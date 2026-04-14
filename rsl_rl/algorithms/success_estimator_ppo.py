# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage.success_estimator_rollout_storage import SuccessEstimatorRolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


class SuccessEstimatorPPO(PPO):
    """PPO extended with a success estimator — a gamma=1 value function trained on binary episode outcomes.

    The success estimator predicts P(success | state) as a scalar in [0, 1] (via sigmoid). It is
    trained with BCE loss against gamma=1 return targets derived from episode termination labels:
    success=1, timeout/failure=0. Envs whose termination is influenced by the estimator itself
    are excluded from its training via ``extras["success_train_mask"]``.
    """

    success_estimator: MLPModel
    """The success estimator model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: SuccessEstimatorRolloutStorage,
        success_estimator: MLPModel,
        success_estimator_learning_rate: float = 1e-4,
        success_loss_coef: float = 1.0,
        success_returns_method: str = "hindsight_mc",
        **kwargs,
    ) -> None:
        """Initialize PPO with an additional success estimator network and optimizer.

        Args:
            success_returns_method: How to compute return targets for the success
                estimator. ``"bootstrap"`` uses TD(0)-style single-step bootstrap
                at every step. ``"hindsight_mc"`` uses the true episode outcome for
                completed episodes and only bootstraps in-progress ones.
        """
        super().__init__(actor, critic, storage, **kwargs)

        self.success_estimator = success_estimator.to(self.device)
        self.success_optimizer = optim.Adam(self.success_estimator.parameters(), lr=success_estimator_learning_rate)
        self.success_loss_coef = success_loss_coef
        self.success_returns_method = success_returns_method

        self.success_predictions = torch.zeros(storage.num_envs, device=self.device)
        """Shared buffer of raw (pre-sigmoid) success predictions, shape ``(num_envs,)``.

        Written by :meth:`act` every step. External consumers (e.g. an env termination term)
        can hold a reference to this tensor and read it without knowing about the algorithm.
        Bind via :meth:`construct_algorithm` or manually after construction.
        """

        self._state_buffer = None
        self._loss_callback: list = []

        self.transition = SuccessEstimatorRolloutStorage.Transition()

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions, compute value estimates, and compute success estimates."""
        actions = super().act(obs)
        self.transition.success_values = self.success_estimator(obs).detach()
        self.success_predictions[:] = self.transition.success_values.squeeze(-1)
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step, including success reward construction.

        The environment must provide:
        - ``extras["successes"]``: ``(num_envs,)`` bool/float tensor, True/1.0 for envs that just succeeded.

        Optional:
        - ``extras["success_train_mask"]``: ``(num_envs,)`` float tensor (1.0 = train,
          0.0 = exclude). Used to mask out envs whose termination was influenced by the
          success estimator itself.
        """
        self.success_estimator.update_normalization(obs)

        success_rewards = torch.zeros(rewards.shape[0], device=self.device)

        if "successes" in extras:
            success_rewards += extras["successes"].float().to(self.device)

        self.transition.success_rewards = success_rewards

        if "success_train_mask" in extras:
            self.transition.success_mask = extras["success_train_mask"].float().to(self.device)
        else:
            self.transition.success_mask = None

        super().process_env_step(obs, rewards, dones, extras)
        self.success_estimator.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute critic returns/advantages and success estimator returns."""
        super().compute_returns(obs)
        self._compute_success_returns(obs)

    def _compute_success_returns(self, obs: TensorDict) -> None:
        """Compute gamma=1 return targets for the success estimator.

        Dispatches to the method selected by :attr:`success_returns_method`.
        """
        if self.success_returns_method == "bootstrap":
            self._compute_success_returns_bootstrap(obs)
        elif self.success_returns_method == "hindsight_mc":
            self._compute_success_returns_hindsight_mc(obs)
        else:
            raise ValueError(f"Unknown success_returns_method: {self.success_returns_method!r}")

    def _compute_success_returns_bootstrap(self, obs: TensorDict) -> None:
        """TD(0)-style bootstrap: target = reward + (1 - done) * P(s_{t+1}).

        At every non-terminal step the target depends on the network's own next-step
        prediction. Accurate only when the network is already well-trained; converges
        slowly early in training due to the bootstrap chicken-and-egg.
        """
        st: SuccessEstimatorRolloutStorage = self.storage  # type: ignore
        last_success_probs = torch.sigmoid(self.success_estimator(obs).detach())

        for step in reversed(range(st.num_transitions_per_env)):
            next_probs = last_success_probs if step == st.num_transitions_per_env - 1 else torch.sigmoid(st.success_values[step + 1])
            next_is_not_terminal = 1.0 - st.dones[step].float()
            st.success_returns[step] = st.success_rewards[step] + next_is_not_terminal * next_probs

    def _compute_success_returns_hindsight_mc(self, obs: TensorDict) -> None:
        """Hindsight Monte Carlo: use ground-truth outcomes for completed episodes.

        Walks backwards through the rollout. When a terminal step (done=1) is
        encountered, ``outcome`` is set to the true episode result (success_reward:
        1 for success, 0 for failure/timeout). All preceding steps in that episode
        inherit the same outcome — this is exact because gamma=1.

        Only episodes still in-progress at the rollout boundary are bootstrapped
        with the network's current estimate.
        """
        st: SuccessEstimatorRolloutStorage = self.storage  # type: ignore
        # Bootstrap for episodes that haven't terminated within this rollout
        outcome = torch.sigmoid(self.success_estimator(obs).detach())

        for step in reversed(range(st.num_transitions_per_env)):
            is_done = st.dones[step].float()
            # At terminal steps, replace the propagated outcome with ground truth.
            # At non-terminal steps, keep propagating the outcome from later steps.
            outcome = is_done * st.success_rewards[step] + (1.0 - is_done) * outcome
            st.success_returns[step] = outcome

    def update(self) -> dict[str, float]:
        """Run PPO update and additionally train the success estimator."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_success_loss = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]
            original_observations = batch.observations

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Adaptive learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # Symmetry loss
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )
                mean_actions = self.actor(batch.observations.detach().clone())
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # PPO backward + step
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Success estimator loss (separate optimizer, separate backward)
            # Use the pre-augmentation observations to avoid interference from symmetry transforms.
            # The network outputs raw logits; BCE with logits applies sigmoid internally
            # and is numerically stable near 0 and 1.
            # When success_train_mask is provided (via extras), excluded envs are masked out.
            success_logits = self.success_estimator(original_observations[:original_batch_size])
            success_targets = batch.success_returns[:original_batch_size].clamp(0.0, 1.0)
            mask = batch.success_mask[:original_batch_size] if batch.success_mask is not None else None
            if mask is not None and not mask.all():
                per_sample_loss = F.binary_cross_entropy_with_logits(success_logits, success_targets, reduction="none")
                success_loss = (per_sample_loss * mask).sum() / mask.sum().clamp(min=1)
            else:
                success_loss = F.binary_cross_entropy_with_logits(success_logits, success_targets)

            self.success_optimizer.zero_grad()
            (self.success_loss_coef * success_loss).backward()
            nn.utils.clip_grad_norm_(self.success_estimator.parameters(), self.max_grad_norm)
            self.success_optimizer.step()

            # Accumulate losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_success_loss += success_loss.item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_success_loss /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        self.storage.clear()

        loss_dict: dict[str, float] = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "success": mean_success_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss  # type: ignore
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss  # type: ignore

        for cb in self._loss_callback:
            cb(mean_success_loss)

        self._update_buffer_success_rates(loss_dict)
        return loss_dict

    @torch.no_grad()
    def _update_buffer_success_rates(self, loss_dict: dict[str, float]) -> None:
        """Evaluate the success estimator on all buffer states and cache the rates."""
        if self._state_buffer is None or len(self._state_buffer) == 0:
            return
        n = len(self._state_buffer)
        time_left = torch.ones(n, 1, device=self.device)
        obs_td = TensorDict({"success": torch.cat([self._state_buffer.data[:n].to(self.device), time_left], dim=-1)}, batch_size=[n])

        p = torch.sigmoid(self.success_estimator(obs_td).squeeze(-1))
        self._state_buffer.success_rates = p

        loss_dict["se_p_min"] = p.min().item()
        loss_dict["se_p_mean"] = p.mean().item()
        loss_dict["se_p_max"] = p.max().item()
        loss_dict["se_p_std"] = p.std().item()

    def train_mode(self) -> None:
        super().train_mode()
        self.success_estimator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.success_estimator.eval()

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["success_estimator_state_dict"] = self.success_estimator.state_dict()
        saved_dict["success_optimizer_state_dict"] = self.success_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        result = super().load(loaded_dict, load_cfg, strict)
        should_load = load_cfg is None or load_cfg.get("success_estimator", True)
        if should_load and "success_estimator_state_dict" in loaded_dict:
            self.success_estimator.load_state_dict(loaded_dict["success_estimator_state_dict"], strict=strict)
            self.success_optimizer.load_state_dict(loaded_dict["success_optimizer_state_dict"])
        return result

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        model_params = [self.success_estimator.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.success_estimator.load_state_dict(model_params[0])

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> SuccessEstimatorPPO:
        """Construct SuccessEstimatorPPO with actor, critic, success estimator, and storage."""
        alg_class: type[SuccessEstimatorPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore
        success_estimator_class: type[MLPModel] = resolve_callable(cfg["success_estimator"].pop("class_name"))  # type: ignore
        for deprecated_key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            cfg["success_estimator"].pop(deprecated_key, None)

        default_sets = ["actor", "critic", "success_estimator"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config

        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        success_estimator: MLPModel = success_estimator_class(
            obs, cfg["obs_groups"], "success_estimator", 1, **cfg["success_estimator"]
        ).to(device)
        print(f"Success Estimator Model: {success_estimator}")

        storage = SuccessEstimatorRolloutStorage(
            "rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        alg: SuccessEstimatorPPO = alg_class(
            actor,
            critic,
            storage,
            success_estimator=success_estimator,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        # Bind success estimator to env components via user-supplied expressions
        bind_ns = {"env": env, "alg": alg, "setattr": setattr}
        for key in ("success_estimator_bind", "state_buffer_bind"):
            bind_expr = cfg.get(key)
            if bind_expr is not None:
                try:
                    eval(bind_expr, bind_ns)  # noqa: S307
                except Exception as e:
                    print(f"[WARNING] {key} skipped: {e}")

        return alg
