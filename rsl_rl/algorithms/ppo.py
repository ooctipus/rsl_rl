# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import (
    RandomNetworkDistillation,
    SuccessorFeatures,
    Symmetry,
    ValueShift,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class PPO:
    """Proximal Policy Optimization algorithm.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: MLPModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        log_weight_decay_metrics: bool = True,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Value-shift parameters (post-update critic value-drift signal)
        value_shift_cfg: dict | None = None,
        # Successor-feature parameters (auxiliary head + optional value coupling)
        successor_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND extension
        self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg) if rnd_cfg else None

        # Symmetry extension
        if symmetry_cfg is not None and (actor.is_recurrent or critic.is_recurrent):
            raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg else None

        # Value-shift extension (post-update critic value-drift signal; no loss/gradient)
        self.value_shift = ValueShift(**value_shift_cfg) if value_shift_cfg else None

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Handles to the uncompiled modules for state_dict operations and export. If compilation is disabled, these
        # simply alias ``self.actor`` / ``self.critic``.
        self._raw_actor = self.actor
        self._raw_critic = self.critic

        # Forward-backward successor-representation critic. The F/B maps live on the critic model; this
        # extension owns only the reward-free occupancy losses and the forward-backward target net (built in
        # ``construct_algorithm``). The value V = <F(s, z), z> is derived -- the extension has no learnable params.
        if successor_cfg is not None:
            if actor.is_recurrent or critic.is_recurrent:
                raise ValueError("Successor augmentation is not supported for recurrent policies.")
            if symmetry_cfg is not None:
                raise ValueError("Successor augmentation is not supported together with symmetry augmentation.")
        self.successor = SuccessorFeatures(gamma=gamma, device=self.device, **successor_cfg) if successor_cfg else None

        if weight_decay < 0.0:
            raise ValueError(f"Weight decay must be non-negative; got {weight_decay}.")
        self.weight_decay = weight_decay
        self.log_weight_decay_metrics = log_weight_decay_metrics

        # Create the optimizer (the successor F/B heads are on the critic, already covered).
        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

    def _state_value(self, obs: TensorDict) -> torch.Tensor:
        """Return the state value ``V(s)`` -- the single source of truth for value across the algorithm.

        This is the successor value ``<F(s, z), z>`` when the successor extension is enabled, else the critic's
        scalar head. Shared by GAE bootstrapping (``act`` / ``compute_returns``) and the
        value-shift hook so they never diverge. No masks: the successor path is non-recurrent and the
        value-shift cache is a flat observation batch.
        """
        if self.successor is not None:
            return self.successor.value(self.critic, obs)
        return self.critic(obs)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Record the hidden states for recurrent policies
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        # Compute the actions and values
        if self.successor is not None:
            # z-conditioned actor + value share ONE detached goal embedding z = B(current goal); the actor reads
            # the goal only through z (its obs is goal-free), and the value is V = <F(s, z), z>. The per-env
            # commanded task index is stored so the update recomputes the SAME z with the still-training B.
            cmd_indices = self.successor.cmd_indices_fn() if self.successor.cmd_indices_fn is not None else None
            self.transition.command_indices = cmd_indices
            z = self.successor.goal_z(self.critic, cmd_indices)
            self.transition.actions = self.actor(obs, z, stochastic_output=True).detach()
            self.transition.values = self.successor.state_value(self.critic, obs, z).detach()
        else:
            self.transition.actions = self.actor(obs, stochastic_output=True).detach()
            self.transition.values = self._state_value(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Successor-feature augmentation: record the realized next state s' for the occupancy objective and
        # OVERRIDE the policy reward with the FB-native goal-alignment <B(s'), z> (w = z; no extrinsic success
        # reward). The timeout value-bootstrap below still mutates ``transition.rewards`` (used for GAE).
        if self.successor is not None:
            # Realized next state s' for the FB occupancy. CRITICAL reset-state handling: the env auto-reset has
            # overwritten ``obs`` with the NEXT EPISODE's reset state on every done env. Never let that reset
            # obs be read as the successor of the last step. So:
            #   - true terminal (absorbing): alias next_obs := the CURRENT obs (self-loop). No reset state, no
            #     captured terminal obs (which poisoned training); the absorbing self-visit is the correct object.
            #   - timeout: not absorbing and we lack the true continuation, so it is DROPPED from the FB loss
            #     (keep mask); its stored next_obs is irrelevant.
            #   - non-done: keep the real post-step next obs.
            timeout = extras["time_outs"] if "time_outs" in extras else torch.zeros_like(dones)
            terminal_mask = dones.bool() & ~timeout.bool()
            next_obs = obs
            if terminal_mask.any():
                next_obs = obs.clone()
                cur = self.transition.observations  # pre-step obs s_t (set in act())
                for key in next_obs.keys():  # noqa: SIM118  (TensorDict iteration walks the batch dim, not keys)
                    next_obs[key][terminal_mask] = cur[key][terminal_mask]
            self.transition.next_observations = next_obs
            self.transition.time_outs = timeout
            # w = z reward: r(s') = <B(s'), z>, the goal-alignment whose successor value is V = <F(s, z), z>.
            # z = the commanded goal embedding (same as act()). Replaces the extrinsic reward set above.
            z = self.successor.goal_z(self.critic, self.transition.command_indices)
            self.transition.rewards = self.successor.goal_alignment_reward(self.critic, next_obs, z)

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        # Compute values for the last step
        critic_hidden_state = self.critic.get_hidden_state()
        last_values = self._state_value(obs).detach()
        # Restore the critic's hidden state so the next rollout is not affected by the forward pass
        self.critic.reset(hidden_state=critic_hidden_state)
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # Successor-representation losses (FB measure / orthonormality) + ``||F(s,z)||``, the early-warning
        # signal that must SELF-BOUND under z-conditioning (it inflates well before the loss does if F diverges).
        mean_fb_loss = 0 if self.successor else None
        mean_ortho_loss = 0 if self.successor else None
        mean_f_norm = 0 if self.successor else None
        mean_m_diag = 0 if self.successor else None  # E[M(s,s')] diagonal -> should converge to 1/(1-gamma)

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over mini-batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with new parameters
            if self.successor is not None:
                # z-conditioned actor + value share ONE detached goal embedding z = B(goal_cache[stored
                # cmd_indices]) -- recomputed with the current (still-training) backward map (raw goal obs are
                # cached, not stale embeddings). The actor reads the goal only through z; the value V=<F(s,z),z>
                # is derived from F (no read-out), so F stays owned by the reward-free FB measure + ortho.
                # (Successor forbids symmetry, so the batch is un-augmented.)
                z_goal = self.successor.goal_z(self.critic, batch.command_indices)  # type: ignore
                self.actor(
                    batch.observations,
                    z_goal,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )
            else:
                self.actor(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            if self.successor is not None:
                obs_s = batch.observations[:original_batch_size]
                values = self.successor.state_value(self.critic, obs_s, z_goal[:original_batch_size])
            else:
                values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
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

            loss = surrogate_loss - self.entropy_coef * entropy.mean()
            # Critic objective (mutually exclusive): a scalar critic regresses V to returns; the successor critic
            # instead trains the reward-free z-conditioned forward-backward occupancy + orthonormality, and its
            # value V=<F(s,z),z> is DERIVED from F (no value regression -- ``value_loss`` stays only a diagnostic).
            # ``representation_loss`` does its own F/B forwards on obs_s/next_obs, conditioned on the ON-POLICY
            # goal z (the env's commanded goal via ``command_indices``) -- the z that generated the transition
            # under the z-conditioned actor, which is what makes z load-bearing in on-policy PPO.
            if self.successor is None:
                loss = loss + self.value_loss_coef * value_loss
            else:
                next_obs = batch.next_observations[:original_batch_size]  # type: ignore
                timeout = batch.time_outs[:original_batch_size].float()  # type: ignore
                # true (absorbing) terminals = dones that are not timeouts (dones lumps both)
                terminal = (batch.dones[:original_batch_size].bool() & ~timeout.bool()).float()  # type: ignore
                cmd_idx = batch.command_indices[:original_batch_size] if batch.command_indices is not None else None  # type: ignore
                fb_loss, ortho_loss, f_norm, m_diag = self.successor.representation_loss(
                    self.critic, obs_s, next_obs, terminal, timeout, cmd_idx
                )
                loss = loss + fb_loss + self.successor.ortho_coef * ortho_loss

            # RND loss
            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

            # Symmetry loss
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()
            # Polyak-update the successor forward-backward target network toward the live critic.
            if self.successor is not None:
                self.successor.update_target(self.critic)

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            # Successor-representation losses
            if mean_fb_loss is not None:
                mean_fb_loss += fb_loss.item()
                mean_ortho_loss += ortho_loss.item()
                mean_f_norm += f_norm.item()
                mean_m_diag += m_diag.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if mean_fb_loss is not None:
            mean_fb_loss /= num_updates
            mean_ortho_loss /= num_updates
            mean_f_norm /= num_updates
            mean_m_diag /= num_updates

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.successor:
            loss_dict["fb"] = mean_fb_loss
            loss_dict["ortho"] = mean_ortho_loss
            loss_dict["f_norm"] = mean_f_norm
            loss_dict["m_diag"] = mean_m_diag

        # Weight-decay diagnostics: total L2 norm of the optimized parameters, so the regularization
        # pressure is visible. Only logged when weight decay is actually active.
        if self.weight_decay > 0.0 and self.log_weight_decay_metrics:
            with torch.no_grad():
                weight_norm = torch.norm(
                    torch.stack([p.detach().norm() for g in self.optimizer.param_groups for p in g["params"]])
                )
            loss_dict["weight_norm"] = weight_norm.item()

        # Clear the storage
        self.storage.clear()

        # Value-shift augmentation: post-update per-state value-drift signal (no gradient). Pass the algorithm's
        # value function (``<F(s, z), z>`` under successor, scalar head otherwise) so the two extensions compose.
        if self.value_shift:
            self.value_shift.after_update(self._state_value)

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd.optimizer.state_dict()
        if self.successor:
            saved_dict["successor_state_dict"] = self.successor.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
                "successor": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd.optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if load_cfg.get("successor") and self.successor and "successor_state_dict" in loaded_dict:
            self.successor.load_state_dict(loaded_dict["successor_state_dict"], strict=strict)
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile actor and critic with ``torch.compile``.

        See :func:`~rsl_rl.utils.compile_model` for the set of accepted modes.

        Args:
            mode: ``torch.compile`` mode. Defaults to ``None``, in which case compilation is disabled.
        """
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct the PPO algorithm."""
        # Resolve class callables
        alg_class: type[PPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Initialize the policy
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        # The critic outputs a scalar by contract; in successor mode it is the SuccessorFeatureCriticModel,
        # which ignores ``output_dim`` and exposes ``forward_map``/``backward`` (value = extension's <F(s,z), z>).
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm
        alg: PPO = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        # Build the successor forward-backward target network (a frozen Polyak copy of the critic). The F/B
        # heads are on the critic, so they are already in the optimizer; the extension has no learnable params.
        # Copy the UNCOMPILED critic: the target net runs under no_grad (no need to compile it) and
        # ``copy.deepcopy`` of a ``torch.compile`` OptimizedModule is fragile.
        if alg.successor is not None:
            alg.successor.build(alg._raw_critic)
            # Bind the goal library for the z-conditioned value V = <F(s, z), z>, z = B(goal_cache[cmd_indices]).
            # The command term named by ``successor_cfg.goal_command_name`` (a StateCommand) builds the per-task
            # target-obs cache once (a teleport-to-target sweep, restored after) and exposes the per-env task
            # index. Single deterministic lookup -- no discovery scan, no fallback: a wrong/missing name or a
            # term lacking ``get_target_obs_cache`` fails loudly here at the boundary.
            term = env.unwrapped.command_manager.get_term(alg.successor.goal_command_name)
            alg.successor.bind_goals(term.get_target_obs_cache(), lambda: term.cmd_indices)

        # Bind the value-shift cache/buffers (owned by an external consumer) onto the extension
        if alg.value_shift is not None:
            alg.value_shift.bind(env, alg)

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self._raw_actor.state_dict(), self._raw_critic.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic.load_state_dict(model_params[1])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
