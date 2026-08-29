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
    StateCurriculum,
    Symmetry,
    resolve_rnd_config,
    resolve_symmetry_config,
)
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_class, resolve_obs_groups, resolve_optimizer

from .sigreg import SIGReg
from .value_loss import HLGaussValueLoss


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
        optimizer_kwargs: dict | None = None,
        weight_decay: float | None = None,
        weight_decay_mode: str = "all",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        use_mixed_precision: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Reset-state curriculum parameters
        state_curriculum_cfg: dict | None = None,
        # Value and representation parameters
        value_loss_cfg: dict | None = None,
        actor_sigreg_cfg: dict | None = None,
        critic_sigreg_cfg: dict | None = None,
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

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Handles to the uncompiled modules for state_dict operations and export
        self._raw_actor = self.actor
        self._raw_critic = self.critic

        self.value_loss = HLGaussValueLoss(**value_loss_cfg).to(self.device) if value_loss_cfg is not None else None
        if self.value_loss is not None and use_clipped_value_loss:
            raise ValueError("HL-Gauss value loss does not support clipped value loss.")
        self.actor_sigreg = SIGReg(**actor_sigreg_cfg).to(self.device) if actor_sigreg_cfg is not None else None
        self.critic_sigreg = SIGReg(**critic_sigreg_cfg).to(self.device) if critic_sigreg_cfg is not None else None
        for name, model, sigreg in (
            ("actor", self.actor, self.actor_sigreg),
            ("critic", self.critic, self.critic_sigreg),
        ):
            if sigreg is not None and not callable(getattr(model, "forward_with_features", None)):
                raise TypeError(f"{name} SIGReg requires {type(model).__name__}.forward_with_features().")

        # Create the optimizer
        if weight_decay is not None and weight_decay < 0.0:
            raise ValueError(f"Weight decay must be non-negative; got {weight_decay}.")
        if weight_decay_mode not in ("all", "matrix"):
            raise ValueError("weight_decay_mode must be 'all' or 'matrix'.")
        optimizer_kwargs = dict(optimizer_kwargs or {})
        conflicting = {"lr", "weight_decay"}.intersection(optimizer_kwargs)
        if conflicting:
            raise ValueError(f"Use dedicated PPO fields instead of optimizer_kwargs for: {sorted(conflicting)}")
        optimizer_kwargs["lr"] = learning_rate
        if weight_decay is not None:
            optimizer_kwargs["weight_decay"] = weight_decay
        parameters = chain(self.actor.parameters(), self.critic.parameters())
        if weight_decay_mode == "matrix":
            unique_parameters = list({id(parameter): parameter for parameter in parameters}.values())
            parameters = [
                {"params": [parameter for parameter in unique_parameters if parameter.ndim >= 2]},
                {"params": [parameter for parameter in unique_parameters if parameter.ndim < 2], "weight_decay": 0.0},
            ]
        self.optimizer = resolve_optimizer(optimizer)(parameters, **optimizer_kwargs)  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()
        self.state_curriculum = (
            StateCurriculum(
                storage,
                device,
                self.is_multi_gpu,
                **state_curriculum_cfg,
            )
            if state_curriculum_cfg is not None
            else None
        )
        if self.state_curriculum is not None and not self.state_curriculum.enabled:
            self.state_curriculum = None

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
        self.use_mixed_precision = use_mixed_precision

    def _state_value(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        predictions = self.critic(obs, masks=masks, hidden_state=hidden_state)
        return self.value_loss.decode(predictions) if self.value_loss is not None else predictions

    def _normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        if not self.is_multi_gpu:
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        local_mean = advantages.mean()
        moments = torch.stack((advantages.new_tensor(advantages.numel()), local_mean, advantages.var(unbiased=False)))
        gathered = torch.empty(self.gpu_world_size * 3, device=advantages.device, dtype=advantages.dtype)
        torch.distributed.all_gather_into_tensor(gathered, moments)
        count = advantages.new_zeros(())
        mean = advantages.new_zeros(())
        squared_deviation = advantages.new_zeros(())
        for rank_count, rank_mean, rank_variance in gathered.view(-1, 3):
            total = count + rank_count
            delta = rank_mean - mean
            mean += delta * rank_count / total
            squared_deviation += rank_count * rank_variance + delta.square() * count * rank_count / total
            count = total
        variance = squared_deviation / (count - 1.0).clamp_min(1.0)
        return (advantages - mean) / (variance.sqrt() + 1e-8)

    @torch.no_grad()
    def _project_models(self) -> None:
        for model in (self._raw_actor, self._raw_critic):
            project = getattr(model, "project_parameters", None)
            if project is not None:
                project()

    def _average_losses(self, losses: dict[str, float]) -> dict[str, float]:
        if not self.is_multi_gpu:
            return losses
        values = torch.tensor(list(losses.values()), device=self.device)
        torch.distributed.all_reduce(values)
        values /= self.gpu_world_size
        return dict(zip(losses, values.tolist()))

    @torch.no_grad()
    def _value_target_statistics(self) -> dict[str, float]:
        targets = self.storage.returns.float().flatten()
        minimum = targets.min()
        maximum = targets.max()
        if not self.is_multi_gpu:
            quantiles = torch.quantile(targets, targets.new_tensor((0.01, 0.5, 0.99)))
        else:
            torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
            if maximum == minimum:
                quantiles = minimum.repeat(3)
            else:
                num_bins = 2048
                histogram = torch.histc(targets, bins=num_bins, min=minimum.item(), max=maximum.item())
                torch.distributed.all_reduce(histogram)
                indices = torch.searchsorted(
                    histogram.cumsum(0), histogram.sum() * histogram.new_tensor((0.01, 0.5, 0.99))
                )
                quantiles = minimum + (indices + 0.5) * (maximum - minimum) / num_bins
        return {
            "value_target_min": minimum.item(),
            "value_target_p01": quantiles[0].item(),
            "value_target_median": quantiles[1].item(),
            "value_target_p99": quantiles[2].item(),
            "value_target_max": maximum.item(),
        }

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Record the hidden states for recurrent policies
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        # Compute the actions and values
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self._state_value(obs).detach()
        if self.state_curriculum is not None:
            self.state_curriculum.record_episode_starts(self.storage.step, self.transition.values)
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step."""
        if self.state_curriculum is not None:
            self.state_curriculum.collect_success_outcomes()
        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

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
            st.advantages = self._normalize_advantages(st.advantages)

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_value_support_clip_fraction = 0 if self.value_loss is not None else None
        mean_actor_sigreg_loss = 0 if self.actor_sigreg is not None else None
        mean_critic_sigreg_loss = 0 if self.critic_sigreg is not None else None
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over mini-batches
        for batch in generator:
            assert batch.values is not None and batch.returns is not None
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = self._normalize_advantages(batch.advantages)  # type: ignore

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Optionally use mixed precision for the forward pass and loss computation
            with torch.amp.autocast(  # type: ignore
                device_type=torch.device(self.device).type, enabled=self.use_mixed_precision, dtype=torch.bfloat16
            ):
                # Recompute actions log prob and entropy for current batch of transitions
                # Note: We need to do this because we updated the policy with new parameters
                if self.actor_sigreg is not None:
                    _, actor_features = self.actor.forward_with_features(  # type: ignore[attr-defined]
                        batch.observations,
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
                    actor_features = None
                actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
                if self.critic_sigreg is not None:
                    value_predictions, critic_features = self.critic.forward_with_features(  # type: ignore[attr-defined]
                        batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
                    )
                else:
                    value_predictions = self.critic(
                        batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
                    )
                    critic_features = None
                values = self.value_loss.decode(value_predictions) if self.value_loss is not None else value_predictions
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
                if self.value_loss is not None:
                    value_loss, value_support_clip_fraction = self.value_loss(value_predictions, batch.returns)
                elif self.use_clipped_value_loss:
                    value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - batch.returns).pow(2)
                    value_losses_clipped = (value_clipped - batch.returns).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (batch.returns - values).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
                actor_sigreg_loss = None
                if self.actor_sigreg is not None:
                    actor_sigreg_loss = self.actor_sigreg(actor_features[:original_batch_size])  # type: ignore[index]
                    loss = loss + self.actor_sigreg.loss_coef * actor_sigreg_loss
                critic_sigreg_loss = None
                if self.critic_sigreg is not None:
                    critic_sigreg_loss = self.critic_sigreg(critic_features[:original_batch_size])  # type: ignore[index]
                    loss = loss + self.critic_sigreg.loss_coef * critic_sigreg_loss

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
            self._project_models()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_value_support_clip_fraction is not None:
                mean_value_support_clip_fraction += value_support_clip_fraction.item()
            if mean_actor_sigreg_loss is not None:
                mean_actor_sigreg_loss += actor_sigreg_loss.item()  # type: ignore[union-attr]
            if mean_critic_sigreg_loss is not None:
                mean_critic_sigreg_loss += critic_sigreg_loss.item()  # type: ignore[union-attr]
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        if mean_value_support_clip_fraction is not None:
            mean_value_support_clip_fraction /= num_updates
        if mean_actor_sigreg_loss is not None:
            mean_actor_sigreg_loss /= num_updates
        if mean_critic_sigreg_loss is not None:
            mean_critic_sigreg_loss /= num_updates

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
        if mean_value_support_clip_fraction is not None:
            loss_dict["value_support_clip_fraction"] = mean_value_support_clip_fraction
        if mean_actor_sigreg_loss is not None:
            loss_dict["actor_sigreg"] = mean_actor_sigreg_loss
        if mean_critic_sigreg_loss is not None:
            loss_dict["critic_sigreg"] = mean_critic_sigreg_loss
        if self.value_loss is not None:
            loss_dict.update(self._value_target_statistics())

        loss_dict = self._average_losses(loss_dict)

        if self.state_curriculum is not None:
            self.state_curriculum.update_value_shift(self._state_value)
            success_loss = self.state_curriculum.update_success_estimator(
                self.num_learning_epochs, self.num_mini_batches
            )
            if success_loss is not None:
                loss_dict["success_estimator"] = success_loss

        # Update normalization only after every consumer of the rollout's frame has finished.
        obs = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(obs)  # type: ignore
        self.critic.update_normalization(obs)  # type: ignore
        if self.rnd:
            self.rnd.update_normalization(obs)  # type: ignore

        # Clear the storage
        self.storage.clear()

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()
        if self.state_curriculum is not None:
            self.state_curriculum.train_mode()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()
        if self.state_curriculum is not None:
            self.state_curriculum.eval_mode()

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
        if self.state_curriculum is not None:
            saved_dict["state_curriculum_state_dict"] = self.state_curriculum.save()
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
                "state_curriculum": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd.optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if load_cfg.get("state_curriculum") and self.state_curriculum is not None:
            self.state_curriculum.load(loaded_dict.get("state_curriculum_state_dict", {}), strict=strict)
        if load_cfg.get("actor") or load_cfg.get("critic"):
            self._project_models()
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
        # Resolve class callables and configs
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        critic_class, critic_cfg = resolve_class(cfg["critic"])

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in alg_cfg and alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        alg_cfg = resolve_symmetry_config(alg_cfg, env)

        # Initialize the policy
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        print(f"Actor Model: {actor}")
        if alg_cfg.pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            critic_cfg["cnns"] = actor.cnns
        value_loss_cfg = alg_cfg.get("value_loss_cfg")
        critic_output_dim = int(value_loss_cfg.get("num_bins", 101)) if value_loss_cfg is not None else 1
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", critic_output_dim, **critic_cfg).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm
        alg: PPO = alg_class(actor, critic, storage, device=device, **alg_cfg, multi_gpu_cfg=cfg["multi_gpu"])

        if alg.state_curriculum is not None:
            provider = env.get_state_curriculum()
            if provider is None:
                raise ValueError("state_curriculum_cfg is enabled, but the environment returned no state curriculum.")
            alg.state_curriculum.bind(provider, env.episode_length_buf, alg.critic)

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

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
        self._project_models()
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])
        if self.state_curriculum is not None:
            self.state_curriculum.broadcast_parameters()

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
