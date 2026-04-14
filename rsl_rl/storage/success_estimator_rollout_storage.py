# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.storage.rollout_storage import RolloutStorage


class SuccessEstimatorRolloutStorage(RolloutStorage):
    """Rollout storage extended with buffers for a success estimator (gamma=1 value function)."""

    class Transition(RolloutStorage.Transition):
        """Transition with an additional success-value estimate."""

        def __init__(self) -> None:
            super().__init__()
            self.success_values: torch.Tensor | None = None
            """Success-estimator prediction at the current step."""

            self.success_rewards: torch.Tensor | None = None
            """Success reward signal (1 on success, 0 otherwise, bootstrap on predictor truncation)."""

            self.success_mask: torch.Tensor | None = None
            """Per-env mask for success estimator training (1.0 = train, 0.0 = exclude)."""

    class Batch(RolloutStorage.Batch):
        """Batch extended with success-estimator fields."""

        def __init__(
            self,
            *,
            success_values: torch.Tensor | None = None,
            success_returns: torch.Tensor | None = None,
            success_mask: torch.Tensor | None = None,
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)
            self.success_values: torch.Tensor | None = success_values
            """Batch of success-estimator predictions."""

            self.success_returns: torch.Tensor | None = success_returns
            """Batch of success-estimator return targets (gamma=1)."""

            self.success_mask: torch.Tensor | None = success_mask
            """Batch of per-transition training masks for the success estimator."""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
    ) -> None:
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)

        if training_type == "rl":
            self.success_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.success_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.success_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.success_mask = torch.ones(num_transitions_per_env, num_envs, 1, device=self.device)

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        """Add a transition, including success-estimator fields when available."""
        step = self.step
        super().add_transition(transition)

        if self.training_type == "rl" and isinstance(transition, SuccessEstimatorRolloutStorage.Transition):
            if transition.success_values is not None:
                self.success_values[step].copy_(transition.success_values)
            if transition.success_rewards is not None:
                self.success_rewards[step].copy_(transition.success_rewards.view(-1, 1))
            if transition.success_mask is not None:
                self.success_mask[step].copy_(transition.success_mask.view(-1, 1))

    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[Batch, None, None]:
        """Yield shuffled mini-batches including success-estimator data."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore
        success_values = self.success_values.flatten(0, 1)
        success_returns = self.success_returns.flatten(0, 1)
        success_mask = self.success_mask.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                yield SuccessEstimatorRolloutStorage.Batch(
                    observations=observations[batch_idx],  # type: ignore
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                    success_values=success_values[batch_idx],
                    success_returns=success_returns[batch_idx],
                    success_mask=success_mask[batch_idx],
                )
