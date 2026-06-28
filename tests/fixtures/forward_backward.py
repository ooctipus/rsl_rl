# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared forward-backward contract fixtures."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models.forward_backward_model import ForwardBackwardObservationSchema
from rsl_rl.modules.reward_channels import ForwardBackwardRewardChannel, ForwardBackwardRewardSchema
from rsl_rl.storage.forward_backward_replay import (
    ForwardBackwardAutoresetMode,
    ForwardBackwardReplayBatch,
    ForwardBackwardTransitionBatch,
    ForwardBackwardTransitionSchema,
)

META_FIELD_WIDTHS = {"state": 358}
META_ROUTES = {
    "actor": ("state",),
    "forward": ("state",),
    "backward": ("state",),
    "discriminator": ("state",),
    "critic_discriminator": ("state",),
}
META_ROUTE_WIDTHS = {
    "actor": 358,
    "forward": 358,
    "backward": 358,
    "discriminator": 358,
    "critic_discriminator": 358,
}

BFM_FIELD_WIDTHS = {
    "state": 64,
    "last_action": 29,
    "history_actor": 372,
    "privileged_state": 463,
}
BFM_ROUTES = {
    "actor": ("state", "last_action", "history_actor"),
    "forward": ("state", "privileged_state", "last_action", "history_actor"),
    "backward": ("state", "privileged_state"),
    "discriminator": ("state", "privileged_state"),
    "critic_discriminator": ("state", "privileged_state", "last_action", "history_actor"),
    "critic_auxiliary": ("state", "privileged_state", "last_action", "history_actor"),
}
BFM_ROUTE_WIDTHS = {
    "actor": 465,
    "forward": 928,
    "backward": 527,
    "discriminator": 527,
    "critic_discriminator": 928,
    "critic_auxiliary": 928,
}


def make_meta_schema() -> ForwardBackwardObservationSchema:
    """Create the MetaMotivo observation schema."""
    return ForwardBackwardObservationSchema.from_config(META_FIELD_WIDTHS, META_ROUTES)


def make_bfm_schema() -> ForwardBackwardObservationSchema:
    """Create the BFM-Zero observation schema."""
    return ForwardBackwardObservationSchema.from_config(BFM_FIELD_WIDTHS, BFM_ROUTES)


def make_reward_schema() -> ForwardBackwardRewardSchema:
    """Create a reward schema with stored and recomputed channels."""
    return ForwardBackwardRewardSchema(
        channels=(
            ForwardBackwardRewardChannel(
                name="environment",
                provider_name="environment",
                source="environment",
                timing="transition",
                context_dependent=False,
                sign=1,
            ),
            ForwardBackwardRewardChannel(
                name="discriminator",
                provider_name="discriminator",
                source="recomputed",
                timing="next_state",
                context_dependent=True,
                sign=1,
            ),
            ForwardBackwardRewardChannel(
                name="action_rate",
                provider_name="auxiliary",
                source="stored_evidence",
                timing="transition",
                context_dependent=False,
                sign=-1,
            ),
            ForwardBackwardRewardChannel(
                name="slip",
                provider_name="auxiliary",
                source="stored_evidence",
                timing="transition",
                context_dependent=False,
                sign=-1,
            ),
        )
    )


def make_observations(
    field_widths: dict[str, int],
    batch_size: int = 4,
    *,
    zeros: bool = False,
) -> TensorDict:
    """Create flat FP32 observations for one field-width mapping."""
    factory = torch.zeros if zeros else torch.randn
    return TensorDict(
        {name: factory(batch_size, width) for name, width in field_widths.items()},
        batch_size=[batch_size],
    )


def make_transition(
    mode: ForwardBackwardAutoresetMode = ForwardBackwardAutoresetMode.SAME_STEP,
    batch_size: int = 4,
) -> tuple[
    ForwardBackwardTransitionBatch,
    ForwardBackwardTransitionSchema,
    ForwardBackwardObservationSchema,
    ForwardBackwardRewardSchema,
]:
    """Create one normalized transition batch and its static schemas."""
    if batch_size < 4:
        raise ValueError("The transition fixture requires at least four rows.")
    observation_schema = make_meta_schema()
    reward_schema = make_reward_schema()
    schema = ForwardBackwardTransitionSchema(
        observation_schema_hash=observation_schema.schema_hash,
        reward_schema_hash=reward_schema.schema_hash,
        action_width=2,
        context_width=3,
        environment_reward_name="environment",
        auxiliary_evidence_names=("action_rate", "slip"),
        autoreset_mode=mode,
    )
    observations = make_observations(META_FIELD_WIDTHS, batch_size)
    next_observations = make_observations(META_FIELD_WIDTHS, batch_size)
    # Invalid rows are intentionally unspecified rather than zero-filled.
    final_observations = make_observations(META_FIELD_WIDTHS, batch_size)
    terminated = torch.zeros(batch_size, 1, dtype=torch.bool)
    truncated = torch.zeros(batch_size, 1, dtype=torch.bool)
    terminated[1] = True
    truncated[2] = True
    final_observation_valid = torch.zeros(batch_size, 1, dtype=torch.bool)
    if mode is ForwardBackwardAutoresetMode.SAME_STEP:
        final_observation_valid = terminated | truncated
    context_changed = torch.zeros(batch_size, 1, dtype=torch.bool)
    context_changed[-1] = True
    transition = ForwardBackwardTransitionBatch(
        observations=observations,
        next_observations=next_observations,
        final_observations=final_observations,
        actions=torch.randn(batch_size, schema.action_width),
        behavior_context=torch.randn(batch_size, schema.context_width),
        environment_reward=torch.randn(batch_size, 1),
        auxiliary_reward_evidence=torch.rand(batch_size, len(schema.auxiliary_evidence_names)),
        terminated=terminated,
        truncated=truncated,
        context_changed=context_changed,
        action_applied=torch.ones(batch_size, 1, dtype=torch.bool),
        final_observation_valid=final_observation_valid,
    )
    return transition, schema, observation_schema, reward_schema


class ForwardBackwardReplayOracle:
    """Small explicit current-and-next replay used as a correctness oracle."""

    def __init__(
        self,
        capacity_steps: int,
        num_envs: int,
        observation_schema: ForwardBackwardObservationSchema,
        transition_schema: ForwardBackwardTransitionSchema,
        reward_schema: ForwardBackwardRewardSchema,
        device: str | torch.device,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Allocate the dense oracle on one device."""
        if capacity_steps < 1 or num_envs < 1:
            raise ValueError("capacity_steps and num_envs must be positive.")
        transition_schema.assert_compatible(observation_schema, reward_schema)
        self.capacity_steps = capacity_steps
        self.num_envs = num_envs
        self.observation_schema = observation_schema
        self.transition_schema = transition_schema
        self.device = torch.device(device)
        self.dtype = dtype
        shape = (capacity_steps, num_envs)
        self.observations = _allocate_oracle_observations(observation_schema, shape, self.device, dtype)
        self.next_observations = _allocate_oracle_observations(observation_schema, shape, self.device, dtype)
        self.actions = torch.zeros(*shape, transition_schema.action_width, device=self.device, dtype=dtype)
        self.behavior_context = torch.zeros(*shape, transition_schema.context_width, device=self.device, dtype=dtype)
        self.environment_reward = torch.zeros(*shape, 1, device=self.device, dtype=dtype)
        self.auxiliary_reward_evidence = torch.zeros(
            *shape, len(transition_schema.auxiliary_evidence_names), device=self.device, dtype=dtype
        )
        self.terminated = torch.zeros(*shape, 1, device=self.device, dtype=torch.bool)
        self.truncated = torch.zeros_like(self.terminated)
        self.context_changed = torch.zeros_like(self.terminated)
        self.successor_uses_current = torch.zeros_like(self.terminated)
        self.valid = torch.zeros_like(self.terminated)
        self.step_ids = torch.full(shape, -1, device=self.device, dtype=torch.long)
        self._total_steps = 0
        self._size_steps = 0
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    @property
    def total_steps(self) -> int:
        """Number of vector steps observed since construction."""
        return self._total_steps

    def add(self, transition: ForwardBackwardTransitionBatch) -> None:
        """Insert one vector step with an explicit logical reached state."""
        row = self._total_steps % self.capacity_steps
        replay = transition.replay_mask(self.transition_schema)
        if self.transition_schema.autoreset_mode is ForwardBackwardAutoresetMode.SAME_STEP:
            done = transition.done_mask()
            uses_current = replay & done & ~transition.final_observation_valid
        else:
            done = torch.zeros_like(transition.action_applied)
            uses_current = torch.zeros_like(transition.action_applied)
        for name, _width in self.observation_schema.field_widths:
            self.observations[name][row].copy_(transition.observations[name])
            same_step_reached = torch.where(
                transition.final_observation_valid,
                transition.final_observations[name],
                transition.observations[name],
            )
            reached = torch.where(done, same_step_reached, transition.next_observations[name])
            self.next_observations[name][row].copy_(reached)
        self.actions[row].copy_(transition.actions)
        self.behavior_context[row].copy_(transition.behavior_context)
        self.environment_reward[row].copy_(transition.environment_reward)
        self.auxiliary_reward_evidence[row].copy_(transition.auxiliary_reward_evidence)
        self.terminated[row].copy_(transition.terminated)
        self.truncated[row].copy_(transition.truncated)
        self.context_changed[row].copy_(transition.context_changed)
        self.successor_uses_current[row].copy_(uses_current)
        self.valid[row].copy_(replay)
        self.step_ids[row].fill_(self._total_steps)
        self._total_steps += 1
        self._size_steps = min(self._size_steps + 1, self.capacity_steps)

    def sample(self, step_ids: torch.Tensor, env_ids: torch.Tensor) -> ForwardBackwardReplayBatch:
        """Sample explicit logical step/environment pairs."""
        rows = torch.remainder(step_ids, self.capacity_steps)
        valid = self.valid[rows, env_ids] & (self.step_ids[rows, env_ids] == step_ids).unsqueeze(-1)
        observations = TensorDict(
            {name: self.observations[name][rows, env_ids] for name, _width in self.observation_schema.field_widths},
            batch_size=[step_ids.shape[0]],
            device=self.device,
        )
        next_observations = TensorDict(
            {
                name: self.next_observations[name][rows, env_ids]
                for name, _width in self.observation_schema.field_widths
            },
            batch_size=[step_ids.shape[0]],
            device=self.device,
        )
        return ForwardBackwardReplayBatch(
            observations=observations,
            next_observations=next_observations,
            actions=self.actions[rows, env_ids],
            behavior_context=self.behavior_context[rows, env_ids],
            environment_reward=self.environment_reward[rows, env_ids],
            auxiliary_reward_evidence=self.auxiliary_reward_evidence[rows, env_ids],
            terminated=self.terminated[rows, env_ids],
            truncated=self.truncated[rows, env_ids],
            context_changed=self.context_changed[rows, env_ids],
            successor_uses_current=self.successor_uses_current[rows, env_ids],
            valid=valid,
        )

    def sample_random(self, batch_size: int) -> ForwardBackwardReplayBatch:
        """Sample physical rows uniformly, returning a validity mask for holes."""
        if self._size_steps == 0:
            raise RuntimeError("Cannot sample an empty replay.")
        oldest = self._total_steps - self._size_steps
        step_ids = torch.randint(oldest, self._total_steps, (batch_size,), device=self.device, generator=self.generator)
        env_ids = torch.randint(self.num_envs, (batch_size,), device=self.device, generator=self.generator)
        return self.sample(step_ids, env_ids)


def _allocate_oracle_observations(
    schema: ForwardBackwardObservationSchema,
    batch_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> TensorDict:
    return TensorDict(
        {name: torch.zeros(*batch_shape, width, device=device, dtype=dtype) for name, width in schema.field_widths},
        batch_size=list(batch_shape),
        device=device,
    )
