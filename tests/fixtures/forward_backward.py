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
