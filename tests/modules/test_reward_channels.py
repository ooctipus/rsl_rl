# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for forward-backward reward and value-channel contracts."""

from __future__ import annotations

import torch
from dataclasses import replace

import pytest

from rsl_rl.modules.reward_channels import (
    ForwardBackwardRewardChannel,
    ForwardBackwardRewardNormalizer,
    ForwardBackwardRewardSchema,
    ForwardBackwardValueSpec,
    get_forward_backward_schema_hash,
)


def _make_reward_schema() -> ForwardBackwardRewardSchema:
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
                name="energy_penalty",
                provider_name="auxiliary",
                source="stored_evidence",
                timing="state",
                context_dependent=False,
                sign=-1,
            ),
        )
    )


def _make_value_spec() -> ForwardBackwardValueSpec:
    return ForwardBackwardValueSpec(
        name="auxiliary",
        kind="critic",
        route="critic_auxiliary",
        reward_channels=("environment", "energy_penalty"),
        ensemble_size=2,
        has_target=True,
    )


def test_reward_schema_preserves_channel_and_provider_order() -> None:
    """Tensor columns and first-use provider order should remain explicit."""
    schema = _make_reward_schema()

    assert schema.channel_names == ("environment", "discriminator", "energy_penalty")
    assert schema.provider_names == ("environment", "discriminator", "auxiliary")


def test_reward_schema_copies_the_channel_sequence() -> None:
    """Later edits to a caller list should not change a constructed schema."""
    channels = list(_make_reward_schema().channels)
    schema = ForwardBackwardRewardSchema(channels=channels)
    channels.reverse()

    assert schema.channel_names == ("environment", "discriminator", "energy_penalty")


def test_reward_channel_order_changes_schema_identity() -> None:
    """Reward column order should be part of the schema fingerprint."""
    schema = _make_reward_schema()
    reordered = ForwardBackwardRewardSchema(channels=tuple(reversed(schema.channels)))

    assert reordered.schema_hash != schema.schema_hash


def test_reward_semantics_change_schema_identity() -> None:
    """Every field that changes reward meaning should affect the fingerprint."""
    schema = _make_reward_schema()
    channel = schema.channels[0]
    variants = (
        replace(channel, name="task"),
        replace(channel, provider_name="task_provider"),
        replace(channel, source="stored_evidence"),
        replace(channel, timing="next_state"),
        replace(channel, context_dependent=True),
        replace(channel, sign=-1),
    )

    assert all(
        ForwardBackwardRewardSchema(channels=(variant, *schema.channels[1:])).schema_hash != schema.schema_hash
        for variant in variants
    )


def test_duplicate_reward_channel_name_fails() -> None:
    """Two semantic rewards should not share one tensor-column name."""
    channel = _make_reward_schema().channels[0]

    with pytest.raises(ValueError, match="must be unique"):
        ForwardBackwardRewardSchema(channels=(channel, replace(channel, provider_name="other")))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("source", "implicit", "Unsupported reward source"),
        ("timing", "predictive", "Unsupported reward timing"),
        ("sign", 0, "either -1 or 1"),
        ("sign", True, "either -1 or 1"),
    ),
)
def test_invalid_reward_semantics_fail(field: str, value: object, message: str) -> None:
    """Invalid source, timing, and sign choices should fail at construction."""
    channel = _make_reward_schema().channels[0]

    with pytest.raises(ValueError, match=message):
        replace(channel, **{field: value})


def test_value_spec_describes_one_reward_subset() -> None:
    """A value source should retain only its compact static description."""
    schema = _make_reward_schema()
    spec = _make_value_spec()

    spec.validate_reward_schema(schema)
    assert spec.reward_channels == ("environment", "energy_penalty")
    assert spec.output_width == 2
    assert spec.ensemble_size == 2
    assert spec.has_target


def test_value_source_rejects_unknown_reward_channel() -> None:
    """A value source should not consume a reward outside the schema."""
    spec = replace(_make_value_spec(), reward_channels=("environment", "ghost"))

    with pytest.raises(ValueError, match="unknown reward channels"):
        spec.validate_reward_schema(_make_reward_schema())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("kind", "state_value", "Unsupported value kind"),
        ("reward_channels", (), "At least one reward channel"),
        ("reward_channels", ("environment", "environment"), "must be unique"),
        ("ensemble_size", 0, "must be positive"),
        ("reward_composition", "matrix", "Unsupported reward composition"),
    ),
)
def test_invalid_value_semantics_fail(field: str, value: object, message: str) -> None:
    """Ambiguous value-source descriptions should fail at construction."""
    with pytest.raises(ValueError, match=message):
        replace(_make_value_spec(), **{field: value})


def test_scalar_reward_composition_predicts_one_value() -> None:
    """A scalar helper should combine its declared channels before TD propagation."""
    spec = replace(_make_value_spec(), reward_composition="scalar")

    assert spec.reward_channels == ("environment", "energy_penalty")
    assert spec.output_width == 1


def test_schema_hash_is_mapping_order_independent_but_sequence_sensitive() -> None:
    """Schema fingerprints should preserve semantic sequence order only."""
    left = get_forward_backward_schema_hash({"b": 2, "a": [1, 3]})
    right = get_forward_backward_schema_hash({"a": [1, 3], "b": 2})
    reordered = get_forward_backward_schema_hash({"a": [3, 1], "b": 2})

    assert left == right
    assert reordered != left


def test_reward_normalizer_separates_update_from_apply_and_preserves_composition() -> None:
    """One shared scale should commute with fixed linear reward composition."""
    normalizer = ForwardBackwardRewardNormalizer((0.5, 2.0), decay=0.5, epsilon=1e-12)
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    before = normalizer(rewards)
    normalizer.update(rewards)
    after = normalizer(rewards)
    composed_after = after @ normalizer.coefficients
    direct_after = (rewards @ normalizer.coefficients) / normalizer.scale

    torch.testing.assert_close(before, rewards)
    torch.testing.assert_close(composed_after, direct_after)
    assert normalizer.count.item() == 1
    normalizer(rewards)
    assert normalizer.count.item() == 1


def test_reward_normalizer_matches_bias_corrected_ema_reference() -> None:
    """Running mean-square variance should match the released BFM formula."""
    normalizer = ForwardBackwardRewardNormalizer((1.0, -0.5), decay=0.75, epsilon=1e-8)
    batches = (
        torch.tensor([[2.0, 0.0], [4.0, 2.0]]),
        torch.tensor([[1.0, 2.0], [5.0, 0.0]]),
    )
    mean = torch.zeros(1)
    mean_square = torch.zeros(1)
    for count, rewards in enumerate(batches, start=1):
        composed = rewards @ normalizer.coefficients
        mean = 0.75 * mean + 0.25 * composed.mean()
        mean_square = 0.75 * mean_square + 0.25 * composed.square().mean()
        correction = 1.0 - 0.75**count
        expected = torch.sqrt(torch.clamp(mean_square / correction - (mean / correction).square(), min=1e-8))
        normalizer.update(rewards)
        torch.testing.assert_close(normalizer.scale, expected)


def test_reward_normalizer_rejects_wrong_channel_width() -> None:
    """A value head must not silently normalize another reward schema."""
    normalizer = ForwardBackwardRewardNormalizer((1.0, 2.0))

    with pytest.raises(ValueError, match="channel_count"):
        normalizer.update(torch.ones(3, 1))
