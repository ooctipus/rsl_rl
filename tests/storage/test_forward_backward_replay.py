# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for forward-backward transition semantics."""

from __future__ import annotations

import torch
from dataclasses import fields, replace

import pytest

from rsl_rl.storage.forward_backward_replay import (
    ForwardBackwardAutoresetMode,
    ForwardBackwardTransitionBatch,
)
from tests.fixtures.forward_backward import make_transition


@pytest.mark.parametrize("mode", tuple(ForwardBackwardAutoresetMode))
def test_all_autoreset_modes_have_one_compatible_contract(mode: ForwardBackwardAutoresetMode) -> None:
    """Every autoreset mode should share one normalized transition type."""
    transition, schema, observation_schema, reward_schema = make_transition(mode)

    schema.assert_compatible(observation_schema, reward_schema)
    transition.assert_valid(schema, observation_schema)
    assert transition.contract_error_mask(schema).squeeze(-1).tolist() == [False] * 4


def test_transition_fields_are_direct_tensor_evidence() -> None:
    """Collection should not store speculative values or replay-slot metadata."""
    assert tuple(field.name for field in fields(ForwardBackwardTransitionBatch)) == (
        "observations",
        "next_observations",
        "final_observations",
        "actions",
        "behavior_context",
        "environment_reward",
        "auxiliary_reward_evidence",
        "terminated",
        "truncated",
        "context_changed",
        "action_applied",
        "final_observation_valid",
    )


def test_masks_separate_termination_truncation_and_context_boundaries() -> None:
    """Termination, truncation, and context changes should have distinct roles."""
    transition, schema, _observation_schema, _reward_schema = make_transition()

    assert transition.replay_mask(schema).squeeze(-1).tolist() == [True, True, True, True]
    assert transition.done_mask().squeeze(-1).tolist() == [False, True, True, False]
    assert transition.bootstrap_mask(schema).squeeze(-1).tolist() == [True, False, True, True]
    assert transition.episode_continuation_mask(schema).squeeze(-1).tolist() == [True, False, False, True]
    assert transition.segment_continuation_mask(schema).squeeze(-1).tolist() == [True, False, False, False]


@pytest.mark.parametrize(
    ("mode", "reached_final", "bootstrap_final"),
    (
        (ForwardBackwardAutoresetMode.DISABLED, [False] * 4, [False] * 4),
        (ForwardBackwardAutoresetMode.SAME_STEP, [False, True, True, False], [False, False, True, False]),
        (ForwardBackwardAutoresetMode.NEXT_STEP, [False] * 4, [False] * 4),
    ),
)
def test_successor_source_never_uses_same_step_reset_observation(
    mode: ForwardBackwardAutoresetMode,
    reached_final: list[bool],
    bootstrap_final: list[bool],
) -> None:
    """Reached and bootstrap states should select true finals explicitly."""
    transition, schema, _observation_schema, _reward_schema = make_transition(mode)

    assert transition.reached_observation_uses_final(schema).squeeze(-1).tolist() == reached_final
    assert transition.bootstrap_observation_uses_final(schema).squeeze(-1).tolist() == bootstrap_final


def test_termination_dominates_truncation_for_bootstrap() -> None:
    """A row carrying both done flags should never bootstrap."""
    transition, schema, observation_schema, _reward_schema = make_transition()
    terminated = transition.terminated.clone()
    terminated[2] = True
    transition = replace(transition, terminated=terminated)

    transition.assert_valid(schema, observation_schema)
    assert transition.reached_observation_uses_final(schema)[2]
    assert not transition.bootstrap_mask(schema)[2]


def test_missing_same_step_final_is_masked_without_reset_fallback() -> None:
    """Malformed done rows should be excluded on-device and fail debug validation."""
    transition, schema, observation_schema, _reward_schema = make_transition()
    valid = transition.final_observation_valid.clone()
    valid[2] = False
    malformed = replace(transition, final_observation_valid=valid)

    assert malformed.contract_error_mask(schema).squeeze(-1).tolist() == [False, False, True, False]
    assert not malformed.replay_mask(schema)[2]
    assert not malformed.bootstrap_mask(schema)[2]
    assert not malformed.reached_observation_uses_final(schema)[2]
    with pytest.raises(ValueError, match="true final observation"):
        malformed.assert_valid(schema, observation_schema)


def test_same_step_rejects_stale_final_validity_on_continuing_row() -> None:
    """Debug validation should detect a final payload attached to a continuing row."""
    transition, schema, observation_schema, _reward_schema = make_transition()
    valid = transition.final_observation_valid.clone()
    valid[0] = True
    malformed = replace(transition, final_observation_valid=valid)

    assert malformed.contract_error_mask(schema).squeeze(-1).tolist() == [True, False, False, False]
    assert not malformed.replay_mask(schema)[0]
    with pytest.raises(ValueError, match="true final observation"):
        malformed.assert_valid(schema, observation_schema)


def test_invalid_final_observation_values_are_unspecified() -> None:
    """Masks, rather than zero-fill scans, should control reads from dense final storage."""
    transition, schema, observation_schema, _reward_schema = make_transition()
    finals = transition.final_observations.clone()
    finals["state"][~transition.final_observation_valid.squeeze(-1)] = float("nan")

    replace(transition, final_observations=finals).assert_valid(schema, observation_schema)


@pytest.mark.parametrize("mode", (ForwardBackwardAutoresetMode.DISABLED, ForwardBackwardAutoresetMode.NEXT_STEP))
def test_non_same_step_modes_reject_separate_final_payload(mode: ForwardBackwardAutoresetMode) -> None:
    """Non-same-step modes receive the reached state directly from step."""
    transition, schema, observation_schema, _reward_schema = make_transition(mode)
    valid = transition.final_observation_valid.clone()
    valid[1] = True
    malformed = replace(transition, final_observation_valid=valid)

    assert malformed.contract_error_mask(schema).squeeze(-1).tolist() == [False, True, False, False]
    assert not malformed.replay_mask(schema)[1]
    with pytest.raises(ValueError, match="must not provide separate final observations"):
        malformed.assert_valid(schema, observation_schema)


def test_next_step_reset_only_row_is_ignored_without_zero_fill_requirements() -> None:
    """A reset-only row may contain stale storage but never enters replay or bootstrap."""
    transition, schema, observation_schema, _reward_schema = make_transition(ForwardBackwardAutoresetMode.NEXT_STEP)
    applied = transition.action_applied.clone()
    applied[0] = False
    reset_batch = replace(transition, action_applied=applied)

    reset_batch.assert_valid(schema, observation_schema)
    assert reset_batch.contract_error_mask(schema).squeeze(-1).tolist() == [False] * 4
    assert not reset_batch.replay_mask(schema)[0]
    assert not reset_batch.bootstrap_mask(schema)[0]
    assert not reset_batch.episode_continuation_mask(schema)[0]


def test_same_step_rejects_reset_only_row_in_debug_validation() -> None:
    """Reset-only rows belong only to next-step autoreset."""
    transition, schema, observation_schema, _reward_schema = make_transition()
    applied = transition.action_applied.clone()
    applied[0] = False
    malformed = replace(transition, action_applied=applied)

    assert malformed.contract_error_mask(schema).squeeze(-1).tolist() == [True, False, False, False]
    with pytest.raises(ValueError, match="cannot contain reset-only rows"):
        malformed.assert_valid(schema, observation_schema)


def test_disabled_autoreset_rejects_reset_only_row() -> None:
    """Only next-step autoreset may emit a row without an applied action."""
    transition, schema, observation_schema, _reward_schema = make_transition(ForwardBackwardAutoresetMode.DISABLED)
    applied = transition.action_applied.clone()
    applied[0] = False
    malformed = replace(transition, action_applied=applied)

    assert malformed.contract_error_mask(schema).squeeze(-1).tolist() == [True, False, False, False]
    assert not malformed.replay_mask(schema)[0]
    with pytest.raises(ValueError, match="reset-only rows"):
        malformed.assert_valid(schema, observation_schema)


def test_debug_validation_checks_tensor_layout_once() -> None:
    """A malformed evidence width should fail the optional adapter check."""
    transition, schema, observation_schema, _reward_schema = make_transition()
    malformed = replace(transition, auxiliary_reward_evidence=torch.randn(4, 1))

    with pytest.raises(ValueError, match="auxiliary_reward_evidence"):
        malformed.assert_valid(schema, observation_schema)


def test_debug_validation_reduces_contract_errors_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean debug check should perform one control-boundary device reduction."""
    transition, schema, observation_schema, _reward_schema = make_transition()
    original_any = torch.any
    num_calls = 0

    def counted_any(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal num_calls
        num_calls += 1
        return original_any(*args, **kwargs)

    monkeypatch.setattr(torch, "any", counted_any)

    transition.assert_valid(schema, observation_schema)

    assert num_calls == 1


def test_hot_masks_do_not_call_reduction_validators(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collection masks should remain free of GPU-to-host reduction branches."""
    transition, schema, _observation_schema, _reward_schema = make_transition()

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("hot masks called a reduction validator")

    monkeypatch.setattr(torch, "any", forbidden)
    monkeypatch.setattr(torch, "all", forbidden)
    monkeypatch.setattr(torch, "equal", forbidden)
    transition.contract_error_mask(schema)
    transition.replay_mask(schema)
    transition.bootstrap_mask(schema)
    transition.episode_continuation_mask(schema)
    transition.segment_continuation_mask(schema)
    transition.reached_observation_uses_final(schema)
    transition.bootstrap_observation_uses_final(schema)


def test_transition_schema_copies_auxiliary_evidence_names() -> None:
    """Caller list mutation must not change a frozen schema or stale its hash."""
    _transition, schema, _observation_schema, _reward_schema = make_transition()
    names = list(schema.auxiliary_evidence_names)
    copied = replace(schema, auxiliary_evidence_names=names)
    schema_hash = copied.schema_hash

    names.append("late_channel")

    assert copied.auxiliary_evidence_names == schema.auxiliary_evidence_names
    assert copied.schema_hash == schema_hash
    assert copied.schema_hash == schema.schema_hash


def test_transition_schema_identity_includes_autoreset_mode() -> None:
    """Replay must not restore under a different reset convention."""
    _transition, same_step, _observation_schema, _reward_schema = make_transition()
    next_step = replace(same_step, autoreset_mode=ForwardBackwardAutoresetMode.NEXT_STEP)

    assert next_step.schema_hash != same_step.schema_hash


def test_recurrent_information_state_requires_schema_bump() -> None:
    """Feedforward finals do not imply correct recurrent terminal state."""
    _transition, schema, _observation_schema, _reward_schema = make_transition()

    with pytest.raises(ValueError, match="new transition schema version"):
        replace(schema, information_state="recurrent")


def test_schema_compatibility_rejects_observation_or_reward_mismatch() -> None:
    """Construction should reject replay interpreted under different schemas."""
    _transition, schema, observation_schema, reward_schema = make_transition()

    with pytest.raises(ValueError, match="observation schemas"):
        replace(schema, observation_schema_hash="different").assert_compatible(observation_schema, reward_schema)
    reversed_reward = replace(reward_schema, channels=tuple(reversed(reward_schema.channels)))
    with pytest.raises(ValueError, match="reward schemas"):
        schema.assert_compatible(observation_schema, reversed_reward)


def test_schema_compatibility_preserves_reward_source_semantics() -> None:
    """Stored tensor columns must not masquerade as learned reward providers."""
    _transition, schema, observation_schema, reward_schema = make_transition()

    with pytest.raises(ValueError, match="environment reward channel"):
        replace(schema, environment_reward_name="discriminator").assert_compatible(observation_schema, reward_schema)
    with pytest.raises(ValueError, match="stored-evidence"):
        replace(schema, auxiliary_evidence_names=("action_rate", "discriminator")).assert_compatible(
            observation_schema, reward_schema
        )


def test_unknown_reward_evidence_channel_fails_at_construction() -> None:
    """Evidence column names should resolve once when replay is built."""
    _transition, schema, observation_schema, reward_schema = make_transition()

    with pytest.raises(ValueError, match="unknown reward channel"):
        replace(schema, auxiliary_evidence_names=("action_rate", "ghost")).assert_compatible(
            observation_schema, reward_schema
        )
