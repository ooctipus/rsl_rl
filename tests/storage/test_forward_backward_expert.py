# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for forward-backward expert-corpus metadata."""

from __future__ import annotations

import torch
from dataclasses import replace

import pytest

from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer, ForwardBackwardExpertSchema


def _make_schema() -> ForwardBackwardExpertSchema:
    return ForwardBackwardExpertSchema(
        dataset_id="motion-corpus-v1",
        data_hash="data-v1",
        feature_schema_hash="features-v1",
        clip_offsets_hash="clips-v1",
        expert_feature_width=527,
        num_frames=100_000,
        num_clips=200,
        window_lengths=(1, 8, 250),
    )


def test_expert_schema_identity_is_stable() -> None:
    """Equivalent corpus metadata should have the same fingerprint."""
    assert _make_schema().schema_hash == _make_schema().schema_hash


def test_expert_schema_retains_concrete_shape_metadata() -> None:
    """Phase 1D should receive only corpus identity, shape, and windows."""
    schema = _make_schema()

    assert schema.dataset_id == "motion-corpus-v1"
    assert schema.data_hash == "data-v1"
    assert schema.feature_schema_hash == "features-v1"
    assert schema.clip_offsets_hash == "clips-v1"
    assert schema.expert_feature_width == 527
    assert schema.num_frames == 100_000
    assert schema.num_clips == 200
    assert schema.window_lengths == (1, 8, 250)


def test_window_lengths_are_copied_and_canonicalized() -> None:
    """Sampling-window sets should have one stable representation."""
    lengths = [8, 1, 8]
    schema = replace(_make_schema(), window_lengths=lengths)
    lengths.append(16)

    assert schema.window_lengths == (1, 8)


@pytest.mark.parametrize("window_lengths", ((), (0, 8), (-1, 8)))
def test_window_lengths_must_be_positive(window_lengths: tuple[int, ...]) -> None:
    """A corpus must expose at least one positive sampling window."""
    with pytest.raises(ValueError, match="window lengths"):
        replace(_make_schema(), window_lengths=window_lengths)


@pytest.mark.parametrize("field", ("expert_feature_width", "num_frames", "num_clips"))
def test_shape_metadata_must_be_positive(field: str) -> None:
    """Concrete corpus dimensions and counts must be positive."""
    with pytest.raises(ValueError, match="must be positive"):
        replace(_make_schema(), **{field: 0})


def test_corpus_must_have_at_least_one_frame_per_clip() -> None:
    """Impossible aggregate clip topology should fail early."""
    with pytest.raises(ValueError, match="at least num_clips"):
        replace(_make_schema(), num_frames=100, num_clips=200)


def test_concrete_corpus_metadata_changes_schema_identity() -> None:
    """Every retained identity or shape field should affect the fingerprint."""
    schema = _make_schema()
    variants = (
        replace(schema, dataset_id="motion-corpus-v2"),
        replace(schema, data_hash="data-v2"),
        replace(schema, feature_schema_hash="features-v2"),
        replace(schema, clip_offsets_hash="clips-v2"),
        replace(schema, expert_feature_width=526),
        replace(schema, num_frames=99_999),
        replace(schema, num_clips=199),
        replace(schema, window_lengths=(1, 8)),
    )

    assert all(variant.schema_hash != schema.schema_hash for variant in variants)


def _make_buffer(seed: int = 3) -> ForwardBackwardExpertBuffer:
    schema = replace(
        _make_schema(),
        expert_feature_width=3,
        num_frames=14,
        num_clips=3,
        window_lengths=(1, 3, 5),
    )
    frames = torch.arange(42, dtype=torch.float32).reshape(14, 3)
    clip_offsets = torch.tensor([0, 2, 8, 14], dtype=torch.long)
    priorities = torch.tensor([100.0, 1.0, 2.0])
    return ForwardBackwardExpertBuffer(frames, clip_offsets, priorities, schema, seed)


def test_expert_windows_are_contiguous_and_never_cross_clips() -> None:
    """Every sampled index matrix should stay within its selected ragged clip."""
    buffer = _make_buffer()
    batch = buffer.sample(128, 3)
    assert batch.frame_indices.shape == (128, 4)
    starts = buffer.clip_offsets[batch.clip_ids]
    stops = buffer.clip_offsets[batch.clip_ids + 1]

    assert torch.all(batch.frame_indices[:, 1:] == batch.frame_indices[:, :-1] + 1)
    assert torch.all(batch.frame_indices >= starts.unsqueeze(-1))
    assert torch.all(batch.frame_indices < stops.unsqueeze(-1))
    assert torch.all(batch.clip_ids != 0)
    torch.testing.assert_close(batch.observations, buffer.frames[batch.frame_indices[:, :-1]])
    torch.testing.assert_close(batch.next_observations, buffer.frames[batch.frame_indices[:, 1:]])


def test_expert_sampler_resumes_exactly() -> None:
    """The dedicated device generator should produce the same next windows after restore."""
    buffer = _make_buffer()
    buffer.set_priorities(torch.tensor([0.0, 3.0, 1.0]))
    buffer.sample(7, 3)
    state = buffer.state_dict()
    restored = _make_buffer(seed=999)
    restored.load_state_dict(state)

    expected = buffer.sample(11, 5)
    actual = restored.sample(11, 5)
    torch.testing.assert_close(actual.observations, expected.observations)
    torch.testing.assert_close(actual.next_observations, expected.next_observations)
    torch.testing.assert_close(actual.clip_ids, expected.clip_ids)
    torch.testing.assert_close(actual.frame_indices, expected.frame_indices)


def test_expert_priority_event_updates_all_windows_and_checkpoint_state() -> None:
    """External weights should immediately govern every window and survive restore."""
    buffer = _make_buffer()
    buffer.set_priorities(torch.tensor([0.0, 1.0, 0.0]))

    assert torch.all(buffer.sample(128, 1).clip_ids == 1)
    assert torch.all(buffer.sample(128, 3).clip_ids == 1)
    assert torch.all(buffer.sample(128, 5).clip_ids == 1)

    state = buffer.state_dict()
    buffer.set_priorities(torch.tensor([0.0, 0.0, 1.0]))
    restored = _make_buffer(seed=999)
    restored.load_state_dict(state)

    torch.testing.assert_close(restored.priorities, torch.tensor([0.0, 1.0, 0.0]))
    assert torch.all(restored.sample(128, 5).clip_ids == 1)


@pytest.mark.parametrize(
    "priorities",
    (
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, -1.0, 1.0]),
        torch.tensor([1.0, float("nan"), 1.0]),
        torch.tensor([1.0, 0.0]),
    ),
)
def test_expert_priority_event_rejects_invalid_weights(priorities: torch.Tensor) -> None:
    """Invalid external weights should fail before mutating sampler state."""
    buffer = _make_buffer()
    original = buffer.priorities.clone()

    with pytest.raises(ValueError, match="priorities"):
        buffer.set_priorities(priorities)

    torch.testing.assert_close(buffer.priorities, original)


def test_expert_sampler_state_rejects_another_corpus() -> None:
    """Mutable RNG state must not move between different immutable corpora."""
    buffer = _make_buffer()
    state = buffer.state_dict()
    changed = _make_buffer()
    changed.schema = replace(changed.schema, dataset_id="another-corpus")

    with pytest.raises(ValueError, match="corpus schema"):
        changed.load_state_dict(state)


def test_expert_sampler_rejects_unavailable_window() -> None:
    """Only lengths frozen in the expert schema should be sampleable."""
    with pytest.raises(ValueError, match="Unsupported expert sequence length"):
        _make_buffer().sample(4, 4)
