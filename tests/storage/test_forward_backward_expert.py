# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for forward-backward expert-corpus metadata."""

from __future__ import annotations

from dataclasses import replace

import pytest

from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertSchema


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
