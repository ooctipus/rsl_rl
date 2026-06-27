# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expert-corpus metadata for forward-backward learning."""

from __future__ import annotations

from dataclasses import dataclass, field

from rsl_rl.modules.reward_channels import get_forward_backward_schema_hash


@dataclass(frozen=True, slots=True)
class ForwardBackwardExpertSchema:
    """Identity and compact shape of one ragged expert corpus."""

    dataset_id: str
    data_hash: str
    feature_schema_hash: str
    clip_offsets_hash: str
    expert_feature_width: int
    num_frames: int
    num_clips: int
    window_lengths: tuple[int, ...]
    schema_hash: str = field(init=False)

    def __post_init__(self) -> None:
        """Normalize sampling windows and fingerprint concrete corpus metadata."""
        if not self.dataset_id:
            raise ValueError("dataset_id must not be empty.")
        if self.expert_feature_width < 1:
            raise ValueError("expert_feature_width must be positive.")
        if self.num_frames < 1:
            raise ValueError("num_frames must be positive.")
        if self.num_clips < 1:
            raise ValueError("num_clips must be positive.")
        if self.num_frames < self.num_clips:
            raise ValueError("num_frames must be at least num_clips.")

        window_lengths = tuple(sorted(set(self.window_lengths)))
        if not window_lengths or any(length < 1 for length in window_lengths):
            raise ValueError("Expert window lengths must contain positive values.")
        object.__setattr__(self, "window_lengths", window_lengths)
        object.__setattr__(
            self,
            "schema_hash",
            get_forward_backward_schema_hash({
                "clip_offsets_hash": self.clip_offsets_hash,
                "data_hash": self.data_hash,
                "dataset_id": self.dataset_id,
                "expert_feature_width": self.expert_feature_width,
                "feature_schema_hash": self.feature_schema_hash,
                "num_clips": self.num_clips,
                "num_frames": self.num_frames,
                "window_lengths": window_lengths,
            }),
        )
