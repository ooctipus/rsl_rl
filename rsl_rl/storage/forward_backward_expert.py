# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expert-corpus metadata for forward-backward learning."""

from __future__ import annotations

import torch
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


@dataclass(frozen=True, slots=True)
class ForwardBackwardExpertBatch:
    """Contiguous expert windows and their source indices."""

    frames: torch.Tensor
    clip_ids: torch.Tensor
    frame_indices: torch.Tensor


class ForwardBackwardExpertBuffer:
    """Immutable ragged expert frames with mutable sampling weights and a device generator."""

    def __init__(
        self,
        frames: torch.Tensor,
        clip_offsets: torch.Tensor,
        priorities: torch.Tensor,
        schema: ForwardBackwardExpertSchema,
        seed: int = 0,
    ) -> None:
        """Validate and retain one GPU corpus without copying it to the host."""
        if frames.ndim != 2 or tuple(frames.shape) != (schema.num_frames, schema.expert_feature_width):
            raise ValueError("frames do not match the expert schema.")
        if clip_offsets.shape != (schema.num_clips + 1,) or clip_offsets.dtype is not torch.long:
            raise ValueError("clip_offsets must be int64 with num_clips + 1 entries.")
        if priorities.shape != (schema.num_clips,) or not priorities.is_floating_point():
            raise ValueError("priorities must be floating point with one entry per clip.")
        if frames.device != clip_offsets.device or frames.device != priorities.device:
            raise ValueError("Expert frames, offsets, and priorities must share one device.")
        if frames.requires_grad or priorities.requires_grad:
            raise ValueError("Expert corpus tensors must be detached.")
        if clip_offsets[0] != 0 or clip_offsets[-1] != schema.num_frames:
            raise ValueError("clip_offsets must span the complete frame tensor.")
        clip_lengths = clip_offsets[1:] - clip_offsets[:-1]
        if torch.any(clip_lengths <= 0):
            raise ValueError("Every expert clip must contain at least one frame.")
        if not torch.all(torch.isfinite(priorities)):
            raise ValueError("Expert priorities must be finite.")
        if torch.any(priorities < 0) or not torch.any(priorities > 0):
            raise ValueError("Expert priorities must be non-negative with positive total mass.")

        self.frames = frames
        self.clip_offsets = clip_offsets
        self.priorities = priorities
        self.schema = schema
        self.device = frames.device
        self.clip_lengths = clip_lengths
        self._sequence_offsets = {
            length: torch.arange(length, device=self.device, dtype=torch.long) for length in schema.window_lengths
        }
        self._eligible_priorities = {
            length: torch.where(clip_lengths >= length, priorities, torch.zeros_like(priorities))
            for length in schema.window_lengths
        }
        if any(not torch.any(value > 0) for value in self._eligible_priorities.values()):
            raise ValueError("Every configured window length needs a positive-priority eligible clip.")
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def set_priorities(self, priorities: torch.Tensor) -> None:
        """Replace clip-sampling weights at one declared external priority event."""
        if priorities.shape != self.priorities.shape or not priorities.is_floating_point():
            raise ValueError("priorities must be floating point with one entry per clip.")
        if priorities.device != self.device:
            raise ValueError("Expert priorities must remain on the corpus device.")
        if priorities.requires_grad:
            raise ValueError("Expert priorities must be detached.")
        if not torch.all(torch.isfinite(priorities)):
            raise ValueError("Expert priorities must be finite.")
        if torch.any(priorities < 0) or not torch.any(priorities > 0):
            raise ValueError("Expert priorities must be non-negative with positive total mass.")
        eligible_priorities = {
            length: torch.where(self.clip_lengths >= length, priorities, torch.zeros_like(priorities))
            for length in self.schema.window_lengths
        }
        if any(not torch.any(value > 0) for value in eligible_priorities.values()):
            raise ValueError("Every configured window length needs a positive-priority eligible clip.")
        self.priorities.copy_(priorities)
        for length, values in eligible_priorities.items():
            self._eligible_priorities[length].copy_(values)

    def sample(self, batch_size: int, sequence_length: int) -> ForwardBackwardExpertBatch:
        """Sample clips by priority and starts uniformly within each selected clip."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        try:
            sequence_offsets = self._sequence_offsets[sequence_length]
        except KeyError as error:
            raise ValueError(f"Unsupported expert sequence length: {sequence_length}.") from error
        num_starts = self.clip_lengths - sequence_length + 1
        clip_ids = torch.multinomial(
            self._eligible_priorities[sequence_length],
            batch_size,
            replacement=True,
            generator=self.generator,
        )
        start_counts = num_starts[clip_ids]
        start_offsets = torch.floor(
            torch.rand(batch_size, device=self.device, generator=self.generator) * start_counts
        ).long()
        starts = self.clip_offsets[clip_ids] + start_offsets
        frame_indices = starts.unsqueeze(-1) + sequence_offsets
        return ForwardBackwardExpertBatch(
            frames=self.frames[frame_indices],
            clip_ids=clip_ids,
            frame_indices=frame_indices,
        )

    def state_dict(self) -> dict[str, object]:
        """Capture only mutable sampling state; corpus tensors are immutable inputs."""
        return {
            "schema_hash": self.schema.schema_hash,
            "priorities": self.priorities.clone(),
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore the exact next sample under the same corpus identity."""
        if state["schema_hash"] != self.schema.schema_hash:
            raise ValueError("Expert sampler state does not match the corpus schema.")
        priorities = state["priorities"]
        if not isinstance(priorities, torch.Tensor):
            raise TypeError("Expert priorities state must be a tensor.")
        self.set_priorities(priorities)
        generator_state = state["generator_state"]
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("Expert generator_state must be a tensor.")
        self.generator.set_state(generator_state)
