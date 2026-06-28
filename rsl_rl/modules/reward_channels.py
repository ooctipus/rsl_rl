# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static reward and value-channel contracts for forward-backward learning."""

from __future__ import annotations

import hashlib
import json
import torch
from dataclasses import dataclass, field
from typing import Literal

ForwardBackwardRewardTiming = Literal["state", "next_state", "transition"]
ForwardBackwardRewardSource = Literal["environment", "stored_evidence", "recomputed"]
ForwardBackwardValueKind = Literal["forward_readout", "critic"]
ForwardBackwardRewardComposition = Literal["vector", "scalar"]


def get_forward_backward_schema_hash(value: object) -> str:
    """Return a stable SHA-256 fingerprint for JSON-compatible schema data.

    Mapping order does not affect the fingerprint. Sequence order remains
    significant because it commonly defines tensor-column order.

    Args:
        value: JSON-compatible schema data.

    Returns:
        Lowercase hexadecimal SHA-256 fingerprint.
    """
    payload = json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ForwardBackwardRewardChannel:
    """Semantic definition of one scalar immediate-reward channel.

    State timing means the pre-action information state. Next-state timing
    means the logical reached information state: a true final state on done,
    never a same-step reset observation. Transition timing may consume the
    complete applied edge. Providers emit pre-sign evidence and composition
    applies sign exactly once.
    """

    name: str
    provider_name: str
    source: ForwardBackwardRewardSource
    timing: ForwardBackwardRewardTiming
    context_dependent: bool
    sign: Literal[-1, 1]

    def __post_init__(self) -> None:
        """Reject invalid channel semantics at construction."""
        if not self.name:
            raise ValueError("Reward channel name must not be empty.")
        if not self.provider_name:
            raise ValueError("Reward provider name must not be empty.")
        if self.source not in ("environment", "stored_evidence", "recomputed"):
            raise ValueError(f"Unsupported reward source: {self.source!r}.")
        if self.timing not in ("state", "next_state", "transition"):
            raise ValueError(f"Unsupported reward timing: {self.timing!r}.")
        if isinstance(self.sign, bool) or self.sign not in (-1, 1):
            raise ValueError("sign must be either -1 or 1.")

    def to_schema_data(self) -> dict[str, object]:
        """Return the semantic fields used by the schema fingerprint."""
        return {
            "context_dependent": self.context_dependent,
            "name": self.name,
            "provider_name": self.provider_name,
            "source": self.source,
            "sign": self.sign,
            "timing": self.timing,
        }


@dataclass(frozen=True, slots=True)
class ForwardBackwardRewardSchema:
    """Ordered immediate-reward channels used for tensor composition."""

    channels: tuple[ForwardBackwardRewardChannel, ...]
    schema_hash: str = field(init=False)

    def __post_init__(self) -> None:
        """Copy channels and fingerprint their static semantics."""
        channels = tuple(self.channels)
        if not channels:
            raise ValueError("At least one reward channel is required.")
        channel_names = tuple(channel.name for channel in channels)
        if len(channel_names) != len(set(channel_names)):
            raise ValueError("Reward channel names must be unique.")
        object.__setattr__(self, "channels", channels)
        object.__setattr__(
            self,
            "schema_hash",
            get_forward_backward_schema_hash([channel.to_schema_data() for channel in channels]),
        )

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Ordered reward-channel names."""
        return tuple(channel.name for channel in self.channels)

    @property
    def provider_names(self) -> tuple[str, ...]:
        """Provider names in first-use order."""
        return tuple(dict.fromkeys(channel.provider_name for channel in self.channels))


@dataclass(frozen=True, slots=True)
class ForwardBackwardValueSpec:
    """Compact description of one value source and its reward channels."""

    name: str
    kind: ForwardBackwardValueKind
    route: str
    reward_channels: tuple[str, ...]
    ensemble_size: int
    has_target: bool
    reward_composition: ForwardBackwardRewardComposition = "vector"

    def __post_init__(self) -> None:
        """Normalize channel names and reject ambiguous value semantics."""
        if not self.name:
            raise ValueError("Value source name must not be empty.")
        if self.kind not in ("forward_readout", "critic"):
            raise ValueError(f"Unsupported value kind: {self.kind!r}.")
        if not self.route:
            raise ValueError("Value source route must not be empty.")
        reward_channels = tuple(self.reward_channels)
        if not reward_channels:
            raise ValueError("At least one reward channel is required by a value source.")
        if len(reward_channels) != len(set(reward_channels)):
            raise ValueError("Value-source reward channels must be unique.")
        if self.ensemble_size < 1:
            raise ValueError("ensemble_size must be positive.")
        if self.reward_composition not in ("vector", "scalar"):
            raise ValueError(f"Unsupported reward composition: {self.reward_composition!r}.")
        object.__setattr__(self, "reward_channels", reward_channels)

    @property
    def output_width(self) -> int:
        """Number of values predicted by this head."""
        return len(self.reward_channels) if self.reward_composition == "vector" else 1

    def validate_reward_schema(self, reward_schema: ForwardBackwardRewardSchema) -> None:
        """Reject references to channels outside the reward schema."""
        unknown = set(self.reward_channels).difference(reward_schema.channel_names)
        if unknown:
            raise ValueError(f"Value source uses unknown reward channels: {tuple(sorted(unknown))}.")


class ForwardBackwardRewardNormalizer(torch.nn.Module):
    """Bias-corrected EMA scale shared by one reward-value vector.

    Statistics are estimated from a fixed linear composition, while the same
    scalar standard deviation is applied to every channel. This preserves exact
    linear reward composition and matches BFM-Zero's scale-only normalization.
    Updating and applying are separate operations so one learner update uses one
    immutable scale.
    """

    def __init__(
        self,
        coefficients: tuple[float, ...] | list[float],
        decay: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize one scalar running scale for a reward vector.

        Args:
            coefficients: Fixed channel composition used to estimate scale.
            decay: Exponential moving-average decay in ``[0, 1)``.
            epsilon: Minimum variance before taking the square root.
        """
        super().__init__()
        coefficients = tuple(coefficients)
        if not coefficients:
            raise ValueError("Reward-normalizer coefficients must not be empty.")
        if not 0.0 <= decay < 1.0:
            raise ValueError("Reward-normalizer decay must be in [0, 1).")
        if epsilon <= 0.0:
            raise ValueError("Reward-normalizer epsilon must be positive.")
        self.decay = decay
        self.epsilon = epsilon
        self.register_buffer("coefficients", torch.tensor(coefficients, dtype=torch.float32))
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("mean_square", torch.zeros(1))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, rewards: torch.Tensor) -> None:
        """Update running moments from one raw reward-vector batch."""
        if rewards.ndim != 2 or rewards.shape[1] != self.coefficients.shape[0]:
            raise ValueError("rewards must have shape [batch, channel_count].")
        composed = rewards @ self.coefficients.to(dtype=rewards.dtype)
        self.mean.mul_(self.decay).add_(composed.mean(), alpha=1.0 - self.decay)
        self.mean_square.mul_(self.decay).add_(composed.square().mean(), alpha=1.0 - self.decay)
        self.count.add_(1)

    @property
    def scale(self) -> torch.Tensor:
        """Return the bias-corrected scalar standard deviation."""
        correction = 1.0 - self.decay ** self.count.clamp_min(1)
        mean = self.mean / correction
        mean_square = self.mean_square / correction
        scale = torch.sqrt(torch.clamp(mean_square - mean.square(), min=self.epsilon))
        return torch.where(self.count == 0, torch.ones_like(scale), scale)

    def forward(self, rewards: torch.Tensor) -> torch.Tensor:
        """Apply the frozen scalar scale without updating statistics."""
        if rewards.ndim != 2 or rewards.shape[1] != self.coefficients.shape[0]:
            raise ValueError("rewards must have shape [batch, channel_count].")
        return rewards / self.scale
