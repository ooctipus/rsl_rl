# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Static reward and value-channel contracts for forward-backward learning."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

ForwardBackwardRewardTiming = Literal["state", "next_state", "transition"]
ForwardBackwardRewardSource = Literal["environment", "stored_evidence", "recomputed"]
ForwardBackwardValueKind = Literal["forward_readout", "critic"]


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
        object.__setattr__(self, "reward_channels", reward_channels)

    def validate_reward_schema(self, reward_schema: ForwardBackwardRewardSchema) -> None:
        """Reject references to channels outside the reward schema."""
        unknown = set(self.reward_channels).difference(reward_schema.channel_names)
        if unknown:
            raise ValueError(f"Value source uses unknown reward channels: {tuple(sorted(unknown))}.")
