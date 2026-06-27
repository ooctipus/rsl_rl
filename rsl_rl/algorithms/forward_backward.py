# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL lifecycle and checkpoint boundary for forward-backward learning."""

from __future__ import annotations

import torch
from collections.abc import Mapping
from dataclasses import dataclass
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models.forward_backward_model import ForwardBackwardModel
from rsl_rl.modules.reward_channels import get_forward_backward_schema_hash

FORWARD_BACKWARD_CHECKPOINT_FORMAT = "rsl_rl.forward_backward"
FORWARD_BACKWARD_CHECKPOINT_VERSION = 1
FORWARD_BACKWARD_CHECKPOINT_HEADER = "forward_backward_header"
_CHECKPOINT_MANIFEST_KEYS = frozenset({
    "config",
    "expert_schema_hash",
    "observation_schema_hash",
    "reward_schema_hash",
    "transition_schema_hash",
    "value_specs",
})


@dataclass(frozen=True)
class ForwardBackwardCheckpointHeader:
    """Small compatibility header stored with a forward-backward checkpoint.

    The schema hash should cover the configuration and schemas that affect
    state interpretation. Concrete learners decide which state dictionaries
    they save; the header deliberately does not prescribe a nested manifest.
    """

    schema_hash: str
    format_name: str = FORWARD_BACKWARD_CHECKPOINT_FORMAT
    format_version: int = FORWARD_BACKWARD_CHECKPOINT_VERSION

    def __post_init__(self) -> None:
        """Reject checkpoint identities that this implementation cannot load."""
        if self.format_name != FORWARD_BACKWARD_CHECKPOINT_FORMAT:
            raise ValueError(f"Unsupported checkpoint format: {self.format_name!r}.")
        if not isinstance(self.format_version, int) or isinstance(self.format_version, bool):
            raise TypeError("format_version must be an integer.")
        if self.format_version != FORWARD_BACKWARD_CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported checkpoint format version: {self.format_version!r}.")
        if not isinstance(self.schema_hash, str):
            raise TypeError("schema_hash must be a string.")
        if len(self.schema_hash) != 64 or any(character not in "0123456789abcdef" for character in self.schema_hash):
            raise ValueError("schema_hash must be a lowercase SHA-256 digest.")

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, object]) -> ForwardBackwardCheckpointHeader:
        """Build a header from the complete compatibility manifest.

        Args:
            manifest: Resolved configuration, schema hashes, and value specifications.

        Returns:
            Header containing one stable aggregate fingerprint.
        """
        keys = set(manifest)
        missing = _CHECKPOINT_MANIFEST_KEYS.difference(keys)
        unknown = keys.difference(_CHECKPOINT_MANIFEST_KEYS)
        if missing or unknown:
            raise ValueError(
                "Checkpoint manifest keys do not match the contract; "
                f"missing={tuple(sorted(missing))}, unknown={tuple(sorted(unknown))}."
            )
        data = {key: manifest[key] for key in sorted(_CHECKPOINT_MANIFEST_KEYS)}
        return cls(schema_hash=get_forward_backward_schema_hash(data))

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ForwardBackwardCheckpointHeader:
        """Parse a serialized checkpoint header."""
        try:
            schema_hash = data["schema_hash"]
            format_name = data["format_name"]
            format_version = data["format_version"]
        except KeyError as error:
            raise ValueError(f"Checkpoint header is missing {error.args[0]!r}.") from error
        if not isinstance(schema_hash, str):
            raise TypeError("checkpoint schema_hash must be a string.")
        if not isinstance(format_name, str):
            raise TypeError("checkpoint format_name must be a string.")
        if not isinstance(format_version, int) or isinstance(format_version, bool):
            raise TypeError("checkpoint format_version must be an integer.")
        return cls(schema_hash=schema_hash, format_name=format_name, format_version=format_version)

    def to_dict(self) -> dict[str, object]:
        """Return the plain mapping stored under the checkpoint header key."""
        return {
            "format_name": self.format_name,
            "format_version": self.format_version,
            "schema_hash": self.schema_hash,
        }

    def validate_checkpoint(self, checkpoint: Mapping[str, object]) -> None:
        """Check checkpoint format/version and compatibility fingerprint."""
        try:
            data = checkpoint[FORWARD_BACKWARD_CHECKPOINT_HEADER]
        except KeyError as error:
            raise ValueError(f"Checkpoint is missing {FORWARD_BACKWARD_CHECKPOINT_HEADER!r}.") from error
        if not isinstance(data, Mapping):
            raise TypeError("forward-backward checkpoint header must be a mapping.")
        loaded = self.from_dict(data)
        if loaded.schema_hash != self.schema_hash:
            raise ValueError("Checkpoint schema is incompatible with the current learner.")


class ForwardBackward:
    """Forward-backward algorithm shell using the ordinary RSL-RL protocol.

    Phase 1A defines only the runner-facing method signatures. Later phases
    add concrete models, storage, equations, and update behavior directly to
    this class. Constructor fields remain explicit so misspelled algorithm
    configuration cannot disappear into ``**kwargs``.
    """

    model: ForwardBackwardModel

    def __init__(self, *, multi_gpu_cfg: dict | None = None) -> None:
        """Create the Phase 1A shell and reject unsupported distributed training."""
        if multi_gpu_cfg is not None:
            raise NotImplementedError("Forward-backward multi-GPU synchronization is not implemented.")

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> ForwardBackward:
        """Construct the learner through the standard RSL-RL factory signature.

        Concrete construction will copy the component sections it mutates,
        resolve their ``class_name`` entries, and pass the remaining fields to
        explicit constructors. Runner-owned fields may remain in ``cfg``.
        """
        raise NotImplementedError("Forward-backward construction is implemented in Phase 1E.")

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and record behavior data for the pending transition."""
        raise NotImplementedError

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Consume one vector-environment step using the standard runner call."""
        raise NotImplementedError

    def compute_returns(self, obs: TensorDict) -> None:
        """Prepare return-like state required by the configured actor update."""
        raise NotImplementedError

    def update(self) -> dict[str, float]:
        """Run one learner update and return scalar diagnostics."""
        raise NotImplementedError

    def train_mode(self) -> None:
        """Put learnable modules in training mode."""
        raise NotImplementedError

    def eval_mode(self) -> None:
        """Put learnable modules in evaluation mode."""
        raise NotImplementedError

    def save(self) -> dict[str, object]:
        """Return model and optimizer state for an RSL-RL runner checkpoint."""
        raise NotImplementedError

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Restore selected state and report whether the runner iteration should load."""
        raise NotImplementedError

    def get_policy(self) -> ForwardBackwardModel:
        """Return the uncompiled policy model used for inference and export."""
        raise NotImplementedError

    def compile(self, mode: str | None = None) -> None:
        """Compile eligible models using the requested ``torch.compile`` mode."""
        raise NotImplementedError
