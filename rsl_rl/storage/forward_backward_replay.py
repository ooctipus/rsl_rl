# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Transition semantics for the forward-backward replay.

Phase 1A defines only the transition dataclasses and their masks. Phase 1D will
add one concrete GPU replay container here; this module does not wrap or alias
legacy buffer implementations.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from enum import Enum
from tensordict import TensorDict

from rsl_rl.models.forward_backward_model import ForwardBackwardObservationSchema
from rsl_rl.modules.reward_channels import ForwardBackwardRewardSchema, get_forward_backward_schema_hash


class ForwardBackwardAutoresetMode(str, Enum):
    """Environment autoreset behavior represented by a transition batch."""

    DISABLED = "disabled"
    SAME_STEP = "same_step"
    NEXT_STEP = "next_step"


@dataclass(frozen=True)
class ForwardBackwardTransitionSchema:
    """Static layout and reset semantics of one replay stream."""

    observation_schema_hash: str
    reward_schema_hash: str
    action_width: int
    context_width: int
    environment_reward_name: str
    auxiliary_evidence_names: tuple[str, ...]
    autoreset_mode: ForwardBackwardAutoresetMode
    schema_version: int = 1
    information_state: str = "feedforward"
    schema_hash: str = field(init=False)

    def __post_init__(self) -> None:
        """Check the few invariants that affect storage interpretation."""
        auxiliary_evidence_names = tuple(self.auxiliary_evidence_names)
        object.__setattr__(self, "auxiliary_evidence_names", auxiliary_evidence_names)
        if self.action_width < 1 or self.context_width < 1:
            raise ValueError("Action and context widths must be positive.")
        if len(auxiliary_evidence_names) != len(set(auxiliary_evidence_names)):
            raise ValueError("Auxiliary evidence names must be unique.")
        if self.environment_reward_name in auxiliary_evidence_names:
            raise ValueError("The environment reward must not also be auxiliary evidence.")
        if not isinstance(self.autoreset_mode, ForwardBackwardAutoresetMode):
            raise ValueError(f"Unsupported autoreset mode: {self.autoreset_mode!r}.")
        if self.schema_version != 1:
            raise ValueError(f"Unsupported transition schema version: {self.schema_version!r}.")
        if self.information_state != "feedforward":
            raise ValueError("Recurrent information state requires a new transition schema version.")

        data = {
            "action_width": self.action_width,
            "autoreset_mode": self.autoreset_mode.value,
            "auxiliary_evidence_names": self.auxiliary_evidence_names,
            "context_width": self.context_width,
            "environment_reward_name": self.environment_reward_name,
            "information_state": self.information_state,
            "observation_schema_hash": self.observation_schema_hash,
            "reward_schema_hash": self.reward_schema_hash,
            "schema_version": self.schema_version,
        }
        object.__setattr__(self, "schema_hash", get_forward_backward_schema_hash(data))

    def assert_compatible(
        self,
        observation_schema: ForwardBackwardObservationSchema,
        reward_schema: ForwardBackwardRewardSchema,
    ) -> None:
        """Check stream schemas once when constructing replay.

        This method examines only CPU metadata. It is not part of collection or
        sampling hot paths.
        """
        if self.observation_schema_hash != observation_schema.schema_hash:
            raise ValueError("Transition and observation schemas do not match.")
        if self.reward_schema_hash != reward_schema.schema_hash:
            raise ValueError("Transition and reward schemas do not match.")

        channel_by_name = {channel.name: channel for channel in reward_schema.channels}
        try:
            environment_channel = channel_by_name[self.environment_reward_name]
            evidence_channels = tuple(channel_by_name[name] for name in self.auxiliary_evidence_names)
        except KeyError as error:
            raise ValueError(f"Transition references unknown reward channel {error.args[0]!r}.") from error
        if environment_channel.source != "environment":
            raise ValueError("environment_reward_name must identify an environment reward channel.")
        if any(channel.source != "stored_evidence" for channel in evidence_channels):
            raise ValueError("Auxiliary evidence names must identify stored-evidence reward channels.")


@dataclass(frozen=True)
class ForwardBackwardTransitionBatch:
    """One fixed-row vector-environment step before replay insertion.

    ``next_observations`` is always the value returned by ``env.step``. In
    same-step autoreset mode, ``final_observations`` contains the true reached
    state on done rows and ``final_observation_valid`` identifies those rows.
    Invalid final rows are unspecified and must only be read through the masks
    below.

    ``action_applied`` is false only for reset-only rows emitted by next-step
    autoreset. Such rows seed the next action but never enter replay.
    """

    observations: TensorDict
    next_observations: TensorDict
    final_observations: TensorDict
    actions: torch.Tensor
    behavior_context: torch.Tensor
    environment_reward: torch.Tensor
    auxiliary_reward_evidence: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    context_changed: torch.Tensor
    action_applied: torch.Tensor
    final_observation_valid: torch.Tensor

    @property
    def batch_size(self) -> int:
        """Number of vector-environment rows."""
        return self.actions.shape[0]

    def done_mask(self) -> torch.Tensor:
        """Return applied rows that end an episode."""
        return self.action_applied & (self.terminated | self.truncated)

    def contract_error_mask(self, schema: ForwardBackwardTransitionSchema) -> torch.Tensor:
        """Return rows that violate final-observation or autoreset contracts.

        This method contains only device-side elementwise operations. Collection
        should OR the result into a device-resident error accumulator and reduce
        it once at a rollout or other control boundary, failing the run if any
        bit is set. It must never introduce a per-step host synchronization.

        Args:
            schema: Reset convention used by this transition stream.

        Returns:
            Boolean error mask with shape ``[batch_size, 1]``.
        """
        if schema.autoreset_mode is ForwardBackwardAutoresetMode.SAME_STEP:
            final_mismatch = self.final_observation_valid != self.done_mask()
            return final_mismatch | ~self.action_applied
        if schema.autoreset_mode is ForwardBackwardAutoresetMode.DISABLED:
            return self.final_observation_valid | ~self.action_applied
        return self.final_observation_valid

    def reached_observation_valid_mask(self, schema: ForwardBackwardTransitionSchema) -> torch.Tensor:
        """Return applied rows with a valid logical reached observation.

        Contract-invalid rows are excluded on-device rather than falling back
        to a post-reset observation before the runner reports the accumulated
        error at its next control boundary.
        """
        return self.action_applied & ~self.contract_error_mask(schema)

    def replay_mask(self, schema: ForwardBackwardTransitionSchema) -> torch.Tensor:
        """Return applied rows safe to insert into replay."""
        return self.reached_observation_valid_mask(schema)

    def bootstrap_mask(self, schema: ForwardBackwardTransitionSchema) -> torch.Tensor:
        """Return replay rows whose true successor may bootstrap.

        Truncations bootstrap from the true reached observation; terminations
        do not. Reset-only rows and malformed same-step finals never bootstrap.
        """
        return self.replay_mask(schema) & ~self.terminated

    def episode_continuation_mask(self, schema: ForwardBackwardTransitionSchema) -> torch.Tensor:
        """Return replay rows that continue the same episode."""
        return self.replay_mask(schema) & ~self.done_mask()

    def segment_continuation_mask(self, schema: ForwardBackwardTransitionSchema) -> torch.Tensor:
        """Return replay rows that continue the same context segment."""
        return self.episode_continuation_mask(schema) & ~self.context_changed

    def reached_observation_uses_final(self, schema: ForwardBackwardTransitionSchema) -> torch.Tensor:
        """Return replay rows whose reached state is in ``final_observations``."""
        if schema.autoreset_mode is ForwardBackwardAutoresetMode.SAME_STEP:
            return self.replay_mask(schema) & self.done_mask()
        return torch.zeros_like(self.action_applied)

    def bootstrap_observation_uses_final(self, schema: ForwardBackwardTransitionSchema) -> torch.Tensor:
        """Return bootstrap rows whose state is in ``final_observations``."""
        if schema.autoreset_mode is ForwardBackwardAutoresetMode.SAME_STEP:
            return self.bootstrap_mask(schema) & self.truncated
        return torch.zeros_like(self.action_applied)

    def assert_valid(
        self,
        schema: ForwardBackwardTransitionSchema,
        observation_schema: ForwardBackwardObservationSchema,
    ) -> None:
        """Run expensive tensor assertions for tests and adapter debugging.

        This method performs device reductions and therefore must not be called
        from the learner's per-step collection path.
        """
        batch_size = self.batch_size
        for name, observations in (
            ("observations", self.observations),
            ("next_observations", self.next_observations),
            ("final_observations", self.final_observations),
        ):
            if tuple(observations.batch_size) != (batch_size,):
                raise ValueError(f"{name} must have batch size ({batch_size},).")
            observation_schema.assert_valid(observations)

        float_fields = (
            ("actions", self.actions, (batch_size, schema.action_width)),
            ("behavior_context", self.behavior_context, (batch_size, schema.context_width)),
            ("environment_reward", self.environment_reward, (batch_size, 1)),
            (
                "auxiliary_reward_evidence",
                self.auxiliary_reward_evidence,
                (batch_size, len(schema.auxiliary_evidence_names)),
            ),
        )
        bool_fields = (
            ("terminated", self.terminated),
            ("truncated", self.truncated),
            ("context_changed", self.context_changed),
            ("action_applied", self.action_applied),
            ("final_observation_valid", self.final_observation_valid),
        )
        for name, value, shape in float_fields:
            if value.shape != shape or not value.is_floating_point():
                raise ValueError(f"{name} must be a floating-point tensor with shape {shape}.")
        for name, value in bool_fields:
            if value.shape != (batch_size, 1) or value.dtype is not torch.bool:
                raise ValueError(f"{name} must be a bool tensor with shape ({batch_size}, 1).")

        devices = {value.device for _name, value in self._tensors()}
        if len(devices) != 1:
            raise ValueError("All transition tensors must be on one device.")
        if any(value.requires_grad for _name, value in self._tensors()):
            raise ValueError("Transition tensors must be detached.")

        if torch.any(self.contract_error_mask(schema)):
            if schema.autoreset_mode is ForwardBackwardAutoresetMode.SAME_STEP:
                raise ValueError(
                    "Same-step autoreset requires a true final observation exactly on done rows "
                    "and cannot contain reset-only rows."
                )
            if schema.autoreset_mode is ForwardBackwardAutoresetMode.NEXT_STEP:
                raise ValueError("next_step mode must not provide separate final observations.")
            raise ValueError("disabled mode must not provide separate final observations or contain reset-only rows.")

    def _tensors(self) -> tuple[tuple[str, torch.Tensor], ...]:
        tensors = [
            (f"{group_name}.{key}", observations[key])
            for group_name, observations in (
                ("observations", self.observations),
                ("next_observations", self.next_observations),
                ("final_observations", self.final_observations),
            )
            for key in observations.keys(include_nested=False, leaves_only=True)
        ]
        tensors.extend(
            (name, value)
            for name, value in (
                ("actions", self.actions),
                ("behavior_context", self.behavior_context),
                ("environment_reward", self.environment_reward),
                ("auxiliary_reward_evidence", self.auxiliary_reward_evidence),
                ("terminated", self.terminated),
                ("truncated", self.truncated),
                ("context_changed", self.context_changed),
                ("action_applied", self.action_applied),
                ("final_observation_valid", self.final_observation_valid),
            )
        )
        return tuple(tensors)
