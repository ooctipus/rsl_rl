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

_EDGE_TERMINATED = 1 << 0
_EDGE_TRUNCATED = 1 << 1
_EDGE_CONTEXT_CHANGED = 1 << 2
_EDGE_FINAL_OBSERVATION_VALID = 1 << 3
_EDGE_APPLIED = 1 << 4


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
    same-step autoreset mode, ``final_observations`` may contain the true reached
    state on done rows and ``final_observation_valid`` identifies those rows.
    When a final observation is unavailable, replay explicitly falls back to the
    pre-step observation; it never substitutes the returned reset observation.

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
            stale_final = self.final_observation_valid & ~self.done_mask()
            return stale_final | ~self.action_applied
        if schema.autoreset_mode is ForwardBackwardAutoresetMode.DISABLED:
            return self.final_observation_valid | ~self.action_applied
        return self.final_observation_valid

    def replay_mask(self, schema: ForwardBackwardTransitionSchema) -> torch.Tensor:
        """Return applied rows safe to insert into replay."""
        return self.action_applied & ~self.contract_error_mask(schema)

    def bootstrap_mask(self, schema: ForwardBackwardTransitionSchema) -> torch.Tensor:
        """Return replay rows whose normalized successor may bootstrap."""
        return self.replay_mask(schema) & ~self.terminated

    def episode_continuation_mask(self, schema: ForwardBackwardTransitionSchema) -> torch.Tensor:
        """Return replay rows that continue the same episode."""
        return self.replay_mask(schema) & ~self.done_mask()

    def segment_continuation_mask(self, schema: ForwardBackwardTransitionSchema) -> torch.Tensor:
        """Return replay rows that continue the same context segment."""
        return self.episode_continuation_mask(schema) & ~self.context_changed

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
                    "Same-step autoreset only permits final observations on done rows and cannot contain "
                    "reset-only rows."
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


@dataclass(frozen=True, slots=True)
class ForwardBackwardHistoryLayout:
    """Versioned reconstruction of derived last-action and history fields.

    ``include_seed_observations`` controls whether observations introduced
    without an applied transition, at stream initialization or an external
    reset, become future observation-history sources. Reached observations
    produced by applied transitions are always history sources.
    """

    @dataclass(frozen=True, slots=True)
    class Source:
        """One field-major history source.

        ``observation_name=None`` selects applied actions. Otherwise ``start``
        and ``stop`` select an emitted observation slice.
        """

        observation_name: str | None
        start: int
        stop: int

        def __post_init__(self) -> None:
            """Reject empty source slices."""
            if self.start < 0 or self.stop <= self.start:
                raise ValueError("History source slices must be non-empty and non-negative.")

    history_field: str
    history_length: int
    sources: tuple[Source, ...]
    last_action_field: str | None = None
    include_seed_observations: bool = True
    version: int = 1
    schema_hash: str = field(init=False)

    def __post_init__(self) -> None:
        """Freeze and identify one reconstruction convention."""
        sources = tuple(self.sources)
        object.__setattr__(self, "sources", sources)
        if not self.history_field:
            raise ValueError("history_field must not be empty.")
        if self.history_length < 1:
            raise ValueError("history_length must be positive.")
        if not sources:
            raise ValueError("At least one history source is required.")
        if self.last_action_field == self.history_field:
            raise ValueError("last_action_field and history_field must differ.")
        if self.version != 1:
            raise ValueError(f"Unsupported history layout version: {self.version!r}.")
        object.__setattr__(
            self,
            "schema_hash",
            get_forward_backward_schema_hash({
                "history_field": self.history_field,
                "history_length": self.history_length,
                "include_seed_observations": self.include_seed_observations,
                "last_action_field": self.last_action_field,
                "sources": tuple((source.observation_name, source.start, source.stop) for source in sources),
                "version": self.version,
            }),
        )


@dataclass(frozen=True, slots=True)
class ForwardBackwardReplayBatch:
    """Logical transitions sampled from dense or compact replay storage."""

    observations: TensorDict
    next_observations: TensorDict
    actions: torch.Tensor
    behavior_context: torch.Tensor
    environment_reward: torch.Tensor
    auxiliary_reward_evidence: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    context_changed: torch.Tensor
    successor_uses_current: torch.Tensor
    valid: torch.Tensor

    def bootstrap_mask(self) -> torch.Tensor:
        """Return sampled rows whose reached state may bootstrap."""
        return self.valid & ~self.terminated

    def episode_continuation_mask(self) -> torch.Tensor:
        """Return sampled rows that continue the same episode."""
        return self.valid & ~self.terminated & ~self.truncated

    def segment_continuation_mask(self) -> torch.Tensor:
        """Return sampled rows that continue the same context segment."""
        return self.episode_continuation_mask() & ~self.context_changed


class ForwardBackwardReplay:
    """GPU-first time-major replay with adjacent nodes and sparse true finals.

    The edge ring has ``capacity_steps`` vector rows. Emitted observation nodes
    retain one successor guard row plus the optional history horizon. Derived
    last-action and actor-history fields are reconstructed from emitted nodes
    and applied actions when a :class:`ForwardBackwardHistoryLayout` is given.

    Calls to :meth:`add` must form one contiguous vector-environment stream:
    except for explicit-reset mode, one call's ``next_observations`` are the
    next call's ``observations``. Replay stores that shared node only once.
    """

    class Sampling(str, Enum):
        """Probability law used to choose one retained logical transition."""

        TRANSITION_UNIFORM = "transition_uniform"
        EPISODE_UNIFORM = "episode_uniform"

    def __init__(
        self,
        capacity_steps: int,
        num_envs: int,
        terminal_capacity_per_env: int,
        observation_schema: ForwardBackwardObservationSchema,
        transition_schema: ForwardBackwardTransitionSchema,
        reward_schema: ForwardBackwardRewardSchema,
        device: str | torch.device,
        *,
        history_layout: ForwardBackwardHistoryLayout | None = None,
        sampling: Sampling | str = Sampling.TRANSITION_UNIFORM,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Allocate replay tensors and fixed sampling state."""
        if capacity_steps < 1 or num_envs < 1 or terminal_capacity_per_env < 1:
            raise ValueError("Replay, environment, and terminal capacities must be positive.")
        transition_schema.assert_compatible(observation_schema, reward_schema)
        self.capacity_steps = capacity_steps
        self.num_envs = num_envs
        self.terminal_capacity_per_env = terminal_capacity_per_env
        self.observation_schema = observation_schema
        self.transition_schema = transition_schema
        self.reward_schema = reward_schema
        self.history_layout = history_layout
        self.sampling = self.Sampling(sampling)
        if (
            self.sampling is self.Sampling.EPISODE_UNIFORM
            and transition_schema.autoreset_mode is ForwardBackwardAutoresetMode.NEXT_STEP
        ):
            raise ValueError("Episode-uniform replay does not support reset-only next-step rows.")
        self.device = torch.device(device)
        self.dtype = dtype
        self._field_widths = dict(observation_schema.field_widths)
        self._validate_history_layout()

        history_length = history_layout.history_length if history_layout is not None else 0
        self.node_capacity_steps = capacity_steps + history_length + 1
        self.action_capacity_steps = capacity_steps + history_length
        derived_fields = set()
        if history_layout is not None:
            derived_fields.add(history_layout.history_field)
            if history_layout.last_action_field is not None:
                derived_fields.add(history_layout.last_action_field)
        self._stored_fields = tuple(
            name for name, _width in observation_schema.field_widths if name not in derived_fields
        )

        self.nodes = _allocate_observations(
            observation_schema,
            (self.node_capacity_steps, num_envs),
            self.device,
            dtype,
            self._stored_fields,
        )
        self.device = next(iter(self.nodes.values())).device
        self.node_episode_ids = torch.full(
            (self.node_capacity_steps, num_envs), -1, device=self.device, dtype=torch.long
        )
        self.node_episode_steps = torch.full_like(self.node_episode_ids, -1)
        self.node_history_valid = (
            torch.zeros_like(self.node_episode_ids, dtype=torch.bool)
            if history_layout is not None and not history_layout.include_seed_observations
            else None
        )

        edge_shape = (capacity_steps, num_envs)
        self.edge_flags = torch.zeros(edge_shape, device=self.device, dtype=torch.uint8)
        self.behavior_context = torch.zeros(
            *edge_shape, transition_schema.context_width, device=self.device, dtype=dtype
        )
        self.environment_reward = torch.zeros(*edge_shape, 1, device=self.device, dtype=dtype)
        self.auxiliary_reward_evidence = torch.zeros(
            *edge_shape, len(transition_schema.auxiliary_evidence_names), device=self.device, dtype=dtype
        )

        action_shape = (self.action_capacity_steps, num_envs)
        self.actions = torch.zeros(*action_shape, transition_schema.action_width, device=self.device, dtype=dtype)
        terminal_capacity = (terminal_capacity_per_env + 1) * num_envs
        self.terminals = _allocate_observations(
            observation_schema,
            (terminal_capacity,),
            self.device,
            dtype,
            self._stored_fields,
        )
        self.terminal_owner_steps = torch.full((terminal_capacity,), -1, device=self.device, dtype=torch.long)
        self._terminal_bases = torch.arange(num_envs, device=self.device, dtype=torch.long) * (
            terminal_capacity_per_env + 1
        )
        self._terminal_scratch = self._terminal_bases + terminal_capacity_per_env

        self.episode_ids = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.episode_steps = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.contract_errors = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.terminal_overflow = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._total_steps = 0
        self._size_steps = 0
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self._sample_valid_counts: torch.Tensor | None = None
        self._sample_valid_cdf: torch.Tensor | None = None
        self._sample_generator: torch.Generator | None = None
        if transition_schema.autoreset_mode is ForwardBackwardAutoresetMode.NEXT_STEP:
            self._sample_valid_counts = torch.zeros(capacity_steps, device=self.device, dtype=torch.int32)
            self._sample_valid_cdf = torch.empty(capacity_steps, device=self.device, dtype=torch.int32)
            self._sample_generator = torch.Generator(device=self.device)
            self._sample_generator.manual_seed(seed + 1)
        if self.sampling is self.Sampling.EPISODE_UNIFORM:
            slots = torch.arange(terminal_capacity, device=self.device, dtype=torch.long)
            self._episode_slot_env_ids = torch.div(
                slots,
                terminal_capacity_per_env + 1,
                rounding_mode="floor",
            )
            self._episode_slot_is_terminal = torch.remainder(slots, terminal_capacity_per_env + 1) < (
                terminal_capacity_per_env
            )
            self._episode_segment_starts = torch.empty_like(slots)
            self._episode_segment_lengths = torch.empty_like(slots)
            self._episode_segment_valid = torch.empty(terminal_capacity, device=self.device, dtype=torch.bool)
            self._episode_segment_cdf = torch.empty(terminal_capacity, device=self.device, dtype=torch.int32)
            self._sample_indices = self._sample_episode_uniform_indices
        else:
            self._sample_indices = self._sample_transition_uniform_indices

    @property
    def total_steps(self) -> int:
        """Number of vector steps observed since construction."""
        return self._total_steps

    @property
    def size_steps(self) -> int:
        """Number of retained vector-step rows."""
        return self._size_steps

    @property
    def num_transitions(self) -> int:
        """Maximum number of retained applied and reset-only edge positions."""
        return self._size_steps * self.num_envs

    def add(self, transition: ForwardBackwardTransitionBatch) -> None:
        """Insert the next contiguous vector step without host synchronization."""
        step = self._total_steps
        edge_row = step % self.capacity_steps
        node_row = step % self.node_capacity_steps
        next_node_row = (step + 1) % self.node_capacity_steps
        action_row = step % self.action_capacity_steps

        contract_error = transition.contract_error_mask(self.transition_schema).squeeze(-1)
        replay = transition.action_applied.squeeze(-1) & ~contract_error
        done = replay & (transition.terminated.squeeze(-1) | transition.truncated.squeeze(-1))
        replay_int = replay.long()
        done_int = done.long()
        self.contract_errors.logical_or_(contract_error)

        write_current = step == 0 or self.transition_schema.autoreset_mode is ForwardBackwardAutoresetMode.DISABLED
        if write_current:
            for name in self._stored_fields:
                self.nodes[name][node_row].copy_(transition.observations[name])
            self.node_episode_ids[node_row].copy_(self.episode_ids)
            self.node_episode_steps[node_row].copy_(self.episode_steps)

        for name in self._stored_fields:
            self.nodes[name][next_node_row].copy_(transition.next_observations[name])
        next_episode_ids = self.episode_ids
        next_episode_steps = self.episode_steps + replay_int
        if self.transition_schema.autoreset_mode is ForwardBackwardAutoresetMode.SAME_STEP:
            next_episode_ids = self.episode_ids + done_int
            next_episode_steps = torch.where(done, 0, next_episode_steps)
        self.node_episode_ids[next_node_row].copy_(next_episode_ids)
        self.node_episode_steps[next_node_row].copy_(next_episode_steps)
        if self.node_history_valid is not None:
            self.node_history_valid[next_node_row].copy_(replay)

        self.actions[action_row].copy_(transition.actions)
        edge_flags = transition.terminated.squeeze(-1).to(torch.uint8) * _EDGE_TERMINATED
        edge_flags.bitwise_or_(transition.truncated.squeeze(-1).to(torch.uint8) * _EDGE_TRUNCATED)
        edge_flags.bitwise_or_(transition.context_changed.squeeze(-1).to(torch.uint8) * _EDGE_CONTEXT_CHANGED)
        edge_flags.bitwise_or_(
            transition.final_observation_valid.squeeze(-1).to(torch.uint8) * _EDGE_FINAL_OBSERVATION_VALID
        )
        edge_flags.bitwise_or_(replay.to(torch.uint8) * _EDGE_APPLIED)
        self.edge_flags[edge_row].copy_(edge_flags)
        if self._sample_valid_counts is not None:
            self._sample_valid_counts[edge_row].copy_(replay_int.sum(dtype=torch.int32))
        self.behavior_context[edge_row].copy_(transition.behavior_context)
        self.environment_reward[edge_row].copy_(transition.environment_reward)
        self.auxiliary_reward_evidence[edge_row].copy_(transition.auxiliary_reward_evidence)

        terminal_slots = self._terminal_bases + torch.remainder(self.episode_ids, self.terminal_capacity_per_env)
        previous_owner_steps = self.terminal_owner_steps[terminal_slots]
        previous_owner_is_live = (previous_owner_steps >= 0) & (previous_owner_steps > step - self.capacity_steps)
        self.terminal_overflow.logical_or_(done & previous_owner_is_live)
        terminal_slots = torch.where(done, terminal_slots, self._terminal_scratch)
        for name in self._stored_fields:
            if self.transition_schema.autoreset_mode is ForwardBackwardAutoresetMode.SAME_STEP:
                terminal_source = torch.where(
                    transition.final_observation_valid,
                    transition.final_observations[name],
                    transition.observations[name],
                )
            else:
                terminal_source = transition.next_observations[name]
            self.terminals[name].index_copy_(0, terminal_slots, terminal_source)
        self.terminal_owner_steps.index_fill_(0, terminal_slots, step)

        self.episode_ids.add_(done_int)
        self.episode_steps.copy_(torch.where(done, 0, torch.where(replay, self.episode_steps + 1, self.episode_steps)))
        self._total_steps += 1
        self._size_steps = min(self._size_steps + 1, self.capacity_steps)

    def process_env_reset(self, observations: TensorDict, reset: torch.Tensor) -> None:
        """Close the latest reached states and seed an externally reset stream.

        This boundary is for resets performed between calls to ``env.step``,
        such as an algorithm-controlled reset after an evaluation. Applied
        latest rows become truncations whose exact final states are the reached
        nodes already in replay. The supplied post-reset observations then
        replace those nodes as the starts of new episodes.
        """
        if self.transition_schema.autoreset_mode is ForwardBackwardAutoresetMode.NEXT_STEP:
            raise ValueError("Next-step autoreset represents external resets with reset-only rows.")
        if tuple(observations.batch_size) != (self.num_envs,):
            raise ValueError(f"Reset observations must have batch size ({self.num_envs},).")
        self.observation_schema.assert_valid(observations)
        if reset.shape not in ((self.num_envs,), (self.num_envs, 1)) or reset.dtype is not torch.bool:
            raise ValueError(f"reset must be bool with shape ({self.num_envs},) or ({self.num_envs}, 1).")
        if reset.device != self.device or any(observations[name].device != self.device for name in self._stored_fields):
            raise ValueError("Reset observations and mask must be on the replay device.")
        if self._total_steps == 0:
            return

        reset = reset.reshape(self.num_envs)
        step = self._total_steps - 1
        edge_row = step % self.capacity_steps
        node_row = step % self.node_capacity_steps
        reached_node_row = (step + 1) % self.node_capacity_steps
        edge_flags = self.edge_flags[edge_row]
        applied = (edge_flags & _EDGE_APPLIED) != 0
        already_done = (edge_flags & (_EDGE_TERMINATED | _EDGE_TRUNCATED)) != 0
        converted = reset & applied & ~already_done
        self.contract_errors.logical_or_(reset & ~applied)

        edge_episode_ids = self.node_episode_ids[node_row]
        terminal_slots = self._terminal_bases + torch.remainder(
            edge_episode_ids,
            self.terminal_capacity_per_env,
        )
        previous_owner_steps = self.terminal_owner_steps[terminal_slots]
        previous_owner_is_live = (previous_owner_steps >= 0) & (previous_owner_steps > step - self.capacity_steps)
        self.terminal_overflow.logical_or_(converted & previous_owner_is_live)
        terminal_slots = torch.where(converted, terminal_slots, self._terminal_scratch)
        for name in self._stored_fields:
            self.terminals[name].index_copy_(0, terminal_slots, self.nodes[name][reached_node_row])
        self.terminal_owner_steps.index_fill_(0, terminal_slots, step)

        edge_flags.bitwise_or_(converted.to(torch.uint8) * _EDGE_TRUNCATED)
        edge_flags.bitwise_or_(converted.to(torch.uint8) * _EDGE_FINAL_OBSERVATION_VALID)
        self.episode_ids.add_(converted.long())
        self.episode_steps.copy_(torch.where(reset, 0, self.episode_steps))
        reset_rows = reset.unsqueeze(-1)
        for name in self._stored_fields:
            reached = self.nodes[name][reached_node_row]
            reached.copy_(torch.where(reset_rows, observations[name], reached))
        self.node_episode_ids[reached_node_row].copy_(
            torch.where(reset, self.episode_ids, self.node_episode_ids[reached_node_row])
        )
        self.node_episode_steps[reached_node_row].copy_(
            torch.where(reset, self.episode_steps, self.node_episode_steps[reached_node_row])
        )
        if self.node_history_valid is not None:
            self.node_history_valid[reached_node_row].logical_and_(~reset)

    def sample(self, step_ids: torch.Tensor, env_ids: torch.Tensor) -> ForwardBackwardReplayBatch:
        """Resolve logical steps through edge, node, and terminal generations."""
        edge_rows = torch.remainder(step_ids, self.capacity_steps)
        node_rows = torch.remainder(step_ids, self.node_capacity_steps)
        next_node_rows = torch.remainder(step_ids + 1, self.node_capacity_steps)
        action_rows = torch.remainder(step_ids, self.action_capacity_steps)
        episode_ids = self.node_episode_ids[node_rows, env_ids]
        episode_steps = self.node_episode_steps[node_rows, env_ids]
        edge_flags = self.edge_flags[edge_rows, env_ids]
        oldest_step = self._total_steps - self._size_steps
        valid = (step_ids >= oldest_step) & (step_ids < self._total_steps) & ((edge_flags & _EDGE_APPLIED) != 0)
        terminated = ((edge_flags & _EDGE_TERMINATED) != 0).unsqueeze(-1)
        truncated = ((edge_flags & _EDGE_TRUNCATED) != 0).unsqueeze(-1)
        done = terminated | truncated
        final_observation_valid = ((edge_flags & _EDGE_FINAL_OBSERVATION_VALID) != 0).unsqueeze(-1)
        successor_uses_current = (
            done & ~final_observation_valid
            if self.transition_schema.autoreset_mode is ForwardBackwardAutoresetMode.SAME_STEP
            else torch.zeros_like(done)
        )
        terminal_slots = self._terminal_bases[env_ids] + torch.remainder(episode_ids, self.terminal_capacity_per_env)
        terminal_valid = self.terminal_owner_steps[terminal_slots] == step_ids
        next_node_valid = (self.node_episode_ids[next_node_rows, env_ids] == episode_ids) & (
            self.node_episode_steps[next_node_rows, env_ids] == episode_steps + 1
        )
        successor_valid = torch.where(done.squeeze(-1), terminal_valid, next_node_valid)
        valid &= successor_valid

        current_base = {name: self.nodes[name][node_rows, env_ids] for name in self._stored_fields}
        next_base = {
            name: torch.where(
                done,
                self.terminals[name][terminal_slots],
                self.nodes[name][next_node_rows, env_ids],
            )
            for name in self._stored_fields
        }
        current_derived, current_history_valid = self._reconstruct_derived(
            step_ids, env_ids, episode_ids, episode_steps
        )
        next_derived, next_history_valid = self._reconstruct_derived(
            step_ids + 1, env_ids, episode_ids, episode_steps + 1
        )
        for name, values in next_derived.items():
            next_derived[name] = torch.where(successor_uses_current, current_derived[name], values)
        next_history_valid = torch.where(successor_uses_current, current_history_valid, next_history_valid)
        valid = valid.unsqueeze(-1) & current_history_valid & next_history_valid
        observations = self._make_observations(current_base, current_derived, step_ids.shape[0])
        next_observations = self._make_observations(next_base, next_derived, step_ids.shape[0])
        return ForwardBackwardReplayBatch(
            observations=observations,
            next_observations=next_observations,
            actions=self.actions[action_rows, env_ids],
            behavior_context=self.behavior_context[edge_rows, env_ids],
            environment_reward=self.environment_reward[edge_rows, env_ids],
            auxiliary_reward_evidence=self.auxiliary_reward_evidence[edge_rows, env_ids],
            terminated=terminated,
            truncated=truncated,
            context_changed=((edge_flags & _EDGE_CONTEXT_CHANGED) != 0).unsqueeze(-1),
            successor_uses_current=successor_uses_current,
            valid=valid,
        )

    def sample_random(self, batch_size: int) -> ForwardBackwardReplayBatch:
        """Sample logical transitions under the configured replay probability law."""
        if self._size_steps == 0:
            raise RuntimeError("Cannot sample an empty replay.")
        step_ids, env_ids = self._sample_indices(batch_size)
        return self.sample(step_ids, env_ids)

    def _sample_transition_uniform_indices(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose retained applied positions uniformly."""
        oldest = self._total_steps - self._size_steps
        step_ids = torch.randint(oldest, self._total_steps, (batch_size,), device=self.device, generator=self.generator)
        env_ids = torch.randint(self.num_envs, (batch_size,), device=self.device, generator=self.generator)
        if self._sample_valid_counts is not None:
            edge_rows = torch.remainder(step_ids, self.capacity_steps)
            selected_valid = (self.edge_flags[edge_rows, env_ids] & _EDGE_APPLIED) != 0
            valid_cdf = self._sample_valid_cdf[: self._size_steps]
            torch.cumsum(
                self._sample_valid_counts[: self._size_steps],
                dim=0,
                dtype=torch.int32,
                out=valid_cdf,
            )
            valid_positions = (
                torch.rand(batch_size, device=self.device, generator=self._sample_generator) * valid_cdf[-1]
            ).to(torch.int32)
            valid_rows = torch.searchsorted(valid_cdf, valid_positions, right=True)
            valid_rows.clamp_max_(self._size_steps - 1)
            previous_rows = torch.clamp(valid_rows - 1, min=0)
            previous_counts = torch.where(
                valid_rows > 0,
                valid_cdf[previous_rows],
                torch.zeros_like(valid_positions),
            )
            valid_ranks = valid_positions - previous_counts
            valid_flags = (self.edge_flags[valid_rows] & _EDGE_APPLIED) != 0
            valid_prefix = torch.cumsum(valid_flags, dim=1, dtype=torch.int32)
            valid_env_ids = torch.argmax((valid_prefix > valid_ranks.unsqueeze(-1)).to(torch.uint8), dim=1)
            if self._size_steps < self.capacity_steps:
                valid_step_ids = valid_rows
            else:
                newest_row = (self._total_steps - 1) % self.capacity_steps
                ages = torch.remainder(newest_row - valid_rows, self.capacity_steps)
                valid_step_ids = self._total_steps - 1 - ages
            step_ids = torch.where(selected_valid, step_ids, valid_step_ids)
            env_ids = torch.where(selected_valid, env_ids, valid_env_ids)
        return step_ids, env_ids

    def _sample_episode_uniform_indices(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Choose a retained episode segment, then one transition in that segment."""
        oldest = self._total_steps - self._size_steps
        owners = self.terminal_owner_steps
        owner_rows = torch.remainder(owners, self.node_capacity_steps)
        owner_episode_steps = self.node_episode_steps[owner_rows, self._episode_slot_env_ids]
        starts = self._episode_segment_starts
        starts.copy_(owners - owner_episode_steps)
        starts.clamp_min_(oldest)
        lengths = self._episode_segment_lengths
        lengths.copy_(owners).sub_(starts).add_(1)
        valid = self._episode_segment_valid
        valid.copy_(self._episode_slot_is_terminal)
        valid.logical_and_(owners >= oldest)
        valid.logical_and_(owners < self._total_steps)
        valid.logical_and_(owner_episode_steps >= 0)

        current_starts = torch.clamp(self._total_steps - self.episode_steps, min=oldest)
        starts.index_copy_(0, self._terminal_scratch, current_starts)
        lengths.index_copy_(0, self._terminal_scratch, self._total_steps - current_starts)
        valid.index_copy_(0, self._terminal_scratch, self.episode_steps > 0)
        torch.cumsum(valid, dim=0, dtype=torch.int32, out=self._episode_segment_cdf)

        segment_ranks = (
            torch.rand(batch_size, device=self.device, generator=self.generator) * self._episode_segment_cdf[-1]
        ).to(torch.int32)
        segment_slots = torch.searchsorted(self._episode_segment_cdf, segment_ranks, right=True)
        selected_lengths = lengths[segment_slots]
        relative_steps = (torch.rand(batch_size, device=self.device, generator=self.generator) * selected_lengths).to(
            torch.long
        )
        return starts[segment_slots] + relative_steps, self._episode_slot_env_ids[segment_slots]

    def assert_no_errors(self) -> None:
        """Reduce deferred collection and terminal errors at a control boundary."""
        if torch.any(self.contract_errors):
            raise RuntimeError("Replay observed an invalid autoreset/final-observation contract.")
        if torch.any(self.terminal_overflow):
            raise RuntimeError("Sparse terminal capacity was reused while its owning edge was still live.")

    def storage_bytes(self) -> int:
        """Return allocated tensor payload bytes, excluding allocator overhead."""
        return sum(tensor.numel() * tensor.element_size() for tensor in self._tensors())

    def state_dict(self) -> dict[str, object]:
        """Capture exact replay, sparse-terminal, and sampling state."""
        state: dict[str, object] = {
            "transition_schema_hash": self.transition_schema.schema_hash,
            "history_schema_hash": self.history_layout.schema_hash if self.history_layout is not None else None,
            "capacity_steps": self.capacity_steps,
            "num_envs": self.num_envs,
            "terminal_capacity_per_env": self.terminal_capacity_per_env,
            "sampling": self.sampling.value,
            "dtype": str(self.dtype),
            "total_steps": self._total_steps,
            "size_steps": self._size_steps,
            "nodes": self.nodes,
            "node_episode_ids": self.node_episode_ids,
            "node_episode_steps": self.node_episode_steps,
            "edge_flags": self.edge_flags,
            "behavior_context": self.behavior_context,
            "environment_reward": self.environment_reward,
            "auxiliary_reward_evidence": self.auxiliary_reward_evidence,
            "actions": self.actions,
            "terminals": self.terminals,
            "terminal_owner_steps": self.terminal_owner_steps,
            "episode_ids": self.episode_ids,
            "episode_steps": self.episode_steps,
            "contract_errors": self.contract_errors,
            "terminal_overflow": self.terminal_overflow,
            "generator_state": self.generator.get_state(),
        }
        if self._sample_generator is not None:
            state["sample_generator_state"] = self._sample_generator.get_state()
        if self.node_history_valid is not None:
            state["node_history_valid"] = self.node_history_valid
        return state

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore an exact replay state with strict static compatibility."""
        expected = (
            self.transition_schema.schema_hash,
            self.history_layout.schema_hash if self.history_layout is not None else None,
            self.capacity_steps,
            self.num_envs,
            self.terminal_capacity_per_env,
            self.sampling.value,
            str(self.dtype),
        )
        actual = (
            state["transition_schema_hash"],
            state["history_schema_hash"],
            state["capacity_steps"],
            state["num_envs"],
            state["terminal_capacity_per_env"],
            state.get("sampling", self.Sampling.TRANSITION_UNIFORM.value),
            state["dtype"],
        )
        if actual != expected:
            raise ValueError("Replay state is incompatible with the configured schemas or capacities.")
        self._total_steps = int(state["total_steps"])
        self._size_steps = int(state["size_steps"])
        _copy_tensor_dict(self.nodes, state["nodes"])
        _copy_tensor_dict(self.terminals, state["terminals"])
        if self.node_history_valid is not None:
            value = state["node_history_valid"]
            if not isinstance(value, torch.Tensor):
                raise TypeError("Replay state 'node_history_valid' must be a tensor.")
            self.node_history_valid.copy_(value)
        for name in (
            "node_episode_ids",
            "node_episode_steps",
            "edge_flags",
            "behavior_context",
            "environment_reward",
            "auxiliary_reward_evidence",
            "actions",
            "terminal_owner_steps",
            "episode_ids",
            "episode_steps",
            "contract_errors",
            "terminal_overflow",
        ):
            value = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Replay state {name!r} must be a tensor.")
            getattr(self, name).copy_(value)
        generator_state = state["generator_state"]
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("Replay generator_state must be a tensor.")
        self.generator.set_state(generator_state)
        if self._sample_generator is not None:
            sample_generator_state = state["sample_generator_state"]
            if not isinstance(sample_generator_state, torch.Tensor):
                raise TypeError("Replay sample_generator_state must be a tensor.")
            self._sample_generator.set_state(sample_generator_state)
            self._sample_valid_counts.copy_(((self.edge_flags & _EDGE_APPLIED) != 0).sum(dim=1, dtype=torch.int32))

    def _validate_history_layout(self) -> None:
        layout = self.history_layout
        if layout is None:
            return
        try:
            expected_history_width = self._field_widths[layout.history_field]
        except KeyError as error:
            raise ValueError(f"Unknown history field: {layout.history_field!r}.") from error
        source_width = 0
        for source in layout.sources:
            width = (
                self.transition_schema.action_width
                if source.observation_name is None
                else self._field_widths.get(source.observation_name, 0)
            )
            if source.stop > width:
                raise ValueError("History source slice exceeds its observation or action width.")
            source_width += source.stop - source.start
        if expected_history_width != layout.history_length * source_width:
            raise ValueError("History field width does not match its source slices and history length.")
        if layout.last_action_field is not None:
            try:
                last_action_width = self._field_widths[layout.last_action_field]
            except KeyError as error:
                raise ValueError(f"Unknown last-action field: {layout.last_action_field!r}.") from error
            if last_action_width != self.transition_schema.action_width:
                raise ValueError("The reconstructed last-action field must match action_width.")

    def _reconstruct_derived(
        self,
        state_steps: torch.Tensor,
        env_ids: torch.Tensor,
        episode_ids: torch.Tensor,
        episode_steps: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        layout = self.history_layout
        complete = torch.ones(state_steps.shape[0], 1, device=self.device, dtype=torch.bool)
        if layout is None:
            return {}, complete
        derived: dict[str, torch.Tensor] = {}
        if layout.last_action_field is not None:
            source_steps = state_steps - 1
            rows = torch.remainder(source_steps, self.action_capacity_steps)
            node_rows = torch.remainder(source_steps, self.node_capacity_steps)
            source_valid = (self.node_episode_ids[node_rows, env_ids] == episode_ids) & (
                self.node_episode_steps[node_rows, env_ids] == episode_steps - 1
            )
            required = episode_steps >= 1
            complete &= (~required | source_valid).unsqueeze(-1)
            values = self.actions[rows, env_ids]
            derived[layout.last_action_field] = torch.where(
                source_valid.unsqueeze(-1), values, torch.zeros_like(values)
            )

        history_parts = []
        for source in layout.sources:
            lag_parts = []
            for lag in range(1, layout.history_length + 1):
                source_steps = state_steps - lag
                required = episode_steps >= lag
                if source.observation_name is None:
                    rows = torch.remainder(source_steps, self.action_capacity_steps)
                    node_rows = torch.remainder(source_steps, self.node_capacity_steps)
                    source_valid = (self.node_episode_ids[node_rows, env_ids] == episode_ids) & (
                        self.node_episode_steps[node_rows, env_ids] == episode_steps - lag
                    )
                    source_complete = source_valid
                    values = self.actions[rows, env_ids, source.start : source.stop]
                else:
                    rows = torch.remainder(source_steps, self.node_capacity_steps)
                    source_valid = (self.node_episode_ids[rows, env_ids] == episode_ids) & (
                        self.node_episode_steps[rows, env_ids] == episode_steps - lag
                    )
                    source_complete = source_valid
                    if self.node_history_valid is not None:
                        source_valid = source_valid & self.node_history_valid[rows, env_ids]
                    values = self.nodes[source.observation_name][rows, env_ids, source.start : source.stop]
                complete &= (~required | source_complete).unsqueeze(-1)
                lag_parts.append(torch.where(source_valid.unsqueeze(-1), values, torch.zeros_like(values)))
            history_parts.append(torch.cat(lag_parts, dim=-1))
        derived[layout.history_field] = torch.cat(history_parts, dim=-1)
        return derived, complete

    def _make_observations(
        self,
        base: dict[str, torch.Tensor],
        derived: dict[str, torch.Tensor],
        batch_size: int,
    ) -> TensorDict:
        values = {
            name: derived[name] if name in derived else base[name]
            for name, _width in self.observation_schema.field_widths
        }
        return TensorDict(values, batch_size=[batch_size], device=self.device)

    def _tensors(self) -> tuple[torch.Tensor, ...]:
        tensors = [value for value in self.nodes.values()] + [value for value in self.terminals.values()]
        tensors.extend([
            self.node_episode_ids,
            self.node_episode_steps,
            self.edge_flags,
            self.behavior_context,
            self.environment_reward,
            self.auxiliary_reward_evidence,
            self.actions,
            self.terminal_owner_steps,
            self.episode_ids,
            self.episode_steps,
            self.contract_errors,
            self.terminal_overflow,
        ])
        if self.node_history_valid is not None:
            tensors.append(self.node_history_valid)
        if self._sample_valid_counts is not None:
            tensors.append(self._sample_valid_counts)
            tensors.append(self._sample_valid_cdf)
        if self.sampling is self.Sampling.EPISODE_UNIFORM:
            tensors.extend([
                self._episode_slot_env_ids,
                self._episode_slot_is_terminal,
                self._episode_segment_starts,
                self._episode_segment_lengths,
                self._episode_segment_valid,
                self._episode_segment_cdf,
            ])
        return tuple(tensors)


def _allocate_observations(
    schema: ForwardBackwardObservationSchema,
    batch_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    fields: tuple[str, ...] | None = None,
) -> TensorDict:
    selected = set(fields) if fields is not None else None
    values = {
        name: torch.zeros(*batch_shape, width, device=device, dtype=dtype)
        for name, width in schema.field_widths
        if selected is None or name in selected
    }
    return TensorDict(values, batch_size=list(batch_shape), device=device)


def _copy_tensor_dict(destination: TensorDict, source: object) -> None:
    if not isinstance(source, TensorDict):
        raise TypeError("Replay TensorDict state has the wrong type.")
    for name in destination.keys(include_nested=False, leaves_only=True):
        destination[name].copy_(source[name])
