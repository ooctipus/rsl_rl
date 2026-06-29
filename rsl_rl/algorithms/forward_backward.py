# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL lifecycle and checkpoint boundary for forward-backward learning."""

from __future__ import annotations

import numpy as np
import random
import torch
from collections.abc import Mapping
from dataclasses import dataclass
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models.forward_backward_model import ForwardBackwardModel
from rsl_rl.modules.forward_backward import (
    actor_direct_loss,
    backward_implied_reward,
    backward_orthogonality_loss,
    discriminator_gradient_penalty,
    discriminator_logistic_loss,
    ensemble_pessimistic,
    forward_backward_loss,
    reward_value_td_loss,
    soft_update,
    trajectory_context,
    trajectory_context_sequence,
)
from rsl_rl.modules.reward_channels import (
    ForwardBackwardRewardChannel,
    ForwardBackwardRewardNormalizer,
    ForwardBackwardRewardSchema,
    ForwardBackwardValueSpec,
    get_forward_backward_schema_hash,
)
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer
from rsl_rl.storage.forward_backward_replay import (
    ForwardBackwardAutoresetMode,
    ForwardBackwardHistoryLayout,
    ForwardBackwardReplay,
    ForwardBackwardReplayBatch,
    ForwardBackwardTransitionBatch,
    ForwardBackwardTransitionSchema,
)
from rsl_rl.utils import resolve_callable, resolve_optimizer

FORWARD_BACKWARD_CHECKPOINT_FORMAT = "rsl_rl.forward_backward"
FORWARD_BACKWARD_CHECKPOINT_VERSION = 2
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
    """Unified off-policy MetaMotivo/BFM-Zero learner."""

    @dataclass(frozen=True, slots=True)
    class ValueCfg:
        """Optimization and actor composition for one named reward-value head."""

        learning_rate: float = 1e-4
        pessimism: float = 0.5
        actor_coefficient: float = 0.0
        reward_coefficients: tuple[float, ...] = (1.0,)
        normalize_rewards: bool = False
        reward_normalization_decay: float = 0.99
        reward_normalization_epsilon: float = 1e-8
        target_tau: float = 0.005

        def __post_init__(self) -> None:
            """Reject value settings that cannot define one stable TD update."""
            object.__setattr__(self, "reward_coefficients", tuple(self.reward_coefficients))
            if not self.reward_coefficients:
                raise ValueError("Value reward_coefficients must not be empty.")

    model: ForwardBackwardModel

    def __init__(
        self,
        model: ForwardBackwardModel,
        replay: ForwardBackwardReplay,
        expert: ForwardBackwardExpertBuffer,
        checkpoint_header: ForwardBackwardCheckpointHeader,
        *,
        batch_size: int = 1024,
        expert_sequence_length: int = 8,
        gamma: float = 0.98,
        learning_rate: float = 1e-4,
        backward_learning_rate: float = 1e-5,
        discriminator_learning_rate: float = 1e-5,
        value_cfg: Mapping[str, ValueCfg] | None = None,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        discriminator_weight_decay: float = 0.0,
        fb_pessimism: float = 0.5,
        actor_pessimism: float = 0.5,
        orthogonality_coefficient: float = 1.0,
        implied_value_coefficient: float = 0.0,
        implied_reward_ridge: float = 0.0,
        discriminator_gradient_penalty_coefficient: float = 10.0,
        context_goal_fraction: float = 0.2,
        context_expert_fraction: float = 0.6,
        relabel_fraction: float = 0.8,
        context_buffer_capacity: int = 10_000,
        fb_target_tau: float = 0.01,
        scale_actor_helpers: bool = True,
        max_grad_norm: float | None = None,
        random_action_range: tuple[float, float] | None = None,
        seed: int = 0,
        rollout_context_refresh_steps: int = 100,
        rollout_expert_fraction: float = 0.0,
        rollout_expert_steps: int = 250,
        rollout_expert_context_steps: int = 8,
        device: str | torch.device = "cpu",
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        """Create the learner and assign every trainable module to one optimizer."""
        if multi_gpu_cfg is not None:
            raise NotImplementedError("Forward-backward multi-GPU synchronization is not implemented.")
        if batch_size < 2 or expert_sequence_length < 1 or batch_size % expert_sequence_length:
            raise ValueError("batch_size must be at least two and divisible by expert_sequence_length.")
        if expert_sequence_length not in expert.schema.window_lengths:
            raise ValueError("expert_sequence_length is not available from the expert corpus.")
        if (
            min(
                context_goal_fraction,
                context_expert_fraction,
                1.0 - context_goal_fraction - context_expert_fraction,
            )
            < 0.0
            or not 0.0 <= relabel_fraction <= 1.0
        ):
            raise ValueError("Context mixture and relabel fractions must define probabilities.")
        if context_buffer_capacity < batch_size:
            raise ValueError("context_buffer_capacity must hold at least one update batch.")
        if rollout_context_refresh_steps < 1 or rollout_expert_steps < 1 or rollout_expert_context_steps < 1:
            raise ValueError("Rollout context periods must be positive.")
        if not 0.0 <= rollout_expert_fraction <= 1.0:
            raise ValueError("rollout_expert_fraction must be in [0, 1].")
        if random_action_range is None:
            random_action_range = getattr(model.action_distribution, "action_range", None)
        if random_action_range is not None:
            random_action_range = tuple(float(bound) for bound in random_action_range)
            if len(random_action_range) != 2 or random_action_range[0] >= random_action_range[1]:
                raise ValueError("random_action_range must contain ordered lower and upper bounds.")
        self.random_action_range = random_action_range

        requested_device = torch.device(device)
        self.model = model.to(requested_device)
        self.device = next(self.model.parameters()).device
        self._raw_model = self.model
        self.replay = replay
        self.expert = expert
        self.checkpoint_header = checkpoint_header
        if replay.device != self.device or expert.device != self.device:
            raise ValueError("Model, replay, and expert corpus must share the learner device.")
        if model.observation_schema.schema_hash != replay.observation_schema.schema_hash:
            raise ValueError("Model and replay observation schemas do not match.")

        self.batch_size = batch_size
        self.expert_sequence_length = expert_sequence_length
        self.gamma = gamma
        self.fb_pessimism = fb_pessimism
        self.actor_pessimism = actor_pessimism
        self.orthogonality_coefficient = orthogonality_coefficient
        self.implied_value_coefficient = implied_value_coefficient
        self.implied_reward_ridge = implied_reward_ridge
        self.discriminator_gradient_penalty_coefficient = discriminator_gradient_penalty_coefficient
        self.relabel_fraction = relabel_fraction
        self.fb_target_tau = fb_target_tau
        self.scale_actor_helpers = scale_actor_helpers
        self.max_grad_norm = max_grad_norm
        self.rollout_context_refresh_steps = rollout_context_refresh_steps
        self.rollout_expert_steps = rollout_expert_steps
        self.rollout_expert_context_steps = rollout_expert_context_steps

        backward_fields = model.observation_schema.route("backward")
        if (
            model.discriminator_network is not None
            and model.observation_schema.route("discriminator") != backward_fields
        ):
            raise ValueError("Phase 1 expert frames require matching backward and discriminator routes.")
        field_widths = dict(model.observation_schema.field_widths)
        if sum(field_widths[name] for name in backward_fields) != expert.schema.expert_feature_width:
            raise ValueError("Expert feature width does not match the backward observation route.")
        self._expert_fields = backward_fields
        self._expert_field_widths = tuple(field_widths[name] for name in backward_fields)

        specs = {spec.name: spec for spec in model.value_specs}
        configs = dict(value_cfg or {})
        if set(configs) != set(specs):
            raise ValueError("value_cfg keys must match the model's named value heads.")
        for name, spec in specs.items():
            spec.validate_reward_schema(replay.reward_schema)
            if not spec.has_target:
                raise ValueError(f"Off-policy value head {name!r} requires a target network.")
            if len(configs[name].reward_coefficients) != len(spec.reward_channels):
                raise ValueError(f"Value head {name!r} needs one coefficient per reward channel.")
        self.value_cfg = configs
        self._value_specs = specs
        reward_indices = {name: index for index, name in enumerate(replay.reward_schema.channel_names)}
        self._value_reward_indices = {
            name: tuple(reward_indices[channel] for channel in spec.reward_channels) for name, spec in specs.items()
        }
        self._value_coefficients = {
            name: torch.tensor(config.reward_coefficients, device=self.device, dtype=replay.dtype)
            for name, config in configs.items()
        }
        self._value_actor_coefficients = {
            name: coefficients if specs[name].reward_composition == "vector" else coefficients.new_ones(1)
            for name, coefficients in self._value_coefficients.items()
        }

        optimizer_class = resolve_optimizer(optimizer)
        actor_parameters = tuple(model.actor_network.parameters()) + tuple(model.action_distribution.parameters())
        self.actor_optimizer = optimizer_class(actor_parameters, lr=learning_rate, weight_decay=weight_decay)
        self.forward_optimizer = optimizer_class(
            model.forward_network.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.backward_optimizer = optimizer_class(
            model.backward_network.parameters(), lr=backward_learning_rate, weight_decay=weight_decay
        )
        self.discriminator_optimizer: torch.optim.Optimizer | None = None
        if model.discriminator_network is not None:
            self.discriminator_optimizer = optimizer_class(
                model.discriminator_network.parameters(),
                lr=discriminator_learning_rate,
                weight_decay=discriminator_weight_decay,
            )
        self.value_optimizers = {
            name: optimizer_class(
                model.value_networks[name].parameters(), lr=config.learning_rate, weight_decay=weight_decay
            )
            for name, config in configs.items()
        }
        self.reward_normalizers = torch.nn.ModuleDict({
            name: ForwardBackwardRewardNormalizer(
                config.reward_coefficients,
                decay=config.reward_normalization_decay,
                epsilon=config.reward_normalization_epsilon,
            )
            for name, config in configs.items()
            if config.normalize_rewards
        }).to(self.device)

        for channel in replay.reward_schema.channels:
            if channel.source == "recomputed" and channel.provider_name != "discriminator":
                raise ValueError(f"Unsupported recomputed reward provider: {channel.provider_name!r}.")
            if channel.provider_name == "discriminator" and model.discriminator_network is None:
                raise ValueError("A discriminator reward channel requires a discriminator network.")
            if channel.provider_name == "discriminator" and channel.timing == "transition":
                raise ValueError("Discriminator reward timing must be state or next_state.")

        self._context_mix_probabilities = torch.tensor(
            (context_goal_fraction, context_expert_fraction, 1.0 - context_goal_fraction - context_expert_fraction),
            device=self.device,
        )
        self.context_buffer = torch.zeros(
            context_buffer_capacity, model.context_dim, device=self.device, dtype=replay.dtype
        )
        self.context_buffer_cursor = 0
        self.context_buffer_size = 0
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.behavior_generator = torch.Generator(device=self.device)
        self.behavior_generator.manual_seed(seed)
        self.rollout_contexts = model.context_random(replay.num_envs, generator=self.generator)
        self._rollout_next_contexts = torch.empty_like(self.rollout_contexts)
        self._rollout_context_changed = torch.zeros(replay.num_envs, device=self.device, dtype=torch.bool)
        self._rollout_tracking_mask = torch.zeros_like(self._rollout_context_changed)
        self._empty_auxiliary_evidence = torch.empty(replay.num_envs, 0, device=self.device, dtype=replay.dtype)
        self.rollout_schedule_step = 0
        self._rollout_tracking_count = round(replay.num_envs * rollout_expert_fraction)
        self._rollout_tracking_env_ids = torch.empty(0, device=self.device, dtype=torch.long)
        self._rollout_tracking_contexts = torch.empty(
            0, rollout_expert_steps, model.context_dim, device=self.device, dtype=replay.dtype
        )
        if self._rollout_tracking_count:
            tracking_window = rollout_expert_steps + rollout_expert_context_steps - 1
            if tracking_window not in expert.schema.window_lengths:
                raise ValueError("The expert corpus does not provide the rollout tracking window.")
            self._rollout_tracking_env_ids, self._rollout_tracking_contexts = self._sample_rollout_tracking()
            self.rollout_contexts[self._rollout_tracking_env_ids] = self._rollout_tracking_contexts[:, 0]
        self._collection_observations: TensorDict | None = None
        self._collection_actions: torch.Tensor | None = None
        self.update_step = 0
        self.versions = {
            "actor": 0,
            "context": 0,
            "discriminator": 0,
            "normalizer": 0,
            "representation": 0,
            "target": 0,
            **{f"value/{name}": 0 for name in specs},
            **{f"reward_normalizer/{name}": 0 for name in self.reward_normalizers},
        }
        self._update_in_progress = False
        self.train_mode()

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> ForwardBackward:
        """Build the strict model, storage, and learner stack from runner config."""
        return _construct_forward_backward(obs, env, cfg, device)

    @property
    def ready_to_update(self) -> bool:
        """Return whether replay holds at least one complete update batch."""
        return self.replay.num_transitions >= self.batch_size

    @property
    def learning_rate(self) -> float:
        """Return the current actor learning rate for ordinary runner logging."""
        return float(self.actor_optimizer.param_groups[0]["lr"])

    @property
    def action_std(self) -> torch.Tensor:
        """Return the current action spread for ordinary runner logging."""
        return self.model.action_distribution.std

    def validate_collection(self) -> None:
        """Reduce deferred replay contract errors at the runner control boundary."""
        self.replay.assert_no_errors()

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample one behavior action and retain its immutable transition fields."""
        if self._collection_observations is not None:
            raise RuntimeError("The previous behavior action has not been processed.")
        observations = obs.to(self.device)
        actions = self.model.action_sample(observations, self.rollout_contexts)
        self._collection_observations = observations
        self._collection_actions = actions
        return actions

    def act_random(self, obs: TensorDict) -> torch.Tensor:
        """Sample uniform bounded behavior actions without advancing learner RNG."""
        if self._collection_observations is not None:
            raise RuntimeError("The previous behavior action has not been processed.")
        if self.random_action_range is None:
            raise TypeError("Random behavior requires explicit or distribution-owned action bounds.")
        observations = obs.to(self.device)
        lower, upper = self.random_action_range
        actions = torch.empty(
            self.replay.num_envs,
            self.model.action_dim,
            device=self.device,
            dtype=self.replay.dtype,
        ).uniform_(lower, upper, generator=self.behavior_generator)
        self._collection_observations = observations
        self._collection_actions = actions
        return actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict,
    ) -> None:
        """Normalize one environment result and append it to replay."""
        if self._collection_observations is None or self._collection_actions is None:
            raise RuntimeError("process_env_step requires a preceding act call.")

        next_observations = obs.to(self.device)
        current_observations = self._collection_observations
        actions = self._collection_actions
        num_envs = self.replay.num_envs
        done = dones.to(self.device).bool().reshape(num_envs, 1)
        timeout_value = extras.get("time_outs")
        timeouts = (
            torch.zeros_like(done)
            if timeout_value is None
            else timeout_value.to(self.device).bool().reshape(num_envs, 1)
        )
        truncated = done & timeouts
        terminated = done & ~truncated

        if self.replay.transition_schema.autoreset_mode is ForwardBackwardAutoresetMode.NEXT_STEP:
            action_applied = extras["action_applied"].to(self.device).bool().reshape(num_envs, 1)
        else:
            applied_value = extras.get("action_applied", torch.ones_like(done))
            action_applied = applied_value.to(self.device).bool().reshape(num_envs, 1)

        evidence_names = self.replay.transition_schema.auxiliary_evidence_names
        if evidence_names:
            evidence = extras["auxiliary_reward_evidence"].to(self.device)
        else:
            evidence = self._empty_auxiliary_evidence

        final_observations = current_observations
        final_observation_valid = torch.zeros_like(done)
        if (
            self.replay.transition_schema.autoreset_mode is ForwardBackwardAutoresetMode.SAME_STEP
            and "final_obs" in extras
        ):
            final_observations = _as_observations(extras["final_obs"], num_envs, self.device)
            valid_value = extras.get("final_obs_valid", done)
            final_observation_valid = valid_value.to(self.device).bool().reshape(num_envs, 1) & done

        behavior_context = self.rollout_contexts
        episode_steps = extras["episode_steps"].to(self.device).reshape(num_envs)
        context_changed = self._advance_rollout_contexts(action_applied, episode_steps)
        self.replay.add(
            ForwardBackwardTransitionBatch(
                observations=current_observations,
                next_observations=next_observations,
                final_observations=final_observations,
                actions=actions,
                behavior_context=behavior_context,
                environment_reward=rewards.to(self.device).reshape(num_envs, 1),
                auxiliary_reward_evidence=evidence,
                terminated=terminated,
                truncated=truncated,
                context_changed=context_changed,
                action_applied=action_applied,
                final_observation_valid=final_observation_valid,
            )
        )
        self._collection_observations = None
        self._collection_actions = None

    def process_env_reset(self, obs: TensorDict, reset: torch.Tensor) -> None:
        """Record an algorithm-controlled reset performed between environment steps."""
        if self._collection_observations is not None or self._collection_actions is not None:
            raise RuntimeError("process_env_reset cannot interrupt a pending behavior action.")
        reset = reset.to(self.device)
        observations = obs.to(self.device)
        self.replay.process_env_reset(observations, reset)
        reset = reset.reshape(self.replay.num_envs)

        tracking = self._rollout_tracking_mask
        tracking.zero_()
        tracking[self._rollout_tracking_env_ids] = True
        env_ids = (reset & ~tracking).nonzero(as_tuple=False).squeeze(-1)
        self.rollout_contexts[env_ids] = self._sample_rollout_contexts(env_ids.shape[0])

    def compute_returns(self, obs: TensorDict) -> None:
        """Do nothing because direct-Q learning has no rollout-return phase."""
        del obs

    @torch.no_grad()
    def _sample_rollout_tracking(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample env assignments and rolling expert contexts for one cycle."""
        env_ids = torch.randperm(self.replay.num_envs, device=self.device, generator=self.generator)[
            : self._rollout_tracking_count
        ]
        window_length = self.rollout_expert_steps + self.rollout_expert_context_steps - 1
        windows = self.expert.sample(self._rollout_tracking_count, window_length).frames
        observations = self._make_expert_observations(windows)
        backward = self.model.backward_map(observations).reshape(
            self._rollout_tracking_count, window_length, self.model.context_dim
        )
        contexts = trajectory_context_sequence(
            backward,
            self.rollout_expert_context_steps,
            include_partial=False,
            radius=self.model.context_dim**0.5 if self.model.context_normalization else None,
        )
        return env_ids, contexts

    def _sample_rollout_contexts(self, count: int) -> torch.Tensor:
        """Sample the learned update mixture, or the prior before its first update."""
        if count == 0:
            return self.context_buffer[:0]
        if self.context_buffer_size == 0:
            return self.model.context_random(count, generator=self.generator)
        indices = torch.randint(
            self.context_buffer_size,
            (count,),
            device=self.device,
            generator=self.generator,
        )
        return self.context_buffer[indices]

    @torch.no_grad()
    def _advance_rollout_contexts(self, action_applied: torch.Tensor, episode_steps: torch.Tensor) -> torch.Tensor:
        """Advance learned-mixture and rolling-expert contexts for the reached states."""
        next_step = self.rollout_schedule_step + 1
        next_contexts = self._rollout_next_contexts
        next_contexts.copy_(self.rollout_contexts)
        changed = self._rollout_context_changed
        changed.zero_()
        tracking = self._rollout_tracking_mask
        tracking.zero_()
        tracking[self._rollout_tracking_env_ids] = True

        refresh = episode_steps.remainder(self.rollout_context_refresh_steps) == 0
        env_ids = (refresh & ~tracking).nonzero(as_tuple=False).squeeze(-1)
        next_contexts[env_ids] = self._sample_rollout_contexts(env_ids.shape[0])
        changed[env_ids] = True

        if self._rollout_tracking_count:
            position = next_step % self.rollout_expert_steps
            if position:
                next_contexts[self._rollout_tracking_env_ids] = self._rollout_tracking_contexts[:, position]
                changed[self._rollout_tracking_env_ids] = True
            else:
                previous_env_ids = self._rollout_tracking_env_ids
                next_contexts[previous_env_ids] = self.model.context_random(
                    previous_env_ids.shape[0], generator=self.generator
                )
                self._rollout_tracking_env_ids, self._rollout_tracking_contexts = self._sample_rollout_tracking()
                next_contexts[self._rollout_tracking_env_ids] = self._rollout_tracking_contexts[:, 0]
                changed[previous_env_ids] = True
                changed[self._rollout_tracking_env_ids] = True

        self.rollout_contexts, self._rollout_next_contexts = next_contexts, self.rollout_contexts
        self.rollout_schedule_step = next_step
        return changed.unsqueeze(-1) & action_applied

    def update(self) -> dict[str, torch.Tensor]:
        """Run one visible off-policy mutation sequence."""
        if self._update_in_progress:
            raise RuntimeError("A forward-backward update is already in progress.")
        self._update_in_progress = True
        try:
            batch = self.replay.sample_random(self.batch_size)
            if not torch.all(batch.valid):
                raise RuntimeError("Replay sampled reset-only or otherwise invalid rows.")
            expert_batch = self.expert.sample(
                self.batch_size // self.expert_sequence_length,
                self.expert_sequence_length,
            )

            self.model.normalization_train(True)
            self.model.update_normalization(batch.observations)
            self.model.update_normalization(batch.next_observations)
            self.model.normalization_train(False)

            expert_observations = self._make_expert_observations(expert_batch.frames)
            with torch.no_grad():
                expert_contexts = trajectory_context(
                    self.model.backward_map(expert_observations),
                    radius=self.model.context_dim**0.5 if self.model.context_normalization else None,
                )
                expert_row_contexts = expert_contexts.repeat_interleave(self.expert_sequence_length, dim=0)

            metrics: dict[str, torch.Tensor] = {}
            if self.model.discriminator_network is not None:
                metrics.update(self._update_discriminator(expert_observations.reshape(-1), expert_row_contexts, batch))

            learner_contexts = self._sample_mixed_contexts(batch.next_observations, expert_row_contexts)
            self._append_contexts(learner_contexts)
            relabel = (
                torch.rand(
                    self.batch_size,
                    1,
                    device=self.device,
                    generator=self.generator,
                )
                < self.relabel_fraction
            )
            learner_contexts = torch.where(relabel, learner_contexts, batch.behavior_context)

            with torch.no_grad():
                next_actions = self.model.action_sample(batch.next_observations, learner_contexts)
            fb_metrics = self._update_forward_backward(batch, learner_contexts, next_actions)
            metrics.update({name: value.clone() for name, value in fb_metrics.items()})

            raw_rewards = self._materialize_rewards(batch, learner_contexts)
            for name in self._value_specs:
                metrics.update(self._update_value(name, batch, learner_contexts, next_actions, raw_rewards))

            metrics.update(self._update_actor(batch.observations, learner_contexts))
            self._update_targets()
            self._commit_versions()
        finally:
            self._update_in_progress = False

        metric_names = tuple(metrics)
        metric_values = torch.stack(tuple(metrics[name].detach() for name in metric_names))
        return dict(zip(metric_names, metric_values.unbind()))

    def _make_expert_observations(self, frames: torch.Tensor) -> TensorDict:
        chunks = torch.split(frames, self._expert_field_widths, dim=-1)
        return TensorDict(
            dict(zip(self._expert_fields, chunks)),
            batch_size=list(frames.shape[:-1]),
            device=self.device,
        )

    def _update_discriminator(
        self,
        expert_observations: TensorDict,
        expert_contexts: torch.Tensor,
        batch: ForwardBackwardReplayBatch,
    ) -> dict[str, torch.Tensor]:
        discriminator = self.model.discriminator_network
        optimizer = self.discriminator_optimizer
        assert discriminator is not None and optimizer is not None
        expert_logits = self.model.discriminator_logits(expert_observations, expert_contexts)
        replay_logits = self.model.discriminator_logits(batch.observations, batch.behavior_context)
        logistic = discriminator_logistic_loss(expert_logits, replay_logits)
        gradient_penalty = torch.zeros((), device=self.device)
        if self.discriminator_gradient_penalty_coefficient > 0.0:
            expert_route = self.model.get_normalized_observations(expert_observations, "discriminator")
            replay_route = self.model.get_normalized_observations(batch.observations, "discriminator")
            alpha = torch.rand(self.batch_size, 1, device=self.device, generator=self.generator)
            interpolated_route = (alpha * expert_route + (1.0 - alpha) * replay_route).requires_grad_(True)
            interpolated_context = (alpha * expert_contexts + (1.0 - alpha) * batch.behavior_context).requires_grad_(
                True
            )
            interpolated_logits = discriminator(interpolated_route, interpolated_context)
            gradient_penalty = discriminator_gradient_penalty(
                interpolated_logits,
                (interpolated_route, interpolated_context),
            )
        loss = logistic + self.discriminator_gradient_penalty_coefficient * gradient_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return {
            "discriminator/loss": loss.detach(),
            "discriminator/logistic": logistic.detach(),
            "discriminator/gradient_penalty": gradient_penalty.detach(),
        }

    @torch.no_grad()
    def _sample_mixed_contexts(
        self,
        goals: TensorDict,
        expert_contexts: torch.Tensor,
    ) -> torch.Tensor:
        goal_contexts = self.model.context_project(self.model.backward_map(goals))
        random_contexts = self.model.context_random(
            self.batch_size,
            device=self.device,
            dtype=goal_contexts.dtype,
            generator=self.generator,
        )
        mixture = torch.multinomial(
            self._context_mix_probabilities,
            self.batch_size,
            replacement=True,
            generator=self.generator,
        )
        goal_contexts = goal_contexts[torch.randperm(self.batch_size, device=self.device, generator=self.generator)]
        expert_contexts = expert_contexts[torch.randperm(self.batch_size, device=self.device, generator=self.generator)]
        contexts = torch.where(mixture.eq(0).unsqueeze(-1), goal_contexts, random_contexts)
        return torch.where(mixture.eq(1).unsqueeze(-1), expert_contexts, contexts)

    @torch.no_grad()
    def _append_contexts(self, contexts: torch.Tensor) -> None:
        first_count = min(self.batch_size, self.context_buffer.shape[0] - self.context_buffer_cursor)
        self.context_buffer[self.context_buffer_cursor : self.context_buffer_cursor + first_count].copy_(
            contexts[:first_count]
        )
        remaining = self.batch_size - first_count
        if remaining:
            self.context_buffer[:remaining].copy_(contexts[first_count:])
        self.context_buffer_cursor = (self.context_buffer_cursor + self.batch_size) % self.context_buffer.shape[0]
        self.context_buffer_size = min(self.context_buffer_size + self.batch_size, self.context_buffer.shape[0])

    def _update_forward_backward(
        self,
        batch: ForwardBackwardReplayBatch,
        contexts: torch.Tensor,
        next_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        continuation = self.gamma * batch.bootstrap_mask().to(dtype=batch.actions.dtype)
        current_forward = self.model.forward_map(batch.observations, contexts, batch.actions)
        current_backward = self.model.backward_map(batch.next_observations)
        with torch.no_grad():
            target_forward = self.model.forward_map(
                batch.next_observations,
                contexts,
                next_actions,
                target=True,
            )
            target_backward = self.model.backward_map(batch.next_observations, target=True)
        fb_loss, fb_off_diagonal, fb_diagonal = forward_backward_loss(
            current_forward,
            current_backward,
            target_forward,
            target_backward,
            continuation,
            self.fb_pessimism,
        )
        orthogonality, orthogonality_off_diagonal, orthogonality_diagonal = backward_orthogonality_loss(
            current_backward
        )
        total_loss = fb_loss + self.orthogonality_coefficient * orthogonality
        implied_value_loss = torch.zeros((), device=self.device)
        if self.implied_value_coefficient > 0.0:
            detached_backward = current_backward.detach()
            covariance = detached_backward.mT @ detached_backward / detached_backward.shape[0]
            implied_reward = backward_implied_reward(
                detached_backward,
                contexts,
                covariance,
                self.implied_reward_ridge,
            )
            implied_value_loss, _target = reward_value_td_loss(
                (current_forward * contexts).sum(dim=-1).unsqueeze(-1),
                (target_forward * contexts).sum(dim=-1).unsqueeze(-1),
                implied_reward.unsqueeze(-1),
                continuation,
                self.fb_pessimism,
            )
            total_loss = total_loss + self.implied_value_coefficient * implied_value_loss

        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.forward_network.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.model.backward_network.parameters(), self.max_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()
        return {
            "fb/loss": total_loss.detach(),
            "fb/measure": fb_loss.detach(),
            "fb/off_diagonal": fb_off_diagonal.detach(),
            "fb/diagonal": fb_diagonal.detach(),
            "fb/orthogonality": orthogonality.detach(),
            "fb/orthogonality_off_diagonal": orthogonality_off_diagonal.detach(),
            "fb/orthogonality_diagonal": orthogonality_diagonal.detach(),
            "fb/implied_value": implied_value_loss.detach(),
        }

    @torch.no_grad()
    def _materialize_rewards(
        self,
        batch: ForwardBackwardReplayBatch,
        contexts: torch.Tensor,
    ) -> torch.Tensor:
        evidence_indices = {
            name: index for index, name in enumerate(self.replay.transition_schema.auxiliary_evidence_names)
        }
        values = []
        discriminator_rewards: dict[str, torch.Tensor] = {}
        for channel in self.replay.reward_schema.channels:
            if channel.source == "environment":
                if channel.name != self.replay.transition_schema.environment_reward_name:
                    raise RuntimeError("Replay contains one declared environment reward channel.")
                value = batch.environment_reward
            elif channel.source == "stored_evidence":
                value = batch.auxiliary_reward_evidence[
                    :, evidence_indices[channel.name] : evidence_indices[channel.name] + 1
                ]
            else:
                try:
                    value = discriminator_rewards[channel.timing]
                except KeyError:
                    observations = batch.observations if channel.timing == "state" else batch.next_observations
                    value = self.model.discriminator_logits(observations, contexts).detach()
                    discriminator_rewards[channel.timing] = value
            values.append(channel.sign * value)
        return torch.cat(values, dim=-1)

    def _update_value(
        self,
        name: str,
        batch: ForwardBackwardReplayBatch,
        contexts: torch.Tensor,
        next_actions: torch.Tensor,
        raw_rewards: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        config = self.value_cfg[name]
        rewards = raw_rewards[:, self._value_reward_indices[name]]
        if name in self.reward_normalizers:
            normalizer = self.reward_normalizers[name]
            normalizer.update(rewards)
            rewards = normalizer(rewards)
        if self._value_specs[name].reward_composition == "scalar":
            rewards = rewards @ self._value_coefficients[name].unsqueeze(-1)
        continuation = self.gamma * batch.bootstrap_mask().to(dtype=batch.actions.dtype)
        values = self.model.critic_values(name, batch.observations, contexts, batch.actions)
        with torch.no_grad():
            target_values = self.model.critic_values(
                name,
                batch.next_observations,
                contexts,
                next_actions,
                target=True,
            )
        loss, target = reward_value_td_loss(values, target_values, rewards, continuation, config.pessimism)
        optimizer = self.value_optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.value_networks[name].parameters(), self.max_grad_norm)
        optimizer.step()
        return {
            f"value/{name}/loss": loss.detach(),
            f"value/{name}/reward": rewards.mean().detach(),
            f"value/{name}/target": target.mean().detach(),
        }

    def _update_actor(
        self,
        observations: TensorDict,
        contexts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        active_value_names = tuple(name for name, config in self.value_cfg.items() if config.actor_coefficient > 0.0)
        evaluators = (self.model.forward_network, *(self.model.value_networks[name] for name in active_value_names))
        for evaluator in evaluators:
            evaluator.zero_grad(set_to_none=True)
            evaluator.requires_grad_(False)
        try:
            actions = self.model.action_sample(observations, contexts, pathwise=True)
            forward_values = (self.model.forward_map(observations, contexts, actions) * contexts).sum(dim=-1)
            _mean, _disagreement, fb_values = ensemble_pessimistic(forward_values, self.actor_pessimism)
            value_channels = []
            coefficients = []
            for name in active_value_names:
                values = self.model.critic_values(name, observations, contexts, actions)
                _mean, _disagreement, pessimistic_values = ensemble_pessimistic(
                    values,
                    self.value_cfg[name].pessimism,
                )
                value_channels.append(pessimistic_values)
                coefficients.append(self.value_cfg[name].actor_coefficient * self._value_actor_coefficients[name])
            if value_channels:
                helper_values = torch.cat(value_channels, dim=-1)
                helper_coefficients = torch.cat(coefficients)
            else:
                helper_values = fb_values.new_empty(self.batch_size, 0)
                helper_coefficients = fb_values.new_empty(0)
            loss = actor_direct_loss(
                fb_values,
                helper_values,
                helper_coefficients,
                scale_channels=self.scale_actor_helpers,
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
        finally:
            for evaluator in evaluators:
                evaluator.requires_grad_(True)
        if self.max_grad_norm is not None:
            parameters = tuple(self.model.actor_network.parameters()) + tuple(
                self.model.action_distribution.parameters()
            )
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        self.actor_optimizer.step()
        return {
            "actor/loss": loss.detach(),
            "actor/fb_value": fb_values.mean().detach(),
            "actor/helper_value": (
                (helper_values * helper_coefficients).sum(dim=-1).mean().detach()
                if value_channels
                else torch.zeros((), device=self.device)
            ),
        }

    @torch.no_grad()
    def _update_targets(self) -> None:
        soft_update(
            tuple(self.model.forward_network.parameters()),
            tuple(self.model.forward_target_network.parameters()),
            self.fb_target_tau,
        )
        soft_update(
            tuple(self.model.backward_network.parameters()),
            tuple(self.model.backward_target_network.parameters()),
            self.fb_target_tau,
        )
        for name, config in self.value_cfg.items():
            soft_update(
                tuple(self.model.value_networks[name].parameters()),
                tuple(self.model.value_target_networks[name].parameters()),
                config.target_tau,
            )

    def _commit_versions(self) -> None:
        self.update_step += 1
        for name in ("actor", "context", "normalizer", "representation", "target"):
            self.versions[name] += 1
        if self.model.discriminator_network is not None:
            self.versions["discriminator"] += 1
        for name in self._value_specs:
            self.versions[f"value/{name}"] += 1
        for name in self.reward_normalizers:
            self.versions[f"reward_normalizer/{name}"] += 1

    def train_mode(self) -> None:
        """Train live modules while keeping targets and normalizers frozen."""
        self.model.train()
        self.model.normalization_train(False)
        self.model.forward_target_network.eval()
        self.model.backward_target_network.eval()
        self.model.value_target_networks.eval()

    def eval_mode(self) -> None:
        """Put every model component in evaluation mode."""
        self.model.eval()

    def save(self) -> dict[str, object]:
        """Return learner-exact state at a canonical update boundary."""
        if self._update_in_progress:
            raise RuntimeError("Cannot checkpoint during a forward-backward update.")
        if self._collection_observations is not None:
            raise RuntimeError("Cannot checkpoint with an unresolved environment transition.")
        optimizers: dict[str, object] = {
            "actor": self.actor_optimizer.state_dict(),
            "forward": self.forward_optimizer.state_dict(),
            "backward": self.backward_optimizer.state_dict(),
            "values": {name: optimizer.state_dict() for name, optimizer in self.value_optimizers.items()},
        }
        if self.discriminator_optimizer is not None:
            optimizers["discriminator"] = self.discriminator_optimizer.state_dict()
        rng: dict[str, object] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "learner": self.generator.get_state(),
            "behavior": self.behavior_generator.get_state(),
        }
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        return {
            FORWARD_BACKWARD_CHECKPOINT_HEADER: self.checkpoint_header.to_dict(),
            "model_state_dict": self._raw_model.state_dict(),
            "optimizer_state_dicts": optimizers,
            "reward_normalizer_state_dict": self.reward_normalizers.state_dict(),
            "replay_state_dict": self.replay.state_dict(),
            "expert_state_dict": self.expert.state_dict(),
            "context_buffer": self.context_buffer,
            "context_buffer_cursor": self.context_buffer_cursor,
            "context_buffer_size": self.context_buffer_size,
            "rollout_contexts": self.rollout_contexts,
            "rollout_schedule_step": self.rollout_schedule_step,
            "rollout_tracking_env_ids": self._rollout_tracking_env_ids,
            "rollout_tracking_contexts": self._rollout_tracking_contexts,
            "update_step": self.update_step,
            "versions": dict(self.versions),
            "rng_state": rng,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Restore selected learner state and report iteration ownership."""
        if self._update_in_progress:
            raise RuntimeError("Cannot restore during a forward-backward update.")
        self.checkpoint_header.validate_checkpoint(loaded_dict)
        if load_cfg is None:
            load_cfg = {
                "model": True,
                "optimizer": True,
                "storage": True,
                "context": True,
                "rng": True,
                "iteration": True,
            }
        if load_cfg.get("model"):
            self._raw_model.load_state_dict(loaded_dict["model_state_dict"], strict=strict)
            self.reward_normalizers.load_state_dict(loaded_dict["reward_normalizer_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            optimizers = loaded_dict["optimizer_state_dicts"]
            self.actor_optimizer.load_state_dict(optimizers["actor"])
            self.forward_optimizer.load_state_dict(optimizers["forward"])
            self.backward_optimizer.load_state_dict(optimizers["backward"])
            for name, optimizer in self.value_optimizers.items():
                optimizer.load_state_dict(optimizers["values"][name])
            if self.discriminator_optimizer is not None:
                self.discriminator_optimizer.load_state_dict(optimizers["discriminator"])
        if load_cfg.get("storage"):
            self.replay.load_state_dict(loaded_dict["replay_state_dict"])
            self.expert.load_state_dict(loaded_dict["expert_state_dict"])
        if load_cfg.get("context"):
            self.context_buffer.copy_(loaded_dict["context_buffer"])
            self.context_buffer_cursor = int(loaded_dict["context_buffer_cursor"])
            self.context_buffer_size = int(loaded_dict["context_buffer_size"])
            self.rollout_contexts.copy_(loaded_dict["rollout_contexts"])
            self.rollout_schedule_step = int(loaded_dict["rollout_schedule_step"])
            self._rollout_tracking_env_ids.copy_(loaded_dict["rollout_tracking_env_ids"])
            self._rollout_tracking_contexts.copy_(loaded_dict["rollout_tracking_contexts"])
            self.update_step = int(loaded_dict["update_step"])
            self.versions = dict(loaded_dict["versions"])
        if load_cfg.get("rng"):
            rng = loaded_dict["rng_state"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            self.generator.set_state(rng["learner"])
            self.behavior_generator.set_state(rng.get("behavior", rng["learner"]))
            if "cuda" in rng:
                torch.cuda.set_rng_state_all(rng["cuda"])
        return bool(load_cfg.get("iteration", False))

    def get_policy(self) -> ForwardBackwardModel:
        """Return the uncompiled composite policy model."""
        return self._raw_model

    def compile(self, mode: str | None = None) -> None:
        """Compile the FB and actor mutation blocks without wrapping model state."""
        if mode is None:
            return
        self._update_forward_backward = torch.compile(self._update_forward_backward, mode=mode)
        self._update_actor = torch.compile(self._update_actor, mode=mode)


def _construct_forward_backward(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> ForwardBackward:
    """Construct one forward-backward learner from ordinary RSL-RL sections."""
    model_class = resolve_callable(cfg["model"]["class_name"])
    if not isinstance(model_class, type) or not issubclass(model_class, ForwardBackwardModel):
        raise TypeError("The configured model class must derive from ForwardBackwardModel.")
    model = model_class.from_config(
        obs.to(device),
        cfg["obs_groups"],
        env.num_actions,
        cfg["model"],
    )

    replay_cfg = dict(cfg["replay"])
    replay_class = resolve_callable(replay_cfg.pop("class_name"))
    reward_schema = ForwardBackwardRewardSchema(
        tuple(ForwardBackwardRewardChannel(**dict(channel)) for channel in replay_cfg.pop("reward_channels"))
    )
    autoreset_mode = ForwardBackwardAutoresetMode(replay_cfg.pop("autoreset_mode"))
    transition_schema = ForwardBackwardTransitionSchema(
        observation_schema_hash=model.observation_schema.schema_hash,
        reward_schema_hash=reward_schema.schema_hash,
        action_width=env.num_actions,
        context_width=model.context_dim,
        environment_reward_name=replay_cfg.pop("environment_reward_name"),
        auxiliary_evidence_names=tuple(replay_cfg.pop("auxiliary_evidence_names")),
        autoreset_mode=autoreset_mode,
    )
    history_layout = _make_history_layout(replay_cfg.pop("history_layout", None))
    replay = replay_class(
        num_envs=env.num_envs,
        observation_schema=model.observation_schema,
        transition_schema=transition_schema,
        reward_schema=reward_schema,
        device=device,
        history_layout=history_layout,
        **replay_cfg,
    )
    if not isinstance(replay, ForwardBackwardReplay):
        raise TypeError("The configured replay must be a ForwardBackwardReplay.")

    expert_cfg = dict(cfg["expert"])
    provider = resolve_callable(expert_cfg.pop("provider"))
    expert = provider(env, model.observation_schema, device, **expert_cfg)
    if not isinstance(expert, ForwardBackwardExpertBuffer):
        raise TypeError("The expert provider must return ForwardBackwardExpertBuffer.")

    algorithm_cfg = dict(cfg["algorithm"])
    algorithm_class = resolve_callable(algorithm_cfg.pop("class_name"))
    value_cfg = {
        name: ForwardBackward.ValueCfg(**dict(value))
        for name, value in dict(algorithm_cfg.pop("value_cfg", {})).items()
    }
    manifest = {
        "config": {
            "algorithm": _checkpoint_config(cfg["algorithm"]),
            "model": _checkpoint_config(cfg["model"]),
            "obs_groups": _checkpoint_config(cfg["obs_groups"]),
            "replay": _checkpoint_config(cfg["replay"]),
        },
        "expert_schema_hash": expert.schema.schema_hash,
        "observation_schema_hash": model.observation_schema.schema_hash,
        "reward_schema_hash": reward_schema.schema_hash,
        "transition_schema_hash": transition_schema.schema_hash,
        "value_specs": tuple(_value_spec_data(spec) for spec in model.value_specs),
    }
    algorithm = algorithm_class(
        model,
        replay,
        expert,
        ForwardBackwardCheckpointHeader.from_manifest(manifest),
        value_cfg=value_cfg,
        device=device,
        multi_gpu_cfg=cfg.get("multi_gpu"),
        **algorithm_cfg,
    )
    if not isinstance(algorithm, ForwardBackward):
        raise TypeError("The configured algorithm must be ForwardBackward.")
    algorithm.compile(cfg.get("torch_compile_mode"))
    return algorithm


def _make_history_layout(value: object) -> ForwardBackwardHistoryLayout | None:
    """Build the optional compact-history reconstruction contract."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("history_layout must be a mapping or None.")
    options = dict(value)
    sources = tuple(ForwardBackwardHistoryLayout.Source(**dict(source)) for source in options.pop("sources"))
    return ForwardBackwardHistoryLayout(sources=sources, **options)


def _checkpoint_config(value: object) -> object:
    """Convert one resolved config section into stable JSON-compatible data."""
    if isinstance(value, Mapping):
        return {str(key): _checkpoint_config(item) for key, item in value.items() if key != "provider"}
    if isinstance(value, (tuple, list)):
        return tuple(_checkpoint_config(item) for item in value)
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if isinstance(value, torch.dtype):
        return str(value)
    return value


def _value_spec_data(spec: ForwardBackwardValueSpec) -> dict[str, object]:
    """Return the checkpoint-compatible fields of one value specification."""
    return {
        "ensemble_size": spec.ensemble_size,
        "has_target": spec.has_target,
        "kind": spec.kind,
        "name": spec.name,
        "reward_channels": spec.reward_channels,
        "reward_composition": spec.reward_composition,
        "route": spec.route,
    }


def _as_observations(value: object, num_envs: int, device: torch.device) -> TensorDict:
    """Normalize an environment final-observation payload to one TensorDict."""
    if isinstance(value, TensorDict):
        return value.to(device)
    if not isinstance(value, Mapping):
        raise TypeError("final_obs must be a TensorDict or tensor mapping.")
    return TensorDict(
        {str(name): tensor.to(device) for name, tensor in value.items()},
        batch_size=[num_envs],
        device=device,
    )
