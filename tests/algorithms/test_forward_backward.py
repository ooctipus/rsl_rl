# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the forward-backward runner and checkpoint boundary."""

from __future__ import annotations

import ast
import copy
import inspect
import numpy as np
import torch
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from tensordict import TensorDict, TensorDictBase
from typing import Literal

import pytest

import rsl_rl.algorithms
import rsl_rl.extensions
import rsl_rl.models
import rsl_rl.runners
import rsl_rl.storage
from rsl_rl.algorithms.forward_backward import (
    FORWARD_BACKWARD_CHECKPOINT_FORMAT,
    FORWARD_BACKWARD_CHECKPOINT_HEADER,
    FORWARD_BACKWARD_CHECKPOINT_VERSION,
    ForwardBackward,
    ForwardBackwardCheckpointHeader,
    _checkpoint_config,
    _ForwardBackwardOnlineHistory,
    forward_backward_model_from_config,
)
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models.forward_backward_model import (
    ForwardBackwardDualNetworkCfg,
    ForwardBackwardModel,
    ForwardBackwardObservationSchema,
    ForwardBackwardValueHeadCfg,
)
from rsl_rl.modules.reward_channels import ForwardBackwardRewardSchema, ForwardBackwardValueSpec
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer, ForwardBackwardExpertSchema
from rsl_rl.storage.forward_backward_replay import (
    ForwardBackwardAutoresetMode,
    ForwardBackwardHistoryLayout,
    ForwardBackwardReplay,
    ForwardBackwardTransitionBatch,
    ForwardBackwardTransitionSchema,
)
from tests.fixtures.forward_backward import META_ROUTES, make_meta_schema, make_reward_schema


def _make_expert_schema() -> ForwardBackwardExpertSchema:
    return ForwardBackwardExpertSchema(
        dataset_id="motion-corpus-v1",
        data_hash="data-v1",
        feature_schema_hash="features-v1",
        clip_offsets_hash="clips-v1",
        expert_feature_width=358,
        num_frames=1_000,
        num_clips=10,
        window_lengths=(1, 8),
    )


def _make_transition_schema(
    observation_schema_hash: str,
    reward_schema_hash: str,
    action_width: int = 2,
) -> ForwardBackwardTransitionSchema:
    return ForwardBackwardTransitionSchema(
        observation_schema_hash=observation_schema_hash,
        reward_schema_hash=reward_schema_hash,
        action_width=action_width,
        context_width=256,
        environment_reward_name="environment",
        auxiliary_evidence_names=("action_rate", "slip"),
        autoreset_mode=ForwardBackwardAutoresetMode.SAME_STEP,
    )


def _value_spec_data(spec: ForwardBackwardValueSpec) -> dict[str, object]:
    return {
        "ensemble_size": spec.ensemble_size,
        "has_target": spec.has_target,
        "kind": spec.kind,
        "name": spec.name,
        "reward_channels": spec.reward_channels,
        "reward_composition": spec.reward_composition,
        "route": spec.route,
    }


def _make_manifest() -> dict[str, object]:
    observation_schema = make_meta_schema()
    reward_schema = make_reward_schema()
    transition_schema = _make_transition_schema(observation_schema.schema_hash, reward_schema.schema_hash)
    value_spec = ForwardBackwardValueSpec(
        name="fb",
        kind="forward_readout",
        route="forward",
        reward_channels=("environment",),
        ensemble_size=2,
        has_target=True,
    )
    return {
        "config": {"algorithm": {"gamma": 0.99}, "model": {"context_width": 256}},
        "observation_schema_hash": observation_schema.schema_hash,
        "transition_schema_hash": transition_schema.schema_hash,
        "reward_schema_hash": reward_schema.schema_hash,
        "expert_schema_hash": _make_expert_schema().schema_hash,
        "value_specs": (_value_spec_data(value_spec),),
    }


def _make_header() -> ForwardBackwardCheckpointHeader:
    return ForwardBackwardCheckpointHeader.from_manifest(_make_manifest())


def _make_learner(
    *,
    include_auxiliary: bool = True,
    multi_gpu_cfg: dict | None = None,
    normalization_type: Literal["empirical", "exponential"] = "empirical",
    auxiliary_composition: Literal["vector", "scalar"] = "vector",
    random_action_range: tuple[float, float] | None = None,
    random_action_transitions: int = 0,
    optimization: dict[str, object] | None = None,
    context: dict[str, object] | None = None,
    exploration: dict[str, object] | None = None,
    prefill_steps: int = 5,
    include_environment_reward: bool = True,
) -> ForwardBackward:
    """Create a small deterministic learner with live replay and expert data."""
    torch.manual_seed(13)
    generator = torch.Generator().manual_seed(29)
    num_envs = 4
    action_width = 2
    context_width = 4
    state_width = 6
    routes = {
        "actor": ("state",),
        "forward": ("state",),
        "backward": ("state",),
        "discriminator": ("state",),
    }
    reward_schema = make_reward_schema()
    if not include_environment_reward:
        reward_schema = ForwardBackwardRewardSchema(
            tuple(channel for channel in reward_schema.channels if channel.source != "environment")
        )
    states = [
        torch.randn(num_envs, state_width, generator=generator) + step for step in range(max(6, prefill_steps + 1))
    ]
    observations = TensorDict({"state": states[0]}, batch_size=[num_envs])
    network = ForwardBackwardDualNetworkCfg(hidden_dim=16, hidden_layers=1, embedding_layers=2)
    value_heads = [
        ForwardBackwardValueHeadCfg(
            ForwardBackwardValueSpec(
                name="discriminator",
                kind="critic",
                route="critic_discriminator",
                reward_channels=("discriminator",),
                ensemble_size=2,
                has_target=True,
            ),
            network,
        )
    ]
    if include_auxiliary:
        value_heads.append(
            ForwardBackwardValueHeadCfg(
                ForwardBackwardValueSpec(
                    name="auxiliary",
                    kind="critic",
                    route="critic_auxiliary",
                    reward_channels=("action_rate", "slip"),
                    ensemble_size=2,
                    reward_composition=auxiliary_composition,
                    has_target=True,
                ),
                network,
            )
        )
    routes.update({head.spec.route: ("state",) for head in value_heads})
    model = ForwardBackwardModel(
        observations,
        routes,
        action_dim=action_width,
        context_dim=context_width,
        actor_cfg=network,
        forward_cfg=network,
        backward_hidden_dims=(16, 16),
        discriminator_hidden_dims=(16, 16),
        value_heads=tuple(value_heads),
        normalization_type=normalization_type,
        normalization_eps=1e-5 if normalization_type == "exponential" else 1e-2,
        normalization_momentum=0.01,
    )
    transition_schema = ForwardBackwardTransitionSchema(
        observation_schema_hash=model.observation_schema.schema_hash,
        reward_schema_hash=reward_schema.schema_hash,
        action_width=action_width,
        context_width=context_width,
        environment_reward_name="environment" if include_environment_reward else None,
        auxiliary_evidence_names=("action_rate", "slip"),
        autoreset_mode=ForwardBackwardAutoresetMode.SAME_STEP,
    )
    replay = ForwardBackwardReplay(
        5,
        num_envs,
        5,
        model.observation_schema,
        transition_schema,
        reward_schema,
        "cpu",
        seed=41,
    )
    false = torch.zeros(num_envs, 1, dtype=torch.bool)
    for step in range(prefill_steps):
        replay.add(
            ForwardBackwardTransitionBatch(
                observations=TensorDict({"state": states[step]}, batch_size=[num_envs]),
                next_observations=TensorDict({"state": states[step + 1]}, batch_size=[num_envs]),
                final_observations=TensorDict(
                    {"state": torch.full_like(states[step], float("nan"))}, batch_size=[num_envs]
                ),
                actions=torch.randn(num_envs, action_width, generator=generator),
                behavior_context=model.context_project(torch.randn(num_envs, context_width, generator=generator)),
                environment_reward=(
                    torch.randn(num_envs, 1, generator=generator)
                    if include_environment_reward
                    else torch.empty(num_envs, 0)
                ),
                auxiliary_reward_evidence=torch.rand(num_envs, 2, generator=generator),
                terminated=false,
                truncated=false,
                context_changed=false,
                action_applied=torch.ones_like(false),
                final_observation_valid=false,
            )
        )

    expert_schema = ForwardBackwardExpertSchema(
        dataset_id="small-motion-corpus",
        data_hash="small-data",
        feature_schema_hash=model.observation_schema.schema_hash,
        clip_offsets_hash="two-equal-clips",
        expert_feature_width=state_width,
        num_frames=16,
        num_clips=2,
        window_lengths=(2,),
    )
    expert = ForwardBackwardExpertBuffer(
        torch.randn(16, state_width, generator=generator),
        torch.tensor([0, 8, 16]),
        torch.ones(2),
        expert_schema,
        seed=43,
        clip_ids=("clip_0", "clip_1"),
        clip_length_values=(8, 8),
    )
    manifest = {
        "config": {"algorithm": {"gamma": 0.98}, "model": {"context_width": context_width}},
        "observation_schema_hash": model.observation_schema.schema_hash,
        "transition_schema_hash": transition_schema.schema_hash,
        "reward_schema_hash": reward_schema.schema_hash,
        "expert_schema_hash": expert_schema.schema_hash,
        "value_specs": tuple(_value_spec_data(spec) for spec in model.value_specs),
    }
    value_cfg = {
        "discriminator": ForwardBackward.ValueCfg(
            actor_coefficient=0.05,
            reward_coefficients=(1.0,),
        )
    }
    if include_auxiliary:
        value_cfg["auxiliary"] = ForwardBackward.ValueCfg(
            actor_coefficient=0.02,
            reward_coefficients=(0.1, 0.4),
            normalize_rewards=True,
        )
    if optimization is None:
        optimization = {
            "learning_rate": 1.0e-4,
            "backward_learning_rate": 1.0e-5,
            "discriminator_learning_rate": 1.0e-5,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "discriminator_weight_decay": 0.0,
            "max_grad_norm": None,
        }
    if context is None:
        context = {
            "goal_fraction": 0.2,
            "expert_fraction": 0.6,
            "relabel_fraction": 0.8,
            "buffer_capacity": 16,
            "refresh_steps": 100,
            "rollout_expert_fraction": 0.0,
            "rollout_expert_steps": 250,
            "rollout_expert_context_steps": 8,
        }
    if exploration is None:
        exploration = {
            "random_action_range": random_action_range,
            "random_action_transitions": random_action_transitions,
        }
    return ForwardBackward(
        model,
        replay,
        expert,
        ForwardBackwardCheckpointHeader.from_manifest(manifest),
        auxiliary_evidence_observation_group="transition",
        optimization=optimization,
        context=context,
        exploration=exploration,
        batch_size=8,
        expert_sequence_length=2,
        value_cfg=value_cfg,
        implied_value_coefficient=0.1,
        implied_reward_ridge=0.1,
        discriminator_gradient_penalty_coefficient=0.1,
        seed=47,
        multi_gpu_cfg=multi_gpu_cfg,
    )


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("optimization", "learning_rate"),
        ("optimization", "backward_learning_rate"),
        ("optimization", "discriminator_learning_rate"),
        ("optimization", "optimizer"),
        ("optimization", "weight_decay"),
        ("optimization", "discriminator_weight_decay"),
        ("optimization", "max_grad_norm"),
        ("context", "goal_fraction"),
        ("context", "expert_fraction"),
        ("context", "relabel_fraction"),
        ("context", "buffer_capacity"),
        ("context", "refresh_steps"),
        ("context", "rollout_expert_fraction"),
        ("context", "rollout_expert_steps"),
        ("context", "rollout_expert_context_steps"),
        ("exploration", "random_action_range"),
        ("exploration", "random_action_transitions"),
    ),
)
def test_grouped_policy_requires_every_declared_member(section: str, field: str) -> None:
    """Incomplete grouped policies must not recover hidden algorithm defaults."""
    records = {
        "optimization": {
            "learning_rate": 1.0e-4,
            "backward_learning_rate": 1.0e-5,
            "discriminator_learning_rate": 1.0e-5,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "discriminator_weight_decay": 0.0,
            "max_grad_norm": None,
        },
        "context": {
            "goal_fraction": 0.2,
            "expert_fraction": 0.6,
            "relabel_fraction": 0.8,
            "buffer_capacity": 16,
            "refresh_steps": 100,
            "rollout_expert_fraction": 0.0,
            "rollout_expert_steps": 250,
            "rollout_expert_context_steps": 8,
        },
        "exploration": {
            "random_action_range": None,
            "random_action_transitions": 0,
        },
    }
    del records[section][field]

    with pytest.raises(KeyError, match=field):
        _make_learner(**{section: records[section]})


def _evaluation_history_layout() -> ForwardBackwardHistoryLayout:
    """Return the compact reached-transition history used by evaluator tests."""
    return ForwardBackwardHistoryLayout(
        history_field="history_actor",
        history_length=2,
        sources=(ForwardBackwardHistoryLayout.Source("state"),),
        include_seed_observations=False,
    )


def test_evaluation_history_allocates_from_sources_when_tensordict_device_is_none() -> None:
    """Derived tensors should follow source tensors rather than TensorDict metadata."""
    observations = TensorDict({"state": torch.ones(2, 2)}, batch_size=[2])
    assert observations.device is None

    history = ForwardBackward.EvaluationHistory(_evaluation_history_layout(), observations)
    decorated = history.decorate_current(observations)

    assert decorated["history_actor"].device == observations["state"].device
    assert decorated["history_actor"].shape == (2, 4)
    torch.testing.assert_close(decorated["history_actor"], torch.zeros(2, 4))


def test_evaluation_history_advances_exactly_once_and_resets_done_rows() -> None:
    """Same-step evaluation should shift reached sources and clear autoreset rows."""
    initial = TensorDict({"state": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}, batch_size=[2])
    history = ForwardBackward.EvaluationHistory(_evaluation_history_layout(), initial)
    history.decorate_current(initial)

    first = TensorDict({"state": torch.tensor([[10.0, 11.0], [12.0, 13.0]])}, batch_size=[2])
    history.advance(initial, first, torch.tensor([False, True]))
    torch.testing.assert_close(first["history_actor"], torch.zeros(2, 4))

    second = TensorDict({"state": torch.tensor([[20.0, 21.0], [22.0, 23.0]])}, batch_size=[2])
    history.advance(first, second, torch.tensor([False, False]))
    torch.testing.assert_close(
        second["history_actor"],
        torch.tensor([[10.0, 11.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
    )

    third = TensorDict({"state": torch.tensor([[30.0, 31.0], [32.0, 33.0]])}, batch_size=[2])
    history.advance(second, third, torch.tensor([False, False]))
    torch.testing.assert_close(
        third["history_actor"],
        torch.tensor([[20.0, 21.0, 10.0, 11.0], [22.0, 23.0, 0.0, 0.0]]),
    )


def test_evaluation_history_factory_is_optional_and_independent_from_collection() -> None:
    """Evaluation should allocate a fresh session without aliasing collection history."""
    learner = _make_learner()
    raw = TensorDict({"state": torch.ones(4, 6)}, batch_size=[4])
    assert learner.evaluation_history(raw) is None

    layout = ForwardBackwardHistoryLayout(
        history_field="history_actor",
        history_length=1,
        sources=(ForwardBackwardHistoryLayout.Source("state"),),
        include_seed_observations=False,
    )
    learner._online_history = _ForwardBackwardOnlineHistory(layout, raw)
    learner._online_history.current.fill_(7.0)
    evaluation = learner.evaluation_history(raw)

    assert evaluation is not None
    decorated = evaluation.decorate_current(raw.clone())
    torch.testing.assert_close(decorated["history_actor"], torch.zeros_like(learner._online_history.current))
    assert decorated["history_actor"].data_ptr() != learner._online_history.current.data_ptr()


def test_evaluation_history_factory_rejects_non_same_step_replay() -> None:
    """The evaluator must not guess history semantics for another autoreset mode."""
    learner = _make_learner()
    raw = TensorDict({"state": torch.ones(4, 6)}, batch_size=[4])
    layout = _evaluation_history_layout()
    learner._online_history = _ForwardBackwardOnlineHistory(layout, raw)
    learner.replay.transition_schema = replace(
        learner.replay.transition_schema,
        autoreset_mode=ForwardBackwardAutoresetMode.NEXT_STEP,
    )

    with pytest.raises(NotImplementedError, match="same-step"):
        learner.evaluation_history(raw)


def test_vanilla_fb_default_uses_the_ensemble_mean_target() -> None:
    """The base FB algorithm should not add pessimism absent an explicit choice."""
    assert _make_learner().fb_pessimism == 0.0


def test_algorithm_constructor_rejects_unknown_config_fields() -> None:
    """An algorithm typo should fail through Python's explicit constructor semantics."""
    with pytest.raises(TypeError, match="learnig_rate"):
        ForwardBackward(learnig_rate=1.0e-4)  # type: ignore[call-arg]


def test_algorithm_owns_random_warmup_and_released_update_boundary() -> None:
    """Policy behavior starts at warm-up, while updates start after its following row."""
    before = _make_learner(
        random_action_range=(20.0, 21.0),
        random_action_transitions=8,
        prefill_steps=0,
    )
    at_boundary = _make_learner(
        random_action_range=(20.0, 21.0),
        random_action_transitions=8,
        prefill_steps=2,
    )
    following_row = _make_learner(
        random_action_range=(20.0, 21.0),
        random_action_transitions=8,
        prefill_steps=3,
    )
    after_following_row = _make_learner(
        random_action_range=(20.0, 21.0),
        random_action_transitions=8,
        prefill_steps=4,
    )
    observations = TensorDict({"state": torch.zeros(4, 6)}, batch_size=[4])

    random_actions = before.act(observations)
    policy_actions = at_boundary.act(observations)

    assert torch.all((random_actions >= 20.0) & (random_actions < 21.0))
    assert torch.all(policy_actions.abs() <= 1.0)
    assert before.ready_to_update is False
    assert at_boundary.replay.total_steps * at_boundary.replay.num_envs == 8
    assert at_boundary.ready_to_update is False
    assert following_row.replay.total_steps * following_row.replay.num_envs == 12
    assert following_row.ready_to_update is False
    assert after_following_row.replay.total_steps * after_following_row.replay.num_envs == 16
    assert after_following_row.ready_to_update is True

    wrapped = _make_learner(
        random_action_range=(20.0, 21.0),
        random_action_transitions=24,
        prefill_steps=8,
    )
    assert wrapped.replay.num_transitions == 20
    assert wrapped.replay.total_steps * wrapped.replay.num_envs == 32
    assert wrapped.ready_to_update is True


def test_runtime_materializes_canonical_helpers_and_root_seed_once() -> None:
    """The runtime composition root should derive every helper consumer from one declaration."""

    class Env:
        num_envs = 4
        num_actions = 2

    def provider(
        env: object,
        observation_schema: ForwardBackwardObservationSchema,
        device: str,
        *,
        clock: dict[str, object],
        window_lengths: tuple[int, ...],
        seed: int,
    ) -> ForwardBackwardExpertBuffer:
        del env
        assert clock == {"sampling_mode": "source_rows", "sampling_step_seconds": None}
        width = observation_schema.route_width("backward")
        schema = ForwardBackwardExpertSchema(
            dataset_id="canonical-helper",
            data_hash="canonical-data",
            feature_schema_hash=observation_schema.schema_hash,
            clip_offsets_hash="two-clips",
            expert_feature_width=width,
            num_frames=16,
            num_clips=2,
            window_lengths=window_lengths,
        )
        return ForwardBackwardExpertBuffer(
            torch.zeros(16, width, device=device),
            torch.tensor([0, 8, 16], device=device),
            torch.ones(2, device=device),
            schema,
            seed=seed,
            clip_ids=("first", "second"),
            clip_length_values=(8, 8),
        )

    network = {"hidden_dim": 16, "hidden_layers": 1, "embedding_layers": 2}
    config = {
        "seed": 73,
        "obs_groups": {
            "actor": ["state"],
            "forward": ["state"],
            "backward": ["state"],
            "discriminator": ["state"],
            "critic_discriminator": ["state"],
        },
        "model": {
            "class_name": "rsl_rl.models.forward_backward_model:ForwardBackwardModel",
            "context_dim": 4,
            "actor_cfg": network,
            "forward_cfg": network,
            "backward_hidden_dims": [16],
            "discriminator_hidden_dims": [16],
        },
        "replay": {
            "class_name": "rsl_rl.storage.forward_backward_replay:ForwardBackwardReplay",
            "policy": {
                "capacity_transitions": 32,
                "terminal_capacity_per_env": 4,
                "sampling": "transition_uniform",
            },
            "autoreset_mode": "same_step",
        },
        "expert": {
            "provider": provider,
            "clock": {"sampling_mode": "source_rows", "sampling_step_seconds": None},
            "window_lengths": (2,),
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms.forward_backward:ForwardBackward",
            "batch_size": 8,
            "expert_sequence_length": 2,
            "optimization": {
                "learning_rate": 3.0e-4,
                "backward_learning_rate": 1.0e-5,
                "discriminator_learning_rate": 1.0e-5,
                "optimizer": "adam",
                "weight_decay": 0.0,
                "discriminator_weight_decay": 0.0,
                "max_grad_norm": None,
            },
            "context": {
                "goal_fraction": 0.2,
                "expert_fraction": 0.6,
                "relabel_fraction": 0.8,
                "buffer_capacity": 16,
                "refresh_steps": 100,
                "rollout_expert_fraction": 0.0,
                "rollout_expert_steps": 250,
                "rollout_expert_context_steps": 8,
            },
            "exploration": {
                "random_action_range": None,
                "random_action_transitions": 0,
            },
            "discriminator_gradient_penalty_coefficient": 0.0,
        },
        "value_helpers": (
            {
                "name": "discriminator",
                "learning_rate": 3.0e-4,
                "route": "critic_discriminator",
                "reward_composition": "vector",
                "terms": (
                    {
                        "name": "discriminator",
                        "coefficient": 1.0,
                        "source": "recomputed",
                        "timing": "next_state",
                        "context_dependent": True,
                        "sign": 1,
                    },
                ),
                "pessimism": 0.5,
                "actor_coefficient": 0.05,
                "target_tau": 0.005,
            },
            {
                "name": "shared_discriminator",
                "learning_rate": 3.0e-4,
                "route": "critic_discriminator",
                "reward_composition": "vector",
                "terms": (
                    {
                        "name": "discriminator",
                        "coefficient": 0.5,
                        "source": "recomputed",
                        "timing": "next_state",
                        "context_dependent": True,
                        "sign": 1,
                    },
                ),
                "pessimism": 0.25,
                "actor_coefficient": 0.02,
                "target_tau": 0.01,
            },
        ),
        "torch_compile_mode": None,
    }
    original = copy.deepcopy(config)

    learner = ForwardBackward.construct_algorithm(
        TensorDict({"state": torch.zeros(4, 6)}, batch_size=[4]),
        Env(),
        config,
        "cpu",
    )
    evaluation_model = forward_backward_model_from_config(
        TensorDict({"state": torch.zeros(4, 6)}, batch_size=[4]),
        config["obs_groups"],
        Env.num_actions,
        config["model"],
        config["value_helpers"],
    )

    assert config == original
    assert tuple(evaluation_model.state_dict()) == tuple(learner.model.state_dict())
    assert learner.replay.reward_schema.channel_names == ("discriminator",)
    assert learner.replay.transition_schema.environment_reward_name is None
    assert learner.replay.transition_schema.auxiliary_evidence_names == ()
    assert tuple(learner.model.value_networks) == ("discriminator", "shared_discriminator")
    assert learner.model.value_specs[0].reward_composition == "vector"
    assert learner.value_cfg["discriminator"].learning_rate == 3.0e-4
    assert learner.actor_optimizer.param_groups[0]["lr"] == 3.0e-4
    assert learner.context_buffer.shape == (16, 4)
    assert learner.rollout_context_refresh_steps == 100
    assert learner.random_action_transitions == 0
    assert learner.value_cfg["discriminator"].reward_coefficients == (1.0,)
    assert learner.value_cfg["shared_discriminator"].reward_coefficients == (0.5,)
    assert learner.replay.generator.initial_seed() == 73
    assert learner.expert.generator.initial_seed() == 73
    assert learner.generator.initial_seed() == 73
    expected_header = ForwardBackwardCheckpointHeader.from_manifest({
        "config": {
            "algorithm": _checkpoint_config(config["algorithm"]),
            "model": _checkpoint_config(config["model"]),
            "obs_groups": _checkpoint_config(config["obs_groups"]),
            "replay": _checkpoint_config(config["replay"]),
            "seed": 73,
            "value_helpers": _checkpoint_config(config["value_helpers"]),
        },
        "expert_schema_hash": learner.expert.schema.schema_hash,
        "observation_schema_hash": learner.model.observation_schema.schema_hash,
        "reward_schema_hash": learner.replay.reward_schema.schema_hash,
        "transition_schema_hash": learner.replay.transition_schema.schema_hash,
        "value_specs": tuple(_value_spec_data(spec) for spec in learner.model.value_specs),
    })
    assert learner.checkpoint_header == expected_header

    invalid = copy.deepcopy(config)
    invalid["value_helpers"][0]["reward_composition"] = None
    with pytest.raises(ValueError, match="Unsupported reward composition"):
        ForwardBackward.construct_algorithm(
            TensorDict({"state": torch.zeros(4, 6)}, batch_size=[4]),
            Env(),
            invalid,
            "cpu",
        )

    invalid = copy.deepcopy(config)
    invalid["replay"]["auxiliary_evidence_observation_group"] = None
    with pytest.raises(ValueError, match="derived from value_helpers"):
        ForwardBackward.construct_algorithm(
            TensorDict({"state": torch.zeros(4, 6)}, batch_size=[4]),
            Env(),
            invalid,
            "cpu",
        )

    invalid = copy.deepcopy(config)
    invalid["expert"]["seed"] = 74
    with pytest.raises(ValueError, match="expert seed is owned by the runner root"):
        ForwardBackward.construct_algorithm(
            TensorDict({"state": torch.zeros(4, 6)}, batch_size=[4]),
            Env(),
            invalid,
            "cpu",
        )


def test_multi_gpu_fails_until_synchronization_is_implemented() -> None:
    """The shell should not silently run unsynchronized distributed training."""
    with pytest.raises(NotImplementedError, match="multi-GPU"):
        _make_learner(multi_gpu_cfg={"world_size": 2})


@pytest.mark.parametrize(
    "method_name",
    (
        "construct_algorithm",
        "act",
        "process_env_step",
        "compute_returns",
        "update",
        "train_mode",
        "eval_mode",
        "save",
        "load",
        "get_policy",
        "compile",
    ),
)
def test_algorithm_methods_follow_the_rsl_protocol(method_name: str) -> None:
    """The generic runner should call ForwardBackward exactly like PPO."""
    forward_backward_parameters = tuple(inspect.signature(getattr(ForwardBackward, method_name)).parameters)
    ppo_parameters = tuple(inspect.signature(getattr(PPO, method_name)).parameters)

    assert forward_backward_parameters == ppo_parameters


def test_phase_1f_collection_retains_one_immutable_pending_action() -> None:
    """Collection should bind one observation/action/context tuple until env.step resolves it."""
    algorithm = _make_learner()
    assert not inspect.isabstract(ForwardBackward)
    actions = algorithm.act(TensorDict({"state": torch.zeros(4, 6)}, batch_size=[4]))
    assert actions.shape == (4, 2)
    with pytest.raises(RuntimeError, match="unresolved environment transition"):
        algorithm.save()


def test_collection_ignores_environment_rewards_when_channel_is_absent() -> None:
    """A helper-only learner should neither inspect nor materialize environment rewards."""
    algorithm = _make_learner(include_environment_reward=False)
    observations = TensorDict(
        {
            "state": torch.zeros(4, 6),
            "transition": TensorDict(
                {
                    "action_rate": torch.ones(4),
                    "slip": 2.0 * torch.ones(4),
                },
                batch_size=[4],
            ),
        },
        batch_size=[4],
    )
    algorithm.act(observations)

    algorithm.process_env_step(
        observations,
        torch.tensor(float("nan")),
        torch.zeros(4, dtype=torch.bool),
        {},
    )
    batch = algorithm.replay.sample(torch.full((4,), 5), torch.arange(4))
    rewards = algorithm._materialize_rewards(batch, algorithm.rollout_contexts)

    assert algorithm.replay.environment_reward.shape == (5, 4, 0)
    assert batch.environment_reward.shape == (4, 0)
    assert rewards.shape == (4, 3)
    assert torch.all(torch.isfinite(rewards))


def test_random_behavior_uses_independent_explicit_action_bounds() -> None:
    """Warm-up support should match the environment without widening learned actor actions."""
    algorithm = _make_learner(random_action_range=(-5.0, 5.0))
    observations = TensorDict({"state": torch.zeros(4, 6)}, batch_size=[4])

    actions = algorithm.act_random(observations)

    assert torch.all(actions >= -5.0)
    assert torch.all(actions <= 5.0)
    assert torch.any(actions.abs() > 1.0)
    assert algorithm.model.action_distribution.action_range == (-1.0, 1.0)


def test_scalar_auxiliary_composes_channels_before_value_propagation() -> None:
    """Released BFM-style helpers should predict one weighted reward return."""
    learner = _make_learner(auxiliary_composition="scalar")
    observations = TensorDict({"state": torch.zeros(4, 6)}, batch_size=[4])
    actions = torch.zeros(4, 2)

    values = learner.model.critic_values("auxiliary", observations, learner.rollout_contexts, actions)

    assert values.shape == (2, 4, 1)
    torch.testing.assert_close(learner._value_actor_coefficients["auxiliary"], torch.ones(1))
    assert torch.isfinite(learner.update()["value/auxiliary/loss"])


def _parameter_snapshot(module: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().clone() for parameter in module.parameters())


def _parameters_changed(before: tuple[torch.Tensor, ...], module: torch.nn.Module) -> bool:
    return any(not torch.equal(previous, current) for previous, current in zip(before, module.parameters()))


def test_complete_update_sequence_matches_reference_dependency_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """D, representations, values, actor, and targets should mutate in source order."""
    learner = _make_learner()
    events: list[str] = []

    original_normalization = learner.model.update_normalization

    def update_normalization(observations: TensorDict) -> None:
        events.append("normalization")
        original_normalization(observations)

    monkeypatch.setattr(learner.model, "update_normalization", update_normalization)

    def record(method_name: str) -> None:
        original: Callable[..., object] = getattr(learner, method_name)

        def wrapped(*args: object, **kwargs: object) -> object:
            if method_name == "_update_value":
                events.append(f"value/{args[0]}")
            else:
                events.append(method_name.removeprefix("_"))
            return original(*args, **kwargs)

        monkeypatch.setattr(learner, method_name, wrapped)

    for method_name in (
        "_update_discriminator",
        "_sample_mixed_contexts",
        "_append_contexts",
        "_update_forward_backward",
        "_materialize_rewards",
        "_update_value",
        "_update_actor",
        "_update_targets",
        "_commit_versions",
    ):
        record(method_name)

    learner.update()

    assert events == [
        "normalization",
        "normalization",
        "update_discriminator",
        "sample_mixed_contexts",
        "append_contexts",
        "update_forward_backward",
        "materialize_rewards",
        "value/discriminator",
        "value/auxiliary",
        "update_actor",
        "update_targets",
        "commit_versions",
    ]


def test_one_update_mutates_every_declared_owner_and_no_actor_evaluator_grads() -> None:
    """The ordered update should step live owners, EMA targets, and only actor gradients last."""
    learner = _make_learner()
    model = learner.model
    live_before = {
        "actor": _parameter_snapshot(model.actor_network),
        "forward": _parameter_snapshot(model.forward_network),
        "backward": _parameter_snapshot(model.backward_network),
        "discriminator": _parameter_snapshot(model.discriminator_network),  # type: ignore[arg-type]
        "value_discriminator": _parameter_snapshot(model.value_networks["discriminator"]),
        "value_auxiliary": _parameter_snapshot(model.value_networks["auxiliary"]),
    }
    target_before = {
        "forward": _parameter_snapshot(model.forward_target_network),
        "backward": _parameter_snapshot(model.backward_target_network),
        "discriminator": _parameter_snapshot(model.value_target_networks["discriminator"]),
        "auxiliary": _parameter_snapshot(model.value_target_networks["auxiliary"]),
    }

    metrics = learner.update()

    assert all(value.ndim == 0 and not value.requires_grad for value in metrics.values())
    assert {
        "discriminator/loss",
        "fb/loss",
        "fb/implied_value",
        "value/discriminator/loss",
        "value/auxiliary/loss",
        "actor/loss",
    }.issubset(metrics)
    assert _parameters_changed(live_before["actor"], model.actor_network)
    assert _parameters_changed(live_before["forward"], model.forward_network)
    assert _parameters_changed(live_before["backward"], model.backward_network)
    assert _parameters_changed(live_before["discriminator"], model.discriminator_network)  # type: ignore[arg-type]
    assert _parameters_changed(live_before["value_discriminator"], model.value_networks["discriminator"])
    assert _parameters_changed(live_before["value_auxiliary"], model.value_networks["auxiliary"])
    assert _parameters_changed(target_before["forward"], model.forward_target_network)
    assert _parameters_changed(target_before["backward"], model.backward_target_network)
    assert _parameters_changed(target_before["discriminator"], model.value_target_networks["discriminator"])
    assert _parameters_changed(target_before["auxiliary"], model.value_target_networks["auxiliary"])
    assert all(parameter.grad is None for parameter in model.forward_network.parameters())
    assert all(
        parameter.grad is None for network in model.value_networks.values() for parameter in network.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.actor_network.parameters())
    assert learner.update_step == 1
    assert learner.context_buffer_size == learner.batch_size
    assert learner.versions["actor"] == 1
    assert model.observation_normalizers["state"].count.item() == 2 * learner.batch_size
    assert learner.reward_normalizers["auxiliary"].count.item() == 1


def test_meta_and_bfm_component_sets_use_the_same_update_class() -> None:
    """Optional auxiliary values should disappear without a learner subclass or dummy loss."""
    meta = _make_learner(include_auxiliary=False)
    bfm = _make_learner(include_auxiliary=True)

    meta_metrics = meta.update()
    bfm_metrics = bfm.update()

    assert type(meta) is type(bfm) is ForwardBackward
    assert "value/auxiliary/loss" not in meta_metrics
    assert "value/auxiliary/loss" in bfm_metrics
    assert "auxiliary" not in meta.value_optimizers
    assert "auxiliary" in bfm.value_optimizers


def _assert_nested_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    elif isinstance(expected, TensorDictBase):
        assert isinstance(actual, TensorDictBase)
        assert actual.batch_size == expected.batch_size
        assert actual.keys() == expected.keys()
        for key in expected.keys():  # noqa: SIM118
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_learner_checkpoint_reproduces_the_exact_next_update() -> None:
    """Model, optimizers, samplers, contexts, and RNG should resume one update exactly."""
    expected = _make_learner()
    restored = _make_learner()
    state = copy.deepcopy(expected.save())

    expected_metrics = expected.update()
    restored.load(state, load_cfg=None, strict=True)
    restored_metrics = restored.update()

    assert restored_metrics == expected_metrics
    assert restored.update_step == expected.update_step
    assert restored.versions == expected.versions
    torch.testing.assert_close(restored.context_buffer, expected.context_buffer, rtol=0.0, atol=0.0)
    for name, value in expected.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[name], value, rtol=0.0, atol=0.0)
    expected_state = expected.save()
    restored_state = restored.save()
    for name in (
        "optimizer_state_dicts",
        "reward_normalizer_state_dict",
        "replay_state_dict",
        "expert_state_dict",
        "context_buffer_cursor",
        "context_buffer_size",
        "update_step",
        "versions",
        "rng_state",
    ):
        _assert_nested_equal(restored_state[name], expected_state[name])


def test_source_matched_normalizer_uses_two_ordered_updates_in_full_sequence() -> None:
    """The complete update should mutate EMA statistics in current-then-next order."""
    learner = _make_learner(normalization_type="exponential")
    replay_rng = learner.replay.generator.get_state()
    batch = learner.replay.sample_random(learner.batch_size)
    learner.replay.generator.set_state(replay_rng)
    mean = torch.zeros(6)
    variance = torch.ones(6)
    for observations in (batch.observations["state"], batch.next_observations["state"]):
        mean = 0.99 * mean + 0.01 * observations.mean(dim=0)
        variance = 0.99 * variance + 0.01 * observations.var(dim=0)

    learner.update()

    normalizer = learner.model.observation_normalizers["state"]
    torch.testing.assert_close(normalizer.running_mean, mean)
    torch.testing.assert_close(normalizer.running_var, variance)
    assert normalizer.num_batches_tracked.item() == 2


def test_checkpoint_header_has_one_compatibility_fingerprint() -> None:
    """Checkpoint identity should stay small until concrete learner state exists."""


def test_discriminator_negative_uses_behavior_context_but_reward_uses_learner_context() -> None:
    """Relabeling must happen after the discriminator negative pair is consumed."""
    learner = _make_learner()
    learner.relabel_fraction = 1.0
    replay_rng = learner.replay.generator.get_state()
    expected_behavior = learner.replay.sample_random(learner.batch_size).behavior_context.clone()
    learner.replay.generator.set_state(replay_rng)
    captured_contexts = []

    def capture_context(_module: torch.nn.Module, inputs: tuple[torch.Tensor, torch.Tensor]) -> None:
        captured_contexts.append(inputs[1].detach().clone())

    assert learner.model.discriminator_network is not None
    handle = learner.model.discriminator_network.register_forward_pre_hook(capture_context)
    try:
        learner.update()
    finally:
        handle.remove()

    torch.testing.assert_close(captured_contexts[1], expected_behavior)
    assert not torch.equal(captured_contexts[-1], expected_behavior)


def test_compile_wraps_mutation_blocks_without_replacing_model_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compilation should target update blocks rather than wrap the state-owning model."""
    learner = _make_learner()
    model = learner.model
    compiled = []

    def fake_compile(function: Callable[..., object], *, mode: str) -> Callable[..., object]:
        compiled.append((function.__name__, mode))
        return function

    monkeypatch.setattr(torch, "compile", fake_compile)
    learner.compile("reduce-overhead")

    assert compiled == [
        ("_update_forward_backward", "reduce-overhead"),
        ("_update_actor", "reduce-overhead"),
    ]
    assert learner.get_policy() is model

    header = _make_header()

    assert header.format_name == FORWARD_BACKWARD_CHECKPOINT_FORMAT
    assert header.format_version == FORWARD_BACKWARD_CHECKPOINT_VERSION
    assert set(header.to_dict()) == {"format_name", "format_version", "schema_hash"}
    assert ForwardBackwardCheckpointHeader.from_dict(header.to_dict()) == header


def test_checkpoint_schema_hash_ignores_mapping_insertion_order() -> None:
    """Equivalent resolved schemas should have the same compatibility identity."""
    first_manifest = _make_manifest()
    second_manifest = dict(reversed(tuple(first_manifest.items())))
    second_manifest["config"] = {
        "model": {"context_width": 256},
        "algorithm": {"gamma": 0.99},
    }

    assert ForwardBackwardCheckpointHeader.from_manifest(
        first_manifest
    ) == ForwardBackwardCheckpointHeader.from_manifest(second_manifest)


def test_checkpoint_manifest_rejects_missing_and_unknown_fields() -> None:
    """The aggregate identity should never omit or silently accept one owner."""
    missing = _make_manifest()
    missing.pop("value_specs")
    with pytest.raises(ValueError, match=r"missing=.*value_specs"):
        ForwardBackwardCheckpointHeader.from_manifest(missing)

    unknown = _make_manifest()
    unknown["typo"] = True
    with pytest.raises(ValueError, match=r"unknown=.*typo"):
        ForwardBackwardCheckpointHeader.from_manifest(unknown)


def test_checkpoint_manifest_tracks_actual_schema_changes() -> None:
    """Every concrete schema and value specification should affect compatibility."""
    base_manifest = _make_manifest()
    base_header = ForwardBackwardCheckpointHeader.from_manifest(base_manifest)
    base_observation = make_meta_schema()
    base_reward = make_reward_schema()

    changed_observation = ForwardBackwardObservationSchema.from_config({"state": 359}, META_ROUTES)
    changed_reward_channels = list(base_reward.channels)
    changed_reward_channels[0] = replace(changed_reward_channels[0], timing="state")
    changed_reward = ForwardBackwardRewardSchema(channels=tuple(changed_reward_channels))
    changed_transition = _make_transition_schema(
        base_observation.schema_hash,
        base_reward.schema_hash,
        action_width=3,
    )
    changed_expert = replace(_make_expert_schema(), data_hash="data-v2")
    changed_value_spec = ForwardBackwardValueSpec(
        name="fb",
        kind="forward_readout",
        route="forward",
        reward_channels=("environment",),
        ensemble_size=3,
        has_target=True,
    )
    variants = (
        ("config", {"algorithm": {"gamma": 0.95}, "model": {"context_width": 256}}),
        ("observation_schema_hash", changed_observation.schema_hash),
        ("transition_schema_hash", changed_transition.schema_hash),
        ("reward_schema_hash", changed_reward.schema_hash),
        ("expert_schema_hash", changed_expert.schema_hash),
        ("value_specs", (_value_spec_data(changed_value_spec),)),
    )

    for name, value in variants:
        changed_manifest = dict(base_manifest)
        changed_manifest[name] = value
        assert ForwardBackwardCheckpointHeader.from_manifest(changed_manifest) != base_header, name


def test_checkpoint_validation_allows_runner_and_learner_state() -> None:
    """The header should validate compatibility without prescribing a deep manifest."""
    header = _make_header()
    checkpoint = {
        FORWARD_BACKWARD_CHECKPOINT_HEADER: header.to_dict(),
        "actor_state_dict": {},
        "optimizer_state_dict": {},
        "iter": 12,
        "infos": None,
    }

    header.validate_checkpoint(checkpoint)


def test_checkpoint_validation_rejects_an_incompatible_schema() -> None:
    """A learner should not interpret state produced for another schema."""
    expected = _make_header()
    loaded_manifest = _make_manifest()
    loaded_manifest["config"] = {"algorithm": {"gamma": 0.99}, "model": {"context_width": 128}}
    loaded = ForwardBackwardCheckpointHeader.from_manifest(loaded_manifest)

    with pytest.raises(ValueError, match="schema is incompatible"):
        expected.validate_checkpoint({FORWARD_BACKWARD_CHECKPOINT_HEADER: loaded.to_dict()})


def test_checkpoint_parser_rejects_unknown_format_version() -> None:
    """Format evolution should remain an explicit load-time decision."""
    data = _make_header().to_dict()
    data["format_version"] = FORWARD_BACKWARD_CHECKPOINT_VERSION + 1

    with pytest.raises(ValueError, match="format version"):
        ForwardBackwardCheckpointHeader.from_dict(data)


def test_checkpoint_parser_rejects_unknown_format_name() -> None:
    """Another checkpoint family should not be accepted based on hash alone."""
    data = _make_header().to_dict()
    data["format_name"] = "other.algorithm"

    with pytest.raises(ValueError, match="checkpoint format"):
        ForwardBackwardCheckpointHeader.from_dict(data)


def test_checkpoint_validation_requires_its_small_header() -> None:
    """A legacy checkpoint needs an explicit migration rather than guessed semantics."""
    with pytest.raises(ValueError, match=FORWARD_BACKWARD_CHECKPOINT_HEADER):
        _make_header().validate_checkpoint({"actor_state_dict": {}})


def test_phase_1g_publishes_only_explicit_forward_backward_boundaries() -> None:
    """The public API should expose explicit owners without legacy replay aliases."""
    assert rsl_rl.algorithms.__all__ == ["PPO", "Distillation", "ForwardBackward"]
    assert rsl_rl.models.__all__ == [
        "CNNModel",
        "ForwardBackwardInferenceModel",
        "ForwardBackwardModel",
        "MLPModel",
        "RNNModel",
    ]
    assert rsl_rl.runners.__all__ == [
        "DistillationRunner",
        "ForwardBackwardRunner",
        "OffPolicyRunner",
        "OnPolicyRunner",
    ]
    assert rsl_rl.storage.__all__ == ["ForwardBackwardExpertBuffer", "ForwardBackwardReplay", "RolloutStorage"]
    assert "SuccessorFeatures" in rsl_rl.extensions.__all__
    assert rsl_rl.algorithms.ForwardBackward is ForwardBackward
    assert rsl_rl.models.ForwardBackwardModel is ForwardBackwardModel


def test_successor_features_deprecation_points_to_the_unified_replacement() -> None:
    """The retained public prototype should emit concrete migration guidance."""
    with pytest.warns(DeprecationWarning, match="ForwardBackward.*OffPolicyRunner"):
        rsl_rl.extensions.SuccessorFeatures()


def test_new_rsl_modules_import_no_environment_or_reference_repository() -> None:
    """The reusable implementation boundary should depend only on RSL-RL and declared dependencies."""
    repository = Path(__file__).parents[2]
    files = (
        repository / "rsl_rl/algorithms/forward_backward.py",
        repository / "rsl_rl/models/forward_backward_model.py",
        repository / "rsl_rl/modules/reward_channels.py",
        repository / "rsl_rl/storage/forward_backward_replay.py",
        repository / "rsl_rl/storage/forward_backward_expert.py",
    )
    forbidden_roots = {
        "humanoidverse",
        "humenv",
        "isaaclab",
        "isaaclab_tasks",
        "metamotivo",
        "metamotivo_fb",
        "fbmzero",
        "fbm_zero",
    }

    for path in files:
        tree = ast.parse(path.read_text())
        imported_roots = {
            node.module.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        imported_roots.update(
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert imported_roots.isdisjoint(forbidden_roots), f"{path.name}: {imported_roots}"
        assert path.read_text().startswith("# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION")
