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
        "critic_discriminator": ("state",),
        "critic_auxiliary": ("state",),
    }
    reward_schema = make_reward_schema()
    states = [torch.randn(num_envs, state_width, generator=generator) + step for step in range(6)]
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
                    has_target=True,
                ),
                network,
            )
        )
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
        environment_reward_name="environment",
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
    for step in range(5):
        replay.add(
            ForwardBackwardTransitionBatch(
                observations=TensorDict({"state": states[step]}, batch_size=[num_envs]),
                next_observations=TensorDict({"state": states[step + 1]}, batch_size=[num_envs]),
                final_observations=TensorDict(
                    {"state": torch.full_like(states[step], float("nan"))}, batch_size=[num_envs]
                ),
                actions=torch.randn(num_envs, action_width, generator=generator),
                behavior_context=model.context_project(torch.randn(num_envs, context_width, generator=generator)),
                environment_reward=torch.randn(num_envs, 1, generator=generator),
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
    return ForwardBackward(
        model,
        replay,
        expert,
        ForwardBackwardCheckpointHeader.from_manifest(manifest),
        batch_size=8,
        expert_sequence_length=2,
        value_cfg=value_cfg,
        context_buffer_capacity=16,
        implied_value_coefficient=0.1,
        implied_reward_ridge=0.1,
        discriminator_gradient_penalty_coefficient=0.1,
        seed=47,
        multi_gpu_cfg=multi_gpu_cfg,
    )


def test_algorithm_constructor_rejects_unknown_config_fields() -> None:
    """An algorithm typo should fail through Python's explicit constructor semantics."""
    with pytest.raises(TypeError, match="learnig_rate"):
        ForwardBackward(learnig_rate=1.0e-4)  # type: ignore[call-arg]


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


def test_phase_1g_publishes_only_concrete_forward_backward_boundaries() -> None:
    """The public API should expose concrete owners without legacy replay aliases."""
    assert rsl_rl.algorithms.__all__ == ["PPO", "Distillation", "ForwardBackward"]
    assert rsl_rl.models.__all__ == ["CNNModel", "ForwardBackwardModel", "MLPModel", "RNNModel"]
    assert rsl_rl.runners.__all__ == ["DistillationRunner", "OffPolicyRunner", "OnPolicyRunner"]
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
