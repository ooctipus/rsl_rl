# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the forward-backward runner and checkpoint boundary."""

from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from pathlib import Path

import pytest

import rsl_rl.algorithms
import rsl_rl.extensions
import rsl_rl.models
import rsl_rl.storage
from rsl_rl.algorithms.forward_backward import (
    FORWARD_BACKWARD_CHECKPOINT_FORMAT,
    FORWARD_BACKWARD_CHECKPOINT_HEADER,
    FORWARD_BACKWARD_CHECKPOINT_VERSION,
    ForwardBackward,
    ForwardBackwardCheckpointHeader,
)
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models.forward_backward_model import ForwardBackwardObservationSchema
from rsl_rl.modules.reward_channels import ForwardBackwardRewardSchema, ForwardBackwardValueSpec
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertSchema
from rsl_rl.storage.forward_backward_replay import ForwardBackwardAutoresetMode, ForwardBackwardTransitionSchema
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


def test_algorithm_constructor_rejects_unknown_config_fields() -> None:
    """An algorithm typo should fail through Python's explicit constructor semantics."""
    with pytest.raises(TypeError, match="learnig_rate"):
        ForwardBackward(learnig_rate=1.0e-4)  # type: ignore[call-arg]


def test_multi_gpu_fails_until_synchronization_is_implemented() -> None:
    """The shell should not silently run unsynchronized distributed training."""
    with pytest.raises(NotImplementedError, match="multi-GPU"):
        ForwardBackward(multi_gpu_cfg={"world_size": 2})


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


def test_phase_1a_shell_does_not_pretend_to_implement_learning() -> None:
    """The non-abstract shell should remain direct while concrete work is deferred."""
    algorithm = ForwardBackward()

    assert not inspect.isabstract(ForwardBackward)
    with pytest.raises(NotImplementedError):
        algorithm.update()


def test_checkpoint_header_has_one_compatibility_fingerprint() -> None:
    """Checkpoint identity should stay small until concrete learner state exists."""
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


def test_public_exports_are_unchanged_during_phase_1a() -> None:
    """The new internals should not expand the baseline public API prematurely."""
    assert rsl_rl.algorithms.__all__ == ["PPO", "Distillation"]
    assert rsl_rl.models.__all__ == ["CNNModel", "MLPModel", "RNNModel"]
    assert rsl_rl.storage.__all__ == ["RolloutStorage"]
    assert "SuccessorFeatures" in rsl_rl.extensions.__all__
    assert not hasattr(rsl_rl.algorithms, "ForwardBackward")
    assert not hasattr(rsl_rl.models, "ForwardBackwardModel")


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
