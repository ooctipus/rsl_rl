# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for reset-state curriculum learner signals."""

from __future__ import annotations

import copy
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from pathlib import Path
from tensordict import TensorDict

import pytest

from rsl_rl.extensions import StateCurriculum
from rsl_rl.storage import RolloutStorage


class _Provider:
    def __init__(self, features: torch.Tensor | None, num_envs: int, num_states: int | None = None) -> None:
        num_states = len(features) if features is not None else num_states
        assert num_states is not None
        self.sampled_state = torch.zeros(num_envs, dtype=torch.long)
        self.state_features = features
        self.success_rate = torch.zeros(num_states)
        self.success_size = torch.zeros(num_states, dtype=torch.long)
        self.value_shift = torch.zeros(num_states)
        self.estimated_success_rate = torch.empty(num_states)
        feature_dim = features.shape[1] if features is not None else 0
        self.outcome_state_ids = torch.full((num_envs,), -1, dtype=torch.long)
        self.outcome_next_features = torch.empty((num_envs, feature_dim))
        self.outcome_hard_targets = torch.zeros(num_envs)
        self.outcome_grounded = torch.zeros(num_envs, dtype=torch.bool)
        self.recorded_state_ids = torch.empty(0, dtype=torch.long)
        self.recorded_targets = torch.empty(0)
        self.recorded_valid = torch.empty(0, dtype=torch.bool)

    def record_success_targets(self, env_ids: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> None:
        self.recorded_state_ids = self.outcome_state_ids[env_ids].clone()
        self.recorded_targets = targets.clone()
        self.recorded_valid = valid.clone()
        self.outcome_state_ids[env_ids] = -1


class _Critic(nn.Module):
    is_recurrent = False

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, obs: TensorDict) -> torch.Tensor:
        return obs["policy"][:, :1] * self.scale


def _make_storage(num_envs: int = 2, num_steps: int = 2) -> RolloutStorage:
    obs = TensorDict({"policy": torch.zeros(num_envs, 1)}, batch_size=[num_envs])
    return RolloutStorage("rl", num_envs, num_steps, obs, [1])


def _distributed_estimator_worker(rank: int, world_size: int, init_file: str, output_dir: str) -> None:
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    torch.manual_seed(7)
    features = torch.tensor([[-2.0], [-1.0]]) if rank == 0 else torch.tensor([[1.0], [2.0]])
    provider = _Provider(features, num_envs=1)
    provider.success_rate.copy_(torch.tensor([0.0, 0.25]) if rank == 0 else torch.tensor([0.75, 1.0]))
    if rank == 0:
        provider.success_size.fill_(8)
    curriculum = StateCurriculum(
        _make_storage(num_envs=1, num_steps=1),
        "cpu",
        distributed=True,
        success_estimator_cfg={
            "hidden_dims": [8],
            "learning_rate": 1.0e-2,
            "num_batches": 2,
            "batch_size": 8,
            "evaluation_batch_size": 2,
        },
    )
    curriculum.bind(provider, torch.zeros(1, dtype=torch.long), _Critic(1.0))
    loss = curriculum.update_success_estimator()
    assert loss is not None
    state = curriculum.save()["success_estimator_state_dict"]
    torch.save({"state": state, "loss": torch.tensor(loss)}, Path(output_dir) / f"rank_{rank}.pt")
    torch.distributed.destroy_process_group()


def test_value_shift_uses_episode_starts_from_the_same_update() -> None:
    """Average duplicate start states and clear stale unvisited scores."""
    storage = _make_storage()
    provider = _Provider(None, num_envs=2, num_states=4)
    provider.value_shift.fill_(9.0)
    episode_length = torch.tensor([0, 1])
    curriculum = StateCurriculum(
        storage,
        "cpu",
        distributed=False,
        value_shift_cfg={"evaluation_batch_size": 1},
    )
    curriculum.bind(provider, episode_length, _Critic(1.0))

    first_obs = TensorDict({"policy": torch.tensor([[1.0], [2.0]])}, batch_size=[2])
    storage.observations[0].copy_(first_obs)
    provider.sampled_state.copy_(torch.tensor([1, 2]))
    curriculum.record_episode_starts(0, _Critic(1.0)(first_obs))

    second_obs = TensorDict({"policy": torch.tensor([[3.0], [4.0]])}, batch_size=[2])
    storage.observations[1].copy_(second_obs)
    provider.sampled_state.copy_(torch.tensor([1, 3]))
    episode_length.zero_()
    curriculum.record_episode_starts(1, _Critic(1.0)(second_obs))

    curriculum.update_value_shift(_Critic(2.0))

    torch.testing.assert_close(provider.value_shift, torch.tensor([0.0, 2.0, 0.0, 4.0]))


def test_enabled_curriculum_requires_an_environment_provider() -> None:
    """Fail at construction rather than silently running an unbound curriculum."""
    curriculum = StateCurriculum(_make_storage(), "cpu", distributed=False, value_shift_cfg={})

    with pytest.raises(ValueError, match="returned no state curriculum"):
        curriculum.bind(None, torch.zeros(2, dtype=torch.long), _Critic(1.0))


def test_success_estimator_learns_empirical_rates_and_restores_checkpoint() -> None:
    """Train from full-episode bank statistics and restore model, optimizer, and normalization state."""
    torch.manual_seed(3)
    features = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    provider = _Provider(features, num_envs=1)
    cfg = {
        "hidden_dims": [16],
        "learning_rate": 2.0e-2,
        "num_batches": 80,
        "batch_size": 32,
        "evaluation_batch_size": 2,
    }
    curriculum = StateCurriculum(
        _make_storage(num_envs=1, num_steps=1),
        "cpu",
        distributed=False,
        success_estimator_cfg=cfg,
    )
    prediction_storage = provider.estimated_success_rate.data_ptr()
    curriculum.bind(provider, torch.zeros(1, dtype=torch.long), _Critic(1.0))

    torch.testing.assert_close(provider.estimated_success_rate, torch.full((4,), 0.5))
    provider.success_rate.copy_(torch.tensor([0.0, 0.0, 1.0, 1.0]))
    provider.success_size.fill_(20)
    assert curriculum.update_success_estimator() is not None
    assert provider.estimated_success_rate[:2].max() < provider.estimated_success_rate[2:].min()
    assert provider.estimated_success_rate.data_ptr() == prediction_storage

    checkpoint = copy.deepcopy(curriculum.save())
    restored_provider = _Provider(features.clone(), num_envs=1)
    restored = StateCurriculum(
        _make_storage(num_envs=1, num_steps=1),
        "cpu",
        distributed=False,
        success_estimator_cfg=cfg,
    )
    restored.bind(restored_provider, torch.zeros(1, dtype=torch.long), _Critic(1.0))
    restored.load({})
    torch.testing.assert_close(restored_provider.estimated_success_rate, torch.full((4,), 0.5))

    restored_provider.success_rate.copy_(provider.success_rate)
    restored_provider.success_size.copy_(provider.success_size)
    restored.load(checkpoint)
    torch.testing.assert_close(restored_provider.estimated_success_rate, provider.estimated_success_rate)
    torch.manual_seed(11)
    curriculum.update_success_estimator()
    torch.manual_seed(11)
    restored.update_success_estimator()
    torch.testing.assert_close(restored_provider.estimated_success_rate, provider.estimated_success_rate)


def test_success_estimate_blends_model_prior_with_recorded_targets() -> None:
    """Use the model for unseen rows and let recorded targets dominate as evidence grows."""
    provider = _Provider(torch.tensor([[-1.0], [1.0]]), num_envs=1)
    curriculum = StateCurriculum(
        _make_storage(num_envs=1, num_steps=1),
        "cpu",
        distributed=False,
        success_estimator_cfg={"hidden_dims": [4], "prior_count": 1.0},
    )
    curriculum.bind(provider, torch.zeros(1, dtype=torch.long), _Critic(1.0))
    torch.testing.assert_close(provider.estimated_success_rate, torch.full((2,), 0.5))

    provider.success_rate.copy_(torch.tensor([0.0, 1.0]))
    provider.success_size.copy_(torch.tensor([1, 50]))
    curriculum.load(copy.deepcopy(curriculum.save()))
    torch.testing.assert_close(provider.estimated_success_rate, torch.tensor([0.25, 50.5 / 51.0]))


def test_success_targets_mix_ground_truth_with_detached_timeout_estimates() -> None:
    """Keep semantic outcomes exact and bootstrap only pure fixed-horizon timeouts."""
    provider = _Provider(torch.tensor([[-1.0], [1.0]]), num_envs=4)
    provider.outcome_state_ids.copy_(torch.tensor([1, 0, 1, -1]))
    provider.outcome_next_features.copy_(torch.tensor([[-2.0], [-1.0], [2.0], [0.0]]))
    provider.outcome_hard_targets.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    provider.outcome_grounded.copy_(torch.tensor([True, True, False, False]))
    curriculum = StateCurriculum(
        _make_storage(num_envs=4, num_steps=1),
        "cpu",
        distributed=False,
        success_estimator_cfg={"hidden_dims": [4]},
    )
    curriculum.bind(provider, torch.zeros(4, dtype=torch.long), _Critic(1.0))

    curriculum.update_success_targets()

    torch.testing.assert_close(provider.recorded_state_ids, torch.tensor([1, 0, 1]))
    torch.testing.assert_close(provider.recorded_targets, torch.tensor([1.0, 0.0, 0.5]))
    torch.testing.assert_close(provider.recorded_valid, torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(provider.outcome_state_ids, torch.full((4,), -1, dtype=torch.long))


def test_success_targets_discard_non_finite_timeout_endpoints() -> None:
    """Release a corrupt timeout without feeding it into the reset monitor."""
    provider = _Provider(torch.tensor([[-1.0], [1.0]]), num_envs=1)
    provider.outcome_state_ids[0] = 1
    provider.outcome_next_features[0, 0] = torch.nan
    curriculum = StateCurriculum(
        _make_storage(num_envs=1, num_steps=1),
        "cpu",
        distributed=False,
        success_estimator_cfg={"hidden_dims": [4]},
    )
    curriculum.bind(provider, torch.zeros(1, dtype=torch.long), _Critic(1.0))

    curriculum.update_success_targets()

    torch.testing.assert_close(provider.recorded_valid, torch.tensor([False]))
    torch.testing.assert_close(provider.outcome_state_ids, torch.tensor([-1]))


def test_distributed_estimator_parameters_and_normalization_match(tmp_path: Path) -> None:
    """Match a local reference when one distributed rank has no completed outcomes."""
    init_file = tmp_path / "process_group"
    mp.spawn(_distributed_estimator_worker, args=(2, str(init_file), str(tmp_path)), nprocs=2, join=True)
    rank_0 = torch.load(tmp_path / "rank_0.pt", weights_only=True)
    rank_1 = torch.load(tmp_path / "rank_1.pt", weights_only=True)

    assert rank_0["state"].keys() == rank_1["state"].keys()
    for name in rank_0["state"]:
        torch.testing.assert_close(rank_0["state"][name], rank_1["state"][name])
    torch.testing.assert_close(rank_0["loss"], rank_1["loss"])
    torch.testing.assert_close(rank_0["state"]["normalizer._mean"], torch.zeros(1, 1))

    torch.manual_seed(7)
    reference_provider = _Provider(torch.tensor([[-2.0], [-1.0], [1.0], [2.0]]), num_envs=1)
    reference_provider.success_rate.copy_(torch.tensor([0.0, 0.25, 0.75, 1.0]))
    reference_provider.success_size.copy_(torch.tensor([8, 8, 0, 0]))
    reference = StateCurriculum(
        _make_storage(num_envs=1, num_steps=1),
        "cpu",
        distributed=False,
        success_estimator_cfg={
            "hidden_dims": [8],
            "learning_rate": 1.0e-2,
            "num_batches": 2,
            "batch_size": 8,
            "evaluation_batch_size": 4,
        },
    )
    reference.bind(reference_provider, torch.zeros(1, dtype=torch.long), _Critic(1.0))
    reference_loss = reference.update_success_estimator()
    assert reference_loss is not None
    reference_state = reference.save()["success_estimator_state_dict"]

    for name in rank_0["state"]:
        torch.testing.assert_close(rank_0["state"][name], reference_state[name])
    torch.testing.assert_close(rank_0["loss"], torch.tensor(reference_loss))
