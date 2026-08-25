# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for reset-state curriculum learner signals."""

from __future__ import annotations

import copy
import math
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
        self.value_shift = torch.zeros(num_states)
        self.estimated_success_rate = torch.empty(num_states) if features is not None else None
        self.mean_success_target = torch.zeros(())
        self.success_target_grounded_fraction = torch.zeros(())
        feature_dim = features.shape[1] if features is not None else 0
        self.outcome_state_ids = torch.full((num_envs,), -1, dtype=torch.long)
        self.outcome_next_features = torch.zeros((num_envs, feature_dim))
        self.outcome_hard_targets = torch.zeros(num_envs)
        self.outcome_grounded = torch.zeros(num_envs, dtype=torch.bool)


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
    features = torch.tensor([[-3.0], [-2.0], [-1.0], [1.0], [2.0], [3.0]])[2 * rank : 2 * rank + 2]
    provider = _Provider(features, num_envs=3)
    if rank == 0:
        provider.outcome_state_ids.copy_(torch.tensor([0, 1, 0]))
        provider.outcome_hard_targets.fill_(1.0)
        provider.outcome_grounded.fill_(True)
    elif rank == 1:
        provider.outcome_state_ids[0] = 0
        provider.outcome_hard_targets[0] = 0.0
        provider.outcome_grounded[0] = True
    curriculum = StateCurriculum(
        _make_storage(num_envs=3, num_steps=1),
        "cpu",
        distributed=True,
        success_estimator_cfg={
            "hidden_dims": [8],
            "learning_rate": 1.0e-2,
        },
    )
    curriculum.bind(provider, torch.zeros(3, dtype=torch.long), _Critic(1.0))
    curriculum.collect_success_outcomes()
    loss = curriculum.update_success_estimator(2, 4)
    assert loss is not None
    checkpoint = curriculum.save()
    state = checkpoint["success_estimator_state_dict"]
    optimizer_state = next(iter(checkpoint["success_optimizer_state_dict"]["state"].values()))
    torch.save(
        {
            "state": state,
            "loss": torch.tensor(loss),
            "optimizer_steps": optimizer_state["step"],
            "mean_success_target": provider.mean_success_target,
            "grounded_fraction": provider.success_target_grounded_fraction,
        },
        Path(output_dir) / f"rank_{rank}.pt",
    )
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


def test_success_estimator_learns_rollout_outcomes_and_restores_checkpoint() -> None:
    """Train directly from rollout outcomes and restore model, optimizer, and normalization state."""
    torch.manual_seed(3)
    features = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    provider = _Provider(features, num_envs=4)
    cfg = {
        "hidden_dims": [16],
        "learning_rate": 2.0e-2,
    }
    curriculum = StateCurriculum(
        _make_storage(num_envs=4, num_steps=1),
        "cpu",
        distributed=False,
        success_estimator_cfg=cfg,
    )
    prediction_storage = provider.estimated_success_rate.data_ptr()
    curriculum.bind(provider, torch.zeros(4, dtype=torch.long), _Critic(1.0))

    torch.testing.assert_close(provider.estimated_success_rate, torch.full((4,), 0.5))
    provider.outcome_state_ids.copy_(torch.arange(4))
    provider.outcome_hard_targets.copy_(torch.tensor([0.0, 0.0, 1.0, 1.0]))
    provider.outcome_grounded.fill_(True)
    curriculum.collect_success_outcomes()
    assert curriculum.update_success_estimator(80, 1) is not None
    assert provider.estimated_success_rate[:2].max() < provider.estimated_success_rate[2:].min()
    assert provider.estimated_success_rate.data_ptr() == prediction_storage
    assert curriculum._success_state_ids is not None
    torch.testing.assert_close(curriculum._success_state_ids, torch.full((1, 4), -1, dtype=torch.long))

    checkpoint = copy.deepcopy(curriculum.save())
    restored_provider = _Provider(features.clone(), num_envs=4)
    restored = StateCurriculum(
        _make_storage(num_envs=4, num_steps=1),
        "cpu",
        distributed=False,
        success_estimator_cfg=cfg,
    )
    restored.bind(restored_provider, torch.zeros(4, dtype=torch.long), _Critic(1.0))
    restored.load({})
    torch.testing.assert_close(restored_provider.estimated_success_rate, torch.full((4,), 0.5))

    restored.load(checkpoint)
    torch.testing.assert_close(restored_provider.estimated_success_rate, provider.estimated_success_rate)
    for outcome_provider in (provider, restored_provider):
        outcome_provider.outcome_state_ids.copy_(torch.arange(4))
        outcome_provider.outcome_hard_targets.copy_(torch.tensor([0.0, 0.0, 1.0, 1.0]))
        outcome_provider.outcome_grounded.fill_(True)
    curriculum.collect_success_outcomes()
    restored.collect_success_outcomes()
    torch.manual_seed(11)
    curriculum.update_success_estimator(80, 1)
    torch.manual_seed(11)
    restored.update_success_estimator(80, 1)
    torch.testing.assert_close(restored_provider.estimated_success_rate, provider.estimated_success_rate)


def test_success_estimator_uses_the_ppo_update_schedule() -> None:
    """Visit every rollout outcome once per PPO epoch across its minibatches."""
    num_outcomes = 7
    num_learning_epochs = 3
    num_mini_batches = 4
    features = torch.arange(num_outcomes, dtype=torch.float32).unsqueeze(1)
    provider = _Provider(features, num_envs=num_outcomes)
    curriculum = StateCurriculum(
        _make_storage(num_envs=num_outcomes, num_steps=1),
        "cpu",
        distributed=False,
        success_estimator_cfg={"hidden_dims": [4], "learning_rate": 0.0},
    )
    curriculum.bind(provider, torch.zeros(num_outcomes, dtype=torch.long), _Critic(1.0))
    provider.outcome_state_ids.copy_(torch.arange(num_outcomes))
    provider.outcome_hard_targets.copy_(torch.arange(num_outcomes).remainder(2).float())
    provider.outcome_grounded.fill_(True)
    curriculum.collect_success_outcomes()

    training_batches: list[torch.Tensor] = []

    def record_training_batch(_module: nn.Module, inputs: tuple[torch.Tensor]) -> None:
        if torch.is_grad_enabled():
            training_batches.append(inputs[0][:, 0].to(dtype=torch.long))

    assert curriculum._success_estimator is not None
    handle = curriculum._success_estimator.register_forward_pre_hook(record_training_batch)
    loss = curriculum.update_success_estimator(num_learning_epochs, num_mini_batches)
    handle.remove()

    assert loss == pytest.approx(math.log(2.0))
    assert [len(batch) for batch in training_batches] == [2, 2, 2, 1] * num_learning_epochs
    counts = torch.bincount(torch.cat(training_batches), minlength=num_outcomes)
    torch.testing.assert_close(counts, torch.full((num_outcomes,), num_learning_epochs, dtype=torch.long))
    optimizer_state = next(iter(curriculum.save()["success_optimizer_state_dict"]["state"].values()))
    assert optimizer_state["step"] == num_learning_epochs * num_mini_batches


@pytest.mark.parametrize("option", ["num_batches", "batch_size", "evaluation_batch_size"])
def test_success_estimator_rejects_an_independent_batch_schedule(option: str) -> None:
    """Keep PPO as the only owner of learning epochs and minibatches."""
    provider = _Provider(torch.zeros(1, 1), num_envs=1)
    curriculum = StateCurriculum(
        _make_storage(num_envs=1, num_steps=1),
        "cpu",
        distributed=False,
        success_estimator_cfg={"hidden_dims": [4], option: 1},
    )

    with pytest.raises(TypeError, match=option):
        curriculum.bind(provider, torch.zeros(1, dtype=torch.long), _Critic(1.0))


def test_success_estimate_is_the_raw_model_prediction() -> None:
    """Write model probabilities directly without blending per-row outcome history."""
    provider = _Provider(torch.tensor([[-1.0], [1.0]]), num_envs=1)
    curriculum = StateCurriculum(
        _make_storage(num_envs=1, num_steps=1),
        "cpu",
        distributed=False,
        success_estimator_cfg={"hidden_dims": [4]},
    )
    curriculum.bind(provider, torch.zeros(1, dtype=torch.long), _Critic(1.0))
    torch.testing.assert_close(provider.estimated_success_rate, torch.full((2,), 0.5))

    estimator = curriculum._success_estimator
    assert estimator is not None
    output = next(module for module in reversed(estimator.mlp) if isinstance(module, nn.Linear))
    with torch.no_grad():
        output.weight.zero_()
        output.bias.fill_(torch.logit(torch.tensor(0.8)))
    curriculum.load(copy.deepcopy(curriculum.save()))
    torch.testing.assert_close(provider.estimated_success_rate, torch.full((2,), 0.8))


def test_success_outcomes_mix_ground_truth_with_detached_timeout_estimates() -> None:
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

    class _EndpointEstimator(nn.Module):
        def forward(self, features: torch.Tensor) -> torch.Tensor:
            return features[:, 0]

    curriculum._success_estimator = _EndpointEstimator()

    curriculum.collect_success_outcomes()
    curriculum.collect_success_outcomes()

    assert curriculum._success_state_ids is not None
    assert curriculum._success_targets is not None and curriculum._success_grounded is not None
    torch.testing.assert_close(curriculum._success_state_ids[0], torch.tensor([1, 0, 1, -1]))
    expected = torch.tensor([1.0, 0.0, torch.sigmoid(torch.tensor(2.0))])
    torch.testing.assert_close(curriculum._success_targets[0, :3], expected)
    torch.testing.assert_close(curriculum._success_grounded[0, :3], torch.tensor([True, True, False]))
    assert not curriculum._success_targets.requires_grad
    torch.testing.assert_close(provider.outcome_state_ids, torch.full((4,), -1, dtype=torch.long))


def test_success_outcomes_discard_invalid_targets_and_release_all_slots() -> None:
    """Discard invalid outcomes, retain grounded results, and release every pending slot."""
    provider = _Provider(torch.tensor([[-1.0], [1.0]]), num_envs=3)
    provider.outcome_state_ids.copy_(torch.tensor([1, 0, 1]))
    provider.outcome_next_features.fill_(torch.nan)
    provider.outcome_hard_targets.copy_(torch.tensor([0.0, torch.nan, 1.0]))
    provider.outcome_grounded.copy_(torch.tensor([False, True, True]))
    curriculum = StateCurriculum(
        _make_storage(num_envs=3, num_steps=1),
        "cpu",
        distributed=False,
        success_estimator_cfg={"hidden_dims": [4]},
    )
    curriculum.bind(provider, torch.zeros(3, dtype=torch.long), _Critic(1.0))

    curriculum.collect_success_outcomes()

    assert curriculum._success_state_ids is not None
    torch.testing.assert_close(curriculum._success_state_ids[0], torch.tensor([-1, -1, 1]))
    torch.testing.assert_close(provider.outcome_state_ids, torch.full((3,), -1, dtype=torch.long))
    assert curriculum.update_success_estimator(1, 1) is not None
    assert provider.mean_success_target == 1.0
    assert provider.success_target_grounded_fraction == 1.0


def test_success_metrics_are_episode_weighted_and_rollout_scoped() -> None:
    """Count duplicate rows as episodes and forget their targets after each update."""
    provider = _Provider(torch.tensor([[-1.0], [1.0]]), num_envs=2)
    storage = _make_storage(num_envs=2, num_steps=2)
    curriculum = StateCurriculum(
        storage,
        "cpu",
        distributed=False,
        success_estimator_cfg={"hidden_dims": [4]},
    )
    curriculum.bind(provider, torch.zeros(2, dtype=torch.long), _Critic(1.0))

    provider.outcome_state_ids.copy_(torch.tensor([0, 0]))
    provider.outcome_hard_targets.fill_(1.0)
    provider.outcome_grounded.copy_(torch.tensor([True, False]))
    provider.outcome_next_features.zero_()
    curriculum.collect_success_outcomes()
    curriculum.collect_success_outcomes()
    storage.step = 1
    provider.outcome_state_ids.copy_(torch.tensor([1, -1]))
    provider.outcome_hard_targets[0] = 0.0
    provider.outcome_grounded[0] = True
    curriculum.collect_success_outcomes()

    assert curriculum.update_success_estimator(1, 1) is not None
    torch.testing.assert_close(provider.mean_success_target, torch.tensor(0.5))
    torch.testing.assert_close(provider.success_target_grounded_fraction, torch.tensor(2.0 / 3.0))
    assert curriculum.update_success_estimator(1, 1) is None
    assert provider.mean_success_target.isnan()
    assert provider.success_target_grounded_fraction.isnan()

    storage.step = 0
    provider.outcome_state_ids.copy_(torch.tensor([1, -1]))
    provider.outcome_hard_targets[0] = 0.0
    curriculum.collect_success_outcomes()
    assert curriculum.update_success_estimator(1, 1) is not None
    torch.testing.assert_close(provider.mean_success_target, torch.tensor(0.0))


def test_distributed_estimator_reduces_uneven_outcome_counts(tmp_path: Path) -> None:
    """Synchronize an empty rank and compute metrics from uneven global outcome totals."""
    init_file = tmp_path / "process_group"
    mp.spawn(_distributed_estimator_worker, args=(3, str(init_file), str(tmp_path)), nprocs=3, join=True)
    ranks = [torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True) for rank in range(3)]
    rank_0 = ranks[0]

    for rank in ranks[1:]:
        assert rank_0["state"].keys() == rank["state"].keys()
        for name in rank_0["state"]:
            torch.testing.assert_close(rank_0["state"][name], rank["state"][name])
        torch.testing.assert_close(rank_0["loss"], rank["loss"])
    torch.testing.assert_close(rank_0["state"]["normalizer._mean"], torch.zeros(1, 1))
    for rank in ranks:
        assert rank["optimizer_steps"] == 8
        torch.testing.assert_close(rank["mean_success_target"], torch.tensor(0.75))
        torch.testing.assert_close(rank["grounded_fraction"], torch.tensor(1.0))
