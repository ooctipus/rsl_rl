# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for successor goal-library binding."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from types import SimpleNamespace

import pytest

from rsl_rl.algorithms import PPO
from rsl_rl.extensions import SuccessorFeatures
from rsl_rl.storage import RolloutStorage


def test_successor_binds_goal_observations_and_indices_from_expressions() -> None:
    """Explicit expressions should replace command-specific cache discovery."""
    cache = TensorDict({"critic": torch.arange(6, dtype=torch.float32).reshape(3, 2)}, batch_size=[3])
    indices = torch.tensor([2, 0], dtype=torch.long)
    env = SimpleNamespace(num_envs=2, unwrapped=SimpleNamespace(goal_cache=cache, goal_indices=indices))

    with pytest.warns(DeprecationWarning, match="SuccessorFeatures is deprecated"):
        successor = SuccessorFeatures(
            goal_observation_bind="env.unwrapped.goal_cache",
            goal_indices_bind="env.unwrapped.goal_indices",
        )
    successor.bind(env)

    torch.testing.assert_close(successor.goal_cache["critic"], cache["critic"])
    assert successor.cmd_indices_fn is not None
    assert successor.cmd_indices_fn().data_ptr() == indices.data_ptr()


@pytest.mark.parametrize(
    ("observation_bind", "indices_bind"),
    [("env.unwrapped.goal_cache", None), (None, "env.unwrapped.goal_indices")],
)
def test_successor_rejects_partial_goal_bindings(observation_bind: str | None, indices_bind: str | None) -> None:
    """A half-configured goal library must fail at the learner boundary."""
    env = SimpleNamespace(num_envs=2, unwrapped=SimpleNamespace())
    with pytest.warns(DeprecationWarning, match="SuccessorFeatures is deprecated"):
        successor = SuccessorFeatures(
            goal_observation_bind=observation_bind,
            goal_indices_bind=indices_bind,
        )

    with pytest.raises(ValueError, match="goal_observation_bind and goal_indices_bind"):
        successor.bind(env)


def test_successor_keeps_deprecated_command_lookup_boundary() -> None:
    """Existing environments retain one release boundary for command-owned goal caches."""
    cache = TensorDict({"critic": torch.zeros(3, 2)}, batch_size=[3])
    indices = torch.tensor([1, 2], dtype=torch.long)
    term = SimpleNamespace(get_target_obs_cache=lambda: cache, cmd_indices=indices)
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _name: term)),
    )
    with pytest.warns(DeprecationWarning, match="SuccessorFeatures is deprecated"):
        successor = SuccessorFeatures(goal_command_name="goal_point")

    with pytest.warns(DeprecationWarning, match="goal_command_name lookup is deprecated"):
        successor.bind(env)

    assert successor.cmd_indices_fn is not None
    assert successor.cmd_indices_fn().data_ptr() == indices.data_ptr()


def test_successor_preserves_existing_positional_constructor_order() -> None:
    """New expression bindings must not shift deprecated public positional parameters."""
    with pytest.warns(DeprecationWarning, match="SuccessorFeatures is deprecated"):
        successor = SuccessorFeatures(64, 0.95, "vector_td", 2.0, 0.25, "task", 321, 0.02, "cpu")

    assert successor.goal_command_name == "task"
    assert successor.fb_batch_size == 321
    assert successor.target_tau == 0.02
    assert successor.goal_observation_bind is None
    assert successor.goal_indices_bind is None


@pytest.mark.parametrize(
    ("cache", "indices", "error_type", "match"),
    [
        ({"critic": torch.zeros(3, 2)}, torch.zeros(2, dtype=torch.long), TypeError, "TensorDict"),
        (TensorDict({"critic": torch.zeros(3, 2)}, batch_size=[3]), torch.zeros(2), TypeError, "torch.long"),
        (
            TensorDict({"critic": torch.zeros(3, 2)}, batch_size=[3]),
            torch.zeros(3, dtype=torch.long),
            ValueError,
            "one index per environment",
        ),
    ],
)
def test_successor_rejects_malformed_expression_results(
    cache: object,
    indices: torch.Tensor,
    error_type: type[Exception],
    match: str,
) -> None:
    """Expression results should fail once at the learner boundary."""
    env = SimpleNamespace(num_envs=2, unwrapped=SimpleNamespace(goal_cache=cache, goal_indices=indices))
    with pytest.warns(DeprecationWarning, match="SuccessorFeatures is deprecated"):
        successor = SuccessorFeatures(
            goal_observation_bind="env.unwrapped.goal_cache",
            goal_indices_bind="env.unwrapped.goal_indices",
        )

    with pytest.raises(error_type, match=match):
        successor.bind(env)


class _Actor:
    def __init__(self) -> None:
        self.output_distribution_params = (torch.zeros(2, 1),)

    def get_hidden_state(self) -> None:
        return None

    def __call__(self, obs: TensorDict, z: torch.Tensor, stochastic_output: bool) -> torch.Tensor:
        del obs, z, stochastic_output
        return torch.zeros(2, 1)

    def get_output_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.zeros(actions.shape[0])


class _Critic:
    def get_hidden_state(self) -> None:
        return None


class _Successor:
    def __init__(self, indices: torch.Tensor) -> None:
        self.cmd_indices_fn = lambda: indices

    def goal_z(self, critic: object, indices: torch.Tensor) -> torch.Tensor:
        del critic
        return indices.unsqueeze(-1).float()

    def state_value(self, critic: object, obs: TensorDict, z: torch.Tensor) -> torch.Tensor:
        del critic, obs
        return z


def test_ppo_act_snapshots_successor_goal_indices() -> None:
    """An in-place command reset after acting must not rewrite the behavior goal."""
    indices = torch.tensor([0, 2], dtype=torch.long)
    ppo = PPO.__new__(PPO)
    ppo.actor = _Actor()
    ppo.critic = _Critic()
    ppo.successor = _Successor(indices)
    ppo.transition = RolloutStorage.Transition()

    PPO.act(ppo, TensorDict({"policy": torch.zeros(2, 1)}, batch_size=[2]))
    indices.copy_(torch.tensor([1, 1]))

    torch.testing.assert_close(ppo.transition.command_indices, torch.tensor([0, 2]))
