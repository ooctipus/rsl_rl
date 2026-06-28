# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the generic off-policy runner and forward-backward collection adapter."""

from __future__ import annotations

import copy
import tempfile
import torch
from pathlib import Path
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models.forward_backward_model import ForwardBackwardObservationSchema
from rsl_rl.runners.off_policy_runner import OffPolicyRunner
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer, ForwardBackwardExpertSchema

NUM_ENVS = 4
STATE_DIM = 6
ACTION_DIM = 2


class ForwardBackwardDummyEnv(VecEnv):
    """Small deterministic same-step environment with optional true finals."""

    def __init__(self, provide_final: bool = True) -> None:
        """Initialize deterministic vector state and final-observation behavior."""
        self.num_envs = NUM_ENVS
        self.num_actions = ACTION_DIM
        self.max_episode_length = 3
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long)
        self.device = torch.device("cpu")
        self.cfg = {}
        self.provide_final = provide_final
        self.state = torch.zeros(NUM_ENVS, STATE_DIM)

    def get_observations(self) -> TensorDict:
        """Return the current emitted state."""
        return TensorDict({"state": self.state.clone()}, batch_size=[NUM_ENVS])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Advance every row and reset completed rows in the returned observation."""
        del actions
        reached = self.state + 1.0
        self.episode_length_buf += 1
        dones = self.episode_length_buf == self.max_episode_length
        self.state = reached.clone()
        self.state[dones] = 0.0
        self.episode_length_buf[dones] = 0
        extras: dict = {
            "time_outs": dones.clone(),
            "auxiliary_reward_evidence": reached[:, :1].square(),
            "episode_steps": self.episode_length_buf,
        }
        if self.provide_final:
            extras["final_obs"] = TensorDict({"state": reached}, batch_size=[NUM_ENVS])
            extras["final_obs_valid"] = dones.clone()
        rewards = reached[:, 0]
        return self.get_observations(), rewards, dones, extras

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return exact deterministic environment state for runner checkpoints."""
        return {
            "state": self.state.clone(),
            "episode_length_buf": self.episode_length_buf.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore exact deterministic environment state."""
        self.state.copy_(state["state"])
        self.episode_length_buf.copy_(state["episode_length_buf"])


def _expert_provider(
    env: VecEnv,
    observation_schema: ForwardBackwardObservationSchema,
    device: str,
    *,
    window_lengths: tuple[int, ...],
) -> ForwardBackwardExpertBuffer:
    """Return one deterministic two-clip corpus on the learner device."""
    del env
    frame_count = 32
    width = observation_schema.route_width("backward")
    frames = torch.arange(frame_count * width, device=device, dtype=torch.float32).reshape(frame_count, width) / 100
    offsets = torch.tensor([0, 16, 32], device=device)
    priorities = torch.ones(2, device=device)
    schema = ForwardBackwardExpertSchema(
        dataset_id="runner-fixture",
        data_hash="runner-data",
        feature_schema_hash=observation_schema.schema_hash,
        clip_offsets_hash="two-clips",
        expert_feature_width=width,
        num_frames=frame_count,
        num_clips=2,
        window_lengths=window_lengths,
    )
    return ForwardBackwardExpertBuffer(frames, offsets, priorities, schema, seed=17)


def _make_cfg(*, rollout_expert_fraction: float = 0.0, random_action_steps: int = 0) -> dict:
    """Return a tiny strict configuration using every Phase 1F section."""
    network = {"hidden_dim": 16, "hidden_layers": 1, "embedding_layers": 2}
    value_network = {"hidden_dim": 16, "hidden_layers": 1, "embedding_layers": 2}
    return {
        "num_steps_per_env": 2,
        "num_updates_per_iteration": 1,
        "random_action_steps": random_action_steps,
        "save_interval": 100,
        "obs_groups": {
            "actor": ["state"],
            "forward": ["state"],
            "backward": ["state"],
            "discriminator": ["state"],
            "critic_discriminator": ["state"],
            "critic_auxiliary": ["state"],
        },
        "model": {
            "class_name": "rsl_rl.models.forward_backward_model:ForwardBackwardModel",
            "context_dim": 4,
            "actor_cfg": network,
            "forward_cfg": network,
            "backward_hidden_dims": [16],
            "discriminator_hidden_dims": [16],
            "value_heads": [
                {
                    "spec": {
                        "name": "discriminator",
                        "kind": "critic",
                        "route": "critic_discriminator",
                        "reward_channels": ["discriminator"],
                        "ensemble_size": 2,
                        "has_target": True,
                    },
                    "network": value_network,
                },
                {
                    "spec": {
                        "name": "auxiliary",
                        "kind": "critic",
                        "route": "critic_auxiliary",
                        "reward_channels": ["effort"],
                        "ensemble_size": 2,
                        "has_target": True,
                    },
                    "network": value_network,
                },
            ],
        },
        "replay": {
            "class_name": "rsl_rl.storage.forward_backward_replay:ForwardBackwardReplay",
            "capacity_steps": 8,
            "terminal_capacity_per_env": 4,
            "autoreset_mode": "same_step",
            "environment_reward_name": "environment",
            "auxiliary_evidence_names": ["effort"],
            "reward_channels": [
                {
                    "name": "environment",
                    "provider_name": "environment",
                    "source": "environment",
                    "timing": "transition",
                    "context_dependent": False,
                    "sign": 1,
                },
                {
                    "name": "discriminator",
                    "provider_name": "discriminator",
                    "source": "recomputed",
                    "timing": "next_state",
                    "context_dependent": True,
                    "sign": 1,
                },
                {
                    "name": "effort",
                    "provider_name": "effort",
                    "source": "stored_evidence",
                    "timing": "transition",
                    "context_dependent": False,
                    "sign": -1,
                },
            ],
            "seed": 19,
        },
        "expert": {"provider": _expert_provider, "window_lengths": (2, 6)},
        "algorithm": {
            "class_name": "rsl_rl.algorithms.forward_backward:ForwardBackward",
            "batch_size": 8,
            "expert_sequence_length": 2,
            "context_buffer_capacity": 16,
            "discriminator_gradient_penalty_coefficient": 0.0,
            "rollout_context_refresh_steps": 2,
            "rollout_expert_fraction": rollout_expert_fraction,
            "rollout_expert_steps": 4,
            "rollout_expert_context_steps": 3,
            "value_cfg": {
                "discriminator": {"actor_coefficient": 0.05},
                "auxiliary": {
                    "actor_coefficient": 0.02,
                    "reward_coefficients": [0.1],
                    "normalize_rewards": True,
                },
            },
            "seed": 23,
        },
        "torch_compile_mode": None,
    }


def _collect(runner: OffPolicyRunner, steps: int) -> None:
    """Collect a fixed number of transitions without invoking the runner loop."""
    obs = runner.env.get_observations()
    for _ in range(steps):
        actions = runner.alg.act(obs)
        obs, rewards, dones, extras = runner.env.step(actions)
        runner.alg.process_env_step(obs, rewards, dones, extras)


def test_runner_constructs_collects_and_updates_through_public_lifecycle() -> None:
    """The generic runner should resolve the algorithm and mutate it after replay is ready."""
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(), _make_cfg(), log_dir=None, device="cpu")
    actor_before = copy.deepcopy(runner.alg.model.actor_network.state_dict())

    runner.learn(2)

    assert runner.alg.replay.total_steps == 4
    assert runner.alg.update_step == 2
    assert any(
        not torch.equal(actor_before[name], value)
        for name, value in runner.alg.model.actor_network.state_dict().items()
    )


def test_runner_uses_random_seed_phase_and_delays_updates_one_iteration() -> None:
    """Uniform source actions should precede actor behavior and the first update."""
    runner = OffPolicyRunner(
        ForwardBackwardDummyEnv(),
        _make_cfg(random_action_steps=2 * NUM_ENVS),
        log_dir=None,
        device="cpu",
    )
    random_calls = 0
    original = runner.alg.act_random

    def count_random_actions(obs: TensorDict) -> torch.Tensor:
        nonlocal random_calls
        random_calls += 1
        return original(obs)

    runner.alg.act_random = count_random_actions
    runner.learn(3)

    assert random_calls == 2
    assert runner.collected_transitions == 6 * NUM_ENVS
    assert runner.alg.update_step == 1


def test_runner_counts_completed_iterations_and_saves_each_boundary_once() -> None:
    """Checkpoint names and resume state should count completed iterations."""
    cfg = _make_cfg()
    cfg["save_interval"] = 2
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(), cfg, log_dir="/tmp/off_policy_runner", device="cpu")
    saved: list[tuple[str, int]] = []

    runner.logger.writer = object()
    runner.logger.init_logging_writer = lambda: None
    runner.logger.process_env_step = lambda *args, **kwargs: None
    runner.logger.log = lambda *args, **kwargs: None
    runner.logger.stop_logging_writer = lambda: None

    def record_save(path: str, infos: dict | None = None) -> None:
        del infos
        saved.append((Path(path).name, runner.current_learning_iteration))

    runner.save = record_save
    runner.learn(3)

    assert runner.current_learning_iteration == 3
    assert saved == [("model_2.pt", 2), ("model_3.pt", 3)]

    runner.learn(1)

    assert runner.current_learning_iteration == 4
    assert saved[-1] == ("model_4.pt", 4)


def test_checkpoint_hook_reset_refreshes_runner_and_replay_stream() -> None:
    """A save hook may reset the environment without leaving stale runner observations."""
    cfg = _make_cfg()
    cfg["save_interval"] = 1
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(), cfg, log_dir="/tmp/off_policy_runner", device="cpu")
    runner.logger.writer = object()
    runner.logger.init_logging_writer = lambda: None
    runner.logger.process_env_step = lambda *args, **kwargs: None
    runner.logger.log = lambda *args, **kwargs: None
    runner.logger.stop_logging_writer = lambda: None
    acted_observations: list[torch.Tensor] = []
    act = runner.alg.act

    def record_act(obs: TensorDict) -> torch.Tensor:
        acted_observations.append(obs["state"].clone())
        return act(obs)

    def reset_on_save(_path: str, _infos: dict | None = None) -> None:
        runner.env.state = torch.full((NUM_ENVS, STATE_DIM), 50.0)
        runner.env.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long)
        runner.alg.process_env_reset(
            runner.env.get_observations(),
            torch.ones(NUM_ENVS, dtype=torch.bool),
        )

    runner.alg.act = record_act
    runner.save = reset_on_save
    runner.learn(2)

    torch.testing.assert_close(acted_observations[2], torch.full((NUM_ENVS, STATE_DIM), 50.0))
    boundary = runner.alg.replay.sample(torch.ones(NUM_ENVS, dtype=torch.long), torch.arange(NUM_ENVS))
    assert torch.all(boundary.truncated)
    torch.testing.assert_close(boundary.next_observations["state"], torch.full((NUM_ENVS, STATE_DIM), 2.0))


def test_same_step_collection_uses_true_final_observation_when_available() -> None:
    """A done edge should reach the pre-reset final observation, never the reset observation."""
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(), _make_cfg(), log_dir=None, device="cpu")
    _collect(runner, 3)
    env_ids = torch.arange(NUM_ENVS)
    batch = runner.alg.replay.sample(torch.full((NUM_ENVS,), 2), env_ids)

    torch.testing.assert_close(batch.next_observations["state"], torch.full((NUM_ENVS, STATE_DIM), 3.0))
    assert not torch.any(batch.successor_uses_current)


def test_same_step_collection_falls_back_to_pre_step_when_final_is_unavailable() -> None:
    """Missing final_obs should use the marked pre-step approximation, not reset state."""
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(provide_final=False), _make_cfg(), log_dir=None, device="cpu")
    _collect(runner, 3)
    env_ids = torch.arange(NUM_ENVS)
    batch = runner.alg.replay.sample(torch.full((NUM_ENVS,), 2), env_ids)

    torch.testing.assert_close(batch.next_observations["state"], torch.full((NUM_ENVS, STATE_DIM), 2.0))
    assert torch.all(batch.successor_uses_current)


def test_rollout_refresh_samples_the_learned_context_mixture_per_episode() -> None:
    """Reached episode steps should refresh behavior from contexts produced by updates."""
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(), _make_cfg(), log_dir=None, device="cpu")
    learner = runner.alg
    learned_contexts = torch.arange(learner.context_buffer.shape[1], dtype=learner.context_buffer.dtype).repeat(
        NUM_ENVS, 1
    )
    learner.context_buffer[:NUM_ENVS].copy_(learned_contexts)
    learner.context_buffer_size = NUM_ENVS

    _collect(runner, 2)

    assert all(
        any(torch.equal(context, learned) for learned in learned_contexts) for context in learner.rollout_contexts
    )
    batch = learner.replay.sample(torch.ones(NUM_ENVS, dtype=torch.long), torch.arange(NUM_ENVS))
    assert torch.all(batch.context_changed)


def test_rolling_expert_schedule_changes_only_assigned_context_segments() -> None:
    """Half of the envs should advance rolling expert contexts on every applied edge."""
    runner = OffPolicyRunner(
        ForwardBackwardDummyEnv(), _make_cfg(rollout_expert_fraction=0.5), log_dir=None, device="cpu"
    )
    _collect(runner, 1)
    env_ids = torch.arange(NUM_ENVS)
    batch = runner.alg.replay.sample(torch.zeros(NUM_ENVS, dtype=torch.long), env_ids)

    assert batch.context_changed.sum() == NUM_ENVS // 2
    assert runner.alg._rollout_tracking_contexts.shape == (NUM_ENVS // 2, 4, 4)


def test_collection_schedule_is_learner_exact_across_checkpoint() -> None:
    """Rollout contexts, expert assignments, and the next update should restore exactly."""
    expected = OffPolicyRunner(
        ForwardBackwardDummyEnv(), _make_cfg(rollout_expert_fraction=0.5), log_dir=None, device="cpu"
    )
    restored = OffPolicyRunner(
        ForwardBackwardDummyEnv(), _make_cfg(rollout_expert_fraction=0.5), log_dir=None, device="cpu"
    )
    _collect(expected, 2)
    state = copy.deepcopy(expected.alg.save())
    restored.alg.load(state, load_cfg=None, strict=True)

    torch.testing.assert_close(restored.alg.rollout_contexts, expected.alg.rollout_contexts)
    torch.testing.assert_close(restored.alg._rollout_tracking_env_ids, expected.alg._rollout_tracking_env_ids)
    torch.testing.assert_close(restored.alg._rollout_tracking_contexts, expected.alg._rollout_tracking_contexts)
    assert restored.alg.rollout_schedule_step == expected.alg.rollout_schedule_step


def test_runner_checkpoint_restores_environment_and_iteration_exactly() -> None:
    """A restorable env should resume at the same collection boundary as its learner."""
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(), _make_cfg(), log_dir=None, device="cpu")
    _collect(runner, 2)
    runner.current_learning_iteration = 7
    expected_state = runner.env.state.clone()

    with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
        runner.save(checkpoint.name)
        runner.env.state.add_(10.0)
        runner.current_learning_iteration = 11
        runner.load(checkpoint.name)

    assert runner.environment_resume_exact
    assert runner.current_learning_iteration == 7
    assert runner.collected_transitions == 0
    torch.testing.assert_close(runner.env.state, expected_state)
