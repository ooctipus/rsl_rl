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

import pytest

from rsl_rl.env import VecEnv
from rsl_rl.models.forward_backward_model import ForwardBackwardModel, ForwardBackwardObservationSchema
from rsl_rl.runners.lifecycle import RunnerLifecycleExtension
from rsl_rl.runners.off_policy_runner import OffPolicyRunner
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer, ForwardBackwardExpertSchema
from rsl_rl.storage.forward_backward_replay import ForwardBackwardReplay

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
        self.last_actions = torch.zeros(NUM_ENVS, ACTION_DIM)

    def get_observations(self) -> TensorDict:
        """Return the current emitted state."""
        return TensorDict(
            {
                "state": self.state.clone(),
                "transition": self.state[:, :1].square(),
            },
            batch_size=[NUM_ENVS],
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Advance every row and reset completed rows in the returned observation."""
        self.last_actions = actions
        reached = self.state + 1.0
        self.episode_length_buf += 1
        dones = self.episode_length_buf == self.max_episode_length
        self.state = reached.clone()
        self.state[dones] = 0.0
        self.episode_length_buf[dones] = 0
        extras: dict = {"time_outs": dones.clone()}
        if self.provide_final:
            extras["final_obs"] = TensorDict(
                {
                    "state": reached,
                    "transition": reached[:, :1].square(),
                },
                batch_size=[NUM_ENVS],
            )
            extras["final_obs_valid"] = dones.clone()
        rewards = reached[:, 0]
        observations = self.get_observations()
        return observations, rewards, dones, extras

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


class NamedEvidenceDummyEnv(ForwardBackwardDummyEnv):
    """Expose two named scalar evidence fields in non-schema insertion order."""

    def get_observations(self) -> TensorDict:
        """Return mixed-rank evidence fields with impact inserted before effort."""
        observations = super().get_observations()
        effort = observations["transition"][:, 0]
        impact = effort.unsqueeze(-1) + 100.0
        observations.set(
            "transition",
            TensorDict({"impact": impact, "effort": effort}, batch_size=[NUM_ENVS]),
        )
        return observations

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Replace flat final evidence with the same named mixed-rank layout."""
        observations, rewards, dones, extras = super().step(actions)
        final = extras.get("final_obs")
        if final is not None:
            effort = final["transition"][:, 0]
            impact = effort.unsqueeze(-1) + 100.0
            final.set(
                "transition",
                TensorDict({"impact": impact, "effort": effort}, batch_size=[NUM_ENVS]),
            )
        return observations, rewards, dones, extras


class RecordingLifecycleExtension(RunnerLifecycleExtension):
    """Record exact events and return deterministic reset observations."""

    def __init__(
        self,
        env: VecEnv,
        algorithm: object,
        log_dir: str | None,
        device: str,
    ) -> None:
        """Initialize the base resources and empty event history."""
        super().__init__(env, algorithm, log_dir, device)
        self.transitions: list[int] = []

    def on_transition(self, transition: int) -> TensorDict:
        """Record one event, reset the fixture, and return its observations."""
        self.transitions.append(transition)
        self.env.state.fill_(float(transition))
        self.env.episode_length_buf.zero_()
        return self.env.get_observations()

    def state_dict(self) -> dict[str, object]:
        """Return the exact recorded event history."""
        return {"transitions": tuple(self.transitions)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore the exact recorded event history."""
        transitions = state["transitions"]
        if not isinstance(transitions, tuple) or not all(isinstance(value, int) for value in transitions):
            raise TypeError("Lifecycle transition state must be a tuple of integers.")
        self.transitions = list(transitions)


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
    return ForwardBackwardExpertBuffer(
        frames,
        offsets,
        priorities,
        schema,
        seed=17,
        clip_ids=("clip_0", "clip_1"),
        clip_length_values=(16, 16),
    )


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
            "capacity_transitions": 8 * NUM_ENVS,
            "terminal_capacity_per_env": 4,
            "autoreset_mode": "same_step",
            "environment_reward_name": "environment",
            "auxiliary_evidence_names": ["effort"],
            "auxiliary_evidence_observation_group": "transition",
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


def _make_named_evidence_cfg() -> dict:
    """Return the tiny config with a two-channel named evidence schema."""
    cfg = _make_cfg()
    cfg["replay"]["auxiliary_evidence_names"] = ["effort", "impact"]
    cfg["replay"]["reward_channels"].append({
        "name": "impact",
        "provider_name": "impact",
        "source": "stored_evidence",
        "timing": "transition",
        "context_dependent": False,
        "sign": -1,
    })
    cfg["model"]["value_heads"][1]["spec"]["reward_channels"] = ["effort", "impact"]
    cfg["algorithm"]["value_cfg"]["auxiliary"]["reward_coefficients"] = [0.1, 0.2]
    return cfg


def _make_lifecycle_cfg(transition_interval: int = 2 * NUM_ENVS) -> dict:
    cfg = _make_cfg()
    cfg["lifecycle_extension"] = {
        "class_name": RecordingLifecycleExtension,
        "transition_interval": transition_interval,
    }
    return cfg


def _make_history_cfg() -> dict:
    """Return a runner config whose actor history is derived by the learner."""
    cfg = _make_cfg()
    cfg["obs_groups"]["actor"] = ["state", "history_actor"]
    cfg["obs_groups"]["forward"] = ["state", "history_actor"]
    cfg["replay"]["history_layout"] = {
        "history_field": "history_actor",
        "history_length": 2,
        "sources": [{"observation_name": "state"}],
        "include_seed_observations": False,
    }
    return cfg


class ObservedOffPolicyRunner(OffPolicyRunner):
    """Record the exact runner-loop boundaries exposed to specialized subclasses."""

    boundary_events: list[tuple]

    def _observe_iteration_start(self, iteration: int, start_transitions: int) -> None:
        """Record the boundary immediately before collection."""
        self.boundary_events.append(("start", iteration, start_transitions))

    def _observe_iteration_learning_complete(self, iteration: int, end_transitions: int) -> None:
        """Record the boundary after updates and before metric materialization."""
        self.boundary_events.append(("learning_complete", iteration, end_transitions))

    def _observe_iteration_complete(
        self,
        iteration: int,
        end_transitions: int,
        collect_time: float,
        learn_time: float,
    ) -> None:
        """Record existing decomposition timers before logging and checkpointing."""
        self.boundary_events.append(("complete", iteration, end_transitions, collect_time, learn_time))


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


def test_constructor_derives_time_major_rows_from_transition_capacity() -> None:
    """High-level capacity should count transitions while replay stores vector steps."""
    cfg = _make_cfg()
    assert "capacity_steps" not in cfg["replay"]
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(), cfg, log_dir=None, device="cpu")

    assert runner.alg.replay.capacity_steps == 8
    assert runner.alg.replay.capacity_steps * runner.env.num_envs == 8 * NUM_ENVS


@pytest.mark.parametrize(
    ("group", "error"),
    (
        (None, "non-empty string"),
        ("state", "non-model observation group"),
        ("missing", "was not returned by the environment"),
    ),
)
def test_constructor_rejects_invalid_auxiliary_evidence_observation_group(
    group: str | None,
    error: str,
) -> None:
    """Evidence should be bound once to an existing non-policy observation group."""
    cfg = _make_cfg()
    cfg["replay"]["auxiliary_evidence_observation_group"] = group

    with pytest.raises(ValueError, match=error):
        OffPolicyRunner(ForwardBackwardDummyEnv(), cfg, log_dir=None, device="cpu")


def test_constructor_rejects_auxiliary_evidence_width_mismatch() -> None:
    """The configured channel order should exactly determine the observation width."""
    cfg = _make_cfg()
    cfg["replay"]["auxiliary_evidence_names"].append("impact")
    cfg["replay"]["reward_channels"].append({
        "name": "impact",
        "provider_name": "impact",
        "source": "stored_evidence",
        "timing": "transition",
        "context_dependent": False,
        "sign": -1,
    })

    with pytest.raises(ValueError, match=r"must have shape \(4, 2\)"):
        OffPolicyRunner(ForwardBackwardDummyEnv(), cfg, log_dir=None, device="cpu")


def test_constructor_rejects_evidence_from_non_actor_model_route() -> None:
    """Transition evidence must not leak into a representation or value input route."""
    cfg = _make_cfg()
    cfg["obs_groups"]["forward"] = ["state", "transition"]

    with pytest.raises(ValueError, match="non-model observation group"):
        OffPolicyRunner(ForwardBackwardDummyEnv(), cfg, log_dir=None, device="cpu")


@pytest.mark.parametrize("capacity_transitions", (False, 0, -1, 32.0, 8 * NUM_ENVS - 1))
def test_constructor_rejects_invalid_transition_capacity(
    capacity_transitions: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Invalid transition counts should fail before replay storage is allocated."""
    cfg = _make_cfg()
    cfg["replay"]["capacity_transitions"] = capacity_transitions
    monkeypatch.setattr(
        ForwardBackwardReplay, "__init__", lambda *_args, **_kwargs: pytest.fail("replay storage was allocated")
    )
    monkeypatch.setattr(
        ForwardBackwardModel,
        "from_config",
        classmethod(lambda *_args, **_kwargs: pytest.fail("model storage was allocated")),
    )

    with pytest.raises(ValueError, match="capacity_transitions"):
        OffPolicyRunner(ForwardBackwardDummyEnv(), cfg, log_dir=None, device="cpu")


def test_lifecycle_extension_runs_at_zero_each_interval_and_final() -> None:
    """Lifecycle work should use completed transitions independently of save hooks."""
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(), _make_lifecycle_cfg(), log_dir=None, device="cpu")
    acted_from: list[torch.Tensor] = []
    original_act = runner.alg.act

    def record_act(observations: TensorDict) -> torch.Tensor:
        """Record collection input before delegating to the real algorithm."""
        acted_from.append(observations["state"].clone())
        return original_act(observations)

    runner.alg.act = record_act
    runner.learn(2)

    extension = runner.lifecycle_extension
    assert isinstance(extension, RecordingLifecycleExtension)
    assert extension.transitions == [0, 2 * NUM_ENVS, 4 * NUM_ENVS]
    assert runner._lifecycle_last_transition == runner.collected_transitions == 4 * NUM_ENVS
    torch.testing.assert_close(acted_from[2], torch.full_like(acted_from[2], 2 * NUM_ENVS))


def test_one_vector_step_advances_transition_clock_by_exactly_num_envs() -> None:
    """One physical vector step must contribute one transition per environment."""
    cfg = _make_lifecycle_cfg(NUM_ENVS)
    cfg["num_steps_per_env"] = 1
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(), cfg, log_dir=None, device="cpu")

    runner.learn(1)

    extension = runner.lifecycle_extension
    assert isinstance(extension, RecordingLifecycleExtension)
    assert runner.alg.replay.total_steps == 1
    assert runner.collected_transitions == NUM_ENVS
    assert extension.transitions == [0, NUM_ENVS]


def test_lifecycle_extension_rejects_unreachable_transition_cadence() -> None:
    """An interval must land on an exact completed collection boundary."""
    with pytest.raises(ValueError, match="positive multiple of one collection block"):
        OffPolicyRunner(ForwardBackwardDummyEnv(), _make_lifecycle_cfg(NUM_ENVS), log_dir=None, device="cpu")


def test_lifecycle_extension_checkpoint_before_first_event_replays_zero_once() -> None:
    """A pre-learning checkpoint should preserve the pending transition-zero event."""
    fresh = OffPolicyRunner(ForwardBackwardDummyEnv(), _make_lifecycle_cfg(), log_dir=None, device="cpu")
    restored = OffPolicyRunner(ForwardBackwardDummyEnv(), _make_lifecycle_cfg(), log_dir=None, device="cpu")

    with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
        fresh.save(checkpoint.name)
        restored.load(checkpoint.name)

    extension = restored.lifecycle_extension
    assert isinstance(extension, RecordingLifecycleExtension)
    assert extension.transitions == []

    restored.learn(1)

    assert extension.transitions == [0, 2 * NUM_ENVS]


def test_lifecycle_extension_checkpoint_resumes_without_replaying_event_zero() -> None:
    """Extension state and cadence should resume at the exact next transition event."""
    expected = OffPolicyRunner(ForwardBackwardDummyEnv(), _make_lifecycle_cfg(), log_dir=None, device="cpu")
    restored = OffPolicyRunner(ForwardBackwardDummyEnv(), _make_lifecycle_cfg(), log_dir=None, device="cpu")
    expected.learn(1)

    with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
        expected.save(checkpoint.name)
        restored.load(checkpoint.name)

    restored_extension = restored.lifecycle_extension
    assert isinstance(restored_extension, RecordingLifecycleExtension)
    assert restored_extension.transitions == [0, 2 * NUM_ENVS]

    restored.learn(1)

    assert restored_extension.transitions == [0, 2 * NUM_ENVS, 4 * NUM_ENVS]
    assert restored._lifecycle_last_transition == 4 * NUM_ENVS


def test_collection_does_not_leak_inference_tensors_into_environment_state() -> None:
    """State retained by an environment should remain mutable outside collection."""
    env = ForwardBackwardDummyEnv()
    runner = OffPolicyRunner(env, _make_cfg(), log_dir=None, device="cpu")

    runner.learn(1)

    assert not torch.is_inference(env.last_actions)
    env.last_actions.zero_()


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


def test_runner_training_summary_persists_exact_updates_and_finite_metric_keys() -> None:
    """The completion boundary should expose counters and every emitted learner metric."""
    runner = OffPolicyRunner(
        ForwardBackwardDummyEnv(),
        _make_cfg(random_action_steps=2 * NUM_ENVS),
        log_dir=None,
        device="cpu",
    )

    runner.learn(3)

    summary = runner.training_summary()
    assert summary["completed_iterations"] == 3
    assert summary["collected_transitions"] == 6 * NUM_ENVS
    assert summary["update_calls"] == runner.alg.update_step == 1
    assert summary["all_metrics_finite"] is True
    assert set(summary["metric_names"]) == set(summary["last_metrics"])
    assert summary["metric_names"]


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


def test_same_step_collection_reads_completed_edge_evidence_from_observations() -> None:
    """Done-edge evidence should come from final_obs, not the reset observation."""
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(), _make_cfg(), log_dir=None, device="cpu")
    _collect(runner, 3)
    env_ids = torch.arange(NUM_ENVS)

    live = runner.alg.replay.sample(torch.full((NUM_ENVS,), 1), env_ids)
    done = runner.alg.replay.sample(torch.full((NUM_ENVS,), 2), env_ids)

    torch.testing.assert_close(live.auxiliary_reward_evidence, torch.full((NUM_ENVS, 1), 4.0))
    torch.testing.assert_close(done.auxiliary_reward_evidence, torch.full((NUM_ENVS, 1), 9.0))


def test_same_step_collection_reads_named_auxiliary_evidence_fields() -> None:
    """Named fields should retain schema order through live, final, and fallback paths."""
    runner = OffPolicyRunner(NamedEvidenceDummyEnv(), _make_named_evidence_cfg(), log_dir=None, device="cpu")
    _collect(runner, 3)
    env_ids = torch.arange(NUM_ENVS)

    live = runner.alg.replay.sample(torch.full((NUM_ENVS,), 1), env_ids)
    done = runner.alg.replay.sample(torch.full((NUM_ENVS,), 2), env_ids)
    torch.testing.assert_close(
        live.auxiliary_reward_evidence,
        torch.tensor([4.0, 104.0]).expand(NUM_ENVS, -1),
    )
    torch.testing.assert_close(
        done.auxiliary_reward_evidence,
        torch.tensor([9.0, 109.0]).expand(NUM_ENVS, -1),
    )

    fallback_runner = OffPolicyRunner(
        NamedEvidenceDummyEnv(provide_final=False),
        _make_named_evidence_cfg(),
        log_dir=None,
        device="cpu",
    )
    _collect(fallback_runner, 3)
    fallback = fallback_runner.alg.replay.sample(torch.full((NUM_ENVS,), 2), env_ids)
    torch.testing.assert_close(
        fallback.auxiliary_reward_evidence,
        torch.tensor([4.0, 104.0]).expand(NUM_ENVS, -1),
    )


def test_same_step_collection_owns_action_and_reached_episode_steps() -> None:
    """Same-step collection should require neither action nor episode metadata from the environment."""
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(), _make_cfg(), log_dir=None, device="cpu")

    _collect(runner, 1)
    assert runner.alg.replay.episode_steps.tolist() == [1] * NUM_ENVS
    _collect(runner, 1)
    assert runner.alg.replay.episode_steps.tolist() == [2] * NUM_ENVS
    _collect(runner, 1)
    assert runner.alg.replay.episode_steps.tolist() == [0] * NUM_ENVS


def test_same_step_collection_falls_back_to_pre_step_when_final_is_unavailable() -> None:
    """Missing final_obs should use the pre-step evidence approximation, not reset state."""
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(provide_final=False), _make_cfg(), log_dir=None, device="cpu")
    _collect(runner, 3)
    env_ids = torch.arange(NUM_ENVS)
    batch = runner.alg.replay.sample(torch.full((NUM_ENVS,), 2), env_ids)

    torch.testing.assert_close(batch.next_observations["state"], torch.full((NUM_ENVS, STATE_DIM), 2.0))
    torch.testing.assert_close(batch.auxiliary_reward_evidence, torch.full((NUM_ENVS, 1), 4.0))
    assert torch.all(batch.successor_uses_current)


def test_online_history_is_learner_owned_and_matches_replay_order() -> None:
    """Raw environments should not materialize derived actor history."""
    env = ForwardBackwardDummyEnv()
    runner = OffPolicyRunner(env, _make_history_cfg(), log_dir=None, device="cpu")
    observations = env.get_observations()

    actions = runner.alg.act_random(observations)
    assert "history_actor" not in observations
    torch.testing.assert_close(
        runner.alg._collection_observations["history_actor"],
        torch.zeros(NUM_ENVS, 2 * STATE_DIM),
    )
    observations, rewards, dones, extras = env.step(actions)
    runner.alg.process_env_step(observations, rewards, dones, extras)

    actions = runner.alg.act_random(observations)
    torch.testing.assert_close(
        runner.alg._collection_observations["history_actor"],
        torch.zeros(NUM_ENVS, 2 * STATE_DIM),
    )
    observations, rewards, dones, extras = env.step(actions)
    runner.alg.process_env_step(observations, rewards, dones, extras)

    actions = runner.alg.act_random(observations)
    expected = torch.cat((torch.ones(NUM_ENVS, STATE_DIM), torch.zeros(NUM_ENVS, STATE_DIM)), dim=-1)
    torch.testing.assert_close(runner.alg._collection_observations["history_actor"], expected)
    observations, rewards, dones, extras = env.step(actions)
    runner.alg.process_env_step(observations, rewards, dones, extras)

    torch.testing.assert_close(runner.alg._online_history.current, torch.zeros(NUM_ENVS, 2 * STATE_DIM))
    expected_final = torch.cat((torch.full((NUM_ENVS, STATE_DIM), 2.0), torch.ones(NUM_ENVS, STATE_DIM)), dim=-1)
    torch.testing.assert_close(runner.alg._online_history.reached, expected_final)


def test_online_history_external_reset_preserves_final_and_zeros_new_episode() -> None:
    """An external reset should retain the closed edge and zero only reset streams."""
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(), _make_history_cfg(), log_dir=None, device="cpu")
    _collect(runner, 2)
    reset = torch.tensor([True, False, True, False])
    final_history = runner.alg._online_history.current.clone()

    runner.env.state[reset] = 50.0
    runner.env.episode_length_buf[reset] = 0
    runner.alg.process_env_reset(runner.env.get_observations(), reset)

    expected_current = final_history.clone()
    expected_current[reset] = 0.0
    torch.testing.assert_close(runner.alg._online_history.current, expected_current)

    reset_env_ids = reset.nonzero(as_tuple=False).squeeze(-1)
    closed = runner.alg.replay.sample(torch.ones(reset_env_ids.shape[0], dtype=torch.long), reset_env_ids)
    assert torch.all(closed.truncated)
    torch.testing.assert_close(closed.next_observations["state"], torch.full((reset_env_ids.shape[0], STATE_DIM), 2.0))
    torch.testing.assert_close(closed.next_observations["history_actor"], final_history[reset_env_ids])

    runner.alg.act_random(runner.env.get_observations())
    torch.testing.assert_close(runner.alg._collection_observations["history_actor"], expected_current)


def test_online_history_checkpoint_resumes_the_exact_next_edge() -> None:
    """Checkpoint restore must preserve the learner-owned history boundary exactly."""
    expected = OffPolicyRunner(ForwardBackwardDummyEnv(), _make_history_cfg(), log_dir=None, device="cpu")
    restored = OffPolicyRunner(ForwardBackwardDummyEnv(), _make_history_cfg(), log_dir=None, device="cpu")
    _collect(expected, 2)
    algorithm_state = copy.deepcopy(expected.alg.save())
    environment_state = copy.deepcopy(expected.env.state_dict())
    restored.env.load_state_dict(environment_state)
    restored.alg.load(algorithm_state, load_cfg=None, strict=True)

    torch.testing.assert_close(restored.alg._online_history.current, expected.alg._online_history.current)
    torch.testing.assert_close(
        restored.alg._online_history.current_source_valid,
        expected.alg._online_history.current_source_valid,
    )

    expected_observations = expected.env.get_observations()
    restored_observations = restored.env.get_observations()
    expected_actions = expected.alg.act_random(expected_observations)
    restored_actions = restored.alg.act_random(restored_observations)
    torch.testing.assert_close(restored_actions, expected_actions, rtol=0.0, atol=0.0)

    expected_observations, expected_rewards, expected_dones, expected_extras = expected.env.step(expected_actions)
    restored_observations, restored_rewards, restored_dones, restored_extras = restored.env.step(restored_actions)
    expected.alg.process_env_step(expected_observations, expected_rewards, expected_dones, expected_extras)
    restored.alg.process_env_step(restored_observations, restored_rewards, restored_dones, restored_extras)

    torch.testing.assert_close(restored.alg._online_history.current, expected.alg._online_history.current)
    torch.testing.assert_close(
        restored.alg._online_history.current_source_valid,
        expected.alg._online_history.current_source_valid,
    )
    assert restored.alg.replay.total_steps == expected.alg.replay.total_steps


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


def test_rolling_expert_context_restarts_at_episode_reset() -> None:
    """Tracking contexts should restart from reached-frame position zero after reset."""
    runner = OffPolicyRunner(
        ForwardBackwardDummyEnv(), _make_cfg(rollout_expert_fraction=0.5), log_dir=None, device="cpu"
    )

    _collect(runner, 2)
    assert torch.all(runner.alg._rollout_tracking_positions == 2)
    _collect(runner, 1)

    assert torch.all(runner.alg._rollout_tracking_positions == 0)
    assert runner.alg._rollout_tracking_env_ids.unique().numel() == NUM_ENVS // 2


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
    torch.testing.assert_close(restored.alg._rollout_tracking_positions, expected.alg._rollout_tracking_positions)
    assert restored.alg.rollout_schedule_step == expected.alg.rollout_schedule_step


def test_runner_materializes_mean_metrics_at_logging_boundary() -> None:
    """Repeated device metrics should become ordinary logging scalars only after averaging."""
    metrics = [
        {"loss": torch.tensor(1.0), "value": torch.tensor(3.0)},
        {"loss": torch.tensor(3.0), "value": torch.tensor(7.0)},
    ]

    assert OffPolicyRunner._mean_metrics(metrics) == {"loss": 2.0, "value": 5.0}


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
        runner.load(checkpoint.name, map_location="cpu", mmap=True)

    assert runner.environment_resume_exact
    assert runner.current_learning_iteration == 7
    assert runner.collected_transitions == 0
    torch.testing.assert_close(runner.env.state, expected_state)
    assert runner.checkpoint_load_summary() == {
        "environment_resume": "exact",
        "environment_state_dict_is_none": False,
        "map_location": "cpu",
        "mmap": True,
        "strict": True,
    }


def test_runner_exposes_exact_iteration_boundaries() -> None:
    """Hooks should bracket unchanged collection/update work and precede logging."""
    runner = ObservedOffPolicyRunner(
        ForwardBackwardDummyEnv(),
        _make_cfg(),
        log_dir=None,
        device="cpu",
    )
    runner.boundary_events = []
    act = runner.alg.act
    mean_metrics = runner._mean_metrics

    def record_act(obs: TensorDict) -> torch.Tensor:
        runner.boundary_events.append(("act", runner.collected_transitions))
        return act(obs)

    def record_update() -> dict[str, torch.Tensor]:
        runner.boundary_events.append(("update", runner.collected_transitions))
        return {"loss": torch.tensor(1.0)}

    def record_mean_metrics(metrics: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        runner.boundary_events.append(("mean_metrics", runner.collected_transitions))
        return mean_metrics(metrics)

    def record_log(**kwargs: object) -> None:
        runner.boundary_events.append(("log", kwargs["it"], runner.collected_transitions))

    runner.alg.act = record_act
    runner.alg.update = record_update
    runner._mean_metrics = record_mean_metrics
    runner.logger.log = record_log

    runner.learn(1)

    assert [event[0] for event in runner.boundary_events] == [
        "start",
        "act",
        "act",
        "update",
        "learning_complete",
        "mean_metrics",
        "complete",
        "log",
    ]
    assert runner.boundary_events[0] == ("start", 0, 0)
    assert runner.boundary_events[4] == ("learning_complete", 0, 2 * NUM_ENVS)
    complete = runner.boundary_events[6]
    assert complete[:3] == ("complete", 0, 2 * NUM_ENVS)
    assert complete[3] >= 0.0
    assert complete[4] >= 0.0


def test_default_iteration_observer_is_state_and_rng_inert() -> None:
    """The default hooks should not mutate runner state or any owned RNG stream."""
    runner = OffPolicyRunner(
        ForwardBackwardDummyEnv(),
        _make_cfg(),
        log_dir=None,
        device="cpu",
    )
    torch_rng = torch.get_rng_state().clone()
    owned_rngs = tuple(
        generator.get_state().clone()
        for generator in (
            runner.alg.generator,
            runner.alg.behavior_generator,
            runner.alg.replay.generator,
            runner.alg.expert.generator,
        )
    )
    state = (runner.current_learning_iteration, runner.collected_transitions)

    runner._observe_iteration_start(0, 0)
    runner._observe_iteration_learning_complete(0, 2 * NUM_ENVS)
    runner._observe_iteration_complete(0, 2 * NUM_ENVS, 0.1, 0.2)

    assert torch.equal(torch.get_rng_state(), torch_rng)
    for generator, expected in zip(
        (
            runner.alg.generator,
            runner.alg.behavior_generator,
            runner.alg.replay.generator,
            runner.alg.expert.generator,
        ),
        owned_rngs,
    ):
        assert torch.equal(generator.get_state(), expected)
    assert (runner.current_learning_iteration, runner.collected_transitions) == state
