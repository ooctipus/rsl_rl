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
from rsl_rl.runners.off_policy_runner import OffPolicyRunner
from rsl_rl.storage.forward_backward_expert import ForwardBackwardExpertBuffer, ForwardBackwardExpertSchema
from rsl_rl.storage.forward_backward_replay import ForwardBackwardReplay
from rsl_rl.utils.utils import check_nan

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


def _expert_provider(
    env: VecEnv,
    observation_schema: ForwardBackwardObservationSchema,
    device: str,
    *,
    clock: dict[str, object],
    window_lengths: tuple[int, ...],
    seed: int,
) -> ForwardBackwardExpertBuffer:
    """Return one deterministic two-clip corpus on the learner device."""
    del env
    assert clock == {"sampling_mode": "source_rows", "sampling_step_seconds": None}
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
        seed=seed,
        clip_ids=("clip_0", "clip_1"),
        clip_length_values=(16, 16),
    )


def _make_cfg(*, rollout_expert_fraction: float = 0.0) -> dict:
    """Return a tiny canonical forward-backward runner configuration."""
    network = {"hidden_dim": 16, "hidden_layers": 1, "embedding_layers": 2}
    return {
        "seed": 23,
        "num_steps_per_env": 2,
        "num_updates_per_iteration": 1,
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
        },
        "replay": {
            "class_name": "rsl_rl.storage.forward_backward_replay:ForwardBackwardReplay",
            "policy": {
                "capacity_transitions": 8 * NUM_ENVS,
                "terminal_capacity_per_env": 4,
                "sampling": "transition_uniform",
            },
            "autoreset_mode": "same_step",
        },
        "expert": {
            "provider": _expert_provider,
            "clock": {"sampling_mode": "source_rows", "sampling_step_seconds": None},
            "window_lengths": (2, 6),
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms.forward_backward:ForwardBackward",
            "batch_size": 8,
            "expert_sequence_length": 2,
            "optimization": {},
            "context": {
                "buffer_capacity": 16,
                "refresh_steps": 2,
                "rollout_expert_fraction": rollout_expert_fraction,
                "rollout_expert_steps": 4,
                "rollout_expert_context_steps": 3,
            },
            "exploration": {"random_action_transitions": 0},
            "discriminator_gradient_penalty_coefficient": 0.0,
        },
        "value_helpers": [
            {
                "name": "discriminator",
                "learning_rate": 1.0e-4,
                "route": "critic_discriminator",
                "terms": [
                    {
                        "name": "discriminator",
                        "coefficient": 1.0,
                        "source": "recomputed",
                        "timing": "next_state",
                        "context_dependent": True,
                        "sign": 1,
                    }
                ],
                "reward_composition": "vector",
                "pessimism": 0.5,
                "actor_coefficient": 0.05,
                "normalize_rewards": False,
                "reward_normalization_decay": None,
                "reward_normalization_epsilon": None,
                "target_tau": 0.005,
            },
            {
                "name": "auxiliary",
                "learning_rate": 1.0e-4,
                "route": "critic_auxiliary",
                "terms": [
                    {
                        "name": "effort",
                        "coefficient": 0.1,
                        "source": "stored_evidence",
                        "timing": "transition",
                        "context_dependent": False,
                        "sign": -1,
                    }
                ],
                "reward_composition": "vector",
                "pessimism": 0.5,
                "actor_coefficient": 0.02,
                "normalize_rewards": True,
                "reward_normalization_decay": 0.99,
                "reward_normalization_epsilon": 1.0e-8,
                "target_tau": 0.005,
            },
        ],
        "torch_compile_mode": None,
    }


def _make_named_evidence_cfg() -> dict:
    """Return the tiny config with a two-channel named evidence schema."""
    cfg = _make_cfg()
    cfg["value_helpers"][1]["terms"].append({
        "name": "impact",
        "coefficient": 0.2,
        "source": "stored_evidence",
        "timing": "transition",
        "context_dependent": False,
        "sign": -1,
    })
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


class UpdatingOffPolicyRunner(OffPolicyRunner):
    """Replace collection observations at the ordinary update boundary."""

    def _update(
        self,
        observations: TensorDict,
    ) -> tuple[TensorDict, list[dict[str, torch.Tensor]]]:
        observations, metrics = super()._update(observations)
        observations = observations.clone()
        observations["state"].fill_(float(self.collected_transitions))
        return observations, metrics


class StatefulOffPolicyRunner(OffPolicyRunner):
    """Extend ordinary runner checkpoints with one fixture-owned value."""

    fixture_state = 0

    def state_dict(self) -> dict[str, object]:
        """Add fixture-owned state to the ordinary checkpoint."""
        state = super().state_dict()
        state["fixture_state"] = self.fixture_state
        return state

    def load_state_dict(
        self,
        state_dict: dict[str, object],
        load_cfg: dict | None = None,
        strict: bool = True,
    ) -> None:
        """Restore ordinary and fixture-owned checkpoint state."""
        super().load_state_dict(state_dict, load_cfg, strict)
        self.fixture_state = int(state_dict["fixture_state"])


def _collect(runner: OffPolicyRunner, steps: int) -> None:
    """Collect a fixed number of transitions without invoking the runner loop."""
    obs = runner.env.get_observations()
    for _ in range(steps):
        actions = runner.alg.act(obs)
        obs, rewards, dones, extras = runner.env.step(actions)
        runner.alg.process_env_step(obs, rewards, dones, extras)


def test_runner_constructs_collects_and_updates() -> None:
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
    assert "capacity_steps" not in cfg["replay"]["policy"]
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(), cfg, log_dir=None, device="cpu")

    assert runner.alg.replay.capacity_steps == 8
    assert runner.alg.replay.capacity_steps * runner.env.num_envs == 8 * NUM_ENVS
    assert runner.alg.replay.sampling is ForwardBackwardReplay.Sampling.TRANSITION_UNIFORM


def test_constructor_derives_auxiliary_evidence_route_from_helper_terms() -> None:
    """Stored transition terms should select the fixed transition observation group."""
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(), _make_cfg(), log_dir=None, device="cpu")

    assert runner.alg._auxiliary_evidence_observation_group == "transition"
    assert runner.alg.replay.transition_schema.auxiliary_evidence_names == ("effort",)


def test_constructor_rejects_auxiliary_evidence_width_mismatch() -> None:
    """The configured channel order should exactly determine the observation width."""
    cfg = _make_cfg()
    cfg["value_helpers"][1]["terms"].append({
        "name": "impact",
        "coefficient": 0.2,
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
    cfg["replay"]["policy"]["capacity_transitions"] = capacity_transitions
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


def test_one_vector_step_advances_transition_clock_by_exactly_num_envs() -> None:
    """One physical vector step must contribute one transition per environment."""
    cfg = _make_cfg()
    cfg["num_steps_per_env"] = 1
    runner = OffPolicyRunner(ForwardBackwardDummyEnv(), cfg, log_dir=None, device="cpu")

    runner.learn(1)

    assert runner.alg.replay.total_steps == 1
    assert runner.collected_transitions == NUM_ENVS


def test_collection_does_not_leak_inference_tensors_into_environment_state() -> None:
    """State retained by an environment should remain mutable outside collection."""
    env = ForwardBackwardDummyEnv()
    runner = OffPolicyRunner(env, _make_cfg(), log_dir=None, device="cpu")

    runner.learn(1)

    assert not torch.is_inference(env.last_actions)
    env.last_actions.zero_()


def test_runner_training_summary_persists_exact_updates_and_finite_metric_keys() -> None:
    """The completion boundary should expose counters and every emitted learner metric."""
    runner = OffPolicyRunner(
        ForwardBackwardDummyEnv(),
        _make_cfg(),
        log_dir=None,
        device="cpu",
    )

    runner.learn(3)

    summary = runner.training_summary()
    assert summary["completed_iterations"] == 3
    assert summary["collected_transitions"] == 6 * NUM_ENVS
    assert summary["update_calls"] == runner.alg.update_step == 3
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


def test_nan_check_accepts_nested_observation_groups() -> None:
    """Finite named evidence should pass the real runner NaN check."""
    runner = OffPolicyRunner(NamedEvidenceDummyEnv(), _make_named_evidence_cfg(), log_dir=None, device="cpu")
    runner.learn(1)


def test_nan_check_reports_nested_observation_path() -> None:
    """A nested NaN should identify its complete observation path."""
    observations = NamedEvidenceDummyEnv().get_observations()
    observations["transition", "impact"][0, 0] = torch.nan
    with pytest.raises(ValueError, match=r"transition\.impact"):
        check_nan(observations, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS))


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


def test_update_boundary_can_replace_collection_observations() -> None:
    """A specialized runner can resume collection from observations returned by update."""
    runner = UpdatingOffPolicyRunner(ForwardBackwardDummyEnv(), _make_cfg(), log_dir=None, device="cpu")
    acted_from: list[torch.Tensor] = []
    original_act = runner.alg.act

    def record_act(observations: TensorDict) -> torch.Tensor:
        acted_from.append(observations["state"].clone())
        return original_act(observations)

    runner.alg.act = record_act
    runner.learn(2)

    torch.testing.assert_close(acted_from[2], torch.full_like(acted_from[2], 2 * NUM_ENVS))


def test_subclass_state_round_trips_through_runner_checkpoint() -> None:
    """Specialized runner state should extend the ordinary checkpoint contract."""
    saved = StatefulOffPolicyRunner(ForwardBackwardDummyEnv(), _make_cfg(), log_dir=None, device="cpu")
    restored = StatefulOffPolicyRunner(ForwardBackwardDummyEnv(), _make_cfg(), log_dir=None, device="cpu")
    saved.fixture_state = 41

    with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
        saved.save(checkpoint.name)
        restored.load(checkpoint.name)

    assert restored.fixture_state == 41
