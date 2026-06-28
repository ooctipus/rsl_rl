# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Phase 1D forward-backward GPU data plane."""

from __future__ import annotations

import torch
from dataclasses import replace
from tensordict import TensorDict

import pytest

from rsl_rl.models.forward_backward_model import ForwardBackwardObservationSchema
from rsl_rl.storage.forward_backward_replay import (
    ForwardBackwardAutoresetMode,
    ForwardBackwardHistoryLayout,
    ForwardBackwardReplay,
    ForwardBackwardReplayBatch,
    ForwardBackwardTransitionBatch,
    ForwardBackwardTransitionSchema,
)
from tests.fixtures.forward_backward import ForwardBackwardReplayOracle, make_reward_schema


def _contracts(
    mode: ForwardBackwardAutoresetMode = ForwardBackwardAutoresetMode.SAME_STEP,
    field_widths: dict[str, int] | None = None,
    action_width: int = 2,
) -> tuple[ForwardBackwardObservationSchema, ForwardBackwardTransitionSchema]:
    if field_widths is None:
        field_widths = {"state": 2}
    observation_schema = ForwardBackwardObservationSchema.from_config(field_widths, {"actor": tuple(field_widths)})
    reward_schema = make_reward_schema()
    transition_schema = ForwardBackwardTransitionSchema(
        observation_schema_hash=observation_schema.schema_hash,
        reward_schema_hash=reward_schema.schema_hash,
        action_width=action_width,
        context_width=3,
        environment_reward_name="environment",
        auxiliary_evidence_names=("action_rate", "slip"),
        autoreset_mode=mode,
    )
    transition_schema.assert_compatible(observation_schema, reward_schema)
    return observation_schema, transition_schema


def _state(step: int, num_envs: int, width: int = 2, offset: float = 0.0) -> torch.Tensor:
    env = torch.arange(num_envs, dtype=torch.float32).unsqueeze(-1)
    component = torch.arange(width, dtype=torch.float32).unsqueeze(0) / 10
    return step * 10 + env + component + offset


def _transition(
    step: int,
    mode: ForwardBackwardAutoresetMode,
    observation_schema: ForwardBackwardObservationSchema,
    transition_schema: ForwardBackwardTransitionSchema,
    *,
    done_envs: tuple[int, ...] = (),
    truncated_envs: tuple[int, ...] = (),
    action_applied: torch.Tensor | None = None,
    current: dict[str, torch.Tensor] | None = None,
    reached: dict[str, torch.Tensor] | None = None,
    reset: dict[str, torch.Tensor] | None = None,
) -> ForwardBackwardTransitionBatch:
    num_envs = 3
    if current is None:
        current = {name: _state(step, num_envs, width) for name, width in observation_schema.field_widths}
    if reached is None:
        reached = {name: _state(step + 1, num_envs, width) for name, width in observation_schema.field_widths}
    if reset is None:
        reset = {name: value.clone() for name, value in reached.items()}
    terminated = torch.zeros(num_envs, 1, dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    if done_envs:
        terminated[list(done_envs)] = True
    if truncated_envs:
        truncated[list(truncated_envs)] = True
        terminated[list(truncated_envs)] = False
    done = terminated | truncated
    if mode is ForwardBackwardAutoresetMode.SAME_STEP:
        next_observations = reset
        final_observations = reached
        final_valid = done
    else:
        next_observations = reached
        final_observations = {name: torch.full_like(value, float("nan")) for name, value in reached.items()}
        final_valid = torch.zeros_like(done)
    if action_applied is None:
        action_applied = torch.ones(num_envs, 1, dtype=torch.bool)
    context_changed = torch.zeros_like(done)
    context_changed[2] = step % 2 == 1
    env = torch.arange(num_envs, dtype=torch.float32).unsqueeze(-1)
    actions = env + step + torch.arange(transition_schema.action_width, dtype=torch.float32).unsqueeze(0) / 10
    return ForwardBackwardTransitionBatch(
        observations=TensorDict(current, batch_size=[num_envs]),
        next_observations=TensorDict(next_observations, batch_size=[num_envs]),
        final_observations=TensorDict(final_observations, batch_size=[num_envs]),
        actions=actions,
        behavior_context=torch.cat((env + step, env + step + 1, env + step + 2), dim=-1),
        environment_reward=env + step / 10,
        auxiliary_reward_evidence=torch.cat((env + step / 100, env + step / 50), dim=-1),
        terminated=terminated,
        truncated=truncated,
        context_changed=context_changed,
        action_applied=action_applied,
        final_observation_valid=final_valid,
    )


def _make_replays(
    capacity_steps: int,
    terminal_capacity_per_env: int,
    observation_schema: ForwardBackwardObservationSchema,
    transition_schema: ForwardBackwardTransitionSchema,
    history_layout: ForwardBackwardHistoryLayout | None = None,
) -> tuple[ForwardBackwardReplayOracle, ForwardBackwardReplay]:
    reward_schema = make_reward_schema()
    oracle = ForwardBackwardReplayOracle(
        capacity_steps, 3, observation_schema, transition_schema, reward_schema, "cpu", seed=7
    )
    replay = ForwardBackwardReplay(
        capacity_steps,
        3,
        terminal_capacity_per_env,
        observation_schema,
        transition_schema,
        reward_schema,
        "cpu",
        history_layout=history_layout,
        seed=7,
    )
    return oracle, replay


def _assert_batches_equal(expected: ForwardBackwardReplayBatch, actual: ForwardBackwardReplayBatch) -> None:
    assert expected.observations.keys() == actual.observations.keys()
    assert expected.next_observations.keys() == actual.next_observations.keys()
    for name in expected.observations.keys(include_nested=False, leaves_only=True):
        torch.testing.assert_close(actual.observations[name], expected.observations[name])
        torch.testing.assert_close(actual.next_observations[name], expected.next_observations[name])
    for name in (
        "actions",
        "behavior_context",
        "environment_reward",
        "auxiliary_reward_evidence",
        "terminated",
        "truncated",
        "context_changed",
        "successor_uses_current",
        "valid",
    ):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))


def test_node_edge_replay_matches_dense_oracle_through_wrap_and_boundaries() -> None:
    """Adjacent nodes and sparse finals should preserve every live logical transition."""
    observation_schema, transition_schema = _contracts()
    oracle, replay = _make_replays(3, 3, observation_schema, transition_schema)
    for step in range(6):
        done_envs = (1,) if step == 1 else ((2,) if step == 4 else ())
        transition = _transition(
            step, transition_schema.autoreset_mode, observation_schema, transition_schema, done_envs=done_envs
        )
        oracle.add(transition)
        replay.add(transition)

    step_ids = torch.tensor([3, 3, 4, 4, 5, 5])
    env_ids = torch.tensor([0, 2, 0, 2, 1, 2])
    _assert_batches_equal(oracle.sample(step_ids, env_ids), replay.sample(step_ids, env_ids))
    assert not replay.sample(torch.tensor([2]), torch.tensor([0])).valid.item()
    replay.assert_no_errors()


@pytest.mark.parametrize("mode", tuple(ForwardBackwardAutoresetMode))
def test_all_autoreset_modes_produce_the_same_logical_done_successor(mode: ForwardBackwardAutoresetMode) -> None:
    """Returned reset observations must never replace a done edge's reached state."""
    observation_schema, transition_schema = _contracts(mode)
    _oracle, replay = _make_replays(3, 3, observation_schema, transition_schema)
    reached = {"state": _state(1, 3, offset=1000)}
    reset = {"state": _state(1, 3, offset=-1000)}
    replay.add(
        _transition(
            0,
            mode,
            observation_schema,
            transition_schema,
            truncated_envs=(1,),
            reached=reached,
            reset=reset,
        )
    )

    batch = replay.sample(torch.tensor([0]), torch.tensor([1]))
    torch.testing.assert_close(batch.next_observations["state"], reached["state"][[1]])
    assert batch.bootstrap_mask().item()


def test_missing_same_step_final_bootstraps_from_pre_step_with_provenance() -> None:
    """The no-final approximation should use current state and remain visible to the learner."""
    observation_schema, transition_schema = _contracts()
    _oracle, replay = _make_replays(3, 3, observation_schema, transition_schema)
    transition = _transition(
        0,
        transition_schema.autoreset_mode,
        observation_schema,
        transition_schema,
        truncated_envs=(1,),
        reached={"state": _state(1, 3, offset=1000)},
        reset={"state": _state(1, 3, offset=-1000)},
    )
    valid = transition.final_observation_valid.clone()
    valid[1] = False
    transition = replace(transition, final_observation_valid=valid)
    replay.add(transition)

    batch = replay.sample(torch.tensor([0]), torch.tensor([1]))
    torch.testing.assert_close(batch.next_observations["state"], transition.observations["state"][[1]])
    assert batch.bootstrap_mask().item()
    assert batch.successor_uses_current.item()
    replay.assert_no_errors()


def test_next_step_reset_only_rows_seed_nodes_but_never_enter_replay() -> None:
    """An ignored action during deferred reset should not become a replay edge."""
    mode = ForwardBackwardAutoresetMode.NEXT_STEP
    observation_schema, transition_schema = _contracts(mode)
    _oracle, replay = _make_replays(4, 4, observation_schema, transition_schema)
    replay.add(_transition(0, mode, observation_schema, transition_schema, done_envs=(0,)))
    applied = torch.ones(3, 1, dtype=torch.bool)
    applied[0] = False
    replay.add(_transition(1, mode, observation_schema, transition_schema, action_applied=applied))

    assert replay.sample(torch.tensor([0]), torch.tensor([0])).valid.item()
    assert not replay.sample(torch.tensor([1]), torch.tensor([0])).valid.item()
    assert replay.sample(torch.tensor([1]), torch.tensor([1])).valid.item()


def test_next_step_random_sampling_is_uniform_over_only_applied_edges() -> None:
    """Reset-only positions should be absent rather than rejected after sampling."""
    mode = ForwardBackwardAutoresetMode.NEXT_STEP
    observation_schema, transition_schema = _contracts(mode)
    _oracle, replay = _make_replays(4, 4, observation_schema, transition_schema)
    reset_envs = (None, 0, 1, 2, None, 1, 2, 0)
    expected_rewards = []
    for step, reset_env in enumerate(reset_envs):
        if step == replay.capacity_steps:
            expected_rewards.clear()
        applied = torch.ones(3, 1, dtype=torch.bool)
        if reset_env is not None:
            applied[reset_env] = False
        replay.add(
            _transition(
                step,
                mode,
                observation_schema,
                transition_schema,
                action_applied=applied,
            )
        )
        expected_rewards.extend(env + step / 10 for env in range(3) if env != reset_env)

    batch = replay.sample_random(90_000)
    rewards, counts = torch.unique(batch.environment_reward.squeeze(-1), return_counts=True)
    expected_count = batch.environment_reward.shape[0] / len(expected_rewards)

    assert torch.all(batch.valid)
    torch.testing.assert_close(rewards, torch.tensor(sorted(expected_rewards)))
    assert torch.all(torch.abs(counts - expected_count) < expected_count * 0.04)


def test_next_step_valid_sampling_does_not_change_primary_rng_sequence() -> None:
    """All-applied next-step streams should retain the ordinary sampler sequence."""
    mode = ForwardBackwardAutoresetMode.NEXT_STEP
    observation_schema, transition_schema = _contracts(mode)
    _oracle, replay = _make_replays(4, 4, observation_schema, transition_schema)
    for step in range(4):
        replay.add(_transition(step, mode, observation_schema, transition_schema))
    expected_generator = torch.Generator().manual_seed(7)
    step_ids = torch.randint(0, 4, (128,), generator=expected_generator)
    env_ids = torch.randint(3, (128,), generator=expected_generator)

    expected = replay.sample(step_ids, env_ids)
    actual = replay.sample_random(128)

    _assert_batches_equal(expected, actual)
    torch.testing.assert_close(replay.generator.get_state(), expected_generator.get_state(), rtol=0.0, atol=0.0)


def test_sparse_terminal_overflow_is_deferred_and_generation_safe() -> None:
    """Reusing a live per-environment terminal slot should fail at the control boundary."""
    observation_schema, transition_schema = _contracts()
    _oracle, replay = _make_replays(4, 1, observation_schema, transition_schema)
    replay.add(_transition(0, transition_schema.autoreset_mode, observation_schema, transition_schema, done_envs=(0,)))
    replay.add(_transition(1, transition_schema.autoreset_mode, observation_schema, transition_schema, done_envs=(0,)))

    assert not replay.sample(torch.tensor([0]), torch.tensor([0])).valid.item()
    assert replay.sample(torch.tensor([1]), torch.tensor([0])).valid.item()
    with pytest.raises(RuntimeError, match="terminal capacity"):
        replay.assert_no_errors()


def test_all_done_rows_remain_exact_at_terminal_capacity_bound() -> None:
    """Worst-case one-step episodes should remain exact when the terminal bound is dense."""
    observation_schema, transition_schema = _contracts()
    oracle, replay = _make_replays(3, 3, observation_schema, transition_schema)
    current = {"state": _state(0, 3)}
    for step in range(6):
        reached = {"state": _state(step + 1, 3, offset=1000 + step)}
        reset = {"state": _state(step + 1, 3, offset=-1000 - step)}
        transition = _transition(
            step,
            transition_schema.autoreset_mode,
            observation_schema,
            transition_schema,
            done_envs=(0, 1, 2),
            current=current,
            reached=reached,
            reset=reset,
        )
        oracle.add(transition)
        replay.add(transition)
        current = reset

    step_ids = torch.tensor([3, 3, 4, 4, 5, 5])
    env_ids = torch.tensor([0, 2, 0, 1, 1, 2])
    _assert_batches_equal(oracle.sample(step_ids, env_ids), replay.sample(step_ids, env_ids))
    replay.assert_no_errors()


def _history_layout() -> ForwardBackwardHistoryLayout:
    source = ForwardBackwardHistoryLayout.Source
    return ForwardBackwardHistoryLayout(
        history_field="history_actor",
        history_length=2,
        sources=(source(None, 0, 2), source("state", 1, 3)),
        last_action_field="last_action",
    )


def _history_observation(
    step: int,
    states: list[torch.Tensor],
    actions: list[torch.Tensor],
    episode_start: int,
) -> dict[str, torch.Tensor]:
    state = states[step]
    zeros_action = torch.zeros_like(actions[0])
    zeros_state = torch.zeros_like(state[:, 1:3])
    last_action = actions[step - 1] if step - 1 >= episode_start else zeros_action
    action_history = [actions[step - lag] if step - lag >= episode_start else zeros_action for lag in (1, 2)]
    state_history = [states[step - lag][:, 1:3] if step - lag >= episode_start else zeros_state for lag in (1, 2)]
    return {
        "state": state,
        "last_action": last_action,
        "history_actor": torch.cat((*action_history, *state_history), dim=-1),
    }


def test_versioned_history_reconstruction_matches_dense_emitted_noise_after_wrap() -> None:
    """Compact history must use retained emitted tensors, including their noise."""
    fields = {"state": 4, "last_action": 2, "history_actor": 8}
    observation_schema, transition_schema = _contracts(field_widths=fields)
    layout = _history_layout()
    oracle, replay = _make_replays(3, 3, observation_schema, transition_schema, layout)
    generator = torch.Generator().manual_seed(19)
    states = [torch.randn(3, 4, generator=generator) + step * 10 for step in range(7)]
    actions = [torch.randn(3, 2, generator=generator) + step for step in range(6)]
    for step in range(6):
        current = _history_observation(step, states, actions, 0)
        reached = _history_observation(step + 1, states, actions, 0)
        transition = replace(
            _transition(
                step,
                transition_schema.autoreset_mode,
                observation_schema,
                transition_schema,
                current=current,
                reached=reached,
            ),
            actions=actions[step],
        )
        oracle.add(transition)
        replay.add(transition)

    step_ids = torch.tensor([3, 3, 4, 5, 5])
    env_ids = torch.tensor([0, 2, 1, 0, 2])
    _assert_batches_equal(oracle.sample(step_ids, env_ids), replay.sample(step_ids, env_ids))
    dense_replay = ForwardBackwardReplay(
        3,
        3,
        3,
        observation_schema,
        transition_schema,
        make_reward_schema(),
        "cpu",
    )
    assert replay.storage_bytes() < dense_replay.storage_bytes()


def _bfm_history_layout() -> ForwardBackwardHistoryLayout:
    source = ForwardBackwardHistoryLayout.Source
    return ForwardBackwardHistoryLayout(
        history_field="history_actor",
        history_length=4,
        sources=(
            source(None, 0, 29),
            source("state", 61, 64),
            source("state", 0, 29),
            source("state", 29, 58),
            source("state", 58, 61),
        ),
        last_action_field="last_action",
    )


def _bfm_observation(
    step: int,
    states: list[torch.Tensor],
    privileged_states: list[torch.Tensor],
    actions: list[torch.Tensor],
) -> dict[str, torch.Tensor]:
    zeros_action = torch.zeros_like(actions[0])
    history_parts = []
    for source in _bfm_history_layout().sources:
        for lag in range(1, 5):
            source_step = step - lag
            if source_step < 0:
                width = source.stop - source.start
                history_parts.append(torch.zeros(states[0].shape[0], width))
            elif source.observation_name is None:
                history_parts.append(actions[source_step][:, source.start : source.stop])
            else:
                history_parts.append(states[source_step][:, source.start : source.stop])
    return {
        "state": states[step],
        "privileged_state": privileged_states[step],
        "last_action": actions[step - 1] if step > 0 else zeros_action,
        "history_actor": torch.cat(history_parts, dim=-1),
    }


def test_bfm_field_major_history_layout_matches_dense_oracle() -> None:
    """The compact layout should reproduce BFM's sorted field-major, newest-first history."""
    fields = {"state": 64, "last_action": 29, "history_actor": 372, "privileged_state": 463}
    observation_schema, transition_schema = _contracts(field_widths=fields, action_width=29)
    layout = _bfm_history_layout()
    oracle, replay = _make_replays(3, 3, observation_schema, transition_schema, layout)
    generator = torch.Generator().manual_seed(23)
    states = [torch.randn(3, 64, generator=generator) + step for step in range(7)]
    privileged_states = [torch.randn(3, 463, generator=generator) + step for step in range(7)]
    actions = [torch.randn(3, 29, generator=generator) + step for step in range(6)]
    for step in range(6):
        transition = replace(
            _transition(
                step,
                transition_schema.autoreset_mode,
                observation_schema,
                transition_schema,
                current=_bfm_observation(step, states, privileged_states, actions),
                reached=_bfm_observation(step + 1, states, privileged_states, actions),
            ),
            actions=actions[step],
        )
        oracle.add(transition)
        replay.add(transition)

    step_ids = torch.tensor([3, 3, 4, 4, 5, 5])
    env_ids = torch.tensor([0, 2, 0, 1, 1, 2])
    _assert_batches_equal(oracle.sample(step_ids, env_ids), replay.sample(step_ids, env_ids))


def test_history_and_last_action_zero_pad_after_reset_but_final_keeps_old_episode() -> None:
    """The final state should advance old history while the reset state starts from zero."""
    fields = {"state": 4, "last_action": 2, "history_actor": 8}
    observation_schema, transition_schema = _contracts(field_widths=fields)
    layout = _history_layout()
    oracle, replay = _make_replays(5, 5, observation_schema, transition_schema, layout)
    states = [_state(step, 3, 4) for step in range(5)]
    actions = [_state(step, 3, 2, offset=0.5) for step in range(4)]
    for step in range(2):
        current = _history_observation(step, states, actions, 0)
        reached = _history_observation(step + 1, states, actions, 0)
        transition = replace(
            _transition(
                step,
                transition_schema.autoreset_mode,
                observation_schema,
                transition_schema,
                current=current,
                reached=reached,
            ),
            actions=actions[step],
        )
        oracle.add(transition)
        replay.add(transition)

    current = _history_observation(2, states, actions, 0)
    final = _history_observation(3, states, actions, 0)
    reset = {
        "state": states[3] + 1000,
        "last_action": torch.zeros(3, 2),
        "history_actor": torch.zeros(3, 8),
    }
    done_transition = replace(
        _transition(
            2,
            transition_schema.autoreset_mode,
            observation_schema,
            transition_schema,
            done_envs=(0,),
            current=current,
            reached=final,
            reset=reset,
        ),
        actions=actions[2],
    )
    oracle.add(done_transition)
    replay.add(done_transition)
    next_current = {name: value.clone() for name, value in final.items()}
    for name in reset:
        next_current[name][0] = reset[name][0]
    next_reached = _history_observation(4, states, actions, 3)
    next_reached["state"][0] = states[4][0] + 1000
    next_transition = replace(
        _transition(
            3,
            transition_schema.autoreset_mode,
            observation_schema,
            transition_schema,
            current=next_current,
            reached=next_reached,
        ),
        actions=actions[3],
    )
    oracle.add(next_transition)
    replay.add(next_transition)

    _assert_batches_equal(
        oracle.sample(torch.tensor([2]), torch.tensor([0])), replay.sample(torch.tensor([2]), torch.tensor([0]))
    )
    reset_batch = replay.sample(torch.tensor([3]), torch.tensor([0]))
    torch.testing.assert_close(reset_batch.observations["last_action"], torch.zeros(1, 2))
    torch.testing.assert_close(reset_batch.observations["history_actor"], torch.zeros(1, 8))


def test_replay_state_restores_exact_next_random_sample() -> None:
    """Storage generations and the device RNG should resume exactly."""
    observation_schema, transition_schema = _contracts()
    _oracle, replay = _make_replays(4, 4, observation_schema, transition_schema)
    for step in range(4):
        replay.add(_transition(step, transition_schema.autoreset_mode, observation_schema, transition_schema))
    replay.sample_random(7)
    state = replay.state_dict()
    _other_oracle, restored = _make_replays(4, 4, observation_schema, transition_schema)
    restored.load_state_dict(state)

    _assert_batches_equal(replay.sample_random(11), restored.sample_random(11))
    assert replay.storage_bytes() == restored.storage_bytes()


def test_next_step_replay_state_restores_valid_population_and_rng() -> None:
    """A resumed sampler should reconstruct holes and continue its correction RNG exactly."""
    mode = ForwardBackwardAutoresetMode.NEXT_STEP
    observation_schema, transition_schema = _contracts(mode)
    _oracle, replay = _make_replays(4, 4, observation_schema, transition_schema)
    for step, reset_env in enumerate((None, 0, 1, 2)):
        applied = torch.ones(3, 1, dtype=torch.bool)
        if reset_env is not None:
            applied[reset_env] = False
        replay.add(_transition(step, mode, observation_schema, transition_schema, action_applied=applied))
    replay.sample_random(17)
    state = replay.state_dict()
    _other_oracle, restored = _make_replays(4, 4, observation_schema, transition_schema)
    restored.load_state_dict(state)

    expected = replay.sample_random(257)
    actual = restored.sample_random(257)

    _assert_batches_equal(expected, actual)
    assert torch.all(actual.valid)


def test_replay_state_rejects_history_layout_mismatch() -> None:
    """A checkpoint must not reinterpret history under another layout."""
    fields = {"state": 4, "last_action": 2, "history_actor": 8}
    observation_schema, transition_schema = _contracts(field_widths=fields)
    _oracle, replay = _make_replays(3, 3, observation_schema, transition_schema, _history_layout())
    state = replay.state_dict()
    changed_layout = replace(_history_layout(), version=1, sources=tuple(reversed(_history_layout().sources)))
    _other_oracle, restored = _make_replays(3, 3, observation_schema, transition_schema, changed_layout)

    with pytest.raises(ValueError, match="incompatible"):
        restored.load_state_dict(state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fixed_shape_cuda_sampling_is_graph_capturable() -> None:
    """Fixed logical indices should replay through a CUDA graph without host work."""
    observation_schema, transition_schema = _contracts()
    replay = ForwardBackwardReplay(
        3,
        3,
        3,
        observation_schema,
        transition_schema,
        make_reward_schema(),
        "cuda",
    )
    for step in range(3):
        transition = _transition(step, transition_schema.autoreset_mode, observation_schema, transition_schema)
        transition = replace(
            transition,
            observations=transition.observations.cuda(),
            next_observations=transition.next_observations.cuda(),
            final_observations=transition.final_observations.cuda(),
            actions=transition.actions.cuda(),
            behavior_context=transition.behavior_context.cuda(),
            environment_reward=transition.environment_reward.cuda(),
            auxiliary_reward_evidence=transition.auxiliary_reward_evidence.cuda(),
            terminated=transition.terminated.cuda(),
            truncated=transition.truncated.cuda(),
            context_changed=transition.context_changed.cuda(),
            action_applied=transition.action_applied.cuda(),
            final_observation_valid=transition.final_observation_valid.cuda(),
        )
        replay.add(transition)
    step_ids = torch.tensor([0, 1, 2], device="cuda")
    env_ids = torch.tensor([0, 1, 2], device="cuda")
    replay.sample(step_ids, env_ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        batch = replay.sample(step_ids, env_ids)
    graph.replay()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    assert torch.all(batch.valid)
