# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the forward-backward runner's optional tracking curriculum."""

import torch
from tensordict import TensorDict, TensorDictBase
from types import SimpleNamespace

import pytest

from rsl_rl.runners import ForwardBackwardRunner
from rsl_rl.runners.off_policy_runner import OffPolicyRunner


class FakeTrackingCurriculum:
    """Small dynamically resolved tracking provider."""

    def __init__(self, env: "_Env", algorithm: object, device: str, *, marker: int) -> None:
        """Retain the generic constructor arguments and provider marker."""
        self.env = env
        self.algorithm = algorithm
        self.device = device
        self.marker = marker
        self.updates = 0
        self.loaded: dict[str, object] | None = None

    def update(self) -> TensorDictBase:
        """Count one update and return reset observations."""
        self.updates += 1
        return self.env.get_observations()

    def state_dict(self) -> dict[str, object]:
        """Return the provider marker and update count."""
        return {"marker": self.marker, "updates": self.updates}

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore and retain provider state."""
        self.loaded = state_dict
        self.updates = int(state_dict["updates"])


class _Env:
    num_envs = 2

    def get_observations(self) -> TensorDictBase:
        return TensorDict({"state": torch.zeros(2, 1)}, batch_size=[2])


def _runner(monkeypatch: pytest.MonkeyPatch, *, interval: int = 8) -> ForwardBackwardRunner:
    def initialize_base(
        self: OffPolicyRunner,
        env: _Env,
        train_cfg: dict[str, object],
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        del log_dir
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.alg = SimpleNamespace()
        self.collected_transitions = 0

    monkeypatch.setattr(OffPolicyRunner, "__init__", initialize_base)
    return ForwardBackwardRunner(
        _Env(),
        {
            "num_steps_per_env": 2,
            "tracking_curriculum": {
                "class_name": "tests.runners.test_forward_backward_runner:FakeTrackingCurriculum",
                "interval_transitions": interval,
                "marker": 17,
            },
        },
        device="cpu",
    )


def test_runner_constructs_tracking_provider_from_class_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Construct the configured provider through its import path."""
    runner = _runner(monkeypatch)

    assert isinstance(runner.tracking_curriculum, FakeTrackingCurriculum)
    assert runner.tracking_curriculum.marker == 17
    assert runner.tracking_interval_transitions == 8


def test_runner_owns_transition_zero_and_periodic_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run tracking at transition zero and exact periodic boundaries."""
    runner = _runner(monkeypatch)
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(
        OffPolicyRunner,
        "learn",
        lambda _self, iterations, _random=False: calls.append(("collection", iterations)),
    )
    runner.learn(3)
    assert runner.tracking_curriculum is not None
    assert runner.tracking_curriculum.updates == 1
    assert calls == [("collection", 3)]

    current = TensorDict({"state": torch.ones(2, 1)}, batch_size=[2])
    monkeypatch.setattr(OffPolicyRunner, "_update", lambda _self, observations: (observations, []))
    runner.collected_transitions = 4
    returned, _ = runner._update(current)
    assert returned is current
    assert runner.tracking_curriculum.updates == 1
    runner.collected_transitions = 8
    returned, _ = runner._update(current)
    assert returned is not current
    assert runner.tracking_curriculum.updates == 2
    with pytest.raises(RuntimeError, match="increase monotonically"):
        runner._run_tracking_curriculum(8, current)


def test_runner_round_trips_tracking_checkpoint_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round-trip provider cadence and provider-owned checkpoint state."""
    runner = _runner(monkeypatch)
    runner._run_tracking_curriculum(0, None)
    monkeypatch.setattr(OffPolicyRunner, "state_dict", lambda _self: {"base": 1})
    state = runner.state_dict()

    assert state["tracking_curriculum"] == {
        "last_transition": 0,
        "state_dict": {"marker": 17, "updates": 1},
    }

    restored = _runner(monkeypatch)
    restored.collected_transitions = 8
    monkeypatch.setattr(OffPolicyRunner, "load_state_dict", lambda *_args, **_kwargs: None)
    restored.load_state_dict({
        "base": 1,
        "tracking_curriculum": {
            "last_transition": 8,
            "state_dict": {"marker": 17, "updates": 3},
        },
    })
    assert restored._tracking_last_transition == 8
    assert restored.tracking_curriculum is not None
    assert restored.tracking_curriculum.loaded == {"marker": 17, "updates": 3}
