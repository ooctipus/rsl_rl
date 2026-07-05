# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forward-backward runner with an optional configured tracking curriculum."""

from __future__ import annotations

import torch
from collections.abc import Mapping
from tensordict import TensorDictBase
from typing import Protocol

from rsl_rl.env import VecEnv
from rsl_rl.runners.off_policy_runner import OffPolicyRunner
from rsl_rl.utils import resolve_callable


class _TrackingCurriculum(Protocol):
    """Configured tracking work owned by an environment connector."""

    def update(self) -> TensorDictBase:
        """Evaluate tracking, update sampling state, and reset the environment."""

    def state_dict(self) -> dict[str, object]:
        """Return provider-owned checkpoint state."""

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        """Restore provider-owned checkpoint state."""


class ForwardBackwardRunner(OffPolicyRunner):
    """Run forward-backward learning with optional periodic tracking work."""

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        """Construct the learner and its explicitly configured tracking provider."""
        super().__init__(env, train_cfg, log_dir, device)
        self.tracking_curriculum: _TrackingCurriculum | None = None
        self.tracking_interval_transitions: int | None = None
        self._tracking_last_transition: int | None = None

        configured = self.cfg.get("tracking_curriculum")
        if configured is None:
            return
        if not isinstance(configured, Mapping):
            raise TypeError("tracking_curriculum must be null or a mapping.")
        curriculum_cfg = dict(configured)
        if "class_type" in curriculum_cfg:
            raise ValueError("tracking_curriculum uses class_name as its single provider boundary.")
        provider = resolve_callable(curriculum_cfg.pop("class_name"))
        interval = curriculum_cfg.pop("interval_transitions")
        if type(interval) is not int:
            raise TypeError("Tracking curriculum interval_transitions must be an integer.")
        collection_block = self.env.num_envs * int(self.cfg["num_steps_per_env"])
        if interval < 1 or interval % collection_block:
            raise ValueError("Tracking curriculum interval must be a positive multiple of one collection block.")
        curriculum = provider(self.env, self.alg, self.device, **curriculum_cfg)
        if not all(callable(getattr(curriculum, name, None)) for name in ("update", "state_dict", "load_state_dict")):
            raise TypeError(
                "Tracking curriculum provider must implement update(), state_dict(), and load_state_dict()."
            )
        self.tracking_curriculum = curriculum
        self.tracking_interval_transitions = interval

    def _run_tracking_curriculum(
        self,
        transition: int,
        observations: TensorDictBase | None,
    ) -> TensorDictBase:
        """Run one due tracking update and return current reset observations."""
        curriculum = self.tracking_curriculum
        interval = self.tracking_interval_transitions
        if curriculum is None or interval is None:
            raise RuntimeError("Tracking curriculum work requires a configured provider.")
        if self._tracking_last_transition is not None and transition <= self._tracking_last_transition:
            raise RuntimeError("Tracking curriculum transitions must increase monotonically.")
        if transition and transition % interval:
            if observations is None:
                raise ValueError("Periodic tracking requires current observations when no update is due.")
            return observations
        reset_observations = curriculum.update()
        if not isinstance(reset_observations, TensorDictBase):
            raise TypeError("Tracking curriculum update must return TensorDict observations.")
        if tuple(reset_observations.batch_size) != (self.env.num_envs,):
            raise ValueError("Tracking reset observations must contain one row per environment.")
        self._tracking_last_transition = transition
        return reset_observations.to(self.device)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run transition-zero tracking once, then use the ordinary off-policy loop."""
        if self.tracking_curriculum is not None and self._tracking_last_transition is None:
            self._run_tracking_curriculum(0, None)
        super().learn(num_learning_iterations, init_at_random_ep_len)

    def _update(
        self,
        observations: TensorDictBase,
    ) -> tuple[TensorDictBase, list[dict[str, torch.Tensor]]]:
        """Update the learner, then run tracking at due transition counts."""
        observations, metrics = super()._update(observations)
        if self.tracking_curriculum is not None:
            observations = self._run_tracking_curriculum(self.collected_transitions, observations)
        return observations, metrics

    def state_dict(self) -> dict[str, object]:
        """Extend the ordinary runner checkpoint with tracking cadence and provider state."""
        state_dict = super().state_dict()
        state_dict["tracking_curriculum"] = (
            None
            if self.tracking_curriculum is None
            else {
                "last_transition": self._tracking_last_transition,
                "state_dict": self.tracking_curriculum.state_dict(),
            }
        )
        return state_dict

    def load_state_dict(
        self,
        state_dict: dict[str, object],
        load_cfg: dict | None = None,
        strict: bool = True,
    ) -> None:
        """Restore learner state followed by the configured tracking provider state."""
        super().load_state_dict(state_dict, load_cfg, strict)
        tracking_state = state_dict["tracking_curriculum"]
        curriculum = self.tracking_curriculum
        interval = self.tracking_interval_transitions
        if curriculum is None:
            if tracking_state is not None:
                raise ValueError("Checkpoint tracking curriculum differs from the configured runner.")
            return
        if interval is None or not isinstance(tracking_state, dict):
            raise ValueError("Checkpoint is missing configured tracking curriculum state.")
        if set(tracking_state) != {"last_transition", "state_dict"}:
            raise ValueError("Checkpoint tracking curriculum fields differ from the runner contract.")
        last_transition = tracking_state["last_transition"]
        if last_transition is not None:
            if type(last_transition) is not int or last_transition < 0:
                raise ValueError("Tracking curriculum last_transition must be null or a non-negative integer.")
            if last_transition and last_transition % interval:
                raise ValueError("Checkpoint tracking transition is outside the configured cadence.")
            if last_transition > self.collected_transitions:
                raise ValueError("Checkpoint tracking transition exceeds collected transitions.")
        elif self.collected_transitions:
            raise ValueError("A checkpoint without a tracking event cannot contain transitions.")
        curriculum_state = tracking_state["state_dict"]
        if not isinstance(curriculum_state, dict):
            raise TypeError("Checkpoint tracking curriculum state_dict must be a dictionary.")
        curriculum.load_state_dict(curriculum_state)
        self._tracking_last_transition = last_transition
