# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic fixed-row off-policy environment runner."""

from __future__ import annotations

import math
import os
import time
import torch
from tensordict import TensorDictBase

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import check_nan


class OffPolicyRunner(OnPolicyRunner):
    """Collect fixed vector rows and update an off-policy algorithm when ready.

    The algorithm owns replay, behavior selection, collection semantics, and
    readiness. The runner owns collection cadence, repeated updates, logging,
    and checkpoints.
    """

    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cpu",
    ) -> None:
        """Construct the ordinary runner stack and retain update cadence."""
        super().__init__(env, train_cfg, log_dir, device)
        self.environment_resume_exact = callable(getattr(env, "state_dict", None)) and callable(
            getattr(env, "load_state_dict", None)
        )
        self.num_updates_per_iteration = int(self.cfg["num_updates_per_iteration"])
        if self.num_updates_per_iteration < 1:
            raise ValueError("num_updates_per_iteration must be positive.")
        self.collected_transitions = 0
        self._update_calls = 0
        self._metric_names: tuple[str, ...] = ()
        self._all_metrics_finite = True
        self._last_metrics: dict[str, float] = {}
        self._checkpoint_load: dict[str, object] | None = None

    def _update(
        self,
        observations: TensorDictBase,
    ) -> tuple[TensorDictBase, list[dict[str, torch.Tensor]]]:
        """Run the configured learner updates and retain current observations."""
        metrics: list[dict[str, torch.Tensor]] = []
        if self.alg.ready_to_update:
            metrics = [self.alg.update() for _ in range(self.num_updates_per_iteration)]
            self._update_calls += len(metrics)
        return observations, metrics

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Alternate fixed-size collection blocks with zero or more updates."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            raise NotImplementedError("Off-policy multi-GPU synchronization is not implemented.")
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        last_saved_iteration: int | None = None
        for it in range(start_it, total_it):
            start = time.time()
            with torch.no_grad():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras)
                    self.collected_transitions += self.env.num_envs
            collect_time = time.time() - start

            start = time.time()
            obs, metrics = self._update(obs)
            loss_dict = self._mean_metrics(metrics)
            if loss_dict:
                names = tuple(loss_dict)
                if self._metric_names and names != self._metric_names:
                    raise RuntimeError("Off-policy updates changed metric keys between iterations.")
                self._metric_names = names
                self._all_metrics_finite &= all(math.isfinite(value) for value in loss_dict.values())
                self._last_metrics = loss_dict
            learn_time = time.time() - start
            self.current_learning_iteration = it + 1

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.action_std,
                rnd_weight=None,
            )

            if self.logger.writer is not None and self.current_learning_iteration % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
                obs = self.env.get_observations().to(self.device)
                last_saved_iteration = self.current_learning_iteration

        if self.logger.writer is not None:
            if self.current_learning_iteration != last_saved_iteration:
                self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    def state_dict(self) -> dict[str, object]:
        """Return learner, runner, and exact environment state."""
        state = self.alg.save()
        state["iter"] = self.current_learning_iteration
        state["environment_resume"] = "exact" if self.environment_resume_exact else "restart"
        state["environment_state_dict"] = self.env.state_dict() if self.environment_resume_exact else None
        state["collected_transitions"] = self.collected_transitions
        return state

    def load_state_dict(
        self,
        state_dict: dict[str, object],
        load_cfg: dict | None = None,
        strict: bool = True,
    ) -> None:
        """Restore learner, runner, and exact environment state."""
        load_iteration = self.alg.load(state_dict, load_cfg, strict)
        environment_state = state_dict.get("environment_state_dict")
        if environment_state is not None:
            if not self.environment_resume_exact:
                raise RuntimeError("Checkpoint contains environment state but this environment cannot restore it.")
            self.env.load_state_dict(environment_state)
        if load_iteration:
            self.current_learning_iteration = int(state_dict["iter"])
        self.collected_transitions = int(state_dict["collected_transitions"])

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the complete runner state."""
        state = self.state_dict()
        state["infos"] = infos
        torch.save(state, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
        mmap: bool | None = None,
    ) -> dict | None:
        """Load the complete runner state."""
        state = torch.load(path, weights_only=False, map_location=map_location, mmap=mmap)
        self.load_state_dict(state, load_cfg, strict)
        self._checkpoint_load = {
            "environment_resume": state.get("environment_resume"),
            "environment_state_dict_is_none": state.get("environment_state_dict") is None,
            "map_location": map_location,
            "mmap": mmap,
            "strict": strict,
        }
        return state["infos"]

    def training_summary(self) -> dict[str, object]:
        """Return materialized counters and metrics from completed runner work."""
        return {
            "completed_iterations": self.current_learning_iteration,
            "collected_transitions": self.collected_transitions,
            "update_calls": self._update_calls,
            "metric_names": list(self._metric_names),
            "all_metrics_finite": self._all_metrics_finite,
            "last_metrics": dict(self._last_metrics),
        }

    def checkpoint_load_summary(self) -> dict[str, object]:
        """Return resume semantics observed during the most recent checkpoint load."""
        if self._checkpoint_load is None:
            raise RuntimeError("No checkpoint has been loaded by this runner.")
        return dict(self._checkpoint_load)

    @staticmethod
    def _mean_metrics(metrics: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        """Average GPU metrics and materialize them at one logging boundary."""
        if not metrics:
            return {}
        names = tuple(metrics[0])
        if any(tuple(sample) != names for sample in metrics[1:]):
            raise RuntimeError("Off-policy updates returned inconsistent metric keys.")
        values = torch.stack(tuple(sample[name] for sample in metrics for name in names)).view(len(metrics), len(names))
        means = values.mean(dim=0).tolist()
        return dict(zip(names, means))
