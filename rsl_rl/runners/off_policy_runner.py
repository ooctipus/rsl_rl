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
from rsl_rl.runners.lifecycle import RunnerLifecycleExtension
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import check_nan, resolve_callable


class OffPolicyRunner(OnPolicyRunner):
    """Collect fixed vector rows and update an off-policy algorithm when ready.

    The algorithm owns replay, collection semantics, and readiness. The runner
    owns only cadence, logging, and the checkpoint boundary.
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
        self.random_action_steps = int(self.cfg.get("random_action_steps", 0))
        if self.random_action_steps < 0:
            raise ValueError("random_action_steps must be non-negative.")
        if self.random_action_steps and not callable(getattr(self.alg, "act_random", None)):
            raise TypeError("random_action_steps requires an algorithm with act_random().")
        self.collected_transitions = 0
        self._update_calls = 0
        self._metric_names: tuple[str, ...] = ()
        self._all_metrics_finite = True
        self._last_metrics: dict[str, float] = {}
        self._checkpoint_load: dict[str, object] | None = None
        self.lifecycle_extension: RunnerLifecycleExtension | None = None
        self.lifecycle_transition_interval: int | None = None
        self._lifecycle_last_transition: int | None = None
        extension_cfg = self.cfg.get("lifecycle_extension")
        if extension_cfg is not None:
            extension_cfg = dict(extension_cfg)
            extension_class = resolve_callable(extension_cfg.pop("class_name"))
            interval = int(extension_cfg.pop("transition_interval"))
            collection_block = self.env.num_envs * int(self.cfg["num_steps_per_env"])
            if interval < 1 or interval % collection_block:
                raise ValueError("lifecycle transition_interval must be a positive multiple of one collection block.")
            extension = extension_class(self.env, self.alg, log_dir, self.device, **extension_cfg)
            if not isinstance(extension, RunnerLifecycleExtension):
                raise TypeError("The configured lifecycle extension must derive from RunnerLifecycleExtension.")
            self.lifecycle_extension = extension
            self.lifecycle_transition_interval = interval

    def _observe_iteration_start(self, iteration: int, start_transitions: int) -> None:
        """Observe the exact boundary immediately before one collection block.

        Args:
            iteration: Zero-based runner iteration.
            start_transitions: Environment transitions collected before this iteration.
        """

    def _observe_iteration_learning_complete(self, iteration: int, end_transitions: int) -> None:
        """Observe the exact boundary after updates and before metric materialization.

        Args:
            iteration: Zero-based runner iteration.
            end_transitions: Environment transitions collected through this iteration.
        """

    def _observe_iteration_complete(
        self,
        iteration: int,
        end_transitions: int,
        collect_time: float,
        learn_time: float,
    ) -> None:
        """Observe existing decomposition timers before logging and checkpointing.

        Args:
            iteration: Zero-based runner iteration.
            end_transitions: Environment transitions collected through this iteration.
            collect_time: Collection wall time [s].
            learn_time: Update and metric-materialization wall time [s].
        """

    def _run_lifecycle_extension(self, transition: int, observations: TensorDictBase) -> TensorDictBase:
        """Run one due extension event and adopt any returned reset observations."""
        extension = self.lifecycle_extension
        interval = self.lifecycle_transition_interval
        if extension is None or interval is None or (transition and transition % interval):
            return observations
        if self._lifecycle_last_transition is not None and transition <= self._lifecycle_last_transition:
            if transition == self._lifecycle_last_transition:
                return observations
            raise RuntimeError("Lifecycle transitions must increase monotonically.")
        reset_observations = extension.on_transition(transition)
        self._lifecycle_last_transition = transition
        if reset_observations is None:
            return observations
        if tuple(reset_observations.batch_size) != (self.env.num_envs,):
            raise ValueError("Lifecycle reset observations must contain one row per environment.")
        return reset_observations.to(self.device)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Alternate fixed-size collection blocks with zero or more updates."""
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        if self._lifecycle_last_transition is None:
            obs = self._run_lifecycle_extension(0, obs)
        self.alg.train_mode()
        if self.is_distributed:
            raise NotImplementedError("Off-policy multi-GPU synchronization is not implemented.")
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        last_saved_iteration: int | None = None
        for it in range(start_it, total_it):
            iteration_start_transitions = self.collected_transitions
            self._observe_iteration_start(it, iteration_start_transitions)
            start = time.time()
            with torch.no_grad():
                for _ in range(self.cfg["num_steps_per_env"]):
                    if self.collected_transitions < self.random_action_steps:
                        actions = self.alg.act_random(obs)
                    else:
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
            metrics: list[dict[str, torch.Tensor]] = []
            seed_phase_complete = not self.random_action_steps or iteration_start_transitions > self.random_action_steps
            if self.alg.ready_to_update and seed_phase_complete:
                self.alg.validate_collection()
                metrics = [self.alg.update() for _ in range(self.num_updates_per_iteration)]
                self._update_calls += len(metrics)
            obs = self._run_lifecycle_extension(self.collected_transitions, obs)
            self._observe_iteration_learning_complete(it, self.collected_transitions)
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
            self._observe_iteration_complete(it, self.collected_transitions, collect_time, learn_time)

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
                self.save(  # type: ignore[arg-type]
                    os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
                )
                obs = self.env.get_observations().to(self.device)
                last_saved_iteration = self.current_learning_iteration

        if self.logger.writer is not None:
            if self.current_learning_iteration != last_saved_iteration:
                self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore[arg-type]
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save learner state and exact environment state when the env exposes it."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        saved_dict["environment_resume"] = "exact" if self.environment_resume_exact else "restart"
        saved_dict["environment_state_dict"] = self.env.state_dict() if self.environment_resume_exact else None
        saved_dict["collected_transitions"] = self.collected_transitions
        saved_dict["lifecycle_extension"] = (
            None
            if self.lifecycle_extension is None
            else {
                "last_transition": self._lifecycle_last_transition,
                "state_dict": self.lifecycle_extension.state_dict(),
            }
        )
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
        mmap: bool | None = None,
    ) -> dict | None:
        """Restore learner and environment state or retain explicit restart semantics."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location, mmap=mmap)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        environment_state = loaded_dict.get("environment_state_dict")
        checkpoint_load = {
            "environment_resume": loaded_dict.get("environment_resume"),
            "environment_state_dict_is_none": environment_state is None,
            "map_location": map_location,
            "mmap": mmap,
            "strict": strict,
        }
        if environment_state is not None:
            if not self.environment_resume_exact:
                raise RuntimeError("Checkpoint contains environment state but this environment cannot restore it.")
            self.env.load_state_dict(environment_state)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        legacy_vector_steps = loaded_dict.get("rollout_schedule_step")
        if legacy_vector_steps is None:
            legacy_vector_steps = (self.current_learning_iteration + 1) * self.cfg["num_steps_per_env"]
        self.collected_transitions = int(
            loaded_dict.get("collected_transitions", int(legacy_vector_steps) * self.env.num_envs)
        )
        extension_state = loaded_dict.get("lifecycle_extension")
        if self.lifecycle_extension is None:
            if extension_state is not None:
                raise ValueError("Checkpoint lifecycle extension is not configured by this runner.")
        else:
            if not isinstance(extension_state, dict):
                raise ValueError("Checkpoint is missing configured lifecycle extension state.")
            last_transition = extension_state.get("last_transition")
            if last_transition is None:
                if self.collected_transitions:
                    raise ValueError("A checkpoint without a lifecycle event cannot contain transitions.")
            elif not isinstance(last_transition, int) or last_transition < 0:
                raise ValueError("Checkpoint lifecycle last_transition must be null or a non-negative integer.")
            else:
                interval = self.lifecycle_transition_interval
                if interval is None or (last_transition and last_transition % interval):
                    raise ValueError("Checkpoint lifecycle transition is outside the configured cadence.")
                if last_transition > self.collected_transitions:
                    raise ValueError("Checkpoint lifecycle transition exceeds collected transitions.")
            state = extension_state.get("state_dict")
            if not isinstance(state, dict):
                raise TypeError("Checkpoint lifecycle state_dict must be a dictionary.")
            self.lifecycle_extension.load_state_dict(state)
            self._lifecycle_last_transition = last_transition
        self._checkpoint_load = checkpoint_load
        return loaded_dict["infos"]

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
