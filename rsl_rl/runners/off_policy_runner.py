# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generic fixed-row off-policy environment runner."""

from __future__ import annotations

import os
import time
import torch

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import check_nan


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
            iteration_start_transitions = self.collected_transitions
            start = time.time()
            with torch.inference_mode():
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
            loss_dict = self._mean_metrics(metrics)
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
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict | None:
        """Restore learner and environment state or retain explicit restart semantics."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        environment_state = loaded_dict.get("environment_state_dict")
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
        return loaded_dict["infos"]

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
