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
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras)
            collect_time = time.time() - start

            start = time.time()
            metrics: list[dict[str, float]] = []
            if self.alg.ready_to_update:
                self.alg.validate_collection()
                metrics = [self.alg.update() for _ in range(self.num_updates_per_iteration)]
            loss_dict = self._mean_metrics(metrics)
            learn_time = time.time() - start
            self.current_learning_iteration = it

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

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore[arg-type]
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save learner state and exact environment state when the env exposes it."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        saved_dict["environment_resume"] = "exact" if self.environment_resume_exact else "restart"
        saved_dict["environment_state_dict"] = self.env.state_dict() if self.environment_resume_exact else None
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
        return loaded_dict["infos"]

    @staticmethod
    def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
        """Average repeated-update metrics without imposing a logging schema."""
        if not metrics:
            return {}
        names = metrics[0].keys()
        if any(sample.keys() != names for sample in metrics[1:]):
            raise RuntimeError("Off-policy updates returned inconsistent metric keys.")
        return {name: sum(sample[name] for sample in metrics) / len(metrics) for name in names}
