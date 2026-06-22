# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Off-policy runner for FB-CPR: seed phase (random actions) -> interleaved collect/gradient-step
# loop, with TensorBoard logging + checkpoints. Mirrors metamotivo's online training cadence.

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithms.fbcpr import FBCPR
from rsl_rl.env import VecEnv


class OffPolicyRunner:
    """Runner for off-policy algorithms (FB-CPR)."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.log_dir = log_dir

        # cadence
        self.num_env_steps = int(train_cfg.get("num_env_steps", 10_000_000))
        self.update_agent_every = int(train_cfg.get("update_agent_every", 10 * env.num_envs))
        self.num_agent_updates = int(train_cfg.get("num_agent_updates", env.num_envs))
        self.num_seed_steps = int(train_cfg.get("num_seed_steps", 1000 * env.num_envs))
        self.log_every_steps = int(train_cfg.get("log_every_steps", 50_000))
        self.checkpoint_every_steps = int(train_cfg.get("checkpoint_every_steps", 1_000_000))
        train_cfg.setdefault("algorithm", {})
        train_cfg["algorithm"].setdefault("num_seed_steps", self.num_seed_steps)
        train_cfg.setdefault("multi_gpu", None)

        obs = env.get_observations()
        self.alg: FBCPR = FBCPR.construct_algorithm(obs, env, train_cfg, device)

        self.writer = SummaryWriter(log_dir) if log_dir is not None else None
        self.total_env_steps = 0
        self.it = 0

    def learn(self, num_env_steps: int | None = None) -> None:
        max_steps = num_env_steps if num_env_steps is not None else self.num_env_steps
        obs = self.env.get_observations()
        self.alg.train_mode()
        collect_per_iter = max(1, self.update_agent_every // self.env.num_envs)
        ep_reward = deque(maxlen=100)
        last_log = -self.log_every_steps
        last_ckpt = 0
        t_start = time.time()
        acc, n_acc = None, 0

        while self.total_env_steps < max_steps:
            # ---- collect ----
            with torch.inference_mode():
                for _ in range(collect_per_iter):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs = obs.to(self.device)
                    self.alg.process_env_step(obs, rewards.to(self.device), dones.to(self.device), extras)
                    self.total_env_steps += self.env.num_envs
                    if "log" in extras and isinstance(extras["log"], dict):
                        r = extras["log"].get("episode_return")
                        if r is not None:
                            ep_reward.append(float(r))

            # ---- update ----
            if (not self.alg.seeding) and len(self.alg.replay_buffer) > self.alg.batch_size:
                for _ in range(self.num_agent_updates):
                    loss = self.alg.update()
                    if acc is None:
                        acc = {k: 0.0 for k in loss}
                    for k, v in loss.items():
                        acc[k] += v
                    n_acc += 1
            self.it += 1

            # ---- log ----
            if self.total_env_steps - last_log >= self.log_every_steps and n_acc > 0:
                fps = self.total_env_steps / max(1e-6, time.time() - t_start)
                if self.writer is not None:
                    for k, v in acc.items():
                        self.writer.add_scalar(f"Loss/{k}", v / n_acc, self.total_env_steps)
                    self.writer.add_scalar("Perf/fps", fps, self.total_env_steps)
                    self.writer.add_scalar("Perf/total_env_steps", self.total_env_steps, self.total_env_steps)
                    if len(ep_reward) > 0:
                        self.writer.add_scalar("Train/mean_episode_return", statistics.mean(ep_reward), self.total_env_steps)
                    self.writer.flush()
                msg = {k: round(v / n_acc, 5) for k, v in acc.items()}
                print(f"[{self.total_env_steps}] fps={fps:.0f} " + str(msg), flush=True)
                acc, n_acc = None, 0
                last_log = self.total_env_steps

            # ---- checkpoint ----
            if self.total_env_steps - last_ckpt >= self.checkpoint_every_steps and self.log_dir is not None:
                self.save(os.path.join(self.log_dir, f"model_{self.total_env_steps}.pt"))
                last_ckpt = self.total_env_steps

        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.total_env_steps}.pt"))

    def save(self, path: str, infos: dict | None = None) -> None:
        saved = self.alg.save()
        saved["total_env_steps"] = self.total_env_steps
        saved["infos"] = infos
        torch.save(saved, path)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None) -> dict:
        loaded = torch.load(path, weights_only=False, map_location=map_location or self.device)
        self.alg.load(loaded, load_cfg, strict)
        self.total_env_steps = loaded.get("total_env_steps", 0)
        return loaded.get("infos", {})

    def get_inference_policy(self, device: str | None = None):
        self.alg.eval_mode()
        if device is not None:
            self.alg._model.to(device)
        return self.alg._model
