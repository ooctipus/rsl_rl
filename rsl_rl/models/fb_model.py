# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# FB-CPR model ported from Meta Motivo (facebookresearch/metamotivo, CC BY-NC 4.0):
# merges metamotivo.fb.model.FBModel + metamotivo.fb_cpr.model.FBcprModel into a single
# self-contained rsl_rl model. Submodule names are kept identical to the reference so the
# ported FB-CPR update math transfers verbatim and weights are state-dict compatible.

from __future__ import annotations

import copy
import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..modules.fb_modules import build_actor, build_backward, build_discriminator, build_forward, eval_mode


# --------------------------------------------------------------------------------------
# config (merged from metamotivo fb/model.py + fb_cpr/model.py)
# --------------------------------------------------------------------------------------
def dict_to_config(source: Mapping, target: Any) -> None:
    target_fields = {field.name for field in dataclasses.fields(target)}
    for field in target_fields:
        if field in source.keys() and dataclasses.is_dataclass(getattr(target, field)):
            dict_to_config(source[field], getattr(target, field))
        elif field in source.keys():
            setattr(target, field, source[field])


def config_from_dict(source: dict, config_class: Any) -> Any:
    target = config_class()
    dict_to_config(source, target)
    return target


@dataclasses.dataclass
class ActorArchiConfig:
    hidden_dim: int = 1024
    model: str = "simple"
    hidden_layers: int = 1
    embedding_layers: int = 2


@dataclasses.dataclass
class ForwardArchiConfig:
    hidden_dim: int = 1024
    model: str = "simple"
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2
    ensemble_mode: str = "batch"


@dataclasses.dataclass
class BackwardArchiConfig:
    hidden_dim: int = 256
    hidden_layers: int = 2
    norm: bool = True


@dataclasses.dataclass
class CriticArchiConfig:
    hidden_dim: int = 1024
    model: str = "simple"
    hidden_layers: int = 1
    embedding_layers: int = 2
    num_parallel: int = 2
    ensemble_mode: str = "batch"


@dataclasses.dataclass
class DiscriminatorArchiConfig:
    hidden_dim: int = 1024
    hidden_layers: int = 2


@dataclasses.dataclass
class ArchiConfig:
    z_dim: int = 100
    norm_z: bool = True
    f: ForwardArchiConfig = dataclasses.field(default_factory=ForwardArchiConfig)
    b: BackwardArchiConfig = dataclasses.field(default_factory=BackwardArchiConfig)
    actor: ActorArchiConfig = dataclasses.field(default_factory=ActorArchiConfig)
    critic: CriticArchiConfig = dataclasses.field(default_factory=CriticArchiConfig)
    discriminator: DiscriminatorArchiConfig = dataclasses.field(default_factory=DiscriminatorArchiConfig)


@dataclasses.dataclass
class Config:
    obs_dim: int = -1
    action_dim: int = -1
    device: str = "cpu"
    archi: ArchiConfig = dataclasses.field(default_factory=ArchiConfig)
    inference_batch_size: int = 500_000
    seq_length: int = 1
    actor_std: float = 0.2
    norm_obs: bool = True


class ForwardBackwardModel(nn.Module):
    """FB-CPR model: forward map F, backward map B, actor, critic ensemble, discriminator.

    Faithful merge of metamotivo FBModel + FBcprModel. Networks (and their state-dict keys)
    are identical to the reference implementation.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = config_from_dict(kwargs, Config)
        obs_dim, action_dim = self.cfg.obs_dim, self.cfg.action_dim
        arch = self.cfg.archi

        # networks (same builders as metamotivo)
        self._backward_map = build_backward(obs_dim, arch.z_dim, arch.b)
        self._forward_map = build_forward(obs_dim, arch.z_dim, action_dim, arch.f)
        self._actor = build_actor(obs_dim, arch.z_dim, action_dim, arch.actor)
        self._discriminator = build_discriminator(obs_dim, arch.z_dim, arch.discriminator)
        self._critic = build_forward(obs_dim, arch.z_dim, action_dim, arch.critic, output_dim=1)
        self._obs_normalizer = (
            nn.BatchNorm1d(obs_dim, affine=False, momentum=0.01) if self.cfg.norm_obs else nn.Identity()
        )

        self.train(False)
        self.requires_grad_(False)
        self.to(self.cfg.device)

    def _prepare_for_train(self) -> None:
        # target networks (created after weight_init so they match the trained nets)
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map = copy.deepcopy(self._forward_map)
        self._target_critic = copy.deepcopy(self._critic)

    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.cfg.device = device.type
        return super().to(*args, **kwargs)

    def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    # ----- inference accessors (no grad, normalized obs) -----
    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor) -> torch.Tensor:
        return self._backward_map(self._normalize(obs))

    @torch.no_grad()
    def forward_map(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._forward_map(self._normalize(obs), z, action)

    @torch.no_grad()
    def actor(self, obs: torch.Tensor, z: torch.Tensor, std: float):
        return self._actor(self._normalize(obs), z, std)

    @torch.no_grad()
    def critic(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self._critic(self._normalize(obs), z, action)

    @torch.no_grad()
    def discriminator(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self._discriminator(self._normalize(obs), z)

    # ----- z helpers -----
    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def act(self, obs: torch.Tensor, z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, z, self.cfg.actor_std)
        if mean:
            return dist.mean
        return dist.sample()

    # ----- prompting (used for eval; identical to metamotivo) -----
    def reward_inference(self, next_obs, reward, weight=None) -> torch.Tensor:
        num_batches = int(np.ceil(next_obs.shape[0] / self.cfg.inference_batch_size))
        z = 0
        wr = reward if weight is None else reward * weight
        for i in range(num_batches):
            s, e = i * self.cfg.inference_batch_size, (i + 1) * self.cfg.inference_batch_size
            B = self.backward_map(next_obs[s:e].to(self.cfg.device))
            z += torch.matmul(wr[s:e].to(self.cfg.device).T, B)
        return self.project_z(z)

    def reward_wr_inference(self, next_obs, reward) -> torch.Tensor:
        return self.reward_inference(next_obs, reward, F.softmax(10 * reward, dim=0))

    def goal_inference(self, next_obs: torch.Tensor) -> torch.Tensor:
        return self.project_z(self.backward_map(next_obs))

    def tracking_inference(self, next_obs: torch.Tensor) -> torch.Tensor:
        z = self.backward_map(next_obs)
        for step in range(z.shape[0]):
            end_idx = min(step + self.cfg.seq_length, z.shape[0])
            z[step] = z[step:end_idx].mean(dim=0)
        return self.project_z(z)
