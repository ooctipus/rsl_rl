# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional architecture checks against the local reference repositories."""

from __future__ import annotations

import os
import sys
import torch
from pathlib import Path
from tensordict import TensorDict

import pytest

from rsl_rl.models.forward_backward_model import (
    ForwardBackwardDualNetworkCfg,
    ForwardBackwardModel,
    ForwardBackwardValueHeadCfg,
)
from rsl_rl.modules.reward_channels import ForwardBackwardValueSpec

_META_ROOT = os.getenv("METAMOTIVO_ORACLE_DIR")
_BFM_ROOT = os.getenv("BFM_ZERO_ORACLE_DIR")
_META_REPO = os.getenv("METAMOTIVO_REPO", _META_ROOT)
_BFM_REPO = os.getenv("BFM_ZERO_REPO", _BFM_ROOT)


def _count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _model(
    dual_cfg: ForwardBackwardDualNetworkCfg,
    *,
    discriminator: bool = False,
    value_head: bool = False,
) -> ForwardBackwardModel:
    observations = TensorDict({"state": torch.randn(2, 11)}, batch_size=[2])
    routes = {
        "actor": ("state",),
        "forward": ("state",),
        "backward": ("state",),
        "discriminator": ("state",),
        "critic_discriminator": ("state",),
    }
    heads = ()
    if value_head:
        heads = (
            ForwardBackwardValueHeadCfg(
                spec=ForwardBackwardValueSpec(
                    name="critic",
                    kind="critic",
                    route="critic_discriminator",
                    reward_channels=("reward",),
                    ensemble_size=2,
                    has_target=True,
                ),
                network=dual_cfg,
            ),
        )
    return ForwardBackwardModel(
        observations,
        routes,
        action_dim=3,
        context_dim=4,
        actor_cfg=dual_cfg,
        forward_cfg=dual_cfg,
        backward_hidden_dims=(8, 8),
        forward_ensemble_size=2,
        discriminator_hidden_dims=(8, 8) if discriminator else None,
        value_heads=heads,
        normalization_type="none",
    )


@pytest.mark.skipif(_META_ROOT is None, reason="METAMOTIVO_ORACLE_DIR is not set")
@pytest.mark.parametrize("residual", (False, True))
def test_meta_forward_parameter_count_matches_reference(residual: bool) -> None:
    """Simple and residual forward maps should own the same parameters as MetaMotivo."""
    sys.path.insert(0, str(Path(_META_REPO)))
    from metamotivo.nn_models import ForwardMap, ResidualForwardMap

    cfg = ForwardBackwardDualNetworkCfg(16, 3, 2, residual)
    model = _model(cfg)
    reference_type = ResidualForwardMap if residual else ForwardMap
    reference = reference_type(11, 4, 3, 16, 3, 2, 2)

    assert _count(model.forward_network) == _count(reference)


@pytest.mark.skipif(_META_ROOT is None, reason="METAMOTIVO_ORACLE_DIR is not set")
def test_meta_actor_backward_discriminator_and_critic_counts_match_reference() -> None:
    """Every non-forward MetaMotivo component should preserve architecture ownership."""
    sys.path.insert(0, str(Path(_META_REPO)))
    from metamotivo.nn_models import Actor, BackwardMap, Discriminator, ForwardMap

    cfg = ForwardBackwardDualNetworkCfg(16, 3, 2)
    model = _model(cfg, discriminator=True, value_head=True)

    assert _count(model.actor_network) == _count(Actor(11, 4, 3, 16, 3, 2))
    assert _count(model.backward_network) == _count(BackwardMap(11, 4, 8, 2, True))
    assert _count(model.discriminator_network) == _count(Discriminator(11, 4, 8, 2))
    assert _count(model.value_networks["critic"]) == _count(ForwardMap(11, 4, 3, 16, 3, 2, 2, output_dim=1))


@pytest.mark.skipif(_BFM_ROOT is None, reason="BFM_ZERO_ORACLE_DIR is not set")
def test_bfm_residual_topology_is_reproduced_with_explicit_embedding_depth() -> None:
    """The successful BFM topology should be explicit instead of preserving its ignored config field."""
    sys.path.insert(0, str(Path(_BFM_REPO)))
    import gymnasium
    from humanoidverse.agents.nn_models import ForwardArchiConfig

    space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(11,))
    reference_cfg = ForwardArchiConfig(
        hidden_dim=16,
        model="residual",
        hidden_layers=3,
        embedding_layers=2,
        num_parallel=2,
    )
    reference = reference_cfg.build(space, z_dim=4, action_dim=3)

    configured = _model(ForwardBackwardDualNetworkCfg(16, 3, 2, True))
    reproduced = _model(ForwardBackwardDualNetworkCfg(16, 3, 3, True))

    assert _count(configured.forward_network) != _count(reference)
    assert _count(reproduced.forward_network) == _count(reference)
