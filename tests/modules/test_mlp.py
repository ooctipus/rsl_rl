# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for reusable MLP ensemble and residual composition."""

import torch
import torch.nn.functional as functional

import pytest

from rsl_rl.modules import MLP
from rsl_rl.modules.mlp import MLPBlock, MLPEnsembleLayerNorm, MLPEnsembleLinear


def test_default_mlp_shape_and_layout_remain_unchanged() -> None:
    """The new optional controls should not alter ordinary RSL-RL MLPs."""
    model = MLP(5, 3, (7, 6), "elu")

    assert model(torch.randn(4, 5)).shape == (4, 3)
    assert isinstance(model[0], torch.nn.Linear)
    assert isinstance(model[1], torch.nn.ELU)
    assert isinstance(model[2], torch.nn.Linear)
    assert isinstance(model[3], torch.nn.ELU)
    assert isinstance(model[4], torch.nn.Linear)


def test_ensemble_linear_matches_independent_linear_heads() -> None:
    """Batched ensemble evaluation should equal explicit independent heads."""
    torch.manual_seed(3)
    layer = MLPEnsembleLinear(5, 4, ensemble_size=3)
    inputs = torch.randn(7, 5)

    actual = layer(inputs)
    expected = torch.stack([functional.linear(inputs, layer.weight[index], layer.bias[index, 0]) for index in range(3)])

    torch.testing.assert_close(actual, expected)


def test_ensemble_mlp_accepts_shared_or_member_specific_inputs() -> None:
    """The first layer should broadcast shared rows and preserve ensemble rows."""
    model = MLP(
        5,
        2,
        (8, 8),
        ("tanh", "relu"),
        ensemble_size=3,
        normalization=("layer_norm", None),
    )
    shared = torch.randn(7, 5)
    member_specific = shared.unsqueeze(0).expand(3, -1, -1).clone()

    shared_output = model(shared)
    specific_output = model(member_specific)

    assert shared_output.shape == (3, 7, 2)
    torch.testing.assert_close(shared_output, specific_output)
    assert any(isinstance(module, MLPEnsembleLayerNorm) for module in model)


def test_mlp_accepts_per_hidden_layer_structure() -> None:
    """Each hidden layer should independently select normalization and activation."""
    model = MLP(
        5,
        3,
        (7, 6, 4),
        activation=("tanh", None, "relu"),
        normalization=(None, "layer_norm", "layer_norm"),
    )

    assert [type(module) for module in model] == [
        torch.nn.Linear,
        torch.nn.Tanh,
        torch.nn.Linear,
        torch.nn.LayerNorm,
        torch.nn.Linear,
        torch.nn.LayerNorm,
        torch.nn.ReLU,
        torch.nn.Linear,
    ]


@pytest.mark.parametrize(
    ("argument", "setting"),
    [("activation", ("relu",)), ("normalization", ("layer_norm",))],
)
def test_mlp_rejects_per_layer_setting_with_wrong_length(argument: str, setting: tuple[str]) -> None:
    """Per-layer settings should align exactly with the hidden dimensions."""
    with pytest.raises(ValueError, match=f"{argument} must contain one entry per hidden layer"):
        MLP(5, 3, (7, 6), **{argument: setting})


def test_mlp_rejects_unknown_normalization() -> None:
    """Unknown normalization names should fail at model construction."""
    with pytest.raises(ValueError, match="Unsupported MLP normalization"):
        MLP(5, 3, (7, 6), normalization="batch_norm")


def test_residual_block_identity_when_projection_is_zero() -> None:
    """A residual block should expose its skip connection exactly."""
    block = MLPBlock(6, 6, ensemble_size=2, residual=True)
    torch.nn.init.zeros_(block.linear.weight)
    torch.nn.init.zeros_(block.linear.bias)
    inputs = torch.randn(2, 5, 6)

    torch.testing.assert_close(block(inputs), inputs)


def test_residual_block_rejects_mismatched_widths() -> None:
    """Residual ownership should be explicit rather than silently projected."""
    with pytest.raises(ValueError, match="equal input and output widths"):
        MLPBlock(5, 6, residual=True)


def test_ensemble_path_keeps_gradients_per_member() -> None:
    """Each member should receive gradients without a Python model loop."""
    model = MLP(4, 2, (8,), "relu", ensemble_size=3)
    loss = model(torch.randn(6, 4)).square().sum()

    loss.backward()

    ensemble_layers = [module for module in model if isinstance(module, MLPEnsembleLinear)]
    assert ensemble_layers
    assert all(layer.weight.grad is not None for layer in ensemble_layers)
    assert all(torch.count_nonzero(layer.weight.grad).item() > 0 for layer in ensemble_layers)
