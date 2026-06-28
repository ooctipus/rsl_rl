# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as functional
from functools import reduce

from rsl_rl.utils import get_param, resolve_nn_activation


class MLPEnsembleLinear(nn.Module):
    """Independent linear layers evaluated as one ensemble.

    The leading output dimension indexes ensemble members. A two-dimensional
    input is broadcast to every member; an input that already starts with the
    ensemble dimension is consumed directly.
    """

    def __init__(self, input_dim: int, output_dim: int, ensemble_size: int, bias: bool = True) -> None:
        """Allocate one weight matrix per ensemble member."""
        super().__init__()
        if ensemble_size < 2:
            raise ValueError("ensemble_size must be at least two.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.empty(ensemble_size, output_dim, input_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(ensemble_size, 1, output_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use the same initialization independently for every member."""
        for weight in self.weight:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.input_dim)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all ensemble members."""
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return torch.matmul(x, self.weight.transpose(-1, -2)) + (self.bias if self.bias is not None else 0.0)

    def extra_repr(self) -> str:
        """Return constructor dimensions for module summaries."""
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"ensemble_size={self.ensemble_size}, bias={self.bias is not None}"
        )


class MLPEnsembleLayerNorm(nn.Module):
    """Layer normalization with independent affine parameters per ensemble member."""

    def __init__(self, width: int, ensemble_size: int, eps: float = 1e-5) -> None:
        """Allocate affine parameters for each ensemble member."""
        super().__init__()
        if ensemble_size < 2:
            raise ValueError("ensemble_size must be at least two.")
        self.width = width
        self.ensemble_size = ensemble_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(ensemble_size, 1, width))
        self.bias = nn.Parameter(torch.zeros(ensemble_size, 1, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the feature dimension and apply member-specific affine terms."""
        return functional.layer_norm(x, (self.width,), eps=self.eps) * self.weight + self.bias


def _linear(input_dim: int, output_dim: int, ensemble_size: int) -> nn.Module:
    if ensemble_size == 1:
        return nn.Linear(input_dim, output_dim)
    return MLPEnsembleLinear(input_dim, output_dim, ensemble_size)


def _layer_norm(width: int, ensemble_size: int) -> nn.Module:
    if ensemble_size == 1:
        return nn.LayerNorm(width)
    return MLPEnsembleLayerNorm(width, ensemble_size)


def _normalization(name: str, width: int, ensemble_size: int) -> nn.Module:
    if name != "layer_norm":
        raise ValueError(f"Unsupported MLP normalization: {name!r}.")
    return _layer_norm(width, ensemble_size)


def _hidden_layer_settings(
    setting: str | tuple[str | None, ...] | list[str | None] | None,
    num_hidden_layers: int,
    name: str,
) -> tuple[str | None, ...]:
    if isinstance(setting, (tuple, list)):
        if len(setting) != num_hidden_layers:
            raise ValueError(f"{name} must contain one entry per hidden layer.")
        return tuple(setting)
    return (setting,) * num_hidden_layers


class MLPBlock(nn.Module):
    """Pre-normalized linear block with optional residual connection."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str | None = "mish",
        ensemble_size: int = 1,
        residual: bool = False,
    ) -> None:
        """Build a reusable pre-normalized MLP block."""
        super().__init__()
        if residual and input_dim != output_dim:
            raise ValueError("A residual MLP block requires equal input and output widths.")
        self.normalization = _layer_norm(input_dim, ensemble_size)
        self.linear = _linear(input_dim, output_dim, ensemble_size)
        self.activation = resolve_nn_activation(activation) if activation is not None else nn.Identity()
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization, projection, activation, and the optional skip."""
        output = self.activation(self.linear(self.normalization(x)))
        return x + output if self.residual else output


class MLP(nn.Sequential):
    """Multi-Layer Perceptron with optional ensemble evaluation.

    Hidden-layer activations and normalizations may be scalars, which broadcast
    to every hidden layer, or sequences aligned with ``hidden_dims``.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int],
        activation: str | tuple[str | None, ...] | list[str | None] | None = "elu",
        last_activation: str | None = None,
        *,
        ensemble_size: int = 1,
        normalization: str | tuple[str | None, ...] | list[str | None] | None = None,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension or shape of the output.
            hidden_dims: Dimensions of the hidden layers. A value of -1 uses
                the input width.
            activation: Hidden-layer activation or one activation per hidden
                layer. None omits activation for the corresponding layer.
            last_activation: Optional activation after the output layer.
            ensemble_size: Number of independent MLPs evaluated together.
            normalization: Hidden-layer normalization or one normalization per
                hidden layer. Supported values are None and "layer_norm".
        """
        super().__init__()
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be positive.")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer.")

        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]
        if any(dim < 1 for dim in hidden_dims_processed):
            raise ValueError("All hidden dimensions must be positive.")

        hidden_activations = _hidden_layer_settings(activation, len(hidden_dims_processed), "activation")
        hidden_normalizations = _hidden_layer_settings(normalization, len(hidden_dims_processed), "normalization")
        layers: list[nn.Module] = []
        layer_input_dim = input_dim
        for layer_output_dim, layer_activation, layer_normalization in zip(
            hidden_dims_processed, hidden_activations, hidden_normalizations, strict=True
        ):
            layers.append(_linear(layer_input_dim, layer_output_dim, ensemble_size))
            if layer_normalization is not None:
                layers.append(_normalization(layer_normalization, layer_output_dim, ensemble_size))
            if layer_activation is not None:
                layers.append(resolve_nn_activation(layer_activation))
            layer_input_dim = layer_output_dim

        if isinstance(output_dim, int):
            layers.append(_linear(hidden_dims_processed[-1], output_dim, ensemble_size))
        else:
            total_output_dim = reduce(lambda x, y: x * y, output_dim)
            layers.append(_linear(hidden_dims_processed[-1], total_output_dim, ensemble_size))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        if last_activation is not None:
            layers.append(resolve_nn_activation(last_activation))

        for index, layer in enumerate(layers):
            self.add_module(f"{index}", layer)

    def init_weights(self, scales: float | tuple[float]) -> None:
        """Initialize linear weights orthogonally."""
        for index, module in enumerate(self):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_param(scales, index))
                nn.init.zeros_(module.bias)
            elif isinstance(module, MLPEnsembleLinear):
                for weight in module.weight:
                    nn.init.orthogonal_(weight, gain=get_param(scales, index))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MLP."""
        for layer in self:
            x = layer(x)
        return x
