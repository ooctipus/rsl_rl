# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional


class SIGReg(nn.Module):
    """Match random one-dimensional feature projections to a standard Gaussian."""

    frequencies: torch.Tensor

    def __init__(
        self,
        loss_coef: float,
        num_slices: int = 16,
        num_frequencies: int = 8,
        frequency_max: float = 5.0,
    ) -> None:
        """Initialize SIGReg's projection and frequency sampling."""
        super().__init__()
        if loss_coef < 0.0:
            raise ValueError("SIGReg loss_coef must be non-negative.")
        if min(num_slices, num_frequencies) <= 0:
            raise ValueError("SIGReg num_slices and num_frequencies must be positive.")
        if frequency_max <= 0.0:
            raise ValueError("SIGReg frequency_max must be positive.")
        self.loss_coef = loss_coef
        self.num_slices = num_slices
        frequencies = torch.linspace(-frequency_max, frequency_max, num_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the characteristic-function matching loss."""
        with torch.autocast(device_type=features.device.type, enabled=False):
            features = features.float().flatten(0, -2)
            directions = functional.normalize(
                torch.randn(self.num_slices, features.shape[-1], device=features.device, dtype=features.dtype), dim=-1
            )
            phases = (features @ directions.T).unsqueeze(-1) * self.frequencies
            real = phases.cos().mean(dim=0)
            imaginary = phases.sin().mean(dim=0)
            target = torch.exp(-0.5 * self.frequencies.square())
            return ((real - target).square() + imaginary.square()).mean()
