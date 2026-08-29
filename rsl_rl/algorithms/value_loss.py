# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.nn import functional


class HLGaussValueLoss(nn.Module):
    """Train categorical value logits from Gaussian-smoothed scalar targets."""

    edges: torch.Tensor
    centers: torch.Tensor

    def __init__(
        self,
        min_value: float,
        max_value: float,
        num_bins: int = 101,
        sigma: float | None = None,
    ) -> None:
        """Initialize a fixed categorical support."""
        super().__init__()
        if max_value <= min_value:
            raise ValueError("HL-Gauss max_value must be greater than min_value.")
        if num_bins < 2:
            raise ValueError("HL-Gauss num_bins must be at least two.")

        bin_width = (max_value - min_value) / num_bins
        sigma = 0.75 * bin_width if sigma is None else sigma
        if sigma <= 0.0:
            raise ValueError("HL-Gauss sigma must be positive.")

        edges = torch.linspace(min_value, max_value, num_bins + 1, dtype=torch.float32)
        self.register_buffer("edges", edges)
        self.register_buffer("centers", 0.5 * (edges[:-1] + edges[1:]))
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.num_bins = num_bins
        self.sigma = float(sigma)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode categorical logits to scalar expectations."""
        self._check_logits(logits)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            probabilities = logits.float().softmax(dim=-1)
            return (probabilities * self.centers).sum(dim=-1, keepdim=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cross-entropy loss and the fraction of targets clipped to the fixed support."""
        self._check_logits(logits)
        with torch.autocast(device_type=logits.device.type, enabled=False):
            targets = targets.float()
            clipped_targets = targets.clamp(self.min_value, self.max_value)
            normalized_edges = (self.edges - clipped_targets) / self.sigma
            cdf = 0.5 * (1.0 + torch.erf(normalized_edges / math.sqrt(2.0)))
            masses = cdf[..., 1:] - cdf[..., :-1]
            masses /= masses.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
            loss = -(masses * functional.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()
            clipped = ((targets < self.min_value) | (targets > self.max_value)).float().mean()
            return loss, clipped

    def _check_logits(self, logits: torch.Tensor) -> None:
        if logits.shape[-1] != self.num_bins:
            raise ValueError(f"Expected {self.num_bins} value logits, got {logits.shape[-1]}.")
