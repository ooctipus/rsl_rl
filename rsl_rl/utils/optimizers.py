# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Callable
from torch.optim.optimizer import ParamsT
from typing import Any


class MuonAdamW(torch.optim.Optimizer):
    """Apply native Muon to matrices and AdamW to all remaining parameters."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0e-3,
        weight_decay: float = 0.1,
        adjust_lr_fn: str | None = None,
        muon_kwargs: dict[str, Any] | None = None,
        adamw_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the two native optimizers behind one optimizer interface."""
        muon_class = getattr(torch.optim, "Muon", None)
        if muon_class is None:
            raise RuntimeError("MuonAdamW requires a Torch build that provides torch.optim.Muon.")

        self._muon_class = muon_class
        self._muon_kwargs = dict(muon_kwargs or {})
        self._adamw_kwargs = dict(adamw_kwargs or {})
        self._check_options("muon_kwargs", self._muon_kwargs, {"lr", "weight_decay", "adjust_lr_fn"})
        self._check_options("adamw_kwargs", self._adamw_kwargs, {"lr", "weight_decay"})
        self._adjust_lr_fn = adjust_lr_fn
        self._children_ready = False
        self._muon: torch.optim.Optimizer | None = None
        self._adamw: torch.optim.Optimizer | None = None

        super().__init__(params, {"lr": lr, "weight_decay": weight_decay})
        source_groups = self.param_groups
        self.param_groups = []
        self._children_ready = True
        for group in source_groups:
            self._add_partitioned_group(group)
        self._share_state()

    @staticmethod
    def _check_options(name: str, options: dict[str, Any], reserved: set[str]) -> None:
        conflicts = reserved.intersection(options)
        if conflicts:
            raise ValueError(f"Pass {sorted(conflicts)} directly instead of through {name}.")

    @staticmethod
    def _partition(group: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        parameters = group["params"]
        matrix_indices = [index for index, parameter in enumerate(parameters) if parameter.ndim == 2]
        vector_indices = [index for index, parameter in enumerate(parameters) if parameter.ndim != 2]

        def select(indices: list[int]) -> dict[str, Any] | None:
            if not indices:
                return None
            selected = {key: value for key, value in group.items() if key not in ("params", "param_names")}
            selected["params"] = [parameters[index] for index in indices]
            if "param_names" in group:
                selected["param_names"] = [group["param_names"][index] for index in indices]
            return selected

        return select(matrix_indices), select(vector_indices)

    def _add_partitioned_group(self, group: dict[str, Any]) -> None:
        matrix_group, vector_group = self._partition(group)
        if matrix_group is not None:
            if self._muon is None:
                self._muon = self._muon_class(
                    [matrix_group],
                    lr=self.defaults["lr"],
                    weight_decay=self.defaults["weight_decay"],
                    adjust_lr_fn=self._adjust_lr_fn,
                    **self._muon_kwargs,
                )
            else:
                self._muon.add_param_group(matrix_group)
        if vector_group is not None:
            if self._adamw is None:
                self._adamw = torch.optim.AdamW(
                    [vector_group],
                    lr=self.defaults["lr"],
                    weight_decay=self.defaults["weight_decay"],
                    **self._adamw_kwargs,
                )
            else:
                self._adamw.add_param_group(vector_group)
        self._refresh_param_groups()

    def _refresh_param_groups(self) -> None:
        self.param_groups = []
        if self._muon is not None:
            self.param_groups.extend(self._muon.param_groups)
        if self._adamw is not None:
            self.param_groups.extend(self._adamw.param_groups)

    def _share_state(self) -> None:
        if self._muon is not None:
            self._muon.state = self.state
        if self._adamw is not None:
            self._adamw.state = self.state

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """Add and route a parameter group according to tensor rank."""
        if not self._children_ready:
            super().add_param_group(param_group)
            return

        super().add_param_group(param_group)
        group = self.param_groups.pop()
        self._add_partitioned_group(group)
        self._share_state()

    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """Run one Muon step and one AdamW step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self._muon is not None:
            self._muon.step()
        if self._adamw is not None:
            self._adamw.step()
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the flat optimizer state and reconnect both native optimizers."""
        super().load_state_dict(state_dict)
        if self._muon is not None:
            self._muon.param_groups = [group for group in self.param_groups if group["params"][0].ndim == 2]
        if self._adamw is not None:
            self._adamw.param_groups = [group for group in self.param_groups if group["params"][0].ndim != 2]
        self._share_state()
