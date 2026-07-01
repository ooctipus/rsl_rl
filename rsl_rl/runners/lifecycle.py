# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional lifecycle work at exact off-policy transition boundaries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from tensordict import TensorDictBase
from typing import Any

from rsl_rl.env import VecEnv


class RunnerLifecycleExtension(ABC):
    """One runner-owned extension invoked at declared transition boundaries."""

    def __init__(
        self,
        env: VecEnv,
        algorithm: Any,
        log_dir: str | None,
        device: str,
    ) -> None:
        """Retain the narrow runner resources available to lifecycle work."""
        self.env = env
        self.algorithm = algorithm
        self.log_dir = log_dir
        self.device = device

    @abstractmethod
    def on_transition(self, transition: int) -> TensorDictBase | None:
        """Run one event and optionally return observations for resumed collection."""
        raise NotImplementedError

    def state_dict(self) -> dict[str, object]:
        """Return mutable extension state required for exact resume."""
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore mutable extension state required for exact resume."""
        if state:
            raise ValueError(f"{type(self).__name__} does not define checkpoint state.")


__all__ = ["RunnerLifecycleExtension"]
