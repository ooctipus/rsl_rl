# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Value-shift signal, a config-gated PPO augmentation.

Like :class:`~rsl_rl.extensions.RandomNetworkDistillation`, this is an optional
component PPO turns on via ``value_shift_cfg`` -- it is *not* a separate
algorithm. Unlike RND it adds **no network, loss, or gradient**: after every
``update()`` it evaluates the algorithm's value function on a fixed observation cache and writes the
per-state ``|V_new - V_prev|`` magnitude into a buffer for an external consumer
(e.g. a prioritized curriculum sampler).

The cache and the two buffers are owned externally (by whatever consumes the
signal). They are wired in after construction by ``eval``-ing the bind
expressions carried in ``value_shift_cfg`` against ``{env, alg, self}`` -- e.g.
``"setattr(self, 'obs_cache', env.unwrapped.curriculum_manager...)"``. When no
cache is bound, :meth:`after_update` is a no-op.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from tensordict import TensorDict

    from rsl_rl.algorithms import PPO
    from rsl_rl.env import VecEnv


class ValueShift:
    """Post-update per-state critic value-drift tracker (no loss/gradient)."""

    def __init__(
        self,
        observation_bind: str | None = None,
        current_value_bind: str | None = None,
        value_diff_bind: str | None = None,
    ) -> None:
        """Initialize the value-shift extension.

        Args:
            observation_bind: Expression setting ``self.obs_cache`` (the cached critic-group observations, one
                row per tracked state).
            current_value_bind: Expression setting ``self.cur_val``, shape ``[num_states]``.
            value_diff_bind: Expression setting ``self.diff_val`` (shape ``[num_states]``), the buffer the
                external consumer reads.
        """
        self._binds = (observation_bind, current_value_bind, value_diff_bind)
        self.obs_cache: TensorDict | None = None
        self.cur_val: torch.Tensor | None = None
        self.diff_val: torch.Tensor | None = None

    def bind(self, env: VecEnv, alg: PPO) -> None:
        """Resolve the bind expressions against ``{env, alg, self}`` (called once after construction)."""
        ns = {"env": env, "alg": alg, "self": self, "setattr": setattr}
        for expr in self._binds:
            if expr is not None:
                eval(expr, ns)
        if self.obs_cache is not None:
            assert self.cur_val is not None and self.diff_val is not None, (
                "ValueShift: observation bound but current_value/value_diff are not."
            )
            n = self.obs_cache.batch_size[0]
            assert tuple(self.cur_val.shape) == (n,) and tuple(self.diff_val.shape) == (n,), (
                f"ValueShift cur_val/diff_val must have shape ({n},)."
            )

    def after_update(self, value_fn: Callable[[TensorDict], torch.Tensor]) -> None:
        """Write ``|V_new - V_prev|`` per state into ``diff_val`` and update ``cur_val``.

        ``value_fn`` is the algorithm's state-value function (``PPO._state_value``): the scalar critic head, or
        the successor read-out ``<psi, w>`` when the successor extension is enabled -- so this hook tracks the
        value the curriculum actually sees, regardless of critic type, and the two extensions compose.
        """
        if self.obs_cache is None:
            return
        with torch.inference_mode():
            v_new = value_fn(self.obs_cache).squeeze(-1)
        self.diff_val.copy_((v_new - self.cur_val).abs())
        self.cur_val.copy_(v_new)
