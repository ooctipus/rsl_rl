# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2020 Preferred Networks, Inc.


from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape: int | tuple[int, ...] | list[int], eps: float = 1e-2, until: int | None = None) -> None:
        """Initialize EmpiricalNormalization module.

        .. note:: The normalization parameters are computed over the whole batch, not for each environment separately.

        Args:
            shape: Shape of input values except batch axis.
            eps: Small value for stability.
            until: If this arg is specified, the module learns input values until the sum of batch sizes exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_pending_mean", torch.zeros(shape).unsqueeze(0), persistent=False)
        self.register_buffer("_pending_m2", torch.zeros(shape).unsqueeze(0), persistent=False)
        self.register_buffer("_pending_count", torch.tensor(0, dtype=torch.long), persistent=False)
        self._defer_updates = False

    @property
    def mean(self) -> torch.Tensor:
        """Return the current running mean."""
        return self._mean.squeeze(0).clone()  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the current running standard deviation."""
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize mean and variance of values based on empirical values."""
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """Learn input values without computing the output values of them."""
        if not self.training:
            return
        observed = self.count + self._pending_count if self._defer_updates else self.count
        if self.until is not None and observed >= self.until:
            return

        count_x = x.shape[0]
        mean_x = torch.mean(x, dim=0, keepdim=True)
        m2_x = torch.var(x, dim=0, unbiased=False, keepdim=True) * count_x
        if self._defer_updates:
            self._merge_pending(mean_x, m2_x, count_x)
        else:
            self._merge_running(mean_x, m2_x, count_x)

    @torch.jit.unused
    def set_deferred_updates(self, enabled: bool = True) -> None:
        """Defer updates until the owning learner synchronizes a rollout."""
        self._defer_updates = enabled

    def _merge_pending(self, mean: torch.Tensor, m2: torch.Tensor, count: int | torch.Tensor) -> None:
        count_t = torch.as_tensor(count, device=self.count.device, dtype=self.count.dtype)
        total = self._pending_count + count_t
        delta = mean - self._pending_mean
        ratio = count_t.to(mean.dtype) / total.clamp_min(1).to(mean.dtype)
        self._pending_mean.add_(delta * ratio)
        cross = delta.square() * self._pending_count.to(mean.dtype) * count_t.to(mean.dtype) / total.clamp_min(1)
        self._pending_m2.add_(m2 + cross)
        self._pending_count.copy_(total)

    def _merge_running(self, mean: torch.Tensor, m2: torch.Tensor, count: int | torch.Tensor) -> None:
        count_t = torch.as_tensor(count, device=self.count.device, dtype=self.count.dtype)
        if not bool(count_t):
            return
        total = self.count + count_t
        delta = mean - self._mean
        ratio = count_t.to(mean.dtype) / total.to(mean.dtype)
        new_mean = self._mean + delta * ratio
        old_m2 = self._var * self.count.to(mean.dtype)
        cross = delta.square() * self.count.to(mean.dtype) * count_t.to(mean.dtype) / total
        self._mean.copy_(new_mean)
        self._var.copy_((old_m2 + m2 + cross) / total)
        self._std.copy_(torch.sqrt(self._var.clamp_min(0.0)))
        self.count.copy_(total)

    def _clear_pending(self) -> None:
        self._pending_mean.zero_()
        self._pending_m2.zero_()
        self._pending_count.zero_()

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """De-normalize values based on empirical values."""
        return y * (self._std + self.eps) + self._mean


def set_deferred_normalization(modules: tuple[nn.Module | None, ...]) -> None:
    """Make normalization statistics update only at learner synchronization boundaries."""
    for module in modules:
        if module is not None:
            for child in module.modules():
                if isinstance(child, EmpiricalNormalization):
                    child.set_deferred_updates()


@torch.no_grad()
def synchronize_normalization(modules: tuple[nn.Module | None, ...], distributed: bool = False) -> None:
    """Merge every deferred normalizer using one collective."""
    normalizers: list[EmpiricalNormalization] = []
    seen: set[int] = set()
    for module in modules:
        if module is None:
            continue
        for child in module.modules():
            if isinstance(child, EmpiricalNormalization) and id(child) not in seen:
                normalizers.append(child)
                seen.add(id(child))
    if not normalizers:
        return

    payload = torch.cat([
        torch.cat((
            norm._pending_count.to(norm._pending_mean.dtype).view(1),
            norm._pending_mean.flatten(),
            norm._pending_m2.flatten(),
        ))
        for norm in normalizers
    ])
    if distributed:
        world_size = torch.distributed.get_world_size()
        gathered = torch.empty(world_size * payload.numel(), device=payload.device, dtype=payload.dtype)
        torch.distributed.all_gather_into_tensor(gathered, payload)
        gathered = gathered.view(world_size, -1)
    else:
        gathered = payload.unsqueeze(0)

    for rank_payload in gathered:
        offset = 0
        for norm in normalizers:
            size = norm._pending_mean.numel()
            count = rank_payload[offset].round().to(torch.long)
            mean = rank_payload[offset + 1 : offset + 1 + size].view_as(norm._pending_mean)
            m2 = rank_payload[offset + 1 + size : offset + 1 + 2 * size].view_as(norm._pending_m2)
            norm._merge_running(mean, m2, count)
            offset += 1 + 2 * size
    for norm in normalizers:
        norm._clear_pending()


@torch.no_grad()
def distributed_mean_var(
    x: torch.Tensor, distributed: bool = False, unbiased: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scalar moments over the local tensor or all learner ranks."""
    mean = x.mean()
    moments = torch.stack((x.new_tensor(x.numel()), mean, (x - mean).square().sum()))
    if distributed:
        world_size = torch.distributed.get_world_size()
        gathered = torch.empty(world_size * 3, device=x.device, dtype=x.dtype)
        torch.distributed.all_gather_into_tensor(gathered, moments)
        gathered = gathered.view(world_size, 3)
    else:
        gathered = moments.unsqueeze(0)

    count = x.new_zeros(())
    mean = x.new_zeros(())
    m2 = x.new_zeros(())
    for rank_count, rank_mean, rank_m2 in gathered:
        total = count + rank_count
        delta = rank_mean - mean
        mean = mean + delta * rank_count / total.clamp_min(1.0)
        m2 = m2 + rank_m2 + delta.square() * count * rank_count / total.clamp_min(1.0)
        count = total
    denominator = count - 1.0 if unbiased else count
    return mean, m2 / denominator.clamp_min(1.0)


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize the scale of the
    rewards so that the value function can learn quickly. We did this by dividing the rewards by a running estimate of
    the standard deviation of the sum of discounted rewards.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...] | list[int],
        eps: float = 1e-2,
        gamma: float = 0.99,
        until: int | None = None,
    ) -> None:
        """Initialize discounted-reward normalization with running moments."""
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = _DiscountedAverage(gamma)

    def forward(self, rew: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using the running std of discounted returns."""
        if self.training:
            # Update discounted rewards
            avg = self.disc_avg.update(rew)
            # Update moments from discounted rewards
            self.emp_norm.update(avg)

        # Normalize rewards with the empirical std
        if self.emp_norm._std > 0:  # type: ignore
            return rew / self.emp_norm._std  # type: ignore
        else:
            return rew


class _DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t
    """

    def __init__(self, gamma: float) -> None:
        """Initialize discounted accumulation with a fixed discount factor."""
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        """Update and return the discounted running average."""
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg
