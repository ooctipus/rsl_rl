# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learner signals for reset-state curricula."""

from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Iterable
from typing import Protocol

from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer


class StateCurriculumProvider(Protocol):
    """Environment-owned reset bank consumed by :class:`StateCurriculum`."""

    sampled_state: torch.Tensor
    """Current reset-state index for every environment, shape ``[num_envs]``."""

    state_features: torch.Tensor | None
    """Fixed reset-state features, or ``None`` when success estimation is disabled."""

    success_rate: torch.Tensor
    """Mean eventual-success target per state, shape ``[num_states]``."""

    success_size: torch.Tensor
    """Number of targets represented by each success rate, shape ``[num_states]``."""

    value_shift: torch.Tensor
    """Sampled-start critic drift written in place, shape ``[num_states]``.

    Each observed row receives the absolute difference between its critic value
    before and after the same PPO update. Unobserved rows decay by the configured
    momentum instead of retaining stale priorities.
    """

    estimated_success_rate: torch.Tensor
    """Model-prior and empirical success estimate written in place, shape ``[num_states]``."""

    outcome_state_ids: torch.Tensor
    """Reset-bank rows awaiting targets, shape ``[num_envs]``; ``-1`` marks no outcome."""

    outcome_next_features: torch.Tensor
    """Physical endpoint features, shape ``[num_envs, feature_dim]``."""

    outcome_hard_targets: torch.Tensor
    """Ground-truth success targets for semantic terminations, shape ``[num_envs]``."""

    outcome_grounded: torch.Tensor
    """Whether each pending target comes from a semantic task termination, shape ``[num_envs]``."""

    def record_success_targets(self, env_ids: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> None:
        """Commit valid targets and release all pending outcome slots."""


class _SuccessEstimator(nn.Module):
    """Predict reset-state success from compact bank features."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] | list[int], activation: str) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("Success-estimator hidden_dims must not be empty.")
        self.normalizer = EmpiricalNormalization(input_dim)
        self.mlp = MLP(input_dim, 1, hidden_dims, activation)
        output = next(module for module in reversed(self.mlp) if isinstance(module, nn.Linear))
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.normalizer(features)).squeeze(-1)


class StateCurriculum:
    """Optional PPO-side signals for an environment-owned reset-state bank."""

    def __init__(
        self,
        storage: RolloutStorage,
        device: str,
        distributed: bool,
        value_shift_cfg: dict | None = None,
        success_estimator_cfg: dict | None = None,
    ) -> None:
        """Initialize enabled signals and preallocate rollout metadata."""
        self.device = torch.device(device)
        self.distributed = distributed
        self._storage = storage
        self._provider: StateCurriculumProvider | None = None
        self._episode_length: torch.Tensor | None = None

        self._value_shift_enabled = value_shift_cfg is not None
        value_shift_cfg = dict(value_shift_cfg or {})
        self._value_momentum = float(value_shift_cfg.pop("momentum", 0.0))
        self._value_batch_size = int(value_shift_cfg.pop("evaluation_batch_size", 16384))
        if value_shift_cfg:
            raise TypeError(f"Unexpected value-shift options: {sorted(value_shift_cfg)}")
        if not 0.0 <= self._value_momentum < 1.0:
            raise ValueError("Value-shift momentum must be in [0, 1).")
        if self._value_batch_size <= 0:
            raise ValueError("Value-shift evaluation_batch_size must be positive.")
        self._start_state = (
            torch.full((storage.num_transitions_per_env, storage.num_envs), -1, dtype=torch.long, device=self.device)
            if self._value_shift_enabled
            else None
        )
        self._start_value = (
            torch.empty((storage.num_transitions_per_env, storage.num_envs), device=self.device)
            if self._value_shift_enabled
            else None
        )
        self._episode_start = (
            torch.empty(storage.num_envs, dtype=torch.bool, device=self.device) if self._value_shift_enabled else None
        )

        self._success_cfg = success_estimator_cfg
        self._success_estimator: _SuccessEstimator | None = None
        self._success_optimizer: torch.optim.Optimizer | None = None
        self._success_num_batches = 0
        self._success_batch_size = 0
        self._success_eval_batch_size = 0
        self._success_max_grad_norm = 0.0
        self._success_prior_count = 0.0

    @property
    def enabled(self) -> bool:
        """Whether at least one curriculum signal is enabled."""
        return self._value_shift_enabled or self._success_cfg is not None

    def bind(self, provider: StateCurriculumProvider | None, episode_length: torch.Tensor, critic: nn.Module) -> None:
        """Bind environment state after PPO and its rollout storage are constructed."""
        if provider is None:
            raise ValueError("state_curriculum_cfg is enabled, but the environment returned no state curriculum.")
        if self._value_shift_enabled:
            self._validate_value_shift_provider(provider, episode_length)
        if self._success_cfg is not None:
            self._validate_success_estimator_provider(provider)
        if self._value_shift_enabled and self._success_cfg is not None:
            assert provider.state_features is not None
            if len(provider.value_shift) != len(provider.state_features):
                raise ValueError("Value-shift and success-estimator buffers must describe the same reset-state bank.")
        if self._value_shift_enabled and getattr(critic, "is_recurrent", False):
            raise ValueError("Value shift does not support recurrent critics.")
        self._provider = provider
        self._episode_length = episode_length
        if self._success_cfg is not None:
            self._build_success_estimator(self._success_cfg.copy())

    def record_episode_starts(self, step: int, values: torch.Tensor) -> None:
        """Mark sampled episode starts and their pre-update values in the existing rollout storage."""
        if not self._value_shift_enabled:
            return
        assert self._provider is not None and self._episode_length is not None
        assert self._start_state is not None and self._start_value is not None and self._episode_start is not None
        row = self._start_state[step]
        row.fill_(-1)
        torch.eq(self._episode_length, 0, out=self._episode_start)
        torch.where(self._episode_start, self._provider.sampled_state, row, out=row)
        self._start_value[step].copy_(values.squeeze(-1))

    @torch.no_grad()
    def update_success_targets(self) -> None:
        """Bootstrap pure timeouts from their endpoint state and commit pending outcomes."""
        if self._success_estimator is None:
            return
        assert self._provider is not None
        env_ids = (self._provider.outcome_state_ids >= 0).nonzero().flatten()
        if env_ids.numel() == 0:
            return

        targets = self._provider.outcome_hard_targets[env_ids].clone()
        grounded = self._provider.outcome_grounded[env_ids]
        valid = grounded & torch.isfinite(targets)
        endpoint_features = self._provider.outcome_next_features[env_ids]
        bootstrap = ~grounded & torch.isfinite(endpoint_features).all(dim=1)
        bootstrap_env_ids = env_ids[bootstrap]
        if bootstrap_env_ids.numel() > 0:
            targets[bootstrap] = self._success_estimator(
                self._provider.outcome_next_features[bootstrap_env_ids]
            ).sigmoid()
            valid[bootstrap] = torch.isfinite(targets[bootstrap])
        self._provider.record_success_targets(env_ids, targets, valid)

    @torch.no_grad()
    def update_value_shift(self, critic: nn.Module) -> None:
        """Update priorities only from sampled episode starts seen by the current PPO update."""
        if not self._value_shift_enabled:
            return
        assert self._provider is not None and self._start_state is not None and self._start_value is not None
        self._provider.value_shift.mul_(self._value_momentum)
        selected = self._start_state.flatten() >= 0
        if not bool(selected.any()):
            return

        state_ids = self._start_state.flatten()[selected]
        old_values = self._start_value.flatten()[selected]
        observations = self._storage.observations.reshape(-1)[selected]
        new_values = torch.empty_like(old_values)
        for start in range(0, len(state_ids), self._value_batch_size):
            stop = min(start + self._value_batch_size, len(state_ids))
            new_values[start:stop] = critic(observations[start:stop]).squeeze(-1)

        shift = (new_values - old_values).abs()
        unique_ids, inverse = torch.unique(state_ids, return_inverse=True)
        sums = torch.zeros(len(unique_ids), device=self.device).scatter_add_(0, inverse, shift)
        counts = torch.zeros_like(sums).scatter_add_(0, inverse, torch.ones_like(shift))
        mean_shift = sums / counts
        self._provider.value_shift[unique_ids] += mean_shift * (1.0 - self._value_momentum)
        self._start_state.fill_(-1)

    def update_success_estimator(self) -> float | None:
        """Fit recorded success targets and refresh full-bank predictions."""
        if self._success_estimator is None:
            return None
        assert self._provider is not None and self._success_optimizer is not None

        local_outcomes = self._provider.success_size.sum().to(dtype=torch.float32)
        global_outcomes = local_outcomes.clone()
        if self.distributed:
            torch.distributed.all_reduce(global_outcomes)
        if not bool(global_outcomes):
            return None

        has_local_outcomes = bool(local_outcomes)
        weights = self._provider.success_size.to(dtype=torch.float32) if has_local_outcomes else None
        weighted_loss = torch.zeros((), device=self.device)
        for _ in range(self._success_num_batches):
            self._success_optimizer.zero_grad()
            if has_local_outcomes:
                assert weights is not None and self._provider.state_features is not None
                state_ids = torch.multinomial(weights, self._success_batch_size, replacement=True)
                logits = self._success_estimator(self._provider.state_features[state_ids])
                targets = self._provider.success_rate[state_ids]
                local_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets)
                weighted_loss += local_loss.detach() * local_outcomes
                objective = local_loss * local_outcomes if self.distributed else local_loss
            else:
                objective = sum(param.sum() * 0.0 for param in self._success_estimator.parameters())
            objective.backward()
            if self.distributed:
                self._reduce_gradients(self._success_estimator.parameters(), global_outcomes)
            nn.utils.clip_grad_norm_(self._success_estimator.parameters(), self._success_max_grad_norm)
            self._success_optimizer.step()
        weighted_loss /= self._success_num_batches

        if self.distributed:
            torch.distributed.all_reduce(weighted_loss)
        self._predict_success_rates()
        return (weighted_loss / global_outcomes).item()

    def train_mode(self) -> None:
        """Set the success estimator to training mode."""
        if self._success_estimator is not None:
            self._success_estimator.train()

    def eval_mode(self) -> None:
        """Set the success estimator to evaluation mode."""
        if self._success_estimator is not None:
            self._success_estimator.eval()

    def save(self) -> dict:
        """Return checkpoint state owned by the learner-side extension."""
        if self._success_estimator is None:
            return {}
        assert self._success_optimizer is not None
        return {
            "success_estimator_state_dict": self._success_estimator.state_dict(),
            "success_optimizer_state_dict": self._success_optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, strict: bool = True) -> None:
        """Restore success-estimator state when present in a checkpoint."""
        if self._success_estimator is None or not loaded_dict:
            return
        assert self._success_optimizer is not None
        self._success_estimator.load_state_dict(loaded_dict["success_estimator_state_dict"], strict=strict)
        self._success_optimizer.load_state_dict(loaded_dict["success_optimizer_state_dict"])
        self._predict_success_rates()

    def broadcast_parameters(self) -> None:
        """Broadcast success-estimator parameters and normalization buffers from rank zero."""
        if self._success_estimator is None:
            return
        state = [self._success_estimator.state_dict()]
        torch.distributed.broadcast_object_list(state, src=0)
        self._success_estimator.load_state_dict(state[0])
        self._predict_success_rates()

    def _build_success_estimator(self, cfg: dict) -> None:
        assert self._provider is not None
        hidden_dims = cfg.pop("hidden_dims", (256, 256))
        activation = cfg.pop("activation", "elu")
        learning_rate = float(cfg.pop("learning_rate", 1.0e-4))
        optimizer = cfg.pop("optimizer", "adam")
        self._success_num_batches = int(cfg.pop("num_batches", 4))
        self._success_batch_size = int(cfg.pop("batch_size", 4096))
        self._success_eval_batch_size = int(cfg.pop("evaluation_batch_size", 16384))
        self._success_max_grad_norm = float(cfg.pop("max_grad_norm", 1.0))
        self._success_prior_count = float(cfg.pop("prior_count", 1.0))
        if cfg:
            raise TypeError(f"Unexpected success-estimator options: {sorted(cfg)}")
        if min(self._success_num_batches, self._success_batch_size, self._success_eval_batch_size) <= 0:
            raise ValueError("Success-estimator batch counts and sizes must be positive.")
        if self._success_max_grad_norm <= 0.0:
            raise ValueError("Success-estimator max_grad_norm must be positive.")
        if self._success_prior_count <= 0.0:
            raise ValueError("Success-estimator prior_count must be positive.")

        assert self._provider.state_features is not None
        feature_dim = self._provider.state_features.shape[1]
        self._success_estimator = _SuccessEstimator(feature_dim, hidden_dims, activation).to(self.device)
        self._success_optimizer = resolve_optimizer(optimizer)(self._success_estimator.parameters(), lr=learning_rate)  # type: ignore
        self._success_estimator.normalizer.update(self._provider.state_features)
        self._predict_success_rates()

    @torch.no_grad()
    def _predict_success_rates(self) -> None:
        assert self._provider is not None and self._success_estimator is not None
        features = self._provider.state_features
        assert features is not None
        estimates = self._provider.estimated_success_rate
        for start in range(0, len(features), self._success_eval_batch_size):
            stop = min(start + self._success_eval_batch_size, len(features))
            model_rate = self._success_estimator(features[start:stop]).sigmoid()
            count = self._provider.success_size[start:stop].to(dtype=model_rate.dtype)
            observed_rate = torch.where(count > 0, self._provider.success_rate[start:stop], 0.0)
            estimates[start:stop].copy_(
                (count * observed_rate + self._success_prior_count * model_rate) / (count + self._success_prior_count)
            )

    def _reduce_gradients(self, parameters: Iterable[nn.Parameter], denominator: torch.Tensor) -> None:
        params = [param for param in parameters if param.grad is not None]
        flat = torch.cat([param.grad.view(-1) for param in params])
        torch.distributed.all_reduce(flat)
        flat /= denominator
        offset = 0
        for param in params:
            size = param.numel()
            param.grad.copy_(flat[offset : offset + size].view_as(param.grad))
            offset += size

    def _validate_value_shift_provider(self, provider: StateCurriculumProvider, episode_length: torch.Tensor) -> None:
        if provider.value_shift.ndim != 1 or not provider.value_shift.is_floating_point():
            raise ValueError("State-curriculum value_shift must be a floating-point [num_states] tensor.")
        if provider.value_shift.device != self.device:
            raise ValueError(
                f"State-curriculum value_shift is on {provider.value_shift.device}, expected {self.device}."
            )
        if provider.sampled_state.shape != (self._storage.num_envs,):
            raise ValueError(
                f"State-curriculum sampled_state must have shape ({self._storage.num_envs},), "
                f"got {tuple(provider.sampled_state.shape)}."
            )
        if provider.sampled_state.dtype != torch.long:
            raise ValueError("State-curriculum sampled_state must use torch.long indices.")
        if provider.sampled_state.device != self.device or episode_length.device != self.device:
            raise ValueError("State-curriculum sampled states and episode lengths must be on the learner device.")
        if episode_length.shape != (self._storage.num_envs,):
            raise ValueError(
                f"Episode lengths must have shape ({self._storage.num_envs},), got {tuple(episode_length.shape)}."
            )

    def _validate_success_estimator_provider(self, provider: StateCurriculumProvider) -> None:
        features = provider.state_features
        if features is None or features.ndim != 2 or not features.is_floating_point():
            raise ValueError("Success estimation requires floating-point state_features[num_states, feature_dim].")
        num_states = len(features)
        vectors = {
            "success_rate": provider.success_rate,
            "success_size": provider.success_size,
            "estimated_success_rate": provider.estimated_success_rate,
        }
        for name, value in vectors.items():
            if value.shape != (num_states,):
                raise ValueError(f"State-curriculum {name} must have shape ({num_states},), got {tuple(value.shape)}.")
            if value.device != self.device:
                raise ValueError(f"State-curriculum {name} is on {value.device}, expected {self.device}.")
        if not provider.success_rate.is_floating_point() or not provider.estimated_success_rate.is_floating_point():
            raise ValueError("State-curriculum success rates must use a floating-point dtype.")
        if features.device != self.device:
            raise ValueError(f"State-curriculum features are on {features.device}, expected {self.device}.")
        if provider.success_size.dtype != torch.long:
            raise ValueError("State-curriculum success_size must use torch.long counts.")

        outcome_vectors = {
            "outcome_state_ids": provider.outcome_state_ids,
            "outcome_hard_targets": provider.outcome_hard_targets,
            "outcome_grounded": provider.outcome_grounded,
        }
        for name, value in outcome_vectors.items():
            if value.shape != (self._storage.num_envs,):
                raise ValueError(
                    f"State-curriculum {name} must have shape ({self._storage.num_envs},), got {tuple(value.shape)}."
                )
            if value.device != self.device:
                raise ValueError(f"State-curriculum {name} is on {value.device}, expected {self.device}.")
        if provider.outcome_state_ids.dtype != torch.long:
            raise ValueError("State-curriculum outcome_state_ids must use torch.long indices.")
        if not provider.outcome_hard_targets.is_floating_point():
            raise ValueError("State-curriculum outcome_hard_targets must use a floating-point dtype.")
        if provider.outcome_grounded.dtype != torch.bool:
            raise ValueError("State-curriculum outcome_grounded must use torch.bool values.")
        expected = (self._storage.num_envs, features.shape[1])
        if provider.outcome_next_features.shape != expected:
            raise ValueError(
                f"State-curriculum outcome_next_features must have shape {expected}, "
                f"got {tuple(provider.outcome_next_features.shape)}."
            )
        if (
            provider.outcome_next_features.device != self.device
            or not provider.outcome_next_features.is_floating_point()
        ):
            raise ValueError("State-curriculum outcome_next_features must be floating-point on the learner device.")
