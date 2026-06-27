# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Core forward-backward tensor operators."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F  # noqa: N812
from numbers import Real


def _logit_batch_size(logits: torch.Tensor, name: str) -> int:
    """Validate one-logit-per-row shape and return its batch size."""
    if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
        batch_size = logits.shape[0]
    else:
        raise ValueError(f"{name} must have shape [batch] or [batch, 1].")
    if batch_size < 1:
        raise ValueError(f"{name} must contain at least one row.")
    return batch_size


def _validate_finite_scalar(value: float, name: str, *, minimum: float, maximum: float | None = None) -> None:
    """Validate a finite Python scalar against inclusive bounds."""
    if not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar.")
    if value < minimum or (maximum is not None and value > maximum):
        if maximum is None:
            raise ValueError(f"{name} must be at least {minimum}.")
        raise ValueError(f"{name} must be in [{minimum}, {maximum}].")


def ensemble_pessimistic(
    values: torch.Tensor,
    penalty: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce an ensemble to its mean, disagreement, and pessimistic value.

    Disagreement is the mean absolute difference over ordered pairs of
    distinct ensemble members. The operation is differentiable with respect
    to its inputs; callers choose whether values belong to a frozen target or
    a live actor path.

    Args:
        values: Ensemble values, shape [ensemble, ...].
        penalty: Non-negative disagreement coefficient broadcastable to the
            output shape.

    Returns:
        Mean, disagreement, and mean-minus-penalty-disagreement tensors, each
        with the leading ensemble dimension removed.
    """
    if values.ndim < 1 or values.shape[0] < 2:
        raise ValueError("Pessimistic reduction requires at least two ensemble members.")

    ensemble_size = values.shape[0]
    mean = values.mean(dim=0)
    if ensemble_size == 2:
        disagreement = (values[0] - values[1]).abs()
    else:
        pairwise_difference = (values.unsqueeze(0) - values.unsqueeze(1)).abs()
        disagreement = pairwise_difference.sum(dim=(0, 1)) / (ensemble_size * (ensemble_size - 1))
    return mean, disagreement, mean - penalty * disagreement


def forward_backward_loss(
    current_forward: torch.Tensor,
    current_backward: torch.Tensor,
    target_forward: torch.Tensor,
    target_backward: torch.Tensor,
    continuation: torch.Tensor,
    pessimism_penalty: float | torch.Tensor = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the finite-batch forward-backward measure loss.

    The target measure and continuation are detached inside this function. The
    resulting semi-gradient therefore updates only current forward and
    backward features. The loss sums over ensemble members rather than
    averaging them, matching the finite-batch objective.

    Args:
        current_forward: Current forward features, shape [ensemble, batch, latent].
        current_backward: Current backward features, shape [batch, latent].
        target_forward: Target forward features, shape [ensemble, batch, latent].
        target_backward: Target backward features, shape [batch, latent].
        continuation: Discount times bootstrap mask, shape [batch, 1].
        pessimism_penalty: Target-measure disagreement coefficient.

    Returns:
        Total, off-diagonal, and diagonal loss scalars.
    """
    if current_forward.ndim != 3:
        raise ValueError("current_forward must have shape [ensemble, batch, latent].")
    ensemble_size, batch_size, latent_size = current_forward.shape
    if ensemble_size < 2:
        raise ValueError("Forward-backward loss requires at least two ensemble members.")
    if batch_size < 2:
        raise ValueError("Forward-backward loss requires at least two batch rows.")
    if current_backward.shape != (batch_size, latent_size):
        raise ValueError("current_backward must have shape [batch, latent].")
    if target_forward.shape != current_forward.shape:
        raise ValueError("target_forward must match current_forward shape.")
    if target_backward.shape != current_backward.shape:
        raise ValueError("target_backward must match current_backward shape.")
    if continuation.shape != (batch_size, 1):
        raise ValueError("continuation must have shape [batch, 1].")

    with torch.no_grad():
        target_scores = target_forward @ target_backward.mT
        _, _, target_score = ensemble_pessimistic(target_scores, pessimism_penalty)

    current_scores = current_forward @ current_backward.mT
    residual = current_scores - continuation.detach().unsqueeze(0) * target_score
    residual_diagonal = residual.diagonal(dim1=-2, dim2=-1)
    off_diagonal_loss = (
        0.5 * (residual.square().sum() - residual_diagonal.square().sum()) / (batch_size * (batch_size - 1))
    )
    diagonal_loss = -residual_diagonal.sum() / batch_size
    return off_diagonal_loss + diagonal_loss, off_diagonal_loss, diagonal_loss


def backward_orthogonality_loss(
    backward: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the unbiased backward-feature whitening objective.

    This uses the exact covariance identity for the off-diagonal pair sum and
    avoids materializing a batch-by-batch Gram matrix.

    Args:
        backward: Backward features, shape [batch, latent].

    Returns:
        Total, off-diagonal, and diagonal loss scalars.
    """
    if backward.ndim != 2 or backward.shape[0] < 2 or backward.shape[1] < 1:
        raise ValueError("backward must have shape [batch, latent] with batch at least two.")

    batch_size = backward.shape[0]
    row_norm_squared = backward.square().sum(dim=-1)
    feature_scatter = backward.mT @ backward
    off_diagonal_numerator = feature_scatter.square().sum() - row_norm_squared.square().sum()
    off_diagonal_loss = 0.5 * off_diagonal_numerator / (batch_size * (batch_size - 1))
    diagonal_loss = -row_norm_squared.mean()
    return off_diagonal_loss + diagonal_loss, off_diagonal_loss, diagonal_loss


@torch.no_grad()
def backward_implied_reward(
    backward: torch.Tensor,
    contexts: torch.Tensor,
    covariance: torch.Tensor,
    ridge: float = 0.0,
) -> torch.Tensor:
    """Compute covariance-correct rewards implied by backward features.

    The covariance is supplied explicitly so an algorithm can estimate and
    freeze it independently from the states being rewarded. A zero ridge is
    the unregularized reference equation. Singular unregularized covariance
    raises from torch.linalg.solve; this function never silently substitutes a
    pseudoinverse.

    Args:
        backward: Backward features at states being rewarded, shape [batch, latent].
        contexts: Reward contexts aligned with those states, shape [batch, latent].
        covariance: Normalized backward-feature covariance, shape [latent, latent].
        ridge: Non-negative absolute diagonal regularizer.

    Returns:
        Detached scalar rewards, shape [batch].
    """
    if backward.ndim != 2:
        raise ValueError("backward must have shape [batch, latent].")
    batch_size, latent_size = backward.shape
    if contexts.shape != (batch_size, latent_size):
        raise ValueError("contexts must match backward shape.")
    if covariance.shape != (latent_size, latent_size):
        raise ValueError("covariance must have shape [latent, latent].")
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative.")

    solve_matrix = covariance
    if ridge > 0.0:
        solve_matrix = covariance + ridge * torch.eye(
            latent_size,
            dtype=covariance.dtype,
            device=covariance.device,
        )
    coefficients = torch.linalg.solve(solve_matrix, contexts.mT).mT
    return (backward * coefficients).sum(dim=-1)


def reward_value_td_loss(
    values: torch.Tensor,
    target_values: torch.Tensor,
    rewards: torch.Tensor,
    continuation: torch.Tensor,
    pessimism: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a detached pessimistic Bellman target and vector reward-value loss.

    Args:
        values: Live value ensemble, shape [ensemble, batch, channels].
        target_values: Target value ensemble, shape [ensemble, batch, channels].
        rewards: Frozen immediate reward channels, shape [batch, channels].
        continuation: Discount times bootstrap mask, shape [batch, 1].
        pessimism: Target-ensemble disagreement coefficient.

    Returns:
        Loss scalar and detached Bellman target with shape [batch, channels].
    """
    if values.ndim != 3:
        raise ValueError("values must have shape [ensemble, batch, channels].")
    ensemble_size, batch_size, channel_count = values.shape
    if ensemble_size < 2 or batch_size < 1 or channel_count < 1:
        raise ValueError("values must contain at least two ensemble members, one batch row, and one channel.")
    if target_values.shape != values.shape:
        raise ValueError("target_values must match values shape.")
    if rewards.shape != (batch_size, channel_count):
        raise ValueError("rewards must have shape [batch, channels].")
    if continuation.shape != (batch_size, 1):
        raise ValueError("continuation must have shape [batch, 1].")
    _validate_finite_scalar(pessimism, "pessimism", minimum=0.0)

    with torch.no_grad():
        _, _, pessimistic_target = ensemble_pessimistic(target_values, pessimism)
        target = rewards + continuation * pessimistic_target

    residual = values - target
    loss = 0.5 * residual.square().sum(dim=(0, 2)).mean()
    return loss, target


def discriminator_logistic_loss(
    expert_logits: torch.Tensor,
    replay_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute equal-class-weight logistic discrimination loss from logits.

    Args:
        expert_logits: Expert-pair logits, shape [expert_batch] or [expert_batch, 1].
        replay_logits: Replay-pair logits, shape [replay_batch] or [replay_batch, 1].

    Returns:
        Sum of the expert-positive and replay-negative mean losses.
    """
    _logit_batch_size(expert_logits, "expert_logits")
    _logit_batch_size(replay_logits, "replay_logits")
    return F.softplus(-expert_logits).mean() + F.softplus(replay_logits).mean()


def discriminator_gradient_penalty(
    interpolated_logits: torch.Tensor,
    interpolated_inputs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Compute the WGAN gradient penalty over discriminator-consumed inputs.

    Args:
        interpolated_logits: One discriminator logit per row, shape [batch] or [batch, 1].
        interpolated_inputs: Interpolated route tensors and context. Each tensor has batch as its first dimension
            and requires gradients.

    Returns:
        Mean squared deviation of the concatenated input-gradient norm from one.
    """
    batch_size = _logit_batch_size(interpolated_logits, "interpolated_logits")
    if not interpolated_logits.requires_grad:
        raise ValueError("interpolated_logits must require gradients.")
    if len(interpolated_inputs) < 1:
        raise ValueError("interpolated_inputs must contain at least one tensor.")
    for interpolated_input in interpolated_inputs:
        if interpolated_input.ndim < 1 or interpolated_input.shape[0] != batch_size:
            raise ValueError("Each interpolated input must have the same non-empty batch dimension as the logits.")
        if interpolated_input.numel() == 0:
            raise ValueError("Interpolated inputs must not be empty.")
        if not interpolated_input.requires_grad:
            raise ValueError("Interpolated inputs must require gradients.")

    gradients = torch.autograd.grad(
        outputs=interpolated_logits,
        inputs=interpolated_inputs,
        grad_outputs=torch.ones_like(interpolated_logits),
        create_graph=True,
        retain_graph=True,
    )
    gradient_norm_squared = gradients[0].reshape(batch_size, -1).square().sum(dim=1)
    for gradient in gradients[1:]:
        gradient_norm_squared = gradient_norm_squared + gradient.reshape(batch_size, -1).square().sum(dim=1)
    return (gradient_norm_squared.sqrt() - 1.0).square().mean()


def trajectory_context(
    backward_features: torch.Tensor,
    *,
    radius: float | None,
) -> torch.Tensor:
    """Construct a detached context from each contiguous backward-feature window.

    Args:
        backward_features: Backward features, shape [..., sequence_length, latent].
        radius: Projected context norm. ``None`` leaves the sequence mean unprojected.

    Returns:
        Detached trajectory contexts, shape [..., latent].
    """
    if backward_features.ndim < 2:
        raise ValueError("backward_features must have shape [..., sequence_length, latent].")
    if backward_features.numel() == 0 or backward_features.shape[-2] < 1 or backward_features.shape[-1] < 1:
        raise ValueError("backward_features must contain a non-empty sequence and latent dimension.")
    if radius is not None:
        _validate_finite_scalar(radius, "radius", minimum=0.0)

    context = backward_features.detach().mean(dim=-2)
    if radius is not None:
        context = radius * F.normalize(context, dim=-1)
    return context


def actor_direct_loss(
    fb_values: torch.Tensor,
    value_channels: torch.Tensor,
    channel_coefficients: torch.Tensor,
    *,
    scale_channels: bool,
) -> torch.Tensor:
    """Compute the direct-Q actor objective from separately pessimistic values.

    The owner must omit zero-configured helper channels instead of evaluating
    dummy heads. It must also freeze evaluator parameters and buffers outside
    this function while retaining their action-input graph.

    Args:
        fb_values: Pessimistic forward-backward values, shape [batch].
        value_channels: Separately pessimistic direct-Q helper channels, shape [batch, channels].
        channel_coefficients: Fixed base coefficient for each helper channel, shape [channels].
        scale_channels: Whether to scale helper channels by detached mean absolute FB value.

    Returns:
        Actor loss scalar. Value graphs remain live with respect to the sampled action.
    """
    if fb_values.ndim != 1 or fb_values.shape[0] < 1:
        raise ValueError("fb_values must have shape [batch] with at least one row.")
    batch_size = fb_values.shape[0]
    if value_channels.ndim != 2 or value_channels.shape[0] != batch_size:
        raise ValueError("value_channels must have shape [batch, channels].")
    if channel_coefficients.shape != (value_channels.shape[1],):
        raise ValueError("channel_coefficients must have shape [channels].")

    scale = fb_values.abs().mean().detach() if scale_channels else 1.0
    channel_value = (value_channels * channel_coefficients.detach()).sum(dim=-1).mean()
    return -fb_values.mean() - scale * channel_value


@torch.no_grad()
def soft_update(
    source: tuple[torch.Tensor, ...],
    target: tuple[torch.Tensor, ...],
    tau: float,
) -> None:
    """Move target tensors toward matching source tensors in place.

    Args:
        source: Ordered live tensors.
        target: Matching ordered target tensors to mutate.
        tau: Polyak interpolation coefficient.
    """
    if len(source) < 1 or len(target) < 1:
        raise ValueError("source and target must each contain at least one tensor.")
    if len(source) != len(target):
        raise ValueError("source and target must contain the same number of tensors.")
    for source_tensor, target_tensor in zip(source, target):
        if source_tensor.shape != target_tensor.shape:
            raise ValueError("Each source tensor must match its target tensor shape.")
    _validate_finite_scalar(tau, "tau", minimum=0.0, maximum=1.0)

    torch._foreach_mul_(target, 1.0 - tau)
    torch._foreach_add_(target, source, alpha=tau)
