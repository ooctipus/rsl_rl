# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for core forward-backward tensor operators."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F  # noqa: N812

import pytest

from rsl_rl.modules.forward_backward import (
    actor_direct_loss,
    backward_implied_reward,
    backward_orthogonality_loss,
    discriminator_gradient_penalty,
    discriminator_logistic_loss,
    ensemble_pessimistic,
    forward_backward_loss,
    reward_value_td_loss,
    soft_update,
    trajectory_context,
    trajectory_context_sequence,
)

_VALUE_RTOL = 1.0e-10
_VALUE_ATOL = 1.0e-12
_GRAD_RTOL = 1.0e-9
_GRAD_ATOL = 1.0e-11


def _pessimistic_loop(
    values: torch.Tensor,
    penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent ordered-pair pessimism oracle."""
    ensemble_size = values.shape[0]
    mean = sum(values[index] for index in range(ensemble_size)) / ensemble_size
    disagreement = sum(
        (values[left] - values[right]).abs()
        for left in range(ensemble_size)
        for right in range(ensemble_size)
        if left != right
    ) / (ensemble_size * (ensemble_size - 1))
    return mean, disagreement, mean - penalty * disagreement


def _forward_backward_loop(
    current_forward: torch.Tensor,
    current_backward: torch.Tensor,
    target_forward: torch.Tensor,
    target_backward: torch.Tensor,
    continuation: torch.Tensor,
    pessimism_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent scalar-loop forward-backward oracle."""
    ensemble_size, batch_size, latent_size = current_forward.shape

    with torch.no_grad():
        target_scores = torch.stack([
            torch.stack([
                torch.stack([
                    sum(
                        target_forward[ensemble, row, feature] * target_backward[goal, feature]
                        for feature in range(latent_size)
                    )
                    for goal in range(batch_size)
                ])
                for row in range(batch_size)
            ])
            for ensemble in range(ensemble_size)
        ])
        _, _, target_score = _pessimistic_loop(target_scores, pessimism_penalty)

    current_scores = torch.stack([
        torch.stack([
            torch.stack([
                sum(
                    current_forward[ensemble, row, feature] * current_backward[goal, feature]
                    for feature in range(latent_size)
                )
                for goal in range(batch_size)
            ])
            for row in range(batch_size)
        ])
        for ensemble in range(ensemble_size)
    ])
    residual = current_scores - continuation.detach().unsqueeze(0) * target_score
    off_diagonal_loss = sum(
        residual[ensemble, row, goal].square()
        for ensemble in range(ensemble_size)
        for row in range(batch_size)
        for goal in range(batch_size)
        if row != goal
    ) / (2 * batch_size * (batch_size - 1))
    diagonal_loss = (
        -sum(residual[ensemble, row, row] for ensemble in range(ensemble_size) for row in range(batch_size))
        / batch_size
    )
    return off_diagonal_loss + diagonal_loss, off_diagonal_loss, diagonal_loss


def _orthogonality_loop(
    backward: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent pairwise backward-orthogonality oracle."""
    batch_size = backward.shape[0]
    off_diagonal_loss = sum(
        (backward[row] * backward[other]).sum().square()
        for row in range(batch_size)
        for other in range(batch_size)
        if row != other
    ) / (2 * batch_size * (batch_size - 1))
    diagonal_loss = -sum(backward[row].square().sum() for row in range(batch_size)) / batch_size
    return off_diagonal_loss + diagonal_loss, off_diagonal_loss, diagonal_loss


def test_ensemble_pessimistic_two_member_values_and_gradients() -> None:
    """The two-head shortcut should equal the ordered-pair definition."""
    values = torch.tensor([[1.0, 4.0], [3.0, 0.0]], dtype=torch.float64, requires_grad=True)

    mean, disagreement, pessimistic = ensemble_pessimistic(values, penalty=0.25)

    torch.testing.assert_close(mean, torch.tensor([2.0, 2.0], dtype=torch.float64))
    torch.testing.assert_close(disagreement, torch.tensor([2.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(pessimistic, torch.tensor([1.5, 1.0], dtype=torch.float64))
    pessimistic.sum().backward()
    torch.testing.assert_close(
        values.grad,
        torch.tensor([[0.75, 0.25], [0.25, 0.75]], dtype=torch.float64),
        rtol=_GRAD_RTOL,
        atol=_GRAD_ATOL,
    )


def test_ensemble_pessimistic_three_member_values_and_gradients() -> None:
    """The general ordered-pair reduction should preserve exact ensemble scaling."""
    values = torch.tensor([0.0, 2.0, 5.0], dtype=torch.float64, requires_grad=True)

    mean, disagreement, pessimistic = ensemble_pessimistic(values, penalty=0.25)

    torch.testing.assert_close(mean, torch.tensor(7.0 / 3.0, dtype=torch.float64))
    torch.testing.assert_close(disagreement, torch.tensor(10.0 / 3.0, dtype=torch.float64))
    torch.testing.assert_close(pessimistic, torch.tensor(1.5, dtype=torch.float64))
    pessimistic.backward()
    torch.testing.assert_close(
        values.grad,
        torch.tensor([0.5, 1.0 / 3.0, 1.0 / 6.0], dtype=torch.float64),
        rtol=_GRAD_RTOL,
        atol=_GRAD_ATOL,
    )


def test_ensemble_pessimistic_preserves_trailing_shape() -> None:
    """Only the leading ensemble dimension should be reduced."""
    values = torch.arange(24, dtype=torch.float64).reshape(3, 2, 4)

    outputs = ensemble_pessimistic(values, penalty=0.5)

    assert all(output.shape == (2, 4) for output in outputs)


def test_ensemble_pessimistic_rejects_one_member() -> None:
    """Disagreement is undefined for a one-member ensemble."""
    with pytest.raises(ValueError, match="at least two"):
        ensemble_pessimistic(torch.ones(1, 3), penalty=0.5)


def test_forward_backward_loss_hand_oracle_and_gradient_ownership() -> None:
    """The finite-batch loss should match hand values and detach all targets."""
    current_forward = torch.tensor(
        [[[1.0], [2.0]], [[3.0], [4.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    current_backward = torch.tensor([[5.0], [7.0]], dtype=torch.float64, requires_grad=True)
    target_forward = torch.tensor(
        [[[1.0], [-1.0]], [[2.0], [3.0]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target_backward = torch.tensor([[2.0], [4.0]], dtype=torch.float64, requires_grad=True)
    continuation = torch.tensor([[0.5], [0.0]], dtype=torch.float64, requires_grad=True)
    penalty = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)

    total, off_diagonal, diagonal = forward_backward_loss(
        current_forward,
        current_backward,
        target_forward,
        target_backward,
        continuation,
        pessimism_penalty=penalty,
    )

    torch.testing.assert_close(off_diagonal, torch.tensor(215.625, dtype=torch.float64))
    torch.testing.assert_close(diagonal, torch.tensor(-29.75, dtype=torch.float64))
    torch.testing.assert_close(total, torch.tensor(185.875, dtype=torch.float64))
    total.backward()
    torch.testing.assert_close(
        current_forward.grad,
        torch.tensor([[[13.25], [21.5]], [[62.25], [46.5]]], dtype=torch.float64),
        rtol=_GRAD_RTOL,
        atol=_GRAD_ATOL,
    )
    torch.testing.assert_close(
        current_backward.grad,
        torch.tensor([[48.0], [27.0]], dtype=torch.float64),
        rtol=_GRAD_RTOL,
        atol=_GRAD_ATOL,
    )
    assert target_forward.grad is None
    assert target_backward.grad is None
    assert continuation.grad is None
    assert penalty.grad is None


def test_forward_backward_loss_matches_independent_loop_oracle() -> None:
    """Vectorized E=3 values and gradients should match explicit scalar loops."""
    current_forward_data = (torch.arange(18, dtype=torch.float64).reshape(3, 3, 2) - 7.0) / 5.0
    current_backward_data = (torch.arange(6, dtype=torch.float64).reshape(3, 2) - 2.0) / 4.0
    target_forward = (torch.arange(18, dtype=torch.float64).reshape(3, 3, 2) + 1.0) / 9.0
    target_backward = (torch.arange(6, dtype=torch.float64).reshape(3, 2) - 1.0) / 7.0
    continuation = torch.tensor([[0.9], [0.0], [0.4]], dtype=torch.float64)

    actual_forward = current_forward_data.clone().requires_grad_(True)
    actual_backward = current_backward_data.clone().requires_grad_(True)
    actual = forward_backward_loss(
        actual_forward,
        actual_backward,
        target_forward,
        target_backward,
        continuation,
        pessimism_penalty=0.3,
    )
    actual_gradients = torch.autograd.grad(actual[0], (actual_forward, actual_backward))

    oracle_forward = current_forward_data.clone().requires_grad_(True)
    oracle_backward = current_backward_data.clone().requires_grad_(True)
    expected = _forward_backward_loop(
        oracle_forward,
        oracle_backward,
        target_forward,
        target_backward,
        continuation,
        pessimism_penalty=0.3,
    )
    expected_gradients = torch.autograd.grad(expected[0], (oracle_forward, oracle_backward))

    for actual_term, expected_term in zip(actual, expected):
        torch.testing.assert_close(actual_term, expected_term, rtol=_VALUE_RTOL, atol=_VALUE_ATOL)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=_GRAD_RTOL, atol=_GRAD_ATOL)


def test_forward_backward_loss_rejects_ambiguous_continuation_shape() -> None:
    """A flat continuation could silently broadcast over goals instead of transitions."""
    forward = torch.ones(2, 3, 2)
    backward = torch.ones(3, 2)

    with pytest.raises(ValueError, match=r"\[batch, 1\]"):
        forward_backward_loss(forward, backward, forward, backward, torch.ones(3))


def test_forward_backward_loss_requires_two_rows_and_heads() -> None:
    """Both off-diagonal sampling and pessimism require at least two items."""
    with pytest.raises(ValueError, match="ensemble"):
        forward_backward_loss(
            torch.ones(1, 2, 1),
            torch.ones(2, 1),
            torch.ones(1, 2, 1),
            torch.ones(2, 1),
            torch.ones(2, 1),
        )
    with pytest.raises(ValueError, match="batch"):
        forward_backward_loss(
            torch.ones(2, 1, 1),
            torch.ones(1, 1),
            torch.ones(2, 1, 1),
            torch.ones(1, 1),
            torch.ones(1, 1),
        )


def test_backward_orthogonality_loss_hand_oracle() -> None:
    """The covariance identity should match a hand-computed pairwise gradient."""
    backward = torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=torch.float64, requires_grad=True)

    total, off_diagonal, diagonal = backward_orthogonality_loss(backward)

    torch.testing.assert_close(off_diagonal, torch.tensor(0.5, dtype=torch.float64))
    torch.testing.assert_close(diagonal, torch.tensor(-7.5, dtype=torch.float64))
    torch.testing.assert_close(total, torch.tensor(-7.0, dtype=torch.float64))
    total.backward()
    torch.testing.assert_close(
        backward.grad,
        torch.tensor([[2.0, -3.0], [-2.0, 3.0]], dtype=torch.float64),
        rtol=_GRAD_RTOL,
        atol=_GRAD_ATOL,
    )


def test_backward_orthogonality_loss_matches_pairwise_oracle() -> None:
    """The non-pairwise identity should preserve pairwise values and gradients."""
    data = (torch.arange(15, dtype=torch.float64).reshape(5, 3) - 6.0) / 7.0
    actual_backward = data.clone().requires_grad_(True)
    actual = backward_orthogonality_loss(actual_backward)
    actual_gradient = torch.autograd.grad(actual[0], actual_backward)[0]

    oracle_backward = data.clone().requires_grad_(True)
    expected = _orthogonality_loop(oracle_backward)
    expected_gradient = torch.autograd.grad(expected[0], oracle_backward)[0]

    for actual_term, expected_term in zip(actual, expected):
        torch.testing.assert_close(actual_term, expected_term, rtol=_VALUE_RTOL, atol=_VALUE_ATOL)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=_GRAD_RTOL, atol=_GRAD_ATOL)


def test_backward_orthogonality_loss_rejects_one_row() -> None:
    """Independent off-diagonal pairs require at least two rows."""
    with pytest.raises(ValueError, match="at least two"):
        backward_orthogonality_loss(torch.ones(1, 3))


def test_backward_implied_reward_spd_and_ridge_oracles() -> None:
    """Unregularized and ridge solves should match hand-computed rewards."""
    backward = torch.tensor(
        [[1.0, 2.0], [3.0, -1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    contexts = torch.tensor(
        [[2.0, 4.0], [4.0, 8.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    covariance = torch.diag(torch.tensor([2.0, 4.0], dtype=torch.float64)).requires_grad_(True)

    reward = backward_implied_reward(backward, contexts, covariance)
    ridge_reward = backward_implied_reward(backward, contexts, covariance, ridge=2.0)

    torch.testing.assert_close(reward, torch.tensor([3.0, 4.0], dtype=torch.float64))
    torch.testing.assert_close(
        ridge_reward,
        torch.tensor([11.0 / 6.0, 5.0 / 3.0], dtype=torch.float64),
        rtol=_VALUE_RTOL,
        atol=_VALUE_ATOL,
    )
    assert not reward.requires_grad
    assert not ridge_reward.requires_grad
    assert backward.grad is None
    assert contexts.grad is None
    assert covariance.grad is None


def test_backward_implied_reward_singular_and_ridge_behavior() -> None:
    """A singular exact solve should fail while an explicit ridge should succeed."""
    backward = torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=torch.float64)
    contexts = torch.tensor([[1.0, 2.0], [2.0, 1.0]], dtype=torch.float64)
    singular = torch.diag(torch.tensor([1.0, 0.0], dtype=torch.float64))

    with pytest.raises(torch.linalg.LinAlgError):
        backward_implied_reward(backward, contexts, singular)

    reward = backward_implied_reward(backward, contexts, singular, ridge=0.5)
    expected_coefficients = torch.tensor(
        [[2.0 / 3.0, 4.0], [4.0 / 3.0, 2.0]],
        dtype=torch.float64,
    )
    expected = (backward * expected_coefficients).sum(dim=-1)
    torch.testing.assert_close(reward, expected, rtol=_VALUE_RTOL, atol=_VALUE_ATOL)


def test_backward_implied_reward_rejects_negative_ridge_and_wrong_shapes() -> None:
    """Ridge and matrix shapes should preserve the stated solve semantics."""
    backward = torch.ones(2, 3)
    contexts = torch.ones(2, 3)
    covariance = torch.eye(3)

    with pytest.raises(ValueError, match="non-negative"):
        backward_implied_reward(backward, contexts, covariance, ridge=-1.0)
    with pytest.raises(ValueError, match="contexts"):
        backward_implied_reward(backward, torch.ones(3, 3), covariance)
    with pytest.raises(ValueError, match="covariance"):
        backward_implied_reward(backward, contexts, torch.eye(2))


@torch.no_grad()
def _td_target_loop(
    target_values: torch.Tensor,
    rewards: torch.Tensor,
    continuation: torch.Tensor,
    pessimism: float,
) -> torch.Tensor:
    """Independent scalar-loop pessimistic Bellman-target oracle."""
    ensemble_size, batch_size, channel_count = target_values.shape
    rows = []
    for batch_index in range(batch_size):
        channels = []
        for channel_index in range(channel_count):
            predictions = [
                target_values[ensemble_index, batch_index, channel_index] for ensemble_index in range(ensemble_size)
            ]
            mean = sum(predictions) / ensemble_size
            disagreement = sum(
                (predictions[left] - predictions[right]).abs()
                for left in range(ensemble_size)
                for right in range(ensemble_size)
                if left != right
            ) / (ensemble_size * (ensemble_size - 1))
            channels.append(
                rewards[batch_index, channel_index] + continuation[batch_index, 0] * (mean - pessimism * disagreement)
            )
        rows.append(torch.stack(channels))
    return torch.stack(rows)


def test_reward_value_td_loss_matches_fp64_loop_and_gradient_oracle() -> None:
    """Vector reward TD values and live gradients should match scalar loops."""
    ensemble_size, batch_size, channel_count = 3, 4, 2
    values = ((torch.arange(24, dtype=torch.float64).reshape(3, 4, 2) - 9.0) / 7.0).requires_grad_(True)
    target_values = ((torch.arange(24, dtype=torch.float64).reshape(3, 4, 2) + 2.0) / 11.0).requires_grad_(True)
    rewards = torch.tensor(
        [[0.5, -1.0], [1.5, 0.25], [-0.75, 2.0], [0.0, -0.5]],
        dtype=torch.float64,
        requires_grad=True,
    )
    continuation = torch.tensor([[0.0], [0.9], [0.4], [0.9]], dtype=torch.float64, requires_grad=True)

    loss, target = reward_value_td_loss(values, target_values, rewards, continuation, pessimism=0.3)
    expected_target = _td_target_loop(target_values, rewards, continuation, pessimism=0.3)
    expected_loss = sum(
        (values[ensemble, batch, channel].detach() - expected_target[batch, channel]).square()
        for ensemble in range(ensemble_size)
        for batch in range(batch_size)
        for channel in range(channel_count)
    ) / (2 * batch_size)

    torch.testing.assert_close(target, expected_target, rtol=_VALUE_RTOL, atol=_VALUE_ATOL)
    torch.testing.assert_close(loss, expected_loss, rtol=_VALUE_RTOL, atol=_VALUE_ATOL)
    loss.backward()
    torch.testing.assert_close(
        values.grad,
        (values.detach() - expected_target) / batch_size,
        rtol=_GRAD_RTOL,
        atol=_GRAD_ATOL,
    )
    assert not target.requires_grad
    assert target_values.grad is None
    assert rewards.grad is None
    assert continuation.grad is None


def test_reward_value_td_loss_scalar_reference_and_channel_sum() -> None:
    """J=1 should match released scaling and vector loss should sum scalar channels."""
    values = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[2.0, 0.0, 5.0], [3.0, 7.0, 1.0]],
        ],
        dtype=torch.float64,
    )
    target_values = torch.tensor(
        [
            [[3.0, 5.0, 1.0], [8.0, 2.0, 4.0]],
            [[1.0, 7.0, 6.0], [6.0, 9.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    rewards = torch.tensor([[0.5, -0.5, 1.0], [2.0, 1.5, -1.0]], dtype=torch.float64)
    continuation = torch.tensor([[0.0], [0.8]], dtype=torch.float64)

    vector_loss, vector_target = reward_value_td_loss(
        values,
        target_values,
        rewards,
        continuation,
        pessimism=0.5,
    )
    scalar_results = [
        reward_value_td_loss(
            values[:, :, channel : channel + 1],
            target_values[:, :, channel : channel + 1],
            rewards[:, channel : channel + 1],
            continuation,
            pessimism=0.5,
        )
        for channel in range(values.shape[-1])
    ]

    torch.testing.assert_close(vector_loss, sum(result[0] for result in scalar_results))
    torch.testing.assert_close(vector_target, torch.cat([result[1] for result in scalar_results], dim=-1))
    torch.testing.assert_close(vector_target[0], rewards[0])
    first_scalar_target = scalar_results[0][1]
    reference_loss = (
        0.5
        * values.shape[0]
        * F.mse_loss(
            values[:, :, :1],
            first_scalar_target.expand(values.shape[0], -1, -1),
        )
    )
    torch.testing.assert_close(scalar_results[0][0], reference_loss)


def test_reward_value_td_loss_rejects_ambiguous_shapes_and_range() -> None:
    """Malformed TD metadata must fail instead of broadcasting into another axis."""
    values = torch.zeros(2, 2, 2)
    target_values = torch.ones_like(values)
    rewards = torch.zeros(2, 2)

    with pytest.raises(ValueError, match="continuation"):
        reward_value_td_loss(values, target_values, rewards, torch.ones(2), pessimism=0.0)
    with pytest.raises(ValueError, match="target_values"):
        reward_value_td_loss(values, target_values[:, :1], rewards, torch.ones(2, 1), pessimism=0.0)
    with pytest.raises(ValueError, match="rewards"):
        reward_value_td_loss(values, target_values, rewards[:, :1], torch.ones(2, 1), pessimism=0.0)
    with pytest.raises(ValueError, match="one batch row"):
        reward_value_td_loss(values[:, :0], target_values[:, :0], rewards[:0], torch.ones(0, 1), pessimism=0.0)
    with pytest.raises(ValueError, match="pessimism"):
        reward_value_td_loss(values, target_values, rewards, torch.ones(2, 1), pessimism=-0.1)


def test_discriminator_logistic_loss_is_stable_for_extreme_logits() -> None:
    """Logit-space softplus should avoid probability saturation and logarithms."""
    expert_logits = torch.tensor([-1000.0, 1000.0], dtype=torch.float64, requires_grad=True)
    replay_logits = torch.tensor([-1000.0, 1000.0], dtype=torch.float64, requires_grad=True)

    loss = discriminator_logistic_loss(expert_logits, replay_logits)
    expected = F.softplus(-expert_logits).mean() + F.softplus(replay_logits).mean()

    torch.testing.assert_close(loss, expected)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(expert_logits.grad).all()
    assert torch.isfinite(replay_logits.grad).all()


def test_discriminator_logistic_loss_uses_separate_class_means_and_gradients() -> None:
    """Unequal batch counts should retain equal class weighting and exact logit gradients."""
    expert_logits = torch.tensor([-2.0, 0.5], dtype=torch.float64, requires_grad=True)
    replay_logits = torch.tensor([-1.0, 0.0, 3.0], dtype=torch.float64, requires_grad=True)

    loss = discriminator_logistic_loss(expert_logits, replay_logits)
    loss.backward()

    torch.testing.assert_close(
        expert_logits.grad,
        (expert_logits.detach().sigmoid() - 1.0) / expert_logits.numel(),
        rtol=_GRAD_RTOL,
        atol=_GRAD_ATOL,
    )
    torch.testing.assert_close(
        replay_logits.grad,
        replay_logits.detach().sigmoid() / replay_logits.numel(),
        rtol=_GRAD_RTOL,
        atol=_GRAD_ATOL,
    )


def test_discriminator_logistic_loss_rejects_empty_and_multi_logit_rows() -> None:
    """Each discriminator class must provide one non-empty logit per row."""
    with pytest.raises(ValueError, match="at least one row"):
        discriminator_logistic_loss(torch.empty(0), torch.ones(1))
    with pytest.raises(ValueError, match=r"\[batch\] or \[batch, 1\]"):
        discriminator_logistic_loss(torch.ones(2, 2), torch.ones(2))


def test_discriminator_gradient_penalty_matches_multi_route_linear_oracle() -> None:
    """The GP norm and second-order parameter gradient should use all consumed routes."""
    batch_size = 5
    route = torch.linspace(-1.0, 1.0, batch_size * 2, dtype=torch.float64).reshape(batch_size, 2)
    context = torch.linspace(0.5, 1.5, batch_size, dtype=torch.float64).reshape(batch_size, 1)
    route.requires_grad_(True)
    context.requires_grad_(True)
    weights = torch.tensor([0.3, -0.4, 1.2], dtype=torch.float64, requires_grad=True)
    logits = (route * weights[:2]).sum(dim=-1, keepdim=True) + context * weights[2]

    penalty = discriminator_gradient_penalty(logits, (route, context))
    weight_norm = weights.detach().norm()
    expected_penalty = (weight_norm - 1.0).square()

    torch.testing.assert_close(penalty, expected_penalty, rtol=_VALUE_RTOL, atol=_VALUE_ATOL)
    penalty.backward()
    expected_gradient = 2.0 * (weight_norm - 1.0) * weights.detach() / weight_norm
    torch.testing.assert_close(
        weights.grad,
        expected_gradient,
        rtol=_GRAD_RTOL,
        atol=_GRAD_ATOL,
    )


def test_discriminator_gradient_penalty_rejects_invalid_route_metadata() -> None:
    """GP inputs must be non-empty differentiable routes aligned by row."""
    route = torch.ones(2, 1, requires_grad=True)
    logits = route.square()

    with pytest.raises(ValueError, match="at least one tensor"):
        discriminator_gradient_penalty(logits, ())
    mismatched = torch.ones(3, 1, requires_grad=True)
    with pytest.raises(ValueError, match="same non-empty batch"):
        discriminator_gradient_penalty(logits, (mismatched,))
    empty = torch.ones(2, 0, requires_grad=True)
    with pytest.raises(ValueError, match="must not be empty"):
        discriminator_gradient_penalty(logits, (empty,))
    detached = torch.ones(2, 1)
    with pytest.raises(ValueError, match="must require gradients"):
        discriminator_gradient_penalty(logits, (detached,))


def test_trajectory_context_length_eight_projection_and_detach() -> None:
    """Released length-eight contexts should mean first, project once, and detach B."""
    features = ((torch.arange(64, dtype=torch.float64).reshape(2, 8, 4) - 20.0) / 13.0).requires_grad_(True)

    context = trajectory_context(features, radius=2.0)
    expected_mean = features.detach().mean(dim=1)
    expected = 2.0 * F.normalize(expected_mean, dim=-1)

    torch.testing.assert_close(context, expected, rtol=_VALUE_RTOL, atol=_VALUE_ATOL)
    torch.testing.assert_close(context.norm(dim=-1), torch.full((2,), 2.0, dtype=torch.float64))
    assert not context.requires_grad
    assert features.grad is None

    cancelling = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]], dtype=torch.float64)
    torch.testing.assert_close(
        trajectory_context(cancelling, radius=math.sqrt(2.0)),
        torch.zeros(1, 2, dtype=torch.float64),
    )


def test_trajectory_context_matches_sliding_window_loop() -> None:
    """One operator should serve both expert batches and rolling BFM windows."""
    sequence = (torch.arange(36, dtype=torch.float64).reshape(12, 3) - 12.0) / 9.0
    windows = torch.stack([sequence[start : start + 8] for start in range(5)])

    actual = trajectory_context(windows, radius=math.sqrt(3.0))
    expected = torch.stack([
        math.sqrt(3.0) * F.normalize(sequence[start : start + 8].mean(dim=0), dim=-1) for start in range(5)
    ])

    torch.testing.assert_close(actual, expected, rtol=_VALUE_RTOL, atol=_VALUE_ATOL)
    torch.testing.assert_close(trajectory_context(windows, radius=None), windows.mean(dim=1))


def test_trajectory_context_sequence_matches_full_and_partial_window_loops() -> None:
    """One prefix-sum operator should serve rollout and evaluator context sequences."""
    features = (torch.arange(30, dtype=torch.float64).reshape(2, 5, 3) - 8.0) / 7.0
    full = trajectory_context_sequence(features, 3, include_partial=False, radius=None)
    partial = trajectory_context_sequence(features, 3, include_partial=True, radius=math.sqrt(3.0))

    expected_full = torch.stack([
        torch.stack([row[start : start + 3].mean(dim=0) for start in range(3)]) for row in features
    ])
    expected_partial = torch.stack([
        torch.stack([
            math.sqrt(3.0) * F.normalize(row[start : min(start + 3, 5)].mean(dim=0), dim=-1) for start in range(5)
        ])
        for row in features
    ])

    torch.testing.assert_close(full, expected_full, rtol=_VALUE_RTOL, atol=_VALUE_ATOL)
    torch.testing.assert_close(partial, expected_partial, rtol=_VALUE_RTOL, atol=_VALUE_ATOL)


def test_trajectory_context_rejects_empty_shapes_and_invalid_radius() -> None:
    """Trajectory contexts require an actual sequence and a finite non-negative radius."""
    with pytest.raises(ValueError, match="must have shape"):
        trajectory_context(torch.ones(3), radius=None)
    with pytest.raises(ValueError, match="non-empty sequence"):
        trajectory_context(torch.empty(2, 0, 3), radius=None)
    with pytest.raises(ValueError, match="radius"):
        trajectory_context(torch.ones(2, 3), radius=-1.0)


def test_signed_cost_is_applied_once_from_td_target_to_actor() -> None:
    """A non-negative cost becomes a negative reward once and keeps that sign in the actor."""
    values = torch.zeros(2, 2, 2, dtype=torch.float64)
    target_values = torch.ones_like(values)
    rewards = torch.tensor([[2.0, -3.0], [4.0, -1.0]], dtype=torch.float64)
    continuation = torch.zeros(2, 1, dtype=torch.float64)

    _, signed_channels = reward_value_td_loss(values, target_values, rewards, continuation, pessimism=0.0)
    torch.testing.assert_close(signed_channels, rewards)
    actor_loss = actor_direct_loss(
        torch.zeros(2, dtype=torch.float64),
        signed_channels[:, 1:],
        torch.tensor([0.5], dtype=torch.float64),
        scale_channels=False,
    )
    expected_loss = -0.5 * signed_channels[:, 1].mean()
    torch.testing.assert_close(actor_loss, expected_loss)
    assert actor_loss > 0.0


def test_actor_direct_loss_rejects_empty_and_mismatched_metadata() -> None:
    """Actor value tensors must share a non-empty batch and exact channel width."""
    with pytest.raises(ValueError, match="at least one row"):
        actor_direct_loss(torch.empty(0), torch.empty(0, 0), torch.empty(0), scale_channels=False)
    with pytest.raises(ValueError, match="value_channels"):
        actor_direct_loss(torch.ones(2), torch.ones(3, 1), torch.ones(1), scale_channels=False)
    with pytest.raises(ValueError, match="channel_coefficients"):
        actor_direct_loss(torch.ones(2), torch.ones(2, 2), torch.ones(1), scale_channels=False)


def test_actor_direct_loss_uses_separate_pessimism_before_composition() -> None:
    """Meta/BFM channels should be reduced separately even when ensemble heads are anti-aligned."""
    fb_ensemble = torch.tensor([[2.0, 2.0], [4.0, 4.0]], dtype=torch.float64)
    channel_ensembles = torch.tensor(
        [
            [[10.0, 0.0], [10.0, 0.0]],
            [[0.0, 10.0], [0.0, 10.0]],
        ],
        dtype=torch.float64,
    )
    coefficients = torch.tensor([0.05, 0.02], dtype=torch.float64)
    _, _, fb_values = ensemble_pessimistic(fb_ensemble, penalty=0.5)
    _, _, channel_values = ensemble_pessimistic(channel_ensembles, penalty=0.5)

    separate_loss = actor_direct_loss(
        fb_values,
        channel_values,
        coefficients,
        scale_channels=True,
    )
    scale = fb_values.abs().mean()
    expected = -fb_values.mean() - scale * (channel_values * coefficients).sum(dim=-1).mean()
    composed_ensemble = fb_ensemble + scale * (channel_ensembles * coefficients).sum(dim=-1)
    _, _, jointly_pessimistic = ensemble_pessimistic(composed_ensemble, penalty=0.5)

    torch.testing.assert_close(separate_loss, expected)
    assert not torch.isclose(separate_loss, -jointly_pessimistic.mean())


def test_actor_direct_loss_detaches_scale_and_coefficients() -> None:
    """Adaptive scale and fixed weights should not acquire actor gradients."""
    fb_values = torch.tensor([2.0, 4.0], dtype=torch.float64, requires_grad=True)
    value_channels = torch.tensor([[3.0, 5.0], [7.0, 11.0]], dtype=torch.float64, requires_grad=True)
    coefficients = torch.tensor([0.5, 0.0], dtype=torch.float64, requires_grad=True)

    loss = actor_direct_loss(fb_values, value_channels, coefficients, scale_channels=True)
    loss.backward()

    torch.testing.assert_close(fb_values.grad, torch.full_like(fb_values, -0.5))
    torch.testing.assert_close(
        value_channels.grad,
        torch.tensor([[-0.75, 0.0], [-0.75, 0.0]], dtype=torch.float64),
    )
    assert coefficients.grad is None


def test_actor_direct_loss_empty_helper_channels_equal_pure_fb() -> None:
    """An empty helper schema should reduce exactly to the pure FB actor objective."""
    fb_values = torch.tensor([1.0, -2.0, 4.0], dtype=torch.float64, requires_grad=True)
    value_channels = torch.empty((3, 0), dtype=torch.float64, requires_grad=True)
    coefficients = torch.empty(0, dtype=torch.float64, requires_grad=True)

    loss = actor_direct_loss(fb_values, value_channels, coefficients, scale_channels=True)

    torch.testing.assert_close(loss, -fb_values.mean())
    loss.backward()
    torch.testing.assert_close(fb_values.grad, torch.full_like(fb_values, -1.0 / 3.0))
    assert value_channels.grad is not None
    assert value_channels.grad.numel() == 0
    assert coefficients.grad is None


def test_actor_direct_loss_zero_fb_scale_removes_helper_gradient() -> None:
    """Reference scaling should disable helper contribution exactly when FB scale is zero."""
    fb_values = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    value_channels = torch.tensor(
        [[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    coefficients = torch.tensor([0.5, 0.25], dtype=torch.float64, requires_grad=True)

    loss = actor_direct_loss(fb_values, value_channels, coefficients, scale_channels=True)

    torch.testing.assert_close(loss, torch.zeros((), dtype=torch.float64))
    loss.backward()
    torch.testing.assert_close(fb_values.grad, torch.full_like(fb_values, -1.0 / 3.0))
    torch.testing.assert_close(value_channels.grad, torch.zeros_like(value_channels))
    assert coefficients.grad is None


def test_actor_direct_loss_keeps_action_path_with_frozen_value_parameters() -> None:
    """Frozen F/Q parameters should retain a nonzero derivative through the sampled action."""
    actor_parameter = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))
    forward_parameter = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64), requires_grad=False)
    critic_parameter = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64), requires_grad=False)
    coefficients = torch.tensor([0.2], dtype=torch.float64, requires_grad=True)
    observations = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    actions = actor_parameter * observations
    fb_values = forward_parameter * actions
    value_channels = (critic_parameter * actions).unsqueeze(-1)

    loss = actor_direct_loss(fb_values, value_channels, coefficients, scale_channels=False)
    loss.backward()

    expected_actor_gradient = -(forward_parameter + 0.2 * critic_parameter) * observations.mean()
    torch.testing.assert_close(actor_parameter.grad, expected_actor_gradient)
    assert actor_parameter.grad.abs() > 0.0
    assert forward_parameter.grad is None
    assert critic_parameter.grad is None
    assert coefficients.grad is None


def test_soft_update_rejects_empty_and_mismatched_tensor_sequences() -> None:
    """Polyak updates require non-empty one-to-one tensor metadata."""
    tensors = (torch.ones(1),)

    with pytest.raises(ValueError, match="at least one tensor"):
        soft_update((), (), tau=0.5)
    with pytest.raises(ValueError, match="same number"):
        soft_update(tensors, (torch.ones(1), torch.ones(1)), tau=0.5)
    with pytest.raises(ValueError, match="match its target tensor shape"):
        soft_update(tensors, (torch.ones(2),), tau=0.5)


@pytest.mark.parametrize("tau", (-0.1, 1.1, float("nan")))
def test_soft_update_rejects_out_of_range_tau(tau: float) -> None:
    """A Polyak coefficient must be finite and remain inside its convex range."""
    source = (torch.ones(1),)
    target = (torch.zeros(1),)

    with pytest.raises(ValueError, match="tau"):
        soft_update(source, target, tau)


@pytest.mark.parametrize("tau", (0.0, 1.0, 0.25))
def test_soft_update_matches_exact_one_step_mutation(tau: float) -> None:
    """Tau endpoints and an intermediate update should preserve source and avoid autograd state."""
    source = (
        torch.tensor([1.0, -2.0], dtype=torch.float64, requires_grad=True),
        torch.tensor([[3.0], [5.0]], dtype=torch.float64, requires_grad=True),
    )
    target = (
        torch.nn.Parameter(torch.tensor([9.0, 4.0], dtype=torch.float64)),
        torch.nn.Parameter(torch.tensor([[-1.0], [7.0]], dtype=torch.float64)),
    )
    source_before = tuple(tensor.detach().clone() for tensor in source)
    target_before = tuple(tensor.detach().clone() for tensor in target)

    soft_update(source, target, tau)

    for source_tensor, original_source, target_tensor, original_target in zip(
        source,
        source_before,
        target,
        target_before,
    ):
        expected = (1.0 - tau) * original_target + tau * original_source
        if tau in (0.0, 1.0):
            torch.testing.assert_close(target_tensor, expected, rtol=0.0, atol=0.0)
        else:
            torch.testing.assert_close(
                target_tensor,
                expected,
                rtol=_VALUE_RTOL,
                atol=_VALUE_ATOL,
            )
        torch.testing.assert_close(source_tensor, original_source, rtol=0.0, atol=0.0)
        assert target_tensor.grad is None
        assert target_tensor.grad_fn is None
