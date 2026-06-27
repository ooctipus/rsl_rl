# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in parity checks against the MetaMotivo and BFM-Zero Phase 0 oracles.

Set ``METAMOTIVO_ORACLE_DIR`` and/or ``BFM_ZERO_ORACLE_DIR`` to an oracle
directory (or repository containing it). The external tensors remain owned by
their source repositories; this test never copies them into RSL-RL.
"""

from __future__ import annotations

import os
import torch
import torch.nn.functional as F  # noqa: N812
from collections.abc import Callable
from pathlib import Path

import pytest

from rsl_rl.modules.forward_backward import (
    actor_direct_loss,
    backward_implied_reward,
    backward_orthogonality_loss,
    discriminator_logistic_loss,
    ensemble_pessimistic,
    forward_backward_loss,
    reward_value_td_loss,
    trajectory_context,
)


def _load_candidate(path: Path, sentinel: str) -> dict[str, torch.Tensor] | None:
    """Load a matching safetensor candidate, or return none."""
    try:
        from safetensors import safe_open
    except ImportError:
        pytest.skip("safetensors is unavailable")

    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            if sentinel in keys:
                return {key: handle.get_tensor(key) for key in keys}
    except Exception:  # A repository may contain unrelated safetensor files.
        return None
    return None


def _load_oracle(env_name: str, sentinel: str) -> dict[str, torch.Tensor]:
    root_value = os.environ.get(env_name)
    if not root_value:
        pytest.skip(f"{env_name} is not set")

    root = Path(root_value).expanduser()
    if not root.exists():
        pytest.skip(f"{env_name} does not exist: {root}")
    candidates = [root] if root.is_file() else sorted(root.rglob("*.safetensors"))
    for path in candidates:
        tensors = _load_candidate(path, sentinel)
        if tensors is not None:
            return tensors
    pytest.skip(f"no oracle containing {sentinel!r} was found below {root}")


@pytest.fixture(scope="module")
def meta_oracle() -> dict[str, torch.Tensor]:
    """Load MetaMotivo oracle tensors."""
    return _load_oracle("METAMOTIVO_ORACLE_DIR", "fb.current_forward")


@pytest.fixture(scope="module")
def bfm_oracle() -> dict[str, torch.Tensor]:
    """Load BFM-Zero oracle tensors."""
    return _load_oracle("BFM_ZERO_ORACLE_DIR", "loss.fb.measures")


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=2.0e-5, atol=2.0e-6)


def _assert_gradient_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=5.0e-5, atol=5.0e-6)


def test_meta_fb_values_and_pessimism(meta_oracle: dict[str, torch.Tensor]) -> None:
    """Match MetaMotivo FB values and pessimistic reductions."""
    total, off_diagonal, diagonal = forward_backward_loss(
        meta_oracle["fb.current_forward"],
        meta_oracle["fb.current_backward"],
        meta_oracle["fb.target_forward"],
        meta_oracle["fb.target_backward"],
        meta_oracle["input.derived.discount"],
    )
    orthogonal, orthogonal_off, orthogonal_diagonal = backward_orthogonality_loss(meta_oracle["fb.current_backward"])

    _assert_close(off_diagonal, meta_oracle["fb.off_diagonal"])
    _assert_close(diagonal, meta_oracle["fb.diagonal"])
    _assert_close(total, meta_oracle["fb.off_diagonal"] + meta_oracle["fb.diagonal"])
    _assert_close(orthogonal, meta_oracle["fb.orthogonal_loss"])
    _assert_close(orthogonal_off, meta_oracle["fb.orthogonal_off_diagonal"])
    _assert_close(orthogonal_diagonal, meta_oracle["fb.orthogonal_diagonal"])

    mean, uncertainty, pessimistic = ensemble_pessimistic(meta_oracle["fb.target_measures"], 0.0)
    _assert_close(mean, meta_oracle["fb.target_measure_mean"])
    _assert_close(uncertainty, meta_oracle["fb.target_measure_uncertainty"])
    _assert_close(pessimistic, meta_oracle["fb.target_measure"])

    _, actor_uncertainty, actor_value = ensemble_pessimistic(meta_oracle["actor.fb_ensemble"], 0.5)
    _assert_close(actor_uncertainty, meta_oracle["actor.fb_uncertainty"])
    _assert_close(actor_value, meta_oracle["actor.fb_q"])


def test_bfm_fb_values_pessimism_and_output_gradients(
    bfm_oracle: dict[str, torch.Tensor], record_property: Callable[[str, object], None]
) -> None:
    """Match BFM-Zero FB values, reductions, and output gradients."""
    current_forward = bfm_oracle["model.forward.behavior"].clone().requires_grad_()
    current_backward = bfm_oracle["model.backward.goal"].clone().requires_grad_()
    fb_total, off_diagonal, diagonal = forward_backward_loss(
        current_forward,
        current_backward,
        bfm_oracle["model.target_forward.next"],
        bfm_oracle["model.target_backward.goal"],
        bfm_oracle["input.discount"],
    )
    orthogonal, orthogonal_off, orthogonal_diagonal = backward_orthogonality_loss(current_backward)
    total = fb_total + 100.0 * orthogonal
    forward_gradient, backward_gradient = torch.autograd.grad(total, (current_forward, current_backward))

    _assert_close(off_diagonal, bfm_oracle["loss.fb.off_diagonal"].squeeze())
    _assert_close(diagonal, bfm_oracle["loss.fb.diagonal"].squeeze())
    _assert_close(orthogonal, bfm_oracle["loss.fb.orthogonal"].squeeze())
    _assert_close(orthogonal_off, bfm_oracle["loss.fb.orthogonal_off_diagonal"].squeeze())
    _assert_close(orthogonal_diagonal, bfm_oracle["loss.fb.orthogonal_diagonal"].squeeze())
    _assert_close(total, bfm_oracle["loss.fb.total"].squeeze())
    _assert_gradient_close(forward_gradient, bfm_oracle["loss.fb.grad_forward_output"])

    expected_backward_gradient = bfm_oracle["loss.fb.grad_backward_output"]
    cosine = F.cosine_similarity(
        backward_gradient.flatten().double(), expected_backward_gradient.flatten().double(), dim=0
    )
    normalized_error = (backward_gradient - expected_backward_gradient).double().norm() / (
        expected_backward_gradient.double().norm()
    )
    record_property("bfm_backward_gradient_cosine", float(cosine))
    record_property("bfm_backward_gradient_normalized_l2_error", float(normalized_error))
    reduction_order_note = (
        "elementwise equality is not expected because covariance and pairwise reductions use different FP32 "
        "reduction orders"
    )
    assert cosine >= 0.99999, reduction_order_note
    assert normalized_error <= 1.0e-4, reduction_order_note

    mean, uncertainty, pessimistic = ensemble_pessimistic(bfm_oracle["loss.fb.target_measures"], 0.0)
    _assert_close(mean, bfm_oracle["loss.fb.target_measure"])
    _assert_close(uncertainty, bfm_oracle["loss.fb.target_measures"].diff(dim=0).squeeze(0).abs())
    _assert_close(pessimistic, bfm_oracle["loss.fb.target_measure"])
    _, actor_uncertainty, actor_value = ensemble_pessimistic(bfm_oracle["actor.q_fb_all"], 0.5)
    _assert_close(actor_uncertainty, bfm_oracle["actor.q_fb_uncertainty"])
    _assert_close(actor_value, bfm_oracle["actor.q_fb"])


def test_meta_cpr_td_target_and_loss(meta_oracle: dict[str, torch.Tensor]) -> None:
    """Match the MetaMotivo CPR target and loss."""
    loss, target = reward_value_td_loss(
        meta_oracle["critic.current_ensemble"],
        meta_oracle["critic.target_ensemble"],
        meta_oracle["critic.discriminator_reward"],
        meta_oracle["input.derived.discount"],
        pessimism=0.5,
    )
    _assert_close(target, meta_oracle["critic.target_q"])
    _assert_close(loss, meta_oracle["critic.loss"])


@pytest.mark.parametrize(
    ("prefix", "value_key", "target_value_key", "reward_key"),
    [
        ("loss.cpr", "model.critic.behavior", "model.target_critic.next", "model.discriminator.reward"),
        ("loss.aux", "model.aux_critic.behavior", "model.target_aux_critic.next", "loss.aux.normalized_reward"),
    ],
)
def test_bfm_reward_td_targets_losses_and_output_gradients(
    bfm_oracle: dict[str, torch.Tensor], prefix: str, value_key: str, target_value_key: str, reward_key: str
) -> None:
    """Match BFM-Zero CPR and auxiliary TD values and gradients."""
    values = bfm_oracle[value_key].clone().requires_grad_()
    loss, target = reward_value_td_loss(
        values,
        bfm_oracle[target_value_key],
        bfm_oracle[reward_key],
        bfm_oracle["input.discount"],
        pessimism=0.5,
    )
    (gradient,) = torch.autograd.grad(loss, (values,))
    _assert_close(target, bfm_oracle[f"{prefix}.target"])
    _assert_close(loss, bfm_oracle[f"{prefix}.total"].squeeze())
    _assert_gradient_close(gradient, bfm_oracle[f"{prefix}.grad_q_output"])


def test_meta_discriminator_total_from_stored_penalty(meta_oracle: dict[str, torch.Tensor]) -> None:
    """Match the MetaMotivo discriminator total using its stored GP."""
    logistic = discriminator_logistic_loss(
        meta_oracle["discriminator.expert_logits"], meta_oracle["discriminator.unlabeled_logits"]
    )
    total = logistic + 10.0 * meta_oracle["discriminator.gradient_penalty"]
    _assert_close(total, meta_oracle["discriminator.loss"])


def test_bfm_discriminator_total_from_stored_penalty(bfm_oracle: dict[str, torch.Tensor]) -> None:
    """Match the BFM-Zero discriminator total using its stored GP."""
    logistic = discriminator_logistic_loss(
        bfm_oracle["model.discriminator.expert_logits"], bfm_oracle["model.discriminator.train_logits"]
    )
    total = logistic + 10.0 * bfm_oracle["loss.discriminator.gradient_penalty"].squeeze()
    _assert_close(total, bfm_oracle["loss.discriminator.total"].squeeze())


def test_meta_trajectory_context(meta_oracle: dict[str, torch.Tensor]) -> None:
    """Match MetaMotivo's released trajectory contexts."""
    backward = meta_oracle["context.backward_expert"].reshape(-1, 8, 256)
    context = trajectory_context(backward, radius=16.0)
    _assert_close(context, meta_oracle["context.expert_per_sequence"])
    _assert_close(context.repeat_interleave(8, dim=0), meta_oracle["context.expert_z"])


def test_bfm_trajectory_context(bfm_oracle: dict[str, torch.Tensor]) -> None:
    """Match BFM-Zero's released trajectory context."""
    context = trajectory_context(bfm_oracle["model.backward.expert"].unsqueeze(0), radius=16.0)
    _assert_close(context.repeat_interleave(8, dim=0), bfm_oracle["latent.expert_z"])


def test_meta_actor_direct_loss(meta_oracle: dict[str, torch.Tensor]) -> None:
    """Match MetaMotivo's released direct actor loss."""
    coefficients = meta_oracle["actor.fb_q"].new_tensor([0.01])
    loss = actor_direct_loss(
        meta_oracle["actor.fb_q"], meta_oracle["actor.critic_q"], coefficients, scale_channels=True
    )
    _assert_close(meta_oracle["actor.fb_q"].abs().mean(), meta_oracle["actor.cpr_scale"])
    _assert_close(loss, meta_oracle["actor.loss"])


def test_bfm_actor_direct_loss(bfm_oracle: dict[str, torch.Tensor]) -> None:
    """Match BFM-Zero's released direct actor loss."""
    value_channels = torch.cat((bfm_oracle["actor.q_discriminator"], bfm_oracle["actor.q_aux"]), dim=-1)
    coefficients = value_channels.new_tensor([0.05, 0.02])
    loss = actor_direct_loss(bfm_oracle["actor.q_fb"], value_channels, coefficients, scale_channels=True)
    _assert_close(bfm_oracle["actor.q_fb"].abs().mean(), bfm_oracle["actor.q_fb_weight"].squeeze())
    _assert_close(loss, bfm_oracle["actor.loss_total"].squeeze())


def test_meta_implied_reward_solve_residual(
    meta_oracle: dict[str, torch.Tensor], record_property: Callable[[str, object], None]
) -> None:
    """Check the MetaMotivo covariance solve without matching stored inverse bits."""
    backward = meta_oracle["fb.current_backward"].double()
    contexts = meta_oracle["context.relabeled_z"].double()
    covariance = meta_oracle["fb.latent_covariance"].double()
    reward = backward_implied_reward(backward, contexts, covariance)
    coefficients = torch.linalg.solve(covariance, contexts.mT)
    residual = (covariance @ coefficients - contexts.mT).norm() / contexts.norm()
    record_property("meta_implied_reward_relative_solve_residual", float(residual))
    assert torch.isfinite(reward).all()
    assert residual < 1.0e-10
    _assert_close(reward, (backward * coefficients.mT).sum(dim=-1))
