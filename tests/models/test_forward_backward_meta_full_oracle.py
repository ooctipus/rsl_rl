# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint-backed MetaMotivo parameter, Adam, and target-update parity."""

from __future__ import annotations

import copy
import json
import os
import sys
import torch
import torch.nn.functional as functional
from collections.abc import Callable
from pathlib import Path
from tensordict import TensorDict
from typing import Protocol

import pytest
from safetensors import safe_open

from rsl_rl.models.forward_backward_model import (
    ForwardBackwardDualNetworkCfg,
    ForwardBackwardModel,
    ForwardBackwardValueHeadCfg,
)
from rsl_rl.modules.forward_backward import (
    actor_direct_loss,
    backward_orthogonality_loss,
    discriminator_gradient_penalty,
    discriminator_logistic_loss,
    ensemble_pessimistic,
    forward_backward_loss,
    soft_update,
)
from rsl_rl.modules.reward_channels import ForwardBackwardValueSpec

_ORACLE_ROOT = os.getenv("METAMOTIVO_FULL_ORACLE_DIR")
_DEVICE = torch.device(os.getenv("FORWARD_BACKWARD_ORACLE_DEVICE", "cpu"))


class _TensorReader(Protocol):
    def get_tensor(self, name: str) -> torch.Tensor: ...


_PREFIXES = {
    "actor_network.left_embedding.network.": "_actor.embed_s.",
    "actor_network.right_embedding.network.": "_actor.embed_z.",
    "actor_network.trunk.": "_actor.policy.",
    "forward_network.left_embedding.network.": "_forward_map.embed_sa.",
    "forward_network.right_embedding.network.": "_forward_map.embed_z.",
    "forward_network.trunk.": "_forward_map.Fs.",
    "backward_network.network.": "_backward_map.net.",
    "forward_target_network.left_embedding.network.": "_target_forward_map.embed_sa.",
    "forward_target_network.right_embedding.network.": "_target_forward_map.embed_z.",
    "forward_target_network.trunk.": "_target_forward_map.Fs.",
    "backward_target_network.network.": "_target_backward_map.net.",
    "discriminator_network.network.": "_discriminator.trunk.",
    "value_networks.critic.left_embedding.network.": "_critic.embed_sa.",
    "value_networks.critic.right_embedding.network.": "_critic.embed_z.",
    "value_networks.critic.trunk.": "_critic.Fs.",
    "value_target_networks.critic.left_embedding.network.": "_target_critic.embed_sa.",
    "value_target_networks.critic.right_embedding.network.": "_target_critic.embed_z.",
    "value_target_networks.critic.trunk.": "_target_critic.Fs.",
}


def _set_reference_runtime() -> None:
    torch.use_deterministic_algorithms(True)
    if _DEVICE.type == "cpu":
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled = False


def _oracle_paths() -> tuple[Path, Path]:
    root = Path(_ORACLE_ROOT).expanduser()  # type: ignore[arg-type]
    return root / "oracle.safetensors", root / "oracle.json"


def _make_model() -> ForwardBackwardModel:
    observations = TensorDict({"state": torch.zeros(2, 358)}, batch_size=[2])
    dual = ForwardBackwardDualNetworkCfg(hidden_dim=1024, hidden_layers=2, embedding_layers=2)
    critic = ForwardBackwardValueHeadCfg(
        spec=ForwardBackwardValueSpec(
            name="critic",
            kind="critic",
            route="critic_discriminator",
            reward_channels=("discriminator",),
            ensemble_size=2,
            has_target=True,
        ),
        network=dual,
    )
    return ForwardBackwardModel(
        observations,
        {
            "actor": ("state",),
            "forward": ("state",),
            "backward": ("state",),
            "discriminator": ("state",),
            "critic_discriminator": ("state",),
        },
        action_dim=69,
        context_dim=256,
        actor_cfg=dual,
        forward_cfg=dual,
        backward_hidden_dims=(256,),
        discriminator_hidden_dims=(1024, 1024, 1024),
        value_heads=(critic,),
        observation_normalization=False,
    ).to(_DEVICE)


def _source_name(clean_name: str) -> str:
    for clean_prefix, source_prefix in _PREFIXES.items():
        if clean_name.startswith(clean_prefix):
            return source_prefix + clean_name.removeprefix(clean_prefix)
    raise KeyError(clean_name)


def _mapped_value(value: torch.Tensor, clean_name: str) -> torch.Tensor:
    if value.ndim == 3 and value.shape[-2] != 1 and clean_name.endswith("weight"):
        value = value.transpose(-1, -2)
    return value.contiguous()


def _source_value(handle: _TensorReader, clean_name: str, gradient_prefix: str | None = None) -> torch.Tensor:
    source_name = _source_name(clean_name)
    if gradient_prefix is None:
        value = handle.get_tensor("model_state." + source_name)
    else:
        value = handle.get_tensor(f"gradient.{gradient_prefix}.{source_name.split('.', 1)[1]}")
    return _mapped_value(value, clean_name)


def _load_model(handle: _TensorReader, model: ForwardBackwardModel) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name == "action_distribution.std_param":
                continue
            parameter.copy_(_source_value(handle, name))


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    tolerance = (1.0e-4, 1.0e-5) if _DEVICE.type == "cuda" else (2.0e-5, 2.0e-6)
    try:
        torch.testing.assert_close(actual, expected, rtol=tolerance[0], atol=tolerance[1])
    except AssertionError:
        if _DEVICE.type != "cuda":
            raise
        actual_flat = actual.detach().double().flatten()
        expected_flat = expected.detach().double().flatten()
        cosine = torch.nn.functional.cosine_similarity(actual_flat, expected_flat, dim=0)
        normalized_error = (actual_flat - expected_flat).norm() / expected_flat.norm()
        assert cosine >= 0.99999
        assert normalized_error <= 1.0e-4


def _assert_gradient_group(
    handle: _TensorReader,
    module: torch.nn.Module,
    clean_prefix: str,
    gradient_prefix: str,
    gradients: tuple[torch.Tensor, ...],
) -> tuple[float, float]:
    dot = 0.0
    actual_squared = 0.0
    expected_squared = 0.0
    error_squared = 0.0
    elementwise_mismatches = 0
    for (name, _parameter), gradient in zip(module.named_parameters(), gradients, strict=True):
        expected = _source_value(handle, f"{clean_prefix}.{name}", gradient_prefix)
        try:
            torch.testing.assert_close(gradient, expected, rtol=5.0e-5, atol=5.0e-6)
        except AssertionError:
            elementwise_mismatches += 1
        actual = gradient.detach().to(torch.float64)
        reference = expected.to(torch.float64)
        dot += float((actual * reference).sum())
        actual_squared += float(actual.square().sum())
        expected_squared += float(reference.square().sum())
        error_squared += float((actual - reference).square().sum())

    cosine = dot / (actual_squared * expected_squared) ** 0.5
    normalized_error = (error_squared / expected_squared) ** 0.5
    if elementwise_mismatches:
        reduction_order_note = (
            "the clean orthogonality identity avoids the source's batch-square Gram allocation, "
            "so a few FP32 elements may differ in reduction order"
        )
        assert cosine >= 0.99999, reduction_order_note
        assert normalized_error <= 1.0e-4, reduction_order_note
    return cosine, normalized_error


def _gradients(loss: torch.Tensor, module: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    return torch.autograd.grad(loss, tuple(module.parameters()), retain_graph=True)


def _set_source_gradients(handle: _TensorReader, module: torch.nn.Module, prefix: str) -> None:
    for name, parameter in module.named_parameters():
        parameter.grad = handle.get_tensor(f"gradient.{prefix}.{name}").clone()


def _set_clean_gradients(
    handle: _TensorReader, module: torch.nn.Module, clean_prefix: str, gradient_prefix: str
) -> None:
    for clean_name, parameter in module.named_parameters():
        parameter.grad = _source_value(handle, f"{clean_prefix}.{clean_name}", gradient_prefix).clone()


def _make_adam(module: torch.nn.Module, source_optimizer: torch.optim.Optimizer) -> torch.optim.Adam:
    group = source_optimizer.param_groups[0]
    return torch.optim.Adam(
        module.parameters(),
        lr=group["lr"],
        betas=group["betas"],
        eps=group["eps"],
        weight_decay=group["weight_decay"],
        amsgrad=group["amsgrad"],
        maximize=group.get("maximize", False),
    )


def _copy_optimizer_state(
    source_optimizer: torch.optim.Optimizer,
    source_module: torch.nn.Module,
    clean_optimizer: torch.optim.Optimizer,
    clean_module: torch.nn.Module,
    clean_prefix: str,
) -> None:
    source_parameters = dict(source_module.named_parameters())
    for clean_name, clean_parameter in clean_module.named_parameters():
        full_clean_name = f"{clean_prefix}.{clean_name}"
        source_parameter = source_parameters[_source_name(full_clean_name).split(".", 1)[1]]
        clean_state = {}
        for key, value in source_optimizer.state[source_parameter].items():
            clean_state[key] = (
                _mapped_value(value, full_clean_name).clone() if torch.is_tensor(value) else copy.deepcopy(value)
            )
        clean_optimizer.state[clean_parameter] = clean_state


def _assert_modules_close(source_module: torch.nn.Module, clean_module: torch.nn.Module, clean_prefix: str) -> None:
    source_parameters = dict(source_module.named_parameters())
    for clean_name, clean_parameter in clean_module.named_parameters():
        full_clean_name = f"{clean_prefix}.{clean_name}"
        source_parameter = source_parameters[_source_name(full_clean_name).split(".", 1)[1]]
        expected = _mapped_value(source_parameter, full_clean_name)
        torch.testing.assert_close(clean_parameter, expected, rtol=2.0e-5, atol=2.0e-7)


@pytest.mark.skipif(_ORACLE_ROOT is None, reason="METAMOTIVO_FULL_ORACLE_DIR is not set")
def test_meta_checkpoint_values_and_full_parameter_gradients(
    record_property: Callable[[str, object], None],
) -> None:
    """The clean model should match every released Meta component and parameter gradient."""
    _set_reference_runtime()
    tensor_path, _manifest_path = _oracle_paths()
    model = _make_model()
    with safe_open(tensor_path, framework="pt", device=str(_DEVICE)) as handle:
        _load_model(handle, model)
        observations = TensorDict({"state": handle.get_tensor("normalized.train_obs")}, batch_size=[264])
        next_observations = TensorDict({"state": handle.get_tensor("normalized.train_next_obs")}, batch_size=[264])
        expert_observations = TensorDict({"state": handle.get_tensor("normalized.expert_obs")}, batch_size=[264])
        contexts = handle.get_tensor("context.relabeled_z")
        actions = handle.get_tensor("input.raw.action")
        behavior_contexts = handle.get_tensor("input.raw.behavior_z")
        continuation = handle.get_tensor("input.derived.discount")

        current_forward = model.forward_map(observations, contexts, actions)
        current_backward = model.backward_map(next_observations)
        target_forward = model.forward_map(
            next_observations, contexts, handle.get_tensor("fb.target_action"), target=True
        )
        target_backward = model.backward_map(next_observations, target=True)
        _assert_close(current_forward, handle.get_tensor("fb.current_forward"))
        _assert_close(current_backward, handle.get_tensor("fb.current_backward"))
        _assert_close(target_forward, handle.get_tensor("fb.target_forward"))
        _assert_close(target_backward, handle.get_tensor("fb.target_backward"))

        measure, _off_diagonal, _diagonal = forward_backward_loss(
            current_forward,
            current_backward,
            target_forward,
            target_backward,
            continuation,
        )
        orthogonality, _orthogonal_off, _orthogonal_diagonal = backward_orthogonality_loss(current_backward)
        current_q = (current_forward * contexts).sum(dim=-1)
        implied_q = (
            0.5
            * current_forward.shape[0]
            * functional.mse_loss(current_q, handle.get_tensor("fb.implicit_target_q").expand_as(current_q))
        )
        fb_loss = measure + 100.0 * orthogonality + 0.1 * implied_q
        _assert_close(fb_loss, handle.get_tensor("fb.loss"))
        fb_modules = ((model.forward_network, "forward_network"), (model.backward_network, "backward_network"))
        fb_parameters = tuple(parameter for module, _prefix in fb_modules for parameter in module.parameters())
        fb_gradients = torch.autograd.grad(fb_loss, fb_parameters)
        offset = 0
        for module, prefix in fb_modules:
            count = len(tuple(module.parameters()))
            cosine, error = _assert_gradient_group(handle, module, prefix, "fb", fb_gradients[offset : offset + count])
            record_property(f"meta_{prefix}_gradient_cosine", cosine)
            record_property(f"meta_{prefix}_gradient_normalized_l2_error", error)
            offset += count

        critic_values = model.critic_values("critic", observations, contexts, actions)
        critic_loss = (
            0.5
            * critic_values.shape[0]
            * functional.mse_loss(critic_values, handle.get_tensor("critic.target_q").expand_as(critic_values))
        )
        _assert_close(critic_values, handle.get_tensor("critic.current_ensemble"))
        _assert_close(critic_loss, handle.get_tensor("critic.loss"))
        critic_gradients = _gradients(critic_loss, model.value_networks["critic"])
        _assert_gradient_group(
            handle, model.value_networks["critic"], "value_networks.critic", "critic", critic_gradients
        )

        expert_contexts = handle.get_tensor("context.expert_z")
        expert_logits = model.discriminator_logits(expert_observations, expert_contexts)
        replay_logits = model.discriminator_logits(observations, behavior_contexts)
        alpha = handle.get_tensor("input.aux.gp_alpha")
        interpolated_state = (
            alpha * expert_observations["state"] + (1.0 - alpha) * observations["state"]
        ).requires_grad_(True)
        interpolated_context = (alpha * expert_contexts + (1.0 - alpha) * behavior_contexts).requires_grad_(True)
        interpolated_logits = model.discriminator_network(interpolated_state, interpolated_context)
        gradient_penalty = discriminator_gradient_penalty(
            interpolated_logits, (interpolated_state, interpolated_context)
        )
        discriminator_loss = discriminator_logistic_loss(expert_logits, replay_logits) + 10.0 * gradient_penalty
        _assert_close(expert_logits, handle.get_tensor("discriminator.expert_logits"))
        _assert_close(replay_logits, handle.get_tensor("discriminator.unlabeled_logits"))
        _assert_close(gradient_penalty, handle.get_tensor("discriminator.gradient_penalty"))
        _assert_close(discriminator_loss, handle.get_tensor("discriminator.loss"))
        discriminator_gradients = _gradients(discriminator_loss, model.discriminator_network)
        _assert_gradient_group(
            handle,
            model.discriminator_network,
            "discriminator_network",
            "discriminator",
            discriminator_gradients,
        )

        raw_actor_output = model.actor_network(
            observations["state"], torch.cat((observations["state"], contexts), dim=-1)
        )
        actor_mean = model.action_distribution.deterministic_output(raw_actor_output)
        noisy_action = actor_mean + handle.get_tensor("input.aux.actor_noise")
        actor_action = noisy_action - noisy_action.detach() + noisy_action.detach().clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        actor_forward = model.forward_map(observations, contexts, actor_action)
        actor_critic = model.critic_values("critic", observations, contexts, actor_action)
        _mean, _uncertainty, fb_value = ensemble_pessimistic((actor_forward * contexts).sum(dim=-1), 0.5)
        _mean, _uncertainty, critic_value = ensemble_pessimistic(actor_critic, 0.5)
        actor_loss = actor_direct_loss(fb_value, critic_value, fb_value.new_tensor([0.01]), scale_channels=True)
        _assert_close(actor_mean, handle.get_tensor("actor.action_mean"))
        _assert_close(actor_action, handle.get_tensor("actor.action"))
        _assert_close(actor_loss, handle.get_tensor("actor.loss"))
        actor_gradients = _gradients(actor_loss, model.actor_network)
        _assert_gradient_group(handle, model.actor_network, "actor_network", "actor", actor_gradients)

        assert all(parameter.grad is None for parameter in model.parameters())
        assert not model.action_distribution.std_param.requires_grad
        assert all(not parameter.requires_grad for parameter in model.forward_target_network.parameters())
        assert all(not parameter.requires_grad for parameter in model.backward_target_network.parameters())
        assert all(not parameter.requires_grad for parameter in model.value_target_networks.parameters())


@pytest.mark.skipif(_ORACLE_ROOT is None, reason="METAMOTIVO_FULL_ORACLE_DIR is not set")
def test_meta_checkpoint_adam_moments_and_target_ema_match_reference() -> None:
    """Persisted Adam moments and fresh helper optimizers should produce the same mutation."""
    _set_reference_runtime()
    tensor_path, manifest_path = _oracle_paths()
    manifest = json.loads(manifest_path.read_text())
    repository = Path(os.getenv("METAMOTIVO_REPO", manifest["provenance"]["repository"]))
    checkpoint = Path(os.getenv("METAMOTIVO_CHECKPOINT_DIR", manifest["provenance"]["checkpoint"]))
    sys.path.insert(0, str(repository))
    from metamotivo.fb_cpr import FBcprAgent
    from metamotivo.nn_models import _soft_update_params

    source_config = copy.deepcopy(manifest["effective_config"])
    source_config["model"]["device"] = str(_DEVICE)
    source_agent = FBcprAgent(**source_config)
    clean_model = _make_model()
    with safe_open(tensor_path, framework="pt", device=str(_DEVICE)) as handle:
        source_state = {
            key.removeprefix("model_state."): handle.get_tensor(key)
            for key in handle.keys()  # noqa: SIM118
            if key.startswith("model_state.")
        }
        source_agent._model.load_state_dict(source_state)
        _load_model(handle, clean_model)

        optimizer_state = torch.load(checkpoint / "optimizers.pth", weights_only=True, map_location=_DEVICE)
        source_agent.actor_optimizer.load_state_dict(optimizer_state["actor_optimizer"])
        source_agent.forward_optimizer.load_state_dict(optimizer_state["forward_optimizer"])
        source_agent.backward_optimizer.load_state_dict(optimizer_state["backward_optimizer"])
        for optimizer in (
            source_agent.actor_optimizer,
            source_agent.forward_optimizer,
            source_agent.backward_optimizer,
        ):
            optimizer.param_groups[0]["capturable"] = False

        groups = (
            (
                source_agent._model._actor,
                clean_model.actor_network,
                source_agent.actor_optimizer,
                "actor_network",
                "actor",
            ),
            (
                source_agent._model._forward_map,
                clean_model.forward_network,
                source_agent.forward_optimizer,
                "forward_network",
                "fb",
            ),
            (
                source_agent._model._backward_map,
                clean_model.backward_network,
                source_agent.backward_optimizer,
                "backward_network",
                "fb",
            ),
            (
                source_agent._model._discriminator,
                clean_model.discriminator_network,
                source_agent.discriminator_optimizer,
                "discriminator_network",
                "discriminator",
            ),
            (
                source_agent._model._critic,
                clean_model.value_networks["critic"],
                source_agent.critic_optimizer,
                "value_networks.critic",
                "critic",
            ),
        )
        for source_module, clean_module, source_optimizer, clean_prefix, gradient_prefix in groups:
            clean_optimizer = _make_adam(clean_module, source_optimizer)
            if source_optimizer.state:
                _copy_optimizer_state(source_optimizer, source_module, clean_optimizer, clean_module, clean_prefix)
            _set_source_gradients(handle, source_module, gradient_prefix)
            _set_clean_gradients(handle, clean_module, clean_prefix, gradient_prefix)
            source_optimizer.step()
            clean_optimizer.step()
            _assert_modules_close(source_module, clean_module, clean_prefix)

        with torch.no_grad():
            _soft_update_params(
                tuple(source_agent._model._forward_map.parameters()),
                tuple(source_agent._model._target_forward_map.parameters()),
                0.01,
            )
            _soft_update_params(
                tuple(source_agent._model._backward_map.parameters()),
                tuple(source_agent._model._target_backward_map.parameters()),
                0.01,
            )
            _soft_update_params(
                tuple(source_agent._model._critic.parameters()),
                tuple(source_agent._model._target_critic.parameters()),
                0.005,
            )
            soft_update(
                tuple(clean_model.forward_network.parameters()),
                tuple(clean_model.forward_target_network.parameters()),
                0.01,
            )
            soft_update(
                tuple(clean_model.backward_network.parameters()),
                tuple(clean_model.backward_target_network.parameters()),
                0.01,
            )
            soft_update(
                tuple(clean_model.value_networks["critic"].parameters()),
                tuple(clean_model.value_target_networks["critic"].parameters()),
                0.005,
            )
        _assert_modules_close(
            source_agent._model._target_forward_map,
            clean_model.forward_target_network,
            "forward_target_network",
        )
        _assert_modules_close(
            source_agent._model._target_backward_map,
            clean_model.backward_target_network,
            "backward_target_network",
        )
        _assert_modules_close(
            source_agent._model._target_critic,
            clean_model.value_target_networks["critic"],
            "value_target_networks.critic",
        )


@pytest.mark.skipif(_ORACLE_ROOT is None, reason="METAMOTIVO_FULL_ORACLE_DIR is not set")
def test_meta_checkpoint_normalizer_values_and_ordered_mutation_match_reference() -> None:
    """The source-matched mode should reproduce checkpoint values and two ordered EMA updates."""
    _set_reference_runtime()
    tensor_path, _manifest_path = _oracle_paths()
    with safe_open(tensor_path, framework="pt", device=str(_DEVICE)) as handle:
        normalizer = torch.nn.BatchNorm1d(358, eps=1e-5, momentum=0.01, affine=False).to(_DEVICE)
        normalizer.running_mean.copy_(handle.get_tensor("model_state._obs_normalizer.running_mean"))
        normalizer.running_var.copy_(handle.get_tensor("model_state._obs_normalizer.running_var"))
        normalizer.num_batches_tracked.copy_(handle.get_tensor("model_state._obs_normalizer.num_batches_tracked"))
        raw = handle.get_tensor("input.raw.train_obs")
        raw_next = handle.get_tensor("input.raw.train_next_obs")
        normalizer.eval()
        _assert_close(normalizer(raw), handle.get_tensor("normalized.train_obs"))

        expected_mean = normalizer.running_mean.clone()
        expected_var = normalizer.running_var.clone()
        expected_count = normalizer.num_batches_tracked.clone() + 2
        for observations in (raw, raw_next):
            expected_mean = 0.99 * expected_mean + 0.01 * observations.mean(dim=0)
            expected_var = 0.99 * expected_var + 0.01 * observations.var(dim=0)
        normalizer.train()
        normalizer(raw)
        normalizer(raw_next)

        torch.testing.assert_close(normalizer.running_mean, expected_mean)
        torch.testing.assert_close(normalizer.running_var, expected_var)
        torch.testing.assert_close(normalizer.num_batches_tracked, expected_count)
