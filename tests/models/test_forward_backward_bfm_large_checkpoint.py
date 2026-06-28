# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Released-scale BFM-Zero F/B/Q checkpoint and gradient parity."""

from __future__ import annotations

import gc
import json
import os
import sys
import torch
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
from rsl_rl.modules.forward_backward import soft_update
from rsl_rl.modules.reward_channels import ForwardBackwardValueSpec

_ORACLE_ROOT = os.getenv("BFM_ZERO_ORACLE_DIR")
_DEVICE = torch.device(os.getenv("FORWARD_BACKWARD_ORACLE_DEVICE", "cpu"))


class _TensorReader(Protocol):
    def get_tensor(self, name: str) -> torch.Tensor: ...


def _paths() -> tuple[Path, Path, Path]:
    root = Path(_ORACLE_ROOT).expanduser()  # type: ignore[arg-type]
    manifest_path = root / "oracle.json"
    manifest = json.loads(manifest_path.read_text())
    return (
        root / "oracle_tensors.safetensors",
        manifest_path,
        Path(os.getenv("BFM_ZERO_CHECKPOINT_DIR", manifest["source"]["checkpoint"])),
    )


def _set_runtime() -> None:
    torch.use_deterministic_algorithms(True)
    if _DEVICE.type == "cpu":
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled = False


def _observations(handle: _TensorReader, prefix: str) -> TensorDict:
    fields = {
        name: handle.get_tensor(f"{prefix}.{name}")
        for name in ("state", "privileged_state", "last_action", "history_actor")
    }
    return TensorDict(fields, batch_size=[fields["state"].shape[0]], device=_DEVICE)


def _routes() -> dict[str, tuple[str, ...]]:
    value_route = ("state", "privileged_state", "last_action", "history_actor")
    return {
        "actor": ("state",),
        "forward": value_route,
        "backward": ("state", "privileged_state"),
        "critic_discriminator": value_route,
    }


def _make_representation_model() -> ForwardBackwardModel:
    observations = TensorDict(
        {
            "state": torch.zeros(2, 64, device=_DEVICE),
            "privileged_state": torch.zeros(2, 463, device=_DEVICE),
            "last_action": torch.zeros(2, 29, device=_DEVICE),
            "history_actor": torch.zeros(2, 372, device=_DEVICE),
        },
        batch_size=[2],
        device=_DEVICE,
    )
    return ForwardBackwardModel(
        observations,
        _routes(),
        action_dim=29,
        context_dim=256,
        actor_cfg=ForwardBackwardDualNetworkCfg(16, 1, 2),
        forward_cfg=ForwardBackwardDualNetworkCfg(2048, 6, 6, True),
        backward_hidden_dims=(256,),
        normalization_type="none",
    ).to(_DEVICE)


def _make_value_model() -> ForwardBackwardModel:
    observations = TensorDict(
        {
            "state": torch.zeros(2, 64, device=_DEVICE),
            "privileged_state": torch.zeros(2, 463, device=_DEVICE),
            "last_action": torch.zeros(2, 29, device=_DEVICE),
            "history_actor": torch.zeros(2, 372, device=_DEVICE),
        },
        batch_size=[2],
        device=_DEVICE,
    )
    value_cfg = ForwardBackwardDualNetworkCfg(2048, 6, 6, True)
    head = ForwardBackwardValueHeadCfg(
        spec=ForwardBackwardValueSpec(
            name="value",
            kind="critic",
            route="critic_discriminator",
            reward_channels=("reward",),
            ensemble_size=2,
            has_target=True,
        ),
        network=value_cfg,
    )
    return ForwardBackwardModel(
        observations,
        _routes(),
        action_dim=29,
        context_dim=256,
        actor_cfg=ForwardBackwardDualNetworkCfg(16, 1, 2),
        forward_cfg=ForwardBackwardDualNetworkCfg(16, 1, 2),
        backward_hidden_dims=(16,),
        value_heads=(head,),
        normalization_type="none",
    ).to(_DEVICE)


def _dual_source_name(clean_name: str, clean_prefix: str, source_prefix: str) -> str:
    relative = clean_name.removeprefix(clean_prefix + ".")
    replacements = {
        "left_embedding.network.": "embed_sa.",
        "right_embedding.network.": "embed_z.",
        "trunk.": "Fs.",
    }
    for clean_stem, source_stem in replacements.items():
        if relative.startswith(clean_stem):
            suffix = relative.removeprefix(clean_stem)
            suffix = suffix.replace(".normalization.", ".mlp.0.")
            suffix = suffix.replace(".linear.", ".mlp.1.")
            return f"{source_prefix}.{source_stem}{suffix}"
    raise KeyError(clean_name)


def _backward_source_name(clean_name: str, source_prefix: str) -> str:
    return f"{source_prefix}.net.{clean_name.removeprefix('backward_network.network.')}"


def _to_clean(value: torch.Tensor, clean_name: str) -> torch.Tensor:
    if value.ndim == 3 and clean_name.endswith("linear.weight"):
        value = value.transpose(-1, -2)
    return value.contiguous()


def _to_source(value: torch.Tensor, clean_name: str) -> torch.Tensor:
    return _to_clean(value, clean_name)


def _load_clean(
    weights: _TensorReader,
    module: torch.nn.Module,
    clean_prefix: str,
    source_prefix: str,
    *,
    backward: bool = False,
) -> None:
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            full_name = f"{clean_prefix}.{name}"
            source_name = (
                _backward_source_name(full_name, source_prefix)
                if backward
                else _dual_source_name(full_name, clean_prefix, source_prefix)
            )
            parameter.copy_(_to_clean(weights.get_tensor(source_name), full_name))


def _load_source(module: torch.nn.Module, weights: _TensorReader, source_prefix: str) -> None:
    marker = source_prefix + "."
    state = {
        key.removeprefix(marker): weights.get_tensor(key)
        for key in weights.keys()  # noqa: SIM118
        if key.startswith(marker)
    }
    module.load_state_dict(state, strict=True, assign=True)
    module.to(_DEVICE).eval().requires_grad_(True)


def _assert_output(actual: torch.Tensor, expected: torch.Tensor) -> None:
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


def _assert_gradient_parity(
    source_module: torch.nn.Module,
    source_gradients: tuple[torch.Tensor, ...],
    clean_module: torch.nn.Module,
    clean_prefix: str,
    source_prefix: str,
    clean_gradients: tuple[torch.Tensor, ...],
    *,
    backward: bool = False,
) -> None:
    source = dict(zip((name for name, _parameter in source_module.named_parameters()), source_gradients, strict=True))
    dot = 0.0
    source_squared = 0.0
    clean_squared = 0.0
    error_squared = 0.0
    for (name, _parameter), clean_gradient in zip(clean_module.named_parameters(), clean_gradients, strict=True):
        full_name = f"{clean_prefix}.{name}"
        source_name = (
            _backward_source_name(full_name, source_prefix).removeprefix(source_prefix + ".")
            if backward
            else _dual_source_name(full_name, clean_prefix, source_prefix).removeprefix(source_prefix + ".")
        )
        expected = source[source_name].double()
        actual = _to_source(clean_gradient, full_name).double()
        dot += float((actual * expected).sum())
        source_squared += float(expected.square().sum())
        clean_squared += float(actual.square().sum())
        error_squared += float((actual - expected).square().sum())
    assert dot / (source_squared * clean_squared) ** 0.5 >= 0.99999
    assert (error_squared / source_squared) ** 0.5 <= 1.0e-4


def _set_optimizer_gradients(
    source_module: torch.nn.Module,
    source_gradients: tuple[torch.Tensor, ...],
    clean_module: torch.nn.Module,
    clean_prefix: str,
    source_prefix: str,
    *,
    backward: bool = False,
) -> None:
    source = dict(zip((name for name, _parameter in source_module.named_parameters()), source_gradients, strict=True))
    for parameter, gradient in zip(source_module.parameters(), source_gradients, strict=True):
        parameter.grad = gradient
    for name, parameter in clean_module.named_parameters():
        full_name = f"{clean_prefix}.{name}"
        source_name = (
            _backward_source_name(full_name, source_prefix).removeprefix(source_prefix + ".")
            if backward
            else _dual_source_name(full_name, clean_prefix, source_prefix).removeprefix(source_prefix + ".")
        )
        parameter.grad = _to_clean(source[source_name], full_name).clone()


def _assert_parameters(
    source_module: torch.nn.Module,
    clean_module: torch.nn.Module,
    clean_prefix: str,
    source_prefix: str,
    *,
    backward: bool = False,
) -> None:
    source = dict(source_module.named_parameters())
    for name, parameter in clean_module.named_parameters():
        full_name = f"{clean_prefix}.{name}"
        source_name = (
            _backward_source_name(full_name, source_prefix).removeprefix(source_prefix + ".")
            if backward
            else _dual_source_name(full_name, clean_prefix, source_prefix).removeprefix(source_prefix + ".")
        )
        torch.testing.assert_close(parameter, _to_clean(source[source_name], full_name), rtol=2e-5, atol=2e-7)


@pytest.mark.skipif(_ORACLE_ROOT is None, reason="BFM_ZERO_ORACLE_DIR is not set")
def test_bfm_released_forward_backward_values_gradients_adam_and_ema() -> None:
    """Released-scale F/B and targets should match through one optimizer/EMA mutation."""
    _set_runtime()
    tensor_path, manifest_path, checkpoint = _paths()
    repository = Path(os.getenv("BFM_ZERO_REPO", json.loads(manifest_path.read_text())["source"]["repository"]))
    sys.path.insert(0, str(repository))
    from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelConfig
    from humanoidverse.agents.nn_models import _soft_update_params

    config_data = json.loads((checkpoint / "model/config.json").read_text())
    config_data["device"] = _DEVICE.type
    config = FBcprAuxModelConfig(**config_data)
    init_kwargs = json.loads((checkpoint / "model/init_kwargs.json").read_text())
    observation_space = json_to_space(init_kwargs["obs_space"])
    action_dim = int(init_kwargs["action_dim"])
    source_forward = config.archi.f.build(observation_space, config.archi.z_dim, action_dim)
    source_target_forward = config.archi.f.build(observation_space, config.archi.z_dim, action_dim)
    source_backward = config.archi.b.build(observation_space, config.archi.z_dim)
    source_target_backward = config.archi.b.build(observation_space, config.archi.z_dim)
    clean = _make_representation_model()

    with safe_open(checkpoint / "model/model.safetensors", framework="pt", device=str(_DEVICE)) as weights:
        _load_source(source_forward, weights, "_forward_map")
        _load_source(source_target_forward, weights, "_target_forward_map")
        _load_source(source_backward, weights, "_backward_map")
        _load_source(source_target_backward, weights, "_target_backward_map")
        _load_clean(weights, clean.forward_network, "forward_network", "_forward_map")
        _load_clean(weights, clean.forward_target_network, "forward_target_network", "_target_forward_map")
        _load_clean(weights, clean.backward_network, "backward_network", "_backward_map", backward=True)
        _load_clean(
            weights,
            clean.backward_target_network,
            "backward_network",
            "_target_backward_map",
            backward=True,
        )

    with safe_open(tensor_path, framework="pt", device=str(_DEVICE)) as handle:
        observations = _observations(handle, "input.obs")
        next_observations = _observations(handle, "input.next_obs")
        contexts = handle.get_tensor("latent.training_z")
        actions = handle.get_tensor("input.behavior_action")
        source_value = source_forward(dict(observations.items()), contexts, actions)
        clean_value = clean.forward_map(observations, contexts, actions)
        _assert_output(clean_value, source_value)
        if _DEVICE.type == "cpu":
            _assert_output(clean_value, handle.get_tensor("model.forward.behavior"))
        forward_adjoint = handle.get_tensor("loss.fb.grad_forward_output")
        source_forward_gradients = torch.autograd.grad(
            source_value, tuple(source_forward.parameters()), grad_outputs=forward_adjoint
        )
        clean_forward_gradients = torch.autograd.grad(
            clean_value, tuple(clean.forward_network.parameters()), grad_outputs=forward_adjoint
        )
        _assert_gradient_parity(
            source_forward,
            source_forward_gradients,
            clean.forward_network,
            "forward_network",
            "_forward_map",
            clean_forward_gradients,
        )

        source_backward_value = source_backward(dict(next_observations.items()))
        clean_backward_value = clean.backward_map(next_observations)
        _assert_output(clean_backward_value, source_backward_value)
        backward_adjoint = handle.get_tensor("loss.fb.grad_backward_output")
        source_backward_gradients = torch.autograd.grad(
            source_backward_value,
            tuple(source_backward.parameters()),
            grad_outputs=backward_adjoint,
        )
        clean_backward_gradients = torch.autograd.grad(
            clean_backward_value,
            tuple(clean.backward_network.parameters()),
            grad_outputs=backward_adjoint,
        )
        _assert_gradient_parity(
            source_backward,
            source_backward_gradients,
            clean.backward_network,
            "backward_network",
            "_backward_map",
            clean_backward_gradients,
            backward=True,
        )

        target_actions = handle.get_tensor("model.actor.next_sample")
        source_target_value = source_target_forward(dict(next_observations.items()), contexts, target_actions)
        clean_target_value = clean.forward_map(next_observations, contexts, target_actions, target=True)
        _assert_output(clean_target_value, source_target_value)
        source_target_backward_value = source_target_backward(dict(next_observations.items()))
        clean_target_backward_value = clean.backward_map(next_observations, target=True)
        _assert_output(clean_target_backward_value, source_target_backward_value)

    source_forward_optimizer = torch.optim.Adam(source_forward.parameters(), lr=3e-4)
    clean_forward_optimizer = torch.optim.Adam(clean.forward_network.parameters(), lr=3e-4)
    _set_optimizer_gradients(
        source_forward, source_forward_gradients, clean.forward_network, "forward_network", "_forward_map"
    )
    source_forward_optimizer.step()
    clean_forward_optimizer.step()
    _assert_parameters(source_forward, clean.forward_network, "forward_network", "_forward_map")

    source_backward_optimizer = torch.optim.Adam(source_backward.parameters(), lr=1e-5)
    clean_backward_optimizer = torch.optim.Adam(clean.backward_network.parameters(), lr=1e-5)
    _set_optimizer_gradients(
        source_backward,
        source_backward_gradients,
        clean.backward_network,
        "backward_network",
        "_backward_map",
        backward=True,
    )
    source_backward_optimizer.step()
    clean_backward_optimizer.step()
    _assert_parameters(source_backward, clean.backward_network, "backward_network", "_backward_map", backward=True)

    with torch.no_grad():
        _soft_update_params(tuple(source_forward.parameters()), tuple(source_target_forward.parameters()), 0.01)
        _soft_update_params(tuple(source_backward.parameters()), tuple(source_target_backward.parameters()), 0.01)
        soft_update(tuple(clean.forward_network.parameters()), tuple(clean.forward_target_network.parameters()), 0.01)
        soft_update(tuple(clean.backward_network.parameters()), tuple(clean.backward_target_network.parameters()), 0.01)
    _assert_parameters(
        source_target_forward, clean.forward_target_network, "forward_target_network", "_target_forward_map"
    )
    _assert_parameters(
        source_target_backward,
        clean.backward_target_network,
        "backward_network",
        "_target_backward_map",
        backward=True,
    )


@pytest.mark.skipif(_ORACLE_ROOT is None, reason="BFM_ZERO_ORACLE_DIR is not set")
@pytest.mark.parametrize(
    ("source_prefix", "target_prefix", "output_key", "target_key", "gradient_key"),
    [
        ("_critic", "_target_critic", "model.critic.behavior", "model.target_critic.next", "loss.cpr.grad_q_output"),
        (
            "_aux_critic",
            "_target_aux_critic",
            "model.aux_critic.behavior",
            "model.target_aux_critic.next",
            "loss.aux.grad_q_output",
        ),
    ],
)
def test_bfm_released_value_head_values_gradients_adam_and_ema(
    source_prefix: str,
    target_prefix: str,
    output_key: str,
    target_key: str,
    gradient_key: str,
) -> None:
    """Each released scalar value head should match through one optimizer/EMA mutation."""
    _set_runtime()
    tensor_path, manifest_path, checkpoint = _paths()
    repository = Path(os.getenv("BFM_ZERO_REPO", json.loads(manifest_path.read_text())["source"]["repository"]))
    sys.path.insert(0, str(repository))
    from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelConfig
    from humanoidverse.agents.nn_models import _soft_update_params

    config_data = json.loads((checkpoint / "model/config.json").read_text())
    config_data["device"] = _DEVICE.type
    config = FBcprAuxModelConfig(**config_data)
    init_kwargs = json.loads((checkpoint / "model/init_kwargs.json").read_text())
    observation_space = json_to_space(init_kwargs["obs_space"])
    action_dim = int(init_kwargs["action_dim"])
    source_value = config.archi.critic.build(observation_space, config.archi.z_dim, action_dim, output_dim=1)
    source_target = config.archi.critic.build(observation_space, config.archi.z_dim, action_dim, output_dim=1)
    clean = _make_value_model()
    with safe_open(checkpoint / "model/model.safetensors", framework="pt", device=str(_DEVICE)) as weights:
        _load_source(source_value, weights, source_prefix)
        _load_source(source_target, weights, target_prefix)
        _load_clean(weights, clean.value_networks["value"], "value_networks.value", source_prefix)
        _load_clean(
            weights,
            clean.value_target_networks["value"],
            "value_target_networks.value",
            target_prefix,
        )

    with safe_open(tensor_path, framework="pt", device=str(_DEVICE)) as handle:
        observations = _observations(handle, "input.obs")
        next_observations = _observations(handle, "input.next_obs")
        contexts = handle.get_tensor("latent.training_z")
        actions = handle.get_tensor("input.behavior_action")
        next_actions = handle.get_tensor("model.actor.next_sample")
        source_output = source_value(dict(observations.items()), contexts, actions)
        clean_output = clean.critic_values("value", observations, contexts, actions)
        _assert_output(clean_output, source_output)
        if _DEVICE.type == "cpu":
            _assert_output(clean_output, handle.get_tensor(output_key))
        adjoint = handle.get_tensor(gradient_key)
        source_gradients = torch.autograd.grad(source_output, tuple(source_value.parameters()), grad_outputs=adjoint)
        clean_gradients = torch.autograd.grad(
            clean_output, tuple(clean.value_networks["value"].parameters()), grad_outputs=adjoint
        )
        _assert_gradient_parity(
            source_value,
            source_gradients,
            clean.value_networks["value"],
            "value_networks.value",
            source_prefix,
            clean_gradients,
        )
        source_target_output = source_target(dict(next_observations.items()), contexts, next_actions)
        clean_target_output = clean.critic_values("value", next_observations, contexts, next_actions, target=True)
        _assert_output(clean_target_output, source_target_output)
        if _DEVICE.type == "cpu":
            _assert_output(clean_target_output, handle.get_tensor(target_key))

    source_optimizer = torch.optim.Adam(source_value.parameters(), lr=3e-4)
    clean_optimizer = torch.optim.Adam(clean.value_networks["value"].parameters(), lr=3e-4)
    _set_optimizer_gradients(
        source_value,
        source_gradients,
        clean.value_networks["value"],
        "value_networks.value",
        source_prefix,
    )
    source_optimizer.step()
    clean_optimizer.step()
    _assert_parameters(source_value, clean.value_networks["value"], "value_networks.value", source_prefix)
    with torch.no_grad():
        _soft_update_params(tuple(source_value.parameters()), tuple(source_target.parameters()), 0.005)
        soft_update(
            tuple(clean.value_networks["value"].parameters()),
            tuple(clean.value_target_networks["value"].parameters()),
            0.005,
        )
    _assert_parameters(
        source_target,
        clean.value_target_networks["value"],
        "value_target_networks.value",
        target_prefix,
    )
    del source_value, source_target, clean
    gc.collect()
