# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Released BFM-Zero actor, discriminator, Adam, and normalizer parity."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import torch
from collections.abc import Mapping
from pathlib import Path
from tensordict import TensorDict
from typing import Protocol

import pytest
from safetensors import safe_open

from rsl_rl.models.forward_backward_model import ForwardBackwardDualNetworkCfg, ForwardBackwardModel
from rsl_rl.modules.forward_backward import discriminator_gradient_penalty, discriminator_logistic_loss

_ORACLE_ROOT = os.getenv("BFM_ZERO_ORACLE_DIR")
_DEVICE = torch.device(os.getenv("FORWARD_BACKWARD_ORACLE_DEVICE", "cpu"))


class _TensorReader(Protocol):
    def get_tensor(self, name: str) -> torch.Tensor: ...


def _set_reference_runtime() -> None:
    torch.use_deterministic_algorithms(True)
    if _DEVICE.type == "cpu":
        torch.set_num_threads(1)
        torch.backends.mkldnn.enabled = False


def _paths() -> tuple[Path, Path, Path]:
    oracle_root = Path(_ORACLE_ROOT).expanduser()  # type: ignore[arg-type]
    manifest_path = oracle_root / "oracle.json"
    manifest = json.loads(manifest_path.read_text())
    checkpoint = Path(os.getenv("BFM_ZERO_CHECKPOINT_DIR", manifest["source"]["checkpoint"]))
    return oracle_root / "oracle_tensors.safetensors", manifest_path, checkpoint


def _make_model(*, normalization: bool = False) -> ForwardBackwardModel:
    observations = TensorDict(
        {
            "state": torch.zeros(2, 64),
            "privileged_state": torch.zeros(2, 463),
            "last_action": torch.zeros(2, 29),
            "history_actor": torch.zeros(2, 372),
        },
        batch_size=[2],
    )
    return ForwardBackwardModel(
        observations,
        {
            "actor": ("state", "last_action", "history_actor"),
            "forward": ("state",),
            "backward": ("state", "privileged_state"),
            "discriminator": ("state", "privileged_state"),
        },
        action_dim=29,
        context_dim=256,
        actor_cfg=ForwardBackwardDualNetworkCfg(2048, 6, 2, True),
        forward_cfg=ForwardBackwardDualNetworkCfg(16, 1, 2),
        backward_hidden_dims=(16,),
        discriminator_hidden_dims=(1024, 1024, 1024),
        normalization_type="exponential" if normalization else "none",
        normalization_eps=1e-5,
        normalization_momentum=0.01,
        distribution_cfg={"class_name": "ClippedGaussianDistribution", "init_std": 0.05},
    ).to(_DEVICE)


def _source_name(clean_name: str) -> str:
    actor_prefixes = {
        "actor_network.left_embedding.network.": "embed_s.",
        "actor_network.right_embedding.network.": "embed_z.",
        "actor_network.trunk.": "policy.",
    }
    for clean_prefix, source_prefix in actor_prefixes.items():
        if clean_name.startswith(clean_prefix):
            suffix = clean_name.removeprefix(clean_prefix)
            suffix = suffix.replace(".normalization.", ".mlp.0.")
            suffix = suffix.replace(".linear.", ".mlp.1.")
            return source_prefix + suffix
    if clean_name.startswith("discriminator_network.network."):
        return "trunk." + clean_name.removeprefix("discriminator_network.network.")
    raise KeyError(clean_name)


def _load_prefix(module: torch.nn.Module, checkpoint: Path, prefix: str) -> None:
    state = {}
    marker = f"{prefix}."
    with safe_open(checkpoint / "model/model.safetensors", framework="pt", device=str(_DEVICE)) as weights:
        for key in weights.keys():  # noqa: SIM118
            if key.startswith(marker):
                state[key.removeprefix(marker)] = weights.get_tensor(key)
    module.load_state_dict(state, strict=True, assign=True)
    module.to(_DEVICE).eval()
    module.requires_grad_(True)


def _load_clean_component(model: ForwardBackwardModel, checkpoint: Path, clean_prefix: str) -> None:
    module = model.actor_network if clean_prefix == "actor_network" else model.discriminator_network
    marker = "_actor." if clean_prefix == "actor_network" else "_discriminator."
    with (
        safe_open(checkpoint / "model/model.safetensors", framework="pt", device=str(_DEVICE)) as weights,
        torch.no_grad(),
    ):
        for clean_name, parameter in module.named_parameters():
            source_name = marker + _source_name(f"{clean_prefix}.{clean_name}")
            parameter.copy_(weights.get_tensor(source_name))


def _observations(handle: _TensorReader, prefix: str) -> TensorDict:
    fields = {
        name: handle.get_tensor(f"{prefix}.{name}")
        for name in ("state", "privileged_state", "last_action", "history_actor")
    }
    return TensorDict(fields, batch_size=[fields["state"].shape[0]])


def _gradient_hash(named_gradients: list[tuple[str, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for name, gradient in sorted(named_gradients):
        digest.update(name.encode())
        digest.update(gradient.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _gradient_hash_provenance() -> dict[str, object]:
    """Return the execution identity required for byte-exact gradient hashes."""
    return {
        "policy": "named-autograd-grad-v1",
        "node": platform.node(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_config_sha256": hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _gradient_hash_is_comparable(runtime: Mapping[str, object]) -> bool:
    """Return whether an oracle hash was produced by this exact execution identity."""
    return runtime.get("gradient_hash_provenance") == _gradient_hash_provenance()


def _assert_gradient_parity(
    source: list[tuple[str, torch.Tensor]],
    clean: list[tuple[str, torch.Tensor]],
    summary: dict[str, float | int | str],
    runtime: Mapping[str, object],
) -> None:
    source_by_name = dict(source)
    clean_by_name = dict(clean)
    assert source_by_name.keys() == clean_by_name.keys()
    squared_norm = 0.0
    max_abs = 0.0
    count = 0
    squared_error = 0.0
    dot = 0.0
    clean_squared = 0.0
    for name in source_by_name:
        reference = source_by_name[name]
        actual = clean_by_name[name]
        squared_norm += float(reference.double().square().sum())
        max_abs = max(max_abs, float(reference.abs().max()))
        count += reference.numel()
        squared_error += float((actual.double() - reference.double()).square().sum())
        dot += float((actual.double() * reference.double()).sum())
        clean_squared += float(actual.double().square().sum())
    assert count == summary["numel"]
    assert squared_norm**0.5 == pytest.approx(summary["l2"], rel=5e-5, abs=5e-6)
    assert max_abs == pytest.approx(summary["max_abs"], rel=5e-5, abs=5e-6)
    assert dot / (clean_squared * squared_norm) ** 0.5 >= 0.99999
    assert (squared_error / squared_norm) ** 0.5 <= 1e-4
    if _DEVICE.type == "cpu" and _gradient_hash_is_comparable(runtime):
        assert _gradient_hash(source) == summary["sha256"]
        assert _gradient_hash(clean) == summary["sha256"]


def test_frozen_gradient_hash_requires_matching_runtime_provenance() -> None:
    """A Torch version alone or a different host must not enable byte-exact checks."""
    gradients = [("weight", torch.ones(1))]
    summary: dict[str, float | int | str] = {
        "l2": 1.0,
        "max_abs": 1.0,
        "numel": 1,
        "sha256": "deliberately-not-the-gradient-hash",
    }
    provenance = _gradient_hash_provenance()

    _assert_gradient_parity(gradients, gradients, summary, {"torch": torch.__version__})
    _assert_gradient_parity(
        gradients,
        gradients,
        summary,
        {"gradient_hash_provenance": provenance | {"node": f"{provenance['node']}-different"}},
    )
    with pytest.raises(AssertionError):
        _assert_gradient_parity(gradients, gradients, summary, {"gradient_hash_provenance": provenance})


def _clean_named_gradients(
    module: torch.nn.Module, clean_prefix: str, gradients: tuple[torch.Tensor, ...]
) -> list[tuple[str, torch.Tensor]]:
    return [
        (_source_name(f"{clean_prefix}.{name}"), gradient)
        for (name, _parameter), gradient in zip(module.named_parameters(), gradients, strict=True)
    ]


def _set_mapped_gradients(
    source_gradients: list[tuple[str, torch.Tensor]], clean: torch.nn.Module, clean_prefix: str
) -> None:
    source_by_name = dict(source_gradients)
    for clean_name, parameter in clean.named_parameters():
        parameter.grad = source_by_name[_source_name(f"{clean_prefix}.{clean_name}")].clone()


def _assert_parameters_close(source: torch.nn.Module, clean: torch.nn.Module, clean_prefix: str) -> None:
    source_parameters = dict(source.named_parameters())
    for clean_name, clean_parameter in clean.named_parameters():
        source_parameter = source_parameters[_source_name(f"{clean_prefix}.{clean_name}")]
        torch.testing.assert_close(clean_parameter, source_parameter, rtol=2.0e-5, atol=2.0e-7)


@pytest.mark.skipif(_ORACLE_ROOT is None, reason="BFM_ZERO_ORACLE_DIR is not set")
def test_bfm_actor_discriminator_gradients_and_fresh_adam_match_reference() -> None:
    """Released actor/D values, full gradient hashes, and fresh Adam steps should match."""
    _set_reference_runtime()
    tensor_path, manifest_path, checkpoint = _paths()
    manifest = json.loads(manifest_path.read_text())
    repository = Path(os.getenv("BFM_ZERO_REPO", manifest["source"]["repository"]))
    sys.path.insert(0, str(repository))
    from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
    from humanoidverse.agents.fb_cpr_aux.model import FBcprAuxModelConfig

    config_data = json.loads((checkpoint / "model/config.json").read_text())
    config_data["device"] = _DEVICE.type
    config = FBcprAuxModelConfig(**config_data)
    init_kwargs = json.loads((checkpoint / "model/init_kwargs.json").read_text())
    observation_space = json_to_space(init_kwargs["obs_space"])
    action_dim = int(init_kwargs["action_dim"])
    source_actor = config.archi.actor.build(observation_space, config.archi.z_dim, action_dim)
    source_discriminator = config.archi.discriminator.build(observation_space, config.archi.z_dim)
    _load_prefix(source_actor, checkpoint, "_actor")
    _load_prefix(source_discriminator, checkpoint, "_discriminator")

    clean = _make_model()
    _load_clean_component(clean, checkpoint, "actor_network")
    _load_clean_component(clean, checkpoint, "discriminator_network")

    with safe_open(tensor_path, framework="pt", device=str(_DEVICE)) as handle:
        observations = _observations(handle, "input.obs")
        expert_observations = _observations(handle, "input.expert_obs")
        contexts = handle.get_tensor("latent.training_z")
        behavior_contexts = handle.get_tensor("input.behavior_z")
        expert_contexts = handle.get_tensor("latent.expert_z")

        source_distribution = source_actor(dict(observations.items()), contexts, 0.05)
        source_mean = source_distribution.mean
        source_noisy_action = source_mean + handle.get_tensor("model.actor.sample_noise")
        source_action = (
            source_noisy_action
            - source_noisy_action.detach()
            + source_noisy_action.detach().clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        )
        clean_raw = clean.actor_network(
            clean.get_observations(observations, "actor"),
            torch.cat((clean.get_observations(observations, "actor"), contexts), dim=-1),
        )
        clean_mean = clean.action_distribution.deterministic_output(clean_raw)
        noisy_action = clean_mean + handle.get_tensor("model.actor.sample_noise")
        clean_action = noisy_action - noisy_action.detach() + noisy_action.detach().clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        torch.testing.assert_close(source_mean, handle.get_tensor("model.actor.mean"), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(clean_mean, source_mean, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(source_action, handle.get_tensor("model.actor.sample"))
        torch.testing.assert_close(clean_action, source_action, rtol=2e-5, atol=2e-6)

        action_gradient = handle.get_tensor("actor.action_grad_total")
        source_actor_gradients = torch.autograd.grad(
            source_action, tuple(source_actor.parameters()), grad_outputs=action_gradient
        )
        clean_actor_gradients = torch.autograd.grad(
            clean_action, tuple(clean.actor_network.parameters()), grad_outputs=action_gradient
        )
        source_actor_named = [
            (name, gradient)
            for (name, _parameter), gradient in zip(
                source_actor.named_parameters(), source_actor_gradients, strict=True
            )
        ]
        clean_actor_named = _clean_named_gradients(clean.actor_network, "actor_network", clean_actor_gradients)
        _assert_gradient_parity(
            source_actor_named,
            clean_actor_named,
            manifest["gradient_summaries"]["actor"],
            manifest["runtime"],
        )

        source_expert_logits = source_discriminator.compute_logits(dict(expert_observations.items()), expert_contexts)
        source_train_logits = source_discriminator.compute_logits(dict(observations.items()), behavior_contexts)
        clean_expert_logits = clean.discriminator_logits(expert_observations, expert_contexts)
        clean_train_logits = clean.discriminator_logits(observations, behavior_contexts)
        torch.testing.assert_close(
            clean_expert_logits, handle.get_tensor("model.discriminator.expert_logits"), rtol=2e-5, atol=2e-6
        )
        torch.testing.assert_close(
            clean_train_logits, handle.get_tensor("model.discriminator.train_logits"), rtol=2e-5, atol=2e-6
        )

        alpha = handle.get_tensor("loss.discriminator.alpha")
        source_interpolated = {
            key: (alpha * expert_observations[key] + (1.0 - alpha) * observations[key]).requires_grad_(True)
            for key in observations.keys()  # noqa: SIM118
        }
        source_context = (alpha * expert_contexts + (1.0 - alpha) * behavior_contexts).requires_grad_(True)
        source_interpolated_logits = source_discriminator.compute_logits(source_interpolated, source_context)
        source_used_inputs = [source_interpolated["state"], source_interpolated["privileged_state"], source_context]
        source_input_gradients = torch.autograd.grad(
            source_interpolated_logits,
            source_used_inputs,
            grad_outputs=torch.ones_like(source_interpolated_logits),
            create_graph=True,
            retain_graph=True,
        )
        source_gp = (torch.cat(source_input_gradients, dim=1).norm(2, dim=1) - 1.0).square().mean()
        source_loss = discriminator_logistic_loss(source_expert_logits, source_train_logits) + 10.0 * source_gp
        source_discriminator_gradients = torch.autograd.grad(source_loss, tuple(source_discriminator.parameters()))

        clean_state = (
            alpha * clean.get_observations(expert_observations, "discriminator")
            + (1.0 - alpha) * clean.get_observations(observations, "discriminator")
        ).requires_grad_(True)
        clean_context = (alpha * expert_contexts + (1.0 - alpha) * behavior_contexts).requires_grad_(True)
        clean_interpolated_logits = clean.discriminator_network(clean_state, clean_context)
        clean_gp = discriminator_gradient_penalty(clean_interpolated_logits, (clean_state, clean_context))
        clean_loss = discriminator_logistic_loss(clean_expert_logits, clean_train_logits) + 10.0 * clean_gp
        clean_discriminator_gradients = torch.autograd.grad(clean_loss, tuple(clean.discriminator_network.parameters()))
        torch.testing.assert_close(clean_gp, handle.get_tensor("loss.discriminator.gradient_penalty").squeeze())
        torch.testing.assert_close(clean_loss, handle.get_tensor("loss.discriminator.total").squeeze())
        discriminator_summary = manifest["gradient_summaries"]["discriminator_with_gp"]
        source_discriminator_named = [
            (name, gradient)
            for (name, _parameter), gradient in zip(
                source_discriminator.named_parameters(), source_discriminator_gradients, strict=True
            )
        ]
        clean_discriminator_named = _clean_named_gradients(
            clean.discriminator_network, "discriminator_network", clean_discriminator_gradients
        )
        _assert_gradient_parity(
            source_discriminator_named,
            clean_discriminator_named,
            discriminator_summary,
            manifest["runtime"],
        )

        source_actor_optimizer = torch.optim.Adam(source_actor.parameters(), lr=3e-4)
        clean_actor_optimizer = torch.optim.Adam(clean.actor_network.parameters(), lr=3e-4)
        for parameter, gradient in zip(source_actor.parameters(), source_actor_gradients, strict=True):
            parameter.grad = gradient
        _set_mapped_gradients(source_actor_named, clean.actor_network, "actor_network")
        source_actor_optimizer.step()
        clean_actor_optimizer.step()
        _assert_parameters_close(source_actor, clean.actor_network, "actor_network")

        source_discriminator_optimizer = torch.optim.Adam(source_discriminator.parameters(), lr=1e-5)
        clean_discriminator_optimizer = torch.optim.Adam(clean.discriminator_network.parameters(), lr=1e-5)
        for parameter, gradient in zip(source_discriminator.parameters(), source_discriminator_gradients, strict=True):
            parameter.grad = gradient
        _set_mapped_gradients(source_discriminator_named, clean.discriminator_network, "discriminator_network")
        source_discriminator_optimizer.step()
        clean_discriminator_optimizer.step()
        _assert_parameters_close(source_discriminator, clean.discriminator_network, "discriminator_network")


@pytest.mark.skipif(_ORACLE_ROOT is None, reason="BFM_ZERO_ORACLE_DIR is not set")
def test_bfm_field_normalizers_match_checkpoint_values_and_ordered_mutation() -> None:
    """All four asymmetric fields should match released values and post-update statistics."""
    _set_reference_runtime()
    tensor_path, _manifest_path, checkpoint = _paths()
    model = _make_model(normalization=True)
    fields = ("state", "privileged_state", "last_action", "history_actor")
    with safe_open(checkpoint / "model/model.safetensors", framework="pt", device=str(_DEVICE)) as weights:
        for field in fields:
            prefix = f"_obs_normalizer._normalizers.{field}._normalizer"
            normalizer = model.observation_normalizers[field]
            normalizer.running_mean.copy_(weights.get_tensor(f"{prefix}.running_mean"))
            normalizer.running_var.copy_(weights.get_tensor(f"{prefix}.running_var"))
            normalizer.num_batches_tracked.copy_(weights.get_tensor(f"{prefix}.num_batches_tracked"))

    with safe_open(tensor_path, framework="pt", device=str(_DEVICE)) as handle:
        raw = _observations(handle, "input.raw_obs")
        raw_next = _observations(handle, "input.raw_next_obs")
        model.update_normalization(raw)
        model.update_normalization(raw_next)
        model.normalization_train(False)
        for field in fields:
            normalizer = model.observation_normalizers[field]
            torch.testing.assert_close(normalizer.running_mean, handle.get_tensor(f"normalizer.updated.{field}.mean"))
            torch.testing.assert_close(normalizer.running_var, handle.get_tensor(f"normalizer.updated.{field}.var"))
            torch.testing.assert_close(normalizer(raw[field]), handle.get_tensor(f"input.obs.{field}"))
