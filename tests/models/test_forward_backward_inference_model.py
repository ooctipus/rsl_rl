# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the minimal forward-backward deterministic inference view."""

from __future__ import annotations

import copy
import torch
from tensordict import TensorDict

import pytest

from rsl_rl.models.forward_backward_model import (
    ForwardBackwardDualNetworkCfg,
    ForwardBackwardInferenceModel,
    ForwardBackwardModel,
)

_ROUTES = {
    "actor": ("state", "history"),
    "forward": ("state",),
    "backward": ("state",),
    "discriminator": ("state",),
}


def _observations(seed: int, batch_size: int = 5) -> TensorDict:
    generator = torch.Generator().manual_seed(seed)
    return TensorDict(
        {
            "state": torch.randn(batch_size, 7, generator=generator),
            "history": torch.randn(batch_size, 3, generator=generator),
        },
        batch_size=[batch_size],
    )


def _model(seed: int) -> ForwardBackwardModel:
    torch.manual_seed(seed)
    observations = _observations(seed)
    model = ForwardBackwardModel(
        observations,
        _ROUTES,
        action_dim=2,
        context_dim=4,
        actor_cfg=ForwardBackwardDualNetworkCfg(hidden_dim=16, hidden_layers=2, embedding_layers=2),
        forward_cfg=ForwardBackwardDualNetworkCfg(hidden_dim=16, hidden_layers=2, embedding_layers=2),
        backward_hidden_dims=(8, 8),
        discriminator_hidden_dims=(8, 8),
        normalization_type="empirical",
        distribution_cfg={
            "class_name": "ClippedGaussianDistribution",
            "init_std": 0.2,
            "action_range": (-0.7, 0.9),
        },
    )
    model.update_normalization(_observations(seed + 100, batch_size=11))
    model.eval()
    return model


def test_inference_view_matches_independent_models_under_vmap() -> None:
    """Stacked inference should numerically match each independently loaded policy."""
    models = [_model(seed) for seed in (11, 22, 33)]
    inference_models = [model.as_inference_model() for model in models]
    observations = TensorDict.stack([_observations(seed) for seed in (44, 55, 66)])

    expected_backward = torch.stack([model.backward_map(observations[index]) for index, model in enumerate(models)])
    expected_context = torch.stack([
        model.context_project(expected_backward[index]) for index, model in enumerate(models)
    ])
    expected_action = torch.stack([
        model.action_sample(observations[index], expected_context[index], deterministic=True)
        for index, model in enumerate(models)
    ])

    parameters, buffers = torch.func.stack_module_state(inference_models)
    base = copy.deepcopy(inference_models[0]).to("meta")

    def backward_one(
        model_parameters: dict[str, torch.Tensor],
        model_buffers: dict[str, torch.Tensor],
        model_observations: TensorDict,
    ) -> torch.Tensor:
        return torch.func.functional_call(
            base,
            (model_parameters, model_buffers),
            (model_observations,),
            {"output": "backward"},
        )

    actual_backward = torch.vmap(backward_one)(parameters, buffers, observations)
    contexts = inference_models[0].context_project(actual_backward)

    def action_one(
        model_parameters: dict[str, torch.Tensor],
        model_buffers: dict[str, torch.Tensor],
        model_observations: TensorDict,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return torch.func.functional_call(
            base,
            (model_parameters, model_buffers),
            (model_observations, context),
            {"output": "action"},
        )

    actual_action = torch.vmap(action_one)(parameters, buffers, observations, contexts)

    torch.testing.assert_close(actual_backward, expected_backward, rtol=1.0e-5, atol=2.0e-7)
    torch.testing.assert_close(actual_action, expected_action, rtol=1.0e-5, atol=2.0e-7)
    assert actual_backward.shape == (3, 5, 4)
    assert actual_action.shape == (3, 5, 2)


def test_inference_view_contains_only_deterministic_policy_state() -> None:
    """The inference state should omit training-only and stochastic modules."""
    model = _model(7)
    inference = model.as_inference_model()
    inference_state = inference.state_dict()
    selected_modules = [
        *(
            (f"observation_normalizers.{field}", model.observation_normalizers[field])
            for field in set(_ROUTES["actor"] + _ROUTES["backward"])
        ),
        ("actor_network", model.actor_network),
        ("backward_network", model.backward_network),
        ("deterministic_output", inference.deterministic_output),
    ]
    expected_state = {
        f"{prefix}.{name}": value for prefix, module in selected_modules for name, value in module.state_dict().items()
    }
    inference_bytes = sum(value.numel() * value.element_size() for value in inference_state.values())
    expected_bytes = sum(value.numel() * value.element_size() for value in expected_state.values())
    full_model_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values())

    assert isinstance(inference, ForwardBackwardInferenceModel)
    assert inference.observation_schema is model.observation_schema
    assert inference.action_dim == model.action_dim
    assert inference.context_dim == model.context_dim
    assert inference.context_normalization is model.context_normalization
    assert set(inference_state) == set(expected_state)
    assert inference_bytes == expected_bytes
    assert inference_bytes < full_model_bytes
    assert all(
        not key.startswith(("forward_network.", "forward_target_network.", "discriminator_network."))
        for key in inference_state
    )
    assert all("std_param" not in key and "value_network" not in key for key in inference_state)


def test_inference_view_reuses_model_modules_without_copying_tensors() -> None:
    """Creating the inference boundary should not duplicate learned tensors."""
    model = _model(5)
    inference = model.as_inference_model()

    assert inference.actor_network is model.actor_network
    assert inference.backward_network is model.backward_network
    for field in set(_ROUTES["actor"] + _ROUTES["backward"]):
        assert inference.observation_normalizers[field] is model.observation_normalizers[field]


def test_inference_forward_rejects_ambiguous_requests() -> None:
    """The functional entry point should reject missing context and unknown outputs."""
    inference = _model(9).as_inference_model()
    observations = _observations(10)

    with pytest.raises(ValueError, match="requires context"):
        inference(observations)
    with pytest.raises(ValueError, match="Unknown inference output"):
        inference(observations, output="unknown")
