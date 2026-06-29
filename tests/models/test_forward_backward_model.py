# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for forward-backward observation routing."""

from __future__ import annotations

import copy
import inspect
import torch
from tensordict import TensorDict
from typing import Literal

import pytest

from rsl_rl.models.forward_backward_model import (
    ForwardBackwardDualNetworkCfg,
    ForwardBackwardModel,
    ForwardBackwardObservationSchema,
    ForwardBackwardValueHeadCfg,
)
from rsl_rl.modules.mlp import MLPEnsembleLinear
from rsl_rl.modules.reward_channels import ForwardBackwardValueSpec
from tests.fixtures.forward_backward import BFM_FIELD_WIDTHS, BFM_ROUTES, META_FIELD_WIDTHS, META_ROUTES


def _make_schema(field_widths: dict[str, int], routes: dict[str, tuple[str, ...]]) -> ForwardBackwardObservationSchema:
    return ForwardBackwardObservationSchema.from_config(field_widths, routes)


def _make_bfm_observations(batch_size: int = 2) -> TensorDict:
    return TensorDict(
        {
            name: torch.full((batch_size, width), float(index + 1))
            for index, (name, width) in enumerate(BFM_FIELD_WIDTHS.items())
        },
        batch_size=[batch_size],
    )


def _dual(
    hidden_dim: int = 16,
    hidden_layers: int = 2,
    embedding_layers: int = 2,
    residual: bool = False,
) -> ForwardBackwardDualNetworkCfg:
    return ForwardBackwardDualNetworkCfg(hidden_dim, hidden_layers, embedding_layers, residual)


def _hidden_dims(hidden_dim: int = 8, hidden_layers: int = 2) -> tuple[int, ...]:
    return (hidden_dim,) * hidden_layers


def _make_model(
    observations: TensorDict,
    routes: dict[str, tuple[str, ...]],
    *,
    actor_cfg: ForwardBackwardDualNetworkCfg | None = None,
    forward_cfg: ForwardBackwardDualNetworkCfg | None = None,
    backward_hidden_dims: tuple[int, ...] | None = None,
    forward_ensemble_size: int = 2,
    discriminator_hidden_dims: tuple[int, ...] | None = None,
    value_heads: tuple[ForwardBackwardValueHeadCfg, ...] = (),
    normalization_type: Literal["none", "empirical", "exponential"] = "none",
    normalization_eps: float = 1e-2,
    normalization_momentum: float = 0.1,
    distribution_cfg: dict[str, object] | None = None,
    initialization_type: Literal["default", "orthogonal"] = "default",
) -> ForwardBackwardModel:
    return ForwardBackwardModel(
        observations,
        routes,
        action_dim=2,
        context_dim=4,
        actor_cfg=actor_cfg or _dual(),
        forward_cfg=forward_cfg or _dual(),
        backward_hidden_dims=backward_hidden_dims or _hidden_dims(),
        forward_ensemble_size=forward_ensemble_size,
        discriminator_hidden_dims=discriminator_hidden_dims,
        value_heads=value_heads,
        normalization_type=normalization_type,
        normalization_eps=normalization_eps,
        normalization_momentum=normalization_momentum,
        distribution_cfg=distribution_cfg,
        initialization_type=initialization_type,
    )


def _value_head(
    name: str,
    route: str,
    *,
    reward_channels: tuple[str, ...] = ("reward",),
    reward_composition: Literal["vector", "scalar"] = "vector",
    ensemble_size: int = 2,
    has_target: bool = True,
    network: ForwardBackwardDualNetworkCfg | None = None,
) -> ForwardBackwardValueHeadCfg:
    return ForwardBackwardValueHeadCfg(
        spec=ForwardBackwardValueSpec(
            name=name,
            kind="critic",
            route=route,
            reward_channels=reward_channels,
            reward_composition=reward_composition,
            ensemble_size=ensemble_size,
            has_target=has_target,
        ),
        network=network or _dual(),
    )


def test_meta_routes_have_exact_width_and_order() -> None:
    """MetaMotivo should route its one 358-wide state field everywhere."""
    schema = _make_schema(META_FIELD_WIDTHS, META_ROUTES)

    assert schema.routes == (
        ("actor", ("state",)),
        ("forward", ("state",)),
        ("backward", ("state",)),
        ("discriminator", ("state",)),
        ("critic_discriminator", ("state",)),
    )
    assert tuple(schema.route_width(name) for name, _ in schema.routes) == (358, 358, 358, 358, 358)


def test_bfm_routes_have_exact_width_and_checkpoint_order() -> None:
    """BFM-Zero should preserve every asymmetric concatenation order."""
    schema = _make_schema(BFM_FIELD_WIDTHS, BFM_ROUTES)

    assert schema.field_widths == (
        ("history_actor", 372),
        ("last_action", 29),
        ("privileged_state", 463),
        ("state", 64),
    )
    assert schema.routes == (
        ("actor", ("state", "last_action", "history_actor")),
        ("forward", ("state", "privileged_state", "last_action", "history_actor")),
        ("backward", ("state", "privileged_state")),
        ("discriminator", ("state", "privileged_state")),
        ("critic_discriminator", ("state", "privileged_state", "last_action", "history_actor")),
        ("critic_auxiliary", ("state", "privileged_state", "last_action", "history_actor")),
    )
    assert tuple(schema.route_width(name) for name, _ in schema.routes) == (465, 928, 527, 527, 928, 928)


def test_route_field_order_changes_checkpoint_identity() -> None:
    """Swapping equal-total-width fields should change the schema hash."""
    reordered = dict(BFM_ROUTES)
    reordered["forward"] = ("privileged_state", "state", "last_action", "history_actor")

    schema = _make_schema(BFM_FIELD_WIDTHS, reordered)

    assert schema.route_width("forward") == 928
    assert schema.schema_hash != _make_schema(BFM_FIELD_WIDTHS, BFM_ROUTES).schema_hash


def test_mapping_insertion_order_does_not_change_checkpoint_identity() -> None:
    """Only route concatenation order, not dictionary order, should affect identity."""
    reversed_fields = dict(reversed(tuple(BFM_FIELD_WIDTHS.items())))
    reversed_routes = dict(reversed(tuple(BFM_ROUTES.items())))

    assert (
        _make_schema(reversed_fields, reversed_routes).schema_hash
        == _make_schema(BFM_FIELD_WIDTHS, BFM_ROUTES).schema_hash
    )


def test_direct_schema_construction_copies_mutable_inputs() -> None:
    """A direct constructor should own the mappings and nested route lists it hashes."""
    field_widths = {"state": 358, "diagnostic": 7}
    routes = {"actor": ["state"]}
    schema = ForwardBackwardObservationSchema(
        field_widths=field_widths,  # type: ignore[arg-type]
        routes=routes,  # type: ignore[arg-type]
    )
    expected = _make_schema(field_widths, routes)

    field_widths["state"] = 1
    routes["actor"].append("diagnostic")

    assert schema.field_widths == (("state", 358),)
    assert schema.routes == (("actor", ("state",)),)
    assert schema.schema_hash == expected.schema_hash


def test_unrouted_environment_field_is_not_owned_by_model_schema() -> None:
    """Environment-only observations should not enter the model checkpoint identity."""
    fields = {**META_FIELD_WIDTHS, "diagnostic": 7}

    schema = _make_schema(fields, META_ROUTES)

    assert schema.field_widths == (("state", 358),)
    assert schema.schema_hash == _make_schema(META_FIELD_WIDTHS, META_ROUTES).schema_hash


def test_schema_derives_widths_without_redundant_expected_values() -> None:
    """Route widths should have one source of truth."""
    fields = {**BFM_FIELD_WIDTHS, "privileged_state": 462}

    schema = _make_schema(fields, BFM_ROUTES)

    assert schema.route_width("forward") == 927
    assert "expected_route_widths" not in inspect.signature(ForwardBackwardObservationSchema.from_config).parameters


def test_model_uses_tensordict_route_order_without_runtime_shape_checks() -> None:
    """Runtime tensors should be concatenated directly in configured order."""
    observations = _make_bfm_observations()
    model = _make_model(observations, BFM_ROUTES)

    actual = model.get_observations(observations, "forward")
    expected = torch.cat(
        [
            observations["state"],
            observations["privileged_state"],
            observations["last_action"],
            observations["history_actor"],
        ],
        dim=-1,
    )

    assert torch.equal(actual, expected)
    assert actual.shape == (2, 928)
    assert not hasattr(model.observation_schema, "validate_observations")


def test_single_field_route_returns_tensordict_tensor_directly() -> None:
    """A one-field route should not allocate a redundant concatenation."""
    observations = TensorDict({"state": torch.randn(3, 358)}, batch_size=[3])
    model = _make_model(observations, META_ROUTES)

    assert model.get_observations(observations, "actor") is observations["state"]


def test_model_surface_exposes_phase_1c_component_methods() -> None:
    """The learner should use public component methods instead of private reach-through."""
    assert not inspect.isabstract(ForwardBackwardModel)
    assert {
        "action_sample",
        "actor_distribution",
        "backward_map",
        "context_project",
        "context_random",
        "critic_values",
        "discriminator_logits",
        "forward_map",
        "get_normalized_observations",
        "update_normalization",
    }.issubset(ForwardBackwardModel.__dict__)


def test_unknown_route_and_field_fail_at_construction_boundary() -> None:
    """Configuration typos should fail once when the model is built."""
    with pytest.raises(ValueError, match="Unknown observation routes"):
        _make_schema(META_FIELD_WIDTHS, {"typo": ("state",)})
    with pytest.raises(ValueError, match="unknown fields"):
        _make_schema(META_FIELD_WIDTHS, {"actor": ("missing",)})


def test_nonflat_construction_observation_fails_once() -> None:
    """Flat feedforward inputs should be checked at construction, not every batch."""
    observations = TensorDict({"state": torch.randn(2, 3, 4)}, batch_size=[2])

    with pytest.raises(ValueError, match="only support flat observations"):
        _make_model(observations, {"actor": ("state",)})


def test_random_context_accepts_a_learner_owned_generator() -> None:
    """Context sampling should resume independently from the global Torch RNG."""
    observations = TensorDict({"state": torch.randn(3, 358)}, batch_size=[3])
    model = _make_model(observations, META_ROUTES)
    first = torch.Generator().manual_seed(17)
    second = torch.Generator().manual_seed(17)

    torch.testing.assert_close(model.context_random(5, generator=first), model.context_random(5, generator=second))


def test_debug_validator_checks_recorded_shape() -> None:
    """Cold debug validation should catch a malformed runtime batch."""
    schema = _make_schema(META_FIELD_WIDTHS, META_ROUTES)
    observations = TensorDict({"state": torch.randn(2, 358)}, batch_size=[2])

    schema.assert_valid(observations)

    with pytest.raises(ValueError, match="must have shape"):
        schema.assert_valid(TensorDict({"state": torch.randn(2, 357)}, batch_size=[2]))


def test_composite_model_outputs_match_named_meta_routes() -> None:
    """Meta-style components should expose the expected ensemble and batch axes."""
    observations = TensorDict({"state": torch.randn(3, 358)}, batch_size=[3])
    critic = _value_head("discriminator", "critic_discriminator")
    model = _make_model(
        observations,
        META_ROUTES,
        discriminator_hidden_dims=_hidden_dims(),
        value_heads=(critic,),
    )
    context = model.context_random(3)
    actions = model.action_sample(observations, context, deterministic=True)

    assert actions.shape == (3, 2)
    assert model.forward_map(observations, context, actions).shape == (2, 3, 4)
    assert model.backward_map(observations).shape == (3, 4)
    assert model.discriminator_logits(observations, context).shape == (3, 1)
    assert model.critic_values("discriminator", observations, context, actions).shape == (2, 3, 1)


def test_orthogonal_initialization_precedes_exact_target_copies() -> None:
    """From-scratch FB models should expose one initialization law for every live owner."""
    observations = TensorDict({"state": torch.randn(3, 12)}, batch_size=[3])
    routes = {name: ("state",) for name in META_ROUTES}
    value_head = _value_head("discriminator", "critic_discriminator")
    model = _make_model(
        observations,
        routes,
        discriminator_hidden_dims=_hidden_dims(),
        value_heads=(value_head,),
        initialization_type="orthogonal",
    )

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            weights = (module.weight,)
            gain = 1.0
        elif isinstance(module, MLPEnsembleLinear):
            weights = tuple(module.weight)
            gain = torch.nn.init.calculate_gain("relu")
        else:
            continue
        if module.bias is not None:
            torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))
        for weight in weights:
            rows, columns = weight.shape
            gram = weight @ weight.mT if rows <= columns else weight.mT @ weight
            expected = gain**2 * torch.eye(min(rows, columns), dtype=weight.dtype)
            torch.testing.assert_close(gram, expected, rtol=1e-5, atol=1e-5)

    for live, target in (
        (model.forward_network, model.forward_target_network),
        (model.backward_network, model.backward_target_network),
        (model.value_networks["discriminator"], model.value_target_networks["discriminator"]),
    ):
        for live_value, target_value in zip(live.state_dict().values(), target.state_dict().values(), strict=True):
            torch.testing.assert_close(live_value, target_value, rtol=0.0, atol=0.0)


def test_scaled_bfm_residual_model_uses_every_asymmetric_route() -> None:
    """The BFM topology should support residual networks and a distinct auxiliary head."""
    observations = _make_bfm_observations(batch_size=3)
    residual = _dual(hidden_dim=16, hidden_layers=3, embedding_layers=2, residual=True)
    discriminator_head = _value_head("discriminator", "critic_discriminator", network=residual)
    auxiliary_head = _value_head(
        "auxiliary",
        "critic_auxiliary",
        reward_channels=("tracking", "regularization"),
        network=_dual(hidden_dim=24, hidden_layers=2, embedding_layers=3, residual=True),
    )
    model = _make_model(
        observations,
        BFM_ROUTES,
        actor_cfg=residual,
        forward_cfg=residual,
        discriminator_hidden_dims=_hidden_dims(hidden_dim=12, hidden_layers=3),
        value_heads=(discriminator_head, auxiliary_head),
    )
    context = model.context_random(3)
    actions = model.action_sample(observations, context, pathwise=True)

    assert model.forward_map(observations, context, actions).shape == (2, 3, 4)
    assert model.critic_values("discriminator", observations, context, actions).shape == (2, 3, 1)
    assert model.critic_values("auxiliary", observations, context, actions).shape == (2, 3, 2)

    discriminator_parameters = sum(
        parameter.numel() for parameter in model.value_networks["discriminator"].parameters()
    )
    auxiliary_parameters = sum(parameter.numel() for parameter in model.value_networks["auxiliary"].parameters())
    assert discriminator_parameters != auxiliary_parameters


def test_field_normalizers_update_once_freeze_and_round_trip() -> None:
    """Shared fields should own one restorable statistic regardless of route reuse."""
    observations = _make_bfm_observations(batch_size=5)
    model = _make_model(observations, BFM_ROUTES, normalization_type="empirical")
    model.update_normalization(observations)

    for name in BFM_FIELD_WIDTHS:
        assert model.observation_normalizers[name].count.item() == 5

    model.normalization_train(False)
    model.update_normalization(_make_bfm_observations(batch_size=7))
    for name in BFM_FIELD_WIDTHS:
        assert model.observation_normalizers[name].count.item() == 5

    restored = _make_model(observations, BFM_ROUTES, normalization_type="empirical")
    restored.load_state_dict(copy.deepcopy(model.state_dict()))

    probe = _make_bfm_observations(batch_size=3)
    context = model.context_random(3)
    torch.testing.assert_close(
        model.action_sample(probe, context, deterministic=True),
        restored.action_sample(probe, context, deterministic=True),
    )


def test_exponential_field_normalizer_matches_released_two_batch_update() -> None:
    """Meta/BFM mode should reproduce two ordered BatchNorm statistic updates."""
    observations = _make_bfm_observations(batch_size=5)
    next_observations = TensorDict(
        {name: value + torch.arange(5, dtype=value.dtype).unsqueeze(-1) for name, value in observations.items()},
        batch_size=[5],
    )
    model = _make_model(
        observations,
        BFM_ROUTES,
        normalization_type="exponential",
        normalization_eps=1e-5,
        normalization_momentum=0.01,
    )
    model.update_normalization(observations)
    model.update_normalization(next_observations)
    model.normalization_train(False)

    for name in BFM_FIELD_WIDTHS:
        normalizer = model.observation_normalizers[name]
        expected_mean = 0.99 * (0.01 * observations[name].mean(dim=0)) + 0.01 * next_observations[name].mean(dim=0)
        expected_var = 0.99 * (0.99 * torch.ones_like(expected_mean) + 0.01 * observations[name].var(dim=0))
        expected_var += 0.01 * next_observations[name].var(dim=0)
        torch.testing.assert_close(normalizer.running_mean, expected_mean)
        torch.testing.assert_close(normalizer.running_var, expected_var)
        assert normalizer.num_batches_tracked.item() == 2
        expected = (next_observations[name] - expected_mean) / torch.sqrt(expected_var + 1e-5)
        torch.testing.assert_close(normalizer(next_observations[name]), expected)


def test_exponential_normalizer_forward_does_not_update_statistics() -> None:
    """Reading normalized observations should not advance temporal state."""
    observations = _make_bfm_observations(batch_size=5)
    model = _make_model(
        observations,
        BFM_ROUTES,
        normalization_type="exponential",
        normalization_eps=1e-5,
        normalization_momentum=0.01,
    )
    model.update_normalization(observations)
    batches_before = {
        name: normalizer.num_batches_tracked.clone() for name, normalizer in model.observation_normalizers.items()
    }

    model.get_normalized_observations(observations, "forward")

    for name, normalizer in model.observation_normalizers.items():
        assert torch.equal(normalizer.num_batches_tracked, batches_before[name])


def test_normalized_bfm_route_matches_independent_field_composition() -> None:
    """Field-level normalizers should compose in declared route order bit-for-bit."""
    observations = _make_bfm_observations(batch_size=5)
    model = _make_model(observations, BFM_ROUTES, normalization_type="empirical")
    model.update_normalization(observations)

    actual = model.get_normalized_observations(observations, "forward")
    expected = torch.cat(
        [model.observation_normalizers[field](observations[field]) for field in BFM_ROUTES["forward"]],
        dim=-1,
    )

    assert torch.equal(actual, expected)


def test_optional_components_vanish_without_dummy_modules() -> None:
    """The Meta FB subset should not allocate discriminator or critic placeholders."""
    observations = TensorDict({"state": torch.randn(3, 358)}, batch_size=[3])
    model = _make_model(observations, META_ROUTES)

    assert model.discriminator_network is None
    assert len(model.value_networks) == 0
    assert len(model.value_target_networks) == 0
    assert not hasattr(model, "actor_target_network")
    with pytest.raises(RuntimeError, match="No discriminator"):
        model.discriminator_logits(observations, model.context_random(3))


def test_targets_exist_only_for_mathematically_targeted_components() -> None:
    """F, B, and opted-in critics should own frozen target copies."""
    observations = TensorDict({"state": torch.randn(3, 358)}, batch_size=[3])
    with_target = _value_head("with_target", "critic_discriminator")
    without_target = _value_head("without_target", "critic_discriminator", has_target=False)
    model = _make_model(observations, META_ROUTES, value_heads=(with_target, without_target))

    assert not any(parameter.requires_grad for parameter in model.forward_target_network.parameters())
    assert not any(parameter.requires_grad for parameter in model.backward_target_network.parameters())
    assert not any(parameter.requires_grad for parameter in model.value_target_networks["with_target"].parameters())
    assert "without_target" not in model.value_target_networks
    assert sum(parameter.numel() for parameter in model.forward_network.parameters()) == sum(
        parameter.numel() for parameter in model.forward_target_network.parameters()
    )


def test_pathwise_actor_gradient_crosses_frozen_value_network() -> None:
    """Frozen evaluators should preserve action gradients without parameter gradients."""
    observations = TensorDict({"state": torch.randn(6, 358)}, batch_size=[6])
    critic = _value_head("critic", "critic_discriminator")
    model = _make_model(observations, META_ROUTES, value_heads=(critic,))
    model.forward_network.requires_grad_(False)
    model.value_networks["critic"].requires_grad_(False)
    context = model.context_random(6).detach()

    actions = model.action_sample(observations, context, pathwise=True)
    forward_value = model.forward_map(observations, context, actions).sum(dim=-1)
    critic_value = model.critic_values("critic", observations, context, actions)
    loss = -forward_value.mean() - critic_value.mean()
    loss.backward()

    actor_gradients = [parameter.grad for parameter in model.actor_network.parameters()]
    assert any(gradient is not None and torch.count_nonzero(gradient).item() > 0 for gradient in actor_gradients)
    assert all(parameter.grad is None for parameter in model.forward_network.parameters())
    assert all(parameter.grad is None for parameter in model.value_networks["critic"].parameters())


def test_clipped_actor_config_is_owned_and_explicitly_not_ppo_compatible() -> None:
    """The direct-Q default should be bounded and must reject an inexact PPO density."""
    observations = TensorDict({"state": torch.randn(32, 358)}, batch_size=[32])
    distribution_cfg: dict[str, object] = {
        "class_name": "ClippedGaussianDistribution",
        "init_std": 5.0,
    }
    expected = distribution_cfg.copy()
    model = _make_model(observations, META_ROUTES, distribution_cfg=distribution_cfg)
    context = model.context_random(32)
    actions = model.action_sample(observations, context, pathwise=True)

    assert distribution_cfg == expected
    assert torch.all(actions > -1.0)
    assert torch.all(actions < 1.0)
    with pytest.raises(NotImplementedError, match="exact bounded density"):
        model.action_distribution.log_prob(actions)


def test_context_projection_uses_sqrt_dimension_radius() -> None:
    """Context projection should match the FB sphere convention exactly."""
    observations = TensorDict({"state": torch.randn(3, 358)}, batch_size=[3])
    model = _make_model(observations, META_ROUTES)
    context = model.context_project(torch.randn(7, 4))

    torch.testing.assert_close(context.norm(dim=-1), torch.full((7,), 2.0))


def test_reward_context_inference_uses_shared_integration_and_projection() -> None:
    """Model reward inference should add only configured context geometry."""
    observations = TensorDict({"state": torch.randn(3, 358)}, batch_size=[3])
    model = _make_model(observations, META_ROUTES)
    backward = torch.randn(7, 4)
    rewards = torch.randn(7, 2)
    weights = torch.softmax(10.0 * rewards, dim=0)

    actual = model.context_infer_reward(backward, rewards, weights)
    expected = model.context_project((rewards * weights).mT @ backward)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.norm(dim=-1), torch.full((2,), 2.0))


def test_model_from_config_is_shared_by_training_and_inference() -> None:
    """The ordinary model config should construct the same inference topology without a learner."""
    observations = TensorDict({"state": torch.randn(3, 358)}, batch_size=[3])
    config = {
        "class_name": "rsl_rl.models.forward_backward_model:ForwardBackwardModel",
        "context_dim": 4,
        "actor_cfg": {"hidden_dim": 16, "hidden_layers": 2, "embedding_layers": 2},
        "forward_cfg": {"hidden_dim": 16, "hidden_layers": 2, "embedding_layers": 2},
        "backward_hidden_dims": [8, 8],
        "normalization_type": "exponential",
        "normalization_eps": 1e-5,
        "normalization_momentum": 0.01,
    }

    model = ForwardBackwardModel.from_config(observations, META_ROUTES, 2, config)

    assert model.observation_schema.route_width("actor") == 358
    assert model.action_dim == 2
    assert model.context_dim == 4
    assert model.observation_normalizers["state"].momentum == 0.01


def _simple_dual_parameter_count(
    left_input: int,
    right_input: int,
    output_dim: int,
    hidden_dim: int,
    hidden_layers: int,
    embedding_layers: int,
    ensemble_size: int,
) -> int:
    def embedding(input_dim: int) -> int:
        count = input_dim * hidden_dim + hidden_dim + 2 * hidden_dim
        count += (embedding_layers - 2) * (hidden_dim * hidden_dim + hidden_dim)
        count += hidden_dim * (hidden_dim // 2) + hidden_dim // 2
        return count

    trunk = hidden_layers * (hidden_dim * hidden_dim + hidden_dim)
    trunk += hidden_dim * output_dim + output_dim
    return ensemble_size * (embedding(left_input) + embedding(right_input) + trunk)


def test_simple_forward_parameter_count_matches_independent_formula() -> None:
    """Ensemble parameter ownership should match the Meta-style architecture exactly."""
    observations = TensorDict({"state": torch.randn(2, 358)}, batch_size=[2])
    cfg = _dual(hidden_dim=16, hidden_layers=2, embedding_layers=3)
    model = _make_model(observations, META_ROUTES, forward_cfg=cfg, forward_ensemble_size=3)

    expected = _simple_dual_parameter_count(360, 362, 4, 16, 2, 3, 3)
    actual = sum(parameter.numel() for parameter in model.forward_network.parameters())

    assert actual == expected


def test_scalar_composed_value_head_has_one_output() -> None:
    """A scalar helper should compose several reward channels into one propagated value."""
    observations = _make_bfm_observations(batch_size=3)
    head = _value_head(
        "auxiliary",
        "critic_auxiliary",
        reward_channels=("action_rate", "slippage"),
        reward_composition="scalar",
    )
    model = _make_model(observations, BFM_ROUTES, value_heads=(head,))
    context = model.context_random(3)
    actions = torch.zeros(3, 2)

    assert model.critic_values("auxiliary", observations, context, actions).shape == (2, 3, 1)
