# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for forward-backward observation routing."""

from __future__ import annotations

import inspect
import torch
from tensordict import TensorDict

import pytest

from rsl_rl.models.forward_backward_model import ForwardBackwardModel, ForwardBackwardObservationSchema
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
    model = ForwardBackwardModel(observations, BFM_ROUTES)

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
    model = ForwardBackwardModel(observations, META_ROUTES)

    assert model.get_observations(observations, "actor") is observations["state"]


def test_model_surface_stays_concrete_and_small() -> None:
    """Phase 1A should not predict the eventual learner-facing network API."""
    assert not inspect.isabstract(ForwardBackwardModel)
    assert set(ForwardBackwardModel.__dict__).isdisjoint({
        "action_sample",
        "actor_distribution",
        "backward_map",
        "context_project",
        "context_random",
        "critic_values",
        "discriminator_logits",
        "discriminator_reward",
        "forward_map",
    })


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
        ForwardBackwardModel(observations, {"actor": ("state",)})


def test_debug_validator_checks_recorded_shape() -> None:
    """Cold debug validation should catch a malformed runtime batch."""
    schema = _make_schema(META_FIELD_WIDTHS, META_ROUTES)
    observations = TensorDict({"state": torch.randn(2, 358)}, batch_size=[2])

    schema.assert_valid(observations)

    with pytest.raises(ValueError, match="must have shape"):
        schema.assert_valid(TensorDict({"state": torch.randn(2, 357)}, batch_size=[2]))
