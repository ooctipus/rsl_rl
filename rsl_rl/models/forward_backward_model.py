# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation routing shared by forward-backward models."""

from __future__ import annotations

import torch
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from tensordict import TensorDict
from typing import Literal

from rsl_rl.modules.reward_channels import get_forward_backward_schema_hash

ForwardBackwardRouteName = Literal[
    "actor",
    "forward",
    "backward",
    "discriminator",
    "critic_discriminator",
    "critic_auxiliary",
]

_ROUTE_ORDER: tuple[ForwardBackwardRouteName, ...] = (
    "actor",
    "forward",
    "backward",
    "discriminator",
    "critic_discriminator",
    "critic_auxiliary",
)


@dataclass(frozen=True)
class ForwardBackwardObservationSchema:
    """Ordered observation groups and their input widths.

    Field widths are recorded once when the model is constructed. Runtime
    batches keep ordinary :class:`TensorDict` semantics and are not rechecked
    by this object.
    """

    field_widths: tuple[tuple[str, int], ...]
    routes: tuple[tuple[ForwardBackwardRouteName, tuple[str, ...]], ...]
    schema_hash: str = field(init=False)

    def __post_init__(self) -> None:
        """Copy, normalize, and identify direct or factory construction data."""
        field_items = self.field_widths.items() if isinstance(self.field_widths, Mapping) else self.field_widths
        route_items = self.routes.items() if isinstance(self.routes, Mapping) else self.routes
        raw_fields = tuple((name, width) for name, width in field_items)
        raw_routes = tuple((name, tuple(fields)) for name, fields in route_items)

        field_names = tuple(name for name, _width in raw_fields)
        if len(field_names) != len(set(field_names)):
            raise ValueError("Observation field names must be unique.")
        route_names = tuple(name for name, _fields in raw_routes)
        if len(route_names) != len(set(route_names)):
            raise ValueError("Observation route names must be unique.")
        unknown_routes = set(route_names).difference(_ROUTE_ORDER)
        if unknown_routes:
            raise ValueError(f"Unknown observation routes: {tuple(sorted(unknown_routes))}.")

        route_by_name = dict(raw_routes)
        routes = tuple((name, route_by_name[name]) for name in _ROUTE_ORDER if name in route_by_name)
        if not routes:
            raise ValueError("At least one observation route is required.")
        for name, route_fields in routes:
            if not route_fields:
                raise ValueError(f"Observation route {name!r} must not be empty.")

        used_fields = {field_name for _route_name, route_fields in routes for field_name in route_fields}
        field_by_name = dict(raw_fields)
        missing_fields = used_fields.difference(field_by_name)
        if missing_fields:
            raise ValueError(f"Observation routes use unknown fields: {tuple(sorted(missing_fields))}.")
        fields = tuple(sorted((name, field_by_name[name]) for name in used_fields))
        for name, width in fields:
            if not isinstance(width, int) or isinstance(width, bool) or width < 1:
                raise ValueError(f"Observation field {name!r} must have a positive integer width.")

        object.__setattr__(self, "field_widths", fields)
        object.__setattr__(self, "routes", routes)
        object.__setattr__(
            self,
            "schema_hash",
            get_forward_backward_schema_hash({"field_widths": fields, "routes": routes}),
        )

    @classmethod
    def from_config(
        cls,
        field_widths: Mapping[str, int],
        obs_groups: Mapping[str, Sequence[str]],
    ) -> ForwardBackwardObservationSchema:
        """Create a schema from named observation groups.

        Args:
            field_widths: Width of each observation used by a route.
            obs_groups: Observation names for each model route, in concatenation order.

        Returns:
            Canonically ordered route schema.
        """
        fields = tuple(field_widths.items())
        routes = tuple((name, tuple(route_fields)) for name, route_fields in obs_groups.items())
        return cls(field_widths=fields, routes=routes)  # type: ignore[arg-type]

    @classmethod
    def from_observations(
        cls,
        observations: TensorDict,
        obs_groups: Mapping[str, Sequence[str]],
    ) -> ForwardBackwardObservationSchema:
        """Infer route widths from the model's construction observations.

        Args:
            observations: Initial environment observations.
            obs_groups: Observation names for each model route, in concatenation order.

        Returns:
            Canonically ordered route schema.
        """
        field_names = {name for fields in obs_groups.values() for name in fields}
        field_widths: dict[str, int] = {}
        for name in field_names:
            value = observations[name]
            if value.ndim != len(observations.batch_size) + 1:
                raise ValueError(
                    "Forward-backward models only support flat observations, "
                    f"got shape {tuple(value.shape)} for {name!r}."
                )
            field_widths[name] = value.shape[-1]
        return cls.from_config(field_widths, obs_groups)

    def assert_valid(self, observations: TensorDict) -> None:
        """Check a batch at a construction or debug boundary.

        This method is intentionally excluded from model and replay hot paths.

        Args:
            observations: Flat observations to check against the recorded widths.
        """
        batch_size = tuple(observations.batch_size)
        if len(batch_size) != 1:
            raise ValueError(f"Observations must have one batch dimension, got {batch_size}.")
        device: torch.device | None = None
        for name, width in self.field_widths:
            value = observations[name]
            expected_shape = (*batch_size, width)
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"Observation {name!r} must have shape {expected_shape}, got {tuple(value.shape)}.")
            if device is None:
                device = value.device
            elif value.device != device:
                raise ValueError(f"Observation {name!r} is on {value.device}, expected {device}.")

    def route(self, name: ForwardBackwardRouteName) -> tuple[str, ...]:
        """Return the ordered observations for a route."""
        for route_name, fields in self.routes:
            if route_name == name:
                return fields
        raise KeyError(f"Observation route {name!r} is not configured.")

    def route_width(self, name: ForwardBackwardRouteName) -> int:
        """Return the concatenated width of a route."""
        widths = dict(self.field_widths)
        return sum(widths[field_name] for field_name in self.route(name))

    def get_observations(self, observations: TensorDict, name: ForwardBackwardRouteName) -> torch.Tensor:
        """Select and concatenate one route from a runtime batch."""
        values = [observations[field_name] for field_name in self.route(name)]
        if len(values) == 1:
            return values[0]
        return torch.cat(values, dim=-1)


class ForwardBackwardModel(torch.nn.Module):
    """Construction-time observation routing for a forward-backward model.

    Concrete network modules are added by the model implementation. This base
    only follows the same observation-group convention as other RSL-RL models.
    """

    is_recurrent: bool = False

    def __init__(self, observations: TensorDict, obs_groups: Mapping[str, Sequence[str]]) -> None:
        """Initialize observation routes from the first environment batch.

        Args:
            observations: Initial environment observations.
            obs_groups: Observation names for each model route, in concatenation order.
        """
        super().__init__()
        self.observation_schema = ForwardBackwardObservationSchema.from_observations(observations, obs_groups)
        self.observation_schema.assert_valid(observations)

    def get_observations(self, observations: TensorDict, name: ForwardBackwardRouteName) -> torch.Tensor:
        """Select and concatenate observations for one model route."""
        return self.observation_schema.get_observations(observations, name)
