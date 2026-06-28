# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation routing shared by forward-backward models."""

from __future__ import annotations

import copy
import math
import torch
import torch.nn.functional as functional
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from tensordict import TensorDict
from typing import Literal, cast

from rsl_rl.modules import MLP, EmpiricalNormalization, ExponentialNormalization, IdentityNormalization
from rsl_rl.modules.distribution import Distribution
from rsl_rl.modules.mlp import MLPBlock
from rsl_rl.modules.reward_channels import ForwardBackwardValueSpec, get_forward_backward_schema_hash
from rsl_rl.utils import resolve_callable

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


@dataclass(frozen=True, slots=True)
class ForwardBackwardDualNetworkCfg:
    """Architecture shared by actor, forward map, and named value heads."""

    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2
    residual: bool = False

    def __post_init__(self) -> None:
        """Reject dimensions that cannot form two equal-width embeddings."""
        if self.hidden_dim < 2 or self.hidden_dim % 2:
            raise ValueError("hidden_dim must be a positive even integer.")
        if self.hidden_layers < 1:
            raise ValueError("hidden_layers must be positive.")
        if self.embedding_layers < 2:
            raise ValueError("embedding_layers must be at least two.")


@dataclass(frozen=True, slots=True)
class ForwardBackwardValueHeadCfg:
    """Named reward-value head and its network architecture."""

    spec: ForwardBackwardValueSpec
    network: ForwardBackwardDualNetworkCfg

    def __post_init__(self) -> None:
        """Keep forward readouts on the main F map rather than a dummy critic."""
        if self.spec.kind != "critic":
            raise ValueError("Only critic value specs create named value-head networks.")


class _ForwardBackwardEmbedding(torch.nn.Module):
    """One half of a state/context or state/action dual encoder."""

    def __init__(
        self,
        input_dim: int,
        cfg: ForwardBackwardDualNetworkCfg,
        ensemble_size: int,
    ) -> None:
        super().__init__()
        if cfg.residual:
            layers: list[torch.nn.Module] = [
                MLPBlock(input_dim, cfg.hidden_dim, ensemble_size=ensemble_size),
            ]
            layers.extend(
                MLPBlock(cfg.hidden_dim, cfg.hidden_dim, ensemble_size=ensemble_size, residual=True)
                for _ in range(cfg.embedding_layers - 2)
            )
            layers.append(MLPBlock(cfg.hidden_dim, cfg.hidden_dim // 2, ensemble_size=ensemble_size))
            self.network = torch.nn.Sequential(*layers)
        else:
            self.network = MLP(
                input_dim,
                cfg.hidden_dim // 2,
                (cfg.hidden_dim,) * (cfg.embedding_layers - 1),
                activation=("tanh",) + ("relu",) * (cfg.embedding_layers - 2),
                last_activation="relu",
                ensemble_size=ensemble_size,
                normalization=("layer_norm",) + (None,) * (cfg.embedding_layers - 2),
            )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Encode one dual-network input."""
        return self.network(value)


class _ForwardBackwardDualNetwork(torch.nn.Module):
    """Two embeddings followed by a shared simple or residual trunk."""

    def __init__(
        self,
        left_input_dim: int,
        right_input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        cfg: ForwardBackwardDualNetworkCfg,
        ensemble_size: int = 1,
    ) -> None:
        super().__init__()
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be positive.")
        self.left_embedding = _ForwardBackwardEmbedding(left_input_dim, cfg, ensemble_size)
        self.right_embedding = _ForwardBackwardEmbedding(right_input_dim, cfg, ensemble_size)

        if cfg.residual:
            output_shape = output_dim if isinstance(output_dim, int) else tuple(output_dim)
            flat_output_dim = output_shape if isinstance(output_shape, int) else math.prod(output_shape)
            layers: list[torch.nn.Module] = [
                MLPBlock(cfg.hidden_dim, cfg.hidden_dim, ensemble_size=ensemble_size, residual=True)
                for _ in range(cfg.hidden_layers)
            ]
            layers.append(MLPBlock(cfg.hidden_dim, flat_output_dim, activation=None, ensemble_size=ensemble_size))
            if not isinstance(output_shape, int):
                layers.append(torch.nn.Unflatten(dim=-1, unflattened_size=output_shape))
            self.trunk = torch.nn.Sequential(*layers)
        else:
            self.trunk = MLP(
                cfg.hidden_dim,
                output_dim,
                (cfg.hidden_dim,) * cfg.hidden_layers,
                activation="relu",
                ensemble_size=ensemble_size,
            )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Fuse two encoded inputs and evaluate the trunk."""
        return self.trunk(torch.cat((self.left_embedding(left), self.right_embedding(right)), dim=-1))


class _ForwardBackwardBackwardNetwork(torch.nn.Module):
    """Backward representation with optional sphere projection."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        normalize_output: bool,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.normalize_output = normalize_output
        self.network = MLP(
            input_dim,
            context_dim,
            hidden_dims,
            activation=("tanh",) + ("relu",) * (len(hidden_dims) - 1),
            normalization=("layer_norm",) + (None,) * (len(hidden_dims) - 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Map observations into the context space."""
        context = self.network(observations)
        if self.normalize_output:
            context = math.sqrt(self.context_dim) * functional.normalize(context, dim=-1)
        return context


class _ForwardBackwardDiscriminatorNetwork(torch.nn.Module):
    """Context-conditioned state discriminator returning logits."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
    ) -> None:
        super().__init__()
        self.network = MLP(
            input_dim + context_dim,
            1,
            hidden_dims,
            activation=("tanh",) + ("relu",) * (len(hidden_dims) - 1),
            normalization=("layer_norm",) + (None,) * (len(hidden_dims) - 1),
        )

    def forward(self, observations: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Return one discriminator logit per row."""
        return self.network(torch.cat((context, observations), dim=-1))


class ForwardBackwardModel(torch.nn.Module):
    """Composite model shared by MetaMotivo and BFM-Zero configurations.

    Observation fields are normalized once and reused by every named route.
    Network modules remain public attributes so the learner can assign explicit
    optimizers and freeze boundaries without private-member reach-through.
    """

    is_recurrent: bool = False

    @classmethod
    def from_config(
        cls,
        observations: TensorDict,
        obs_groups: Mapping[str, Sequence[str]],
        action_dim: int,
        config: Mapping[str, object],
    ) -> ForwardBackwardModel:
        """Build a training or inference model from one ordinary config section."""
        options = dict(config)
        configured_class = resolve_callable(options.pop("class_name", cls))
        if configured_class is not cls:
            raise TypeError(f"Configured model class must be {cls.__name__}, got {configured_class!r}.")
        actor_cfg = ForwardBackwardDualNetworkCfg(**dict(options.pop("actor_cfg")))
        forward_cfg = ForwardBackwardDualNetworkCfg(**dict(options.pop("forward_cfg")))
        value_heads = []
        for value in options.pop("value_heads", ()):
            value_options = dict(value)
            spec = ForwardBackwardValueSpec(**dict(value_options.pop("spec")))
            network = ForwardBackwardDualNetworkCfg(**dict(value_options.pop("network")))
            if value_options:
                raise ValueError(f"Unknown value-head configuration: {tuple(value_options)}.")
            value_heads.append(ForwardBackwardValueHeadCfg(spec, network))
        return cls(
            observations,
            obs_groups,
            action_dim,
            actor_cfg=actor_cfg,
            forward_cfg=forward_cfg,
            value_heads=tuple(value_heads),
            **options,
        )

    def __init__(
        self,
        observations: TensorDict,
        obs_groups: Mapping[str, Sequence[str]],
        action_dim: int,
        context_dim: int,
        actor_cfg: ForwardBackwardDualNetworkCfg,
        forward_cfg: ForwardBackwardDualNetworkCfg,
        backward_hidden_dims: tuple[int, ...] | list[int],
        *,
        forward_ensemble_size: int = 2,
        backward_normalization: bool = True,
        distribution_cfg: Mapping[str, object] | None = None,
        discriminator_hidden_dims: tuple[int, ...] | list[int] | None = None,
        value_heads: Sequence[ForwardBackwardValueHeadCfg] = (),
        normalization_type: Literal["none", "empirical", "exponential"] = "empirical",
        normalization_eps: float = 1e-2,
        normalization_momentum: float = 0.1,
        normalization_until: int | None = None,
        context_normalization: bool = True,
    ) -> None:
        """Build the composite model from explicit component configurations."""
        super().__init__()
        if action_dim < 1 or context_dim < 1 or forward_ensemble_size < 1:
            raise ValueError("action_dim, context_dim, and forward_ensemble_size must be positive.")
        self.observation_schema = ForwardBackwardObservationSchema.from_observations(observations, obs_groups)
        self.observation_schema.assert_valid(observations)
        for route in ("actor", "forward", "backward"):
            self.observation_schema.route(cast(ForwardBackwardRouteName, route))

        self.action_dim = action_dim
        self.context_dim = context_dim
        self.context_normalization = context_normalization
        self.value_specs = tuple(head.spec for head in value_heads)

        if normalization_type == "none":
            normalizers = {name: IdentityNormalization() for name, _width in self.observation_schema.field_widths}
        elif normalization_type == "empirical":
            normalizers = {
                name: EmpiricalNormalization(width, eps=normalization_eps, until=normalization_until)
                for name, width in self.observation_schema.field_widths
            }
        elif normalization_type == "exponential":
            normalizers = {
                name: ExponentialNormalization(width, eps=normalization_eps, momentum=normalization_momentum)
                for name, width in self.observation_schema.field_widths
            }
        else:
            raise ValueError(f"Unknown normalization_type: {normalization_type!r}.")
        self.observation_normalizers = torch.nn.ModuleDict(normalizers)

        options = dict(
            distribution_cfg
            or {
                "class_name": "ClippedGaussianDistribution",
                "init_std": 0.2,
            }
        )
        try:
            distribution_class = resolve_callable(cast(str, options.pop("class_name")))
        except KeyError as error:
            raise ValueError("distribution_cfg requires class_name.") from error
        self.action_distribution = distribution_class(action_dim, **options)
        if not isinstance(self.action_distribution, Distribution):
            raise TypeError("The action distribution must derive from Distribution.")

        actor_output_dim = self.action_distribution.input_dim
        self.actor_network = _ForwardBackwardDualNetwork(
            self.observation_schema.route_width("actor"),
            self.observation_schema.route_width("actor") + context_dim,
            actor_output_dim,
            actor_cfg,
        )
        self.forward_network = _ForwardBackwardDualNetwork(
            self.observation_schema.route_width("forward") + action_dim,
            self.observation_schema.route_width("forward") + context_dim,
            context_dim,
            forward_cfg,
            forward_ensemble_size,
        )
        self.backward_network = _ForwardBackwardBackwardNetwork(
            self.observation_schema.route_width("backward"),
            context_dim,
            backward_hidden_dims,
            backward_normalization,
        )

        self.forward_target_network = self._make_target(self.forward_network)
        self.backward_target_network = self._make_target(self.backward_network)

        if discriminator_hidden_dims is None:
            self.discriminator_network: torch.nn.Module | None = None
        else:
            self.observation_schema.route("discriminator")
            self.discriminator_network = _ForwardBackwardDiscriminatorNetwork(
                self.observation_schema.route_width("discriminator"),
                context_dim,
                discriminator_hidden_dims,
            )

        names = tuple(head.spec.name for head in value_heads)
        if len(names) != len(set(names)):
            raise ValueError("Value-head names must be unique.")
        self.value_networks = torch.nn.ModuleDict()
        self.value_target_networks = torch.nn.ModuleDict()
        self._value_specs_by_name = {head.spec.name: head.spec for head in value_heads}
        for head in value_heads:
            route = cast(ForwardBackwardRouteName, head.spec.route)
            self.observation_schema.route(route)
            network = _ForwardBackwardDualNetwork(
                self.observation_schema.route_width(route) + action_dim,
                self.observation_schema.route_width(route) + context_dim,
                len(head.spec.reward_channels),
                head.network,
                head.spec.ensemble_size,
            )
            self.value_networks[head.spec.name] = network
            if head.spec.has_target:
                self.value_target_networks[head.spec.name] = self._make_target(network)

    @staticmethod
    def _make_target(module: torch.nn.Module) -> torch.nn.Module:
        target = copy.deepcopy(module)
        target.requires_grad_(False)
        target.eval()
        return target

    def get_observations(self, observations: TensorDict, name: ForwardBackwardRouteName) -> torch.Tensor:
        """Select and concatenate one raw observation route."""
        return self.observation_schema.get_observations(observations, name)

    def get_normalized_observations(
        self,
        observations: TensorDict,
        name: ForwardBackwardRouteName,
    ) -> torch.Tensor:
        """Normalize shared fields and concatenate one route."""
        fields = self.observation_schema.route(name)
        values = [self.observation_normalizers[field](observations[field]) for field in fields]
        return values[0] if len(values) == 1 else torch.cat(values, dim=-1)

    @torch.no_grad()
    def update_normalization(self, observations: TensorDict) -> None:
        """Update each field normalizer exactly once from raw observations."""
        for field_name, _width in self.observation_schema.field_widths:
            self.observation_normalizers[field_name].update(observations[field_name])

    def normalization_train(self, mode: bool = True) -> None:
        """Enable or freeze field-normalizer updates."""
        self.observation_normalizers.train(mode)

    def context_project(self, context: torch.Tensor) -> torch.Tensor:
        """Project context vectors onto the configured sphere."""
        if not self.context_normalization:
            return context
        return math.sqrt(self.context_dim) * functional.normalize(context, dim=-1)

    def context_random(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample random projected context vectors."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        parameter = next(self.parameters())
        context = torch.randn(
            batch_size,
            self.context_dim,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
            generator=generator,
        )
        return self.context_project(context)

    def actor_distribution(
        self,
        observations: TensorDict,
        context: torch.Tensor,
    ) -> Distribution:
        """Update and return the actor action distribution."""
        actor_observations = self.get_normalized_observations(observations, "actor")
        output = self.actor_network(
            actor_observations,
            torch.cat((actor_observations, context), dim=-1),
        )
        self.action_distribution.update(output)
        return self.action_distribution

    def action_sample(
        self,
        observations: TensorDict,
        context: torch.Tensor,
        *,
        deterministic: bool = False,
        pathwise: bool = False,
    ) -> torch.Tensor:
        """Return deterministic, ordinary stochastic, or pathwise actions."""
        distribution = self.actor_distribution(observations, context)
        if deterministic:
            return distribution.mean
        if pathwise:
            return distribution.rsample()
        return distribution.sample()

    def forward_map(
        self,
        observations: TensorDict,
        context: torch.Tensor,
        actions: torch.Tensor,
        *,
        target: bool = False,
    ) -> torch.Tensor:
        """Evaluate the live or target forward representation ensemble."""
        route = self.get_normalized_observations(observations, "forward")
        network = self.forward_target_network if target else self.forward_network
        return network(torch.cat((route, actions), dim=-1), torch.cat((route, context), dim=-1))

    def backward_map(
        self,
        observations: TensorDict,
        *,
        target: bool = False,
    ) -> torch.Tensor:
        """Evaluate the live or target backward representation."""
        route = self.get_normalized_observations(observations, "backward")
        network = self.backward_target_network if target else self.backward_network
        return network(route)

    def discriminator_logits(
        self,
        observations: TensorDict,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate discriminator logits for one state/context batch."""
        if self.discriminator_network is None:
            raise RuntimeError("No discriminator is configured.")
        route = self.get_normalized_observations(observations, "discriminator")
        return self.discriminator_network(route, context)

    def critic_values(
        self,
        name: str,
        observations: TensorDict,
        context: torch.Tensor,
        actions: torch.Tensor,
        *,
        target: bool = False,
    ) -> torch.Tensor:
        """Evaluate one named live or target reward-value ensemble."""
        try:
            spec = self._value_specs_by_name[name]
        except KeyError as error:
            raise KeyError(f"Unknown value head {name!r}.") from error
        if target:
            try:
                network = self.value_target_networks[name]
            except KeyError as error:
                raise RuntimeError(f"Value head {name!r} has no target network.") from error
        else:
            network = self.value_networks[name]
        route_name = cast(ForwardBackwardRouteName, spec.route)
        route = self.get_normalized_observations(observations, route_name)
        return network(torch.cat((route, actions), dim=-1), torch.cat((route, context), dim=-1))
