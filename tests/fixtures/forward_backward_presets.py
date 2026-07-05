# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen non-public MetaMotivo and BFM-Zero Phase 1F configurations."""

from __future__ import annotations

import copy
from collections.abc import Callable


def metamotivo_config(expert_provider: Callable) -> dict:
    """Return the frozen MetaMotivo HumEnv FB-CPR configuration."""
    dual = {"hidden_dim": 1024, "hidden_layers": 2, "embedding_layers": 2}
    return {
        "seed": 0,
        "num_steps_per_env": 1,
        "num_updates_per_iteration": 1,
        "save_interval": 100_000,
        "obs_groups": {
            "actor": ["policy"],
            "forward": ["policy"],
            "backward": ["policy"],
            "discriminator": ["policy"],
            "critic_discriminator": ["policy"],
        },
        "model": {
            "class_name": "rsl_rl.models.forward_backward_model:ForwardBackwardModel",
            "context_dim": 256,
            "actor_cfg": dual,
            "forward_cfg": dual,
            "backward_hidden_dims": [256],
            "discriminator_hidden_dims": [1024, 1024, 1024],
            "distribution_cfg": {
                "class_name": "ClippedGaussianDistribution",
                "init_std": 0.2,
                "noise_clip": 0.3,
            },
            "initialization_type": "orthogonal",
            "normalization_type": "exponential",
            "normalization_eps": 1e-5,
            "normalization_momentum": 0.01,
        },
        "replay": {
            "class_name": "rsl_rl.storage.forward_backward_replay:ForwardBackwardReplay",
            "policy": {
                "capacity_transitions": 2_000_000,
                "terminal_capacity_per_env": 16,
                "sampling": "transition_uniform",
            },
            "autoreset_mode": "same_step",
        },
        "expert": {
            "provider": expert_provider,
            "clock": {"sampling_mode": "source_rows", "sampling_step_seconds": None},
            "window_lengths": (8,),
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms.forward_backward:ForwardBackward",
            "batch_size": 1024,
            "expert_sequence_length": 8,
            "gamma": 0.98,
            "optimization": {
                "learning_rate": 1e-4,
                "backward_learning_rate": 1e-5,
                "discriminator_learning_rate": 1e-5,
            },
            "context": {
                "goal_fraction": 0.2,
                "expert_fraction": 0.6,
                "relabel_fraction": 0.8,
                "buffer_capacity": 10_000,
                "refresh_steps": 150,
                "rollout_expert_fraction": 0.0,
                "rollout_expert_steps": 250,
                "rollout_expert_context_steps": 8,
            },
            "exploration": {
                "random_action_range": (-1.0, 1.0),
                "random_action_transitions": 50_000,
            },
            "fb_pessimism": 0.0,
            "orthogonality_coefficient": 100.0,
            "implied_value_coefficient": 0.1,
            "discriminator_gradient_penalty_coefficient": 10.0,
        },
        "value_helpers": (
            {
                "name": "discriminator",
                "learning_rate": 1e-4,
                "route": "critic_discriminator",
                "reward_composition": "vector",
                "terms": (
                    {
                        "name": "discriminator",
                        "coefficient": 1.0,
                        "source": "recomputed",
                        "timing": "next_state",
                        "context_dependent": True,
                        "sign": 1,
                    },
                ),
                "pessimism": 0.5,
                "actor_coefficient": 0.01,
                "target_tau": 0.005,
            },
        ),
        "torch_compile_mode": None,
    }


def bfm_zero_native_config(expert_provider: Callable) -> dict:
    """Return the frozen released-effective BFM-Zero configuration."""
    actor = {"hidden_dim": 2048, "hidden_layers": 6, "embedding_layers": 2, "residual": True}
    value = {"hidden_dim": 2048, "hidden_layers": 6, "embedding_layers": 6, "residual": True}
    evidence = (
        "penalty_torques",
        "penalty_action_rate",
        "limits_dof_pos",
        "limits_torque",
        "penalty_undesired_contact",
        "penalty_feet_ori",
        "penalty_ankle_roll",
        "penalty_slippage",
    )
    magnitudes = (0.0, 0.1, 10.0, 0.0, 1.0, 0.4, 4.0, 2.0)
    state = ["joint_position", "joint_velocity", "projected_gravity", "base_angular_velocity"]
    actor_route = [*state, "last_action", "history_actor"]
    forward_route = [*state, "privileged_state", "last_action", "history_actor"]
    backward_route = [*state, "privileged_state"]
    routes = {
        "actor": actor_route,
        "forward": forward_route,
        "backward": backward_route,
        "discriminator": backward_route,
        "critic_discriminator": forward_route,
        "critic_auxiliary": forward_route,
    }
    return {
        "seed": 4728,
        "num_steps_per_env": 1,
        "num_updates_per_iteration": 1,
        "save_interval": 5_000,
        "obs_groups": routes,
        "model": {
            "class_name": "rsl_rl.models.forward_backward_model:ForwardBackwardModel",
            "context_dim": 256,
            "actor_cfg": actor,
            "forward_cfg": value,
            "backward_hidden_dims": [256],
            "discriminator_hidden_dims": [1024, 1024, 1024],
            "distribution_cfg": {
                "class_name": "ClippedGaussianDistribution",
                "init_std": 0.05,
                "noise_clip": 0.3,
            },
            "initialization_type": "orthogonal",
            "normalization_type": "exponential",
            "normalization_eps": 1e-5,
            "normalization_momentum": 0.01,
            "normalization_groups": [{"name": "state", "fields": state}],
        },
        "replay": {
            "class_name": "rsl_rl.storage.forward_backward_replay:ForwardBackwardReplay",
            "policy": {
                "capacity_transitions": 5_120_000,
                "terminal_capacity_per_env": 16,
                "sampling": "transition_uniform",
            },
            "autoreset_mode": "same_step",
            "history_layout": {
                "history_field": "history_actor",
                "history_length": 4,
                "sources": [
                    {"observation_name": "last_action"},
                    {"observation_name": "base_angular_velocity"},
                    {"observation_name": "joint_position"},
                    {"observation_name": "joint_velocity"},
                    {"observation_name": "projected_gravity"},
                ],
            },
        },
        "expert": {
            "provider": expert_provider,
            "clock": {"sampling_mode": "uniform_before_source_end", "sampling_step_seconds": 0.02},
            "window_lengths": (8, 257),
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms.forward_backward:ForwardBackward",
            "batch_size": 1024,
            "expert_sequence_length": 8,
            "gamma": 0.98,
            "optimization": {
                "learning_rate": 3e-4,
                "backward_learning_rate": 1e-5,
                "discriminator_learning_rate": 1e-5,
            },
            "context": {
                "goal_fraction": 0.2,
                "expert_fraction": 0.6,
                "relabel_fraction": 0.8,
                "buffer_capacity": 8_192,
                "refresh_steps": 100,
                "rollout_expert_fraction": 0.5,
                "rollout_expert_steps": 250,
                "rollout_expert_context_steps": 8,
            },
            "exploration": {
                "random_action_range": (-5.0, 5.0),
                "random_action_transitions": 10_240,
            },
            "fb_pessimism": 0.0,
            "orthogonality_coefficient": 100.0,
            "discriminator_gradient_penalty_coefficient": 10.0,
        },
        "value_helpers": (
            {
                "name": "discriminator",
                "learning_rate": 3e-4,
                "route": "critic_discriminator",
                "reward_composition": "vector",
                "terms": (
                    {
                        "name": "discriminator",
                        "coefficient": 1.0,
                        "source": "recomputed",
                        "timing": "next_state",
                        "context_dependent": True,
                        "sign": 1,
                    },
                ),
                "pessimism": 0.5,
                "actor_coefficient": 0.05,
                "target_tau": 0.005,
            },
            {
                "name": "auxiliary",
                "learning_rate": 3e-4,
                "route": "critic_auxiliary",
                "terms": tuple(
                    {
                        "name": name,
                        "coefficient": coefficient,
                        "source": "stored_evidence",
                        "timing": "transition",
                        "context_dependent": False,
                        "sign": -1,
                    }
                    for name, coefficient in zip(evidence, magnitudes, strict=True)
                ),
                "reward_composition": "scalar",
                "pessimism": 0.5,
                "actor_coefficient": 0.02,
                "normalize_rewards": True,
                "reward_normalization_decay": 0.99,
                "reward_normalization_epsilon": 1e-8,
                "target_tau": 0.005,
            },
        ),
        "torch_compile_mode": "reduce-overhead",
    }


def bfm_zero_corrected_terminal_config(expert_provider: Callable) -> dict:
    """Return the measured 32 GiB path with true finals and CUDA graphs disabled."""
    config = copy.deepcopy(bfm_zero_native_config(expert_provider))
    config["torch_compile_mode"] = None
    return config
