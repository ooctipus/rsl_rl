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
            "distribution_cfg": {"class_name": "ClippedGaussianDistribution", "init_std": 0.2},
            "normalization_eps": 1e-5,
            "normalization_momentum": 0.01,
            "value_heads": [
                {
                    "spec": {
                        "name": "discriminator",
                        "kind": "critic",
                        "route": "critic_discriminator",
                        "reward_channels": ["discriminator"],
                        "ensemble_size": 2,
                        "has_target": True,
                    },
                    "network": dual,
                }
            ],
        },
        "replay": {
            "class_name": "rsl_rl.storage.forward_backward_replay:ForwardBackwardReplay",
            "capacity_steps": 2_000_000,
            "terminal_capacity_per_env": 16,
            "autoreset_mode": "same_step",
            "environment_reward_name": "environment",
            "auxiliary_evidence_names": [],
            "reward_channels": [
                {
                    "name": "environment",
                    "provider_name": "environment",
                    "source": "environment",
                    "timing": "transition",
                    "context_dependent": False,
                    "sign": 1,
                },
                {
                    "name": "discriminator",
                    "provider_name": "discriminator",
                    "source": "recomputed",
                    "timing": "next_state",
                    "context_dependent": True,
                    "sign": 1,
                },
            ],
        },
        "expert": {"provider": expert_provider, "window_lengths": (8,)},
        "algorithm": {
            "class_name": "rsl_rl.algorithms.forward_backward:ForwardBackward",
            "batch_size": 1024,
            "expert_sequence_length": 8,
            "gamma": 0.98,
            "learning_rate": 1e-4,
            "backward_learning_rate": 1e-5,
            "discriminator_learning_rate": 1e-5,
            "orthogonality_coefficient": 100.0,
            "implied_value_coefficient": 0.1,
            "discriminator_gradient_penalty_coefficient": 10.0,
            "context_goal_fraction": 0.2,
            "context_expert_fraction": 0.6,
            "relabel_fraction": 0.8,
            "rollout_context_refresh_steps": 150,
            "value_cfg": {"discriminator": {"learning_rate": 1e-4, "actor_coefficient": 0.01}},
        },
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
    routes = {
        "actor": ["state", "last_action", "history_actor"],
        "forward": ["state", "privileged_state", "last_action", "history_actor"],
        "backward": ["state", "privileged_state"],
        "discriminator": ["state", "privileged_state"],
        "critic_discriminator": ["state", "privileged_state", "last_action", "history_actor"],
        "critic_auxiliary": ["state", "privileged_state", "last_action", "history_actor"],
    }
    reward_channels = [
        {
            "name": "environment",
            "provider_name": "environment",
            "source": "environment",
            "timing": "transition",
            "context_dependent": False,
            "sign": 1,
        },
        {
            "name": "discriminator",
            "provider_name": "discriminator",
            "source": "recomputed",
            "timing": "next_state",
            "context_dependent": True,
            "sign": 1,
        },
    ]
    reward_channels.extend(
        {
            "name": name,
            "provider_name": name,
            "source": "stored_evidence",
            "timing": "transition",
            "context_dependent": False,
            "sign": -1,
        }
        for name in evidence
    )
    return {
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
            "distribution_cfg": {"class_name": "ClippedGaussianDistribution", "init_std": 0.05},
            "normalization_eps": 1e-5,
            "normalization_momentum": 0.01,
            "value_heads": [
                {
                    "spec": {
                        "name": "discriminator",
                        "kind": "critic",
                        "route": "critic_discriminator",
                        "reward_channels": ["discriminator"],
                        "ensemble_size": 2,
                        "has_target": True,
                    },
                    "network": value,
                },
                {
                    "spec": {
                        "name": "auxiliary",
                        "kind": "critic",
                        "route": "critic_auxiliary",
                        "reward_channels": list(evidence),
                        "ensemble_size": 2,
                        "has_target": True,
                    },
                    "network": value,
                },
            ],
        },
        "replay": {
            "class_name": "rsl_rl.storage.forward_backward_replay:ForwardBackwardReplay",
            "capacity_steps": 5_000,
            "terminal_capacity_per_env": 16,
            "autoreset_mode": "same_step",
            "environment_reward_name": "environment",
            "auxiliary_evidence_names": list(evidence),
            "reward_channels": reward_channels,
            "history_layout": {
                "history_field": "history_actor",
                "history_length": 4,
                "last_action_field": "last_action",
                "sources": [
                    {"observation_name": None, "start": 0, "stop": 29},
                    {"observation_name": "state", "start": 61, "stop": 64},
                    {"observation_name": "state", "start": 0, "stop": 29},
                    {"observation_name": "state", "start": 29, "stop": 58},
                    {"observation_name": "state", "start": 58, "stop": 61},
                ],
            },
        },
        "expert": {"provider": expert_provider, "window_lengths": (8, 257)},
        "algorithm": {
            "class_name": "rsl_rl.algorithms.forward_backward:ForwardBackward",
            "batch_size": 1024,
            "expert_sequence_length": 8,
            "gamma": 0.98,
            "learning_rate": 3e-4,
            "backward_learning_rate": 1e-5,
            "discriminator_learning_rate": 1e-5,
            "orthogonality_coefficient": 100.0,
            "discriminator_gradient_penalty_coefficient": 10.0,
            "context_goal_fraction": 0.2,
            "context_expert_fraction": 0.6,
            "relabel_fraction": 0.8,
            "context_buffer_capacity": 8_192,
            "rollout_context_refresh_steps": 100,
            "rollout_expert_fraction": 0.5,
            "rollout_expert_steps": 250,
            "rollout_expert_context_steps": 8,
            "value_cfg": {
                "discriminator": {
                    "learning_rate": 3e-4,
                    "actor_coefficient": 0.05,
                },
                "auxiliary": {
                    "learning_rate": 3e-4,
                    "actor_coefficient": 0.02,
                    "reward_coefficients": magnitudes,
                    "normalize_rewards": True,
                },
            },
        },
        "torch_compile_mode": "reduce-overhead",
    }


def bfm_zero_corrected_terminal_config(expert_provider: Callable) -> dict:
    """Return the measured 32 GiB path with true finals and CUDA graphs disabled."""
    config = copy.deepcopy(bfm_zero_native_config(expert_provider))
    config["torch_compile_mode"] = None
    return config
