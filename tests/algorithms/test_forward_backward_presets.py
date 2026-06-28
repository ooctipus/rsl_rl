# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for frozen non-public Phase 1F reference configurations."""

from tests.fixtures.forward_backward_presets import (
    bfm_zero_corrected_terminal_config,
    bfm_zero_native_config,
    metamotivo_config,
)


def _provider(
    env: object,
    observation_schema: object,
    device: str,
    *,
    window_lengths: tuple[int, ...],
) -> None:
    """Raise if a static configuration test tries to load expert data."""
    del env, observation_schema, device, window_lengths
    raise AssertionError("Static preset tests must not load an expert corpus.")


def test_metamotivo_preset_freezes_humenv_reference_choices() -> None:
    """The Meta preset should retain its 1024-wide simple networks and 20/60/20 contexts."""
    cfg = metamotivo_config(_provider)

    assert cfg["model"]["context_dim"] == 256
    assert cfg["model"]["forward_cfg"] == {
        "hidden_dim": 1024,
        "hidden_layers": 2,
        "embedding_layers": 2,
    }
    assert cfg["model"]["initialization_type"] == "orthogonal"
    assert cfg["model"]["normalization_type"] == "exponential"
    assert cfg["model"]["normalization_eps"] == 1e-5
    assert cfg["model"]["normalization_momentum"] == 0.01
    assert cfg["algorithm"]["batch_size"] == 1024
    assert cfg["algorithm"]["expert_sequence_length"] == 8
    assert cfg["algorithm"]["context_goal_fraction"] == 0.2
    assert cfg["algorithm"]["context_expert_fraction"] == 0.6
    assert cfg["algorithm"]["relabel_fraction"] == 0.8


def test_bfm_preset_freezes_released_topology_and_compact_replay() -> None:
    """The BFM preset should retain released depth, rolling contexts, and proven history layout."""
    cfg = bfm_zero_native_config(_provider)

    assert cfg["model"]["actor_cfg"]["embedding_layers"] == 2
    assert cfg["model"]["forward_cfg"] == {
        "hidden_dim": 2048,
        "hidden_layers": 6,
        "embedding_layers": 6,
        "residual": True,
    }
    assert cfg["model"]["initialization_type"] == "orthogonal"
    assert cfg["model"]["normalization_type"] == "exponential"
    assert cfg["model"]["normalization_eps"] == 1e-5
    assert cfg["model"]["normalization_momentum"] == 0.01
    assert cfg["replay"]["capacity_steps"] == 5_000
    assert cfg["replay"]["history_layout"]["history_length"] == 4
    assert cfg["algorithm"]["rollout_expert_fraction"] == 0.5
    assert cfg["algorithm"]["rollout_expert_steps"] == 250
    assert cfg["algorithm"]["rollout_expert_context_steps"] == 8
    assert cfg["algorithm"]["random_action_range"] == (-5.0, 5.0)
    assert cfg["model"]["value_heads"][1]["spec"]["reward_composition"] == "scalar"
    assert cfg["algorithm"]["value_cfg"]["auxiliary"]["reward_coefficients"] == (
        0.0,
        0.1,
        10.0,
        0.0,
        1.0,
        0.4,
        4.0,
        2.0,
    )


def test_corrected_terminal_preset_records_the_measured_32_gib_compile_ablation() -> None:
    """The viable true-final path should disable the slower over-budget CUDA graphs explicitly."""
    native = bfm_zero_native_config(_provider)
    corrected = bfm_zero_corrected_terminal_config(_provider)

    assert native["torch_compile_mode"] == "reduce-overhead"
    assert corrected["torch_compile_mode"] is None
    native["torch_compile_mode"] = None
    assert corrected == native
    assert corrected is not native
    assert corrected["replay"] is not native["replay"]
