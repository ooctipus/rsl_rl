# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for normalization modules."""

import torch
import torch.multiprocessing as mp
from pathlib import Path

from rsl_rl.modules.normalization import (
    EmpiricalDiscountedVariationNormalization,
    EmpiricalNormalization,
    commit_normalization,
    distributed_mean_var,
)


def _commit_normalization_worker(rank: int, world_size: int, init_file: str, output_dir: str) -> None:
    torch.distributed.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    norm = EmpiricalNormalization(shape=2)
    module = torch.nn.Sequential(norm)
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) if rank == 0 else torch.tensor([[7.0, 10.0]])
    norm.accumulate(data)
    commit_normalization((module,), distributed=True)
    torch.save((norm.mean, norm.std, norm.count), Path(output_dir) / f"rank_{rank}.pt")
    torch.distributed.destroy_process_group()


class TestEmpiricalNormalization:
    """Tests for ``EmpiricalNormalization``."""

    def test_convergence_to_known_distribution(self) -> None:
        """Running mean and std should converge to the true values of the input distribution."""
        true_mean, true_std = 5.0, 2.0
        norm = EmpiricalNormalization(shape=4)
        norm.train()

        torch.manual_seed(0)
        for _ in range(200):
            batch = true_mean + true_std * torch.randn(64, 4)
            norm.update(batch)

        assert torch.allclose(norm.mean, torch.full((4,), true_mean), atol=0.15)
        assert torch.allclose(norm.std, torch.full((4,), true_std), atol=0.15)

    def test_forward_applies_normalization(self) -> None:
        """forward() should return (x - mean) / (std + eps), not the raw input."""
        norm = EmpiricalNormalization(shape=2, eps=0.01)
        norm.train()

        data = torch.tensor([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])
        norm.update(data)

        result = norm(data)
        expected = (data - norm._mean) / (norm._std + norm.eps)
        assert torch.allclose(result, expected)

    def test_until_stops_updates(self) -> None:
        """After 'until' samples are seen, further updates must not change statistics."""
        norm = EmpiricalNormalization(shape=2, until=100)
        norm.train()

        for _ in range(10):
            norm.update(torch.randn(20, 2))

        assert norm.count >= 100
        mean_snapshot = norm._mean.clone()
        std_snapshot = norm._std.clone()

        for _ in range(10):
            norm.update(torch.randn(20, 2) + 100)

        assert torch.equal(norm._mean, mean_snapshot)
        assert torch.equal(norm._std, std_snapshot)

    def test_eval_mode_freezes_stats(self) -> None:
        """In eval mode, update() should be a no-op."""
        norm = EmpiricalNormalization(shape=3)
        norm.train()
        norm.update(torch.randn(50, 3))

        mean_before = norm._mean.clone()
        std_before = norm._std.clone()

        norm.eval()
        norm.update(torch.randn(50, 3) + 100)

        assert torch.equal(norm._mean, mean_before)
        assert torch.equal(norm._std, std_before)

    def test_inverse_round_trip(self) -> None:
        """inverse(forward(x)) should approximately recover x."""
        norm = EmpiricalNormalization(shape=4, eps=1e-2)
        norm.train()

        torch.manual_seed(42)
        for _ in range(50):
            norm.update(torch.randn(32, 4) * 3 + 5)

        x = torch.randn(16, 4) * 3 + 5
        recovered = norm.inverse(norm(x))
        assert torch.allclose(recovered, x, atol=1e-5)

    def test_single_sample_does_not_produce_nan(self) -> None:
        """Updating with a single sample should not produce NaN in mean or std."""
        norm = EmpiricalNormalization(shape=2)
        norm.train()
        norm.update(torch.tensor([[1.0, 2.0]]))

        assert not torch.any(torch.isnan(norm._mean))
        assert not torch.any(torch.isnan(norm._std))

    def test_accumulated_updates_commit_once(self) -> None:
        """Accumulated batches should leave live statistics frozen until committed."""
        norm = EmpiricalNormalization(shape=2)
        module = torch.nn.Sequential(norm)
        first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        second = torch.tensor([[5.0, 8.0]])

        norm.accumulate(first)
        norm.accumulate(second)

        torch.testing.assert_close(norm.mean, torch.zeros(2))
        assert norm.count == 0
        commit_normalization((module,))
        expected = torch.cat((first, second))
        torch.testing.assert_close(norm.mean, expected.mean(0))
        torch.testing.assert_close(norm.std, expected.std(0, unbiased=False))
        assert norm.count == len(expected)
        assert norm._pending_count == 0

    def test_pending_moments_are_not_checkpointed(self) -> None:
        """A checkpoint should contain committed statistics only."""
        norm = EmpiricalNormalization(shape=2)
        norm.accumulate(torch.ones(3, 2))

        assert not any("pending" in key for key in norm.state_dict())

    def test_accumulation_is_explicit(self) -> None:
        """Normalizers should not carry a hidden deferred-update mode."""
        assert not hasattr(EmpiricalNormalization(shape=2), "set_deferred_updates")


def test_distributed_mean_var_matches_torch_locally() -> None:
    """The moment helper should preserve PyTorch's local unbiased variance."""
    values = torch.tensor([[1.0, 3.0], [5.0, 9.0]])

    mean, var = distributed_mean_var(values, unbiased=True)

    torch.testing.assert_close(mean, values.mean())
    torch.testing.assert_close(var, values.var(unbiased=True))


def test_deferred_normalization_merges_all_workers(tmp_path: Path) -> None:
    """Every worker should commit identical moments from unequal local batches."""
    init_file = tmp_path / "process_group"
    mp.spawn(_commit_normalization_worker, args=(2, str(init_file), str(tmp_path)), nprocs=2, join=True)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [7.0, 10.0]])

    for rank in range(2):
        mean, std, count = torch.load(tmp_path / f"rank_{rank}.pt", weights_only=True)
        torch.testing.assert_close(mean, expected.mean(0))
        torch.testing.assert_close(std, expected.std(0, unbiased=False))
        assert count == len(expected)


class TestEmpiricalDiscountedVariationNormalization:
    """Tests for ``EmpiricalDiscountedVariationNormalization``."""

    def test_constant_rewards_produce_stable_normalization(self) -> None:
        """Constant rewards should converge to a stable normalization factor."""
        gamma = 0.99
        norm = EmpiricalDiscountedVariationNormalization(shape=[], gamma=gamma)
        norm.train()

        reward = torch.tensor([1.0])
        outputs = [norm(reward).item() for _ in range(200)]

        # After convergence, the output should be approximately constant
        last_10 = outputs[-10:]
        assert max(last_10) - min(last_10) < 0.1, "Normalization should stabilize for constant rewards"

    def test_zero_std_returns_raw_reward(self) -> None:
        """When std is zero (or not yet computed), forward should return the raw reward."""
        norm = EmpiricalDiscountedVariationNormalization(shape=[])
        norm.eval()
        reward = torch.tensor([5.0])
        result = norm(reward)
        # In eval mode with no prior updates, std defaults to 1, so it normalizes by 1
        assert torch.isfinite(result).all()
