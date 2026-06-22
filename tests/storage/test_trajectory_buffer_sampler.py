# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Equivalence proof for the vectorized TrajectoryBuffer.sample (FB-CPR expert buffer).
#
# The original sampler was a per-slice Python loop with a `.item()` GPU sync per episode and a
# per-episode `torch.randint` for the start frame. The vectorized rewrite replaces that with a
# single advanced-index gather over a flat concatenation of all episodes. These tests prove the
# rewrite is equivalent:
#
#   1. Shapes + consumed keys ("observation", "next.observation") match the original.
#   2. STRUCTURAL correctness: every output row block is a real contiguous slice of some episode,
#      and the t+1 block equals the t block shifted by offset=1 in the SAME episode -- i.e. the
#      vectorized output is drawn from exactly the same set of valid slices the loop produces.
#   3. DISTRIBUTION equivalence: episode-selection frequencies match multinomial(priorities) and
#      start-frame positions are uniform on [0, valid_i) -- the same distribution as the original.

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

try:
    import pytest
except ModuleNotFoundError:  # allow standalone run in venvs without pytest
    class _NoPytest:
        class mark:
            @staticmethod
            def parametrize(*_a, **_k):
                def deco(fn):
                    return fn

                return deco

    pytest = _NoPytest()  # type: ignore

from rsl_rl.storage._fb_buffers import TrajectoryBuffer, dict_cat


# ----------------------------------------------------------------------------------------------
# Reference: the ORIGINAL per-slice loop sampler (verbatim, pre-optimization) for A/B comparison.
# ----------------------------------------------------------------------------------------------
def _original_sample(buf: TrajectoryBuffer, batch_size: int):
    num_slices = batch_size // buf.seq_length
    ep_ind = torch.multinomial(buf.priorities, num_slices, replacement=True)
    output = defaultdict(list)
    offset = 0
    if len(buf.output_key_tp1) > 0:
        offset = 1
        output["next"] = defaultdict(list)
    for ep_idx in ep_ind:
        _ep = buf.storage[ep_idx.item()]
        length = _ep[buf.output_key_t[0]].shape[0]
        time_idx = torch.randint(0, length - buf.seq_length - offset, (1,))
        for k in buf.output_key_t:
            output[k].append(_ep[k][time_idx : time_idx + buf.seq_length])
        for k in buf.output_key_tp1:
            output["next"][k].append(_ep[k][time_idx + offset : time_idx + offset + buf.seq_length])
    return dict_cat(output), ep_ind


def _make_buffer(seq_length=8, lengths=(50, 73, 120, 31, 200, 64, 99), obs_dim=358, device="cpu"):
    # Fill each episode with an identifiable encoding so a sampled row reveals its origin:
    #   obs[e][f, 0] = e*100000 + f    (episode id + frame)
    #   obs[e][f, j>0] = f             (frame index, for contiguity checks)
    eps = []
    for e, L in enumerate(lengths):
        obs = np.zeros((L, obs_dim), dtype=np.float32)
        obs[:, 0] = e * 100000 + np.arange(L)
        obs[:, 1:] = np.arange(L)[:, None]
        eps.append({"observation": obs})
    buf = TrajectoryBuffer(capacity=len(eps), seq_length=seq_length, device=device)
    buf.extend(eps)
    return buf, lengths, obs_dim


def _devices():
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


@pytest.mark.parametrize("device", _devices())
def test_shapes_and_keys_match(device):
    buf, lengths, obs_dim = _make_buffer(device=device)
    batch_size = 1024
    torch.manual_seed(0)
    out = buf.sample(batch_size)
    torch.manual_seed(0)
    ref, _ = _original_sample(buf, batch_size)

    # consumed keys identical
    assert set(out.keys()) == set(ref.keys()) == {"observation", "next"}
    assert set(out["next"].keys()) == set(ref["next"].keys()) == {"observation"}
    # shapes identical
    assert out["observation"].shape == ref["observation"].shape == (batch_size, obs_dim)
    assert out["next"]["observation"].shape == ref["next"]["observation"].shape == (batch_size, obs_dim)
    assert out["observation"].dtype == ref["observation"].dtype
    assert str(out["observation"].device).startswith(device)


@pytest.mark.parametrize("device", _devices())
def test_structural_correctness(device):
    """Every output row block is a real contiguous slice of some episode, and next == t shifted by 1."""
    buf, lengths, obs_dim = _make_buffer(device=device)
    seq = buf.seq_length
    batch_size = 2048
    num_slices = batch_size // seq

    torch.manual_seed(123)
    out = buf.sample(batch_size)
    obs = out["observation"].cpu().numpy().reshape(num_slices, seq, obs_dim)
    nxt = out["next"]["observation"].cpu().numpy().reshape(num_slices, seq, obs_dim)

    # decode episode id + start frame from the identifiable encoding (col0 = e*1e5 + f)
    for s in range(num_slices):
        col0 = obs[s, :, 0]
        # within a slice, frames must be contiguous & increasing by 1 -> same episode
        frames = obs[s, :, 1]  # == f
        assert np.allclose(frames, np.arange(frames[0], frames[0] + seq)), "t-slice not contiguous"
        ep_from_col0 = (col0 - frames) / 100000.0
        assert np.allclose(ep_from_col0, np.round(ep_from_col0)), "row not from a single episode"
        e = int(round(float(ep_from_col0[0])))
        assert np.all(np.round(ep_from_col0) == e), "t-slice spans >1 episode"
        f0 = int(round(float(frames[0])))
        # start frame must be within the valid range used by the original randint(0, L - seq - 1)
        assert 0 <= f0 <= lengths[e] - seq - 1, f"start {f0} out of valid range for ep {e} (L={lengths[e]})"
        # next-slice == t-slice shifted by offset=1 in the SAME episode
        assert np.allclose(nxt[s, :, 1], np.arange(f0 + 1, f0 + 1 + seq)), "next-slice not t+1"
        assert np.allclose((nxt[s, :, 0] - nxt[s, :, 1]) / 100000.0, e), "next-slice changed episode"


@pytest.mark.parametrize("device", _devices())
def test_episode_distribution_matches_multinomial(device):
    """Episode-selection frequency must match the priorities (same multinomial as original)."""
    buf, lengths, obs_dim = _make_buffer(device=device)
    # set non-uniform priorities to make the test discriminative
    pri = torch.tensor([1.0, 2.0, 3.0, 1.0, 4.0, 2.0, 1.0], device=device)
    buf.priorities = pri / pri.sum()
    expected = (pri / pri.sum()).cpu().numpy()

    torch.manual_seed(7)
    counts = np.zeros(len(lengths))
    n_draws = 4000
    batch_size = 8 * 64  # 64 slices per sample
    total_slices = 0
    for _ in range(n_draws // 64):
        buf.sample(batch_size)
        ep = buf.ep_ind.cpu().numpy()
        for e in ep:
            counts[e] += 1
        total_slices += len(ep)
    freq = counts / total_slices
    # within 3% absolute of the multinomial probabilities
    assert np.allclose(freq, expected, atol=0.03), f"freq {freq} vs expected {expected}"


@pytest.mark.parametrize("device", _devices())
def test_start_frame_uniform(device):
    """Start frames must be ~uniform on [0, L - seq - offset) for a chosen episode (matches randint)."""
    # single-episode buffer so every draw exercises the same valid range
    L = 300
    obs = np.zeros((L, 4), dtype=np.float32)
    obs[:, 1] = np.arange(L)
    buf = TrajectoryBuffer(capacity=1, seq_length=8, device=device)
    buf.extend([{"observation": obs}])

    torch.manual_seed(11)
    starts = []
    for _ in range(2000):
        out = buf.sample(8)  # 1 slice
        f0 = int(round(float(out["observation"][0, 1].item())))
        starts.append(f0)
    starts = np.array(starts)
    valid = L - 8 - 1  # 291
    assert starts.min() >= 0 and starts.max() <= valid - 1
    # mean of uniform[0, valid) ~ (valid-1)/2; allow generous tolerance for 2000 draws
    assert abs(starts.mean() - (valid - 1) / 2.0) < 0.1 * valid, f"mean {starts.mean()} not ~uniform"
    # coverage: should hit both low and high ends
    assert starts.min() < valid * 0.1 and starts.max() > valid * 0.9


@pytest.mark.parametrize("device", _devices())
def test_cache_invalidated_on_extend(device):
    """Adding episodes (filling the buffer) after a sample must rebuild the flat cache.

    The expert buffer is constructed with capacity == number of episodes and filled in one
    extend(), but it may also be filled across several extend() calls; the cache must rebuild
    against the full set once it is sampled again.
    """
    obs_dim = 16
    # capacity == total episodes (the expert-buffer invariant: always exactly full once loaded)
    buf = TrajectoryBuffer(capacity=3, seq_length=8, device=device)
    e0 = np.zeros((40, obs_dim), dtype=np.float32); e0[:, 0] = np.arange(40)
    buf.extend([{"observation": e0}, {"observation": e0.copy()}])  # 2 of 3 filled
    e1 = np.zeros((50, obs_dim), dtype=np.float32); e1[:, 0] = np.arange(50)
    buf.extend([{"observation": e1}])  # now full (3/3); cache must be dropped by extend
    assert buf._flat is None
    buf.sample(64)
    assert buf._flat is not None
    assert buf._flat["observation"].shape[0] == 130  # 40 + 40 + 50


def _run_standalone():
    """Run all tests without pytest (for venvs lacking pytest). Returns nonzero on failure."""
    import traceback

    tests = [
        test_shapes_and_keys_match,
        test_structural_correctness,
        test_episode_distribution_matches_multinomial,
        test_start_frame_uniform,
        test_cache_invalidated_on_extend,
    ]
    failures = 0
    for fn in tests:
        for dev in _devices():
            try:
                fn(dev)
                print(f"PASS  {fn.__name__}[{dev}]")
            except Exception:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {fn.__name__}[{dev}]")
                traceback.print_exc()
    print(f"\n{'ALL PASSED' if failures == 0 else str(failures) + ' FAILURES'}")
    return failures


if __name__ == "__main__":
    import sys

    try:
        import pytest as _pt  # noqa: F401

        sys.exit(pytest.main([__file__, "-v"]))
    except ModuleNotFoundError:
        sys.exit(1 if _run_standalone() else 0)
