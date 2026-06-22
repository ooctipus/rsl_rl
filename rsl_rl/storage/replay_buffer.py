# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Off-policy storage for FB-CPR. Thin re-export of the Meta Motivo buffers
# (ported verbatim in _fb_buffers.py / _zbuffer.py) under rsl_rl-friendly names.

from ._fb_buffers import DictBuffer, TrajectoryBuffer
from ._zbuffer import ZBuffer

# rsl_rl-facing aliases
ReplayBuffer = DictBuffer  # flat dict ring buffer for online transitions
ExpertTrajectoryBuffer = TrajectoryBuffer  # length-seq_length contiguous expert slices

__all__ = ["ReplayBuffer", "ExpertTrajectoryBuffer", "ZBuffer", "DictBuffer", "TrajectoryBuffer"]
