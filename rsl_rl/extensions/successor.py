# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forward-backward successor-representation critic, a config-gated PPO augmentation.

Replaces the scalar critic with a goal-conditioned successor-*representation* value (Touati-Ollivier
2021 / Meta-Motivo). Two maps on the critic model -- the backward ``B(s)`` (hard-normed to ``sqrt(d)``,
"what a state is") and the z-conditioned forward ``F(s, z)`` (free, "where ``s`` is headed under the
goal ``z``") -- are learned **reward-free** so their inner product approximates the discounted
state-occupancy measure of the goal-``z`` policy::

    <F(s, z), B(s')>  ~prop~  M^{pi_z}(s -> s') = sum_k gamma^k Pr(s_k = s' | s_0 = s, pi_z).

GOAL-REACHING IS INTRINSIC -- there is no learned reward read-out. A goal is the embedding
``z = project_z(B(goal))``, and the value is the occupancy of goal-aligned states::

    V(s; z) = <F(s, z), z>   ( = E[ sum_k gamma^k <B(s_k), z> ] ),

i.e. the successor read-out with ``w = z``. The implied per-step reward ``r(s') = <B(s'), z>`` is the
dense goal-alignment, and the successor measure propagates that (sparse) alignment into a dense value --
so the policy reaches goals with NO hand-designed success reward (Meta-Motivo's zero-shot identity, for
the goal-reaching reward ``z_r = B(goal)``). This module owns only the reward-free losses and the target
network; ``F``/``B`` live on the critic model (which must expose ``forward_map``/``backward``).

The objective is forward-backward (TD on the successor measure): it grounds one-step reachability and
chains it through the recursion, with an orthonormality penalty ``E[B B^T]=I`` for anti-collapse /
decorrelation (``B``'s scale is pinned by its hard ``sqrt(d)`` norm).
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from tensordict import TensorDict

    from rsl_rl.models import MLPModel


class SuccessorFeatures(nn.Module):
    """Reward-free forward-backward occupancy losses + the target network for ``V = <F(s, z), z>``.

    Args:
        feature_dim: Width ``d`` of ``F`` / ``B``.
        gamma: Discount (matches the policy's); sets the occupancy horizon.
        occupancy: Which reward-free objective trains ``F``/``B`` -- ``"bilinear_fb"`` (the ``[n, n]``
            forward-backward measure) or ``"vector_td"`` (the successor-feature vector TD). Switches only
            :meth:`representation_loss`; the model, ``z``, value, and actor are shared. See
            :class:`~isaaclab_rl.rsl_rl.RslRlSuccessorCfg` for the full description.
        ortho_coef: Weight of the backward orthonormality penalty ``E[B B^T] -> I`` (decorrelates states).
            Meta-Motivo's default is ``1.0`` (DMC) up to ``100`` (humanoid).
        train_goal_ratio: Fraction of the FB ``z`` drawn from goal embeddings ``project_z(B(s')[perm])``; the
            remaining ``1 - ratio`` are uniform on the ``sqrt(d)`` sphere. The random-sphere half is what bounds
            ``F`` across the latent space (Meta-Motivo ``train_goal_ratio``, default ``0.5``).
        goal_command_name: Name of the command-manager term (a ``StateCommand``) exposing the per-task target
            observation cache via ``get_target_obs_cache()`` and the per-env ``cmd_indices``; bound once at
            construction. Must match the env's command term (default ``"goal_point"`` for the position task).
        fb_batch_size: Cap on the forward-backward batch-matrix size (Meta-Motivo trains FB at 1024). If the
            PPO minibatch is larger, the FB loss subsamples to this many states; otherwise it's a no-op.
        target_tau: Polyak rate for the forward-backward target network, applied PER GRADIENT STEP. The PPO
            update calls :meth:`update_target` once per minibatch (= one gradient step), so this matches
            Meta-Motivo's per-step ``fb_target_tau`` (default ``0.01``).
        device: Torch device.
    """

    def __init__(
        self,
        feature_dim: int = 128,
        gamma: float = 0.99,
        occupancy: str = "bilinear_fb",
        ortho_coef: float = 1.0,
        train_goal_ratio: float = 0.5,
        goal_command_name: str = "goal_point",
        fb_batch_size: int = 1024,
        target_tau: float = 0.01,
        device: str = "cpu",
    ) -> None:
        """Store hyperparameters; the forward-backward target network is built later in :meth:`build`."""
        warnings.warn(
            "SuccessorFeatures is deprecated. Migrate off-policy forward-backward training to "
            "rsl_rl.algorithms.ForwardBackward with rsl_rl.runners.OffPolicyRunner. The APIs are not "
            "drop-in compatible because ForwardBackward owns replay, reward channels, and optimizer state.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self.feature_dim = feature_dim
        self.gamma = gamma
        if occupancy not in ("bilinear_fb", "vector_td"):
            raise ValueError(f"occupancy must be 'bilinear_fb' or 'vector_td', got {occupancy!r}")
        self.occupancy = occupancy
        self.ortho_coef = ortho_coef
        self.train_goal_ratio = train_goal_ratio
        self.goal_command_name = goal_command_name
        self.fb_batch_size = fb_batch_size
        self.target_tau = target_tau
        self._device = device
        # No learned reward read-out: the value is V(s) = <F(s, z), z> (w = z, the goal embedding), derived from
        # the reward-free F. There are no learnable parameters here -- F/B live on the critic model.
        self.target_critic: MLPModel | None = None  # frozen Polyak copy for the FB bootstrap
        # z-conditioned VALUE goal: the per-task target observation library (raw obs at each goal, delta-0) and a
        # callable returning the current per-env task index. Bound (REQUIRED) at ``construct_algorithm`` via
        # :meth:`bind_goals`; the value and actor read it directly, with no fallback.
        self.goal_cache: TensorDict | None = None
        self.cmd_indices_fn = None

    def build(self, critic: MLPModel) -> None:
        """Create the frozen forward-backward target network (a Polyak copy of the critic)."""
        self.target_critic = copy.deepcopy(critic).to(self._device).requires_grad_(False)
        self.target_critic.eval()

    def bind_goals(self, goal_cache: TensorDict, cmd_indices_fn: Callable[[], torch.Tensor]) -> None:
        """Bind the goal library for the z-conditioned value ``V(s) = <F(s, z), w>``, ``z = B(goal)``.

        Args:
            goal_cache: ``[num_tasks, ...]`` raw observations at each task's TARGET config (delta-0, the
                "arrived" state); ``z`` is recomputed live as ``project_z(B(goal_cache[task]))`` each update so it
                tracks the still-training backward map (raw obs are cached, not stale embeddings).
            cmd_indices_fn: Zero-arg callable returning the current ``[num_envs]`` per-env task index (which goal
                each env is commanded to reach), read at rollout time and stored per transition.
        """
        self.goal_cache = goal_cache.to(self._device)
        self.cmd_indices_fn = cmd_indices_fn

    @torch.no_grad()
    def update_target(self, critic: MLPModel) -> None:
        """Polyak-update the forward-backward target network toward the live critic.

        Parameters are Polyak-lagged (slow target). Buffers are NOT: ``obs_normalizer`` running stats
        (``EmpiricalNormalization`` registers ``_mean``/``_var``/``count`` as buffers, not parameters) must
        track the LIVE normalization, else the target net scores ``next_obs`` under stale init stats
        (mean 0/std 1) while the live net normalizes -- a scale-mismatched bootstrap that flips the FB
        residual sign. Hard-copy every buffer each step so both nets normalize identically.
        """
        if self.target_critic is not None:
            for param, target_param in zip(critic.parameters(), self.target_critic.parameters()):
                target_param.lerp_(param.detach(), self.target_tau)
            for buf, target_buf in zip(critic.buffers(), self.target_critic.buffers()):
                target_buf.copy_(buf)

    def _project_z(self, z: torch.Tensor) -> torch.Tensor:
        """Project ``z`` onto the ``sqrt(d)`` sphere (Meta-Motivo ``project_z``, ``norm_z=True``)."""
        return math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)

    def goal_z(self, critic: MLPModel, cmd_indices: torch.Tensor) -> torch.Tensor:
        """Return the goal embedding ``z = project_z(B(goal_cache[cmd_indices]))`` ``[N, d]``, DETACHED.

        The cache holds each task's TARGET obs (delta-0, "arrived"), so ``B`` of it is the goal-state embedding.
        Recomputed live (``B`` is still training); ``z`` is detached -- it conditions ``F``/the actor, shared by
        the z-conditioned actor and the value, and is never trained by these consumers.
        """
        return self._project_z(critic.backward(self.goal_cache[cmd_indices])).detach()

    def state_value(self, critic: MLPModel, obs: TensorDict, z: torch.Tensor) -> torch.Tensor:
        """Return the PPO state value ``V(s; z) = <F(s, z), z>`` (shape ``[B, 1]``) for goal embedding ``z = B(goal)``.

        ``w = z``: goal-reaching needs no learned read-out. The value is the discounted occupancy of
        goal-aligned states and is consistent with the implied reward ``<B(s'), z>`` by construction (the FB
        loss already trains ``F`` to be the successor measure of ``B``). ``F`` is DETACHED: V is DERIVED, the
        value path trains nothing; the representation is owned solely by the reward-free FB + ortho loss.
        """
        psi = critic.forward_map(obs, z).detach()  # detached: V is derived from F; no value-side training
        return (psi * z).sum(dim=-1, keepdim=True)

    def goal_alignment_reward(self, critic: MLPModel, next_obs: TensorDict, z: torch.Tensor) -> torch.Tensor:
        """Return the intrinsic per-step reward ``r(s') = <B(s'), z>`` (shape ``[N]``), DETACHED.

        The goal-alignment of the realized next state with the goal embedding ``z``. This is the reward whose
        successor value is ``V = <F(s, z), z>`` -- the policy maximizing it reaches the goal with no extrinsic
        success reward. Detached (a reward target, not trained).
        """
        return (critic.backward(next_obs).detach() * z).sum(dim=-1)

    def value(self, critic: MLPModel, obs: TensorDict) -> torch.Tensor:
        """Return the state value ``V(s)`` at rollout/bootstrap time, ``z = B(current task goal)``.

        Reads the per-env commanded goal via the bound ``cmd_indices_fn`` and the goal cache, both bound
        deterministically at construction (:meth:`bind_goals`) -- no fallback; an unbound cache is a
        construction bug, not a runtime branch.
        """
        return self.state_value(critic, obs, self.goal_z(critic, self.cmd_indices_fn()))

    def representation_loss(
        self,
        critic: MLPModel,
        obs_s: TensorDict,
        next_obs: TensorDict,
        terminal: torch.Tensor,
        timeout: torch.Tensor,
        cmd_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the reward-free occupancy objective training ``F``/``B``, dispatched by :attr:`occupancy`.

        Both variants share the z-mixing, the model (``F(s,z)``/``B(s)``), and the value ``V=<F(s,z),z>``; they
        differ ONLY in how ``F`` is trained. ``B`` is hard-normed to ``sqrt(d)`` + decorrelated by the
        orthonormality penalty.

        ``z`` is MIXED (Meta-Motivo ``sample_mixed_z``): ``train_goal_ratio`` of the rows use the env's on-policy
        goal ``z = B(goal_cache[cmd_indices])`` (the z that generated this transition, kept LOAD-BEARING) and the
        rest a uniform ``sqrt(d)``-sphere z (the sphere fraction keeps the representation full-rank). ``z`` is
        DETACHED. Terminals bootstrap to 0 (``disc=0``); timeout rows are dropped.

        * ``"bilinear_fb"`` -- Touati-Ollivier / Meta-Motivo FB measure (``fb/agent.py:204-207``) on the
          ``[n, n]`` batch matrix ``M[i,j] = <F(s_i, z_i), B(s'_j)>`` (frozen target ``M_bar``,
          ``diff = M - gamma*(1-term)*M_bar``)::

              fb_offdiag = 0.5 * mean_{i != j}( diff[i,j]^2 )    # diagonal MASKED OUT of the TD
              fb_diag    = - mean_i( diff[i,i] )                 # coefficient 1, the +1 immediate term

        * ``"vector_td"`` -- Barreto/Borsa successor-feature vector TD: regress the whole vector
          ``F(s,z) -> B(s) + gamma*(1-term)*F_bar(s',z)`` (``B(s)`` detached -> trains only ``F``;
          gamma-contraction self-bounds, no batch matrix). The single-task special case of FB.

        Returns ``(loss, ortho, f_norm, m_diag)``; ``m_diag = E[<F(s,z), B(s')>]`` (the realized measure
        diagonal, shared across variants) should converge toward ``1/(1-gamma)``.
        """
        # Drop timeout rows (their next_obs is the auto-reset state, not a real continuation).
        keep = timeout.squeeze(-1) == 0
        obs_s, next_obs, terminal = obs_s[keep], next_obs[keep], terminal[keep]
        if cmd_indices is not None:
            cmd_indices = cmd_indices[keep]
        # Cap the O(n^2) measure matrix (Meta-Motivo trains FB at 1024).
        n = terminal.shape[0]
        if n > self.fb_batch_size:
            idx = torch.randperm(n, device=terminal.device)[: self.fb_batch_size]
            obs_s, next_obs, terminal = obs_s[idx], next_obs[idx], terminal[idx]
            if cmd_indices is not None:
                cmd_indices = cmd_indices[idx]

        phi_next = critic.backward(next_obs)  # B(s'): the MEASURE columns (states reached, LIVE grad to B)
        # MIXED z (Meta-Motivo ``sample_mixed_z``): a ``train_goal_ratio`` fraction uses the env's on-policy goal
        # z = B(goal_cache[cmd_indices]) -- the z that generated this transition under the z-conditioned actor,
        # which keeps z LOAD-BEARING (F is trained on its own data, the PPO analog of MM's replay relabeling) --
        # and the rest uses a uniform sqrt(d)-sphere z. The sphere fraction is essential: with the curriculum
        # concentrating cmd_indices on a few goals, an all-goal z collapses the [n, n] measure to near rank-1, the
        # (n-1) off-diagonal terms per B-column outvote the single diagonal pull, and m_diag is dragged negative.
        # The sphere z diversifies the rows so the measure stays full-rank and bounded. z is DETACHED.
        z_rand = self._project_z(torch.randn_like(phi_next))
        if cmd_indices is not None and self.goal_cache is not None:
            z_goal = self.goal_z(critic, cmd_indices)
            use_goal = torch.rand((z_goal.shape[0], 1), device=z_goal.device) < self.train_goal_ratio
            z = torch.where(use_goal, z_goal, z_rand)
        else:
            z = z_rand
        f_s = critic.forward_map(obs_s, z)  # F(s, z) = psi, LIVE (grad to F); the successor in both variants
        disc = (self.gamma * (1.0 - terminal.squeeze(-1))).unsqueeze(-1)  # [n, 1] per-row (s_i), 0 at terminals
        f_norm = f_s.norm(dim=-1).mean().detach()  # shared diagnostic: ||F|| self-bounds
        m_diag = (f_s * phi_next).sum(dim=-1).mean().detach()  # shared diag: realized <F(s,z),B(s')> -> 1/(1-g)

        if self.occupancy == "vector_td":
            # Successor-feature vector TD (Barreto/Borsa): F(s,z) -> B(s) + gamma*(1-term)*F_bar(s',z). B(s) is
            # detached in the target so the TD trains only F; B is owned by the orthonormality penalty. The
            # gamma-contraction self-bounds F (no clamp, no batch matrix).
            phi_s = critic.backward(obs_s)  # B(s) = phi, the per-step feature
            with torch.no_grad():
                psi_bar_next = self.target_critic.forward_map(next_obs, z)  # F_bar(s', z), target net
            td_target = phi_s.detach() + disc * psi_bar_next
            loss = (f_s - td_target).pow(2).sum(dim=-1).mean()
            ortho = self._orthonormality(phi_s)
        else:  # bilinear_fb
            with torch.no_grad():
                f_next_bar = self.target_critic.forward_map(next_obs, z)  # F_bar(s', z), target net
                phi_next_bar = self.target_critic.backward(next_obs)  # B_bar(s'), target net
                m_tgt = f_next_bar @ phi_next_bar.transpose(0, 1)  # [n, n] frozen target measure M_bar
            m = f_s @ phi_next.transpose(0, 1)  # [n, n] live measure M (grad to F and B)
            diff = m - disc * m_tgt  # [n, n] FB residual
            off_diag = 1.0 - torch.eye(m.shape[0], device=m.device, dtype=m.dtype)  # post-subsample size
            fb_offdiag = 0.5 * (diff * off_diag).pow(2).sum() / off_diag.sum()  # mean over n*(n-1); diag excluded
            fb_diag = -diff.diagonal().mean()  # coefficient 1; the +1 immediate term
            loss = fb_offdiag + fb_diag
            ortho = self._orthonormality(phi_next)
        return loss, ortho, f_norm, m_diag

    def _orthonormality(self, phi: torch.Tensor) -> torch.Tensor:
        """Meta-Motivo backward orthonormality on the batch Gram ``phi phi^T -> I`` (decorrelate states).

        The Touati-Ollivier estimator ``E[(B(s).B(s'))^2] - 2 E[||B(s)||^2]`` (equivalent in expectation to
        ``||E[phi phi^T] - I||^2``). With ``phi`` hard-normed to sqrt(d) the diagonal is fixed (``-diag`` is
        inert, kept for parity); the off-diagonal penalty drives different states' embeddings apart.
        """
        bs = phi.shape[0]
        cov = phi @ phi.transpose(0, 1)  # [bs, bs] state Gram
        off = 0.5 * (cov * (1.0 - torch.eye(bs, device=phi.device, dtype=phi.dtype))).pow(2).sum() / (bs * (bs - 1))
        diag = -cov.diagonal().mean()
        return off + diag
