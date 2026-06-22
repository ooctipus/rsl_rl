# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# FB-CPR (Forward-Backward representations with Conditional Policy Regularization), an
# off-policy algorithm. The update math is a faithful port of Meta Motivo
# (facebookresearch/metamotivo, CC BY-NC 4.0): metamotivo.fb.agent.FBAgent +
# metamotivo.fb_cpr.agent.FBcprAgent, restructured to follow rsl_rl conventions
# (act / process_env_step / compute_returns / update / save / load / construct_algorithm).

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import autograd

from rsl_rl.models.fb_model import ForwardBackwardModel
from rsl_rl.modules.fb_modules import _soft_update_params, eval_mode, weight_init
from rsl_rl.storage.replay_buffer import ReplayBuffer, ZBuffer


class FBCPR:
    """Off-policy FB-CPR algorithm."""

    policy: ForwardBackwardModel

    def __init__(
        self,
        model: ForwardBackwardModel,
        replay_buffer: ReplayBuffer,
        expert_buffer,
        num_envs: int,
        action_dim: int,
        # optim
        lr_f: float = 1e-4,
        lr_b: float = 1e-5,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-4,
        lr_discriminator: float = 1e-5,
        weight_decay: float = 0.0,
        weight_decay_discriminator: float = 0.0,
        clip_grad_norm: float = 0.0,
        # fb / actor / critic
        fb_target_tau: float = 0.01,
        critic_target_tau: float = 0.005,
        ortho_coef: float = 100.0,
        fb_pessimism_penalty: float = 0.0,
        actor_pessimism_penalty: float = 0.5,
        critic_pessimism_penalty: float = 0.5,
        stddev_clip: float = 0.3,
        q_loss_coef: float = 0.1,
        reg_coeff: float = 0.01,
        scale_reg: bool = True,
        # discriminator
        grad_penalty_discriminator: float = 10.0,
        # z mixing
        train_goal_ratio: float = 0.2,
        expert_asm_ratio: float = 0.6,
        relabel_ratio: float = 0.8,
        use_mix_rollout: bool = True,
        update_z_every_step: int = 150,
        # rollout
        batch_size: int = 1024,
        discount: float = 0.98,
        num_seed_steps: int = 50_000,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs,  # swallow unused config keys
    ) -> None:
        self.device = device
        self._model = model.to(device)
        self.policy = self._model
        self.replay_buffer = replay_buffer
        self.expert_buffer = expert_buffer
        self.num_envs = num_envs
        self.action_dim = action_dim

        # hyperparameters
        self.batch_size = batch_size
        self.discount = discount
        self.fb_target_tau = float(min(max(fb_target_tau, 0), 1))
        self.critic_target_tau = float(min(max(critic_target_tau, 0), 1))
        self.ortho_coef = ortho_coef
        self.fb_pessimism_penalty = fb_pessimism_penalty
        self.actor_pessimism_penalty = actor_pessimism_penalty
        self.critic_pessimism_penalty = critic_pessimism_penalty
        self.stddev_clip = stddev_clip
        self.q_loss_coef = q_loss_coef
        self.reg_coeff = reg_coeff
        self.scale_reg = scale_reg
        self.grad_penalty_discriminator = grad_penalty_discriminator
        self.train_goal_ratio = train_goal_ratio
        self.expert_asm_ratio = expert_asm_ratio
        self.relabel_ratio = relabel_ratio
        self.use_mix_rollout = use_mix_rollout
        self.update_z_every_step = update_z_every_step
        self.clip_grad_norm = clip_grad_norm if clip_grad_norm > 0 else None
        self.num_seed_steps = num_seed_steps
        self.seq_length = self._model.cfg.seq_length
        self.actor_std = self._model.cfg.actor_std

        # enable training on the model + create targets (mirrors FBAgent.setup_training)
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()

        self.forward_optimizer = torch.optim.Adam(self._model._forward_map.parameters(), lr=lr_f, weight_decay=weight_decay)
        self.backward_optimizer = torch.optim.Adam(self._model._backward_map.parameters(), lr=lr_b, weight_decay=weight_decay)
        self.actor_optimizer = torch.optim.Adam(self._model._actor.parameters(), lr=lr_actor, weight_decay=weight_decay)
        self.critic_optimizer = torch.optim.Adam(self._model._critic.parameters(), lr=lr_critic, weight_decay=weight_decay)
        self.discriminator_optimizer = torch.optim.Adam(
            self._model._discriminator.parameters(), lr=lr_discriminator, weight_decay=weight_decay_discriminator
        )

        # param lists for soft updates
        self._forward_map_paramlist = tuple(self._model._forward_map.parameters())
        self._target_forward_map_paramlist = tuple(self._model._target_forward_map.parameters())
        self._backward_map_paramlist = tuple(self._model._backward_map.parameters())
        self._target_backward_map_paramlist = tuple(self._model._target_backward_map.parameters())
        self._critic_paramlist = tuple(self._model._critic.parameters())
        self._target_critic_paramlist = tuple(self._model._target_critic.parameters())

        self.off_diag = 1 - torch.eye(batch_size, batch_size, device=device)
        self.off_diag_sum = self.off_diag.sum()
        # cache the z-mixing categorical (train_goal / expert_asm / random) once on-device so
        # sample_mixed_z doesn't rebuild it (a tiny H2D) every update. Values are unchanged.
        self._mix_prob = torch.tensor(
            [train_goal_ratio, expert_asm_ratio, 1.0 - train_goal_ratio - expert_asm_ratio],
            dtype=torch.float32, device=device,
        )
        self.z_buffer = ZBuffer(int(kwargs.get("z_buffer_size", 10000)), self._model.cfg.archi.z_dim, device)

        # rollout state
        self._z = None  # [num_envs, z_dim]
        self._step_count = torch.zeros(num_envs, device=device)
        self._prev_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._collected_steps = 0
        self._pending = {}  # transition pieces being assembled this step

        self.is_multi_gpu = multi_gpu_cfg is not None
        self._cudagraphs = False  # set by compile() when mode enables cuda-graphs

    # ============================ rollout interface ============================
    @torch.no_grad()
    def maybe_update_rollout_context(self, z, step_count):
        if z is not None:
            mask_reset_z = (step_count % self.update_z_every_step == 0).reshape(-1, 1)
            if self.use_mix_rollout and not self.z_buffer.empty():
                new_z = self.z_buffer.sample(z.shape[0], device=self.device)
            else:
                new_z = self._model.sample_z(z.shape[0], device=self.device)
            z = torch.where(mask_reset_z, new_z, z.to(self.device))
        else:
            z = self._model.sample_z(step_count.shape[0], device=self.device)
        return z

    @torch.no_grad()
    def act(self, obs: TensorDict) -> torch.Tensor:
        obs_t = obs["policy"].to(self.device)
        self._z = self.maybe_update_rollout_context(self._z, self._step_count)
        if self._collected_steps < self.num_seed_steps:
            action = 2.0 * torch.rand((self.num_envs, self.action_dim), device=self.device) - 1.0
        else:
            action = self._model.act(obs_t, self._z, mean=False)
        # stash pieces for process_env_step
        self._pending = {"obs": obs_t, "action": action, "z": self._z.clone()}
        return action

    @torch.no_grad()
    def process_env_step(self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        next_obs = obs["policy"].to(self.device)
        dones = dones.to(self.device).bool().reshape(-1)
        time_outs = extras.get("time_outs", torch.zeros_like(dones)).to(self.device).bool().reshape(-1)
        terminated = dones & ~time_outs

        # add only transitions for envs that were NOT reset in the previous step
        keep = ~self._prev_done
        if keep.any():
            idx = keep.nonzero(as_tuple=False).reshape(-1)
            data = {
                "observation": self._pending["obs"][idx],
                "action": self._pending["action"][idx],
                "z": self._pending["z"][idx],
                "next": {
                    "observation": next_obs[idx],
                    "terminated": terminated[idx].reshape(-1, 1),
                    "truncated": time_outs[idx].reshape(-1, 1),
                },
            }
            self.replay_buffer.extend(data)

        # advance per-env step counters / done bookkeeping
        self._step_count = self._step_count + 1
        self._step_count = torch.where(dones, torch.zeros_like(self._step_count), self._step_count)
        self._prev_done = dones
        self._collected_steps += self.num_envs

    def compute_returns(self, obs: TensorDict) -> None:
        pass  # off-policy: no GAE

    @property
    def seeding(self) -> bool:
        return self._collected_steps < self.num_seed_steps

    # ============================ losses (ported) ============================
    def get_targets_uncertainty(self, preds, pessimism_penalty):
        dim = 0
        preds_mean = preds.mean(dim=dim)
        preds_uns = preds.unsqueeze(dim=dim)
        preds_uns2 = preds.unsqueeze(dim=dim + 1)
        preds_diffs = torch.abs(preds_uns - preds_uns2)
        scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = preds_diffs.sum(dim=(dim, dim + 1)) / scaling
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc

    @torch.no_grad()
    def encode_expert(self, next_obs):
        B_expert = self._model._backward_map(next_obs).detach()
        B_expert = B_expert.view(self.batch_size // self.seq_length, self.seq_length, B_expert.shape[-1])
        z_expert = B_expert.mean(dim=1)
        z_expert = self._model.project_z(z_expert)
        z_expert = torch.repeat_interleave(z_expert, self.seq_length, dim=0)
        return z_expert

    @torch.no_grad()
    def sample_mixed_z(self, train_goal, expert_encodings):
        z = self._model.sample_z(self.batch_size, device=self.device)
        mix_idxs = torch.multinomial(self._mix_prob, num_samples=self.batch_size, replacement=True).reshape(-1, 1)
        perm = torch.randperm(self.batch_size, device=self.device)
        goals = self._model.project_z(self._model._backward_map(train_goal[perm]))
        z = torch.where(mix_idxs == 0, goals, z)
        perm = torch.randperm(self.batch_size, device=self.device)
        z = torch.where(mix_idxs == 1, expert_encodings[perm], z)
        return z

    def update_fb(self, obs, action, discount, next_obs, goal, z, q_loss_coef, clip_grad_norm):
        with torch.no_grad():
            dist = self._model._actor(next_obs, z, self.actor_std)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Fs = self._model._target_forward_map(next_obs, z, next_action)
            target_B = self._model._target_backward_map(goal)
            target_Ms = torch.matmul(target_Fs, target_B.T)
            _, _, target_M = self.get_targets_uncertainty(target_Ms, self.fb_pessimism_penalty)

        Fs = self._model._forward_map(obs, z, action)
        B = self._model._backward_map(goal)
        Ms = torch.matmul(Fs, B.T)

        diff = Ms - discount * target_M
        fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
        fb_loss = fb_offdiag + fb_diag

        Cov = torch.matmul(B, B.T)
        orth_loss_diag = -Cov.diag().mean()
        orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += self.ortho_coef * orth_loss

        q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
        if q_loss_coef is not None:
            with torch.no_grad():
                next_Qs = (target_Fs * z).sum(dim=-1)
                _, _, next_Q = self.get_targets_uncertainty(next_Qs, self.fb_pessimism_penalty)
                cov = torch.matmul(B.T, B) / B.shape[0]
                inv_cov = torch.inverse(cov)
                implicit_reward = (torch.matmul(B, inv_cov) * z).sum(dim=-1)
                target_Q = implicit_reward.detach() + discount.squeeze() * next_Q
                expanded_targets = target_Q.expand(Fs.shape[0], -1)
            Qs = (Fs * z).sum(dim=-1)
            q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)
            fb_loss += q_loss_coef * q_loss

        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._model._backward_map.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        with torch.no_grad():
            return {
                "target_M": target_M.mean(), "M1": Ms[0].mean(), "F1": Fs[0].mean(), "B": B.mean(),
                "B_norm": torch.norm(B, dim=-1).mean(), "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss": fb_loss, "fb_diag": fb_diag, "fb_offdiag": fb_offdiag,
                "orth_loss": orth_loss, "orth_loss_diag": orth_loss_diag, "orth_loss_offdiag": orth_loss_offdiag,
                "q_loss": q_loss,
            }

    @torch.compiler.disable
    def gradient_penalty_wgan(self, real_obs, real_z, fake_obs, fake_z):
        bs = real_obs.shape[0]
        alpha = torch.rand(bs, 1, device=real_obs.device)
        interpolates = torch.cat(
            [(alpha * real_obs + (1 - alpha) * fake_obs).requires_grad_(True),
             (alpha * real_z + (1 - alpha) * fake_z).requires_grad_(True)], dim=1)
        d_interpolates = self._model._discriminator.compute_logits(
            interpolates[:, 0 : real_obs.shape[1]], interpolates[:, real_obs.shape[1] :])
        gradients = autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones_like(d_interpolates),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    def update_discriminator(self, expert_obs, expert_z, train_obs, train_z, grad_penalty):
        expert_logits = self._model._discriminator.compute_logits(obs=expert_obs, z=expert_z)
        unlabeled_logits = self._model._discriminator.compute_logits(obs=train_obs, z=train_z)
        expert_loss = -torch.nn.functional.logsigmoid(expert_logits)
        unlabeled_loss = torch.nn.functional.softplus(unlabeled_logits)
        loss = torch.mean(expert_loss + unlabeled_loss)
        if grad_penalty is not None:
            wgan_gp = self.gradient_penalty_wgan(expert_obs, expert_z, train_obs, train_z)
            loss = loss + grad_penalty * wgan_gp
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_optimizer.step()
        with torch.no_grad():
            out = {"disc_loss": loss.detach(), "disc_expert_loss": expert_loss.detach().mean(),
                   "disc_train_loss": unlabeled_loss.detach().mean()}
            if grad_penalty is not None:
                out["disc_wgan_gp_loss"] = wgan_gp.detach()
        return out

    def update_critic(self, obs, action, discount, next_obs, z):
        num_parallel = self._model.cfg.archi.critic.num_parallel
        with torch.no_grad():
            reward = self._model._discriminator.compute_reward(obs=obs, z=z)
            dist = self._model._actor(next_obs, z, self.actor_std)
            next_action = dist.sample(clip=self.stddev_clip)
            next_Qs = self._model._target_critic(next_obs, z, next_action)
            Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, self.critic_pessimism_penalty)
            target_Q = reward + discount * next_V
            expanded_targets = target_Q.expand(num_parallel, -1, -1)
        Qs = self._model._critic(obs, z, action)
        critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        with torch.no_grad():
            return {"target_Q": target_Q.mean(), "Q1": Qs.mean(), "mean_next_Q": Q_mean.mean(),
                    "unc_Q": Q_unc.mean(), "critic_loss": critic_loss.mean(), "mean_disc_reward": reward.mean()}

    def update_actor(self, obs, action, z, clip_grad_norm):
        dist = self._model._actor(obs, z, self.actor_std)
        action = dist.sample(clip=self.stddev_clip)
        Qs_discriminator = self._model._critic(obs, z, action)
        _, _, Q_discriminator = self.get_targets_uncertainty(Qs_discriminator, self.actor_pessimism_penalty)
        Fs = self._model._forward_map(obs, z, action)
        Qs_fb = (Fs * z).sum(-1)
        _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.actor_pessimism_penalty)
        weight = Q_fb.abs().mean().detach() if self.scale_reg else 1.0
        actor_loss = -Q_discriminator.mean() * self.reg_coeff * weight - Q_fb.mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()
        with torch.no_grad():
            return {"actor_loss": actor_loss.detach(), "Q_discriminator": Q_discriminator.mean(), "Q_fb": Q_fb.mean()}

    def update(self) -> Dict[str, float]:
        """One FB-CPR gradient step (discriminator -> fb -> critic -> actor -> soft updates)."""
        # When the per-network update fns are compiled with cuda-graphs (mode="reduce-overhead"),
        # each gradient step must open a new graph-capture region so the static input/output
        # buffers from the previous step are not aliased into this one. No-op when not compiled
        # or when compiled without cuda-graphs. Mirrors metamotivo FBcprAgent.update.
        if self._cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()

        expert_batch = self.expert_buffer.sample(self.batch_size)
        train_batch = self.replay_buffer.sample(self.batch_size)

        train_obs = train_batch["observation"].to(self.device)
        train_action = train_batch["action"].to(self.device)
        train_next_obs = train_batch["next"]["observation"].to(self.device)
        discount = self.discount * ~train_batch["next"]["terminated"].to(self.device)
        expert_obs = expert_batch["observation"].to(self.device)
        expert_next_obs = expert_batch["next"]["observation"].to(self.device)

        # update + apply obs normalizer (only train data updates running stats)
        self._model._obs_normalizer(train_obs)
        self._model._obs_normalizer(train_next_obs)
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            train_obs, train_next_obs = self._model._obs_normalizer(train_obs), self._model._obs_normalizer(train_next_obs)
            expert_obs, expert_next_obs = self._model._obs_normalizer(expert_obs), self._model._obs_normalizer(expert_next_obs)

        expert_z = self.encode_expert(next_obs=expert_next_obs)
        train_z = train_batch["z"].to(self.device)

        grad_penalty = self.grad_penalty_discriminator if self.grad_penalty_discriminator > 0 else None
        metrics = self.update_discriminator(expert_obs, expert_z, train_obs, train_z, grad_penalty)

        z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
        self.z_buffer.add(z)
        if self.relabel_ratio is not None:
            mask = torch.rand((self.batch_size, 1), device=self.device) <= self.relabel_ratio
            train_z = torch.where(mask, z, train_z)

        q_loss_coef = self.q_loss_coef if self.q_loss_coef > 0 else None
        metrics.update(self.update_fb(train_obs, train_action, discount, train_next_obs, train_next_obs, train_z,
                                      q_loss_coef, self.clip_grad_norm))
        metrics.update(self.update_critic(train_obs, train_action, discount, train_next_obs, train_z))
        metrics.update(self.update_actor(train_obs, train_action, train_z, self.clip_grad_norm))

        with torch.no_grad():
            _soft_update_params(self._forward_map_paramlist, self._target_forward_map_paramlist, self.fb_target_tau)
            _soft_update_params(self._backward_map_paramlist, self._target_backward_map_paramlist, self.fb_target_tau)
            _soft_update_params(self._critic_paramlist, self._target_critic_paramlist, self.critic_target_tau)

        # Batch the GPU->CPU readback: stack all scalar metrics into one tensor and sync once,
        # instead of one .item() (== one device sync) per key. Under cuda-graphs the per-metric
        # syncs would also stall graph replay, so this matters more there. ``torch.stack`` also
        # copies the values out of any cuda-graph static output buffers, making the read safe.
        tensor_items = [(k, v) for k, v in metrics.items() if torch.is_tensor(v)]
        scalar_vals = torch.stack([v.reshape(()) for _, v in tensor_items]).tolist()
        out = {k: float(v) for k, v in metrics.items() if not torch.is_tensor(v)}
        out.update({k: val for (k, _), val in zip(tensor_items, scalar_vals)})
        return out

    # ============================ rsl_rl glue ============================
    def train_mode(self) -> None:
        self._model.train(True)

    def eval_mode(self) -> None:
        self._model.train(False)

    def get_policy(self) -> ForwardBackwardModel:
        return self._model

    def compile(self, mode: str | None = None) -> None:
        """Compile the per-network update functions with ``torch.compile``.

        Mirrors metamotivo ``FBcprAgent.setup_compile``: each of the five hot update
        functions (encode_expert / sample_mixed_z / discriminator / fb / critic / actor) is
        compiled independently rather than compiling ``update()`` as one graph, because the
        gradient-penalty double-backward (``gradient_penalty_wgan``) is graph-breaking and is
        kept eager via ``@torch.compiler.disable``.

        ``mode`` follows rsl_rl's ``torch_compile_mode`` convention:
          * ``None``                          -> no compilation (eager).
          * ``"reduce-overhead"``             -> inductor + cuda-graphs (fastest; needs the
                                                 ``cudagraph_mark_step_begin`` guard in update()).
          * ``"default"`` / ``"max-autotune-no-cudagraphs"`` -> inductor fusion, no cuda-graphs.
        """
        if mode is None:
            return
        self._cudagraphs = mode == "reduce-overhead"
        # fullgraph for the two pure-tensor helpers (no python branching / buffer side effects);
        # the gradient-step fns have an in-graph optimizer.step that inductor handles fine but
        # may contain guarded branches (clip_grad_norm), so they are compiled without fullgraph.
        self.encode_expert = torch.compile(self.encode_expert, mode=mode, fullgraph=True)
        self.sample_mixed_z = torch.compile(self.sample_mixed_z, mode=mode, fullgraph=True)
        self.update_discriminator = torch.compile(self.update_discriminator, mode=mode)
        self.update_fb = torch.compile(self.update_fb, mode=mode)
        self.update_critic = torch.compile(self.update_critic, mode=mode)
        self.update_actor = torch.compile(self.update_actor, mode=mode)

    def save(self) -> dict:
        return {
            "model_state_dict": self._model.state_dict(),
            "forward_optimizer": self.forward_optimizer.state_dict(),
            "backward_optimizer": self.backward_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
        self._model.load_state_dict(loaded_dict["model_state_dict"], strict=strict)
        for key in ["forward_optimizer", "backward_optimizer", "actor_optimizer", "critic_optimizer", "discriminator_optimizer"]:
            if key in loaded_dict:
                getattr(self, key).load_state_dict(loaded_dict[key])
        return False

    def broadcast_parameters(self) -> None:
        pass

    def reduce_parameters(self) -> None:
        pass

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> "FBCPR":
        """rsl_rl-style factory. Expects cfg['model'] (FB model config dict) and an expert
        buffer either in cfg['expert_buffer'] (pre-built) or built by the caller."""
        from rsl_rl.utils import resolve_obs_groups

        cfg["obs_groups"] = resolve_obs_groups(obs, cfg.get("obs_groups", {"actor": ["policy"], "critic": ["policy"]}),
                                               ["actor", "critic"])
        obs_dim = obs[cfg["obs_groups"]["actor"][0]].shape[-1]
        model_cfg = dict(cfg["model"])
        model_cfg["obs_dim"] = obs_dim
        model_cfg["action_dim"] = env.num_actions
        model_cfg["device"] = device
        model = ForwardBackwardModel(**model_cfg)

        replay = ReplayBuffer(capacity=int(cfg.get("replay_buffer_size", 5_000_000)), device=cfg.get("buffer_device", "cpu"))
        expert_buffer = cfg["expert_buffer"]

        alg_cfg = dict(cfg["algorithm"])
        alg_cfg.pop("class_name", None)
        alg = FBCPR(model=model, replay_buffer=replay, expert_buffer=expert_buffer,
                    num_envs=env.num_envs, action_dim=env.num_actions, device=device,
                    multi_gpu_cfg=cfg.get("multi_gpu"), **alg_cfg)
        # Compile the per-network update fns if requested (rsl_rl `torch_compile_mode` convention).
        alg.compile(cfg.get("torch_compile_mode"))
        return alg
