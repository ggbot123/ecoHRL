# rl/algos/hiro/hiro_high_replay_buffer.py
from __future__ import annotations

from typing import Dict, Any, List
import gymnasium as gym
import numpy as np
import torch as th

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from rl.utils import utils


class HiROHighReplayBuffer(ReplayBuffer):
    """ReplayBuffer with HiRO off-policy correction for the *high-level* policy.

    This buffer stores, for each high-level transition, the low-level trajectory
    (low-level observations and low-level actions) over the high-level interval.

    When sampling, it performs the HIRO off-policy correction by re-labeling the
    high-level action (goal) to maximize the likelihood of the recorded low-level
    actions under the current low-level policy.

    Notes
    -----
    * This implementation follows HiRO (Nachum et al., NeurIPS 2018) Section 3.3:
      evaluate candidate goals: the original goal, the achieved delta-goal, and
      Gaussian samples around the achieved goal; select the goal that maximizes
      the (approximate) log-likelihood of the low-level actions.
    * SB3 stores actions in replay buffers in the *scaled* [-1, 1] space; this
      buffer therefore expects/returns scaled actions for the high-level policy.
    """

    # Keys used in infos dict passed to ReplayBuffer.add()
    _INFO_KEY_SEQ_LEN = "high_interval_len"
    _INFO_KEY_DISCOUNT = "high_transition_discount"
    _INFO_KEY_LOW_OBS_SEQ = "opc_low_obs_seq"
    _INFO_KEY_ACT_SEQ = "opc_low_act_seq"

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: str | th.device = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = False,
        *,
        max_seq_len: int,
        kin_flat_dim: int,
        obs_extra_dim: int,
        low_action_dim: int,
        feat_dim: int,
        ego_feature_idx: List[int],
        lane_center_ys: np.ndarray,
        low_policy=None,
        high_interval: int,
        n_candidates: int = 10,
        noise_std: float = 0.5,
        enable_off_policy_correction: bool = True,
        dynamic_feasible_lane_intervals: bool = False,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=False,
        )

        # This implementation is only used with a 1-env dummy vec env for the high-level policy.
        if int(n_envs) != 1:
            raise ValueError(f"HiROHighReplayBuffer only supports n_envs=1, got n_envs={n_envs}.")

        self.max_seq_len = int(max_seq_len)
        self.kin_flat_dim = int(kin_flat_dim)
        self.obs_extra_dim = int(obs_extra_dim)
        self.low_action_dim = int(low_action_dim)
        self.feat_dim = int(feat_dim)
        self.ego_feature_idx = np.asarray(ego_feature_idx, dtype=int).reshape(-1)
        self.ego_dim = int(self.ego_feature_idx.size)
        self.lane_center_ys = np.asarray(lane_center_ys, dtype=np.float32).reshape(-1)
        self.low_policy = low_policy
        self.high_interval = int(high_interval)
        self.n_candidates = int(n_candidates)
        self.noise_std = float(noise_std)
        self.enable_off_policy_correction = bool(enable_off_policy_correction)
        self.dynamic_feasible_lane_intervals = bool(dynamic_feasible_lane_intervals)
        self.low_obs_dim = int(1 + self.kin_flat_dim + self.obs_extra_dim + self.ego_dim)

        # [buffer_size, max_seq_len, ...]
        self._opc_low_obs = np.zeros((self.buffer_size, self.max_seq_len, self.low_obs_dim), dtype=np.float32)
        self._opc_low_actions = np.zeros((self.buffer_size, self.max_seq_len, self.low_action_dim), dtype=np.float32)
        self._opc_seq_len = np.zeros((self.buffer_size,), dtype=np.int32)
        self._debug_env_id = np.full((self.buffer_size,), -1, dtype=np.int32)
        self._debug_global_step = np.full((self.buffer_size,), -1, dtype=np.int64)
        self._debug_segment_id = np.full((self.buffer_size,), -1, dtype=np.int64)
        self._debug_interval_len = np.zeros((self.buffer_size,), dtype=np.int32)
        self._transition_discounts = np.ones((self.buffer_size,), dtype=np.float32)
        self._debug_comp_names = [
            "collision_reward",
            "progress_reward",
            "speed_ref_aux_reward",
            "comfort_reward_for_high",
            "lane_change_reward",
            "goal_lane_dense_reward",
            "punctual_reward",
            "wrong_lane_terminal_penalty",
        ]
        self._debug_components = {
            name: np.zeros((self.buffer_size,), dtype=np.float32)
            for name in self._debug_comp_names
        }
        self._debug_acc_stats = {
            "acc_min": np.zeros((self.buffer_size,), dtype=np.float32),
            "acc_max": np.zeros((self.buffer_size,), dtype=np.float32),
            "acc_mean": np.zeros((self.buffer_size,), dtype=np.float32),
            "acc_abs_mean": np.zeros((self.buffer_size,), dtype=np.float32),
            "hard_brake_frac": np.zeros((self.buffer_size,), dtype=np.float32),
            "hard_accel_frac": np.zeros((self.buffer_size,), dtype=np.float32),
        }
        self.last_sample_debug: Dict[str, Any] | None = None

        # Cache action bounds for faster clipping
        self._high_low = np.asarray(self.action_space.low, dtype=np.float32).reshape(1, 1, -1)
        self._high_high = np.asarray(self.action_space.high, dtype=np.float32).reshape(1, 1, -1)
        
        # RNG for OPC noise
        self.rng = np.random.default_rng()

    def set_seed(self, seed: int) -> None:
        """Set the random seed for the buffer's RNG."""
        self.rng = np.random.default_rng(seed)

    # ------------------------- storing extra OPC data -------------------------
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        pos = int(self.pos)
        super().add(obs, next_obs, action, reward, done, infos)

        # Extras are provided through infos[0]
        info = infos[0] if isinstance(infos, list) and len(infos) > 0 else {}
        self._transition_discounts[pos] = float(info.get(self._INFO_KEY_DISCOUNT, 1.0))
        self._store_debug_info(pos, info)
        seq_len = int(info.get(self._INFO_KEY_SEQ_LEN, 0))
        seq_len = max(0, min(seq_len, self.max_seq_len))

        low_obs_seq = info.get(self._INFO_KEY_LOW_OBS_SEQ)
        act_seq = info.get(self._INFO_KEY_ACT_SEQ)

        if act_seq is None or seq_len == 0 or low_obs_seq is None:
            # leave as zeros
            self._opc_seq_len[pos] = 0
            return

        low_obs_seq = np.asarray(low_obs_seq, dtype=np.float32)
        act_seq = np.asarray(act_seq, dtype=np.float32)

        L = int(min(seq_len, low_obs_seq.shape[0], act_seq.shape[0]))
        if L <= 0:
            self._opc_seq_len[pos] = 0
            return

        # Pad/truncate into fixed-size arrays
        self._opc_low_obs[pos, :L] = low_obs_seq[:L, : self.low_obs_dim]
        self._opc_low_actions[pos, :L] = act_seq[:L]
        self._opc_seq_len[pos] = L

        if L < self.max_seq_len:
            self._opc_low_obs[pos, L:] = 0.0
            self._opc_low_actions[pos, L:] = 0.0

    def _store_debug_info(self, pos: int, info: Dict[str, Any]) -> None:
        self._debug_env_id[pos] = int(info.get("high_env_id", -1))
        self._debug_global_step[pos] = int(info.get("high_global_step", -1))
        self._debug_segment_id[pos] = int(info.get("high_segment_id", -1))
        self._debug_interval_len[pos] = int(info.get("high_interval_len", 0))

        comp = info.get("high_components", {})
        if not isinstance(comp, dict):
            comp = {}
        for name in self._debug_comp_names:
            self._debug_components[name][pos] = float(comp.get(name, 0.0))

        acc = info.get("high_acc_stats", {})
        if not isinstance(acc, dict):
            acc = {}
        for name, arr in self._debug_acc_stats.items():
            arr[pos] = float(acc.get(name, 0.0))

    def _build_sample_debug(
        self,
        batch_inds: np.ndarray,
        original_actions: th.Tensor,
        returned_actions: th.Tensor,
    ) -> Dict[str, Any]:
        inds = np.asarray(batch_inds, dtype=np.int64).reshape(-1)
        debug: Dict[str, Any] = {
            "batch_inds": inds.copy(),
            "env_id": self._debug_env_id[inds].copy(),
            "global_step": self._debug_global_step[inds].copy(),
            "segment_id": self._debug_segment_id[inds].copy(),
            "interval_len": self._debug_interval_len[inds].copy(),
            "original_actions": original_actions.detach().cpu().numpy().astype(np.float32),
            "returned_actions": returned_actions.detach().cpu().numpy().astype(np.float32),
        }
        for name, arr in self._debug_components.items():
            debug[f"comp_{name}"] = arr[inds].copy()
        for name, arr in self._debug_acc_stats.items():
            debug[name] = arr[inds].copy()
        return debug

    # ------------------------- low-level deterministic action -------------------------
    def _low_policy_deterministic(self, obs_tensor: th.Tensor, obs_np: np.ndarray | None = None) -> th.Tensor:
        """Return deterministic low-level action in *scaled* space, matching SB3 replay format."""
        if self.low_policy is None:
            raise RuntimeError("HiROHighReplayBuffer.low_policy is None; cannot run off-policy correction.")

        # Prefer direct actor forward pass (fast, returns scaled actions).
        actor = getattr(self.low_policy, "actor", None)
        if actor is not None:
            # Ensure observations are on the same device as the low-level actor
            try:
                actor_device = next(actor.parameters()).device
                obs_tensor = obs_tensor.to(actor_device)
            except Exception:
                pass

            with th.no_grad():
                # SB3 SAC Actor.forward(obs, deterministic=bool) exists in common versions
                try:
                    return actor(obs_tensor, deterministic=True)
                except TypeError:
                    try:
                        return actor.forward(obs_tensor, deterministic=True)
                    except TypeError:
                        pass  # fallback below

        # Fallback: use policy.predict (returns env actions), then scale back.
        if obs_np is None:
            obs_np = obs_tensor.detach().cpu().numpy()
        act_env, _ = self.low_policy.predict(obs_np, deterministic=True)
        if hasattr(self.low_policy, "scale_action"):
            act_scaled = self.low_policy.scale_action(act_env)
        else:
            # Last resort: assume already scaled
            act_scaled = act_env
        return th.as_tensor(act_scaled, device=obs_tensor.device, dtype=obs_tensor.dtype)

    # ------------------------- off-policy correction -------------------------
    def _apply_off_policy_correction(self, batch_inds: np.ndarray, samples: ReplayBufferSamples) -> th.Tensor:
        """Return corrected high-level actions (scaled) for the sampled batch."""
        if (not self.enable_off_policy_correction) or (self.low_policy is None):
            return samples.actions

        # Tensors from SB3 samples
        obs_t = samples.observations
        next_obs_t = samples.next_observations
        act_t = samples.actions

        batch_size = int(batch_inds.shape[0])
        act_dim = int(self.action_space.shape[0])
        n_cand = int(self.n_candidates)
        max_L = int(self.max_seq_len)

        # Convert high-level (start/end) observations and stored actions to numpy
        act_np = act_t.detach().cpu().numpy().reshape(batch_size, act_dim).astype(np.float32)
        obs_np = obs_t.detach().cpu().numpy().reshape(batch_size, -1).astype(np.float32)
        next_np = next_obs_t.detach().cpu().numpy().reshape(batch_size, -1).astype(np.float32)

        # --- achieved goal in env-space (batch, act_dim) ---
        # high_obs layout is assumed: [t, kinematics_flat, ...]; we only need ego state from the start/end.
        kin_flat_start = obs_np[:, 1 : 1 + self.kin_flat_dim]
        kin_flat_end = next_np[:, 1 : 1 + self.kin_flat_dim]

        ego0_full = kin_flat_start[:, : self.feat_dim]
        ego1_full = kin_flat_end[:, : self.feat_dim]
        ego0 = ego0_full[:, self.ego_feature_idx]  # (B, ego_dim=[x,y,vx,vy])
        ego1 = ego1_full[:, self.ego_feature_idx]

        dx = (ego1[:, 0] - ego0[:, 0]).astype(np.float32)
        vx_target = ego1[:, 2].astype(np.float32)

        lanes = self.lane_center_ys.reshape(1, -1)  # (1, n_lanes)
        y0 = ego0[:, 1].reshape(-1, 1)
        y1 = ego1[:, 1].reshape(-1, 1)
        k0 = np.argmin(np.abs(y0 - lanes), axis=1)
        k1 = np.argmin(np.abs(y1 - lanes), axis=1)
        y_code = np.where(k1 > k0, 1.0, np.where(k1 < k0, -1.0, 0.0)).astype(np.float32)

        achieved_env = np.stack([dx, y_code, vx_target], axis=1).astype(np.float32)

        # --- build candidate set in scaled space: (B, n_candidates, act_dim) ---
        cand_scaled = np.empty((batch_size, n_cand, act_dim), dtype=np.float32)
        cand_scaled[:, 0, :] = np.clip(act_np, -1.0, 1.0)

        if n_cand >= 2:
            achieved_scaled = utils.scale_action(achieved_env, self.action_space).astype(np.float32)
            cand_scaled[:, 1, :] = np.clip(achieved_scaled, -1.0, 1.0)

        if n_cand > 2:
            noise = self.rng.normal(loc=0.0, scale=self.noise_std, size=(batch_size, n_cand - 2, act_dim)).astype(np.float32)
            cand_scaled[:, 2:, :] = cand_scaled[:, 1:2, :] + noise

        cand_scaled = np.clip(cand_scaled, -1.0, 1.0)

        # --- convert candidate goals to env space: (B, n_candidates, act_dim) ---
        cand_env = utils.unscale_action(cand_scaled, self.action_space).astype(np.float32)
        cand_env = np.clip(cand_env, self._high_low, self._high_high)

        # --- absolute goal state (x*, y*, vx*, 0) in physical coordinates: (B, n_candidates, ego_dim) ---
        ego0_rep = np.repeat(ego0, repeats=n_cand, axis=0)  # (B*n_cand, ego_dim)
        cand_env_flat = cand_env.reshape(batch_size * n_cand, act_dim)
        goal_phys_flat = utils.goal_action_to_abs(
            ego0_rep,
            cand_env_flat,
            self.lane_center_ys,
            dynamic_feasible_intervals=self.dynamic_feasible_lane_intervals,
        ).astype(np.float32)
        goal_phys = goal_phys_flat.reshape(batch_size, n_cand, self.ego_dim)

        # --- load stored low-level sequences for these transitions ---
        low_obs_seq = self._opc_low_obs[batch_inds.astype(int)]          # (B, max_L, low_obs_dim)
        low_act_seq = self._opc_low_actions[batch_inds.astype(int)]      # (B, max_L, low_action_dim)
        seq_len = self._opc_seq_len[batch_inds.astype(int)].astype(int)  # (B,)

        # mask padded timesteps
        t_idx = np.arange(max_L, dtype=np.int32).reshape(1, max_L)
        mask = (t_idx < seq_len.reshape(-1, 1))  # (B, max_L) bool

        # Split low_obs into components
        t_norm = low_obs_seq[:, :, 0:1]  # (B, max_L, 1)
        kin_seq = low_obs_seq[:, :, 1 : 1 + self.kin_flat_dim]  # (B, max_L, kin_flat_dim)
        extra_seq = low_obs_seq[:, :, 1 + self.kin_flat_dim : 1 + self.kin_flat_dim + self.obs_extra_dim]

        # Ego substate at each low-level step, extracted from kin_seq
        ego_full_seq = kin_seq[:, :, : self.feat_dim]  # (B, max_L, feat_dim)
        ego_seq = ego_full_seq[:, :, self.ego_feature_idx]  # (B, max_L, ego_dim)

        # goal_rel for each candidate
        goal_rel = (goal_phys[:, :, None, :] - ego_seq[:, None, :, :]).astype(np.float32)  # (B, n_cand, max_L, ego_dim)

        # Build candidate low-level obs: (B, n_cand, max_L, low_obs_dim)
        t_rep = np.broadcast_to(t_norm[:, None, :, :], (batch_size, n_cand, max_L, 1))
        kin_rep = np.broadcast_to(kin_seq[:, None, :, :], (batch_size, n_cand, max_L, self.kin_flat_dim))
        extra_rep = np.broadcast_to(extra_seq[:, None, :, :], (batch_size, n_cand, max_L, self.obs_extra_dim))
        low_obs_all = np.concatenate([t_rep, kin_rep, extra_rep, goal_rel], axis=-1).astype(np.float32)

        low_obs_all_flat = low_obs_all.reshape(batch_size * n_cand * max_L, self.low_obs_dim)

        # --- evaluate likelihood under current low-level policy ---
        obs_tensor = th.as_tensor(low_obs_all_flat, device=self.device, dtype=th.float32)
        pred_low = self._low_policy_deterministic(obs_tensor, obs_np=low_obs_all_flat)  # (B*n_cand*max_L, low_action_dim)
        pred_low = pred_low.reshape(batch_size, n_cand, max_L, self.low_action_dim)

        act_true = th.as_tensor(low_act_seq, device=pred_low.device, dtype=pred_low.dtype).unsqueeze(1)  # (B,1,max_L,low_action_dim)
        mask_t = th.as_tensor(mask, device=pred_low.device, dtype=pred_low.dtype).unsqueeze(1).unsqueeze(-1)  # (B,1,max_L,1)

        mse = th.sum(((pred_low - act_true) ** 2) * mask_t, dim=(2, 3))  # (B, n_cand)
        best_idx = th.argmin(mse, dim=1).detach().cpu().numpy()  # (B,)

        best_scaled = cand_scaled[np.arange(batch_size), best_idx].astype(np.float32)  # (B, act_dim)
        return th.as_tensor(best_scaled, device=act_t.device, dtype=act_t.dtype)

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        # Mirror SB3 ReplayBuffer.sample() but keep batch indices for OPC.
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = self.rng.integers(0, upper_bound, size=batch_size)
        samples = self._get_samples(batch_inds, env=env)
        discounts = self.to_torch(
            self._transition_discounts[batch_inds].reshape(-1, 1)
        )

        if not self.enable_off_policy_correction:
            self.last_sample_debug = self._build_sample_debug(batch_inds, samples.actions, samples.actions)
            return ReplayBufferSamples(
                observations=samples.observations,
                actions=samples.actions,
                next_observations=samples.next_observations,
                dones=samples.dones,
                rewards=samples.rewards,
                discounts=discounts,
            )

        corrected_actions = self._apply_off_policy_correction(batch_inds, samples)
        self.last_sample_debug = self._build_sample_debug(batch_inds, samples.actions, corrected_actions)

        return ReplayBufferSamples(
            observations=samples.observations,
            actions=corrected_actions,
            next_observations=samples.next_observations,
            dones=samples.dones,
            rewards=samples.rewards,
            discounts=discounts,
        )
