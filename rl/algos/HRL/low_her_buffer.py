from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
import torch as th

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples

from rl.utils import utils


class HiROLowHERReplayBuffer(ReplayBuffer):
    """Low-level replay buffer with HER relabeling at sampling time.

    Stored per transition:
    - segment id and timestep inside segment (for future/final goal sampling)
    - segment start ego state (for intrinsic reward recomputation)
    - external reward r_ext (kept unchanged)

    On sampling:
    - with probability her_ratio, relabel goal in obs/next_obs using a future/final achieved goal
      from the same segment
    - recompute reward = r_ext + r_goal
    """

    _INFO_KEY_SEG_ID = "low_seg_id"
    _INFO_KEY_T_IN_SEG = "low_t_in_seg"
    _INFO_KEY_EGO_START = "low_ego_start"
    _INFO_KEY_EGO_NOW = "low_ego_now"
    _INFO_KEY_EGO_NEXT = "low_ego_next"
    _INFO_KEY_R_EXT = "low_r_ext"

    def __init__(
        self,
        buffer_size: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        device: str | th.device = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        *,
        feat_dim: int,
        kin_flat_dim: int,
        ego_feature_idx: List[int],
        intrinsic_coef: float,
        intrinsic_norm_ranges: np.ndarray,
        intrinsic_weights: np.ndarray | None,
        intrinsic_type: str,
        low_gamma: float,
        her_ratio: float = 0.8,
        her_strategy: str = "future",
        enable_her: bool = True,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

        self.feat_dim = int(feat_dim)
        self.kin_flat_dim = int(kin_flat_dim)
        self.ego_feature_idx = np.asarray(ego_feature_idx, dtype=np.int32).reshape(-1)
        self.ego_dim = int(self.ego_feature_idx.size)

        self.intrinsic_coef = float(intrinsic_coef)
        self.intrinsic_norm_ranges = np.asarray(intrinsic_norm_ranges, dtype=np.float32)
        self.intrinsic_weights = None if intrinsic_weights is None else np.asarray(intrinsic_weights, dtype=np.float32)
        self.intrinsic_type = str(intrinsic_type).lower()
        self.low_gamma = float(low_gamma)

        self.her_ratio = float(her_ratio)
        self.her_strategy = str(her_strategy).lower()
        if self.her_strategy not in {"future", "final"}:
            raise ValueError(f"Unknown her_strategy: {self.her_strategy}")
        self.enable_her = bool(enable_her)

        self.goal_start = int(1 + self.kin_flat_dim)
        self.goal_end = int(self.goal_start + self.ego_dim)

        self._seg_id = np.full((self.buffer_size, self.n_envs), -1, dtype=np.int64)
        self._t_in_seg = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        self._ego_start = np.zeros((self.buffer_size, self.n_envs, self.ego_dim), dtype=np.float32)
        self._ego_now = np.zeros((self.buffer_size, self.n_envs, self.ego_dim), dtype=np.float32)
        self._ego_next = np.zeros((self.buffer_size, self.n_envs, self.ego_dim), dtype=np.float32)
        self._r_ext = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self._seg_index: dict[int, dict[tuple[int, int], int]] = defaultdict(dict)

        self.rng = np.random.default_rng()

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _extract_ego_sub_from_low_obs(self, low_obs: np.ndarray) -> np.ndarray:
        kin_flat = low_obs[..., 1 : 1 + self.kin_flat_dim]
        ego_full = kin_flat[..., : self.feat_dim]
        return ego_full[..., self.ego_feature_idx].astype(np.float32)

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

        for env_i in range(int(self.n_envs)):
            old_seg = int(self._seg_id[pos, env_i])
            if old_seg >= 0:
                old_map = self._seg_index.get(old_seg)
                if old_map is not None:
                    old_map.pop((pos, env_i), None)
                    if len(old_map) == 0:
                        self._seg_index.pop(old_seg, None)

            info = infos[env_i] if env_i < len(infos) else {}
            self._seg_id[pos, env_i] = int(info.get(self._INFO_KEY_SEG_ID, -1))
            self._t_in_seg[pos, env_i] = int(info.get(self._INFO_KEY_T_IN_SEG, 0))

            ego_start = np.asarray(info.get(self._INFO_KEY_EGO_START, np.zeros(self.ego_dim, dtype=np.float32)), dtype=np.float32).reshape(-1)
            if ego_start.size >= self.ego_dim:
                self._ego_start[pos, env_i] = ego_start[: self.ego_dim]
            else:
                self._ego_start[pos, env_i] = 0.0

            ego_now = np.asarray(info.get(self._INFO_KEY_EGO_NOW, self._extract_ego_sub_from_low_obs(obs[env_i : env_i + 1])[0]), dtype=np.float32).reshape(-1)
            if ego_now.size >= self.ego_dim:
                self._ego_now[pos, env_i] = ego_now[: self.ego_dim]
            else:
                self._ego_now[pos, env_i] = 0.0

            ego_next = np.asarray(info.get(self._INFO_KEY_EGO_NEXT, self._extract_ego_sub_from_low_obs(next_obs[env_i : env_i + 1])[0]), dtype=np.float32).reshape(-1)
            if ego_next.size >= self.ego_dim:
                self._ego_next[pos, env_i] = ego_next[: self.ego_dim]
            else:
                self._ego_next[pos, env_i] = 0.0

            self._r_ext[pos, env_i] = float(info.get(self._INFO_KEY_R_EXT, reward[env_i]))

            new_seg = int(self._seg_id[pos, env_i])
            if new_seg >= 0:
                self._seg_index[new_seg][(pos, env_i)] = int(self._t_in_seg[pos, env_i])

    def _sample_batch_indices(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        if self.optimize_memory_usage:
            if self.full:
                batch_inds = (self.rng.integers(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
            else:
                batch_inds = self.rng.integers(0, self.pos, size=batch_size)
        else:
            upper_bound = self.buffer_size if self.full else self.pos
            batch_inds = self.rng.integers(0, upper_bound, size=batch_size)

        env_indices = self.rng.integers(0, self.n_envs, size=batch_size)
        return batch_inds.astype(np.int64), env_indices.astype(np.int64)

    def _get_next_obs_at(self, row: int, col: int) -> np.ndarray:
        if self.optimize_memory_usage:
            return self.observations[(int(row) + 1) % self.buffer_size, int(col)]
        return self.next_observations[int(row), int(col)]

    def _get_future_goal(self, seg_id: int, t_in_seg: int, upper_bound: int) -> np.ndarray | None:
        seg_map = self._seg_index.get(int(seg_id))
        if not seg_map:
            return None

        entries = [(int(t), int(r), int(c)) for (r, c), t in seg_map.items()]
        if len(entries) == 0:
            return None

        if self.her_strategy == "future":
            future_entries = [e for e in entries if e[0] > int(t_in_seg)]
            if len(future_entries) > 0:
                pick = int(self.rng.integers(0, len(future_entries)))
                _, r, c = future_entries[pick]
                return self._ego_next[int(r), int(c)].astype(np.float32, copy=True)

        _, r, c = max(entries, key=lambda x: x[0])
        return self._ego_next[int(r), int(c)].astype(np.float32, copy=True)

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        batch_inds, env_indices = self._sample_batch_indices(batch_size)

        obs = self.observations[batch_inds, env_indices, :]
        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]

        actions = self.actions[batch_inds, env_indices, :]
        dones = (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1)

        if not self.enable_her or self.her_ratio <= 0.0:
            rewards = self.rewards[batch_inds, env_indices].reshape(-1, 1)
            data = (
                self._normalize_obs(obs, env),
                actions,
                self._normalize_obs(next_obs, env),
                dones,
                self._normalize_reward(rewards, env),
            )
            return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

        obs_relabeled = np.array(obs, copy=True)
        next_obs_relabeled = np.array(next_obs, copy=True)

        seg_ids = self._seg_id[batch_inds, env_indices]
        t_in_seg = self._t_in_seg[batch_inds, env_indices]
        upper_bound = self.buffer_size if self.full else self.pos

        her_mask = (self.rng.random(batch_size) < self.her_ratio) & (seg_ids >= 0)
        ego_now_all = self._ego_now[batch_inds, env_indices].astype(np.float32, copy=False)
        ego_next_all = self._ego_next[batch_inds, env_indices].astype(np.float32, copy=False)

        for i in np.flatnonzero(her_mask):
            g_new_abs = self._get_future_goal(int(seg_ids[i]), int(t_in_seg[i]), int(upper_bound))
            if g_new_abs is None:
                continue

            # Keep HER goal semantics aligned with high-level goal_action_to_abs:
            # goal = [x*, y*, vx*, 0], i.e., target vy is always zero.
            if g_new_abs.shape[0] >= 4:
                g_new_abs = np.array(g_new_abs, copy=True)
                g_new_abs[3] = 0.0

            obs_relabeled[i, self.goal_start : self.goal_end] = (g_new_abs - ego_now_all[i]).astype(np.float32)
            next_obs_relabeled[i, self.goal_start : self.goal_end] = (g_new_abs - ego_next_all[i]).astype(np.float32)

        goal_rel_all = obs_relabeled[:, self.goal_start : self.goal_end]
        goal_abs_all = ego_now_all + goal_rel_all

        ego_start_all = self._ego_start[batch_inds, env_indices]
        ego_rel_now = ego_now_all - ego_start_all
        ego_rel_next = ego_next_all - ego_start_all
        goal_rel_seg = goal_abs_all - ego_start_all

        if self.intrinsic_type == "huber_shaping":
            r_goal, _, _, _ = utils.intrinsic_reward_shaping_huber(
                ego_rel_now,
                ego_rel_next,
                goal_rel_seg,
                self.intrinsic_norm_ranges,
                self.intrinsic_coef,
                self.intrinsic_weights,
                gamma=float(self.low_gamma),
                is_terminal=dones.reshape(-1).astype(bool),
            )
        else:
            r_goal = np.zeros(batch_size, dtype=np.float32)
            terminal_mask = dones.reshape(-1).astype(bool)
            if terminal_mask.any():
                r_goal_term, _, _ = utils.intrinsic_reward_l2(
                    ego_rel_next[terminal_mask],
                    goal_rel_seg[terminal_mask],
                    self.intrinsic_norm_ranges,
                    self.intrinsic_coef,
                    self.intrinsic_weights,
                )
                r_goal[terminal_mask] = r_goal_term.astype(np.float32)

        r_ext = self._r_ext[batch_inds, env_indices]
        rewards = (r_ext + r_goal).astype(np.float32).reshape(-1, 1)

        data = (
            self._normalize_obs(obs_relabeled, env),
            actions,
            self._normalize_obs(next_obs_relabeled, env),
            dones,
            self._normalize_reward(rewards, env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
