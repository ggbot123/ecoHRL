from __future__ import annotations

from collections import defaultdict, deque
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
        - with probability her_ratio, relabel goal in obs/next_obs
        - strategy="future" supports three modes:
            * episode_timeaware: sample i in [min_steps, max_steps], use achieved state at
                (same env, same episode, ep_step + i), and sync t_norm
            * segment_timeaware: sample a future achieved state only in current segment,
                and sync t_norm
            * segment_legacy: legacy same-segment future relabeling, do not sync t_norm
        - strategy="final": use final achieved goal from the same segment
        - recompute reward = r_ext + r_goal
    """

    _INFO_KEY_SEG_ID = "low_seg_id"
    _INFO_KEY_T_IN_SEG = "low_t_in_seg"
    _INFO_KEY_EGO_START = "low_ego_start"
    _INFO_KEY_EGO_NOW = "low_ego_now"
    _INFO_KEY_EGO_NEXT = "low_ego_next"
    _INFO_KEY_R_EXT = "low_r_ext"
    _INFO_KEY_SKIP_REPLAY = "skip_replay"
    _INFO_KEY_EP_ID = "low_ep_id"
    _INFO_KEY_EP_STEP = "low_ep_step"
    _INFO_KEY_INTRINSIC_TERMINAL = "low_intrinsic_terminal"

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
        feat_dim: int,
        kin_flat_dim: int,
        obs_extra_dim: int,
        ego_feature_idx: List[int],
        intrinsic_coef: float,
        intrinsic_norm_ranges: np.ndarray,
        intrinsic_weights: np.ndarray | None,
        intrinsic_type: str,
        low_gamma: float,
        high_interval: int,
        fixed_goal_vx: float | None = None,
        her_ratio: float = 0.8,
        her_strategy: str = "future",
        her_future_mode: str | None = None,
        her_episode_timeaware_steps_ahead_range: tuple[int, int] | list[int] | None = None,
        her_future_timeaware: bool = True,
        her_debug_enabled: bool = False,
        her_debug_max_records: int = 20000,
        her_debug_sample_prob: float = 1.0,
        enable_her: bool = True,
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

        self.feat_dim = int(feat_dim)
        self.kin_flat_dim = int(kin_flat_dim)
        self.obs_extra_dim = int(obs_extra_dim)
        self.ego_feature_idx = np.asarray(ego_feature_idx, dtype=np.int32).reshape(-1)
        self.ego_dim = int(self.ego_feature_idx.size)

        self.intrinsic_coef = float(intrinsic_coef)
        self.intrinsic_norm_ranges = np.asarray(intrinsic_norm_ranges, dtype=np.float32)
        self.intrinsic_weights = None if intrinsic_weights is None else np.asarray(intrinsic_weights, dtype=np.float32)
        self.intrinsic_type = str(intrinsic_type).lower()
        self.low_gamma = float(low_gamma)
        self.high_interval = int(high_interval)
        if self.high_interval <= 0:
            raise ValueError(f"high_interval must be positive, got {self.high_interval}")
        self.fixed_goal_vx = None if fixed_goal_vx is None else float(fixed_goal_vx)

        self.her_ratio = float(her_ratio)
        self.her_strategy = str(her_strategy).lower()
        if self.her_strategy not in {"future", "final"}:
            raise ValueError(f"Unknown her_strategy: {self.her_strategy}")

        if her_future_mode is None:
            self.her_future_mode = "episode_timeaware" if bool(her_future_timeaware) else "segment_legacy"
        else:
            self.her_future_mode = str(her_future_mode).lower()
        if self.her_future_mode not in {"episode_timeaware", "segment_timeaware", "segment_legacy"}:
            raise ValueError(f"Unknown her_future_mode: {self.her_future_mode}")

        if her_episode_timeaware_steps_ahead_range is None:
            self.her_episode_timeaware_min_steps_ahead = 1
            self.her_episode_timeaware_max_steps_ahead = int(self.high_interval)
        else:
            if not isinstance(her_episode_timeaware_steps_ahead_range, (tuple, list)) or len(her_episode_timeaware_steps_ahead_range) != 2:
                raise ValueError(
                    "her_episode_timeaware_steps_ahead_range must be None or a 2-item tuple/list like (1, 8)"
                )
            min_steps = int(her_episode_timeaware_steps_ahead_range[0])
            max_steps = int(her_episode_timeaware_steps_ahead_range[1])
            min_steps = max(1, min_steps)
            max_steps = min(int(self.high_interval), max_steps)
            if max_steps < min_steps:
                raise ValueError(
                    f"Invalid her_episode_timeaware_steps_ahead_range after clamping: ({min_steps}, {max_steps})"
                )
            self.her_episode_timeaware_min_steps_ahead = min_steps
            self.her_episode_timeaware_max_steps_ahead = max_steps

        # Keep legacy flag for backward compatibility in configs/checkpoints.
        self.her_future_timeaware = bool(her_future_timeaware)
        self.her_debug_enabled = bool(her_debug_enabled)
        self.her_debug_max_records = int(max(1, her_debug_max_records))
        self.her_debug_sample_prob = float(np.clip(float(her_debug_sample_prob), 0.0, 1.0))
        self.enable_her = bool(enable_her)

        self.goal_start = int(1 + self.kin_flat_dim + self.obs_extra_dim)
        self.goal_end = int(self.goal_start + self.ego_dim)

        self._seg_id = np.full((self.buffer_size, self.n_envs), -1, dtype=np.int64)
        self._t_in_seg = np.zeros((self.buffer_size, self.n_envs), dtype=np.int32)
        self._ego_start = np.zeros((self.buffer_size, self.n_envs, self.ego_dim), dtype=np.float32)
        self._ego_now = np.zeros((self.buffer_size, self.n_envs, self.ego_dim), dtype=np.float32)
        self._ego_next = np.zeros((self.buffer_size, self.n_envs, self.ego_dim), dtype=np.float32)
        self._r_ext = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self._valid = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self._ep_id = np.full((self.buffer_size, self.n_envs), -1, dtype=np.int64)
        self._ep_step = np.full((self.buffer_size, self.n_envs), -1, dtype=np.int64)
        self._intrinsic_terminal = np.zeros((self.buffer_size, self.n_envs), dtype=bool)
        self._seg_index: dict[int, dict[tuple[int, int], int]] = defaultdict(dict)
        self._time_index: dict[tuple[int, int, int], tuple[int, int]] = {}
        self._her_debug_records: deque[dict[str, Any]] = deque(maxlen=self.her_debug_max_records)
        self._training_debug_stats = self._new_training_debug_stats()

        self.rng = np.random.default_rng()

    @staticmethod
    def _new_training_debug_stats() -> dict[str, float]:
        return {
            "sample_calls": 0.0,
            "sampled_transitions": 0.0,
            "her_candidates": 0.0,
            "her_applied": 0.0,
            "skip_missing_metadata": 0.0,
            "skip_goal_lookup": 0.0,
            "skip_start_ref": 0.0,
            "skip_invalid_steps": 0.0,
            "steps_sum": 0.0,
            "steps_sq_sum": 0.0,
            "steps_min": np.inf,
            "steps_max": -np.inf,
            "relabeled_terminal": 0.0,
            "intrinsic_sum": 0.0,
            "intrinsic_sq_sum": 0.0,
            "intrinsic_abs_max": 0.0,
            "external_sum": 0.0,
            "external_sq_sum": 0.0,
            "total_reward_sum": 0.0,
            "total_reward_sq_sum": 0.0,
            "goal_error_l2_sum": 0.0,
            "goal_error_l2_sq_sum": 0.0,
            "her_debug_records_dropped": 0.0,
        }

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

            old_ep_id = int(self._ep_id[pos, env_i])
            old_ep_step = int(self._ep_step[pos, env_i])
            if old_ep_id >= 0 and old_ep_step >= 0:
                old_key = (int(env_i), old_ep_id, old_ep_step)
                loc = self._time_index.get(old_key)
                if loc == (pos, env_i):
                    self._time_index.pop(old_key, None)

            info = infos[env_i] if env_i < len(infos) else {}
            is_valid = not bool(info.get(self._INFO_KEY_SKIP_REPLAY, False))
            self._valid[pos, env_i] = is_valid

            if not is_valid:
                self._seg_id[pos, env_i] = -1
                self._t_in_seg[pos, env_i] = 0
                self._ep_id[pos, env_i] = -1
                self._ep_step[pos, env_i] = -1
                self._ego_start[pos, env_i] = 0.0
                self._ego_now[pos, env_i] = 0.0
                self._ego_next[pos, env_i] = 0.0
                self._r_ext[pos, env_i] = 0.0
                self._intrinsic_terminal[pos, env_i] = False
                continue

            self._seg_id[pos, env_i] = int(info.get(self._INFO_KEY_SEG_ID, -1))
            self._t_in_seg[pos, env_i] = int(info.get(self._INFO_KEY_T_IN_SEG, 0))
            self._ep_id[pos, env_i] = int(info.get(self._INFO_KEY_EP_ID, -1))
            self._ep_step[pos, env_i] = int(info.get(self._INFO_KEY_EP_STEP, -1))

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
            self._intrinsic_terminal[pos, env_i] = bool(
                info.get(self._INFO_KEY_INTRINSIC_TERMINAL, bool(done[env_i]))
            )

            new_seg = int(self._seg_id[pos, env_i])
            if new_seg >= 0:
                self._seg_index[new_seg][(pos, env_i)] = int(self._t_in_seg[pos, env_i])

            new_ep_id = int(self._ep_id[pos, env_i])
            new_ep_step = int(self._ep_step[pos, env_i])
            if new_ep_id >= 0 and new_ep_step >= 0:
                self._time_index[(int(env_i), new_ep_id, new_ep_step)] = (pos, env_i)

    def _sample_batch_indices(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound <= 0:
            raise RuntimeError("Replay buffer is empty")

        valid_mask = np.array(self._valid[:upper_bound], copy=True)
        if self.optimize_memory_usage and self.full:
            valid_mask[int(self.pos), :] = False

        valid_pairs = np.argwhere(valid_mask)
        if valid_pairs.shape[0] == 0:
            raise RuntimeError("Replay buffer has no valid low-level transitions to sample")

        picks = self.rng.integers(0, int(valid_pairs.shape[0]), size=batch_size)
        chosen = valid_pairs[picks]
        batch_inds = chosen[:, 0].astype(np.int64)
        env_indices = chosen[:, 1].astype(np.int64)
        return batch_inds, env_indices

    @staticmethod
    def _to_list(arr: np.ndarray) -> list[float]:
        return np.asarray(arr, dtype=np.float32).reshape(-1).tolist()

    def pop_her_debug_records(self, max_items: int | None = None) -> list[dict[str, Any]]:
        if not self._her_debug_records:
            return []
        if max_items is None or int(max_items) <= 0:
            items = list(self._her_debug_records)
            self._her_debug_records.clear()
            return items

        n = min(int(max_items), len(self._her_debug_records))
        out: list[dict[str, Any]] = []
        for _ in range(n):
            out.append(self._her_debug_records.popleft())
        return out

    def pop_training_debug_stats(self) -> dict[str, float]:
        stats = dict(self._training_debug_stats)
        self._training_debug_stats = self._new_training_debug_stats()
        stats["buffer_size"] = float(self.buffer_size)
        stats["buffer_entries"] = float(self.buffer_size if self.full else self.pos)
        stats["buffer_valid_entries"] = float(np.count_nonzero(self._valid))
        stats["pending_her_debug_records"] = float(len(self._her_debug_records))
        return stats

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
            # A transition at t reaches its achieved next state in one action,
            # so it is already a valid one-step future goal for state s_t.
            future_entries = [e for e in entries if e[0] >= int(t_in_seg)]
            if len(future_entries) > 0:
                pick = int(self.rng.integers(0, len(future_entries)))
                _, r, c = future_entries[pick]
                return self._ego_next[int(r), int(c)].astype(np.float32, copy=True)

        _, r, c = max(entries, key=lambda x: x[0])
        return self._ego_next[int(r), int(c)].astype(np.float32, copy=True)

    def _get_future_goal_with_meta(self, seg_id: int, t_in_seg: int) -> tuple[np.ndarray, int, int, int] | None:
        seg_map = self._seg_index.get(int(seg_id))
        if not seg_map:
            return None

        entries = [(int(t), int(r), int(c)) for (r, c), t in seg_map.items()]
        if len(entries) == 0:
            return None

        if self.her_strategy == "future":
            future_entries = [e for e in entries if e[0] >= int(t_in_seg)]
            if len(future_entries) > 0:
                pick = int(self.rng.integers(0, len(future_entries)))
                t_target, r, c = future_entries[pick]
                steps_ahead = int(t_target - int(t_in_seg) + 1)
                return self._ego_next[int(r), int(c)].astype(np.float32, copy=True), int(r), int(c), steps_ahead

        t_target, r, c = max(entries, key=lambda x: x[0])
        steps_ahead = int(t_target - int(t_in_seg) + 1)
        return self._ego_next[int(r), int(c)].astype(np.float32, copy=True), int(r), int(c), steps_ahead

    def _get_future_goal_in_segment_with_steps(self, seg_id: int, t_in_seg: int) -> tuple[np.ndarray, int, int, int] | None:
        seg_map = self._seg_index.get(int(seg_id))
        if not seg_map:
            return None

        entries = [(int(t), int(r), int(c)) for (r, c), t in seg_map.items()]
        if len(entries) == 0:
            return None

        future_entries = [e for e in entries if e[0] >= int(t_in_seg)]
        if len(future_entries) == 0:
            return None

        pick = int(self.rng.integers(0, len(future_entries)))
        t_target, r, c = future_entries[pick]
        steps_ahead = int(t_target - int(t_in_seg) + 1)
        if steps_ahead <= 0:
            return None

        return self._ego_next[int(r), int(c)].astype(np.float32, copy=True), int(r), int(c), steps_ahead

    def _get_future_goal_by_episode_time(self, env_i: int, ep_id: int, ep_step: int) -> tuple[np.ndarray, int, int, int] | None:
        # Randomly choose the true action distance to the achieved goal.
        # Transition k stores s_k -> s_(k+1), so an H-step goal from s_t
        # is the next state of transition t + H - 1.
        steps_ahead = int(
            self.rng.integers(
                int(self.her_episode_timeaware_min_steps_ahead),
                int(self.her_episode_timeaware_max_steps_ahead) + 1,
            )
        )
        target_transition_step = int(ep_step + steps_ahead - 1)
        target_key = (int(env_i), int(ep_id), target_transition_step)
        loc = self._time_index.get(target_key)
        if loc is None:
            return None

        r, c = int(loc[0]), int(loc[1])
        if c != int(env_i):
            return None
        if int(self._ep_id[r, c]) != int(ep_id) or int(self._ep_step[r, c]) != target_transition_step:
            # Stale index (e.g. ring-buffer overwrite): clean and skip relabel for this sample.
            self._time_index.pop(target_key, None)
            return None

        return self._ego_next[r, c].astype(np.float32, copy=True), int(r), int(c), steps_ahead

    def _get_episode_step_ego_now(self, env_i: int, ep_id: int, ep_step: int) -> tuple[np.ndarray, int, int] | None:
        if int(ep_step) < 0:
            return None

        target_key = (int(env_i), int(ep_id), int(ep_step))
        loc = self._time_index.get(target_key)
        if loc is None:
            return None

        r, c = int(loc[0]), int(loc[1])
        if c != int(env_i):
            return None
        if int(self._ep_id[r, c]) != int(ep_id) or int(self._ep_step[r, c]) != int(ep_step):
            # Stale index (e.g. ring-buffer overwrite): clean and skip for this sample.
            self._time_index.pop(target_key, None)
            return None

        return self._ego_now[r, c].astype(np.float32, copy=True), int(r), int(c)

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
        ep_ids = self._ep_id[batch_inds, env_indices]
        ep_steps = self._ep_step[batch_inds, env_indices]
        upper_bound = self.buffer_size if self.full else self.pos

        her_mask = (self.rng.random(batch_size) < self.her_ratio) & (seg_ids >= 0)
        her_applied_mask = np.zeros(batch_size, dtype=bool)
        applied_steps: list[int] = []
        debug_stats = self._training_debug_stats
        debug_stats["sample_calls"] += 1.0
        debug_stats["sampled_transitions"] += float(batch_size)
        debug_stats["her_candidates"] += float(np.count_nonzero(her_mask))
        ego_now_all = self._ego_now[batch_inds, env_indices].astype(np.float32, copy=False)
        ego_next_all = self._ego_next[batch_inds, env_indices].astype(np.float32, copy=False)
        ego_start_ref_all = self._ego_start[batch_inds, env_indices].astype(np.float32, copy=True)
        source_done_mask = dones.reshape(-1).astype(bool)
        relabeled_done_mask = np.array(source_done_mask, copy=True)
        source_intrinsic_terminal_mask = (
            self._intrinsic_terminal[batch_inds, env_indices].astype(bool, copy=False)
            & source_done_mask
        )
        intrinsic_terminal_mask = np.array(source_intrinsic_terminal_mask, copy=True)
        relabel_debug_pending: list[dict[str, Any]] = []

        for i in np.flatnonzero(her_mask):
            relabeled_t_norm = None
            goal_row = -1
            goal_col = -1
            steps_ahead = -1
            start_ref_row = -1
            start_ref_col = -1
            start_ref_step = -1
            if self.her_strategy == "future":
                if self.her_future_mode == "episode_timeaware":
                    if int(ep_ids[i]) < 0 or int(ep_steps[i]) < 0:
                        debug_stats["skip_missing_metadata"] += 1.0
                        continue
                    goal_with_steps = self._get_future_goal_by_episode_time(
                        int(env_indices[i]),
                        int(ep_ids[i]),
                        int(ep_steps[i]),
                    )
                    if goal_with_steps is None:
                        debug_stats["skip_goal_lookup"] += 1.0
                        continue
                    g_new_abs, goal_row, goal_col, i_steps = goal_with_steps
                    steps_ahead = int(i_steps)

                    # Time-aware relabel should use a virtual segment start that is H steps before
                    # the relabeled goal step, so shaping remains consistent with H-step semantics.
                    start_ref_step = int(ep_steps[i]) + steps_ahead - int(self.high_interval)
                    start_ref = self._get_episode_step_ego_now(
                        int(env_indices[i]),
                        int(ep_ids[i]),
                        int(start_ref_step),
                    )
                    if start_ref is None:
                        debug_stats["skip_start_ref"] += 1.0
                        continue
                    ego_start_ref, start_ref_row, start_ref_col = start_ref
                    ego_start_ref_all[i] = ego_start_ref

                    c_relabel = self.high_interval - int(steps_ahead)
                    relabeled_t_norm = float(c_relabel) / float(self.high_interval)
                elif self.her_future_mode == "segment_timeaware":
                    goal_with_steps = self._get_future_goal_in_segment_with_steps(
                        int(seg_ids[i]),
                        int(t_in_seg[i]),
                    )
                    if goal_with_steps is None:
                        debug_stats["skip_goal_lookup"] += 1.0
                        continue
                    g_new_abs, goal_row, goal_col, i_steps = goal_with_steps
                    steps_ahead = int(i_steps)
                    c_relabel = self.high_interval - int(steps_ahead)
                    relabeled_t_norm = float(c_relabel) / float(self.high_interval)
                else:
                    # Legacy behavior: sample achieved goal from future steps in the same segment.
                    goal_meta = self._get_future_goal_with_meta(int(seg_ids[i]), int(t_in_seg[i]))
                    if goal_meta is None:
                        debug_stats["skip_goal_lookup"] += 1.0
                        continue
                    g_new_abs, goal_row, goal_col, steps_ahead = goal_meta
            else:
                goal_meta = self._get_future_goal_with_meta(int(seg_ids[i]), int(t_in_seg[i]))
                if goal_meta is None:
                    debug_stats["skip_goal_lookup"] += 1.0
                    continue
                g_new_abs, goal_row, goal_col, steps_ahead = goal_meta
            if g_new_abs is None:
                debug_stats["skip_goal_lookup"] += 1.0
                continue
            if int(steps_ahead) < 1:
                debug_stats["skip_invalid_steps"] += 1.0
                continue

            # Keep HER goal semantics aligned with high-level goal_action_to_abs:
            # goal = [x*, y*, vx*, 0], i.e., target vy is always zero.
            if g_new_abs.shape[0] >= 4:
                g_new_abs = np.array(g_new_abs, copy=True)
                if self.fixed_goal_vx is not None and g_new_abs.shape[0] >= 3:
                    g_new_abs[2] = float(self.fixed_goal_vx)
                g_new_abs[3] = 0.0

            obs_relabeled[i, self.goal_start : self.goal_end] = (g_new_abs - ego_now_all[i]).astype(np.float32)
            next_obs_relabeled[i, self.goal_start : self.goal_end] = (g_new_abs - ego_next_all[i]).astype(np.float32)
            relabeled_done_mask[i] = (int(steps_ahead) == 1)
            intrinsic_terminal_mask[i] = bool(relabeled_done_mask[i])
            her_applied_mask[i] = True
            applied_steps.append(int(steps_ahead))
            if relabeled_t_norm is not None:
                obs_relabeled[i, 0] = np.float32(np.clip(relabeled_t_norm, 0.0, 1.0))
                next_obs_relabeled[i, 0] = np.float32(np.clip(relabeled_t_norm + 1.0 / float(self.high_interval), 0.0, 1.0))

            if (
                self.her_debug_enabled
                and goal_row >= 0
                and goal_col >= 0
                and self.rng.random() < self.her_debug_sample_prob
            ):
                src_row = int(batch_inds[i])
                src_col = int(env_indices[i])
                relabel_debug_pending.append(
                    {
                        "batch_i": int(i),
                        "source_row": src_row,
                        "source_col": src_col,
                        "source_seg_id": int(self._seg_id[src_row, src_col]),
                        "source_t_in_seg": int(self._t_in_seg[src_row, src_col]),
                        "source_ep_id": int(self._ep_id[src_row, src_col]),
                        "source_ep_step": int(self._ep_step[src_row, src_col]),
                        "source_obs": self._to_list(obs[i]),
                        "source_next_obs": self._to_list(next_obs[i]),
                        "source_action": self._to_list(actions[i]),
                        "source_ego_abs": self._to_list(self._ego_now[src_row, src_col]),
                        "source_ego_next_abs": self._to_list(self._ego_next[src_row, src_col]),
                        "source_reward_stored": float(self.rewards[src_row, src_col]),
                        "source_done": bool(source_done_mask[i]),
                        "source_intrinsic_terminal": bool(source_intrinsic_terminal_mask[i]),
                        "relabeled_done": bool(relabeled_done_mask[i]),
                        "relabeled_intrinsic_terminal": bool(intrinsic_terminal_mask[i]),
                        "goal_row": int(goal_row),
                        "goal_col": int(goal_col),
                        "goal_seg_id": int(self._seg_id[int(goal_row), int(goal_col)]),
                        "goal_t_in_seg": int(self._t_in_seg[int(goal_row), int(goal_col)]),
                        "goal_ep_id": int(self._ep_id[int(goal_row), int(goal_col)]),
                        "goal_ep_step": int(self._ep_step[int(goal_row), int(goal_col)]),
                        "goal_obs": self._to_list(self.observations[int(goal_row), int(goal_col), :]),
                        "goal_next_obs": self._to_list(self._get_next_obs_at(int(goal_row), int(goal_col))),
                        "goal_action": self._to_list(self.actions[int(goal_row), int(goal_col), :]),
                        "goal_transition_ego_abs": self._to_list(self._ego_now[int(goal_row), int(goal_col)]),
                        "goal_ego_abs": self._to_list(g_new_abs),
                        "goal_reward_stored": float(self.rewards[int(goal_row), int(goal_col)]),
                        "start_ref_row": int(start_ref_row),
                        "start_ref_col": int(start_ref_col),
                        "start_ref_ep_step": int(start_ref_step),
                        "start_ref_ego_abs": self._to_list(ego_start_ref_all[i]),
                        "steps_ahead": int(steps_ahead),
                        "relabeled_obs": self._to_list(obs_relabeled[i]),
                        "relabeled_next_obs": self._to_list(next_obs_relabeled[i]),
                        "relabeled_goal_rel": self._to_list(obs_relabeled[i, self.goal_start : self.goal_end]),
                        "relabeled_ego_abs": self._to_list(ego_now_all[i]),
                        "relabel_t_norm_obs": float(obs_relabeled[i, 0]),
                        "relabel_t_norm_next": float(next_obs_relabeled[i, 0]),
                        "her_strategy": str(self.her_strategy),
                        "her_future_mode": str(self.her_future_mode),
                    }
                )

        goal_rel_all = obs_relabeled[:, self.goal_start : self.goal_end]
        goal_abs_all = ego_now_all + goal_rel_all

        ego_rel_now = ego_now_all - ego_start_ref_all
        ego_rel_next = ego_next_all - ego_start_ref_all
        goal_rel_seg = goal_abs_all - ego_start_ref_all

        if self.intrinsic_type == "huber_shaping":
            r_goal, _, _, _ = utils.intrinsic_reward_shaping_huber(
                ego_rel_now,
                ego_rel_next,
                goal_rel_seg,
                self.intrinsic_norm_ranges,
                self.intrinsic_coef,
                self.intrinsic_weights,
                gamma=float(self.low_gamma),
                is_terminal=intrinsic_terminal_mask,
            )
        else:
            r_goal = np.zeros(batch_size, dtype=np.float32)
            terminal_mask = intrinsic_terminal_mask
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

        applied_count = int(np.count_nonzero(her_applied_mask))
        debug_stats["her_applied"] += float(applied_count)
        if applied_steps:
            steps_np = np.asarray(applied_steps, dtype=np.float32)
            debug_stats["steps_sum"] += float(np.sum(steps_np))
            debug_stats["steps_sq_sum"] += float(np.sum(steps_np * steps_np))
            debug_stats["steps_min"] = min(float(debug_stats["steps_min"]), float(np.min(steps_np)))
            debug_stats["steps_max"] = max(float(debug_stats["steps_max"]), float(np.max(steps_np)))
        if applied_count > 0:
            applied_intrinsic = np.asarray(r_goal[her_applied_mask], dtype=np.float32)
            applied_external = np.asarray(r_ext[her_applied_mask], dtype=np.float32)
            applied_total = np.asarray(rewards.reshape(-1)[her_applied_mask], dtype=np.float32)
            goal_error_l2 = np.linalg.norm(
                goal_abs_all[her_applied_mask] - ego_next_all[her_applied_mask],
                axis=1,
            ).astype(np.float32)
            debug_stats["relabeled_terminal"] += float(np.count_nonzero(relabeled_done_mask & her_applied_mask))
            debug_stats["intrinsic_sum"] += float(np.sum(applied_intrinsic))
            debug_stats["intrinsic_sq_sum"] += float(np.sum(applied_intrinsic * applied_intrinsic))
            debug_stats["intrinsic_abs_max"] = max(
                float(debug_stats["intrinsic_abs_max"]),
                float(np.max(np.abs(applied_intrinsic))),
            )
            debug_stats["external_sum"] += float(np.sum(applied_external))
            debug_stats["external_sq_sum"] += float(np.sum(applied_external * applied_external))
            debug_stats["total_reward_sum"] += float(np.sum(applied_total))
            debug_stats["total_reward_sq_sum"] += float(np.sum(applied_total * applied_total))
            debug_stats["goal_error_l2_sum"] += float(np.sum(goal_error_l2))
            debug_stats["goal_error_l2_sq_sum"] += float(np.sum(goal_error_l2 * goal_error_l2))

        if self.her_debug_enabled and relabel_debug_pending:
            for rec in relabel_debug_pending:
                bi = int(rec["batch_i"])
                rec["intrinsic_reward_new"] = float(r_goal[bi])
                rec["reward_ext"] = float(r_ext[bi])
                rec["reward_total_new"] = float(rewards[bi, 0])
                if len(self._her_debug_records) >= self._her_debug_records.maxlen:
                    debug_stats["her_debug_records_dropped"] += 1.0
                self._her_debug_records.append(rec)

        dones_relabeled = relabeled_done_mask.astype(dones.dtype, copy=False).reshape(-1, 1)

        data = (
            self._normalize_obs(obs_relabeled, env),
            actions,
            self._normalize_obs(next_obs_relabeled, env),
            dones_relabeled,
            self._normalize_reward(rewards, env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
