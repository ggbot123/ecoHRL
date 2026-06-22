# rl/algos/hiro/hiro_callbacks.py
from __future__ import annotations

import os
import csv
import json
from collections import deque
from typing import Any, Dict, List
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from rl.utils import utils


class HIROLoggingCallback(BaseCallback):
    """Unified logging callback for HIRO.

    - TensorBoard:
        - High-level: logs true env episode stats aggregated across envs.
        - Low-level: logs low-episode (high_interval) stats.
    - CSV (optional):
        - Logs high-level and low-level trajectories for env 0 periodically.
        - Logs low_obs at each high-interval start (configurable interval sampling).
        - Logs high-interval debug snapshots (state/goal/safe-bounds) for plotting.
    """

    def __init__(
        self,
        high_log_interval_episodes: int = 1,
        low_log_interval_hi: int = 1,
        verbose: int = 0,
        # CSV Logging args
        csv_log_freq_episodes: int = 0,  # 0 to disable
        csv_save_dir: str | None = None,
        mpc_fail_csv_path: str | None = None,
        # Low-obs snapshot CSV (high-interval starts)
        low_obs_csv_interval_hi: int = 1,
        low_obs_csv_env0_only: bool = True,
        # Record env0 full high-episode every k episodes (k<=0 disables)
        high_transition_csv_all: int = 1,
        high_transition_csv_envs: str = "env0",
        low_transition_detail_csv: bool = False,
        low_transition_detail_envs: str = "env0",
        # HER relabel debug CSV
        her_debug_csv_interval_steps: int = 0,
        her_debug_csv_max_rows_per_flush: int = 200,
        low_debug_summary_interval_steps: int = 10000,
        low_debug_env_step_interval_steps: int = 1000,
    ):
        super().__init__(verbose)
        self.high_log_interval_episodes = int(high_log_interval_episodes)
        self.low_log_interval_hi = int(low_log_interval_hi)
        
        self.csv_log_freq = int(csv_log_freq_episodes)
        self.csv_save_dir = csv_save_dir
        self.mpc_fail_csv_path = mpc_fail_csv_path
        self.low_obs_csv_interval_hi = max(1, int(low_obs_csv_interval_hi))
        self.low_obs_csv_env0_only = bool(low_obs_csv_env0_only)
        self.high_transition_csv_all = max(0, int(high_transition_csv_all))
        self.high_transition_csv_envs = str(high_transition_csv_envs).strip().lower()
        if self.high_transition_csv_envs not in {"env0", "all"}:
            raise ValueError("high_transition_csv_envs must be 'env0' or 'all'")
        self.low_transition_detail_csv = bool(low_transition_detail_csv)
        self.low_transition_detail_envs = str(low_transition_detail_envs).strip().lower()
        if self.low_transition_detail_envs not in {"env0", "all"}:
            raise ValueError("low_transition_detail_envs must be 'env0' or 'all'")
        self._high_transition_capture_active = False
        self._env0_high_episode_idx = 0
        self._env0_hi_comp_seq: dict[str, list[float]] = {}
        self._env0_hi_acc_seq: list[float] = []
        self._env0_hi_low_ext_seq: list[float] = []
        self._env0_hi_intrinsic_seq: list[float] = []
        self._env0_hi_low_total_seq: list[float] = []
        self._hi_comp_seq_by_env: dict[int, dict[str, list[float]]] = {}
        self._hi_acc_seq_by_env: dict[int, list[float]] = {}
        self._hi_low_ext_seq_by_env: dict[int, list[float]] = {}
        self._hi_intrinsic_seq_by_env: dict[int, list[float]] = {}
        self._hi_low_total_seq_by_env: dict[int, list[float]] = {}
        self._acc_min = -5.0
        self._acc_max = 5.0
        self.her_debug_csv_interval_steps = max(0, int(her_debug_csv_interval_steps))
        self.her_debug_csv_max_rows_per_flush = max(1, int(her_debug_csv_max_rows_per_flush))
        self.low_debug_summary_interval_steps = max(0, int(low_debug_summary_interval_steps))
        self.low_debug_env_step_interval_steps = max(0, int(low_debug_env_step_interval_steps))

        # Effective high-level component keys (weighted values)
        # Sum over these components equals high_reward per interval.
        self.high_comp_keys = [
            "collision_reward",
            "progress_reward",
            "speed_ref_aux_reward",
            "comfort_reward_for_high",
            "lane_change_reward",
            "goal_lane_dense_reward",
            "punctual_reward",
            "wrong_lane_terminal_penalty",
        ]

        self._traj_csv_enabled = (self.csv_log_freq > 0 or self.high_transition_csv_all > 0) and bool(self.csv_save_dir)
        self._low_obs_csv_enabled = bool(self.csv_save_dir)
        self._diagnostic_csv_enabled = bool(self.csv_save_dir)
        self._low_transition_detail_csv_enabled = self.low_transition_detail_csv and bool(self.csv_save_dir)
        self._her_debug_csv_enabled = self.her_debug_csv_interval_steps > 0 and bool(self.csv_save_dir)
        self._low_debug_summary_enabled = self.low_debug_summary_interval_steps > 0 and bool(self.csv_save_dir)

        self.csv_active = False  # Whether current episode is being logged to CSV (env 0)
        self.csv_low_traj_recorded = False # Only record first interval's low traj per logged episode
        self._hi_start_seen = 0
        self._hi_start_saved = 0
        
        self._episode_counter = 0     # Counts finished episodes (across all envs)
        self._env0_ep_count = 0
        
        self._rollout_counter = 0
        self._last_dump_high = 0
        self._last_dump_low = 0
        self._last_her_debug_dump_step = 0
        self._last_low_debug_summary_step = 0
        self._last_env_diag_step = 0
        self._high_buffers, self._low_buffers = {}, {}

        # CSV Headers
        self.comp_keys = [
            "collision_reward",
            "progress_reward",
            "comfort_reward",
            "lane_change_reward",
            "goal_lane_dense_reward",
            "punctual_reward",
            "wrong_lane_terminal_penalty",
        ]
        self.ego_keys = ["ego_x", "ego_y", "ego_vx", "ego_vy"]

        if self._traj_csv_enabled or self._low_obs_csv_enabled:
            os.makedirs(self.csv_save_dir, exist_ok=True)

        if self._traj_csv_enabled:
            self.high_csv_path = os.path.join(self.csv_save_dir, "high_traj.csv")
            self.low_csv_path = os.path.join(self.csv_save_dir, "low_traj.csv")
            self.high_transition_csv_path = os.path.join(self.csv_save_dir, "high_interval_transitions.csv")
            
            base_header = ["episode", "step", "s", "a", "r", "next_s", "done"] + self.ego_keys
            comp_headers = [f"comp_{k}" for k in self.comp_keys]
            self._init_csv(self.high_csv_path, base_header + comp_headers)
            self._init_csv(self.low_csv_path, base_header + comp_headers)
            self._init_csv(
                self.high_transition_csv_path,
                [
                    "global_step",
                    "env_id",
                    "segment_id",
                    "high_obs",
                    "high_action",
                    "high_reward",
                    "high_comp_sum",
                    "high_comp_collision_reward",
                    "high_comp_progress_reward",
                    "high_comp_speed_ref_aux_reward",
                    "high_comp_comfort_reward_for_high",
                    "high_comp_lane_change_reward",
                    "high_comp_goal_lane_dense_reward",
                    "high_comp_punctual_reward",
                    "high_comp_wrong_lane_terminal_penalty",
                    "low_seq_collision_reward",
                    "low_seq_progress_reward",
                    "low_seq_speed_ref_aux_reward",
                    "low_seq_comfort_reward_for_high",
                    "low_seq_lane_change_reward",
                    "low_seq_goal_lane_dense_reward",
                    "low_seq_punctual_reward",
                    "low_seq_wrong_lane_terminal_penalty",
                    "low_seq_ego_acceleration",
                    "next_high_obs",
                    "done_env",
                ],
            )

        if self._low_obs_csv_enabled:
            self.low_obs_start_csv_path = os.path.join(self.csv_save_dir, "low_obs_hi_start.csv")
            low_obs_header = ["hi_start_seen", "hi_start_saved", "env_id", "step", "episode_env0", "low_obs"]
            self._init_csv(self.low_obs_start_csv_path, low_obs_header)

            self.high_interval_debug_csv_path = os.path.join(self.csv_save_dir, "high_interval_debug.csv")
            high_debug_header = [
                "hi_start_seen",
                "hi_start_saved",
                "env_id",
                "step",
                "episode_env0",
                "segment_id",
                "c",
                "ego_sub",
                "high_obs",
                "kin",
                "goal_action",
                "goal_phys",
                "safe_l1",
                "safe_u1",
                "safe_l2",
                "safe_u2",
                "safe_dx_l2",
                "safe_dx_u2",
            ]
            self._init_csv(self.high_interval_debug_csv_path, high_debug_header)

        if self._diagnostic_csv_enabled:
            self.env_step_diag_csv_path = os.path.join(self.csv_save_dir, "env_step_diagnostics.csv")
            self.episode_diag_csv_path = os.path.join(self.csv_save_dir, "episode_env_stats.csv")
            self.transition_health_csv_path = os.path.join(self.csv_save_dir, "transition_health.csv")
            env_diag_fields = [
                "time",
                "signal_time_global",
                "signal_episode_base",
                "initial_lane",
                "goal_lane",
                "ego_x",
                "ego_y",
                "ego_speed",
                "ego_lane",
                "bg_count",
                "bg_valid_count",
                "bg_low_speed_1",
                "bg_low_speed_2",
                "bg_near_goal_low",
                "bg_min_speed",
                "bg_mean_speed",
                "bg_max_x",
                "virtual_stop_count",
                "signal_is_green",
                "signal_remaining",
                "inter_episode_active",
            ]
            self.env_diag_fields = env_diag_fields
            self._init_csv(
                self.env_step_diag_csv_path,
                [
                    "global_step",
                    "env_id",
                    "segment_id",
                    "c",
                    "reward_env",
                    "low_reward_ext",
                    "intrinsic",
                    "low_reward_total",
                    "high_ret_running",
                    "done_env",
                    "done_low",
                    "replay",
                    "skip_replay",
                    "next_obs_is_dummy",
                    "low_action",
                    "goal_action",
                    "goal_phys",
                ] + [f"diag_{k}" for k in env_diag_fields],
            )
            self._init_csv(
                self.episode_diag_csv_path,
                [
                    "global_step",
                    "env_id",
                    "episode_index_global",
                    "ep_rew",
                    "ep_len",
                    "done_env",
                    "terminal_observation_present",
                    "terminal_signal_features",
                ] + [f"diag_{k}" for k in env_diag_fields],
            )
            self._init_csv(
                self.transition_health_csv_path,
                [
                    "global_step",
                    "env_id",
                    "segment_id",
                    "low_len",
                    "done_env",
                    "done_low",
                    "high_reward",
                    "component_sum",
                    "reward_minus_components",
                    "low_reward_ext_sum",
                    "intrinsic_sum",
                    "low_reward_total_sum",
                    "low_seq_acc_min",
                    "low_seq_acc_max",
                ] + [f"diag_{k}" for k in env_diag_fields],
            )

        if self._low_transition_detail_csv_enabled:
            self.low_transition_detail_csv_path = os.path.join(self.csv_save_dir, "low_step_transition_details.csv")
            self._init_csv(
                self.low_transition_detail_csv_path,
                [
                    "global_step",
                    "env_id",
                    "segment_id",
                    "c",
                    "replay",
                    "skip_replay",
                    "next_obs_is_dummy",
                    "done_env",
                    "done_low",
                    "reward_env",
                    "low_reward_ext",
                    "intrinsic",
                    "low_reward_total",
                    "high_ret_running",
                    "obs",
                    "next_obs",
                    "next_obs_tr",
                    "terminal_observation",
                    "high_obs",
                    "high_obs_start",
                    "next_high_obs",
                    "goal_action",
                    "goal_buffer_action",
                    "goal_phys",
                    "low_obs",
                    "low_action_raw",
                    "low_action",
                    "low_buffer_action",
                    "next_low_obs",
                    "ego_now_sub",
                    "ego_next_sub",
                    "reward_components",
                    "info_keys",
                ] + [f"diag_{k}" for k in env_diag_fields],
            )

        if self._her_debug_csv_enabled:
            self.her_debug_csv_path = os.path.join(self.csv_save_dir, "her_relabel_debug.csv")
            her_debug_header = [
                "global_step",
                "batch_i",
                "her_strategy",
                "her_future_mode",
                "steps_ahead",
                "source_seg_id",
                "source_t_in_seg",
                "source_ep_id",
                "source_ep_step",
                "source_done",
                "source_reward_stored",
                "source_ego_abs",
                "source_ego_next_abs",
                "source_obs",
                "source_action",
                "source_next_obs",
                "goal_seg_id",
                "goal_t_in_seg",
                "goal_ep_id",
                "goal_ep_step",
                "goal_reward_stored",
                "goal_transition_ego_abs",
                "goal_ego_abs",
                "goal_obs",
                "goal_action",
                "goal_next_obs",
                "relabeled_obs",
                "relabeled_next_obs",
                "relabeled_goal_rel",
                "relabeled_ego_abs",
                "relabel_t_norm_obs",
                "relabel_t_norm_next",
                "intrinsic_reward_new",
                "reward_ext",
                "reward_total_new",
                "relabeled_done",
                "start_ref_ep_step",
                "start_ref_ego_abs",
            ]
            self._init_csv(self.her_debug_csv_path, her_debug_header)

        if self._low_debug_summary_enabled:
            self.low_training_health_csv_path = os.path.join(self.csv_save_dir, "low_training_health.csv")
            self._init_csv(
                self.low_training_health_csv_path,
                [
                    "global_step",
                    "sample_calls",
                    "sampled_transitions",
                    "her_candidates",
                    "her_applied",
                    "her_apply_rate",
                    "skip_missing_metadata",
                    "skip_goal_lookup",
                    "skip_start_ref",
                    "skip_invalid_steps",
                    "steps_mean",
                    "steps_std",
                    "steps_min",
                    "steps_max",
                    "relabeled_terminal_rate",
                    "intrinsic_mean",
                    "intrinsic_std",
                    "intrinsic_abs_max",
                    "external_reward_mean",
                    "total_reward_mean",
                    "total_reward_std",
                    "goal_error_l2_mean",
                    "goal_error_l2_std",
                    "buffer_entries",
                    "buffer_valid_entries",
                    "buffer_occupancy",
                    "pending_her_debug_records",
                    "her_debug_records_dropped",
                ],
            )

    def _init_csv(self, path, header):
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

    def _fmt_arr(self, arr):
        # Format array as string with 2 decimal places
        arr = np.asarray(arr).reshape(-1)
        return np.array2string(
            arr, 
            separator=',', 
            max_line_width=np.inf, 
            formatter={'float_kind': lambda x: f"{x:.2f}"}
        ).replace('\n', '')

    def _append_csv(self, path, row) -> None:
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    @staticmethod
    def _stats_mean_std(total: float, total_sq: float, count: float) -> tuple[float, float]:
        if count <= 0:
            return np.nan, np.nan
        mean = float(total) / float(count)
        variance = max(float(total_sq) / float(count) - mean * mean, 0.0)
        return mean, float(np.sqrt(variance))

    def _flush_low_training_debug(self, step_now: int, *, force: bool = False) -> None:
        if not self._low_debug_summary_enabled or not hasattr(self, "low_training_health_csv_path"):
            return
        if not force and step_now - self._last_low_debug_summary_step < self.low_debug_summary_interval_steps:
            return

        rb = getattr(getattr(self.model, "low_agent", None), "replay_buffer", None)
        if rb is None or not hasattr(rb, "pop_training_debug_stats"):
            self._last_low_debug_summary_step = step_now
            return

        stats = rb.pop_training_debug_stats()
        if force and float(stats.get("sample_calls", 0.0)) <= 0:
            self._last_low_debug_summary_step = step_now
            return
        applied = float(stats.get("her_applied", 0.0))
        candidates = float(stats.get("her_candidates", 0.0))
        steps_mean, steps_std = self._stats_mean_std(
            stats.get("steps_sum", 0.0), stats.get("steps_sq_sum", 0.0), applied
        )
        intrinsic_mean, intrinsic_std = self._stats_mean_std(
            stats.get("intrinsic_sum", 0.0), stats.get("intrinsic_sq_sum", 0.0), applied
        )
        external_mean, _ = self._stats_mean_std(
            stats.get("external_sum", 0.0), stats.get("external_sq_sum", 0.0), applied
        )
        total_mean, total_std = self._stats_mean_std(
            stats.get("total_reward_sum", 0.0), stats.get("total_reward_sq_sum", 0.0), applied
        )
        goal_error_mean, goal_error_std = self._stats_mean_std(
            stats.get("goal_error_l2_sum", 0.0), stats.get("goal_error_l2_sq_sum", 0.0), applied
        )
        buffer_size = float(stats.get("buffer_size", 0.0))
        buffer_entries = float(stats.get("buffer_entries", 0.0))
        buffer_occupancy = buffer_entries / buffer_size if buffer_size > 0 else 0.0
        apply_rate = applied / candidates if candidates > 0 else 0.0
        terminal_rate = float(stats.get("relabeled_terminal", 0.0)) / applied if applied > 0 else 0.0
        steps_min = float(stats.get("steps_min", np.nan))
        steps_max = float(stats.get("steps_max", np.nan))
        steps_min = steps_min if np.isfinite(steps_min) else np.nan
        steps_max = steps_max if np.isfinite(steps_max) else np.nan

        self._append_csv(
            self.low_training_health_csv_path,
            [
                int(step_now),
                int(stats.get("sample_calls", 0.0)),
                int(stats.get("sampled_transitions", 0.0)),
                int(candidates),
                int(applied),
                apply_rate,
                int(stats.get("skip_missing_metadata", 0.0)),
                int(stats.get("skip_goal_lookup", 0.0)),
                int(stats.get("skip_start_ref", 0.0)),
                int(stats.get("skip_invalid_steps", 0.0)),
                steps_mean,
                steps_std,
                steps_min,
                steps_max,
                terminal_rate,
                intrinsic_mean,
                intrinsic_std,
                float(stats.get("intrinsic_abs_max", 0.0)),
                external_mean,
                total_mean,
                total_std,
                goal_error_mean,
                goal_error_std,
                int(buffer_entries),
                int(stats.get("buffer_valid_entries", 0.0)),
                buffer_occupancy,
                int(stats.get("pending_her_debug_records", 0.0)),
                int(stats.get("her_debug_records_dropped", 0.0)),
            ],
        )

        logger = getattr(self.model, "low_logger", None)
        if logger is not None:
            logger.record("her/apply_rate", apply_rate)
            logger.record("her/candidates", candidates)
            logger.record("her/applied", applied)
            logger.record("her/skip_missing_metadata", float(stats.get("skip_missing_metadata", 0.0)))
            logger.record("her/skip_goal_lookup", float(stats.get("skip_goal_lookup", 0.0)))
            logger.record("her/skip_start_ref", float(stats.get("skip_start_ref", 0.0)))
            logger.record("her/steps_mean", steps_mean)
            logger.record("her/terminal_rate", terminal_rate)
            logger.record("her/intrinsic_mean", intrinsic_mean)
            logger.record("her/intrinsic_std", intrinsic_std)
            logger.record("her/goal_error_l2_mean", goal_error_mean)
            logger.record("replay/buffer_occupancy", buffer_occupancy)
            logger.record("replay/valid_entries", float(stats.get("buffer_valid_entries", 0.0)))
            logger.record("her/debug_records_dropped", float(stats.get("her_debug_records_dropped", 0.0)))

        self._last_low_debug_summary_step = step_now

    def _on_training_end(self) -> None:
        step_now = int(getattr(self.model, "num_timesteps", 0))
        self._flush_low_training_debug(step_now, force=True)
        logger = getattr(self.model, "low_logger", None)
        if logger is not None:
            logger.dump(step=step_now)

    @staticmethod
    def _json_arr(arr) -> str:
        a = np.asarray(arr, dtype=np.float32)
        return json.dumps(a.tolist(), ensure_ascii=True)

    @staticmethod
    def _json_obj(obj: Any) -> str:
        def _default(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
            try:
                return float(value)
            except (TypeError, ValueError):
                return str(value)

        return json.dumps(obj, ensure_ascii=True, default=_default)

    @staticmethod
    def _diag_value(diag: dict[str, Any], key: str) -> float:
        val = diag.get(key, np.nan)
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan

    @staticmethod
    def _safe_norm_to_dx(norm_val: np.ndarray, dx_low: float, dx_high: float) -> np.ndarray:
        n = np.asarray(norm_val, dtype=np.float32)
        return ((n + 1.0) * 0.5 * float(dx_high - dx_low) + float(dx_low)).astype(np.float32)

    def _reset_env0_hi_buffers(self) -> None:
        self._reset_hi_buffers(0)
        self._env0_hi_comp_seq = self._hi_comp_seq_by_env[0]
        self._env0_hi_acc_seq = self._hi_acc_seq_by_env[0]
        self._env0_hi_low_ext_seq = self._hi_low_ext_seq_by_env[0]
        self._env0_hi_intrinsic_seq = self._hi_intrinsic_seq_by_env[0]
        self._env0_hi_low_total_seq = self._hi_low_total_seq_by_env[0]

    def _reset_hi_buffers(self, env_i: int) -> None:
        env_i = int(env_i)
        self._hi_comp_seq_by_env[env_i] = {k: [] for k in self.high_comp_keys}
        self._hi_acc_seq_by_env[env_i] = []
        self._hi_low_ext_seq_by_env[env_i] = []
        self._hi_intrinsic_seq_by_env[env_i] = []
        self._hi_low_total_seq_by_env[env_i] = []

    def _csv_env_indices(self, n_envs: int) -> list[int]:
        if self.high_transition_csv_envs == "all":
            return list(range(int(n_envs)))
        return [0] if int(n_envs) > 0 else []

    def _detail_csv_env_indices(self, n_envs: int) -> list[int]:
        if self.low_transition_detail_envs == "all":
            return list(range(int(n_envs)))
        return [0] if int(n_envs) > 0 else []

    def _capture_active(self, env_i: int) -> bool:
        active = self._high_transition_capture_active
        if isinstance(active, np.ndarray):
            return int(env_i) < active.size and bool(active[int(env_i)])
        return int(env_i) == 0 and bool(active)

    def _effective_high_components(self, rc: dict[str, Any]) -> dict[str, float]:
        comp = {
            "collision_reward": float(rc.get("collision_reward", 0.0)),
            "progress_reward": float(rc.get("progress_reward", 0.0)),
            "speed_ref_aux_reward": float(rc.get("speed_ref_aux_reward", 0.0)),
            "lane_change_reward": float(rc.get("lane_change_reward", 0.0)),
            "goal_lane_dense_reward": float(rc.get("goal_lane_dense_reward", 0.0)),
            "punctual_reward": float(rc.get("punctual_reward", 0.0)),
            "wrong_lane_terminal_penalty": float(rc.get("wrong_lane_terminal_penalty", 0.0)),
        }
        comp["comfort_reward_for_high"] = float(rc.get("comfort_reward", 0.0))
        return comp

    def _to_physical_acc(self, a1: float) -> float:
        v = float(a1)
        if -1.0001 <= v <= 1.0001:
            return float(((v + 1.0) * 0.5 * (self._acc_max - self._acc_min)) + self._acc_min)
        return float(np.clip(v, self._acc_min, self._acc_max))

    def _on_training_start(self) -> None:
        n_envs = int(getattr(self.model, "n_envs", 1))
        self._ep_ret = np.zeros(n_envs, dtype=np.float32)
        self._ep_len = np.zeros(n_envs, dtype=np.int32)
        self._ep_comp_sums: dict[str, np.ndarray] = {}
        self._env0_high_episode_idx = 0
        self._high_episode_idx = np.zeros(n_envs, dtype=np.int64)
        self._high_transition_capture_active = np.zeros(n_envs, dtype=bool)
        for env_i in self._csv_env_indices(n_envs):
            self._high_transition_capture_active[env_i] = self.high_transition_csv_all > 0
            self._reset_hi_buffers(env_i)
        self._reset_env0_hi_buffers()

        # Read action acceleration bounds once for low-level acceleration sequence logging.
        try:
            env_cfg = self.model.env.get_attr("config", indices=0)[0]
            acc_range = env_cfg.get("action", {}).get("acceleration_range", [-5.0, 5.0])
            if isinstance(acc_range, (list, tuple)) and len(acc_range) >= 2:
                self._acc_min = float(acc_range[0])
                self._acc_max = float(acc_range[1])
        except Exception:
            self._acc_min, self._acc_max = -5.0, 5.0

        if self.mpc_fail_csv_path:
            os.makedirs(os.path.dirname(self.mpc_fail_csv_path), exist_ok=True)
            self._init_csv(
                self.mpc_fail_csv_path,
                [
                    "global_step",
                    "env_id",
                    "segment_id",
                    "planner",
                    "message",
                    "goal_phys",
                    "ego_sub_now",
                    "low_obs",
                ],
            )

    @staticmethod
    def _record_smooth(logger, buffers: dict, tag: str, value: float, window: int = 50):
        buf = buffers.setdefault(tag, deque(maxlen=window))
        buf.append(float(value))
        # logger.record_mean(tag, float(sum(buf) / len(buf)))
        logger.record(tag, float(sum(buf) / len(buf)))

    def _on_rollout_end(self) -> None:
        loc = self.locals
        low_ret = np.asarray(loc.get("low_ret", []), dtype=np.float32).reshape(-1)
        if low_ret.size == 0:
            return

        low_len = np.asarray(loc.get("low_len", []), dtype=np.float32).reshape(-1)
        low_safety_clip_ratio = np.asarray(loc.get("low_safety_clip_ratio", []), dtype=np.float32).reshape(-1)
        low_comp_sums = loc.get("low_comp_sums", {})
        mpc_plan_attempt_count = int(loc.get("mpc_plan_attempt_count", 0))
        mpc_plan_fail_count = int(loc.get("mpc_plan_fail_count", 0))

        # --- TensorBoard Logging ---
        self._rollout_counter += int(low_ret.size)
        self._record_smooth(self.model.low_logger, self._low_buffers, "rollout/ep_rew", float(low_ret.mean()))
        self._record_smooth(self.model.low_logger, self._low_buffers, "rollout/ep_len", float(low_len.mean()) if low_len.size else 0.0)
        if low_safety_clip_ratio.size:
            self._record_smooth(
                self.model.low_logger,
                self._low_buffers,
                "rollout/safety_clip_ratio",
                float(low_safety_clip_ratio.mean()),
            )
        for k, v in low_comp_sums.items():
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            if arr.size:
                self._record_smooth(self.model.low_logger, self._low_buffers, f"rollout/{k}", float(arr.mean()))

        if mpc_plan_attempt_count > 0:
            fail_rate = float(mpc_plan_fail_count) / float(max(mpc_plan_attempt_count, 1))
            self._record_smooth(
                self.model.low_logger,
                self._low_buffers,
                "rollout/mpc_plan_fail_rate",
                fail_rate,
            )
            self.model.low_logger.record("rollout/mpc_plan_fail_count", float(mpc_plan_fail_count))
            self.model.low_logger.record("rollout/mpc_plan_attempt_count", float(mpc_plan_attempt_count))

        # goal tracking error at the end of each high-interval
        goal_err = np.asarray(loc.get("goal_err", []), dtype=np.float32)
        if goal_err.size:
            self._record_smooth(self.model.low_logger, self._low_buffers, "goal_err/x", float(goal_err[:, 0].mean()))
            self._record_smooth(self.model.low_logger, self._low_buffers, "goal_err/y", float(goal_err[:, 1].mean()))
            self._record_smooth(self.model.low_logger, self._low_buffers, "goal_err/vx", float(goal_err[:, 2].mean()))
            self._record_smooth(self.model.low_logger, self._low_buffers, "goal_err/vy", float(goal_err[:, 3].mean()))

        intrinsic_unweighted = np.asarray(loc.get("intrinsic_unweighted", []), dtype=np.float32).reshape(-1)
        if intrinsic_unweighted.size:
            self._record_smooth(
                self.model.low_logger,
                self._low_buffers,
                "goal_err/intrinsic_unweighted",
                float(intrinsic_unweighted.mean()),
                window=1,
            )

        goal_dist_start = np.asarray(loc.get("goal_dist_start", []), dtype=np.float32)
        if goal_dist_start.size:
            self._record_smooth(self.model.low_logger, self._low_buffers, "goal_err/start_dist_x", float(goal_dist_start[:, 0].mean()))
            self._record_smooth(self.model.low_logger, self._low_buffers, "goal_err/start_dist_y", float(goal_dist_start[:, 1].mean()))
            self._record_smooth(self.model.low_logger, self._low_buffers, "goal_err/start_dist_vx", float(goal_dist_start[:, 2].mean()))
            self._record_smooth(self.model.low_logger, self._low_buffers, "goal_err/start_dist_vy", float(goal_dist_start[:, 3].mean()))

        if self._rollout_counter - self._last_dump_low >= self.low_log_interval_hi:
            self.model.low_logger.dump(step=self.model.num_timesteps)
            self._last_dump_low = self._rollout_counter

        # --- CSV Logging (High Level Transition) ---
        done_low = loc.get("done_low", [])
        if self.csv_active and len(done_low) > 0 and done_low[0]:
            if hasattr(self.model, "high_logger"):
                # Env 0 finished a high interval
                high_obs_start = loc.get("high_obs_start")[0]
                goal_action = loc.get("goal_action")[0]
                high_ret = loc.get("high_ret")[0]
                next_high_obs = loc.get("next_high_obs")[0]
                done = loc.get("done")[0]
                
                # Ego State
                ego_s = loc.get("ego_start")[0]

                row_high = [
                    self._env0_ep_count,
                    self.model.num_timesteps,
                    self._fmt_arr(high_obs_start),
                    self._fmt_arr(goal_action),
                    f"{float(high_ret):.2f}",
                    self._fmt_arr(next_high_obs),
                    int(done)
                ]
                for v in ego_s:
                    row_high.append(f"{float(v):.2f}")

                for k in self.comp_keys:
                    arr = low_comp_sums.get(k, np.array([]))
                    val = float(arr[0]) if arr.size > 0 else 0.0
                    row_high.append(f"{val:.2f}")

                self._append_csv(self.high_csv_path, row_high)

            self.csv_low_traj_recorded = True

        done = loc.get("done", [])
        if len(done) > 0 and done[0]:
            self._env0_ep_count += 1
            # Determines if NEXT episode should be logged
            if self.csv_log_freq > 0 and (self._env0_ep_count % self.csv_log_freq == 0):
                self.csv_active = True
                self.csv_low_traj_recorded = False # Reset for new episode
            else:
                self.csv_active = False

    def _on_step(self) -> bool:
        loc = self.locals
        step_now = int(getattr(self.model, "num_timesteps", 0))
        reward_env = np.asarray(loc.get("reward_env", 0.0), dtype=np.float32).reshape(-1)
        dones = np.asarray(loc.get("done", False), dtype=bool).reshape(-1)
        infos = loc.get("infos", [])
        c = np.asarray(loc.get("c", []), dtype=np.int32).reshape(-1)
        low_action_all = np.asarray(loc.get("low_action", []), dtype=np.float32)
        replay_mask = np.asarray(loc.get("replay_mask", np.ones_like(dones, dtype=bool)), dtype=bool).reshape(-1)
        if replay_mask.size != dones.size:
            replay_mask = np.ones_like(dones, dtype=bool)
        if not replay_mask.any():
            return True

        self._flush_low_training_debug(step_now)

        # Start of a new high interval for captured envs.
        if c.size > 0:
            for env_i in self._csv_env_indices(min(c.size, replay_mask.size)):
                if env_i < replay_mask.size and bool(replay_mask[env_i]) and int(c[env_i]) == 0:
                    self._reset_hi_buffers(env_i)

        if self._her_debug_csv_enabled and hasattr(self, "her_debug_csv_path"):
            if (step_now - self._last_her_debug_dump_step) >= self.her_debug_csv_interval_steps:
                rb = getattr(getattr(self.model, "low_agent", None), "replay_buffer", None)
                if rb is not None and hasattr(rb, "pop_her_debug_records"):
                    records = rb.pop_her_debug_records(max_items=self.her_debug_csv_max_rows_per_flush)
                    for rec in records:
                        row = [
                            step_now,
                            int(rec.get("batch_i", -1)),
                            str(rec.get("her_strategy", "")),
                            str(rec.get("her_future_mode", "")),
                            int(rec.get("steps_ahead", -1)),
                            int(rec.get("source_seg_id", -1)),
                            int(rec.get("source_t_in_seg", -1)),
                            int(rec.get("source_ep_id", -1)),
                            int(rec.get("source_ep_step", -1)),
                            int(bool(rec.get("source_done", False))),
                            float(rec.get("source_reward_stored", 0.0)),
                            json.dumps(rec.get("source_ego_abs", []), ensure_ascii=True),
                            json.dumps(rec.get("source_ego_next_abs", []), ensure_ascii=True),
                            json.dumps(rec.get("source_obs", []), ensure_ascii=True),
                            json.dumps(rec.get("source_action", []), ensure_ascii=True),
                            json.dumps(rec.get("source_next_obs", []), ensure_ascii=True),
                            int(rec.get("goal_seg_id", -1)),
                            int(rec.get("goal_t_in_seg", -1)),
                            int(rec.get("goal_ep_id", -1)),
                            int(rec.get("goal_ep_step", -1)),
                            float(rec.get("goal_reward_stored", 0.0)),
                            json.dumps(rec.get("goal_transition_ego_abs", []), ensure_ascii=True),
                            json.dumps(rec.get("goal_ego_abs", []), ensure_ascii=True),
                            json.dumps(rec.get("goal_obs", []), ensure_ascii=True),
                            json.dumps(rec.get("goal_action", []), ensure_ascii=True),
                            json.dumps(rec.get("goal_next_obs", []), ensure_ascii=True),
                            json.dumps(rec.get("relabeled_obs", []), ensure_ascii=True),
                            json.dumps(rec.get("relabeled_next_obs", []), ensure_ascii=True),
                            json.dumps(rec.get("relabeled_goal_rel", []), ensure_ascii=True),
                            json.dumps(rec.get("relabeled_ego_abs", []), ensure_ascii=True),
                            float(rec.get("relabel_t_norm_obs", 0.0)),
                            float(rec.get("relabel_t_norm_next", 0.0)),
                            float(rec.get("intrinsic_reward_new", 0.0)),
                            float(rec.get("reward_ext", 0.0)),
                            float(rec.get("reward_total_new", 0.0)),
                            int(bool(rec.get("relabeled_done", False))),
                            int(rec.get("start_ref_ep_step", -1)),
                            json.dumps(rec.get("start_ref_ego_abs", []), ensure_ascii=True),
                        ]
                        self._append_csv(self.her_debug_csv_path, row)
                self._last_her_debug_dump_step = step_now

        if self.mpc_fail_csv_path:
            fail_records = list(loc.get("mpc_fail_records", []) or [])
            if fail_records:
                for rec in fail_records:
                    self._append_csv(
                        self.mpc_fail_csv_path,
                        [
                            int(rec.get("global_step", 0)),
                            int(rec.get("env_id", -1)),
                            int(rec.get("segment_id", -1)),
                            str(rec.get("planner", "")),
                            str(rec.get("message", "")),
                            json.dumps(rec.get("goal_phys", []), ensure_ascii=True),
                            json.dumps(rec.get("ego_sub_now", []), ensure_ascii=True),
                            json.dumps(rec.get("low_obs", []), ensure_ascii=True),
                        ],
                    )

        # --- Update TB stats & Capture env 0 components ---
        rc_env0 = {}
        if reward_env.size:
            self._ep_ret[replay_mask] += reward_env[replay_mask]
            self._ep_len[replay_mask] += 1

        if infos:
            for i, info in enumerate(infos):
                if i >= replay_mask.size or not replay_mask[i]:
                    continue
                rc = info.get("reward_components", {})
                for name, val in rc.items():
                    self._ep_comp_sums.setdefault(name, np.zeros_like(self._ep_ret))[i] += float(val)
                if i == 0 and self.csv_active:
                    rc_env0 = rc

        # --- Transition and environment health diagnostics ---
        low_reward_ext_arr = np.asarray(loc.get("low_reward_ext", []), dtype=np.float32).reshape(-1)
        intrinsic_arr = np.asarray(loc.get("intrinsic", []), dtype=np.float32).reshape(-1)
        low_reward_total_arr = np.asarray(loc.get("low_reward_total", []), dtype=np.float32).reshape(-1)
        high_ret_arr = np.asarray(loc.get("high_ret", []), dtype=np.float32).reshape(-1)
        seg_id_arr = np.asarray(loc.get("seg_id", np.full_like(c, -1)), dtype=np.int64).reshape(-1)
        skip_replay_arr = np.asarray(loc.get("skip_replay_mask", ~replay_mask), dtype=bool).reshape(-1)
        next_obs_is_dummy_arr = np.asarray(loc.get("next_obs_is_dummy", np.zeros_like(replay_mask)), dtype=bool).reshape(-1)

        goal_action_arr = np.asarray(loc.get("goal_action", []), dtype=np.float32)
        goal_phys_arr = np.asarray(loc.get("goal_phys", []), dtype=np.float32)
        done_low_arr = np.asarray(loc.get("done_low", np.zeros_like(replay_mask)), dtype=bool).reshape(-1)

        write_env_diag = (
            self._diagnostic_csv_enabled
            and self.low_debug_env_step_interval_steps > 0
            and step_now - self._last_env_diag_step >= self.low_debug_env_step_interval_steps
        )
        if write_env_diag and infos and replay_mask.size > 0 and bool(replay_mask[0]):
            i = 0
            diag = infos[i].get("env_diagnostics", {}) if isinstance(infos[i], dict) else {}
            row = [
                int(getattr(self.model, "num_timesteps", 0)),
                i,
                int(seg_id_arr[i]) if seg_id_arr.size > i else -1,
                int(c[i]) if c.size > i else -1,
                float(reward_env[i]) if reward_env.size > i else np.nan,
                float(low_reward_ext_arr[i]) if low_reward_ext_arr.size > i else np.nan,
                float(intrinsic_arr[i]) if intrinsic_arr.size > i else np.nan,
                float(low_reward_total_arr[i]) if low_reward_total_arr.size > i else np.nan,
                float(high_ret_arr[i]) if high_ret_arr.size > i else np.nan,
                int(dones[i]) if dones.size > i else 0,
                int(done_low_arr[i]) if done_low_arr.size > i else 0,
                int(replay_mask[i]),
                int(skip_replay_arr[i]) if skip_replay_arr.size > i else int(not replay_mask[i]),
                int(next_obs_is_dummy_arr[i]) if next_obs_is_dummy_arr.size > i else 0,
                self._json_arr(low_action_all[i]) if low_action_all.ndim == 2 and low_action_all.shape[0] > i else "[]",
                self._json_arr(goal_action_arr[i]) if goal_action_arr.ndim == 2 and goal_action_arr.shape[0] > i else "[]",
                self._json_arr(goal_phys_arr[i]) if goal_phys_arr.ndim == 2 and goal_phys_arr.shape[0] > i else "[]",
            ]
            row.extend(self._diag_value(diag, k) for k in getattr(self, "env_diag_fields", []))
            self._append_csv(self.env_step_diag_csv_path, row)
            self._last_env_diag_step = step_now

        if self._low_transition_detail_csv_enabled and hasattr(self, "low_transition_detail_csv_path"):
            obs_arr = np.asarray(loc.get("obs", []), dtype=np.float32)
            next_obs_arr = np.asarray(loc.get("next_obs", []), dtype=np.float32)
            next_obs_tr_arr = np.asarray(loc.get("next_obs_tr", []), dtype=np.float32)
            high_obs_arr = np.asarray(loc.get("high_obs", []), dtype=np.float32)
            high_obs_start_arr = np.asarray(loc.get("high_obs_start", []), dtype=np.float32)
            next_high_obs_arr = np.asarray(loc.get("next_high_obs", []), dtype=np.float32)
            low_obs_arr = np.asarray(loc.get("low_obs", []), dtype=np.float32)
            low_action_raw_arr = np.asarray(loc.get("low_action_raw", []), dtype=np.float32)
            low_buffer_action_arr = np.asarray(loc.get("low_buffer_action", []), dtype=np.float32)
            next_low_obs_arr = np.asarray(loc.get("next_low_obs", []), dtype=np.float32)
            goal_buffer_action_arr = np.asarray(loc.get("goal_buffer_action", []), dtype=np.float32)
            ego_now_sub_arr = np.asarray(loc.get("ego_now_sub", []), dtype=np.float32)
            ego_next_sub_arr = np.asarray(loc.get("ego_next_sub", []), dtype=np.float32)

            def arr_row(arr: np.ndarray, env_i: int) -> str:
                return self._json_arr(arr[env_i]) if arr.ndim >= 2 and arr.shape[0] > env_i else "[]"

            max_rows = max(int(replay_mask.size), int(dones.size), int(c.size))
            for env_i in self._detail_csv_env_indices(max_rows):
                if env_i >= replay_mask.size or not bool(replay_mask[env_i]):
                    continue
                info_i = infos[env_i] if infos and len(infos) > env_i and isinstance(infos[env_i], dict) else {}
                diag = info_i.get("env_diagnostics", {}) if isinstance(info_i, dict) else {}
                terminal_obs = info_i.get("terminal_observation", [])
                rc_i = info_i.get("reward_components", {}) if isinstance(info_i, dict) else {}
                row_detail = [
                    int(getattr(self.model, "num_timesteps", 0)),
                    int(env_i),
                    int(seg_id_arr[env_i]) if seg_id_arr.size > env_i else -1,
                    int(c[env_i]) if c.size > env_i else -1,
                    int(replay_mask[env_i]),
                    int(skip_replay_arr[env_i]) if skip_replay_arr.size > env_i else int(not replay_mask[env_i]),
                    int(next_obs_is_dummy_arr[env_i]) if next_obs_is_dummy_arr.size > env_i else 0,
                    int(dones[env_i]) if dones.size > env_i else 0,
                    int(done_low_arr[env_i]) if done_low_arr.size > env_i else 0,
                    float(reward_env[env_i]) if reward_env.size > env_i else np.nan,
                    float(low_reward_ext_arr[env_i]) if low_reward_ext_arr.size > env_i else np.nan,
                    float(intrinsic_arr[env_i]) if intrinsic_arr.size > env_i else np.nan,
                    float(low_reward_total_arr[env_i]) if low_reward_total_arr.size > env_i else np.nan,
                    float(high_ret_arr[env_i]) if high_ret_arr.size > env_i else np.nan,
                    arr_row(obs_arr, env_i),
                    arr_row(next_obs_arr, env_i),
                    arr_row(next_obs_tr_arr, env_i),
                    self._json_arr(terminal_obs) if terminal_obs is not None and np.asarray(terminal_obs).size else "[]",
                    arr_row(high_obs_arr, env_i),
                    arr_row(high_obs_start_arr, env_i),
                    arr_row(next_high_obs_arr, env_i),
                    arr_row(goal_action_arr, env_i),
                    arr_row(goal_buffer_action_arr, env_i),
                    arr_row(goal_phys_arr, env_i),
                    arr_row(low_obs_arr, env_i),
                    arr_row(low_action_raw_arr, env_i),
                    arr_row(low_action_all, env_i),
                    arr_row(low_buffer_action_arr, env_i),
                    arr_row(next_low_obs_arr, env_i),
                    arr_row(ego_now_sub_arr, env_i),
                    arr_row(ego_next_sub_arr, env_i),
                    self._json_obj(rc_i),
                    self._json_obj(sorted(list(info_i.keys()))),
                ]
                row_detail.extend(self._diag_value(diag, k) for k in getattr(self, "env_diag_fields", []))
                self._append_csv(self.low_transition_detail_csv_path, row_detail)

        # Collect per-step weighted component values and low acceleration in current high interval.
        if (
            self.high_transition_csv_all > 0
            and self._traj_csv_enabled
            and infos
        ):
            for env_i in self._csv_env_indices(len(infos)):
                if env_i >= replay_mask.size or not bool(replay_mask[env_i]) or not self._capture_active(env_i):
                    continue
                rc_i = infos[env_i].get("reward_components", {}) if isinstance(infos[env_i], dict) else {}
                eff_i = self._effective_high_components(rc_i)
                comp_seq = self._hi_comp_seq_by_env.setdefault(env_i, {k: [] for k in self.high_comp_keys})
                for k in self.high_comp_keys:
                    comp_seq.setdefault(k, []).append(float(eff_i.get(k, 0.0)))

                acc_phys = 0.0
                if low_action_all.ndim == 2 and low_action_all.shape[0] > env_i and low_action_all.shape[1] >= 2:
                    acc_phys = self._to_physical_acc(float(low_action_all[env_i, 1]))
                self._hi_acc_seq_by_env.setdefault(env_i, []).append(float(acc_phys))
                if low_reward_ext_arr.size > env_i:
                    self._hi_low_ext_seq_by_env.setdefault(env_i, []).append(float(low_reward_ext_arr[env_i]))
                if intrinsic_arr.size > env_i:
                    self._hi_intrinsic_seq_by_env.setdefault(env_i, []).append(float(intrinsic_arr[env_i]))
                if low_reward_total_arr.size > env_i:
                    self._hi_low_total_seq_by_env.setdefault(env_i, []).append(float(low_reward_total_arr[env_i]))

        # --- CSV Logs (low_obs snapshot at high-interval start) ---
        if self._low_obs_csv_enabled and hasattr(self, "low_obs_start_csv_path"):
            c = np.asarray(loc.get("c", []), dtype=np.int32).reshape(-1)
            low_obs = np.asarray(loc.get("low_obs", []), dtype=np.float32)
            if c.size and low_obs.size and low_obs.ndim == 2 and low_obs.shape[0] == c.size:
                start_idx = np.flatnonzero((c == 0) & replay_mask)
                if self.low_obs_csv_env0_only:
                    start_idx = start_idx[start_idx == 0]

                high_obs = np.asarray(loc.get("high_obs", []), dtype=np.float32)
                kin = np.asarray(loc.get("kin", []), dtype=np.float32)
                goal_action = np.asarray(loc.get("goal_action", []), dtype=np.float32)
                goal_phys = np.asarray(loc.get("goal_phys", []), dtype=np.float32)
                seg_id = np.asarray(loc.get("seg_id", []), dtype=np.int64).reshape(-1)

                safe_bounds = None
                safe_dx_l2 = None
                safe_dx_u2 = None
                if (
                    start_idx.size
                    and hasattr(self.model, "high_goal_safe_bounds")
                    and high_obs.ndim == 2
                    and high_obs.shape[0] == c.size
                ):
                    try:
                        safe_bounds = self.model.high_goal_safe_bounds.compute_np(high_obs[start_idx])
                        l2_np = np.asarray(safe_bounds.get("l2", []), dtype=np.float32)
                        u2_np = np.asarray(safe_bounds.get("u2", []), dtype=np.float32)
                        if l2_np.size and u2_np.size:
                            dx_low = float(getattr(self.model.high_goal_safe_bounds, "dx_low", 0.0))
                            dx_high = float(getattr(self.model.high_goal_safe_bounds, "dx_high", 1.0))
                            safe_dx_l2 = self._safe_norm_to_dx(l2_np, dx_low, dx_high)
                            safe_dx_u2 = self._safe_norm_to_dx(u2_np, dx_low, dx_high)
                            empty_mask = l2_np > u2_np
                            safe_dx_l2 = np.where(empty_mask, np.nan, safe_dx_l2)
                            safe_dx_u2 = np.where(empty_mask, np.nan, safe_dx_u2)
                    except Exception:
                        safe_bounds = None
                        safe_dx_l2 = None
                        safe_dx_u2 = None

                start_pos = {int(env_i): pos for pos, env_i in enumerate(start_idx.tolist())}

                for env_i in start_idx:
                    self._hi_start_seen += 1
                    if ((self._hi_start_seen - 1) % self.low_obs_csv_interval_hi) != 0:
                        continue

                    self._hi_start_saved += 1
                    row = [
                        self._hi_start_seen,
                        self._hi_start_saved,
                        int(env_i),
                        int(self.model.num_timesteps),
                        int(self._env0_ep_count),
                        self._fmt_arr(low_obs[env_i]),
                    ]
                    self._append_csv(self.low_obs_start_csv_path, row)

                    p = start_pos.get(int(env_i), -1)
                    if p < 0:
                        continue

                    ego_sub = np.asarray([], dtype=np.float32)
                    if kin.ndim == 3 and kin.shape[0] > int(env_i):
                        try:
                            ego_sub = utils.extract_ego_substate(
                                kin[int(env_i): int(env_i) + 1],
                                self.model.ego_feature_idx,
                            )[0]
                        except Exception:
                            ego_sub = np.asarray([], dtype=np.float32)

                    safe_l1 = np.asarray([], dtype=np.float32)
                    safe_u1 = np.asarray([], dtype=np.float32)
                    safe_l2 = np.asarray([], dtype=np.float32)
                    safe_u2 = np.asarray([], dtype=np.float32)
                    safe_dx_l2_row = np.asarray([], dtype=np.float32)
                    safe_dx_u2_row = np.asarray([], dtype=np.float32)
                    if safe_bounds is not None:
                        safe_l1 = np.asarray(safe_bounds.get("l1", []), dtype=np.float32)[p:p + 1]
                        safe_u1 = np.asarray(safe_bounds.get("u1", []), dtype=np.float32)[p:p + 1]
                        safe_l2 = np.asarray(safe_bounds.get("l2", []), dtype=np.float32)[p:p + 1]
                        safe_u2 = np.asarray(safe_bounds.get("u2", []), dtype=np.float32)[p:p + 1]
                    if safe_dx_l2 is not None:
                        safe_dx_l2_row = np.asarray(safe_dx_l2, dtype=np.float32)[p:p + 1]
                    if safe_dx_u2 is not None:
                        safe_dx_u2_row = np.asarray(safe_dx_u2, dtype=np.float32)[p:p + 1]

                    debug_row = [
                        self._hi_start_seen,
                        self._hi_start_saved,
                        int(env_i),
                        int(self.model.num_timesteps),
                        int(self._env0_ep_count),
                        int(seg_id[int(env_i)]) if seg_id.size > int(env_i) else -1,
                        int(c[int(env_i)]),
                        self._json_arr(ego_sub),
                        self._json_arr(high_obs[int(env_i)]) if high_obs.ndim == 2 and high_obs.shape[0] > int(env_i) else "[]",
                        self._json_arr(kin[int(env_i)]) if kin.ndim == 3 and kin.shape[0] > int(env_i) else "[]",
                        self._json_arr(goal_action[int(env_i)]) if goal_action.ndim == 2 and goal_action.shape[0] > int(env_i) else "[]",
                        self._json_arr(goal_phys[int(env_i)]) if goal_phys.ndim == 2 and goal_phys.shape[0] > int(env_i) else "[]",
                        self._json_arr(safe_l1),
                        self._json_arr(safe_u1),
                        self._json_arr(safe_l2),
                        self._json_arr(safe_u2),
                        self._json_arr(safe_dx_l2_row),
                        self._json_arr(safe_dx_u2_row),
                    ]
                    self._append_csv(self.high_interval_debug_csv_path, debug_row)

        # --- CSV Logs (Low Level Step) ---
        # Record only if active and not yet done with first interval
        if self.csv_active and not self.csv_low_traj_recorded and replay_mask.size > 0 and replay_mask[0]:
            low_obs = loc.get("low_obs")[0]
            low_action = loc.get("low_action")[0]
            low_reward = loc.get("low_reward_total")[0]
            next_low_obs = loc.get("next_low_obs")[0]
            done_low = loc.get("done_low")[0]
            
            # Ego
            kin = loc.get("kin")
            ego_s = utils.extract_ego_substate(kin, self.model.ego_feature_idx)[0]

            row = [
                self._env0_ep_count,
                self.model.num_timesteps,
                self._fmt_arr(low_obs),
                self._fmt_arr(low_action),
                f"{float(low_reward):.2f}",
                self._fmt_arr(next_low_obs),
                int(done_low)
            ]
            for v in ego_s:
                row.append(f"{float(v):.2f}")

            for k in self.comp_keys:
                row.append(f"{float(rc_env0.get(k, 0.0)):.2f}")
            self._append_csv(self.low_csv_path, row)

        # --- CSV Logs (captured env high-episodes, sampled every k episodes) ---
        if self.high_transition_csv_all > 0 and self._traj_csv_enabled:
            done_low = np.asarray(loc.get("done_low", False), dtype=bool).reshape(-1)
            if done_low.size > 0 and replay_mask.size > 0:
                high_obs_start = np.asarray(loc.get("high_obs_start", []), dtype=np.float32)
                goal_action = np.asarray(loc.get("goal_action", []), dtype=np.float32)
                high_ret = np.asarray(loc.get("high_ret", []), dtype=np.float32).reshape(-1)
                next_high_obs = np.asarray(loc.get("next_high_obs", []), dtype=np.float32)
                done_env = np.asarray(loc.get("done", False), dtype=bool).reshape(-1)
                seg_id = np.asarray(loc.get("seg_id", np.full_like(done_low, -1)), dtype=np.int64).reshape(-1)

                if (
                    high_obs_start.ndim == 2
                    and goal_action.ndim == 2
                    and next_high_obs.ndim == 2
                ):
                    max_rows = min(
                        done_low.size,
                        replay_mask.size,
                        high_obs_start.shape[0],
                        goal_action.shape[0],
                        next_high_obs.shape[0],
                    )
                    for env_i in self._csv_env_indices(max_rows):
                        if not (bool(done_low[env_i]) and bool(replay_mask[env_i]) and self._capture_active(env_i)):
                            continue
                        comp_seq = self._hi_comp_seq_by_env.get(env_i, {k: [] for k in self.high_comp_keys})
                        comp_sums = {k: float(np.sum(np.asarray(comp_seq.get(k, []), dtype=np.float32))) for k in self.high_comp_keys}
                        comp_sum_total = float(sum(comp_sums.values()))

                        row_h = [
                            int(getattr(self.model, "num_timesteps", 0)),
                            int(env_i),
                            int(seg_id[env_i]) if seg_id.size > env_i else -1,
                            self._json_arr(high_obs_start[env_i]),
                            self._json_arr(goal_action[env_i]),
                            float(high_ret[env_i]) if high_ret.size > env_i else 0.0,
                            comp_sum_total,
                            comp_sums["collision_reward"],
                            comp_sums["progress_reward"],
                            comp_sums["speed_ref_aux_reward"],
                            comp_sums["comfort_reward_for_high"],
                            comp_sums["lane_change_reward"],
                            comp_sums["goal_lane_dense_reward"],
                            comp_sums["punctual_reward"],
                            comp_sums["wrong_lane_terminal_penalty"],
                            self._json_arr(comp_seq.get("collision_reward", [])),
                            self._json_arr(comp_seq.get("progress_reward", [])),
                            self._json_arr(comp_seq.get("speed_ref_aux_reward", [])),
                            self._json_arr(comp_seq.get("comfort_reward_for_high", [])),
                            self._json_arr(comp_seq.get("lane_change_reward", [])),
                            self._json_arr(comp_seq.get("goal_lane_dense_reward", [])),
                            self._json_arr(comp_seq.get("punctual_reward", [])),
                            self._json_arr(comp_seq.get("wrong_lane_terminal_penalty", [])),
                            self._json_arr(self._hi_acc_seq_by_env.get(env_i, [])),
                            self._json_arr(next_high_obs[env_i]),
                            int(done_env[env_i]) if done_env.size > env_i else 0,
                        ]
                        self._append_csv(self.high_transition_csv_path, row_h)

                        if self._diagnostic_csv_enabled:
                            low_len_arr = np.asarray(loc.get("low_len", []), dtype=np.int32).reshape(-1)
                            low_ext_sum = float(np.sum(np.asarray(self._hi_low_ext_seq_by_env.get(env_i, []), dtype=np.float32)))
                            intrinsic_sum = float(np.sum(np.asarray(self._hi_intrinsic_seq_by_env.get(env_i, []), dtype=np.float32)))
                            low_total_sum = float(np.sum(np.asarray(self._hi_low_total_seq_by_env.get(env_i, []), dtype=np.float32)))
                            acc_seq = np.asarray(self._hi_acc_seq_by_env.get(env_i, []), dtype=np.float32)
                            diag_i = infos[env_i].get("env_diagnostics", {}) if infos and len(infos) > env_i and isinstance(infos[env_i], dict) else {}
                            health_row = [
                                int(getattr(self.model, "num_timesteps", 0)),
                                int(env_i),
                                int(seg_id[env_i]) if seg_id.size > env_i else -1,
                                int(low_len_arr[env_i]) if low_len_arr.size > env_i else int(acc_seq.size),
                                int(done_env[env_i]) if done_env.size > env_i else 0,
                                int(done_low[env_i]) if done_low.size > env_i else 0,
                                float(high_ret[env_i]) if high_ret.size > env_i else 0.0,
                                comp_sum_total,
                                (float(high_ret[env_i]) if high_ret.size > env_i else 0.0) - comp_sum_total,
                                low_ext_sum,
                                intrinsic_sum,
                                low_total_sum,
                                float(np.min(acc_seq)) if acc_seq.size else np.nan,
                                float(np.max(acc_seq)) if acc_seq.size else np.nan,
                            ]
                            health_row.extend(self._diag_value(diag_i, k) for k in getattr(self, "env_diag_fields", []))
                            self._append_csv(self.transition_health_csv_path, health_row)
                        self._reset_hi_buffers(env_i)

            # Switch capture window at each captured env episode boundary.
            if dones.size > 0:
                for env_i in self._csv_env_indices(min(dones.size, replay_mask.size)):
                    if not bool(dones[env_i]):
                        continue
                    if not hasattr(self, "_high_episode_idx") or env_i >= self._high_episode_idx.size:
                        continue
                    self._high_episode_idx[env_i] += 1
                    self._high_transition_capture_active[env_i] = (
                        self.high_transition_csv_all > 0
                        and (int(self._high_episode_idx[env_i]) % self.high_transition_csv_all) == 0
                    )
                    if env_i == 0:
                        self._env0_high_episode_idx = int(self._high_episode_idx[env_i])

        # --- Episode End Logic ---
        dones_real = dones & replay_mask
        if dones_real.any():
            idx = np.flatnonzero(dones_real)
            self._episode_counter += int(idx.size)

            if self._diagnostic_csv_enabled:
                for env_i in idx:
                    info_i = infos[int(env_i)] if infos and len(infos) > int(env_i) and isinstance(infos[int(env_i)], dict) else {}
                    diag_i = info_i.get("env_diagnostics", {}) if isinstance(info_i, dict) else {}
                    terminal_signal = info_i.get("terminal_signal_features", "")
                    row_ep = [
                        int(getattr(self.model, "num_timesteps", 0)),
                        int(env_i),
                        int(self._episode_counter),
                        float(self._ep_ret[int(env_i)]),
                        int(self._ep_len[int(env_i)]),
                        int(dones[int(env_i)]) if dones.size > int(env_i) else 0,
                        int("terminal_observation" in info_i),
                        json.dumps(list(terminal_signal), ensure_ascii=True) if isinstance(terminal_signal, (list, tuple, np.ndarray)) else str(terminal_signal),
                    ]
                    row_ep.extend(self._diag_value(diag_i, k) for k in getattr(self, "env_diag_fields", []))
                    self._append_csv(self.episode_diag_csv_path, row_ep)
            
            # Only log environment-level stats if high_logger exists
            if hasattr(self.model, "high_logger"):
                buffers = self._high_buffers
                self._record_smooth(self.model.high_logger, buffers, "rollout/ep_rew", float(self._ep_ret[idx].mean()))
                self._record_smooth(self.model.high_logger, buffers, "rollout/ep_len", float(self._ep_len[idx].mean()))
                for name, arr in self._ep_comp_sums.items():
                    self._record_smooth(self.model.high_logger, buffers, f"rollout/{name}", float(arr[idx].mean()))
                
                if self._episode_counter - self._last_dump_high >= self.high_log_interval_episodes:
                    self.model.high_logger.dump(step=self.model.num_timesteps)
                    self._last_dump_high = self._episode_counter

            # Reset TB buffers
            self._ep_ret[idx] = 0.0
            self._ep_len[idx] = 0
            for arr in self._ep_comp_sums.values():
                arr[idx] = 0.0

        return True


class HIROCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_dir: str, prefix: str = "hiro", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_dir = save_dir
        self.prefix = prefix
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            train_mode = getattr(self.model.cfg, "train_mode", "joint")
            if train_mode != "low_only":
                high_path = os.path.join(self.save_dir, f"{self.prefix}_high_step_{self.num_timesteps}.zip")
                self.model.high_agent.save(high_path)

            if train_mode != "high_only":
                low_path = os.path.join(self.save_dir, f"{self.prefix}_low_step_{self.num_timesteps}.zip")
                self.model.low_agent.save(low_path)
            if self.verbose:
                print(f"[Checkpoint] Saved HIRO models at step={self.num_timesteps}")
        return True


class HIROLowEpisodeTrajectoryCallback(BaseCallback):
    """Record full low-episode trajectories for all envs to JSONL."""

    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path

    def _append_jsonl(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=True) + "\n"
        try:
            with open(self.save_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line)
        except OSError as e:
            # Some Windows environments can intermittently raise Errno 22 for text writes.
            if getattr(e, "errno", None) != 22:
                raise
            with open(self.save_path, "ab") as f:
                f.write(line.encode("utf-8", errors="replace"))
            if self.verbose:
                print(f"[HIROLowEpisodeTrajectoryCallback] Fallback binary append due to OSError(22): {e}")

    def _on_training_start(self) -> None:
        n_envs = int(getattr(self.model, "n_envs", 1))
        self._episode_seq = np.zeros(n_envs, dtype=np.int64)
        self._ep_ret = np.zeros(n_envs, dtype=np.float32)
        self._ep_len = np.zeros(n_envs, dtype=np.int32)
        self._episode_steps: List[List[Dict[str, Any]]] = [[] for _ in range(n_envs)]
        self._episode_meta: List[Dict[str, Any]] = [
            {
                "episode_id": int(i),
                "env_id": int(i),
                "segment_id": -1,
                "global_step_start": int(getattr(self.model, "num_timesteps", 0)),
            }
            for i in range(n_envs)
        ]
        save_dir = os.path.dirname(self.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        loc = self.locals

        c = np.asarray(loc.get("c", []), dtype=np.int32).reshape(-1)
        if c.size == 0:
            return True
        replay_mask = np.asarray(loc.get("replay_mask", np.ones_like(c, dtype=bool)), dtype=bool).reshape(-1)
        if replay_mask.size != c.size:
            replay_mask = np.ones_like(c, dtype=bool)
        if not replay_mask.any():
            return True

        low_obs = np.asarray(loc.get("low_obs", []), dtype=np.float32)
        low_action = np.asarray(loc.get("low_action", []), dtype=np.float32)
        low_buffer_action = np.asarray(loc.get("low_buffer_action", []), dtype=np.float32)
        next_low_obs = np.asarray(loc.get("next_low_obs", []), dtype=np.float32)
        reward_env = np.asarray(loc.get("reward_env", []), dtype=np.float32).reshape(-1)
        low_reward_ext = np.asarray(loc.get("low_reward_ext", []), dtype=np.float32).reshape(-1)
        intrinsic = np.asarray(loc.get("intrinsic", []), dtype=np.float32).reshape(-1)
        low_reward_total = np.asarray(loc.get("low_reward_total", []), dtype=np.float32).reshape(-1)
        done = np.asarray(loc.get("done", False), dtype=bool).reshape(-1)
        done_low = np.asarray(loc.get("done_low", False), dtype=bool).reshape(-1)
        ego_now_sub = np.asarray(loc.get("ego_now_sub", []), dtype=np.float32)
        ego_next_sub = np.asarray(loc.get("ego_next_sub", []), dtype=np.float32)

        seg_id = np.asarray(loc.get("seg_id", np.full_like(c, -1)), dtype=np.int64).reshape(-1)
        goal_action = np.asarray(loc.get("goal_action", []), dtype=np.float32)
        goal_phys = np.asarray(loc.get("goal_phys", []), dtype=np.float32)
        ego_start = np.asarray(loc.get("ego_start", []), dtype=np.float32)
        goal_err_all = np.asarray(loc.get("goal_err_all", []), dtype=np.float32)
        intrinsic_unweighted = np.asarray(loc.get("intrinsic_unweighted", []), dtype=np.float32).reshape(-1)

        n_envs = int(c.size)
        if low_obs.ndim != 2 or low_obs.shape[0] != n_envs:
            return True

        idx_start = np.flatnonzero((c == 0) & replay_mask)
        for i in idx_start:
            ii = int(i)
            self._ep_ret[ii] = 0.0
            self._ep_len[ii] = 0
            self._episode_steps[ii] = []

            meta = {
                "episode_id": int(self._episode_seq[ii]),
                "env_id": ii,
                "segment_id": int(seg_id[ii]) if seg_id.size > ii else -1,
                "global_step_start": int(getattr(self.model, "num_timesteps", 0)),
            }
            if goal_action.ndim == 2 and goal_action.shape[0] > ii:
                meta["goal_action"] = np.asarray(goal_action[ii], dtype=np.float32).reshape(-1).tolist()
            if goal_phys.ndim == 2 and goal_phys.shape[0] > ii:
                meta["goal_phys"] = np.asarray(goal_phys[ii], dtype=np.float32).reshape(-1).tolist()
            if ego_start.ndim == 2 and ego_start.shape[0] > ii:
                meta["ego_start"] = np.asarray(ego_start[ii], dtype=np.float32).reshape(-1).tolist()
            self._episode_meta[ii] = meta

        for i in range(n_envs):
            ii = int(i)
            if not replay_mask[ii]:
                continue
            r_tot = float(low_reward_total[ii]) if low_reward_total.size > ii else 0.0
            self._ep_ret[ii] += r_tot
            self._ep_len[ii] += 1

            row = {
                "t_in_episode": int(c[ii]),
                "global_step": int(getattr(self.model, "num_timesteps", 0)),
                "low_obs": np.asarray(low_obs[ii], dtype=np.float32).reshape(-1).tolist(),
                "action": np.asarray(low_action[ii], dtype=np.float32).reshape(-1).tolist() if low_action.ndim == 2 and low_action.shape[0] > ii else [],
                "buffer_action": np.asarray(low_buffer_action[ii], dtype=np.float32).reshape(-1).tolist() if low_buffer_action.ndim == 2 and low_buffer_action.shape[0] > ii else [],
                "reward_env": float(reward_env[ii]) if reward_env.size > ii else 0.0,
                "reward_ext": float(low_reward_ext[ii]) if low_reward_ext.size > ii else 0.0,
                "reward_intrinsic": float(intrinsic[ii]) if intrinsic.size > ii else 0.0,
                "reward_total": r_tot,
                "done_env": bool(done[ii]) if done.size > ii else False,
                "done_low": bool(done_low[ii]) if done_low.size > ii else False,
                "next_low_obs": np.asarray(next_low_obs[ii], dtype=np.float32).reshape(-1).tolist() if next_low_obs.ndim == 2 and next_low_obs.shape[0] > ii else [],
                "ego_now_sub": np.asarray(ego_now_sub[ii], dtype=np.float32).reshape(-1).tolist() if ego_now_sub.ndim == 2 and ego_now_sub.shape[0] > ii else [],
                "ego_next_sub": np.asarray(ego_next_sub[ii], dtype=np.float32).reshape(-1).tolist() if ego_next_sub.ndim == 2 and ego_next_sub.shape[0] > ii else [],
            }
            self._episode_steps[ii].append(row)

        idx_end = np.flatnonzero(done_low & replay_mask)
        if idx_end.size:
            for j in idx_end:
                jj = int(j)
                payload = {
                    "episode": self._episode_meta[jj],
                    "length": int(self._ep_len[jj]),
                    "return": float(self._ep_ret[jj]),
                    "steps": self._episode_steps[jj],
                }
                if goal_err_all.ndim == 2 and goal_err_all.shape[0] > jj:
                    payload["goal_err"] = np.asarray(goal_err_all[jj], dtype=np.float32).reshape(-1).tolist()
                if intrinsic_unweighted.size > jj:
                    payload["intrinsic_unweighted"] = float(intrinsic_unweighted[jj])

                self._append_jsonl(payload)
                self._episode_seq[jj] += 1
                self._episode_steps[jj] = []
                self._ep_ret[jj] = 0.0
                self._ep_len[jj] = 0

        return True

