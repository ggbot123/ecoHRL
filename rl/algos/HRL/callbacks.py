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
    ):
        super().__init__(verbose)
        self.high_log_interval_episodes = int(high_log_interval_episodes)
        self.low_log_interval_hi = int(low_log_interval_hi)
        
        self.csv_log_freq = int(csv_log_freq_episodes)
        self.csv_save_dir = csv_save_dir
        self.mpc_fail_csv_path = mpc_fail_csv_path
        self.low_obs_csv_interval_hi = max(1, int(low_obs_csv_interval_hi))
        self.low_obs_csv_env0_only = bool(low_obs_csv_env0_only)

        self._traj_csv_enabled = self.csv_log_freq > 0 and bool(self.csv_save_dir)
        self._low_obs_csv_enabled = bool(self.csv_save_dir)

        self.csv_active = False  # Whether current episode is being logged to CSV (env 0)
        self.csv_low_traj_recorded = False # Only record first interval's low traj per logged episode
        self._hi_start_seen = 0
        self._hi_start_saved = 0
        
        self._episode_counter = 0     # Counts finished episodes (across all envs)
        self._env0_ep_count = 0
        
        self._rollout_counter = 0
        self._last_dump_high = 0
        self._last_dump_low = 0
        self._high_buffers, self._low_buffers = {}, {}

        # CSV Headers
        self.comp_keys = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "punctual_reward"]
        self.ego_keys = ["ego_x", "ego_y", "ego_vx", "ego_vy"]

        if self._traj_csv_enabled or self._low_obs_csv_enabled:
            os.makedirs(self.csv_save_dir, exist_ok=True)

        if self._traj_csv_enabled:
            self.high_csv_path = os.path.join(self.csv_save_dir, "high_traj.csv")
            self.low_csv_path = os.path.join(self.csv_save_dir, "low_traj.csv")
            
            base_header = ["episode", "step", "s", "a", "r", "next_s", "done"] + self.ego_keys
            comp_headers = [f"comp_{k}" for k in self.comp_keys]
            self._init_csv(self.high_csv_path, base_header + comp_headers)
            self._init_csv(self.low_csv_path, base_header + comp_headers)

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

    def _init_csv(self, path, header):
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def _fmt_arr(self, arr):
        # Format array as string with 2 decimal places
        arr = np.asarray(arr).reshape(-1)
        return np.array2string(
            arr, 
            separator=',', 
            max_line_width=np.inf, 
            formatter={'float_kind': lambda x: f"{x:.2f}"}
        ).replace('\n', '')

    def _append_csv(self, path, row):
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    @staticmethod
    def _json_arr(arr) -> str:
        a = np.asarray(arr, dtype=np.float32)
        return json.dumps(a.tolist(), ensure_ascii=True)

    @staticmethod
    def _safe_norm_to_dx(norm_val: np.ndarray, dx_low: float, dx_high: float) -> np.ndarray:
        n = np.asarray(norm_val, dtype=np.float32)
        return ((n + 1.0) * 0.5 * float(dx_high - dx_low) + float(dx_low)).astype(np.float32)

    def _on_training_start(self) -> None:
        n_envs = int(getattr(self.model, "n_envs", 1))
        self._ep_ret = np.zeros(n_envs, dtype=np.float32)
        self._ep_len = np.zeros(n_envs, dtype=np.int32)
        self._ep_comp_sums: dict[str, np.ndarray] = {}

        if self.mpc_fail_csv_path:
            os.makedirs(os.path.dirname(self.mpc_fail_csv_path), exist_ok=True)
            if not os.path.exists(self.mpc_fail_csv_path):
                with open(self.mpc_fail_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
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
                    writer.writeheader()

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
        reward_env = np.asarray(loc.get("reward_env", 0.0), dtype=np.float32).reshape(-1)
        dones = np.asarray(loc.get("done", False), dtype=bool).reshape(-1)
        infos = loc.get("infos", [])
        replay_mask = np.asarray(loc.get("replay_mask", np.ones_like(dones, dtype=bool)), dtype=bool).reshape(-1)
        if replay_mask.size != dones.size:
            replay_mask = np.ones_like(dones, dtype=bool)
        if not replay_mask.any():
            return True

        if self.mpc_fail_csv_path:
            fail_records = list(loc.get("mpc_fail_records", []) or [])
            if fail_records:
                with open(self.mpc_fail_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
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
                    for rec in fail_records:
                        writer.writerow(
                            {
                                "global_step": int(rec.get("global_step", 0)),
                                "env_id": int(rec.get("env_id", -1)),
                                "segment_id": int(rec.get("segment_id", -1)),
                                "planner": str(rec.get("planner", "")),
                                "message": str(rec.get("message", "")),
                                "goal_phys": json.dumps(rec.get("goal_phys", []), ensure_ascii=True),
                                "ego_sub_now": json.dumps(rec.get("ego_sub_now", []), ensure_ascii=True),
                                "low_obs": json.dumps(rec.get("low_obs", []), ensure_ascii=True),
                            }
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

        # --- Episode End Logic ---
        dones_real = dones & replay_mask
        if dones_real.any():
            idx = np.flatnonzero(dones_real)
            self._episode_counter += int(idx.size)
            
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
            if getattr(self.model.cfg, "train_mode", "joint") != "low_only":
                high_path = os.path.join(self.save_dir, f"{self.prefix}_high_step_{self.num_timesteps}.zip")
                self.model.high_agent.save(high_path)
            
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

