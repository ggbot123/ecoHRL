# rl/algos/hiro/hiro_callbacks.py
from __future__ import annotations

import os
import csv
from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class HIROLoggingCallback(BaseCallback):
    """Unified logging callback for HIRO.

    - TensorBoard:
        - High-level: logs true env episode stats aggregated across envs.
        - Low-level: logs low-episode (high_interval) stats.
    - CSV (optional):
        - Logs high-level and low-level trajectories for env 0 periodically.
    """

    def __init__(
        self,
        high_log_interval_episodes: int = 1,
        low_log_interval_hi: int = 1,
        verbose: int = 0,
        # CSV Logging args
        csv_log_freq_episodes: int = 0,  # 0 to disable
        csv_save_dir: str | None = None,
    ):
        super().__init__(verbose)
        self.high_log_interval_episodes = int(high_log_interval_episodes)
        self.low_log_interval_hi = int(low_log_interval_hi)
        
        self.csv_log_freq = int(csv_log_freq_episodes)
        self.csv_save_dir = csv_save_dir
        self.csv_active = False  # Whether current episode is being logged to CSV (env 0)
        self.csv_low_traj_recorded = False # Only record first interval's low traj per logged episode
        
        self._episode_counter = 0     # Counts finished episodes (across all envs)
        self._env0_ep_count = 0
        
        self._rollout_counter = 0
        self._last_dump_high = 0
        self._last_dump_low = 0
        self._high_buffers, self._low_buffers = {}, {}

        # CSV Headers
        self.comp_keys = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "punctual_reward"]
        if self.csv_log_freq > 0 and self.csv_save_dir:
            os.makedirs(self.csv_save_dir, exist_ok=True)
            self.high_csv_path = os.path.join(self.csv_save_dir, "high_traj.csv")
            self.low_csv_path = os.path.join(self.csv_save_dir, "low_traj.csv")
            
            base_header = ["episode", "step", "s", "a", "r", "next_s", "done"]
            comp_headers = [f"comp_{k}" for k in self.comp_keys]
            self._init_csv(self.high_csv_path, base_header + comp_headers)
            self._init_csv(self.low_csv_path, base_header + comp_headers)

    def _init_csv(self, path, header):
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def _fmt_arr(self, arr):
        return np.array2string(np.asarray(arr).reshape(-1), separator=',', max_line_width=np.inf).replace('\n', '')

    def _append_csv(self, path, row):
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def _on_training_start(self) -> None:
        n_envs = int(getattr(self.model, "n_envs", 1))
        self._ep_ret = np.zeros(n_envs, dtype=np.float32)
        self._ep_len = np.zeros(n_envs, dtype=np.int32)
        self._ep_comp_sums: dict[str, np.ndarray] = {}

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
        low_comp_sums = loc.get("low_comp_sums", {})

        # --- TensorBoard Logging ---
        self._rollout_counter += int(low_ret.size)
        self._record_smooth(self.model.low_logger, self._low_buffers, "rollout/ep_rew", float(low_ret.mean()))
        self._record_smooth(self.model.low_logger, self._low_buffers, "rollout/ep_len", float(low_len.mean()) if low_len.size else 0.0)
        for k, v in low_comp_sums.items():
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            if arr.size:
                self._record_smooth(self.model.low_logger, self._low_buffers, f"rollout/{k}", float(arr.mean()))

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
            # Env 0 finished a high interval
            high_obs_start = loc.get("high_obs_start")[0]
            goal_action = loc.get("goal_action")[0]
            high_ret = loc.get("high_ret")[0]
            next_high_obs = loc.get("next_high_obs")[0]
            done = loc.get("done")[0]

            row_high = [
                self._env0_ep_count,
                self.model.num_timesteps,
                self._fmt_arr(high_obs_start),
                self._fmt_arr(goal_action),
                float(high_ret),
                self._fmt_arr(next_high_obs),
                int(done)
            ]
            
            for k in self.comp_keys:
                arr = low_comp_sums.get(k, np.array([]))
                val = float(arr[0]) if arr.size > 0 else 0.0
                row_high.append(val)
            
            self._append_csv(self.high_csv_path, row_high)
            
            # If we just finished the first interval, stop recording low traj
            self.csv_low_traj_recorded = True

    def _on_step(self) -> bool:
        loc = self.locals
        reward_env = np.asarray(loc.get("reward_env", 0.0), dtype=np.float32).reshape(-1)
        episode_end = np.asarray(loc.get("episode_end", False), dtype=bool).reshape(-1)
        infos = loc.get("infos", [])

        # --- Update TB stats & Capture env 0 components ---
        rc_env0 = {}
        if reward_env.size:
            self._ep_ret += reward_env
            self._ep_len += 1

        if infos:
            for i, info in enumerate(infos):
                rc = info.get("reward_components", {})
                for name, val in rc.items():
                    self._ep_comp_sums.setdefault(name, np.zeros_like(self._ep_ret))[i] += float(val)
                if i == 0 and self.csv_active:
                    rc_env0 = rc

        # --- CSV Logs (Low Level Step) ---
        # Record only if active and not yet done with first interval
        if self.csv_active and not self.csv_low_traj_recorded:
            low_obs = loc.get("low_obs")[0]
            low_action = loc.get("low_action")[0]
            low_reward = loc.get("low_reward_total")[0]
            next_low_obs = loc.get("next_low_obs")[0]
            done_low = loc.get("done_low")[0]
            
            row = [
                self._env0_ep_count,
                self.model.num_timesteps,
                self._fmt_arr(low_obs),
                self._fmt_arr(low_action),
                float(low_reward),
                self._fmt_arr(next_low_obs),
                int(done_low)
            ]
            for k in self.comp_keys:
                row.append(float(rc_env0.get(k, 0.0)))
            self._append_csv(self.low_csv_path, row)

        # --- Episode End Logic ---
        if episode_end.any():
            idx = np.flatnonzero(episode_end)
            self._episode_counter += int(idx.size)

            # TB Logging
            self._record_smooth(self.model.high_logger, self._high_buffers, "rollout/ep_rew", float(self._ep_ret[idx].mean()))
            self._record_smooth(self.model.high_logger, self._high_buffers, "rollout/ep_len", float(self._ep_len[idx].mean()))
            for name, arr in self._ep_comp_sums.items():
                self._record_smooth(self.model.high_logger, self._high_buffers, f"rollout/{name}", float(arr[idx].mean()))

            # Reset TB buffers
            self._ep_ret[idx] = 0.0
            self._ep_len[idx] = 0
            for arr in self._ep_comp_sums.values():
                arr[idx] = 0.0

            if self._episode_counter - self._last_dump_high >= self.high_log_interval_episodes:
                self.model.high_logger.dump(step=self.model.num_timesteps)
                self._last_dump_high = self._episode_counter
            
            # CSV Episode Counter Logic (Env 0)
            if episode_end[0]:
                self._env0_ep_count += 1
                # Determines if NEXT episode should be logged
                if self.csv_log_freq > 0 and (self._env0_ep_count % self.csv_log_freq == 0):
                    self.csv_active = True
                    self.csv_low_traj_recorded = False # Reset for new episode
                else:
                    self.csv_active = False

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
            high_path = os.path.join(self.save_dir, f"{self.prefix}_high_step_{self.num_timesteps}.zip")
            low_path = os.path.join(self.save_dir, f"{self.prefix}_low_step_{self.num_timesteps}.zip")
            self.model.high_agent.save(high_path)
            self.model.low_agent.save(low_path)
            if self.verbose:
                print(f"[Checkpoint] Saved HIRO models at step={self.num_timesteps}")
        return True

