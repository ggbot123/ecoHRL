# rl/algos/hiro/hiro_callbacks.py
from __future__ import annotations

import os
import csv
from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class HIROLoggingCallback(BaseCallback):
    """VecEnv-friendly logging callback for HIRO.

    - High-level: logs true env episode stats (done from VecEnv) aggregated across envs.
    - Low-level: logs low-episode (= one high_interval or early termination) stats for envs that finished a low-episode at the current step.

    Expect locals provided by HIROSAC.learn:
    - reward_env: np.ndarray (n_envs,)
    - episode_end: np.ndarray (n_envs,) bool
    - infos: list[dict] length n_envs
    - low_ret: np.ndarray (k,) for finished low-episodes at this step
    - low_len: np.ndarray (k,)
    - low_comp_sums: dict[str, np.ndarray (k,)]
    - goal_err: np.ndarray (k, ego_dim) for finished low-episodes at this step
        Signed tracking error at the end of each high-interval:
        goal_err = ego_next_rel - goal_rel.
        With the default ego_dim==4, components correspond to (x, y, vx, vy).
    - intrinsic_unweighted: np.ndarray (k,)
        intrinsic / intrinsic_coef at the same boundary.
    """

    def __init__(self, high_log_interval_episodes: int = 1, low_log_interval_hi: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.high_log_interval_episodes = int(high_log_interval_episodes)
        self.low_log_interval_hi = int(low_log_interval_hi)
        self._episode_counter = 0
        self._rollout_counter = 0
        self._last_dump_high = 0
        self._last_dump_low = 0
        self._high_buffers, self._low_buffers = {}, {}

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

        self._rollout_counter += int(low_ret.size)
        self._record_smooth(self.model.low_logger, self._low_buffers, "rollout/ep_rew", float(low_ret.mean()))
        self._record_smooth(self.model.low_logger, self._low_buffers, "rollout/ep_len", float(low_len.mean()) if low_len.size else 0.0)
        for k, v in low_comp_sums.items():
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            if arr.size:
                self._record_smooth(self.model.low_logger, self._low_buffers, f"rollout/{k}", float(arr.mean()))

        # goal tracking error at the end of each high-interval
        goal_err = np.asarray(loc.get("goal_err", []), dtype=np.float32)
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

    def _on_step(self) -> bool:
        loc = self.locals
        reward_env = np.asarray(loc.get("reward_env", 0.0), dtype=np.float32).reshape(-1)
        episode_end = np.asarray(loc.get("episode_end", False), dtype=bool).reshape(-1)
        infos = loc.get("infos", [])

        if reward_env.size:
            self._ep_ret += reward_env
            self._ep_len += 1

        if infos:
            for i, info in enumerate(infos):
                rc = info.get("reward_components", {})
                for name, val in rc.items():
                    self._ep_comp_sums.setdefault(name, np.zeros_like(self._ep_ret))[i] += float(val)

        if episode_end.any():
            idx = np.flatnonzero(episode_end)
            self._episode_counter += int(idx.size)

            self._record_smooth(self.model.high_logger, self._high_buffers, "rollout/ep_rew", float(self._ep_ret[idx].mean()))
            self._record_smooth(self.model.high_logger, self._high_buffers, "rollout/ep_len", float(self._ep_len[idx].mean()))
            for name, arr in self._ep_comp_sums.items():
                self._record_smooth(self.model.high_logger, self._high_buffers, f"rollout/{name}", float(arr[idx].mean()))

            self._ep_ret[idx] = 0.0
            self._ep_len[idx] = 0
            for arr in self._ep_comp_sums.values():
                arr[idx] = 0.0

            if self._episode_counter - self._last_dump_high >= self.high_log_interval_episodes:
                self.model.high_logger.dump(step=self.model.num_timesteps)
                self._last_dump_high = self._episode_counter

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


class HIROTrajectoryLogger(BaseCallback):
    def __init__(self, log_freq_episodes: int, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = int(log_freq_episodes)
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.ep_count = 0
        self.current_ep_high_traj = []
        self.current_ep_low_traj = []
        self.logging_active = False
        self.low_traj_recorded = False

        # CSV Headers
        self.high_header = ["episode", "step", "start_x", "start_y", "start_vx", "start_vy", 
                            "goal_x", "goal_y", "goal_vx", "goal_vy", 
                            "end_x", "end_y", "end_vx", "end_vy", "intrinsic_reward"]
        self.low_header = ["episode", "step_in_episode", "x", "y", "vx", "vy", 
                           "action_acc", "action_steer", "reward"]

        # Initialize files with headers if they don't exist
        self.high_csv_path = os.path.join(self.save_dir, "high_traj.csv")
        self.low_csv_path = os.path.join(self.save_dir, "low_traj.csv")
        
        if not os.path.exists(self.high_csv_path):
            with open(self.high_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self.high_header)
        
        if not os.path.exists(self.low_csv_path):
            with open(self.low_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self.low_header)

    def _on_step(self) -> bool:
        # We only track env 0
        loc = self.locals
        
        # Check if episode ended for env 0
        done = loc.get("done")[0]
        
        # Check if we should be logging this episode
        if self.ep_count % self.log_freq == 0:
            self.logging_active = True
        else:
            self.logging_active = False

        if self.logging_active:
            # Collect Low Trajectory
            # We want to record ONE high interval's low trajectory.
            # Let's record the FIRST interval of the episode.
            
            # If we haven't recorded low traj yet, we record every step until done_low[0] is True.
            if not self.low_traj_recorded:
                ego_idx = self.model.ego_feature_idx
                kin_next = loc.get("kin_next")[0]
                # Assuming ego is vehicle 0
                ego_state = kin_next[0][ego_idx]
                
                # Low action
                low_action = loc.get("low_action")[0]
                reward = loc.get("reward")[0]
                
                # Store
                self.current_ep_low_traj.append({
                    "episode": self.ep_count,
                    "step_in_episode": len(self.current_ep_low_traj),
                    "x": float(ego_state[0]),
                    "y": float(ego_state[1]),
                    "vx": float(ego_state[2]),
                    "vy": float(ego_state[3]),
                    "action_acc": float(low_action[0]),
                    "action_steer": float(low_action[1]),
                    "reward": float(reward)
                })
                
                if loc.get("done_low")[0]:
                    self.low_traj_recorded = True

            # Collect High Trajectory (at end of interval)
            if loc.get("done_low")[0]:
                # Get interval data
                ego_start = loc.get("ego_start")[0]
                goal_phys = loc.get("goal_phys")[0]
                intrinsic = loc.get("intrinsic")[0]
                
                # End state
                ego_idx = self.model.ego_feature_idx
                kin_next = loc.get("kin_next")[0]
                ego_end = kin_next[0][ego_idx]
                
                self.current_ep_high_traj.append({
                    "episode": self.ep_count,
                    "step": len(self.current_ep_high_traj),
                    "start_x": float(ego_start[0]),
                    "start_y": float(ego_start[1]),
                    "start_vx": float(ego_start[2]),
                    "start_vy": float(ego_start[3]),
                    "goal_x": float(goal_phys[0]),
                    "goal_y": float(goal_phys[1]),
                    "goal_vx": float(goal_phys[2]),
                    "goal_vy": float(goal_phys[3]),
                    "end_x": float(ego_end[0]),
                    "end_y": float(ego_end[1]),
                    "end_vx": float(ego_end[2]),
                    "end_vy": float(ego_end[3]),
                    "intrinsic_reward": float(intrinsic)
                })

        if done:
            if self.logging_active:
                # Write to CSV
                if self.current_ep_high_traj:
                    with open(self.high_csv_path, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=self.high_header)
                        writer.writerows(self.current_ep_high_traj)
                
                if self.current_ep_low_traj:
                    with open(self.low_csv_path, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=self.low_header)
                        writer.writerows(self.current_ep_low_traj)
            
            # Reset for next episode
            self.ep_count += 1
            self.current_ep_high_traj = []
            self.current_ep_low_traj = []
            self.low_traj_recorded = False
            
        return True
