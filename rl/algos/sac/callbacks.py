from collections import deque
import csv
import json
import os
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class RewardComponentsTensorboardCallback(BaseCallback):
    """
    Logs reward components and episode stats with smoothing, similar to HIROLoggingCallback.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._buffers = {}
        self._ep_ret = None
        self._ep_len = None
        self._ep_comp_sums = {}

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self._ep_ret = np.zeros(n_envs, dtype=np.float32)
        self._ep_len = np.zeros(n_envs, dtype=np.int32)
        self._ep_comp_sums = {}

    @staticmethod
    def _record_smooth(logger, buffers: dict, tag: str, value: float, window: int = 50):
        buf = buffers.setdefault(tag, deque(maxlen=window))
        buf.append(float(value))
        logger.record(tag, float(sum(buf) / len(buf)))

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if rewards is None or dones is None:
            return True

        n_envs = len(rewards)
        # Initialize if needed (e.g. if _on_training_start wasn't called properly or env changed)
        if self._ep_ret is None or len(self._ep_ret) != n_envs:
            self._ep_ret = np.zeros(n_envs, dtype=np.float32)
            self._ep_len = np.zeros(n_envs, dtype=np.int32)
            self._ep_comp_sums = {}

        replay_mask = np.asarray(
            [
                not bool(info.get("skip_replay", False))
                if isinstance(info, dict)
                else True
                for info in (infos or [{} for _ in range(n_envs)])
            ],
            dtype=bool,
        )

        # Accumulate only transitions eligible for replay/training.
        self._ep_ret[replay_mask] += rewards[replay_mask]
        self._ep_len[replay_mask] += 1

        # Accumulate components
        if infos:
            for i, info in enumerate(infos):
                if not replay_mask[i]:
                    continue
                rc = info.get("reward_components", {})
                for name, val in rc.items():
                    self._ep_comp_sums.setdefault(name, np.zeros(n_envs, dtype=np.float32))[i] += float(val)

        # Process finished episodes
        if dones.any():
            idx = np.flatnonzero(dones)
            
            # Log smoothed stats
            self._record_smooth(self.logger, self._buffers, "rollout/ep_rew", float(self._ep_ret[idx].mean()))
            self._record_smooth(self.logger, self._buffers, "rollout/ep_len", float(self._ep_len[idx].mean()))
            
            for name, arr in self._ep_comp_sums.items():
                self._record_smooth(self.logger, self._buffers, f"rollout/{name}", float(arr[idx].mean()))

            # Reset stats for finished envs
            self._ep_ret[idx] = 0.0
            self._ep_len[idx] = 0
            for arr in self._ep_comp_sums.values():
                arr[idx] = 0.0

        return True


class SACTransitionLoggingCallback(BaseCallback):
    """Write standalone SAC replay transitions and episode summaries to CSV."""

    def __init__(
        self,
        save_dir: str,
        episode_freq: int = 1,
        envs: str = "env0",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_dir = str(save_dir)
        self.episode_freq = max(1, int(episode_freq))
        self.envs = str(envs).strip().lower()
        if self.envs not in {"env0", "all"}:
            raise ValueError("SAC transition CSV envs must be 'env0' or 'all'")

        self.transition_path = os.path.join(self.save_dir, "sac_replay_transitions.csv")
        self.episode_path = os.path.join(self.save_dir, "sac_episode_stats.csv")
        self._episode_idx = None
        self._episode_step = None
        self._episode_return = None
        self._component_sums: list[dict[str, float]] = []

    @staticmethod
    def _json_value(value: Any) -> str:
        def _default(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            return str(obj)

        return json.dumps(value, ensure_ascii=True, default=_default, separators=(",", ":"))

    @staticmethod
    def _init_csv(path: str, header: list[str]) -> None:
        if os.path.exists(path):
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

    @staticmethod
    def _append_csv(path: str, row: list[Any]) -> None:
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def _selected_envs(self, n_envs: int) -> range:
        return range(n_envs) if self.envs == "all" else range(min(1, n_envs))

    def _capture_active(self, env_i: int) -> bool:
        return bool(self._episode_idx[env_i] % self.episode_freq == 0)

    def _on_training_start(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        n_envs = int(self.training_env.num_envs)
        self._episode_idx = np.zeros(n_envs, dtype=np.int64)
        self._episode_step = np.zeros(n_envs, dtype=np.int64)
        self._episode_return = np.zeros(n_envs, dtype=np.float64)
        self._component_sums = [{} for _ in range(n_envs)]

        self._init_csv(
            self.transition_path,
            [
                "global_step",
                "env_id",
                "episode_index",
                "episode_step",
                "obs",
                "action_env",
                "buffer_action",
                "reward",
                "done",
                "timeout",
                "next_obs",
                "replay_next_obs",
                "terminal_observation",
                "reward_components",
                "env_diagnostics",
                "info_keys",
            ],
        )
        self._init_csv(
            self.episode_path,
            [
                "global_step",
                "env_id",
                "episode_index",
                "episode_return",
                "episode_length",
                "timeout",
                "reward_component_sums",
                "terminal_env_diagnostics",
            ],
        )

    def _on_step(self) -> bool:
        loc = self.locals
        obs = np.asarray(getattr(self.model, "_last_obs", []), dtype=np.float32)
        actions = np.asarray(loc.get("actions", []), dtype=np.float32)
        buffer_actions = np.asarray(loc.get("buffer_actions", []), dtype=np.float32)
        new_obs = np.asarray(loc.get("new_obs", []), dtype=np.float32)
        rewards = np.asarray(loc.get("rewards", []), dtype=np.float32).reshape(-1)
        dones = np.asarray(loc.get("dones", []), dtype=bool).reshape(-1)
        infos = list(loc.get("infos", []) or [])
        n_envs = int(rewards.size)
        if n_envs == 0:
            return True

        for env_i in range(n_envs):
            info = infos[env_i] if env_i < len(infos) and isinstance(infos[env_i], dict) else {}
            if bool(info.get("skip_replay", False)):
                continue
            reward_components = info.get("reward_components", {})
            env_diagnostics = info.get("env_diagnostics", {})
            timeout = bool(info.get("TimeLimit.truncated", False))
            terminal_obs = info.get("terminal_observation")
            replay_next_obs = terminal_obs if bool(dones[env_i]) and terminal_obs is not None else new_obs[env_i]

            self._episode_step[env_i] += 1
            self._episode_return[env_i] += float(rewards[env_i])
            if isinstance(reward_components, dict):
                sums = self._component_sums[env_i]
                for name, value in reward_components.items():
                    try:
                        sums[str(name)] = sums.get(str(name), 0.0) + float(value)
                    except (TypeError, ValueError):
                        continue

            if env_i in self._selected_envs(n_envs) and self._capture_active(env_i):
                self._append_csv(
                    self.transition_path,
                    [
                        int(getattr(self.model, "num_timesteps", 0)),
                        int(env_i),
                        int(self._episode_idx[env_i]),
                        int(self._episode_step[env_i]),
                        self._json_value(obs[env_i]),
                        self._json_value(actions[env_i]),
                        self._json_value(buffer_actions[env_i]),
                        float(rewards[env_i]),
                        int(dones[env_i]),
                        int(timeout),
                        self._json_value(new_obs[env_i]),
                        self._json_value(replay_next_obs),
                        self._json_value(terminal_obs) if terminal_obs is not None else "[]",
                        self._json_value(reward_components),
                        self._json_value(env_diagnostics),
                        self._json_value(sorted(info.keys())),
                    ],
                )

            if bool(dones[env_i]):
                if env_i in self._selected_envs(n_envs):
                    self._append_csv(
                        self.episode_path,
                        [
                            int(getattr(self.model, "num_timesteps", 0)),
                            int(env_i),
                            int(self._episode_idx[env_i]),
                            float(self._episode_return[env_i]),
                            int(self._episode_step[env_i]),
                            int(timeout),
                            self._json_value(self._component_sums[env_i]),
                            self._json_value(env_diagnostics),
                        ],
                    )
                self._episode_idx[env_i] += 1
                self._episode_step[env_i] = 0
                self._episode_return[env_i] = 0.0
                self._component_sums[env_i] = {}

        return True
