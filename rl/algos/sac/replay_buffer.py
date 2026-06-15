from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer


class SkipReplayBuffer(ReplayBuffer):
    """Compact replay buffer that discards transitions marked skip_replay."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: str | th.device = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ) -> None:
        self.source_n_envs = int(n_envs)
        # Store valid transitions one by one so skipped vector-env slots do not
        # consume replay capacity.
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=1,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=handle_timeout_termination,
        )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        obs_arr = np.asarray(obs)
        next_obs_arr = np.asarray(next_obs)
        action_arr = np.asarray(action)
        reward_arr = np.asarray(reward).reshape(-1)
        done_arr = np.asarray(done).reshape(-1)

        for env_i in range(self.source_n_envs):
            info = (
                infos[env_i]
                if env_i < len(infos) and isinstance(infos[env_i], dict)
                else {}
            )
            if bool(info.get("skip_replay", False)):
                continue
            super().add(
                obs_arr[env_i : env_i + 1],
                next_obs_arr[env_i : env_i + 1],
                action_arr[env_i : env_i + 1],
                reward_arr[env_i : env_i + 1],
                done_arr[env_i : env_i + 1],
                [info],
            )

    def valid_count(self) -> int:
        return int(self.buffer_size if self.full else self.pos)
