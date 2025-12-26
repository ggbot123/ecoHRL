from collections import deque
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

        # Accumulate rewards
        self._ep_ret += rewards
        self._ep_len += 1

        # Accumulate components
        if infos:
            for i, info in enumerate(infos):
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