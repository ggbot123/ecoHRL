# rl/algos/hiro/hiro_callbacks.py
from __future__ import annotations

import os
from typing import Dict, Any
from collections import deque
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback


class HIROLoggingCallback(BaseCallback):
    def __init__(self, log_interval: int = 1, verbose: int = 0, smooth_window: int = 100):
        super().__init__(verbose=verbose)
        self.log_interval = log_interval
        self.smooth_window = int(smooth_window)
        self._episode_counter: int = 0
        self.low_buffers: Dict[str, deque] = {}
        self.high_buffers: Dict[str, deque] = {}

    def _record_smooth(self, logger, buffers: Dict[str, deque], tag: str, value: float):
        """
        辅助函数：更新缓冲区，计算均值，并写入对应的 smooth 选项卡
        """
        if tag not in buffers:
            buffers[tag] = deque(maxlen=self.smooth_window)
        
        buffers[tag].append(value)
        mean_val = np.mean(buffers[tag])

        # 映射到新的 Tag 前缀
        if tag.startswith("rollout/"):
            new_tag = tag.replace("rollout/", "1_rollout_smooth/")
        elif tag.startswith("reward_components/"):
            new_tag = tag.replace("reward_components/", "2_reward_smooth/")
        else:
            new_tag = f"smooth/{tag}"

        logger.record(new_tag, mean_val)

    def _on_training_start(self) -> None:
        # 简单健壮性检查
        assert hasattr(self.model, "low_logger"), "HIROLoggingCallback 需要 model.low_logger"
        assert hasattr(self.model, "high_logger"), "HIROLoggingCallback 需要 model.high_logger"

        # 初始化 episode 累计量
        self.ep_return = 0.0
        self.ep_len = 0
        self.ep_comp_sums = {}
        self._episode_counter = 0
        self.low_buffers.clear()
        self.high_buffers.clear()

    def _on_rollout_end(self) -> None:
        # 低层“episode”日志（每个 high_interval 一次）
        loc: Dict[str, Any] = self.locals
        if (
            "low_ret" not in loc
            or "low_len" not in loc
            or "low_comp_sums" not in loc
        ):
            return
        low_logger = self.model.low_logger

        # 1. 记录原始值
        ep_rew = float(loc["low_ret"])
        ep_len = int(loc["low_len"])
        low_logger.record("rollout/ep_rew", ep_rew)
        low_logger.record("rollout/ep_len", ep_len)

        # 2. 记录滑动平均值
        self._record_smooth(low_logger, self.low_buffers, "rollout/ep_rew", ep_rew)
        self._record_smooth(low_logger, self.low_buffers, "rollout/ep_len", float(ep_len))

        comp_sums: Dict[str, float] = loc["low_comp_sums"]
        for name, value in comp_sums.items():
            val = float(value)
            # 记录原始分量
            tag = f"reward_components/{name}"
            low_logger.record(tag, val)
            # 记录平滑分量
            self._record_smooth(low_logger, self.low_buffers, tag, val)

        # 这里直接按 model.num_timesteps 写 step，与 SB3 保持一致
        low_logger.dump(step=self.model.num_timesteps)

    def _on_step(self) -> bool:
        loc: Dict[str, Any] = self.locals

        reward_env = float(loc.get("reward_env", 0.0))
        r_components: Dict[str, float] = loc.get("r_components", {})

        # 累积 episode 级别统计
        self.ep_return += reward_env
        self.ep_len += 1
        for name, val in r_components.items():
            self.ep_comp_sums[name] = self.ep_comp_sums.get(name, 0.0) + float(val)

        # 只有 episode_end=True 的那一步才真正写高层日志
        if not loc.get("episode_end", False):
            return True

        self._episode_counter += 1
        high_logger = self.model.high_logger

        # 1. 记录原始值
        high_logger.record("rollout/ep_rew", self.ep_return)
        high_logger.record("rollout/ep_len", self.ep_len)
        
        # 2. 记录滑动平均值
        self._record_smooth(high_logger, self.high_buffers, "rollout/ep_rew", self.ep_return)
        self._record_smooth(high_logger, self.high_buffers, "rollout/ep_len", float(self.ep_len))

        for name, value in self.ep_comp_sums.items():
            val = float(value)
            tag = f"reward_components/{name}"
            high_logger.record(tag, val)
            self._record_smooth(high_logger, self.high_buffers, tag, val)

        # 控制 dump 频率（默认每个 episode 都 dump）
        if self._episode_counter % self.log_interval == 0:
            high_logger.dump(step=self.model.num_timesteps)

        # reset episode 统计
        self.ep_return = 0.0
        self.ep_len = 0
        self.ep_comp_sums = {}

        return True

class HIROCheckpointCallback(BaseCallback):
    """
    按固定 step 间隔保存 HIRO 的高层/低层模型。
    """
    def __init__(self, save_freq: int, save_dir: str, prefix: str = "hiro", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_dir = save_dir
        self.prefix = prefix

    def _on_step(self) -> bool:
        # 使用 HIROSAC.num_timesteps 作为 step 计数
        step = self.model.num_timesteps
        if step > 0 and step % self.save_freq == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            high_path = os.path.join(self.save_dir, f"{self.prefix}_high_{step}.zip")
            low_path = os.path.join(self.save_dir, f"{self.prefix}_low_{step}.zip")
            # 直接保存 SB3 的 SAC 子模型
            self.model.high_agent.save(high_path)
            self.model.low_agent.save(low_path)
            if self.verbose > 0:
                print(f"[HIROCheckpoint] Saved checkpoint at step={step} to:\n  {high_path}\n  {low_path}")
        return True