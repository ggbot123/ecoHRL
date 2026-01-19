from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Union

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class GoalSamplerConfig:
    type: str = "uniform"  # "uniform" | "pretrained" | "pretrained_sac"
    path: Optional[str] = None
    device: str = "auto"
    deterministic: bool = True
    action: Optional[list[float]] = None  # used when type == "fixed"

class GoalSampler:
    """Base class for goal samplers."""
    def __init__(self, action_space: gym.spaces.Box):
        self.action_space = action_space

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class UniformGoalSampler(GoalSampler):
    """Samples goals uniformly from the high-level action space."""
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        n = obs.shape[0]
        low = self.action_space.low
        high = self.action_space.high
        # Uniform sample in [low, high]
        return np.random.uniform(low, high, size=(n, low.shape[0])).astype(np.float32)


class PretrainedSACGoalSampler(GoalSampler):
    """Samples goals using a pretrained high-level SAC policy."""

    def __init__(self, action_space: gym.spaces.Box, model_path: str, device: str = "auto", deterministic: bool = True):
        super().__init__(action_space)
        if not model_path:
            raise ValueError("model_path is required for pretrained goal sampler")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pretrained goal sampler model not found: {model_path}")

        from rl.algos.sac.sac import SAC

        self._model = SAC.load(model_path, device=device)
        self._deterministic = bool(deterministic)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        actions, _ = self._model.predict(obs, deterministic=self._deterministic)
        return np.asarray(actions, dtype=np.float32)


class FixedGoalSampler(GoalSampler):
    """Always returns a fixed goal action (broadcasted to batch)."""

    def __init__(self, action_space: gym.spaces.Box, action: list[float]):
        super().__init__(action_space)
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        expected_dim = int(np.prod(self.action_space.shape))
        if a.size != expected_dim:
            raise ValueError(f"Fixed goal action dim mismatch: got {a.size}, expected {expected_dim}")
        low = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
        self._action = np.clip(a, low, high).astype(np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        n = int(np.asarray(obs).shape[0])
        return np.repeat(self._action[None, :], n, axis=0)

def get_goal_sampler(
    cfg: Union[GoalSamplerConfig, str],
    action_space: gym.spaces.Box,
) -> GoalSampler:
    if isinstance(cfg, str):
        cfg = GoalSamplerConfig(type=cfg)

    sampler_type = str(cfg.type).lower()
    if sampler_type == "uniform":
        return UniformGoalSampler(action_space)
    if sampler_type in {"fixed", "constant"}:
        if cfg.action is None:
            raise ValueError("GoalSamplerConfig.action is required for type='fixed'")
        return FixedGoalSampler(action_space, action=list(cfg.action))
    if sampler_type in {"pretrained", "pretrained_sac"}:
        return PretrainedSACGoalSampler(
            action_space,
            model_path=cfg.path or "",
            device=str(cfg.device),
            deterministic=bool(cfg.deterministic),
        )
    raise ValueError(f"Unknown goal sampler type: {sampler_type}")
