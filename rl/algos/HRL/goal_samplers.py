import numpy as np
import gymnasium as gym

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

def get_goal_sampler(sampler_type: str, action_space: gym.spaces.Box) -> GoalSampler:
    if sampler_type == "uniform":
        return UniformGoalSampler(action_space)
    else:
        raise ValueError(f"Unknown goal sampler type: {sampler_type}")
