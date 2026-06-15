from __future__ import annotations

from typing import Any

import numpy as np


def is_random_goal_lane(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "random"


def sample_goal_lane_id(
    rng,
    *,
    goal_lane_id: Any,
    lanes_count: int,
    goal_lane_probs: Any = None,
) -> int:
    lanes = int(lanes_count)
    if lanes <= 0:
        raise ValueError(f"lanes_count must be positive, got {lanes}")

    if goal_lane_probs is not None:
        probs = np.asarray(goal_lane_probs, dtype=np.float64).reshape(-1)
        if probs.size != lanes:
            raise ValueError(
                f"goal_lane_probs must contain {lanes} values, got {probs.size}"
            )
        if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
            raise ValueError("goal_lane_probs must be finite and non-negative")
        total = float(np.sum(probs))
        if total <= 0.0:
            raise ValueError("goal_lane_probs must have a positive sum")
        return int(rng.choice(lanes, p=probs / total))

    if is_random_goal_lane(goal_lane_id):
        return int(rng.integers(lanes))
    return int(np.clip(int(goal_lane_id), 0, lanes - 1))
