# rl/algos/hiro/utils.py
from __future__ import annotations

from typing import Sequence, List, Tuple
import numpy as np


def init_kinematics_meta(
    env,
    obs_flat: np.ndarray,
    keep_features: Sequence[str] = ("x", "y", "vx", "vy"),
) -> Tuple[int, int, int, List[str], List[int]]:
    """Infer kinematics meta from VecEnv config + one flattened observation.

    Assumes observation is 1D: [t, kinematics_flat], where kinematics_flat
    corresponds to K vehicles and F features per-vehicle.

    For SubprocVecEnv, prefer reading picklable dict config from `env.get_attr`.
    """
    env_cfg = env.get_attr("config", indices=0)[0] if hasattr(env, "get_attr") else getattr(env.unwrapped, "config", {})
    obs_cfg = env_cfg.get("observation", {})

    feature_names = list(obs_cfg["features"])
    feat_dim = int(len(feature_names))
    n_veh = int(obs_cfg["vehicles_count"])
    n_veh_local = int(obs_cfg["vehicles_count_local"])
    ego_feature_idx = [feature_names.index(name) for name in keep_features]

    return n_veh, n_veh_local, feat_dim, feature_names, ego_feature_idx


def split_time_kinematics(obs_flat: np.ndarray, n_veh: int, feat_dim: int):
    arr = np.asarray(obs_flat, dtype=np.float32)
    t = arr[:, 0]
    kin_flat = arr[:, 1:]
    kin = kin_flat.reshape(arr.shape[0], n_veh, feat_dim)
    return t, kin, kin_flat


def extract_ego_substate(kin: np.ndarray, ego_feature_idx: Sequence[int]) -> np.ndarray:
    """Extract ego substate from kinematics tensor (B, K, F) -> (B, len(idx))."""
    idx = np.asarray(ego_feature_idx, dtype=int)
    return kin[:, 0, idx].astype(np.float32)


def _normalize_vec_phys(vec: np.ndarray, feature_ranges: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    ranges = np.asarray(feature_ranges, dtype=np.float32)
    v_min = ranges[:, 0]
    v_max = ranges[:, 1]
    denom = v_max - v_min
    norm = 2.0 * (vec - v_min) / denom - 1.0
    norm = np.where(denom > 0.0, norm, 0.0)
    return np.clip(norm, -1.0, 1.0).astype(np.float32)


def intrinsic_reward_l2(
    ego_next_sub_rel: np.ndarray,
    goal_rel: np.ndarray,
    norm_ranges: np.ndarray,
    coef: float,
    weights: np.ndarray | Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute HIRO intrinsic reward based on weighted L2 distance.

    Returns
    -------
    reward : (B,)
        Intrinsic reward = -coef * dist.
    goal_err_phys : (B, D)
        Physical error (ego_next_sub_rel - goal_rel) without normalization.
    reward_unweighted : (B,)
        Reward divided by coef (i.e. -dist). This is well-defined even when coef=0.
    """
    ego = np.asarray(ego_next_sub_rel, dtype=np.float32)
    goal = np.asarray(goal_rel, dtype=np.float32)

    norm_ranges = np.asarray(norm_ranges, dtype=np.float32)
    ego_norm = _normalize_vec_phys(ego, norm_ranges)
    goal_norm = _normalize_vec_phys(goal, norm_ranges)
    delta = ego_norm - goal_norm
    n = int(delta.shape[1])

    if weights is None:
        w = np.ones(n, dtype=np.float32) / float(n)
    else:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        s = float(w.sum())
        w = (np.ones(n, dtype=np.float32) / float(n)) if s == 0.0 else (w / s)

    dist = np.sqrt(np.sum((delta * delta) * w[None, :], axis=1)).astype(np.float32)
    reward_unweighted = (-dist).astype(np.float32)
    reward = (-float(coef) * dist).astype(np.float32)

    goal_err_phys = (ego - goal).astype(np.float32)
    return reward, goal_err_phys, reward_unweighted


def map_y_code_to_target_y(y_code: np.ndarray, y_current: np.ndarray, lane_center_ys: np.ndarray) -> np.ndarray:
    y_code = np.asarray(y_code, dtype=np.float32).reshape(-1)
    y_current = np.asarray(y_current, dtype=np.float32).reshape(-1)
    lanes_y = np.asarray(lane_center_ys, dtype=np.float32).reshape(-1)

    diff = np.abs(y_current[:, None] - lanes_y[None, :])
    k = diff.argmin(axis=1)
    k_target = k.copy()
    k_target[(y_code < -1 / 3) & (k > 0)] -= 1
    k_target[(y_code > 1 / 3) & (k < len(lanes_y) - 1)] += 1
    return lanes_y[k_target].astype(np.float32)


def goal_action_to_abs(ego_sub: np.ndarray, goal_action: np.ndarray, lane_center_ys: np.ndarray) -> np.ndarray:
    """Convert high-level goal action a=[Δx, y_code, vx_target] to absolute goal state [x*, y*, vx*, 0].

    ego_sub: current absolute ego state [x0, y0, vx0, vy0]
    goal_action: high-level action vector
    lane_center_ys: lane center y coordinates
    """
    ego_sub = np.asarray(ego_sub, dtype=np.float32)
    goal_action = np.asarray(goal_action, dtype=np.float32)

    dx = goal_action[:, 0]
    y_code = goal_action[:, 1]
    vx_target = goal_action[:, 2]

    x_target = ego_sub[:, 0] + dx
    y_target = map_y_code_to_target_y(y_code, ego_sub[:, 1], lane_center_ys)
    return np.stack([x_target, y_target, vx_target, np.zeros_like(x_target)], axis=1).astype(np.float32)


# ----------------------------------------------------------------------
# Action scaling helpers
# ----------------------------------------------------------------------
def unscale_action(scaled_action: np.ndarray, action_space) -> np.ndarray:
    """Map scaled [-1, 1] action -> env action space (Box bounds)."""
    a = np.asarray(scaled_action, dtype=np.float32)
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    denom = (high - low)
    # Avoid division-by-zero when some dims have identical bounds
    return low + 0.5 * (a + 1.0) * np.where(denom != 0.0, denom, 1.0)


def scale_action(env_action: np.ndarray, action_space) -> np.ndarray:
    """Map env action space (Box bounds) -> scaled [-1, 1] action.

    This follows the same affine mapping used by SB3:
        scaled = 2 * (a - low) / (high - low) - 1
    """
    a = np.asarray(env_action, dtype=np.float32)
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    denom = (high - low)
    scaled = 2.0 * (a - low) / np.where(denom != 0.0, denom, 1.0) - 1.0
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)
