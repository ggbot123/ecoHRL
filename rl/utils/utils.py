# rl/algos/hiro/utils.py
from __future__ import annotations

from typing import Sequence, List, Dict, Tuple
import numpy as np
import pandas as pd

from custom_env.envs.common.observation import KinematicObservation


def init_kinematics_meta(
    env,
    obs_flat: np.ndarray,
    keep_features: Sequence[str] = ("x", "y", "vx", "vy", "heading"),
) -> Tuple[int, int, List[str], List[int], np.ndarray]:
    """
    初始化与 Kinematics 相关的元信息，并为高层 goal 构造 feature_range。

    假设 env.reset() 后观测为一维:
        obs_flat = [t, kinematics_flat]
    其中 kinematics_flat 长度 = K * F，K = vehicles_count，F = len(features)。

    返回:
        n_veh: 车辆个数 K
        feat_dim: 特征维数 F
        feature_names: Kinematics 的 feature 名称列表
        ego_feature_idx: ego 子状态在一行中的索引（只保留 x/y/vx/vy/heading）
        goal_feature_ranges: shape = (ego_dim, 2)，每维 [min, max] 物理范围
    """
    obs_type = getattr(env.unwrapped, "observation_type", None)
    assert isinstance(
        obs_type, KinematicObservation
    ), f"HIRO 目前只支持 KinematicObservation，实际是 {type(obs_type)}"

    feature_names = list(obs_type.features)
    feat_dim = len(feature_names)
    n_veh = int(obs_type.vehicles_count)

    obs_arr = np.asarray(obs_flat, dtype=np.float32).reshape(-1)
    expected_dim = 1 + n_veh * feat_dim
    assert (
        obs_arr.shape[0] == expected_dim
    ), f"obs 维度应为 {expected_dim}，实际为 {obs_arr.shape[0]}"

    kin_flat = obs_arr[1:]
    kin = kin_flat.reshape(n_veh, feat_dim)
    ego_full = kin[0]  # ego 一行完整 feature

    # 确保 features_range 已初始化：KinematicObservation.normalize_obs 内部会根据道路和车辆填好范围【:contentReference[oaicite:3]{index=3}】
    if not getattr(obs_type, "features_range", None):
        df_tmp = pd.DataFrame([dict(zip(feature_names, ego_full))])
        _ = obs_type.normalize_obs(df_tmp)

    features_range: Dict[str, Sequence[float]] = dict(obs_type.features_range)

    keep_set = set(keep_features)
    ego_feature_idx = [i for i, name in enumerate(feature_names) if name in keep_set]
    if not ego_feature_idx:
        raise RuntimeError(
            f"在特征 {feature_names} 中没有找到 keep_features={keep_features}，"
            "无法构造高层 goal 子状态"
        )

    # 为 ego 子状态构造每维的 [min, max]
    goal_ranges: List[List[float]] = []
    for idx in ego_feature_idx:
        name = feature_names[idx]
        goal_ranges.append(list(features_range[name]))
    goal_feature_ranges = np.asarray(goal_ranges, dtype=np.float32)

    return n_veh, feat_dim, feature_names, ego_feature_idx, goal_feature_ranges


def split_time_kinematics(
    obs_flat: np.ndarray, n_veh: int, feat_dim: int
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    将 flatten 观测拆成:
        t: 当前时间标量
        kin: (K, F) 的 kinematics 矩阵
        kin_flat: 一维 kinematics_flat
    假设 obs_flat = [t, kin_flat]
    """
    arr = np.asarray(obs_flat, dtype=np.float32).reshape(-1)
    t = float(arr[0])
    kin_flat = arr[1:]
    assert kin_flat.shape[0] == n_veh * feat_dim, (
        f"kin_flat 维度 {kin_flat.shape[0]} != n_veh*feat_dim {n_veh * feat_dim}"
    )
    kin = kin_flat.reshape(n_veh, feat_dim)
    return t, kin, kin_flat


def extract_ego_substate(kin: np.ndarray, ego_feature_idx: Sequence[int]) -> np.ndarray:
    """
    从 kinematics 矩阵 (K, F) 中提取 ego 的子状态（只保留指定列）。
    """
    ego_full = kin[0].astype(np.float32)
    idx = np.asarray(ego_feature_idx, dtype=int)
    return ego_full[idx]


def denormalize_goal(
    norm_goal: np.ndarray, goal_feature_ranges: np.ndarray
) -> np.ndarray:
    """
    将高层策略输出的归一化动作 norm_goal ∈ [-1,1]^d
    映射到物理 goal_phys，使用与 Kinematics 相同的 feature_range。

    这是 KinematicObservation.normalize_obs 映射的逆过程：
        物理 x ∈ [v_min, v_max]  ->  x_norm ∈ [-1,1]
    这里是：
        x_norm ∈ [-1,1] -> 物理 x ∈ [v_min, v_max]
    """
    norm_goal = np.asarray(norm_goal, dtype=np.float32).reshape(-1)
    assert norm_goal.shape[0] == goal_feature_ranges.shape[0]
    phys = []
    for g, (v_min, v_max) in zip(norm_goal, goal_feature_ranges):
        v_min = float(v_min)
        v_max = float(v_max)
        if v_max > v_min:
            # [-1, 1] -> [v_min, v_max]
            x = 0.5 * (g + 1.0) * (v_max - v_min) + v_min
        else:
            x = 0.0
        phys.append(x)
    return np.asarray(phys, dtype=np.float32)


def _normalize_vec_phys(
    vec_phys: np.ndarray, feature_ranges: np.ndarray
) -> np.ndarray:
    """
    将物理空间中的向量 vec_phys 按每一维的 [v_min, v_max]
    线性映射到 [-1, 1]，映射方式与 Kinematics 的 normalize_obs 一致。
    """
    vec_phys = np.asarray(vec_phys, dtype=np.float32).reshape(-1)
    assert feature_ranges.shape[0] == vec_phys.shape[0], (
        f"feature_ranges 维度 {feature_ranges.shape[0]} 与向量维度 "
        f"{vec_phys.shape[0]} 不一致"
    )
    norm = []
    for val, (v_min, v_max) in zip(vec_phys, feature_ranges):
        v_min = float(v_min)
        v_max = float(v_max)
        if v_max > v_min:
            # [v_min, v_max] -> [-1, 1]
            x = 2.0 * (val - v_min) / (v_max - v_min) - 1.0
            x = float(np.clip(x, -1.0, 1.0))
        else:
            x = 0.0
        norm.append(x)
    return np.asarray(norm, dtype=np.float32)


def intrinsic_reward_l2(
    ego_next_sub_phys: np.ndarray,
    goal_phys: np.ndarray,
    norm_ranges: np.ndarray,
    coef: float = 1.0,
    weights: np.ndarray | Sequence[float] | None = None,
) -> float:
    ego_next_sub_phys = np.asarray(ego_next_sub_phys, dtype=np.float32).reshape(-1)
    goal_phys = np.asarray(goal_phys, dtype=np.float32).reshape(-1)
    norm_ranges = np.asarray(norm_ranges, dtype=np.float32)

    assert ego_next_sub_phys.shape == goal_phys.shape, (
        f"形状不一致: ego {ego_next_sub_phys.shape}, goal {goal_phys.shape}"
    )
    assert norm_ranges.shape[0] == ego_next_sub_phys.shape[0], (
        f"norm_ranges 维度 {norm_ranges.shape[0]} 与状态维度 {ego_next_sub_phys.shape[0]} 不一致"
    )

    ego_norm = _normalize_vec_phys(ego_next_sub_phys, norm_ranges)
    goal_norm = _normalize_vec_phys(goal_phys, norm_ranges)
    delta = ego_norm - goal_norm  # (n,)
    n = delta.shape[0]

    if weights is None:
        w = np.ones(n, dtype=np.float32) / float(n)
    else:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        assert w.shape == delta.shape, (
            f"权重维度 {w.shape} 与状态维度 {delta.shape} 不一致"
        )
        s = float(w.sum())
        # 避免全 0 权重
        if s == 0.0:
            w = np.ones(n, dtype=np.float32) / float(n)
        else:
            w = w / s
    dist_sq = float(np.sum(w * (delta * delta)))
    dist = float(np.sqrt(dist_sq))

    return float(-coef * dist)


def map_y_code_to_target_y(y_code: float, y_current: float, lane_center_ys: np.ndarray) -> float:
    """
    y_code ∈ [-1, 1] 映射到“当前车道/左/右相邻车道”的中心线 y。
    """
    lanes_y = np.asarray(lane_center_ys, dtype=np.float32)
    # 当前 ego 所在车道：y 轴上最近的一条
    k = int(np.argmin(np.abs(lanes_y - y_current)))
    if y_code > 0 and k < len(lanes_y) - 1:
        k_target = k + 1   # 左侧相邻车道
    elif y_code < 0 and k > 0:
        k_target = k - 1   # 右侧相邻车道
    else:
        k_target = k       # 保持当前车道
    return float(lanes_y[k_target])


def goal_action_to_abs(
    ego_sub: np.ndarray,
    goal_action: np.ndarray,
    lane_center_ys: np.ndarray,
) -> np.ndarray:
    """
    将高层动作向量 a=[Δx, y_code, vx_target, _] 转成绝对目标状态 [x*, y*, vx*, 0]。

    ego_sub: 自车当前绝对状态 [x0, y0, vx0, vy0]
    goal_action: SAC 输出动作向量
    lane_center_ys: 各车道中心线 y
    """
    ego_sub = np.asarray(ego_sub, dtype=np.float32).reshape(-1)
    goal_action = np.asarray(goal_action, dtype=np.float32).reshape(-1)

    dx = float(goal_action[0])
    y_code = float(goal_action[1])
    vx_target = float(goal_action[2])

    x_target = ego_sub[0] + dx
    y_target = map_y_code_to_target_y(y_code, ego_sub[1], lane_center_ys)
    return np.array([x_target, y_target, vx_target, 0.0], dtype=np.float32)
