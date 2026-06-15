from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from custom_env import utils as c_utils
from rl.utils import utils as rl_utils


@dataclass
class VirtualVehicle:
    position: np.ndarray  # (2,) absolute
    velocity: np.ndarray  # (2,) absolute
    heading: float
    speed: float
    lane_index: int
    target_speed: float
    length: float = 5.0

    def lane_distance_to(self, other: "VirtualVehicle") -> float:
        # Assume road axis aligned with +x
        return float(other.position[0] - self.position[0])


class RuleBasedController:
    """基于观测(obs)的 rule-based 低层控制器。

    目标：兼容 SubprocVecEnv（不访问 env.unwrapped.vehicle/road 等不可跨进程对象）。

    支持两类动作空间：
    - ContinuousAction: action = [acc_norm, steer_norm]
    - ParamLaneAccelAction: action = [lane_scalar, acc_norm]
    """

    def __init__(self, env_or_config: Any, low_safety_filter: Any = None):
        # Backward-compatible: allow passing a gym env (used by older eval scripts)
        if isinstance(env_or_config, dict):
            env_config = env_or_config
        else:
            env = env_or_config
            env_config = getattr(getattr(env, "unwrapped", env), "config", None)
            if env_config is None:
                raise TypeError("RuleBasedController expects an env with .unwrapped.config or a config dict")

        self.config = dict(env_config)

        if low_safety_filter is not None:
            if isinstance(low_safety_filter, dict):
                sf = dict(low_safety_filter)
            else:
                sf = {
                    "type": getattr(low_safety_filter, "type", None),
                    "lane_change_min_front_gap": getattr(low_safety_filter, "lane_change_min_front_gap", None),
                    "lane_change_min_rear_gap": getattr(low_safety_filter, "lane_change_min_rear_gap", None),
                    "lane_change_min_front_ttc": getattr(low_safety_filter, "lane_change_min_front_ttc", None),
                    "lane_change_min_rear_ttc": getattr(low_safety_filter, "lane_change_min_rear_ttc", None),
                    "safe_gap_d_min": getattr(low_safety_filter, "safe_gap_d_min", None),
                    "safe_gap_tau": getattr(low_safety_filter, "safe_gap_tau", None),
                    "safe_gap_b_ego": getattr(low_safety_filter, "safe_gap_b_ego", None),
                    "safe_gap_b_front": getattr(low_safety_filter, "safe_gap_b_front", None),
                    "safe_gap_comfort_decel": getattr(low_safety_filter, "safe_gap_comfort_decel", None),
                    "safe_gap_emergency_decel": getattr(low_safety_filter, "safe_gap_emergency_decel", None),
                    "safe_gap_emergency_ttc": getattr(low_safety_filter, "safe_gap_emergency_ttc", None),
                    "safe_gap_emergency_distance": getattr(low_safety_filter, "safe_gap_emergency_distance", None),
                }
            sf_type = sf.get("type")
            if sf_type is not None:
                self.config["low_safety_filter_type"] = str(sf_type)
            for key in (
                "lane_change_min_front_gap",
                "lane_change_min_rear_gap",
                "lane_change_min_front_ttc",
                "lane_change_min_rear_ttc",
                "safe_gap_d_min",
                "safe_gap_tau",
                "safe_gap_b_ego",
                "safe_gap_b_front",
                "safe_gap_comfort_decel",
                "safe_gap_emergency_decel",
                "safe_gap_emergency_ttc",
                "safe_gap_emergency_distance",
            ):
                val = sf.get(key)
                if val is not None:
                    self.config[key] = float(val)

        # Action config
        act_cfg = dict(self.config.get("action", {}) or {})
        self.action_type = str(act_cfg.get("type", "ContinuousAction"))

        # ContinuousAction
        self.acc_range = tuple(act_cfg.get("acceleration_range", (-5.0, 5.0)))
        self.steer_range = tuple(act_cfg.get("steering_range", (-np.pi / 4, np.pi / 4)))

        # ParamLaneAccelAction
        self.acc_min, self.acc_max = float(self.acc_range[0]), float(self.acc_range[1])

        # Controller parameters (aligned with highway-env controlled vehicle style)
        self.TAU_ACC = 0.6
        self.TAU_HEADING = 0.2
        self.TAU_LATERAL = 0.6
        self.TAU_PURSUIT = 0.5 * self.TAU_HEADING
        self.KP_A = 1.0 / self.TAU_ACC
        self.KP_HEADING = 1.0 / self.TAU_HEADING
        self.KP_LATERAL = 1.0 / self.TAU_LATERAL
        self.MAX_STEERING_ANGLE = np.pi / 3
        self.LENGTH = 5.0

        # IDM/MOBIL params
        self.COMFORT_ACC_MAX = 3.0
        self.COMFORT_ACC_MIN = -5.0
        self.DISTANCE_WANTED = 10.0
        self.TIME_WANTED = 0.5
        self.DELTA = 4.0

        self.POLITENESS = 0.0
        self.LANE_CHANGE_MIN_ACC_GAIN = -4.0
        self.LANE_CHANGE_MAX_BRAKING_IMPOSED = 4.0

        # Road geometry from config
        self.lanes_count = int(self.config.get("lanes_count", 3))
        self.lane_width = float(self.config.get("lane_width", 4.0))
        self.speed_limit = float(self.config.get("speed_limit", 30.0))
        self.lane_center_ys = (np.arange(self.lanes_count, dtype=np.float32) * self.lane_width).astype(np.float32)

        # Safety filter mode
        self.low_safety_filter_type = str(self.config.get("low_safety_filter_type", "legacy")).lower().strip()
        self.compute_action_mode = str(
            self.config.get("rule_based_compute_action_mode", "target_speed_lane")
        ).lower().strip()

        # Observation meta (for act(obs, ...))
        obs_cfg = dict(self.config.get("observation", {}) or {})
        self.obs_features = list(obs_cfg.get("features", ["presence", "x", "y", "vx", "vy", "acceleration"]))
        self.obs_feat_dim = int(len(self.obs_features))
        self.obs_vehicles_count = int(obs_cfg.get("vehicles_count", 5))
        self.idx_presence = self.obs_features.index("presence")
        self.idx_x = self.obs_features.index("x")
        self.idx_y = self.obs_features.index("y")
        self.idx_vx = self.obs_features.index("vx")
        self.idx_vy = self.obs_features.index("vy")

    def act(self, obs: np.ndarray, goal_phys: np.ndarray, remaining_time: Optional[float] = None) -> np.ndarray:
        """兼容旧接口：直接从 env 的 flattened obs 计算动作。"""
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        goal_phys = np.asarray(goal_phys, dtype=np.float32).reshape(-1)

        # KinematicObservation(include_time=True) => [t, vehicles_flat]
        # ego row is absolute, others are relative to ego.
        _, kin, _ = rl_utils.split_time_kinematics(obs, self.obs_vehicles_count, self.obs_feat_dim)
        ego_feat = kin[0, 0]
        ego_abs = np.array(
            [
                ego_feat[self.idx_x],
                ego_feat[self.idx_y],
                ego_feat[self.idx_vx],
                ego_feat[self.idx_vy],
            ],
            dtype=np.float32,
        )
        others_feat = kin[0, 1:]
        others_rel: List[List[float]] = []
        for j in range(int(others_feat.shape[0])):
            d = others_feat[j]
            if d[self.idx_presence] == 0:
                continue
            others_rel.append([float(d[self.idx_x]), float(d[self.idx_y]), float(d[self.idx_vx]), float(d[self.idx_vy])])
        others_rel_arr = np.asarray(others_rel, dtype=np.float32).reshape(-1, 4)

        dt = 1.0 / float(self.config.get("policy_frequency", 10.0))
        return self.compute_action(ego_abs, others_rel_arr, goal_phys, dt, remaining_time=remaining_time)

    # -------------------------- geometry helpers --------------------------
    def get_lane_index(self, y: float) -> int:
        return int(np.argmin(np.abs(self.lane_center_ys - float(y))))

    def get_lane_y(self, lane_index: int) -> float:
        idx = int(np.clip(int(lane_index), 0, self.lanes_count - 1))
        return float(self.lane_center_ys[idx])

    def _get_neighbors(
        self,
        ego: VirtualVehicle,
        others: List[VirtualVehicle],
        lane_index: int,
    ) -> Tuple[Optional[VirtualVehicle], Optional[VirtualVehicle]]:
        front_vehicle: Optional[VirtualVehicle] = None
        rear_vehicle: Optional[VirtualVehicle] = None
        min_front = float("inf")
        min_rear = float("inf")

        for v in others:
            if v.lane_index != lane_index:
                continue
            dist = float(v.position[0] - ego.position[0])
            if dist > 0.0:
                if dist < min_front:
                    min_front = dist
                    front_vehicle = v
            else:
                if -dist < min_rear:
                    min_rear = -dist
                    rear_vehicle = v
        return front_vehicle, rear_vehicle

    # -------------------------- IDM / MOBIL --------------------------
    def desired_gap(self, ego: VirtualVehicle, front: VirtualVehicle) -> float:
        d0 = float(self.DISTANCE_WANTED)
        tau = float(self.TIME_WANTED)
        ab = -float(self.COMFORT_ACC_MAX) * float(self.COMFORT_ACC_MIN)
        dv = float(ego.speed - front.speed)
        return float(d0 + ego.speed * tau + ego.speed * dv / (2.0 * math.sqrt(max(ab, 1e-6))))

    def idm_acceleration(self, ego: VirtualVehicle, front: Optional[VirtualVehicle]) -> float:
        v0 = float(np.clip(ego.target_speed, 0.0, self.speed_limit))
        v = float(max(ego.speed, 0.0))

        # Free-road term
        acc = float(self.COMFORT_ACC_MAX) * (1.0 - (v / float(abs(c_utils.not_zero(v0)))) ** float(self.DELTA))

        # Interaction term
        if front is not None:
            d = max(ego.lane_distance_to(front), 0.1)
            s_star = self.desired_gap(ego, front)
            acc -= float(self.COMFORT_ACC_MAX) * (s_star / float(c_utils.not_zero(d))) ** 2

        return float(acc)

    def mobil_ok(self, ego: VirtualVehicle, target_lane: int, others: List[VirtualVehicle]) -> bool:
        # Safety for new following vehicle
        new_front, new_rear = self._get_neighbors(ego, others, target_lane)
        if new_rear is not None:
            # assume rear vehicle keeps its current speed as desired speed
            new_rear_v0 = max(new_rear.speed, 0.1)
            new_rear_eval = VirtualVehicle(
                position=new_rear.position,
                velocity=new_rear.velocity,
                heading=new_rear.heading,
                speed=new_rear.speed,
                lane_index=new_rear.lane_index,
                target_speed=new_rear_v0,
                length=new_rear.length,
            )
            a_rear_now = self.idm_acceleration(new_rear_eval, new_front)
            # predicted if ego cuts in
            a_rear_pred = self.idm_acceleration(new_rear_eval, ego)
            if a_rear_pred < -float(self.LANE_CHANGE_MAX_BRAKING_IMPOSED):
                return False
        else:
            a_rear_now = 0.0
            a_rear_pred = 0.0

        # Ego incentive + politeness
        old_front, old_rear = self._get_neighbors(ego, others, ego.lane_index)
        a_ego_now = self.idm_acceleration(ego, old_front)
        a_ego_pred = self.idm_acceleration(ego, new_front)
        if a_ego_pred < -float(self.LANE_CHANGE_MAX_BRAKING_IMPOSED):
            return False

        if old_rear is not None:
            old_rear_v0 = max(old_rear.speed, 0.1)
            old_rear_eval = VirtualVehicle(
                position=old_rear.position,
                velocity=old_rear.velocity,
                heading=old_rear.heading,
                speed=old_rear.speed,
                lane_index=old_rear.lane_index,
                target_speed=old_rear_v0,
                length=old_rear.length,
            )
            a_old_rear_now = self.idm_acceleration(old_rear_eval, old_front)
            a_old_rear_pred = self.idm_acceleration(old_rear_eval, old_front)
        else:
            a_old_rear_now = 0.0
            a_old_rear_pred = 0.0

        jerk = (a_ego_pred - a_ego_now) + float(self.POLITENESS) * ((a_rear_pred - a_rear_now) + (a_old_rear_pred - a_old_rear_now))
        return bool(jerk >= float(self.LANE_CHANGE_MIN_ACC_GAIN))

    # -------------------------- action mapping --------------------------
    def _acc_phys_to_norm(self, acc_phys: float) -> float:
        lo, hi = float(self.acc_min), float(self.acc_max)
        if hi == lo:
            return 0.0
        x = 2.0 * (float(acc_phys) - lo) / (hi - lo) - 1.0
        return float(np.clip(x, -1.0, 1.0))

    def _acc_norm_to_phys(self, acc_norm: float) -> float:
        lo, hi = float(self.acc_min), float(self.acc_max)
        if hi == lo:
            return float(lo)
        return float(lo + 0.5 * (float(acc_norm) + 1.0) * (hi - lo))

    def _lane_to_scalar(self, ego_lane: int, target_lane: int) -> float:
        if target_lane < ego_lane:
            return -1.0
        if target_lane > ego_lane:
            return 1.0
        return 0.0

    def _scalar_to_lane(self, ego_lane: int, lane_scalar: float) -> int:
        if lane_scalar < -1.0 / 3.0:
            return int(np.clip(ego_lane - 1, 0, self.lanes_count - 1))
        if lane_scalar > 1.0 / 3.0:
            return int(np.clip(ego_lane + 1, 0, self.lanes_count - 1))
        return int(np.clip(ego_lane, 0, self.lanes_count - 1))

    def _as_ego_abs4(self, ego_abs: np.ndarray) -> np.ndarray:
        ego = np.asarray(ego_abs, dtype=np.float32).reshape(-1)
        if ego.size != 4:
            raise ValueError(
                f"ego_abs must be exactly 4 elements [x, y, vx, vy], got shape {ego.shape}"
            )
        return ego.astype(np.float32, copy=False)

    def safety_filter_action(
        self,
        ego_abs: np.ndarray,
        others_rel: np.ndarray,
        goal_phys: np.ndarray,
        action: np.ndarray,
        dt: float,
        remaining_time: Optional[float] = None,
    ) -> np.ndarray:
        if self.low_safety_filter_type == "legacy":
            return self._safety_filter_action_legacy(
                ego_abs=ego_abs,
                others_rel=others_rel,
                goal_phys=goal_phys,
                action=action,
                dt=dt,
                remaining_time=remaining_time,
            )
        if self.low_safety_filter_type in {"legacy_mpc_max", "mpc_legacy_max"}:
            return self._safety_filter_action_legacy_mpc_max(
                ego_abs=ego_abs,
                others_rel=others_rel,
                goal_phys=goal_phys,
                action=action,
                dt=dt,
                remaining_time=remaining_time,
            )
        if self.low_safety_filter_type in {"rss", "braking_distance", "safe_gap", "safe_gap_braking"}:
            return self._safety_filter_action_rss(
                ego_abs=ego_abs,
                others_rel=others_rel,
                goal_phys=goal_phys,
                action=action,
                dt=dt,
                remaining_time=remaining_time,
            )

        ego_abs = self._as_ego_abs4(ego_abs)
        goal_phys = np.asarray(goal_phys, dtype=np.float32).reshape(-1)
        ego_vel = np.asarray([float(ego_abs[2]), float(ego_abs[3])], dtype=np.float32)
        goal_rel = (goal_phys - ego_abs).astype(np.float32)
        ego_y = float(ego_abs[1])
        ego_lane = int(self.get_lane_index(ego_y))
        return self.safety_filter_action_relative(
            ego_vel=ego_vel,
            others_rel=others_rel,
            goal_rel=goal_rel,
            action=action,
            dt=dt,
            remaining_time=remaining_time,
            ego_lane=ego_lane,
            ego_y=ego_y,
        )

    def _safety_filter_action_legacy_mpc_max(
        self,
        ego_abs: np.ndarray,
        others_rel: np.ndarray,
        goal_phys: np.ndarray,
        action: np.ndarray,
        dt: float,
        remaining_time: Optional[float] = None,
    ) -> np.ndarray:
        """Combine legacy and MPC constraints, using the larger acceleration upper bound."""
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_type != "ParamLaneAccelAction":
            return self._safety_filter_action_legacy(
                ego_abs=ego_abs,
                others_rel=others_rel,
                goal_phys=goal_phys,
                action=act,
                dt=dt,
                remaining_time=remaining_time,
            )

        legacy_action = self._safety_filter_action_legacy(
            ego_abs=ego_abs,
            others_rel=others_rel,
            goal_phys=goal_phys,
            action=act,
            dt=dt,
            remaining_time=remaining_time,
        )

        ego_abs_arr = self._as_ego_abs4(ego_abs)
        goal_phys_arr = np.asarray(goal_phys, dtype=np.float32).reshape(-1)
        ego_vel = np.asarray([float(ego_abs_arr[2]), float(ego_abs_arr[3])], dtype=np.float32)
        goal_rel = (goal_phys_arr - ego_abs_arr).astype(np.float32)
        ego_y = float(ego_abs_arr[1])
        ego_lane = int(self.get_lane_index(ego_y))
        mpc_action = self.safety_filter_action_relative(
            ego_vel=ego_vel,
            others_rel=others_rel,
            goal_rel=goal_rel,
            action=act,
            dt=dt,
            remaining_time=remaining_time,
            ego_lane=ego_lane,
            ego_y=ego_y,
        )

        lane_scalar = float(act[0])
        legacy_allows_lane = abs(float(legacy_action[0]) - lane_scalar) <= 1e-6
        mpc_allows_lane = abs(float(mpc_action[0]) - lane_scalar) <= 1e-6
        if not (legacy_allows_lane or mpc_allows_lane):
            lane_scalar = 0.0

        acc_req = self._acc_norm_to_phys(float(act[1]))
        acc_legacy = self._acc_norm_to_phys(float(legacy_action[1]))
        acc_mpc = self._acc_norm_to_phys(float(mpc_action[1]))
        acc_phys = min(float(acc_req), max(float(acc_legacy), float(acc_mpc)))
        acc_phys = float(np.clip(acc_phys, self.acc_min, self.acc_max))
        return np.array([lane_scalar, self._acc_phys_to_norm(acc_phys)], dtype=np.float32)

    def _collect_lane_rel_neighbors(
        self,
        others_rel: np.ndarray,
        lane_offset: int,
        ego_lane: int,
        ego_y: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        front: Optional[np.ndarray] = None
        rear: Optional[np.ndarray] = None
        front_dx = float("inf")
        rear_dx = float("inf")
        for row in np.asarray(others_rel, dtype=np.float32).reshape(-1, 4):
            dx, dy, dvx, dvy = [float(v) for v in row]
            lane_idx = int(self.get_lane_index(float(ego_y) + dy))
            row_lane_offset = int(lane_idx - int(ego_lane))

            if row_lane_offset != int(lane_offset):
                continue
            if dx >= 0.0:
                if dx < front_dx:
                    front_dx = dx
                    front = np.array([dx, dy, dvx, dvy], dtype=np.float32)
            else:
                if -dx < rear_dx:
                    rear_dx = -dx
                    rear = np.array([dx, dy, dvx, dvy], dtype=np.float32)
        return front, rear

    def _lane_constraints_ok_relative(
        self,
        ego_x_next: float,
        ego_vx_next: float,
        lane_offset: int,
        ego_vx_now: float,
        others_rel: np.ndarray,
        dt: float,
        ego_lane: int,
        ego_y: float,
    ) -> bool:
        front, rear = self._collect_lane_rel_neighbors(
            others_rel,
            lane_offset,
            ego_lane=ego_lane,
            ego_y=ego_y,
        )

        if front is not None:
            dx, _dy, dvx, _dvy = [float(v) for v in front]
            d_front = float(dx + dvx * dt - ego_x_next)
            rel_front = max(float(ego_vx_next) - float(ego_vx_now + dvx), 0.0)
            if d_front < float(self.config.get("lane_change_min_front_gap", 10.0)):
                return False
            if d_front < float(self.config.get("lane_change_min_front_ttc", 3.0)) * rel_front:
                return False

        if rear is not None:
            dx, _dy, dvx, _dvy = [float(v) for v in rear]
            d_rear = float(ego_x_next - (dx + dvx * dt))
            rel_rear = max(float(ego_vx_now + dvx) - float(ego_vx_next), 0.0)
            if d_rear < float(self.config.get("lane_change_min_rear_gap", 8.0)):
                return False
            if d_rear < float(self.config.get("lane_change_min_rear_ttc", 2.0)) * rel_rear:
                return False

        return True

    def safety_filter_action_relative(
        self,
        ego_vel: np.ndarray,
        others_rel: np.ndarray,
        goal_rel: np.ndarray,
        action: np.ndarray,
        dt: float,
        remaining_time: Optional[float] = None,
        ego_lane: int = 0,
        ego_y: float = 0.0,
    ) -> np.ndarray:
        """Safety layer using only ego velocity, relative neighbors and goal_rel.

        This path avoids dependency on ego absolute x/y in low_obs.
        """
        _ = goal_rel
        _ = remaining_time

        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_type != "ParamLaneAccelAction":
            return act.astype(np.float32)

        ego_vel = np.asarray(ego_vel, dtype=np.float32).reshape(-1)
        ego_vx = float(ego_vel[0])

        lane_scalar = float(act[0])
        acc_norm = float(act[1])
        acc_phys_req = self._acc_norm_to_phys(acc_norm)

        dt_safe = max(float(dt), 1e-6)
        acc_for_gate = float(np.clip(acc_phys_req, self.acc_min, self.acc_max))
        ego_x_next = float(ego_vx * dt_safe + 0.5 * acc_for_gate * (dt_safe ** 2))
        ego_vx_next = float(ego_vx + acc_for_gate * dt_safe)

        ego_lane_clip = int(np.clip(int(ego_lane), 0, self.lanes_count - 1))
        target_lane = self._scalar_to_lane(ego_lane_clip, lane_scalar)
        lane_step = int(target_lane - ego_lane_clip)

        if lane_step != 0:
            ok_origin = self._lane_constraints_ok_relative(
                ego_x_next,
                ego_vx_next,
                0,
                ego_vx,
                others_rel,
                dt_safe,
                ego_lane=ego_lane,
                ego_y=ego_y,
            )
            ok_target = self._lane_constraints_ok_relative(
                ego_x_next,
                ego_vx_next,
                lane_step,
                ego_vx,
                others_rel,
                dt_safe,
                ego_lane=ego_lane,
                ego_y=ego_y,
            )
            if not (ok_origin and ok_target):
                lane_scalar = 0.0
                lane_step = 0

        lane_front_gap_min = float(self.config.get("lane_change_min_front_gap", 10.0))
        lane_front_ttc_min = float(self.config.get("lane_change_min_front_ttc", 3.0))

        a_upper = float(self.acc_max)
        a_upper = min(a_upper, float((self.speed_limit - ego_vx) / dt_safe))

        den_gap = 0.5 * (dt_safe ** 2)
        den_ttc = den_gap + lane_front_ttc_min * dt_safe

        for row in np.asarray(others_rel, dtype=np.float32).reshape(-1, 4):
            dx, dy, dvx, _dvy = [float(v) for v in row]
            lane_idx = int(self.get_lane_index(float(ego_y) + dy))
            row_lane_offset = int(lane_idx - int(ego_lane))

            if row_lane_offset != int(lane_step):
                continue
            if dx <= 0.0:
                continue

            xj1_rel = float(dx + dvx * dt_safe)

            if den_gap > 1e-9:
                a_gap = (xj1_rel - lane_front_gap_min - ego_vx * dt_safe) / den_gap
                a_upper = min(a_upper, float(a_gap))

            if den_ttc > 1e-9:
                a_ttc = (xj1_rel - ego_vx * dt_safe + lane_front_ttc_min * dvx) / den_ttc
                a_upper = min(a_upper, float(a_ttc))

        acc_phys = min(float(acc_phys_req), float(a_upper))
        acc_phys = float(np.clip(acc_phys, self.acc_min, self.acc_max))
        acc_norm = self._acc_phys_to_norm(acc_phys)
        return np.array([lane_scalar, acc_norm], dtype=np.float32)

    def _safe_gap_required(self, follower_vx: float, leader_vx: float) -> float:
        d_min = float(self.config.get("safe_gap_d_min", 6.0))
        tau = float(self.config.get("safe_gap_tau", 0.6))
        b_ego = max(float(self.config.get("safe_gap_b_ego", 3.0)), 1e-6)
        b_front = max(float(self.config.get("safe_gap_b_front", 3.0)), 1e-6)
        fv = max(float(follower_vx), 0.0)
        lv = max(float(leader_vx), 0.0)
        return float(d_min + tau * fv + fv * fv / (2.0 * b_ego) - lv * lv / (2.0 * b_front))

    def _safe_gap_status(self, distance: float, follower_vx: float, leader_vx: float) -> Tuple[bool, bool]:
        d = float(distance)
        closing = max(float(follower_vx) - float(leader_vx), 0.0)
        ttc = float("inf") if closing <= 1e-6 else d / closing
        required_gap = self._safe_gap_required(follower_vx, leader_vx)
        emergency = (
            d < float(self.config.get("safe_gap_emergency_distance", 10.0))
            or ttc < float(self.config.get("safe_gap_emergency_ttc", 1.0))
        )
        return bool(d < required_gap), bool(emergency)

    def _safety_filter_action_rss(
        self,
        ego_abs: np.ndarray,
        others_rel: np.ndarray,
        goal_phys: np.ndarray,
        action: np.ndarray,
        dt: float,
        remaining_time: Optional[float] = None,
    ) -> np.ndarray:
        """RSS-style safety layer based on braking-distance safe gaps.

        Current/target-lane front vehicles can cap longitudinal acceleration.
        Target-lane front and rear vehicles can reject an unsafe lane change.
        """
        _ = goal_phys
        _ = remaining_time

        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_type != "ParamLaneAccelAction":
            return act.astype(np.float32)

        ego = self._as_ego_abs4(ego_abs)
        ego_y = float(ego[1])
        ego_vx = float(ego[2])
        ego_lane = int(self.get_lane_index(ego_y))
        ego_lane_clip = int(np.clip(ego_lane, 0, self.lanes_count - 1))

        lane_scalar = float(act[0])
        acc_phys_req = self._acc_norm_to_phys(float(act[1]))
        dt_safe = max(float(dt), 1e-6)
        acc_for_gate = float(np.clip(acc_phys_req, self.acc_min, self.acc_max))
        ego_x_next = float(ego_vx * dt_safe + 0.5 * acc_for_gate * (dt_safe ** 2))

        target_lane = self._scalar_to_lane(ego_lane_clip, lane_scalar)
        lane_step = int(target_lane - ego_lane_clip)

        if lane_step != 0:
            front_t, rear_t = self._collect_lane_rel_neighbors(
                others_rel,
                lane_step,
                ego_lane=ego_lane,
                ego_y=ego_y,
            )
            lane_ok = True

            if front_t is not None:
                dx, _dy, dvx, _dvy = [float(v) for v in front_t]
                front_vx = float(ego_vx + dvx)
                d_front = float(dx + dvx * dt_safe - ego_x_next)
                unsafe, emergency = self._safe_gap_status(d_front, ego_vx, front_vx)
                if unsafe or emergency:
                    lane_ok = False

            if rear_t is not None:
                dx, _dy, dvx, _dvy = [float(v) for v in rear_t]
                rear_vx = float(ego_vx + dvx)
                d_rear = float(ego_x_next - (dx + dvx * dt_safe))
                unsafe, emergency = self._safe_gap_status(d_rear, rear_vx, ego_vx)
                if unsafe or emergency:
                    lane_ok = False

            if not lane_ok:
                lane_scalar = 0.0
                lane_step = 0

        acc_phys = float(acc_phys_req)
        front, _rear = self._collect_lane_rel_neighbors(
            others_rel,
            lane_step,
            ego_lane=ego_lane,
            ego_y=ego_y,
        )
        if front is not None:
            dx, _dy, dvx, _dvy = [float(v) for v in front]
            front_vx = float(ego_vx + dvx)
            d_front = float(dx + dvx * dt_safe - ego_x_next)
            unsafe, emergency = self._safe_gap_status(d_front, ego_vx, front_vx)
            if emergency:
                acc_phys = min(acc_phys, float(self.config.get("safe_gap_emergency_decel", -5.0)))
            elif unsafe:
                acc_phys = min(acc_phys, float(self.config.get("safe_gap_comfort_decel", -3.0)))

        acc_phys = float(np.clip(acc_phys, self.acc_min, self.acc_max))
        acc_norm = self._acc_phys_to_norm(acc_phys)
        return np.array([lane_scalar, acc_norm], dtype=np.float32)

    @staticmethod
    def _ttc_ok(distance: float, rel_speed_closing: float, min_ttc: float) -> bool:
        return bool(float(distance) >= float(min_ttc) * max(float(rel_speed_closing), 0.0))

    def _lane_change_constraints_ok(
        self,
        ego: VirtualVehicle,
        front_origin: Optional[VirtualVehicle],
        front_target: Optional[VirtualVehicle],
        rear_target: Optional[VirtualVehicle],
    ) -> bool:
        ex = float(ego.position[0])
        evx = float(ego.velocity[0])

        if front_origin is not None:
            d_front_origin = float(front_origin.position[0]) - ex
            rel_front_origin = max(evx - float(front_origin.velocity[0]), 0.0)
            if d_front_origin < float(self.config.get("lane_change_min_front_gap", 10.0)):
                return False
            if not self._ttc_ok(d_front_origin, rel_front_origin, float(self.config.get("lane_change_min_front_ttc", 3.0))):
                return False

        if front_target is not None:
            d_front_target = float(front_target.position[0]) - ex
            rel_front_target = max(evx - float(front_target.velocity[0]), 0.0)
            if d_front_target < float(self.config.get("lane_change_min_front_gap", 10.0)):
                return False
            if not self._ttc_ok(d_front_target, rel_front_target, float(self.config.get("lane_change_min_front_ttc", 3.0))):
                return False

        if rear_target is not None:
            d_rear_target = ex - float(rear_target.position[0])
            rel_rear_target = max(float(rear_target.velocity[0]) - evx, 0.0)
            if d_rear_target < float(self.config.get("lane_change_min_rear_gap", 8.0)):
                return False
            if not self._ttc_ok(d_rear_target, rel_rear_target, float(self.config.get("lane_change_min_rear_ttc", 2.0))):
                return False

        return True

    def _safety_filter_action_legacy(
        self,
        ego_abs: np.ndarray,
        others_rel: np.ndarray,
        goal_phys: np.ndarray,
        action: np.ndarray,
        dt: float,
        remaining_time: Optional[float] = None,
    ) -> np.ndarray:
        """Safety layer: clamp unsafe lane change & longitudinal acceleration.

        For ParamLaneAccelAction: adjust lane_scalar if MOBIL disallows, and
        clamp acc by IDM safety for current/target lane.
        """
        ego = self._as_ego_abs4(ego_abs)
        ego_x, ego_y, ego_vx, ego_vy = [float(v) for v in ego]
        ego_speed = float(math.sqrt(ego_vx * ego_vx + ego_vy * ego_vy))
        ego_heading = float(math.atan2(ego_vy, ego_vx)) if ego_speed > 0.1 else 0.0
        ego_lane = self.get_lane_index(ego_y)

        goal_phys = np.asarray(goal_phys, dtype=np.float32).reshape(-1)
        goal_x = float(goal_phys[0])
        goal_vx = float(goal_phys[2])

        _ = goal_x
        _ = goal_vx
        _ = remaining_time
        _ = dt
        target_speed = float(self.speed_limit)

        others: List[VirtualVehicle] = []
        for row in np.asarray(others_rel, dtype=np.float32).reshape(-1, 4):
            dx, dy, dvx, dvy = [float(v) for v in row]
            ox = ego_x + dx
            oy = ego_y + dy
            ovx = ego_vx + dvx
            ovy = ego_vy + dvy
            ospeed = float(math.sqrt(ovx * ovx + ovy * ovy))
            oheading = float(math.atan2(ovy, ovx)) if ospeed > 0.1 else 0.0
            olane = self.get_lane_index(oy)
            others.append(
                VirtualVehicle(
                    position=np.array([ox, oy], dtype=np.float32),
                    velocity=np.array([ovx, ovy], dtype=np.float32),
                    heading=oheading,
                    speed=ospeed,
                    lane_index=olane,
                    target_speed=max(ospeed, 0.1),
                    length=self.LENGTH,
                )
            )

        ego = VirtualVehicle(
            position=np.array([ego_x, ego_y], dtype=np.float32),
            velocity=np.array([ego_vx, ego_vy], dtype=np.float32),
            heading=ego_heading,
            speed=ego_speed,
            lane_index=ego_lane,
            target_speed=target_speed,
            length=self.LENGTH,
        )

        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_type == "ParamLaneAccelAction":
            lane_scalar = float(act[0])
            acc_norm = float(act[1])

            target_lane = self._scalar_to_lane(ego_lane, lane_scalar)
            if target_lane != ego_lane:
                if not self.mobil_ok(ego, target_lane, others):
                    lane_scalar = 0.0
                    target_lane = ego_lane

            acc_phys = self._acc_norm_to_phys(acc_norm)
            front, _ = self._get_neighbors(ego, others, ego_lane)
            acc_idm = self.idm_acceleration(ego, front)
            if target_lane != ego_lane:
                front_t, _ = self._get_neighbors(ego, others, target_lane)
                acc_idm = min(acc_idm, self.idm_acceleration(ego, front_t))

            if acc_idm < acc_phys:
                acc_phys = acc_idm
            acc_phys = float(np.clip(acc_phys, self.acc_min, self.acc_max))
            acc_norm = self._acc_phys_to_norm(acc_phys)
            return np.array([lane_scalar, acc_norm], dtype=np.float32)

        return act.astype(np.float32)

    def _compute_action_idm_mobil(
        self,
        ego_abs: np.ndarray,
        others_rel: np.ndarray,
        goal_phys: np.ndarray,
        dt: float,
        remaining_time: Optional[float] = None,
    ) -> np.ndarray:
        """完整规则：使用 IDM/MOBIL 与目标约束。"""

        ego = self._as_ego_abs4(ego_abs)
        ego_x, ego_y, ego_vx, ego_vy = [float(v) for v in ego]
        ego_speed = float(math.sqrt(ego_vx * ego_vx + ego_vy * ego_vy))
        ego_heading = float(math.atan2(ego_vy, ego_vx)) if ego_speed > 0.1 else 0.0
        ego_lane = self.get_lane_index(ego_y)

        others: List[VirtualVehicle] = []
        for row in np.asarray(others_rel, dtype=np.float32).reshape(-1, 4):
            dx, dy, dvx, dvy = [float(v) for v in row]
            ox = ego_x + dx
            oy = ego_y + dy
            ovx = ego_vx + dvx
            ovy = ego_vy + dvy
            ospeed = float(math.sqrt(ovx * ovx + ovy * ovy))
            oheading = float(math.atan2(ovy, ovx)) if ospeed > 0.1 else 0.0
            olane = self.get_lane_index(oy)
            others.append(
                VirtualVehicle(
                    position=np.array([ox, oy], dtype=np.float32),
                    velocity=np.array([ovx, ovy], dtype=np.float32),
                    heading=oheading,
                    speed=ospeed,
                    lane_index=olane,
                    target_speed=max(ospeed, 0.1),
                    length=self.LENGTH,
                )
            )

        goal_x = float(goal_phys[0])
        goal_y = float(goal_phys[1])
        goal_vx = float(goal_phys[2])

        if remaining_time is not None:
            rt = max(float(remaining_time), float(dt))
            target_speed = max((goal_x - ego_x) / rt, 0.0)
        else:
            target_speed = abs(goal_vx)
        target_speed = float(np.clip(target_speed, 0.0, self.speed_limit))

        ego = VirtualVehicle(
            position=np.array([ego_x, ego_y], dtype=np.float32),
            velocity=np.array([ego_vx, ego_vy], dtype=np.float32),
            heading=ego_heading,
            speed=ego_speed,
            lane_index=ego_lane,
            target_speed=target_speed,
            length=self.LENGTH,
        )

        target_lane = self.get_lane_index(goal_y)
        if abs(target_lane - ego_lane) > 1:
            target_lane = ego_lane + int(np.sign(target_lane - ego_lane))
            target_lane = int(np.clip(target_lane, 0, self.lanes_count - 1))

        if target_lane != ego_lane and not self.mobil_ok(ego, target_lane, others):
            target_lane = ego_lane

        acc_pid = float(self.KP_A) * (target_speed - ego_speed)
        front, _ = self._get_neighbors(ego, others, ego_lane)
        acc_idm = self.idm_acceleration(ego, front)
        if target_lane != ego_lane:
            front_t, _ = self._get_neighbors(ego, others, target_lane)
            acc_idm = min(acc_idm, self.idm_acceleration(ego, front_t))

        acc_phys = acc_pid if acc_idm >= float(self.COMFORT_ACC_MIN) else acc_idm
        acc_phys = float(np.clip(acc_phys, self.acc_min, self.acc_max))

        if self.action_type == "ParamLaneAccelAction":
            lane_scalar = self._lane_to_scalar(ego_lane, target_lane)
            acc_norm = self._acc_phys_to_norm(acc_phys)
            return np.array([lane_scalar, acc_norm], dtype=np.float32)

        target_lane_y = self.get_lane_y(target_lane)
        lateral_error = float(target_lane_y - ego_y)
        lateral_speed_command = float(self.KP_LATERAL) * lateral_error
        v_s = float(c_utils.not_zero(ego_speed))
        heading_ref = float(math.asin(float(np.clip(lateral_speed_command / v_s, -1.0, 1.0))))
        heading_rate_command = float(self.KP_HEADING) * float(c_utils.wrap_to_pi(heading_ref - ego_heading))
        slip_angle = float(math.asin(float(np.clip(self.LENGTH / 2.0 / v_s * heading_rate_command, -1.0, 1.0))))
        steering_angle = float(math.atan(2.0 * math.tan(slip_angle)))
        steering_angle = float(np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE))

        acc_norm = self._acc_phys_to_norm(acc_phys)
        steer_min, steer_max = float(self.steer_range[0]), float(self.steer_range[1])
        if steer_max == steer_min:
            steer_norm = 0.0
        else:
            steer_norm = 2.0 * (steering_angle - steer_min) / (steer_max - steer_min) - 1.0
        steer_norm = float(np.clip(steer_norm, -1.0, 1.0))
        return np.array([acc_norm, steer_norm], dtype=np.float32)

    def _compute_action_goal_x_accel(
        self,
        ego_abs: np.ndarray,
        others_rel: np.ndarray,
        goal_phys: np.ndarray,
        dt: float,
        remaining_time: Optional[float] = None,
    ) -> np.ndarray:
        """简化规则：按匀加速到目标 x，不使用 IDM/MOBIL。"""

        _ = others_rel

        ego = self._as_ego_abs4(ego_abs)
        ego_x, ego_y, ego_vx, ego_vy = [float(v) for v in ego]
        ego_speed = float(math.sqrt(ego_vx * ego_vx + ego_vy * ego_vy))
        ego_heading = float(math.atan2(ego_vy, ego_vx)) if ego_speed > 0.1 else 0.0
        ego_lane = self.get_lane_index(ego_y)

        goal_x = float(goal_phys[0])
        goal_y = float(goal_phys[1])

        rt = max(float(remaining_time), float(dt)) if remaining_time is not None else float(dt)
        acc_phys = 2.0 * (goal_x - ego_x - ego_vx * rt) / max(rt * rt, 1e-6)

        target_lane = self.get_lane_index(goal_y)
        if abs(target_lane - ego_lane) > 1:
            target_lane = ego_lane + int(np.sign(target_lane - ego_lane))
            target_lane = int(np.clip(target_lane, 0, self.lanes_count - 1))

        acc_phys = float(np.clip(acc_phys, self.acc_min, self.acc_max))

        if self.action_type == "ParamLaneAccelAction":
            lane_scalar = self._lane_to_scalar(ego_lane, target_lane)
            acc_norm = self._acc_phys_to_norm(acc_phys)
            return np.array([lane_scalar, acc_norm], dtype=np.float32)

        target_lane_y = self.get_lane_y(target_lane)
        lateral_error = float(target_lane_y - ego_y)
        lateral_speed_command = float(self.KP_LATERAL) * lateral_error
        v_s = float(c_utils.not_zero(ego_speed))
        heading_ref = float(math.asin(float(np.clip(lateral_speed_command / v_s, -1.0, 1.0))))
        heading_rate_command = float(self.KP_HEADING) * float(c_utils.wrap_to_pi(heading_ref - ego_heading))
        slip_angle = float(math.asin(float(np.clip(self.LENGTH / 2.0 / v_s * heading_rate_command, -1.0, 1.0))))
        steering_angle = float(math.atan(2.0 * math.tan(slip_angle)))
        steering_angle = float(np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE))

        acc_norm = self._acc_phys_to_norm(acc_phys)
        steer_min, steer_max = float(self.steer_range[0]), float(self.steer_range[1])
        if steer_max == steer_min:
            steer_norm = 0.0
        else:
            steer_norm = 2.0 * (steering_angle - steer_min) / (steer_max - steer_min) - 1.0
        steer_norm = float(np.clip(steer_norm, -1.0, 1.0))
        return np.array([acc_norm, steer_norm], dtype=np.float32)

    def _compute_action_target_speed_lane(
        self,
        ego_abs: np.ndarray,
        others_rel: np.ndarray,
        goal_phys: np.ndarray,
        dt: float,
        remaining_time: Optional[float] = None,
    ) -> np.ndarray:
        """简化版动作计算：不使用 IDM/MOBIL，直接按目标速度与目标车道输出动作。"""

        _ = others_rel

        ego = self._as_ego_abs4(ego_abs)
        ego_x, ego_y, ego_vx, ego_vy = [float(v) for v in ego]
        ego_speed = float(math.sqrt(ego_vx * ego_vx + ego_vy * ego_vy))
        ego_heading = float(math.atan2(ego_vy, ego_vx)) if ego_speed > 0.1 else 0.0
        ego_lane = self.get_lane_index(ego_y)

        goal_x = float(goal_phys[0])
        goal_y = float(goal_phys[1])
        goal_vx = float(goal_phys[2])

        if remaining_time is not None:
            rt = max(float(remaining_time), float(dt))
            target_speed = max((goal_x - ego_x) / rt, 0.0)
        else:
            target_speed = abs(goal_vx)
        target_speed = float(np.clip(target_speed, 0.0, self.speed_limit))

        target_lane = self.get_lane_index(goal_y)
        if abs(target_lane - ego_lane) > 1:
            target_lane = ego_lane + int(np.sign(target_lane - ego_lane))
            target_lane = int(np.clip(target_lane, 0, self.lanes_count - 1))

        acc_phys = float(self.KP_A) * (target_speed - ego_speed)
        acc_phys = float(np.clip(acc_phys, self.acc_min, self.acc_max))

        if self.action_type == "ParamLaneAccelAction":
            lane_scalar = self._lane_to_scalar(ego_lane, target_lane)
            acc_norm = self._acc_phys_to_norm(acc_phys)
            return np.array([lane_scalar, acc_norm], dtype=np.float32)

        target_lane_y = self.get_lane_y(target_lane)
        lateral_error = float(target_lane_y - ego_y)

        lateral_speed_command = float(self.KP_LATERAL) * lateral_error
        v_s = float(c_utils.not_zero(ego_speed))
        heading_ref = float(math.asin(float(np.clip(lateral_speed_command / v_s, -1.0, 1.0))))
        heading_rate_command = float(self.KP_HEADING) * float(c_utils.wrap_to_pi(heading_ref - ego_heading))
        slip_angle = float(math.asin(float(np.clip(self.LENGTH / 2.0 / v_s * heading_rate_command, -1.0, 1.0))))
        steering_angle = float(math.atan(2.0 * math.tan(slip_angle)))
        steering_angle = float(np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE))

        acc_norm = self._acc_phys_to_norm(acc_phys)
        steer_min, steer_max = float(self.steer_range[0]), float(self.steer_range[1])
        if steer_max == steer_min:
            steer_norm = 0.0
        else:
            steer_norm = 2.0 * (steering_angle - steer_min) / (steer_max - steer_min) - 1.0
        steer_norm = float(np.clip(steer_norm, -1.0, 1.0))
        return np.array([acc_norm, steer_norm], dtype=np.float32)

    def compute_action(
        self,
        ego_abs: np.ndarray,
        others_rel: np.ndarray,
        goal_phys: np.ndarray,
        dt: float,
        remaining_time: Optional[float] = None,
    ) -> np.ndarray:
        """从配置选择 rule-based 动作策略。"""
        mode = self.compute_action_mode
        if mode in {"idm_mobil", "full"}:
            return self._compute_action_idm_mobil(ego_abs, others_rel, goal_phys, dt, remaining_time=remaining_time)
        if mode in {"goal_x_accel", "const_accel_to_goal_x", "goal_x_accel_follow", "follow_goal_x_accel"}:
            return self._compute_action_goal_x_accel(ego_abs, others_rel, goal_phys, dt, remaining_time=remaining_time)
        if mode in {"target_speed_lane", "simple"}:
            return self._compute_action_target_speed_lane(ego_abs, others_rel, goal_phys, dt, remaining_time=remaining_time)
        raise ValueError(
            f"Unknown rule_based_compute_action_mode: {mode}. "
            "Expected one of: idm_mobil, goal_x_accel, goal_x_accel_follow, target_speed_lane"
        )


class RuleBasedAgentWrapper:
    def __init__(self, vec_env, n_envs: int, high_interval: int, low_safety_filter: Any = None):
        self.vec_env = vec_env
        self.n_envs = int(n_envs)
        self.high_interval = int(high_interval)

        # SubprocVecEnv 也支持 get_attr
        env_cfg = vec_env.get_attr("config", indices=0)[0]
        self.dt = 1.0 / float(env_cfg.get("policy_frequency", 10.0))

        obs_cfg = dict(env_cfg.get("observation", {}) or {})
        self.n_veh_local = int(obs_cfg.get("vehicles_count_local", obs_cfg.get("vehicles_count", 5)))
        self.feature_names = list(obs_cfg.get("features", ["presence", "x", "y", "vx", "vy", "acceleration"]))
        self.feat_dim = int(len(self.feature_names))
        self.append_front_vehicle_features = bool(
            obs_cfg.get("append_front_vehicle_features", False)
        )
        self.obs_extra_dim = (
            (2 if self.append_front_vehicle_features else 0)
            + (1 if bool(obs_cfg.get("append_goal_lane_id", False)) else 0)
        )
        self.obs_extra_normalize = bool(obs_cfg.get("normalize", False))
        self.front_distance_range = float(obs_cfg.get("front_vehicle_distance_range", 150.0))
        self.front_ttc_range = float(obs_cfg.get("front_vehicle_ttc_range", 30.0))

        self.controller = RuleBasedController(env_cfg, low_safety_filter=low_safety_filter)
        mode = str(env_cfg.get("rule_based_compute_action_mode", "")).lower().strip()
        self.follow_mode_enabled = mode == "goal_x_accel_follow"
        self.follow_enter_gap = float(env_cfg.get("rule_follow_enter_gap", 18.0))
        self.follow_release_gap = float(env_cfg.get("rule_follow_release_gap", 23.0))
        self.follow_enter_ttc = float(env_cfg.get("rule_follow_enter_ttc", 2.0))
        self.follow_release_ttc = float(env_cfg.get("rule_follow_release_ttc", 4.0))
        self.follow_max_acc = float(env_cfg.get("rule_follow_max_acc", 0.0))
        self.follow_same_lane_dy = float(env_cfg.get("rule_follow_same_lane_dy", 0.5 * float(env_cfg.get("lane_width", 4.0))))
        self.follow_reset_on_high_interval = bool(env_cfg.get("rule_follow_reset_on_high_interval", True))
        self._follow_active = np.zeros(self.n_envs, dtype=bool)

        # feature indices
        def _idx(name: str, default: int) -> int:
            try:
                return int(self.feature_names.index(name))
            except ValueError:
                return int(default)

        self.idx_presence = _idx("presence", -1)
        self.idx_x = _idx("x", 0)
        self.idx_y = _idx("y", 1)
        self.idx_vx = _idx("vx", 2)
        self.idx_vy = _idx("vy", 3)
        self.goal_dim = 4

    def _decode_front_extra(self, extra: np.ndarray) -> tuple[float, float] | None:
        if not self.append_front_vehicle_features:
            return None
        vals = np.asarray(extra, dtype=np.float32).reshape(-1)
        if vals.size < 2:
            return None
        distance = float(vals[0])
        ttc = float(vals[1])
        if self.obs_extra_normalize:
            distance = 0.5 * (distance + 1.0) * max(self.front_distance_range, 1e-6)
            ttc = 0.5 * (ttc + 1.0) * max(self.front_ttc_range, 1e-6)
        distance = float(np.clip(distance, 0.0, max(self.front_distance_range, 1e-6)))
        ttc = float(np.clip(ttc, 0.0, max(self.front_ttc_range, 1e-6)))
        if distance >= self.front_distance_range - 1e-6 and ttc >= self.front_ttc_range - 1e-6:
            return None
        return distance, ttc

    def _augment_others_with_front_extra(
        self,
        others_rel: np.ndarray,
        ego_vx: float,
        extra: np.ndarray,
    ) -> np.ndarray:
        decoded = self._decode_front_extra(extra)
        if decoded is None:
            return np.asarray(others_rel, dtype=np.float32).reshape(-1, 4)
        distance, ttc = decoded
        closing = 0.0 if ttc >= self.front_ttc_range - 1e-6 else distance / max(ttc, 1e-6)
        rel_vx = -max(float(closing), 0.0)
        synthetic = np.array([[float(distance), 0.0, rel_vx, 0.0]], dtype=np.float32)
        base = np.asarray(others_rel, dtype=np.float32).reshape(-1, 4)
        return np.concatenate([base, synthetic], axis=0)

    def _front_follow_metrics(
        self,
        others_rel: np.ndarray,
        ego_vx: float,
        extra: np.ndarray,
    ) -> tuple[float | None, float | None]:
        """Return nearest same-lane front center distance and TTC for follow-mode."""
        best_gap: float | None = None
        best_ttc: float | None = None

        for row in np.asarray(others_rel, dtype=np.float32).reshape(-1, 4):
            dx, dy, dvx, _dvy = [float(v) for v in row]
            if dx <= 0.0 or abs(dy) > self.follow_same_lane_dy:
                continue
            gap = max(dx, 0.0)
            closing = max(-dvx, 0.0)
            ttc = gap / max(closing, 1e-6) if closing > 1e-6 else self.front_ttc_range
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_ttc = ttc

        decoded = self._decode_front_extra(extra)
        if decoded is not None:
            gap, ttc = decoded
            if best_gap is None or gap < best_gap:
                best_gap = float(gap)
                best_ttc = float(ttc)

        return best_gap, best_ttc

    def _update_follow_active(self, env_i: int, gap: float | None, ttc: float | None, use_state: bool = True) -> bool:
        if not self.follow_mode_enabled:
            return False
        if gap is None:
            if use_state and 0 <= env_i < self._follow_active.size:
                self._follow_active[env_i] = False
            return False

        if not use_state or env_i < 0 or env_i >= self._follow_active.size:
            ttc_val = self.front_ttc_range if ttc is None else float(ttc)
            return bool(gap <= self.follow_enter_gap or ttc_val <= self.follow_enter_ttc)

        active = bool(self._follow_active[env_i])
        ttc_val = self.front_ttc_range if ttc is None else float(ttc)
        if active:
            release = gap >= self.follow_release_gap and ttc_val >= self.follow_release_ttc
            if release:
                active = False
        else:
            enter = gap <= self.follow_enter_gap or ttc_val <= self.follow_enter_ttc
            if enter:
                active = True
        self._follow_active[env_i] = active
        return active

    def _reset_follow_active_at_interval_start(self, env_i: int, t_norm: float, use_state: bool) -> None:
        if (
            not use_state
            or not self.follow_reset_on_high_interval
            or env_i < 0
            or env_i >= self._follow_active.size
        ):
            return
        if float(t_norm) <= 1e-6:
            self._follow_active[env_i] = False

    def _apply_follow_mode_cap(self, action: np.ndarray) -> np.ndarray:
        safe = np.asarray(action, dtype=np.float32).reshape(-1).copy()
        if safe.size < 2:
            return safe
        acc_phys = self.controller._acc_norm_to_phys(float(safe[1]))
        acc_phys = min(acc_phys, self.follow_max_acc)
        safe[1] = self.controller._acc_phys_to_norm(acc_phys)
        return safe.astype(np.float32)

    def act(self, low_obs: np.ndarray, goal_phys: np.ndarray) -> np.ndarray:
        low_obs = np.asarray(low_obs, dtype=np.float32)
        goal_phys = np.asarray(goal_phys, dtype=np.float32)

        # low_obs = [t_norm, local_kin_flat, goal_rel]
        kin_slice = low_obs[:, : 1 + self.n_veh_local * self.feat_dim]
        _, kin, _ = rl_utils.split_time_kinematics(kin_slice, self.n_veh_local, self.feat_dim)

        actions: List[np.ndarray] = []
        for i in range(int(low_obs.shape[0])):
            t_norm = float(low_obs[i, 0])
            rem_time = float(self.high_interval) * (1.0 - t_norm) * float(self.dt)

            ego_feat = kin[i, 0]
            ego_abs = np.array(
                [
                    ego_feat[self.idx_x],
                    ego_feat[self.idx_y],
                    ego_feat[self.idx_vx],
                    ego_feat[self.idx_vy],
                ],
                dtype=np.float32,
            )

            others_feat = kin[i, 1:]
            others_rel: List[List[float]] = []
            for j in range(int(others_feat.shape[0])):
                d = others_feat[j]
                if d[self.idx_presence] == 0:
                    continue
                others_rel.append([float(d[self.idx_x]), float(d[self.idx_y]), float(d[self.idx_vx]), float(d[self.idx_vy])])

            others_rel_arr = np.asarray(others_rel, dtype=np.float32).reshape(-1, 4)
            extra_start = int(1 + self.n_veh_local * self.feat_dim)
            extra = low_obs[i, extra_start : extra_start + self.obs_extra_dim]
            others_rel_arr = self._augment_others_with_front_extra(others_rel_arr, float(ego_abs[2]), extra)
            a = self.controller.compute_action(ego_abs, others_rel_arr, goal_phys[i], self.dt, remaining_time=rem_time)
            actions.append(a)

        return np.asarray(actions, dtype=np.float32)

    def apply_safety_layer(self, low_obs: np.ndarray, goal_phys: np.ndarray, action: np.ndarray) -> np.ndarray:
        low_obs = np.asarray(low_obs, dtype=np.float32)
        goal_phys = np.asarray(goal_phys, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)

        kin_slice = low_obs[:, : 1 + self.n_veh_local * self.feat_dim]
        _, kin, _ = rl_utils.split_time_kinematics(kin_slice, self.n_veh_local, self.feat_dim)
        use_follow_state = int(low_obs.shape[0]) == int(self.n_envs)

        safe_actions: List[np.ndarray] = []
        for i in range(int(low_obs.shape[0])):
            t_norm = float(low_obs[i, 0])
            self._reset_follow_active_at_interval_start(i, t_norm, use_follow_state)
            rem_time = float(self.high_interval) * (1.0 - t_norm) * float(self.dt)

            ego_feat = kin[i, 0]
            ego_vel = np.array(
                [
                    ego_feat[self.idx_vx],
                    ego_feat[self.idx_vy],
                ],
                dtype=np.float32,
            )

            others_feat = kin[i, 1:]
            others_rel: List[List[float]] = []
            for j in range(int(others_feat.shape[0])):
                d = others_feat[j]
                if d[self.idx_presence] == 0:
                    continue
                others_rel.append([float(d[self.idx_x]), float(d[self.idx_y]), float(d[self.idx_vx]), float(d[self.idx_vy])])

            others_rel_arr = np.asarray(others_rel, dtype=np.float32).reshape(-1, 4)
            extra_start = int(1 + self.n_veh_local * self.feat_dim)
            extra = low_obs[i, extra_start : extra_start + self.obs_extra_dim]
            front_gap, front_ttc = self._front_follow_metrics(others_rel_arr, float(ego_vel[0]), extra)
            others_rel_arr = self._augment_others_with_front_extra(others_rel_arr, float(ego_vel[0]), extra)
            g0 = int(1 + self.n_veh_local * self.feat_dim + self.obs_extra_dim)
            g1 = int(g0 + self.goal_dim)
            goal_rel = low_obs[i, g0:g1]
            ego_abs = (np.asarray(goal_phys[i], dtype=np.float32) - np.asarray(goal_rel, dtype=np.float32)).astype(np.float32)
            safe_a = self.controller.safety_filter_action(
                ego_abs=ego_abs,
                others_rel=others_rel_arr,
                goal_phys=goal_phys[i],
                action=action[i],
                dt=self.dt,
                remaining_time=rem_time,
            )
            if self._update_follow_active(i, front_gap, front_ttc, use_state=use_follow_state):
                safe_a = self._apply_follow_mode_cap(safe_a)
            safe_actions.append(safe_a)

        return np.asarray(safe_actions, dtype=np.float32)

    @property
    def action_space(self):
        return self.vec_env.action_space

    def save(self, path: str):
        # 无可训练参数，保持与 CheckpointCallback 兼容
        return None
