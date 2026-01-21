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
        return float(other.position[0] - self.position[0] - self.length)


class RuleBasedController:
    """基于观测(obs)的 rule-based 低层控制器。

    目标：兼容 SubprocVecEnv（不访问 env.unwrapped.vehicle/road 等不可跨进程对象）。

    支持两类动作空间：
    - ContinuousAction: action = [acc_norm, steer_norm]
    - ParamLaneAccelAction: action = [lane_scalar, acc_norm]
    """

    def __init__(self, env_or_config: Any):
        # Backward-compatible: allow passing a gym env (used by older eval scripts)
        if isinstance(env_or_config, dict):
            env_config = env_or_config
        else:
            env = env_or_config
            env_config = getattr(getattr(env, "unwrapped", env), "config", None)
            if env_config is None:
                raise TypeError("RuleBasedController expects an env with .unwrapped.config or a config dict")

        self.config = dict(env_config)

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

        # Observation meta (for act(obs, ...))
        obs_cfg = dict(self.config.get("observation", {}) or {})
        self.obs_features = list(obs_cfg.get("features", ["presence", "x", "y", "vx", "vy"]))
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

    def safety_filter_action(
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
        ego_x, ego_y, ego_vx, ego_vy = [float(v) for v in np.asarray(ego_abs).reshape(-1)]
        ego_speed = float(math.sqrt(ego_vx * ego_vx + ego_vy * ego_vy))
        ego_heading = float(math.atan2(ego_vy, ego_vx)) if ego_speed > 0.1 else 0.0
        ego_lane = self.get_lane_index(ego_y)

        # Align target_speed with compute_action (use goal and remaining_time if provided)
        goal_phys = np.asarray(goal_phys, dtype=np.float32).reshape(-1)
        goal_x = float(goal_phys[0])
        goal_vx = float(goal_phys[2])
        if remaining_time is not None:
            rt = max(float(remaining_time), float(dt))
            target_speed = max((goal_x - ego_x) / rt, 0.0)
        else:
            target_speed = abs(goal_vx)
        target_speed = float(np.clip(target_speed, 0.0, self.speed_limit))

        # Build other vehicles in absolute frame
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

    def compute_action(
        self,
        ego_abs: np.ndarray,
        others_rel: np.ndarray,
        goal_phys: np.ndarray,
        dt: float,
        remaining_time: Optional[float] = None,
    ) -> np.ndarray:
        """从 (ego, others, goal) 计算一个动作。

        ego_abs: [x, y, vx, vy] (绝对)
        others_rel: (M, 4) [dx, dy, dvx, dvy] (相对 ego)
        goal_phys: [x*, y*, vx*, vy*] (绝对目标)
        """

        ego_x, ego_y, ego_vx, ego_vy = [float(v) for v in ego_abs]
        ego_speed = float(math.sqrt(ego_vx * ego_vx + ego_vy * ego_vy))
        ego_heading = float(math.atan2(ego_vy, ego_vx)) if ego_speed > 0.1 else 0.0
        ego_lane = self.get_lane_index(ego_y)

        # Reconstruct other vehicles in absolute frame for neighbor queries
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

        # Parse goal
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

        # Desired lane (nearest lane center to goal_y)
        target_lane = self.get_lane_index(goal_y)
        if abs(target_lane - ego_lane) > 1:
            target_lane = ego_lane + int(np.sign(target_lane - ego_lane))
            target_lane = int(np.clip(target_lane, 0, self.lanes_count - 1))

        # MOBIL safety/incentive gate
        if target_lane != ego_lane:
            if not self.mobil_ok(ego, target_lane, others):
                target_lane = ego_lane

        # Longitudinal control: PID towards target speed
        acc_pid = float(self.KP_A) * (target_speed - ego_speed)

        # IDM safety constraint
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

        # Default: ContinuousAction [acc_norm, steer_norm]
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


class RuleBasedAgentWrapper:
    """给 HIRO 调用的 low-level wrapper。

    - 输入：HIRO 的 low_obs（包含 t_norm + local kinematics + goal_rel）
    - 输出：env action space 对应的 2D action
    """

    def __init__(self, vec_env, n_envs: int, high_interval: int):
        self.vec_env = vec_env
        self.n_envs = int(n_envs)
        self.high_interval = int(high_interval)

        # SubprocVecEnv 也支持 get_attr
        env_cfg = vec_env.get_attr("config", indices=0)[0]
        self.dt = 1.0 / float(env_cfg.get("policy_frequency", 10.0))

        obs_cfg = dict(env_cfg.get("observation", {}) or {})
        self.n_veh_local = int(obs_cfg.get("vehicles_count_local", obs_cfg.get("vehicles_count", 5)))
        self.feature_names = list(obs_cfg.get("features", ["presence", "x", "y", "vx", "vy"]))
        self.feat_dim = int(len(self.feature_names))

        self.controller = RuleBasedController(env_cfg)

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
            a = self.controller.compute_action(ego_abs, others_rel_arr, goal_phys[i], self.dt, remaining_time=rem_time)
            actions.append(a)

        return np.asarray(actions, dtype=np.float32)

    def apply_safety_layer(self, low_obs: np.ndarray, goal_phys: np.ndarray, action: np.ndarray) -> np.ndarray:
        low_obs = np.asarray(low_obs, dtype=np.float32)
        goal_phys = np.asarray(goal_phys, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)

        kin_slice = low_obs[:, : 1 + self.n_veh_local * self.feat_dim]
        _, kin, _ = rl_utils.split_time_kinematics(kin_slice, self.n_veh_local, self.feat_dim)

        safe_actions: List[np.ndarray] = []
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
            safe_a = self.controller.safety_filter_action(
                ego_abs,
                others_rel_arr,
                goal_phys[i],
                action[i],
                self.dt,
                remaining_time=rem_time,
            )
            safe_actions.append(safe_a)

        return np.asarray(safe_actions, dtype=np.float32)

    @property
    def action_space(self):
        return self.vec_env.action_space

    def save(self, path: str):
        # 无可训练参数，保持与 CheckpointCallback 兼容
        return None
