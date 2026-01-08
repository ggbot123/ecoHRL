from __future__ import annotations
import numpy as np
from typing import Any, Dict, Tuple
from custom_env import utils as c_utils

class RuleBasedController:
    """
    A controller that mimics ControlledVehicle's logic but operates externally.
    It takes target speed and target lane index (from HIRO goal) and outputs 
    acceleration and steering control actions.
    """
    def __init__(self, env):
        self.env = env
        self.config = env.unwrapped.config
        
        # Determine continuous action bounds from config
        act_config = self.config.get("action", {})
        self.acc_bound = act_config.get("acceleration_range", (-5.0, 5.0))[1]
        self.steer_bound = act_config.get("steering_range", (-np.pi/4, np.pi/4))[1]
        
        # Controller parameters (copied from ControlledVehicle)
        self.TAU_ACC = 0.6  # [s]
        self.TAU_HEADING = 0.2  # [s]
        self.TAU_LATERAL = 0.6  # [s]
        self.TAU_PURSUIT = 0.5 * self.TAU_HEADING  # [s]
        self.KP_A = 1 / self.TAU_ACC
        self.KP_HEADING = 1 / self.TAU_HEADING
        self.KP_LATERAL = 1 / self.TAU_LATERAL  # [1/s]
        self.MAX_STEERING_ANGLE = np.pi / 3  # [rad]
        self.LENGTH = 5.0 # Approximate vehicle length

        # --- IDM/MOBIL Configuration ---
        # Longitudinal (IDM)
        self.ACC_MAX = 6.0              # Max acceleration [m/s2]
        self.COMFORT_ACC_MAX = 3.0      # Desired max acceleration [m/s2]
        self.COMFORT_ACC_MIN = -5.0     # Desired max deceleration [m/s2]
        self.DISTANCE_WANTED = 5.0 + 5.0 # Desired jam distance [m] (5.0 + car_length)
        self.TIME_WANTED = 0.5          # Desired time gap [s]
        self.DELTA = 4.0                # Exponent of the velocity term
        
        # Lateral (MOBIL)
        self.POLITENESS = 0           # Politeness factor [0, 1] (0.5 = considerate)
        self.LANE_CHANGE_MIN_ACC_GAIN = -4  # Min acceleration gain to change lane [m/s2]
        self.LANE_CHANGE_MAX_BRAKING_IMPOSED = 4.0 # Max braking imposed on others [m/s2]

    def acceleration(self, ego_vehicle, front_vehicle=None, rear_vehicle=None) -> float:
        """Compute an acceleration command with the Intelligent Driver Model."""
        if not ego_vehicle:
            return 0
            
        ego_target_speed = getattr(ego_vehicle, "target_speed", 0)
        
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            ego_target_speed = np.clip(ego_target_speed, 0, ego_vehicle.lane.speed_limit)
            
        acceleration = self.COMFORT_ACC_MAX * (
            1 - np.power(max(ego_vehicle.speed, 0) / abs(c_utils.not_zero(ego_target_speed)), self.DELTA)
        )

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * np.power(
                self.desired_gap(ego_vehicle, front_vehicle) / c_utils.not_zero(d), 2
            )
        return acceleration

    def desired_gap(self, ego_vehicle, front_vehicle=None, projected=True) -> float:
        """Compute the desired distance between a vehicle and its leading vehicle."""
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = (
            np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction)
            if projected and hasattr(ego_vehicle, "direction") and hasattr(front_vehicle, "velocity")
            else ego_vehicle.speed - front_vehicle.speed
        )
        d_star = d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        return d_star

    def mobil(self, vehicle, lane_index) -> bool:
        """
        MOBIL lane change decision.
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = vehicle.road.neighbour_vehicles(vehicle, lane_index)
        new_following_a = self.acceleration(ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = self.acceleration(ego_vehicle=new_following, front_vehicle=vehicle)
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = vehicle.road.neighbour_vehicles(vehicle)
        self_pred_a = self.acceleration(ego_vehicle=vehicle, front_vehicle=new_preceding)
        
        route = getattr(vehicle, "route", None)
        target_lane_index = getattr(vehicle, "target_lane_index", None)
        
        if route and route[0][2] is not None and target_lane_index:
             # Wrong direction
            if np.sign(lane_index[2] - target_lane_index[2]) != np.sign(route[0][2] - target_lane_index[2]):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=vehicle, front_vehicle=old_preceding)
            old_following_a = self.acceleration(ego_vehicle=old_following, front_vehicle=vehicle)
            old_following_pred_a = self.acceleration(ego_vehicle=old_following, front_vehicle=old_preceding)
            jerk = (
                self_pred_a
                - self_a
                + self.POLITENESS
                * (
                    new_following_pred_a
                    - new_following_a
                    + old_following_pred_a
                    - old_following_a
                )
            )
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        return True

    def act(self, obs: Dict[str, Any], goal_phys: np.ndarray) -> np.ndarray:
        vehicle = self.env.unwrapped.vehicle
        if vehicle is None:
            return np.array([0.0, 0.0], dtype=np.float32)

        # 1. Parse Goal
        target_vx = goal_phys[2]
        target_vy = goal_phys[3] if len(goal_phys) > 3 else 0.0
        target_speed = np.sqrt(target_vx**2 + target_vy**2) 
        target_y = goal_phys[1]
        
        # Find closest lane to target_y
        query_pos = np.array([vehicle.position[0], target_y])
        raw_target_lane_index = self.env.unwrapped.road.network.get_closest_lane_index(query_pos)
        
        vehicle.target_speed = target_speed

        # --- Lateral Safety Check (MOBIL) ---
        target_lane_index = raw_target_lane_index
        
        if target_lane_index != vehicle.lane_index:
            if not self.env.unwrapped.road.network.get_lane(target_lane_index).is_reachable_from(vehicle.position):
                target_lane_index = vehicle.lane_index
            else:
                if not self.mobil(vehicle, target_lane_index):
                    target_lane_index = vehicle.lane_index
        
        # 2. Compute Controls (Mimic ControlledVehicle.act)
        
        # --- Longitudinal Control (Speed) ---
        acc_pid = self.KP_A * (target_speed - vehicle.speed)
        
        # --- Longitudinal Safety Check (IDM) ---
        front, rear = self.env.unwrapped.road.neighbour_vehicles(vehicle, vehicle.lane_index)
        acc_idm = self.acceleration(vehicle, front, rear)
        
        if target_lane_index != vehicle.lane_index:
             front_t, rear_t = self.env.unwrapped.road.neighbour_vehicles(vehicle, target_lane_index)
             acc_idm_target = self.acceleration(vehicle, front_t, rear_t)
             acc_idm = min(acc_idm, acc_idm_target)
             
        # Combined Acceleration: Apply safety constraint
        if acc_idm < self.COMFORT_ACC_MIN:
            acceleration = acc_idm
        else:
            acceleration = acc_pid
        
        # --- Lateral Control (Steering) ---
        target_lane = self.env.unwrapped.road.network.get_lane(target_lane_index)
        
        lane_coords = target_lane.local_coordinates(vehicle.position)
        lane_next_coords = lane_coords[0] + vehicle.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        lateral_speed_command = -self.KP_LATERAL * lane_coords[1]
        v_s = c_utils.not_zero(vehicle.speed)
        heading_command = np.arcsin(np.clip(lateral_speed_command / v_s, -1, 1))
        
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi / 4, np.pi / 4)
        heading_rate_command = self.KP_HEADING * c_utils.wrap_to_pi(heading_ref - vehicle.heading)
        slip_angle = np.arcsin(np.clip(self.LENGTH / 2 / v_s * heading_rate_command, -1, 1))
        
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # 3. Normalize for Action Space
        acc_norm = np.clip(acceleration / self.acc_bound, -1.0, 1.0)
        steer_norm = np.clip(steering_angle / self.steer_bound, -1.0, 1.0)
        
        return np.array([acc_norm, steer_norm], dtype=np.float32)

class RuleBasedAgentWrapper:
    def __init__(self, vec_env, n_envs):
        self.vec_env = vec_env
        self.n_envs = n_envs
        self.controllers = []
        
        if hasattr(vec_env, "envs"):
             for i in range(n_envs):
                 self.controllers.append(RuleBasedController(vec_env.envs[i]))
        else:
             import warnings
             warnings.warn("RuleBasedAgentWrapper: vec_env does not have .envs attribute. RuleBasedController might fail if it relies on internal env access. Ensure you are using DummyVecEnv.")

    def act(self, obs, goal_phys):
        actions = []
        for i in range(self.n_envs):
            # goal_phys[i] is [x, y, vx, vy]
            if i < len(self.controllers):
                a = self.controllers[i].act(None, goal_phys[i])
                actions.append(a)
            else:
                # Fallback zero
                actions.append(np.zeros(2))
        return np.array(actions, dtype=np.float32)

    @property
    def action_space(self):
        return self.vec_env.action_space
