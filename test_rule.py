import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import scenarios.multi_lane  # Trigger registration

import numpy as np
import os
import shutil
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from util.plot_result import *

from rl.algos.sac.sac import SAC
from rl.utils import utils
from custom_env import utils as c_utils
from rl.algos.HRL.hiro_infer import HIROPolicyRunner
from configs.conf import get_env_config, get_hiro_config
from custom_env.vehicle.controller import ControlledVehicle
from custom_env.vehicle.behavior import NormalIDMVehicle
from custom_env.road.road import LaneIndex

def _resolve_hiro_model_paths(model_dir: str, model_name: str) -> Tuple[str, str]:
    name = str(model_name)
    if name.endswith(".zip"):
        if "_high" in name and "_low" not in name:
            high_name = name
            low_name = name.replace("_high_final", "_low_final").replace("_high_", "_low_").replace("_high", "_low", 1)
        elif "_low" in name:
            low_name = name
            high_name = name.replace("_low_final", "_high_final").replace("_low_", "_high_").replace("_low", "_high", 1)
        else:
            prefix = name[:-4]
            high_name, low_name = f"{prefix}_high_final.zip", f"{prefix}_low_final.zip"
    else:
        high_name, low_name = f"{name}_high_final.zip", f"{name}_low_final.zip"
    return os.path.join(model_dir, high_name), os.path.join(model_dir, low_name)


def _load_hiro_models(model_dir: str, model_name: str) -> Tuple[SAC, SAC]:
    high_path, low_path = _resolve_hiro_model_paths(model_dir, model_name)
    return SAC.load(high_path), SAC.load(low_path)

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
            
        # Use target speed from HIRO if available on ego_vehicle (set in act), else internal default?
        # In this external controller, we set `ego_vehicle.target_speed` dynamically or pass it.
        # But IDM logic reads `ego_vehicle.target_speed`.
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
        """
        Calculate low-level control action based on current state and goal.
        
        Args:
            obs: Observation dict (we need direct access to vehicle state, usually not available in obs vector)
                 Wait, this is external control. We need the vehicle object or exact kinematic state.
                 Since we are in a test script, we can access env.vehicle.
            goal_phys: [target_x, target_y, target_vx, target_vy] (absolute coords)
                       HIRO outputs relative goal, but runner.goal_phys is absolute.
                       Wait, HIRO goal is [rel_x, rel_y, vx] usually. 
                       Let's check utils.goal_action_to_abs.
        """
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
        
        # --- IDM/MOBIL Params Injection ---
        # Temporarily inject target_speed into vehicle for IDM calculation
        vehicle.target_speed = target_speed

        # --- Lateral Safety Check (MOBIL) ---
        target_lane_index = raw_target_lane_index
        
        if target_lane_index != vehicle.lane_index:
            if not self.env.unwrapped.road.network.get_lane(target_lane_index).is_reachable_from(vehicle.position):
                target_lane_index = vehicle.lane_index
            else:
                # Check MOBIL
                if not self.mobil(vehicle, target_lane_index):
                    target_lane_index = vehicle.lane_index
        
        # 2. Compute Controls (Mimic ControlledVehicle.act)
        
        # --- Longitudinal Control (Speed) ---
        acc_pid = self.KP_A * (target_speed - vehicle.speed)
        
        # --- Longitudinal Safety Check (IDM) ---
        front, rear = self.env.unwrapped.road.neighbour_vehicles(vehicle, vehicle.lane_index)
        acc_idm = self.acceleration(vehicle, front, rear)
        
        # If changing lane, IDM also checks target lane safety
        if target_lane_index != vehicle.lane_index:
             front_t, rear_t = self.env.unwrapped.road.neighbour_vehicles(vehicle, target_lane_index)
             acc_idm_target = self.acceleration(vehicle, front_t, rear_t)

             acc_idm = min(acc_idm, acc_idm_target)
             
        # Combined Acceleration: Apply safety constraint
        acceleration = min(acc_pid, acc_idm)
        
        # --- Lateral Control (Steering) ---
        target_lane = self.env.unwrapped.road.network.get_lane(target_lane_index)
        
        lane_coords = target_lane.local_coordinates(vehicle.position)
        lane_next_coords = lane_coords[0] + vehicle.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Lateral position control
        lateral_speed_command = -self.KP_LATERAL * lane_coords[1]
        
        # Lateral speed to heading
        v_s = c_utils.not_zero(vehicle.speed)
        heading_command = np.arcsin(np.clip(lateral_speed_command / v_s, -1, 1))
        
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi / 4, np.pi / 4)
        
        # Heading control
        heading_rate_command = self.KP_HEADING * c_utils.wrap_to_pi(heading_ref - vehicle.heading)
        
        # Heading rate to steering angle
        # slip_angle = arcsin(clip(l/2/v * w, -1, 1))
        slip_angle = np.arcsin(np.clip(self.LENGTH / 2 / v_s * heading_rate_command, -1, 1))
        
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)

        # 3. Normalize for Action Space
        # Action space is ContinuousAction: [acceleration, steering] (normalized)
        acc_norm = np.clip(acceleration / self.acc_bound, -1.0, 1.0)
        steer_norm = np.clip(steering_angle / self.steer_bound, -1.0, 1.0)
        
        return np.array([acc_norm, steer_norm], dtype=np.float32)


def main(
    model_dir: str,
    model_name: str,
    episodes: int,
    record_episodes: Optional[Sequence[int]] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
):
    log_path = os.path.join(model_dir, "eval_rule.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    def log(msg: str = ""):
        print(msg)
        log_file.write(msg + "\n")

    test_overrides: Dict[str, Any] = {
        "initial_lane_id": 1,
        "duration": 70.0,
        "warmup_each_episode": False,
        "screen_width": 1800,
        "screen_height": 300,
        "scaling": 3,
        "centering_position": [0.5, 0.5],
        "show_trajectories": True,
        "warmup_render": False,
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
            "acceleration_range": [-5.0, 5.0],
            "steering_range": [-0.7853981633974483, 0.7853981633974483], # [-pi/4, pi/4]
        }
    }
    if env_overrides:
        test_overrides.update(env_overrides)
    env_config = get_env_config(test_overrides)

    if not record_episodes:
        def trigger(ep_id: int) -> bool: return False
    else:
        record_set = {int(ep_idx) - 1 for ep_idx in record_episodes}
        def trigger(ep_id: int) -> bool: return ep_id in record_set

    # 1. Create Environment
    base_env = gym.make("multi-lane-custom-v0", render_mode="rgb_array", config=env_config)
    
    # Define and clear video/result directory
    video_dir = os.path.join(model_dir, "goal_distribution_rule")
    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
    os.makedirs(video_dir, exist_ok=True)
    
    env = RecordVideo(base_env, video_folder=video_dir, episode_trigger=trigger, name_prefix="rule")

    # 2. Load HIRO Models (for High-Level Goal Sampling)
    high_model, low_model = _load_hiro_models(model_dir, model_name)
    hiro_cfg = get_hiro_config()
    high_interval = int(getattr(hiro_cfg, "high_interval", 25))
    runner = HIROPolicyRunner(high_model, low_model, high_interval)

    # 3. Initialize Rule Based Controller
    controller = RuleBasedController(env)

    reward_keys_high = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "punctual_reward", "on_road_reward"]
    # Controller doesn't have intrinsic reward in environment return, but we can compute it
    reward_keys_low = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "on_road_reward", "intrinsic_reward"]

    log("=" * 80)
    log(f"Eval RuleBasedController with HIRO High-Level Goals")
    log(f"Model Dir: {model_dir}")
    log(f"Episodes: {episodes}")
    log("=" * 80)

    ep_lens: list[int] = []
    high_ep_rets: list[float] = []
    low_ep_ext_rets: list[float] = []
    low_ep_int_rets: list[float] = []
    low_ep_total_rets: list[float] = []
    high_comp_sum = {k: 0.0 for k in reward_keys_high}
    low_comp_sum = {k: 0.0 for k in reward_keys_low}

    arrived_count, arrival_times = 0, []
    viewer_initialized = False
    seed_base = 42

    for ep in range(1, int(episodes) + 1):
        obs, _ = env.reset(seed=seed_base + ep)
        runner.reset(env, obs, float(getattr(hiro_cfg, "intrinsic_coef", 1.0)))

        terminated, truncated, steps = False, False, 0
        high_ret, low_ext_ret, low_int_ret, low_total_ret = 0.0, 0.0, 0.0, 0.0
        high_comp = {k: 0.0 for k in reward_keys_high}
        low_comp = {k: 0.0 for k in reward_keys_low}
        high_interval_rets, low_interval_rets = [], []
        cur_high_interval_ret, cur_low_interval_ret = 0.0, 0.0
        
        last_intrinsic_viz = None
        prev_goal_phys = None
        
        if not viewer_initialized:
            class Dummy:
                def __init__(self, pos): self.position = np.array(pos, dtype=float)
            base = env.unwrapped
            base.render()
            base.viewer.observer_vehicle = Dummy([base.config["road_length"] / 2, 5.0])
            viewer_initialized = True

        while not (terminated or truncated):
            if ep == 2 and steps == 25:
                pass

            # Capture prev goal logic
            if runner.need_high:
                 if len(runner.goal_phys) > 0 and not (runner.c == 0 and steps == 0):
                      prev_goal_phys = runner.goal_phys.copy()
            
            # 1. Update Goal via Runner (dummy act call)
            # This ensures goal is sampled and state c is maintained
            _ = runner.act(env, obs)
            
            # 2. Get Goal
            goal_phys = runner.goal_phys
            
            # 3. Compute Control Action
            # Use runner's goal to determine target speed and lane
            # runner.goal_phys is [x, y, vx, vy]
            action = controller.act(obs, goal_phys)

            # Snapshot
            if runner.c == 0:
                save_goal_snapshot(env, runner, ep, steps, model_dir, prev_goal_phys=prev_goal_phys, intrinsic_reward=last_intrinsic_viz, folder_name="goal_distribution_rule")

            # Step
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            rc = info.get("reward_components", {})
            punctual = float(rc.get("punctual_reward", 0.0))
            low_ext = float(reward) - punctual

            last_step = bool(done or runner.c == runner.hi - 1)
            # Calculate intrinsic manually
            intrinsic = runner.intrinsic_if_last(obs_next) if last_step else 0.0
            
            if last_step:
                last_intrinsic_viz = intrinsic

            high_ret += float(reward)
            low_ext_ret += low_ext
            low_int_ret += intrinsic
            low_total_ret += low_ext + intrinsic
            cur_high_interval_ret += float(reward)
            cur_low_interval_ret += low_ext + intrinsic

            for k in reward_keys_high:
                high_comp[k] += float(rc.get(k, 0.0))
            for k in reward_keys_low:
                if k == "intrinsic_reward":
                    low_comp[k] += float(intrinsic)
                elif k == "punctual_reward":
                    continue
                else:
                    low_comp[k] += float(rc.get(k, 0.0))

            steps += 1

            if last_step:
                high_interval_rets.append(float(cur_high_interval_ret))
                low_interval_rets.append(float(cur_low_interval_ret))
                cur_high_interval_ret, cur_low_interval_ret = 0.0, 0.0
            runner.step_end(done)
            obs = obs_next

        n_low_intervals = len(low_interval_rets) or 1
        low_ext_mean = low_ext_ret / float(n_low_intervals)
        low_int_mean = low_int_ret / float(n_low_intervals)
        low_total_mean = low_total_ret / float(n_low_intervals)

        ep_lens.append(int(steps))
        high_ep_rets.append(float(high_ret))
        low_ep_ext_rets.append(float(low_ext_mean))
        low_ep_int_rets.append(float(low_int_mean))
        low_ep_total_rets.append(float(low_total_mean))
        for k in reward_keys_high:
            high_comp_sum[k] += high_comp[k]
        for k in reward_keys_low:
            low_comp_sum[k] += low_comp[k] / float(n_low_intervals)

        base_env = env.unwrapped
        arrived = bool(getattr(base_env, "_has_arrived", False))
        arrival_time = getattr(base_env, "_arrival_time", None)
        if arrived:
            arrived_count += 1
            if arrival_time is not None:
                arrival_times.append(float(arrival_time))

        reason = "terminated" if terminated else ("truncated(time limit)" if truncated else "unknown")
        log("=" * 60)
        log(f"Episode {ep}:")
        log(f"  length                  : {steps}")
        log(f"  high total reward       : {high_ret:.6f}")
        log(f"  low  ext mean           : {low_ext_mean:.6f}")
        log(f"  low  int mean           : {low_int_mean:.6f}")
        log(f"  low  tot mean           : {low_total_mean:.6f}")

        if base_env.config.get("show_trajectories", False):
            save_speed_acc_curves(env, ep_idx=ep, model_path=model_dir)

    n = int(episodes)
    log("=" * 80)
    log("Summary (RuleBased Control):")
    log(f"  episodes                : {n}")
    log(f"  mean length             : {float(np.mean(ep_lens)):.3f}")
    log(f"  mean high total         : {float(np.mean(high_ep_rets)):.6f}")
    log(f"  arrival rate            : {arrived_count / n * 100 if n else 0:.2f}%")
    log("=" * 80)
    log_file.close()
    env.close()

if __name__ == "__main__":
    main(
        model_dir="./models/hiro_1e7_lane1_localObs_opc_seed42_0106", 
        model_name="hiro",
        episodes=10,
        record_episodes=[1, 2, 3],
    )
