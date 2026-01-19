import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import scenarios.multi_lane  # Trigger registration

import numpy as np
import os
import shutil
from typing import Any, Dict, Optional, Sequence, Tuple

from util.plot_result import *
from util.hiro_utils import load_hiro_high_model, unique_path

from rl.utils import utils
from custom_env import utils as c_utils
from rl.algos.HRL.hiro_infer import HIROPolicyRunner
from rl.algos.HRL.rule_based import RuleBasedController
from configs.conf import get_env_config, get_hiro_config
from custom_env.vehicle.controller import ControlledVehicle
from custom_env.vehicle.behavior import NormalIDMVehicle
from custom_env.road.road import LaneIndex



def main(
    model_dir: str,
    episodes: int,
    record_episodes: Optional[Sequence[int]] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
    high_model_dir: Optional[str] = None,
):
    eval_dir = os.path.join(model_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    log_path = unique_path(os.path.join(eval_dir, "eval_rule.txt"))
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
    video_dir = os.path.join(model_dir, "goal_distribution_rule_variableTV")
    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
    os.makedirs(video_dir, exist_ok=True)
    
    env = RecordVideo(base_env, video_folder=video_dir, episode_trigger=trigger, name_prefix="rule")

    # 2. Load HIRO High Model (for High-Level Goal Sampling)
    high_model = load_hiro_high_model(high_model_dir or model_dir)
    hiro_cfg = get_hiro_config()
    high_interval = int(getattr(hiro_cfg, "high_interval", 25))
    runner = HIROPolicyRunner(high_model, low_model=None, high_interval=high_interval)

    # 3. Initialize Rule Based Controller
    controller = RuleBasedController(env)

    reward_keys_high = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "punctual_reward", "on_road_reward"]
    # Controller doesn't have intrinsic reward in environment return, but we can compute it
    reward_keys_low = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "on_road_reward", "intrinsic_reward"]

    log("=" * 80)
    log(f"Eval RuleBasedController with HIRO High-Level Goals")
    log(f"Model Dir: {model_dir}")
    log(f"Eval Results Dir: {eval_dir}")
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
            
            # 1. Update Goal via Runner (side-effect: updates runner.goal_phys)
            _ = runner.act(env, obs)
            
            # 2. Get Goal
            goal_phys = runner.goal_phys
            
            # Calculate remaining time in current high-level step for target tracking
            rem_steps = high_interval - runner.c
            dt = 1.0 / env_config["policy_frequency"]
            rem_time = rem_steps * dt

            # 3. Compute Control Action
            action = controller.act(obs, goal_phys, remaining_time=rem_time)

            # Snapshot
            if runner.c == 0:
                save_goal_snapshot(env, runner, ep, steps, model_dir, prev_goal_phys=prev_goal_phys, intrinsic_reward=last_intrinsic_viz, folder_name="goal_distribution_rule_variableTV")

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
        model_dir="./models/hiro_0112_onlyhigh_rule_varTarV", 
        episodes=10,
        record_episodes=[i for i in range(1, 11)],
    )
