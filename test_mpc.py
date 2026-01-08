import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import scenarios.multi_lane  # Trigger registration

import numpy as np
import os
import shutil
from typing import Any, Dict, Optional, Sequence, Tuple

from util.plot_result import *
from util.mpc import MPCController  # Custom MPC

from rl.algos.sac.sac import SAC
from rl.utils import utils
from rl.algos.HRL.hiro_infer import HIROPolicyRunner
from configs.conf import get_env_config, get_hiro_config

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


def main(
    model_dir: str,
    model_name: str,
    episodes: int,
    record_episodes: Optional[Sequence[int]] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
):
    log_path = os.path.join(model_dir, "eval_mpc.txt")
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
    video_dir = os.path.join(model_dir, "goal_distribution_mpc")
    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
    os.makedirs(video_dir, exist_ok=True)

    env = RecordVideo(base_env, video_folder=video_dir, episode_trigger=trigger, name_prefix="mpc")

    # 2. Load HIRO Models (for High-Level Goal Sampling)
    high_model, low_model = _load_hiro_models(model_dir, model_name)
    hiro_cfg = get_hiro_config()
    high_interval = int(getattr(hiro_cfg, "high_interval", 25))
    runner = HIROPolicyRunner(high_model, low_model, high_interval)

    # 3. Initialize MPC
    mpc = MPCController(
        base_env.unwrapped, 
        horizon=high_interval, 
        dt=1.0/base_env.unwrapped.config["policy_frequency"],
        intrinsic_coef=float(getattr(hiro_cfg, "intrinsic_coef", 1.0)),
        intrinsic_weights=getattr(hiro_cfg, "intrinsic_weights", None)
    )
    
    # Action Normalization Helper
    # Assuming standard ContinuousAction with symmetric bounds
    act_config = base_env.unwrapped.config.get("action", {})
    acc_bound = act_config.get("acceleration_range", (-5.0, 5.0))[1] # assume symmetric
    steer_bound = act_config.get("steering_range", (-np.pi/4, np.pi/4))[1]
    
    def normalize_action(phys_action):
        # phys_action: [acc, steering]
        acc_norm = np.clip(phys_action[0] / acc_bound, -1.0, 1.0)
        steer_norm = np.clip(phys_action[1] / steer_bound, -1.0, 1.0)
        return np.array([acc_norm, steer_norm], dtype=np.float32)

    reward_keys_high = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "punctual_reward", "on_road_reward"]
    # MPC doesn't have intrinsic reward in environment return, but we can compute it
    reward_keys_low = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "on_road_reward", "intrinsic_reward"]

    log("=" * 80)
    log(f"Eval MPC with HIRO High-Level Goals")
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
            # Capture prev goal logic
            if runner.need_high:
                 if len(runner.goal_phys) > 0 and not (runner.c == 0 and steps == 0):
                      prev_goal_phys = runner.goal_phys.copy()
            
            # 1. Update Goal via Runner (dummy act call)
            # This ensures goal is sampled and state c is maintained
            _ = runner.act(env, obs)
            
            # 2. Get Goal
            goal_phys = runner.goal_phys
            
            # 3. Plan with MPC using CURRENT goal
            steps_to_goal = runner.hi - runner.c # Steps remaining in HIRO interval
            mpc_action_phys, mpc_pred_next_state = mpc.act(obs, goal_phys, steps_to_goal)
            
            # 4. Normalize for Env
            action = normalize_action(mpc_action_phys)

            # Snapshot
            if runner.c == 0:
                save_goal_snapshot(env, runner, ep, steps, model_dir, prev_goal_phys=prev_goal_phys, intrinsic_reward=last_intrinsic_viz, folder_name="goal_distribution_mpc")

            # Step
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            rc = info.get("reward_components", {})
            punctual = float(rc.get("punctual_reward", 0.0))
            low_ext = float(reward) - punctual

            last_step = bool(done or runner.c == runner.hi - 1)
            # Calculate intrinsic manually since MPC took action (runner.intrinsic_if_last computes it correctly based on transitions and its internal goal)
            # We can still use runner.intrinsic_if_last because it uses runner state (ego_start, goal_phys) and next_obs
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
    log("Summary (MPC Control):")
    log(f"  episodes                : {n}")
    log(f"  mean length             : {float(np.mean(ep_lens)):.3f}")
    log(f"  mean high total         : {float(np.mean(high_ep_rets)):.6f}")
    log(f"  arrival rate            : {arrived_count / n * 100 if n else 0:.2f}%")
    log("=" * 80)
    log_file.close()
    env.close()

if __name__ == "__main__":
    main(
        model_dir="./models/hiro_1e7_lane1_localObs_opc_seed42_0106", # Change to your actual model path
        model_name="hiro",
        episodes=10,
        record_episodes=[1, 2, 3],
    )
