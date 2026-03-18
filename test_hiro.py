import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import scenarios.multi_lane  # 触发 __init__.py 里的 register

import numpy as np
import os
import csv
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple

from util.plot_result import *
from util.hiro_utils import unique_path, load_hiro_models
from rl.algos.HRL.hiro_infer import HIROPolicyRunner
from configs.conf import get_env_config, get_hiro_config


def main(
    model_dir: str,
    episodes: int,
    record_episodes: Optional[Sequence[int]] = None,
    record_trajectory_episodes: Optional[Sequence[int]] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
    high_model_dir: Optional[str] = None,
    low_model_dir: Optional[str] = None,
    model_suffix: Optional[str] = "final",
    use_low_safety_layer: Optional[bool] = None,
):
    eval_root_dir = os.path.join(model_dir, "eval_results")
    os.makedirs(eval_root_dir, exist_ok=True)
    run_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = unique_path(os.path.join(eval_root_dir, run_folder_name))
    os.makedirs(eval_dir, exist_ok=True)

    log_path = os.path.join(eval_dir, "eval_hiro.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    def log(msg: str = ""):
        print(msg)
        log_file.write(msg + "\n")

    test_overrides: Dict[str, Any] = {
        "initial_lane_id": 1,
        # "PERCEPTION_DISTANCE": 200,
        # "observation": {
        #     "vehicles_count": 20,
        #     "vehicles_count_local": 5,
        #     "features_range": {
        #         "x": [-200, 200],
        #         "y": [-10, 10],
        #         "vx": [-15, 15],
        #         "vy": [-10, 10],
        #     },
        # },
        "duration": 70.0,
        "warmup_each_episode": False,
        "screen_width": 1800,
        "screen_height": 300,
        "scaling": 3,
        "centering_position": [0.5, 0.5],
        "show_trajectories": True,
        "warmup_render": False,  
    }
    if env_overrides:
        test_overrides.update(env_overrides)
    env_config = get_env_config(test_overrides)

    if not record_episodes:
        def trigger(ep_id: int) -> bool: return False
    else:
        record_set = {int(ep_idx) - 1 for ep_idx in record_episodes}
        def trigger(ep_id: int) -> bool: return ep_id in record_set

    trajectory_record_set = set()
    if record_trajectory_episodes:
        trajectory_record_set = {int(ep_idx) for ep_idx in record_trajectory_episodes}

    base_env = gym.make("multi-lane-custom-v0", render_mode="rgb_array", config=env_config)
    env = RecordVideo(base_env, video_folder=eval_dir, episode_trigger=trigger, name_prefix="hiro")

    high_model, low_model = load_hiro_models(
        model_dir,
        high_model_dir=high_model_dir,
        low_model_dir=low_model_dir,
        model_suffix=model_suffix,
    )
    hiro_cfg = get_hiro_config()
    runner = HIROPolicyRunner(
        high_model,
        low_model,
        int(getattr(hiro_cfg, "high_interval", 25)),
        use_low_safety_layer=use_low_safety_layer,
    )

    reward_keys_high = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "punctual_reward", "on_road_reward"]
    reward_keys_low = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "on_road_reward", "intrinsic_reward"]

    log("=" * 80)
    log(f"Eval HIRO model dir: {model_dir}")
    log(f"Eval run folder    : {run_folder_name}")
    log(f"Eval results dir   : {eval_dir}")
    log(f"Episodes           : {episodes}")
    hd = high_model_dir or model_dir
    ld = low_model_dir or model_dir
    suffix = model_suffix or "final"
    hp = os.path.join(hd, f"hiro_high_{suffix}.zip")
    lp = os.path.join(ld, f"hiro_low_{suffix}.zip")
    log(f"HIRO high          : {hp}")
    log(f"HIRO low           : {lp}")
    log(f"Low safety layer   : {runner.use_low_safety_layer}")
    log(f"High interval      : {runner.hi}")
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
        should_record_trajectory = ep in trajectory_record_set
        trajectory_rows: list[Dict[str, Any]] = []

        def _build_low_obs_for_logging(obs_raw: np.ndarray) -> np.ndarray:
            """Build low_obs exactly as HIROPolicyRunner.act() does for current step."""
            _, kin_local, kin_flat_local = runner._split(obs_raw)
            ego_sub_local = runner._ego_sub(kin_local)
            t_norm_local = np.array([runner.c / float(runner.hi)], dtype=np.float32)
            goal_rel_local = (runner.goal_phys - ego_sub_local).astype(np.float32)

            local_kin_flat_local = np.asarray(
                kin_flat_local[0, : runner.local_kin_flat_dim], dtype=np.float32
            ).copy()

            if bool(getattr(runner.cfg, "mask_ego_position_in_low_obs", False)):
                if int(runner.feat_dim) > 0 and local_kin_flat_local.shape[0] >= int(runner.feat_dim):
                    idx_x_local = int(runner.feature_names.index("x"))
                    idx_y_local = int(runner.feature_names.index("y"))
                    local_kin_flat_local[idx_x_local] = 0.0
                    local_kin_flat_local[idx_y_local] = 0.0

            return np.concatenate([t_norm_local, local_kin_flat_local, goal_rel_local]).astype(np.float32)

        terminated, truncated, steps = False, False, 0
        high_ret, low_ext_ret, low_int_ret, low_total_ret = 0.0, 0.0, 0.0, 0.0
        high_comp = {k: 0.0 for k in reward_keys_high}
        low_comp = {k: 0.0 for k in reward_keys_low}
        high_interval_rets, low_interval_rets = [], []
        cur_high_interval_ret, cur_low_interval_ret = 0.0, 0.0
        
        # Track previous goal and intrinsic reward for visualization
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
            # Capture prev goal before runner.act updates it (if need_high is True)
            if runner.need_high:
                 # Check if we have a valid current goal to save as "previous"
                 if len(runner.goal_phys) > 0 and not (runner.c == 0 and steps == 0):
                      prev_goal_phys = runner.goal_phys.copy()
            
            action = runner.act(env, obs)

            # Snapshot goal at the beginning of each interval (or every few intervals)
            # runner.c is 0 immediately after sampling a new goal.
            if runner.c == 0:
                # k = 1: save every interval
                save_goal_snapshot(env, runner, ep, steps, eval_dir, prev_goal_phys=prev_goal_phys, intrinsic_reward=last_intrinsic_viz)

            obs_next, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            rc = info.get("reward_components", {})
            punctual = float(rc.get("punctual_reward", 0.0))
            low_ext = float(reward) - punctual

            last_step = bool(done or runner.c == runner.hi - 1)
            intrinsic = runner.intrinsic_if_last(obs_next) if last_step else 0.0
            
            if last_step:
                last_intrinsic_viz = intrinsic
                if ep == 4 and steps > 20:
                    pass

            if should_record_trajectory:
                low_obs_now = _build_low_obs_for_logging(obs)
                action_before_safety = np.asarray(getattr(runner, "last_action_pre_safety", action), dtype=np.float32).reshape(-1)
                action_after_safety = np.asarray(getattr(runner, "last_action_post_safety", action), dtype=np.float32).reshape(-1)
                row: Dict[str, Any] = {
                    "episode": int(ep),
                    "step": int(steps),
                    "done": int(done),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "reward": float(reward),
                    "punctual_reward": float(punctual),
                    "low_ext_reward": float(low_ext),
                    "intrinsic_reward": float(intrinsic),
                    "low_total_step_reward": float(low_ext + intrinsic),
                }
                for i, v in enumerate(low_obs_now):
                    row[f"low_obs_{i}"] = float(v)
                for i, v in enumerate(action_before_safety):
                    row[f"action_pre_safety_{i}"] = float(v)
                for i, v in enumerate(action_after_safety):
                    row[f"action_post_safety_{i}"] = float(v)
                trajectory_rows.append(row)

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
        log(f"  length (steps)          : {steps}")
        log(f"  terminated info         : {reason}")
        log(f"  high total reward       : {high_ret:.6f}")
        log(f"  low  ext reward (per interval mean)       : {low_ext_mean:.6f}   (env_reward - punctual)")
        log(f"  low  intrinsic reward (per interval mean) : {low_int_mean:.6f}")
        log(f"  low  total reward (per interval mean)     : {low_total_mean:.6f}   (ext + intrinsic)")
        if high_interval_rets:
            log(f"  high intervals          : {len(high_interval_rets)}  (mean={float(np.mean(high_interval_rets)):.6f})")
        if low_interval_rets:
            log(f"  low  intervals          : {len(low_interval_rets)}  (mean={float(np.mean(low_interval_rets)):.6f})")

        log("  high reward components (sum over episode):")
        for k in reward_keys_high:
            log(f"    {k:18s}: {high_comp[k]: .6f}")

        log("  low reward components (mean per interval):")
        for k in reward_keys_low:
            log(f"    {k:18s}: {low_comp[k] / float(n_low_intervals): .6f}")

        if arrived and arrival_time is not None:
            log(f"  ARRIVED at t = {float(arrival_time):.3f} s")
        if base_env.config.get("show_trajectories", False):
            save_speed_acc_curves(env, ep_idx=ep, model_path=eval_dir)
        if should_record_trajectory:
            csv_path = os.path.join(eval_dir, f"hiro_ep_{ep:04d}_trajectory.csv")
            if trajectory_rows:
                with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=list(trajectory_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(trajectory_rows)
                log(f"  saved trajectory csv    : {csv_path}")
            else:
                log(f"  saved trajectory csv    : skipped (episode {ep} has no trajectory rows)")

    n = int(episodes)
    log("=" * 80)
    log("Summary over all episodes:")
    log(f"  episodes                : {n}")
    log(f"  mean length             : {float(np.mean(ep_lens)):.3f} steps")
    log(f"  mean high total reward  : {float(np.mean(high_ep_rets)):.6f}")
    log(f"  mean low  ext (per interval mean)    : {float(np.mean(low_ep_ext_rets)):.6f}")
    log(f"  mean low  intrinsic (per interval)   : {float(np.mean(low_ep_int_rets)):.6f}")
    log(f"  mean low  total (per interval mean)  : {float(np.mean(low_ep_total_rets)):.6f}")

    log("  mean high reward components (per episode):")
    for k in reward_keys_high:
        log(f"    {k:18s}: {high_comp_sum[k] / n: .6f}")
    log("  mean low reward components (per interval):")
    for k in reward_keys_low:
        log(f"    {k:18s}: {low_comp_sum[k] / n: .6f}")

    arrive_rate = arrived_count / n if n else 0.0
    log(f"  arrival rate            : {arrive_rate * 100:.2f}%")
    if arrived_count:
        log(f"  mean arrival time       : {float(np.mean(arrival_times)):.3f} s (over {arrived_count} success episodes)")
    else:
        log("  mean arrival time       : N/A (no successful episodes)")
    log("=" * 80)

    log_file.close()
    env.close()


if __name__ == "__main__":
    # 用法示例：
    # - model_dir: 训练产物目录（默认从该目录读取 hiro_high_final.zip / hiro_low_final.zip）
    main(
        # model_dir="./models/hiro_260120_joint_safetyLayer_noOpc_rewShaping",
        model_dir="./models",
        high_model_dir="./models/hiro_260311_highonly_pretrained_newSLv2_lowDet_vioPenalty03_HER", 
        # high_model_dir="./models/hiro_test_260211_highonly_pretrained_vmin0", 
        low_model_dir="./models/hiro_260311_lowonly_uniform_RS_newSLv2_vioPenalty03_HER", 
        # low_model_dir="./models/hiro_260122_onlyLow_uniform_safetyLayer_rewShaping", 
        # model_suffix="step_6400000",
        use_low_safety_layer=True,
        episodes=10, 
        # record_episodes=[1, 2, 3],
        record_episodes=[i for i in range(1, 11)],
        # record_trajectory_episodes=[6],
        record_trajectory_episodes=[i for i in range(1, 11)],
    )
