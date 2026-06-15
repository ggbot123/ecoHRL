import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import importlib

import numpy as np
import os
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple

from util.plot_result import *
from util.hiro_utils import load_hiro_high_model, unique_path

from rl.utils import utils
from custom_env import utils as c_utils
from rl.algos.HRL.hiro_infer import HIROPolicyRunner
from rl.algos.HRL.rule_based import RuleBasedController
from configs.conf import get_env_config_for_scenario, get_hiro_config, get_scenario_spec
from custom_env.vehicle.controller import ControlledVehicle
from custom_env.vehicle.behavior import NormalIDMVehicle
from custom_env.road.road import LaneIndex



def main(
    model_dir: str,
    episodes: int,
    record_episodes: Optional[Sequence[int]] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
    high_model_dir: Optional[str] = None,
    enable_rendering: bool = True,
    scenario_name: str = "multi_lane",
):
    eval_root_dir = os.path.join(model_dir, "eval_results")
    os.makedirs(eval_root_dir, exist_ok=True)
    run_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = unique_path(os.path.join(eval_root_dir, run_folder_name))
    os.makedirs(eval_dir, exist_ok=True)

    log_path = os.path.join(eval_dir, "eval_rule.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    def log(msg: str = ""):
        print(msg)
        log_file.write(msg + "\n")

    test_overrides: Dict[str, Any] = {
        # "initial_lane_id": 0,
        # "initial_lane_id": 1,
        # "initial_lane_id": 2,
        "initial_lane_id": "random",
        "duration": 70.0,
        # "warmup_each_episode": False,
        "warmup_each_episode": True,
        "screen_width": 1800,
        "screen_height": 300,
        "scaling": 3,
        "centering_position": [0.5, 0.5],
        "show_trajectories": enable_rendering,
        "warmup_render": False,
        "offscreen_rendering": enable_rendering,
        "action": {
            "type": "ParamLaneAccelAction",
            "acceleration_range": [-5.0, 5.0],
            "lane_actions": ["KEEP", "LANE_LEFT", "LANE_RIGHT"],
        }
    }
    if env_overrides:
        test_overrides.update(env_overrides)
    scenario_spec = get_scenario_spec(scenario_name)
    importlib.import_module(str(scenario_spec["module"]))
    env_id = str(scenario_spec["env_id"])
    if not enable_rendering:
        test_overrides["show_trajectories"] = False
        test_overrides["warmup_render"] = False
        test_overrides["offscreen_rendering"] = False
    env_config = get_env_config_for_scenario(scenario_name, test_overrides)

    if not record_episodes:
        def trigger(ep_id: int) -> bool: return False
    else:
        record_set = {int(ep_idx) - 1 for ep_idx in record_episodes}
        def trigger(ep_id: int) -> bool: return ep_id in record_set

    # 1. Create Environment
    render_mode = "rgb_array" if enable_rendering else None
    base_env = gym.make(env_id, render_mode=render_mode, config=env_config)
    
    env = RecordVideo(base_env, video_folder=eval_dir, episode_trigger=trigger, name_prefix="rule") if enable_rendering else base_env

    # 2. Load HIRO High Model (for High-Level Goal Sampling)
    high_model = load_hiro_high_model(high_model_dir or model_dir)
    hiro_cfg = get_hiro_config()
    high_interval = int(getattr(hiro_cfg, "high_interval", 25))
    runner = HIROPolicyRunner(high_model, low_model=None, high_interval=high_interval)

    # 3. Initialize Rule Based Controller
    controller = RuleBasedController(
        env_config,
        low_safety_filter=getattr(hiro_cfg, "low_safety_filter", None),
    )
    dt = 1.0 / float(env_config["policy_frequency"])
    kin_meta = None

    reward_keys_high = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "punctual_reward", "on_road_reward"]
    # Controller doesn't have intrinsic reward in environment return, but we can compute it
    reward_keys_low = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "on_road_reward", "intrinsic_reward"]
    punctual_time_window = env_config.get("punctual_time_window", [20.0, 30.0])
    t_min = float(punctual_time_window[0])
    t_max = float(punctual_time_window[1])

    log("=" * 80)
    log(f"Eval Rule-based model dir: {model_dir}")
    log(f"Eval run folder      : {run_folder_name}")
    log(f"Eval results dir     : {eval_dir}")
    log(f"Episodes             : {episodes}")
    hd = high_model_dir or model_dir
    hp = os.path.join(hd, "hiro_high_final.zip")
    log(f"HIRO high            : {hp}")
    log("HIRO low             : N/A (rule-based low controller)")
    log(f"Low safety layer     : {bool(getattr(hiro_cfg, 'low_safety_filter', None))}")
    log(f"High interval        : {runner.hi}")
    log(f"Rendering enabled    : {enable_rendering}")
    log("=" * 80)

    ep_lens: list[int] = []
    high_ep_rets: list[float] = []
    low_ep_ext_rets: list[float] = []
    low_ep_int_rets: list[float] = []
    low_ep_total_rets: list[float] = []
    high_comp_sum = {k: 0.0 for k in reward_keys_high}
    low_comp_sum = {k: 0.0 for k in reward_keys_low}
    exclude_collision_mean_keys = {"comfort_reward", "lane_change_reward"}
    high_comp_sum_no_collision = {k: 0.0 for k in exclude_collision_mean_keys}
    low_comp_sum_no_collision = {k: 0.0 for k in exclude_collision_mean_keys}
    non_collision_episode_count = 0

    lane_group_stats: Dict[int, Dict[str, Any]] = {}

    def get_terminal_lane_id(base_env: Any) -> Optional[int]:
        ego_vehicle = getattr(base_env, "vehicle", None)
        if ego_vehicle is not None:
            lane_index = getattr(ego_vehicle, "lane_index", None)
            if lane_index is not None and len(lane_index) >= 3:
                try:
                    return int(lane_index[2])
                except (TypeError, ValueError):
                    pass
            if hasattr(ego_vehicle, "position"):
                lane_w = float(base_env.config.get("lane_width", 4.0))
                lanes_n = int(base_env.config.get("lanes_count", 3))
                return int(
                    np.clip(
                        int(round(float(ego_vehicle.position[1]) / max(lane_w, 1e-6))),
                        0,
                        lanes_n - 1,
                    )
                )
        return None

    def classify_failure(crashed: bool, arrived: bool, arrival_time: Optional[float], final_lane_id: Optional[int], goal_lane_id: Optional[int]) -> Tuple[bool, bool, bool, bool, bool]:
        if crashed:
            return True, True, False, False, False
        on_time_arrival = bool(arrived and arrival_time is not None and t_min <= float(arrival_time) <= t_max)
        failed = not on_time_arrival
        wrong_lane = bool(
            failed
            and final_lane_id is not None
            and goal_lane_id is not None
            and int(final_lane_id) != int(goal_lane_id)
        )
        late = bool(failed and arrived and arrival_time is not None and float(arrival_time) > t_max)
        early = bool(failed and arrived and arrival_time is not None and float(arrival_time) < t_min)
        return failed, False, wrong_lane, late, early

    def log_failed_breakdown(prefix: str, failed_count: int, collision_count: int, wrong_lane_count: int, late_count: int, early_count: int) -> None:
        other_count = int(failed_count) - int(collision_count) - int(wrong_lane_count) - int(late_count) - int(early_count)
        other_count = max(other_count, 0)
        if failed_count <= 0:
            log(f"{prefix}failed episodes       : 0")
            log(f"{prefix}collision            : 0")
            log(f"{prefix}wrong-lane at end    : 0")
            log(f"{prefix}late arrival         : 0")
            log(f"{prefix}early arrival        : 0")
            return
        log(f"{prefix}failed episodes       : {failed_count}")
        log(f"{prefix}collision            : {collision_count} ({collision_count / failed_count * 100:.2f}% of failed)")
        log(f"{prefix}wrong-lane at end    : {wrong_lane_count} ({wrong_lane_count / failed_count * 100:.2f}% of failed)")
        log(f"{prefix}late arrival         : {late_count} ({late_count / failed_count * 100:.2f}% of failed)")
        log(f"{prefix}early arrival        : {early_count} ({early_count / failed_count * 100:.2f}% of failed)")
        if other_count > 0:
            log(f"{prefix}other failures       : {other_count} ({other_count / failed_count * 100:.2f}% of failed)")

    def format_component_mean(
        key: str,
        total_sum: float,
        total_count: int,
        no_collision_sum: float,
        no_collision_count: int,
    ) -> str:
        if key in exclude_collision_mean_keys:
            if no_collision_count > 0:
                return f"{no_collision_sum / no_collision_count: .6f}"
            return "N/A (all episodes collided)"
        return f"{total_sum / total_count: .6f}"

    def ensure_lane_group(lane_id: int) -> Dict[str, Any]:
        if lane_id not in lane_group_stats:
            lane_group_stats[lane_id] = {
                "episodes": 0,
                "ep_lens": [],
                "high_ep_rets": [],
                "low_ep_ext_rets": [],
                "low_ep_int_rets": [],
                "low_ep_total_rets": [],
                "high_comp_sum": {k: 0.0 for k in reward_keys_high},
                "low_comp_sum": {k: 0.0 for k in reward_keys_low},
                "high_comp_sum_no_collision": {k: 0.0 for k in exclude_collision_mean_keys},
                "low_comp_sum_no_collision": {k: 0.0 for k in exclude_collision_mean_keys},
                "non_collision_episode_count": 0,
                "arrived_count": 0,
                "arrival_times": [],
                "failed_count": 0,
                "failed_collision_count": 0,
                "failed_wrong_lane_count": 0,
                "failed_late_count": 0,
                "failed_early_count": 0,
            }
        return lane_group_stats[lane_id]

    arrived_count, arrival_times = 0, []
    failed_count = 0
    failed_collision_count = 0
    failed_wrong_lane_count = 0
    failed_late_count = 0
    failed_early_count = 0
    viewer_initialized = False
    seed_base = 42

    for ep in range(1, int(episodes) + 1):
        obs, _ = env.reset(seed=seed_base + ep)
        reset_base_env = env.unwrapped
        init_lane = None
        ego_vehicle = getattr(reset_base_env, "vehicle", None)
        if ego_vehicle is not None:
            lane_index = getattr(ego_vehicle, "lane_index", None)
            if lane_index is not None and len(lane_index) >= 3:
                init_lane = int(lane_index[2])
            elif hasattr(ego_vehicle, "position"):
                lane_w = float(reset_base_env.config.get("lane_width", 4.0))
                lanes_n = int(reset_base_env.config.get("lanes_count", 3))
                init_lane = int(np.clip(int(round(float(ego_vehicle.position[1]) / max(lane_w, 1e-6))), 0, lanes_n - 1))
        if init_lane is None:
            init_lane = -1

        if kin_meta is None:
            keep = ("x", "y", "vx", "vy")
            n_veh, n_veh_local, feat_dim, feature_names, ego_feature_idx = utils.init_kinematics_meta(env, obs, keep)

            def _idx(name: str, default: int) -> int:
                try:
                    return int(feature_names.index(name))
                except ValueError:
                    return int(default)

            idx_presence = _idx("presence", -1)
            idx_x = _idx("x", 0)
            idx_y = _idx("y", 1)
            idx_vx = _idx("vx", 2)
            idx_vy = _idx("vy", 3)
            kin_meta = {
                "n_veh": int(n_veh),
                "n_veh_local": int(n_veh_local),
                "feat_dim": int(feat_dim),
                "ego_feature_idx": list(ego_feature_idx),
                "idx_presence": int(idx_presence),
                "idx_x": int(idx_x),
                "idx_y": int(idx_y),
                "idx_vx": int(idx_vx),
                "idx_vy": int(idx_vy),
            }
        runner.reset(env, obs, float(getattr(hiro_cfg, "intrinsic_coef", 1.0)))

        terminated, truncated, steps = False, False, 0
        high_ret, low_ext_ret, low_int_ret, low_total_ret = 0.0, 0.0, 0.0, 0.0
        high_comp = {k: 0.0 for k in reward_keys_high}
        low_comp = {k: 0.0 for k in reward_keys_low}
        high_interval_rets, low_interval_rets = [], []
        cur_high_interval_ret, cur_low_interval_ret = 0.0, 0.0
        
        last_intrinsic_viz = None
        prev_goal_phys = None
        
        if enable_rendering and not viewer_initialized:
            class Dummy:
                def __init__(self, pos): self.position = np.array(pos, dtype=float)
            base = env.unwrapped
            base.render()
            base.viewer.observer_vehicle = Dummy([base.config["road_length"] / 2, 5.0])
            viewer_initialized = True

        while not (terminated or truncated):
            if ep == 5:
                pass

            # Capture prev goal logic
            if runner.need_high:
                 if len(runner.goal_phys) > 0 and not (runner.c == 0 and steps == 0):
                      prev_goal_phys = runner.goal_phys.copy()
            
            # 1. Update Goal via Runner (side-effect: updates runner.goal_phys)
            _ = runner.act(env, obs)
            
            # 2. Get Goal
            goal_phys = runner.goal_phys
            
            # 3. Compute control action exactly as RuleBasedAgentWrapper in training:
            #    raw action -> safety-filtered action
            t_norm = float(runner.c) / float(high_interval)
            rem_time = float(high_interval) * (1.0 - t_norm) * float(dt)

            obs_arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            n_veh = int(kin_meta["n_veh"])
            n_veh_local = int(kin_meta["n_veh_local"])
            feat_dim = int(kin_meta["feat_dim"])
            kin_slice = obs_arr[:, : 1 + n_veh * feat_dim]
            _, kin_full, kin_flat_full = utils.split_time_kinematics(kin_slice, n_veh, feat_dim)

            ego_feature_idx = kin_meta["ego_feature_idx"]
            ego_sub = utils.extract_ego_substate(kin_full, ego_feature_idx)[0]
            goal_rel = (np.asarray(goal_phys, dtype=np.float32) - ego_sub).astype(np.float32)
            local_kin_flat = np.asarray(kin_flat_full[0, : n_veh_local * feat_dim], dtype=np.float32)
            # Keep low_obs format aligned with training/inference: [t_norm, local_kin_flat, goal_rel].
            low_obs = np.concatenate(
                [np.array([t_norm], dtype=np.float32), local_kin_flat, goal_rel],
                axis=0,
            ).astype(np.float32)

            kin_local_slice = low_obs.reshape(1, -1)[:, : 1 + n_veh_local * feat_dim]
            _, kin_local, _ = utils.split_time_kinematics(kin_local_slice, n_veh_local, feat_dim)

            idx_presence = int(kin_meta["idx_presence"])
            idx_x = int(kin_meta["idx_x"])
            idx_y = int(kin_meta["idx_y"])
            idx_vx = int(kin_meta["idx_vx"])
            idx_vy = int(kin_meta["idx_vy"])

            ego_feat = kin_local[0, 0]
            ego_abs = np.array(
                [
                    ego_feat[idx_x],
                    ego_feat[idx_y],
                    ego_feat[idx_vx],
                    ego_feat[idx_vy],
                ],
                dtype=np.float32,
            )

            others_feat = kin_local[0, 1:]
            others_rel = []
            for j in range(int(others_feat.shape[0])):
                d = others_feat[j]
                if idx_presence >= 0 and d[idx_presence] == 0:
                    continue
                others_rel.append([float(d[idx_x]), float(d[idx_y]), float(d[idx_vx]), float(d[idx_vy])])
            others_rel_arr = np.asarray(others_rel, dtype=np.float32).reshape(-1, 4)

            action_raw = controller.compute_action(
                ego_abs,
                others_rel_arr,
                goal_phys,
                dt,
                remaining_time=rem_time,
            )
            action = controller.safety_filter_action(
                ego_abs=ego_abs,
                others_rel=others_rel_arr,
                goal_phys=goal_phys,
                action=action_raw,
                dt=dt,
                remaining_time=rem_time,
            )

            # Snapshot
            if enable_rendering and runner.c == 0:
                save_goal_snapshot(env, runner, ep, steps, eval_dir, prev_goal_phys=prev_goal_phys, intrinsic_reward=last_intrinsic_viz)

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
            runner.step_end(
                done,
                queue_takeover_active=bool(info.get("queue_takeover_active", False)),
            )
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
        crashed = bool(getattr(getattr(env.unwrapped, "vehicle", None), "crashed", False))
        for k in reward_keys_high:
            high_comp_sum[k] += high_comp[k]
        for k in reward_keys_low:
            low_comp_sum[k] += low_comp[k] / float(n_low_intervals)
        if not crashed:
            non_collision_episode_count += 1
            for k in exclude_collision_mean_keys:
                high_comp_sum_no_collision[k] += high_comp[k]
                low_comp_sum_no_collision[k] += low_comp[k] / float(n_low_intervals)

        base_env = env.unwrapped
        arrived = bool(getattr(base_env, "_has_arrived", False))
        arrival_time = getattr(base_env, "_arrival_time", None)
        final_lane_id = get_terminal_lane_id(base_env)
        goal_lane_id = int(base_env.get_goal_lane_id())
        failed, failed_collision, failed_wrong_lane, failed_late, failed_early = classify_failure(
            crashed,
            arrived,
            arrival_time,
            final_lane_id,
            goal_lane_id,
        )
        if arrived:
            arrived_count += 1
            if arrival_time is not None:
                arrival_times.append(float(arrival_time))
        if failed:
            failed_count += 1
        if failed_collision:
            failed_collision_count += 1
        if failed_wrong_lane:
            failed_wrong_lane_count += 1
        if failed_late:
            failed_late_count += 1
        if failed_early:
            failed_early_count += 1

        reason = "terminated" if terminated else ("truncated(time limit)" if truncated else "unknown")
        log("=" * 60)
        log(f"Episode {ep}:")
        log(f"  initial lane            : {init_lane}")
        log(f"  terminal lane           : {final_lane_id if final_lane_id is not None else 'N/A'}")
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
        if failed:
            log(
                "  failed flags            : "
                f"collision={int(failed_collision)}, wrong_lane={int(failed_wrong_lane)}, late={int(failed_late)}, early={int(failed_early)}"
            )

        if enable_rendering and base_env.config.get("show_trajectories", False):
            save_speed_acc_curves(env, ep_idx=ep, model_path=eval_dir)

        group = ensure_lane_group(int(init_lane))
        group["episodes"] += 1
        group["ep_lens"].append(int(steps))
        group["high_ep_rets"].append(float(high_ret))
        group["low_ep_ext_rets"].append(float(low_ext_mean))
        group["low_ep_int_rets"].append(float(low_int_mean))
        group["low_ep_total_rets"].append(float(low_total_mean))
        for k in reward_keys_high:
            group["high_comp_sum"][k] += high_comp[k]
        for k in reward_keys_low:
            group["low_comp_sum"][k] += low_comp[k] / float(n_low_intervals)
        if not crashed:
            group["non_collision_episode_count"] += 1
            for k in exclude_collision_mean_keys:
                group["high_comp_sum_no_collision"][k] += high_comp[k]
                group["low_comp_sum_no_collision"][k] += low_comp[k] / float(n_low_intervals)
        if arrived:
            group["arrived_count"] += 1
            if arrival_time is not None:
                group["arrival_times"].append(float(arrival_time))
        if failed:
            group["failed_count"] += 1
        if failed_collision:
            group["failed_collision_count"] += 1
        if failed_wrong_lane:
            group["failed_wrong_lane_count"] += 1
        if failed_late:
            group["failed_late_count"] += 1
        if failed_early:
            group["failed_early_count"] += 1

    n = int(episodes)
    log("=" * 80)
    log("Summary by initial lane:")
    lanes_for_summary = int(env_config.get("lanes_count", 3))
    for lane_id in range(lanes_for_summary):
        group = lane_group_stats.get(lane_id)
        if group is None or int(group["episodes"]) == 0:
            log(f"  lane {lane_id}: no episodes")
            continue

        n_lane = int(group["episodes"])
        log("-" * 80)
        log(f"  lane {lane_id}:")
        log(f"    episodes              : {n_lane}")
        log(f"    mean length           : {float(np.mean(group['ep_lens'])):.3f} steps")
        log(f"    mean high total reward: {float(np.mean(group['high_ep_rets'])):.6f}")
        log(f"    mean low ext          : {float(np.mean(group['low_ep_ext_rets'])):.6f}")
        log(f"    mean low intrinsic    : {float(np.mean(group['low_ep_int_rets'])):.6f}")
        log(f"    mean low total        : {float(np.mean(group['low_ep_total_rets'])):.6f}")
        log("    mean high reward components (per episode):")
        for k in reward_keys_high:
            log(
                f"      {k:16s}: "
                f"{format_component_mean(k, group['high_comp_sum'][k], n_lane, group['high_comp_sum_no_collision'].get(k, 0.0), int(group['non_collision_episode_count']))}"
            )
        log("    mean low reward components (per interval):")
        for k in reward_keys_low:
            log(
                f"      {k:16s}: "
                f"{format_component_mean(k, group['low_comp_sum'][k], n_lane, group['low_comp_sum_no_collision'].get(k, 0.0), int(group['non_collision_episode_count']))}"
            )

        lane_arrive_rate = group["arrived_count"] / n_lane if n_lane else 0.0
        log(f"    arrival rate          : {lane_arrive_rate * 100:.2f}%")
        if group["arrived_count"] > 0:
            log(
                f"    mean arrival time     : {float(np.mean(group['arrival_times'])):.3f} s "
                f"(over {int(group['arrived_count'])} success episodes)"
            )
        else:
            log("    mean arrival time     : N/A (no successful episodes)")
        log_failed_breakdown(
            "    ",
            int(group["failed_count"]),
            int(group["failed_collision_count"]),
            int(group["failed_wrong_lane_count"]),
            int(group["failed_late_count"]),
            int(group["failed_early_count"]),
        )

    log("=" * 80)
    log("Overall summary:")
    log("Summary over all episodes:")
    log(f"  episodes                : {n}")
    log(f"  mean length             : {float(np.mean(ep_lens)):.3f} steps")
    log(f"  mean high total reward  : {float(np.mean(high_ep_rets)):.6f}")
    log(f"  mean low  ext (per interval mean)    : {float(np.mean(low_ep_ext_rets)):.6f}")
    log(f"  mean low  intrinsic (per interval)   : {float(np.mean(low_ep_int_rets)):.6f}")
    log(f"  mean low  total (per interval mean)  : {float(np.mean(low_ep_total_rets)):.6f}")

    log("  mean high reward components (per episode):")
    for k in reward_keys_high:
        log(
            f"    {k:18s}: "
            f"{format_component_mean(k, high_comp_sum[k], n, high_comp_sum_no_collision.get(k, 0.0), non_collision_episode_count)}"
        )
    log("  mean low reward components (per interval):")
    for k in reward_keys_low:
        log(
            f"    {k:18s}: "
            f"{format_component_mean(k, low_comp_sum[k], n, low_comp_sum_no_collision.get(k, 0.0), non_collision_episode_count)}"
        )

    arrive_rate = arrived_count / n if n else 0.0
    log(f"  arrival rate            : {arrive_rate * 100:.2f}%")
    if arrived_count:
        log(f"  mean arrival time       : {float(np.mean(arrival_times)):.3f} s (over {arrived_count} success episodes)")
    else:
        log("  mean arrival time       : N/A (no successful episodes)")
    log_failed_breakdown(
        "  ",
        failed_count,
        failed_collision_count,
        failed_wrong_lane_count,
        failed_late_count,
        failed_early_count,
    )
    log("=" * 80)
    log_file.close()
    env.close()

if __name__ == "__main__":
    main(
        # model_dir="./models/hiro_260330_highonly_rule_withSL", 
        model_dir="./models/hiro_260331_highonly_rule_accwithSL_randomLane", 
        episodes=300,
        record_episodes=[i for i in range(1, 301)],
        # enable_rendering=False,
    )
