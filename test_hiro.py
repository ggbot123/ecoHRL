import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import scenarios.multi_lane  # 触发 __init__.py 里的 register

import numpy as np
import os
import csv
import json
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
    enable_rendering: bool = True,
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

    high_interval_debug_csv_path = os.path.join(eval_dir, "high_interval_debug.csv")
    high_interval_debug_header = [
        "hi_start_seen",
        "hi_start_saved",
        "env_id",
        "step",
        "episode_env0",
        "segment_id",
        "c",
        "ego_sub",
        "high_obs",
        "kin",
        "goal_action",
        "goal_phys",
        "safe_l1",
        "safe_u1",
        "safe_l2",
        "safe_u2",
        "safe_dx_l2",
        "safe_dx_u2",
    ]
    with open(high_interval_debug_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        csv.writer(csv_file).writerow(high_interval_debug_header)

    def _json_arr(arr: Any) -> str:
        return json.dumps(np.asarray(arr, dtype=np.float32).tolist(), ensure_ascii=True)

    def _safe_norm_to_dx(norm_val: np.ndarray, dx_low: float, dx_high: float) -> np.ndarray:
        n = np.asarray(norm_val, dtype=np.float32)
        return ((n + 1.0) * 0.5 * float(dx_high - dx_low) + float(dx_low)).astype(np.float32)

    test_overrides: Dict[str, Any] = {
        # "initial_lane_id": 1,
        "initial_lane_id": "random",
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
        # "warmup_each_episode": False,
        "warmup_each_episode": True,
        "screen_width": 1800,
        "screen_height": 300,
        "scaling": 3,
        "centering_position": [0.5, 0.5],
        "show_trajectories": enable_rendering,
        "warmup_render": False,  
        "offscreen_rendering": enable_rendering,
        # Goal-distribution snapshot controls
        "goal_snapshot_use_focus_window": True,
        "goal_snapshot_front_distance": 100.0,
        "goal_snapshot_back_distance": 50.0,
        "goal_snapshot_show_history": True,
        "goal_snapshot_history_duration": 2.0,
        "goal_snapshot_history_frequency": 3.0,
        "goal_snapshot_goal_marker_size": 24.0,
        "goal_snapshot_show_prev_goal": False,
        "goal_snapshot_prev_goal_marker_size": 18.0,
        "goal_snapshot_fig_width": 15.0,
        "goal_snapshot_fig_height": 3.0,
    }
    if env_overrides:
        test_overrides.update(env_overrides)
    if not enable_rendering:
        test_overrides["show_trajectories"] = False
        test_overrides["warmup_render"] = False
        test_overrides["offscreen_rendering"] = False
    env_config = get_env_config(test_overrides)

    if not record_episodes:
        def trigger(ep_id: int) -> bool: return False
    else:
        record_set = {int(ep_idx) - 1 for ep_idx in record_episodes}
        def trigger(ep_id: int) -> bool: return ep_id in record_set

    trajectory_record_set = set()
    if record_trajectory_episodes:
        trajectory_record_set = {int(ep_idx) for ep_idx in record_trajectory_episodes}

    render_mode = "rgb_array" if enable_rendering else None
    base_env = gym.make("multi-lane-custom-v0", render_mode=render_mode, config=env_config)
    env = RecordVideo(base_env, video_folder=eval_dir, episode_trigger=trigger, name_prefix="hiro") if enable_rendering else base_env

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
    goal_lane_id = int(env_config.get("goal_lane_id", 2))
    punctual_time_window = env_config.get("punctual_time_window", [20.0, 30.0])
    t_min = float(punctual_time_window[0])
    t_max = float(punctual_time_window[1])

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
    log(f"Rendering enabled  : {enable_rendering}")
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

    def classify_failure(crashed: bool, arrived: bool, arrival_time: Optional[float], final_lane_id: Optional[int]) -> Tuple[bool, bool, bool, bool, bool]:
        if crashed:
            return True, True, False, False, False
        on_time_arrival = bool(arrived and arrival_time is not None and t_min <= float(arrival_time) <= t_max)
        failed = not on_time_arrival
        wrong_lane = bool(failed and final_lane_id is not None and int(final_lane_id) != goal_lane_id)
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
    hi_start_seen = 0
    hi_start_saved = 0
    high_segment_id = 0
    total_env_steps = 0
    policy_frequency = float(env_config.get("policy_frequency", 1.0))
    warmup_time = float(env_config.get("warmup_time", 0.0))
    warmup_each_episode = bool(env_config.get("warmup_each_episode", False))
    initial_vid = int(env.unwrapped.config.get("vid", 0))

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
        
        if enable_rendering and not viewer_initialized:
            class Dummy:
                def __init__(self, pos): self.position = np.array(pos, dtype=float)
            base = env.unwrapped
            base.render()
            base.viewer.observer_vehicle = Dummy([base.config["road_length"] / 2, 5.0])
            viewer_initialized = True

        while not (terminated or truncated):
            is_hi_start = bool(runner.need_high)
            hi_start_high_obs = None
            hi_start_kin = None
            hi_start_ego_sub = np.asarray([], dtype=np.float32)
            if is_hi_start:
                hi_start_high_obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
                try:
                    _, hi_start_kin, _ = runner._split(obs)
                    hi_start_ego_sub = runner._ego_sub(hi_start_kin).astype(np.float32)
                except Exception:
                    hi_start_kin = None

            # Capture prev goal before runner.act updates it (if need_high is True)
            if runner.need_high:
                 # Check if we have a valid current goal to save as "previous"
                 if len(runner.goal_phys) > 0 and not (runner.c == 0 and steps == 0):
                      prev_goal_phys = runner.goal_phys.copy()
            
            action = runner.act(env, obs)

            if is_hi_start and hi_start_high_obs is not None:
                hi_start_seen += 1
                hi_start_saved += 1

                safe_l1 = np.asarray([], dtype=np.float32)
                safe_u1 = np.asarray([], dtype=np.float32)
                safe_l2 = np.asarray([], dtype=np.float32)
                safe_u2 = np.asarray([], dtype=np.float32)
                safe_dx_l2 = np.asarray([], dtype=np.float32)
                safe_dx_u2 = np.asarray([], dtype=np.float32)
                bounds_calc = getattr(runner, "high_goal_safe_bounds", None)
                if bounds_calc is not None:
                    try:
                        safe_bounds = bounds_calc.compute_np(hi_start_high_obs)
                        safe_l1 = np.asarray(safe_bounds.get("l1", []), dtype=np.float32)
                        safe_u1 = np.asarray(safe_bounds.get("u1", []), dtype=np.float32)
                        safe_l2 = np.asarray(safe_bounds.get("l2", []), dtype=np.float32)
                        safe_u2 = np.asarray(safe_bounds.get("u2", []), dtype=np.float32)
                        if safe_l2.size and safe_u2.size:
                            safe_dx_l2 = _safe_norm_to_dx(safe_l2, float(bounds_calc.dx_low), float(bounds_calc.dx_high))
                            safe_dx_u2 = _safe_norm_to_dx(safe_u2, float(bounds_calc.dx_low), float(bounds_calc.dx_high))
                            empty_mask = safe_l2 > safe_u2
                            safe_dx_l2 = np.where(empty_mask, np.nan, safe_dx_l2)
                            safe_dx_u2 = np.where(empty_mask, np.nan, safe_dx_u2)
                    except Exception:
                        safe_l1 = np.asarray([], dtype=np.float32)
                        safe_u1 = np.asarray([], dtype=np.float32)
                        safe_l2 = np.asarray([], dtype=np.float32)
                        safe_u2 = np.asarray([], dtype=np.float32)
                        safe_dx_l2 = np.asarray([], dtype=np.float32)
                        safe_dx_u2 = np.asarray([], dtype=np.float32)

                goal_action_log = np.asarray(getattr(runner, "last_goal_action", []), dtype=np.float32)
                goal_phys_log = np.asarray(getattr(runner, "goal_phys", []), dtype=np.float32)
                kin_log = np.asarray([], dtype=np.float32)
                if hi_start_kin is not None:
                    kin_log = np.asarray(hi_start_kin[0], dtype=np.float32)

                debug_row = [
                    int(hi_start_seen),
                    int(hi_start_saved),
                    0,
                    int(total_env_steps),
                    int(ep - 1),
                    int(high_segment_id),
                    int(runner.c),
                    _json_arr(hi_start_ego_sub),
                    _json_arr(hi_start_high_obs[0]),
                    _json_arr(kin_log),
                    _json_arr(goal_action_log),
                    _json_arr(goal_phys_log),
                    _json_arr(safe_l1),
                    _json_arr(safe_u1),
                    _json_arr(safe_l2),
                    _json_arr(safe_u2),
                    _json_arr(safe_dx_l2),
                    _json_arr(safe_dx_u2),
                ]
                with open(high_interval_debug_csv_path, "a", newline="", encoding="utf-8") as csv_file:
                    csv.writer(csv_file).writerow(debug_row)
                high_segment_id += 1

            # Snapshot goal at the beginning of each interval (or every few intervals)
            # runner.c is 0 immediately after sampling a new goal.
            if enable_rendering and runner.c == 0:
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
            total_env_steps += 1

            if last_step:
                high_interval_rets.append(float(cur_high_interval_ret))
                low_interval_rets.append(float(cur_low_interval_ret))
                cur_high_interval_ret, cur_low_interval_ret = 0.0, 0.0
            runner.step_end(done)
            obs = obs_next

        if enable_rendering:
            # Save the terminal frame snapshot for each episode.
            save_goal_snapshot(
                env,
                runner,
                ep,
                steps,
                eval_dir,
                prev_goal_phys=prev_goal_phys,
                intrinsic_reward=last_intrinsic_viz,
            )

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
        failed, failed_collision, failed_wrong_lane, failed_late, failed_early = classify_failure(crashed, arrived, arrival_time, final_lane_id)
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

    final_vid = int(env.unwrapped.config.get("vid", initial_vid))
    generated_vehicle_count = max(final_vid - initial_vid, 0)
    warmup_runs = int(episodes) if warmup_each_episode else (1 if warmup_time > 0.0 else 0)
    total_warmup_time = warmup_time * float(warmup_runs)
    total_episode_time = float(total_env_steps) / max(policy_frequency, 1e-6)
    total_sim_time = total_episode_time + total_warmup_time
    traffic_flow_veh_per_s = (
        float(generated_vehicle_count) / total_sim_time if total_sim_time > 0.0 else 0.0
    )

    log("  traffic flow stats      :")
    log(f"    generated vehicles    : {generated_vehicle_count}")
    log(f"    total sim time        : {total_sim_time:.3f} s (episode={total_episode_time:.3f} s, warmup={total_warmup_time:.3f} s)")
    log(f"    flow                  : {traffic_flow_veh_per_s:.6f} veh/s ({traffic_flow_veh_per_s * 3600.0:.3f} veh/h)")
    log("=" * 80)

    log_file.close()
    env.close()


if __name__ == "__main__":
    # 用法示例：
    # - model_dir: 训练产物目录（默认从该目录读取 hiro_high_final.zip / hiro_low_final.zip）
    main(
        # model_dir="./models/hiro_260120_joint_safetyLayer_noOpc_rewShaping",
        model_dir="./models",
        # high_model_dir="./models/hiro_260401_highonly_UniformLane1_Rainbow_randomLane",
        # high_model_dir="./models/hiro_260329_highonly_reachablePretrainedV2_Rainbow_amax3_dmin15_10",
        high_model_dir="./models/hiro_260331_highonly_reachableUniformLane1_Rainbow_amax3_dmin15_10_randomlane",
        # high_model_dir="./models/hiro_260319_highonly_pretrained_newSLv2_vio03_HER_reDim_lc10",
        # high_model_dir="./models/hiro_test_260211_highonly_pretrained_vmin0",

        # low_model_dir="./models/hiro_260325_lowonly_reachableUniformv2_Rainbow_dmin10_8",
        low_model_dir="./models/hiro_260328_lowonly_reachablePretrainedV2_Rainbow_amax3_dmin15_10",
        # low_model_dir="./models/hiro_260318_lowonly_uniform_RS_newSLv2_vio03_HER_reDim_v2",
        # low_model_dir="./models/hiro_260122_onlyLow_uniform_safetyLayer_rewShaping",
        # model_suffix="step_6400000",
        use_low_safety_layer=True,
        episodes=300, 
        # record_episodes=[],
        # record_trajectory_episodes=[],
        record_episodes=[i for i in range(1, 11)],
        record_trajectory_episodes=[i for i in range(1, 11)],
        # enable_rendering=False,
    )
