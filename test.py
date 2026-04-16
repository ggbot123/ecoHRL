import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import os
import importlib
import csv
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple
from util.plot_result import *
from configs.conf import get_env_config_for_scenario, get_hiro_config, get_scenario_spec

from rl.algos.ppo.ppo import PPO
from rl.algos.sac.sac import SAC


def load_model(algo: str, model_path: str, env):
    algo = algo.lower()
    if algo == "ppo":
        model = PPO.load(model_path, env=env)
    elif algo == "sac":
        model = SAC.load(model_path, env=env)
    else:
        raise ValueError(f"未知算法类型: {algo}")
    return model


def _unique_path(base_path: str) -> str:
    if not os.path.exists(base_path):
        return base_path
    idx = 1
    while True:
        candidate = f"{base_path}_{idx:02d}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def main(
    model_path: str,
    model_name: str,
    algo: str,
    episodes: int,
    record_episodes: Optional[Sequence[int]] = None,
    record_trajectory_episodes: Optional[Sequence[int]] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
    enable_rendering: bool = True,
    scenario_name: str = "multi_lane",
):
    algo = str(algo).lower()

    eval_root_dir = os.path.join(model_path, "eval_results")
    os.makedirs(eval_root_dir, exist_ok=True)
    run_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = _unique_path(os.path.join(eval_root_dir, run_folder_name))
    os.makedirs(eval_dir, exist_ok=True)

    log_path = os.path.join(eval_dir, f"eval_{algo}.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg: str = ""):
        print(msg)
        log_file.write(msg + "\n")

    test_overrides: Dict[str, Any] = {
        "initial_lane_id": "random",
        # "PERCEPTION_DISTANCE": 200,
        # "observation": {
        #     "type": "Kinematics",
        #     "vehicles_count": 20,
        #     "vehicles_count_local": 5,
        #     "features": ["presence", "x", "y", "vx", "vy"],
        #     "features_range": {
        #         "x": [-200, 200],
        #         "y": [-10, 10],
        #         "vx": [-15, 15],
        #         "vy": [-10, 10]
        #     },
        #     "normalize": False,
        #     "see_behind": False,
        #     "include_obstacles": False,
        #     "include_time": True,
        #     "time_range": [0.0, 50.0],
        # },
        "duration": 70.0,
        "warmup_each_episode": True,
        "screen_width": 1800,
        "screen_height": 300,
        "scaling": 3,
        "centering_position": [0.5, 0.5],
        "show_trajectories": enable_rendering,
        "warmup_render": False,
        "offscreen_rendering": enable_rendering,
    }
    scenario_spec = get_scenario_spec(scenario_name)
    importlib.import_module(str(scenario_spec["module"]))
    env_id = str(scenario_spec["env_id"])

    if env_overrides:
        test_overrides.update(env_overrides)
    if not enable_rendering:
        test_overrides["show_trajectories"] = False
        test_overrides["warmup_render"] = False
        test_overrides["offscreen_rendering"] = False
    env_config = get_env_config_for_scenario(scenario_name, test_overrides)

    # Keep SAC evaluation safety behavior consistent with training.
    if algo == "sac" and bool(env_config.get("enable_sac_low_safety_filter", False)):
        hiro_cfg_for_sac = get_hiro_config()
        if getattr(hiro_cfg_for_sac, "low_safety_filter", None) is not None:
            test_overrides.update(
                {
                    "enable_low_safety_filter": True,
                    "lane_change_min_front_gap": float(hiro_cfg_for_sac.low_safety_filter.lane_change_min_front_gap),
                    "lane_change_min_rear_gap": float(hiro_cfg_for_sac.low_safety_filter.lane_change_min_rear_gap),
                    "lane_change_min_front_ttc": float(hiro_cfg_for_sac.low_safety_filter.lane_change_min_front_ttc),
                    "lane_change_min_rear_ttc": float(hiro_cfg_for_sac.low_safety_filter.lane_change_min_rear_ttc),
                }
            )
            env_config = get_env_config_for_scenario(scenario_name, test_overrides)

    # 视频录制触发器
    if record_episodes is None or len(record_episodes) == 0:
        def trigger(ep_id: int) -> bool:
            return False
    else:
        record_set = {int(ep_idx) - 1 for ep_idx in record_episodes}

        def trigger(ep_id: int) -> bool:
            return ep_id in record_set

    trajectory_record_set = set()
    if record_trajectory_episodes:
        trajectory_record_set = {int(ep_idx) for ep_idx in record_trajectory_episodes}

    render_mode = "rgb_array" if enable_rendering else None
    base_env = gym.make(env_id, render_mode=render_mode, config=env_config)
    env = RecordVideo(base_env, video_folder=eval_dir, episode_trigger=trigger, name_prefix=f"{algo}") if enable_rendering else base_env

    # 加载模型
    model = load_model(algo, os.path.join(model_path, model_name), env)

    # reward key 列表，保持与 MultiLaneEnv._rewards 一致
    reward_keys = [
        "collision_reward",
        "progress_reward",
        "speed_ref_aux_reward",
        "comfort_reward",
        "lane_change_reward",
        "punctual_reward",
        "on_road_reward",
    ]
    goal_lane_id = int(env_config.get("goal_lane_id", 2))
    punctual_time_window = env_config.get("punctual_time_window", [20.0, 30.0])
    t_min = float(punctual_time_window[0])
    t_max = float(punctual_time_window[1])

    exclude_collision_mean_keys = {"comfort_reward", "lane_change_reward"}

    def get_terminal_lane_id(base: Any) -> Optional[int]:
        ego_vehicle = getattr(base, "vehicle", None)
        if ego_vehicle is not None:
            lane_index = getattr(ego_vehicle, "lane_index", None)
            if lane_index is not None and len(lane_index) >= 3:
                try:
                    return int(lane_index[2])
                except (TypeError, ValueError):
                    pass
            if hasattr(ego_vehicle, "position"):
                lane_w = float(base.config.get("lane_width", 4.0))
                lanes_n = int(base.config.get("lanes_count", 3))
                return int(np.clip(int(round(float(ego_vehicle.position[1]) / max(lane_w, 1e-6))), 0, lanes_n - 1))
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

    lane_group_stats: Dict[int, Dict[str, Any]] = {}

    def ensure_lane_group(lane_id: int) -> Dict[str, Any]:
        if lane_id not in lane_group_stats:
            lane_group_stats[lane_id] = {
                "episodes": 0,
                "ep_lens": [],
                "ep_rets": [],
                "comp_sum": {k: 0.0 for k in reward_keys},
                "comp_sum_no_collision": {k: 0.0 for k in exclude_collision_mean_keys},
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

    log("=" * 80)
    log(f"Eval model dir     : {model_path}")
    log(f"Eval run folder    : {run_folder_name}")
    log(f"Eval results dir   : {eval_dir}")
    log(f"Model file         : {os.path.join(model_path, model_name)}")
    log(f"Algo               : {algo}")
    log(f"Episodes           : {episodes}")
    log(f"Rendering enabled  : {enable_rendering}")
    log(f"Low safety filter  : {bool(env_config.get('enable_low_safety_filter', False))}")
    log("=" * 80)

    # 用于统计均值
    episode_lengths: list[int] = []
    episode_total_rewards: list[float] = []
    agg_components = {k: 0.0 for k in reward_keys}
    agg_components_no_collision = {k: 0.0 for k in exclude_collision_mean_keys}
    non_collision_episode_count = 0
    arrived_count = 0
    arrival_times: list[float] = []
    failed_count = 0
    failed_collision_count = 0
    failed_wrong_lane_count = 0
    failed_late_count = 0
    failed_early_count = 0

    seed_base = 42
    viewer_initialized = False
    for ep in range(1, int(episodes) + 1):
        episode_seed = seed_base + ep   # 按 ep 设置 seed
        obs, _ = env.reset(seed=episode_seed)
        reset_base_env = env.unwrapped
        init_lane = get_terminal_lane_id(reset_base_env)
        if init_lane is None:
            init_lane = -1

        terminated = False
        truncated = False
        step_count = 0
        ep_total_reward = 0.0
        ep_components = {k: 0.0 for k in reward_keys}
        should_record_trajectory = ep in trajectory_record_set
        trajectory_rows: list[Dict[str, Any]] = []

        if enable_rendering and not viewer_initialized:
            # 定义一个“假车”，把摄像头锁在中心点
            class Dummy:
                def __init__(self, pos):
                    self.position = np.array(pos, dtype=float)
            base = env.unwrapped
            base.render()
            base.viewer.observer_vehicle = Dummy([base.config["road_length"] / 2, 5.0])
            viewer_initialized = True

        while not (terminated or truncated):
            # 用训练好的模型选择动作
            action, _ = model.predict(obs, deterministic=True)
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # 从 env 中取出刚刚这个 step 的 reward 分量
            r_dict = info.get("reward_components", None)
            if r_dict is None:
                r_dict = getattr(env.unwrapped, "_last_weighted_rewards", None)
            if r_dict is not None:
                for k in reward_keys:
                    ep_components[k] += float(r_dict.get(k, 0.0))

            if should_record_trajectory:
                row: Dict[str, Any] = {
                    "episode": int(ep),
                    "step": int(step_count),
                    "done": int(done),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "reward": float(reward),
                }
                flat_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                flat_act = np.asarray(action, dtype=np.float32).reshape(-1)
                for i, v in enumerate(flat_obs):
                    row[f"obs_{i}"] = float(v)
                for i, v in enumerate(flat_act):
                    row[f"action_{i}"] = float(v)
                for k in reward_keys:
                    row[k] = float((r_dict or {}).get(k, 0.0))
                trajectory_rows.append(row)

            ep_total_reward += float(reward)
            step_count += 1
            obs = obs_next

        # Episode 结束，判断是否成功到达
        base_env = env.unwrapped
        crashed = getattr(base_env.vehicle, "crashed", False)
        arrived = bool(getattr(base_env, "_has_arrived", False))
        arrival_time = getattr(base_env, "_arrival_time", None)
        final_lane_id = get_terminal_lane_id(base_env)

        failed, failed_collision, failed_wrong_lane, failed_late, failed_early = classify_failure(
            bool(crashed), bool(arrived), arrival_time, final_lane_id
        )

        # 打印本 episode 结果
        reason = "unknown"
        if truncated:
            reason = "truncated(time limit)"
        else:
            # terminated
            if crashed:
                reason = "terminated(crash)"
            elif arrived:
                reason = "terminated(goal)"
            else:
                reason = "terminated(other)"
        # 统计到全局
        episode_lengths.append(step_count)
        episode_total_rewards.append(ep_total_reward)
        for k in reward_keys:
            agg_components[k] += ep_components[k]
        if not crashed:
            non_collision_episode_count += 1
            for k in exclude_collision_mean_keys:
                agg_components_no_collision[k] += ep_components[k]
        if arrived and (arrival_time is not None):
            arrived_count += 1
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

        group = ensure_lane_group(int(init_lane))
        group["episodes"] += 1
        group["ep_lens"].append(int(step_count))
        group["ep_rets"].append(float(ep_total_reward))
        for k in reward_keys:
            group["comp_sum"][k] += ep_components[k]
        if not crashed:
            group["non_collision_episode_count"] += 1
            for k in exclude_collision_mean_keys:
                group["comp_sum_no_collision"][k] += ep_components[k]
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

        # ---- 打印 / 写日志：单个 episode 结果 ----
        log("=" * 60)
        log(f"Episode {ep}:")
        log(f"  initial lane            : {init_lane}")
        log(f"  terminal lane           : {final_lane_id if final_lane_id is not None else 'N/A'}")
        log(f"  length (steps)          : {step_count}")
        log(f"  total reward            : {ep_total_reward:.6f}")
        log(f"  terminated info         : {reason}")
        log("  reward components (sum over episode):")
        for k in reward_keys:
            log(f"    {k:18s}: {ep_components[k]: .6f}")
        # 成功到达目标时，打印到达时间
        if arrived and arrival_time is not None:
            log(f"  ARRIVED at t = {arrival_time:.3f} s")
        if failed:
            log(
                "  failed flags            : "
                f"collision={int(failed_collision)}, wrong_lane={int(failed_wrong_lane)}, late={int(failed_late)}, early={int(failed_early)}"
            )
        # 如需画速度轨迹
        if enable_rendering and base_env.config["show_trajectories"]:
            save_speed_acc_curves(env, ep_idx=ep, model_path=eval_dir)
        if should_record_trajectory:
            csv_path = os.path.join(eval_dir, f"{algo}_ep_{ep:04d}_trajectory.csv")
            if trajectory_rows:
                with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=list(trajectory_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(trajectory_rows)
                log(f"  saved trajectory csv    : {csv_path}")
            else:
                log(f"  saved trajectory csv    : skipped (episode {ep} has no trajectory rows)")
    
    # ====== 统计所有 episode 的均值并打印 ======
    n = int(episodes)
    mean_len = float(np.mean(episode_lengths)) if n > 0 else 0.0
    mean_total_rew = float(np.mean(episode_total_rewards)) if n > 0 else 0.0
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
        log(f"    mean total reward     : {float(np.mean(group['ep_rets'])):.6f}")
        log("    mean reward components (per episode):")
        for k in reward_keys:
            log(
                f"      {k:16s}: "
                f"{format_component_mean(k, group['comp_sum'][k], n_lane, group['comp_sum_no_collision'].get(k, 0.0), int(group['non_collision_episode_count']))}"
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
    log(f"  mean length             : {mean_len:.3f} steps")
    log(f"  mean total reward       : {mean_total_rew:.6f}")
    log("  mean reward components (per episode):")
    for k in reward_keys:
        log(
            f"    {k:18s}: "
            f"{format_component_mean(k, agg_components[k], n, agg_components_no_collision.get(k, 0.0), non_collision_episode_count)}"
        )
    arrive_rate = arrived_count / n
    log(f"  arrival rate            : {arrive_rate * 100:.2f}%")
    if arrived_count > 0:
        mean_arrival_time = float(np.mean(arrival_times))
        log(f"  mean arrival time       : {mean_arrival_time:.3f} s (over {arrived_count} success episodes)")
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
    # main(
    #     model_path="./models/ppo_1e6",
    #     model_name="best_model.zip",
    #     algo="ppo",
    #     episodes=30,
    #     record_episode=3,
    # )
    main(
        # model_path="./models/sac_260403_base_SLv2_randomlane",
        # model_path="./models/sac_260403_withPrior_SLv2_randomlane",
        model_path="./models/sac_260403_withConstPrior_SLv2_randomlane",
        model_name="best_model.zip",
        algo="sac",
        episodes=300,
        # record_episodes=[1, 3, 5],
        record_episodes=[i for i in range(1, 301)],
        record_trajectory_episodes=[i for i in range(1, 301)],
        # enable_rendering=False,
    )