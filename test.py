import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import os
import importlib
import csv
import json
import random
import torch as th
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from util.plot_result import save_speed_acc_curves
from configs.builders import get_env_config_for_scenario, get_hiro_config, get_scenario_spec
from util.config_utils import deep_update

from rl.algos.ppo.ppo import PPO
from rl.algos.sac.sac import SAC


class OneBasedEvalRecordVideo(RecordVideo):
    """Name evaluation videos with the same 1-based episode id as trajectory CSVs."""

    def __init__(self, *args, eval_episode_number: Optional[int] = None, name_prefix: str = "eval", **kwargs):
        self.eval_episode_number = eval_episode_number
        self.eval_name_prefix = str(name_prefix)
        super().__init__(*args, **kwargs)

    def start_recording(self, video_name: str):
        episode_number = (
            int(self.eval_episode_number)
            if self.eval_episode_number is not None
            else int(self.episode_id) + 1
        )
        return super().start_recording(f"{self.eval_name_prefix}_ep_{episode_number:04d}")


def load_model(algo: str, model_path: str, env):
    algo = algo.lower()
    if algo == "ppo":
        model = PPO.load(model_path, env=env)
    elif algo == "sac":
        model = SAC.load(model_path, env=env)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
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


def find_run_config(model_path: str) -> Optional[str]:
    model_dir_abs = os.path.abspath(model_path)
    run_name = os.path.basename(os.path.normpath(model_dir_abs))
    repo_root = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(model_dir_abs, "run_config.json"),
        os.path.join(repo_root, "logs", "current", run_name, "run_config.json"),
        os.path.join(repo_root, "logs", run_name, "run_config.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def load_run_config(model_path: str) -> Tuple[Dict[str, Any], str]:
    path = find_run_config(model_path)
    if path is None:
        raise FileNotFoundError(
            "Missing run_config.json beside the model and in same-name log "
            f"directories for: {os.path.abspath(model_path)}"
        )
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"run_config.json must be a JSON object: {path}")
    return payload, path


def env_config_from_run_config(payload: Mapping[str, Any]) -> Dict[str, Any]:
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("run_config.json is missing the 'environment' object")
    env_cfg = environment.get("env0_config")
    if not isinstance(env_cfg, Mapping):
        raise ValueError("run_config.json is missing 'environment.env0_config'")
    return deepcopy(dict(env_cfg))


def legacy_run_config(model_path: str, scenario_name: str) -> Tuple[Dict[str, Any], str]:
    return (
        {
            "run_metadata": {
                "algo": "sac",
                "scenario_name": scenario_name,
                "legacy_config_fallback": True,
                "legacy_model_dir": os.path.abspath(model_path),
            },
            "environment": {"env0_config": get_env_config_for_scenario(scenario_name)},
        },
        f"<legacy fallback for {os.path.abspath(model_path)}>",
    )


def load_run_config_or_legacy(model_path: str, scenario_name: str) -> Tuple[Dict[str, Any], str]:
    try:
        return load_run_config(model_path)
    except FileNotFoundError:
        return legacy_run_config(model_path, scenario_name)


def strict_deep_update(dst: Dict[str, Any], src: Mapping[str, Any], path: str) -> None:
    for key, value in src.items():
        key_path = f"{path}.{key}"
        if key not in dst:
            raise ValueError(f"Unknown config override: {key_path}")
        if isinstance(value, Mapping):
            if not isinstance(dst[key], dict):
                raise TypeError(f"{key_path} cannot be overridden with an object")
            strict_deep_update(dst[key], value, key_path)
        else:
            dst[key] = deepcopy(value)


def main(
    model_path: str,
    episodes: int,
    model_name: str = "best_model.zip",
    algo: str = "sac",
    record_episodes: Optional[Sequence[int]] = None,
    record_trajectory_episodes: Optional[Sequence[int]] = None,
    config_overrides: Optional[Mapping[str, Any]] = None,
    env_overrides: Optional[Mapping[str, Any]] = None,
    enable_rendering: bool = True,
    scenario_name: Optional[str] = None,
    config_model_dir: Optional[str] = None,
    env_config_model_dir: Optional[str] = None,
    seed_base: int = 42,
    episode_seeds: Optional[Sequence[int]] = None,
    independent_episodes: bool = True,
    deterministic: bool = True,
    eval_root_dir: str = "./results/eval_results",
) -> str:
    algo = str(algo).lower()

    overrides = deepcopy(dict(config_overrides or {}))
    if env_overrides:
        overrides.setdefault("environment", {})
        deep_update(overrides["environment"], deepcopy(dict(env_overrides)))
    allowed_override_sections = {"environment", "evaluation"}
    unknown_sections = set(overrides) - allowed_override_sections
    if unknown_sections:
        raise ValueError(
            "Unknown config_overrides section(s): "
            f"{sorted(unknown_sections)}. Supported: {sorted(allowed_override_sections)}"
        )
    env_eval_overrides = deepcopy(dict(overrides.get("environment", {}) or {}))
    evaluation_overrides = deepcopy(dict(overrides.get("evaluation", {}) or {}))
    if evaluation_overrides:
        unknown_eval = set(evaluation_overrides) - {"deterministic"}
        if unknown_eval:
            raise ValueError(
                "Unknown evaluation override(s): "
                f"{sorted(unknown_eval)}. Supported: ['deterministic']"
            )
        deterministic = bool(evaluation_overrides.get("deterministic", deterministic))

    default_scenario_name = scenario_name or "multi_lane"
    config_source_dir = config_model_dir or model_path
    run_config, run_config_path = load_run_config_or_legacy(config_source_dir, default_scenario_name)
    env_run_config, env_run_config_path = (
        load_run_config_or_legacy(env_config_model_dir, default_scenario_name)
        if env_config_model_dir
        else (run_config, run_config_path)
    )
    saved_metadata = env_run_config.get("run_metadata")
    saved_scenario_name = None
    if isinstance(saved_metadata, Mapping) and saved_metadata.get("scenario_name"):
        saved_scenario_name = str(saved_metadata["scenario_name"])
    effective_scenario_name = scenario_name or saved_scenario_name or default_scenario_name

    os.makedirs(eval_root_dir, exist_ok=True)
    run_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = _unique_path(os.path.join(eval_root_dir, run_folder_name))
    os.makedirs(eval_dir, exist_ok=True)

    log_path = os.path.join(eval_dir, f"eval_{algo}.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg: str = ""):
        print(msg)
        log_file.write(msg + "\n")

    runtime_overrides: Dict[str, Any] = {
        "screen_width": 1800,
        "screen_height": 300,
        "scaling": 3,
        "centering_position": [0.5, 0.5],
        "show_trajectories": enable_rendering,
        "warmup_render": False,
        "offscreen_rendering": enable_rendering,
    }
    scenario_spec = get_scenario_spec(effective_scenario_name)
    importlib.import_module(str(scenario_spec["module"]))
    env_id = str(scenario_spec["env_id"])

    scenario_changed = bool(
        scenario_name is not None
        and saved_scenario_name is not None
        and effective_scenario_name != saved_scenario_name
    )
    if scenario_changed:
        env_config = get_env_config_for_scenario(effective_scenario_name)
    else:
        env_config = get_env_config_for_scenario(effective_scenario_name)
        deep_update(env_config, env_config_from_run_config(env_run_config))
        env_config.pop("_env_seed", None)
        env_config.pop("actual_episode_start_phase_offset", None)
    deep_update(env_config, runtime_overrides)
    if env_eval_overrides:
        strict_deep_update(env_config, env_eval_overrides, "config_overrides.environment")
    if not enable_rendering:
        env_config["show_trajectories"] = False
        env_config["warmup_render"] = False
        env_config["offscreen_rendering"] = False

    # Keep SAC evaluation safety behavior consistent with training.
    if algo == "sac" and bool(env_config.get("enable_sac_low_safety_filter", False)):
        hiro_cfg_for_sac = get_hiro_config()
        if getattr(hiro_cfg_for_sac, "low_safety_filter", None) is not None:
            env_config.update(
                {
                    "enable_low_safety_filter": True,
                    "lane_change_min_front_gap": float(hiro_cfg_for_sac.low_safety_filter.lane_change_min_front_gap),
                    "lane_change_min_rear_gap": float(hiro_cfg_for_sac.low_safety_filter.lane_change_min_rear_gap),
                    "lane_change_min_front_ttc": float(hiro_cfg_for_sac.low_safety_filter.lane_change_min_front_ttc),
                    "lane_change_min_rear_ttc": float(hiro_cfg_for_sac.low_safety_filter.lane_change_min_rear_ttc),
                }
            )

    # Video recording trigger uses 1-based episode ids from the caller.
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

    def make_eval_env(episode_number: Optional[int] = None):
        base_env = gym.make(env_id, render_mode=render_mode, config=deepcopy(env_config))
        if not enable_rendering:
            return base_env
        if episode_number is None:
            episode_trigger = trigger
        else:
            should_record = trigger(int(episode_number) - 1)
            episode_trigger = lambda _ep_id, enabled=should_record: enabled
        return OneBasedEvalRecordVideo(
            base_env,
            video_folder=os.path.join(
                eval_dir,
                "videos",
                f"ep_{int(episode_number):04d}" if episode_number is not None else "all",
            ),
            episode_trigger=episode_trigger,
            name_prefix=algo,
            eval_episode_number=episode_number,
        )

    env = make_eval_env(1 if independent_episodes else None)

    # Load model after the evaluation env is built so SB3 can validate spaces.
    model = load_model(algo, os.path.join(model_path, model_name), env)
    expected_shape = getattr(getattr(model, "observation_space", None), "shape", None)
    actual_shape = getattr(getattr(env, "observation_space", None), "shape", None)
    if expected_shape is not None and actual_shape is not None and tuple(expected_shape) != tuple(actual_shape):
        env.close()
        log_file.close()
        raise ValueError(
            "Observation dimension mismatch: "
            f"test config builds {actual_shape}, model expects {expected_shape}. "
            f"Config source={run_config_path}, env config source={env_run_config_path}."
        )

    # Reward keys produced by MultiLaneEnv._rewards; missing keys default to 0.
    reward_keys = [
        "collision_reward",
        "progress_reward",
        "speed_ref_aux_reward",
        "comfort_reward",
        "lane_change_reward",
        "goal_lane_dense_reward",
        "punctual_reward",
        "wrong_lane_terminal_penalty",
        "on_road_reward",
    ]
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

    lane_group_stats: Dict[int, Dict[str, Any]] = {}
    goal_lane_group_stats: Dict[int, Dict[str, Any]] = {}

    def ensure_lane_group(group_stats: Dict[int, Dict[str, Any]], lane_id: int) -> Dict[str, Any]:
        if lane_id not in group_stats:
            group_stats[lane_id] = {
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
        return group_stats[lane_id]

    log("=" * 80)
    log(f"Eval model dir     : {model_path}")
    log(f"Eval run folder    : {run_folder_name}")
    log(f"Eval results dir   : {eval_dir}")
    log(f"Model file         : {os.path.join(model_path, model_name)}")
    log(f"Algo               : {algo}")
    log(f"Episodes           : {episodes}")
    log(f"Scenario           : {effective_scenario_name} ({env_id})")
    log(f"Config source      : {run_config_path}")
    log(f"Env config source  : {env_run_config_path}")
    log(f"Config overrides   : {json.dumps(overrides, ensure_ascii=False, sort_keys=True)}")
    log(f"Independent eps    : {independent_episodes}")
    log(f"Deterministic      : {deterministic}")
    log(f"Rendering enabled  : {enable_rendering}")
    log(f"Low safety filter  : {bool(env_config.get('enable_low_safety_filter', False))}")
    log("=" * 80)

    # Accumulators for overall evaluation statistics.
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

    if episode_seeds is None:
        resolved_episode_seeds = [int(seed_base) + ep for ep in range(1, int(episodes) + 1)]
    else:
        resolved_episode_seeds = [int(seed) for seed in episode_seeds]
        if len(resolved_episode_seeds) != int(episodes):
            raise ValueError(
                f"episode_seeds length ({len(resolved_episode_seeds)}) must equal episodes ({episodes})"
            )

    with open(os.path.join(eval_dir, "effective_eval_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "algo": algo,
                "model_path": model_path,
                "model_name": model_name,
                "scenario_name": effective_scenario_name,
                "env_id": env_id,
                "config_source": run_config_path,
                "env_config_source": env_run_config_path,
                "config_overrides": overrides,
                "episode_seeds": resolved_episode_seeds,
                "independent_episodes": bool(independent_episodes),
                "deterministic": bool(deterministic),
                "enable_rendering": bool(enable_rendering),
                "environment": env_config,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    viewer_initialized = False
    for ep in range(1, int(episodes) + 1):
        if independent_episodes and ep > 1:
            env.close()
            env = make_eval_env(ep)
            viewer_initialized = False

        episode_seed = resolved_episode_seeds[ep - 1]
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        th.manual_seed(episode_seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(episode_seed)
        obs, _ = env.reset(seed=episode_seed)
        reset_base_env = env.unwrapped
        episode_time_window = reset_base_env.config.get("punctual_time_window", punctual_time_window)
        t_min = float(episode_time_window[0])
        t_max = float(episode_time_window[1])
        actual_offset_fn = getattr(reset_base_env, "get_actual_episode_start_phase_offset", None)
        actual_offset = float(actual_offset_fn()) if callable(actual_offset_fn) else None
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
            # Keep the camera centered for global trajectory videos.
            class Dummy:
                def __init__(self, pos):
                    self.position = np.array(pos, dtype=float)
            base = env.unwrapped
            base.render()
            base.viewer.observer_vehicle = Dummy([base.config["road_length"] / 2, 5.0])
            viewer_initialized = True

        while not (terminated or truncated):
            # Select action from the trained policy.
            action, _ = model.predict(obs, deterministic=bool(deterministic))
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            # Collect weighted reward components for this step.
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
                    "queue_takeover_active": int(bool(info.get("queue_takeover_active", False))),
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

        # Episode finished; classify success and failure reason.
        base_env = env.unwrapped
        crashed = getattr(base_env.vehicle, "crashed", False)
        arrived = bool(getattr(base_env, "_has_arrived", False))
        arrival_time = getattr(base_env, "_arrival_time", None)
        final_lane_id = get_terminal_lane_id(base_env)
        goal_lane_id = int(base_env.get_goal_lane_id())

        failed, failed_collision, failed_wrong_lane, failed_late, failed_early = classify_failure(
            bool(crashed), bool(arrived), arrival_time, final_lane_id, goal_lane_id
        )

        # Log this episode.
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
        # Update overall statistics.
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

        for group in (
            ensure_lane_group(lane_group_stats, int(init_lane)),
            ensure_lane_group(goal_lane_group_stats, int(goal_lane_id)),
        ):
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

        # Per-episode text and optional trajectory artifacts.
        log("=" * 60)
        log(f"Episode {ep}:")
        log(f"  seed                    : {episode_seed}")
        if actual_offset is not None:
            log(f"  start phase offset      : {actual_offset:.6f} s")
        log(f"  punctual window         : [{t_min:.3f}, {t_max:.3f}] s")
        log(f"  initial lane            : {init_lane}")
        log(f"  goal lane               : {goal_lane_id}")
        log(f"  terminal lane           : {final_lane_id if final_lane_id is not None else 'N/A'}")
        log(f"  length (steps)          : {step_count}")
        log(f"  total reward            : {ep_total_reward:.6f}")
        log(f"  terminated info         : {reason}")
        log("  reward components (sum over episode):")
        for k in reward_keys:
            log(f"    {k:18s}: {ep_components[k]: .6f}")
        # Successful arrivals include the arrival time.
        if arrived and arrival_time is not None:
            log(f"  ARRIVED at t = {arrival_time:.3f} s")
        if failed:
            log(
                "  failed flags            : "
                f"collision={int(failed_collision)}, wrong_lane={int(failed_wrong_lane)}, late={int(failed_late)}, early={int(failed_early)}"
            )
        # Save speed/acceleration curves when trajectory rendering is enabled.
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
    
    # Summary over all evaluated episodes.
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
    log("Summary by goal lane:")
    for lane_id in range(lanes_for_summary):
        group = goal_lane_group_stats.get(lane_id)
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
    return eval_dir


@dataclass(frozen=True)
class EvalModel:
    name: str
    model_path: str
    model_name: str = "best_model.zip"
    algo: str = "sac"
    config_model_dir: Optional[str] = None


def run_batch(
    models: Sequence[EvalModel],
    episodes: int,
    *,
    seed_base: int = 42,
    episode_seeds: Optional[Sequence[int]] = None,
    batch_output_dir: str = "./results/sac_batch",
    shared_env_config_model_dir: Optional[str] = None,
    use_each_model_env_config: bool = False,
    **eval_kwargs: Any,
) -> Dict[str, str]:
    """Evaluate several models with identical per-episode seeds."""
    if not models:
        raise ValueError("models must contain at least one EvalModel")
    eval_kwargs.pop("independent_episodes", None)
    eval_kwargs.pop("episode_seeds", None)
    eval_kwargs.pop("seed_base", None)
    eval_kwargs.pop("env_config_model_dir", None)
    seeds = (
        [int(seed) for seed in episode_seeds]
        if episode_seeds is not None
        else [int(seed_base) + ep for ep in range(1, int(episodes) + 1)]
    )
    if len(seeds) != int(episodes):
        raise ValueError(f"episode_seeds length ({len(seeds)}) must equal episodes ({episodes})")

    batch_dir = _unique_path(os.path.join(batch_output_dir, datetime.now().strftime("%Y%m%d_%H%M%S")))
    os.makedirs(batch_dir, exist_ok=True)
    results: Dict[str, str] = {}
    manifest_models = []
    shared_env_source = None
    if not use_each_model_env_config:
        shared_env_source = shared_env_config_model_dir or models[0].config_model_dir or models[0].model_path

    for spec in models:
        if spec.name in results:
            raise ValueError(f"Duplicate model name in batch: {spec.name}")
        env_config_source = (
            (spec.config_model_dir or spec.model_path)
            if use_each_model_env_config
            else shared_env_source
        )
        eval_dir = main(
            model_path=spec.model_path,
            model_name=spec.model_name,
            algo=spec.algo,
            config_model_dir=spec.config_model_dir,
            env_config_model_dir=env_config_source,
            episodes=int(episodes),
            episode_seeds=seeds,
            independent_episodes=True,
            **eval_kwargs,
        )
        results[spec.name] = eval_dir
        manifest_models.append(
            {
                "name": spec.name,
                "algo": spec.algo,
                "model_path": spec.model_path,
                "model_name": spec.model_name,
                "config_model_dir": spec.config_model_dir,
                "env_config_model_dir": env_config_source,
                "eval_dir": eval_dir,
            }
        )

    with open(os.path.join(batch_dir, "batch_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "episodes": int(episodes),
                "episode_seeds": seeds,
                "shared_env_config_model_dir": shared_env_source,
                "use_each_model_env_config": use_each_model_env_config,
                "config_overrides": deepcopy(eval_kwargs.get("config_overrides", {})),
                "models": manifest_models,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return results


if __name__ == "__main__":
    run_batch(
        models=[
            EvalModel(
                name="sac_260622_withPrior_2to2_noGoalReshape",
                model_path="./models/sac_260622_withPrior_2to2_noGoalReshape",
                model_name="best_model.zip",
                algo="sac",
            ),
        ],
        episodes=30,
        record_episodes=[i for i in range(1, 31)],
        record_trajectory_episodes=[i for i in range(1, 31)],
        scenario_name="multi_lane_stop_to_int",
        shared_env_config_model_dir=None,
        use_each_model_env_config=False,
        config_overrides={
            "environment": {
                "initial_lane_id": "2",
                "goal_lane_id": "2",
            },
            "evaluation": {
                "deterministic": True,
            },
        },
        # enable_rendering=False,
    )
