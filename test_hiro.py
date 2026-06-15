import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import importlib

import numpy as np
import os
import csv
import json
import random
import torch as th
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from util.plot_result import *
from util.config_utils import deep_update
from util.hiro_utils import (
    apply_hiro_config_overrides,
    env_config_from_run_config,
    hiro_config_from_run_config,
    load_hiro_high_model,
    load_hiro_low_model,
    load_hiro_run_config,
    unique_path,
)
from rl.algos.HRL.hiro_infer import HIROPolicyRunner
from rl.algos.HRL.goal_samplers import GoalSamplerConfig, get_goal_sampler
from configs.conf import get_env_config_for_scenario, get_scenario_spec


class OneBasedEvalRecordVideo(RecordVideo):
    """Name evaluation videos with the same 1-based episode id as trajectory CSVs."""

    def __init__(self, *args, eval_episode_number: Optional[int] = None, **kwargs):
        self.eval_episode_number = eval_episode_number
        super().__init__(*args, **kwargs)

    def start_recording(self, video_name: str):
        episode_number = (
            int(self.eval_episode_number)
            if self.eval_episode_number is not None
            else int(self.episode_id) + 1
        )
        return super().start_recording(f"hiro_ep_{episode_number:04d}")


def main(
    model_dir: str,
    episodes: int,
    record_episodes: Optional[Sequence[int]] = None,
    record_trajectory_episodes: Optional[Sequence[int]] = None,
    config_overrides: Optional[Mapping[str, Any]] = None,
    high_model_dir: Optional[str] = None,
    low_model_dir: Optional[str] = None,
    model_suffix: Optional[str] = "final",
    enable_rendering: bool = True,
    scenario_name: Optional[str] = None,
    config_model_dir: Optional[str] = None,
    env_config_model_dir: Optional[str] = None,
    seed_base: int = 42,
    episode_seeds: Optional[Sequence[int]] = None,
    independent_episodes: bool = True,
    eval_root_dir: str = "./results/eval_results",
) -> str:
    def _strict_deep_update(
        dst: Dict[str, Any],
        src: Mapping[str, Any],
        path: str,
    ) -> None:
        for key, value in src.items():
            key_path = f"{path}.{key}"
            if key not in dst:
                raise ValueError(f"Unknown config override: {key_path}")
            if isinstance(value, Mapping):
                if not isinstance(dst[key], dict):
                    raise TypeError(f"{key_path} cannot be overridden with an object")
                _strict_deep_update(dst[key], value, key_path)
            else:
                dst[key] = deepcopy(value)

    overrides = deepcopy(dict(config_overrides or {}))
    allowed_override_sections = {"environment", "hiro", "evaluation"}
    unknown_sections = set(overrides) - allowed_override_sections
    if unknown_sections:
        raise ValueError(
            "Unknown config_overrides section(s): "
            f"{sorted(unknown_sections)}. Supported: {sorted(allowed_override_sections)}"
        )

    def _override_section(name: str) -> Dict[str, Any]:
        section = overrides.get(name, {})
        if not isinstance(section, Mapping):
            raise TypeError(f"config_overrides['{name}'] must be a mapping")
        return deepcopy(dict(section))

    env_overrides = _override_section("environment")
    hiro_overrides = _override_section("hiro")
    evaluation_overrides = _override_section("evaluation")
    allowed_evaluation_overrides = {"high_policy_source"}
    unknown_evaluation_keys = set(evaluation_overrides) - allowed_evaluation_overrides
    if unknown_evaluation_keys:
        raise ValueError(
            "Unknown evaluation override(s): "
            f"{sorted(unknown_evaluation_keys)}. "
            f"Supported: {sorted(allowed_evaluation_overrides)}"
        )
    high_policy_source = str(
        evaluation_overrides.get("high_policy_source", "high_model")
    ).strip().lower()
    if high_policy_source not in {"high_model", "goal_sampler"}:
        raise ValueError(
            "config_overrides['evaluation']['high_policy_source'] must be "
            "'high_model' or 'goal_sampler'"
        )

    config_source_dir = config_model_dir or high_model_dir or model_dir
    run_config, run_config_path = load_hiro_run_config(config_source_dir)
    env_run_config, env_run_config_path = (
        load_hiro_run_config(env_config_model_dir)
        if env_config_model_dir
        else (run_config, run_config_path)
    )
    saved_metadata = env_run_config.get("run_metadata")
    if not isinstance(saved_metadata, Mapping):
        raise ValueError("run_config.json is missing the 'run_metadata' object")
    if not saved_metadata.get("scenario_name"):
        raise ValueError(
            "run_config.json is missing 'run_metadata.scenario_name'"
        )
    saved_scenario_name = str(saved_metadata["scenario_name"])
    effective_scenario_name = scenario_name or saved_scenario_name or "multi_lane"

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

    runtime_overrides: Dict[str, Any] = {
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
    scenario_spec = get_scenario_spec(effective_scenario_name)
    importlib.import_module(str(scenario_spec["module"]))
    env_id = str(scenario_spec["env_id"])

    scenario_changed = bool(
        scenario_name is not None
        and saved_scenario_name is not None
        and effective_scenario_name != saved_scenario_name
    )
    saved_env_config = (
        None
        if scenario_changed
        else env_config_from_run_config(env_run_config)
    )
    if saved_env_config is not None:
        env_config = saved_env_config
        env_config.pop("_env_seed", None)
        env_config.pop("actual_episode_start_phase_offset", None)
    else:
        env_config = get_env_config_for_scenario(effective_scenario_name)
    deep_update(env_config, runtime_overrides)
    if env_overrides:
        _strict_deep_update(
            env_config,
            env_overrides,
            "config_overrides.environment",
        )
    if not enable_rendering:
        env_config["show_trajectories"] = False
        env_config["warmup_render"] = False
        env_config["offscreen_rendering"] = False

    if record_episodes and any(int(ep_idx) < 1 for ep_idx in record_episodes):
        raise ValueError("record_episodes uses 1-based episode numbers; values must be >= 1")
    if (
        record_trajectory_episodes
        and any(int(ep_idx) < 1 for ep_idx in record_trajectory_episodes)
    ):
        raise ValueError(
            "record_trajectory_episodes uses 1-based episode numbers; values must be >= 1"
        )

    if not record_episodes:
        def trigger(ep_id: int) -> bool: return False
    else:
        record_set = {int(ep_idx) - 1 for ep_idx in record_episodes}
        def trigger(ep_id: int) -> bool: return ep_id in record_set

    trajectory_record_set = set()
    if record_trajectory_episodes:
        trajectory_record_set = {int(ep_idx) for ep_idx in record_trajectory_episodes}

    render_mode = "rgb_array" if enable_rendering else None

    def make_eval_env(episode_number: Optional[int] = None):
        base = gym.make(env_id, render_mode=render_mode, config=deepcopy(env_config))
        if not enable_rendering:
            return base
        if episode_number is None:
            episode_trigger = trigger
        else:
            should_record = trigger(int(episode_number) - 1)
            episode_trigger = lambda _episode_id, enabled=should_record: enabled
        return OneBasedEvalRecordVideo(
            base,
            video_folder=eval_dir,
            episode_trigger=episode_trigger,
            name_prefix="hiro",
            eval_episode_number=episode_number,
        )

    env = make_eval_env(1 if independent_episodes else None)

    hiro_cfg = hiro_config_from_run_config(run_config)
    if hiro_overrides:
        hiro_cfg = apply_hiro_config_overrides(hiro_cfg, hiro_overrides)
    low_level_type = str(getattr(hiro_cfg, "low_level_type", "sac")).lower()
    if low_level_type not in {"sac", "rule_based"}:
        raise ValueError(f"Unknown low_level_type: {low_level_type}")
    suffix = model_suffix or "final"
    high_model = load_hiro_high_model(
        high_model_dir or model_dir,
        model_suffix=suffix,
    )
    low_model = (
        load_hiro_low_model(
            low_model_dir or model_dir,
            model_suffix=suffix,
        )
        if low_level_type == "sac"
        else None
    )
    runner = HIROPolicyRunner(
        high_model,
        low_model,
        int(getattr(hiro_cfg, "high_interval", 25)),
        use_low_safety_layer=None,
        config=hiro_cfg,
    )

    if high_policy_source == "goal_sampler":
        cfg_goal_sampler = getattr(hiro_cfg, "goal_sampler", GoalSamplerConfig(type="uniform"))
        if isinstance(cfg_goal_sampler, GoalSamplerConfig):
            sampler_cfg = cfg_goal_sampler
        else:
            sampler_cfg = GoalSamplerConfig(
                type=str(getattr(cfg_goal_sampler, "type", "uniform")),
                path=getattr(cfg_goal_sampler, "path", None),
                device=str(getattr(cfg_goal_sampler, "device", "auto")),
                deterministic=bool(getattr(cfg_goal_sampler, "deterministic", True)),
                action=getattr(cfg_goal_sampler, "action", None),
                gaussian_mean_x_m=float(getattr(cfg_goal_sampler, "gaussian_mean_x_m", 27.0)),
                gaussian_half_range_m=float(getattr(cfg_goal_sampler, "gaussian_half_range_m", 5.0)),
            )

        def _goal_bounds_fn(high_obs_batch: np.ndarray) -> Dict[str, np.ndarray]:
            calc = getattr(runner, "high_goal_safe_bounds", None)
            if calc is None:
                raise RuntimeError("high_goal_safe_bounds is not initialized")
            return calc.compute_np(np.asarray(high_obs_batch, dtype=np.float32))

        def _goal_speed_fn(high_obs_batch: np.ndarray) -> np.ndarray:
            arr = np.asarray(high_obs_batch, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            vx_idx_in_high_obs = 1 + int(getattr(runner, "idx_vx", 2))
            if arr.shape[1] <= vx_idx_in_high_obs:
                return np.zeros((arr.shape[0],), dtype=np.float32)
            return np.maximum(arr[:, vx_idx_in_high_obs], 0.0).astype(np.float32)

        high_policy = get_goal_sampler(
            sampler_cfg,
            action_space=high_model.action_space,
            bounds_fn=_goal_bounds_fn,
            speed_fn=_goal_speed_fn,
            dynamic_feasible_lane_intervals=bool(
                getattr(hiro_cfg, "high_goal_dynamic_feasible_lane_intervals", False)
            ),
        )
        runner.high_policy = high_policy

    reward_keys_high = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "punctual_reward", "wrong_lane_terminal_penalty", "on_road_reward"]
    reward_keys_low = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "on_road_reward", "intrinsic_reward"]
    punctual_time_window = env_config.get("punctual_time_window", [20.0, 30.0])
    t_min = float(punctual_time_window[0])
    t_max = float(punctual_time_window[1])

    log("=" * 80)
    log(f"Eval HIRO model dir: {model_dir}")
    log(f"Eval run folder    : {run_folder_name}")
    log(f"Eval results dir   : {eval_dir}")
    log(f"Episodes           : {episodes}")
    log(f"Scenario           : {effective_scenario_name} ({env_id})")
    log(f"HIRO config source : {run_config_path}")
    log(f"Env config source  : {env_run_config_path}")
    log(f"Config overrides   : {json.dumps(overrides, ensure_ascii=False, sort_keys=True)}")
    log(f"Independent eps    : {independent_episodes}")
    hd = high_model_dir or model_dir
    ld = low_model_dir or model_dir
    hp = os.path.join(hd, f"hiro_high_{suffix}.zip")
    lp = os.path.join(ld, f"hiro_low_{suffix}.zip")
    log(f"HIRO high          : {hp}")
    if low_level_type == "rule_based":
        log("HIRO low           : N/A (rule-based controller)")
    else:
        log(f"HIRO low           : {lp}")
    if high_policy_source == "goal_sampler":
        sampler_name = type(runner.high_policy).__name__ if runner.high_policy is not None else "None"
        log(f"High policy source : goal sampler ({sampler_name})")
    else:
        log("High policy source : high model")
    if low_level_type == "rule_based":
        log("Low safety layer   : rule-based built-in filter")
    else:
        log(f"Low safety layer   : {runner.use_low_safety_layer}")
    log(f"Low policy source  : {low_level_type}")
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

    initial_lane_group_stats: Dict[int, Dict[str, Any]] = {}
    goal_lane_group_stats: Dict[int, Dict[str, Any]] = {}

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

    def ensure_lane_group(
        group_stats: Dict[int, Dict[str, Any]],
        lane_id: int,
    ) -> Dict[str, Any]:
        if lane_id not in group_stats:
            group_stats[lane_id] = {
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
        return group_stats[lane_id]

    def update_lane_group(
        group: Dict[str, Any],
        *,
        steps: int,
        high_ret: float,
        low_ext_mean: float,
        low_int_mean: float,
        low_total_mean: float,
        high_comp: Mapping[str, float],
        low_comp: Mapping[str, float],
        n_low_intervals: int,
        crashed: bool,
        arrived: bool,
        arrival_time: Optional[float],
        failed: bool,
        failed_collision: bool,
        failed_wrong_lane: bool,
        failed_late: bool,
        failed_early: bool,
    ) -> None:
        group["episodes"] += 1
        group["ep_lens"].append(int(steps))
        group["high_ep_rets"].append(float(high_ret))
        group["low_ep_ext_rets"].append(float(low_ext_mean))
        group["low_ep_int_rets"].append(float(low_int_mean))
        group["low_ep_total_rets"].append(float(low_total_mean))
        for key in reward_keys_high:
            group["high_comp_sum"][key] += float(high_comp[key])
        for key in reward_keys_low:
            group["low_comp_sum"][key] += float(low_comp[key]) / float(n_low_intervals)
        if not crashed:
            group["non_collision_episode_count"] += 1
            for key in exclude_collision_mean_keys:
                group["high_comp_sum_no_collision"][key] += float(high_comp[key])
                group["low_comp_sum_no_collision"][key] += (
                    float(low_comp[key]) / float(n_low_intervals)
                )
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

    def log_lane_group_summary(
        title: str,
        group_stats: Mapping[int, Dict[str, Any]],
        lanes_count: int,
    ) -> None:
        log("=" * 80)
        log(title)
        for lane_id in range(lanes_count):
            group = group_stats.get(lane_id)
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
            for key in reward_keys_high:
                log(
                    f"      {key:16s}: "
                    f"{format_component_mean(key, group['high_comp_sum'][key], n_lane, group['high_comp_sum_no_collision'].get(key, 0.0), int(group['non_collision_episode_count']))}"
                )
            log("    mean low reward components (per interval):")
            for key in reward_keys_low:
                log(
                    f"      {key:16s}: "
                    f"{format_component_mean(key, group['low_comp_sum'][key], n_lane, group['low_comp_sum_no_collision'].get(key, 0.0), int(group['non_collision_episode_count']))}"
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

    arrived_count, arrival_times = 0, []
    failed_count = 0
    failed_collision_count = 0
    failed_wrong_lane_count = 0
    failed_late_count = 0
    failed_early_count = 0
    viewer_initialized = False
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
                "hiro_config_source": run_config_path,
                "env_config_source": env_run_config_path,
                "scenario_name": effective_scenario_name,
                "env_id": env_id,
                "episode_seeds": resolved_episode_seeds,
                "independent_episodes": bool(independent_episodes),
                "config_overrides": overrides,
                "evaluation": {
                    "high_policy_source": high_policy_source,
                    "enable_rendering": bool(enable_rendering),
                },
                "environment": env_config,
                "hiro": vars(hiro_cfg),
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=lambda value: vars(value) if hasattr(value, "__dict__") else str(value),
        )

    hi_start_seen = 0
    hi_start_saved = 0
    high_segment_id = 0
    total_env_steps = 0
    policy_frequency = float(env_config.get("policy_frequency", 1.0))
    warmup_time = float(env_config.get("warmup_time", 0.0))
    warmup_each_episode = bool(env_config.get("warmup_each_episode", False))
    initial_vid = int(env.unwrapped.config.get("vid", 0))
    generated_vehicle_count = 0

    for ep in range(1, int(episodes) + 1):
        if independent_episodes and ep > 1:
            env.close()
            env = make_eval_env(ep)
            runner._inited = False
            runner.safety_controller = None
            runner.rule_based_agent = None
            viewer_initialized = False
            initial_vid = int(env.unwrapped.config.get("vid", 0))

        episode_seed = resolved_episode_seeds[ep - 1]
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        th.manual_seed(episode_seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(episode_seed)
        obs, _ = env.reset(seed=episode_seed)
        reset_base_env = env.unwrapped
        episode_time_window = reset_base_env.config.get(
            "punctual_time_window",
            punctual_time_window,
        )
        t_min = float(episode_time_window[0])
        t_max = float(episode_time_window[1])
        actual_offset_fn = getattr(reset_base_env, "get_actual_episode_start_phase_offset", None)
        actual_offset = float(actual_offset_fn()) if callable(actual_offset_fn) else None
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
        if low_model is not None:
            expected_low_dim = (
                1
                + int(runner.local_kin_flat_dim)
                + int(runner.obs_extra_dim)
                + int(runner.ego_dim)
            )
            low_shape = getattr(getattr(low_model, "observation_space", None), "shape", None)
            trained_low_dim = int(np.prod(low_shape)) if low_shape else None
            if trained_low_dim is not None and trained_low_dim != expected_low_dim:
                raise ValueError(
                    "Low-level observation dimension mismatch: "
                    f"test config builds {expected_low_dim}, model expects {trained_low_dim}. "
                    f"HIRO config source={run_config_path}, "
                    f"env config source={env_run_config_path}. "
                    "Use config_model_dir/env_config_model_dir with a complete saved config, "
                    "or provide "
                    "matching config_overrides['environment']."
                )
        should_record_trajectory = ep in trajectory_record_set
        trajectory_rows: list[Dict[str, Any]] = []

        def _build_low_obs_for_logging(obs_raw: np.ndarray) -> np.ndarray:
            """Build low_obs as [t_norm, local_kin_flat, goal_rel] (no signal dims)."""
            _, kin_local, kin_flat_local = runner._split(obs_raw)
            ego_sub_local = runner._ego_sub(kin_local)
            t_norm_local = np.array([runner.c / float(runner.hi)], dtype=np.float32)
            goal_rel_local = (runner.goal_phys - ego_sub_local).astype(np.float32)

            local_kin_flat_local = np.asarray(
                kin_flat_local[0, : runner.local_kin_flat_dim], dtype=np.float32
            ).copy()

            if (
                str(getattr(runner.cfg, "low_level_type", "sac")).lower() == "sac"
                and bool(getattr(runner.cfg, "mask_ego_position_in_low_obs", False))
            ):
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
            wrong_lane_penalty = float(rc.get("wrong_lane_terminal_penalty", 0.0))
            low_ext = float(reward) - punctual - wrong_lane_penalty

            queue_takeover_next = bool(info.get("queue_takeover_active", False))
            last_step = bool(
                done
                or (
                    runner.c == runner.hi - 1
                    and not queue_takeover_next
                )
            )
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
                    "queue_takeover_active": int(queue_takeover_next),
                    "reward": float(reward),
                    "punctual_reward": float(punctual),
                    "wrong_lane_terminal_penalty": float(wrong_lane_penalty),
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
            runner.step_end(done, queue_takeover_active=queue_takeover_next)
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
        log(f"  seed                    : {episode_seed}")
        if actual_offset is not None:
            log(f"  start phase offset      : {actual_offset:.6f} s")
        log(f"  punctual window         : [{t_min:.3f}, {t_max:.3f}] s")
        log(f"  initial lane            : {init_lane}")
        log(f"  goal lane               : {goal_lane_id}")
        log(f"  terminal lane           : {final_lane_id if final_lane_id is not None else 'N/A'}")
        log(f"  length (steps)          : {steps}")
        log(f"  terminated info         : {reason}")
        log(f"  high total reward       : {high_ret:.6f}")
        log(f"  low  ext reward (per interval mean)       : {low_ext_mean:.6f}   (env_reward - high-only task rewards)")
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

        group_update = {
            "steps": int(steps),
            "high_ret": float(high_ret),
            "low_ext_mean": float(low_ext_mean),
            "low_int_mean": float(low_int_mean),
            "low_total_mean": float(low_total_mean),
            "high_comp": high_comp,
            "low_comp": low_comp,
            "n_low_intervals": int(n_low_intervals),
            "crashed": crashed,
            "arrived": arrived,
            "arrival_time": arrival_time,
            "failed": failed,
            "failed_collision": failed_collision,
            "failed_wrong_lane": failed_wrong_lane,
            "failed_late": failed_late,
            "failed_early": failed_early,
        }
        update_lane_group(
            ensure_lane_group(initial_lane_group_stats, int(init_lane)),
            **group_update,
        )
        update_lane_group(
            ensure_lane_group(goal_lane_group_stats, int(goal_lane_id)),
            **group_update,
        )

        if independent_episodes:
            final_ep_vid = int(env.unwrapped.config.get("vid", initial_vid))
            generated_vehicle_count += max(final_ep_vid - initial_vid, 0)

    n = int(episodes)
    lanes_for_summary = int(env_config.get("lanes_count", 3))
    log_lane_group_summary(
        "Summary by initial lane:",
        initial_lane_group_stats,
        lanes_for_summary,
    )
    log_lane_group_summary(
        "Summary by goal lane:",
        goal_lane_group_stats,
        lanes_for_summary,
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

    if not independent_episodes:
        final_vid = int(env.unwrapped.config.get("vid", initial_vid))
        generated_vehicle_count = max(final_vid - initial_vid, 0)
    warmup_runs = (
        int(episodes)
        if independent_episodes or warmup_each_episode
        else (1 if warmup_time > 0.0 else 0)
    )
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
    return eval_dir


@dataclass(frozen=True)
class HIROEvalModel:
    name: str
    model_dir: str
    high_model_dir: Optional[str] = None
    low_model_dir: Optional[str] = None
    model_suffix: str = "final"
    config_model_dir: Optional[str] = None


def run_batch(
    models: Sequence[HIROEvalModel],
    episodes: int,
    *,
    seed_base: int = 42,
    episode_seeds: Optional[Sequence[int]] = None,
    batch_output_dir: str = "./results/hiro_batch",
    shared_env_config_model_dir: Optional[str] = None,
    use_each_model_env_config: bool = False,
    **eval_kwargs: Any,
) -> Dict[str, str]:
    """Evaluate several model pairs with identical per-episode seeds."""
    if not models:
        raise ValueError("models must contain at least one HIROEvalModel")
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

    batch_dir = unique_path(
        os.path.join(batch_output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    os.makedirs(batch_dir, exist_ok=True)
    results: Dict[str, str] = {}
    manifest_models = []
    shared_env_source = None
    if not use_each_model_env_config:
        shared_env_source = (
            shared_env_config_model_dir
            or models[0].config_model_dir
            or models[0].high_model_dir
            or models[0].model_dir
        )

    for spec in models:
        if spec.name in results:
            raise ValueError(f"Duplicate model name in batch: {spec.name}")
        if use_each_model_env_config:
            env_config_source = (
                spec.config_model_dir
                or spec.high_model_dir
                or spec.model_dir
            )
        else:
            env_config_source = shared_env_source
        eval_dir = main(
            model_dir=spec.model_dir,
            high_model_dir=spec.high_model_dir,
            low_model_dir=spec.low_model_dir,
            model_suffix=spec.model_suffix,
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
                "model_dir": spec.model_dir,
                "high_model_dir": spec.high_model_dir,
                "low_model_dir": spec.low_model_dir,
                "model_suffix": spec.model_suffix,
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
            # HIROEvalModel(
            #     name="oldEnv_lane2to1_randomstart",
            #     model_dir="./models/hiro_260611_rule_oldEnv_lane2to1_005_randomstart",
            # ),
            # HIROEvalModel(
            #     name="oldEnv_lane2to1_wronglanePen",
            #     model_dir="./models/hiro_260611_rule_oldEnv_lane2to1_005_wronglanePen",
            # ),
            # HIROEvalModel(
            #     name="lateGreen_lane2to2",
            #     model_dir="./models/hiro_260613_highonly_lateGreen_2to2",
            # ),
            HIROEvalModel(
                name="lateGreen_lane2to0",
                model_dir="./models/hiro_260613_highonly_lateGreen_2to0",
            ),
        ],
        episodes=100,
        record_episodes=[i for i in range(1, 101)],
        record_trajectory_episodes=[i for i in range(1, 101)],
        # enable_rendering=False,
        # scenario_name="multi_lane",
        scenario_name="multi_lane_stop_to_int",
        # shared_env_config_model_dir=(
        #     "./models/hiro_260611_rule_oldEnv_lane2to1_005_randomstart"
        # ),
        # use_each_model_env_config=False,
        # config_overrides={
        #     "environment": {
        #         # Probability configs take precedence over fixed lane IDs.
        #         "initial_lane_id": 2,
        #         "initial_lane_probs": None,
        #         "goal_lane_id": 1,
        #         # Use one reward definition so total returns are comparable.
        #         "wrong_lane_terminal_penalty": 0,
        #     },
        #     "hiro": {
        #         "use_low_safety_layer": True,
        #         "goal_sampler": {
        #             "type": "reachable_uniform",
        #         },
        #     },
        #     "evaluation": {
        #         "high_policy_source": "high_model",
        #         # "high_policy_source": "goal_sampler",
        #     },
        # },
    )
