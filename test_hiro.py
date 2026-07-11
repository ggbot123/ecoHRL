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
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from util.plot_result import save_goal_snapshot, save_speed_acc_curves
from util.config_utils import deep_update
from util.hiro_utils import (
    env_config_from_run_config,
    hiro_checkpoint_path,
    hiro_config_from_run_config,
    load_hiro_high_model,
    load_hiro_low_model,
    load_hiro_run_config,
    unique_path,
)
from rl.algos.HRL.hiro_infer import HIROPolicyRunner
from rl.algos.HRL.goal_samplers import GoalSamplerConfig, get_goal_sampler
from rl.algos.sac.sac import SAC
from scenarios.goal_lane_logic import sample_goal_lane_id
from configs.builders import get_env_config_for_scenario, get_hiro_config, get_scenario_spec
from configs.conf import TRAIN_CONFIG


LANE_TRAFFIC_CONFIG_KEYS = (
    "behavior_probs",
    "behavior_lane_probs",
    "behavior_vehicle_types",
    "flow_speed_range",
    "speed_distribution",
    "spawn_probability",
    "spawn_min_gap",
    "spawn_min_t_headway",
    "spawn_check_adjacent_cutins",
    "spawn_adjacent_cutin_front_gap",
    "spawn_adjacent_cutin_back_gap",
    "movement_lanes",
    "movement_behavior_probs",
    "background_vehicle_respect_movement_lanes",
    "background_snapshot_reset",
    "background_snapshot_path",
    "background_snapshot_paths",
    "background_snapshot_max_resample_attempts",
    "background_snapshot_chunk_reuse_enabled",
    "background_snapshot_chunk_reuse_count",
    "background_snapshot_chunk_cache_size",
    "background_snapshot_phase_offset",
    "signal_plan",
    "enable_signal_green_launch_behavior",
    "signal_green_launch_approach_distance",
    "signal_green_launch_end_margin",
    "signal_green_launch_target_speed",
    "enable_signal_cycle_spawn_probability",
    "signal_cycle_spawn_probability",
)


class OneBasedEvalRecordVideo(RecordVideo):
    """Name evaluation videos with the same 1-based episode id as trajectory CSVs."""

    def __init__(
        self,
        *args,
        eval_episode_number: Optional[int] = None,
        name_prefix: str = "hiro",
        **kwargs,
    ):
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


def _legacy_hiro_run_config(model_dir: str) -> Tuple[Dict[str, Any], str]:
    """Build a minimal config for old HIRO checkpoints saved before run_config.json."""
    hiro_cfg = asdict(get_hiro_config())
    hiro_cfg.update(
        {
            "train_mode": "high_only",
            "low_level_type": "rule_based",
            "high_obs_use_signal_features": False,
            "use_low_safety_layer": True,
        }
    )
    env_cfg = get_env_config_for_scenario("multi_lane")
    payload = {
        "run_metadata": {
            "scenario_name": "multi_lane",
            "legacy_config_fallback": True,
            "legacy_model_dir": os.path.abspath(model_dir),
            "high_obs_time_mode": "elapsed",
            "high_obs_x_mode": "elapsed",
        },
        "environment": {"env0_config": env_cfg},
        "hiro": {"config": hiro_cfg},
    }
    return payload, f"<legacy fallback for {os.path.abspath(model_dir)}>"


def _load_hiro_run_config_or_legacy(model_dir: str) -> Tuple[Dict[str, Any], str]:
    try:
        return load_hiro_run_config(model_dir)
    except FileNotFoundError:
        return _legacy_hiro_run_config(model_dir)


def _legacy_sac_run_config(model_dir: str, scenario_name: str) -> Tuple[Dict[str, Any], str]:
    env_cfg = get_env_config_for_scenario(scenario_name)
    payload = {
        "run_metadata": {
            "algo": "sac",
            "scenario_name": scenario_name,
            "legacy_config_fallback": True,
            "legacy_model_dir": os.path.abspath(model_dir),
        },
        "environment": {"env0_config": env_cfg},
    }
    return payload, f"<legacy SAC fallback for {os.path.abspath(model_dir)}>"


def _load_sac_run_config_or_legacy(
    model_dir: str,
    scenario_name: str,
) -> Tuple[Dict[str, Any], str]:
    try:
        return load_hiro_run_config(model_dir)
    except FileNotFoundError:
        return _legacy_sac_run_config(model_dir, scenario_name)


def _high_model_source_path(source: str, model_suffix: str) -> str:
    """Return the concrete high-model zip path for a directory or zip source."""
    src = str(source)
    if src.lower().endswith(".zip"):
        return src
    return hiro_checkpoint_path(src, "hiro_high", model_suffix)


def _high_model_config_source_dir(source: str) -> str:
    src = str(source)
    if src.lower().endswith(".zip"):
        return os.path.dirname(src) or "."
    return src


def _load_hiro_high_model_source(source: str, model_suffix: str):
    src = str(source)
    if src.lower().endswith(".zip"):
        return SAC.load(src)
    return load_hiro_high_model(src, model_suffix=model_suffix)


def _sac_model_source_path(source: str, model_name: str) -> str:
    """Return the concrete SAC zip path for a directory or zip source."""
    src = str(source)
    if src.lower().endswith(".zip"):
        return src
    return os.path.join(src, model_name)


def _sac_model_config_source_dir(source: str) -> str:
    src = str(source)
    if src.lower().endswith(".zip"):
        return os.path.dirname(src) or "."
    return src


def _load_sac_model_source(source: str, model_name: str, env: Any):
    return SAC.load(_sac_model_source_path(source, model_name), env=env)


def _clear_high_model_env_bindings(high_model: Any) -> None:
    actor = getattr(high_model, "actor", None)
    if actor is not None and hasattr(actor, "goal_safe_bounds_fn"):
        actor.goal_safe_bounds_fn = None


def _normalize_high_model_by_goal_lane(
    high_model_by_goal_lane: Optional[Mapping[Any, str]],
) -> Dict[int, str]:
    if not high_model_by_goal_lane:
        return {}
    normalized: Dict[int, str] = {}
    for lane_id, source in high_model_by_goal_lane.items():
        lane = int(lane_id)
        if lane in normalized:
            raise ValueError(f"Duplicate high model mapping for goal lane {lane}")
        normalized[lane] = str(source)
    return normalized


def _normalize_sac_model_by_goal_lane(
    sac_model_by_goal_lane: Optional[Mapping[Any, str]],
) -> Dict[int, str]:
    if not sac_model_by_goal_lane:
        return {}
    normalized: Dict[int, str] = {}
    for lane_id, source in sac_model_by_goal_lane.items():
        lane = int(lane_id)
        if lane in normalized:
            raise ValueError(f"Duplicate SAC model mapping for goal lane {lane}")
        normalized[lane] = str(source)
    return normalized


def _extract_lane_traffic_config(saved_env_config: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: deepcopy(saved_env_config[key])
        for key in LANE_TRAFFIC_CONFIG_KEYS
        if key in saved_env_config
    }


def _extract_sac_goal_lane_env_config(
    saved_env_config: Mapping[str, Any],
) -> Dict[str, Any]:
    cfg = _extract_lane_traffic_config(saved_env_config)
    action_cfg = saved_env_config.get("action", {})
    if isinstance(action_cfg, Mapping) and "acceleration_range" in action_cfg:
        cfg.setdefault("action", {})["acceleration_range"] = deepcopy(
            action_cfg["acceleration_range"]
        )
    return cfg


def _sac_goal_lane_env_config_keys() -> list[str]:
    return [*LANE_TRAFFIC_CONFIG_KEYS, "action.acceleration_range"]


def _train_eval_config_overrides(*, algo: str) -> Dict[str, Any]:
    train_overrides = TRAIN_CONFIG.get("config_overrides", {}) or {}
    if not isinstance(train_overrides, Mapping):
        raise TypeError("TRAIN_CONFIG['config_overrides'] must be a mapping")
    sections = (
        ("environment", "hiro", "evaluation")
        if str(algo).lower() == "hiro"
        else ("environment", "sac_environment", "evaluation")
    )
    normalized: Dict[str, Any] = {}
    for section in sections:
        value = train_overrides.get(section, {})
        if value:
            if not isinstance(value, Mapping):
                raise TypeError(
                    f"TRAIN_CONFIG['config_overrides']['{section}'] must be a mapping"
                )
            normalized[section] = deepcopy(dict(value))
    return normalized


def _normalize_eval_config_overrides(
    config_overrides: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    explicit = config_overrides or {}
    if not isinstance(explicit, Mapping):
        raise TypeError("config_overrides must be a mapping")
    normalized: Dict[str, Any] = {}
    for section, value in explicit.items():
        if not isinstance(value, Mapping):
            raise TypeError(f"config_overrides['{section}'] must be a mapping")
        normalized[str(section)] = deepcopy(dict(value))
    return normalized


def _merge_config_override_layers(*layers: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for layer in layers:
        for section, value in layer.items():
            if not isinstance(value, Mapping):
                raise TypeError(f"config_overrides['{section}'] must be a mapping")
            target = merged.setdefault(str(section), {})
            if not isinstance(target, dict):
                raise TypeError(f"Merged config_overrides['{section}'] must be a mapping")
            deep_update(target, deepcopy(dict(value)))
    return merged


def _is_legacy_run_config(payload: Mapping[str, Any]) -> bool:
    metadata = payload.get("run_metadata")
    return bool(
        isinstance(metadata, Mapping)
        and metadata.get("legacy_config_fallback")
    )


def _hiro_config_mapping_from_run_config(payload: Mapping[str, Any]) -> Dict[str, Any]:
    hiro_section = payload.get("hiro")
    if not isinstance(hiro_section, Mapping):
        raise ValueError("run_config.json is missing the 'hiro' object")
    saved = hiro_section.get("config")
    if not isinstance(saved, Mapping):
        raise ValueError("run_config.json is missing 'hiro.config'")
    return deepcopy(dict(saved))


def _override_section(
    overrides: Mapping[str, Any],
    name: str,
    *,
    label: str = "config_overrides",
) -> Dict[str, Any]:
    section = overrides.get(name, {})
    if not isinstance(section, Mapping):
        raise TypeError(f"{label}['{name}'] must be a mapping")
    return deepcopy(dict(section))


def _env_overrides_for_algo(
    overrides: Mapping[str, Any],
    *,
    algo: str,
) -> Dict[str, Any]:
    env_overrides = _override_section(overrides, "environment")
    if str(algo).lower() == "sac":
        deep_update(env_overrides, _override_section(overrides, "sac_environment"))
    return env_overrides


def _pop_hiro_eval_modes(
    hiro_overrides: Dict[str, Any],
    *,
    label: str,
) -> Dict[str, Any]:
    hiro_eval_modes: Dict[str, Any] = {}
    for mode_key in ("high_obs_time_mode", "high_obs_x_mode"):
        if mode_key in hiro_overrides:
            mode_val = str(hiro_overrides.pop(mode_key)).lower().strip()
            if mode_val not in {"remaining", "elapsed"}:
                raise ValueError(
                    f"{label}['hiro']['{mode_key}'] must be "
                    "'remaining' or 'elapsed'"
                )
            hiro_eval_modes[mode_key] = mode_val
    return hiro_eval_modes


def main(
    model_dir: str,
    episodes: int,
    record_episodes: Optional[Sequence[int]] = None,
    record_trajectory_episodes: Optional[Sequence[int]] = None,
    config_overrides: Optional[Mapping[str, Any]] = None,
    high_model_dir: Optional[str] = None,
    high_model_by_goal_lane: Optional[Mapping[Any, str]] = None,
    low_model_dir: Optional[str] = None,
    low_model_path: Optional[str] = None,
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
    conf_overrides = _train_eval_config_overrides(algo="hiro")
    test_overrides = _normalize_eval_config_overrides(config_overrides)
    overrides = _merge_config_override_layers(conf_overrides, test_overrides)
    suffix = model_suffix or "final"
    high_model_sources_by_goal_lane = _normalize_high_model_by_goal_lane(
        high_model_by_goal_lane
    )
    if high_model_sources_by_goal_lane and not independent_episodes:
        raise ValueError(
            "high_model_by_goal_lane requires independent_episodes=True so each "
            "episode can use its goal-lane-specific environment traffic config."
        )
    allowed_override_sections = {"environment", "hiro", "evaluation"}
    unknown_sections = set(overrides) - allowed_override_sections
    if unknown_sections:
        raise ValueError(
            "Unknown config_overrides section(s): "
            f"{sorted(unknown_sections)}. Supported: {sorted(allowed_override_sections)}"
        )

    conf_env_overrides = _env_overrides_for_algo(conf_overrides, algo="hiro")
    test_env_overrides = _env_overrides_for_algo(test_overrides, algo="hiro")
    conf_hiro_overrides = _override_section(
        conf_overrides,
        "hiro",
        label="TRAIN_CONFIG['config_overrides']",
    )
    test_hiro_overrides = _override_section(test_overrides, "hiro")
    conf_hiro_eval_modes = _pop_hiro_eval_modes(
        conf_hiro_overrides,
        label="TRAIN_CONFIG['config_overrides']",
    )
    test_hiro_eval_modes = _pop_hiro_eval_modes(
        test_hiro_overrides,
        label="config_overrides",
    )
    evaluation_overrides = _merge_config_override_layers(
        {"evaluation": _override_section(conf_overrides, "evaluation")},
        {"evaluation": _override_section(test_overrides, "evaluation")},
    ).get("evaluation", {})
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
    run_config, run_config_path = _load_hiro_run_config_or_legacy(config_source_dir)
    env_run_config, env_run_config_path = (
        _load_hiro_run_config_or_legacy(env_config_model_dir)
        if env_config_model_dir
        else (run_config, run_config_path)
    )
    run_is_legacy = _is_legacy_run_config(run_config)
    env_run_is_legacy = _is_legacy_run_config(env_run_config)
    saved_hiro_config = _hiro_config_mapping_from_run_config(run_config)
    hiro_config_mapping = asdict(get_hiro_config())
    if run_is_legacy:
        deep_update(hiro_config_mapping, saved_hiro_config)
        deep_update(hiro_config_mapping, conf_hiro_overrides)
    else:
        deep_update(hiro_config_mapping, conf_hiro_overrides)
        deep_update(hiro_config_mapping, saved_hiro_config)
    deep_update(hiro_config_mapping, test_hiro_overrides)
    hiro_cfg = hiro_config_from_run_config({"hiro": {"config": hiro_config_mapping}})
    run_metadata = run_config.get("run_metadata")
    hiro_eval_modes: Dict[str, Any] = {}
    if run_is_legacy:
        hiro_eval_modes.update(conf_hiro_eval_modes)
        hiro_eval_modes.setdefault(
            "high_obs_time_mode",
            str(run_metadata.get("high_obs_time_mode", "elapsed")),
        )
        hiro_eval_modes.setdefault(
            "high_obs_x_mode",
            str(run_metadata.get("high_obs_x_mode", "elapsed")),
        )
    hiro_eval_modes.update(test_hiro_eval_modes)
    for mode_key, mode_val in hiro_eval_modes.items():
        setattr(hiro_cfg, mode_key, mode_val)
    saved_metadata = run_config.get("run_metadata")
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

    env_metadata = env_run_config.get("run_metadata")
    env_saved_scenario_name = (
        str(env_metadata["scenario_name"])
        if isinstance(env_metadata, Mapping) and env_metadata.get("scenario_name")
        else saved_scenario_name
    )
    scenario_changed = bool(
        env_config_model_dir
        and scenario_name is not None
        and env_saved_scenario_name
        and effective_scenario_name != env_saved_scenario_name
    )
    if scenario_changed:
        raise ValueError(
            "scenario_name does not match the saved environment config: "
            f"requested scenario_name={effective_scenario_name!r}, "
            f"saved scenario_name={env_saved_scenario_name!r} "
            f"from {env_run_config_path}. "
            "Use a matching scenario_name or a matching env_config_model_dir."
        )
    env_config = get_env_config_for_scenario(
        effective_scenario_name,
        conf_env_overrides,
    )
    env_config_source_parts = [
        f"configs/conf.py scenario={effective_scenario_name}",
        "TRAIN_CONFIG.config_overrides.environment",
    ]
    if not env_run_is_legacy:
        saved_env_config = env_config_from_run_config(env_run_config)
        deep_update(env_config, saved_env_config)
        env_config.pop("_env_seed", None)
        env_config.pop("actual_episode_start_phase_offset", None)
        env_config_source_parts.append(
            f"{env_run_config_path} environment.env0_config"
        )
    else:
        env_config_source_parts.append(
            f"{env_run_config_path} (legacy fallback; no saved env override)"
        )
    deep_update(env_config, test_env_overrides)
    env_config_source_parts.append("test config_overrides.environment")
    env_config_source_desc = " + ".join(env_config_source_parts)
    deep_update(env_config, runtime_overrides)
    if not enable_rendering:
        env_config["show_trajectories"] = False
        env_config["warmup_render"] = False
        env_config["offscreen_rendering"] = False

    lane_traffic_overrides_by_goal_lane: Dict[int, Dict[str, Any]] = {}
    lane_traffic_source_by_goal_lane: Dict[int, str] = {}
    if high_model_sources_by_goal_lane:
        for lane, source in sorted(high_model_sources_by_goal_lane.items()):
            lane_run_config, lane_run_config_path = _load_hiro_run_config_or_legacy(
                _high_model_config_source_dir(source)
            )
            lane_metadata = lane_run_config.get("run_metadata")
            lane_scenario_name = (
                str(lane_metadata["scenario_name"])
                if isinstance(lane_metadata, Mapping)
                and lane_metadata.get("scenario_name")
                else saved_scenario_name
            )
            if lane_scenario_name != effective_scenario_name:
                raise ValueError(
                    "Goal-lane model config scenario does not match evaluation "
                    f"scenario: goal_lane={lane}, "
                    f"saved scenario_name={lane_scenario_name!r}, "
                    f"evaluation scenario_name={effective_scenario_name!r}, "
                    f"source={lane_run_config_path}"
                )
            lane_traffic_overrides_by_goal_lane[lane] = (
                {}
                if _is_legacy_run_config(lane_run_config)
                else _extract_lane_traffic_config(env_config_from_run_config(lane_run_config))
            )
            lane_traffic_source_by_goal_lane[lane] = lane_run_config_path

    def sample_eval_goal_lane_id(seed: int) -> int:
        return int(
            sample_goal_lane_id(
                np.random.default_rng(int(seed)),
                goal_lane_id=env_config.get("goal_lane_id", 0),
                lanes_count=int(env_config.get("lanes_count", 1)),
                goal_lane_probs=env_config.get("goal_lane_probs", None),
            )
        )

    def env_config_for_goal_lane(goal_lane_id: int) -> Dict[str, Any]:
        lane = int(goal_lane_id)
        cfg = deepcopy(env_config)
        if lane_traffic_overrides_by_goal_lane:
            if lane not in lane_traffic_overrides_by_goal_lane:
                raise ValueError(
                    f"No traffic config configured for goal_lane_id={lane}. "
                    f"Available lanes: {sorted(lane_traffic_overrides_by_goal_lane)}"
                )
            deep_update(cfg, deepcopy(lane_traffic_overrides_by_goal_lane[lane]))
            deep_update(cfg, test_env_overrides)
        cfg["goal_lane_id"] = lane
        cfg["goal_lane_probs"] = None
        return cfg

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

    def make_eval_env(
        episode_number: Optional[int] = None,
        config: Optional[Mapping[str, Any]] = None,
    ):
        base = gym.make(
            env_id,
            render_mode=render_mode,
            config=deepcopy(dict(config or env_config)),
        )
        if not enable_rendering:
            return base
        if episode_number is None:
            episode_trigger = trigger
        else:
            should_record = trigger(int(episode_number) - 1)
            episode_trigger = lambda _episode_id, enabled=should_record: enabled
        return OneBasedEvalRecordVideo(
            base,
            video_folder=os.path.join(
                eval_dir,
                "videos",
                f"ep_{int(episode_number):04d}" if episode_number is not None else "all",
            ),
            episode_trigger=episode_trigger,
            name_prefix="hiro",
            eval_episode_number=episode_number,
        )

    env = make_eval_env(1 if independent_episodes else None)

    low_level_type = str(getattr(hiro_cfg, "low_level_type", "sac")).lower()
    if low_level_type not in {"sac", "rule_based"}:
        raise ValueError(f"Unknown low_level_type: {low_level_type}")
    if high_model_sources_by_goal_lane and high_policy_source == "goal_sampler":
        raise ValueError(
            "high_model_by_goal_lane cannot be combined with "
            "config_overrides['evaluation']['high_policy_source']='goal_sampler'"
        )
    high_load_dir = high_model_dir or model_dir
    high_models_by_goal_lane = {
        lane: _load_hiro_high_model_source(source, suffix)
        for lane, source in high_model_sources_by_goal_lane.items()
    }
    if high_models_by_goal_lane:
        active_high_goal_lane: Optional[int] = sorted(high_models_by_goal_lane)[0]
        high_model = high_models_by_goal_lane[active_high_goal_lane]
    else:
        active_high_goal_lane = None
        high_model = load_hiro_high_model(
            high_load_dir,
            model_suffix=suffix,
        )
    _clear_high_model_env_bindings(high_model)
    low_model = None
    low_load_path = None
    if low_level_type == "sac":
        if low_model_path and low_model_dir:
            raise ValueError("Use either low_model_path or low_model_dir, not both")
        if low_model_path:
            low_load_path = str(low_model_path)
            low_load_dir = os.path.dirname(low_load_path) or "."
            low_load_suffix = os.path.basename(low_load_path)
        else:
            low_load_dir = low_model_dir or model_dir
            low_load_suffix = suffix
            low_load_path = hiro_checkpoint_path(low_load_dir, "hiro_low", low_load_suffix)
        if not os.path.isfile(low_load_path):
            low_pretrained_path = getattr(hiro_cfg, "low_pretrained_path", None)
            hint = (
                f" run_config low_pretrained_path={low_pretrained_path!r}; "
                "pass it via low_model_path explicitly if this is the intended low model."
                if low_pretrained_path
                else ""
            )
            raise FileNotFoundError(
                f"Low-level HIRO checkpoint not found: {low_load_path}.{hint}"
            )
        low_model = load_hiro_low_model(
            low_load_dir,
            model_suffix=low_load_suffix,
        )
    runner = HIROPolicyRunner(
        high_model,
        low_model,
        int(getattr(hiro_cfg, "high_interval", 25)),
        use_low_safety_layer=None,
        config=hiro_cfg,
    )

    def select_high_model_for_goal_lane(goal_lane_id: int) -> None:
        nonlocal active_high_goal_lane
        if not high_models_by_goal_lane:
            return
        lane = int(goal_lane_id)
        if lane not in high_models_by_goal_lane:
            raise ValueError(
                f"No high-level model configured for goal_lane_id={lane}. "
                f"Available lanes: {sorted(high_models_by_goal_lane)}"
            )
        if (
            active_high_goal_lane == lane
            and runner.high_model is high_models_by_goal_lane[lane]
            and runner._inited
        ):
            return
        runner.high_model = high_models_by_goal_lane[lane]
        active_high_goal_lane = lane
        _clear_high_model_env_bindings(runner.high_model)
        runner._inited = False

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

    reward_keys_high = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "goal_lane_dense_reward", "punctual_reward", "wrong_lane_terminal_penalty", "on_road_reward"]
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
    log(f"Env config source  : {env_config_source_desc}")
    log(f"Config overrides   : {json.dumps(overrides, ensure_ascii=False, sort_keys=True)}")
    log(f"Independent eps    : {independent_episodes}")
    hd = high_model_dir or model_dir
    ld = low_model_dir or model_dir
    hp = hiro_checkpoint_path(hd, "hiro_high", suffix)
    lp = low_load_path or hiro_checkpoint_path(ld, "hiro_low", suffix)
    if high_model_sources_by_goal_lane:
        log("HIRO high          : by goal lane")
        for lane, source in sorted(high_model_sources_by_goal_lane.items()):
            log(f"  goal_lane={lane:<2d}       : {_high_model_source_path(source, suffix)}")
        log("Env traffic config : by goal lane")
        for lane, source in sorted(lane_traffic_source_by_goal_lane.items()):
            log(f"  goal_lane={lane:<2d}       : {source}")
    else:
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
    log(
        "High obs modes     : "
        f"time={getattr(hiro_cfg, 'high_obs_time_mode', 'remaining')}, "
        f"x={getattr(hiro_cfg, 'high_obs_x_mode', 'remaining')}, "
        f"signal_pref={getattr(hiro_cfg, 'high_obs_use_signal_features', True)}"
    )
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
                "env_config_source": env_config_source_desc,
                "scenario_name": effective_scenario_name,
                "env_id": env_id,
                "episode_seeds": resolved_episode_seeds,
                "independent_episodes": bool(independent_episodes),
                "config_overrides": overrides,
                "evaluation": {
                    "high_policy_source": high_policy_source,
                    "enable_rendering": bool(enable_rendering),
                    "high_model_by_goal_lane": {
                        str(lane): _high_model_source_path(source, suffix)
                        for lane, source in sorted(high_model_sources_by_goal_lane.items())
                    },
                    "lane_traffic_config_source": {
                        str(lane): source
                        for lane, source in sorted(
                            lane_traffic_source_by_goal_lane.items()
                        )
                    },
                    "lane_traffic_config_keys": list(LANE_TRAFFIC_CONFIG_KEYS),
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
    observed_background_only_time = 0.0
    observed_background_time_available = True

    def get_background_only_time(current_env) -> Optional[float]:
        value = getattr(current_env.unwrapped, "_background_only_sim_time", None)
        if value is None:
            return None
        return float(value)

    for ep in range(1, int(episodes) + 1):
        episode_seed = resolved_episode_seeds[ep - 1]
        planned_goal_lane_id: Optional[int] = None
        episode_env_config: Mapping[str, Any] = env_config
        if lane_traffic_overrides_by_goal_lane:
            planned_goal_lane_id = sample_eval_goal_lane_id(episode_seed)
            episode_env_config = env_config_for_goal_lane(planned_goal_lane_id)

        if independent_episodes and (ep > 1 or lane_traffic_overrides_by_goal_lane):
            env.close()
            env = make_eval_env(ep, episode_env_config)
            runner._inited = False
            runner.safety_controller = None
            runner.rule_based_agent = None
            viewer_initialized = False
            initial_vid = int(env.unwrapped.config.get("vid", 0))

        random.seed(episode_seed)
        np.random.seed(episode_seed)
        th.manual_seed(episode_seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(episode_seed)
        background_time_before_reset = get_background_only_time(env)
        obs, _ = env.reset(seed=episode_seed)
        reset_base_env = env.unwrapped
        goal_lane_getter = getattr(reset_base_env, "get_goal_lane_id", None)
        if callable(goal_lane_getter):
            goal_lane_id = int(goal_lane_getter())
        else:
            goal_lane_id = int(reset_base_env.config.get("goal_lane_id", 0))
        if planned_goal_lane_id is not None and goal_lane_id != planned_goal_lane_id:
            raise RuntimeError(
                "Planned goal lane and environment goal lane diverged: "
                f"planned={planned_goal_lane_id}, actual={goal_lane_id}"
            )
        select_high_model_for_goal_lane(goal_lane_id)
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
                    f"env config source={env_config_source_desc}. "
                    "Provide "
                    "matching config_overrides['environment']."
                )
        should_record_trajectory = ep in trajectory_record_set
        trajectory_rows: list[Dict[str, Any]] = []

        def _build_low_obs_for_logging(obs_raw: np.ndarray) -> np.ndarray:
            """Build the same low_obs layout that HIROPolicyRunner sends to the low model."""
            _, kin_local, kin_flat_local = runner._split(obs_raw)
            ego_sub_local = runner._ego_sub(kin_local)
            t_norm_local = np.array([runner.c / float(runner.hi)], dtype=np.float32)
            goal_rel_local = (runner.goal_phys - ego_sub_local).astype(np.float32)
            obs_arr_local = np.asarray(obs_raw, dtype=np.float32).reshape(-1)
            extra_local = obs_arr_local[
                1 + runner.kin_flat_dim : 1 + runner.kin_flat_dim + runner.obs_extra_dim
            ].astype(np.float32)

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

            return np.concatenate(
                [t_norm_local, local_kin_flat_local, extra_local, goal_rel_local]
            ).astype(np.float32)

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
                try:
                    hi_start_high_obs = runner._build_high_obs(
                        np.asarray(obs, dtype=np.float32),
                        env,
                    )
                    _, hi_start_kin, _ = runner._split(obs)
                    hi_start_ego_sub = runner._ego_sub(hi_start_kin).astype(np.float32)
                except Exception:
                    hi_start_high_obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
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
            goal_lane_dense = float(rc.get("goal_lane_dense_reward", 0.0))
            wrong_lane_penalty = float(rc.get("wrong_lane_terminal_penalty", 0.0))
            low_ext = float(reward) - goal_lane_dense - punctual - wrong_lane_penalty

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
                    "goal_lane_dense_reward": float(goal_lane_dense),
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
        if high_models_by_goal_lane:
            log(f"  high model goal lane    : {active_high_goal_lane}")
            log(
                "  traffic config source   : "
                f"{lane_traffic_source_by_goal_lane.get(int(goal_lane_id), 'N/A')}"
            )
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
            background_time_after_episode = get_background_only_time(env)
            if background_time_before_reset is None or background_time_after_episode is None:
                observed_background_time_available = False
            else:
                observed_background_only_time += max(
                    background_time_after_episode - background_time_before_reset,
                    0.0,
                )

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
        background_time_after_eval = get_background_only_time(env)
        if background_time_after_eval is None:
            observed_background_time_available = False
        else:
            observed_background_only_time = max(background_time_after_eval, 0.0)
    warmup_runs = (
        int(episodes)
        if independent_episodes or warmup_each_episode
        else (1 if warmup_time > 0.0 else 0)
    )
    total_warmup_time = warmup_time * float(warmup_runs)
    total_episode_time = float(total_env_steps) / max(policy_frequency, 1e-6)
    total_background_only_time = (
        observed_background_only_time
        if observed_background_time_available
        else total_warmup_time
    )
    total_sim_time = total_episode_time + total_background_only_time
    traffic_flow_veh_per_s = (
        float(generated_vehicle_count) / total_sim_time if total_sim_time > 0.0 else 0.0
    )

    log("  traffic flow stats      :")
    log(f"    generated vehicles    : {generated_vehicle_count}")
    log(f"    total sim time        : {total_sim_time:.3f} s (episode={total_episode_time:.3f} s, background-only={total_background_only_time:.3f} s)")
    log(f"    flow                  : {traffic_flow_veh_per_s:.6f} veh/s ({traffic_flow_veh_per_s * 3600.0:.3f} veh/h)")
    log("=" * 80)

    log_file.close()
    env.close()
    return eval_dir


def main_sac(
    model_dir: str,
    episodes: int,
    model_name: str = "best_model.zip",
    sac_model_by_goal_lane: Optional[Mapping[Any, str]] = None,
    record_episodes: Optional[Sequence[int]] = None,
    record_trajectory_episodes: Optional[Sequence[int]] = None,
    config_overrides: Optional[Mapping[str, Any]] = None,
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
    conf_overrides = _train_eval_config_overrides(algo="sac")
    test_overrides = _normalize_eval_config_overrides(config_overrides)
    overrides = _merge_config_override_layers(conf_overrides, test_overrides)
    sac_model_sources_by_goal_lane = _normalize_sac_model_by_goal_lane(
        sac_model_by_goal_lane
    )
    if sac_model_sources_by_goal_lane and not independent_episodes:
        raise ValueError(
            "sac_model_by_goal_lane requires independent_episodes=True so each "
            "episode can use its goal-lane-specific environment traffic config."
        )
    allowed_override_sections = {"environment", "sac_environment", "evaluation"}
    unknown_sections = set(overrides) - allowed_override_sections
    if unknown_sections:
        raise ValueError(
            "Unknown config_overrides section(s): "
            f"{sorted(unknown_sections)}. Supported: {sorted(allowed_override_sections)}"
        )

    conf_env_overrides = _env_overrides_for_algo(conf_overrides, algo="sac")
    test_env_overrides = _env_overrides_for_algo(test_overrides, algo="sac")
    evaluation_overrides = _merge_config_override_layers(
        {"evaluation": _override_section(conf_overrides, "evaluation")},
        {"evaluation": _override_section(test_overrides, "evaluation")},
    ).get("evaluation", {})
    unknown_eval = set(evaluation_overrides) - {"deterministic"}
    if unknown_eval:
        raise ValueError(
            "Unknown evaluation override(s): "
            f"{sorted(unknown_eval)}. Supported: ['deterministic']"
        )
    deterministic = bool(evaluation_overrides.get("deterministic", deterministic))

    default_scenario_name = scenario_name or "multi_lane"
    config_source_dir = config_model_dir or model_dir
    run_config, run_config_path = _load_sac_run_config_or_legacy(
        config_source_dir,
        default_scenario_name,
    )
    env_run_config, env_run_config_path = (
        _load_sac_run_config_or_legacy(env_config_model_dir, default_scenario_name)
        if env_config_model_dir
        else (run_config, run_config_path)
    )
    env_run_is_legacy = _is_legacy_run_config(env_run_config)

    saved_metadata = env_run_config.get("run_metadata")
    saved_scenario_name = (
        str(saved_metadata["scenario_name"])
        if isinstance(saved_metadata, Mapping) and saved_metadata.get("scenario_name")
        else default_scenario_name
    )
    effective_scenario_name = scenario_name or saved_scenario_name or default_scenario_name
    if (
        scenario_name is not None
        and saved_scenario_name
        and effective_scenario_name != saved_scenario_name
    ):
        raise ValueError(
            "scenario_name does not match the saved SAC environment config: "
            f"requested scenario_name={effective_scenario_name!r}, "
            f"saved scenario_name={saved_scenario_name!r} "
            f"from {env_run_config_path}."
        )

    os.makedirs(eval_root_dir, exist_ok=True)
    run_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = unique_path(os.path.join(eval_root_dir, run_folder_name))
    os.makedirs(eval_dir, exist_ok=True)

    log_path = os.path.join(eval_dir, "eval_sac.txt")
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

    env_config = get_env_config_for_scenario(
        effective_scenario_name,
        conf_env_overrides,
    )
    if not env_run_is_legacy:
        deep_update(env_config, env_config_from_run_config(env_run_config))
        env_config.pop("_env_seed", None)
        env_config.pop("actual_episode_start_phase_offset", None)
    deep_update(env_config, test_env_overrides)
    deep_update(env_config, runtime_overrides)
    if not enable_rendering:
        env_config["show_trajectories"] = False
        env_config["warmup_render"] = False
        env_config["offscreen_rendering"] = False

    if bool(env_config.get("enable_sac_low_safety_filter", False)):
        hiro_cfg_for_sac = get_hiro_config()
        if getattr(hiro_cfg_for_sac, "low_safety_filter", None) is not None:
            env_config.update(
                {
                    "enable_low_safety_filter": True,
                    "lane_change_min_front_gap": float(
                        hiro_cfg_for_sac.low_safety_filter.lane_change_min_front_gap
                    ),
                    "lane_change_min_rear_gap": float(
                        hiro_cfg_for_sac.low_safety_filter.lane_change_min_rear_gap
                    ),
                    "lane_change_min_front_ttc": float(
                        hiro_cfg_for_sac.low_safety_filter.lane_change_min_front_ttc
                    ),
                    "lane_change_min_rear_ttc": float(
                        hiro_cfg_for_sac.low_safety_filter.lane_change_min_rear_ttc
                    ),
                }
            )

    lane_traffic_overrides_by_goal_lane: Dict[int, Dict[str, Any]] = {}
    lane_traffic_source_by_goal_lane: Dict[int, str] = {}
    if sac_model_sources_by_goal_lane:
        for lane, source in sorted(sac_model_sources_by_goal_lane.items()):
            lane_run_config, lane_run_config_path = _load_sac_run_config_or_legacy(
                _sac_model_config_source_dir(source),
                effective_scenario_name,
            )
            lane_metadata = lane_run_config.get("run_metadata")
            lane_scenario_name = (
                str(lane_metadata["scenario_name"])
                if isinstance(lane_metadata, Mapping)
                and lane_metadata.get("scenario_name")
                else saved_scenario_name
            )
            if lane_scenario_name != effective_scenario_name:
                raise ValueError(
                    "Goal-lane SAC model config scenario does not match evaluation "
                    f"scenario: goal_lane={lane}, "
                    f"saved scenario_name={lane_scenario_name!r}, "
                    f"evaluation scenario_name={effective_scenario_name!r}, "
                    f"source={lane_run_config_path}"
                )
            lane_traffic_overrides_by_goal_lane[lane] = (
                {}
                if _is_legacy_run_config(lane_run_config)
                else _extract_sac_goal_lane_env_config(
                    env_config_from_run_config(lane_run_config)
                )
            )
            lane_traffic_source_by_goal_lane[lane] = lane_run_config_path

    def sample_eval_goal_lane_id(seed: int) -> int:
        return int(
            sample_goal_lane_id(
                np.random.default_rng(int(seed)),
                goal_lane_id=env_config.get("goal_lane_id", 0),
                lanes_count=int(env_config.get("lanes_count", 1)),
                goal_lane_probs=env_config.get("goal_lane_probs", None),
            )
        )

    def env_config_for_goal_lane(goal_lane_id: int) -> Dict[str, Any]:
        lane = int(goal_lane_id)
        cfg = deepcopy(env_config)
        if lane_traffic_overrides_by_goal_lane:
            if lane not in lane_traffic_overrides_by_goal_lane:
                raise ValueError(
                    f"No traffic config configured for goal_lane_id={lane}. "
                    f"Available lanes: {sorted(lane_traffic_overrides_by_goal_lane)}"
                )
            deep_update(cfg, deepcopy(lane_traffic_overrides_by_goal_lane[lane]))
            deep_update(cfg, test_env_overrides)
        cfg["goal_lane_id"] = lane
        cfg["goal_lane_probs"] = None
        return cfg

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

    trajectory_record_set = (
        {int(ep_idx) for ep_idx in record_trajectory_episodes}
        if record_trajectory_episodes
        else set()
    )
    render_mode = "rgb_array" if enable_rendering else None

    def make_eval_env(
        episode_number: Optional[int] = None,
        config: Optional[Mapping[str, Any]] = None,
    ):
        base = gym.make(
            env_id,
            render_mode=render_mode,
            config=deepcopy(dict(config or env_config)),
        )
        if not enable_rendering:
            return base
        if episode_number is None:
            episode_trigger = trigger
        else:
            should_record = trigger(int(episode_number) - 1)
            episode_trigger = lambda _episode_id, enabled=should_record: enabled
        return OneBasedEvalRecordVideo(
            base,
            video_folder=os.path.join(
                eval_dir,
                "videos",
                f"ep_{int(episode_number):04d}" if episode_number is not None else "all",
            ),
            episode_trigger=episode_trigger,
            name_prefix="sac",
            eval_episode_number=episode_number,
        )

    env = make_eval_env(1 if independent_episodes else None)
    model_path = os.path.join(model_dir, model_name)
    sac_models_by_goal_lane = {
        lane: _load_sac_model_source(source, model_name, env)
        for lane, source in sac_model_sources_by_goal_lane.items()
    }
    if sac_models_by_goal_lane:
        active_sac_goal_lane: Optional[int] = sorted(sac_models_by_goal_lane)[0]
        model = sac_models_by_goal_lane[active_sac_goal_lane]
    else:
        active_sac_goal_lane = None
        model = SAC.load(model_path, env=env)

    actual_shape = getattr(getattr(env, "observation_space", None), "shape", None)
    models_for_shape_check = (
        sorted(sac_models_by_goal_lane.items())
        if sac_models_by_goal_lane
        else [(None, model)]
    )
    for lane_for_check, model_for_check in models_for_shape_check:
        expected_shape = getattr(
            getattr(model_for_check, "observation_space", None),
            "shape",
            None,
        )
        if (
            expected_shape is not None
            and actual_shape is not None
            and tuple(expected_shape) != tuple(actual_shape)
        ):
            env.close()
            log_file.close()
            lane_hint = (
                f" for goal_lane_id={lane_for_check}"
                if lane_for_check is not None
                else ""
            )
            raise ValueError(
                "SAC observation dimension mismatch"
                f"{lane_hint}: test config builds {actual_shape}, "
                f"model expects {expected_shape}. Config source={run_config_path}, "
                f"env config source={env_run_config_path}."
            )

    def select_sac_model_for_goal_lane(goal_lane_id: int) -> None:
        nonlocal model, active_sac_goal_lane
        if not sac_models_by_goal_lane:
            return
        lane = int(goal_lane_id)
        if lane not in sac_models_by_goal_lane:
            raise ValueError(
                f"No SAC model configured for goal_lane_id={lane}. "
                f"Available lanes: {sorted(sac_models_by_goal_lane)}"
            )
        model = sac_models_by_goal_lane[lane]
        active_sac_goal_lane = lane

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
    exclude_collision_mean_keys = {"comfort_reward", "lane_change_reward"}
    punctual_time_window = env_config.get("punctual_time_window", [20.0, 30.0])
    t_min = float(punctual_time_window[0])
    t_max = float(punctual_time_window[1])

    def get_terminal_lane_id(base: Any) -> Optional[int]:
        ego_vehicle = getattr(base, "vehicle", None)
        if ego_vehicle is not None:
            lane_index = getattr(ego_vehicle, "lane_index", None)
            if lane_index is not None and len(lane_index) >= 3:
                return int(lane_index[2])
            if hasattr(ego_vehicle, "position"):
                lane_w = float(base.config.get("lane_width", 4.0))
                lanes_n = int(base.config.get("lanes_count", 3))
                return int(
                    np.clip(
                        int(round(float(ego_vehicle.position[1]) / max(lane_w, 1e-6))),
                        0,
                        lanes_n - 1,
                    )
                )
        return None

    def classify_failure(
        crashed: bool,
        arrived: bool,
        arrival_time: Optional[float],
        final_lane_id: Optional[int],
        goal_lane_id: Optional[int],
    ) -> Tuple[bool, bool, bool, bool, bool]:
        if crashed:
            return True, True, False, False, False
        on_time_arrival = bool(
            arrived and arrival_time is not None and t_min <= float(arrival_time) <= t_max
        )
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

    def log_failed_breakdown(
        prefix: str,
        failed_count: int,
        collision_count: int,
        wrong_lane_count: int,
        late_count: int,
        early_count: int,
    ) -> None:
        other_count = max(
            int(failed_count)
            - int(collision_count)
            - int(wrong_lane_count)
            - int(late_count)
            - int(early_count),
            0,
        )
        log(f"{prefix}failed episodes       : {failed_count}")
        if failed_count <= 0:
            log(f"{prefix}collision            : 0")
            log(f"{prefix}wrong-lane at end    : 0")
            log(f"{prefix}late arrival         : 0")
            log(f"{prefix}early arrival        : 0")
            return
        log(f"{prefix}collision            : {collision_count} ({collision_count / failed_count * 100:.2f}% of failed)")
        log(f"{prefix}wrong-lane at end    : {wrong_lane_count} ({wrong_lane_count / failed_count * 100:.2f}% of failed)")
        log(f"{prefix}late arrival         : {late_count} ({late_count / failed_count * 100:.2f}% of failed)")
        log(f"{prefix}early arrival        : {early_count} ({early_count / failed_count * 100:.2f}% of failed)")
        if other_count > 0:
            log(f"{prefix}other failures       : {other_count} ({other_count / failed_count * 100:.2f}% of failed)")

    initial_lane_group_stats: Dict[int, Dict[str, Any]] = {}
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

    def update_lane_group(group: Dict[str, Any], **kwargs: Any) -> None:
        group["episodes"] += 1
        group["ep_lens"].append(int(kwargs["steps"]))
        group["ep_rets"].append(float(kwargs["ret"]))
        for key in reward_keys:
            group["comp_sum"][key] += float(kwargs["comp"][key])
        if not bool(kwargs["crashed"]):
            group["non_collision_episode_count"] += 1
            for key in exclude_collision_mean_keys:
                group["comp_sum_no_collision"][key] += float(kwargs["comp"][key])
        if bool(kwargs["arrived"]):
            group["arrived_count"] += 1
            if kwargs["arrival_time"] is not None:
                group["arrival_times"].append(float(kwargs["arrival_time"]))
        for key in (
            "failed",
            "failed_collision",
            "failed_wrong_lane",
            "failed_late",
            "failed_early",
        ):
            group[f"{key}_count"] += int(bool(kwargs[key]))

    def log_lane_group_summary(title: str, group_stats: Dict[int, Dict[str, Any]], lanes: int) -> None:
        log("=" * 80)
        log(title)
        for lane_id in range(int(lanes)):
            group = group_stats.get(lane_id)
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
            for key in reward_keys:
                log(
                    f"      {key:16s}: "
                    f"{format_component_mean(key, group['comp_sum'][key], n_lane, group['comp_sum_no_collision'].get(key, 0.0), int(group['non_collision_episode_count']))}"
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

    if episode_seeds is None:
        resolved_episode_seeds = [int(seed_base) + ep for ep in range(1, int(episodes) + 1)]
    else:
        resolved_episode_seeds = [int(seed) for seed in episode_seeds]
        if len(resolved_episode_seeds) != int(episodes):
            raise ValueError(
                f"episode_seeds length ({len(resolved_episode_seeds)}) must equal episodes ({episodes})"
            )

    log("=" * 80)
    log(f"Eval SAC model dir : {model_dir}")
    log(f"Eval run folder    : {run_folder_name}")
    log(f"Eval results dir   : {eval_dir}")
    if sac_model_sources_by_goal_lane:
        log("SAC model          : by goal lane")
        for lane, source in sorted(sac_model_sources_by_goal_lane.items()):
            log(f"  goal_lane={lane:<2d}       : {_sac_model_source_path(source, model_name)}")
        log("Env traffic config : by goal lane")
        for lane, source in sorted(lane_traffic_source_by_goal_lane.items()):
            lane_cfg = lane_traffic_overrides_by_goal_lane.get(lane, {})
            action_cfg = lane_cfg.get("action", {})
            acc_range = (
                action_cfg.get("acceleration_range")
                if isinstance(action_cfg, Mapping)
                else None
            )
            acc_suffix = f" | acc_range={acc_range}" if acc_range is not None else ""
            log(f"  goal_lane={lane:<2d}       : {source}{acc_suffix}")
    else:
        log(f"Model file         : {model_path}")
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

    with open(os.path.join(eval_dir, "effective_eval_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "algo": "sac",
                "model_dir": model_dir,
                "model_name": model_name,
                "model_path": model_path,
                "sac_model_by_goal_lane": {
                    str(lane): _sac_model_source_path(source, model_name)
                    for lane, source in sorted(sac_model_sources_by_goal_lane.items())
                },
                "scenario_name": effective_scenario_name,
                "env_id": env_id,
                "config_source": run_config_path,
                "env_config_source": env_run_config_path,
                "lane_traffic_config_source": {
                    str(lane): source
                    for lane, source in sorted(lane_traffic_source_by_goal_lane.items())
                },
                "lane_traffic_config_keys": _sac_goal_lane_env_config_keys(),
                "lane_action_acceleration_range_by_goal_lane": {
                    str(lane): deepcopy(
                        lane_cfg.get("action", {}).get("acceleration_range")
                    )
                    for lane, lane_cfg in sorted(
                        lane_traffic_overrides_by_goal_lane.items()
                    )
                    if isinstance(lane_cfg.get("action", {}), Mapping)
                    and "acceleration_range" in lane_cfg.get("action", {})
                },
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

    ep_lens: list[int] = []
    ep_rets: list[float] = []
    comp_sum = {k: 0.0 for k in reward_keys}
    comp_sum_no_collision = {k: 0.0 for k in exclude_collision_mean_keys}
    non_collision_episode_count = 0
    arrived_count = 0
    arrival_times: list[float] = []
    failed_count = 0
    failed_collision_count = 0
    failed_wrong_lane_count = 0
    failed_late_count = 0
    failed_early_count = 0
    viewer_initialized = False

    for ep in range(1, int(episodes) + 1):
        episode_seed = resolved_episode_seeds[ep - 1]
        planned_goal_lane_id: Optional[int] = None
        episode_env_config: Mapping[str, Any] = env_config
        if lane_traffic_overrides_by_goal_lane:
            planned_goal_lane_id = sample_eval_goal_lane_id(episode_seed)
            episode_env_config = env_config_for_goal_lane(planned_goal_lane_id)

        if independent_episodes and (ep > 1 or lane_traffic_overrides_by_goal_lane):
            env.close()
            env = make_eval_env(ep, episode_env_config)
            viewer_initialized = False

        random.seed(episode_seed)
        np.random.seed(episode_seed)
        th.manual_seed(episode_seed)
        if th.cuda.is_available():
            th.cuda.manual_seed_all(episode_seed)

        obs, _ = env.reset(seed=episode_seed)
        reset_base_env = env.unwrapped
        goal_lane_getter = getattr(reset_base_env, "get_goal_lane_id", None)
        if callable(goal_lane_getter):
            goal_lane_id = int(goal_lane_getter())
        else:
            goal_lane_id = int(reset_base_env.config.get("goal_lane_id", 0))
        if planned_goal_lane_id is not None and goal_lane_id != planned_goal_lane_id:
            raise RuntimeError(
                "Planned goal lane and environment goal lane diverged: "
                f"planned={planned_goal_lane_id}, actual={goal_lane_id}"
            )
        select_sac_model_for_goal_lane(goal_lane_id)
        episode_time_window = reset_base_env.config.get(
            "punctual_time_window",
            punctual_time_window,
        )
        t_min = float(episode_time_window[0])
        t_max = float(episode_time_window[1])
        actual_offset_fn = getattr(reset_base_env, "get_actual_episode_start_phase_offset", None)
        actual_offset = float(actual_offset_fn()) if callable(actual_offset_fn) else None
        init_lane = get_terminal_lane_id(reset_base_env)
        if init_lane is None:
            init_lane = -1

        if enable_rendering and not viewer_initialized:
            class Dummy:
                def __init__(self, pos):
                    self.position = np.array(pos, dtype=float)
            base = env.unwrapped
            base.render()
            base.viewer.observer_vehicle = Dummy([base.config["road_length"] / 2, 5.0])
            viewer_initialized = True

        terminated, truncated, steps = False, False, 0
        ep_ret = 0.0
        comp = {k: 0.0 for k in reward_keys}
        should_record_trajectory = ep in trajectory_record_set
        trajectory_rows: list[Dict[str, Any]] = []

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=bool(deterministic))
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            rc = info.get("reward_components", None)
            if rc is None:
                rc = getattr(env.unwrapped, "_last_weighted_rewards", None)
            rc = rc or {}
            for key in reward_keys:
                comp[key] += float(rc.get(key, 0.0))

            if should_record_trajectory:
                row: Dict[str, Any] = {
                    "episode": int(ep),
                    "step": int(steps),
                    "done": int(done),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "queue_takeover_active": int(bool(info.get("queue_takeover_active", False))),
                    "reward": float(reward),
                }
                flat_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                flat_action = np.asarray(action, dtype=np.float32).reshape(-1)
                for i, value in enumerate(flat_obs):
                    row[f"obs_{i}"] = float(value)
                for i, value in enumerate(flat_action):
                    row[f"action_{i}"] = float(value)
                for key in reward_keys:
                    row[key] = float(rc.get(key, 0.0))
                trajectory_rows.append(row)

            ep_ret += float(reward)
            steps += 1
            obs = obs_next

        base_env = env.unwrapped
        crashed = bool(getattr(base_env.vehicle, "crashed", False))
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

        ep_lens.append(int(steps))
        ep_rets.append(float(ep_ret))
        for key in reward_keys:
            comp_sum[key] += comp[key]
        if not crashed:
            non_collision_episode_count += 1
            for key in exclude_collision_mean_keys:
                comp_sum_no_collision[key] += comp[key]
        if arrived:
            arrived_count += 1
            if arrival_time is not None:
                arrival_times.append(float(arrival_time))
        failed_count += int(failed)
        failed_collision_count += int(failed_collision)
        failed_wrong_lane_count += int(failed_wrong_lane)
        failed_late_count += int(failed_late)
        failed_early_count += int(failed_early)

        group_update = {
            "steps": int(steps),
            "ret": float(ep_ret),
            "comp": comp,
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

        reason = "terminated" if terminated else ("truncated(time limit)" if truncated else "unknown")
        log("=" * 60)
        log(f"Episode {ep}:")
        log(f"  seed                    : {episode_seed}")
        if actual_offset is not None:
            log(f"  start phase offset      : {actual_offset:.6f} s")
        log(f"  punctual window         : [{t_min:.3f}, {t_max:.3f}] s")
        log(f"  initial lane            : {init_lane}")
        log(f"  goal lane               : {goal_lane_id}")
        if sac_model_sources_by_goal_lane:
            log(f"  SAC model goal lane     : {active_sac_goal_lane}")
            log(
                "  traffic config source   : "
                f"{lane_traffic_source_by_goal_lane.get(int(goal_lane_id), 'N/A')}"
            )
            action_cfg = base_env.config.get("action", {})
            if isinstance(action_cfg, Mapping) and "acceleration_range" in action_cfg:
                log(
                    "  action acc range        : "
                    f"{action_cfg.get('acceleration_range')}"
                )
        log(f"  terminal lane           : {final_lane_id if final_lane_id is not None else 'N/A'}")
        log(f"  length (steps)          : {steps}")
        log(f"  total reward            : {ep_ret:.6f}")
        log(f"  terminated info         : {reason}")
        log("  reward components (sum over episode):")
        for key in reward_keys:
            log(f"    {key:18s}: {comp[key]: .6f}")
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
            csv_path = os.path.join(eval_dir, f"sac_ep_{ep:04d}_trajectory.csv")
            if trajectory_rows:
                with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=list(trajectory_rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(trajectory_rows)
                log(f"  saved trajectory csv    : {csv_path}")
            else:
                log(f"  saved trajectory csv    : skipped (episode {ep} has no trajectory rows)")

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
    log(f"  mean total reward       : {float(np.mean(ep_rets)):.6f}")
    log("  mean reward components (per episode):")
    for key in reward_keys:
        log(
            f"    {key:18s}: "
            f"{format_component_mean(key, comp_sum[key], n, comp_sum_no_collision.get(key, 0.0), non_collision_episode_count)}"
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
    return eval_dir


@dataclass(frozen=True)
class HIROEvalModel:
    name: str
    model_dir: str
    high_model_dir: Optional[str] = None
    high_model_by_goal_lane: Optional[Mapping[Any, str]] = None
    low_model_dir: Optional[str] = None
    low_model_path: Optional[str] = None
    model_suffix: str = "final"
    config_model_dir: Optional[str] = None
    low_level_type: Optional[str] = None
    high_obs_time_mode: Optional[str] = None
    high_obs_x_mode: Optional[str] = None
    high_obs_use_signal_features: Optional[bool] = None
    high_goal_infeasible_action_mode: Optional[str] = None


@dataclass(frozen=True)
class SACEvalModel:
    name: str
    model_dir: str
    model_name: str = "best_model.zip"
    sac_model_by_goal_lane: Optional[Mapping[Any, str]] = None
    config_model_dir: Optional[str] = None


def run_sac_batch(
    models: Sequence[SACEvalModel],
    episodes: int,
    *,
    seed_base: int = 42,
    episode_seeds: Optional[Sequence[int]] = None,
    batch_output_dir: str = "./results/sac_batch",
    shared_env_config_model_dir: Optional[str] = None,
    use_each_model_env_config: bool = False,
    **eval_kwargs: Any,
) -> Dict[str, str]:
    """Evaluate several single-level SAC models with identical per-episode seeds."""
    if not models:
        raise ValueError("models must contain at least one SACEvalModel")
    eval_kwargs.pop("independent_episodes", None)
    eval_kwargs.pop("episode_seeds", None)
    eval_kwargs.pop("seed_base", None)
    explicit_env_config_model_dir = eval_kwargs.pop("env_config_model_dir", None)
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
    if explicit_env_config_model_dir:
        shared_env_source = explicit_env_config_model_dir
    elif shared_env_config_model_dir:
        shared_env_source = shared_env_config_model_dir
    elif not use_each_model_env_config:
        shared_env_source = models[0].config_model_dir or models[0].model_dir

    for spec in models:
        if spec.name in results:
            raise ValueError(f"Duplicate SAC model name in batch: {spec.name}")
        env_config_source = (
            (spec.config_model_dir or spec.model_dir)
            if use_each_model_env_config
            else shared_env_source
        )
        eval_dir = main_sac(
            model_dir=spec.model_dir,
            model_name=spec.model_name,
            sac_model_by_goal_lane=spec.sac_model_by_goal_lane,
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
                "model_name": spec.model_name,
                "sac_model_by_goal_lane": (
                    {
                        str(lane): _sac_model_source_path(source, spec.model_name)
                        for lane, source in sorted(
                            _normalize_sac_model_by_goal_lane(
                                spec.sac_model_by_goal_lane
                            ).items()
                        )
                    }
                    if spec.sac_model_by_goal_lane
                    else None
                ),
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
    explicit_env_config_model_dir = eval_kwargs.pop("env_config_model_dir", None)
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
    if explicit_env_config_model_dir:
        shared_env_source = explicit_env_config_model_dir
    elif shared_env_config_model_dir:
        shared_env_source = shared_env_config_model_dir
    env_config_source_note = (
        str(shared_env_source)
        if shared_env_source is not None
        else "configs/conf.py scenario defaults + config_overrides.environment"
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
        model_eval_kwargs = dict(eval_kwargs)
        if (
            spec.low_level_type is not None
            or spec.high_obs_time_mode is not None
            or spec.high_obs_x_mode is not None
            or spec.high_obs_use_signal_features is not None
            or spec.high_goal_infeasible_action_mode is not None
        ):
            merged_overrides = deepcopy(model_eval_kwargs.get("config_overrides", {}))
            hiro_section = merged_overrides.setdefault("hiro", {})
            if not isinstance(hiro_section, dict):
                raise TypeError("config_overrides['hiro'] must be a mapping")
            if spec.low_level_type is not None:
                hiro_section["low_level_type"] = str(spec.low_level_type)
            if spec.high_obs_time_mode is not None:
                hiro_section["high_obs_time_mode"] = str(spec.high_obs_time_mode)
            if spec.high_obs_x_mode is not None:
                hiro_section["high_obs_x_mode"] = str(spec.high_obs_x_mode)
            if spec.high_obs_use_signal_features is not None:
                hiro_section["high_obs_use_signal_features"] = bool(
                    spec.high_obs_use_signal_features
                )
            if spec.high_goal_infeasible_action_mode is not None:
                high_goal_safety = hiro_section.setdefault("high_goal_safety", {})
                if not isinstance(high_goal_safety, dict):
                    raise TypeError("config_overrides['hiro']['high_goal_safety'] must be a mapping")
                high_goal_safety["infeasible_action_mode"] = str(
                    spec.high_goal_infeasible_action_mode
                )
            model_eval_kwargs["config_overrides"] = merged_overrides
        eval_dir = main(
            model_dir=spec.model_dir,
            high_model_dir=spec.high_model_dir,
            high_model_by_goal_lane=spec.high_model_by_goal_lane,
            low_model_dir=spec.low_model_dir,
            low_model_path=spec.low_model_path,
            model_suffix=spec.model_suffix,
            config_model_dir=spec.config_model_dir,
            env_config_model_dir=env_config_source,
            episodes=int(episodes),
            episode_seeds=seeds,
            independent_episodes=True,
            **model_eval_kwargs,
        )
        results[spec.name] = eval_dir
        manifest_models.append(
            {
                "name": spec.name,
                "model_dir": spec.model_dir,
                "high_model_dir": spec.high_model_dir,
                "high_model_by_goal_lane": (
                    {
                        str(lane): source
                        for lane, source in sorted(
                            _normalize_high_model_by_goal_lane(
                                spec.high_model_by_goal_lane
                            ).items()
                        )
                    }
                    if spec.high_model_by_goal_lane
                    else None
                ),
                "low_model_dir": spec.low_model_dir,
                "low_model_path": spec.low_model_path,
                "model_suffix": spec.model_suffix,
                "config_model_dir": spec.config_model_dir,
                "low_level_type": spec.low_level_type,
                "high_obs_time_mode": spec.high_obs_time_mode,
                "high_obs_x_mode": spec.high_obs_x_mode,
                "high_obs_use_signal_features": spec.high_obs_use_signal_features,
                "high_goal_infeasible_action_mode": spec.high_goal_infeasible_action_mode,
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
                "env_config_source": env_config_source_note,
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
            #     name="hiro_260331_highonly_rule_accwithSL_randomLane",
            #     model_dir="./models/hiro_260331_highonly_rule_accwithSL_randomLane",
            #     low_level_type="rule_based",
            # ),
            # HIROEvalModel(
            #     name="hiro_260706_highonly_ruleReUni_oldEnv",
            #     model_dir="./models/hiro_260706_highonly_ruleReUni_oldEnv",
            #     low_level_type="rule_based",
            # ),
            # HIROEvalModel(
            #     name="uni_fixedHER_oldEnv_old",
            #     model_dir="./models/hiro_260331_highonly_reachableUniformLane1_Rainbow_amax3_dmin15_10_randomlane",
            #     low_model_path="./models/hiro_260328_lowonly_reachablePretrainedV2_Rainbow_amax3_dmin15_10/hiro_low_final.zip",
            #     low_level_type="sac",
            #     high_obs_time_mode="elapsed",
            #     high_obs_x_mode="elapsed",
            #     high_obs_use_signal_features=False,
            #     high_goal_infeasible_action_mode="preserve",
            # )
            # HIROEvalModel(
            #     name="uni_fixedHER_oldEnv",
            #     model_dir="./models/hiro_260628_highonly_pretrained_uni_oldEnv_fixedHER_SLmpc_noaugObs",
            #     low_model_path="./models/hiro_260627_lowonly_uni_oldEnv_fixedHER_SLmpc_noaugObs/hiro_low_final.zip",
            # ),
            # HIROEvalModel(
            #     name="uni_noHER_oldEnv",
            #     model_dir="./models/hiro_260628_highonly_pretrained_uni_oldEnv_noHER_SLmpc_noaugObs",
            #     low_model_path="./models/hiro_260627_lowonly_uni_oldEnv_noHER_SLmpc_noaugObs/hiro_low_final.zip",
            # ),
            # HIROEvalModel(
            #     name="reUni_fixedHER_oldEnv",
            #     model_dir="./models/hiro_260702_highonly_reUni_lowSnapshot_oldEnv_fixedHER_newPolicy_withPrior",
            #     low_model_path="./models/hiro_260630_lowonly_reUni_oldEnv_fixedHER_snapshot/hiro_low_final.zip",
            # ),
            # HIROEvalModel(
            #     name="uni_fixedHER_newEnv",
            #     model_dir="./models/hiro_260703_highonly_uniOld_fixedHER_newEnv_2to0",  # 用作 config/run_config 来源，任选一个同环境模型
            #     high_model_by_goal_lane={
            #         0: "./models/hiro_260703_highonly_uniOld_fixedHER_newEnv_2to0",
            #         1: "./models/hiro_260706_highonly_uniOld_fixedHER_newEnv_2to1",
            #         2: "./models/hiro_260630_highonly_pretrained_uniOld_fixedHER_newEnv_2to2",
            #     },
            #     low_model_path="./models/hiro_260627_lowonly_uni_oldEnv_fixedHER_SLmpc_noaugObs/hiro_low_final.zip",
            # ),
            # HIROEvalModel(
            #     name="uni_noHER_newEnv",
            #     model_dir="./models/hiro_260703_highonly_uniOld_noHER_newEnv_2to0",  # 用作 config/run_config 来源，任选一个同环境模型
            #     high_model_by_goal_lane={
            #         0: "./models/hiro_260703_highonly_uniOld_noHER_newEnv_2to0",
            #         1: "./models/hiro_260703_highonly_uniOld_noHER_newEnv_2to1",
            #         2: "./models/hiro_260703_highonly_uniOld_noHER_newEnv_2to2",
            #     },
            #     low_model_path="./models/hiro_260627_lowonly_uni_oldEnv_noHER_SLmpc_noaugObs/hiro_low_final.zip",
            # ),
            # HIROEvalModel(
            #     name="uni_rule_newEnv",
            #     model_dir="./models/hiro_260702_highonly_ruleReUni_newEnv_SLmpc_noaugObs_2to0",
            #     high_model_by_goal_lane={
            #         0: "./models/hiro_260702_highonly_ruleReUni_newEnv_SLmpc_noaugObs_2to0",
            #         1: "./models/hiro_260702_highonly_ruleReUni_newEnv_SLmpc_noaugObs_2to1",
            #         2: "./models/hiro_260702_highonly_ruleReUni_newEnv_SLmpc_noaugObs_2to2",
            #     },
            #     low_level_type="rule_based",
            # ),
            HIROEvalModel(
                name="reUni_fixedHER_newEnv",
                model_dir="./models/hiro_260708_highonly_reUni_oldLow_newEnv_2to0",  # 用作 config/run_config 来源，任选一个同环境模型
                high_model_by_goal_lane={
                    0: "./models/hiro_260708_highonly_reUni_oldLow_newEnv_2to0",
                    1: "./models/hiro_260708_highonly_reUni_oldLow_newEnv_2to1",
                    2: "./models/hiro_260708_highonly_reUni_oldLow_newEnv_2to2",
                },
                low_model_path="./models/hiro_260328_lowonly_reachablePretrainedV2_Rainbow_amax3_dmin15_10/hiro_low_final.zip",
            ),
        ],
        # seed_base=343,
        episodes=300,
        record_episodes=[i for i in range(1, 301)],
        record_trajectory_episodes=[i for i in range(1, 301)],
        # enable_rendering=False,
        # scenario_name="multi_lane",
        scenario_name="multi_lane_stop_to_int",
        shared_env_config_model_dir=None,
        use_each_model_env_config=True,
        config_overrides={
            "environment": {
                # "initial_lane_probs": None,
                # "initial_lane_id": "random",
                # "goal_lane_id": 2,
                # "behavior_lane_probs": [
                #     [0.6, 0.3, 0.1],
                #     [0.6, 0.3, 0.1],
                #     [0.4, 0.3, 0.3],
                # ],

                "initial_lane_id": 2,
                "goal_lane_id": "random",
                "goal_lane_probs": None,
                # # "goal_lane_probs": [0.5, 0.0, 0.5],

                "action": {
                    "acceleration_range": [-5.0, 5.0],
                    # "acceleration_range": [-2.0, 3.0],
                },
            },
            "hiro": {
                "high_goal_safety": {
                    "enabled": True,
                    # "enabled": False,
                    # "dynamic_feasible_lane_intervals": False,
                    # "infeasible_action_mode": "reroute",
                    # "infeasible_action_mode": "shield_penalty",
                    # "infeasible_action_penalty": 3.0,
                    "max_accel": 3.0,
                    "max_decel": 3.0,
                    # "front_dmin": 20.0,
                    # "lane_change_rear_dmin": 12.0,
                    "comfort_prior_enabled": False,
                },
            },
        },
    )

    # run_sac_batch(
    #     models=[
    #         SACEvalModel(
    #             name="sac_newEnv_by_lane",
    #             model_dir="./models/sac_260624_withPrior_2to0",  # 用作基础 config 来源，任选同环境模型
    #             model_name="best_model.zip",
    #             sac_model_by_goal_lane={
    #                 0: "./models/sac_260624_withPrior_2to0",
    #                 1: "./models/sac_260704_withPrior_2to1",
    #                 2: "./models/sac_260622_withPrior_2to2_noGoalReshape",
    #             },
    #         ),
    #     ],
    #     episodes=300,
    #     scenario_name="multi_lane_stop_to_int",
    #     use_each_model_env_config=True,
    #     config_overrides={
    #         "environment": {
    #             "initial_lane_id": 2,
    #             "goal_lane_id": "random",
    #             "goal_lane_probs": None,
    #         },
    #         "evaluation": {
    #             "deterministic": True,
    #         },
    #     },
    # )
