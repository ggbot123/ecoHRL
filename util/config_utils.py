from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence


def deep_update(dst: Dict[str, Any], src: Mapping[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _drop_ignored_override_paths(
    overrides: Mapping[str, Any],
    ignored_paths: Sequence[str] | None,
) -> Dict[str, Any]:
    """Return overrides with scenario-specific unsupported keys removed."""
    filtered = deepcopy(dict(overrides))
    if not ignored_paths:
        return filtered

    for raw_path in ignored_paths:
        parts = [part for part in str(raw_path).split(".") if part]
        if not parts:
            continue
        cursor = filtered
        for part in parts[:-1]:
            next_cursor = cursor.get(part)
            if not isinstance(next_cursor, dict):
                cursor = None
                break
            cursor = next_cursor
        if isinstance(cursor, dict):
            cursor.pop(parts[-1], None)
    return filtered


def sync_lane_slot_observation_switch(cfg: Dict[str, Any]) -> None:
    """Optionally switch to fixed lane-slot kinematics without touching legacy mode."""
    if not bool(cfg.get("use_lane_slot_observation", False)):
        return

    obs_cfg = cfg.setdefault("observation", {})
    obs_cfg["type"] = "LaneSlotKinematics"
    obs_cfg["vehicles_count"] = 9
    obs_cfg["vehicles_count_local"] = 9
    obs_cfg.setdefault("lane_slot_front_range", 150.0)
    obs_cfg.setdefault("lane_slot_rear_range", 50.0)
    obs_cfg.setdefault("lane_slot_lateral_margin", 0.75)
    # LaneSlotKinematics observes real vehicles only by construction.
    obs_cfg["include_obstacles"] = False


def sync_goal_lane_observation_switch(cfg: Dict[str, Any]) -> None:
    """Expose the episode goal lane whenever random goals are configured."""
    obs_cfg = cfg.setdefault("observation", {})
    goal_lane = cfg.get("goal_lane_id", None)
    if isinstance(goal_lane, str) and goal_lane.strip().lower() == "random":
        obs_cfg["append_goal_lane_id"] = True
    else:
        obs_cfg.setdefault("append_goal_lane_id", False)


def sync_punctual_time_with_phase_offset(
    cfg: Dict[str, Any],
    phase_offset: float | None = None,
) -> None:
    """Derive the punctual target/window from a signal phase offset."""
    profile = cfg.get("punctual_time_offset_profile", None)
    if not isinstance(profile, Mapping) or not bool(profile.get("enabled", False)):
        return

    offset = float(
        cfg.get("episode_start_phase_offset", 0.0)
        if phase_offset is None
        else phase_offset
    )
    signal_plan = cfg.get("signal_plan", [])
    cycle = 0.0
    if isinstance(signal_plan, list):
        for phase in signal_plan:
            if not isinstance(phase, Mapping):
                continue
            for duration in phase.values():
                try:
                    cycle += float(duration)
                except (TypeError, ValueError):
                    continue
    if cycle > 1e-9:
        offset %= cycle

    left_end = float(profile.get("left_end", 3.0))
    low_plateau_end = float(profile.get("low_plateau_end", 25.0))
    high_plateau_end = float(profile.get("high_plateau_end", 30.0))
    low_level = float(profile.get("low_level", 35.0))
    high_level = float(profile.get("high_level", 75.0))
    shared_slope = float(profile.get("shared_slope", -0.55330067))
    window_length = max(float(profile.get("window_length", 10.0)), 0.0)

    if offset < left_end:
        target = low_level + shared_slope * (offset - left_end)
    elif offset < low_plateau_end:
        target = low_level
    elif offset < high_plateau_end:
        target = high_level
    else:
        target = high_level + shared_slope * (offset - high_plateau_end)

    half_window = 0.5 * window_length
    cfg["punctual_time_target"] = float(target)
    cfg["punctual_time_window"] = [
        float(target - half_window),
        float(target + half_window),
    ]


def get_scenario_spec_from_specs(
    scenario_specs: Mapping[str, Mapping[str, Any]],
    scenario_name: str,
) -> Dict[str, Any]:
    key = str(scenario_name).strip().lower()
    if key not in scenario_specs:
        supported = ", ".join(sorted(scenario_specs.keys()))
        raise ValueError(f"Unknown scenario '{scenario_name}'. Supported: {supported}")
    return deepcopy(scenario_specs[key])


def build_env_config(
    base_config: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = deepcopy(base_config)
    if overrides:
        deep_update(cfg, overrides)
    sync_lane_slot_observation_switch(cfg)
    sync_goal_lane_observation_switch(cfg)
    sync_punctual_time_with_phase_offset(cfg)
    return cfg


def build_env_config_for_scenario(
    base_config: Mapping[str, Any],
    scenario_specs: Mapping[str, Mapping[str, Any]],
    scenario_name: str,
    overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    spec = get_scenario_spec_from_specs(scenario_specs, scenario_name)
    merged_overrides: Dict[str, Any] = deepcopy(spec.get("env_overrides", {}))
    if overrides:
        filtered_overrides = _drop_ignored_override_paths(
            overrides,
            spec.get("ignored_env_override_keys", ()),
        )
        deep_update(merged_overrides, filtered_overrides)
    return build_env_config(base_config, merged_overrides)
