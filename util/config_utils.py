from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping


def deep_update(dst: Dict[str, Any], src: Mapping[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def sync_observation_with_comfort_switch(cfg: Dict[str, Any]) -> None:
    """Keep observation features consistent with comfort_use_jerk."""
    use_jerk = bool(cfg.get("comfort_use_jerk", False))
    obs_cfg = cfg.setdefault("observation", {})
    features = list(obs_cfg.get("features", []))
    features_range = dict(obs_cfg.get("features_range", {}))

    if use_jerk:
        if "acceleration" not in features:
            features.append("acceleration")
        features_range.setdefault("acceleration", [-5.0, 5.0])
    else:
        features = [f for f in features if f != "acceleration"]
        features_range.pop("acceleration", None)

    obs_cfg["features"] = features
    obs_cfg["features_range"] = features_range


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
    sync_observation_with_comfort_switch(cfg)
    sync_lane_slot_observation_switch(cfg)
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
        deep_update(merged_overrides, overrides)
    return build_env_config(base_config, merged_overrides)
