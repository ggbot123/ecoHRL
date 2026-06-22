from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Union

from configs.conf import (
    _HIRO_CONFIG,
    _HIRO_GOAL_SAMPLER_CONFIG,
    _HIRO_HIGH_GOAL_SAFETY_CONFIG,
    _HIRO_HIGH_REPLAY_BUFFER_KWARGS,
    _HIRO_INTRINSIC_PRESETS,
    _HIRO_LOW_SAFETY_FILTER_CONFIG,
    _MULTILANE_BASE_ENV_CONFIG,
    _PPO_KWARGS,
    _SAC_KWARGS_BY_LEVEL,
    _SAC_NUMERICS_GUARD,
    _SCENARIO_SPECS,
)
from util.config_utils import (
    build_env_config,
    build_env_config_for_scenario,
    get_scenario_spec_from_specs,
)


def get_ppo_kwargs(log_dir: str, seed: int) -> Dict[str, Any]:
    kwargs = dict(_PPO_KWARGS)
    kwargs.update(tensorboard_log=log_dir, seed=seed)
    return kwargs


def get_sac_kwargs(log_dir: str, seed: int, level: str = "default") -> Dict[str, Any]:
    key = str(level).strip().lower()
    kwargs = dict(_SAC_KWARGS_BY_LEVEL.get(key, _SAC_KWARGS_BY_LEVEL["default"]))
    rb_kwargs = dict(kwargs.get("replay_buffer_kwargs", {}) or {})
    rb_kwargs["handle_timeout_termination"] = False
    kwargs["replay_buffer_kwargs"] = rb_kwargs
    numerics_guard_cfg = dict(_SAC_NUMERICS_GUARD)
    numerics_guard_cfg["save_dir"] = log_dir
    kwargs.update(tensorboard_log=log_dir, seed=seed, numerics_guard=numerics_guard_cfg)
    return kwargs


def get_scenario_spec(scenario_name: str) -> Dict[str, Any]:
    return get_scenario_spec_from_specs(_SCENARIO_SPECS, scenario_name)


def _normalize_rule_follow_mode(config: Dict[str, Any]) -> Dict[str, Any]:
    mode = str(config.get("rule_based_compute_action_mode", "")).lower().strip()
    config["rule_follow_mode_enabled"] = mode == "goal_x_accel_follow"
    return config


def _apply_fixed_env_config(config: Dict[str, Any]) -> Dict[str, Any]:
    config.update(
        {
            "offroad_terminal": True,
            "single_road_network": True,
            "enable_signal_virtual_stops": True,
            "enable_sac_low_safety_filter": True,
            "signal_cycle_offset": 0.0,
        }
    )
    return config


def get_env_config_for_scenario(
    scenario_name: str,
    overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return _apply_fixed_env_config(
        _normalize_rule_follow_mode(
            build_env_config_for_scenario(
                _MULTILANE_BASE_ENV_CONFIG,
                _SCENARIO_SPECS,
                scenario_name,
                overrides,
            )
        )
    )


def get_env_config(overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Return a full env config dict for MultiLaneEnv."""
    return _apply_fixed_env_config(
        _normalize_rule_follow_mode(
            build_env_config(_MULTILANE_BASE_ENV_CONFIG, overrides)
        )
    )


def get_hiro_config():
    """Centralized HiRO algorithm config."""
    from rl.algos.HRL.goal_samplers import GoalSamplerConfig
    from rl.algos.HRL.hiro import HIROConfig, HighGoalSafetyConfig, LowSafetyFilterConfig

    kwargs = dict(_HIRO_CONFIG)
    reward_shaping_enabled = bool(kwargs.pop("reward_shaping_enabled", True))
    intrinsic = dict(_HIRO_INTRINSIC_PRESETS["huber_shaping" if reward_shaping_enabled else "l2"])
    kwargs.update(intrinsic)
    kwargs["goal_sampler"] = GoalSamplerConfig(**_HIRO_GOAL_SAMPLER_CONFIG)
    kwargs["low_safety_filter"] = (
        LowSafetyFilterConfig(**_HIRO_LOW_SAFETY_FILTER_CONFIG)
        if _HIRO_LOW_SAFETY_FILTER_CONFIG is not None
        else None
    )
    kwargs["high_goal_safety"] = HighGoalSafetyConfig(**_HIRO_HIGH_GOAL_SAFETY_CONFIG)
    return HIROConfig(**kwargs)


def get_hiro_high_sac_kwargs(
    log_dir: str,
    seed: int,
    q_replay_debug_enabled: bool = True,
) -> Dict[str, Any]:
    """Get SAC kwargs for HiRO high-level agent, including static buffer config."""
    kwargs = get_sac_kwargs(log_dir, seed, level="high")
    run_log_dir = os.path.dirname(log_dir) if os.path.basename(log_dir) == "hiro_high" else log_dir

    numerics_guard = dict(kwargs.get("numerics_guard", {}) or {})
    numerics_guard["save_dir"] = run_log_dir
    kwargs["numerics_guard"] = numerics_guard

    q_replay_debug = {
        "file_name": "q_replay_debug.csv",
        "target_q_lte": -20.0,
        "next_q_lte": -20.0,
        "max_rows_per_update": 8,
        "max_total_rows": 200_000,
        "period_updates": 0,
        "record_full_obs": True,
        "enabled": bool(q_replay_debug_enabled),
    }
    q_replay_debug["save_dir"] = run_log_dir
    kwargs["q_replay_debug"] = q_replay_debug
    replay_buffer_kwargs = dict(_HIRO_HIGH_REPLAY_BUFFER_KWARGS)
    replay_buffer_kwargs["handle_timeout_termination"] = False
    kwargs["replay_buffer_kwargs"] = replay_buffer_kwargs
    return kwargs


def get_hiro_low_sac_kwargs(
    log_dir: str,
    seed: int,
    target_entropy: Union[str, float] = "auto",
    target_entropy_scale: Optional[float] = 0.5,
) -> Dict[str, Any]:
    """Get SAC kwargs for HiRO low-level agent."""
    kwargs = get_sac_kwargs(log_dir, seed, level="low")
    run_log_dir = os.path.dirname(log_dir) if os.path.basename(log_dir) == "hiro_low" else log_dir

    numerics_guard = dict(kwargs.get("numerics_guard", {}) or {})
    numerics_guard["save_dir"] = run_log_dir
    kwargs["numerics_guard"] = numerics_guard
    kwargs["target_entropy"] = target_entropy
    kwargs["target_entropy_scale"] = target_entropy_scale
    return kwargs
