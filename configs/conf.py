from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Union

from util.config_utils import (
    build_env_config,
    build_env_config_for_scenario,
    get_scenario_spec_from_specs,
)


MASTER_SEED = 42


# =========================
# Algorithm kwargs
# =========================

_PPO_KWARGS: Dict[str, Any] = {
    "policy": "MlpPolicy",
    "device": "cpu",
    "verbose": 1,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "n_epochs": 10,
    "clip_range": 0.2,
    "ent_coef": 0.0,
}

_SAC_NUMERICS_GUARD: Dict[str, Any] = {
    "enabled": False,
    "file_name": "sac_non_finite_debug.csv",
    "max_rows_per_event": 8,
}

_SAC_KWARGS_BY_LEVEL: Dict[str, Dict[str, Any]] = {
    "high": {
        "policy": "MlpPolicy",
        "verbose": 0,
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "learning_rate": 3e-4,
        "learning_starts": 2000,
        "train_freq": (1, "step"),
        "gradient_steps": 1,
    },
    "low": {
        "policy": "MlpPolicy",
        "verbose": 0,
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "learning_rate": 3e-4,
        "train_freq": (1, "step"),
        "gradient_steps": 1,
    },
    "default": {
        "policy": "MlpPolicy",
        "verbose": 0,
        "buffer_size": 1_000_000,
        "batch_size": 256,
        # "gamma": 0.9995,
        "gamma": 0.99,
        "tau": 0.005,
        "learning_rate": 3e-4,
        "train_freq": (1, "step"),
        "gradient_steps": 1,
        "replay_buffer_kwargs": {
            "handle_timeout_termination": True,
            # "handle_timeout_termination": False,
        },
    },
}


# =========================
# Environment config (MultiLaneEnv)
# =========================

# Base env config shared by scenario.py default_config and train.py
# Train-time overrides should be applied via get_env_config(overrides=...).
_MULTILANE_BASE_ENV_CONFIG: Dict[str, Any] = {
    # Basic
    "simulation_frequency": 10,
    "policy_frequency": 10,
    "duration": 50.0,
    "warmup_time": 100.0,
    "warmup_each_episode": False,
    "inter_episode_as_steps": False,
    "inter_episode_step_seconds": 0.1,
    "inter_episode_zero_obs": True,

    # Road
    "lanes_count": 3,
    "road_length": 500.0,
    "speed_limit": 15.0,

    # Traffic flow
    "spawn_probability": 0.07,
    "flow_speed_range": [10.0, 10.0],
    "speed_distribution": "Uniform",
    "spawn_min_gap": 10.0,
    "spawn_min_t_headway": 1.5,
    "spawn_check_adjacent_cutins": False,
    "behavior_vehicle_types": [
        "custom_env.vehicle.behavior.NormalIDMVehicle",
        "custom_env.vehicle.behavior.AggressiveIDMVehicle",
        "custom_env.vehicle.behavior.DefensiveIDMVehicle",
    ],
    "behavior_probs": [0.4, 0.3, 0.3],
    "behavior_lane_probs": [
        [0.6, 0.3, 0.1],
        [0.6, 0.3, 0.1],
        [0.4, 0.3, 0.3],
    ],
    "vid": 0,
    "ego_clear_radius": 20.0,
    # "ego_clear_radius": 10.0,
    # "ego_clear_radius": "auto",
    # "ego_clear_margin": 0.5,

    # Ego
    "controlled_vehicles": 1,
    "ego_speed": 10.0,
    "ego_speed_range": None,
    "initial_lane_id": "random",

    # Observation / Action
    "PERCEPTION_DISTANCE": None,
    "use_lane_slot_observation": False,
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "vehicles_count_local": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-200, 200],
            "y": [-10, 10],
            "vx": [-15, 15],
            "vy": [-10, 10],
        },
        "normalize": False,
        "see_behind": False,
        "include_obstacles": False,
        "include_time": True,
        "time_range": [0.0, 50.0],
        "append_front_vehicle_features": True,
        "front_vehicle_distance_range": 150.0,
        "front_vehicle_ttc_range": 30.0,
    },
    "action": {
        "type": "ParamLaneAccelAction",
        "acceleration_range": [-5.0, 5.0],
        "lane_actions": ["KEEP", "LANE_LEFT", "LANE_RIGHT"],
    },

    # Task / goal
    "goal_longitudinal": 400.0,
    "goal_lane_id": 2,
    "punctual_time_window": [30.0, 40.0],
    "punctual_time_target": 35.0,
    "punctual_reward": 10.0,
    # "wrong_lane_terminal_penalty": -5.0,
    "wrong_lane_terminal_penalty": 0,

    # Termination
    "offroad_terminal": False,

    # Reward weights (used by MultiLaneEnv._reward gating logic)
    "collision_reward": -10.0,
    "progress_reward": 10.0,
    "speed_ref_aux_reward": 0.0,

    "comfort_reward": 0.7,
    "comfort_max_accel": 3.0,
    "comfort_use_jerk": False,
    "high_use_acc_only_comfort": True,
    "comfort_max_jerk": 5.0,
    "comfort_acc_weight": 1.0,
    "comfort_jerk_weight": 0.1,
    
    # RuleBasedController compute_action strategy: "target_speed_lane" | "goal_x_accel" | "goal_x_accel_follow" | "idm_mobil"
    # "rule_based_compute_action_mode": "goal_x_accel",
    "rule_based_compute_action_mode": "goal_x_accel_follow",
    "rule_follow_mode_enabled": True,
    "rule_follow_enter_gap": 17.0,
    "rule_follow_release_gap": 20.0,
    "rule_follow_enter_ttc": 2.0,
    "rule_follow_release_ttc": 4.0,
    "rule_follow_max_acc": 0.0,
    "rule_follow_reset_on_high_interval": True,

    # SAC can optionally reuse HIRO low-safety-filter lane-change constraints.
    "enable_sac_low_safety_filter": True,
}


_SCENARIO_SPECS: Dict[str, Dict[str, Any]] = {
    "multi_lane": {
        "module": "scenarios.multi_lane",
        "env_id": "multi-lane-custom-v0",
        "env_overrides": {
            "spawn_probability": 0.07,
            "behavior_lane_probs": [
                [0.6, 0.3, 0.1],
                [0.6, 0.3, 0.1],
                [0.4, 0.3, 0.3],
            ],
            # Task
            # "initial_lane_id": 2,
            # "initial_lane_id": "1",
            # "goal_lane_id": 1,
            "goal_lane_id": 2,
            "lane_change_reward": -1.0,
            # "lane_change_reward": -0.5,
            # "rule_based_compute_action_mode": "goal_x_accel",
            "rule_based_compute_action_mode": "goal_x_accel_follow",
            "observation": {
                "append_front_vehicle_features": False,
            }
        },
    },
    "multi_lane_stop_to_int": {
        "module": "scenarios.multi_lane_stop_to_int",
        "env_id": "multi-lane-stop-to-int-v0",
        "env_overrides": {
            # Road & traffic
            "lanes_count": 3,
            "spawn_probability": 0.05,
            "behavior_lane_probs": [
                [0.6, 0.3, 0.1],
                [0.6, 0.3, 0.1],
                [0.4, 0.3, 0.3],
                # [0.6, 0.3, 0.1],
            ],
            "single_road_network": True,
            "intersection_length": 50.0,
            "movement_lanes": {
                "straight": [0, 1, 2],
            },
            "background_vehicle_respect_movement_lanes": False,
            "enable_signal_virtual_stops": True,
            "spawn_check_adjacent_cutins": True,
            "spawn_adjacent_cutin_front_gap": 15.0,
            "spawn_adjacent_cutin_back_gap": 5.0,
            # Task
            "start_longitudinal": 0.0,
            "goal_longitudinal": 400.0,
            "duration": 85.0,
            "punctual_time_window": [30.0, 40.0],
            "punctual_time_target": 35.0,
            "signal_plan": [
                {"straight": 63.0},
                {"left": 37.0},
            ],
            "signal_cycle_offset": 0.0,
            # "align_ego_spawn_to_signal_offset": True,
            "align_ego_spawn_to_signal_offset": False,
            "inter_episode_as_steps": True,
            # "episode_start_phase_offset": 20.0,   # late green pass
            # "episode_start_phase_offset": 90.0,     # mid green pass
            "episode_start_phase_offset": 40.0,     # early green pass
            "punctual_time_offset_profile": {
                "enabled": True,
                "left_end": 3.0,
                "low_plateau_end": 25.0,
                "high_plateau_end": 30.0,
                "low_level": 35.0,
                "high_level": 75.0,
                "shared_slope": -0.55330067,
                "window_length": 10.0,
            },
            "initial_lane_id": 2,
            "goal_lane_id": 1,
            "lane_change_reward": -1.0,
            # "lane_change_reward": -0.5,
            # "rule_based_compute_action_mode": "goal_x_accel",
            "rule_based_compute_action_mode": "goal_x_accel_follow",
        },
    },
}


# =========================
# HiRO centralized configs
# =========================

_HIRO_REWARD_SHAPING_ENABLED = True

_HIRO_INTRINSIC_PRESETS: Dict[str, Dict[str, Any]] = {
    "huber_shaping": {
        "intrinsic_type": "huber_shaping",
        "intrinsic_coef": 10.0,
        "intrinsic_norm_ranges": [
            [0.0, 10.0],
            [-4.0, 4.0],
            [-10.0, 10.0],
            [-2.0, 2.0],
        ],
        "intrinsic_weights": [1.0, 1.0, 0.0, 0.2],
    },
    "l2": {
        "intrinsic_type": "l2",
        "intrinsic_coef": 10.0,
        "intrinsic_norm_ranges": [
            [0.0, 37.5],
            [-8.0, 8.0],
            [-8.0, 8.0],
            [-2.0, 2.0],
        ],
        "intrinsic_weights": [1.0, 2.0, 0.0, 0.3],
    },
}


_HIRO_GOAL_SAMPLER_CONFIG: Dict[str, Any] = {
    "type": "uniform",
    # "type": "reachable_uniform",
    # "type": "reachable_gaussian",
    # "gaussian_mean_x_m": 27.0,
    # "gaussian_half_range_m": 5.0,
    # "type": "speed_near_cruise",
    # "type": "pretrained",
    # "path": "./models/hiro_test_260211_highonly_pretrained_vmin0/hiro_high_final.zip",
    # "device": "auto",
    # "deterministic": False,
    # "type": "fixed",
    # "action": [25.0, 0.0, 10.0],
}

_HIRO_LOW_SAFETY_FILTER_CONFIG: Dict[str, Any] | None = {
    "type": "mpc_constraints",
    "lane_change_min_front_gap": 15.0,
    "lane_change_min_rear_gap": 10.0,
    "lane_change_min_front_ttc": 3.0,
    "lane_change_min_rear_ttc": 2.0,
    # "type": "RSS",
    # "safe_gap_d_min": 6.0,
    # "safe_gap_tau": 0.6,
    # "safe_gap_b_ego": 3.0,
    # "safe_gap_b_front": 3.0,
    # "safe_gap_comfort_decel": -3.0,
    # "safe_gap_emergency_decel": -5.0,
    # "safe_gap_emergency_ttc": 1.0,
    # "safe_gap_emergency_distance": 10.0,
    # "type": "legacy",
    # "type": "legacy_mpc_max",
}

_HIRO_CONFIG: Dict[str, Any] = {
    "high_interval": 25,
    "batch_size": 256,
    "gradient_steps_high": 1,
    "gradient_steps_low": 1,
    "train_freq": 1,
    "device": "auto",

    # "train_mode": "joint",
    "train_mode": "high_only",
    # "train_mode": "low_only",

    "low_level_type": "rule_based",
    # "low_level_type": "sac",
    # "low_sac_impl": "sac",

    # "low_use_her": False,
    "low_use_her": True,
    "low_her_ratio": 0.8,
    "low_her_strategy": "future",

    # "use_off_policy_correction": True,
    "use_off_policy_correction": False,

    "use_low_safety_layer": True,
    # "use_low_safety_layer": False,

    "use_high_goal_safety_layer": False,
    # "use_high_goal_safety_layer": True,

    "high_goal_safe_use_custom_kinematics": True,
    "high_goal_safe_max_accel": 3.0,
    "high_goal_safe_max_decel": 3.0,
    "high_goal_safe_front_dmin": 15.0,
    "high_goal_safe_lane_change_rear_dmin": 10.0,
    "high_goal_safe_min_goal_x_span": 0,

    "low_safety_violation_penalty": 0.3,

    "mask_ego_position_in_low_obs": True,
    "fixed_goal_vx": 0.0,
    # "fixed_goal_vx": None,

    # "high_obs_use_signal_features": False,
    "high_obs_use_signal_features": True,
}

_HIRO_HIGH_Q_REPLAY_DEBUG: Dict[str, Any] = {
    "enabled": True,
    "file_name": "q_replay_debug.csv",
    "target_q_lte": -20.0,
    "next_q_lte": -20.0,
    "max_rows_per_update": 8,
    "max_total_rows": 200_000,
    "period_updates": 0,
    "record_full_obs": True,
}

_HIRO_HIGH_REPLAY_BUFFER_KWARGS: Dict[str, Any] = {
    "n_candidates": 20,
    "noise_std": 0.5,
    "enable_off_policy_correction": True,
    "handle_timeout_termination": False,
}


# =========================
# Default train entry config
# =========================

TRAIN_CONFIG: Dict[str, Any] = {
    # "algo": "hiro",
    "algo": "sac",
    "log_root": "./logs/current",
    "save_root": "./models",
    "total_timesteps": 5_000_000,
    # "total_timesteps": 10_000_000,
    "eval_freq": 10_000,
    "save_freq": 50_000,
    "n_envs": 8,
    "render": False,

    # "run_name": "hiro_260607_highonly_ruleFollow_sigFeat_earlyGreen",
    # "run_name": "hiro_260607_highonly_ruleFollow_sigFeat_midGreen",
    # "run_name": "hiro_260607_highonly_ruleFollow_sigFeat_varOffset",
    # "run_name": "hiro_260604_sac_withPrior_oldEnv_fixTimeout",
    "run_name": "hiro_260608_sac_base_oldEnv_test",
    "scenario_name": "multi_lane",
    # "scenario_name": "multi_lane_stop_to_int",

    # Train-time env overrides. Keep empty unless you want to override scenario defaults.
    "env_overrides": {
    },

    # SAC-specific env overrides used only when algo="sac".
    "sac_env_overrides": {
        # "speed_ref_aux_reward": 0.1,
        "speed_ref_aux_reward": 0,
    },
    # 0 disables SAC transition/episode CSV logging; N records every Nth episode.
    "sac_transition_csv_episode_freq": 1,
    "sac_transition_csv_envs": "env0",

    # Optional pretrained / implementation switches.
    "hiro_high_pretrained_path": None,
    "hiro_low_pretrained_path": None,
    "hiro_low_target_entropy": "auto",
    "hiro_low_target_entropy_scale": 1,
    "hiro_low_sac_impl": None,

    # HIRO debug CSV switches.
    "hiro_high_transition_csv_all": True,
    "hiro_high_transition_csv_envs": "all",
    "hiro_low_transition_detail_csv": False,
    "hiro_low_transition_detail_envs": "env0",

    # Train-time video recording.
    "record_video": True,
    "record_video_envs": "env0",
    "record_video_global_view": True,
    "video_episode_freq": 20,
    "record_video_collision_episodes": False,
    "record_video_collision_envs": "all",
}


# =========================
# Config builders
# =========================

def get_ppo_kwargs(log_dir: str, seed: int) -> Dict[str, Any]:
    kwargs = dict(_PPO_KWARGS)
    kwargs.update(tensorboard_log=log_dir, seed=seed)
    return kwargs


def get_sac_kwargs(log_dir: str, seed: int, level: str = "default") -> Dict[str, Any]:
    key = str(level).strip().lower()
    kwargs = dict(_SAC_KWARGS_BY_LEVEL.get(key, _SAC_KWARGS_BY_LEVEL["default"]))
    numerics_guard_cfg = dict(_SAC_NUMERICS_GUARD)
    numerics_guard_cfg["save_dir"] = log_dir
    kwargs.update(tensorboard_log=log_dir, seed=seed, numerics_guard=numerics_guard_cfg)
    return kwargs


def get_scenario_spec(scenario_name: str) -> Dict[str, Any]:
    return get_scenario_spec_from_specs(_SCENARIO_SPECS, scenario_name)


def get_env_config_for_scenario(
    scenario_name: str,
    overrides: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    return build_env_config_for_scenario(
        _MULTILANE_BASE_ENV_CONFIG,
        _SCENARIO_SPECS,
        scenario_name,
        overrides,
    )


def get_env_config(overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Return a full env config dict for MultiLaneEnv."""
    return build_env_config(_MULTILANE_BASE_ENV_CONFIG, overrides)


def get_hiro_config():
    """Centralized HiRO algorithm config."""
    from rl.algos.HRL.hiro import HIROConfig, LowSafetyFilterConfig
    from rl.algos.HRL.goal_samplers import GoalSamplerConfig

    intrinsic = dict(_HIRO_INTRINSIC_PRESETS["huber_shaping" if _HIRO_REWARD_SHAPING_ENABLED else "l2"])
    kwargs = dict(_HIRO_CONFIG)
    kwargs.update(intrinsic)
    kwargs["goal_sampler"] = GoalSamplerConfig(**_HIRO_GOAL_SAMPLER_CONFIG)
    kwargs["low_safety_filter"] = (
        LowSafetyFilterConfig(**_HIRO_LOW_SAFETY_FILTER_CONFIG)
        if _HIRO_LOW_SAFETY_FILTER_CONFIG is not None
        else None
    )
    return HIROConfig(**kwargs)


def get_hiro_high_sac_kwargs(log_dir: str, seed: int) -> Dict[str, Any]:
    """Get SAC kwargs for HiRO high-level agent, including static buffer config."""
    kwargs = get_sac_kwargs(log_dir, seed, level="high")
    run_log_dir = os.path.dirname(log_dir) if os.path.basename(log_dir) == "hiro_high" else log_dir

    numerics_guard = dict(kwargs.get("numerics_guard", {}) or {})
    numerics_guard["save_dir"] = run_log_dir
    kwargs["numerics_guard"] = numerics_guard

    q_replay_debug = dict(_HIRO_HIGH_Q_REPLAY_DEBUG)
    q_replay_debug["save_dir"] = run_log_dir
    kwargs["q_replay_debug"] = q_replay_debug
    kwargs["replay_buffer_kwargs"] = dict(_HIRO_HIGH_REPLAY_BUFFER_KWARGS)
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
