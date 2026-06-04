from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Union

from util.config_utils import (
    build_env_config,
    build_env_config_for_scenario,
    get_scenario_spec_from_specs,
)


# =========================
# Algorithm kwargs
# =========================

def get_ppo_kwargs(log_dir: str, seed: int) -> Dict[str, Any]:
    return dict(
        policy="MlpPolicy",
        device="cpu",
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.0,
    )


def get_sac_kwargs(log_dir: str, seed: int, level: str = "high") -> Dict[str, Any]:
    numerics_guard_cfg = dict(
        enabled=True,
        save_dir=log_dir,
        file_name="sac_non_finite_debug.csv",
        max_rows_per_event=8,
    )

    if level == "high":
        sac_kwargs = dict(
            policy="MlpPolicy",
            verbose=0,
            tensorboard_log=log_dir,
            seed=seed,
            # buffer_size=100_000,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            learning_rate=3e-4,
            learning_starts=2000,
            # learning_rate=1e-4,
            train_freq=(1, "step"),
            # train_freq=(4, "step"),
            gradient_steps=1,
            numerics_guard=numerics_guard_cfg,
        )
    elif level == "low":
        sac_kwargs = dict(
            policy="MlpPolicy",
            verbose=0,
            tensorboard_log=log_dir,
            seed=seed,
            # buffer_size=100_000,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            learning_rate=3e-4,
            train_freq=(1, "step"),
            gradient_steps=1,
            numerics_guard=numerics_guard_cfg,
        )
    else:
        sac_kwargs = dict(
            policy="MlpPolicy",
            verbose=0,
            tensorboard_log=log_dir,
            seed=seed,
            buffer_size=1_000_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            learning_rate=3e-4,
            train_freq=(1, "step"),
            gradient_steps=1,
            numerics_guard=numerics_guard_cfg,
        )
    return sac_kwargs


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
    "spawn_adjacent_cutin_front_gap": 15.0,
    "spawn_adjacent_cutin_back_gap": 5.0,
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

    # Ego
    "controlled_vehicles": 1,
    "ego_speed": 10.0,
    "ego_speed_range": None,
    "initial_lane_id": "random",
    "warmup_time": 100.0,
    "warmup_each_episode": False,
    # Defer long episode-start offset alignment into multiple lightweight env.step calls.
    "inter_episode_as_steps": False,
    "inter_episode_step_seconds": 0.0,
    "inter_episode_zero_obs": True,
    
    "ego_clear_radius": 20.0,
    # "ego_clear_radius": 10.0,
    # "ego_clear_radius": "auto",
    # "ego_clear_margin": 0.5,

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
    # "comfort_use_jerk": True,
    "comfort_use_jerk": False,
    "high_use_acc_only_comfort": True,
    "comfort_max_jerk": 5.0,
    "comfort_acc_weight": 1.0,
    "comfort_jerk_weight": 0.1,

    "lane_change_reward": -1.0,
    # "lane_change_reward": -0.5,
    
    # RuleBasedController compute_action strategy: "target_speed_lane" | "goal_x_accel" | "goal_x_accel_follow" | "idm_mobil"
    # "rule_based_compute_action_mode": "goal_x_accel",
    "rule_based_compute_action_mode": "goal_x_accel_follow",
    "rule_follow_mode_enabled": True,
    "rule_follow_enter_gap": 12.0,
    "rule_follow_release_gap": 15.0,
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
            "spawn_probability": 0.05,
            "initial_lane_id": 2,
            "goal_lane_id": 1,
        },
        "use_lane_slot_observation": False,
    },
    "multi_lane_stop_to_int": {
        "module": "scenarios.multi_lane_stop_to_int",
        "env_id": "multi-lane-stop-to-int-v0",
        "env_overrides": {
            "lanes_count": 3,
            "spawn_probability": 0.05,
            # "spawn_probability": 0.07,
            "behavior_lane_probs": [
                [0.6, 0.3, 0.1],
                [0.6, 0.3, 0.1],
                [0.4, 0.3, 0.3],
            ],
            "initial_lane_id": 2,
            "goal_lane_id": 1,
            "single_road_network": True,
            "intersection_length": 50.0,
            "movement_lanes": {
                "straight": [0, 1, 2],
            },
            "background_vehicle_respect_movement_lanes": False,
            "start_longitudinal": 0.0,
            "goal_longitudinal": 400.0,
            "punctual_time_window": [30.0, 40.0],
            "punctual_time_target": 35.0,
            "signal_plan": [
                {"straight": 63.0},
                {"left": 37.0},
            ],
            "enable_signal_virtual_stops": True,
            "signal_cycle_offset": 0.0,
            "align_ego_spawn_to_signal_offset": True,
            # "episode_start_phase_offset": 0.0,
            "episode_start_phase_offset": 20.0,
            "inter_episode_as_steps": True,
            "inter_episode_step_seconds": 0.1,
            "inter_episode_zero_obs": True,
            
            "spawn_check_adjacent_cutins": True,
            "spawn_adjacent_cutin_front_gap": 15.0,
            "spawn_adjacent_cutin_back_gap": 5.0,

            "use_lane_slot_observation": False,
            # "use_lane_slot_observation": True,
        },
    },
}


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
    """Return a full env config dict for MultiLaneEnv.

    This function performs a deep-merge, so nested keys like observation/action can be overridden partially
    without losing required defaults (e.g. vehicles_count/features/include_time).
    """
    return build_env_config(_MULTILANE_BASE_ENV_CONFIG, overrides)


# =========================
# HiRO centralized configs
# =========================

def get_hiro_config():
    """Centralized HiRO algorithm config."""
    from rl.algos.HRL.hiro import HIROConfig, LowSafetyFilterConfig
    from rl.algos.HRL.goal_samplers import GoalSamplerConfig

    # Intrinsic reward presets. (ego feature order in HIRO: x, y, vx, vy)
    ENABLE_HIRO_REWARD_SHAPING = True
    # ENABLE_HIRO_REWARD_SHAPING = False
    if ENABLE_HIRO_REWARD_SHAPING:
        intrinsic_type = "huber_shaping"
        intrinsic_coef = 10.0
        intrinsic_norm_ranges = [
            [0.0, 10.0],
            [-4.0, 4.0],
            [-10.0, 10.0],
            [-2.0, 2.0],
        ]
        intrinsic_weights = [1.0, 1.0, 0.0, 0.2]
        # intrinsic_weights = [1.0, 1.0, 0.0, 0.1]
    else:
        intrinsic_type = "l2"
        intrinsic_coef = 10.0
        # intrinsic_coef = 8.0
        intrinsic_norm_ranges = [
            [0.0, 37.5],
            [-8.0, 8.0],
            [-8.0, 8.0],
            [-2.0, 2.0],
        ]
        # intrinsic_weights = [1.0, 2.0, 0.0, 1.0]
        intrinsic_weights = [1.0, 2.0, 0.0, 0.3]
        # intrinsic_weights = [1.0, 2.0, 8.0, 1.0]

    return HIROConfig(
        high_interval=25,
        batch_size=256,
        gradient_steps_high=1,
        gradient_steps_low=1,
        train_freq=1,
        device="auto",

        # train_mode="joint",
        train_mode="high_only",
        # train_mode="low_only",

        intrinsic_coef=intrinsic_coef,
        intrinsic_norm_ranges=intrinsic_norm_ranges,
        intrinsic_weights=intrinsic_weights,
        intrinsic_type=intrinsic_type,

        goal_sampler=GoalSamplerConfig(
            type="uniform",
        ),
        # goal_sampler=GoalSamplerConfig(
        #     type="reachable_uniform",
        # ),
        # goal_sampler=GoalSamplerConfig(
        #     type="reachable_gaussian",
        #     gaussian_mean_x_m=27.0,
        #     gaussian_half_range_m=5.0,
        # ),
        # goal_sampler=GoalSamplerConfig(
        #     type="speed_near_cruise",
        # ),
        # goal_sampler=GoalSamplerConfig(
        #     type="pretrained",
        #     path="./models/hiro_test_260211_highonly_pretrained_vmin0/hiro_high_final.zip",
        #     device="auto",
        #     deterministic=False,
        # ),
        # goal_sampler=GoalSamplerConfig(
        #     type="fixed",
        #     action=[25.0, 0.0, 10.0],
        # ),

        low_level_type="rule_based",
        # low_level_type="sac",
        # low_sac_impl="sac",

        # low_use_her=False,
        low_use_her=True,
        low_her_ratio=0.8,
        low_her_strategy="future",

        # use_off_policy_correction=True,
        use_off_policy_correction=False,

        use_low_safety_layer=True,
        # use_low_safety_layer=False,

        use_high_goal_safety_layer=False,
        # use_high_goal_safety_layer=True,
        # high_goal_safe_eps=1e-6,

        high_goal_safe_use_custom_kinematics=True,
        high_goal_safe_max_accel=3.0,
        high_goal_safe_max_decel=3.0,
        high_goal_safe_front_dmin=15.0,
        high_goal_safe_lane_change_rear_dmin=10.0,
        high_goal_safe_min_goal_x_span=0,

        low_safety_violation_penalty=0.3,

        low_safety_filter=LowSafetyFilterConfig(
            type="mpc_constraints",
            lane_change_min_front_gap=15.0,
            lane_change_min_rear_gap=10.0,
            lane_change_min_front_ttc=3.0,
            lane_change_min_rear_ttc=2.0,
            # lane_change_min_front_ttc=0.0,
            # lane_change_min_rear_ttc=0.0,
        ),
        # low_safety_filter=LowSafetyFilterConfig(
        #     type="RSS",
        #     safe_gap_d_min=6.0,
        #     safe_gap_tau=0.6,
        #     safe_gap_b_ego=3.0,
        #     safe_gap_b_front=3.0,
        #     safe_gap_comfort_decel=-3.0,
        #     safe_gap_emergency_decel=-5.0,
        #     safe_gap_emergency_ttc=1.0,
        #     safe_gap_emergency_distance=10.0,
        # ),
        # low_safety_filter=LowSafetyFilterConfig(
        #     type="legacy",
        # ),
        # low_safety_filter=LowSafetyFilterConfig(
        #     type="legacy_mpc_max",
        #     lane_change_min_front_gap=15.0,
        #     lane_change_min_rear_gap=10.0,
        #     lane_change_min_front_ttc=3.0,
        #     lane_change_min_rear_ttc=2.0,
        # ),

        mask_ego_position_in_low_obs=True,
        fixed_goal_vx=0.0,
        # fixed_goal_vx=None,

        # high_obs_use_signal_features=False,
        high_obs_use_signal_features=True,
    )

def get_hiro_high_sac_kwargs(log_dir: str, seed: int) -> Dict[str, Any]:
    """Get SAC kwargs for HiRO high-level agent, including static buffer config."""
    kwargs = get_sac_kwargs(log_dir, seed, level="high")

    # Keep numerics guard CSV at run-level log dir (same level as high_interval_debug.csv).
    run_log_dir = os.path.dirname(log_dir) if os.path.basename(log_dir) == "hiro_high" else log_dir
    numerics_guard = dict(kwargs.get("numerics_guard", {}) or {})
    numerics_guard["save_dir"] = run_log_dir
    kwargs["numerics_guard"] = numerics_guard
    kwargs["q_replay_debug"] = dict(
        enabled=True,
        save_dir=run_log_dir,
        file_name="q_replay_debug.csv",
        target_q_lte=-20.0,
        next_q_lte=-20.0,
        max_rows_per_update=8,
        max_total_rows=200_000,
        period_updates=0,
        record_full_obs=True,
    )
    
    # Static config for HiROHighReplayBuffer
    kwargs["replay_buffer_kwargs"] = dict(
        n_candidates=20,
        noise_std=0.5,
        enable_off_policy_correction=True,
        # In this task, the 50s episode timeout is a real failure/terminal
        handle_timeout_termination=False,
    )
    return kwargs


def get_hiro_low_sac_kwargs(
    log_dir: str,
    seed: int,
    target_entropy: Union[str, float] = "auto",
    target_entropy_scale: Optional[float] = 0.5,
) -> Dict[str, Any]:
    """Get SAC kwargs for HiRO low-level agent.

    By default, we use a scaled auto target entropy: target_entropy = -target_entropy_scale * action_dim.
    The actual action_dim is only known inside HIRO at runtime, so the scaling is applied there.
    """
    kwargs = get_sac_kwargs(log_dir, seed, level="low")

    # Keep numerics guard CSV at run-level log dir (same level as high_interval_debug.csv).
    run_log_dir = os.path.dirname(log_dir) if os.path.basename(log_dir) == "hiro_low" else log_dir
    numerics_guard = dict(kwargs.get("numerics_guard", {}) or {})
    numerics_guard["save_dir"] = run_log_dir
    kwargs["numerics_guard"] = numerics_guard

    kwargs["target_entropy"] = target_entropy
    kwargs["target_entropy_scale"] = target_entropy_scale
    return kwargs
