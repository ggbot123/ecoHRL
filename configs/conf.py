from __future__ import annotations
from typing import Any, Dict


MASTER_SEED = 42

# =========================
# Basic algorithm kwargs
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
    },
}


# =========================
# Environment config
# =========================

# Base env config shared by scenario.py default_config and train.py
_MULTILANE_BASE_ENV_CONFIG: Dict[str, Any] = {
    # Basic
    "simulation_frequency": 10,
    "policy_frequency": 10,
    "warmup_time": 100.0,
    "warmup_each_episode": False,
    "inter_episode_as_steps": False,
    "inter_episode_step_seconds": 0.1,
    "inter_episode_zero_obs": True,

    # Optional fast reset path for signalized scenarios. When enabled by a
    # scenario, reset restores a pre-generated background traffic snapshot at
    # the requested signal offset, then inserts ego immediately.
    "background_snapshot_reset": False,
    "background_snapshot_path": None,
    "background_snapshot_paths": None,

    # Road
    "lanes_count": 3,
    "road_length": 500.0,
    "speed_limit": 15.0,

    # Traffic flow
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
    # Optional per-lane episode sampling probabilities. When set, this takes precedence over initial_lane_id and is normalized automatically.
    "initial_lane_probs": None,

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
        "append_goal_lane_id": False,
        "goal_lane_feature_encoding": "one_hot",  # "scalar" | "one_hot"
        "front_vehicle_distance_range": 150.0,
        "front_vehicle_ttc_range": 30.0,
    },
    "action": {
        "type": "ParamLaneAccelAction",
        "acceleration_range": [-3.0, 2.0],
        # "acceleration_range": [-5.0, 5.0],
        "lane_actions": ["KEEP", "LANE_LEFT", "LANE_RIGHT"],
    },

    # Termination
    "offroad_terminal": True,

    # Reward weights (used by MultiLaneEnv._reward gating logic)
    "collision_reward": -10.0,
    "punctual_reward": 10.0,
    "progress_reward": 10.0,
    "speed_ref_aux_reward": 0.0,
    "comfort_reward": 0.7,
    "comfort_max_accel": 3.0,
    "comfort_acc_weight": 1.0,
    "wrong_lane_terminal_penalty": 0,
    "wrong_lane_penalty_only_at_goal_longitudinal": False,

    # RuleBasedController compute_action strategy: "target_speed_lane" | "goal_x_accel" | "goal_x_accel_follow" | "idm_mobil"
    "rule_based_compute_action_mode": "goal_x_accel",
    # "rule_based_compute_action_mode": "goal_x_accel_follow",
    "rule_follow_enter_gap": 17.0,
    "rule_follow_release_gap": 20.0,
    "rule_follow_enter_ttc": 2.0,
    "rule_follow_release_ttc": 4.0,
    "rule_follow_max_acc": 0.0,
    "rule_follow_reset_on_high_interval": True,

    # Optional environment-owned queue controller. Scenarios that enable it may take over after ego has joined a stopped signal queue.
    "enable_queue_takeover": False,
    "queue_takeover_front_speed": 2.0,
    "queue_takeover_front_gap": 30.0,
    "queue_takeover_enter_steps": 3,
    "queue_takeover_release_x_margin": 3.0,
    "queue_takeover_desired_speed": 10.0,
    "queue_takeover_max_accel": 2.0,
    "queue_takeover_comfort_brake": 3.0,
    "queue_takeover_min_gap": 4.0,
    "queue_takeover_time_headway": 1.2,

    # SAC always reuses HIRO low-safety-filter lane-change constraints.
    "enable_sac_low_safety_filter": True,
}

_SCENARIO_SPECS: Dict[str, Dict[str, Any]] = {
    "multi_lane": {
        "module": "scenarios.multi_lane",
        "env_id": "multi-lane-custom-v0",
        "env_overrides": {
            "spawn_probability": 0.07,
            "goal_longitudinal": 400.0,
            "goal_lane_id": 2,
            "goal_lane_probs": None,
            "duration": 50.0,
            "punctual_time_window": [30.0, 40.0],
            "punctual_time_target": 35.0,
        },
    },
    "multi_lane_stop_to_int": {
        "module": "scenarios.multi_lane_stop_to_int",
        "env_id": "multi-lane-stop-to-int-v0",
        "env_overrides": {
            # Road & traffic
            "intersection_length": 50.0,
            "movement_lanes": {
                "straight": [0, 1, 2],
            },
            "spawn_check_adjacent_cutins": True,
            "spawn_adjacent_cutin_front_gap": 15.0,
            "spawn_adjacent_cutin_back_gap": 5.0,
            "background_vehicle_respect_movement_lanes": False,
            "enable_signal_green_launch_behavior": True,
            "signal_green_launch_approach_distance": 80,
            "signal_green_launch_end_margin": 5.0,
            "signal_green_launch_target_speed": None,
            "enable_signal_cycle_spawn_probability": False,
            "signal_cycle_spawn_probability": None,
            "enable_queue_takeover": True,
            # Task
            "start_longitudinal": 0.0,
            "goal_longitudinal": 400.0,
            "initial_lane_id": 2,
            "duration": 85.0,
            "punctual_time_window": [30.0, 40.0],
            "punctual_time_target": 35.0,
            "signal_plan": [
                {"straight": 63.0},
                {"left": 57.0},
            ],
            "align_ego_spawn_to_signal_offset": True,
            "inter_episode_as_steps": True,
            "episode_start_phase_offset": 20.0,     # early green pass
            "punctual_time_offset_profile": {
                "enabled": False,
                "left_end": 3.0,
                "low_plateau_end": 25.0,
                "high_plateau_end": 30.0,
                "low_level": 35.0,
                "high_level": 75.0,
                "shared_slope": -0.55330067,
                "window_length": 10.0,
            },
        },
    },
}


# =========================
# HiRO centralized configs
# =========================

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

_HIRO_HIGH_GOAL_SAFETY_CONFIG: Dict[str, Any] = {
    "enabled": False,
    # "enabled": True,
    "eps": 1e-6,
    "use_custom_kinematics": True,
    "max_accel": 2.0,
    "max_decel": 3.0,
    "front_dmin": 15.0,
    "lane_change_rear_dmin": 10.0,
    "min_goal_x_span": 0.0,
    "enable_goal_vx_bounds": False,
    "dynamic_feasible_lane_intervals": True,
}

_HIRO_HIGH_REPLAY_BUFFER_KWARGS: Dict[str, Any] = {
    "n_candidates": 20,
    "noise_std": 0.5,
    "enable_off_policy_correction": True,
}

_HIRO_CONFIG: Dict[str, Any] = {
    "high_interval": 25,
    "batch_size": 256,
    "gradient_steps_high": 1,
    "gradient_steps_low": 1,
    "train_freq": 1,
    "device": "auto",
    "use_off_policy_correction": False,
    "use_low_safety_layer": True,
    "reward_shaping_enabled": True,
    "high_obs_use_signal_features": True,

    # "train_mode": "joint",
    "train_mode": "high_only",
    # "train_mode": "low_only",

    "low_level_type": "rule_based",
    # "low_level_type": "sac",

    "low_use_her": True,
    # "low_use_her": False,
    "low_her_ratio": 0.6,
    "low_her_strategy": "future",
    "low_her_future_mode": "episode_timeaware",
    "low_her_episode_timeaware_steps_ahead_range": None,

    "low_safety_violation_penalty": 0.3,
    "mask_ego_position_in_low_obs": True,
    "fixed_goal_vx": 0.0,
    # "fixed_goal_vx": None,
}


# =========================
# Default train entry config
# =========================

TRAIN_CONFIG: Dict[str, Any] = {
    # "algo": "hiro",
    "algo": "sac",
    "log_root": "./logs/current",
    "save_root": "./models",
    # "total_timesteps": 5_000_000,
    "total_timesteps": 10_000_000,
    "eval_freq": 10_000,
    "save_freq": 50_000,
    "n_envs": 8,
    "render": False,

    # "run_name": "sac_260621_withPrior_2to2",
    # "run_name": "sac_260621_withPrior_2to0",
    # "run_name": "sac_260621_withPrior_2to0_noGoalReshape",
    "run_name": "sac_260621_withPrior_2to2_noGoalReshape",
    # "run_name": "hiro_260620_highonly_lateGreen_2to0_noGoalReshape",
    # "run_name": "hiro_260620_highonly_lateGreen_2to1_newReset_dyna_buf100k_randStart_slowLane1",
    # "run_name": "hiro_260618_highonly_lateGreen_2to2_newReset_largeBuf",
    # "run_name": "hiro_260619_highonly_lateGreen_2to2_newReset_acc2",
    # "run_name": "hiro_260619_highonly_lateGreen_2to0_newReset",
    # "run_name": "hiro_260616_highonly_lateGreen_2toR_newFlowProb",
    # "run_name": "hiro_260611_lowonly_noSigFeat_lateGreen_noHER",
    # "run_name": "hiro_260620_lowonly_uniform_randomStart_snapshot02_queue_lc05_fixedHER",

    # "scenario_name": "multi_lane",
    "scenario_name": "multi_lane_stop_to_int",

    # Train-time config overrides. Keep sections empty unless you want to override defaults.
    "config_overrides": {
        "environment": {
            "rule_based_compute_action_mode": "goal_x_accel_follow",
            # "rule_based_compute_action_mode": "goal_x_accel",
            "observation": {
                # "append_front_vehicle_features": False,
                "append_front_vehicle_features": True,
                "goal_lane_feature_encoding": "one_hot",
            },
            # "initial_lane_probs": None,
            # "initial_lane_id": "random",
            # "goal_lane_id": 0,
            # "goal_lane_id": 1,
            "goal_lane_id": 2,
            # "goal_lane_id": "random",
            "goal_lane_probs": None,
            # "behavior_lane_probs": [
            #     [0.4, 0.3, 0.3],
            #     [0.6, 0.3, 0.1],
            #     [0.6, 0.3, 0.1],
            # ],
            "behavior_lane_probs": [
                [0.6, 0.3, 0.1],
                [0.6, 0.3, 0.1],
                [0.4, 0.3, 0.3],
            ],
            # "behavior_lane_probs": [
            #     [0.6, 0.3, 0.1],
            #     [0.4, 0.3, 0.3],
            #     [0.6, 0.3, 0.1],
            # ],
            "signal_plan": [
                {"straight": 63.0},
                {"left": 57.0},
            ],
            "enable_signal_cycle_spawn_probability": True,
            "signal_cycle_spawn_probability": [
                {"start": 0.0, "end": 27.0, "spawn_probability": 0.07},
                {"start": 27.0, "end": 84.0, "spawn_probability": 0.03},
                {"start": 84.0, "end": 120.0, "spawn_probability": 0.07},
            ],
            "align_ego_spawn_to_signal_offset": True,
            "background_snapshot_reset": True,
            "background_snapshot_path": None,
            "background_snapshot_paths": [
                # "debug/background_snapshot_pool_slowlane0",
                "debug/background_snapshot_pool_slowlane2",
            ],
            "episode_start_phase_offset": 20.0,   # late green pass
            # "enable_queue_takeover": True,
            "enable_queue_takeover": False,
            "goal_lane_dense_reward": 0,
            # "goal_lane_dense_reward": 1.0,
            "lane_change_reward": -1.0,
            # "lane_change_reward": -0.5,
            "action": {
                "acceleration_range": [-3.0, 2.0],
                # "acceleration_range": [-5.0, 5.0],
            },
        },
        # SAC-only environment overrides, used only when algo="sac".
        "sac_environment": {
            "speed_ref_aux_reward": 0.1,
        },
        "hiro": {
            # "train_mode": "high_only",
            # "low_level_type": "rule_based",

            "train_mode": "low_only",
            "low_level_type": "sac",
            "goal_sampler": {
                "type": "uniform",
            },
            "low_use_her": True,
            "fixed_goal_vx": 0.0,
            "use_low_safety_layer": True,
            "high_goal_safety": {
                "dynamic_feasible_lane_intervals": True,  # 或 True
            },
        },
        "hiro_high_sac_kwargs": {
            # "buffer_size": 1_000_000,
            "buffer_size": 100_000,
        },
    },
    "sac_transition_csv_episode_freq": 1,
    "sac_transition_csv_envs": "env0",

    # Optional pretrained / implementation switches.
    "hiro_high_pretrained_path": None,
    "hiro_low_pretrained_path": None,
    "hiro_low_target_entropy": "auto",
    "hiro_low_target_entropy_scale": 1,

    # HIRO debug CSV switches.
    "hiro_high_transition_csv_all": True,
    "hiro_high_transition_csv_envs": "env0",
    "hiro_high_q_replay_debug": True,
    "hiro_low_transition_detail_csv": False,
    "hiro_low_her_debug_csv_interval_steps": 5_000,
    "hiro_low_her_debug_sample_prob": 0.001,
    "hiro_low_debug_summary_interval_steps": 10_000,

    # Train-time video recording.
    "record_video": False,
    # "record_video": True,
    "record_video_envs": "env0",
    "video_episode_freq": 20,
}
