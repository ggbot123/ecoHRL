from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Optional


def _deep_update(dst: Dict[str, Any], src: Mapping[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, Mapping) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


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
    "initial_lane_id": "random",
    "warmup_time": 100.0,
    "warmup_each_episode": False,
    "ego_clear_radius": 10.0,

    # Observation / Action
    "PERCEPTION_DISTANCE": None,
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "vehicles_count_local": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-200, 200],
            "y": [-10, 10],
            "vx": [-15, 15],
            "vy": [-10, 10]
        },
        "normalize": False,
        "see_behind": False,
        "include_obstacles": False,
        "include_time": True,
        "time_range": [0.0, 50.0],
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

    # Reward weights (used by MultiLaneEnv._reward gating logic)
    "collision_reward": -10.0,
    "progress_reward": 10.0,
    "comfort_reward": 0.7,
    "comfort_max_accel": 3.0,
    "lane_change_reward": -0.5,

    # Termination
    "offroad_terminal": False,
}


def get_env_config(overrides: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Return a full env config dict for MultiLaneEnv.

    This function performs a deep-merge, so nested keys like observation/action can be overridden partially
    without losing required defaults (e.g. vehicles_count/features/include_time).
    """
    cfg = deepcopy(_MULTILANE_BASE_ENV_CONFIG)
    if overrides:
        _deep_update(cfg, overrides)
    return cfg


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
    )
    if level == "low":
        sac_kwargs["verbose"] = 0
    return sac_kwargs


# =========================
# HiRO centralized configs
# =========================

def get_hiro_replay_buffer_kwargs() -> Dict[str, Any]:
    """Centralized kwargs for the HiRO high-level replay buffer (OPC)."""
    return dict(
        n_candidates=10,
        noise_std=0.5,  # std in *scaled* action space [-1, 1]
        enable_off_policy_correction=True,
    )


def get_hiro_config():
    """Centralized HiRO algorithm config."""
    from rl.algos.HRL.hiro import HIROConfig

    intrinsic_norm_ranges = [
        [0.0, 37.5],
        [-8.0, 8.0],
        [-8.0, 8.0],
        [-2.0, 2.0],
    ]
    intrinsic_weights = [1.0, 2.0, 8.0, 1.0]

    return HIROConfig(
        high_interval=25,
        batch_size=256,
        gradient_steps_high=1,
        gradient_steps_low=1,
        train_freq=1,
        intrinsic_coef=8.0,
        device="auto",
        intrinsic_norm_ranges=intrinsic_norm_ranges,
        intrinsic_weights=intrinsic_weights,
        use_off_policy_correction=True,
    )
