from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from configs.builders import get_env_config
from configs.builders import get_hiro_config
from rl.algos.HRL.high_goal_safe_bounds import HighGoalSafeBoundsCalculator
from rl.algos.sac.sac import SAC
from rl.utils.utils import goal_action_to_abs
from util.plot_result import render_high_interval_debug_snapshot


# Edit these paths directly when you want to switch test data/output location.
CSV_PATH = Path(r"d:\workspace\python\ecoHRL\logs\current\hiro_260322_highonly_reachableUniform_newSLv2_vio03_HER_reDim_lc10_amax3_dmin10_8\high_interval_debug.csv")
DEFAULT_OUT_DIR = Path(r"d:\workspace\python\ecoHRL\debug\test_high")
ROW_N = 5064
HIGH_MODEL_PATH = Path(r"d:\workspace\python\ecoHRL\models\hiro_260322_highonly_reachableUniform_newSLv2_vio03_HER_reDim_lc10_amax3_dmin10_8\hiro_high_final.zip")
# USE_STOCHASTIC = False
USE_STOCHASTIC = True
STOCHASTIC_SAMPLES = 100


def _parse_json_array(raw: str) -> np.ndarray:
    text = str(raw).strip()
    if not text:
        return np.asarray([], dtype=np.float32)
    try:
        obj = json.loads(text)
        return np.asarray(obj, dtype=np.float32)
    except Exception:
        if text.startswith("[") and text.endswith("]"):
            return np.fromstring(text[1:-1], sep=",", dtype=np.float32)
        return np.asarray([], dtype=np.float32)


def _load_csv_row(csv_path: Path, row_n: int) -> dict[str, str]:
    if row_n <= 0:
        raise ValueError(f"row_n must be >= 1, got {row_n}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            if i == row_n:
                return row

    raise IndexError(f"row_n={row_n} is out of range for file: {csv_path}")


def _extract_ego_sub(row: dict[str, Any]) -> np.ndarray:
    ego_sub = _parse_json_array(str(row.get("ego_sub", ""))).reshape(-1)
    if ego_sub.size >= 4:
        return ego_sub[:4].astype(np.float32)

    kin = _parse_json_array(str(row.get("kin", "")))
    if kin.ndim == 2 and kin.shape[0] >= 1 and kin.shape[1] >= 5:
        # kin row format: [presence, x, y, vx, vy]
        return np.asarray([kin[0, 1], kin[0, 2], kin[0, 3], kin[0, 4]], dtype=np.float32)

    raise ValueError("Cannot extract ego_sub from row: missing valid ego_sub and kin columns")


def _lane_centers_from_env_config() -> np.ndarray:
    env_cfg = get_env_config()
    n_lanes = int(env_cfg.get("lanes_count", 3))
    lane_w = float(env_cfg.get("lane_width", 4.0))
    return (np.arange(n_lanes, dtype=np.float32) * lane_w).astype(np.float32)


def _bind_high_goal_safe_bounds_if_needed(high_model: SAC, high_obs: np.ndarray) -> None:
    actor = getattr(high_model, "actor", None)
    if actor is None or not hasattr(actor, "goal_safe_sampling_enabled"):
        return

    if not bool(getattr(actor, "goal_safe_sampling_enabled", False)):
        return

    if getattr(actor, "goal_safe_bounds_fn", None) is not None:
        return

    try:
        env_cfg = get_env_config()
        hiro_cfg = get_hiro_config()

        n_lanes = int(env_cfg.get("lanes_count", 3))
        lane_w = float(env_cfg.get("lane_width", 4.0))
        speed_limit = float(env_cfg.get("speed_limit", 15.0))
        policy_freq = float(env_cfg.get("policy_frequency", 10.0))
        dt = 1.0 / max(policy_freq, 1e-6)
        hi = int(getattr(hiro_cfg, "high_interval", 25))
        horizon_t = float(hi) * float(dt)
        dx_low = 0.0
        dx_high = float(speed_limit * horizon_t)

        obs_features = list(env_cfg.get("observation", {}).get("features", ["presence", "x", "y", "vx", "vy"]))

        def _fidx(name: str, default: int) -> int:
            try:
                return int(obs_features.index(name))
            except ValueError:
                return int(default)

        action_cfg = env_cfg.get("action", {})
        accel_range = action_cfg.get("acceleration_range", [-5.0, 5.0])
        default_max_accel = float(max(abs(float(accel_range[0])), abs(float(accel_range[1]))))
        use_custom_kin = bool(getattr(hiro_cfg, "high_goal_safe_use_custom_kinematics", False))
        if use_custom_kin:
            cfg_max_accel = getattr(hiro_cfg, "high_goal_safe_max_accel", None)
            cfg_max_decel = getattr(hiro_cfg, "high_goal_safe_max_decel", None)
            max_accel = float(default_max_accel if cfg_max_accel is None else max(float(cfg_max_accel), 0.0))
            max_decel = float(default_max_accel if cfg_max_decel is None else max(float(cfg_max_decel), 0.0))
        else:
            max_accel = float(default_max_accel)
            max_decel = float(default_max_accel)

        feat_dim = int(len(obs_features))
        calc = HighGoalSafeBoundsCalculator(
            n_lanes=n_lanes,
            lane_width=lane_w,
            high_interval=hi,
            dt=dt,
            speed_min=0.0,
            speed_max=speed_limit,
            max_accel=max_accel,
            max_decel=max_decel,
            front_dmin=float(max(0.0, getattr(hiro_cfg, "high_goal_safe_front_dmin", 0.0))),
            lane_change_rear_dmin=float(max(0.0, getattr(hiro_cfg, "high_goal_safe_lane_change_rear_dmin", 0.0))),
            dx_low=dx_low,
            dx_high=dx_high,
            feat_dim=feat_dim,
            presence_idx=int(_fidx("presence", 0)),
            x_idx=int(_fidx("x", 1)),
            y_idx=int(_fidx("y", 2)),
            vx_idx=int(_fidx("vx", 3)),
            vy_idx=int(_fidx("vy", 4)),
        )

        # Warm-check once with current sample to fail fast on malformed layouts.
        _ = calc.compute_np(np.asarray(high_obs, dtype=np.float32).reshape(1, -1))

        actor.goal_safe_eps = float(getattr(hiro_cfg, "high_goal_safe_eps", 1e-6))
        actor.goal_safe_bounds_fn = calc.compute_torch
        actor.goal_safe_sampling_enabled = True
    except Exception:
        # Fallback for legacy checkpoints: disable safe sampling instead of crashing.
        actor.goal_safe_sampling_enabled = False


def main() -> None:
    csv_path = CSV_PATH.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}. Please edit CSV_PATH in this script.")

    model_path = HIGH_MODEL_PATH.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Please edit HIGH_MODEL_PATH in this script.")

    row = _load_csv_row(csv_path, int(ROW_N))

    high_obs = _parse_json_array(str(row.get("high_obs", ""))).reshape(-1)
    if high_obs.size == 0:
        raise ValueError("Column high_obs is empty or invalid for the selected row")

    ego_sub = _extract_ego_sub(row)
    lane_centers = _lane_centers_from_env_config()

    high_model = SAC.load(str(model_path))
    _bind_high_goal_safe_bounds_if_needed(high_model, high_obs)
    deterministic = not bool(USE_STOCHASTIC)
    high_obs_arr = np.asarray(high_obs, dtype=np.float32)

    goal_phys_samples: np.ndarray | None = None
    if deterministic:
        goal_action, _ = high_model.predict(high_obs_arr, deterministic=True)
        goal_action = np.asarray(goal_action, dtype=np.float32).reshape(1, -1)
        goal_phys = goal_action_to_abs(ego_sub.reshape(1, -1), goal_action, lane_centers).reshape(-1)
    else:
        sampled_actions: list[np.ndarray] = []
        sampled_goals: list[np.ndarray] = []
        n_samples = int(max(1, STOCHASTIC_SAMPLES))
        for _ in range(n_samples):
            a_s, _ = high_model.predict(high_obs_arr, deterministic=False)
            a_s = np.asarray(a_s, dtype=np.float32).reshape(1, -1)
            g_s = goal_action_to_abs(ego_sub.reshape(1, -1), a_s, lane_centers).reshape(-1)
            sampled_actions.append(a_s.reshape(-1).copy())
            sampled_goals.append(g_s.copy())

        goal_phys_samples = np.asarray(sampled_goals, dtype=np.float32).reshape(-1, 4)
        goal_phys = np.mean(goal_phys_samples, axis=0).astype(np.float32)
        goal_action = np.mean(np.asarray(sampled_actions, dtype=np.float32), axis=0, keepdims=True).astype(np.float32)

    # Reuse existing renderer by replacing row goal columns with model-predicted values.
    row_for_plot = dict(row)
    row_for_plot["goal_action"] = json.dumps(goal_action.reshape(-1).tolist())
    row_for_plot["goal_phys"] = json.dumps(goal_phys.tolist())
    if goal_phys_samples is not None:
        row_for_plot["goal_phys_samples"] = json.dumps(goal_phys_samples.tolist())

    run_dir = DEFAULT_OUT_DIR.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"hi_debug_row{int(ROW_N):04d}_pred_goal.png"
    out_path = run_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    render_high_interval_debug_snapshot(row=row_for_plot, save_path=str(out_path))

    print(f"csv_path       : {csv_path}")
    print(f"row            : {int(ROW_N)}")
    print(f"model_path     : {model_path}")
    print(f"deterministic  : {deterministic}")
    if goal_phys_samples is not None:
        print(f"stochastic_n   : {int(goal_phys_samples.shape[0])}")
        xy = np.asarray(goal_phys_samples[:, :2], dtype=np.float32)
        xy_unique = np.unique(np.round(xy, 4), axis=0)
        x_min, x_max = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
        y_min, y_max = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))
        x_std, y_std = float(np.std(xy[:, 0])), float(np.std(xy[:, 1]))
        print(f"samples_unique : {int(xy_unique.shape[0])} (rounded 1e-4)")
        print(f"samples_x_range: [{x_min:.6f}, {x_max:.6f}], std={x_std:.6f}")
        print(f"samples_y_range: [{y_min:.6f}, {y_max:.6f}], std={y_std:.6f}")
    print(f"goal_action    : {goal_action.reshape(-1).tolist()}")
    print(f"goal_phys_pred : {goal_phys.tolist()}")
    print(f"saved_plot     : {out_path}")


if __name__ == "__main__":
    main()
