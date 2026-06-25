from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from configs.builders import get_env_config_for_scenario
from custom_env.vehicle.kinematics import Vehicle
from rl.utils import utils
from util.hiro_low_test_utils import make_vehicle


DEFAULT_POOL_PATHS = [
    "debug/background_snapshot_pool_slowlane0",
    "debug/background_snapshot_pool_slowlane1",
    "debug/background_snapshot_pool_slowlane2",
]

DEFAULT_ACCELERATION_RANGE = (-3.0, 2.0)
DEFAULT_CRUISE_ACCEL_ABS = 0.2
DEFAULT_TREND_MIN_ABS_ACCEL = 0.25

DEFAULT_BUILD_ENV_ID = "multi-lane-stop-to-int-v0"
DEFAULT_BUILD_SCENARIO = "multi_lane_stop_to_int"
DEFAULT_BUILD_SCENARIO_MODULE = "scenarios.multi_lane_stop_to_int"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _vehicle_state(data: Mapping[str, Any]) -> list[float]:
    pos = np.asarray(data.get("position", [0.0, 0.0]), dtype=np.float32).reshape(-1)
    heading = float(data.get("heading", 0.0))
    speed = float(data.get("speed", 0.0))
    return [
        float(pos[0]),
        float(pos[1] if pos.size > 1 else 0.0),
        float(speed * math.cos(heading)),
        float(speed * math.sin(heading)),
    ]


def _lane_id_from_vehicle(data: Mapping[str, Any], lane_width: float) -> int:
    lane_index = data.get("lane_index", None)
    if isinstance(lane_index, (list, tuple)) and len(lane_index) >= 3:
        try:
            return int(lane_index[2])
        except Exception:
            pass
    y = float(_vehicle_state(data)[1])
    return int(round(y / max(float(lane_width), 1e-6)))


def _lane_center_y(lane_id: int, lane_width: float) -> float:
    return float(int(lane_id) * float(lane_width))


def _vehicle_state_from_obj(vehicle: Any) -> list[float]:
    pos = np.asarray(getattr(vehicle, "position", [0.0, 0.0]), dtype=np.float32).reshape(-1)
    speed = float(getattr(vehicle, "speed", 0.0))
    heading = float(getattr(vehicle, "heading", 0.0))
    return [
        float(pos[0]),
        float(pos[1] if pos.size > 1 else 0.0),
        float(speed * math.cos(heading)),
        float(speed * math.sin(heading)),
    ]


def _lane_id_from_obj(vehicle: Any, lane_width: float) -> int:
    lane_index = getattr(vehicle, "lane_index", None)
    if isinstance(lane_index, (list, tuple)) and len(lane_index) >= 3:
        try:
            return int(lane_index[2])
        except Exception:
            pass
    return int(round(float(_vehicle_state_from_obj(vehicle)[1]) / max(float(lane_width), 1e-6)))


def _kinematic_dx(v0: float, accel: float, horizon_s: float, speed_limit: float) -> float:
    v0 = max(float(v0), 0.0)
    accel = float(accel)
    horizon_s = max(float(horizon_s), 0.0)
    speed_limit = max(float(speed_limit), 0.0)
    if horizon_s <= 1e-9:
        return 0.0
    if abs(accel) <= 1e-9:
        return v0 * horizon_s
    v_end_free = v0 + accel * horizon_s
    if accel > 0.0 and v_end_free > speed_limit:
        t_to_limit = max((speed_limit - v0) / accel, 0.0)
        t_to_limit = min(t_to_limit, horizon_s)
        return v0 * t_to_limit + 0.5 * accel * t_to_limit * t_to_limit + speed_limit * (horizon_s - t_to_limit)
    if accel < 0.0 and v_end_free < 0.0:
        t_to_stop = max(-v0 / accel, 0.0)
        t_to_stop = min(t_to_stop, horizon_s)
        return v0 * t_to_stop + 0.5 * accel * t_to_stop * t_to_stop
    return v0 * horizon_s + 0.5 * accel * horizon_s * horizon_s


def _normalize_acceleration_range(acceleration_range: Sequence[float] | None) -> tuple[float, float]:
    if acceleration_range is None:
        return float(DEFAULT_ACCELERATION_RANGE[0]), float(DEFAULT_ACCELERATION_RANGE[1])
    values = list(acceleration_range)
    if len(values) < 2:
        return float(DEFAULT_ACCELERATION_RANGE[0]), float(DEFAULT_ACCELERATION_RANGE[1])
    lo = float(values[0])
    hi = float(values[1])
    if not np.isfinite(lo) or not np.isfinite(hi):
        return float(DEFAULT_ACCELERATION_RANGE[0]), float(DEFAULT_ACCELERATION_RANGE[1])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _acceleration_range_from_overrides(
    env_overrides: Mapping[str, Any] | None,
    acceleration_range: Sequence[float] | None,
) -> tuple[float, float]:
    if acceleration_range is not None:
        return _normalize_acceleration_range(acceleration_range)
    action_cfg = (env_overrides or {}).get("action", {})
    if isinstance(action_cfg, Mapping):
        return _normalize_acceleration_range(action_cfg.get("acceleration_range"))
    return _normalize_acceleration_range(None)


def _sample_goal_accel(
    trend: str,
    acceleration_range: Sequence[float],
    rng: np.random.Generator,
    *,
    cruise_accel_abs: float = DEFAULT_CRUISE_ACCEL_ABS,
    trend_min_abs_accel: float = DEFAULT_TREND_MIN_ABS_ACCEL,
) -> float:
    lo, hi = _normalize_acceleration_range(acceleration_range)
    trend = str(trend).lower().strip()
    cruise_abs = max(float(cruise_accel_abs), 0.0)
    min_abs = max(float(trend_min_abs_accel), 0.0)

    if trend == "decel":
        upper = min(hi, -min_abs)
        if lo <= upper:
            return float(rng.uniform(lo, upper))
        upper = min(hi, 0.0)
        if lo <= upper:
            return float(rng.uniform(lo, upper))
        return float(lo)

    if trend == "accel":
        lower = max(lo, min_abs)
        if lower <= hi:
            return float(rng.uniform(lower, hi))
        lower = max(lo, 0.0)
        if lower <= hi:
            return float(rng.uniform(lower, hi))
        return float(hi)

    lower = max(lo, -cruise_abs)
    upper = min(hi, cruise_abs)
    if lower <= upper:
        return float(rng.uniform(lower, upper))
    return float(np.clip(0.0, lo, hi))


def _iter_chunk_files(pool_path: Path, rng: np.random.Generator) -> list[tuple[str, dict[str, Any]]]:
    meta_path = pool_path / "meta.pkl"
    if not meta_path.is_file():
        chunk_files = [(str(p.relative_to(pool_path)), {"count": None}) for p in pool_path.rglob("*.pkl") if p.name != "meta.pkl"]
        rng.shuffle(chunk_files)
        return chunk_files

    with meta_path.open("rb") as f:
        meta = pickle.load(f)
    shards = meta.get("shards", {}) if isinstance(meta, dict) else {}
    out: list[tuple[str, dict[str, Any]]] = []
    for shard in shards.values():
        if not isinstance(shard, Mapping):
            continue
        if isinstance(shard.get("chunks", None), list):
            for chunk in shard["chunks"]:
                if isinstance(chunk, Mapping) and chunk.get("file"):
                    out.append((str(chunk["file"]), dict(chunk)))
        elif shard.get("file"):
            out.append((str(shard["file"]), dict(shard)))
    rng.shuffle(out)
    return out


def _load_snapshots_from_file(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    snapshots = payload.get("snapshots", payload) if isinstance(payload, dict) else payload
    if not isinstance(snapshots, list):
        return []
    return [s for s in snapshots if isinstance(s, dict)]


def _nearest_neighbors(
    vehicles: list[dict[str, Any]],
    ego_index: int,
    *,
    max_neighbors: int,
) -> list[dict[str, Any]]:
    ego_state = _vehicle_state(vehicles[ego_index])
    ego_x = float(ego_state[0])
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for i, data in enumerate(vehicles):
        if i == ego_index:
            continue
        try:
            state = _vehicle_state(data)
        except Exception:
            continue
        scored.append((abs(float(state[0]) - ego_x), i, data))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [deepcopy(item[2]) for item in scored[: max(0, int(max_neighbors))]]


def _make_build_env(
    *,
    env_id: str,
    scenario_name: str,
    scenario_module: str,
    env_overrides: Mapping[str, Any] | None,
):
    importlib.import_module(str(scenario_module))
    cfg = get_env_config_for_scenario(str(scenario_name), dict(env_overrides or {}))
    cfg["background_snapshot_reset"] = False
    return gym.make(str(env_id), config=cfg)


def _restore_snapshot_as_background(env, snapshot: Mapping[str, Any]) -> list[Any]:
    env.reset()
    base_env = env.unwrapped
    base_env._create_road()
    base_env.road.vehicles = []
    base_env.controlled_vehicles = []
    if hasattr(base_env, "_clear_virtual_stops"):
        base_env._clear_virtual_stops()

    cycle = base_env._snapshot_cycle_seconds_from_config() if hasattr(base_env, "_snapshot_cycle_seconds_from_config") else 120.0
    if "signal_time_global" in snapshot:
        signal_time = float(snapshot["signal_time_global"])
    else:
        phase = float(snapshot.get("phase_offset", base_env.config.get("episode_start_phase_offset", 0.0)))
        cycle_offset = float(getattr(getattr(base_env, "_signal_controller", None), "cycle_offset", 0.0))
        signal_time = (phase - cycle_offset) % cycle if cycle > 1e-9 else phase
    if hasattr(base_env, "_signal_time_global"):
        base_env._signal_time_global = signal_time
    if hasattr(base_env, "_signal_episode_base"):
        base_env._signal_episode_base = signal_time
    base_env.time = 0.0
    base_env.steps = 0

    restored: list[Any] = []
    for raw in list(snapshot.get("vehicles", []) or []):
        try:
            vehicle = base_env._vehicle_from_background_snapshot(dict(raw))
        except Exception:
            vehicle = make_vehicle(base_env.road, _vehicle_state(raw), Vehicle)
        base_env.road.vehicles.append(vehicle)
        restored.append(vehicle)
    if hasattr(base_env, "_sync_episode_punctual_time"):
        base_env._sync_episode_punctual_time()
    if hasattr(base_env, "_update_signal_virtual_stops"):
        base_env._update_signal_virtual_stops(query_time=0.0)
    return restored


def _forecast_snapshot(
    env,
    snapshot: Mapping[str, Any],
    ego_index: int,
    *,
    horizon_s: float,
    lane_width: float,
) -> dict[str, Any] | None:
    vehicles = _restore_snapshot_as_background(env, snapshot)
    if not (0 <= int(ego_index) < len(vehicles)):
        return None
    base_env = env.unwrapped
    ego_vehicle = vehicles[int(ego_index)]
    sim_freq = float(base_env.config.get("simulation_frequency", 10.0))
    dt = 1.0 / max(sim_freq, 1e-6)
    steps = int(math.ceil(max(float(horizon_s), 0.0) * sim_freq))
    for k in range(steps):
        query_time = float(k) * dt
        if hasattr(base_env, "_update_signal_virtual_stops"):
            base_env._update_signal_virtual_stops(query_time=query_time)
        base_env.road.act()
        base_env.road.step(dt)
        base_env.time = float(getattr(base_env, "time", 0.0)) + dt
        if hasattr(base_env, "_signal_time_global"):
            base_env._signal_time_global = float(getattr(base_env, "_signal_time_global", 0.0)) + dt

    forecast_vehicles: list[dict[str, Any]] = []
    for i, vehicle in enumerate(vehicles):
        if getattr(vehicle, "crashed", False):
            continue
        state = _vehicle_state_from_obj(vehicle)
        forecast_vehicles.append(
            {
                "index": int(i),
                "lane_id": _lane_id_from_obj(vehicle, lane_width),
                "state": state,
            }
        )
    return {
        "ego_state": _vehicle_state_from_obj(ego_vehicle),
        "ego_lane_id": _lane_id_from_obj(ego_vehicle, lane_width),
        "vehicles": forecast_vehicles,
    }


def _traffic_constrained_goal(
    *,
    ego_state: list[float],
    target_lane: int,
    goal_accel: float,
    forecast: Mapping[str, Any] | None,
    ego_index: int,
    horizon_s: float,
    lane_width: float,
    speed_limit: float,
    front_gap: float,
    rear_gap: float,
    acceleration_range: Sequence[float],
) -> dict[str, Any]:
    v0 = max(0.0, float(np.hypot(ego_state[2], ego_state[3])))
    accel = float(goal_accel)
    acc_min, acc_max = _normalize_acceleration_range(acceleration_range)
    desired_dx = max(_kinematic_dx(v0, accel, horizon_s, speed_limit), 2.0)
    desired_x = float(ego_state[0] + desired_dx)
    reach_dx_min = _kinematic_dx(v0, acc_min, horizon_s, speed_limit)
    reach_dx_max = _kinematic_dx(v0, acc_max, horizon_s, speed_limit)
    reach_low_dx = max(min(reach_dx_min, reach_dx_max), 0.0)
    reach_high_dx = max(max(reach_dx_min, reach_dx_max), desired_dx, 2.0)
    reach_low = float(ego_state[0] + reach_low_dx)
    reach_high = float(ego_state[0] + reach_high_dx)
    target_y = _lane_center_y(int(target_lane), lane_width)

    intervals: list[tuple[float, float]] = [(reach_low, reach_high)]
    target_vehicle_xs: list[float] = []
    if forecast is not None:
        for item in list(forecast.get("vehicles", []) or []):
            if int(item.get("index", -1)) == int(ego_index):
                continue
            if int(item.get("lane_id", -999)) != int(target_lane):
                continue
            state = item.get("state", None)
            if not isinstance(state, (list, tuple)) or len(state) < 1:
                continue
            x = float(state[0])
            if reach_low - 80.0 <= x <= reach_high + 80.0:
                target_vehicle_xs.append(x)

    if target_vehicle_xs:
        xs = sorted(target_vehicle_xs)
        bounds = [(-np.inf, xs[0] - front_gap)]
        for prev_x, next_x in zip(xs[:-1], xs[1:]):
            bounds.append((prev_x + rear_gap, next_x - front_gap))
        bounds.append((xs[-1] + rear_gap, np.inf))
        intervals = [
            (max(float(lo), reach_low), min(float(hi), reach_high))
            for lo, hi in bounds
            if max(float(lo), reach_low) <= min(float(hi), reach_high)
        ]

    if intervals:
        containing = [interval for interval in intervals if interval[0] <= desired_x <= interval[1]]
        if containing:
            lo, hi = containing[0]
        else:
            lo, hi = min(intervals, key=lambda interval: min(abs(desired_x - interval[0]), abs(desired_x - interval[1])))
        constrained_x = float(np.clip(desired_x, lo, hi))
        traffic_feasible = True
        selected_interval = [float(lo), float(hi)]
    else:
        constrained_x = float(np.clip(desired_x, reach_low, reach_high))
        traffic_feasible = False
        selected_interval = [float(reach_low), float(reach_high)]

    v_goal = float(np.clip(v0 + accel * horizon_s, 0.0, speed_limit))
    return {
        "goal_phys": [constrained_x, target_y, v_goal, 0.0],
        "goal_unconstrained": [desired_x, target_y, v_goal, 0.0],
        "goal_accel": accel,
        "reach_interval_x": [float(reach_low), float(reach_high)],
        "traffic_interval_x": selected_interval,
        "traffic_feasible": bool(traffic_feasible),
        "traffic_adjusted": bool(abs(constrained_x - desired_x) > 1e-6),
        "traffic_vehicle_count": int(len(target_vehicle_xs)),
    }


def _candidate_bucket(
    ego_data: Mapping[str, Any],
    target_lane: int,
    trend: str,
    *,
    lane_width: float,
) -> tuple[int, int, str]:
    return (
        _lane_id_from_vehicle(ego_data, lane_width),
        int(target_lane),
        str(trend),
    )


def _speed_label(speed: float) -> str:
    speed = float(speed)
    if speed < 6.0:
        return "低初速度"
    if speed < 11.0:
        return "中初速度"
    return "高初速度"


def _trend_label(trend: str) -> str:
    text = str(trend).lower().strip()
    if text == "accel":
        return "加速"
    if text == "decel":
        return "减速"
    return "匀速"


def _lane_change_label(ego_lane: int, target_lane: int) -> str:
    delta = int(target_lane) - int(ego_lane)
    if delta == 0:
        return "保持车道"
    # Lane 0 is the leftmost lane in the current road layout.
    return "向左换道" if delta < 0 else "向右换道"


def _scenario_label(ego_state: Sequence[float], ego_lane: int, target_lane: int, trend: str) -> str:
    speed = float(np.hypot(float(ego_state[2]), float(ego_state[3])))
    return "_".join(
        [
            _speed_label(speed),
            _trend_label(trend),
            _lane_change_label(int(ego_lane), int(target_lane)),
        ]
    )


def _make_case(
    *,
    case_id: int,
    pool_name: str,
    chunk_file: str,
    snapshot_index: int,
    snapshot: Mapping[str, Any],
    ego_index: int,
    target_lane: int,
    trend: str,
    goal_accel: float,
    horizon_s: float,
    lane_width: float,
    speed_limit: float,
    max_neighbors: int,
    forecast: Mapping[str, Any] | None = None,
    traffic_front_gap: float = 12.0,
    traffic_rear_gap: float = 8.0,
    acceleration_range: Sequence[float] = DEFAULT_ACCELERATION_RANGE,
) -> dict[str, Any]:
    vehicles = list(snapshot.get("vehicles", []) or [])
    ego_data = vehicles[ego_index]
    ego_state = _vehicle_state(ego_data)
    goal_meta = _traffic_constrained_goal(
        ego_state=ego_state,
        target_lane=int(target_lane),
        goal_accel=float(goal_accel),
        forecast=forecast,
        ego_index=int(ego_index),
        horizon_s=float(horizon_s),
        lane_width=float(lane_width),
        speed_limit=float(speed_limit),
        front_gap=float(traffic_front_gap),
        rear_gap=float(traffic_rear_gap),
        acceleration_range=acceleration_range,
    )
    goal_phys = goal_meta["goal_phys"]
    ego_lane_id = _lane_id_from_vehicle(ego_data, lane_width)
    neighbors_snapshot = _nearest_neighbors(
        vehicles,
        ego_index,
        max_neighbors=max_neighbors,
    )
    return {
        "case_id": int(case_id),
        "source_pool": str(pool_name),
        "source_chunk": str(chunk_file),
        "source_snapshot_index": int(snapshot_index),
        "source_ego_index": int(ego_index),
        "phase_offset": float(snapshot.get("phase_offset", 0.0)),
        "signal_time_global": float(snapshot.get("signal_time_global", snapshot.get("phase_offset", 0.0))),
        "ego_lane_id": int(ego_lane_id),
        "target_lane_id": int(target_lane),
        "trend": str(trend),
        "scenario_label": _scenario_label(ego_state, int(ego_lane_id), int(target_lane), str(trend)),
        "horizon_s": float(horizon_s),
        "goal_accel": float(goal_meta["goal_accel"]),
        "ego_state": [float(v) for v in ego_state],
        "goal_phys": [float(v) for v in goal_phys],
        "goal_unconstrained": [float(v) for v in goal_meta["goal_unconstrained"]],
        "reach_interval_x": [float(v) for v in goal_meta["reach_interval_x"]],
        "traffic_interval_x": [float(v) for v in goal_meta["traffic_interval_x"]],
        "traffic_feasible": bool(goal_meta["traffic_feasible"]),
        "traffic_adjusted": bool(goal_meta["traffic_adjusted"]),
        "traffic_vehicle_count": int(goal_meta["traffic_vehicle_count"]),
        "forecast_ego_state": None
        if forecast is None
        else [float(v) for v in list(forecast.get("ego_state", []))[:4]],
        "forecast_ego_lane_id": None if forecast is None else int(forecast.get("ego_lane_id", -1)),
        "ego_snapshot": _json_safe(deepcopy(ego_data)),
        "neighbors_state": [_vehicle_state(v) for v in neighbors_snapshot],
        "neighbors_snapshot": _json_safe(neighbors_snapshot),
    }


def build_low_eval_cases(
    *,
    pool_paths: Iterable[str | os.PathLike[str]] = DEFAULT_POOL_PATHS,
    output_path: str | os.PathLike[str],
    env_id: str = DEFAULT_BUILD_ENV_ID,
    scenario_name: str = DEFAULT_BUILD_SCENARIO,
    scenario_module: str = DEFAULT_BUILD_SCENARIO_MODULE,
    env_overrides: Mapping[str, Any] | None = None,
    cases_per_bucket: int = 8,
    seed: int = 42,
    ego_x_range: tuple[float, float] = (0.0, 350.0),
    target_lanes: Iterable[int] = (0, 1, 2),
    trends: Iterable[str] = ("decel", "cruise", "accel"),
    horizon_s: float = 2.5,
    lane_width: float = 4.0,
    speed_limit: float = 15.0,
    max_neighbors: int = 24,
    max_chunks_per_pool: int = 24,
    use_forward_sim_constraints: bool = True,
    traffic_front_gap: float = 12.0,
    traffic_rear_gap: float = 8.0,
    acceleration_range: Sequence[float] | None = None,
    cruise_accel_abs: float = DEFAULT_CRUISE_ACCEL_ABS,
    trend_min_abs_accel: float = DEFAULT_TREND_MIN_ABS_ACCEL,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    pool_paths = [Path(p) for p in pool_paths]
    target_lanes = [int(v) for v in target_lanes]
    trends = [str(v) for v in trends]
    accel_min, accel_max = _acceleration_range_from_overrides(env_overrides, acceleration_range)
    resolved_acceleration_range = (float(accel_min), float(accel_max))
    desired_keys = {
        (ego_lane, target_lane, trend)
        for ego_lane in target_lanes
        for target_lane in target_lanes
        for trend in trends
    }
    buckets: dict[tuple[int, int, str], list[dict[str, Any]]] = {key: [] for key in desired_keys}

    case_counter = 0
    forecast_env = None
    if bool(use_forward_sim_constraints):
        forecast_env = _make_build_env(
            env_id=str(env_id),
            scenario_name=str(scenario_name),
            scenario_module=str(scenario_module),
            env_overrides=dict(env_overrides or {}),
        )

    try:
        for pool_path in pool_paths:
            chunk_entries = _iter_chunk_files(pool_path, rng)[: max(1, int(max_chunks_per_pool))]
            for chunk_file, _chunk_info in chunk_entries:
                snapshots = _load_snapshots_from_file(pool_path / chunk_file)
                if not snapshots:
                    continue
                order = np.arange(len(snapshots))
                rng.shuffle(order)
                for snap_idx in order:
                    snapshot = snapshots[int(snap_idx)]
                    vehicles = list(snapshot.get("vehicles", []) or [])
                    if not vehicles:
                        continue
                    ego_indices = list(range(len(vehicles)))
                    rng.shuffle(ego_indices)
                    for ego_index in ego_indices:
                        ego_data = vehicles[int(ego_index)]
                        try:
                            ego_state = _vehicle_state(ego_data)
                            ego_lane = _lane_id_from_vehicle(ego_data, lane_width)
                        except Exception:
                            continue
                        if ego_lane not in target_lanes:
                            continue
                        if not (float(ego_x_range[0]) <= float(ego_state[0]) <= float(ego_x_range[1])):
                            continue
                        open_keys = [
                            (ego_lane, target_lane, trend)
                            for target_lane in target_lanes
                            for trend in trends
                            if (ego_lane, target_lane, trend) in buckets
                            and len(buckets[(ego_lane, target_lane, trend)]) < int(cases_per_bucket)
                        ]
                        if not open_keys:
                            continue
                        forecast = None
                        if forecast_env is not None:
                            try:
                                forecast = _forecast_snapshot(
                                    forecast_env,
                                    snapshot,
                                    int(ego_index),
                                    horizon_s=float(horizon_s),
                                    lane_width=float(lane_width),
                                )
                            except Exception:
                                forecast = None
                        key = open_keys[int(rng.integers(len(open_keys)))]
                        _ego_lane, target_lane, trend = key
                        goal_accel = _sample_goal_accel(
                            str(trend),
                            resolved_acceleration_range,
                            rng,
                            cruise_accel_abs=float(cruise_accel_abs),
                            trend_min_abs_accel=float(trend_min_abs_accel),
                        )
                        case = _make_case(
                            case_id=case_counter,
                            pool_name=pool_path.name,
                            chunk_file=chunk_file,
                            snapshot_index=int(snap_idx),
                            snapshot=snapshot,
                            ego_index=int(ego_index),
                            target_lane=int(target_lane),
                            trend=str(trend),
                            goal_accel=float(goal_accel),
                            horizon_s=float(horizon_s),
                            lane_width=float(lane_width),
                            speed_limit=float(speed_limit),
                            max_neighbors=int(max_neighbors),
                            forecast=forecast,
                            traffic_front_gap=float(traffic_front_gap),
                            traffic_rear_gap=float(traffic_rear_gap),
                            acceleration_range=resolved_acceleration_range,
                        )
                        buckets[key].append(case)
                        case_counter += 1
                        if all(len(v) >= int(cases_per_bucket) for v in buckets.values()):
                            break
                    if all(len(v) >= int(cases_per_bucket) for v in buckets.values()):
                        break
                if all(len(v) >= int(cases_per_bucket) for v in buckets.values()):
                    break
    finally:
        if forecast_env is not None:
            forecast_env.close()

    cases: list[dict[str, Any]] = []
    for key in sorted(buckets):
        cases.extend(buckets[key])
    for i, case in enumerate(cases):
        case["case_id"] = int(i)

    payload = {
        "version": 1,
        "description": "Offline low-level HIRO evaluation cases sampled from background snapshot pools.",
        "build_config": {
            "pool_paths": [str(p) for p in pool_paths],
            "env_id": str(env_id),
            "scenario_name": str(scenario_name),
            "scenario_module": str(scenario_module),
            "env_overrides": _json_safe(dict(env_overrides or {})),
            "cases_per_bucket": int(cases_per_bucket),
            "seed": int(seed),
            "ego_x_range": [float(ego_x_range[0]), float(ego_x_range[1])],
            "target_lanes": target_lanes,
            "trends": trends,
            "horizon_s": float(horizon_s),
            "lane_width": float(lane_width),
            "speed_limit": float(speed_limit),
            "max_neighbors": int(max_neighbors),
            "max_chunks_per_pool": int(max_chunks_per_pool),
            "use_forward_sim_constraints": bool(use_forward_sim_constraints),
            "traffic_front_gap": float(traffic_front_gap),
            "traffic_rear_gap": float(traffic_rear_gap),
            "acceleration_range": [float(accel_min), float(accel_max)],
            "cruise_accel_abs": float(cruise_accel_abs),
            "trend_min_abs_accel": float(trend_min_abs_accel),
        },
        "bucket_counts": {
            f"ego{key[0]}_target{key[1]}_{key[2]}": len(value)
            for key, value in sorted(buckets.items())
        },
        "cases": _json_safe(cases),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def load_low_eval_cases(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No low eval cases found in {path}")
    return [dict(case) for case in cases if isinstance(case, Mapping)]


class HIROLowOfflineEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        cases_path: str,
        env_id: str,
        scenario_name: str,
        scenario_module: str,
        env_overrides: Mapping[str, Any] | None,
        eval_freq: int = 200_000,
        sample_size: int = 90,
        seed: int = 42,
        build_if_missing: bool = True,
        build_config: Mapping[str, Any] | None = None,
        deterministic: bool = True,
        log_prefix: str = "low_eval",
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.cases_path = str(cases_path)
        self.env_id = str(env_id)
        self.scenario_name = str(scenario_name)
        self.scenario_module = str(scenario_module)
        self.env_overrides = deepcopy(dict(env_overrides or {}))
        self.eval_freq = max(1, int(eval_freq))
        self.sample_size = max(1, int(sample_size))
        self.seed = int(seed)
        self.build_if_missing = bool(build_if_missing)
        self.build_config = deepcopy(dict(build_config or {}))
        self.deterministic = bool(deterministic)
        self.log_prefix = str(log_prefix).rstrip("/")
        self.rng = np.random.default_rng(self.seed)
        self.cases: list[dict[str, Any]] = []
        self.eval_env = None
        self._next_eval_step = self.eval_freq

    def _tb_logger(self):
        for attr in ("low_logger", "high_logger", "logger"):
            logger = getattr(self.model, attr, None)
            if logger is not None:
                return logger
        raise AttributeError(
            "HIROLowOfflineEvalCallback could not find a logger on the model "
            "(expected low_logger/high_logger/logger)"
        )

    def _on_training_start(self) -> None:
        if not os.path.isfile(self.cases_path):
            if not self.build_if_missing:
                raise FileNotFoundError(f"Low eval cases not found: {self.cases_path}")
            cfg = dict(self.build_config)
            cfg.setdefault("output_path", self.cases_path)
            cfg.setdefault("env_id", self.env_id)
            cfg.setdefault("scenario_name", self.scenario_name)
            cfg.setdefault("scenario_module", self.scenario_module)
            cfg.setdefault("env_overrides", self.env_overrides)
            build_low_eval_cases(**cfg)
            if self.verbose:
                print(f"[HIRO Low Eval] Built cases: {self.cases_path}")
        self.cases = load_low_eval_cases(self.cases_path)
        importlib.import_module(self.scenario_module)
        env_cfg = get_env_config_for_scenario(self.scenario_name, self.env_overrides)
        env_cfg["background_snapshot_reset"] = False
        self.eval_env = gym.make(self.env_id, config=env_cfg)
        if self.verbose:
            print(f"[HIRO Low Eval] Loaded {len(self.cases)} cases from {self.cases_path}")

    def _on_training_end(self) -> None:
        if self.eval_env is not None:
            self.eval_env.close()
            self.eval_env = None

    def _setup_case(self, case: Mapping[str, Any]) -> np.ndarray:
        assert self.eval_env is not None
        env = self.eval_env
        obs, _info = env.reset()
        base_env = env.unwrapped
        ego_state = np.asarray(case["ego_state"], dtype=np.float32).reshape(4)
        neighbors_snapshot = list(case.get("neighbors_snapshot", []) or [])
        neighbors_state = list(case.get("neighbors_state", []) or [])

        road = base_env.road
        road.vehicles = []
        base_env.controlled_vehicles = []
        ego_cls = base_env.action_type.vehicle_class
        ego = make_vehicle(road, ego_state, ego_cls)
        neighbors = []
        if neighbors_snapshot and hasattr(base_env, "_vehicle_from_background_snapshot"):
            for raw in neighbors_snapshot:
                try:
                    neighbors.append(base_env._vehicle_from_background_snapshot(dict(raw)))
                except Exception:
                    state = _vehicle_state(raw)
                    neighbors.append(make_vehicle(road, state, Vehicle))
        else:
            for state in neighbors_state:
                neighbors.append(make_vehicle(road, state, Vehicle))

        base_env.controlled_vehicles = [ego]
        base_env.vehicle = ego
        road.vehicles = [ego] + neighbors
        for vehicle in road.vehicles:
            if hasattr(vehicle, "history"):
                vehicle.history.clear()
                vehicle.history.appendleft(Vehicle.create_from(vehicle))

        signal_time = float(case.get("signal_time_global", case.get("phase_offset", 0.0)))
        if hasattr(base_env, "_signal_time_global"):
            base_env._signal_time_global = signal_time
        if hasattr(base_env, "_signal_episode_base"):
            base_env._signal_episode_base = signal_time
        if hasattr(base_env, "_sync_episode_punctual_time"):
            base_env._sync_episode_punctual_time()
        if hasattr(base_env, "_update_signal_virtual_stops"):
            base_env._update_signal_virtual_stops(query_time=0.0)

        if hasattr(base_env, "set_hiro_goal"):
            goal = np.asarray(case["goal_phys"], dtype=np.float32).reshape(-1)
            if goal.size >= 4:
                try:
                    base_env.set_hiro_goal(goal.copy())
                except Exception:
                    pass

        obs = base_env.observation_type.observe()
        return np.asarray(obs, dtype=np.float32)

    def _low_obs(self, obs: np.ndarray, goal_phys: np.ndarray, t_rel: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        _, kin, kin_flat = utils.split_time_kinematics(arr, self.model.n_veh, self.model.feat_dim)
        obs_extra = arr[:, 1 + self.model.kin_flat_dim : 1 + self.model.kin_flat_dim + self.model.obs_extra_dim]
        low_obs = self.model._build_low_obs(
            np.asarray([int(t_rel)], dtype=np.int32),
            kin_flat,
            kin,
            goal_phys.reshape(1, -1),
            obs_extra,
        )
        return low_obs, kin, kin_flat

    def _ego_sub_from_obs(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        _, kin, _ = utils.split_time_kinematics(arr, self.model.n_veh, self.model.feat_dim)
        return utils.extract_ego_substate(kin, self.model.ego_feature_idx)[0].astype(np.float32)

    def _evaluate_case(self, case: Mapping[str, Any]) -> dict[str, float | str | int]:
        assert self.eval_env is not None
        obs = self._setup_case(case)
        goal_phys = np.asarray(case["goal_phys"], dtype=np.float32).reshape(-1)
        start_ego = self._ego_sub_from_obs(obs)
        rewards: list[float] = []
        intrinsic_rewards: list[float] = []
        comfort_rewards: list[float] = []
        safety_clips = 0
        done = False
        info: dict[str, Any] = {}
        horizon = int(getattr(self.model.cfg, "high_interval", 25))

        for t in range(horizon):
            low_obs, kin, _kin_flat = self._low_obs(obs, goal_phys, t)
            action_raw = self.model.low_agent.predict_action(low_obs, deterministic=self.deterministic)
            action = np.asarray(action_raw, dtype=np.float32)
            if bool(getattr(self.model.cfg, "use_low_safety_layer", False)) and hasattr(self.model, "low_safety"):
                safe_action = self.model.low_safety.apply_safety_layer(low_obs, goal_phys.reshape(1, -1), action)
                if np.any(np.abs(safe_action - action) > 1e-6):
                    safety_clips += 1
                action = np.asarray(safe_action, dtype=np.float32)
            if action.ndim == 2:
                step_action = action[0]
            else:
                step_action = action.reshape(-1)
            obs, reward, terminated, truncated, info = self.eval_env.step(step_action)
            next_arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
            _, kin_next, _ = utils.split_time_kinematics(next_arr, self.model.n_veh, self.model.feat_dim)
            try:
                intrinsic, _goal_err, _intrinsic_unw = self.model._compute_intrinsic(
                    kin,
                    kin_next,
                    goal_phys.reshape(1, -1),
                    start_ego.reshape(1, -1),
                    intrinsic_terminal_mask=np.asarray([t == horizon - 1], dtype=bool),
                    goal_err_mask=np.asarray([t == horizon - 1], dtype=bool),
                )
                intrinsic_rewards.append(float(np.asarray(intrinsic, dtype=np.float32).reshape(-1)[0]))
            except Exception:
                pass
            if isinstance(info, dict):
                rc = info.get("reward_components", {}) or {}
                if isinstance(rc, Mapping):
                    comfort_rewards.append(float(rc.get("comfort_reward", 0.0)))
            rewards.append(float(reward))
            done = bool(terminated or truncated)
            if done:
                break

        final_ego = self._ego_sub_from_obs(obs)
        goal_err = goal_phys[: final_ego.size] - final_ego
        abs_dx = float(abs(goal_err[0])) if goal_err.size >= 1 else 0.0
        abs_dy = float(abs(goal_err[1])) if goal_err.size >= 2 else 0.0
        collision = float(bool(getattr(self.eval_env.unwrapped.vehicle, "crashed", False)))
        queue_terminal = float(bool(info.get("queue_takeover_terminal", False))) if isinstance(info, dict) else 0.0
        ego_lane_id = int(case.get("ego_lane_id", -1))
        target_lane_id = int(case.get("target_lane_id", -1))
        final_lane_id = _lane_id_from_obj(
            self.eval_env.unwrapped.vehicle,
            float(self.eval_env.unwrapped.config.get("lane_width", 4.0)),
        )
        success = float(
            abs_dx <= 5.0
            and abs_dy <= 1.2
            and collision <= 0.0
            and queue_terminal <= 0.0
        )
        lane_change_success = None
        if ego_lane_id >= 0 and target_lane_id >= 0 and ego_lane_id != target_lane_id:
            lane_change_success = float(
                final_lane_id == target_lane_id
                and abs_dy <= 1.2
                and collision <= 0.0
                and queue_terminal <= 0.0
            )
        return {
            "case_id": int(case.get("case_id", -1)),
            "trend": str(case.get("trend", "unknown")),
            "target_lane_id": target_lane_id,
            "ego_lane_id": ego_lane_id,
            "goal_err_x": abs_dx,
            "success": success,
            "safety_clip": float(safety_clips / max(len(rewards), 1)),
            "intrinsic_reward": float(np.sum(intrinsic_rewards)) if intrinsic_rewards else 0.0,
            "comfort_reward": float(np.sum(comfort_rewards)) if comfort_rewards else 0.0,
            **({} if lane_change_success is None else {"lane_change_success": lane_change_success}),
        }

    def _record_group(self, name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        numeric_keys = [
            "intrinsic_reward",
            "goal_err_x",
            "lane_change_success",
            "comfort_reward",
            "success",
            "safety_clip",
        ]
        prefix = f"{self.log_prefix}/{name}" if name else self.log_prefix
        for key in numeric_keys:
            vals = [float(row[key]) for row in rows if key in row and np.isfinite(float(row[key]))]
            if vals:
                self._tb_logger().record(f"{prefix}/{key}_mean", float(np.mean(vals)))

    def _run_eval(self) -> None:
        n = min(self.sample_size, len(self.cases))
        indices = self.rng.choice(len(self.cases), size=n, replace=False if n <= len(self.cases) else True)
        rows = [self._evaluate_case(self.cases[int(i)]) for i in indices]
        self._record_group("all", rows)
        logger = self._tb_logger()
        logger.dump(step=int(getattr(self.model, "num_timesteps", self.num_timesteps)))

    def _on_step(self) -> bool:
        current = int(getattr(self.model, "num_timesteps", self.num_timesteps))
        if current >= self._next_eval_step:
            self._run_eval()
            while self._next_eval_step <= current:
                self._next_eval_step += self.eval_freq
        return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline HIRO low-level eval cases.")
    parser.add_argument("--output", default="debug/hiro_low_eval_cases_snapshot012.json")
    parser.add_argument("--pool", action="append", dest="pools", default=None)
    parser.add_argument("--env-id", default=DEFAULT_BUILD_ENV_ID)
    parser.add_argument("--scenario-name", default=DEFAULT_BUILD_SCENARIO)
    parser.add_argument("--scenario-module", default=DEFAULT_BUILD_SCENARIO_MODULE)
    parser.add_argument("--cases-per-bucket", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-chunks-per-pool", type=int, default=24)
    parser.add_argument("--max-neighbors", type=int, default=24)
    parser.add_argument("--no-forward-sim-constraints", action="store_true")
    parser.add_argument("--traffic-front-gap", type=float, default=12.0)
    parser.add_argument("--traffic-rear-gap", type=float, default=8.0)
    parser.add_argument("--acceleration-range", type=float, nargs=2, default=list(DEFAULT_ACCELERATION_RANGE))
    parser.add_argument("--cruise-accel-abs", type=float, default=DEFAULT_CRUISE_ACCEL_ABS)
    parser.add_argument("--trend-min-abs-accel", type=float, default=DEFAULT_TREND_MIN_ABS_ACCEL)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_low_eval_cases(
        pool_paths=args.pools or DEFAULT_POOL_PATHS,
        output_path=args.output,
        env_id=args.env_id,
        scenario_name=args.scenario_name,
        scenario_module=args.scenario_module,
        cases_per_bucket=args.cases_per_bucket,
        seed=args.seed,
        max_chunks_per_pool=args.max_chunks_per_pool,
        max_neighbors=args.max_neighbors,
        use_forward_sim_constraints=not bool(args.no_forward_sim_constraints),
        traffic_front_gap=args.traffic_front_gap,
        traffic_rear_gap=args.traffic_rear_gap,
        acceleration_range=args.acceleration_range,
        cruise_accel_abs=args.cruise_accel_abs,
        trend_min_abs_accel=args.trend_min_abs_accel,
    )
    print(f"[HIRO Low Eval] Wrote {len(payload['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()
