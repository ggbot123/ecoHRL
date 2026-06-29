from __future__ import annotations

import copy
import os
import pickle
import queue
import shutil
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import Manager
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.utils import seeding

from configs.builders import get_env_config_for_scenario
from configs.conf import MASTER_SEED, TRAIN_CONFIG
from scenarios.multi_lane.scenario import MultiLaneEnv
from scenarios.multi_lane_stop_to_int.scenario import MultiLaneStopToIntEnv
from util.config_utils import deep_update

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency fallback
    tqdm = None


SNAPSHOT_OFFSET_KEY_SCALE = 1000


# Edit this dict directly, then run:
#   python util/build_background_snapshot_pool.py
SNAPSHOT_POOL_CONFIG: dict[str, Any] = {
    # "multi_lane_stop_to_int" uses signal offsets. "multi_lane" has no signal
    # offset and records one background-only stream snapshot every
    # snapshot_interval_seconds.

    # "scenario_name": "multi_lane_stop_to_int",
    "scenario_name": "multi_lane",

    # Explicit offsets take precedence. Use [20.0, 40.0, 90.0] for selected offsets.
    # Leave as None to either use the offset grid below or, if the grid is disabled,
    # TRAIN_CONFIG["config_overrides"]["environment"]["episode_start_phase_offset"].
    "offsets": None,
    # Optional offset grid. Set start/stop to cover a signal cycle at 1s resolution,
    # for example offset_start=0.0, offset_stop=119.0, offset_step=1.0.
    "offset_start": 0,
    "offset_stop": 119,
    "offset_step": 1,
    "count_per_offset": 50000,
    "snapshot_interval_seconds": 50.0,
    "workers": 8,
    "seed": MASTER_SEED,
    "out_path": Path("debug") / "background_snapshot_pool_slowlane2_oldEnv",
    # "out_path": Path("debug") / "background_snapshot_pool_slowlane1",
    "overwrite": True,
    # Training reset only loads one chunk at a time. Smaller chunks use less
    # memory but read from disk more often.
    "chunk_size": 500,
    # None means keep the value from configs/conf.py after TRAIN_CONFIG environment overrides.
    "warmup_time": None,
    # True means start from TRAIN_CONFIG["config_overrides"]["environment"],
    # then apply "env_overrides" below.
    "use_train_env_overrides": True,
    # Extra overrides for snapshot generation. Keep these aligned with training.
    "env_overrides": {
        "background_snapshot_reset": False,
        "inter_episode_as_steps": False,
        "align_ego_spawn_to_signal_offset": True,
    },
    "log_every": 100,
    "use_tqdm": True,
    # Worker-to-main progress update frequency in snapshots. Set higher if
    # progress IPC overhead becomes noticeable on very large full-cycle pools.
    "progress_every": 1,
    # Optional offline filtering mode. When enabled, this script does not run
    # simulation; it reads filter_source_path and writes a smaller compatible
    # sharded pool containing only snapshots with at least one eligible ego
    # candidate.
    "filter_existing_pool_enabled": True,
    "filter_source_path": Path("debug") / "background_snapshot_pool_slowlane2",
    "filter_out_path": Path("debug") / "background_snapshot_pool_slowlane2_x0_200_v7_15",
    # Optional shard filter for offline filtering. filter_offsets uses seconds
    # and is converted to the internal key scale, e.g. 20.0 -> "20000".
    "filter_offsets": [20.0],
    "filter_offset_keys": None,
    "filter_ego_x_range": [0.0, 200.0],
    "filter_ego_speed_range": [7.0, 15.0],
    "filter_chunk_size": 500,
    "filter_overwrite": True,
}


def _configured_offsets(config: dict[str, Any]) -> list[float]:
    offsets = config.get("offsets", None)
    if offsets is not None:
        if isinstance(offsets, (int, float)):
            return [float(offsets)]
        return [float(x) for x in offsets]

    start = config.get("offset_start", None)
    stop = config.get("offset_stop", None)
    if start is not None or stop is not None:
        if start is None or stop is None:
            raise ValueError("offset_start and offset_stop must be set together")
        step = float(config.get("offset_step", 1.0))
        if step <= 0.0:
            raise ValueError("offset_step must be positive")
        values: list[float] = []
        current = float(start)
        stop_f = float(stop)
        eps = step * 1e-6
        while current <= stop_f + eps:
            values.append(round(current, 9))
            current += step
        return values

    default_offset = (
        TRAIN_CONFIG.get("config_overrides", {})
        .get("environment", {})
        .get("episode_start_phase_offset", 0.0)
    )
    return [float(default_offset)]


def _offset_key(offset: float) -> str:
    return str(int(round(float(offset) * SNAPSHOT_OFFSET_KEY_SCALE)))


def _configured_filter_offset_keys(config: dict[str, Any]) -> set[str] | None:
    raw_keys = config.get("filter_offset_keys", None)
    if raw_keys is not None:
        if isinstance(raw_keys, (str, int, float)):
            return {str(raw_keys)}
        return {str(key) for key in raw_keys}

    raw_offsets = config.get("filter_offsets", None)
    if raw_offsets is None:
        return None
    if isinstance(raw_offsets, (int, float)):
        offsets = [float(raw_offsets)]
    else:
        offsets = [float(offset) for offset in raw_offsets]
    return {_offset_key(offset) for offset in offsets}


def _chunk_path(temp_dir: Path, worker_id: int, offset_key: str, chunk_index: int) -> Path:
    return temp_dir / f"worker_{int(worker_id):03d}_offset_{offset_key}_chunk_{int(chunk_index):06d}.pkl"


def _flush_chunk(
    temp_dir: Path,
    worker_id: int,
    offset_key: str,
    buffers: dict[str, list[dict[str, Any]]],
    chunk_counts: dict[str, int],
) -> Path | None:
    buffer = buffers.get(offset_key, [])
    if not buffer:
        return None
    chunk_index = chunk_counts.get(offset_key, 0)
    path = _chunk_path(temp_dir, worker_id, offset_key, chunk_index)
    with path.open("wb") as f:
        pickle.dump(buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
    buffers[offset_key] = []
    chunk_counts[offset_key] = chunk_index + 1
    return path


def _seed_env_without_reset(env: Any, seed: int) -> None:
    env.np_random, _ = seeding.np_random(int(seed))
    if getattr(env, "road", None) is not None:
        env.road.np_random = env.np_random


def _make_env_config(offset: float, config: dict[str, Any]) -> dict:
    scenario_name = str(config.get("scenario_name", "multi_lane_stop_to_int"))
    overrides: dict[str, Any] = {}
    if bool(config.get("use_train_env_overrides", True)):
        overrides = copy.deepcopy(
            TRAIN_CONFIG.get("config_overrides", {}).get("environment", {}) or {}
        )
    extra_overrides = copy.deepcopy(config.get("env_overrides", {}) or {})
    deep_update(overrides, extra_overrides)
    overrides["background_snapshot_reset"] = False
    if scenario_name == "multi_lane_stop_to_int":
        overrides["episode_start_phase_offset"] = float(offset)
        overrides["inter_episode_as_steps"] = False
        overrides["align_ego_spawn_to_signal_offset"] = True
    if config.get("warmup_time", None) is not None:
        overrides["warmup_time"] = float(config["warmup_time"])
    return get_env_config_for_scenario(scenario_name, overrides)


def _make_env_instance(scenario_name: str, cfg: dict[str, Any]):
    if scenario_name == "multi_lane":
        return MultiLaneEnv(config=cfg)
    if scenario_name == "multi_lane_stop_to_int":
        return MultiLaneStopToIntEnv(config=cfg)
    raise ValueError(f"Unsupported snapshot scenario_name: {scenario_name!r}")


def _initialize_background_only_stream(env: Any, seed: int) -> None:
    """Create a road and evolve only background traffic; ego is never inserted."""
    _seed_env_without_reset(env, seed)
    env.time = 0.0
    env.steps = 0
    env.done = False
    if hasattr(env, "_signal_time_global"):
        env._signal_time_global = 0.0
    if hasattr(env, "_signal_episode_base"):
        env._signal_episode_base = 0.0
    if hasattr(env, "_episodes_started"):
        env._episodes_started = 0
    if hasattr(env, "_inter_episode_active"):
        env._inter_episode_active = False
    if hasattr(env, "_inter_episode_remaining"):
        env._inter_episode_remaining = 0.0
    env._background_only_sim_time = 0.0

    env._create_road()
    env.road.vehicles = []
    env.controlled_vehicles = []
    if hasattr(env, "_clear_virtual_stops"):
        env._clear_virtual_stops()

    env._warmup(render=False)


def _advance_background_to_offset(
    env: MultiLaneStopToIntEnv,
    offset: float,
    *,
    strict_next: bool,
) -> None:
    env.config["episode_start_phase_offset"] = float(offset)
    env._advance_to_episode_start_offset(strict_next=bool(strict_next))


def _advance_background_interval(env: Any, seconds: float) -> None:
    sim_freq = float(env.config.get("simulation_frequency", 10.0))
    dt = 1.0 / max(sim_freq, 1e-6)
    steps = int(round(max(float(seconds), 0.0) * sim_freq))
    for _ in range(max(steps, 0)):
        env._clear_background()
        env._spawn_background()
        if hasattr(env, "_update_signal_virtual_stops"):
            env._update_signal_virtual_stops(query_time=float(getattr(env, "time", 0.0)))
        env.road.act()
        env.road.step(dt)
        env.time = float(getattr(env, "time", 0.0)) + dt
        if hasattr(env, "_signal_time_global"):
            env._signal_time_global = float(getattr(env, "_signal_time_global", 0.0)) + dt
    env._clear_background()
    env._background_only_sim_time = float(getattr(env, "_background_only_sim_time", 0.0)) + (
        max(steps, 0) * dt
    )


def _collect_worker(
    args: tuple[int, tuple[float, ...], int, int, dict[str, Any], str, Any],
) -> dict[str, Any]:
    worker_id, offsets, count_per_offset, seed, config, temp_dir_raw, progress_sink = args
    if count_per_offset <= 0:
        return {
            "worker_id": int(worker_id),
            "offsets": [float(x) for x in offsets],
            "snapshot_count": 0,
            "counts_by_offset": {},
            "background_min": None,
            "background_max": None,
            "background_sum": 0,
            "config_signature": None,
        }

    temp_dir = Path(temp_dir_raw)
    scenario_name = str(config.get("scenario_name", "multi_lane_stop_to_int"))
    offsets_sorted = tuple(sorted(float(x) for x in offsets))
    if scenario_name == "multi_lane":
        offsets_sorted = (0.0,)
    chunk_size = max(int(config.get("chunk_size", 1000)), 1)
    progress_every = max(int(config.get("progress_every", 1)), 1)
    cfg = _make_env_config(offsets_sorted[0], config)
    env = _make_env_instance(scenario_name, cfg)
    buffers: dict[str, list[dict[str, Any]]] = {_offset_key(offset): [] for offset in offsets_sorted}
    chunk_counts: dict[str, int] = {}
    counts_by_offset: dict[str, int] = {_offset_key(offset): 0 for offset in offsets_sorted}
    snapshot_count = 0
    pending_progress = 0
    background_min = None
    background_max = None
    background_sum = 0
    try:
        _initialize_background_only_stream(env, seed)
        config_signature = env._snapshot_config_signature()
        if scenario_name == "multi_lane":
            interval = float(config.get("snapshot_interval_seconds", 50.0))
            key = _offset_key(0.0)
            for local_index in range(count_per_offset):
                if local_index > 0:
                    _advance_background_interval(env, interval)
                snapshot = env.export_background_snapshot()
                snapshot["worker_id"] = int(worker_id)
                snapshot["worker_local_index"] = int(local_index)
                snapshot["worker_offset_index"] = 0
                snapshot["snapshot_interval_seconds"] = float(interval)
                buffers[key].append(snapshot)
                counts_by_offset[key] = counts_by_offset.get(key, 0) + 1
                snapshot_count += 1
                pending_progress += 1
                if progress_sink is not None and pending_progress >= progress_every:
                    progress_sink.put(int(pending_progress))
                    pending_progress = 0
                bg_count = int(snapshot.get("background_count", 0))
                background_sum += bg_count
                background_min = bg_count if background_min is None else min(background_min, bg_count)
                background_max = bg_count if background_max is None else max(background_max, bg_count)
                if len(buffers[key]) >= chunk_size:
                    _flush_chunk(temp_dir, worker_id, key, buffers, chunk_counts)
        else:
            first_record = True
            previous_key = None
            for local_index in range(count_per_offset):
                for offset_index, offset in enumerate(offsets_sorted):
                    key = _offset_key(offset)
                    strict_next = (not first_record) and key == previous_key
                    _advance_background_to_offset(env, offset, strict_next=strict_next)
                    snapshot = env.export_background_snapshot()
                    snapshot["worker_id"] = int(worker_id)
                    snapshot["worker_local_index"] = int(local_index)
                    snapshot["worker_offset_index"] = int(offset_index)
                    buffers[key].append(snapshot)
                    counts_by_offset[key] = counts_by_offset.get(key, 0) + 1
                    snapshot_count += 1
                    pending_progress += 1
                    if progress_sink is not None and pending_progress >= progress_every:
                        progress_sink.put(int(pending_progress))
                        pending_progress = 0
                    bg_count = int(snapshot.get("background_count", 0))
                    background_sum += bg_count
                    background_min = bg_count if background_min is None else min(background_min, bg_count)
                    background_max = bg_count if background_max is None else max(background_max, bg_count)
                    if len(buffers[key]) >= chunk_size:
                        _flush_chunk(temp_dir, worker_id, key, buffers, chunk_counts)
                    first_record = False
                    previous_key = key
        for key in list(buffers.keys()):
            _flush_chunk(temp_dir, worker_id, key, buffers, chunk_counts)
        if progress_sink is not None and pending_progress > 0:
            progress_sink.put(int(pending_progress))
    finally:
        env.close()

    return {
        "worker_id": int(worker_id),
        "offsets": [float(x) for x in offsets_sorted],
        "snapshot_count": int(snapshot_count),
        "counts_by_offset": counts_by_offset,
        "background_min": background_min,
        "background_max": background_max,
        "background_sum": int(background_sum),
        "config_signature": config_signature,
    }


def _split_counts(total: int, workers: int) -> list[int]:
    workers = max(1, int(workers))
    base = int(total) // workers
    rem = int(total) % workers
    return [base + (1 if i < rem else 0) for i in range(workers)]


def _update_collection_stats(result: dict[str, Any], stats: dict[str, Any]) -> None:
    stats["completed"] += int(result["snapshot_count"])
    for key, value in result["counts_by_offset"].items():
        stats["counts_by_offset"][key] = stats["counts_by_offset"].get(key, 0) + int(value)
    if result["background_min"] is not None:
        bg_min = int(result["background_min"])
        bg_max = int(result["background_max"])
        stats["bg_min"] = bg_min if stats["bg_min"] is None else min(stats["bg_min"], bg_min)
        stats["bg_max"] = bg_max if stats["bg_max"] is None else max(stats["bg_max"], bg_max)
        stats["bg_sum"] += int(result["background_sum"])


def _make_progress_state(
    total: int,
    desc: str,
    log_every: int,
    enabled: bool,
    unit: str,
) -> dict[str, Any]:
    pbar = None
    if enabled and tqdm is not None:
        pbar = tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True, ascii=True)
    elif enabled:
        print("tqdm is not installed; falling back to periodic progress logs")
    return {
        "completed": 0,
        "total": int(total),
        "log_every": max(int(log_every), 1),
        "next_log": max(int(log_every), 1),
        "pbar": pbar,
        "desc": desc,
    }


def _advance_progress(state: dict[str, Any], amount: int) -> None:
    amount = int(amount)
    if amount <= 0:
        return
    state["completed"] += amount
    pbar = state.get("pbar", None)
    if pbar is not None:
        pbar.update(amount)
        return

    total = int(state["total"])
    completed = int(state["completed"])
    if completed >= int(state["next_log"]) or completed >= total:
        print(f"{state['desc']}: {completed}/{total}")
        while state["next_log"] <= completed:
            state["next_log"] += int(state["log_every"])


def _close_progress(state: dict[str, Any] | None) -> None:
    if state is None:
        return
    pbar = state.get("pbar", None)
    if pbar is not None:
        pbar.close()
        state["pbar"] = None


class _LocalProgressSink:
    def __init__(self, state: dict[str, Any]):
        self.state = state

    def put(self, amount: int) -> None:
        _advance_progress(self.state, amount)


def _drain_progress_queue(progress_queue: Any, state: dict[str, Any]) -> None:
    if progress_queue is None:
        return
    while True:
        try:
            amount = progress_queue.get_nowait()
        except queue.Empty:
            break
        _advance_progress(state, int(amount))


def _write_final_shards(
    *,
    scenario_name: str,
    offsets: list[float],
    count_per_offset: int,
    temp_dir: Path,
    out_path: Path,
    config_signature: dict[str, Any] | None,
    progress_state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    shards: dict[str, dict[str, Any]] = {}
    for offset in offsets:
        key = _offset_key(offset)
        chunk_paths = sorted(temp_dir.glob(f"worker_*_offset_{key}_chunk_*.pkl"))
        offset_dir = out_path / f"offset_{key}"
        offset_dir.mkdir(parents=True, exist_ok=False)
        chunk_infos: list[dict[str, Any]] = []
        offset_count = 0
        offset_bg_sum = 0
        offset_bg_min = None
        offset_bg_max = None
        for chunk_path in chunk_paths:
            with chunk_path.open("rb") as f:
                chunk = pickle.load(f)
            if not isinstance(chunk, list):
                raise ValueError(f"Invalid chunk payload in {chunk_path}")
            chunk.sort(
                key=lambda item: (
                    int(item.get("worker_id", 0)),
                    int(item.get("worker_local_index", 0)),
                    int(item.get("worker_offset_index", 0)),
                )
            )
            chunk_bg_counts = [int(item.get("background_count", 0)) for item in chunk]
            chunk_file = offset_dir / f"chunk_{len(chunk_infos):06d}.pkl"
            chunk_payload = {
                "version": 3,
                "scenario_name": str(scenario_name),
                "offset": float(offset),
                "offset_key": key,
                "chunk_index": len(chunk_infos),
                "count": len(chunk),
                "config_signature": config_signature,
                "offset_key_scale": SNAPSHOT_OFFSET_KEY_SCALE,
                "snapshots": chunk,
            }
            with chunk_file.open("wb") as f:
                pickle.dump(chunk_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            chunk_info = {
                "file": f"{offset_dir.name}/{chunk_file.name}",
                "count": len(chunk),
                "background_count_min": min(chunk_bg_counts) if chunk_bg_counts else None,
                "background_count_max": max(chunk_bg_counts) if chunk_bg_counts else None,
                "background_count_mean": (
                    sum(chunk_bg_counts) / len(chunk_bg_counts)
                )
                if chunk_bg_counts
                else None,
            }
            chunk_infos.append(chunk_info)
            offset_count += len(chunk)
            offset_bg_sum += sum(chunk_bg_counts)
            if chunk_bg_counts:
                cmin = min(chunk_bg_counts)
                cmax = max(chunk_bg_counts)
                offset_bg_min = cmin if offset_bg_min is None else min(offset_bg_min, cmin)
                offset_bg_max = cmax if offset_bg_max is None else max(offset_bg_max, cmax)

        if offset_count != count_per_offset:
            raise RuntimeError(
                f"Expected {count_per_offset} snapshots for offset {offset}, got {offset_count}"
            )

        shards[key] = {
            "format": "chunks",
            "offset": float(offset),
            "count": offset_count,
            "chunks": chunk_infos,
            "background_count_min": offset_bg_min,
            "background_count_max": offset_bg_max,
            "background_count_mean": (offset_bg_sum / offset_count) if offset_count else None,
        }
        _advance_progress(progress_state, 1)
    return shards


def _snapshot_lane_id(value: Any) -> int | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    if value[2] is None:
        return None
    try:
        return int(value[2])
    except (TypeError, ValueError):
        return None


def _snapshot_has_eligible_ego_candidate(
    snapshot: dict[str, Any],
    *,
    x_range: tuple[float, float],
    speed_range: tuple[float, float],
    lanes_count: int,
) -> bool:
    vehicles = snapshot.get("vehicles", [])
    if not isinstance(vehicles, list):
        return False
    x_min, x_max = x_range
    speed_min, speed_max = speed_range
    for data in vehicles:
        if not isinstance(data, dict) or bool(data.get("crashed", False)):
            continue
        lane_id = _snapshot_lane_id(data.get("lane_index", None))
        if lane_id is None or not (0 <= lane_id < lanes_count):
            continue
        pos = data.get("position", None)
        if pos is None:
            continue
        try:
            x = float(np.asarray(pos, dtype=float).reshape(-1)[0])
            speed = float(data.get("speed", 0.0))
        except (TypeError, ValueError, IndexError):
            continue
        if not (np.isfinite(x) and np.isfinite(speed)):
            continue
        if x_min <= x <= x_max and speed_min <= speed <= speed_max:
            return True
    return False


def _write_filtered_chunk(
    *,
    out_path: Path,
    offset_key: str,
    offset: float,
    chunk_index: int,
    snapshots: list[dict[str, Any]],
    scenario_name: str,
    config_signature: dict[str, Any] | None,
) -> dict[str, Any]:
    offset_dir = out_path / f"offset_{offset_key}"
    offset_dir.mkdir(parents=True, exist_ok=True)
    chunk_file = offset_dir / f"chunk_{chunk_index:06d}.pkl"
    chunk_bg_counts = [int(item.get("background_count", 0)) for item in snapshots]
    payload = {
        "version": 3,
        "scenario_name": str(scenario_name),
        "offset": float(offset),
        "offset_key": str(offset_key),
        "chunk_index": int(chunk_index),
        "count": len(snapshots),
        "config_signature": config_signature,
        "offset_key_scale": SNAPSHOT_OFFSET_KEY_SCALE,
        "snapshots": snapshots,
    }
    with chunk_file.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "file": f"{offset_dir.name}/{chunk_file.name}",
        "count": len(snapshots),
        "background_count_min": min(chunk_bg_counts) if chunk_bg_counts else None,
        "background_count_max": max(chunk_bg_counts) if chunk_bg_counts else None,
        "background_count_mean": (
            sum(chunk_bg_counts) / len(chunk_bg_counts)
        )
        if chunk_bg_counts
        else None,
    }


def build_filtered_snapshot_pool(config: dict[str, Any] = SNAPSHOT_POOL_CONFIG) -> None:
    source_path = Path(config.get("filter_source_path", config.get("source_path", "")))
    out_path = Path(config.get("filter_out_path", config.get("out_path", "")))
    if not source_path.exists():
        raise FileNotFoundError(f"filter_source_path does not exist: {source_path}")
    if not source_path.is_dir():
        raise ValueError("filter_source_path must be a sharded snapshot pool directory")
    if not out_path:
        raise ValueError("filter_out_path must be set")

    overwrite = bool(config.get("filter_overwrite", config.get("overwrite", True)))
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"{out_path} already exists and overwrite=False")
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()
    out_path.mkdir(parents=True, exist_ok=False)

    x_raw = config.get("filter_ego_x_range", [0.0, float("inf")])
    speed_raw = config.get("filter_ego_speed_range", [0.0, float("inf")])
    if not isinstance(x_raw, (list, tuple)) or len(x_raw) != 2:
        raise ValueError("filter_ego_x_range must be a two-item list/tuple")
    if not isinstance(speed_raw, (list, tuple)) or len(speed_raw) != 2:
        raise ValueError("filter_ego_speed_range must be a two-item list/tuple")
    x_range = (float(x_raw[0]), float(x_raw[1]))
    speed_range = (float(speed_raw[0]), float(speed_raw[1]))
    chunk_size = max(1, int(config.get("filter_chunk_size", config.get("chunk_size", 500))))

    with (source_path / "meta.pkl").open("rb") as f:
        meta = pickle.load(f)
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid source snapshot metadata: {source_path / 'meta.pkl'}")
    scenario_name = str(meta.get("scenario_name", config.get("scenario_name", "")))
    config_signature = meta.get("config_signature", None)
    lanes_count = int(
        (config_signature or {}).get(
            "lanes_count",
            config.get("filter_lanes_count", 3),
        )
    )
    shards_src = meta.get("shards", {})
    if not isinstance(shards_src, dict) or not shards_src:
        raise ValueError(f"No source shards found in {source_path}")

    selected_offset_keys = _configured_filter_offset_keys(config)
    if selected_offset_keys is None:
        shard_items = list(shards_src.items())
    else:
        missing_keys = sorted(key for key in selected_offset_keys if key not in shards_src)
        if missing_keys:
            raise ValueError(
                "filter_offsets/filter_offset_keys requested missing source shard keys: "
                f"{missing_keys}; available keys include {sorted(shards_src)[:10]}"
            )
        shard_items = [
            (key, shard)
            for key, shard in shards_src.items()
            if str(key) in selected_offset_keys
        ]
    if not shard_items:
        raise ValueError("No source shards selected by filter_offsets/filter_offset_keys")

    total_chunks = sum(
        len(shard.get("chunks", []))
        for _, shard in shard_items
        if isinstance(shard, dict) and isinstance(shard.get("chunks", None), list)
    )
    progress = _make_progress_state(
        total_chunks,
        "filtering chunks",
        max(1, int(config.get("log_every", 100))),
        bool(config.get("use_tqdm", True)),
        "chunk",
    )

    shards_out: dict[str, dict[str, Any]] = {}
    total_in = 0
    total_out = 0
    try:
        for offset_key, shard in shard_items:
            if not isinstance(shard, dict):
                continue
            chunks_src = shard.get("chunks", [])
            if not isinstance(chunks_src, list):
                continue
            offset = float(shard.get("offset", 0.0))
            out_chunk_infos: list[dict[str, Any]] = []
            buffer: list[dict[str, Any]] = []
            offset_out = 0
            offset_in = 0
            offset_bg_sum = 0
            offset_bg_min = None
            offset_bg_max = None

            for chunk_info in chunks_src:
                if not isinstance(chunk_info, dict):
                    continue
                chunk_file = str(chunk_info.get("file", ""))
                if not chunk_file:
                    continue
                with (source_path / chunk_file).open("rb") as f:
                    payload = pickle.load(f)
                snapshots = payload.get("snapshots", None) if isinstance(payload, dict) else None
                if not isinstance(snapshots, list):
                    raise ValueError(f"Invalid source chunk payload: {source_path / chunk_file}")
                offset_in += len(snapshots)
                total_in += len(snapshots)
                for snapshot in snapshots:
                    if not isinstance(snapshot, dict):
                        continue
                    if _snapshot_has_eligible_ego_candidate(
                        snapshot,
                        x_range=x_range,
                        speed_range=speed_range,
                        lanes_count=lanes_count,
                    ):
                        buffer.append(snapshot)
                        if len(buffer) >= chunk_size:
                            info = _write_filtered_chunk(
                                out_path=out_path,
                                offset_key=str(offset_key),
                                offset=offset,
                                chunk_index=len(out_chunk_infos),
                                snapshots=buffer,
                                scenario_name=scenario_name,
                                config_signature=config_signature,
                            )
                            out_chunk_infos.append(info)
                            bg_counts = [
                                int(item.get("background_count", 0))
                                for item in buffer
                            ]
                            offset_out += len(buffer)
                            total_out += len(buffer)
                            offset_bg_sum += sum(bg_counts)
                            if bg_counts:
                                bmin = min(bg_counts)
                                bmax = max(bg_counts)
                                offset_bg_min = bmin if offset_bg_min is None else min(offset_bg_min, bmin)
                                offset_bg_max = bmax if offset_bg_max is None else max(offset_bg_max, bmax)
                            buffer = []
                _advance_progress(progress, 1)

            if buffer:
                info = _write_filtered_chunk(
                    out_path=out_path,
                    offset_key=str(offset_key),
                    offset=offset,
                    chunk_index=len(out_chunk_infos),
                    snapshots=buffer,
                    scenario_name=scenario_name,
                    config_signature=config_signature,
                )
                out_chunk_infos.append(info)
                bg_counts = [int(item.get("background_count", 0)) for item in buffer]
                offset_out += len(buffer)
                total_out += len(buffer)
                offset_bg_sum += sum(bg_counts)
                if bg_counts:
                    bmin = min(bg_counts)
                    bmax = max(bg_counts)
                    offset_bg_min = bmin if offset_bg_min is None else min(offset_bg_min, bmin)
                    offset_bg_max = bmax if offset_bg_max is None else max(offset_bg_max, bmax)

            if out_chunk_infos:
                shards_out[str(offset_key)] = {
                    "format": "chunks",
                    "offset": offset,
                    "count": offset_out,
                    "source_count": offset_in,
                    "chunks": out_chunk_infos,
                    "background_count_min": offset_bg_min,
                    "background_count_max": offset_bg_max,
                    "background_count_mean": (
                        offset_bg_sum / offset_out
                    )
                    if offset_out
                    else None,
                }
    finally:
        _close_progress(progress)

    if total_out <= 0:
        raise RuntimeError(
            "Filtered snapshot pool is empty; relax filter_ego_x_range/"
            "filter_ego_speed_range or check source pool"
        )

    offsets_out = [float(shard["offset"]) for shard in shards_out.values()]
    meta_out = {
        "version": 3,
        "format": "offset_chunk_shards",
        "scenario_name": scenario_name,
        "seed": meta.get("seed", None),
        "source_path": str(source_path),
        "source_format": meta.get("format", None),
        "source_count": total_in,
        "filtered_count": total_out,
        "filter": {
            "ego_x_range": [float(x_range[0]), float(x_range[1])],
            "ego_speed_range": [float(speed_range[0]), float(speed_range[1])],
            "offset_keys": (
                sorted(selected_offset_keys) if selected_offset_keys is not None else None
            ),
        },
        "offsets": offsets_out,
        "count_per_offset": None,
        "workers": 1,
        "config_signature": config_signature,
        "offset_key_scale": SNAPSHOT_OFFSET_KEY_SCALE,
        "shards": shards_out,
    }
    with (out_path / "meta.pkl").open("wb") as f:
        pickle.dump(meta_out, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"filtered {total_out}/{total_in} snapshots into {out_path.resolve()}")
    print(f"kept shards: {len(shards_out)}/{len(shard_items)} selected, {len(shards_src)} source")


def build_snapshot_pool(config: dict[str, Any] = SNAPSHOT_POOL_CONFIG) -> None:
    if bool(config.get("filter_existing_pool_enabled", False)):
        build_filtered_snapshot_pool(config)
        return

    scenario_name = str(config.get("scenario_name", "multi_lane_stop_to_int"))
    offsets = [0.0] if scenario_name == "multi_lane" else _configured_offsets(config)
    count_per_offset = int(config["count_per_offset"])
    if count_per_offset <= 0:
        raise ValueError("SNAPSHOT_POOL_CONFIG['count_per_offset'] must be positive")

    requested_workers = int(config.get("workers", 1))
    workers = max(1, min(requested_workers, os.cpu_count() or 1, count_per_offset))
    seed = int(config.get("seed", MASTER_SEED))
    log_every = max(int(config.get("log_every", 100)), 1)
    total_expected = len(offsets) * count_per_offset
    show_progress = bool(config.get("use_tqdm", True))

    out_path = Path(config["out_path"])
    temp_dir = out_path.parent / f".{out_path.name}.chunks_{uuid.uuid4().hex}"
    if out_path.exists():
        if not bool(config.get("overwrite", True)):
            raise FileExistsError(f"{out_path} already exists and overwrite=False")
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()
    out_path.mkdir(parents=True, exist_ok=False)
    temp_dir.mkdir(parents=True, exist_ok=False)

    collect_progress = _make_progress_state(
        total_expected,
        "collecting snapshots",
        log_every,
        show_progress,
        "snap",
    )
    progress_manager = None
    progress_queue = None
    if workers > 1 and show_progress:
        progress_manager = Manager()
        progress_queue = progress_manager.Queue()
    local_progress_sink = _LocalProgressSink(collect_progress) if workers == 1 and show_progress else None

    counts = _split_counts(count_per_offset, workers)
    tasks: list[tuple[int, tuple[float, ...], int, int, dict[str, Any], str, Any]] = []
    for worker_id, count in enumerate(counts):
        if count <= 0:
            continue
        worker_seed = seed + int(worker_id)
        tasks.append(
            (
                int(worker_id),
                tuple(float(offset) for offset in offsets),
                int(count),
                int(worker_seed),
                copy.deepcopy(config),
                str(temp_dir),
                progress_queue if workers > 1 else local_progress_sink,
            )
        )

    print(
        f"building scenario={scenario_name}, {len(offsets)} offset(s), "
        f"{count_per_offset} snapshots/offset "
        f"with {workers} continuous worker stream(s)"
    )
    if scenario_name == "multi_lane":
        print(
            "old multi_lane has no signal offset; recording one continuous stream "
            f"every {float(config.get('snapshot_interval_seconds', 50.0)):.3f}s"
        )
    config_signature = None
    stats: dict[str, Any] = {
        "completed": 0,
        "counts_by_offset": {_offset_key(offset): 0 for offset in offsets},
        "bg_min": None,
        "bg_max": None,
        "bg_sum": 0,
    }

    try:
        if workers == 1:
            for task in tasks:
                result = _collect_worker(task)
                config_signature = result["config_signature"] or config_signature
                _update_collection_stats(result, stats)
                if not show_progress and (
                    stats["completed"] % log_every == 0 or stats["completed"] == total_expected
                ):
                    print(f"collected {stats['completed']}/{total_expected}")
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_collect_worker, task) for task in tasks]
                pending = set(futures)
                while pending:
                    done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                    _drain_progress_queue(progress_queue, collect_progress)
                    for future in done:
                        result = future.result()
                        config_signature = result["config_signature"] or config_signature
                        _update_collection_stats(result, stats)
                        if not show_progress and (
                            stats["completed"] % log_every == 0
                            or stats["completed"] == total_expected
                        ):
                            print(f"collected {stats['completed']}/{total_expected}")
                _drain_progress_queue(progress_queue, collect_progress)

        if show_progress and int(collect_progress["completed"]) < int(stats["completed"]):
            _advance_progress(
                collect_progress,
                int(stats["completed"]) - int(collect_progress["completed"]),
            )
        _close_progress(collect_progress)
        if progress_manager is not None:
            progress_manager.shutdown()
            progress_manager = None

        if stats["completed"] != total_expected:
            raise RuntimeError(f"Expected {total_expected} snapshots, got {stats['completed']}")

        merge_progress = _make_progress_state(
            len(offsets),
            "writing shards",
            max(1, min(log_every, len(offsets))),
            show_progress,
            "offset",
        )
        try:
            shards = _write_final_shards(
                scenario_name=scenario_name,
                offsets=offsets,
                count_per_offset=count_per_offset,
                temp_dir=temp_dir,
                out_path=out_path,
                config_signature=config_signature,
                progress_state=merge_progress,
            )
        finally:
            _close_progress(merge_progress)

        meta = {
            "version": 3,
            "format": "offset_chunk_shards",
            "scenario_name": scenario_name,
            "seed": seed,
            "offsets": [float(x) for x in offsets],
            "count_per_offset": count_per_offset,
            "workers": workers,
            "config_signature": config_signature,
            "offset_key_scale": SNAPSHOT_OFFSET_KEY_SCALE,
            "shards": shards,
        }
        with (out_path / "meta.pkl").open("wb") as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        _close_progress(collect_progress)
        if progress_manager is not None:
            progress_manager.shutdown()
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    mean_bg = stats["bg_sum"] / max(stats["completed"], 1)
    print(f"wrote {stats['completed']} snapshots to sharded pool {out_path.resolve()}")
    print(f"per-offset counts: {stats['counts_by_offset']}")
    print(f"background vehicles: min={stats['bg_min']} mean={mean_bg:.2f} max={stats['bg_max']}")


if __name__ == "__main__":
    build_snapshot_pool()
