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

from gymnasium.utils import seeding

from configs.builders import get_env_config_for_scenario
from configs.conf import MASTER_SEED, TRAIN_CONFIG
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
    "workers": 8,
    "seed": MASTER_SEED,
    "out_path": Path("debug") / "background_snapshot_pool_slowlane1",
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


def _seed_env_without_reset(env: MultiLaneStopToIntEnv, seed: int) -> None:
    env.np_random, _ = seeding.np_random(int(seed))
    if getattr(env, "road", None) is not None:
        env.road.np_random = env.np_random


def _make_env_config(offset: float, config: dict[str, Any]) -> dict:
    overrides: dict[str, Any] = {}
    if bool(config.get("use_train_env_overrides", True)):
        overrides = copy.deepcopy(
            TRAIN_CONFIG.get("config_overrides", {}).get("environment", {}) or {}
        )
    extra_overrides = copy.deepcopy(config.get("env_overrides", {}) or {})
    deep_update(overrides, extra_overrides)
    overrides["episode_start_phase_offset"] = float(offset)
    overrides["background_snapshot_reset"] = False
    overrides["inter_episode_as_steps"] = False
    overrides["align_ego_spawn_to_signal_offset"] = True
    if config.get("warmup_time", None) is not None:
        overrides["warmup_time"] = float(config["warmup_time"])
    return get_env_config_for_scenario("multi_lane_stop_to_int", overrides)


def _initialize_background_only_stream(env: MultiLaneStopToIntEnv, seed: int) -> None:
    """Create a road and evolve only background traffic; ego is never inserted."""
    _seed_env_without_reset(env, seed)
    env.time = 0.0
    env.steps = 0
    env.done = False
    env._signal_time_global = 0.0
    env._signal_episode_base = 0.0
    env._episodes_started = 0
    env._inter_episode_active = False
    env._inter_episode_remaining = 0.0
    env._background_only_sim_time = 0.0

    env._create_road()
    env.road.vehicles = []
    env.controlled_vehicles = []
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
    offsets_sorted = tuple(sorted(float(x) for x in offsets))
    chunk_size = max(int(config.get("chunk_size", 1000)), 1)
    progress_every = max(int(config.get("progress_every", 1)), 1)
    cfg = _make_env_config(offsets_sorted[0], config)
    env = MultiLaneStopToIntEnv(config=cfg)
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
                "scenario_name": "multi_lane_stop_to_int",
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


def build_snapshot_pool(config: dict[str, Any] = SNAPSHOT_POOL_CONFIG) -> None:
    offsets = _configured_offsets(config)
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
        f"building {len(offsets)} offset(s), {count_per_offset} snapshots/offset "
        f"with {workers} continuous worker stream(s)"
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
            "scenario_name": "multi_lane_stop_to_int",
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
