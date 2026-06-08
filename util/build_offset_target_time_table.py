from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from configs.conf import MASTER_SEED, get_env_config_for_scenario
from custom_env.vehicle.behavior import IDMVehicle
from scenarios.multi_lane_stop_to_int.scenario import MultiLaneStopToIntEnv

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


# =========================
# Manual test configuration
# =========================

TEST_CONFIG = {
    "offset_start": 0.0,
    "offset_stop": 99.0,
    "offset_step": 1.0,
    "lanes": [0, 1, 2],
    "seed": MASTER_SEED,
    "episodes": 10,
    "workers": 0,  # 0: min(cpu_count, 8); 1: serial
    "target_speed": 12.0,
    "spawn_probability": None,  # None: use scenario default
    "warmup_time": None,  # None: use scenario default
    "duration": 160.0,
    "out_dir": Path("debug") / "offset_target_time_table_with_traffic",
}


@dataclass(frozen=True)
class RolloutResult:
    offset: float
    initial_lane_id: int
    seed: int
    episode_index: int
    target_time: float | None
    final_x: float
    final_speed: float
    signal_state_at_start: str
    signal_remaining_at_start: float
    background_count_at_start: int
    queue_count_at_start: int
    same_lane_front_gap_at_start: float | None
    terminated: bool
    truncated: bool
    crashed: bool


@dataclass(frozen=True)
class RolloutOutput:
    result: RolloutResult
    trajectory: tuple[tuple[float, float, float, float], ...]


def _float_grid(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("TEST_CONFIG['offset_step'] must be > 0")
    values: list[float] = []
    x = float(start)
    eps = step * 1e-6
    while x <= float(stop) + eps:
        values.append(round(x, 10))
        x += step
    return values


def _replace_ego_with_idm(env: MultiLaneStopToIntEnv, target_speed: float) -> IDMVehicle:
    old = env.vehicle
    lane_index = old.lane_index
    lane = env.road.network.get_lane(lane_index)
    ego = IDMVehicle(
        env.road,
        np.asarray(old.position, dtype=float).copy(),
        heading=float(old.heading),
        speed=float(old.speed),
        target_lane_index=lane_index,
        target_speed=float(target_speed),
        enable_lane_change=False,
    )
    ego.lane = lane
    ego.lane_index = lane_index
    ego.target_lane_index = lane_index

    try:
        idx = env.road.vehicles.index(old)
        env.road.vehicles[idx] = ego
    except ValueError:
        env.road.vehicles.append(ego)

    env.vehicle = ego
    env.controlled_vehicles = [ego]
    env.action_type.controlled_vehicle = ego
    env._last_speed = float(ego.speed)
    env._last_acc = 0.0
    env._last_longitudinal = float(ego.position[0])
    env._last_lane_id = int(lane_index[2])
    env._has_arrived = False
    env._arrival_time = None
    env._update_signal_virtual_stops(query_time=0.0)
    return ego


def _signal_state_for_lane(env: MultiLaneStopToIntEnv, lane_id: int) -> tuple[str, float]:
    controller = env._signal_controller
    if controller is None:
        return "unknown", float("nan")
    phase = controller.phase_at(float(env._signal_time_global))
    direction = controller.lane_direction(int(lane_id))
    item = phase.get(direction, {"state": "unknown", "remaining": np.nan})
    return str(item["state"]), float(item["remaining"])


def _traffic_diagnostics(env: MultiLaneStopToIntEnv) -> tuple[int, int, float | None]:
    ego = env.vehicle
    lane_index = ego.lane_index
    lane = env.road.network.get_lane(lane_index)
    ego_s, _ = lane.local_coordinates(ego.position)
    goal_x = float(env._goal_longitudinal())
    background = [v for v in env.road.vehicles if v not in env.controlled_vehicles]

    queue_count = 0
    same_lane_front_gap: float | None = None
    for v in background:
        if getattr(v, "crashed", False):
            continue
        li = getattr(v, "lane_index", None)
        if li is None or len(li) < 3:
            continue
        speed = float(getattr(v, "speed", np.linalg.norm(getattr(v, "velocity", np.zeros(2)))))
        x = float(np.asarray(getattr(v, "position", [np.nan, np.nan]), dtype=float)[0])
        if goal_x - 40.0 <= x <= goal_x + 5.0 and speed < 2.0:
            queue_count += 1
        if li[0] == lane_index[0] and li[1] == lane_index[1] and int(li[2]) == int(lane_index[2]):
            s_v, _ = lane.local_coordinates(v.position)
            gap = float(s_v - ego_s)
            if gap >= 0.0 and (same_lane_front_gap is None or gap < same_lane_front_gap):
                same_lane_front_gap = gap

    return int(len(background)), int(queue_count), same_lane_front_gap


def run_condition(
    offset: float,
    initial_lane_id: int,
    seed: int,
    episodes: int,
    target_speed: float,
    spawn_probability: float | None,
    duration: float,
    warmup_time: float | None,
) -> tuple[RolloutOutput, ...]:
    overrides = {
        "episode_start_phase_offset": float(offset),
        "initial_lane_id": int(initial_lane_id),
        "ego_speed": float(target_speed),
        "warmup_each_episode": False,
        "inter_episode_as_steps": False,
        "duration": float(duration),
    }
    if spawn_probability is not None:
        overrides["spawn_probability"] = float(spawn_probability)
    if warmup_time is not None:
        overrides["warmup_time"] = float(warmup_time)
    cfg = get_env_config_for_scenario("multi_lane_stop_to_int", overrides)
    env = MultiLaneStopToIntEnv(config=cfg)
    try:
        outputs: list[RolloutOutput] = []
        for episode_index in range(int(episodes)):
            if episode_index == 0:
                env.reset(seed=int(seed))
            else:
                env.reset()
            _replace_ego_with_idm(env, target_speed=target_speed)
            state, remaining = _signal_state_for_lane(env, initial_lane_id)
            bg_count, queue_count, same_lane_front_gap = _traffic_diagnostics(env)
            trajectory: list[tuple[float, float, float, float]] = [
                (
                    float(env.time),
                    float(env.vehicle.position[0]),
                    float(env.vehicle.position[1]),
                    float(env.vehicle.speed),
                )
            ]

            action = np.zeros(env.action_space.shape, dtype=np.float32)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                _, _, terminated, truncated, _ = env.step(action)
                trajectory.append(
                    (
                        float(env.time),
                        float(env.vehicle.position[0]),
                        float(env.vehicle.position[1]),
                        float(env.vehicle.speed),
                    )
                )

            target_time = float(env.time) if env._goal_longitudinal_reached() else None
            outputs.append(
                RolloutOutput(
                    result=RolloutResult(
                        offset=float(offset),
                        initial_lane_id=int(initial_lane_id),
                        seed=int(seed),
                        episode_index=int(episode_index),
                        target_time=target_time,
                        final_x=float(env.vehicle.position[0]),
                        final_speed=float(env.vehicle.speed),
                        signal_state_at_start=state,
                        signal_remaining_at_start=remaining,
                        background_count_at_start=bg_count,
                        queue_count_at_start=queue_count,
                        same_lane_front_gap_at_start=same_lane_front_gap,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        crashed=bool(env.vehicle.crashed),
                    ),
                    trajectory=tuple(trajectory),
                )
            )
        return tuple(outputs)
    finally:
        env.close()


def _run_task(
    task: tuple[float, int, int, int, float, float | None, float, float | None],
) -> tuple[RolloutOutput, ...]:
    return run_condition(*task)


def write_csv(path: Path, rows: Iterable[RolloutResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "offset",
        "initial_lane_id",
        "seed",
        "episode_index",
        "target_time",
        "final_x",
        "final_speed",
        "signal_state_at_start",
        "signal_remaining_at_start",
        "background_count_at_start",
        "queue_count_at_start",
        "same_lane_front_gap_at_start",
        "terminated",
        "truncated",
        "crashed",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = row.__dict__.copy()
            data["target_time"] = "" if row.target_time is None else f"{row.target_time:.3f}"
            data["final_x"] = f"{row.final_x:.3f}"
            data["final_speed"] = f"{row.final_speed:.3f}"
            data["signal_remaining_at_start"] = f"{row.signal_remaining_at_start:.3f}"
            data["same_lane_front_gap_at_start"] = (
                "" if row.same_lane_front_gap_at_start is None else f"{row.same_lane_front_gap_at_start:.3f}"
            )
            writer.writerow(data)


def write_markdown_pivot(path: Path, rows: list[RolloutResult], lane_ids: list[int]) -> None:
    by_key = {(r.offset, r.initial_lane_id): r for r in rows if r.target_time is not None}
    offsets = sorted({r.offset for r in rows})
    with path.open("w", encoding="utf-8") as f:
        f.write("| episode_start_phase_offset | " + " | ".join(f"lane {i}" for i in lane_ids) + " |\n")
        f.write("|---" + "|---" * len(lane_ids) + "|\n")
        for offset in offsets:
            vals = []
            for lane_id in lane_ids:
                row = by_key.get((offset, lane_id))
                vals.append("" if row is None else f"{row.target_time:.1f}")
            f.write(f"| {offset:.1f} | " + " | ".join(vals) + " |\n")


def write_summary_csv(path: Path, rows: list[RolloutResult]) -> list[dict[str, float | int]]:
    summary: list[dict[str, float | int]] = []
    keys = sorted({(row.offset, row.initial_lane_id) for row in rows})
    for offset, lane_id in keys:
        subset = [row for row in rows if row.offset == offset and row.initial_lane_id == lane_id]
        target_times = np.asarray(
            [float(row.target_time) for row in subset if row.target_time is not None],
            dtype=float,
        )
        front_gaps = np.asarray(
            [
                float(row.same_lane_front_gap_at_start)
                for row in subset
                if row.same_lane_front_gap_at_start is not None
            ],
            dtype=float,
        )
        summary.append(
            {
                "offset": float(offset),
                "initial_lane_id": int(lane_id),
                "seed": int(subset[0].seed),
                "episode_count": int(len(subset)),
                "success_count": int(target_times.size),
                "target_time_mean": float(np.mean(target_times)) if target_times.size else np.nan,
                "target_time_std": float(np.std(target_times)) if target_times.size else np.nan,
                "target_time_min": float(np.min(target_times)) if target_times.size else np.nan,
                "target_time_max": float(np.max(target_times)) if target_times.size else np.nan,
                "background_count_at_start_mean": float(
                    np.mean([row.background_count_at_start for row in subset])
                ),
                "queue_count_at_start_mean": float(np.mean([row.queue_count_at_start for row in subset])),
                "same_lane_front_gap_at_start_mean": (
                    float(np.mean(front_gaps)) if front_gaps.size else np.nan
                ),
                "crash_count": int(sum(row.crashed for row in subset)),
                "truncated_count": int(sum(row.truncated for row in subset)),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        if summary:
            writer.writeheader()
            writer.writerows(summary)
    return summary


def write_summary_markdown(
    path: Path,
    summary: list[dict[str, float | int]],
    lane_ids: list[int],
) -> None:
    by_key = {
        (float(row["offset"]), int(row["initial_lane_id"])): float(row["target_time_mean"])
        for row in summary
    }
    offsets = sorted({float(row["offset"]) for row in summary})
    with path.open("w", encoding="utf-8") as f:
        f.write("| episode_start_phase_offset | " + " | ".join(f"lane {i}" for i in lane_ids) + " |\n")
        f.write("|---" + "|---" * len(lane_ids) + "|\n")
        for offset in offsets:
            values = [by_key.get((offset, lane_id), np.nan) for lane_id in lane_ids]
            f.write(
                f"| {offset:.1f} | "
                + " | ".join("" if not np.isfinite(value) else f"{value:.2f}" for value in values)
                + " |\n"
            )


def write_raw_trajectory_row(
    writer: csv.DictWriter,
    result: RolloutResult,
    point: tuple[float, float, float, float],
) -> None:
    time_s, x, y, speed = point
    writer.writerow(
        {
            "offset": f"{result.offset:.3f}",
            "initial_lane_id": result.initial_lane_id,
            "seed": result.seed,
            "episode_index": result.episode_index,
            "time": f"{time_s:.3f}",
            "absolute_time": f"{result.offset + time_s:.3f}",
            "ego_x": f"{x:.6f}",
            "ego_y": f"{y:.6f}",
            "ego_speed": f"{speed:.6f}",
        }
    )


def write_mean_trajectories(
    path: Path,
    grouped: dict[
        tuple[float, int],
        list[tuple[int, tuple[tuple[float, float, float, float], ...]]],
    ],
) -> None:
    fieldnames = [
        "offset",
        "initial_lane_id",
        "time",
        "absolute_time",
        "ego_x_mean",
        "ego_x_std",
        "ego_y_mean",
        "ego_speed_mean",
        "episode_count",
        "active_episode_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (offset, lane_id), trajectories in sorted(grouped.items()):
            if not trajectories:
                continue
            max_len = max(len(points) for _, points in trajectories)
            for index in range(max_len):
                padded_points = [points[min(index, len(points) - 1)] for _, points in trajectories]
                time_s = max(point[0] for point in padded_points)
                xs = np.asarray([point[1] for point in padded_points], dtype=float)
                ys = np.asarray([point[2] for point in padded_points], dtype=float)
                speeds = np.asarray([point[3] for point in padded_points], dtype=float)
                writer.writerow(
                    {
                        "offset": f"{offset:.3f}",
                        "initial_lane_id": lane_id,
                        "time": f"{time_s:.3f}",
                        "absolute_time": f"{offset + time_s:.3f}",
                        "ego_x_mean": f"{np.mean(xs):.6f}",
                        "ego_x_std": f"{np.std(xs):.6f}",
                        "ego_y_mean": f"{np.mean(ys):.6f}",
                        "ego_speed_mean": f"{np.mean(speeds):.6f}",
                        "episode_count": len(trajectories),
                        "active_episode_count": sum(index < len(points) for _, points in trajectories),
                    }
                )


def main() -> None:
    cfg = dict(TEST_CONFIG)
    episodes = int(cfg["episodes"])
    if episodes <= 0:
        raise ValueError("TEST_CONFIG['episodes'] must be > 0")

    offsets = _float_grid(
        float(cfg["offset_start"]),
        float(cfg["offset_stop"]),
        float(cfg["offset_step"]),
    )
    lane_ids = [int(x) for x in cfg["lanes"]]
    if not lane_ids:
        raise ValueError("TEST_CONFIG['lanes'] must contain at least one lane id")

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            float(offset),
            int(lane_id),
            int(cfg["seed"]),
            episodes,
            float(cfg["target_speed"]),
            cfg["spawn_probability"],
            float(cfg["duration"]),
            cfg["warmup_time"],
        )
        for offset in offsets
        for lane_id in lane_ids
    ]
    total_conditions = len(tasks)
    total_episodes = total_conditions * episodes
    workers = int(cfg["workers"])
    if workers <= 0:
        workers = min(os.cpu_count() or 1, 8)
    workers = max(1, min(workers, total_conditions))

    rows: list[RolloutResult] = []
    grouped_trajectories: dict[
        tuple[float, int],
        list[tuple[int, tuple[tuple[float, float, float, float], ...]]],
    ] = {}
    trajectory_raw_path = out_dir / "ego_position_time_raw.csv"
    trajectory_fields = [
        "offset",
        "initial_lane_id",
        "seed",
        "episode_index",
        "time",
        "absolute_time",
        "ego_x",
        "ego_y",
        "ego_speed",
    ]

    print(
        f"running {total_conditions} conditions x {episodes} episodes "
        f"({total_episodes} episodes total) with {workers} worker process(es)"
    )
    with trajectory_raw_path.open("w", newline="", encoding="utf-8") as trajectory_file:
        trajectory_writer = csv.DictWriter(trajectory_file, fieldnames=trajectory_fields)
        trajectory_writer.writeheader()

        progress = (
            tqdm(
                total=total_conditions,
                desc="Simulating conditions",
                unit="condition",
                dynamic_ncols=True,
            )
            if tqdm is not None
            else None
        )

        def collect_outputs(outputs: tuple[RolloutOutput, ...]) -> None:
            for output in outputs:
                rows.append(output.result)
                grouped_trajectories.setdefault(
                    (output.result.offset, output.result.initial_lane_id), []
                ).append((output.result.episode_index, output.trajectory))
                for point in output.trajectory:
                    write_raw_trajectory_row(trajectory_writer, output.result, point)

        try:
            if workers == 1:
                completed_conditions = (_run_task(task) for task in tasks)
                for done, outputs in enumerate(completed_conditions, start=1):
                    collect_outputs(outputs)
                    if progress is not None:
                        progress.update(1)
                        progress.set_postfix(episodes=f"{done * episodes}/{total_episodes}")
                    elif done % 25 == 0 or done == total_conditions:
                        print(f"completed {done}/{total_conditions} conditions")
            else:
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    future_to_task = {executor.submit(_run_task, task): task for task in tasks}
                    for done, future in enumerate(as_completed(future_to_task), start=1):
                        collect_outputs(future.result())
                        if progress is not None:
                            progress.update(1)
                            progress.set_postfix(episodes=f"{done * episodes}/{total_episodes}")
                        elif done % 25 == 0 or done == total_conditions:
                            print(f"completed {done}/{total_conditions} conditions")
        finally:
            if progress is not None:
                progress.close()

    rows.sort(key=lambda row: (row.offset, row.initial_lane_id, row.episode_index))
    raw_csv = out_dir / "offset_target_time_raw.csv"
    write_csv(raw_csv, rows)

    if episodes == 1:
        md_path = out_dir / "offset_target_time_pivot.md"
        write_markdown_pivot(md_path, rows, lane_ids)
        trajectory_mean_path = out_dir / "ego_position_time_mean.csv"
        write_mean_trajectories(trajectory_mean_path, grouped_trajectories)
        print(f"raw csv: {raw_csv.resolve()}")
        print(f"pivot md: {md_path.resolve()}")
        print(f"raw trajectories: {trajectory_raw_path.resolve()}")
        print(f"mean trajectories: {trajectory_mean_path.resolve()}")
        return

    summary_csv = out_dir / "offset_target_time_mean.csv"
    summary_md = out_dir / "offset_target_time_mean_pivot.md"
    summary = write_summary_csv(summary_csv, rows)
    write_summary_markdown(summary_md, summary, lane_ids)
    trajectory_mean_path = out_dir / "ego_position_time_mean.csv"
    write_mean_trajectories(trajectory_mean_path, grouped_trajectories)
    print(f"raw csv: {raw_csv.resolve()}")
    print(f"mean csv: {summary_csv.resolve()}")
    print(f"mean pivot md: {summary_md.resolve()}")
    print(f"raw trajectories: {trajectory_raw_path.resolve()}")
    print(f"mean trajectories: {trajectory_mean_path.resolve()}")


if __name__ == "__main__":
    main()
