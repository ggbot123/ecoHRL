"""Summarize high-interval terminal goal errors near the end of training.

Training transition CSVs in these runs contain every high-level transition for
env0.  A tail window is selected by the transition terminal ``global_step``.
Signed errors are physical errors ``ego_terminal - goal``; absolute-error
means use ``abs(ego_terminal - goal)``. Standard deviations are population
standard deviations (ddof=0).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable


DEFAULT_RUNS = (
    "hiro_260708_highonly_reUni_oldLow_newEnv_2to1",
    "hiro_260706_highonly_uniOld_fixedHER_newEnv_2to1",
)


def _json_vector(raw: str) -> list[float]:
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"Expected JSON list, got {type(value).__name__}")
    return [float(item) for item in value]


def _lane_id(y: float, lane_centers: list[float]) -> int:
    return min(range(len(lane_centers)), key=lambda i: abs(y - lane_centers[i]))


def _target_lane(
    ego_y: float,
    y_code: float,
    lane_centers: list[float],
    *,
    dynamic_feasible_intervals: bool,
) -> int:
    lane = _lane_id(ego_y, lane_centers)
    target = lane
    last_lane = len(lane_centers) - 1
    if not dynamic_feasible_intervals:
        if y_code < -1.0 / 3.0 and lane > 0:
            target -= 1
        elif y_code > 1.0 / 3.0 and lane < last_lane:
            target += 1
    elif last_lane > 0:
        if 0 < lane < last_lane:
            if y_code < -1.0 / 3.0:
                target -= 1
            elif y_code > 1.0 / 3.0:
                target += 1
        elif lane == 0 and y_code > 0.0:
            target += 1
        elif lane == last_lane and y_code < 0.0:
            target -= 1
    return target


def _load_run_settings(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    environment = config["environment"]["env0_config"]
    hiro = config["hiro"]["config"]
    trainer = config["trainer"]
    lanes_count = int(environment.get("lanes_count", 3))
    lane_width = float(environment.get("lane_width", 4.0))
    return {
        "goal_longitudinal": float(environment.get("goal_longitudinal", 400.0)),
        "lane_width": lane_width,
        "lane_centers": [lane_width * lane for lane in range(lanes_count)],
        "dynamic_feasible_intervals": bool(
            hiro.get("high_goal_safety", {}).get("dynamic_feasible_lane_intervals", False)
        ),
        "n_envs": int(trainer.get("n_envs", 1)),
        "total_timesteps": int(trainer["total_timesteps"]),
        "captured_envs": str(trainer.get("high_transition_csv_envs", "env0")),
    }


def analyze_run(run_dir: Path, tail_steps: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = _load_run_settings(run_dir)
    if settings["captured_envs"] != "env0":
        raise ValueError(
            f"Expected env0 transition capture in {run_dir}, got {settings['captured_envs']}"
        )

    total_timesteps = int(settings["total_timesteps"])
    window_start = max(0, total_timesteps - int(tail_steps))
    goal_longitudinal = float(settings["goal_longitudinal"])
    lane_centers = list(settings["lane_centers"])
    dynamic = bool(settings["dynamic_feasible_intervals"])
    transition_path = run_dir / "high_interval_transitions.csv"
    details: list[dict[str, Any]] = []

    with transition_path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            global_step = int(row["global_step"])
            if not (window_start < global_step <= total_timesteps):
                continue

            high_obs = _json_vector(row["high_obs"])
            high_action = _json_vector(row["high_action"])
            next_high_obs = _json_vector(row["next_high_obs"])
            if len(high_obs) < 6 or len(high_action) < 3 or len(next_high_obs) < 6:
                raise ValueError(f"Malformed transition at global_step={global_step}")

            ego_start_x = goal_longitudinal - high_obs[2]
            ego_start_y = high_obs[3]
            goal_x = ego_start_x + high_action[0]
            goal_lane = _target_lane(
                ego_start_y,
                high_action[1],
                lane_centers,
                dynamic_feasible_intervals=dynamic,
            )
            goal_y = lane_centers[goal_lane]
            ego_terminal_x = goal_longitudinal - next_high_obs[2]
            ego_terminal_y = next_high_obs[3]
            signed_x_error = ego_terminal_x - goal_x
            signed_y_error = ego_terminal_y - goal_y
            terminal_lane = _lane_id(ego_terminal_y, lane_centers)

            details.append(
                {
                    "run": run_dir.name,
                    "global_step": global_step,
                    "env_id": int(row["env_id"]),
                    "segment_id": int(row["segment_id"]),
                    "done_env": int(row["done_env"]),
                    "ego_start_x": ego_start_x,
                    "ego_start_y": ego_start_y,
                    "goal_x": goal_x,
                    "goal_y": goal_y,
                    "ego_terminal_x": ego_terminal_x,
                    "ego_terminal_y": ego_terminal_y,
                    "signed_terminal_x_error_ego_minus_goal": signed_x_error,
                    "signed_terminal_y_error_ego_minus_goal": signed_y_error,
                    "abs_terminal_x_error": abs(signed_x_error),
                    "abs_terminal_y_error": abs(signed_y_error),
                    "goal_lane": goal_lane,
                    "ego_terminal_lane": terminal_lane,
                    "failed_lane_mismatch": int(goal_lane != terminal_lane),
                }
            )

    if not details:
        raise ValueError(f"No transitions in ({window_start}, {total_timesteps}] for {run_dir}")

    x_errors = [float(row["signed_terminal_x_error_ego_minus_goal"]) for row in details]
    y_errors = [float(row["signed_terminal_y_error_ego_minus_goal"]) for row in details]
    failures = sum(int(row["failed_lane_mismatch"]) for row in details)
    summary = {
        "run": run_dir.name,
        "tail_global_low_steps": int(tail_steps),
        "window_start_exclusive": window_start,
        "window_end_inclusive": total_timesteps,
        "first_saved_terminal_step": min(int(row["global_step"]) for row in details),
        "last_saved_terminal_step": max(int(row["global_step"]) for row in details),
        "captured_envs": settings["captured_envs"],
        "training_n_envs": settings["n_envs"],
        "intervals": len(details),
        "mean_signed_terminal_x_error_ego_minus_goal": fmean(x_errors),
        "std_signed_terminal_x_error_ego_minus_goal": pstdev(x_errors),
        "mean_signed_terminal_y_error_ego_minus_goal": fmean(y_errors),
        "std_signed_terminal_y_error_ego_minus_goal": pstdev(y_errors),
        "mean_abs_terminal_x_error": fmean(abs(value) for value in x_errors),
        "mean_abs_terminal_y_error": fmean(abs(value) for value in y_errors),
        "lane_mismatch_failures": failures,
        "lane_mismatch_failure_ratio": failures / len(details),
        "lane_centers": json.dumps(lane_centers),
        "terminal_definition": "exact next_high_obs post-action state",
    }
    return details, summary


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="Training log directories.")
    parser.add_argument("--log-root", type=Path, default=Path("logs/current"))
    parser.add_argument("--tail-steps", type=int, default=500_000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/training_tail_goal_errors"),
    )
    args = parser.parse_args()
    if args.tail_steps <= 0:
        parser.error("--tail-steps must be positive")

    runs = args.runs or [args.log_root / name for name in DEFAULT_RUNS]
    summaries: list[dict[str, Any]] = []
    for run_dir in runs:
        details, summary = analyze_run(run_dir, args.tail_steps)
        suffix = f"last_{args.tail_steps}_steps"
        _write_rows(args.output_dir / f"{run_dir.name}_{suffix}.csv", details)
        summaries.append(summary)
        print(
            f"{summary['run']}: intervals={summary['intervals']}, "
            f"signed_x_mean={summary['mean_signed_terminal_x_error_ego_minus_goal']:.6f}, "
            f"signed_x_std={summary['std_signed_terminal_x_error_ego_minus_goal']:.6f}, "
            f"signed_y_mean={summary['mean_signed_terminal_y_error_ego_minus_goal']:.6f}, "
            f"signed_y_std={summary['std_signed_terminal_y_error_ego_minus_goal']:.6f}, "
            f"abs_x_mean={summary['mean_abs_terminal_x_error']:.6f}, "
            f"abs_y_mean={summary['mean_abs_terminal_y_error']:.6f}, "
            f"failure_ratio={summary['lane_mismatch_failure_ratio']:.6%}"
        )

    _write_rows(args.output_dir / f"summary_last_{args.tail_steps}_steps.csv", summaries)


if __name__ == "__main__":
    main()
