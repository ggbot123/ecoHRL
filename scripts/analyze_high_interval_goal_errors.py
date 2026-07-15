"""Measure signed high-level goal error at each interval terminal state.

The evaluator writes one ``high_interval_debug.csv`` row at every high-level
interval start and one trajectory row per low-level step.  For intervals that
have a following high-level start, that next start is the exact post-action
terminal state.  The final interval of each episode lacks a saved terminal
observation, so its terminal position is extrapolated for one policy step from
the final logged state and velocity.  The script reports the mean and
population standard deviation of signed x/y terminal errors (ego minus goal),
plus lane misses.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable


DEFAULT_EXPERIMENTS = (
    "20260707_164522_uniHERnew",
    "20260709_102056_HRLnew",
)


def _json_vector(raw: str) -> list[float]:
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list, got {type(value).__name__}")
    return [float(item) for item in value]


def _lane_id(y: float, lane_centers: list[float]) -> int:
    return min(range(len(lane_centers)), key=lambda i: abs(y - lane_centers[i]))


def _load_environment_config(experiment_dir: Path) -> dict[str, Any]:
    path = experiment_dir / "effective_eval_config.json"
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    environment = config.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError(f"Invalid environment config in {path}")
    return environment


def _trajectory_files(experiment_dir: Path) -> list[Path]:
    files = sorted(experiment_dir.glob("hiro_ep_*_trajectory.csv"))
    if not files:
        raise FileNotFoundError(f"No HIRO trajectory CSV files in {experiment_dir}")
    return files


def _load_high_starts(experiment_dir: Path) -> dict[int, list[dict[str, str]]]:
    path = experiment_dir / "high_interval_debug.csv"
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped[int(row["episode_env0"]) + 1].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["step"]))
    return grouped


def analyze_experiment(experiment_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    environment = _load_environment_config(experiment_dir)
    lanes_count = int(environment.get("lanes_count", 3))
    lane_width = float(environment.get("lane_width", 4.0))
    high_interval = int(environment.get("high_interval", 25))
    policy_frequency = float(environment.get("policy_frequency", 10.0))
    if policy_frequency <= 0.0:
        raise ValueError(f"policy_frequency must be positive in {experiment_dir}")
    policy_dt = 1.0 / policy_frequency
    lane_centers = [lane_width * lane for lane in range(lanes_count)]

    high_by_episode = _load_high_starts(experiment_dir)
    details: list[dict[str, Any]] = []

    for trajectory_path in _trajectory_files(experiment_dir):
        episode = int(trajectory_path.stem.split("_")[2])
        starts = high_by_episode.get(episode, [])
        if not starts:
            raise ValueError(f"Episode {episode} has no high-level starts")

        with trajectory_path.open("r", newline="", encoding="utf-8") as stream:
            trajectory = list(csv.DictReader(stream))
        if not trajectory:
            raise ValueError(f"Empty trajectory: {trajectory_path}")

        low_obs_columns = sorted(
            (name for name in trajectory[0] if name.startswith("low_obs_")),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
        first_global_step = int(starts[0]["step"])

        for interval_in_episode, start in enumerate(starts):
            goal = _json_vector(start["goal_phys"])
            if len(goal) < 2 or len(low_obs_columns) < len(goal):
                raise ValueError(f"Cannot decode goal-relative state in {trajectory_path}")

            start_local = int(start["step"]) - first_global_step
            if interval_in_episode + 1 < len(starts):
                end_local = int(starts[interval_in_episode + 1]["step"]) - first_global_step - 1
            else:
                end_local = len(trajectory) - 1
            if not (0 <= start_local <= end_local < len(trajectory)):
                raise ValueError(
                    f"Invalid interval [{start_local}, {end_local}] in {trajectory_path}"
                )

            if interval_in_episode + 1 < len(starts):
                terminal_ego = _json_vector(starts[interval_in_episode + 1]["ego_sub"])
                terminal_state_source = "exact_next_high_start"
            else:
                endpoint = trajectory[end_local]
                goal_rel_columns = low_obs_columns[-len(goal) :]
                goal_rel = [float(endpoint[name]) for name in goal_rel_columns]
                terminal_ego = [goal[i] - goal_rel[i] for i in range(len(goal))]
                if len(terminal_ego) < 4:
                    raise ValueError(
                        f"Cannot extrapolate terminal state in {trajectory_path}"
                    )
                terminal_ego[0] += terminal_ego[2] * policy_dt
                terminal_ego[1] += terminal_ego[3] * policy_dt
                terminal_state_source = "velocity_extrapolated_episode_terminal"

            ego_terminal_x = terminal_ego[0]
            ego_terminal_y = terminal_ego[1]
            signed_x_error = ego_terminal_x - goal[0]
            signed_y_error = ego_terminal_y - goal[1]
            goal_lane = _lane_id(goal[1], lane_centers)
            ego_lane = _lane_id(ego_terminal_y, lane_centers)
            interval_steps = end_local - start_local + 1

            details.append(
                {
                    "experiment": experiment_dir.name,
                    "episode": episode,
                    "interval_in_episode": interval_in_episode,
                    "segment_id": int(start["segment_id"]),
                    "start_step": start_local,
                    "end_step": end_local,
                    "interval_steps": interval_steps,
                    "is_regular_length": int(interval_steps == high_interval),
                    "terminal_state_source": terminal_state_source,
                    "goal_x": goal[0],
                    "goal_y": goal[1],
                    "ego_terminal_x": ego_terminal_x,
                    "ego_terminal_y": ego_terminal_y,
                    "terminal_signed_x_error_ego_minus_goal": signed_x_error,
                    "terminal_signed_y_error_ego_minus_goal": signed_y_error,
                    "terminal_abs_x_error": abs(signed_x_error),
                    "terminal_abs_y_error": abs(signed_y_error),
                    "goal_lane": goal_lane,
                    "ego_terminal_lane": ego_lane,
                    "failed_lane_mismatch": int(goal_lane != ego_lane),
                }
            )

    expected = sum(len(rows) for rows in high_by_episode.values())
    if len(details) != expected:
        raise ValueError(f"Decoded {len(details)} intervals, expected {expected}")

    failures = sum(int(row["failed_lane_mismatch"]) for row in details)
    summary = {
        "experiment": experiment_dir.name,
        "episodes": len(high_by_episode),
        "intervals": len(details),
        "regular_length_intervals": sum(int(row["is_regular_length"]) for row in details),
        "nonregular_length_intervals": sum(
            1 - int(row["is_regular_length"]) for row in details
        ),
        "exact_terminal_states": sum(
            row["terminal_state_source"] == "exact_next_high_start" for row in details
        ),
        "extrapolated_terminal_states": sum(
            row["terminal_state_source"] == "velocity_extrapolated_episode_terminal"
            for row in details
        ),
        "mean_signed_terminal_x_error_ego_minus_goal": fmean(
            float(row["terminal_signed_x_error_ego_minus_goal"]) for row in details
        ),
        "std_signed_terminal_x_error_ego_minus_goal": pstdev(
            float(row["terminal_signed_x_error_ego_minus_goal"]) for row in details
        ),
        "mean_signed_terminal_y_error_ego_minus_goal": fmean(
            float(row["terminal_signed_y_error_ego_minus_goal"]) for row in details
        ),
        "std_signed_terminal_y_error_ego_minus_goal": pstdev(
            float(row["terminal_signed_y_error_ego_minus_goal"]) for row in details
        ),
        "lane_mismatch_failures": failures,
        "lane_mismatch_failure_ratio": failures / len(details),
        "lane_width": lane_width,
        "lane_centers": json.dumps(lane_centers),
        "terminal_definition": (
            "post-action state: exact next high start; episode-final interval "
            "uses one-step constant-velocity extrapolation"
        ),
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
    parser.add_argument(
        "experiments",
        nargs="*",
        type=Path,
        help="Experiment directories (defaults to the two requested eval groups).",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("results/eval_results"),
        help="Root used for default experiment names and the combined summary.",
    )
    args = parser.parse_args()

    experiments = args.experiments or [args.eval_root / name for name in DEFAULT_EXPERIMENTS]
    summaries: list[dict[str, Any]] = []
    for experiment in experiments:
        details, summary = analyze_experiment(experiment)
        detail_path = experiment / "high_interval_goal_errors.csv"
        _write_rows(detail_path, details)
        summaries.append(summary)
        print(
            f"{summary['experiment']}: intervals={summary['intervals']}, "
            f"signed_x_mean={summary['mean_signed_terminal_x_error_ego_minus_goal']:.6f}, "
            f"signed_x_std={summary['std_signed_terminal_x_error_ego_minus_goal']:.6f}, "
            f"signed_y_mean={summary['mean_signed_terminal_y_error_ego_minus_goal']:.6f}, "
            f"signed_y_std={summary['std_signed_terminal_y_error_ego_minus_goal']:.6f}, "
            f"failures={summary['lane_mismatch_failures']}, "
            f"failure_ratio={summary['lane_mismatch_failure_ratio']:.6%}"
        )

    summary_path = args.eval_root / "high_interval_goal_error_summary.csv"
    _write_rows(summary_path, summaries)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
