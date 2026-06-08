from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_curves(csv_path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    rows_by_lane: dict[int, list[tuple[float, float]]] = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target_time_raw = str(
                row.get("target_time_mean", row.get("target_time", ""))
            ).strip()
            if not target_time_raw:
                continue
            departure_time = float(row["offset"])
            travel_time = float(target_time_raw)
            lane_id = int(row["initial_lane_id"])
            crossing_time = departure_time + travel_time
            rows_by_lane[lane_id].append((departure_time, crossing_time))

    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for lane_id, rows in rows_by_lane.items():
        rows.sort(key=lambda item: item[0])
        curves[lane_id] = (
            np.asarray([row[0] for row in rows], dtype=float),
            np.asarray([row[1] for row in rows], dtype=float),
        )
    return curves


def plot_curves(csv_path: Path, output_path: Path) -> None:
    curves = load_curves(csv_path)
    if not curves:
        raise RuntimeError(f"No valid target_time rows found in {csv_path}")

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    fig, ax = plt.subplots(figsize=(11.5, 6.8), constrained_layout=True)

    all_departures: list[np.ndarray] = []
    all_crossings: list[np.ndarray] = []
    for color, lane_id in zip(colors, sorted(curves)):
        departures, crossings = curves[lane_id]
        all_departures.append(departures)
        all_crossings.append(crossings)
        ax.plot(
            departures,
            crossings,
            color=color,
            linewidth=1.9,
            marker="o",
            markersize=2.8,
            label=f"Initial lane {lane_id}",
        )

    x = np.concatenate(all_departures)
    y = np.concatenate(all_crossings)
    ref_min = float(min(np.min(x), np.min(y)))
    ref_max = float(max(np.max(x), np.max(y)))
    ax.plot(
        [ref_min, ref_max],
        [ref_min, ref_max],
        color="#666666",
        linewidth=1.0,
        linestyle="--",
        alpha=0.65,
        label="Departure time",
    )

    ax.set_title("Ego Stop-Line Crossing Time by Departure Time")
    ax.set_xlabel("Ego departure time / episode_start_phase_offset (s)")
    ax.set_ylabel("Ego stop-line crossing time (s)")
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if np.isclose(x_min, x_max):
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
    else:
        ax.set_xlim(x_min, x_max)
    ax.grid(True, color="#D0D0D0", linewidth=0.7, alpha=0.7)
    ax.legend(loc="upper left", frameon=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    default_dir = Path("debug") / "offset_target_time_table_with_traffic"
    parser = argparse.ArgumentParser(
        description="Plot departure time against absolute stop-line crossing time."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_dir / "offset_target_time_mean.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dir / "departure_vs_crossing_time.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_curves(args.csv, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
