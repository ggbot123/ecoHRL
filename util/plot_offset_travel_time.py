from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path("debug") / "offset_target_time_table_with_traffic"
INPUT_CSV = DATA_DIR / "offset_target_time_mean.csv"
RAW_INPUT_CSV = DATA_DIR / "offset_target_time_raw.csv"
OUTPUT_PNG = DATA_DIR / "offset_vs_travel_time_mean_std.png"
ALL_LANES_OUTPUT_PNG = DATA_DIR / "offset_vs_travel_time_all_lanes_mean_std.png"


def load_lane_curves(
    csv_path: Path,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rows_by_lane: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mean_raw = str(row.get("target_time_mean", "")).strip()
            std_raw = str(row.get("target_time_std", "")).strip()
            if not mean_raw:
                continue
            offset = float(row["offset"])
            lane_id = int(row["initial_lane_id"])
            mean_time = float(mean_raw)
            std_time = float(std_raw) if std_raw else 0.0
            rows_by_lane[lane_id].append((offset, mean_time, std_time))

    curves: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for lane_id, rows in rows_by_lane.items():
        rows.sort(key=lambda item: item[0])
        curves[lane_id] = (
            np.asarray([row[0] for row in rows], dtype=float),
            np.asarray([row[1] for row in rows], dtype=float),
            np.asarray([row[2] for row in rows], dtype=float),
        )
    return curves


def load_all_lanes_curve(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_by_offset: dict[float, list[float]] = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target_time_raw = str(row.get("target_time", "")).strip()
            if not target_time_raw:
                continue
            values_by_offset[float(row["offset"])].append(float(target_time_raw))

    offsets = np.asarray(sorted(values_by_offset), dtype=float)
    means = np.asarray(
        [np.mean(values_by_offset[offset]) for offset in offsets],
        dtype=float,
    )
    stds = np.asarray(
        [np.std(values_by_offset[offset]) for offset in offsets],
        dtype=float,
    )
    return offsets, means, stds


def main() -> None:
    curves = load_lane_curves(INPUT_CSV)
    if not curves:
        raise RuntimeError(f"No mean travel-time rows found in {INPUT_CSV}")

    colors = {
        0: "#0072B2",
        1: "#D55E00",
        2: "#009E73",
    }
    fig, ax = plt.subplots(figsize=(12.0, 7.0), constrained_layout=True)

    all_offsets: list[np.ndarray] = []
    for lane_id in sorted(curves):
        offsets, means, stds = curves[lane_id]
        color = colors.get(lane_id, f"C{lane_id % 10}")
        all_offsets.append(offsets)
        ax.fill_between(
            offsets,
            np.maximum(means - stds, 0.0),
            means + stds,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            offsets,
            means,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=2.6,
            label=f"Initial lane {lane_id}",
        )

    x = np.concatenate(all_offsets)
    ax.set_title("Ego Travel Time to Stop Line by Departure Offset")
    ax.set_xlabel("Episode start phase offset (s)")
    ax.set_ylabel("Travel time to stop line (s)")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.grid(True, color="#D0D0D0", linewidth=0.7, alpha=0.7)
    ax.legend(title="Mean curve; shaded band = mean +/- std", loc="upper right")

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=200)
    plt.close(fig)
    print(OUTPUT_PNG.resolve())

    offsets, means, stds = load_all_lanes_curve(RAW_INPUT_CSV)
    if offsets.size == 0:
        raise RuntimeError(f"No raw travel-time rows found in {RAW_INPUT_CSV}")

    fig, ax = plt.subplots(figsize=(12.0, 7.0), constrained_layout=True)
    color = "#0072B2"
    ax.fill_between(
        offsets,
        np.maximum(means - stds, 0.0),
        means + stds,
        color=color,
        alpha=0.2,
        linewidth=0,
    )
    ax.plot(
        offsets,
        means,
        color=color,
        linewidth=2.2,
        marker="o",
        markersize=2.8,
        label="Mean across all lanes and episodes",
    )
    ax.set_title("Mean Ego Travel Time Across All Initial Lanes")
    ax.set_xlabel("Episode start phase offset (s)")
    ax.set_ylabel("Travel time to stop line (s)")
    ax.set_xlim(float(np.min(offsets)), float(np.max(offsets)))
    ax.grid(True, color="#D0D0D0", linewidth=0.7, alpha=0.7)
    ax.legend(title="Shaded band = mean +/- std", loc="upper right")

    fig.savefig(ALL_LANES_OUTPUT_PNG, dpi=200)
    plt.close(fig)
    print(ALL_LANES_OUTPUT_PNG.resolve())


if __name__ == "__main__":
    main()
