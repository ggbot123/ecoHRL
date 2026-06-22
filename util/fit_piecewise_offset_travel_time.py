from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path("debug") / "offset_target_time_table_spawn006"
DATA_DIRS = [
    DATA_ROOT / "lane_probs_060_030_010",
    DATA_ROOT / "lane_probs_040_030_030",
]

LEFT_END = 3.0
LOW_PLATEAU_END = 25.0
HIGH_PLATEAU_END = 30.0
LOW_LEVEL = 35.0
HIGH_LEVEL = 75.0


def load_all_lanes_mean(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_by_offset: dict[float, list[float]] = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            target_time_raw = str(row.get("target_time", "")).strip()
            if target_time_raw:
                values_by_offset[float(row["offset"])].append(float(target_time_raw))

    offsets = np.asarray(sorted(values_by_offset), dtype=float)
    means = np.asarray([np.mean(values_by_offset[x]) for x in offsets], dtype=float)
    stds = np.asarray([np.std(values_by_offset[x]) for x in offsets], dtype=float)
    return offsets, means, stds


def fit_shared_slope(offsets: np.ndarray, means: np.ndarray) -> float:
    left = offsets <= LEFT_END
    right = offsets >= HIGH_PLATEAU_END

    design = np.concatenate(
        [
            offsets[left] - LEFT_END,
            offsets[right] - HIGH_PLATEAU_END,
        ]
    )
    residual = np.concatenate(
        [
            means[left] - LOW_LEVEL,
            means[right] - HIGH_LEVEL,
        ]
    )
    denominator = float(np.dot(design, design))
    if denominator <= 1e-12:
        raise RuntimeError("Not enough outer-segment data to fit the shared slope.")
    return float(np.dot(design, residual) / denominator)


def piecewise_value(offsets: np.ndarray, slope: float) -> np.ndarray:
    x = np.asarray(offsets, dtype=float)
    y = np.empty_like(x)

    left = x < LEFT_END
    low = (x >= LEFT_END) & (x < LOW_PLATEAU_END)
    high = (x >= LOW_PLATEAU_END) & (x < HIGH_PLATEAU_END)
    right = x >= HIGH_PLATEAU_END

    y[left] = LOW_LEVEL + slope * (x[left] - LEFT_END)
    y[low] = LOW_LEVEL
    y[high] = HIGH_LEVEL
    y[right] = HIGH_LEVEL + slope * (x[right] - HIGH_PLATEAU_END)
    return y


def write_fit_csv(path: Path, offsets: np.ndarray, fitted: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "offset",
                "fitted_window_center",
                "fitted_window_start",
                "fitted_window_end",
            ],
        )
        writer.writeheader()
        for offset, travel_time in zip(offsets, fitted):
            writer.writerow(
                {
                    "offset": f"{offset:.6f}",
                    "fitted_window_center": f"{travel_time:.6f}",
                    "fitted_window_start": f"{travel_time - 5.0:.6f}",
                    "fitted_window_end": f"{travel_time + 5.0:.6f}",
                }
            )


def fit_data_dir(data_dir: Path) -> None:
    raw_input_csv = data_dir / "offset_target_time_raw.csv"
    output_png = data_dir / "offset_travel_time_piecewise_fit.png"
    output_csv = data_dir / "offset_travel_time_piecewise_fit.csv"

    offsets, means, stds = load_all_lanes_mean(raw_input_csv)
    if offsets.size == 0:
        raise RuntimeError(f"No travel-time rows found in {raw_input_csv}")

    slope = fit_shared_slope(offsets, means)
    fitted_at_samples = piecewise_value(offsets, slope)
    rmse_outer_mask = (offsets <= LEFT_END) | (offsets >= HIGH_PLATEAU_END)
    rmse_outer = float(
        np.sqrt(np.mean((fitted_at_samples[rmse_outer_mask] - means[rmse_outer_mask]) ** 2))
    )

    x_plot = np.linspace(float(np.min(offsets)), float(np.max(offsets)), 1200)
    y_plot = piecewise_value(x_plot, slope)

    fig, ax = plt.subplots(figsize=(12.0, 7.0), constrained_layout=True)
    ax.fill_between(
        offsets,
        np.maximum(means - stds, 0.0),
        means + stds,
        color="#0072B2",
        alpha=0.16,
        linewidth=0,
        label="Episode mean +/- std",
    )
    ax.plot(
        offsets,
        means,
        color="#0072B2",
        linewidth=1.7,
        marker="o",
        markersize=2.5,
        label="Mean across all lanes and episodes",
    )
    ax.plot(
        x_plot,
        y_plot,
        color="#D55E00",
        linewidth=2.8,
        label=f"Piecewise fit (shared slope k={slope:.4f})",
    )
    for boundary in [LEFT_END, LOW_PLATEAU_END, HIGH_PLATEAU_END]:
        ax.axvline(boundary, color="#777777", linewidth=0.9, linestyle="--", alpha=0.7)

    ax.set_title("Piecewise Fit of Mean Ego Travel Time")
    ax.set_xlabel("Episode start phase offset (s)")
    ax.set_ylabel("Travel time to stop line (s)")
    ax.set_xlim(float(np.min(offsets)), float(np.max(offsets)))
    ax.grid(True, color="#D0D0D0", linewidth=0.7, alpha=0.7)
    ax.legend(loc="upper right")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200)
    plt.close(fig)
    write_fit_csv(output_csv, offsets, fitted_at_samples)

    print(f"\n=== {data_dir} ===")
    print(f"shared slope: {slope:.8f}")
    print(f"outer-segment RMSE: {rmse_outer:.6f} s")
    print("piecewise function:")
    print(f"  offset < 3   : y = {LOW_LEVEL:.1f} + ({slope:.8f}) * (offset - 3)")
    print("  3 <= offset < 25  : y = 35")
    print("  25 <= offset < 30 : y = 75")
    print(f"  offset >= 30 : y = {HIGH_LEVEL:.1f} + ({slope:.8f}) * (offset - 30)")
    print(output_png.resolve())
    print(output_csv.resolve())


def main() -> None:
    for data_dir in DATA_DIRS:
        fit_data_dir(data_dir)


if __name__ == "__main__":
    main()
