from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle


COMPONENTS = [("left", -1), ("keep", 0), ("right", 1)]
LANE_COLORS = ["#4C78A8", "#59A14F", "#F28E2B", "#B07AA1", "#76B7B2"]
VEH_LENGTH = 5.0
VEH_WIDTH = 2.0
X_MARGIN = 70.0


def parse_array(value: str) -> list[Any]:
    if value is None or value == "":
        return []
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Python's json parser accepts NaN by default, but keep this fallback for
        # slightly malformed historical debug rows.
        return []


def flatten_once(value: Any) -> list[Any]:
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    if isinstance(value, list):
        return value
    return []


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def first_mapping_with_key(obj: Any, key: str) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        if key in obj:
            return obj
        for child in obj.values():
            found = first_mapping_with_key(child, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for child in obj:
            found = first_mapping_with_key(child, key)
            if found is not None:
                return found
    return None


def load_config(eval_dir: Path) -> tuple[int, float, float | None, int | None, float, float]:
    config_path = eval_dir / "effective_eval_config.json"
    if not config_path.exists():
        return 3, 4.0, None, None, 15.0, 0.1

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    env_cfg = first_mapping_with_key(cfg, "lanes_count") or {}
    hiro_cfg = first_mapping_with_key(cfg, "high_interval") or {}

    n_lanes = int(env_cfg.get("lanes_count", 3))
    lane_width = float(env_cfg.get("lane_width", 4.0))
    goal_x_raw = env_cfg.get("goal_longitudinal", None)
    goal_x = float(goal_x_raw) if goal_x_raw is not None else None
    high_interval = hiro_cfg.get("high_interval", None)
    high_interval_int = int(high_interval) if high_interval is not None else None
    speed_limit = float(env_cfg.get("speed_limit", 15.0))
    policy_frequency = float(env_cfg.get("policy_frequency", 10.0))
    dt = 1.0 / max(policy_frequency, 1e-6)
    return n_lanes, lane_width, goal_x, high_interval_int, speed_limit, dt


def load_episode_rows(debug_csv: Path, episode: int) -> list[dict[str, Any]]:
    episode_env0 = int(episode) - 1
    rows: list[dict[str, Any]] = []
    with debug_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if int(raw.get("episode_env0", -1)) != episode_env0:
                continue

            ego = flatten_once(parse_array(raw.get("ego_sub", "")))
            goal_phys = flatten_once(parse_array(raw.get("goal_phys", "")))
            goal_action = flatten_once(parse_array(raw.get("goal_action", "")))
            dx_lo = flatten_once(parse_array(raw.get("safe_dx_l2", "")))
            dx_hi = flatten_once(parse_array(raw.get("safe_dx_u2", "")))
            kin = parse_array(raw.get("kin", ""))

            rows.append(
                {
                    "step": int(raw["step"]),
                    "segment_id": int(raw["segment_id"]),
                    "ego": ego,
                    "ego_x": float(ego[0]) if len(ego) > 0 else float("nan"),
                    "ego_y": float(ego[1]) if len(ego) > 1 else float("nan"),
                    "ego_vx": float(ego[2]) if len(ego) > 2 else float("nan"),
                    "goal_phys": goal_phys,
                    "goal_action": goal_action,
                    "dx_lo": dx_lo,
                    "dx_hi": dx_hi,
                    "kin": kin,
                }
            )
    return rows


def lane_index_from_y(y: float, lane_width: float, n_lanes: int) -> int:
    if not math.isfinite(y):
        return 0
    return max(0, min(n_lanes - 1, int(round(y / lane_width))))


def row_intervals(row: dict[str, Any], lane_width: float, n_lanes: int) -> list[dict[str, Any]]:
    ego_lane = lane_index_from_y(float(row["ego_y"]), lane_width, n_lanes)
    intervals: list[dict[str, Any]] = []
    for comp_idx, (name, offset) in enumerate(COMPONENTS):
        target_lane = ego_lane + offset
        if target_lane < 0 or target_lane >= n_lanes:
            continue
        lo = finite_float(row["dx_lo"][comp_idx] if comp_idx < len(row["dx_lo"]) else None)
        hi = finite_float(row["dx_hi"][comp_idx] if comp_idx < len(row["dx_hi"]) else None)
        if lo is None or hi is None or hi <= lo:
            continue
        intervals.append(
            {
                "component": name,
                "target_lane": target_lane,
                "dx_lo": lo,
                "dx_hi": hi,
                "x_lo": float(row["ego_x"]) + lo,
                "x_hi": float(row["ego_x"]) + hi,
            }
        )
    return intervals


def kin_to_absolute(row: dict[str, Any], lane_width: float) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    kin_raw = row.get("kin", [])
    if not isinstance(kin_raw, list) or len(kin_raw) == 0:
        return [], []

    ego_x = float(row["ego_x"])
    ego_y = float(row["ego_y"])
    ego_vx = float(row["ego_vx"])
    vehicles: list[dict[str, float]] = []
    for idx, item in enumerate(kin_raw):
        if not isinstance(item, list) or len(item) < 4:
            continue
        presence = finite_float(item[0])
        if presence is not None and presence <= 0.5:
            continue

        raw_x = finite_float(item[1])
        raw_y = finite_float(item[2])
        raw_vx = finite_float(item[3])
        raw_vy = finite_float(item[4]) if len(item) > 4 else 0.0
        if raw_x is None or raw_y is None or raw_vx is None:
            continue

        if idx == 0:
            x_abs = raw_x
            y_abs = raw_y
            vx_abs = raw_vx
            vy_abs = raw_vy if raw_vy is not None else 0.0
        else:
            x_abs = ego_x + raw_x
            # Historical HIRO debug rows store neighbor y as ego-relative offsets.
            y_abs = ego_y + raw_y if abs(raw_y) <= lane_width * 1.5 else raw_y
            vx_abs = ego_vx + raw_vx
            vy_abs = raw_vy if raw_vy is not None else 0.0

        vehicles.append(
            {
                "idx": float(idx),
                "x": float(x_abs),
                "y": float(y_abs),
                "vx": float(vx_abs),
                "vy": float(vy_abs),
            }
        )
    neighbors = [v for v in vehicles if int(v["idx"]) != 0]
    return vehicles, neighbors


def predicted_neighbors(
    neighbors: list[dict[str, float]],
    high_interval: int | None,
    dt: float,
) -> list[dict[str, float]]:
    horizon_t = float(high_interval or 25) * float(dt)
    return [
        {
            **v,
            "x": float(v["x"] + v["vx"] * horizon_t),
            "y": float(v["y"] + v["vy"] * horizon_t),
        }
        for v in neighbors
    ]


def add_lane_background(ax: plt.Axes, n_lanes: int, lane_width: float, xlim: tuple[float, float]) -> None:
    for lane in range(n_lanes):
        y0 = lane * lane_width - 0.5 * lane_width
        ax.add_patch(
            Rectangle(
                (xlim[0], y0),
                xlim[1] - xlim[0],
                lane_width,
                facecolor="#666666",
                edgecolor="none",
                zorder=0,
            )
        )
        ax.axhline(lane * lane_width, color="white", lw=1.0, ls="--", zorder=1)
    ax.axhline(-0.5 * lane_width, color="white", lw=1.0, zorder=1)
    ax.axhline((n_lanes - 0.5) * lane_width, color="white", lw=1.0, zorder=1)
    ax.set_yticks([lane * lane_width for lane in range(n_lanes)])
    ax.set_yticklabels([f"lane {lane}" for lane in range(n_lanes)])


def draw_vehicles(
    ax: plt.Axes,
    vehicles: list[dict[str, float]],
    pred: list[dict[str, float]],
    cmap: Any,
    norm: mcolors.Normalize,
    compact: bool,
) -> None:
    for v in vehicles:
        idx = int(v["idx"])
        color = cmap(norm(max(0.0, float(v["vx"]))))
        edge = "red" if idx == 0 else "black"
        lw = 1.6 if idx == 0 else 1.0
        rect = Rectangle(
            (float(v["x"]) - VEH_LENGTH / 2.0, float(v["y"]) - VEH_WIDTH / 2.0),
            VEH_LENGTH,
            VEH_WIDTH,
            facecolor=color,
            edgecolor=edge,
            linewidth=lw,
            alpha=0.95,
            zorder=5 if idx == 0 else 4,
        )
        ax.add_patch(rect)

    pred_size = float(max(1.6, VEH_WIDTH * 1.05))
    for v in pred:
        idx = int(v["idx"])
        color = cmap(norm(max(0.0, float(v["vx"]))))
        rect = Rectangle(
            (float(v["x"]) - pred_size / 2.0, float(v["y"]) - pred_size / 2.0),
            pred_size,
            pred_size,
            facecolor=color,
            edgecolor="white",
            linewidth=0.9,
            alpha=0.36,
            zorder=6,
        )
        ax.add_patch(rect)
        if not compact:
            ax.text(
                float(v["x"]) + pred_size * 0.55,
                float(v["y"]) - pred_size * 0.55,
                f"pred{idx}: x={float(v['x']):.1f}, v={float(v['vx']):.1f}",
                fontsize=8,
                color="white",
                ha="left",
                va="center",
                zorder=8,
                bbox=dict(facecolor="black", alpha=0.35, pad=1.5, edgecolor="none"),
            )


def draw_interval_panel(
    ax: plt.Axes,
    row: dict[str, Any],
    intervals: list[dict[str, Any]],
    n_lanes: int,
    lane_width: float,
    goal_x: float | None,
    high_interval: int | None,
    dt: float,
    speed_limit: float,
    xlim: tuple[float, float],
    compact: bool = False,
) -> None:
    add_lane_background(ax, n_lanes, lane_width, xlim)
    ego_x = float(row["ego_x"])
    ego_y = float(row["ego_y"])
    vehicles, neighbors = kin_to_absolute(row, lane_width)
    pred = predicted_neighbors(neighbors, high_interval, dt)
    norm = mcolors.Normalize(vmin=0.0, vmax=max(float(speed_limit), 1e-3))
    cmap = plt.get_cmap("jet")

    for item in intervals:
        lane = int(item["target_lane"])
        y0 = lane * lane_width - 0.5 * lane_width
        rect = Rectangle(
            (item["x_lo"], y0),
            item["x_hi"] - item["x_lo"],
            lane_width,
            facecolor="#66ff66",
            edgecolor="#00dd00",
            lw=1.5,
            linestyle="--",
            alpha=0.28,
            zorder=2,
        )
        ax.add_patch(rect)
        if compact:
            ax.text(
                0.5 * (item["x_lo"] + item["x_hi"]),
                lane * lane_width,
                f"{item['dx_lo']:.1f}-{item['dx_hi']:.1f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white",
                zorder=7,
                bbox=dict(facecolor="black", alpha=0.25, pad=0.6, edgecolor="none"),
            )

    draw_vehicles(ax, vehicles, pred, cmap, norm, compact=compact)

    goal = row.get("goal_phys", [])
    if len(goal) >= 2:
        gx = finite_float(goal[0])
        gy = finite_float(goal[1])
        if gx is not None and gy is not None:
            ax.scatter([gx], [gy], marker="o", s=75, c="orangered", edgecolors="white", linewidths=1.2, zorder=9)
            if not compact:
                ax.text(
                    gx + 1.8,
                    gy - 0.4,
                    f"goal: ({gx:.1f}, {gy:.1f})",
                    fontsize=9,
                    color="white",
                    ha="left",
                    va="center",
                    zorder=10,
                    bbox=dict(facecolor="black", alpha=0.45, pad=1.8, edgecolor="none"),
                )

    if goal_x is not None:
        ax.axvline(goal_x, color="white", lw=1.1, alpha=0.6, zorder=3)

    if not compact:
        range_lines = []
        for item in intervals:
            range_lines.append(
                f"{item['component']} (lane{item['target_lane']}): x=[{item['x_lo']:.1f}, {item['x_hi']:.1f}]"
            )
        if range_lines:
            ax.text(
                0.01,
                0.98,
                "feasible x-ranges\n" + "\n".join(range_lines),
                transform=ax.transAxes,
                fontsize=9,
                color="white",
                ha="left",
                va="top",
                zorder=12,
                bbox=dict(facecolor="black", alpha=0.45, pad=2.5, edgecolor="none"),
            )
        ax.text(
            0.01,
            0.02,
            f"ego: x={ego_x:.1f}, y={ego_y:.1f}, vx={float(row['ego_vx']):.1f}",
            transform=ax.transAxes,
            fontsize=9,
            color="white",
            ha="left",
            va="bottom",
            zorder=12,
            bbox=dict(facecolor="black", alpha=0.45, pad=2.5, edgecolor="none"),
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(-0.65 * lane_width, (n_lanes - 0.35) * lane_width)
    ax.set_title(
        f"interval {row['interval_idx']}  local_step {row['local_step']}  global_step {row['step']}  ego_x {ego_x:.1f}",
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")


def write_summary_csv(output_dir: Path, episode: int, rows: list[dict[str, Any]], interval_rows: list[list[dict[str, Any]]], lane_width: float, n_lanes: int) -> Path:
    path = output_dir / f"ep{episode:04d}_high_reachable_feasible_regions.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "interval_idx",
                "local_step",
                "global_step",
                "segment_id",
                "ego_x",
                "ego_y",
                "ego_lane",
                "component",
                "target_lane",
                "dx_lo",
                "dx_hi",
                "x_lo",
                "x_hi",
            ]
        )
        for row, intervals in zip(rows, interval_rows):
            ego_lane = lane_index_from_y(float(row["ego_y"]), lane_width, n_lanes)
            for item in intervals:
                writer.writerow(
                    [
                        row["interval_idx"],
                        row["local_step"],
                        row["step"],
                        row["segment_id"],
                        f"{row['ego_x']:.6f}",
                        f"{row['ego_y']:.6f}",
                        ego_lane,
                        item["component"],
                        item["target_lane"],
                        f"{item['dx_lo']:.6f}",
                        f"{item['dx_hi']:.6f}",
                        f"{item['x_lo']:.6f}",
                        f"{item['x_hi']:.6f}",
                    ]
                )
    return path


def plot_all(eval_dir: Path, episode: int, output_dir: Path) -> None:
    n_lanes, lane_width, goal_x, high_interval, speed_limit, dt = load_config(eval_dir)
    rows = load_episode_rows(eval_dir / "high_interval_debug.csv", episode)
    if not rows:
        raise RuntimeError(f"No high-interval debug rows found for episode {episode}")
    first_global_step = int(rows[0]["step"])
    for idx, row in enumerate(rows, start=1):
        row["interval_idx"] = idx
        row["local_step"] = (idx - 1) * high_interval if high_interval is not None else int(row["step"]) - first_global_step

    interval_rows = [row_intervals(row, lane_width, n_lanes) for row in rows]
    all_x = [float(row["ego_x"]) for row in rows]
    for intervals in interval_rows:
        for item in intervals:
            all_x.extend([item["x_lo"], item["x_hi"]])
    for row in rows:
        vehicles, neighbors = kin_to_absolute(row, lane_width)
        pred = predicted_neighbors(neighbors, high_interval, dt)
        all_x.extend([float(v["x"]) for v in vehicles])
        all_x.extend([float(v["x"]) for v in pred])
    for row in rows:
        goal = row.get("goal_phys", [])
        if len(goal) >= 1:
            gx = finite_float(goal[0])
            if gx is not None:
                all_x.append(gx)
    if goal_x is not None:
        all_x.append(goal_x)
    x_min, x_max = min(all_x), max(all_x)
    pad = max(10.0, 0.05 * (x_max - x_min))
    global_xlim = (x_min - pad, x_max + pad)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = write_summary_csv(output_dir, episode, rows, interval_rows, lane_width, n_lanes)

    ncols = 4
    nrows = math.ceil(len(rows) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 2.5 * nrows), constrained_layout=True)
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]
    for ax, row, intervals in zip(axes_list, rows, interval_rows):
        local_x = [float(row["ego_x"])]
        for item in intervals:
            local_x.extend([item["x_lo"], item["x_hi"]])
        vehicles, neighbors = kin_to_absolute(row, lane_width)
        pred = predicted_neighbors(neighbors, high_interval, dt)
        local_x.extend([float(v["x"]) for v in vehicles])
        local_x.extend([float(v["x"]) for v in pred])
        goal = row.get("goal_phys", [])
        if len(goal) >= 1:
            gx = finite_float(goal[0])
            if gx is not None:
                local_x.append(gx)
        local_pad = 8.0
        draw_interval_panel(
            ax,
            row,
            intervals,
            n_lanes,
            lane_width,
            goal_x,
            high_interval,
            dt,
            speed_limit,
            (min(local_x) - local_pad, max(local_x) + local_pad),
            compact=True,
        )
    for ax in axes_list[len(rows) :]:
        ax.axis("off")
    hi_text = f", high_interval={high_interval}" if high_interval is not None else ""
    fig.suptitle(f"Episode {episode:04d} high-level reachable goal feasible regions{hi_text}", fontsize=14)
    sm = cm.ScalarMappable(cmap=plt.get_cmap("jet"), norm=mcolors.Normalize(vmin=0.0, vmax=max(float(speed_limit), 1e-3)))
    sm.set_array([])
    fig.colorbar(sm, ax=axes_list[: len(rows)], aspect=35, shrink=0.75, pad=0.01, label="Speed [m/s]")
    overview_path = output_dir / f"ep{episode:04d}_high_reachable_feasible_regions_overview.png"
    fig.savefig(overview_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.5, max(6.0, 0.38 * len(rows))), constrained_layout=True)
    ax.set_xlim(*global_xlim)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"int {row['interval_idx']} / t {row['local_step']}" for row in rows], fontsize=8)
    ax.invert_yaxis()
    for i, (row, intervals) in enumerate(zip(rows, interval_rows)):
        ax.axhline(i, color="#EEEEEE", lw=0.7, zorder=0)
        ax.axvline(float(row["ego_x"]), color="#333333", lw=0.7, alpha=0.45, zorder=1)
        for item in intervals:
            lane = int(item["target_lane"])
            ax.barh(
                i,
                item["x_hi"] - item["x_lo"],
                left=item["x_lo"],
                height=0.22,
                color=LANE_COLORS[lane % len(LANE_COLORS)],
                edgecolor="#222222",
                linewidth=0.4,
                alpha=0.75,
                label=f"lane {lane}" if i == 0 else None,
            )
        goal = row.get("goal_phys", [])
        if len(goal) >= 1:
            gx = finite_float(goal[0])
            if gx is not None:
                ax.scatter([gx], [i], marker="*", s=55, c="#D62728", edgecolors="#7F0000", linewidths=0.4, zorder=4)
    if goal_x is not None:
        ax.axvline(goal_x, color="#D62728", lw=1.0, alpha=0.45, label="episode goal x")
    ax.set_xlabel("absolute x (m)")
    ax.set_title(f"Episode {episode:04d} feasible x intervals by high interval")
    ax.grid(axis="x", color="#DDDDDD", lw=0.5)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(unique.values(), unique.keys(), loc="lower right", fontsize=8)
    timeline_path = output_dir / f"ep{episode:04d}_high_reachable_feasible_regions_timeline.png"
    fig.savefig(timeline_path, dpi=180)
    plt.close(fig)

    pdf_path = output_dir / f"ep{episode:04d}_high_reachable_feasible_regions_each_interval.pdf"
    with PdfPages(pdf_path) as pdf:
        for idx, (row, intervals) in enumerate(zip(rows, interval_rows), start=1):
            local_x = [float(row["ego_x"])]
            for item in intervals:
                local_x.extend([item["x_lo"], item["x_hi"]])
            vehicles, neighbors = kin_to_absolute(row, lane_width)
            pred = predicted_neighbors(neighbors, high_interval, dt)
            local_x.extend([float(v["x"]) for v in vehicles])
            local_x.extend([float(v["x"]) for v in pred])
            goal = row.get("goal_phys", [])
            if len(goal) >= 1:
                gx = finite_float(goal[0])
                if gx is not None:
                    local_x.append(gx)
            fig, ax = plt.subplots(figsize=(10.5, 3.2), constrained_layout=True)
            draw_interval_panel(
                ax,
                row,
                intervals,
                n_lanes,
                lane_width,
                goal_x,
                high_interval,
                dt,
                speed_limit,
                (min(local_x) - 10.0, max(local_x) + 10.0),
            )
            fig.suptitle(f"Episode {episode:04d}, high interval {idx}/{len(rows)}")
            sm = cm.ScalarMappable(
                cmap=plt.get_cmap("jet"),
                norm=mcolors.Normalize(vmin=0.0, vmax=max(float(speed_limit), 1e-3)),
            )
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, aspect=30, shrink=0.8, pad=0.02)
            cbar.set_label("Speed [m/s]", fontsize=10)
            pdf.savefig(fig)
            png_path = output_dir / f"ep{episode:04d}_interval{row['interval_idx']:02d}_localstep{row['local_step']:05d}.png"
            fig.savefig(png_path, dpi=180)
            plt.close(fig)

    print(f"Wrote {overview_path}")
    print(f"Wrote {timeline_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {summary_csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot high-level reachable goal feasible regions for one HIRO evaluation episode.")
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.eval_dir / f"high_reachable_feasible_ep{args.episode:04d}"
    plot_all(args.eval_dir, args.episode, output_dir)


if __name__ == "__main__":
    main()
