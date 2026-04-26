import os
import shutil
import csv
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from typing import Any
from pathlib import Path
from datetime import datetime
from configs.conf import get_hiro_config
from rl.algos.HRL.high_goal_safe_bounds import HighGoalSafeBoundsCalculator


def _parse_json_array(raw: str) -> np.ndarray:
    text = str(raw).strip()
    if not text:
        return np.asarray([], dtype=np.float32)
    try:
        obj = json.loads(text)
        return np.asarray(obj, dtype=np.float32)
    except Exception:
        if text.startswith("[") and text.endswith("]"):
            return np.fromstring(text[1:-1], sep=",", dtype=np.float32)
        return np.asarray([], dtype=np.float32)


def _squeeze_to_1d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    while a.ndim > 1 and a.shape[0] == 1:
        a = a[0]
    return a


def _infer_lane_centers_from_kin(kin: np.ndarray, lane_count: int = 3) -> np.ndarray:
    if kin.ndim != 2 or kin.shape[1] < 3:
        return np.asarray([0.0, 4.0, 8.0], dtype=np.float32)

    presence = kin[:, 0] if kin.shape[1] >= 1 else np.ones((kin.shape[0],), dtype=np.float32)
    y_all = kin[presence > 0.5, 2]
    if y_all.size == 0:
        return np.asarray([0.0, 4.0, 8.0], dtype=np.float32)

    y_round = np.round(y_all.astype(np.float32), 1)
    uniq, counts = np.unique(y_round, return_counts=True)
    order = np.argsort(counts)[::-1]
    top = np.sort(uniq[order[: max(1, int(lane_count))]])

    if top.size >= lane_count:
        return top[:lane_count].astype(np.float32)

    # Fallback: extend by estimated lane width around observed values.
    if top.size >= 2:
        lane_w = float(np.median(np.diff(top)))
        if lane_w < 1e-3:
            lane_w = 4.0
    else:
        lane_w = 4.0

    vals = list(top.tolist())
    while len(vals) < lane_count:
        vals.append(vals[-1] + lane_w)
    return np.asarray(vals[:lane_count], dtype=np.float32)


def _infer_lane_width(lane_centers: np.ndarray) -> float:
    c = np.asarray(lane_centers, dtype=np.float32).reshape(-1)
    if c.size >= 2:
        d = np.diff(np.sort(c))
        d = d[d > 1e-3]
        if d.size:
            return float(np.median(d))
    return 4.0


def _convert_kin_xy_to_absolute(kin: np.ndarray, x_margin: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Convert kinematics x/y to absolute coordinates when neighbors are ego-relative.

    In this project, ego x/y is often absolute while neighbors may be stored as relative offsets.
    This helper detects that pattern and converts neighbors to absolute coordinates.
    """
    if kin.ndim != 2 or kin.shape[0] < 1 or kin.shape[1] < 3:
        return None, None

    x_abs = kin[:, 1].astype(np.float32).copy()
    y_abs = kin[:, 2].astype(np.float32).copy()

    ego_x = float(x_abs[0])
    ego_y = float(y_abs[0])

    if kin.shape[0] > 1:
        # x conversion: if others are clustered far from ego_x scale, treat as relative.
        med_other_x = float(np.median(x_abs[1:]))
        if abs(med_other_x - ego_x) > float(max(20.0, x_margin * 0.4)):
            x_abs[1:] = x_abs[1:] + ego_x

        # y conversion: if there are clear negative lane offsets while ego is on positive lane center,
        # neighbors are likely ego-relative y.
        y_other = y_abs[1:]
        if np.any(y_other < -1.0) and ego_y > 1.0:
            y_abs[1:] = y_abs[1:] + ego_y

    return x_abs, y_abs


def _stable_lane_centers_from_abs_y(y_abs: np.ndarray, ego_y: float, lane_count: int = 3) -> np.ndarray:
    """Get stable lane centers to avoid visual gaps caused by noisy inferred widths."""
    y_abs = np.asarray(y_abs, dtype=np.float32).reshape(-1)
    if y_abs.size == 0:
        return np.asarray([0.0, 4.0, 8.0], dtype=np.float32)

    # Prefer canonical 3-lane centers if data fits that regime.
    if float(np.nanmin(y_abs)) > -2.0 and float(np.nanmax(y_abs)) < 10.0:
        return np.asarray([0.0, 4.0, 8.0], dtype=np.float32)

    # Otherwise build centers around ego lane with fixed 4m spacing.
    lane_w = 4.0
    center_idx = int(np.round(float(ego_y) / lane_w))
    return np.asarray([(center_idx - 1) * lane_w, center_idx * lane_w, (center_idx + 1) * lane_w], dtype=np.float32)


def _neighbor_abs_vx_from_kin(kin: np.ndarray) -> np.ndarray | None:
    """Convert neighbor vx to absolute speed for coloring.

    In HIRO kinematic observations, ego vx is absolute while neighbor vx is ego-relative.
    """
    if kin.ndim != 2 or kin.shape[0] < 1 or kin.shape[1] < 4:
        return None

    vx_abs = kin[:, 3].astype(np.float32).copy()
    if kin.shape[0] > 1:
        ego_vx = float(vx_abs[0])
        vx_abs[1:] = vx_abs[1:] + ego_vx
    return vx_abs


def _predict_neighbor_positions_one_interval(
    kin: np.ndarray,
    x_abs: np.ndarray,
    y_abs: np.ndarray,
    vx_abs: np.ndarray | None,
    high_interval: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict neighbor positions after one high interval in world coordinates.

    Uses x_abs(t+h) = x_abs(t) + vx_abs * h.
    """
    if kin.ndim != 2 or kin.shape[0] < 2 or kin.shape[1] < 4:
        return (
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
        )

    horizon_t = float(high_interval) * float(dt)
    presence = kin[:, 0] if kin.shape[1] >= 1 else np.ones((kin.shape[0],), dtype=np.float32)
    if vx_abs is None or vx_abs.size != kin.shape[0]:
        vx_abs = _neighbor_abs_vx_from_kin(kin)
    if vx_abs is None or vx_abs.size != kin.shape[0]:
        return (
            np.asarray([], dtype=np.int32),
            np.asarray([], dtype=np.float32),
            np.asarray([], dtype=np.float32),
        )

    idx_list: list[int] = []
    px_list: list[float] = []
    py_list: list[float] = []
    for i in range(1, kin.shape[0]):
        if float(presence[i]) <= 0.5:
            continue
        idx_list.append(i)
        px_list.append(float(x_abs[i] + vx_abs[i] * horizon_t))
        py_list.append(float(y_abs[i]))

    return (
        np.asarray(idx_list, dtype=np.int32),
        np.asarray(px_list, dtype=np.float32),
        np.asarray(py_list, dtype=np.float32),
    )


def render_high_interval_debug_snapshot(
    row: dict[str, Any],
    save_path: str,
    x_margin: float = 70.0,
    veh_length: float = 5.0,
    veh_width: float = 2.0,
    goal_marker_size: float = 85.0,
    high_interval: int = 25,
    dt: float = 0.1,
) -> None:
    """Render one high-interval-start debug snapshot.

    Figure content:
    - Ego and nearby vehicles from `kin`
    - Goal position from `goal_phys`
    - Safe goal region (3 lane rectangles) from `safe_dx_l2/safe_dx_u2`
    """
    kin = _parse_json_array(row.get("kin", ""))
    goal_phys = _squeeze_to_1d(_parse_json_array(row.get("goal_phys", "")))
    goal_phys_samples = _parse_json_array(row.get("goal_phys_samples", ""))
    safe_dx_l2 = _squeeze_to_1d(_parse_json_array(row.get("safe_dx_l2", "")))
    safe_dx_u2 = _squeeze_to_1d(_parse_json_array(row.get("safe_dx_u2", "")))

    fig, ax = plt.subplots(figsize=(15, 3))
    fig.patch.set_facecolor("#d9d9d9")
    ax.set_facecolor("#d9d9d9")

    ego_x = 0.0
    ego_y = 0.0
    if kin.ndim == 2 and kin.shape[1] >= 3 and kin.shape[0] >= 1:
        ego_x = float(kin[0, 1])
        ego_y = float(kin[0, 2])

    x_abs, y_abs = _convert_kin_xy_to_absolute(kin, x_margin=float(x_margin))

    if y_abs is not None and y_abs.size:
        lane_centers = _stable_lane_centers_from_abs_y(y_abs, ego_y=ego_y, lane_count=3)
    else:
        lane_centers = _infer_lane_centers_from_kin(kin, lane_count=3)
    lane_w = 4.0

    x0 = ego_x - float(x_margin)
    x1 = ego_x + float(x_margin)

    # Draw lane backgrounds and separators.
    for y in lane_centers:
        rect = patches.Rectangle(
            (x0, float(y - lane_w / 2.0)),
            float(x1 - x0),
            float(lane_w),
            facecolor="#666666",
            edgecolor="none",
            zorder=0,
        )
        ax.add_patch(rect)

    y_min = float(np.min(lane_centers) - lane_w / 2.0)
    y_max = float(np.max(lane_centers) + lane_w / 2.0)
    for y in lane_centers:
        ax.plot([x0, x1], [float(y), float(y)], color="white", linestyle="--", linewidth=1.0, zorder=1)
    ax.plot([x0, x1], [y_min, y_min], color="white", linewidth=1.0, zorder=1)
    ax.plot([x0, x1], [y_max, y_max], color="white", linewidth=1.0, zorder=1)

    # Draw safe-goal regions: components are relative [left, keep, right].
    safe_colors = ["#66ff66", "#44dd44", "#66ff66"]
    lane_range_lines: list[str] = []
    rel_names = ["left", "keep", "right"]
    rel_offsets = [-1, 0, 1]
    ego_lane_idx = int(np.argmin(np.abs(lane_centers - float(ego_y))))
    for comp_i in range(min(3, safe_dx_l2.size, safe_dx_u2.size)):
        target_lane = ego_lane_idx + int(rel_offsets[comp_i])
        if target_lane < 0 or target_lane >= lane_centers.size:
            lane_range_lines.append(f"{rel_names[comp_i]}: infeasible (out of road)")
            continue

        # safe_dx_* in debug CSV is rel_x; convert to absolute x by adding ego_x.
        xl = float(ego_x + safe_dx_l2[comp_i])
        xu = float(ego_x + safe_dx_u2[comp_i])
        if np.isnan(xl) or np.isnan(xu) or xl >= xu:
            lane_range_lines.append(f"{rel_names[comp_i]}: infeasible")
            continue
        lane_range_lines.append(
            f"{rel_names[comp_i]} (lane{target_lane}): x=[{xl:.1f}, {xu:.1f}]"
        )
        y = float(lane_centers[target_lane])
        safe_rect = patches.Rectangle(
            (xl, y - lane_w / 2.0),
            xu - xl,
            lane_w,
            facecolor=safe_colors[comp_i % len(safe_colors)],
            edgecolor="#00dd00",
            linestyle="--",
            linewidth=1.8,
            alpha=0.28,
            zorder=2,
        )
        ax.add_patch(safe_rect)

    # Draw vehicles.
    speed_limit = 15.0
    norm = mcolors.Normalize(vmin=0.0, vmax=speed_limit)
    cmap = cm.get_cmap("jet")

    if kin.ndim == 2 and kin.shape[1] >= 4 and x_abs is not None and y_abs is not None:
        presence = kin[:, 0] if kin.shape[1] >= 1 else np.ones((kin.shape[0],), dtype=np.float32)
        vx_abs = _neighbor_abs_vx_from_kin(kin)
        for i in range(kin.shape[0]):
            if presence[i] <= 0.5:
                continue
            x_i = float(x_abs[i])
            y_i = float(y_abs[i])
            vx_i = float(vx_abs[i]) if vx_abs is not None else float(kin[i, 3])
            color = cmap(norm(max(0.0, vx_i)))
            edge = "red" if i == 0 else "black"
            lw = 1.6 if i == 0 else 1.0
            veh_rect = patches.Rectangle(
                (x_i - veh_length / 2.0, y_i - veh_width / 2.0),
                veh_length,
                veh_width,
                facecolor=color,
                edgecolor=edge,
                linewidth=lw,
                alpha=0.95,
                zorder=4,
            )
            ax.add_patch(veh_rect)

        # Draw predicted neighbor positions after one high interval.
        pred_idx, pred_x, pred_y = _predict_neighbor_positions_one_interval(
            kin=kin,
            x_abs=x_abs,
            y_abs=y_abs,
            vx_abs=vx_abs,
            high_interval=int(high_interval),
            dt=float(dt),
        )
        pred_size = float(max(1.6, veh_width * 1.05))
        for k in range(pred_idx.size):
            i = int(pred_idx[k])
            vx_i = float(vx_abs[i]) if vx_abs is not None else float(kin[i, 3])
            pred_color = cmap(norm(max(0.0, vx_i)))
            px_i = float(pred_x[k])
            py_i = float(pred_y[k])
            pred_rect = patches.Rectangle(
                (px_i - pred_size / 2.0, py_i - pred_size / 2.0),
                pred_size,
                pred_size,
                facecolor=pred_color,
                edgecolor="white",
                linewidth=0.9,
                alpha=0.35,
                zorder=5,
            )
            ax.add_patch(pred_rect)
            ax.text(
                px_i + pred_size * 0.55,
                py_i - pred_size * 0.55,
                f"pred{i}: x={px_i:.1f}, v={vx_i:.1f}",
                fontsize=8,
                color="white",
                ha="left",
                va="center",
                zorder=7,
                bbox=dict(facecolor="black", alpha=0.35, pad=1.5, edgecolor="none"),
            )

    # Draw goal.
    samples_vis_jittered = False
    samples_spread_x = 0.0
    samples_spread_y = 0.0
    if goal_phys_samples.size > 0:
        g_samples = np.asarray(goal_phys_samples, dtype=np.float32).reshape(-1, 4)
        g_vis = g_samples.copy()
        samples_spread_x = float(np.ptp(g_samples[:, 0])) if g_samples.shape[0] > 0 else 0.0
        samples_spread_y = float(np.ptp(g_samples[:, 1])) if g_samples.shape[0] > 0 else 0.0

        # If samples almost overlap exactly, spread markers on a tiny ring for visualization only.
        if g_samples.shape[0] > 1 and samples_spread_x < 1e-3 and samples_spread_y < 1e-3:
            n = int(g_samples.shape[0])
            theta = np.linspace(0.0, 2.0 * np.pi, num=n, endpoint=False, dtype=np.float32)
            radius_x = 0.45
            radius_y = 0.22
            g_vis[:, 0] = g_samples[:, 0] + radius_x * np.cos(theta)
            g_vis[:, 1] = g_samples[:, 1] + radius_y * np.sin(theta)
            samples_vis_jittered = True

        ax.scatter(
            g_vis[:, 0],
            g_vis[:, 1],
            s=max(float(goal_marker_size) * 0.38, 20.0),
            marker="o",
            facecolors="none",
            edgecolors="gold",
            linewidths=1.0,
            alpha=0.75,
            zorder=5,
        )

    if goal_phys.size >= 2:
        gx = float(goal_phys[0])
        gy = float(goal_phys[1])
        ax.scatter([gx], [gy], s=goal_marker_size, marker="o", c="orangered", edgecolors="white", linewidths=1.4, zorder=6)
        goal_label = f"goal: ({gx:.1f}, {gy:.1f})"
        if goal_phys.size >= 4:
            gvx = float(goal_phys[2])
            gvy = float(goal_phys[3])
            goal_label = f"goal: ({gx:.1f}, {gy:.1f}, {gvx:.1f}, {gvy:.1f})"
        ax.text(
            gx + 1.8,
            gy - 0.4,
            goal_label,
            fontsize=9,
            color="white",
            ha="left",
            va="center",
            zorder=8,
            bbox=dict(facecolor="black", alpha=0.45, pad=1.8, edgecolor="none"),
        )

    if goal_phys_samples.size > 0:
        g_samples = np.asarray(goal_phys_samples, dtype=np.float32).reshape(-1, 4)
        jitter_note = " (vis spread)" if samples_vis_jittered else ""
        ax.text(
            0.99,
            0.98,
            (
                f"goal samples: n={int(g_samples.shape[0])}{jitter_note}\n"
                f"spread: dx={samples_spread_x:.4f}, dy={samples_spread_y:.4f}"
            ),
            transform=ax.transAxes,
            fontsize=9,
            color="white",
            ha="right",
            va="top",
            zorder=9,
            bbox=dict(facecolor="black", alpha=0.45, pad=2.2, edgecolor="none"),
        )

    # Annotate ego current state.
    if kin.ndim == 2 and kin.shape[0] >= 1 and kin.shape[1] >= 4 and x_abs is not None and y_abs is not None:
        ego_vx_abs = float(kin[0, 3])
        ego_info = f"ego: x={float(x_abs[0]):.1f}, y={float(y_abs[0]):.1f}, vx={ego_vx_abs:.1f}"
        ax.text(
            0.01,
            0.02,
            ego_info,
            transform=ax.transAxes,
            fontsize=9,
            color="white",
            ha="left",
            va="bottom",
            zorder=9,
            bbox=dict(facecolor="black", alpha=0.45, pad=2.5, edgecolor="none"),
        )

    # Formatting similar to existing debug snapshots.
    step = row.get("step", "")
    hi_s = row.get("hi_start_saved", "")
    seg = row.get("segment_id", "")
    ax.set_title(f"High-Interval Start {hi_s} | Step {step} | Seg {seg}", fontsize=12)

    if lane_range_lines:
        ax.text(
            0.01,
            0.98,
            "feasible x-ranges\n" + "\n".join(lane_range_lines),
            transform=ax.transAxes,
            fontsize=9,
            color="white",
            ha="left",
            va="top",
            zorder=9,
            bbox=dict(facecolor="black", alpha=0.45, pad=2.5, edgecolor="none"),
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y_min - 1.0, y_max + 1.0)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, aspect=30, shrink=0.8, pad=0.02)
    cbar.set_label("Speed [m/s]", fontsize=10)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def batch_render_high_interval_debug_csv(
    csv_path: str,
    debug_root: str,
    n_last: int = 100,
) -> str:
    """Render last N rows in high_interval_debug.csv to a datetime folder.

    Returns the output directory path.
    """
    csv_p = Path(csv_path)
    if not csv_p.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_p.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"CSV is empty: {csv_path}")

    rows = rows[-int(max(1, n_last)):]

    out_dir = Path(debug_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(rows):
        fn = f"{i:03d}_hiStart{row.get('hi_start_saved', i)}_step{row.get('step', 'na')}.png"
        render_high_interval_debug_snapshot(row=row, save_path=str(out_dir / fn))

    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"csv_path={csv_p}\n")
        f.write(f"samples={len(rows)}\n")
        f.write(f"output_dir={out_dir}\n")

    return str(out_dir)


def save_low_step_snapshot(
    env,
    runner,
    step: int,
    out_dir: str,
    goal_phys: np.ndarray,
    reward_sums: dict[str, float] | None = None,
    title_suffix: str = "",
):
    base_env = env.unwrapped
    road = base_env.road
    ego = base_env.vehicle

    os.makedirs(out_dir, exist_ok=True)

    p_dist = getattr(base_env, "PERCEPTION_DISTANCE", 200.0)
    if p_dist is None:
        p_dist = 200.0
    p_dist = float(p_dist)

    all_draw_vehs = list(road.vehicles)

    fig, ax = plt.subplots(figsize=(15, 3))

    lanes = road.network.lanes_list()
    ys = []
    for lane in lanes:
        x0, y0 = lane.start
        x1, y1 = lane.end
        w = lane.width
        heading = lane.heading_at(0)
        c, s = np.cos(heading), np.sin(heading)

        normal = np.array([-s, c])

        p0 = lane.start - normal * w / 2
        p1 = lane.start + normal * w / 2
        p2 = lane.end + normal * w / 2
        p3 = lane.end - normal * w / 2
        poly = patches.Polygon([p0, p1, p2, p3], closed=True, facecolor="#666666", edgecolor="none", zorder=0)
        ax.add_patch(poly)

        types = [str(t) for t in lane.line_types]

        def _draw_line(pa, pb, ltype):
            if "NONE" in ltype:
                return
            style = "solid" if "CONTINUOUS" in ltype else "dashed"
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="white", linestyle=style, linewidth=1, zorder=1)

        _draw_line(p0, p3, types[0])
        _draw_line(p1, p2, types[1])

        ys.append(y0)
        ys.append(y1)

    speed_limit = float(base_env.config.get("speed_limit", 30.0))
    norm = mcolors.Normalize(vmin=0.0, vmax=speed_limit)
    cmap = cm.get_cmap("jet")

    for v in all_draw_vehs:
        hist = list(reversed(getattr(v, "history", [])))
        if len(hist) >= 2:
            traj = np.array([h.position for h in hist], dtype=float)
            if v is ego:
                ax.plot(traj[:, 0], traj[:, 1], color="red", linewidth=1.2, alpha=0.9, zorder=3)
            else:
                ax.plot(traj[:, 0], traj[:, 1], color="#A0A0A0", linewidth=0.8, alpha=0.6, zorder=2)

        edge_c = "black"
        lw = 1
        z = 5
        alpha_v = 0.9

        if v is ego:
            color = cmap(norm(v.speed))
            edge_c = "red"
            z = 6
        else:
            color = cmap(norm(v.speed))

        l, w = v.LENGTH, v.WIDTH
        rect = patches.Rectangle((-l / 2, -w / 2), l, w, facecolor=color, edgecolor=edge_c, linewidth=lw, alpha=alpha_v, zorder=z)
        t = transforms.Affine2D().rotate(v.heading).translate(v.position[0], v.position[1]) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    gx, gy, gvx, gvy = [float(x) for x in goal_phys[:4]]
    goal_color = cmap(norm(gvx))
    ax.scatter([gx], [gy], c=[goal_color], marker="o", s=50, linewidth=1.5, edgecolors="white", zorder=10)

    # Overlay high-level goal feasible region computed from current reachable-set bounds.
    try:
        hiro_cfg = get_hiro_config()
        lane_width = float(base_env.config.get("lane_width", 4.0))
        n_lanes = int(base_env.config.get("lanes_count", 3))
        lane_centers = np.arange(n_lanes, dtype=np.float32) * lane_width
        ego_lane_idx = int(np.argmin(np.abs(lane_centers - float(ego.position[1]))))

        obs_features = list(base_env.config.get("observation", {}).get("features", ["presence", "x", "y", "vx", "vy"]))

        def _fidx(name: str, default: int) -> int:
            try:
                return int(obs_features.index(name))
            except ValueError:
                return int(default)

        policy_freq = float(base_env.config.get("policy_frequency", 10.0))
        dt = 1.0 / max(policy_freq, 1e-6)
        hi = int(getattr(runner, "hi", getattr(hiro_cfg, "high_interval", 25)))
        horizon_t = float(hi) * float(dt)

        speed_limit = float(base_env.config.get("speed_limit", 15.0))
        v_min = 0.0
        v_max = speed_limit
        dx_low = float(v_min * horizon_t)
        dx_high = float(v_max * horizon_t)

        action_cfg = base_env.config.get("action", {})
        accel_range = action_cfg.get("acceleration_range", [-5.0, 5.0])
        default_max_accel = float(max(abs(float(accel_range[0])), abs(float(accel_range[1]))))
        use_custom_kin = bool(getattr(hiro_cfg, "high_goal_safe_use_custom_kinematics", False))
        if use_custom_kin:
            cfg_max_accel = getattr(hiro_cfg, "high_goal_safe_max_accel", None)
            cfg_max_decel = getattr(hiro_cfg, "high_goal_safe_max_decel", None)
            max_accel = float(default_max_accel if cfg_max_accel is None else max(float(cfg_max_accel), 0.0))
            max_decel = float(default_max_accel if cfg_max_decel is None else max(float(cfg_max_decel), 0.0))
        else:
            max_accel = float(default_max_accel)
            max_decel = float(default_max_accel)

        calc = HighGoalSafeBoundsCalculator(
            n_lanes=n_lanes,
            lane_width=lane_width,
            high_interval=hi,
            dt=dt,
            speed_min=v_min,
            speed_max=v_max,
            max_accel=max_accel,
            max_decel=max_decel,
            front_dmin=float(max(0.0, getattr(hiro_cfg, "high_goal_safe_front_dmin", 0.0))),
            lane_change_rear_dmin=float(max(0.0, getattr(hiro_cfg, "high_goal_safe_lane_change_rear_dmin", 0.0))),
            dx_low=dx_low,
            dx_high=dx_high,
            feat_dim=int(len(obs_features)),
            presence_idx=int(_fidx("presence", 0)),
            x_idx=int(_fidx("x", 1)),
            y_idx=int(_fidx("y", 2)),
            vx_idx=int(_fidx("vx", 3)),
            vy_idx=int(_fidx("vy", 4)),
        )

        high_obs_now = np.asarray(base_env.observation_type.observe(), dtype=np.float32).reshape(1, -1)
        bounds = calc.compute_np(high_obs_now)
        l2 = np.asarray(bounds.get("l2", [[1.0, 1.0, 1.0]]), dtype=np.float32).reshape(1, 3)[0]
        u2 = np.asarray(bounds.get("u2", [[-1.0, -1.0, -1.0]]), dtype=np.float32).reshape(1, 3)[0]

        rel_offsets = [-1, 0, 1]
        rel_names = ["left", "keep", "right"]
        feasible_lines: list[str] = []
        denom = max(float(dx_high - dx_low), 1e-6)
        for comp_i in range(3):
            target_lane = ego_lane_idx + int(rel_offsets[comp_i])
            if target_lane < 0 or target_lane >= n_lanes:
                feasible_lines.append(f"{rel_names[comp_i]}: infeasible (out of road)")
                continue

            lo_n = float(l2[comp_i])
            hi_n = float(u2[comp_i])
            lo_dx = float(dx_low + 0.5 * (lo_n + 1.0) * denom)
            hi_dx = float(dx_low + 0.5 * (hi_n + 1.0) * denom)
            xl = float(ego.position[0] + lo_dx)
            xu = float(ego.position[0] + hi_dx)
            if (not np.isfinite(xl)) or (not np.isfinite(xu)) or (xl >= xu):
                feasible_lines.append(f"{rel_names[comp_i]}: infeasible")
                continue

            feasible_lines.append(f"{rel_names[comp_i]}: x=[{xl:.1f}, {xu:.1f}]")
            y_center = float(lane_centers[target_lane])
            rect = patches.Rectangle(
                (xl, y_center - lane_width / 2.0),
                xu - xl,
                lane_width,
                facecolor="#66ff66",
                edgecolor="#00cc00",
                linestyle="--",
                linewidth=1.5,
                alpha=0.25,
                zorder=2,
            )
            ax.add_patch(rect)

        if feasible_lines:
            ax.text(
                0.01,
                0.98,
                "goal feasible set\n" + "\n".join(feasible_lines),
                transform=ax.transAxes,
                fontsize=9,
                color="white",
                bbox=dict(facecolor="black", alpha=0.55, pad=3, edgecolor="none"),
                ha="left",
                va="top",
                zorder=20,
            )
    except Exception:
        # Snapshot rendering should remain robust even if feasible-set computation fails.
        pass

    dx = float(gx - ego.position[0])
    dy = float(gy - ego.position[1])
    ego_vx, ego_vy = [float(v) for v in ego.velocity]
    info_text = (
        f"Δx={dx:.2f} m, Δy={dy:.2f} m\n"
        f"ego vx={ego_vx:.2f} m/s, vy={ego_vy:.2f} m/s"
    )
    if reward_sums:
        reward_lines = ["cum rewards:"]
        for k, v in reward_sums.items():
            reward_lines.append(f"{k}: {v:.4f}")
        info_text = info_text + "\n" + "\n".join(reward_lines)
    ax.text(
        0.01,
        0.01,
        info_text,
        transform=ax.transAxes,
        fontsize=10,
        color="white",
        bbox=dict(facecolor="black", alpha=0.6, pad=3, edgecolor="none"),
        ha="left",
        va="bottom",
        zorder=20,
    )

    x_min = ego.position[0] - p_dist
    x_max = ego.position[0] + p_dist
    ax.set_xlim(x_min, x_max)

    if ys:
        mean_y = np.mean(ys)
        ax.set_ylim(mean_y - 12, mean_y + 12)
    else:
        ax.set_ylim(-10, 10)

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, aspect=30, shrink=0.8, pad=0.02)
    cbar.set_label("Speed [m/s]", fontsize=10)

    title = f"Step {step}" + (f" | {title_suffix}" if title_suffix else "")
    plt.title(title)

    save_path = os.path.join(out_dir, f"step{step:05d}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def save_goal_candidates_snapshot(
    env,
    step: int,
    out_dir: str,
    goals_phys: np.ndarray,
    scores: np.ndarray,
    title_suffix: str = "goal candidates",
):
    base_env = env.unwrapped
    road = base_env.road
    ego = base_env.vehicle

    os.makedirs(out_dir, exist_ok=True)

    p_dist = getattr(base_env, "PERCEPTION_DISTANCE", 200.0)
    if p_dist is None:
        p_dist = 200.0
    p_dist = float(p_dist)

    all_draw_vehs = list(road.vehicles)

    fig, ax = plt.subplots(figsize=(15, 3))

    lanes = road.network.lanes_list()
    ys = []
    for lane in lanes:
        x0, y0 = lane.start
        x1, y1 = lane.end
        w = lane.width
        heading = lane.heading_at(0)
        c, s = np.cos(heading), np.sin(heading)
        normal = np.array([-s, c])

        p0 = lane.start - normal * w / 2
        p1 = lane.start + normal * w / 2
        p2 = lane.end + normal * w / 2
        p3 = lane.end - normal * w / 2
        poly = patches.Polygon([p0, p1, p2, p3], closed=True, facecolor="#666666", edgecolor="none", zorder=0)
        ax.add_patch(poly)

        types = [str(t) for t in lane.line_types]

        def _draw_line(pa, pb, ltype):
            if "NONE" in ltype:
                return
            style = "solid" if "CONTINUOUS" in ltype else "dashed"
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="white", linestyle=style, linewidth=1, zorder=1)

        _draw_line(p0, p3, types[0])
        _draw_line(p1, p2, types[1])

        ys.append(y0)
        ys.append(y1)

    speed_limit = float(base_env.config.get("speed_limit", 30.0))
    norm = mcolors.Normalize(vmin=0.0, vmax=speed_limit)
    cmap = cm.get_cmap("jet")

    for v in all_draw_vehs:
        edge_c = "black"
        lw = 1
        z = 5
        alpha_v = 0.9

        if v is ego:
            color = cmap(norm(v.speed))
            edge_c = "red"
            z = 6
        else:
            color = cmap(norm(v.speed))

        l, w = v.LENGTH, v.WIDTH
        rect = patches.Rectangle((-l / 2, -w / 2), l, w, facecolor=color, edgecolor=edge_c, linewidth=lw, alpha=alpha_v, zorder=z)
        t = transforms.Affine2D().rotate(v.heading).translate(v.position[0], v.position[1]) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    goals_phys = np.asarray(goals_phys, dtype=np.float32).reshape(-1, 4)
    if goals_phys.size > 0:
        s_min = float(np.min(scores))
        s_max = float(np.max(scores))
        denom = (s_max - s_min) if s_max > s_min else 1.0
        s_norm = (scores - s_min) / denom
        score_cmap = cm.get_cmap("viridis")

        order = np.argsort(scores)
        for idx in order:
            gx, gy = goals_phys[idx, 0], goals_phys[idx, 1]
            color = score_cmap(float(s_norm[idx]))
            ax.scatter([gx], [gy], c=[color], marker="o", s=45, linewidth=0.5, edgecolors="white", zorder=9)

        sm = cm.ScalarMappable(cmap=score_cmap, norm=mcolors.Normalize(vmin=s_min, vmax=s_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, aspect=30, shrink=0.8, pad=0.02)
        cbar.set_label("Goal score", fontsize=10)

    x_min = ego.position[0] - p_dist
    x_max = ego.position[0] + p_dist
    ax.set_xlim(x_min, x_max)

    if ys:
        mean_y = np.mean(ys)
        ax.set_ylim(mean_y - 12, mean_y + 12)
    else:
        ax.set_ylim(-10, 10)

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    title = f"Step {step} | {title_suffix}"
    plt.title(title)

    save_path = os.path.join(out_dir, f"step{step:05d}_goal_candidates.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def save_q_sa_surface_plot(
    out_dir: str,
    step: int,
    a0_mesh: np.ndarray,
    a1_mesh: np.ndarray,
    q_surface: np.ndarray,
    selected_action: np.ndarray | None,
):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf(a0_mesh, a1_mesh, q_surface, levels=40, cmap="viridis")
    ax.contour(a0_mesh, a1_mesh, q_surface, levels=12, colors="k", linewidths=0.3, alpha=0.35)

    if selected_action is not None and selected_action.size >= 2:
        a0 = float(selected_action[0])
        a1 = float(selected_action[1])
        ax.scatter([a0], [a1], color="tab:red", s=30, label="chosen action", edgecolors="white", linewidths=0.7)
        ax.legend(loc="best")

    ax.set_xlabel("a[0]")
    ax.set_ylabel("a[1]")
    ax.set_title(f"Q_min(s,a0,a1) Contour - step {int(step)}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Q_min")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"q_sa_surface_step{int(step):05d}.png"), dpi=150)
    plt.close(fig)


def save_q_sa_global_summary(
    out_dir: str,
    a0_mesh: np.ndarray,
    a1_mesh: np.ndarray,
    q_surface_mean: np.ndarray,
    q_surface_std: np.ndarray,
):
    if q_surface_mean.size == 0:
        return

    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf(a0_mesh, a1_mesh, q_surface_mean, levels=40, cmap="viridis")
    ax.set_xlabel("a[0]")
    ax.set_ylabel("a[1]")
    ax.set_title("Mean Q_min(s,a0,a1) across rollout")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean Q_min")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "q_sa_surface_mean_contour.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(a0_mesh, a1_mesh, q_surface_mean, cmap="viridis", linewidth=0.0, antialiased=True, alpha=0.95)
    ax.set_xlabel("a[0]")
    ax.set_ylabel("a[1]")
    ax.set_zlabel("mean Q_min")
    ax.set_title("Mean Q_min(s,a0,a1) Surface")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "q_sa_surface_mean_3d.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.contourf(a0_mesh, a1_mesh, q_surface_std, levels=40, cmap="magma")
    ax.set_xlabel("a[0]")
    ax.set_ylabel("a[1]")
    ax.set_title("Std(Q_min) across rollout")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("std Q_min")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "q_sa_surface_std_contour.png"), dpi=150)
    plt.close(fig)


def save_goal_metric_summary(
    env,
    goals_phys: list[np.ndarray],
    metrics: list[float],
    out_path: str,
    metric_name: str,
):
    base_env = env.unwrapped
    road = base_env.road

    if not goals_phys:
        return

    fig, ax = plt.subplots(figsize=(15, 3))

    lanes = road.network.lanes_list()
    ys = []
    for lane in lanes:
        x0, y0 = lane.start
        x1, y1 = lane.end
        w = lane.width
        heading = lane.heading_at(0)
        c, s = np.cos(heading), np.sin(heading)
        normal = np.array([-s, c])

        p0 = lane.start - normal * w / 2
        p1 = lane.start + normal * w / 2
        p2 = lane.end + normal * w / 2
        p3 = lane.end - normal * w / 2
        poly = patches.Polygon([p0, p1, p2, p3], closed=True, facecolor="#666666", edgecolor="none", zorder=0)
        ax.add_patch(poly)

        types = [str(t) for t in lane.line_types]

        def _draw_line(pa, pb, ltype):
            if "NONE" in ltype:
                return
            style = "solid" if "CONTINUOUS" in ltype else "dashed"
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="white", linestyle=style, linewidth=1, zorder=1)

        _draw_line(p0, p3, types[0])
        _draw_line(p1, p2, types[1])

        ys.append(y0)
        ys.append(y1)

    goals_arr = np.asarray(goals_phys, dtype=np.float32).reshape(-1, 4)
    xs = goals_arr[:, 0]
    ys_goal = goals_arr[:, 1]
    metrics_arr = np.asarray(metrics, dtype=np.float32).reshape(-1)
    if metrics_arr.size == 0:
        return

    vmin = float(np.min(metrics_arr))
    vmax = float(np.max(metrics_arr))
    if abs(vmax - vmin) < 1e-6:
        vmax = vmin + 1e-6
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("viridis")

    ax.scatter(xs, ys_goal, c=metrics_arr, cmap=cmap, norm=norm, s=60, edgecolors="white", linewidths=0.8, zorder=5)

    x_min = 0.0
    x_max = float(base_env.config.get("road_length", 500.0))
    ax.set_xlim(x_min, x_max)

    if ys:
        mean_y = np.mean(ys)
        ax.set_ylim(mean_y - 12, mean_y + 12)
    else:
        ax.set_ylim(-10, 10)

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, aspect=30, shrink=0.8, pad=0.02)
    cbar.set_label(metric_name, fontsize=10)

    plt.title(f"Goal metric summary: {metric_name}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close()

def plot_ego_speed_history(env):
    ego = env.unwrapped.vehicle          # ego 车对象
    hist = list(reversed(ego.history))   # deque -> list
    speeds = [v.speed for v in hist]
    dt = 1.0 / env.unwrapped.config["simulation_frequency"]
    times = [i * dt for i in range(len(speeds))]
    plt.plot(times, speeds)
    plt.xlabel("Time [s]")
    plt.ylabel("Ego speed [m/s]")
    plt.grid(True)
    plt.show()

def plot_all_speed_history(env):
    dt = 1.0 / env.unwrapped.config["simulation_frequency"]
    vehs = env.unwrapped.road.vehicles  # 所有车辆
    for v in vehs:
        hist = list(reversed(v.history))
        speeds = [t.speed for t in hist]
        times = [i * dt for i in range(len(speeds))]
        if v == env.unwrapped.vehicle:
            plt.plot(times, speeds, color='r')
        else:
            plt.plot(times, speeds, color='b')

    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.grid(True)
    plt.show()


def plot_warmup_avg_speed(env, show=True, save_path=None):
    base_env = env.unwrapped
    times = getattr(base_env, "_warmup_times", None)
    avg_speeds = getattr(base_env, "_warmup_avg_speeds", None)
    if times is None or avg_speeds is None:
        raise RuntimeError("env 中没有 warmup 统计信息，请确认已经执行过第一次 reset。")
    plt.figure()
    plt.plot(times, avg_speeds)
    plt.xlabel("Time [s]")
    plt.ylabel("Average speed [m/s]")
    plt.title("Warmup average speed vs time")
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def save_speed_acc_curves(env, ep_idx: int, model_path: str, comparison_data: dict | None = None):
    """
    在 show_trajectories 打开的情况下，将当前 episode 的
    车速曲线、加速度曲线和所在车道随时间变化曲线保存到：
        model_path/speed_curve/epXXX_speed.png
        model_path/acc_curve/epXXX_acc.png
        model_path/lane_curve/epXXX_lane.png

    - show_trajectories == 'all'：一张图上画所有车辆（ego 为红色，其它车辆为蓝色）
    - show_trajectories == True：只画 ego 车辆
    - show_trajectories == False：不做任何事
    """
    base_env = env.unwrapped
    road = base_env.road

    show_mode = base_env.config.get("show_trajectories", False)
    if not show_mode:
        # 未开启轨迹记录，直接返回
        return

    # 创建保存目录
    speed_dir = os.path.join(model_path, "speed_curve")
    acc_dir = os.path.join(model_path, "acc_curve")
    lane_dir = os.path.join(model_path, "lane_curve")
    os.makedirs(speed_dir, exist_ok=True)
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(lane_dir, exist_ok=True)

    speed_path = os.path.join(speed_dir, f"ep{ep_idx:03d}_speed.png")
    acc_path = os.path.join(acc_dir, f"ep{ep_idx:03d}_acc.png")
    lane_path = os.path.join(lane_dir, f"ep{ep_idx:03d}_lane.png")

    # 时间步长按 simulation_frequency 计算（与 history 记录频率一致）
    dt = 1.0 / float(base_env.config["simulation_frequency"])

    # 对比模式：RL / RL+safety / RL safety upper / MPC / MPC safety upper
    if isinstance(comparison_data, dict) and comparison_data:
        speed_rl = np.asarray(comparison_data.get("speed_rl", []), dtype=float).reshape(-1)
        speed_rl_safety_output = np.asarray(comparison_data.get("speed_rl_safety_output", []), dtype=float).reshape(-1)
        speed_safety_upper_rl = np.asarray(comparison_data.get("speed_safety_upper_rl", []), dtype=float).reshape(-1)
        speed_mpc = np.asarray(comparison_data.get("speed_mpc", []), dtype=float).reshape(-1)
        speed_safety_upper_mpc = np.asarray(comparison_data.get("speed_safety_upper_mpc", []), dtype=float).reshape(-1)
        speed_mpc_alternatives_raw = comparison_data.get("speed_mpc_alternatives", [])

        acc_rl = np.asarray(comparison_data.get("acc_rl", []), dtype=float).reshape(-1)
        acc_rl_safety_output = np.asarray(comparison_data.get("acc_rl_safety_output", []), dtype=float).reshape(-1)
        acc_safety_upper_rl = np.asarray(comparison_data.get("acc_safety_upper_rl", []), dtype=float).reshape(-1)
        acc_mpc = np.asarray(comparison_data.get("acc_mpc", []), dtype=float).reshape(-1)
        acc_safety_upper_mpc = np.asarray(comparison_data.get("acc_safety_upper_mpc", []), dtype=float).reshape(-1)
        acc_mpc_alternatives_raw = comparison_data.get("acc_mpc_alternatives", [])

        intrinsic_rl_safety_output = np.asarray(comparison_data.get("intrinsic_rl_safety_output", []), dtype=float).reshape(-1)
        comfort_rl_safety_output = np.asarray(comparison_data.get("comfort_rl_safety_output", []), dtype=float).reshape(-1)
        intrinsic_mpc = np.asarray(comparison_data.get("intrinsic_mpc", []), dtype=float).reshape(-1)
        comfort_mpc = np.asarray(comparison_data.get("comfort_mpc", []), dtype=float).reshape(-1)

        lane_rl = np.asarray(comparison_data.get("lane_rl", []), dtype=float).reshape(-1)
        lane_rl_safety_output = np.asarray(comparison_data.get("lane_rl_safety_output", []), dtype=float).reshape(-1)
        lane_safety_upper_rl = np.asarray(comparison_data.get("lane_safety_upper_rl", []), dtype=float).reshape(-1)
        lane_mpc = np.asarray(comparison_data.get("lane_mpc", []), dtype=float).reshape(-1)
        lane_safety_upper_mpc = np.asarray(comparison_data.get("lane_safety_upper_mpc", []), dtype=float).reshape(-1)
        lane_mpc_alternatives_raw = comparison_data.get("lane_mpc_alternatives", [])

        def _to_curve_list(raw: Any, max_n: int = 3) -> list[np.ndarray]:
            if not isinstance(raw, (list, tuple)):
                return []
            out: list[np.ndarray] = []
            for item in list(raw)[: int(max(0, max_n))]:
                arr = np.asarray(item, dtype=float).reshape(-1)
                if arr.size > 0:
                    out.append(arr)
            return out

        speed_mpc_alternatives = _to_curve_list(speed_mpc_alternatives_raw, max_n=3)
        acc_mpc_alternatives = _to_curve_list(acc_mpc_alternatives_raw, max_n=3)
        lane_mpc_alternatives = _to_curve_list(lane_mpc_alternatives_raw, max_n=3)

        def _to_lane_target_code(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr
            bins = np.array([-1.0, 0.0, 1.0], dtype=float)
            idx = np.argmin(np.abs(arr.reshape(-1, 1) - bins.reshape(1, -1)), axis=1)
            return bins[idx]

        lane_rl = _to_lane_target_code(lane_rl)
        lane_rl_safety_output = _to_lane_target_code(lane_rl_safety_output)
        lane_safety_upper_rl = _to_lane_target_code(lane_safety_upper_rl)
        lane_mpc = _to_lane_target_code(lane_mpc)
        lane_safety_upper_mpc = _to_lane_target_code(lane_safety_upper_mpc)

        # --------- 速度曲线（五组） --------- #
        plt.figure()
        if speed_rl.size > 0:
            t_rl = np.arange(speed_rl.size, dtype=float) * dt
            plt.plot(t_rl, speed_rl, color="tab:blue", linewidth=1.6, label="RL output")
        if speed_rl_safety_output.size > 0:
            t_rl_sf = np.arange(speed_rl_safety_output.size, dtype=float) * dt
            plt.plot(t_rl_sf, speed_rl_safety_output, color="tab:purple", linewidth=1.6, label="RL+safety output")
        if speed_safety_upper_rl.size > 0:
            t_su_rl = np.arange(speed_safety_upper_rl.size, dtype=float) * dt
            plt.plot(t_su_rl, speed_safety_upper_rl, color="tab:orange", linewidth=1.4, linestyle="--", label="RL safety upper")
        if speed_mpc.size > 0:
            t_mpc = np.arange(speed_mpc.size, dtype=float) * dt
            plt.plot(t_mpc, speed_mpc, color="tab:green", linewidth=1.6, label="MPC optimal")
        for i, speed_alt in enumerate(speed_mpc_alternatives, start=1):
            t_alt = np.arange(speed_alt.size, dtype=float) * dt
            plt.plot(
                t_alt,
                speed_alt,
                color="tab:green",
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
                label=f"MPC optimal alt#{i}",
            )
        if speed_safety_upper_mpc.size > 0:
            t_su_mpc = np.arange(speed_safety_upper_mpc.size, dtype=float) * dt
            plt.plot(t_su_mpc, speed_safety_upper_mpc, color="tab:red", linewidth=1.4, linestyle="--", label="MPC safety upper")
        plt.xlabel("Time [s]")
        plt.ylabel("Speed [m/s]")
        plt.ylim(0.0, 16.0)
        plt.title(f"Ego Speed Comparison (ep {ep_idx})")
        plt.grid(True)
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(speed_path)
        plt.close()

        # --------- 加速度曲线（五组） --------- #
        fig, ax = plt.subplots()
        if acc_rl.size > 0:
            t_rl = np.arange(acc_rl.size, dtype=float) * dt
            ax.plot(t_rl, acc_rl, color="tab:blue", linewidth=1.6, label="RL output")
        if acc_rl_safety_output.size > 0:
            t_rl_sf = np.arange(acc_rl_safety_output.size, dtype=float) * dt
            ax.plot(t_rl_sf, acc_rl_safety_output, color="tab:purple", linewidth=1.6, label="RL+safety output")
        if acc_safety_upper_rl.size > 0:
            t_su_rl = np.arange(acc_safety_upper_rl.size, dtype=float) * dt
            ax.plot(t_su_rl, acc_safety_upper_rl, color="tab:orange", linewidth=1.4, linestyle="--", label="RL safety upper")
        if acc_mpc.size > 0:
            t_mpc = np.arange(acc_mpc.size, dtype=float) * dt
            ax.plot(t_mpc, acc_mpc, color="tab:green", linewidth=1.6, label="MPC optimal")
        for i, acc_alt in enumerate(acc_mpc_alternatives, start=1):
            t_alt = np.arange(acc_alt.size, dtype=float) * dt
            ax.plot(
                t_alt,
                acc_alt,
                color="tab:green",
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
                label=f"MPC optimal alt#{i}",
            )
        if acc_safety_upper_mpc.size > 0:
            t_su_mpc = np.arange(acc_safety_upper_mpc.size, dtype=float) * dt
            ax.plot(t_su_mpc, acc_safety_upper_mpc, color="tab:red", linewidth=1.4, linestyle="--", label="MPC safety upper")

        def _fmt_reward_sum(arr: np.ndarray) -> str:
            if arr.size == 0:
                return "N/A"
            return f"{float(np.sum(arr)):.4f}"

        reward_text = (
            "Reward summary\n"
            f"intrinsic (RL+safety): {_fmt_reward_sum(intrinsic_rl_safety_output)}\n"
            f"comfort   (RL+safety): {_fmt_reward_sum(comfort_rl_safety_output)}\n"
            f"intrinsic (MPC): {_fmt_reward_sum(intrinsic_mpc)}\n"
            f"comfort   (MPC): {_fmt_reward_sum(comfort_mpc)}"
        )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Acceleration [m/s²]")
        ax.set_title(f"Ego Acceleration Comparison (ep {ep_idx})")
        ax.grid(True)

        lines_l, labels_l = ax.get_legend_handles_labels()
        if lines_l:
            ax.legend(lines_l, labels_l, loc="best")

        fig.subplots_adjust(right=0.77)
        ax.text(
            1.02,
            0.98,
            reward_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.7"),
        )

        fig.tight_layout(rect=[0.0, 0.0, 0.77, 1.0])
        fig.savefig(acc_path)
        plt.close(fig)

        # --------- 换道曲线（五组） --------- #
        plt.figure()
        if lane_rl.size > 0:
            t_rl = np.arange(lane_rl.size, dtype=float) * dt
            plt.step(t_rl, lane_rl, where="post", color="tab:blue", linewidth=1.6, label="RL output")
        if lane_rl_safety_output.size > 0:
            t_rl_sf = np.arange(lane_rl_safety_output.size, dtype=float) * dt
            plt.step(t_rl_sf, lane_rl_safety_output, where="post", color="tab:purple", linewidth=1.6, label="RL+safety output")
        if lane_safety_upper_rl.size > 0:
            t_su_rl = np.arange(lane_safety_upper_rl.size, dtype=float) * dt
            plt.step(t_su_rl, lane_safety_upper_rl, where="post", color="tab:orange", linewidth=1.4, linestyle="--", label="RL safety upper")
        if lane_mpc.size > 0:
            t_mpc = np.arange(lane_mpc.size, dtype=float) * dt
            plt.step(t_mpc, lane_mpc, where="post", color="tab:green", linewidth=1.6, label="MPC optimal")
        for i, lane_alt in enumerate(lane_mpc_alternatives, start=1):
            t_alt = np.arange(lane_alt.size, dtype=float) * dt
            plt.step(
                t_alt,
                lane_alt,
                where="post",
                color="tab:green",
                linewidth=1.2,
                linestyle="--",
                alpha=0.8,
                label=f"MPC optimal alt#{i}",
            )
        if lane_safety_upper_mpc.size > 0:
            t_su_mpc = np.arange(lane_safety_upper_mpc.size, dtype=float) * dt
            plt.step(t_su_mpc, lane_safety_upper_mpc, where="post", color="tab:red", linewidth=1.4, linestyle="--", label="MPC safety upper")
        plt.xlabel("Time [s]")
        plt.ylabel("Target lane cmd (-1/0/1)")
        plt.title(f"Ego Lane-Change Comparison (ep {ep_idx})")
        plt.grid(True)
        plt.yticks([-1, 0, 1])
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(lane_path)
        plt.close()
        return

    # 根据 show_trajectories 的取值决定绘制哪些车辆
    if show_mode == "all":
        vehicles = list(road.vehicles)
        title_prefix = "All vehicles"
    else:
        vehicles = [base_env.vehicle]
        title_prefix = "Ego"

    # --------- 速度曲线 --------- #
    plt.figure()
    for v in vehicles:
        hist = list(reversed(getattr(v, "history", [])))
        if not hist:
            continue
        speeds = np.asarray([snap.speed for snap in hist], dtype=float)
        if speeds.size == 0:
            continue
        t = np.arange(speeds.size, dtype=float) * dt
        if v is base_env.vehicle:
            plt.plot(t, speeds, color="r", label="ego")
        else:
            plt.plot(t, speeds, color="b", alpha=0.6)

    plt.xlabel("Time [s]")
    plt.ylabel("Speed [m/s]")
    plt.ylim(0.0, 16.0)
    plt.title(f"{title_prefix} Speed vs Time (ep {ep_idx})")
    plt.grid(True)
    if show_mode == "all":
        plt.legend()
    plt.tight_layout()
    plt.savefig(speed_path)
    plt.close()

    # --------- 加速度曲线（由速度数值微分算出） --------- #
    plt.figure()
    for v in vehicles:
        hist = list(reversed(getattr(v, "history", [])))
        if not hist:
            continue
        speeds = np.asarray([snap.speed for snap in hist], dtype=float)
        if speeds.size < 2:
            continue
        # 数值微分：a_t ≈ (v_t - v_{t-1}) / dt
        accs = np.diff(speeds) / dt          # 长度 N-1
        t_acc = np.arange(accs.size, dtype=float) * dt

        if v is base_env.vehicle:
            plt.plot(t_acc, accs, color="r", label="ego")
        else:
            plt.plot(t_acc, accs, color="b", alpha=0.6)

    plt.xlabel("Time [s]")
    plt.ylabel("Acceleration [m/s²]")
    ### set ylim here ###
    plt.ylim(-2.5, 3.5)
    plt.title(f"{title_prefix} Acceleration vs Time (ep {ep_idx})")
    plt.grid(True)
    if show_mode == "all":
        plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    # --------- 车道随时间变化曲线 --------- #
    def _get_lane_id(snap):
        li = getattr(snap, "lane_index", None)
        if li is None:
            return np.nan
        # highwayEnv 风格：lane_index = (from, to, lane_id)
        try:
            if isinstance(li, (tuple, list)) and len(li) >= 3:
                return float(li[2])
            # 其他情况尝试直接转为数值
            return float(li)
        except Exception:
            return np.nan

    plt.figure()
    for v in vehicles:
        hist = list(reversed(getattr(v, "history", [])))
        if not hist:
            continue
        lane_ids = np.asarray([_get_lane_id(snap) for snap in hist], dtype=float)
        if lane_ids.size == 0:
            continue
        t_lane = np.arange(lane_ids.size, dtype=float) * dt

        if v is base_env.vehicle:
            plt.step(t_lane, lane_ids, where="post", color="r", label="ego")
        else:
            plt.step(t_lane, lane_ids, where="post", color="b", alpha=0.6)

    plt.xlabel("Time [s]")
    plt.ylabel("Lane ID")
    plt.title(f"{title_prefix} Lane vs Time (ep {ep_idx})")
    plt.grid(True)
    if show_mode == "all":
        plt.legend()
    plt.tight_layout()
    plt.savefig(lane_path)
    plt.close()


def save_goal_snapshot(env, runner, ep_idx: int, step: int, model_dir: str, prev_goal_phys=None, intrinsic_reward=None, folder_name="goal_distribution"):
    """
    保存 HIRO Goal 可视化快照 (Vector Graphics version).
    使用 Matplotlib 直接绘制道路和车辆，获得清晰的矢量图/高分辨率位图，
    """
    import itertools
    import matplotlib.transforms as transforms
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    
    # 1. Directory Structure: separated by episode
    # Save directly under the run directory (e.g., eval_results/<datetime>/goal_distribution)
    # and clear output folder once at the beginning of a run
    base_debug_dir = os.path.join(model_dir, folder_name)
    if int(ep_idx) == 1 and int(step) == 0:
        if os.path.exists(base_debug_dir):
            shutil.rmtree(base_debug_dir)
    debug_dir = os.path.join(base_debug_dir, f"ep{ep_idx:03d}")
    os.makedirs(debug_dir, exist_ok=True)
    
    base_env = env.unwrapped
    road = base_env.road
    ego = base_env.vehicle
    
    if runner.goal_phys is None:
        return

    # 获取感知范围
    p_dist = getattr(base_env, "PERCEPTION_DISTANCE", 200.0)
    if p_dist is None:
        p_dist = 200.0
    p_dist = float(p_dist)

    # 视野窗口配置：仅显示 ego 前方/后方指定范围，可通过 config 开关关闭
    use_focus_window = bool(base_env.config.get("goal_snapshot_use_focus_window", True))
    front_dist = float(base_env.config.get("goal_snapshot_front_distance", 200.0))
    back_dist = float(base_env.config.get("goal_snapshot_back_distance", 50.0))

    forward = np.array([np.cos(float(ego.heading)), np.sin(float(ego.heading))], dtype=np.float32)

    def _in_focus_window(pos_xy: np.ndarray) -> bool:
        if not use_focus_window:
            return True
        rel = np.asarray(pos_xy, dtype=np.float32) - np.asarray(ego.position, dtype=np.float32)
        longi = float(np.dot(rel, forward))
        return (-back_dist <= longi <= front_dist)
    
    # 获取范围内车辆
    # close_vehicles_to 返回按距离排序的车辆列表 (不含 ego)
    neighbors = road.close_vehicles_to(ego, p_dist)
    neighbors = [v for v in neighbors if _in_focus_window(v.position)]
    
    # 确定哪些是 "Local Prob" (Observation 内的车辆)
    # runner.n_veh_local 是观察空间中包含的邻车数量
    n_local = getattr(runner, "n_veh_local", 0)
    local_neighbors_set = set(neighbors[:n_local])
    
    # 绘图列表：ego + neighbors
    # 注意：绘制顺序影响遮挡，这里不严格区分，因为大家都在车道上
    all_draw_vehs = [ego] + neighbors

    # 2. Setup Plot
    fig_w = float(base_env.config.get("goal_snapshot_fig_width", 15.0))
    fig_h = float(base_env.config.get("goal_snapshot_fig_height", 3.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    # 3. Draw Road Planes
    lanes = road.network.lanes_list()
    ys = []
    
    for lane in lanes:
        x0, y0 = lane.start
        x1, y1 = lane.end
        w = lane.width
        
        heading = lane.heading_at(0)
        c, s = np.cos(heading), np.sin(heading)
        normal = np.array([-s, c])
        
        p0 = lane.start - normal * w / 2
        p1 = lane.start + normal * w / 2
        p2 = lane.end + normal * w / 2
        p3 = lane.end - normal * w / 2
        
        poly = patches.Polygon([p0, p1, p2, p3], closed=True, facecolor='#666666', edgecolor='none', zorder=0)
        ax.add_patch(poly)
        
        types = [str(t) for t in lane.line_types]
        
        def draw_line(pa, pb, ltype):
            if 'NONE' in ltype: return
            style = 'solid' if 'CONTINUOUS' in ltype else 'dashed'
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color='white', linestyle=style, linewidth=1, zorder=1)
            
        draw_line(p0, p3, types[0])
        draw_line(p1, p2, types[1])
        
        ys.append(y0)
        ys.append(y1)
        
    # 4. Draw History (ego + observed neighbors)
    show_history = bool(base_env.config.get("goal_snapshot_show_history", True))
    hist_duration = float(base_env.config.get("goal_snapshot_history_duration", 2.0))
    hist_frequency = float(base_env.config.get("goal_snapshot_history_frequency", 3.0))
    sim_frequency = float(base_env.config.get("simulation_frequency", 15.0))
    hist_len = max(int(sim_frequency * max(hist_duration, 0.0)), 1)
    hist_stride = max(int(sim_frequency / max(hist_frequency, 1e-6)), 1)

    if show_history:
        trail_vehicles = [ego] + [v for v in neighbors if v in local_neighbors_set]
        for hv in trail_vehicles:
            hist = getattr(hv, "history", None)
            if not hist:
                continue
            sampled = list(itertools.islice(hist, 0, hist_len, hist_stride))
            if len(sampled) < 2:
                continue
            sampled = list(reversed(sampled))
            hist_xy = np.asarray([s.position for s in sampled], dtype=np.float32)
            if hist_xy.ndim != 2 or hist_xy.shape[1] < 2:
                continue
            if use_focus_window:
                mask = [bool(_in_focus_window(p)) for p in hist_xy]
                if not any(mask):
                    continue
                hist_xy = hist_xy[np.asarray(mask, dtype=bool)]
                if hist_xy.shape[0] < 2:
                    continue

            if hv is ego:
                trail_color = '#ff6b6b'
                trail_alpha = 0.75
                trail_lw = 2.0
                trail_z = 4
            else:
                trail_color = '#66c2ff'
                trail_alpha = 0.5
                trail_lw = 1.4
                trail_z = 3
            ax.plot(hist_xy[:, 0], hist_xy[:, 1], color=trail_color, linewidth=trail_lw, alpha=trail_alpha, zorder=trail_z)

    # 5. Draw Bus Stop
    for obj in getattr(road, "objects", []):
        obj_name = type(obj).__name__.lower()
        if "busstop" not in obj_name and "bus_stop" not in obj_name:
            continue
        if not _in_focus_window(getattr(obj, "position", np.zeros(2, dtype=np.float32))):
            continue
        l_obj = float(getattr(obj, "LENGTH", 10.0))
        w_obj = float(getattr(obj, "WIDTH", 3.0))
        rect_obj = patches.Rectangle(
            (-l_obj / 2.0, -w_obj / 2.0),
            l_obj,
            w_obj,
            facecolor='#d8c3a5',
            edgecolor='#8d6e63',
            linewidth=1.2,
            alpha=0.95,
            zorder=2,
        )
        t_obj = transforms.Affine2D().rotate(float(getattr(obj, "heading", 0.0))).translate(float(obj.position[0]), float(obj.position[1])) + ax.transData
        rect_obj.set_transform(t_obj)
        ax.add_patch(rect_obj)

    # 6. Draw Vehicles
    # 动态 Colorbar Range: 使用 HIRO High-Level Output 绝对速度范围 [0, speed_limit]
    # 索引获取: init_kinematics_meta 中 keep = ("x", "y", "vx", "vy")
    sx, sy, svx, svy = runner.ego_start[:4]
    
    # 获取速度上限 (默认 30 m/s，如果 config 中未定义)
    speed_limit = float(base_env.config.get("speed_limit", 30.0))
    norm = mcolors.Normalize(vmin=0.0, vmax=speed_limit)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'speed_red_to_green',
        ['#d73027', '#fdae61', '#1a9850'],
        N=256,
    )
    
    for v in all_draw_vehs:
        # Color Logic
        edge_c = 'black'
        lw = 1
        z = 5
        alpha_v = 0.9
        
        if v is ego:
            # Ego: Color by speed, Red Edge (thin)
            color = cmap(norm(v.speed))
            edge_c = 'red'
            lw = 1
            z = 6
        elif v in local_neighbors_set:
            # Observed Neighbor: Color by speed
            color = cmap(norm(v.speed))
        elif getattr(v, "crashed", False):
            color = 'black'
        else:
            # Unobserved: Gray
            color = '#E0E0E0'
        
        l, w = v.LENGTH, v.WIDTH
        rect = patches.Rectangle((-l/2, -w/2), l, w, facecolor=color, edgecolor=edge_c, linewidth=lw, alpha=alpha_v, zorder=z)
        
        t = transforms.Affine2D().rotate(v.heading).translate(v.position[0], v.position[1]) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

    # 7. Draw Goal
    # Unpack first 4 elements: x, y, vx, vy
    gx, gy, gvx, gvy = runner.goal_phys[:4]

    goal_marker_size = float(base_env.config.get("goal_snapshot_goal_marker_size", 24.0))
    show_prev_goal = bool(base_env.config.get("goal_snapshot_show_prev_goal", False))
    prev_goal_marker_size = float(base_env.config.get("goal_snapshot_prev_goal_marker_size", 18.0))

    # Draw Goal (Dot with color by Absolute Speed)
    goal_color = cmap(norm(gvx))
    ax.scatter([gx], [gy], c=[goal_color], marker='o', s=goal_marker_size, linewidth=1.2, edgecolors='white', zorder=10)
    
    # Draw Previous Goal (Transparent Dot)
    if show_prev_goal and prev_goal_phys is not None and len(prev_goal_phys) >= 4:
         px, py, pvx, pvy = prev_goal_phys[:4]
         if pvx != 0 or px != 0: # 简单过滤初始全0的情况
            p_color = cmap(norm(pvx))
            ax.scatter([px], [py], c=[p_color], marker='o', s=prev_goal_marker_size, linewidth=1.0, edgecolors='white', zorder=9, alpha=0.45)

    # 8. View Settings
    if use_focus_window:
        x_min = float(ego.position[0]) - back_dist
        x_max = float(ego.position[0]) + front_dist
    else:
        x_min = float(ego.position[0]) - p_dist
        x_max = float(ego.position[0]) + p_dist
    ax.set_xlim(x_min, x_max)
    
    if ys:
        mean_y = np.mean(ys)
        ax.set_ylim(mean_y - 12, mean_y + 12)
    else:
        ax.set_ylim(-10, 10)
        
    ax.invert_yaxis()  # HighwayEnv Y轴正方向向下，需翻转 Matplotlib 默认行为
    ax.set_aspect('equal')
    ax.axis('off')

    # Right-top annotation: current simulation time and ego longitudinal position
    pol_freq = float(base_env.config.get("policy_frequency", 10.0))
    t_now = float(getattr(base_env, "time", (float(step) / max(pol_freq, 1e-6))))
    x_ego = float(ego.position[0])
    ax.text(
        0.99,
        0.98,
        f"t={t_now:.1f}s, x_ego={x_ego:.1f}m",
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=20,
        color='black',
        bbox=dict(facecolor='white', alpha=0.55, edgecolor='none', boxstyle='round,pad=0.25'),
        zorder=20,
    )
    
    save_path = os.path.join(debug_dir, f"step{step:05d}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
