from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

from configs.builders import get_env_config
from rl.utils.utils import goal_action_to_abs


# =========================
# Config (edit here directly)
# =========================
CSV_PATH = Path(r"./high_interval_transitions.csv")
OUT_DIR = Path(r"./debug/high_interval_transitions_from_csv")
ROWS_PER_FIGURE = 18
X_MARGIN = 70.0
VEH_LENGTH = 5.0
VEH_WIDTH = 2.0
DPI = 140
MAX_EPISODES = 0  # 0 means render all


@dataclass
class SnapshotRow:
    global_step: int
    segment_id: int
    done_env: bool
    t_remaining: float
    signal_color_code: float
    signal_remaining: float
    ego_x_abs: float
    ego_y_abs: float
    ego_vx: float
    kin: np.ndarray
    goal_phys: np.ndarray


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


def _signal_text_and_color(signal_code: float) -> tuple[str, str]:
    if signal_code < -0.5:
        return "N/A", "#7f8c8d"
    if signal_code >= 0.5:
        return "GREEN", "#2ecc71"
    return "RED/YELLOW", "#e74c3c"


def _decode_high_obs(
    high_obs: np.ndarray,
    goal_longitudinal: float,
) -> tuple[float, np.ndarray, float, float, float, float, float]:
    if high_obs.ndim != 1 or high_obs.size < 8:
        raise ValueError("Invalid high_obs shape")

    t_remaining = float(high_obs[0])
    signal_color = float(high_obs[-2])
    signal_remaining = float(high_obs[-1])
    kin_flat = np.asarray(high_obs[1:-2], dtype=np.float32)
    if kin_flat.size % 5 != 0:
        raise ValueError(f"high_obs kinematics size is not multiple of 5: {kin_flat.size}")
    kin = kin_flat.reshape(-1, 5)

    # high_obs has ego x overwritten by goal_remaining_x, recover absolute x.
    ego_goal_remaining_x = float(kin[0, 1])
    ego_x_abs = float(goal_longitudinal - ego_goal_remaining_x)
    ego_y_abs = float(kin[0, 2])
    ego_vx = float(kin[0, 3])
    ego_vy = float(kin[0, 4])
    return t_remaining, kin, signal_color, signal_remaining, ego_x_abs, ego_y_abs, ego_vx if np.isfinite(ego_vx) else 0.0


def _build_snapshot_row(
    row: dict[str, str],
    lane_centers: np.ndarray,
    goal_longitudinal: float,
) -> SnapshotRow:
    high_obs = _parse_json_array(row.get("high_obs", "")).reshape(-1)
    high_action = _parse_json_array(row.get("high_action", "")).reshape(-1)
    if high_action.size < 3:
        raise ValueError("high_action must contain at least 3 values: [dx, y_code, vx_target]")

    t_remaining, kin, signal_color, signal_remaining, ego_x_abs, ego_y_abs, ego_vx = _decode_high_obs(
        high_obs=high_obs,
        goal_longitudinal=goal_longitudinal,
    )

    ego_sub = np.asarray([[ego_x_abs, ego_y_abs, ego_vx, 0.0]], dtype=np.float32)
    goal_action = np.asarray(high_action[:3], dtype=np.float32).reshape(1, 3)
    goal_phys = goal_action_to_abs(ego_sub, goal_action, lane_centers).reshape(-1)

    return SnapshotRow(
        global_step=int(float(row.get("global_step", 0))),
        segment_id=int(float(row.get("segment_id", 0))),
        done_env=bool(int(float(row.get("done_env", 0)))),
        t_remaining=t_remaining,
        signal_color_code=signal_color,
        signal_remaining=signal_remaining,
        ego_x_abs=ego_x_abs,
        ego_y_abs=ego_y_abs,
        ego_vx=ego_vx,
        kin=kin,
        goal_phys=goal_phys,
    )


def _draw_snapshot_axis(
    ax: Any,
    snap: SnapshotRow,
    lane_centers: np.ndarray,
    lane_width: float,
    speed_limit: float,
    stop_lines_x: list[float],
    x_margin: float,
    veh_length: float,
    veh_width: float,
    episode_id: int,
    idx_in_episode: int,
    idx_global: int,
) -> None:
    x0 = float(snap.ego_x_abs - x_margin)
    x1 = float(snap.ego_x_abs + x_margin)

    y_min = float(np.min(lane_centers) - lane_width / 2.0)
    y_max = float(np.max(lane_centers) + lane_width / 2.0)

    # Match high-interval-debug style.
    ax.set_facecolor("#d9d9d9")

    # Road background.
    for y in lane_centers:
        rect = patches.Rectangle(
            (x0, float(y - lane_width / 2.0)),
            float(x1 - x0),
            float(lane_width),
            facecolor="#666666",
            edgecolor="none",
            zorder=0,
        )
        ax.add_patch(rect)
        ax.plot([x0, x1], [float(y), float(y)], color="white", linestyle="--", linewidth=1.0, zorder=1)
    ax.plot([x0, x1], [y_min, y_min], color="white", linewidth=1.0, zorder=1)
    ax.plot([x0, x1], [y_max, y_max], color="white", linewidth=1.0, zorder=1)

    # Draw stop lines (left at goal x, right at intersection end when available).
    for sx in stop_lines_x:
        if not np.isfinite(sx):
            continue
        if sx < x0 - 2.0 or sx > x1 + 2.0:
            continue
        ax.plot([sx, sx], [y_min, y_max], color="white", linewidth=2.0, linestyle="-", alpha=0.95, zorder=3)

    norm = mcolors.Normalize(vmin=0.0, vmax=max(float(speed_limit), 1e-3))
    cmap = cm.get_cmap("jet")

    # Vehicles: kin rows [presence, x, y, vx, vy].
    kin = snap.kin
    if kin.ndim == 2 and kin.shape[1] >= 5:
        ego_y = float(snap.ego_y_abs)
        for i in range(kin.shape[0]):
            presence = float(kin[i, 0])
            if presence <= 0.5:
                continue

            if i == 0:
                vx_abs = float(snap.ego_vx)
                x_abs = float(snap.ego_x_abs)
                y_abs = float(ego_y)
                face = cmap(norm(max(0.0, vx_abs)))
                edge = "red"
                lw = 1.6
                z = 5
            else:
                x_abs = float(snap.ego_x_abs + kin[i, 1])
                y_abs = float(ego_y + kin[i, 2])
                vx_abs = float(snap.ego_vx + kin[i, 3])
                face = cmap(norm(max(0.0, vx_abs)))
                edge = "black"
                lw = 1.0
                z = 4

            rect = patches.Rectangle(
                (x_abs - veh_length / 2.0, y_abs - veh_width / 2.0),
                veh_length,
                veh_width,
                facecolor=face,
                edgecolor=edge,
                linewidth=lw,
                alpha=0.95,
                zorder=z,
            )
            ax.add_patch(rect)

            if i == 0:
                ax.text(
                    x_abs + 2.5,
                    y_abs - 0.9,
                    f"ego x={x_abs:.1f}, vx={vx_abs:.2f}",
                    fontsize=8,
                    color="white",
                    ha="left",
                    va="center",
                    zorder=7,
                    bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1.4),
                )

    # Goal marker.
    gx, gy = float(snap.goal_phys[0]), float(snap.goal_phys[1])
    ax.scatter([gx], [gy], s=70, c="#f39c12", edgecolors="white", linewidths=1.2, zorder=7)
    ax.text(
        gx + 1.8,
        gy - 0.4,
        f"goal ({gx:.1f}, {gy:.1f})",
        fontsize=8,
        color="white",
        ha="left",
        va="center",
        zorder=8,
        bbox=dict(facecolor="black", alpha=0.4, edgecolor="none", pad=1.2),
    )

    signal_text, signal_color = _signal_text_and_color(snap.signal_color_code)
    sig_msg = f"signal={signal_text}, remain={snap.signal_remaining:.1f}s"
    ax.text(
        0.99,
        0.95,
        sig_msg,
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        ha="right",
        va="top",
        zorder=9,
        bbox=dict(facecolor=signal_color, alpha=0.85, edgecolor="none", pad=2.0),
    )

    ax.set_title(
        (
            f"Episode {episode_id:03d} | in-ep {idx_in_episode:03d} | global_idx {idx_global:04d} | "
            f"step {snap.global_step} | seg {snap.segment_id} | t_rem={snap.t_remaining:.1f}s"
        ),
        fontsize=10,
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y_min - 1.2, y_max + 1.2)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")


def _split_episodes(rows: list[SnapshotRow]) -> list[list[SnapshotRow]]:
    episodes: list[list[SnapshotRow]] = []
    cur: list[SnapshotRow] = []
    for r in rows:
        cur.append(r)
        if r.done_env:
            episodes.append(cur)
            cur = []
    if cur:
        episodes.append(cur)
    return episodes


def _load_rows(csv_path: Path, lane_centers: np.ndarray, goal_longitudinal: float) -> list[SnapshotRow]:
    out: list[SnapshotRow] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out.append(_build_snapshot_row(row, lane_centers=lane_centers, goal_longitudinal=goal_longitudinal))
            except Exception as exc:
                raise RuntimeError(f"Failed to parse row with global_step={row.get('global_step', 'NA')}: {exc}") from exc
    return out


def main() -> None:
    csv_path = CSV_PATH.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    env_cfg = get_env_config()
    n_lanes = int(env_cfg.get("lanes_count", 3))
    lane_width = float(env_cfg.get("lane_width", 4.0))
    speed_limit = float(env_cfg.get("speed_limit", 15.0))
    goal_longitudinal = float(env_cfg.get("goal_longitudinal", 400.0))
    intersection_length = float(env_cfg.get("intersection_length", 50.0))
    road_length = float(env_cfg.get("road_length", goal_longitudinal + intersection_length))
    left_stop_x = float(goal_longitudinal)
    right_stop_x = float(min(max(goal_longitudinal + intersection_length, left_stop_x), road_length))
    stop_lines_x = [left_stop_x]
    if right_stop_x > left_stop_x + 1e-3:
        stop_lines_x.append(right_stop_x)

    lane_centers = (np.arange(n_lanes, dtype=np.float32) * lane_width).astype(np.float32)

    rows = _load_rows(csv_path=csv_path, lane_centers=lane_centers, goal_longitudinal=goal_longitudinal)
    if not rows:
        raise RuntimeError(f"CSV has no data rows: {csv_path}")

    episodes = _split_episodes(rows)
    if int(MAX_EPISODES) > 0:
        episodes = episodes[: int(MAX_EPISODES)]

    out_dir = OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_per_figure = int(max(1, ROWS_PER_FIGURE))
    total_saved = 0
    global_idx = 0

    for ep_i, ep_rows in enumerate(episodes, start=1):
        ep_dir = out_dir / f"ep{ep_i:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        start = 0
        page = 1
        while start < len(ep_rows):
            chunk = ep_rows[start : start + rows_per_figure]
            n = len(chunk)
            fig_h = max(2.6 * n, 3.0)
            fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(15, fig_h), squeeze=False)
            fig.patch.set_facecolor("#d9d9d9")
            axes_flat = axes.reshape(-1)

            for i, (ax, snap) in enumerate(zip(axes_flat, chunk, strict=True)):
                _draw_snapshot_axis(
                    ax=ax,
                    snap=snap,
                    lane_centers=lane_centers,
                    lane_width=lane_width,
                    speed_limit=speed_limit,
                    stop_lines_x=stop_lines_x,
                    x_margin=float(X_MARGIN),
                    veh_length=float(VEH_LENGTH),
                    veh_width=float(VEH_WIDTH),
                    episode_id=ep_i,
                    idx_in_episode=start + i,
                    idx_global=global_idx,
                )
                global_idx += 1

            fig.suptitle(
                f"high_interval_transitions.csv | Episode {ep_i:03d} | page {page:02d}",
                fontsize=12,
                y=0.995,
            )
            # Reserve right margin for an external colorbar to avoid covering snapshots.
            fig.tight_layout(rect=(0.0, 0.0, 0.91, 0.985))

            # One colorbar per page, placed in a dedicated right-side axis.
            sm = cm.ScalarMappable(cmap=cm.get_cmap("jet"), norm=mcolors.Normalize(vmin=0.0, vmax=max(speed_limit, 1e-3)))
            sm.set_array([])
            cax = fig.add_axes([0.925, 0.14, 0.015, 0.72])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label("Speed [m/s]", fontsize=10)

            out_path = ep_dir / f"ep{ep_i:03d}_page{page:02d}.png"
            fig.savefig(out_path, dpi=int(DPI), bbox_inches="tight", pad_inches=0.08)
            plt.close(fig)

            total_saved += 1
            page += 1
            start += rows_per_figure

    print(f"CSV             : {csv_path}")
    print(f"Episodes rendered: {len(episodes)}")
    print(f"Pages saved     : {total_saved}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
