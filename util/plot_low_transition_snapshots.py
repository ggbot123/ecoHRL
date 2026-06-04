from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


DEFAULT_RUN_DIR = Path("./logs/current/hiro_260601_highonly_rule_debug1M_video")


@dataclass
class Snapshot:
    episode: int
    index: int
    global_step: int
    segment_id: int
    c: int
    done_env: bool
    done_low: bool
    time_s: float
    remaining_time_s: float
    remaining_distance_m: float
    signal_is_green: float
    signal_remaining_s: float
    reward_env: float
    low_reward_total: float
    high_ret_running: float
    kin: np.ndarray


def _json_array(raw: str) -> np.ndarray:
    text = str(raw or "").strip()
    if not text or text == "[]":
        return np.asarray([], dtype=np.float32)
    try:
        return np.asarray(json.loads(text), dtype=np.float32)
    except Exception:
        if text.startswith("[") and text.endswith("]"):
            return np.fromstring(text[1:-1], sep=",", dtype=np.float32)
        return np.asarray([], dtype=np.float32)


def _json_obj(raw: str) -> Any:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _float(row: dict[str, str], key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def _int(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except Exception:
        return default


def _parse_episode_set(raw: str) -> set[int]:
    out: set[int] = set()
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _env0_config(run_config: dict[str, Any]) -> dict[str, Any]:
    env = run_config.get("environment", {}) if isinstance(run_config, dict) else {}
    cfg = env.get("env0_config", {}) if isinstance(env, dict) else {}
    return cfg if isinstance(cfg, dict) else {}


def _iter_snapshots(csv_path: Path, episodes: set[int]) -> Any:
    episode = 1
    index = 0
    max_episode = max(episodes) if episodes else 0

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            obs = _json_array(row.get("next_obs_tr", ""))
            if obs.size < 26:
                if _int(row, "done_env"):
                    episode += 1
                    index = 0
                continue

            if not episodes or episode in episodes:
                kin = obs[1:].reshape(-1, 5)
                next_high_obs = _json_array(row.get("next_high_obs", ""))
                if next_high_obs.size >= 3:
                    remaining_time = float(next_high_obs[0])
                    remaining_dist = float(next_high_obs[2])
                else:
                    remaining_time = np.nan
                    remaining_dist = np.nan

                yield Snapshot(
                    episode=episode,
                    index=index,
                    global_step=_int(row, "global_step"),
                    segment_id=_int(row, "segment_id", -1),
                    c=_int(row, "c", -1),
                    done_env=bool(_int(row, "done_env")),
                    done_low=bool(_int(row, "done_low")),
                    time_s=_float(row, "diag_time"),
                    remaining_time_s=remaining_time,
                    remaining_distance_m=remaining_dist,
                    signal_is_green=_float(row, "diag_signal_is_green"),
                    signal_remaining_s=_float(row, "diag_signal_remaining"),
                    reward_env=_float(row, "reward_env"),
                    low_reward_total=_float(row, "low_reward_total"),
                    high_ret_running=_float(row, "high_ret_running"),
                    kin=kin,
                )

            if _int(row, "done_env"):
                episode += 1
                index = 0
                if episodes and max_episode > 0 and episode > max_episode:
                    break
            else:
                index += 1


def _signal_label(signal_is_green: float) -> tuple[str, str]:
    if signal_is_green >= 0.5:
        return "GREEN", "#2ecc71"
    return "RED/YELLOW", "#e74c3c"


def _draw_snapshot(
    snap: Snapshot,
    out_path: Path,
    *,
    lanes_count: int,
    lane_width: float,
    goal_lane_id: int,
    goal_longitudinal: float,
    speed_limit: float,
    x_margin: float,
    vehicle_length: float,
    vehicle_width: float,
    dpi: int,
) -> None:
    lane_centers = np.arange(lanes_count, dtype=float) * lane_width
    y_min = float(lane_centers[0] - lane_width / 2.0)
    y_max = float(lane_centers[-1] + lane_width / 2.0)

    ego = snap.kin[0]
    ego_x = float(ego[1])
    ego_y = float(ego[2])
    ego_vx = float(ego[3])
    ego_vy = float(ego[4])

    x0 = max(0.0, ego_x - x_margin)
    x1 = max(x0 + 20.0, ego_x + x_margin)
    if x0 <= goal_longitudinal <= x1:
        x0 = min(x0, goal_longitudinal - x_margin * 0.35)
        x1 = max(x1, goal_longitudinal + x_margin * 0.35)

    fig, ax = plt.subplots(figsize=(13.5, 4.2), dpi=dpi)
    ax.set_facecolor("#d6d6d6")

    for lane_id, y in enumerate(lane_centers):
        face = "#707070" if lane_id != goal_lane_id else "#676f77"
        rect = patches.Rectangle(
            (x0, y - lane_width / 2.0),
            x1 - x0,
            lane_width,
            facecolor=face,
            edgecolor="none",
            zorder=0,
        )
        ax.add_patch(rect)
        ax.plot([x0, x1], [y + lane_width / 2.0, y + lane_width / 2.0], color="white", linewidth=1.0, zorder=1)
        if lane_id < lanes_count - 1:
            ax.plot([x0, x1], [y + lane_width / 2.0, y + lane_width / 2.0], color="white", linestyle="--", linewidth=1.0, zorder=2)

    if x0 <= goal_longitudinal <= x1:
        ax.plot([goal_longitudinal, goal_longitudinal], [y_min, y_max], color="white", linewidth=2.2, zorder=3)
        ax.text(goal_longitudinal + 1.0, y_max - 0.3, "stop/goal x", color="white", fontsize=8, va="top")

    norm = mcolors.Normalize(vmin=0.0, vmax=max(speed_limit, 1e-3))
    cmap = plt.get_cmap("viridis")

    speeds: list[float] = []
    for i in range(snap.kin.shape[0]):
        row = snap.kin[i]
        if float(row[0]) <= 0.5:
            continue
        if i == 0:
            x = ego_x
            y = ego_y
            vx = ego_vx
            vy = ego_vy
            edge = "#ff2d2d"
            label = "ego"
            z = 6
            lw = 2.0
        else:
            x = ego_x + float(row[1])
            y = ego_y + float(row[2])
            vx = ego_vx + float(row[3])
            vy = ego_vy + float(row[4])
            edge = "#111111"
            label = f"v{i}"
            z = 5
            lw = 1.0

        speed = float(np.hypot(vx, vy))
        speeds.append(speed)
        color = cmap(norm(max(0.0, speed)))
        rect = patches.Rectangle(
            (x - vehicle_length / 2.0, y - vehicle_width / 2.0),
            vehicle_length,
            vehicle_width,
            facecolor=color,
            edgecolor=edge,
            linewidth=lw,
            zorder=z,
        )
        ax.add_patch(rect)
        ax.text(x, y, label, fontsize=7, color="white", ha="center", va="center", zorder=z + 1)
        ax.text(x + 2.8, y - 0.95, f"{speed:.1f}m/s", fontsize=7, color="white", ha="left", va="center", zorder=z + 1)

    signal_text, signal_color = _signal_label(snap.signal_is_green)
    info_lines = [
        f"episode={snap.episode} frame={snap.index} global_step={snap.global_step}",
        f"t={snap.time_s:.1f}s  t_rem={snap.remaining_time_s:.1f}s  x_rem={snap.remaining_distance_m:.1f}m",
        f"signal={signal_text} rem={snap.signal_remaining_s:.1f}s  seg={snap.segment_id} c={snap.c}",
        f"reward_env={snap.reward_env:.3f} low_total={snap.low_reward_total:.3f} high_ret={snap.high_ret_running:.3f}",
        f"done_env={int(snap.done_env)} done_low={int(snap.done_low)} goal_lane={goal_lane_id}",
    ]
    ax.text(
        0.012,
        0.97,
        "\n".join(info_lines),
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#1f2933", edgecolor=signal_color, linewidth=1.4, alpha=0.88),
        zorder=10,
    )

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.015)
    cbar.set_label("speed [m/s]")

    ax.set_xlim(x0, x1)
    ax.set_ylim(y_min - 0.8, y_max + 0.8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Low-step transition snapshot reconstructed from next_obs_tr")
    ax.grid(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render episode snapshots from low_step_transition_details.csv")
    parser.add_argument("--run-dir", type=str, default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--episodes", type=str, default="578,579,580", help="Comma/range list, e.g. 10,12-15")
    parser.add_argument("--stride", type=int, default=25, help="Render every Nth timestep")
    parser.add_argument("--max-frames-per-episode", type=int, default=16)
    parser.add_argument("--x-margin", type=float, default=80.0)
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    csv_path = run_dir / "low_step_transition_details.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    run_config = _load_run_config(run_dir)
    cfg = _env0_config(run_config)
    lanes_count = int(cfg.get("lanes_count", 3))
    lane_width = float(cfg.get("lane_width", 4.0))
    goal_lane_id = int(cfg.get("goal_lane_id", 1))
    goal_longitudinal = float(cfg.get("goal_longitudinal", 400.0))
    speed_limit = float(cfg.get("speed_limit", 15.0))

    episodes = _parse_episode_set(args.episodes)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "transition_snapshots"
    stride = max(1, int(args.stride))
    max_frames = max(1, int(args.max_frames_per_episode))

    saved_by_episode: dict[int, int] = {}
    for snap in _iter_snapshots(csv_path, episodes):
        count = saved_by_episode.get(snap.episode, 0)
        should_save = (snap.index % stride == 0) or snap.done_env or snap.done_low
        if count >= max_frames and not snap.done_env:
            should_save = False
        if not should_save:
            continue

        saved_by_episode[snap.episode] = count + 1
        out_path = out_dir / f"episode_{snap.episode:04d}" / f"ep{snap.episode:04d}_frame{snap.index:04d}_step{snap.global_step}.png"
        _draw_snapshot(
            snap,
            out_path,
            lanes_count=lanes_count,
            lane_width=lane_width,
            goal_lane_id=goal_lane_id,
            goal_longitudinal=goal_longitudinal,
            speed_limit=speed_limit,
            x_margin=float(args.x_margin),
            vehicle_length=5.0,
            vehicle_width=2.0,
            dpi=int(args.dpi),
        )

    print(f"CSV used: {csv_path}")
    print(f"Saved snapshots to: {out_dir}")
    print(f"Saved counts: {dict(sorted(saved_by_episode.items()))}")


if __name__ == "__main__":
    main()
