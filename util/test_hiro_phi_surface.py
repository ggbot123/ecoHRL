from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from configs.builders import get_env_config, get_hiro_config
from rl.utils import utils


TAU_HEADING = 0.2
TAU_LATERAL = 0.6
TAU_PURSUIT = 0.5 * TAU_HEADING
KP_HEADING = 1.0 / TAU_HEADING
KP_LATERAL = 1.0 / TAU_LATERAL
MAX_STEERING_ANGLE = np.pi / 3
VEHICLE_LENGTH = 5.0


def _not_zero(v: float, eps: float = 1e-2) -> float:
    return float(v) if abs(float(v)) > eps else (eps if float(v) >= 0 else -eps)


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _scalar_to_lane_change_cmd(lane_scalar: float) -> int:
    x = float(np.clip(lane_scalar, -1.0, 1.0))
    if x < -1.0 / 3.0:
        return 1  # LANE_LEFT
    if x > 1.0 / 3.0:
        return 2  # LANE_RIGHT
    return 0  # KEEP


def _apply_lane_change_command(cur_lane_id: int, lane_change_cmd: int, max_lane_id: int) -> int:
    if lane_change_cmd == 0:
        desired_id = int(cur_lane_id)
    elif lane_change_cmd == 1:
        desired_id = int(np.clip(cur_lane_id - 1, 0, max_lane_id))
    elif lane_change_cmd == 2:
        desired_id = int(np.clip(cur_lane_id + 1, 0, max_lane_id))
    else:
        desired_id = int(cur_lane_id)
    return desired_id


def _steering_control_straight_lane(
    x: float,
    y: float,
    heading: float,
    speed: float,
    target_lane_center_y: float,
) -> float:
    del x
    lane_lateral = float(y - target_lane_center_y)
    lane_next_longitudinal = float(speed) * TAU_PURSUIT
    del lane_next_longitudinal
    lane_future_heading = 0.0

    lateral_speed_command = -KP_LATERAL * lane_lateral
    heading_command = np.arcsin(np.clip(lateral_speed_command / _not_zero(speed), -1.0, 1.0))
    heading_ref = lane_future_heading + np.clip(heading_command, -np.pi / 4.0, np.pi / 4.0)

    heading_rate_command = KP_HEADING * _wrap_to_pi(float(heading_ref - heading))
    slip_angle = np.arcsin(
        np.clip(
            (VEHICLE_LENGTH / 2.0) / _not_zero(speed) * heading_rate_command,
            -1.0,
            1.0,
        )
    )
    steering_angle = np.arctan(2.0 * np.tan(slip_angle))
    steering_angle = np.clip(steering_angle, -MAX_STEERING_ANGLE, MAX_STEERING_ANGLE)
    return float(steering_angle)


@dataclass
class SceneState:
    """Initial scene state.

    Notes
    -----
    - `ego` uses [x, y, vx, vy].
    - `others` is accepted for completeness of state s, but current Φ depends on ego-goal error only.
    """

    ego: np.ndarray
    others: np.ndarray


def _closest_lane_y(y_now: float, lane_center_ys: np.ndarray, lane_scalar: float) -> float:
    y_arr = np.asarray(lane_center_ys, dtype=np.float32).reshape(-1)
    idx = int(np.argmin(np.abs(y_arr - float(y_now))))
    if lane_scalar < -1.0 / 3.0 and idx > 0:
        idx -= 1
    elif lane_scalar > 1.0 / 3.0 and idx < len(y_arr) - 1:
        idx += 1
    return float(y_arr[idx])


def _predict_ego_next(
    ego_now: np.ndarray,
    action_lane_scalar: float,
    action_accel_scalar: float,
    dt: float,
    lane_center_ys: np.ndarray,
    accel_range: Sequence[float],
    simulation_frequency: float,
    policy_frequency: float,
    speed_limit: float,
) -> np.ndarray:
    """One policy-step ego rollout aligned with scenario.py + AbstractEnv._simulate logic."""
    x, y, vx, vy = [float(v) for v in ego_now]

    heading = float(np.arctan2(vy, _not_zero(vx)))
    speed = float(np.hypot(vx, vy))

    a_min, a_max = float(accel_range[0]), float(accel_range[1])
    acc = float(np.interp(float(np.clip(action_accel_scalar, -1.0, 1.0)), [-1.0, 1.0], [a_min, a_max]))

    max_lane_id = int(len(lane_center_ys) - 1)
    cur_lane_id = int(np.argmin(np.abs(np.asarray(lane_center_ys, dtype=np.float32) - y)))
    lane_change_cmd = _scalar_to_lane_change_cmd(action_lane_scalar)
    target_lane_id = _apply_lane_change_command(cur_lane_id, lane_change_cmd, max_lane_id)
    target_lane_center_y = float(lane_center_ys[target_lane_id])

    frames = int(float(simulation_frequency) // float(policy_frequency))
    sim_dt = 1.0 / float(simulation_frequency)

    for _ in range(frames):
        steering = _steering_control_straight_lane(
            x=x,
            y=y,
            heading=heading,
            speed=speed,
            target_lane_center_y=target_lane_center_y,
        )

        beta = np.arctan(0.5 * np.tan(steering))
        x += speed * np.cos(heading + beta) * sim_dt
        y += speed * np.sin(heading + beta) * sim_dt
        heading += speed * np.sin(beta) / (VEHICLE_LENGTH / 2.0) * sim_dt
        speed = float(np.clip(speed + acc * sim_dt, 0.0, float(speed_limit)))

    vx_next = float(speed * np.cos(heading))
    vy_next = float(speed * np.sin(heading))

    del dt
    return np.asarray([x, y, vx_next, vy_next], dtype=np.float32)


def plot_phi_surface(
    state: SceneState,
    goal_phys: np.ndarray,
    lane_res: int = 121,
    accel_res: int = 121,
    show_intrinsic_surface: bool = False,
    csv_out: str = "phi_surface_points.csv",
) -> None:
    hiro_cfg = get_hiro_config()
    env_cfg = get_env_config()

    if str(getattr(hiro_cfg, "intrinsic_type", "")).lower() != "huber_shaping":
        raise ValueError("当前 conf 未启�?huber_shaping，请先设�?ENABLE_HIRO_REWARD_SHAPING=True�?)

    dt = 1.0 / float(env_cfg["policy_frequency"])
    simulation_frequency = float(env_cfg["simulation_frequency"])
    policy_frequency = float(env_cfg["policy_frequency"])
    speed_limit = float(env_cfg["speed_limit"])
    lanes = int(env_cfg["lanes_count"])
    lane_w = float(env_cfg.get("lane_width", 4.0))
    lane_center_ys = (np.arange(lanes, dtype=np.float32) * lane_w).astype(np.float32)
    accel_range = tuple(env_cfg["action"]["acceleration_range"])

    norm_ranges = np.asarray(hiro_cfg.intrinsic_norm_ranges, dtype=np.float32)
    weights = np.asarray(hiro_cfg.intrinsic_weights, dtype=np.float32)

    ego_now = np.asarray(state.ego, dtype=np.float32).reshape(1, -1)
    goal = np.asarray(goal_phys, dtype=np.float32).reshape(1, -1)

    lane_axis = np.linspace(-1.0, 1.0, int(lane_res), dtype=np.float32)
    accel_axis = np.linspace(-1.0, 1.0, int(accel_res), dtype=np.float32)
    lane_grid, accel_grid = np.meshgrid(lane_axis, accel_axis, indexing="xy")

    phi_grid = np.zeros_like(lane_grid, dtype=np.float32)
    intrinsic_grid = np.zeros_like(lane_grid, dtype=np.float32)
    phi_now = float(utils.huber_potential(ego_now - goal, norm_ranges=norm_ranges, weights=weights)[0])

    for i in range(lane_grid.shape[0]):
        for j in range(lane_grid.shape[1]):
            ego_next = _predict_ego_next(
                ego_now=ego_now[0],
                action_lane_scalar=float(lane_grid[i, j]),
                action_accel_scalar=float(accel_grid[i, j]),
                dt=dt,
                lane_center_ys=lane_center_ys,
                accel_range=accel_range,
                simulation_frequency=simulation_frequency,
                policy_frequency=policy_frequency,
                speed_limit=speed_limit,
            ).reshape(1, -1)

            err_next = ego_next - goal
            # if lane_grid[i, j] < 0.33 and lane_grid[i, j] > -0.33:
            if lane_grid[i, j] < -0.99:
                if accel_grid[i, j] < -0.99:
                    pass
                elif accel_grid[i, j] > -0.5:
                    pass
            phi_next = float(utils.huber_potential(err_next, norm_ranges=norm_ranges, weights=weights)[0])
            phi_grid[i, j] = phi_next

            intrinsic, _, _, _ = utils.intrinsic_reward_shaping_huber(
                ego_rel_now=ego_now,
                ego_rel_next=ego_next,
                goal_rel=goal,
                norm_ranges=norm_ranges,
                coef=float(hiro_cfg.intrinsic_coef),
                weights=weights,
                gamma=0.99,
                is_terminal=np.asarray([False]),
            )
            intrinsic_grid[i, j] = float(intrinsic[0])

    n_cols = 3 if show_intrinsic_surface else 2
    fig = plt.figure(figsize=(6 * n_cols, 6))
    ax1 = fig.add_subplot(1, n_cols, 1, projection="3d")
    surf1 = ax1.plot_surface(lane_grid, accel_grid, phi_grid, cmap="viridis", linewidth=0, antialiased=True)
    phi_max = float(np.max(phi_grid))
    max_mask = np.isclose(phi_grid, phi_max, rtol=1e-6, atol=1e-8)
    max_i, max_j = np.where(max_mask)

    out_path = Path(csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = np.column_stack(
        [
            lane_grid.ravel(),
            accel_grid.ravel(),
            phi_grid.ravel(),
            intrinsic_grid.ravel(),
            max_mask.astype(np.int32).ravel(),
        ]
    )
    np.savetxt(
        out_path,
        rows,
        delimiter=",",
        header="lane_scalar,accel_scalar,phi,intrinsic,is_global_max",
        comments="",
    )
    print(f"[CSV] Saved {rows.shape[0]} points to {out_path}")
    ax1.scatter(
        lane_grid[max_i, max_j],
        accel_grid[max_i, max_j],
        phi_grid[max_i, max_j],
        c="red",
        s=28,
        depthshade=False,
        label="global max",
    )
    ax1.set_xlabel("lane_scalar a[0]")
    ax1.set_ylabel("accel_scalar a[1]")
    ax1.set_zlabel("Phi(s,a) = Phi(s_next(a))")
    ax1.set_title("Huber Potential Surface")
    fig.colorbar(surf1, ax=ax1, shrink=0.7, pad=0.1)
    ax1.legend(loc="upper right")

    axc = fig.add_subplot(1, n_cols, 2)
    contour = axc.contourf(lane_grid, accel_grid, phi_grid, levels=30, cmap="viridis")
    axc.contour(lane_grid, accel_grid, phi_grid, levels=10, colors="k", linewidths=0.3, alpha=0.5)
    axc.scatter(
        lane_grid[max_i, max_j],
        accel_grid[max_i, max_j],
        c="red",
        s=24,
        label="global max",
    )
    axc.set_xlabel("lane_scalar a[0]")
    axc.set_ylabel("accel_scalar a[1]")
    axc.set_title("Phi(s,a) Contour")
    fig.colorbar(contour, ax=axc, shrink=0.9, pad=0.02, label="Phi(s,a)")
    axc.legend(loc="upper right")

    if show_intrinsic_surface:
        ax2 = fig.add_subplot(1, n_cols, 3, projection="3d")
        surf2 = ax2.plot_surface(lane_grid, accel_grid, intrinsic_grid, cmap="plasma", linewidth=0, antialiased=True)
        ax2.set_xlabel("lane_scalar a[0]")
        ax2.set_ylabel("accel_scalar a[1]")
        ax2.set_zlabel("intrinsic shaping reward")
        ax2.set_title("r_shape(s,a)")
        fig.colorbar(surf2, ax=ax2, shrink=0.7, pad=0.1)

    others_note = "none" if state.others.size == 0 else str(state.others.shape)
    fig.suptitle(
        f"state ego={state.ego.tolist()}, others={others_note}, goal={goal_phys.tolist()}\n"
        f"Phi(s)={phi_now:.4f}, dt={dt}, lanes={lanes}, accel_range={accel_range}"
    )
    plt.tight_layout()
    plt.show()


def plot_phi_xy_surface(
    state: SceneState,
    goal_phys: np.ndarray,
    xy_res: int = 121,
    x_min: float | None = None,
    x_max: float | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    """Plot Phi over (x, y) grid with ego (vx, vy) fixed to state's values."""
    hiro_cfg = get_hiro_config()

    if str(getattr(hiro_cfg, "intrinsic_type", "")).lower() != "huber_shaping":
        raise ValueError("当前 conf 未启�?huber_shaping，请先设�?ENABLE_HIRO_REWARD_SHAPING=True�?)

    norm_ranges = np.asarray(hiro_cfg.intrinsic_norm_ranges, dtype=np.float32)
    weights = np.asarray(hiro_cfg.intrinsic_weights, dtype=np.float32)

    ego_ref = np.asarray(state.ego, dtype=np.float32).reshape(-1)
    goal = np.asarray(goal_phys, dtype=np.float32).reshape(-1)

    if x_min is None:
        x_min = float(goal[0] + norm_ranges[0, 0])
    if x_max is None:
        x_max = float(goal[0] + norm_ranges[0, 1])
    if y_min is None:
        y_min = float(goal[1] + norm_ranges[1, 0])
    if y_max is None:
        y_max = float(goal[1] + norm_ranges[1, 1])

    x_axis = np.linspace(float(x_min), float(x_max), int(xy_res), dtype=np.float32)
    y_axis = np.linspace(float(y_min), float(y_max), int(xy_res), dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")

    phi_xy = np.zeros_like(x_grid, dtype=np.float32)
    fixed_vx, fixed_vy = float(ego_ref[2]), float(ego_ref[3])

    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            ego_candidate = np.asarray([x_grid[i, j], y_grid[i, j], fixed_vx, fixed_vy], dtype=np.float32).reshape(1, -1)
            err = ego_candidate - goal.reshape(1, -1)
            phi_xy[i, j] = float(utils.huber_potential(err, norm_ranges=norm_ranges, weights=weights)[0])

    phi_max = float(np.max(phi_xy))
    max_mask = np.isclose(phi_xy, phi_max, rtol=1e-6, atol=1e-8)
    max_i, max_j = np.where(max_mask)

    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax3d.plot_surface(x_grid, y_grid, phi_xy, cmap="viridis", linewidth=0, antialiased=True)
    ax3d.scatter(x_grid[max_i, max_j], y_grid[max_i, max_j], phi_xy[max_i, max_j], c="red", s=28, depthshade=False, label="global max")
    ax3d.set_xlabel("ego x")
    ax3d.set_ylabel("ego y")
    ax3d.set_zlabel("Phi")
    ax3d.set_title("Phi(x, y) Surface")
    ax3d.legend(loc="upper right")
    fig.colorbar(surf, ax=ax3d, shrink=0.75, pad=0.1)

    ax2d = fig.add_subplot(1, 2, 2)
    contour = ax2d.contourf(x_grid, y_grid, phi_xy, levels=30, cmap="viridis")
    ax2d.contour(x_grid, y_grid, phi_xy, levels=10, colors="k", linewidths=0.3, alpha=0.5)
    ax2d.scatter(x_grid[max_i, max_j], y_grid[max_i, max_j], c="red", s=24, label="global max")
    ax2d.set_xlabel("ego x")
    ax2d.set_ylabel("ego y")
    ax2d.set_title("Phi(x, y) Contour")
    ax2d.legend(loc="upper right")
    fig.colorbar(contour, ax=ax2d, shrink=0.9, pad=0.02, label="Phi")

    fig.suptitle(
        f"goal={goal.tolist()}, fixed (vx, vy)=({fixed_vx:.3f}, {fixed_vy:.3f}), grid={xy_res}x{xy_res}\n"
        f"x_range=[{x_min:.3f}, {x_max:.3f}], y_range=[{y_min:.3f}, {y_max:.3f}]"
    )
    plt.tight_layout()
    plt.show()


def plot_phi_xvy_surface(
    state: SceneState,
    goal_phys: np.ndarray,
    xvy_res: int = 121,
    x_min: float | None = None,
    x_max: float | None = None,
    vy_min: float | None = None,
    vy_max: float | None = None,
) -> None:
    """Plot Phi over (x, vy) grid with ego y/vx fixed to state's values."""
    hiro_cfg = get_hiro_config()

    if str(getattr(hiro_cfg, "intrinsic_type", "")).lower() != "huber_shaping":
        raise ValueError("当前 conf 未启�?huber_shaping，请先设�?ENABLE_HIRO_REWARD_SHAPING=True�?)

    norm_ranges = np.asarray(hiro_cfg.intrinsic_norm_ranges, dtype=np.float32)
    weights = np.asarray(hiro_cfg.intrinsic_weights, dtype=np.float32)

    ego_ref = np.asarray(state.ego, dtype=np.float32).reshape(-1)
    goal = np.asarray(goal_phys, dtype=np.float32).reshape(-1)

    if x_min is None:
        x_min = float(goal[0] + norm_ranges[0, 0])
    if x_max is None:
        x_max = float(goal[0] + norm_ranges[0, 1])
    if vy_min is None:
        vy_min = float(goal[3] + norm_ranges[3, 0])
    if vy_max is None:
        vy_max = float(goal[3] + norm_ranges[3, 1])

    x_axis = np.linspace(float(x_min), float(x_max), int(xvy_res), dtype=np.float32)
    vy_axis = np.linspace(float(vy_min), float(vy_max), int(xvy_res), dtype=np.float32)
    x_grid, vy_grid = np.meshgrid(x_axis, vy_axis, indexing="xy")

    phi_xvy = np.zeros_like(x_grid, dtype=np.float32)
    fixed_y, fixed_vx = float(ego_ref[1]), float(ego_ref[2])

    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            ego_candidate = np.asarray([x_grid[i, j], fixed_y, fixed_vx, vy_grid[i, j]], dtype=np.float32).reshape(1, -1)
            err = ego_candidate - goal.reshape(1, -1)
            phi_xvy[i, j] = float(utils.huber_potential(err, norm_ranges=norm_ranges, weights=weights)[0])

    phi_max = float(np.max(phi_xvy))
    max_mask = np.isclose(phi_xvy, phi_max, rtol=1e-6, atol=1e-8)
    max_i, max_j = np.where(max_mask)

    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax3d.plot_surface(x_grid, vy_grid, phi_xvy, cmap="viridis", linewidth=0, antialiased=True)
    ax3d.scatter(x_grid[max_i, max_j], vy_grid[max_i, max_j], phi_xvy[max_i, max_j], c="red", s=28, depthshade=False, label="global max")
    ax3d.set_xlabel("ego x")
    ax3d.set_ylabel("ego vy")
    ax3d.set_zlabel("Phi")
    ax3d.set_title("Phi(x, vy) Surface")
    ax3d.legend(loc="upper right")
    fig.colorbar(surf, ax=ax3d, shrink=0.75, pad=0.1)

    ax2d = fig.add_subplot(1, 2, 2)
    contour = ax2d.contourf(x_grid, vy_grid, phi_xvy, levels=30, cmap="viridis")
    ax2d.contour(x_grid, vy_grid, phi_xvy, levels=10, colors="k", linewidths=0.3, alpha=0.5)
    ax2d.scatter(x_grid[max_i, max_j], vy_grid[max_i, max_j], c="red", s=24, label="global max")
    ax2d.set_xlabel("ego x")
    ax2d.set_ylabel("ego vy")
    ax2d.set_title("Phi(x, vy) Contour")
    ax2d.legend(loc="upper right")
    fig.colorbar(contour, ax=ax2d, shrink=0.9, pad=0.02, label="Phi")

    fig.suptitle(
        f"goal={goal.tolist()}, fixed (y, vx)=({fixed_y:.3f}, {fixed_vx:.3f}), grid={xvy_res}x{xvy_res}\n"
        f"x_range=[{x_min:.3f}, {x_max:.3f}], vy_range=[{vy_min:.3f}, {vy_max:.3f}]"
    )
    plt.tight_layout()
    plt.show()


def plot_phi_yvy_surface(
    state: SceneState,
    goal_phys: np.ndarray,
    yvy_res: int = 121,
    y_min: float | None = None,
    y_max: float | None = None,
    vy_min: float | None = None,
    vy_max: float | None = None,
) -> None:
    """Plot Phi over (y, vy) grid with ego x/vx fixed to state's values."""
    hiro_cfg = get_hiro_config()

    if str(getattr(hiro_cfg, "intrinsic_type", "")).lower() != "huber_shaping":
        raise ValueError("当前 conf 未启�?huber_shaping，请先设�?ENABLE_HIRO_REWARD_SHAPING=True�?)

    norm_ranges = np.asarray(hiro_cfg.intrinsic_norm_ranges, dtype=np.float32)
    weights = np.asarray(hiro_cfg.intrinsic_weights, dtype=np.float32)

    ego_ref = np.asarray(state.ego, dtype=np.float32).reshape(-1)
    goal = np.asarray(goal_phys, dtype=np.float32).reshape(-1)

    if y_min is None:
        y_min = float(goal[1] + norm_ranges[1, 0])
    if y_max is None:
        y_max = float(goal[1] + norm_ranges[1, 1])
    if vy_min is None:
        vy_min = float(goal[3] + norm_ranges[3, 0])
    if vy_max is None:
        vy_max = float(goal[3] + norm_ranges[3, 1])

    y_axis = np.linspace(float(y_min), float(y_max), int(yvy_res), dtype=np.float32)
    vy_axis = np.linspace(float(vy_min), float(vy_max), int(yvy_res), dtype=np.float32)
    y_grid, vy_grid = np.meshgrid(y_axis, vy_axis, indexing="xy")

    phi_yvy = np.zeros_like(y_grid, dtype=np.float32)
    fixed_x, fixed_vx = float(ego_ref[0]), float(ego_ref[2])

    for i in range(y_grid.shape[0]):
        for j in range(y_grid.shape[1]):
            ego_candidate = np.asarray([fixed_x, y_grid[i, j], fixed_vx, vy_grid[i, j]], dtype=np.float32).reshape(1, -1)
            err = ego_candidate - goal.reshape(1, -1)
            phi_yvy[i, j] = float(utils.huber_potential(err, norm_ranges=norm_ranges, weights=weights)[0])

    phi_max = float(np.max(phi_yvy))
    max_mask = np.isclose(phi_yvy, phi_max, rtol=1e-6, atol=1e-8)
    max_i, max_j = np.where(max_mask)

    fig = plt.figure(figsize=(12, 6))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax3d.plot_surface(y_grid, vy_grid, phi_yvy, cmap="viridis", linewidth=0, antialiased=True)
    ax3d.scatter(y_grid[max_i, max_j], vy_grid[max_i, max_j], phi_yvy[max_i, max_j], c="red", s=28, depthshade=False, label="global max")
    ax3d.set_xlabel("ego y")
    ax3d.set_ylabel("ego vy")
    ax3d.set_zlabel("Phi")
    ax3d.set_title("Phi(y, vy) Surface")
    ax3d.legend(loc="upper right")
    fig.colorbar(surf, ax=ax3d, shrink=0.75, pad=0.1)

    ax2d = fig.add_subplot(1, 2, 2)
    contour = ax2d.contourf(y_grid, vy_grid, phi_yvy, levels=30, cmap="viridis")
    ax2d.contour(y_grid, vy_grid, phi_yvy, levels=10, colors="k", linewidths=0.3, alpha=0.5)
    ax2d.scatter(y_grid[max_i, max_j], vy_grid[max_i, max_j], c="red", s=24, label="global max")
    ax2d.set_xlabel("ego y")
    ax2d.set_ylabel("ego vy")
    ax2d.set_title("Phi(y, vy) Contour")
    ax2d.legend(loc="upper right")
    fig.colorbar(contour, ax=ax2d, shrink=0.9, pad=0.02, label="Phi")

    fig.suptitle(
        f"goal={goal.tolist()}, fixed (x, vx)=({fixed_x:.3f}, {fixed_vx:.3f}), grid={yvy_res}x{yvy_res}\n"
        f"y_range=[{y_min:.3f}, {y_max:.3f}], vy_range=[{vy_min:.3f}, {vy_max:.3f}]"
    )
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Huber potential Phi(s,a) surface over low-level action space.")
    parser.add_argument("--lane-res", type=int, default=121, help="Grid resolution on lane action axis.")
    parser.add_argument("--accel-res", type=int, default=121, help="Grid resolution on acceleration action axis.")
    parser.add_argument("--show-intrinsic", action="store_true", help="Also plot shaping reward surface.")
    parser.add_argument("--csv-out", type=str, default="phi_surface_points.csv", help="Output CSV path for all sampled points.")
    parser.add_argument("--plot-xy-phi", action="store_true", help="Also plot Phi over full (x, y) grid with fixed (vx, vy).")
    parser.add_argument("--xy-res", type=int, default=121, help="Grid resolution for x/y Phi surface.")
    parser.add_argument("--plot-xvy-phi", action="store_true", help="Also plot Phi over (x, vy) grid with fixed (y, vx).")
    parser.add_argument("--xvy-res", type=int, default=121, help="Grid resolution for x/vy Phi surface.")
    parser.add_argument("--plot-yvy-phi", action="store_true", help="Also plot Phi over (y, vy) grid with fixed (x, vx).")
    parser.add_argument("--yvy-res", type=int, default=121, help="Grid resolution for y/vy Phi surface.")
    parser.add_argument("--x-min", type=float, default=20, help="Optional x lower bound for Phi(x,y).")
    parser.add_argument("--x-max", type=float, default=30, help="Optional x upper bound for Phi(x,y).")
    parser.add_argument("--y-min", type=float, default=0, help="Optional y lower bound for Phi(x,y).")
    parser.add_argument("--y-max", type=float, default=8.0, help="Optional y upper bound for Phi(x,y).")
    parser.add_argument("--vy-min", type=float, default=-2, help="Optional vy lower bound for Phi(x,vy)/Phi(y,vy).")
    parser.add_argument("--vy-max", type=float, default=2, help="Optional vy upper bound for Phi(x,vy)/Phi(y,vy).")
    args = parser.parse_args()

    # --------------------
    # 可在这里替换你的给定状�?s �?goal_phys
    # --------------------
    state = SceneState(
        ego=np.asarray([0, 4.0, 10.0, 0.0], dtype=np.float32),
        others=np.asarray(
            [
                [30.0, 4.0, 10.0, 0.0],
                [60.0, 8.0, 12.0, 0.0],
                [30.0, 0.0, 10.0, 0.0],
                [10.0, 8.0, 15.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    goal_phys = np.asarray([25, 4.0, 12.0, 0.0], dtype=np.float32)

    plot_phi_surface(
        state=state,
        goal_phys=goal_phys,
        lane_res=args.lane_res,
        accel_res=args.accel_res,
        show_intrinsic_surface=args.show_intrinsic,
        csv_out=args.csv_out,
    )

    if args.plot_xy_phi:
        plot_phi_xy_surface(
            state=state,
            goal_phys=goal_phys,
            xy_res=args.xy_res,
            x_min=args.x_min,
            x_max=args.x_max,
            y_min=args.y_min,
            y_max=args.y_max,
        )

    if args.plot_xvy_phi:
        plot_phi_xvy_surface(
            state=state,
            goal_phys=goal_phys,
            xvy_res=args.xvy_res,
            x_min=args.x_min,
            x_max=args.x_max,
            vy_min=args.vy_min,
            vy_max=args.vy_max,
        )

    if args.plot_yvy_phi:
        plot_phi_yvy_surface(
            state=state,
            goal_phys=goal_phys,
            yvy_res=args.yvy_res,
            y_min=args.y_min,
            y_max=args.y_max,
            vy_min=args.vy_min,
            vy_max=args.vy_max,
        )


if __name__ == "__main__":
    main()
