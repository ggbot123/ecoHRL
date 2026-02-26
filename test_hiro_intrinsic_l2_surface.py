from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from rl.utils import utils


@dataclass
class SceneState:
    """State template for grid plotting.

    ego layout: [x, y, vx, vy]
    """

    ego: np.ndarray


# Params corresponding to ENABLE_HIRO_REWARD_SHAPING=False in configs/conf.py
L2_INTRINSIC_COEF = 10.0
L2_INTRINSIC_NORM_RANGES = np.asarray(
    [
        [0.0, 37.5],
        [-8.0, 8.0],
        [-8.0, 8.0],
        [-2.0, 2.0],
    ],
    dtype=np.float32,
)
L2_INTRINSIC_WEIGHTS = np.asarray([1.0, 2.0, 0.0, 0.3], dtype=np.float32)


def _plot_surface_and_contour(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_grid: np.ndarray,
    x_label: str,
    y_label: str,
    title_prefix: str,
) -> None:
    z_max = float(np.max(z_grid))
    max_mask = np.isclose(z_grid, z_max, rtol=1e-6, atol=1e-8)
    max_i, max_j = np.where(max_mask)

    fig = plt.figure(figsize=(12, 6))

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    surf = ax3d.plot_surface(x_grid, y_grid, z_grid, cmap="viridis", linewidth=0, antialiased=True)
    ax3d.scatter(
        x_grid[max_i, max_j],
        y_grid[max_i, max_j],
        z_grid[max_i, max_j],
        c="red",
        s=28,
        depthshade=False,
        label="global max",
    )
    ax3d.set_xlabel(x_label)
    ax3d.set_ylabel(y_label)
    ax3d.set_zlabel("intrinsic reward")
    ax3d.set_title(f"{title_prefix} Surface")
    ax3d.legend(loc="upper right")
    fig.colorbar(surf, ax=ax3d, shrink=0.75, pad=0.1)

    ax2d = fig.add_subplot(1, 2, 2)
    contour = ax2d.contourf(x_grid, y_grid, z_grid, levels=30, cmap="viridis")
    ax2d.contour(x_grid, y_grid, z_grid, levels=10, colors="k", linewidths=0.3, alpha=0.5)
    ax2d.scatter(
        x_grid[max_i, max_j],
        y_grid[max_i, max_j],
        c="red",
        s=24,
        label="global max",
    )
    ax2d.set_xlabel(x_label)
    ax2d.set_ylabel(y_label)
    ax2d.set_title(f"{title_prefix} Contour")
    ax2d.legend(loc="upper right")
    fig.colorbar(contour, ax=ax2d, shrink=0.9, pad=0.02, label="intrinsic reward")

    plt.tight_layout()
    plt.show()


def _intrinsic_reward_grid(ego_candidates: np.ndarray, goal_phys: np.ndarray) -> np.ndarray:
    """Compute L2 intrinsic reward for candidates of shape (N, 4)."""
    goal = np.asarray(goal_phys, dtype=np.float32).reshape(1, -1)
    ego = np.asarray(ego_candidates, dtype=np.float32)
    reward, _, _ = utils.intrinsic_reward_l2(
        ego_next_sub_rel=ego,
        goal_rel=np.repeat(goal, ego.shape[0], axis=0),
        norm_ranges=L2_INTRINSIC_NORM_RANGES,
        coef=float(L2_INTRINSIC_COEF),
        weights=L2_INTRINSIC_WEIGHTS,
    )
    return reward.astype(np.float32)


def plot_intrinsic_xy(
    state: SceneState,
    goal_phys: np.ndarray,
    res: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    x_axis = np.linspace(float(x_min), float(x_max), int(res), dtype=np.float32)
    y_axis = np.linspace(float(y_min), float(y_max), int(res), dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_axis, y_axis, indexing="xy")

    fixed_vx = float(state.ego[2])
    fixed_vy = float(state.ego[3])
    ego_candidates = np.stack(
        [
            x_grid.ravel(),
            y_grid.ravel(),
            np.full(x_grid.size, fixed_vx, dtype=np.float32),
            np.full(x_grid.size, fixed_vy, dtype=np.float32),
        ],
        axis=1,
    )
    rewards = _intrinsic_reward_grid(ego_candidates, goal_phys).reshape(x_grid.shape)

    _plot_surface_and_contour(
        x_grid,
        y_grid,
        rewards,
        x_label="x",
        y_label="y",
        title_prefix="Intrinsic Reward (x, y)",
    )


def plot_intrinsic_xvy(
    state: SceneState,
    goal_phys: np.ndarray,
    res: int,
    x_min: float,
    x_max: float,
    vy_min: float,
    vy_max: float,
) -> None:
    x_axis = np.linspace(float(x_min), float(x_max), int(res), dtype=np.float32)
    vy_axis = np.linspace(float(vy_min), float(vy_max), int(res), dtype=np.float32)
    x_grid, vy_grid = np.meshgrid(x_axis, vy_axis, indexing="xy")

    fixed_y = float(state.ego[1])
    fixed_vx = float(state.ego[2])
    ego_candidates = np.stack(
        [
            x_grid.ravel(),
            np.full(x_grid.size, fixed_y, dtype=np.float32),
            np.full(x_grid.size, fixed_vx, dtype=np.float32),
            vy_grid.ravel(),
        ],
        axis=1,
    )
    rewards = _intrinsic_reward_grid(ego_candidates, goal_phys).reshape(x_grid.shape)

    _plot_surface_and_contour(
        x_grid,
        vy_grid,
        rewards,
        x_label="x",
        y_label="vy",
        title_prefix="Intrinsic Reward (x, vy)",
    )


def plot_intrinsic_yvy(
    state: SceneState,
    goal_phys: np.ndarray,
    res: int,
    y_min: float,
    y_max: float,
    vy_min: float,
    vy_max: float,
) -> None:
    y_axis = np.linspace(float(y_min), float(y_max), int(res), dtype=np.float32)
    vy_axis = np.linspace(float(vy_min), float(vy_max), int(res), dtype=np.float32)
    y_grid, vy_grid = np.meshgrid(y_axis, vy_axis, indexing="xy")

    fixed_x = float(state.ego[0])
    fixed_vx = float(state.ego[2])
    ego_candidates = np.stack(
        [
            np.full(y_grid.size, fixed_x, dtype=np.float32),
            y_grid.ravel(),
            np.full(y_grid.size, fixed_vx, dtype=np.float32),
            vy_grid.ravel(),
        ],
        axis=1,
    )
    rewards = _intrinsic_reward_grid(ego_candidates, goal_phys).reshape(y_grid.shape)

    _plot_surface_and_contour(
        y_grid,
        vy_grid,
        rewards,
        x_label="y",
        y_label="vy",
        title_prefix="Intrinsic Reward (y, vy)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot L2 intrinsic reward surfaces for ENABLE_HIRO_REWARD_SHAPING=False style settings."
    )
    parser.add_argument("--res", type=int, default=121, help="Grid resolution for all spaces.")
    parser.add_argument("--x-min", type=float, default=20.0)
    parser.add_argument("--x-max", type=float, default=30.0)
    parser.add_argument("--y-min", type=float, default=0.0)
    parser.add_argument("--y-max", type=float, default=8.0)
    parser.add_argument("--vy-min", type=float, default=-2.0)
    parser.add_argument("--vy-max", type=float, default=2.0)
    args = parser.parse_args()

    state = SceneState(ego=np.asarray([0.0, 4.0, 10.0, 0.0], dtype=np.float32))
    goal_phys = np.asarray([25.0, 4.0, 12.0, 0.0], dtype=np.float32)

    print(
        "[L2 intrinsic params] "
        f"coef={L2_INTRINSIC_COEF}, "
        f"norm_ranges={L2_INTRINSIC_NORM_RANGES.tolist()}, "
        f"weights={L2_INTRINSIC_WEIGHTS.tolist()}"
    )

    plot_intrinsic_xy(
        state=state,
        goal_phys=goal_phys,
        res=args.res,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
    )
    plot_intrinsic_xvy(
        state=state,
        goal_phys=goal_phys,
        res=args.res,
        x_min=args.x_min,
        x_max=args.x_max,
        vy_min=args.vy_min,
        vy_max=args.vy_max,
    )
    plot_intrinsic_yvy(
        state=state,
        goal_phys=goal_phys,
        res=args.res,
        y_min=args.y_min,
        y_max=args.y_max,
        vy_min=args.vy_min,
        vy_max=args.vy_max,
    )


if __name__ == "__main__":
    main()
