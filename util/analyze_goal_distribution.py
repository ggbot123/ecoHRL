from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ----------------------------
# User-editable configuration
# ----------------------------
CSV_PATH = Path(
	r"d:\workspace\python\ecoHRL\logs\current\hiro_260311_lowonly_uniform_RS_newSLv2_vioPenalty03_HER\low_obs_hi_start.csv"
)
GOAL_DIM = 4
GOAL_X_IDX = 0
GOAL_Y_IDX = 1
PLOT_MODE = "both"  # "rel" | "abs" | "both"

# low_obs = [t_norm, local_kin_flat, goal_rel]
# In local_kin_flat ego vehicle features are typically [presence, x, y, vx, vy].
EGO_X_IN_LOW_OBS = 2
EGO_Y_IN_LOW_OBS = 3

PLOT_ALPHA = 0.35
PLOT_SIZE = 10.0
PLOT_TITLE = "Goal Distribution on X-Y Plane"
PLOT_SURFACE = True
SURFACE_BINS_X = 60
SURFACE_BINS_Y = 40
SURFACE_CMAP = "viridis"

SAVE_PATH: Path | None = Path(
	r"d:\workspace\python\ecoHRL\logs\current\hiro_260311_lowonly_uniform_RS_newSLv2_vioPenalty03_HER\goal_xy_dist.png"
)
SHOW_FIGURE = True


def parse_low_obs(raw: str) -> np.ndarray:
	"""Parse one low_obs field string like "[0.00,1.00,...]" into float array."""
	text = str(raw).strip()
	if not text:
		raise ValueError("empty low_obs")

	if text.startswith("[") and text.endswith("]"):
		text = text[1:-1]

	arr = np.fromstring(text, sep=",", dtype=np.float32)
	if arr.size == 0:
		raise ValueError(f"failed to parse low_obs: {raw!r}")
	return arr


def iter_goal_xy(
	csv_path: Path,
	goal_dim: int,
	goal_x_idx: int,
	goal_y_idx: int,
) -> Iterable[tuple[float, float]]:
	"""Yield (goal_x, goal_y) parsed from each valid row's low_obs."""
	with csv_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		if "low_obs" not in (reader.fieldnames or []):
			raise KeyError(f"'low_obs' column not found in {csv_path}")

		for row in reader:
			raw = row.get("low_obs", "")
			try:
				obs = parse_low_obs(raw)
			except Exception:
				continue

			if obs.size < goal_dim:
				continue

			goal = obs[-goal_dim:]
			if goal_x_idx >= goal.size or goal_y_idx >= goal.size:
				continue

			yield float(goal[goal_x_idx]), float(goal[goal_y_idx])


def iter_goal_abs_xy(
	csv_path: Path,
	goal_dim: int,
	goal_x_idx: int,
	goal_y_idx: int,
	ego_x_idx: int,
	ego_y_idx: int,
) -> Iterable[tuple[float, float]]:
	"""Yield reconstructed absolute goal (x, y): goal_abs = ego_abs + goal_rel."""
	with csv_path.open("r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		if "low_obs" not in (reader.fieldnames or []):
			raise KeyError(f"'low_obs' column not found in {csv_path}")

		for row in reader:
			raw = row.get("low_obs", "")
			try:
				obs = parse_low_obs(raw)
			except Exception:
				continue

			if obs.size < goal_dim:
				continue

			if ego_x_idx >= obs.size or ego_y_idx >= obs.size:
				continue

			goal_rel = obs[-goal_dim:]
			if goal_x_idx >= goal_rel.size or goal_y_idx >= goal_rel.size:
				continue

			ego_x = float(obs[ego_x_idx])
			ego_y = float(obs[ego_y_idx])
			yield ego_x + float(goal_rel[goal_x_idx]), ego_y + float(goal_rel[goal_y_idx])


def plot_goal_xy(
	points: np.ndarray,
	alpha: float = 0.35,
	size: float = 10.0,
	title: str = "Goal Distribution on X-Y Plane",
	) -> plt.Figure:
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.scatter(points[:, 0], points[:, 1], s=size, alpha=alpha)
	ax.set_xlabel("goal_x")
	ax.set_ylabel("goal_y")
	ax.set_title(title)
	ax.grid(True, linestyle="--", alpha=0.3)
	plt.tight_layout()
	return fig


def plot_goal_density_surface(
	points: np.ndarray,
	title: str,
	bins_x: int,
	bins_y: int,
	cmap: str,
) -> plt.Figure:
	"""Plot a 3D surface of P(goal_x, goal_y) estimated by a 2D histogram density."""
	x = points[:, 0]
	y = points[:, 1]

	hist, x_edges, y_edges = np.histogram2d(x, y, bins=[int(bins_x), int(bins_y)], density=True)
	x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
	y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
	x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing="ij")

	fig = plt.figure(figsize=(9, 6))
	ax = fig.add_subplot(111, projection="3d")
	ax.plot_surface(x_grid, y_grid, hist, cmap=cmap, linewidth=0, antialiased=True)
	ax.set_xlabel("goal_x")
	ax.set_ylabel("goal_y")
	ax.set_zlabel("density")
	ax.set_title(title)
	plt.tight_layout()
	return fig


def _save_figure(fig: plt.Figure, base_path: Path, suffix: str) -> Path:
	if suffix:
		out = base_path.with_name(f"{base_path.stem}_{suffix}{base_path.suffix}")
	else:
		out = base_path
	fig.savefig(out, dpi=180)
	return out


def main() -> None:
	mode = str(PLOT_MODE).lower()
	if mode not in {"rel", "abs", "both"}:
		raise ValueError(f"Invalid PLOT_MODE={PLOT_MODE!r}, expected 'rel' | 'abs' | 'both'.")

	figs_to_save: list[tuple[plt.Figure, str]] = []

	if mode in {"rel", "both"}:
		points_rel = np.asarray(
			list(
				iter_goal_xy(
					csv_path=CSV_PATH,
					goal_dim=GOAL_DIM,
					goal_x_idx=GOAL_X_IDX,
					goal_y_idx=GOAL_Y_IDX,
				)
			),
			dtype=np.float32,
		)
		if points_rel.size == 0:
			raise RuntimeError("No valid relative goal points were extracted.")
		fig_rel = plot_goal_xy(points_rel, alpha=PLOT_ALPHA, size=PLOT_SIZE, title=f"{PLOT_TITLE} (Relative)")
		figs_to_save.append((fig_rel, "rel_scatter"))
		if PLOT_SURFACE:
			fig_rel_surface = plot_goal_density_surface(
				points_rel,
				title=f"Goal Density Surface (Relative)",
				bins_x=SURFACE_BINS_X,
				bins_y=SURFACE_BINS_Y,
				cmap=SURFACE_CMAP,
			)
			figs_to_save.append((fig_rel_surface, "rel_surface"))

	if mode in {"abs", "both"}:
		points_abs = np.asarray(
			list(
				iter_goal_abs_xy(
					csv_path=CSV_PATH,
					goal_dim=GOAL_DIM,
					goal_x_idx=GOAL_X_IDX,
					goal_y_idx=GOAL_Y_IDX,
					ego_x_idx=EGO_X_IN_LOW_OBS,
					ego_y_idx=EGO_Y_IN_LOW_OBS,
				)
			),
			dtype=np.float32,
		)
		if points_abs.size == 0:
			raise RuntimeError("No valid absolute goal points were reconstructed.")
		fig_abs = plot_goal_xy(points_abs, alpha=PLOT_ALPHA, size=PLOT_SIZE, title=f"{PLOT_TITLE} (Absolute)")
		figs_to_save.append((fig_abs, "abs_scatter"))
		if PLOT_SURFACE:
			fig_abs_surface = plot_goal_density_surface(
				points_abs,
				title=f"Goal Density Surface (Absolute)",
				bins_x=SURFACE_BINS_X,
				bins_y=SURFACE_BINS_Y,
				cmap=SURFACE_CMAP,
			)
			figs_to_save.append((fig_abs_surface, "abs_surface"))

	if SAVE_PATH is not None:
		SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
		for fig, suffix in figs_to_save:
			out = _save_figure(fig, SAVE_PATH, suffix)
			print(f"Saved figure to: {out}")

	if SHOW_FIGURE:
		plt.show()


if __name__ == "__main__":
	main()
