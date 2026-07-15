"""Evaluate trained HIRO low-level policies on one-interval snapshot tasks.

The test mirrors low-level snapshot training:

1. sample distinct background snapshots from a sharded pool;
2. choose one eligible background vehicle uniformly as ego;
3. sample one goal with the reachable-uniform goal sampler;
4. let each low-level model track the same (snapshot, ego, goal) case for one
   high-level interval, then reset.

Per-case results are written to CSV.  The JSON summary reports signed terminal
errors (terminal - goal), terminal MAE, the mean comfort-reward sum per
interval, and lane-change success among cases whose goal is in another lane.
"""

from __future__ import annotations

import bisect
import copy
import csv
import importlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import gymnasium as gym
import numpy as np

from configs.builders import get_env_config_for_scenario, get_scenario_spec
from rl.algos.HRL.goal_samplers import get_goal_sampler
from rl.algos.HRL.hiro_infer import HIROPolicyRunner
from rl.algos.sac.sac import SAC
from rl.utils import utils as hiro_utils
from stable_baselines3.common.buffers import ReplayBuffer
from util.config_utils import deep_update
from util.hiro_utils import (
    apply_hiro_config_overrides,
    env_config_from_run_config,
    hiro_config_from_run_config,
    load_hiro_run_config,
)


# ---------------------------------------------------------------------------
# Run configuration: edit values here, then run this script directly.
# ---------------------------------------------------------------------------
SNAPSHOT_POOL = Path("debug/background_snapshot_pool_slowlane2_oldEnv")
MODEL_PATHS = (
    # Path("models/hiro_260318_lowonly_uniform_RS_newSLv2_vio03_HER_reDim_v2/hiro_low_final.zip"),
    # Path("models/hiro_260328_lowonly_reachablePretrainedV2_Rainbow_amax3_dmin15_10/hiro_low_final.zip"),
    # Path("models/hiro_260630_lowonly_reUni_oldEnv_fixedHER_snapshot/hiro_low_final.zip"),
    # Path("models/hiro_260627_lowonly_uni_oldEnv_noHER_SLmpc_noaugObs/hiro_low_final.zip"),
    # Path("models/hiro_260627_lowonly_uni_oldEnv_fixedHER_SLmpc_noaugObs/hiro_low_final.zip"),
)
CONFIG_MODEL_DIR = Path("models/hiro_260627_lowonly_uni_oldEnv_noHER_SLmpc_noaugObs")
NUM_SNAPSHOTS = 1000
RANDOM_SEED = 42
DEVICE = "auto"
OUTPUT_DIR = Path("debug/low_snapshot_goal_tracking_eval")
# Save cases 50, 100, 150, ... for every model. Set to 0 to disable plotting.
PLOT_EVERY_N = 50


class _DummyHigh:
    def predict(self, obs: np.ndarray, deterministic: bool = True):
        del obs, deterministic
        return np.zeros((1, 3), dtype=np.float32), None


@dataclass(frozen=True)
class SnapshotRef:
    global_index: int
    chunk_file: str
    chunk_index: int


@dataclass
class EvalCase:
    case_id: int
    snapshot_ref: SnapshotRef
    snapshot: dict[str, Any]
    ego_index: int
    reset_seed: int
    goal_seed: int
    ego_start: list[float]
    goal_action: list[float]
    goal_phys: list[float]


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _pool_chunks(pool_path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    meta_path = pool_path / "meta.pkl"
    meta = _load_pickle(meta_path)
    if not isinstance(meta, Mapping):
        raise ValueError(f"Invalid snapshot-pool metadata: {meta_path}")

    chunks: list[dict[str, Any]] = []
    shards = meta.get("shards", {})
    if isinstance(shards, Mapping):
        for shard in shards.values():
            if not isinstance(shard, Mapping):
                continue
            for chunk in list(shard.get("chunks", []) or []):
                if isinstance(chunk, Mapping) and chunk.get("file"):
                    chunks.append({"file": str(chunk["file"]), "count": int(chunk["count"])})
    if not chunks:
        raise ValueError(f"No chunk entries found in {meta_path}")

    cumulative: list[int] = []
    total = 0
    for chunk in chunks:
        total += int(chunk["count"])
        cumulative.append(total)
    return chunks, cumulative


def _snapshot_ref(global_index: int, chunks: Sequence[Mapping[str, Any]], cumulative: Sequence[int]) -> SnapshotRef:
    chunk_pos = bisect.bisect_right(cumulative, int(global_index))
    previous = 0 if chunk_pos == 0 else int(cumulative[chunk_pos - 1])
    return SnapshotRef(
        global_index=int(global_index),
        chunk_file=str(chunks[chunk_pos]["file"]),
        chunk_index=int(global_index) - previous,
    )


def _load_snapshot_batch(pool_path: Path, refs: Sequence[SnapshotRef]) -> dict[int, dict[str, Any]]:
    grouped: dict[str, list[SnapshotRef]] = {}
    for ref in refs:
        grouped.setdefault(ref.chunk_file, []).append(ref)

    loaded: dict[int, dict[str, Any]] = {}
    for chunk_file, chunk_refs in grouped.items():
        payload = _load_pickle(pool_path / chunk_file)
        snapshots = payload.get("snapshots", None) if isinstance(payload, Mapping) else payload
        if not isinstance(snapshots, list):
            raise ValueError(f"Invalid snapshot chunk: {pool_path / chunk_file}")
        for ref in chunk_refs:
            snapshot = snapshots[ref.chunk_index]
            if isinstance(snapshot, Mapping):
                loaded[ref.global_index] = dict(snapshot)
    return loaded


def _force_snapshot_reset(env: gym.Env, snapshot: Mapping[str, Any], ego_index: int, seed: int) -> tuple[np.ndarray, dict]:
    """Use the environment's training reset path while forcing one case."""
    base_env = env.unwrapped
    base_env._sample_background_snapshot = lambda: snapshot
    base_env._snapshot_ego_candidate_indices = lambda vehicles: [int(ego_index)]
    obs, info = env.reset(seed=int(seed))
    return np.asarray(obs, dtype=np.float32), dict(info or {})


def _ego_substate(runner: HIROPolicyRunner, obs: np.ndarray) -> np.ndarray:
    _, kin, _ = runner._split(np.asarray(obs, dtype=np.float32))
    return np.asarray(runner._ego_sub(kin), dtype=np.float32).reshape(-1)


def _lane_id_from_y(y: float, lane_center_ys: Sequence[float]) -> int:
    centers = np.asarray(lane_center_ys, dtype=np.float32).reshape(-1)
    if centers.size == 0:
        raise ValueError("lane_center_ys must not be empty")
    return int(np.argmin(np.abs(centers - float(y))))


def _road_vehicle_states(env: gym.Env) -> list[dict[str, float | bool]]:
    base_env = env.unwrapped
    ego = getattr(base_env, "vehicle", None)
    states: list[dict[str, float | bool]] = []
    for vehicle in list(getattr(getattr(base_env, "road", None), "vehicles", []) or []):
        position = np.asarray(getattr(vehicle, "position", [np.nan, np.nan]), dtype=float).reshape(-1)
        if position.size < 2 or not np.all(np.isfinite(position[:2])):
            continue
        states.append(
            {
                "x": float(position[0]),
                "y": float(position[1]),
                "speed": float(getattr(vehicle, "speed", 0.0)),
                "is_ego": bool(vehicle is ego),
            }
        )
    return states


def _save_interval_tracking_snapshot(
    *,
    save_path: Path,
    model_name: str,
    case: EvalCase,
    trajectory: Sequence[Mapping[str, float]],
    initial_vehicles: Sequence[Mapping[str, float | bool]],
    final_vehicles: Sequence[Mapping[str, float | bool]],
    goal: np.ndarray,
    lane_center_ys: Sequence[float],
    lane_width: float,
    speed_limit: float,
    comfort_sum: float,
    lane_change_success: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    save_path.parent.mkdir(parents=True, exist_ok=True)
    centers = np.asarray(lane_center_ys, dtype=float).reshape(-1)
    traj_x = np.asarray([float(row["x"]) for row in trajectory], dtype=float)
    traj_y = np.asarray([float(row["y"]) for row in trajectory], dtype=float)
    steps = np.asarray([int(row["step"]) for row in trajectory], dtype=int)
    x_error = traj_x - float(goal[0])
    y_error = traj_y - float(goal[1])
    comfort_cumulative = np.cumsum(
        np.asarray([float(row.get("comfort_reward", 0.0)) for row in trajectory], dtype=float)
    )

    # Keep the road panel focused on the tracked interval.  A full snapshot can
    # contain vehicles across the entire 500 m road and would squash the lanes.
    core_x = list(traj_x) + [float(goal[0])]
    x_min, x_max = min(core_x), max(core_x)
    x_pad = max(18.0, 0.2 * max(x_max - x_min, 1.0))

    fig = plt.figure(figsize=(14.0, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 1, height_ratios=[2.1, 1.0])
    ax = fig.add_subplot(grid[0])
    ax_err = fig.add_subplot(grid[1])

    road_left, road_right = x_min - x_pad, x_max + x_pad
    for center in centers:
        ax.add_patch(
            Rectangle(
                (road_left, float(center) - 0.5 * lane_width),
                road_right - road_left,
                lane_width,
                facecolor="#666666",
                edgecolor="none",
                zorder=0,
            )
        )
        ax.axhline(float(center), color="white", lw=1.0, ls="--", alpha=0.8, zorder=1)
    ax.axhline(float(centers[0]) - 0.5 * lane_width, color="white", lw=1.1, zorder=1)
    ax.axhline(float(centers[-1]) + 0.5 * lane_width, color="white", lw=1.1, zorder=1)

    cmap = plt.get_cmap("jet")
    speed_norm = mcolors.Normalize(vmin=0.0, vmax=max(float(speed_limit), 1e-3))
    veh_length, veh_width = 5.0, 2.0
    for row in initial_vehicles:
        if not (road_left - veh_length <= float(row["x"]) <= road_right + veh_length):
            continue
        color = cmap(speed_norm(max(0.0, float(row["speed"]))))
        is_ego = bool(row["is_ego"])
        ax.add_patch(
            Rectangle(
                (float(row["x"]) - veh_length / 2.0, float(row["y"]) - veh_width / 2.0),
                veh_length,
                veh_width,
                facecolor=color,
                edgecolor="#00BFFF" if is_ego else "black",
                linewidth=2.0 if is_ego else 0.8,
                alpha=0.38 if not is_ego else 0.9,
                zorder=4 if is_ego else 2,
            )
        )
    for row in final_vehicles:
        if bool(row["is_ego"]):
            continue
        if not (road_left - veh_length <= float(row["x"]) <= road_right + veh_length):
            continue
        ax.add_patch(
            Rectangle(
                (float(row["x"]) - veh_length / 2.0, float(row["y"]) - veh_width / 2.0),
                veh_length,
                veh_width,
                facecolor="none",
                edgecolor="#E6E6E6",
                linewidth=0.8,
                linestyle=":",
                alpha=0.7,
                zorder=3,
            )
        )

    ax.plot(traj_x, traj_y, color="white", lw=2.0, alpha=0.85, zorder=6)
    scatter = ax.scatter(
        traj_x,
        traj_y,
        c=steps,
        cmap="viridis",
        s=38,
        edgecolors="black",
        linewidths=0.35,
        zorder=7,
    )
    ax.scatter(
        [float(goal[0])],
        [float(goal[1])],
        marker="*",
        s=180,
        c="#FF6F00",
        edgecolors="white",
        linewidths=1.2,
        zorder=9,
    )
    ax.scatter(
        [traj_x[-1]],
        [traj_y[-1]],
        marker="s",
        s=90,
        c="#D62728",
        edgecolors="white",
        linewidths=1.0,
        zorder=9,
    )
    ax.text(
        float(goal[0]) + 1.5,
        float(goal[1]) - 0.35,
        f"goal ({float(goal[0]):.1f}, {float(goal[1]):.1f})",
        fontsize=9,
        color="white",
        zorder=10,
        bbox=dict(facecolor="black", alpha=0.4, pad=1.5, edgecolor="none"),
    )
    ax.set_xlim(road_left, road_right)
    ax.set_ylim(float(centers[0]) - 0.65 * lane_width, float(centers[-1]) + 0.65 * lane_width)
    ax.set_yticks(centers)
    ax.set_yticklabels([f"lane {i}" for i in range(len(centers))])
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("absolute x [m]")
    ax.set_title(
        f"{model_name}\n"
        f"case {case.case_id} | terminal error dx={x_error[-1]:.2f} m, "
        f"dy={y_error[-1]:.2f} m | comfort sum={comfort_sum:.3f} | "
        f"lane-change success={lane_change_success}",
        fontsize=10,
    )
    ax.legend(
        handles=[
            Patch(facecolor="#777777", edgecolor="black", alpha=0.5, label="vehicles at start"),
            Patch(facecolor="none", edgecolor="#E6E6E6", linestyle=":", label="neighbors at terminal"),
            Line2D([0], [0], color="white", marker="o", label="ego trajectory"),
            Line2D([0], [0], marker="*", color="none", markerfacecolor="#FF6F00", markeredgecolor="white", markersize=12, label="goal"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="#D62728", markeredgecolor="white", markersize=8, label="terminal ego"),
        ],
        loc="upper left",
        fontsize=8,
        framealpha=0.75,
    )
    fig.colorbar(scatter, ax=ax, aspect=28, shrink=0.75, pad=0.01, label="low step")

    ax_err.axhline(0.0, color="#666666", lw=0.9, ls="--")
    ax_err.plot(steps, x_error, color="#4C78A8", lw=1.8, marker="o", ms=3, label="x error")
    ax_err.plot(steps, y_error, color="#F28E2B", lw=1.8, marker="o", ms=3, label="y error")
    ax_err.set_xlabel("low step (0 is interval start)")
    ax_err.set_ylabel("signed error: ego - goal [m]")
    ax_err.grid(True, alpha=0.25)
    ax_err.legend(loc="best", fontsize=8)
    ax_comfort = ax_err.twinx()
    ax_comfort.plot(
        steps,
        comfort_cumulative,
        color="#59A14F",
        lw=1.5,
        alpha=0.9,
        label="cumulative comfort reward",
    )
    ax_comfort.set_ylabel("cumulative comfort reward", color="#2E7D32")
    ax_comfort.tick_params(axis="y", labelcolor="#2E7D32")

    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def _training_high_action_space(env: gym.Env, hiro_cfg: Any) -> gym.spaces.Box:
    cfg = env.unwrapped.config
    high_interval = int(getattr(hiro_cfg, "high_interval", 25))
    dt = 1.0 / float(cfg.get("policy_frequency", 10.0))
    speed_min = 0.0
    speed_max = float(cfg.get("speed_limit", 15.0))
    horizon = float(high_interval) * dt
    low = np.array([speed_min * horizon, -1.0, speed_min], dtype=np.float32)
    high = np.array([speed_max * horizon, 1.0, speed_max], dtype=np.float32)
    fixed_goal_vx = getattr(hiro_cfg, "fixed_goal_vx", None)
    if fixed_goal_vx is not None:
        fixed_goal_vx = float(np.clip(float(fixed_goal_vx), speed_min, speed_max))
        low[2] = fixed_goal_vx - 0.01
        high[2] = fixed_goal_vx + 0.01
    return gym.spaces.Box(low, high, dtype=np.float32)


def _make_reachable_sampler(env: gym.Env, runner: HIROPolicyRunner, hiro_cfg: Any):
    action_space = _training_high_action_space(env, hiro_cfg)
    enable_vx_bounds = bool(getattr(hiro_cfg, "high_goal_safe_enable_goal_vx_bounds", True))
    fixed_goal_vx = getattr(hiro_cfg, "fixed_goal_vx", None)
    if fixed_goal_vx is not None and np.isclose(float(fixed_goal_vx), 0.0):
        enable_vx_bounds = False
    return get_goal_sampler(
        "reachable_uniform",
        action_space,
        bounds_fn=runner.high_goal_safe_bounds.compute_np,
        enable_vx_bounds=enable_vx_bounds,
        dynamic_feasible_lane_intervals=bool(
            getattr(hiro_cfg, "high_goal_dynamic_feasible_lane_intervals", False)
        ),
    )


def _align_low_observation_dim(runner: HIROPolicyRunner, low_model: SAC) -> None:
    shape = getattr(getattr(low_model, "observation_space", None), "shape", None)
    if not shape:
        return
    trained_dim = int(np.prod(shape))
    with_extra = int(1 + runner.local_kin_flat_dim + runner.obs_extra_dim + runner.ego_dim)
    without_extra = int(1 + runner.local_kin_flat_dim + runner.ego_dim)
    if trained_dim == with_extra:
        return
    if trained_dim == without_extra and runner.obs_extra_dim > 0:
        runner.obs_extra_dim = 0
        return
    raise ValueError(
        "Low-level observation dimension mismatch: "
        f"model={trained_dim}, env_with_extra={with_extra}, env_without_extra={without_extra}"
    )


def _load_low_model(path: Path, device: str) -> SAC:
    return SAC.load(
        str(path),
        device=device,
        custom_objects={"replay_buffer_class": ReplayBuffer, "replay_buffer_kwargs": {}},
    )


def _load_test_config(config_model_dir: Path, pool_path: Path) -> tuple[str, str, dict[str, Any], Any, str]:
    run_config, run_config_path = load_hiro_run_config(str(config_model_dir))
    hiro_cfg = hiro_config_from_run_config(run_config)
    hiro_cfg = apply_hiro_config_overrides(
        hiro_cfg,
        {"low_level_type": "sac", "goal_sampler": {"type": "reachable_uniform"}},
    )

    metadata = run_config.get("run_metadata", {})
    scenario_name = str(metadata.get("scenario_name", "multi_lane")) if isinstance(metadata, Mapping) else "multi_lane"
    scenario_spec = get_scenario_spec(scenario_name)
    importlib.import_module(str(scenario_spec["module"]))

    env_config = get_env_config_for_scenario(scenario_name)
    deep_update(env_config, env_config_from_run_config(run_config))
    env_config.pop("_env_seed", None)
    env_config.pop("actual_episode_start_phase_offset", None)
    env_config.update(
        {
            "background_snapshot_reset": True,
            "background_snapshot_path": str(pool_path),
            "background_snapshot_paths": [str(pool_path)],
            "low_snapshot_ego_from_background": True,
            "low_snapshot_ego_x_range": list(getattr(hiro_cfg, "low_snapshot_ego_x_range", None) or [0.0, 200.0]),
            "low_snapshot_ego_speed_range": list(
                getattr(hiro_cfg, "low_snapshot_ego_speed_range", None) or [7.0, 15.0]
            ),
            "inter_episode_as_steps": False,
            "warmup_each_episode": False,
            "enable_queue_takeover": False,
        }
    )
    policy_frequency = float(env_config.get("policy_frequency", 10.0))
    env_config["duration"] = float(int(getattr(hiro_cfg, "high_interval", 25))) / policy_frequency
    return scenario_name, str(scenario_spec["env_id"]), env_config, hiro_cfg, str(run_config_path)


def _iter_progress(items: Iterable[Any], total: int, label: str):
    try:
        from tqdm import tqdm

        return tqdm(items, total=total, desc=label)
    except ImportError:
        return items


def _select_snapshot_cases(
    env: gym.Env,
    pool_path: Path,
    num_cases: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[SnapshotRef, dict[str, Any], int, int, int]], int]:
    chunks, cumulative = _pool_chunks(pool_path)
    total_snapshots = int(cumulative[-1])
    if num_cases > total_snapshots:
        raise ValueError(f"Requested {num_cases} distinct snapshots, pool only has {total_snapshots}")

    order = rng.permutation(total_snapshots)
    selected: list[tuple[SnapshotRef, dict[str, Any], int, int, int]] = []
    skipped = 0
    cursor = 0
    batch_size = max(1000, num_cases)
    while len(selected) < num_cases and cursor < total_snapshots:
        batch_ids = order[cursor : min(cursor + batch_size, total_snapshots)]
        cursor += int(batch_ids.size)
        refs = [_snapshot_ref(int(index), chunks, cumulative) for index in batch_ids]
        snapshots = _load_snapshot_batch(pool_path, refs)
        for ref in refs:
            snapshot = snapshots.get(ref.global_index)
            if snapshot is None:
                skipped += 1
                continue
            candidates = env.unwrapped._snapshot_ego_candidate_indices(list(snapshot.get("vehicles", []) or []))
            if not candidates:
                skipped += 1
                continue
            ego_index = int(rng.choice(np.asarray(candidates, dtype=np.int64)))
            reset_seed = int(rng.integers(0, 2**31 - 1))
            goal_seed = int(rng.integers(0, 2**31 - 1))
            selected.append((ref, snapshot, ego_index, reset_seed, goal_seed))
            if len(selected) >= num_cases:
                break
    if len(selected) < num_cases:
        raise RuntimeError(f"Only found {len(selected)} usable snapshots out of {total_snapshots}")
    return selected, skipped


def _generate_goals(
    env: gym.Env,
    selected: Sequence[tuple[SnapshotRef, dict[str, Any], int, int, int]],
    low_model: SAC,
    hiro_cfg: Any,
) -> list[EvalCase]:
    runner = HIROPolicyRunner(
        _DummyHigh(),
        low_model,
        int(getattr(hiro_cfg, "high_interval", 25)),
        use_low_safety_layer=bool(getattr(hiro_cfg, "use_low_safety_layer", False)),
        config=copy.deepcopy(hiro_cfg),
    )
    cases: list[EvalCase] = []
    for case_id, (ref, snapshot, ego_index, reset_seed, goal_seed) in enumerate(
        _iter_progress(selected, len(selected), "generate goals"), start=1
    ):
        obs0, _ = _force_snapshot_reset(env, snapshot, ego_index, reset_seed)
        runner.init_from_env(env, obs0, float(getattr(hiro_cfg, "intrinsic_coef", 1.0)))
        ego_start = _ego_substate(runner, obs0)
        high_obs = runner._build_high_obs(obs0, env)
        sampler = _make_reachable_sampler(env, runner, hiro_cfg)

        random_state = np.random.get_state()
        try:
            np.random.seed(goal_seed)
            goal_action = np.asarray(sampler(high_obs.reshape(1, -1)), dtype=np.float32).reshape(1, -1)
        finally:
            np.random.set_state(random_state)
        goal_phys = hiro_utils.goal_action_to_abs(
            ego_start.reshape(1, -1),
            goal_action,
            runner.lane_center_ys,
            dynamic_feasible_intervals=bool(
                getattr(hiro_cfg, "high_goal_dynamic_feasible_lane_intervals", False)
            ),
        ).reshape(-1)

        cases.append(
            EvalCase(
                case_id=case_id,
                snapshot_ref=ref,
                snapshot=snapshot,
                ego_index=ego_index,
                reset_seed=reset_seed,
                goal_seed=goal_seed,
                ego_start=ego_start.tolist(),
                goal_action=goal_action.reshape(-1).tolist(),
                goal_phys=goal_phys.tolist(),
            )
        )
    return cases


def _evaluate_model(
    env: gym.Env,
    cases: Sequence[EvalCase],
    model_path: Path,
    low_model: SAC,
    hiro_cfg: Any,
    plot_every_n: int = 0,
    plot_root: Path | None = None,
) -> list[dict[str, Any]]:
    model_name = model_path.parent.name
    runner = HIROPolicyRunner(
        _DummyHigh(),
        low_model,
        int(getattr(hiro_cfg, "high_interval", 25)),
        use_low_safety_layer=bool(getattr(hiro_cfg, "use_low_safety_layer", False)),
        config=copy.deepcopy(hiro_cfg),
    )
    high_interval = int(getattr(hiro_cfg, "high_interval", 25))
    results: list[dict[str, Any]] = []

    for case in _iter_progress(cases, len(cases), model_name):
        obs, _ = _force_snapshot_reset(env, case.snapshot, case.ego_index, case.reset_seed)
        runner.init_from_env(env, obs, float(getattr(hiro_cfg, "intrinsic_coef", 1.0)))
        _align_low_observation_dim(runner, low_model)
        goal = np.asarray(case.goal_phys, dtype=np.float32)
        runner.goal_phys = goal.copy()
        runner.ego_start = _ego_substate(runner, obs).copy()
        runner.need_high = False
        runner.c = 0
        if hasattr(env.unwrapped, "set_hiro_goal"):
            env.unwrapped.set_hiro_goal(goal)

        should_plot = int(plot_every_n) > 0 and case.case_id % int(plot_every_n) == 0
        initial_vehicles = _road_vehicle_states(env) if should_plot else []
        ego_initial = _ego_substate(runner, obs)
        trajectory: list[dict[str, float]] = []
        if should_plot:
            trajectory.append(
                {
                    "step": 0.0,
                    "x": float(ego_initial[0]),
                    "y": float(ego_initial[1]),
                    "speed": float(ego_initial[2]),
                    "comfort_reward": 0.0,
                }
            )

        comfort_sum = 0.0
        terminated = False
        truncated = False
        collision = False
        steps = 0
        for _ in range(high_interval):
            runner.goal_phys = goal.copy()
            runner.need_high = False
            action = runner.act(env, obs)
            obs_next, _, terminated, truncated, info = env.step(action)
            obs = np.asarray(obs_next, dtype=np.float32)
            info = dict(info or {})
            reward_components = info.get("reward_components", {})
            comfort_step = 0.0
            if isinstance(reward_components, Mapping):
                comfort_step = float(reward_components.get("comfort_reward", 0.0))
                comfort_sum += comfort_step
            collision = bool(collision or info.get("crashed", False) or getattr(env.unwrapped.vehicle, "crashed", False))
            steps += 1
            if should_plot:
                ego_step = _ego_substate(runner, obs)
                trajectory.append(
                    {
                        "step": float(steps),
                        "x": float(ego_step[0]),
                        "y": float(ego_step[1]),
                        "speed": float(ego_step[2]),
                        "comfort_reward": float(comfort_step),
                    }
                )
            done = bool(terminated or truncated)
            runner.step_end(done, queue_takeover_active=bool(info.get("queue_takeover_active", False)))
            if done:
                break

        terminal = _ego_substate(runner, obs)
        signed_error = terminal - goal
        start_lane = _lane_id_from_y(float(case.ego_start[1]), runner.lane_center_ys)
        goal_lane = _lane_id_from_y(float(goal[1]), runner.lane_center_ys)
        terminal_lane = _lane_id_from_y(float(terminal[1]), runner.lane_center_ys)
        lane_change_required = goal_lane != start_lane
        goal_lane_reached = terminal_lane == goal_lane
        lane_change_success = bool(lane_change_required and goal_lane_reached)
        if should_plot:
            if plot_root is None:
                raise ValueError("plot_root is required when plot_every_n > 0")
            _save_interval_tracking_snapshot(
                save_path=plot_root / model_name / f"case_{case.case_id:05d}.png",
                model_name=model_name,
                case=case,
                trajectory=trajectory,
                initial_vehicles=initial_vehicles,
                final_vehicles=_road_vehicle_states(env),
                goal=goal,
                lane_center_ys=runner.lane_center_ys,
                lane_width=float(env.unwrapped.config.get("lane_width", 4.0)),
                speed_limit=float(env.unwrapped.config.get("speed_limit", 15.0)),
                comfort_sum=comfort_sum,
                lane_change_success=lane_change_success,
            )
        results.append(
            {
                "model": model_name,
                "model_path": str(model_path),
                "case_id": case.case_id,
                "snapshot_global_index": case.snapshot_ref.global_index,
                "chunk_file": case.snapshot_ref.chunk_file,
                "chunk_index": case.snapshot_ref.chunk_index,
                "ego_index": case.ego_index,
                "reset_seed": case.reset_seed,
                "goal_seed": case.goal_seed,
                "start_x": case.ego_start[0],
                "start_y": case.ego_start[1],
                "goal_x": float(goal[0]),
                "goal_y": float(goal[1]),
                "terminal_x": float(terminal[0]),
                "terminal_y": float(terminal[1]),
                "start_lane": int(start_lane),
                "goal_lane": int(goal_lane),
                "terminal_lane": int(terminal_lane),
                "lane_change_required": bool(lane_change_required),
                "goal_lane_reached": bool(goal_lane_reached),
                "lane_change_success": lane_change_success,
                "signed_error_x": float(signed_error[0]),
                "signed_error_y": float(signed_error[1]),
                "abs_error_x": float(abs(signed_error[0])),
                "abs_error_y": float(abs(signed_error[1])),
                "comfort_reward_sum": float(comfort_sum),
                "comfort_reward_per_step": float(comfort_sum / max(steps, 1)),
                "high_interval": int(high_interval),
                "steps": int(steps),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "collision": bool(collision),
            }
        )
    return results


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=0))


def _summarize(rows: Sequence[Mapping[str, Any]], model_path: Path) -> dict[str, Any]:
    x_signed = [float(row["signed_error_x"]) for row in rows]
    y_signed = [float(row["signed_error_y"]) for row in rows]
    lane_change_rows = [row for row in rows if bool(row["lane_change_required"])]
    lane_change_success_count = sum(bool(row["lane_change_success"]) for row in lane_change_rows)
    return {
        "model": model_path.parent.name,
        "model_path": str(model_path),
        "interval_count": len(rows),
        "terminal_error_definition": "terminal_ego - goal",
        "signed_error_x_mean": _mean(x_signed),
        "signed_error_x_std": _std(x_signed),
        "signed_error_y_mean": _mean(y_signed),
        "signed_error_y_std": _std(y_signed),
        "terminal_abs_error_x_mean": _mean([float(row["abs_error_x"]) for row in rows]),
        "terminal_abs_error_y_mean": _mean([float(row["abs_error_y"]) for row in rows]),
        "comfort_reward_interval_sum_mean": _mean([float(row["comfort_reward_sum"]) for row in rows]),
        "comfort_reward_step_mean": _mean([float(row["comfort_reward_per_step"]) for row in rows]),
        "goal_lane_success_ratio_all_intervals": _mean(
            [float(bool(row["goal_lane_reached"])) for row in rows]
        ),
        "lane_change_required_count": len(lane_change_rows),
        "lane_change_success_count": int(lane_change_success_count),
        "lane_change_success_ratio": (
            float(lane_change_success_count / len(lane_change_rows)) if lane_change_rows else None
        ),
        "lane_change_success_definition": (
            "among intervals with goal_lane != start_lane, terminal_lane == goal_lane"
        ),
        "executed_steps_mean": _mean([float(row["steps"]) for row in rows]),
        "early_done_ratio": _mean(
            [float(int(row["steps"]) < int(row["high_interval"])) for row in rows]
        ),
        "collision_ratio": _mean([float(bool(row["collision"])) for row in rows]),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_cases(path: Path, cases: Sequence[EvalCase]) -> None:
    payload = [
        {
            "case_id": case.case_id,
            "snapshot_global_index": case.snapshot_ref.global_index,
            "chunk_file": case.snapshot_ref.chunk_file,
            "chunk_index": case.snapshot_ref.chunk_index,
            "ego_index": case.ego_index,
            "reset_seed": case.reset_seed,
            "goal_seed": case.goal_seed,
            "ego_start": case.ego_start,
            "goal_action": case.goal_action,
            "goal_phys": case.goal_phys,
        }
        for case in cases
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    if NUM_SNAPSHOTS <= 0:
        raise ValueError("NUM_SNAPSHOTS must be positive")
    if PLOT_EVERY_N < 0:
        raise ValueError("PLOT_EVERY_N must be non-negative")
    if not SNAPSHOT_POOL.is_dir():
        raise FileNotFoundError(SNAPSHOT_POOL)
    if not MODEL_PATHS:
        raise ValueError("MODEL_PATHS must contain at least one checkpoint")
    for model_path in MODEL_PATHS:
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

    scenario_name, env_id, env_config, hiro_cfg, config_source = _load_test_config(
        CONFIG_MODEL_DIR, SNAPSHOT_POOL
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"scenario={scenario_name}, env_id={env_id}, high_interval={hiro_cfg.high_interval}, "
        f"pool={SNAPSHOT_POOL}, cases={NUM_SNAPSHOTS}"
    )
    print(f"config_source={config_source}")
    print(
        "ego_filter="
        f"x{env_config['low_snapshot_ego_x_range']}, speed{env_config['low_snapshot_ego_speed_range']}; "
        "goal_sampler=reachable_uniform"
    )

    env = gym.make(env_id, render_mode=None, config=env_config)
    loaded_models: dict[Path, SAC] = {}
    try:
        rng = np.random.default_rng(RANDOM_SEED)
        selected, skipped = _select_snapshot_cases(env, SNAPSHOT_POOL, NUM_SNAPSHOTS, rng)
        print(f"selected={len(selected)}, skipped_without_eligible_ego={skipped}")

        first_model_path = MODEL_PATHS[0]
        loaded_models[first_model_path] = _load_low_model(first_model_path, DEVICE)
        cases = _generate_goals(env, selected, loaded_models[first_model_path], hiro_cfg)
        _write_cases(OUTPUT_DIR / "cases.json", cases)

        all_rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for model_path in MODEL_PATHS:
            if model_path not in loaded_models:
                loaded_models[model_path] = _load_low_model(model_path, DEVICE)
            rows = _evaluate_model(
                env,
                cases,
                model_path,
                loaded_models[model_path],
                hiro_cfg,
                plot_every_n=PLOT_EVERY_N,
                plot_root=OUTPUT_DIR / "tracking_snapshots",
            )
            all_rows.extend(rows)
            summaries.append(_summarize(rows, model_path))
            print(json.dumps(summaries[-1], ensure_ascii=False, indent=2))

        _write_csv(OUTPUT_DIR / "interval_results.csv", all_rows)
        summary = {
            "pool": str(SNAPSHOT_POOL),
            "seed": int(RANDOM_SEED),
            "requested_snapshot_count": int(NUM_SNAPSHOTS),
            "usable_snapshot_count": len(cases),
            "skipped_without_eligible_ego": int(skipped),
            "high_interval": int(getattr(hiro_cfg, "high_interval", 25)),
            "ego_x_range": env_config["low_snapshot_ego_x_range"],
            "ego_speed_range": env_config["low_snapshot_ego_speed_range"],
            "goal_sampler": "reachable_uniform",
            "same_cases_for_all_models": True,
            "plot_every_n": int(PLOT_EVERY_N),
            "tracking_snapshot_dir": (
                str(OUTPUT_DIR / "tracking_snapshots") if PLOT_EVERY_N > 0 else None
            ),
            "config_source": config_source,
            "models": summaries,
        }
        (OUTPUT_DIR / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"results={OUTPUT_DIR / 'interval_results.csv'}")
        print(f"summary={OUTPUT_DIR / 'summary.json'}")
        if PLOT_EVERY_N > 0:
            print(f"tracking_snapshots={OUTPUT_DIR / 'tracking_snapshots'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
