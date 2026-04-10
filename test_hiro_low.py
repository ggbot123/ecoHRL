import os
import csv
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import scenarios.multi_lane  # 触发 __init__.py 里的 register

from configs.conf import get_env_config, get_hiro_config
from rl.algos.sac.sac import SAC
from rl.algos.HRL.hiro_infer import HIROPolicyRunner
from rl.algos.HRL.goal_samplers import UniformGoalSampler
from rl.utils import utils as hiro_utils
from util.mpc import MPCController
from util.plot_result import (
    save_speed_acc_curves,
    save_low_step_snapshot,
    save_goal_metric_summary,
    save_q_sa_surface_plot,
    save_q_sa_global_summary,
)
from util.hiro_low_test_utils import (
    setup_env_with_state,
    build_high_action_space,
    default_metric_fn,
    abs_dx_metric_fn,
    abs_dy_metric_fn,
    load_test_cases_from_csv,
    evaluate_q_sa_surface,
)
from util.hiro_low_batch_utils import run_batch_random_neighbors_acc_eval


class _DummyHigh:
    """占位高层模型，避免低层测试时强制依赖高层模型。"""

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        # HIRO high action 维度通常为 3：[dx, y_code, vx]
        return np.zeros((1, 3), dtype=np.float32), None


_abs_dx_metric_fn = abs_dx_metric_fn
_abs_dy_metric_fn = abs_dy_metric_fn


def run_mpc_theoretical_optimal(
    env,
    ego_state: Sequence[float],
    neighbors_state: Sequence[Sequence[float]],
    goal_phys: Sequence[float],
    out_dir: str,
    horizon: int,
    steps_to_goal: int,
    mpc_mode: str = "qp",
    mpc_global_maxiter: int = 250,
    mpc_plot_alternative_optima: bool = False,
    mpc_max_alternative_optima: int = 3,
):
    os.makedirs(out_dir, exist_ok=True)

    base_env, _, _ = setup_env_with_state(env, ego_state, neighbors_state)
    hiro_cfg = get_hiro_config()
    mpc = MPCController(
        base_env,
        horizon=int(max(1, horizon)),
        dt=1.0 / float(base_env.config.get("policy_frequency", 10.0)),
        intrinsic_coef=float(getattr(hiro_cfg, "intrinsic_coef", 1.0)),
        intrinsic_type=str(getattr(hiro_cfg, "intrinsic_type", "l2")),
        intrinsic_norm_ranges=getattr(hiro_cfg, "intrinsic_norm_ranges", None),
        intrinsic_weights=getattr(hiro_cfg, "intrinsic_weights", None),
    )

    mode = str(mpc_mode).lower().strip()
    if mode == "joint_global":
        result = mpc.plan_joint_optimal(
            goal_phys=goal_phys,
            steps_to_goal=int(max(1, steps_to_goal)),
            maxiter=int(max(1, mpc_global_maxiter)),
            enumerate_alternative_optima=bool(mpc_plot_alternative_optima),
            max_alternative_optima=int(max(0, mpc_max_alternative_optima)),
        )
    else:
        result = mpc.plan(goal_phys=goal_phys, steps_to_goal=int(max(1, steps_to_goal)))

    summary = {
        "success": bool(result.get("success", False)),
        "message": str(result.get("message", "")),
        "iterations": int(result.get("iterations", -1)),
        "horizon": int(result.get("horizon", horizon)),
        "steps_to_goal": int(result.get("steps_to_goal", steps_to_goal)),
        "mpc_mode": mode,
        "intrinsic_type": str(getattr(hiro_cfg, "intrinsic_type", "l2")),
        "progress_objective_enabled": bool(result.get("progress_objective_enabled", True)),
        "approximation_notes": list(result.get("approximation_notes", [])),
        "sum_low_ext": float(result.get("sum_low_ext", 0.0)),
        "sum_intrinsic": float(result.get("sum_intrinsic", 0.0)),
        "sum_low_total": float(result.get("sum_low_total", 0.0)),
        "reward_components": dict(result.get("reward_components", {})),
        "solver": dict(result.get("solver", {})),
        "alternative_optima": [
            {
                "index": int(i + 1),
                "fun": float(sol.get("fun", 0.0)),
                "sum_low_total": float(sol.get("sum_low_total", 0.0)),
                "sum_low_ext": float(sol.get("sum_low_ext", 0.0)),
                "sum_intrinsic": float(sol.get("sum_intrinsic", 0.0)),
            }
            for i, sol in enumerate(list(result.get("alternative_optima", [])))
        ],
        "start_state": np.asarray(result.get("start_state", []), dtype=np.float32).reshape(-1).tolist(),
        "goal_phys": np.asarray(result.get("goal_phys", []), dtype=np.float32).reshape(-1).tolist(),
    }

    summary_path = os.path.join(out_dir, "mpc_low_theoretical_optimal_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    actions = np.asarray(result.get("best_actions_cont", []), dtype=np.float32)
    states = np.asarray(result.get("states", []), dtype=np.float32)
    acc_norm = np.asarray(result.get("acc_norm", []), dtype=np.float32).reshape(-1)
    acc_phys = np.asarray(result.get("acc_phys", []), dtype=np.float32).reshape(-1)
    intrinsic_step = np.asarray(result.get("intrinsic_step", []), dtype=np.float32).reshape(-1)
    comfort_step = np.asarray(result.get("comfort_step", []), dtype=np.float32).reshape(-1)
    low_ext_step = np.asarray(result.get("low_ext_step", []), dtype=np.float32).reshape(-1)
    low_total_step = np.asarray(result.get("low_total_step", []), dtype=np.float32).reshape(-1)

    trajectory_rows: List[Dict[str, Any]] = []
    n_rows = int(min(actions.shape[0], states.shape[0] - 1)) if states.ndim == 2 else int(actions.shape[0])
    for i in range(n_rows):
        s = states[i + 1] if states.ndim == 2 and i + 1 < states.shape[0] else np.zeros(4, dtype=np.float32)
        a = actions[i] if actions.ndim == 2 and i < actions.shape[0] else np.zeros(2, dtype=np.float32)
        trajectory_rows.append(
            {
                "step": int(i + 1),
                "lane_scalar": float(a[0]),
                "acc_norm": float(a[1]) if a.shape[0] > 1 else (float(acc_norm[i]) if i < acc_norm.shape[0] else 0.0),
                "acc_phys": float(acc_phys[i]) if i < acc_phys.shape[0] else 0.0,
                "pred_x": float(s[0]) if s.shape[0] > 0 else 0.0,
                "pred_y": float(s[1]) if s.shape[0] > 1 else 0.0,
                "pred_vx": float(s[2]) if s.shape[0] > 2 else 0.0,
                "pred_vy": float(s[3]) if s.shape[0] > 3 else 0.0,
                "low_ext_step": float(low_ext_step[i]) if i < low_ext_step.shape[0] else 0.0,
                "comfort_step": float(comfort_step[i]) if i < comfort_step.shape[0] else 0.0,
                "intrinsic_step": float(intrinsic_step[i]) if i < intrinsic_step.shape[0] else 0.0,
                "low_total_step": float(low_total_step[i]) if i < low_total_step.shape[0] else 0.0,
            }
        )

    traj_csv = os.path.join(out_dir, "mpc_low_theoretical_optimal_trajectory.csv")
    if trajectory_rows:
        with open(traj_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(trajectory_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trajectory_rows)

    # 保存 MPC 理论最优对应的速度/加速度/换道曲线
    lane_width = float(base_env.config.get("lane_width", 4.0))
    n_steps = int(acc_phys.shape[0])
    t = np.arange(1, n_steps + 1, dtype=np.int32)
    pred_vx = states[1 : n_steps + 1, 2] if states.ndim == 2 and states.shape[0] >= n_steps + 1 else np.zeros((n_steps,), dtype=np.float32)
    lane_scalar = actions[:n_steps, 0] if actions.ndim == 2 and actions.shape[0] >= n_steps else np.zeros((n_steps,), dtype=np.float32)
    lane_id = np.rint(states[1 : n_steps + 1, 1] / max(lane_width, 1e-6)).astype(np.int32) if states.ndim == 2 and states.shape[0] >= n_steps + 1 else np.zeros((n_steps,), dtype=np.int32)

    alt_solutions_raw = list(result.get("alternative_optima", []))
    alt_solutions = alt_solutions_raw[: int(max(0, mpc_max_alternative_optima))]

    # 1) 速度曲线
    speed_fig = os.path.join(out_dir, "mpc_speed_curve.png")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, pred_vx, linewidth=1.6, label="pred_vx")
    for idx, alt in enumerate(alt_solutions, start=1):
        alt_states = np.asarray(alt.get("states", []), dtype=np.float32)
        if alt_states.ndim == 2 and alt_states.shape[0] >= n_steps + 1 and alt_states.shape[1] > 2:
            alt_vx = alt_states[1 : n_steps + 1, 2]
            ax.plot(t, alt_vx, linewidth=1.2, linestyle="--", alpha=0.9, label=f"pred_vx alt#{idx}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("MPC Theoretical Optimal - Speed")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(speed_fig, dpi=150)
    plt.close(fig)

    # 2) 加速度曲线（物理 + 归一化）
    acc_fig = os.path.join(out_dir, "mpc_acc_curve.png")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(t, acc_phys[:n_steps], linewidth=1.6, label="acc_phys (m/s^2)")
    ax.plot(t, acc_norm[:n_steps], linewidth=1.2, linestyle="--", label="acc_norm")
    for idx, alt in enumerate(alt_solutions, start=1):
        alt_acc_phys = np.asarray(alt.get("acc_phys", []), dtype=np.float32).reshape(-1)
        if alt_acc_phys.size >= n_steps:
            ax.plot(t, alt_acc_phys[:n_steps], linewidth=1.2, linestyle=":", alpha=0.9, label=f"acc_phys alt#{idx}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Acceleration")
    ax.set_title("MPC Theoretical Optimal - Acceleration")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(acc_fig, dpi=150)
    plt.close(fig)

    # 3) 换道曲线（lane_scalar + lane_id）
    lane_fig = os.path.join(out_dir, "mpc_lane_change_curve.png")
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(t, lane_scalar, color="tab:blue", linewidth=1.6, label="lane_scalar")
    alt_lane_id_series: List[np.ndarray] = []
    for idx, alt in enumerate(alt_solutions, start=1):
        alt_actions = np.asarray(alt.get("best_actions_cont", []), dtype=np.float32)
        alt_states = np.asarray(alt.get("states", []), dtype=np.float32)
        if alt_actions.ndim == 2 and alt_actions.shape[0] >= n_steps:
            ax1.plot(
                t,
                alt_actions[:n_steps, 0],
                linewidth=1.2,
                linestyle="--",
                alpha=0.85,
                label=f"lane_scalar alt#{idx}",
            )
        if alt_states.ndim == 2 and alt_states.shape[0] >= n_steps + 1 and alt_states.shape[1] > 1:
            alt_lane_id_series.append(
                np.rint(alt_states[1 : n_steps + 1, 1] / max(lane_width, 1e-6)).astype(np.int32)
            )
    ax1.set_xlabel("Step")
    ax1.set_ylabel("lane_scalar", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.step(t, lane_id, where="mid", color="tab:orange", linewidth=1.4, label="lane_id")
    for idx, alt_lane_id in enumerate(alt_lane_id_series, start=1):
        if alt_lane_id.shape[0] >= n_steps:
            ax2.step(
                t,
                alt_lane_id[:n_steps],
                where="mid",
                linewidth=1.1,
                linestyle=":",
                alpha=0.85,
                label=f"lane_id alt#{idx}",
            )
    ax2.set_ylabel("lane_id", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title("MPC Theoretical Optimal - Lane Change")
    fig.tight_layout()
    fig.savefig(lane_fig, dpi=150)
    plt.close(fig)

    print("MPC low-level 理论最优解结果：")
    print(f"  success         : {summary['success']}")
    print(f"  message         : {summary['message']}")
    print(f"  horizon         : {summary['horizon']}")
    print(f"  steps_to_goal   : {summary['steps_to_goal']}")
    print(f"  low_ext_sum     : {summary['sum_low_ext']:.6f}")
    print(f"  intrinsic_sum   : {summary['sum_intrinsic']:.6f}")
    print(f"  low_total_sum   : {summary['sum_low_total']:.6f}")
    solver_info = dict(summary.get("solver", {}))
    uniqueness_checked = bool(solver_info.get("uniqueness_checked", False))
    alt_found = int(solver_info.get("alternative_optima_found", len(alt_solutions)))
    print(f"  uniqueness_checked : {uniqueness_checked}")
    print(f"  alternative_optima_found : {alt_found}")
    if uniqueness_checked:
        print(f"  has_multiple_optima : {alt_found > 0}")
    else:
        print("  has_multiple_optima : N/A (uniqueness check disabled)")
    print(f"  summary_path    : {summary_path}")
    if trajectory_rows:
        print(f"  trajectory_csv  : {traj_csv}")
    print(f"  speed_curve     : {speed_fig}")
    print(f"  acc_curve       : {acc_fig}")
    print(f"  lane_curve      : {lane_fig}")

    mpc_speed = states[:, 2] if states.ndim == 2 and states.shape[1] > 2 else np.asarray([], dtype=np.float32)
    mpc_acc = acc_phys.copy()
    mpc_lane = lane_scalar.copy()
    neighbors_state = np.asarray(result.get("neighbors_state", []), dtype=np.float32).reshape(-1, 4)
    mpc_actions_cont = np.asarray(result.get("best_actions_cont", []), dtype=np.float32).reshape(-1, 2)
    mpc_alt_curves: List[Dict[str, np.ndarray]] = []
    for alt in alt_solutions:
        alt_states = np.asarray(alt.get("states", []), dtype=np.float32)
        alt_actions = np.asarray(alt.get("best_actions_cont", []), dtype=np.float32)
        alt_acc = np.asarray(alt.get("acc_phys", []), dtype=np.float32).reshape(-1)
        alt_speed = alt_states[:, 2] if alt_states.ndim == 2 and alt_states.shape[1] > 2 else np.asarray([], dtype=np.float32)
        alt_lane = alt_actions[:, 0] if alt_actions.ndim == 2 and alt_actions.shape[1] > 0 else np.asarray([], dtype=np.float32)
        mpc_alt_curves.append(
            {
                "speed": np.asarray(alt_speed, dtype=np.float32).reshape(-1),
                "acc": np.asarray(alt_acc, dtype=np.float32).reshape(-1),
                "lane": np.asarray(alt_lane, dtype=np.float32).reshape(-1),
            }
        )
    return {
        "speed": np.asarray(mpc_speed, dtype=np.float32).reshape(-1),
        "acc": np.asarray(mpc_acc, dtype=np.float32).reshape(-1),
        "lane": np.asarray(mpc_lane, dtype=np.float32).reshape(-1),
        "intrinsic_step": np.asarray(intrinsic_step, dtype=np.float32).reshape(-1),
        "comfort_step": np.asarray(comfort_step, dtype=np.float32).reshape(-1),
        "states": np.asarray(states, dtype=np.float32),
        "actions_cont": np.asarray(mpc_actions_cont, dtype=np.float32),
        "alternative_curves": mpc_alt_curves,
        "neighbors_state": neighbors_state,
        "dt": float(1.0 / float(base_env.config.get("policy_frequency", 10.0))),
    }


def run_uniform_goal_trials(
    env,
    runner: HIROPolicyRunner,
    ego_state: Sequence[float],
    neighbors_state: Sequence[Sequence[float]],
    n_trials: int,
    out_dir: str,
    steps_per_trial: Optional[int] = None,
    metric_fn=None,
    metric_name: str = "intrinsic_reward",
):
    if n_trials <= 0:
        return

    steps_per_trial = int(steps_per_trial or runner.hi)
    metric_fn = metric_fn or default_metric_fn

    high_act_space = build_high_action_space(env, runner.hi)
    sampler = UniformGoalSampler(high_act_space)

    goals_phys: List[np.ndarray] = []
    metrics: List[float] = []

    for i in range(int(n_trials)):
        base_env, _, _ = setup_env_with_state(env, ego_state, neighbors_state)
        obs0 = base_env.observation_type.observe()

        runner.init_from_env(env, obs0, float(getattr(get_hiro_config(), "intrinsic_coef", 1.0)))
        _, kin0, _ = runner._split(obs0)
        ego_sub = runner._ego_sub(kin0)
        runner.ego_start = ego_sub.copy()
        runner.need_high = False
        runner.c = 0

        goal_action = sampler(np.asarray(obs0, dtype=np.float32)[None, :])
        goal_phys = hiro_utils.goal_action_to_abs(ego_sub[None, :], goal_action, runner.lane_center_ys).reshape(-1)
        runner.goal_phys = goal_phys.copy()

        if hasattr(env.unwrapped, "set_hiro_goal"):
            env.unwrapped.set_hiro_goal(runner.goal_phys)

        trial_dir = os.path.join(out_dir, f"trial_{i + 1:03d}")

        obs = obs0
        for step in range(1, steps_per_trial + 1):
            runner.goal_phys = goal_phys.copy()
            runner.need_high = False

            action = runner.act(env, obs)
            obs_next, _, _, _, _ = env.step(action)

            save_low_step_snapshot(env, runner, step, trial_dir, goal_phys)

            runner.c = int((runner.c + 1) % max(runner.hi, 1))
            runner.need_high = False
            obs = obs_next

        metric_val = float(metric_fn(runner, obs))
        goals_phys.append(goal_phys.copy())
        metrics.append(metric_val)

    summary_path = os.path.join(out_dir, "goal_trials_summary.png")
    save_goal_metric_summary(env, goals_phys, metrics, summary_path, metric_name=metric_name)

    # 打印 goal_phys 的 y 分布（0/4/8）
    y_counts = {0.0: 0, 4.0: 0, 8.0: 0}
    for g in goals_phys:
        if g is None or len(g) < 2:
            continue
        y = float(g[1])
        # 允许微小数值误差，按最近值归类
        closest = min(y_counts.keys(), key=lambda v: abs(y - v))
        if abs(y - closest) <= 1e-3:
            y_counts[closest] += 1

    total = sum(y_counts.values())
    print("goal_phys y 分布统计 (0/4/8):")
    print(f"  y=0  : {y_counts[0.0]}")
    print(f"  y=4  : {y_counts[4.0]}")
    print(f"  y=8  : {y_counts[8.0]}")
    print(f"  total: {total}")


def run_mpc_action_sequence_evaluation(
    env,
    ego_state: Sequence[float],
    neighbors_state: Sequence[Sequence[float]],
    goal_phys: Sequence[float],
    action_sequence: Sequence[Sequence[float]],
    out_dir: str,
    steps_to_goal: int,
):
    os.makedirs(out_dir, exist_ok=True)

    base_env, _, _ = setup_env_with_state(env, ego_state, neighbors_state)
    hiro_cfg = get_hiro_config()
    mpc = MPCController(
        base_env,
        horizon=max(1, len(action_sequence)),
        dt=1.0 / float(base_env.config.get("policy_frequency", 10.0)),
        intrinsic_coef=float(getattr(hiro_cfg, "intrinsic_coef", 1.0)),
        intrinsic_type=str(getattr(hiro_cfg, "intrinsic_type", "l2")),
        intrinsic_norm_ranges=getattr(hiro_cfg, "intrinsic_norm_ranges", None),
        intrinsic_weights=getattr(hiro_cfg, "intrinsic_weights", None),
    )

    result = mpc.evaluate_action_sequence(
        actions_cont=action_sequence,
        goal_phys=goal_phys,
        steps_to_goal=int(max(1, steps_to_goal)),
    )

    summary = {
        "success": bool(result.get("success", False)),
        "message": str(result.get("message", "")),
        "horizon": int(result.get("horizon", len(action_sequence))),
        "steps_to_goal": int(result.get("steps_to_goal", steps_to_goal)),
        "intrinsic_type": str(getattr(hiro_cfg, "intrinsic_type", "l2")),
        "progress_objective_enabled": bool(result.get("progress_objective_enabled", True)),
        "approximation_notes": list(result.get("approximation_notes", [])),
        "sequence_check": dict(result.get("sequence_check", {})),
        "sum_low_ext": float(result.get("sum_low_ext", 0.0)),
        "sum_intrinsic": float(result.get("sum_intrinsic", 0.0)),
        "sum_low_total": float(result.get("sum_low_total", 0.0)),
        "reward_components": dict(result.get("reward_components", {})),
        "start_state": np.asarray(result.get("start_state", []), dtype=np.float32).reshape(-1).tolist(),
        "goal_phys": np.asarray(result.get("goal_phys", []), dtype=np.float32).reshape(-1).tolist(),
    }

    summary_path = os.path.join(out_dir, "mpc_action_sequence_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    actions = np.asarray(result.get("best_actions_cont", []), dtype=np.float32)
    states = np.asarray(result.get("states", []), dtype=np.float32)
    acc_norm = np.asarray(result.get("acc_norm", []), dtype=np.float32).reshape(-1)
    acc_phys = np.asarray(result.get("acc_phys", []), dtype=np.float32).reshape(-1)
    intrinsic_step = np.asarray(result.get("intrinsic_step", []), dtype=np.float32).reshape(-1)
    low_ext_step = np.asarray(result.get("low_ext_step", []), dtype=np.float32).reshape(-1)
    low_total_step = np.asarray(result.get("low_total_step", []), dtype=np.float32).reshape(-1)

    trajectory_rows: List[Dict[str, Any]] = []
    n_rows = int(min(actions.shape[0], states.shape[0] - 1)) if states.ndim == 2 else int(actions.shape[0])
    for i in range(n_rows):
        s = states[i + 1] if states.ndim == 2 and i + 1 < states.shape[0] else np.zeros(4, dtype=np.float32)
        a = actions[i] if actions.ndim == 2 and i < actions.shape[0] else np.zeros(2, dtype=np.float32)
        trajectory_rows.append(
            {
                "step": int(i + 1),
                "lane_scalar": float(a[0]),
                "acc_norm": float(a[1]) if a.shape[0] > 1 else (float(acc_norm[i]) if i < acc_norm.shape[0] else 0.0),
                "acc_phys": float(acc_phys[i]) if i < acc_phys.shape[0] else 0.0,
                "pred_x": float(s[0]) if s.shape[0] > 0 else 0.0,
                "pred_y": float(s[1]) if s.shape[0] > 1 else 0.0,
                "pred_vx": float(s[2]) if s.shape[0] > 2 else 0.0,
                "pred_vy": float(s[3]) if s.shape[0] > 3 else 0.0,
                "low_ext_step": float(low_ext_step[i]) if i < low_ext_step.shape[0] else 0.0,
                "intrinsic_step": float(intrinsic_step[i]) if i < intrinsic_step.shape[0] else 0.0,
                "low_total_step": float(low_total_step[i]) if i < low_total_step.shape[0] else 0.0,
            }
        )

    traj_csv = os.path.join(out_dir, "mpc_action_sequence_eval_trajectory.csv")
    if trajectory_rows:
        with open(traj_csv, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(trajectory_rows[0].keys()))
            writer.writeheader()
            writer.writerows(trajectory_rows)

    print("MPC 动作序列评估结果：")
    print(f"  success         : {summary['success']}")
    print(f"  message         : {summary['message']}")
    print(f"  horizon         : {summary['horizon']}")
    print(f"  steps_to_goal   : {summary['steps_to_goal']}")
    print(f"  low_ext_sum     : {summary['sum_low_ext']:.6f}")
    print(f"  intrinsic_sum   : {summary['sum_intrinsic']:.6f}")
    print(f"  low_total_sum   : {summary['sum_low_total']:.6f}")
    print(f"  summary_path    : {summary_path}")
    if trajectory_rows:
        print(f"  trajectory_csv  : {traj_csv}")


def main(
    low_model_path: str,
    steps: int,
    ego_state: Sequence[float],
    neighbors_state: Sequence[Sequence[float]],
    goal_phys: Sequence[float],
    batch_cases_csv: Optional[str] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
    out_dir: str = "./debug/low_level_rollout",
    use_low_safety_layer: Optional[bool] = None,
    seed: int = 0,
    save_initial: bool = True,
    uniform_trials: int = 0,
    uniform_steps: Optional[int] = None,
    uniform_metric_name: str = "intrinsic_reward",
    uniform_metric_fn=None,
    record_interval_csv: bool = False,
    record_interval_index: int = 1,
    run_mpc_optimal: bool = False,
    mpc_horizon: Optional[int] = None,
    mpc_steps_to_goal: Optional[int] = None,
    mpc_mode: str = "qp",
    mpc_global_maxiter: int = 250,
    mpc_plot_alternative_optima: bool = False,
    mpc_max_alternative_optima: int = 3,
    mpc_eval_actions_cont: Optional[Sequence[Sequence[float]]] = None,
    lane_change_min_front_gap: float = 10.0,
    lane_change_min_rear_gap: float = 8.0,
    lane_change_min_front_ttc: float = 3.0,
    lane_change_min_rear_ttc: float = 2.0,
    record_q_sa_curve: bool = True,
    q_sa_a0_min: float = -1.0,
    q_sa_a0_max: float = 1.0,
    q_sa_a1_min: float = -1.0,
    q_sa_a1_max: float = 1.0,
    q_sa_points_per_axis: int = 51,
    random_neighbors_batch_size: int = 0,
    random_neighbors_seed: int = 0,
    _is_subrun: bool = False,
):
    run_out_dir = out_dir
    if not bool(_is_subrun):
        run_tag = datetime.now().strftime("run_%Y%m%d-%H%M%S")
        run_out_dir = os.path.join(out_dir, run_tag)
        os.makedirs(run_out_dir, exist_ok=True)
        print(f"[RUN] output dir: {run_out_dir}")

    if batch_cases_csv:
        cases = load_test_cases_from_csv(batch_cases_csv)
        base_out_dir = os.path.join(run_out_dir, "batch_cases")
        os.makedirs(base_out_dir, exist_ok=True)

        print(f"Loaded {len(cases)} cases from CSV: {batch_cases_csv}")
        for i, case in enumerate(cases, start=1):
            case_id = str(case.get("case_id", "")).strip() or f"case_{i:03d}"
            safe_case_id = case_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
            case_out_dir = os.path.join(base_out_dir, safe_case_id)
            print(f"[{i}/{len(cases)}] Running case: {case_id}")

            main(
                low_model_path=low_model_path,
                steps=steps,
                ego_state=case["ego_state"],
                neighbors_state=case["neighbors_state"],
                goal_phys=case["goal_phys"],
                batch_cases_csv=None,
                env_overrides=env_overrides,
                out_dir=case_out_dir,
                use_low_safety_layer=use_low_safety_layer,
                seed=seed,
                save_initial=save_initial,
                uniform_trials=uniform_trials,
                uniform_steps=uniform_steps,
                uniform_metric_name=uniform_metric_name,
                uniform_metric_fn=uniform_metric_fn,
                run_mpc_optimal=run_mpc_optimal,
                mpc_horizon=mpc_horizon,
                mpc_steps_to_goal=mpc_steps_to_goal,
                mpc_mode=mpc_mode,
                mpc_global_maxiter=mpc_global_maxiter,
                mpc_plot_alternative_optima=mpc_plot_alternative_optima,
                mpc_max_alternative_optima=mpc_max_alternative_optima,
                mpc_eval_actions_cont=mpc_eval_actions_cont,
                lane_change_min_front_gap=lane_change_min_front_gap,
                lane_change_min_rear_gap=lane_change_min_rear_gap,
                lane_change_min_front_ttc=lane_change_min_front_ttc,
                lane_change_min_rear_ttc=lane_change_min_rear_ttc,
                record_q_sa_curve=record_q_sa_curve,
                q_sa_a0_min=q_sa_a0_min,
                q_sa_a0_max=q_sa_a0_max,
                q_sa_a1_min=q_sa_a1_min,
                q_sa_a1_max=q_sa_a1_max,
                q_sa_points_per_axis=q_sa_points_per_axis,
                _is_subrun=True,
            )
        return

    if int(random_neighbors_batch_size) > 0:
        run_batch_random_neighbors_acc_eval(
            low_model_path=low_model_path,
            ego_state=ego_state,
            goal_phys=goal_phys,
            out_dir=run_out_dir,
            n_sets=int(random_neighbors_batch_size),
            steps=int(steps),
            n_neighbors=max(int(len(neighbors_state)), 1),
            seed=int(random_neighbors_seed),
            env_overrides=env_overrides,
            use_low_safety_layer=use_low_safety_layer,
        )
        return

    # 环境配置：禁用背景车流，保持仅有指定车辆
    test_overrides: Dict[str, Any] = {
        "spawn_probability": 0.0,
        "warmup_time": 0.0,
        "warmup_each_episode": True,
        "show_trajectories": True,
        "screen_width": 1800,
        "screen_height": 300,
        "scaling": 3,
        "centering_position": [0.5, 0.5],
        "show_trajectories": True,
        "initial_lane_id": 1,
        "lane_change_min_front_gap": float(lane_change_min_front_gap),
        "lane_change_min_rear_gap": float(lane_change_min_rear_gap),
        "lane_change_min_front_ttc": float(lane_change_min_front_ttc),
        "lane_change_min_rear_ttc": float(lane_change_min_rear_ttc),
    }
    if env_overrides:
        test_overrides.update(env_overrides)

    env_config = get_env_config(test_overrides)
    n_local = int(env_config.get("observation", {}).get("vehicles_count_local", 1))
    if len(neighbors_state) > max(0, n_local - 1):
        raise ValueError(
            f"neighbors_state 数量 ({len(neighbors_state)}) 超过 vehicles_count_local-1 ({max(0, n_local - 1)})。"
        )
    if len(goal_phys) < 4:
        raise ValueError("goal_phys 需要至少包含 [x, y, vx, vy] 四个元素。")
    env = gym.make("multi-lane-custom-v0", render_mode=None, config=env_config)

    obs0, _ = env.reset(seed=seed)
    base_env, ego, neighbors = setup_env_with_state(env, ego_state, neighbors_state)
    obs0 = base_env.observation_type.observe()

    low_model = SAC.load(low_model_path)
    hiro_cfg = get_hiro_config()

    runner = HIROPolicyRunner(
        _DummyHigh(),
        low_model,
        int(getattr(hiro_cfg, "high_interval", 25)),
        use_low_safety_layer=use_low_safety_layer,
    )
    runner.init_from_env(env, obs0, float(getattr(hiro_cfg, "intrinsic_coef", 1.0)))

    goal_phys_arr = np.asarray(goal_phys, dtype=np.float32).reshape(-1)
    runner.goal_phys = goal_phys_arr.copy()

    # 初始化 ego_start，用于可视化范围框
    _, kin0, _ = runner._split(obs0)
    runner.ego_start = runner._ego_sub(kin0).copy()
    runner.need_high = False
    runner.c = 0

    if hasattr(env.unwrapped, "set_hiro_goal"):
        env.unwrapped.set_hiro_goal(runner.goal_phys)

    if save_initial:
        save_low_step_snapshot(env, runner, 0, run_out_dir, goal_phys_arr, title_suffix="init")

    q_sa_dir = os.path.join(run_out_dir, "q_sa_curve")
    q_sa_rows: List[Dict[str, Any]] = []
    q_sa_step_indices: List[int] = []
    q_sa_surface_rows: List[np.ndarray] = []
    q_sa_a0_mesh_ref: Optional[np.ndarray] = None
    q_sa_a1_mesh_ref: Optional[np.ndarray] = None

    def _build_low_obs_for_q(obs_raw: np.ndarray) -> np.ndarray:
        """Build low-level observation exactly as HIROPolicyRunner.act() does."""
        _, kin_local, kin_flat_local = runner._split(obs_raw)
        ego_sub_local = runner._ego_sub(kin_local)
        t_norm_local = np.array([runner.c / float(runner.hi)], dtype=np.float32)
        goal_rel_local = (runner.goal_phys - ego_sub_local).astype(np.float32)
        local_kin_flat_local = np.asarray(kin_flat_local[0, :runner.local_kin_flat_dim], dtype=np.float32).copy()

        if bool(getattr(runner.cfg, "mask_ego_position_in_low_obs", False)):
            if int(runner.feat_dim) > 0 and local_kin_flat_local.shape[0] >= int(runner.feat_dim):
                idx_x_local = int(runner.feature_names.index("x"))
                idx_y_local = int(runner.feature_names.index("y"))
                local_kin_flat_local[idx_x_local] = 0.0
                local_kin_flat_local[idx_y_local] = 0.0

        return np.concatenate([t_norm_local, local_kin_flat_local, goal_rel_local]).astype(np.float32)

    def _record_q_sa_for_step(step_idx: int, low_obs_state: np.ndarray, action_ref: np.ndarray, chosen_action: Optional[np.ndarray] = None):
        nonlocal q_sa_a0_mesh_ref, q_sa_a1_mesh_ref
        if not bool(record_q_sa_curve):
            return

        q_data = evaluate_q_sa_surface(
            low_model=low_model,
            state=low_obs_state,
            action_template=action_ref,
            a0_min=float(q_sa_a0_min),
            a0_max=float(q_sa_a0_max),
            a1_min=float(q_sa_a1_min),
            a1_max=float(q_sa_a1_max),
            n_points_per_axis=int(q_sa_points_per_axis),
        )

        a0_mesh = np.asarray(q_data["a0_mesh"], dtype=np.float32)
        a1_mesh = np.asarray(q_data["a1_mesh"], dtype=np.float32)
        q_surface = np.asarray(q_data["q_min_surface"], dtype=np.float32)

        if q_sa_a0_mesh_ref is None or q_sa_a1_mesh_ref is None:
            q_sa_a0_mesh_ref = a0_mesh.copy()
            q_sa_a1_mesh_ref = a1_mesh.copy()

        q_sa_step_indices.append(int(step_idx))
        q_sa_surface_rows.append(q_surface.copy())

        save_q_sa_surface_plot(
            out_dir=q_sa_dir,
            step=int(step_idx),
            a0_mesh=a0_mesh,
            a1_mesh=a1_mesh,
            q_surface=q_surface,
            selected_action=(np.asarray(chosen_action, dtype=np.float32).reshape(-1) if chosen_action is not None else None),
        )

        chosen_idx_0 = -1
        chosen_idx_1 = -1
        if chosen_action is not None:
            chosen_action_arr = np.asarray(chosen_action, dtype=np.float32).reshape(-1)
            if chosen_action_arr.size >= 2:
                chosen_idx_0 = int(np.argmin(np.abs(a0_mesh[0, :] - float(chosen_action_arr[0]))))
                chosen_idx_1 = int(np.argmin(np.abs(a1_mesh[:, 0] - float(chosen_action_arr[1]))))

        n0 = int(a0_mesh.shape[1])
        n1 = int(a1_mesh.shape[0])
        for i in range(n1):
            for j in range(n0):
                q_sa_rows.append(
                    {
                        "step": int(step_idx),
                        "a0": float(a0_mesh[i, j]),
                        "a1": float(a1_mesh[i, j]),
                        "q_min": float(q_surface[i, j]),
                        "is_chosen_action": int(i == chosen_idx_1 and j == chosen_idx_0),
                    }
                )

    if bool(record_q_sa_curve):
        low_obs_init = _build_low_obs_for_q(np.asarray(obs0, dtype=np.float32))
        action_init, _ = low_model.predict(low_obs_init, deterministic=True)
        action_init_arr = np.asarray(action_init, dtype=np.float32).reshape(-1)
        _record_q_sa_for_step(0, low_obs_init, action_init_arr, chosen_action=action_init_arr)

    if bool(run_mpc_optimal):
        mpc_dir = os.path.join(run_out_dir, "mpc_theoretical_optimal")
        mpc_hi = int(mpc_horizon) if mpc_horizon is not None else int(runner.hi)
        mpc_goal_steps = int(mpc_steps_to_goal) if mpc_steps_to_goal is not None else int(runner.hi)
        mpc_curve_data = run_mpc_theoretical_optimal(
            env=env,
            ego_state=ego_state,
            neighbors_state=neighbors_state,
            goal_phys=goal_phys_arr,
            out_dir=mpc_dir,
            horizon=mpc_hi,
            steps_to_goal=mpc_goal_steps,
            mpc_mode=mpc_mode,
            mpc_global_maxiter=mpc_global_maxiter,
            mpc_plot_alternative_optima=mpc_plot_alternative_optima,
            mpc_max_alternative_optima=mpc_max_alternative_optima,
        )
    else:
        mpc_curve_data = None

    if mpc_eval_actions_cont is not None:
        mpc_dir = os.path.join(run_out_dir, "mpc_theoretical_optimal")
        mpc_goal_steps = int(mpc_steps_to_goal) if mpc_steps_to_goal is not None else int(runner.hi)
        run_mpc_action_sequence_evaluation(
            env=env,
            ego_state=ego_state,
            neighbors_state=neighbors_state,
            goal_phys=goal_phys_arr,
            action_sequence=mpc_eval_actions_cont,
            out_dir=mpc_dir,
            steps_to_goal=mpc_goal_steps,
        )

    if bool(run_mpc_optimal) or (mpc_eval_actions_cont is not None):

        # 重新设置一次环境状态，确保后续 RL rollout 从同一起点开始
        base_env, ego, neighbors = setup_env_with_state(env, ego_state, neighbors_state)
        obs0 = base_env.observation_type.observe()
        runner.init_from_env(env, obs0, float(getattr(get_hiro_config(), "intrinsic_coef", 1.0)))
        runner.goal_phys = goal_phys_arr.copy()
        _, kin0, _ = runner._split(obs0)
        runner.ego_start = runner._ego_sub(kin0).copy()
        runner.need_high = False
        runner.c = 0
        if hasattr(env.unwrapped, "set_hiro_goal"):
            env.unwrapped.set_hiro_goal(runner.goal_phys)
        if save_initial:
            save_low_step_snapshot(env, runner, 0, run_out_dir, goal_phys_arr, title_suffix="init")

    obs = obs0
    trajectory_rows: List[Dict[str, Any]] = []
    interval_len = max(int(runner.hi), 1)
    selected_interval_idx = max(int(record_interval_index), 1)
    selected_start_step = (selected_interval_idx - 1) * interval_len + 1
    selected_end_step = selected_interval_idx * interval_len

    reward_keys_low = [
        "collision_reward",
        "progress_reward",
        "comfort_reward",
        "lane_change_reward",
        "on_road_reward",
        "intrinsic_reward",
    ]
    reward_sums = {k: 0.0 for k in reward_keys_low}

    def _acc_norm_to_phys(acc_norm_val: float) -> float:
        if getattr(runner, "safety_controller", None) is not None:
            return float(runner.safety_controller._acc_norm_to_phys(float(acc_norm_val)))
        acc_min = float(env.unwrapped.config.get("acceleration_range", [-5.0, 5.0])[0])
        acc_max = float(env.unwrapped.config.get("acceleration_range", [-5.0, 5.0])[1])
        if abs(acc_max - acc_min) < 1e-8:
            return float(acc_min)
        return float(acc_min + 0.5 * (float(acc_norm_val) + 1.0) * (acc_max - acc_min))

    rl_speed_curve: List[float] = []
    rl_acc_curve: List[float] = []
    rl_lane_curve: List[float] = []
    rl_safety_speed_curve: List[float] = []
    rl_safety_acc_curve: List[float] = []
    rl_safety_lane_curve: List[float] = []
    rl_safety_intrinsic_curve: List[float] = []
    rl_safety_comfort_curve: List[float] = []
    safety_speed_upper_curve: List[float] = []
    safety_acc_upper_curve: List[float] = []
    safety_lane_upper_curve: List[float] = []

    comfort_weight = float(env.unwrapped.config.get("comfort_reward", 0.0))
    comfort_max_accel = max(float(env.unwrapped.config.get("comfort_max_accel", 3.0)), 1e-6)

    def _comfort_reward_from_acc(acc_phys_val: float) -> float:
        comfort_base = -(min(abs(float(acc_phys_val)) / comfort_max_accel, 1.0) ** 2) * float(runner.dt)
        return float(comfort_base * comfort_weight)

    _, kin_init, _ = runner._split(obs)
    ego_init = runner._ego_sub(kin_init)
    init_speed = float(ego_init[2]) if ego_init.shape[0] > 2 else 0.0
    rl_speed_curve.append(init_speed)
    rl_safety_speed_curve.append(init_speed)

    for step in range(1, int(steps) + 1):
        runner.goal_phys = goal_phys_arr.copy()
        runner.need_high = False

        state_now = np.asarray(obs, dtype=np.float32).reshape(-1)
        low_obs_now = _build_low_obs_for_q(obs)
        _, kin_now, _ = runner._split(obs)
        ego_abs_now, others_rel_now = runner._extract_ego_others(kin_now)
        action = runner.act(env, obs)

        action_pre = np.asarray(getattr(runner, "last_action_pre_safety", action), dtype=np.float32).reshape(-1)
        action_post = np.asarray(getattr(runner, "last_action_post_safety", action), dtype=np.float32).reshape(-1)
        lane_rl = float(action_pre[0]) if action_pre.shape[0] > 0 else 0.0
        acc_norm_rl = float(action_pre[1]) if action_pre.shape[0] > 1 else 0.0

        _record_q_sa_for_step(int(step), low_obs_now, action_pre, chosen_action=action_pre)

        rl_lane_curve.append(lane_rl)
        acc_phys_rl = _acc_norm_to_phys(acc_norm_rl)
        rl_acc_curve.append(acc_phys_rl)

        lane_rl_safety = float(action_post[0]) if action_post.shape[0] > 0 else lane_rl
        acc_norm_rl_safety = float(action_post[1]) if action_post.shape[0] > 1 else acc_norm_rl
        acc_phys_rl_safety = _acc_norm_to_phys(acc_norm_rl_safety)
        rl_safety_lane_curve.append(lane_rl_safety)
        rl_safety_acc_curve.append(acc_phys_rl_safety)

        if getattr(runner, "safety_controller", None) is not None:
            safety_upper_in = np.array([lane_rl, 1.0], dtype=np.float32)
            safety_upper = np.asarray(
                runner.safety_controller.safety_filter_action(
                    ego_abs_now,
                    others_rel_now,
                    runner.goal_phys,
                    safety_upper_in,
                    runner.dt,
                    remaining_time=float(runner.hi - runner.c) * float(runner.dt),
                ),
                dtype=np.float32,
            ).reshape(-1)
            lane_safety_upper = float(safety_upper[0]) if safety_upper.shape[0] > 0 else 0.0
            acc_norm_safety_upper = float(safety_upper[1]) if safety_upper.shape[0] > 1 else 0.0
        else:
            lane_safety_upper = lane_rl
            acc_norm_safety_upper = 1.0

        safety_lane_upper_curve.append(lane_safety_upper)
        acc_phys_safety_upper = _acc_norm_to_phys(acc_norm_safety_upper)
        safety_acc_upper_curve.append(acc_phys_safety_upper)

        vx_now = float(ego_abs_now[2]) if ego_abs_now.shape[0] > 2 else 0.0
        rl_speed_curve.append(vx_now + acc_phys_rl * float(runner.dt))
        safety_speed_upper_curve.append(vx_now + acc_phys_safety_upper * float(runner.dt))

        obs_next, reward, terminated, truncated, info = env.step(action)

        _, kin_next, _ = runner._split(obs_next)
        ego_next = runner._ego_sub(kin_next)
        rl_safety_speed_curve.append(float(ego_next[2]) if ego_next.shape[0] > 2 else 0.0)

        rc = info.get("reward_components", {}) if isinstance(info, dict) else {}
        punctual = float(rc.get("punctual_reward", 0.0))
        low_ext = float(reward) - punctual
        for k in reward_keys_low:
            if k == "intrinsic_reward":
                continue
            reward_sums[k] += float(rc.get(k, 0.0))

        last_step = bool(runner.c == runner.hi - 1)
        intrinsic = runner.intrinsic_if_last(obs_next) if last_step else 0.0
        rl_safety_intrinsic_curve.append(float(intrinsic))
        rl_safety_comfort_curve.append(_comfort_reward_from_acc(acc_phys_rl_safety))
        reward_sums["intrinsic_reward"] += float(intrinsic)

        if record_interval_csv and selected_start_step <= int(step) <= selected_end_step:
            action_before_safety = np.asarray(getattr(runner, "last_action_pre_safety", action), dtype=np.float32).reshape(-1)
            action_after_safety = np.asarray(getattr(runner, "last_action_post_safety", action), dtype=np.float32).reshape(-1)
            row: Dict[str, Any] = {
                "step": int(step),
                "interval_index": int(selected_interval_idx),
                "step_in_interval": int(((step - 1) % interval_len) + 1),
                "done": int(bool(terminated or truncated)),
                "terminated": int(terminated),
                "truncated": int(truncated),
                "reward": float(reward),
                "punctual_reward": float(punctual),
                "low_ext_reward": float(low_ext),
                "intrinsic_reward": float(intrinsic),
                "low_total_step_reward": float(low_ext + intrinsic),
            }
            for i, v in enumerate(state_now):
                row[f"state_{i}"] = float(v)
            for i, v in enumerate(action_before_safety):
                row[f"action_pre_safety_{i}"] = float(v)
            for i, v in enumerate(action_after_safety):
                row[f"action_post_safety_{i}"] = float(v)
            trajectory_rows.append(row)

        save_low_step_snapshot(env, runner, step, run_out_dir, goal_phys_arr, reward_sums=reward_sums)

        runner.c = int((runner.c + 1) % max(runner.hi, 1))
        runner.need_high = False

        obs = obs_next

    # 保存三组对比曲线：RL 输出 / Safety Layer 上界 / MPC 最优
    mpc_speed_curve = np.asarray([], dtype=np.float32)
    mpc_acc_curve = np.asarray([], dtype=np.float32)
    mpc_lane_curve = np.asarray([], dtype=np.float32)
    mpc_speed_safety_upper_curve = np.asarray([], dtype=np.float32)
    mpc_acc_safety_upper_curve = np.asarray([], dtype=np.float32)
    mpc_lane_safety_upper_curve = np.asarray([], dtype=np.float32)
    mpc_intrinsic_curve = np.asarray([], dtype=np.float32)
    mpc_comfort_curve = np.asarray([], dtype=np.float32)
    mpc_alt_speed_curves: List[np.ndarray] = []
    mpc_alt_acc_curves: List[np.ndarray] = []
    mpc_alt_lane_curves: List[np.ndarray] = []
    if isinstance(mpc_curve_data, dict):
        mpc_speed_curve = np.asarray(mpc_curve_data.get("speed", []), dtype=np.float32).reshape(-1)
        mpc_acc_curve = np.asarray(mpc_curve_data.get("acc", []), dtype=np.float32).reshape(-1)
        mpc_lane_curve = np.asarray(mpc_curve_data.get("lane", []), dtype=np.float32).reshape(-1)
        mpc_intrinsic_curve = np.asarray(mpc_curve_data.get("intrinsic_step", []), dtype=np.float32).reshape(-1)
        mpc_comfort_curve = np.asarray(mpc_curve_data.get("comfort_step", []), dtype=np.float32).reshape(-1)
        for alt_curve in list(mpc_curve_data.get("alternative_curves", [])):
            mpc_alt_speed_curves.append(np.asarray(alt_curve.get("speed", []), dtype=np.float32).reshape(-1))
            mpc_alt_acc_curves.append(np.asarray(alt_curve.get("acc", []), dtype=np.float32).reshape(-1))
            mpc_alt_lane_curves.append(np.asarray(alt_curve.get("lane", []), dtype=np.float32).reshape(-1))

        mpc_states = np.asarray(mpc_curve_data.get("states", []), dtype=np.float32)
        mpc_actions = np.asarray(mpc_curve_data.get("actions_cont", []), dtype=np.float32)
        mpc_neighbors0 = np.asarray(mpc_curve_data.get("neighbors_state", []), dtype=np.float32).reshape(-1, 4)
        mpc_dt = float(mpc_curve_data.get("dt", runner.dt))

        if (
            getattr(runner, "safety_controller", None) is not None
            and mpc_states.ndim == 2
            and mpc_states.shape[0] >= 2
            and mpc_actions.ndim == 2
            and mpc_actions.shape[0] >= 1
        ):
            mpc_acc_upper_list: List[float] = []
            mpc_lane_upper_list: List[float] = []
            mpc_speed_upper_list: List[float] = []

            n_mpc = int(min(mpc_actions.shape[0], mpc_states.shape[0] - 1))
            for t in range(n_mpc):
                ego_abs_t = np.asarray(mpc_states[t, :4], dtype=np.float32)

                others_rel_rows: List[List[float]] = []
                for j in range(int(mpc_neighbors0.shape[0])):
                    nx0, ny0, nvx0, nvy0 = [float(v) for v in mpc_neighbors0[j]]
                    nx_t = nx0 + nvx0 * mpc_dt * float(t)
                    ny_t = ny0 + nvy0 * mpc_dt * float(t)
                    rel_dx = nx_t - float(ego_abs_t[0])
                    rel_dy = ny_t - float(ego_abs_t[1])
                    rel_dvx = nvx0 - float(ego_abs_t[2])
                    rel_dvy = nvy0 - float(ego_abs_t[3])
                    others_rel_rows.append([rel_dx, rel_dy, rel_dvx, rel_dvy])
                others_rel_t = np.asarray(others_rel_rows, dtype=np.float32).reshape(-1, 4)

                lane_mpc_t = float(mpc_actions[t, 0]) if mpc_actions.shape[1] > 0 else 0.0
                safety_upper_in_mpc = np.array([lane_mpc_t, 1.0], dtype=np.float32)
                safety_upper_mpc = np.asarray(
                    runner.safety_controller.safety_filter_action(
                        ego_abs_t,
                        others_rel_t,
                        goal_phys_arr,
                        safety_upper_in_mpc,
                        mpc_dt,
                        remaining_time=max(float(n_mpc - t), 1.0) * mpc_dt,
                    ),
                    dtype=np.float32,
                ).reshape(-1)

                lane_upper_t = float(safety_upper_mpc[0]) if safety_upper_mpc.shape[0] > 0 else 0.0
                acc_norm_upper_t = float(safety_upper_mpc[1]) if safety_upper_mpc.shape[0] > 1 else 0.0
                acc_phys_upper_t = _acc_norm_to_phys(acc_norm_upper_t)

                mpc_lane_upper_list.append(lane_upper_t)
                mpc_acc_upper_list.append(acc_phys_upper_t)
                mpc_speed_upper_list.append(float(ego_abs_t[2]) + acc_phys_upper_t * mpc_dt)

            mpc_acc_safety_upper_curve = np.asarray(mpc_acc_upper_list, dtype=np.float32)
            mpc_lane_safety_upper_curve = np.asarray(mpc_lane_upper_list, dtype=np.float32)
            if mpc_states.shape[0] > 0:
                mpc_speed0 = float(mpc_states[0, 2])
                mpc_speed_safety_upper_curve = np.asarray([mpc_speed0] + mpc_speed_upper_list, dtype=np.float32)

    save_speed_acc_curves(
        env,
        ep_idx=1,
        model_path=run_out_dir,
        comparison_data={
            "speed_rl": np.asarray(rl_speed_curve, dtype=np.float32),
            "speed_rl_safety_output": np.asarray(rl_safety_speed_curve, dtype=np.float32),
            "speed_safety_upper_rl": np.asarray([rl_speed_curve[0]] + safety_speed_upper_curve, dtype=np.float32) if rl_speed_curve else np.asarray([], dtype=np.float32),
            "speed_mpc": mpc_speed_curve,
            "speed_safety_upper_mpc": mpc_speed_safety_upper_curve,
            "acc_rl": np.asarray(rl_acc_curve, dtype=np.float32),
            "acc_rl_safety_output": np.asarray(rl_safety_acc_curve, dtype=np.float32),
            "acc_safety_upper_rl": np.asarray(safety_acc_upper_curve, dtype=np.float32),
            "acc_mpc": mpc_acc_curve,
            "acc_safety_upper_mpc": mpc_acc_safety_upper_curve,
            "lane_rl": np.asarray(rl_lane_curve, dtype=np.float32),
            "lane_rl_safety_output": np.asarray(rl_safety_lane_curve, dtype=np.float32),
            "lane_safety_upper_rl": np.asarray(safety_lane_upper_curve, dtype=np.float32),
            "lane_mpc": mpc_lane_curve,
            "lane_safety_upper_mpc": mpc_lane_safety_upper_curve,
            "intrinsic_rl_safety_output": np.asarray(rl_safety_intrinsic_curve, dtype=np.float32),
            "comfort_rl_safety_output": np.asarray(rl_safety_comfort_curve, dtype=np.float32),
            "intrinsic_mpc": mpc_intrinsic_curve,
            "comfort_mpc": mpc_comfort_curve,
            "speed_mpc_alternatives": mpc_alt_speed_curves,
            "acc_mpc_alternatives": mpc_alt_acc_curves,
            "lane_mpc_alternatives": mpc_alt_lane_curves,
        },
    )

    if int(uniform_trials) > 0:
        trials_dir = os.path.join(run_out_dir, "uniform_goal_trials")
        run_uniform_goal_trials(
            env,
            runner,
            ego_state,
            neighbors_state,
            n_trials=int(uniform_trials),
            out_dir=trials_dir,
            steps_per_trial=uniform_steps or runner.hi,
            metric_fn=uniform_metric_fn,
            metric_name=uniform_metric_name,
        )

    if record_interval_csv:
        csv_path = os.path.join(run_out_dir, f"low_interval_{selected_interval_idx:03d}_trajectory.csv")
        if trajectory_rows:
            with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(trajectory_rows[0].keys()))
                writer.writeheader()
                writer.writerows(trajectory_rows)
            print(f"Saved interval trajectory csv: {csv_path}")
        else:
            print(
                "Saved interval trajectory csv: skipped "
                f"(interval={selected_interval_idx}, steps={steps}, hi={interval_len})"
            )

    if bool(record_q_sa_curve):
        os.makedirs(q_sa_dir, exist_ok=True)

        q_csv_path = os.path.join(q_sa_dir, "q_sa_surface_all_steps.csv")
        if q_sa_rows:
            with open(q_csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(q_sa_rows[0].keys()))
                writer.writeheader()
                writer.writerows(q_sa_rows)
            print(f"Saved Q(s,a0,a1) csv: {q_csv_path}")

        if q_sa_a0_mesh_ref is not None and q_sa_a1_mesh_ref is not None and q_sa_surface_rows:
            q_stack = np.stack([np.asarray(s, dtype=np.float32) for s in q_sa_surface_rows], axis=0)
            q_surface_mean = np.mean(q_stack, axis=0)
            q_surface_std = np.std(q_stack, axis=0)
            save_q_sa_global_summary(
                q_sa_dir,
                np.asarray(q_sa_a0_mesh_ref, dtype=np.float32),
                np.asarray(q_sa_a1_mesh_ref, dtype=np.float32),
                np.asarray(q_surface_mean, dtype=np.float32),
                np.asarray(q_surface_std, dtype=np.float32),
            )
            print(f"Saved Q(s,a0,a1) surfaces to: {q_sa_dir}")

    env.close()
    print(f"Saved {steps + (1 if save_initial else 0)} frames to: {run_out_dir}")


if __name__ == "__main__":
    # 用法示例：
    # 1) 指定低层模型路径
    # 2) 指定 ego + 观测车辆初始状态（x, y, vx, vy）
    # 3) 指定 goal_phys（x, y, vx, vy）
    # 4) 若需要批量测试，可传 batch_cases_csv（列支持：
    #    - ego_state / neighbors_state / goal_phys 三列，值为列表字符串；或
    #    - ego_x,ego_y,ego_vx,ego_vy + goal_x,goal_y,goal_vx,goal_vy + neighbors_state(可选)
    #    - 可选 case_id 列作为输出目录名）

    main(
        low_model_path="./models/hiro_260311_lowonly_uniform_RS_newSLv2_vioPenalty03/hiro_low_final.zip",
        # low_model_path="./models/hiro_260321_lowonly_reachableUniform_newSLv2_vio03_HER_reDim_amax3_dmin0/hiro_low_final.zip",
        # low_model_path="./models/hiro_260318_lowonly_uniform_RS_newSLv2_vio03_HER_reDim_v2/hiro_low_final.zip",
        steps=25,
        ego_state=[0.0, 4.0, 10.0, 0.0],
        neighbors_state=[
            [30.0, 4.0, 10.0, 0.0],
            [60.0, 8.0, 12.0, 0.0],
            [30.0, 0.0, 10.0, 0.0],
            [5.0, 8.0, 15.0, 0.0],
        ],
        goal_phys=[30, 4, 10, 0],
        # ego_state=[0.000000, 8.000000, 12.320744, 0],
        # neighbors_state=[
        #     [33.611660, 0, 12.699497, 0.000000],
        #     [55.454609, 4, 12.963794, 0.000000],
        #     [73.182320, 8, 12.303981, 0.000000],
        #     [85.691025, 0, 13.218258, 0.000000],
        # ],
        # goal_phys=[31.408527, 8, 0, 0.000000],
        # batch_cases_csv="low_test_cases.csv",
        # batch_cases_csv="low_test_cases_debug.csv",
        use_low_safety_layer=True,
        out_dir="./models/debug/low_level_rollout",
        uniform_trials=0,
        record_interval_csv=True,
        record_interval_index=1,
        run_mpc_optimal=True,
        # run_mpc_optimal=False,
        # mpc_mode="qp",
        mpc_mode="joint_global",
        mpc_horizon=25,
        mpc_steps_to_goal=25,
        mpc_global_maxiter=200,  # joint_global 模式才会使用
        mpc_plot_alternative_optima=False,
        mpc_max_alternative_optima=3,
        random_neighbors_batch_size=0,
        random_neighbors_seed=42,
        # mpc_eval_actions_cont=[
        #     [0.0, 0.0],
        #     [0.0, 0.0],
        #     [0.0, 0.0],
        #     [0.0, 0.0],
        #     [0.0, 0.0],
        #     [0.0, 0.0],
        #     [0.0, 0.0],
        #     [0.0, 0.0],
        #     [0.0, 0.0],
        #     [0.0, 0.0],
        #     [0.0, 0.0],
        #     [1.0, 0.0],
        #     [1.0, 0.0],
        #     [1.0, 0.0],
        #     [1.0, 0.0],
        #     [1.0, 0.0],
        #     [1.0, 0.0],
        #     [0.0, 0.0],
        #     [0.0, 0.0],
        #     [0.0, 0.0],
        #     [0.0, 0.0],
        #     [0.0, 0.0],
        #     [0.0, 0.0],
        #     [0.0, 0.0],
        #     [0.0, 0.0]
        # ],
        lane_change_min_front_gap=15.0,
        lane_change_min_rear_gap=10.0,
        lane_change_min_front_ttc=3.0,
        lane_change_min_rear_ttc=2.0,
        # uniform_trials=1000,
        # uniform_metric_name="abs_dy",
        # uniform_metric_fn=_abs_dy_metric_fn,
    )
