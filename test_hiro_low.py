import os
import csv
from typing import Any, Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np

import scenarios.multi_lane  # 触发 __init__.py 里的 register

from configs.conf import get_env_config, get_hiro_config
from rl.algos.sac.sac import SAC
from rl.algos.HRL.hiro_infer import HIROPolicyRunner
from rl.algos.HRL.goal_samplers import UniformGoalSampler
from rl.utils import utils as hiro_utils
from util.plot_result import save_speed_acc_curves, save_low_step_snapshot, save_goal_metric_summary
from util.hiro_low_test_utils import (
    setup_env_with_state,
    build_high_action_space,
    default_metric_fn,
    abs_dx_metric_fn,
    abs_dy_metric_fn,
    load_test_cases_from_csv,
)


class _DummyHigh:
    """占位高层模型，避免低层测试时强制依赖高层模型。"""

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        # HIRO high action 维度通常为 3：[dx, y_code, vx]
        return np.zeros((1, 3), dtype=np.float32), None


_abs_dx_metric_fn = abs_dx_metric_fn
_abs_dy_metric_fn = abs_dy_metric_fn


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


def main(
    low_model_path: str,
    steps: int,
    ego_state: Sequence[float],
    neighbors_state: Sequence[Sequence[float]],
    goal_phys: Sequence[float],
    batch_cases_csv: Optional[str] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
    out_dir: str = "./debug/low_level_rollout",
    batch_out_dir: Optional[str] = None,
    use_low_safety_layer: Optional[bool] = None,
    seed: int = 0,
    save_initial: bool = True,
    uniform_trials: int = 0,
    uniform_steps: Optional[int] = None,
    uniform_out_dir: Optional[str] = None,
    uniform_metric_name: str = "intrinsic_reward",
    uniform_metric_fn=None,
    record_interval_csv: bool = False,
    record_interval_index: int = 1,
):
    if batch_cases_csv:
        cases = load_test_cases_from_csv(batch_cases_csv)
        base_out_dir = batch_out_dir or os.path.join(out_dir, "batch_cases")
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
                batch_out_dir=None,
                use_low_safety_layer=use_low_safety_layer,
                seed=seed,
                save_initial=save_initial,
                uniform_trials=uniform_trials,
                uniform_steps=uniform_steps,
                uniform_out_dir=None,
                uniform_metric_name=uniform_metric_name,
                uniform_metric_fn=uniform_metric_fn,
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
        save_low_step_snapshot(env, runner, 0, out_dir, goal_phys_arr, title_suffix="init")

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
    for step in range(1, int(steps) + 1):
        runner.goal_phys = goal_phys_arr.copy()
        runner.need_high = False

        state_now = np.asarray(obs, dtype=np.float32).reshape(-1)
        action = runner.act(env, obs)
        obs_next, reward, terminated, truncated, info = env.step(action)

        rc = info.get("reward_components", {}) if isinstance(info, dict) else {}
        punctual = float(rc.get("punctual_reward", 0.0))
        low_ext = float(reward) - punctual
        for k in reward_keys_low:
            if k == "intrinsic_reward":
                continue
            reward_sums[k] += float(rc.get(k, 0.0))

        last_step = bool(runner.c == runner.hi - 1)
        intrinsic = runner.intrinsic_if_last(obs_next) if last_step else 0.0
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

        save_low_step_snapshot(env, runner, step, out_dir, goal_phys_arr, reward_sums=reward_sums)

        runner.c = int((runner.c + 1) % max(runner.hi, 1))
        runner.need_high = False

        obs = obs_next

    # 保存速度曲线/加速度曲线
    save_speed_acc_curves(env, ep_idx=1, model_path=out_dir)

    if int(uniform_trials) > 0:
        trials_dir = uniform_out_dir or os.path.join(out_dir, "uniform_goal_trials")
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
        csv_path = os.path.join(out_dir, f"low_interval_{selected_interval_idx:03d}_trajectory.csv")
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

    env.close()
    print(f"Saved {steps + (1 if save_initial else 0)} frames to: {out_dir}")


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
        # low_model_path="./models/hiro_260122_onlyLow_uniform_safetyLayer_rewShaping/hiro_low_final.zip",
        low_model_path="./models/hiro_260226_lowonly_uniform_SL_RS_newIDM/hiro_low_final.zip",
        steps=25,
        ego_state=[0.0, 4.0, 10.0, 0.0],
        neighbors_state=[
            [30.0, 4.0, 10.0, 0.0],
            [60.0, 8.0, 12.0, 0.0],
            [30.0, 0.0, 10.0, 0.0],
            [10.0, 8.0, 15.0, 0.0],
        ],
        goal_phys=[25, 8.0, 12.0, 0.0],
        # batch_cases_csv="./configs/low_test_cases.csv",
        use_low_safety_layer=True,
        out_dir="./models/debug/low_level_rollout",
        uniform_trials=0,
        record_interval_csv=True,
        record_interval_index=1,
        # uniform_trials=1000,
        # uniform_out_dir="./models/debug/low_level_rollout/0225/uniform_goal_trials",
        # uniform_metric_name="abs_dy",
        # uniform_metric_fn=_abs_dy_metric_fn,
    )
