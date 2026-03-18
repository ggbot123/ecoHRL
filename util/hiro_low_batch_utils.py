import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from configs.conf import get_env_config, get_hiro_config
from rl.algos.HRL.hiro_infer import HIROPolicyRunner
from rl.algos.sac.sac import SAC
from util.hiro_low_test_utils import setup_env_with_state


class _BatchDummyHigh:
    """Placeholder high-level policy for low-level-only rollout in batch eval."""

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        return np.zeros((1, 3), dtype=np.float32), None


def is_neighbor_set_reasonable(
    neighbors: List[np.ndarray],
    min_same_lane_gap: float = 8.0,
    min_ttc: float = 2.0,
) -> bool:
    """Basic plausibility checks on generated neighbors in each lane."""
    if not neighbors:
        return True

    arr = np.asarray(neighbors, dtype=np.float32).reshape(-1, 4)
    lane_vals = sorted({float(v) for v in arr[:, 1]})
    for lane_y in lane_vals:
        lane_mask = np.isclose(arr[:, 1], lane_y)
        lane_cars = arr[lane_mask]
        if lane_cars.shape[0] <= 1:
            continue

        order = np.argsort(lane_cars[:, 0])
        lane_sorted = lane_cars[order]
        for i in range(lane_sorted.shape[0] - 1):
            rear = lane_sorted[i]
            front = lane_sorted[i + 1]
            dx = float(front[0] - rear[0])
            if dx < float(min_same_lane_gap):
                return False

            dv = float(rear[2] - front[2])
            if dv > 1e-6:
                ttc = dx / dv
                if ttc < float(min_ttc):
                    return False
    return True


def generate_random_neighbors_state_sets(
    ego_state: Sequence[float],
    n_sets: int,
    n_neighbors: int,
    seed: int,
    y_candidates: Sequence[float] = (0.0, 8.0),
    vx_min: float = 8.0,
    vx_max: float = 15.0,
    front_gap_range: Sequence[float] = (12.0, 70.0),
    rear_gap_range: Sequence[float] = (10.0, 40.0),
    min_same_lane_gap: float = 8.0,
) -> List[List[List[float]]]:
    """Generate neighbor sets with lane/speed constraints and simple plausibility checks."""
    rng = np.random.default_rng(int(seed))
    ego_x = float(ego_state[0])

    all_sets: List[List[List[float]]] = []
    max_trials_per_set = 400

    for _ in range(int(n_sets)):
        accepted = None
        for _trial in range(max_trials_per_set):
            neighbors: List[np.ndarray] = []
            for _k in range(int(n_neighbors)):
                y = float(y_candidates[int(rng.integers(0, len(y_candidates)))])
                is_front = bool(rng.random() < 0.65)

                if is_front:
                    dx = float(rng.uniform(float(front_gap_range[0]), float(front_gap_range[1])))
                else:
                    dx = -float(rng.uniform(float(rear_gap_range[0]), float(rear_gap_range[1])))

                vx = float(rng.uniform(float(vx_min), float(vx_max)))
                neighbors.append(np.array([ego_x + dx, y, vx, 0.0], dtype=np.float32))

            if is_neighbor_set_reasonable(
                neighbors,
                min_same_lane_gap=float(min_same_lane_gap),
                min_ttc=2.0,
            ):
                accepted = neighbors
                break

        if accepted is None:
            accepted = neighbors

        all_sets.append([np.asarray(v, dtype=np.float32).reshape(-1).tolist() for v in accepted])

    return all_sets


def run_batch_random_neighbors_acc_eval(
    low_model_path: str,
    ego_state: Sequence[float],
    goal_phys: Sequence[float],
    out_dir: str,
    n_sets: int = 100,
    steps: int = 25,
    n_neighbors: int = 4,
    seed: int = 0,
    env_overrides: Optional[Dict[str, Any]] = None,
    use_low_safety_layer: Optional[bool] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    test_overrides: Dict[str, Any] = {
        "spawn_probability": 0.0,
        "warmup_time": 0.0,
        "warmup_each_episode": True,
        "show_trajectories": False,
        "screen_width": 1800,
        "screen_height": 300,
        "scaling": 3,
        "centering_position": [0.5, 0.5],
        "initial_lane_id": 1,
    }
    if env_overrides:
        test_overrides.update(env_overrides)

    env_config = get_env_config(test_overrides)
    env = gym.make("multi-lane-custom-v0", render_mode=None, config=env_config)

    low_model = SAC.load(low_model_path)
    hiro_cfg = get_hiro_config()

    neighbor_sets = generate_random_neighbors_state_sets(
        ego_state=ego_state,
        n_sets=int(n_sets),
        n_neighbors=int(n_neighbors),
        seed=int(seed),
    )

    all_acc_curves: List[np.ndarray] = []

    def _acc_norm_to_phys_local(runner_local: HIROPolicyRunner, env_local, acc_norm_val: float) -> float:
        if getattr(runner_local, "safety_controller", None) is not None:
            return float(runner_local.safety_controller._acc_norm_to_phys(float(acc_norm_val)))
        acc_min = float(env_local.unwrapped.config.get("acceleration_range", [-5.0, 5.0])[0])
        acc_max = float(env_local.unwrapped.config.get("acceleration_range", [-5.0, 5.0])[1])
        if abs(acc_max - acc_min) < 1e-8:
            return float(acc_min)
        return float(acc_min + 0.5 * (float(acc_norm_val) + 1.0) * (acc_max - acc_min))

    for i, neighbors_state in enumerate(neighbor_sets, start=1):
        env.reset(seed=int(seed) + i)
        base_env, _, _ = setup_env_with_state(env, ego_state, neighbors_state)
        obs = base_env.observation_type.observe()

        runner = HIROPolicyRunner(
            _BatchDummyHigh(),
            low_model,
            int(getattr(hiro_cfg, "high_interval", 25)),
            use_low_safety_layer=use_low_safety_layer,
        )
        runner.init_from_env(env, obs, float(getattr(hiro_cfg, "intrinsic_coef", 1.0)))
        runner.goal_phys = np.asarray(goal_phys, dtype=np.float32).reshape(-1)
        _, kin0, _ = runner._split(obs)
        runner.ego_start = runner._ego_sub(kin0).copy()
        runner.need_high = False
        runner.c = 0

        if hasattr(env.unwrapped, "set_hiro_goal"):
            env.unwrapped.set_hiro_goal(runner.goal_phys)

        acc_curve: List[float] = []
        for _step in range(int(steps)):
            runner.goal_phys = np.asarray(goal_phys, dtype=np.float32).reshape(-1)
            runner.need_high = False

            action = runner.act(env, obs)
            action_post = np.asarray(getattr(runner, "last_action_post_safety", action), dtype=np.float32).reshape(-1)
            acc_norm = float(action_post[1]) if action_post.shape[0] > 1 else 0.0
            acc_curve.append(_acc_norm_to_phys_local(runner, env, acc_norm))

            obs_next, _, terminated, truncated, _ = env.step(action)
            runner.c = int((runner.c + 1) % max(runner.hi, 1))
            runner.need_high = False
            obs = obs_next

            if bool(terminated or truncated):
                break

        arr = np.asarray(acc_curve, dtype=np.float32)
        if arr.shape[0] < int(steps):
            pad = np.full((int(steps) - arr.shape[0],), np.nan, dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        all_acc_curves.append(arr)

    neighbors_csv_path = os.path.join(out_dir, "random_neighbors_100_sets.csv")
    with open(neighbors_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["set_id", "neighbors_state"])
        writer.writeheader()
        for idx, ns in enumerate(neighbor_sets, start=1):
            writer.writerow({"set_id": idx, "neighbors_state": json.dumps(ns, ensure_ascii=False)})

    acc_plot_path = os.path.join(out_dir, "rl_safety_acc_curves_100_sets.png")
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(1, int(steps) + 1, dtype=np.int32)
    for curve in all_acc_curves:
        ax.plot(x, curve, color="tab:blue", alpha=0.20, linewidth=1.0)

    if all_acc_curves:
        stack = np.stack(all_acc_curves, axis=0)
        mean_curve = np.nanmean(stack, axis=0)
        ax.plot(x, mean_curve, color="tab:red", linewidth=2.0, label="mean")
        ax.legend(loc="best")

    ax.set_xlabel("Step")
    ax.set_ylabel("Acceleration (m/s^2)")
    ax.set_title("RL + Safety Output Acc Curves (100 Neighbor Sets)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(acc_plot_path, dpi=160)
    plt.close(fig)

    env.close()
    print(f"Saved random neighbors csv: {neighbors_csv_path}")
    print(f"Saved acceleration curves : {acc_plot_path}")
