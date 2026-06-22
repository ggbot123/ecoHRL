import os
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple

import cvxpy as cp
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
import importlib

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable

from configs.builders import get_env_config_for_scenario, get_scenario_spec
from custom_env.vehicle.behavior import IDMVehicle
from rl.algos.HRL.rule_based import RuleBasedController
from util.hiro_utils import unique_path
from util.plot_result import save_speed_acc_curves


class RuleMPCController:
    """Rule + MPC baseline.

    1) Lateral lane target is always RIGHT unless ego is already on the rightmost lane.
    2) Longitudinal motion is obtained by solving a convex QP over acceleration sequence.
       Objective: sum(a^2) + sum((v-v_ref)^2)
       Constraints: min distance & TTC to nearby vehicles (front/rear), speed and accel bounds.
         If lane-change-target MPC is infeasible, fallback target lane is current lane.
    """

    def __init__(
        self,
        env,
        horizon: int = 15,
        dt: float = 0.1,
        w_acc: float = 1.0,
        w_speed: float = 2.0,
        lane_change_settle_steps: int = 4,
        lane_strategy: str = "right_bias",
    ):
        self.env = env
        self.horizon = int(max(2, horizon))
        self.dt = float(max(dt, 1e-3))
        self.w_acc = float(max(w_acc, 1e-8))
        self.w_speed = float(max(w_speed, 1e-8))
        self.lane_change_settle_steps = int(max(1, lane_change_settle_steps))
        lane_strategy_norm = str(lane_strategy).strip().lower()
        if lane_strategy_norm not in {"right_bias", "mobil"}:
            raise ValueError(f"Unsupported lane_strategy: {lane_strategy}")
        self.lane_strategy = lane_strategy_norm

        cfg = getattr(self.env, "config", {})
        action_cfg = cfg.get("action", {})
        acc_range = action_cfg.get("acceleration_range", [-5.0, 5.0])
        self.acc_min = float(acc_range[0])
        self.acc_max = float(acc_range[1])
        self.speed_limit = float(cfg.get("speed_limit", 15.0))
        self.goal_longitudinal = float(cfg.get("goal_longitudinal", 400.0))
        self.punctual_target = float(cfg.get("punctual_time_target", cfg.get("duration", 35.0)))

        self.front_gap_min = float(cfg.get("lane_change_min_front_gap", 10.0))
        self.rear_gap_min = float(cfg.get("lane_change_min_rear_gap", 8.0))
        self.front_ttc_min = float(cfg.get("lane_change_min_front_ttc", 3.0))
        self.rear_ttc_min = float(cfg.get("lane_change_min_rear_ttc", 2.0))

        self.lanes_count = int(cfg.get("lanes_count", 3))
        self.lane_width = float(cfg.get("lane_width", 4.0))

    def _lane_scalar_from_target(self, current_lane: int, target_lane: int) -> float:
        delta = int(target_lane) - int(current_lane)
        if delta < 0:
            return -1.0
        if delta > 0:
            return 1.0
        return 0.0

    def _acc_phys_to_norm(self, acc_phys: float) -> float:
        den = max(self.acc_max - self.acc_min, 1e-6)
        return float(np.clip(2.0 * (acc_phys - self.acc_min) / den - 1.0, -1.0, 1.0))

    def _longitudinal(self, vehicle) -> float:
        lane = self.env.road.network.get_lane(vehicle.lane_index)
        longi, _ = lane.local_coordinates(vehicle.position)
        return float(longi)

    def _infer_lane_id(self, vehicle) -> int:
        lane_index = getattr(vehicle, "lane_index", None)
        if lane_index is not None and len(lane_index) >= 3:
            return int(np.clip(lane_index[2], 0, self.lanes_count - 1))
        y = float(vehicle.position[1])
        return int(np.clip(int(round(y / max(self.lane_width, 1e-6))), 0, self.lanes_count - 1))

    def _right_bias_target_lane(self, current_lane: int) -> int:
        if int(current_lane) >= self.lanes_count - 1:
            return int(current_lane)
        return int(current_lane) + 1

    def _mobil_target_lane(self, ego, current_lane: int) -> int:
        try:
            shadow = IDMVehicle.create_from(ego)
            road = getattr(shadow, "road", None)
            original_vehicles = None
            if road is not None and hasattr(road, "vehicles"):
                original_vehicles = road.vehicles
                road.vehicles = [v for v in original_vehicles if v is not ego]

            try:
                shadow.target_lane_index = shadow.lane_index
                shadow.follow_road()
                if shadow.enable_lane_change:
                    shadow.change_lane_policy()
            finally:
                if original_vehicles is not None:
                    road.vehicles = original_vehicles

            lane_index = getattr(shadow, "target_lane_index", None)
            if lane_index is not None and len(lane_index) >= 3:
                return int(np.clip(lane_index[2], 0, self.lanes_count - 1))
        except Exception:
            pass
        return int(current_lane)

    def _neighbors_state(self) -> np.ndarray:
        ego = self.env.vehicle
        rows = []
        for v in self.env.road.vehicles:
            if v is ego:
                continue
            rows.append(
                [
                    self._longitudinal(v),
                    self._infer_lane_id(v),
                    float(v.velocity[0]),
                ]
            )
        if not rows:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(rows, dtype=np.float32)

    def _solve_longitudinal_qp(
        self,
        x0: float,
        v0: float,
        current_lane: int,
        target_lane: int,
        v_ref: float,
        neighbors_state: np.ndarray,
    ) -> Tuple[float, int]:
        def _solve_for_lane(planned_target_lane: int) -> Optional[float]:
            H = self.horizon
            a = cp.Variable(H)
            x = cp.Variable(H + 1)
            v = cp.Variable(H + 1)

            constraints = [x[0] == x0, v[0] == v0]
            constraints += [a >= self.acc_min, a <= self.acc_max]
            constraints += [v >= 0.0, v <= self.speed_limit]

            for t in range(H):
                constraints += [x[t + 1] == x[t] + v[t] * self.dt + 0.5 * a[t] * (self.dt ** 2)]
                constraints += [v[t + 1] == v[t] + a[t] * self.dt]

            settle = int(min(H, self.lane_change_settle_steps))

            for t in range(1, H + 1):
                if t == H:
                    active_lanes = {int(planned_target_lane)}
                elif current_lane == planned_target_lane:
                    active_lanes = {int(current_lane)}
                elif t <= settle:
                    active_lanes = {int(current_lane), int(planned_target_lane)}
                else:
                    active_lanes = {int(planned_target_lane)}

                tt = float(t) * self.dt
                for j in range(neighbors_state.shape[0]):
                    xj0 = float(neighbors_state[j, 0])
                    lane_j = int(neighbors_state[j, 1])
                    vj = float(neighbors_state[j, 2])
                    if lane_j not in active_lanes:
                        continue

                    xj_t = xj0 + vj * tt
                    ahead_now = xj0 >= x0

                    if ahead_now:
                        d_front = xj_t - x[t]
                        constraints += [d_front >= self.front_gap_min]
                        constraints += [d_front >= self.front_ttc_min * (v[t] - vj)]
                    else:
                        d_rear = x[t] - xj_t
                        constraints += [d_rear >= self.rear_gap_min]
                        constraints += [d_rear >= self.rear_ttc_min * (vj - v[t])]

            objective = cp.Minimize(
                self.w_acc * cp.sum_squares(a)
                + self.w_speed * cp.sum_squares(v[1:] - float(v_ref))
            )

            problem = cp.Problem(objective, constraints)
            solver_sequence = [cp.OSQP, cp.ECOS, cp.SCS]

            for solver in solver_sequence:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message=r"Solution may be inaccurate.*",
                            category=UserWarning,
                        )
                        problem.solve(solver=solver, warm_start=True, verbose=False)
                except Exception:
                    continue
                if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and a.value is not None:
                    a_seq = np.asarray(a.value, dtype=np.float32).reshape(-1)
                    return float(np.clip(a_seq[0], self.acc_min, self.acc_max))
            return None

        primary = _solve_for_lane(int(target_lane))
        if primary is not None:
            return primary, int(target_lane)

        if int(target_lane) != int(current_lane):
            fallback = _solve_for_lane(int(current_lane))
            if fallback is not None:
                return fallback, int(current_lane)

        heuristic = float(np.clip((float(v_ref) - float(v0)) / self.dt, self.acc_min, self.acc_max))
        return heuristic, int(current_lane)

    def act(self) -> np.ndarray:
        ego = self.env.vehicle
        x0 = self._longitudinal(ego)
        v0 = float(ego.velocity[0])
        current_lane = self._infer_lane_id(ego)

        if self.lane_strategy == "mobil":
            target_lane = self._mobil_target_lane(ego, current_lane)
        else:
            target_lane = self._right_bias_target_lane(current_lane)

        # Reference speed = remaining distance / remaining planned arrival time.
        remain_dist = max(self.goal_longitudinal - x0, 0.0)
        remain_time = max(self.punctual_target - float(getattr(self.env, "time", 0.0)), self.dt)
        v_ref = float(np.clip(remain_dist / remain_time, 0.0, self.speed_limit))

        neighbors_state = self._neighbors_state()
        acc_phys, used_target_lane = self._solve_longitudinal_qp(
            x0=x0,
            v0=v0,
            current_lane=current_lane,
            target_lane=target_lane,
            v_ref=v_ref,
            neighbors_state=neighbors_state,
        )

        lane_scalar = self._lane_scalar_from_target(current_lane, used_target_lane)
        acc_norm = self._acc_phys_to_norm(acc_phys)
        return np.array([lane_scalar, acc_norm], dtype=np.float32)


def main(
    model_dir: str,
    episodes: int = 10,
    record_episodes: Optional[Sequence[int]] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
    horizon: int = 15,
    lane_strategy: str = "right_bias",
    use_low_safety_layer: bool = False,
    enable_rendering: bool = True,
    scenario_name: str = "multi_lane",
):
    eval_root_dir = os.path.join(model_dir, "eval_results")
    os.makedirs(eval_root_dir, exist_ok=True)
    run_folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = unique_path(os.path.join(eval_root_dir, run_folder_name))
    os.makedirs(eval_dir, exist_ok=True)

    log_path = os.path.join(eval_dir, "eval_rule_mpc.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    def log(msg: str = "") -> None:
        print(msg)
        log_file.write(msg + "\n")

    test_overrides: Dict[str, Any] = {
        "initial_lane_id": "random",
        "duration": 70.0,
        "warmup_each_episode": False,
        "screen_width": 1800,
        "screen_height": 300,
        "scaling": 3,
        "centering_position": [0.5, 0.5],
        "show_trajectories": enable_rendering,
        "warmup_render": enable_rendering,
        "offscreen_rendering": enable_rendering,
        "action": {
            "type": "ParamLaneAccelAction",
            "acceleration_range": [-5.0, 5.0],
            "lane_actions": ["KEEP", "LANE_LEFT", "LANE_RIGHT"],
        },
    }
    if env_overrides:
        test_overrides.update(env_overrides)
    scenario_spec = get_scenario_spec(scenario_name)
    importlib.import_module(str(scenario_spec["module"]))
    env_id = str(scenario_spec["env_id"])
    if not enable_rendering:
        test_overrides["show_trajectories"] = False
        test_overrides["warmup_render"] = False
        test_overrides["offscreen_rendering"] = False
    env_config = get_env_config_for_scenario(scenario_name, test_overrides)

    if not record_episodes:
        def trigger(ep_id: int) -> bool:
            return False
    else:
        record_set = {int(ep_idx) - 1 for ep_idx in record_episodes}

        def trigger(ep_id: int) -> bool:
            return ep_id in record_set

    render_mode = "rgb_array" if enable_rendering else None
    base_env = gym.make(env_id, render_mode=render_mode, config=env_config)
    env = RecordVideo(base_env, video_folder=eval_dir, episode_trigger=trigger, name_prefix="rule_mpc") if enable_rendering else base_env

    controller = RuleMPCController(
        env=base_env.unwrapped,
        horizon=int(horizon),
        dt=1.0 / float(env_config["policy_frequency"]),
        w_acc=1.0,
        w_speed=2.0,
        lane_change_settle_steps=4,
        lane_strategy=lane_strategy,
    )
    safety_controller = RuleBasedController(env_config) if use_low_safety_layer else None

    reward_keys = [
        "collision_reward",
        "progress_reward",
        "comfort_reward",
        "lane_change_reward",
        "punctual_reward",
        "on_road_reward",
    ]
    punctual_time_window = env_config.get("punctual_time_window", [20.0, 30.0])
    t_min = float(punctual_time_window[0])
    t_max = float(punctual_time_window[1])

    def get_terminal_lane_id(base_env: Any) -> Optional[int]:
        ego_vehicle = getattr(base_env, "vehicle", None)
        if ego_vehicle is not None:
            lane_index = getattr(ego_vehicle, "lane_index", None)
            if lane_index is not None and len(lane_index) >= 3:
                try:
                    return int(lane_index[2])
                except (TypeError, ValueError):
                    pass
            if hasattr(ego_vehicle, "position"):
                lane_w = float(base_env.config.get("lane_width", 4.0))
                lanes_n = int(base_env.config.get("lanes_count", 3))
                return int(
                    np.clip(
                        int(round(float(ego_vehicle.position[1]) / max(lane_w, 1e-6))),
                        0,
                        lanes_n - 1,
                    )
                )
        return None

    def classify_failure(
        crashed: bool,
        arrived: bool,
        arrival_time: Optional[float],
        final_lane_id: Optional[int],
        goal_lane_id: Optional[int],
    ) -> Tuple[bool, bool, bool, bool, bool]:
        if crashed:
            return True, True, False, False, False
        on_time_arrival = bool(arrived and arrival_time is not None and t_min <= float(arrival_time) <= t_max)
        failed = not on_time_arrival
        wrong_lane = bool(
            failed
            and final_lane_id is not None
            and goal_lane_id is not None
            and int(final_lane_id) != int(goal_lane_id)
        )
        late = bool(failed and arrived and arrival_time is not None and float(arrival_time) > t_max)
        early = bool(failed and arrived and arrival_time is not None and float(arrival_time) < t_min)
        return failed, False, wrong_lane, late, early

    def log_failed_breakdown(
        prefix: str,
        failed_count: int,
        collision_count: int,
        wrong_lane_count: int,
        late_count: int,
        early_count: int,
    ) -> None:
        other_count = int(failed_count) - int(collision_count) - int(wrong_lane_count) - int(late_count) - int(early_count)
        other_count = max(other_count, 0)
        if failed_count <= 0:
            log(f"{prefix}failed episodes       : 0")
            log(f"{prefix}collision            : 0")
            log(f"{prefix}wrong-lane at end    : 0")
            log(f"{prefix}late arrival         : 0")
            log(f"{prefix}early arrival        : 0")
            return
        log(f"{prefix}failed episodes       : {failed_count}")
        log(f"{prefix}collision            : {collision_count} ({collision_count / failed_count * 100:.2f}% of failed)")
        log(f"{prefix}wrong-lane at end    : {wrong_lane_count} ({wrong_lane_count / failed_count * 100:.2f}% of failed)")
        log(f"{prefix}late arrival         : {late_count} ({late_count / failed_count * 100:.2f}% of failed)")
        log(f"{prefix}early arrival        : {early_count} ({early_count / failed_count * 100:.2f}% of failed)")
        if other_count > 0:
            log(f"{prefix}other failures       : {other_count} ({other_count / failed_count * 100:.2f}% of failed)")

    exclude_collision_mean_keys = {"comfort_reward", "lane_change_reward"}

    def format_component_mean(
        key: str,
        total_sum: float,
        total_count: int,
        no_collision_sum: float,
        no_collision_count: int,
    ) -> str:
        if key in exclude_collision_mean_keys:
            if no_collision_count > 0:
                return f"{no_collision_sum / no_collision_count: .6f}"
            return "N/A (all episodes collided)"
        return f"{total_sum / total_count: .6f}"

    log("=" * 80)
    log("Eval Rule+MPC baseline")
    log(f"Eval Rule+MPC model dir: {model_dir}")
    log(f"Eval run folder     : {run_folder_name}")
    log(f"Eval results dir    : {eval_dir}")
    log(f"Episodes            : {episodes}")
    log(f"Lane strategy       : {lane_strategy}")
    log(f"Low safety layer    : {use_low_safety_layer}")
    log(f"Rendering enabled   : {enable_rendering}")
    log("=" * 80)

    ep_lens: list[int] = []
    ep_rets: list[float] = []
    comp_sum = {k: 0.0 for k in reward_keys}
    comp_sum_no_collision = {k: 0.0 for k in exclude_collision_mean_keys}
    non_collision_episode_count = 0

    lane_group_stats: Dict[int, Dict[str, Any]] = {}

    def ensure_lane_group(lane_id: int) -> Dict[str, Any]:
        if lane_id not in lane_group_stats:
            lane_group_stats[lane_id] = {
                "episodes": 0,
                "ep_lens": [],
                "ep_rets": [],
                "comp_sum": {k: 0.0 for k in reward_keys},
                "comp_sum_no_collision": {k: 0.0 for k in exclude_collision_mean_keys},
                "non_collision_episode_count": 0,
                "arrived_count": 0,
                "arrival_times": [],
                "failed_count": 0,
                "failed_collision_count": 0,
                "failed_wrong_lane_count": 0,
                "failed_late_count": 0,
                "failed_early_count": 0,
            }
        return lane_group_stats[lane_id]

    arrived_count, arrival_times = 0, []
    failed_count = 0
    failed_collision_count = 0
    failed_wrong_lane_count = 0
    failed_late_count = 0
    failed_early_count = 0

    viewer_initialized = False
    seed_base = 42

    episode_iter = tqdm(
        range(1, int(episodes) + 1),
        total=int(episodes),
        desc="Eval episodes",
        dynamic_ncols=True,
    )
    for ep in episode_iter:
        obs, _ = env.reset(seed=seed_base + ep)
        del obs

        reset_base_env = env.unwrapped
        init_lane = None
        ego_vehicle = getattr(reset_base_env, "vehicle", None)
        if ego_vehicle is not None:
            lane_index = getattr(ego_vehicle, "lane_index", None)
            if lane_index is not None and len(lane_index) >= 3:
                init_lane = int(lane_index[2])
            elif hasattr(ego_vehicle, "position"):
                lane_w = float(reset_base_env.config.get("lane_width", 4.0))
                lanes_n = int(reset_base_env.config.get("lanes_count", 3))
                init_lane = int(np.clip(int(round(float(ego_vehicle.position[1]) / max(lane_w, 1e-6))), 0, lanes_n - 1))
        if init_lane is None:
            init_lane = -1

        terminated = False
        truncated = False
        step_count = 0
        ep_total_reward = 0.0
        ep_components = {k: 0.0 for k in reward_keys}

        if enable_rendering and not viewer_initialized:
            class Dummy:
                def __init__(self, pos):
                    self.position = np.array(pos, dtype=float)

            base = env.unwrapped
            base.render()
            base.viewer.observer_vehicle = Dummy([base.config["road_length"] / 2.0, 5.0])
            viewer_initialized = True

        while not (terminated or truncated):
            action = controller.act()
            if use_low_safety_layer and safety_controller is not None:
                ego = env.unwrapped.vehicle
                ego_abs = np.array(
                    [
                        float(ego.position[0]),
                        float(ego.position[1]),
                        float(ego.velocity[0]),
                        float(ego.velocity[1]),
                    ],
                    dtype=np.float32,
                )

                others_rel_rows = []
                for v in env.unwrapped.road.vehicles:
                    if v is ego:
                        continue
                    others_rel_rows.append(
                        [
                            float(v.position[0] - ego.position[0]),
                            float(v.position[1] - ego.position[1]),
                            float(v.velocity[0] - ego.velocity[0]),
                            float(v.velocity[1] - ego.velocity[1]),
                        ]
                    )
                if others_rel_rows:
                    others_rel = np.asarray(others_rel_rows, dtype=np.float32)
                else:
                    others_rel = np.zeros((0, 4), dtype=np.float32)

                remain_dist = max(controller.goal_longitudinal - float(ego.position[0]), 0.0)
                remain_time = max(
                    controller.punctual_target - float(getattr(env.unwrapped, "time", 0.0)),
                    controller.dt,
                )
                v_ref = float(np.clip(remain_dist / remain_time, 0.0, controller.speed_limit))
                goal_phys = np.array(
                    [
                        controller.goal_longitudinal,
                        float(ego.position[1]),
                        v_ref,
                        0.0,
                    ],
                    dtype=np.float32,
                )

                action = safety_controller.safety_filter_action(
                    ego_abs=ego_abs,
                    others_rel=others_rel,
                    goal_phys=goal_phys,
                    action=action,
                    dt=controller.dt,
                    remaining_time=remain_time,
                )
            _, reward, terminated, truncated, info = env.step(action)

            rc = info.get("reward_components", {}) if isinstance(info, dict) else {}
            for k in reward_keys:
                ep_components[k] += float(rc.get(k, 0.0))

            ep_total_reward += float(reward)
            step_count += 1

        base_env_unwrapped = env.unwrapped
        arrived = bool(getattr(base_env_unwrapped, "_has_arrived", False))
        arrival_time = getattr(base_env_unwrapped, "_arrival_time", None)
        crashed = bool(getattr(base_env_unwrapped.vehicle, "crashed", False))

        reason = "terminated(crash)" if crashed else ("truncated(time limit)" if truncated else "terminated(goal_or_other)")

        ep_lens.append(step_count)
        ep_rets.append(ep_total_reward)
        for k in reward_keys:
            comp_sum[k] += ep_components[k]

        if not crashed:
            non_collision_episode_count += 1
            for k in exclude_collision_mean_keys:
                comp_sum_no_collision[k] += ep_components[k]

        final_lane_id = get_terminal_lane_id(base_env_unwrapped)
        goal_lane_id = int(base_env_unwrapped.get_goal_lane_id())
        failed, failed_collision, failed_wrong_lane, failed_late, failed_early = classify_failure(
            crashed,
            arrived,
            arrival_time,
            final_lane_id,
            goal_lane_id,
        )
        if arrived:
            arrived_count += 1
            if arrival_time is not None:
                arrival_times.append(float(arrival_time))
        if failed:
            failed_count += 1
        if failed_collision:
            failed_collision_count += 1
        if failed_wrong_lane:
            failed_wrong_lane_count += 1
        if failed_late:
            failed_late_count += 1
        if failed_early:
            failed_early_count += 1

        reason = "terminated" if terminated else ("truncated(time limit)" if truncated else "unknown")
        log("=" * 60)
        log(f"Episode {ep}:")
        log(f"  initial lane            : {init_lane}")
        log(f"  terminal lane           : {final_lane_id if final_lane_id is not None else 'N/A'}")
        log(f"  length (steps)          : {step_count}")
        log(f"  terminated info         : {reason}")
        log(f"  mpc total reward        : {ep_total_reward:.6f}")
        log("  reward components (sum over episode):")
        for k in reward_keys:
            log(f"    {k:18s}: {ep_components[k]: .6f}")

        if arrived and arrival_time is not None:
            log(f"  ARRIVED at t = {float(arrival_time):.3f} s")
        if failed:
            log(
                "  failed flags            : "
                f"collision={int(failed_collision)}, wrong_lane={int(failed_wrong_lane)}, late={int(failed_late)}, early={int(failed_early)}"
            )

        if enable_rendering and base_env_unwrapped.config.get("show_trajectories", False):
            save_speed_acc_curves(env, ep_idx=ep, model_path=eval_dir)

        group = ensure_lane_group(int(init_lane))
        group["episodes"] += 1
        group["ep_lens"].append(int(step_count))
        group["ep_rets"].append(float(ep_total_reward))
        for k in reward_keys:
            group["comp_sum"][k] += ep_components[k]
        if not crashed:
            group["non_collision_episode_count"] += 1
            for k in exclude_collision_mean_keys:
                group["comp_sum_no_collision"][k] += ep_components[k]
        if arrived:
            group["arrived_count"] += 1
            if arrival_time is not None:
                group["arrival_times"].append(float(arrival_time))
        if failed:
            group["failed_count"] += 1
        if failed_collision:
            group["failed_collision_count"] += 1
        if failed_wrong_lane:
            group["failed_wrong_lane_count"] += 1
        if failed_late:
            group["failed_late_count"] += 1
        if failed_early:
            group["failed_early_count"] += 1

    n = int(episodes)
    log("=" * 80)
    log("Summary by initial lane:")
    lanes_for_summary = int(env_config.get("lanes_count", 3))
    for lane_id in range(lanes_for_summary):
        group = lane_group_stats.get(lane_id)
        if group is None or int(group["episodes"]) == 0:
            log(f"  lane {lane_id}: no episodes")
            continue

        n_lane = int(group["episodes"])
        log("-" * 80)
        log(f"  lane {lane_id}:")
        log(f"    episodes              : {n_lane}")
        log(f"    mean length           : {float(np.mean(group['ep_lens'])):.3f} steps")
        log(f"    mean mpc total reward : {float(np.mean(group['ep_rets'])):.6f}")
        log("    mean reward components (per episode):")
        for k in reward_keys:
            log(
                f"      {k:16s}: "
                f"{format_component_mean(k, group['comp_sum'][k], n_lane, group['comp_sum_no_collision'].get(k, 0.0), int(group['non_collision_episode_count']))}"
            )

        lane_arrive_rate = group["arrived_count"] / n_lane if n_lane else 0.0
        log(f"    arrival rate          : {lane_arrive_rate * 100:.2f}%")
        if group["arrived_count"] > 0:
            log(
                f"    mean arrival time     : {float(np.mean(group['arrival_times'])):.3f} s "
                f"(over {int(group['arrived_count'])} success episodes)"
            )
        else:
            log("    mean arrival time     : N/A (no successful episodes)")
        log_failed_breakdown(
            "    ",
            int(group["failed_count"]),
            int(group["failed_collision_count"]),
            int(group["failed_wrong_lane_count"]),
            int(group["failed_late_count"]),
            int(group["failed_early_count"]),
        )

    log("=" * 80)
    log("Overall summary:")
    log("Summary over all episodes:")
    log(f"  episodes                : {n}")
    log(f"  mean length             : {float(np.mean(ep_lens)) if n else 0.0:.3f} steps")
    log(f"  mean mpc total reward   : {float(np.mean(ep_rets)) if n else 0.0:.6f}")
    log("  mean reward components (per episode):")
    for k in reward_keys:
        log(
            f"    {k:18s}: "
            f"{format_component_mean(k, comp_sum[k], n, comp_sum_no_collision.get(k, 0.0), non_collision_episode_count) if n else 'N/A'}"
        )
    arrive_rate = arrived_count / n if n else 0.0
    log(f"  arrival rate            : {arrive_rate * 100.0:.2f}%")
    if arrived_count:
        log(f"  mean arrival time       : {float(np.mean(arrival_times)):.3f} s (over {arrived_count} success episodes)")
    else:
        log("  mean arrival time       : N/A (no successful episodes)")
    log_failed_breakdown(
        "  ",
        failed_count,
        failed_collision_count,
        failed_wrong_lane_count,
        failed_late_count,
        failed_early_count,
    )
    log("=" * 80)

    log_file.close()
    env.close()


if __name__ == "__main__":
    main(
        # model_dir="./models/baseline_260407_mlc_mpc",
        # model_dir="./models/baseline_260407_mobil_mpc",
        model_dir="./models/baseline_260407_mlc_mpc_withSL",

        episodes=301,
        record_episodes=[i for i in range(1, 301)],
        horizon=15,

        # lane_strategy="mobil",
        lane_strategy="right_bias",

        # use_low_safety_layer=False,
        use_low_safety_layer=True,

        enable_rendering=False,
    )
