import os
import warnings
import random
import json
import csv
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

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
from configs.conf import TRAIN_CONFIG
from custom_env.vehicle.behavior import IDMVehicle
from rl.algos.HRL.rule_based import RuleBasedController
from scenarios.goal_lane_logic import sample_goal_lane_id
from util.hiro_utils import env_config_from_run_config, load_hiro_run_config, unique_path
from util.config_utils import deep_update
from util.plot_result import save_speed_acc_curves


# Kept in sync with test_hiro.py: these are the only per-goal-lane fields that
# may differ between the three SAC training environments.
LANE_TRAFFIC_CONFIG_KEYS = (
    "behavior_probs", "behavior_lane_probs", "behavior_vehicle_types",
    "flow_speed_range", "speed_distribution", "spawn_probability",
    "spawn_min_gap", "spawn_min_t_headway", "spawn_check_adjacent_cutins",
    "spawn_adjacent_cutin_front_gap", "spawn_adjacent_cutin_back_gap",
    "movement_lanes", "movement_behavior_probs",
    "background_vehicle_respect_movement_lanes", "background_snapshot_reset",
    "background_snapshot_path", "background_snapshot_paths",
    "background_snapshot_max_resample_attempts",
    "background_snapshot_chunk_reuse_enabled",
    "background_snapshot_chunk_reuse_count",
    "background_snapshot_chunk_cache_size", "background_snapshot_phase_offset",
    "signal_plan", "enable_signal_green_launch_behavior",
    "signal_green_launch_approach_distance", "signal_green_launch_end_margin",
    "signal_green_launch_target_speed", "enable_signal_cycle_spawn_probability",
    "signal_cycle_spawn_probability",
)


def _extract_goal_lane_environment_config(saved_env_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Match test_hiro.py's per-goal-lane SAC environment extraction."""
    extracted = {
        key: deepcopy(saved_env_config[key])
        for key in LANE_TRAFFIC_CONFIG_KEYS
        if key in saved_env_config
    }
    for key in ("rule_based_compute_action_mode", "rule_follow_mode_enabled"):
        if key in saved_env_config:
            extracted[key] = deepcopy(saved_env_config[key])
    action_cfg = saved_env_config.get("action", {})
    if isinstance(action_cfg, Mapping) and "acceleration_range" in action_cfg:
        extracted.setdefault("action", {})["acceleration_range"] = deepcopy(
            action_cfg["acceleration_range"]
        )
    return extracted


class RuleMPCController:
    """Rule + MPC baseline.

    1) Lateral lane target moves one lane at a time toward the episode goal lane.
    2) Longitudinal motion is obtained by solving a convex QP over acceleration sequence.
       Objective: sum(a^2) + sum((v-v_ref)^2)
       Constraints: min distance & TTC to nearby vehicles (front/rear), speed and accel bounds.
         If lane-change-target MPC is infeasible, fallback target lane is current lane.
    3) When the ego vehicle is within ``forced_goal_lane_change_distance`` of the
       goal x but is still in the wrong lane, it attempts one adjacent merge toward
       the goal lane. Its QP predicts that a rear vehicle in the target lane
       brakes at ``forced_goal_lane_rear_braking``.
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
        if lane_strategy_norm not in {"right_bias", "mobil", "benefit_urgency"}:
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
        self.forced_goal_lane_change_distance = float(
            max(cfg.get("forced_goal_lane_change_distance", 30.0), 0.0)
        )
        self.enable_forced_goal_lane_change = bool(
            cfg.get("enable_forced_goal_lane_change", True)
        )
        self.forced_goal_lane_rear_braking = float(
            max(cfg.get("forced_goal_lane_rear_braking", 3.0), 0.0)
        )
        self.mobil_goal_lane_bias_distance = float(
            max(cfg.get("mobil_goal_lane_bias_distance", 100.0), 0.0)
        )

        self.front_gap_min = float(cfg.get("lane_change_min_front_gap", 10.0))
        self.rear_gap_min = float(cfg.get("lane_change_min_rear_gap", 8.0))
        self.front_ttc_min = float(cfg.get("lane_change_min_front_ttc", 3.0))
        self.rear_ttc_min = float(cfg.get("lane_change_min_rear_ttc", 2.0))

        self.lanes_count = int(cfg.get("lanes_count", 3))
        self.lane_width = float(cfg.get("lane_width", 4.0))
        # MOBIL-like benefit + goal-lane urgency strategy parameters.
        self.lane_score_politeness = float(cfg.get("lane_score_politeness", 0.2))
        self.lane_score_traffic_weight = float(cfg.get("lane_score_traffic_weight", 1.0))
        self.lane_score_goal_weight = float(cfg.get("lane_score_goal_weight", 0.25))
        self.lane_score_goal_urgency_weight = float(
            cfg.get("lane_score_goal_urgency_weight", 2.5)
        )
        self.lane_score_goal_approach_distance = float(
            max(cfg.get("lane_score_goal_approach_distance", 150.0), 1e-3)
        )
        self.lane_score_change_cost = float(cfg.get("lane_score_change_cost", 0.1))
        self.last_lane_scores: Dict[int, float] = {}
        self.last_forced_goal_lane_change = False
        self.mobil_failure_count = 0
        self.mobil_last_error: Optional[str] = None
        self.mobil_failed_this_episode = False

    def begin_episode(self) -> None:
        self.mobil_failed_this_episode = False

    def _goal_target_lane(self, current_lane: int) -> int:
        try:
            goal_lane = int(self.env.get_goal_lane_id())
        except (AttributeError, TypeError, ValueError):
            goal_lane = int(self.env.config.get("goal_lane_id", current_lane))
        goal_lane = int(np.clip(goal_lane, 0, self.lanes_count - 1))
        if goal_lane < int(current_lane):
            return int(current_lane) - 1
        if goal_lane > int(current_lane):
            return int(current_lane) + 1
        return int(current_lane)

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
        except Exception as exc:
            self.mobil_failure_count += 1
            self.mobil_failed_this_episode = True
            self.mobil_last_error = f"{type(exc).__name__}: {exc}"
        return int(current_lane)

    @staticmethod
    def _vehicle_speed(vehicle: Any) -> float:
        return float(getattr(vehicle, "speed", np.linalg.norm(vehicle.velocity)))

    def _idm_acceleration_estimate(self, vehicle: Any, front_vehicle: Optional[Any]) -> float:
        """Deterministic IDM estimate used only for relative lane scoring."""
        if vehicle is None:
            return 0.0
        speed = max(self._vehicle_speed(vehicle), 0.0)
        desired_speed = float(
            np.clip(getattr(vehicle, "target_speed", self.speed_limit), 0.1, self.speed_limit)
        )
        comfort_acc = 2.0
        comfort_brake = 5.0
        acc = comfort_acc * (1.0 - (speed / desired_speed) ** 4)
        if front_vehicle is None:
            return float(acc)

        try:
            distance = max(float(vehicle.lane_distance_to(front_vehicle)), 0.1)
        except Exception:
            distance = max(float(front_vehicle.position[0] - vehicle.position[0]), 0.1)
        front_speed = max(self._vehicle_speed(front_vehicle), 0.0)
        desired_gap = 5.0 + speed * 1.5 + speed * (speed - front_speed) / (
            2.0 * np.sqrt(comfort_acc * comfort_brake)
        )
        return float(acc - comfort_acc * (max(desired_gap, 0.1) / distance) ** 2)

    def _benefit_urgency_target_lane(self, ego: Any, current_lane: int, x0: float) -> int:
        """Score adjacent lanes by MOBIL-like benefit plus goal urgency."""
        goal_lane = self._goal_target_lane(current_lane)
        # _goal_target_lane returns an adjacent step; recover the actual endpoint
        # to score all candidate lanes by their distance from the route goal.
        try:
            final_goal_lane = int(self.env.get_goal_lane_id())
        except (AttributeError, TypeError, ValueError):
            final_goal_lane = int(goal_lane)
        final_goal_lane = int(np.clip(final_goal_lane, 0, self.lanes_count - 1))
        remaining_distance = max(self.goal_longitudinal - x0, 0.0)
        urgency = float(np.clip(
            1.0 - remaining_distance / self.lane_score_goal_approach_distance,
            0.0,
            1.0,
        ))

        old_front, old_rear = self.env.road.neighbour_vehicles(ego)
        ego_acc_now = self._idm_acceleration_estimate(ego, old_front)
        old_rear_acc_now = self._idm_acceleration_estimate(old_rear, ego)
        old_rear_acc_after = self._idm_acceleration_estimate(old_rear, old_front)
        scores: Dict[int, float] = {}

        for lane in range(max(0, current_lane - 1), min(self.lanes_count - 1, current_lane + 1) + 1):
            if lane == current_lane:
                traffic_benefit = 0.0
            else:
                lane_index = (ego.lane_index[0], ego.lane_index[1], int(lane))
                try:
                    target_lane_obj = self.env.road.network.get_lane(lane_index)
                    if not target_lane_obj.is_reachable_from(ego.position):
                        scores[lane] = -np.inf
                        continue
                    new_front, new_rear = self.env.road.neighbour_vehicles(ego, lane_index)
                except Exception:
                    scores[lane] = -np.inf
                    continue

                ego_acc_after = self._idm_acceleration_estimate(ego, new_front)
                new_rear_acc_now = self._idm_acceleration_estimate(new_rear, new_front)
                new_rear_acc_after = self._idm_acceleration_estimate(new_rear, ego)
                traffic_benefit = (
                    ego_acc_after - ego_acc_now
                    + self.lane_score_politeness
                    * ((new_rear_acc_after - new_rear_acc_now) + (old_rear_acc_after - old_rear_acc_now))
                )

            goal_progress = abs(current_lane - final_goal_lane) - abs(lane - final_goal_lane)
            goal_score = (
                self.lane_score_goal_weight + self.lane_score_goal_urgency_weight * urgency
            ) * float(goal_progress)
            change_cost = self.lane_score_change_cost if lane != current_lane else 0.0
            scores[lane] = self.lane_score_traffic_weight * traffic_benefit + goal_score - change_cost

        self.last_lane_scores = scores
        best_score = max(scores.values())
        best_lanes = [lane for lane, score in scores.items() if np.isclose(score, best_score)]
        # Prefer keeping the current lane on ties to prevent lane oscillation.
        return int(current_lane if current_lane in best_lanes else min(best_lanes))

    def _safe_fallback_acceleration(
        self,
        x0: float,
        v0: float,
        current_lane: int,
        v_ref: float,
        neighbors_state: np.ndarray,
    ) -> float:
        """One-step safety-constrained fallback, with front-safety priority if infeasible."""
        dt = self.dt
        lower = self.acc_min
        upper = min(self.acc_max, (self.speed_limit - v0) / dt)
        desired = float(np.clip((v_ref - v0) / dt, self.acc_min, self.acc_max))

        for xj0, lane_j_raw, vj in neighbors_state:
            if int(lane_j_raw) != int(current_lane):
                continue
            if float(xj0) >= x0:
                gap_free = float(xj0) - x0 + (float(vj) - v0) * dt
                upper = min(upper, 2.0 * (gap_free - self.front_gap_min) / (dt ** 2))
                upper = min(
                    upper,
                    (gap_free - self.front_ttc_min * (v0 - float(vj)))
                    / (0.5 * dt ** 2 + self.front_ttc_min * dt),
                )
            else:
                gap_free = x0 - float(xj0) + (v0 - float(vj)) * dt
                lower = max(lower, 2.0 * (self.rear_gap_min - gap_free) / (dt ** 2))
                lower = max(
                    lower,
                    (self.rear_ttc_min * (float(vj) - v0) - gap_free)
                    / (0.5 * dt ** 2 + self.rear_ttc_min * dt),
                )

        upper = max(self.acc_min, min(self.acc_max, upper))
        lower = max(self.acc_min, min(self.acc_max, lower))
        if lower <= upper:
            return float(np.clip(desired, lower, upper))
        # Conflicting front/rear constraints: protect against the frontal collision.
        return float(upper)

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
        target_lane_rear_braking: float = 0.0,
    ) -> Tuple[float, int]:
        """Solve the longitudinal MPC.

        ``target_lane_rear_braking`` is a positive braking magnitude.  It is
        only applied to a vehicle that starts behind ego in the requested
        target lane; all other surrounding vehicles retain constant-speed
        prediction.
        """
        def _solve_for_lane(planned_target_lane: int) -> Optional[float]:
            H = self.horizon
            a = cp.Variable(H)
            x = cp.Variable(H + 1)
            v = cp.Variable(H + 1)

            constraints = [x[0] == x0, v[0] == v0]
            constraints += [a >= self.acc_min, a <= self.acc_max]
            # v0 is measured state and cannot be made feasible retroactively.
            # For an externally overridden overspeed state, permit only the
            # fastest physically possible monotone recovery toward the limit.
            recovery_upper = np.maximum(
                self.speed_limit,
                v0 + self.acc_min * self.dt * np.arange(1, H + 1),
            )
            constraints += [v[1:] >= 0.0, v[1:] <= recovery_upper]

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

                    ahead_now = xj0 >= x0
                    target_rear_brakes = bool(
                        target_lane_rear_braking > 0.0
                        and lane_j == int(target_lane)
                        and not ahead_now
                    )
                    if target_rear_brakes:
                        # Constant -b prediction until the rear vehicle stops;
                        # do not extrapolate it backwards after that instant.
                        brake = float(target_lane_rear_braking)
                        stop_time = max(vj, 0.0) / brake
                        moving_time = min(tt, stop_time)
                        xj_t = xj0 + max(vj, 0.0) * moving_time - 0.5 * brake * moving_time ** 2
                        if tt > stop_time:
                            xj_t = xj0 + max(vj, 0.0) ** 2 / (2.0 * brake)
                        vj_t = max(vj - brake * tt, 0.0)
                    else:
                        xj_t = xj0 + vj * tt
                        vj_t = vj

                    if ahead_now:
                        d_front = xj_t - x[t]
                        constraints += [d_front >= self.front_gap_min]
                        constraints += [d_front >= self.front_ttc_min * (v[t] - vj_t)]
                    else:
                        d_rear = x[t] - xj_t
                        constraints += [d_rear >= self.rear_gap_min]
                        constraints += [d_rear >= self.rear_ttc_min * (vj_t - v[t])]

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

        safe_fallback = self._safe_fallback_acceleration(
            x0, v0, current_lane, v_ref, neighbors_state
        )
        return safe_fallback, int(current_lane)

    def act(self) -> np.ndarray:
        ego = self.env.vehicle
        x0 = self._longitudinal(ego)
        v0 = float(ego.velocity[0])
        current_lane = self._infer_lane_id(ego)
        remaining_distance = max(self.goal_longitudinal - x0, 0.0)
        try:
            final_goal_lane = int(self.env.get_goal_lane_id())
        except (AttributeError, TypeError, ValueError):
            final_goal_lane = current_lane
        final_goal_lane = int(np.clip(final_goal_lane, 0, self.lanes_count - 1))
        forced_goal_lane_change = bool(
            self.enable_forced_goal_lane_change
            and 0.0 < remaining_distance <= self.forced_goal_lane_change_distance
            and current_lane != final_goal_lane
        )
        self.last_forced_goal_lane_change = forced_goal_lane_change

        if forced_goal_lane_change:
            # Near goal x, force one adjacent step toward the required lane.
            # The low speed created by max braking makes the forced merge easier.
            target_lane = self._goal_target_lane(current_lane)
        elif self.lane_strategy == "mobil":
            # Near the route endpoint, MOBIL's traffic-efficiency objective is
            # subordinated to route completion. right_bias is goal-directed in
            # this script, so it advances one adjacent lane toward goal_lane.
            if 0.0 < remaining_distance <= self.mobil_goal_lane_bias_distance:
                target_lane = self._goal_target_lane(current_lane)
            else:
                target_lane = self._mobil_target_lane(ego, current_lane)
        elif self.lane_strategy == "benefit_urgency":
            target_lane = self._benefit_urgency_target_lane(ego, current_lane, x0)
        else:
            target_lane = self._goal_target_lane(current_lane)

        # Reference speed = remaining distance / remaining planned arrival time.
        remain_dist = remaining_distance
        remain_time = max(self.punctual_target - float(getattr(self.env, "time", 0.0)), self.dt)
        v_ref = float(np.clip(remain_dist / remain_time, 0.0, self.speed_limit))

        neighbors_state = self._neighbors_state()
        # The forced merge keeps its lateral command even if the QP falls back,
        # but uses a more realistic target-lane rear-vehicle prediction: the
        # rear vehicle is assumed to brake at 3 m/s^2 (configurable) for the
        # horizon. Normal driving keeps the previous constant-speed model.
        acc_phys, used_target_lane = self._solve_longitudinal_qp(
            x0=x0,
            v0=v0,
            current_lane=current_lane,
            target_lane=target_lane,
            v_ref=v_ref,
            neighbors_state=neighbors_state,
            target_lane_rear_braking=(
                self.forced_goal_lane_rear_braking if forced_goal_lane_change else 0.0
            ),
        )

        # benefit_urgency deliberately issues its selected lateral action
        # directly. Its longitudinal QP fallback must not cancel that action.
        lateral_target_lane = (
            target_lane
            if forced_goal_lane_change or self.lane_strategy == "benefit_urgency"
            else used_target_lane
        )
        lane_scalar = self._lane_scalar_from_target(current_lane, lateral_target_lane)
        acc_norm = self._acc_phys_to_norm(acc_phys)
        return np.array([lane_scalar, acc_norm], dtype=np.float32)


def main(
    model_dir: str,
    episodes: int = 10,
    record_episodes: Optional[Sequence[int]] = None,
    record_trajectory_episodes: Optional[Sequence[int]] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
    horizon: int = 15,
    lane_strategy: str = "right_bias",
    use_low_safety_layer: bool = False,
    enable_rendering: bool = True,
    scenario_name: str = "multi_lane",
    initial_lane_id: Optional[Any] = None,
    goal_lane_id: Optional[Any] = None,
    duration: Optional[float] = None,
    goal_longitudinal: Optional[float] = None,
    forced_goal_lane_change_distance: Optional[float] = None,
    enable_forced_goal_lane_change: Optional[bool] = None,
    forced_goal_lane_rear_braking: Optional[float] = None,
    mobil_goal_lane_bias_distance: Optional[float] = None,
    punctual_time_target: Optional[float] = None,
    punctual_time_window: Optional[Sequence[float]] = None,
    spawn_probability: Optional[float] = None,
    start_longitudinal: Optional[float] = None,
    episode_start_phase_offset: Optional[float] = None,
    enable_queue_takeover: Optional[bool] = None,
    reference_env_model_dir: Optional[str] = None,
    goal_lane_reference_env_model_dirs: Optional[Mapping[int, str]] = None,
    snapshot_pool_by_goal_lane: Optional[Mapping[int, str]] = None,
):
    if record_episodes and any(int(ep_idx) < 1 for ep_idx in record_episodes):
        raise ValueError(
            "record_episodes uses 1-based episode numbers; values must be >= 1"
        )
    if record_trajectory_episodes and any(
        int(ep_idx) < 1 for ep_idx in record_trajectory_episodes
    ):
        raise ValueError(
            "record_trajectory_episodes uses 1-based episode numbers; "
            "values must be >= 1"
        )

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

    runtime_overrides: Dict[str, Any] = {
        "screen_width": 1800,
        "screen_height": 300,
        "scaling": 3,
        "centering_position": [0.5, 0.5],
        "show_trajectories": enable_rendering,
        "warmup_render": False,
        "offscreen_rendering": enable_rendering,
    }
    test_env_overrides: Dict[str, Any] = {}
    explicit_overrides = {
        "initial_lane_id": initial_lane_id,
        "goal_lane_id": goal_lane_id,
        "duration": duration,
        "goal_longitudinal": goal_longitudinal,
        "forced_goal_lane_change_distance": forced_goal_lane_change_distance,
        "enable_forced_goal_lane_change": enable_forced_goal_lane_change,
        "forced_goal_lane_rear_braking": forced_goal_lane_rear_braking,
        "mobil_goal_lane_bias_distance": mobil_goal_lane_bias_distance,
        "punctual_time_target": punctual_time_target,
        "spawn_probability": spawn_probability,
        "start_longitudinal": start_longitudinal,
        "episode_start_phase_offset": episode_start_phase_offset,
        "enable_queue_takeover": enable_queue_takeover,
    }
    test_env_overrides.update({k: v for k, v in explicit_overrides.items() if v is not None})
    if punctual_time_window is not None:
        if len(punctual_time_window) != 2:
            raise ValueError("punctual_time_window must contain exactly two values")
        test_env_overrides["punctual_time_window"] = [
            float(punctual_time_window[0]),
            float(punctual_time_window[1]),
        ]
    if env_overrides:
        deep_update(test_env_overrides, env_overrides)
    scenario_spec = get_scenario_spec(scenario_name)
    importlib.import_module(str(scenario_spec["module"]))
    env_id = str(scenario_spec["env_id"])
    if not enable_rendering:
        runtime_overrides["show_trajectories"] = False
        runtime_overrides["warmup_render"] = False
        runtime_overrides["offscreen_rendering"] = False
    conf_config_overrides = TRAIN_CONFIG.get("config_overrides", {}) or {}
    conf_env_overrides = deepcopy(
        dict(conf_config_overrides.get("environment", {}) or {})
    )
    # Normal evaluation precedence: base -> scenario -> conf -> this script.
    # A SAC-reference profile then replaces the conf-derived environment with
    # its saved training environment, just as test_hiro.py does.
    env_config = get_env_config_for_scenario(scenario_name, conf_env_overrides)
    reference_env_config_path: Optional[str] = None
    if reference_env_model_dir:
        reference_run_config, reference_env_config_path = load_hiro_run_config(
            reference_env_model_dir
        )
        deep_update(env_config, env_config_from_run_config(reference_run_config))
        env_config.pop("_env_seed", None)
        env_config.pop("actual_episode_start_phase_offset", None)
    deep_update(env_config, test_env_overrides)
    deep_update(env_config, runtime_overrides)

    lane_env_overrides: Dict[int, Dict[str, Any]] = {}
    lane_env_config_sources: Dict[int, str] = {}
    for raw_lane, source_dir in (goal_lane_reference_env_model_dirs or {}).items():
        lane = int(raw_lane)
        lane_run_config, lane_source_path = load_hiro_run_config(str(source_dir))
        lane_env_overrides[lane] = _extract_goal_lane_environment_config(
            env_config_from_run_config(lane_run_config)
        )
        lane_env_config_sources[lane] = lane_source_path

    normalized_snapshot_pools = {
        int(lane): str(path)
        for lane, path in (snapshot_pool_by_goal_lane or {}).items()
    }

    record_set: set[int] = set()
    if not record_episodes:
        def trigger(ep_id: int) -> bool:
            return False
    else:
        record_set = {int(ep_idx) - 1 for ep_idx in record_episodes}

        def trigger(ep_id: int) -> bool:
            return ep_id in record_set

    trajectory_record_set = (
        {int(ep_idx) for ep_idx in record_trajectory_episodes}
        if record_trajectory_episodes
        else set()
    )

    render_mode = "rgb_array" if enable_rendering else None
    per_goal_lane_environment = bool(lane_env_overrides or normalized_snapshot_pools)

    def config_for_goal_lane(goal_lane: int) -> Dict[str, Any]:
        cfg = deepcopy(env_config)
        lane = int(goal_lane)
        if lane_env_overrides:
            if lane not in lane_env_overrides:
                raise ValueError(
                    f"No reference environment config for goal_lane_id={lane}. "
                    f"Available lanes: {sorted(lane_env_overrides)}"
                )
            deep_update(cfg, deepcopy(lane_env_overrides[lane]))
            # Match test_hiro.py: user test settings override lane model traffic.
            deep_update(cfg, test_env_overrides)
            deep_update(cfg, runtime_overrides)
        if normalized_snapshot_pools:
            if lane not in normalized_snapshot_pools:
                raise ValueError(
                    f"No snapshot pool configured for goal_lane_id={lane}. "
                    f"Available lanes: {sorted(normalized_snapshot_pools)}"
                )
            cfg["background_snapshot_reset"] = True
            cfg["background_snapshot_path"] = None
            cfg["background_snapshot_paths"] = [normalized_snapshot_pools[lane]]
        cfg["goal_lane_id"] = lane
        cfg["goal_lane_probs"] = None
        return cfg

    def make_runtime_environment(config: Mapping[str, Any], episode_number: Optional[int] = None):
        base = gym.make(env_id, render_mode=render_mode, config=dict(config))
        wrapped = base
        if enable_rendering:
            if episode_number is None:
                wrapped = RecordVideo(base, video_folder=eval_dir, episode_trigger=trigger, name_prefix="rule_mpc")
            else:
                should_record = int(episode_number) in record_set
                wrapped = RecordVideo(
                    base,
                    video_folder=eval_dir,
                    episode_trigger=lambda _ep_id: should_record,
                    name_prefix=f"rule_mpc_ep_{int(episode_number):04d}",
                )
        active_controller = RuleMPCController(
            env=base.unwrapped,
            horizon=int(horizon),
            dt=1.0 / float(config["policy_frequency"]),
            w_acc=1.0,
            w_speed=2.0,
            lane_change_settle_steps=4,
            lane_strategy=lane_strategy,
        )
        active_safety_controller = RuleBasedController(config) if use_low_safety_layer else None
        return wrapped, active_controller, active_safety_controller

    env, controller, safety_controller = make_runtime_environment(env_config)

    reward_keys = [
        "collision_reward",
        "progress_reward",
        "speed_ref_aux_reward",
        "comfort_reward",
        "lane_change_reward",
        "goal_lane_dense_reward",
        "punctual_reward",
        "wrong_lane_terminal_penalty",
        "on_road_reward",
    ]
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
        t_min: float,
        t_max: float,
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
    log(f"Scenario            : {scenario_name} ({env_id})")
    log(f"Initial lane config : {env_config.get('initial_lane_id')}")
    log(f"Goal lane config    : {env_config.get('goal_lane_id')}")
    log(f"Goal longitudinal   : {env_config.get('goal_longitudinal')}")
    log(f"Duration            : {env_config.get('duration')} s")
    log(f"Punctual target     : {env_config.get('punctual_time_target')} s")
    log(f"Punctual window     : {env_config.get('punctual_time_window')}")
    if reference_env_config_path:
        log(f"Reference env config: {reference_env_config_path}")
    if lane_env_config_sources:
        log("Per-goal env config sources:")
        for lane, source in sorted(lane_env_config_sources.items()):
            log(f"  goal_lane={lane}: {source}")
    if normalized_snapshot_pools:
        log("Snapshot pool by goal lane:")
        for lane, pool in sorted(normalized_snapshot_pools.items()):
            log(f"  goal_lane={lane}: {pool}")
    log(f"Lane strategy       : {lane_strategy}")
    log(f"Low safety layer    : {use_low_safety_layer}")
    log(f"Rendering enabled   : {enable_rendering}")
    log("=" * 80)

    seed_base = 42
    with open(os.path.join(eval_dir, "effective_eval_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "controller": "rule_mpc",
                "scenario_name": scenario_name,
                "env_id": env_id,
                "reference_env_config_source": reference_env_config_path,
                "goal_lane_env_config_sources": {
                    str(lane): source for lane, source in sorted(lane_env_config_sources.items())
                },
                "snapshot_pool_by_goal_lane": {
                    str(lane): pool for lane, pool in sorted(normalized_snapshot_pools.items())
                },
                "test_env_overrides": test_env_overrides,
                "runtime_overrides": runtime_overrides,
                "base_environment": env_config,
                "goal_lane_environment_overrides": lane_env_overrides,
                "episode_seeds": [seed_base + ep for ep in range(1, int(episodes) + 1)],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

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
    mobil_failed_episode_count = 0
    mobil_failure_call_count = 0
    mobil_last_failure_error: Optional[str] = None

    viewer_initialized = False
    episode_iter = tqdm(
        range(1, int(episodes) + 1),
        total=int(episodes),
        desc="Eval episodes",
        dynamic_ncols=True,
    )
    for ep in episode_iter:
        episode_seed = seed_base + ep
        planned_goal_lane: Optional[int] = None
        if per_goal_lane_environment:
            planned_goal_lane = int(
                sample_goal_lane_id(
                    np.random.default_rng(int(episode_seed)),
                    goal_lane_id=env_config.get("goal_lane_id", 0),
                    lanes_count=int(env_config.get("lanes_count", 1)),
                    goal_lane_probs=env_config.get("goal_lane_probs", None),
                )
            )
            env.close()
            env, controller, safety_controller = make_runtime_environment(
                config_for_goal_lane(planned_goal_lane), ep
            )
            viewer_initialized = False

        # Mirrors test_hiro.py's episode seeding for comparable traffic streams.
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        obs, _ = env.reset(seed=episode_seed)
        controller.begin_episode()

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
        forced_goal_lane_change_steps = 0
        ep_total_reward = 0.0
        ep_components = {k: 0.0 for k in reward_keys}
        should_record_trajectory = ep in trajectory_record_set
        trajectory_rows: list[Dict[str, Any]] = []

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
            if controller.last_forced_goal_lane_change:
                forced_goal_lane_change_steps += 1
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
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            rc = info.get("reward_components", None) if isinstance(info, dict) else None
            if rc is None:
                rc = getattr(env.unwrapped, "_last_weighted_rewards", None)
            rc = rc or {}
            for k in reward_keys:
                ep_components[k] += float(rc.get(k, 0.0))

            if should_record_trajectory:
                row: Dict[str, Any] = {
                    "episode": int(ep),
                    "step": int(step_count),
                    "done": int(done),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "queue_takeover_active": int(
                        bool(info.get("queue_takeover_active", False))
                        if isinstance(info, dict)
                        else False
                    ),
                    "reward": float(reward),
                }
                flat_obs = np.asarray(obs, dtype=np.float32).reshape(-1)
                flat_action = np.asarray(action, dtype=np.float32).reshape(-1)
                for i, value in enumerate(flat_obs):
                    row[f"obs_{i}"] = float(value)
                for i, value in enumerate(flat_action):
                    row[f"action_{i}"] = float(value)
                for key in reward_keys:
                    row[key] = float(rc.get(key, 0.0))
                trajectory_rows.append(row)

            ep_total_reward += float(reward)
            step_count += 1
            obs = obs_next

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
        if planned_goal_lane is not None and goal_lane_id != planned_goal_lane:
            raise RuntimeError(
                "Planned goal lane and environment goal lane diverged: "
                f"planned={planned_goal_lane}, actual={goal_lane_id}"
            )
        episode_time_window = base_env_unwrapped.config.get(
            "punctual_time_window", env_config.get("punctual_time_window", [20.0, 30.0])
        )
        t_min = float(episode_time_window[0])
        t_max = float(episode_time_window[1])
        failed, failed_collision, failed_wrong_lane, failed_late, failed_early = classify_failure(
            crashed,
            arrived,
            arrival_time,
            final_lane_id,
            goal_lane_id,
            t_min,
            t_max,
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

        if controller.mobil_failed_this_episode:
            mobil_failed_episode_count += 1
        mobil_failure_call_count += controller.mobil_failure_count
        if controller.mobil_last_error is not None:
            mobil_last_failure_error = controller.mobil_last_error

        reason = "terminated" if terminated else ("truncated(time limit)" if truncated else "unknown")
        log("=" * 60)
        log(f"Episode {ep}:")
        log(f"  initial lane            : {init_lane}")
        log(f"  terminal lane           : {final_lane_id if final_lane_id is not None else 'N/A'}")
        log(f"  goal lane               : {goal_lane_id}")
        if planned_goal_lane is not None:
            log(f"  snapshot pool           : {base_env_unwrapped.config.get('background_snapshot_paths')}")
        log(f"  length (steps)          : {step_count}")
        if forced_goal_lane_change_steps:
            log(
                "  forced goal-lane merge  : "
                f"{forced_goal_lane_change_steps} control steps "
                f"(target rear assumes {controller.forced_goal_lane_rear_braking:.1f} m/s^2 braking "
                f"within {controller.forced_goal_lane_change_distance:.1f} m)"
            )
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
        if controller.mobil_failed_this_episode:
            log(
                "  WARNING: MOBIL lane decision failed in this episode; "
                f"kept current lane. Last error: {controller.mobil_last_error}"
            )

        if enable_rendering and base_env_unwrapped.config.get("show_trajectories", False):
            save_speed_acc_curves(env, ep_idx=ep, model_path=eval_dir)
        if should_record_trajectory:
            csv_path = os.path.join(eval_dir, f"rule_mpc_ep_{ep:04d}_trajectory.csv")
            if trajectory_rows:
                with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                    writer = csv.DictWriter(
                        csv_file,
                        fieldnames=list(trajectory_rows[0].keys()),
                    )
                    writer.writeheader()
                    writer.writerows(trajectory_rows)
                log(f"  saved trajectory csv    : {csv_path}")
            else:
                log(
                    "  saved trajectory csv    : skipped "
                    f"(episode {ep} has no trajectory rows)"
                )

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
    if lane_strategy == "mobil":
        if mobil_failed_episode_count:
            log(
                "  WARNING: MOBIL silently-fallback episodes: "
                f"{mobil_failed_episode_count}/{n}; calls failed: "
                f"{mobil_failure_call_count}; last error: {mobil_last_failure_error}"
            )
        else:
            log("  MOBIL fallback failures : 0")
    log("=" * 80)

    log_file.close()
    env.close()


if __name__ == "__main__":
    # Select a test_hiro.py SAC reference condition. Environment settings are
    # loaded from its saved run_config.json; only Rule+MPC controls ego.
    EVAL_PROFILE = "sac_withPrior_newEnv_by_lane"
    # EVAL_PROFILE = "sac_withPrior_oldEnv_withWrongLanePen"

    EVAL_PROFILES: Dict[str, Dict[str, Any]] = {
        "sac_withPrior_oldEnv_withWrongLanePen": {
            "scenario_name": "multi_lane",
            "model_dir": "./results/rule_mpc_sacPrior_oldEnv_wrongLanePen",
            "reference_env_model_dir": "./models/sac_260613_withPrior_oldEnv_randomto2_wronglanePen_1e7",
            "initial_lane_id": "random",
            # Keep the string form used by test_hiro.py's explicit override.
            "goal_lane_id": "2",
            "env_overrides": {
                "goal_lane_probs": None,
            },
        },
        "sac_withPrior_newEnv_by_lane": {
            "scenario_name": "multi_lane_stop_to_int",
            "model_dir": "./results/rule_mpc_sacPrior_newEnv_by_lane",
            "reference_env_model_dir": "./models/sac_260624_withPrior_2to0",
            "goal_lane_reference_env_model_dirs": {
                0: "./models/sac_260624_withPrior_2to0",
                1: "./models/sac_260704_withPrior_2to1",
                2: "./models/sac_260622_withPrior_2to2_noGoalReshape",
            },
            # Required new-environment snapshot mapping.
            "snapshot_pool_by_goal_lane": {
                0: "debug/background_snapshot_pool_slowlane0",
                1: "debug/background_snapshot_pool_slowlane2",
                2: "debug/background_snapshot_pool_slowlane2",
            },
            "initial_lane_id": 2,
            "goal_lane_id": "random",
            "env_overrides": {
                "goal_lane_probs": None,
            },
        },
    }

    common_config: Dict[str, Any] = {
        "episodes": 300,
        # "record_episodes": [],
        # "enable_rendering": False,
        "record_episodes": [i for i in range(1, 301)],
        "record_trajectory_episodes": [i for i in range(1, 301)],
        "horizon": 15,
        "enable_forced_goal_lane_change": False,
        # "enable_forced_goal_lane_change": True,
        "forced_goal_lane_change_distance": 30.0,
        "forced_goal_lane_rear_braking": 3.0,

        "mobil_goal_lane_bias_distance": 50.0,
        # "mobil_goal_lane_bias_distance": 100.0,

        # "lane_strategy": "right_bias",
        "lane_strategy": "mobil",
        # "lane_strategy": "benefit_urgency",

        "use_low_safety_layer": True,
    }
    main(
        **common_config,
        **EVAL_PROFILES[EVAL_PROFILE],
    )
