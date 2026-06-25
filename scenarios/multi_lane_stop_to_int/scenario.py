import pickle
from pathlib import Path
from typing import Any

import numpy as np

from custom_env.envs.common.abstract import AbstractEnv
from custom_env.road.road import Road, RoadNetwork
from custom_env.road.lane import LineType, StraightLane
from custom_env.envs.common.action import Action
from custom_env import utils
from custom_env.vehicle.behavior import AggressiveIDMVehicle
from custom_env.vehicle.objects import Obstacle, Landmark

from configs.builders import get_env_config
from scenarios.goal_lane_logic import sample_goal_lane_id
from scenarios.reward_logic import goal_lane_dense_progress, wrong_lane_terminal_triggered
from util.config_utils import sync_punctual_time_with_phase_offset
from util.safety_utils import compute_ego_clear_distance_for_front_vehicle

Observation = np.ndarray

class GoalMarker(Landmark):
    """
    Visual marker for HIRO high-level goal.
    """
    def __init__(self, road, position, heading=0, velocity=0):
        super().__init__(road, position, heading, velocity)
        self.color = (255, 0, 0)  # Red color for goal
        self.LENGTH = 2.0
        self.WIDTH = 2.0
        self.collidable = False  # Purely visual, no collision physics

class BusStop(Obstacle):
    affects_traffic = False
    """
    Static rectangular bus stop placed along the road.
    length: longitudinal size.
    width: lateral size.
    """
    LENGTH = 20.0
    WIDTH = 3.0

    def __init__(self, road, position, heading=0, speed=0):
        super().__init__(road, position, heading, speed)
        self.collidable = False


class VirtualStopVehicle(Obstacle):
    """Invisible, non-collidable virtual front vehicle used for signalized stopping."""
    LENGTH = 4.5
    WIDTH = 2.0

    def __init__(self, road, position, heading=0, speed=0):
        super().__init__(road, position, heading, speed)
        self.collidable = False
        self.solid = False
        self.check_collisions = False
        self.hidden = True


class IntersectionSignalController:
    """Signal phase controller driven by signal_plan list.

    signal_plan format:
    [
        {"straight": phase_total_seconds},
        {"left": phase_total_seconds},
        ...
    ]

    Each phase total is (green + yellow). Yellow is fixed to 3s for all phases.
    """

    YELLOW_DURATION = 3.0

    DIRECTION_ALIAS = {
        "left": "left",
        "left_turn": "left",
        "turn_left": "left",
        "zuozhuan": "left",
        "左转": "left",
        "straight": "straight",
        "go_straight": "straight",
        "through": "straight",
        "zhixing": "straight",
        "直行": "straight",
    }

    def __init__(self, config: dict, lanes_count: int):
        self.lanes_count = int(lanes_count)
        self.yellow = self.YELLOW_DURATION
        self.cycle_offset = 0.0

        self.direction_lane_groups = self._parse_direction_lane_groups(config, lanes_count)

        default_plan = [{"straight": 21.0}, {"left": 15.0}]
        raw_plan = config.get("signal_plan", default_plan)
        self.signal_plan: list[tuple[str, float]] = self._parse_signal_plan(raw_plan)

    def _config_lane_to_internal(self, lane_id: int) -> int:
        # Fixed convention: lane ids are always 0..N-1 from left to right.
        return int(np.clip(int(lane_id), 0, self.lanes_count - 1))

    def _parse_direction_lane_groups(self, config: dict, lanes_count: int) -> dict[str, set[int]]:
        default_groups = {
            "left": {0} if lanes_count >= 1 else set(),
            "straight": set(range(1, lanes_count)) if lanes_count >= 2 else set(),
        }

        raw = config.get("movement_lanes", None)
        if not isinstance(raw, dict):
            return default_groups

        parsed = {"left": set(), "straight": set()}
        for key, value in raw.items():
            direction = self._normalize_direction(key)
            if direction not in parsed:
                continue
            if not isinstance(value, (list, tuple, set)):
                continue
            for lane_id in value:
                try:
                    lane_i = self._config_lane_to_internal(int(lane_id))
                except (TypeError, ValueError):
                    continue
                if 0 <= lane_i < lanes_count:
                    parsed[direction].add(lane_i)

        # Remove overlap by prioritizing left-turn membership.
        parsed["straight"] -= parsed["left"]
        all_lanes = set(range(lanes_count))
        assigned = parsed["left"] | parsed["straight"]
        parsed["straight"] |= (all_lanes - assigned)

        if not parsed["left"] and not parsed["straight"]:
            return default_groups
        return parsed

    def _normalize_direction(self, direction: str) -> str | None:
        key = str(direction).strip().lower()
        return self.DIRECTION_ALIAS.get(key, None)

    def _parse_signal_plan(self, raw_plan) -> list[tuple[str, float]]:
        phases: list[tuple[str, float]] = []
        if not isinstance(raw_plan, list):
            raw_plan = []

        for item in raw_plan:
            if not isinstance(item, dict) or len(item) != 1:
                continue
            direction_raw, total_raw = next(iter(item.items()))
            direction = self._normalize_direction(direction_raw)
            if direction is None:
                continue
            try:
                total = float(total_raw)
            except (TypeError, ValueError):
                continue
            total = max(total, self.yellow + 0.1)
            phases.append((direction, total))

        if not phases:
            phases = [("straight", 21.0), ("left", 15.0)]
        return phases

    def lane_direction(self, lane_id: int | None) -> str:
        if lane_id is not None and lane_id in self.direction_lane_groups["left"]:
            return "left"
        if lane_id is not None and lane_id in self.direction_lane_groups["straight"]:
            return "straight"
        if self.direction_lane_groups["straight"]:
            return "straight"
        return "left"

    def phase_at(self, t: float) -> dict[str, dict[str, float | str]]:
        cycle = sum(total for _, total in self.signal_plan)
        tau = (float(t) + self.cycle_offset) % cycle

        # Default all directions to red; 'remaining' means time to next phase boundary.
        phase = {
            "straight": {"state": "red", "remaining": 0.0},
            "left": {"state": "red", "remaining": 0.0},
        }

        elapsed = 0.0
        for direction, total in self.signal_plan:
            start = elapsed
            end = elapsed + total
            if start <= tau < end:
                green_dur = max(total - self.yellow, 0.1)
                local_tau = tau - start
                rem = end - tau
                if local_tau < green_dur:
                    phase[direction] = {"state": "green", "remaining": green_dur - local_tau}
                else:
                    phase[direction] = {"state": "yellow", "remaining": rem}

                other = "left" if direction == "straight" else "straight"
                phase[other] = {"state": "red", "remaining": rem}
                return phase
            elapsed = end

        # Numerical fallback at boundary
        direction, total = self.signal_plan[0]
        phase[direction] = {"state": "green", "remaining": max(total - self.yellow, 0.1)}
        return phase

    @staticmethod
    def state_color(state: str) -> tuple[int, int, int]:
        if state == "green":
            return (60, 200, 80)
        if state == "yellow":
            return (255, 205, 60)
        return (220, 70, 70)

class MultiLaneStopToIntEnv(AbstractEnv):
    """
    多车道直�?+ 顺序交通流 + 预热 + 更安全的生成逻辑
    - 道路：节�?"0" -> "1" 的四车道直路，长�?road_length
    - 环境车：从左端（x=0）按概率生成，跑到右端后删除
    - warmup：先只跑环境�?warmup_time 秒，再在公交站插�?ego
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,  # 例如 10fps，对应你�?policy_frequency=10Hz
    }
    SIGNAL_GREEN_LAUNCH_ATTRS = (
        "target_speed",
        "TIME_WANTED",
        "DISTANCE_WANTED",
        "COMFORT_ACC_MAX",
        "COMFORT_ACC_MIN",
        "DELTA",
        "imperfection",
    )
    BACKGROUND_SNAPSHOT_VEHICLE_ATTRS = (
        "target_speed",
        "TIME_WANTED",
        "DISTANCE_WANTED",
        "COMFORT_ACC_MAX",
        "COMFORT_ACC_MIN",
        "DELTA",
        "POLITENESS",
        "LANE_CHANGE_MIN_ACC_GAIN",
        "LANE_CHANGE_MAX_BRAKING_IMPOSED",
        "ACC_MAX",
        "imperfection",
    )
    BACKGROUND_SNAPSHOT_MATCH_TOLERANCE = 1e-3
    BACKGROUND_SNAPSHOT_OFFSET_KEY_SCALE = 1000

    def __init__(self, config: dict = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)
        if self.config['PERCEPTION_DISTANCE'] is not None:
            self.PERCEPTION_DISTANCE = self.config['PERCEPTION_DISTANCE']
        self._virtual_stops: dict[tuple[str, int], VirtualStopVehicle] = {}
        self._signal_controller: IntersectionSignalController | None = None
        self._signal_render_x = 0.0
        self._signal_render_y = 0.0
        # Signal clock state. Episode start alignment is controlled by config.
        self._signal_time_global = 0.0
        self._signal_episode_base = 0.0
        self._episodes_started = 0
        self._inter_episode_active = False
        self._inter_episode_remaining = 0.0
        self._queue_takeover_active = False
        self._queue_takeover_enter_count = 0
        self._background_only_sim_time = 0.0
        self._signal_green_launch_reset_vehicles: dict[int, object] = {}

    # ----------------- 配置 ----------------- #
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update(get_env_config())
        cfg.setdefault("movement_lanes", None)
        cfg.setdefault("movement_behavior_probs", None)
        cfg.setdefault("background_vehicle_respect_movement_lanes", True)
        cfg.setdefault("enable_signal_green_launch_behavior", True)
        cfg.setdefault("signal_green_launch_approach_distance", None)
        cfg.setdefault("signal_green_launch_end_margin", 5.0)
        cfg.setdefault("signal_green_launch_target_speed", None)
        cfg.setdefault("enable_signal_cycle_spawn_probability", False)
        cfg.setdefault("signal_cycle_spawn_probability", None)
        cfg.setdefault("inter_episode_as_steps", False)
        cfg.setdefault("inter_episode_step_seconds", 0.0)
        cfg.setdefault("inter_episode_zero_obs", True)
        cfg.setdefault("background_snapshot_reset", False)
        cfg.setdefault("background_snapshot_path", None)
        cfg.setdefault("background_snapshot_paths", None)
        return cfg

    # ----------------- 建路 ----------------- #
    def _create_road(self):
        self.config["single_road_network"] = True
        # 路网由三段组成：
        # 1) 进路段（0->1）：有车道线，长度由 goal x 决定
        # 2) 路口段（1->2）：长度 intersection_length，仅保留道路上下边沿�?
        # 3) 路口后段�?->3）：长度 road_length - goal_x - intersection_length
        lanes = int(self.config["lanes_count"])
        lane_w = float(StraightLane.DEFAULT_WIDTH)
        speed_limit = float(self.config["speed_limit"])
        stop_x = max(float(self._goal_longitudinal()), 0.0)
        intersection_length = max(float(self.config.get("intersection_length", 50.0)), 1.0)
        road_total = max(float(self.config.get("road_length", stop_x + intersection_length)), 1.0)

        approach_end = float(np.clip(stop_x, 0.0, road_total))
        intersection_end = float(np.clip(approach_end + intersection_length, approach_end, road_total))
        post_length = max(road_total - intersection_end, 0.0)

        net = RoadNetwork()

        def _normal_line_types(lane_id: int) -> tuple[int, int]:
            return (
                LineType.CONTINUOUS_LINE if lane_id == 0 else LineType.STRIPED,
                LineType.CONTINUOUS_LINE if lane_id == lanes - 1 else LineType.NONE,
            )

        def _intersection_line_types(lane_id: int) -> tuple[int, int]:
            # 路口区：只保留道路边沿两条横线，内部车道线不显示
            if lanes == 1:
                return (LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE)
            return (
                LineType.CONTINUOUS_LINE if lane_id == 0 else LineType.NONE,
                LineType.CONTINUOUS_LINE if lane_id == lanes - 1 else LineType.NONE,
            )

        # Main forward road (ego direction)
        if bool(self.config.get("single_road_network", True)):
            for lane_id in range(lanes):
                y = lane_id * lane_w
                net.add_lane(
                    "0",
                    "1",
                    StraightLane(
                        np.array([0.0, y]),
                        np.array([road_total, y]),
                        line_types=_normal_line_types(lane_id),
                        speed_limit=speed_limit,
                    ),
                )
            # Render metadata remains in global x, while every physical lane is
            # on one road id. This avoids segment-boundary blind spots.
            self._main_segments = [("0", "1", 0.0, road_total)]
        else:
            for lane_id in range(lanes):
                y = lane_id * lane_w
                net.add_lane(
                    "0",
                    "1",
                    StraightLane(
                        np.array([0.0, y]),
                        np.array([approach_end, y]),
                        line_types=_normal_line_types(lane_id),
                        speed_limit=speed_limit,
                    ),
                )
                net.add_lane(
                    "1",
                    "2",
                    StraightLane(
                        np.array([approach_end, y]),
                        np.array([intersection_end, y]),
                        line_types=_intersection_line_types(lane_id),
                        speed_limit=speed_limit,
                    ),
                )

                if post_length > 1e-6:
                    net.add_lane(
                        "2",
                        "3",
                        StraightLane(
                            np.array([intersection_end, y]),
                            np.array([road_total, y]),
                            line_types=_normal_line_types(lane_id),
                            speed_limit=speed_limit,
                        ),
                    )

            # Store longitudinal segmentation metadata in global x.
            self._main_segments = [
                ("0", "1", 0.0, approach_end),
                ("1", "2", approach_end, intersection_end),
            ]
            if post_length > 1e-6:
                self._main_segments.append(("2", "3", intersection_end, road_total))
        self._intersection_start_x = approach_end
        self._intersection_end_x = intersection_end
        self._road_end_x = road_total
        self._signal_controller = IntersectionSignalController(self.config, lanes)
        self._signal_render_x = 0.5 * (approach_end + intersection_end)
        self._signal_render_y = lanes * lane_w + float(self.config.get("signal_render_y_offset", 7.0))

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self.road.signal_render_items = []
        self._virtual_stops = {}
        self._create_bus_stop()
        self._update_goal_highlight_regions()

    # ----------------- reset：预�?+ 插入 ego ----------------- #
    def _reset(self):
        """
        - 第一�?reset：建�?+ 全局 warmup 交通流 + 插入 ego�?
        - 后续 reset：保留现有路网和交通流，只移除�?ego、清理一下车流，再插入新�?ego�?
        """
        first_reset = not getattr(self, "_did_global_warmup", False)
        has_previous_episode = int(getattr(self, "_episodes_started", 0)) > 0
        self._episode_initial_lane_id = self._sample_initial_lane_id()
        self._episode_goal_lane_id = self._sample_goal_lane_id()
        self._inter_episode_active = False
        self._inter_episode_remaining = 0.0
        self._signal_green_launch_reset_vehicles = {}

        if bool(self.config.get("background_snapshot_reset", False)):
            self._reset_from_background_snapshot()
            return

        # 每次都重置交通流，用于测试，以保证各个episode之间独立
        if self.config["warmup_each_episode"] is True:
            self._create_road()
            self.road.vehicles = []
            self.controlled_vehicles = []
            self._warmup(render=self.config.get("warmup_render", False))
        else:
            if first_reset:
                # ------- 第一次：建立路网 + 清空所有车�?+ 预热交通流 -------
                self._create_road()
                self.road.vehicles = []
                self.controlled_vehicles = []

                # 只跑环境�?warmup_time �?
                self._warmup(render=self.config.get("warmup_render", False))

                # 打标记：后续 reset 不再重建 & warmup
                self._did_global_warmup = True
            else:
                # 把上一回合�?ego �?road.vehicles 里移�?
                if getattr(self, "vehicle", None) is not None:
                    try:
                        self.road.vehicles.remove(self.vehicle)
                    except ValueError:
                        pass
                self.controlled_vehicles = []
                self._clear_virtual_stops()
                self._clear_background()

        self._update_goal_highlight_regions()

        # First episode: align to target offset (may be zero extra wait if already aligned).
        # Later episodes: optionally defer long alignment into inter-episode dummy env.step calls.
        if has_previous_episode and bool(self.config.get("inter_episode_as_steps", False)):
            delta = self._compute_episode_start_offset_delta(strict_next=True)
            if delta > 1e-9:
                self._begin_inter_episode_phase(delta)
                return

        self._advance_to_episode_start_offset(strict_next=has_previous_episode)
        self._signal_episode_base = float(self._signal_time_global)
        self._sync_episode_punctual_time()
        self._create_ego()
        self._episodes_started += 1
        self._mark_signal_green_launch_reset_vehicles()
        self._update_signal_virtual_stops(query_time=0.0)

    def _compute_episode_start_offset_delta(self, strict_next: bool) -> float:
        if not bool(self.config.get("align_ego_spawn_to_signal_offset", True)):
            return 0.0
        controller = getattr(self, "_signal_controller", None)
        if controller is None:
            return 0.0

        cycle = float(sum(total for _, total in controller.signal_plan))
        if cycle <= 1e-9:
            return 0.0

        target_tau = float(self.config.get("episode_start_phase_offset", 0.0)) % cycle
        tau_now = (float(self._signal_time_global) + float(controller.cycle_offset)) % cycle
        delta = (target_tau - tau_now) % cycle
        if strict_next and delta <= 1e-9:
            delta = cycle
        return max(float(delta), 0.0)

    def _current_signal_phase_offset(self) -> float:
        controller = getattr(self, "_signal_controller", None)
        if controller is None:
            return float(self.config.get("episode_start_phase_offset", 0.0))
        cycle = float(sum(total for _, total in controller.signal_plan))
        if cycle <= 1e-9:
            return float(self.config.get("episode_start_phase_offset", 0.0))
        phase_offset = (
            float(self._signal_time_global) + float(controller.cycle_offset)
        ) % cycle
        phase_offset = round(float(phase_offset), 9)
        if phase_offset >= cycle:
            phase_offset = 0.0
        return float(phase_offset)

    def _current_signal_cycle_spawn_probability(self) -> float:
        base = float(self.config.get("spawn_probability", 0.0))
        if not bool(self.config.get("enable_signal_cycle_spawn_probability", False)):
            return base

        controller = getattr(self, "_signal_controller", None)
        profile = self.config.get("signal_cycle_spawn_probability", None)
        if controller is None or not isinstance(profile, (list, tuple)):
            return base

        cycle = float(sum(total for _, total in controller.signal_plan))
        if cycle <= 1e-9:
            return base

        tau = (float(self._signal_time_global) + float(controller.cycle_offset)) % cycle
        for item in profile:
            if isinstance(item, dict):
                start_raw = item.get("start", item.get("from", item.get("begin", None)))
                end_raw = item.get("end", item.get("to", item.get("until", None)))
                prob_raw = item.get(
                    "spawn_probability",
                    item.get("probability", item.get("prob", None)),
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                start_raw, end_raw, prob_raw = item[:3]
            else:
                continue

            try:
                start = float(start_raw) % cycle
                end = float(end_raw) % cycle
                prob = float(prob_raw)
            except (TypeError, ValueError):
                continue

            if not np.isfinite(prob):
                continue

            if abs(end - start) <= 1e-9:
                in_window = True
            elif start < end:
                in_window = start <= tau < end
            else:
                in_window = tau >= start or tau < end

            if in_window:
                return float(np.clip(prob, 0.0, 1.0))

        return base

    def _sync_episode_punctual_time(self) -> None:
        actual_offset = self._current_signal_phase_offset()
        self.config["actual_episode_start_phase_offset"] = float(actual_offset)
        sync_punctual_time_with_phase_offset(
            self.config,
            phase_offset=actual_offset,
        )

    def get_punctual_time_target(self) -> float:
        return float(
            self.config.get(
                "punctual_time_target",
                self.config.get("duration", 0.0),
            )
        )

    def get_punctual_time_window(self) -> tuple[float, float]:
        window = self.config.get("punctual_time_window", [0.0, 0.0])
        return float(window[0]), float(window[1])

    def get_actual_episode_start_phase_offset(self) -> float:
        return float(
            self.config.get(
                "actual_episode_start_phase_offset",
                self._current_signal_phase_offset(),
            )
        )

    def _snapshot_cycle_seconds_from_config(self) -> float:
        raw_plan = self.config.get("signal_plan", [])
        cycle = 0.0
        if isinstance(raw_plan, list):
            for item in raw_plan:
                if not isinstance(item, dict):
                    continue
                for value in item.values():
                    try:
                        cycle += float(value)
                    except (TypeError, ValueError):
                        continue
        return max(float(cycle), 0.0)

    @staticmethod
    def _snapshot_scalar(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _snapshot_lane_index(value: Any) -> tuple[str, str, int | None] | None:
        if value is None:
            return None
        if not isinstance(value, (list, tuple)) or len(value) < 3:
            return None
        lane_id = None if value[2] is None else int(value[2])
        return (str(value[0]), str(value[1]), lane_id)

    def _snapshot_route(self, value: Any) -> list[tuple[str, str, int | None]] | None:
        if value is None:
            return None
        route: list[tuple[str, str, int | None]] = []
        for lane_index in value:
            parsed = self._snapshot_lane_index(lane_index)
            if parsed is not None:
                route.append(parsed)
        return route

    def _snapshot_config_signature(self) -> dict[str, Any]:
        keys = [
            "lanes_count",
            "road_length",
            "speed_limit",
            "start_longitudinal",
            "goal_longitudinal",
            "intersection_length",
            "flow_speed_range",
            "speed_distribution",
            "spawn_min_gap",
            "spawn_min_t_headway",
            "spawn_check_adjacent_cutins",
            "spawn_adjacent_cutin_front_gap",
            "spawn_adjacent_cutin_back_gap",
            "movement_lanes",
            "movement_behavior_probs",
            "signal_plan",
            "behavior_vehicle_types",
            "behavior_lane_probs",
            "background_vehicle_respect_movement_lanes",
            "enable_signal_green_launch_behavior",
            "signal_green_launch_approach_distance",
            "signal_green_launch_end_margin",
            "signal_green_launch_target_speed",
            "enable_signal_cycle_spawn_probability",
            "signal_cycle_spawn_probability",
        ]
        if not bool(self.config.get("enable_signal_cycle_spawn_probability", False)):
            keys.append("spawn_probability")
        return {key: self.config.get(key, None) for key in keys}

    def _vehicle_to_background_snapshot(self, vehicle) -> dict[str, Any]:
        original = getattr(vehicle, "_signal_green_launch_original", None)
        attrs: dict[str, Any] = {}
        for attr in self.BACKGROUND_SNAPSHOT_VEHICLE_ATTRS:
            if isinstance(original, dict) and attr in original:
                value = original[attr]
            elif hasattr(vehicle, attr):
                value = getattr(vehicle, attr)
            else:
                continue
            attrs[attr] = self._snapshot_scalar(value)

        action = dict(getattr(vehicle, "action", {}) or {})
        action = {
            str(k): float(v)
            for k, v in action.items()
            if isinstance(v, (int, float, np.floating))
        }
        cls = vehicle.__class__
        return {
            "class_path": f"{cls.__module__}.{cls.__qualname__}",
            "position": np.asarray(vehicle.position, dtype=float).copy(),
            "heading": float(getattr(vehicle, "heading", 0.0)),
            "speed": float(getattr(vehicle, "speed", 0.0)),
            "lane_index": self._snapshot_lane_index(getattr(vehicle, "lane_index", None)),
            "target_lane_index": self._snapshot_lane_index(getattr(vehicle, "target_lane_index", None)),
            "target_speed": float(getattr(vehicle, "target_speed", getattr(vehicle, "speed", 0.0))),
            "route": self._snapshot_route(getattr(vehicle, "route", None)),
            "enable_lane_change": bool(getattr(vehicle, "enable_lane_change", True)),
            "timer": float(getattr(vehicle, "timer", 0.0)),
            "movement_direction": getattr(vehicle, "movement_direction", None),
            "vid": int(getattr(vehicle, "vid", -1)),
            "action": action,
            "crashed": bool(getattr(vehicle, "crashed", False)),
            "attrs": attrs,
        }

    def export_background_snapshot(self) -> dict[str, Any]:
        """Return a restorable snapshot of background traffic at the current signal phase."""
        if not hasattr(self, "road") or self.road is None:
            raise RuntimeError("Cannot export a background snapshot before the road exists")

        self._clear_background()
        controlled = tuple(getattr(self, "controlled_vehicles", []) or ())
        vehicles = [
            self._vehicle_to_background_snapshot(vehicle)
            for vehicle in list(getattr(self.road, "vehicles", []) or [])
            if not any(vehicle is controlled_vehicle for controlled_vehicle in controlled)
        ]
        return {
            "version": 1,
            "phase_offset": float(self._current_signal_phase_offset()),
            "signal_time_global": float(self._signal_time_global),
            "vid": int(self.config.get("vid", 0)),
            "background_count": int(len(vehicles)),
            "config_signature": self._snapshot_config_signature(),
            "vehicles": vehicles,
        }

    def _background_snapshot_offset_key(self, offset: float) -> str:
        return str(int(round(float(offset) * self.BACKGROUND_SNAPSHOT_OFFSET_KEY_SCALE)))

    def _build_background_snapshot_index(
        self,
        snapshots: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        index: dict[str, list[dict[str, Any]]] = {}
        for snapshot in snapshots:
            if not isinstance(snapshot, dict) or "phase_offset" not in snapshot:
                continue
            key = self._background_snapshot_offset_key(float(snapshot["phase_offset"]))
            index.setdefault(key, []).append(snapshot)
        return index

    def _validate_background_snapshot_config_signature(self, saved: Any) -> None:
        if not isinstance(saved, dict):
            raise ValueError("Background snapshot pool is missing config_signature")

        current = self._snapshot_config_signature()
        ignored_keys: set[str] = {"enable_signal_green_launch_behavior"}
        try:
            if len(self._background_snapshot_paths()) > 1:
                ignored_keys.add("behavior_lane_probs")
        except ValueError:
            pass
        mismatches: list[str] = []
        for key, current_value in current.items():
            if key in ignored_keys:
                continue
            if saved.get(key, None) != current_value:
                mismatches.append(key)
        if mismatches:
            detail = ", ".join(mismatches[:12])
            if len(mismatches) > 12:
                detail += f", ... (+{len(mismatches) - 12} more)"
            raise ValueError(
                "Background snapshot pool config does not match current env config: "
                + detail
            )

    def _background_snapshot_paths(self) -> list[Path]:
        path_raw = self.config.get("background_snapshot_paths", None)
        if path_raw is None:
            path_raw = self.config.get("background_snapshot_path", None)
        if not path_raw:
            raise ValueError(
                "background_snapshot_reset=True requires background_snapshot_path "
                "or background_snapshot_paths"
            )
        if isinstance(path_raw, (list, tuple)):
            paths = [Path(p) for p in path_raw if p]
        else:
            paths = [Path(path_raw)]
        if not paths:
            raise ValueError("background_snapshot_paths must contain at least one path")
        return paths

    def _load_background_snapshot_pool(self, path_raw: Any = None) -> dict[str, list[dict[str, Any]]]:
        """Load a legacy single-file pool and keep its full in-memory index."""
        if path_raw is None:
            path_raw = self.config.get("background_snapshot_path", None)
        if not path_raw:
            raise ValueError("background_snapshot_reset=True requires background_snapshot_path")
        path = str(Path(path_raw))
        cached_path = getattr(self, "_background_snapshot_cache_path", None)
        cached_index = getattr(self, "_background_snapshot_index", None)
        if cached_path == path and isinstance(cached_index, dict):
            return cached_index

        with Path(path).open("rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            self._validate_background_snapshot_config_signature(data.get("config_signature", None))
        snapshot_index = None
        if isinstance(data, dict) and isinstance(data.get("snapshots_by_offset", None), dict):
            snapshot_index = {
                str(key): list(value)
                for key, value in data["snapshots_by_offset"].items()
                if isinstance(value, list)
            }
        if isinstance(data, dict) and "snapshots" in data:
            snapshots = data["snapshots"]
        else:
            snapshots = data
        if not isinstance(snapshots, list) or not snapshots:
            raise ValueError(f"No background snapshots found in {path}")
        if not snapshot_index:
            snapshot_index = self._build_background_snapshot_index(snapshots)
        if not snapshot_index:
            raise ValueError(f"No indexed background snapshots found in {path}")

        self._background_snapshot_cache_path = path
        self._background_snapshot_pool = snapshots
        self._background_snapshot_index = snapshot_index
        return snapshot_index

    def _load_background_snapshot_shard_meta(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        path_key = str(path)
        cached_path = getattr(self, "_background_snapshot_meta_cache_path", None)
        cached_meta = getattr(self, "_background_snapshot_meta", None)
        if cached_path == path_key and isinstance(cached_meta, dict):
            return cached_meta

        meta_path = path / "meta.pkl"
        with meta_path.open("rb") as f:
            meta = pickle.load(f)
        if not isinstance(meta, dict) or meta.get("format") not in {
            "offset_shards",
            "offset_chunk_shards",
        }:
            raise ValueError(f"Invalid sharded background snapshot pool metadata in {meta_path}")
        if int(meta.get("offset_key_scale", self.BACKGROUND_SNAPSHOT_OFFSET_KEY_SCALE)) != int(
            self.BACKGROUND_SNAPSHOT_OFFSET_KEY_SCALE
        ):
            raise ValueError(
                "Background snapshot pool offset_key_scale does not match current env"
            )
        self._validate_background_snapshot_config_signature(meta.get("config_signature", None))

        self._background_snapshot_meta_cache_path = path_key
        self._background_snapshot_meta = meta
        return meta

    def _load_background_snapshot_shard(self, path: Path, offset_key: str) -> list[dict[str, Any]]:
        path = Path(path)
        path_key = str(path)
        cached_path = getattr(self, "_background_snapshot_shard_cache_path", None)
        cached_key = getattr(self, "_background_snapshot_shard_cache_key", None)
        cached_snapshots = getattr(self, "_background_snapshot_shard_snapshots", None)
        if cached_path == path_key and cached_key == offset_key and isinstance(cached_snapshots, list):
            return cached_snapshots

        meta = self._load_background_snapshot_shard_meta(path)
        shards = meta.get("shards", {})
        if not isinstance(shards, dict) or offset_key not in shards:
            raise ValueError(f"No background snapshot shard for offset key {offset_key} in {path}")
        shard_info = shards[offset_key]
        if not isinstance(shard_info, dict) or not shard_info.get("file"):
            raise ValueError(f"Invalid background snapshot shard metadata for offset key {offset_key}")

        shard_path = path / str(shard_info["file"])
        with shard_path.open("rb") as f:
            shard = pickle.load(f)
        if not isinstance(shard, dict):
            raise ValueError(f"Invalid background snapshot shard payload in {shard_path}")
        self._validate_background_snapshot_config_signature(shard.get("config_signature", None))
        snapshots = shard.get("snapshots", None)
        if not isinstance(snapshots, list) or not snapshots:
            raise ValueError(f"No background snapshots found in shard {shard_path}")

        self._background_snapshot_shard_cache_path = path_key
        self._background_snapshot_shard_cache_key = offset_key
        self._background_snapshot_shard_snapshots = snapshots
        return snapshots

    def _load_background_snapshot_chunk(
        self,
        path: Path,
        offset_key: str,
        chunk_info: dict[str, Any],
    ) -> list[dict[str, Any]]:
        path = Path(path)
        path_key = str(path)
        chunk_file = str(chunk_info.get("file", ""))
        if not chunk_file:
            raise ValueError(f"Invalid background snapshot chunk metadata for offset key {offset_key}")

        cached_path = getattr(self, "_background_snapshot_chunk_cache_path", None)
        cached_key = getattr(self, "_background_snapshot_chunk_cache_key", None)
        cached_file = getattr(self, "_background_snapshot_chunk_cache_file", None)
        cached_snapshots = getattr(self, "_background_snapshot_chunk_snapshots", None)
        if (
            cached_path == path_key
            and cached_key == offset_key
            and cached_file == chunk_file
            and isinstance(cached_snapshots, list)
        ):
            return cached_snapshots

        chunk_path = path / chunk_file
        with chunk_path.open("rb") as f:
            chunk = pickle.load(f)
        if not isinstance(chunk, dict):
            raise ValueError(f"Invalid background snapshot chunk payload in {chunk_path}")
        self._validate_background_snapshot_config_signature(chunk.get("config_signature", None))
        snapshots = chunk.get("snapshots", None)
        if not isinstance(snapshots, list) or not snapshots:
            raise ValueError(f"No background snapshots found in chunk {chunk_path}")

        self._background_snapshot_chunk_cache_path = path_key
        self._background_snapshot_chunk_cache_key = offset_key
        self._background_snapshot_chunk_cache_file = chunk_file
        self._background_snapshot_chunk_snapshots = snapshots
        return snapshots

    def _sample_background_snapshot_from_chunked_shard(
        self,
        path: Path,
        offset_key: str,
        shard_info: dict[str, Any],
    ) -> dict[str, Any]:
        chunks = shard_info.get("chunks", None)
        if not isinstance(chunks, list) or not chunks:
            raise ValueError(f"No background snapshot chunks for offset key {offset_key}")

        total = int(shard_info.get("count", 0))
        if total <= 0:
            total = sum(int(chunk.get("count", 0)) for chunk in chunks if isinstance(chunk, dict))
        if total <= 0:
            raise ValueError(f"Empty background snapshot chunk shard for offset key {offset_key}")

        selected = int(self.np_random.integers(total))
        remaining = selected
        for chunk_info in chunks:
            if not isinstance(chunk_info, dict):
                continue
            count = int(chunk_info.get("count", 0))
            if count <= 0:
                continue
            if remaining >= count:
                remaining -= count
                continue
            snapshots = self._load_background_snapshot_chunk(path, offset_key, chunk_info)
            if remaining >= len(snapshots):
                # Metadata drift should not happen, but keep the failure explicit.
                raise ValueError(
                    f"Background snapshot chunk count mismatch for offset key {offset_key}"
                )
            return snapshots[remaining]

        raise ValueError(f"Failed to sample background snapshot for offset key {offset_key}")

    def _load_background_snapshot_candidates_from_path(
        self,
        path: Path,
        target: float,
        cycle: float,
        tolerance: float,
    ) -> list[dict[str, Any]]:
        target_key = int(round(float(target) * self.BACKGROUND_SNAPSHOT_OFFSET_KEY_SCALE))

        if path.is_dir():
            exact_key = str(target_key)
            meta = self._load_background_snapshot_shard_meta(path)
            shards = meta.get("shards", {})
            if isinstance(shards, dict) and exact_key in shards:
                shard_info = shards[exact_key]
                if isinstance(shard_info, dict) and shard_info.get("format") == "chunks":
                    return [
                        self._sample_background_snapshot_from_chunked_shard(
                            path, exact_key, shard_info
                        )
                    ]
                return self._load_background_snapshot_shard(path, exact_key)
            # Compatibility guard for pools created with tiny floating-point drift.
            candidates: list[dict[str, Any]] = []
            for key in (str(target_key - 1), str(target_key + 1)):
                if not isinstance(shards, dict) or key not in shards:
                    continue
                shard_info = shards[key]
                if isinstance(shard_info, dict) and shard_info.get("format") == "chunks":
                    snapshot = self._sample_background_snapshot_from_chunked_shard(
                        path, key, shard_info
                    )
                    if self._snapshot_phase_diff(
                        float(snapshot["phase_offset"]), target, cycle
                    ) <= tolerance:
                        candidates.append(snapshot)
                    continue
                shard = self._load_background_snapshot_shard(path, key)
                candidates.extend(
                    snapshot
                    for snapshot in shard
                    if self._snapshot_phase_diff(
                        float(snapshot["phase_offset"]), target, cycle
                    )
                    <= tolerance
                )
            return candidates

        snapshot_index = self._load_background_snapshot_pool(path)
        candidates = snapshot_index.get(str(target_key), [])
        if not candidates:
            neighbor_keys = (str(target_key - 1), str(target_key + 1))
            candidates = [
                snapshot
                for key in neighbor_keys
                for snapshot in snapshot_index.get(key, [])
                if self._snapshot_phase_diff(float(snapshot["phase_offset"]), target, cycle) <= tolerance
            ]
        return candidates

    def _load_background_snapshot_candidates(
        self,
        target: float,
        cycle: float,
        tolerance: float,
    ) -> list[dict[str, Any]]:
        paths = self._background_snapshot_paths()
        if len(paths) == 1:
            return self._load_background_snapshot_candidates_from_path(
                paths[0], target, cycle, tolerance
            )

        first = int(self.np_random.integers(len(paths)))
        order = [first] + [idx for idx in range(len(paths)) if idx != first]
        for idx in order:
            candidates = self._load_background_snapshot_candidates_from_path(
                paths[idx], target, cycle, tolerance
            )
            if candidates:
                return candidates
        return []

    def _snapshot_phase_diff(self, phase_a: float, phase_b: float, cycle: float) -> float:
        if cycle <= 1e-9:
            return abs(float(phase_a) - float(phase_b))
        raw = abs((float(phase_a) - float(phase_b)) % cycle)
        return min(raw, cycle - raw)

    def _sample_background_snapshot(self) -> dict[str, Any]:
        cycle = self._snapshot_cycle_seconds_from_config()
        target = float(self.config.get("episode_start_phase_offset", 0.0))
        if cycle > 1e-9:
            target %= cycle
        tolerance = float(self.BACKGROUND_SNAPSHOT_MATCH_TOLERANCE)

        candidates = self._load_background_snapshot_candidates(target, cycle, tolerance)
        if not candidates:
            raise ValueError(
                "No background snapshot matches "
                f"episode_start_phase_offset={target:.6f} within {tolerance:.6f}s"
            )

        idx = int(self.np_random.integers(len(candidates)))
        return candidates[idx]

    def _vehicle_from_background_snapshot(self, data: dict[str, Any]):
        class_path = str(data["class_path"])
        vehicle_cls = utils.class_from_path(class_path)
        position = np.asarray(data.get("position", [0.0, 0.0]), dtype=float).copy()
        heading = float(data.get("heading", 0.0))
        speed = float(data.get("speed", 0.0))
        lane_index = self._snapshot_lane_index(data.get("lane_index", None))
        target_lane_index = self._snapshot_lane_index(data.get("target_lane_index", lane_index))
        route = self._snapshot_route(data.get("route", None))
        target_speed = float(data.get("target_speed", speed))
        enable_lane_change = bool(data.get("enable_lane_change", True))
        timer = float(data.get("timer", 0.0))

        try:
            vehicle = vehicle_cls(
                self.road,
                position,
                heading,
                speed,
                target_lane_index=target_lane_index,
                target_speed=target_speed,
                route=route,
                enable_lane_change=enable_lane_change,
                timer=timer,
            )
        except TypeError:
            try:
                vehicle = vehicle_cls(
                    self.road,
                    position,
                    heading,
                    speed,
                    target_lane_index=target_lane_index,
                    target_speed=target_speed,
                    route=route,
                )
            except TypeError:
                vehicle = vehicle_cls(self.road, position, heading, speed)

        if lane_index is None:
            lane_index = self.road.network.get_closest_lane_index(position, heading)
        vehicle.lane_index = lane_index
        vehicle.lane = self.road.network.get_lane(lane_index)
        if target_lane_index is not None:
            vehicle.target_lane_index = target_lane_index
        if route is not None:
            vehicle.route = list(route)
        if hasattr(vehicle, "enable_lane_change"):
            vehicle.enable_lane_change = enable_lane_change
        if hasattr(vehicle, "timer"):
            vehicle.timer = timer
        vehicle.target_speed = target_speed
        action = {"steering": 0.0, "acceleration": 0.0}
        action.update(dict(data.get("action", {}) or {}))
        vehicle.action = action
        vehicle.crashed = bool(data.get("crashed", False))
        vehicle.impact = None
        if data.get("movement_direction", None) is not None:
            vehicle.movement_direction = str(data["movement_direction"])
        if int(data.get("vid", -1)) >= 0:
            vehicle.vid = int(data["vid"])
        for attr, value in dict(data.get("attrs", {}) or {}).items():
            setattr(vehicle, str(attr), self._snapshot_scalar(value))
        return vehicle

    def _reset_from_background_snapshot(self) -> bool:
        snapshot = self._sample_background_snapshot()
        if not isinstance(snapshot, dict):
            raise ValueError("Background snapshot entries must be dictionaries")

        snapshot_signature = snapshot.get("config_signature", None)
        if isinstance(snapshot_signature, dict) and "behavior_lane_probs" in snapshot_signature:
            self.config["behavior_lane_probs"] = snapshot_signature["behavior_lane_probs"]

        self._create_road()
        self.road.vehicles = []
        self.controlled_vehicles = []
        self._clear_virtual_stops()

        cycle = self._snapshot_cycle_seconds_from_config()
        if "signal_time_global" in snapshot:
            self._signal_time_global = float(snapshot["signal_time_global"])
        else:
            phase = float(snapshot.get("phase_offset", self.config.get("episode_start_phase_offset", 0.0)))
            cycle_offset = float(getattr(self._signal_controller, "cycle_offset", 0.0))
            self._signal_time_global = (phase - cycle_offset) % cycle if cycle > 1e-9 else phase

        max_vid = int(snapshot.get("vid", self.config.get("vid", 0)))
        for vehicle_data in list(snapshot.get("vehicles", []) or []):
            vehicle = self._vehicle_from_background_snapshot(vehicle_data)
            self.road.vehicles.append(vehicle)
            max_vid = max(max_vid, int(getattr(vehicle, "vid", -1)))
        self.config["vid"] = max(int(self.config.get("vid", 0)), max_vid)

        self.time = 0.0
        self.steps = 0
        self._signal_episode_base = float(self._signal_time_global)
        self._sync_episode_punctual_time()
        self._create_ego()
        self._episodes_started += 1
        self._did_global_warmup = True
        self._mark_signal_green_launch_reset_vehicles()
        self._update_signal_virtual_stops(query_time=0.0)
        return True

    def _inter_episode_step_seconds(self) -> float:
        configured = float(self.config.get("inter_episode_step_seconds", 0.0))
        if configured > 1e-9:
            return configured
        return 1.0 / float(self.config["policy_frequency"])

    def _dummy_action(self):
        shape = getattr(self.action_space, "shape", None)
        if shape is not None:
            return np.zeros(shape, dtype=np.float32)
        return 0

    def _dummy_observation(self) -> np.ndarray:
        if not bool(self.config.get("inter_episode_zero_obs", True)):
            return self.observation_type.observe()
        shape = getattr(self.observation_space, "shape", None)
        if shape is not None:
            dtype = getattr(self.observation_space, "dtype", np.float32)
            return np.zeros(shape, dtype=dtype)
        obs = self.observation_type.observe()
        return np.zeros_like(np.asarray(obs, dtype=np.float32))

    def _begin_inter_episode_phase(self, seconds: float) -> None:
        self._inter_episode_active = True
        self._inter_episode_remaining = max(float(seconds), 0.0)

        lane_id = self._initial_lane_id()
        lane_index = ("0", "1", int(lane_id))
        lane = self.road.network.get_lane(lane_index)
        position = lane.position(0.0, 0.0)
        heading = lane.heading_at(0.0)

        dummy = self.action_type.vehicle_class(self.road, position, heading, 0.0)
        dummy.lane = lane
        dummy.lane_index = lane_index
        self.vehicle = dummy

        self._last_speed = 0.0
        self._last_longitudinal = 0.0
        self._last_lane_id = int(lane_id)
        self._has_arrived = False
        self._arrival_time = None

        self._update_signal_virtual_stops(query_time=0.0)

    def _finish_inter_episode_phase(self) -> None:
        self._inter_episode_active = False
        self._inter_episode_remaining = 0.0
        self.time = 0.0
        self.steps = 0
        self._signal_episode_base = float(self._signal_time_global)
        self._sync_episode_punctual_time()
        self._create_ego()
        self._episodes_started += 1
        self._mark_signal_green_launch_reset_vehicles()
        self._update_signal_virtual_stops(query_time=0.0)

    def _nearest_same_lane_front(self):
        ego = getattr(self, "vehicle", None)
        if ego is None or not hasattr(self, "road") or self.road is None:
            return None, None
        ego_lane = getattr(ego, "lane_index", None)
        if ego_lane is None or len(ego_lane) < 3:
            return None, None

        ego_x = float(np.asarray(ego.position, dtype=float)[0])
        best_vehicle = None
        best_gap = None
        for vehicle in self.road.vehicles:
            if vehicle is ego or any(
                vehicle is controlled for controlled in self.controlled_vehicles
            ):
                continue
            lane_index = getattr(vehicle, "lane_index", None)
            if lane_index is None or len(lane_index) < 3:
                continue
            if tuple(lane_index[:3]) != tuple(ego_lane[:3]):
                continue
            gap = float(np.asarray(vehicle.position, dtype=float)[0]) - ego_x
            if gap <= 0.0:
                continue
            if best_gap is None or gap < best_gap:
                best_vehicle = vehicle
                best_gap = gap
        return best_vehicle, best_gap

    def _queue_takeover_enter_candidate(self) -> bool:
        if not bool(self.config.get("enable_queue_takeover", False)):
            return False
        ego = getattr(self, "vehicle", None)
        if ego is None:
            return False

        stop_x = float(self._goal_longitudinal())

        front, front_gap = self._nearest_same_lane_front()
        if front is None or front_gap is None:
            return False

        front_x = float(np.asarray(front.position, dtype=float)[0])
        front_speed = max(float(getattr(front, "speed", 0.0)), 0.0)
        return bool(
            front_gap <= float(self.config.get("queue_takeover_front_gap", 30.0))
            and front_speed <= float(self.config.get("queue_takeover_front_speed", 2.0))
            and front_x
            <= stop_x + float(self.config.get("queue_takeover_release_x_margin", 3.0))
        )

    def _update_queue_takeover_state(self) -> None:
        if not bool(self.config.get("enable_queue_takeover", False)):
            self._queue_takeover_active = False
            self._queue_takeover_enter_count = 0
            return

        ego = getattr(self, "vehicle", None)
        if ego is None:
            return
        if bool(getattr(self, "_queue_takeover_active", False)):
            release_x = float(self._goal_longitudinal()) + float(
                self.config.get("queue_takeover_release_x_margin", 3.0)
            )
            if float(np.asarray(ego.position, dtype=float)[0]) >= release_x:
                self._queue_takeover_active = False
                self._queue_takeover_enter_count = 0
            return

        if self._queue_takeover_enter_candidate():
            self._queue_takeover_enter_count += 1
        else:
            self._queue_takeover_enter_count = 0
        required = max(int(self.config.get("queue_takeover_enter_steps", 3)), 1)
        if self._queue_takeover_enter_count >= required:
            self._queue_takeover_active = True

    def get_queue_takeover_active(self) -> bool:
        return bool(
            self.config.get("enable_queue_takeover", False)
            and getattr(self, "_queue_takeover_active", False)
        )

    def get_queue_takeover_action(self) -> np.ndarray:
        """Return a lane-keeping IDM-like action for the latched queue phase."""
        ego = getattr(self, "vehicle", None)
        if ego is None:
            return np.zeros(2, dtype=np.float32)

        speed = max(float(getattr(ego, "speed", 0.0)), 0.0)
        desired_speed = max(
            float(self.config.get("queue_takeover_desired_speed", 10.0)),
            1e-3,
        )
        max_accel = max(
            float(self.config.get("queue_takeover_max_accel", 2.0)),
            1e-3,
        )
        comfort_brake = max(
            float(self.config.get("queue_takeover_comfort_brake", 3.0)),
            1e-3,
        )
        min_gap = max(float(self.config.get("queue_takeover_min_gap", 4.0)), 0.0)
        time_headway = max(
            float(self.config.get("queue_takeover_time_headway", 1.2)),
            0.0,
        )

        obstacle_gap = None
        obstacle_speed = 0.0
        front, front_gap = self._nearest_same_lane_front()
        if front is not None and front_gap is not None:
            obstacle_gap = max(float(front_gap), 1e-3)
            obstacle_speed = max(float(getattr(front, "speed", 0.0)), 0.0)

        signal_is_green, _ = self.get_hiro_signal_features()
        if signal_is_green < 0.5:
            stop_gap = float(self._goal_longitudinal()) - float(
                np.asarray(ego.position, dtype=float)[0]
            )
            if stop_gap > 0.0 and (obstacle_gap is None or stop_gap < obstacle_gap):
                obstacle_gap = max(stop_gap, 1e-3)
                obstacle_speed = 0.0

        free_road_term = (speed / desired_speed) ** 4
        interaction_term = 0.0
        if obstacle_gap is not None:
            closing_speed = speed - obstacle_speed
            dynamic_gap = (
                min_gap
                + speed * time_headway
                + speed
                * closing_speed
                / (2.0 * np.sqrt(max_accel * comfort_brake))
            )
            desired_gap = max(dynamic_gap, min_gap)
            interaction_term = (desired_gap / obstacle_gap) ** 2

        acc_phys = max_accel * (1.0 - free_road_term - interaction_term)
        acc_range = self.config.get("action", {}).get(
            "acceleration_range",
            [-5.0, 5.0],
        )
        acc_min, acc_max = float(acc_range[0]), float(acc_range[1])
        acc_phys = float(np.clip(acc_phys, acc_min, acc_max))
        acc_norm = 2.0 * (acc_phys - acc_min) / max(acc_max - acc_min, 1e-6) - 1.0
        return np.asarray([0.0, np.clip(acc_norm, -1.0, 1.0)], dtype=np.float32)

    def _step_inter_episode_dummy(self, action):
        del action
        chunk = min(self._inter_episode_step_seconds(), float(self._inter_episode_remaining))
        if chunk > 1e-9:
            self._simulate_background_for(chunk)
            self._inter_episode_remaining = max(float(self._inter_episode_remaining - chunk), 0.0)

        finished = bool(self._inter_episode_remaining <= 1e-9)
        if finished:
            self._finish_inter_episode_phase()
            obs = self.observation_type.observe()
        else:
            obs = self._dummy_observation()

        info = {
            "speed": float(getattr(self.vehicle, "speed", 0.0)),
            "crashed": bool(getattr(self.vehicle, "crashed", False)),
            "action": self._dummy_action(),
            "reward_components": {},
            "inter_episode": True,
            "skip_replay": True,
            "next_obs_is_dummy": not finished,
            "queue_takeover_active": False,
            "env_diagnostics": self._env_diagnostics(),
        }
        return obs, 0.0, False, False, info

    def _advance_to_episode_start_offset(self, strict_next: bool) -> None:
        """Keep background traffic evolving and align spawn time to configured signal phase offset."""
        delta = self._compute_episode_start_offset_delta(strict_next=strict_next)
        if delta <= 1e-9:
            return

        self._simulate_background_for(delta)
        self._background_only_sim_time += delta

    def _simulate_background_for(self, seconds: float) -> None:
        """Simulate road/background traffic for given wall-clock seconds without ego control."""
        remain = max(float(seconds), 0.0)
        if remain <= 0.0:
            return

        sim_freq = float(self.config["simulation_frequency"])
        base_dt = 1.0 / sim_freq

        while remain > 1e-9:
            dt = min(base_dt, remain)
            self._update_signal_virtual_stops(query_time=None)
            self._clear_background()
            base_spawn_p = self._current_signal_cycle_spawn_probability()
            spawn_p = base_spawn_p * (dt / base_dt)
            self._spawn_background(spawn_probability=spawn_p)
            self.road.act()
            self.road.step(dt)
            self._signal_time_global += dt
            remain -= dt

        self._clear_background()
        self._update_signal_virtual_stops(query_time=None)

    def _warmup(self, render: bool = False):
        """Run background-only warmup before inserting ego."""
        warmup_time = float(self.config["warmup_time"])
        sim_freq = float(self.config["simulation_frequency"])
        sim_dt = 1.0 / sim_freq
        steps = int(warmup_time * sim_freq)

        avg_speeds = []
        times = []
        for k in range(steps):
            self._update_signal_virtual_stops(query_time=k / sim_freq)
            self._clear_background()
            self._spawn_background()

            speeds = [
                float(v.speed)
                for v in self.road.vehicles
                if not getattr(v, "crashed", False)
            ]
            if speeds:
                avg_speed = float(np.mean(speeds))
            else:
                avg_speed = 0.0
            t = k / sim_freq  # 当前 warmup 时间 [s]
            times.append(t)
            avg_speeds.append(avg_speed)

            self.road.act()
            self.road.step(sim_dt)
            self._signal_time_global += sim_dt
            # 调试模式：在 reset 期间也渲�?warmup 的画�?
            if render and self.render_mode is not None:
                self.render()

        # 再做一次清理，避免 warmup 结束时残�?crash 车辆
        self._clear_virtual_stops()
        self._clear_background()

        self._background_only_sim_time += float(steps) * sim_dt
        self._warmup_times = np.asarray(times, dtype=float)
        self._warmup_avg_speeds = np.asarray(avg_speeds, dtype=float)

    # ----------------- RL step：在 AbstractEnv 的基础上维护车�?----------------- #
    def step(self, action):
        if bool(getattr(self, "_inter_episode_active", False)):
            return self._step_inter_episode_dummy(action)

        # 在当前决策步生效的信号相位（�?_simulate �?IDM 使用�?
        dt = 1.0 / float(self.config["policy_frequency"])
        self._update_signal_virtual_stops(query_time=self.time + dt)

        # �?AbstractEnv 完成 ego 控制 + 仿真
        obs, reward, terminated, truncated, info = super().step(action)

        # Sync persistent signal clock to the current episode local time after step().
        self._signal_time_global = float(self._signal_episode_base + self.time)

        # 维持渲染与下一步前的一致信号状�?
        self._update_signal_virtual_stops(query_time=self.time)
        queue_takeover_was_active = self.get_queue_takeover_active()
        self._update_queue_takeover_state()
        queue_takeover_active = self.get_queue_takeover_active()
        queue_takeover_terminal = bool(
            self.config.get("terminate_on_queue_takeover", False)
            and queue_takeover_active
            and not queue_takeover_was_active
            and not bool(terminated or truncated)
        )
        if queue_takeover_terminal:
            terminated = True

        # 把“加权后的分项奖励”塞�?info，方�?callback �?infos 里读
        weighted = getattr(self, "_last_weighted_rewards", None)
        if isinstance(info, dict) and weighted is not None:
            info["reward_components"] = dict(weighted)

        next_obs_is_dummy = False
        use_inter_episode_steps = (
            bool(self.config.get("inter_episode_as_steps", False))
            and not bool(self.config.get("background_snapshot_reset", False))
        )
        if bool(terminated or truncated) and use_inter_episode_steps:
            pending = self._compute_episode_start_offset_delta(strict_next=True)
            next_obs_is_dummy = bool(pending > 1e-9)
            if isinstance(info, dict):
                info["inter_episode_pending_seconds"] = float(max(pending, 0.0))

        if isinstance(info, dict):
            info["inter_episode"] = False
            info["skip_replay"] = False
            info["next_obs_is_dummy"] = bool(next_obs_is_dummy)
            info["queue_takeover_active"] = bool(queue_takeover_active)
            info["queue_takeover_terminal"] = bool(queue_takeover_terminal)
            info["env_diagnostics"] = self._env_diagnostics()
            if bool(terminated or truncated):
                info["terminal_signal_features"] = tuple(self.get_hiro_signal_features())

        sim_freq = float(self.config["simulation_frequency"])
        pol_freq = float(self.config["policy_frequency"])

        # 在每个决策步之后，更新一次车流：清除驶离 & crashed，按概率增车
        self._clear_background()
        self._spawn_background(self._current_signal_cycle_spawn_probability() * (sim_freq / pol_freq))    # TODO: 完善增车策略，现在是按policy_freq集总生成，不是按simu_freq生成

        return obs, reward, terminated, truncated, info

    def _env_diagnostics(self) -> dict:
        bg = [v for v in self.road.vehicles if v not in self.controlled_vehicles]
        valid = [v for v in bg if not getattr(v, "crashed", False)]
        speeds = [
            float(getattr(v, "speed", np.linalg.norm(getattr(v, "velocity", np.zeros(2)))))
            for v in valid
        ]
        xs = [float(np.asarray(getattr(v, "position", [np.nan, np.nan]), dtype=float)[0]) for v in valid]
        goal_x = float(self._goal_longitudinal())
        near_goal_low = sum(1 for x, s in zip(xs, speeds) if goal_x - 40.0 <= x <= goal_x + 10.0 and s < 2.0)
        ego_pos = np.asarray(getattr(self.vehicle, "position", [np.nan, np.nan]), dtype=float)
        lane_index = getattr(self.vehicle, "lane_index", (None, None, -1))
        signal_is_green, signal_remaining = self.get_hiro_signal_features()
        punctual_window = self.get_punctual_time_window()
        return {
            "time": float(getattr(self, "time", 0.0)),
            "signal_time_global": float(getattr(self, "_signal_time_global", np.nan)),
            "signal_episode_base": float(getattr(self, "_signal_episode_base", np.nan)),
            "actual_episode_start_phase_offset": self.get_actual_episode_start_phase_offset(),
            "current_spawn_probability": self._current_signal_cycle_spawn_probability(),
            "punctual_time_target": self.get_punctual_time_target(),
            "punctual_time_window_start": float(punctual_window[0]),
            "punctual_time_window_end": float(punctual_window[1]),
            "initial_lane": int(self._initial_lane_id()),
            "goal_lane": int(self._goal_lane_id()),
            "ego_x": float(ego_pos[0]),
            "ego_y": float(ego_pos[1]),
            "ego_speed": float(getattr(self.vehicle, "speed", 0.0)),
            "ego_lane": int(lane_index[2]) if lane_index is not None and len(lane_index) >= 3 else -1,
            "bg_count": int(len(bg)),
            "bg_valid_count": int(len(valid)),
            "bg_low_speed_1": int(sum(1 for s in speeds if s < 1.0)),
            "bg_low_speed_2": int(sum(1 for s in speeds if s < 2.0)),
            "bg_near_goal_low": int(near_goal_low),
            "bg_min_speed": float(min(speeds)) if speeds else np.nan,
            "bg_mean_speed": float(np.mean(speeds)) if speeds else np.nan,
            "bg_max_x": float(max(xs)) if xs else np.nan,
            "virtual_stop_count": int(len(getattr(self, "_virtual_stops", {}) or {})),
            "signal_is_green": float(signal_is_green),
            "signal_remaining": float(signal_remaining),
            "queue_takeover_active": float(self.get_queue_takeover_active()),
            "inter_episode_active": float(bool(getattr(self, "_inter_episode_active", False))),
        }

    def _update_signal_render_items(
        self,
        phase: dict[str, dict[str, float | str]],
    ) -> None:
        if not hasattr(self, "road") or self.road is None or self._signal_controller is None:
            return

        radius_m = float(self.config.get("signal_render_radius_m", 1.2))
        gap_m = float(self.config.get("signal_render_gap_m", 6.0))
        order = ["left", "straight"]
        labels = {"left": "左转", "straight": "直行"}

        items = []
        for idx, direction in enumerate(order):
            if direction not in phase:
                continue
            state = str(phase[direction]["state"])
            remaining = max(float(phase[direction]["remaining"]), 0.0)
            remaining_s = int(np.ceil(remaining))
            x = self._signal_render_x + (idx - 0.5 * (len(order) - 1)) * gap_m
            items.append(
                {
                    "position": [x, self._signal_render_y],
                    "color": self._signal_controller.state_color(state),
                    "label": f"{labels.get(direction, direction)} {remaining_s}s",
                    "state": state,
                    "direction": direction,
                    "radius_m": radius_m,
                }
            )

        self.road.signal_render_items = items

    def _ego_can_pass_in_yellow(self, yellow_remaining: float) -> bool:
        ego = getattr(self, "vehicle", None)
        if ego is None:
            return False
        stop_x = self._goal_longitudinal()
        dist = stop_x - float(ego.position[0])
        if dist <= 0.0:
            return True
        speed = max(float(getattr(ego, "speed", 0.0)), 0.0)
        if speed <= 1e-6:
            return False
        return dist <= speed * max(float(yellow_remaining), 0.0)

    def _ensure_virtual_stop(self, direction: str, lane_id: int) -> None:
        key = (direction, int(lane_id))
        if key in self._virtual_stops:
            return

        lane_index = ("0", "1", int(lane_id))
        lane = self.road.network.get_lane(lane_index)
        stop_local = float(np.clip(self._goal_longitudinal() - 0.5, 0.0, lane.length))
        pos = lane.position(stop_local, 0.0)
        heading = lane.heading_at(stop_local)

        v = VirtualStopVehicle(self.road, pos, heading, speed=0.0)
        v.lane = lane
        v.lane_index = lane_index
        self.road.objects.append(v)
        self._virtual_stops[key] = v

    def _remove_virtual_stop(self, direction: str, lane_id: int) -> None:
        key = (direction, int(lane_id))
        v = self._virtual_stops.pop(key, None)
        if v is not None and hasattr(self, "road") and self.road is not None:
            try:
                self.road.objects.remove(v)
            except ValueError:
                pass

    def _clear_virtual_stops(self) -> None:
        if not hasattr(self, "road") or self.road is None:
            self._virtual_stops = {}
            return
        self.road.signal_render_items = []
        for _, v in list(self._virtual_stops.items()):
            try:
                self.road.objects.remove(v)
            except ValueError:
                pass
        self._virtual_stops = {}

    def _restore_signal_green_launch_behavior(self, vehicle) -> None:
        original = getattr(vehicle, "_signal_green_launch_original", None)
        if isinstance(original, dict):
            for attr, value in original.items():
                setattr(vehicle, attr, value)
        if hasattr(vehicle, "_signal_green_launch_original"):
            delattr(vehicle, "_signal_green_launch_original")

    def _vehicle_signal_direction(self, vehicle) -> str | None:
        direction = getattr(vehicle, "movement_direction", None)
        if direction in {"left", "straight"}:
            return str(direction)

        controller = getattr(self, "_signal_controller", None)
        if controller is None:
            return None
        lane_index = getattr(vehicle, "lane_index", None)
        if lane_index is None or len(lane_index) < 3:
            return None
        return controller.lane_direction(int(lane_index[2]))

    def _signal_green_launch_target_speed(self) -> float:
        configured = self.config.get("signal_green_launch_target_speed", None)
        if configured is not None:
            return max(float(configured), 0.1)

        target_speed = float(AggressiveIDMVehicle.DESIRED_SPEED_MAX)
        speed_limit = float(self.config.get("speed_limit", target_speed))
        if np.isfinite(speed_limit) and speed_limit > 0.0:
            target_speed = min(target_speed, speed_limit)
        return max(target_speed, 0.1)

    def _green_launch_region_candidate(self, vehicle) -> bool:
        direction = self._vehicle_signal_direction(vehicle)
        if direction is None:
            return False

        try:
            x = float(np.asarray(vehicle.position, dtype=float)[0])
        except (TypeError, ValueError, IndexError):
            return False

        stop_x = float(self._goal_longitudinal())
        approach_raw = self.config.get("signal_green_launch_approach_distance", None)
        approach = stop_x if approach_raw is None else max(float(approach_raw), 0.0)
        intersection = max(float(self.config.get("intersection_length", 0.0)), 0.0)
        end_margin = max(float(self.config.get("signal_green_launch_end_margin", 5.0)), 0.0)
        return (stop_x - approach) <= x <= (stop_x + intersection + end_margin)

    def _mark_signal_green_launch_reset_vehicles(self) -> None:
        eligible: dict[int, object] = {}
        if not hasattr(self, "road") or self.road is None:
            self._signal_green_launch_reset_vehicles = eligible
            return

        controlled = tuple(getattr(self, "controlled_vehicles", []) or ())
        for vehicle in list(getattr(self.road, "vehicles", []) or []):
            if any(vehicle is controlled_vehicle for controlled_vehicle in controlled):
                continue
            if self._green_launch_region_candidate(vehicle):
                eligible[id(vehicle)] = vehicle
        self._signal_green_launch_reset_vehicles = eligible

    def _green_launch_candidate(self, vehicle, phase: dict[str, dict[str, float | str]]) -> bool:
        eligible = getattr(self, "_signal_green_launch_reset_vehicles", {})
        if not isinstance(eligible, dict) or eligible.get(id(vehicle)) is not vehicle:
            return False
        direction = self._vehicle_signal_direction(vehicle)
        if direction is None:
            return False
        state = str(phase.get(direction, {}).get("state", "red"))
        if state != "green":
            return False
        return self._green_launch_region_candidate(vehicle)

    def _apply_signal_green_launch_behavior(self, vehicle) -> None:
        if not isinstance(getattr(vehicle, "_signal_green_launch_original", None), dict):
            original = {
                attr: getattr(vehicle, attr)
                for attr in self.SIGNAL_GREEN_LAUNCH_ATTRS
                if hasattr(vehicle, attr)
            }
            vehicle._signal_green_launch_original = original

        vehicle.target_speed = max(
            float(getattr(vehicle, "target_speed", 0.0)),
            self._signal_green_launch_target_speed(),
        )
        vehicle.TIME_WANTED = float(AggressiveIDMVehicle.TIME_WANTED_MEAN)
        vehicle.DISTANCE_WANTED = float(AggressiveIDMVehicle.DISTANCE_WANTED_MEAN)
        vehicle.COMFORT_ACC_MAX = float(AggressiveIDMVehicle.COMFORT_ACC_MAX_MEAN)
        vehicle.COMFORT_ACC_MIN = float(AggressiveIDMVehicle.COMFORT_ACC_MIN_MEAN)
        vehicle.DELTA = 0.5 * (
            float(AggressiveIDMVehicle.DELTA_LOW)
            + float(AggressiveIDMVehicle.DELTA_UPP)
        )
        vehicle.imperfection = float(AggressiveIDMVehicle.IMPERFECTION_MEAN)

    def _update_signal_green_launch_behavior(self, phase: dict[str, dict[str, float | str]]) -> None:
        if not hasattr(self, "road") or self.road is None:
            return

        controlled = tuple(getattr(self, "controlled_vehicles", []) or ())
        enabled = bool(self.config.get("enable_signal_green_launch_behavior", True))
        for vehicle in list(getattr(self.road, "vehicles", []) or []):
            if any(vehicle is controlled_vehicle for controlled_vehicle in controlled):
                continue
            if enabled and self._green_launch_candidate(vehicle, phase):
                self._apply_signal_green_launch_behavior(vehicle)
            else:
                self._restore_signal_green_launch_behavior(vehicle)

    def _update_signal_virtual_stops(self, query_time: float | None = None) -> None:
        if not hasattr(self, "road") or self.road is None or self._signal_controller is None:
            return

        if query_time is None:
            t = float(self._signal_time_global)
        else:
            t = float(self._signal_episode_base + float(query_time))
        phase = self._signal_controller.phase_at(t)
        groups = self._signal_controller.direction_lane_groups

        self._update_signal_render_items(phase)
        self._update_signal_green_launch_behavior(phase)

        ego = getattr(self, "vehicle", None)
        ego_lane_id = None
        if ego is not None:
            li = getattr(ego, "lane_index", None)
            if li is not None and len(li) >= 3:
                ego_lane_id = int(li[2])

        for direction, lane_ids in groups.items():
            state = str(phase[direction]["state"])
            yellow_remaining = float(phase[direction]["remaining"])

            if state == "green":
                for lid in lane_ids:
                    self._remove_virtual_stop(direction, lid)
                continue

            if state == "yellow":
                ego_can_pass = self._ego_can_pass_in_yellow(yellow_remaining)
                for lid in lane_ids:
                    if ego_lane_id == lid and ego_can_pass:
                        self._remove_virtual_stop(direction, lid)
                    else:
                        self._ensure_virtual_stop(direction, lid)
                continue

            # red
            for lid in lane_ids:
                self._ensure_virtual_stop(direction, lid)

    def get_hiro_signal_features(self) -> tuple[float, float]:
        """Return (is_green, remaining_seconds) for ego direction.

        Color encoding: 0 for red/yellow, 1 for green.
        Remaining semantics: for red add one yellow duration so red/yellow are
        treated as one merged non-green stage.
        """
        controller = getattr(self, "_signal_controller", None)
        if controller is None:
            return 1.0, 0.0

        t = float(getattr(self, "_signal_time_global", getattr(self, "time", 0.0)))
        phase = controller.phase_at(t)

        ego = getattr(self, "vehicle", None)
        lane_id = None
        if ego is not None:
            li = getattr(ego, "lane_index", None)
            if li is not None and len(li) >= 3:
                lane_id = int(li[2])
        direction = controller.lane_direction(lane_id)

        state = str(phase.get(direction, {}).get("state", "green"))
        remaining = float(phase.get(direction, {}).get("remaining", 0.0))
        yellow = float(getattr(controller, "yellow", 3.0))

        if state == "green":
            return 1.0, max(remaining, 0.0)
        if state == "red":
            return 0.0, max(remaining + yellow, 0.0)
        return 0.0, max(remaining, 0.0)

    # ----------------- RL task 定义 ----------------- #
    def _reward(self, action: Action) -> float:
        raw = self._rewards(action)
        on_road = float(raw["on_road_reward"])

        weighted: dict[str, float] = {}
        for name, val in raw.items():
            w = float(self.config.get(name, 0.0))
            # 各项真实贡献 = 权重 * 原始分项 * on_road gating
            weighted[name] = w * float(val) * on_road
        total = sum(weighted.values())

        # Auxiliary metric for HIRO high-level reward shaping:
        # keep env reward unchanged, but expose acc-only comfort contribution in info.
        # 特殊记录 on_road_reward
        weighted["on_road_reward"] = on_road
        self._last_raw_rewards = raw
        self._last_weighted_rewards = weighted

        return total

    def _rewards(self, action: Action) -> dict[str, float]:
        # 当前纵向位置（使用全局 x，避免分段路网局部坐标重置）
        longi = float(self.vehicle.position[0])

        # ---------- 1) 进度奖励 ----------
        last_longi = getattr(self, "_last_longitudinal", longi)
        delta_s = max(longi - last_longi, 0.0)
        target_long = self._goal_longitudinal()
        route_length = max(target_long - self._start_longitudinal(), 1e-6)
        progress = np.clip(delta_s / route_length, 0.0, 1.0)

        # ---------- 2) 舒适性奖励（加速度 / 加速度+jerk�?----------
        dt = 1.0 / float(self.config["policy_frequency"])
        cur_speed = self.vehicle.speed
        last_speed = getattr(self, "_last_speed", cur_speed)
        acc = (cur_speed - last_speed) / dt

        # 参考车速辅助奖励：超时后参考车速退化为限�?
        remaining_distance = max(target_long - longi, 0.0)
        remaining_expected_time = float(self.config.get("punctual_time_target", self.config.get("duration", 0.0))) - float(self.time)
        speed_limit = float(self.config.get("speed_limit", 0.0))
        if remaining_expected_time <= 0.0:
            ref_speed = speed_limit
        else:
            ref_speed = remaining_distance / max(remaining_expected_time, 1e-6)
        # ref_speed = route_length / float(self.config.get("punctual_time_target", self.config.get("duration", 0.0)))
        speed_ref_aux = -abs(float(cur_speed) - float(ref_speed)) * dt
        # speed_ref_aux = 0


        a_max = float(self.config["comfort_max_accel"])
        acc_term = (abs(acc) / max(a_max, 1e-6)) ** 2

        comfort = -(acc_term) * dt
        # comfort = - (min(abs(acc) / a_max, 1.0) ** 2) * dt

        # ---------- 3) 换道惩罚 ----------
        curr_lane_id = self.vehicle.lane_index[2]
        last_lane_id = getattr(self, "_last_lane_id", curr_lane_id)
        lane_changed = 1.0 if curr_lane_id != last_lane_id else 0.0
        goal_lane_dense = goal_lane_dense_progress(
            previous_lane_id=last_lane_id,
            current_lane_id=curr_lane_id,
            goal_lane_id=self._goal_lane_id(),
        )

        # ---------- 4) 准时性奖励（只在首次到达目标时给�?----------
        punctual = 0.0
        if not getattr(self, "_has_arrived", False) and self._goal_reached():
            self._has_arrived = True
            self._arrival_time = self.time
            punctual = self._punctual_factor(self._arrival_time)
        wrong_lane_terminal = float(self._wrong_lane_terminal_triggered())

        self._last_speed = cur_speed
        self._last_lane_id = curr_lane_id
        self._last_longitudinal = longi
        return {
            "collision_reward": float(self.vehicle.crashed),
            "progress_reward": progress,
            "speed_ref_aux_reward": float(speed_ref_aux),
            "comfort_reward": comfort,
            "lane_change_reward": lane_changed,
            "goal_lane_dense_reward": goal_lane_dense,
            "punctual_reward": punctual,
            "wrong_lane_terminal_penalty": wrong_lane_terminal,
            "on_road_reward": float(self.vehicle.on_road),
        }


    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed, reached the goal, or went off-road."""
        return (
            self.vehicle.crashed
            or self._goal_longitudinal_reached()
            or not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the episode time limit is reached."""
        return self.time >= self.config["duration"]

    def _wrong_lane_terminal_triggered(self) -> bool:
        longitudinal_reached = self._goal_longitudinal_reached()
        episode_ending = (
            self.vehicle.crashed
            or self._is_truncated()
            or not self.vehicle.on_road
        )
        return wrong_lane_terminal_triggered(
            longitudinal_reached=longitudinal_reached,
            goal_reached=self._goal_reached(),
            episode_ending=bool(episode_ending),
            only_at_goal_longitudinal=bool(
                self.config.get("wrong_lane_penalty_only_at_goal_longitudinal", False)
            ),
        )

    # ----------------- 入口生成环境车（安全间距版） ----------------- #
    def _spawn_background(self, spawn_probability=None):
        cfg = self.config
        if spawn_probability is None:
            spawn_probability = self._current_signal_cycle_spawn_probability()
        if self.np_random.uniform() > spawn_probability:
            return
        lanes = int(cfg["lanes_count"])

        behavior_types = cfg.get(
            "behavior_vehicle_types",
            [cfg["other_vehicles_type"]],
        )
        n_types = len(behavior_types)
        lane_probs_all = cfg.get("behavior_lane_probs", None)   # 各车道独立的行为分布（可选）
        movement_probs_all = cfg.get("movement_behavior_probs", None)  # 按通行方向配置行为分布（可选）
        uniform_probs = np.full(n_types, 1.0 / max(n_types, 1), dtype=float)

        def _normalize_probs(raw_probs) -> np.ndarray | None:
            try:
                arr = np.asarray(raw_probs, dtype=float)
            except (TypeError, ValueError):
                return None
            if arr.shape[0] != n_types:
                return None
            s = float(arr.sum())
            if not np.isfinite(s) or s <= 0.0:
                return None
            return arr / s

        def _get_lane_behavior_probs(lane_id: int, direction: str) -> np.ndarray:
            """Return behavior probabilities for a lane."""
            if lane_probs_all is not None:
                try:
                    lane_row = _normalize_probs(lane_probs_all[lane_id])
                    if lane_row is not None:
                        return lane_row
                except (IndexError, TypeError, ValueError):
                    pass

            if isinstance(movement_probs_all, dict):
                movement_row = _normalize_probs(movement_probs_all.get(direction, None))
                if movement_row is not None:
                    return movement_row

            # Fallback to uniform distribution.
            return uniform_probs

        def _lane_direction(lane_id: int) -> str:
            if self._signal_controller is None:
                return "left" if lane_id == 0 else "straight"
            return self._signal_controller.lane_direction(lane_id)

        # Try several lane/speed samples and insert the first safe vehicle.
        for _ in range(2 * lanes):
            lane_id = int(self.np_random.integers(lanes))
            direction = _lane_direction(lane_id)
            lane_index = ("0", "1", lane_id)
            lane = self.road.network.get_lane(lane_index)
            if cfg["speed_distribution"] == 'Uniform':
                speed_min, speed_max = cfg["flow_speed_range"]
                speed = float(self.np_random.uniform(speed_min, speed_max))
            elif cfg["speed_distribution"] == 'Gaussian':
                speed_mean, speed_dev, speed_bound = cfg["flow_speed_range"]
                speed = float(np.clip(self.np_random.normal(speed_mean, speed_dev), speed_mean-speed_bound, speed_mean+speed_bound))
            else:
                raise RuntimeError

            # 检查是否有安全车头时距
            if not self._can_spawn_on_lane(lane, lane_index, speed):
                continue

            # 按概率抽一类风格，创建车辆
            probs_lane = _get_lane_behavior_probs(lane_id, direction)
            style_idx = int(self.np_random.choice(len(behavior_types), p=probs_lane))
            vehicle_cls = utils.class_from_path(behavior_types[style_idx])
            position = lane.position(0.0, 0.0)
            heading = lane.heading_at(0.0)
            v = vehicle_cls(
                self.road,
                position,
                heading,
                speed,
            )
            v.lane = lane
            v.lane_index = lane_index
            cfg["vid"] += 1
            v.vid = cfg["vid"]
            v.movement_direction = direction
            if hasattr(v, "randomize_behavior"):    # 随机化车辆参�?
                v.randomize_behavior()

            # 可选：锁定环境车在其所属通行方向车道，避免跨方向变道
            if bool(cfg.get("background_vehicle_respect_movement_lanes", True)) and hasattr(v, "enable_lane_change"):
                v.enable_lane_change = False

            self.road.vehicles.append(v)
            break

    def _can_spawn_on_lane(self, lane, lane_index, new_speed: float) -> bool:
        """Return whether inserting a vehicle at x=0 on the lane is safe."""
        cfg = self.config
        min_gap = float(cfg.get("spawn_min_gap", 10.0))          # 纯空�?
        min_t_headway = float(cfg.get("spawn_min_t_headway", 1.5))  # 车头时距
        check_cutins = bool(cfg.get("spawn_check_adjacent_cutins", False))
        cutin_front_gap = float(cfg.get("spawn_adjacent_cutin_front_gap", 15.0))
        cutin_back_gap = float(cfg.get("spawn_adjacent_cutin_back_gap", 5.0))
        front_dist = None

        # 计算最近的前车距离
        for v in self.road.vehicles:
            li = getattr(v, "lane_index", None)
            if li is None or len(li) < 3:
                continue
            if check_cutins:
                target_li = getattr(v, "target_lane_index", None)
                is_targeting_spawn_lane = (
                    isinstance(target_li, tuple)
                    and len(target_li) >= 3
                    and target_li[0] == lane_index[0]
                    and target_li[1] == lane_index[1]
                    and target_li[2] == lane_index[2]
                )
                is_adjacent_lane = (
                    li[0] == lane_index[0]
                    and li[1] == lane_index[1]
                    and abs(int(li[2]) - int(lane_index[2])) == 1
                )
                if is_targeting_spawn_lane and is_adjacent_lane:
                    longi_cutin, _ = lane.local_coordinates(v.position)
                    if -cutin_back_gap <= float(longi_cutin) <= cutin_front_gap:
                        return False
            if li[0] != lane_index[0] or li[1] != lane_index[1] or li[2] != lane_index[2]:
                continue
            longi, _ = lane.local_coordinates(v.position)
            if longi < 0.0: # 理论上不会有 <0 的车，这里略�?
                continue
            if longi < min_gap: # 入口附近有车，直接视为不安全
                return False
            if front_dist is None or longi < front_dist:
                front_dist = longi

        if front_dist is None:  # 该车道入口前方暂时没人，可以安全插入
            return True

        # 安全距离 = 最小空间间�?+ v * t_headway
        safe_dist = min_gap + new_speed * min_t_headway
        return front_dist >= safe_dist

    # ----------------- 清除驶离 / 碰撞的环境车 ----------------- #
    def _clear_background(self):
        L = float(getattr(self, "_road_end_x", self.config.get("road_length", 500.0)))
        margin = 5.0

        remaining = []
        for v in self.road.vehicles:
            # ego 一定保�?
            if v in self.controlled_vehicles:
                remaining.append(v)
                continue
            # 已经 crash 的环境车直接移除，避免堆成“连环车祸山�?
            if getattr(v, "crashed", False):
                continue
            # 判断是否驶离场景
            longi = float(v.position[0])
            if longi > L + margin:
                continue
            remaining.append(v)
        self.road.vehicles = remaining

    # ----------------- 预热后，在公交站插入 ego ----------------- #
    def _sample_ego_speed(self) -> float:
        cfg = self.config
        speed_range = cfg.get("ego_speed_range", None)
        if isinstance(speed_range, (list, tuple, np.ndarray)) and len(speed_range) == 2:
            s0 = float(speed_range[0])
            s1 = float(speed_range[1])
            lo = min(s0, s1)
            hi = max(s0, s1)
            if np.isfinite(lo) and np.isfinite(hi):
                if hi > lo:
                    return float(self.np_random.uniform(lo, hi))
                return float(lo)
        return float(cfg["ego_speed"])

    def _create_ego(self):
        cfg = self.config
        lane_id = self._initial_lane_id()
        lane_index = ("0", "1", int(lane_id))
        lane = self.road.network.get_lane(lane_index)
        ego_speed = self._sample_ego_speed()

        # 清理入口前方车辆�?
        # - ego_clear_radius 为数值时，使用固定半径（兼容旧配置）
        # - ego_clear_radius="auto" 时，�?safety-layer 约束与前车速度动态计�?
        clear_radius_cfg = cfg.get("ego_clear_radius", "auto")
        use_fixed_radius = isinstance(clear_radius_cfg, (int, float, np.floating))
        cleaned = []
        for v in self.road.vehicles:
            lane_v = getattr(v, "lane", None)
            if lane_v is lane:
                longi, _ = lane.local_coordinates(v.position)
                if 0.0 <= longi:
                    if use_fixed_radius:
                        if longi < float(clear_radius_cfg):
                            continue
                    else:
                        front_speed = float(getattr(v, "speed", np.linalg.norm(getattr(v, "velocity", np.zeros(2)))))
                        required_clear_dist = compute_ego_clear_distance_for_front_vehicle(
                            cfg,
                            ego_speed=ego_speed,
                            front_speed=front_speed,
                        )
                        if longi < required_clear_dist:
                            continue
            cleaned.append(v)
        self.road.vehicles = cleaned

        # 初始�?ego
        longi0 = 0
        position = lane.position(longi0, 0.0)
        heading = lane.heading_at(longi0)
        ego = self.action_type.vehicle_class(self.road, position, heading, ego_speed)

        self.vehicle = ego
        self.controlled_vehicles = [ego]
        self.road.vehicles.append(ego)

        # 初始化奖励相关的历史�?
        self._last_speed = ego_speed
        self._last_longitudinal = longi0
        self._has_arrived = False
        self._arrival_time = None
        self._queue_takeover_active = False
        self._queue_takeover_enter_count = 0

    def _goal_reached(self) -> bool:
        """Return whether ego is on the goal lane and past the goal position."""
        if self.vehicle.lane_index[2] != self._goal_lane_id():
            return False
        return self._goal_longitudinal_reached()

    def _goal_longitudinal_reached(self) -> bool:
        """Return whether ego has reached the configured longitudinal goal."""
        longi = float(self.vehicle.position[0])
        return longi >= self._goal_longitudinal()

    def _punctual_factor(self, t: float) -> float:
        """Compute punctuality factor in [0, 1]."""
        t_min, t_max = self.config.get("punctual_time_window", [20.0, 30.0])
        t_target = float(self.config.get("punctual_time_target", 25.0))

        if t < t_min or t > t_max:
            return 0.0

        half_width = min(t_target - t_min, t_max - t_target)
        if half_width <= 0:
            return 0.0

        d = abs(t - t_target) / half_width   # in [0,1]
        d = min(d, 1.0)
        return 1.0 - 0.5 * d

    def _create_bus_stop(self):
        lane_id = self._initial_lane_id()
        lanes = int(self.config["lanes_count"])
        lane_index = ("0", "1", lane_id)
        lane = self.road.network.get_lane(lane_index)

        center_long = 0.0
        bus_width = BusStop.WIDTH
        lane_half_width = getattr(lane, "width", 4.0) / 2.0
        margin = 0.5  # 车道右缘和站台中线之间留一点间�?

        # Always place the stop on the outer side of the selected initial lane.
        side_sign = 1.0 if (2 * lane_id) >= (lanes - 1) else -1.0

        lateral_center = side_sign * (lane_half_width + margin + bus_width / 2.0)

        position = lane.position(center_long, lateral_center)
        heading = lane.heading_at(center_long)

        # 创建 BusStop 对象并加�?road.objects，交�?viewer 渲染
        bus_stop = BusStop(self.road, position, heading)
        if not hasattr(self.road, "objects"):
            self.road.objects = []
        self.road.objects.append(bus_stop)

    def _update_goal_highlight_regions(self):
        """Define render-only goal zone metadata (no physical road entity)."""
        target_x = self._goal_longitudinal()
        length = 10.0
        half_len = max(length, 0.1) * 0.5
        x0 = target_x - half_len
        x1 = target_x + half_len
        lanes = int(self.config["lanes_count"])

        regions = []
        lane_id = self._goal_lane_id()
        segments = getattr(self, "_main_segments", [])

        # Render-only intersection band. In single-road mode this preserves the
        # visual cue without splitting the physical road network.
        ix0 = float(getattr(self, "_intersection_start_x", target_x))
        ix1 = float(getattr(self, "_intersection_end_x", target_x))
        if ix1 > ix0:
            for lane_idx in range(lanes):
                for _from, _to, seg_start_x, seg_end_x in segments:
                    overlap_start = max(ix0, seg_start_x)
                    overlap_end = min(ix1, seg_end_x)
                    if overlap_end <= overlap_start:
                        continue
                    regions.append(
                        {
                            "lane_index": (_from, _to, lane_idx),
                            "s_start": float(overlap_start - seg_start_x),
                            "s_end": float(max(overlap_end - seg_start_x, overlap_start - seg_start_x + 1e-3)),
                            "color": (120, 120, 120, 45),
                        }
                    )

        for _from, _to, seg_start_x, seg_end_x in segments:
            overlap_start = max(x0, seg_start_x)
            overlap_end = min(x1, seg_end_x)
            if overlap_end <= overlap_start:
                continue
            regions.append(
                {
                    "lane_index": (_from, _to, lane_id),
                    "s_start": float(overlap_start - seg_start_x),
                    "s_end": float(max(overlap_end - seg_start_x, overlap_start - seg_start_x + 1e-3)),
                    "color": (255, 80, 80, 90),
                }
            )

        # Explicit stop lines:
        # - left stop line at goal x
        # - right stop line at the end of intersection segment
        stop_line_half_len = 0.4

        def _append_vertical_line(x_line: float, color=(255, 255, 255, 180)) -> None:
            sx0 = x_line - stop_line_half_len
            sx1 = x_line + stop_line_half_len
            for lane_idx in range(lanes):
                for _from, _to, seg_start_x, seg_end_x in segments:
                    overlap_start = max(sx0, seg_start_x)
                    overlap_end = min(sx1, seg_end_x)
                    if overlap_end <= overlap_start:
                        continue
                    regions.append(
                        {
                            "lane_index": (_from, _to, lane_idx),
                            "s_start": float(overlap_start - seg_start_x),
                            "s_end": float(max(overlap_end - seg_start_x, overlap_start - seg_start_x + 1e-3)),
                            "color": color,
                        }
                    )

        # Left stop line (goal x)
        _append_vertical_line(target_x)

        # Right stop line (intersection right boundary)
        right_stop_x = float(getattr(self, "_intersection_end_x", np.nan))
        if not np.isfinite(right_stop_x):
            right_stop_x = None
        for _from, _to, _, seg_end_x in segments:
            if _from == "1" and _to == "2":
                right_stop_x = float(seg_end_x)
                break
        if right_stop_x is not None:
            _append_vertical_line(right_stop_x)

        self.road.highlight_regions = regions

    def _sample_initial_lane_id(self) -> int:
        cfg = self.config
        lanes = int(cfg["lanes_count"])
        lane_probs = cfg.get("initial_lane_probs", None)
        if lane_probs is not None:
            probs = np.asarray(lane_probs, dtype=np.float64).reshape(-1)
            if probs.size != lanes:
                raise ValueError(
                    f"initial_lane_probs must contain {lanes} values, got {probs.size}"
                )
            if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
                raise ValueError("initial_lane_probs must be finite and non-negative")
            total = float(np.sum(probs))
            if total <= 0.0:
                raise ValueError("initial_lane_probs must have a positive sum")
            return int(self.np_random.choice(lanes, p=probs / total))

        lane_cfg = cfg.get("initial_lane_id", "random")
        if lane_cfg == "random":
            return int(self.np_random.integers(lanes))
        return int(np.clip(int(lane_cfg), 0, lanes - 1))

    def _initial_lane_id(self) -> int:
        if not hasattr(self, "_episode_initial_lane_id"):
            self._episode_initial_lane_id = self._sample_initial_lane_id()
        return int(self._episode_initial_lane_id)

    def _start_longitudinal(self) -> float:
        # Stop is fixed at x=0 for this scenario.
        return 0.0

    def _goal_longitudinal(self) -> float:
        cfg = self.config
        default_target = float(cfg.get("road_length", 500.0)) - float(cfg.get("intersection_buffer", 20.0))
        return float(cfg.get("goal_longitudinal", default_target))

    def _goal_lane_id(self) -> int:
        if not hasattr(self, "_episode_goal_lane_id"):
            self._episode_goal_lane_id = self._sample_goal_lane_id()
        return int(self._episode_goal_lane_id)

    def _sample_goal_lane_id(self) -> int:
        return sample_goal_lane_id(
            self.np_random,
            goal_lane_id=self.config.get("goal_lane_id", 0),
            lanes_count=int(self.config["lanes_count"]),
            goal_lane_probs=self.config.get("goal_lane_probs", None),
        )

    def get_goal_lane_id(self) -> int:
        return self._goal_lane_id()

    def set_hiro_goal(self, goal_phys: np.ndarray):
        """
        Update the visual marker for HIRO goal.
        goal_phys: [x, y, vx, vy]
        """
        if not hasattr(self, "road") or self.road is None:
            return

        # Remove existing goal marker
        if hasattr(self, "_goal_marker") and self._goal_marker in self.road.objects:
            self.road.objects.remove(self._goal_marker)

        # Create new marker
        # goal_phys is absolute [x, y, vx, vy]
        position = np.array([goal_phys[0], goal_phys[1]])

        # We can use heading 0 for point goal, or calculate if needed
        heading = 0

        self._goal_marker = GoalMarker(self.road, position, heading)

        if not hasattr(self.road, "objects"):
            self.road.objects = []
        self.road.objects.append(self._goal_marker)
