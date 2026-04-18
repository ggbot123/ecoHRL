import numpy as np

from custom_env.envs.common.abstract import AbstractEnv
from custom_env.road.road import Road, RoadNetwork
from custom_env.road.lane import LineType, StraightLane
from custom_env.envs.common.action import Action
from custom_env import utils
from custom_env.vehicle.objects import Obstacle, Landmark

from configs.conf import get_env_config
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
    """
    静态矩形公交站台，沿道路方向放置。
    length: 沿道路方向长度
    width:  垂直道路方向宽度（从路缘向右侧延伸）
    """
    LENGTH = 20.0  # m，沿 x 方向
    WIDTH = 3.0    # m，可以自己调宽一点，比如 3~4m

    def __init__(self, road, position, heading=0, speed=0):
        super().__init__(road, position, heading, speed)
        self.collidable = False  # 设为 False，使其成为纯视觉物体，避免意外碰撞


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
        self.cycle_offset = float(config.get("signal_cycle_offset", 0.0))

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
    多车道直路 + 顺序交通流 + 预热 + 更安全的生成逻辑
    - 道路：节点 "0" -> "1" 的四车道直路，长度 road_length
    - 环境车：从左端（x=0）按概率生成，跑到右端后删除
    - warmup：先只跑环境车 warmup_time 秒，再在公交站插入 ego
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,  # 例如 10fps，对应你的 policy_frequency=10Hz
    }
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

    # ----------------- 配置 ----------------- #
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update(get_env_config())
        cfg.setdefault("movement_lanes", None)
        cfg.setdefault("movement_behavior_probs", None)
        cfg.setdefault("background_vehicle_respect_movement_lanes", True)
        cfg.setdefault("inter_episode_as_steps", False)
        cfg.setdefault("inter_episode_step_seconds", 0.0)
        cfg.setdefault("inter_episode_zero_obs", True)
        return cfg

    # ----------------- 建路 ----------------- #
    def _create_road(self):
        # 路网由三段组成：
        # 1) 进路段（0->1）：有车道线，长度由 goal x 决定
        # 2) 路口段（1->2）：长度 intersection_length，仅保留道路上下边沿线
        # 3) 路口后段（2->3）：长度 road_length - goal_x - intersection_length
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

    # ----------------- reset：预热 + 插入 ego ----------------- #
    def _reset(self):
        """
        - 第一次 reset：建路 + 全局 warmup 交通流 + 插入 ego；
        - 后续 reset：保留现有路网和交通流，只移除旧 ego、清理一下车流，再插入新的 ego。
        """
        first_reset = not getattr(self, "_did_global_warmup", False)
        has_previous_episode = int(getattr(self, "_episodes_started", 0)) > 0
        self._inter_episode_active = False
        self._inter_episode_remaining = 0.0

        # 每次都重置交通流，用于测试，以保证各个episode之间独立
        if self.config["warmup_each_episode"] is True:
            self._create_road()
            self.road.vehicles = []
            self.controlled_vehicles = []
            self._warmup(render=self.config.get("warmup_render", False))
        else:
            if first_reset:
                # ------- 第一次：建立路网 + 清空所有车辆 + 预热交通流 -------
                self._create_road()
                self.road.vehicles = []
                self.controlled_vehicles = []

                # 只跑环境车 warmup_time 秒
                self._warmup(render=self.config.get("warmup_render", False))

                # 打标记：后续 reset 不再重建 & warmup
                self._did_global_warmup = True
            else:
                # 把上一回合的 ego 从 road.vehicles 里移除
                if getattr(self, "vehicle", None) is not None:
                    try:
                        self.road.vehicles.remove(self.vehicle)
                    except ValueError:
                        pass
                self.controlled_vehicles = []
                self._clear_virtual_stops()
                self._clear_background()

        # First episode: align to target offset (may be zero extra wait if already aligned).
        # Later episodes: optionally defer long alignment into inter-episode dummy env.step calls.
        if has_previous_episode and bool(self.config.get("inter_episode_as_steps", False)):
            delta = self._compute_episode_start_offset_delta(strict_next=True)
            if delta > 1e-9:
                self._begin_inter_episode_phase(delta)
                return

        self._advance_to_episode_start_offset(strict_next=has_previous_episode)
        self._signal_episode_base = float(self._signal_time_global)
        self._create_ego()
        self._episodes_started += 1
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

        lane_id = self._start_lane_id()
        lane_index = ("0", "1", int(lane_id))
        lane = self.road.network.get_lane(lane_index)
        position = lane.position(0.0, 0.0)
        heading = lane.heading_at(0.0)

        dummy = self.action_type.vehicle_class(self.road, position, heading, 0.0)
        dummy.lane = lane
        dummy.lane_index = lane_index
        self.vehicle = dummy

        self._last_speed = 0.0
        self._last_acc = 0.0
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
        self._create_ego()
        self._episodes_started += 1
        self._update_signal_virtual_stops(query_time=0.0)

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
        }
        return obs, 0.0, False, False, info

    def _advance_to_episode_start_offset(self, strict_next: bool) -> None:
        """Keep background traffic evolving and align spawn time to configured signal phase offset."""
        delta = self._compute_episode_start_offset_delta(strict_next=strict_next)
        if delta <= 1e-9:
            return

        self._simulate_background_for(delta)

    def _simulate_background_for(self, seconds: float) -> None:
        """Simulate road/background traffic for given wall-clock seconds without ego control."""
        remain = max(float(seconds), 0.0)
        if remain <= 0.0:
            return

        sim_freq = float(self.config["simulation_frequency"])
        base_dt = 1.0 / sim_freq
        base_spawn_p = float(self.config.get("spawn_probability", 0.0))

        while remain > 1e-9:
            dt = min(base_dt, remain)
            self._update_signal_virtual_stops(query_time=None)
            self._clear_background()
            spawn_p = base_spawn_p * (dt / base_dt)
            self._spawn_background(spawn_probability=spawn_p)
            self.road.act()
            self.road.step(dt)
            self._signal_time_global += dt
            remain -= dt

        self._clear_background()
        self._update_signal_virtual_stops(query_time=None)

    def _warmup(self, render: bool = False):
        """只跑环境车 warmup_time 秒，可以选择是否渲染出来看。"""
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
            # 调试模式：在 reset 期间也渲染 warmup 的画面
            if render and self.render_mode is not None:
                self.render()

        # 再做一次清理，避免 warmup 结束时残留 crash 车辆
        self._clear_virtual_stops()
        self._clear_background()

        self._warmup_times = np.asarray(times, dtype=float)
        self._warmup_avg_speeds = np.asarray(avg_speeds, dtype=float)

    # ----------------- RL step：在 AbstractEnv 的基础上维护车流 ----------------- #
    def step(self, action):
        if bool(getattr(self, "_inter_episode_active", False)):
            return self._step_inter_episode_dummy(action)

        # 在当前决策步生效的信号相位（供 _simulate 内 IDM 使用）
        dt = 1.0 / float(self.config["policy_frequency"])
        self._update_signal_virtual_stops(query_time=self.time + dt)

        # 让 AbstractEnv 完成 ego 控制 + 仿真
        obs, reward, terminated, truncated, info = super().step(action)

        # Sync persistent signal clock to the current episode local time after step().
        self._signal_time_global = float(self._signal_episode_base + self.time)

        # 维持渲染与下一步前的一致信号状态
        self._update_signal_virtual_stops(query_time=self.time)

        # 把“加权后的分项奖励”塞进 info，方便 callback 从 infos 里读
        weighted = getattr(self, "_last_weighted_rewards", None)
        if isinstance(info, dict) and weighted is not None:
            info["reward_components"] = dict(weighted)

        next_obs_is_dummy = False
        if bool(terminated or truncated) and bool(self.config.get("inter_episode_as_steps", False)):
            pending = self._compute_episode_start_offset_delta(strict_next=True)
            next_obs_is_dummy = bool(pending > 1e-9)
            if isinstance(info, dict):
                info["inter_episode_pending_seconds"] = float(max(pending, 0.0))

        if isinstance(info, dict):
            info["inter_episode"] = False
            info["skip_replay"] = False
            info["next_obs_is_dummy"] = bool(next_obs_is_dummy)

        sim_freq = float(self.config["simulation_frequency"])
        pol_freq = float(self.config["policy_frequency"])

        # 在每个决策步之后，更新一次车流：清除驶离 & crashed，按概率增车
        self._clear_background()
        self._spawn_background(self.config["spawn_probability"] * (sim_freq / pol_freq))    # TODO: 完善增车策略，现在是按policy_freq集总生成，不是按simu_freq生成

        return obs, reward, terminated, truncated, info

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
        stop_local = max(lane.length - 0.5, 0.0)
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
        comfort_w = float(self.config.get("comfort_reward", 0.0))
        comfort_acc_only = float(raw.get("comfort_reward_acc_only", raw.get("comfort_reward", 0.0)))
        weighted["comfort_reward_acc_only_for_high"] = comfort_w * comfort_acc_only * on_road

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

        # ---------- 2) 舒适性奖励（加速度 / 加速度+jerk） ----------
        dt = 1.0 / float(self.config["policy_frequency"])
        cur_speed = self.vehicle.speed
        last_speed = getattr(self, "_last_speed", cur_speed)
        acc = (cur_speed - last_speed) / dt

        # 参考车速辅助奖励：超时后参考车速退化为限速
        remaining_distance = max(target_long - longi, 0.0)
        remaining_expected_time = float(self.config.get("punctual_time_target", self.config.get("duration", 0.0))) - float(self.time)
        speed_limit = float(self.config.get("speed_limit", 0.0))
        if remaining_expected_time <= 0.0:
            ref_speed = speed_limit
        else:
            ref_speed = remaining_distance / max(remaining_expected_time, 1e-6)
        ref_speed = route_length / float(self.config.get("punctual_time_target", self.config.get("duration", 0.0)))
        # speed_ref_aux = -abs(float(cur_speed) - float(ref_speed)) * dt
        speed_ref_aux = 0


        a_max = float(self.config["comfort_max_accel"])
        acc_term = (abs(acc) / max(a_max, 1e-6)) ** 2

        use_jerk = bool(self.config.get("comfort_use_jerk", False))
        comfort_acc_only = -(acc_term) * dt
        if use_jerk:
            last_acc = float(getattr(self, "_last_acc", acc))
            jerk = (acc - last_acc) / dt
            j_max = float(self.config.get("comfort_max_jerk", 5.0))
            jerk_term = (abs(jerk) / max(j_max, 1e-6)) ** 2

            w_acc = float(self.config.get("comfort_acc_weight", 1.0))
            w_jerk = float(self.config.get("comfort_jerk_weight", 1.0))
            w_sum = max(w_acc + w_jerk, 1e-6)
            comfort = -((w_acc * acc_term + w_jerk * jerk_term) / w_sum) * dt
        else:
            comfort = -(acc_term) * dt
        # comfort = - (min(abs(acc) / a_max, 1.0) ** 2) * dt

        # ---------- 3) 换道惩罚 ----------
        curr_lane_id = self.vehicle.lane_index[2]
        last_lane_id = getattr(self, "_last_lane_id", curr_lane_id)
        lane_changed = 1.0 if curr_lane_id != last_lane_id else 0.0

        # ---------- 4) 准时性奖励（只在首次到达目标时给） ----------
        punctual = 0.0
        if not getattr(self, "_has_arrived", False) and self._goal_reached():
            self._has_arrived = True
            self._arrival_time = self.time
            punctual = self._punctual_factor(self._arrival_time)

        self._last_speed = cur_speed
        self._last_acc = acc
        self._last_lane_id = curr_lane_id
        self._last_longitudinal = longi
        return {
            "collision_reward": float(self.vehicle.crashed),
            "progress_reward": progress,
            "speed_ref_aux_reward": float(speed_ref_aux),
            "comfort_reward": comfort,
            "comfort_reward_acc_only": comfort_acc_only,
            "lane_change_reward": lane_changed,
            "punctual_reward": punctual,
            "on_road_reward": float(self.vehicle.on_road),
        }
    

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed, reached the goal, or went off-road."""
        return (
            self.vehicle.crashed
            or self._goal_longitudinal_reached()
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the episode time limit is reached."""
        return self.time >= self.config["duration"]

    # ----------------- 入口生成环境车（安全间距版） ----------------- #
    def _spawn_background(self, spawn_probability=None):
        cfg = self.config
        if spawn_probability is None:
            spawn_probability = float(cfg["spawn_probability"])
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
        global_probs = np.array(
            cfg.get("behavior_probs", [1.0] * n_types),
            dtype=float,
        )
        global_probs = global_probs / global_probs.sum()

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
            """返回当前 lane_id 的 behavior 概率向量"""
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

            # 回退：使用全局分布
            return global_probs

        def _lane_direction(lane_id: int) -> str:
            if self._signal_controller is None:
                return "left" if lane_id == 0 else "straight"
            return self._signal_controller.lane_direction(lane_id)

        # 尝试若干次（不同车道+速度），找一个符合安全间距的插入点，成功生成一辆就退出循环
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
            if hasattr(v, "randomize_behavior"):    # 随机化车辆参数
                v.randomize_behavior()

            # 可选：锁定环境车在其所属通行方向车道，避免跨方向变道
            if bool(cfg.get("background_vehicle_respect_movement_lanes", True)) and hasattr(v, "enable_lane_change"):
                v.enable_lane_change = False
                
            self.road.vehicles.append(v)
            break

    def _can_spawn_on_lane(self, lane, lane_index, new_speed: float) -> bool:
        """
        判断在给定 lane 上、以 new_speed 从 x=0 插入是否安全：
        - 入口附近必须没有太近的车（空间间距）
        - 最近前车与入口距离 >= min_gap + new_speed * min_t_headway（时间车头时距约束）
        """
        cfg = self.config
        min_gap = float(cfg.get("spawn_min_gap", 10.0))          # 纯空间
        min_t_headway = float(cfg.get("spawn_min_t_headway", 1.5))  # 车头时距
        front_dist = None

        # 计算最近的前车距离
        for v in self.road.vehicles:
            li = getattr(v, "lane_index", None)
            if li is None or len(li) < 3:
                continue
            if li[0] != lane_index[0] or li[1] != lane_index[1] or li[2] != lane_index[2]:
                continue
            longi, _ = lane.local_coordinates(v.position)
            if longi < 0.0: # 理论上不会有 <0 的车，这里略过
                continue
            if longi < min_gap: # 入口附近有车，直接视为不安全
                return False
            if front_dist is None or longi < front_dist:
                front_dist = longi

        if front_dist is None:  # 该车道入口前方暂时没人，可以安全插入
            return True

        # 安全距离 = 最小空间间距 + v * t_headway
        safe_dist = min_gap + new_speed * min_t_headway
        return front_dist >= safe_dist

    # ----------------- 清除驶离 / 碰撞的环境车 ----------------- #
    def _clear_background(self):
        L = float(getattr(self, "_road_end_x", self.config.get("road_length", 500.0)))
        margin = 5.0

        remaining = []
        for v in self.road.vehicles:
            # ego 一定保留
            if v in self.controlled_vehicles:
                remaining.append(v)
                continue
            # 已经 crash 的环境车直接移除，避免堆成“连环车祸山”
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
        lane_id = self._start_lane_id()
        lane_index = ("0", "1", int(lane_id))
        lane = self.road.network.get_lane(lane_index)
        ego_speed = self._sample_ego_speed()

        # 清理入口前方车辆：
        # - ego_clear_radius 为数值时，使用固定半径（兼容旧配置）
        # - ego_clear_radius="auto" 时，按 safety-layer 约束与前车速度动态计算
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

        # 初始化 ego
        longi0 = 0
        position = lane.position(longi0, 0.0)
        heading = lane.heading_at(longi0)
        ego = self.action_type.vehicle_class(self.road, position, heading, ego_speed)

        self.vehicle = ego
        self.controlled_vehicles = [ego]
        self.road.vehicles.append(ego)

        # 初始化奖励相关的历史量
        self._last_speed = ego_speed
        self._last_acc = 0.0
        self._last_longitudinal = longi0
        self._has_arrived = False
        self._arrival_time = None

    def _goal_reached(self) -> bool:
        """在目标车道且 x >= goal_longitudinal（默认路口前）"""
        if self.vehicle.lane_index[2] != self._target_lane_id():
            return False
        return self._goal_longitudinal_reached()
    
    def _goal_longitudinal_reached(self) -> bool:
        """x >= goal_longitudinal（不要求在目标车道）"""
        longi = float(self.vehicle.position[0])
        return longi >= self._goal_longitudinal()
    
    def _punctual_factor(self, t: float) -> float:
        """根据到达时间 t 计算 [0,1] 上的准时性系数"""
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
        lane_id = self._start_lane_id()
        lanes = int(self.config["lanes_count"])
        lane_index = ("0", "1", lane_id)
        lane = self.road.network.get_lane(lane_index)

        center_long = 0.0
        bus_width = BusStop.WIDTH
        lane_half_width = getattr(lane, "width", 4.0) / 2.0
        margin = 0.5  # 车道右缘和站台中线之间留一点间隙

        # Always place the stop on the outer side of the selected start lane.
        side_sign = 1.0 if (2 * lane_id) >= (lanes - 1) else -1.0

        lateral_center = side_sign * (lane_half_width + margin + bus_width / 2.0)

        position = lane.position(center_long, lateral_center)
        heading = lane.heading_at(center_long)

        # 创建 BusStop 对象并加入 road.objects，交给 viewer 渲染
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
        lane_id = self._target_lane_id()
        segments = getattr(self, "_main_segments", [])
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
        right_stop_x = None
        for _from, _to, _, seg_end_x in segments:
            if _from == "1" and _to == "2":
                right_stop_x = float(seg_end_x)
                break
        if right_stop_x is not None:
            _append_vertical_line(right_stop_x)

        self.road.highlight_regions = regions

    def _start_lane_id(self) -> int:
        cfg = self.config
        lane_cfg = cfg.get("start_lane_id", int(cfg["lanes_count"]) - 1)
        lanes = int(cfg["lanes_count"])
        if lane_cfg == "random":
            return int(self.np_random.integers(lanes))
        return int(np.clip(int(lane_cfg), 0, lanes - 1))

    def _start_longitudinal(self) -> float:
        # Stop is fixed at x=0 for this scenario.
        return 0.0

    def _goal_longitudinal(self) -> float:
        cfg = self.config
        default_target = float(cfg.get("road_length", 500.0)) - float(cfg.get("intersection_buffer", 20.0))
        return float(cfg.get("goal_longitudinal", default_target))

    def _target_lane_id(self) -> int:
        cfg = self.config
        lanes = int(cfg["lanes_count"])
        lane_id = int(cfg.get("target_lane_id", cfg.get("goal_lane_id", 0)))
        return int(np.clip(lane_id, 0, lanes - 1))

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
