import numpy as np

from custom_env.envs.common.abstract import AbstractEnv
from custom_env.road.road import Road, RoadNetwork
from custom_env.envs.common.action import Action
from custom_env import utils
from custom_env.vehicle.objects import Obstacle, Landmark

from configs.builders import get_env_config
from scenarios.goal_lane_logic import sample_goal_lane_id
from scenarios.reward_logic import goal_lane_dense_progress, wrong_lane_terminal_triggered
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
    静态矩形公交站台，沿道路方向放置。
    length: 沿道路方向长度
    width:  垂直道路方向宽度（从路缘向右侧延伸）
    """
    LENGTH = 20.0  # m x
    WIDTH = 3.0    # m 3~4m

    def __init__(self, road, position, heading=0, speed=0):
        super().__init__(road, position, heading, speed)
        self.collidable = False  #  False?

class MultiLaneEnv(AbstractEnv):
    """
    四车道直路 + 顺序交通流 + 预热 + 更安全的生成逻辑
    - 道路：节点 "0" -> "1" 的四车道直路，长度 road_length
    - 环境车：从左端（x=0）按概率生成，跑到右端后删除
    - warmup：先只跑环境车 warmup_time 秒，再在入口插入 ego
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,  #  10fps?policy_frequency=10Hz
    }
    def __init__(self, config: dict = None, render_mode: str | None = None):
        super().__init__(config=config, render_mode=render_mode)
        if self.config['PERCEPTION_DISTANCE'] is not None:
            self.PERCEPTION_DISTANCE = self.config['PERCEPTION_DISTANCE']
        self._background_only_sim_time = 0.0

    # -----------------  ----------------- #
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update(get_env_config())
        return cfg

    # -----------------  ----------------- #
    def _create_road(self):
        # 四车道直路，从节点 "0" 到 "1"
        net = RoadNetwork.straight_road_network(
            lanes=int(self.config["lanes_count"]),
            start=0.0,
            length=float(self.config["road_length"]),
            speed_limit=self.config["speed_limit"],
            nodes_str=("0", "1"),
        )
        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )
        self._create_bus_stop()

    # ----------------- reset?+  ego ----------------- #
    def _reset(self):
        """
        - 第一次 reset：建路 + 全局 warmup 交通流 + 插入 ego；
        - 后续 reset：保留现有路网和交通流，只移除旧 ego、清理一下车流，再插入新的 ego。
        """
        # episode
        self._episode_initial_lane_id = self._sample_initial_lane_id()
        self._episode_goal_lane_id = self._sample_goal_lane_id()

        if self.config["warmup_each_episode"] is True:
            self._create_road()
            self.road.vehicles = []
            self.controlled_vehicles = []
            self._warmup(render=self.config.get("warmup_render", False))
        else:
            first_reset = not getattr(self, "_did_global_warmup", False)
            if first_reset:
                # ------- 第一次：建立路网 + 清空所有车辆 + 预热交通流 -------
                self._create_road()
                self.road.vehicles = []
                self.controlled_vehicles = []

                # 只跑环境车 warmup_time 秒
                self._warmup(render=self.config.get("warmup_render", False))

                #  reset  & warmup
                self._did_global_warmup = True
            else:
                # 把上一回合的 ego 从 road.vehicles 里移除
                if getattr(self, "vehicle", None) is not None:
                    try:
                        self.road.vehicles.remove(self.vehicle)
                    except ValueError:
                        pass
                self.controlled_vehicles = []
                self._clear_background()
        self._create_ego()

    def _warmup(self, render: bool = False):
        """Warm up background traffic before inserting ego."""
        warmup_time = float(self.config["warmup_time"])
        sim_freq = float(self.config["simulation_frequency"])
        steps = int(warmup_time * sim_freq)

        avg_speeds = []
        times = []
        for k in range(steps):
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
            t = k / sim_freq  #  warmup  [s]
            times.append(t)
            avg_speeds.append(avg_speed)

            self.road.act()
            self.road.step(1.0 / sim_freq)
            #  reset ?warmup ?
            if render and self.render_mode is not None:
                self.render()

        # 再做一次清理，避免 warmup 结束时残留 crash 车辆
        self._clear_background()

        self._background_only_sim_time += float(steps) / max(sim_freq, 1e-6)
        self._warmup_times = np.asarray(times, dtype=float)
        self._warmup_avg_speeds = np.asarray(avg_speeds, dtype=float)

    # ----------------- RL step AbstractEnv ?----------------- #
    def step(self, action):
        # 让 AbstractEnv 完成 ego 控制 + 仿真
        obs, reward, terminated, truncated, info = super().step(action)

        # 把“加权后的分项奖励”塞进 info，方便 callback 从 infos 里读
        weighted = getattr(self, "_last_weighted_rewards", None)
        if isinstance(info, dict) and weighted is not None:
            info["reward_components"] = dict(weighted)
        if isinstance(info, dict):
            info["env_diagnostics"] = self._env_diagnostics()
        if isinstance(info, dict) and bool(terminated or truncated):
            info["terminal_signal_features"] = tuple(self.get_hiro_signal_features())

        sim_freq = float(self.config["simulation_frequency"])
        pol_freq = float(self.config["policy_frequency"])

        # 在每个决策步之后，更新一次车流：清除驶离 & crashed，按概率增车
        self._clear_background()
        self._spawn_background(self.config["spawn_probability"] * (sim_freq / pol_freq))    # TODO: 完善增车策略，现在是按policy_freq集总生成，不是按simu_freq生成

        return obs, reward, terminated, truncated, info

    def _env_diagnostics(self) -> dict:
        bg = [v for v in self.road.vehicles if v not in self.controlled_vehicles]
        valid = [v for v in bg if not getattr(v, "crashed", False)]
        speeds = [
            float(getattr(v, "speed", np.linalg.norm(getattr(v, "velocity", np.zeros(2)))))
            for v in valid
        ]
        xs = [float(np.asarray(getattr(v, "position", [np.nan, np.nan]), dtype=float)[0]) for v in valid]
        near_goal_low = sum(
            1
            for x, s in zip(xs, speeds)
            if float(self.config.get("goal_longitudinal", self.config["road_length"])) - 40.0 <= x <= float(self.config.get("goal_longitudinal", self.config["road_length"])) + 10.0
            and s < 2.0
        )
        ego_pos = np.asarray(getattr(self.vehicle, "position", [np.nan, np.nan]), dtype=float)
        lane_index = getattr(self.vehicle, "lane_index", (None, None, -1))
        return {
            "time": float(getattr(self, "time", 0.0)),
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
            "virtual_stop_count": 0,
            "signal_is_green": -1.0,
            "signal_remaining": -1.0,
        }

    def get_hiro_signal_features(self) -> tuple[float, float]:
        """
        Base multi-lane scenario has no traffic light control.
        Use fixed sentinel values to indicate "no signal": (-1, -1).
        """
        return -1.0, -1.0

    def get_punctual_time_target(self) -> float:
        """Return the punctual target used for the current episode."""
        return float(
            self.config.get(
                "punctual_time_target",
                self.config.get("duration", 0.0),
            )
        )

    def get_punctual_time_window(self) -> tuple[float, float]:
        """Return the configured punctual arrival window."""
        window = self.config.get("punctual_time_window", [0.0, 0.0])
        return float(window[0]), float(window[1])

    # ----------------- RL task  ----------------- #
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
        # 当前车道和纵向位置
        lane = self.road.network.get_lane(self.vehicle.lane_index)
        longi, _ = lane.local_coordinates(self.vehicle.position)

        # ---------- 1) 进度奖励 ----------
        last_longi = getattr(self, "_last_longitudinal", longi)
        delta_s = max(longi - last_longi, 0.0)
        goal_long = float(self.config.get("goal_longitudinal", self.config["road_length"]))
        progress = np.clip(delta_s / goal_long, 0.0, 1.0)

        # ---------- 2) 舒适性奖励（加速度 / 加速度+jerk） ----------
        dt = 1.0 / float(self.config["policy_frequency"])
        cur_speed = self.vehicle.speed
        last_speed = getattr(self, "_last_speed", cur_speed)
        acc = (cur_speed - last_speed) / dt

        # 参考车速辅助奖励：超时后参考车速退化为限速
        remaining_distance = max(goal_long - longi, 0.0)
        remaining_expected_time = float(self.config.get("punctual_time_target", self.config.get("duration", 0.0))) - float(self.time)
        speed_limit = float(self.config.get("speed_limit", 0.0))
        if remaining_expected_time <= 0.0:
            ref_speed = speed_limit
        else:
            ref_speed = remaining_distance / max(remaining_expected_time, 1e-6)
        # ref_speed = goal_long / float(self.config.get("punctual_time_target", self.config.get("duration", 0.0)))
        speed_ref_aux = -abs(float(cur_speed) - float(ref_speed)) * dt
        # speed_ref_aux = 0


        a_max = float(self.config["comfort_max_accel"])
        acc_term = (abs(acc) / max(a_max, 1e-6)) ** 2

        comfort = -(acc_term) * dt
        # comfort = - (min(abs(acc) / a_max, 1.0) ** 2) * dt

        # ---------- 3)  ----------
        curr_lane_id = self.vehicle.lane_index[2]
        last_lane_id = getattr(self, "_last_lane_id", curr_lane_id)
        lane_changed = 1.0 if curr_lane_id != last_lane_id else 0.0
        goal_lane_dense = goal_lane_dense_progress(
            previous_lane_id=last_lane_id,
            current_lane_id=curr_lane_id,
            goal_lane_id=self._goal_lane_id(),
        )

        # ---------- 4) ?----------
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
        """Compute punctuality factor in [0, 1]."""
        return (
            self.vehicle.crashed
            or self._goal_longitudinal_reached()
            # or self._goal_reached()
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

    # -----------------  ----------------- #
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
        lane_probs_all = cfg.get("behavior_lane_probs", None)   #
        uniform_probs = np.full(n_types, 1.0 / max(n_types, 1), dtype=float)

        def _get_lane_behavior_probs(lane_id: int) -> np.ndarray:
            """Return behavior probabilities for a lane."""
            if lane_probs_all is not None:
                try:
                    lane_row = np.asarray(lane_probs_all[lane_id], dtype=float)
                    if lane_row.shape[0] == n_types:
                        lane_row = lane_row / lane_row.sum()
                        return lane_row
                except (IndexError, TypeError, ValueError):
                    pass
            return uniform_probs

        # 尝试若干次（不同车道+速度），找一个符合安全间距的插入点，成功生成一辆就退出循环
        for _ in range(2 * lanes):
            lane_id = int(self.np_random.integers(lanes))
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
            probs_lane = _get_lane_behavior_probs(lane_id)
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
            if hasattr(v, "randomize_behavior"):    # 随机化车辆参数
                v.randomize_behavior()

            self.road.vehicles.append(v)
            break

    def _can_spawn_on_lane(self, lane, lane_index, new_speed: float) -> bool:
        """
        判断在给定 lane 上、以 new_speed 从 x=0 插入是否安全：
        - 入口附近必须没有太近的车（空间间距）
        - 最近前车与入口距离 >= min_gap + new_speed * min_t_headway（时间车头时距约束）
        """
        cfg = self.config
        min_gap = float(cfg.get("spawn_min_gap", 10.0))          # ?
        min_t_headway = float(cfg.get("spawn_min_t_headway", 1.5))  #
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
        base_lane = self.road.network.get_lane(("0", "1", 0))
        L = base_lane.length
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
            lane = getattr(v, "lane", None)
            longi, _ = lane.local_coordinates(v.position)
            if longi > L + margin:
                continue
            remaining.append(v)
        self.road.vehicles = remaining

    # ----------------- 预热后，在入口插入 ego ----------------- #
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

    def _sample_goal_lane_id(self) -> int:
        return sample_goal_lane_id(
            self.np_random,
            goal_lane_id=self.config.get("goal_lane_id", 0),
            lanes_count=int(self.config["lanes_count"]),
            goal_lane_probs=self.config.get("goal_lane_probs", None),
        )

    def _goal_lane_id(self) -> int:
        if not hasattr(self, "_episode_goal_lane_id"):
            self._episode_goal_lane_id = self._sample_goal_lane_id()
        return int(self._episode_goal_lane_id)

    def get_goal_lane_id(self) -> int:
        return self._goal_lane_id()

    def _create_ego(self):
        cfg = self.config
        lane_id = self._initial_lane_id()
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
        self._last_longitudinal = longi0
        self._has_arrived = False
        self._arrival_time = None

    def _goal_reached(self) -> bool:
        """在目标车道且 x >= goal_longitudinal"""
        if self.vehicle.lane_index[2] != self._goal_lane_id():
            return False
        return self._goal_longitudinal_reached()

    def _goal_longitudinal_reached(self) -> bool:
        """x >= goal_longitudinal（不要求在目标车道）"""
        lane = self.road.network.get_lane(self.vehicle.lane_index)
        longi, _ = lane.local_coordinates(self.vehicle.position)
        goal_long = float(self.config["goal_longitudinal"])
        return longi >= goal_long

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
        lane_index = ("0", "1", int(self.config["lanes_count"]) - 1)
        lane = self.road.network.get_lane(lane_index)

        center_long = float(self.config.get("goal_longitudinal", 300.0))  # ?x=300 ?
        bus_length = BusStop.LENGTH
        bus_width = BusStop.WIDTH
        lane_half_width = getattr(lane, "width", 4.0) / 2.0
        margin = 0.5  # ?
        lateral_center = lane_half_width + margin + bus_width / 2.0

        position = lane.position(center_long, lateral_center)
        heading = lane.heading_at(center_long)

        # 创建 BusStop 对象并加入 road.objects，交给 viewer 渲染
        bus_stop = BusStop(self.road, position, heading)
        if not hasattr(self.road, "objects"):
            self.road.objects = []
        self.road.objects.append(bus_stop)

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
