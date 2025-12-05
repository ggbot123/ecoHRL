import numpy as np

from custom_env.envs.common.abstract import AbstractEnv
from custom_env.road.road import Road, RoadNetwork
from custom_env.vehicle.controller import ControlledVehicle
from custom_env.envs.common.action import Action
from custom_env import utils

Observation = np.ndarray

class MultiLaneEnv(AbstractEnv):
    """
    四车道直路 + 顺序交通流 + 30s 预热 + 更安全的生成逻辑
    - 道路：节点 "0" -> "1" 的四车道直路，长度 road_length
    - 环境车：从左端（x=0）按概率生成，跑到右端后删除
    - warmup：先只跑环境车 warmup_time 秒，再在入口插入 ego
    """
    # ----------------- 配置 ----------------- #
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update(
            {
                # 基本设置
                "simulation_frequency": 10,    # [Hz]
                "policy_frequency": 1,         # [Hz]
                "duration": 50,               # [s]

                # 道路设置
                "lanes_count": 3,
                "road_length": 500.0,
                "speed_limit": 15.0,          # 限速 [m/s]

                # 交通流设置
                "spawn_probability": 0.07,       # 每个仿真步生成一辆新车的概率
                "flow_speed_range": [10.0, 15.0],   # 环境车 初始 + 目标 速度
                "spawn_min_gap": 10.0,           # 入口附近的最小空间间距 [m]
                "spawn_min_t_headway": 1.5,      # 最小时间车头时距 [s]
                # "other_vehicles_type": "custom_env.vehicle.behavior.IDMVehicle",
                "behavior_vehicle_types": [     # 三种风格 IDM 类型及其概率
                    "custom_env.vehicle.behavior.NormalIDMVehicle",
                    "custom_env.vehicle.behavior.AggressiveIDMVehicle",
                    "custom_env.vehicle.behavior.DefensiveIDMVehicle",
                ],
                "behavior_probs": [0.5, 0.2, 0.3],
                "vid": 0,

                # ego设置
                "controlled_vehicles": 1,
                "ego_speed": 10.0,        # [m/s]
                "initial_lane_id": 1,
                "warmup_time": 40.0,             # 只跑环境车的时间 [s]
                "ego_clear_radius": 10.0,        # 在插入ego前，清除入口附近多远范围内的车辆 [m]
                
                # 观测-动作-奖励空间配置
                "observation": {
                    "type": "Kinematics",
                    "normalize": False,
                },
                # "action": {
                #     "type": "DiscreteMetaAction",
                #     "target_speeds": np.linspace(10, 20, 3),  # [m/s]，设置MDPVehicle可选目标速度
                # },
                "action": {
                    "type": "ParamLaneAccelAction",
                    "acceleration_range": [-5.0, 5.0],
                    "lane_actions": ["KEEP", "LANE_LEFT", "LANE_RIGHT"],
                },
                "goal_longitudinal": 300.0,        # 任务 / 目标设置
                "goal_lane_id": 2,
                "punctual_time_window": [20.0, 30.0],   # punctual arrival reward
                "punctual_time_target": 25.0,
                "punctual_reward": 10.0,
                "collision_reward": -10.0,  # collision penalty
                "progress_reward": 1,  # progress reward
                "comfort_reward": 0.2,  # comfort reward
                "comfort_max_accel": 3.0, 
                "lane_change_reward": -0.5,  # lane change penalty
                # "reward_speed_range": [10, 15], # TODO: calculated by goal position
                "offroad_terminal": False,
            }
        )
        return cfg

    # ----------------- 建路 ----------------- #
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

    # ----------------- reset：预热 + 插入 ego ----------------- #
    def _reset(self):
        # 1) 建路 & 清空车辆
        self._create_road()
        self.road.vehicles = []
        self.controlled_vehicles = []

        # 2) 预热 warmup_time 秒
        self._warmup(render=self.config.get("warmup_render", False))

        # 3) warmup 结束后，从入口插入 ego
        self._create_ego()

    def _warmup(self, render: bool = False):
        """只跑环境车 warmup_time 秒，可以选择是否渲染出来看。"""
        warmup_time = float(self.config["warmup_time"])
        sim_freq = float(self.config["simulation_frequency"])
        steps = int(warmup_time * sim_freq)

        for _ in range(steps):
            self._clear_background()
            self._spawn_background()
            self.road.act()
            self.road.step(1.0 / sim_freq)
            # 调试模式：在 reset 期间也渲染 warmup 的画面
            if render and self.render_mode is not None:
                self.render()

        # 再做一次清理，避免 warmup 结束时残留 crash 车辆
        self._clear_background()

    # ----------------- RL step：在 AbstractEnv 的基础上维护车流 ----------------- #
    def step(self, action):
        # 让 AbstractEnv 完成 ego 控制 + 仿真
        obs, reward, terminated, truncated, info = super().step(action)
        sim_freq = float(self.config["simulation_frequency"])
        pol_freq = float(self.config["policy_frequency"])

        # 在每个决策步之后，更新一次车流：清除驶离 & crashed，按概率增车
        self._clear_background()
        self._spawn_background(self.config["spawn_probability"] * (sim_freq / pol_freq))    # TODO: 完善增车策略，现在是按policy_freq集总生成，不是按simu_freq生成

        return obs, reward, terminated, truncated, info
    
    # ----------------- RL task 定义 ----------------- #
    def _reward(self, action: Action) -> float:
        rewards = self._rewards(action)
        reward = sum(
            self.config.get(name, 0) * reward for name, reward in rewards.items()
        )
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        # 当前车道和纵向位置
        lane = self.road.network.get_lane(self.vehicle.lane_index)
        longi, _ = lane.local_coordinates(self.vehicle.position)
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)

        # ---------- 1) 进度奖励 ----------
        last_longi = getattr(self, "_last_longitudinal", longi)
        delta_s = max(longi - last_longi, 0.0)
        goal_long = float(self.config.get("goal_longitudinal", self.config["road_length"]))
        progress = np.clip(delta_s / goal_long, 0.0, 1.0)

        # ---------- 2) 舒适性奖励（加速度） ----------
        dt = 1.0 / float(self.config["policy_frequency"])
        cur_speed = self.vehicle.speed
        last_speed = getattr(self, "_last_speed", cur_speed)
        acc = (cur_speed - last_speed) / dt

        a_max = float(self.config["comfort_max_accel"])
        comfort = - (min(abs(acc) / a_max, 1.0) ** 2) * dt

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
        self._last_lane_id = curr_lane_id
        self._last_longitudinal = longi
        return {
        "collision_reward": float(self.vehicle.crashed),
        "progress_reward": progress,
        "comfort_reward": comfort,
        "lane_change_reward": lane_changed,
        "punctual_reward": punctual,
        "on_road_reward": float(self.vehicle.on_road),
    }
    

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed, reached the goal, or went off-road."""
        return (
            self.vehicle.crashed
            or self._goal_reached()
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
        speed_min, speed_max = cfg["flow_speed_range"]
        
        behavior_types = cfg.get(
            "behavior_vehicle_types",
            [cfg["other_vehicles_type"]],  # IDM三种风格类型 + 概率
        )
        probs = np.array(cfg.get("behavior_probs", [1.0] * len(behavior_types)),
                        dtype=float)
        probs = probs / probs.sum()

        # 尝试若干次（不同车道+速度），找一个符合安全间距的插入点，成功生成一辆就退出循环
        for _ in range(2 * lanes):
            lane_id = int(self.np_random.integers(lanes))
            lane_index = ("0", "1", lane_id)
            lane = self.road.network.get_lane(lane_index)
            speed = float(self.np_random.uniform(speed_min, speed_max))

            # 检查是否有安全车头时距
            if not self._can_spawn_on_lane(lane, lane_index, speed):
                continue

            # 按概率抽一类风格，创建车辆
            style_idx = int(self.np_random.choice(len(behavior_types), p=probs))
            vehicle_cls = utils.class_from_path(behavior_types[style_idx])
            position = lane.position(0.0, 0.0)
            heading = lane.heading_at(0.0)
            v = vehicle_cls(
                self.road,
                position,
                heading,
                speed,
                target_speed=speed,
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
    def _create_ego(self):
        cfg = self.config
        lane_id = cfg["initial_lane_id"] if cfg["initial_lane_id"] != "random" else int(self.np_random.integers(int(cfg["lanes_count"])))
        lane_index = ("0", "1", int(lane_id))
        lane = self.road.network.get_lane(lane_index)

        # 先把入口附近的一段距离清空，为 ego 腾位置
        clear_radius = cfg["ego_clear_radius"]
        cleaned = []
        for v in self.road.vehicles:
            lane_v = getattr(v, "lane", None)
            if lane_v is lane:
                longi, _ = lane.local_coordinates(v.position)
                if 0.0 <= longi < clear_radius: # 清除入口附近的车辆
                    continue
            cleaned.append(v)
        self.road.vehicles = cleaned

        # 初始化 ego
        longi0 = 0
        position = lane.position(longi0, 0.0)
        heading = lane.heading_at(longi0)
        ego_speed = cfg["ego_speed"]
        ego = self.action_type.vehicle_class(self.road, position, heading, ego_speed, target_speed=ego_speed)

        self.vehicle = ego
        self.controlled_vehicles = [ego]
        self.road.vehicles.append(ego)

        # ====== 初始化奖励相关的历史量 ======
        self._last_speed = ego_speed
        self._last_lane_id = lane_id
        self._last_longitudinal = longi0
        self._has_arrived = False
        self._arrival_time = None

    def _goal_reached(self) -> bool:
        """判断是否到达右侧车道的目标位置 x >= goal_longitudinal。"""
        goal_lane_id = self.config["goal_lane_id"]

        lane_index = self.vehicle.lane_index
        if lane_index[2] != goal_lane_id:
            return False

        lane = self.road.network.get_lane(lane_index)
        longi, _ = lane.local_coordinates(self.vehicle.position)

        goal_long = float(self.config.get("goal_longitudinal", 300.0))
        return longi >= goal_long
    
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