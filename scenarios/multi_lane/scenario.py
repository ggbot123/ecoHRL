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
                "lanes_count": 4,
                "road_length": 500.0,
                "speed_limit": 15.0,          # 限速 [m/s]

                # 交通流设置
                "spawn_probability": 0.07,       # 每个仿真步生成一辆新车的概率
                "flow_speed_range": [10.0, 15.0],   # 环境车 初始 + 目标 速度
                "spawn_min_gap": 10.0,           # 入口附近的最小空间间距 [m]
                "spawn_min_t_headway": 1.5,      # 最小时间车头时距 [s]
                "other_vehicles_type": "custom_env.vehicle.behavior.IDMVehicle",
                "vid": 0,

                # ego设置
                "controlled_vehicles": 1,
                "ego_speed": 10.0,        # [m/s]
                "initial_lane_id": 2,
                "warmup_time": 30.0,             # 只跑环境车的时间 [s]
                "ego_clear_radius": 10.0,        # 在插入ego前，清除入口附近多远范围内的车辆 [m]
                
                # 渲染设置
                "show_trajectories": False,      # True 时记录并显示车辆轨迹
                "warmup_render": False,          # True 时在 reset 期间也渲染 warmup 画面
                "real_time_rendering": False,     # True 时在 step 期间渲染时加 sleep，变成肉眼可看速度

                # 观测-动作-奖励空间配置
                "observation": {
                    "type": "Kinematics",
                    "normalize": False,
                },
                "action": {
                    "type": "DiscreteMetaAction",
                    "target_speeds": np.linspace(10, 20, 3),  # [m/s]，设置MDPVehicle可选目标速度
                },
                "collision_reward": -10,  # The reward received when colliding with a vehicle.
                "progress_reward": 1,  # The reward received when moving forward.
                "comfort_reward": 0.2,  # The reward received when accelerating / decelerating.
                "lane_change_reward": -0.5,  # The reward received at each lane change action.
                "punctual_reward": 10,  # The reward received when reaching the goal on time.
                "reward_speed_range": [10, 15], # TODO: calculated by goal position
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
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )
        return {
            "collision_reward": float(self.vehicle.crashed),
            "on_road_reward": float(self.vehicle.on_road),
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed or reached the goal."""
        return (
            self.vehicle.crashed
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

        # 尝试若干次（不同车道+速度），找一个符合安全间距的插入点
        for _ in range(2 * lanes):
            lane_id = int(self.np_random.integers(lanes))
            lane_index = ("0", "1", lane_id)
            lane = self.road.network.get_lane(lane_index)
            speed = float(self.np_random.uniform(speed_min, speed_max))

            # 检查初始速度下该车道是否有安全车头时距
            if not self._can_spawn_on_lane(lane, lane_index, speed):
                continue

            vehicle_type = utils.class_from_path(cfg["other_vehicles_type"])
            position = lane.position(0.0, 0.0)
            heading = lane.heading_at(0.0)
            v = vehicle_type(self.road, position, heading, speed, target_speed=speed)   # TODO: target_speed按不同风格车辆设定，设为自由流车速/道路限速的一定比率，附加风格内部的随机性
            cfg["vid"] += 1

            # TODO: 更丰富的随机参数（delta + desire distance & time headway + accel range + politeness...），按不同风格设置，附加风格内部的随机性
            if hasattr(v, "randomize_behavior"):
                v.randomize_behavior()
            v.lane = lane
            v.lane_index = lane_index
            v.vid = cfg["vid"]

            self.road.vehicles.append(v)
            break  # 成功生成一辆，就退出循环

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
        lane_id = cfg.get("initial_lane_id")
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