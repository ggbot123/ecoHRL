import numpy as np

from custom_env.envs.common.abstract import AbstractEnv
from custom_env.road.road import Road, RoadNetwork
from custom_env.envs.common.action import Action
from custom_env import utils
from custom_env.vehicle.objects import Obstacle

Observation = np.ndarray

class BusStop(Obstacle):
    """
    静态矩形公交站台，沿道路方向放置。
    length: 沿道路方向长度
    width:  垂直道路方向宽度（从路缘向右侧延伸）
    """
    LENGTH = 20.0  # m，沿 x 方向
    WIDTH = 3.0    # m，可以自己调宽一点，比如 3~4m

class MultiLaneEnv(AbstractEnv):
    """
    四车道直路 + 顺序交通流 + 30s 预热 + 更安全的生成逻辑
    - 道路：节点 "0" -> "1" 的四车道直路，长度 road_length
    - 环境车：从左端（x=0）按概率生成，跑到右端后删除
    - warmup：先只跑环境车 warmup_time 秒，再在入口插入 ego
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,  # 例如 10fps，对应你的 policy_frequency=10Hz
    }
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
                "flow_speed_range": [10.0, 10.0],   # 环境车 初始速度
                "speed_distribution": "Uniform",
                # "flow_speed_range": [12.0, 2, 4],   # 环境车 初始速度
                # "speed_distribution": "Gaussian",
                "spawn_min_gap": 10.0,           # 入口附近的最小空间间距 [m]
                "spawn_min_t_headway": 1.5,      # 最小时间车头时距 [s]
                "behavior_vehicle_types": [     # 三种风格 IDM 类型及其概率
                    "custom_env.vehicle.behavior.NormalIDMVehicle",
                    "custom_env.vehicle.behavior.AggressiveIDMVehicle",
                    "custom_env.vehicle.behavior.DefensiveIDMVehicle",
                ],
                "behavior_probs": [0.4, 0.3, 0.3],
                "behavior_lane_probs": [
                    [0.6, 0.3, 0.1],   # 0 号车道
                    [0.6, 0.3, 0.1],   # 1 号车道
                    [0.4, 0.3, 0.3],   # 2 号车道
                ],
                "vid": 0,

                # ego设置
                "controlled_vehicles": 1,
                "ego_speed": 10.0,        # [m/s]
                # "initial_lane_id": 1,
                "initial_lane_id": "random",
                "warmup_time": 100.0,             # 只跑环境车的时间 [s]
                "warmup_each_episode": False,
                "ego_clear_radius": 10.0,        # 在插入ego前，清除入口附近多远范围内的车辆 [m]
                
                # 观测-动作-奖励空间配置
                "observation": {
                    "type": "Kinematics",
                    "normalize": False,
                    "see_behind": False,
                    "include_obstacles": False,
                },
                "action": {
                    "type": "ParamLaneAccelAction",
                    "acceleration_range": [-5.0, 5.0],
                    "lane_actions": ["KEEP", "LANE_LEFT", "LANE_RIGHT"],
                },
                "goal_longitudinal": 400.0,        # 任务 / 目标设置
                "goal_lane_id": 2,
                "punctual_time_window": [30.0, 40.0],   # punctual arrival reward
                "punctual_time_target": 35.0,
                "punctual_reward": 10.0,
                "collision_reward": -10.0,  # collision penalty
                "progress_reward": 10.0,  # progress reward
                "comfort_reward": 0.7,  # comfort reward
                "comfort_max_accel": 3.0, 
                "lane_change_reward": -0.5,  # lane change penalty
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
        self._create_bus_stop()

    # ----------------- reset：预热 + 插入 ego ----------------- #
    def _reset(self):
        """
        - 第一次 reset：建路 + 全局 warmup 交通流 + 插入 ego；
        - 后续 reset：保留现有路网和交通流，只移除旧 ego、清理一下车流，再插入新的 ego。
        """
        # 每次都重置交通流，用于测试，以保证各个episode之间独立
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
                self._clear_background()
        self._create_ego()

    def _warmup(self, render: bool = False):
        """只跑环境车 warmup_time 秒，可以选择是否渲染出来看。"""
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
            t = k / sim_freq  # 当前 warmup 时间 [s]
            times.append(t)
            avg_speeds.append(avg_speed)
            
            self.road.act()
            self.road.step(1.0 / sim_freq)
            # 调试模式：在 reset 期间也渲染 warmup 的画面
            if render and self.render_mode is not None:
                self.render()

        # 再做一次清理，避免 warmup 结束时残留 crash 车辆
        self._clear_background()

        self._warmup_times = np.asarray(times, dtype=float)
        self._warmup_avg_speeds = np.asarray(avg_speeds, dtype=float)

    # ----------------- RL step：在 AbstractEnv 的基础上维护车流 ----------------- #
    def step(self, action):
        # 让 AbstractEnv 完成 ego 控制 + 仿真
        obs, reward, terminated, truncated, info = super().step(action)

        # 把“加权后的分项奖励”塞进 info，方便 callback 从 infos 里读
        weighted = getattr(self, "_last_weighted_rewards", None)
        if isinstance(info, dict) and weighted is not None:
            info["reward_components"] = dict(weighted)

        sim_freq = float(self.config["simulation_frequency"])
        pol_freq = float(self.config["policy_frequency"])

        # 在每个决策步之后，更新一次车流：清除驶离 & crashed，按概率增车
        self._clear_background()
        self._spawn_background(self.config["spawn_probability"] * (sim_freq / pol_freq))    # TODO: 完善增车策略，现在是按policy_freq集总生成，不是按simu_freq生成

        return obs, reward, terminated, truncated, info
    
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
        global_probs = np.array(
            cfg.get("behavior_probs", [1.0] * n_types),
            dtype=float,
        )
        global_probs = global_probs / global_probs.sum()

        def _get_lane_behavior_probs(lane_id: int) -> np.ndarray:
            """返回当前 lane_id 的 behavior 概率向量"""
            if lane_probs_all is not None:
                try:
                    lane_row = np.asarray(lane_probs_all[lane_id], dtype=float)
                    if lane_row.shape[0] == n_types:
                        lane_row = lane_row / lane_row.sum()
                        return lane_row
                except (IndexError, TypeError, ValueError):
                    pass
            # 回退：使用全局分布
            return global_probs

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
        lanes = int(cfg["lanes_count"])
        lane_id = int(self.np_random.integers(lanes)) if cfg["initial_lane_id"] == "random" else int(cfg["initial_lane_id"])
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

        # 初始化奖励相关的历史量
        self._last_speed = ego_speed
        self._last_lane_id = lane_id
        self._last_longitudinal = longi0
        self._has_arrived = False
        self._arrival_time = None

    def _goal_reached(self) -> bool:
        """在目标车道且 x >= goal_longitudinal"""
        if self.vehicle.lane_index[2] != int(self.config["goal_lane_id"]):
            return False
        return self._goal_longitudinal_reached()
    
    def _goal_longitudinal_reached(self) -> bool:
        """x >= goal_longitudinal（不要求在目标车道）"""
        lane = self.road.network.get_lane(self.vehicle.lane_index)
        longi, _ = lane.local_coordinates(self.vehicle.position)
        goal_long = float(self.config["goal_longitudinal"])
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
    
    def _create_bus_stop(self):
        lane_index = ("0", "1", int(self.config["lanes_count"]) - 1)
        lane = self.road.network.get_lane(lane_index)

        center_long = float(self.config.get("goal_longitudinal", 300.0))  # 以 x=300 为中心
        bus_length = BusStop.LENGTH
        bus_width = BusStop.WIDTH
        lane_half_width = getattr(lane, "width", 4.0) / 2.0
        margin = 0.5  # 车道右缘和站台中线之间留一点间隙
        lateral_center = lane_half_width + margin + bus_width / 2.0

        position = lane.position(center_long, lateral_center)
        heading = lane.heading_at(center_long)

        # 创建 BusStop 对象并加入 road.objects，交给 viewer 渲染
        bus_stop = BusStop(self.road, position, heading)
        if not hasattr(self.road, "objects"):
            self.road.objects = []
        self.road.objects.append(bus_stop)