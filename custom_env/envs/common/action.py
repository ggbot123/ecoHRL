from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING, Callable, Union

import numpy as np
from gymnasium import spaces

from custom_env import utils
from custom_env.utils import Vector
from custom_env.vehicle.controller import MDPVehicle
from custom_env.vehicle.dynamics import BicycleVehicle
from custom_env.vehicle.kinematics import Vehicle


if TYPE_CHECKING:
    from custom_env.envs.common.abstract import AbstractEnv

Action = Union[int, np.ndarray]


class ActionType:
    """A type of action specifies its definition space, and how actions are executed in the environment"""

    def __init__(self, env: AbstractEnv, **kwargs) -> None:
        self.env = env
        self.__controlled_vehicle = None

    def space(self) -> spaces.Space:
        """The action space."""
        raise NotImplementedError

    @property
    def vehicle_class(self) -> Callable:
        """
        The class of a vehicle able to execute the action.

        Must return a subclass of :py:class:`highway_env.vehicle.kinematics.Vehicle`.
        """
        raise NotImplementedError

    def act(self, action: Action) -> None:
        """
        Execute the action on the ego-vehicle.

        Most of the action mechanics are actually implemented in vehicle.act(action), where
        vehicle is an instance of the specified :py:class:`highway_env.envs.common.action.ActionType.vehicle_class`.
        Must some pre-processing can be applied to the action based on the ActionType configurations.

        :param action: the action to execute
        """
        raise NotImplementedError

    def get_available_actions(self):
        """
        For discrete action space, return the list of available actions.
        """
        raise NotImplementedError

    @property
    def controlled_vehicle(self):
        """The vehicle acted upon.

        If not set, the first controlled vehicle is used by default."""
        return self.__controlled_vehicle or self.env.vehicle

    @controlled_vehicle.setter
    def controlled_vehicle(self, vehicle):
        self.__controlled_vehicle = vehicle


class ContinuousAction(ActionType):
    """
    An continuous action space for throttle and/or steering angle.

    If both throttle and steering are enabled, they are set in this order: [throttle, steering]

    The space intervals are always [-1, 1], but are mapped to throttle/steering intervals through configurations.
    """

    ACCELERATION_RANGE = (-5, 5.0)
    """Acceleration range: [-x, x], in m/s²."""

    STEERING_RANGE = (-np.pi / 4, np.pi / 4)
    """Steering angle range: [-x, x], in rad."""

    def __init__(
        self,
        env: AbstractEnv,
        acceleration_range: tuple[float, float] | None = None,
        steering_range: tuple[float, float] | None = None,
        speed_range: tuple[float, float] | None = None,
        longitudinal: bool = True,
        lateral: bool = True,
        dynamical: bool = False,
        clip: bool = True,
        **kwargs,
    ) -> None:
        """
        Create a continuous action space.

        :param env: the environment
        :param acceleration_range: the range of acceleration values [m/s²]
        :param steering_range: the range of steering values [rad]
        :param speed_range: the range of reachable speeds [m/s]
        :param longitudinal: enable throttle control
        :param lateral: enable steering control
        :param dynamical: whether to simulate dynamics (i.e. friction) rather than kinematics
        :param clip: clip action to the defined range
        """
        super().__init__(env)
        self.acceleration_range = (
            acceleration_range if acceleration_range else self.ACCELERATION_RANGE
        )
        self.steering_range = steering_range if steering_range else self.STEERING_RANGE
        self.speed_range = speed_range
        self.lateral = lateral
        self.longitudinal = longitudinal
        if not self.lateral and not self.longitudinal:
            raise ValueError(
                "Either longitudinal and/or lateral control must be enabled"
            )
        self.dynamical = dynamical
        self.clip = clip
        self.size = 2 if self.lateral and self.longitudinal else 1
        self.last_action = np.zeros(self.size)

    def space(self) -> spaces.Box:
        return spaces.Box(-1.0, 1.0, shape=(self.size,), dtype=np.float32)

    @property
    def vehicle_class(self) -> Callable:
        return Vehicle if not self.dynamical else BicycleVehicle

    def get_action(self, action: np.ndarray):
        if self.clip:
            action = np.clip(action, -1, 1)
        if self.speed_range:
            (
                self.controlled_vehicle.MIN_SPEED,
                self.controlled_vehicle.MAX_SPEED,
            ) = self.speed_range
        if self.longitudinal and self.lateral:
            return {
                "acceleration": utils.lmap(action[0], [-1, 1], self.acceleration_range),
                "steering": utils.lmap(action[1], [-1, 1], self.steering_range),
            }
        elif self.longitudinal:
            return {
                "acceleration": utils.lmap(action[0], [-1, 1], self.acceleration_range),
                "steering": 0,
            }
        elif self.lateral:
            return {
                "acceleration": 0,
                "steering": utils.lmap(action[0], [-1, 1], self.steering_range),
            }

    def act(self, action: np.ndarray) -> None:
        self.controlled_vehicle.act(self.get_action(action))
        self.last_action = action


class DiscreteAction(ContinuousAction):
    def __init__(
        self,
        env: AbstractEnv,
        acceleration_range: tuple[float, float] | None = None,
        steering_range: tuple[float, float] | None = None,
        longitudinal: bool = True,
        lateral: bool = True,
        dynamical: bool = False,
        clip: bool = True,
        actions_per_axis: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(
            env,
            acceleration_range=acceleration_range,
            steering_range=steering_range,
            longitudinal=longitudinal,
            lateral=lateral,
            dynamical=dynamical,
            clip=clip,
        )
        self.actions_per_axis = actions_per_axis

    def space(self) -> spaces.Discrete:
        return spaces.Discrete(self.actions_per_axis**self.size)

    def act(self, action: int) -> None:
        cont_space = super().space()
        axes = np.linspace(cont_space.low, cont_space.high, self.actions_per_axis).T
        all_actions = list(itertools.product(*axes))
        super().act(all_actions[action])


class DiscreteMetaAction(ActionType):
    """
    An discrete action space of meta-actions: lane changes, and cruise control set-point.
    """

    ACTIONS_ALL = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
    """A mapping of action indexes to labels."""

    ACTIONS_LONGI = {0: "SLOWER", 1: "IDLE", 2: "FASTER"}
    """A mapping of longitudinal action indexes to labels."""

    ACTIONS_LAT = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT"}
    """A mapping of lateral action indexes to labels."""

    def __init__(
        self,
        env: AbstractEnv,
        longitudinal: bool = True,
        lateral: bool = True,
        target_speeds: Vector | None = None,
        **kwargs,
    ) -> None:
        """
        Create a discrete action space of meta-actions.

        :param env: the environment
        :param longitudinal: include longitudinal actions
        :param lateral: include lateral actions
        :param target_speeds: the list of speeds the vehicle is able to track
        """
        super().__init__(env)
        self.longitudinal = longitudinal
        self.lateral = lateral
        self.target_speeds = (
            np.array(target_speeds)
            if target_speeds is not None
            else MDPVehicle.DEFAULT_TARGET_SPEEDS
        )
        self.actions = (
            self.ACTIONS_ALL
            if longitudinal and lateral
            else (
                self.ACTIONS_LONGI
                if longitudinal
                else self.ACTIONS_LAT if lateral else None
            )
        )
        if self.actions is None:
            raise ValueError(
                "At least longitudinal or lateral actions must be included"
            )
        self.actions_indexes = {v: k for k, v in self.actions.items()}

    def space(self) -> spaces.Space:
        return spaces.Discrete(len(self.actions))

    @property
    def vehicle_class(self) -> Callable:
        return functools.partial(MDPVehicle, target_speeds=self.target_speeds)

    def act(self, action: int | np.ndarray) -> None:
        self.controlled_vehicle.act(self.actions[int(action)])

    def get_available_actions(self) -> list[int]:
        """
        Get the list of currently available actions.

        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.

        :return: the list of available actions
        """
        actions = [self.actions_indexes["IDLE"]]
        network = self.controlled_vehicle.road.network
        for l_index in network.side_lanes(self.controlled_vehicle.lane_index):
            if (
                l_index[2] < self.controlled_vehicle.lane_index[2]
                and network.get_lane(l_index).is_reachable_from(
                    self.controlled_vehicle.position
                )
                and self.lateral
            ):
                actions.append(self.actions_indexes["LANE_LEFT"])
            if (
                l_index[2] > self.controlled_vehicle.lane_index[2]
                and network.get_lane(l_index).is_reachable_from(
                    self.controlled_vehicle.position
                )
                and self.lateral
            ):
                actions.append(self.actions_indexes["LANE_RIGHT"])
        if (
            self.controlled_vehicle.speed_index
            < self.controlled_vehicle.target_speeds.size - 1
            and self.longitudinal
        ):
            actions.append(self.actions_indexes["FASTER"])
        if self.controlled_vehicle.speed_index > 0 and self.longitudinal:
            actions.append(self.actions_indexes["SLOWER"])
        return actions


class ParamLaneAccelAction(ActionType):
    """
    Parameterized action for RL:
    （默认）动作空间：Box([-1, -1], [1, 1])
        - action[0] ∈ [-1, 1] : 车道决策，三等分区间
            [-1,   -1/3)  -> LANE_LEFT
            [-1/3,  1/3]  -> KEEP
            ( 1/3,  1]    -> LANE_RIGHT
        - action[1] ∈ [-1, 1] : 纵向加速度的归一化值，经线性映射到物理加速度
    同时兼容 Dict 形式：
        {
            "lane_change": 0/1/2,          # 0: KEEP, 1: LANE_LEFT, 2: LANE_RIGHT
            "acceleration": acc_phys,      # 物理加速度 [m/s^2]
        }
    steering 始终由车模内部（如 RLControlledVehicle）调用自己的 steering/换道控制器计算。
    """

    ACCELERATION_RANGE = (-5, 5.0)
    """Acceleration range: [-x, x], in m/s²."""
    LANE_ACTIONS = ["KEEP", "LANE_LEFT", "LANE_RIGHT"]
    """Default lane actions: KEEP, LANE_LEFT, LANE_RIGHT."""

    def __init__(
        self,
        env: AbstractEnv,
        acceleration_range: tuple[float, float] | None = None,
        lane_actions: list[str] | None = None,
        clip: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(env)

        # 加速度物理范围
        self.acc_min, self.acc_max = acceleration_range if acceleration_range is not None else self.ACCELERATION_RANGE
        # 车道动作语义（默认：0: KEEP, 1: LANE_LEFT, 2: LANE_RIGHT）
        self.lane_actions = lane_actions if lane_actions is not None else self.LANE_ACTIONS
        assert (len(self.lane_actions) == 3), "lane_actions 期望 3 个元素：KEEP / LANE_LEFT / LANE_RIGHT"
        assert "KEEP" in self.lane_actions
        assert "LANE_LEFT" in self.lane_actions
        assert "LANE_RIGHT" in self.lane_actions
        # 动作空间定义为 2 维 Box： [lane_scalar, acceleration_scalar]
        self._space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._lane_index_by_name = {
            name: idx for idx, name in enumerate(self.lane_actions)
        }
        self.clip = clip
    
    def space(self) -> spaces.Space:
        """
        默认暴露给 RL 的动作空间：Box([-1, -1], [1, 1])。
        对 action[0] 做三等分映射到车道选择，对 action[1] 线性映射到物理加速度。
        """
        return self._space
    
    @property
    def vehicle_class(self) -> Callable:
        from custom_env.vehicle.controller import RLControlledVehicle  # 延迟导入，避免循环引用
        return RLControlledVehicle
    
    # ---------- 内部工具函数 ----------
    def _scalar_to_lane_index(self, x: float) -> int:
        """将 [-1, 1] 上的连续标量映射为 lane_actions 中的一个离散 index。"""
        x = float(np.clip(x, -1.0, 1.0))
        if x < -1/3:
            return self._lane_index_by_name["LANE_LEFT"]
        elif x > 1/3:
            return self._lane_index_by_name["LANE_RIGHT"]
        else:
            return self._lane_index_by_name["KEEP"]

    def _map_acceleration(self, a: float, normalized: bool) -> float:
        """
        将加速度标量映射到物理加速度范围 [acc_min, acc_max]。
        - 如果 normalized=True，则认为 a ∈ [-1, 1]，先截断再线性映射；
        - 如果 normalized=False，则认为 a 已经是物理加速度，只做范围截断。
        """
        if normalized:
            if self.clip:
                a = float(np.clip(a, -1.0, 1.0))
            return float(utils.lmap(a, [-1.0, 1.0], [self.acc_min, self.acc_max]))
        else:
            return float(np.clip(a, self.acc_min, self.acc_max))
    
    # ---------- 动作执行 ----------
    def act(self, action: Action) -> None:
        """
        将 RL 的动作翻译给受控车辆（一般是 RLControlledVehicle）。
        支持以下形式：
        1) Dict：
           {
               "lane_change": 0/1/2,        # 对应 lane_actions 中元素
               "acceleration": acc_phys,    # 物理加速度 [m/s^2]
           }
        2) np.ndarray / list / tuple，形状 (2,)：
           action[0] ∈ [-1, 1] : 映射车道选择
           action[1] ∈ [-1, 1] : 映射到物理加速度
        """
        ego = self.controlled_vehicle
        lane_cmd_idx = self._lane_index_by_name["KEEP"]
        acc_phys = 0.0

        # ---- 1) Dict 形式：直接读取离散 + 物理加速度 ----
        if isinstance(action, dict):
            if "lane_change" in action:
                lane_cmd_idx = int(action["lane_change"])
            if "acceleration" in action:
                # 认为已经是物理加速度
                acc_phys = self._map_acceleration(float(action["acceleration"]), normalized=False)

        # ---- 2) Box / 序列形式：[-1,1]^2 ----
        elif isinstance(action, (np.ndarray, list, tuple)):
            if len(action) >= 1:
                a0 = float(action[0])
                lane_cmd_idx = self._scalar_to_lane_index(a0)
            if len(action) >= 2:
                a1 = float(action[1])
                acc_phys = self._map_acceleration(a1, normalized=True)

        # 防御性截断 lane index
        lane_cmd_idx = int(np.clip(lane_cmd_idx, 0, len(self.lane_actions) - 1))

        # 把解析出的 “离散车道选择 + 物理加速度” 交给车模
        ego.act({"lane_change": lane_cmd_idx, "acceleration": acc_phys})

class MultiAgentAction(ActionType):
    def __init__(self, env: AbstractEnv, action_config: dict, **kwargs) -> None:
        super().__init__(env)
        self.action_config = action_config
        self.agents_action_types = []
        for vehicle in self.env.controlled_vehicles:
            action_type = action_factory(self.env, self.action_config)
            action_type.controlled_vehicle = vehicle
            self.agents_action_types.append(action_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple(
            [action_type.space() for action_type in self.agents_action_types]
        )

    @property
    def vehicle_class(self) -> Callable:
        return action_factory(self.env, self.action_config).vehicle_class

    def act(self, action: Action) -> None:
        assert isinstance(action, tuple)
        for agent_action, action_type in zip(action, self.agents_action_types):
            action_type.act(agent_action)

    def get_available_actions(self):
        return itertools.product(
            *[
                action_type.get_available_actions()
                for action_type in self.agents_action_types
            ]
        )


def action_factory(env: AbstractEnv, config: dict) -> ActionType:
    if config["type"] == "ContinuousAction":
        return ContinuousAction(env, **config)
    if config["type"] == "DiscreteAction":
        return DiscreteAction(env, **config)
    elif config["type"] == "DiscreteMetaAction":
        return DiscreteMetaAction(env, **config)
    elif config["type"] == "MultiAgentAction":
        return MultiAgentAction(env, **config)
    elif config["type"] == "ParamLaneAccelAction":
        return ParamLaneAccelAction(env, **config)
    else:
        raise ValueError("Unknown action type")
