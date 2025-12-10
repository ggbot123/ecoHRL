from __future__ import annotations

import numpy as np

from custom_env import utils
from custom_env.road.road import LaneIndex, Road, Route
from custom_env.utils import Vector
from custom_env.vehicle.controller import ControlledVehicle
from custom_env.vehicle.kinematics import Vehicle


class IDMVehicle(ControlledVehicle):
    """
    A vehicle using both a longitudinal and a lateral decision policies.

    - Longitudinal: the IDM model computes an acceleration given the preceding vehicle's distance and speed.
    - Lateral: the MOBIL model decides when to change lane by maximizing the acceleration of nearby vehicles.
    """

    # Longitudinal policy parameters
    ACC_MAX = 6.0  # [m/s2]
    """Maximum acceleration."""

    COMFORT_ACC_MAX = 3.0  # [m/s2]
    """Desired maximum acceleration."""

    COMFORT_ACC_MIN = -5.0  # [m/s2]
    """Desired maximum deceleration."""

    DISTANCE_WANTED = 5.0 + ControlledVehicle.LENGTH  # [m]
    """Desired jam distance to the front vehicle."""

    TIME_WANTED = 1.5  # [s]
    """Desired time gap to the front vehicle."""

    DELTA = 4.0  # []
    """Exponent of the velocity term."""

    DELTA_RANGE = [3.5, 4.5]
    """Range of delta when chosen randomly."""

    # Lateral policy parameters
    POLITENESS = 0.0  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(
        self,
        road: Road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index: int = None,
        target_speed: float = None,
        route: Route = None,
        enable_lane_change: bool = True,
        timer: float = None,
    ):
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route
        )
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position) * np.pi) % self.LANE_CHANGE_DELAY

    def randomize_behavior(self):
        self.DELTA = self.road.np_random.uniform(
            low=self.DELTA_RANGE[0], high=self.DELTA_RANGE[1]
        )

    @classmethod
    def create_from(cls, vehicle: ControlledVehicle) -> IDMVehicle:
        """
        Create a new vehicle from an existing one.

        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(
            vehicle.road,
            vehicle.position,
            heading=vehicle.heading,
            speed=vehicle.speed,
            target_lane_index=vehicle.target_lane_index,
            target_speed=vehicle.target_speed,
            route=vehicle.route,
            timer=getattr(vehicle, "timer", None),
        )
        return v

    def act(self, action: dict | str = None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions
        of acceleration and lane changes on its own, based on the IDM and MOBIL models.

        :param action: the action
        """
        if self.crashed:
            return
        action = {}
        # Lateral: MOBIL
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()
        action["steering"] = self.steering_control(self.target_lane_index)
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )

        # Longitudinal: IDM
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
            self, self.lane_index
        )
        action["acceleration"] = self.acceleration(
            ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
        )
        # When changing lane, check both current and target lanes
        if self.lane_index != self.target_lane_index:
            front_vehicle, rear_vehicle = self.road.neighbour_vehicles(
                self, self.target_lane_index
            )
            target_idm_acceleration = self.acceleration(
                ego_vehicle=self, front_vehicle=front_vehicle, rear_vehicle=rear_vehicle
            )
            action["acceleration"] = min(
                action["acceleration"], target_idm_acceleration
            )
        # action['acceleration'] = self.recover_from_stop(action['acceleration'])
        action["acceleration"] = np.clip(
            action["acceleration"], -self.ACC_MAX, self.ACC_MAX
        )
        # Skip ControlledVehicle.act(), or the command will be overridden.
        Vehicle.act(self, action)

    def step(self, dt: float):
        """
        Step the simulation.

        Increases a timer used for decision policies, and step the vehicle dynamics.

        :param dt: timestep
        """
        self.timer += dt
        super().step(dt)

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """
        Compute an acceleration command with the Intelligent Driver Model.

        The acceleration is chosen so as to:
        - reach a target speed;
        - maintain a minimum safety distance (and safety time) w.r.t the front vehicle.

        :param ego_vehicle: the vehicle whose desired acceleration is to be computed. It does not have to be an
                            IDM vehicle, which is why this method is a class method. This allows an IDM vehicle to
                            reason about other vehicles behaviors even though they may not IDMs.
        :param front_vehicle: the vehicle preceding the ego-vehicle
        :param rear_vehicle: the vehicle following the ego-vehicle
        :return: the acceleration command for the ego-vehicle [m/s2]
        """
        if not ego_vehicle or not isinstance(ego_vehicle, Vehicle):
            return 0
        ego_target_speed = getattr(ego_vehicle, "target_speed", 0)
        if ego_vehicle.lane and ego_vehicle.lane.speed_limit is not None:
            ego_target_speed = np.clip(
                ego_target_speed, 0, ego_vehicle.lane.speed_limit
            )
        acceleration = self.COMFORT_ACC_MAX * (
            1
            - np.power(
                max(ego_vehicle.speed, 0) / abs(utils.not_zero(ego_target_speed)),
                self.DELTA,
            )
        )

        if front_vehicle:
            d = ego_vehicle.lane_distance_to(front_vehicle)
            acceleration -= self.COMFORT_ACC_MAX * np.power(
                self.desired_gap(ego_vehicle, front_vehicle) / utils.not_zero(d), 2
            )
        return acceleration

    def desired_gap(
        self,
        ego_vehicle: Vehicle,
        front_vehicle: Vehicle = None,
        projected: bool = True,
    ) -> float:
        """
        Compute the desired distance between a vehicle and its leading vehicle.

        :param ego_vehicle: the vehicle being controlled
        :param front_vehicle: its leading vehicle
        :param projected: project 2D velocities in 1D space
        :return: the desired distance between the two [m]
        """
        d0 = self.DISTANCE_WANTED
        tau = self.TIME_WANTED
        ab = -self.COMFORT_ACC_MAX * self.COMFORT_ACC_MIN
        dv = (
            np.dot(ego_vehicle.velocity - front_vehicle.velocity, ego_vehicle.direction)
            if projected
            else ego_vehicle.speed - front_vehicle.speed
        )
        d_star = (
            d0 + ego_vehicle.speed * tau + ego_vehicle.speed * dv / (2 * np.sqrt(ab))
        )
        return d_star

    def change_lane_policy(self) -> None:
        """
        Decide when to change lane.

        Based on:
        - frequency;
        - closeness of the target lane;
        - MOBIL model.
        """
        # If a lane change is already ongoing
        if self.lane_index != self.target_lane_index:
            # If we are on correct route but bad lane: abort it if someone else is already changing into the same lane
            if self.lane_index[:2] == self.target_lane_index[:2]:
                for v in self.road.vehicles:
                    if (
                        v is not self
                        and v.lane_index != self.target_lane_index
                        and isinstance(v, ControlledVehicle)
                        and v.target_lane_index == self.target_lane_index
                    ):
                        d = self.lane_distance_to(v)
                        d_star = self.desired_gap(self, v)
                        if 0 < d < d_star:
                            self.target_lane_index = self.lane_index
                            break
            return

        # else, at a given frequency,
        if not utils.do_every(self.LANE_CHANGE_DELAY, self.timer) or isinstance(self, DefensiveIDMVehicle):
            return
        self.timer = 0

        # decide to make a lane change
        for lane_index in self.road.network.side_lanes(self.lane_index):
            # Is the candidate lane close enough?
            if not self.road.network.get_lane(lane_index).is_reachable_from(
                self.position
            ):
                continue
            # Only change lane when the vehicle is moving
            if np.abs(self.speed) < 1:
                continue
            # Does the MOBIL model recommend a lane change?
            if self.mobil(lane_index):
                self.target_lane_index = lane_index

    def mobil(self, lane_index: LaneIndex) -> bool:
        """
        MOBIL lane change model: Minimizing Overall Braking Induced by a Lane change

            The vehicle should change lane only if:
            - after changing it (and/or following vehicles) can accelerate more;
            - it doesn't impose an unsafe braking on its new following vehicle.

        :param lane_index: the candidate lane for the change
        :return: whether the lane change should be performed
        """
        # Is the maneuver unsafe for the new following vehicle?
        new_preceding, new_following = self.road.neighbour_vehicles(self, lane_index)
        new_following_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=new_preceding
        )
        new_following_pred_a = self.acceleration(
            ego_vehicle=new_following, front_vehicle=self
        )
        if new_following_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False

        # Do I have a planned route for a specific lane which is safe for me to access?
        old_preceding, old_following = self.road.neighbour_vehicles(self)
        self_pred_a = self.acceleration(ego_vehicle=self, front_vehicle=new_preceding)
        if self.route and self.route[0][2] is not None:
            # Wrong direction
            if np.sign(lane_index[2] - self.target_lane_index[2]) != np.sign(
                self.route[0][2] - self.target_lane_index[2]
            ):
                return False
            # Unsafe braking required
            elif self_pred_a < -self.LANE_CHANGE_MAX_BRAKING_IMPOSED:
                return False

        # Is there an acceleration advantage for me and/or my followers to change lane?
        else:
            self_a = self.acceleration(ego_vehicle=self, front_vehicle=old_preceding)
            old_following_a = self.acceleration(
                ego_vehicle=old_following, front_vehicle=self
            )
            old_following_pred_a = self.acceleration(
                ego_vehicle=old_following, front_vehicle=old_preceding
            )
            jerk = (
                self_pred_a
                - self_a
                + self.POLITENESS
                * (
                    new_following_pred_a
                    - new_following_a
                    + old_following_pred_a
                    - old_following_a
                )
            )
            if jerk < self.LANE_CHANGE_MIN_ACC_GAIN:
                return False

        # All clear, let's go!
        return True

    def recover_from_stop(self, acceleration: float) -> float:
        """
        If stopped on the wrong lane, try a reversing maneuver.

        :param acceleration: desired acceleration from IDM
        :return: suggested acceleration to recover from being stuck
        """
        stopped_speed = 5
        safe_distance = 200
        # Is the vehicle stopped on the wrong lane?
        if self.target_lane_index != self.lane_index and self.speed < stopped_speed:
            _, rear = self.road.neighbour_vehicles(self)
            _, new_rear = self.road.neighbour_vehicles(
                self, self.road.network.get_lane(self.target_lane_index)
            )
            # Check for free room behind on both lanes
            if (not rear or rear.lane_distance_to(self) > safe_distance) and (
                not new_rear or new_rear.lane_distance_to(self) > safe_distance
            ):
                # Reverse
                return -self.COMFORT_ACC_MAX / 2
        return acceleration


class StyledIDMVehicle(IDMVehicle):
    """
    在 IDM 基础上加：
    - 不同风格的参数均值（期望速度/间距/时距/加减速/礼貌等）
    - 围绕这些均值进行采样的 std
    - imperfection: 加速度噪声强度
    """

    DELTA_LOW = 3.5
    DELTA_UPP = 4.5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.imperfection = getattr(self, "IMPERFECTION_MEAN", 0.0)

    def randomize_behavior(self):
        """在每辆车的风格均值附近采样具体参数。"""
        rng = self.road.np_random

        # 期望速度
        v0 = rng.uniform(self.DESIRED_SPEED_MIN, self.DESIRED_SPEED_MAX)
        self.target_speed = max(0.1, v0)

        # 期望车头时距 / 间距
        self.TIME_WANTED = max(
            0.2, rng.normal(self.TIME_WANTED_MEAN, self.TIME_WANTED_SIGMA)
        )
        self.DISTANCE_WANTED = max(
            1.0, rng.normal(self.DISTANCE_WANTED_MEAN, self.DISTANCE_WANTED_SIGMA)
        )

        # IDM 纵向参数
        self.DELTA = max(1.0, rng.uniform(self.DELTA_LOW, self.DELTA_UPP))
        self.COMFORT_ACC_MAX = max(
            0.1, rng.normal(self.COMFORT_ACC_MAX_MEAN, self.COMFORT_ACC_MAX_SIGMA)
        )
        self.COMFORT_ACC_MIN = min(
            -0.1, rng.normal(self.COMFORT_ACC_MIN_MEAN, self.COMFORT_ACC_MIN_SIGMA)
        )

        # MOBIL 换道参数
        self.POLITENESS = float(
            np.clip(
                rng.normal(self.POLITENESS_MEAN, self.POLITENESS_SIGMA),
                0.0,
                1.0,
            )
        )
        self.LANE_CHANGE_MIN_ACC_GAIN = max(
            0.0,
            rng.normal(
                self.LANE_CHANGE_MIN_ACC_GAIN_MEAN,
                self.LANE_CHANGE_MIN_ACC_GAIN_SIGMA,
            ),
        )
        self.LANE_CHANGE_MAX_BRAKING_IMPOSED = max(
            0.1,
            rng.normal(
                self.LANE_CHANGE_MAX_BRAKING_IMPOSED_MEAN,
                self.LANE_CHANGE_MAX_BRAKING_IMPOSED_SIGMA,
            ),
        )

        # 驾驶员不完美性
        self.imperfection = max(
            0.0, rng.normal(self.IMPERFECTION_MEAN, self.IMPERFECTION_SIGMA)
        )

    def acceleration(
        self,
        ego_vehicle: ControlledVehicle,
        front_vehicle: Vehicle = None,
        rear_vehicle: Vehicle = None,
    ) -> float:
        """
        先用 IDM 算“理想加速度”，再叠加一个与 imperfection 成正比的噪声。
        """
        base_acc = super().acceleration(ego_vehicle, front_vehicle, rear_vehicle)  # IDM 原公式:contentReference[oaicite:3]{index=3}

        imp = getattr(self, "imperfection", 0.0)
        if imp > 0 and self.road is not None:
            noise = self.road.np_random.normal(0.0, imp * self.COMFORT_ACC_MAX)
            base_acc += noise

        return base_acc

class NormalIDMVehicle(StyledIDMVehicle):
    """正常驾驶：接近期望速度，适中时距，适中礼貌。"""
    DESIRED_SPEED_MIN = 12
    DESIRED_SPEED_MAX = 15

    TIME_WANTED_MEAN = 1.5
    TIME_WANTED_SIGMA = 0.15

    DISTANCE_WANTED_MEAN = 5.0 + ControlledVehicle.LENGTH
    DISTANCE_WANTED_SIGMA = 1.0

    COMFORT_ACC_MAX_MEAN = 3.0
    COMFORT_ACC_MAX_SIGMA = 0.2

    COMFORT_ACC_MIN_MEAN = -5.0
    COMFORT_ACC_MIN_SIGMA = 0.2

    POLITENESS_MEAN = 0.2
    POLITENESS_SIGMA = 0.1

    LANE_CHANGE_MIN_ACC_GAIN_MEAN = 0.3    # 不会为了很小的收益就换道
    LANE_CHANGE_MIN_ACC_GAIN_SIGMA = 0.0

    LANE_CHANGE_MAX_BRAKING_IMPOSED_MEAN = 2.0
    LANE_CHANGE_MAX_BRAKING_IMPOSED_SIGMA = 0.3

    LANE_CHANGE_DELAY = 1.0

    IMPERFECTION_MEAN = 0.0
    IMPERFECTION_SIGMA = 0.1


class AggressiveIDMVehicle(StyledIDMVehicle):
    """激进驾驶：高期望速度，短时距，大加速度，小礼貌。"""
    DESIRED_SPEED_MIN = 12
    DESIRED_SPEED_MAX = 15

    TIME_WANTED_MEAN = 1.0
    TIME_WANTED_SIGMA = 0.15

    DISTANCE_WANTED_MEAN = 3.0 + ControlledVehicle.LENGTH
    DISTANCE_WANTED_SIGMA = 0.5

    COMFORT_ACC_MAX_MEAN = 4.0
    COMFORT_ACC_MAX_SIGMA = 0.3

    COMFORT_ACC_MIN_MEAN = -6.0
    COMFORT_ACC_MIN_SIGMA = 0.3

    POLITENESS_MEAN = 0.0           # 不管别人
    POLITENESS_SIGMA = 0.05

    LANE_CHANGE_MIN_ACC_GAIN_MEAN = 0.2   # 为了很小的加速收益也愿意变道（speed_gain 大）
    LANE_CHANGE_MIN_ACC_GAIN_SIGMA = 0.0

    LANE_CHANGE_MAX_BRAKING_IMPOSED_MEAN = 3.0  # 允许后车为自己猛踩一点刹车
    LANE_CHANGE_MAX_BRAKING_IMPOSED_SIGMA = 0.3

    LANE_CHANGE_DELAY = 1.0

    IMPERFECTION_MEAN = 0.0          # 更不稳定
    IMPERFECTION_SIGMA = 0.1


class DefensiveIDMVehicle(StyledIDMVehicle):
    """保守驾驶：低期望速度，长时距，小加速度，高礼貌。"""
    DESIRED_SPEED_MIN = 8
    DESIRED_SPEED_MAX = 10

    TIME_WANTED_MEAN = 2.0
    TIME_WANTED_SIGMA = 0.15

    DISTANCE_WANTED_MEAN = 7.0 + ControlledVehicle.LENGTH
    DISTANCE_WANTED_SIGMA = 1.0

    COMFORT_ACC_MAX_MEAN = 2.0
    COMFORT_ACC_MAX_SIGMA = 0.3

    COMFORT_ACC_MIN_MEAN = -3.0
    COMFORT_ACC_MIN_SIGMA = 0.3

    POLITENESS_MEAN = 0.5         # 很讲礼貌
    POLITENESS_SIGMA = 0.1

    LANE_CHANGE_MIN_ACC_GAIN_MEAN = 3.0    # 必须收益明显才愿意变道
    LANE_CHANGE_MIN_ACC_GAIN_SIGMA = 0.0

    LANE_CHANGE_MAX_BRAKING_IMPOSED_MEAN = 1.0  # 不愿让别人为自己猛刹
    LANE_CHANGE_MAX_BRAKING_IMPOSED_SIGMA = 0.3

    LANE_CHANGE_DELAY = 1.0

    IMPERFECTION_MEAN = 0.0       # 也有一点反应误差，但比激进小
    IMPERFECTION_SIGMA = 0.1
