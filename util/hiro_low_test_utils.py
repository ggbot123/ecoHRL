import ast
import csv
import json
from typing import Sequence, Any

import gymnasium as gym
import numpy as np

from custom_env.vehicle.kinematics import Vehicle


def make_vehicle(
    road,
    state: Sequence[float],
    vehicle_cls,
    *,
    target_lane_index=None,
) -> Vehicle:
    x, y, vx, vy = [float(s) for s in state]
    speed = float(np.hypot(vx, vy))
    if speed > 1e-6:
        heading = float(np.arctan2(vy, vx))
    else:
        heading = 0.0
    pos = np.array([x, y], dtype=float)

    lane_index = road.network.get_closest_lane_index(pos, heading)
    lane = road.network.get_lane(lane_index)

    if speed <= 1e-6:
        longi, _ = lane.local_coordinates(pos)
        heading = float(lane.heading_at(longi))

    if target_lane_index is None:
        target_lane_index = lane_index

    try:
        v = vehicle_cls(
            road,
            pos,
            heading=heading,
            speed=speed,
            target_lane_index=target_lane_index,
        )
    except TypeError:
        v = vehicle_cls(
            road,
            pos,
            heading=heading,
            speed=speed,
        )
    v.lane_index = lane_index
    v.lane = lane
    v.on_state_update()
    return v


def setup_env_with_state(
    env,
    ego_state: Sequence[float],
    neighbors_state: Sequence[Sequence[float]],
):
    base_env = env.unwrapped
    road = base_env.road
    road.vehicles = []
    base_env.controlled_vehicles = []

    ego_cls = base_env.action_type.vehicle_class
    ego = make_vehicle(road, ego_state, ego_cls)

    neighbors: list[Vehicle] = []
    for s in neighbors_state:
        neighbors.append(make_vehicle(road, s, Vehicle))

    base_env.controlled_vehicles = [ego]
    base_env.vehicle = ego
    road.vehicles = [ego] + neighbors

    for v in road.vehicles:
        if hasattr(v, "history"):
            v.history.clear()
            v.history.appendleft(Vehicle.create_from(v))

    return base_env, ego, neighbors


def build_high_action_space(env, high_interval: int) -> gym.spaces.Box:
    cfg = env.unwrapped.config
    v_min = 0.0
    v_max = float(cfg.get("speed_limit", 30.0))
    dt = 1.0 / float(cfg.get("policy_frequency", 10.0))
    t_h = float(high_interval) * dt
    goal_low = np.array([v_min * t_h, -1.0, v_min], dtype=np.float32)
    goal_high = np.array([v_max * t_h, 1.0, v_max], dtype=np.float32)
    return gym.spaces.Box(goal_low, goal_high, dtype=np.float32)


def default_metric_fn(runner: Any, obs_next: np.ndarray) -> float:
    return float(runner.intrinsic_if_last(obs_next))


def abs_dx_metric_fn(runner: Any, obs_next: np.ndarray) -> float:
    if runner.goal_phys is None or runner.goal_phys.size < 1:
        return 0.0
    _, kin, _ = runner._split(obs_next)
    ego_sub = runner._ego_sub(kin)
    ego_x = float(ego_sub[0])
    dx = float(runner.goal_phys[0] - ego_x)
    return abs(dx)


def abs_dy_metric_fn(runner: Any, obs_next: np.ndarray) -> float:
    if runner.goal_phys is None or runner.goal_phys.size < 2:
        return 0.0
    _, kin, _ = runner._split(obs_next)
    ego_sub = runner._ego_sub(kin)
    ego_y = float(ego_sub[1]) if ego_sub.size > 1 else 0.0
    dy = float(runner.goal_phys[1] - ego_y)
    return abs(dy)


def _parse_literal(raw: str):
    text = str(raw).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return ast.literal_eval(text)


def _to_vec4(value, field_name: str) -> list[float]:
    if value is None:
        raise ValueError(f"{field_name} 为空。")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} 需要是长度为 4 的列表。")
    if len(value) < 4:
        raise ValueError(f"{field_name} 需要至少 4 个元素 [x, y, vx, vy]。")
    return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]


def _parse_neighbors(value) -> list[list[float]]:
    if value is None or value == "":
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("neighbors_state 需要是二维列表，如 [[x,y,vx,vy], ...]。")
    out: list[list[float]] = []
    for i, item in enumerate(value):
        out.append(_to_vec4(item, f"neighbors_state[{i}]"))
    return out


def _pick_float(row: dict[str, str], key: str, default=None):
    value = row.get(key, None)
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    return float(text)


def load_test_cases_from_csv(csv_path: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV 缺少表头。")

        for row_idx, row in enumerate(reader, start=2):
            if row is None:
                continue

            row_text = "".join(str(v).strip() for v in row.values() if v is not None)
            if row_text == "":
                continue

            try:
                ego_raw = row.get("ego_state", "")
                goal_raw = row.get("goal_phys", "")
                neighbors_raw = row.get("neighbors_state", "")

                if str(ego_raw).strip() != "":
                    ego_state = _to_vec4(_parse_literal(str(ego_raw)), "ego_state")
                else:
                    ego_state = [
                        float(_pick_float(row, "ego_x")),
                        float(_pick_float(row, "ego_y")),
                        float(_pick_float(row, "ego_vx")),
                        float(_pick_float(row, "ego_vy")),
                    ]

                if str(goal_raw).strip() != "":
                    goal_phys = _to_vec4(_parse_literal(str(goal_raw)), "goal_phys")
                else:
                    goal_phys = [
                        float(_pick_float(row, "goal_x")),
                        float(_pick_float(row, "goal_y")),
                        float(_pick_float(row, "goal_vx")),
                        float(_pick_float(row, "goal_vy")),
                    ]

                if str(neighbors_raw).strip() != "":
                    neighbors_state = _parse_neighbors(_parse_literal(str(neighbors_raw)))
                else:
                    neighbors_state = []

                case_id_raw = row.get("case_id", "")
                case_id = str(case_id_raw).strip() if case_id_raw is not None else ""

                cases.append(
                    {
                        "case_id": case_id,
                        "ego_state": ego_state,
                        "neighbors_state": neighbors_state,
                        "goal_phys": goal_phys,
                    }
                )
            except Exception as e:
                raise ValueError(f"解析 CSV 第 {row_idx} 行失败: {e}") from e

    if not cases:
        raise ValueError("CSV 中没有可用测试用例。")
    return cases
