from gymnasium.envs.registration import register

from .scenario import MultiLaneEnv

register(
    id="multi-lane-custom-v0",
    entry_point="scenarios.multi_lane.scenario:MultiLaneEnv",
)
