from gymnasium.envs.registration import register

from .scenario import MultiLaneStopToIntEnv

register(
    id="multi-lane-stop-to-int-v0",
    entry_point="scenarios.multi_lane_stop_to_int.scenario:MultiLaneStopToIntEnv",
)
