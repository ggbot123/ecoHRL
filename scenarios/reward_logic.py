def wrong_lane_terminal_triggered(
    *,
    longitudinal_reached: bool,
    goal_reached: bool,
    episode_ending: bool,
    only_at_goal_longitudinal: bool,
) -> bool:
    if longitudinal_reached:
        return not goal_reached
    if only_at_goal_longitudinal:
        return False
    return episode_ending


def goal_lane_dense_progress(
    *,
    previous_lane_id: int,
    current_lane_id: int,
    goal_lane_id: int,
) -> float:
    """Potential progress toward the episode goal lane in lane-index units."""
    previous_distance = abs(int(previous_lane_id) - int(goal_lane_id))
    current_distance = abs(int(current_lane_id) - int(goal_lane_id))
    return float(previous_distance - current_distance)
