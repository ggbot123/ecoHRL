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
