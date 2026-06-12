def wrong_lane_terminal_triggered(
    *,
    longitudinal_reached: bool,
    goal_reached: bool,
    episode_ending: bool,
) -> bool:
    if longitudinal_reached:
        return not goal_reached
    return episode_ending
