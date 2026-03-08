from __future__ import annotations

from typing import Any, Mapping


def compute_ego_clear_distance_for_front_vehicle(
    env_cfg: Mapping[str, Any],
    ego_speed: float,
    front_speed: float,
) -> float:
    """Compute minimal front distance to avoid immediate safety-layer braking at spawn.

    This aligns with mpc-constraints safety filter in rule_based.py where front constraints are:
    - front distance >= lane_change_min_front_gap
    - front TTC >= lane_change_min_front_ttc

    The filter evaluates one-step-ahead quantities, so we include dt coupling terms.
    """
    dt = 1.0 / max(float(env_cfg.get("policy_frequency", 10.0)), 1e-6)
    lane_front_gap_min = float(env_cfg.get("lane_change_min_front_gap", 10.0))
    lane_front_ttc_min = float(env_cfg.get("lane_change_min_front_ttc", 3.0))
    margin = float(env_cfg.get("ego_clear_margin", 0.0))

    rel_speed_closing = max(float(ego_speed) - float(front_speed), 0.0)

    # Ensure a_upper from longitudinal safety constraints is non-negative (no forced braking).
    # From safety filter derivation:
    # 1) d0 >= lane_front_gap_min + rel_speed * dt
    # 2) d0 >= rel_speed * (lane_front_ttc_min + dt)
    d_req_gap = lane_front_gap_min + rel_speed_closing * dt
    d_req_ttc = rel_speed_closing * (lane_front_ttc_min + dt)

    return float(max(lane_front_gap_min, d_req_gap, d_req_ttc) + margin)
