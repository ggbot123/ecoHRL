import importlib
import os
import time
from typing import Any, Dict

import gymnasium as gym
import numpy as np

from configs.conf import get_env_config
from custom_env.vehicle.behavior import IDMVehicle, NormalIDMVehicle


# =========================
# In-code runtime selection
# =========================
# Choose which scenario package to import (for gym register side-effect)
SCENARIO_MODULE = "scenarios.multi_lane_stop_to_int"
# SCENARIO_MODULE = "scenarios.multi_lane"  # Example: original multi-lane

ENV_ID = "multi-lane-stop-to-int-v0"
# ENV_ID = "multi-lane-custom-v0"  # Example: original multi-lane

# Episode runtime config (no CLI args)
SEED = 42
N_EPISODES = 20
STEP_SLEEP_S = 0.0

# Scenario flow control: probability of spawning a background vehicle per generation check.
TRAFFIC_FLOW_SPAWN_PROB = 0.05

# Render robustness toggles
RENDER_MODE = "human"
# RENDER_MODE = "rgb_array"
FORCE_DISABLE_DUMMY_SDL = True
EXPLICIT_RENDER_EACH_STEP = True


# Common overrides used by compatible scenarios
COMMON_OVERRIDES: Dict[str, Any] = {
    "duration": 70.0,
    # Keep a single warmup for the first reset, then continue across episodes.
    "warmup_each_episode": False,
    "show_trajectories": True,
    "warmup_render": False,
    "offscreen_rendering": False,
    "screen_width": 1800,
    "screen_height": 320,
    "scaling": 3,
    "centering_position": [0.5, 0.5],
}


# Scenario-specific overrides. Unknown keys are ignored by unrelated scenarios.
SCENARIO_OVERRIDES_BY_ENV_ID: Dict[str, Dict[str, Any]] = {
    "multi-lane-stop-to-int-v0": {
        "lanes_count": 3,
        "spawn_probability": TRAFFIC_FLOW_SPAWN_PROB,
        "start_lane_id": 2,
        "start_longitudinal": 0.0,
        "target_lane_id": 0,
        "goal_longitudinal": 400.0,
        "intersection_length": 50.0,
        # Movement-role lane mapping (lane ids). Any unlisted lanes default to straight.
        "movement_lanes": {
            # "left": [0],
            "straight": [0, 1, 2],
        },
        # Optional movement-specific behavior distribution for background vehicles.
        # "movement_behavior_probs": {
        #     "left": [0.2, 0.3, 0.5],
        #     "straight": [0.5, 0.3, 0.2],
        # },
        # Keep background vehicles in their movement-role lanes.
        "background_vehicle_respect_movement_lanes": True,
        # Signal plan format: [{direction: green+yellow total seconds}, ...]
        "signal_plan": [
            {"straight": 63.0},  # 18s green + 3s yellow
            {"left": 37.0},      # 12s green + 3s yellow
        ],
        "signal_cycle_offset": 0.0,
        "align_ego_spawn_to_signal_offset": True,
        "episode_start_phase_offset": 0.0,
    },
    "multi-lane-custom-v0": {
        "initial_lane_id": "random",
        "goal_lane_id": 2,
        "goal_longitudinal": 400.0,
    },
}


def register_scenario(module_path: str) -> None:
    importlib.import_module(module_path)


def ensure_render_backend(render_mode: str) -> None:
    if render_mode != "human" or not FORCE_DISABLE_DUMMY_SDL:
        return
    # In some environments SDL_VIDEODRIVER=dummy disables the pygame window completely.
    if os.environ.get("SDL_VIDEODRIVER", "").lower() == "dummy":
        os.environ.pop("SDL_VIDEODRIVER", None)
        print("[render] Removed SDL_VIDEODRIVER=dummy for human rendering.")


def build_env_config(env_id: str, user_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    overrides: Dict[str, Any] = dict(COMMON_OVERRIDES)
    overrides.update(SCENARIO_OVERRIDES_BY_ENV_ID.get(env_id, {}))
    if user_overrides:
        overrides.update(user_overrides)
    return get_env_config(overrides)


def switch_ego_to_idm(env: gym.Env) -> None:
    """Replace ego with an IDM-type vehicle so no RL action is required."""
    base_env = env.unwrapped
    ego_old = base_env.vehicle
    if ego_old is None:
        raise RuntimeError("Ego vehicle is not initialized after reset().")

    if isinstance(ego_old, IDMVehicle):
        return

    ego_new = NormalIDMVehicle.create_from(ego_old)
    ego_new.vid = getattr(ego_old, "vid", -1)

    try:
        idx = base_env.road.vehicles.index(ego_old)
    except ValueError as exc:
        raise RuntimeError("Old ego vehicle was not found in road.vehicles.") from exc

    base_env.road.vehicles[idx] = ego_new
    base_env.vehicle = ego_new
    base_env.controlled_vehicles = [ego_new]

    # Keep action/observation internals pointing to the new ego.
    if getattr(base_env, "action_type", None) is not None:
        base_env.action_type.controlled_vehicle = ego_new
    if getattr(base_env, "observation_type", None) is not None:
        base_env.observation_type.observer_vehicle = ego_new


def set_fixed_global_camera(env: gym.Env) -> None:
    """Keep viewer camera fixed at a global road center instead of following ego."""
    base_env = env.unwrapped
    if getattr(base_env, "road", None) is None:
        return

    road_len = float(getattr(base_env, "_road_end_x", base_env.config.get("road_length", 500.0)))
    lanes = int(base_env.config.get("lanes_count", 1))
    try:
        lane0 = base_env.road.network.get_lane(("0", "1", 0))
        lane_width = float(getattr(lane0, "width", 4.0))
    except Exception:
        lane_width = 4.0

    center_x = 0.5 * road_len
    center_y = 0.5 * max(lanes - 1, 0) * lane_width

    anchor = getattr(base_env, "_fixed_global_camera_anchor", None)
    if anchor is None:
        class _CameraAnchor:
            pass

        anchor = _CameraAnchor()
        base_env._fixed_global_camera_anchor = anchor

    anchor.position = np.array([center_x, center_y], dtype=float)

    if getattr(base_env, "viewer", None) is not None:
        base_env.viewer.observer_vehicle = anchor


def _get_straight_lane_ids(base_env: gym.Env) -> set[int]:
    controller = getattr(base_env, "_signal_controller", None)
    if controller is not None:
        groups = getattr(controller, "direction_lane_groups", None)
        if isinstance(groups, dict) and "straight" in groups:
            return {int(lid) for lid in groups["straight"]}
    lanes = int(base_env.config.get("lanes_count", 0))
    return set(range(lanes))


def compute_straight_queue_length(env: gym.Env, speed_threshold: float = 0.8) -> float:
    """Estimate straight-direction queue length [m] before the stop line."""
    base_env = env.unwrapped
    if getattr(base_env, "road", None) is None:
        return 0.0

    goal_fn = getattr(base_env, "_goal_longitudinal", None)
    if callable(goal_fn):
        stop_x = float(goal_fn())
    else:
        stop_x = float(base_env.config.get("goal_longitudinal", 0.0))

    straight_lane_ids = _get_straight_lane_ids(base_env)
    if not straight_lane_ids:
        return 0.0

    max_queue_len = 0.0
    for lane_id in straight_lane_ids:
        rear_x = None
        for v in base_env.road.vehicles:
            if getattr(v, "crashed", False):
                continue

            lane_index = getattr(v, "lane_index", None)
            if lane_index is None or len(lane_index) < 3:
                continue
            if lane_index[0] != "0" or lane_index[1] != "1" or int(lane_index[2]) != int(lane_id):
                continue

            x = float(v.position[0])
            if x < 0.0 or x > stop_x:
                continue

            speed = float(getattr(v, "speed", np.linalg.norm(getattr(v, "velocity", np.zeros(2)))))
            if speed > speed_threshold:
                continue

            if rear_x is None or x < rear_x:
                rear_x = x

        if rear_x is not None:
            lane_queue_len = max(stop_x - rear_x, 0.0)
            if lane_queue_len > max_queue_len:
                max_queue_len = lane_queue_len

    return float(max_queue_len)


def plot_queue_length_timeseries(times_s: list[float], queue_m: list[float], output_path: str) -> None:
    if not times_s or not queue_m:
        print("[queue] No samples collected; skip plotting.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[queue] matplotlib not installed; skip plotting.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times_s, queue_m, color="#1f77b4", linewidth=1.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Straight Queue Length [m]")
    ax.set_title("Straight-Direction Queue Length Over Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_n_episodes(
    scenario_module: str,
    env_id: str,
    n_episodes: int,
    seed: int,
    sleep_s: float,
    env_overrides: Dict[str, Any] | None = None,
) -> None:
    register_scenario(scenario_module)
    ensure_render_backend(RENDER_MODE)
    cfg = build_env_config(env_id, env_overrides)
    # Ensure only the first reset triggers warmup in this run.
    cfg["warmup_each_episode"] = False
    sim_dt = 1.0 / float(cfg["policy_frequency"])

    env = gym.make(
        env_id,
        render_mode=RENDER_MODE,
        config=cfg,
    )

    try:
        global_time_s = 0.0
        queue_times_s: list[float] = []
        queue_lengths_m: list[float] = []

        for ep_idx in range(int(n_episodes)):
            # Seed only the first reset to preserve continuity for later episodes.
            reset_seed = seed if ep_idx == 0 else None
            obs, info = env.reset(seed=reset_seed)
            switch_ego_to_idm(env)

            # Sample queue length at episode start.
            q0 = compute_straight_queue_length(env)
            queue_times_s.append(global_time_s)
            queue_lengths_m.append(q0)
            ep_queue_samples = [q0]

            if RENDER_MODE == "human":
                # Create viewer first, then lock camera to a fixed global anchor.
                env.render()
                set_fixed_global_camera(env)
                if EXPLICIT_RENDER_EACH_STEP:
                    env.render()

            terminated = False
            truncated = False
            step_count = 0
            total_reward = 0.0

            while not (terminated or truncated):
                # action=None means env.action_type.act(...) is skipped;
                # ego is driven by road.act() calling IDMVehicle.act().
                obs, reward, terminated, truncated, info = env.step(None)

                global_time_s += sim_dt
                q_t = compute_straight_queue_length(env)
                queue_times_s.append(global_time_s)
                queue_lengths_m.append(q_t)
                ep_queue_samples.append(q_t)

                if EXPLICIT_RENDER_EACH_STEP and RENDER_MODE == "human":
                    set_fixed_global_camera(env)
                    env.render()

                total_reward += float(reward)
                step_count += 1

                if sleep_s > 0.0:
                    time.sleep(sleep_s)

            base_env = env.unwrapped
            final_lane = None
            if getattr(base_env, "vehicle", None) is not None:
                lane_index = getattr(base_env.vehicle, "lane_index", None)
                if lane_index is not None and len(lane_index) >= 3:
                    final_lane = int(lane_index[2])

            arrived = bool(getattr(base_env, "_has_arrived", False))
            arrival_time = getattr(base_env, "_arrival_time", None)

            print("=" * 80)
            print(f"Episode {ep_idx + 1}/{n_episodes} finished")
            print(f"steps       : {step_count}")
            print(f"reward_sum  : {total_reward:.6f}")
            print(f"terminated  : {terminated}")
            print(f"truncated   : {truncated}")
            print(f"crashed     : {bool(info.get('crashed', False))}")
            print(f"arrived     : {arrived}")
            print(f"arrival_time: {arrival_time}")
            print(f"final_lane  : {final_lane}")
            print(f"queue_max_m : {max(ep_queue_samples):.3f}")
            print(f"queue_mean_m: {float(np.mean(ep_queue_samples)):.3f}")
            print("=" * 80)

        plot_path = os.path.join(os.getcwd(), "straight_queue_length_over_time.png")
        plot_queue_length_timeseries(queue_times_s, queue_lengths_m, plot_path)
        if queue_lengths_m:
            print(f"[queue] samples={len(queue_lengths_m)}")
            print(f"[queue] max={max(queue_lengths_m):.3f} m, mean={float(np.mean(queue_lengths_m)):.3f} m")
        print(f"[queue] plot saved: {plot_path}")

    finally:
        env.close()


def main() -> None:
    run_n_episodes(
        scenario_module=SCENARIO_MODULE,
        env_id=ENV_ID,
        n_episodes=N_EPISODES,
        seed=SEED,
        sleep_s=STEP_SLEEP_S,
        env_overrides=None,
    )


if __name__ == "__main__":
    main()
