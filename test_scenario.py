import importlib
import os
import time
from typing import Any, Dict

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo

from configs.conf import get_env_config_for_scenario, get_scenario_spec
from custom_env.vehicle.behavior import IDMVehicle, NormalIDMVehicle


# =========================
# In-code runtime selection
# =========================
# Choose which centralized scenario config to use from configs/conf.py.
SCENARIO_NAME = "multi_lane_stop_to_int"
# SCENARIO_NAME = "multi_lane"

# Episode runtime config (no CLI args)
SEED = 42
N_EPISODES = 20
STEP_SLEEP_S = 0.0

# Render robustness toggles
RENDER_MODE = "human"
# RENDER_MODE = "rgb_array"
FORCE_DISABLE_DUMMY_SDL = True
EXPLICIT_RENDER_EACH_STEP = True
EGO_HIGHLIGHT_COLOR = (255, 215, 0)  # Yellow

# Video capture (records exactly one episode by index)
SAVE_EPISODE_VIDEO = False
# SAVE_EPISODE_VIDEO = True
VIDEO_EPISODE_INDEX = 2  # 0-based: 0 means the first episode
VIDEO_OUTPUT_DIR = os.path.join("results", "videos")
VIDEO_NAME_PREFIX = "scenario"


# Only override values that should differ from configs/conf.py for this script.
TEST_ENV_OVERRIDES: Dict[str, Any] = {
    "show_trajectories": True,
    "warmup_render": False,
    "offscreen_rendering": False,
    "screen_width": 1800,
    "screen_height": 320,
    "scaling": 3,
    "centering_position": [0.5, 0.5],
}


def _set_vehicle_color_yellow(vehicle) -> None:
    """Best-effort ego highlight for renderer implementations that honor `color`."""
    if vehicle is None:
        return
    try:
        vehicle.color = EGO_HIGHLIGHT_COLOR
    except Exception:
        pass


def register_scenario(module_path: str) -> None:
    importlib.import_module(module_path)


def ensure_render_backend(render_mode: str) -> None:
    if render_mode != "human" or not FORCE_DISABLE_DUMMY_SDL:
        return
    # In some environments SDL_VIDEODRIVER=dummy disables the pygame window completely.
    if os.environ.get("SDL_VIDEODRIVER", "").lower() == "dummy":
        os.environ.pop("SDL_VIDEODRIVER", None)
        print("[render] Removed SDL_VIDEODRIVER=dummy for human rendering.")


def build_env_config(scenario_name: str, user_overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return get_env_config_for_scenario(scenario_name, user_overrides or {})


def switch_ego_to_idm(env: gym.Env) -> None:
    """Replace ego with an IDM-type vehicle so no RL action is required."""
    base_env = env.unwrapped
    ego_old = base_env.vehicle
    if ego_old is None:
        raise RuntimeError("Ego vehicle is not initialized after reset().")
    if bool(getattr(base_env, "_inter_episode_active", False)):
        raise RuntimeError("Cannot switch ego while inter-episode dummy phase is active.")

    if isinstance(ego_old, IDMVehicle):
        _set_vehicle_color_yellow(ego_old)
        return

    ego_new = NormalIDMVehicle.create_from(ego_old)
    ego_new.vid = getattr(ego_old, "vid", -1)
    _set_vehicle_color_yellow(ego_new)

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


def finish_inter_episode_phase(
    env: gym.Env,
    render_mode: str | None = None,
    sleep_s: float = 0.0,
) -> float:
    """Step through deferred inter-episode simulation until the real ego is created."""
    base_env = env.unwrapped
    if not bool(getattr(base_env, "_inter_episode_active", False)):
        return 0.0

    remaining_start = float(getattr(base_env, "_inter_episode_remaining", 0.0))
    steps = 0
    while bool(getattr(base_env, "_inter_episode_active", False)):
        env.step(None)
        set_fixed_global_camera(env)
        if render_mode == "human" and EXPLICIT_RENDER_EACH_STEP:
            env.render()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        steps += 1

    if steps > 0:
        print(f"[inter_episode] advanced {remaining_start:.3f}s in {steps} dummy steps")
    return remaining_start


def _signal_cycle_length(base_env: gym.Env) -> float | None:
    controller = getattr(base_env, "_signal_controller", None)
    if controller is None:
        return None
    plan = getattr(controller, "signal_plan", None)
    if not isinstance(plan, list) or not plan:
        return None
    return float(sum(total for _, total in plan))


def _signal_phase_tau(base_env: gym.Env) -> float | None:
    controller = getattr(base_env, "_signal_controller", None)
    if controller is None:
        return None
    cycle = _signal_cycle_length(base_env)
    if cycle is None or cycle <= 1e-9:
        return None
    t_global = float(getattr(base_env, "_signal_time_global", 0.0))
    cycle_offset = float(getattr(controller, "cycle_offset", 0.0))
    return float((t_global + cycle_offset) % cycle)


def _phase_err_mod(a: float, b: float, cycle: float) -> float:
    d = abs(a - b) % cycle
    return float(min(d, cycle - d))


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
    scenario_name: str,
    n_episodes: int,
    seed: int,
    sleep_s: float,
    env_overrides: Dict[str, Any] | None = None,
) -> None:
    scenario_spec = get_scenario_spec(scenario_name)
    scenario_module = str(scenario_spec["module"])
    env_id = str(scenario_spec["env_id"])
    register_scenario(scenario_module)
    effective_render_mode = "rgb_array" if SAVE_EPISODE_VIDEO else RENDER_MODE
    ensure_render_backend(effective_render_mode)
    cfg = build_env_config(scenario_name, env_overrides)

    sim_dt = 1.0 / float(cfg["policy_frequency"])

    env = gym.make(
        env_id,
        render_mode=effective_render_mode,
        config=cfg,
    )

    video_dir_abs = os.path.join(os.getcwd(), VIDEO_OUTPUT_DIR)
    if SAVE_EPISODE_VIDEO:
        os.makedirs(video_dir_abs, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=video_dir_abs,
            episode_trigger=lambda ep: int(ep) == int(VIDEO_EPISODE_INDEX),
            name_prefix=VIDEO_NAME_PREFIX,
            disable_logger=True,
        )

    try:
        global_time_s = 0.0
        queue_times_s: list[float] = []
        queue_lengths_m: list[float] = []

        for ep_idx in range(int(n_episodes)):
            # Seed only the first reset to preserve continuity for later episodes.
            reset_seed = seed if ep_idx == 0 else None
            obs, info = env.reset(seed=reset_seed)
            global_time_s += finish_inter_episode_phase(
                env,
                render_mode=effective_render_mode,
                sleep_s=sleep_s,
            )
            switch_ego_to_idm(env)
            base_env = env.unwrapped

            # Optional sanity check: ego should be spawned at the configured phase offset.
            if env_id == "multi-lane-stop-to-int-v0" and bool(cfg.get("align_ego_spawn_to_signal_offset", False)):
                cycle = _signal_cycle_length(base_env)
                tau = _signal_phase_tau(base_env)
                target = float(cfg.get("episode_start_phase_offset", 0.0))
                if cycle is not None and tau is not None:
                    err = _phase_err_mod(tau, target % cycle, cycle)
                    print(
                        f"[phase] ep={ep_idx + 1} tau={tau:.3f}s target={(target % cycle):.3f}s "
                        f"cycle={cycle:.3f}s err={err:.6f}s"
                    )

            # Sample queue length at episode start.
            q0 = compute_straight_queue_length(env)
            queue_times_s.append(global_time_s)
            queue_lengths_m.append(q0)
            ep_queue_samples = [q0]

            # Keep camera fixed for both on-screen rendering and off-screen video recording.
            set_fixed_global_camera(env)
            if SAVE_EPISODE_VIDEO and effective_render_mode == "rgb_array":
                # Ensure viewer exists in off-screen mode so fixed camera applies from the first recorded frame.
                env.render()
                set_fixed_global_camera(env)

            if effective_render_mode == "human":
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

                # RecordVideo uses rgb_array rendering under the hood; re-apply fixed camera each step.
                set_fixed_global_camera(env)

                if EXPLICIT_RENDER_EACH_STEP and effective_render_mode == "human":
                    set_fixed_global_camera(env)
                    env.render()

                total_reward += float(reward)
                step_count += 1

                if sleep_s > 0.0:
                    time.sleep(sleep_s)

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
        if SAVE_EPISODE_VIDEO:
            mp4_files = sorted(
                f for f in os.listdir(video_dir_abs)
                if f.lower().endswith(".mp4") and f.startswith(VIDEO_NAME_PREFIX)
            )
            if mp4_files:
                print("[video] saved files:")
                for filename in mp4_files:
                    print(f"[video] {os.path.join(video_dir_abs, filename)}")
            else:
                print(
                    "[video] no mp4 found. "
                    "Make sure the configured episode index exists and env supports rgb_array rendering."
                )


def main() -> None:
    run_n_episodes(
        scenario_name=SCENARIO_NAME,
        n_episodes=N_EPISODES,
        seed=SEED,
        sleep_s=STEP_SLEEP_S,
        env_overrides=TEST_ENV_OVERRIDES,
    )


if __name__ == "__main__":
    main()
