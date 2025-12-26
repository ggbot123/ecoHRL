import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import scenarios.multi_lane  # 触发 __init__.py 里的 register

import numpy as np
import os
from typing import Any, Dict, Optional, Sequence, Tuple

from util.plot_result import *

from rl.algos.sac.sac import SAC
from rl.utils import utils
from configs.conf import get_env_config, get_hiro_config

def _resolve_hiro_model_paths(model_dir: str, model_name: str) -> Tuple[str, str]:
    name = str(model_name)
    if name.endswith(".zip"):
        if "_high" in name and "_low" not in name:
            high_name = name
            low_name = name.replace("_high_final", "_low_final").replace("_high_", "_low_").replace("_high", "_low", 1)
        elif "_low" in name:
            low_name = name
            high_name = name.replace("_low_final", "_high_final").replace("_low_", "_high_").replace("_low", "_high", 1)
        else:
            prefix = name[:-4]
            high_name, low_name = f"{prefix}_high_final.zip", f"{prefix}_low_final.zip"
    else:
        high_name, low_name = f"{name}_high_final.zip", f"{name}_low_final.zip"
    return os.path.join(model_dir, high_name), os.path.join(model_dir, low_name)


def _load_hiro_models(model_dir: str, model_name: str) -> Tuple[SAC, SAC]:
    high_path, low_path = _resolve_hiro_model_paths(model_dir, model_name)
    return SAC.load(high_path), SAC.load(low_path)


class HIROPolicyRunner:
    """Single-env HIRO inference runner.

    - High-level: sample goal every `high_interval` env steps.
    - Low-level: sample primitive action every env step.
    - Maintains per-interval state needed for intrinsic reward logging.
    """

    def __init__(self, high_model: SAC, low_model: SAC, high_interval: int):
        self.high_model, self.low_model, self.hi = high_model, low_model, int(high_interval)
        self._inited = False
        self.need_high, self.c = True, 0
        self.n_veh, self.feat_dim, self.feature_names, self.ego_feature_idx, self.ego_dim = 0, 0, [], [], 0
        self.lane_center_ys = np.zeros(0, dtype=np.float32)
        self.goal_phys = np.zeros(0, dtype=np.float32)
        self.ego_start = np.zeros(0, dtype=np.float32)

        # dx_max is set from env config in init_from_env (speed_limit * high_interval * dt)
        self.norm_ranges = np.asarray([[0.0, 1.0], [-8.0, 8.0], [-8.0, 8.0], [-2.0, 2.0]], dtype=np.float32)
        self.weights = np.asarray([1.0, 2.0, 8.0, 1.0], dtype=np.float32)
        self.intrinsic_coef = 1.0

    def init_from_env(self, env, obs0: np.ndarray, intrinsic_coef: float):
        keep = ("x", "y", "vx", "vy")
        n_veh, feat_dim, feature_names, ego_idx = utils.init_kinematics_meta(env, obs0, keep)
        self.n_veh, self.feat_dim, self.feature_names, self.ego_feature_idx = int(n_veh), int(feat_dim), list(feature_names), list(ego_idx)
        self.ego_dim = int(len(self.ego_feature_idx))
        cfg = getattr(env.unwrapped, "config", getattr(env, "config", {}))
        lanes, lane_w = int(cfg["lanes_count"]), float(cfg.get("lane_width", 4.0))
        self.lane_center_ys = (np.arange(lanes, dtype=np.float32) * lane_w).astype(np.float32)

        v_max = float(cfg.get("speed_limit", 15.0))
        dt = 1.0 / float(cfg.get("policy_frequency", 10.0))
        self.norm_ranges[0, 1] = max(v_max * float(self.hi) * dt, 1e-6)
        self.goal_phys = np.zeros(self.ego_dim, dtype=np.float32)
        self.ego_start = np.zeros(self.ego_dim, dtype=np.float32)
        self.intrinsic_coef = float(intrinsic_coef)
        self._inited = True

    def reset(self, env, obs0: np.ndarray, intrinsic_coef: float):
        if not self._inited:
            self.init_from_env(env, obs0, intrinsic_coef)
        self.need_high, self.c = True, 0

    def _split(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(obs, dtype=np.float32)
        t, kin, kin_flat = utils.split_time_kinematics(arr[None, :], self.n_veh, self.feat_dim)
        return t, kin, kin_flat

    def _ego_sub(self, kin: np.ndarray) -> np.ndarray:
        return utils.extract_ego_substate(kin, self.ego_feature_idx)[0]

    def _sample_goal(self, obs: np.ndarray, kin: np.ndarray, env=None):
        ego_sub = self._ego_sub(kin)
        goal_action, _ = self.high_model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        goal_action = np.asarray(goal_action, dtype=np.float32).reshape(1, -1)
        goal_phys = utils.goal_action_to_abs(ego_sub[None, :], goal_action, self.lane_center_ys)
        self.goal_phys = np.asarray(goal_phys, dtype=np.float32).reshape(-1)
        self.ego_start = ego_sub.astype(np.float32, copy=True)
        self.need_high, self.c = False, 0
        
        # Update env with goal for rendering
        if env is not None:
            # Handle RecordVideo wrapper or other wrappers
            unwrapped = env.unwrapped
            if hasattr(unwrapped, "set_hiro_goal"):
                unwrapped.set_hiro_goal(self.goal_phys)

    def act(self, env, obs: np.ndarray) -> np.ndarray:
        _, kin, kin_flat = self._split(obs)
        if self.need_high:
            self._sample_goal(obs, kin, env)
        ego_sub = self._ego_sub(kin)
        t_norm = np.array([self.c / float(self.hi)], dtype=np.float32)
        goal_rel = (self.goal_phys - ego_sub).astype(np.float32)
        low_obs = np.concatenate([t_norm, kin_flat[0], goal_rel]).astype(np.float32)
        action, _ = self.low_model.predict(low_obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    def intrinsic_if_last(self, obs_next: np.ndarray) -> float:
        _, kin_next, _ = self._split(obs_next)
        ego_next = self._ego_sub(kin_next)
        ego_rel = (ego_next - self.ego_start).astype(np.float32)
        goal_rel = (self.goal_phys - self.ego_start).astype(np.float32)
        r = utils.intrinsic_reward_l2(ego_rel[None, :], goal_rel[None, :], self.norm_ranges, self.intrinsic_coef, self.weights)
        return float(np.asarray(r, dtype=np.float32).reshape(-1)[0])

    def step_end(self, done: bool):
        self.c += 1
        if done or self.c >= self.hi:
            self.need_high, self.c = True, 0


def main(
    model_dir: str,
    model_name: str,
    episodes: int,
    record_episodes: Optional[Sequence[int]] = None,
    env_overrides: Optional[Dict[str, Any]] = None,
):
    log_path = os.path.join(model_dir, "eval_hiro.txt")
    log_file = open(log_path, "w", encoding="utf-8")
    def log(msg: str = ""):
        print(msg)
        log_file.write(msg + "\n")

    test_overrides: Dict[str, Any] = {
        "duration": 70.0,
        "initial_lane_id": 1,
        "warmup_each_episode": False,
        "screen_width": 1800,
        "screen_height": 300,
        "scaling": 3,
        "centering_position": [0.5, 0.5],
        "show_trajectories": True,
        "warmup_render": False,
    }
    if env_overrides:
        test_overrides.update(env_overrides)
    env_config = get_env_config(test_overrides)

    if not record_episodes:
        def trigger(ep_id: int) -> bool: return False
    else:
        record_set = {int(ep_idx) - 1 for ep_idx in record_episodes}
        def trigger(ep_id: int) -> bool: return ep_id in record_set

    base_env = gym.make("multi-lane-custom-v0", render_mode="rgb_array", config=env_config)
    env = RecordVideo(base_env, video_folder=model_dir, episode_trigger=trigger, name_prefix="hiro")

    high_model, low_model = _load_hiro_models(model_dir, model_name)
    hiro_cfg = get_hiro_config()
    runner = HIROPolicyRunner(high_model, low_model, int(getattr(hiro_cfg, "high_interval", 25)))

    reward_keys_high = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "punctual_reward", "on_road_reward"]
    reward_keys_low = ["collision_reward", "progress_reward", "comfort_reward", "lane_change_reward", "on_road_reward", "intrinsic_reward"]

    log("=" * 80)
    log(f"Eval HIRO model dir: {model_dir}")
    log(f"Episodes           : {episodes}")
    hp, lp = _resolve_hiro_model_paths(model_dir, model_name)
    log(f"HIRO high          : {os.path.basename(hp)}")
    log(f"HIRO low           : {os.path.basename(lp)}")
    log(f"High interval      : {runner.hi}")
    log("=" * 80)

    ep_lens: list[int] = []
    high_ep_rets: list[float] = []
    low_ep_ext_rets: list[float] = []
    low_ep_int_rets: list[float] = []
    low_ep_total_rets: list[float] = []
    high_comp_sum = {k: 0.0 for k in reward_keys_high}
    low_comp_sum = {k: 0.0 for k in reward_keys_low}

    arrived_count, arrival_times = 0, []
    viewer_initialized = False
    seed_base = 42

    for ep in range(1, int(episodes) + 1):
        obs, _ = env.reset(seed=seed_base + ep)
        runner.reset(env, obs, float(getattr(hiro_cfg, "intrinsic_coef", 1.0)))

        terminated, truncated, steps = False, False, 0
        high_ret, low_ext_ret, low_int_ret, low_total_ret = 0.0, 0.0, 0.0, 0.0
        high_comp = {k: 0.0 for k in reward_keys_high}
        low_comp = {k: 0.0 for k in reward_keys_low}
        high_interval_rets, low_interval_rets = [], []
        cur_high_interval_ret, cur_low_interval_ret = 0.0, 0.0

        if not viewer_initialized:
            class Dummy:
                def __init__(self, pos): self.position = np.array(pos, dtype=float)
            base = env.unwrapped
            base.render()
            base.viewer.observer_vehicle = Dummy([base.config["road_length"] / 2, 5.0])
            viewer_initialized = True

        while not (terminated or truncated):
            action = runner.act(env, obs)
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            rc = info.get("reward_components", {})
            punctual = float(rc.get("punctual_reward", 0.0))
            low_ext = float(reward) - punctual

            last_step = bool(done or runner.c == runner.hi - 1)
            intrinsic = runner.intrinsic_if_last(obs_next) if last_step else 0.0

            high_ret += float(reward)
            low_ext_ret += low_ext
            low_int_ret += intrinsic
            low_total_ret += low_ext + intrinsic
            cur_high_interval_ret += float(reward)
            cur_low_interval_ret += low_ext + intrinsic

            for k in reward_keys_high:
                high_comp[k] += float(rc.get(k, 0.0))
            for k in reward_keys_low:
                if k == "intrinsic_reward":
                    low_comp[k] += float(intrinsic)
                elif k == "punctual_reward":
                    continue
                else:
                    low_comp[k] += float(rc.get(k, 0.0))

            steps += 1

            if last_step:
                high_interval_rets.append(float(cur_high_interval_ret))
                low_interval_rets.append(float(cur_low_interval_ret))
                cur_high_interval_ret, cur_low_interval_ret = 0.0, 0.0
            runner.step_end(done)
            obs = obs_next

        n_low_intervals = len(low_interval_rets) or 1
        low_ext_mean = low_ext_ret / float(n_low_intervals)
        low_int_mean = low_int_ret / float(n_low_intervals)
        low_total_mean = low_total_ret / float(n_low_intervals)

        ep_lens.append(int(steps))
        high_ep_rets.append(float(high_ret))
        low_ep_ext_rets.append(float(low_ext_mean))
        low_ep_int_rets.append(float(low_int_mean))
        low_ep_total_rets.append(float(low_total_mean))
        for k in reward_keys_high:
            high_comp_sum[k] += high_comp[k]
        for k in reward_keys_low:
            low_comp_sum[k] += low_comp[k] / float(n_low_intervals)

        base_env = env.unwrapped
        arrived = bool(getattr(base_env, "_has_arrived", False))
        arrival_time = getattr(base_env, "_arrival_time", None)
        if arrived:
            arrived_count += 1
            if arrival_time is not None:
                arrival_times.append(float(arrival_time))

        reason = "terminated" if terminated else ("truncated(time limit)" if truncated else "unknown")
        log("=" * 60)
        log(f"Episode {ep}:")
        log(f"  length (steps)          : {steps}")
        log(f"  terminated info         : {reason}")
        log(f"  high total reward       : {high_ret:.6f}")
        log(f"  low  ext reward (per interval mean)       : {low_ext_mean:.6f}   (env_reward - punctual)")
        log(f"  low  intrinsic reward (per interval mean) : {low_int_mean:.6f}")
        log(f"  low  total reward (per interval mean)     : {low_total_mean:.6f}   (ext + intrinsic)")
        if high_interval_rets:
            log(f"  high intervals          : {len(high_interval_rets)}  (mean={float(np.mean(high_interval_rets)):.6f})")
        if low_interval_rets:
            log(f"  low  intervals          : {len(low_interval_rets)}  (mean={float(np.mean(low_interval_rets)):.6f})")

        log("  high reward components (sum over episode):")
        for k in reward_keys_high:
            log(f"    {k:18s}: {high_comp[k]: .6f}")

        log("  low reward components (mean per interval):")
        for k in reward_keys_low:
            log(f"    {k:18s}: {low_comp[k] / float(n_low_intervals): .6f}")

        if arrived and arrival_time is not None:
            log(f"  ARRIVED at t = {float(arrival_time):.3f} s")
        if base_env.config.get("show_trajectories", False):
            save_speed_acc_curves(env, ep_idx=ep, model_path=model_dir)

    n = int(episodes)
    log("=" * 80)
    log("Summary over all episodes:")
    log(f"  episodes                : {n}")
    log(f"  mean length             : {float(np.mean(ep_lens)):.3f} steps")
    log(f"  mean high total reward  : {float(np.mean(high_ep_rets)):.6f}")
    log(f"  mean low  ext (per interval mean)    : {float(np.mean(low_ep_ext_rets)):.6f}")
    log(f"  mean low  intrinsic (per interval)   : {float(np.mean(low_ep_int_rets)):.6f}")
    log(f"  mean low  total (per interval mean)  : {float(np.mean(low_ep_total_rets)):.6f}")

    log("  mean high reward components (per episode):")
    for k in reward_keys_high:
        log(f"    {k:18s}: {high_comp_sum[k] / n: .6f}")
    log("  mean low reward components (per interval):")
    for k in reward_keys_low:
        log(f"    {k:18s}: {low_comp_sum[k] / n: .6f}")

    arrive_rate = arrived_count / n if n else 0.0
    log(f"  arrival rate            : {arrive_rate * 100:.2f}%")
    if arrived_count:
        log(f"  mean arrival time       : {float(np.mean(arrival_times)):.3f} s (over {arrived_count} success episodes)")
    else:
        log("  mean arrival time       : N/A (no successful episodes)")
    log("=" * 80)

    log_file.close()
    env.close()


if __name__ == "__main__":
    # 用法示例：
    # - model_dir: 训练产物目录（包含 *_high_final.zip / *_low_final.zip）
    # - model_name: 前缀（如 "hiro"）或其中一个 zip 文件名
    main(
        model_dir="./models/hiro_20251222-164124", 
        model_name="hiro", 
        episodes=10, 
        record_episodes=[1, 2, 3]
    )
