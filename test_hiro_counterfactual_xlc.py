"""Evaluate fixed lane-change positions against a trained HIRO high policy.

Run this file directly. Experiment settings are constants below; no command-line
arguments are required.
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.wrappers import RecordVideo
from tqdm.auto import tqdm

from configs.conf import get_scenario_spec
from rl.algos.HRL.hiro_infer import HIROPolicyRunner
from util.hiro_utils import (
    env_config_from_run_config,
    hiro_config_from_run_config,
    load_hiro_high_model,
    load_hiro_low_model,
    load_hiro_run_config,
    unique_path,
)


# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

MODEL_DIR = "./models/hiro_260604_highonly_ruleFollow_augObs_SigFeat_newFlowProbs"
MODEL_SUFFIX = "final"

EPISODES = 50
SEED_BASE = 42
X_LC_VALUES = [0.0, 100.0, 200.0, 225.0, 250.0, 275.0, 300.0, 325.0, 350.0, 375.0]

TARGET_LANE_ID = 1
INCLUDE_TRAINED_BASELINE = True
HIGH_RETURN_GAMMA = 0.99
# "saved" uses the controller configuration stored with the model.
# "idm_mobil" switches the rule-based low level to pure IDM+MOBIL.
RULE_CONTROLLER_MODES = ("saved", "idm_mobil")

OUTPUT_ROOT = "./debug/counterfactual_xlc"
RECORD_VIDEOS = True
# None records every episode. Use e.g. [1, 2, 3] to limit disk usage.
VIDEO_EPISODES: Optional[Sequence[int]] = None


REWARD_COMPONENT_KEYS = (
    "collision_reward",
    "progress_reward",
    "speed_ref_aux_reward",
    "comfort_reward_for_high",
    "comfort_reward",
    "lane_change_reward",
    "punctual_reward",
    "wrong_lane_terminal_penalty",
    "on_road_reward",
)


@dataclass(frozen=True)
class ExperimentCondition:
    x_lc: Optional[float]
    controller_mode: str
    trained_baseline: bool = False


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def _std(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def _vehicle_x(base_env: Any) -> float:
    vehicle = getattr(base_env, "vehicle", None)
    if vehicle is None or not hasattr(vehicle, "position"):
        raise RuntimeError("Cannot read ego longitudinal position from environment")
    return float(vehicle.position[0])


def _vehicle_lane_id(base_env: Any) -> Optional[int]:
    vehicle = getattr(base_env, "vehicle", None)
    if vehicle is None:
        return None
    lane_index = getattr(vehicle, "lane_index", None)
    if lane_index is not None and len(lane_index) >= 3:
        try:
            return int(lane_index[2])
        except (TypeError, ValueError):
            pass
    if hasattr(vehicle, "position"):
        lane_width = float(base_env.config.get("lane_width", 4.0))
        lanes_count = int(base_env.config.get("lanes_count", 3))
        return int(
            np.clip(
                round(float(vehicle.position[1]) / max(lane_width, 1e-6)),
                0,
                lanes_count - 1,
            )
        )
    return None


class FixedLaneChangePositionPolicy:
    """Keep the model's longitudinal goal and override only its lane decision."""

    def __init__(
        self,
        env: gym.Env,
        high_model: Any,
        action_space: gym.spaces.Box,
        *,
        x_lc: float,
        target_lane_id: int,
    ):
        self.env = env
        self.high_model = high_model
        self.action_space = action_space
        self.x_lc = float(x_lc)
        self.target_lane_id = int(target_lane_id)
        self.first_attempt_x: Optional[float] = None
        self.success_x: Optional[float] = None
        self.attempt_count = 0
        self.lane_change_succeeded = False
        self.last_model_action = np.zeros(0, dtype=np.float32)
        self.last_overridden_action = np.zeros(0, dtype=np.float32)

    def _mark_success_if_reached(self, x_now: float, lane_now: Optional[int]) -> None:
        if lane_now == self.target_lane_id and not self.lane_change_succeeded:
            self.lane_change_succeeded = True
            self.success_x = float(x_now)

    def __call__(self, high_obs: np.ndarray) -> np.ndarray:
        base_env = self.env.unwrapped
        x_now = _vehicle_x(base_env)
        lane_now = _vehicle_lane_id(base_env)
        self._mark_success_if_reached(x_now, lane_now)

        model_action, _ = self.high_model.predict(
            np.asarray(high_obs, dtype=np.float32),
            deterministic=True,
        )
        action = np.asarray(model_action, dtype=np.float32).reshape(-1).copy()
        if action.size < 3:
            raise ValueError(
                f"Expected a 3D HIRO high action, got shape {action.shape}"
            )
        self.last_model_action = action.copy()

        lane_code = 0.0
        should_attempt = x_now >= self.x_lc and not self.lane_change_succeeded
        if should_attempt and lane_now is not None:
            if lane_now > self.target_lane_id:
                lane_code = -1.0
            elif lane_now < self.target_lane_id:
                lane_code = 1.0
            else:
                self._mark_success_if_reached(x_now, lane_now)

        if lane_code != 0.0:
            self.attempt_count += 1
            if self.first_attempt_x is None:
                self.first_attempt_x = float(x_now)

        action[1] = float(lane_code)
        action = np.clip(
            action,
            self.action_space.low,
            self.action_space.high,
        ).astype(np.float32)
        self.last_overridden_action = action.copy()
        return action

    def finalize(self) -> None:
        x_now = _vehicle_x(self.env.unwrapped)
        lane_now = _vehicle_lane_id(self.env.unwrapped)
        self._mark_success_if_reached(x_now, lane_now)


def _write_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _condition_name(condition: ExperimentCondition) -> str:
    if condition.trained_baseline:
        return "trained_high_policy"
    if condition.x_lc is None:
        raise ValueError("Rule intervention condition requires x_lc")
    base = f"x_lc_{float(condition.x_lc):g}"
    return (
        base
        if condition.controller_mode == "saved"
        else f"{base}_{condition.controller_mode}"
    )


def _commands_toward_target_lane(
    lane_code: float,
    current_lane: Optional[int],
    target_lane: int,
) -> bool:
    if current_lane is None or int(current_lane) == int(target_lane):
        return False
    if int(current_lane) > int(target_lane):
        return float(lane_code) < -1.0 / 3.0
    return float(lane_code) > 1.0 / 3.0


def _build_runner(
    env: gym.Env,
    obs: np.ndarray,
    high_model: Any,
    low_model: Any,
    hiro_cfg: Any,
    x_lc: Optional[float],
) -> tuple[HIROPolicyRunner, Optional[FixedLaneChangePositionPolicy]]:
    runner = HIROPolicyRunner(
        high_model,
        low_model,
        int(getattr(hiro_cfg, "high_interval", 25)),
        use_low_safety_layer=None,
        config=hiro_cfg,
    )
    runner.reset(env, obs, float(getattr(hiro_cfg, "intrinsic_coef", 1.0)))

    policy = None
    if x_lc is not None:
        policy = FixedLaneChangePositionPolicy(
            env,
            high_model,
            high_model.action_space,
            x_lc=float(x_lc),
            target_lane_id=TARGET_LANE_ID,
        )
        runner.high_policy = policy
    return runner, policy


def _run_episode(
    *,
    env_id: str,
    env_config: Dict[str, Any],
    high_model: Any,
    low_model: Any,
    hiro_cfg: Any,
    seed: int,
    episode: int,
    condition_spec: ExperimentCondition,
    video_root: Optional[str] = None,
    step_callback: Optional[
        Callable[[int, float, float, Optional[int]], None]
    ] = None,
) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

    condition = _condition_name(condition_spec)
    x_lc = condition_spec.x_lc
    record_video = bool(
        video_root is not None
        and (
            VIDEO_EPISODES is None
            or int(episode) in {int(value) for value in VIDEO_EPISODES}
        )
    )
    episode_env_config = deepcopy(env_config)
    if condition_spec.controller_mode == "idm_mobil":
        episode_env_config["rule_based_compute_action_mode"] = "idm_mobil"
        episode_env_config["rule_follow_mode_enabled"] = False
    elif condition_spec.controller_mode != "saved":
        raise ValueError(
            f"Unknown rule controller mode: {condition_spec.controller_mode}"
        )

    env = gym.make(
        env_id,
        render_mode="rgb_array" if record_video else None,
        config=episode_env_config,
    )
    if record_video:
        condition_video_dir = os.path.join(str(video_root), condition)
        env = RecordVideo(
            env,
            video_folder=condition_video_dir,
            episode_trigger=lambda _episode_id: True,
            name_prefix=f"{condition}_ep_{int(episode):04d}",
            disable_logger=True,
        )
    try:
        obs, _ = env.reset(seed=seed)
        runner, rule_policy = _build_runner(
            env, obs, high_model, low_model, hiro_cfg, x_lc
        )

        total_reward = 0.0
        post_attempt_reward = 0.0
        first_command_x: Optional[float] = None
        first_command_interval: Optional[int] = None
        interval_index = 0
        interval_reward = 0.0
        interval_start_x = _vehicle_x(env.unwrapped)
        interval_start_lane = _vehicle_lane_id(env.unwrapped)
        interval_goal_action: list[float] = []
        interval_model_goal_action: list[float] = []
        interval_goal_phys: list[float] = []
        interval_components = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
        episode_components = {key: 0.0 for key in REWARD_COMPONENT_KEYS}
        high_interval_rewards: list[float] = []
        interval_rows: list[Dict[str, Any]] = []
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated):
            is_high_start = bool(runner.need_high)
            if is_high_start:
                interval_start_x = _vehicle_x(env.unwrapped)
                interval_start_lane = _vehicle_lane_id(env.unwrapped)

            action = runner.act(env, obs)

            if is_high_start:
                interval_goal_action = (
                    np.asarray(runner.last_goal_action, dtype=np.float32)
                    .reshape(-1)
                    .tolist()
                )
                if rule_policy is None:
                    interval_model_goal_action = list(interval_goal_action)
                else:
                    interval_model_goal_action = (
                        np.asarray(
                            rule_policy.last_model_action,
                            dtype=np.float32,
                        )
                        .reshape(-1)
                        .tolist()
                    )
                interval_goal_phys = (
                    np.asarray(runner.goal_phys, dtype=np.float32)
                    .reshape(-1)
                    .tolist()
                )
                lane_code = (
                    float(interval_goal_action[1])
                    if len(interval_goal_action) >= 2
                    else 0.0
                )
                if (
                    first_command_x is None
                    and _commands_toward_target_lane(
                        lane_code,
                        interval_start_lane,
                        TARGET_LANE_ID,
                    )
                ):
                    first_command_x = float(interval_start_x)
                    first_command_interval = int(interval_index)

            obs_next, reward, terminated, truncated, info = env.step(action)
            reward_value = float(reward)
            total_reward += reward_value
            interval_reward += reward_value
            if first_command_interval is not None:
                post_attempt_reward += reward_value

            reward_components = info.get("reward_components", {})
            for key in REWARD_COMPONENT_KEYS:
                value = float(reward_components.get(key, 0.0))
                interval_components[key] += value
                episode_components[key] += value

            done = bool(terminated or truncated)
            last_interval_step = bool(done or runner.c == runner.hi - 1)
            steps += 1
            runner.step_end(
                done,
                queue_takeover_active=bool(info.get("queue_takeover_active", False)),
            )
            obs = obs_next
            if step_callback is not None:
                step_callback(
                    steps,
                    total_reward,
                    _vehicle_x(env.unwrapped),
                    _vehicle_lane_id(env.unwrapped),
                )

            if last_interval_step:
                high_interval_rewards.append(float(interval_reward))
                interval_row: Dict[str, Any] = {
                    "condition": condition,
                    "controller_mode": condition_spec.controller_mode,
                    "x_lc": "" if x_lc is None else float(x_lc),
                    "episode": int(episode),
                    "seed": int(seed),
                    "interval": int(interval_index),
                    "start_x": float(interval_start_x),
                    "start_lane": (
                        "" if interval_start_lane is None else int(interval_start_lane)
                    ),
                    "model_goal_action": json.dumps(
                        interval_model_goal_action
                    ),
                    "goal_action": json.dumps(interval_goal_action),
                    "goal_phys": json.dumps(interval_goal_phys),
                    "interval_reward": float(interval_reward),
                }
                for key in REWARD_COMPONENT_KEYS:
                    interval_row[key] = float(interval_components[key])
                interval_rows.append(interval_row)

                interval_index += 1
                interval_reward = 0.0
                interval_components = {
                    key: 0.0 for key in REWARD_COMPONENT_KEYS
                }

        if rule_policy is not None:
            rule_policy.finalize()

        base_env = env.unwrapped
        final_lane = _vehicle_lane_id(base_env)
        crashed = bool(getattr(getattr(base_env, "vehicle", None), "crashed", False))
        arrived = bool(getattr(base_env, "_has_arrived", False))
        arrival_time_raw = getattr(base_env, "_arrival_time", None)
        arrival_time = (
            None if arrival_time_raw is None else float(arrival_time_raw)
        )
        punctual_window = base_env.config.get("punctual_time_window", [0.0, 0.0])
        on_time = bool(
            arrived
            and arrival_time is not None
            and float(punctual_window[0])
            <= arrival_time
            <= float(punctual_window[1])
        )
        discounted_high_return = float(
            sum(
                (HIGH_RETURN_GAMMA**idx) * value
                for idx, value in enumerate(high_interval_rewards)
            )
        )

        row: Dict[str, Any] = {
            "condition": condition,
            "controller_mode": condition_spec.controller_mode,
            "x_lc": "" if x_lc is None else float(x_lc),
            "episode": int(episode),
            "seed": int(seed),
            "steps": int(steps),
            "high_intervals": int(len(high_interval_rewards)),
            "episode_return": float(total_reward),
            "discounted_high_return": discounted_high_return,
            "return_from_first_lane_command": (
                ""
                if first_command_interval is None
                else float(post_attempt_reward)
            ),
            "first_lane_command_x": (
                "" if first_command_x is None else float(first_command_x)
            ),
            "lane_change_success_x": (
                ""
                if rule_policy is None or rule_policy.success_x is None
                else float(rule_policy.success_x)
            ),
            "lane_change_attempts": (
                ""
                if rule_policy is None
                else int(rule_policy.attempt_count)
            ),
            "arrived": int(arrived),
            "on_time": int(on_time),
            "crashed": int(crashed),
            "final_lane": "" if final_lane is None else int(final_lane),
            "target_lane_success": int(final_lane == TARGET_LANE_ID),
            "arrival_time": "" if arrival_time is None else arrival_time,
        }
        for key in REWARD_COMPONENT_KEYS:
            row[key] = float(episode_components[key])
        return row, interval_rows
    finally:
        env.close()


def _build_summary(
    episode_rows: Sequence[Dict[str, Any]],
    conditions: Sequence[ExperimentCondition],
) -> list[Dict[str, Any]]:
    baseline_by_seed = {
        int(row["seed"]): float(row["episode_return"])
        for row in episode_rows
        if row["condition"] == "trained_high_policy"
    }
    summaries: list[Dict[str, Any]] = []

    for condition_spec in conditions:
        condition = _condition_name(condition_spec)
        x_lc = condition_spec.x_lc
        rows = [row for row in episode_rows if row["condition"] == condition]
        returns = [float(row["episode_return"]) for row in rows]
        discounted = [float(row["discounted_high_return"]) for row in rows]
        paired_delta = [
            float(row["episode_return"]) - baseline_by_seed[int(row["seed"])]
            for row in rows
            if int(row["seed"]) in baseline_by_seed
        ]
        post_command = [
            float(row["return_from_first_lane_command"])
            for row in rows
            if row["return_from_first_lane_command"] != ""
        ]
        first_x = [
            float(row["first_lane_command_x"])
            for row in rows
            if row["first_lane_command_x"] != ""
        ]
        success_x = [
            float(row["lane_change_success_x"])
            for row in rows
            if row["lane_change_success_x"] != ""
        ]
        n = len(rows)
        stderr = _std(paired_delta) / np.sqrt(len(paired_delta)) if paired_delta else float("nan")
        summaries.append(
            {
                "condition": condition,
                "controller_mode": condition_spec.controller_mode,
                "x_lc": "" if x_lc is None else float(x_lc),
                "episodes": n,
                "mean_episode_return": _mean(returns),
                "std_episode_return": _std(returns),
                "mean_discounted_high_return": _mean(discounted),
                "mean_return_from_first_lane_command": _mean(post_command),
                "mean_first_lane_command_x": _mean(first_x),
                "mean_lane_change_success_x": _mean(success_x),
                "arrival_rate": _mean(row["arrived"] for row in rows),
                "on_time_rate": _mean(row["on_time"] for row in rows),
                "collision_rate": _mean(row["crashed"] for row in rows),
                "target_lane_success_rate": _mean(
                    row["target_lane_success"] for row in rows
                ),
                "mean_paired_return_delta_vs_trained": _mean(paired_delta),
                "paired_delta_std": _std(paired_delta),
                "paired_delta_95ci_low": (
                    _mean(paired_delta) - 1.96 * stderr
                    if paired_delta
                    else float("nan")
                ),
                "paired_delta_95ci_high": (
                    _mean(paired_delta) + 1.96 * stderr
                    if paired_delta
                    else float("nan")
                ),
            }
        )
    return summaries


def run_experiment(
    *,
    episodes: int = EPISODES,
    x_lc_values: Sequence[float] = X_LC_VALUES,
    include_trained_baseline: bool = INCLUDE_TRAINED_BASELINE,
    controller_modes: Sequence[str] = RULE_CONTROLLER_MODES,
) -> str:
    run_config, run_config_path = load_hiro_run_config(MODEL_DIR)
    metadata = run_config.get("run_metadata", {})
    scenario_name = str(metadata.get("scenario_name", "multi_lane"))
    scenario_spec = get_scenario_spec(scenario_name)
    importlib.import_module(str(scenario_spec["module"]))
    env_id = str(scenario_spec["env_id"])

    env_config = env_config_from_run_config(run_config)
    env_config.pop("_env_seed", None)
    env_config.pop("actual_episode_start_phase_offset", None)
    env_config["show_trajectories"] = False
    env_config["warmup_render"] = False
    env_config["offscreen_rendering"] = bool(RECORD_VIDEOS)

    hiro_cfg = hiro_config_from_run_config(run_config)
    low_level_type = str(getattr(hiro_cfg, "low_level_type", "sac")).lower()
    if low_level_type != "rule_based":
        raise ValueError(
            "This experiment requires the saved HIRO config to use "
            f"rule_based low level, got {low_level_type!r}"
        )

    high_model = load_hiro_high_model(MODEL_DIR, model_suffix=MODEL_SUFFIX)
    low_model = (
        load_hiro_low_model(MODEL_DIR, model_suffix=MODEL_SUFFIX)
        if low_level_type == "sac"
        else None
    )

    output_dir = unique_path(
        os.path.join(OUTPUT_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    os.makedirs(output_dir, exist_ok=True)

    seeds = [int(SEED_BASE) + episode for episode in range(1, int(episodes) + 1)]
    resolved_controller_modes = [
        str(mode).strip().lower() for mode in controller_modes
    ]
    unknown_controller_modes = set(resolved_controller_modes) - {
        "saved",
        "idm_mobil",
    }
    if unknown_controller_modes:
        raise ValueError(
            f"Unknown controller mode(s): {sorted(unknown_controller_modes)}"
        )

    conditions: list[ExperimentCondition] = []
    if include_trained_baseline:
        conditions.append(
            ExperimentCondition(
                x_lc=None,
                controller_mode="saved",
                trained_baseline=True,
            )
        )
    for controller_mode in resolved_controller_modes:
        conditions.extend(
            ExperimentCondition(
                x_lc=float(value),
                controller_mode=controller_mode,
            )
            for value in x_lc_values
        )

    with open(
        os.path.join(output_dir, "experiment_config.json"),
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(
            {
                "model_dir": MODEL_DIR,
                "model_suffix": MODEL_SUFFIX,
                "run_config_path": run_config_path,
                "scenario_name": scenario_name,
                "env_id": env_id,
                "episodes": int(episodes),
                "episode_seeds": seeds,
                "x_lc_values": [float(value) for value in x_lc_values],
                "rule_controller_modes": resolved_controller_modes,
                "target_lane_id": TARGET_LANE_ID,
                "intervention": (
                    "Preserve trained high-policy dx/vx and override only "
                    "the lane-action dimension."
                ),
                "include_trained_baseline": bool(include_trained_baseline),
                "high_return_gamma": HIGH_RETURN_GAMMA,
                "record_videos": bool(RECORD_VIDEOS),
                "video_episodes": (
                    None
                    if VIDEO_EPISODES is None
                    else [int(value) for value in VIDEO_EPISODES]
                ),
            },
            config_file,
            ensure_ascii=False,
            indent=2,
        )

    episode_rows: list[Dict[str, Any]] = []
    interval_rows: list[Dict[str, Any]] = []
    total_runs = len(conditions) * len(seeds)
    expected_episode_steps = max(
        1,
        int(
            np.ceil(
                float(env_config.get("duration", 1.0))
                * float(env_config.get("policy_frequency", 1.0))
            )
        ),
    )
    overall_progress = tqdm(
        total=total_runs,
        desc="Counterfactual x_lc",
        unit="episode",
        position=0,
        dynamic_ncols=True,
    )
    try:
        for condition_spec in conditions:
            condition = _condition_name(condition_spec)
            for episode, seed in enumerate(seeds, start=1):
                episode_progress = tqdm(
                    total=expected_episode_steps,
                    desc=f"{condition} | ep={episode}/{episodes} seed={seed}",
                    unit="step",
                    position=1,
                    leave=False,
                    dynamic_ncols=True,
                )

                def update_step_progress(
                    step: int,
                    total_reward: float,
                    ego_x: float,
                    lane_id: Optional[int],
                ) -> None:
                    episode_progress.update(1)
                    episode_progress.set_postfix(
                        x=f"{ego_x:.1f}",
                        lane="?" if lane_id is None else int(lane_id),
                        ret=f"{total_reward:.2f}",
                        refresh=False,
                    )

                try:
                    episode_row, episode_intervals = _run_episode(
                        env_id=env_id,
                        env_config=env_config,
                        high_model=high_model,
                        low_model=low_model,
                        hiro_cfg=hiro_cfg,
                        seed=seed,
                        episode=episode,
                        condition_spec=condition_spec,
                        video_root=(
                            os.path.join(output_dir, "videos")
                            if RECORD_VIDEOS
                            else None
                        ),
                        step_callback=update_step_progress,
                    )
                finally:
                    episode_progress.close()

                episode_rows.append(episode_row)
                interval_rows.extend(episode_intervals)
                overall_progress.update(1)
                overall_progress.set_postfix(
                    condition=condition,
                    seed=seed,
                    ret=f"{float(episode_row['episode_return']):.2f}",
                    refresh=True,
                )
    finally:
        overall_progress.close()

    summaries = _build_summary(episode_rows, conditions)
    _write_csv(os.path.join(output_dir, "episodes.csv"), episode_rows)
    _write_csv(os.path.join(output_dir, "high_intervals.csv"), interval_rows)
    _write_csv(os.path.join(output_dir, "summary.csv"), summaries)

    print("\nSummary:")
    for row in summaries:
        delta = row["mean_paired_return_delta_vs_trained"]
        delta_text = "N/A" if np.isnan(delta) else f"{delta:+.3f}"
        print(
            f"  {row['condition']:>20s}: "
            f"return={row['mean_episode_return']: .3f}, "
            f"paired_delta={delta_text}, "
            f"on_time={100.0 * row['on_time_rate']:.1f}%, "
            f"target_lane={100.0 * row['target_lane_success_rate']:.1f}%"
        )
    print(f"\nSaved results to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    run_experiment()
