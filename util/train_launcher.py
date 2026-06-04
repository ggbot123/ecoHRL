from __future__ import annotations

import importlib
import os
import random
from datetime import datetime
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from configs.conf import (
    MASTER_SEED,
    get_env_config_for_scenario,
    get_hiro_config,
    get_hiro_high_sac_kwargs,
    get_hiro_low_sac_kwargs,
    get_ppo_kwargs,
    get_sac_kwargs,
    get_scenario_spec,
)
from rl.algos.HRL.hiro import HIROConfig
from rl.algos.HRL.trainer import train_hiro
from rl.algos.ppo.trainer import train_ppo
from rl.algos.sac.trainer import train_sac
from util.train_video import (
    env_index_set,
    global_view_video_config,
    validate_video_env_choice,
    wrap_training_video_env,
)

master_rng: np.random.Generator = np.random.default_rng(MASTER_SEED)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def make_env(
    env_id: str,
    scenario_name: str,
    env_overrides: Mapping[str, Any] | None = None,
    render_mode: str | None = None,
    record_video: bool = False,
    record_video_global_view: bool = True,
    video_folder: str | None = None,
    video_name_prefix: str = "train",
    video_episode_freq: int = 1,
    record_video_scheduled: bool = True,
    record_video_collision_episodes: bool = False,
):
    """Return an env constructor compatible with DummyVecEnv/SubprocVecEnv."""
    env_seed = int(master_rng.integers(0, 2**31 - 1))
    scenario_module = str(get_scenario_spec(scenario_name)["module"])

    def _init():
        # Windows spawn mode needs each subprocess to register the scenario env ID.
        importlib.import_module(scenario_module)
        cfg = get_env_config_for_scenario(scenario_name, env_overrides or {})
        if record_video and record_video_global_view:
            cfg.update(global_view_video_config())
        cfg["_env_seed"] = env_seed

        env = gym.make(env_id, render_mode=render_mode, config=cfg)
        raw_env = env.unwrapped
        raw_env.np_random = np.random.default_rng(env_seed)
        if hasattr(raw_env, "_np_random_seed"):
            raw_env._np_random_seed = env_seed

        if record_video:
            env = wrap_training_video_env(
                env,
                video_folder=video_folder,
                video_name_prefix=video_name_prefix,
                video_episode_freq=video_episode_freq,
                record_video_global_view=bool(record_video_global_view),
                record_video_scheduled=bool(record_video_scheduled),
                record_video_collision_episodes=bool(record_video_collision_episodes),
            )

        env = Monitor(env)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(env_seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(env_seed)
        return env

    return _init


def apply_low_safety_filter_overrides(
    env_overrides: Mapping[str, Any],
    hiro_cfg: HIROConfig,
) -> dict[str, Any]:
    if hiro_cfg.low_safety_filter is None:
        return dict(env_overrides)

    updated = dict(env_overrides)
    updated.update(
        {
            "lane_change_min_front_gap": float(hiro_cfg.low_safety_filter.lane_change_min_front_gap),
            "lane_change_min_rear_gap": float(hiro_cfg.low_safety_filter.lane_change_min_rear_gap),
            "lane_change_min_front_ttc": float(hiro_cfg.low_safety_filter.lane_change_min_front_ttc),
            "lane_change_min_rear_ttc": float(hiro_cfg.low_safety_filter.lane_change_min_rear_ttc),
        }
    )
    return updated


def apply_sac_env_overrides(
    scenario_name: str,
    env_overrides: Mapping[str, Any],
    sac_env_overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    updated = dict(env_overrides)
    updated.update(dict(sac_env_overrides or {}))

    cfg_for_sac = get_env_config_for_scenario(scenario_name, updated)
    if not bool(cfg_for_sac.get("enable_sac_low_safety_filter", False)):
        return updated

    hiro_cfg = get_hiro_config()
    if hiro_cfg.low_safety_filter is None:
        print("[SAC] enable_sac_low_safety_filter=True, but HIRO low_safety_filter is None; skipped")
        return updated

    updated["enable_low_safety_filter"] = True
    updated = apply_low_safety_filter_overrides(updated, hiro_cfg)
    print("[SAC] Enabled low safety filter from HIRO low_safety_filter")
    return updated


def build_hiro_video_env_fns(
    *,
    env_id: str,
    scenario_name: str,
    env_overrides: Mapping[str, Any],
    n_envs: int,
    render_mode: str | None,
    log_dir: str,
    record_video: bool,
    record_video_envs: str,
    record_video_global_view: bool,
    video_episode_freq: int,
    record_video_collision_episodes: bool,
    record_video_collision_envs: str,
):
    video_dir = os.path.join(log_dir, "videos")
    video_envs = env_index_set(record_video_envs, n_envs)
    collision_video_envs = (
        env_index_set(record_video_collision_envs, n_envs)
        if bool(record_video_collision_episodes)
        else set()
    )
    wrapped_video_envs = video_envs | collision_video_envs
    hiro_render_mode = "rgb_array" if (bool(record_video) and bool(wrapped_video_envs)) else render_mode

    env_fns = [
        make_env(
            env_id,
            scenario_name,
            env_overrides,
            render_mode=hiro_render_mode,
            record_video=bool(record_video) and i in wrapped_video_envs,
            record_video_global_view=bool(record_video_global_view),
            video_folder=os.path.join(video_dir, f"env{i}") if (bool(record_video) and i in wrapped_video_envs) else None,
            video_name_prefix=f"train_env{i}",
            video_episode_freq=video_episode_freq,
            record_video_scheduled=i in video_envs,
            record_video_collision_episodes=i in collision_video_envs,
        )
        for i in range(n_envs)
    ]
    return env_fns, video_dir


def run_training(
    *,
    algo: str,
    total_timesteps: int,
    eval_freq: int,
    save_freq: int,
    n_envs: int,
    render: bool = False,
    log_root: str = "./logs",
    save_root: str = "./models",
    run_name: str | None = None,
    scenario_name: str = "multi_lane",
    env_overrides: Mapping[str, Any] | None = None,
    sac_env_overrides: Mapping[str, Any] | None = None,
    hiro_high_pretrained_path: str | None = None,
    hiro_low_pretrained_path: str | None = None,
    hiro_low_target_entropy: str | float = "auto",
    hiro_low_target_entropy_scale: float | None = 0.5,
    hiro_low_sac_impl: str | None = None,
    hiro_high_transition_csv_all: int = 1,
    hiro_high_transition_csv_envs: str = "env0",
    hiro_low_transition_detail_csv: bool = False,
    hiro_low_transition_detail_envs: str = "env0",
    record_video: bool = False,
    record_video_envs: str = "env0",
    record_video_global_view: bool = True,
    video_episode_freq: int = 1,
    record_video_collision_episodes: bool = False,
    record_video_collision_envs: str = "all",
) -> None:
    global master_rng
    set_global_seed(MASTER_SEED)
    master_rng = np.random.default_rng(MASTER_SEED)

    algo = algo.lower()
    env_overrides = dict(env_overrides or {})
    record_video_envs = validate_video_env_choice(record_video_envs, "record_video_envs")
    record_video_collision_envs = validate_video_env_choice(record_video_collision_envs, "record_video_collision_envs")

    scenario_spec = get_scenario_spec(scenario_name)
    scenario_module = str(scenario_spec["module"])
    env_id = str(scenario_spec["env_id"])
    importlib.import_module(scenario_module)

    if run_name is None:
        run_name = f"{algo}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    log_dir = os.path.join(log_root, run_name)
    save_dir = os.path.join(save_root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[MAIN] algo={algo}")
    print(f"[MAIN] scenario={scenario_name}, env_id={env_id}")
    print(f"[MAIN] log_dir={log_dir}")
    print(f"[MAIN] save_dir={save_dir}")

    render_mode = "human" if bool(render) else None
    if record_video:
        print(f"[MAIN] record_video=True, envs={record_video_envs}, every {max(1, int(video_episode_freq))} episode(s)")
        if bool(record_video_collision_episodes):
            print(f"[MAIN] record collision episodes from {record_video_collision_envs}")
    if bool(render) and int(n_envs) != 1:
        print(f"[MAIN] render=True is usually best with n_envs=1, current n_envs={n_envs}")

    if algo == "ppo":
        env_fns = [make_env(env_id, scenario_name, env_overrides, render_mode=render_mode) for _ in range(n_envs)]
        eval_env_fn = make_env(env_id, scenario_name, env_overrides, render_mode=render_mode)
        train_ppo(
            env_fns=env_fns,
            eval_env_fn=eval_env_fn,
            total_timesteps=total_timesteps,
            log_dir=log_dir,
            save_dir=save_dir,
            ppo_kwargs=get_ppo_kwargs(log_dir=log_dir, seed=MASTER_SEED),
            eval_freq=eval_freq,
            save_freq=save_freq,
            save_name_prefix="ppo",
        )

    elif algo == "sac":
        sac_overrides = apply_sac_env_overrides(scenario_name, env_overrides, sac_env_overrides)
        env_fns = [make_env(env_id, scenario_name, sac_overrides, render_mode=render_mode) for _ in range(n_envs)]
        eval_env_fn = make_env(env_id, scenario_name, sac_overrides, render_mode=render_mode)
        train_sac(
            env_fns=env_fns,
            eval_env_fn=eval_env_fn,
            total_timesteps=total_timesteps,
            log_dir=log_dir,
            save_dir=save_dir,
            sac_kwargs=get_sac_kwargs(log_dir=log_dir, seed=MASTER_SEED),
            eval_freq=eval_freq,
            save_freq=save_freq,
            save_name_prefix="sac",
        )

    elif algo == "hiro":
        run_hiro_training(
            env_id=env_id,
            scenario_name=scenario_name,
            scenario_module=scenario_module,
            env_overrides=env_overrides,
            n_envs=n_envs,
            render=render,
            render_mode=render_mode,
            log_root=log_root,
            save_root=save_root,
            log_dir=log_dir,
            save_dir=save_dir,
            run_name=run_name,
            total_timesteps=total_timesteps,
            hiro_high_pretrained_path=hiro_high_pretrained_path,
            hiro_low_pretrained_path=hiro_low_pretrained_path,
            hiro_low_target_entropy=hiro_low_target_entropy,
            hiro_low_target_entropy_scale=hiro_low_target_entropy_scale,
            hiro_low_sac_impl=hiro_low_sac_impl,
            hiro_high_transition_csv_all=hiro_high_transition_csv_all,
            hiro_high_transition_csv_envs=hiro_high_transition_csv_envs,
            hiro_low_transition_detail_csv=hiro_low_transition_detail_csv,
            hiro_low_transition_detail_envs=hiro_low_transition_detail_envs,
            record_video=record_video,
            record_video_envs=record_video_envs,
            record_video_global_view=record_video_global_view,
            video_episode_freq=video_episode_freq,
            record_video_collision_episodes=record_video_collision_episodes,
            record_video_collision_envs=record_video_collision_envs,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    print("[MAIN] Training finished")


def run_hiro_training(
    *,
    env_id: str,
    scenario_name: str,
    scenario_module: str,
    env_overrides: Mapping[str, Any],
    n_envs: int,
    render: bool,
    render_mode: str | None,
    log_root: str,
    save_root: str,
    log_dir: str,
    save_dir: str,
    run_name: str,
    total_timesteps: int,
    hiro_high_pretrained_path: str | None,
    hiro_low_pretrained_path: str | None,
    hiro_low_target_entropy: str | float,
    hiro_low_target_entropy_scale: float | None,
    hiro_low_sac_impl: str | None,
    hiro_high_transition_csv_all: int,
    hiro_high_transition_csv_envs: str,
    hiro_low_transition_detail_csv: bool,
    hiro_low_transition_detail_envs: str,
    record_video: bool,
    record_video_envs: str,
    record_video_global_view: bool,
    video_episode_freq: int,
    record_video_collision_episodes: bool,
    record_video_collision_envs: str,
) -> None:
    hiro_cfg = get_hiro_config()
    if hiro_high_pretrained_path:
        hiro_cfg.high_pretrained_path = hiro_high_pretrained_path
    if hiro_low_pretrained_path:
        hiro_cfg.low_pretrained_path = hiro_low_pretrained_path
    if hiro_low_sac_impl is not None:
        hiro_cfg.low_sac_impl = str(hiro_low_sac_impl)

    env_overrides = apply_low_safety_filter_overrides(env_overrides, hiro_cfg)

    print(f"[HIRO] Train Mode: {hiro_cfg.train_mode}, Goal Sampler: {hiro_cfg.goal_sampler.type}")
    print(f"[HIRO] Low SAC Impl: {hiro_cfg.low_sac_impl}")
    print(f"[HIRO] High pretrained: {hiro_cfg.high_pretrained_path}")
    print(f"[HIRO] Low pretrained: {hiro_cfg.low_pretrained_path}")

    env_fns, video_dir = build_hiro_video_env_fns(
        env_id=env_id,
        scenario_name=scenario_name,
        env_overrides=env_overrides,
        n_envs=n_envs,
        render_mode=render_mode,
        log_dir=log_dir,
        record_video=record_video,
        record_video_envs=record_video_envs,
        record_video_global_view=record_video_global_view,
        video_episode_freq=video_episode_freq,
        record_video_collision_episodes=record_video_collision_episodes,
        record_video_collision_envs=record_video_collision_envs,
    )
    env = DummyVecEnv(env_fns) if bool(render) else SubprocVecEnv(env_fns)

    try:
        train_hiro(
            env=env,
            total_timesteps=total_timesteps,
            log_dir=log_dir,
            save_dir=save_dir,
            high_sac_kwargs=get_hiro_high_sac_kwargs(log_dir=os.path.join(log_dir, "hiro_high"), seed=MASTER_SEED),
            low_sac_kwargs=get_hiro_low_sac_kwargs(
                log_dir=os.path.join(log_dir, "hiro_low"),
                seed=MASTER_SEED,
                target_entropy=hiro_low_target_entropy,
                target_entropy_scale=hiro_low_target_entropy_scale,
            ),
            cfg=hiro_cfg,
            save_name_prefix="hiro",
            seed=MASTER_SEED,
            high_transition_csv_all=max(0, int(hiro_high_transition_csv_all)),
            high_transition_csv_envs=hiro_high_transition_csv_envs,
            low_transition_detail_csv=bool(hiro_low_transition_detail_csv),
            low_transition_detail_envs=hiro_low_transition_detail_envs,
            run_metadata={
                "algo": "hiro",
                "run_name": run_name,
                "scenario_name": scenario_name,
                "scenario_module": scenario_module,
                "env_id": env_id,
                "log_root": log_root,
                "save_root": save_root,
                "n_envs": int(n_envs),
                "render": bool(render),
                "master_seed": int(MASTER_SEED),
                "train_time_env_overrides": dict(env_overrides),
                "hiro_high_pretrained_path": hiro_high_pretrained_path,
                "hiro_low_pretrained_path": hiro_low_pretrained_path,
                "hiro_low_target_entropy": hiro_low_target_entropy,
                "hiro_low_target_entropy_scale": hiro_low_target_entropy_scale,
                "hiro_low_sac_impl_arg": hiro_low_sac_impl,
                "hiro_high_transition_csv_envs": hiro_high_transition_csv_envs,
                "hiro_low_transition_detail_csv": bool(hiro_low_transition_detail_csv),
                "hiro_low_transition_detail_envs": hiro_low_transition_detail_envs,
                "record_video": bool(record_video),
                "record_video_envs": record_video_envs,
                "record_video_global_view": bool(record_video_global_view),
                "video_episode_freq": int(video_episode_freq),
                "record_video_collision_episodes": bool(record_video_collision_episodes),
                "record_video_collision_envs": record_video_collision_envs,
                "video_dir": video_dir if bool(record_video) else None,
            },
        )
    finally:
        print("[MAIN] Closing training env...")
        env.close()
        print("[MAIN] Training env closed")
