from __future__ import annotations

import importlib
import os
import random
from copy import deepcopy
from datetime import datetime
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from configs.builders import (
    get_env_config_for_scenario,
    get_hiro_config,
    get_hiro_high_sac_kwargs,
    get_hiro_low_sac_kwargs,
    get_ppo_kwargs,
    get_sac_kwargs,
    get_scenario_spec,
)
from configs.conf import MASTER_SEED
from rl.algos.HRL.hiro import HIROConfig
from rl.algos.HRL.trainer import train_hiro
from rl.algos.ppo.trainer import train_ppo
from rl.algos.sac.trainer import train_sac
from util.config_utils import deep_update
from util.hiro_utils import apply_hiro_config_overrides
from util.train_video import (
    env_index_set,
    global_view_video_config,
    validate_video_env_choice,
    wrap_training_video_env,
)

master_rng: np.random.Generator = np.random.default_rng(MASTER_SEED)

_HIRO_LOW_TRANSITION_DETAIL_ENVS = "env0"
_HIRO_LOW_HER_DEBUG_CSV_MAX_ROWS_PER_FLUSH = 200
_HIRO_LOW_HER_DEBUG_MAX_RECORDS = 20_000
_HIRO_LOW_DEBUG_ENV_STEP_INTERVAL_STEPS = 1_000

_CONFIG_OVERRIDE_SECTIONS = {
    "environment",
    "sac_environment",
    "hiro",
    "ppo_kwargs",
    "sac_kwargs",
    "hiro_high_sac_kwargs",
    "hiro_low_sac_kwargs",
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def _normalize_config_overrides(
    config_overrides: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    if config_overrides is None:
        return {}
    if not isinstance(config_overrides, Mapping):
        raise TypeError("config_overrides must be a mapping")

    unknown = set(config_overrides) - _CONFIG_OVERRIDE_SECTIONS
    if unknown:
        raise ValueError(
            "Unknown config_overrides section(s): "
            f"{sorted(unknown)}. Supported: {sorted(_CONFIG_OVERRIDE_SECTIONS)}"
        )

    normalized: dict[str, dict[str, Any]] = {}
    for section, value in config_overrides.items():
        if not isinstance(value, Mapping):
            raise TypeError(f"config_overrides['{section}'] must be a mapping")
        normalized[section] = deepcopy(dict(value))
    return normalized


def _override_section(
    config_overrides: Mapping[str, dict[str, Any]],
    section: str,
) -> dict[str, Any]:
    return deepcopy(dict(config_overrides.get(section, {}) or {}))


def _apply_kwargs_overrides(
    kwargs: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    merged = deepcopy(dict(kwargs))
    if overrides:
        deep_update(merged, deepcopy(dict(overrides)))
    return merged


def make_env(
    env_id: str,
    scenario_name: str,
    env_overrides: Mapping[str, Any] | None = None,
    render_mode: str | None = None,
    record_video: bool = False,
    video_folder: str | None = None,
    video_name_prefix: str = "train",
    video_episode_freq: int = 1,
    record_video_scheduled: bool = True,
):
    """Return an env constructor compatible with DummyVecEnv/SubprocVecEnv."""
    env_seed = int(master_rng.integers(0, 2**31 - 1))
    scenario_module = str(get_scenario_spec(scenario_name)["module"])

    def _init():
        # Windows spawn mode needs each subprocess to register the scenario env ID.
        importlib.import_module(scenario_module)
        cfg = get_env_config_for_scenario(scenario_name, env_overrides or {})
        if record_video:
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
                record_video_scheduled=bool(record_video_scheduled),
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


def apply_sac_environment_overrides(
    scenario_name: str,
    env_overrides: Mapping[str, Any],
    sac_environment_overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    updated = dict(env_overrides)
    updated.update(dict(sac_environment_overrides or {}))

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
    video_episode_freq: int,
):
    video_dir = os.path.join(log_dir, "videos")
    video_envs = env_index_set(record_video_envs, n_envs)
    hiro_render_mode = "rgb_array" if (bool(record_video) and bool(video_envs)) else render_mode

    env_fns = [
        make_env(
            env_id,
            scenario_name,
            env_overrides,
            render_mode=hiro_render_mode,
            record_video=bool(record_video) and i in video_envs,
            video_folder=os.path.join(video_dir, f"env{i}") if (bool(record_video) and i in video_envs) else None,
            video_name_prefix=f"train_env{i}",
            video_episode_freq=video_episode_freq,
            record_video_scheduled=i in video_envs,
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
    config_overrides: Mapping[str, Any] | None = None,
    sac_transition_csv_episode_freq: int = 1,
    sac_transition_csv_envs: str = "env0",
    hiro_high_pretrained_path: str | None = None,
    hiro_low_pretrained_path: str | None = None,
    hiro_low_target_entropy: str | float = "auto",
    hiro_low_target_entropy_scale: float | None = 0.5,
    hiro_high_transition_csv_all: int = 1,
    hiro_high_transition_csv_envs: str = "env0",
    hiro_high_q_replay_debug: bool = True,
    hiro_low_transition_detail_csv: bool = False,
    hiro_low_her_debug_csv_interval_steps: int = 0,
    hiro_low_her_debug_sample_prob: float = 0.0,
    hiro_low_debug_summary_interval_steps: int = 10000,
    record_video: bool = False,
    record_video_envs: str = "env0",
    video_episode_freq: int = 1,
) -> None:
    global master_rng
    set_global_seed(MASTER_SEED)
    master_rng = np.random.default_rng(MASTER_SEED)

    algo = algo.lower()
    normalized_config_overrides = _normalize_config_overrides(config_overrides)
    env_overrides = _override_section(normalized_config_overrides, "environment")
    sac_environment_overrides = _override_section(normalized_config_overrides, "sac_environment")
    hiro_config_overrides = _override_section(normalized_config_overrides, "hiro")
    ppo_kwargs_overrides = _override_section(normalized_config_overrides, "ppo_kwargs")
    sac_kwargs_overrides = _override_section(normalized_config_overrides, "sac_kwargs")
    hiro_high_sac_kwargs_overrides = _override_section(
        normalized_config_overrides,
        "hiro_high_sac_kwargs",
    )
    hiro_low_sac_kwargs_overrides = _override_section(
        normalized_config_overrides,
        "hiro_low_sac_kwargs",
    )
    record_video_envs = validate_video_env_choice(record_video_envs, "record_video_envs")

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
            ppo_kwargs=_apply_kwargs_overrides(
                get_ppo_kwargs(log_dir=log_dir, seed=MASTER_SEED),
                ppo_kwargs_overrides,
            ),
            eval_freq=eval_freq,
            save_freq=save_freq,
            save_name_prefix="ppo",
        )

    elif algo == "sac":
        sac_overrides = apply_sac_environment_overrides(
            scenario_name,
            env_overrides,
            sac_environment_overrides,
        )
        env_fns = [make_env(env_id, scenario_name, sac_overrides, render_mode=render_mode) for _ in range(n_envs)]
        eval_env_fn = make_env(env_id, scenario_name, sac_overrides, render_mode=render_mode)
        train_sac(
            env_fns=env_fns,
            eval_env_fn=eval_env_fn,
            total_timesteps=total_timesteps,
            log_dir=log_dir,
            save_dir=save_dir,
            sac_kwargs=_apply_kwargs_overrides(
                get_sac_kwargs(log_dir=log_dir, seed=MASTER_SEED, level="default"),
                sac_kwargs_overrides,
            ),
            eval_freq=eval_freq,
            save_freq=save_freq,
            save_name_prefix="sac",
            transition_csv_episode_freq=sac_transition_csv_episode_freq,
            transition_csv_envs=sac_transition_csv_envs,
            run_metadata={
                "algo": "sac",
                "run_name": run_name,
                "scenario_name": scenario_name,
                "scenario_module": scenario_module,
                "env_id": env_id,
                "log_root": log_root,
                "save_root": save_root,
                "n_envs": int(n_envs),
                "render": bool(render),
                "master_seed": int(MASTER_SEED),
                "environment_overrides": dict(env_overrides),
                "sac_environment_overrides": dict(sac_environment_overrides or {}),
                "effective_sac_environment_overrides": dict(sac_overrides),
                "config_overrides": deepcopy(dict(normalized_config_overrides)),
            },
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
            hiro_high_transition_csv_all=hiro_high_transition_csv_all,
            hiro_high_transition_csv_envs=hiro_high_transition_csv_envs,
            hiro_high_q_replay_debug=hiro_high_q_replay_debug,
            hiro_config_overrides=hiro_config_overrides,
            hiro_high_sac_kwargs_overrides=hiro_high_sac_kwargs_overrides,
            hiro_low_sac_kwargs_overrides=hiro_low_sac_kwargs_overrides,
            hiro_low_transition_detail_csv=hiro_low_transition_detail_csv,
            hiro_low_her_debug_csv_interval_steps=hiro_low_her_debug_csv_interval_steps,
            hiro_low_her_debug_sample_prob=hiro_low_her_debug_sample_prob,
            hiro_low_debug_summary_interval_steps=hiro_low_debug_summary_interval_steps,
            record_video=record_video,
            record_video_envs=record_video_envs,
            video_episode_freq=video_episode_freq,
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
    hiro_high_transition_csv_all: int,
    hiro_high_transition_csv_envs: str,
    hiro_high_q_replay_debug: bool,
    hiro_config_overrides: Mapping[str, Any],
    hiro_high_sac_kwargs_overrides: Mapping[str, Any],
    hiro_low_sac_kwargs_overrides: Mapping[str, Any],
    hiro_low_transition_detail_csv: bool,
    hiro_low_her_debug_csv_interval_steps: int,
    hiro_low_her_debug_sample_prob: float,
    hiro_low_debug_summary_interval_steps: int,
    record_video: bool,
    record_video_envs: str,
    video_episode_freq: int,
) -> None:
    hiro_cfg = get_hiro_config()
    if hiro_config_overrides:
        hiro_cfg = apply_hiro_config_overrides(hiro_cfg, hiro_config_overrides)
    if hiro_high_pretrained_path:
        hiro_cfg.high_pretrained_path = hiro_high_pretrained_path
    if hiro_low_pretrained_path:
        hiro_cfg.low_pretrained_path = hiro_low_pretrained_path

    env_overrides = apply_low_safety_filter_overrides(env_overrides, hiro_cfg)

    print(f"[HIRO] Train Mode: {hiro_cfg.train_mode}, Goal Sampler: {hiro_cfg.goal_sampler.type}")
    print("[HIRO] Low SAC Impl: sac")
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
        video_episode_freq=video_episode_freq,
    )
    env = DummyVecEnv(env_fns) if bool(render) else SubprocVecEnv(env_fns)

    try:
        high_sac_kwargs = _apply_kwargs_overrides(
            get_hiro_high_sac_kwargs(
                log_dir=os.path.join(log_dir, "hiro_high"),
                seed=MASTER_SEED,
                q_replay_debug_enabled=bool(hiro_high_q_replay_debug),
            ),
            hiro_high_sac_kwargs_overrides,
        )
        low_sac_kwargs = _apply_kwargs_overrides(
            get_hiro_low_sac_kwargs(
                log_dir=os.path.join(log_dir, "hiro_low"),
                seed=MASTER_SEED,
                target_entropy=hiro_low_target_entropy,
                target_entropy_scale=hiro_low_target_entropy_scale,
            ),
            hiro_low_sac_kwargs_overrides,
        )
        train_hiro(
            env=env,
            total_timesteps=total_timesteps,
            log_dir=log_dir,
            save_dir=save_dir,
            high_sac_kwargs=high_sac_kwargs,
            low_sac_kwargs=low_sac_kwargs,
            cfg=hiro_cfg,
            save_name_prefix="hiro",
            seed=MASTER_SEED,
            high_transition_csv_all=max(0, int(hiro_high_transition_csv_all)),
            high_transition_csv_envs=hiro_high_transition_csv_envs,
            low_transition_detail_csv=bool(hiro_low_transition_detail_csv),
            low_transition_detail_envs=_HIRO_LOW_TRANSITION_DETAIL_ENVS,
            low_her_debug_csv_interval_steps=int(hiro_low_her_debug_csv_interval_steps),
            low_her_debug_csv_max_rows_per_flush=_HIRO_LOW_HER_DEBUG_CSV_MAX_ROWS_PER_FLUSH,
            low_her_debug_max_records=_HIRO_LOW_HER_DEBUG_MAX_RECORDS,
            low_her_debug_sample_prob=float(hiro_low_her_debug_sample_prob),
            low_debug_summary_interval_steps=int(hiro_low_debug_summary_interval_steps),
            low_debug_env_step_interval_steps=_HIRO_LOW_DEBUG_ENV_STEP_INTERVAL_STEPS,
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
                "environment_overrides": dict(env_overrides),
                "hiro_high_pretrained_path": hiro_high_pretrained_path,
                "hiro_low_pretrained_path": hiro_low_pretrained_path,
                "hiro_low_target_entropy": hiro_low_target_entropy,
                "hiro_low_target_entropy_scale": hiro_low_target_entropy_scale,
                "hiro_high_transition_csv_envs": hiro_high_transition_csv_envs,
                "hiro_high_q_replay_debug": bool(hiro_high_q_replay_debug),
                "hiro_low_transition_detail_csv": bool(hiro_low_transition_detail_csv),
                "hiro_low_transition_detail_envs": _HIRO_LOW_TRANSITION_DETAIL_ENVS,
                "hiro_low_her_debug_csv_interval_steps": int(hiro_low_her_debug_csv_interval_steps),
                "hiro_low_her_debug_csv_max_rows_per_flush": _HIRO_LOW_HER_DEBUG_CSV_MAX_ROWS_PER_FLUSH,
                "hiro_low_her_debug_max_records": _HIRO_LOW_HER_DEBUG_MAX_RECORDS,
                "hiro_low_her_debug_sample_prob": float(hiro_low_her_debug_sample_prob),
                "hiro_low_debug_summary_interval_steps": int(hiro_low_debug_summary_interval_steps),
                "hiro_low_debug_env_step_interval_steps": _HIRO_LOW_DEBUG_ENV_STEP_INTERVAL_STEPS,
                "hiro_config_overrides": deepcopy(dict(hiro_config_overrides or {})),
                "hiro_high_sac_kwargs_overrides": deepcopy(dict(hiro_high_sac_kwargs_overrides or {})),
                "hiro_low_sac_kwargs_overrides": deepcopy(dict(hiro_low_sac_kwargs_overrides or {})),
                "record_video": bool(record_video),
                "record_video_envs": record_video_envs,
                "video_episode_freq": int(video_episode_freq),
                "video_dir": video_dir if bool(record_video) else None,
            },
        )
    finally:
        print("[MAIN] Closing training env...")
        env.close()
        print("[MAIN] Training env closed")
