# rl/algos/hiro/trainer.py
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, is_dataclass, replace
from typing import Any, Dict

import numpy as np

from rl.algos.HRL.hiro import HIROSAC
from rl.algos.HRL.buffer import HiROHighReplayBuffer
from rl.algos.HRL.callbacks import HIROLoggingCallback, HIROCheckpointCallback, HIROLowEpisodeTrajectoryCallback
from rl.algos.HRL.goal_samplers import GoalSamplerConfig, get_goal_sampler
from rl.utils import utils
from util.hiro_low_offline_eval import HIROLowOfflineEvalCallback
from stable_baselines3.common.callbacks import CallbackList


def _json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        return getattr(value, "__name__", repr(value))
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def _write_hiro_run_config(
    *,
    log_dir: str,
    env,
    total_timesteps: int,
    save_dir: str,
    save_name_prefix: str,
    seed: int,
    high_transition_csv_all: int,
    high_transition_csv_envs: str,
    low_transition_detail_csv: bool,
    low_transition_detail_envs: str,
    low_debug_config: Dict[str, Any],
    effective_cfg: Any,
    high_sac_kwargs: Dict[str, Any],
    low_sac_kwargs: Dict[str, Any],
    run_metadata: Dict[str, Any] | None = None,
    checkpoint_save_freq: int = 50_000,
) -> None:
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    try:
        env_configs = env.get_attr("config")
    except Exception as exc:
        env_configs = [{"error": f"env.get_attr('config') failed: {exc!r}"}]

    payload = {
        "run_metadata": dict(run_metadata or {}),
        "trainer": {
            "total_timesteps": int(total_timesteps),
            "save_dir": save_dir,
            "save_name_prefix": save_name_prefix,
            "seed": int(seed),
            "n_envs": int(getattr(env, "num_envs", len(env_configs) if isinstance(env_configs, list) else 1)),
            "high_transition_csv_all": int(high_transition_csv_all),
            "high_transition_csv_envs": str(high_transition_csv_envs),
            "low_transition_detail_csv": bool(low_transition_detail_csv),
            "low_transition_detail_envs": str(low_transition_detail_envs),
            "low_debug_config": dict(low_debug_config),
            "checkpoint_save_freq": int(checkpoint_save_freq),
        },
        "environment": {
            "env0_config": env_configs[0] if isinstance(env_configs, list) and env_configs else env_configs,
            "all_env_configs": env_configs,
        },
        "hiro": {
            "config": effective_cfg,
            "high_sac_kwargs": high_sac_kwargs,
            "low_sac_kwargs": low_sac_kwargs,
        },
    }
    out_path = os.path.join(log_dir, "run_config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[HIRO Trainer] Saved run config: {out_path}")
    model_config_path = os.path.join(save_dir, "run_config.json")
    if os.path.abspath(model_config_path) != os.path.abspath(out_path):
        shutil.copyfile(out_path, model_config_path)
        print(f"[HIRO Trainer] Saved model-side run config: {model_config_path}")


def train_hiro(
    env,
    total_timesteps: int,
    log_dir: str,
    save_dir: str,
    high_sac_kwargs: Dict[str, Any],
    low_sac_kwargs: Dict[str, Any],
    cfg,
    save_name_prefix: str,
    seed: int = 42,
    high_transition_csv_all: int = 1,
    high_transition_csv_envs: str = "env0",
    low_transition_detail_csv: bool = False,
    low_transition_detail_envs: str = "env0",
    low_her_debug_csv_interval_steps: int = 0,
    low_her_debug_csv_max_rows_per_flush: int = 200,
    low_her_debug_max_records: int = 20000,
    low_her_debug_sample_prob: float = 0.0,
    low_debug_summary_interval_steps: int = 10000,
    low_debug_env_step_interval_steps: int = 1000,
    low_offline_eval_config: Dict[str, Any] | None = None,
    run_metadata: Dict[str, Any] | None = None,
):
    """Train HiRO (SAC high + SAC low).

    The HiRO high-level replay buffer (with OPC) is configured here (trainer),
    so hiro.py stays focused on the algorithm logic.
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    train_mode = str(getattr(cfg, "train_mode", "joint")).lower()
    if train_mode not in {"joint", "low_only", "high_only"}:
        raise ValueError(f"Unknown train_mode: {train_mode}")
    env_configs = env.get_attr("config")
    terminate_on_queue_envs = [
        int(i)
        for i, env_cfg in enumerate(env_configs)
        if isinstance(env_cfg, dict)
        and bool(env_cfg.get("terminate_on_queue_takeover", False))
    ]
    if train_mode != "low_only" and terminate_on_queue_envs:
        raise ValueError(
            "terminate_on_queue_takeover=True is only allowed for "
            f"HIRO train_mode='low_only'; got train_mode={train_mode!r}, "
            f"enabled_envs={terminate_on_queue_envs}"
        )

    # Resolve mode-dependent effective settings here; HIRO should not contain redundant guardrails.
    opc_enabled = train_mode == "joint" and bool(getattr(cfg, "use_off_policy_correction", True))
    low_level_type = str(getattr(cfg, "low_level_type", "sac")).lower()
    goal_sampler_cfg = getattr(cfg, "goal_sampler", GoalSamplerConfig()) if train_mode == "low_only" else GoalSamplerConfig()

    effective_cfg = replace(
        cfg,
        train_mode=train_mode,
        use_off_policy_correction=bool(opc_enabled),
        low_level_type=low_level_type,
        goal_sampler=goal_sampler_cfg,
    )

    high_sac_kwargs = dict(high_sac_kwargs)
    if train_mode == "low_only":
        # The high agent is construction-only in this mode; goals come from the
        # configured sampler, so a full high replay buffer only wastes memory.
        high_sac_kwargs["buffer_size"] = 1
        q_debug_cfg = dict(high_sac_kwargs.get("q_replay_debug", {}) or {})
        q_debug_cfg["enabled"] = False
        high_sac_kwargs["q_replay_debug"] = q_debug_cfg

    rb_kwargs = dict(high_sac_kwargs.get("replay_buffer_kwargs", {}) or {})

    # High-level transitions may span multiple standard high intervals during
    # queue takeover. The custom buffer carries gamma**interval_count.
    high_sac_kwargs["replay_buffer_class"] = HiROHighReplayBuffer
    rb_kwargs["enable_off_policy_correction"] = bool(opc_enabled)
    high_sac_kwargs["replay_buffer_kwargs"] = rb_kwargs

    low_debug_config = {
        "her_debug_enabled": int(low_her_debug_csv_interval_steps) > 0,
        "her_debug_max_records": int(low_her_debug_max_records),
        "her_debug_sample_prob": float(low_her_debug_sample_prob),
        "her_debug_csv_interval_steps": int(low_her_debug_csv_interval_steps),
        "her_debug_csv_max_rows_per_flush": int(low_her_debug_csv_max_rows_per_flush),
        "summary_interval_steps": int(low_debug_summary_interval_steps),
        "env_step_interval_steps": int(low_debug_env_step_interval_steps),
    }
    model = HIROSAC(
        env,
        high_sac_kwargs,
        low_sac_kwargs,
        effective_cfg,
        low_debug_config=low_debug_config,
    )

    sampler_type = str(getattr(effective_cfg.goal_sampler, "type", "")).lower()
    if sampler_type in {"speed_near_cruise", "near_cruise", "cruise_nearby"}:
        # For speed-near-cruise goals, diversify x by randomizing ego initial speed.
        cfg_list = env.get_attr("config")
        for i, cfg_i in enumerate(cfg_list):
            cfg_new = dict(cfg_i)
            cfg_new["ego_speed_range"] = [8.0, 12.0]
            env.set_attr("config", cfg_new, indices=i)
        print("[HIRO Trainer] Enabled ego_speed_range=[8,12] for speed_near_cruise sampler")

    checkpoint_save_freq = 50_000
    logging_high_transition_csv_all = max(0, int(high_transition_csv_all))
    _write_hiro_run_config(
        log_dir=log_dir,
        env=env,
        total_timesteps=total_timesteps,
        save_dir=save_dir,
        save_name_prefix=save_name_prefix,
        seed=seed,
        high_transition_csv_all=logging_high_transition_csv_all,
        high_transition_csv_envs=high_transition_csv_envs,
        low_transition_detail_csv=bool(low_transition_detail_csv),
        low_transition_detail_envs=low_transition_detail_envs,
        low_debug_config=low_debug_config,
        effective_cfg=effective_cfg,
        high_sac_kwargs=high_sac_kwargs,
        low_sac_kwargs=low_sac_kwargs,
        run_metadata={
            **dict(run_metadata or {}),
            "low_offline_eval_config": _json_safe(dict(low_offline_eval_config or {})),
        },
        checkpoint_save_freq=checkpoint_save_freq,
    )

    def _extract_ego_speed(high_obs_batch: np.ndarray) -> np.ndarray:
        arr = np.asarray(high_obs_batch, dtype=np.float32)
        _, kin, _ = utils.split_time_kinematics(arr, model.n_veh, model.feat_dim)
        ego_sub = utils.extract_ego_substate(kin, model.ego_feature_idx)
        if ego_sub.shape[1] >= 4:
            vx = ego_sub[:, 2]
            vy = ego_sub[:, 3]
            speed = np.sqrt(np.maximum(vx * vx + vy * vy, 0.0))
        elif ego_sub.shape[1] >= 3:
            speed = np.abs(ego_sub[:, 2])
        else:
            raise ValueError("Cannot extract ego speed from high observation")
        return np.asarray(speed, dtype=np.float32)

    # Set seed for high-level replay buffer if it exists (for OPC noise reproducibility)
    if hasattr(model.high_agent.replay_buffer, "set_seed"):
        model.high_agent.replay_buffer.set_seed(seed)
    if hasattr(model.low_agent, "replay_buffer") and hasattr(model.low_agent.replay_buffer, "set_seed"):
        model.low_agent.replay_buffer.set_seed(seed)

    n_envs = int(env.num_envs)

    logging_cb = HIROLoggingCallback(
        high_log_interval_episodes=n_envs * 1,
        low_log_interval_hi=n_envs * 4,
        # csv_log_freq_episodes=20,
        csv_log_freq_episodes=0,
        csv_save_dir=log_dir,
        low_obs_csv_interval_hi=10,
        low_obs_csv_env0_only=True,
        high_transition_csv_all=logging_high_transition_csv_all,
        high_transition_csv_envs=high_transition_csv_envs,
        low_transition_detail_csv=bool(low_transition_detail_csv),
        low_transition_detail_envs=low_transition_detail_envs,
        her_debug_csv_interval_steps=int(low_debug_config["her_debug_csv_interval_steps"]),
        her_debug_csv_max_rows_per_flush=int(low_debug_config["her_debug_csv_max_rows_per_flush"]),
        low_debug_summary_interval_steps=int(low_debug_config["summary_interval_steps"]),
        low_debug_env_step_interval_steps=int(low_debug_config["env_step_interval_steps"]),
        verbose=1,
    )
    checkpoint_cb = HIROCheckpointCallback(
        save_freq=checkpoint_save_freq,
        save_dir=save_dir,
        prefix=save_name_prefix,
        verbose=1,
    )
    low_traj_cb = HIROLowEpisodeTrajectoryCallback(
        save_path=os.path.join(log_dir, "low_episode_trajectories.jsonl"),
        verbose=0,
    )

    callback_items = [logging_cb, checkpoint_cb]
    low_eval_cfg = dict(low_offline_eval_config or {})
    if train_mode == "low_only" and bool(low_eval_cfg.get("enabled", False)):
        run_meta = dict(run_metadata or {})
        cases_path = str(low_eval_cfg.get("cases_path", "debug/hiro_low_eval_cases_snapshot012.json"))
        callback_items.insert(
            1,
            HIROLowOfflineEvalCallback(
                cases_path=cases_path,
                env_id=str(low_eval_cfg.get("env_id", run_meta.get("env_id", ""))),
                scenario_name=str(
                    low_eval_cfg.get(
                        "scenario_name",
                        run_meta.get("scenario_name", "multi_lane_stop_to_int"),
                    )
                ),
                scenario_module=str(
                    low_eval_cfg.get(
                        "scenario_module",
                        run_meta.get("scenario_module", "scenarios.multi_lane_stop_to_int"),
                    )
                ),
                env_overrides=dict(
                    low_eval_cfg.get(
                        "env_overrides",
                        run_meta.get("environment_overrides", {}) or {},
                    )
                ),
                eval_freq=int(low_eval_cfg.get("eval_freq", 200_000)),
                sample_size=int(low_eval_cfg.get("sample_size", 90)),
                seed=int(low_eval_cfg.get("seed", seed)),
                build_if_missing=bool(low_eval_cfg.get("build_if_missing", True)),
                build_config=dict(low_eval_cfg.get("build_config", {}) or {}),
                deterministic=bool(low_eval_cfg.get("deterministic", True)),
                verbose=int(low_eval_cfg.get("verbose", 1)),
            ),
        )
        print(
            "[HIRO Trainer] Enabled low offline eval: "
            f"cases={cases_path}, "
            f"freq={int(low_eval_cfg.get('eval_freq', 200_000))}, "
            f"sample_size={int(low_eval_cfg.get('sample_size', 90))}"
        )

    # callback = CallbackList([logging_cb, low_traj_cb, checkpoint_cb])
    callback = CallbackList(callback_items)
    
    if train_mode == "joint":
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )
    elif train_mode == "low_only":
        print(f"[HIRO Trainer] Training Low-Level Only. Goal Sampler: {effective_cfg.goal_sampler.type}")
        enable_vx_bounds = bool(getattr(effective_cfg, "high_goal_safe_enable_goal_vx_bounds", True))
        fixed_goal_vx = getattr(effective_cfg, "fixed_goal_vx", None)
        if fixed_goal_vx is not None and np.isclose(float(fixed_goal_vx), 0.0):
            enable_vx_bounds = False
        sampler = get_goal_sampler(
            effective_cfg.goal_sampler,
            model.high_agent.action_space,
            bounds_fn=model.high_goal_safe_bounds.compute_np,
            speed_fn=_extract_ego_speed,
            enable_vx_bounds=enable_vx_bounds,
            dynamic_feasible_lane_intervals=bool(
                getattr(effective_cfg, "high_goal_dynamic_feasible_lane_intervals", False)
            ),
        )
        model.learn_low(
            total_timesteps=total_timesteps,
            goal_sampler=sampler,
            callback=callback,
            progress_bar=True,
        )
    elif train_mode == "high_only":
        print(f"[HIRO Trainer] Training High-Level Only.")
        model.learn_high(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    if train_mode != "low_only":
        print("[HIRO Trainer] Saving final high-level model...")
        model.high_agent.save(os.path.join(save_dir, f"{save_name_prefix}_high_final.zip"))
    if train_mode != "high_only":
        print("[HIRO Trainer] Saving final low-level model...")
        model.low_agent.save(os.path.join(save_dir, f"{save_name_prefix}_low_final.zip"))
    print("[HIRO Trainer] Final model save finished")
