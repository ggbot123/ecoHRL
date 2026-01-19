# rl/algos/hiro/trainer.py
from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Dict

from rl.algos.HRL.hiro import HIROSAC
from rl.algos.HRL.buffer import HiROHighReplayBuffer
from rl.algos.HRL.callbacks import HIROLoggingCallback, HIROCheckpointCallback
from rl.algos.HRL.goal_samplers import GoalSamplerConfig, get_goal_sampler
from stable_baselines3.common.callbacks import CallbackList


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
):
    """Train HiRO (SAC high + SAC low).

    The HiRO high-level replay buffer (with OPC) is configured here (trainer),
    so hiro.py stays focused on the algorithm logic.
    """
    os.makedirs(save_dir, exist_ok=True)

    train_mode = str(getattr(cfg, "train_mode", "joint")).lower()
    if train_mode not in {"joint", "low_only", "high_only"}:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    # Resolve mode-dependent effective settings here; HIRO should not contain redundant guardrails.
    opc_enabled = train_mode == "joint" and bool(getattr(cfg, "use_off_policy_correction", True))
    low_level_type = str(getattr(cfg, "low_level_type", "sac")).lower() if train_mode == "high_only" else "sac"
    goal_sampler_cfg = getattr(cfg, "goal_sampler", GoalSamplerConfig()) if train_mode == "low_only" else GoalSamplerConfig()

    effective_cfg = replace(
        cfg,
        train_mode=train_mode,
        use_off_policy_correction=bool(opc_enabled),
        low_level_type=low_level_type,
        goal_sampler=goal_sampler_cfg,
    )

    high_sac_kwargs = dict(high_sac_kwargs)
    rb_kwargs = dict(high_sac_kwargs.get("replay_buffer_kwargs", {}) or {})

    if opc_enabled:
        high_sac_kwargs["replay_buffer_class"] = HiROHighReplayBuffer
        rb_kwargs["enable_off_policy_correction"] = True
        high_sac_kwargs["replay_buffer_kwargs"] = rb_kwargs
    else:
        high_sac_kwargs.pop("replay_buffer_class", None)
        for k in ("n_candidates", "noise_std", "enable_off_policy_correction"):
            rb_kwargs.pop(k, None)
        if rb_kwargs:
            high_sac_kwargs["replay_buffer_kwargs"] = rb_kwargs
        else:
            high_sac_kwargs.pop("replay_buffer_kwargs", None)

    model = HIROSAC(env, high_sac_kwargs, low_sac_kwargs, effective_cfg)
    
    # Set seed for high-level replay buffer if it exists (for OPC noise reproducibility)
    if hasattr(model.high_agent.replay_buffer, "set_seed"):
        model.high_agent.replay_buffer.set_seed(seed)

    n_envs = int(env.num_envs)

    logging_cb = HIROLoggingCallback(
        high_log_interval_episodes=n_envs * 1,
        low_log_interval_hi=n_envs * 4,
        csv_log_freq_episodes=20,
        csv_save_dir=log_dir,
        verbose=1,
    )
    checkpoint_cb = HIROCheckpointCallback(
        save_freq=50_000,
        save_dir=save_dir,
        prefix=save_name_prefix,
        verbose=1,
    )

    callback = CallbackList([logging_cb, checkpoint_cb])
    
    if train_mode == "joint":
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )
    elif train_mode == "low_only":
        print(f"[HIRO Trainer] Training Low-Level Only. Goal Sampler: {effective_cfg.goal_sampler.type}")
        sampler = get_goal_sampler(effective_cfg.goal_sampler, model.high_agent.action_space)
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
        model.high_agent.save(os.path.join(save_dir, f"{save_name_prefix}_high_final.zip"))
    if train_mode != "high_only":
        model.low_agent.save(os.path.join(save_dir, f"{save_name_prefix}_low_final.zip"))
