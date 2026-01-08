# rl/algos/hiro/trainer.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from rl.algos.HRL.hiro import HIROSAC
from rl.algos.HRL.buffer import HiROHighReplayBuffer
from rl.algos.HRL.callbacks import HIROLoggingCallback, HIROCheckpointCallback
from rl.algos.HRL.goal_samplers import get_goal_sampler
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

    high_sac_kwargs = dict(high_sac_kwargs)
    if bool(getattr(cfg, 'use_off_policy_correction', True)):
        # Inject the custom buffer class.
        # The static buffer kwargs (n_candidates, etc.) should already be in high_sac_kwargs["replay_buffer_kwargs"]
        # from get_hiro_high_sac_kwargs() in conf.py.
        high_sac_kwargs["replay_buffer_class"] = HiROHighReplayBuffer

    model = HIROSAC(env, high_sac_kwargs, low_sac_kwargs, cfg)
    
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
    
    # Retrieve mode from config
    train_mode = getattr(cfg, "train_mode", "joint")
    goal_sampler_type = getattr(cfg, "goal_sampler_type", "uniform")

    if train_mode == "joint":
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
        )
    elif train_mode == "low_only":
        print(f"[HIRO Trainer] Training Low-Level Only. Goal Sampler: {goal_sampler_type}")
        sampler = get_goal_sampler(goal_sampler_type, model.high_agent.action_space)
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
