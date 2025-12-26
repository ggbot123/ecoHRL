# rl/algos/hiro/trainer.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from rl.algos.HRL.hiro import HIROSAC
from rl.algos.HRL.buffer import HiROHighReplayBuffer
from rl.algos.HRL.callbacks import HIROLoggingCallback, HIROCheckpointCallback
from stable_baselines3.common.callbacks import CallbackList


def train_hiro(
    env,
    total_timesteps: int,
    log_dir: str,
    save_dir: str,
    high_sac_kwargs: Dict[str, Any],
    low_sac_kwargs: Dict[str, Any],
    cfg,
    high_rb_kwargs: Optional[Dict[str, Any]],
    save_name_prefix: str,
):
    """Train HiRO (SAC high + SAC low).

    The HiRO high-level replay buffer (with OPC) is configured here (trainer),
    so hiro.py stays focused on the algorithm logic.
    """
    os.makedirs(save_dir, exist_ok=True)

    high_sac_kwargs = dict(high_sac_kwargs)
    if bool(getattr(cfg, 'use_off_policy_correction', True)):
        rb_kwargs = dict(high_sac_kwargs.get("replay_buffer_kwargs", {}) or {})
        rb_kwargs.update(dict(high_rb_kwargs))
        high_sac_kwargs["replay_buffer_class"] = HiROHighReplayBuffer
        high_sac_kwargs["replay_buffer_kwargs"] = rb_kwargs

    model = HIROSAC(env, high_sac_kwargs, low_sac_kwargs, cfg)
    n_envs = int(env.num_envs)

    logging_cb = HIROLoggingCallback(high_log_interval_episodes=n_envs * 1, low_log_interval_hi=n_envs * 4)
    checkpoint_cb = HIROCheckpointCallback(
        save_freq=50_000,  # 或从参数传进来
        save_dir=save_dir,
        prefix=save_name_prefix,
        verbose=1,
    )
    callback = CallbackList([logging_cb, checkpoint_cb])

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    model.high_agent.save(os.path.join(save_dir, f"{save_name_prefix}_high_final.zip"))
    model.low_agent.save(os.path.join(save_dir, f"{save_name_prefix}_low_final.zip"))
