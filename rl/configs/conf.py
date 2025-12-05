from __future__ import annotations
from typing import Dict, Any


def get_ppo_kwargs(log_dir: str, seed: int) -> Dict[str, Any]:
    """
    返回 PPO 初始化所需的 keyword arguments（不包括 env）。
    """
    return dict(
        policy="MlpPolicy",
        device="cpu",      # PPO 默认用 cpu
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,
        # ==== 下面是你当前用的 PPO 超参数 ====
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.0,
    )


def get_sac_kwargs(log_dir: str, seed: int) -> Dict[str, Any]:
    """
    返回 SAC 初始化所需的 keyword arguments（不包括 env）。
    """
    return dict(
        policy="MlpPolicy",
        verbose=1,
        tensorboard_log=log_dir,
        seed=seed,
        # ==== 下面是你当前用的 SAC 超参数 ====
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        learning_rate=3e-4,
        train_freq=(1, "step"),
        gradient_steps=1,
    )
