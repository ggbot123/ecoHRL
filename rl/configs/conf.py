from __future__ import annotations
from typing import Dict, Any
from rl.algos.HRL.hiro import HIROConfig

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


def get_sac_kwargs(log_dir: str, seed: int, level: str = "high") -> Dict[str, Any]:
    """
    返回 SAC 初始化所需的 keyword arguments（不包括 env）。
    """
    sac_kwargs = dict(
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
    if level == 'low':
        sac_kwargs['verbose'] = 0
    return sac_kwargs

def get_hiro_config() -> HIROConfig:
    """
    返回 HIRO 的超参数配置（高层间隔 / intrinsic 系数等）。
    """
    return HIROConfig(
        high_interval=25,      # 0.1s * 25 = 2.5s，一次高层决策（0.4Hz）
        gamma_high=0.99,
        gamma_low=0.99,
        buffer_size=1_000_000,
        batch_size=256,
        learning_starts=10_000,
        gradient_steps_high=1,
        gradient_steps_low=1,
        train_freq=1,
        intrinsic_coef=1.0,   # 你之前设置的 intrinsic reward 系数
        device="auto",
    )