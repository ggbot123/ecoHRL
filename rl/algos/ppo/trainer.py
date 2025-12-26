# rl/algos/ppo/trainer.py
from __future__ import annotations

import os
from typing import Dict, Any, List, Callable

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from rl.algos.ppo.ppo import PPO
from rl.algos.ppo.callbacks import RewardComponentsTensorboardCallback  # 你已有的 callback


def train_ppo(
    env_fns: List[Callable[[], gym.Env]],
    eval_env_fn: Callable[[], gym.Env],
    total_timesteps: int,
    log_dir: str,
    save_dir: str,
    ppo_kwargs: Dict[str, Any],
    eval_freq: int = 10_000,
    save_freq: int = 50_000,
    save_name_prefix: str = "ppo",
) -> None:
    """
    PPO 训练入口（不负责创建 env，也不设种子）：

    参数：
        env_fns        : DummyVecEnv 使用的一组环境构造函数（make_env() 返回的 _init）
        eval_env_fn    : 用于 EvalCallback 的单环境构造函数
        total_timesteps: 总训练步数
        log_dir        : 日志目录（给 tensorboard_log）
        save_dir       : 模型保存目录
        ppo_kwargs     : 传给 PPO 的 kwargs（不包含 env）
        eval_freq      : 评估频率（以 environment steps 计）
        save_freq      : checkpoint 频率
    """
    # 训练环境：多进程并行（n_envs>1），否则退回 DummyVecEnv
    if len(env_fns) > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    eval_env = DummyVecEnv([eval_env_fn])

    # 评估 + checkpoint 回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir,
        name_prefix=f"{save_name_prefix}_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    rc_tb_callback = RewardComponentsTensorboardCallback(log_freq=1, verbose=0)

    # 不要把 env 写在 kwargs 里
    ppo_kwargs = dict(ppo_kwargs)
    ppo_kwargs.pop("env", None)
    ppo_kwargs.setdefault("tensorboard_log", log_dir)

    model = PPO(
        env=vec_env,
        **ppo_kwargs,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, rc_tb_callback],
        log_interval=1,
        progress_bar=True,
    )

    final_path = os.path.join(save_dir, f"{save_name_prefix}_final")
    model.save(final_path)
    print(f"[PPO] 训练完成，模型已保存到: {final_path}")

    vec_env.close()
    eval_env.close()
