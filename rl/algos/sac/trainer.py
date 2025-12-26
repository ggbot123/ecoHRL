# rl/algos/sac/trainer.py
from __future__ import annotations

import os
from typing import Dict, Any, List, Callable

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from rl.algos.sac.sac import SAC
from rl.algos.sac.callbacks import RewardComponentsTensorboardCallback


def train_sac(
    env_fns: List[Callable[[], gym.Env]],
    eval_env_fn: Callable[[], gym.Env],
    total_timesteps: int,
    log_dir: str,
    save_dir: str,
    sac_kwargs: Dict[str, Any],
    eval_freq: int = 10_000,
    save_freq: int = 50_000,
    save_name_prefix: str = "sac",
) -> None:
    """
    SAC 训练入口（不负责创建 env 和 seed）。

    参数定义同 PPO 版本。
    """
    if len(env_fns) > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)
    eval_env = DummyVecEnv([eval_env_fn])

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
        save_replay_buffer=True,
        save_vecnormalize=False,
    )
    rc_tb_callback = RewardComponentsTensorboardCallback(verbose=0)

    sac_kwargs = dict(sac_kwargs)
    sac_kwargs.pop("env", None)
    sac_kwargs.setdefault("tensorboard_log", log_dir)

    # SAC 只支持 Box 动作
    assert isinstance(
        vec_env.action_space, gym.spaces.Box
    ), "train_sac: env.action_space 必须是 Box (连续动作)"

    model = SAC(
        env=vec_env,
        **sac_kwargs,
    )

    # 对 off-policy 算法，log_interval 以“episode数”为单位；设为1让 tensorboard 点更密
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, rc_tb_callback],
        log_interval=1,
        progress_bar=True,
    )

    final_path = os.path.join(save_dir, f"{save_name_prefix}_final")
    model.save(final_path)
    print(f"[SAC] 训练完成，模型已保存到: {final_path}")

    vec_env.close()
    eval_env.close()
