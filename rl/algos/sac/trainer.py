# rl/algos/sac/trainer.py
from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from rl.algos.sac.sac import SAC
from rl.algos.sac.callbacks import RewardComponentsTensorboardCallback, SACTransitionLoggingCallback


def _json_safe(value: Any) -> Any:
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


def _write_sac_run_config(
    *,
    log_dir: str,
    vec_env,
    total_timesteps: int,
    save_dir: str,
    save_name_prefix: str,
    eval_freq: int,
    save_freq: int,
    sac_kwargs: Dict[str, Any],
    run_metadata: Dict[str, Any] | None,
) -> None:
    os.makedirs(log_dir, exist_ok=True)
    try:
        env_configs = vec_env.get_attr("config")
    except Exception as exc:
        env_configs = [{"error": f"vec_env.get_attr('config') failed: {exc!r}"}]

    payload = {
        "run_metadata": dict(run_metadata or {}),
        "trainer": {
            "total_timesteps": int(total_timesteps),
            "save_dir": save_dir,
            "save_name_prefix": save_name_prefix,
            "n_envs": int(getattr(vec_env, "num_envs", len(env_configs) if isinstance(env_configs, list) else 1)),
            "eval_freq": int(eval_freq),
            "save_freq": int(save_freq),
        },
        "environment": {
            "env0_config": env_configs[0] if isinstance(env_configs, list) and env_configs else env_configs,
            "all_env_configs": env_configs,
        },
        "sac": {
            "kwargs": sac_kwargs,
        },
    }
    out_path = os.path.join(log_dir, "run_config.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[SAC Trainer] Saved run config: {out_path}")


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
    run_metadata: Dict[str, Any] | None = None,
    transition_csv_episode_freq: int = 1,
    transition_csv_envs: str = "env0",
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
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    rc_tb_callback = RewardComponentsTensorboardCallback(verbose=0)
    callbacks = [eval_callback, checkpoint_callback, rc_tb_callback]
    if int(transition_csv_episode_freq) > 0:
        callbacks.append(
            SACTransitionLoggingCallback(
                save_dir=log_dir,
                episode_freq=transition_csv_episode_freq,
                envs=transition_csv_envs,
                verbose=0,
            )
        )

    sac_kwargs = dict(sac_kwargs)
    sac_kwargs.pop("env", None)
    sac_kwargs.setdefault("tensorboard_log", log_dir)

    _write_sac_run_config(
        log_dir=log_dir,
        vec_env=vec_env,
        total_timesteps=total_timesteps,
        save_dir=save_dir,
        save_name_prefix=save_name_prefix,
        eval_freq=eval_freq,
        save_freq=save_freq,
        sac_kwargs=sac_kwargs,
        run_metadata={
            **dict(run_metadata or {}),
            "sac_transition_csv_episode_freq": int(transition_csv_episode_freq),
            "sac_transition_csv_envs": str(transition_csv_envs),
        },
    )

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
        callback=callbacks,
        log_interval=1,
        progress_bar=True,
    )

    final_path = os.path.join(save_dir, f"{save_name_prefix}_final")
    model.save(final_path)
    print(f"[SAC] 训练完成，模型已保存到: {final_path}")

    vec_env.close()
    eval_env.close()
