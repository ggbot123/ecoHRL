# train.py
from __future__ import annotations
import os
import random
from datetime import datetime
import gymnasium as gym
import numpy as np
import torch as th

import scenarios.multi_lane  # 注册 multi-lane-custom-v0
from stable_baselines3.common.monitor import Monitor

from rl.configs.conf import (
    get_ppo_kwargs,
    get_sac_kwargs,
    get_hiro_config,
)
from rl.algos.ppo.trainer import train_ppo
from rl.algos.sac.trainer import train_sac
from rl.algos.HRL.trainer import train_hiro  # 你之前的 HIRO trainer
from rl.algos.HRL.hiro import HIROConfig

MASTER_SEED = 42
master_rng: np.random.Generator

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def make_env():
    """
    返回一个“环境构造函数”（与 DummyVecEnv 兼容）：
    - 在外层用 master_rng 生成一个固定 env_seed
    - _init() 内部创建 MultiLaneEnv，并用该 seed reset 一次
    """
    env_seed = int(master_rng.integers(0, 2**31 - 1))
    def _init():
        env = gym.make(
            "multi-lane-custom-v0",
            render_mode=None,
            config={
                "policy_frequency": 10,
                "duration": 50.0,
                "observation": {
                    "type": "Kinematics",
                    "normalize": False,
                    "include_time": True,
                    "time_range": [0.0, 40.0],
                    "include_obstacles": False,
                },
                "initial_lane_id": 1,
                # "initial_lane_id": "random",
            },
        )
        env = Monitor(env)
        env.reset(seed=env_seed)
        return env
    return _init


def main(
    algo: str,
    total_timesteps: int,
    eval_freq: int,
    save_freq: int,
    n_envs: int,
    log_root: str = "./logs",
    save_root: str = "./models",
) -> None:
    global master_rng

    set_global_seed(MASTER_SEED)
    master_rng = np.random.default_rng(MASTER_SEED)

    algo = algo.lower()

    # 每次运行一个新的时间戳目录
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{algo}_{time_str}"
    log_dir = os.path.join(log_root, run_name)
    save_dir = os.path.join(save_root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[MAIN] algo={algo}")
    print(f"[MAIN] log_dir={log_dir}")
    print(f"[MAIN] save_dir={save_dir}")

    if algo == "ppo":
        ppo_kwargs = get_ppo_kwargs(log_dir=log_dir, seed=MASTER_SEED)
        env_fns = [make_env() for _ in range(n_envs)]
        eval_env_fn = make_env()
        train_ppo(
            env_fns=env_fns,
            eval_env_fn=eval_env_fn,
            total_timesteps=total_timesteps,
            log_dir=log_dir,
            save_dir=save_dir,
            ppo_kwargs=ppo_kwargs,
            eval_freq=eval_freq,
            save_freq=save_freq,
            save_name_prefix="ppo",
        )

    elif algo == "sac":
        sac_kwargs = get_sac_kwargs(log_dir=log_dir, seed=MASTER_SEED)
        env_fns = [make_env() for _ in range(n_envs)]
        eval_env_fn = make_env()
        train_sac(
            env_fns=env_fns,
            eval_env_fn=eval_env_fn,
            total_timesteps=total_timesteps,
            log_dir=log_dir,
            save_dir=save_dir,
            sac_kwargs=sac_kwargs,
            eval_freq=eval_freq,
            save_freq=save_freq,
            save_name_prefix="sac",
        )

    elif algo == "hiro":
        sac_kwargs_high = get_sac_kwargs(log_dir=os.path.join(log_dir, "hiro_high"), seed=MASTER_SEED, level="high")
        sac_kwargs_low = get_sac_kwargs(log_dir=os.path.join(log_dir, "hiro_low"), seed=MASTER_SEED, level="low")
        hiro_cfg: HIROConfig = get_hiro_config()

        # 单环境实例：make_env() 返回 _init 函数，这里调用一次得到 env
        env = make_env()()

        train_hiro(
            env=env,
            total_timesteps=total_timesteps,
            log_dir=log_dir,
            save_dir=save_dir,
            high_sac_kwargs=sac_kwargs_high,
            low_sac_kwargs=sac_kwargs_low,
            cfg=hiro_cfg,
            save_name_prefix="hiro",
        )
        env.close()

    else:
        raise ValueError(f"未知算法类型: {algo}, 只支持 ppo / sac / hiro")

    print("[MAIN] 训练完成")


if __name__ == "__main__":
    # main(
    #     algo="ppo",
    #     total_timesteps=1_000_000,
    #     eval_freq=10_000,
    #     save_freq=50_000,
    #     n_envs=8,
    # )
    # main(
    #     algo="sac",
    #     total_timesteps=5_000_000,
    #     eval_freq=10_000,
    #     save_freq=50_000,
    #     n_envs=4,
    # )
    main(
        algo="hiro",
        total_timesteps=1_000_000,
        eval_freq=10_000,
        save_freq=50_000,
        n_envs=1,
    )
