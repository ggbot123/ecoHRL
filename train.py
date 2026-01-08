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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from configs.conf import (
    get_env_config,
    get_ppo_kwargs,
    get_sac_kwargs,
    get_hiro_config,
    get_hiro_high_sac_kwargs,
    get_hiro_low_sac_kwargs,
)
from rl.algos.ppo.trainer import train_ppo
from rl.algos.sac.trainer import train_sac
from rl.algos.HRL.trainer import train_hiro
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


def make_env(env_overrides: dict | None = None, render_mode: str | None = None):
    """Return an env constructor compatible with DummyVecEnv/SubprocVecEnv."""
    env_seed = int(master_rng.integers(0, 2**31 - 1))

    def _init():
        cfg = get_env_config(env_overrides or {})
        env = gym.make("multi-lane-custom-v0", render_mode=render_mode, config=cfg)
        env = Monitor(env)
        env.reset(seed=env_seed)
        return env

    return _init


def main(algo: str, total_timesteps: int, eval_freq: int, save_freq: int, n_envs: int, log_root: str = "./logs", save_root: str = "./models", run_name: str | None = None) -> None:
    global master_rng
    set_global_seed(MASTER_SEED)
    master_rng = np.random.default_rng(MASTER_SEED)

    algo = algo.lower()

    if run_name is None:
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{algo}_{time_str}"

    log_dir = os.path.join(log_root, run_name)
    save_dir = os.path.join(save_root, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[MAIN] algo={algo}")
    print(f"[MAIN] log_dir={log_dir}")
    print(f"[MAIN] save_dir={save_dir}")

    #### ================ Train-time overrides (optional) ================ ####
    env_overrides = {
        # "initial_lane_id": "random",
        "initial_lane_id": 1,
        "PERCEPTION_DISTANCE": 200,
        "observation": {
            "vehicles_count": 20,
            "vehicles_count_local": 5,
            "features_range": {
                "x": [-200, 200],
                "y": [-10, 10],
                "vx": [-15, 15],
                "vy": [-10, 10],
            },
        },
        "goal_lane_id": 2,
    }
    #### ================================================================= ####

    if algo == "ppo":
        ppo_kwargs = get_ppo_kwargs(log_dir=log_dir, seed=MASTER_SEED)
        env_fns = [make_env(env_overrides) for _ in range(n_envs)]
        eval_env_fn = make_env(env_overrides)
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
        env_fns = [make_env(env_overrides) for _ in range(n_envs)]
        eval_env_fn = make_env(env_overrides)
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
        sac_kwargs_high = get_hiro_high_sac_kwargs(log_dir=os.path.join(log_dir, "hiro_high"), seed=MASTER_SEED)
        sac_kwargs_low = get_hiro_low_sac_kwargs(log_dir=os.path.join(log_dir, "hiro_low"), seed=MASTER_SEED)

        hiro_cfg: HIROConfig = get_hiro_config()
        print(f"[HIRO] Train Mode: {hiro_cfg.train_mode}, Goal Sampler: {hiro_cfg.goal_sampler_type}")

        # Use DummyVecEnv when using RuleBasedController because it requires direct access to env objects
        # Note: DummyVecEnv runs environments sequentially (serial execution), so it will be slower than SubprocVecEnv.
        if hiro_cfg.low_level_type == "rule_based":
            print("[HIRO] Using DummyVecEnv for Rule-Based Low-Level Controller (Direct object access required)")
            env = DummyVecEnv([make_env(env_overrides) for _ in range(n_envs)])
        else:
            env = SubprocVecEnv([make_env(env_overrides) for _ in range(n_envs)])

        train_hiro(
            env=env,
            total_timesteps=total_timesteps,
            log_dir=log_dir,
            save_dir=save_dir,
            high_sac_kwargs=sac_kwargs_high,
            low_sac_kwargs=sac_kwargs_low,
            cfg=hiro_cfg,
            save_name_prefix="hiro",
            seed=MASTER_SEED,
        )
        env.close()

    else:
        raise ValueError(f"未知算法类型: {algo}")

    print("[MAIN] 训练完成")


if __name__ == "__main__":
    # main(
    #     algo="ppo",
    #     total_timesteps=1_000_000,
    #     eval_freq=10_000,
    #     save_freq=50_000,
    #     n_envs=8,
    #     run_name=f"ppo_1e7_lane1_seed2"
    # )
    # main(
    #     algo="sac",
    #     log_root="./logs/current",
    #     total_timesteps=5_000_000,
    #     eval_freq=10_000,
    #     save_freq=50_000,
    #     n_envs=8,
    #     run_name=f"sac_1e7_lane1_seed2"
    # )
    main(
        algo="hiro",
        log_root="./logs/current",
        total_timesteps=10_000_000,
        eval_freq=10_000,
        save_freq=50_000,
        n_envs=8,
        run_name=f"hiro_260108_onlyHigh_rule_varTargetV_hierObs"
    )
