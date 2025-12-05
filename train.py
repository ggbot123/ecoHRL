import os, random
from datetime import datetime
import gymnasium as gym
import numpy as np
import torch as th

from rl.configs.conf import *
from rl.algos.ppo.ppo import PPO
from rl.algos.sac.sac import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from rl.utils.callbacks import RewardComponentsTensorboardCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import scenarios.multi_lane  # 触发 __init__.py 里的 register

MASTER_SEED = 42

# ---- 固定所有库的随机种子 ----
random.seed(MASTER_SEED)
np.random.seed(MASTER_SEED)
th.manual_seed(MASTER_SEED)
th.cuda.manual_seed_all(MASTER_SEED)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False
master_rng = np.random.default_rng(MASTER_SEED)

# ------------------------- 环境构造函数 ------------------------- #
def make_env():
    # 确保每次运行都是同一串 seeds
    env_seed = int(master_rng.integers(0, 2**31 - 1))
    def _init():
        env = gym.make(
            "multi-lane-custom-v0",
            render_mode=None,
            # render_mode="human",
            config={
                "policy_frequency": 10, 
                "duration": 30,               # [s]
            },
        )
        env = Monitor(env)
        env.reset(seed=env_seed)
        return env
    return _init

# ------------------------- 训练主流程 ------------------------- #
def train(
    algo: str = "ppo",
    total_timesteps: int = 500_000,
    eval_freq: int = 10_000,
    save_freq: int = 50_000,
    n_envs: int = 1,
    log_dir: str = "./logs",
    save_root: str = "./models",
):
    algo = algo.lower()
    vec_env = DummyVecEnv([make_env() for _ in range(n_envs)])
    eval_env = DummyVecEnv([make_env()])  # 评估环境

    # 为每次训练创建独立的 log / model 目录
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(save_root, f"{algo}_{time_str}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 定义回调：定期评估 + 存 checkpoint
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
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    rc_tb_callback = RewardComponentsTensorboardCallback(log_freq=1, verbose=0)

    # 根据 algo 选择算法和超参数
    if algo == "ppo":
        ppo_kwargs = get_ppo_kwargs(log_dir=log_dir, seed=MASTER_SEED)
        model = PPO(
            env=vec_env,
            **ppo_kwargs,
        )
    else:  # SAC 只支持连续 Box 动作，请确保 ParamLaneAccelAction 把 action_space 暴露为 Box([-1,1]^2)
        assert isinstance(vec_env.action_space, gym.spaces.Box), "使用 SAC 时 env.action_space 必须是 gym.spaces.Box"
        sac_kwargs = get_sac_kwargs(log_dir=log_dir, seed=MASTER_SEED)
        model = SAC(
            env=vec_env,
            **sac_kwargs,
        )

    # 开始训练
    print(f"[INFO] {algo.upper()} 训练开始")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, rc_tb_callback],
        progress_bar=True,
    )
    # 保存模型
    final_model_path = os.path.join(save_dir, f"{algo}_final")
    model.save(final_model_path)
    print(f"[INFO] {algo.upper()} 训练完成，模型已保存到：{final_model_path}")
    vec_env.close()
    eval_env.close()

if __name__ == "__main__":
    train(
        algo="ppo", 
        total_timesteps=1_000_000, 
        eval_freq=10_000, 
        save_freq=50_000,
        n_envs=4,
    )
    # train(
    #     algo="sac", 
    #     total_timesteps=1_000_000, 
    #     eval_freq=10_000, 
    #     save_freq=50_000,
    #     n_envs=4,
    # )