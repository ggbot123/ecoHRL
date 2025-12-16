# rl/algos/hiro/hiro.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import gymnasium as gym
import numpy as np
import torch as th

from rl.algos.sac.sac import SAC
from rl.utils import utils
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import get_device, configure_logger
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback


class DummyEnv(gym.Env):
    """
    一个只提供 observation_space / action_space 的 dummy Env，
    供 SB3 的 SAC 构造函数使用（我们不会调用它的 step/reset）。
    """
    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, **kwargs):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError


class SB3AgentWrapper:
    """
    SB3 算法的包装器，用于将 SB3 的 Off-Policy 算法（如 SAC）适配到 HRL 框架中。
    隐藏了对 _last_obs, _sample_action 等私有属性的访问。
    """
    def __init__(self, agent: SAC, config_train_freq: int, gradient_steps: int, batch_size: int):
        self.agent = agent
        self.train_freq = config_train_freq
        self.gradient_steps = gradient_steps
        self.batch_size = batch_size
        self.replay_buffer = agent.replay_buffer
    
    @property
    def num_timesteps(self) -> int:
        return self.agent.num_timesteps
    
    @num_timesteps.setter
    def num_timesteps(self, value: int):
        self.agent.num_timesteps = value

    def set_logger(self, logger):
        self.agent.set_logger(logger)

    def sample_action(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        obs_vec = obs[None, :]  # 增加 batch 维度 (1, dim)
        self.agent._last_obs = obs_vec
        
        # 调用 SB3 内部采样逻辑 (处理了 exploration noise 和 learning_starts)
        action_vec, buffer_action_vec = self.agent._sample_action(
            learning_starts=self.agent.learning_starts,
            action_noise=self.agent.action_noise,
        )
        return action_vec[0], buffer_action_vec[0]

    def store_transition(self, obs: np.ndarray, action: np.ndarray, 
                         next_obs: np.ndarray, reward: float, done: bool, info: Dict[str, Any]):
        # 构造符合 SB3 buffer 接口的数据
        obs_vec = obs[None, :]
        next_obs_vec = next_obs[None, :]
        action_vec = action[None, :] # 注意：这里通常存 buffer_action
        reward_arr = np.array([reward], dtype=np.float32)
        done_arr = np.array([done], dtype=np.bool_)
        infos = [info]

        self.agent._store_transition(
            replay_buffer=self.replay_buffer,
            buffer_action=action_vec,
            new_obs=next_obs_vec,
            reward=reward_arr,
            dones=done_arr,
            infos=infos
        )

    def train_if_needed(self):
        if (self.num_timesteps > self.agent.learning_starts and 
            self.num_timesteps % self.train_freq == 0):
            self.agent.train(
                gradient_steps=self.gradient_steps,
                batch_size=self.batch_size
            )
    
    def save(self, path: str):
        self.agent.save(path)

    def __getattr__(self, name):
        return getattr(self.agent, name)

@dataclass
class HIROConfig:
    high_interval: int         # 高层每 high_interval 个 env.step 决策一次
    batch_size: int
    gradient_steps_high: int
    gradient_steps_low: int
    train_freq: int
    intrinsic_coef: float     # 末状态距离 goal 的 intrinsic reward 系数
    device: str


class HIROSAC:
    def __init__(
        self,
        env: gym.Env,
        high_sac_kwargs: Dict[str, Any],
        low_sac_kwargs: Dict[str, Any],
        config: HIROConfig,
    ):
        self.env = env
        self.cfg = config
        self.device = get_device(config.device)
        self.total_timesteps: int = 0      # 以 env.step 计的步数（低层）
        self.n_envs: int = getattr(env, "num_envs", 1)
        if not hasattr(self.env, "num_envs"):
            self.env.num_envs = self.n_envs

        # ---- 用一次 reset 的 obs 初始化 Kinematics 元信息 ----
        obs0, _ = env.reset()
        obs0_flat = np.asarray(obs0, dtype=np.float32).reshape(-1)
        keep_features = ("x", "y", "vx", "vy")  # ego 子状态中参与 HIRO goal 的特征
        (
            self.n_veh,
            self.feat_dim,
            self.feature_names,
            self.ego_feature_idx,
            self.goal_feature_ranges,
        ) = utils.init_kinematics_meta(env, obs0_flat, keep_features)
        self.kin_flat_dim: int = self.n_veh * self.feat_dim
        self.ego_dim: int = len(self.ego_feature_idx)

        # ---- 从env中获取的必要变量 --- #
        env_cfg = getattr(env.unwrapped, "config", {})
        self.v_min: float = 0.0
        self.v_max: float = float(env_cfg["speed_limit"])
        policy_freq = float(env_cfg["policy_frequency"])
        self.dt: float = 1.0 / policy_freq
        lanes = int(env_cfg["lanes_count"])
        self.lane_center_ys = np.array(
            [
                env.unwrapped.road.network.get_lane(("0", "1", lane_id)).position(0.0, 0.0)[1]
                for lane_id in range(lanes)
            ],
            dtype=np.float32,
        )
        ego_feature_order = [self.feature_names[i] for i in self.ego_feature_idx]
        assert (self.ego_dim == 4 and ego_feature_order == ["x", "y", "vx", "vy"])

        # ----- 定义 Spaces -----  #
        high_obs_dim = self.kin_flat_dim + 1
        high_obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(high_obs_dim,), dtype=np.float32)

        t_h = float(self.cfg.high_interval) * self.dt
        # 目标空间：[rel_x, rel_y, vx]，rel_y从[-1, 1]映射到左右车道中心线
        goal_low = np.array([self.v_min * t_h, -1, self.v_min], dtype=np.float32)
        goal_high = np.array([self.v_max * t_h, 1, self.v_max], dtype=np.float32)
        high_act_space = gym.spaces.Box(goal_low, goal_high, dtype=np.float32)

        low_obs_dim = self.kin_flat_dim + self.ego_dim + 1
        low_obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(low_obs_dim,), dtype=np.float32)
        low_act_space = env.action_space

        # ----- 创建 Wrapper 实例 ----- #
        high_dummy = DummyEnv(high_obs_space, high_act_space)
        high_sac = SAC(env=high_dummy, **high_sac_kwargs)
        self.high_agent = SB3AgentWrapper(
            high_sac, 
            config.train_freq, 
            config.gradient_steps_high, 
            config.batch_size
        )
        low_dummy = DummyEnv(low_obs_space, low_act_space)
        low_sac = SAC(env=low_dummy, **low_sac_kwargs)
        self.low_agent = SB3AgentWrapper(
            low_sac, 
            config.train_freq, 
            config.gradient_steps_low, 
            config.batch_size
        )

        # === logger ===
        self.high_logger = configure_logger(high_sac.verbose, high_sac_kwargs.get("tensorboard_log"), "hiro_high", True)
        self.low_logger = configure_logger(low_sac.verbose, low_sac_kwargs.get("tensorboard_log"), "hiro_low", True)
        self.high_agent.set_logger(self.high_logger)
        self.low_agent.set_logger(self.low_logger)

    # ------------------------------------------------------------------
    # SB3 Callback 兼容接口：让 BaseCallback 可以把 HIROSAC 当作 BaseAlgorithm 用
    # ------------------------------------------------------------------
    @property
    def num_timesteps(self) -> int:
        return self.total_timesteps

    def get_env(self) -> gym.Env:
        return self.env

    def _init_callback(
        self,
        callback,
        progress_bar: bool = False,
    ):
        if isinstance(callback, list):
            callback = CallbackList(callback)

        if callback is None:
            callback = CallbackList([])
        elif not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        if progress_bar and ProgressBarCallback is not None:
            callback = CallbackList([callback, ProgressBarCallback()])

        callback.init_callback(self)
        return callback

    
    # ------------------------------------------------------------------
    # 内部工具函数：从 obs 中解析高/低层观测
    # ------------------------------------------------------------------
    def _build_high_obs(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """与 env.obs 完全对齐"""
        obs_flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        t, kin, kin_flat = utils.split_time_kinematics(obs_flat, self.n_veh, self.feat_dim)
        return obs_flat, kin_flat, kin, float(t)

    def _build_low_obs(self, t: float, kin_flat: np.ndarray, kin: np.ndarray, goal_action: np.ndarray) -> np.ndarray:
        """
        低层观测 = flatten(kinematics) + goal_rel
        其中 goal_action 是高层 SAC 的动作向量 a=[Δx, y_code, vx_target]，
        goal_rel = goal_abs - ego_abs，y 维在这里按照 y_code 映射到相邻车道中心线。
        """
        t_arr = np.array([t], dtype=np.float32)
        kin_flat = np.asarray(kin_flat, dtype=np.float32)
        goal_action = np.asarray(goal_action, dtype=np.float32).reshape(-1)
        ego_sub = utils.extract_ego_substate(kin, self.ego_feature_idx).astype(np.float32)
        goal_phys = utils.goal_action_to_abs(ego_sub, goal_action, self.lane_center_ys)
        goal_rel = (goal_phys - ego_sub).astype(np.float32)     # 相对目标：goal_rel = goal_abs - ego_sub

        return np.concatenate([t_arr, kin_flat, goal_rel], axis=0)


    # ------------------------------------------------------------------
    # 核心逻辑
    # ------------------------------------------------------------------
    def learn(self, total_timesteps: int, callback=None, log_interval=1, progress_bar=False):
        callback = self._init_callback(callback, progress_bar=progress_bar)
        env = self.env
        obs, _ = env.reset()
        done, truncated = False, False

        callback.on_training_start(locals(), globals())

        while self.total_timesteps < total_timesteps:
            # === 1. High Level Decision ===
            high_obs, kin_flat, kin, t = self._build_high_obs(obs)
            goal_action, goal_buffer_action = self.high_agent.sample_action(high_obs)

            # 缓存开始状态，用于计算 intrinsic reward
            ego_start = utils.extract_ego_substate(kin, self.ego_feature_idx).astype(np.float32)

            # === 2. Low Level Rollout (Interval) ===
            high_ret = 0.0
            low_ret = 0.0
            low_len = 0
            low_comp_sums = {}
            continue_training = True

            callback.on_rollout_start()

            for c in range(self.cfg.high_interval):
                low_obs = self._build_low_obs(t, kin_flat, kin, goal_action)
                low_action, low_buffer_action = self.low_agent.sample_action(low_obs)
                
                next_obs, reward, done, truncated, info = env.step(low_action)
                _, kin_flat_next, kin_next, t_next = self._build_high_obs(next_obs)
                
                self.total_timesteps += self.n_envs
                self.low_agent.num_timesteps += self.n_envs
                
                # --- Reward Calculation ---
                reward_env = float(reward)
                high_ret += reward_env  # 高层累积环境原始奖励

                r_components = info.get("reward_components", {})
                punctual_contrib = float(r_components.get("punctual_reward", 0.0))
                low_reward_ext = reward_env - punctual_contrib
                for name, val in r_components.items():
                    low_comp_sums[name] = low_comp_sums.get(name, 0.0) + float(val)     # 用于callback统计
                
                intrinsic = 0.0
                terminated = bool(done or truncated)
                is_last_step = (c == self.cfg.high_interval - 1) or terminated      # 目前采用sparse reward
                if is_last_step:
                    ego_next_sub_rel = utils.extract_ego_substate(kin_next, self.ego_feature_idx) - ego_start
                    goal_rel = utils.goal_action_to_abs(ego_start, goal_action, self.lane_center_ys) - ego_start
                    intrinsic = utils.intrinsic_reward_l2(
                        ego_next_sub_rel=ego_next_sub_rel,
                        goal_rel=goal_rel,
                        norm_ranges=[[0, 37.5], [-8, 8], [-8, 8], [-2, 2]], # 根据实际修改
                        coef=self.cfg.intrinsic_coef,
                        weights=[1, 2, 8, 1],
                    )
                
                low_reward_total = low_reward_ext + intrinsic
                low_ret += low_reward_total
                low_len += 1
                
                # --- Store & Train Low Level ---
                next_low_obs = self._build_low_obs(t_next, kin_flat_next, kin_next, goal_action)
                
                self.low_agent.store_transition(
                    obs=low_obs,
                    action=low_buffer_action, # 存 buffer action
                    next_obs=next_low_obs,
                    reward=low_reward_total,
                    done=terminated,
                    info=info
                )
                self.low_agent.train_if_needed()

                # --- Callback & Loop Control ---
                episode_end = terminated
                callback.update_locals(locals())
                if callback.on_step() is False:
                    continue_training = False
                    break
                
                obs = next_obs
                kin_flat, kin, t = kin_flat_next, kin_next, t_next

                if terminated:
                    break

            callback.on_rollout_end()

            if not continue_training:
                break

            # === 3. High Level Store & Train ===
            next_high_obs, _, _, _ = self._build_high_obs(obs)
            terminated_high = bool(done or truncated)
            
            self.high_agent.num_timesteps += 1
            self.high_agent.store_transition(
                obs=high_obs,
                action=goal_buffer_action,
                next_obs=next_high_obs,
                reward=high_ret,
                done=terminated_high,
                info={"high_interval_len": low_len}
            )
            self.high_agent.train_if_needed()

            # 4) 若 episode 结束：reset 环境 & episode 级日志累计量
            if done or truncated:
                obs, _ = env.reset()
                done, truncated = False, False

        callback.on_training_end()

        print(f"[HIROSAC] 训练结束: env_steps={self.total_timesteps}")
        return self
