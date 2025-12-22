# rl/algos/hiro/hiro.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import gymnasium as gym
import numpy as np

from rl.algos.sac.sac import SAC
from rl.utils import utils
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import get_device, configure_logger
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ConvertCallback, ProgressBarCallback
from stable_baselines3.common.vec_env import DummyVecEnv


class DummyEnv(gym.Env):
    """Minimal gymnasium Env used only for building SB3 off-policy agents."""
    def __init__(self, observation_space: gym.spaces.Box, action_space: gym.spaces.Box):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        return obs, {}

    def step(self, action):
        raise RuntimeError("DummyEnv.step() was called. This env is only for building SB3 agents; stepping it indicates a bug.")


def _make_dummy_vec_env(obs_space: gym.spaces.Box, act_space: gym.spaces.Box, n_envs: int) -> DummyVecEnv:
    return DummyVecEnv([(lambda: DummyEnv(obs_space, act_space)) for _ in range(int(n_envs))])


class SB3AgentWrapper:
    """Thin wrapper exposing SB3 private APIs (_sample_action/_store_transition) in batch form."""
    
    def __init__(self, agent: SAC, config_train_freq: int, gradient_steps: int, batch_size: int):
        self.agent = agent
        self.train_freq = int(config_train_freq)
        self.gradient_steps = int(gradient_steps)
        self.batch_size = int(batch_size)
        self.replay_buffer = agent.replay_buffer
        self._last_train_step = 0
    
    @property
    def num_timesteps(self) -> int:
        return int(self.agent.num_timesteps)
    
    @num_timesteps.setter
    def num_timesteps(self, value: int):
        self.agent.num_timesteps = int(value)

    def set_logger(self, logger):
        self.agent.set_logger(logger)

    def sample_action(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(obs, dtype=np.float32)
        n = int(obs.shape[0])

        if int(getattr(self.agent, "n_envs", n)) == 1 and n > 1:
            act_dim = int(self.agent.action_space.shape[0])
            action = np.empty((n, act_dim), dtype=np.float32)
            buffer_action = np.empty_like(action)
            for i in range(n):
                self.agent._last_obs = obs[i:i + 1]
                a, a_buf = self.agent._sample_action(learning_starts=self.agent.learning_starts, action_noise=self.agent.action_noise, n_envs=1)
                action[i] = a[0]; buffer_action[i] = a_buf[0]
            return action, buffer_action

        self.agent._last_obs = obs
        return self.agent._sample_action(
            learning_starts=self.agent.learning_starts,
            action_noise=self.agent.action_noise,
            n_envs=n,
        )

    def store_transition(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, reward: np.ndarray, done: np.ndarray, infos: List[Dict[str, Any]]):
        self.agent._last_obs = obs
        self.agent._store_transition(self.replay_buffer, action, next_obs, reward, done, infos)

    def train_if_needed(self):
        if self.num_timesteps <= self.agent.learning_starts:
            return
        if self.num_timesteps // self.train_freq > self._last_train_step // self.train_freq:
            self.agent.train(gradient_steps=self.gradient_steps, batch_size=self.batch_size)
            self._last_train_step = self.num_timesteps
    
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
    def __init__(self, env, high_sac_kwargs: Dict[str, Any], low_sac_kwargs: Dict[str, Any], config: HIROConfig):
        self.env = env
        self.cfg = config
        self.device = get_device(config.device)
        self.total_timesteps = 0
        self.n_envs = int(env.num_envs)

        # ---- 用一次 reset 的 obs 初始化 Kinematics 元信息 ----
        obs0 = env.reset()
        obs0_flat = np.asarray(obs0[0], dtype=np.float32).reshape(-1)
        keep_features = ("x", "y", "vx", "vy")  # ego 子状态中参与 HIRO goal 的特征
        (
            self.n_veh,
            self.feat_dim,
            self.feature_names,
            self.ego_feature_idx,
        ) = utils.init_kinematics_meta(env, obs0_flat, keep_features)
        self.kin_flat_dim = int(self.n_veh * self.feat_dim)
        self.ego_dim = len(self.ego_feature_idx)

        # ---- 从env中获取的必要变量 --- #
        env_cfg = env.get_attr("config", indices=0)[0]
        self.v_min, self.v_max = 0.0, float(env_cfg["speed_limit"])
        self.dt = 1.0 / float(env_cfg["policy_frequency"])
        lanes = int(env_cfg["lanes_count"])
        lane_w = float(env_cfg.get("lane_width", 4.0))
        self.lane_center_ys = (np.arange(lanes, dtype=np.float32) * lane_w).astype(np.float32)

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
        high_sac = SAC(env=_make_dummy_vec_env(high_obs_space, high_act_space, 1), **high_sac_kwargs)
        low_sac = SAC(env=_make_dummy_vec_env(low_obs_space, low_act_space, self.n_envs), **low_sac_kwargs)
        self.high_agent = SB3AgentWrapper(high_sac, config.train_freq, config.gradient_steps_high, config.batch_size)
        self.low_agent = SB3AgentWrapper(low_sac, config.train_freq, config.gradient_steps_low, config.batch_size)

        # ----- logger ----- #
        self.high_logger = configure_logger(high_sac.verbose, high_sac_kwargs.get("tensorboard_log"), "hiro_high", True)
        self.low_logger = configure_logger(low_sac.verbose, low_sac_kwargs.get("tensorboard_log"), "hiro_low", True)
        self.high_agent.set_logger(self.high_logger)
        self.low_agent.set_logger(self.low_logger)

        self._intrinsic_norm_ranges = np.asarray([[0.0, 37.5], [-8.0, 8.0], [-8.0, 8.0], [-2.0, 2.0]], dtype=np.float32)
        self._intrinsic_weights = np.asarray([1.0, 2.0, 8.0, 1.0], dtype=np.float32)

    # ------------------------------------------------------------------
    # SB3 Callback 兼容接口：让 BaseCallback 可以把 HIROSAC 当作 BaseAlgorithm 用
    # ------------------------------------------------------------------
    @property
    def num_timesteps(self) -> int:
        return self.total_timesteps

    def get_env(self) -> gym.Env:
        return self.env

    def _init_callback(self, callback, progress_bar: bool = False):
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
    # 内部工具函数：obs 处理
    # ------------------------------------------------------------------
    def _build_high_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def _build_low_obs(self, t_rel: float, kin_flat: np.ndarray, kin: np.ndarray, goal_phys: np.ndarray) -> np.ndarray:
        """
        低层观测 = t_norm + kin_flat + goal_rel
        接收绝对坐标系 goal_phys，根据当前 ego 状态计算 goal_rel
        """
        t_norm = (np.asarray(t_rel, dtype=np.float32) / float(self.cfg.high_interval)).reshape(-1, 1)
        ego_sub = utils.extract_ego_substate(kin, self.ego_feature_idx)
        goal_rel = (np.asarray(goal_phys, dtype=np.float32) - ego_sub).astype(np.float32)
        return np.concatenate([t_norm, np.asarray(kin_flat, dtype=np.float32), goal_rel], axis=1)
    
    @staticmethod
    def _terminal_obs(next_obs: np.ndarray, dones: np.ndarray, infos: List[Dict[str, Any]]) -> np.ndarray:
        if not np.any(dones):
            return next_obs
        term = np.array(next_obs, copy=True)
        for i in np.flatnonzero(dones):
            tobs = infos[i].get("terminal_observation")
            if tobs is not None:
                term[i] = tobs
        return term


    def learn(self, total_timesteps: int, callback=None, log_interval: int = 1, progress_bar: bool = False):

        # ========== 0. initialization ========== #
        callback = self._init_callback(callback, progress_bar=progress_bar)
        env = self.env
        obs = env.reset()
        done, truncated = False, False

        n_envs = self.n_envs
        hi = int(self.cfg.high_interval)

        need_high = np.ones(n_envs, dtype=bool)
        c = np.zeros(n_envs, dtype=np.int32)

        high_obs_start = np.zeros_like(obs, dtype=np.float32)
        goal_action = np.zeros((n_envs, int(self.high_agent.action_space.shape[0])), dtype=np.float32)
        goal_buffer_action = np.zeros_like(goal_action)
        goal_phys = np.zeros((n_envs, self.ego_dim), dtype=np.float32)
        ego_start = np.zeros((n_envs, self.ego_dim), dtype=np.float32)

        high_ret = np.zeros(n_envs, dtype=np.float32)
        low_ret = np.zeros(n_envs, dtype=np.float32)
        low_len = np.zeros(n_envs, dtype=np.int32)
        low_comp_sums: dict[str, np.ndarray] = {}

        callback.on_training_start(locals(), globals())

        # ========== 1. Main Loop ========== #
        while self.total_timesteps < total_timesteps:

            # === 1.1 High Level Decision ===
            high_obs = self._build_high_obs(obs)
            _, kin, kin_flat = utils.split_time_kinematics(high_obs, self.n_veh, self.feat_dim)

            # step a high interval for required envs
            if need_high.any():
                idx = np.flatnonzero(need_high)
                a, a_buf = self.high_agent.sample_action(high_obs[idx])

                high_obs_start[idx] = high_obs[idx]
                goal_action[idx] = a
                goal_buffer_action[idx] = a_buf

                ego_sub = utils.extract_ego_substate(kin[idx], self.ego_feature_idx)
                ego_start[idx] = ego_sub
                goal_phys[idx] = utils.goal_action_to_abs(ego_sub, a, self.lane_center_ys)

                high_ret[idx] = 0.0
                low_ret[idx] = 0.0
                low_len[idx] = 0
                for v in low_comp_sums.values():
                    v[idx] = 0.0
                
                c[idx] = 0
                need_high[idx] = False

            # === 1.2 Low Level Decision ===
            low_obs = self._build_low_obs(c, kin_flat, kin, goal_phys)
            low_action, low_buffer_action = self.low_agent.sample_action(low_obs)

            next_obs, reward, done, infos = env.step(low_action)
            reward = np.asarray(reward, dtype=np.float32)
            done = np.asarray(done, dtype=bool)

            next_obs_tr = self._terminal_obs(next_obs, done, infos)
            next_high_obs = self._build_high_obs(next_obs_tr)
            _, kin_next, kin_flat_next = utils.split_time_kinematics(next_high_obs, self.n_veh, self.feat_dim)


            # === 1.3 Calculate Rewards ===
            high_ret += reward

            r_components = [info.get("reward_components", {}) for info in infos]
            punctual = np.asarray([rc.get("punctual_reward", 0.0) for rc in r_components], dtype=np.float32)
            low_reward_ext = reward - punctual

            # calculate intrinsic reward at last step for required envs
            is_last_step = (c == hi - 1) | done
            intrinsic = np.zeros(n_envs, dtype=np.float32)
            if is_last_step.any():
                idx_last = np.flatnonzero(is_last_step)
                ego_next_rel = utils.extract_ego_substate(kin_next[idx_last], self.ego_feature_idx) - ego_start[idx_last]
                goal_rel = goal_phys[idx_last] - ego_start[idx_last]

                intrinsic[idx_last] = utils.intrinsic_reward_l2(ego_next_rel, goal_rel, self._intrinsic_norm_ranges, self.cfg.intrinsic_coef, self._intrinsic_weights)

            low_reward_total = low_reward_ext + intrinsic
            low_ret += low_reward_total
            low_len += 1
            
            # record logs in callbacks
            for i, rc in enumerate(r_components):
                for name, val in rc.items():
                    if name == "punctual_reward":
                        continue
                    low_comp_sums.setdefault(name, np.zeros(n_envs, dtype=np.float32))[i] += float(val)
            if is_last_step.any():
                low_comp_sums.setdefault("intrinsic_reward", np.zeros(n_envs, dtype=np.float32))[is_last_step] += intrinsic[is_last_step]
            
            # === 1.4 Low Level Store & Train ===
            next_low_obs = self._build_low_obs(c + 1, kin_flat_next, kin_next, goal_phys)
            done_low = is_last_step.astype(np.bool_)

            low_infos = infos
            if done_low.any():
                low_infos = [dict(info) for info in infos]
                for i in np.flatnonzero(done_low):
                    low_infos[i]["terminal_observation"] = next_low_obs[i]
                
            self.low_agent.store_transition(
                low_obs, 
                low_buffer_action, 
                next_low_obs, 
                low_reward_total.astype(np.float32), 
                done_low, 
                low_infos
            )
            self.total_timesteps += n_envs
            self.low_agent.num_timesteps += n_envs

            # train lower model
            self.low_agent.train_if_needed()

            reward_env = reward
            episode_end = done
            callback.update_locals(locals())
            if callback.on_step() is False:
                break

            # === 1.5. High Level Store & Train ===
            if done_low.any():
                idx_end = np.flatnonzero(done_low)
                low_ret_end = low_ret[idx_end].copy()
                low_len_end = low_len[idx_end].copy()
                low_comp_end = {k: v[idx_end].copy() for k, v in low_comp_sums.items()}

                callback.update_locals({**locals(), "low_ret": low_ret_end, "low_len": low_len_end, "low_comp_sums": low_comp_end})
                callback.on_rollout_end()

                self.high_agent.num_timesteps += int(idx_end.size)
                for j in idx_end:
                    info_h = dict(infos[j]); info_h["high_interval_len"] = int(low_len[j])
                    self.high_agent.store_transition(
                        high_obs_start[j:j + 1], 
                        goal_buffer_action[j:j + 1], 
                        next_high_obs[j:j + 1], 
                        np.asarray([high_ret[j]], dtype=np.float32), 
                        np.asarray([done[j]], dtype=np.bool_), 
                        [info_h]
                    )

                # train higher model
                self.high_agent.train_if_needed()

                need_high[idx_end] = True
                high_ret[idx_end] = 0.0
                low_ret[idx_end] = 0.0
                low_len[idx_end] = 0
                for v in low_comp_sums.values():
                    v[idx_end] = 0.0

                c[idx_end] = 0

            c[~done_low] += 1
            obs = next_obs

        callback.on_training_end()
        print(f"[HIROSAC] 训练结束: env_steps={self.total_timesteps}")
        return self
