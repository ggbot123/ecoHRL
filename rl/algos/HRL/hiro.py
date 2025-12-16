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


@dataclass
class HIROConfig:
    high_interval: int = 25          # 高层每 high_interval 个 env.step 决策一次
    gamma_high: float = 0.99
    gamma_low: float = 0.99
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 10_000
    gradient_steps_high: int = 1
    gradient_steps_low: int = 1
    train_freq: int = 1              # 每个 env.step 都尝试更新一次
    intrinsic_coef: float = 1.0      # 末状态距离 goal 的 intrinsic reward 系数
    device: str = "auto"


class HIROSAC:
    def __init__(
        self,
        env: gym.Env,
        high_sac_kwargs: Dict[str, Any],
        low_sac_kwargs: Dict[str, Any],
        config: HIROConfig,
    ):
        """
        :param env: 真实环境（MultiLaneEnv），低层直接与它交互
        :param high_sac_kwargs: 传给高层 SAC 的参数（除 env 外）
        :param low_sac_kwargs: 传给低层 SAC 的参数（除 env 外）
        :param config: HIRO 的超参数
        """
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

        # ----- 定义高层 / 低层的 spaces（用于 DummyEnv） -----  #
        high_obs_dim = self.kin_flat_dim + 1
        high_obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(high_obs_dim,),
            dtype=np.float32,
        )
        t_h: float = float(self.cfg.high_interval) * self.dt
        goal_low = np.array([self.v_min * t_h, -1, self.v_min], dtype=np.float32)
        goal_high = np.array([self.v_max * t_h, 1, self.v_max], dtype=np.float32)
        high_act_space = gym.spaces.Box(
            low=goal_low,
            high=goal_high,
            dtype=np.float32,
        )
        low_obs_dim = self.kin_flat_dim + self.ego_dim
        low_obs_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(low_obs_dim,),
            dtype=np.float32,
        )
        low_act_space: gym.spaces.Box = env.action_space

        # ----- 创建 SAC 实例（内部是 SB3 的 OffPolicyAlgorithm 派生类） -----
        high_dummy_env = DummyEnv(high_obs_space, high_act_space)
        low_dummy_env = DummyEnv(low_obs_space, low_act_space)
        self.high_agent = SAC(env=high_dummy_env, **high_sac_kwargs)
        self.low_agent = SAC(env=low_dummy_env, **low_sac_kwargs)

        # === 缓存上下层 logger，供回调使用 ===
        self.high_logger = configure_logger(
            self.high_agent.verbose,
            high_sac_kwargs.get("tensorboard_log", None),
            tb_log_name="hiro_high",
            reset_num_timesteps=True,
        )
        self.low_logger = configure_logger(
            self.low_agent.verbose,
            low_sac_kwargs.get("tensorboard_log", None),
            tb_log_name="hiro_low",
            reset_num_timesteps=True,
        )
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

    # 复用 SB3 BaseAlgorithm._init_callback 的逻辑
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
        t, kin, kin_flat = utils.split_time_kinematics(
            obs_flat, self.n_veh, self.feat_dim
        )
        return obs_flat, kin_flat, kin, float(t)

    def _build_low_obs(self, kin_flat: np.ndarray, kin: np.ndarray, goal_action: np.ndarray) -> np.ndarray:
        """
        低层观测 = flatten(kinematics) + goal_rel
        其中 goal_action 是高层 SAC 的动作向量 a=[Δx, y_code, vx_target, _]，
        goal_rel = goal_abs - ego_abs，y 维在这里按照 y_code 映射到相邻车道中心线。
        """
        kin_flat = np.asarray(kin_flat, dtype=np.float32)
        goal_action = np.asarray(goal_action, dtype=np.float32).reshape(-1)
        ego_sub = utils.extract_ego_substate(kin, self.ego_feature_idx).astype(np.float32)
        goal_phys = utils.goal_action_to_abs(ego_sub, goal_action, self.lane_center_ys)

        # 相对目标：goal_rel = goal_abs - ego_sub
        goal_rel = (goal_phys - ego_sub).astype(np.float32)

        return np.concatenate([kin_flat, goal_rel], axis=0)
    
    # ---------------------------------------------------------
    # 高层：一次决策的采样（给出 goal_phys）
    # ---------------------------------------------------------
    def _sample_high_action(
        self,
        obs: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        
        high_obs, kin_flat, kin, t = self._build_high_obs(obs)
        high_obs_vec = high_obs[None, :]
        self.high_agent._last_obs = high_obs_vec
        self.high_agent._last_original_obs = high_obs_vec

        # TODO achievable & constrained goal space
        goal_action_vec, goal_buffer_action = self.high_agent._sample_action(
            learning_starts=self.high_agent.learning_starts,
            action_noise=self.high_agent.action_noise,
        )
        goal_action = np.asarray(goal_action_vec[0], dtype=np.float32)

        return high_obs_vec, kin_flat, kin, float(t), goal_action, goal_buffer_action

    # ---------------------------------------------------------
    # 低层：一个 high_interval 内的 rollouts + 写低层 buffer + 训练低层
    # ---------------------------------------------------------
    def _run_low_level_interval(
        self,
        obs: np.ndarray,
        kin_flat: np.ndarray,
        kin: np.ndarray,
        goal_action: np.ndarray,
        callback: BaseCallback,
        log_interval: int,
    ) -> Tuple[float, float, int, Dict[str, float], bool, np.ndarray, bool, bool]:

        assert isinstance(callback, BaseCallback)
        env = self.env
        high_interval = self.cfg.high_interval
        high_ret: float = 0.0
        low_ret: float = 0.0
        low_len: int = 0
        low_comp_sums: Dict[str, float] = {}
        continue_training: bool = True

        ego_start = utils.extract_ego_substate(kin, self.ego_feature_idx).astype(np.float32)

        callback.on_rollout_start()

        for c in range(high_interval):
            # ---------- 低层观测 + 采样 ----------
            low_obs = self._build_low_obs(kin_flat, kin, goal_action)
            low_obs_vec = low_obs[None, :]

            self.low_agent._last_obs = low_obs_vec
            self.low_agent._last_original_obs = low_obs_vec
            low_action_vec, low_buffer_action = self.low_agent._sample_action(
                learning_starts=self.low_agent.learning_starts,
                action_noise=self.low_agent.action_noise,
            )
            low_action = np.asarray(low_action_vec[0])

            # ---------- 与环境交互 ----------
            next_obs, reward, done, truncated, info = env.step(low_action)
            _, kin_flat_next, kin_next, _ = self._build_high_obs(next_obs)

            step_inc = self.n_envs
            self.total_timesteps += step_inc
            self.low_agent.num_timesteps += step_inc
            
            # ---------- 计算并统计低层 reward：去掉准时性 + intrinsic ----------
            reward_env = float(reward)
            r_components = info.get("reward_components", {})
            punctual_contrib = float(r_components.get("punctual_reward", 0.0))
            low_reward_ext = reward_env - punctual_contrib
            intrinsic = 0.0
            terminated = bool(done or truncated)
            is_last_low_step = (c == high_interval - 1) or terminated
            if is_last_low_step:
                ego_next_sub = utils.extract_ego_substate(
                    kin_next, self.ego_feature_idx
                )
                goal_phys = utils.goal_action_to_abs(ego_start, goal_action, self.lane_center_ys)
                intrinsic = utils.intrinsic_reward_l2(
                    ego_next_sub_phys=ego_next_sub,
                    goal_phys=goal_phys,
                    norm_ranges=[[-2, 2], [-1, 1], [-0.2, 0.2], [-0.1, 0.1]],
                    coef=self.cfg.intrinsic_coef,
                    weights=None,
                )
            low_reward_total = low_reward_ext + intrinsic
            low_ret += low_reward_total
            low_len += 1
            for name, val in r_components.items():
                low_comp_sums[name] = low_comp_sums.get(name, 0.0) + float(val)
            low_comp_sums["intrinsic_reward"] = low_comp_sums.get("intrinsic_reward", 0.0) + intrinsic

            # ---------- 写低层 replay buffer ----------
            next_low_obs = self._build_low_obs(kin_flat_next, kin_next, goal_action)
            next_low_obs_vec = next_low_obs[None, :]
            reward_arr = np.array([low_reward_total], dtype=np.float32)
            done_arr = np.array([terminated], dtype=np.bool_)
            infos_list = [info]
            self.low_agent._store_transition(
                replay_buffer=self.low_agent.replay_buffer,
                buffer_action=low_buffer_action,
                new_obs=next_low_obs_vec,
                reward=reward_arr,
                dones=done_arr,
                infos=infos_list,
            )
            # 累计高层 reward
            high_ret += reward_env

            # ---------- 训练低层策略 ----------
            if (
                self.total_timesteps > self.cfg.learning_starts
                and self.total_timesteps % self.cfg.train_freq == 0
            ):
                self.low_agent.train(
                    gradient_steps=self.cfg.gradient_steps_low,
                    batch_size=self.cfg.batch_size,
                )

            # callback.on_step
            episode_end = terminated
            callback.update_locals(locals())
            if callback.on_step() is False:
                continue_training = False
                obs = next_obs
                break

            # 准备下一步
            obs = next_obs
            kin_flat, kin = kin_flat_next, kin_next
            if terminated:
                break

        # interval 结束，返回最终状态
        return high_ret, low_ret, low_len, low_comp_sums, continue_training, obs, bool(done), bool(truncated)

    # ---------------------------------------------------------
    # 高层：写 replay buffer + 训练
    # ---------------------------------------------------------
    def _update_high_level(
        self,
        goal_buffer_action: np.ndarray,
        high_ret: float,
        low_len: int,
        obs: np.ndarray,
        done: bool,
        truncated: bool,
    ) -> None:
        """
        将一个 high_interval 汇总成一条高层 transition，写入高层 replay buffer，并按需要训练高层。
        """
        next_high_obs, _, _, _ = self._build_high_obs(obs)
        next_high_obs_vec = next_high_obs[None, :]
        terminated = bool(done or truncated)
        high_reward_arr = np.array([high_ret], dtype=np.float32)
        high_done_arr = np.array([terminated], dtype=np.bool_)
        high_infos_list = [
            {"high_interval": low_len, "env_steps": self.total_timesteps}
        ]
        self.high_agent.num_timesteps += 1
        
        # 写入replay buffer
        self.high_agent._store_transition(
            replay_buffer=self.high_agent.replay_buffer,
            buffer_action=goal_buffer_action,
            new_obs=next_high_obs_vec,
            reward=high_reward_arr,
            dones=high_done_arr,
            infos=high_infos_list,
        )

        # 训练高层策略
        if (self.total_timesteps > self.cfg.learning_starts and self.total_timesteps % self.cfg.train_freq == 0):
            self.high_agent.train(
                gradient_steps=self.cfg.gradient_steps_high,
                batch_size=self.cfg.batch_size,
            )


    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 1,
        progress_bar: bool = False,
    ) -> "HIROSAC":
        """
        :param total_timesteps: 以 env.step 计的总训练步数（低层）
        :param callback: SB3 风格的 MaybeCallback（BaseCallback / list / 函数 / None）
        :param log_interval: 暂未使用，保留接口与 SB3 一致
        :param progress_bar: 若为 True，则自动附加 ProgressBarCallback
        """
        callback = self._init_callback(callback, progress_bar=progress_bar)
        env = self.env
        obs, _ = env.reset()
        done = False
        truncated = False

        callback.on_training_start(locals(), globals())

        while self.total_timesteps < total_timesteps:
            # 1) 高层观测 + 采样（得到 goal_phys）
            high_obs_vec, kin_flat, kin, t, goal_action, goal_buffer_action = self._sample_high_action(obs)

            # 2) 低层 interval：rollout + 低层训练
            high_ret, low_ret, low_len, low_comp_sums, continue_training, obs, done, truncated = self._run_low_level_interval(obs, kin_flat, kin, goal_action, callback, log_interval)
            callback.update_locals(
                dict(
                    low_ret=low_ret,
                    low_len=low_len,
                    low_comp_sums=low_comp_sums,
                )
            )
            callback.on_rollout_end()
            if not continue_training:
                break

            # 3) 高层：存一条 transition，并按需要训练高层
            self._update_high_level(goal_buffer_action, high_ret, low_len, obs, done, truncated)

            # 4) 若 episode 结束：reset 环境 & episode 级日志累计量
            if done or truncated:
                obs, _ = env.reset()
                done, truncated = False, False

        callback.on_training_end()

        print(f"[HIROSAC] 训练结束: env_steps={self.total_timesteps}")
        return self
