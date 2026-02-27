# rl/algos/hiro/hiro.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional, Callable

import gymnasium as gym
import numpy as np

from rl.algos.sac.sac import SAC
from rl.algos.sac.safety_sac import SafetyLayerSAC
from rl.utils import utils
from rl.algos.HRL.rule_based import RuleBasedAgentWrapper
from rl.algos.HRL.goal_samplers import GoalSamplerConfig
from rl.algos.HRL.low_her_buffer import HiROLowHERReplayBuffer
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
                a, a_buf = self.agent._sample_action(
                    learning_starts=self.agent.learning_starts,
                    action_noise=self.agent.action_noise,
                    n_envs=1,
                )
                action[i] = a[0]
                buffer_action[i] = a_buf[0]
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
    intrinsic_coef: float      # 末状态距离 goal 的 intrinsic reward 系数
    device: str
    use_off_policy_correction: bool
    intrinsic_norm_ranges: Optional[np.ndarray | List[List[float]]]
    intrinsic_weights: Optional[np.ndarray | List[float]]
    intrinsic_type: str # "l2" | "huber_shaping"
    train_mode: str  # "joint", "low_only", "high_only"
    goal_sampler: GoalSamplerConfig
    low_level_type: str = "sac" # "sac" or "rule_based" or "mpc"
    low_sac_impl: str = "auto" # "auto" | "sac" | "safety_sac"
    low_use_her: bool = False
    low_her_ratio: float = 0.8
    low_her_strategy: str = "future"  # "future" | "final"
    high_pretrained_path: Optional[str] = None
    low_pretrained_path: Optional[str] = None
    mask_ego_position_in_low_obs: bool = False
    use_low_safety_layer: bool = False


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
        (self.n_veh, self.n_veh_local, self.feat_dim, self.feature_names, self.ego_feature_idx) = \
            utils.init_kinematics_meta(env, obs0_flat, keep_features)
        self.kin_flat_dim = int(self.n_veh * self.feat_dim)
        self.local_kin_flat_dim = int(self.n_veh_local * self.feat_dim)
        self.ego_dim = len(self.ego_feature_idx)
        self._intrinsic_norm_ranges = np.asarray(self.cfg.intrinsic_norm_ranges, dtype=np.float32)
        w = getattr(self.cfg, "intrinsic_weights", None)
        self._intrinsic_weights = None if w is None else np.asarray(w, dtype=np.float32)

        # ---- 从env中获取的必要变量 --- #
        env_cfg = env.get_attr("config", indices=0)[0]
        # self.v_min, self.v_max = 8.0, float(env_cfg["speed_limit"])
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

        low_obs_dim = self.local_kin_flat_dim + self.ego_dim + 1
        low_obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(low_obs_dim,), dtype=np.float32)
        low_act_space = env.action_space

        # ----- Build low-level agent ----- #
        self.train_mode = str(getattr(config, "train_mode", "joint")).lower()
        self.low_level_type = str(getattr(config, "low_level_type", "sac")).lower()
        self.low_sac_impl = str(getattr(config, "low_sac_impl", "auto")).lower()
        self.low_use_her = bool(getattr(self.cfg, "low_use_her", False))
        self.use_off_policy_correction = bool(getattr(self.cfg, "use_off_policy_correction", False))
        self.low_gamma = float(low_sac_kwargs.get("gamma", 0.99))
        if self.low_level_type == "rule_based":
            self.low_agent = RuleBasedAgentWrapper(env, self.n_envs, high_interval=int(config.high_interval))
        elif self.low_level_type == "sac":
            if self.low_sac_impl not in {"auto", "sac", "safety_sac"}:
                raise ValueError(f"Unknown low_sac_impl: {self.low_sac_impl}")

            use_low_safety_layer = bool(getattr(self.cfg, "use_low_safety_layer", False))
            use_safety_sac = use_low_safety_layer if self.low_sac_impl == "auto" else (self.low_sac_impl == "safety_sac")
            if use_safety_sac and not use_low_safety_layer:
                raise ValueError("low_sac_impl='safety_sac' requires use_low_safety_layer=True")

            low_sac_cls = SafetyLayerSAC if use_safety_sac else SAC

            # Resolve scaled auto target entropy for low-level SAC: -scale * action_dim.
            low_sac_kwargs = dict(low_sac_kwargs)

            use_low_her = bool(self.low_use_her)
            if use_low_her:
                rb_kwargs_low = dict(low_sac_kwargs.get("replay_buffer_kwargs", {}) or {})
                rb_kwargs_low.update(
                    dict(
                        feat_dim=int(self.feat_dim),
                        kin_flat_dim=int(self.local_kin_flat_dim),
                        ego_feature_idx=list(self.ego_feature_idx),
                        intrinsic_coef=float(self.cfg.intrinsic_coef),
                        intrinsic_norm_ranges=np.asarray(self.cfg.intrinsic_norm_ranges, dtype=np.float32),
                        intrinsic_weights=None
                        if getattr(self.cfg, "intrinsic_weights", None) is None
                        else np.asarray(self.cfg.intrinsic_weights, dtype=np.float32),
                        intrinsic_type=str(getattr(self.cfg, "intrinsic_type", "l2")),
                        low_gamma=float(self.low_gamma),
                        her_ratio=float(getattr(self.cfg, "low_her_ratio", 0.8)),
                        her_strategy=str(getattr(self.cfg, "low_her_strategy", "future")),
                        enable_her=True,
                    )
                )
                low_sac_kwargs["replay_buffer_class"] = HiROLowHERReplayBuffer
                low_sac_kwargs["replay_buffer_kwargs"] = rb_kwargs_low

            target_entropy = low_sac_kwargs.get("target_entropy", "auto")
            target_entropy_scale = low_sac_kwargs.pop("target_entropy_scale", None)
            if isinstance(target_entropy, str) and target_entropy == "auto" and target_entropy_scale is not None:
                act_dim = float(np.prod(low_act_space.shape))
                low_sac_kwargs["target_entropy"] = float(-float(target_entropy_scale) * act_dim)
            low_pretrained_path = getattr(self.cfg, "low_pretrained_path", None)
            if low_pretrained_path:
                if not os.path.isfile(low_pretrained_path):
                    raise FileNotFoundError(f"Low-level pretrained model not found: {low_pretrained_path}")
                print(f"[HIRO] Load low-level pretrained model from: {low_pretrained_path}")
                low_sac = low_sac_cls.load(
                    low_pretrained_path,
                    env=_make_dummy_vec_env(low_obs_space, low_act_space, self.n_envs),
                    device=self.device,
                    **low_sac_kwargs,
                )
            else:
                low_sac = low_sac_cls(env=_make_dummy_vec_env(low_obs_space, low_act_space, self.n_envs), **low_sac_kwargs)
            self.low_agent = SB3AgentWrapper(low_sac, config.train_freq, config.gradient_steps_low, config.batch_size)
            if use_low_safety_layer:
                self.low_safety = RuleBasedAgentWrapper(env, self.n_envs, high_interval=int(config.high_interval))
                if use_safety_sac:
                    low_sac.target_action_filter = self._low_target_action_filter
        else:
            raise ValueError(f"Unknown low_level_type: {self.low_level_type}")

        # ----- High-level buffer config (dynamic OPC metadata only) ----- #
        high_sac_kwargs = dict(high_sac_kwargs)
        if self.use_off_policy_correction:
            rb_kwargs = dict(high_sac_kwargs.get("replay_buffer_kwargs", {}) or {})
            rb_kwargs.update(
                dict(
                    max_seq_len=int(self.cfg.high_interval),
                    kin_flat_dim=int(self.local_kin_flat_dim),
                    low_action_dim=int(np.prod(low_act_space.shape)),
                    feat_dim=int(self.feat_dim),
                    ego_feature_idx=list(self.ego_feature_idx),
                    lane_center_ys=self.lane_center_ys,
                    high_interval=int(self.cfg.high_interval),
                    low_policy=self.low_agent.policy,
                )
            )
            rb_kwargs["enable_off_policy_correction"] = True
            high_sac_kwargs["replay_buffer_kwargs"] = rb_kwargs

        high_pretrained_path = getattr(self.cfg, "high_pretrained_path", None)
        if high_pretrained_path:
            if not os.path.isfile(high_pretrained_path):
                raise FileNotFoundError(f"High-level pretrained model not found: {high_pretrained_path}")
            print(f"[HIRO] Load high-level pretrained model from: {high_pretrained_path}")
            high_sac = SAC.load(
                high_pretrained_path,
                env=_make_dummy_vec_env(high_obs_space, high_act_space, 1),
                device=self.device,
                **high_sac_kwargs,
            )
        else:
            high_sac = SAC(env=_make_dummy_vec_env(high_obs_space, high_act_space, 1), **high_sac_kwargs)

        self.high_agent = SB3AgentWrapper(high_sac, config.train_freq, config.gradient_steps_high, config.batch_size)

        # ----- logger ----- #
        self.high_logger = configure_logger(high_sac.verbose, high_sac_kwargs.get("tensorboard_log"), "hiro_high", True)
        self.high_agent.set_logger(self.high_logger)
        self.low_logger = configure_logger(0, low_sac_kwargs.get("tensorboard_log"), "hiro_low", True)
        if self.low_level_type == "sac":
            self.low_agent.set_logger(self.low_logger)

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

    @staticmethod
    def _propagate_log_interval(callback, log_interval: int):
        if isinstance(callback, CallbackList):
            for cb in callback.callbacks: HIROSAC._propagate_log_interval(cb, log_interval)
        elif hasattr(callback, "log_interval"):
            callback.log_interval = int(log_interval)

    # ------------------------------------------------------------------
    # 内部工具函数：obs 处理
    # ------------------------------------------------------------------
    def _build_high_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float32)

    def _build_low_obs(self, t_rel: np.ndarray, kin_flat: np.ndarray, kin: np.ndarray, goal_phys: np.ndarray) -> np.ndarray:
        """
        低层观测 = t_norm + local_kin_flat + goal_rel
        接收绝对坐标系 goal_phys，根据当前 ego 状态计算 goal_rel
        """
        t_norm = (np.asarray(t_rel, dtype=np.float32) / float(self.cfg.high_interval)).reshape(-1, 1)
        ego_sub = utils.extract_ego_substate(kin, self.ego_feature_idx)
        goal_rel = (np.asarray(goal_phys, dtype=np.float32) - ego_sub).astype(np.float32)
        local_kin_flat = np.asarray(kin_flat[:, :self.local_kin_flat_dim], dtype=np.float32).copy()

        # Mask ego absolute position in low_obs
        mask_ego_pos = bool(getattr(self.cfg, "mask_ego_position_in_low_obs", False))
        if self.low_level_type == "rule_based" or bool(getattr(self.cfg, "use_low_safety_layer", False)):
            mask_ego_pos = False
        if mask_ego_pos:
            idx_x = int(self.feature_names.index("x"))
            idx_y = int(self.feature_names.index("y"))
            local_kin_flat[:, idx_x] = 0.0
            local_kin_flat[:, idx_y] = 0.0
        return np.concatenate([t_norm, np.asarray(local_kin_flat, dtype=np.float32), goal_rel], axis=1)

    def _low_target_action_filter(self, low_obs: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Safety-filter next actions for low-level SAC target-Q.

        low_obs format: [t_norm, local_kin_flat, goal_rel].
        Reconstruct goal_phys = ego_sub + goal_rel, then apply safety layer.
        """
        if not bool(getattr(self.cfg, "use_low_safety_layer", False)) or self.low_level_type != "sac":
            return np.asarray(action, dtype=np.float32)

        low_obs = np.asarray(low_obs, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)

        kin_slice = low_obs[:, : 1 + self.local_kin_flat_dim]
        _, kin, _ = utils.split_time_kinematics(kin_slice, self.n_veh_local, self.feat_dim)
        ego_sub = utils.extract_ego_substate(kin, self.ego_feature_idx)

        goal_rel = low_obs[:, 1 + self.local_kin_flat_dim : 1 + self.local_kin_flat_dim + self.ego_dim]
        goal_phys = (ego_sub + goal_rel).astype(np.float32)

        return self.low_safety.apply_safety_layer(low_obs, goal_phys, action)

    def _compute_intrinsic(self, kin: np.ndarray, kin_next: np.ndarray, goal_phys: np.ndarray, ego_start: np.ndarray, is_last_step: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_envs = int(kin.shape[0])
        intrinsic = np.zeros(n_envs, dtype=np.float32)
        goal_err = np.zeros((n_envs, self.ego_dim), dtype=np.float32)
        intrinsic_unweighted = np.zeros(n_envs, dtype=np.float32)

        intrinsic_type = str(getattr(self.cfg, "intrinsic_type", "l2")).lower()
        if intrinsic_type == "huber_shaping":
            ego_rel_now = utils.extract_ego_substate(kin, self.ego_feature_idx) - ego_start
            ego_rel_next = utils.extract_ego_substate(kin_next, self.ego_feature_idx) - ego_start
            goal_rel_all = goal_phys - ego_start

            intrinsic_val, goal_err_val, intrinsic_unw, _terminal_bonus = utils.intrinsic_reward_shaping_huber(ego_rel_now, ego_rel_next, goal_rel_all,
                self._intrinsic_norm_ranges, self.cfg.intrinsic_coef, self._intrinsic_weights, gamma=float(self.low_gamma), is_terminal=is_last_step)
            intrinsic = intrinsic_val

            if is_last_step.any():
                idx_last = np.flatnonzero(is_last_step)
                goal_err[idx_last] = goal_err_val[idx_last]
                intrinsic_unweighted[idx_last] = intrinsic_unw[idx_last]
        else:
            if is_last_step.any():
                idx_last = np.flatnonzero(is_last_step)
                ego_next_rel = utils.extract_ego_substate(kin_next[idx_last], self.ego_feature_idx) - ego_start[idx_last]
                goal_rel = goal_phys[idx_last] - ego_start[idx_last]

                intrinsic[idx_last], goal_err[idx_last], intrinsic_unweighted[idx_last] = utils.intrinsic_reward_l2(
                    ego_next_rel, goal_rel, self._intrinsic_norm_ranges, self.cfg.intrinsic_coef, self._intrinsic_weights
                )

        return intrinsic, goal_err, intrinsic_unweighted

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
        """
        Standard HIRO training (Joint).
        """
        return self._train(total_timesteps, callback, log_interval, progress_bar, train_high=True, train_low=True)

    def learn_low(self, total_timesteps: int, goal_sampler: Optional[Callable[[np.ndarray], np.ndarray]] = None, callback=None, log_interval: int = 1, progress_bar: bool = False):
        """
        Only train low-level agent. High-level agent is disabled; goals are sampled using goal_sampler.
        """
        return self._train(total_timesteps, callback, log_interval, progress_bar, train_high=False, train_low=True, high_policy=goal_sampler)

    def learn_high(self, total_timesteps: int, callback=None, log_interval: int = 1, progress_bar: bool = False):
        """
        Only train high-level agent. Low-level agent is fixed (inference only).
        """
        return self._train(total_timesteps, callback, log_interval, progress_bar, train_high=True, train_low=False)

    def _train(self, total_timesteps: int, callback, log_interval: int, progress_bar: bool, train_high: bool, train_low: bool, high_policy: Optional[Callable] = None):

        # ========== 0. initialization ========== #
        callback = self._init_callback(callback, progress_bar=progress_bar)
        self._propagate_log_interval(callback, log_interval)
        env = self.env
        obs = env.reset()
        done, truncated = False, False

        n_envs = self.n_envs
        hi = int(self.cfg.high_interval)

        need_high = np.ones(n_envs, dtype=bool)
        c = np.zeros(n_envs, dtype=np.int32)
        seg_id = np.zeros(n_envs, dtype=np.int64)
        seg_counter = 0

        high_obs_start = np.zeros_like(obs, dtype=np.float32)
        goal_action = np.zeros((n_envs, int(self.high_agent.action_space.shape[0])), dtype=np.float32)
        goal_buffer_action = np.zeros_like(goal_action)
        goal_phys = np.zeros((n_envs, self.ego_dim), dtype=np.float32)
        ego_start = np.zeros((n_envs, self.ego_dim), dtype=np.float32)
        goal_dist_start = np.zeros((n_envs, self.ego_dim), dtype=np.float32)

        high_ret = np.zeros(n_envs, dtype=np.float32)
        low_ret = np.zeros(n_envs, dtype=np.float32)
        low_len = np.zeros(n_envs, dtype=np.int32)
        low_comp_sums: dict[str, np.ndarray] = {}
        goal_err_all = np.zeros((n_envs, self.ego_dim), dtype=np.float32)
        intrinsic_unweighted = np.zeros(n_envs, dtype=np.float32)

        # HiRO high-level off-policy correction (OPC) requires low-level (obs, act) sequences per high-level transition.
        opc_enabled = bool(getattr(self, "use_off_policy_correction", False))
        if opc_enabled:
            low_act_dim = int(np.prod(env.action_space.shape))
            low_obs_dim = int(1 + self.local_kin_flat_dim + self.ego_dim)
            opc_low_obs_seq = np.zeros((n_envs, hi, low_obs_dim), dtype=np.float32)
            opc_low_act_seq = np.zeros((n_envs, hi, low_act_dim), dtype=np.float32)
        else:
            low_act_dim = 0
            opc_low_obs_seq = None
            opc_low_act_seq = None

        callback.on_training_start(locals(), globals())

        # ========== 1. Main Loop ========== #
        while self.total_timesteps < total_timesteps:

            # === 1.1 High Level Decision ===
            high_obs = self._build_high_obs(obs)
            _, kin, kin_flat = utils.split_time_kinematics(high_obs, self.n_veh, self.feat_dim)

            # step a high interval for required envs
            if need_high.any():
                idx = np.flatnonzero(need_high)

                if high_policy is not None:
                    # Use custom high-level policy (e.g. random sampler)
                    a = high_policy(high_obs[idx])
                    # For buffer action, if not training high, exact value doesn't matter much unless logged
                    a_buf = a.copy()
                else:
                    a, a_buf = self.high_agent.sample_action(high_obs[idx])

                high_obs_start[idx] = high_obs[idx]
                goal_action[idx] = a
                goal_buffer_action[idx] = a_buf

                ego_sub = utils.extract_ego_substate(kin[idx], self.ego_feature_idx)
                ego_start[idx] = ego_sub
                goal_phys[idx] = utils.goal_action_to_abs(ego_sub, a, self.lane_center_ys)
                goal_dist_start[idx] = goal_phys[idx] - ego_start[idx]

                high_ret[idx] = 0.0
                low_ret[idx] = 0.0
                low_len[idx] = 0
                for v in low_comp_sums.values():
                    v[idx] = 0.0

                if opc_enabled:
                    opc_low_obs_seq[idx] = 0.0
                    opc_low_act_seq[idx] = 0.0

                c[idx] = 0
                seg_id[idx] = np.arange(seg_counter, seg_counter + int(idx.size), dtype=np.int64)
                seg_counter += int(idx.size)
                need_high[idx] = False

            # === 1.2 Low Level Decision ===
            low_obs = self._build_low_obs(c, kin_flat, kin, goal_phys)
            
            if self.low_level_type == "rule_based":
                low_action = self.low_agent.act(low_obs, goal_phys)
                low_buffer_action = low_action.copy()
            elif self.low_level_type == "sac":
                low_action, low_buffer_action = self.low_agent.sample_action(low_obs)
                if bool(getattr(self.cfg, "use_low_safety_layer", False)):
                    low_action = self.low_safety.apply_safety_layer(low_obs, goal_phys, low_action)
                    if self.low_sac_impl == "safety_sac" or self.low_sac_impl == "auto":
                        low_buffer_action = low_action.copy()
            else:
                raise ValueError(f"Unknown low_level_type: {self.low_level_type}")

            if opc_enabled:
                # record (o_i, a_i) for off-policy correction
                opc_low_obs_seq[np.arange(n_envs), c] = low_obs
                opc_low_act_seq[np.arange(n_envs), c] = low_buffer_action

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
            if str(getattr(self.cfg, "intrinsic_type", "l2")).lower() == "huber_shaping":
                progress = np.asarray([rc.get("progress_reward", 0.0) for rc in r_components], dtype=np.float32)
                low_reward_ext = low_reward_ext - progress

            # calculate intrinsic reward
            is_last_step = (c == hi - 1) | done
            intrinsic, goal_err, intrinsic_unw = self._compute_intrinsic(kin, kin_next, goal_phys, ego_start, is_last_step)
            if is_last_step.any():
                idx_last = np.flatnonzero(is_last_step)
                goal_err_all[idx_last] = goal_err[idx_last]
                intrinsic_unweighted[idx_last] = intrinsic_unw[idx_last]

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

            if train_low and self.low_level_type == "sac":
                if low_infos is infos:
                    low_infos = [dict(info) for info in infos]
                for i in range(n_envs):
                    low_infos[i]["low_seg_id"] = int(seg_id[i])
                    low_infos[i]["low_t_in_seg"] = int(c[i])
                    low_infos[i]["low_ego_start"] = np.asarray(ego_start[i], dtype=np.float32)
                    low_infos[i]["low_r_ext"] = float(low_reward_ext[i])

            if train_low and self.low_level_type == "sac":
                self.low_agent.store_transition(
                    low_obs,
                    low_buffer_action,
                    next_low_obs,
                    (low_reward_ext if self.low_use_her else low_reward_total).astype(np.float32),
                    done_low,
                    low_infos,
                )
                self.low_agent.num_timesteps += n_envs
                self.low_agent.train_if_needed()

            self.total_timesteps += n_envs

            reward_env = reward
            callback.update_locals(locals())
            if callback.on_step() is False:
                break

            # === 1.5. High Level Store & Train ===
            if done_low.any():
                idx_end = np.flatnonzero(done_low)
                low_ret_end = low_ret[idx_end].copy()
                low_len_end = low_len[idx_end].copy()
                low_comp_end = {k: v[idx_end].copy() for k, v in low_comp_sums.items()}

                # goal tracking diagnostics for these finished low-episodes
                goal_err_end = goal_err_all[idx_end].copy()
                intrinsic_unweighted_end = intrinsic_unweighted[idx_end].copy()
                goal_dist_start_end = goal_dist_start[idx_end].copy()

                callback.update_locals({**locals(), "low_ret": low_ret_end, "low_len": low_len_end, "low_comp_sums": low_comp_end, "goal_err": goal_err_end, "intrinsic_unweighted": intrinsic_unweighted_end, "goal_dist_start": goal_dist_start_end})
                callback.on_rollout_end()

                if train_high:
                    self.high_agent.num_timesteps += int(idx_end.size)
                    for j in idx_end:
                        info_h = dict(infos[j])
                        info_h["high_interval_len"] = int(low_len[j])
                        if opc_enabled:
                            info_h["opc_low_obs_seq"] = opc_low_obs_seq[j, : int(low_len[j])].copy()
                            info_h["opc_low_act_seq"] = opc_low_act_seq[j, : int(low_len[j])].copy()
                        self.high_agent.store_transition(
                            high_obs_start[j:j + 1],
                            goal_buffer_action[j:j + 1],
                            next_high_obs[j:j + 1],
                            np.asarray([high_ret[j]], dtype=np.float32),
                            np.asarray([done[j]], dtype=np.bool_),
                            [info_h],
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
