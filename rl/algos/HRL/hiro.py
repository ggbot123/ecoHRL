# rl/algos/hiro/hiro.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional, Callable

import gymnasium as gym
import numpy as np

from rl.algos.sac.sac import SAC
from rl.algos.sac.safe_goal_policies import SafeGoalMlpPolicy
from rl.utils import utils
from rl.algos.HRL.rule_based import RuleBasedAgentWrapper
from rl.algos.HRL.goal_samplers import GoalSamplerConfig
from rl.algos.HRL.high_goal_safe_bounds import HighGoalSafeBoundsCalculator
from rl.algos.HRL.low_her_buffer import HiROLowHERReplayBuffer
from stable_baselines3.common.utils import get_device, configure_logger
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    ConvertCallback,
    ProgressBarCallback,
)
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


class HIROProgressBarCallback(ProgressBarCallback):
    """SB3 Rich progress bar driven by HIRO effective replay/train timesteps."""

    def __init__(self):
        super().__init__()
        self._last_num_timesteps = 0

    def _on_training_start(self) -> None:
        super()._on_training_start()
        self._last_num_timesteps = int(getattr(self.model, "num_timesteps", 0))

    def _on_step(self) -> bool:
        current = int(getattr(self.model, "num_timesteps", 0))
        delta = max(current - self._last_num_timesteps, 0)
        if delta > 0:
            remaining = max(int(self.pbar.total or 0) - int(self.pbar.n), 0)
            self.pbar.update(min(delta, remaining))
        self._last_num_timesteps = current
        return True


class SB3AgentWrapper:
    """Thin wrapper exposing SB3 private APIs (_sample_action/_store_transition) in batch form."""

    def __init__(self, agent: SAC, config_train_freq: int, gradient_steps: int, batch_size: int):
        self.agent = agent
        self.train_freq = int(config_train_freq)
        self.gradient_steps = int(gradient_steps)
        self.batch_size = int(batch_size)
        self.replay_buffer = agent.replay_buffer
        self._last_train_step = 0
        self._deferred_train_credit = 0.0

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

    def predict_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        n = int(obs.shape[0])

        if int(getattr(self.agent, "n_envs", n)) == 1 and n > 1:
            act_dim = int(self.agent.action_space.shape[0])
            action = np.empty((n, act_dim), dtype=np.float32)
            for i in range(n):
                a, _ = self.agent.predict(obs[i:i + 1], deterministic=deterministic)
                action[i] = np.asarray(a, dtype=np.float32).reshape(-1)
            return action

        a, _ = self.agent.predict(obs, deterministic=deterministic)
        return np.asarray(a, dtype=np.float32)

    def store_transition(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray, reward: np.ndarray, done: np.ndarray, infos: List[Dict[str, Any]]):
        self.agent._last_obs = obs
        self.agent._store_transition(self.replay_buffer, action, next_obs, reward, done, infos)

    def store_transition_direct(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ):
        self.replay_buffer.add(obs, next_obs, action, reward, done, infos)

    def train_if_needed(self):
        if self.num_timesteps <= self.agent.learning_starts:
            return
        due_updates = int(
            self.num_timesteps // self.train_freq
            > self._last_train_step // self.train_freq
        )
        if due_updates > 0:
            self.agent.train(
                gradient_steps=self.gradient_steps * int(due_updates),
                batch_size=self.batch_size,
            )
            self._last_train_step = self.num_timesteps

    def train_from_committed(self, committed_transitions: int, n_envs: int) -> None:
        """Train at the same rate as one update per original vector env step."""
        committed_transitions = max(int(committed_transitions), 0)
        if committed_transitions == 0:
            return

        previous_steps = self.num_timesteps
        self.num_timesteps = previous_steps + committed_transitions
        learning_starts = int(self.agent.learning_starts)
        eligible_transitions = (
            max(self.num_timesteps - learning_starts, 0)
            - max(previous_steps - learning_starts, 0)
        )
        self._deferred_train_credit += (
            float(eligible_transitions)
            / float(max(int(n_envs), 1) * max(self.train_freq, 1))
        )
        due_updates = int(np.floor(self._deferred_train_credit + 1e-12))
        if due_updates <= 0:
            return

        self.agent.train(
            gradient_steps=self.gradient_steps * due_updates,
            batch_size=self.batch_size,
        )
        self._deferred_train_credit -= float(due_updates)
        self._last_train_step = self.num_timesteps

    def save(self, path: str):
        self.agent.save(path)

    def __getattr__(self, name):
        return getattr(self.agent, name)


def discounted_option_reward_update(
    accumulated: np.ndarray,
    step_reward: np.ndarray,
    gamma: float,
    elapsed_steps: np.ndarray,
    high_interval: int,
) -> np.ndarray:
    interval_index = np.floor_divide(
        np.asarray(elapsed_steps, dtype=np.int64),
        max(int(high_interval), 1),
    )
    return accumulated + np.power(
        float(gamma),
        interval_index.astype(np.float32),
    ) * np.asarray(step_reward, dtype=np.float32)


def compute_low_level_external_reward(
    reward_env: np.ndarray,
    reward_components: List[Dict[str, Any]],
    replay_mask: np.ndarray,
    *,
    exclude_progress: bool,
) -> np.ndarray:
    """Remove task-level rewards that should only train the high level."""
    replay_mask_f = np.asarray(replay_mask, dtype=np.float32)
    low_reward = np.asarray(reward_env, dtype=np.float32) * replay_mask_f
    excluded_names = [
        "goal_lane_dense_reward",
        "punctual_reward",
        "wrong_lane_terminal_penalty",
    ]
    if exclude_progress:
        excluded_names.append("progress_reward")
    for name in excluded_names:
        component = np.asarray(
            [rc.get(name, 0.0) for rc in reward_components],
            dtype=np.float32,
        )
        low_reward -= component * replay_mask_f
    return low_reward


def option_bootstrap_discount(
    gamma: float,
    duration: int,
    high_interval: int,
) -> float:
    interval_count = max(
        1,
        int(np.ceil(max(int(duration), 0) / max(int(high_interval), 1))),
    )
    return float(float(gamma) ** interval_count)


@dataclass
class PendingLowTransition:
    obs: np.ndarray
    action: np.ndarray
    next_obs: np.ndarray
    reward: float
    done: bool
    info: Dict[str, Any]


class PendingLowEpisodes:
    def __init__(self, n_envs: int):
        self._episodes: list[list[PendingLowTransition]] = [
            [] for _ in range(int(n_envs))
        ]

    def append(self, env_i: int, transition: PendingLowTransition) -> None:
        self._episodes[int(env_i)].append(transition)

    def discard(self, env_i: int) -> int:
        env_i = int(env_i)
        count = len(self._episodes[env_i])
        self._episodes[env_i].clear()
        return count

    def commit(self, env_i: int) -> list[PendingLowTransition]:
        env_i = int(env_i)
        transitions = self._episodes[env_i]
        self._episodes[env_i] = []
        return transitions

    def size(self, env_i: int) -> int:
        return len(self._episodes[int(env_i)])


@dataclass
class LowSafetyFilterConfig:
    type: str = "mpc_constraints"  # "legacy" | "mpc_constraints" | "RSS" | "legacy_mpc_max"
    lane_change_min_front_gap: float = 15.0
    lane_change_min_rear_gap: float = 10.0
    lane_change_min_front_ttc: float = 3.0
    lane_change_min_rear_ttc: float = 2.0
    safe_gap_d_min: float = 6.0
    safe_gap_tau: float = 0.6
    safe_gap_b_ego: float = 3.0
    safe_gap_b_front: float = 3.0
    safe_gap_comfort_decel: float = -3.0
    safe_gap_emergency_decel: float = -5.0
    safe_gap_emergency_ttc: float = 1.0
    safe_gap_emergency_distance: float = 10.0


@dataclass
class HighGoalSafetyConfig:
    enabled: bool = False
    eps: float = 1e-6
    use_custom_kinematics: bool = True
    max_accel: Optional[float] = 2.0
    max_decel: Optional[float] = 3.0
    front_dmin: float = 15.0
    lane_change_rear_dmin: float = 10.0
    min_goal_x_span: float = 0.0
    enable_goal_vx_bounds: bool = False
    dynamic_feasible_lane_intervals: bool = True


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
    low_level_type: str = "sac" # "sac" | "rule_based"
    low_use_her: bool = False
    low_her_ratio: float = 0.8
    low_her_strategy: str = "future"  # "future" | "final"
    low_her_future_mode: str | None = None  # None->compat map, or "episode_timeaware"|"segment_timeaware"|"segment_legacy"
    low_her_episode_timeaware_steps_ahead_range: Optional[tuple[int, int] | List[int]] = None  # e.g. (1, 8)
    low_her_future_timeaware: bool = True  # True: new episode-time HER, False: legacy segment-future HER
    high_pretrained_path: Optional[str] = None
    low_pretrained_path: Optional[str] = None
    mask_ego_position_in_low_obs: bool = False
    use_low_safety_layer: bool = False
    low_safety_filter: Optional[LowSafetyFilterConfig] = None
    low_safety_violation_penalty: float = 0.0
    fixed_goal_vx: Optional[float] = None
    high_goal_safety: HighGoalSafetyConfig = field(default_factory=HighGoalSafetyConfig)
    high_obs_use_signal_features: bool = True

    @property
    def use_high_goal_safety_layer(self) -> bool:
        return bool(self.high_goal_safety.enabled)

    @property
    def high_goal_safe_eps(self) -> float:
        return float(self.high_goal_safety.eps)

    @property
    def high_goal_safe_use_custom_kinematics(self) -> bool:
        return bool(self.high_goal_safety.use_custom_kinematics)

    @property
    def high_goal_safe_max_accel(self) -> Optional[float]:
        return self.high_goal_safety.max_accel

    @property
    def high_goal_safe_max_decel(self) -> Optional[float]:
        return self.high_goal_safety.max_decel

    @property
    def high_goal_safe_front_dmin(self) -> float:
        return float(self.high_goal_safety.front_dmin)

    @property
    def high_goal_safe_lane_change_rear_dmin(self) -> float:
        return float(self.high_goal_safety.lane_change_rear_dmin)

    @property
    def high_goal_safe_min_goal_x_span(self) -> float:
        return float(self.high_goal_safety.min_goal_x_span)

    @property
    def high_goal_safe_enable_goal_vx_bounds(self) -> bool:
        return bool(self.high_goal_safety.enable_goal_vx_bounds)

    @property
    def high_goal_dynamic_feasible_lane_intervals(self) -> bool:
        return bool(self.high_goal_safety.dynamic_feasible_lane_intervals)


class HIROSAC:
    def __init__(
        self,
        env,
        high_sac_kwargs: Dict[str, Any],
        low_sac_kwargs: Dict[str, Any],
        config: HIROConfig,
        low_debug_config: Dict[str, Any] | None = None,
    ):
        self.env = env
        self.cfg = config
        self.low_debug_config = dict(low_debug_config or {})
        self.device = get_device(config.device)
        self.total_timesteps = 0
        self.n_envs = int(env.num_envs)

        # ---- 初始化 Kinematics 元信息 ----
        # Avoid a metadata reset: scenarios may update episode counters in reset().
        obs_shape = getattr(env.observation_space, "shape", None)
        if obs_shape is None:
            raise ValueError("HIROSAC requires a flat Box observation space.")
        obs0_flat = np.zeros(int(np.prod(obs_shape)), dtype=np.float32)
        keep_features = ("x", "y", "vx", "vy")  # ego 子状态中参与 HIRO goal 的特征
        (self.n_veh, self.n_veh_local, self.feat_dim, self.feature_names, self.ego_feature_idx) = \
            utils.init_kinematics_meta(env, obs0_flat, keep_features)
        self.kin_flat_dim = int(self.n_veh * self.feat_dim)
        self.local_kin_flat_dim = int(self.n_veh_local * self.feat_dim)
        self.obs_extra_dim = int(max(0, int(np.prod(obs_shape)) - (1 + self.kin_flat_dim)))
        self.ego_dim = len(self.ego_feature_idx)
        self._intrinsic_norm_ranges = np.asarray(self.cfg.intrinsic_norm_ranges, dtype=np.float32)
        w = getattr(self.cfg, "intrinsic_weights", None)
        self._intrinsic_weights = None if w is None else np.asarray(w, dtype=np.float32)

        # ---- 从env中获取的必要变量 --- #
        env_cfg = env.get_attr("config", indices=0)[0]
        self.queue_takeover_enabled = bool(env_cfg.get("enable_queue_takeover", False))
        self.high_gamma = float(high_sac_kwargs.get("gamma", 0.99))
        self.punctual_time_target = float(env_cfg.get("punctual_time_target", env_cfg.get("duration", 0.0)))
        self.goal_longitudinal = float(env_cfg.get("goal_longitudinal", env_cfg.get("road_length", 0.0)))
        self.ego_x_idx = int(self.feature_names.index("x"))
        # self.v_min, self.v_max = 8.0, float(env_cfg["speed_limit"])
        self.v_min, self.v_max = 0.0, float(env_cfg["speed_limit"])
        self.dt = 1.0 / float(env_cfg["policy_frequency"])
        lanes = int(env_cfg["lanes_count"])
        lane_w = float(env_cfg.get("lane_width", 4.0))
        self.n_lanes = int(lanes)
        self.max_lane_id = int(max(self.n_lanes - 1, 1))
        self.lane_center_ys = (np.arange(lanes, dtype=np.float32) * lane_w).astype(np.float32)

        # ----- 定义 Spaces -----  #
        high_obs_dim = self.kin_flat_dim + self.obs_extra_dim + 1 + 2
        self.high_obs_dim = int(high_obs_dim)
        high_obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(high_obs_dim,), dtype=np.float32)

        t_h = float(self.cfg.high_interval) * self.dt
        # 目标空间：[rel_x, rel_y, vx]，rel_y从[-1, 1]映射到左右车道中心线
        goal_low = np.array([self.v_min * t_h, -1, self.v_min], dtype=np.float32)
        goal_high = np.array([self.v_max * t_h, 1, self.v_max], dtype=np.float32)
        fixed_goal_vx = getattr(self.cfg, "fixed_goal_vx", None)
        if fixed_goal_vx is not None:
            fixed_goal_vx = float(np.clip(float(fixed_goal_vx), self.v_min, self.v_max))
            goal_low[2] = fixed_goal_vx - 0.01
            goal_high[2] = fixed_goal_vx + 0.01
        high_act_space = gym.spaces.Box(goal_low, goal_high, dtype=np.float32)
        enable_goal_vx_bounds = bool(getattr(self.cfg, "high_goal_safe_enable_goal_vx_bounds", True))
        if fixed_goal_vx is not None and np.isclose(float(fixed_goal_vx), 0.0):
            enable_goal_vx_bounds = False

        action_cfg = env_cfg.get("action", {})
        accel_range = action_cfg.get("acceleration_range", [-5.0, 5.0])
        self._acc_min = float(accel_range[0])
        self._acc_max = float(accel_range[1])
        default_max_accel = float(max(abs(float(accel_range[0])), abs(float(accel_range[1]))))
        use_custom_kin = bool(getattr(self.cfg, "high_goal_safe_use_custom_kinematics", False))
        if use_custom_kin:
            cfg_max_accel = getattr(self.cfg, "high_goal_safe_max_accel", None)
            cfg_max_decel = getattr(self.cfg, "high_goal_safe_max_decel", None)
            max_accel = float(default_max_accel if cfg_max_accel is None else max(float(cfg_max_accel), 0.0))
            max_decel = float(default_max_accel if cfg_max_decel is None else max(float(cfg_max_decel), 0.0))
        else:
            max_accel = float(default_max_accel)
            max_decel = float(default_max_accel)
        self.high_goal_safe_bounds = HighGoalSafeBoundsCalculator(
            n_lanes=self.n_lanes,
            lane_width=lane_w,
            high_interval=int(self.cfg.high_interval),
            dt=self.dt,
            speed_min=self.v_min,
            speed_max=self.v_max,
            max_accel=max_accel,
            max_decel=max_decel,
            front_dmin=float(max(0.0, getattr(self.cfg, "high_goal_safe_front_dmin", 0.0))),
            lane_change_rear_dmin=float(max(0.0, getattr(self.cfg, "high_goal_safe_lane_change_rear_dmin", 0.0))),
            min_goal_x_span=float(max(0.0, getattr(self.cfg, "high_goal_safe_min_goal_x_span", 0.0))),
            dx_low=float(goal_low[0]),
            dx_high=float(goal_high[0]),
            feat_dim=int(self.feat_dim),
            n_veh=int(self.n_veh),
            presence_idx=int(self.feature_names.index("presence")),
            x_idx=int(self.feature_names.index("x")),
            y_idx=int(self.feature_names.index("y")),
            vx_idx=int(self.feature_names.index("vx")),
            vy_idx=int(self.feature_names.index("vy")),
            enable_goal_vx_bounds=bool(enable_goal_vx_bounds),
        )

        low_obs_dim = self.local_kin_flat_dim + self.obs_extra_dim + self.ego_dim + 1
        self.low_obs_dim = int(low_obs_dim)
        low_obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(low_obs_dim,), dtype=np.float32)
        low_act_space = env.action_space

        # ----- Build low-level agent ----- #
        self.train_mode = str(getattr(config, "train_mode", "joint")).lower()
        self.low_level_type = str(getattr(config, "low_level_type", "sac")).lower()
        self.low_use_her = bool(getattr(self.cfg, "low_use_her", False))
        self.use_off_policy_correction = bool(getattr(self.cfg, "use_off_policy_correction", False))
        if self.queue_takeover_enabled and self.use_off_policy_correction:
            raise ValueError(
                "enable_queue_takeover is incompatible with HIRO off-policy correction "
                "because the extended option contains environment-controller actions"
            )
        self.low_gamma = float(low_sac_kwargs.get("gamma", 0.99))
        if self.low_level_type == "rule_based":
            self.low_agent = RuleBasedAgentWrapper(
                env,
                self.n_envs,
                high_interval=int(config.high_interval),
                low_safety_filter=getattr(config, "low_safety_filter", None),
            )
        elif self.low_level_type == "sac":
            use_low_safety_layer = bool(getattr(self.cfg, "use_low_safety_layer", False))

            # Resolve scaled auto target entropy for low-level SAC: -scale * action_dim.
            low_sac_kwargs = dict(low_sac_kwargs)

            # In high_only mode, low SAC is inference-only: avoid allocating large replay memory.
            low_inference_only = self.train_mode == "high_only"
            low_sac_n_envs = self.n_envs
            if low_inference_only:
                low_sac_kwargs["buffer_size"] = int(min(int(low_sac_kwargs.get("buffer_size", 1000000)), 1024))
                low_sac_kwargs.pop("replay_buffer_class", None)
                rb_kwargs_low = dict(low_sac_kwargs.get("replay_buffer_kwargs", {}) or {})
                rb_kwargs_low["handle_timeout_termination"] = False
                low_sac_kwargs["replay_buffer_kwargs"] = rb_kwargs_low
                print("[HIRO] Low SAC inference-only mode in high_only: keep n_envs, use small replay buffer")

            use_low_her = bool(self.low_use_her)
            if low_inference_only:
                use_low_her = False
            if use_low_her:
                rb_kwargs_low = dict(low_sac_kwargs.get("replay_buffer_kwargs", {}) or {})
                rb_kwargs_low.update(
                    dict(
                        feat_dim=int(self.feat_dim),
                        kin_flat_dim=int(self.local_kin_flat_dim),
                        obs_extra_dim=int(self.obs_extra_dim),
                        ego_feature_idx=list(self.ego_feature_idx),
                        intrinsic_coef=float(self.cfg.intrinsic_coef),
                        intrinsic_norm_ranges=np.asarray(self.cfg.intrinsic_norm_ranges, dtype=np.float32),
                        intrinsic_weights=None
                        if getattr(self.cfg, "intrinsic_weights", None) is None
                        else np.asarray(self.cfg.intrinsic_weights, dtype=np.float32),
                        intrinsic_type=str(getattr(self.cfg, "intrinsic_type", "l2")),
                        low_gamma=float(self.low_gamma),
                        high_interval=int(self.cfg.high_interval),
                        fixed_goal_vx=getattr(self.cfg, "fixed_goal_vx", None),
                        her_ratio=float(getattr(self.cfg, "low_her_ratio", 0.8)),
                        her_strategy=str(getattr(self.cfg, "low_her_strategy", "future")),
                        her_future_mode=getattr(self.cfg, "low_her_future_mode", None),
                        her_episode_timeaware_steps_ahead_range=getattr(self.cfg, "low_her_episode_timeaware_steps_ahead_range", None),
                        her_future_timeaware=bool(getattr(self.cfg, "low_her_future_timeaware", True)),
                        her_debug_enabled=bool(self.low_debug_config.get("her_debug_enabled", False)),
                        her_debug_max_records=int(self.low_debug_config.get("her_debug_max_records", 20000)),
                        her_debug_sample_prob=float(self.low_debug_config.get("her_debug_sample_prob", 0.0)),
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
                low_sac = SAC.load(
                    low_pretrained_path,
                    env=_make_dummy_vec_env(low_obs_space, low_act_space, low_sac_n_envs),
                    device=self.device,
                    **low_sac_kwargs,
                )
            else:
                low_sac = SAC(
                    env=_make_dummy_vec_env(low_obs_space, low_act_space, low_sac_n_envs),
                    **low_sac_kwargs,
                )
            self.low_agent = SB3AgentWrapper(low_sac, config.train_freq, config.gradient_steps_low, config.batch_size)
            if use_low_safety_layer:
                self.low_safety = RuleBasedAgentWrapper(
                    env,
                    self.n_envs,
                    high_interval=int(config.high_interval),
                    low_safety_filter=getattr(config, "low_safety_filter", None),
                )
        else:
            raise ValueError(f"Unknown low_level_type: {self.low_level_type}")

        # ----- High-level buffer config (dynamic OPC metadata only) ----- #
        high_sac_kwargs = dict(high_sac_kwargs)
        if bool(getattr(self.cfg, "use_high_goal_safety_layer", False)):
            high_sac_kwargs["policy"] = SafeGoalMlpPolicy
            high_sac_kwargs["safe_warmup_sampling"] = True

            policy_kwargs = dict(high_sac_kwargs.get("policy_kwargs", {}) or {})
            policy_kwargs["goal_safe_eps"] = float(getattr(self.cfg, "high_goal_safe_eps", 1e-6))
            policy_kwargs["dynamic_feasible_lane_intervals"] = bool(
                getattr(self.cfg, "high_goal_dynamic_feasible_lane_intervals", False)
            )
            high_sac_kwargs["policy_kwargs"] = policy_kwargs

        rb_kwargs = dict(high_sac_kwargs.get("replay_buffer_kwargs", {}) or {})
        rb_kwargs.update(
            dict(
                max_seq_len=int(self.cfg.high_interval),
                kin_flat_dim=int(self.local_kin_flat_dim),
                obs_extra_dim=int(self.obs_extra_dim),
                low_action_dim=int(np.prod(low_act_space.shape)),
                feat_dim=int(self.feat_dim),
                ego_feature_idx=list(self.ego_feature_idx),
                lane_center_ys=self.lane_center_ys,
                high_interval=int(self.cfg.high_interval),
                dynamic_feasible_lane_intervals=bool(
                    getattr(self.cfg, "high_goal_dynamic_feasible_lane_intervals", False)
                ),
                low_policy=self.low_agent.policy if self.use_off_policy_correction else None,
            )
        )
        rb_kwargs["enable_off_policy_correction"] = bool(self.use_off_policy_correction)
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

        if bool(getattr(self.cfg, "use_high_goal_safety_layer", False)):
            high_sac.actor.goal_safe_eps = float(getattr(self.cfg, "high_goal_safe_eps", 1e-6))
            high_sac.actor.goal_safe_bounds_fn = self.high_goal_safe_bounds.compute_torch
            high_sac.actor.dynamic_feasible_lane_intervals = bool(
                getattr(self.cfg, "high_goal_dynamic_feasible_lane_intervals", False)
            )

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
        if progress_bar:
            callback = CallbackList([callback, HIROProgressBarCallback()])
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
    def _get_signal_features(self) -> np.ndarray:
        """Get per-env traffic-signal features [is_green, remaining_seconds]."""
        n = int(self.n_envs)
        out = np.zeros((n, 2), dtype=np.float32)
        out[:] = -1.0

        if not bool(getattr(self.cfg, "high_obs_use_signal_features", True)):
            return out

        if not hasattr(self.env, "env_method"):
            return out

        try:
            vals = self.env.env_method("get_hiro_signal_features")
        except Exception:
            return out

        if vals is None:
            return out
        for i, v in enumerate(vals):
            if i >= n:
                break
            try:
                color, remain = v
                out[i, 0] = float(color)
                out[i, 1] = float(remain)
            except Exception:
                continue
        return out

    def _get_goal_longitudinal(self) -> np.ndarray:
        """Get per-env goal x for constructing remaining-x high observation."""
        n = int(self.n_envs)
        # Avoid calling private env methods through wrappers (Monitor/OrderEnforcing)
        # in SubprocVecEnv workers on Windows spawn mode.
        return np.full((n,), float(self.goal_longitudinal), dtype=np.float32)

    def _get_punctual_time_targets(self) -> np.ndarray:
        """Get the current episode punctual target for each parallel env."""
        n = int(self.n_envs)
        fallback = np.full((n,), float(self.punctual_time_target), dtype=np.float32)
        if not hasattr(self.env, "env_method"):
            return fallback
        try:
            values = self.env.env_method("get_punctual_time_target")
        except Exception:
            return fallback
        if values is None:
            return fallback
        out = fallback.copy()
        for i, value in enumerate(values):
            if i >= n:
                break
            try:
                out[i] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    def _get_queue_takeover_mask(self) -> np.ndarray:
        if not self.queue_takeover_enabled or not hasattr(self.env, "env_method"):
            return np.zeros(self.n_envs, dtype=bool)
        try:
            values = self.env.env_method("get_queue_takeover_active")
        except Exception:
            return np.zeros(self.n_envs, dtype=bool)
        out = np.zeros(self.n_envs, dtype=bool)
        for i, value in enumerate(values or []):
            if i >= self.n_envs:
                break
            out[i] = bool(value)
        return out

    def _get_queue_takeover_actions(self) -> np.ndarray:
        actions = np.zeros((self.n_envs, int(np.prod(self.env.action_space.shape))), dtype=np.float32)
        if not self.queue_takeover_enabled or not hasattr(self.env, "env_method"):
            return actions
        try:
            values = self.env.env_method("get_queue_takeover_action")
        except Exception:
            return actions
        for i, value in enumerate(values or []):
            if i >= self.n_envs:
                break
            actions[i] = np.asarray(value, dtype=np.float32).reshape(-1)
        return actions

    def _build_high_obs(self, obs: np.ndarray, signal_feat: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        t_cur = arr[:, :1]
        punctual_targets = self._get_punctual_time_targets().reshape(-1, 1)
        t_remaining = (punctual_targets - t_cur).astype(np.float32)
        kin_flat = np.asarray(arr[:, 1 : 1 + self.kin_flat_dim], dtype=np.float32).copy()
        extra = np.asarray(arr[:, 1 + self.kin_flat_dim : 1 + self.kin_flat_dim + self.obs_extra_dim], dtype=np.float32)
        ego_x = kin_flat[:, self.ego_x_idx]
        goal_x = self._get_goal_longitudinal()
        kin_flat[:, self.ego_x_idx] = (goal_x - ego_x).astype(np.float32)
        sig = np.asarray(signal_feat, dtype=np.float32).reshape(arr.shape[0], 2)
        return np.concatenate([t_remaining, kin_flat, extra, sig], axis=1).astype(np.float32)

    def _build_low_obs(
        self,
        t_rel: np.ndarray,
        kin_flat: np.ndarray,
        kin: np.ndarray,
        goal_phys: np.ndarray,
        obs_extra: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        低层观测 = t_norm + local_kin_flat + goal_rel
        接收绝对坐标系 goal_phys，根据当前 ego 状态计算 goal_rel
        """
        t_norm = (np.asarray(t_rel, dtype=np.float32) / float(self.cfg.high_interval)).reshape(-1, 1)
        ego_sub = utils.extract_ego_substate(kin, self.ego_feature_idx)
        goal_rel = (np.asarray(goal_phys, dtype=np.float32) - ego_sub).astype(np.float32)
        local_kin_flat = np.asarray(kin_flat[:, :self.local_kin_flat_dim], dtype=np.float32).copy()
        if self.obs_extra_dim > 0:
            if obs_extra is None:
                extra = np.zeros((local_kin_flat.shape[0], self.obs_extra_dim), dtype=np.float32)
            else:
                extra = np.asarray(obs_extra, dtype=np.float32).reshape(local_kin_flat.shape[0], self.obs_extra_dim)
        else:
            extra = np.zeros((local_kin_flat.shape[0], 0), dtype=np.float32)

        # Mask ego absolute position in low_obs
        mask_ego_pos = bool(getattr(self.cfg, "mask_ego_position_in_low_obs", False))
        if self.low_level_type == "rule_based":
            mask_ego_pos = False
        if mask_ego_pos:
            idx_x = int(self.feature_names.index("x"))
            idx_y = int(self.feature_names.index("y"))
            local_kin_flat[:, idx_x] = 0.0
            local_kin_flat[:, idx_y] = 0.0
        return np.concatenate([t_norm, np.asarray(local_kin_flat, dtype=np.float32), extra, goal_rel], axis=1)

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
        if (
            self.queue_takeover_enabled
            and train_low
            and self.low_level_type == "sac"
            and not self.low_use_her
        ):
            raise ValueError(
                "Training low-level SAC with enable_queue_takeover requires "
                "low_use_her=True so takeover steps can be excluded from replay"
            )
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
        ep_id = np.zeros(n_envs, dtype=np.int64)
        ep_step = np.zeros(n_envs, dtype=np.int64)

        high_obs_start = np.zeros((n_envs, int(self.high_obs_dim)), dtype=np.float32)
        goal_action = np.zeros((n_envs, int(self.high_agent.action_space.shape[0])), dtype=np.float32)
        goal_buffer_action = np.zeros_like(goal_action)
        goal_phys = np.zeros((n_envs, self.ego_dim), dtype=np.float32)
        ego_start = np.zeros((n_envs, self.ego_dim), dtype=np.float32)
        goal_dist_start = np.zeros((n_envs, self.ego_dim), dtype=np.float32)

        high_ret = np.zeros(n_envs, dtype=np.float32)
        low_ret = np.zeros(n_envs, dtype=np.float32)
        low_len = np.zeros(n_envs, dtype=np.int32)
        high_len = np.zeros(n_envs, dtype=np.int32)
        low_safety_clip_count = np.zeros(n_envs, dtype=np.int32)
        low_comp_sums: dict[str, np.ndarray] = {}
        high_comp_keys = [
            "collision_reward",
            "progress_reward",
            "speed_ref_aux_reward",
            "comfort_reward_for_high",
            "lane_change_reward",
            "goal_lane_dense_reward",
            "punctual_reward",
            "wrong_lane_terminal_penalty",
        ]
        high_comp_sums = {k: np.zeros(n_envs, dtype=np.float32) for k in high_comp_keys}
        high_acc_min = np.full(n_envs, np.inf, dtype=np.float32)
        high_acc_max = np.full(n_envs, -np.inf, dtype=np.float32)
        high_acc_sum = np.zeros(n_envs, dtype=np.float32)
        high_acc_abs_sum = np.zeros(n_envs, dtype=np.float32)
        high_acc_count = np.zeros(n_envs, dtype=np.int32)
        high_acc_hard_brake = np.zeros(n_envs, dtype=np.int32)
        high_acc_hard_accel = np.zeros(n_envs, dtype=np.int32)
        goal_err_all = np.zeros((n_envs, self.ego_dim), dtype=np.float32)
        intrinsic_unweighted = np.zeros(n_envs, dtype=np.float32)

        # HiRO high-level off-policy correction (OPC) requires low-level (obs, act) sequences per high-level transition.
        opc_enabled = bool(getattr(self, "use_off_policy_correction", False))
        if opc_enabled:
            low_act_dim = int(np.prod(env.action_space.shape))
            low_obs_dim = int(1 + self.local_kin_flat_dim + self.obs_extra_dim + self.ego_dim)
            opc_low_obs_seq = np.zeros((n_envs, hi, low_obs_dim), dtype=np.float32)
            opc_low_act_seq = np.zeros((n_envs, hi, low_act_dim), dtype=np.float32)
        else:
            low_act_dim = 0
            opc_low_obs_seq = None
            opc_low_act_seq = None

        callback.on_training_start(locals(), globals())
        obs_skip_mask = np.zeros(n_envs, dtype=bool)
        pending_low = PendingLowEpisodes(n_envs)
        use_pending_low = bool(
            self.queue_takeover_enabled
            and train_low
            and self.low_level_type == "sac"
        )

        def _commit_pending_low(env_indices: np.ndarray) -> int:
            committed_by_env = {
                int(env_i): pending_low.commit(int(env_i))
                for env_i in np.asarray(env_indices, dtype=np.int64).reshape(-1)
            }
            max_rows = max((len(items) for items in committed_by_env.values()), default=0)
            committed_count = sum(len(items) for items in committed_by_env.values())
            for row_i in range(max_rows):
                obs_batch = np.zeros((n_envs, self.low_obs_dim), dtype=np.float32)
                action_batch = np.zeros(
                    (n_envs, int(np.prod(env.action_space.shape))),
                    dtype=np.float32,
                )
                next_obs_batch = np.zeros_like(obs_batch)
                reward_batch = np.zeros(n_envs, dtype=np.float32)
                done_batch = np.zeros(n_envs, dtype=np.bool_)
                info_batch = [{"skip_replay": True} for _ in range(n_envs)]

                for env_i, transitions in committed_by_env.items():
                    if row_i >= len(transitions):
                        continue
                    transition = transitions[row_i]
                    obs_batch[env_i] = transition.obs
                    action_batch[env_i] = transition.action
                    next_obs_batch[env_i] = transition.next_obs
                    reward_batch[env_i] = float(transition.reward)
                    done_batch[env_i] = bool(transition.done)
                    info_batch[env_i] = dict(transition.info)
                self.low_agent.store_transition_direct(
                    obs_batch,
                    action_batch,
                    next_obs_batch,
                    reward_batch,
                    done_batch,
                    info_batch,
                )
            return committed_count

        # ========== 1. Main Loop ========== #
        while self.total_timesteps < total_timesteps:
            active_obs_mask = ~obs_skip_mask

            # === 1.1 High Level Decision ===
            signal_feat = self._get_signal_features()
            high_obs = self._build_high_obs(obs, signal_feat)
            _, kin, kin_flat = utils.split_time_kinematics(obs, self.n_veh, self.feat_dim)
            obs_extra = np.asarray(obs[:, 1 + self.kin_flat_dim : 1 + self.kin_flat_dim + self.obs_extra_dim], dtype=np.float32)

            # step a high interval for required envs
            need_high_now = need_high & active_obs_mask
            if need_high_now.any():
                idx = np.flatnonzero(need_high_now)

                if high_policy is not None:
                    # Use custom high-level policy (e.g. random sampler)
                    a = high_policy(high_obs[idx])
                    # For buffer action, if not training high, exact value doesn't matter much unless logged
                    a_buf = a.copy()
                else:
                    a, a_buf = self.high_agent.sample_action(high_obs[idx])

                a = np.asarray(a, dtype=np.float32)
                a_buf = np.asarray(a_buf, dtype=np.float32)

                high_obs_start[idx] = high_obs[idx]
                goal_action[idx] = a
                goal_buffer_action[idx] = a_buf

                ego_sub = utils.extract_ego_substate(kin[idx], self.ego_feature_idx)
                ego_start[idx] = ego_sub
                goal_phys[idx] = utils.goal_action_to_abs(
                    ego_sub,
                    a,
                    self.lane_center_ys,
                    dynamic_feasible_intervals=bool(
                        getattr(self.cfg, "high_goal_dynamic_feasible_lane_intervals", False)
                    ),
                )
                goal_dist_start[idx] = goal_phys[idx] - ego_start[idx]

                high_ret[idx] = 0.0
                low_ret[idx] = 0.0
                low_len[idx] = 0
                high_len[idx] = 0
                low_safety_clip_count[idx] = 0
                for v in low_comp_sums.values():
                    v[idx] = 0.0
                for v in high_comp_sums.values():
                    v[idx] = 0.0
                high_acc_min[idx] = np.inf
                high_acc_max[idx] = -np.inf
                high_acc_sum[idx] = 0.0
                high_acc_abs_sum[idx] = 0.0
                high_acc_count[idx] = 0
                high_acc_hard_brake[idx] = 0
                high_acc_hard_accel[idx] = 0
                goal_err_all[idx] = 0.0
                intrinsic_unweighted[idx] = 0.0

                if opc_enabled:
                    opc_low_obs_seq[idx] = 0.0
                    opc_low_act_seq[idx] = 0.0

                c[idx] = 0
                seg_id[idx] = np.arange(seg_counter, seg_counter + int(idx.size), dtype=np.int64)
                seg_counter += int(idx.size)
                need_high[idx] = False

            # === 1.2 Low Level Decision ===
            queue_takeover_mask = self._get_queue_takeover_mask() & active_obs_mask
            low_obs = self._build_low_obs(c, kin_flat, kin, goal_phys, obs_extra)
            if obs_skip_mask.any():
                low_obs[obs_skip_mask] = 0.0
            low_safety_clipped = np.zeros(n_envs, dtype=bool)
            
            if self.low_level_type == "rule_based":
                low_action_raw = self.low_agent.act(low_obs, goal_phys)
                low_action = self.low_agent.apply_safety_layer(low_obs, goal_phys, low_action_raw)
                low_safety_clipped = np.any(np.abs(low_action - low_action_raw) > 1e-6, axis=1)
                safety_count_mask = active_obs_mask & (~queue_takeover_mask)
                if safety_count_mask.any():
                    low_safety_clip_count[safety_count_mask] += low_safety_clipped[safety_count_mask].astype(np.int32)
                low_buffer_action = low_action.copy()
            elif self.low_level_type == "sac":
                if train_low:
                    low_action, low_buffer_action = self.low_agent.sample_action(low_obs)
                else:
                    low_action = self.low_agent.predict_action(low_obs, deterministic=True)
                    low_buffer_action = low_action.copy()
                if bool(getattr(self.cfg, "use_low_safety_layer", False)):
                    low_action_raw = low_action.copy()
                    low_action = self.low_safety.apply_safety_layer(low_obs, goal_phys, low_action)
                    low_safety_clipped = np.any(np.abs(low_action - low_action_raw) > 1e-6, axis=1)
                    safety_count_mask = active_obs_mask & (~queue_takeover_mask)
                    if safety_count_mask.any():
                        low_safety_clip_count[safety_count_mask] += low_safety_clipped[safety_count_mask].astype(np.int32)
            else:
                raise ValueError(f"Unknown low_level_type: {self.low_level_type}")

            if queue_takeover_mask.any():
                queue_actions = self._get_queue_takeover_actions()
                low_action[queue_takeover_mask] = queue_actions[queue_takeover_mask]
                low_buffer_action[queue_takeover_mask] = queue_actions[queue_takeover_mask]
                low_safety_clipped[queue_takeover_mask] = False

            if obs_skip_mask.any():
                low_action = np.asarray(low_action, dtype=np.float32)
                low_buffer_action = np.asarray(low_buffer_action, dtype=np.float32)
                low_action[obs_skip_mask] = 0.0
                low_buffer_action[obs_skip_mask] = 0.0
                low_safety_clipped[obs_skip_mask] = False

            if opc_enabled and active_obs_mask.any():
                # record (o_i, a_i) for off-policy correction
                idx_active = np.flatnonzero(active_obs_mask)
                opc_low_obs_seq[idx_active, c[idx_active]] = low_obs[idx_active]
                opc_low_act_seq[idx_active, c[idx_active]] = low_buffer_action[idx_active]

            next_obs, reward_env, done, infos = env.step(low_action)
            reward_env = np.asarray(reward_env, dtype=np.float32)
            done = np.asarray(done, dtype=bool)
            infos = list(infos) if isinstance(infos, (list, tuple)) else [{} for _ in range(n_envs)]
            if len(infos) != n_envs:
                infos = [{} for _ in range(n_envs)]
            infos = [info if isinstance(info, dict) else {} for info in infos]

            skip_replay_mask = np.asarray([bool(info.get("skip_replay", False)) for info in infos], dtype=bool)
            replay_mask = ~skip_replay_mask
            next_obs_is_dummy = np.asarray([bool(info.get("next_obs_is_dummy", False)) for info in infos], dtype=bool)
            inter_episode_mask = np.asarray([bool(info.get("inter_episode", False)) for info in infos], dtype=bool)
            queue_takeover_next_mask = np.asarray(
                [bool(info.get("queue_takeover_active", False)) for info in infos],
                dtype=bool,
            )
            queue_enter_mask = replay_mask & (~queue_takeover_mask) & queue_takeover_next_mask
            queue_exit_mask = replay_mask & queue_takeover_mask & (~queue_takeover_next_mask)
            low_replay_mask = replay_mask & (~queue_takeover_mask)
            dummy_finished_mask = skip_replay_mask & inter_episode_mask & (~next_obs_is_dummy)
            replay_count = int(np.sum(replay_mask))

            next_obs_tr = self._terminal_obs(next_obs, done, infos)
            signal_feat_next = self._get_signal_features()
            if bool(getattr(self.cfg, "high_obs_use_signal_features", True)) and done.any():
                for i in np.flatnonzero(done):
                    tsig = infos[i].get("terminal_signal_features")
                    if tsig is None:
                        continue
                    try:
                        sig_i = np.asarray(tsig, dtype=np.float32).reshape(-1)
                        if sig_i.size >= 2:
                            signal_feat_next[i, 0] = float(sig_i[0])
                            signal_feat_next[i, 1] = float(sig_i[1])
                    except Exception:
                        continue
            next_high_obs = self._build_high_obs(next_obs_tr, signal_feat_next)
            _, kin_next, kin_flat_next = utils.split_time_kinematics(next_obs_tr, self.n_veh, self.feat_dim)
            next_obs_extra = np.asarray(next_obs_tr[:, 1 + self.kin_flat_dim : 1 + self.kin_flat_dim + self.obs_extra_dim], dtype=np.float32)

            # === 1.3 Calculate Rewards ===
            r_components = [info.get("reward_components", {}) for info in infos]
            physical_acc = np.zeros(n_envs, dtype=np.float32)
            try:
                acc_cmd = np.asarray(low_action, dtype=np.float32)
                if acc_cmd.ndim == 2 and acc_cmd.shape[1] >= 2:
                    physical_acc = acc_cmd[:, 1].astype(np.float32)
                    scaled_mask = (physical_acc >= -1.0001) & (physical_acc <= 1.0001)
                    physical_acc[scaled_mask] = (
                        (physical_acc[scaled_mask] + 1.0)
                        * 0.5
                        * (float(self._acc_max) - float(self._acc_min))
                        + float(self._acc_min)
                    )
                    physical_acc = np.clip(physical_acc, float(self._acc_min), float(self._acc_max))
            except Exception:
                physical_acc = np.zeros(n_envs, dtype=np.float32)

            replay_mask_f = replay_mask.astype(np.float32)
            high_step_reward = reward_env * replay_mask_f
            high_ret = discounted_option_reward_update(
                high_ret,
                high_step_reward,
                self.high_gamma,
                high_len,
                hi,
            )
            high_step_discount = np.power(
                self.high_gamma,
                np.floor_divide(high_len, hi).astype(np.float32),
            )

            if replay_mask.any():
                for i, rc in enumerate(r_components):
                    if not replay_mask[i]:
                        continue
                    discount_i = float(high_step_discount[i])
                    high_comp_sums["collision_reward"][i] += discount_i * float(rc.get("collision_reward", 0.0))
                    high_comp_sums["progress_reward"][i] += discount_i * float(rc.get("progress_reward", 0.0))
                    high_comp_sums["speed_ref_aux_reward"][i] += discount_i * float(rc.get("speed_ref_aux_reward", 0.0))
                    high_comp_sums["comfort_reward_for_high"][i] += discount_i * float(rc.get("comfort_reward", 0.0))
                    high_comp_sums["lane_change_reward"][i] += discount_i * float(rc.get("lane_change_reward", 0.0))
                    high_comp_sums["goal_lane_dense_reward"][i] += discount_i * float(rc.get("goal_lane_dense_reward", 0.0))
                    high_comp_sums["punctual_reward"][i] += discount_i * float(rc.get("punctual_reward", 0.0))
                    high_comp_sums["wrong_lane_terminal_penalty"][i] += discount_i * float(rc.get("wrong_lane_terminal_penalty", 0.0))
                idx_replay = np.flatnonzero(replay_mask)
                acc_replay = physical_acc[idx_replay]
                high_acc_min[idx_replay] = np.minimum(high_acc_min[idx_replay], acc_replay)
                high_acc_max[idx_replay] = np.maximum(high_acc_max[idx_replay], acc_replay)
                high_acc_sum[idx_replay] += acc_replay
                high_acc_abs_sum[idx_replay] += np.abs(acc_replay)
                high_acc_count[idx_replay] += 1
                high_acc_hard_brake[idx_replay] += (acc_replay <= -3.0).astype(np.int32)
                high_acc_hard_accel[idx_replay] += (acc_replay >= 3.0).astype(np.int32)

            low_reward_ext = compute_low_level_external_reward(
                reward_env,
                r_components,
                replay_mask,
                exclude_progress=(
                    str(getattr(self.cfg, "intrinsic_type", "l2")).lower()
                    == "huber_shaping"
                ),
            )

            safety_penalty_coef = float(getattr(self.cfg, "low_safety_violation_penalty", 0.0))
            safety_penalty = np.where(replay_mask & low_safety_clipped, safety_penalty_coef, 0.0).astype(np.float32)
            low_reward_ext = low_reward_ext - safety_penalty

            # calculate intrinsic reward
            regular_interval_end = (c == hi - 1) & (~queue_takeover_next_mask)
            low_transition_end = (
                regular_interval_end | done
            ) & low_replay_mask & (~queue_enter_mask)
            high_interval_end = (
                regular_interval_end | done | queue_exit_mask
            ) & replay_mask
            # Callback-facing name retained for compatibility: it denotes the
            # end of the current high/low option, not the low replay terminal.
            done_low = high_interval_end
            intrinsic, goal_err, intrinsic_unw = self._compute_intrinsic(
                kin,
                kin_next,
                goal_phys,
                ego_start,
                low_transition_end,
            )
            intrinsic[~low_replay_mask] = 0.0
            if low_transition_end.any():
                idx_last = np.flatnonzero(low_transition_end)
                goal_err_all[idx_last] = goal_err[idx_last]
                intrinsic_unweighted[idx_last] = intrinsic_unw[idx_last]

            low_reward_total = low_reward_ext + intrinsic
            if low_replay_mask.any():
                low_ret[low_replay_mask] += low_reward_total[low_replay_mask]
                low_len[low_replay_mask] += 1
            if replay_mask.any():
                high_len[replay_mask] += 1

            # record logs in callbacks
            for i, rc in enumerate(r_components):
                if not low_replay_mask[i]:
                    continue
                for name, val in rc.items():
                    if name in {"goal_lane_dense_reward", "punctual_reward", "wrong_lane_terminal_penalty"}:
                        continue
                    low_comp_sums.setdefault(name, np.zeros(n_envs, dtype=np.float32))[i] += float(val)
            if low_transition_end.any():
                low_comp_sums.setdefault("intrinsic_reward", np.zeros(n_envs, dtype=np.float32))[low_transition_end] += intrinsic[low_transition_end]
            if np.any((safety_penalty > 0.0) & low_replay_mask):
                low_comp_sums.setdefault("safety_violation_penalty", np.zeros(n_envs, dtype=np.float32))[low_replay_mask] += (-safety_penalty[low_replay_mask])

            # === 1.4 Low Level Store & Train ===
            next_low_obs = self._build_low_obs(c + 1, kin_flat_next, kin_next, goal_phys, next_obs_extra)
            if next_obs_is_dummy.any():
                next_low_obs[next_obs_is_dummy] = 0.0
            done_low_store = low_transition_end.astype(np.bool_)
            ego_now_sub = utils.extract_ego_substate(kin, self.ego_feature_idx)
            ego_next_sub = utils.extract_ego_substate(kin_next, self.ego_feature_idx)

            low_infos = [dict(info) for info in infos]
            if done_low_store.any():
                for i in np.flatnonzero(done_low_store):
                    low_infos[i]["terminal_observation"] = next_low_obs[i]

            if train_low and self.low_level_type == "sac":
                for i in np.flatnonzero(replay_mask):
                    low_infos[i]["low_seg_id"] = int(seg_id[i])
                    low_infos[i]["low_t_in_seg"] = int(c[i])
                    low_infos[i]["low_ep_id"] = int(ep_id[i])
                    low_infos[i]["low_ep_step"] = int(ep_step[i])
                    low_infos[i]["low_ego_start"] = np.asarray(ego_start[i], dtype=np.float32)
                    low_infos[i]["low_ego_now"] = np.asarray(ego_now_sub[i], dtype=np.float32)
                    low_infos[i]["low_ego_next"] = np.asarray(ego_next_sub[i], dtype=np.float32)
                    low_infos[i]["low_r_ext"] = float(low_reward_ext[i])

            committed_low_count = 0
            if use_pending_low:
                reward_for_store = (low_reward_ext if self.low_use_her else low_reward_total).astype(np.float32)
                for i in np.flatnonzero(low_replay_mask):
                    pending_low.append(
                        int(i),
                        PendingLowTransition(
                            obs=np.asarray(low_obs[i], dtype=np.float32).copy(),
                            action=np.asarray(low_buffer_action[i], dtype=np.float32).copy(),
                            next_obs=np.asarray(next_low_obs[i], dtype=np.float32).copy(),
                            reward=float(reward_for_store[i]),
                            done=bool(done_low_store[i]),
                            info=dict(low_infos[i]),
                        ),
                    )

                for i in np.flatnonzero(queue_enter_mask):
                    pending_low.discard(int(i))
                    low_ret[i] = 0.0
                    low_len[i] = 0
                    low_safety_clip_count[i] = 0
                    goal_err_all[i] = 0.0
                    intrinsic_unweighted[i] = 0.0
                    for values in low_comp_sums.values():
                        values[i] = 0.0

                commit_env_indices = np.flatnonzero(low_transition_end)
                if commit_env_indices.size > 0:
                    committed_low_count = _commit_pending_low(commit_env_indices)

                if committed_low_count > 0:
                    self.low_agent.train_from_committed(
                        committed_low_count,
                        n_envs=n_envs,
                    )
            elif train_low and self.low_level_type == "sac" and low_replay_mask.any():
                reward_for_store = (low_reward_ext if self.low_use_her else low_reward_total).astype(np.float32)
                for i in np.flatnonzero(~low_replay_mask):
                    low_infos[i]["skip_replay"] = True
                self.low_agent.store_transition(
                    low_obs,
                    low_buffer_action,
                    next_low_obs,
                    reward_for_store,
                    done_low_store,
                    low_infos,
                )
                self.low_agent.num_timesteps += int(np.sum(low_replay_mask))
                self.low_agent.train_if_needed()

            self.total_timesteps += replay_count
            target_reached = self.total_timesteps >= total_timesteps
            close_to_target = (total_timesteps - self.total_timesteps) <= n_envs
            dummy_tail_only = replay_count == 0 and close_to_target

            if replay_count > 0:
                callback.update_locals(locals())
                if callback.on_step() is False:
                    break

            # Track per-env episode timeline for HER future-step relabeling.
            ep_step += 1
            if done.any():
                idx_done_env = np.flatnonzero(done)
                ep_id[idx_done_env] += 1
                ep_step[idx_done_env] = 0

            # === 1.5. High Level Store & Train ===
            if high_interval_end.any():
                idx_end = np.flatnonzero(high_interval_end)
                low_ret_end = low_ret[idx_end].copy()
                low_len_end = low_len[idx_end].copy()
                low_safety_clip_ratio_end = (
                    low_safety_clip_count[idx_end].astype(np.float32)
                    / np.maximum(low_len_end.astype(np.float32), 1.0)
                )
                low_comp_end = {k: v[idx_end].copy() for k, v in low_comp_sums.items()}

                # goal tracking diagnostics for these finished low-episodes
                goal_err_end = goal_err_all[idx_end].copy()
                intrinsic_unweighted_end = intrinsic_unweighted[idx_end].copy()
                goal_dist_start_end = goal_dist_start[idx_end].copy()

                callback.update_locals({**locals(), "low_ret": low_ret_end, "low_len": low_len_end, "low_safety_clip_ratio": low_safety_clip_ratio_end, "low_comp_sums": low_comp_end, "goal_err": goal_err_end, "intrinsic_unweighted": intrinsic_unweighted_end, "goal_dist_start": goal_dist_start_end})
                callback.on_rollout_end()

                if train_high:
                    self.high_agent.num_timesteps += int(idx_end.size)
                    for j in idx_end:
                        info_h = dict(infos[j])
                        # SB3 off-policy store uses info["terminal_observation"] for done transitions;
                        # ensure it matches high-level observation shape instead of raw env obs shape.
                        if bool(done[j]):
                            info_h["terminal_observation"] = np.asarray(next_high_obs[j], dtype=np.float32).copy()
                        info_h["high_interval_len"] = int(high_len[j])
                        info_h["high_transition_discount"] = option_bootstrap_discount(
                            self.high_gamma,
                            int(high_len[j]),
                            hi,
                        )
                        info_h["queue_takeover_used"] = bool(
                            high_len[j] > low_len[j]
                        )
                        info_h["high_env_id"] = int(j)
                        info_h["high_global_step"] = int(self.total_timesteps)
                        info_h["high_segment_id"] = int(seg_id[j])
                        info_h["high_components"] = {
                            k: float(v[j])
                            for k, v in high_comp_sums.items()
                        }
                        acc_n = max(int(high_acc_count[j]), 1)
                        acc_min_j = float(high_acc_min[j]) if np.isfinite(high_acc_min[j]) else 0.0
                        acc_max_j = float(high_acc_max[j]) if np.isfinite(high_acc_max[j]) else 0.0
                        info_h["high_acc_stats"] = {
                            "acc_min": acc_min_j,
                            "acc_max": acc_max_j,
                            "acc_mean": float(high_acc_sum[j] / acc_n),
                            "acc_abs_mean": float(high_acc_abs_sum[j] / acc_n),
                            "hard_brake_frac": float(high_acc_hard_brake[j] / acc_n),
                            "hard_accel_frac": float(high_acc_hard_accel[j] / acc_n),
                        }
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
                high_len[idx_end] = 0
                low_safety_clip_count[idx_end] = 0
                for v in low_comp_sums.values():
                    v[idx_end] = 0.0
                for v in high_comp_sums.values():
                    v[idx_end] = 0.0
                high_acc_min[idx_end] = np.inf
                high_acc_max[idx_end] = -np.inf
                high_acc_sum[idx_end] = 0.0
                high_acc_abs_sum[idx_end] = 0.0
                high_acc_count[idx_end] = 0
                high_acc_hard_brake[idx_end] = 0
                high_acc_hard_accel[idx_end] = 0

                c[idx_end] = 0

            c[
                replay_mask
                & (~high_interval_end)
                & (~queue_takeover_next_mask)
            ] += 1
            if dummy_finished_mask.any():
                idx_dummy_done = np.flatnonzero(dummy_finished_mask)
                need_high[idx_dummy_done] = True
                c[idx_dummy_done] = 0
                high_ret[idx_dummy_done] = 0.0
                low_ret[idx_dummy_done] = 0.0
                low_len[idx_dummy_done] = 0
                high_len[idx_dummy_done] = 0
                low_safety_clip_count[idx_dummy_done] = 0
                for v in low_comp_sums.values():
                    v[idx_dummy_done] = 0.0
                for v in high_comp_sums.values():
                    v[idx_dummy_done] = 0.0
                high_acc_min[idx_dummy_done] = np.inf
                high_acc_max[idx_dummy_done] = -np.inf
                high_acc_sum[idx_dummy_done] = 0.0
                high_acc_abs_sum[idx_dummy_done] = 0.0
                high_acc_count[idx_dummy_done] = 0
                high_acc_hard_brake[idx_dummy_done] = 0
                high_acc_hard_accel[idx_dummy_done] = 0
            obs = next_obs
            obs_skip_mask = next_obs_is_dummy

            if target_reached or dummy_tail_only:
                if dummy_tail_only:
                    print(
                        "[HIROSAC] Stop near target without draining inter-episode dummy steps: "
                        f"env_steps={self.total_timesteps}, target={total_timesteps}"
                    )
                break
        
        print(f"[HIROSAC] Training loop finished, running callbacks: env_steps={self.total_timesteps}")
        callback.on_training_end()
        print(f"[HIROSAC] Training callbacks finished: env_steps={self.total_timesteps}")
        print(f"[HIROSAC] 训练结束: env_steps={self.total_timesteps}")
        return self
