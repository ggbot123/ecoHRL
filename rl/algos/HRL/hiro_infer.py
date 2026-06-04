import numpy as np
from typing import Tuple, Optional, Callable
from rl.algos.sac.sac import SAC
from rl.utils import utils
from rl.algos.HRL.rule_based import RuleBasedController
from rl.algos.HRL.high_goal_safe_bounds import HighGoalSafeBoundsCalculator
from configs.conf import get_hiro_config

class HIROPolicyRunner:
    """Single-env HIRO inference runner.

    - High-level: sample goal every `high_interval` env steps.
    - Low-level: sample primitive action every env step.
    - Maintains per-interval state needed for intrinsic reward logging.
    """

    def __init__(
        self,
        high_model: SAC,
        low_model: Optional[SAC],
        high_interval: int,
        use_low_safety_layer: Optional[bool] = None,
        high_policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.high_model, self.low_model, self.hi = high_model, low_model, int(high_interval)
        self.high_policy = high_policy
        self.cfg = get_hiro_config()
        self.use_low_safety_layer = bool(use_low_safety_layer) if use_low_safety_layer is not None else bool(getattr(self.cfg, "use_low_safety_layer", False))
        self._inited = False
        self.need_high, self.c = True, 0
        self.n_veh, self.feat_dim, self.feature_names, self.ego_feature_idx, self.ego_dim = 0, 0, [], [], 0
        self.n_veh_local, self.kin_flat_dim, self.local_kin_flat_dim, self.obs_extra_dim = 0, 0, 0, 0
        self.lane_center_ys = np.zeros(0, dtype=np.float32)
        self.goal_phys = np.zeros(0, dtype=np.float32)
        self.ego_start = np.zeros(0, dtype=np.float32)
        self.norm_ranges: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.intrinsic_coef = 1.0
        self.dt = 0.0
        self.idx_presence = -1
        self.idx_x = 0
        self.idx_y = 1
        self.idx_vx = 2
        self.idx_vy = 3
        self.safety_controller: Optional[RuleBasedController] = None
        self.last_action_pre_safety = np.zeros(0, dtype=np.float32)
        self.last_action_post_safety = np.zeros(0, dtype=np.float32)
        self.last_goal_action = np.zeros(0, dtype=np.float32)
        self.high_goal_safe_bounds: Optional[HighGoalSafeBoundsCalculator] = None
        self.punctual_time_target = 0.0
        self.goal_longitudinal_default = 0.0

    def init_from_env(self, env, obs0: np.ndarray, intrinsic_coef: float):
        keep = ("x", "y", "vx", "vy")
        n_veh, n_veh_local, feat_dim, feature_names, ego_idx = utils.init_kinematics_meta(env, obs0, keep)
        self.n_veh, self.feat_dim, self.feature_names, self.ego_feature_idx = int(n_veh), int(feat_dim), list(feature_names), list(ego_idx)
        self.n_veh_local = int(n_veh_local)
        self.kin_flat_dim = self.n_veh * self.feat_dim
        self.local_kin_flat_dim = self.n_veh_local * self.feat_dim
        self.obs_extra_dim = int(max(0, int(np.asarray(obs0, dtype=np.float32).reshape(-1).shape[0]) - (1 + self.kin_flat_dim)))
        self.ego_dim = int(len(self.ego_feature_idx))
        cfg = getattr(env.unwrapped, "config", getattr(env, "config", {}))
        self.punctual_time_target = float(cfg.get("punctual_time_target", cfg.get("duration", 0.0)))
        self.goal_longitudinal_default = float(cfg.get("goal_longitudinal", cfg.get("road_length", 0.0)))
        lanes, lane_w = int(cfg["lanes_count"]), float(cfg.get("lane_width", 4.0))
        self.lane_center_ys = (np.arange(lanes, dtype=np.float32) * lane_w).astype(np.float32)
        self.dt = 1.0 / float(cfg.get("policy_frequency", 10.0))

        def _idx(name: str, default: int) -> int:
            try:
                return int(self.feature_names.index(name))
            except ValueError:
                return int(default)

        self.idx_presence = _idx("presence", -1)
        self.idx_x = _idx("x", 0)
        self.idx_y = _idx("y", 1)
        self.idx_vx = _idx("vx", 2)
        self.idx_vy = _idx("vy", 3)

        if self.use_low_safety_layer and self.safety_controller is None:
            self.safety_controller = RuleBasedController(
                cfg,
                low_safety_filter=getattr(self.cfg, "low_safety_filter", None),
            )

        action_cfg = cfg.get("action", {}) if isinstance(cfg, dict) else {}
        accel_range = action_cfg.get("acceleration_range", [-5.0, 5.0])
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

        t_h = float(self.hi) * float(self.dt)
        v_min = 0.0
        v_max = float(cfg.get("speed_limit", 15.0)) if isinstance(cfg, dict) else 15.0
        dx_low = float(v_min * t_h)
        dx_high = float(v_max * t_h)
        self.high_goal_safe_bounds = HighGoalSafeBoundsCalculator(
            n_lanes=int(cfg.get("lanes_count", len(self.lane_center_ys))) if isinstance(cfg, dict) else int(len(self.lane_center_ys)),
            lane_width=float(cfg.get("lane_width", 4.0)) if isinstance(cfg, dict) else 4.0,
            high_interval=int(self.hi),
            dt=float(self.dt),
            speed_min=float(v_min),
            speed_max=float(v_max),
            max_accel=float(max_accel),
            max_decel=float(max_decel),
            front_dmin=float(max(0.0, getattr(self.cfg, "high_goal_safe_front_dmin", 0.0))),
            lane_change_rear_dmin=float(max(0.0, getattr(self.cfg, "high_goal_safe_lane_change_rear_dmin", 0.0))),
            min_goal_x_span=float(max(0.0, getattr(self.cfg, "high_goal_safe_min_goal_x_span", 0.0))),
            dx_low=float(dx_low),
            dx_high=float(dx_high),
            feat_dim=int(self.feat_dim),
            presence_idx=int(_idx("presence", 0)),
            x_idx=int(self.idx_x),
            y_idx=int(self.idx_y),
            vx_idx=int(self.idx_vx),
            vy_idx=int(self.idx_vy),
        )

        # In training we bind high-goal safe bounds explicitly. During standalone
        # inference we must rebind it after model deserialization.
        high_actor = getattr(self.high_model, "actor", None)
        if high_actor is not None and hasattr(high_actor, "goal_safe_sampling_enabled"):
            need_bind_bounds = bool(getattr(high_actor, "goal_safe_bounds_fn", None) is None)
            if need_bind_bounds and bool(getattr(self.cfg, "use_high_goal_safety_layer", False)):
                high_actor.goal_safe_eps = float(getattr(self.cfg, "high_goal_safe_eps", 1e-6))
                high_actor.goal_safe_bounds_fn = self.high_goal_safe_bounds.compute_torch
                high_actor.goal_safe_sampling_enabled = True
            elif need_bind_bounds:
                # Fallback for legacy checkpoints: disable safe sampling when
                # no bounds function can be reconstructed.
                high_actor.goal_safe_sampling_enabled = False

        intrinsic_norm = getattr(self.cfg, "intrinsic_norm_ranges", None)
        self.norm_ranges = np.asarray(intrinsic_norm, dtype=np.float32)

        w = getattr(self.cfg, "intrinsic_weights", None)
        self.weights = None if w is None else np.asarray(w, dtype=np.float32)
        self.goal_phys = np.zeros(self.ego_dim, dtype=np.float32)
        self.ego_start = np.zeros(self.ego_dim, dtype=np.float32)
        self.intrinsic_coef = float(intrinsic_coef)
        self._inited = True

    def reset(self, env, obs0: np.ndarray, intrinsic_coef: float):
        if not self._inited:
            self.init_from_env(env, obs0, intrinsic_coef)
        self.need_high, self.c = True, 0

    def _split(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        arr = np.asarray(obs, dtype=np.float32)
        t, kin, kin_flat = utils.split_time_kinematics(arr[None, :], self.n_veh, self.feat_dim)
        return t, kin, kin_flat

    def _ego_sub(self, kin: np.ndarray) -> np.ndarray:
        return utils.extract_ego_substate(kin, self.ego_feature_idx)[0]

    def _extract_ego_others(self, kin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ego_feat = kin[0, 0]
        ego_abs = np.array(
            [
                ego_feat[self.idx_x],
                ego_feat[self.idx_y],
                ego_feat[self.idx_vx],
                ego_feat[self.idx_vy],
            ],
            dtype=np.float32,
        )
        others_feat = kin[0, 1:]
        others_rel = []
        for j in range(int(others_feat.shape[0])):
            d = others_feat[j]
            if self.idx_presence >= 0 and d[self.idx_presence] == 0:
                continue
            others_rel.append([float(d[self.idx_x]), float(d[self.idx_y]), float(d[self.idx_vx]), float(d[self.idx_vy])])
        others_rel_arr = np.asarray(others_rel, dtype=np.float32).reshape(-1, 4)
        return ego_abs, others_rel_arr

    def _sample_goal(self, obs: np.ndarray, kin: np.ndarray, env=None):
        ego_sub = self._ego_sub(kin)
        high_obs = self._build_high_obs(np.asarray(obs, dtype=np.float32), env)
        if self.high_policy is not None:
            goal_action = self.high_policy(high_obs)
        else:
            goal_action, _ = self.high_model.predict(high_obs, deterministic=True)
        goal_action = np.asarray(goal_action, dtype=np.float32).reshape(1, -1)
        self.last_goal_action = np.asarray(goal_action[0], dtype=np.float32).copy()
        goal_phys = utils.goal_action_to_abs(ego_sub[None, :], goal_action, self.lane_center_ys)
        self.goal_phys = np.asarray(goal_phys, dtype=np.float32).reshape(-1)
        self.ego_start = ego_sub.astype(np.float32, copy=True)
        self.need_high, self.c = False, 0
        
        # Update env with goal for rendering
        if env is not None:
            # Handle RecordVideo wrapper or other wrappers
            unwrapped = env.unwrapped
            if hasattr(unwrapped, "set_hiro_goal"):
                unwrapped.set_hiro_goal(self.goal_phys)

    def _get_signal_features(self, env) -> np.ndarray:
        if not bool(getattr(self.cfg, "high_obs_use_signal_features", True)):
            return np.array([-1.0, -1.0], dtype=np.float32)
        if env is None:
            return np.array([-1.0, -1.0], dtype=np.float32)
        base = getattr(env, "unwrapped", env)
        fn = getattr(base, "get_hiro_signal_features", None)
        if callable(fn):
            try:
                color, remain = fn()
                return np.array([float(color), float(remain)], dtype=np.float32)
            except Exception:
                return np.array([-1.0, -1.0], dtype=np.float32)
        return np.array([-1.0, -1.0], dtype=np.float32)

    def _get_goal_longitudinal(self, env) -> float:
        if env is None:
            return float(self.goal_longitudinal_default)
        base = getattr(env, "unwrapped", env)
        fn = getattr(base, "_goal_longitudinal", None)
        if callable(fn):
            try:
                return float(fn())
            except Exception:
                return float(self.goal_longitudinal_default)
        return float(self.goal_longitudinal_default)

    def _build_high_obs(self, obs: np.ndarray, env=None) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        t_remaining = (float(self.punctual_time_target) - arr[:, :1]).astype(np.float32)
        kin_flat = np.asarray(arr[:, 1 : 1 + self.kin_flat_dim], dtype=np.float32).copy()
        extra = np.asarray(arr[:, 1 + self.kin_flat_dim : 1 + self.kin_flat_dim + self.obs_extra_dim], dtype=np.float32)
        ego_x = float(kin_flat[0, self.idx_x])
        goal_x = self._get_goal_longitudinal(env)
        kin_flat[0, self.idx_x] = float(goal_x - ego_x)
        signal = self._get_signal_features(env).reshape(1, 2)
        return np.concatenate([t_remaining, kin_flat, extra, signal], axis=1).astype(np.float32)

    def act(self, env, obs: np.ndarray) -> np.ndarray:
        _, kin, kin_flat = self._split(obs)
        if self.need_high:
            self._sample_goal(obs, kin, env)

        if self.low_model is None:
            # Some evaluation scripts call runner.act() only to update self.goal_phys.
            # In that case, allow low_model to be absent and return a dummy action.
            self.last_action_pre_safety = np.zeros(0, dtype=np.float32)
            self.last_action_post_safety = np.zeros(0, dtype=np.float32)
            return np.zeros(0, dtype=np.float32)

        ego_sub = self._ego_sub(kin)
        t_norm = np.array([self.c / float(self.hi)], dtype=np.float32)
        goal_rel = (self.goal_phys - ego_sub).astype(np.float32)
        
        local_kin_flat = np.asarray(kin_flat[0, :self.local_kin_flat_dim], dtype=np.float32).copy()
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        extra = obs_arr[1 + self.kin_flat_dim : 1 + self.kin_flat_dim + self.obs_extra_dim].astype(np.float32)

        # Keep inference low_obs consistent with training: mask ego absolute position (x/y).
        if bool(getattr(self.cfg, "mask_ego_position_in_low_obs", False)):
            if int(self.feat_dim) > 0 and local_kin_flat.shape[0] >= int(self.feat_dim):
                idx_x = int(self.feature_names.index("x"))
                idx_y = int(self.feature_names.index("y"))
                local_kin_flat[idx_x] = 0.0
                local_kin_flat[idx_y] = 0.0
        low_obs = np.concatenate([t_norm, local_kin_flat, extra, goal_rel]).astype(np.float32)
        
        action, _ = self.low_model.predict(low_obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        self.last_action_pre_safety = action.copy()

        if self.use_low_safety_layer and self.safety_controller is not None:
            ego_abs, others_rel = self._extract_ego_others(kin)
            rem_time = float(self.hi - self.c) * float(self.dt)
            action = self.safety_controller.safety_filter_action(
                ego_abs,
                others_rel,
                self.goal_phys,
                action,
                self.dt,
                remaining_time=rem_time,
            )
        self.last_action_post_safety = np.asarray(action, dtype=np.float32).copy()
        return np.asarray(action, dtype=np.float32)

    def intrinsic_if_last(self, obs_next: np.ndarray) -> float:
        _, kin_next, _ = self._split(obs_next)
        ego_next = self._ego_sub(kin_next)
        ego_rel = (ego_next - self.ego_start).astype(np.float32)
        goal_rel = (self.goal_phys - self.ego_start).astype(np.float32)
        r, _, _ = utils.intrinsic_reward_l2(ego_rel[None, :], goal_rel[None, :], self.norm_ranges, self.intrinsic_coef, self.weights)
        return float(np.asarray(r, dtype=np.float32).reshape(-1)[0])

    def step_end(self, done: bool):
        self.c += 1
        if done or self.c >= self.hi:
            self.need_high, self.c = True, 0
