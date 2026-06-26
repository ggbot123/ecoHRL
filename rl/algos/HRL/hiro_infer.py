import numpy as np
from typing import Tuple, Optional, Callable, Any
from rl.algos.sac.sac import SAC
from rl.utils import utils
from rl.algos.HRL.rule_based import RuleBasedAgentWrapper, RuleBasedController
from rl.algos.HRL.high_goal_safe_bounds import HighGoalSafeBoundsCalculator
from configs.builders import get_hiro_config


class _SingleEnvAdapter:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space

    def get_attr(self, name, indices=None):
        return [getattr(self.env.unwrapped, name)]

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
        config: Optional[Any] = None,
    ):
        self.high_model, self.low_model, self.hi = high_model, low_model, int(high_interval)
        self.high_policy = high_policy
        self.cfg = config if config is not None else get_hiro_config()
        self.use_low_safety_layer = bool(use_low_safety_layer) if use_low_safety_layer is not None else bool(getattr(self.cfg, "use_low_safety_layer", False))
        self._inited = False
        self.need_high, self.c = True, 0
        self.n_veh, self.feat_dim, self.feature_names, self.ego_feature_idx, self.ego_dim = 0, 0, [], [], 0
        self.n_veh_local, self.kin_flat_dim, self.local_kin_flat_dim, self.obs_extra_dim = 0, 0, 0, 0
        self.high_obs_extra_dim = 0
        self.high_obs_include_signal_features = True
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
        self.low_safety_agent: Optional[RuleBasedAgentWrapper] = None
        self.rule_based_agent: Optional[RuleBasedAgentWrapper] = None
        self.last_action_pre_safety = np.zeros(0, dtype=np.float32)
        self.last_action_post_safety = np.zeros(0, dtype=np.float32)
        self.last_goal_action = np.zeros(0, dtype=np.float32)
        self.last_goal_action_raw = np.zeros(0, dtype=np.float32)
        self.last_goal_shielded = False
        self.high_goal_safe_bounds: Optional[HighGoalSafeBoundsCalculator] = None
        self.punctual_time_target = 0.0
        self.goal_longitudinal_default = 0.0
        self.queue_takeover_enabled = False
        self._last_action_queue_takeover = False

    def init_from_env(self, env, obs0: np.ndarray, intrinsic_coef: float):
        keep = ("x", "y", "vx", "vy")
        n_veh, n_veh_local, feat_dim, feature_names, ego_idx = utils.init_kinematics_meta(env, obs0, keep)
        self.n_veh, self.feat_dim, self.feature_names, self.ego_feature_idx = int(n_veh), int(feat_dim), list(feature_names), list(ego_idx)
        self.n_veh_local = int(n_veh_local)
        self.kin_flat_dim = self.n_veh * self.feat_dim
        self.local_kin_flat_dim = self.n_veh_local * self.feat_dim
        self.obs_extra_dim = int(max(0, int(np.asarray(obs0, dtype=np.float32).reshape(-1).shape[0]) - (1 + self.kin_flat_dim)))
        self.high_obs_extra_dim = int(self.obs_extra_dim)
        self.high_obs_include_signal_features = True
        model_obs_shape = getattr(getattr(self.high_model, "observation_space", None), "shape", None)
        model_obs_dim = int(model_obs_shape[0]) if model_obs_shape else 0
        if model_obs_dim > 0:
            base_dim = int(1 + self.kin_flat_dim)
            extra_plus_signal = int(model_obs_dim - base_dim)
            prefer_signal = bool(getattr(self.cfg, "high_obs_use_signal_features", True))
            layouts = [
                (int(self.obs_extra_dim + 2), (int(self.obs_extra_dim), True)),
                (int(self.obs_extra_dim), (int(self.obs_extra_dim), False)),
                (2, (0, True)),
                (0, (0, False)),
            ]
            if not prefer_signal:
                layouts = [layouts[1], layouts[3], layouts[0], layouts[2]]
            matched_layout = next(
                (layout for dim, layout in layouts if dim == extra_plus_signal),
                None,
            )
            if matched_layout is None:
                raise ValueError(
                    "High-level observation dimension mismatch: "
                    f"model expects {model_obs_dim}, env base={base_dim}, "
                    f"env extra={self.obs_extra_dim}. Supported layouts are "
                    "base, base+signal, base+extra, or base+extra+signal."
                )
            self.high_obs_extra_dim, self.high_obs_include_signal_features = matched_layout
        self.ego_dim = int(len(self.ego_feature_idx))
        cfg = getattr(env.unwrapped, "config", getattr(env, "config", {}))
        self.punctual_time_target = float(cfg.get("punctual_time_target", cfg.get("duration", 0.0)))
        self.goal_longitudinal_default = float(cfg.get("goal_longitudinal", cfg.get("road_length", 0.0)))
        self.queue_takeover_enabled = bool(cfg.get("enable_queue_takeover", False))
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

        if self.use_low_safety_layer and self.low_safety_agent is None:
            self.low_safety_agent = RuleBasedAgentWrapper(
                _SingleEnvAdapter(env),
                n_envs=1,
                high_interval=int(self.hi),
                low_safety_filter=getattr(self.cfg, "low_safety_filter", None),
            )
            self.safety_controller = self.low_safety_agent.controller
        if (
            str(getattr(self.cfg, "low_level_type", "sac")).lower() == "rule_based"
            and self.rule_based_agent is None
        ):
            if self.low_safety_agent is not None:
                self.rule_based_agent = self.low_safety_agent
            else:
                self.rule_based_agent = RuleBasedAgentWrapper(
                    _SingleEnvAdapter(env),
                    n_envs=1,
                    high_interval=int(self.hi),
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
        fixed_goal_vx = getattr(self.cfg, "fixed_goal_vx", None)
        if fixed_goal_vx is not None:
            fixed_goal_vx = float(np.clip(float(fixed_goal_vx), v_min, v_max))
        enable_goal_vx_bounds = bool(getattr(self.cfg, "high_goal_safe_enable_goal_vx_bounds", True))
        if fixed_goal_vx is not None and np.isclose(float(fixed_goal_vx), 0.0):
            enable_goal_vx_bounds = False
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
            use_idm_dynamic_margins=bool(getattr(self.cfg.high_goal_safety, "use_idm_dynamic_margins", False)),
            front_standstill_dmin=float(max(0.0, getattr(self.cfg.high_goal_safety, "front_standstill_dmin", 8.0))),
            rear_standstill_dmin=float(max(0.0, getattr(self.cfg.high_goal_safety, "rear_standstill_dmin", 6.0))),
            idm_time_headway=float(max(0.0, getattr(self.cfg.high_goal_safety, "idm_time_headway", 0.5))),
            idm_accel=float(max(1e-6, getattr(self.cfg.high_goal_safety, "idm_accel", 3.0))),
            idm_decel=float(max(1e-6, getattr(self.cfg.high_goal_safety, "idm_decel", 5.0))),
            rear_imposed_decel=float(max(0.0, getattr(self.cfg.high_goal_safety, "rear_imposed_decel", 4.0))),
            min_goal_x_span=float(max(0.0, getattr(self.cfg, "high_goal_safe_min_goal_x_span", 0.0))),
            dx_low=float(dx_low),
            dx_high=float(dx_high),
            feat_dim=int(self.feat_dim),
            n_veh=int(self.n_veh),
            presence_idx=int(_idx("presence", 0)),
            x_idx=int(self.idx_x),
            y_idx=int(self.idx_y),
            vx_idx=int(self.idx_vx),
            vy_idx=int(self.idx_vy),
            enable_goal_vx_bounds=bool(enable_goal_vx_bounds),
        )

        # In training we bind high-goal safe bounds explicitly. During standalone
        # inference we must rebind it after model deserialization.
        high_actor = getattr(self.high_model, "actor", None)
        use_high_goal_safety = bool(
            getattr(self.cfg, "use_high_goal_safety_layer", False)
        )
        if use_high_goal_safety and (
            high_actor is None
            or not hasattr(high_actor, "goal_safe_sampling_enabled")
        ):
            raise RuntimeError(
                "The saved high-level model does not support the configured "
                "high-goal safety layer"
            )
        if high_actor is not None and hasattr(high_actor, "goal_safe_sampling_enabled"):
            if hasattr(high_actor, "dynamic_feasible_lane_intervals"):
                high_actor.dynamic_feasible_lane_intervals = bool(
                    getattr(self.cfg, "high_goal_dynamic_feasible_lane_intervals", False)
                )
            need_bind_bounds = bool(getattr(high_actor, "goal_safe_bounds_fn", None) is None)
            if need_bind_bounds and use_high_goal_safety:
                high_actor.goal_safe_eps = float(getattr(self.cfg, "high_goal_safe_eps", 1e-6))
                high_actor.goal_safe_bounds_fn = self.high_goal_safe_bounds.compute_torch
                high_actor.goal_safe_sampling_enabled = True
            elif need_bind_bounds:
                high_actor.goal_safe_sampling_enabled = False
            if hasattr(high_actor, "infeasible_action_mode"):
                high_actor.infeasible_action_mode = (
                    "preserve"
                    if self._high_goal_infeasible_mode() == "shield_penalty"
                    else "reroute"
                )

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
        self._last_action_queue_takeover = False

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

    def _high_goal_infeasible_mode(self) -> str:
        mode = str(getattr(self.cfg, "high_goal_infeasible_action_mode", "reroute")).lower().strip()
        if mode in {"shield", "shield_penalty", "penalty", "fallback"}:
            return "shield_penalty"
        return "reroute"

    def _goal_component_index(self, action: np.ndarray, ego_lane_idx: int, n_lanes: int) -> int:
        y = float(np.asarray(action, dtype=np.float32).reshape(-1)[1])
        if not bool(getattr(self.cfg, "high_goal_dynamic_feasible_lane_intervals", False)):
            if y <= -1.0 / 3.0:
                return 0
            if y < 1.0 / 3.0:
                return 1
            return 2
        if int(n_lanes) <= 1:
            return 1
        if int(ego_lane_idx) == 0:
            return 2 if y > 0.0 else 1
        if int(ego_lane_idx) == int(n_lanes) - 1:
            return 0 if y < 0.0 else 1
        if y <= -1.0 / 3.0:
            return 0
        if y < 1.0 / 3.0:
            return 1
        return 2

    def _shield_goal_action_if_needed(self, high_obs: np.ndarray, goal_action: np.ndarray) -> tuple[np.ndarray, bool]:
        if (
            self.high_goal_safe_bounds is None
            or self._high_goal_infeasible_mode() != "shield_penalty"
            or not bool(getattr(self.cfg, "use_high_goal_safety_layer", False))
        ):
            return goal_action, False
        action = np.asarray(goal_action, dtype=np.float32).reshape(1, -1)
        stats = self.high_goal_safe_bounds.compute_np(np.asarray(high_obs, dtype=np.float32).reshape(1, -1))
        l2 = np.asarray(stats["l2"], dtype=np.float32)
        u2 = np.asarray(stats["u2"], dtype=np.float32)
        valid = u2 > l2
        if action.shape[1] >= 3 and "l_vx" in stats and "u_vx" in stats:
            valid = valid & (np.asarray(stats["u_vx"], dtype=np.float32) > np.asarray(stats["l_vx"], dtype=np.float32))

        lane_idx = int(np.asarray(stats.get("ego_lane_idx", [0]), dtype=np.int64).reshape(-1)[0])
        n_lanes = int(np.asarray(stats.get("n_lanes", len(self.lane_center_ys)), dtype=np.int64).reshape(-1)[0])
        comp = self._goal_component_index(action[0], lane_idx, n_lanes)
        if bool(valid[0, comp]):
            return action, False

        low = np.asarray(self.high_model.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.high_model.action_space.high, dtype=np.float32).reshape(-1)
        fallback = action.copy()
        y0, y1 = utils.semantic_y_interval(
            1,
            lane_idx,
            n_lanes,
            bool(getattr(self.cfg, "high_goal_dynamic_feasible_lane_intervals", False)),
        )
        fallback[0, 1] = float(np.clip(0.5 * (y0 + y1), low[1], high[1]))
        if bool(valid[0, 1]):
            denom = max(float(high[0] - low[0]), 1e-6)
            x_norm = np.clip(2.0 * (fallback[0, 0] - low[0]) / denom - 1.0, l2[0, 1], u2[0, 1])
            fallback[0, 0] = float(np.clip(low[0] + 0.5 * (x_norm + 1.0) * (high[0] - low[0]), low[0], high[0]))
        else:
            fallback[0, 0] = low[0]
            if fallback.shape[1] >= 3:
                fallback[0, 2] = float(np.clip(0.0, low[2], high[2]))
        return fallback.astype(np.float32), True

    def _sample_goal(self, obs: np.ndarray, kin: np.ndarray, env=None):
        ego_sub = self._ego_sub(kin)
        high_obs = self._build_high_obs(np.asarray(obs, dtype=np.float32), env)
        if self.high_policy is not None:
            goal_action = self.high_policy(high_obs)
        else:
            goal_action, _ = self.high_model.predict(high_obs, deterministic=True)
        goal_action = np.asarray(goal_action, dtype=np.float32).reshape(1, -1)
        self.last_goal_action_raw = np.asarray(goal_action[0], dtype=np.float32).copy()
        goal_action, shielded = self._shield_goal_action_if_needed(high_obs, goal_action)
        self.last_goal_action = np.asarray(goal_action[0], dtype=np.float32).copy()
        self.last_goal_shielded = bool(shielded)
        goal_phys = utils.goal_action_to_abs(
            ego_sub[None, :],
            goal_action,
            self.lane_center_ys,
            dynamic_feasible_intervals=bool(
                getattr(self.cfg, "high_goal_dynamic_feasible_lane_intervals", False)
            ),
        )
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

    def _get_punctual_time_target(self, env) -> float:
        if env is None:
            return float(self.punctual_time_target)
        base = getattr(env, "unwrapped", env)
        fn = getattr(base, "get_punctual_time_target", None)
        if callable(fn):
            try:
                return float(fn())
            except Exception:
                return float(self.punctual_time_target)
        return float(self.punctual_time_target)

    def _queue_takeover_active(self, env) -> bool:
        if not self.queue_takeover_enabled or env is None:
            return False
        base = getattr(env, "unwrapped", env)
        fn = getattr(base, "get_queue_takeover_active", None)
        return bool(fn()) if callable(fn) else False

    def _queue_takeover_action(self, env) -> np.ndarray:
        base = getattr(env, "unwrapped", env)
        fn = getattr(base, "get_queue_takeover_action", None)
        if not callable(fn):
            raise RuntimeError("Queue takeover is active but the environment has no controller action")
        return np.asarray(fn(), dtype=np.float32).reshape(-1)

    def _build_high_obs(self, obs: np.ndarray, env=None) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        time_mode = str(getattr(self.cfg, "high_obs_time_mode", "remaining")).lower().strip()
        if time_mode == "remaining":
            punctual_target = self._get_punctual_time_target(env)
            high_t = (punctual_target - arr[:, :1]).astype(np.float32)
        elif time_mode == "elapsed":
            high_t = arr[:, :1].astype(np.float32)
        else:
            raise ValueError("high_obs_time_mode must be 'remaining' or 'elapsed'")

        kin_flat = np.asarray(arr[:, 1 : 1 + self.kin_flat_dim], dtype=np.float32).copy()
        extra = np.asarray(arr[:, 1 + self.kin_flat_dim : 1 + self.kin_flat_dim + self.obs_extra_dim], dtype=np.float32)
        x_mode = str(getattr(self.cfg, "high_obs_x_mode", "remaining")).lower().strip()
        if x_mode == "remaining":
            ego_x = float(kin_flat[0, self.idx_x])
            goal_x = self._get_goal_longitudinal(env)
            kin_flat[0, self.idx_x] = float(goal_x - ego_x)
        elif x_mode != "elapsed":
            raise ValueError("high_obs_x_mode must be 'remaining' or 'elapsed'")

        parts = [high_t, kin_flat]
        if int(getattr(self, "high_obs_extra_dim", self.obs_extra_dim)) > 0:
            parts.append(extra[:, : int(self.high_obs_extra_dim)])
        if bool(getattr(self, "high_obs_include_signal_features", True)):
            parts.append(self._get_signal_features(env).reshape(1, 2))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def act(self, env, obs: np.ndarray) -> np.ndarray:
        self._last_action_queue_takeover = False
        _, kin, kin_flat = self._split(obs)
        if self.need_high:
            self._sample_goal(obs, kin, env)

        ego_sub = self._ego_sub(kin)
        t_norm = np.array([self.c / float(self.hi)], dtype=np.float32)
        goal_rel = (self.goal_phys - ego_sub).astype(np.float32)
        
        local_kin_flat = np.asarray(kin_flat[0, :self.local_kin_flat_dim], dtype=np.float32).copy()
        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        extra = obs_arr[1 + self.kin_flat_dim : 1 + self.kin_flat_dim + self.obs_extra_dim].astype(np.float32)

        # Keep inference low_obs consistent with training: mask ego absolute position (x/y).
        low_level_type = str(getattr(self.cfg, "low_level_type", "sac")).lower()
        if (
            low_level_type == "sac"
            and bool(getattr(self.cfg, "mask_ego_position_in_low_obs", False))
        ):
            if int(self.feat_dim) > 0 and local_kin_flat.shape[0] >= int(self.feat_dim):
                idx_x = int(self.feature_names.index("x"))
                idx_y = int(self.feature_names.index("y"))
                local_kin_flat[idx_x] = 0.0
                local_kin_flat[idx_y] = 0.0
        low_obs = np.concatenate([t_norm, local_kin_flat, extra, goal_rel]).astype(np.float32)

        if self._queue_takeover_active(env):
            action = self._queue_takeover_action(env)
            self._last_action_queue_takeover = True
            self.last_action_pre_safety = action.copy()
            self.last_action_post_safety = action.copy()
            return action

        if low_level_type == "rule_based":
            if self.rule_based_agent is None:
                raise RuntimeError("Rule-based low-level agent is not initialized")
            low_obs_batch = low_obs.reshape(1, -1)
            goal_batch = self.goal_phys.reshape(1, -1)
            action_raw = self.rule_based_agent.act(low_obs_batch, goal_batch)[0]
            action = self.rule_based_agent.apply_safety_layer(
                low_obs_batch,
                goal_batch,
                np.asarray(action_raw, dtype=np.float32).reshape(1, -1),
            )[0]
            self.last_action_pre_safety = np.asarray(action_raw, dtype=np.float32).copy()
            self.last_action_post_safety = np.asarray(action, dtype=np.float32).copy()
            return np.asarray(action, dtype=np.float32)
        if low_level_type != "sac":
            raise ValueError(f"Unknown low_level_type: {low_level_type}")
        if self.low_model is None:
            # Goal-only evaluation scripts use the runner to update goal_phys
            # and compute their own primitive action.
            self.last_action_pre_safety = np.zeros(0, dtype=np.float32)
            self.last_action_post_safety = np.zeros(0, dtype=np.float32)
            return np.zeros(0, dtype=np.float32)

        action, _ = self.low_model.predict(low_obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        self.last_action_pre_safety = action.copy()

        if self.use_low_safety_layer and self.low_safety_agent is not None:
            action = self.low_safety_agent.apply_safety_layer(
                low_obs.reshape(1, -1),
                self.goal_phys.reshape(1, -1),
                action.reshape(1, -1),
            )[0]
        self.last_action_post_safety = np.asarray(action, dtype=np.float32).copy()
        return np.asarray(action, dtype=np.float32)

    def intrinsic_if_last(self, obs_next: np.ndarray) -> float:
        _, kin_next, _ = self._split(obs_next)
        ego_next = self._ego_sub(kin_next)
        ego_rel = (ego_next - self.ego_start).astype(np.float32)
        goal_rel = (self.goal_phys - self.ego_start).astype(np.float32)
        r, _, _ = utils.intrinsic_reward_l2(ego_rel[None, :], goal_rel[None, :], self.norm_ranges, self.intrinsic_coef, self.weights)
        return float(np.asarray(r, dtype=np.float32).reshape(-1)[0])

    def step_end(self, done: bool, queue_takeover_active: bool = False):
        queue_released = self._last_action_queue_takeover and not queue_takeover_active
        if not queue_takeover_active:
            self.c += 1
        if done or queue_released or (not queue_takeover_active and self.c >= self.hi):
            self.need_high, self.c = True, 0
