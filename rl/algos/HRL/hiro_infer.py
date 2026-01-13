import numpy as np
from typing import Tuple, Optional
from rl.algos.sac.sac import SAC
from rl.utils import utils

class HIROPolicyRunner:
    """Single-env HIRO inference runner.

    - High-level: sample goal every `high_interval` env steps.
    - Low-level: sample primitive action every env step.
    - Maintains per-interval state needed for intrinsic reward logging.
    """

    def __init__(self, high_model: SAC, low_model: Optional[SAC], high_interval: int):
        self.high_model, self.low_model, self.hi = high_model, low_model, int(high_interval)
        self._inited = False
        self.need_high, self.c = True, 0
        self.n_veh, self.feat_dim, self.feature_names, self.ego_feature_idx, self.ego_dim = 0, 0, [], [], 0
        self.n_veh_local, self.local_kin_flat_dim = 0, 0
        self.lane_center_ys = np.zeros(0, dtype=np.float32)
        self.goal_phys = np.zeros(0, dtype=np.float32)
        self.ego_start = np.zeros(0, dtype=np.float32)

        # dx_max is set from env config in init_from_env (speed_limit * high_interval * dt)
        self.norm_ranges = np.asarray([[0.0, 1.0], [-8.0, 8.0], [-8.0, 8.0], [-2.0, 2.0]], dtype=np.float32)
        self.weights = np.asarray([1.0, 2.0, 8.0, 1.0], dtype=np.float32)
        self.intrinsic_coef = 1.0

    def init_from_env(self, env, obs0: np.ndarray, intrinsic_coef: float):
        keep = ("x", "y", "vx", "vy")
        n_veh, n_veh_local, feat_dim, feature_names, ego_idx = utils.init_kinematics_meta(env, obs0, keep)
        self.n_veh, self.feat_dim, self.feature_names, self.ego_feature_idx = int(n_veh), int(feat_dim), list(feature_names), list(ego_idx)
        self.n_veh_local = int(n_veh_local)
        self.local_kin_flat_dim = self.n_veh_local * self.feat_dim
        self.ego_dim = int(len(self.ego_feature_idx))
        cfg = getattr(env.unwrapped, "config", getattr(env, "config", {}))
        lanes, lane_w = int(cfg["lanes_count"]), float(cfg.get("lane_width", 4.0))
        self.lane_center_ys = (np.arange(lanes, dtype=np.float32) * lane_w).astype(np.float32)

        v_max = float(cfg.get("speed_limit", 15.0))
        dt = 1.0 / float(cfg.get("policy_frequency", 10.0))
        self.norm_ranges[0, 1] = max(v_max * float(self.hi) * dt, 1e-6)
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

    def _sample_goal(self, obs: np.ndarray, kin: np.ndarray, env=None):
        ego_sub = self._ego_sub(kin)
        goal_action, _ = self.high_model.predict(np.asarray(obs, dtype=np.float32), deterministic=True)
        goal_action = np.asarray(goal_action, dtype=np.float32).reshape(1, -1)
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

    def act(self, env, obs: np.ndarray) -> np.ndarray:
        _, kin, kin_flat = self._split(obs)
        if self.need_high:
            self._sample_goal(obs, kin, env)

        if self.low_model is None:
            # Some evaluation scripts call runner.act() only to update self.goal_phys.
            # In that case, allow low_model to be absent and return a dummy action.
            return np.zeros(0, dtype=np.float32)

        ego_sub = self._ego_sub(kin)
        t_norm = np.array([self.c / float(self.hi)], dtype=np.float32)
        goal_rel = (self.goal_phys - ego_sub).astype(np.float32)
        
        local_kin_flat = kin_flat[0, :self.local_kin_flat_dim]
        low_obs = np.concatenate([t_norm, local_kin_flat, goal_rel]).astype(np.float32)
        
        action, _ = self.low_model.predict(low_obs, deterministic=True)
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
