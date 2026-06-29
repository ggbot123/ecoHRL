from __future__ import annotations

import os
from statistics import NormalDist
from dataclasses import dataclass
from typing import Callable, Optional, Union

import gymnasium as gym
import numpy as np

from rl.utils.utils import semantic_y_interval


@dataclass(frozen=True)
class GoalSamplerConfig:
    type: str = "uniform"  # "uniform" | "reachable_uniform" | "reachable_gaussian" | "reachable_cruise_mix" | "speed_near_cruise" | "pretrained" | "pretrained_sac"
    path: Optional[str] = None
    device: str = "auto"
    deterministic: bool = True
    action: Optional[list[float]] = None  # used when type == "fixed"
    gaussian_mean_x_m: float = 27.0  # used when type == "reachable_gaussian"
    gaussian_half_range_m: float = 5.0  # used when type == "reachable_gaussian"
    enable_vx_bounds: bool = True  # used by reachable samplers
    cruise_horizon_s: float = 2.5  # used by cruise-centered samplers
    cruise_keep_prob: float = 0.6  # used by reachable_cruise_mix
    cruise_accel_prob: float = 0.2  # used by reachable_cruise_mix
    cruise_decel_prob: float = 0.2  # used by reachable_cruise_mix
    cruise_window_m: float = 2.0  # used by reachable_cruise_mix
    cruise_window_mapping: str = "symmetric"  # "symmetric" | "balanced_reachable"

class GoalSampler:
    """Base class for goal samplers."""
    def __init__(self, action_space: gym.spaces.Box):
        self.action_space = action_space

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class UniformGoalSampler(GoalSampler):
    """Samples goals uniformly from the high-level action space."""
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        n = obs.shape[0]
        low = self.action_space.low
        high = self.action_space.high
        # Uniform sample in [low, high]
        return np.random.uniform(low, high, size=(n, low.shape[0])).astype(np.float32)


class ReachableUniformGoalSampler(GoalSampler):
    """Sample goals from reachable set with uniform segment selection.

    Segment (lane choice) is sampled uniformly over feasible components, then x/y are
    sampled uniformly inside the selected component bounds.
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        bounds_fn: Callable[[np.ndarray], dict[str, np.ndarray]],
        enable_vx_bounds: bool = True,
        dynamic_feasible_intervals: bool = False,
    ):
        super().__init__(action_space)
        self._bounds_fn = bounds_fn
        self._enable_vx_bounds = bool(enable_vx_bounds)
        self._dynamic_feasible_intervals = bool(dynamic_feasible_intervals)

    @staticmethod
    def _sample_component_uniform(valid_mask: np.ndarray) -> np.ndarray:
        """Sample one component index per row uniformly among feasible components."""
        k = np.zeros((valid_mask.shape[0],), dtype=np.int64)
        for i in range(valid_mask.shape[0]):
            candidates = np.flatnonzero(valid_mask[i])
            k[i] = int(np.random.choice(candidates))
        return k

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        n = int(obs.shape[0])
        act_dim = int(np.prod(self.action_space.shape))

        low = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)

        # Start from full-space uniform for unconstrained dims and fallback rows.
        actions = np.random.uniform(low, high, size=(n, act_dim)).astype(np.float32)

        stats = self._bounds_fn(obs)
        l2 = np.asarray(stats["l2"], dtype=np.float32)
        u2 = np.asarray(stats["u2"], dtype=np.float32)
        if self._enable_vx_bounds and ("l_vx" in stats and "u_vx" in stats):
            l_vx = np.asarray(stats.get("l_vx"), dtype=np.float32)
            u_vx = np.asarray(stats.get("u_vx"), dtype=np.float32)
        else:
            l_vx = None
            u_vx = None

        valid = u2 > l2
        valid_any = np.any(valid, axis=1)

        if np.any(valid_any):
            idx = np.flatnonzero(valid_any)
            k = self._sample_component_uniform(valid[idx])

            lane_idx = np.asarray(stats.get("ego_lane_idx", np.zeros(n)), dtype=np.int64).reshape(-1)
            n_lanes = int(np.asarray(stats.get("n_lanes", 3)).reshape(-1)[0])
            use_dynamic = self._dynamic_feasible_intervals and {
                "ego_lane_idx",
                "n_lanes",
            }.issubset(stats)
            y_intervals = np.asarray(
                [
                    semantic_y_interval(
                        int(comp),
                        int(lane_idx[obs_idx]),
                        n_lanes,
                        use_dynamic,
                    )
                    for obs_idx, comp in zip(idx, k)
                ],
                dtype=np.float32,
            )
            yl = y_intervals[:, 0]
            yh = y_intervals[:, 1]
            xl = l2[idx, k]
            xh = u2[idx, k]

            y_code = yl + np.random.rand(idx.size).astype(np.float32) * (yh - yl)
            x_norm = xl + np.random.rand(idx.size).astype(np.float32) * (xh - xl)

            # dim-1 is already in env range [-1, 1].
            actions[idx, 1] = y_code

            # dim-0 bounds are in normalized space [-1, 1], map back to env action space.
            if act_dim >= 1:
                denom = max(float(high[0] - low[0]), 1e-6)
                x_env = low[0] + 0.5 * (x_norm + 1.0) * denom
                actions[idx, 0] = np.clip(x_env, low[0], high[0])

            # dim-2 (goal vx) bounds are optional normalized intervals in [-1, 1].
            if act_dim >= 3 and l_vx is not None and u_vx is not None:
                vl = l_vx[idx, k]
                vh = u_vx[idx, k]
                v_valid = vh > vl
                if np.any(v_valid):
                    ii = idx[v_valid]
                    vv_l = vl[v_valid]
                    vv_h = vh[v_valid]
                    v_norm = vv_l + np.random.rand(ii.size).astype(np.float32) * (vv_h - vv_l)
                    denom_v = max(float(high[2] - low[2]), 1e-6)
                    v_env = low[2] + 0.5 * (v_norm + 1.0) * denom_v
                    actions[ii, 2] = np.clip(v_env, low[2], high[2])

        return actions.astype(np.float32)


class ReachableGaussianGoalSampler(GoalSampler):
    """Sample goals from reachable set with x following truncated Gaussian mass.

    x is sampled from N(mu=27m, sigma) with sigma chosen such that
    P(|X - mu| <= 5m) = 0.7, then truncated and renormalized on the selected
    reachable segment interval. Segment (lane choice) is sampled uniformly over
    feasible components, and y is sampled uniformly in the chosen segment code range.
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        bounds_fn: Callable[[np.ndarray], dict[str, np.ndarray]],
        mean_x_m: float = 27.0,
        half_range_m: float = 5.0,
        enable_vx_bounds: bool = True,
        dynamic_feasible_intervals: bool = False,
    ):
        super().__init__(action_space)
        self._bounds_fn = bounds_fn
        self._mu_x_m = float(mean_x_m)
        self._enable_vx_bounds = bool(enable_vx_bounds)
        self._dynamic_feasible_intervals = bool(dynamic_feasible_intervals)
        half = float(max(half_range_m, 1e-6))
        # Keep 70% probability mass inside [mu-half, mu+half].
        z = float(NormalDist().inv_cdf(0.85))
        sigma = max(half / max(z, 1e-6), 1e-6)
        self._gauss = NormalDist(mu=self._mu_x_m, sigma=sigma)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        n = int(obs.shape[0])
        act_dim = int(np.prod(self.action_space.shape))

        low = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)

        # Fallback values for unconstrained dims and rows with no feasible component.
        actions = np.random.uniform(low, high, size=(n, act_dim)).astype(np.float32)
        if act_dim < 2:
            return actions

        stats = self._bounds_fn(obs)
        l2 = np.asarray(stats["l2"], dtype=np.float32)
        u2 = np.asarray(stats["u2"], dtype=np.float32)
        if self._enable_vx_bounds and ("l_vx" in stats and "u_vx" in stats):
            l_vx = np.asarray(stats.get("l_vx"), dtype=np.float32)
            u_vx = np.asarray(stats.get("u_vx"), dtype=np.float32)
        else:
            l_vx = None
            u_vx = None

        valid = u2 > l2
        denom = max(float(high[0] - low[0]), 1e-6)
        lane_idx = np.asarray(stats.get("ego_lane_idx", np.zeros(n)), dtype=np.int64).reshape(-1)
        n_lanes = int(np.asarray(stats.get("n_lanes", 3)).reshape(-1)[0])
        use_dynamic = self._dynamic_feasible_intervals and {
            "ego_lane_idx",
            "n_lanes",
        }.issubset(stats)

        for i in range(n):
            candidates = np.flatnonzero(valid[i])
            if candidates.size == 0:
                continue

            xl_norm = l2[i, candidates]
            xh_norm = u2[i, candidates]
            xl_env = low[0] + 0.5 * (xl_norm + 1.0) * denom
            xh_env = low[0] + 0.5 * (xh_norm + 1.0) * denom

            # Keep segment selection uniform among feasible components.
            cidx = int(np.random.randint(0, candidates.size))
            k = int(candidates[cidx])

            lo = float(min(xl_env[cidx], xh_env[cidx]))
            hi = float(max(xl_env[cidx], xh_env[cidx]))
            p_lo = self._gauss.cdf(lo)
            p_hi = self._gauss.cdf(hi)
            dp = max(p_hi - p_lo, 0.0)

            if dp <= 1e-15:
                x_env = float(np.clip(self._mu_x_m, lo, hi))
            else:
                p = p_lo + float(np.random.rand()) * dp
                p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
                x_env = float(self._gauss.inv_cdf(p))
                x_env = float(np.clip(x_env, lo, hi))

            y_low, y_high = semantic_y_interval(
                k,
                int(lane_idx[i]),
                n_lanes,
                use_dynamic,
            )
            y_code = float(y_low + np.random.rand() * (y_high - y_low))
            actions[i, 0] = np.float32(np.clip(x_env, low[0], high[0]))
            actions[i, 1] = np.float32(y_code)
            if act_dim >= 3 and l_vx is not None and u_vx is not None:
                v_lo_n = float(l_vx[i, k])
                v_hi_n = float(u_vx[i, k])
                if v_hi_n > v_lo_n:
                    v_norm = float(v_lo_n + np.random.rand() * (v_hi_n - v_lo_n))
                    denom_v = max(float(high[2] - low[2]), 1e-6)
                    v_env = float(low[2] + 0.5 * (v_norm + 1.0) * denom_v)
                    actions[i, 2] = np.float32(np.clip(v_env, low[2], high[2]))

        return actions.astype(np.float32)


class ReachableCruiseMixGoalSampler(GoalSampler):
    """Reachable goal sampler centered on constant-speed displacement.

    For each sampled component, x is drawn around x_cruise = ego_speed * horizon.
    The keep/accel/decel modes use symmetric offsets around x_cruise whenever both
    sides are reachable, so equal accel/decel probabilities preserve the expected
    displacement near constant speed while still exposing harder targets.
    """

    def __init__(
        self,
        action_space: gym.spaces.Box,
        bounds_fn: Callable[[np.ndarray], dict[str, np.ndarray]],
        speed_fn: Callable[[np.ndarray], np.ndarray],
        *,
        horizon_s: float = 2.5,
        keep_prob: float = 0.6,
        accel_prob: float = 0.2,
        decel_prob: float = 0.2,
        window_m: float = 2.0,
        window_mapping: str = "symmetric",
        enable_vx_bounds: bool = True,
        dynamic_feasible_intervals: bool = False,
    ):
        super().__init__(action_space)
        self._bounds_fn = bounds_fn
        self._speed_fn = speed_fn
        self._horizon_s = float(max(horizon_s, 0.0))
        probs = np.asarray([keep_prob, accel_prob, decel_prob], dtype=np.float32)
        probs = np.maximum(probs, 0.0)
        total = float(np.sum(probs))
        if total <= 0.0:
            probs[:] = np.asarray([0.6, 0.2, 0.2], dtype=np.float32)
            total = float(np.sum(probs))
        self._mode_probs = (probs / total).astype(np.float32)
        self._window_m = float(max(window_m, 0.0))
        self._window_mapping = str(window_mapping).lower().strip()
        self._enable_vx_bounds = bool(enable_vx_bounds)
        self._dynamic_feasible_intervals = bool(dynamic_feasible_intervals)

    def _sample_keep_x(self, cruise_x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return float(np.clip(cruise_x, lo, hi))

        left = max(float(cruise_x) - float(lo), 0.0)
        right = max(float(hi) - float(cruise_x), 0.0)
        if self._window_m <= 0.0 or (left <= 1e-6 and right <= 1e-6):
            return float(np.clip(cruise_x, lo, hi))

        left = min(left, self._window_m)
        right = min(right, self._window_m)

        if self._window_mapping in {"balanced_reachable", "reachable_balanced", "balanced"}:
            # Preserve E[x] ~= cruise_x while using asymmetric reachable spans.
            if left <= 1e-6 or right <= 1e-6:
                return float(np.clip(cruise_x, lo, hi))
            p_left = right / max(left + right, 1e-6)
            if float(np.random.rand()) < p_left:
                return float(cruise_x - np.random.rand() * left)
            return float(cruise_x + np.random.rand() * right)

        if self._window_mapping in {"reachable", "stretch", "stretch_reachable"}:
            if float(np.random.rand()) < 0.5:
                return float(cruise_x - np.random.rand() * left)
            return float(cruise_x + np.random.rand() * right)

        half = min(left, right)
        if half <= 1e-6:
            return float(np.clip(cruise_x, lo, hi))
        return float(cruise_x + np.random.uniform(-half, half))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        n = int(obs.shape[0])
        act_dim = int(np.prod(self.action_space.shape))

        low = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
        actions = np.random.uniform(low, high, size=(n, act_dim)).astype(np.float32)
        if act_dim < 2:
            return actions

        stats = self._bounds_fn(obs)
        l2 = np.asarray(stats["l2"], dtype=np.float32)
        u2 = np.asarray(stats["u2"], dtype=np.float32)
        if self._enable_vx_bounds and ("l_vx" in stats and "u_vx" in stats):
            l_vx = np.asarray(stats.get("l_vx"), dtype=np.float32)
            u_vx = np.asarray(stats.get("u_vx"), dtype=np.float32)
        else:
            l_vx = None
            u_vx = None

        valid = u2 > l2
        valid_any = np.any(valid, axis=1)
        if not np.any(valid_any):
            return actions.astype(np.float32)

        speed = np.asarray(self._speed_fn(obs), dtype=np.float32).reshape(-1)
        if speed.shape[0] != n:
            raise ValueError(f"speed_fn output size mismatch: got {speed.shape[0]}, expected {n}")

        denom_x = max(float(high[0] - low[0]), 1e-6)
        lane_idx = np.asarray(stats.get("ego_lane_idx", np.zeros(n)), dtype=np.int64).reshape(-1)
        n_lanes = int(np.asarray(stats.get("n_lanes", 3)).reshape(-1)[0])
        use_dynamic = self._dynamic_feasible_intervals and {
            "ego_lane_idx",
            "n_lanes",
        }.issubset(stats)

        idx = np.flatnonzero(valid_any)
        modes = np.random.choice(3, size=idx.size, p=self._mode_probs)

        for row_i, mode in zip(idx, modes):
            candidates = np.flatnonzero(valid[row_i])
            if candidates.size == 0:
                continue

            lo_all = low[0] + 0.5 * (l2[row_i, candidates].astype(np.float32) + 1.0) * denom_x
            hi_all = low[0] + 0.5 * (u2[row_i, candidates].astype(np.float32) + 1.0) * denom_x
            lo_all, hi_all = np.minimum(lo_all, hi_all), np.maximum(lo_all, hi_all)
            cruise_raw = float(speed[row_i]) * self._horizon_s

            if int(mode) == 1:
                mode_ok = hi_all > cruise_raw + 1e-6
            elif int(mode) == 2:
                mode_ok = lo_all < cruise_raw - 1e-6
            else:
                mode_ok = np.ones_like(hi_all, dtype=bool)
            eligible_idx = np.flatnonzero(mode_ok)
            if eligible_idx.size == 0:
                eligible_idx = np.arange(candidates.size)

            choice = int(np.random.choice(eligible_idx))
            comp = int(candidates[choice])
            lo = float(lo_all[choice])
            hi = float(hi_all[choice])

            cruise_x = float(np.clip(cruise_raw, lo, hi))
            shared_span = max(0.0, min(cruise_x - lo, hi - cruise_x))

            if shared_span <= 1e-6:
                x_env = cruise_x
            elif int(mode) == 0:
                x_env = self._sample_keep_x(cruise_x, lo, hi)
            else:
                offset = float(np.random.uniform(0.0, shared_span))
                x_env = cruise_x + offset if int(mode) == 1 else cruise_x - offset

            y_low, y_high = semantic_y_interval(
                comp,
                int(lane_idx[row_i]),
                n_lanes,
                use_dynamic,
            )
            y_code = float(y_low + np.random.rand() * (y_high - y_low))
            actions[row_i, 0] = np.float32(np.clip(x_env, low[0], high[0]))
            actions[row_i, 1] = np.float32(y_code)

            if act_dim >= 3 and l_vx is not None and u_vx is not None:
                v_lo_n = float(l_vx[row_i, comp])
                v_hi_n = float(u_vx[row_i, comp])
                if v_hi_n > v_lo_n:
                    v_norm = float(v_lo_n + np.random.rand() * (v_hi_n - v_lo_n))
                    denom_v = max(float(high[2] - low[2]), 1e-6)
                    v_env = float(low[2] + 0.5 * (v_norm + 1.0) * denom_v)
                    actions[row_i, 2] = np.float32(np.clip(v_env, low[2], high[2]))

        return actions.astype(np.float32)


class SpeedNearCruiseGoalSampler(GoalSampler):
    """Sample goals with x concentrated near ego cruise distance (v * horizon)."""

    def __init__(
        self,
        action_space: gym.spaces.Box,
        speed_fn: Callable[[np.ndarray], np.ndarray],
        horizon_s: float = 2.5,
        x_window: float = 2.0,
    ):
        super().__init__(action_space)
        self._speed_fn = speed_fn
        self._horizon_s = float(max(horizon_s, 0.0))
        self._x_window = float(max(x_window, 0.0))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        n = int(obs.shape[0])
        low = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
        act_dim = int(low.size)

        actions = np.random.uniform(low, high, size=(n, act_dim)).astype(np.float32)
        if act_dim < 1:
            return actions

        speed = np.asarray(self._speed_fn(obs), dtype=np.float32).reshape(-1)
        if speed.shape[0] != n:
            raise ValueError(f"speed_fn output size mismatch: got {speed.shape[0]}, expected {n}")

        target_x = speed * self._horizon_s
        x_low = np.maximum(low[0], target_x - self._x_window)
        x_high = np.minimum(high[0], target_x + self._x_window)
        valid = x_high > x_low

        if np.any(valid):
            idx = np.flatnonzero(valid)
            rand = np.random.rand(idx.size).astype(np.float32)
            actions[idx, 0] = x_low[idx] + rand * (x_high[idx] - x_low[idx])

        if np.any(~valid):
            idx = np.flatnonzero(~valid)
            actions[idx, 0] = np.clip(target_x[idx], low[0], high[0]).astype(np.float32)

        return actions.astype(np.float32)


class PretrainedSACGoalSampler(GoalSampler):
    """Samples goals using a pretrained high-level SAC policy."""

    def __init__(self, action_space: gym.spaces.Box, model_path: str, device: str = "auto", deterministic: bool = True):
        super().__init__(action_space)
        if not model_path:
            raise ValueError("model_path is required for pretrained goal sampler")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pretrained goal sampler model not found: {model_path}")

        from rl.algos.sac.sac import SAC

        self._model = SAC.load(model_path, device=device)
        self._deterministic = bool(deterministic)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        actions, _ = self._model.predict(obs, deterministic=self._deterministic)
        return np.asarray(actions, dtype=np.float32)


class FixedGoalSampler(GoalSampler):
    """Always returns a fixed goal action (broadcasted to batch)."""

    def __init__(self, action_space: gym.spaces.Box, action: list[float]):
        super().__init__(action_space)
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        expected_dim = int(np.prod(self.action_space.shape))
        if a.size != expected_dim:
            raise ValueError(f"Fixed goal action dim mismatch: got {a.size}, expected {expected_dim}")
        low = np.asarray(self.action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.action_space.high, dtype=np.float32).reshape(-1)
        self._action = np.clip(a, low, high).astype(np.float32)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        n = int(np.asarray(obs).shape[0])
        return np.repeat(self._action[None, :], n, axis=0)

def get_goal_sampler(
    cfg: Union[GoalSamplerConfig, str],
    action_space: gym.spaces.Box,
    bounds_fn: Optional[Callable[[np.ndarray], dict[str, np.ndarray]]] = None,
    speed_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    enable_vx_bounds: Optional[bool] = None,
    dynamic_feasible_lane_intervals: bool = False,
) -> GoalSampler:
    if isinstance(cfg, str):
        cfg = GoalSamplerConfig(type=cfg)

    sampler_type = str(cfg.type).lower()
    use_vx_bounds = bool(cfg.enable_vx_bounds if enable_vx_bounds is None else enable_vx_bounds)
    if sampler_type == "uniform":
        return UniformGoalSampler(action_space)
    if sampler_type == "reachable_uniform":
        if bounds_fn is None:
            raise ValueError("bounds_fn is required for type='reachable_uniform'")
        return ReachableUniformGoalSampler(
            action_space,
            bounds_fn=bounds_fn,
            enable_vx_bounds=use_vx_bounds,
            dynamic_feasible_intervals=dynamic_feasible_lane_intervals,
        )
    if sampler_type in {"reachable_gaussian", "gaussian_reachable", "reachable_trunc_gaussian"}:
        if bounds_fn is None:
            raise ValueError("bounds_fn is required for type='reachable_gaussian'")
        return ReachableGaussianGoalSampler(
            action_space,
            bounds_fn=bounds_fn,
            mean_x_m=float(cfg.gaussian_mean_x_m),
            half_range_m=float(cfg.gaussian_half_range_m),
            enable_vx_bounds=use_vx_bounds,
            dynamic_feasible_intervals=dynamic_feasible_lane_intervals,
        )
    if sampler_type in {"reachable_cruise_mix", "cruise_mix", "reachable_speed_mix"}:
        if bounds_fn is None:
            raise ValueError("bounds_fn is required for type='reachable_cruise_mix'")
        if speed_fn is None:
            raise ValueError("speed_fn is required for type='reachable_cruise_mix'")
        return ReachableCruiseMixGoalSampler(
            action_space,
            bounds_fn=bounds_fn,
            speed_fn=speed_fn,
            horizon_s=float(cfg.cruise_horizon_s),
            keep_prob=float(cfg.cruise_keep_prob),
            accel_prob=float(cfg.cruise_accel_prob),
            decel_prob=float(cfg.cruise_decel_prob),
            window_m=float(cfg.cruise_window_m),
            window_mapping=str(cfg.cruise_window_mapping),
            enable_vx_bounds=use_vx_bounds,
            dynamic_feasible_intervals=dynamic_feasible_lane_intervals,
        )
    if sampler_type in {"speed_near_cruise", "near_cruise", "cruise_nearby"}:
        if speed_fn is None:
            raise ValueError("speed_fn is required for type='speed_near_cruise'")
        return SpeedNearCruiseGoalSampler(action_space, speed_fn=speed_fn)
    if sampler_type in {"fixed", "constant"}:
        if cfg.action is None:
            raise ValueError("GoalSamplerConfig.action is required for type='fixed'")
        return FixedGoalSampler(action_space, action=list(cfg.action))
    if sampler_type in {"pretrained", "pretrained_sac"}:
        return PretrainedSACGoalSampler(
            action_space,
            model_path=cfg.path or "",
            device=str(cfg.device),
            deterministic=bool(cfg.deterministic),
        )
    raise ValueError(f"Unknown goal sampler type: {sampler_type}")
