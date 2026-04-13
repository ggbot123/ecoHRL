from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch as th


class HighGoalSafeBoundsCalculator:
    """Compute high-level goal safe bounds for 3-segment union boxes.

    This module intentionally keeps a placeholder reachable-interval hook.
    Replace `lane_reachable_x_interval()` with real business logic later.
    """

    def __init__(
        self,
        n_lanes: int = 3,
        lane_width: float = 4.0,
        high_interval: int = 25,
        dt: float = 0.1,
        speed_min: float = 0.0,
        speed_max: float = 15.0,
        max_accel: float = 5.0,
        max_decel: float = 5.0,
        front_dmin: float = 0.0,
        lane_change_rear_dmin: float = 0.0,
        min_goal_x_span: float = 0.0,
        dx_low: float = 0.0,
        dx_high: float = 37.5,
        feat_dim: int = 5,
        presence_idx: int = 0,
        x_idx: int = 1,
        y_idx: int = 2,
        vx_idx: int = 3,
        vy_idx: int = 4,
        enable_goal_vx_bounds: bool = True,
    ):
        self.n_lanes = int(max(1, n_lanes))
        self.lane_width = float(lane_width)
        self.high_interval = int(max(1, high_interval))
        self.dt = float(dt)
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.max_accel = float(max_accel)
        self.max_decel = float(max_decel)
        self.front_dmin = float(max(0.0, front_dmin))
        self.lane_change_rear_dmin = float(max(0.0, lane_change_rear_dmin))
        self.min_goal_x_span = float(max(0.0, min_goal_x_span))
        self.dx_low = float(dx_low)
        self.dx_high = float(dx_high)

        self.feat_dim = int(feat_dim)
        self.presence_idx = int(presence_idx)
        self.x_idx = int(x_idx)
        self.y_idx = int(y_idx)
        self.vx_idx = int(vx_idx)
        self.vy_idx = int(vy_idx)
        self.enable_goal_vx_bounds = bool(enable_goal_vx_bounds)
        # Hardcoded threshold for vy-based target-lane classification.
        self.lane_assign_vy_eps = 0.05
        self.lane_center_ys = (np.arange(self.n_lanes, dtype=np.float32) * self.lane_width).astype(np.float32)

    def lane_future_front_rear(self, high_obs_np: np.ndarray, lane_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Predict nearest front/rear vehicle longitudinal bounds for a target lane.

        Steps:
        1) Find front/rear vehicles in target lane from observation kinematics.
          Returns rear/front in ego-t0 frame (same frame as goal dx):
          - rear_dx: nearest rear vehicle longitudinal position at t+h
          - front_dx: nearest front vehicle longitudinal position at t+h

        NOTE: In HIRO high_obs used here, ego kinematics are absolute while neighboring
        vehicles are represented in ego-relative coordinates for x/y/vx/vy.
        """
        high_obs_np = np.asarray(high_obs_np, dtype=np.float32)
        batch = int(high_obs_np.shape[0])
        kin = self._extract_kinematics(high_obs_np)
        horizon_t = float(self.high_interval) * float(self.dt)

        ego_y = kin[:, 0, self.y_idx]
        ego_vx = kin[:, 0, self.vx_idx]
        ego_vy = kin[:, 0, self.vy_idx] if self.vy_idx < kin.shape[2] else np.zeros((batch,), dtype=np.float32)

        rear_dx = np.full((batch,), -1e9, dtype=np.float32)
        front_dx = np.full((batch,), 1e9, dtype=np.float32)

        n_veh = int(kin.shape[1])
        for j in range(1, n_veh):
            present = kin[:, j, self.presence_idx] > 0.5
            # Neighbor states are ego-relative in high_obs.
            veh_x_rel = kin[:, j, self.x_idx]
            veh_y_rel = kin[:, j, self.y_idx]
            veh_vx_rel = kin[:, j, self.vx_idx]
            veh_vy_rel = kin[:, j, self.vy_idx] if self.vy_idx < kin.shape[2] else np.zeros((batch,), dtype=np.float32)
            veh_vx_abs = veh_vx_rel + ego_vx
            veh_vy_abs = veh_vy_rel + ego_vy

            # Use absolute y for lane assignment.
            veh_y_abs = veh_y_rel + ego_y

            # Classify lane by lane-change direction when lateral velocity is significant:
            # if vy > eps -> target right lane; if vy < -eps -> target left lane.
            lane_idx_now = np.argmin(np.abs(veh_y_abs[:, None] - self.lane_center_ys[None, :]), axis=1)
            lane_delta = np.where(
                veh_vy_abs > self.lane_assign_vy_eps,
                1,
                np.where(veh_vy_abs < -self.lane_assign_vy_eps, -1, 0),
            )
            lane_idx = np.clip(lane_idx_now + lane_delta, 0, self.n_lanes - 1)
            in_lane = lane_idx == int(np.clip(lane_id, 0, self.n_lanes - 1))
            valid = present & in_lane
            if not np.any(valid):
                continue

            # Use future relative ordering at t+h (constant longitudinal velocities)
            # to decide front/rear membership.
            # rel_h = x_veh(t+h) - x_ego(t+h) = rel_x(t0) + rel_vx * h
            rel_h = veh_x_rel + veh_vx_rel * horizon_t
            is_front = rel_h >= 0.0

            # Keep bound values in ego-t0 frame (same frame as goal dx):
            # x_veh(t+h) - x_ego(t0) = rel_x(t0) + v_veh_abs * h
            rel = veh_x_rel + veh_vx_abs * horizon_t

            upd_front = valid & is_front
            upd_rear = valid & (~is_front)
            front_dx = np.where(upd_front, np.minimum(front_dx, rel), front_dx)
            rear_dx = np.where(upd_rear, np.maximum(rear_dx, rel), rear_dx)

        return rear_dx.astype(np.float32), front_dx.astype(np.float32)

    def _extract_kinematics(self, high_obs_np: np.ndarray) -> np.ndarray:
        kin_flat = np.asarray(high_obs_np[:, 1:], dtype=np.float32)
        total_dim = int(kin_flat.shape[1])
        if total_dim % self.feat_dim != 0:
            raise ValueError(
                f"high_obs kinematics dim {total_dim} is not divisible by feat_dim {self.feat_dim}"
            )
        n_veh = total_dim // self.feat_dim
        return kin_flat.reshape(high_obs_np.shape[0], n_veh, self.feat_dim)

    def _disp_const_accel_with_speed_cap(self, v0: np.ndarray, a: float, t: float) -> np.ndarray:
        v0 = np.asarray(v0, dtype=np.float32)
        if np.isclose(a, 0.0):
            return (v0 * t).astype(np.float32)

        if a > 0.0:
            t_cap = np.maximum((self.speed_max - v0) / a, 0.0)
            t1 = np.minimum(t, t_cap)
            s = v0 * t1 + 0.5 * a * (t1 ** 2)
            rem = np.maximum(t - t1, 0.0)
            s = s + self.speed_max * rem
            return s.astype(np.float32)

        t_cap = np.maximum((self.speed_min - v0) / a, 0.0)
        t1 = np.minimum(t, t_cap)
        s = v0 * t1 + 0.5 * a * (t1 ** 2)
        rem = np.maximum(t - t1, 0.0)
        s = s + self.speed_min * rem
        return s.astype(np.float32)

    def _ego_displacement_bounds(self, ego_vx: np.ndarray, horizon_t: float) -> Tuple[np.ndarray, np.ndarray]:
        s_low = self._disp_const_accel_with_speed_cap(ego_vx, -abs(self.max_decel), horizon_t)
        s_high = self._disp_const_accel_with_speed_cap(ego_vx, abs(self.max_accel), horizon_t)
        return s_low.astype(np.float32), s_high.astype(np.float32)

    def _ego_speed_bounds(self, ego_vx: np.ndarray, horizon_t: float) -> Tuple[np.ndarray, np.ndarray]:
        ego_vx = np.asarray(ego_vx, dtype=np.float32)
        v_low = np.clip(ego_vx - abs(self.max_decel) * horizon_t, self.speed_min, self.speed_max)
        v_high = np.clip(ego_vx + abs(self.max_accel) * horizon_t, self.speed_min, self.speed_max)
        return v_low.astype(np.float32), v_high.astype(np.float32)

    def _to_normalized_dx(self, dx: np.ndarray) -> np.ndarray:
        dx = np.asarray(dx, dtype=np.float32)
        denom = max(self.dx_high - self.dx_low, 1e-6)
        y = 2.0 * (dx - self.dx_low) / denom - 1.0
        return np.clip(y, -0.999999, 0.999999).astype(np.float32)

    def _to_normalized_vx(self, vx: np.ndarray) -> np.ndarray:
        vx = np.asarray(vx, dtype=np.float32)
        denom = max(self.speed_max - self.speed_min, 1e-6)
        y = 2.0 * (vx - self.speed_min) / denom - 1.0
        return np.clip(y, -0.999999, 0.999999).astype(np.float32)

    def _vx_bounds_from_distance_interval(
        self,
        v0: np.ndarray,
        disp_low: np.ndarray,
        disp_high: np.ndarray,
        horizon_t: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Infer terminal vx bounds induced by displacement interval [l, u].

        Piecewise profiles follow the requested construction:
        1) s0<u  : coast, then max-accelerate to hit u -> upper vx
        2) s0>u  : max-decelerate, then coast to hit u -> upper vx
        3) s0>l  : coast, then max-decelerate to hit l -> lower vx
        4) s0<l  : max-accelerate, then coast to hit l -> lower vx
        """
        v0 = np.asarray(v0, dtype=np.float32)
        l = np.asarray(disp_low, dtype=np.float32)
        u = np.asarray(disp_high, dtype=np.float32)

        T = float(max(horizon_t, 1e-6))
        a = float(max(abs(self.max_accel), 1e-6))
        d = float(max(abs(self.max_decel), 1e-6))
        s0 = v0 * T

        # Upper bound from distance upper limit u
        upper_case1 = s0 <= u
        du_plus = np.maximum(u - s0, 0.0)
        tau_u_1 = np.sqrt(np.maximum(2.0 * du_plus / a, 0.0))
        tau_u_1 = np.clip(tau_u_1, 0.0, T)
        v_up_1 = v0 + a * tau_u_1

        du_minus = np.maximum(s0 - u, 0.0)
        disc_u = np.maximum(T * T - 2.0 * du_minus / d, 0.0)
        tau_u_2 = T - np.sqrt(disc_u)
        tau_u_2 = np.clip(tau_u_2, 0.0, T)
        v_up_2 = v0 - d * tau_u_2

        v_up = np.where(upper_case1, v_up_1, v_up_2)

        # Lower bound from distance lower limit l
        lower_case3 = s0 >= l
        dl_minus = np.maximum(s0 - l, 0.0)
        tau_l_3 = np.sqrt(np.maximum(2.0 * dl_minus / d, 0.0))
        tau_l_3 = np.clip(tau_l_3, 0.0, T)
        v_lo_3 = v0 - d * tau_l_3

        dl_plus = np.maximum(l - s0, 0.0)
        disc_l = np.maximum(T * T - 2.0 * dl_plus / a, 0.0)
        tau_l_4 = T - np.sqrt(disc_l)
        tau_l_4 = np.clip(tau_l_4, 0.0, T)
        v_lo_4 = v0 + a * tau_l_4

        v_lo = np.where(lower_case3, v_lo_3, v_lo_4)

        # Numerical guard for rare branch boundary jitter.
        lo = np.minimum(v_lo, v_up)
        hi = np.maximum(v_lo, v_up)
        return lo.astype(np.float32), hi.astype(np.float32)

    def compute_np(self, high_obs_np: np.ndarray) -> Dict[str, np.ndarray]:
        """Build union-of-3-rectangles bounds in normalized action space."""
        high_obs_np = np.asarray(high_obs_np, dtype=np.float32)
        n = int(high_obs_np.shape[0])
        kin = self._extract_kinematics(high_obs_np)
        ego_y = kin[:, 0, self.y_idx]
        ego_vx = kin[:, 0, self.vx_idx]
        ego_lane_idx = np.argmin(np.abs(ego_y[:, None] - self.lane_center_ys[None, :]), axis=1)
        horizon_t = float(self.high_interval) * float(self.dt)
        s_min, s_max = self._ego_displacement_bounds(ego_vx, horizon_t)
        if self.enable_goal_vx_bounds:
            v_dyn_min, v_dyn_max = self._ego_speed_bounds(ego_vx, horizon_t)
        else:
            v_dyn_min, v_dyn_max = None, None

        # Lane-conditioned interval for rel_x dimension (action dim 0).
        # Mixture component order is relative semantics: [left, keep, right].
        l2 = np.zeros((n, 3), dtype=np.float32)
        u2 = np.zeros((n, 3), dtype=np.float32)
        if self.enable_goal_vx_bounds:
            l_vx = np.zeros((n, 3), dtype=np.float32)
            u_vx = np.zeros((n, 3), dtype=np.float32)
        else:
            l_vx = None
            u_vx = None

        # Precompute absolute-lane nearest front/rear predictions once, then remap
        # per-sample to relative components [left, keep, right].
        abs_rear = np.zeros((n, self.n_lanes), dtype=np.float32)
        abs_front = np.zeros((n, self.n_lanes), dtype=np.float32)
        for lane_id in range(self.n_lanes):
            rear_dx, front_dx = self.lane_future_front_rear(high_obs_np, lane_id)
            abs_rear[:, lane_id] = rear_dx
            abs_front[:, lane_id] = front_dx

        rel_offsets = np.asarray([-1, 0, 1], dtype=np.int32)
        for comp_idx, rel_off in enumerate(rel_offsets):
            target_lane = ego_lane_idx + int(rel_off)
            valid_lane = (target_lane >= 0) & (target_lane < self.n_lanes)

            # Out-of-road direction is explicitly infeasible for this component.
            lane_clamped = np.clip(target_lane, 0, self.n_lanes - 1)
            front_dx = np.where(valid_lane, abs_front[np.arange(n), lane_clamped], 1e9)
            rear_dx = np.where(valid_lane, abs_rear[np.arange(n), lane_clamped], -1e9)

            # Front safety margin applies to all components.
            hi = np.minimum(front_dx - self.front_dmin, s_max)

            # Rear safety margin applies only when changing lane.
            if int(rel_off) == 0:
                lo = s_min
            else:
                lo = np.maximum(s_min, rear_dx + self.lane_change_rear_dmin)

            if self.enable_goal_vx_bounds:
                v_lo_x, v_hi_x = self._vx_bounds_from_distance_interval(ego_vx, lo, hi, horizon_t)
                v_lo = np.maximum(v_dyn_min, v_lo_x)
                v_hi = np.minimum(v_dyn_max, v_hi_x)
            else:
                v_lo = None
                v_hi = None

            lo_n = self._to_normalized_dx(lo)
            hi_n = self._to_normalized_dx(hi)
            if self.enable_goal_vx_bounds:
                v_lo_n = self._to_normalized_vx(v_lo)
                v_hi_n = self._to_normalized_vx(v_hi)
            else:
                v_lo_n = None
                v_hi_n = None

            # Keep empty interval explicit: l2 > u2 means infeasible for that component.
            # Also discard very narrow reachable intervals to avoid unstable sampling.
            span = hi - lo
            empty = (~valid_lane) | (lo >= hi) | (span < self.min_goal_x_span)
            if self.enable_goal_vx_bounds:
                empty = empty | (v_lo >= v_hi)
            lo_n = np.where(empty, 1.0, lo_n)
            hi_n = np.where(empty, -1.0, hi_n)
            if self.enable_goal_vx_bounds:
                v_lo_n = np.where(empty, 1.0, v_lo_n)
                v_hi_n = np.where(empty, -1.0, v_hi_n)
            l2[:, comp_idx] = lo_n
            u2[:, comp_idx] = hi_n
            if self.enable_goal_vx_bounds:
                l_vx[:, comp_idx] = v_lo_n
                u_vx[:, comp_idx] = v_hi_n

        out: Dict[str, np.ndarray] = {"l2": l2, "u2": u2}
        if self.enable_goal_vx_bounds:
            out["l_vx"] = l_vx
            out["u_vx"] = u_vx
        return out

    def compute_torch(self, high_obs_t: th.Tensor) -> Dict[str, th.Tensor]:
        bounds_np = self.compute_np(high_obs_t.detach().cpu().numpy())
        out = {
            "l2": th.as_tensor(bounds_np["l2"], dtype=high_obs_t.dtype, device=high_obs_t.device),
            "u2": th.as_tensor(bounds_np["u2"], dtype=high_obs_t.dtype, device=high_obs_t.device),
        }
        if self.enable_goal_vx_bounds:
            out["l_vx"] = th.as_tensor(bounds_np["l_vx"], dtype=high_obs_t.dtype, device=high_obs_t.device)
            out["u_vx"] = th.as_tensor(bounds_np["u_vx"], dtype=high_obs_t.dtype, device=high_obs_t.device)
        return out
