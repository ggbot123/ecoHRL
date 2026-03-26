from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import numpy as np
import cvxpy as cp  # type: ignore[import-not-found]
from rl.utils import utils as hiro_utils


class MPCController:
    """
    QP-compatible MPC with explicit analytic objective/constraints.

    Design choices to keep the optimization strictly QP-form:
    - Decision variable: normalized acceleration sequence acc_norm[0:H-1] (same as RL action[1] in [-1, 1]).
    - Dynamics: linear double-integrator on x-vx; y-vy kept constant (no lane change decision).
    - Objective: QP proxy of low-level reward (progress + comfort + terminal intrinsic).
    - Constraints: linear inequalities (speed bounds + hard collision/headway bounds).

    Notes
    -----
     1) Keep optimization in strict QP form. Some non-quadratic RL terms are approximated to quadratic proxies.
     2) No safety layer.
     3) Lane-change action is fixed to KEEP (lane_scalar=0); only longitudinal control is optimized.
    """

    def __init__(
        self,
        env,
        horizon: int = 10,
        dt: float = 0.1,
        weights: Optional[Dict[str, float]] = None,
        intrinsic_coef: float = 10.0,
        intrinsic_type: str = "l2",
        intrinsic_norm_ranges: Optional[Sequence[Sequence[float]]] = None,
        intrinsic_weights: Optional[Sequence[float]] = None,
        min_front_gap: float = 8.0,
    ):
        self.env = env
        self.horizon = int(max(1, horizon))
        self.dt = float(dt)

        env_cfg = getattr(self.env, "config", {})
        action_cfg = env_cfg.get("action", {})

        self.acc_range = np.asarray(action_cfg.get("acceleration_range", [-5.0, 5.0]), dtype=np.float32)
        self.acc_min = float(self.acc_range[0])
        self.acc_max = float(self.acc_range[1])
        self.acc_scale = 0.5 * (self.acc_max - self.acc_min)
        self.acc_bias = 0.5 * (self.acc_max + self.acc_min)

        self.speed_limit = float(env_cfg.get("speed_limit", 15.0))
        self.lane_width = float(env_cfg.get("lane_width", 4.0))
        self.lanes_count = int(env_cfg.get("lanes_count", 3))
        self.lane_center_ys = (np.arange(self.lanes_count, dtype=np.float32) * self.lane_width).astype(np.float32)
        self.goal_longitudinal = float(env_cfg.get("goal_longitudinal", 400.0))
        self.comfort_max_accel = float(env_cfg.get("comfort_max_accel", 3.0))

        # Lane-change safety thresholds (common autonomous-driving defaults)
        # These are applied as hard constraints in joint_global MIQP.
        self.lc_front_gap_min = float(env_cfg.get("lane_change_min_front_gap", 10.0))
        self.lc_rear_gap_min = float(env_cfg.get("lane_change_min_rear_gap", 8.0))
        self.lc_front_ttc_min = float(env_cfg.get("lane_change_min_front_ttc", 3.0))
        self.lc_rear_ttc_min = float(env_cfg.get("lane_change_min_rear_ttc", 2.0))

        # Vehicle geometry for hard non-overlap constraints
        self.ego_length = 5.0
        self.neighbor_length = 5.0

        # Requested hard front-gap threshold (in addition to non-overlap)
        self.min_front_gap = float(min_front_gap)

        self.intrinsic_type = str(intrinsic_type).lower()
        self.disable_progress_when_huber = self.intrinsic_type == "huber_shaping"
        if intrinsic_norm_ranges is None:
            intrinsic_norm_ranges = [
                [0.0, 37.5],
                [-8.0, 8.0],
                [-8.0, 8.0],
                [-2.0, 2.0],
            ]
        self.intrinsic_norm_ranges = np.asarray(intrinsic_norm_ranges, dtype=np.float32)
        if self.intrinsic_norm_ranges.shape != (4, 2):
            raise ValueError("intrinsic_norm_ranges must have shape (4,2).")

        if intrinsic_weights is None:
            iw = np.ones((4,), dtype=np.float32) / 4.0
        else:
            iw = np.asarray(intrinsic_weights, dtype=np.float32).reshape(-1)
            if iw.shape[0] != 4:
                raise ValueError("intrinsic_weights must have 4 elements for [x, y, vx, vy].")
            s = float(np.sum(iw))
            iw = (np.ones((4,), dtype=np.float32) / 4.0) if s <= 1e-12 else (iw / s)
        self.intrinsic_weights = iw

        # QP objective weights
        base_w = {
            "intrinsic": float(intrinsic_coef),
            "progress_reward": float(env_cfg.get("progress_reward", 10.0)),
            "comfort_reward": float(env_cfg.get("comfort_reward", 0.7)),
            "lane_change_reward": float(env_cfg.get("lane_change_reward", -0.5)),
            "collision_reward": float(env_cfg.get("collision_reward", -10.0)),
            "on_road_reward": 1.0,
        }
        if weights:
            base_w.update({k: float(v) for k, v in weights.items()})
        self.w = base_w

    def _current_state(self) -> Tuple[np.ndarray, np.ndarray]:
        ego = self.env.vehicle
        start_state = np.asarray(
            [
                float(ego.position[0]),
                float(ego.position[1]),
                float(ego.velocity[0]),
                float(ego.velocity[1]),
            ],
            dtype=np.float32,
        )

        neighbors = []
        for v in self.env.road.vehicles:
            if v is ego:
                continue
            neighbors.append(
                [
                    float(v.position[0]),
                    float(v.position[1]),
                    float(v.velocity[0]),
                    float(v.velocity[1]),
                ]
            )
        neighbors_state = np.asarray(neighbors, dtype=np.float32).reshape(-1, 4)
        return start_state, neighbors_state

    def _predict_neighbors_all(self, neighbors_state: np.ndarray, horizon: int) -> np.ndarray:
        H = int(max(1, horizon))
        if neighbors_state.size == 0:
            return np.zeros((H, 0, 4), dtype=np.float32)
        out = np.zeros((H, neighbors_state.shape[0], 4), dtype=np.float32)
        for t in range(1, H + 1):
            tt = float(t) * self.dt
            n = neighbors_state.copy()
            n[:, 0] = n[:, 0] + n[:, 2] * tt
            n[:, 1] = n[:, 1] + n[:, 3] * tt
            out[t - 1] = n
        return out

    def _x_affine_coeff(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build affine map x_t = c_t + M_t @ u for t=1..H, and vx_t similarly.

        Decision u = [u_0, ..., u_{H-1}] (normalized acceleration sequence in [-1, 1]).
        Physical acceleration is a_x = acc_scale * u + acc_bias.
        """
        H = self.horizon
        dt = self.dt

        Mx = np.zeros((H, H), dtype=np.float32)
        Mv = np.zeros((H, H), dtype=np.float32)

        for t in range(1, H + 1):
            for k in range(0, t):
                Mx[t - 1, k] = (t - k - 0.5) * (dt ** 2)
                Mv[t - 1, k] = dt
        return Mx, Mv

    def _build_qp(
        self,
        start_state: np.ndarray,
        neighbors_state: np.ndarray,
        goal_phys: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Build standard QP:

        min 0.5 u^T H u + f^T u
        s.t. A u <= b
             lb <= u <= ub
        """
        Hn = self.horizon
        x0, y0, vx0, vy0 = [float(v) for v in start_state[:4]]
        gx, gy, gvx, gvy = [float(v) for v in goal_phys[:4]]

        Mx_base, Mv_base = self._x_affine_coeff()  # [H,H], [H,H]
        Mx = (self.acc_scale * Mx_base).astype(np.float32)
        Mv = (self.acc_scale * Mv_base).astype(np.float32)

        t_idx = np.arange(1, Hn + 1, dtype=np.float32)
        cx = x0 + vx0 * self.dt * t_idx + 0.5 * self.acc_bias * (self.dt ** 2) * (t_idx ** 2)
        cv = vx0 + self.acc_bias * self.dt * t_idx

        # Terminal state affine forms:
        # x_H = cx_H + mx @ u, vx_H = cv_H + mv @ u, y_H=y0, vy_H=vy0
        mx = Mx[-1]
        mv = Mv[-1]
        cxH = float(cx[-1])

        # Objective proxy for low-level reward:
        # maximize sum(progress_reward + comfort_reward) + terminal_intrinsic
        # => minimize negative of the above in QP form.
        H_mat = np.zeros((Hn, Hn), dtype=np.float32)
        f_vec = np.zeros((Hn,), dtype=np.float32)

        # 1) progress reward proxy (scenario.py): progress = delta_s / goal_longitudinal
        # under vx>=0 constraints, sum_t delta_s = x_H - x_0
        w_prog = 0.0 if self.disable_progress_when_huber else float(self.w.get("progress_reward", 0.0))
        goal_long = max(float(self.goal_longitudinal), 1e-6)
        k_prog = w_prog / goal_long
        # maximize k_prog * x_H  => minimize -k_prog * x_H
        f_vec += (-k_prog) * mx

        # 2) comfort reward proxy (scenario.py):
        # comfort_t = - (|acc|/comfort_max_accel)^2 * dt  (ignore min(.,1) clip for QP)
        # weighted term in env reward: w_comf * comfort_t
        # maximize -> minimize + w_comf * dt * (acc/amax)^2
        w_comf = float(self.w.get("comfort_reward", 0.0))
        amax = max(float(self.comfort_max_accel), 1e-6)
        c2 = w_comf * self.dt * (self.acc_scale ** 2) / (amax ** 2)
        c1 = w_comf * self.dt * (2.0 * self.acc_scale * self.acc_bias) / (amax ** 2)
        H_mat += (2.0 * c2) * np.eye(Hn, dtype=np.float32)
        f_vec += c1 * np.ones((Hn,), dtype=np.float32)

        # 3) terminal intrinsic proxy aligned with HIRO config:
        # utils.intrinsic_reward_l2 uses sqrt(weighted squared normalized error), which is non-QP.
        # here we use quadratic proxy: -0.5*coef*sum_i w_i*(alpha_i*err_i)^2
        # with err=[x_H-gx, y_H-gy, vx_H-gvx, vy_H-gvy].
        w_int = float(self.w.get("intrinsic", 0.0))
        ranges = self.intrinsic_norm_ranges
        span = np.maximum(ranges[:, 1] - ranges[:, 0], 1e-6)
        alpha = (2.0 / span).astype(np.float32)
        wi = self.intrinsic_weights.astype(np.float32)

        kx = float(w_int * wi[0] * (alpha[0] ** 2))
        kvx = float(w_int * wi[2] * (alpha[2] ** 2))

        H_mat += kx * np.outer(mx, mx).astype(np.float32)
        H_mat += kvx * np.outer(mv, mv).astype(np.float32)
        f_vec += (kx * (cxH - gx)) * mx
        f_vec += (kvx * (float(cv[-1]) - gvx)) * mv

        # Small regularization for numerical stability
        H_mat += 1e-6 * np.eye(Hn, dtype=np.float32)

        # Linear constraints A a <= b
        A_rows: List[np.ndarray] = []
        b_rows: List[float] = []

        # 1) Speed bounds: 0 <= vx_t <= speed_limit
        for t in range(Hn):
            mvt = Mv[t]
            cvt = float(cv[t])
            # upper: mvt @ u <= speed_limit - cvt
            A_rows.append(mvt.copy())
            b_rows.append(float(self.speed_limit - cvt))
            # lower: -mvt @ u <= cvt
            A_rows.append((-mvt).copy())
            b_rows.append(float(cvt))

        # 2) Hard collision + front-gap + front-TTC constraints with same-lane ahead vehicles.
        #    Distance: x_ego_t + d_req <= x_nei_t  ->  mxt @ u <= x_nei_t - d_req - cxt
        #    TTC: x_nei_t - x_ego_t >= ttc_min * (vx_ego_t - vx_nei_t)
        #         -> (mxt + ttc_min*mvt) @ u <= x_nei_t - cxt - ttc_min*(cvt - vx_nei_t)
        #    d_req uses the same front-gap threshold as lane-change safety gate.
        neighbors_pred = self._predict_neighbors_all(neighbors_state, Hn)
        d_req = float(self.lc_front_gap_min)
        ttc_min = float(self.lc_front_ttc_min)

        for j in range(neighbors_pred.shape[1]):
            yj0 = float(neighbors_state[j, 1])
            same_lane = abs(yj0 - y0) <= 0.5 * self.lane_width
            ahead_now = float(neighbors_state[j, 0]) > x0
            if not (same_lane and ahead_now):
                continue

            for t in range(Hn):
                xj_t = float(neighbors_pred[t, j, 0])
                vxj_t = float(neighbors_pred[t, j, 2])
                mxt = Mx[t]
                mvt = Mv[t]
                cxt = float(cx[t])
                cvt = float(cv[t])
                A_rows.append(mxt.copy())
                b_rows.append(float(xj_t - d_req - cxt))
                A_rows.append((mxt + ttc_min * mvt).copy())
                b_rows.append(float(xj_t - cxt - ttc_min * (cvt - vxj_t)))

        if A_rows:
            A = np.vstack(A_rows).astype(np.float32)
            b = np.asarray(b_rows, dtype=np.float32)
        else:
            A = np.zeros((0, Hn), dtype=np.float32)
            b = np.zeros((0,), dtype=np.float32)

        # decision variable is normalized acceleration command in RL action space
        lb = np.full((Hn,), -1.0, dtype=np.float32)
        ub = np.full((Hn,), 1.0, dtype=np.float32)

        return {
            "H": H_mat,
            "f": f_vec,
            "A": A,
            "b": b,
            "lb": lb,
            "ub": ub,
            "Mx": Mx,
            "Mv": Mv,
            "cx": cx,
            "cv": cv,
        }

    def _solve_qp_solver(
        self,
        H: np.ndarray,
        f: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        maxiter: int,
    ) -> Dict[str, Any]:
        if cp is None:
            raise ImportError("QP solver path requires cvxpy with GUROBI backend.")

        n = int(H.shape[0])
        u = cp.Variable(n)

        objective = 0.5 * cp.quad_form(u, H.astype(np.float64)) + cp.sum(cp.multiply(f.astype(np.float64), u))
        constraints = [u >= lb.astype(np.float64), u <= ub.astype(np.float64)]
        if A.shape[0] > 0:
            constraints.append(A.astype(np.float64) @ u <= b.astype(np.float64))

        prob = cp.Problem(cp.Minimize(objective), constraints)

        if "GUROBI" not in cp.installed_solvers():
            raise RuntimeError("GUROBI solver is required but not installed in cvxpy backends.")

        solver_tried: List[str] = ["GUROBI"]
        try:
            prob.solve(solver=cp.GUROBI, verbose=False)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"qp solve failed with GUROBI. error={e}") from e

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"qp solve failed with GUROBI. status={prob.status}")

        x_sol = np.asarray(u.value, dtype=np.float32).reshape(-1)
        viol = 0.0
        if A.shape[0] > 0:
            viol = float(np.max(np.maximum(A @ x_sol - b, 0.0)))

        return {
            "x": x_sol,
            "obj": float(prob.value),
            "max_violation": viol,
            "status": str(prob.status),
            "solver_tried": solver_tried,
        }

    def _rollout_from_u(self, start_state: np.ndarray, u_seq: np.ndarray) -> Dict[str, np.ndarray]:
        Hn = int(u_seq.shape[0])
        ax_seq = (self.acc_scale * np.asarray(u_seq, dtype=np.float32) + self.acc_bias).astype(np.float32)
        states = np.zeros((Hn + 1, 4), dtype=np.float32)
        states[0] = np.asarray(start_state[:4], dtype=np.float32)

        x, y, vx, vy = [float(v) for v in states[0]]
        for t in range(Hn):
            ax = float(ax_seq[t])
            x = x + vx * self.dt + 0.5 * ax * (self.dt ** 2)
            vx = vx + ax * self.dt
            states[t + 1] = np.asarray([x, y, vx, vy], dtype=np.float32)

        # RL action output: [lane_scalar, acc_norm]
        actions_cont = np.zeros((Hn, 2), dtype=np.float32)
        actions_cont[:, 0] = 0.0  # KEEP lane in this simplified MPC
        actions_cont[:, 1] = np.asarray(u_seq, dtype=np.float32)

        return {
            "states": states,
            "actions_cont": actions_cont,
            "acc_norm": np.asarray(u_seq, dtype=np.float32).copy(),
            "acc_phys": ax_seq.astype(np.float32).copy(),
        }

    def _y_to_lane_index(self, y: float) -> int:
        return int(np.argmin(np.abs(self.lane_center_ys - float(y))))

    @staticmethod
    def _lane_scalar_to_delta(lane_scalar: float) -> int:
        s = float(np.clip(lane_scalar, -1.0, 1.0))
        if s < -1.0 / 3.0:
            return -1
        if s > 1.0 / 3.0:
            return 1
        return 0

    def _rollout_joint_from_actions(
        self,
        start_state: np.ndarray,
        lane_scalar_seq: np.ndarray,
        u_seq: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        Hn = int(min(lane_scalar_seq.shape[0], u_seq.shape[0]))
        lane_scalar_seq = np.asarray(lane_scalar_seq[:Hn], dtype=np.float32)
        u_seq = np.asarray(u_seq[:Hn], dtype=np.float32)
        u_seq = np.clip(u_seq, -1.0, 1.0)
        ax_seq = (self.acc_scale * u_seq + self.acc_bias).astype(np.float32)

        states = np.zeros((Hn + 1, 4), dtype=np.float32)
        states[0] = np.asarray(start_state[:4], dtype=np.float32)
        lane_idx_seq = np.zeros((Hn + 1,), dtype=np.int32)
        lane_change_step = np.zeros((Hn,), dtype=np.float32)

        x, y, vx, vy = [float(v) for v in states[0]]
        lane_idx = self._y_to_lane_index(y)
        lane_idx_seq[0] = lane_idx

        for t in range(Hn):
            delta_lane = self._lane_scalar_to_delta(float(lane_scalar_seq[t]))
            next_lane_idx = int(np.clip(lane_idx + delta_lane, 0, self.lanes_count - 1))
            lane_change_step[t] = 1.0 if next_lane_idx != lane_idx else 0.0
            lane_idx = next_lane_idx

            y = float(self.lane_center_ys[lane_idx])
            vy = 0.0

            ax = float(ax_seq[t])
            x = x + vx * self.dt + 0.5 * ax * (self.dt ** 2)
            vx = vx + ax * self.dt

            states[t + 1] = np.asarray([x, y, vx, vy], dtype=np.float32)
            lane_idx_seq[t + 1] = lane_idx

        actions_cont = np.zeros((Hn, 2), dtype=np.float32)
        actions_cont[:, 0] = np.clip(lane_scalar_seq, -1.0, 1.0)
        actions_cont[:, 1] = u_seq

        return {
            "states": states,
            "actions_cont": actions_cont,
            "acc_norm": u_seq.astype(np.float32).copy(),
            "acc_phys": ax_seq.astype(np.float32).copy(),
            "lane_idx_seq": lane_idx_seq,
            "lane_change_step": lane_change_step,
        }

    def _collision_step_flags(
        self,
        states: np.ndarray,
        lane_idx_seq: np.ndarray,
        neighbors_state: np.ndarray,
    ) -> np.ndarray:
        Hn = int(states.shape[0] - 1)
        flags = np.zeros((Hn,), dtype=np.float32)
        if neighbors_state.size == 0 or Hn <= 0:
            return flags

        neighbors_pred = self._predict_neighbors_all(neighbors_state, Hn)
        neighbors_lane_idx = np.argmin(
            np.abs(neighbors_pred[:, :, 1][:, :, None] - self.lane_center_ys.reshape(1, 1, -1)), axis=2
        ).astype(np.int32)
        d_req = 0.5 * (self.ego_length + self.neighbor_length)

        crashed = False
        for t in range(Hn):
            if crashed:
                break
            ego_x = float(states[t + 1, 0])
            ego_lane = int(lane_idx_seq[t + 1])
            for j in range(neighbors_pred.shape[1]):
                if int(neighbors_lane_idx[t, j]) != ego_lane:
                    continue
                xj = float(neighbors_pred[t, j, 0])
                if abs(xj - ego_x) < d_req:
                    flags[t] = 1.0
                    crashed = True
                    break
        return flags

    def _collect_step_constraint_violations(
        self,
        states: np.ndarray,
        lane_idx_seq: np.ndarray,
        lane_scalar_raw: np.ndarray,
        acc_norm_raw: np.ndarray,
        neighbors_state: np.ndarray,
    ) -> Dict[str, Any]:
        Hn = int(states.shape[0] - 1)
        if Hn <= 0:
            return {
                "per_step": [],
                "max_violation": 0.0,
                "violation_counts": {},
                "violated_steps": [],
            }

        lane_scalar_raw = np.asarray(lane_scalar_raw, dtype=np.float32).reshape(-1)[:Hn]
        acc_norm_raw = np.asarray(acc_norm_raw, dtype=np.float32).reshape(-1)[:Hn]

        neighbors_pred = self._predict_neighbors_all(neighbors_state, Hn)
        neighbors_lane_idx = np.argmin(
            np.abs(neighbors_pred[:, :, 1][:, :, None] - self.lane_center_ys.reshape(1, 1, -1)), axis=2
        ).astype(np.int32)

        # Keep violation checking semantics consistent with plan_joint_optimal:
        # - front-gap hard constraints only consider vehicles that are ahead at t=0
        # - lane-change front/rear references are preselected by x_ref(t)
        ahead_now_mask = np.zeros((neighbors_state.shape[0],), dtype=bool)
        if neighbors_state.size > 0:
            x0 = float(states[0, 0])
            ahead_now_mask = np.asarray(neighbors_state[:, 0] > x0, dtype=bool)

        L = int(self.lanes_count)
        x_ref = float(states[0, 0]) + float(states[0, 2]) * self.dt * np.arange(1, Hn + 1, dtype=np.float32)
        front_info: List[List[Optional[Tuple[float, float]]]] = [[None for _ in range(L)] for _ in range(Hn)]
        rear_info: List[List[Optional[Tuple[float, float]]]] = [[None for _ in range(L)] for _ in range(Hn)]
        for ti in range(Hn):
            x_ref_t = float(x_ref[ti])
            for lane_i in range(L):
                best_front_x = None
                best_front_v = 0.0
                best_rear_x = None
                best_rear_v = 0.0
                for j in range(neighbors_pred.shape[1]):
                    if int(neighbors_lane_idx[ti, j]) != lane_i:
                        continue
                    xj_t = float(neighbors_pred[ti, j, 0])
                    vj_t = float(neighbors_pred[ti, j, 2])
                    if xj_t >= x_ref_t:
                        if (best_front_x is None) or (xj_t < best_front_x):
                            best_front_x = xj_t
                            best_front_v = vj_t
                    else:
                        if (best_rear_x is None) or (xj_t > best_rear_x):
                            best_rear_x = xj_t
                            best_rear_v = vj_t
                if best_front_x is not None:
                    front_info[ti][lane_i] = (float(best_front_x), float(best_front_v))
                if best_rear_x is not None:
                    rear_info[ti][lane_i] = (float(best_rear_x), float(best_rear_v))

        collision_step = self._collision_step_flags(states, lane_idx_seq, neighbors_state)
        d_req_front = float(self.lc_front_gap_min)

        keys = [
            "speed_limit",
            "acc_norm_bound",
            "lane_scalar_discrete",
            "collision",
            "front_gap_same_lane",
            "front_ttc_same_lane",
            "lc_front_gap_origin",
            "lc_front_gap_target",
            "lc_front_ttc_origin",
            "lc_front_ttc_target",
            "lc_rear_gap_origin",
            "lc_rear_gap_target",
            "lc_rear_ttc_origin",
            "lc_rear_ttc_target",
        ]
        violation_counts: Dict[str, int] = {k: 0 for k in keys}
        per_step: List[Dict[str, Any]] = []
        violated_steps: List[int] = []
        max_violation = 0.0

        for t in range(Hn):
            step_id = int(t + 1)
            ego_x = float(states[t + 1, 0])
            ego_vx = float(states[t + 1, 2])
            lane_prev = int(lane_idx_seq[t])
            lane_now = int(lane_idx_seq[t + 1])
            changed = lane_now != lane_prev

            speed_v = max(ego_vx - self.speed_limit, 0.0)
            acc_v = max(abs(float(acc_norm_raw[t])) - 1.0, 0.0)
            lane_disc_v = min(
                abs(float(lane_scalar_raw[t]) + 1.0),
                abs(float(lane_scalar_raw[t]) - 0.0),
                abs(float(lane_scalar_raw[t]) - 1.0),
            )
            collision_v = float(collision_step[t])

            # Same-lane hard front-gap used in MPC constraints.
            front_gap_v = 0.0
            front_ttc_v = 0.0
            for j in range(neighbors_pred.shape[1]):
                if j < ahead_now_mask.shape[0] and not bool(ahead_now_mask[j]):
                    continue
                if int(neighbors_lane_idx[t, j]) != lane_now:
                    continue
                xj = float(neighbors_pred[t, j, 0])
                vj = float(neighbors_pred[t, j, 2])
                gap = xj - ego_x
                front_gap_v = max(front_gap_v, max(d_req_front - gap, 0.0))
                rel_front_same = max(ego_vx - vj, 0.0)
                front_ttc_v = max(front_ttc_v, max(float(self.lc_front_ttc_min) * rel_front_same - gap, 0.0))

            lc_front_gap_origin_v = 0.0
            lc_front_gap_target_v = 0.0
            lc_front_ttc_origin_v = 0.0
            lc_front_ttc_target_v = 0.0
            lc_rear_gap_origin_v = 0.0
            lc_rear_gap_target_v = 0.0
            lc_rear_ttc_origin_v = 0.0
            lc_rear_ttc_target_v = 0.0

            if changed and neighbors_pred.shape[1] > 0:
                for lane_role, lane_i in (("origin", lane_prev), ("target", lane_now)):
                    front_x = None
                    front_v = 0.0
                    rear_x = None
                    rear_v = 0.0
                    front = front_info[t][lane_i] if t < len(front_info) and lane_i < L else None
                    rear = rear_info[t][lane_i] if t < len(rear_info) and lane_i < L else None
                    if front is not None:
                        front_x, front_v = front
                    if rear is not None:
                        rear_x, rear_v = rear

                    if front_x is not None:
                        d_front = float(front_x) - ego_x
                        rel_front = max(ego_vx - float(front_v), 0.0)
                        gap_v = max(float(self.lc_front_gap_min) - d_front, 0.0)
                        ttc_v = max(float(self.lc_front_ttc_min) * rel_front - d_front, 0.0)
                        if lane_role == "origin":
                            lc_front_gap_origin_v = gap_v
                            lc_front_ttc_origin_v = ttc_v
                        else:
                            lc_front_gap_target_v = gap_v
                            lc_front_ttc_target_v = ttc_v

                    if rear_x is not None:
                        d_rear = ego_x - float(rear_x)
                        rel_rear = max(float(rear_v) - ego_vx, 0.0)
                        gap_v = max(float(self.lc_rear_gap_min) - d_rear, 0.0)
                        ttc_v = max(float(self.lc_rear_ttc_min) * rel_rear - d_rear, 0.0)
                        if lane_role == "origin":
                            lc_rear_gap_origin_v = gap_v
                            lc_rear_ttc_origin_v = ttc_v
                        else:
                            lc_rear_gap_target_v = gap_v
                            lc_rear_ttc_target_v = ttc_v

            violations = {
                "speed_limit": float(speed_v),
                "acc_norm_bound": float(acc_v),
                "lane_scalar_discrete": float(lane_disc_v),
                "collision": float(collision_v),
                "front_gap_same_lane": float(front_gap_v),
                "front_ttc_same_lane": float(front_ttc_v),
                "lc_front_gap_origin": float(lc_front_gap_origin_v),
                "lc_front_gap_target": float(lc_front_gap_target_v),
                "lc_front_ttc_origin": float(lc_front_ttc_origin_v),
                "lc_front_ttc_target": float(lc_front_ttc_target_v),
                "lc_rear_gap_origin": float(lc_rear_gap_origin_v),
                "lc_rear_gap_target": float(lc_rear_gap_target_v),
                "lc_rear_ttc_origin": float(lc_rear_ttc_origin_v),
                "lc_rear_ttc_target": float(lc_rear_ttc_target_v),
            }

            violated = [k for k, v in violations.items() if float(v) > 1e-6]
            if violated:
                violated_steps.append(step_id)
                for k in violated:
                    violation_counts[k] += 1
            step_max = float(max(violations.values())) if violations else 0.0
            max_violation = max(max_violation, step_max)

            per_step.append(
                {
                    "step": step_id,
                    "lane_prev": int(lane_prev),
                    "lane_now": int(lane_now),
                    "lane_change": bool(changed),
                    "violations": violations,
                    "violated_constraints": violated,
                }
            )

        return {
            "per_step": per_step,
            "max_violation": float(max_violation),
            "violation_counts": violation_counts,
            "violated_steps": violated_steps,
        }

    def _evaluate_joint_result(
        self,
        states: np.ndarray,
        lane_change_step: np.ndarray,
        lane_idx_seq: np.ndarray,
        goal_phys: np.ndarray,
        acc_phys: np.ndarray,
        neighbors_state: np.ndarray,
    ) -> Dict[str, Any]:
        Hn = int(acc_phys.shape[0])
        collision_step = self._collision_step_flags(states, lane_idx_seq, neighbors_state)

        valid_mask = np.ones((Hn,), dtype=np.float32)
        crash_idx = np.flatnonzero(collision_step > 0.5)
        if crash_idx.size > 0:
            first = int(crash_idx[0])
            if first + 1 < Hn:
                valid_mask[first + 1 :] = 0.0

        sH = states[int(np.sum(valid_mask)), :4] if Hn > 0 else states[-1, :4]
        goal = np.asarray(goal_phys[:4], dtype=np.float32)
        ego_start = states[0, :4].astype(np.float32)
        ego_rel_H = (sH - ego_start).astype(np.float32)
        goal_rel = (goal - ego_start).astype(np.float32)

        intrinsic_type = str(self.intrinsic_type).lower()
        if intrinsic_type == "huber_shaping":
            prev_idx = max(int(np.sum(valid_mask)) - 1, 0)
            ego_rel_prev = (states[prev_idx, :4] - ego_start).astype(np.float32)
            r_huber, _, _, _ = hiro_utils.intrinsic_reward_shaping_huber(
                ego_rel_now=ego_rel_prev[None, :],
                ego_rel_next=ego_rel_H[None, :],
                goal_rel=goal_rel[None, :],
                norm_ranges=self.intrinsic_norm_ranges,
                coef=float(self.w.get("intrinsic", 0.0)),
                weights=self.intrinsic_weights,
                gamma=1.0,
                is_terminal=np.array([True], dtype=bool),
            )
            intrinsic = float(np.asarray(r_huber, dtype=np.float32).reshape(-1)[0])
        else:
            r_l2, _, _ = hiro_utils.intrinsic_reward_l2(
                ego_next_sub_rel=ego_rel_H[None, :],
                goal_rel=goal_rel[None, :],
                norm_ranges=self.intrinsic_norm_ranges,
                coef=float(self.w.get("intrinsic", 0.0)),
                weights=self.intrinsic_weights,
            )
            intrinsic = float(np.asarray(r_l2, dtype=np.float32).reshape(-1)[0])

        comfort_step = -(np.clip(np.abs(acc_phys) / max(self.comfort_max_accel, 1e-6), 0.0, 1.0) ** 2) * self.dt
        comfort_step = comfort_step * valid_mask

        progress_step = np.zeros((Hn,), dtype=np.float32)
        if Hn > 0:
            delta_x = np.diff(states[:, 0], prepend=states[0, 0]).astype(np.float32)
            progress_step = np.clip(delta_x[1:] / max(self.goal_longitudinal, 1e-6), 0.0, 1.0) * valid_mask

        on_road_step = valid_mask.copy()
        lane_change_step = np.asarray(lane_change_step, dtype=np.float32) * valid_mask
        collision_step = collision_step.astype(np.float32) * valid_mask

        progress_w = 0.0 if self.disable_progress_when_huber else float(self.w.get("progress_reward", 0.0))
        low_ext_step = np.zeros((Hn,), dtype=np.float32)
        low_ext_step += np.asarray(progress_step * progress_w, dtype=np.float32)
        low_ext_step += np.asarray(comfort_step * float(self.w.get("comfort_reward", 0.0)), dtype=np.float32)
        low_ext_step += np.asarray(lane_change_step * float(self.w.get("lane_change_reward", 0.0)), dtype=np.float32)
        low_ext_step += np.asarray(collision_step * float(self.w.get("collision_reward", 0.0)), dtype=np.float32)

        intrinsic_step = np.zeros((Hn,), dtype=np.float32)
        if Hn > 0:
            last_valid = max(int(np.sum(valid_mask)) - 1, 0)
            intrinsic_step[last_valid] = np.float32(intrinsic)

        low_total_step = low_ext_step + intrinsic_step

        return {
            "intrinsic_step": intrinsic_step,
            "comfort_step": np.asarray(comfort_step * float(self.w.get("comfort_reward", 0.0)), dtype=np.float32),
            "low_ext_step": low_ext_step,
            "low_total_step": low_total_step,
            "sum_low_ext": float(np.sum(low_ext_step)),
            "sum_intrinsic": float(np.sum(intrinsic_step)),
            "sum_low_total": float(np.sum(low_total_step)),
            "reward_components": {
                "progress_reward": float(np.sum(progress_step) * progress_w),
                "comfort_reward": float(np.sum(comfort_step) * float(self.w.get("comfort_reward", 0.0))),
                "lane_change_reward": float(np.sum(lane_change_step) * float(self.w.get("lane_change_reward", 0.0))),
                "collision_reward": float(np.sum(collision_step) * float(self.w.get("collision_reward", 0.0))),
                "on_road_reward": float(np.sum(on_road_step) * float(self.w.get("on_road_reward", 1.0))),
                "intrinsic_reward": float(np.sum(intrinsic_step)),
            },
            "goal_rel": (goal - states[0, :4]).astype(np.float32),
            "eval_step_idx": int(max(int(np.sum(valid_mask)), 0)),
            "collision_step": collision_step,
        }

    def plan_joint_optimal(
        self,
        goal_phys: Sequence[float],
        steps_to_goal: int,
        maxiter: int = 250,
        enumerate_alternative_optima: bool = False,
        max_alternative_optima: int = 3,
        alternative_objective_tol: float = 1e-5,
    ) -> Dict[str, Any]:
        _ = steps_to_goal
        goal = np.asarray(goal_phys, dtype=np.float32).reshape(-1)
        if goal.shape[0] < 4:
            raise ValueError("goal_phys must contain at least [x, y, vx, vy].")

        Hn = int(self.horizon)
        start_state, neighbors_state = self._current_state()
        x0, y0, vx0, vy0 = [float(v) for v in start_state[:4]]
        gx, gy, gvx, gvy = [float(v) for v in goal[:4]]

        lane0 = self._y_to_lane_index(y0)
        L = int(self.lanes_count)

        u = cp.Variable(Hn)
        x = cp.Variable(Hn + 1)
        vx = cp.Variable(Hn + 1)
        k = cp.Variable(Hn + 1, integer=True)

        z_left = cp.Variable(Hn, boolean=True)
        z_keep = cp.Variable(Hn, boolean=True)
        z_right = cp.Variable(Hn, boolean=True)

        y_lane = cp.Variable((Hn + 1, L), boolean=True)

        constraints = []
        constraints += [x[0] == x0, vx[0] == vx0, k[0] == lane0]
        constraints += [vx >= 0.0, vx <= self.speed_limit]
        constraints += [u >= -1.0, u <= 1.0]
        constraints += [k >= 0, k <= L - 1]

        for t in range(Hn + 1):
            constraints += [cp.sum(y_lane[t, :]) == 1]
            constraints += [k[t] == cp.sum(cp.multiply(np.arange(L, dtype=np.float32), y_lane[t, :]))]

        for t in range(Hn):
            constraints += [z_left[t] + z_keep[t] + z_right[t] == 1]
            delta_lane = -z_left[t] + z_right[t]  # in {-1,0,1}
            constraints += [k[t + 1] == k[t] + delta_lane]

            a_phys_t = self.acc_scale * u[t] + self.acc_bias
            constraints += [x[t + 1] == x[t] + vx[t] * self.dt + 0.5 * a_phys_t * (self.dt ** 2)]
            constraints += [vx[t + 1] == vx[t] + a_phys_t * self.dt]

        # Hard collision constraints (same lane + ahead vehicle)
        neighbors_pred = self._predict_neighbors_all(neighbors_state, Hn)
        d_req = float(self.lc_front_gap_min)
        ttc_min = float(self.lc_front_ttc_min)
        M = float(self.goal_longitudinal + 2.0 * self.speed_limit * Hn * self.dt + 1000.0)

        for j in range(neighbors_pred.shape[1]):
            ahead_now = float(neighbors_state[j, 0]) > x0
            if not ahead_now:
                continue
            for t in range(1, Hn + 1):
                yj_t = float(neighbors_pred[t - 1, j, 1])
                lane_j_t = int(np.argmin(np.abs(self.lane_center_ys - yj_t)))
                xj_t = float(neighbors_pred[t - 1, j, 0])
                # If ego in same lane (y_lane[t, lane_j_t]==1), enforce front-gap hard constraint.
                constraints += [x[t] + d_req <= xj_t + M * (1.0 - y_lane[t, lane_j_t])]
                # If ego in same lane, also enforce front TTC hard constraint.
                vj_t = float(neighbors_pred[t - 1, j, 2])
                rel_front_same = cp.Variable(nonneg=True)
                constraints += [rel_front_same >= vx[t] - vj_t]
                d_front_same = xj_t - x[t]
                constraints += [d_front_same >= ttc_min * rel_front_same - M * (1.0 - y_lane[t, lane_j_t])]

        # Lane-change safety constraints:
        # when changing lane at step t, origin lane (t) and target lane (t+1) must satisfy
        # front/rear distance and TTC thresholds.
        x_ref = x0 + vx0 * self.dt * np.arange(1, Hn + 1, dtype=np.float32)
        front_info: List[List[Optional[Tuple[float, float]]]] = [[None for _ in range(L)] for _ in range(Hn)]
        rear_info: List[List[Optional[Tuple[float, float]]]] = [[None for _ in range(L)] for _ in range(Hn)]

        for ti in range(Hn):
            for lane_i in range(L):
                best_front_x = None
                best_front_v = 0.0
                best_rear_x = None
                best_rear_v = 0.0
                x_ref_t = float(x_ref[ti])
                for j in range(neighbors_pred.shape[1]):
                    yj_t = float(neighbors_pred[ti, j, 1])
                    lane_j_t = int(np.argmin(np.abs(self.lane_center_ys - yj_t)))
                    if lane_j_t != lane_i:
                        continue
                    xj_t = float(neighbors_pred[ti, j, 0])
                    vj_t = float(neighbors_pred[ti, j, 2])
                    if xj_t >= x_ref_t:
                        if (best_front_x is None) or (xj_t < best_front_x):
                            best_front_x = xj_t
                            best_front_v = vj_t
                    else:
                        if (best_rear_x is None) or (xj_t > best_rear_x):
                            best_rear_x = xj_t
                            best_rear_v = vj_t
                if best_front_x is not None:
                    front_info[ti][lane_i] = (float(best_front_x), float(best_front_v))
                if best_rear_x is not None:
                    rear_info[ti][lane_i] = (float(best_rear_x), float(best_rear_v))

        for t in range(Hn):
            z_change_t = z_left[t] + z_right[t]  # 1 iff lane change at step t
            for lane_i in range(L):
                # Activate constraints for origin lane and target lane when changing.
                gate_origin = 2.0 - z_change_t - y_lane[t, lane_i]
                gate_target = 2.0 - z_change_t - y_lane[t + 1, lane_i]

                for gate in (gate_origin, gate_target):
                    front = front_info[t][lane_i]
                    if front is not None:
                        x_front, v_front = front
                        rel_front = cp.Variable(nonneg=True)
                        constraints += [rel_front >= vx[t + 1] - float(v_front)]
                        d_front = float(x_front) - x[t + 1]
                        constraints += [d_front >= float(self.lc_front_gap_min) - M * gate]
                        constraints += [d_front >= float(self.lc_front_ttc_min) * rel_front - M * gate]

                    rear = rear_info[t][lane_i]
                    if rear is not None:
                        x_rear, v_rear = rear
                        rel_rear = cp.Variable(nonneg=True)
                        constraints += [rel_rear >= float(v_rear) - vx[t + 1]]
                        d_rear = x[t + 1] - float(x_rear)
                        constraints += [d_rear >= float(self.lc_rear_gap_min) - M * gate]
                        constraints += [d_rear >= float(self.lc_rear_ttc_min) * rel_rear - M * gate]

        # Objective: minimize negative low-level reward proxy (MIQP)
        objective = 0

        # progress reward proxy
        progress_w = 0.0 if self.disable_progress_when_huber else float(self.w.get("progress_reward", 0.0))
        objective += -progress_w * (x[Hn] - x0) / max(self.goal_longitudinal, 1e-6)

        # comfort reward
        comfort_w = float(self.w.get("comfort_reward", 0.0))
        amax = max(float(self.comfort_max_accel), 1e-6)
        objective += comfort_w * self.dt * cp.sum_squares((self.acc_scale * u + self.acc_bias) / amax)

        # lane change penalty (lane_change_reward is typically negative)
        lane_penalty = -float(self.w.get("lane_change_reward", 0.0))
        objective += lane_penalty * cp.sum(z_left + z_right)

        # terminal intrinsic quadratic proxy
        w_int = float(self.w.get("intrinsic", 0.0))
        ranges = self.intrinsic_norm_ranges
        span = np.maximum(ranges[:, 1] - ranges[:, 0], 1e-6)
        alpha = (2.0 / span).astype(np.float32)
        wi = self.intrinsic_weights.astype(np.float32)

        y_H_expr = self.lane_width * k[Hn]
        err_x = x[Hn] - gx
        err_y = y_H_expr - gy
        err_vx = vx[Hn] - gvx
        err_vy = 0.0 - gvy

        intrinsic_type = str(self.intrinsic_type).lower()
        if intrinsic_type == "l2":
            # Match utils.intrinsic_reward_l2 exactly:
            # r_int = -coef * sqrt(sum_i w_i * (norm_err_i)^2)
            # where norm_err_i = alpha_i * err_i.
            weighted_delta = cp.hstack([
                np.sqrt(float(wi[0])) * float(alpha[0]) * err_x,
                np.sqrt(float(wi[1])) * float(alpha[1]) * err_y,
                np.sqrt(float(wi[2])) * float(alpha[2]) * err_vx,
                np.sqrt(float(wi[3])) * float(alpha[3]) * err_vy,
            ])
            intrinsic_dist = cp.norm(weighted_delta, 2)
            objective += float(w_int) * intrinsic_dist
        else:
            # Keep convex quadratic proxy for non-l2 intrinsic types (e.g., huber_shaping)
            objective += 0.5 * w_int * wi[0] * (alpha[0] ** 2) * cp.square(err_x)
            objective += 0.5 * w_int * wi[1] * (alpha[1] ** 2) * cp.square(err_y)
            objective += 0.5 * w_int * wi[2] * (alpha[2] ** 2) * cp.square(err_vx)
            objective += 0.5 * w_int * wi[3] * (alpha[3] ** 2) * (err_vy ** 2)

        prob = cp.Problem(cp.Minimize(objective), constraints)

        if "GUROBI" not in cp.installed_solvers():
            raise RuntimeError("joint_global requires GUROBI solver, but it is not installed in cvxpy backends.")

        solver_tried: List[str] = ["GUROBI"]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob.solve(solver=cp.GUROBI, verbose=False)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"joint_global GUROBI solve failed. error={e}") from e

        if prob.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"joint_global GUROBI solve failed. status={prob.status}")

        u_best = np.asarray(u.value, dtype=np.float32).reshape(-1)

        z_left_raw = np.asarray(z_left.value, dtype=np.float32).reshape(-1)
        z_keep_raw = np.asarray(z_keep.value, dtype=np.float32).reshape(-1)
        z_right_raw = np.asarray(z_right.value, dtype=np.float32).reshape(-1)
        k_raw = np.asarray(k.value, dtype=np.float32).reshape(-1)

        # Decode lane action from solved lane-index trajectory first (more robust than
        # independently rounding z_left/z_right when MIP tolerance leaves near-fractional values).
        if k_raw.shape[0] >= Hn + 1:
            lane_delta = np.diff(k_raw[: Hn + 1])
            lane_best = np.where(lane_delta > 0.5, 1.0, np.where(lane_delta < -0.5, -1.0, 0.0)).astype(np.float32)
        else:
            z_left_val = np.rint(z_left_raw).astype(np.int32)
            z_right_val = np.rint(z_right_raw).astype(np.int32)
            lane_best = (-z_left_val + z_right_val).astype(np.float32)
            lane_best = np.where(lane_best > 0.5, 1.0, np.where(lane_best < -0.5, -1.0, 0.0)).astype(np.float32)

        z_integrality_max = float(
            np.max(
                np.concatenate([
                    np.abs(z_left_raw - np.rint(z_left_raw)),
                    np.abs(z_keep_raw - np.rint(z_keep_raw)),
                    np.abs(z_right_raw - np.rint(z_right_raw)),
                ])
            )
        ) if Hn > 0 else 0.0
        k_integrality_max = float(np.max(np.abs(k_raw - np.rint(k_raw)))) if k_raw.size > 0 else 0.0

        rollout = self._rollout_joint_from_actions(start_state, lane_best, u_best)
        eval_res = self._evaluate_joint_result(
            states=rollout["states"],
            lane_change_step=rollout["lane_change_step"],
            lane_idx_seq=rollout["lane_idx_seq"],
            goal_phys=goal[:4],
            acc_phys=rollout["acc_phys"],
            neighbors_state=neighbors_state,
        )

        alternative_optima: List[Dict[str, Any]] = []
        uniqueness_checked = bool(enumerate_alternative_optima)
        max_alt = int(max(0, max_alternative_optima))
        obj_tol = max(float(alternative_objective_tol), 0.0)
        primary_obj = float(prob.value)

        if uniqueness_checked and max_alt > 0:
            cut_constraints: List[Any] = []

            def _build_lane_no_good_cut(lane_idx_seq: np.ndarray) -> Any:
                lane_idx_seq = np.asarray(lane_idx_seq, dtype=np.int32).reshape(-1)
                terms = []
                for tt in range(min(Hn + 1, lane_idx_seq.shape[0])):
                    lane_idx_t = int(np.clip(lane_idx_seq[tt], 0, L - 1))
                    terms.append(y_lane[tt, lane_idx_t])
                if not terms:
                    return cp.sum(y_lane[0, :]) <= 0.0
                return cp.sum(cp.hstack(terms)) <= float(len(terms) - 1)

            cut_constraints.append(_build_lane_no_good_cut(rollout["lane_idx_seq"]))

            for _ in range(max_alt):
                constraints_alt = list(constraints) + list(cut_constraints)
                constraints_alt.append(objective <= primary_obj + obj_tol)
                prob_alt = cp.Problem(cp.Minimize(objective), constraints_alt)
                try:
                    prob_alt.solve(solver=cp.GUROBI, verbose=False)
                except Exception:
                    break

                if prob_alt.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                    break
                if float(prob_alt.value) > primary_obj + obj_tol + 1e-9:
                    break

                u_alt = np.asarray(u.value, dtype=np.float32).reshape(-1)
                k_alt_raw = np.asarray(k.value, dtype=np.float32).reshape(-1)
                if k_alt_raw.shape[0] >= Hn + 1:
                    lane_delta_alt = np.diff(k_alt_raw[: Hn + 1])
                    lane_alt = np.where(
                        lane_delta_alt > 0.5,
                        1.0,
                        np.where(lane_delta_alt < -0.5, -1.0, 0.0),
                    ).astype(np.float32)
                else:
                    lane_alt = np.zeros((Hn,), dtype=np.float32)

                rollout_alt = self._rollout_joint_from_actions(start_state, lane_alt, u_alt)
                eval_alt = self._evaluate_joint_result(
                    states=rollout_alt["states"],
                    lane_change_step=rollout_alt["lane_change_step"],
                    lane_idx_seq=rollout_alt["lane_idx_seq"],
                    goal_phys=goal[:4],
                    acc_phys=rollout_alt["acc_phys"],
                    neighbors_state=neighbors_state,
                )
                alternative_optima.append(
                    {
                        "fun": float(prob_alt.value),
                        "states": rollout_alt["states"],
                        "acc_norm": rollout_alt["acc_norm"],
                        "acc_phys": rollout_alt["acc_phys"],
                        "lane_idx_seq": rollout_alt["lane_idx_seq"],
                        "lane_change_step": rollout_alt["lane_change_step"],
                        "best_actions_cont": rollout_alt["actions_cont"],
                        "sum_low_total": float(eval_alt.get("sum_low_total", 0.0)),
                        "sum_low_ext": float(eval_alt.get("sum_low_ext", 0.0)),
                        "sum_intrinsic": float(eval_alt.get("sum_intrinsic", 0.0)),
                    }
                )

                cut_constraints.append(_build_lane_no_good_cut(rollout_alt["lane_idx_seq"]))

        return {
            "success": bool(prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}),
            "message": str(prob.status),
            "iterations": int(maxiter),
            "fun": float(prob.value),
            "best_actions_cont": rollout["actions_cont"],
            "start_state": start_state,
            "neighbors_state": neighbors_state,
            "goal_phys": goal[:4].copy(),
            "horizon": int(Hn),
            "steps_to_goal": int(steps_to_goal),
            "enforce_collision_constraints": True,
            "collision_relax_eps": 0.0,
            "states": rollout["states"],
            "acc_norm": rollout["acc_norm"],
            "acc_phys": rollout["acc_phys"],
            "lane_idx_seq": rollout["lane_idx_seq"],
            "lane_change_step": rollout["lane_change_step"],
            "max_constraint_violation": 0.0,
            "progress_objective_enabled": bool(not self.disable_progress_when_huber),
            "approximation_notes": [
                "joint optimization solved as MIQP (branch-and-bound on integer variables)",
                "lane_scalar is strictly discrete in {-1, 0, 1}",
                "collision is enforced as hard constraints with same-lane gating",
                "lane-change gating enforces origin/target lane front-rear distance and TTC thresholds",
                "neighbors predicted with constant velocity and lane inferred from predicted y",
                "for intrinsic_type=l2, optimization uses exact weighted L2 terminal intrinsic (MISOCP form)",
            ],
            "solver": {
                "type": "miqp",
                "maxiter": int(maxiter),
                "installed": list(cp.installed_solvers()),
                "tried": solver_tried,
                "status": str(prob.status),
                "z_integrality_max": z_integrality_max,
                "k_integrality_max": k_integrality_max,
                "uniqueness_checked": uniqueness_checked,
                "objective_optimal": primary_obj,
                "alternative_objective_tol": obj_tol,
                "alternative_optima_found": int(len(alternative_optima)),
            },
            "alternative_optima": alternative_optima,
            **eval_res,
        }

    def _evaluate_result(self, states: np.ndarray, goal_phys: np.ndarray, acc_phys: np.ndarray) -> Dict[str, Any]:
        Hn = int(acc_phys.shape[0])
        sH = states[-1, :4]
        goal = np.asarray(goal_phys[:4], dtype=np.float32)
        ego_start = states[0, :4].astype(np.float32)
        ego_rel_H = (sH - ego_start).astype(np.float32)
        goal_rel = (goal - ego_start).astype(np.float32)

        intrinsic_type = str(self.intrinsic_type).lower()
        if intrinsic_type == "huber_shaping":
            ego_rel_prev = ego_rel_H if Hn <= 0 else (states[-2, :4] - ego_start).astype(np.float32)
            r_huber, _, _, _ = hiro_utils.intrinsic_reward_shaping_huber(
                ego_rel_now=ego_rel_prev[None, :],
                ego_rel_next=ego_rel_H[None, :],
                goal_rel=goal_rel[None, :],
                norm_ranges=self.intrinsic_norm_ranges,
                coef=float(self.w.get("intrinsic", 0.0)),
                weights=self.intrinsic_weights,
                gamma=1.0,
                is_terminal=np.array([True], dtype=bool),
            )
            intrinsic = float(np.asarray(r_huber, dtype=np.float32).reshape(-1)[0])
        else:
            r_l2, _, _ = hiro_utils.intrinsic_reward_l2(
                ego_next_sub_rel=ego_rel_H[None, :],
                goal_rel=goal_rel[None, :],
                norm_ranges=self.intrinsic_norm_ranges,
                coef=float(self.w.get("intrinsic", 0.0)),
                weights=self.intrinsic_weights,
            )
            intrinsic = float(np.asarray(r_l2, dtype=np.float32).reshape(-1)[0])

        # external reward proxy terms aligned with scenario reward definitions
        comfort_step = -(np.clip(np.abs(acc_phys) / max(self.comfort_max_accel, 1e-6), 0.0, 1.0) ** 2) * self.dt
        comfort_sum = float(np.sum(comfort_step)) * float(self.w.get("comfort_reward", 0.0))

        progress_step = np.zeros((Hn,), dtype=np.float32)
        if Hn > 0:
            delta_x = np.diff(states[:, 0], prepend=states[0, 0]).astype(np.float32)
            progress_step = np.clip(delta_x[1:] / max(self.goal_longitudinal, 1e-6), 0.0, 1.0)
        progress_w = 0.0 if self.disable_progress_when_huber else float(self.w.get("progress_reward", 0.0))
        progress_sum = float(np.sum(progress_step)) * progress_w

        on_road = 1.0  # with fixed y in-lane model
        on_road_sum = float(Hn) * on_road * float(self.w.get("on_road_reward", 1.0))

        low_ext_step = np.zeros((Hn,), dtype=np.float32)
        if Hn > 0:
            low_ext_step += np.asarray(progress_step * progress_w, dtype=np.float32)
            low_ext_step += np.asarray(comfort_step * float(self.w.get("comfort_reward", 0.0)), dtype=np.float32)
        intrinsic_step = np.zeros((Hn,), dtype=np.float32)
        if Hn > 0:
            intrinsic_step[-1] = np.float32(intrinsic)
        low_total_step = low_ext_step + intrinsic_step

        return {
            "intrinsic_step": intrinsic_step,
            "comfort_step": np.asarray(comfort_step * float(self.w.get("comfort_reward", 0.0)), dtype=np.float32),
            "low_ext_step": low_ext_step,
            "low_total_step": low_total_step,
            "sum_low_ext": float(np.sum(low_ext_step)),
            "sum_intrinsic": float(np.sum(intrinsic_step)),
            "sum_low_total": float(np.sum(low_total_step)),
            "reward_components": {
                "progress_reward": progress_sum,
                "comfort_reward": comfort_sum,
                "on_road_reward": on_road_sum,
                "intrinsic_reward": float(np.sum(intrinsic_step)),
            },
            "goal_rel": (goal - states[0, :4]).astype(np.float32),
            "eval_step_idx": int(Hn),
        }

    def plan(
        self,
        goal_phys: Sequence[float],
        steps_to_goal: int,
        maxiter: int = 200,
        enforce_collision_constraints: bool = True,
        collision_relax_eps: float = 0.0,
    ) -> Dict[str, Any]:
        _ = steps_to_goal
        _ = collision_relax_eps

        goal = np.asarray(goal_phys, dtype=np.float32).reshape(-1)
        if goal.shape[0] < 4:
            raise ValueError("goal_phys must contain at least [x, y, vx, vy].")

        start_state, neighbors_state = self._current_state()
        qp = self._build_qp(start_state, neighbors_state, goal[:4])

        if not bool(enforce_collision_constraints):
            qp["A"] = np.zeros((0, self.horizon), dtype=np.float32)
            qp["b"] = np.zeros((0,), dtype=np.float32)

        sol = self._solve_qp_solver(
            H=qp["H"],
            f=qp["f"],
            A=qp["A"],
            b=qp["b"],
            lb=qp["lb"],
            ub=qp["ub"],
            maxiter=int(maxiter),
        )

        u_seq = np.asarray(sol["x"], dtype=np.float32).reshape(-1)
        rollout = self._rollout_from_u(start_state, u_seq)
        eval_res = self._evaluate_result(rollout["states"], goal[:4], rollout["acc_phys"])

        ok = float(sol["max_violation"]) <= 1e-3
        msg = "qp solver converged" if ok else "qp solver finished with constraint violation"

        return {
            "success": bool(ok),
            "message": msg,
            "iterations": int(maxiter),
            "fun": float(sol["obj"]),
            "best_actions_cont": rollout["actions_cont"],
            "start_state": start_state,
            "neighbors_state": neighbors_state,
            "goal_phys": goal[:4].copy(),
            "horizon": int(self.horizon),
            "steps_to_goal": int(steps_to_goal),
            "enforce_collision_constraints": bool(enforce_collision_constraints),
            "collision_relax_eps": float(collision_relax_eps),
            "states": rollout["states"],
            "acc_norm": rollout["acc_norm"],
            "acc_phys": rollout["acc_phys"],
            "max_constraint_violation": float(sol["max_violation"]),
            "progress_objective_enabled": bool(not self.disable_progress_when_huber),
            "approximation_notes": [
                "lane-change fixed to KEEP (lane_scalar=0); only longitudinal action optimized",
                "neighbors predicted with constant velocity",
                "collision penalty converted to hard linear headway constraints",
                f"intrinsic_type={self.intrinsic_type} uses quadratic terminal proxy in solver",
                "progress objective disabled when intrinsic_type is huber_shaping",
            ],
            "solver": {
                "type": "qp",
                "status": str(sol.get("status", "")),
                "tried": list(sol.get("solver_tried", [])),
            },
            **eval_res,
        }

    def evaluate_action_sequence(
        self,
        actions_cont: Sequence[Sequence[float]],
        goal_phys: Sequence[float],
        steps_to_goal: int,
    ) -> Dict[str, Any]:
        """Evaluate a given low-level action sequence without optimization.

        Parameters
        ----------
        actions_cont:
            Sequence of low-level actions in RL format [lane_scalar, acc_norm].
        goal_phys:
            Absolute goal [x, y, vx, vy].
        steps_to_goal:
            Metadata only, kept for interface consistency.
        """
        goal = np.asarray(goal_phys, dtype=np.float32).reshape(-1)
        if goal.shape[0] < 4:
            raise ValueError("goal_phys must contain at least [x, y, vx, vy].")

        actions = np.asarray(actions_cont, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < 2:
            raise ValueError("actions_cont must have shape [T, 2+] with columns [lane_scalar, acc_norm].")

        lane_scalar_seq = np.asarray(actions[:, 0], dtype=np.float32).reshape(-1)
        acc_norm_seq_raw = np.asarray(actions[:, 1], dtype=np.float32).reshape(-1)
        acc_norm_seq = np.clip(acc_norm_seq_raw, -1.0, 1.0).astype(np.float32)

        start_state, neighbors_state = self._current_state()
        rollout = self._rollout_joint_from_actions(start_state, lane_scalar_seq, acc_norm_seq)

        eval_res = self._evaluate_joint_result(
            states=rollout["states"],
            lane_change_step=rollout["lane_change_step"],
            lane_idx_seq=rollout["lane_idx_seq"],
            goal_phys=goal[:4],
            acc_phys=rollout["acc_phys"],
            neighbors_state=neighbors_state,
        )

        speed = rollout["states"][1:, 2] if rollout["states"].shape[0] > 1 else np.zeros((0,), dtype=np.float32)
        speed_violation = float(np.max(np.maximum(speed - self.speed_limit, 0.0))) if speed.size > 0 else 0.0
        acc_norm_violation = float(np.max(np.maximum(np.abs(acc_norm_seq_raw) - 1.0, 0.0))) if acc_norm_seq_raw.size > 0 else 0.0

        dist_to_discrete = np.minimum(
            np.minimum(np.abs(lane_scalar_seq + 1.0), np.abs(lane_scalar_seq - 0.0)),
            np.abs(lane_scalar_seq - 1.0),
        )
        lane_discrete_violation = float(np.max(dist_to_discrete)) if dist_to_discrete.size > 0 else 0.0

        collision_step = np.asarray(eval_res.get("collision_step", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        collision_count = int(np.sum(collision_step > 0.5))

        constraints_report = self._collect_step_constraint_violations(
            states=rollout["states"],
            lane_idx_seq=rollout["lane_idx_seq"],
            lane_scalar_raw=lane_scalar_seq,
            acc_norm_raw=acc_norm_seq_raw,
            neighbors_state=neighbors_state,
        )

        ok = (
            speed_violation <= 1e-6
            and acc_norm_violation <= 1e-6
            and lane_discrete_violation <= 1e-6
            and collision_count == 0
            and float(constraints_report.get("max_violation", 0.0)) <= 1e-6
        )

        return {
            "success": bool(ok),
            "message": "action sequence evaluation finished",
            "iterations": 0,
            "fun": float(-eval_res.get("sum_low_total", 0.0)),
            "best_actions_cont": rollout["actions_cont"],
            "start_state": start_state,
            "neighbors_state": neighbors_state,
            "goal_phys": goal[:4].copy(),
            "horizon": int(rollout["actions_cont"].shape[0]),
            "steps_to_goal": int(steps_to_goal),
            "enforce_collision_constraints": True,
            "collision_relax_eps": 0.0,
            "states": rollout["states"],
            "acc_norm": rollout["acc_norm"],
            "acc_phys": rollout["acc_phys"],
            "lane_idx_seq": rollout["lane_idx_seq"],
            "lane_change_step": rollout["lane_change_step"],
            "max_constraint_violation": float(constraints_report.get("max_violation", 0.0)),
            "progress_objective_enabled": bool(not self.disable_progress_when_huber),
            "sequence_check": {
                "constraint_check_mode": "optimizer_equivalent",
                "speed_violation": speed_violation,
                "acc_norm_violation": acc_norm_violation,
                "lane_discrete_violation": lane_discrete_violation,
                "collision_count": int(collision_count),
                "constraint_max_violation": float(constraints_report.get("max_violation", 0.0)),
                "violated_steps": list(constraints_report.get("violated_steps", [])),
                "violation_counts": dict(constraints_report.get("violation_counts", {})),
                "step_constraint_violations": list(constraints_report.get("per_step", [])),
            },
            "approximation_notes": [
                "provided action sequence is rolled out and evaluated without optimization",
                "acc_norm is clipped to [-1,1] for dynamics rollout",
                "lane_scalar is mapped to {-1,0,1} by threshold for lane transitions",
            ],
            **eval_res,
        }

    def act(self, goal_phys: Sequence[float], steps_to_goal: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        result = self.plan(goal_phys=goal_phys, steps_to_goal=steps_to_goal)
        return np.asarray(result["best_actions_cont"][0], dtype=np.float32), result
