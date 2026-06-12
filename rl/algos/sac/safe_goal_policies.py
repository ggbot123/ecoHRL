from typing import Any, Callable, Optional, Union

import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
)
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from rl.algos.sac.policies import Actor, SACPolicy

LOG_2PI = 1.8378770664093453


def _atanh_clipped(x: th.Tensor, eps: float) -> th.Tensor:
    x = th.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (th.log1p(x) - th.log1p(-x))


class SafeGoalActor(Actor):
    """Actor that samples dim-1 from the base policy and maps dim-0 into safe bounds.
    1) sample a full base squashed-Gaussian action;
    2) determine k = seg(a1) from dim-1 action using either fixed thirds or
       state-dependent feasible intervals at boundary lanes;
    3) conditioned on k, sample latent xi0 ~ N(mu0, sigma0^2), then map
       a0 = m_k + r_k * tanh(xi0), where m_k=(l_k+u_k)/2 and r_k=(u_k-l_k)/2;
     4) keep dim-1 unchanged from the base sample, and map dim-2 (goal vx)
         into safe bounds when provided by bounds_fn.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
        goal_safe_eps: float = 1e-6,
        goal_safe_log_eps: float = 1e-30,
        goal_safe_bounds_fn: Optional[Callable[[th.Tensor], dict[str, th.Tensor]]] = None,
        dynamic_feasible_lane_intervals: bool = False,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=net_arch,
            features_extractor=features_extractor,
            features_dim=features_dim,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            clip_mean=clip_mean,
            normalize_images=normalize_images,
        )
        self.goal_safe_sampling_enabled = True
        self.goal_safe_eps = float(goal_safe_eps)
        self.goal_safe_log_eps = float(goal_safe_log_eps)
        self.goal_safe_bounds_fn = goal_safe_bounds_fn
        self.dynamic_feasible_lane_intervals = bool(dynamic_feasible_lane_intervals)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        if self.goal_safe_sampling_enabled and not self.use_sde:
            if not isinstance(obs, th.Tensor):
                raise TypeError("Safe goal sampling currently expects tensor observations")
            safe_actions, _ = self._safe_action_log_prob_from_params(
                obs,
                mean_actions,
                log_std,
                deterministic=deterministic,
            )
            return safe_actions
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: PyTorchObs) -> tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        if self.goal_safe_sampling_enabled and not self.use_sde:
            if not isinstance(obs, th.Tensor):
                raise TypeError("Safe goal sampling currently expects tensor observations")
            return self._safe_action_log_prob_from_params(obs, mean_actions, log_std, deterministic=False)
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _segment_index_from_action(
        self,
        a1: th.Tensor,
        safe_stats: dict[str, th.Tensor],
    ) -> th.Tensor:
        """Map dim-1 action to relative semantic component [LEFT, KEEP, RIGHT].

        Fixed-third boundary convention:
            a1 <= -1/3 -> 0
            -1/3 < a1 < 1/3 -> 1
            a1 >= 1/3 -> 2
        """
        k = th.full_like(a1, 2, dtype=th.long)
        k = th.where(a1 < (1.0 / 3.0), th.ones_like(k), k)
        k = th.where(a1 <= (-1.0 / 3.0), th.zeros_like(k), k)
        if not self.dynamic_feasible_lane_intervals:
            return k

        lane_idx = safe_stats.get("ego_lane_idx")
        n_lanes_t = safe_stats.get("n_lanes")
        if lane_idx is None or n_lanes_t is None:
            return k
        lane_idx = lane_idx.to(device=a1.device, dtype=th.long).reshape(-1)
        n_lanes = int(n_lanes_t.reshape(-1)[0].item())
        if n_lanes <= 1:
            return th.ones_like(k)

        left_edge = lane_idx == 0
        right_edge = lane_idx == (n_lanes - 1)
        k = th.where(left_edge, th.where(a1 > 0.0, 2, 1), k)
        k = th.where(right_edge, th.where(a1 < 0.0, 0, 1), k)
        return k

    def _segment_geometry(
        self,
        safe_stats: dict[str, th.Tensor],
        reference: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Return per-row center/radius for relative semantic y-code intervals."""
        batch = int(reference.shape[0])
        seg_m = reference.new_tensor([-2.0 / 3.0, 0.0, 2.0 / 3.0]).repeat(batch, 1)
        seg_r = reference.new_full((batch, 3), 1.0 / 3.0)
        if not self.dynamic_feasible_lane_intervals:
            return seg_m, seg_r

        lane_idx = safe_stats.get("ego_lane_idx")
        n_lanes_t = safe_stats.get("n_lanes")
        if lane_idx is None or n_lanes_t is None:
            return seg_m, seg_r
        lane_idx = lane_idx.to(device=reference.device, dtype=th.long).reshape(-1)
        n_lanes = int(n_lanes_t.reshape(-1)[0].item())
        if n_lanes <= 1:
            seg_m[:, 1] = 0.0
            seg_r[:, 1] = 1.0
            return seg_m, seg_r

        left_edge = lane_idx == 0
        right_edge = lane_idx == (n_lanes - 1)
        seg_m[left_edge, 1] = -0.5
        seg_r[left_edge, 1] = 0.5
        seg_m[left_edge, 2] = 0.5
        seg_r[left_edge, 2] = 0.5
        seg_m[right_edge, 0] = -0.5
        seg_r[right_edge, 0] = 0.5
        seg_m[right_edge, 1] = 0.5
        seg_r[right_edge, 1] = 0.5
        return seg_m, seg_r

    def _safe_action_log_prob_from_params(
        self,
        obs: th.Tensor,
        mean_actions: th.Tensor,
        log_std: th.Tensor,
        deterministic: bool,
    ) -> tuple[th.Tensor, th.Tensor]:
        eps = float(self.goal_safe_eps)
        if self.goal_safe_bounds_fn is None:
            raise RuntimeError("goal_safe_sampling_enabled=True requires goal_safe_bounds_fn")
        if mean_actions.shape[1] < 2:
            raise RuntimeError("Safe goal sampling requires action_dim >= 2")

        std = th.exp(log_std)
        batch = mean_actions.shape[0]
        idx = th.arange(batch, device=mean_actions.device)
        safe_stats = self._safe_stats(obs, mean_actions)

        if deterministic:
            z_base = mean_actions.clone()
        else:
            z_base = mean_actions + std * th.randn_like(mean_actions)

        y_base = th.tanh(z_base)
        k_base = self._segment_index_from_action(y_base[:, 1], safe_stats)
        valid_k = safe_stats["valid_k"]
        valid_sel_base = valid_k[idx, k_base]
        feasible_any = th.any(valid_k, dim=1)

        # If selected segment is infeasible, reroute to the nearest feasible segment.
        seg_m, seg_r = self._segment_geometry(safe_stats, mean_actions)
        y1_base = y_base[:, 1]
        dist_to_center = th.abs(y1_base[:, None] - seg_m)
        inf = th.full_like(dist_to_center, 1e9)
        dist_masked = th.where(valid_k, dist_to_center, inf)
        k_fallback = th.argmin(dist_masked, dim=1)
        use_fallback = (~valid_sel_base) & feasible_any
        k = th.where(use_fallback, k_fallback, k_base)
        valid_sel = valid_k[idx, k]

        m2_sel = safe_stats["m2"][idx, k]
        r2_sel = safe_stats["r2"][idx, k]
        r2_safe = th.clamp(r2_sel, min=eps)

        if deterministic:
            xi0 = mean_actions[:, 0]
        else:
            xi0 = mean_actions[:, 0] + std[:, 0] * th.randn((batch,), dtype=mean_actions.dtype, device=mean_actions.device)

        t0 = th.tanh(xi0)
        a0_safe = m2_sel + r2_sel * t0

        y = y_base.clone()
        y[:, 0] = th.where(valid_sel, a0_safe, y_base[:, 0])

        # For rerouted rows, map dim-1 latent sample into the selected feasible segment.
        t1 = th.tanh(z_base[:, 1])
        m1_sel = seg_m[idx, k]
        r1_sel = seg_r[idx, k]
        r1_safe = th.clamp(r1_sel, min=eps)
        a1_safe = m1_sel + r1_sel * t1
        y[:, 1] = th.where(use_fallback, a1_safe, y_base[:, 1])

        has_safe_vx = bool(safe_stats.get("has_safe_vx", False)) and mean_actions.shape[1] >= 3
        if has_safe_vx:
            m_vx_sel = safe_stats["m_vx"][idx, k]
            r_vx_sel = safe_stats["r_vx"][idx, k]
            r_vx_safe = th.clamp(r_vx_sel, min=eps)
            if deterministic:
                xi2 = mean_actions[:, 2]
            else:
                xi2 = mean_actions[:, 2] + std[:, 2] * th.randn((batch,), dtype=mean_actions.dtype, device=mean_actions.device)
            t2 = th.tanh(xi2)
            a2_safe = m_vx_sel + r_vx_sel * t2
            y[:, 2] = th.where(valid_sel, a2_safe, y_base[:, 2])
        else:
            xi2 = None
            t2 = None
            r_vx_safe = None

        dim_mask_other = th.ones_like(mean_actions, dtype=th.bool)
        dim_mask_other[:, 0] = False
        if has_safe_vx:
            dim_mask_other[:, 2] = False
        log_prob_other = self._log_prob_base_with_mask(mean_actions, log_std, z_base, y_base, eps, dim_mask_other)

        # Replace dim-1 base contribution only for rerouted rows.
        dim_mask_1 = th.zeros_like(mean_actions, dtype=th.bool)
        dim_mask_1[:, 1] = True
        log_prob_dim1_base = self._log_prob_base_with_mask(mean_actions, log_std, z_base, y_base, eps, dim_mask_1)
        normalized1 = (z_base[:, 1] - mean_actions[:, 1]) / std[:, 1]
        log_gauss1 = -0.5 * (normalized1.pow(2) + LOG_2PI) - log_std[:, 1]
        log_det1 = th.log(r1_safe) + th.log(1.0 - t1.pow(2) + eps)
        log_prob_dim1_safe = log_gauss1 - log_det1
        log_prob_other = log_prob_other + th.where(use_fallback, log_prob_dim1_safe - log_prob_dim1_base, th.zeros_like(log_prob_dim1_base))

        dim_mask_0 = th.zeros_like(mean_actions, dtype=th.bool)
        dim_mask_0[:, 0] = True
        log_prob_dim0_base = self._log_prob_base_with_mask(mean_actions, log_std, z_base, y_base, eps, dim_mask_0)

        normalized0 = (xi0 - mean_actions[:, 0]) / std[:, 0]
        log_gauss0 = -0.5 * (normalized0.pow(2) + LOG_2PI) - log_std[:, 0]
        log_det0 = th.log(r2_safe) + th.log(1.0 - t0.pow(2) + eps)
        log_prob_dim0_safe = log_gauss0 - log_det0

        log_prob_safe = log_prob_other + th.where(valid_sel, log_prob_dim0_safe, log_prob_dim0_base)
        if has_safe_vx and xi2 is not None and t2 is not None and r_vx_safe is not None:
            dim_mask_2 = th.zeros_like(mean_actions, dtype=th.bool)
            dim_mask_2[:, 2] = True
            log_prob_dim2_base = self._log_prob_base_with_mask(mean_actions, log_std, z_base, y_base, eps, dim_mask_2)
            normalized2 = (xi2 - mean_actions[:, 2]) / std[:, 2]
            log_gauss2 = -0.5 * (normalized2.pow(2) + LOG_2PI) - log_std[:, 2]
            log_det2 = th.log(r_vx_safe) + th.log(1.0 - t2.pow(2) + eps)
            log_prob_dim2_safe = log_gauss2 - log_det2
            log_prob_safe = log_prob_safe + th.where(valid_sel, log_prob_dim2_safe, log_prob_dim2_base)

        return y, log_prob_safe

    def base_log_prob_from_action(self, mean_actions: th.Tensor, log_std: th.Tensor, actions: th.Tensor) -> th.Tensor:
        eps = float(self.goal_safe_eps)
        z = _atanh_clipped(actions, eps)
        return self._log_prob_base(mean_actions, log_std, z, actions, eps)

    def _safe_stats(self, obs: th.Tensor, mean_actions: th.Tensor) -> dict[str, th.Tensor]:
        eps = float(self.goal_safe_eps)
        if self.goal_safe_bounds_fn is None:
            raise RuntimeError("goal_safe_bounds_fn is required")
        stats = self.goal_safe_bounds_fn(obs)

        required = ("l2", "u2")
        if any(key not in stats for key in required):
            raise KeyError(f"goal_safe_bounds_fn must return keys {required}")

        l2 = th.clamp(stats["l2"].to(dtype=mean_actions.dtype, device=mean_actions.device), -1.0 + eps, 1.0 - eps)
        u2 = th.clamp(stats["u2"].to(dtype=mean_actions.dtype, device=mean_actions.device), -1.0 + eps, 1.0 - eps)

        if l2.ndim != 2 or u2.ndim != 2 or l2.shape[1] != 3 or u2.shape[1] != 3:
            raise ValueError("l2/u2 must have shape [batch, 3]")

        batch = mean_actions.shape[0]
        if l2.shape[0] != batch or u2.shape[0] != batch:
            raise ValueError("Bounds batch dimension must match mean_actions")

        valid_k = u2 > l2
        m2 = 0.5 * (l2 + u2)
        r2 = 0.5 * (u2 - l2)

        has_safe_vx = False
        if mean_actions.shape[1] >= 3 and ("l_vx" in stats) and ("u_vx" in stats):
            l_vx = th.clamp(stats["l_vx"].to(dtype=mean_actions.dtype, device=mean_actions.device), -1.0 + eps, 1.0 - eps)
            u_vx = th.clamp(stats["u_vx"].to(dtype=mean_actions.dtype, device=mean_actions.device), -1.0 + eps, 1.0 - eps)
            if l_vx.shape != l2.shape or u_vx.shape != u2.shape:
                raise ValueError("l_vx/u_vx must have same shape as l2/u2")
            valid_k_vx = u_vx > l_vx
            valid_k = valid_k & valid_k_vx
            m_vx = 0.5 * (l_vx + u_vx)
            r_vx = 0.5 * (u_vx - l_vx)
            has_safe_vx = True
        else:
            l_vx = th.zeros_like(l2)
            u_vx = th.zeros_like(u2)
            m_vx = th.zeros_like(l2)
            r_vx = th.zeros_like(u2)

        out = {
            "l2": l2,
            "u2": u2,
            "valid_k": valid_k,
            "m2": m2,
            "r2": r2,
            "l_vx": l_vx,
            "u_vx": u_vx,
            "m_vx": m_vx,
            "r_vx": r_vx,
            "has_safe_vx": th.tensor(has_safe_vx, device=mean_actions.device).item(),
        }
        if "ego_lane_idx" in stats:
            out["ego_lane_idx"] = stats["ego_lane_idx"].to(device=mean_actions.device, dtype=th.long)
        if "n_lanes" in stats:
            out["n_lanes"] = stats["n_lanes"].to(device=mean_actions.device, dtype=th.long)
        return out

    @classmethod
    def _log_prob_base(cls, mean_actions: th.Tensor, log_std: th.Tensor, z: th.Tensor, y: th.Tensor, eps: float) -> th.Tensor:
        """Base (unconstrained) log-prob across all action dims."""
        dim_mask = th.ones_like(mean_actions, dtype=th.bool)
        return cls._log_prob_base_with_mask(mean_actions, log_std, z, y, eps, dim_mask)

    @classmethod
    def _log_prob_base_with_mask(
        cls,
        mean_actions: th.Tensor,
        log_std: th.Tensor,
        z: th.Tensor,
        y: th.Tensor,
        eps: float,
        dim_mask: th.Tensor,
    ) -> th.Tensor:
        """Base squashed-Gaussian log-prob over selected action dims."""
        std = th.exp(log_std)
        normalized = (z - mean_actions) / std
        log_phi = -0.5 * (normalized.pow(2) + LOG_2PI)
        log_terms = (log_phi - log_std) - th.log(1.0 - y.pow(2) + eps)
        masked_terms = th.where(dim_mask, log_terms, th.zeros_like(log_terms))
        return th.sum(masked_terms, dim=1)


class SafeGoalSACPolicy(SACPolicy):
    """SACPolicy that swaps base Actor for SafeGoalActor."""

    actor: SafeGoalActor

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
        goal_safe_eps: float = 1e-6,
        goal_safe_log_eps: float = 1e-30,
        dynamic_feasible_lane_intervals: bool = False,
    ):
        self.goal_safe_eps = float(goal_safe_eps)
        self.goal_safe_log_eps = float(goal_safe_log_eps)
        self.dynamic_feasible_lane_intervals = bool(dynamic_feasible_lane_intervals)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            use_sde=use_sde,
            log_std_init=log_std_init,
            use_expln=use_expln,
            clip_mean=clip_mean,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            n_critics=n_critics,
            share_features_extractor=share_features_extractor,
        )

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> SafeGoalActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return SafeGoalActor(
            **actor_kwargs,
            goal_safe_eps=self.goal_safe_eps,
            goal_safe_log_eps=self.goal_safe_log_eps,
            dynamic_feasible_lane_intervals=self.dynamic_feasible_lane_intervals,
        ).to(self.device)


SafeGoalMlpPolicy = SafeGoalSACPolicy
