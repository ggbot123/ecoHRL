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

SQRT_2 = 1.4142135623730951
LOG_2PI = 1.8378770664093453


def _phi_cdf(x: th.Tensor) -> th.Tensor:
    return 0.5 * (1.0 + th.erf(x / SQRT_2))


def _phi_icdf(p: th.Tensor) -> th.Tensor:
    return SQRT_2 * th.erfinv(2.0 * p - 1.0)


def _atanh_clipped(x: th.Tensor, eps: float) -> th.Tensor:
    x = th.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (th.log1p(x) - th.log1p(-x))


def _logsubexp(x: th.Tensor, y: th.Tensor) -> th.Tensor:
    """Compute log(exp(x) - exp(y)) stably for x >= y."""
    return x + th.log1p(-th.exp(th.clamp(y - x, max=0.0)))


def _log_trunc_standard_normal_mass(a: th.Tensor, b: th.Tensor, mass_eps: float) -> th.Tensor:
    """Stable log(Phi(max(a,b)) - Phi(min(a,b)))."""
    a64 = a.to(dtype=th.float64)
    b64 = b.to(dtype=th.float64)
    lo = th.minimum(a64, b64)
    hi = th.maximum(a64, b64)

    log_phi_hi = th.special.log_ndtr(hi)
    log_phi_lo = th.special.log_ndtr(lo)
    log_m1 = _logsubexp(log_phi_hi, log_phi_lo)

    log_phi_neg_lo = th.special.log_ndtr(-lo)
    log_phi_neg_hi = th.special.log_ndtr(-hi)
    log_m2 = _logsubexp(log_phi_neg_lo, log_phi_neg_hi)

    log_m = th.maximum(log_m1, log_m2)
    floor = th.log(th.tensor(float(mass_eps), dtype=log_m.dtype, device=log_m.device))
    if th.any(log_m < floor):
        pass
    log_m = th.maximum(log_m, floor)
    return log_m.to(dtype=a.dtype)


def _sample_trunc_standard_normal(a: th.Tensor, b: th.Tensor, xi: th.Tensor, eps: float) -> th.Tensor:
    """Sample epsilon from N(0,1) truncated to [a, b] robustly."""
    phi_a = _phi_cdf(a)
    phi_b = _phi_cdf(b)

    lo = th.minimum(phi_a, phi_b)
    hi = th.maximum(phi_a, phi_b)
    span = th.clamp(hi - lo, min=0.0)

    p = lo + xi * span
    p = th.where(span > eps, th.clamp(p, lo + eps, hi - eps), lo)
    p = th.clamp(p, eps, 1.0 - eps)

    e = _phi_icdf(p)
    return th.clamp(e, min=th.minimum(a, b), max=th.maximum(a, b))


class SafeGoalActor(Actor):
    """Actor that samples dim-1 from the base policy and constrains dim-0 afterwards.

    User-defined sampling process:
    1) sample a full base squashed-Gaussian action;
    2) determine k = seg(a1) from dim-1 action using the fixed three segments
       [-1, -1/3], [-1/3, 1/3], [1/3, 1];
    3) conditioned on k, resample dim-0 inside the safe interval of that segment;
    4) keep dim-1 and dims >= 2 unchanged from the base sample.

    For samples whose selected segment has a valid dim-0 safe interval:
        log pi_safe(a|s) = log pi_base(a|s) - log Z2_k(s)

    where Z2_k is the base Gaussian mass of dim-0 over the selected segment's
    safe interval. If the selected segment has no valid dim-0 interval, this
    implementation falls back to the original base sample on dim-0 and applies
    no log-prob correction for that sample.
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

    @staticmethod
    def _segment_index_from_action(a1: th.Tensor) -> th.Tensor:
        """Map dim-1 action in [-1, 1] to one of the three fixed segments.

        Boundary convention:
            a1 <= -1/3 -> 0
            -1/3 < a1 < 1/3 -> 1
            a1 >= 1/3 -> 2
        Exact-boundary probability is zero under continuous sampling, so this
        convention only matters for deterministic evaluation.
        """
        k = th.full_like(a1, 2, dtype=th.long)
        k = th.where(a1 < (1.0 / 3.0), th.ones_like(k), k)
        k = th.where(a1 <= (-1.0 / 3.0), th.zeros_like(k), k)
        return k

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
        safe_stats = self._safe_stats(obs, mean_actions, std)

        if deterministic:
            z = mean_actions.clone()
        else:
            z = mean_actions + std * th.randn_like(mean_actions)

        y_base = th.tanh(z)
        k = self._segment_index_from_action(y_base[:, 1])
        valid_sel = safe_stats["valid_k"][idx, k]

        a2 = safe_stats["a2"][idx, k]
        b2 = safe_stats["b2"][idx, k]

        if deterministic:
            z0_safe = th.clamp(mean_actions[:, 0], min=safe_stats["alpha2"][idx, k], max=safe_stats["beta2"][idx, k])
        else:
            xi2 = th.rand((batch,), dtype=mean_actions.dtype, device=mean_actions.device)
            eps2 = _sample_trunc_standard_normal(a2, b2, xi2, eps)
            z0_safe = mean_actions[:, 0] + std[:, 0] * eps2

        z[:, 0] = th.where(valid_sel, z0_safe, z[:, 0])
        y = th.tanh(z)

        log_prob_base = self._log_prob_base(mean_actions, log_std, z, y, eps)
        log_z2 = safe_stats["log_z2_mass"][idx, k]
        corr = th.where(valid_sel, -log_z2, th.zeros_like(log_z2))
        log_prob_safe = log_prob_base + corr
        print(log_z2.min())
        if th.any(log_z2 < -32):
            pass
        # print(log_prob_safe.min(), log_prob_safe.max())
        # if log_prob_safe.max() < 23.39 and log_prob_safe.max() > 23.38:
        #     pass
        return y, log_prob_safe

    def base_log_prob_from_action(self, mean_actions: th.Tensor, log_std: th.Tensor, actions: th.Tensor) -> th.Tensor:
        eps = float(self.goal_safe_eps)
        z = _atanh_clipped(actions, eps)
        return self._log_prob_base(mean_actions, log_std, z, actions, eps)

    def _safe_stats(self, obs: th.Tensor, mean_actions: th.Tensor, std: th.Tensor) -> dict[str, th.Tensor]:
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

        alpha2 = _atanh_clipped(l2, eps)
        beta2 = _atanh_clipped(u2, eps)
        mu2 = mean_actions[:, 0:1]
        sigma2 = std[:, 0:1]
        a2 = (alpha2 - mu2) / sigma2
        b2 = (beta2 - mu2) / sigma2
        log_z2_mass = _log_trunc_standard_normal_mass(a2, b2, mass_eps=self.goal_safe_log_eps)

        return {
            "l2": l2,
            "u2": u2,
            "valid_k": valid_k,
            "alpha2": alpha2,
            "beta2": beta2,
            "a2": a2,
            "b2": b2,
            "log_z2_mass": log_z2_mass,
        }

    @classmethod
    def _log_prob_base(cls, mean_actions: th.Tensor, log_std: th.Tensor, z: th.Tensor, y: th.Tensor, eps: float) -> th.Tensor:
        """Base (unconstrained) log-prob across all action dims."""
        std = th.exp(log_std)
        normalized = (z - mean_actions) / std
        log_phi = -0.5 * (normalized.pow(2) + LOG_2PI)

        log_pz = th.sum(log_phi - log_std, dim=1)
        log_det = th.sum(th.log(1.0 - y.pow(2) + eps), dim=1)
        return log_pz - log_det


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
    ):
        self.goal_safe_eps = float(goal_safe_eps)
        self.goal_safe_log_eps = float(goal_safe_log_eps)
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
        ).to(self.device)


SafeGoalMlpPolicy = SafeGoalSACPolicy
