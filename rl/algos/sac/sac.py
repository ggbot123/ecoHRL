import csv
import json
import os
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import (
    GymEnv,
    MaybeCallback,
    RolloutReturn,
    Schedule,
    TrainFreq,
    TrainFrequencyUnit,
)
from stable_baselines3.common.utils import (
    get_parameters_by_name,
    polyak_update,
    should_collect_more_steps,
)
from stable_baselines3.common.vec_env import VecEnv
from rl.algos.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy
from rl.algos.sac.replay_buffer import SkipReplayBuffer
from rl.utils.numerics_guard import SACNumericsGuard
from rl.utils.utils import semantic_y_interval

SelfSAC = TypeVar("SelfSAC", bound="SAC")


class SAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param n_steps: When n_step > 1, uses n-step return (with the NStepReplayBuffer) when updating the Q-value network.
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`sac_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param safe_warmup_sampling: When True, warmup actions (before learning_starts) are sampled from policy
        instead of uniform random actions. Useful for hard safety constraints from step 0.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        safe_warmup_sampling: bool = False,
        numerics_guard: Optional[dict[str, Any]] = None,
        q_replay_debug: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class or SkipReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None
        self.safe_warmup_sampling = bool(safe_warmup_sampling)
        self.numerics_guard = SACNumericsGuard.from_dict(numerics_guard)
        self.q_replay_debug = dict(q_replay_debug or {})
        self.q_replay_debug_enabled = bool(self.q_replay_debug.get("enabled", False))
        self.q_replay_debug_file = None
        self.q_replay_debug_header_written = False
        self.q_replay_debug_rows_written = 0

        if _init_setup_model:
            self._setup_model()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """Collect rollouts while excluding inter-episode dummy transitions."""
        self.policy.set_training_mode(False)
        num_collected_steps = 0
        num_collected_transitions = 0
        num_collected_episodes = 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."
        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, (
                "You must use only one env when doing episodic training."
            )

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)  # type: ignore[operator]

        callback.on_rollout_start()
        target_transitions = int(train_freq.frequency) * int(env.num_envs)
        while (
            num_collected_transitions < target_transitions
            if train_freq.unit == TrainFrequencyUnit.STEP
            else should_collect_more_steps(
                train_freq,
                num_collected_steps,
                num_collected_episodes,
            )
        ):
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and num_collected_steps % self.sde_sample_freq == 0
            ):
                self.actor.reset_noise(env.num_envs)  # type: ignore[operator]

            actions, buffer_actions = self._sample_action(
                learning_starts,
                action_noise,
                env.num_envs,
            )
            new_obs, rewards, dones, infos = env.step(actions)
            infos = [
                info if isinstance(info, dict) else {}
                for info in list(infos)
            ]
            replay_mask = np.asarray(
                [not bool(info.get("skip_replay", False)) for info in infos],
                dtype=bool,
            )
            replay_count = int(np.count_nonzero(replay_mask))

            if replay_count == 0:
                self._store_transition(
                    replay_buffer,
                    buffer_actions,
                    new_obs,
                    rewards,
                    dones,
                    infos,
                )
                continue

            self.num_timesteps += replay_count
            num_collected_steps += 1
            num_collected_transitions += replay_count

            callback.update_locals(locals())
            if not callback.on_step():
                return RolloutReturn(
                    num_collected_transitions,
                    num_collected_episodes,
                    continue_training=False,
                )

            valid_infos = [
                info if replay_mask[i] else {}
                for i, info in enumerate(infos)
            ]
            valid_dones = np.asarray(dones, dtype=bool) & replay_mask
            self._update_info_buffer(valid_infos, valid_dones)

            self._store_transition(
                replay_buffer,
                buffer_actions,
                new_obs,
                rewards,
                dones,
                infos,
            )

            self._update_current_progress_remaining(
                self.num_timesteps,
                self._total_timesteps,
            )
            self._on_step()

            for idx in np.flatnonzero(valid_dones):
                num_collected_episodes += 1
                self._episode_num += 1
                if action_noise is not None:
                    kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                    action_noise.reset(**kwargs)
                if (
                    log_interval is not None
                    and self._episode_num % log_interval == 0
                ):
                    self.dump_logs()

        callback.on_rollout_end()
        return RolloutReturn(
            num_collected_transitions,
            num_collected_episodes,
            continue_training=True,
        )

    @staticmethod
    def _json_array(x: Any, precision: int = 6) -> str:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        return json.dumps([round(float(v), precision) for v in arr], separators=(",", ":"))

    def _q_debug_cfg(self, key: str, default: Any) -> Any:
        return self.q_replay_debug.get(key, default)

    def _init_q_replay_debug_writer(self) -> None:
        if self.q_replay_debug_file is not None:
            return
        save_dir = self._q_debug_cfg("save_dir", self.tensorboard_log)
        if not save_dir:
            self.q_replay_debug_enabled = False
            return
        os.makedirs(str(save_dir), exist_ok=True)
        file_name = str(self._q_debug_cfg("file_name", "q_replay_debug.csv"))
        self.q_replay_debug_file = os.path.join(str(save_dir), file_name)
        if not os.path.exists(self.q_replay_debug_file):
            header = [
                "algo_update",
                "gradient_step",
                "rank",
                "batch_row",
                "buffer_index",
                "env_id",
                "global_step",
                "segment_id",
                "interval_len",
                "target_q",
                "reward",
                "done",
                "discount",
                "next_q",
                "next_log_prob",
                "ent_coef",
                "current_q_min",
                "current_q_mean",
                "td_error_min",
                "td_error_mean",
                "next_q1",
                "next_q2",
                "current_q1",
                "current_q2",
                "comp_collision_reward",
                "comp_progress_reward",
                "comp_speed_ref_aux_reward",
                "comp_comfort_reward_for_high",
                "comp_lane_change_reward",
                "comp_goal_lane_dense_reward",
                "comp_punctual_reward",
                "comp_wrong_lane_terminal_penalty",
                "acc_min",
                "acc_max",
                "acc_mean",
                "acc_abs_mean",
                "hard_brake_frac",
                "hard_accel_frac",
                "ego_x",
                "ego_y",
                "ego_v",
                "next_ego_x",
                "next_ego_y",
                "next_ego_v",
                "action_scaled",
                "action_env",
                "original_action_scaled",
                "next_action_scaled",
                "obs",
                "next_obs",
            ]
            with open(self.q_replay_debug_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(header)

    def _unscale_debug_action(self, action_scaled: np.ndarray) -> np.ndarray:
        action_scaled = np.asarray(action_scaled, dtype=np.float32)
        try:
            return np.asarray(self.policy.unscale_action(action_scaled), dtype=np.float32)
        except Exception:
            return action_scaled

    def _extract_debug_ego(self, obs: np.ndarray) -> tuple[float, float, float]:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)
        # Flat high obs layout: [remaining_time, ego_presence, ego_x, ego_y, ego_vx, ego_vy, ...].
        if arr.size >= 5:
            return float(arr[2]), float(arr[3]), float(arr[4])
        return 0.0, 0.0, 0.0

    def _maybe_log_q_replay_debug(
        self,
        *,
        gradient_step: int,
        replay_data,
        discounts: Any,
        next_actions: th.Tensor,
        next_log_prob: th.Tensor,
        next_q_values: th.Tensor,
        target_q_values_all: th.Tensor,
        target_q_values: th.Tensor,
        current_q_values_all: th.Tensor,
        ent_coef: th.Tensor,
    ) -> dict[str, float]:
        if not self.q_replay_debug_enabled:
            return {}

        target_np = target_q_values.detach().cpu().numpy().reshape(-1)
        next_q_np = next_q_values.detach().cpu().numpy().reshape(-1)
        done_np = replay_data.dones.detach().cpu().numpy().reshape(-1)
        reward_np = replay_data.rewards.detach().cpu().numpy().reshape(-1)
        threshold = float(self._q_debug_cfg("target_q_lte", -20.0))
        next_threshold = float(self._q_debug_cfg("next_q_lte", threshold))
        low_mask = target_np <= threshold
        next_low_mask = (done_np < 0.5) & (next_q_np <= next_threshold)
        low_or_next = low_mask | next_low_mask

        summary: dict[str, float] = {
            "q_debug_low_target_frac": float(np.mean(low_mask)) if target_np.size else 0.0,
            "q_debug_low_next_frac": float(np.mean(next_low_mask)) if target_np.size else 0.0,
        }
        if np.any(low_or_next):
            summary["q_debug_low_reward_mean"] = float(np.mean(reward_np[low_or_next]))
            summary["q_debug_low_done_frac"] = float(np.mean(done_np[low_or_next]))

        sample_debug = getattr(self.replay_buffer, "last_sample_debug", None)
        comp_names = [
            "collision_reward",
            "progress_reward",
            "speed_ref_aux_reward",
            "comfort_reward_for_high",
            "lane_change_reward",
            "goal_lane_dense_reward",
            "punctual_reward",
            "wrong_lane_terminal_penalty",
        ]
        if isinstance(sample_debug, dict) and np.any(low_or_next):
            for name in comp_names:
                key = f"comp_{name}"
                if key in sample_debug:
                    vals = np.asarray(sample_debug[key], dtype=np.float32).reshape(-1)
                    if vals.size == low_or_next.size:
                        summary[f"q_debug_low_comp_{name}"] = float(np.mean(vals[low_or_next]))
            for name in ("acc_abs_mean", "hard_brake_frac", "hard_accel_frac"):
                if name in sample_debug:
                    vals = np.asarray(sample_debug[name], dtype=np.float32).reshape(-1)
                    if vals.size == low_or_next.size:
                        summary[f"q_debug_low_{name}"] = float(np.mean(vals[low_or_next]))

        period = int(self._q_debug_cfg("period_updates", 0))
        should_periodic = period > 0 and (int(self._n_updates) % period == 0)
        if not (np.any(low_or_next) or should_periodic):
            return summary

        self._init_q_replay_debug_writer()
        if self.q_replay_debug_file is None:
            return summary

        max_rows = int(self._q_debug_cfg("max_rows_per_update", 8))
        max_rows = max(max_rows, 1)
        max_total_rows = int(self._q_debug_cfg("max_total_rows", 200_000))
        if max_total_rows > 0 and self.q_replay_debug_rows_written >= max_total_rows:
            return summary
        candidate = np.flatnonzero(low_or_next)
        if candidate.size == 0:
            candidate = np.arange(target_np.size)
        order = candidate[np.argsort(target_np[candidate])[:max_rows]]
        if max_total_rows > 0:
            remaining = max_total_rows - self.q_replay_debug_rows_written
            if remaining <= 0:
                return summary
            order = order[:remaining]

        obs_np = replay_data.observations.detach().cpu().numpy()
        next_obs_np = replay_data.next_observations.detach().cpu().numpy()
        action_np = replay_data.actions.detach().cpu().numpy()
        next_action_np = next_actions.detach().cpu().numpy()
        next_log_np = next_log_prob.detach().cpu().numpy().reshape(-1)
        target_all_np = target_q_values_all.detach().cpu().numpy()
        current_all_np = current_q_values_all.detach().cpu().numpy()
        td_np = current_all_np - target_np.reshape(-1, 1)
        if isinstance(discounts, th.Tensor):
            discount_np = discounts.detach().cpu().numpy().reshape(-1)
        else:
            discount_np = np.full_like(target_np, float(discounts), dtype=np.float32)

        def meta(name: str, default: float | int = -1):
            if isinstance(sample_debug, dict) and name in sample_debug:
                return np.asarray(sample_debug[name]).reshape(-1)
            return np.full((target_np.size,), default)

        buffer_index = meta("batch_inds", -1)
        env_id = meta("env_id", -1)
        global_step = meta("global_step", -1)
        segment_id = meta("segment_id", -1)
        interval_len = meta("interval_len", 0)
        original_actions = (
            np.asarray(sample_debug.get("original_actions"), dtype=np.float32)
            if isinstance(sample_debug, dict) and "original_actions" in sample_debug
            else action_np
        )

        rows = []
        ent_coef_val = float(ent_coef.detach().cpu().item()) if isinstance(ent_coef, th.Tensor) else float(ent_coef)
        record_full_obs = bool(self._q_debug_cfg("record_full_obs", True))
        for rank, i in enumerate(order):
            ego_x, ego_y, ego_v = self._extract_debug_ego(obs_np[i])
            next_ego_x, next_ego_y, next_ego_v = self._extract_debug_ego(next_obs_np[i])
            comp_vals = []
            for name in comp_names:
                vals = meta(f"comp_{name}", 0.0)
                comp_vals.append(float(vals[i]))
            acc_vals = [float(meta(name, 0.0)[i]) for name in (
                "acc_min",
                "acc_max",
                "acc_mean",
                "acc_abs_mean",
                "hard_brake_frac",
                "hard_accel_frac",
            )]
            act_env = self._unscale_debug_action(action_np[i])
            rows.append([
                int(self._n_updates),
                int(gradient_step),
                int(rank),
                int(i),
                int(buffer_index[i]),
                int(env_id[i]),
                int(global_step[i]),
                int(segment_id[i]),
                int(interval_len[i]),
                float(target_np[i]),
                float(reward_np[i]),
                float(done_np[i]),
                float(discount_np[i]) if discount_np.size > i else float(discount_np[0]),
                float(next_q_np[i]),
                float(next_log_np[i]),
                ent_coef_val,
                float(np.min(current_all_np[i])),
                float(np.mean(current_all_np[i])),
                float(np.min(td_np[i])),
                float(np.mean(td_np[i])),
                float(target_all_np[i, 0]) if target_all_np.shape[1] > 0 else 0.0,
                float(target_all_np[i, 1]) if target_all_np.shape[1] > 1 else 0.0,
                float(current_all_np[i, 0]) if current_all_np.shape[1] > 0 else 0.0,
                float(current_all_np[i, 1]) if current_all_np.shape[1] > 1 else 0.0,
                *comp_vals,
                *acc_vals,
                ego_x,
                ego_y,
                ego_v,
                next_ego_x,
                next_ego_y,
                next_ego_v,
                self._json_array(action_np[i]),
                self._json_array(act_env),
                self._json_array(original_actions[i]),
                self._json_array(next_action_np[i]),
                self._json_array(obs_np[i]) if record_full_obs else "",
                self._json_array(next_obs_np[i]) if record_full_obs else "",
            ])
        with open(self.q_replay_debug_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        self.q_replay_debug_rows_written += len(rows)
        return summary

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def _sample_safe_reachable_warmup_action(self, n_envs: int) -> Optional[np.ndarray]:
        """Sample warmup actions uniformly from reachable safe segments when available."""
        if self._last_obs is None or not isinstance(self.action_space, spaces.Box):
            return None

        goal_safe_bounds_fn = getattr(self.policy.actor, "goal_safe_bounds_fn", None)
        if goal_safe_bounds_fn is None:
            return None

        obs_np = np.asarray(self._last_obs, dtype=np.float32)
        if obs_np.ndim == 1:
            obs_np = obs_np.reshape(1, -1)
        if obs_np.shape[0] != n_envs:
            return None

        with th.no_grad():
            obs_tensor = th.as_tensor(obs_np, device=self.device)
            bounds = goal_safe_bounds_fn(obs_tensor)

        required = ("l2", "u2")
        if any(key not in bounds for key in required):
            return None

        l2 = bounds["l2"].detach().cpu().numpy().astype(np.float32)
        u2 = bounds["u2"].detach().cpu().numpy().astype(np.float32)
        l_vx = bounds.get("l_vx")
        u_vx = bounds.get("u_vx")
        if l_vx is not None and u_vx is not None:
            l_vx = l_vx.detach().cpu().numpy().astype(np.float32)
            u_vx = u_vx.detach().cpu().numpy().astype(np.float32)
        else:
            l_vx = None
            u_vx = None

        if l2.ndim != 2 or u2.ndim != 2 or l2.shape[1] != 3 or u2.shape[1] != 3:
            return None

        low = self.action_space.low.astype(np.float32)
        high = self.action_space.high.astype(np.float32)
        actions = np.random.uniform(low=low, high=high, size=(n_envs, low.shape[0])).astype(np.float32)

        valid_box2 = u2 > l2
        valid_k = valid_box2
        valid_any = np.any(valid_k, axis=1)
        if not np.all(valid_any):
            # Fall back to policy sampling when any env has no feasible component.
            return None

        k = np.zeros((n_envs,), dtype=np.int64)
        for i in range(n_envs):
            candidates = np.flatnonzero(valid_k[i])
            k[i] = int(np.random.choice(candidates))

        row = np.arange(n_envs)
        dynamic_intervals = bool(
            getattr(self.policy.actor, "dynamic_feasible_lane_intervals", False)
        )
        if dynamic_intervals:
            lane_idx_value = bounds.get("ego_lane_idx")
            n_lanes_value = bounds.get("n_lanes")
            if lane_idx_value is None or n_lanes_value is None:
                return None
            if isinstance(lane_idx_value, th.Tensor):
                lane_idx_value = lane_idx_value.detach().cpu().numpy()
            if isinstance(n_lanes_value, th.Tensor):
                n_lanes_value = n_lanes_value.detach().cpu().numpy()
            lane_idx = np.asarray(lane_idx_value, dtype=np.int64).reshape(-1)
            n_lanes = int(np.asarray(n_lanes_value).reshape(-1)[0])
        else:
            lane_idx = np.zeros(n_envs, dtype=np.int64)
            n_lanes = 3
        y_intervals = np.asarray(
            [
                semantic_y_interval(
                    int(k[i]),
                    int(lane_idx[i]),
                    n_lanes,
                    dynamic_feasible_intervals=dynamic_intervals,
                )
                for i in range(n_envs)
            ],
            dtype=np.float32,
        )
        y_low = y_intervals[:, 0]
        y_high = y_intervals[:, 1]
        x_low_n = l2[row, k]
        x_high_n = u2[row, k]

        y_code = y_low + np.random.rand(n_envs).astype(np.float32) * (y_high - y_low)
        x_norm = x_low_n + np.random.rand(n_envs).astype(np.float32) * (x_high_n - x_low_n)
        x_norm = np.clip(x_norm, -1.0, 1.0)

        if actions.shape[1] >= 1:
            actions[:, 0] = low[0] + 0.5 * (x_norm + 1.0) * (high[0] - low[0])
        if actions.shape[1] >= 2:
            actions[:, 1] = np.clip(y_code, low[1], high[1])
        if actions.shape[1] >= 3 and l_vx is not None and u_vx is not None and l_vx.shape == l2.shape and u_vx.shape == u2.shape:
            v_low_n = l_vx[row, k]
            v_high_n = u_vx[row, k]
            v_valid = v_high_n > v_low_n
            if not np.all(v_valid):
                return None
            v_norm = v_low_n + np.random.rand(n_envs).astype(np.float32) * (v_high_n - v_low_n)
            actions[:, 2] = low[2] + 0.5 * (v_norm + 1.0) * (high[2] - low[2])

        return actions

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample action with optional policy-based warmup for hard safety from step 0."""
        if not self.safe_warmup_sampling or self.num_timesteps >= learning_starts:
            return super()._sample_action(learning_starts=learning_starts, action_noise=action_noise, n_envs=n_envs)

        if self._last_obs is None:
            return super()._sample_action(learning_starts=learning_starts, action_noise=action_noise, n_envs=n_envs)

        if not isinstance(self.action_space, spaces.Box):
            return super()._sample_action(learning_starts=learning_starts, action_noise=action_noise, n_envs=n_envs)

        action = self._sample_safe_reachable_warmup_action(n_envs=n_envs)
        if action is None:
            action, _ = self.predict(self._last_obs, deterministic=False)
            action = np.asarray(action, dtype=np.float32)

        buffer_action = self.policy.scale_action(action)
        buffer_action = np.clip(buffer_action, -1.0, 1.0)
        action = self.policy.unscale_action(buffer_action)
        return action, buffer_action

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if not hasattr(self.numerics_guard, "check_and_raise"):
            cfg = self.numerics_guard if isinstance(self.numerics_guard, dict) else None
            self.numerics_guard = SACNumericsGuard.from_dict(cfg)

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        target_q_means, target_q_absmaxs = [], []
        target_q_stds, target_q_p05s, target_q_p95s = [], [], []
        target_q_terminal_means, target_q_terminal_stds = [], []
        target_q_nonterminal_means, target_q_nonterminal_stds = [], []
        current_q_means, current_q_absmaxs = [], []
        td_error_means, td_error_stds, td_error_absmaxs = [], [], []
        done_batch_fracs = []
        reward_means, reward_absmaxs = [], []
        reward_stds = []
        next_q_means, next_q_absmaxs = [], []
        next_q_stds = []
        log_prob_means, log_prob_absmaxs = [], []
        q_replay_debug_summaries: dict[str, list[float]] = {}

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                target_q_values_all = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(target_q_values_all, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
                target_q_means.append(float(target_q_values.mean().detach().cpu().item()))
                target_q_absmaxs.append(float(target_q_values.abs().max().detach().cpu().item()))
                target_q_stds.append(float(target_q_values.std(unbiased=False).detach().cpu().item()))
                target_q_p05s.append(float(th.quantile(target_q_values.reshape(-1), 0.05).detach().cpu().item()))
                target_q_p95s.append(float(th.quantile(target_q_values.reshape(-1), 0.95).detach().cpu().item()))
                done_mask = replay_data.dones.reshape(-1) > 0.5
                done_batch_fracs.append(float(done_mask.float().mean().detach().cpu().item()))
                target_flat = target_q_values.reshape(-1)
                if bool(done_mask.any()):
                    target_terminal = target_flat[done_mask]
                    target_q_terminal_means.append(float(target_terminal.mean().detach().cpu().item()))
                    target_q_terminal_stds.append(float(target_terminal.std(unbiased=False).detach().cpu().item()))
                if bool((~done_mask).any()):
                    target_nonterminal = target_flat[~done_mask]
                    target_q_nonterminal_means.append(float(target_nonterminal.mean().detach().cpu().item()))
                    target_q_nonterminal_stds.append(float(target_nonterminal.std(unbiased=False).detach().cpu().item()))
                next_q_means.append(float(next_q_values.mean().detach().cpu().item()))
                next_q_absmaxs.append(float(next_q_values.abs().max().detach().cpu().item()))
                next_q_stds.append(float(next_q_values.std(unbiased=False).detach().cpu().item()))

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            current_q_values_all = th.cat(current_q_values, dim=1)
            current_q_means.append(float(current_q_values_all.mean().detach().cpu().item()))
            current_q_absmaxs.append(float(current_q_values_all.abs().max().detach().cpu().item()))
            reward_means.append(float(replay_data.rewards.mean().detach().cpu().item()))
            reward_absmaxs.append(float(replay_data.rewards.abs().max().detach().cpu().item()))
            reward_stds.append(float(replay_data.rewards.std(unbiased=False).detach().cpu().item()))
            log_prob_means.append(float(log_prob.mean().detach().cpu().item()))
            log_prob_absmaxs.append(float(log_prob.abs().max().detach().cpu().item()))
            td_error = current_q_values_all - target_q_values.detach()
            td_error_means.append(float(td_error.mean().detach().cpu().item()))
            td_error_stds.append(float(td_error.std(unbiased=False).detach().cpu().item()))
            td_error_absmaxs.append(float(td_error.abs().max().detach().cpu().item()))
            q_debug_summary = self._maybe_log_q_replay_debug(
                gradient_step=gradient_step,
                replay_data=replay_data,
                discounts=discounts,
                next_actions=next_actions,
                next_log_prob=next_log_prob.reshape(-1, 1),
                next_q_values=next_q_values,
                target_q_values_all=target_q_values_all,
                target_q_values=target_q_values,
                current_q_values_all=current_q_values_all,
                ent_coef=ent_coef,
            )
            for key, value in q_debug_summary.items():
                q_replay_debug_summaries.setdefault(key, []).append(float(value))

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            self.numerics_guard.check_and_raise(
                algo_update=self._n_updates,
                gradient_step=gradient_step,
                replay_data=replay_data,
                next_actions=next_actions,
                next_log_prob=next_log_prob.reshape(-1, 1),
                target_q_values_all=target_q_values_all,
                target_q_values=target_q_values,
                current_q_values_all=current_q_values_all,
                critic_loss=critic_loss,
            )

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss using the same sampled action/log-prob pair as in standard SAC.
            # The safe actor already returns actions sampled from the desired safe policy
            # together with the matching safe log-prob.
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/target_q_mean", np.mean(target_q_means))
        self.logger.record("train/target_q_std", np.mean(target_q_stds))
        self.logger.record("train/target_q_p05", np.mean(target_q_p05s))
        self.logger.record("train/target_q_p95", np.mean(target_q_p95s))
        self.logger.record("train/target_q_absmax", np.mean(target_q_absmaxs))
        if target_q_terminal_means:
            self.logger.record("train/target_q_terminal_mean", np.mean(target_q_terminal_means))
            self.logger.record("train/target_q_terminal_std", np.mean(target_q_terminal_stds))
        if target_q_nonterminal_means:
            self.logger.record("train/target_q_nonterminal_mean", np.mean(target_q_nonterminal_means))
            self.logger.record("train/target_q_nonterminal_std", np.mean(target_q_nonterminal_stds))
        self.logger.record("train/current_q_mean", np.mean(current_q_means))
        self.logger.record("train/current_q_absmax", np.mean(current_q_absmaxs))
        self.logger.record("train/td_error_mean", np.mean(td_error_means))
        self.logger.record("train/td_error_std", np.mean(td_error_stds))
        self.logger.record("train/td_error_absmax", np.mean(td_error_absmaxs))
        self.logger.record("train/done_batch_frac", np.mean(done_batch_fracs))
        self.logger.record("train/reward_batch_mean", np.mean(reward_means))
        self.logger.record("train/reward_batch_std", np.mean(reward_stds))
        self.logger.record("train/reward_batch_absmax", np.mean(reward_absmaxs))
        self.logger.record("train/next_q_mean", np.mean(next_q_means))
        self.logger.record("train/next_q_std", np.mean(next_q_stds))
        self.logger.record("train/next_q_absmax", np.mean(next_q_absmaxs))
        self.logger.record("train/log_prob_mean", np.mean(log_prob_means))
        self.logger.record("train/log_prob_absmax", np.mean(log_prob_absmaxs))
        for key, values in q_replay_debug_summaries.items():
            if values:
                self.logger.record(f"train/{key}", np.mean(values))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
