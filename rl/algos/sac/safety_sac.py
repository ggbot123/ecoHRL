from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.utils import polyak_update
from rl.algos.sac.sac import SAC


class SafetyLayerSAC(SAC):
    """SAC variant whose target-Q uses safety-filtered next actions."""

    def __init__(self, *args, target_action_filter: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_action_filter = target_action_filter

    def _filter_actions(self, observations: th.Tensor, actions: th.Tensor, keep_grad: bool) -> th.Tensor:
        if self.target_action_filter is None:
            return actions

        obs_np = observations.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        filtered_np = self.target_action_filter(obs_np, actions_np)
        filtered = th.as_tensor(filtered_np, device=actions.device, dtype=actions.dtype).view_as(actions)

        if not keep_grad:
            return filtered

        # Straight-through: forward uses filtered action, backward keeps d(filtered)/d(actions) ~= I.
        return actions + (filtered - actions).detach()

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

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

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            actions_pi_for_q = self._filter_actions(replay_data.observations, actions_pi, keep_grad=True)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_actions = self._filter_actions(replay_data.next_observations, next_actions, keep_grad=False)

                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
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
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi_for_q), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
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
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["target_action_filter"]
