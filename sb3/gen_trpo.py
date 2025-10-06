import copy
import random
from functools import partial

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib.common.utils import conjugate_gradient_solver
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.utils import explained_variance
from torch import nn

from sb3.networks import ForwardDynamicsModel, GenerativeReplayBuffer
from sb3.noise import MonitoredEntropyInjectionWrapper
from sb3.strategy import sampling_strategy
from sb3.trpo import TRPO


class GenTRPO(TRPO):
  """
    Trust Region Policy Optimization with Generative Replay Buffer and entropy regulirized.
    """

  def __init__(
    self,
    epsilon=0.2,
    entropy_coef=0.01,
    sampling_coef=0.5,
    buffer_capacity=10000,
    batch_size=32,
    normalized_advantage=False,
    noise_configs=None,
    dynamics_updates=5,
    fd_config=None,
    **kwargs
  ):

    # Initializes TRPO with replay buffer integration.
    super().__init__(noise_configs=noise_configs, **kwargs)

    self.epsilon = epsilon
    self.batch_size = batch_size
    self.entropy_coef = entropy_coef
    self.normalize_advantage = normalized_advantage
    self.sampling_coef = sampling_coef
    if fd_config is None:
      fd_config = {}
    self.forward_dynamics_model = ForwardDynamicsModel(observation_space=self.observation_space, action_space=self.action_space, **fd_config).to(self.device)
    self.dynamics_updates = dynamics_updates  # Number of update steps per training call, approximating lightweight updates (~5% of policy steps)

    self.replay_buffer = GenerativeReplayBuffer(
      real_capacity=buffer_capacity,
      synthetic_capacity=buffer_capacity,
      relevance_function=self._compute_relevance,
      generative_model=self.policy,
      batch_size=batch_size,
      device=self.device,
    )

  def _compute_relevance(self, transition):
    obs, action, next_obs, _, _, _ = transition
    obs = obs.to(self.device)
    action = action.to(self.device)
    next_obs = next_obs.to(self.device)
    with th.no_grad():
      h_s, pred_h_next = self.forward_dynamics_model(obs.unsqueeze(0), action.unsqueeze(0))
      h_next = self.forward_dynamics_model.encoder(next_obs.unsqueeze(0))
    curiosity_score = 0.5 * th.norm(pred_h_next - h_next, p=2).item() ** 2
    return curiosity_score

  def _compute_policy_objective(self, advantages, ratio, distribution):
    return super()._compute_policy_objective(advantages, ratio, distribution) + self.entropy_coef * distribution.entropy().mean()

  def _augment_data(self, on_policy_obs, on_policy_actions, on_policy_returns, on_policy_advantages, on_policy_old_log_prob):
    # Compute entropy and determine how many replay samples to mix in.
    distribution = self.policy.get_distribution(on_policy_obs)
    entropy_mean = distribution.entropy().mean().item()
    num_replay_samples = sampling_strategy(
      entropy=entropy_mean,
      sampling_coef=self.sampling_coef,
      min_samples=0,
      max_samples=self.batch_size,
    )

    if num_replay_samples > 0:
      # Only generate synthetic samples if we need replay samples (optimization: generate only what we need)
      # Aim to generate roughly half the num_replay_samples, since sample() takes half real/half synthetic
      num_to_generate = max(1, num_replay_samples // 2)
      self.replay_buffer.generate_synthetic(num_to_generate=num_to_generate)

      replay_samples = self.replay_buffer.sample(num_replay_samples)
      # unzip
      (
        replay_obs,
        replay_actions,
        replay_next_obs,  # Note: added but not used in policy update; for future extensions
        replay_returns,
        replay_advantages,
        replay_old_log_prob,
      ) = zip(*replay_samples)

      replay_obs = th.stack(replay_obs).to(self.device)
      replay_actions = th.stack(replay_actions).to(self.device)
      replay_returns = th.stack(replay_returns).to(self.device)
      replay_advantages = th.stack(replay_advantages).to(self.device)
      replay_old_log_prob = th.stack(replay_old_log_prob).to(self.device)

      # Concatenate replay samples with the current on-policy batch.
      on_policy_obs = th.cat([on_policy_obs, replay_obs])
      on_policy_actions = th.cat([on_policy_actions, replay_actions])
      on_policy_returns = th.cat([on_policy_returns, replay_returns])
      on_policy_advantages = th.cat([on_policy_advantages, replay_advantages])
      on_policy_old_log_prob = th.cat([on_policy_old_log_prob, replay_old_log_prob])

    return on_policy_obs, on_policy_actions, on_policy_returns, on_policy_advantages, on_policy_old_log_prob

  def train(self) -> None:
    # Updates the policy using on-policy rollouts augmented with prioritized replay samples.

    # Gather all on-policy rollout data and add each batch to the replay buffer.
    rollout_data_list = list(self.rollout_buffer.get())
    on_policy_obs = th.cat([rd.observations for rd in rollout_data_list])
    on_policy_actions = th.cat([rd.actions for rd in rollout_data_list])
    on_policy_returns = th.cat([rd.returns for rd in rollout_data_list])
    on_policy_advantages = th.cat([rd.advantages for rd in rollout_data_list])
    on_policy_old_log_prob = th.cat([rd.old_log_prob for rd in rollout_data_list])

    # Compute next_obs: shift on_policy_obs and append the final _last_obs
    last_next_obs = th.as_tensor(self._last_obs, dtype=th.float32, device=self.device)
    if last_next_obs.ndim == 1:  # Handle single env case
      last_next_obs = last_next_obs.unsqueeze(0)
    on_policy_next_obs = th.cat([on_policy_obs[1:], last_next_obs])

    # Add real transitions including next_obs
    for i in range(on_policy_obs.shape[0]):
      transition = (
        on_policy_obs[i].cpu(),  # single observation
        on_policy_actions[i].cpu(),  # single action
        on_policy_next_obs[i].cpu(),  # single next observation
        on_policy_returns[i].cpu(),  # single return
        on_policy_advantages[i].cpu(),  # single advantage
        on_policy_old_log_prob[i].cpu(),  # single old_log_prob
      )
      self.replay_buffer.add_real(transition)

    # Train the forward dynamics model on real data (online update as in the paper)
    # Only train if we have enough data and based on lightweight updates
    if len(self.replay_buffer.real_buffer) >= self.batch_size:
      batch = random.sample(self.replay_buffer.real_buffer, self.batch_size)
      obs_b = th.stack([t[0].to(self.device) for t in batch])
      act_b = th.stack([t[1].to(self.device) for t in batch])
      next_b = th.stack([t[2].to(self.device) for t in batch])
      self.forward_dynamics_model.learn(obs_b, act_b, next_b, updates=self.dynamics_updates)

    super().train()
