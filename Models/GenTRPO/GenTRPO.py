import bisect
import copy
import random
from functools import partial

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib.common.utils import conjugate_gradient_solver
from stable_baselines3.common.distributions import kl_divergence
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F

from Models.SB3 import PPO, TRPO
from Models.Strategy import sampling_strategy


class ForwardDynamicsModel(nn.Module):
  def __init__(self, observation_space, action_space, hidden_dim=128):
    super().__init__()

    self.state_dim = np.prod(observation_space.shape)  # Compute total state dimension
    self.action_dim = action_space.shape[0]  # Continuous action space

    self.encoder = nn.Sequential(
      nn.Linear(self.state_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
    )

    self.forward_model = nn.Sequential(
      nn.Linear(hidden_dim + self.action_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
    )

  def forward(self, state, action):
    state = state.view(state.size(0), -1)  # Ensure correct shape
    h_s = self.encoder(state)
    action = action.view(action.size(0), -1)  # Ensure action is 2D
    x = th.cat([h_s, action], dim=-1)
    pred_h_next = self.forward_model(x)
    return h_s, pred_h_next


class GenerativeReplayBuffer:
  def __init__(self, real_capacity, synthetic_capacity, relevance_function, generative_model, batch_size):
    self.real_capacity = real_capacity
    self.synthetic_capacity = synthetic_capacity
    self.real_buffer = []
    self.synthetic_buffer = []
    self.relevance_function = relevance_function
    self.generative_model = generative_model
    self.batch_size = batch_size

  def add_real(self, transition):
    # transition is: (obs, action, returns, advantages, old_log_prob)
    obs, action, returns, advantages, old_log_prob = transition

    # Convert all to tensors if not already
    if not isinstance(obs, th.Tensor):
      obs = th.as_tensor(obs, dtype=th.float32)
    if not isinstance(action, th.Tensor):
      action = th.as_tensor(action, dtype=th.float32)
    if not isinstance(returns, th.Tensor):
      returns = th.as_tensor(returns, dtype=th.float32)
    if not isinstance(advantages, th.Tensor):
      advantages = th.as_tensor(advantages, dtype=th.float32)
    if not isinstance(old_log_prob, th.Tensor):
      old_log_prob = th.as_tensor(old_log_prob, dtype=th.float32)

    self.real_buffer.append((obs, action, returns, advantages, old_log_prob))
    if len(self.real_buffer) > self.real_capacity:
      self.real_buffer.pop(0)

  def generate_synthetic(self):
    if len(self.real_buffer) == 0:
      return

    # Sort by relevance
    scored_samples = sorted(self.real_buffer, key=self.relevance_function, reverse=True)
    # Pick top 10
    sampled_real = scored_samples[:10]

    synthetic_transitions = []
    for obs, action, returns, advantages, old_log_prob in sampled_real:
      with th.no_grad():
        # obs is shape [obs_dim], expand to [1, obs_dim] for get_distribution
        dist = self.generative_model.get_distribution(obs.unsqueeze(0))
        synthetic_action = dist.sample()[0]
      synthetic_transitions.append((obs, synthetic_action, returns, advantages, old_log_prob))

    self.synthetic_buffer.extend(synthetic_transitions)
    self.synthetic_buffer = self.synthetic_buffer[-self.synthetic_capacity :]

  def sample(self, num_samples):
    real_sample_size = num_samples // 2
    synthetic_sample_size = num_samples // 2

    real_sample = random.sample(self.real_buffer, min(real_sample_size, len(self.real_buffer)))
    synthetic_sample = random.sample(self.synthetic_buffer, min(synthetic_sample_size, len(self.synthetic_buffer)))

    return real_sample + synthetic_sample


class GenTRPO(TRPO):
  """
    Trust Region Policy Optimization with Generative Replay Buffer and entropy regulirized.
    """

  def __init__(self, epsilon=0.2, entropy_coef=0.01, sampling_coef=0.5, buffer_capacity=10000, batch_size=32, normalized_advantage=False, **kwargs):

    # Initializes TRPO with replay buffer integration.
    super().__init__(**kwargs)
    self.epsilon = epsilon
    self.batch_size = batch_size
    self.entropy_coef = entropy_coef
    self.normalize_advantage = normalized_advantage
    self.sampling_coef = sampling_coef
    self.forward_dynamics_model = ForwardDynamicsModel(observation_space=self.observation_space, action_space=self.action_space).to(self.device)

    self.replay_buffer = GenerativeReplayBuffer(
      real_capacity=buffer_capacity,
      synthetic_capacity=buffer_capacity,
      relevance_function=self._compute_relevance,
      generative_model=self.policy,
      batch_size=batch_size,
    )

  def _compute_relevance(self, transition):
    obs, action, _, _, _ = transition
    with th.no_grad():
      h_s, pred_h_next = self.forward_dynamics_model(obs.unsqueeze(0), action.unsqueeze(0))
    curiosity_score = 0.5 * th.norm(pred_h_next - h_s, p=2).item() ** 2
    return curiosity_score

  def train(self):
    """
        Perform a single policy and value update using PPO's clipped objective,
        augmented with generative replay samples.
        """
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)

    # Collect rollout data
    rollout_data_list = list(self.rollout_buffer.get())
    on_policy_obs = th.cat([rd.observations for rd in rollout_data_list])
    on_policy_actions = th.cat([rd.actions for rd in rollout_data_list])
    on_policy_returns = th.cat([rd.returns for rd in rollout_data_list])
    on_policy_advantages = th.cat([rd.advantages for rd in rollout_data_list])
    on_policy_old_log_prob = th.cat([rd.old_log_prob for rd in rollout_data_list])

    # Add to replay buffer
    for i in range(on_policy_obs.shape[0]):
      transition = (
        on_policy_obs[i].cpu(),
        on_policy_actions[i].cpu(),
        on_policy_returns[i].cpu(),
        on_policy_advantages[i].cpu(),
        on_policy_old_log_prob[i].cpu(),
      )
      self.replay_buffer.add_real(transition)

    # Generate synthetic samples
    self.replay_buffer.generate_synthetic()

    # Determine replay sample size based on entropy
    distribution = self.policy.get_distribution(on_policy_obs)
    entropy_mean = distribution.entropy().mean().item()
    num_replay_samples = sampling_strategy(entropy=entropy_mean, sampling_coef=self.sampling_coef, min_samples=0, max_samples=self.batch_size)

    if num_replay_samples > 0:
      replay_samples = self.replay_buffer.sample(num_replay_samples)
      if replay_samples:
        replay_obs, replay_actions, replay_returns, replay_advantages, replay_old_log_prob = zip(*replay_samples)
        replay_obs = th.stack(replay_obs).to(self.device)
        replay_actions = th.stack(replay_actions).to(self.device)
        replay_returns = th.stack(replay_returns).to(self.device)
        replay_advantages = th.stack(replay_advantages).to(self.device)
        replay_old_log_prob = th.stack(replay_old_log_prob).to(self.device)
        on_policy_obs = th.cat([on_policy_obs, replay_obs])
        on_policy_actions = th.cat([on_policy_actions, replay_actions])
        on_policy_returns = th.cat([on_policy_returns, replay_returns])
        on_policy_advantages = th.cat([on_policy_advantages, replay_advantages])
        on_policy_old_log_prob = th.cat([on_policy_old_log_prob, replay_old_log_prob])

    if isinstance(self.action_space, spaces.Discrete):
      on_policy_actions = on_policy_actions.long().flatten()

    # Normalize advantages if specified
    if self.normalize_advantage:
      on_policy_advantages = (on_policy_advantages - on_policy_advantages.mean()) / (on_policy_advantages.std() + 1e-8)

    # Single epoch update
    distribution = self.policy.get_distribution(on_policy_obs)
    log_prob = distribution.log_prob(on_policy_actions)
    entropy = distribution.entropy().mean()
    ratio = th.exp(log_prob - on_policy_old_log_prob)

    # PPO clipped objective
    surr1 = ratio * on_policy_advantages
    surr2 = th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * on_policy_advantages
    policy_loss = -th.min(surr1, surr2).mean()
    policy_loss -= self.entropy_coef * entropy  # Entropy regularization

    # Value loss
    values = self.policy.predict_values(on_policy_obs)
    value_loss = F.mse_loss(on_policy_returns, values.flatten())

    # Total loss
    loss = policy_loss + self.value_loss_coef * value_loss

    # Optimization step
    self.policy.optimizer.zero_grad()
    loss.backward()
    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
    self.policy.optimizer.step()

    # Logging
    with th.no_grad():
      approx_kl = kl_divergence(distribution, self.policy.get_distribution(on_policy_obs)).mean().item()

    self._n_updates += 1
    explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

    self.logger.record("train/loss", loss.item())
    self.logger.record("train/policy_loss", policy_loss.item())
    self.logger.record("train/value_loss", value_loss.item())
    self.logger.record("train/entropy", entropy.item())
    self.logger.record("train/approx_kl", approx_kl)
    self.logger.record("train/explained_variance", explained_var)
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


def sample_gentrpo_params(trial, n_actions, n_envs, additional_args):
  # Sampling core hyperparameters
  n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
  gamma = trial.suggest_categorical("gamma", [0.8, 0.85, 0.9, 0.95, 0.99])
  learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
  n_critic_updates = trial.suggest_categorical("n_critic_updates", [5, 10, 20, 25, 30])
  cg_max_steps = trial.suggest_categorical("cg_max_steps", [5, 10, 20, 25, 30])
  target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
  gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])

  batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])

  # Neural network architecture selection
  net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
  net_arch = {
    "small": dict(pi=[64, 64], vf=[64, 64]),
    "medium": dict(pi=[256, 256], vf=[256, 256]),
    "large": dict(pi=[400, 300], vf=[400, 300]),
  }[net_arch_type]

  # Activation function selection
  activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
  activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]

  # Entropy coefficient for regularization
  entropy_coef = trial.suggest_float("entropy_coef", -1, 1, step=0.01)
  sampling_coef = trial.suggest_float("sampling_coef", -1, 1, step=0.01)

  # Replay buffer capacity and reward threshold for buffer clearing
  buffer_capacity = trial.suggest_int("buffer_capacity", 10000, 100000, step=1000)

  epsilon = trial.suggest_float("epsilon", 0.1, 0.9, step=0.05)

  orthogonal_init = trial.suggest_categorical("orthogonal_init", [True, False])

  n_timesteps = trial.suggest_int("n_timesteps", 100000, 1000000, step=100000)
  n_envs_choice = [2, 4, 6, 8, 10]
  n_envs = trial.suggest_categorical("n_envs", n_envs_choice)

  normalized_advantage = trial.suggest_categorical("normalize_advantage", [True, False])

  # Returning the sampled hyperparameters as a dictionary
  return {
    "policy": "MlpPolicy",
    "n_timesteps": n_timesteps,
    "n_envs": n_envs,
    "epsilon": epsilon,
    "entropy_coef": entropy_coef,
    "sampling_coef": sampling_coef,
    "n_steps": n_steps,
    "batch_size": batch_size,
    "gamma": gamma,
    "cg_max_steps": cg_max_steps,
    "n_critic_updates": n_critic_updates,
    "target_kl": target_kl,
    "learning_rate": learning_rate,
    "gae_lambda": gae_lambda,
    "buffer_capacity": buffer_capacity,
    "policy_kwargs": dict(
      net_arch=net_arch,
      activation_fn=activation_fn,
      ortho_init=orthogonal_init,
    ),
    "normalize_advantage": normalized_advantage,
    **additional_args,
  }


class GenPPO(PPO):
  """
    Proximal Policy Optimization with Generative Replay Buffer and entropy regularization.
    """

  def __init__(self, entropy_coef=0.01, sampling_coef=0.5, buffer_capacity=10000, batch_size=64, normalize_advantage=True, **kwargs):
    super().__init__(**kwargs)
    self.entropy_coef = entropy_coef
    self.sampling_coef = sampling_coef
    self.batch_size = batch_size
    self.normalize_advantage = normalize_advantage
    self.forward_dynamics_model = ForwardDynamicsModel(observation_space=self.observation_space, action_space=self.action_space).to(self.device)
    self.replay_buffer = GenerativeReplayBuffer(
      real_capacity=buffer_capacity,
      synthetic_capacity=buffer_capacity,
      relevance_function=self._compute_relevance,
      generative_model=self.policy,
      batch_size=batch_size,
    )

  def _compute_relevance(self, transition):
    obs, action, _, _, _ = transition
    with th.no_grad():
      h_s, pred_h_next = self.forward_dynamics_model(obs.unsqueeze(0), action.unsqueeze(0))
    curiosity_score = 0.5 * th.norm(pred_h_next - h_s, p=2).item() ** 2
    return curiosity_score

  def train(self) -> None:
    """
        Update policy using the currently gathered rollout buffer.
        """
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    # Update optimizer learning rate
    self._update_learning_rate(self.policy.optimizer)
    # Compute current clip range
    clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
    # Optional: clip range for the value function
    if self.clip_range_vf is not None:
      clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

    entropy_losses = []
    pg_losses, value_losses = [], []
    clip_fractions = []

    # Collect rollout data
    rollout_data_list = list(self.rollout_buffer.get())
    on_policy_obs = th.cat([rd.observations for rd in rollout_data_list])
    on_policy_actions = th.cat([rd.actions for rd in rollout_data_list])
    on_policy_returns = th.cat([rd.returns for rd in rollout_data_list])
    on_policy_advantages = th.cat([rd.advantages for rd in rollout_data_list])
    on_policy_old_log_prob = th.cat([rd.old_log_prob for rd in rollout_data_list])

    # Add to replay buffer
    for i in range(on_policy_obs.shape[0]):
      transition = (
        on_policy_obs[i].cpu(),
        on_policy_actions[i].cpu(),
        on_policy_returns[i].cpu(),
        on_policy_advantages[i].cpu(),
        on_policy_old_log_prob[i].cpu(),
      )
      self.replay_buffer.add_real(transition)

    # Generate synthetic samples
    self.replay_buffer.generate_synthetic()

    # Determine replay sample size based on entropy
    distribution = self.policy.get_distribution(on_policy_obs)
    entropy_mean = distribution.entropy().mean().item()
    num_replay_samples = sampling_strategy(entropy=entropy_mean, sampling_coef=self.sampling_coef, min_samples=0, max_samples=self.batch_size)

    if num_replay_samples > 0:
      replay_samples = self.replay_buffer.sample(num_replay_samples)
      if replay_samples:
        replay_obs, replay_actions, replay_returns, replay_advantages, replay_old_log_prob = zip(*replay_samples)
        replay_obs = th.stack(replay_obs).to(self.device)
        replay_actions = th.stack(replay_actions).to(self.device)
        replay_returns = th.stack(replay_returns).to(self.device)
        replay_advantages = th.stack(replay_advantages).to(self.device)
        replay_old_log_prob = th.stack(replay_old_log_prob).to(self.device)
        on_policy_obs = th.cat([on_policy_obs, replay_obs])
        on_policy_actions = th.cat([on_policy_actions, replay_actions])
        on_policy_returns = th.cat([on_policy_returns, replay_returns])
        on_policy_advantages = th.cat([on_policy_advantages, replay_advantages])
        on_policy_old_log_prob = th.cat([on_policy_old_log_prob, replay_old_log_prob])

    if isinstance(self.action_space, spaces.Discrete):
      on_policy_actions = on_policy_actions.long().flatten()

    # Normalize advantages if specified
    if self.normalize_advantage and len(on_policy_advantages) > 1:
      on_policy_advantages = (on_policy_advantages - on_policy_advantages.mean()) / (on_policy_advantages.std() + 1e-8)

    # Train for n_epochs epochs
    continue_training = True

    # train for n_epochs epochs
    for epoch in range(self.n_epochs):
      approx_kl_divs = []
      for start_idx in range(0, len(on_policy_obs), self.batch_size):
        end_idx = min(start_idx + self.batch_size, len(on_policy_obs))
        batch_indices = th.arange(start_idx, end_idx)

        obs_batch = on_policy_obs[batch_indices]
        actions_batch = on_policy_actions[batch_indices]
        returns_batch = on_policy_returns[batch_indices]
        advantages_batch = on_policy_advantages[batch_indices]
        old_log_prob_batch = on_policy_old_log_prob[batch_indices]

        values, log_prob, entropy = self.policy.evaluate_actions(obs_batch, actions_batch)
        values = values.flatten()

        # Ratio between old and new policy
        ratio = th.exp(log_prob - old_log_prob_batch)

        # Clipped surrogate loss
        policy_loss_1 = advantages_batch * ratio
        policy_loss_2 = advantages_batch * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        # Logging
        pg_losses.append(policy_loss.item())
        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
        clip_fractions.append(clip_fraction)

        if self.clip_range_vf is None:
          # No clipping
          values_pred = values
        else:
          # Clip the difference between old and new value
          # NOTE: this depends on the reward scaling
          values_pred = returns_batch + th.clamp(values - returns_batch, -clip_range_vf, clip_range_vf)

        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(returns_batch, values_pred)
        value_losses.append(value_loss.item())

        # Entropy loss favor exploration
        if entropy is None:
          # Approximate entropy when no analytical form
          entropy_loss = -th.mean(-log_prob)
        else:
          entropy_loss = -th.mean(entropy)

        entropy_losses.append(entropy_loss.item())

        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with th.no_grad():
          log_ratio = log_prob - old_log_prob_batch
          approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
          approx_kl_divs.append(approx_kl_div)

        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
          continue_training = False
          if self.verbose >= 1:
            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
          break

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()

        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

      self._n_updates += 1
      if not continue_training:
        break

    explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

    # Logs
    self.logger.record("train/entropy_loss", np.mean(entropy_losses))
    self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
    self.logger.record("train/clip_fraction", np.mean(clip_fractions))
    self.logger.record("train/loss", loss.item())
    self.logger.record("train/explained_variance", explained_var)
    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/clip_range", clip_range)
    if self.clip_range_vf is not None:
      self.logger.record("train/clip_range_vf", clip_range_vf)


def sample_genppo_params(trial, n_actions, n_envs, additional_args):
  n_steps = trial.suggest_categorical("n_steps", [64, 128, 256, 512, 1024])
  gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.99])
  learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
  n_epochs = 10
  clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
  gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.99])
  batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

  net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium"])
  net_arch = {"small": [64, 64], "medium": [256, 256]}[net_arch_type]
  activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
  activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn_name]

  entropy_coef = trial.suggest_float("entropy_coef", 0.0, 0.1, step=0.01)
  sampling_coef = trial.suggest_float("sampling_coef", 0.1, 1.0, step=0.1)
  buffer_capacity = trial.suggest_int("buffer_capacity", 5000, 50000, step=5000)

  # always 100000 timesteps
  n_timesteps = trial.suggest_categorical("n_timesteps", [100000])

  return {
    "policy": "MlpPolicy",
    "n_timesteps": n_timesteps,
    "n_steps": n_steps,
    "batch_size": batch_size,
    "gamma": gamma,
    "learning_rate": learning_rate,
    "n_epochs": n_epochs,
    "clip_range": clip_range,
    "gae_lambda": gae_lambda,
    "entropy_coef": entropy_coef,
    "sampling_coef": sampling_coef,
    "buffer_capacity": buffer_capacity,
    "policy_kwargs": dict(net_arch=dict(pi=net_arch, vf=net_arch), activation_fn=activation_fn),
    "normalize_advantage": True,
    **additional_args,
  }
