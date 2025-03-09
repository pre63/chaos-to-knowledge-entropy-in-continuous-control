import random

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import explained_variance
from torch import nn
from torch.nn import functional as F

from Models.SB3 import PPO  # Assuming this is defined elsewhere
from Models.Strategy import sampling_strategy  # Assuming this is defined elsewhere

# ForwardDynamicsModel (for relevance scoring in GenGRPO)


class ForwardDynamicsModel(nn.Module):
  def __init__(self, observation_space, action_space, hidden_dim=128):
    super().__init__()
    self.state_dim = np.prod(observation_space.shape)
    self.action_dim = action_space.shape[0] if isinstance(action_space, spaces.Box) else 1
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
    state = state.view(state.size(0), -1)
    h_s = self.encoder(state)
    action = action.view(action.size(0), -1)
    x = th.cat([h_s, action], dim=-1)
    pred_h_next = self.forward_model(x)
    return h_s, pred_h_next


# GenerativeReplayBuffer (used by GenGRPO)


class GenerativeReplayBuffer:
  def __init__(self, real_capacity, synthetic_capacity, relevance_function, generative_model, batch_size, group_size, env):
    self.real_capacity = real_capacity
    self.synthetic_capacity = synthetic_capacity
    self.real_buffer = []
    self.synthetic_buffer = []
    self.relevance_function = relevance_function
    self.generative_model = generative_model
    self.batch_size = batch_size
    self.group_size = group_size
    self.env = env

  def add_real(self, transition):
    state, actions, rewards, old_log_probs = transition
    if not isinstance(state, th.Tensor):
      state = th.as_tensor(state, dtype=th.float32)
    if not isinstance(actions, th.Tensor):
      actions = th.as_tensor(actions, dtype=th.float32)
    if not isinstance(rewards, th.Tensor):
      rewards = th.as_tensor(rewards, dtype=th.float32)
    if not isinstance(old_log_probs, th.Tensor):
      old_log_probs = th.as_tensor(old_log_probs, dtype=th.float32)
    self.real_buffer.append((state, actions, rewards, old_log_probs))
    if len(self.real_buffer) > self.real_capacity:
      self.real_buffer.pop(0)

  def generate_synthetic(self):
    if len(self.real_buffer) == 0:
      return
    scored_samples = sorted(self.real_buffer, key=self.relevance_function, reverse=True)
    sampled_real = scored_samples[:10]
    synthetic_transitions = []
    for state, _, _, _ in sampled_real:
      with th.no_grad():
        dist = self.generative_model.get_distribution(state.unsqueeze(0))
        synthetic_actions = dist.sample(sample_shape=(self.group_size,))
        synthetic_log_probs = dist.log_prob(synthetic_actions)
        synthetic_rewards = []
        self.env.reset()
        self.env.unwrapped.state = state.numpy()
        for action in synthetic_actions:
          _, reward, _, _ = self.env.step(action.numpy())
          synthetic_rewards.append(reward)
          self.env.unwrapped.state = state.numpy()
        synthetic_rewards = th.tensor(synthetic_rewards, dtype=th.float32)
      synthetic_transitions.append((state, synthetic_actions, synthetic_rewards, synthetic_log_probs))
    self.synthetic_buffer.extend(synthetic_transitions)
    self.synthetic_buffer = self.synthetic_buffer[-self.synthetic_capacity :]

  def sample(self, num_samples):
    real_sample_size = num_samples // 2
    synthetic_sample_size = num_samples // 2
    real_sample = random.sample(self.real_buffer, min(real_sample_size, len(self.real_buffer)))
    synthetic_sample = random.sample(self.synthetic_buffer, min(synthetic_sample_size, len(self.synthetic_buffer)))
    return real_sample + synthetic_sample


# Base GRPO class


class GRPO(PPO):
  """
    Group Relative Policy Optimization for robotic control.
    - Samples group_size actions per state, steps the environment for each.
    """

  def __init__(
    self,
    policy,
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.0,
    max_grad_norm=0.5,
    target_kl=None,
    group_size=4,
    verbose=0,
    device="auto",
    **kwargs
  ):
    super().__init__(
      policy=policy,
      env=env,
      learning_rate=learning_rate,
      n_steps=n_steps,
      batch_size=batch_size,
      n_epochs=n_epochs,
      gamma=gamma,
      gae_lambda=1.0,
      clip_range=clip_range,
      clip_range_vf=None,
      normalize_advantage=False,
      ent_coef=ent_coef,
      vf_coef=0.0,
      max_grad_norm=max_grad_norm,
      use_sde=False,
      target_kl=target_kl,
      verbose=verbose,
      device=device,
      **kwargs
    )
    self.group_size = group_size

  def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps):
    """
        Collect rollouts by stepping the environment group_size times per state.
        Store each action in the group as a separate buffer entry with repeated observations.
        Adjust step counting to fit buffer size.
        """
    assert self._last_obs is not None, "No previous observation"
    n_steps_collected = 0
    rollout_buffer.reset()

    # Adjust for group_size to avoid buffer overflow
    effective_steps = n_rollout_steps // self.group_size
    while n_steps_collected < effective_steps:
      with th.no_grad():
        obs_tensor = th.as_tensor(self._last_obs).to(self.device)
        dist = self.policy.get_distribution(obs_tensor)
        actions = th.stack([dist.sample() for _ in range(self.group_size)])  # [group_size, action_dim]
        log_probs = dist.log_prob(actions)  # [group_size]

      # Step the environment group_size times from the same state
      rewards = []
      next_obs = None
      dones = []
      initial_state = self._last_obs.copy()
      for i, action in enumerate(actions):
        action_np = action.cpu().numpy()
        new_obs, reward, done, info = env.step(action_np)
        rewards.append(reward)
        dones.append(done)
        if i < self.group_size - 1:
          env.reset()
          env.unwrapped.state = initial_state
        else:
          next_obs = new_obs

      rewards = np.array(rewards, dtype=np.float32)
      dones = np.array(dones, dtype=np.bool_)

      # Add each action in the group as a separate entry
      for i in range(self.group_size):
        self.num_timesteps += env.num_envs
        dummy_value = th.zeros(1, device=self.device)
        rollout_buffer.add(
          self._last_obs, actions[i].cpu().numpy(), rewards[i], self._last_episode_starts if i == 0 else False, dummy_value, log_probs[i]  # Keep as tensor
        )

      self._last_obs = next_obs
      self._last_episode_starts = dones[-1]
      n_steps_collected += 1

    return True

  def train(self):
    """
        Train with group-based advantages using real rewards.
        """
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)
    clip_range = self.clip_range(self._current_progress_remaining)

    entropy_losses, pg_losses = [], []
    clip_fractions, approx_kl_divs = []

    # Collect rollout data
    rollout_data = list(self.rollout_buffer.get())
    states = th.tensor(np.concatenate([rd.observations for rd in rollout_data]), device=self.device)
    actions = th.tensor(np.concatenate([rd.actions for rd in rollout_data]), device=self.device)
    rewards = th.tensor(np.concatenate([rd.rewards for rd in rollout_data]), device=self.device)
    old_log_probs = th.tensor(np.concatenate([rd.old_log_prob for rd in rollout_data]), device=self.device)

    # Ensure data aligns with group_size
    n_states = len(rewards) // self.group_size
    states = states[:n_states]
    actions = actions[: n_states * self.group_size].reshape(n_states, self.group_size, -1)
    rewards = rewards[: n_states * self.group_size].reshape(n_states, self.group_size)
    old_log_probs = old_log_probs[: n_states * self.group_size].reshape(n_states, self.group_size)

    # Compute group-based advantages
    group_means = rewards.mean沿着(dim=1, keepdim=True)
    advantages = rewards - group_means
    advantages = advantages.flatten()

    # Flatten for training
    obs = states.repeat_interleave(self.group_size, dim=0)
    actions = actions.reshape(-1, *actions.shape[2:])
    old_log_probs = old_log_probs.flatten()

    if isinstance(self.action_space, spaces.Discrete):
      actions = actions.long()

    # Training loop
    for epoch in range(self.n_epochs):
      indices = th.randperm(len(obs))
      for start in range(0, len(obs), self.batch_size):
        end = min(start + self.batch_size, len(obs))
        batch_idx = indices[start:end]

        obs_batch = obs[batch_idx]
        actions_batch = actions[batch_idx]
        advantages_batch = advantages[batch_idx]
        old_log_prob_batch = old_log_probs[batch_idx]

        _, log_prob, entropy = self.policy.evaluate_actions(obs_batch, actions_batch)

        # PPO clipped objective
        ratio = th.exp(log_prob - old_log_prob_batch)
        surr1 = ratio * advantages_batch
        surr2 = th.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages_batch
        policy_loss = -th.min(surr1, surr2).mean()

        # Entropy loss
        entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)

        # Total loss
        loss = policy_loss + self.ent_coef * entropy_loss

        # Logging
        pg_losses.append(policy_loss.item())
        entropy_losses.append(entropy_loss.item())
        clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())
        with th.no_grad():
          approx_kl = th.mean((th.exp(log_prob - old_log_prob_batch) - 1) - (log_prob - old_log_prob_batch)).item()
          approx_kl_divs.append(approx_kl)

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

      self._n_updates += 1

    # Logging
    self.logger.record("train/entropy_loss", np.mean(entropy_losses))
    self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
    self.logger.record("train/clip_fraction", np.mean(clip_fractions))
    self.logger.record("train/loss", loss.item())
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/clip_range", clip_range)


# GenGRPO inheriting from GRPO


class GenGRPO(GRPO):
  """
    Generative Group Relative Policy Optimization for robotic control.
    - Extends GRPO with generative replay.
    """

  def __init__(
    self,
    policy,
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    clip_range=0.2,
    ent_coef=0.0,
    entropy_coef=0.01,
    sampling_coef=0.5,
    buffer_capacity=10000,
    max_grad_norm=0.5,
    target_kl=None,
    group_size=4,
    verbose=0,
    device="auto",
    **kwargs
  ):
    super().__init__(
      policy=policy,
      env=env,
      learning_rate=learning_rate,
      n_steps=n_steps,
      batch_size=batch_size,
      n_epochs=n_epochs,
      gamma=gamma,
      clip_range=clip_range,
      ent_coef=ent_coef,
      max_grad_norm=max_grad_norm,
      target_kl=target_kl,
      group_size=group_size,
      verbose=verbose,
      device=device,
      **kwargs
    )
    self.entropy_coef = entropy_coef
    self.sampling_coef = sampling_coef
    self.forward_dynamics_model = ForwardDynamicsModel(observation_space=self.observation_space, action_space=self.action_space).to(self.device)
    self.replay_buffer = GenerativeReplayBuffer(
      real_capacity=buffer_capacity,
      synthetic_capacity=buffer_capacity,
      relevance_function=self._compute_relevance,
      generative_model=self.policy,
      batch_size=batch_size,
      group_size=group_size,
      env=env,
    )

  def _compute_relevance(self, transition):
    state, actions, _, _ = transition
    with th.no_grad():
      h_s, pred_h_next = self.forward_dynamics_model(state.unsqueeze(0), actions[0].unsqueeze(0))
    curiosity_score = 0.5 * th.norm(pred_h_next - h_s, p=2).item() ** 2
    return curiosity_score

  def train(self):
    """
        Train with group-based advantages and generative replay, overriding GRPO's train.
        """
    self.policy.set_training_mode(True)
    self._update_learning_rate(self.policy.optimizer)
    clip_range = self.clip_range(self._current_progress_remaining)

    entropy_losses, pg_losses = [], []
    clip_fractions, approx_kl_divs = []

    # Collect rollout data
    rollout_data = list(self.rollout_buffer.get())
    states = th.tensor(np.concatenate([rd.observations for rd in rollout_data]), device=self.device)
    actions = th.tensor(np.concatenate([rd.actions for rd in rollout_data]), device=self.device)
    rewards = th.tensor(np.concatenate([rd.rewards for rd in rollout_data]), device=self.device)
    old_log_probs = th.tensor(np.concatenate([rd.old_log_prob for rd in rollout_data]), device=self.device)

    # Add to replay buffer
    n_states = len(states)
    for i in range(n_states):
      transition = (
        states[i].cpu(),
        actions[i * self.group_size : (i + 1) * self.group_size].cpu(),
        rewards[i * self.group_size : (i + 1) * self.group_size].cpu(),
        old_log_probs[i * self.group_size : (i + 1) * self.group_size].cpu(),
      )
      self.replay_buffer.add_real(transition)

    # Generate and sample synthetic data
    self.replay_buffer.generate_synthetic()
    num_replay_samples = sampling_strategy(
      entropy=self.policy.get_distribution(states).entropy().mean().item(),
      sampling_coef=self.sampling_coef,
      min_samples=0,
      max_samples=self.batch_size // self.group_size,
    )
    if num_replay_samples > 0:
      replay_samples = self.replay_buffer.sample(num_replay_samples)
      if replay_samples:
        replay_states, replay_actions, replay_rewards, replay_old_log_probs = zip(*replay_samples)
        states = th.cat([states, th.stack(replay_states).to(self.device)])
        actions = th.cat([actions] + list(replay_actions)).to(self.device)
        rewards = th.cat([rewards] + list(replay_rewards)).to(self.device)
        old_log_probs = th.cat([old_log_probs] + list(replay_old_log_probs)).to(self.device)

    # Ensure data aligns with group_size
    n_states = len(rewards) // self.group_size
    states = states[:n_states]
    actions = actions[: n_states * self.group_size].reshape(n_states, self.group_size, -1)
    rewards = rewards[: n_states * self.group_size].reshape(n_states, self.group_size)
    old_log_probs = old_log_probs[: n_states * self.group_size].reshape(n_states, self.group_size)

    # Compute group-based advantages
    group_means = rewards.mean(dim=1, keepdim=True)
    advantages = rewards - group_means
    advantages = advantages.flatten()

    # Flatten for training
    obs = states.repeat_interleave(self.group_size, dim=0)
    actions = actions.reshape(-1, *actions.shape[2:])
    old_log_probs = old_log_probs.flatten()

    if isinstance(self.action_space, spaces.Discrete):
      actions = actions.long()

    # Training loop
    for epoch in range(self.n_epochs):
      indices = th.randperm(len(obs))
      for start in range(0, len(obs), self.batch_size):
        end = min(start + self.batch_size, len(obs))
        batch_idx = indices[start:end]

        obs_batch = obs[batch_idx]
        actions_batch = actions[batch_idx]
        advantages_batch = advantages[batch_idx]
        old_log_prob_batch = old_log_probs[batch_idx]

        _, log_prob, entropy = self.policy.evaluate_actions(obs_batch, actions_batch)

        # PPO clipped objective
        ratio = th.exp(log_prob - old_log_prob_batch)
        surr1 = ratio * advantages_batch
        surr2 = th.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages_batch
        policy_loss = -th.min(surr1, surr2).mean()

        # Entropy loss with generative regularization
        entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
        loss = policy_loss + self.ent_coef * entropy_loss - self.entropy_coef * entropy.mean()

        # Logging
        pg_losses.append(policy_loss.item())
        entropy_losses.append(entropy_loss.item())
        clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())
        with th.no_grad():
          approx_kl = th.mean((th.exp(log_prob - old_log_prob_batch) - 1) - (log_prob - old_log_prob_batch)).item()
          approx_kl_divs.append(approx_kl)

        # Optimization step
        self.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

      self._n_updates += 1

    # Logging
    self.logger.record("train/entropy_loss", np.mean(entropy_losses))
    self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
    self.logger.record("train/clip_fraction", np.mean(clip_fractions))
    self.logger.record("train/loss", loss.item())
    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/clip_range", clip_range)


# Example usage
if __name__ == "__main__":
  from stable_baselines3.common.env_util import make_vec_env

  env = make_vec_env("Pendulum-v1", n_envs=1)

  # Test GRPO
  grpo_model = GRPO(policy="MlpPolicy", env=env, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, clip_range=0.2, group_size=4, verbose=1)
  grpo_model.learn(total_timesteps=100000)

  # Test GenGRPO
  gengrpo_model = GenGRPO(
    policy="MlpPolicy", env=env, learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10, clip_range=0.2, group_size=4, buffer_capacity=10000, verbose=1
  )
  gengrpo_model.learn(total_timesteps=100000)
