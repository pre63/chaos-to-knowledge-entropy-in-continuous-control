import random

import numpy as np
import torch as th
from torch import nn


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
    state = state.reshape(-1, self.state_dim)  # Flatten to (effective_batch, state_dim)
    action = action.reshape(-1, self.action_dim)  # Flatten to (effective_batch, action_dim)
    h_s = self.encoder(state)
    x = th.cat([h_s, action], dim=-1)
    pred_h_next = self.forward_model(x)
    return h_s, pred_h_next


class GenerativeReplayBuffer:
  def __init__(
    self,
    real_capacity,
    synthetic_capacity,
    relevance_function,
    generative_model,
    batch_size,
    device,
    action_dim,
  ):
    self.real_capacity = real_capacity
    self.synthetic_capacity = synthetic_capacity
    self.real_buffer = []
    self.synthetic_buffer = []
    self.relevance_function = relevance_function
    self.generative_model = generative_model
    self.batch_size = batch_size
    self.device = device
    self.action_dim = action_dim

  def add_real(self, transition):
    obs, action, return_, advantages, old_log_prob = transition
    if not isinstance(obs, th.Tensor):
      obs = th.as_tensor(obs, dtype=th.float32)
    obs = obs.view(-1)
    if obs.numel() == 0:
      obs = th.zeros(self.generative_model.observation_space.shape[0], dtype=th.float32)

    if not isinstance(action, th.Tensor):
      action = th.as_tensor(action, dtype=th.float32)
    action = action.view(-1)
    if action.numel() == 0:
      action = th.zeros(self.action_dim, dtype=th.float32)

    if not isinstance(return_, th.Tensor):
      return_ = th.as_tensor(return_, dtype=th.float32)
    if not isinstance(advantages, th.Tensor):
      advantages = th.as_tensor(advantages, dtype=th.float32)
    if not isinstance(old_log_prob, th.Tensor):
      old_log_prob = th.as_tensor(old_log_prob, dtype=th.float32)

    self.real_buffer.append((obs, action, return_, advantages, old_log_prob))
    if len(self.real_buffer) > self.real_capacity:
      self.real_buffer.pop(0)

  def generate_synthetic(self):
    if len(self.real_buffer) == 0:
      return

    scored_samples = sorted(self.real_buffer, key=self.relevance_function, reverse=True)
    sampled_real = scored_samples[:10]

    synthetic_transitions = []
    for obs, action, return_, advantages, old_log_prob in sampled_real:
      obs = obs.to(self.device)
      with th.no_grad():
        _, _, outputs = self.generative_model.act({"states": obs.unsqueeze(0)}, role="policy")
      mean = outputs["mean_actions"]
      std = outputs["log_std"].exp()
      dist = th.distributions.Normal(mean, std)
      synthetic_action = dist.sample().squeeze(0)
      synthetic_action = synthetic_action.view(-1)
      if synthetic_action.numel() == 0:
        synthetic_action = th.zeros(self.action_dim, device=self.device)
      synthetic_transitions.append((obs, synthetic_action, return_, advantages, old_log_prob))

    self.synthetic_buffer.extend(synthetic_transitions)
    self.synthetic_buffer = self.synthetic_buffer[-self.synthetic_capacity :]

  def sample(self, num_samples):
    real_sample_size = num_samples // 2
    synthetic_sample_size = num_samples // 2

    real_sample = random.sample(self.real_buffer, min(real_sample_size, len(self.real_buffer)))
    synthetic_sample = random.sample(
      self.synthetic_buffer,
      min(synthetic_sample_size, len(self.synthetic_buffer)),
    )

    real_sample = [
      (
        obs.to(self.device),
        action.to(self.device),
        return_.to(self.device),
        advantages.to(self.device),
        old_log_prob.to(self.device),
      )
      for obs, action, return_, advantages, old_log_prob in real_sample
    ]

    synthetic_sample = [
      (
        obs.to(self.device),
        action.to(self.device),
        return_.to(self.device),
        advantages.to(self.device),
        old_log_prob.to(self.device),
      )
      for obs, action, return_, advantages, old_log_prob in synthetic_sample
    ]

    return real_sample + synthetic_sample
