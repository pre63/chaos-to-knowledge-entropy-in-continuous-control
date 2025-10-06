import random

import numpy as np
import optuna
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from torch import nn
from torch.optim import Adam


class ForwardDynamicsModel(nn.Module):
  def __init__(self, observation_space, action_space, hidden_dim=128, encoder_layers=2, forward_layers=3, activation="ReLU", lr=1e-3):
    super().__init__()

    self.state_dim = np.prod(observation_space.shape)  # Compute total state dimension
    self.action_dim = action_space.shape[0]  # Continuous action space
    self.hidden_dim = hidden_dim

    # Dynamically build encoder
    encoder_modules = []
    encoder_modules.append(nn.Linear(self.state_dim, hidden_dim))
    encoder_modules.append(getattr(nn, activation)())
    for _ in range(encoder_layers - 1):
      encoder_modules.append(nn.Linear(hidden_dim, hidden_dim))
      encoder_modules.append(getattr(nn, activation)())
    self.encoder = nn.Sequential(*encoder_modules)

    # Dynamically build forward_model
    forward_modules = []
    forward_modules.append(nn.Linear(hidden_dim + self.action_dim, hidden_dim))
    forward_modules.append(getattr(nn, activation)())
    for _ in range(forward_layers - 1):
      forward_modules.append(nn.Linear(hidden_dim, hidden_dim))
      forward_modules.append(getattr(nn, activation)())
    forward_modules.append(nn.Linear(hidden_dim, hidden_dim))
    self.forward_model = nn.Sequential(*forward_modules)

    self.optimizer = Adam(self.parameters(), lr=lr)

  def forward(self, state, action):
    state = state.view(state.size(0), -1)  # Ensure correct shape
    h_s = self.encoder(state)
    action = action.view(action.size(0), -1)  # Ensure action is 2D
    x = th.cat([h_s, action], dim=-1)
    pred_h_next = self.forward_model(x)
    return h_s, pred_h_next

  def learn(self, states, actions, next_states, updates=1):
    """
        Train the model for a given number of updates on the provided batch.
        """
    self.train()
    for _ in range(updates):
      h_s, pred_h_next = self(states, actions)
      with th.no_grad():
        h_next = self.encoder(next_states)
      loss = F.mse_loss(pred_h_next, h_next, reduction="mean")

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()


class GenerativeReplayBuffer:
  def __init__(
    self,
    real_capacity,
    synthetic_capacity,
    relevance_function,
    generative_model,
    batch_size,
    device,
  ):
    self.real_capacity = real_capacity
    self.synthetic_capacity = synthetic_capacity
    self.real_buffer = []
    self.synthetic_buffer = []
    self.relevance_function = relevance_function
    self.generative_model = generative_model
    self.batch_size = batch_size
    self.device = device

  def add_real(self, transition):
    obs, action, next_obs, returns, advantages, old_log_prob = transition
    if not isinstance(obs, th.Tensor):
      obs = th.as_tensor(obs, dtype=th.float32)
    if not isinstance(action, th.Tensor):
      action = th.as_tensor(action, dtype=th.float32)
    if not isinstance(next_obs, th.Tensor):
      next_obs = th.as_tensor(next_obs, dtype=th.float32)
    if not isinstance(returns, th.Tensor):
      returns = th.as_tensor(returns, dtype=th.float32)
    if not isinstance(advantages, th.Tensor):
      advantages = th.as_tensor(advantages, dtype=th.float32)
    if not isinstance(old_log_prob, th.Tensor):
      old_log_prob = th.as_tensor(old_log_prob, dtype=th.float32)

    self.real_buffer.append((obs, action, next_obs, returns, advantages, old_log_prob))
    if len(self.real_buffer) > self.real_capacity:
      self.real_buffer.pop(0)

  def generate_synthetic(self, num_to_generate=10):
    if len(self.real_buffer) == 0:
      return

    scored_samples = sorted(self.real_buffer, key=self.relevance_function, reverse=True)
    sampled_real = scored_samples[:num_to_generate]  # Use num_to_generate instead of fixed 10

    synthetic_transitions = []
    for obs, action, next_obs, returns, advantages, old_log_prob in sampled_real:
      with th.no_grad():
        obs = obs.to(self.device)
        dist = self.generative_model.get_distribution(obs.unsqueeze(0))
        synthetic_action = dist.sample()[0]
      # For simplicity, copy next_obs (ideally, generate/predict it to match paper)
      synthetic_transitions.append((obs, synthetic_action, next_obs, returns, advantages, old_log_prob))

    self.synthetic_buffer.extend(synthetic_transitions)
    self.synthetic_buffer = self.synthetic_buffer[-self.synthetic_capacity :]

  def sample(self, num_samples):
    real_sample_size = num_samples // 2
    synthetic_sample_size = num_samples - real_sample_size  # Adjust to handle odd numbers

    real_sample = random.sample(self.real_buffer, min(real_sample_size, len(self.real_buffer)))
    synthetic_sample = random.sample(
      self.synthetic_buffer,
      min(synthetic_sample_size, len(self.synthetic_buffer)),
    )

    real_sample = [
      (
        obs.to(self.device),
        action.to(self.device),
        next_obs.to(self.device),
        returns.to(self.device),
        advantages.to(self.device),
        old_log_prob.to(self.device),
      )
      for obs, action, next_obs, returns, advantages, old_log_prob in real_sample
    ]

    synthetic_sample = [
      (
        obs.to(self.device),
        action.to(self.device),
        next_obs.to(self.device),
        returns.to(self.device),
        advantages.to(self.device),
        old_log_prob.to(self.device),
      )
      for obs, action, next_obs, returns, advantages, old_log_prob in synthetic_sample
    ]

    return real_sample + synthetic_sample


def sample_fd_params(trial):
  """
    Optuna sampling function for ForwardDynamicsModel hyperparameters.
    Suggests values for all tunable parameters.
    """
  hidden_dim = trial.suggest_int("hidden_dim", 64, 512, log=True)
  encoder_layers = trial.suggest_int("encoder_layers", 1, 4)
  forward_layers = trial.suggest_int("forward_layers", 2, 5)
  activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "Tanh", "ELU"])
  lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

  return {"hidden_dim": hidden_dim, "encoder_layers": encoder_layers, "forward_layers": forward_layers, "activation": activation, "lr": lr}
