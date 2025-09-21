import copy
from typing import Mapping, Optional, Tuple, Union

import gymnasium
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from skrl.models.torch import Model
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from skrli.gen_trpo import GEN_TRPO
from skrli.strategy import sampling_strategy


# GenTRPO-NE: Generative TRPO with entropy regularization and noise injection
class GEN_TRPO_NE(GEN_TRPO):
  def __init__(
    self,
    models,
    memory=None,
    cfg=None,
    observation_space=None,
    action_space=None,
    device=None,
    entropy_coef=0.01,
    noise_type="uniform",
    noise_level=0.0,
    **kwargs
  ):
    super().__init__(models=models, memory=memory, cfg=cfg, observation_space=observation_space, action_space=action_space, device=device, **kwargs)
    self._entropy_coef = entropy_coef
    self.noise_type = noise_type
    self.noise_level = noise_level
    self.base_std = 1.0
    self.base_range = 1.0
    self.base_scale = 1.0
    self.base_p = 0.5

  def _update(self, timestep, timesteps):
    super()._update(timestep, timesteps)

  def surrogate_loss(self, policy: Model, states: th.Tensor, actions: th.Tensor, log_prob: th.Tensor, advantages: th.Tensor) -> th.Tensor:
    """Override to include entropy"""
    _, new_log_prob, outputs = policy.act({"states": states, "taken_actions": actions}, role="policy")
    surrogate = (advantages * th.exp(new_log_prob - log_prob.detach())).mean()

    logstd = policy.get_log_std(role="policy")
    action_dim = logstd.shape[-1]
    entropy = 0.5 * action_dim * (th.log(2 * th.pi * th.e)) + logstd.sum(dim=-1)
    entropy = entropy.mean()

    return surrogate + self._entropy_coef * entropy

  def act(self, states, timestep=0, timesteps=1):
    actions, log_prob, infos = super().act(states, timestep, timesteps)
    if self.noise_level != 0:
      actions = self._add_noise(actions, "action")
      actions = th.clamp(actions, th.tensor(self.action_space.low, device=self.device), th.tensor(self.action_space.high, device=self.device))
    return actions, log_prob, infos

  def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
    if self.noise_level != 0:
      rewards = self._add_noise(rewards, "reward")
    super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

  def _add_noise(self, value, component):
    entropy_level = abs(self.noise_level)
    noise_type = self.noise_type
    if component == "action":
      if noise_type == "bernoulli":
        return value
      if noise_type == "gaussian":
        std = entropy_level * self.base_std
        return value + th.normal(0, std, size=value.shape, device=self.device)
      elif noise_type == "uniform":
        range_val = entropy_level * self.base_range
        return value + th.empty(size=value.shape, device=self.device).uniform_(-range_val, range_val)
      elif noise_type == "laplace":
        scale = entropy_level * self.base_scale
        return value + th.distributions.laplace.Laplace(0, scale).sample(value.shape).to(self.device)
    elif component == "reward":
      if noise_type == "gaussian":
        std = entropy_level * self.base_std
        return value + th.normal(0, std, size=value.shape, device=self.device)
      elif noise_type == "uniform":
        range_val = entropy_level * self.base_range
        return value + th.empty(size=value.shape, device=self.device).uniform_(-range_val, range_val)
      elif noise_type == "laplace":
        scale = entropy_level * self.base_scale
        return value + th.distributions.laplace.Laplace(0, scale).sample(value.shape).to(self.device)
      elif noise_type == "bernoulli":
        p = entropy_level * self.base_p
        mask = (th.rand(value.shape, device=self.device) < p).float()
        return value * (1 - mask)
    return value
