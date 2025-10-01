import argparse
import copy
import os
from typing import Any, Mapping, Optional, Tuple, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import yaml
from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.agents.torch.trpo import TRPO as SKRL_TRPO
from skrl.agents.torch.trpo import TRPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import Memory, RandomMemory
from skrl.models.torch import Model
from skrl.trainers.torch import SequentialTrainer
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters


def flat_grad(y, x, retain_graph=False, create_graph=False):
  if create_graph:
    retain_graph = True
  g = th.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
  g = th.cat([t.view(-1) for t in g if t is not None])
  return g


def get_flat_params(model):
  return th.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model, flat_params):
  n = 0
  for p in model.parameters():
    numel = p.numel()
    p.data.copy_(flat_params[n : n + numel].view(p.shape))
    n += numel


def conjugate_gradient(A, b, max_iterations=10, residual_tol=1e-10):
  x = th.zeros_like(b)
  r = b.clone()
  p = b.clone()
  rdotr_old = r.dot(r)
  if rdotr_old == 0:
    return x
  for i in range(max_iterations):
    Ap = A(p)
    alpha = rdotr_old / p.dot(Ap)
    x += alpha * p
    r -= alpha * Ap
    rdotr_new = r.dot(r)
    if rdotr_new < residual_tol:
      break
    beta = rdotr_new / rdotr_old
    p = r + beta * p
    rdotr_old = rdotr_new
  return x


class SKRL_TRPO_WITH_COLLECT(SKRL_TRPO):
  def __init__(self, models, memory=None, cfg=None, observation_space=None, action_space=None, device=None, **kwargs):
    super().__init__(
      models=models,
      memory=memory,
      cfg=cfg,
      observation_space=observation_space,
      action_space=action_space,
      device=device,
    )
    self.kwargs = kwargs
    if str(self.device) != "cuda":
      raise ValueError("Device must be cuda for GPU acceleration")
    print(f"Using device: {self.device} for audit purposes")
    self.rewards = []

  def post_interaction(self, timestep, timesteps):
    super().post_interaction(timestep, timesteps)
    if self.memory is not None:
      rewards = self.memory.get_tensor_by_name("rewards")
      if rewards.numel() > 0:
        mean_reward = th.nanmean(rewards).item()
        self.rewards.append(mean_reward)

    if (timestep + 1) % 100 == 0:
      self._save_rewards()

  def _save_rewards(self):

    variant = self.kwargs.get("variant", "default")
    log_dir = self.kwargs.get("log_dir", "runs")
    env_id = self.kwargs.get("env_id", "env")

    path = os.path.join(log_dir, f"{env_id}_{variant}.yaml")
    with open(path, "w") as f:
      yaml.safe_dump({"rewards": self.rewards}, f, default_flow_style=False)
