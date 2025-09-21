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


class AutoPlotProgressWrapper:
  def __init__(self, env, config, env_id, log_dir=".omega", plot_interval=100):
    self.env = env
    self.config = config
    self.env_id = env_id
    self.log_dir = log_dir
    self.plot_interval = plot_interval
    self.timesteps = 0
    self.rewards = []
    os.makedirs(self.log_dir, exist_ok=True)
    self.step_size = env.num_envs if hasattr(env, "num_envs") else 1

  def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)
    self.timesteps += self.step_size
    mean_reward = np.mean(reward) if isinstance(reward, np.ndarray) else float(reward)
    self.rewards.append(mean_reward)

    if self.timesteps % self.plot_interval == 0:
      self._save_rewards()
      self._plot_progress()

    return obs, reward, terminated, truncated, info

  def reset(self, **kwargs):
    self.timesteps = 0
    return self.env.reset(**kwargs)

  def _save_rewards(self):
    config_name = self.config.upper().replace("_", " ")
    path = os.path.join(self.log_dir, f"rewards_{self.env_id}_{self.config}.yaml")
    with open(path, "w") as f:
      yaml.safe_dump({"rewards": self.rewards}, f, default_flow_style=False)

  def _plot_progress(self):
    plt.figure(figsize=(10, 6))
    plt.clf()

    configs = [
      "trpo",
      "gentrpo",
      "grentrpo-ne",
    ]
    for cfg in configs:
      path = os.path.join(self.log_dir, f"rewards_{self.env_id}_{cfg}.yaml")
      if os.path.exists(path):
        with open(path, "r") as f:
          data = yaml.safe_load(f)
        rewards = data.get("rewards", [])
        if rewards:
          timesteps = np.arange(1, len(rewards) + 1) * self.step_size
          if len(rewards) >= 100:
            smoothed = np.convolve(rewards, np.ones(100) / 100, mode="valid")
            smoothed_timesteps = np.arange(50, len(smoothed) + 50) * self.step_size
            plt.plot(
              smoothed_timesteps,
              smoothed,
              label=cfg.upper().replace("-", " "),
            )
          else:
            plt.plot(timesteps, rewards, label=cfg.upper().replace("-", " "))

    plt.title(f"Training Progress per Configuration - {self.env_id}")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(self.log_dir, f"graph_{self.env_id}.png"))
    plt.close()


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
  def __init__(
    self,
    models,
    memory=None,
    cfg=None,
    observation_space=None,
    action_space=None,
    device=None,
  ):
    super().__init__(
      models=models,
      memory=memory,
      cfg=cfg,
      observation_space=observation_space,
      action_space=action_space,
      device=device,
    )
    if str(self.device) != "cuda":
      raise ValueError("Device must be cuda for GPU acceleration")
    print(f"Using device: {self.device} for audit purposes")
    self.rewards = []

  def post_interaction(self, timestep, timesteps):
    super().post_interaction(timestep, timesteps)
    if self.memory is not None:
      rewards = self.memory.get_tensor_by_name("rewards")
      if rewards.numel() > 0:
        self.rewards.append(float(rewards.mean()))


class TRPOWrapper:
  def __init__(
    self,
    env=None,
    learning_rate=1e-3,
    n_steps=2048,
    batch_size=128,
    gamma=0.99,
    cg_max_steps=15,
    cg_damping=0.1,
    line_search_shrinking_factor=0.8,
    line_search_max_iter=10,
    n_critic_updates=10,
    gae_lambda=0.95,
    normalize_advantage=True,
    target_kl=0.01,
    policy_kwargs=None,
    seed=None,
    device="cuda",
    ent_coef=0.0,
    noise_type="uniform",
    noise_level=0.0,
    **kwargs,
  ):
    self.env = env
    self.learning_rate = learning_rate
    self.n_steps = n_steps
    self.batch_size = batch_size
    self.gamma = gamma
    self.cg_max_steps = cg_max_steps
    self.cg_damping = cg_damping
    self.line_search_shrinking_factor = line_search_shrinking_factor
    self.line_search_max_iter = line_search_max_iter
    self.n_critic_updates = n_critic_updates
    self.gae_lambda = gae_lambda
    self.normalize_advantage = normalize_advantage
    self.target_kl = target_kl
    self.policy_kwargs = policy_kwargs or {}
    self.seed = seed
    self.device = th.device("cuda")
    self.ent_coef = ent_coef
    self.noise_type = noise_type
    self.noise_level = noise_level

    self._setup_model()

  def _setup_model(self):
    if isinstance(self.env, str):
      self.env = gym.make(self.env)
    self.env = wrap_env(self.env, wrapper="gymnasium")

    if self.seed is not None:
      np.random.seed(self.seed)
      th.manual_seed(self.seed)

    net_arch = self.policy_kwargs.get("net_arch", dict(pi=[256, 256], vf=[256, 256]))
    pi_arch = net_arch.get("pi", [256, 256])
    vf_arch = net_arch.get("vf", [256, 256])

    shared_net = nn.Sequential().to(self.device)
    input_size = self.env.observation_space.shape[0]
    for size in pi_arch[:-1]:
      shared_net.append(nn.Linear(input_size, size).to(self.device))
      shared_net.append(nn.Tanh())
      input_size = size

    class Policy(Model):
      def __init__(self, observation_space, action_space, device):
        super().__init__(observation_space, action_space, device)

        self.shared_net = shared_net
        self.mean_layer = nn.Linear(input_size, action_space.shape[0]).to(self.device)
        self.log_std_parameter = nn.Parameter(th.zeros(action_space.shape[0]).to(self.device))

      def compute(self, inputs, role):
        mean = self.mean_layer(self.shared_net(inputs["states"]))
        log_std = th.clamp(self.log_std_parameter, -20, 2).expand_as(mean)
        return mean, log_std, {}

      def act(self, inputs, role, taken_action=None, inference=False):
        mean, log_std, _ = self.compute(inputs, role)
        std = log_std.exp()
        dist = th.distributions.Normal(mean, std)
        if taken_action is None:
          if inference:
            action = mean
          else:
            action = dist.rsample()
        else:
          action = taken_action
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob, {"mean_actions": mean, "log_std": log_std}

      def get_log_std(self, role=""):
        return th.clamp(self.log_std_parameter, -20, 2)

      def distribution(self, role):
        log_std = th.clamp(self.log_std_parameter, -20, 2)
        std = log_std.exp()
        mean = th.zeros_like(std, device=self.device)
        return th.distributions.Normal(mean, std)

    class Value(Model):
      def __init__(self, observation_space, action_space, device):
        super().__init__(observation_space, action_space, device)

        self.shared_net = shared_net
        self.value_layer = nn.Linear(input_size, 1).to(self.device)

      def compute(self, inputs, role):
        return self.value_layer(self.shared_net(inputs["states"]))

      def act(self, inputs, role, taken_action=None, inference=False):
        value = self.compute(inputs, role)
        return value, {}, {}

    policy = Policy(self.env.observation_space, self.env.action_space, self.device)
    value = Value(self.env.observation_space, self.env.action_space, self.device)
    models = {"policy": policy, "value": value}

    memory = RandomMemory(
      memory_size=self.n_steps,
      num_envs=self.env.num_envs,
      device=self.device,
      replacement=False,
    )

    cfg = TRPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = self.n_steps
    cfg["batch_size"] = self.batch_size
    cfg["discount_factor"] = self.gamma
    cfg["conjugate_gradient_iterations"] = self.cg_max_steps
    cfg["damping"] = self.cg_damping
    cfg["backtrack_ratio"] = self.line_search_shrinking_factor
    cfg["line_search_iterations"] = self.line_search_max_iter
    cfg["value_solver_iterations"] = self.n_critic_updates
    cfg["gae_lambda"] = self.gae_lambda
    cfg["normalize_advantage"] = self.normalize_advantage
    cfg["kl_threshold"] = self.target_kl
    cfg["value_learning_rate"] = self.learning_rate
    cfg["value_clip"] = False
    cfg["clip_actions"] = False

    self.agent = agent_class(
      models=models,
      memory=memory,
      cfg=cfg,
      observation_space=self.env.observation_space,
      action_space=self.env.action_space,
      device=self.device,
    )

  def learn(self, total_timesteps: int, log_interval=1):
    trainer = SequentialTrainer(
      cfg={
        "timesteps": total_timesteps,
        "headless": True,
        "log_interval": log_interval,
      },
      env=self.env,
      agents=self.agent,
    )
    trainer.train()
