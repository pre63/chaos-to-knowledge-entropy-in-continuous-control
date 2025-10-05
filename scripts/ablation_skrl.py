# scripts/ablation_skrl.py
import argparse
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import yaml
from gymnasium.vector import SyncVectorEnv
from skrl.agents.torch.trpo import TRPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import Model
from skrl.trainers.torch import SequentialTrainer

from skrli.gen_trpo import GenTRPO as GEN_TRPO_SKRL
from skrli.trpo import SKRL_TRPO_WITH_COLLECT

th.backends.cudnn.benchmark = True
th.set_float32_matmul_precision("high")

TIMESTEPS = 100000
BUFFER_CAPACITY = 3072


def parse_params(p):
  parsed = {}
  for k, v in p.items():
    if isinstance(v, str):
      try:
        v = int(v)
      except ValueError:
        try:
          v = float(v)
        except ValueError:
          pass
    if k in ["normalize_advantage", "ortho_init", "orthogonal_init"]:
      v = bool(v)
    parsed[k] = v
  return parsed


def setup_agent_skrl(variant, p, env, env_id, device):
  net_arch = {
    "pi": [64, 64] if p["net_arch"] == "small" else [400, 300] if p["net_arch"] == "large" else [256, 256],
    "vf": [64, 64] if p["net_arch"] == "small" else [400, 300] if p["net_arch"] == "large" else [256, 256],
  }
  activation_fn_class = {"tanh": nn.Tanh, "relu": nn.ReLU}[p["activation_fn"]]

  shared_net = nn.Sequential().to(device)
  input_size = env.observation_space.shape[0]  # Corrected to use observation dimension
  print(f"Observation dimension for {variant}: {input_size}")  # Verify (should be 348 or 376 depending on Gymnasium version)
  for size in net_arch["pi"]:
    layer = nn.Linear(input_size, size).to(device)
    if p["ortho_init"]:
      gain = nn.init.calculate_gain(p["activation_fn"])
      nn.init.orthogonal_(layer.weight, gain=gain)
      nn.init.constant_(layer.bias, 0)
    shared_net.append(layer)
    shared_net.append(activation_fn_class())
    input_size = size

  class Policy(Model):
    def __init__(self, observation_space, action_space, device):
      super().__init__(observation_space, action_space, device)

      self.shared_net = shared_net
      self.mean_layer = nn.Linear(input_size, action_space.shape[0]).to(self.device)
      self.log_std_parameter = nn.Parameter(th.full((action_space.shape[0],), p.get("log_std_init", 0.0)).to(self.device))

      if p["ortho_init"]:
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0)

    def compute(self, inputs, role):
      x = inputs["states"]
      if not isinstance(x, th.Tensor):
        x = th.tensor(x, dtype=th.float32, device=self.device)
      mean = self.mean_layer(self.shared_net(x))
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

      if p["ortho_init"]:
        nn.init.orthogonal_(self.value_layer.weight, gain=1.0)
        nn.init.constant_(self.value_layer.bias, 0)

    def compute(self, inputs, role):
      x = inputs["states"]
      if not isinstance(x, th.Tensor):
        x = th.tensor(x, dtype=th.float32, device=self.device)
      return self.value_layer(self.shared_net(x))

    def act(self, inputs, role, taken_action=None, inference=False):
      value = self.compute(inputs, role)
      return value, {}, {}

  policy = Policy(env.observation_space, env.action_space, device)
  value = Value(env.observation_space, env.action_space, device)
  models = {"policy": policy, "value": value}

  memory = RandomMemory(
    memory_size=p.get("buffer_capacity", p["n_steps"]),
    num_envs=env.num_envs,
    device=device,
    replacement=False,
  )

  cfg = TRPO_DEFAULT_CONFIG.copy()
  cfg["rollouts"] = p["n_steps"]
  cfg["batch_size"] = p["batch_size"]
  cfg["discount_factor"] = p["gamma"]
  cfg["conjugate_gradient_iterations"] = p["cg_max_steps"]
  cfg["damping"] = p.get("cg_damping", 0.1)
  cfg["backtrack_ratio"] = p.get("line_search_shrinking_factor", 0.8)
  cfg["line_search_iterations"] = 10
  cfg["value_solver_iterations"] = p["n_critic_updates"]
  cfg["gae_lambda"] = p["gae_lambda"]
  cfg["normalize_advantage"] = p["normalize_advantage"]
  cfg["kl_threshold"] = p["target_kl"]
  cfg["value_learning_rate"] = p["learning_rate"]
  cfg["value_clip"] = False
  cfg["clip_actions"] = False

  if variant == "trpo":
    agent_class = SKRL_TRPO_WITH_COLLECT
  elif variant == "gentrpo":
    agent_class = GEN_TRPO_SKRL

  log_dir = f".logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
  os.makedirs(log_dir, exist_ok=True)

  args = {
    "models": models,
    "memory": memory,
    "cfg": cfg,
    "observation_space": env.observation_space,
    "action_space": env.action_space,
    "device": device,
    "variant": variant,
    "log_dir": log_dir,
    "env_id": env_id,
  }
  if variant == "gentrpo":
    args.update(
      {
        "sampling_coef": p.get("sampling_coef", 0.5),
        "buffer_capacity": p.get("buffer_capacity", p["n_steps"]),
        "normalize_advantage": p["normalize_advantage"],
      }
    )

  return agent_class(**args)


def run_ablation_skrl(single_env=None, single_variant=None, device="cuda" if th.cuda.is_available() else "cpu", noise_preset="none"):
  if single_env:
    envs = [single_env]
  else:
    envs = ["Humanoid-v5", "HumanoidStandup-v5"]
  if single_variant:
    variants = [single_variant]
  else:
    variants = ["trpo", "gentrpo"]

  runs = 5

  for env_id in envs:
    for variant in variants:
      if variant == "trpo":
        with open("hyperparameters/trpo.yaml", "r") as f:
          params = yaml.safe_load(f)
      else:
        with open("hyperparameters/gentrpo.yaml", "r") as f:
          params = yaml.safe_load(f)
      if env_id not in params:
        print(f"Skipping invalid combination: {env_id} {variant}")
        continue
      p = parse_params(params[env_id])
      p["n_timesteps"] = TIMESTEPS
      if variant != "trpo" and device.startswith("cuda"):
        p["buffer_capacity"] = BUFFER_CAPACITY
      n_envs = p["n_envs"]

      def make_env():
        return gym.make(env_id)

      env = SyncVectorEnv([make_env for _ in range(n_envs)])

      raw_path = f"results/{env_id}_{variant}_raw.yaml"

      env = wrap_env(env)
      agent = setup_agent_skrl(variant, p, env, env_id, device)
      trainer = SequentialTrainer(
        cfg={
          "timesteps": p["n_timesteps"],
          "headless": True,
          "log_interval": 1,
        },
        env=env,
        agents=agent,
      )

      for run in range(runs):
        trainer.train()
        os.makedirs("models", exist_ok=True)
        # For skrl, saving might be different; assuming agent has save method
        try:
          agent.save(f"models/{env_id}_{variant}_{noise_preset}_run{run}")
        except AttributeError:
          print("skrl agent save not implemented")
        # No callback, so no raw save for skrl

      env.close()
