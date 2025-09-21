add a flag sb3 or skrl

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

from skrli.gen_trpo import GEN_TRPO as GEN_TRPO_SKRL 
from skrli.trpo import SKRL_TRPO_WITH_COLLECT

from sb3.gen_trpo import GEN_TRPO as GEN_TRPO_SB3
from sb3.trpo import TRPO as TRPO_SB3

th.backends.cudnn.benchmark = True
th.set_float32_matmul_precision("high") 

TIMESTEPS = 1000000
BUFFER_CAPACITY = 3072

params_dict = {
  "Humanoid-v5": {
    "trpo": {
      "batch_size": 512,
      "n_steps": 1024,
      "gamma": 0.98,
      "learning_rate": 0.0001001422334677667,
      "line_search_shrinking_factor": 0.7,
      "n_critic_updates": 25,
      "cg_max_steps": 25,
      "cg_damping": 0.5,
      "target_kl": 0.1,
      "gae_lambda": 0.99,
      "net_arch": "small",
      "log_std_init": -1.1830649974937577,
      "ortho_init": False,
      "activation_fn": "relu",
      "lr_schedule": "linear",
      "n_timesteps": TIMESTEPS,
      "n_envs": 2,
      "normalize_advantage": True,  # Assume default if not specified
    },
    "gentrpo": {
      "n_steps": 512,
      "gamma": 0.95,
      "learning_rate": 0.03648542602737527,
      "n_critic_updates": 20,
      "cg_max_steps": 5,
      "target_kl": 0.03,
      "gae_lambda": 0.95,
      "batch_size": 2048,
      "net_arch": "medium",
      "log_std_init": -1.0,
      "activation_fn": "tanh",
      "entropy_coef": -0.59,
      "buffer_capacity": BUFFER_CAPACITY,
      "ortho_init": False,
      "n_timesteps": TIMESTEPS,
      "n_envs": 6,
      "normalize_advantage": True,
    },
    "grentrpo-ne": {
      "n_steps": 512,
      "gamma": 0.95,
      "learning_rate": 0.03648542602737527,
      "n_critic_updates": 20,
      "cg_max_steps": 5,
      "target_kl": 0.03,
      "gae_lambda": 0.95,
      "batch_size": 2048,
      "net_arch": "medium",
      "activation_fn": "tanh",
      "entropy_coef": -0.59,
      "sampling_coef": -0.69,
      "buffer_capacity": 30000,
      "epsilon": -0.30,
      "orthogonal_init": 0,
      "n_timesteps": 600000,
      "n_envs": 6,
      "normalize_advantage": 1,
    },
  },
  "HumanoidStandup-v5": {
    "trpo": {
      "batch_size": 2048,
      "n_steps": 1024,
      "gamma": 0.995,
      "learning_rate": 0.0002661447879672383,
      "line_search_shrinking_factor": 0.8,
      "n_critic_updates": 5,
      "cg_max_steps": 20,
      "cg_damping": 0.5,
      "target_kl": 0.005,
      "gae_lambda": 0.8,
      "net_arch": "medium",
      "log_std_init": -1.3944767094748196,
      "ortho_init": True,
      "activation_fn": "tanh",
      "lr_schedule": "constant",
      "n_timesteps": TIMESTEPS,
      "n_envs": 8,
      "normalize_advantage": True,  # Assume default
    },
    "gentrpo": {
      "n_steps": 8,
      "gamma": 0.8,
      "learning_rate": 0.11959944479404305,
      "n_critic_updates": 30,
      "cg_max_steps": 5,
      "target_kl": 0.1,
      "gae_lambda": 1,
      "batch_size": 2048,
      "net_arch": "large",
      "log_std_init": -1.0,
      "activation_fn": "tanh",
      "entropy_coef": 0.24,
      "buffer_capacity": BUFFER_CAPACITY,
      "ortho_init": True,
      "n_timesteps": TIMESTEPS,
      "n_envs": 4,
      "normalize_advantage": False,
    },
    "grentrpo-ne": {
      "n_steps": "8",
      "gamma": "0.8",
      "learning_rate": "0.11959944479404305",
      "n_critic_updates": "30",
      "cg_max_steps": "5",
      "target_kl": "0.1",
      "gae_lambda": "1",
      "batch_size": "2048",
      "net_arch": "large",
      "activation_fn": "tanh",
      "entropy_coef": "0.24",
      "sampling_coef": "-0.5",
      "epsilon": "0.10",
      "orthogonal_init": "1",
      "n_timesteps": TIMESTEPS,
      "n_envs": "4",
      "normalize_advantage": "0",
    },
  },
}


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
      self.log_std_parameter = nn.Parameter(th.full((action_space.shape[0],), p["log_std_init"]).to(self.device))

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
  elif variant == "grentrpo-ne":
    agent_class = GEN_TRPO_SKRL
    cfg["entropy_coef"] = p["entropy_coef"]

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
  elif variant == "grentrpo-ne":
    args.update(
      {
        "entropy_coef": p["entropy_coef"],
        "noise_type": "uniform",
        "noise_level": abs(p["epsilon"]),
        "sampling_coef": p.get("sampling_coef", 0.5),
        "buffer_capacity": p.get("buffer_capacity", p["n_steps"]),
        "normalize_advantage": p["normalize_advantage"],
      }
    )

  return agent_class(**args)


def setup_agent_sb3(variant, p, env, env_id, device):
  net_arch = {
    "pi": [64, 64] if p["net_arch"] == "small" else [400, 300] if p["net_arch"] == "large" else [256, 256],
    "vf": [64, 64] if p["net_arch"] == "small" else [400, 300] if p["net_arch"] == "large" else [256, 256],
  }
  activation_fn_class = {"tanh": nn.Tanh, "relu": nn.ReLU}[p["activation_fn"]]

  ortho_init = p.get("ortho_init", p.get("orthogonal_init", False))

  policy_kwargs = {
    "net_arch": dict(pi=net_arch["pi"], vf=net_arch["vf"]),
    "activation_fn": activation_fn_class,
    "ortho_init": ortho_init,
    "log_std_init": p.get("log_std_init", 0.0),
  }

  log_dir = f".logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}/"

  kwargs = {
    "policy": "MlpPolicy",
    "env": env,
    "n_steps": p["n_steps"],
    "batch_size": p["batch_size"],
    "gamma": p["gamma"],
    "gae_lambda": p["gae_lambda"],
    "normalize_advantage": p["normalize_advantage"],
    "max_kl": p["target_kl"],
    "cg_max_iters": p["cg_max_steps"],
    "vf_iters": p["n_critic_updates"],
    "vf_stepsize": p["learning_rate"],
    "device": device,
    "tensorboard_log": log_dir,
    "verbose": 1,
    "policy_kwargs": policy_kwargs,
  }

  if "cg_damping" in p:
    kwargs["cg_damping"] = p["cg_damping"]

  if "entropy_coef" in p:
    kwargs["ent_coef"] = p["entropy_coef"]
  else:
    kwargs["ent_coef"] = 0.0

  if variant != "trpo":
    kwargs["buffer_size"] = p.get("buffer_capacity", BUFFER_CAPACITY)
    kwargs["sampling_coef"] = p.get("sampling_coef", 0.5)

  if variant == "grentrpo-ne":
    kwargs["noise_level"] = abs(p["epsilon"])
    kwargs["noise_type"] = "uniform"

  if "line_search_shrinking_factor" in p:
    if "backtrack_coeff" in dir(TRPO_SB3):  # assume if supported
      kwargs["backtrack_coeff"] = p["line_search_shrinking_factor"]

  if variant == "trpo":
    agent_class = TRPO_SB3
  else:
    agent_class = GEN_TRPO_SB3

  return agent_class(**kwargs)


def run_ablation(single_env=None, single_variant=None, device="cuda" if th.cuda.is_available() else "cpu", framework="skrl"):
  if single_env:
    envs = [single_env]
  else:
    envs = ["Humanoid-v5", "HumanoidStandup-v5"]
  if single_variant:
    variants = [single_variant]
  else:
    variants = ["trpo", "gentrpo", "grentrpo-ne"]

  for env_id in envs:
    for variant in variants:
      if env_id not in params_dict or variant not in params_dict[env_id]:
        print(f"Skipping invalid combination: {env_id} {variant}")
        continue
      p = parse_params(params_dict[env_id][variant])
      n_envs = p["n_envs"]

      def make_env():
        return gym.make(env_id)

      env = SyncVectorEnv([make_env for _ in range(n_envs)])

      agent = None
      if framework == "skrl":
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

        trainer.train()
      elif framework == "sb3":
        agent = setup_agent_sb3(variant, p, env, env_id, device)

        agent.learn(
          total_timesteps=p["n_timesteps"],
          log_interval=1,
          progress_bar=True,
        )

      env.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--env", type=str, default=None, help="Environment ID")
  parser.add_argument("--model", type=str, default=None, help="Model variant")
  parser.add_argument("--device", type=str, default="auto", help="Device to use (e.g., cuda:0, cpu)")
  parser.add_argument("--framework", type=str, default="skrl", choices=["sb3", "skrl"], help="Framework to use")
  args = parser.parse_args()
  if args.device == "auto":
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
  else:
    device = th.device(args.device)
  run_ablation(args.env, args.model, device, args.framework)