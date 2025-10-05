# scripts/ablation_sb3.py
import argparse
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import yaml
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3.gen_trpo import GenTRPO as GEN_TRPO_SB3
from sb3.trpo import TRPO as TRPO_SB3

th.backends.cudnn.benchmark = True
th.set_float32_matmul_precision("high")

TIMESTEPS = 100000
BUFFER_CAPACITY = 3072


class TrainingDataCallback(BaseCallback):
  def __init__(self, verbose=0):
    super(TrainingDataCallback, self).__init__(verbose)
    self.rewards = []
    self.entropies = []

  def _on_step(self) -> bool:
    return True

  def _on_rollout_end(self) -> None:
    if hasattr(self.model, "rollout_buffer"):
      rewards = self.model.rollout_buffer.rewards
      if rewards.size > 0:
        self.rewards.append(float(np.mean(rewards)))  # Convert to float

      for rollout_data in self.model.rollout_buffer.get(batch_size=None):
        observations = rollout_data.observations
        distribution = self.model.policy.get_distribution(observations)
        entropy_mean = distribution.entropy().mean().item()
        self.entropies.append(float(entropy_mean))  # Convert to float


def save_raw_run(raw_path, noise_key, run_idx, metrics, noise_configs, total_timesteps):
  raw_data = []
  if os.path.exists(raw_path):
    try:
      with open(raw_path, "r") as file:
        raw_data = yaml.safe_load(file) or []
    except Exception as e:
      print(f"Error loading {raw_path}: {e}")

  new_entry = {
    "noise_type": noise_key,
    "run_index": run_idx,
    "metrics": metrics,
    "noise_configs": noise_configs,
    "total_timesteps": total_timesteps,
    "completed": len(metrics) > 0,
    "timestamp": datetime.now().isoformat(),
  }
  raw_data.append(new_entry)

  try:
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    with open(raw_path, "w") as file:
      yaml.dump(raw_data, file, default_flow_style=False)
    print(f"Saved raw data to {raw_path} for {noise_key}, run {run_idx+1}")
  except Exception as e:
    print(f"Error saving raw data to {raw_path}: {e}")


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


def setup_agent_sb3(variant, p, env, env_id, device, noise_preset="none"):
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
    "normalize_advantage": p.get("normalize_advantage", False),
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

  if "line_search_shrinking_factor" in p:
    if "backtrack_coeff" in dir(TRPO_SB3):  # assume if supported
      kwargs["backtrack_coeff"] = p["line_search_shrinking_factor"]

  noise_configs = []
  if noise_preset != "none":
    entropy_level = abs(p.get("epsilon", 0.1))
    if noise_preset in ["both", "action"]:
      noise_configs.append({"component": "action", "type": "uniform", "entropy_level": entropy_level})
    if noise_preset in ["both", "reward"]:
      noise_configs.append({"component": "reward", "type": "uniform", "entropy_level": entropy_level})

  if noise_configs:
    kwargs["noise_configs"] = noise_configs

  if variant == "trpo":
    agent_class = TRPO_SB3
  else:
    agent_class = GEN_TRPO_SB3

  return agent_class, kwargs, noise_configs


def run_ablation_sb3(single_env=None, single_variant=None, device="cuda" if th.cuda.is_available() else "cpu", noise_preset="none"):
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
      if device == "cuda":
        p["buffer_capacity"] = BUFFER_CAPACITY

      def make_env():
        return gym.make(env_id, render_mode=None)

      env = DummyVecEnv([make_env])

      raw_path = f"results/{env_id}_{variant}_raw.yaml"

      agent_class, agent_kwargs, noise_configs = setup_agent_sb3(variant, p, env, env_id, device, noise_preset)
      for run in range(runs):
        callback = TrainingDataCallback()
        agent = agent_class(**agent_kwargs)
        agent.raw_path = raw_path
        agent.run_idx = run
        agent.noise_configs = noise_configs
        agent.noise_preset = noise_preset
        agent.learn(total_timesteps=p["n_timesteps"], log_interval=1, progress_bar=True, callback=callback)
        os.makedirs("models", exist_ok=True)
        agent.save(f"models/{env_id}_{variant}_{noise_preset}_run{run}.zip")

        # Save final yaml with eval
        save_raw_run(raw_path, noise_preset, run, agent.rollout_metrics, noise_configs, p["n_timesteps"])

      env.close()
