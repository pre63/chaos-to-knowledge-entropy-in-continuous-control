import argparse
import os
from datetime import datetime

import gymnasium as gym
import numpy as np
import optuna
import torch as th
import torch.nn as nn
import yaml
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from sb3.gen_trpo import GenTRPO as GEN_TRPO_SB3

th.backends.cudnn.benchmark = True
th.set_float32_matmul_precision("high")

TIMESTEPS = 100000
BUFFER_CAPACITY = 3072
N_TRIALS = 1000  # Number of Optuna trials
OBJECTIVE_DIRECTION = "maximize"  # Maximize average reward


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


class PruningCallback(TrainingDataCallback):
  def __init__(self, trial, report_interval, verbose=0):
    super(PruningCallback, self).__init__(verbose)
    self.trial = trial
    self.report_interval = report_interval
    self.last_reported_steps = 0

  def _on_rollout_end(self) -> None:
    super()._on_rollout_end()
    current_steps = self.num_timesteps
    if current_steps - self.last_reported_steps >= self.report_interval:
      num_rewards = len(self.rewards)
      if num_rewards > 0:
        last_rewards = self.rewards[-max(1, num_rewards // 10):]
        intermediate_value = np.mean(last_rewards)
        step = current_steps // self.report_interval
        self.trial.report(intermediate_value, step)
        if self.trial.should_prune():
          raise optuna.TrialPruned()
      self.last_reported_steps = current_steps


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


def suggest_fd_hyperparams(trial):
  """
    Optuna sampling function for ForwardDynamicsModel hyperparameters.
    Suggests values for all tunable parameters.
    """
  hidden_dim = trial.suggest_categorical("hidden_dim", [32,64, 128, 256, 512])
  encoder_layers = trial.suggest_int("encoder_layers", 1, 5)
  forward_layers = trial.suggest_int("forward_layers", 2, 10)
  activation = trial.suggest_categorical("activation", ["ReLU", "LeakyReLU", "Tanh", "ELU"])
  lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

  return {"hidden_dim": hidden_dim, "encoder_layers": encoder_layers, "forward_layers": forward_layers, "activation": activation, "lr": lr}


def setup_agent_sb3(variant, p, env, env_id, device, noise_preset="action", fd_config=None):
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

  kwargs["buffer_capacity"] = p.get("buffer_capacity", BUFFER_CAPACITY)
  kwargs["sampling_coef"] = p.get("sampling_coef", 0.5)
  kwargs["dynamics_updates"] = p.get("dynamics_updates", 5)
  if fd_config is None:
    fd_config = {}
  kwargs["fd_config"] = fd_config

  if "line_search_shrinking_factor" in p:
    if hasattr(GEN_TRPO_SB3, "backtrack_coeff"):  # assume if supported
      kwargs["backtrack_coeff"] = p["line_search_shrinking_factor"]

  noise_configs = []
  if noise_preset != "none":
    noise_level = abs(p.get("noise_level", 0.1))
    if noise_preset == "action":
      noise_configs.append({"component": "action", "noise_type": "uniform", "noise_level": noise_level})

  if noise_configs:
    kwargs["noise_configs"] = noise_configs

  agent_class = GEN_TRPO_SB3

  return agent_class, kwargs, noise_configs


def objective(trial, env_id, p, device):
  fd_config = suggest_fd_hyperparams(trial)
  print(f"Trial {trial.number} - FD Config: {fd_config}")

  def make_env():
    return gym.make(env_id, render_mode=None)

  env = DummyVecEnv([make_env])

  agent_class, agent_kwargs, noise_configs = setup_agent_sb3("gentrpo", p, env, env_id, device, noise_preset="action", fd_config=fd_config)

  report_interval = 10000
  callback = PruningCallback(trial, report_interval)
  agent = agent_class(**agent_kwargs)
  agent.learn(total_timesteps=p["n_timesteps"], log_interval=100000, progress_bar=False, callback=callback)

  # Objective: mean of the last 10% of rewards (or all if few)
  num_rewards = len(callback.rewards)
  if num_rewards > 0:
    last_rewards = callback.rewards[-max(1, num_rewards // 10) :]
    score = np.mean(last_rewards)
  else:
    score = -np.inf  # Penalize if no rewards

  env.close()

  return score


def run_optuna_fd_study(env_id="Humanoid-v5", device="cpu"):
  with open("hyperparameters/gentrpo.yaml", "r") as f:
    params = yaml.safe_load(f)
  if env_id not in params:
    raise ValueError(f"No hyperparameters for {env_id} in gentrpo.yaml")
  p = parse_params(params[env_id])
  p["n_timesteps"] = TIMESTEPS

  os.makedirs(".optuna_fd", exist_ok=True)
  file_path = f".optuna_fd/storage"
  backend = JournalFileBackend(file_path)
  storage = JournalStorage(backend)
  pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3, min_early_stopping_rate=0)
  study = optuna.create_study(direction=OBJECTIVE_DIRECTION, storage=storage, study_name=f"{env_id}_gentrpo_fd", load_if_exists=True, pruner=pruner)

  study.optimize(lambda trial: objective(trial, env_id, p, device), n_trials=N_TRIALS)

  print("Best hyperparameters:", study.best_params)
  print("Best value:", study.best_value)

  # Save study visualizations
  optuna.visualization.plot_optimization_history(study).show()
  optuna.visualization.plot_param_importances(study).show()

  # Also save trials dataframe
  study.trials_dataframe().to_csv(f".optuna_fd/{env_id}_gentrpo_fd_optuna.csv", index=False)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Optuna study for FD model in GenTRPO")
  parser.add_argument("--env_id", type=str, default="Humanoid-v5", help="Environment ID")
  parser.add_argument("--device", type=str, default="cpu", help="Device to use")
  args = parser.parse_args()

  run_optuna_fd_study(env_id=args.env_id, device=args.device)