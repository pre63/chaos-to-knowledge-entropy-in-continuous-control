import argparse
import functools
import os
from typing import Any, Dict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import optuna
import optuna.pruners
import optuna.samplers
import optuna.storages
import torch.nn as nn
import yaml
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from stable_baselines3.common.callbacks import BaseCallback

from Environments.Noise import EntropyInjectionWrapper
from Models.SB3 import TRPO
from Models.TRPOR.TRPOR import TRPOR


class AutoPlotProgressWrapper(gym.Wrapper):
  def __init__(self, env, config, config_to_study, log_dir=".omega", plot_interval=100):
    super().__init__(env)
    self.config = config
    self.config_to_study = config_to_study
    self.log_dir = log_dir
    self.plot_interval = plot_interval
    self.timesteps = 0
    self.rewards = []
    os.makedirs(self.log_dir, exist_ok=True)

  def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)
    self.timesteps += 1
    self.rewards.append(reward)

    if self.timesteps % self.plot_interval == 0:
      self._plot_progress()

    return obs, reward, terminated, truncated, info

  def reset(self, **kwargs):
    self.timesteps = 0
    return self.env.reset(**kwargs)

  def _plot_progress(self):
    plt.figure(figsize=(10, 6))
    plt.clf()

    for cfg in self.config_to_study.keys():
      st = self.config_to_study[cfg]
      rewards = st.user_attrs.get(f"best_rewards_{cfg}", [])
      if rewards:
        timesteps = range(1, len(rewards) + 1)
        if len(rewards) >= 100:
          smoothed = np.convolve(rewards, np.ones(100) / 100, mode="valid")
          smoothed_timesteps = range(50, len(smoothed) + 50)
          plt.plot(smoothed_timesteps, smoothed, label=cfg.upper().replace("_", " "))
        else:
          plt.plot(timesteps, rewards, label=cfg.upper().replace("_", " "))

    if self.rewards:
      timesteps = range(1, len(self.rewards) + 1)
      if len(self.rewards) >= 100:
        smoothed = np.convolve(self.rewards, np.ones(100) / 100, mode="valid")
        smoothed_timesteps = range(50, len(smoothed) + 50)
        plt.plot(smoothed_timesteps, smoothed, label=f"{self.config.upper().replace('_', ' ')} (current)")
      else:
        plt.plot(timesteps, self.rewards, label=f"{self.config.upper().replace('_', ' ')} (current)")

    plt.title("Best Run Progress per Configuration")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(self.log_dir, "graph.png"))
    plt.close()


class TrainingDataCallback(BaseCallback):
  def __init__(self, verbose=0):
    super().__init__(verbose)
    self.rewards = []
    self.entropies = []

  def _on_step(self) -> bool:
    return True

  def _on_rollout_end(self) -> None:
    if hasattr(self.model, "rollout_buffer"):
      rewards = self.model.rollout_buffer.rewards
      if rewards.size > 0:
        self.rewards.append(float(np.mean(rewards)))

      entropies = []
      for rollout_data in self.model.rollout_buffer.get(batch_size=None):
        observations = rollout_data.observations
        distribution = self.model.policy.get_distribution(observations)
        entropy_mean = distribution.entropy().mean().item()
        entropies.append(entropy_mean)
      if entropies:
        self.entropies.append(float(np.mean(entropies)))


def create_param_samplers():
  """
    Factory function that creates and returns two sampler functions for TRPO and TRPOR hyperparameters.
    Both samplers share the same dictionary of default values for less impactful hyperparameters via closures.
    """
  # Shared defaults for less impactful hyperparameters
  defaults = {
    "n_critic_updates": 20,
    "cg_max_steps": 20,
    "net_arch": dict(pi=[256, 256], vf=[256, 256]),
    "activation_fn": nn.Tanh,
    "ortho_init": False,
    "n_envs": 4,
  }

  # TRPOR-specific defaults
  trpor_defaults = {
    "epsilon": 0.5,
  }

  # TRPO-specific defaults
  trpo_defaults = {
    "cg_damping": 0.1,
    "line_search_shrinking_factor": 0.8,
    "log_std_init": -2.0,
    "sde_sample_freq": -1,
  }

  def sample_trpor_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
        Sampler for TRPOR hyperparameters using Optuna.
        Uses shared defaults via closure.
        """
    # Sampling core hyperparameters (dynamic for those that matter)
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.8, 0.85, 0.9, 0.95, 0.99])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024])
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.001, step=0.0001)

    if batch_size > n_steps:
      batch_size = n_steps

    # Combine shared defaults, TRPOR-specific, and sampled params
    params = {
      "policy": "MlpPolicy",
      "n_envs": defaults["n_envs"],
      "epsilon": trpor_defaults["epsilon"],
      "ent_coef": ent_coef,
      "n_steps": n_steps,
      "batch_size": batch_size,
      "gamma": gamma,
      "cg_max_steps": defaults["cg_max_steps"],
      "n_critic_updates": defaults["n_critic_updates"],
      "target_kl": target_kl,
      "learning_rate": learning_rate,
      "gae_lambda": gae_lambda,
      "policy_kwargs": dict(
        net_arch=defaults["net_arch"],
        activation_fn=defaults["activation_fn"],
        ortho_init=defaults["ortho_init"],
      ),
    }
    return params

  def sample_trpo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
        Sampler for TRPO hyperparameters using Optuna.
        Uses shared defaults via closure.
        """
    # Sampling core hyperparameters (dynamic for those that matter)
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.8, 0.85, 0.9, 0.95, 0.99])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024])

    if batch_size > n_steps:
      batch_size = n_steps

    # Combine shared defaults, TRPO-specific, and sampled params
    params = {
      "policy": "MlpPolicy",
      "n_envs": defaults["n_envs"],
      "n_steps": n_steps,
      "batch_size": batch_size,
      "gamma": gamma,
      "cg_damping": trpo_defaults["cg_damping"],
      "cg_max_steps": defaults["cg_max_steps"],
      "line_search_shrinking_factor": trpo_defaults["line_search_shrinking_factor"],
      "n_critic_updates": defaults["n_critic_updates"],
      "target_kl": target_kl,
      "learning_rate": learning_rate,
      "gae_lambda": gae_lambda,
      "sde_sample_freq": trpo_defaults["sde_sample_freq"],
      "policy_kwargs": dict(
        log_std_init=trpo_defaults["log_std_init"],
        net_arch=defaults["net_arch"],
        activation_fn=defaults["activation_fn"],
        ortho_init=defaults["ortho_init"],
      ),
    }
    return params

  return sample_trpor_params, sample_trpo_params


def objective(trial, config, env_id, n_timesteps, device, config_to_study):
  study = config_to_study[config]
  entropy_level = -0.3 if "with_noise" in config else 0.0

  env = gym.make(env_id)
  if entropy_level != 0:
    noise_configs = [
      {"component": "reward", "type": "uniform", "entropy_level": entropy_level},
      {"component": "action", "type": "uniform", "entropy_level": entropy_level},
    ]
    env = EntropyInjectionWrapper(env, noise_configs=noise_configs)
  env = AutoPlotProgressWrapper(env, config, config_to_study)

  sample_trpor_params, sample_trpo_params = create_param_samplers()

  if "trpor" in config:
    params = sample_trpor_params(trial)
  else:
    params = sample_trpo_params(trial)

  if "trpor" in config:
    model_class = TRPOR
  else:
    model_class = TRPO

  model = model_class(env=env, device=device, **params)

  callback = TrainingDataCallback()
  model.learn(total_timesteps=n_timesteps, callback=callback)

  max_reward = max(callback.rewards) if callback.rewards else float("-inf")

  if study.user_attrs.get(f"best_value_{config}", float("-inf")) < max_reward:
    study.set_user_attr(f"best_value_{config}", max_reward)
    study.set_user_attr(f"best_rewards_{config}", env.rewards)

  env.close()
  return max_reward


def train_with_best_params(config, env_id, n_timesteps, device, log_dir, config_to_study):
  study = config_to_study[config]
  best_params = study.best_params
  entropy_level = -0.3 if "with_noise" in config else 0.0  # Fixed noise level

  env = gym.make(env_id)
  if entropy_level != 0:
    noise_configs = [
      {"component": "reward", "type": "uniform", "entropy_level": entropy_level},
      {"component": "action", "type": "uniform", "entropy_level": entropy_level},
    ]
    env = EntropyInjectionWrapper(env, noise_configs=noise_configs)
  env = AutoPlotProgressWrapper(env, config, config_to_study)

  if "trpor" in config:
    model_class = TRPOR
  else:
    model_class = TRPO

  model = model_class(env=env, device=device, **best_params)

  callback = TrainingDataCallback()
  model.learn(total_timesteps=n_timesteps, callback=callback)

  # Save results
  results = {"rewards": callback.rewards, "entropies": callback.entropies, "max_reward": max(callback.rewards) if callback.rewards else 0}

  config_name = get_name(config)
  path = os.path.join(log_dir, f"{model_class.__name__}_{env_id.split('-')[0]}_{config_name}.yaml")
  with open(path, "w") as f:
    yaml.dump(results, f)

  env.close()
  return results["max_reward"]


def compare_results(env_id="HumanoidStandup-v5", log_dir=".omega/finetune_logs/"):
  configs = ["trpo_no_noise", "trpo_with_noise", "trpor_no_noise", "trpor_with_noise"]
  max_rewards = {}

  for config in configs:
    config_name = get_name(config)
    model_class_name = "TRPOR" if "trpor" in config else "TRPO"
    path = os.path.join(log_dir, f"{model_class_name}_{env_id.split('-')[0]}_{config_name}.yaml")
    if os.path.exists(path):
      with open(path, "r") as f:
        results = yaml.safe_load(f)
      max_rewards[config] = results["max_reward"]
    else:
      print(f"Missing results file for {config}: {path}")

  if max_rewards:
    # Plot max rewards using Matplotlib
    models = list(max_rewards.keys())
    rewards = list(max_rewards.values())

    plt.figure(figsize=(10, 6))
    plt.bar(models, rewards)
    plt.title(f"Max Rewards Comparison - {env_id}")
    plt.ylabel("Max Reward")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "max_rewards_comparison.png"))
    plt.close()

    print("\nComparison of Max Rewards from Final Training:")
    for config, reward in max_rewards.items():
      print(f"{config.upper().replace('_', ' ')}: {reward:.2f}")


def get_name(cfg):
  return cfg.upper().replace("_", " ")


def setup_studies(env_id):
  configs = ["trpo_no_noise", "trpo_with_noise", "trpor_no_noise", "trpor_with_noise"]

  config_to_study = {}
  optuna_dir = ".omega/optuna_studies/"
  os.makedirs(optuna_dir, exist_ok=True)
  sampler = optuna.samplers.TPESampler()
  pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

  for config in configs:
    storage = JournalStorage(JournalFileBackend(f"{optuna_dir}/{config}_storage"))
    study_name = f"{config}_{env_id}_study"
    study = optuna.create_study(
      sampler=sampler,
      pruner=pruner,
      storage=storage,
      study_name=study_name,
      load_if_exists=True,
      direction="maximize",
    )
    config_to_study[config] = study

  return config_to_study


def main():
  parser = argparse.ArgumentParser(description="Run training for specified model/config.")
  parser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=["trpo_no_noise", "trpo_with_noise", "trpor_no_noise", "trpor_with_noise"],
    help="The model configuration to run.",
  )
  parser.add_argument(
    "--mode",
    type=str,
    default="optimize",
    choices=["optimize", "train_final", "compare"],
    help="Mode: 'optimize' for hyperparam tuning, 'train_final' for training with best params, 'compare' for comparing results.",
  )
  parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for optimization mode.")
  parser.add_argument("--env_id", type=str, default="HumanoidStandup-v5", help="Environment ID.")
  parser.add_argument("--n_timesteps", type=int, default=1_000_000, help="Number of timesteps for training.")
  parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda).")
  parser.add_argument("--log_dir", type=str, default=".omega/finetune_logs/", help="Log directory.")

  args = parser.parse_args()

  os.makedirs(args.log_dir, exist_ok=True)
  config_to_study = setup_studies(args.env_id)

  if args.mode == "optimize":
    print(f"\nRunning optimization for {args.config} with {args.n_trials} trials...")
    study = config_to_study[args.config]
    study.optimize(lambda trial: objective(trial, args.config, args.env_id, args.n_timesteps, args.device, config_to_study), n_trials=args.n_trials)
    print(f"Completed {args.n_trials} trials for {args.config}.")
    print(f"Best params for {args.config}: {study.best_params}")
    print(f"Best value for {args.config}: {study.best_value:.2f}")

  elif args.mode == "train_final":
    print(f"\nTraining final model for {args.config} with best params...")
    max_reward = train_with_best_params(args.config, args.env_id, args.n_timesteps, args.device, args.log_dir, config_to_study)
    print(f"Max reward from final training for {args.config}: {max_reward:.2f}")

  elif args.mode == "compare":
    print("\nRunning comparison across all configs...")
    compare_results(args.env_id, args.log_dir)


if __name__ == "__main__":
  main()
