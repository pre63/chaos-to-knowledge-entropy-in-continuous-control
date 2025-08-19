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
  def __init__(self, env, config, env_id, log_dir=".omega", plot_interval=100):
    super().__init__(env)
    self.config = config
    self.env_id = env_id
    self.log_dir = log_dir
    self.plot_interval = plot_interval
    self.timesteps = 0
    self.rewards = []
    os.makedirs(self.log_dir, exist_ok=True)

  def step(self, action):
    obs, reward, terminated, truncated, info = self.env.step(action)
    self.timesteps += 1
    # Convert reward to native Python float to avoid NumPy serialization issues
    self.rewards.append(float(reward))

    if self.timesteps % self.plot_interval == 0:
      self._save_rewards()
      self._plot_progress()

    return obs, reward, terminated, truncated, info

  def reset(self, **kwargs):
    self.timesteps = 0
    return self.env.reset(**kwargs)

  def _save_rewards(self):
    """Save current rewards to a dedicated YAML file for this configuration."""
    config_name = self.config.upper().replace("_", " ")
    path = os.path.join(self.log_dir, f"rewards_{self.env_id}_{self.config}.yaml")
    # Ensure rewards are native Python types
    with open(path, "w") as f:
      yaml.safe_dump({"rewards": self.rewards}, f, default_flow_style=False)

  def _plot_progress(self):
    """Plot progress for all configurations by reading their reward YAML files."""
    plt.figure(figsize=(10, 6))
    plt.clf()

    configs = ["trpo_no_noise", "trpo_with_noise", "trpor_no_noise", "trpor_with_noise"]
    for cfg in configs:
      path = os.path.join(self.log_dir, f"rewards_{self.env_id}_{cfg}.yaml")
      if os.path.exists(path):
        with open(path, "r") as f:
          data = yaml.safe_load(f)
        rewards = data.get("rewards", [])
        if rewards:
          timesteps = range(1, len(rewards) + 1)
          if len(rewards) >= 100:
            smoothed = np.convolve(rewards, np.ones(100) / 100, mode="valid")
            smoothed_timesteps = range(50, len(smoothed) + 50)
            plt.plot(smoothed_timesteps, smoothed, label=cfg.upper().replace("_", " "))
          else:
            plt.plot(timesteps, rewards, label=cfg.upper().replace("_", " "))

    plt.title(f"Training Progress per Configuration - {self.env_id}")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(self.log_dir, f"graph_{self.env_id}.png"))
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
        # Convert mean reward to native Python float
        self.rewards.append(float(np.mean(rewards)))

      entropies = []
      for rollout_data in self.model.rollout_buffer.get(batch_size=None):
        observations = rollout_data.observations
        distribution = self.model.policy.get_distribution(observations)
        entropy_mean = distribution.entropy().mean().item()
        entropies.append(entropy_mean)
      if entropies:
        # Convert mean entropy to native Python float
        self.entropies.append(float(np.mean(entropies)))


def create_param_samplers(n_timesteps):
  """Factory function that creates and returns two sampler functions for TRPO and TRPOR hyperparameters."""
  defaults = {
    "n_critic_updates": 20,
    "cg_max_steps": 20,
    "net_arch": dict(pi=[256, 256], vf=[256, 256]),
    "activation_fn": nn.Tanh,
    "ortho_init": False,
    "n_timesteps": n_timesteps,
    "n_envs": 4,
  }

  trpor_defaults = {"epsilon": 0.5}
  trpo_defaults = {
    "cg_damping": 0.1,
    "line_search_shrinking_factor": 0.8,
    "log_std_init": -2.0,
    "sde_sample_freq": -1,
  }

  def sample_trpor_params(trial: optuna.Trial) -> Dict[str, Any]:
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.8, 0.85, 0.9, 0.95, 0.99])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024])
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.001, step=0.0001)

    if batch_size > n_steps:
      batch_size = n_steps

    params = {
      "policy": "MlpPolicy",
      "n_timesteps": defaults["n_timesteps"],
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
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.8, 0.85, 0.9, 0.95, 0.99])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    target_kl = trial.suggest_categorical("target_kl", [0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512, 1024])

    if batch_size > n_steps:
      batch_size = n_steps

    params = {
      "policy": "MlpPolicy",
      "n_timesteps": defaults["n_timesteps"],
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


def objective(trial, config, env_id, n_timesteps, device, log_dir):
  entropy_level = -0.3 if "with_noise" in config else 0.0

  env = gym.make(env_id)
  if entropy_level != 0:
    noise_configs = [
      {"component": "reward", "type": "uniform", "entropy_level": entropy_level},
      {"component": "action", "type": "uniform", "entropy_level": entropy_level},
    ]
    env = EntropyInjectionWrapper(env, noise_configs=noise_configs)
  env = AutoPlotProgressWrapper(env, config, env_id, log_dir=log_dir)

  sample_trpor_params, sample_trpo_params = create_param_samplers(n_timesteps)

  if "trpor" in config:
    params = sample_trpor_params(trial)
    model_class = TRPOR
  else:
    params = sample_trpo_params(trial)
    model_class = TRPO

  model = model_class(env=env, device=device, **params)

  callback = TrainingDataCallback()
  model.learn(total_timesteps=n_timesteps, callback=callback)

  max_reward = max(callback.rewards) if callback.rewards else float("-inf")

  env.close()
  return max_reward


def compare_max_rewards(config, env_id="HumanoidStandup-v5", n_timesteps=100_000, n_trials=100, device="cpu"):
  log_dir = ".omega/finetune_logs/"
  os.makedirs(log_dir, exist_ok=True)

  max_rewards = {}
  batch_size = 10  # Number of trials per batch

  # Create study for the specified config
  optuna_dir = ".omega/optuna_studies/"
  os.makedirs(optuna_dir, exist_ok=True)
  sampler = optuna.samplers.TPESampler()
  pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

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

  # Run optimization in batches
  trials_remaining = n_trials
  batch_number = 1
  study_has_one_trial = False

  while trials_remaining > 0:
    current_batch_size = min(batch_size, trials_remaining)
    print(f"\nRunning batch {batch_number} for {config} ({current_batch_size} trials for {n_timesteps} timesteps)...")

    if study.trials and study_has_one_trial:
      print(f"\nBest Trial Stats for {config.upper().replace('_', ' ')} (Batch {batch_number}):")
      bt = study.best_trial
      print(f"  Best Parameters: {bt.params}")
      print(f"  Best Max Reward: {bt.value:.2f}")
    else:
      print(f"{config.upper().replace('_', ' ')}: No trials yet")

    study.optimize(
      lambda trial: objective(trial, config, env_id, n_timesteps, device, log_dir),
      n_trials=current_batch_size,
    )
    study_has_one_trial = True
    print(f"Completed {current_batch_size} trials for {config}.")

    trials_remaining -= current_batch_size
    batch_number += 1

    # Save final results for this config
    results = {
      "best_params": study.best_params,
      "best_value": study.best_value,
    }
    config_name = config.upper().replace("_", " ")
    path = os.path.join(log_dir, f"{config_name}.yaml")
    with open(path, "w") as f:
      yaml.safe_dump(results, f, default_flow_style=False)

  # Print final summary for this config
  print(f"\nFinal Best Trial Stats for {config.upper().replace('_', ' ')}:")
  print(f"  Best Parameters: {study.best_params}")
  print(f"  Best Value: {study.best_value:.2f}")
  max_rewards[config] = study.best_value

  print(f"\nMax Reward for {config.upper().replace('_', ' ')}: {max_rewards[config]:.2f}")


def main():
  parser = argparse.ArgumentParser(description="Run training for specified model/config.")
  parser.add_argument(
    "--config",
    type=str,
    required=True,
    choices=["trpo_no_noise", "trpo_with_noise", "trpor_no_noise", "trpor_with_noise"],
    help="The model configuration to run.",
  )
  parser.add_argument("--env_id", type=str, default="HumanoidStandup-v5", help="Environment ID.")
  parser.add_argument("--n_timesteps", type=int, default=1_000_000, help="Number of timesteps for training.")
  parser.add_argument("--n_trials", type=int, default=100, help="Number of trials for optimization.")
  parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda).")

  args = parser.parse_args()

  compare_max_rewards(
    config=args.config,
    env_id=args.env_id,
    n_timesteps=args.n_timesteps,
    n_trials=args.n_trials,
    device=args.device,
  )


if __name__ == "__main__":
  main()
