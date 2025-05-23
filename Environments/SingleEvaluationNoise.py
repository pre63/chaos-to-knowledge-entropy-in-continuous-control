import json
import os
from datetime import datetime
from itertools import chain, combinations

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import uniform_filter1d
from stable_baselines3.common.callbacks import BaseCallback

from Models.Gen.GenPPO import GenPPO
from Models.Gen.GenTRPO import GenTRPO
from Models.SB3 import PPO, TRPO
from Models.TRPOER.TRPOER import TRPOER
from Models.TRPOR.TRPOR import TRPOR


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
        self.rewards.append(float(np.mean(rewards)))

      for rollout_data in self.model.rollout_buffer.get(batch_size=None):
        observations = rollout_data.observations
        distribution = self.model.policy.get_distribution(observations)
        entropy_mean = distribution.entropy().mean().item()
        self.entropies.append(float(entropy_mean))


class EntropyInjectionWrapper(gym.Wrapper):
  def __init__(self, env, noise_configs=None):
    super().__init__(env)
    if not (isinstance(self.action_space, gym.spaces.Box) and isinstance(self.observation_space, gym.spaces.Box)):
      raise ValueError("This wrapper is designed for continuous action and observation spaces only.")
    self.noise_configs = noise_configs if noise_configs is not None else []
    self._validate_configs()
    self.base_std = 1.0
    self.base_range = 1.0
    self.base_scale = 1.0
    self.base_p = 0.5

  def _validate_configs(self):
    if not self.noise_configs:
      return
    for config in self.noise_configs:
      required_keys = {"component", "type", "entropy_level"}
      if not all(key in config for key in required_keys):
        raise ValueError("Each noise_config must include 'component', 'type', and 'entropy_level'.")
      component, noise_type, entropy_level = config["component"], config["type"], config["entropy_level"]
      if component not in ["obs", "reward", "action"]:
        raise ValueError("Component must be 'obs', 'reward', or 'action'.")
      if noise_type not in ["gaussian", "uniform", "laplace", "bernoulli"]:
        raise ValueError("Noise type must be 'gaussian', 'uniform', 'laplace', or 'bernoulli'.")
      if noise_type == "bernoulli" and component != "reward":
        raise ValueError("Bernoulli noise is only supported for rewards.")
      if not -1 <= entropy_level <= 1:
        raise ValueError("entropy_level must be between -1 and 1.")

  def _add_obs_noise(self, obs):
    return obs

  def _add_reward_noise(self, reward):
    for config in self.noise_configs:
      if config["component"] == "reward":
        noise_type = config["type"]
        entropy_level = abs(config["entropy_level"])
        if noise_type == "gaussian":
          std = entropy_level * self.base_std
          return reward + np.random.normal(0, std)
        elif noise_type == "uniform":
          range_val = entropy_level * self.base_range
          return reward + np.random.uniform(-range_val, range_val)
        elif noise_type == "laplace":
          scale = entropy_level * self.base_scale
          return reward + np.random.laplace(0, scale)
        elif noise_type == "bernoulli":
          p = entropy_level * self.base_p
          return 0 if np.random.uniform() < p else reward
    return reward

  def _add_action_noise(self, action):
    for config in self.noise_configs:
      if config["component"] == "action":
        noise_type = config["type"]
        entropy_level = abs(config["entropy_level"])
        if noise_type == "gaussian":
          std = entropy_level * self.base_std
          return action + np.random.normal(0, std, size=action.shape)
        elif noise_type == "uniform":
          range_val = entropy_level * self.base_range
          return action + np.random.uniform(-range_val, range_val, size=action.shape)
        elif noise_type == "laplace":
          scale = entropy_level * self.base_scale
          return action + np.random.laplace(0, scale, size=action.shape)
    return action

  def reset(self, **kwargs):
    obs, info = self.env.reset(**kwargs)
    return obs, info

  def step(self, action):
    noisy_action = self._add_action_noise(action)
    action_to_use = np.clip(noisy_action, self.action_space.low, self.action_space.high)
    obs, reward, terminated, truncated, info = self.env.step(action_to_use)
    reward = self._add_reward_noise(reward)
    return obs, reward, terminated, truncated, info


def load_raw_runs(raw_path, noise_key):
  runs = []
  if os.path.exists(raw_path):
    try:
      with open(raw_path, "r") as file:
        raw_data = yaml.safe_load(file) or []
        for entry in raw_data:
          if entry["noise_type"] == noise_key:
            runs.append(
              {
                "run_index": entry["run_index"],
                "rewards": entry["rewards"],
                "entropies": entry["entropies"],
                "completed": entry.get("completed", len(entry["rewards"]) > 0),
              }
            )
    except Exception as e:
      print(f"Error loading {raw_path}: {e}")
  return runs


def check_run_exists(runs, run_idx):
  return any(run["run_index"] == run_idx and run["completed"] for run in runs)


def save_raw_run(raw_path, noise_key, run_idx, rewards, entropies, config_list, total_timesteps):
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
    "rewards": rewards,
    "entropies": entropies,
    "config": config_list,
    "total_timesteps": total_timesteps,
    "completed": len(rewards) > 0,
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


def run_training(model_class, env, config, total_timesteps, num_runs, output_path, env_name, noise_key, model_name, config_list, dry_run=False):
  result_path = f"{output_path}/{env_name}/{model_name}_results.yml"
  raw_path = f"{output_path}/{env_name}/{model_name}_raw.yml"

  runs = load_raw_runs(raw_path, noise_key)
  run_rewards = [run["rewards"] for run in runs if run["completed"]]
  run_entropies = [run["entropies"] for run in runs if run["completed"]]

  runs_needed = num_runs - len(run_rewards)
  if runs_needed <= 0:
    print(f"All {num_runs} runs already completed for {model_name} on {env_name}, {noise_key}")
    max_reward_len = max(len(r) for r in run_rewards) if run_rewards else 1
    max_entropy_len = max(len(e) for e in run_entropies) if run_entropies else 1
    padded_rewards = [np.pad(r, (0, max_reward_len - len(r)), mode="edge") for r in run_rewards]
    padded_entropies = [np.pad(e, (0, max_entropy_len - len(e)), mode="edge") for e in run_entropies]
    avg_rewards = [float(x) for x in np.mean(padded_rewards, axis=0)] if padded_rewards else [0.0]
    avg_entropies = [float(x) for x in np.mean(padded_entropies, axis=0)] if padded_entropies else [0.0]
  else:
    print(f"Need {runs_needed} more runs for {model_name} on {env_name}, {noise_key}")
    for run_idx in range(len(run_rewards), num_runs):
      if check_run_exists(runs, run_idx):
        print(f"Run {run_idx+1} already exists for {model_name} on {env_name}, {noise_key}. Skipping.")
        continue

      print(f"Running {model_class.__name__} on {env.spec.id} - Run {run_idx + 1}/{num_runs} for {noise_key}")
      callback = TrainingDataCallback(verbose=1)
      model_config = config.copy()
      model_config["env"] = env
      try:
        model = model_class(**model_config)
        model.learn(total_timesteps=total_timesteps, callback=callback)
      except Exception as e:
        print(f"Error during training for {model_name} on {env_name}, {noise_key}, run {run_idx+1}: {e}")
        continue

      rewards = callback.rewards if callback.rewards else [0.0]
      entropies = callback.entropies if callback.entropies else [0.0]
      run_rewards.append(rewards)
      run_entropies.append(entropies)
      print(f"Run {run_idx+1}/{num_runs} completed for {noise_key}.")

      save_raw_run(raw_path, noise_key, run_idx, rewards, entropies, config_list, total_timesteps)

    max_reward_len = max(len(r) for r in run_rewards) if run_rewards else 1
    max_entropy_len = max(len(e) for e in run_entropies) if run_entropies else 1
    padded_rewards = [np.pad(r, (0, max_reward_len - len(r)), mode="edge") for r in run_rewards]
    padded_entropies = [np.pad(e, (0, max_entropy_len - len(e)), mode="edge") for e in run_entropies]
    avg_rewards = [float(x) for x in np.mean(padded_rewards, axis=0)] if padded_rewards else [0.0]
    avg_entropies = [float(x) for x in np.mean(padded_entropies, axis=0)] if padded_entropies else [0.0]

  # Generate label based on noise configuration
  if noise_key == "none":
    label = "Baseline"
  else:
    try:
      entropy_level = config_list[0]["entropy_level"] if config_list else 0.0
      label = f"reward+action_uniform ({entropy_level:+.1f})"
    except (IndexError, KeyError):
      label = f"reward+action_uniform (unknown)"

  training_data = [{"label": label, "rewards": avg_rewards, "entropies": avg_entropies, "model": model_name}]

  smoothed_data = smooth_data(training_data)
  new_result = {"noise_type": noise_key, "smoothed_data": smoothed_data}
  existing_results = []
  if os.path.exists(result_path):
    try:
      with open(result_path, "r") as file:
        existing_results = yaml.safe_load(file) or []
    except Exception as e:
      print(f"Error loading {result_path}: {e}")
  existing_results = [r for r in existing_results if r["noise_type"] != noise_key]
  existing_results.append(new_result)

  try:
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as file:
      yaml.dump(existing_results, file, default_flow_style=False)
    print(f"Smoothed results updated in {result_path} for {noise_key}")
  except Exception as e:
    print(f"Error saving results to {result_path}: {e}")

  return avg_rewards, avg_entropies


def smooth_data(training_data, window_size=50, pad_mode="edge"):
  smoothed_data = []
  window_size = max(1, window_size)
  if window_size % 2 == 0:
    window_size += 1

  for data in training_data:
    rewards = np.array(data["rewards"])
    entropies = np.array(data["entropies"])

    if len(rewards) <= 1:
      smoothed_rewards = rewards.tolist()
    else:
      pad_size = window_size // 2
      padded_rewards = np.pad(rewards, (pad_size, pad_size), mode=pad_mode)
      smoothed_rewards = uniform_filter1d(padded_rewards, size=window_size, mode="nearest")[pad_size : pad_size + len(rewards)].tolist()

    if len(entropies) <= 1:
      smoothed_entropies = entropies.tolist()
    else:
      pad_size = window_size // 2
      padded_entropies = np.pad(entropies, (pad_size, pad_size), mode=pad_mode)
      smoothed_entropies = uniform_filter1d(padded_entropies, size=window_size, mode="nearest")[pad_size : pad_size + len(entropies)].tolist()

    smoothed_data.append(
      {
        "label": data["label"],
        "rewards": [float(x) for x in smoothed_rewards],
        "entropies": [float(x) for x in smoothed_entropies],
        "model": data["model"],
      }
    )

  return smoothed_data


def plot_results(smoothed_results, model_name, total_timesteps, num_runs, output_path, env_name):
  fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)
  colors = ["b", "g", "r", "c", "m", "y", "k"]
  markers = ["o", "s", "^", "v", "*", "+", "x"]

  label_to_marker = {}
  label_to_color = {}
  marker_idx = 0
  color_idx = 0

  for result in smoothed_results:
    for data in result["smoothed_data"]:
      if data["label"] not in label_to_marker:
        label_to_marker[data["label"]] = markers[marker_idx % len(markers)]
        label_to_color[data["label"]] = colors[color_idx % len(colors)]
        marker_idx += 1
        color_idx = (color_idx + 2) % len(colors)

  for result in smoothed_results:
    for data in result["smoothed_data"]:
      label = data["label"]
      marker = label_to_marker[label]
      color = label_to_color[label]

      rewards = data["rewards"]
      if not rewards:
        print(f"Warning: Empty rewards for label '{label}'")
        continue
      x = np.arange(len(rewards))
      mark_every = max(1, len(x) // 10) if len(x) > 0 else 1
      ax1.plot(x, rewards, label=label, color=color, marker=marker, linewidth=2, markersize=8, markevery=mark_every)

      entropies = data["entropies"]  # Corrected to use entropies
      if not entropies:
        print(f"Warning: Empty entropies for label '{label}'")
        continue
      entropies_len = len(entropies)
      padded_entropies = np.pad(entropies, (0, len(x) - entropies_len), mode="edge") if entropies_len < len(x) else entropies[: len(x)]
      ax2.plot(x, padded_entropies, label=label, color=color, marker=marker, linewidth=2, markersize=8, markevery=mark_every)

  ax1.set_title(f"Rewards (Single Run) - {env_name}")
  ax1.set_ylabel("Mean Reward")
  ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
  ax1.grid(True)
  ax2.set_title(f"Entropy (Single Run) - {env_name}")
  ax2.set_xlabel("Rollout")
  ax2.set_ylabel("Entropy")
  ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
  ax2.grid(True)
  plt.tight_layout()
  plot_path = f"{output_path}/{env_name}/{model_name}_results.png"
  os.makedirs(os.path.dirname(plot_path), exist_ok=True)
  plt.savefig(plot_path)
  plt.close()
  return plot_path


def is_variant_complete(env_name, model_name, noise_type, total_timesteps, num_runs, output_path=".noise/extras"):
  raw_path = f"{output_path}/{env_name}/{model_name}_raw.yml"
  if not os.path.exists(raw_path):
    return False
  try:
    with open(raw_path, "r") as file:
      raw_data = yaml.safe_load(file) or []
      completed_runs = sum(1 for entry in raw_data if entry["noise_type"] == noise_type and entry.get("completed", False))
      return completed_runs >= num_runs
  except Exception as e:
    print(f"Error checking {raw_path}: {e}")
  return False


def generate_noise_config(env_name, model_class, entropy_level=None):
  if entropy_level is None:
    return env_name, model_class, [], "none"
  config_list = [
    {"component": "reward", "type": "uniform", "entropy_level": entropy_level},
    {"component": "action", "type": "uniform", "entropy_level": entropy_level},
  ]
  noise_key = f"reward_action_{entropy_level:+.1f}".replace("+", "").replace("-", "-")
  return env_name, model_class, config_list, noise_key


if __name__ == "__main__":
  env_model_configs = [
    generate_noise_config("HalfCheetah-v5", PPO, None),
    generate_noise_config("Hopper-v5", PPO, -0.1),
    generate_noise_config("Humanoid-v5", TRPO, None),
    generate_noise_config("HumanoidStandup-v5", TRPO, None),
    generate_noise_config("Swimmer-v5", TRPO, None),
    generate_noise_config("Swimmer-v5", TRPO, -0.3),
    generate_noise_config("Swimmer-v5", GenTRPO, 0.3),
    generate_noise_config("HumanoidStandup-v5", TRPOER, -0.5),
    generate_noise_config("HumanoidStandup-v5", TRPOR, 0.1),
    generate_noise_config("Swimmer-v5", GenTRPO, 0.3),
    generate_noise_config("Humanoid-v5", GenTRPO, -0.3),
    generate_noise_config("HalfCheetah-v5", GenPPO, 0.1),
  ]

  dry_run = False
  total_timesteps = 10000000
  num_runs = 1
  output_path = f".noise/extras_{total_timesteps}"

  for env_name, model_class, config_list, noise_key in env_model_configs:
    model_name = model_class.__name__
    os.makedirs(f"{output_path}/{env_name}", exist_ok=True)

    hyperparam_path = f".hyperparameters/{model_name.lower()}.yml"
    if not os.path.exists(hyperparam_path):
      print(f"No hyperparameter file at {hyperparam_path} for {model_name}. Skipping.")
      continue

    try:
      with open(hyperparam_path, "r") as file:
        model_hyperparameters = yaml.safe_load(file) or {}
    except Exception as e:
      print(f"Error loading {hyperparam_path}: {e}. Skipping.")
      continue

    if not model_hyperparameters.get(env_name):
      print(f"No hyperparameters for {model_name} on {env_name}. Skipping.")
      continue

    env_base = gym.make(env_name, render_mode=None)
    env = env_base if not config_list else EntropyInjectionWrapper(env_base, noise_configs=config_list)

    if not is_variant_complete(env_name, model_name, noise_key, total_timesteps, num_runs, output_path):
      print(f"Starting {noise_key} for {model_name} on {env_name}")
      rewards, entropies = run_training(
        model_class, env, model_hyperparameters[env_name], total_timesteps, num_runs, output_path, env_name, noise_key, model_name, config_list, dry_run
      )
      print(f"Completed {noise_key} for {model_name} on {env_name}")
    else:
      print(f"Run already completed for {model_name} on {env_name}, {noise_key}")

    result_path = f"{output_path}/{env_name}/{model_name}_results.yml"
    smoothed_results = []
    if os.path.exists(result_path):
      try:
        with open(result_path, "r") as file:
          smoothed_results = yaml.safe_load(file) or []
      except Exception as e:
        print(f"Error loading results from {result_path}: {e}")
    if smoothed_results:
      try:
        plot_path = plot_results(smoothed_results, model_name, total_timesteps, num_runs, output_path, env_name)
        print(f"Plot generated at {plot_path}")
      except Exception as e:
        print(f"Error generating plot for {model_name} on {env_name}: {e}")

    print(f"Model {model_name} completed for {env_name}")
