import os
from datetime import datetime
from itertools import chain, combinations

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import uniform_filter1d  # For smoothing
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
        mean_reward = np.mean(rewards)
        self.rewards.append(mean_reward)
        print(f"Rollout end: Mean reward = {mean_reward}")

      for rollout_data in self.model.rollout_buffer.get(batch_size=None):
        observations = rollout_data.observations
        distribution = self.model.policy.get_distribution(observations)
        entropy_mean = distribution.entropy().mean().item()
        self.entropies.append(entropy_mean)


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
    return obs  # No obs noise needed

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


def generate_step_configs(components, noise_type, steps, min_level=-1.0, max_level=1.0):
  if steps < 1 or min_level >= max_level:
    raise ValueError("Invalid steps or range.")
  entropy_levels = np.linspace(min_level, max_level, steps)
  configs = []
  for level in entropy_levels:
    config_list = [{"component": comp, "type": noise_type, "entropy_level": float(level)} for comp in components]
    configs.append(config_list)
  return configs


def run_training(model_class, env, config, total_timesteps, num_runs, dry_run=False):
  run_rewards = []
  run_entropies = []
  for run in range(num_runs):
    callback = TrainingDataCallback(verbose=1)
    model_config = config.copy()
    model_config["env"] = env
    model = model_class(**model_config)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    rewards = callback.rewards if callback.rewards else [0]
    entropies = callback.entropies if callback.entropies else [0]
    run_rewards.append(rewards)
    run_entropies.append(entropies)
    print(f"Run {run+1}/{num_runs} completed.")
  max_reward_len = max(len(r) for r in run_rewards)
  max_entropy_len = max(len(e) for e in run_entropies)
  padded_rewards = [np.pad(r, (0, max_reward_len - len(r)), mode="edge") for r in run_rewards]
  padded_entropies = [np.pad(e, (0, max_entropy_len - len(e)), mode="edge") for e in run_entropies]
  return np.mean(padded_rewards, axis=0).tolist(), np.mean(padded_entropies, axis=0).tolist()


def smooth_data(training_data, window_size=50, pad_mode="edge"):
  smoothed_data = []

  # Ensure window_size is positive and odd for symmetry
  window_size = max(1, window_size)
  if window_size % 2 == 0:
    window_size += 1  # Make it odd for centered smoothing

  for data in training_data:
    # Convert to numpy arrays for processing
    rewards = np.array(data["rewards"])
    entropies = np.array(data["entropies"])

    # Skip smoothing for very short sequences
    if len(rewards) <= 1:
      smoothed_rewards = rewards
    else:
      # Pad symmetrically to avoid edge artifacts
      pad_size = window_size // 2
      padded_rewards = np.pad(rewards, (pad_size, pad_size), mode=pad_mode)
      smoothed_rewards = uniform_filter1d(padded_rewards, size=window_size, mode="nearest")[pad_size : pad_size + len(rewards)]

    if len(entropies) <= 1:
      smoothed_entropies = entropies
    else:
      pad_size = window_size // 2
      padded_entropies = np.pad(entropies, (pad_size, pad_size), mode=pad_mode)
      smoothed_entropies = uniform_filter1d(padded_entropies, size=window_size, mode="nearest")[pad_size : pad_size + len(entropies)]

    # Append smoothed data as a new dict
    smoothed_data.append({"label": data["label"], "rewards": smoothed_rewards.tolist(), "entropies": smoothed_entropies.tolist(), "model": data["model"]})

  return smoothed_data


def plot_results(smoothed_results, run_date, model_name, total_timesteps, num_runs, output_path):
  fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)
  colors = ["b", "g", "r", "c", "m", "y", "k"]
  markers = ["o", "s", "^", "v", "*", "+", "x"]

  # Create mappings of labels to markers and colors
  label_to_marker = {}
  label_to_color = {}
  marker_idx = 0
  color_idx = 0

  # Assign markers and colors independently to each unique label
  for result in smoothed_results:
    for data in result["smoothed_data"]:
      if data["label"] not in label_to_marker:
        # Assign marker and color with separate cycling
        label_to_marker[data["label"]] = markers[marker_idx % len(markers)]
        label_to_color[data["label"]] = colors[color_idx % len(colors)]
        marker_idx += 1  # Increment marker independently
        color_idx = (color_idx + 2) % len(colors)  # Offset color increment (e.g., skip 2)

  # Plot each dataset with its assigned marker and color
  for result in smoothed_results:
    for data in result["smoothed_data"]:
      label = data["label"]
      marker = label_to_marker[label]
      color = label_to_color[label]

      # Get rewards and x-axis, with safety check
      rewards = data["rewards"]
      if not rewards:  # Handle empty rewards
        print(f"Warning: Empty rewards for label '{label}'")
        continue
      x = np.arange(len(rewards))

      # Set markevery safely
      mark_every = max(1, len(x) // 10) if len(x) > 0 else 1

      # Plot rewards with markers
      ax1.plot(x, data["rewards"], label=label, color=color, marker=marker, linewidth=2, markersize=8, markevery=mark_every)

      # Handle entropies with padding if needed
      entropies = data["entropies"]
      if not entropies:  # Handle empty entropies
        print(f"Warning: Empty entropies for label '{label}'")
        continue
      entropies_len = len(entropies)
      padded_entropies = np.pad(entropies, (0, len(x) - entropies_len), mode="edge") if entropies_len < len(x) else entropies[: len(x)]
      # Plot entropies with markers
      ax2.plot(x, padded_entropies, label=label, color=color, marker=marker, linewidth=2, markersize=8, markevery=mark_every)

  # Configure axes
  ax1.set_title(f"Reward+Action Noise - Rewards (Avg of {num_runs} Runs)")
  ax1.set_ylabel("Mean Reward")
  ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
  ax1.grid(True)

  ax2.set_title(f"Reward+Action Noise - Entropy (Avg of {num_runs} Runs)")
  ax2.set_xlabel("Rollout")
  ax2.set_ylabel("Entropy")
  ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
  ax2.grid(True)

  plt.tight_layout()

  # Save the plot
  os.makedirs(f".noise/{run_date}", exist_ok=True)
  plot_path = f"{output_path}/{model_name}_{total_timesteps}_reward_action_{num_runs}_runs.png"
  plt.savefig(plot_path)
  plt.close()

  return plot_path


def save_partial_results(run_date, model_name, total_timesteps, num_runs, noise_type, training_data):
  os.makedirs(f"{output_path}/partial", exist_ok=True)
  temp_path = f"{output_path}/partial/{model_name}_{noise_type}_{total_timesteps}_temp.yml"
  with open(temp_path, "w") as file:
    yaml.dump({"noise_type": noise_type, "training_data": training_data}, file)
  return temp_path


def update_summary(run_date, summary_data):
  os.makedirs(f".noise/{run_date}", exist_ok=True)
  temp_path = f"{output_path}/summary_temp.yml"
  with open(temp_path, "w") as file:
    yaml.dump(summary_data, file)


def load_config_from_env(default_path=".noise/config.yml"):
  config_path = os.getenv("NOISE", default_path)
  config = {}
  try:
    with open(config_path, "r") as file:
      config = yaml.safe_load(file)
      print(f"Loaded config from {config_path}")
  except FileNotFoundError:
    print(f"Config file not found at {config_path}, using defaults")
  except Exception as e:
    print(f"Error loading config from {config_path}: {e}, using defaults")
  return config


if __name__ == "__main__":
  env_names = [
    "Humanoid-v5",
    "HalfCheetah-v5",
    "Hopper-v5",
    # "HumanoidStandup-v5",
    # "InvertedPendulum-v5",
    # "Pusher-v5",
    # "Reacher-v5",
    # "Swimmer-v5",
    # "Walker2d-v5",
    # "Humanoid-v5",
    # "RocketLander-v0",
  ]

  # Default values
  dry_run = False
  total_timesteps = 100000
  steps = 6
  min_level = -0.5
  max_level = 0.5
  num_runs = 5

  run_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

  models = [
    # GenTRPO,
    TRPO,
    PPO,
    TRPOR,
    TRPOER,
    # GenPPO,
  ]

  for env_name in env_names:
    config = load_config_from_env()

    # Override defaults with config
    models = config.get("models", models)
    dry_run = config.get("dry_run", dry_run)
    env_name = config.get("env_name", env_name)
    total_timesteps = config.get("total_timesteps", total_timesteps)
    steps = config.get("steps", steps)
    min_level = config.get("min_level", min_level)
    max_level = config.get("max_level", max_level)
    num_runs = config.get("num_runs", num_runs)

    # Adjusted config for reward+action only
    VALID_NOISE_CONFIGS = {"reward": ["uniform"], "action": ["uniform"]}
    ALL_COMPONENTS = ["reward", "action"]  # Only this combo will be tested

    all_configs_results = []
    summary = {
      "run_date": run_date,
      "total_timesteps": total_timesteps,
      "num_runs": num_runs,
      "models_tested": [],
      "best_reward_config": None,
      "best_entropy_config": None,
    }
    max_reward_improvement = float("-inf")
    max_entropy_reduction = float("-inf")
    baseline_dict = {}
    output_path = f".noise/{run_date}/{env_name}"
    os.makedirs(output_path, exist_ok=True)
    used_noise_configs = VALID_NOISE_CONFIGS if not dry_run else {"none": ["none"]}
    config_data = {
      "total_timesteps": total_timesteps,
      "steps": steps,
      "min_level": min_level,
      "max_level": max_level,
      "noise_configs": used_noise_configs,
      "component_combinations": [["reward", "action"]],  # Only this combo
      "num_runs": num_runs,
      "env_name": env_name,
      "dry_run": dry_run,
    }
    with open(f"{output_path}/config.yml", "w") as file:
      yaml.dump(config_data, file)

    for model_class in models:
      with open(f".hyperparameters/{model_class.__name__.lower()}.yml", "r") as file:
        model_hyperparameters = yaml.safe_load(file.read())
      all_results = []
      env_base = gym.make(env_name, render_mode=None)

      # Baseline run
      baseline_rewards, baseline_entropies = run_training(model_class, env_base, model_hyperparameters[env_name], total_timesteps, num_runs, dry_run)
      baseline_data = [{"label": "Baseline", "rewards": baseline_rewards, "entropies": baseline_entropies, "model": model_class.__name__}]
      all_results.append({"noise_type": "none", "training_data": baseline_data})
      save_partial_results(run_date, model_class.__name__, total_timesteps, num_runs, "none", baseline_data)
      baseline_dict[model_class.__name__] = {
        "final_reward": baseline_rewards[-1] if baseline_rewards else 0,
        "initial_entropy": baseline_entropies[0] if baseline_entropies else 0,
      }
      print(f"Baseline completed for {model_class.__name__}")

      if not dry_run:
        # Only test reward+action combo
        combo = tuple(ALL_COMPONENTS)  # ('reward', 'action')
        noise_type = "uniform"  # Fixed to uniform for simplicity
        configs = generate_step_configs(combo, noise_type, steps, min_level, max_level)
        training_data = []
        for config_list in configs:
          env = EntropyInjectionWrapper(env_base, noise_configs=config_list)
          avg_rewards, avg_entropies = run_training(model_class, env, model_hyperparameters[env_name], total_timesteps, num_runs, dry_run)
          label = f"reward+action_{noise_type} ({config_list[0]['entropy_level']:.2f})"
          training_data.append({"label": label, "rewards": avg_rewards, "entropies": avg_entropies, "model": model_class.__name__})
          print(f"Averaged {num_runs} runs for {label}")
        all_results.append({"noise_type": "reward_action", "training_data": training_data})
        save_partial_results(run_date, model_class.__name__, total_timesteps, num_runs, "reward_action", training_data)

      # Process and save results
      smoothed_results = [{"noise_type": r["noise_type"], "smoothed_data": smooth_data(r["training_data"])} for r in all_results]
      plot_path = plot_results(smoothed_results, run_date, model_class.__name__, total_timesteps, num_runs, output_path)
      with open(f"{output_path}/{model_class.__name__}_{total_timesteps}_reward_action_{num_runs}_runs.yml", "w") as file:
        yaml.dump(smoothed_results, file)
      all_configs_results.extend(smoothed_results)

      # Update summary
      summary["models_tested"].append(model_class.__name__)
      for result in smoothed_results:
        if result["noise_type"] == "none":
          continue
        for data in result["smoothed_data"]:
          model_name = data["model"]
          baseline_final_reward = baseline_dict.get(model_name, {}).get("final_reward", 0)
          baseline_initial_entropy = baseline_dict.get(model_name, {}).get("initial_entropy", 0)
          final_reward = data["rewards"][-1] if data["rewards"] else 0
          final_entropy = data["entropies"][-1] if data["entropies"] else 0
          reward_improvement = final_reward - baseline_final_reward
          entropy_reduction = baseline_initial_entropy - final_entropy

          if reward_improvement > max_reward_improvement:
            max_reward_improvement = reward_improvement
            summary["best_reward_config"] = {"config": data["label"], "model": model_name, "improvement": float(reward_improvement)}
          if entropy_reduction > max_entropy_reduction:
            max_entropy_reduction = entropy_reduction
            summary["best_entropy_config"] = {"config": data["label"], "model": model_name, "reduction": float(entropy_reduction)}

      update_summary(run_date, summary)
      print(f"Model {model_class.__name__} completed")

    # Finalize summary
    with open(f"{output_path}/summary.yml", "w") as file:
      yaml.dump(summary, file)

    if os.path.exists(f"{output_path}/summary_temp.yml"):
      os.remove(f"{output_path}/summary_temp.yml")

    print(f"Final summary saved: Best reward config = {summary['best_reward_config']}, Best entropy config = {summary['best_entropy_config']}")
