import csv
import os

import numpy as np
import yaml
from scipy.stats import linregress


def compute_stats(rewards, entropies):
  if not rewards:
    return None, None, None, None, None
  max_reward = np.max(rewards)
  max_index = np.argmax(rewards)
  timestep_at_max = (max_index / len(rewards)) * 100000  # Normalized to a total of 100,000 timesteps for comparison
  mean_reward = np.mean(rewards)
  std_reward = np.std(rewards)
  mean_entropy = np.mean(entropies) if entropies else None
  return max_reward, mean_reward, std_reward, mean_entropy, timestep_at_max


if __name__ == "__main__":
  log_dir = "results"
  envs = ["Humanoid-v5", "HumanoidStandup-v5"]
  variants = ["trpo", "gentrpo", "gentrpo-ne"]
  variant_labels = {"trpo": "TRPO", "gentrpo": "GenTRPO", "gentrpo-ne": "GenTRPO w/ Noise"}  # Not used here, but keeping for consistency
  results = {}
  for env_id in envs:
    results[env_id] = {}
    for cfg in variants:
      path = os.path.join(log_dir, f"rewards_{env_id}_{cfg}.yaml")
      if os.path.exists(path):
        with open(path, "r") as f:
          data = yaml.safe_load(f)
        rewards = data.get("rewards", [])
        entropies = data.get("entropies", [])
        max_r, mean_r, std_r, mean_e, timestep_max = compute_stats(rewards, entropies)
        results[env_id][cfg] = {"max_reward": max_r, "mean_reward": mean_r, "std_reward": std_r, "mean_entropy": mean_e, "timestep_at_max": timestep_max}
      else:
        results[env_id][cfg] = {}

  # Generate CSV
  csv_path = os.path.join(log_dir, "tables.csv")
  with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Environment", "Variant", "Max Reward", "Mean Reward", "Std Reward", "Timestep at Max"])
    for env_id in envs:
      for cfg in variants:
        d = results[env_id].get(cfg, {})
        max_r = d.get("max_reward", "")
        mean_r = d.get("mean_reward", "")
        std_r = d.get("std_reward", "")
        time_max = d.get("timestep_at_max", "")
        writer.writerow([env_id, cfg, max_r, mean_r, std_r, time_max])
