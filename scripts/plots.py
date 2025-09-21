# Script 2: Produce plots for the output data in standard paper plot style (smoothed learning curves with legends, grid, etc.)
import os

import matplotlib.pyplot as plt
import numpy as np
import yaml


def plot_results(log_dir=".logs/"):
  envs = ["Humanoid-v5", "HumanoidStandup-v5"]
  variants = ["trpo", "gentrpo", "grentrpo-ne"]
  for env_id in envs:
    plt.figure(figsize=(10, 6))
    plt.clf()
    for cfg in variants:
      path = os.path.join(log_dir, f"rewards_{env_id}_{cfg}.yaml")
      if os.path.exists(path):
        with open(path, "r") as f:
          data = yaml.safe_load(f)
        rewards = data.get("rewards", [])
        if rewards:
          timesteps = np.arange(1, len(rewards) + 1)
          window_size = 100
          if len(rewards) >= window_size:
            smoothed = np.convolve(rewards, np.ones(window_size) / window_size, mode="valid")
            smoothed_timesteps = timesteps[window_size // 2 : -(window_size // 2) + 1]
            plt.plot(smoothed_timesteps, smoothed, label=cfg.upper().replace("-", " "))
          else:
            plt.plot(timesteps, rewards, label=cfg.upper().replace("-", " "))
    plt.title(f"Training Progress per Variant - {env_id}", fontsize=14)
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"graph_{env_id}.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
  plot_results()
