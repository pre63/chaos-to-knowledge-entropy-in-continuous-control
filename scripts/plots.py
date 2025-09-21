# Methodology:
# This script processes reward and entropy data from YAML files for different environments and variants.
# Data resampling:
# - If the original data length is less than or equal to ntimesteps, it is upsampled by repeating values to reach ntimesteps, distributing the repetitions evenly.
# - If longer, it is downsampled by taking max (for rewards) or mean (for entropies) in non-overlapping windows.
# - Then, further downsampled into 200 points using max/mean in windows for plotting.
# - Finally, a Savitzky-Golay filter is applied for slight smoothing to avoid step-like appearance while preserving peaks.

import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import signal


def resample_data(data, ntimesteps, aggregator="mean"):
  orig_len = len(data)
  if orig_len == 0:
    return []
  if orig_len <= ntimesteps:
    # Upsample by repeating to approximately ntimesteps
    repeat_factor = ntimesteps // orig_len
    remainder = ntimesteps % orig_len
    upsampled = []
    for i in range(orig_len):
      repeats = repeat_factor
      if i < remainder:
        repeats += 1
      for _ in range(repeats):
        upsampled.append(data[i])
    # Trim or pad if necessary to exactly ntimesteps
    if len(upsampled) > ntimesteps:
      upsampled = upsampled[:ntimesteps]
    while len(upsampled) < ntimesteps:
      upsampled.append(upsampled[-1])
  else:
    # Downsample directly
    bin_size = orig_len // ntimesteps
    remainder = orig_len % ntimesteps
    upsampled = []
    idx = 0
    for i in range(ntimesteps):
      extra = 1 if i < remainder else 0
      window = bin_size + extra
      if window == 0:
        continue
      seg = data[idx : idx + window]
      if aggregator == "max":
        upsampled.append(max(seg))
      else:
        upsampled.append(sum(seg) / len(seg))
      idx += window
  # Now downsample with non-overlapping windows
  window = ntimesteps // 200
  if window == 0:
    window = 1
  new_data = []
  for start in range(0, ntimesteps, window):
    end = min(start + window, ntimesteps)
    seg = upsampled[start:end]
    if not seg:
      break
    if aggregator == "max":
      new_data.append(max(seg))
    else:
      new_data.append(sum(seg) / len(seg))
  # Apply smoothing
  new_data_np = np.array(new_data)
  if len(new_data_np) > 5:
    new_data_np = signal.savgol_filter(new_data_np, window_length=5, polyorder=2)
  return new_data_np.tolist()


def plot_results(log_dir, ntimesteps):
  envs = ["Humanoid-v5", "HumanoidStandup-v5"]
  variants = ["trpo", "gentrpo", "gentrpo-ne"]
  variant_labels = {"trpo": "TRPO", "gentrpo": "GenTRPO", "gentrpo-ne": "GenTRPO w/ Noise"}
  for env_id in envs:
    # Plot combined rewards
    plt.figure(figsize=(10, 6))
    plt.clf()
    for cfg in variants:
      path = os.path.join(log_dir, f"rewards_{env_id}_{cfg}.yaml")
      if os.path.exists(path):
        with open(path, "r") as f:
          data = yaml.safe_load(f)
        rewards = data.get("rewards", [])
        if rewards:
          resampled = resample_data(rewards, ntimesteps, aggregator="max")
          if len(resampled) > 0:
            timesteps = np.linspace(1, ntimesteps, len(resampled))
            plt.plot(timesteps, resampled, label=variant_labels[cfg])
    plt.title(f"Training Progress per Variant (Rewards) - {env_id}", fontsize=14)
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"graph_rewards_{env_id}.png"), bbox_inches="tight", dpi=300)
    plt.close()

    # Plot combined entropies
    plt.figure(figsize=(10, 6))
    plt.clf()
    for cfg in variants:
      path = os.path.join(log_dir, f"rewards_{env_id}_{cfg}.yaml")
      if os.path.exists(path):
        with open(path, "r") as f:
          data = yaml.safe_load(f)
        entropies = data.get("entropies", [])
        if entropies:
          resampled = resample_data(entropies, ntimesteps, aggregator="mean")
          if len(resampled) > 0:
            timesteps = np.linspace(1, ntimesteps, len(resampled))
            plt.plot(timesteps, resampled, label=variant_labels[cfg])
    plt.title(f"Training Progress per Variant (Entropies) - {env_id}", fontsize=14)
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Entropy", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"graph_entropies_{env_id}.png"), bbox_inches="tight", dpi=300)
    plt.close()

    # Plot individual for each variant
    for cfg in variants:
      path = os.path.join(log_dir, f"rewards_{env_id}_{cfg}.yaml")
      if os.path.exists(path):
        with open(path, "r") as f:
          data = yaml.safe_load(f)
        rewards = data.get("rewards", [])
        entropies = data.get("entropies", [])
        if rewards and entropies:
          resampled_rewards = resample_data(rewards, ntimesteps, aggregator="max")
          resampled_entropies = resample_data(entropies, ntimesteps, aggregator="mean")
          if len(resampled_rewards) > 0 and len(resampled_entropies) > 0:
            timesteps = np.linspace(1, ntimesteps, len(resampled_rewards))
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.set_xlabel("Timesteps")
            ax1.set_ylabel("Reward", color="tab:blue")
            ax1.plot(timesteps, resampled_rewards, color="tab:blue", label="Rewards")
            ax1.tick_params(axis="y", labelcolor="tab:blue")
            ax2 = ax1.twinx()
            ax2.set_ylabel("Entropy", color="tab:red")
            ax2.plot(timesteps, resampled_entropies, color="tab:red", label="Entropies")
            ax2.tick_params(axis="y", labelcolor="tab:red")
            fig.tight_layout()
            plt.title(f"Rewards and Entropies - {env_id} {variant_labels[cfg]}", fontsize=14)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.savefig(os.path.join(log_dir, f"graph_{env_id}_{cfg}_rewards_entropies.png"), bbox_inches="tight", dpi=300)
            plt.close()


if __name__ == "__main__":

  ntimesteps = 100000

  results_dir = "results"
  plot_results(results_dir, ntimesteps)
