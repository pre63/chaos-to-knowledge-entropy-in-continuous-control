import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import signal


def resample_data(data, aggregator="mean", num_points=200):
  orig_len = len(data)
  if orig_len == 0:
    return []
  bin_size = orig_len // num_points
  remainder = orig_len % num_points
  resampled = []
  idx = 0
  for i in range(num_points):
    extra = 1 if i < remainder else 0
    window = bin_size + extra
    if window == 0:
      break
    seg = data[idx : idx + window]
    if aggregator == "max":
      resampled.append(max(seg))
    else:
      resampled.append(sum(seg) / len(seg))
    idx += window
  new_data_np = np.array(resampled)
  return new_data_np.tolist()


def plot_results(log_dir, ntimesteps):
  envs = ["Humanoid-v5", "HumanoidStandup-v5"]
  variants = ["trpo", "gentrpo", "gentrpo-ne"]
  variant_labels = {"trpo": "TRPO", "gentrpo": "GenTRPO (Noise=0)", "gentrpo-ne": "GenTRPO (Noise=0.1)"}
  all_variant_data = {env: {} for env in envs}

  # Ensure .debug directory exists
  debug_dir = ".debug"
  os.makedirs(debug_dir, exist_ok=True)

  for env_id in envs:
    variant_data = {}
    for cfg in variants:
      path = os.path.join(log_dir, f"rewards_{env_id}_{cfg}.yaml")
      if os.path.exists(path):
        with open(path, "r") as f:
          data = yaml.safe_load(f)
        rewards = data.get("rewards", [])
        entropies = data.get("entropies", [])
        if rewards:
          resampled_rewards = resample_data(rewards, aggregator="max")
        else:
          resampled_rewards = []
        if entropies:
          resampled_entropies = resample_data(entropies, aggregator="mean")
        else:
          resampled_entropies = []
        variant_data[cfg] = {"resampled_rewards": resampled_rewards, "resampled_entropies": resampled_entropies}

        # Individual plot for rewards and entropies
        if resampled_rewards and resampled_entropies:
          timesteps = np.linspace(0, ntimesteps, len(resampled_rewards))
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

          # Dump debug data for individual plot
          debug_data = {"timesteps": timesteps.tolist(), "rewards": resampled_rewards, "entropies": resampled_entropies}
          debug_path = os.path.join(debug_dir, f"graph_{env_id}_{cfg}_rewards_entropies.yaml")
          with open(debug_path, "w") as debug_file:
            yaml.dump(debug_data, debug_file)

    all_variant_data[env_id] = variant_data

    # Comparative rewards plot for the environment
    plt.figure(figsize=(10, 6))
    plt.clf()
    comparative_rewards_data = {}
    for cfg in variants:
      resampled_rewards = variant_data.get(cfg, {}).get("resampled_rewards", [])
      if resampled_rewards:
        timesteps = np.linspace(0, ntimesteps, len(resampled_rewards))
        plt.plot(timesteps, resampled_rewards, label=variant_labels[cfg])
        comparative_rewards_data[cfg] = {"timesteps": timesteps.tolist(), "rewards": resampled_rewards}
    plt.title(f"Training Progress per Variant (Rewards) - {env_id}", fontsize=14)
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"graph_rewards_{env_id}.png"), bbox_inches="tight", dpi=300)
    plt.close()

    # Dump debug data for comparative rewards
    debug_path = os.path.join(debug_dir, f"graph_rewards_{env_id}.yaml")
    with open(debug_path, "w") as debug_file:
      yaml.dump(comparative_rewards_data, debug_file)

    # Plot combined entropies
    plt.figure(figsize=(10, 6))
    plt.clf()
    comparative_entropies_data = {}
    for cfg in variants:
      resampled_entropies = variant_data.get(cfg, {}).get("resampled_entropies", [])
      if resampled_entropies:
        timesteps = np.linspace(0, ntimesteps, len(resampled_entropies))
        plt.plot(timesteps, resampled_entropies, label=variant_labels[cfg])
        comparative_entropies_data[cfg] = {"timesteps": timesteps.tolist(), "entropies": resampled_entropies}
    plt.title(f"Training Progress per Variant (Entropies) - {env_id}", fontsize=14)
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Entropy", fontsize=12)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"graph_entropies_{env_id}.png"), bbox_inches="tight", dpi=300)
    plt.close()

    # Dump debug data for comparative entropies
    debug_path = os.path.join(debug_dir, f"graph_entropies_{env_id}.yaml")
    with open(debug_path, "w") as debug_file:
      yaml.dump(comparative_entropies_data, debug_file)

  # Grid for each model (variant) with plots for each env
  for cfg in variants:
    reward_values = []
    entropy_values = []
    for env_id in envs:
      data = all_variant_data[env_id].get(cfg, {})
      resampled_rewards = data.get("resampled_rewards", [])
      resampled_entropies = data.get("resampled_entropies", [])
      if resampled_rewards:
        reward_values.extend(resampled_rewards)
      if resampled_entropies:
        entropy_values.extend(resampled_entropies)
    global_reward_min = min(reward_values) if reward_values else 0
    global_reward_max = max(reward_values) if reward_values else 1
    global_entropy_min = min(entropy_values) if entropy_values else 0
    global_entropy_max = max(entropy_values) if entropy_values else 1

    num_envs = len(envs)
    fig, axs = plt.subplots(1, num_envs, figsize=(6 * num_envs, 5))
    grid_model_data = {}
    for i, env_id in enumerate(envs):
      data = all_variant_data[env_id].get(cfg, {})
      resampled_rewards = data.get("resampled_rewards", [])
      resampled_entropies = data.get("resampled_entropies", [])
      if resampled_rewards:
        timesteps = np.linspace(0, ntimesteps, len(resampled_rewards))
        ax1 = axs[i] if num_envs > 1 else axs
        ax1.plot(timesteps, resampled_rewards, "b-", label="Reward")
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Reward", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        ax1.set_title(env_id)
        ax1.set_ylim(global_reward_min, global_reward_max)
        if resampled_entropies:
          ax2 = ax1.twinx()
          ax2.plot(timesteps, resampled_entropies, "r-", label="Entropy")
          ax2.set_ylabel("Entropy", color="r")
          ax2.tick_params(axis="y", labelcolor="r")
          ax2.set_ylim(global_entropy_min, global_entropy_max)
        grid_model_data[env_id] = {"timesteps": timesteps.tolist(), "rewards": resampled_rewards, "entropies": resampled_entropies}
    fig.suptitle(f"Plots for {variant_labels[cfg]} across Environments")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"grid_model_{cfg}.png"))
    plt.close()

    # Dump debug data for grid model
    debug_path = os.path.join(debug_dir, f"grid_model_{cfg}.yaml")
    with open(debug_path, "w") as debug_file:
      yaml.dump(grid_model_data, debug_file)

  # Grid for each env with plots for each model
  for env_id in envs:
    reward_values = []
    entropy_values = []
    for cfg in variants:
      data = all_variant_data[env_id].get(cfg, {})
      resampled_rewards = data.get("resampled_rewards", [])
      resampled_entropies = data.get("resampled_entropies", [])
      if resampled_rewards:
        reward_values.extend(resampled_rewards)
      if resampled_entropies:
        entropy_values.extend(resampled_entropies)
    global_reward_min = min(reward_values) if reward_values else 0
    global_reward_max = max(reward_values) if reward_values else 1
    global_entropy_min = min(entropy_values) if entropy_values else 0
    global_entropy_max = max(entropy_values) if entropy_values else 1

    num_variants = len(variants)
    fig, axs = plt.subplots(1, num_variants, figsize=(6 * num_variants, 5))
    grid_env_data = {}
    for i, cfg in enumerate(variants):
      data = all_variant_data[env_id].get(cfg, {})
      resampled_rewards = data.get("resampled_rewards", [])
      resampled_entropies = data.get("resampled_entropies", [])
      if resampled_rewards:
        timesteps = np.linspace(0, ntimesteps, len(resampled_rewards))
        ax1 = axs[i] if num_variants > 1 else axs
        ax1.plot(timesteps, resampled_rewards, "b-", label="Reward")
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Reward", color="b")
        ax1.tick_params(axis="y", labelcolor="b")
        ax1.set_title(variant_labels[cfg])
        ax1.set_ylim(global_reward_min, global_reward_max)
        if resampled_entropies:
          ax2 = ax1.twinx()
          ax2.plot(timesteps, resampled_entropies, "r-", label="Entropy")
          ax2.set_ylabel("Entropy", color="r")
          ax2.tick_params(axis="y", labelcolor="r")
          ax2.set_ylim(global_entropy_min, global_entropy_max)
        grid_env_data[cfg] = {"timesteps": timesteps.tolist(), "rewards": resampled_rewards, "entropies": resampled_entropies}
    fig.suptitle(f"Plots for {env_id} across Models")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f"grid_env_{env_id}.png"))
    plt.close()

    # Dump debug data for grid env
    debug_path = os.path.join(debug_dir, f"grid_env_{env_id}.yaml")
    with open(debug_path, "w") as debug_file:
      yaml.dump(grid_env_data, debug_file)


if __name__ == "__main__":

  ntimesteps = 100000

  results_dir = "results"
  plot_results(results_dir, ntimesteps)
