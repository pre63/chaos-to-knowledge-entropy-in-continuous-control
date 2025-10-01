import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from scipy import signal

# Set seaborn style for aesthetics
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica"]


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


def plot_individual_rewards_entropies(env_id, cfg, resampled_rewards, resampled_entropies, log_dir, ntimesteps, label, n_runs=1):
  if resampled_rewards and resampled_entropies:
    timesteps = np.linspace(0, ntimesteps, len(resampled_rewards))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Timesteps", fontsize=14)
    ax1.set_ylabel("Max Reward", color="#0072B2", fontsize=14)
    ax1.plot(timesteps, resampled_rewards, color="#0072B2", label="Rewards")
    ax1.tick_params(axis="y", labelcolor="#0072B2", labelsize=12)
    ax1.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean Entropy", color="#D55E00", fontsize=14)
    ax2.plot(timesteps, resampled_entropies, color="#D55E00", label="Entropies")
    ax2.tick_params(axis="y", labelcolor="#D55E00", labelsize=12)
    fig.tight_layout()
    plt.title(f"Rewards and Entropies - {env_id} {label} (N={n_runs})", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(log_dir, f"graph_{env_id}_{cfg}_rewards_entropies.png"), bbox_inches="tight", dpi=300)
    plt.close()

    # Dump debug data for individual plot
    debug_data = {"timesteps": timesteps.tolist(), "rewards": resampled_rewards, "entropies": resampled_entropies}
    debug_dir = ".debug"
    os.makedirs(debug_dir, exist_ok=True)
    debug_path = os.path.join(debug_dir, f"graph_{env_id}_{cfg}_rewards_entropies.yaml")
    with open(debug_path, "w") as debug_file:
      yaml.dump(debug_data, debug_file)


def plot_comparative_rewards(env_id, variant_data, log_dir, ntimesteps, labels, suffix=""):
  colorblind_palette = ["#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#009E73", "#56B4E9"]
  plt.figure(figsize=(14, 6))
  plt.clf()
  comparative_rewards_data = {}
  sorted_keys = sorted(variant_data.keys(), key=float) if suffix.endswith("_noise_mod") else list(variant_data.keys())
  global_min = np.inf
  global_max = -np.inf
  for key in sorted_keys:
    data = variant_data.get(key, {})
    resampled_rewards = data.get("resampled_rewards", [])
    if resampled_rewards:
      global_min = min(global_min, min(resampled_rewards))
      global_max = max(global_max, max(resampled_rewards))
  for i, key in enumerate(sorted_keys):
    color = colorblind_palette[i % len(colorblind_palette)]
    data = variant_data.get(key, {})
    resampled_rewards = data.get("resampled_rewards", [])
    if resampled_rewards:
      timesteps = np.linspace(0, ntimesteps, len(resampled_rewards))
      plt.plot(timesteps, resampled_rewards, color=color, label=f"{labels.get(key, key)} (N=1)")
      comparative_rewards_data[key] = {"timesteps": timesteps.tolist(), "rewards": resampled_rewards}

  if "_noise_mod" in suffix:
    parts = suffix.split("_")
    cfg = parts[1]  # assuming suffix = "_{cfg}_noise_mod"
    model_map = {"gentrpo_raw": "GenTRPO", "trpo_raw": "TRPO", "gentrpo": "GenTRPO", "trpo": "TRPO"}
    model_name = model_map.get(cfg, cfg.upper())
    title = f"{model_name} Noise Level Modulation (Rewards) - {env_id}"
    filename = f"graph_{env_id}_{model_name}_noise_mod_rewards.png"
  else:
    title = f"Training Progress per Model (Rewards){suffix} - {env_id} (N=1 per model)"
    filename = f"graph_{env_id}_models_rewards.png"

  plt.title(title, fontsize=16)
  plt.xlabel("Timesteps", fontsize=14)
  plt.ylabel("Max Reward", fontsize=14)
  plt.ylim(global_min, global_max) if global_min != np.inf else None
  plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)
  if len(sorted_keys) > 1:
    plt.legend(loc="best", fontsize=10)
  plt.grid(True, linestyle="--", alpha=0.7)
  plt.tight_layout()
  plt.savefig(os.path.join(log_dir, filename), bbox_inches="tight", dpi=300)
  plt.close()

  # Dump debug data for comparative rewards
  debug_dir = ".debug"
  debug_path = os.path.join(debug_dir, f"{filename[:-4]}.yaml")
  with open(debug_path, "w") as debug_file:
    yaml.dump(comparative_rewards_data, debug_file)


def plot_comparative_entropies(env_id, variant_data, log_dir, ntimesteps, labels, suffix=""):
  colorblind_palette = ["#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#009E73", "#56B4E9"]
  plt.figure(figsize=(14, 6))
  plt.clf()
  comparative_entropies_data = {}
  sorted_keys = sorted(variant_data.keys(), key=float) if suffix.endswith("_noise_mod") else list(variant_data.keys())
  global_min = np.inf
  global_max = -np.inf
  for key in sorted_keys:
    data = variant_data.get(key, {})
    resampled_entropies = data.get("resampled_entropies", [])
    if resampled_entropies:
      global_min = min(global_min, min(resampled_entropies))
      global_max = max(global_max, max(resampled_entropies))
  for i, key in enumerate(sorted_keys):
    color = colorblind_palette[i % len(colorblind_palette)]
    data = variant_data.get(key, {})
    resampled_entropies = data.get("resampled_entropies", [])
    if resampled_entropies:
      timesteps = np.linspace(0, ntimesteps, len(resampled_entropies))
      plt.plot(timesteps, resampled_entropies, color=color, label=f"{labels.get(key, key)} (N=1)")
      comparative_entropies_data[key] = {"timesteps": timesteps.tolist(), "entropies": resampled_entropies}

  if "_noise_mod" in suffix:
    parts = suffix.split("_")
    cfg = parts[1]  # assuming suffix = "_{cfg}_noise_mod"
    model_map = {"gentrpo_raw": "GenTRPO", "trpo_raw": "TRPO", "gentrpo": "GenTRPO", "trpo": "TRPO"}
    model_name = model_map.get(cfg, cfg.upper())
    title = f"{model_name} Noise Level Modulation (Entropies) - {env_id}"
    filename = f"graph_{env_id}_{model_name}_noise_mod_entropies.png"
  else:
    title = f"Training Progress per Model (Entropies){suffix} - {env_id} (N=1 per model)"
    filename = f"graph_{env_id}_models_entropies.png"

  plt.title(title, fontsize=16)
  plt.xlabel("Timesteps", fontsize=14)
  plt.ylabel("Mean Entropy", fontsize=14)
  plt.ylim(global_min, global_max) if global_min != np.inf else None
  plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
  plt.xticks(fontsize=12)
  plt.yticks(fontsize=12)
  if len(sorted_keys) > 1:
    plt.legend(loc="best", fontsize=10)
  plt.grid(True, linestyle="--", alpha=0.7)
  plt.tight_layout()
  plt.savefig(os.path.join(log_dir, filename), bbox_inches="tight", dpi=300)
  plt.close()

  # Dump debug data for comparative entropies
  debug_dir = ".debug"
  debug_path = os.path.join(debug_dir, f"{filename[:-4]}.yaml")
  with open(debug_path, "w") as debug_file:
    yaml.dump(comparative_entropies_data, debug_file)


def plot_noise_mod_rewards_grid(env_id, variant_data, log_dir, ntimesteps, labels, cfg):
  colorblind_palette = ["#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#009E73", "#56B4E9"]
  sorted_keys = sorted(variant_data.keys(), key=float)
  num_levels = len(sorted_keys)
  cols = int(np.ceil(np.sqrt(num_levels)))
  rows = int(np.ceil(num_levels / cols))
  fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
  axs = axs.flatten()
  comparative_rewards_data = {}
  global_min = np.inf
  global_max = -np.inf
  for key in sorted_keys:
    runs_data = variant_data[key]
    rewards_arrays = [d["resampled_rewards"] for d in runs_data if d["resampled_rewards"]]
    if rewards_arrays:
      rewards_arrays = np.array(rewards_arrays)
      means = np.mean(rewards_arrays, axis=0)
      global_min = min(global_min, np.min(means))
      global_max = max(global_max, np.max(means))
  for i, key in enumerate(sorted_keys):
    ax = axs[i]
    runs_data = variant_data[key]
    n_runs = len(runs_data)
    rewards_arrays = [d["resampled_rewards"] for d in runs_data if d["resampled_rewards"]]
    if not rewards_arrays:
      continue
    rewards_arrays = np.array(rewards_arrays)
    means = np.mean(rewards_arrays, axis=0)
    stds = np.std(rewards_arrays, axis=0)
    timesteps = np.linspace(0, ntimesteps, len(means))
    color = colorblind_palette[i % len(colorblind_palette)]
    ax.plot(timesteps, means, color=color)
    ax.fill_between(timesteps, means - stds, means + stds, color=color, alpha=0.2)
    ax.set_title(f"{labels[key]} (N={n_runs})", fontsize=14)
    ax.set_xlabel("Timesteps", fontsize=14)
    ax.set_ylabel("Mean Reward (±SD)", fontsize=14)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.set_ylim(global_min, global_max)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    comparative_rewards_data[key] = {"timesteps": timesteps.tolist(), "means": means.tolist(), "stds": stds.tolist()}
  # Hide unused axes
  for j in range(i + 1, len(axs)):
    axs[j].axis("off")
  model_map = {"gentrpo_raw": "GenTRPO", "trpo_raw": "TRPO", "gentrpo": "GenTRPO", "trpo": "TRPO"}
  model_name = model_map.get(cfg, cfg.upper())
  fig.suptitle(f"{model_name} Noise Level Modulation (Rewards) - {env_id}", fontsize=16)
  plt.subplots_adjust(wspace=0.3, hspace=0.3)
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  filename = f"graph_{env_id}_{model_name}_noise_mod_rewards_grid.png"
  plt.savefig(os.path.join(log_dir, filename), bbox_inches="tight", dpi=300)
  plt.close()
  # Dump debug data
  debug_dir = ".debug"
  debug_path = os.path.join(debug_dir, f"{filename[:-4]}.yaml")
  with open(debug_path, "w") as debug_file:
    yaml.dump(comparative_rewards_data, debug_file)


def plot_noise_mod_entropies_grid(env_id, variant_data, log_dir, ntimesteps, labels, cfg):
  colorblind_palette = ["#0072B2", "#D55E00", "#CC79A7", "#F0E442", "#009E73", "#56B4E9"]
  sorted_keys = sorted(variant_data.keys(), key=float)
  num_levels = len(sorted_keys)
  cols = int(np.ceil(np.sqrt(num_levels)))
  rows = int(np.ceil(num_levels / cols))
  fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
  axs = axs.flatten()
  comparative_entropies_data = {}
  global_min = np.inf
  global_max = -np.inf
  for key in sorted_keys:
    runs_data = variant_data[key]
    entropies_arrays = [d["resampled_entropies"] for d in runs_data if d["resampled_entropies"]]
    if entropies_arrays:
      entropies_arrays = np.array(entropies_arrays)
      means = np.mean(entropies_arrays, axis=0)
      global_min = min(global_min, np.min(means))
      global_max = max(global_max, np.max(means))
  for i, key in enumerate(sorted_keys):
    ax = axs[i]
    runs_data = variant_data[key]
    n_runs = len(runs_data)
    entropies_arrays = [d["resampled_entropies"] for d in runs_data if d["resampled_entropies"]]
    if not entropies_arrays:
      continue
    entropies_arrays = np.array(entropies_arrays)
    means = np.mean(entropies_arrays, axis=0)
    stds = np.std(entropies_arrays, axis=0)
    timesteps = np.linspace(0, ntimesteps, len(means))
    color = colorblind_palette[i % len(colorblind_palette)]
    ax.plot(timesteps, means, color=color)
    ax.fill_between(timesteps, means - stds, means + stds, color=color, alpha=0.2)
    ax.set_title(f"{labels[key]} (N={n_runs})", fontsize=14)
    ax.set_xlabel("Timesteps", fontsize=14)
    ax.set_ylabel("Mean Entropy (±SD)", fontsize=14)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.set_ylim(global_min, global_max)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    comparative_entropies_data[key] = {"timesteps": timesteps.tolist(), "means": means.tolist(), "stds": stds.tolist()}
  # Hide unused axes
  for j in range(i + 1, len(axs)):
    axs[j].axis("off")
  model_map = {"gentrpo_raw": "GenTRPO", "trpo_raw": "TRPO", "gentrpo": "GenTRPO", "trpo": "TRPO"}
  model_name = model_map.get(cfg, cfg.upper())
  fig.suptitle(f"{model_name} Noise Level Modulation (Entropies) - {env_id}", fontsize=16)
  plt.subplots_adjust(wspace=0.3, hspace=0.3)
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  filename = f"graph_{env_id}_{model_name}_noise_mod_entropies_grid.png"
  plt.savefig(os.path.join(log_dir, filename), bbox_inches="tight", dpi=300)
  plt.close()
  # Dump debug data
  debug_dir = ".debug"
  debug_path = os.path.join(debug_dir, f"{filename[:-4]}.yaml")
  with open(debug_path, "w") as debug_file:
    yaml.dump(comparative_entropies_data, debug_file)


def plot_grid_model(cfg, all_variant_data, log_dir, ntimesteps, envs, variant_labels):
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
      ax1.plot(timesteps, resampled_rewards, color="#0072B2", label="Reward")
      ax1.set_xlabel("Timestep", fontsize=14)
      ax1.set_ylabel("Max Reward", color="#0072B2", fontsize=14)
      ax1.tick_params(axis="y", labelcolor="#0072B2", labelsize=12)
      ax1.set_title(f"{env_id} (N=1)", fontsize=14)
      ax1.set_ylim(global_reward_min, global_reward_max)
      ax1.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
      if resampled_entropies:
        ax2 = ax1.twinx()
        ax2.plot(timesteps, resampled_entropies, color="#D55E00", label="Entropy")
        ax2.set_ylabel("Mean Entropy", color="#D55E00", fontsize=14)
        ax2.tick_params(axis="y", labelcolor="#D55E00", labelsize=12)
        ax2.set_ylim(global_entropy_min, global_entropy_max)
      grid_model_data[env_id] = {"timesteps": timesteps.tolist(), "rewards": resampled_rewards, "entropies": resampled_entropies}
  fig.suptitle(f"Plots for {variant_labels[cfg]} across Environments", fontsize=16)
  plt.tight_layout()
  plt.savefig(os.path.join(log_dir, f"grid_model_{cfg}.png"))
  plt.close()

  # Dump debug data for grid model
  debug_dir = ".debug"
  debug_path = os.path.join(debug_dir, f"grid_model_{cfg}.yaml")
  with open(debug_path, "w") as debug_file:
    yaml.dump(grid_model_data, debug_file)


def plot_grid_env(env_id, all_variant_data, log_dir, ntimesteps, variants, variant_labels):
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
      ax1.plot(timesteps, resampled_rewards, color="#0072B2", label="Reward")
      ax1.set_xlabel("Timestep", fontsize=14)
      ax1.set_ylabel("Max Reward", color="#0072B2", fontsize=14)
      ax1.tick_params(axis="y", labelcolor="#0072B2", labelsize=12)
      ax1.set_title(f"{variant_labels[cfg]} (N=1)", fontsize=14)
      ax1.set_ylim(global_reward_min, global_reward_max)
      ax1.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
      if resampled_entropies:
        ax2 = ax1.twinx()
        ax2.plot(timesteps, resampled_entropies, color="#D55E00", label="Entropy")
        ax2.set_ylabel("Mean Entropy", color="#D55E00", fontsize=14)
        ax2.tick_params(axis="y", labelcolor="#D55E00", labelsize=12)
        ax2.set_ylim(global_entropy_min, global_entropy_max)
      grid_env_data[cfg] = {"timesteps": timesteps.tolist(), "rewards": resampled_rewards, "entropies": resampled_entropies}
  fig.suptitle(f"Plots for {env_id} across Models", fontsize=16)
  plt.tight_layout()
  plt.savefig(os.path.join(log_dir, f"grid_env_{env_id}.png"))
  plt.close()

  # Dump debug data for grid env
  debug_dir = ".debug"
  debug_path = os.path.join(debug_dir, f"grid_env_{env_id}.yaml")
  with open(debug_path, "w") as debug_file:
    yaml.dump(grid_env_data, debug_file)


def plot_results(log_dir, ntimesteps):
  envs = ["Humanoid-v5", "HumanoidStandup-v5"]
  variants = ["trpo", "gentrpo", "gentrpo-ne"]
  variant_labels = {"trpo": "TRPO", "gentrpo": "GenTRPO (Noise=0)", "gentrpo-ne": "GenTRPO"}
  all_variant_data = {env: {} for env in envs}

  # Ensure .debug directory exists
  debug_dir = ".debug"
  os.makedirs(debug_dir, exist_ok=True)

  for env_id in envs:
    variant_data = {}
    for cfg in variants:
      path = os.path.join(log_dir, f"{env_id}_{cfg}.yaml")
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
        plot_individual_rewards_entropies(env_id, cfg, resampled_rewards, resampled_entropies, log_dir, ntimesteps, variant_labels[cfg], n_runs=1)

    all_variant_data[env_id] = variant_data

    # Comparative rewards plot for the environment
    plot_comparative_rewards(env_id, variant_data, log_dir, ntimesteps, variant_labels)

    # Plot combined entropies
    plot_comparative_entropies(env_id, variant_data, log_dir, ntimesteps, variant_labels)

  # Grid for each model (variant) with plots for each env
  for cfg in variants:
    plot_grid_model(cfg, all_variant_data, log_dir, ntimesteps, envs, variant_labels)

  # Grid for each env with plots for each model
  for env_id in envs:
    plot_grid_env(env_id, all_variant_data, log_dir, ntimesteps, variants, variant_labels)


def plot_noise_modulation(log_dir, env_id, cfg, ntimesteps):
  path = os.path.join(log_dir, f"{env_id}_{cfg}.yaml")
  if not os.path.exists(path):
    print(f"File not found: {path}")
    return

  with open(path, "r") as f:
    runs = yaml.safe_load(f)

  # Assume runs is a list of dicts
  variant_data = {}
  labels = {}
  for run in runs:
    if not run.get("completed", False):
      continue
    entropy_level = None
    for comp in run.get("config", []):
      if comp.get("component") == "reward":  # or 'action', assuming similar
        entropy_level = comp.get("entropy_level")
        break
    if entropy_level is None:
      continue

    key = str(entropy_level)
    if key not in variant_data:
      variant_data[key] = []
    if key not in labels:
      labels[key] = f"Noise Level {float(key):.2f}"

    rewards = run.get("rewards", [])
    entropies = run.get("entropies", [])

    resampled_rewards = resample_data(rewards, aggregator="mean") if rewards else []
    resampled_entropies = resample_data(entropies, aggregator="mean") if entropies else []

    variant_data[key].append({"resampled_rewards": resampled_rewards, "resampled_entropies": resampled_entropies})

  # Grid rewards plot showing impact of noise
  plot_noise_mod_rewards_grid(env_id, variant_data, log_dir, ntimesteps, labels, cfg)

  # Grid entropies plot
  plot_noise_mod_entropies_grid(env_id, variant_data, log_dir, ntimesteps, labels, cfg)


if __name__ == "__main__":

  ntimesteps = 100000

  results_dir = "results"
  plot_results(results_dir, ntimesteps)
  plot_noise_modulation(results_dir, "Humanoid-v5", "gentrpo_raw", ntimesteps)
  plot_noise_modulation(results_dir, "Humanoid-v5", "trpo_raw", ntimesteps)
  plot_noise_modulation(results_dir, "HumanoidStandup-v5", "gentrpo_raw", ntimesteps)
  plot_noise_modulation(results_dir, "HumanoidStandup-v5", "trpo_raw", ntimesteps)
