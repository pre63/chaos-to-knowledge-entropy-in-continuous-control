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


def load_aggregated_data(env_id, variant, results_dir, noise_type="none", aggregator_rewards="max", aggregator_entropies="mean"):
  path = os.path.join(results_dir, f"{env_id}_{variant}_raw.yaml")
  if not os.path.exists(path):
    print(f"File not found: {path}")
    return None

  with open(path, "r") as f:
    data = yaml.safe_load(f)

  runs = [entry for entry in data if entry.get("noise_type") == noise_type]
  if not runs:
    print(f"No runs for {env_id}_{variant} noise_type={noise_type}")
    return None

  # Resample each run and aggregate
  rewards_per_run = []
  for run in runs:
    rewards = run.get("reward_mean", [])
    if rewards:
      resampled = resample_data(rewards, aggregator=aggregator_rewards)
      rewards_per_run.append(np.array(resampled))

  if not rewards_per_run:
    return None

  # Assume all resampled have same length (200)
  rewards_stack = np.stack(rewards_per_run)
  if aggregator_rewards == "max":
    rewards_aggregate = np.max(rewards_stack, axis=0).tolist()
  else:
    rewards_aggregate = np.mean(rewards_stack, axis=0).tolist()
  rewards_std = np.std(rewards_stack, axis=0).tolist() if len(rewards_per_run) > 1 else [0.0] * len(rewards_aggregate)

  entropies_per_run = []
  for run in runs:
    entropies = run.get("entropy_mean", [])
    if entropies:
      resampled = resample_data(entropies, aggregator=aggregator_entropies)
      entropies_per_run.append(np.array(resampled))

  entropies_stack = np.stack(entropies_per_run) if entropies_per_run else np.array([])
  if aggregator_entropies == "max":
    entropies_aggregate = np.max(entropies_stack, axis=0).tolist() if entropies_per_run else []
  else:
    entropies_aggregate = np.mean(entropies_stack, axis=0).tolist() if entropies_per_run else []
  entropies_std = np.std(entropies_stack, axis=0).tolist() if len(entropies_per_run) > 1 else [0.0] * len(entropies_aggregate)

  return {
    "resampled_rewards_aggregate": rewards_aggregate,
    "resampled_rewards_std": rewards_std,
    "resampled_entropies_aggregate": entropies_aggregate,
    "resampled_entropies_std": entropies_std,
    "n_runs": len(runs),
  }


def plot_individual_rewards_entropies(env_id, cfg, rewards_aggregate, entropies_aggregate, rewards_std, entropies_std, log_dir, ntimesteps, label, n_runs=5):
  if rewards_aggregate and entropies_aggregate:
    timesteps = np.linspace(0, ntimesteps, len(rewards_aggregate))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Timesteps", fontsize=14)
    ax1.set_ylabel("Max Reward", color="#0072B2", fontsize=14)
    ax1.plot(timesteps, rewards_aggregate, color="#0072B2", label="Rewards")
    ax1.fill_between(
      timesteps, np.array(rewards_aggregate) - np.array(rewards_std), np.array(rewards_aggregate) + np.array(rewards_std), color="#0072B2", alpha=0.2
    )
    ax1.tick_params(axis="y", labelcolor="#0072B2", labelsize=12)
    ax1.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax2 = ax1.twinx()
    ax2.set_ylabel("Mean Entropy", color="#D55E00", fontsize=14)
    ax2.plot(timesteps, entropies_aggregate, color="#D55E00", label="Entropies")
    ax2.fill_between(
      timesteps, np.array(entropies_aggregate) - np.array(entropies_std), np.array(entropies_aggregate) + np.array(entropies_std), color="#D55E00", alpha=0.2
    )
    ax2.tick_params(axis="y", labelcolor="#D55E00", labelsize=12)
    fig.tight_layout()
    plt.title(f"Rewards and Entropies - {env_id} {label} (N={n_runs})", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(log_dir, f"graph_{env_id}_{cfg}_rewards_entropies.png"), bbox_inches="tight", dpi=300)
    plt.close()

    # Dump debug data for individual plot
    debug_data = {
      "timesteps": timesteps.tolist(),
      "rewards_aggregate": rewards_aggregate,
      "rewards_std": rewards_std,
      "entropies_aggregate": entropies_aggregate,
      "entropies_std": entropies_std,
    }
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
  sorted_keys = sorted(variant_data.keys(), key=lambda x: labels.get(x, x))
  global_min = np.inf
  global_max = -np.inf
  for key in sorted_keys:
    data = variant_data.get(key, {})
    rewards_aggregate = data.get("resampled_rewards_aggregate", [])
    if rewards_aggregate:
      global_min = min(global_min, min(rewards_aggregate))
      global_max = max(global_max, max(rewards_aggregate))
  for i, key in enumerate(sorted_keys):
    color = colorblind_palette[i % len(colorblind_palette)]
    data = variant_data.get(key, {})
    rewards_aggregate = data.get("resampled_rewards_aggregate", [])
    rewards_std = data.get("resampled_rewards_std", [])
    if rewards_aggregate:
      timesteps = np.linspace(0, ntimesteps, len(rewards_aggregate))
      plt.plot(timesteps, rewards_aggregate, color=color, label=f"{labels.get(key, key)} (N={data['n_runs']})")
      plt.fill_between(
        timesteps, np.array(rewards_aggregate) - np.array(rewards_std), np.array(rewards_aggregate) + np.array(rewards_std), color=color, alpha=0.2
      )
      comparative_rewards_data[key] = {"timesteps": timesteps.tolist(), "rewards_aggregate": rewards_aggregate, "rewards_std": rewards_std}

  if "_noise_mod" in suffix:
    parts = suffix.split("_")
    cfg = parts[1]  # assuming suffix = "_{cfg}_noise_mod"
    model_map = {"gentrpo": "GenTRPO", "trpo": "TRPO"}
    model_name = model_map.get(cfg, cfg.upper())
    title = f"{model_name} Noise Type Modulation (Rewards) - {env_id}"
    filename = f"graph_{env_id}_{model_name}_noise_mod_rewards.png"
  else:
    title = f"Training Progress per Model (Rewards){suffix} - {env_id} (N=5 per model)"
    filename = f"graph_{env_id}_models_rewards{suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"

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
  sorted_keys = sorted(variant_data.keys(), key=lambda x: labels.get(x, x))
  global_min = np.inf
  global_max = -np.inf
  for key in sorted_keys:
    data = variant_data.get(key, {})
    entropies_aggregate = data.get("resampled_entropies_aggregate", [])
    if entropies_aggregate:
      global_min = min(global_min, min(entropies_aggregate))
      global_max = max(global_max, max(entropies_aggregate))
  for i, key in enumerate(sorted_keys):
    color = colorblind_palette[i % len(colorblind_palette)]
    data = variant_data.get(key, {})
    entropies_aggregate = data.get("resampled_entropies_aggregate", [])
    entropies_std = data.get("resampled_entropies_std", [])
    if entropies_aggregate:
      timesteps = np.linspace(0, ntimesteps, len(entropies_aggregate))
      plt.plot(timesteps, entropies_aggregate, color=color, label=f"{labels.get(key, key)} (N={data['n_runs']})")
      plt.fill_between(
        timesteps, np.array(entropies_aggregate) - np.array(entropies_std), np.array(entropies_aggregate) + np.array(entropies_std), color=color, alpha=0.2
      )
      comparative_entropies_data[key] = {"timesteps": timesteps.tolist(), "entropies_aggregate": entropies_aggregate, "entropies_std": entropies_std}

  if "_noise_mod" in suffix:
    parts = suffix.split("_")
    cfg = parts[1]  # assuming suffix = "_{cfg}_noise_mod"
    model_map = {"gentrpo": "GenTRPO", "trpo": "TRPO"}
    model_name = model_map.get(cfg, cfg.upper())
    title = f"{model_name} Noise Type Modulation (Entropies) - {env_id}"
    filename = f"graph_{env_id}_{model_name}_noise_mod_entropies.png"
  else:
    title = f"Training Progress per Model (Entropies){suffix} - {env_id} (N=5 per model)"
    filename = f"graph_{env_id}_models_entropies{suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"

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
  sorted_keys = ["none", "action", "reward", "both"]
  sorted_keys = [k for k in sorted_keys if k in variant_data]
  num_levels = len(sorted_keys)
  cols = int(np.ceil(np.sqrt(num_levels)))
  rows = int(np.ceil(num_levels / cols))
  fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
  axs = np.array([axs]) if num_levels == 1 else axs.flatten()
  comparative_rewards_data = {}
  global_min = np.inf
  global_max = -np.inf
  for key in sorted_keys:
    rewards_arrays = variant_data[key]["rewards_arrays"]
    if len(rewards_arrays) > 0:
      aggregate = np.max(rewards_arrays, axis=0)
      global_min = min(global_min, np.min(aggregate))
      global_max = max(global_max, np.max(aggregate))
  for i, key in enumerate(sorted_keys):
    ax = axs[i]
    rewards_arrays = variant_data[key]["rewards_arrays"]
    n_runs = variant_data[key]["n_runs"]
    if len(rewards_arrays) == 0:
      continue
    aggregate = np.max(rewards_arrays, axis=0)
    stds = np.std(rewards_arrays, axis=0)
    timesteps = np.linspace(0, ntimesteps, len(aggregate))
    color = colorblind_palette[i % len(colorblind_palette)]
    ax.plot(timesteps, aggregate, color=color)
    ax.fill_between(timesteps, aggregate - stds, aggregate + stds, color=color, alpha=0.2)
    ax.set_title(f"{labels[key]} (N={n_runs})", fontsize=14)
    ax.set_xlabel("Timesteps", fontsize=14)
    ax.set_ylabel("Max Reward (±SD)", fontsize=14)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.set_ylim(global_min, global_max)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    comparative_rewards_data[key] = {"timesteps": timesteps.tolist(), "aggregate": aggregate.tolist(), "stds": stds.tolist()}
  # Hide unused axes
  for j in range(i + 1, len(axs)):
    axs[j].axis("off")
  model_map = {"gentrpo": "GenTRPO", "trpo": "TRPO"}
  model_name = model_map.get(cfg, cfg.upper())
  fig.suptitle(f"{model_name} Noise Type Modulation (Rewards) - {env_id}", fontsize=16)
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
  sorted_keys = ["none", "action", "reward", "both"]
  sorted_keys = [k for k in sorted_keys if k in variant_data]
  num_levels = len(sorted_keys)
  cols = int(np.ceil(np.sqrt(num_levels)))
  rows = int(np.ceil(num_levels / cols))
  fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
  axs = np.array([axs]) if num_levels == 1 else axs.flatten()
  comparative_entropies_data = {}
  global_min = np.inf
  global_max = -np.inf
  for key in sorted_keys:
    entropies_arrays = variant_data[key]["entropies_arrays"]
    if len(entropies_arrays) > 0:
      aggregate = np.mean(entropies_arrays, axis=0)
      global_min = min(global_min, np.min(aggregate))
      global_max = max(global_max, np.max(aggregate))
  for i, key in enumerate(sorted_keys):
    ax = axs[i]
    entropies_arrays = variant_data[key]["entropies_arrays"]
    n_runs = variant_data[key]["n_runs"]
    if len(entropies_arrays) == 0:
      continue
    aggregate = np.mean(entropies_arrays, axis=0)
    stds = np.std(entropies_arrays, axis=0)
    timesteps = np.linspace(0, ntimesteps, len(aggregate))
    color = colorblind_palette[i % len(colorblind_palette)]
    ax.plot(timesteps, aggregate, color=color)
    ax.fill_between(timesteps, aggregate - stds, aggregate + stds, color=color, alpha=0.2)
    ax.set_title(f"{labels[key]} (N={n_runs})", fontsize=14)
    ax.set_xlabel("Timesteps", fontsize=14)
    ax.set_ylabel("Mean Entropy (±SD)", fontsize=14)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.set_ylim(global_min, global_max)
    ax.tick_params(labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)
    comparative_entropies_data[key] = {"timesteps": timesteps.tolist(), "aggregate": aggregate.tolist(), "stds": stds.tolist()}
  # Hide unused axes
  for j in range(i + 1, len(axs)):
    axs[j].axis("off")
  model_map = {"gentrpo": "GenTRPO", "trpo": "TRPO"}
  model_name = model_map.get(cfg, cfg.upper())
  fig.suptitle(f"{model_name} Noise Type Modulation (Entropies) - {env_id}", fontsize=16)
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
    rewards_aggregate = data.get("resampled_rewards_aggregate", [])
    entropies_aggregate = data.get("resampled_entropies_aggregate", [])
    if rewards_aggregate:
      reward_values.extend(rewards_aggregate)
    if entropies_aggregate:
      entropy_values.extend(entropies_aggregate)
  global_reward_min = min(reward_values) if reward_values else 0
  global_reward_max = max(reward_values) if reward_values else 1
  global_entropy_min = min(entropy_values) if entropy_values else 0
  global_entropy_max = max(entropy_values) if entropy_values else 1

  num_envs = len(envs)
  fig, axs = plt.subplots(1, num_envs, figsize=(6 * num_envs, 5))
  grid_model_data = {}
  for i, env_id in enumerate(envs):
    data = all_variant_data[env_id].get(cfg, {})
    rewards_aggregate = data.get("resampled_rewards_aggregate", [])
    rewards_std = data.get("resampled_rewards_std", [])
    entropies_aggregate = data.get("resampled_entropies_aggregate", [])
    entropies_std = data.get("resampled_entropies_std", [])
    if rewards_aggregate:
      timesteps = np.linspace(0, ntimesteps, len(rewards_aggregate))
      ax1 = axs[i] if num_envs > 1 else axs
      ax1.plot(timesteps, rewards_aggregate, color="#0072B2", label="Reward")
      ax1.fill_between(
        timesteps, np.array(rewards_aggregate) - np.array(rewards_std), np.array(rewards_aggregate) + np.array(rewards_std), color="#0072B2", alpha=0.2
      )
      ax1.set_xlabel("Timestep", fontsize=14)
      ax1.set_ylabel("Max Reward", color="#0072B2", fontsize=14)
      ax1.tick_params(axis="y", labelcolor="#0072B2", labelsize=12)
      ax1.set_title(f"{env_id} (N={data['n_runs']})", fontsize=14)
      ax1.set_ylim(global_reward_min, global_reward_max)
      ax1.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
      if entropies_aggregate:
        ax2 = ax1.twinx()
        ax2.plot(timesteps, entropies_aggregate, color="#D55E00", label="Entropy")
        ax2.fill_between(
          timesteps,
          np.array(entropies_aggregate) - np.array(entropies_std),
          np.array(entropies_aggregate) + np.array(entropies_std),
          color="#D55E00",
          alpha=0.2,
        )
        ax2.set_ylabel("Mean Entropy", color="#D55E00", fontsize=14)
        ax2.tick_params(axis="y", labelcolor="#D55E00", labelsize=12)
        ax2.set_ylim(global_entropy_min, global_entropy_max)
      grid_model_data[env_id] = {
        "timesteps": timesteps.tolist(),
        "rewards_aggregate": rewards_aggregate,
        "rewards_std": rewards_std,
        "entropies_aggregate": entropies_aggregate,
        "entropies_std": entropies_std,
      }
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
    rewards_aggregate = data.get("resampled_rewards_aggregate", [])
    entropies_aggregate = data.get("resampled_entropies_aggregate", [])
    if rewards_aggregate:
      reward_values.extend(rewards_aggregate)
    if entropies_aggregate:
      entropy_values.extend(entropies_aggregate)
  global_reward_min = min(reward_values) if reward_values else 0
  global_reward_max = max(reward_values) if reward_values else 1
  global_entropy_min = min(entropy_values) if entropy_values else 0
  global_entropy_max = max(entropy_values) if entropy_values else 1

  num_variants = len(variants)
  fig, axs = plt.subplots(1, num_variants, figsize=(6 * num_variants, 5))
  grid_env_data = {}
  for i, cfg in enumerate(variants):
    data = all_variant_data[env_id].get(cfg, {})
    rewards_aggregate = data.get("resampled_rewards_aggregate", [])
    rewards_std = data.get("resampled_rewards_std", [])
    entropies_aggregate = data.get("resampled_entropies_aggregate", [])
    entropies_std = data.get("resampled_entropies_std", [])
    if rewards_aggregate:
      timesteps = np.linspace(0, ntimesteps, len(rewards_aggregate))
      ax1 = axs[i] if num_variants > 1 else axs
      ax1.plot(timesteps, rewards_aggregate, color="#0072B2", label="Reward")
      ax1.fill_between(
        timesteps, np.array(rewards_aggregate) - np.array(rewards_std), np.array(rewards_aggregate) + np.array(rewards_std), color="#0072B2", alpha=0.2
      )
      ax1.set_xlabel("Timestep", fontsize=14)
      ax1.set_ylabel("Max Reward", color="#0072B2", fontsize=14)
      ax1.tick_params(axis="y", labelcolor="#0072B2", labelsize=12)
      ax1.set_title(f"{variant_labels[cfg]} (N={data['n_runs']})", fontsize=14)
      ax1.set_ylim(global_reward_min, global_reward_max)
      ax1.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
      if entropies_aggregate:
        ax2 = ax1.twinx()
        ax2.plot(timesteps, entropies_aggregate, color="#D55E00", label="Entropy")
        ax2.fill_between(
          timesteps,
          np.array(entropies_aggregate) - np.array(entropies_std),
          np.array(entropies_aggregate) + np.array(entropies_std),
          color="#D55E00",
          alpha=0.2,
        )
        ax2.set_ylabel("Mean Entropy", color="#D55E00", fontsize=14)
        ax2.tick_params(axis="y", labelcolor="#D55E00", labelsize=12)
        ax2.set_ylim(global_entropy_min, global_entropy_max)
      grid_env_data[cfg] = {
        "timesteps": timesteps.tolist(),
        "rewards_aggregate": rewards_aggregate,
        "rewards_std": rewards_std,
        "entropies_aggregate": entropies_aggregate,
        "entropies_std": entropies_std,
      }
  fig.suptitle(f"Plots for {env_id} across Models", fontsize=16)
  plt.tight_layout()
  plt.savefig(os.path.join(log_dir, f"grid_env_{env_id}.png"))
  plt.close()

  # Dump debug data for grid env
  debug_dir = ".debug"
  debug_path = os.path.join(debug_dir, f"grid_env_{env_id}.yaml")
  with open(debug_path, "w") as debug_file:
    yaml.dump(grid_env_data, debug_file)


def plot_results(results_dir, ntimesteps):
  envs = ["Humanoid-v5", "HumanoidStandup-v5"]
  variants = ["trpo", "gentrpo"]
  variant_labels = {"trpo": "TRPO Noise=0", "gentrpo": "GenTRPO Noise=0"}
  noise_types = ["none", "action", "reward", "both"]
  all_variant_data = {env: {} for env in envs}

  # Ensure .debug directory exists
  debug_dir = ".debug"
  os.makedirs(debug_dir, exist_ok=True)

  for env_id in envs:
    for noise_type in noise_types:
      variant_data = {}
      for cfg in variants:
        data = load_aggregated_data(env_id, cfg, results_dir, noise_type=noise_type, aggregator_rewards="max", aggregator_entropies="mean")
        if data:
          variant_data[cfg] = data

          # Individual plot for rewards and entropies
          label = variant_labels[cfg].replace("Noise=0", f"Noise {noise_type.capitalize()}")
          plot_individual_rewards_entropies(
            env_id,
            f"{cfg}_{noise_type}",
            data["resampled_rewards_aggregate"],
            data["resampled_entropies_aggregate"],
            data["resampled_rewards_std"],
            data["resampled_entropies_std"],
            results_dir,
            ntimesteps,
            label,
            n_runs=data["n_runs"],
          )

      if variant_data:
        # Comparative rewards plot for the environment and noise_type
        plot_comparative_rewards(env_id, variant_data, results_dir, ntimesteps, variant_labels, suffix=f" - {noise_type.capitalize()} Noise")

        # Plot combined entropies
        plot_comparative_entropies(env_id, variant_data, results_dir, ntimesteps, variant_labels, suffix=f" - {noise_type.capitalize()} Noise")

    # For grid plots, use noise_type="none" as default
    variant_data_none = {}
    for cfg in variants:
      data = load_aggregated_data(env_id, cfg, results_dir, noise_type="none")
      if data:
        variant_data_none[cfg] = data
    all_variant_data[env_id] = variant_data_none

  # Grid for each model (variant) with plots for each env (using none noise)
  for cfg in variants:
    plot_grid_model(cfg, all_variant_data, results_dir, ntimesteps, envs, variant_labels)

  # Grid for each env with plots for each model (using none noise)
  for env_id in envs:
    plot_grid_env(env_id, all_variant_data, results_dir, ntimesteps, variants, variant_labels)


def plot_noise_modulation(results_dir, env_id, cfg, ntimesteps):
  path = os.path.join(results_dir, f"{env_id}_{cfg}_raw.yaml")
  if not os.path.exists(path):
    print(f"File not found: {path}")
    return

  with open(path, "r") as f:
    data = yaml.safe_load(f)

  variant_data = {}
  labels = {}
  for run in data:
    if not run.get("completed", False):
      continue
    noise_type = run.get("noise_type", "none")
    key = noise_type
    if key not in variant_data:
      variant_data[key] = {"rewards_arrays": [], "entropies_arrays": [], "n_runs": 0}
    rewards = run.get("reward_mean", [])
    if rewards:
      resampled_rewards = resample_data(rewards, aggregator="max")
      variant_data[key]["rewards_arrays"].append(np.array(resampled_rewards))
    entropies = run.get("entropy_mean", [])
    if entropies:
      resampled_entropies = resample_data(entropies, aggregator="mean")
      variant_data[key]["entropies_arrays"].append(np.array(resampled_entropies))
    variant_data[key]["n_runs"] += 1
    if key not in labels:
      if key == "none":
        labels[key] = "Noise=0"
      elif key == "action":
        labels[key] = "Action Noise"
      elif key == "reward":
        labels[key] = "Reward Noise"
      elif key == "both":
        labels[key] = "Noise=0.1" if env_id == "Humanoid-v5" else "Noise=-0.5"

  # Grid rewards plot showing impact of noise
  plot_noise_mod_rewards_grid(env_id, variant_data, results_dir, ntimesteps, labels, cfg)

  # Grid entropies plot
  plot_noise_mod_entropies_grid(env_id, variant_data, results_dir, ntimesteps, labels, cfg)


if __name__ == "__main__":

  ntimesteps = 100000

  results_dir = "results"
  plot_results(results_dir, ntimesteps)
  plot_noise_modulation(results_dir, "Humanoid-v5", "gentrpo", ntimesteps)
  plot_noise_modulation(results_dir, "Humanoid-v5", "trpo", ntimesteps)
  plot_noise_modulation(results_dir, "HumanoidStandup-v5", "gentrpo", ntimesteps)
  plot_noise_modulation(results_dir, "HumanoidStandup-v5", "trpo", ntimesteps)
