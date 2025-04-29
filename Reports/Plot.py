import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from matplotlib.ticker import FixedFormatter, FixedLocator
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from scipy.stats import binom_test, ttest_ind


# Normalize data to 100 points (1,000,000 steps)
def normalize_data(values, original_steps, target_steps=1000000, target_points=100):
  """
    Normalize a list of values to a fixed number of points representing 1,000,000 steps.

    Args:
        values (list): Original data (e.g., rewards or entropies).
        original_steps (int): Total steps in the original data (e.g., 1000000).
        target_steps (int): Desired total steps (1000000).
        target_points (int): Number of points to interpolate to (100).

    Returns:
        np.ndarray: Interpolated values with target_points.
    """
  values = np.array(values)
  if len(values) < 2:
    return np.repeat(values, target_points)[:target_points]
  original_points = len(values)
  original_x = np.linspace(0, original_steps, original_points)
  target_x = np.linspace(0, target_steps, target_points)
  interpolator = interp1d(original_x, values, kind="linear", fill_value="extrapolate")
  return interpolator(target_x)


# Load data from YAML files with robust error handling
def load_data(data_dir):
  """
    Load data from YAML files in the specified directory.

    Args:
        data_dir (str): Path to directory containing environment subdirectories.

    Returns:
        dict: {env_name: {algo_name: [{'label': str, 'rewards': list, 'entropies': list}, ...]}}
    """
  data = {}
  for env_dir in os.listdir(data_dir):
    env_path = os.path.join(data_dir, env_dir)
    if not os.path.isdir(env_path):
      continue
    env_name = env_dir
    data[env_name] = {}
    for file in os.listdir(env_path):
      # Skip non-data files
      if not file.endswith(".yml") or file.lower() == "config.yaml" or "summary" in file.lower():
        continue
      algo_name = file.split("_")[0]
      file_path = os.path.join(env_path, file)
      with open(file_path, "r") as f:
        try:
          algo_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
          print(f"Error parsing {file_path}: {e}")
          continue
      if not isinstance(algo_data, list):
        print(f"Warning: {file_path} does not contain a list. Skipping.")
        continue
      for entry in algo_data:
        if not isinstance(entry, dict) or "smoothed_data" not in entry:
          print(f"Warning: Invalid entry in {file_path}. Skipping.")
          continue
        smoothed_data = entry.get("smoothed_data", [])
        if not isinstance(smoothed_data, list):
          print(f"Warning: smoothed_data in {file_path} is not a list. Skipping.")
          continue
        for smoothed in smoothed_data:
          if not isinstance(smoothed, dict) or not all(k in smoothed for k in ["label", "rewards", "entropies"]):
            print(f"Warning: Missing keys in smoothed data of {file_path}. Skipping.")
            continue
          # Normalize rewards and entropies to 100 points
          rewards = normalize_data(smoothed["rewards"], original_steps=1000000)
          entropies = normalize_data(smoothed["entropies"], original_steps=1000000)
          if algo_name not in data[env_name]:
            data[env_name][algo_name] = []
          data[env_name][algo_name].append({"label": smoothed["label"], "rewards": rewards, "entropies": entropies})
  return data


# Get global y-axis bounds for a metric across runs
def get_y_bounds(data, metric, keys):
  """
    Compute global min and max for a metric across specified keys.

    Args:
        data: Loaded data.
        metric (str): 'rewards' or 'entropies'.
        keys: List of (env, algo) tuples to consider.

    Returns:
        tuple: (y_min, y_max)
    """
  y_min, y_max = np.inf, -np.inf
  for env, algo in keys:
    if env in data and algo in data[env]:
      for run in data[env][algo]:
        values = run[metric]
        y_min = min(y_min, np.min(values))
        y_max = max(y_max, np.max(values))
  # Add padding for readability
  padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
  return y_min - padding, y_max + padding


# Format legend labels to exclude noise_type
def format_label(label):
  """
    Format YAML label to show 'Baseline' or 'Noise (value)'.

    Args:
        label (str): Original label from YAML (e.g., 'Baseline', 'reward+action_uniform(-0.5)').

    Returns:
        str: Formatted label (e.g., 'Baseline', 'Noise (-0.5)').
    """
  if label.lower() == "baseline":
    return "Baseline"
  # Match patterns like 'reward+action_uniform(-0.5)' to extract value
  match = re.match(r".*\(([-+]?[0-9]*\.?[0-9]+)\)", label)
  if match:
    value = match.group(1)
    return f"Noise ({value})"
  return label  # Fallback to original label if parsing fails


# New function to plot comparison of two algorithms across shared environments (rewards only)
def plot_algo_pair_comparison(data, algo1, algo2, environments, output_dir, smooth_window=5, markers_per_line=10):
  """
    Plot rewards for two algorithms across shared environments in a 1xN grid, with linear y-axis scale,
    x-axis with decimal ticks and scientific notation note for 1M steps, frequent lower y-ticks, tiny fonts,
    increased offset markers, y-label only on first subplot, and single legend at the bottom.

    Args:
        data: Loaded data.
        algo1, algo2 (str): Algorithm names to compare.
        environments: List of environment names.
        output_dir (str): Where to save the plot.
        smooth_window (int): Smoothing window size (default: 5).
        markers_per_line (int): Number of markers to show per line (default: 10).
    """
  # Find shared environments
  valid_envs = sorted([env for env in environments if env in data and algo1 in data[env] and algo2 in data[env]])
  if not valid_envs:
    print(f"No shared environments for {algo1} and {algo2}.")
    return
  ncols = len(valid_envs)
  fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3), squeeze=False)
  markers = ["o", "s", "^", "v", "*", "+", "x"]
  handles, labels = [], []
  marker_interval = max(1, 100 // markers_per_line)
  for i, env in enumerate(valid_envs):
    ax = axes[0, i]
    keys = [(env, algo1), (env, algo2)]
    y_min, y_max = get_y_bounds(data, "rewards", keys)
    for algo_idx, algo in enumerate([algo1, algo2]):
      runs = data[env][algo]
      for j, run in enumerate(runs):
        rewards = run["rewards"]
        if smooth_window > 1:
          rewards = uniform_filter1d(rewards, size=smooth_window, mode="nearest")
        steps = np.linspace(0, 1000000, len(rewards))
        offset = (j + algo_idx * len(runs)) * marker_interval // max(1, len(runs) * 2)
        linestyle = "-" if algo == algo1 else "--"  # Solid for algo1, dashed for algo2
        (line,) = ax.plot(
          steps,
          rewards,
          label=f"{algo}: {format_label(run['label'])}",
          color="black",
          linestyle=linestyle,
          marker=markers[j % len(markers)],
          markevery=(offset, marker_interval),
          linewidth=2,
          alpha=0.7,
          markersize=6,
        )
        if i == 0:
          handles.append(line)
          labels.append(f"{algo}: {format_label(run['label'])}")
    ax.set_title(f"{env}", fontsize=10)
    if i == 0:
      ax.set_ylabel("Rewards", fontsize=6)
    ax.set_xlabel("Steps", fontsize=6)
    ax.grid(True, linestyle="--", alpha=0.7)
    rewards_range = y_max - y_min
    if rewards_range > 0:
      y_ticks = np.concatenate(
        [
          np.arange(y_min, y_min + rewards_range / 4, rewards_range / 20),
          np.arange(y_min + rewards_range / 4, y_max, rewards_range / 10),
        ]
      )
      y_ticks = np.unique(np.clip(y_ticks, y_min, y_max))
      ax.set_yticks(y_ticks, labels=[f"{tick:.2f}" for tick in y_ticks])
    ax.set_ylim(y_min, y_max)
    x_ticks = [0, 200000, 400000, 600000, 800000, 1000000]
    x_tick_labels = [f"{x // 100000}" for x in x_ticks]
    ax.set_xticks(x_ticks, labels=x_tick_labels)
    ax.tick_params(axis="both", labelsize=5)
    ax.set_xlim(0, 1000000)
  fig.text(0.5, 0.01, r"Steps (\(\times 10^6\))", ha="center", fontsize=6)
  for ax in axes.flatten():
    if ax.get_legend() is not None:
      ax.get_legend().remove()
  if handles:
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=len(labels), fontsize=8)
  plt.tight_layout(rect=[0, 0.1, 1, 0.95])
  os.makedirs(output_dir, exist_ok=True)
  fig.savefig(os.path.join(output_dir, f"{algo1}_vs_{algo2}_rewards.png"), dpi=300, bbox_inches="tight")
  plt.close(fig)


# Plot returns and entropy vs steps for one algorithm across all environments
def plot_algo_across_envs(data, algo, environments, output_dir, smooth_window=5, markers_per_line=10):
  """
    Plot all runs for one algorithm across 6 environments in a 2x7 tiled grid with returns (top row)
    and entropy (bottom row), with linear y-axis scale (per-environment for both rewards and entropy),
    x-axis with decimal ticks and scientific notation note for 1M steps, frequent lower y-ticks, tiny
    fonts, increased offset markers, y-label only on first cell of each row, and single legend at the bottom.

    Args:
        data: Loaded data.
        algo (str): Algorithm name.
        environments: List of environment names (expected up to 7).
        output_dir (str): Where to save the plot.
        smooth_window (int): Smoothing window size (default: 5).
        markers_per_line (int): Number of markers to show per line (default: 10).
    """
  valid_envs = sorted([env for env in environments if env in data and algo in data[env]])
  if not valid_envs:
    print(f"No data for {algo} in any environments.")
    return
  # Fixed 2x7 grid for 12 square tiles
  nrows, ncols = 2, 7
  fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6), squeeze=False)
  # Define markers
  markers = ["o", "s", "^", "v", "*", "+", "x"]  # Markers for distinction
  # Get y-axis bounds for returns and entropy per environment
  handles, labels = [], []
  marker_interval = max(1, 100 // markers_per_line)  # Interval between markers
  for i in range(7):
    # Top row: Returns
    ax_returns = axes[0, i]
    # Bottom row: Entropy
    ax_entropy = axes[1, i]
    if i < len(valid_envs):
      env = valid_envs[i]
      runs = data[env][algo]
      # Compute y-axis bounds for this environment
      keys = [(env, algo)]
      rewards_y_min, rewards_y_max = get_y_bounds(data, "rewards", keys)
      entropies_y_min, entropies_y_max = get_y_bounds(data, "entropies", keys)
      for j, run in enumerate(runs):
        rewards = run["rewards"]
        entropies = run["entropies"]
        if smooth_window > 1:
          rewards = uniform_filter1d(rewards, size=smooth_window, mode="nearest")
          entropies = uniform_filter1d(entropies, size=smooth_window, mode="nearest")
        steps = np.linspace(0, 1000000, len(rewards))  # 1M steps
        # Use increased offset for markers to prevent overlap
        offset = j * marker_interval // max(1, len(runs))
        (line,) = ax_returns.plot(
          steps,
          rewards,
          label=format_label(run["label"]),
          color="black",
          linestyle="-",
          marker=markers[j % len(markers)],
          markevery=(offset, marker_interval),
          linewidth=2,
          alpha=0.7,
          markersize=6,
        )
        ax_entropy.plot(
          steps,
          entropies,
          label=format_label(run["label"]),
          color="black",
          linestyle="-",
          marker=markers[j % len(markers)],
          markevery=(offset, marker_interval),
          linewidth=2,
          alpha=0.7,
          markersize=6,
        )
        # Collect handles and labels from the first returns subplot
        if i == 0:
          handles.append(line)
          labels.append(format_label(run["label"]))
      ax_returns.set_title(f"{env}", fontsize=10)
      # Set y-label only on first cell of each row
      if i == 0:
        ax_returns.set_ylabel("Rewards", fontsize=6)
        ax_entropy.set_ylabel("Entropy", fontsize=6)
      ax_entropy.set_xlabel("Steps", fontsize=6)
      ax_returns.grid(True, linestyle="--", alpha=0.7)
      ax_entropy.grid(True, linestyle="--", alpha=0.7)
      # Set linear y-axis with frequent lower ticks
      rewards_range = rewards_y_max - rewards_y_min
      if rewards_range > 0:
        y_ticks = np.concatenate(
          [
            np.arange(rewards_y_min, rewards_y_min + rewards_range / 4, rewards_range / 20),  # Frequent lower ticks
            np.arange(rewards_y_min + rewards_range / 4, rewards_y_max, rewards_range / 10),  # Infrequent upper ticks
          ]
        )
        y_ticks = np.unique(np.clip(y_ticks, rewards_y_min, rewards_y_max))
        ax_returns.set_yticks(y_ticks, labels=[f"{tick:.2f}" for tick in y_ticks])
      entropies_range = entropies_y_max - entropies_y_min
      if entropies_range > 0:
        y_ticks = np.concatenate(
          [
            np.arange(entropies_y_min, entropies_y_min + entropies_range / 4, entropies_range / 20),  # Frequent lower ticks
            np.arange(entropies_y_min + entropies_range / 4, entropies_y_max, entropies_range / 10),  # Infrequent upper ticks
          ]
        )
        y_ticks = np.unique(np.clip(y_ticks, entropies_y_min, entropies_y_max))
        ax_entropy.set_yticks(y_ticks, labels=[f"{tick:.2f}" for tick in y_ticks])
      ax_returns.set_ylim(rewards_y_min, rewards_y_max)
      ax_entropy.set_ylim(entropies_y_min, entropies_y_max)
      # Set x-axis ticks as decimals (steps in 10^6 units)
      x_ticks = [0, 200000, 400000, 600000, 800000, 1000000]
      x_tick_labels = [f"{x // 100000}" for x in x_ticks]  # Divide by 10^5 for display
      ax_returns.set_xticks(x_ticks, labels=x_tick_labels)
      ax_entropy.set_xticks(x_ticks, labels=x_tick_labels)
      ax_returns.tick_params(axis="both", labelsize=5)
      ax_entropy.tick_params(axis="both", labelsize=5)
      ax_returns.set_xlim(0, 1000000)
      ax_entropy.set_xlim(0, 1000000)
    else:
      ax_returns.set_visible(False)  # Empty subplot for missing data
      ax_entropy.set_visible(False)  # Empty subplot for missing data
  # Add scientific notation note for x-axis
  fig.text(0.5, 0.01, r"Steps (\(\times 10^6\))", ha="center", fontsize=6)
  # Remove individual legends
  for ax in axes.flatten():
    if ax.get_legend() is not None:
      ax.get_legend().remove()
  # Add single legend at the bottom as a horizontal row
  if handles:
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=len(labels), fontsize=8)
  plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for legend and note space
  os.makedirs(output_dir, exist_ok=True)
  fig.savefig(os.path.join(output_dir, f"{algo}_rewards_entropy_vs_steps.png"), dpi=300, bbox_inches="tight")
  plt.close(fig)


# Plot all algorithms for one environment, each in its own subplot
def plot_env_all_algos(data, env, algorithms, output_dir, smooth_window=5, markers_per_line=10):
  """
    Plot all runs for each algorithm in one environment in a 2x7 tiled grid with returns (top row)
    and entropy (bottom row), with linear y-axis scale (shared for rewards, per-algorithm for entropy),
    x-axis with decimal ticks and scientific notation note for 1M steps, frequent lower y-ticks, tiny
    fonts, increased offset markers, y-label only on first cell of each row, and single legend at the bottom.

    Args:
        data: Loaded data.
        env (str): Environment name.
        algorithms: List of algorithm names (expected up to 7).
        output_dir (str): Where to save the plot.
        smooth_window (int): Smoothing window size (default: 5).
        markers_per_line (int): Number of markers to show per line (default: 10).
    """
  if env not in data:
    print(f"No data for {env}.")
    return
  valid_algos = sorted([algo for algo in algorithms if algo in data[env]])
  if not valid_algos:
    print(f"No algorithms for {env}.")
    return
  # Fixed 2x7 grid for 12 square tiles
  nrows, ncols = 2, 7
  fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6), squeeze=False)
  # Define markers
  markers = ["o", "s", "^", "v", "*", "+", "x"]  # Markers for distinction
  # Get global y-axis bounds for rewards across all algorithms
  keys = [(env, algo) for algo in valid_algos]
  rewards_y_min, rewards_y_max = get_y_bounds(data, "rewards", keys)
  marker_interval = max(1, 100 // markers_per_line)  # Interval between markers
  handles, labels = [], []
  for i in range(7):
    ax_returns = axes[0, i]
    ax_entropy = axes[1, i]
    if i < len(valid_algos):
      algo = valid_algos[i]
      runs = data[env][algo]
      # Compute y-axis bounds for entropy for this algorithm
      keys = [(env, algo)]
      entropies_y_min, entropies_y_max = get_y_bounds(data, "entropies", keys)
      for j, run in enumerate(runs):
        rewards = run["rewards"]
        entropies = run["entropies"]
        if smooth_window > 1:
          rewards = uniform_filter1d(rewards, size=smooth_window, mode="nearest")
          entropies = uniform_filter1d(entropies, size=smooth_window, mode="nearest")
        steps = np.linspace(0, 1000000, len(rewards))  # 1M steps
        # Use increased offset for markers to prevent overlap
        offset = j * marker_interval // max(1, len(runs))
        (line,) = ax_returns.plot(
          steps,
          rewards,
          label=format_label(run["label"]),
          color="black",
          linestyle="-",
          marker=markers[j % len(markers)],
          markevery=(offset, marker_interval),
          linewidth=2,
          alpha=0.7,
          markersize=6,
        )
        ax_entropy.plot(
          steps,
          entropies,
          label=format_label(run["label"]),
          color="black",
          linestyle="-",
          marker=markers[j % len(markers)],
          markevery=(offset, marker_interval),
          linewidth=2,
          alpha=0.7,
          markersize=6,
        )
        # Collect handles and labels from the first returns subplot
        if i == 0:
          handles.append(line)
          labels.append(format_label(run["label"]))
      ax_returns.set_title(f"{algo}", fontsize=10)
      # Set y-label only on first cell of each row
      if i == 0:
        ax_returns.set_ylabel("Rewards", fontsize=6)
        ax_entropy.set_ylabel("Entropy", fontsize=6)
      ax_entropy.set_xlabel("Steps", fontsize=6)
      ax_returns.grid(True, linestyle="--", alpha=0.7)
      ax_entropy.grid(True, linestyle="--", alpha=0.7)
      # Set linear y-axis with frequent lower ticks
      rewards_range = rewards_y_max - rewards_y_min
      if rewards_range > 0:
        y_ticks = np.concatenate(
          [
            np.arange(rewards_y_min, rewards_y_min + rewards_range / 4, rewards_range / 20),  # Frequent lower ticks
            np.arange(rewards_y_min + rewards_range / 4, rewards_y_max, rewards_range / 10),  # Infrequent upper ticks
          ]
        )
        y_ticks = np.unique(np.clip(y_ticks, rewards_y_min, rewards_y_max))
        ax_returns.set_yticks(y_ticks, labels=[f"{tick:.2f}" for tick in y_ticks])
      entropies_range = entropies_y_max - entropies_y_min
      if entropies_range > 0:
        y_ticks = np.concatenate(
          [
            np.arange(entropies_y_min, entropies_y_min + entropies_range / 4, entropies_range / 20),  # Frequent lower ticks
            np.arange(entropies_y_min + entropies_range / 4, entropies_y_max, entropies_range / 10),  # Infrequent upper ticks
          ]
        )
        y_ticks = np.unique(np.clip(y_ticks, entropies_y_min, entropies_y_max))
        ax_entropy.set_yticks(y_ticks, labels=[f"{tick:.2f}" for tick in y_ticks])
      ax_returns.set_ylim(rewards_y_min, rewards_y_max)
      ax_entropy.set_ylim(entropies_y_min, entropies_y_max)
      # Set x-axis ticks as decimals (steps in 10^6 units)
      x_ticks = [0, 200000, 400000, 600000, 800000, 1000000]
      x_tick_labels = [f"{x // 100000}" for x in x_ticks]  # Divide by 10^5 for display
      ax_returns.set_xticks(x_ticks, labels=x_tick_labels)
      ax_entropy.set_xticks(x_ticks, labels=x_tick_labels)
      ax_returns.tick_params(axis="both", labelsize=5)
      ax_entropy.tick_params(axis="both", labelsize=5)
      ax_returns.set_xlim(0, 1000000)
      ax_entropy.set_xlim(0, 1000000)
    else:
      ax_returns.set_visible(False)  # Empty subplot for missing data
      ax_entropy.set_visible(False)  # Empty subplot for missing data
  # Add scientific notation note for x-axis
  fig.text(0.5, 0.01, r"Steps (\(\times 10^6\))", ha="center", fontsize=6)
  # Remove individual legends
  for ax in axes.flatten():
    if ax.get_legend() is not None:
      ax.get_legend().remove()
  # Add single legend at the bottom as a horizontal row
  if handles:
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=len(labels), fontsize=8)
  plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for legend and note space
  os.makedirs(output_dir, exist_ok=True)
  fig.savefig(os.path.join(output_dir, f"{env}_rewards_entropy_vs_steps_all_algos.png"), dpi=300, bbox_inches="tight")
  plt.close(fig)


# Reuse load_data from your plotting script, modified to store raw rewards
def load_data(data_dir):
  data = {}
  for env_dir in os.listdir(data_dir):
    env_path = os.path.join(data_dir, env_dir)
    if not os.path.isdir(env_path):
      continue
    env_name = env_dir
    data[env_name] = {}
    for file in os.listdir(env_path):
      if not file.endswith(".yml") or file.lower() == "config.yaml" or "summary" in file.lower():
        continue
      algo_name = file.split("_")[0]
      file_path = os.path.join(env_path, file)
      with open(file_path, "r") as f:
        try:
          algo_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
          print(f"Error parsing {file_path}: {e}")
          continue
      if not isinstance(algo_data, list):
        print(f"Warning: {file_path} does not contain a list. Skipping.")
        continue
      for entry in algo_data:
        if not isinstance(entry, dict) or "smoothed_data" not in entry:
          print(f"Warning: Invalid entry in {file_path}. Skipping.")
          continue
        smoothed_data = entry.get("smoothed_data", [])
        if not isinstance(smoothed_data, list):
          print(f"Warning: smoothed_data in {file_path} is not a list. Skipping.")
          continue
        for smoothed in smoothed_data:
          if not isinstance(smoothed, dict) or not all(k in smoothed for k in ["label", "rewards", "entropies"]):
            print(f"Warning: Missing keys in smoothed data of {file_path}. Skipping.")
            continue
          rewards = normalize_data(smoothed["rewards"], original_steps=1000000)
          if algo_name not in data[env_name]:
            data[env_name][algo_name] = []
          data[env_name][algo_name].append({"label": smoothed["label"], "rewards": rewards})
  return data


# Generate LaTeX table and statistical sections
def generate_latex_outputs(data, output_dir, algorithms, environments):
  os.makedirs(output_dir, exist_ok=True)
  latex_table = []
  latex_significance = []
  latex_comparison = []

  # LaTeX Table: Max reward, variation, best noise level
  latex_table.append("\\begin{table}[h]")
  latex_table.append("\\centering")
  latex_table.append("\\caption{Maximum Rewards, Variation, and Best Noise Level for Each Algorithm and Environment}")
  latex_table.append("\\label{tab:numerical_results}")
  latex_table.append("\\begin{tabular}{|l|" + "c|" * len(environments) + "}")
  latex_table.append("\\hline")
  latex_table.append("Algorithm & " + " & ".join([env.replace("-v5", "") for env in environments]) + " \\\\ \\hline")

  for algo in algorithms:
    row = [algo]
    for env in environments:
      if env not in data or algo not in data[env]:
        row.append("-")
        continue
      max_reward, std_dev, best_noise = 0, 0, "N/A"
      best_avg = -np.inf
      for run_group in data[env][algo]:
        avg_reward = np.mean([np.max(run["rewards"]) for run in run_group])
        if avg_reward > best_avg:
          best_avg = avg_reward
          max_reward = max([np.max(run["rewards"]) for run in run_group])
          std_dev = np.std([np.max(run["rewards"]) for run in run_group])
          best_noise = run_group[0]["label"]
      row.append(f"{max_reward:.2f} $\\pm$ {std_dev:.2f} ({best_noise})")
    latex_table.append(" & ".join(row) + " \\\\ \\hline")

  latex_table.append("\\end{tabular}")
  latex_table.append("\\end{table}")

  # LaTeX Section: Statistical Significance vs. Baseline
  latex_significance.append("\\section{Statistical Significance Against Baseline}")
  latex_significance.append(
    "This section evaluates the statistical significance of each algorithm’s performance compared to its baseline (PPO for GenPPO, TRPO for GenTRPO, TRPOR, TRPOER) using a two-sample t-test. The p-value indicates the probability that a noise-trained run outperforms the baseline."
  )

  for algo, baseline in [("GenPPO", "PPO"), ("GenTRPO", "TRPO"), ("TRPOR", "TRPO"), ("TRPOER", "TRPO")]:
    latex_significance.append(f"\\subsection{{{algo} vs. {baseline}}}")
    for env in environments:
      if env not in data or algo not in data[env] or baseline not in data[env]:
        continue
      baseline_runs = [np.max(run["rewards"]) for run in data[env][baseline] if run["label"].lower() == "baseline"]
      if not baseline_runs:
        continue
      latex_significance.append(f"\\paragraph{{{env.replace('-v5', '')}}}")
      for run_group in data[env][algo]:
        if run_group[0]["label"].lower() == "baseline":
          continue
        noise_level = run_group[0]["label"]
        algo_runs = [np.max(run["rewards"]) for run in run_group]
        t_stat, p_value = ttest_ind(algo_runs, baseline_runs, equal_var=False)
        latex_significance.append(f"For noise level {noise_level}, the p-value is {p_value:.3f}, indicating the likelihood that {algo} outperforms {baseline}.")

  # LaTeX Section: Comparison Statistics (GenPPO vs. PPO, GenTRPO vs. TRPO, Best Model)
  latex_comparison.append("\\section{Comparison Statistics}")
  latex_comparison.append(
    "This section analyzes the probability that GenPPO outperforms PPO and GenTRPO outperforms TRPO based on maximum rewards across all runs, using a binomial test. It also identifies the model and noise configuration most likely to achieve the highest reward per environment."
  )

  for env in environments:
    if env not in data:
      continue
    latex_comparison.append(f"\\subsection{{{env.replace('-v5', '')}}}")
    # GenPPO vs. PPO
    if "PPO" in data[env] and "GenPPO" in data[env]:
      ppo_runs = [np.max(run["rewards"]) for run_group in data[env]["PPO"] for run in run_group]
      genppo_runs = [np.max(run["rewards"]) for run_group in data[env]["GenPPO"] for run in run_group]
      genppo_wins = sum(1 for g in genppo_runs if any(g > p for p in ppo_runs))
      total_comparisons = len(genppo_runs) * len(ppo_runs)
      prob = genppo_wins / total_comparisons if total_comparisons > 0 else 0
      latex_comparison.append(f"The probability that GenPPO outperforms PPO is {prob:.3f} (based on {genppo_wins}/{total_comparisons} comparisons).")
    # GenTRPO vs. TRPO
    if "TRPO" in data[env] and "GenTRPO" in data[env]:
      trpo_runs = [np.max(run["rewards"]) for run_group in data[env]["TRPO"] for run in run_group]
      gentrpo_runs = [np.max(run["rewards"]) for run_group in data[env]["GenTRPO"] for run in run_group]
      gentrpo_wins = sum(1 for g in gentrpo_runs if any(g > t for t in trpo_runs))
      total_comparisons = len(gentrpo_runs) * len(trpo_runs)
      prob = gentrpo_wins / total_comparisons if total_comparisons > 0 else 0
      latex_comparison.append(f"The probability that GenTRPO outperforms TRPO is {prob:.3f} (based on {gentrpo_wins}/{total_comparisons} comparisons).")
    # Best Model/Noise Configuration
    max_reward, best_algo, best_noise = -np.inf, "N/A", "N/A"
    for algo in algorithms:
      if algo not in data[env]:
        continue
      for run_group in data[env][algo]:
        avg_reward = np.mean([np.max(run["rewards"]) for run in run_group])
        if avg_reward > max_reward:
          max_reward = avg_reward
          best_algo = algo
          best_noise = run_group[0]["label"]
    latex_comparison.append(f"The best model is {best_algo} with noise level {best_noise}, achieving a maximum reward of {max_reward:.2f}.")

  # Save LaTeX files
  with open(os.path.join(output_dir, "numerical_results.tex"), "w") as f:
    f.write("\n".join(latex_table))
  with open(os.path.join(output_dir, "statistical_significance.tex"), "w") as f:
    f.write("\n".join(latex_significance))
  with open(os.path.join(output_dir, "comparison_statistics.tex"), "w") as f:
    f.write("\n".join(latex_comparison))


# Main execution
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate publication-quality RL plots for MuJoCo.")
  parser.add_argument("--data_dir", type=str, required=True, help="Directory with YAML files (e.g., '.noise/2025-04-19_18-19-45')")
  parser.add_argument("--smooth_window", type=int, default=5, help="Smoothing window size for curves")
  parser.add_argument("--markers_per_line", type=int, default=10, help="Number of markers to show per line (default: 10)")
  args = parser.parse_args()

  # Validate data directory
  if not os.path.exists(args.data_dir):
    print(f"Error: Directory '{args.data_dir}' does not exist.")
    exit(1)

  # Set output directory
  output_dir = os.path.join(".plots", os.path.basename(args.data_dir))
  os.makedirs(output_dir, exist_ok=True)

  # Load data and set black and white style
  sns.set(style="whitegrid", palette="gray")
  data = load_data(args.data_dir)

  # Get environments and algorithms (limit to 7 each)
  environments = sorted(data.keys())[:7]
  algorithms = sorted(set(algo for env in data for algo in data[env]))[:7]

  # Generate comparison plots
  if "PPO" in algorithms and "GenPPO" in algorithms:
    plot_algo_pair_comparison(data, "PPO", "GenPPO", environments, output_dir, args.smooth_window, args.markers_per_line)
  if "GenTRPO" in algorithms and "TRPO" in algorithms:
    plot_algo_pair_comparison(data, "GenTRPO", "TRPO", environments, output_dir, args.smooth_window, args.markers_per_line)

  # Plot for each algorithm: 7 environments, all runs
  for algo in algorithms:
    plot_algo_across_envs(data, algo, environments, output_dir, args.smooth_window, args.markers_per_line)

  # Plot for each environment:  algorithms, each in its own subplot, all runs
  for env in environments:
    plot_env_all_algos(data, env, algorithms, output_dir, args.smooth_window, args.markers_per_line)

  print(f"Plots saved to '{output_dir}' as high-res PNGs.")

  generate_latex_outputs(data, output_dir, algorithms, environments)
  print(f"LaTeX files saved to '{output_dir}'.")
