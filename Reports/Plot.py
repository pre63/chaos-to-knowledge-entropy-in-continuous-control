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
from scipy.stats import binomtest, ttest_ind


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
  match = re.match(r".*\(([-+]?[0-9]*\.?[0-9]+)\)", label)
  if match:
    value = match.group(1)
    return f"Noise ({value})"
  return label


# Get global y-axis bounds for a metric across runs
def get_y_bounds(data, metric, keys):
  """
    Compute global min and max for a metric across specified keys.

    Args:
        data: List of (env, algo, label, run_idx, rewards, entropies) tuples.
        metric (str): 'rewards' or 'entropies'.
        keys: List of (env, algo, label) tuples to consider.

    Returns:
        tuple: (y_min, y_max)
    """
  y_min, y_max = np.inf, -np.inf
  for env, algo, label in keys:
    for run in data:
      if run[0] == env and run[1] == algo and run[2] == label:
        values = run[4] if metric == "rewards" else run[5]
        y_min = min(y_min, np.min(values))
        y_max = max(y_max, np.max(values))
  padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.1
  return y_min - padding, y_max + padding


# Load and preprocess data upfront
def load_and_preprocess_data(data_dir, algorithms, environments):
  """
    Load YAML files and preprocess data, computing max rewards, variations, and statistical results.

    Args:
        data_dir (str): Path to directory with YAML files.
        algorithms: List of algorithm names.
        environments: List of environment names.

    Returns:
        tuple: (
            runs: List of (env, algo, label, run_idx, rewards, entropies) tuples,
            table_data: Dict of {algo: {env: (max_reward, std_dev, best_noise)}},
            significance_data: Dict of {algo: {env: {label: p_value}}},
            comparison_data: Dict of {env: (genppo_prob, gentrpo_prob, best_algo, best_noise, max_reward)}
        )
    """
  runs = []
  table_data = {algo: {env: None for env in environments} for algo in algorithms}
  significance_data = {algo: {env: {} for env in environments} for algo in ["GenPPO", "GenTRPO", "TRPOR", "TRPOER"]}
  comparison_data = {env: (1.0, 1.0, "N/A", "N/A", -np.inf) for env in environments}

  # Load and normalize data
  for env_dir in os.listdir(data_dir):
    env_path = os.path.join(data_dir, env_dir)
    if not os.path.isdir(env_path) or env_dir not in environments:
      continue
    env_name = env_dir
    for file in os.listdir(env_path):
      if not file.endswith(".yml") or file.lower() == "config.yaml" or "summary" in file.lower():
        continue
      algo_name = file.split("_")[0]
      if algo_name not in algorithms:
        continue
      file_path = os.path.join(env_path, file)
      with open(file_path, "r") as f:
        try:
          algo_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
          print(f"Error parsing {file_path}: {e}")
          continue
      if not isinstance(algo_data, list):
        continue
      for entry in algo_data:
        if not isinstance(entry, dict) or "smoothed_data" not in entry:
          continue
        smoothed_data = entry.get("smoothed_data", [])
        if not isinstance(smoothed_data, list):
          continue
        for smoothed in smoothed_data:
          if not isinstance(smoothed, dict) or not all(k in smoothed for k in ["label", "rewards", "entropies"]):
            continue
          rewards = normalize_data(smoothed["rewards"], original_steps=1000000)
          entropies = normalize_data(smoothed["entropies"], original_steps=1000000)
          label = smoothed["label"]
          runs.append((env_name, algo_name, label, len(runs), rewards, entropies))

  # Group runs by environment, algorithm, and label
  run_groups = {}
  for env, algo, label, run_idx, rewards, entropies in runs:
    key = (env, algo, label)
    if key not in run_groups:
      run_groups[key] = []
    run_groups[key].append((rewards, entropies))

  # Precompute table and statistical data
  for env in environments:
    for algo in algorithms:
      max_reward, std_dev, best_noise = 0, 0, "N/A"
      best_avg = -np.inf
      if algo in ["PPO", "TRPO"]:
        key = (env, algo, "Baseline")
        if key in run_groups and len(run_groups[key]) >= 5:
          rewards = [np.max(run[0]) for run in run_groups[key]]
          avg_reward = np.mean(rewards)
          max_reward = max(rewards)
          std_dev = np.std(rewards)
          best_noise = "Baseline"
          best_avg = avg_reward
      else:
        for label in set(label for e, a, label in run_groups if e == env and a == algo):
          key = (env, algo, label)
          if key in run_groups and len(run_groups[key]) >= 5:
            rewards = [np.max(run[0]) for run in run_groups[key]]
            avg_reward = np.mean(rewards)
            if avg_reward > best_avg:
              best_avg = avg_reward
              max_reward = max(rewards)
              std_dev = np.std(rewards)
              best_noise = format_label(label)
      if best_avg != -np.inf:
        table_data[algo][env] = (max_reward, std_dev, best_noise)
      else:
        table_data[algo][env] = None

    # Statistical significance
    for algo, baseline in [("GenPPO", "PPO"), ("GenTRPO", "TRPO"), ("TRPOR", "TRPO"), ("TRPOER", "TRPO")]:
      baseline_key = (env, baseline, "Baseline")
      if baseline_key not in run_groups or len(run_groups[baseline_key]) < 5:
        continue
      baseline_runs = [np.max(run[0]) for run in run_groups[baseline_key]]
      for label in set(label for e, a, l in run_groups if e == env and a == algo and l.lower() != "baseline"):
        key = (env, algo, label)
        if key in run_groups and len(run_groups[key]) >= 5:
          algo_runs = [np.max(run[0]) for run in run_groups[key]]
          t_stat, p_value = ttest_ind(algo_runs, baseline_runs, equal_var=False, alternative="greater")
          significance_data[algo][env][format_label(label)] = p_value

    # Comparison statistics
    ppo_runs = [np.max(run[0]) for key in run_groups if key[0] == env and key[1] == "PPO" for run in run_groups[key]]
    genppo_runs = [np.max(run[0]) for key in run_groups if key[0] == env and key[1] == "GenPPO" for run in run_groups[key]]
    trpo_runs = [np.max(run[0]) for key in run_groups if key[0] == env and key[1] == "TRPO" for run in run_groups[key]]
    gentrpo_runs = [np.max(run[0]) for key in run_groups if key[0] == env and key[1] == "GenTRPO" for run in run_groups[key]]
    genppo_prob = 1.0
    if ppo_runs and genppo_runs:
      genppo_wins = sum(1 for g in genppo_runs for p in ppo_runs if g > p)
      total_comparisons = len(genppo_runs) * len(ppo_runs)
      genppo_prob = binomtest(genppo_wins, total_comparisons, p=0.5, alternative="greater").pvalue if total_comparisons > 0 else 1.0
    gentrpo_prob = 1.0
    if trpo_runs and gentrpo_runs:
      gentrpo_wins = sum(1 for g in gentrpo_runs for t in trpo_runs if g > t)
      total_comparisons = len(gentrpo_runs) * len(trpo_runs)
      gentrpo_prob = binomtest(gentrpo_wins, total_comparisons, p=0.5, alternative="greater").pvalue if total_comparisons > 0 else 1.0
    max_reward, best_algo, best_noise = -np.inf, "N/A", "N/A"
    for algo in algorithms:
      for label in set(label for e, a, l in run_groups if e == env and a == algo):
        key = (env, algo, label)
        if key in run_groups and len(run_groups[key]) >= 5:
          rewards = [np.max(run[0]) for run in run_groups[key]]
          avg_reward = np.mean(rewards)
          if avg_reward > max_reward:
            max_reward = avg_reward
            best_algo = algo
            best_noise = format_label(label)
    comparison_data[env] = (genppo_prob, gentrpo_prob, best_algo, best_noise, max_reward)

  return runs, table_data, significance_data, comparison_data


# Plot comparison of two algorithms across shared environments, stacked vertically
def plot_algo_pair_comparison(data, algo1, algo2, environments, output_dir, smooth_window=5, markers_per_line=10):
  """
    Plot rewards for two algorithms across shared environments in a 2xN grid (algo1 on top, algo2 below),
    with shared linear y-axis scale per environment, x-axis with decimal ticks and scientific notation note
    for 1M steps, frequent lower y-ticks, tiny fonts, increased offset markers, y-label only on first subplot
    of each row, and single legend at the bottom.

    Args:
        data: List of (env, algo, label, run_idx, rewards, entropies) tuples.
        algo1, algo2 (str): Algorithm names to compare.
        environments: List of environment names.
        output_dir (str): Where to save the plot.
        smooth_window (int): Smoothing window size (default: 5).
        markers_per_line (int): Number of markers to show per line (default: 10).
    """
  valid_envs = sorted([env for env in environments if any(r[0] == env and r[1] == algo1 for r in data) and any(r[0] == env and r[1] == algo2 for r in data)])
  if not valid_envs:
    print(f"No shared environments for {algo1} and {algo2}.")
    return
  ncols = len(valid_envs)
  fig, axes = plt.subplots(2, ncols, figsize=(3 * ncols, 6), squeeze=False)
  markers = ["o", "s", "^", "v", "*", "+", "x"]
  handles, labels = [], []
  marker_interval = max(1, 100 // markers_per_line)
  for i, env in enumerate(valid_envs):
    keys = [(run[0], run[1], run[2]) for run in data if run[0] == env and run[1] in [algo1, algo2]]
    y_min, y_max = get_y_bounds(data, "rewards", keys)
    for row, algo in enumerate([algo1, algo2]):
      ax = axes[row, i]
      run_idx = 0
      labels_seen = set()
      for run in data:
        if run[0] != env or run[1] != algo:
          continue
        label = run[2]
        rewards = run[4]
        if smooth_window > 1:
          rewards = uniform_filter1d(rewards, size=smooth_window, mode="nearest")
        steps = np.linspace(0, 1000000, len(rewards))
        offset = run_idx * marker_interval // max(1, sum(1 for r in data if r[0] == env and r[1] == algo))
        linestyle = "-" if row == 0 else "--"
        (line,) = ax.plot(
          steps,
          rewards,
          label=f"{format_label(label)}" if format_label(label) not in labels_seen else None,
          color="black",
          linestyle=linestyle,
          marker=markers[run_idx % len(markers)],
          markevery=(offset, marker_interval),
          linewidth=2,
          alpha=0.7,
          markersize=6,
        )
        if format_label(label) not in labels_seen and i == 0 and row == 0:
          handles.append(line)
          labels.append(f"{format_label(label)}")
          labels_seen.add(format_label(label))
        run_idx += 1
      ax.set_title(f"{env} ({algo})", fontsize=10)
      if i == 0:
        ax.set_ylabel("Rewards", fontsize=6)
      if row == 1:
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
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=len(labels), fontsize=8)
  plt.tight_layout(rect=[0, 0.05, 1, 0.95])
  os.makedirs(output_dir, exist_ok=True)
  fig.savefig(os.path.join(output_dir, f"{algo1}_vs_{algo2}_rewards.png"), dpi=300, bbox_inches="tight")
  plt.close(fig)


# Plot all algorithms for one environment, each in its own subplot
def plot_env_all_algos(data, env, algorithms, output_dir, smooth_window=5, markers_per_line=10):
  """
    Plot all runs for each algorithm in one environment in a 2x7 tiled grid with returns (top row)
    and entropy (bottom row), with linear y-axis scale (shared for rewards, per-algorithm for entropy),
    x-axis with decimal ticks and scientific notation note for 1M steps, frequent lower y-ticks, tiny
    fonts, increased offset markers, y-label only on first cell of each row, and single legend at the bottom.

    Args:
        data: List of (env, algo, label, run_idx, rewards, entropies) tuples.
        env (str): Environment name.
        algorithms: List of algorithm names (expected up to 7).
        output_dir (str): Where to save the plot.
        smooth_window (int): Smoothing window size (default: 5).
        markers_per_line (int): Number of markers to show per line (default: 10).
    """
  valid_algos = sorted([algo for algo in algorithms if any(run[0] == env and run[1] == algo for run in data)])
  if not valid_algos:
    print(f"No algorithms for {env}.")
    return
  nrows, ncols = 2, 7
  fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6), squeeze=False)
  markers = ["o", "s", "^", "v", "*", "+", "x"]
  keys = [(env, algo, label) for run in data if run[0] == env and run[1] in valid_algos for algo, label in [(run[1], run[2])]]
  rewards_y_min, rewards_y_max = get_y_bounds(data, "rewards", keys)
  marker_interval = max(1, 100 // markers_per_line)
  handles, labels = [], []
  for i in range(7):
    ax_returns = axes[0, i]
    ax_entropy = axes[1, i]
    if i < len(valid_algos):
      algo = valid_algos[i]
      keys = [(env, algo, label) for run in data if run[0] == env and run[1] == algo for label in [run[2]]]
      entropies_y_min, entropies_y_max = get_y_bounds(data, "entropies", keys)
      run_idx = 0
      labels_seen = set()
      for run in data:
        if run[0] != env or run[1] != algo:
          continue
        rewards = run[4]
        entropies = run[5]
        label = run[2]
        if smooth_window > 1:
          rewards = uniform_filter1d(rewards, size=smooth_window, mode="nearest")
          entropies = uniform_filter1d(entropies, size=smooth_window, mode="nearest")
        steps = np.linspace(0, 1000000, len(rewards))
        offset = run_idx * marker_interval // max(1, sum(1 for r in data if r[0] == env and r[1] == algo))
        (line,) = ax_returns.plot(
          steps,
          rewards,
          label=f"{format_label(label)}" if format_label(label) not in labels_seen else None,
          color="black",
          linestyle="-",
          marker=markers[run_idx % len(markers)],
          markevery=(offset, marker_interval),
          linewidth=2,
          alpha=0.7,
          markersize=6,
        )
        ax_entropy.plot(
          steps,
          entropies,
          label=f"{format_label(label)}" if format_label(label) not in labels_seen else None,
          color="black",
          linestyle="-",
          marker=markers[run_idx % len(markers)],
          markevery=(offset, marker_interval),
          linewidth=2,
          alpha=0.7,
          markersize=6,
        )
        if format_label(label) not in labels_seen and i == 0:
          handles.append(line)
          labels.append(f"{format_label(label)}")
          labels_seen.add(format_label(label))
        run_idx += 1
      ax_returns.set_title(f"{algo}", fontsize=10)
      if i == 0:
        ax_returns.set_ylabel("Rewards", fontsize=6)
        ax_entropy.set_ylabel("Entropy", fontsize=6)
      ax_entropy.set_xlabel("Steps", fontsize=6)
      ax_returns.grid(True, linestyle="--", alpha=0.7)
      ax_entropy.grid(True, linestyle="--", alpha=0.7)
      rewards_range = rewards_y_max - rewards_y_min
      if rewards_range > 0:
        y_ticks = np.concatenate(
          [
            np.arange(rewards_y_min, rewards_y_min + rewards_range / 4, rewards_range / 20),
            np.arange(rewards_y_min + rewards_range / 4, rewards_y_max, rewards_range / 10),
          ]
        )
        y_ticks = np.unique(np.clip(y_ticks, rewards_y_min, rewards_y_max))
        ax_returns.set_yticks(y_ticks, labels=[f"{tick:.2f}" for tick in y_ticks])
      entropies_range = entropies_y_max - entropies_y_min
      if entropies_range > 0:
        y_ticks = np.concatenate(
          [
            np.arange(entropies_y_min, entropies_y_min + entropies_range / 4, entropies_range / 20),
            np.arange(entropies_y_min + entropies_range / 4, entropies_y_max, entropies_range / 10),
          ]
        )
        y_ticks = np.unique(np.clip(y_ticks, entropies_y_min, entropies_y_max))
        ax_entropy.set_yticks(y_ticks, labels=[f"{tick:.2f}" for tick in y_ticks])
      ax_returns.set_ylim(rewards_y_min, rewards_y_max)
      ax_entropy.set_ylim(entropies_y_min, entropies_y_max)
      x_ticks = [0, 200000, 400000, 600000, 800000, 1000000]
      x_tick_labels = [f"{x // 100000}" for x in x_ticks]
      ax_returns.set_xticks(x_ticks, labels=x_tick_labels)
      ax_entropy.set_xticks(x_ticks, labels=x_tick_labels)
      ax_returns.tick_params(axis="both", labelsize=5)
      ax_entropy.tick_params(axis="both", labelsize=5)
      ax_returns.set_xlim(0, 1000000)
      ax_entropy.set_xlim(0, 1000000)
    else:
      ax_returns.set_visible(False)
      ax_entropy.set_visible(False)
  fig.text(0.5, 0.01, r"Steps (\(\times 10^6\))", ha="center", fontsize=6)
  for ax in axes.flatten():
    if ax.get_legend() is not None:
      ax.get_legend().remove()
  if handles:
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=len(labels), fontsize=8)
  plt.tight_layout(rect=[0, 0.05, 1, 0.95])
  os.makedirs(output_dir, exist_ok=True)
  fig.savefig(os.path.join(output_dir, f"{env}_rewards_entropy_vs_steps_all_algos.png"), dpi=300, bbox_inches="tight")
  plt.close(fig)


# Plot returns and entropy vs steps for one algorithm across all environments
def plot_algo_across_envs(data, algo, environments, output_dir, smooth_window=5, markers_per_line=10):
  """
    Plot all runs for one algorithm across 7 environments in a 2x7 tiled grid with returns (top row)
    and entropy (bottom row), with linear y-axis scale (per-environment for both rewards and entropy),
    x-axis with decimal ticks and scientific notation note for 1M steps, frequent lower y-ticks, tiny
    fonts, increased offset markers, y-label only on first cell of each row, and single legend at the bottom.

    Args:
        data: List of (env, algo, label, run_idx, rewards, entropies) tuples.
        algo (str): Algorithm name.
        environments: List of environment names (expected up to 7).
        output_dir (str): Where to save the plot.
        smooth_window (int): Smoothing window size (default: 5).
        markers_per_line (int): Number of markers to show per line (default: 10).
    """
  valid_envs = sorted([env for env in environments if any(run[0] == env and run[1] == algo for run in data)])
  if not valid_envs:
    print(f"No data for {algo} in any environments.")
    return
  nrows, ncols = 2, 7
  fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6), squeeze=False)
  markers = ["o", "s", "^", "v", "*", "+", "x"]
  handles, labels = [], []
  marker_interval = max(1, 100 // markers_per_line)
  for i in range(7):
    ax_returns = axes[0, i]
    ax_entropy = axes[1, i]
    if i < len(valid_envs):
      env = valid_envs[i]
      keys = [(env, algo, run[2]) for run in data if run[0] == env and run[1] == algo]
      rewards_y_min, rewards_y_max = get_y_bounds(data, "rewards", keys)
      entropies_y_min, entropies_y_max = get_y_bounds(data, "entropies", keys)
      run_idx = 0
      labels_seen = set()
      for run in data:
        if run[0] != env or run[1] != algo:
          continue
        rewards = run[4]
        entropies = run[5]
        label = run[2]
        if smooth_window > 1:
          rewards = uniform_filter1d(rewards, size=smooth_window, mode="nearest")
          entropies = uniform_filter1d(entropies, size=smooth_window, mode="nearest")
        steps = np.linspace(0, 1000000, len(rewards))
        offset = run_idx * marker_interval // max(1, sum(1 for r in data if r[0] == env and r[1] == algo))
        (line,) = ax_returns.plot(
          steps,
          rewards,
          label=f"{format_label(label)}" if format_label(label) not in labels_seen else None,
          color="black",
          linestyle="-",
          marker=markers[run_idx % len(markers)],
          markevery=(offset, marker_interval),
          linewidth=2,
          alpha=0.7,
          markersize=6,
        )
        ax_entropy.plot(
          steps,
          entropies,
          label=f"{format_label(label)}" if format_label(label) not in labels_seen else None,
          color="black",
          linestyle="-",
          marker=markers[run_idx % len(markers)],
          markevery=(offset, marker_interval),
          linewidth=2,
          alpha=0.7,
          markersize=6,
        )
        if format_label(label) not in labels_seen and i == 0:
          handles.append(line)
          labels.append(f"{format_label(label)}")
          labels_seen.add(format_label(label))
        run_idx += 1
      ax_returns.set_title(f"{env}", fontsize=10)
      if i == 0:
        ax_returns.set_ylabel("Rewards", fontsize=6)
        ax_entropy.set_ylabel("Entropy", fontsize=6)
      ax_entropy.set_xlabel("Steps", fontsize=6)
      ax_returns.grid(True, linestyle="--", alpha=0.7)
      ax_entropy.grid(True, linestyle="--", alpha=0.7)
      rewards_range = rewards_y_max - rewards_y_min
      if rewards_range > 0:
        y_ticks = np.concatenate(
          [
            np.arange(rewards_y_min, rewards_y_min + rewards_range / 4, rewards_range / 20),
            np.arange(rewards_y_min + rewards_range / 4, rewards_y_max, rewards_range / 10),
          ]
        )
        y_ticks = np.unique(np.clip(y_ticks, rewards_y_min, rewards_y_max))
        ax_returns.set_yticks(y_ticks, labels=[f"{tick:.2f}" for tick in y_ticks])
      entropies_range = entropies_y_max - entropies_y_min
      if entropies_range > 0:
        y_ticks = np.concatenate(
          [
            np.arange(entropies_y_min, entropies_y_min + entropies_range / 4, entropies_range / 20),
            np.arange(entropies_y_min + entropies_range / 4, entropies_y_max, entropies_range / 10),
          ]
        )
        y_ticks = np.unique(np.clip(y_ticks, entropies_y_min, entropies_y_max))
        ax_entropy.set_yticks(y_ticks, labels=[f"{tick:.2f}" for tick in y_ticks])
      ax_returns.set_ylim(rewards_y_min, rewards_y_max)
      ax_entropy.set_ylim(entropies_y_min, entropies_y_max)
      x_ticks = [0, 200000, 400000, 600000, 800000, 1000000]
      x_tick_labels = [f"{x // 100000}" for x in x_ticks]
      ax_returns.set_xticks(x_ticks, labels=x_tick_labels)
      ax_entropy.set_xticks(x_ticks, labels=x_tick_labels)
      ax_returns.tick_params(axis="both", labelsize=5)
      ax_entropy.tick_params(axis="both", labelsize=5)
      ax_returns.set_xlim(0, 1000000)
      ax_entropy.set_xlim(0, 1000000)
    else:
      ax_returns.set_visible(False)
      ax_entropy.set_visible(False)
  fig.text(0.5, 0.01, r"Steps (\(\times 10^6\))", ha="center", fontsize=6)
  for ax in axes.flatten():
    if ax.get_legend() is not None:
      ax.get_legend().remove()
  if handles:
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=len(labels), fontsize=8)
  plt.tight_layout(rect=[0, 0.05, 1, 0.95])
  os.makedirs(output_dir, exist_ok=True)
  fig.savefig(os.path.join(output_dir, f"{algo}_rewards_entropy_vs_steps.png"), dpi=300, bbox_inches="tight")
  plt.close(fig)


# Generate LaTeX table and statistical sections
def generate_latex_outputs(table_data, significance_data, comparison_data, output_dir, algorithms, environments):
  """
    Generate a single LaTeX file with numerical results table, statistical significance,
    and comparison statistics sections using precomputed data.

    Args:
        table_data: Dict of {algo: {env: (max_reward, std_dev, best_noise)}}.
        significance_data: Dict of {algo: {env: {label: p_value}}}.
        comparison_data: Dict of {env: (genppo_prob, gentrpo_prob, best_algo, best_noise, max_reward)}.
        output_dir (str): Where to save the LaTeX file.
        algorithms: List of algorithm names.
        environments: List of environment names.
    """
  os.makedirs(output_dir, exist_ok=True)
  latex_content = []

  # LaTeX Table: Max reward, variation, best noise level
  latex_content.append("\\begin{table}[h]")
  latex_content.append("\\centering")
  latex_content.append("\\caption{Maximum Rewards, Variation, and Best Noise Level for Each Algorithm and Environment}")
  latex_content.append("\\label{tab:numerical_results}")
  latex_content.append("\\begin{tabular}{|l|" + "c|" * len(environments) + "}")
  latex_content.append("\\hline")
  latex_content.append("Algorithm & " + " & ".join([env.replace("-v5", "") for env in environments]) + " \\\\ \\hline")

  for algo in algorithms:
    row = [algo]
    for env in environments:
      if table_data[algo][env] is None:
        row.append("-")
      else:
        max_reward, std_dev, best_noise = table_data[algo][env]
        row.append(f"{max_reward:.2f} $\\pm$ {std_dev:.2f} ({best_noise})")
    latex_content.append(" & ".join(row) + " \\\\ \\hline")

  latex_content.append("\\end{tabular}")
  latex_content.append("\\end{table}")

  # LaTeX Section: Statistical Significance vs. Baseline
  latex_content.append("\\section{Statistical Significance Against Baseline}")
  latex_content.append(
    "This section evaluates the statistical significance of each algorithm’s performance compared to its baseline (PPO for GenPPO, TRPO for GenTRPO, TRPOR, TRPOER) using a two-sample t-test. The p-value indicates the likelihood that a noise-trained run outperforms the baseline."
  )

  for algo, baseline in [("GenPPO", "PPO"), ("GenTRPO", "TRPO"), ("TRPOR", "TRPO"), ("TRPOER", "TRPO")]:
    latex_content.append(f"\\subsection{{{algo} vs. {baseline}}}")
    for env in environments:
      if not significance_data[algo][env]:
        continue
      latex_content.append(f"\\paragraph{{{env.replace('-v5', '')}}}")
      for label, p_value in significance_data[algo][env].items():
        latex_content.append(f"For noise level {label}, the p-value is {p_value:.3f}, indicating the likelihood that {algo} outperforms {baseline}.")

  # LaTeX Section: Comparison Statistics (GenPPO vs. PPO, GenTRPO vs. TRPO, Best Model)
  latex_content.append("\\section{Comparison Statistics}")
  latex_content.append(
    "This section analyzes the probability that GenPPO outperforms PPO and GenTRPO outperforms TRPO based on maximum rewards across all runs, using a binomial test. It also identifies the model and noise configuration most likely to achieve the highest reward per environment."
  )

  for env in environments:
    if comparison_data[env][2] == "N/A":
      continue
    latex_content.append(f"\\subsection{{{env.replace('-v5', '')}}}")
    genppo_prob, gentrpo_prob, best_algo, best_noise, max_reward = comparison_data[env]
    if genppo_prob < 1.0:
      latex_content.append(f"The probability that GenPPO outperforms PPO is {genppo_prob:.3f}.")
    if gentrpo_prob < 1.0:
      latex_content.append(f"The probability that GenTRPO outperforms TRPO is {gentrpo_prob:.3f}.")
    latex_content.append(f"The best model is {best_algo} with noise level {best_noise}, achieving an average maximum reward of {max_reward:.2f}.")

  # Save all LaTeX content to a single file
  with open(os.path.join(output_dir, "results.tex"), "w") as f:
    f.write("\n".join(latex_content))


# Main execution
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Generate publication-quality RL plots and LaTeX results for MuJoCo.")
  parser.add_argument("--data_dir", type=str, required=True, help="Directory with YAML files (e.g., '.noise/2025-04-19_18-19-45')")
  parser.add_argument("--smooth_window", type=int, default=5, help="Smoothing window size for curves")
  parser.add_argument("--markers_per_line", type=int, default=10, help="Number of markers to show per line (default: 10)")
  args = parser.parse_args()

  if not os.path.exists(args.data_dir):
    print(f"Error: Directory '{args.data_dir}' does not exist.")
    exit(1)

  output_dir = os.path.join(".plots", os.path.basename(args.data_dir))
  os.makedirs(output_dir, exist_ok=True)

  sns.set(style="whitegrid", palette="gray")

  environments = ["HalfCheetah-v5", "Hopper-v5", "Humanoid-v5", "HumanoidStandup-v5", "Pusher-v5", "Reacher-v5", "Swimmer-v5"]
  algorithms = ["PPO", "GenPPO", "TRPO", "GenTRPO", "TRPOR", "TRPOER"]

  # Load and preprocess data upfront
  runs, table_data, significance_data, comparison_data = load_and_preprocess_data(args.data_dir, algorithms, environments)

  # Generate comparison plots
  if any(run[1] == "PPO" for run in runs) and any(run[1] == "GenPPO" for run in runs):
    plot_algo_pair_comparison(runs, "PPO", "GenPPO", environments, output_dir, args.smooth_window, args.markers_per_line)
  if any(run[1] == "GenTRPO" for run in runs) and any(run[1] == "TRPO" for run in runs):
    plot_algo_pair_comparison(runs, "GenTRPO", "TRPO", environments, output_dir, args.smooth_window, args.markers_per_line)

  for algo in algorithms:
    plot_algo_across_envs(runs, algo, environments, output_dir, args.smooth_window, args.markers_per_line)

  for env in environments:
    plot_env_all_algos(runs, env, algorithms, output_dir, args.smooth_window, args.markers_per_line)

  generate_latex_outputs(table_data, significance_data, comparison_data, output_dir, algorithms, environments)

  print(f"Plots and LaTeX file saved to '{output_dir}'.")
