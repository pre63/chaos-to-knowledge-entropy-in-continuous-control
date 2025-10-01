import csv
import datetime
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import yaml
from scipy.stats import f_oneway, linregress, pearsonr
from sklearn.metrics import roc_auc_score

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def compute_stats(rewards, entropies, ntimesteps=100000, epsilon=1e-5, threshold_pct=0.9, rolling_window=10):
  """
    Compute various statistics from rewards and entropies lists.

    Args:
      rewards (list): List of reward values.
      entropies (list): List of entropy values.
      ntimesteps (int): Total timesteps for normalization (default 100000).
      epsilon (float): Threshold for detecting plateaus.
      threshold_pct (float): Percentage of max_reward for convergence threshold.
      rolling_window (int): Window for rolling std (volatility).

    Returns:
      dict: Dictionary with computed statistics or None if no rewards.
    """
  if not isinstance(rewards, (list, np.ndarray)):
    rewards = [rewards] if rewards is not None else []
  if not isinstance(entropies, (list, np.ndarray)):
    entropies = [entropies] if entropies is not None else []

  if len(rewards) == 0:
    logging.warning("No rewards data provided.")
    return None
  rewards_np = np.array(rewards)
  if rewards_np.ndim == 0:
    rewards_np = np.array([rewards_np])
  min_reward = np.min(rewards_np)
  max_reward = np.max(rewards_np)
  final_reward = np.mean(rewards_np[-10:]) if len(rewards_np) >= 10 else rewards_np[-1]
  max_index = np.argmax(rewards_np)
  timestep_at_max = (max_index / len(rewards_np)) * ntimesteps
  mean_reward = np.mean(rewards_np)
  std_reward = np.std(rewards_np)
  median_reward = np.median(rewards_np)
  iqr_reward = np.percentile(rewards_np, 75) - np.percentile(rewards_np, 25)
  cum_reward = np.sum(rewards_np)
  auc_reward = np.trapezoid(rewards_np, np.linspace(0, ntimesteps, len(rewards_np)))
  norm_auc_reward = auc_reward / (ntimesteps * max_reward) if max_reward != 0 else 0
  cv_reward = std_reward / mean_reward if mean_reward != 0 else np.nan
  volatility = pd.Series(rewards_np).rolling(rolling_window).std().mean()

  # Steps to threshold
  threshold = threshold_pct * max_reward
  steps_to_threshold = np.where(rewards_np >= threshold)[0]
  steps_to_threshold = steps_to_threshold[0] / len(rewards_np) * ntimesteps if steps_to_threshold.size > 0 else ntimesteps

  # Plateau detection (longest segment with small change)
  diffs = np.abs(np.diff(rewards_np))
  plateaus = np.split(diffs, np.where(diffs > epsilon)[0] + 1)
  plateau_length = max(len(p) + 1 for p in plateaus) if len(plateaus) > 0 else 0  # +1 for original segments

  # Linear regression for trend
  timesteps = np.linspace(0, ntimesteps, len(rewards_np))
  slope, intercept, r_value, p_value, std_err = linregress(timesteps, rewards_np)

  # ROC-AUC (illustrative: treating > mean as positive)
  labels = (rewards_np > mean_reward).astype(int)
  roc_auc = roc_auc_score(labels, rewards_np) if len(np.unique(labels)) > 1 else np.nan

  entropy_stats = {}
  entropy_trend = {}
  if len(entropies) > 0:
    entropies_np = np.array(entropies)
    entropy_stats = {
      "Minimum Entropy": np.min(entropies_np),
      "Maximum Entropy": np.max(entropies_np),
      "Mean Entropy": np.mean(entropies_np),
      "Entropy Standard Deviation": np.std(entropies_np),
    }
    # Entropy trend
    e_slope, e_intercept, e_r_value, e_p_value, e_std_err = linregress(timesteps, entropies_np)
    entropy_trend = {
      "Entropy Trend Slope": e_slope,
      "Entropy Trend R Value": e_r_value,
      "Entropy Trend P Value": e_p_value,
      "Entropy Trend Standard Error": e_std_err,
    }

  return {
    "Minimum Reward": min_reward,
    "Maximum Reward": max_reward,
    "Final Reward": final_reward,
    "Timestep at Maximum": timestep_at_max,
    "Mean Reward": mean_reward,
    "Reward Standard Deviation": std_reward,
    "Median Reward": median_reward,
    "Reward Interquartile Range": iqr_reward,
    "Cumulative Reward": cum_reward,
    "Area Under Reward Curve": auc_reward,
    "Normalized Area Under Reward Curve": norm_auc_reward,
    "Coefficient of Variation Reward": cv_reward,
    "Volatility": volatility,
    "Steps to Threshold": steps_to_threshold,
    "Plateau Length": plateau_length,
    "Reward Trend Slope": slope,
    "Reward Trend R Value": r_value,
    "Reward Trend P Value": p_value,
    "Reward Trend Standard Error": std_err,
    "ROC AUC": roc_auc,
    **entropy_stats,
    **entropy_trend,
  }


def process_single_file(path, env_id, cfg):
  """
    Process a single non-raw YAML file and compute stats.

    Args:
      path (str): Path to YAML file.
      env_id (str): Environment ID.
      cfg (str): Variant/config name.

    Returns:
      dict: Stats or None.
    """
  with open(path, "r") as f:
    data = yaml.safe_load(f)
  rewards = data.get("rewards", [])
  entropies = data.get("entropies", [])
  stats = compute_stats(rewards, entropies)
  return stats


def process_raw_file(path, env_id, cfg):
  """
    Process a raw YAML file with multiple runs grouped by entropy level.

    Args:
      path (str): Path to YAML file.
      env_id (str): Environment ID.
      cfg (str): Variant/config name.

    Returns:
      dict: Dict of level: avg_stats, plus overall keys.
    """
  with open(path, "r") as f:
    data = yaml.safe_load(f)
  groups = defaultdict(list)
  for run in data:
    if not run.get("completed", False):
      continue
    entropy_level = None
    for comp in run.get("config", []):
      if comp.get("component") == "reward":
        entropy_level = comp.get("entropy_level")
        break
    if entropy_level is None:
      entropy_level = 0.0  # Default for no noise
    groups[entropy_level].append(run)
  level_stats = {}
  all_means = []
  all_levels = []
  all_entropies = []
  group_means = []  # For ANOVA: list of arrays per group
  for level, group_runs in groups.items():
    level_rewards_means = []
    level_entropies_means = []
    level_group = []  # Per-run means for ANOVA
    for run in group_runs:
      rewards = run.get("rewards", [])
      entropies = run.get("entropies", [])
      if len(rewards) > 0:
        mean_r = np.mean(rewards)
        level_rewards_means.append(mean_r)
        level_group.append(mean_r)
        all_means.append(mean_r)
        all_levels.append(level)
      if len(entropies) > 0:
        mean_e = np.mean(entropies)
        level_entropies_means.append(mean_e)
        all_entropies.append(mean_e)
    if len(level_rewards_means) > 0:
      # Filter runs with rewards/entropies
      reward_runs = [run.get("rewards", []) for run in group_runs if len(run.get("rewards", [])) > 0]
      entropy_runs = [run.get("entropies", []) for run in group_runs if len(run.get("entropies", [])) > 0]
      # To handle variable lengths, truncate to min length
      if reward_runs:
        min_len = min(len(r) for r in reward_runs)
        truncated_rewards = [r[:min_len] for r in reward_runs]
        avg_rewards = np.mean(truncated_rewards, axis=0)
      else:
        avg_rewards = []
      if entropy_runs:
        min_len = min(len(e) for e in entropy_runs)
        truncated_entropies = [e[:min_len] for e in entropy_runs]
        avg_entropies = np.mean(truncated_entropies, axis=0)
      else:
        avg_entropies = []
      stats = compute_stats(avg_rewards, avg_entropies)
      level_stats[level] = stats or {}
      group_means.append(level_group)

  # Extract levels BEFORE adding overall stats
  levels = sorted(list(level_stats.keys()))  # Sort for consistency

  # ANOVA if multiple levels with sufficient data
  if len(group_means) > 1 and all(len(g) > 1 for g in group_means):
    anova_stat, anova_p = f_oneway(*group_means)
    level_stats["Overall ANOVA P Value"] = anova_p

  # Effect size (Cohen's d) for pairwise comparisons if >1 level
  if len(levels) > 1:
    for j in range(1, len(levels)):
      mean1 = level_stats[levels[0]]["Mean Reward"]
      mean2 = level_stats[levels[j]]["Mean Reward"]
      std1 = level_stats[levels[0]]["Reward Standard Deviation"]
      std2 = level_stats[levels[j]]["Reward Standard Deviation"]
      pooled_std = np.sqrt((std1**2 + std2**2) / 2)
      cohen_d = (mean1 - mean2) / pooled_std if pooled_std != 0 else np.nan
      level_stats[f"Cohen's d vs {levels[0]} and {levels[j]}"] = cohen_d

  # Correlation: noise_level vs rewards
  all_levels_num = [float(l) for l in all_levels] if len(all_levels) > 0 else []
  if len(all_levels_num) > 1 and len(all_means) > 1:
    level_r, level_p = pearsonr(all_levels_num, all_means)
    level_stats["Noise vs Reward Correlation Coefficient"] = level_r
    level_stats["Noise vs Reward P Value"] = level_p

  # Correlation: entropy vs rewards (if both present)
  if len(all_entropies) > 1 and len(all_means) > 1 and len(all_entropies) == len(all_means):
    entropy_r, entropy_p = pearsonr(all_entropies, all_means)
    level_stats["Entropy vs Reward Correlation Coefficient"] = entropy_r
    level_stats["Entropy vs Reward P Value"] = entropy_p

  return level_stats


def export_stats_to_files(log_dir, env_id, cfg, stats_data):
  """
    Export statistics to CSV and LaTeX files for a specific env_cfg pair.

    Args:
      log_dir (str): Directory to save files.
      env_id (str): Environment ID.
      cfg (str): Variant/config name.
      stats_data (dict or dict of dict): Stats dict for single, or dict of level:stats for raw.
    """
  if not stats_data:
    logging.warning(f"No stats for {env_id}_{cfg}")
    return

  is_raw = isinstance(stats_data, dict) and any(isinstance(v, dict) for v in stats_data.values())
  if is_raw:
    # Filter to only level: dict entries
    filtered_stats = {k: v for k, v in stats_data.items() if isinstance(v, dict)}
    overall_stats = {k: v for k, v in stats_data.items() if not isinstance(v, dict)}
    if overall_stats:
      logging.info(f"Overall stats for {env_id}_{cfg}: {overall_stats}")
    if not filtered_stats:
      logging.warning(f"No level stats for {env_id}_{cfg}")
      return
    df = pd.DataFrame.from_dict(filtered_stats, orient="index")
    df.index.name = "Noise Level"
    # Pivot the table: stats as rows, noise levels as columns
    pivoted_df = df.transpose()
    pivoted_df.index.name = "Statistic"
  else:
    # For non-raw: single stats dict
    df = pd.DataFrame.from_dict(stats_data, orient="index", columns=["Value"])
    df.index.name = "Statistic"
    pivoted_df = None  # No pivot for non-raw

  base_name = f"{env_id}_{cfg}_stats"
  csv_path = os.path.join(log_dir, f"{base_name}.csv")
  df.to_csv(csv_path)
  logging.info(f"CSV exported to {csv_path}")

  latex_path = os.path.join(log_dir, f"{base_name}.tex")
  with open(latex_path, "w") as f:
    f.write(df.to_latex(escape=True, float_format="%.2f", bold_rows=True))
  logging.info(f"LaTeX table exported to {latex_path}")

  if pivoted_df is not None:
    pivoted_csv_path = os.path.join(log_dir, f"{base_name}.csv")
    pivoted_df.to_csv(pivoted_csv_path)
    logging.info(f"Pivoted CSV exported to {pivoted_csv_path}")

    pivoted_latex_path = os.path.join(log_dir, f"{base_name}.tex")
    with open(pivoted_latex_path, "w") as f:
      f.write(pivoted_df.to_latex(escape=True, float_format="%.2f", bold_rows=True))
    logging.info(f"Pivoted LaTeX table exported to {pivoted_latex_path}")


def export_overall_csv(log_dir, results):
  """
    Export overall aggregated statistics to tables.csv.

    Args:
      log_dir (str): Directory to save file.
      results (dict): Nested dict of env:cfg:stats.
    """
  rows = []
  for env_id in results:
    for cfg in results[env_id]:
      d = results[env_id][cfg]
      row = {"Environment": env_id, "Variant": cfg}
      # For non-raw
      if isinstance(d, dict) and "Maximum Reward" in d:
        row.update(
          {
            "Max Reward": d.get("Maximum Reward", "-"),
            "Mean Reward": d.get("Mean Reward", "-"),
            "Std Reward": d.get("Reward Standard Deviation", "-"),
            "Timestep at Max": d.get("Timestep at Maximum", "-"),
          }
        )
      # For raw, aggregate over levels (e.g., mean of means)
      elif isinstance(d, dict) and any(isinstance(v, dict) for v in d.values()):
        level_means = [v["Mean Reward"] for v in d.values() if isinstance(v, dict)]
        if level_means:
          row.update(
            {
              "Avg Mean Reward": np.mean(level_means),
              "Num Levels": len(level_means),
            }
          )
      rows.append(row)
  if rows:
    df = pd.DataFrame(rows)
    csv_path = os.path.join(log_dir, "tables.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f"Overall CSV exported to {csv_path}")


if __name__ == "__main__":
  # Script version and date for reproducibility
  script_version = "1.0"
  current_date = datetime.date.today().isoformat()
  logging.info(f"Running statistics script version {script_version} on {current_date}")

  log_dir = "results"
  envs = ["Humanoid-v5", "HumanoidStandup-v5"]
  variants = ["trpo", "gentrpo", "gentrpo-ne"]
  raw_variants = ["trpo_raw", "gentrpo_raw"]

  results = {}

  for env_id in envs:
    results[env_id] = {}
    for cfg in variants:
      path = os.path.join(log_dir, f"{env_id}_{cfg}.yaml")
      if os.path.exists(path):
        stats = process_single_file(path, env_id, cfg)
        if stats:
          results[env_id][cfg] = stats
        export_stats_to_files(log_dir, env_id, cfg, stats or {})
    for cfg in raw_variants:
      path = os.path.join(log_dir, f"{env_id}_{cfg}.yaml")
      if os.path.exists(path):
        level_stats = process_raw_file(path, env_id, cfg)
        results[env_id][cfg] = level_stats  # Store for overall CSV
        export_stats_to_files(log_dir, env_id, cfg, level_stats)

  # Export overall tables.csv
  export_overall_csv(log_dir, results)
