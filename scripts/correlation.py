import os

import numpy as np
import pandas as pd
import yaml
from scipy.stats import f_oneway, kruskal, levene, pearsonr, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd


# Data loading function (updated to optionally load individual records instead of means)
def load_data(env_id, raw_cfg, use_individual_records=False, log_dir="results"):
  path = os.path.join(log_dir, f"{env_id}_{raw_cfg}.yaml")
  if not os.path.exists(path):
    print(f"File not found: {path}")
    return None
  with open(path, "r") as f:
    data = yaml.safe_load(f)
  data_list = []
  for run in data:
    if not run.get("completed", False):
      continue
    entropy_level = None
    for comp in run.get("config", []):
      if comp.get("component") == "reward":
        entropy_level = comp.get("entropy_level")
        break
    if entropy_level is None:
      entropy_level = 0.0
    rewards = run.get("rewards", [])
    entropies = run.get("entropies", [])
    if use_individual_records:
      # Use individual values, assuming rewards and entropies are paired (same length)
      if len(rewards) != len(entropies):
        print(f"Warning: Rewards and entropies lengths differ in run for {key}. Skipping pairing.")
        continue
      for r, e in zip(rewards, entropies):
        data_list.append({"noise_level": entropy_level, "reward": r, "entropy": e})
    else:
      # Original: means per run
      mean_reward = np.mean(rewards) if len(rewards) > 0 else np.nan
      mean_entropy = np.mean(entropies) if len(entropies) > 0 else np.nan
      data_list.append({"noise_level": entropy_level, "reward": mean_reward, "entropy": mean_entropy})
  df = pd.DataFrame(data_list)
  return df


# Analysis function focusing on ANOVA (with fallback) and correlation, per environment/config
def analyze_anova_and_correlation(use_individual_records=False):
  envs = ["Humanoid-v5", "HumanoidStandup-v5"]
  cfgs = ["trpo_raw", "gentrpo_raw"]
  results = {}

  for env_id in envs:
    for raw_cfg in cfgs:
      key = f"{env_id}_{raw_cfg}"
      df = load_data(env_id, raw_cfg, use_individual_records=use_individual_records)
      if df is None or df.empty:
        print(f"No data for {key}. Skipping.")
        continue

      # Drop NaNs for relevant columns
      df_rew_ent = df.dropna(subset=["reward", "entropy"])
      df_noise_reward = df.dropna(subset=["reward"])

      # Correlation: Rewards vs Entropy
      if len(df_rew_ent) < 2:
        corr_interpretation = "Insufficient data for correlation in {key}."
        corr_details = ""
      else:
        corr, p_value_corr = pearsonr(df_rew_ent["reward"], df_rew_ent["entropy"])
        corr_interpretation = f"Correlation between rewards and entropy: {corr:.2f} (p={p_value_corr:.2e})"
        if abs(corr) < 0.3:
          corr_type = "no or very weak"
        elif corr > 0:
          corr_type = "positive"
        else:
          corr_type = "negative"
        corr_details = f"This indicates a {corr_type} correlation. P-value suggests it is {'statistically significant' if p_value_corr < 0.05 else 'not statistically significant'}."

      # ANOVA/Kruskal-Wallis: Effect of noise level on rewards
      unique_noises = df_noise_reward["noise_level"].unique()
      reward_groups = [df_noise_reward[df_noise_reward["noise_level"] == n]["reward"] for n in unique_noises]
      reward_groups = [g for g in reward_groups if len(g) > 0]  # Filter empty groups

      if len(reward_groups) < 2:
        anova_interpretation = f"Insufficient groups (noise levels) for statistical test in {key}."
        post_hoc = None
      else:
        # Check assumptions for ANOVA
        normality_p = [shapiro(g)[1] for g in reward_groups if len(g) >= 3]
        levene_stat, levene_p = levene(*reward_groups) if all(len(g) > 0 for g in reward_groups) else (np.nan, np.nan)

        use_parametric = all(p > 0.05 for p in normality_p) and (levene_p > 0.05 if not np.isnan(levene_p) else False)

        # Compute test
        if use_parametric:
          stat, p_value_test = f_oneway(*reward_groups)
          test_name = "One-Way ANOVA"
        else:
          stat, p_value_test = kruskal(*reward_groups)
          test_name = "Kruskal-Wallis"

        significance = "statistically significant (p < 0.05)" if p_value_test < 0.05 else "not statistically significant (p >= 0.05)"
        anova_interpretation = f"{test_name} for noise level effect on rewards: stat={stat:.2f}, p={p_value_test:.2e}. The impact is {significance}."

        # Post-hoc if significant
        post_hoc = None
        if p_value_test < 0.05:
          if use_parametric:
            tukey = pairwise_tukeyhsd(df_noise_reward["reward"], df_noise_reward["noise_level"])
            post_hoc = tukey.summary()
          else:
            # For Kruskal-Wallis, post-hoc could use Dunn's test, but not implemented here; note it
            post_hoc = "Post-hoc not performed for non-parametric test. Consider manual pairwise comparisons."

      # Store results per environment/config
      results[key] = {
        "correlation": {"interpretation": corr_interpretation, "details": corr_details},
        "stat_test": {"interpretation": anova_interpretation, "post_hoc": post_hoc},
      }

  return results


if __name__ == "__main__":
  # Set to True to use individual records (for thousands of data points)
  use_individual = True  # Change to False if you want means per run
  analysis_results = analyze_anova_and_correlation(use_individual_records=use_individual)
  for key, res in analysis_results.items():
    print(f"\nBreakdown for {key}:")
    print("Correlation (Rewards vs Entropy):")
    print(res["correlation"]["interpretation"])
    print(res["correlation"]["details"])
    print("Statistical Test (Noise Level vs Rewards):")
    print(res["stat_test"]["interpretation"])
    if res["stat_test"]["post_hoc"] is not None:
      print("Post-Hoc Analysis:")
      print(res["stat_test"]["post_hoc"])
