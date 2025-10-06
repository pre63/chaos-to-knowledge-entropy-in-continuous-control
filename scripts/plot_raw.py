import os

import matplotlib.pyplot as plt
import numpy as np
import yaml


def main(results_dir="results", plots_dir="plots"):
  os.makedirs(plots_dir, exist_ok=True)

  # Collect all data across files
  all_data = {}
  time_data = {}  # For bar plots, group by env

  for filename in os.listdir(results_dir):
    if filename.endswith("_raw.yaml"):
      filepath = os.path.join(results_dir, filename)
      with open(filepath, "r") as f:
        data = yaml.safe_load(f)

      # Extract env_id and variant from filename, e.g., Humanoid-v5_trpo_raw.yaml -> env_id='Humanoid-v5', variant='trpo'
      base_name = filename.replace("_raw.yaml", "")
      parts = base_name.rsplit("_", 1)
      env_id = parts[0]
      variant = parts[1] if len(parts) > 1 else "unknown"

      # Group by noise_type
      groups = {}
      for entry in data:
        noise_type = entry.get("noise_type", "none")
        run_index = entry.get("run_index", 0)
        key = f"{noise_type}_{run_index}"
        groups[key] = entry

      # For each noise_type, collect from 5 runs
      noise_groups = {}
      for key, entry in groups.items():
        noise_type = entry["noise_type"]
        if noise_type not in noise_groups:
          noise_groups[noise_type] = []
        noise_groups[noise_type].append(entry)

      for noise_type, runs in noise_groups.items():
        if len(runs) != 5:
          print(f"Warning: {env_id}_{variant} noise {noise_type} has {len(runs)} runs, skipping.")
          continue

        # Metric keys
        metric_keys = [
          k
          for k in runs[0]
          if k not in ["noise_type", "run_index", "noise_configs", "total_timesteps", "timestamp", "completed", "total_time", "time_per_timestep"]
        ]

        # For line plots
        averages = {}
        stds = {}
        for metric in metric_keys:
          all_lists = []
          for run_data in runs:
            metric_list = run_data.get(metric, [])
            if isinstance(metric_list, list):
              all_lists.append(np.array(metric_list, dtype=float))
            else:
              # Skip if not list
              break
          else:
            if all_lists:
              lengths = [len(lst) for lst in all_lists]
              if len(set(lengths)) == 1 and lengths[0] > 0:
                stack = np.stack(all_lists)
                averages[metric] = np.mean(stack, axis=0)
                stds[metric] = np.std(stack, axis=0)

        # Save for line plots
        plot_key = f"{env_id}_{variant}_{noise_type}"
        all_data[plot_key] = {"averages": averages, "stds": stds}

        # For time bar
        total_times = [run["total_time"] for run in runs if "total_time" in run]
        if len(total_times) == 5:
          avg_total_time = np.mean(total_times)
          std_total_time = np.std(total_times)
          time_per_timesteps = [run["time_per_timestep"] for run in runs if "time_per_timestep" in run]
          avg_time_per_ts = np.mean(time_per_timesteps)
          std_time_per_ts = np.std(time_per_timesteps)

          if env_id not in time_data:
            time_data[env_id] = {}
          time_label = f"{variant}_{noise_type}"
          time_data[env_id][time_label] = {
            "avg_total": avg_total_time,
            "std_total": std_total_time,
            "avg_per_ts": avg_time_per_ts,
            "std_per_ts": std_time_per_ts,
          }

  # Generate line plots
  for plot_key, d in all_data.items():
    averages = d["averages"]
    stds = d["stds"]
    for metric, avg in averages.items():
      fig, ax = plt.subplots(figsize=(20, 10))
      x = np.arange(len(avg))
      ax.plot(x, avg, label="mean")
      ax.fill_between(x, avg - stds[metric], avg + stds[metric], alpha=0.2, label="± std dev")
      ax.set_title(f"{plot_key} - {metric}")
      ax.set_xlabel("Update Step")
      ax.set_ylabel(metric)
      ax.legend()
      ax.grid(True)
      plot_path = os.path.join(plots_dir, f"{plot_key}_{metric}.png")
      plt.savefig(plot_path)
      plt.close(fig)
      print(f"Saved plot to {plot_path}")

  # Generate bar plots for times, per env
  for env_id, times in time_data.items():
    labels = list(times.keys())
    avg_totals = [times[label]["avg_total"] for label in labels]
    std_totals = [times[label]["std_total"] for label in labels]
    avg_per_ts = [times[label]["avg_per_ts"] for label in labels]
    std_per_ts = [times[label]["std_per_ts"] for label in labels]

    # Bar for total_time
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, avg_totals, yerr=std_totals, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Average Total Time (s)")
    ax.set_title(f"{env_id} - Average Total Training Time")
    plot_path = os.path.join(plots_dir, f"{env_id}_total_time_bar.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved bar plot to {plot_path}")

    # Bar for time_per_timestep
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x_pos, avg_per_ts, yerr=std_per_ts, capsize=5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Average Time per Timestep (s)")
    ax.set_title(f"{env_id} - Average Time per Timestep")
    plot_path = os.path.join(plots_dir, f"{env_id}_time_per_ts_bar.png")
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Saved bar plot to {plot_path}")


if __name__ == "__main__":
  main()
