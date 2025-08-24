import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.ndimage import uniform_filter1d


def smooth_data(smoothed_data, window_size=100, pad_mode="edge"):
  """
    Re-smooth the data with a new window size.
    Assumes input is already in smoothed_data format from the YAML.
    """
  re_smoothed_data = []

  window_size = max(1, window_size)
  if window_size % 2 == 0:
    window_size += 1  # Ensure odd for symmetry

  for data in smoothed_data:
    rewards = np.array(data["rewards"])
    entropies = np.array(data["entropies"])

    if len(rewards) <= 1:
      smoothed_rewards = rewards
    else:
      pad_size = window_size // 2
      padded_rewards = np.pad(rewards, (pad_size, pad_size), mode=pad_mode)
      smoothed_rewards = uniform_filter1d(padded_rewards, size=window_size, mode="nearest")[pad_size : pad_size + len(rewards)]

    if len(entropies) <= 1:
      smoothed_entropies = entropies
    else:
      pad_size = window_size // 2
      padded_entropies = np.pad(entropies, (pad_size, pad_size), mode=pad_mode)
      smoothed_entropies = uniform_filter1d(padded_entropies, size=window_size, mode="nearest")[pad_size : pad_size + len(entropies)]

    re_smoothed_data.append(
      {
        "label": data["label"],
        "rewards": smoothed_rewards.tolist(),
        "entropies": smoothed_entropies.tolist(),
        "model": data["model"],
      }
    )

  return re_smoothed_data


def plot_results(results, run_date, model_name, total_timesteps, num_runs):
  """
    Plotting function from the original script, adapted for this YAML structure.
    """
  fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True)
  colors = ["b", "g", "r", "c", "m", "y", "k"]
  markers = ["o", "s", "^", "v", "*", "+", "x"]

  label_to_marker = {}
  label_to_color = {}
  marker_idx = 0
  color_idx = 0

  # Assign markers and colors to unique labels
  for result in results:
    for data in result["smoothed_data"]:
      if data["label"] not in label_to_marker:
        label_to_marker[data["label"]] = markers[marker_idx % len(markers)]
        label_to_color[data["label"]] = colors[color_idx % len(colors)]
        marker_idx += 1
        color_idx = (color_idx + 2) % len(colors)

  # Plot data
  for result in results:
    for data in result["smoothed_data"]:
      label = data["label"]
      marker = label_to_marker[label]
      color = label_to_color[label]

      rewards = data["rewards"]
      if not rewards:
        print(f"Warning: Empty rewards for label '{label}'")
        continue
      x = np.arange(len(rewards))
      mark_every = max(1, len(x) // 10) if len(x) > 0 else 1

      ax1.plot(
        x,
        rewards,
        label=label,
        color=color,
        marker=marker,
        linewidth=2,
        markersize=8,
        markevery=mark_every,
      )

      entropies = data["entropies"]
      if not entropies:
        print(f"Warning: Empty entropies for label '{label}'")
        continue
      entropies_len = len(entropies)
      padded_entropies = np.pad(entropies, (0, len(x) - entropies_len), mode="edge") if entropies_len < len(x) else entropies[: len(x)]
      ax2.plot(
        x,
        padded_entropies,
        label=label,
        color=color,
        marker=marker,
        linewidth=2,
        markersize=8,
        markevery=mark_every,
      )

  ax1.set_title(f"Reward+Action Noise - Rewards (Avg of {num_runs} Runs)")
  ax1.set_ylabel("Mean Reward")
  ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
  ax1.grid(True)

  ax2.set_title(f"Reward+Action Noise - Entropy (Avg of {num_runs} Runs)")
  ax2.set_xlabel("Rollout")
  ax2.set_ylabel("Entropy")
  ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
  ax2.grid(True)

  plt.tight_layout()

  os.makedirs(f".noise/{run_date}", exist_ok=True)
  plot_path = f".noise/{run_date}/{model_name}_{total_timesteps}_reward_action_{num_runs}_runs_resmoothed.png"
  plt.savefig(plot_path)
  plt.close()

  return plot_path


def load_and_replot(yaml_path, new_window_size=100):
  """
    Load the YAML file and replot with new smoothing.
    """
  try:
    with open(yaml_path, "r") as file:
      results = yaml.safe_load(file)
      print(f"Loaded data from {yaml_path}")
  except FileNotFoundError:
    print(f"Error: YAML file not found at {yaml_path}")
    return
  except Exception as e:
    print(f"Error loading YAML file: {e}")
    return

  # Extract metadata from path
  filename = os.path.basename(yaml_path)
  parts = filename.split("_")
  model_name = parts[0]
  total_timesteps = int(parts[1])
  num_runs = int(parts[-2])
  run_date = os.path.basename(os.path.dirname(yaml_path))

  # Handle the malformed YAML structure
  corrected_results = []
  for item in results:
    if isinstance(item, dict) and "noise_type" in item and "smoothed_data" in item:
      # Re-smooth the existing smoothed_data
      re_smoothed_data = smooth_data(item["smoothed_data"], window_size=new_window_size)
      corrected_results.append({"noise_type": item["noise_type"], "smoothed_data": re_smoothed_data})
    else:
      print(f"Skipping malformed entry: {item}")

  if not corrected_results:
    print("Error: No valid data found in YAML file")
    return

  # Generate new plot
  plot_path = plot_results(corrected_results, run_date, model_name, total_timesteps, num_runs)
  print(f"New plot saved at: {plot_path}")


if __name__ == "__main__":
  YPATH = os.getenv("YPATH")
  WND = os.getenv("WND")
  yaml_path = YPATH
  new_window_size = int(WND)

  load_and_replot(yaml_path, new_window_size)
