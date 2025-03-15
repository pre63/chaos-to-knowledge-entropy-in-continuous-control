import os
import matplotlib.pyplot as plt
import numpy as np
import yaml

def plot_algorithms(all_results, run_date, total_timesteps, num_runs):
    """
    Plot entropy coefficient vs returns for multiple algorithms on a single scatter plot.
    Each entropy coefficient gets its own unique label and marker/color.
    """
    plt.figure(figsize=(12, 8))
    colors = ["b", "g", "r", "c", "m", "y", "k", "orange", "purple", "brown"]
    markers = ["o", "s", "^", "v", "*", "+", "x", "d", "p", "h"]

    label_to_marker = {}
    label_to_color = {}
    marker_idx = 0
    color_idx = 0

    # Assign unique markers and colors to each unique label
    for algo_data in all_results:
        label = algo_data["label"]
        if label not in label_to_marker:
            label_to_marker[label] = markers[marker_idx % len(markers)]
            label_to_color[label] = colors[color_idx % len(colors)]
            marker_idx += 1
            color_idx += 1

    # Plot all data with flipped axes
    for algo_data in all_results:
        label = algo_data["label"]
        marker = label_to_marker[label]
        color = label_to_color[label]

        rewards = np.array(algo_data["rewards"])
        entropies = np.array(algo_data["entropies"])

        if len(rewards) == 0 or len(entropies) == 0:
            print(f"Warning: Empty data for label '{label}'")
            continue

        min_len = min(len(rewards), len(entropies))
        rewards = rewards[:min_len]
        entropies = entropies[:min_len]

        plt.scatter(entropies, rewards, label=label, color=color, marker=marker, s=50)

    # Set x-axis scale to [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]
    plt.xlim(-0.6, 0.6)  # Slightly wider to show all points clearly
    plt.xticks([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5])

    plt.title(f"Returns vs Entropy Coefficient (Avg of {num_runs} Runs) - All Models")
    plt.xlabel("Entropy Coefficient")
    plt.ylabel("Returns")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(f".noise/{run_date}", exist_ok=True)
    plot_path = f".noise/{run_date}/multi_algo_{total_timesteps}_returns_vs_entropy_{num_runs}_runs.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved at: {plot_path}")
    return plot_path

def load_and_plot_yaml_list(yaml_paths):
    """
    Load multiple YAML files and plot each entropy coefficient as a unique model.
    """
    if not yaml_paths:
        print("Error: No YAML paths provided")
        return

    yaml_path_list = yaml_paths.split(";")

    # Convert to absolute paths
    yaml_path_list = [os.path.abspath(path.strip()) for path in yaml_path_list]

    if not yaml_path_list:
        print("Error: Empty YAML path list after splitting")
        return

    all_results = []
    run_date = None
    total_timesteps = None
    num_runs = None

    for yaml_path in yaml_path_list:
        if not os.path.exists(yaml_path):
            print(f"Error: YAML file not found at {yaml_path}")
            continue

        try:
            with open(yaml_path, "r") as file:
                results = yaml.safe_load(file)
                print(f"Loaded data from {yaml_path}")
        except Exception as e:
            print(f"Error loading YAML file {yaml_path}: {e}")
            continue

        filename = os.path.basename(yaml_path)
        parts = filename.split("_")
        algo_name = parts[0]  # e.g., GenPPO, GenTRPO
        total_timesteps = int(parts[1]) if total_timesteps is None else total_timesteps
        num_runs = int(parts[-2]) if num_runs is None else num_runs
        run_date = os.path.basename(os.path.dirname(yaml_path)) if run_date is None else run_date

        for item in results:
            if not (isinstance(item, dict) and "noise_type" in item and "smoothed_data" in item):
                print(f"Skipping malformed entry in {yaml_path}: {item}")
                continue

            noise_type = item["noise_type"]
            smoothed_data_list = item["smoothed_data"]

            if not smoothed_data_list:
                print(f"No smoothed_data found in {yaml_path} for noise_type {noise_type}")
                continue

            if noise_type == "none":
                # Baseline: treat as one entry
                algo_data = smoothed_data_list[0]
                algo_data["label"] = f"{algo_name}_none"
                all_results.append(algo_data)
            else:
                # For noise_type like reward_action, plot each entropy separately
                for i, algo_data in enumerate(smoothed_data_list):
                    # Use the entropy coefficient value as part of the label
                    if "entropies" in algo_data and algo_data["entropies"]:
                        entropy_coeff = algo_data["entropies"][0]  # First entropy value as coefficient
                        algo_data["label"] = f"{algo_name}_{noise_type}_{entropy_coeff:.1f}"
                        all_results.append(algo_data)
                    else:
                        print(f"Skipping entry with no entropies in {yaml_path} for noise_type {noise_type}")

    if not all_results:
        print("Error: No valid data found in any YAML file")
        return

    plot_path = plot_algorithms(all_results, run_date, total_timesteps, num_runs)

if __name__ == "__main__":
    YPATH = os.getenv("YPATH")
    yaml_paths = YPATH if YPATH else ""
    load_and_plot_yaml_list(yaml_paths)