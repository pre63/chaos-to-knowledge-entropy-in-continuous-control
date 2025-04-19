import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import re
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from matplotlib.ticker import FixedFormatter, FixedLocator

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
    interpolator = interp1d(original_x, values, kind='linear', fill_value="extrapolate")
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
            if not file.endswith('.yml') or file.lower() == "config.yaml" or "summary" in file.lower():
                continue
            algo_name = file.split('_')[0]
            file_path = os.path.join(env_path, file)
            with open(file_path, 'r') as f:
                try:
                    algo_data = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    print(f"Error parsing {file_path}: {e}")
                    continue
            if not isinstance(algo_data, list):
                print(f"Warning: {file_path} does not contain a list. Skipping.")
                continue
            for entry in algo_data:
                if not isinstance(entry, dict) or 'smoothed_data' not in entry:
                    print(f"Warning: Invalid entry in {file_path}. Skipping.")
                    continue
                smoothed_data = entry.get('smoothed_data', [])
                if not isinstance(smoothed_data, list):
                    print(f"Warning: smoothed_data in {file_path} is not a list. Skipping.")
                    continue
                for smoothed in smoothed_data:
                    if not isinstance(smoothed, dict) or not all(k in smoothed for k in ['label', 'rewards', 'entropies']):
                        print(f"Warning: Missing keys in smoothed data of {file_path}. Skipping.")
                        continue
                    # Normalize rewards and entropies to 100 points
                    rewards = normalize_data(smoothed['rewards'], original_steps=1000000)
                    entropies = normalize_data(smoothed['entropies'], original_steps=1000000)
                    if algo_name not in data[env_name]:
                        data[env_name][algo_name] = []
                    data[env_name][algo_name].append({
                        'label': smoothed['label'],
                        'rewards': rewards,
                        'entropies': entropies
                    })
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
    match = re.match(r'.*\(([-+]?[0-9]*\.?[0-9]+)\)', label)
    if match:
        value = match.group(1)
        return f"Noise ({value})"
    return label  # Fallback to original label if parsing fails

# Plot returns and entropy vs steps for one algorithm across all environments
def plot_algo_across_envs(data, algo, environments, output_dir, smooth_window=5, markers_per_line=10):
    """
    Plot all runs for one algorithm across 6 environments in a 2x6 tiled grid with returns (top row)
    and entropy (bottom row), with linear y-axis scale (per-environment for both rewards and entropy),
    x-axis with decimal ticks and scientific notation note for 1M steps, frequent lower y-ticks, tiny
    fonts, increased offset markers, y-label only on first cell of each row, and single legend at the bottom.
    
    Args:
        data: Loaded data.
        algo (str): Algorithm name.
        environments: List of environment names (expected up to 6).
        output_dir (str): Where to save the plot.
        smooth_window (int): Smoothing window size (default: 5).
        markers_per_line (int): Number of markers to show per line (default: 10).
    """
    valid_envs = sorted([env for env in environments if env in data and algo in data[env]])
    if not valid_envs:
        print(f"No data for {algo} in any environments.")
        return
    # Fixed 2x6 grid for 12 square tiles
    nrows, ncols = 2, 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6), squeeze=False)
    # Define markers
    markers = ["o", "s", "^", "v", "*", "+", "x"]  # Markers for distinction
    # Get y-axis bounds for returns and entropy per environment
    handles, labels = [], []
    marker_interval = max(1, 100 // markers_per_line)  # Interval between markers
    for i in range(6):
        # Top row: Returns
        ax_returns = axes[0, i]
        # Bottom row: Entropy
        ax_entropy = axes[1, i]
        if i < len(valid_envs):
            env = valid_envs[i]
            runs = data[env][algo]
            # Compute y-axis bounds for this environment
            keys = [(env, algo)]
            rewards_y_min, rewards_y_max = get_y_bounds(data, 'rewards', keys)
            entropies_y_min, entropies_y_max = get_y_bounds(data, 'entropies', keys)
            for j, run in enumerate(runs):
                rewards = run['rewards']
                entropies = run['entropies']
                if smooth_window > 1:
                    rewards = uniform_filter1d(rewards, size=smooth_window, mode='nearest')
                    entropies = uniform_filter1d(entropies, size=smooth_window, mode='nearest')
                steps = np.linspace(0, 1000000, len(rewards))  # 1M steps
                # Use increased offset for markers to prevent overlap
                offset = j * marker_interval // max(1, len(runs))
                line, = ax_returns.plot(steps, rewards, label=format_label(run['label']), color='black', linestyle='-', marker=markers[j % len(markers)], markevery=(offset, marker_interval), linewidth=2, alpha=0.7, markersize=6)
                ax_entropy.plot(steps, entropies, label=format_label(run['label']), color='black', linestyle='-', marker=markers[j % len(markers)], markevery=(offset, marker_interval), linewidth=2, alpha=0.7, markersize=6)
                # Collect handles and labels from the first returns subplot
                if i == 0:
                    handles.append(line)
                    labels.append(format_label(run['label']))
            ax_returns.set_title(f"{algo} - {env}", fontsize=10)
            # Set y-label only on first cell of each row
            if i == 0:
                ax_returns.set_ylabel("Rewards", fontsize=6)
                ax_entropy.set_ylabel("Entropy", fontsize=6)
            ax_entropy.set_xlabel("Steps", fontsize=6)
            ax_returns.grid(True, linestyle='--', alpha=0.7)
            ax_entropy.grid(True, linestyle='--', alpha=0.7)
            # Set linear y-axis with frequent lower ticks
            rewards_range = rewards_y_max - rewards_y_min
            if rewards_range > 0:
                y_ticks = np.concatenate([
                    np.arange(rewards_y_min, rewards_y_min + rewards_range/4, rewards_range/20),  # Frequent lower ticks
                    np.arange(rewards_y_min + rewards_range/4, rewards_y_max, rewards_range/10)  # Infrequent upper ticks
                ])
                y_ticks = np.unique(np.clip(y_ticks, rewards_y_min, rewards_y_max))
                ax_returns.set_yticks(y_ticks, labels=[f"{tick:.2f}" for tick in y_ticks])
            entropies_range = entropies_y_max - entropies_y_min
            if entropies_range > 0:
                y_ticks = np.concatenate([
                    np.arange(entropies_y_min, entropies_y_min + entropies_range/4, entropies_range/20),  # Frequent lower ticks
                    np.arange(entropies_y_min + entropies_range/4, entropies_y_max, entropies_range/10)  # Infrequent upper ticks
                ])
                y_ticks = np.unique(np.clip(y_ticks, entropies_y_min, entropies_y_max))
                ax_entropy.set_yticks(y_ticks, labels=[f"{tick:.2f}" for tick in y_ticks])
            ax_returns.set_ylim(rewards_y_min, rewards_y_max)
            ax_entropy.set_ylim(entropies_y_min, entropies_y_max)
            # Set x-axis ticks as decimals (steps in 10^6 units)
            x_ticks = [0, 200000, 400000, 600000, 800000, 1000000]
            x_tick_labels = [f"{x // 100000}" for x in x_ticks]  # Divide by 10^5 for display
            ax_returns.set_xticks(x_ticks, labels=x_tick_labels)
            ax_entropy.set_xticks(x_ticks, labels=x_tick_labels)
            ax_returns.tick_params(axis='both', labelsize=5)
            ax_entropy.tick_params(axis='both', labelsize=5)
            ax_returns.set_xlim(0, 1000000)
            ax_entropy.set_xlim(0, 1000000)
        else:
            ax_returns.set_visible(False)  # Empty subplot for missing data
            ax_entropy.set_visible(False)  # Empty subplot for missing data
    # Add scientific notation note for x-axis
    fig.text(0.5, 0.01, r"Steps (\(\times 10^6\))", ha='center', fontsize=6)
    # Remove individual legends
    for ax in axes.flatten():
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    # Add single legend at the bottom as a horizontal row
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(labels), fontsize=8)
    fig.suptitle(f"{algo}: Rewards and Entropy vs Steps Across Environments", fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for legend and note space
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{algo}_rewards_entropy_vs_steps.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

# Plot all algorithms for one environment, each in its own subplot
def plot_env_all_algos(data, env, algorithms, output_dir, smooth_window=5, markers_per_line=10):
    """
    Plot all runs for each algorithm in one environment in a 2x6 tiled grid with returns (top row)
    and entropy (bottom row), with linear y-axis scale (shared for rewards, per-algorithm for entropy),
    x-axis with decimal ticks and scientific notation note for 1M steps, frequent lower y-ticks, tiny
    fonts, increased offset markers, y-label only on first cell of each row, and single legend at the bottom.
    
    Args:
        data: Loaded data.
        env (str): Environment name.
        algorithms: List of algorithm names (expected up to 6).
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
    # Fixed 2x6 grid for 12 square tiles
    nrows, ncols = 2, 6
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6), squeeze=False)
    # Define markers
    markers = ["o", "s", "^", "v", "*", "+", "x"]  # Markers for distinction
    # Get global y-axis bounds for rewards across all algorithms
    keys = [(env, algo) for algo in valid_algos]
    rewards_y_min, rewards_y_max = get_y_bounds(data, 'rewards', keys)
    marker_interval = max(1, 100 // markers_per_line)  # Interval between markers
    handles, labels = [], []
    for i in range(6):
        ax_returns = axes[0, i]
        ax_entropy = axes[1, i]
        if i < len(valid_algos):
            algo = valid_algos[i]
            runs = data[env][algo]
            # Compute y-axis bounds for entropy for this algorithm
            keys = [(env, algo)]
            entropies_y_min, entropies_y_max = get_y_bounds(data, 'entropies', keys)
            for j, run in enumerate(runs):
                rewards = run['rewards']
                entropies = run['entropies']
                if smooth_window > 1:
                    rewards = uniform_filter1d(rewards, size=smooth_window, mode='nearest')
                    entropies = uniform_filter1d(entropies, size=smooth_window, mode='nearest')
                steps = np.linspace(0, 1000000, len(rewards))  # 1M steps
                # Use increased offset for markers to prevent overlap
                offset = j * marker_interval // max(1, len(runs))
                line, = ax_returns.plot(steps, rewards, label=format_label(run['label']), color='black', linestyle='-', marker=markers[j % len(markers)], markevery=(offset, marker_interval), linewidth=2, alpha=0.7, markersize=6)
                ax_entropy.plot(steps, entropies, label=format_label(run['label']), color='black', linestyle='-', marker=markers[j % len(markers)], markevery=(offset, marker_interval), linewidth=2, alpha=0.7, markersize=6)
                # Collect handles and labels from the first returns subplot
                if i == 0:
                    handles.append(line)
                    labels.append(format_label(run['label']))
            ax_returns.set_title(f"{algo} in {env}", fontsize=10)
            # Set y-label only on first cell of each row
            if i == 0:
                ax_returns.set_ylabel("Rewards", fontsize=6)
                ax_entropy.set_ylabel("Entropy", fontsize=6)
            ax_entropy.set_xlabel("Steps", fontsize=6)
            ax_returns.grid(True, linestyle='--', alpha=0.7)
            ax_entropy.grid(True, linestyle='--', alpha=0.7)
            # Set linear y-axis with frequent lower ticks
            rewards_range = rewards_y_max - rewards_y_min
            if rewards_range > 0:
                y_ticks = np.concatenate([
                    np.arange(rewards_y_min, rewards_y_min + rewards_range/4, rewards_range/20),  # Frequent lower ticks
                    np.arange(rewards_y_min + rewards_range/4, rewards_y_max, rewards_range/10)  # Infrequent upper ticks
                ])
                y_ticks = np.unique(np.clip(y_ticks, rewards_y_min, rewards_y_max))
                ax_returns.set_yticks(y_ticks, labels=[f"{tick:.2f}" for tick in y_ticks])
            entropies_range = entropies_y_max - entropies_y_min
            if entropies_range > 0:
                y_ticks = np.concatenate([
                    np.arange(entropies_y_min, entropies_y_min + entropies_range/4, entropies_range/20),  # Frequent lower ticks
                    np.arange(entropies_y_min + entropies_range/4, entropies_y_max, entropies_range/10)  # Infrequent upper ticks
                ])
                y_ticks = np.unique(np.clip(y_ticks, entropies_y_min, entropies_y_max))
                ax_entropy.set_yticks(y_ticks, labels=[f"{tick:.2f}" for tick in y_ticks])
            ax_returns.set_ylim(rewards_y_min, rewards_y_max)
            ax_entropy.set_ylim(entropies_y_min, entropies_y_max)
            # Set x-axis ticks as decimals (steps in 10^6 units)
            x_ticks = [0, 200000, 400000, 600000, 800000, 1000000]
            x_tick_labels = [f"{x // 100000}" for x in x_ticks]  # Divide by 10^5 for display
            ax_returns.set_xticks(x_ticks, labels=x_tick_labels)
            ax_entropy.set_xticks(x_ticks, labels=x_tick_labels)
            ax_returns.tick_params(axis='both', labelsize=5)
            ax_entropy.tick_params(axis='both', labelsize=5)
            ax_returns.set_xlim(0, 1000000)
            ax_entropy.set_xlim(0, 1000000)
        else:
            ax_returns.set_visible(False)  # Empty subplot for missing data
            ax_entropy.set_visible(False)  # Empty subplot for missing data
    # Add scientific notation note for x-axis
    fig.text(0.5, 0.01, r"Steps (\(\times 10^6\))", ha='center', fontsize=6)
    # Remove individual legends
    for ax in axes.flatten():
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    # Add single legend at the bottom as a horizontal row
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=len(labels), fontsize=8)
    fig.suptitle(f"Rewards and Entropy vs Steps for Algorithms in {env}", fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for legend and note space
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{env}_rewards_entropy_vs_steps_all_algos.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

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

    # Get environments and algorithms (limit to 6 each)
    environments = sorted(data.keys())[:6]
    algorithms = sorted(set(algo for env in data for algo in data[env]))[:6]

    # Plot for each algorithm: 6 environments, all runs
    for algo in algorithms:
        plot_algo_across_envs(data, algo, environments, output_dir, args.smooth_window, args.markers_per_line)

    # Plot for each environment: 6 algorithms, each in its own subplot, all runs
    for env in environments:
        plot_env_all_algos(data, env, algorithms, output_dir, args.smooth_window, args.markers_per_line)

    print(f"Plots saved to '{output_dir}' as high-res PNGs.")