import enum
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import TypeAlias
from scipy import stats
from matplotlib.ticker import FuncFormatter

from .constants import Sizing, DISPLAY_NAMES, RECOMMENDATIONS, DIRECTORY_RESULTS, SIZES_FRACTIONAL, SIZES_ABSOLUTE
from .results import Results, load_results, setdefault_nested
from .logger import log


MARKERS = ['o', '^', '2', 's', 'P', '*', 'X', 'D', '|']
COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"
] # "tab:brown" removed its so bad
Slopes: TypeAlias = dict[str, dict[str, dict[str, float]]]


def main():
    results = load_results()
    plot_results(results)
    log("Line graph created")


def plot_results(results: Results, *, output: str = "line", title: str = "") -> None:
    mode = results["META"]["MODE"]
    sizing, sampling = DISPLAY_NAMES[mode["SIZING"]], DISPLAY_NAMES[mode["SAMPLING"]]
    prefix = f"{title} - " if title else ""
    title = prefix + f"{sizing} {sampling}"

    results = results["OUTPUT"] # type: ignore
    tools = list(results.keys())
    datasets = sorted(
        {dataset for tool in tools for algorithm in results[tool] for dataset in results[tool][algorithm].keys()}
    )
    
    # Create a mapping of datasets to colors for consistency
    dataset_colors = {dataset: COLORS[i % len(COLORS)] for i, dataset in enumerate(datasets)}
    
    # Calculate number of plots per row (number of algorithms per tool)
    plots_per_row = {}
    for tool in tools:
        plots_per_row[tool] = len(results[tool])
    
    # Create figure with one row per tool
    fig, axes_grid = plt.subplots(len(tools), max(plots_per_row.values()), 
                                figsize=(4 * max(plots_per_row.values()), 6 * len(tools)))
    
    # Add title at the top right in the middle
    fig.suptitle(title, fontsize=16, fontweight="bold", ha="center")
    
    # Make sure axes_grid is always 2D
    if len(tools) == 1:
        axes_grid = [axes_grid] if max(plots_per_row.values()) == 1 else [axes_grid]
    
    for tool_idx, tool in enumerate(tools):
        algorithms = list(results[tool].keys())
        
        # For each algorithm in this tool
        for algo_idx, algorithm in enumerate(algorithms):
            ax = axes_grid[tool_idx][algo_idx] if max(plots_per_row.values()) > 1 else axes_grid[tool_idx]
            
            # Track maximum value for this subplot
            max_value = 0.0
            
            # Plot each dataset as a separate line
            for dataset in datasets:
                # Extract sizes and values for this tool-algorithm-dataset combination
                size_data = results[tool][algorithm].get(dataset, {})
                
                # Sort by size percentage
                sorted_data = dict(sorted(size_data.items(), key=lambda item: float(item[0]))) if size_data else {}
                
                sizes = list(sorted_data.keys())
                values = list(sorted_data.values())
                # Handle None values - but keep zeros as zeros
                sizes_filtered = []
                values_filtered = []
                for i, value in enumerate(values):
                    if value is not None:
                        sizes_filtered.append(sizes[i])
                        values_filtered.append(value)
                sizes = sizes_filtered
                values = values_filtered
                
                # Plot the dataset if we have ANY data points, even if all are zeros
                if sizes:
                
                    # Update max value for this subplot
                    max_value = max(max_value, max(values) if values else 0.0)
                    
                    # Plot the line with consistent dataset color
                    color = dataset_colors[dataset]
                    
                    ax.plot(
                        sizes, values,
                        color=color,
                        marker='o',
                        linewidth=2,
                        markersize=6,
                        label=DISPLAY_NAMES[dataset],
                        alpha=0.8
                    )

                    # Highlight the last data point with a distinct marker
                    if values:
                        ax.scatter(
                            [sizes[-1]], [values[-1]],
                            color=color,
                            marker='s',
                            s=90,
                            edgecolors="white",
                            linewidths=0.8,
                            zorder=3
                        )
            
            # Customize subplot
            title = f"{DISPLAY_NAMES[tool]} - {DISPLAY_NAMES[algorithm]}"
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel("Dataset Size", fontsize=9)
            ax.set_ylabel(f"NDCG@{RECOMMENDATIONS}", fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Show legend regardless of data presence
            ax.legend(fontsize=8, loc="best")
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

            # Set x-axis limits based on sizing type
            sizes = SIZES_FRACTIONAL if sizing == Sizing.FRACTIONAL.name else SIZES_ABSOLUTE
            ax.set_xlim(sizes[0], sizes[-1])
            
            # Set y-axis limits from 0 to local maximum with small padding
            if max_value > 0:
                ax.set_ylim(0, max_value * 1.05)  # Add 5% padding above max value
            else:
                ax.set_ylim(0, 1)  # Fallback if no data
                
        # Hide unused subplots if any
        for extra_idx in range(len(algorithms), max(plots_per_row.values())):
            if max(plots_per_row.values()) > 1:
                axes_grid[tool_idx][extra_idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = DIRECTORY_RESULTS / f"{output}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

def get_slopes(results: Results, log = True) -> Slopes:
    slopes: Slopes = {}
    for tool in results:
        for algorithm in results[tool]:
            for dataset in results[tool][algorithm]:
                setdefault_nested(slopes, [tool, algorithm, dataset], 0.0)

                x = []
                y = []
                zeros = True
                for size, value in results[tool][algorithm][dataset].items():
                    if value is not None:
                        x.append(float(size))
                        y.append(float(value))

                        if value > 0.0:
                            zeros = False

                if zeros or len(x) < 2:
                    continue

                x = np.array(x)
                y = np.array(y)
                if log:
                    x = np.log(x)

                    y = np.log(y)

                slope: np.float64 = stats.linregress(x, y)[0] # type: ignore
                slopes[tool][algorithm][dataset] = float(slope)

    return slopes

def plot_slopes(slopes: Slopes, output: str = "bar") -> None:
    tools = list(slopes.keys())
    datasets = sorted(
        {dataset for tool in tools for algorithm in slopes[tool] for dataset in slopes[tool][algorithm].keys()}
    )
    
    # Calculate number of plots per row (number of algorithms per tool)
    plots_per_row = {}
    for tool in tools:
        plots_per_row[tool] = len(slopes[tool])
    
    # Create figure with one row per tool
    fig, axes_grid = plt.subplots(len(tools), max(plots_per_row.values()), 
                                figsize=(4 * max(plots_per_row.values()), 6 * len(tools)))
    
    # Make sure axes_grid is always 2D
    if len(tools) == 1:
        axes_grid = [axes_grid] if max(plots_per_row.values()) == 1 else [axes_grid]
    
    for tool_idx, tool in enumerate(tools):
        algorithms = list(slopes[tool].keys())
        
        # For each algorithm in this tool
        for algo_idx, algorithm in enumerate(algorithms):
            ax = axes_grid[tool_idx][algo_idx] if max(plots_per_row.values()) > 1 else axes_grid[tool_idx]
            
            # Extract slope values for each dataset
            dataset_labels = []
            slope_values = []
            colors_list = []
            
            for dataset in datasets:
                slope_value = slopes[tool][algorithm].get(dataset, 0.0)
                dataset_labels.append(DISPLAY_NAMES[dataset])
                slope_values.append(slope_value)
                colors_list.append(COLORS[datasets.index(dataset) % len(COLORS)])
            
            # Create bar chart
            x_pos = np.arange(len(dataset_labels))
            bars = ax.bar(x_pos, slope_values, color=colors_list, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add a horizontal line at y=0 for reference
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Customize subplot
            title = f"{DISPLAY_NAMES[tool]} - {DISPLAY_NAMES[algorithm]}"
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel("Dataset", fontsize=9)
            ax.set_ylabel("Slope (NDCG change per size unit)", fontsize=9)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(dataset_labels, rotation=45, ha='right', fontsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            
        # Hide unused subplots if any
        for extra_idx in range(len(algorithms), max(plots_per_row.values())):
            if max(plots_per_row.values()) > 1:
                axes_grid[tool_idx][extra_idx].set_visible(False)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = DIRECTORY_RESULTS / f"{output}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
