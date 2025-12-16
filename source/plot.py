import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from pathlib import Path
from typing import TypeAlias
from scipy import stats
from matplotlib.ticker import FuncFormatter

from .constants import (
    Sizing, Dataset, DISPLAY_NAMES, RECOMMENDATIONS, DIRECTORY_RESULTS, SIZES_FRACTIONAL, SIZES_ABSOLUTE, SEED
)
from .results import (
    Results, Maxima, OUTPUT_KEY, META_KEY, MODE_KEY, SIZING_KEY, SAMPLING_KEY, load_results, setdefault_nested
)
from .logger import log


MAXIMA_MIN_COUNT = 4
TITLE_PREFIX = "Dataset-Size Effectiveness via"
MARKERS = ['o', '^', '2', 's', 'P', '*', 'X', 'D', '|']
# COLORS = [
#     "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
#     "tab:pink", "tab:gray", "tab:olive", "tab:cyan", "tab:brown",
# ]
COLORS = list(mcolors.CSS4_COLORS.keys())
Slopes: TypeAlias = dict[str, dict[str, dict[str, float]]]


def main():
    np.random.seed(SEED)
    np.random.shuffle(COLORS)

    log("Loading latest results")
    results = load_results()
    log("Done")

    mode = results[META_KEY][MODE_KEY]
    sizing, sampling = mode[SIZING_KEY], mode[SAMPLING_KEY]
    title = f"{TITLE_PREFIX} {DISPLAY_NAMES[sizing]} {DISPLAY_NAMES[sampling]}"

    log("Plotting lines")
    plot_results(results, title)
    log("Done")

    log("Plotting maxima")
    maxima = get_maxima(results)
    plot_maxima(maxima, title)
    log("Done\n")


def plot_results(results: Results, title: str = "Results", output: str = "line") -> None:
    sizing = results[META_KEY][MODE_KEY][SIZING_KEY]
    results = results[OUTPUT_KEY] # type: ignore
    tools = list(results.keys())
    datasets = [d.name for d in Dataset]
    
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
                    if value is not None and value >= 0.0 and value <= 1.0:
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
                        alpha=0.8,
                        path_effects=[pe.Stroke(linewidth=2, foreground="black"), pe.Normal()]
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
                            zorder=3,
                            path_effects=[pe.Stroke(linewidth=2, foreground="black"), pe.Normal()]
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

def get_maxima(results: Results) -> Maxima:
    maxima: Maxima = {}

    for tool in results[OUTPUT_KEY]:
        maxima[tool] = {}
        for algorithm in results[OUTPUT_KEY][tool]:
            maxima[tool][algorithm] = {}
            for dataset in results[OUTPUT_KEY][tool][algorithm]:
                max_value_size: float | int = -1
                max_value: float = -1.0
                max_size: float | int = -1
                count: int = 0

                for size, value in results[OUTPUT_KEY][tool][algorithm][dataset].items():
                    if value is not None and value >= 0 and value <= 1.0:
                        count += 1
                        if value > max_value:
                            max_value = value
                            max_value_size = size
                        if size > max_size:
                            max_size = size

                if count >= MAXIMA_MIN_COUNT:
                    maxima[tool][algorithm][dataset] = round(max_value_size / max_size, 4)

    return maxima

def plot_maxima(maxima: Maxima, title: str = "Maxima", output: str = "maxima") -> None:
    subtitle = f"Maximum NDCG@{RECOMMENDATIONS} value reached at x% of maximum size used, n >= {MAXIMA_MIN_COUNT}"

    tools = list(maxima.keys())
    datasets = [d.name for d in Dataset]
    
    # Build column headers: Tool-Algorithm combinations
    columns = []
    tool_boundaries = []  # Track where each tool's columns end for vertical separators
    tool_spans = []  # Track (start_col, end_col, tool_name) for merged headers
    for tool in tools:
        tool_start = len(columns)
        for algorithm in maxima[tool]:
            columns.append((tool, algorithm))
        tool_spans.append((tool_start, len(columns), tool))
        tool_boundaries.append(len(columns))
    
    # Build the data matrix: rows = datasets, cols = tool-algorithm combos
    n_rows = len(datasets)
    n_cols = len(columns)
    
    # Create figure with extra height for two header rows
    header_height = 1.5  # Height for tool header + algorithm header
    fig, ax = plt.subplots(figsize=(1.8 * n_cols + 2, 0.5 * n_rows + header_height + 2))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows + 2)  # +2 for two header rows
    ax.axis('off')
    
    # Define colors for cells
    color_100 = '#c8e6c9'      # Light green for 100%
    color_50_plus = '#fff9c4'  # Light yellow for >= 50%
    color_less_50 = '#ffcdd2'  # Light red for < 50%
    color_missing = '#e0e0e0'  # Light grey for missing
    
    # Draw tool header row (merged headers spanning algorithm columns)
    for start_col, end_col, tool in tool_spans:
        span = end_col - start_col
        # Tool header cell background (white)
        rect = plt.Rectangle((start_col, n_rows + 1), span, 1, facecolor='white', edgecolor='black', linewidth=0.5)  # type: ignore
        ax.add_patch(rect)
        ax.text(start_col + span / 2, n_rows + 1.5, DISPLAY_NAMES[tool], ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Draw algorithm header row
    for col_idx, (tool, algorithm) in enumerate(columns):
        # Algorithm header cell background
        rect = plt.Rectangle((col_idx, n_rows), 1, 1, facecolor='#f5f5f5', edgecolor='black', linewidth=0.5)  # type: ignore
        ax.add_patch(rect)
        # Split algorithm name into two lines for better fit
        algo_name = DISPLAY_NAMES[algorithm]
        # Split at space if possible, otherwise just use as is
        words = algo_name.split(' ')
        if len(words) >= 2:
            mid = len(words) // 2
            algo_name = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
        ax.text(col_idx + 0.5, n_rows + 0.5, algo_name, ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Draw data cells
    for row_idx, dataset in enumerate(datasets):
        # Row position (inverted so first dataset is at top)
        y_pos = n_rows - 1 - row_idx
        
        for col_idx, (tool, algorithm) in enumerate(columns):
            value = maxima[tool][algorithm].get(dataset, None)
            
            # Determine cell color and text
            if value is None:
                cell_color = color_missing
                cell_text = "-"
            else:
                percentage = value * 100
                # Round to 1 decimal place for consistent color/text behavior
                rounded_percentage = round(percentage, 1)
                if rounded_percentage >= 100:
                    cell_color = color_100
                elif rounded_percentage >= 50:
                    cell_color = color_50_plus
                else:
                    cell_color = color_less_50
                # Handle very small values that round to 0
                if percentage > 0 and percentage < 1.0:
                    cell_text = "<1.0%"
                else:
                    cell_text = f"{rounded_percentage:.1f}%"
            
            # Draw cell
            rect = plt.Rectangle((col_idx, y_pos), 1, 1, facecolor=cell_color, edgecolor='black', linewidth=0.5)  # type: ignore
            ax.add_patch(rect)
            ax.text(col_idx + 0.5, y_pos + 0.5, cell_text, ha='center', va='center', fontsize=12)
    
    # Draw vertical separators between different tools (thicker lines)
    for boundary in tool_boundaries[:-1]:  # Skip the last boundary (right edge)
        ax.axvline(x=boundary, color='black', linewidth=2, ymin=0, ymax=(n_rows + 2) / (n_rows + 2))
    
    # Add dataset labels on the left
    for row_idx, dataset in enumerate(datasets):
        y_pos = n_rows - 1 - row_idx
        ax.text(-0.1, y_pos + 0.5, DISPLAY_NAMES[dataset], ha='right', va='center', fontsize=9, fontweight='bold')
    
    # Add title and subtitle
    fig.suptitle(title, fontsize=20, fontweight="bold", x=0.55, y=0.98)
    fig.text(0.55, 0.92, subtitle, ha='center', va='top', fontsize=14, fontstyle='italic')
    
    plt.tight_layout(rect=[0.15, 0, 1, 0.88])  # type: ignore # Leave room for row labels and title

    # Save the plot
    output_path = DIRECTORY_RESULTS / f"{output}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
