import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.artist import Artist

from .constants import (
    Sizing, Dataset, DISPLAY_NAMES, RECOMMENDATIONS, DIRECTORY_RESULTS, SIZES_FRACTIONAL, SIZES_ABSOLUTE, SEED,
    DATASET_FEEDBACK_EXPLICIT, COLUMN_NAMES
)
from .results import load_results
from .load import load as load_dataset
from .logger import log
from .types import (
    Results, ResultsOutput, Maxima, Normalized, Slopes, NormalizedSlopes, RawSlopes,
    HalfNormalized, ElbowPoints, Gain, ScatterMetadata, LegendType,
    OUTPUT_KEY, META_KEY, MODE_KEY, SIZING_KEY, SAMPLING_KEY
)


# Constants
## Common
COLOR_GREEN = "#c8e6c9"
COLOR_YELLOW = "#fff9c4"
COLOR_RED = "#ffcdd2"
COLOR_MISSING = "#e0e0e0"
TITLE_PREFIX = "Dataset-Size Effectiveness via"
MARKERS = ["o", "^", "2", "s", "P", "*", "X", "D", "|"]
COLORS = list(mcolors.CSS4_COLORS.keys())

## Datasets
DATASETS_TITLE = "Datasets Overview"

## (Result) Lines
LINES_TITLE = f"NDCG@{RECOMMENDATIONS} vs. Dataset Size"

## Maxima
MAXIMA_MIN_COUNT = 5
MAXIMA_TITLE = f"Maximum value reached at x% of maximum size used (n ≥ {MAXIMA_MIN_COUNT})"

## Normalized
### Lines
NORMALIZED_LINES_TITLE = f"Min-Max Normalized NDCG@{RECOMMENDATIONS} vs. Dataset Size"
### Scatter
NORMALIZED_SCATTER_TITLE = f"Min-Max Normalized NDCG@{RECOMMENDATIONS} vs. Dataset Size Scatter"
NORMALIZED_SCATTER_DIFFS = [0.05, 0.10]
NORMALIZED_SCATTER_META_TITLE = f"Scatter Point Distribution"
NORMALIZED_SCATTER_META_BINS = [(0.0, 0.0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1.0, 1.0)]

## Slopes
SLOPES_MIN_COUNT = 5
SLOPES_MIN_DIFF = 0.1
SLOPES_MAX_DIFF = 0.3
SLOPES_TITLE_NORMALIZED = (
    "Normalized Values Late-Stage Slope Distribution" +
    f" ({SLOPES_MIN_DIFF:.0%} ≤ x ≤ {SLOPES_MAX_DIFF:.0%}) (n ≥ {SLOPES_MIN_COUNT})"
)
SLOPES_TITLE_RAW = (
    "Raw Values Late-Stage Slope Distribution" +
    f" ({SLOPES_MIN_DIFF:.0%} ≤ x ≤ {SLOPES_MAX_DIFF:.0%}) (n ≥ {SLOPES_MIN_COUNT})"
)

## Elbow
ELBOW_MIN_COUNT = 5
ELBOW_THRESHOLD = 0.85
ELBOW_TITLE = (
    f"Elbow Point: First size reaching {int(ELBOW_THRESHOLD * 100)}% of max NDCG@{RECOMMENDATIONS}" +
    f" (n ≥ {ELBOW_MIN_COUNT})"
)

## Gain
GAIN_MIN_COUNT = 5
GAIN_MIN = 0.70
GAIN_MAX = 0.90
GAIN_THRESHOLD_GREEN = 10.0
GAIN_THRESHOLD_YELLOW = 1.0
GAIN_TITLE = f"NDCG@{RECOMMENDATIONS} gain from {int(GAIN_MIN * 100)}%-{int(GAIN_MAX * 100)}% to 100% of Data (n ≥ {GAIN_MIN_COUNT})"


def main() -> None:
    np.random.seed(SEED)
    np.random.shuffle(COLORS)

    old = False

    log("Loading latest results")
    results = load_results(DIRECTORY_RESULTS / "16-12-2025-latest.yaml") if old else load_results()
    log("Done")

    mode = results[META_KEY][MODE_KEY]
    sizing, sampling = mode[SIZING_KEY], mode[SAMPLING_KEY]
    suffix = f"via {DISPLAY_NAMES[sizing]} {DISPLAY_NAMES[sampling]}"

    # log("Plotting dataset metadata")
    # dataset_meta = get_dataset_metadata()
    # plot_dataset_metadata(dataset_meta, DATASETS_TITLE)
    # log("Done")

    log("Plotting lines")
    plot_lines(results, f"{LINES_TITLE} {suffix}")
    log("Done")

    log("Plotting lines (zoomed)")
    plot_lines(results, f"{LINES_TITLE} {suffix}", zoom=True, output="lines-zoomed")
    log("Done")

    log("Plotting maxima table")
    maxima = get_maxima(results)
    plot_maxima(maxima, f"{MAXIMA_TITLE} {suffix}")
    log("Done")

    log("Plotting normalized lines")
    normalized = get_normalized(results)
    plot_normalized(normalized, f"{NORMALIZED_LINES_TITLE} {suffix}")
    log("Done")

    log("Plotting normalized scatter (by dataset)")
    plot_scatter(normalized, f"{NORMALIZED_SCATTER_TITLE} {suffix}", legend_type=LegendType.DATASETS, output="scatter-datasets")
    log("Done")

    log("Plotting normalized scatter (by algorithm)")
    plot_scatter(normalized, f"{NORMALIZED_SCATTER_TITLE} {suffix}", legend_type=LegendType.ALGORITHMS, output="scatter-algorithms")
    log("Done")

    log("Plotting scatter metadata")
    scatter_meta = get_scatter_metadata(normalized)
    plot_scatter_metadata(scatter_meta, f"{NORMALIZED_SCATTER_META_TITLE} {suffix}")
    log("Done")

    log("Plotting normalized slopes (by dataset)")
    norm_slopes = get_normalized_slopes(normalized)
    plot_slopes(norm_slopes, f"{SLOPES_TITLE_NORMALIZED} {suffix}", legend_type=LegendType.DATASETS, output="slopes-datasets")
    log("Done")

    log("Plotting normalized slopes (by algorithm)")
    plot_slopes(norm_slopes, f"{SLOPES_TITLE_NORMALIZED} {suffix}", legend_type=LegendType.ALGORITHMS, output="slopes-algorithms")
    log("Done")

    # log("Plotting raw slopes (by dataset)")
    # raw_slopes = get_raw_slopes(results)
    # plot_slopes(raw_slopes, f"{SLOPES_TITLE_RAW} {suffix}", legend_type=LegendType.DATASETS, output="slopes-raw-datasets")
    # log("Done")

    # log("Plotting raw slopes (by algorithm)")
    # plot_slopes(raw_slopes, f"{SLOPES_TITLE_RAW} {suffix}", legend_type=LegendType.ALGORITHMS, output="slopes-raw-algorithms")
    # log("Done")

    log("Plotting elbow table")
    elbow = get_elbow_points(results)
    plot_elbow(elbow, f"{ELBOW_TITLE} {suffix}", output="elbow")
    log("Done")

    log("Plotting gain table")
    half_norm = get_half_normalized(results)
    gain = get_gain(half_norm)
    plot_gain(gain, f"{GAIN_TITLE} {suffix}")
    log("Done")
    

def plot_lines(results: Results, title: str = LINES_TITLE, zoom = False, output: str = "lines") -> None:
    sizing = results[META_KEY][MODE_KEY][SIZING_KEY]
    results = results[OUTPUT_KEY] # type: ignore
    tools = list(results.keys())
    datasets = [d.name for d in Dataset]
    
    # Create a mapping of datasets to colors for consistency
    dataset_colors = {dataset: COLORS[i % len(COLORS)] for i, dataset in enumerate(datasets)}
    
    # Build per-row algorithm order with a shared prefix so common algorithms align,
    # while tool-specific algorithms stay contiguous in each row.
    tool_algorithms = {tool: list(results[tool].keys()) for tool in tools}
    common_algorithms = set(tool_algorithms[tools[0]]) if tools else set()
    for tool in tools[1:]:
        common_algorithms &= set(tool_algorithms[tool])

    common_order: list[str] = []
    for tool in tools:
        for algorithm in tool_algorithms[tool]:
            if algorithm in common_algorithms:
                if algorithm not in common_order:
                    common_order.append(algorithm)

    exclusive_by_tool: dict[str, list[str]] = {}
    for tool in tools:
        exclusive_by_tool[tool] = [a for a in tool_algorithms[tool] if a not in common_algorithms]

    n_cols = max((len(common_order) + len(exclusive_by_tool[tool]) for tool in tools), default=1)
    
    # Create figure with one row per tool
    fig, axes_grid = plt.subplots(len(tools), n_cols,
                                figsize=(4 * n_cols, 6 * len(tools)))
    
    # Add title
    fig.suptitle(title, fontsize=16, fontweight="bold", ha="center")
    
    # Make sure axes_grid is always 2D
    axes_grid = np.array(axes_grid, dtype=object)
    if axes_grid.ndim == 0:
        axes_grid = axes_grid.reshape(1, 1)
    elif axes_grid.ndim == 1:
        if len(tools) == 1:
            axes_grid = axes_grid.reshape(1, -1)
        else:
            axes_grid = axes_grid.reshape(-1, 1)
    
    for tool_idx, tool in enumerate(tools):
        algorithms = results[tool]
        row_algorithm_order = common_order + exclusive_by_tool[tool]
        
        # For each algorithm in this tool
        for algo_idx in range(n_cols):
            ax = axes_grid[tool_idx, algo_idx]
            if algo_idx >= len(row_algorithm_order):
                ax.set_visible(False)
                continue
            algorithm = row_algorithm_order[algo_idx]
            
            # First pass: collect all data for this subplot to determine axis limits
            all_sizes_in_subplot = []
            all_values_in_subplot = []
            subplot_data = {}  # Store processed data for second pass
            
            for dataset in datasets:
                size_data = algorithms[algorithm].get(dataset, {})
                sorted_data = dict(sorted(size_data.items(), key=lambda item: float(item[0]))) if size_data else {}
                
                sizes = list(sorted_data.keys())
                values = list(sorted_data.values())
                
                sizes_filtered = []
                values_filtered = []
                for i, value in enumerate(values):
                    if value is not None and value >= 0.0 and value <= 1.0:
                        sizes_filtered.append(sizes[i])
                        values_filtered.append(value)
                
                if sizes_filtered:
                    subplot_data[dataset] = (sizes_filtered, values_filtered)
                    all_sizes_in_subplot.extend(sizes_filtered)
                    all_values_in_subplot.extend(values_filtered)
            
            # Determine x-axis limits from this subplot's data
            if all_sizes_in_subplot:
                x_min = min(all_sizes_in_subplot)
                x_max = max(all_sizes_in_subplot)
            else:
                # Fallback to global sizes if no data
                fallback_sizes = SIZES_FRACTIONAL if sizing == Sizing.FRACTIONAL.name else SIZES_ABSOLUTE
                x_min, x_max = fallback_sizes[0], fallback_sizes[-1]
            
            # Determine y-axis limits: use 90th percentile to handle outliers
            if all_values_in_subplot:
                sorted_values = sorted(all_values_in_subplot)
                # Use 90th percentile as the y-max to avoid outliers squishing the majority
                percentile_90_idx = int(len(sorted_values) * 0.90)
                percentile_90 = sorted_values[min(percentile_90_idx, len(sorted_values) - 1)]
                # But if there's not much variation, just use the max
                actual_max = max(all_values_in_subplot)
                # If 90th percentile is close to max (within 50%), use max; otherwise use percentile
                if actual_max <= 0:
                    y_max = 1.0
                elif percentile_90 >= actual_max * 0.5 or not zoom:
                    y_max = actual_max * 1.05
                else:
                    y_max = percentile_90 * 1.2  # Give some headroom above the 90th percentile
            else:
                y_max = 1.0
            
            # Second pass: plot with clipping for outliers
            for dataset in datasets:
                if dataset not in subplot_data:
                    continue
                
                sizes_filtered, values_filtered = subplot_data[dataset]
                color = dataset_colors[dataset]
                
                # Clip values to y_max for display (outliers will appear at the top border)
                values_clipped = [min(v, y_max) for v in values_filtered]
                
                ax.plot(
                    sizes_filtered, values_clipped,
                    color=color,
                    marker='o',
                    linewidth=2,
                    markersize=6,
                    label=DISPLAY_NAMES[dataset],
                    alpha=0.8,
                    path_effects=[pe.Stroke(linewidth=2, foreground="black"), pe.Normal()]
                )

                # Highlight the last data point with a distinct marker
                if values_clipped:
                    ax.scatter(
                        [sizes_filtered[-1]], [values_clipped[-1]],
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
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

            # Set x-axis limits based on this subplot's data
            ax.set_xlim(x_min, x_max)
            
            # Set y-axis limits from 0 to computed max
            ax.set_ylim(0, y_max)
                
        # No per-row cleanup needed: missing algorithms are hidden above.
    
    # Create a single shared legend for all datasets at the top right
    legend_handles = [
        plt.Line2D( # type: ignore
            [0], [0], color=dataset_colors[dataset], marker='o',
            linewidth=2, markersize=6, label=DISPLAY_NAMES[dataset]
        ) for dataset in datasets
    ]
    fig.legend(handles=legend_handles, loc='upper right', fontsize=9, ncol=1, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 0.90, 1])  # type: ignore # Leave room for legend on the right
    
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

def plot_maxima(maxima: Maxima, title: str = MAXIMA_TITLE, output: str = "maxima") -> None:
    tools = list(maxima.keys())
    datasets = [d.name for d in Dataset]
    
    # Build column headers: Tool-Algorithm combinations
    columns = []
    tool_spans = []  # Track (start_col, end_col, tool_name) for merged headers
    for tool in tools:
        tool_start = len(columns)
        for algorithm in maxima[tool]:
            columns.append((tool, algorithm))
        tool_spans.append((tool_start, len(columns), tool))
    
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
            ax.text(col_idx + 0.5, y_pos + 0.5, cell_text, ha='center', va='center', fontsize=14)
    
    # Add dataset labels on the left
    for row_idx, dataset in enumerate(datasets):
        y_pos = n_rows - 1 - row_idx
        ax.text(-0.1, y_pos + 0.5, DISPLAY_NAMES[dataset], ha='right', va='center', fontsize=12, fontweight='bold')
    
    # Add title and subtitle
    fig.suptitle(title, fontsize=20, fontweight="bold", x=0.55, y=0.98)
    
    plt.tight_layout(rect=[0.15, 0, 1, 0.88])  # type: ignore # Leave room for row labels and title

    # Save the plot
    output_path = DIRECTORY_RESULTS / f"{output}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

def get_normalized(results: Results) -> Normalized:
    """Min-max normalize each dataset-algorithm curve to [0, 1] range."""
    normalized: Normalized = {}
    
    for tool in results[OUTPUT_KEY]:
        normalized[tool] = {}
        for algorithm in results[OUTPUT_KEY][tool]:
            normalized[tool][algorithm] = {}
            for dataset in results[OUTPUT_KEY][tool][algorithm]:
                normalized[tool][algorithm][dataset] = {}
                data = results[OUTPUT_KEY][tool][algorithm][dataset]
                
                valid_points = [
                    (size, val) for size, val in sorted(data.items(), key=lambda x: float(x[0]))
                    if val is not None and 0 <= val <= 1
                ]
                
                if len(valid_points) < 2:
                    continue
                
                sizes = [float(p[0]) for p in valid_points]
                values = [p[1] for p in valid_points]
                
                # Normalize values to [0, 1] based on min/max value
                min_val, max_val = min(values), max(values)
                
                # Skip if all values are the same (no variation)
                if max_val == min_val:
                    continue
                
                # Normalize sizes to [0, 1] based on min/max size
                min_size, max_size = min(sizes), max(sizes)
                
                for size, val in zip(sizes, values):
                    norm_size = (size - min_size) / (max_size - min_size) if max_size != min_size else 0.0
                    norm_val = (val - min_val) / (max_val - min_val)
                    normalized[tool][algorithm][dataset][norm_size] = norm_val
    
    return normalized

def plot_normalized(normalized: Normalized, title: str = NORMALIZED_LINES_TITLE, output: str = "normalized") -> None:
    tools = list(normalized.keys())
    datasets = [d.name for d in Dataset]
    
    # Create a mapping of datasets to colors for consistency
    dataset_colors = {dataset: COLORS[i % len(COLORS)] for i, dataset in enumerate(datasets)}
    
    # Build per-row algorithm order with a shared prefix so common algorithms align,
    # while tool-specific algorithms stay contiguous in each row.
    tool_algorithms = {tool: list(normalized[tool].keys()) for tool in tools}
    common_algorithms = set(tool_algorithms[tools[0]]) if tools else set()
    for tool in tools[1:]:
        common_algorithms &= set(tool_algorithms[tool])

    common_order: list[str] = []
    for tool in tools:
        for algorithm in tool_algorithms[tool]:
            if algorithm in common_algorithms:
                if algorithm not in common_order:
                    common_order.append(algorithm)

    exclusive_by_tool: dict[str, list[str]] = {}
    for tool in tools:
        exclusive_by_tool[tool] = [a for a in tool_algorithms[tool] if a not in common_algorithms]

    n_cols = max((len(common_order) + len(exclusive_by_tool[tool]) for tool in tools), default=1)
    
    # Create figure with one row per tool
    fig, axes_grid = plt.subplots(len(tools), n_cols,
                                figsize=(4 * n_cols, 6 * len(tools)))
    
    # Add title
    fig.suptitle(title, fontsize=16, fontweight="bold", ha="center")
    
    # Make sure axes_grid is always 2D
    axes_grid = np.array(axes_grid, dtype=object)
    if axes_grid.ndim == 0:
        axes_grid = axes_grid.reshape(1, 1)
    elif axes_grid.ndim == 1:
        if len(tools) == 1:
            axes_grid = axes_grid.reshape(1, -1)
        else:
            axes_grid = axes_grid.reshape(-1, 1)
    
    for tool_idx, tool in enumerate(tools):
        algorithms = normalized[tool]
        row_algorithm_order = common_order + exclusive_by_tool[tool]
        
        # For each algorithm in this tool
        for algo_idx in range(n_cols):
            ax = axes_grid[tool_idx, algo_idx]
            if algo_idx >= len(row_algorithm_order):
                ax.set_visible(False)
                continue
            algorithm = row_algorithm_order[algo_idx]
           
            for dataset in datasets:
                size_data = algorithms[algorithm].get(dataset, {})
                sorted_data = dict(sorted(size_data.items(), key=lambda item: float(item[0]))) if size_data else {}
                
                sizes = list(sorted_data.keys())
                values = list(sorted_data.values())
                
                sizes_filtered = []
                values_filtered = []
                for i, value in enumerate(values):
                    if value is not None and 0.0 <= value <= 1.0:
                        sizes_filtered.append(sizes[i])
                        values_filtered.append(value)
                
                if sizes_filtered:                
                    ax.plot(
                        sizes_filtered, values_filtered,
                        color=dataset_colors[dataset],
                        marker='o',
                        linewidth=2,
                        markersize=6,
                        label=DISPLAY_NAMES[dataset],
                        alpha=0.8,
                        path_effects=[pe.Stroke(linewidth=2, foreground="black"), pe.Normal()]
                    )
            
            # Customize subplot
            title = f"{DISPLAY_NAMES[tool]} - {DISPLAY_NAMES[algorithm]}"
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel("Normalized Dataset Size", fontsize=9)
            ax.set_ylabel(f"Normalized NDCG@{RECOMMENDATIONS}", fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

            # Set axis limits
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
    
    # Create a single shared legend for all datasets at the top right
    legend_handles = [
        plt.Line2D( # type: ignore
            [0], [0], color=dataset_colors[dataset], marker='o',
            linewidth=2, markersize=6, label=DISPLAY_NAMES[dataset]
        ) for dataset in datasets
    ]
    fig.legend(handles=legend_handles, loc='upper right', fontsize=9, ncol=1, framealpha=0.9)
    
    plt.tight_layout(rect=[0, 0, 0.90, 1])  # type: ignore # Leave room for legend on the right
    
    # Save the plot
    output_path = DIRECTORY_RESULTS / f"{output}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

def plot_scatter(normalized: Normalized, title: str = NORMALIZED_SCATTER_TITLE, legend_type: LegendType = LegendType.DATASETS, output: str = "scatter") -> None:
    datasets = [d.name for d in Dataset]
    dataset_colors = {dataset: COLORS[i] for i, dataset in enumerate(datasets)}
    
    # Collect all (tool, algorithm) combos for algorithm legend
    all_algos: list[tuple[str, str]] = []
    for tool in normalized:
        for algorithm in normalized[tool]:
            all_algos.append((tool, algorithm))
    algo_colors = {(t, a): COLORS[i % len(COLORS)] for i, (t, a) in enumerate(all_algos)}
    
    fig, ax = plt.subplots(figsize=(12, 8))

    legend_handles: dict[str, tuple[str, Artist]] = {}

    # Collect points in a stable order, then apply deterministic jitter so
    # repeated calls (e.g., dataset-colored vs algorithm-colored) align exactly.
    points: list[tuple[str, str, str, float, float]] = []
    for tool in sorted(normalized.keys()):
        for algorithm in sorted(normalized[tool].keys()):
            for dataset in sorted(normalized[tool][algorithm].keys()):
                for size, value in sorted(normalized[tool][algorithm][dataset].items(), key=lambda item: float(item[0])):
                    if value is None:
                        continue
                    points.append((tool, algorithm, dataset, float(size), float(value)))

    if not points:
        return

    rng = np.random.default_rng(SEED)
    jitter_x = rng.uniform(-0.0125, 0.0125, len(points))
    jitter_y = rng.uniform(-0.0125, 0.0125, len(points))

    # Build jittered points once so plotting and dump files use identical coordinates.
    jittered_points: list[tuple[str, str, str, float, float, float, float]] = []
    for i, (tool, algorithm, dataset, size, value) in enumerate(points):
        x = size + jitter_x[i]
        y = value + jitter_y[i]
        jittered_points.append((tool, algorithm, dataset, size, value, x, y))

    for tool, algorithm, dataset, _, _, x, y in jittered_points:

        if legend_type == LegendType.DATASETS:
            color = dataset_colors.get(dataset, "gray")
            legend_key = dataset
            legend_label = DISPLAY_NAMES[dataset]
        else:
            color = algo_colors.get((tool, algorithm), "gray")
            legend_key = f"{tool}_{algorithm}"
            legend_label = f"{DISPLAY_NAMES[tool]} - {DISPLAY_NAMES[algorithm]}"

        line, = ax.plot(x, y, color=color, alpha=0.4, linewidth=1.5)
        ax.scatter(
            x, y,
            color=color,
            alpha=0.55,
            s=28,
            edgecolors="black",
            linewidths=0.7,
            zorder=3
        )

        if legend_key not in legend_handles:
            legend_handles[legend_key] = (legend_label, line)
    
    # Add diagonal reference line (linear scaling)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=2, label="Linear scaling")
    
    ax.set_xlabel("Min-Max Normalized Dataset Size", fontsize=12)
    ax.set_ylabel("Min-Max Normalized NDCG", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    
    # # Create legend
    # legend_items = list(legend_handles.values())
    # ax.legend(
    #     [h for _, h in legend_items], [n for n, _ in legend_items],
    #     loc="lower right", fontsize=8, ncol=1 if len(legend_items) <= 12 else 2
    # )
    
    plt.tight_layout()
    output_path = DIRECTORY_RESULTS / f"{output}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

def get_normalized_slopes(normalized: Normalized) -> NormalizedSlopes:
    """Calculate the late stage slopes of the normalized curves for each algorithm-dataset combination."""
    slopes: NormalizedSlopes = {}

    for tool in normalized:
        slopes[tool] = {}
        for algorithm in normalized[tool]:
            slopes[tool][algorithm] = {}
            for dataset in normalized[tool][algorithm]:
                slopes[tool][algorithm][dataset] = 0.0
                size_values = normalized[tool][algorithm][dataset]
                sorted_data = dict(sorted(size_values.items(), key=lambda item: float(item[0]))) if size_values else {}
                
                sizes = []
                values = []
                for size, value in sorted_data.items():
                    if value is not None and 0 <= value:
                        sizes.append(float(size))
                        values.append(value)
                
                if not SLOPES_MIN_COUNT <= len(sizes):
                    del slopes[tool][algorithm][dataset]
                    continue

                x_high, y_high = sizes[-1], values[-1]
                x_low, y_low = None, None

                for i in reversed(range(len(sizes))):
                    x_low, y_low = sizes[i], values[i]
                    diff = x_high - x_low
                    if diff < SLOPES_MIN_DIFF:
                        continue
                    if SLOPES_MAX_DIFF < diff:
                        x_low, y_low = None, None
                    break

                if x_low is None or y_low is None:
                    del slopes[tool][algorithm][dataset]
                    continue

                slope = (y_high - y_low) / (x_high - x_low)
                slopes[tool][algorithm][dataset] = slope
    return slopes

def get_raw_slopes(results: Results) -> RawSlopes:
    """Calculate the late stage slopes of the raw curves for each algorithm-dataset combination.
    
    Sizes are normalized to [0, 1] but NDCG values are kept raw.
    """
    slopes: RawSlopes = {}

    for tool in results[OUTPUT_KEY]:
        slopes[tool] = {}
        for algorithm in results[OUTPUT_KEY][tool]:
            slopes[tool][algorithm] = {}
            for dataset in results[OUTPUT_KEY][tool][algorithm]:
                data = results[OUTPUT_KEY][tool][algorithm][dataset]
                
                valid_points = [
                    (float(size), val) for size, val in sorted(data.items(), key=lambda x: float(x[0]))
                    if val is not None and 0 <= val <= 1
                ]
                
                if len(valid_points) < SLOPES_MIN_COUNT:
                    continue
                
                sizes = [p[0] for p in valid_points]
                values = [p[1] for p in valid_points]
                
                # Normalize sizes to [0, 1]
                min_size, max_size = min(sizes), max(sizes)
                if max_size == min_size:
                    continue
                
                norm_sizes = [(s - min_size) / (max_size - min_size) for s in sizes]
                
                x_high, y_high = norm_sizes[-1], values[-1]
                x_low, y_low = None, None

                for i in reversed(range(len(norm_sizes))):
                    x_low, y_low = norm_sizes[i], values[i]
                    diff = x_high - x_low
                    if diff < SLOPES_MIN_DIFF:
                        continue
                    if SLOPES_MAX_DIFF < diff:
                        x_low, y_low = None, None
                    break

                if x_low is None or y_low is None:
                    continue

                slope = (y_high - y_low) / (x_high - x_low)
                slopes[tool][algorithm][dataset] = slope
    return slopes

def plot_slopes(slopes: Slopes, title: str = SLOPES_TITLE_NORMALIZED, legend_type: LegendType = LegendType.DATASETS, output: str = "slopes") -> None:
    deviation = 0.25
    
    datasets = [d.name for d in Dataset]
    dataset_colors = {dataset: COLORS[i % len(COLORS)] for i, dataset in enumerate(datasets)}
    
    # Collect all (tool, algorithm) combos for algorithm legend
    all_algos: list[tuple[str, str]] = []
    for tool in slopes:
        for algorithm in slopes[tool]:
            all_algos.append((tool, algorithm))
    algo_colors = {(t, a): COLORS[i % len(COLORS)] for i, (t, a) in enumerate(all_algos)}
    
    # Collect all slope values with metadata
    all_slopes: list[tuple[float, str, str, str]] = []  # (slope, dataset, tool, algorithm)
    for tool in slopes:
        for algorithm in slopes[tool]:
            for dataset in slopes[tool][algorithm]:
                slope_value = slopes[tool][algorithm][dataset]
                all_slopes.append((slope_value, dataset, tool, algorithm))
    
    if not all_slopes:
        log("No slopes to plot.")
        return
    
    # Auto-adjust y-axis to data range with some padding
    raw_values = [s[0] for s in all_slopes]
    data_min = min(raw_values)
    data_max = max(raw_values)
    padding = max(0.5, (data_max - data_min) * 0.15)
    minimum = data_min - padding
    maximum = data_max + padding
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create box plot (using raw values since axis auto-adjusts)
    bp = ax.boxplot([raw_values], positions=[1], widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgray')
    bp['boxes'][0].set_alpha(0.5)
    bp['medians'][0].set_color('black')
    bp['medians'][0].set_linewidth(2)
    
    # Scatter plot with deterministic jitter so repeated calls with the same data
    # (e.g., dataset-colored vs algorithm-colored views) keep identical x-positions.
    rng = np.random.default_rng(SEED)
    jitter = rng.uniform(-deviation, deviation, len(raw_values))
    x_positions = [1 + j for j in jitter]
    
    legend_handles = {}
    
    for i, (slope_val, dataset, tool, algorithm) in enumerate(all_slopes):
        if legend_type == LegendType.DATASETS:
            color = dataset_colors.get(dataset, "gray")
            legend_key = dataset
            legend_label = DISPLAY_NAMES[dataset]
        else:
            color = algo_colors.get((tool, algorithm), "gray")
            legend_key = f"{tool}_{algorithm}"
            legend_label = f"{DISPLAY_NAMES[tool]} - {DISPLAY_NAMES[algorithm]}"
        
        point = ax.scatter(
            x_positions[i], slope_val,
            color=color,
            s=80,
            alpha=0.7,
            edgecolors='white',
            linewidths=0.8,
            zorder=3,
            path_effects=[pe.Stroke(linewidth=1.5, foreground="black"), pe.Normal()]
        )
        
        if legend_key not in legend_handles:
            legend_handles[legend_key] = (legend_label, point)
    
    # Customize plot
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Late-Stage Slope', fontsize=12)
    ax.set_xticks([])
    ax.set_xlim(0.4, 1.6)
    ax.set_ylim(minimum, maximum)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at zero for reference
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    
    # Legend
    if legend_type == LegendType.DATASETS:
        # Keep all datasets in the legend, even when some are absent from plotted data.
        legend_items: list[tuple[str, Artist]] = []
        for dataset in datasets:
            if dataset in legend_handles:
                _, handle = legend_handles[dataset]
            else:
                handle = ax.scatter(
                    [], [],
                    color=dataset_colors[dataset],
                    s=80,
                    alpha=0.7,
                    edgecolors='white',
                    linewidths=0.8,
                    zorder=3,
                    path_effects=[pe.Stroke(linewidth=1.5, foreground="black"), pe.Normal()]
                )
            legend_items.append((DISPLAY_NAMES[dataset], handle))

        ax.legend(
            [h for _, h in legend_items], [n for n, h in legend_items],
            loc='upper right', fontsize=8, ncol=1 if len(datasets) <= 12 else 2
        )
    else:
        legend_items = list(legend_handles.values())
        ax.legend(
            [h for _, h in legend_items], [n for n, _ in legend_items],
            loc='upper right', fontsize=8, ncol=1 if len(legend_items) <= 12 else 2
        )
    
    plt.tight_layout()
    output_path = DIRECTORY_RESULTS / f"{output}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

def get_half_normalized(results: Results) -> HalfNormalized:
    """Normalize only sizes to [0, 1] range, keeping NDCG values raw."""
    half_normalized: HalfNormalized = {}
    
    for tool in results[OUTPUT_KEY]:
        half_normalized[tool] = {}
        for algorithm in results[OUTPUT_KEY][tool]:
            half_normalized[tool][algorithm] = {}
            for dataset in results[OUTPUT_KEY][tool][algorithm]:
                half_normalized[tool][algorithm][dataset] = {}
                data = results[OUTPUT_KEY][tool][algorithm][dataset]
                
                valid_points = [
                    (size, val) for size, val in sorted(data.items(), key=lambda x: float(x[0]))
                    if val is not None and 0 <= val <= 1
                ]
                
                if len(valid_points) < 2:
                    continue
                
                sizes = [float(p[0]) for p in valid_points]
                values = [p[1] for p in valid_points]
                
                # Normalize values to [0, 1] based on min/max value
                min_val, max_val = min(values), max(values)
                
                # Skip if all values are the same (no variation)
                if max_val == min_val:
                    continue
                
                # Normalize sizes to [0, 1] based on min/max size
                min_size, max_size = min(sizes), max(sizes)
                
                for size, val in zip(sizes, values):
                    norm_size = (size - min_size) / (max_size - min_size) if max_size != min_size else 0.0
                    half_normalized[tool][algorithm][dataset][norm_size] = val
    
    return half_normalized

def get_dataset_metadata() -> list[dict]:
    """Load each dataset and extract metadata (interactions, users, items, sparsity, etc.)."""
    metadata = []

    for dataset in Dataset:
        # log(f"  Loading {dataset.name}...")
        try:
            df = load_dataset(dataset, parquet=True)
        except Exception:
            try:
                df = load_dataset(dataset, parquet=False)
            except Exception as e:
                # log(f"  Could not load {dataset.name}: {e}")
                continue

        explicit = DATASET_FEEDBACK_EXPLICIT[dataset]
        n_interactions = len(df)
        n_users = df[COLUMN_NAMES["user_id"]].nunique()
        n_items = df[COLUMN_NAMES["item_id"]].nunique()
        density = n_interactions / (n_users * n_items) if (n_users * n_items) > 0 else 0.0
        sparsity = 1.0 - density

        entry: dict = {
            "dataset": dataset.name,
            "display_name": DISPLAY_NAMES[dataset.name].split("-")[0],
            "feedback": "Explicit" if explicit else "Implicit",
            "interactions": n_interactions,
            "users": n_users,
            "items": n_items,
            "density": density,
            "sparsity": sparsity,
        }

        if explicit:
            rating_col = COLUMN_NAMES["rating"]
            entry["rating_min"] = float(df[rating_col].min())
            entry["rating_max"] = float(df[rating_col].max())

        metadata.append(entry)
        log(f"  {entry['display_name']}: {n_interactions:,} interactions, {n_users:,} users, {n_items:,} items")

    return metadata

def plot_dataset_metadata(metadata: list[dict], title: str = DATASETS_TITLE, output: str = "datasets") -> None:
    """Draw a table with dataset metadata."""
    headers = ["Dataset", "Feedback", "Interactions", "Users", "Items", "Density", "Sparsity", "Rating"]
    n_rows = len(metadata)
    n_cols = len(headers)

    fig, ax = plt.subplots(figsize=(1.8 * n_cols + 1, 0.7 * n_rows + 2.5))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows + 1)
    ax.axis('off')

    # Header row
    for col_idx, header in enumerate(headers):
        rect = plt.Rectangle((col_idx, n_rows), 1, 1, facecolor='#e3f2fd', edgecolor='black', linewidth=0.5)  # type: ignore
        ax.add_patch(rect)
        ax.text(col_idx + 0.5, n_rows + 0.5, header, ha='center', va='center', fontsize=18, fontweight='bold')

    # Data rows
    for row_idx, entry in enumerate(metadata):
        y_pos = n_rows - 1 - row_idx
        row_color = 'white' if row_idx % 2 == 0 else '#fafafa'

        # Format rating range
        if "rating_min" in entry:
            rating_text = f"{entry['rating_min']:.0f}-{entry['rating_max']:.0f}"
        else:
            rating_text = "Binary (0/1)"

        cells = [
            entry["display_name"],
            entry["feedback"],
            f"{entry['interactions']:,}",
            f"{entry['users']:,}",
            f"{entry['items']:,}",
            f"{entry['density']*100:.4f}%",
            f"{entry['sparsity']*100:.2f}%",
            rating_text,
        ]

        for col_idx, cell_text in enumerate(cells):
            rect = plt.Rectangle((col_idx, y_pos), 1, 1, facecolor=row_color, edgecolor='black', linewidth=0.5)  # type: ignore
            ax.add_patch(rect)
            ax.text(col_idx + 0.5, y_pos + 0.5, cell_text, ha='center', va='center', fontsize=14)

    fig.suptitle(title, fontsize=34, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.92])  # type: ignore

    output_path = DIRECTORY_RESULTS / f"{output}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

def get_scatter_metadata(normalized: Normalized) -> ScatterMetadata:
    """Count scatter points around diagonal bin centers.
    
    For each bin (bx, by) and each diff in NORMALIZED_SCATTER_DIFFS:
    - Count how many points with x in [bx-diff, bx+diff] also have y in [by-diff, by+diff]
    - Percentage is relative to points in the x-band (not total points)
    """
    all_points: list[tuple[float, float]] = []

    for tool in normalized:
        for algorithm in normalized[tool]:
            for dataset in normalized[tool][algorithm]:
                for size, value in normalized[tool][algorithm][dataset].items():
                    if value is not None:
                        all_points.append((float(size), float(value)))

    total = len(all_points)
    metadata: ScatterMetadata = {}

    for bx, by in NORMALIZED_SCATTER_META_BINS:
        metadata[(bx, by)] = {}
        for diff in NORMALIZED_SCATTER_DIFFS:
            # Count points in the x-band
            x_band_count = sum(1 for x, y in all_points if abs(x - bx) <= diff)
            # Count points in both x-band and y-band
            xy_count = sum(
                1 for x, y in all_points
                if abs(x - bx) <= diff and abs(y - by) <= diff
            )
            percentage = round(xy_count / x_band_count * 100, 2) if x_band_count > 0 else 0.0
            metadata[(bx, by)][diff] = {
                "count": xy_count,
                "x_band_count": x_band_count,
                "percentage": percentage,
                "total": total,
            }

    return metadata

def plot_scatter_metadata(scatter_meta: ScatterMetadata, title: str = NORMALIZED_SCATTER_META_TITLE, output: str = "scatter-metadata") -> None:
    """Draw a table showing scatter point distribution around diagonal bins, with columns per diff."""
    bins = NORMALIZED_SCATTER_META_BINS
    diffs = NORMALIZED_SCATTER_DIFFS
    n_rows = len(bins)
    
    # Build headers: Region | (Count ±d1, % ±d1) | (Count ±d2, % ±d2) | ...
    headers = ["Region"]
    for diff in diffs:
        headers.append(f"Count (\u00b1{diff:.2f})")
        headers.append(f"% (\u00b1{diff:.2f})")
    n_cols = len(headers)

    # Get total from first entry
    first_bin = list(scatter_meta.keys())[0]
    first_diff = list(scatter_meta[first_bin].keys())[0]
    total = scatter_meta[first_bin][first_diff]["total"]

    fig, ax = plt.subplots(figsize=(3.0 * len(diffs) + 4, 0.6 * n_rows + 3))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(-0.5, n_rows + 1.5)
    ax.axis('off')

    # Header
    for col_idx, header in enumerate(headers):
        rect = plt.Rectangle((col_idx, n_rows), 1, 1, facecolor='#e3f2fd', edgecolor='black', linewidth=0.5)  # type: ignore
        ax.add_patch(rect)
        ax.text(col_idx + 0.5, n_rows + 0.5, header, ha='center', va='center', fontsize=16, fontweight='bold')

    # Data rows
    for row_idx, (bx, by) in enumerate(bins):
        y_pos = n_rows - 1 - row_idx

        # Region cell
        rect = plt.Rectangle((0, y_pos), 1, 1, facecolor='white', edgecolor='black', linewidth=0.5)  # type: ignore
        ax.add_patch(rect)
        ax.text(0.5, y_pos + 0.5, f"({bx:.2f}, {by:.2f})", ha='center', va='center', fontsize=16)

        for diff_idx, diff in enumerate(diffs):
            info = scatter_meta[(bx, by)][diff]
            pct = info['percentage']
            count = int(info['count'])

            # Color based on percentage of x-band points
            if pct >= 50:
                cell_color = COLOR_GREEN
            elif pct >= 25:
                cell_color = COLOR_YELLOW
            else:
                cell_color = COLOR_RED if count > 0 else COLOR_MISSING

            col_count = 1 + diff_idx * 2
            col_pct = 2 + diff_idx * 2

            # Count cell
            rect = plt.Rectangle((col_count, y_pos), 1, 1, facecolor=cell_color, edgecolor='black', linewidth=0.5)  # type: ignore
            ax.add_patch(rect)
            ax.text(col_count + 0.5, y_pos + 0.5, f"{count:,}", ha='center', va='center', fontsize=16)

            # Percentage cell
            rect = plt.Rectangle((col_pct, y_pos), 1, 1, facecolor=cell_color, edgecolor='black', linewidth=0.5)  # type: ignore
            ax.add_patch(rect)
            ax.text(col_pct + 0.5, y_pos + 0.5, f"{pct:.1f}%", ha='center', va='center', fontsize=16)

    fig.suptitle(title, fontsize=18, fontweight="bold")
    ax.text(n_cols / 2, -0.3, f"Total points: {int(total):,} | % = count / points in x-band", ha='center', va='center', fontsize=16, fontstyle='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])  # type: ignore
    output_path = DIRECTORY_RESULTS / f"{output}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

def get_elbow_points(results: Results, threshold: float = ELBOW_THRESHOLD) -> ElbowPoints:
    """Find the size % at which NDCG first reaches threshold% of the max NDCG value.
    
    Returns an ElbowPoints dict: tool -> algorithm -> dataset -> size_ratio (float in [0, 1]).
    """
    elbow: ElbowPoints = {}

    for tool in results[OUTPUT_KEY]:
        elbow[tool] = {}
        for algorithm in results[OUTPUT_KEY][tool]:
            elbow[tool][algorithm] = {}
            for dataset in results[OUTPUT_KEY][tool][algorithm]:
                max_value: float = -1.0
                max_size: float | int = -1
                count: int = 0

                for size, value in results[OUTPUT_KEY][tool][algorithm][dataset].items():
                    if value is not None and value >= 0 and value <= 1.0:
                        count += 1
                        if size > max_size:
                            max_size = size
                        if value > max_value:
                            max_value = value

                elbow_size: float | int = -1
                for size, value in results[OUTPUT_KEY][tool][algorithm][dataset].items():
                    if value is not None and value >= 0 and value <= 1.0:
                        if value >= max_value * threshold:
                            elbow_size = size
                            break

                if count >= MAXIMA_MIN_COUNT:
                    elbow[tool][algorithm][dataset] = round(elbow_size / max_size, 4)

    return elbow

def plot_elbow(elbow: ElbowPoints, title: str = ELBOW_TITLE, output: str = "elbow") -> None:
    """Plot elbow point table in the same style as plot_maxima.
    
    Each cell shows the size % at which NDCG first reached ELBOW_THRESHOLD% of max.
    """
    plot_maxima(elbow, title=title, output=output)

def get_gain(half_normalized: HalfNormalized) -> Gain:
    """Calculate % NDCG increase from a partial data point to 100% of data.
    
    Uses half-normalized data (sizes normalized to [0,1], NDCG raw).
    Finds the data point with normalized size closest to GAIN_MAX within [GAIN_MIN, GAIN_MAX],
    then computes: (ndcg_full - ndcg_partial) / ndcg_partial * 100.
    """
    gain: Gain = {}

    for tool in half_normalized:
        gain[tool] = {}
        for algorithm in half_normalized[tool]:
            gain[tool][algorithm] = {}
            for dataset in half_normalized[tool][algorithm]:
                size_data = half_normalized[tool][algorithm][dataset]
                sorted_data = sorted(
                    [(float(s), v) for s, v in size_data.items() if v is not None],
                    key=lambda x: x[0]
                )

                if len(sorted_data) < GAIN_MIN_COUNT:
                    continue

                # NDCG at max normalized size (1.0)
                ndcg_full = sorted_data[-1][1]

                # Find point in [GAIN_MIN, GAIN_MAX] range, prioritize closest to GAIN_MAX
                candidates = [(s, v) for s, v in sorted_data if GAIN_MIN <= s <= GAIN_MAX]

                if not candidates:
                    continue

                # Pick closest to midpoint
                midpoint = (GAIN_MIN + GAIN_MAX) / 2
                best = min(candidates, key=lambda x: abs(x[0] - midpoint))
                ndcg_partial = best[1]

                if ndcg_partial <= 0:
                    continue

                pct_increase = (ndcg_full - ndcg_partial) / ndcg_partial * 100
                gain[tool][algorithm][dataset] = round(pct_increase, 2)

    return gain

def plot_gain(gain: Gain, title: str = GAIN_TITLE, output: str = "gain",
              threshold_green: float = GAIN_THRESHOLD_GREEN,
              threshold_yellow: float = GAIN_THRESHOLD_YELLOW) -> None:
    """Plot gain table showing % NDCG increase from partial to full data.
    
    Color thresholds (flipped - high gain is good):
        abs(gain) >= threshold_green -> green (large benefit from more data)
        abs(gain) >= threshold_yellow -> yellow (moderate benefit)
        otherwise -> red (little benefit, < 1%)
    """
    tools = list(gain.keys())
    datasets = [d.name for d in Dataset]

    # Build columns
    columns: list[tuple[str, str]] = []
    tool_spans: list[tuple[int, int, str]] = []
    for tool in tools:
        tool_start = len(columns)
        for algorithm in gain[tool]:
            columns.append((tool, algorithm))
        tool_spans.append((tool_start, len(columns), tool))

    n_rows = len(datasets)
    n_cols = len(columns)

    if n_cols == 0:
        log("No gain data to plot.")
        return

    header_height = 1.5
    fig, ax = plt.subplots(figsize=(1.8 * n_cols + 2, 0.5 * n_rows + header_height + 2))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows + 2)
    ax.axis('off')

    # Tool header row
    for start_col, end_col, tool in tool_spans:
        span = end_col - start_col
        rect = plt.Rectangle((start_col, n_rows + 1), span, 1, facecolor='white', edgecolor='black', linewidth=0.5)  # type: ignore
        ax.add_patch(rect)
        ax.text(start_col + span / 2, n_rows + 1.5, DISPLAY_NAMES[tool], ha='center', va='center', fontsize=14, fontweight='bold')

    # Algorithm header row
    for col_idx, (tool, algorithm) in enumerate(columns):
        rect = plt.Rectangle((col_idx, n_rows), 1, 1, facecolor='#f5f5f5', edgecolor='black', linewidth=0.5)  # type: ignore
        ax.add_patch(rect)
        algo_name = DISPLAY_NAMES[algorithm]
        words = algo_name.split(' ')
        if len(words) >= 2:
            mid = len(words) // 2
            algo_name = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
        ax.text(col_idx + 0.5, n_rows + 0.5, algo_name, ha='center', va='center', fontsize=8, fontweight='bold')

    # Data cells
    for row_idx, dataset in enumerate(datasets):
        y_pos = n_rows - 1 - row_idx

        for col_idx, (tool, algorithm) in enumerate(columns):
            value = gain[tool][algorithm].get(dataset, None)

            if value is None:
                cell_color = COLOR_MISSING
                cell_text = "-"
            else:
                # Positive values: >=10% green, >=1% yellow, <1% red
                # Negative or zero: red
                if value >= threshold_green:
                    cell_color = COLOR_GREEN
                elif value >= threshold_yellow:
                    cell_color = COLOR_YELLOW
                else:
                    cell_color = COLOR_RED

                sign = "+" if value >= 0 else ""
                cell_text = f"{sign}{value:.1f}%"

            rect = plt.Rectangle((col_idx, y_pos), 1, 1, facecolor=cell_color, edgecolor='black', linewidth=0.5)  # type: ignore
            ax.add_patch(rect)
            ax.text(col_idx + 0.5, y_pos + 0.5, cell_text, ha='center', va='center', fontsize=14)

    # Dataset labels
    for row_idx, dataset in enumerate(datasets):
        y_pos = n_rows - 1 - row_idx
        ax.text(-0.1, y_pos + 0.5, DISPLAY_NAMES[dataset], ha='right', va='center', fontsize=12, fontweight='bold')

    fig.suptitle(title, fontsize=20, fontweight="bold", x=0.55, y=0.98)
    plt.tight_layout(rect=[0.15, 0, 1, 0.88])  # type: ignore

    output_path = DIRECTORY_RESULTS / f"{output}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
