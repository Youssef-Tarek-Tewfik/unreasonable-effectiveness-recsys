import torch
import os
# import time
import psutil
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Literal

from .constants import (
    ALLOWED_EXCEPTIONS, DIRECTORY_DATASETS, COLUMN_NAMES, FIGURES, SIGNIFICANT_FIGURES, SIZES_ABSOLUTE,
    Dataset, Model, Scorer, Sizing, Sampling
)
# from .load import load
# from .sample import sample
from .logger import log
# from .use_recbole import use_recbole
# from .use_lenskit import use_lenskit


def main():
    pass

    
def gpu_check() -> None:
    print("=== GPU Detection ===")
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print("\n=== GPU Details ===")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}:")
            print(f"  Name: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi-processors: {props.multi_processor_count}")

            # Test GPU accessibility
            try:
                torch.cuda.set_device(i)
                test_tensor = torch.tensor([1.0]).cuda()
                print(f"  Status: ✓ Accessible")
            except Exception as e:
                print(f"  Status: ✗ Error - {e}")

        print(f"\n=== Current GPU ===")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Current device name: {torch.cuda.get_device_name()}")

        print(f"\n=== Environment Variables ===")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

    else:
        print("No CUDA GPUs detected")

def inter_to_csv(name: str, columns: list[int], output="ratings", input="raw") -> None:
    input_path = DIRECTORY_DATASETS / name / f"{input}.inter"
    output_path = DIRECTORY_DATASETS / name / f"{output}.csv"
    
    df = pd.read_csv(input_path, sep='\t', skiprows=1, header=None)
    
    df = df.iloc[:, columns]
    
    df.to_csv(output_path, index=False, header=False)
    
    log(f"Converted\n{input_path}\nto\n{output_path}")

def remove_last_column(input: str | Path, output: str | Path, *, header = None) -> None:
    df = pd.read_csv(input, header=header)
    df = df.iloc[:, :-1]
    df.to_csv(output, index=False, header=False)
    print(f"Removed last column from\n{input}\nand saved to\n{output}")

def copy_head(input: str | Path, output: str | Path) -> None:
    with open(input, 'r') as input_file:
        lines = [input_file.readline() for _ in range(5)]
    with open(output, 'w') as output_file:
        output_file.writelines(lines)
    log(f"Copied first 5 lines from\n{input}\nto\n{output}")

def safe_run(function: Callable[[], float]) -> float:
    try:
        return function()
    except Exception as exception:
        propagate = True
        for allowed, message in ALLOWED_EXCEPTIONS:
            if type(exception) == allowed and message in str(exception):
                propagate = False
                break
        if propagate:
            raise exception
        log("Exception handled:", exception, sep='\n')
        return -1.0

def test_run_time():
    log("starting")
    log("cpus: 64")
    log("memory: 128")
    log("env variables:")
    
    variables = [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS",
        "LK_NUM_PROCS", "LK_NUM_THREADS",
    ]
    
    for variable in variables:
        log(f"{variable}:", os.environ.get(variable, "Not set"))
    print('\n')
    
    algorithms = [
        # "LINE",

        "NCEPLRec",
        "SimpleX",
        "NCL",
        "Random",
        "DiffRec",
        "LDiffRec",
    ]

    # for dataset in [Dataset.NETFLIX]:
    #     for sampling in Sampling:
    #         log("Dataset:", dataset.name)
    #         log("Sampling:", sampling.name)
    #         start = time.time()
    #         df, _ = sample(load(dataset), 25_000_000, (Sizing.ABSOLUTE, sampling))
    #         log("Loaded and sampled")
    #         log("Elapsed", start=start)
    #         log("Starting")
    #         start = time.time()
    #         result = use_recbole(df, dataset, f"25m-{sampling.name}")
    #         log("Result:", result)
    #         log("Elapsed", start=start, end="\n\n")

def show_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_in_mb = mem_info.rss / (1024 ** 2)
    vms_in_mb = mem_info.vms / (1024 ** 2)
    log(f"Memory Usage: RSS = {rss_in_mb:.2f} MB, VMS = {vms_in_mb:.2f} MB")

def is_contiguous(series: pd.Series) -> bool:
    """
    Return True if the integer values in series are exactly [0, 1, ..., n-1] with no gaps.
    """
    s = series.dropna()
    if s.empty:
        return False

    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.isna().any():
        return False

    s_int = numeric.astype(int)
    unique_vals = np.sort(s_int.unique())
    return unique_vals[0] == 0 and np.array_equal(unique_vals, np.arange(len(unique_vals)))

def check_contiguous(df: pd.DataFrame) -> bool:
    """
    Check if user and item columns are 0-based contiguous.
    Returns (user_ok, item_ok).
    """
    user_ok = is_contiguous(df[COLUMN_NAMES["user_id"]])
    item_ok = is_contiguous(df[COLUMN_NAMES["item_id"]])
    # log(f"user_id contiguous 0-based: {user_ok}")
    # log(f"item_id contiguous 0-based: {item_ok}")
    return user_ok and item_ok

def make_contiguous(df: pd.DataFrame, output: str | Path) -> None:
    """
    Remap user_id and item_id to 0-based contiguous indices and save as CSV
    with no header, columns in the same order as input df.
    """
    user_col = COLUMN_NAMES["user_id"]
    item_col = COLUMN_NAMES["item_id"]

    # Create mappings based on sorted unique IDs, so order is deterministic
    user_ids = pd.Index(sorted(df[user_col].unique()))
    item_ids = pd.Index(sorted(df[item_col].unique()))

    user_map = {old: new for new, old in enumerate(user_ids)}
    item_map = {old: new for new, old in enumerate(item_ids)}

    df_reindexed = df.copy()
    df_reindexed[user_col] = df_reindexed[user_col].map(user_map)
    df_reindexed[item_col] = df_reindexed[item_col].map(item_map)

    # Sanity checks
    contiguous = check_contiguous(df_reindexed)
    if not contiguous:
        log("Warning: reindexed IDs are not contiguous 0-based as expected.")

    # Save with no header, same column order, comma-separated
    output = Path(output)
    df_reindexed.to_csv(output, index=False, header=False)
    log(f"Saved reindexed data to: {output}")

def check_unique(df: pd.DataFrame) -> bool:
    """
    Return True if every (user_id, item_id) pair appears at most once.
    """
    user_col = COLUMN_NAMES["user_id"]
    item_col = COLUMN_NAMES["item_id"]
    has_dupes = df.duplicated(subset=[user_col, item_col]).any()
    # log(f"(user_id, item_id) pairs unique: {not has_dupes}")
    return not has_dupes

def make_unique(
    df: pd.DataFrame,
    output: str | Path,
    keep: Literal["last", "first"] | None = None,
    aggregate: str | None = None
) -> pd.DataFrame:
    """
    Remove duplicate (user_id, item_id) pairs, keeping the last occurrence,
    and save as CSV with no header, columns in the same order as input df.
    """
    user_col = COLUMN_NAMES["user_id"]
    item_col = COLUMN_NAMES["item_id"]
    rating_col = COLUMN_NAMES["rating"]

    if aggregate is not None:
        df_unique = df.groupby([user_col, item_col], as_index=False).agg({rating_col: aggregate})
    elif keep is not None:
        df_unique = df.drop_duplicates(subset=[user_col, item_col], keep=keep)
    else:
        df_unique = df

    # Sanity check
    if not check_unique(df_unique):
        log("Warning: duplicates still present after make_unique")

    # Save with no header, same column order, comma-separated
    # output = Path(output)
    # df_unique.to_csv(output, index=False, header=False)
    # log(f"Saved unique data to\n{output}")

    return df_unique

def save_parquet(df: pd.DataFrame, output: str | Path) -> None:
    """
    Save a DataFrame to a parquet file without the pandas index.
    """
    output = Path(output)
    df.to_parquet(output, index=False)
    log(f"Saved parquet to: {output}")

def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to more memory-efficient types.
    """
    for col in df.select_dtypes(include=["int", "float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="signed")
    return df

def check_sparsity(df: pd.DataFrame) -> dict[str, float]:
    """
    Calculate sparsity metrics for a user-item interaction dataset.
    Returns a dict with sparsity, density, and other statistics.
    """
    user_col = COLUMN_NAMES["user_id"]
    item_col = COLUMN_NAMES["item_id"]
    
    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    n_interactions = len(df)
    
    # Total possible interactions
    total_possible = n_users * n_items
    
    # Sparsity = 1 - (actual / possible)
    density = n_interactions / total_possible
    sparsity = 1 - density
    
    stats = {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,
        "total_possible": total_possible,
        "density": density,
        "sparsity": sparsity,
        "sparsity_pct": sparsity * 100,
    }
    
    log(f"Users: {n_users:,}")
    log(f"Items: {n_items:,}")
    log(f"Interactions: {n_interactions:,}")
    log(f"Density: {density:.6f} ({density*100:.4f}%)")
    log(f"Sparsity: {sparsity:.6f} ({sparsity*100:.4f}%)")
    
    return stats

def round_significant(x: float, figures: int = SIGNIFICANT_FIGURES) -> float:
  if x == 0:
    return 0
  return round(x, figures - 1 - int(math.floor(math.log10(abs(x)))))

def ceil_significant(x: float, figures: int = SIGNIFICANT_FIGURES) -> float:
  if x == 0:
    return 1
  digits = figures - 1 - int(math.floor(math.log10(abs(x))))
  multiplier = 10 ** digits
  return math.ceil(x * multiplier) / multiplier


if __name__ == "__main__":
    main()
