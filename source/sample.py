import pandas as pd

from .constants import SIZES, SEED


def sample(df: pd.DataFrame, size: float = SIZES[(len(SIZES) // 2) - 1]) -> pd.DataFrame:
  if size not in SIZES:
    raise ValueError(f"Size must be one of '{SIZES}'")

  size = int(size * len(df))
  return df.sample(size, random_state=SEED)
