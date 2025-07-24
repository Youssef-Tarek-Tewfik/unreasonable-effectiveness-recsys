import pandas as pd

from .constants import SEED, SIZES


def sample(df: pd.DataFrame, size: float = SIZES[-1]) -> pd.DataFrame:
  if size not in SIZES:
    raise ValueError(f"Size must be one of '{SIZES}'")
  
  size = int(size * df.size)
  return df.sample(size, random_state=SEED)
