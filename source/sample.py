import pandas as pd
from math import ceil

from .constants import Sizing, Sampling, MODE, SIZES_FRACTIONAL, SIZES_ABSOLUTE, SEED


def sample(df: pd.DataFrame, size: float | int, mode: tuple[Sizing, Sampling] = MODE) -> tuple[pd.DataFrame, float | int]:
  sizing, sampling = mode
  fraction: float
  absolute: int

  if sizing == Sizing.FRACTIONAL:
    if size not in SIZES_FRACTIONAL:
      raise ValueError(f"Size '{size}' (fractional) must be one of '{SIZES_FRACTIONAL}'")
    if type(size) is not float or size <= 0.0 or size > 1.0:
      raise ValueError(f"Size '{size}' (fractional) must be a float in the range (0.0, 1.0)")

    fraction = size
    absolute = ceil(len(df) * size)
  elif sizing == Sizing.ABSOLUTE:
    if size not in SIZES_ABSOLUTE:
      raise ValueError(f"Size '{size}' (absolute) must be one of '{SIZES_ABSOLUTE}'")
    if type(size) is not int or size <= 0:
      raise ValueError(f"Size '{size}' (absolute) must be an integer greater than 0")

    fraction = min(size / len(df), 1.0)
    absolute = size if fraction < 1.0 else len(df)

  if fraction > 1.0 or fraction <= 0.0 or absolute > len(df) or absolute <= 0:
    message = f"Invalid fractional '{fraction}' or absolute size value '{absolute}' from size argument '{size}'"
    raise ValueError(message)

  if sampling == Sampling.RANDOM:
    df = df.sample(frac=fraction, random_state=SEED)
  elif sampling == Sampling.STRATIFIED_USER:
    df = df.groupby("user_id").sample(frac=fraction, random_state=SEED)
  elif sampling == Sampling.STRATIFIED_ITEM:
    df = df.groupby("item_id").sample(frac=fraction, random_state=SEED)
  elif sampling == Sampling.STRATIFIED_HYBRID:
    df = df \
      .groupby("item_id").sample(frac=(fraction ** 0.5), random_state=SEED) \
      .groupby("user_id").sample(frac=(fraction ** 0.5), random_state=SEED)

  if sampling in (Sampling.STRATIFIED_USER, Sampling.STRATIFIED_ITEM, Sampling.STRATIFIED_HYBRID):
    df = df if len(df) <= absolute else df.sample(n=absolute, random_state=SEED)

  df = df.reset_index(drop=True)
  return df, fraction if sizing == Sizing.FRACTIONAL else len(df) if sizing == Sizing.ABSOLUTE else -1
