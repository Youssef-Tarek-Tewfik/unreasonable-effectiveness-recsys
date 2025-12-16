import pandas as pd
import yaml
from math import ceil
from typing import TypeAlias
from pathlib import Path

from .constants import (
  Sizing, Sampling, Dataset, COLUMN_NAMES, MODE, SIZES_FRACTIONAL, SIZES_ABSOLUTE, SEED, PATH_SAMPLING_FACTORS
)
from .logger import log
from .results import setdefault_nested


# Definitions
## Constants
FACTOR_FIGURES = 2
FACTOR_MULTIPLIER = 1.05
USER, ITEM = COLUMN_NAMES["user_id"], COLUMN_NAMES["item_id"]


## Types
FactorValue: TypeAlias = float | None
FactorKey: TypeAlias = int  # Size
DatasetKey: TypeAlias = str
DatasetValue: TypeAlias = dict[FactorKey, FactorValue]
Factors: TypeAlias = dict[DatasetKey, DatasetValue]


## Functions
def create_factors() -> Factors:
  factors: Factors = {
    dataset.name: {
      size: None for size in SIZES_ABSOLUTE
    } for dataset in Dataset
  }
  return factors

def load_factors(path: str | Path = PATH_SAMPLING_FACTORS) -> Factors:
  factors: Factors
  try:
    with open(path, 'r') as f:
      factors = yaml.safe_load(f)
  except FileNotFoundError:
    log("File not found: ", path)
    factors = create_factors()
  return factors

def save_factors(factors: Factors, path: str | Path = PATH_SAMPLING_FACTORS) -> None:
  with open(path, 'w') as f:
    yaml.dump(factors, f)

def sample_proportional(dataset: Dataset, full: pd.DataFrame, size: int, column: str = USER) -> pd.DataFrame:
  original = len(full)
  fraction = size / original
  factors = load_factors()
  setdefault_nested(factors, [dataset.name, size], None)
  factor = factors[dataset.name][size]

  if factor is not None:
    sampled = full.groupby(column).sample(frac=fraction * factor, random_state=SEED)
  else:
    factor = 1.0
    iterations = 0
    sampled = full.groupby(column).sample(frac=fraction * factor, random_state=SEED)
    while len(sampled) < size:
      iterations += 1
      factor = round(factor * FACTOR_MULTIPLIER, FACTOR_FIGURES)
      sampled = full.groupby(column).sample(frac=fraction * factor, random_state=SEED)

    log("Factor:", f"{factor:.2f}")
    log("Iterations:", iterations)
    factors[dataset.name][size] = factor
    save_factors(factors)

  return sampled

def sample(dataset: Dataset, df: pd.DataFrame, size: float | int, mode: tuple[Sizing, Sampling] = MODE) -> pd.DataFrame:
  sizing, sampling = mode

  if sizing == Sizing.FRACTIONAL:
    if size not in SIZES_FRACTIONAL:
      raise ValueError(f"Size '{size}' (fractional) must be one of '{SIZES_FRACTIONAL}'")
    if type(size) is not float or size <= 0.0 or size > 1.0:
      raise ValueError(f"Size '{size}' (fractional) must be a float in the range (0.0, 1.0)")
    
    if sampling == Sampling.RANDOM:
      df = df.sample(frac=size, random_state=SEED)
    elif sampling == Sampling.STRATIFIED_USER:
      df = df.groupby(USER).sample(frac=size, random_state=SEED)
    elif sampling == Sampling.STRATIFIED_ITEM:
      df = df.groupby(ITEM).sample(frac=size, random_state=SEED)
    elif sampling == Sampling.STRATIFIED_HYBRID:
      squared = size ** 0.5
      df = df \
        .groupby(ITEM).sample(frac=squared, random_state=SEED) \
        .groupby(USER).sample(frac=squared, random_state=SEED)

  elif sizing == Sizing.ABSOLUTE:
    if size not in SIZES_ABSOLUTE:
      raise ValueError(f"Size '{size}' (absolute) must be one of '{SIZES_ABSOLUTE}'")
    if type(size) is not int or size <= 0:
      raise ValueError(f"Size '{size}' (absolute) must be an integer greater than 0")

    if size >= len(df):
      pass
    elif sampling == Sampling.RANDOM:
      df = df.sample(n=size, random_state=SEED)
    elif sampling == Sampling.STRATIFIED_USER:
      df = sample_proportional(dataset, df, size, USER)
    elif sampling == Sampling.STRATIFIED_ITEM:
      df = sample_proportional(dataset, df, size, ITEM)
    elif sampling == Sampling.STRATIFIED_HYBRID:
      intermediate = round((len(df) + size) / 2)
      df = sample_proportional(dataset, df, intermediate, ITEM)
      df = sample_proportional(dataset, df, size, USER)

    if len(df) > size:
      df = df.sample(n=size, random_state=SEED)

  df = df.reset_index(drop=True)
  return df
