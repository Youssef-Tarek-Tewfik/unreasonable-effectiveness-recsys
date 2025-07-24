from enum import Enum
import pandas as pd

from .constants import COLUMN_NAMES


class Datasets(Enum):
  MOVIELENS = "movielens"
  GOODREADS = "goodreads"
  AMAZON = "amazon"
  MOVIETWEETINGS = "movietweetings"
  PERSONALITY = "personality"

def load(dataset: Datasets = Datasets.MOVIELENS) -> pd.DataFrame:
  path = f"./datasets/{dataset.value}/ratings.csv"
  df = pd.read_csv(path, sep=',', usecols=[0, 1, 2], names=[
    COLUMN_NAMES["user_id"],
    COLUMN_NAMES["item_id"],
    COLUMN_NAMES["rating"]
  ])
  df = df.reset_index(drop=True)
  return df
