from enum import Enum
import pandas as pd

from .constants import COLUMN_NAMES, DIRECTORY_DATASETS, FILE_NAME_RATINGS


class Dataset(Enum):
  GOODREADS = "goodreads"
  AMAZON = "amazon"
  MOVIETWEETINGS = "movietweetings"
  PERSONALITY = "personality"
  MOVIELENS = "movielens"


def load(dataset: Dataset = Dataset.GOODREADS) -> pd.DataFrame:
  path = DIRECTORY_DATASETS / dataset.value / FILE_NAME_RATINGS
  df = pd.read_csv(path, sep=',', usecols=[0, 1, 2], names=[
    COLUMN_NAMES["user_id"],
    COLUMN_NAMES["item_id"],
    COLUMN_NAMES["rating"]
  ])
  df = clean(df)
  df = df.reset_index(drop=True)
  return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
  user_id, item_id, rating = COLUMN_NAMES["user_id"], COLUMN_NAMES["item_id"], COLUMN_NAMES["rating"]
  return df.groupby([user_id, item_id])[rating].mean().round(1).reset_index()
