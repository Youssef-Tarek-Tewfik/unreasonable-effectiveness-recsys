import pandas as pd

from .constants import (
  Dataset, DATASET_FEEDBACK_EXPLICIT, COLUMN_NAMES, DIRECTORY_DATASETS, FILE_NAME_RATINGS, FILE_NAME_RATINGS_PARQUET
)


SEP = ','
USECOLS_EXPLICIT = [0, 1, 2]
USECOLS_IMPLICIT = [0, 1]
NAMES_EXPLICIT = [COLUMN_NAMES["user_id"], COLUMN_NAMES["item_id"], COLUMN_NAMES["rating"]]
NAMES_IMPLICIT = [COLUMN_NAMES["user_id"], COLUMN_NAMES["item_id"]]


def load(dataset: Dataset = Dataset.MOVIELENS, parquet: bool = True) -> pd.DataFrame:
  explicit = DATASET_FEEDBACK_EXPLICIT[dataset]
  names = NAMES_EXPLICIT if explicit else NAMES_IMPLICIT

  if parquet:
    path = DIRECTORY_DATASETS / dataset.value / FILE_NAME_RATINGS_PARQUET
    df = pd.read_parquet(path, columns=names)
  else:
    path = DIRECTORY_DATASETS / dataset.value / FILE_NAME_RATINGS
    usecols = USECOLS_EXPLICIT if explicit else USECOLS_IMPLICIT
    df = pd.read_csv(path, sep=SEP, usecols=usecols, names=names)

  return df
