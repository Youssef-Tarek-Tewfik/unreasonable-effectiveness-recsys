import pandas as pd

from .constants import Dataset, DATASET_FEEDBACK_EXPLICIT, COLUMN_NAMES, DIRECTORY_DATASETS, FILE_NAME_RATINGS


SEP = ','
USECOLS_EXPLICIT = [0, 1, 2]
USECOLS_IMPLICIT = [0, 1]
NAMES_EXPLICIT = [COLUMN_NAMES["user_id"], COLUMN_NAMES["item_id"], COLUMN_NAMES["rating"]]
NAMES_IMPLICIT = [COLUMN_NAMES["user_id"], COLUMN_NAMES["item_id"]]


def load(dataset: Dataset = Dataset.MOVIELENS) -> pd.DataFrame:
  path = DIRECTORY_DATASETS / dataset.value / FILE_NAME_RATINGS
  explicit = DATASET_FEEDBACK_EXPLICIT[dataset]
  usecols = USECOLS_EXPLICIT if explicit else USECOLS_IMPLICIT
  names = NAMES_EXPLICIT if explicit else NAMES_IMPLICIT

  df = pd.read_csv(path, sep=SEP, usecols=usecols, names=names)
  df = clean(df, explicit)
  df = df.reset_index(drop=True)
  return df


def clean(df: pd.DataFrame, explicit: bool) -> pd.DataFrame:
  user_id, item_id = COLUMN_NAMES["user_id"], COLUMN_NAMES["item_id"]
  
  if explicit:
    rating = COLUMN_NAMES["rating"]
    return df.groupby([user_id, item_id])[rating].mean().round(1).reset_index()
  else:
    return df.drop_duplicates(subset=[user_id, item_id]).reset_index(drop=True)
