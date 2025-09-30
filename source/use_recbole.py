import os
import pandas as pd
from recbole.quick_start import run_recbole

from .constants import (
  Dataset, Model, DATASET_FEEDBACK_EXPLICIT, RECBOLE_DIRECTORY_CHECKPOINTS, RECBOLE_DIRECTORY_DATASETS,
  COLUMN_NAMES, RECBOLE_MODEL_CONFIGS, RECBOLE_CONFIGS
)


FLOAT_COLUMNS = [COLUMN_NAMES["rating"]]


def save_as_atomic(df: pd.DataFrame, dataset: Dataset) -> None:
  checkpoints = RECBOLE_DIRECTORY_CHECKPOINTS
  path = RECBOLE_DIRECTORY_DATASETS / dataset.value / f"{dataset.value}.inter"
  os.makedirs(os.path.dirname(checkpoints), exist_ok=True)
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w") as f:
    features = [f"{name}:float" if name in FLOAT_COLUMNS else f"{name}:token" for name in df.columns]
    f.write(','.join(features) + '\n')
  df.to_csv(path, mode="a", index=False, header=False)


def use_recbole(df: pd.DataFrame, dataset: Dataset, model: Model = Model.ITEM_KNN) -> float:
  save_as_atomic(df, dataset)

  explicit = DATASET_FEEDBACK_EXPLICIT[dataset]
  config = RECBOLE_CONFIGS[explicit]
  model_config = RECBOLE_MODEL_CONFIGS[model]
  config = {**config, **model_config}
  
  result = run_recbole(model=model.value, dataset=dataset.value, config_dict=config, saved=False)

  test_result = result["test_result"]
  for key, value in test_result.items():
    if "ndcg" in key.lower():
      return float(value)
  raise ValueError("No NDCG metric found")
