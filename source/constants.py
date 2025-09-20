from enum import Enum
from pathlib import Path


# Enums
## Tools/Libraries
class Tool(Enum):
  LENSKIT = "LensKit"
  RECBOLE = "RecBole"

## LensKit algorithms (scorers)
class Scorer(Enum):
  ITEM_KNN = "Item KNN"
  BIASED_MF = "Biased Matrix Factorization"
  BIAS = "Bias"
  # USER_KNN = "User KNN"

## RecBole algorithms (models)
class Model(Enum):
  ITEM_KNN = "ItemKNN"  # Item K-Nearest Neighbours
  POPULARITY = "Pop"  # Popularity
  FACTORED_ITEM_SIMILARITY = "FISM"  # Factored Item Similarity Models
  NEURAL_ATTENTIVE_ITEM_SIMILARITY = "NAIS"  # Neural Attentive Item Similarity Model


# Naming
COLUMN_NAMES = {"user_id": "user_id", "item_id": "item_id", "rating": "rating"}
FILE_NAME_RATINGS = "ratings.csv"
FILE_NAME_RESULTS_LATEST = "latest.yaml"
FILE_NAME_RESULTS_AGGREGATE = "aggregate.yaml"
FILE_NAME_SCRIPT_SEQUENTIAL = "sequential.sh"


# Directories
DIRECTORY_ROOT = Path(__file__).resolve().parent.parent
DIRECTORY_DATASETS = DIRECTORY_ROOT / "datasets"
DIRECTORY_RESULTS = DIRECTORY_ROOT / "results"
DIRECTORY_ARTIFACTS = DIRECTORY_ROOT / "artifacts"
DIRECTORY_SCRIPTS = DIRECTORY_ROOT / "scripts"


# Paths
PATH_RESULTS_LATEST = DIRECTORY_RESULTS / FILE_NAME_RESULTS_LATEST
PATH_RESULTS_AGGREGATE = DIRECTORY_RESULTS / FILE_NAME_RESULTS_AGGREGATE
PATH_SCRIPT_SEQUENTIAL = DIRECTORY_SCRIPTS / FILE_NAME_SCRIPT_SEQUENTIAL


# Experiment variables
RECOMMENDATIONS = 10
SEED = 42
SIZES = [0.1, 0.25, 0.5, 0.75, 1.0] # [0.1, 0.33, 0.67, 1.0]
FIGURES = 4


# Hyperparameters
TEST_SIZE = 0.2
PARTITIONS = 5 # For cross-validation
TRAINING_RECOMMENDATIONS = 100
KNN_K = TRAINING_RECOMMENDATIONS
KNN_MIN_K = 5
FEATURES = 50


# Recommenders
## RecBole
RECBOLE_DIRECTORY_ROOT = DIRECTORY_ARTIFACTS / "recbole"
RECBOLE_DIRECTORY_DATASETS = RECBOLE_DIRECTORY_ROOT / "datasets"
RECBOLE_DIRECTORY_CHECKPOINTS = RECBOLE_DIRECTORY_ROOT / "checkpoints"

RECBOLE_CONFIG = {
  ### Environment
  "gpu_id": "0,1",
  "worker": 8,
  "seed": SEED,
  "data_path": RECBOLE_DIRECTORY_DATASETS,
  "checkpoint_dir": RECBOLE_DIRECTORY_CHECKPOINTS,
  # "state": "WARNING",
  "show_progress": False,


  ### Data
  "field_separator": ',',
  "seq_separator": '\n',
  "USER_ID_FIELD": COLUMN_NAMES["user_id"],
  "ITEM_ID_FIELD": COLUMN_NAMES["item_id"],
  "RATING_FIELD": COLUMN_NAMES["rating"],
  "TIME_FIELD": None,
  "load_col": {"inter": [COLUMN_NAMES["user_id"], COLUMN_NAMES["item_id"], COLUMN_NAMES["rating"]]},


  ### Training
  "epochs": 128,
  "learner": "sgd",
  "train_batch_size": 2048,
  "stopping_step": 2,
  "eval_step": 2,
  "enable_amp": True,
  "enable_scaler": False,


  ### Evaluation
  # "eval_args": {
  #   "split": {"RS": [10 - (TEST_SIZE * 10), 0, (TEST_SIZE * 10)]},
  # },
  "metrics": ["NDCG"],
  "topk": RECOMMENDATIONS,
  "valid_metric": f"NDCG@{RECOMMENDATIONS}",
  "eval_batch_size": 4096,
  "metric_decimal_place": FIGURES,


  ### Model Hyper-Parameters
  #### Shared
  "embedding_size": 64,
  "split_to": 4,

  #### KNN
  "k": KNN_K,
}
