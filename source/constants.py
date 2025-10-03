from enum import Enum
from pathlib import Path


# Enums
## Tools/Libraries
class Tool(Enum):
  LENSKIT = 0
  RECBOLE = 1

## Datasets
class Dataset(Enum):
  MOVIELENS = "movielens"
  NETFLIX = "netflix"
  ALIBABA = "alibaba"
  GOODREADS = "goodreads"
  MUSIC4ALL = "music4all"

## LensKit algorithms (scorers)
class Scorer(Enum):
  POP = 0
  ITEM_KNN = 1
  BIASED_MF = 2
  IMPLICIT_MF = 3
  BIASED_SVD = 4


## RecBole algorithms (models)
class Model(Enum):
  ASYM_KNN = "AsymKNN" # Asymmetric KNN
  POP = "Pop"  # Popularity
  ITEM_KNN = "ItemKNN"  # Item KNN
  BPR = "BPR"  # Bayesian Personalized Ranking
  NEU_MF = "NeuMF"  # Neural Collaborative Filtering


# Meta
## Datasets
DATASET_FEEDBACK_EXPLICIT = {
  Dataset.MOVIELENS: True,
  Dataset.NETFLIX: True,
  Dataset.ALIBABA: False,
  Dataset.GOODREADS: True,
  Dataset.MUSIC4ALL: True,
}

## Exception handling
ALLOWED_EXCEPTIONS: list[tuple[type[Exception], str]] = [
  (KeyError, "Field \"rating\" does not exist in schema"),
  (KeyError, "Column rating does not exist in schema"),
]


# Naming
COLUMN_NAMES = {
  "user_id": "user_id",
  "item_id": "item_id",
  "rating": "rating",
  "timestamp": "timestamp",
}
DISPLAY_NAMES = {
  Tool.LENSKIT.name: "LensKit",
  Tool.RECBOLE.name: "RecBole",
  Dataset.MOVIELENS.name: "MovieLens-32m",
  Dataset.NETFLIX.name: "Netflix-100m",
  Dataset.ALIBABA.name: "Alibaba-iFashion-191m",
  Dataset.GOODREADS.name: "GoodReads-228m",
  Dataset.MUSIC4ALL.name: "Music4All-Onion-252m",
  Scorer.POP.name: "Popularity",
  Scorer.ITEM_KNN.name: "Item KNN",
  Scorer.BIASED_MF.name: "Biased MF (ALS)",
  Scorer.IMPLICIT_MF.name: "Implicit MF (ALS)",
  Scorer.BIASED_SVD.name: "Biased MF (SVD)",
  Model.ASYM_KNN.name: "Asymmetric KNN (User)",
  Model.POP.name: "Popularity",
  Model.ITEM_KNN.name: "Item KNN",
  Model.BPR.name: "Bayesian Personalized Ranking",
  Model.NEU_MF.name: "Neural Collaborative Filtering",
}
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
SIZES = [0.1, 0.25, 0.5, 0.75, 1.0]
FIGURES = 5


# Environment related
GPUS = 2
WORKERS = 8
BATCH_SIZE = 4096
# SPLIT_TO = 2


# Hyperparameters
TRAIN_SIZE = 0.8
VALID_SIZE = 0.1
TEST_SIZE = 0.1
PARTITIONS = 2
EPOCHS = 4
K = 64 # 128?
MIN_K = K // 2
FEATURES = 64
MF_EMBEDDING_SIZE = 64
REGULARIZATION = 0.75 # 0.5? 1.0?
ALPHA = 0.75 # 0.5? 1.0?
BIASED_MF_DAMPING = 16
CONFIDENCE_WEIGHT = 16
LEARNING_RATE = 0.025 # 0.5?
BETA = 1.0
SHRINK = 0.0
SVD_N_ITER = 16


# Configurations
## LensKit
LENSKIT_CONFIG_POP = {"score": "rank"} # "quantile" | "rank" | "count"
LENSKIT_CONFIG_ITEM_KNN = {"max_nbrs": K, "min_nbrs": K}
LENSKIT_CONFIG_BIASED_MF = {
  "embedding_size": MF_EMBEDDING_SIZE,
  "epochs": EPOCHS,
  "regularization": REGULARIZATION,
  "damping": BIASED_MF_DAMPING,
}
LENSKIT_CONFIG_IMPLICIT_MF = {
  "embedding_size": MF_EMBEDDING_SIZE,
  "epochs": EPOCHS,
  "regularization": REGULARIZATION,
  "weight": CONFIDENCE_WEIGHT,
}
LENSKIT_CONFIG_BIASED_SVD = {
  "embedding_size": MF_EMBEDDING_SIZE,
  "damping": BIASED_MF_DAMPING,
  "algorithm": "randomized", # "arpack" | "randomized"
  "n_iter": SVD_N_ITER,
}

LENSKIT_CONFIGS = {
  Scorer.POP: {
    True: LENSKIT_CONFIG_POP,
    False: LENSKIT_CONFIG_POP,
  },
  Scorer.ITEM_KNN: {
    True: {**LENSKIT_CONFIG_ITEM_KNN, "feedback": "explicit"},
    False: {**LENSKIT_CONFIG_ITEM_KNN, "feedback": "implicit"},
  },
  Scorer.BIASED_MF: {
    True: LENSKIT_CONFIG_BIASED_MF,
    False: LENSKIT_CONFIG_BIASED_MF,
  },
  Scorer.IMPLICIT_MF: {
    True: {**LENSKIT_CONFIG_IMPLICIT_MF, "use_ratings": True},
    False: {**LENSKIT_CONFIG_IMPLICIT_MF, "use_ratings": False},
  },
  Scorer.BIASED_SVD: {
    True: LENSKIT_CONFIG_BIASED_SVD,
    False: LENSKIT_CONFIG_BIASED_SVD,
  },
}

## RecBole
RECBOLE_DIRECTORY_ROOT = DIRECTORY_ARTIFACTS / "recbole"
RECBOLE_DIRECTORY_DATASETS = RECBOLE_DIRECTORY_ROOT / "datasets"
RECBOLE_DIRECTORY_CHECKPOINTS = RECBOLE_DIRECTORY_ROOT / "checkpoints"

RECBOLE_MODEL_CONFIGS = {
  Model.ASYM_KNN: {
    "knn_method": "user", # "item" | "user"; Default: "item"
    "k": K, # Default: 100
    "alpha": ALPHA, # Default: 0.5
    "beta": BETA, # Default:  1.0
    "q": 1, # Default: 1
  },
  Model.POP: {},
  Model.ITEM_KNN: {
    "k": K, # Default: 100
    "shrink": SHRINK, # Default: 0.0
  },
  Model.BPR: {
    "embedding_size": MF_EMBEDDING_SIZE, # Default: 64
  },
  Model.NEU_MF: {
    "mf_embedding_size": MF_EMBEDDING_SIZE, # Default: 64
    "mlp_embedding_size": MF_EMBEDDING_SIZE, # Default: 64
    "mlp_hidden_size": [512, 256], # Default: [128,64]
    "dropout_prob": 0.1, # Default: 0.1 
  },
}

RECBOLE_CONFIG_COMMON = {
  ### Environment
  "gpu_id": ','.join([str(i) for i in range(GPUS)]),
  "worker": WORKERS,
  "seed": SEED,
  "data_path": RECBOLE_DIRECTORY_DATASETS,
  "checkpoint_dir": RECBOLE_DIRECTORY_CHECKPOINTS,
  "show_progress": False,

  ### Data
  "field_separator": ',',
  "seq_separator": '\n',
  "USER_ID_FIELD": COLUMN_NAMES["user_id"],
  "ITEM_ID_FIELD": COLUMN_NAMES["item_id"],
  "TIME_FIELD": None,

  ### Training
  "epochs": EPOCHS,
  "learner": "sgd",
  "learning_rate": LEARNING_RATE,
  "train_batch_size": BATCH_SIZE,
  "stopping_step": 2,
  "eval_step": 4,
  "enable_amp": True,
  "enable_scaler": False,

  ### Evaluation
  "eval_args": {
    "split": {"RS": [TRAIN_SIZE, VALID_SIZE, TEST_SIZE]},
  },
  "metrics": ["NDCG"],
  "topk": RECOMMENDATIONS,
  "valid_metric": f"NDCG@{RECOMMENDATIONS}",
  "eval_batch_size": BATCH_SIZE,
  "metric_decimal_place": FIGURES,

  ### Model
  # "split_to": SPLIT_TO,
}
RECBOLE_CONFIG_EXPLICIT = {
  **RECBOLE_CONFIG_COMMON,
  "RATING_FIELD": COLUMN_NAMES["rating"],
  "load_col": {
    "inter": [COLUMN_NAMES["user_id"], COLUMN_NAMES["item_id"], COLUMN_NAMES["rating"]]
  },
}
RECBOLE_CONFIG_IMPLICIT = {
  **RECBOLE_CONFIG_COMMON,
  "RATING_FIELD": None,
  "load_col": {
    "inter": [COLUMN_NAMES["user_id"], COLUMN_NAMES["item_id"]]
  },
}
RECBOLE_CONFIGS = {
  True: RECBOLE_CONFIG_EXPLICIT,
  False: RECBOLE_CONFIG_IMPLICIT,
}
