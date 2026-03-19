from enum import Enum
from pathlib import Path
from torch.cuda import OutOfMemoryError


# Enums
## Tools/Libraries
class Tool(Enum):
  LENSKIT = 0
  RECBOLE = 1

## Datasets
class Dataset(Enum):
  MYANIMELIST = "myanimelist"
  IPINYOU = "ipinyou"
  TMALL = "tmall"
  MOVIELENS = "movielens"
  MUSIC4ALL = "music4all"
  NETFLIX = "netflix"
  YAHOO = "yahoo"
  ALIBABA = "alibaba"
  GOODREADS = "goodreads"
  LASTFM = "lastfm"
  AMAZON = "amazon"

## Sampling
### Sizing
class Sizing(Enum):
  FRACTIONAL = 0
  ABSOLUTE = 1
### Sampling strategies
class Sampling(Enum):
  RANDOM = 0
  STRATIFIED_USER = 1
  STRATIFIED_ITEM = 2
  STRATIFIED_HYBRID = 3

## LensKit algorithms (scorers)
class Scorer(Enum):
  POP = 0
  ITEM_KNN = 1
  BIASED_MF = 2
  IMPLICIT_MF = 3
  BIASED_SVD = 4


## RecBole algorithms (models)
class Model(Enum):
  SimpleX = "SimpleX" # SimpleX
  POP = "Pop"  # Popularity
  ITEM_KNN = "ItemKNN"  # Item KNN
  BPR = "BPR"  # Bayesian Personalized Ranking
  NEU_MF = "NeuMF"  # Neural Collaborative Filtering


# Meta
## Datasets
DATASET_FEEDBACK_EXPLICIT = {
  Dataset.MYANIMELIST: True,
  Dataset.IPINYOU: True,
  Dataset.TMALL: True,
  Dataset.MOVIELENS: True,
  Dataset.MUSIC4ALL: True,
  Dataset.NETFLIX: True,
  Dataset.YAHOO: True,
  Dataset.ALIBABA: False,
  Dataset.GOODREADS: True,
  Dataset.LASTFM: True,
  Dataset.AMAZON: True,
}

## Exception handling
ALLOWED_EXCEPTIONS = [
  (KeyError, "Field \"rating\" does not exist in schema"),
  (KeyError, "Column rating does not exist in schema"),
  (RuntimeError, "cholesky solve failed"),
  (ValueError, "Some feat is empty"),
  (AssertionError, ""),
  (AttributeError, "'NoneType' object has no attribute 'items'"),
  (RuntimeError, "\"coalesce\" not implemented"),
  (OutOfMemoryError, "CUDA out of memory"),
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
  Dataset.MYANIMELIST.name: "MyAnimeList-7m",
  Dataset.IPINYOU.name: "iPinYou-22m",
  Dataset.TMALL.name: "TMall-24m",
  Dataset.MOVIELENS.name: "MovieLens-32m",
  Dataset.MUSIC4ALL.name: "Music4All Onion-50m",
  Dataset.NETFLIX.name: "Netflix-100m",
  Dataset.YAHOO.name: "Yahoo Music-115m",
  Dataset.ALIBABA.name: "Alibaba iFashion-191m",
  Dataset.GOODREADS.name: "Goodreads-228m",
  Dataset.LASTFM.name: "last.fm-319m",
  Dataset.AMAZON.name: "Amazon-564m",
  Sizing.FRACTIONAL.name: "Fractional",
  Sizing.ABSOLUTE.name: "Absolute",
  Sampling.RANDOM.name: "Random Sampling",
  Sampling.STRATIFIED_USER.name: "Stratified Sampling (User)",
  Sampling.STRATIFIED_ITEM.name: "Stratified Sampling (Item)",
  Sampling.STRATIFIED_HYBRID.name: "Stratified Sampling (Hybrid)",
  Scorer.POP.name: "Popularity",
  Scorer.ITEM_KNN.name: "Item KNN",
  Scorer.BIASED_MF.name: "Biased MF (ALS)",
  Scorer.IMPLICIT_MF.name: "Implicit MF (ALS)",
  Scorer.BIASED_SVD.name: "Biased MF (SVD)",
  Model.SimpleX.name: "SimpleX",
  Model.POP.name: "Popularity",
  Model.ITEM_KNN.name: "Item KNN",
  Model.BPR.name: "Bayesian Personalized Ranking",
  Model.NEU_MF.name: "Neural Collaborative Filtering",
}
FILE_NAME_RATINGS = "ratings.csv"
FILE_NAME_RATINGS_PARQUET = "ratings.parquet"
FILE_NAME_RESULTS_LATEST = "latest.yaml"
FILE_NAME_RESULTS_AGGREGATE = "aggregate.yaml"
FILE_NAME_SCRIPT_SEQUENTIAL = "sequential.sh"
FILE_NAME_SAMPLING_FACTORS = "factors.yaml"


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
PATH_SAMPLING_FACTORS = DIRECTORY_DATASETS / FILE_NAME_SAMPLING_FACTORS


# Experiment related
RECOMMENDATIONS = 10
SEED = 42
FIGURES = 10
SIGNIFICANT_FIGURES = 3
TRAIN_SIZE = 0.8
VALID_SIZE = 0.1
TEST_SIZE = 0.1
SIZES_FRACTIONAL = [0.1, 0.25, 0.5, 0.75, 1.0]
SIZES_ABSOLUTE = [
  *[i * 1_000 for i in [100, 250, 500, 750]],
  *[i * 1_000_000 for i in [1, 25, 50, 75, 100]],
]
MODE = (Sizing.ABSOLUTE, Sampling.STRATIFIED_USER)


# Environment related
GPUS = 1
WORKERS = 4
BATCH_SIZE = 4096 # 2048 4096 8192 16384
SPLIT_TO = 1


# Hyperparameters
PARTITIONS = 1
EPOCHS = 1
K = 16
MIN_K = K
EMBEDDING_SIZE = 32
REGULARIZATION = 1.0
BIASED_MF_DAMPING = 16
CONFIDENCE_WEIGHT = 16
LEARNING_RATE = 1.0
SVD_N_ITER = 16


# Configurations
## LensKit
LENSKIT_CONFIG_POP = {"score": "rank"} # "quantile" | "rank" | "count"
LENSKIT_CONFIG_ITEM_KNN = {"max_nbrs": K, "min_nbrs": MIN_K}
LENSKIT_CONFIG_BIASED_MF = {
  "embedding_size": EMBEDDING_SIZE,
  "epochs": EPOCHS,
  "regularization": REGULARIZATION,
  "damping": BIASED_MF_DAMPING,
}
LENSKIT_CONFIG_IMPLICIT_MF = {
  "embedding_size": EMBEDDING_SIZE,
  "epochs": EPOCHS,
  "regularization": REGULARIZATION,
  "weight": CONFIDENCE_WEIGHT,
}
LENSKIT_CONFIG_BIASED_SVD = {
  "embedding_size": EMBEDDING_SIZE,
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
RECBOLE_SAVED = False

RECBOLE_DIRECTORY_ROOT = DIRECTORY_ARTIFACTS / "recbole"
RECBOLE_DIRECTORY_DATASETS = RECBOLE_DIRECTORY_ROOT / "datasets"
RECBOLE_DIRECTORY_CHECKPOINTS = RECBOLE_DIRECTORY_ROOT / "checkpoints"

RECBOLE_MODEL_CONFIGS = {
  Model.SimpleX: {
    "embedding_size": EMBEDDING_SIZE, # Default: 64
  },
  Model.POP: {},
  Model.ITEM_KNN: {
    "k": K, # Default: 100
  },
  Model.BPR: {
    "embedding_size": EMBEDDING_SIZE, # Default: 64
  },
  Model.NEU_MF: {
    "mf_embedding_size": EMBEDDING_SIZE, # Default: 64
    "mlp_embedding_size": EMBEDDING_SIZE, # Default: 64
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
  "load_col": None,

  ### Training
  "epochs": EPOCHS,
  "learner": "sgd",
  "learning_rate": LEARNING_RATE,
  "train_batch_size": BATCH_SIZE,
  "stopping_step": 2,
  "eval_step": 4,
  "enable_amp": False,
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
}
RECBOLE_CONFIG_EXPLICIT = {
  **RECBOLE_CONFIG_COMMON,
  "RATING_FIELD": COLUMN_NAMES["rating"],
}
RECBOLE_CONFIG_IMPLICIT = {
  **RECBOLE_CONFIG_COMMON,
  "RATING_FIELD": None,
}
RECBOLE_CONFIGS = {
  True: RECBOLE_CONFIG_EXPLICIT,
  False: RECBOLE_CONFIG_IMPLICIT,
}
