# The Unreasonable Effectiveness of Data for Recommender Systems

This repository contains the experimental framework and thesis artifacts for a large-scale study of how recommender-system performance changes as the size of the interaction dataset increases.

The project evaluates classic and modern recommendation algorithms across multiple public datasets, using a fixed ranking-based evaluation protocol while varying training-set size over several orders of magnitude. The current implementation integrates LensKit and RecBole, uses user-stratified absolute-size sampling by default, and records results as NDCG@10.

## What This Project Does

- Loads standardized interaction datasets from the local `datasets/` directory.
- Samples each dataset to a target size.
- Trains and evaluates multiple recommenders through LensKit and RecBole.
- Stores intermediate and latest results in YAML files.
- Aggregates results across parallel runs.
- Generates thesis-ready plots and summary tables.
- Supports batch execution on SLURM clusters with Singularity.

The main research question behind the repository is:

> What is the functional relationship between input dataset size and offline recommender-system performance?

## Current Experimental Setup

The codebase is currently configured for the following default experiment mode:

- Sizing mode: `ABSOLUTE`
- Sampling mode: `STRATIFIED_USER`
- Metric: `NDCG@10`
- Train/validation/test split: `0.8 / 0.1 / 0.1`
- Random seed: `42`

Default absolute target sizes:

- `100k`
- `250k`
- `500k`
- `750k`
- `1m`
- `25m`
- `50m`
- `75m`
- `100m`

If a dataset is smaller than the requested target size, the framework uses the full dataset and records the obtained size instead.

## Implemented Backends and Algorithms

### LensKit

- `POP`
- `ITEM_KNN`
- `BIASED_MF`
- `IMPLICIT_MF`
- `BIASED_SVD`

### RecBole

- `SimpleX`
- `POP`
- `ITEM_KNN`
- `BPR`
- `NEU_MF`

## Supported Datasets

The repository is structured around the following datasets:

- `MYANIMELIST`
- `IPINYOU`
- `TMALL`
- `MOVIELENS`
- `MUSIC4ALL`
- `NETFLIX`
- `YAHOO`
- `ALIBABA`
- `GOODREADS`
- `LASTFM`
- `AMAZON`

The thesis and plotting metadata currently use the following display labels:

- MyAnimeList-7m
- iPinYou-22m
- TMall-24m
- MovieLens-32m
- Music4All Onion-50m
- Netflix-100m
- Yahoo Music-115m
- Alibaba iFashion-191m
- Goodreads-228m
- last.fm-319m
- Amazon-564m

Some datasets are treated as explicit-feedback datasets with a `rating` column, while others are treated as implicit-feedback datasets and use only `user_id` and `item_id`.

## Repository Layout

```text
.
|-- artifacts/              # Runtime artifacts, including RecBole dataset/checkpoint files
|-- datasets/               # Prepared input datasets and dataset notes
|-- docs/
|   |-- poster/             # Poster materials
|   `-- thesis/             # LaTeX thesis sources
|-- output/                 # SLURM stdout/stderr and preprocessing logs
|-- results/                # Latest, aggregated, and archived result files and plots
|-- scripts/                # Sequential and parallel cluster execution helpers
|-- singularity/            # Singularity definition and image placeholder
|-- source/                 # Main Python package
|-- makefile                # Convenience commands
|-- requirements.txt        # Python dependencies
`-- notebook.ipynb          # Early exploratory notebook, not actively maintained
```

Important source modules:

- `source/run.py`: main experiment loop
- `source/load.py`: dataset loading
- `source/sample.py`: sampling logic and proportional-factor persistence
- `source/use_lenskit.py`: LensKit training/evaluation path
- `source/use_recbole.py`: RecBole training/evaluation path
- `source/results.py`: latest/aggregate YAML handling
- `source/plot.py`: plot and table generation
- `source/utilities.py`: helper and preprocessing utilities

## Requirements

The project expects:

- Python `3.11`
- `numpy==1.26.4`
- `pandas==2.3.1`
- `torch==2.5.1`
- `scikit-learn==1.7.1`
- `matplotlib==3.10.3`
- `lenskit==2025.2.0`
- `recbole==1.2.1`

For parquet-based dataset loading, you will also need a parquet engine supported by pandas, typically `pyarrow`.

## Setup

### Option 1: Using Make

```bash
make setup
```

This creates a virtual environment and installs dependencies.

### Option 2: Manual Setup

On Windows:

```powershell
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt -U
```

On Linux:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -U
```

## Data Format and Preparation

The runtime currently loads datasets from parquet by default.

Expected file layout per dataset:

```text
datasets/<dataset-name>/ratings.parquet
```

Expected columns:

- Explicit feedback datasets: `user_id`, `item_id`, `rating`
- Implicit feedback datasets: `user_id`, `item_id`

Important operational detail:

- The repository includes many CSV and raw source files for datasets.
- The main runtime in `source/load.py` expects `ratings.parquet`.
- No parquet files are currently committed in this workspace.

That means a fresh clone is not immediately runnable until you prepare parquet files for the datasets you want to use.

The helper functions in `source/utilities.py` include operations for:

- converting `.inter` files to CSV
- enforcing contiguous user/item IDs
- deduplicating user-item pairs
- downcasting numeric dtypes
- writing parquet files
- checking sparsity and dataset consistency

These helpers are utility functions rather than a polished end-user CLI, so dataset preparation is still a project-level task rather than a one-command setup step.

## Running Experiments

### Run the Default Experiment Sweep

```bash
make run
```

Equivalent command:

```bash
python -m source.run
```

By default, the main loop iterates over:

- all configured target sizes
- all configured datasets
- all LensKit scorers
- all RecBole models

Completed results are skipped if already present in the latest results file.

### Run a Specific Tool or Algorithm

The main entry point accepts a `--tag` argument in the form:

```text
TOOL:ALGORITHM
```

Examples:

```bash
python -m source.run --tag LENSKIT:ITEM_KNN
python -m source.run --tag RECBOLE:NEU_MF
python -m source.run --tag LENSKIT:
python -m source.run --tag RECBOLE:
```

Notes:

- `LENSKIT:` runs all LensKit algorithms.
- `RECBOLE:` runs all RecBole algorithms.
- Omitted or empty tags fall back to the internal default currently defined in `source/run.py`.

## Results and Outputs

### Latest Results

The main run writes YAML results into:

- `results/latest.yaml`
- `results/latest-<tag>.yaml` for tagged runs

Each result is stored under a nested structure like:

```text
OUTPUT -> tool -> algorithm -> dataset -> size -> ndcg
```

Unsuccessful or handled failed runs are recorded as `-1` internally and ignored during aggregation.

### Aggregation

To merge partial result files from multiple runs:

```bash
make aggregate
```

Equivalent command:

```bash
python -m source.results
```

This scans `results/latest*.yaml` and writes the merged output to:

- `results/aggregate.yaml`

### Plots

To generate plots and summary tables:

```bash
make plot
```

Equivalent command:

```bash
python -m source.plot
```

The plotting code produces outputs such as:

- line plots of NDCG vs dataset size
- zoomed line plots
- maxima summary tables
- normalized curves
- scatter plots
- slope distributions
- elbow-point tables
- gain tables

Generated plot files are written to `results/`.

## Cluster and Container Execution

The repository includes an HPC-oriented execution path.

### Sequential SLURM Job

```bash
make run-sbatch-sequential
```

This submits `scripts/sequential.sh`, which currently:

- requests `1` node
- requests `64` CPU cores
- requests `64G` RAM
- requests `1` GPU
- uses a wall time of `23:59:59`
- executes the experiment inside `singularity/container.sif`

### Parallel SLURM Submission

```bash
make run-sbatch-parallel
```

Equivalent command:

```bash
python -m scripts.parallel
```

The parallel helper derives temporary SLURM scripts from `scripts/sequential.sh` and submits one job per selected tag.

## Other Make Targets

```bash
make utilities    # Run source.utilities
make show         # Show current SLURM queue for the active user
make cancel       # Cancel all SLURM jobs for the active user
make clean        # Remove logs and temporary artifacts
```

## Thesis Materials

The thesis source lives in:

- `docs/thesis/main.tex`

That document contains the full research framing, methodology, related work, and interpretation of results. The repository itself is the implementation and artifact companion to that written thesis.

## Practical Notes and Caveats

- The framework is designed for very large interaction datasets. A GPU-equipped machine or HPC environment is strongly recommended.
- Several real runs in this project required many hours, and some SLURM jobs hit the configured 24-hour time limit.
- The notebook in the repository is historical and not the maintained execution path.
- RecBole artifacts are written under `artifacts/recbole/` during runtime.
- The project is optimized for reproducible experimentation, not yet for turnkey public reuse.

## Reproducibility Summary

To reproduce the intended workflow at a high level:

1. Prepare the datasets under `datasets/` with the expected standardized schema.
2. Convert the final prepared ratings files to `ratings.parquet`.
3. Create a Python 3.11 environment and install requirements.
4. Run experiments with `python -m source.run` or the provided Make targets.
5. Aggregate partial result files with `make aggregate`.
6. Generate plots with `make plot`.
7. Consult the thesis in `docs/thesis/` for interpretation and research context.

## Status

This repository is an active thesis/research codebase rather than a packaged library. The implementation is sufficient for running and extending the experiments, but some setup steps, especially dataset preparation, remain manual and assume familiarity with the project.
