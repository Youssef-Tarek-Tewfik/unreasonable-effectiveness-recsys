import yaml
from pathlib import Path

from .constants import (
  Tool, Dataset, Scorer, Model, Sizing, PATH_RESULTS_LATEST, PATH_RESULTS_AGGREGATE, DIRECTORY_RESULTS, MODE,
  SIZES_FRACTIONAL, SIZES_ABSOLUTE
)
from .logger import log
from .types import Results, ResultsSizeValue as Result, OUTPUT_KEY, META_KEY, MODE_KEY, SIZING_KEY, SAMPLING_KEY


def main():
  aggregate_results()
  log("Results files aggregation done")


def create_results(*, default: Result = None, empty: bool = False) -> Results:
  sizing, sampling = MODE
  sizes = SIZES_FRACTIONAL if sizing == Sizing.FRACTIONAL else SIZES_ABSOLUTE if sizing == Sizing.ABSOLUTE else []

  results: Results = {
    META_KEY: {
      MODE_KEY: {
        SIZING_KEY: sizing.name,
        SAMPLING_KEY: sampling.name
      },
    },
    OUTPUT_KEY: {} if empty else {
      tool.name: {
        algorithm.name: {
          dataset.name: {
            size: default for size in sizes
          } for dataset in Dataset
        } for algorithm in (Scorer if tool == Tool.LENSKIT else Model)
      } for tool in Tool
    }
  }
  return results

def load_results(path: str | Path = PATH_RESULTS_LATEST) -> Results:
  results: Results
  try:
    with open(path, 'r') as f:
      results = yaml.safe_load(f)
  except FileNotFoundError:
    log("File not found: ", path)
    results = create_results()
  return results

def save_results(results: Results, tag: str | None = None, path: str | Path = PATH_RESULTS_LATEST) -> None:
  if tag:
    tag = tag.replace(':', '-')
    path = str(path).replace("latest.yaml", f"latest-{tag}.yaml")
  with open(path, 'w') as f:
    yaml.dump(results, f)

def setdefault_nested(input: dict | Results, keys: list[str | int | float], default: Result = None) -> None:
  current: dict = input # type: ignore
  for key in keys[:-1]:
    current = current.setdefault(key, {})
  current.setdefault(keys[-1], default)

def aggregate_results() -> None:
  paths: list[Path] = list(Path(DIRECTORY_RESULTS).glob("latest*.yaml"))
  collection: list[Results] = [load_results(path) for path in paths]
  output: Results = create_results(empty=True)

  for results in collection:
    for tool in results[OUTPUT_KEY]:
      for algorithm in results[OUTPUT_KEY][tool]:
        for dataset in results[OUTPUT_KEY][tool][algorithm]:
          for size, value in results[OUTPUT_KEY][tool][algorithm][dataset].items():
            setdefault_nested(output, [OUTPUT_KEY, tool, algorithm, dataset, size]) # type: ignore
            if value is not None and value >= 0:
              output[OUTPUT_KEY][tool][algorithm][dataset][size] = value

  save_results(output, None, PATH_RESULTS_AGGREGATE)


if __name__ == '__main__':
  main()
