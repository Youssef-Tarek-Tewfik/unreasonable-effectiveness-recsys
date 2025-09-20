import yaml
from typing import Dict, TypeAlias
from pathlib import Path

from .constants import SIZES, PATH_RESULTS_LATEST, PATH_RESULTS_AGGREGATE, DIRECTORY_RESULTS, Tool, Scorer, Model
from .load import Dataset


Result: TypeAlias = float | None
# d[library][algorithm][dataset][size] -> float
Results: TypeAlias = Dict[str, Dict[str, Dict[str, Dict[str, Result]]]]


def main():
  aggregate_results()
  print("Results files aggregation done")


def create_results(sizes = SIZES, default: Result = None) -> Results:
  results: Results = {
    tool.value: {
      algorithm.name: {
        dataset.name: {
          str(int(size * 100.0)) + '%': default for size in sizes
        } for dataset in Dataset
      } for algorithm in (Scorer if tool == Tool.LENSKIT else Model)
    } for tool in Tool
  }
  return results


def load_results(path: str | Path = PATH_RESULTS_LATEST) -> Results:
  results: Results
  try:
    with open(path, 'r') as f:
      results = yaml.safe_load(f) or {}
  except FileNotFoundError:
    results = create_results()
  return results


def save_results(results: Results, tag: str | None = None, path: str | Path = PATH_RESULTS_LATEST) -> None:
  if tag:
    tag = tag.replace(':', '-')
    path = str(path).replace("latest.yaml", f"latest-{tag}.yaml")
  with open(path, 'w') as f:
    yaml.dump(results, f)


def setdefault_results(results: dict, keys: list[str], default: Result = None) -> None:
  current = results
  for key in keys[:-1]:
    current = current.setdefault(key, {})
  current.setdefault(keys[-1], default)


def aggregate_results() -> None:
  paths: list[Path] = list(Path(DIRECTORY_RESULTS).glob("latest*.yaml"))
  collection: list[Results] = [load_results(path) for path in paths]
  output: Results = {}

  for results in collection:
    for tool in results:
      for algorithm in results[tool]:
        for dataset in results[tool][algorithm]:
          for size, value in results[tool][algorithm][dataset].items():
            setdefault_results(output, [tool, algorithm, dataset, size])
            if value is not None and value >= 0:
              output[tool][algorithm][dataset][size] = value

  save_results(output, None, PATH_RESULTS_AGGREGATE)

if __name__ == '__main__':
  main()
