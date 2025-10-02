import sys
import csv
from argparse import ArgumentParser

from .constants import Tool, Dataset, Scorer, Model, SIZES, FIGURES
from .load import load
from .results import Result, load_results, save_results, setdefault_results
from .sample import sample
from .utilities import safe_run
from .use_lenskit import use_lenskit
from .use_recbole import use_recbole


def main():
  csv.field_size_limit(sys.maxsize // 10)

  # Command-line args
  parser = ArgumentParser("Unreasonable Effectiveness of Data for RecSys")
  parser.add_argument('--tag', dest='tag', type=str, required=False)
  args = parser.parse_args()
  tag = args.tag
  tool, algorithm = tag.split(':') if tag else (None, None)
  tool, algorithm = (tool.strip(), algorithm.strip()) if tool and algorithm else (None, None)


  lenskit, recbole = Tool.LENSKIT.name, Tool.RECBOLE.name
  sizes = SIZES
  # excluded = [Dataset.MOVIELENS, Dataset.NETFLIX, Dataset.GOODREADS]
  excluded = []
  datasets = [dataset for dataset in Dataset if dataset not in excluded]
  results = load_results()
  result: Result
  current: str

  for dataset in datasets:
    full = load(dataset)

    for size in sizes:
      sampled = sample(full, size)

      # LensKit
      for scorer in Scorer:
        if tag and (tool != Tool.LENSKIT.name or algorithm != scorer.name):
          continue

        current = f"[{Tool.LENSKIT.name}][{scorer.name}][{dataset.name}][{size}]"
        print(f"Checking {current}")
        setdefault_results(results, [lenskit, scorer.name, dataset.name, size])
        result = results[lenskit][scorer.name][dataset.name][size]
        if result is None or result < 0.0:
          print(f"Starting {current}")
          result = safe_run(lambda: use_lenskit(sampled, dataset, scorer))
          results[lenskit][scorer.name][dataset.name][size] = round(result, FIGURES)
          save_results(results, tag)
          print(f"Finished {current} ({result})")
        else:
          print(f"Skipping {current} ({result})")

      # RecBole
      for model in Model:
        if tag and (tool != Tool.RECBOLE.name or algorithm != model.name):
          continue

        current = f"[{Tool.RECBOLE.name}][{model.name}][{dataset.name}][{size}]"
        print(f"Checking {current}")
        setdefault_results(results, [recbole, model.name, dataset.name, size])
        result = results[recbole][model.name][dataset.name][size]
        if result is None or result < 0.0:
          print(f"Starting {current}")
          result = safe_run(lambda: use_recbole(sampled, dataset, size, model))
          results[recbole][model.name][dataset.name][size] = round(result, FIGURES)
          save_results(results, tag)
          print(f"Finished {current} ({result})")
        else:
          print(f"Skipping {current} ({result})")

  print("Work complete\n\n")
  print(results)


if __name__ == '__main__':
  main()
