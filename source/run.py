from argparse import ArgumentParser

from .constants import SIZES, FIGURES, Tool, Scorer, Model
from .load import Dataset, load
from .results import Result, load_results, save_results, setdefault_results
from .sample import sample
from .use_lenskit import use_lenskit
from .use_recbole import use_recbole


def main():
  # Command-line args
  parser = ArgumentParser("Unreasonable Effectiveness of Data for RecSys")
  parser.add_argument('--tag', dest='tag', type=str, required=False)
  args = parser.parse_args()
  tag = args.tag
  tool, algorithm = tag.split(':') if tag else (None, None)
  tool, algorithm = (tool.strip(), algorithm.strip()) if tool and algorithm else (None, None)


  lenskit, recbole = Tool.LENSKIT.value, Tool.RECBOLE.value
  sizes = SIZES[:1]
  excluded = [Dataset.MOVIELENS]
  # excluded = []
  datasets = [dataset for dataset in Dataset if dataset not in excluded]
  results = load_results()
  result: Result

  for dataset in datasets:
    full = load(dataset)

    for size in sizes:
      sampled = sample(full, size)
      percentage = f"{int(size * 100)}%"

      # LensKit
      for scorer in Scorer:
        if tag and (tool != Tool.LENSKIT.name or algorithm != scorer.name):
          continue

        setdefault_results(results, [lenskit, scorer.name, dataset.name, percentage])
        result = results[lenskit][scorer.name][dataset.name][percentage]
        if result is None or result < 0.0:
          result = use_lenskit(sampled, scorer)
          results[lenskit][scorer.name][dataset.name][percentage] = round(result, FIGURES)
          save_results(results, tag)
          print(f"Finished [{Tool.LENSKIT.name}][{scorer.name}][{dataset.name}][{percentage}]")
        else:
          print(f"Skipping [{Tool.LENSKIT.name}][{scorer.name}][{dataset.name}][{percentage}]")

      # RecBole
      for model in Model:
        if tag and (tool != Tool.RECBOLE.name or algorithm != model.name):
          continue

        setdefault_results(results, [recbole, model.name, dataset.name, percentage])
        result = results[recbole][model.name][dataset.name][percentage]
        if result is None or result < 0.0:
          result = use_recbole(sampled, dataset, model)
          results[recbole][model.name][dataset.name][percentage] = round(result, FIGURES)
          save_results(results, tag)
          print(f"Finished [{Tool.RECBOLE.name}][{model.name}][{dataset.name}][{percentage}]")
        else:
          print(f"Skipping [{Tool.RECBOLE.name}][{model.name}][{dataset.name}][{percentage}]")

  print("Work complete\n\n\n\n\n")
  print(results)


if __name__ == '__main__':
  main()
