import sys
import csv
import time
import gc
from argparse import ArgumentParser

from .constants import Tool, Dataset, Sizing, Scorer, Model, MODE, SIZES_FRACTIONAL, SIZES_ABSOLUTE, FIGURES
from .load import load
from .results import OUTPUT_KEY, SizeValue as Result, load_results, save_results, setdefault_nested
from .sample import sample
from .utilities import safe_run, show_memory
from .use_lenskit import use_lenskit
from .use_recbole import use_recbole
from .logger import log


def main():
  csv.field_size_limit(sys.maxsize // 10)

  # Run options
  parallel = True
  default = (Tool.RECBOLE.name, Model.ITEM_KNN.name)
  # excluded = [Dataset.ALIBABA, Dataset.AMAZON]
  # datasets = [dataset for dataset in Dataset if dataset not in excluded]
  datasets = [Dataset.MOVIELENS, Dataset.NETFLIX, Dataset.MUSIC4ALL, Dataset.GOODREADS, Dataset.AMAZON, Dataset.ALIBABA]

  # Command-line args
  parser = ArgumentParser("Unreasonable Effectiveness of Data for RecSys")
  parser.add_argument("--tag", dest="tag", type=str, required=False)
  args = parser.parse_args()
  tag = args.tag or (':'.join(default) if parallel else None)
  tool, algorithm = tag.split(':') if tag else default
  tool, algorithm = (tool.strip() if tool else None, algorithm.strip() if algorithm else None)

  # Runtime variables
  lenskit, recbole = Tool.LENSKIT.name, Tool.RECBOLE.name
  sizing, sampling = MODE
  sizes = SIZES_FRACTIONAL if sizing == Sizing.FRACTIONAL else SIZES_ABSOLUTE if sizing == Sizing.ABSOLUTE else []
  results = load_results()
  result: Result
  current: str
  stopwatch: float

  log("Mode:", *(m.name for m in MODE))
  log("Tag: ", tag if tag else "none")
  log("Tool:", tool if tool else "all")
  log("Algorithm:", algorithm if algorithm else "all")
  log("Datasets:", ', '.join(d.name for d in datasets))
  log("Sizes:", sizes, end="\n\n")

  for dataset in datasets:
    log(f"Current dataset [{dataset.name}]")
    for size in sizes:
      log("Loading full dataset")
      stopwatch = time.time()
      dataframe = load(dataset)
      log("Loading done")
      show_memory()
      log("Elapsed", start=stopwatch)

      log(f"Sampling for [{size}]")
      stopwatch = time.time()
      dataframe, size = sample(dataframe, size)
      log("Sampling done")
      show_memory()
      log("Elapsed", start=stopwatch)
      if sizing == Sizing.FRACTIONAL:
        size_formatted = f"{int(size * 100)}%"
      elif sizing == Sizing.ABSOLUTE:
        size_formatted = f"{round(size / 1_000_000)}m"

      # LensKit
      for scorer in Scorer:
        if (tool and tool != lenskit) or (algorithm and algorithm != scorer.name):
          continue

        current = f"[{lenskit}][{scorer.name}][{dataset.name}][{size_formatted}]"
        log(f"Checking {current}")
        setdefault_nested(results, [OUTPUT_KEY, lenskit, scorer.name, dataset.name, size])
        result = results[OUTPUT_KEY][lenskit][scorer.name][dataset.name][size]
        if result is None or result < 0.0:
          log("Starting")
          stopwatch = time.time()

          result = safe_run(lambda: use_lenskit(dataframe, dataset, scorer))
          result = round(result, FIGURES)
          results[OUTPUT_KEY][lenskit][scorer.name][dataset.name][size] = result
          save_results(results, tag)

          log(f"Finished ({result})")
          show_memory()
          log("Elapsed", start=stopwatch, end="\n\n")
        else:
          log(f"Skipping ({result})")

      # RecBole
      for model in Model:
        if (tool and tool != recbole) or (algorithm and algorithm != model.name):
          continue

        current = f"[{recbole}][{model.name}][{dataset.name}][{size_formatted}]"
        log(f"Checking {current}")
        setdefault_nested(results, [OUTPUT_KEY, recbole, model.name, dataset.name, size])
        result = results[OUTPUT_KEY][recbole][model.name][dataset.name][size]
        if result is None or result < 0.0:
          log("Starting")
          stopwatch = time.time()

          result = safe_run(lambda: use_recbole(dataframe, dataset, f"{size_formatted}-{sampling.name}", model))
          result = round(result, FIGURES)
          results[OUTPUT_KEY][recbole][model.name][dataset.name][size] = result
          save_results(results, tag)

          log(f"Finished ({result})")
          show_memory()
          log("Elapsed", start=stopwatch, end="\n\n")
        else:
          log(f"Skipping ({result})")

      show_memory()
      log(f"Releasing dataframe from memory")
      del dataframe
      gc.collect()
      show_memory()

  log("Work complete")


if __name__ == '__main__':
  main()
