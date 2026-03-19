import sys
import csv
import time
import gc
from argparse import ArgumentParser

from .constants import Tool, Dataset, Sizing, Scorer, Model, MODE, SIZES_FRACTIONAL, SIZES_ABSOLUTE
from .load import load
from .results import load_results, save_results, setdefault_nested
from .sample import sample
from .utilities import safe_run, show_memory
from .use_lenskit import use_lenskit
from .use_recbole import use_recbole
from .logger import log
from .types import OUTPUT_KEY, ResultsSizeValue as Result


def main():
  csv.field_size_limit(sys.maxsize // 10)

  # Run options
  parallel = True
  default = (Tool.RECBOLE.name, Model.POP.name)
  sizes = SIZES_ABSOLUTE[0:]
  datasets = [
    Dataset.MYANIMELIST,
    Dataset.IPINYOU,
    Dataset.TMALL,
    Dataset.MOVIELENS,
    Dataset.MUSIC4ALL,
    Dataset.NETFLIX,
    Dataset.YAHOO,
    Dataset.ALIBABA,
    Dataset.GOODREADS,
    Dataset.LASTFM,
    Dataset.AMAZON,
  ]

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

  for desired_size in sizes:
    log("Desired size:", desired_size)
    for dataset in datasets:
      log("Dataset:", dataset.name)
      log("Loading")
      stopwatch = time.time()
      dataframe = load(dataset)
      log("Loading done")
      show_memory()
      log("Elapsed", start=stopwatch)
      log("Sampling")
      stopwatch = time.time()
      dataframe = sample(dataset, dataframe, desired_size)
      obtained_size = len(dataframe)
      log("Sampling done, obtained size:", obtained_size)
      show_memory()
      log("Elapsed", start=stopwatch)
      if sizing == Sizing.FRACTIONAL:
        formatted_size = f"{int(obtained_size * 100)}%"
      elif sizing == Sizing.ABSOLUTE:
        if obtained_size >= 1_000_000:
          formatted_size = f"{round(obtained_size / 1_000_000)}m"
        else:
          formatted_size = f"{round(obtained_size / 1_000)}k"

      # LensKit
      for scorer in Scorer:
        if (tool and tool != lenskit) or (algorithm and algorithm != scorer.name):
          continue

        current = f"[{lenskit}][{scorer.name}][{dataset.name}][{formatted_size}]"
        log(f"Checking {current}")
        setdefault_nested(results, [OUTPUT_KEY, lenskit, scorer.name, dataset.name, obtained_size])
        result = results[OUTPUT_KEY][lenskit][scorer.name][dataset.name][obtained_size]
        if result is None:
          log("Starting")
          stopwatch = time.time()

          result = safe_run(lambda: use_lenskit(dataframe, dataset, scorer))
          result = result if 0 <= result <= 1 else -1
          results[OUTPUT_KEY][lenskit][scorer.name][dataset.name][obtained_size] = result
          save_results(results, tag)

          log(f"Finished ({result:.2g})")
          show_memory()
          log("Elapsed", start=stopwatch, end="\n\n")
        else:
          log(f"Skipping ({result:.2g})")

      # RecBole
      for model in Model:
        if (tool and tool != recbole) or (algorithm and algorithm != model.name):
          continue

        current = f"[{recbole}][{model.name}][{dataset.name}][{formatted_size}]"
        log(f"Checking {current}")
        setdefault_nested(results, [OUTPUT_KEY, recbole, model.name, dataset.name, obtained_size])
        result = results[OUTPUT_KEY][recbole][model.name][dataset.name][obtained_size]
        if result is None:
          log("Starting")
          stopwatch = time.time()

          result = safe_run(lambda: use_recbole(dataframe, dataset, f"{formatted_size}-{sampling.name}", model))
          result = result if 0 <= result <= 1 else -1
          results[OUTPUT_KEY][recbole][model.name][dataset.name][obtained_size] = result
          save_results(results, tag)

          log(f"Finished ({result:.2g})")
          show_memory()
          log("Elapsed", start=stopwatch, end="\n\n")
        else:
          log(f"Skipping ({result:.2g})")

      show_memory()
      log(f"Releasing dataframe from memory")
      del dataframe
      gc.collect()
      show_memory()

  log("Work complete")


if __name__ == '__main__':
  main()
