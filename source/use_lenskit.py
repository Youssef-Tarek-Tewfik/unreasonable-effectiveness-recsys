import pandas as pd
from typing import Callable
from lenskit.basic.popularity import PopScorer, PopConfig
from lenskit.knn import ItemKNNScorer, ItemKNNConfig
from lenskit.als import BiasedMFScorer, BiasedMFConfig, ImplicitMFScorer, ImplicitMFConfig
from lenskit.sklearn.svd import BiasedSVDScorer, BiasedSVDConfig
from lenskit.data import from_interactions_df, UserIDKey, ItemListCollection
from lenskit.splitting import SampleFrac, crossfold_users
from lenskit.pipeline import topn_pipeline, Component, RecPipelineBuilder
from lenskit.basic import UserTrainingHistoryLookup, UnratedTrainingItemsCandidateSelector
from lenskit.batch import recommend
from lenskit.metrics import RunAnalysis, NDCG
from lenskit.training import TrainingOptions

from .constants import (
  Scorer, Dataset, DATASET_FEEDBACK_EXPLICIT, COLUMN_NAMES, GPUS, RECOMMENDATIONS, SEED, TEST_SIZE, VALID_SIZE,
  PARTITIONS, LENSKIT_CONFIGS
)


SCORERS: dict[Scorer, Callable[[bool], Component]] = {
  Scorer.POP: lambda e: PopScorer(PopConfig(**LENSKIT_CONFIGS[Scorer.POP][e])),
  Scorer.ITEM_KNN: lambda e: ItemKNNScorer(ItemKNNConfig(**LENSKIT_CONFIGS[Scorer.ITEM_KNN][e])),
  Scorer.BIASED_MF: lambda e: BiasedMFScorer(BiasedMFConfig(**LENSKIT_CONFIGS[Scorer.BIASED_MF][e])),
  Scorer.IMPLICIT_MF: lambda e: ImplicitMFScorer(ImplicitMFConfig(**LENSKIT_CONFIGS[Scorer.IMPLICIT_MF][e])),
  Scorer.BIASED_SVD: lambda e: BiasedSVDScorer(BiasedSVDConfig(**LENSKIT_CONFIGS[Scorer.BIASED_SVD][e])),
}


def use_lenskit(df: pd.DataFrame, dataset: Dataset, scorer: Scorer = Scorer.ITEM_KNN) -> float:
  explicit = DATASET_FEEDBACK_EXPLICIT[dataset]
  data = from_interactions_df(
    df,
    user_col=COLUMN_NAMES["user_id"],
    item_col=COLUMN_NAMES["item_id"],
    rating_col=(COLUMN_NAMES["rating"] if explicit else None),
    timestamp_col=None,
  )

  get = SCORERS[scorer]
  model = get(explicit)
  selector = UnratedTrainingItemsCandidateSelector()
  lookup = UserTrainingHistoryLookup()

  # pipeline = topn_pipeline(model, n=RECOMMENDATIONS, predicts_ratings=False)
  builder = RecPipelineBuilder()
  builder.ranker(n=RECOMMENDATIONS)
  builder.scorer(model)
  builder.candidate_selector(selector)
  builder = builder.build().modify()
  try:
    builder.add_component("history-lookup", lookup)
  except ValueError:
    builder.replace_component("history-lookup", lookup)
  pipeline = builder.build()

  all_test = ItemListCollection.empty(UserIDKey)
  all_recs = ItemListCollection.empty(UserIDKey)
  for split in crossfold_users(data, PARTITIONS, SampleFrac(VALID_SIZE + TEST_SIZE, rng=SEED), rng=SEED):
    all_test.add_from(split.test)
    fit = pipeline.clone()
    fit.train(split.train, TrainingOptions(device=("cuda" if GPUS else None), rng=SEED))
    recs = recommend(fit, split.test.keys(), n=RECOMMENDATIONS)
    all_recs.add_from(recs)

  ran = RunAnalysis()
  ran.add_metric(NDCG(k=RECOMMENDATIONS, gain=(COLUMN_NAMES["rating"] if explicit else None)))
  results = ran.measure(all_recs, all_test)
  result = results.list_summary().iloc[0].iloc[0]

  return float(result)
