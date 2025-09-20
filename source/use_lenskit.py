import pandas as pd
from lenskit.data import from_interactions_df, UserIDKey, ItemListCollection
from lenskit.splitting import SampleFrac, crossfold_users, sample_records
from lenskit.knn import ItemKNNScorer, ItemKNNConfig, UserKNNScorer, UserKNNConfig
from lenskit.als import BiasedMFScorer
from lenskit.pipeline import topn_pipeline, RecPipelineBuilder
from lenskit.batch import recommend
from lenskit.metrics import RunAnalysis, NDCG
from lenskit.basic import BiasScorer

from .constants import (
  KNN_K, KNN_MIN_K, FEATURES, COLUMN_NAMES, TRAINING_RECOMMENDATIONS, TEST_SIZE, RECOMMENDATIONS, SEED, Scorer
)


SCORERS = {
  Scorer.ITEM_KNN.name: lambda: ItemKNNScorer(ItemKNNConfig(
    max_nbrs=KNN_K, min_nbrs=KNN_MIN_K, feedback="explicit",
  )),
  Scorer.BIASED_MF.name: lambda: BiasedMFScorer(features=FEATURES),
  Scorer.BIAS.name: lambda: BiasScorer(),
  # Scorer.USER_KNN.name: lambda: UserKNNScorer(UserKNNConfig(
  #   max_nbrs=KNN_K, min_nbrs=KNN_MIN_K, feedback="explicit",
  # )),
}

def use_lenskit(df: pd.DataFrame, scorer: Scorer = Scorer.ITEM_KNN) -> float:
  dataset = from_interactions_df(
    df,
    user_col=COLUMN_NAMES["user_id"],
    item_col=COLUMN_NAMES["item_id"],
    rating_col=COLUMN_NAMES["rating"],
    timestamp_col=None,
  )

  get = SCORERS[scorer.name]
  model = get()

  pipe = topn_pipeline(model, n=TRAINING_RECOMMENDATIONS, predicts_ratings=True)

  all_test = ItemListCollection.empty([COLUMN_NAMES["user_id"]])
  all_recs = ItemListCollection.empty([COLUMN_NAMES["user_id"]])

  split = sample_records(dataset, size=int(dataset.interaction_count * TEST_SIZE), repeats=None, rng=SEED)
  all_test.add_from(split.test)
  fit = pipe
  fit.train(split.train)
  recs = recommend(fit, split.test.keys(), n=RECOMMENDATIONS)
  all_recs.add_from(recs)

  ran = RunAnalysis()
  ran.add_metric(NDCG(k=RECOMMENDATIONS, gain=COLUMN_NAMES["rating"]))
  results = ran.measure(all_recs, all_test)
  result = results.list_summary().iloc[0].iloc[0]

  return float(result)
