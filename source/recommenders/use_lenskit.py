from enum import Enum
import pandas as pd
from lenskit.data import from_interactions_df, UserIDKey, ItemListCollection
from lenskit.splitting import SampleFrac, crossfold_users
from lenskit.knn import ItemKNNScorer
from lenskit.als import BiasedMFScorer
from lenskit.pipeline import topn_pipeline
from lenskit.batch import recommend
from lenskit.metrics import RunAnalysis, NDCG

from ..constants import TEST_SIZE, COLUMN_NAMES, RECOMMENDATIONS, PARTITIONS


class Scorer(Enum):
  ITEM_KNN = ItemKNNScorer(k=20) # Item K-Nearest Neighbours
  BIASED_MF = BiasedMFScorer(features=50) # Biased Matrix Factorization

def use_lenskit(df: pd.DataFrame, scorer: Scorer = Scorer.ITEM_KNN) -> float:
  dataset = from_interactions_df(
    df,
    user_col=COLUMN_NAMES["user_id"],
    item_col=COLUMN_NAMES["item_id"],
    rating_col=COLUMN_NAMES["rating"],
    timestamp_col=None,
  )

  model = scorer.value
  pipe = topn_pipeline(model, n=RECOMMENDATIONS)

  all_test = ItemListCollection.empty(key=UserIDKey)
  all_recs = ItemListCollection.empty(["user_id"])


  for split in crossfold_users(dataset, PARTITIONS, SampleFrac(TEST_SIZE)):
    all_test.add_from(split.test)

    fit = pipe.clone()
    fit.train(split.train)

    recs = recommend(fit, split.test.keys(), n=RECOMMENDATIONS)
    all_recs.add_from(recs)

  ran = RunAnalysis()
  ran.add_metric(NDCG())
  results = ran.measure(all_recs, all_test)
  result = results.list_metrics().mean()[0]

  return float(result)
