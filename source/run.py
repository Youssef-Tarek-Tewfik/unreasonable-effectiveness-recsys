from .load import Datasets, load
from .constants import SIZES
from .sample import sample
from .recommenders.use_lenskit import use_lenskit


df = load(Datasets.MOVIELENS)
df = sample(df, SIZES[0])
result = use_lenskit(df)
print(result)
