from enum import Enum
from typing import TypeAlias, TypedDict


# Constants
OUTPUT_KEY = "OUTPUT"
META_KEY = "META"
MODE_KEY = "MODE"
SIZING_KEY = "SIZING"
SAMPLING_KEY = "SAMPLING"


# Enums
class LegendType(Enum):
  DATASETS = "datasets"
  ALGORITHMS = "algorithms"


# Types
## Results
ResultsSizeValue: TypeAlias = float | None # NDCG
ResultsSizeKey: TypeAlias = float | int # Fractional or absolute size
ResultsDatasetKey: TypeAlias = str
ResultsAlgorithmKey: TypeAlias = str
ResultsToolKey: TypeAlias = str
ResultsDataset: TypeAlias = dict[ResultsSizeKey, ResultsSizeValue]
ResultsAlgorithm: TypeAlias = dict[ResultsDatasetKey, ResultsDataset]
ResultsTool: TypeAlias = dict[ResultsAlgorithmKey, ResultsAlgorithm]
ResultsOutput: TypeAlias = dict[ResultsToolKey, ResultsTool]
ResultsSizingValue: TypeAlias = str
ResultsSamplingValue: TypeAlias = str

## Maxima
MaximumValue: TypeAlias = float
MaximaAlgorithm: TypeAlias = dict[ResultsDatasetKey, MaximumValue]
MaximaTool: TypeAlias = dict[ResultsAlgorithmKey, MaximaAlgorithm]
Maxima: TypeAlias = dict[ResultsToolKey, MaximaTool]

## Factors
FactorValue: TypeAlias = float | None
FactorKey: TypeAlias = int  # Size
DatasetKey: TypeAlias = str
DatasetValue: TypeAlias = dict[FactorKey, FactorValue]
Factors: TypeAlias = dict[DatasetKey, DatasetValue]

## Normalized
Normalized: TypeAlias = ResultsOutput
HalfNormalized: TypeAlias = Normalized

## Slopes
Slope: TypeAlias = float
SlopesAlgorithmValue: TypeAlias = dict[ResultsDatasetKey, Slope]
SlopesToolValue: TypeAlias = dict[ResultsAlgorithmKey, SlopesAlgorithmValue]
Slopes: TypeAlias = dict[ResultsToolKey, SlopesToolValue]
NormalizedSlopes: TypeAlias = Slopes
RawSlopes: TypeAlias = Slopes

## Elbow
ElbowPoints: TypeAlias = Maxima

## Gain
GainAlgorithm: TypeAlias = dict[ResultsDatasetKey, float]
GainTool: TypeAlias = dict[ResultsAlgorithmKey, GainAlgorithm]
Gain: TypeAlias = dict[ResultsToolKey, GainTool]

## Scatter Metadata
ScatterBinKey: TypeAlias = tuple[float, float]
ScatterBinValue: TypeAlias = dict[str, float | int]
ScatterBinDiffValue: TypeAlias = dict[float, ScatterBinValue]  # diff -> bin value
ScatterMetadata: TypeAlias = dict[ScatterBinKey, ScatterBinDiffValue]

# Classes
class ResultsMetaMode(TypedDict):
  SIZING: ResultsSizingValue
  SAMPLING: ResultsSamplingValue
class ResultsMeta(TypedDict):
  MODE: ResultsMetaMode
class Results(TypedDict):
  META: ResultsMeta
  OUTPUT: ResultsOutput

# d["OUTPUT"][tool][algorithm][dataset][size] -> NDCG
# d["META"]["MODE"] -> (sizing, sampling)
