from typing import List, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class BaselineModel(Protocol):
    targets: List[str]
    partition_dict: dict
    loa: str

    def fit(self, df: pd.DataFrame) -> "BaselineModel": ...

    def predict(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int = 36,
    ) -> pd.DataFrame: ...


@runtime_checkable
class DistributionalBaselineModel(Protocol):
    """Distributional baseline that returns Dict[str, PredictionFrame] from predict()."""

    targets: List[str]
    partition_dict: dict
    loa: str
    distributional: bool

    def fit(self, df: pd.DataFrame) -> "DistributionalBaselineModel": ...

    def predict(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int = 36,
    ) -> dict: ...
