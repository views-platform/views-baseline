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
class DistributionalBaselineModel(BaselineModel, Protocol):
    def predict_prediction_frame(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int = 36,
    ) -> dict: ...
