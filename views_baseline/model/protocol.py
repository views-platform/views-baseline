from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class BaselineModel(Protocol):
    targets: list[str]
    partition_dict: dict
    loa: str

    def fit(self, df: pd.DataFrame) -> BaselineModel: ...

    def predict(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int,
    ) -> pd.DataFrame: ...


@runtime_checkable
class DistributionalBaselineModel(Protocol):
    """Distributional baseline that returns dict[str, PredictionFrame] from predict()."""

    targets: list[str]
    partition_dict: dict
    loa: str
    distributional: bool

    def fit(self, df: pd.DataFrame) -> DistributionalBaselineModel: ...

    def predict(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int,
    ) -> dict: ...
