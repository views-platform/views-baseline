from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd

if TYPE_CHECKING:
    from views_frames import PredictionFrame


@runtime_checkable
class BaselineModel(Protocol):
    """All baseline models return dict[str, PredictionFrame] from predict()."""

    targets: list[str]
    partition_dict: dict
    loa: str

    def fit(self, df: pd.DataFrame) -> BaselineModel: ...

    def predict(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int,
    ) -> dict[str, PredictionFrame]: ...


@runtime_checkable
class DistributionalBaselineModel(Protocol):
    """Distributional baseline with multi-sample y_pred (n_samples > 1)."""

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
    ) -> dict[str, PredictionFrame]: ...
