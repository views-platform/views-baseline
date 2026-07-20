from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame, PredictionFrame


@runtime_checkable
class BaselineModel(Protocol):
    """All baseline models return dict[str, PredictionFrame] from predict()."""

    targets: list[str]
    partition_dict: dict
    loa: str

    def fit(self, df: pd.DataFrame | FeatureFrame) -> BaselineModel: ...

    def predict(
        self,
        df: pd.DataFrame | FeatureFrame,
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

    def fit(self, df: pd.DataFrame | FeatureFrame) -> DistributionalBaselineModel: ...

    def predict(
        self,
        df: pd.DataFrame | FeatureFrame,
        sequence_number: int,
        output_length: int,
    ) -> dict[str, PredictionFrame]: ...
