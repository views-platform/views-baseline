from views_baseline.model.protocol import BaselineModel, DistributionalBaselineModel
from views_baseline.model.baseline import (
    ZeroModel,
    LocfModel,
    AverageModel,
    ConflictologyModel,
)
import pytest


@pytest.mark.parametrize(
    "Model,kwargs",
    [
        (ZeroModel, {"targets": ["y"], "partition_dict": {"test": (493, 540)}, "loa": "pg_id"}),
        (LocfModel, {"targets": ["y"], "partition_dict": {"test": (493, 540)}, "loa": "pg_id"}),
        (AverageModel, {"targets": ["y"], "months": 3, "partition_dict": {"test": (493, 540)}, "loa": "pg_id"}),
        (ConflictologyModel, {"targets": ["y"], "months": 3, "partition_dict": {"test": (493, 540)}, "loa": "pg_id"}),
    ],
)
def test_model_satisfies_baseline_protocol(Model, kwargs):
    instance = Model(**kwargs)
    assert isinstance(instance, BaselineModel)


def test_conflictology_satisfies_distributional_protocol():
    m = ConflictologyModel(
        targets=["y"], months=3, partition_dict={"test": (493, 540)}, loa="pg_id"
    )
    assert isinstance(m, DistributionalBaselineModel)


@pytest.mark.parametrize(
    "Model,kwargs",
    [
        (ZeroModel, {"targets": ["y"], "partition_dict": {"test": (493, 540)}, "loa": "pg_id"}),
        (LocfModel, {"targets": ["y"], "partition_dict": {"test": (493, 540)}, "loa": "pg_id"}),
        (AverageModel, {"targets": ["y"], "months": 3, "partition_dict": {"test": (493, 540)}, "loa": "pg_id"}),
    ],
)
def test_non_distributional_models_reject_distributional_protocol(Model, kwargs):
    assert not isinstance(Model(**kwargs), DistributionalBaselineModel)
