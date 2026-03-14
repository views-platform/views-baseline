import pytest

from views_baseline.model.baseline import (
    AverageModel,
    ConflictologyModel,
    LocfModel,
    MixtureBaseline,
    ZeroModel,
)
from views_baseline.model.protocol import BaselineModel, DistributionalBaselineModel


@pytest.mark.parametrize(
    "Model,kwargs",
    [
        (ZeroModel, {"targets": ["y"], "partition_dict": {"test": (493, 540)}, "loa": "pg_id"}),
        (LocfModel, {"targets": ["y"], "partition_dict": {"test": (493, 540)}, "loa": "pg_id"}),
        (
            AverageModel,
            {
                "targets": ["y"],
                "window_months": 3,
                "partition_dict": {"test": (493, 540)},
                "loa": "pg_id",
            },
        ),
        (
            ConflictologyModel,
            {
                "targets": ["y"],
                "window_months": 3,
                "n_samples": 10,
                "partition_dict": {"test": (493, 540)},
                "loa": "pg_id",
            },
        ),
        (
            MixtureBaseline,
            {
                "targets": ["y"],
                "window_months": 3,
                "lambda_mix": 0.05,
                "n_samples": 10,
                "partition_dict": {"test": (493, 540)},
                "loa": "pg_id",
            },
        ),
    ],
)
def test_model_satisfies_baseline_protocol(Model, kwargs):
    instance = Model(**kwargs)
    assert isinstance(instance, BaselineModel)


def test_conflictology_satisfies_distributional_protocol():
    m = ConflictologyModel(
        targets=["y"],
        window_months=3,
        n_samples=10,
        partition_dict={"test": (493, 540)},
        loa="pg_id",
    )
    assert isinstance(m, DistributionalBaselineModel)


@pytest.mark.parametrize(
    "Model,kwargs",
    [
        (ZeroModel, {"targets": ["y"], "partition_dict": {"test": (493, 540)}, "loa": "pg_id"}),
        (LocfModel, {"targets": ["y"], "partition_dict": {"test": (493, 540)}, "loa": "pg_id"}),
        (
            AverageModel,
            {
                "targets": ["y"],
                "window_months": 3,
                "partition_dict": {"test": (493, 540)},
                "loa": "pg_id",
            },
        ),
    ],
)
def test_non_distributional_models_reject_distributional_protocol(Model, kwargs):
    assert not isinstance(Model(**kwargs), DistributionalBaselineModel)


def test_mixture_satisfies_distributional_protocol():
    m = MixtureBaseline(
        targets=["y"],
        window_months=3,
        lambda_mix=0.05,
        n_samples=10,
        partition_dict={"test": (493, 540)},
        loa="pg_id",
    )
    assert isinstance(m, DistributionalBaselineModel)
