import pytest

from views_baseline.model.models.distributional import (
    ConflictologyModel,
    EmpiricalKDEMixtureConflictology,
    InterpolatedQuantileConflictology,
    KDEConflictology,
    MatryoshkaConflictology,
    MixtureBaseline,
    ParametricConflictology,
    ParametricHurdleConflictology,
    PooledConflictology,
    RollingConflictology,
)
from views_baseline.model.models.point import AverageModel, LocfModel, ZeroModel
from views_baseline.model.protocol import BaselineModel, DistributionalBaselineModel

_PARTITION = {"test": (493, 540)}
_BASE = {"targets": ["y"], "partition_dict": _PARTITION, "loa": "pg_id"}

_POINT_MODELS = [
    (ZeroModel, _BASE),
    (LocfModel, _BASE),
    (AverageModel, {**_BASE, "window_months": 3}),
]

_DISTRIBUTIONAL_MODELS = [
    (ConflictologyModel, {**_BASE, "window_months": 3, "n_samples": 10}),
    (MixtureBaseline, {**_BASE, "window_months": 3, "lambda_mix": 0.05, "n_samples": 10}),
    (
        ParametricConflictology,
        {**_BASE, "window_months": 3, "n_samples": 10, "family": "nb", "transform": "none"},
    ),
    (
        ParametricHurdleConflictology,
        {**_BASE, "window_months": 3, "n_samples": 10, "family": "gumbel", "transform": "log1p"},
    ),
    (InterpolatedQuantileConflictology, {**_BASE, "window_months": 3, "n_samples": 10}),
    (RollingConflictology, {**_BASE, "window_months": 3, "n_samples": 10}),
    (PooledConflictology, {**_BASE, "window_months": 3, "n_samples": 10, "pool_by": "map"}),
    (
        MatryoshkaConflictology,
        {**_BASE, "n_samples": 10, "country_column": "c_id", "backtest_cutoffs": 1,
         "backtest_horizon": 6, "stability_refits": False},
    ),
    (KDEConflictology, {**_BASE, "window_months": 3, "n_samples": 10, "bandwidth": "sj"}),
    (
        EmpiricalKDEMixtureConflictology,
        {**_BASE, "window_months": 3, "n_samples": 10, "alpha": 0.5, "bandwidth": 1.0},
    ),
]

_ALL_MODELS = _POINT_MODELS + _DISTRIBUTIONAL_MODELS


@pytest.mark.parametrize("Model,kwargs", _ALL_MODELS)
def test_model_satisfies_baseline_protocol(Model, kwargs):
    instance = Model(**kwargs)
    assert isinstance(instance, BaselineModel)


@pytest.mark.parametrize("Model,kwargs", _DISTRIBUTIONAL_MODELS)
def test_distributional_models_satisfy_distributional_protocol(Model, kwargs):
    model = Model(**kwargs)
    assert isinstance(model, DistributionalBaselineModel)
    # isinstance on a runtime-checkable Protocol only checks the attribute EXISTS; pin its value.
    assert model.distributional is True


@pytest.mark.parametrize("Model,kwargs", _POINT_MODELS)
def test_non_distributional_models_reject_distributional_protocol(Model, kwargs):
    assert not isinstance(Model(**kwargs), DistributionalBaselineModel)
