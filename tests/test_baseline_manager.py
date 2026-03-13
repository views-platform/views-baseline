import pandas as pd
import pickle
import re
from types import SimpleNamespace
from pathlib import Path

import pytest

from views_baseline.manager.baseline_manager import BaselineForecastingModelManager
from views_baseline.model.baseline import ZeroModel, LocfModel
import views_baseline.manager.baseline_manager as bm
from conftest import make_dummy_df


# ---------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def base_df():
    return make_dummy_df(time_range=range(110, 126))


@pytest.fixture
def partition_dict():
    # test_start = 120, so train_end = 119
    return {"test": (120, 125)}


def make_manager(config, partition_dict):
    """
    Create a BaselineForecastingModelManager instance without calling its __init__,
    and manually attach the attributes we need for our tests.
    """
    from views_pipeline_core.managers.configuration.configuration import ConfigurationManager

    mgr = BaselineForecastingModelManager.__new__(BaselineForecastingModelManager)

    # The base class __init__ creates _config_manager and _sweep.
    # Since we skip __init__, we must create them manually.
    mgr._config_manager = ConfigurationManager(
        config_hyperparameters={},
        config_deployment={},
        config_meta={},
        partition_dict={},
        config_sweep=None,
    )
    mgr._sweep = False

    # Now the property setter works
    mgr.config = config

    mgr._model_path = SimpleNamespace(
        data_raw=Path("dummy_raw_path"),
        artifacts=Path("dummy_artifacts_path"),
    )
    mgr._data_loader = SimpleNamespace(partition_dict=partition_dict)

    def fake_resolve_evaluation_sequence_number(eval_type: str) -> int:
        return config.get("sequence_numbers", 1)

    mgr._resolve_evaluation_sequence_number = fake_resolve_evaluation_sequence_number

    return mgr


# ---------------------------------------------------------------------
# Tests: _evaluate_model_artifact
# ---------------------------------------------------------------------


def test_manager_evaluate_uses_zero_model(monkeypatch, base_df, partition_dict, targets):
    """
    _evaluate_model_artifact should:
    - Use BaselineModelCatalog to get the correct baseline class (ZeroModel here)
    - Call .predict() for each sequence_number
    - Return a list of prediction DataFrames
    """
    # Make manager configuration
    config = {
        "run_type": "eval",
        "level": "pg_id",
        "algorithm": "ZeroModel",
        "targets": targets,
        "sequence_numbers": 2,  # we want two sequences
    }

    manager = make_manager(config, partition_dict)

    # read_dataframe should just return our in-memory df, no files
    monkeypatch.setattr(bm, "read_dataframe", lambda path: base_df)

    # Run evaluation
    preds_list = manager._evaluate_model_artifact(eval_type="temporal")

    assert isinstance(preds_list, list)
    assert len(preds_list) == 2

    # Expected: what ZeroModel would produce directly
    zero_model = ZeroModel(targets=targets, partition_dict=partition_dict, loa="pg_id")
    zero_model.fit(base_df)

    expected0 = zero_model.predict(df=base_df, sequence_number=0)
    expected1 = zero_model.predict(df=base_df, sequence_number=1)

    pd.testing.assert_frame_equal(preds_list[0], expected0)
    pd.testing.assert_frame_equal(preds_list[1], expected1)


def test_manager_evaluate_uses_locf_model(monkeypatch, base_df, partition_dict, targets):
    """
    Same as above but for LocfModel, just to make sure the manager is respecting
    config['algorithm'] and not hard-coding ZeroModel.
    """
    config = {
        "run_type": "eval",
        "level": "pg_id",
        "algorithm": "LocfModel",
        "targets": targets,
        "sequence_numbers": 1,
    }

    manager = make_manager(config, partition_dict)

    monkeypatch.setattr(bm, "read_dataframe", lambda path: base_df)

    preds_list = manager._evaluate_model_artifact(eval_type="temporal")

    assert isinstance(preds_list, list)
    assert len(preds_list) == 1

    locf = LocfModel(targets=targets, partition_dict=partition_dict, loa="pg_id")
    locf.fit(base_df)
    expected = locf.predict(df=base_df, sequence_number=0)

    pd.testing.assert_frame_equal(preds_list[0], expected)


# ---------------------------------------------------------------------
# Tests: _forecast_model_artifact
# ---------------------------------------------------------------------


def test_manager_forecast_uses_baseline_model(monkeypatch, base_df, partition_dict, targets):
    """
    _forecast_model_artifact should:
    - Load the viewser df (here via monkeypatched read_dataframe)
    - Instantiate the correct baseline model via the catalog
    - Call .fit() and then .predict(sequence_number=0)
    - Return that prediction DataFrame
    """
    config = {
        "run_type": "forecast",
        "level": "pg_id",
        "algorithm": "LocfModel",
        "targets": targets,
        # 'months' would be needed for Average/Conflictology, not Locf/Zero
    }

    manager = make_manager(config, partition_dict)

    # Monkeypatch read_dataframe to avoid any disk IO
    monkeypatch.setattr(bm, "read_dataframe", lambda path: base_df)

    forecasts = manager._forecast_model_artifact()

    locf = LocfModel(targets=targets, partition_dict=partition_dict, loa="pg_id")
    locf.fit(base_df)
    expected = locf.predict(df=base_df, sequence_number=0)

    pd.testing.assert_frame_equal(forecasts, expected)


def test_manager_forecast_respects_algorithm_choice(monkeypatch, base_df, partition_dict, targets):
    """
    Smoke test: switching algorithm in config should change the forecast
    (ZeroModel vs LocfModel should not match unless the data are degenerate).
    """
    # ZeroModel config
    config_zero = {
        "run_type": "forecast",
        "level": "pg_id",
        "algorithm": "ZeroModel",
        "targets": targets,
    }
    manager_zero = make_manager(config_zero, partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: base_df)
    forecasts_zero = manager_zero._forecast_model_artifact()

    # LocfModel config
    config_locf = {
        "run_type": "forecast",
        "level": "pg_id",
        "algorithm": "LocfModel",
        "targets": targets,
    }
    manager_locf = make_manager(config_locf, partition_dict)
    # Same monkeypatch, same df
    forecasts_locf = manager_locf._forecast_model_artifact()

    # Shapes should be same
    assert forecasts_zero.shape == forecasts_locf.shape

    # But at least one value should differ (for our deterministic dummy data)
    assert (forecasts_zero.values != forecasts_locf.values).any()


# ---------------------------------------------------------------------
# Tests: _setup_model_and_data
# ---------------------------------------------------------------------


def test_manager_setup_returns_model_and_data(monkeypatch, base_df, partition_dict, targets):
    """
    _setup_model_and_data should instantiate the correct model via the catalog,
    fit it on the data, and return (model, df).
    """
    config = {
        "run_type": "eval",
        "level": "pg_id",
        "algorithm": "ZeroModel",
        "targets": targets,
    }
    manager = make_manager(config, partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: base_df)

    model, df = manager._setup_model_and_data()

    assert isinstance(model, ZeroModel)
    assert df.shape == base_df.shape


# ---------------------------------------------------------------------
# Tests: _train_model_artifact
# ---------------------------------------------------------------------


def test_manager_train_saves_artifact(monkeypatch, base_df, partition_dict, targets, tmp_path):
    """
    _train_model_artifact should pickle the fitted model to artifacts/.
    """
    config = {
        "run_type": "calibration",
        "level": "pg_id",
        "algorithm": "ZeroModel",
        "targets": targets,
    }
    manager = make_manager(config, partition_dict)
    manager._model_path = SimpleNamespace(
        data_raw=Path("dummy_raw_path"),
        artifacts=tmp_path,
    )
    monkeypatch.setattr(bm, "read_dataframe", lambda path: base_df)

    model = manager._train_model_artifact()

    assert isinstance(model, ZeroModel)

    pkl_files = list(tmp_path.glob("calibration_model_*.pkl"))
    assert len(pkl_files) == 1

    # Filename matches expected pattern
    assert re.match(r"calibration_model_\d{8}_\d{6}\.pkl", pkl_files[0].name)

    # Unpickle and verify it's a valid model
    with open(pkl_files[0], "rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, ZeroModel)
    assert loaded.targets == targets
