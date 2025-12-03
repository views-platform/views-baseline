import pandas as pd
from types import SimpleNamespace

import pytest

from views_baseline.manager.baseline_manager import BaselineForecastingModelManager
from views_baseline.model.baseline import ZeroModel, LocfModel
import views_baseline.manager.baseline_manager as bm


# ---------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------


def make_dummy_df():
    """
    Simple MultiIndex dataframe with 2 entities and a range of months.
    Index names matter because the baselines read them from df.index.names[0/1].
    """
    time_idx_name = "month_id"
    entity_idx_name = "pg_id"

    times = list(range(110, 126))  # includes train and test period
    entities = [1, 2]

    index = pd.MultiIndex.from_product(
        [times, entities],
        names=[time_idx_name, entity_idx_name],
    )

    df = pd.DataFrame(index=index)

    # Deterministic targets
    df["y1"] = [
        t * 10 + e
        for t, e in zip(
            df.index.get_level_values(time_idx_name),
            df.index.get_level_values(entity_idx_name),
        )
    ]
    df["y2"] = [
        t * 100 + e
        for t, e in zip(
            df.index.get_level_values(time_idx_name),
            df.index.get_level_values(entity_idx_name),
        )
    ]
    return df


@pytest.fixture
def base_df():
    return make_dummy_df()


@pytest.fixture
def partition_dict():
    # test_start = 120, so train_end = 119
    return {"test": (120, 125)}


@pytest.fixture
def targets():
    return ["y1", "y2"]


def make_manager(config, partition_dict):
    """
    Create a BaselineForecastingModelManager instance without calling its __init__,
    and manually attach the attributes we need for our tests.
    """
    mgr = BaselineForecastingModelManager.__new__(BaselineForecastingModelManager)

    # Attach config and minimal path / data_loader stubs
    mgr.config = config
    mgr._model_path = SimpleNamespace(
        data_raw="dummy_raw_path",
        artifacts="dummy_artifacts_path",
    )
    mgr._data_loader = SimpleNamespace(partition_dict=partition_dict)

    # Stub out the sequence number resolver from the base class
    def fake_resolve_evaluation_sequence_number(eval_type: str) -> int:
        # Use config if present, otherwise default to 1 sequence
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
