import pickle
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import make_dummy_ff, make_manager
from views_pipeline_core.modules.dataloaders.datafactory_contract import (
    DATA_FORMAT_DATAFRAME,
    DATA_FORMAT_FEATURE_FRAME,
)

import views_baseline.manager.baseline_manager as bm
from views_baseline.model.models.point import ZeroModel

# ---------------------------------------------------------------------
# Tests: _evaluate_model_artifact
# ---------------------------------------------------------------------


def test_manager_evaluate_uses_zero_model(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    config = {
        "run_type": "eval",
        "level": "pgm",
        "algorithm": "ZeroModel",
        "regression_targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
        "sequence_numbers": 2,
    }

    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    result = manager._evaluate_model_artifact(eval_type="temporal")

    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)
    for target_key in targets:
        assert isinstance(result[target_key], list)
        assert len(result[target_key]) == 2


def test_manager_evaluate_uses_locf_model(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    config = {
        "run_type": "eval",
        "level": "pgm",
        "algorithm": "LocfModel",
        "regression_targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
        "sequence_numbers": 1,
    }

    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    result = manager._evaluate_model_artifact(eval_type="temporal")

    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)
    for target_key in targets:
        assert len(result[target_key]) == 1


# ---------------------------------------------------------------------
# Tests: _forecast_model_artifact
# ---------------------------------------------------------------------


def test_manager_forecast_uses_baseline_model(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    from views_frames import PredictionFrame

    config = {
        "run_type": "forecast",
        "level": "pgm",
        "algorithm": "LocfModel",
        "regression_targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }

    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    result = manager._forecast_model_artifact()

    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)
    for target_key in targets:
        assert isinstance(result[target_key], PredictionFrame)


# ---------------------------------------------------------------------
# Tests: frame-native manager seam (issue #64)
#
# A model declaring data_format: feature_frame is served the pipeline-core
# FeatureFrame directory cache directly — no pandas in the manager. Every
# other model keeps the byte-identical read_dataframe path. These tests pin
# both read-sites (_setup_model_and_data at :60, _evaluate_sweep at :125),
# and assert the frame path NEVER touches read_dataframe (and vice versa).
# ---------------------------------------------------------------------


def _frame_config(algorithm="LocfModel", run_type="forecast", targets=("y1", "y2"), **extra):
    config = {
        "run_type": run_type,
        "level": "pgm",
        "algorithm": algorithm,
        "regression_targets": list(targets),
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }
    config.update(extra)
    return config


def _no_pandas(path):
    raise AssertionError("read_dataframe was called on a frame-declared model")


def test_manager_setup_flows_feature_frame_from_cache(
    monkeypatch, manager_partition_dict, targets
):
    """Frame-declared model: _setup_model_and_data loads the frame cache and a
    FeatureFrame — not a DataFrame — flows into fit()."""
    from views_frames import FeatureFrame

    manager = make_manager(_frame_config(), manager_partition_dict)
    manager._data_format = DATA_FORMAT_FEATURE_FRAME
    manager._cached_frame_path = Path("dummy_frame_cache")

    seen = {}

    def fake_load_frame_cache(path):
        seen["path"] = path
        return make_dummy_ff(time_range=range(110, 126))

    monkeypatch.setattr(bm, "read_dataframe", _no_pandas)
    monkeypatch.setattr(bm, "load_frame_cache", fake_load_frame_cache)

    model, source = manager._setup_model_and_data()

    assert isinstance(source, FeatureFrame)
    assert seen["path"] == Path("dummy_frame_cache")


def test_manager_forecast_via_frame_cache(
    monkeypatch, manager_partition_dict, targets
):
    """End-to-end forecast from the frame cache yields PredictionFrames with no pandas."""
    from views_frames import PredictionFrame

    manager = make_manager(_frame_config(run_type="forecast"), manager_partition_dict)
    manager._data_format = DATA_FORMAT_FEATURE_FRAME
    manager._cached_frame_path = Path("dummy_frame_cache")

    monkeypatch.setattr(bm, "read_dataframe", _no_pandas)
    monkeypatch.setattr(
        bm, "load_frame_cache", lambda path: make_dummy_ff(time_range=range(110, 126))
    )

    result = manager._forecast_model_artifact()

    assert set(result.keys()) == set(targets)
    for target_key in targets:
        assert isinstance(result[target_key], PredictionFrame)


def test_manager_evaluate_sweep_via_frame_cache(
    monkeypatch, manager_partition_dict, targets
):
    """The sweep read-site (:125) also serves the frame cache, no pandas."""
    manager = make_manager(
        _frame_config(run_type="eval", sequence_numbers=2), manager_partition_dict
    )
    manager._data_format = DATA_FORMAT_FEATURE_FRAME
    manager._cached_frame_path = Path("dummy_frame_cache")

    monkeypatch.setattr(bm, "read_dataframe", _no_pandas)
    monkeypatch.setattr(
        bm, "load_frame_cache", lambda path: make_dummy_ff(time_range=range(110, 126))
    )

    model, _ = manager._setup_model_and_data()
    result = manager._evaluate_sweep(eval_type="temporal", model=model)

    assert set(result.keys()) == set(targets)
    for target_key in targets:
        assert len(result[target_key]) == 2


def test_manager_frame_declared_missing_cache_fails_loud(
    monkeypatch, manager_partition_dict, targets
):
    """A frames-declared model whose cache was never populated must fail loud —
    never silently fall back to the pandas path (pipeline-core register C-214)."""
    manager = make_manager(_frame_config(), manager_partition_dict)
    manager._data_format = DATA_FORMAT_FEATURE_FRAME
    manager._cached_frame_path = None  # declared frame, but cache absent

    monkeypatch.setattr(
        bm, "read_dataframe", lambda path: pytest.fail("silent pandas fallback on frame model")
    )
    monkeypatch.setattr(
        bm, "load_frame_cache", lambda path: pytest.fail("loader reached with no cache path")
    )

    with pytest.raises(RuntimeError):
        manager._load_source()


def test_manager_legacy_format_never_calls_frame_loader(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    """An explicitly dataframe-declared model uses read_dataframe and never the
    frame loader — the branch does not degrade legacy behavior."""
    manager = make_manager(
        _frame_config(algorithm="ZeroModel", run_type="eval"), manager_partition_dict
    )
    manager._data_format = DATA_FORMAT_DATAFRAME

    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)
    monkeypatch.setattr(
        bm, "load_frame_cache", lambda path: pytest.fail("frame loader called on dataframe model")
    )

    model, source = manager._setup_model_and_data()

    assert isinstance(model, ZeroModel)
    assert source.shape == manager_df.shape


def test_manager_default_format_is_dataframe(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    """With no _data_format attribute set at all, the manager defaults to the
    pandas path (byte-identical legacy behavior)."""
    manager = make_manager(
        _frame_config(algorithm="ZeroModel", run_type="eval"), manager_partition_dict
    )
    # deliberately do NOT set manager._data_format

    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)
    monkeypatch.setattr(
        bm, "load_frame_cache", lambda path: pytest.fail("frame loader called by default")
    )

    _, source = manager._setup_model_and_data()

    assert source.shape == manager_df.shape


def test_manager_forecast_respects_algorithm_choice(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    config_zero = {
        "run_type": "forecast",
        "level": "pgm",
        "algorithm": "ZeroModel",
        "regression_targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }
    manager_zero = make_manager(config_zero, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)
    forecasts_zero = manager_zero._forecast_model_artifact()

    config_locf = {
        "run_type": "forecast",
        "level": "pgm",
        "algorithm": "LocfModel",
        "regression_targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }
    manager_locf = make_manager(config_locf, manager_partition_dict)
    forecasts_locf = manager_locf._forecast_model_artifact()

    target = targets[0]
    assert forecasts_zero[target].values.shape == forecasts_locf[target].values.shape
    assert not np.array_equal(forecasts_zero[target].values, forecasts_locf[target].values)


# ---------------------------------------------------------------------
# Tests: _setup_model_and_data
# ---------------------------------------------------------------------


def test_manager_setup_returns_model_and_data(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    config = {
        "run_type": "eval",
        "level": "pgm",
        "algorithm": "ZeroModel",
        "regression_targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }
    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    model, df = manager._setup_model_and_data()

    assert isinstance(model, ZeroModel)
    assert df.shape == manager_df.shape


# ---------------------------------------------------------------------
# Tests: _train_model_artifact
# ---------------------------------------------------------------------


def test_manager_train_saves_artifact(
    monkeypatch, manager_df, manager_partition_dict, targets, tmp_path
):
    config = {
        "run_type": "calibration",
        "level": "pgm",
        "algorithm": "ZeroModel",
        "regression_targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }
    manager = make_manager(config, manager_partition_dict)
    manager._model_path = SimpleNamespace(
        artifacts=tmp_path,
    )
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    model = manager._train_model_artifact()

    assert isinstance(model, ZeroModel)

    pkl_files = list(tmp_path.glob("calibration_model_*.pkl"))
    assert len(pkl_files) == 1

    assert re.match(r"calibration_model_\d{8}_\d{6}\.pkl", pkl_files[0].name)

    with open(pkl_files[0], "rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, ZeroModel)
    assert loaded.targets == targets


# ---------------------------------------------------------------------
# Tests: distributional model paths (unchanged — already returned dict)
# ---------------------------------------------------------------------


def test_manager_evaluate_distributional_model(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    config = {
        "run_type": "eval",
        "level": "pgm",
        "algorithm": "ConflictologyModel",
        "regression_targets": targets,
        "window_months": 6,
        "n_samples": 64,
        "seed": 42,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
        "sequence_numbers": 2,
    }

    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    result = manager._evaluate_model_artifact(eval_type="temporal")

    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)
    for target_key in targets:
        assert isinstance(result[target_key], list)
        assert len(result[target_key]) == 2


def test_manager_evaluate_sweep(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    config = {
        "run_type": "eval",
        "level": "pgm",
        "algorithm": "LocfModel",
        "regression_targets": targets,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
        "sequence_numbers": 2,
    }

    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    model, _ = manager._setup_model_and_data()
    result = manager._evaluate_sweep(eval_type="temporal", model=model)

    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)
    for target_key in targets:
        assert isinstance(result[target_key], list)
        assert len(result[target_key]) == 2


def test_manager_forecast_distributional_model(
    monkeypatch, manager_df, manager_partition_dict, targets
):
    from views_frames import PredictionFrame

    config = {
        "run_type": "forecast",
        "level": "pgm",
        "algorithm": "ConflictologyModel",
        "regression_targets": targets,
        "window_months": 6,
        "n_samples": 64,
        "seed": 42,
        "steps": [*range(1, 37)],
        "time_steps": 36,
        "prediction_format": "prediction_frame",
    }

    manager = make_manager(config, manager_partition_dict)
    monkeypatch.setattr(bm, "read_dataframe", lambda path: manager_df)

    result = manager._forecast_model_artifact()

    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)
    for target_key in targets:
        assert isinstance(result[target_key], PredictionFrame)
