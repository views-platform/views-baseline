from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

MANAGER_BASE_CONFIG = {
    "run_type": "calibration",
    "algorithm": "LocfModel",
    "level": "pgm",
    "time_steps": 36,
    "targets": ["synth_target"],
    "regression_targets": ["synth_target"],
    "regression_point_metrics": ["MSE"],
    "prediction_format": "prediction_frame",
    "steps": [*range(1, 37)],
}

MANAGER_PARTITION = {"test": (120, 125)}

ARTIFACT_TS = "20260101_120000"


def make_dummy_df(entity_id="priogrid_id", time_range=range(440, 540)):
    """
    Create a simple MultiIndex dataframe with 2 entities and a range of months.
    Index names matter because the models read them from df.index.names[0/1].
    """
    time_idx_name = "month_id"
    entity_idx_name = entity_id

    times = list(time_range)
    entities = [1, 2]

    index = pd.MultiIndex.from_product(
        [times, entities],
        names=[time_idx_name, entity_idx_name],
    )

    df = pd.DataFrame(index=index)

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


def make_dummy_ff(entity_id="priogrid_id", time_range=range(440, 540), targets=("y1", "y2"),
                  loa="pgm"):
    """Build a ``views_frames.FeatureFrame`` from :func:`make_dummy_df` via the input adapter.

    The dual-input counterpart of ``make_dummy_df``. Note the FeatureFrame is float32 by
    design (views_frames contract), so magnitudes are float32-rounded — the dummy targets
    here are small integers, hence float32-exact (see C-32).
    """
    from views_baseline.model.frames.input import to_feature_frame

    df = make_dummy_df(entity_id=entity_id, time_range=time_range)
    return to_feature_frame(df, loa=loa, targets=list(targets))


@pytest.fixture
def targets():
    return ["y1", "y2"]


@pytest.fixture
def manager_df():
    """Small DataFrame for manager-level tests (16 timesteps, 2 entities)."""
    return make_dummy_df(time_range=range(110, 126))


@pytest.fixture
def manager_partition_dict():
    """Partition dict aligned with manager_df: test_start=120, train_end=119."""
    return {"test": (120, 125)}


def make_manager(config, partition_dict):
    """
    Create a BaselineForecastingModelManager bypassing __init__,
    with manually attached attributes for testing.
    """
    from views_pipeline_core.managers.configuration.configuration import ConfigurationManager

    from views_baseline.manager.baseline_manager import BaselineForecastingModelManager

    mgr = BaselineForecastingModelManager.__new__(BaselineForecastingModelManager)
    mgr._config_manager = ConfigurationManager(
        config_hyperparameters={},
        config_deployment={},
        config_meta={},
        partition_dict={},
        config_sweep=None,
    )
    mgr._sweep = False
    mgr.config = config
    mgr._model_path = SimpleNamespace(
        artifacts=Path("dummy_artifacts_path"),
        get_latest_model_artifact_path=lambda run_type: Path(
            f"dummy_artifacts_path/{run_type}_model_20260101_120000.pkl"
        ),
    )
    mgr._data_loader = SimpleNamespace(partition_dict=partition_dict)
    mgr._cached_data_path = Path("dummy_raw_path") / "cached_df.parquet"

    def fake_resolve_evaluation_sequence_number(eval_type: str) -> int:
        return config.get("sequence_numbers", 1)

    mgr._resolve_evaluation_sequence_number = fake_resolve_evaluation_sequence_number
    return mgr


def assert_point_prediction_structure(
    result, df, targets, partition_dict, sequence_number, output_length
):
    """Assert common structural properties of prediction PredictionFrame dicts."""
    from views_frames import PredictionFrame

    test_start = partition_dict["test"][0]
    prediction_start = test_start + sequence_number
    prediction_end = prediction_start + output_length - 1

    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)

    train_end = test_start - 1
    time_idx = df.index.names[0]
    entity_idx = df.index.names[1]
    n_entities = (
        df.loc[df.index.get_level_values(time_idx) == train_end]
        .index.get_level_values(entity_idx)
        .nunique()
    )

    for target in targets:
        pf = result[target]
        assert isinstance(pf, PredictionFrame)
        assert pf.values.shape == (n_entities * output_length, 1)
        time_vals = pf.identifiers["time"]
        assert min(time_vals) == prediction_start
        assert max(time_vals) == prediction_end
