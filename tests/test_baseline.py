import pandas as pd
import numpy as np

import pytest

from views_baseline.model.baseline import (
    ZeroModel,
    LocfModel,
    AverageModel,
    ConflictologyModel,
    MixtureBaseline,
)
from conftest import make_dummy_df


@pytest.fixture
def partition_dict():
    return {"test": (493, 540)}


@pytest.fixture
def base_df_pgm():
    return make_dummy_df(entity_id="pg_id")

@pytest.fixture
def base_df_cm():
    return make_dummy_df(entity_id="country_id")


# -----------------------------------------------------------------------
# ZeroModel
# -----------------------------------------------------------------------


def test_zero_model_predicts_zeros_pgm(base_df_pgm, partition_dict, targets):
    model = ZeroModel(targets=targets, partition_dict=partition_dict, loa="pg_id")
    model.fit(base_df_pgm)
    output_length = 36
    preds = model.predict(df=base_df_pgm, sequence_number=0, output_length=output_length)

    time_idx, entity_idx = base_df_pgm.index.names
    test_start, _ = partition_dict["test"]
    train_end = test_start - 1
    train_times = (
        base_df_pgm.index.get_level_values(time_idx)
        [base_df_pgm.index.get_level_values(time_idx) < test_start]
    )
    assert train_times.max() == train_end

    prediction_start = test_start  # sequence_number = 0
    prediction_end = prediction_start + output_length - 1

    # Check columns
    assert list(preds.columns) == [f"pred_{t}" for t in targets]

    # Check index names and time range
    assert preds.index.names == [time_idx, entity_idx]
    assert preds.index.get_level_values(time_idx).min() == test_start
    assert preds.index.get_level_values(time_idx).max() == test_start + output_length - 1
    # ----- assert time ids in preds -----
    pred_time_ids = preds.index.get_level_values(time_idx).unique().tolist()
    assert pred_time_ids == list(range(prediction_start, prediction_end + 1))

    assert base_df_pgm.index.names[1] == "pg_id"

    # All zeros
    assert (preds.values == 0.0).all()

def test_zero_model_predicts_zeros_cm(base_df_cm, partition_dict, targets):
    model = ZeroModel(targets=targets, partition_dict=partition_dict, loa="country_id")
    model.fit(base_df_cm)
    output_length = 36
    preds = model.predict(df=base_df_cm, sequence_number=0, output_length=output_length)

    time_idx, entity_idx = base_df_cm.index.names
    test_start, _ = partition_dict["test"]
    train_end = test_start - 1
    train_times = (
        base_df_cm.index.get_level_values(time_idx)
        [base_df_cm.index.get_level_values(time_idx) < test_start]
    )
    assert train_times.max() == train_end

    prediction_start = test_start  # sequence_number = 0
    prediction_end = prediction_start + output_length - 1

    # Check columns
    assert list(preds.columns) == [f"pred_{t}" for t in targets]

    # Check index names and time range
    assert preds.index.names == [time_idx, entity_idx]
    assert preds.index.get_level_values(time_idx).min() == test_start
    assert preds.index.get_level_values(time_idx).max() == test_start + output_length - 1
    pred_time_ids = preds.index.get_level_values(time_idx).unique().tolist()
    assert pred_time_ids == list(range(prediction_start, prediction_end + 1))
    assert base_df_cm.index.names[1] == "country_id"

    # All zeros
    assert (preds.values == 0.0).all()


def test_zero_model_respects_sequence_number(base_df_pgm, partition_dict, targets):
    model = ZeroModel(targets=targets, partition_dict=partition_dict, loa="pg_id")
    model.fit(base_df_pgm)

    test_start, _ = partition_dict["test"]
    seq_num = 2
    output_length = 36

    preds = model.predict(df=base_df_pgm, sequence_number=seq_num, output_length=output_length)
    time_idx = base_df_pgm.index.names[0]

    assert preds.index.get_level_values(time_idx).min() == test_start + seq_num
    assert preds.index.get_level_values(time_idx).max() == test_start + seq_num + output_length - 1


# -----------------------------------------------------------------------
# LocfModel
# -----------------------------------------------------------------------


def test_locf_model_uses_last_observation(base_df_pgm, partition_dict, targets):
    model = LocfModel(targets=targets, partition_dict=partition_dict, loa="pg_id")
    model.fit(base_df_pgm)
    output_length = 36

    time_idx, entity_idx = base_df_pgm.index.names
    test_start, _ = partition_dict["test"]
    train_end = test_start - 1
    train_times = (
        base_df_pgm.index.get_level_values(time_idx)
        [base_df_pgm.index.get_level_values(time_idx) < test_start]
    )
    assert train_times.max() == train_end

    prediction_start = test_start  # sequence_number = 0
    prediction_end = prediction_start + output_length - 1

    preds = model.predict(df=base_df_pgm, sequence_number=0, output_length=output_length)

    # Expected last obs per entity from training part (< test_start)
    train_df = base_df_pgm[base_df_pgm.index.get_level_values(time_idx) < test_start]
    expected_last = train_df.groupby(level=entity_idx)[targets].last()

    # Check predictions align with last observation for each entity
    for ent in expected_last.index:
        for t in range(test_start, test_start + 2):
            row = preds.loc[(t, ent)]
            for target in targets:
                assert row[f"pred_{target}"] == expected_last.loc[ent, target]

    pred_time_ids = preds.index.get_level_values(time_idx).unique().tolist()
    assert pred_time_ids == list(range(prediction_start, prediction_end + 1))
    assert preds.index.names == [time_idx, entity_idx]
    assert preds.index.get_level_values(time_idx).min() == test_start
    assert preds.index.get_level_values(time_idx).max() == test_start + output_length - 1


def test_locf_model_respects_sequence_number(base_df_pgm, partition_dict, targets):
    model = LocfModel(targets=targets, partition_dict=partition_dict, loa="pg_id")
    model.fit(base_df_pgm)

    test_start, _ = partition_dict["test"]
    seq_num = 1
    output_length = 36
    preds = model.predict(df=base_df_pgm, sequence_number=seq_num, output_length=output_length)

    time_idx = base_df_pgm.index.names[0]
    assert preds.index.get_level_values(time_idx).min() == test_start + seq_num
    assert preds.index.get_level_values(time_idx).max() == test_start + seq_num + output_length - 1


def test_locf_model_time_idx_is_not_tuple_before_fit(partition_dict, targets):
    model = LocfModel(targets=targets, partition_dict=partition_dict, loa="pg_id")
    assert model.time_idx is None


def test_locf_model_fit_handles_unsorted_data(partition_dict, targets):
    time_idx_name, entity_idx_name = "month_id", "pg_id"
    rows = []
    for t in [490, 492, 491]:  # deliberately unsorted
        rows.append({time_idx_name: t, entity_idx_name: 1, "y1": t * 10 + 1, "y2": t * 100 + 1})
    rows.append({time_idx_name: 492, entity_idx_name: 2, "y1": 4921, "y2": 49202})
    df = pd.DataFrame(rows).set_index([time_idx_name, entity_idx_name])

    model = LocfModel(targets=targets, partition_dict=partition_dict, loa="pg_id")
    model.fit(df)
    # Must use month 492 (temporally last), not 491 (positionally last)
    assert model.last_observations.loc[1, "y1"] == 492 * 10 + 1
    assert model.last_observations.loc[1, "y2"] == 492 * 100 + 1


# -----------------------------------------------------------------------
# AverageModel
# -----------------------------------------------------------------------


def test_average_model_uses_mean_of_last_n_months(base_df_pgm, partition_dict, targets):
    months = 3
    model = AverageModel(
        targets=targets,
        months=months,
        partition_dict=partition_dict,
        loa="pg_id",
    )

    output_length = 36
    model.fit(base_df_pgm)

    time_idx, entity_idx = base_df_pgm.index.names
    test_start, _ = partition_dict["test"]
    train_end = test_start - 1
    train_times = (
        base_df_pgm.index.get_level_values(time_idx)
        [base_df_pgm.index.get_level_values(time_idx) < test_start]
    )
    assert train_times.max() == train_end

    prediction_start = test_start  # sequence_number = 0
    prediction_end = prediction_start + output_length - 1

    preds = model.predict(df=base_df_pgm, sequence_number=0, output_length=output_length)

    # Expected means: per entity, mean of last `months` rows in train
    train_df = base_df_pgm[base_df_pgm.index.get_level_values(time_idx) < test_start]
    train_df = train_df.sort_index(level=[entity_idx, time_idx])
    expected_means = (
        train_df.groupby(level=entity_idx, group_keys=False)
        .apply(lambda g: g.tail(months)[targets].mean())
    )

    # Check predictions equal these means for each entity and time
    for ent in expected_means.index:
        for t in range(test_start, test_start + 2):
            row = preds.loc[(t, ent)]
            for target in targets:
                assert row[f"pred_{target}"] == pytest.approx(expected_means.loc[ent, target])

    pred_time_ids = preds.index.get_level_values(time_idx).unique().tolist()
    assert pred_time_ids == list(range(prediction_start, prediction_end + 1))
    assert preds.index.names == [time_idx, entity_idx]
    assert preds.index.get_level_values(time_idx).min() == test_start
    assert preds.index.get_level_values(time_idx).max() == test_start + output_length - 1


def test_average_model_respects_sequence_number(base_df_pgm, partition_dict, targets):
    months = 2
    model = AverageModel(
        targets=targets,
        months=months,
        partition_dict=partition_dict,
        loa="pg_id",
    )
    model.fit(base_df_pgm)

    test_start, _ = partition_dict["test"]
    seq_num = 2
    output_length = 36

    preds = model.predict(df=base_df_pgm, sequence_number=seq_num, output_length=output_length)
    time_idx = base_df_pgm.index.names[0]

    assert preds.index.get_level_values(time_idx).min() == test_start + seq_num
    assert preds.index.get_level_values(time_idx).max() == test_start + seq_num + output_length - 1


# -----------------------------------------------------------------------
# ConflictologyModel (distribution baseline)
# -----------------------------------------------------------------------


def test_conflictology_model_resamples_from_history(base_df_pgm, partition_dict, targets):
    months = 4
    n_samples = 64
    model = ConflictologyModel(
        targets=targets,
        months=months,
        partition_dict=partition_dict,
        loa="pg_id",
        n_samples=n_samples,
        seed=42,
    )
    model.fit(base_df_pgm)

    test_start, _ = partition_dict["test"]
    output_length = 36
    preds = model.predict(df=base_df_pgm, sequence_number=0, output_length=output_length)

    time_idx, entity_idx = base_df_pgm.index.names

    # Basic shape checks
    assert preds.index.names == [time_idx, entity_idx]
    assert list(preds.columns) == [f"pred_{t}" for t in targets]

    # Train part
    train_df = base_df_pgm[base_df_pgm.index.get_level_values(time_idx) < test_start]
    train_df = train_df.sort_index(level=[entity_idx, time_idx])

    for ent in train_df.index.get_level_values(entity_idx).unique():
        ent_history = train_df.xs(ent, level=entity_idx).tail(months)

        for target in targets:
            history_values = set(ent_history[target].tolist())

            first_time = test_start
            cell_value = preds.loc[(first_time, ent), f"pred_{target}"]

            # 1) each prediction is a list (resampled distribution)
            assert isinstance(cell_value, list)

            # 2) list has n_samples entries (not months)
            assert len(cell_value) == n_samples

            # 3) all sampled values come from the history window
            assert set(cell_value).issubset(history_values)

            # 4) all entries are numeric
            assert all(isinstance(x, (int, float, np.integer, np.floating)) for x in cell_value)


def test_conflictology_model_respects_sequence_number(base_df_pgm, partition_dict, targets):
    months = 3
    n_samples = 32
    model = ConflictologyModel(
        targets=targets,
        months=months,
        partition_dict=partition_dict,
        loa="pg_id",
        n_samples=n_samples,
        seed=42,
    )
    model.fit(base_df_pgm)

    time_idx, entity_idx = base_df_pgm.index.names
    test_start, _ = partition_dict["test"]

    seq_num = 2
    output_length = 4

    preds = model.predict(
        df=base_df_pgm,
        sequence_number=seq_num,
        output_length=output_length,
    )

    # --- prediction window checks ---
    prediction_start = test_start + seq_num
    prediction_end = prediction_start + output_length - 1

    assert preds.index.get_level_values(time_idx).min() == prediction_start
    assert preds.index.get_level_values(time_idx).max() == prediction_end

    # --- history window checks (samples come from history, not shifted) ---
    train_end = test_start - 1
    history_start = train_end - (months - 1)

    df_hist = base_df_pgm[
        (base_df_pgm.index.get_level_values(time_idx) >= history_start)
        & (base_df_pgm.index.get_level_values(time_idx) <= train_end)
    ].sort_index(level=[entity_idx, time_idx])

    for ent in df_hist.index.get_level_values(entity_idx).unique():
        ent_hist = df_hist.xs(ent, level=entity_idx)
        for target in targets:
            history_values = set(ent_hist[target].tolist())

            first_time = prediction_start
            cell_value = preds.loc[(first_time, ent), f"pred_{target}"]

            assert isinstance(cell_value, list)
            assert len(cell_value) == n_samples
            assert set(cell_value).issubset(history_values)


def test_conflictology_model_predict_prediction_frame(base_df_pgm, partition_dict, targets):
    from views_pipeline_core.data.prediction_frame import PredictionFrame

    months = 4
    n_samples = 64
    model = ConflictologyModel(
        targets=targets,
        months=months,
        partition_dict=partition_dict,
        loa="pg_id",
        n_samples=n_samples,
        seed=42,
    )
    model.fit(base_df_pgm)

    test_start, _ = partition_dict["test"]
    output_length = 5
    result = model.predict_prediction_frame(
        df=base_df_pgm, sequence_number=0, output_length=output_length,
    )

    time_idx, entity_idx = base_df_pgm.index.names
    n_entities = base_df_pgm.loc[
        base_df_pgm.index.get_level_values(time_idx) == test_start - 1
    ].index.get_level_values(entity_idx).nunique()

    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)

    for target in targets:
        pf = result[target]
        assert isinstance(pf, PredictionFrame)
        assert pf.y_pred.shape == (n_entities * output_length, n_samples)
        assert len(pf.identifiers["time"]) == n_entities * output_length
        assert len(pf.identifiers["unit"]) == n_entities * output_length

    # Verify values match the DataFrame path
    df_preds = model.predict(df=base_df_pgm, sequence_number=0, output_length=output_length)
    for target in targets:
        pf = result[target]
        for i in range(pf.n_rows):
            tid = pf.identifiers["time"][i]
            uid = pf.identifiers["unit"][i]
            expected = df_preds.loc[(tid, uid), f"pred_{target}"]
            assert list(pf.y_pred[i]) == pytest.approx(expected)


def test_conflictology_matches_mixture_lambda_zero(base_df_pgm, partition_dict, targets):
    """
    ConflictologyModel and MixtureBaseline(lambda_mix=0) should draw from
    the same local history pool per entity/target.
    """
    window = 4
    n_samples = 128

    conf = ConflictologyModel(
        targets=targets, months=window, partition_dict=partition_dict,
        loa="pg_id", n_samples=n_samples, seed=42,
    )
    conf.fit(base_df_pgm)

    mix = MixtureBaseline(
        targets=targets, window_months=window, lambda_mix=0.0,
        n_samples=n_samples, partition_dict=partition_dict,
        loa="pg_id", seed=99,  # different seed — we test pools, not samples
    )
    mix.fit(base_df_pgm)

    # 1) Same entities
    assert set(conf.hist_per_entity.keys()) == set(mix.local_pool.keys())

    # 2) Same source pool values per entity per target
    for cid in conf.hist_per_entity:
        for t in targets:
            np.testing.assert_array_equal(
                np.sort(conf.hist_per_entity[cid][t]),
                np.sort(mix.local_pool[cid][t]),
            )

    # 3) Predictions have same shape and all values come from the pool
    output_length = 5
    conf_preds = conf.predict(df=base_df_pgm, sequence_number=0, output_length=output_length)
    mix_preds = mix.predict(df=base_df_pgm, sequence_number=0, output_length=output_length)

    assert conf_preds.shape == mix_preds.shape

    for col in conf_preds.columns:
        for idx in conf_preds.index:
            conf_cell = conf_preds.loc[idx, col]
            mix_cell = mix_preds.loc[idx, col]
            assert len(conf_cell) == n_samples
            assert len(mix_cell) == n_samples


# -----------------------------------------------------------------------
# build_prediction_grid helper
# -----------------------------------------------------------------------


def test_build_prediction_grid_shape_and_values():
    from views_baseline.model.helpers import build_prediction_grid

    df = build_prediction_grid(
        time_idx="month_id",
        entity_idx="pg_id",
        loa_ids=[1, 2],
        time_ids=[100, 101],
        targets=["y1"],
        value_fn=lambda cid, t: float(cid),
    )
    assert df.index.names == ["month_id", "pg_id"]
    assert list(df.columns) == ["pred_y1"]
    assert len(df) == 4  # 2 entities x 2 times
    assert df.loc[(100, 2), "pred_y1"] == 2.0


def test_build_prediction_grid_empty():
    from views_baseline.model.helpers import build_prediction_grid

    df = build_prediction_grid(
        time_idx="month_id",
        entity_idx="pg_id",
        loa_ids=[],
        time_ids=[100, 101],
        targets=["y1"],
        value_fn=lambda cid, t: 0.0,
    )
    assert len(df) == 0
    assert list(df.columns) == ["pred_y1"]
    assert df.index.names == ["month_id", "pg_id"]


# -----------------------------------------------------------------------
# MixtureBaseline
# -----------------------------------------------------------------------


def make_mixture_df():
    """
    DataFrame with 3 entities: two with positive values, one all-zero.
    Entity 3 (all-zero) tests the zero-probability trap.
    """
    time_idx_name = "month_id"
    entity_idx_name = "pg_id"
    times = list(range(440, 540))
    rows = []
    for t in times:
        # Entity 1: positive values
        rows.append({time_idx_name: t, entity_idx_name: 1, "y1": float(t - 439), "y2": float((t - 439) * 2)})
        # Entity 2: positive values (different scale)
        rows.append({time_idx_name: t, entity_idx_name: 2, "y1": float(t - 439) * 0.5, "y2": float(t - 439) * 0.1})
        # Entity 3: all zeros
        rows.append({time_idx_name: t, entity_idx_name: 3, "y1": 0.0, "y2": 0.0})
    df = pd.DataFrame(rows).set_index([time_idx_name, entity_idx_name]).sort_index()
    return df


@pytest.fixture
def mixture_df():
    return make_mixture_df()


def test_mixture_fit_extracts_local_pool(mixture_df, partition_dict, targets):
    model = MixtureBaseline(
        targets=targets, window_months=4, lambda_mix=0.05, n_samples=10,
        partition_dict=partition_dict, loa="pg_id",
    )
    model.fit(mixture_df)

    # Entity 1 local pool should be last 4 training values
    test_start = partition_dict["test"][0]
    train_times = [t for t in range(test_start - 4, test_start)]
    expected_y1 = [float(t - 439) for t in train_times]

    assert 1 in model.local_pool
    np.testing.assert_array_equal(model.local_pool[1]["y1"], expected_y1)

    # Entity 3 local pool should be all zeros
    assert 3 in model.local_pool
    np.testing.assert_array_equal(model.local_pool[3]["y1"], [0.0, 0.0, 0.0, 0.0])


def test_mixture_fit_extracts_global_pool(mixture_df, partition_dict, targets):
    model = MixtureBaseline(
        targets=targets, window_months=4, lambda_mix=0.05, n_samples=10,
        partition_dict=partition_dict, loa="pg_id",
    )
    model.fit(mixture_df)

    # Global pool should contain only positive values
    assert "y1" in model.global_pool
    assert len(model.global_pool["y1"]) > 0
    assert all(v > 0 for v in model.global_pool["y1"])

    # Entity 3 is all-zero, so it contributes nothing to global pool
    # Entities 1 and 2 contribute all their training values (all positive)
    test_start = partition_dict["test"][0]
    n_train = test_start - 440
    # Entity 1: all positive, Entity 2: all positive → 2 * n_train values
    assert len(model.global_pool["y1"]) == 2 * n_train


def test_mixture_fit_global_pool_causal(mixture_df, partition_dict, targets):
    model = MixtureBaseline(
        targets=targets, window_months=4, lambda_mix=0.05, n_samples=10,
        partition_dict=partition_dict, loa="pg_id",
    )
    model.fit(mixture_df)

    test_start = partition_dict["test"][0]
    # Max value in global pool for y1 should correspond to train_end
    # Entity 1 has y1 = t - 439, so max should be (test_start - 1) - 439
    max_expected = float(test_start - 1 - 439)
    assert max(model.global_pool["y1"]) <= max_expected


def test_mixture_fit_returns_self(mixture_df, partition_dict, targets):
    model = MixtureBaseline(
        targets=targets, window_months=4, lambda_mix=0.05, n_samples=10,
        partition_dict=partition_dict, loa="pg_id",
    )
    assert model.fit(mixture_df) is model


def test_mixture_predict_shape(mixture_df, partition_dict, targets):
    model = MixtureBaseline(
        targets=targets, window_months=4, lambda_mix=0.05, n_samples=10,
        partition_dict=partition_dict, loa="pg_id",
    )
    model.fit(mixture_df)
    output_length = 5
    preds = model.predict(df=mixture_df, sequence_number=0, output_length=output_length)

    time_idx, entity_idx = mixture_df.index.names
    assert preds.index.names == [time_idx, entity_idx]
    assert list(preds.columns) == [f"pred_{t}" for t in targets]
    # 3 entities x 5 time steps
    assert len(preds) == 3 * output_length


def test_mixture_predict_cells_are_lists(mixture_df, partition_dict, targets):
    n_samples = 16
    model = MixtureBaseline(
        targets=targets, window_months=4, lambda_mix=0.05, n_samples=n_samples,
        partition_dict=partition_dict, loa="pg_id",
    )
    model.fit(mixture_df)
    preds = model.predict(df=mixture_df, sequence_number=0, output_length=3)

    for col in preds.columns:
        for val in preds[col]:
            assert isinstance(val, list), f"Expected list, got {type(val)}"
            assert len(val) == n_samples


def test_mixture_predict_respects_sequence_number(mixture_df, partition_dict, targets):
    model = MixtureBaseline(
        targets=targets, window_months=4, lambda_mix=0.05, n_samples=10,
        partition_dict=partition_dict, loa="pg_id",
    )
    model.fit(mixture_df)

    test_start = partition_dict["test"][0]
    seq_num = 2
    output_length = 4
    preds = model.predict(df=mixture_df, sequence_number=seq_num, output_length=output_length)

    time_idx = mixture_df.index.names[0]
    assert preds.index.get_level_values(time_idx).min() == test_start + seq_num
    assert preds.index.get_level_values(time_idx).max() == test_start + seq_num + output_length - 1


def test_mixture_predict_lambda_zero_local_only(mixture_df, partition_dict, targets):
    """With lambda_mix=0.0, all samples come from the local pool."""
    model = MixtureBaseline(
        targets=targets, window_months=4, lambda_mix=0.0, n_samples=100,
        partition_dict=partition_dict, loa="pg_id",
    )
    model.fit(mixture_df)
    preds = model.predict(df=mixture_df, sequence_number=0, output_length=1)

    test_start = partition_dict["test"][0]

    # Entity 1: samples should all be from its local pool
    cell = preds.loc[(test_start, 1), "pred_y1"]
    local_vals = set(model.local_pool[1]["y1"].tolist())
    assert set(cell).issubset(local_vals)

    # Entity 3 (all-zero): should be all zeros
    cell_zero = preds.loc[(test_start, 3), "pred_y1"]
    assert all(v == 0.0 for v in cell_zero)


def test_mixture_predict_lambda_one_global_only(mixture_df, partition_dict, targets):
    """With lambda_mix=1.0, all-zero entity gets only positive samples."""
    model = MixtureBaseline(
        targets=targets, window_months=4, lambda_mix=1.0, n_samples=100,
        partition_dict=partition_dict, loa="pg_id",
    )
    model.fit(mixture_df)
    preds = model.predict(df=mixture_df, sequence_number=0, output_length=1)

    test_start = partition_dict["test"][0]
    # Entity 3 (all-zero local pool) should have all positive samples from global pool
    cell = preds.loc[(test_start, 3), "pred_y1"]
    assert all(v > 0 for v in cell)


def test_mixture_predict_reproducible(mixture_df, partition_dict, targets):
    """Same seed produces identical predictions."""
    kwargs = dict(
        targets=targets, window_months=4, lambda_mix=0.05, n_samples=50,
        partition_dict=partition_dict, loa="pg_id", seed=123,
    )
    m1 = MixtureBaseline(**kwargs)
    m1.fit(mixture_df)
    p1 = m1.predict(df=mixture_df, sequence_number=0, output_length=2)

    m2 = MixtureBaseline(**kwargs)
    m2.fit(mixture_df)
    p2 = m2.predict(df=mixture_df, sequence_number=0, output_length=2)

    for col in p1.columns:
        for idx in p1.index:
            assert p1.loc[idx, col] == p2.loc[idx, col]


def test_mixture_predict_prediction_frame_shape(mixture_df, partition_dict, targets):
    from views_pipeline_core.data.prediction_frame import PredictionFrame

    n_samples = 32
    model = MixtureBaseline(
        targets=targets, window_months=4, lambda_mix=0.05, n_samples=n_samples,
        partition_dict=partition_dict, loa="pg_id",
    )
    model.fit(mixture_df)

    output_length = 5
    result = model.predict_prediction_frame(
        df=mixture_df, sequence_number=0, output_length=output_length,
    )

    n_entities = 3
    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)

    for target in targets:
        pf = result[target]
        assert isinstance(pf, PredictionFrame)
        assert pf.y_pred.shape == (n_entities * output_length, n_samples)
        assert len(pf.identifiers["time"]) == n_entities * output_length
        assert len(pf.identifiers["unit"]) == n_entities * output_length


def test_mixture_predict_prediction_frame_matches_predict(mixture_df, partition_dict, targets):
    n_samples = 16
    kwargs = dict(
        targets=targets, window_months=4, lambda_mix=0.05, n_samples=n_samples,
        partition_dict=partition_dict, loa="pg_id", seed=99,
    )
    output_length = 3

    m1 = MixtureBaseline(**kwargs)
    m1.fit(mixture_df)
    df_preds = m1.predict(df=mixture_df, sequence_number=0, output_length=output_length)

    m2 = MixtureBaseline(**kwargs)
    m2.fit(mixture_df)
    pf_result = m2.predict_prediction_frame(df=mixture_df, sequence_number=0, output_length=output_length)

    for target in targets:
        pf = pf_result[target]
        for i in range(pf.y_pred.shape[0]):
            tid = pf.identifiers["time"][i]
            uid = pf.identifiers["unit"][i]
            expected = df_preds.loc[(tid, uid), f"pred_{target}"]
            assert list(pf.y_pred[i]) == pytest.approx(expected)

