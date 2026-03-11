import pandas as pd
import numpy as np
from itertools import product

import pytest

from views_baseline.model.baseline import (
    ZeroModel,
    LocfModel,
    AverageModel,
    ConflictologyModel,
)


def make_dummy_df(entity_id="pg_id"):
    """
    Create a simple MultiIndex dataframe with 2 entities and a range of months.
    Index names matter because the models read them from df.index.names[0/1].
    """
    time_idx_name = "month_id"
    entity_idx_name = entity_id

    times = list(range(440, 540))
    entities = [1, 2]

    tuples = list(product(times, entities))
    index = pd.MultiIndex.from_tuples(
        tuples,
        names=[time_idx_name, entity_idx_name],
    )

    df = pd.DataFrame(index=index)

    # Simple deterministic targets
    # Example: y1 = time * 10 + entity, y2 = time * 100 + entity
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
def partition_dict():
    # test_start = 120, so train_end = 119
    return {"test": (493, 540)}


@pytest.fixture
def targets():
    return ["y1", "y2"]


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
    test_start, test_end = partition_dict["test"]
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
    test_start, test_end = partition_dict["test"]
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
    test_start, test_end = partition_dict["test"]
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
    test_start, test_end = partition_dict["test"]
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


def test_conflictology_model_returns_history_lists(base_df_pgm, partition_dict, targets):
    months = 4
    model = ConflictologyModel(
        targets=targets,
        months=months,
        partition_dict=partition_dict,
        loa="pg_id",
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

    # For each entity, expected history is the last `months` scalar values before test_start
    for ent in train_df.index.get_level_values(entity_idx).unique():
        ent_history = train_df.xs(ent, level=entity_idx).tail(months)

        for target in targets:
            expected_list = ent_history[target].tolist()

            # check first prediction time
            first_time = test_start
            cell_value = preds.loc[(first_time, ent), f"pred_{target}"]

            # 1) each prediction is a list (distribution forecast)
            assert isinstance(cell_value, list)

            # 2) list has the right length
            assert len(cell_value) == months

            # 3) content is exactly the last n months (order preserved)
            assert cell_value == expected_list

            # 4) all entries are numeric (no nested lists, etc.)
            assert all(isinstance(x, (int, float, np.integer, np.floating)) for x in cell_value)

            # 5) for all other forecast times we get the same list
            for t in range(test_start, test_start + output_length):
                assert preds.loc[(t, ent), f"pred_{target}"] == expected_list


def test_conflictology_model_respects_sequence_number(base_df_pgm, partition_dict, targets):
    months = 3
    model = ConflictologyModel(
        targets=targets,
        months=months,
        partition_dict=partition_dict,
        loa="pg_id",
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
    # sequence_number only shifts the prediction window, NOT the history window
    prediction_start = test_start + seq_num
    prediction_end = prediction_start + output_length - 1

    assert preds.index.get_level_values(time_idx).min() == prediction_start
    assert preds.index.get_level_values(time_idx).max() == prediction_end

    # --- history window checks (fixed, does not depend on seq_num) ---
    train_end = test_start - 1
    history_start = train_end - (months - 1)

    df_hist = base_df_pgm[
        (base_df_pgm.index.get_level_values(time_idx) >= history_start)
        & (base_df_pgm.index.get_level_values(time_idx) <= train_end)
    ].sort_index(level=[entity_idx, time_idx])

    for ent in df_hist.index.get_level_values(entity_idx).unique():
        ent_hist = df_hist.xs(ent, level=entity_idx)
        for target in targets:
            expected_list = ent_hist[target].tolist()

            first_time = prediction_start
            cell_value = preds.loc[(first_time, ent), f"pred_{target}"]

            assert isinstance(cell_value, list)
            assert len(cell_value) == months
            assert cell_value == expected_list

            for t in range(prediction_start, prediction_end + 1):
                assert preds.loc[(t, ent), f"pred_{target}"] == expected_list


def test_conflictology_model_predict_prediction_frame(base_df_pgm, partition_dict, targets):
    from views_pipeline_core.data.prediction_frame import PredictionFrame

    months = 4
    model = ConflictologyModel(
        targets=targets,
        months=months,
        partition_dict=partition_dict,
        loa="pg_id",
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
        assert pf.y_pred.shape == (n_entities * output_length, months)
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

