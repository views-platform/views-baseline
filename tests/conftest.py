import pandas as pd
import pytest


def make_dummy_df(entity_id="pg_id", time_range=range(440, 540)):
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


@pytest.fixture
def targets():
    return ["y1", "y2"]


def assert_prediction_structure(
    preds, df, targets, partition_dict, sequence_number, output_length
):
    """Assert common structural properties of point-model prediction DataFrames."""
    time_idx, entity_idx = df.index.names
    test_start = partition_dict["test"][0]
    prediction_start = test_start + sequence_number
    prediction_end = prediction_start + output_length - 1

    assert list(preds.columns) == [f"pred_{t}" for t in targets]
    assert preds.index.names == [time_idx, entity_idx]
    assert preds.index.get_level_values(time_idx).min() == prediction_start
    assert preds.index.get_level_values(time_idx).max() == prediction_end
    pred_time_ids = preds.index.get_level_values(time_idx).unique().tolist()
    assert pred_time_ids == list(range(prediction_start, prediction_end + 1))
