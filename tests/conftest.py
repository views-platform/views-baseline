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
