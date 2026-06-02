import numpy as np
import pandas as pd
import pytest
from conftest import assert_point_prediction_structure, make_dummy_df

from views_baseline.model.baseline import (
    AverageModel,
    ConflictologyModel,
    LocfModel,
    MixtureBaseline,
    ZeroModel,
)


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
    result = model.predict(df=base_df_pgm, sequence_number=0, output_length=output_length)

    assert_point_prediction_structure(
        result, base_df_pgm, targets, partition_dict, 0, output_length
    )
    assert base_df_pgm.index.names[1] == "pg_id"
    for target in targets:
        assert (result[target].y_pred == 0.0).all()


def test_zero_model_predicts_zeros_cm(base_df_cm, partition_dict, targets):
    model = ZeroModel(targets=targets, partition_dict=partition_dict, loa="country_id")
    model.fit(base_df_cm)
    output_length = 36
    result = model.predict(df=base_df_cm, sequence_number=0, output_length=output_length)

    assert_point_prediction_structure(
        result, base_df_cm, targets, partition_dict, 0, output_length
    )
    assert base_df_cm.index.names[1] == "country_id"
    for target in targets:
        assert (result[target].y_pred == 0.0).all()


def test_zero_model_respects_sequence_number(base_df_pgm, partition_dict, targets):
    model = ZeroModel(targets=targets, partition_dict=partition_dict, loa="pg_id")
    model.fit(base_df_pgm)

    test_start = partition_dict["test"][0]
    seq_num = 2
    output_length = 36

    result = model.predict(df=base_df_pgm, sequence_number=seq_num, output_length=output_length)
    pf = result[targets[0]]
    time_vals = pf.identifiers["time"]

    assert min(time_vals) == test_start + seq_num
    assert max(time_vals) == test_start + seq_num + output_length - 1


# -----------------------------------------------------------------------
# LocfModel
# -----------------------------------------------------------------------


def test_locf_model_uses_last_observation(base_df_pgm, partition_dict, targets):
    model = LocfModel(targets=targets, partition_dict=partition_dict, loa="pg_id")
    model.fit(base_df_pgm)
    output_length = 36

    time_idx, entity_idx = base_df_pgm.index.names
    test_start = partition_dict["test"][0]

    result = model.predict(df=base_df_pgm, sequence_number=0, output_length=output_length)

    train_df = base_df_pgm[base_df_pgm.index.get_level_values(time_idx) < test_start]
    expected_last = train_df.groupby(level=entity_idx)[targets].last()

    for target in targets:
        pf = result[target]
        for i in range(pf.y_pred.shape[0]):
            uid = pf.identifiers["unit"][i]
            assert pf.y_pred[i, 0] == expected_last.loc[uid, target]

    assert_point_prediction_structure(
        result, base_df_pgm, targets, partition_dict, 0, output_length
    )


def test_locf_model_respects_sequence_number(base_df_pgm, partition_dict, targets):
    model = LocfModel(targets=targets, partition_dict=partition_dict, loa="pg_id")
    model.fit(base_df_pgm)

    test_start = partition_dict["test"][0]
    seq_num = 1
    output_length = 36
    result = model.predict(df=base_df_pgm, sequence_number=seq_num, output_length=output_length)

    pf = result[targets[0]]
    time_vals = pf.identifiers["time"]
    assert min(time_vals) == test_start + seq_num
    assert max(time_vals) == test_start + seq_num + output_length - 1


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
        window_months=months,
        partition_dict=partition_dict,
        loa="pg_id",
    )

    output_length = 36
    model.fit(base_df_pgm)

    time_idx, entity_idx = base_df_pgm.index.names
    test_start = partition_dict["test"][0]

    result = model.predict(df=base_df_pgm, sequence_number=0, output_length=output_length)

    train_df = base_df_pgm[base_df_pgm.index.get_level_values(time_idx) < test_start]
    train_df = train_df.sort_index(level=[entity_idx, time_idx])
    expected_means = train_df.groupby(level=entity_idx, group_keys=False).apply(
        lambda g: g.tail(months)[targets].mean()
    )

    for target in targets:
        pf = result[target]
        for i in range(pf.y_pred.shape[0]):
            uid = pf.identifiers["unit"][i]
            assert pf.y_pred[i, 0] == pytest.approx(expected_means.loc[uid, target])

    assert_point_prediction_structure(
        result, base_df_pgm, targets, partition_dict, 0, output_length
    )


def test_average_model_respects_sequence_number(base_df_pgm, partition_dict, targets):
    months = 2
    model = AverageModel(
        targets=targets,
        window_months=months,
        partition_dict=partition_dict,
        loa="pg_id",
    )
    model.fit(base_df_pgm)

    test_start = partition_dict["test"][0]
    seq_num = 2
    output_length = 36

    result = model.predict(df=base_df_pgm, sequence_number=seq_num, output_length=output_length)
    pf = result[targets[0]]
    time_vals = pf.identifiers["time"]

    assert min(time_vals) == test_start + seq_num
    assert max(time_vals) == test_start + seq_num + output_length - 1


# -----------------------------------------------------------------------
# ConflictologyModel (distribution baseline)
# -----------------------------------------------------------------------


def test_conflictology_model_resamples_from_history(base_df_pgm, partition_dict, targets):
    from views_pipeline_core.data.prediction_frame import PredictionFrame

    months = 4
    n_samples = 64
    model = ConflictologyModel(
        targets=targets,
        window_months=months,
        partition_dict=partition_dict,
        loa="pg_id",
        n_samples=n_samples,
        seed=42,
    )
    model.fit(base_df_pgm)

    test_start = partition_dict["test"][0]
    time_idx, entity_idx = base_df_pgm.index.names
    output_length = 5
    result = model.predict(df=base_df_pgm, sequence_number=0, output_length=output_length)

    # Returns dict of PredictionFrames
    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)

    n_entities = (
        base_df_pgm.loc[
            base_df_pgm.index.get_level_values(time_idx) == test_start - 1
        ]
        .index.get_level_values(entity_idx)
        .nunique()
    )

    for target in targets:
        pf = result[target]
        assert isinstance(pf, PredictionFrame)
        assert pf.y_pred.shape == (n_entities * output_length, n_samples)

    # All sampled values must come from the history window
    train_df = base_df_pgm[
        base_df_pgm.index.get_level_values(time_idx) < test_start
    ]
    train_df = train_df.sort_index(level=[entity_idx, time_idx])

    pf = result[targets[0]]
    for i in range(pf.y_pred.shape[0]):
        uid = pf.identifiers["unit"][i]
        ent_history = train_df.xs(uid, level=entity_idx).tail(months)
        history_values = set(ent_history[targets[0]].tolist())
        assert set(pf.y_pred[i].tolist()).issubset(history_values)


def test_conflictology_model_respects_sequence_number(base_df_pgm, partition_dict, targets):
    months = 3
    n_samples = 32
    model = ConflictologyModel(
        targets=targets,
        window_months=months,
        partition_dict=partition_dict,
        loa="pg_id",
        n_samples=n_samples,
        seed=42,
    )
    model.fit(base_df_pgm)

    test_start = partition_dict["test"][0]
    seq_num = 2
    output_length = 4

    result = model.predict(
        df=base_df_pgm, sequence_number=seq_num, output_length=output_length,
    )

    pf = result[targets[0]]
    time_vals = pf.identifiers["time"]
    assert min(time_vals) == test_start + seq_num
    assert max(time_vals) == test_start + seq_num + output_length - 1


def test_conflictology_matches_mixture_lambda_zero(base_df_pgm, partition_dict, targets):
    """
    ConflictologyModel and MixtureBaseline(lambda_mix=0) should draw from
    the same local history pool per entity/target.
    """
    window = 4
    n_samples = 128

    conf = ConflictologyModel(
        targets=targets,
        window_months=window,
        partition_dict=partition_dict,
        loa="pg_id",
        n_samples=n_samples,
        seed=42,
    )
    conf.fit(base_df_pgm)

    mix = MixtureBaseline(
        targets=targets,
        window_months=window,
        lambda_mix=0.0,
        n_samples=n_samples,
        partition_dict=partition_dict,
        loa="pg_id",
        seed=99,  # different seed — we test pools, not samples
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


# -----------------------------------------------------------------------
# build_prediction_grid helper
# -----------------------------------------------------------------------


def test_build_prediction_grid_shape_and_values():
    from views_baseline.model.helpers import build_prediction_grid

    df = build_prediction_grid(
        time_idx="month_id",
        entity_idx="pg_id",
        entity_ids=[1, 2],
        time_ids=[100, 101],
        targets=["y1"],
        value_fn=lambda cid, target: float(cid),
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
        entity_ids=[],
        time_ids=[100, 101],
        targets=["y1"],
        value_fn=lambda cid, target: 0.0,
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
        rows.append(
            {
                time_idx_name: t,
                entity_idx_name: 1,
                "y1": float(t - 439),
                "y2": float((t - 439) * 2),
            }
        )
        # Entity 2: positive values (different scale)
        rows.append(
            {
                time_idx_name: t,
                entity_idx_name: 2,
                "y1": float(t - 439) * 0.5,
                "y2": float(t - 439) * 0.1,
            }
        )
        # Entity 3: all zeros
        rows.append({time_idx_name: t, entity_idx_name: 3, "y1": 0.0, "y2": 0.0})
    df = pd.DataFrame(rows).set_index([time_idx_name, entity_idx_name]).sort_index()
    return df


@pytest.fixture
def mixture_df():
    return make_mixture_df()


def test_mixture_fit_extracts_local_pool(mixture_df, partition_dict, targets):
    model = MixtureBaseline(
        targets=targets,
        window_months=4,
        lambda_mix=0.05,
        n_samples=10,
        partition_dict=partition_dict,
        loa="pg_id",
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
        targets=targets,
        window_months=4,
        lambda_mix=0.05,
        n_samples=10,
        partition_dict=partition_dict,
        loa="pg_id",
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
        targets=targets,
        window_months=4,
        lambda_mix=0.05,
        n_samples=10,
        partition_dict=partition_dict,
        loa="pg_id",
    )
    model.fit(mixture_df)

    test_start = partition_dict["test"][0]
    # Max value in global pool for y1 should correspond to train_end
    # Entity 1 has y1 = t - 439, so max should be (test_start - 1) - 439
    max_expected = float(test_start - 1 - 439)
    assert max(model.global_pool["y1"]) <= max_expected


def test_mixture_fit_returns_self(mixture_df, partition_dict, targets):
    model = MixtureBaseline(
        targets=targets,
        window_months=4,
        lambda_mix=0.05,
        n_samples=10,
        partition_dict=partition_dict,
        loa="pg_id",
    )
    assert model.fit(mixture_df) is model


def test_mixture_predict_shape(mixture_df, partition_dict, targets):
    from views_pipeline_core.data.prediction_frame import PredictionFrame

    n_samples = 10
    model = MixtureBaseline(
        targets=targets,
        window_months=4,
        lambda_mix=0.05,
        n_samples=n_samples,
        partition_dict=partition_dict,
        loa="pg_id",
    )
    model.fit(mixture_df)
    output_length = 5
    result = model.predict(df=mixture_df, sequence_number=0, output_length=output_length)

    n_entities = 3
    assert isinstance(result, dict)
    assert set(result.keys()) == set(targets)

    for target in targets:
        pf = result[target]
        assert isinstance(pf, PredictionFrame)
        assert pf.y_pred.shape == (n_entities * output_length, n_samples)
        assert len(pf.identifiers["time"]) == n_entities * output_length
        assert len(pf.identifiers["unit"]) == n_entities * output_length


def test_mixture_predict_respects_sequence_number(mixture_df, partition_dict, targets):
    model = MixtureBaseline(
        targets=targets,
        window_months=4,
        lambda_mix=0.05,
        n_samples=10,
        partition_dict=partition_dict,
        loa="pg_id",
    )
    model.fit(mixture_df)

    test_start = partition_dict["test"][0]
    seq_num = 2
    output_length = 4
    result = model.predict(
        df=mixture_df, sequence_number=seq_num, output_length=output_length
    )

    pf = result[targets[0]]
    time_vals = pf.identifiers["time"]
    assert min(time_vals) == test_start + seq_num
    assert max(time_vals) == test_start + seq_num + output_length - 1


def test_mixture_predict_lambda_zero_local_only(mixture_df, partition_dict, targets):
    """With lambda_mix=0.0, all samples come from the local pool."""
    n_samples = 100
    model = MixtureBaseline(
        targets=targets,
        window_months=4,
        lambda_mix=0.0,
        n_samples=n_samples,
        partition_dict=partition_dict,
        loa="pg_id",
    )
    model.fit(mixture_df)
    result = model.predict(df=mixture_df, sequence_number=0, output_length=1)

    pf = result["y1"]
    for i in range(pf.y_pred.shape[0]):
        uid = pf.identifiers["unit"][i]
        local_vals = set(model.local_pool[uid]["y1"].tolist())
        assert set(pf.y_pred[i].tolist()).issubset(local_vals)

    # Entity 3 (all-zero): should be all zeros
    for i in range(pf.y_pred.shape[0]):
        if pf.identifiers["unit"][i] == 3:
            assert all(v == 0.0 for v in pf.y_pred[i])
            break


def test_mixture_predict_lambda_one_global_only(mixture_df, partition_dict, targets):
    """With lambda_mix=1.0, all-zero entity gets only positive samples."""
    model = MixtureBaseline(
        targets=targets,
        window_months=4,
        lambda_mix=1.0,
        n_samples=100,
        partition_dict=partition_dict,
        loa="pg_id",
    )
    model.fit(mixture_df)
    result = model.predict(df=mixture_df, sequence_number=0, output_length=1)

    # Entity 3 (all-zero local pool) should have all positive samples from global pool
    pf = result["y1"]
    for i in range(pf.y_pred.shape[0]):
        if pf.identifiers["unit"][i] == 3:
            assert all(v > 0 for v in pf.y_pred[i])
            break


def test_mixture_predict_reproducible(mixture_df, partition_dict, targets):
    """Same seed produces identical predictions."""
    kwargs = dict(
        targets=targets,
        window_months=4,
        lambda_mix=0.05,
        n_samples=50,
        partition_dict=partition_dict,
        loa="pg_id",
        seed=123,
    )
    m1 = MixtureBaseline(**kwargs)
    m1.fit(mixture_df)
    r1 = m1.predict(df=mixture_df, sequence_number=0, output_length=2)

    m2 = MixtureBaseline(**kwargs)
    m2.fit(mixture_df)
    r2 = m2.predict(df=mixture_df, sequence_number=0, output_length=2)

    for target in targets:
        np.testing.assert_array_equal(r1[target].y_pred, r2[target].y_pred)


# -----------------------------------------------------------------------
# BaselineModelCatalog
# -----------------------------------------------------------------------


def test_catalog_get_zero_model(partition_dict, targets):
    from views_baseline.model.catalog import BaselineModelCatalog

    config = {"targets": targets}
    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pg_id")
    model = catalog.get_model("ZeroModel")
    assert isinstance(model, ZeroModel)


def test_catalog_get_average_model(partition_dict, targets):
    from views_baseline.model.catalog import BaselineModelCatalog

    config = {"targets": targets, "window_months": 6}
    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pg_id")
    model = catalog.get_model("AverageModel")
    assert isinstance(model, AverageModel)
    assert model.window_months == 6


def test_catalog_unknown_model_raises(partition_dict, targets):
    from views_baseline.model.catalog import BaselineModelCatalog

    config = {"targets": targets}
    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pg_id")
    with pytest.raises(ValueError, match="NoSuchModel"):
        catalog.get_model("NoSuchModel")


def test_catalog_missing_required_key_raises(partition_dict, targets):
    from views_baseline.model.catalog import BaselineModelCatalog

    config = {"targets": targets}  # missing "window_months" required by AverageModel
    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pg_id")
    with pytest.raises(ValueError, match="window_months"):
        catalog.get_model("AverageModel")


def test_catalog_list_models(partition_dict, targets):
    from views_baseline.model.catalog import BaselineModelCatalog

    config = {"targets": targets}
    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pg_id")
    names = catalog.list_models()
    expected = {"ZeroModel", "LocfModel", "AverageModel", "ConflictologyModel", "MixtureBaseline"}
    assert set(names) == expected


# -----------------------------------------------------------------------
# Red team: degenerate parameter tests
# -----------------------------------------------------------------------


def test_average_model_window_months_zero_produces_nan(base_df_pgm, partition_dict, targets):
    """window_months=0 → tail(0) is empty → mean is NaN → all predictions NaN."""
    model = AverageModel(
        targets=targets, window_months=0, partition_dict=partition_dict, loa="pg_id"
    )
    model.fit(base_df_pgm)
    result = model.predict(df=base_df_pgm, sequence_number=0, output_length=5)
    for target in targets:
        assert np.isnan(result[target].y_pred).all()


def test_conflictology_window_months_zero_raises(base_df_pgm, partition_dict, targets):
    """window_months=0 → tail(0) empty → xs() raises KeyError during fit."""
    model = ConflictologyModel(
        targets=targets, window_months=0, partition_dict=partition_dict,
        loa="pg_id", n_samples=10,
    )
    with pytest.raises(KeyError):
        model.fit(base_df_pgm)


def test_mixture_window_months_zero_raises(partition_dict, targets):
    """window_months=0 → empty local pool → rng.choice raises ValueError."""
    df = make_mixture_df()
    model = MixtureBaseline(
        targets=targets, window_months=0, lambda_mix=0.0,
        n_samples=10, partition_dict=partition_dict, loa="pg_id",
    )
    model.fit(df)
    with pytest.raises(ValueError):
        model.predict(df=df, sequence_number=0, output_length=5)


def test_conflictology_n_samples_zero_raises(base_df_pgm, partition_dict, targets):
    """n_samples=0 → PredictionFrame rejects y_pred with 0 sample columns."""
    model = ConflictologyModel(
        targets=targets, window_months=4, partition_dict=partition_dict,
        loa="pg_id", n_samples=0,
    )
    model.fit(base_df_pgm)
    with pytest.raises(ValueError, match="at least one sample column"):
        model.predict(df=base_df_pgm, sequence_number=0, output_length=2)


def test_predict_before_fit_raises(base_df_pgm, partition_dict, targets):
    """predict() before fit() → self.time_idx is None → crash."""
    model = ZeroModel(targets=targets, partition_dict=partition_dict, loa="pg_id")
    with pytest.raises((AttributeError, TypeError, KeyError)):
        model.predict(df=base_df_pgm, sequence_number=0, output_length=36)


# -----------------------------------------------------------------------
# Beige team: entity-drop warning tests
# -----------------------------------------------------------------------


def test_locf_entity_drop_warning(caplog, base_df_pgm, partition_dict, targets):
    """Entities in predict df but not in fitted state trigger a WARNING."""
    import logging

    model = LocfModel(targets=targets, partition_dict=partition_dict, loa="pg_id")
    model.fit(base_df_pgm)

    # Add entity 3 at train_end — it won't be in last_observations
    test_start = partition_dict["test"][0]
    train_end = test_start - 1
    extra = pd.DataFrame(
        {"y1": [99.0], "y2": [99.0]},
        index=pd.MultiIndex.from_tuples(
            [(train_end, 3)], names=["month_id", "pg_id"]
        ),
    )
    df_predict = pd.concat([base_df_pgm, extra]).sort_index()

    with caplog.at_level(logging.WARNING):
        model.predict(df=df_predict, sequence_number=0, output_length=5)

    assert "LocfModel: 1 entities dropped" in caplog.text


def test_average_entity_drop_warning(caplog, base_df_pgm, partition_dict, targets):
    """Entities in predict df but not in fitted state trigger a WARNING."""
    import logging

    model = AverageModel(
        targets=targets, window_months=3, partition_dict=partition_dict, loa="pg_id"
    )
    model.fit(base_df_pgm)

    test_start = partition_dict["test"][0]
    train_end = test_start - 1
    extra = pd.DataFrame(
        {"y1": [99.0], "y2": [99.0]},
        index=pd.MultiIndex.from_tuples(
            [(train_end, 3)], names=["month_id", "pg_id"]
        ),
    )
    df_predict = pd.concat([base_df_pgm, extra]).sort_index()

    with caplog.at_level(logging.WARNING):
        model.predict(df=df_predict, sequence_number=0, output_length=5)

    assert "AverageModel: 1 entities dropped" in caplog.text


# -----------------------------------------------------------------------
# Green team: ConflictologyModel reproducibility
# -----------------------------------------------------------------------


def test_conflictology_predict_reproducible(base_df_pgm, partition_dict, targets):
    """Same seed produces identical predictions."""
    kwargs = dict(
        targets=targets,
        window_months=4,
        partition_dict=partition_dict,
        loa="pg_id",
        n_samples=50,
        seed=123,
    )
    m1 = ConflictologyModel(**kwargs)
    m1.fit(base_df_pgm)
    r1 = m1.predict(df=base_df_pgm, sequence_number=0, output_length=2)

    m2 = ConflictologyModel(**kwargs)
    m2.fit(base_df_pgm)
    r2 = m2.predict(df=base_df_pgm, sequence_number=0, output_length=2)

    for target in targets:
        np.testing.assert_array_equal(r1[target].y_pred, r2[target].y_pred)
