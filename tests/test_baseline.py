import numpy as np
import pandas as pd
import pytest
from conftest import assert_point_prediction_structure, make_dummy_df

from views_baseline.model.models.distributional import ConflictologyModel, MixtureBaseline
from views_baseline.model.models.point import AverageModel, LocfModel, ZeroModel


@pytest.fixture
def partition_dict():
    return {"test": (493, 540)}


@pytest.fixture
def base_df_pgm():
    return make_dummy_df(entity_id="priogrid_id")


@pytest.fixture
def base_df_cm():
    return make_dummy_df(entity_id="country_id")


# -----------------------------------------------------------------------
# ZeroModel
# -----------------------------------------------------------------------


def test_zero_model_predicts_zeros_pgm(base_df_pgm, partition_dict, targets):
    model = ZeroModel(targets=targets, partition_dict=partition_dict, loa="pgm")
    model.fit(base_df_pgm)
    output_length = 36
    result = model.predict(df=base_df_pgm, sequence_number=0, output_length=output_length)

    assert_point_prediction_structure(
        result, base_df_pgm, targets, partition_dict, 0, output_length
    )
    assert base_df_pgm.index.names[1] == "priogrid_id"
    for target in targets:
        assert (result[target].values == 0.0).all()


def test_zero_model_predicts_zeros_cm(base_df_cm, partition_dict, targets):
    model = ZeroModel(targets=targets, partition_dict=partition_dict, loa="cm")
    model.fit(base_df_cm)
    output_length = 36
    result = model.predict(df=base_df_cm, sequence_number=0, output_length=output_length)

    assert_point_prediction_structure(
        result, base_df_cm, targets, partition_dict, 0, output_length
    )
    assert base_df_cm.index.names[1] == "country_id"
    for target in targets:
        assert (result[target].values == 0.0).all()


def test_zero_model_respects_sequence_number(base_df_pgm, partition_dict, targets):
    model = ZeroModel(targets=targets, partition_dict=partition_dict, loa="pgm")
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
    model = LocfModel(targets=targets, partition_dict=partition_dict, loa="pgm")
    model.fit(base_df_pgm)
    output_length = 36

    time_idx, entity_idx = base_df_pgm.index.names
    test_start = partition_dict["test"][0]

    result = model.predict(df=base_df_pgm, sequence_number=0, output_length=output_length)

    train_df = base_df_pgm[base_df_pgm.index.get_level_values(time_idx) < test_start]
    expected_last = train_df.groupby(level=entity_idx)[targets].last()

    for target in targets:
        pf = result[target]
        for i in range(pf.values.shape[0]):
            uid = pf.identifiers["unit"][i]
            assert pf.values[i, 0] == expected_last.loc[uid, target]

    assert_point_prediction_structure(
        result, base_df_pgm, targets, partition_dict, 0, output_length
    )


def test_locf_model_respects_sequence_number(base_df_pgm, partition_dict, targets):
    model = LocfModel(targets=targets, partition_dict=partition_dict, loa="pgm")
    model.fit(base_df_pgm)

    test_start = partition_dict["test"][0]
    seq_num = 1
    output_length = 36
    result = model.predict(df=base_df_pgm, sequence_number=seq_num, output_length=output_length)

    pf = result[targets[0]]
    time_vals = pf.identifiers["time"]
    assert min(time_vals) == test_start + seq_num
    assert max(time_vals) == test_start + seq_num + output_length - 1


def test_locf_model_fit_handles_unsorted_data(partition_dict, targets):
    time_idx_name, entity_idx_name = "month_id", "priogrid_id"
    rows = []
    for t in [490, 492, 491]:  # deliberately unsorted
        rows.append({time_idx_name: t, entity_idx_name: 1, "y1": t * 10 + 1, "y2": t * 100 + 1})
    rows.append({time_idx_name: 492, entity_idx_name: 2, "y1": 4921, "y2": 49202})
    df = pd.DataFrame(rows).set_index([time_idx_name, entity_idx_name])

    model = LocfModel(targets=targets, partition_dict=partition_dict, loa="pgm")
    model.fit(df)
    # Must use month 492 (temporally last), not 491 (positionally last).
    # last_observations is now a numpy-backed dict {entity -> {target -> value}} (PR-2 S7).
    assert model.last_observations[1]["y1"] == 492 * 10 + 1
    assert model.last_observations[1]["y2"] == 492 * 100 + 1


# -----------------------------------------------------------------------
# AverageModel
# -----------------------------------------------------------------------


def test_average_model_uses_mean_of_last_n_months(base_df_pgm, partition_dict, targets):
    months = 3
    model = AverageModel(
        targets=targets,
        window_months=months,
        partition_dict=partition_dict,
        loa="pgm",
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
        for i in range(pf.values.shape[0]):
            uid = pf.identifiers["unit"][i]
            assert pf.values[i, 0] == pytest.approx(expected_means.loc[uid, target])

    assert_point_prediction_structure(
        result, base_df_pgm, targets, partition_dict, 0, output_length
    )


def test_average_model_respects_sequence_number(base_df_pgm, partition_dict, targets):
    months = 2
    model = AverageModel(
        targets=targets,
        window_months=months,
        partition_dict=partition_dict,
        loa="pgm",
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
    from views_frames import PredictionFrame

    months = 4
    n_samples = 64
    model = ConflictologyModel(
        targets=targets,
        window_months=months,
        partition_dict=partition_dict,
        loa="pgm",
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
        assert pf.values.shape == (n_entities * output_length, n_samples)

    # All sampled values must come from the history window
    train_df = base_df_pgm[
        base_df_pgm.index.get_level_values(time_idx) < test_start
    ]
    train_df = train_df.sort_index(level=[entity_idx, time_idx])

    pf = result[targets[0]]
    for i in range(pf.values.shape[0]):
        uid = pf.identifiers["unit"][i]
        ent_history = train_df.xs(uid, level=entity_idx).tail(months)
        history_values = set(ent_history[targets[0]].tolist())
        assert set(pf.values[i].tolist()).issubset(history_values)


def test_conflictology_model_respects_sequence_number(base_df_pgm, partition_dict, targets):
    months = 3
    n_samples = 32
    model = ConflictologyModel(
        targets=targets,
        window_months=months,
        partition_dict=partition_dict,
        loa="pgm",
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
        loa="pgm",
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
        loa="pgm",
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
# MixtureBaseline
# -----------------------------------------------------------------------


def make_mixture_df():
    """
    DataFrame with 3 entities: two with positive values, one all-zero.
    Entity 3 (all-zero) tests the zero-probability trap.
    """
    time_idx_name = "month_id"
    entity_idx_name = "priogrid_id"
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
        loa="pgm",
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
        loa="pgm",
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
        loa="pgm",
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
        loa="pgm",
    )
    assert model.fit(mixture_df) is model


def test_mixture_predict_shape(mixture_df, partition_dict, targets):
    from views_frames import PredictionFrame

    n_samples = 10
    model = MixtureBaseline(
        targets=targets,
        window_months=4,
        lambda_mix=0.05,
        n_samples=n_samples,
        partition_dict=partition_dict,
        loa="pgm",
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
        assert pf.values.shape == (n_entities * output_length, n_samples)
        assert len(pf.identifiers["time"]) == n_entities * output_length
        assert len(pf.identifiers["unit"]) == n_entities * output_length


def test_mixture_predict_respects_sequence_number(mixture_df, partition_dict, targets):
    model = MixtureBaseline(
        targets=targets,
        window_months=4,
        lambda_mix=0.05,
        n_samples=10,
        partition_dict=partition_dict,
        loa="pgm",
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
        loa="pgm",
    )
    model.fit(mixture_df)
    result = model.predict(df=mixture_df, sequence_number=0, output_length=1)

    pf = result["y1"]
    for i in range(pf.values.shape[0]):
        uid = pf.identifiers["unit"][i]
        local_vals = set(model.local_pool[uid]["y1"].tolist())
        assert set(pf.values[i].tolist()).issubset(local_vals)

    # Entity 3 (all-zero): should be all zeros
    for i in range(pf.values.shape[0]):
        if pf.identifiers["unit"][i] == 3:
            assert all(v == 0.0 for v in pf.values[i])
            break


def test_mixture_predict_lambda_one_global_only(mixture_df, partition_dict, targets):
    """With lambda_mix=1.0, all-zero entity gets only positive samples."""
    model = MixtureBaseline(
        targets=targets,
        window_months=4,
        lambda_mix=1.0,
        n_samples=100,
        partition_dict=partition_dict,
        loa="pgm",
    )
    model.fit(mixture_df)
    result = model.predict(df=mixture_df, sequence_number=0, output_length=1)

    # Entity 3 (all-zero local pool) should have all positive samples from global pool
    pf = result["y1"]
    for i in range(pf.values.shape[0]):
        if pf.identifiers["unit"][i] == 3:
            assert all(v > 0 for v in pf.values[i])
            break


def test_mixture_predict_reproducible(mixture_df, partition_dict, targets):
    """Same seed produces identical predictions."""
    kwargs = dict(
        targets=targets,
        window_months=4,
        lambda_mix=0.05,
        n_samples=50,
        partition_dict=partition_dict,
        loa="pgm",
        seed=123,
    )
    m1 = MixtureBaseline(**kwargs)
    m1.fit(mixture_df)
    r1 = m1.predict(df=mixture_df, sequence_number=0, output_length=2)

    m2 = MixtureBaseline(**kwargs)
    m2.fit(mixture_df)
    r2 = m2.predict(df=mixture_df, sequence_number=0, output_length=2)

    for target in targets:
        np.testing.assert_array_equal(r1[target].values, r2[target].values)


# -----------------------------------------------------------------------
# BaselineModelCatalog
# -----------------------------------------------------------------------


def test_catalog_get_zero_model(partition_dict, targets):
    from views_baseline.model.catalog import BaselineModelCatalog

    config = {"targets": targets}
    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pgm")
    model = catalog.get_model("ZeroModel")
    assert isinstance(model, ZeroModel)


def test_catalog_get_average_model(partition_dict, targets):
    from views_baseline.model.catalog import BaselineModelCatalog

    config = {"targets": targets, "window_months": 6}
    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pgm")
    model = catalog.get_model("AverageModel")
    assert isinstance(model, AverageModel)
    assert model.window_months == 6


def test_catalog_unknown_model_raises(partition_dict, targets):
    from views_baseline.model.catalog import BaselineModelCatalog

    config = {"targets": targets}
    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pgm")
    with pytest.raises(ValueError, match="NoSuchModel"):
        catalog.get_model("NoSuchModel")


def test_catalog_missing_required_key_raises(partition_dict, targets):
    from views_baseline.model.catalog import BaselineModelCatalog

    config = {"targets": targets}  # missing "window_months" required by AverageModel
    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pgm")
    with pytest.raises(ValueError, match="window_months"):
        catalog.get_model("AverageModel")


def test_catalog_list_models(partition_dict, targets):
    from views_baseline.model.catalog import BaselineModelCatalog

    config = {"targets": targets}
    catalog = BaselineModelCatalog(config=config, partition_dict=partition_dict, loa="pgm")
    names = catalog.list_models()
    expected = {
        "ZeroModel", "LocfModel", "AverageModel", "ConflictologyModel", "MixtureBaseline",
        "ParametricConflictology", "ParametricHurdleConflictology",
    }
    assert set(names) == expected


# -----------------------------------------------------------------------
# Red team: degenerate parameter tests
# -----------------------------------------------------------------------


def test_average_model_window_months_zero_raises(base_df_pgm, partition_dict, targets):
    """window_months=0 → window_pool fails loud with a ValueError during fit.

    (Pre-PR-2 the empty pandas tail produced silent all-NaN predictions; the numpy port
    validates the degenerate window explicitly, matching ConflictologyModel.)
    """
    model = AverageModel(
        targets=targets, window_months=0, partition_dict=partition_dict, loa="pgm"
    )
    with pytest.raises(ValueError, match="window_months must be >= 1"):
        model.fit(base_df_pgm)


def test_conflictology_window_months_zero_raises(base_df_pgm, partition_dict, targets):
    """window_months=0 → window_pool fails loud with a ValueError during fit.

    (Pre-PR-2 this raised an accidental KeyError from an empty pandas group; the numpy
    port validates the degenerate window explicitly instead.)
    """
    model = ConflictologyModel(
        targets=targets, window_months=0, partition_dict=partition_dict,
        loa="pgm", n_samples=10,
    )
    with pytest.raises(ValueError, match="window_months must be >= 1"):
        model.fit(base_df_pgm)


@pytest.mark.parametrize(
    "Model,kwargs",
    [
        (LocfModel, {}),
        (AverageModel, {"window_months": 3}),
        (ConflictologyModel, {"window_months": 3, "n_samples": 8}),
        (MixtureBaseline, {"window_months": 3, "lambda_mix": 0.1, "n_samples": 8}),
    ],
)
def test_nan_in_training_window_fails_loud(base_df_pgm, partition_dict, targets, Model, kwargs):
    """A NaN target inside the training window fails loud at fit (C-33), rather than being
    silently forward-filled (LOCF) or averaged-around (Average) as the pre-PR pandas path did."""
    df = base_df_pgm.copy()
    train_end = partition_dict["test"][0] - 1  # 492, the last training month
    df.loc[(train_end, 1), "y1"] = np.nan  # NaN in the used window for entity 1
    model = Model(targets=targets, partition_dict=partition_dict, loa="pgm", **kwargs)
    with pytest.raises(ValueError, match="NaN target value"):
        model.fit(df)


def test_mixture_nan_outside_window_fails_loud(base_df_pgm, partition_dict, targets):
    """For MixtureBaseline a NaN OUTSIDE the tail window still fails loud — its global pool
    consumes the whole train panel, so the fail-loud contract extends past the tail (C-33
    re-review fix). tail_pools alone would miss this (NaN is not in the last-3 tail)."""
    df = base_df_pgm.copy()
    train_end = partition_dict["test"][0] - 1  # 492
    df.loc[(train_end - 20, 1), "y1"] = np.nan  # month 472, well outside the 3-month tail
    model = MixtureBaseline(
        targets=targets, window_months=3, lambda_mix=0.1,
        n_samples=8, partition_dict=partition_dict, loa="pgm",
    )
    with pytest.raises(ValueError, match="NaN target value in the training panel"):
        model.fit(df)


def test_mixture_window_months_zero_raises(partition_dict, targets):
    """window_months=0 → window_pool fails loud with a ValueError during fit.

    (Pre-PR-2 the empty local pool surfaced later as a rng.choice ValueError in predict;
    the shared numpy window_pool now validates the degenerate window at fit time.)
    """
    df = make_mixture_df()
    model = MixtureBaseline(
        targets=targets, window_months=0, lambda_mix=0.0,
        n_samples=10, partition_dict=partition_dict, loa="pgm",
    )
    with pytest.raises(ValueError, match="window_months must be >= 1"):
        model.fit(df)


def test_conflictology_n_samples_zero_raises(base_df_pgm, partition_dict, targets):
    """n_samples=0 → PredictionFrame rejects y_pred with 0 sample columns."""
    model = ConflictologyModel(
        targets=targets, window_months=4, partition_dict=partition_dict,
        loa="pgm", n_samples=0,
    )
    model.fit(base_df_pgm)
    with pytest.raises(ValueError, match="at least one sample column"):
        model.predict(df=base_df_pgm, sequence_number=0, output_length=2)


def test_predict_before_fit_raises(base_df_pgm, partition_dict, targets):
    """predict() before fit() crashes for a model with fitted state.

    Post-PR-2 (S7) `ZeroModel` is genuinely stateless — it needs no fit — so the
    "predict before fit fails loud" contract is tested on `LocfModel`, whose predict reads
    the (unset) `last_observations` fitted state.
    """
    model = LocfModel(targets=targets, partition_dict=partition_dict, loa="pgm")
    with pytest.raises((AttributeError, TypeError, KeyError)):
        model.predict(df=base_df_pgm, sequence_number=0, output_length=36)


# -----------------------------------------------------------------------
# Beige team: entity-drop warning tests
# -----------------------------------------------------------------------


def test_locf_entity_drop_warning(caplog, base_df_pgm, partition_dict, targets):
    """Entities in predict df but not in fitted state trigger a WARNING."""
    import logging

    model = LocfModel(targets=targets, partition_dict=partition_dict, loa="pgm")
    model.fit(base_df_pgm)

    # Add entity 3 at train_end — it won't be in last_observations
    test_start = partition_dict["test"][0]
    train_end = test_start - 1
    extra = pd.DataFrame(
        {"y1": [99.0], "y2": [99.0]},
        index=pd.MultiIndex.from_tuples(
            [(train_end, 3)], names=["month_id", "priogrid_id"]
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
        targets=targets, window_months=3, partition_dict=partition_dict, loa="pgm"
    )
    model.fit(base_df_pgm)

    test_start = partition_dict["test"][0]
    train_end = test_start - 1
    extra = pd.DataFrame(
        {"y1": [99.0], "y2": [99.0]},
        index=pd.MultiIndex.from_tuples(
            [(train_end, 3)], names=["month_id", "priogrid_id"]
        ),
    )
    df_predict = pd.concat([base_df_pgm, extra]).sort_index()

    with caplog.at_level(logging.WARNING):
        model.predict(df=df_predict, sequence_number=0, output_length=5)

    assert "AverageModel: 1 entities dropped" in caplog.text


# -----------------------------------------------------------------------
# Empty-entity: fail loud (all 5 models + helper)
# -----------------------------------------------------------------------


def test_require_entities_raises_on_empty():
    from views_baseline.model.grid import require_entities

    with pytest.raises(ValueError, match="no entities to predict"):
        require_entities([], "TestModel")


def test_require_entities_noop_when_present():
    from views_baseline.model.grid import require_entities

    assert require_entities([1, 2], "TestModel") is None


def _train_end_only_entity(train_end, entity_id):
    """A predict df whose only train_end row holds a single entity."""
    return pd.DataFrame(
        {"y1": [99.0], "y2": [99.0]},
        index=pd.MultiIndex.from_tuples(
            [(train_end, entity_id)], names=["month_id", "priogrid_id"]
        ),
    )


def test_zero_model_raises_when_no_entities_at_train_end(
    base_df_pgm, partition_dict, targets
):
    """ZeroModel: no rows at train_end → fail loud, not a cryptic shape error."""
    model = ZeroModel(targets=targets, partition_dict=partition_dict, loa="pgm")
    model.fit(base_df_pgm)

    test_start = partition_dict["test"][0]
    train_end = test_start - 1
    # predict df has rows only well before train_end → no entities at train_end
    df_predict = pd.DataFrame(
        {"y1": [1.0], "y2": [2.0]},
        index=pd.MultiIndex.from_tuples(
            [(train_end - 5, 1)], names=["month_id", "priogrid_id"]
        ),
    )
    with pytest.raises(ValueError, match="ZeroModel: no entities to predict"):
        model.predict(df=df_predict, sequence_number=0, output_length=5)


def test_locf_model_raises_when_all_entities_dropped(
    base_df_pgm, partition_dict, targets
):
    """LocfModel: every train_end entity absent from fitted state → fail loud."""
    model = LocfModel(targets=targets, partition_dict=partition_dict, loa="pgm")
    model.fit(base_df_pgm)  # fitted on entities {1, 2}

    test_start = partition_dict["test"][0]
    train_end = test_start - 1
    df_predict = _train_end_only_entity(train_end, 3)  # only unknown entity 3
    with pytest.raises(ValueError, match="LocfModel: no entities to predict"):
        model.predict(df=df_predict, sequence_number=0, output_length=5)


def test_average_model_raises_when_all_entities_dropped(
    base_df_pgm, partition_dict, targets
):
    """AverageModel: every train_end entity absent from fitted state → fail loud."""
    model = AverageModel(
        targets=targets, window_months=3, partition_dict=partition_dict, loa="pgm"
    )
    model.fit(base_df_pgm)  # fitted on entities {1, 2}

    test_start = partition_dict["test"][0]
    train_end = test_start - 1
    df_predict = _train_end_only_entity(train_end, 3)
    with pytest.raises(ValueError, match="AverageModel: no entities to predict"):
        model.predict(df=df_predict, sequence_number=0, output_length=5)


def test_conflictology_raises_when_no_entities(base_df_pgm, partition_dict, targets):
    """ConflictologyModel: empty fitted pool → fail loud (was a silent return {})."""
    model = ConflictologyModel(
        targets=targets, window_months=4, partition_dict=partition_dict,
        loa="pgm", n_samples=10,
    )
    model.fit(base_df_pgm)
    model.entity_ids = []  # force the all-dropped state
    with pytest.raises(ValueError, match="ConflictologyModel: no entities to predict"):
        model.predict(df=base_df_pgm, sequence_number=0, output_length=5)


def test_mixture_raises_when_no_entities(partition_dict, targets):
    """MixtureBaseline: empty fitted pool → fail loud (was a silent return {})."""
    df = make_mixture_df()
    model = MixtureBaseline(
        targets=targets, window_months=4, lambda_mix=0.05,
        n_samples=10, partition_dict=partition_dict, loa="pgm",
    )
    model.fit(df)
    model.entity_ids = []  # force the all-dropped state
    with pytest.raises(ValueError, match="MixtureBaseline: no entities to predict"):
        model.predict(df=df, sequence_number=0, output_length=5)


# -----------------------------------------------------------------------
# Green team: ConflictologyModel reproducibility
# -----------------------------------------------------------------------


def test_conflictology_predict_reproducible(base_df_pgm, partition_dict, targets):
    """Same seed produces identical predictions."""
    kwargs = dict(
        targets=targets,
        window_months=4,
        partition_dict=partition_dict,
        loa="pgm",
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
        np.testing.assert_array_equal(r1[target].values, r2[target].values)
