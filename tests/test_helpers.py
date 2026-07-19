import logging

import numpy as np
import pytest

from views_baseline.model.helpers import (
    build_identifier_arrays,
    build_time_grid,
    filter_entities,
    resolve_level,
    sample_prediction_grid,
    to_prediction_frames,
)

# -----------------------------------------------------------------------
# build_time_grid
# -----------------------------------------------------------------------


def test_build_time_grid_basic():
    assert build_time_grid(100, 0, 5) == [100, 101, 102, 103, 104]


def test_build_time_grid_with_offset():
    assert build_time_grid(100, 3, 2) == [103, 104]


# -----------------------------------------------------------------------
# filter_entities
# -----------------------------------------------------------------------


def test_filter_entities_no_drops(caplog):
    with caplog.at_level(logging.WARNING):
        result = filter_entities([1, 2], {1, 2, 3}, "TestModel")
    assert result == [1, 2]
    assert "entities dropped" not in caplog.text


def test_filter_entities_with_drops(caplog):
    with caplog.at_level(logging.WARNING):
        result = filter_entities([1, 2, 3], {1, 2}, "TestModel")
    assert result == [1, 2]
    assert "TestModel: 1 entities dropped" in caplog.text


# -----------------------------------------------------------------------
# build_identifier_arrays
# -----------------------------------------------------------------------


def test_build_identifier_arrays_entity_time_order():
    """Verifies entity→time nesting order per ADR-011."""
    time_arr, unit_arr = build_identifier_arrays([1, 2], [100, 101])
    np.testing.assert_array_equal(time_arr, [100, 101, 100, 101])
    np.testing.assert_array_equal(unit_arr, [1, 1, 2, 2])


# -----------------------------------------------------------------------
# resolve_level (ADR-020 spatial-level contract; closes C-18)
# -----------------------------------------------------------------------


def test_resolve_level_pgm():
    """Declared pgm + matching index resolves to SpatialLevel.PGM (independent expected)."""
    from views_frames import SpatialLevel

    assert resolve_level("pgm", ("month_id", "priogrid_id")) is SpatialLevel.PGM


def test_resolve_level_cm():
    """Declared cm + matching index resolves to SpatialLevel.CM (independent expected)."""
    from views_frames import SpatialLevel

    assert resolve_level("cm", ("month_id", "country_id")) is SpatialLevel.CM


def test_resolve_level_rejects_unknown_loa():
    """An loa that is not a SpatialLevel value (e.g. the 'pg_id' shorthand) fails loud."""
    with pytest.raises(ValueError, match="Unknown level of analysis"):
        resolve_level("pg_id", ("month_id", "pg_id"))


def test_resolve_level_rejects_declared_observed_mismatch():
    """Declared pgm but a cm entity index is the C-18 silent-mislabel case — must raise."""
    with pytest.raises(ValueError, match="does not match declared level"):
        resolve_level("pgm", ("month_id", "country_id"))


def test_resolve_level_rejects_reversed_index():
    """A (entity, time) reversal (the C-05 case) must raise, not silently mislabel."""
    with pytest.raises(ValueError, match="does not match declared level"):
        resolve_level("pgm", ("priogrid_id", "month_id"))


def test_resolve_level_rejects_non_two_level_index():
    """A 3-level or flat index does not match the declared (time, entity) vocabulary."""
    with pytest.raises(ValueError, match="does not match declared level"):
        resolve_level("pgm", ("month_id", "priogrid_id", "extra"))


# -----------------------------------------------------------------------
# to_prediction_frames — the single views-frames construction seam (ADR-020)
# -----------------------------------------------------------------------


def test_to_prediction_frames_real_leaf_canary():
    """Canary: build a real views_frames.PredictionFrame through the single seam.

    Fails loudly if the leaf constructor, SpatioTemporalIndex, or SpatialLevel API
    shifts again — the early-warning ADR-020 exists to provide (C-16/C-17).
    """
    from views_frames import PredictionFrame, SpatialLevel

    y_pred_by_target = {"y1": np.zeros((4, 3), dtype=np.float64)}
    time = np.array([100, 101, 100, 101])
    unit = np.array([1, 1, 2, 2])

    result = to_prediction_frames(y_pred_by_target, time=time, unit=unit, level=SpatialLevel.PGM)
    pf = result["y1"]

    assert isinstance(pf, PredictionFrame)
    assert pf.values.shape == (4, 3)
    assert pf.index.level is SpatialLevel.PGM
    assert list(pf.index.time) == [100, 101, 100, 101]
    assert list(pf.index.unit) == [1, 1, 2, 2]


def test_to_prediction_frames_rejects_zero_sample_columns():
    """The seam restores the fail-loud guard views_frames dropped (C-20)."""
    from views_frames import SpatialLevel

    with pytest.raises(ValueError, match="at least one sample column"):
        to_prediction_frames(
            {"y1": np.zeros((2, 0), dtype=np.float64)},
            time=np.array([1, 1]),
            unit=np.array([1, 2]),
            level=SpatialLevel.PGM,
        )


# -----------------------------------------------------------------------
# sample_prediction_grid (shared distributional scaffold — C-19 dedup)
# -----------------------------------------------------------------------


def test_sample_prediction_grid_ordering_shape_and_call_count():
    """The scaffold must fill entity→time→target (ADR-011), shape (N, n_samples), and
    call draw_cell exactly once per (entity, time, target)."""
    calls = []

    def draw(cid, target, rng):
        calls.append((cid, target))
        return np.full(3, float(cid))  # marker = the entity id, n_samples=3

    out = sample_prediction_grid(
        entity_ids=[10, 20], fitted_state={10: True, 20: True}, model_name="Stub",
        targets=["a", "b"], n_samples=3, loa="pgm",
        index_names=["month_id", "priogrid_id"], test_start=100,
        sequence_number=0, output_length=2, seed=1, draw_cell=draw,
    )

    assert set(out) == {"a", "b"}
    assert out["a"].values.shape == (2 * 2, 3)  # 2 entities × 2 timesteps
    # each entity's rows carry its marker -> entity→time fill order preserved
    assert (out["a"].values[np.asarray(out["a"].index.unit) == 10] == 10.0).all()
    assert (out["a"].values[np.asarray(out["a"].index.unit) == 20] == 20.0).all()
    # called once per (entity, time, target): 2 × 2 × 2 = 8
    assert len(calls) == 8


def test_sample_prediction_grid_drops_and_requires_entities():
    """Entities absent from `fitted_state` are dropped; if none remain it fails loud."""
    # entity 20 has no fitted state -> dropped; 10 remains
    out = sample_prediction_grid(
        entity_ids=[10, 20], fitted_state={10: True}, model_name="Stub", targets=["a"],
        n_samples=2, loa="pgm", index_names=["month_id", "priogrid_id"],
        test_start=100, sequence_number=0, output_length=1, seed=1,
        draw_cell=lambda cid, t, rng: np.zeros(2),
    )
    assert set(np.asarray(out["a"].index.unit)) == {10}

    with pytest.raises(ValueError, match="no entities to predict"):
        sample_prediction_grid(
            entity_ids=[10, 20], fitted_state={}, model_name="Stub", targets=["a"], n_samples=2,
            loa="pgm", index_names=["month_id", "priogrid_id"], test_start=100,
            sequence_number=0, output_length=1, seed=1,
            draw_cell=lambda cid, t, rng: np.zeros(2),
        )
