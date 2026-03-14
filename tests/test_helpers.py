import logging

import numpy as np

from views_baseline.model.helpers import (
    build_identifier_arrays,
    build_time_grid,
    filter_entities,
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
