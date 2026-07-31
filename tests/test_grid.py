"""Guards for ``model.grid.train_test_boundary`` — the single home of the train/test
boundary convention AND the single place ``partition_dict`` shape is validated at the model
boundary (C-06).

Two properties are pinned here: (1) the helper's contract (returns ``(test_start, train_end)``
with ``train_end = test_start - 1``) and its fail-loud rejection of a malformed ``partition_dict``;
(2) that the distributional ``predict()`` path now routes through the helper (closing the former
Pattern-B bypass that read ``self.partition_dict["test"][0]`` inline), so a malformed dict fails
loudly at the boundary rather than deep inside sampling.
"""

import numpy as np
import pytest

from views_baseline.model.grid import train_test_boundary


def test_returns_start_and_train_end():
    assert train_test_boundary({"test": (150, 155)}) == (150, 149)


def test_accepts_numpy_integer_start_and_list_value():
    # np.integer start and a list (not tuple) are both legitimate shapes.
    assert train_test_boundary({"test": [np.int64(200), 210]}) == (200, 199)


def test_rejects_non_dict():
    with pytest.raises(ValueError, match="must be a dict with a 'test' key"):
        train_test_boundary(("test", (150, 155)))


def test_rejects_missing_test_key():
    with pytest.raises(ValueError, match="must be a dict with a 'test' key"):
        train_test_boundary({"train": (0, 149)})


def test_rejects_empty_test_tuple():
    with pytest.raises(ValueError, match="non-empty .* tuple of ints"):
        train_test_boundary({"test": ()})


def test_rejects_non_integer_start():
    with pytest.raises(ValueError, match="tuple of ints"):
        train_test_boundary({"test": (150.5, 155)})


def test_rejects_bool_start():
    # bool is an int subclass in Python — reject it explicitly, it is never a valid time id.
    with pytest.raises(ValueError, match="tuple of ints"):
        train_test_boundary({"test": (True, 155)})


def test_distributional_predict_validates_partition_dict():
    """The predict path routes through the guard (Pattern-B bypass closed).

    A ``ConflictologyModel`` fitted with a valid partition, then handed a malformed
    ``partition_dict``, must raise ``ValueError`` from ``predict()`` — not a deep
    ``KeyError``/``IndexError`` inside grid construction.
    """
    pytest.importorskip("views_frames")
    from conftest import make_dummy_df

    from views_baseline.model.models.distributional import ConflictologyModel

    df = make_dummy_df()
    model = ConflictologyModel(
        targets=["y1", "y2"], partition_dict={"test": (493, 540)}, loa="pgm",
        window_months=3, n_samples=8, seed=42,
    ).fit(df)

    model.partition_dict = {"train": (0, 492)}  # malformed: no "test" key
    with pytest.raises(ValueError, match="must be a dict with a 'test' key"):
        model.predict(df=df, sequence_number=0, output_length=2)
