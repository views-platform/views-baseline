"""
Falsification test stubs — seed-wiring bug (falsification audit 2026-06-25).

HARD falsification of ADR-011 §2/§4: `BaselineModelCatalog` does not forward the
`seed` config to `ConflictologyModel`/`MixtureBaseline` (the factory methods
`_get_conflictology_model` / `_get_mixture_model` omit `seed=`), so
`config["seed"]` is silently ignored and every catalog-constructed distributional
model uses the hardcoded default `seed=42`. Consequences: WandB seed sweeps are
inert and any declared non-42 seed is dropped (risk C-10).

These tests encode the corrected contract and FAIL until the catalog forwards
`seed`. Per ADR-021 (Zero-Magic / Explicit-Defaults) the catalog forwards
`config["seed"]` **strictly** and must NOT silently default — the `DEFAULT_SEED`
constant is a sentinel for direct/test construction only.
"""

import numpy as np
import pandas as pd
import pytest

from views_baseline.model.baseline import DEFAULT_SEED
from views_baseline.model.catalog import BaselineModelCatalog

_PARTITION = {"test": (493, 540)}


def _df():
    """Balanced pgm-style panel with varied positive values (Mixture global pool)."""
    idx = pd.MultiIndex.from_product(
        [range(440, 500), [1, 2, 3]], names=["month_id", "priogrid_id"]
    )
    vals = np.random.default_rng(0).integers(0, 6, size=len(idx)).astype(float)
    return pd.DataFrame({"y1": vals}, index=idx)


def _catalog(seed):
    cfg = {
        "targets": ["y1"],
        "window_months": 6,
        "n_samples": 16,
        "lambda_mix": 0.5,
        "seed": seed,
    }
    return BaselineModelCatalog(config=cfg, partition_dict=_PARTITION, loa="pgm")


@pytest.mark.parametrize("algo", ["ConflictologyModel", "MixtureBaseline"])
def test_catalog_forwards_seed(algo):
    """The catalog must forward config['seed'] to the model (ADR-011 §2)."""
    model = _catalog(seed=7).get_model(algo)
    assert model.seed == 7, (
        f"{algo}: catalog did not forward config seed (model.seed={model.seed}); "
        f"config['seed'] is silently ignored (C-10)."
    )


@pytest.mark.parametrize("algo", ["ConflictologyModel", "MixtureBaseline"])
def test_different_seed_produces_different_output(algo):
    """Different config seeds must produce different draws (ADR-011 §4, converse)."""
    df = _df()
    m1 = _catalog(seed=123).get_model(algo)
    m1.fit(df)
    o1 = m1.predict(df=df, sequence_number=0, output_length=3)["y1"].values
    m2 = _catalog(seed=456).get_model(algo)
    m2.fit(df)
    o2 = m2.predict(df=df, sequence_number=0, output_length=3)["y1"].values
    assert not np.array_equal(o1, o2), (
        f"{algo}: seeds 123 and 456 produced identical output — the seed sweep "
        f"is inert because config['seed'] is ignored (C-10)."
    )


@pytest.mark.parametrize("algo", ["ConflictologyModel", "MixtureBaseline"])
def test_catalog_requires_seed(algo):
    """ADR-021: the catalog must NOT silently default seed — a config without it fails loud."""
    cfg = {"targets": ["y1"], "window_months": 6, "n_samples": 8, "lambda_mix": 0.5}
    cat = BaselineModelCatalog(config=cfg, partition_dict=_PARTITION, loa="pgm")
    with pytest.raises(ValueError, match="seed"):
        cat.get_model(algo)


def test_direct_construction_uses_default_seed_sentinel():
    """Direct construction without a seed uses the single DEFAULT_SEED sentinel (ADR-021).

    This is the construction-ergonomics path, not the production path (which is the catalog).
    """
    from views_baseline.model.baseline import ConflictologyModel, MixtureBaseline

    conf = ConflictologyModel(
        targets=["y1"], window_months=6, partition_dict=_PARTITION, loa="pgm", n_samples=8
    )
    mix = MixtureBaseline(
        targets=["y1"], window_months=6, lambda_mix=0.5, n_samples=8,
        partition_dict=_PARTITION, loa="pgm",
    )
    assert conf.seed == DEFAULT_SEED
    assert mix.seed == DEFAULT_SEED
