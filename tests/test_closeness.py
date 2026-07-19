"""Self-validation of the closeness harness (ADR-022 / epic #33 S3).

The machinery is validated with no parametric model: conflictology-vs-conflictology
must sit at the null (C2ST ~0.5), and a deliberately shifted sample must be flagged.
"""

import numpy as np
import pandas as pd
from views_frames import PredictionFrame

from views_baseline.evaluation.closeness import (
    c2st_1nn,
    compare_models,
    equivalence,
    summarize,
)
from views_baseline.model.baseline import ConflictologyModel

_PARTITION = {"test": (493, 540)}
_DELTA = {"c2st": 0.05, "wasserstein": 0.5}


def _df():
    idx = pd.MultiIndex.from_product(
        [range(440, 500), [1, 2, 3]], names=["month_id", "priogrid_id"]
    )
    vals = np.random.default_rng(0).integers(0, 8, size=len(idx)).astype(float)
    return pd.DataFrame({"y1": vals}, index=idx)


def _conflictology(seed):
    m = ConflictologyModel(
        targets=["y1"], window_months=12, partition_dict=_PARTITION,
        loa="pgm", n_samples=1000, seed=seed,
    )
    m.fit(_df())
    return m.predict(df=_df(), sequence_number=0, output_length=2)


def test_c2st_identical_samples_near_half():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=1000), rng.normal(size=1000)
    assert abs(c2st_1nn(a, b, rng) - 0.5) < 0.06


def test_c2st_constant_samples_near_half():
    # both models emit the same constant -> indistinguishable, must NOT score ~1.0
    a = np.full(1000, 3.0)
    b = np.full(1000, 3.0)
    assert abs(c2st_1nn(a, b, np.random.default_rng(2)) - 0.5) < 0.06


def test_c2st_shifted_samples_high():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=1000), rng.normal(size=1000) + 50
    assert c2st_1nn(a, b, rng) > 0.95


def test_same_model_null_is_indistinguishable():
    null = summarize(compare_models(_conflictology(2), _conflictology(1)))
    assert abs(null[("y1", "all")]["c2st"]["median"] - 0.5) < 0.06


def test_closeness_summary_is_activity_stratified():
    # the harness stratifies per-cell by the reference's activity regime (zero/low/high);
    # summarize must expose those strata plus 'all', each with the three metrics.
    summ = summarize(compare_models(_conflictology(2), _conflictology(1)))
    strata = {key[1] for key in summ}
    assert "all" in strata
    assert strata & {"zero", "low", "high"}  # at least one activity stratum populated
    for row in summ.values():
        assert set(row) >= {"n", "wasserstein", "energy", "c2st"}


def test_shifted_model_is_flagged_not_equivalent():
    ref = _conflictology(1)
    shifted = {t: PredictionFrame(pf.values + 100.0, pf.index) for t, pf in ref.items()}
    cand = summarize(compare_models(shifted, ref))
    null = summarize(compare_models(_conflictology(2), _conflictology(3)))
    verdict = equivalence(cand, null, _DELTA)
    assert cand[("y1", "all")]["c2st"]["median"] > 0.9
    assert verdict[("y1", "all")]["equivalent"] is False
