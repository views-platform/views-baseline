"""Integration verification (issue #69, epic #66): a baseline ``PredictionFrame`` is
consumed by pipeline-core's ``PredictionFrameEnsembleManager`` aggregation, and the
point-``(N,1)`` vs distributional-``(N,S)`` contract is pinned.

What this test locks in (the answers to the #69 / stale-#12 open question):

* A baseline **point** model emits an ``(N, 1)`` ``PredictionFrame`` (``sample_count == 1``)
  that ``_aggregate_prediction_frames`` accepts and concatenates with other same-``sample_count``
  frames — i.e. baseline output really does flow through the ensemble aggregation seam.
* Ensembles are **homogeneous in ``sample_count``**: mixing a point ``(N, 1)`` frame with a
  distributional ``(N, S>1)`` frame in one pool fails **loud** (pipeline-core #160 / register
  C-205), never a silent unbalanced pool. Deliberate weighted pooling is future work, not a
  silent default.
* ``point-broadcast`` (a ``sample_count == 1`` forecast broadcast across ``S`` draws) is a
  **separate cross-LEVEL** reconciliation mechanism (cm→pgm, ``reconcile_frames`` /
  ``POINT_BROADCAST``), **not** constituent pooling — so it does not apply to same-level
  ensemble aggregation. This test therefore asserts the *loud raise*, which is the intended
  contract for a mixed same-level pool.

Runs against the editable pipeline-core in the ``views_pipeline`` conda env (the unpublished
3.0.0). ``importorskip`` keeps it a no-op where the ensemble module is absent; CI cannot run
it until pipeline-core 3.0.0 publishes.
"""

import numpy as np
import pytest
from conftest import make_dummy_ff

from views_baseline.model.models.point import ZeroModel

# The ensemble aggregation primitive lives in pipeline-core (editable in the conda env).
pfe = pytest.importorskip(
    "views_pipeline_core.managers.ensemble.prediction_frame_ensemble"
)


def _baseline_point_pf(target: str = "y1"):
    """A genuine baseline point PredictionFrame: ZeroModel over a dummy FeatureFrame."""
    ff = make_dummy_ff(time_range=range(110, 126))
    model = ZeroModel(
        targets=["y1", "y2"], partition_dict={"test": (120, 125)}, loa="pgm"
    ).fit(ff)
    return model.predict(df=ff, sequence_number=0, output_length=3)[target]


def test_baseline_point_pf_is_consumed_by_ensemble_aggregation():
    """A baseline point ``(N, 1)`` frame is a valid ensemble constituent: concatenating two of
    them yields ``(N, 2)`` with the shared ``(time, unit)`` index preserved."""
    from views_frames import PredictionFrame

    pf = _baseline_point_pf()
    assert pf.sample_count == 1 and pf.values.shape[1] == 1

    agg = pfe._aggregate_prediction_frames([pf, pf], method="concat")

    assert isinstance(agg, PredictionFrame)
    assert agg.values.shape == (pf.n_rows, 2)
    assert np.array_equal(agg.index.unit, pf.index.unit)
    assert np.array_equal(agg.index.time, pf.index.time)


def test_point_and_distributional_mix_fails_loud():
    """Mixing a point ``(N, 1)`` baseline frame with a distributional ``(N, S)`` frame in one
    ensemble pool raises ``ValueError`` (sample_count guard, pipeline-core #160 / C-205) —
    the intended contract, not a silent unbalanced pool. Point-broadcast does NOT rescue this
    path; it is a separate cross-level reconciliation mechanism."""
    from views_frames import PredictionFrame

    pf = _baseline_point_pf()
    # A distributional constituent's shape on the same rows/index (S = 4 samples).
    distributional = PredictionFrame(
        np.zeros((pf.n_rows, 4), dtype=np.float32), pf.index
    )

    with pytest.raises(ValueError, match="sample_count mismatch"):
        pfe._aggregate_prediction_frames([pf, distributional], method="concat")
