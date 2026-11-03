from __future__ import annotations

import math
import numbers
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from views_baseline.model.defaults import DEFAULT_SEED
from views_baseline.model.frames.input import to_feature_frame, to_level
from views_baseline.model.frames.output import sample_prediction_grid
from views_baseline.model.frames.pooling import window_pool
from views_baseline.model.grid import train_test_boundary
from views_baseline.model.models.distributional.smoothing import (
    SELECTOR_FAILED,
    SMOOTH,
    classify_window,
    draw_fallback,
    round_positive,
    warn_fallbacks,
)

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame

#: Per-cell bandwidth selectors accepted as `bandwidth` (ADR-023 §3).
BANDWIDTH_SELECTORS = ("sj", "silverman")

#: Reflection point: the lower edge of count 1's rounding bin [0.5, 1.5).
REFLECT_AT = 0.5

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def validate_bandwidth(bandwidth) -> None:
    """Fail loud unless `bandwidth` is a positive finite number or a known selector name."""
    if isinstance(bandwidth, str):
        if bandwidth not in BANDWIDTH_SELECTORS:
            raise ValueError(
                f"bandwidth must be a positive number or one of {BANDWIDTH_SELECTORS}, "
                f"got {bandwidth!r}."
            )
        return
    if isinstance(bandwidth, bool) or not isinstance(bandwidth, numbers.Real):
        raise ValueError(
            f"bandwidth must be a positive number or a selector name, got {bandwidth!r}."
        )
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive and finite, got {bandwidth!r}.")


def silverman_bandwidth(x: np.ndarray) -> float | None:
    """Silverman's rule of thumb, as R's ``bw.nrd0``: ``0.9 · min(sd, IQR/1.34) · n^(−1/5)``.

    If the IQR is 0 (common with tied counts), ``sd`` is used alone, as ``bw.nrd0`` does.
    Known to oversmooth multimodal and skewed data (Jones, Marron & Sheather 1996).
    Returns ``None`` if no positive bandwidth results.
    """
    n = x.size
    if n < 2:
        return None
    sd = float(np.std(x, ddof=1))
    q75, q25 = np.percentile(x, [75, 25])
    iqr_scale = float(q75 - q25) / 1.34
    lo = min(sd, iqr_scale) if iqr_scale > 0 else sd
    h = 0.9 * lo * n ** (-0.2)
    return h if h > 0 and math.isfinite(h) else None


def _phi4(u: np.ndarray) -> np.ndarray:
    return (u**4 - 6 * u**2 + 3) * np.exp(-0.5 * u**2) / _SQRT_2PI


def _phi6(u: np.ndarray) -> np.ndarray:
    return (u**6 - 15 * u**4 + 45 * u**2 - 15) * np.exp(-0.5 * u**2) / _SQRT_2PI


def sheather_jones_bandwidth(x: np.ndarray, max_iter: int = 200) -> float | None:
    """Sheather & Jones (1991) solve-the-equation bandwidth, as R's ``stats::bw.SJ``.

    Follows ``bw.SJ(method = "ste")`` term for term: ``scale = min(sd, IQR/1.349)``,
    ``a = 1.24·scale·n^(−1/7)``, ``b = 1.23·scale·n^(−1/9)``, ``α₂(h) = 1.357 [Ŝ(a)/T̂(b)]^(1/7)
    h^(5/7)``, and the root of ``h = [1 / (2√π · n · Ŝ(α₂(h)))]^(1/5)`` searched from
    ``[0.1·hmax, hmax]``, ``hmax = 1.144·scale·n^(−1/5)``, widening alternately up and down
    as R does. Two deliberate differences, both making it *more* exact: ``Ŝ`` and ``T̂`` are
    exact pairwise sums (R bins into ``nb`` = 1000 cells), and the root is solved to
    relative 1e-10 (R stops at ``0.1·lower``). ``tests/test_smoothed_climatology.py`` pins
    it to R 4.6.2 run with fine binning and tolerance.

    Returns ``None`` where R stops with an error: fewer than 2 values, ``scale`` = 0,
    a non-positive ``T̂`` or ``Ŝ``, or no root after 99 widenings. The caller falls back and
    counts it (ADR-023 §3).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        return None
    q75, q25 = np.percentile(x, [75, 25])  # R's IQR(): quantile type 7 = numpy default
    scale = min(float(np.std(x, ddof=1)), float(q75 - q25) / 1.349)
    if not scale > 0:
        return None
    d = (x[:, None] - x[None, :]).ravel()  # all (i, j), diagonal included, as R
    nn = n * (n - 1)

    def s_hat(alpha):
        return float(np.sum(_phi4(d / alpha))) / (nn * alpha**5)

    def t_hat(b):
        return -float(np.sum(_phi6(d / b))) / (nn * b**7)

    t_b = t_hat(1.23 * scale * n ** (-1 / 9))
    s_a = s_hat(1.24 * scale * n ** (-1 / 7))
    if not (t_b > 0 and s_a > 0):
        return None
    alph2 = 1.357 * (s_a / t_b) ** (1 / 7)
    c1 = 1.0 / (2.0 * math.sqrt(math.pi) * n)

    def f(h):
        s = s_hat(alph2 * h ** (5 / 7))
        if not s > 0:
            return None
        return (c1 / s) ** 0.2 - h

    hmax = 1.144 * scale * n ** (-0.2)
    lo, hi = 0.1 * hmax, hmax
    f_lo, f_hi = f(lo), f(hi)
    for itry in range(1, 100):
        if f_lo is None or f_hi is None or f_lo * f_hi <= 0:
            break
        if itry % 2:
            hi *= 1.2
            f_hi = f(hi)
        else:
            lo /= 1.2
            f_lo = f(lo)
    if f_lo is None or f_hi is None or f_lo * f_hi > 0:
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if f_mid is None:
            return None
        if f_lo * f_mid <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
        if hi - lo < 1e-10 * hi:
            break
    h = 0.5 * (lo + hi)
    return h if h > 0 and math.isfinite(h) else None


def cell_bandwidth(bandwidth, positives: np.ndarray) -> float | None:
    """The bandwidth for one cell: the global number, or the selector's per-cell value."""
    if bandwidth == "sj":
        return sheather_jones_bandwidth(positives)
    if bandwidth == "silverman":
        return silverman_bandwidth(positives)
    return float(bandwidth)


def draw_kde_positives(
    positives: np.ndarray, h: float, k: int, rng: np.random.Generator
) -> np.ndarray:
    """``k`` positive draws from the reflected, rounded Gaussian KDE (ADR-023 §3).

    Pick a positive value uniformly, add ``h·Z``, reflect below 0.5 to ``1 − x``, round
    half up. Equal in law to integrating the reflected density over the bins
    ``[j − ½, j + ½)``, ``j ≥ 1``; reflection at 0.5 keeps every draw ≥ 1.
    RNG order: ``k`` integers, then ``k`` normals.
    """
    picks = positives[rng.integers(0, positives.size, size=k)]
    x = picks + h * rng.standard_normal(k)
    x = np.where(x < REFLECT_AT, 2 * REFLECT_AT - x, x)
    return round_positive(x)


class KDEConflictology:
    """Climatology with a zero-preserving kernel density estimate for the positive part
    (ADR-023, C).

    Per entity and target, on the same window as `ConflictologyModel`:

    - a draw is 0 with probability equal to the window's empirical zero rate (as in A);
    - otherwise it comes from a Gaussian KDE of the window's positive values, reflected at
      0.5 and rounded to a count ≥ 1 (`draw_kde_positives`).

    ``bandwidth`` is a required, audited genome key: a positive number (count units, the
    same for every cell), ``"sj"`` (Sheather–Jones per cell) or ``"silverman"`` (per cell;
    oversmooths). Cells that `classify_window` does not smooth, and cells whose selector
    fails, are drawn exactly as A draws them and counted in one WARN per `predict()`.

    **Known bias.** Reflection removes the O(1) boundary loss but leaves an O(h) bias where
    the density's slope at the boundary is non-zero (Marron & Ruppert 1994). Conflict counts
    decay steeply from 1, so C moves mass from 1 to 2, 3, …: ``P(Y=1 | Y≥1)`` is lower than
    the window's share of 1s, more so for larger ``h``.
    """

    distributional = True

    def __init__(
        self,
        targets: list[str],
        window_months: int,
        partition_dict: dict,
        loa: str,
        n_samples: int,
        bandwidth: float | str,
        seed: int = DEFAULT_SEED,
    ):
        validate_bandwidth(bandwidth)
        self.targets = targets
        self.window_months = window_months
        self.partition_dict = partition_dict
        self.loa = loa
        self.n_samples = n_samples
        self.bandwidth = bandwidth
        self.seed = seed
        self.entity_ids = None
        # per cid/target: {"kind", "zero_rate", "positives", "h"} or {"kind", "pool"}
        self.cells = None
        self.fallback_counts: Counter = Counter()

    def fit(self, df: pd.DataFrame | FeatureFrame) -> KDEConflictology:
        _, train_end = train_test_boundary(self.partition_dict)
        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        # Same pool as ConflictologyModel, byte for byte (ADR-023 §1).
        self.entity_ids, pools = window_pool(ff, self.targets, self.window_months, train_end)

        self.cells = {}
        self.fallback_counts = Counter()
        for cid in self.entity_ids:
            self.cells[cid] = {}
            for t in self.targets:
                pool = pools[cid][t]
                kind, positives = classify_window(pool)
                h = cell_bandwidth(self.bandwidth, positives) if kind == SMOOTH else None
                if kind == SMOOTH and h is None:
                    kind = SELECTOR_FAILED
                self.fallback_counts[kind] += 1
                if kind == SMOOTH:
                    self.cells[cid][t] = {
                        "kind": kind,
                        "zero_rate": float(np.mean(pool == 0.0)),
                        "positives": positives,
                        "h": h,
                    }
                else:
                    self.cells[cid][t] = {"kind": kind, "pool": pool}
        return self

    def predict(
        self, df: pd.DataFrame | FeatureFrame, sequence_number: int, output_length: int
    ) -> dict:
        def draw(cid, t, rng):
            cell = self.cells[cid][t]
            if cell["kind"] != SMOOTH:
                return draw_fallback(cell["pool"], self.n_samples, rng)
            draws = np.zeros(self.n_samples, dtype=np.float64)
            is_positive = rng.random(self.n_samples) >= cell["zero_rate"]
            draws[is_positive] = draw_kde_positives(
                cell["positives"], cell["h"], int(is_positive.sum()), rng
            )
            return draws

        warn_fallbacks("KDEConflictology", self.fallback_counts)
        level = to_level(df, loa=self.loa)
        test_start, _ = train_test_boundary(self.partition_dict)
        return sample_prediction_grid(
            entity_ids=self.entity_ids, fitted_state=self.cells,
            model_name="KDEConflictology", targets=self.targets,
            n_samples=self.n_samples, level=level, test_start=test_start,
            sequence_number=sequence_number, output_length=output_length, seed=self.seed,
            draw_cell=draw,
        )
