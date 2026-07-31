from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from views_baseline.model.defaults import DEFAULT_SEED
from views_baseline.model.distributions import (
    CONTINUOUS_FAMILIES,
    TRANSFORMS,
    clamp_floor,
    clamp_log,
    fit_family,
    sample_family,
    validate_family_transform,
)
from views_baseline.model.frames.input import to_feature_frame, to_level
from views_baseline.model.frames.output import sample_prediction_grid
from views_baseline.model.frames.pooling import window_pool
from views_baseline.model.grid import train_test_boundary

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame


class ParametricHurdleConflictology:
    """Hurdle parametric climatology (ADR-022).

    Per entity/target: a zero-spike (`w` = empirical zero-rate of the window, Bernoulli)
    plus a continuous positive-part family (`family` ∈ {"lognormal","gumbel","gamma"})
    fit to the *positive* window values. `transform` (`none`/`log1p`) is applied to the
    positive part before fitting and **inverted per sampled draw** (Jensen-safe), then
    clamped (`EMIT_LOG_CEIL`). Mirrors Vesco et al. 2026's RVI mixture (spike-at-0 here).
    """

    distributional = True

    def __init__(
        self,
        targets: list[str],
        window_months: int,
        partition_dict: dict,
        loa: str,
        n_samples: int,
        family: str,
        transform: str = "none",
        seed: int = DEFAULT_SEED,
    ):
        if family not in CONTINUOUS_FAMILIES:
            raise ValueError(
                f"ParametricHurdleConflictology supports continuous positive-part families "
                f"{sorted(CONTINUOUS_FAMILIES)}, got {family!r}."
            )
        validate_family_transform(family, transform)
        self.targets = targets
        self.window_months = window_months
        self.partition_dict = partition_dict
        self.loa = loa
        self.n_samples = n_samples
        self.family = family
        self.transform = transform
        self.seed = seed
        self.entity_ids = None
        self.params = None  # per cid/target: {"zero_rate": w, "pos": params-or-None}

    def fit(self, df: pd.DataFrame | FeatureFrame) -> "ParametricHurdleConflictology":
        test_start, train_end = train_test_boundary(self.partition_dict)
        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        # `pools` is local — predict reads only self.params (C-37).
        self.entity_ids, pools = window_pool(
            ff, self.targets, self.window_months, train_end
        )
        forward, _ = TRANSFORMS[self.transform]
        self.params = {}
        for cid in self.entity_ids:
            self.params[cid] = {}
            for t in self.targets:
                pool = pools[cid][t]
                pos = pool[pool > 0]
                if pos.size == 0:  # all-zero window -> point mass at 0 (w=1)
                    self.params[cid][t] = {"zero_rate": 1.0, "pos": None}
                else:
                    self.params[cid][t] = {
                        "zero_rate": float(np.mean(pool == 0.0)),
                        "pos": fit_family(self.family, forward(pos)),
                    }
        return self

    def predict(
        self, df: pd.DataFrame | FeatureFrame, sequence_number: int, output_length: int
    ) -> dict:
        _, inverse = TRANSFORMS[self.transform]

        def draw(cid, t, rng):
            hp = self.params[cid][t]
            draws = np.zeros(self.n_samples, dtype=np.float64)
            is_positive = rng.random(self.n_samples) >= hp["zero_rate"]
            n_pos = int(is_positive.sum())
            if n_pos > 0 and hp["pos"] is not None:
                pos = sample_family(self.family, hp["pos"], n_pos, rng)
                if self.transform != "none":
                    pos = inverse(clamp_log(pos))
                # gumbel_r has ℝ support — floor at 0 so no negative magnitude escapes
                draws[is_positive] = clamp_floor(pos)
            return draws

        level = to_level(df, loa=self.loa)
        test_start, _ = train_test_boundary(self.partition_dict)
        return sample_prediction_grid(
            entity_ids=self.entity_ids, fitted_state=self.params,
            model_name="ParametricHurdleConflictology", targets=self.targets,
            n_samples=self.n_samples, level=level,
            test_start=test_start, sequence_number=sequence_number,
            output_length=output_length, seed=self.seed, draw_cell=draw,
        )
