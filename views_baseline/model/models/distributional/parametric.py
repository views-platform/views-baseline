from __future__ import annotations

from typing import TYPE_CHECKING

from views_baseline.model.defaults import DEFAULT_SEED
from views_baseline.model.distributions import (
    NATIVE_ZERO_FAMILIES,
    TRANSFORMS,
    clamp_log,
    fit_family,
    sample_family,
    validate_family_transform,
)
from views_baseline.model.frames.input import to_feature_frame, to_index
from views_baseline.model.frames.output import sample_prediction_grid
from views_baseline.model.frames.pooling import window_pool

if TYPE_CHECKING:
    import pandas as pd
    from views_frames import FeatureFrame


class ParametricConflictology:
    """No-hurdle parametric climatology (ADR-022).

    Fits a single native-zero distribution (`family` ∈ {"nb", "zinb"}) to each
    entity's `window_pool` and samples `n_samples` i.i.d. per cell — a parametric analogue
    of `ConflictologyModel` (same pool). `transform` must be "none" for count families
    (`log1p` fails loud). Output via the single `to_prediction_frames` seam.
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
        if family not in NATIVE_ZERO_FAMILIES:
            raise ValueError(
                f"ParametricConflictology supports native-zero families "
                f"{sorted(NATIVE_ZERO_FAMILIES)}, got {family!r}."
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
        self.params = None

    def fit(self, df: pd.DataFrame | FeatureFrame) -> "ParametricConflictology":
        test_start = self.partition_dict["test"][0]
        train_end = test_start - 1
        ff = to_feature_frame(df, loa=self.loa, targets=self.targets)
        # `pools` is local — predict reads only self.params, so retaining it would pin the
        # entities x window array on the fitted object for nothing (C-37).
        self.entity_ids, pools = window_pool(
            ff, self.targets, self.window_months, train_end
        )
        forward, _ = TRANSFORMS[self.transform]
        self.params = {
            cid: {t: fit_family(self.family, forward(pools[cid][t])) for t in self.targets}
            for cid in self.entity_ids
        }
        return self

    def predict(
        self, df: pd.DataFrame | FeatureFrame, sequence_number: int, output_length: int
    ) -> dict:
        _, inverse = TRANSFORMS[self.transform]

        def draw(cid, t, rng):
            draws = sample_family(self.family, self.params[cid][t], self.n_samples, rng)
            return inverse(clamp_log(draws)) if self.transform != "none" else draws

        level, _, _ = to_index(df, loa=self.loa)
        return sample_prediction_grid(
            entity_ids=self.entity_ids, fitted_state=self.params,
            model_name="ParametricConflictology",
            targets=self.targets, n_samples=self.n_samples, level=level,
            test_start=self.partition_dict["test"][0],
            sequence_number=sequence_number, output_length=output_length, seed=self.seed,
            draw_cell=draw,
        )
