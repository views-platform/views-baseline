from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from views_baseline.model.distributions import (
    CONTINUOUS_FAMILIES,
    NATIVE_ZERO_FAMILIES,
    TRANSFORMS,
    clamp_floor,
    clamp_log,
    fit_family,
    sample_family,
    validate_family_transform,
)
from views_baseline.model.helpers import (
    build_prediction_frame,
    build_time_grid,
    filter_entities,
    require_entities,
    resolve_level,
    sample_prediction_grid,
)
from views_baseline.model.pooling import window_pool

logger = logging.getLogger(__name__)

# Zero-Magic / Explicit-Defaults (ADR-021): the single source of truth for the
# default RNG seed. It is a **sentinel for direct/test construction only** — production
# configs MUST declare `seed` (a required, audited genome key). The catalog reads
# ``config["seed"]`` strictly and never falls back to this constant.
DEFAULT_SEED = 42


class ZeroModel:
    def __init__(self, targets: list[str], partition_dict: dict, loa: str):
        """
        Baseline model that predicts 0 for all targets.
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.time_idx = None
        self.entity_idx = None

    def fit(self, df: pd.DataFrame) -> ZeroModel:
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]
        return self

    def predict(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int,
    ) -> dict:
        """
        Predicts zero for each target variable over output_length time steps
        starting from test_start + sequence_number.
        """
        test_start = self.partition_dict["test"][0]
        train_end = test_start - 1

        logger.info(f"Generating ZeroModel predictions on level: {self.entity_idx}")

        entity_ids = (
            df.loc[df.index.get_level_values(self.time_idx) == train_end]
            .index.get_level_values(self.entity_idx)
            .unique()
        )
        require_entities(entity_ids, "ZeroModel")
        time_ids = build_time_grid(test_start, sequence_number, output_length)
        level = resolve_level(self.loa, df.index.names)

        return build_prediction_frame(
            entity_ids=entity_ids,
            time_ids=time_ids,
            targets=self.targets,
            value_fn=lambda cid, target: 0.0,
            level=level,
        )


class LocfModel:
    def __init__(self, targets: list[str], partition_dict: dict, loa: str):
        """
        Baseline model that carries forward the last observation for each entity and target.
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.last_observations = None
        self.time_idx = None
        self.entity_idx = None

    def fit(self, df: pd.DataFrame) -> LocfModel:
        """
        Store the last available observation before the test period for each entity.
        """
        test_start = self.partition_dict["test"][0]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]
        df = df[df.index.get_level_values(self.time_idx) < test_start]

        logger.info(f"Fitting LocfModel on level: {self.entity_idx}")
        df = df.sort_index(level=[self.entity_idx, self.time_idx])
        self.last_observations = df.groupby(self.entity_idx)[self.targets].last()
        return self

    def predict(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int,
    ) -> dict:
        """
        Repeats the last observed value for each target and entity over the forecast horizon.
        """
        test_start = self.partition_dict["test"][0]
        train_end = test_start - 1

        logger.info(f"Generating LOCF predictions on level: {self.entity_idx}")

        entity_ids = (
            df.loc[df.index.get_level_values(self.time_idx) == train_end]
            .index.get_level_values(self.entity_idx)
            .unique()
        )
        entity_ids = filter_entities(entity_ids, self.last_observations.index, "LocfModel")
        require_entities(entity_ids, "LocfModel")
        time_ids = build_time_grid(test_start, sequence_number, output_length)
        level = resolve_level(self.loa, df.index.names)

        return build_prediction_frame(
            entity_ids=entity_ids,
            time_ids=time_ids,
            targets=self.targets,
            value_fn=lambda cid, target: self.last_observations.loc[cid, target],
            level=level,
        )


class AverageModel:
    def __init__(self, targets: list[str], window_months: int, partition_dict: dict, loa: str):
        """
        Baseline model that carries forward the average of the last window_months months
        specified in the config file.
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.mean = None
        self.window_months = window_months
        self.time_idx = None
        self.entity_idx = None

    def fit(self, df: pd.DataFrame) -> AverageModel:
        """
        Get the average of the last m observations before the test partition for each entity.
        """
        test_start = self.partition_dict["test"][0]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]
        df = df[df.index.get_level_values(self.time_idx) < test_start]

        logger.info(f"Fitting AverageModel on level: {self.entity_idx}")

        df = df.sort_index(level=[self.entity_idx, self.time_idx])

        # Group by entity and take mean of last `months` rows
        entity_means = df.groupby(level=self.entity_idx, group_keys=False).apply(
            lambda g: g.tail(self.window_months)[self.targets].mean()
        )
        self.mean = entity_means
        return self

    def predict(
        self,
        df: pd.DataFrame,
        sequence_number: int,
        output_length: int,
    ) -> dict:
        """
        Repeats the average over the last m months for each target
        and entity over the forecast horizon.
        """
        test_start = self.partition_dict["test"][0]
        train_end = test_start - 1

        logger.info(f"Generating average predictions on level: {self.entity_idx}")

        entity_ids = (
            df.loc[df.index.get_level_values(self.time_idx) == train_end]
            .index.get_level_values(self.entity_idx)
            .unique()
        )
        entity_ids = filter_entities(entity_ids, self.mean.index, "AverageModel")
        require_entities(entity_ids, "AverageModel")
        time_ids = build_time_grid(test_start, sequence_number, output_length)
        level = resolve_level(self.loa, df.index.names)

        return build_prediction_frame(
            entity_ids=entity_ids,
            time_ids=time_ids,
            targets=self.targets,
            value_fn=lambda cid, target: self.mean.loc[cid, target],
            level=level,
        )


class ConflictologyModel:
    distributional = True

    def __init__(
        self,
        targets: list[str],
        window_months: int,
        partition_dict: dict,
        loa: str,
        n_samples: int,
        seed: int = DEFAULT_SEED,
    ):
        """
        Climatology baseline that resamples with replacement from the last window_months
        of data for a given cm/pgm, producing n_samples i.i.d. draws per cell.
        """
        self.targets = targets
        self.partition_dict = partition_dict
        self.loa = loa
        self.window_months = window_months
        self.n_samples = n_samples
        self.seed = seed
        self.time_idx = None
        self.entity_idx = None
        self.hist_per_entity = None
        self.entity_ids = None

    def fit(self, df: pd.DataFrame) -> ConflictologyModel:
        """
        Extract and store the last `window_months` of history per entity before the test period.
        """
        test_start = self.partition_dict["test"][0]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]
        train_end = test_start - 1

        # Shared with the parametric climatology models so the per-entity pools are
        # byte-identical (ADR-022). Behaviour unchanged from the previous inline form.
        self.entity_ids, self.hist_per_entity = window_pool(
            df, self.time_idx, self.entity_idx, self.targets, self.window_months, train_end
        )

        return self

    def predict(
        self, df: pd.DataFrame, sequence_number: int, output_length: int
    ) -> dict:
        """
        Return predictions as Dict[str, PredictionFrame] — one PF per target.
        Each PF has y_pred shape (N, n_samples) with resampled draws.
        """
        return sample_prediction_grid(
            entity_ids=self.entity_ids, valid=self.hist_per_entity,
            model_name="ConflictologyModel", targets=self.targets, n_samples=self.n_samples,
            loa=self.loa, index_names=df.index.names, test_start=self.partition_dict["test"][0],
            sequence_number=sequence_number, output_length=output_length, seed=self.seed,
            draw_cell=lambda cid, t, rng: rng.choice(
                self.hist_per_entity[cid][t], size=self.n_samples, replace=True
            ),
        )


class MixtureBaseline:
    distributional = True

    def __init__(
        self,
        targets: list[str],
        window_months: int,
        lambda_mix: float,
        n_samples: int,
        partition_dict: dict,
        loa: str,
        seed: int = DEFAULT_SEED,
    ):
        """
        Mixture empirical baseline that combines local history with a global
        positive pool to avoid the zero-probability trap.
        """
        self.targets = targets
        self.window_months = window_months
        self.lambda_mix = lambda_mix
        self.n_samples = n_samples
        self.partition_dict = partition_dict
        self.loa = loa
        self.seed = seed
        self.time_idx = None
        self.entity_idx = None
        self.local_pool = None
        self.global_pool = None
        self.entity_ids = None

    def fit(self, df: pd.DataFrame) -> MixtureBaseline:
        test_start = self.partition_dict["test"][0]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]

        train_df = df[df.index.get_level_values(self.time_idx) < test_start]
        train_df = train_df.sort_index(level=[self.entity_idx, self.time_idx])

        train_end = test_start - 1
        self.entity_ids = (
            train_df.loc[train_df.index.get_level_values(self.time_idx) == train_end]
            .index.get_level_values(self.entity_idx)
            .unique()
        )

        # Local pool: last window_months values per entity per target
        self.local_pool = {}
        for cid in self.entity_ids:
            entity_data = train_df.xs(cid, level=self.entity_idx, drop_level=False)
            tail = entity_data.tail(self.window_months)
            self.local_pool[cid] = {
                t: np.array(tail[t].tolist(), dtype=np.float64) for t in self.targets
            }

        # Global pool: all positive values per target across all entities
        self.global_pool = {}
        for t in self.targets:
            vals = train_df[t].values
            self.global_pool[t] = vals[vals > 0].astype(np.float64)

        return self

    def _sample(self, cid: int, target: str, rng: np.random.Generator) -> np.ndarray:
        """Generate n_samples by mixing local and global pools."""
        local = self.local_pool[cid][target]
        gpool = self.global_pool[target]

        if len(gpool) == 0:
            # No positive values in training data — local only
            return rng.choice(local, size=self.n_samples)

        use_global = rng.random(self.n_samples) < self.lambda_mix
        n_global = int(np.sum(use_global))
        n_local = self.n_samples - n_global

        samples = np.empty(self.n_samples, dtype=np.float64)
        samples[use_global] = rng.choice(gpool, size=n_global)
        samples[~use_global] = rng.choice(local, size=n_local)
        return samples

    def predict(
        self, df: pd.DataFrame, sequence_number: int, output_length: int
    ) -> dict:
        return sample_prediction_grid(
            entity_ids=self.entity_ids, valid=self.local_pool, model_name="MixtureBaseline",
            targets=self.targets, n_samples=self.n_samples, loa=self.loa,
            index_names=df.index.names, test_start=self.partition_dict["test"][0],
            sequence_number=sequence_number, output_length=output_length, seed=self.seed,
            draw_cell=self._sample,
        )


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
        self.time_idx = None
        self.entity_idx = None
        self.entity_ids = None
        self.pools = None
        self.params = None

    def fit(self, df: pd.DataFrame) -> "ParametricConflictology":
        test_start = self.partition_dict["test"][0]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]
        train_end = test_start - 1
        self.entity_ids, self.pools = window_pool(
            df, self.time_idx, self.entity_idx, self.targets, self.window_months, train_end
        )
        forward, _ = TRANSFORMS[self.transform]
        self.params = {
            cid: {t: fit_family(self.family, forward(self.pools[cid][t])) for t in self.targets}
            for cid in self.entity_ids
        }
        return self

    def predict(self, df: pd.DataFrame, sequence_number: int, output_length: int) -> dict:
        _, inverse = TRANSFORMS[self.transform]

        def draw(cid, t, rng):
            draws = sample_family(self.family, self.params[cid][t], self.n_samples, rng)
            return inverse(clamp_log(draws)) if self.transform != "none" else draws

        return sample_prediction_grid(
            entity_ids=self.entity_ids, valid=self.params, model_name="ParametricConflictology",
            targets=self.targets, n_samples=self.n_samples, loa=self.loa,
            index_names=df.index.names, test_start=self.partition_dict["test"][0],
            sequence_number=sequence_number, output_length=output_length, seed=self.seed,
            draw_cell=draw,
        )


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
        self.time_idx = None
        self.entity_idx = None
        self.entity_ids = None
        self.pools = None
        self.params = None  # per cid/target: {"zero_rate": w, "pos": params-or-None}

    def fit(self, df: pd.DataFrame) -> "ParametricHurdleConflictology":
        test_start = self.partition_dict["test"][0]
        self.time_idx = df.index.names[0]
        self.entity_idx = df.index.names[1]
        train_end = test_start - 1
        self.entity_ids, self.pools = window_pool(
            df, self.time_idx, self.entity_idx, self.targets, self.window_months, train_end
        )
        forward, _ = TRANSFORMS[self.transform]
        self.params = {}
        for cid in self.entity_ids:
            self.params[cid] = {}
            for t in self.targets:
                pool = self.pools[cid][t]
                pos = pool[pool > 0]
                if pos.size == 0:  # all-zero window -> point mass at 0 (w=1)
                    self.params[cid][t] = {"zero_rate": 1.0, "pos": None}
                else:
                    self.params[cid][t] = {
                        "zero_rate": float(np.mean(pool == 0.0)),
                        "pos": fit_family(self.family, forward(pos)),
                    }
        return self

    def predict(self, df: pd.DataFrame, sequence_number: int, output_length: int) -> dict:
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

        return sample_prediction_grid(
            entity_ids=self.entity_ids, valid=self.params,
            model_name="ParametricHurdleConflictology", targets=self.targets,
            n_samples=self.n_samples, loa=self.loa, index_names=df.index.names,
            test_start=self.partition_dict["test"][0], sequence_number=sequence_number,
            output_length=output_length, seed=self.seed, draw_cell=draw,
        )
