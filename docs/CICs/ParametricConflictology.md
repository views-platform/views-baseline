# Class Intent Contract: ParametricConflictology

**Date:** 2026-07-18
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-001, ADR-003, ADR-005, ADR-006, ADR-008, ADR-009, ADR-011, ADR-020, ADR-021, ADR-022

---

## Purpose

`ParametricConflictology` is the **no-hurdle, parametric** counterpart of `ConflictologyModel`. Instead of resampling an entity's trailing window empirically, it **fits a single native-zero distribution** (`family`) to that same window (via `window_pool`, byte-identical to conflictology) and draws `n_samples` i.i.d. per cell from the fitted law. It is a proper generative baseline — extensible (covariates, pooling later) — designed to sit as close to conflictology as possible while being a distribution, not a bootstrap.

The shipped families are **`nb`** (negative binomial, Gamma-Poisson mixture) and **`zinb`** (zero-inflated NB). The S8 closeness experiment found `nb` closest to conflictology on the **C2ST / indistinguishability** axis (its discrete integer support matches conflictology's resampled window values); the later platform-native dwarf evaluation found **`zinb` the best forecaster** — its structural zero-spike fixes what plain NB under-models on 99%-zero data (`reports/dwarf_forecast_evaluation/FINDINGS.md`).

---

## Non-Goals

- Does **not** pool across entities; each fit uses only that entity's own window (shared `window_pool`).
- Does **not** add a separate *hurdle* stage (a zero-gate + positive-part family) — the native-zero family carries its own zero mass (that is the hurdle model's job). Note `zinb` includes an in-distribution structural zero-inflation term, but that is part of the single native-zero distribution, not a separate hurdle.
- Does **not** accept a continuous positive-part family (`lognormal`/`gumbel`/`gamma`) — those are hurdle-only; passing one **fails loud** at construction.
- Does **not** accept `transform="log1p"` for a count family — `log1p` of a count is not a count; it **fails loud** (ADR-021).
- Does **not** apply magic defaults to `family`, `transform`, or `seed` — all are required, audited genome keys.
- Does **not** include Tweedie — evaluated in S9 and excluded on family-feasibility grounds (ADR-022).

---

## Responsibilities and Guarantees

- `__init__` validates `family ∈ NATIVE_ZERO_FAMILIES` (else `ValueError`) and `validate_family_transform(family, transform)` (rejects `log1p` on a count family). No numeric validation of `window_months`/`n_samples` (consistent with the other baselines).
- `fit(df)` extracts per-entity/per-target windows via the shared `window_pool(...)` (identical to `ConflictologyModel`), forward-transforms each pool, and stores `self.params[entity][target] = fit_family(family, forward(pool))`. Returns `self`.
- `predict(df, sequence_number, output_length)` returns `dict[str, PredictionFrame]`, one PF per target, each with `y_pred` shape `(N, n_samples)` where `N = len(entities_with_params) * output_length`.
- Sampling uses a **fresh** `np.random.default_rng(self.seed)` per `predict()` call; the RNG advances in **entity→time→target** order (ADR-011), making output deterministic given `seed`.
- When `transform != "none"`, each drawn sample is detransformed **per sample** via `inverse(clamp_log(draw))` (`clamp_log` caps log-space at `EMIT_LOG_CEIL` before `expm1`; Jensen-safe). For the shipped `nb`/`none` configuration this path is inactive.
- For `family="zinb"`, `fit_family` derives a structural zero-inflation `π` — fixing the NB dispersion from the window and solving the inflated NB mean so both the sample **mean** and the empirical **zero-rate** match exactly — and falls back to plain NB when the NB already explains the zeros (no magic `π`, ADR-021).
- Degenerate windows fail **safe and loud-ish** inside `distributions.py`: an all-zero window → point-mass at 0 (matches conflictology exactly); an underdispersed window (`var ≤ mean`, which includes a single/constant positive value) → **Poisson fallback** for the NB (and the ZINB NB component) — note this has spread, so `nb` does *not* point-mass on a single positive value (that is a continuous/positive-part behaviour); all fallbacks emit a `WARNING`.
- Entities without params are dropped with a `WARNING`; if none remain, `require_entities` raises a descriptive `ValueError` (fail-loud).
- Output is constructed exclusively through the single `to_prediction_frames` seam (ADR-020); `level` is resolved and validated from the declared `loa` via `resolve_level`.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets` | `list[str]` | Column names in `df`; must exist at fit time. |
| `window_months` | `int` | Trailing training months per entity (shared `window_pool`). |
| `partition_dict` | `dict` | Must contain `"test"` → `(test_start, test_end)`; `train_end = test_start − 1`. |
| `loa` | `str` | Declared level of analysis; resolved to a `SpatialLevel` via `resolve_level` and validated against the index. |
| `n_samples` | `int` | Draws per cell. Required — no default. |
| `family` | `str` | Required, audited genome key. Must be in `NATIVE_ZERO_FAMILIES` (`{"nb", "zinb"}`). |
| `transform` | `str` | Required, audited genome key. `"none"` or `"log1p"`; `log1p` is illegal for count families. |
| `seed` | `int` | Required, audited genome key (ADR-021). `DEFAULT_SEED` (42) is a direct-construction sentinel only, never the production path. |
| `df` (fit/predict) | `pd.DataFrame` | 2-level MultiIndex; level 0 = time, level 1 = entity. Pandas is confined to the `window_pool`/input seam. |
| `sequence_number` | `int` | Offset from `test_start` for the prediction window start. |
| `output_length` | `int` | Number of forecast timesteps. Required — no default. |

---

## Outputs and Side Effects

- **`fit()`**: returns `self`; sets `self.time_idx`, `self.entity_idx`, `self.entity_ids`, `self.pools`, `self.params`. May emit `WARNING` on degenerate-window fallbacks. No external side effects.
- **`predict()`**: returns `dict[str, PredictionFrame]`; raises `ValueError` (via `require_entities`) if no entities have params; emits `WARNING` on entity drops or `clamp_log` clamping. Output is stochastic but reproducible given `seed`. `views_frames` is imported lazily inside `to_prediction_frames`.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| `family` not native-zero (e.g. `"lognormal"`) | `ValueError` at `__init__` | Fail-loud; continuous families are hurdle-only. |
| `family="nb"` + `transform="log1p"` | `ValueError` at `__init__` | `validate_family_transform` — count + log1p is a contract violation (ADR-021). |
| Unknown `family` | `ValueError` at `fit` | `fit_family` rejects unknown names. |
| All-zero window | Point-mass at 0 + `WARNING` | Matches conflictology exactly. |
| Underdispersed window (`var ≤ mean`, incl. single/constant positive) | Poisson fallback + `WARNING` | NB moment-match undefined; degrades with spread (not a point-mass), never `NaN`. |
| Target column missing / bad index / missing `"test"` | `KeyError`/`IndexError` (crash) | No boundary validation (consistent with the other baselines). |
| No entities have params | `ValueError` (crash) | `require_entities`, fail-loud. |
| `views_frames` not installed | `ImportError` at predict time | Lazy import in the seam. |

---

## Boundaries and Interactions

- **Depends on:** `numpy`; `views_baseline.model.distributions` (strategy registry: `fit_family`/`sample_family`, `NATIVE_ZERO_FAMILIES`, `TRANSFORMS`, `clamp_log`, `validate_family_transform`); `views_baseline.model.pooling.window_pool`; `views_baseline.model.helpers.sample_prediction_grid`.
- **Predict scaffold (ADR-011/ADR-020):** `predict()` delegates the shared distributional shell to `helpers.sample_prediction_grid(..., draw_cell)` (the entity→time→target fill, seeded RNG, entity drop/`require_entities`, and `to_prediction_frames` construction), supplying only a per-cell `draw_cell` closure. No inline `PredictionFrame` construction; pandas confined to `window_pool`.
- **Instantiated by:** `BaselineModelCatalog._get_parametric_conflictology()`.
- **Genome:** `ReproducibilityGate.Config.ALGORITHM_GENOMES["ParametricConflictology"] = ["window_months", "n_samples", "seed", "family", "transform"]`.
- **Pool equivalence:** `self.pools` is built by the same `window_pool` as `ConflictologyModel`, so the closeness measurement (S3/S8 harness) is valid.

---

## Examples of Correct Usage

```python
from views_baseline.model.baseline import ParametricConflictology

model = ParametricConflictology(
    targets=["y1"], window_months=36, partition_dict={"test": (457, 504)},
    loa="pgm", n_samples=1000, family="nb", transform="none", seed=42,
)
model.fit(df)
result = model.predict(df=df, sequence_number=0, output_length=1)
pf = result["y1"]
assert pf.values.shape[1] == 1000
assert (pf.values >= 0).all()   # nb emits non-negative counts
```

---

## Examples of Incorrect Usage

```python
# Continuous family on the no-hurdle model — fails loud (use ParametricHurdleConflictology)
ParametricConflictology(targets=["y1"], window_months=36, partition_dict=p,
                        loa="pgm", n_samples=64, family="gamma", transform="none")
# ValueError: ... supports native-zero families ['nb'] ...

# log1p on a count family — fails loud (ADR-021)
ParametricConflictology(targets=["y1"], window_months=36, partition_dict=p,
                        loa="pgm", n_samples=64, family="nb", transform="log1p")
# ValueError: transform='log1p' is invalid for count family 'nb' ...

# Relying on a default seed in production — the catalog requires it (ADR-021)
# config without "seed" -> BaselineModelCatalog.get_model raises before construction.
```

---

## Test Alignment

Files: `tests/test_parametric.py`, `tests/test_distributions.py`, `tests/test_golden.py`, `tests/test_catalog.py`, `tests/test_reproducibility_gate.py`, `tests/test_protocol.py`.

| Test area | What it verifies (Green/Beige/Red) |
|---|---|
| Shape/type, non-negativity, reproducibility (`test_parametric.py`) | Green — `(N, n_samples)` PF, `nb` draws ≥ 0, identical output under equal `seed`. |
| **Golden / characterization** (`test_golden.py`) | Green — pins the *exact* `y_pred` (nb + zinb paths) on a fixed seed+window; a draw-path regression fails loudly (C-29). |
| `zinb` mean + zero-rate, structural inflation, NB fallback (`test_distributions.py`, `test_parametric.py`) | Green/Red — zero-inflation matches empirical zero-rate; falls back to NB with no excess zeros; `zinb+log1p` illegal. |
| All-zero entity → point-mass zero | Green — degenerate window matches conflictology. |
| `family` / `transform` validation (`nb+log1p`, continuous rejected) | Red — illegal genomes fail loud. |
| Registry fits/fallbacks (`test_distributions.py`) | Green/Red — param recovery, Poisson/point-mass fallbacks, `clamp_log`. |
| Catalog forwarding + illegal-combo (`test_catalog.py`) | Beige — `family`/`transform`/`seed` forwarded; illegal combo fails via the catalog. |
| Genome audit (`test_reproducibility_gate.py`) | Beige — `family`/`transform`/`seed` are required, audited keys. |
| Protocol conformance (`test_protocol.py`) | Beige — satisfies `DistributionalBaselineModel`. |

---

## Evolution Notes

- Family set is a registry entry: adding a native-zero family is a `distributions.py` addition plus a `NATIVE_ZERO_FAMILIES` update — no change to this class. Tweedie was evaluated (S9) and excluded (feasibility, ADR-022); it is a candidate for a future data-rich (pooled/covariate) variant.
- MLE fits are a possible later ablation; the MVP uses method-of-moments (closed-form, transparent).
- If the `views_frames` schema evolves upstream, the output contract may change — hence the **Evolving** stability (ADR-004).
