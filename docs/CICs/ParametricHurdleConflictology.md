# Class Intent Contract: ParametricHurdleConflictology

**Date:** 2026-07-18
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-001, ADR-003, ADR-005, ADR-006, ADR-008, ADR-009, ADR-011, ADR-020, ADR-021, ADR-022

---

## Purpose

`ParametricHurdleConflictology` is the **hurdle** parametric baseline. Per entity/target it fits a two-part law to the entity's `window_pool`: a **zero-spike** (`w` = the window's empirical zero-rate, a Bernoulli gate) plus a **continuous positive-part family** (`lognormal`/`gumbel`/`gamma`) fit to the *positive* window values. It draws `n_samples` per cell — each draw is either an exact 0 (probability `w`) or a sample from the positive part. This mirrors Vesco et al. 2026's RVI mixture (point mass at a distinguished value + continuous part), with the distinguished value being 0 (zero-inflation).

The S8 closeness experiment found `gamma`/`none` the closest on **magnitude fidelity** (Wasserstein/energy) at high activity, with `gumbel`/`none` (the Vesco-motivated, right-skew, lighter-tailed family) a strong second; `log1p` is strictly worse on active cells and should not be a default.

---

## Non-Goals

- Does **not** pool across entities; each fit uses only that entity's own window (shared `window_pool`).
- Does **not** accept a native-zero family (`nb`) — those are no-hurdle-only; passing `nb` **fails loud** at construction.
- Does **not** emit negative magnitudes: the positive part is floored at 0 (`clamp_floor`), even for `gumbel` (whose ℝ support otherwise permits negative draws). See Guarantees.
- Does **not** apply magic defaults to `family`, `transform`, or `seed` — all are required, audited genome keys.
- Does **not** detransform on a summary mean — detransform is **per sample** (Jensen-safe).

---

## Responsibilities and Guarantees

- `__init__` validates `family ∈ CONTINUOUS_FAMILIES` (`{"lognormal","gumbel","gamma"}`; else `ValueError`) and `validate_family_transform(family, transform)` (`log1p` is legal here, `none` too).
- `fit(df)` normalizes the input via `to_feature_frame(df, loa, targets)` (the ADR-019 pandas boundary; `pd.DataFrame` or `FeatureFrame`), records `time_idx`/`entity_idx` from `ff.index.level.index_names`, and extracts per-entity/per-target windows via the shared `window_pool(ff, ...)` on the frame's numpy panel (no pandas). For each cell it stores `{"zero_rate": mean(pool == 0), "pos": fit_family(family, forward(positives))}`; an all-zero window stores `{"zero_rate": 1.0, "pos": None}` (point mass at 0). Returns `self`.
- `predict(df, sequence_number, output_length)` returns `dict[str, PredictionFrame]`, `y_pred` shape `(N, n_samples)`, `N = len(entities_with_params) * output_length`.
- Sampling uses a **fresh** `np.random.default_rng(self.seed)` per call, advancing in **entity→time→target** order (ADR-011). Per cell: `is_positive = rng.random(n_samples) >= zero_rate`; the positive draws come from `sample_family`, then (if `transform != "none"`) are detransformed **per sample** via `inverse(clamp_log(...))`, then **floored** via `clamp_floor` before being written into the output; the rest stay 0.
- **Non-negativity is guaranteed** by `clamp_floor` (single-sourced `EMIT_FLOOR = 0.0`, WARN on floor) — the mirror image of the `EMIT_LOG_CEIL` ceiling. This closes the `gumbel` negative-emission path (register C-26).
- Degenerate positive parts fail **safe and loud-ish** in `distributions/families.py` (single positive value → point-mass; etc.), never `NaN`.
- Entities without params are dropped with a `WARNING`; none remaining → `require_entities` raises (fail-loud). Output is built only through `to_prediction_frames` (ADR-020); the spatial `level` is resolved and validated once at the `to_feature_frame` boundary (`resolve_level` for a DataFrame, or a `level`↔`loa` check for a `FeatureFrame`) and passed into `sample_prediction_grid` as `level=ff.index.level`.

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets` | `list[str]` | Column names in `df`; must exist at fit time. |
| `window_months` | `int` | Trailing training months per entity (shared `window_pool`). |
| `partition_dict` | `dict` | Must contain `"test"` → `(test_start, test_end)`; `train_end = test_start − 1`. |
| `loa` | `str` | Declared level; resolved + validated via `resolve_level`. |
| `n_samples` | `int` | Draws per cell. Required — no default. |
| `family` | `str` | Required, audited genome key. Must be in `CONTINUOUS_FAMILIES`. |
| `transform` | `str` | Required, audited genome key. `"none"` or `"log1p"` (both legal for continuous families). |
| `seed` | `int` | Required, audited genome key (ADR-021). `DEFAULT_SEED` (42) is a direct-construction sentinel only. |
| `df` (fit/predict) | `pd.DataFrame \| FeatureFrame` | Normalized at entry via `to_feature_frame`. A DataFrame needs a 2-level MultiIndex (level 0 = time, level 1 = entity) matching `loa`; a `FeatureFrame` must carry the matching `level` and the `targets` features. pandas confined to the `to_feature_frame` boundary; `window_pool` runs on the numpy panel. |
| `sequence_number` / `output_length` | `int` | Prediction-window offset / number of timesteps. Required. |

---

## Outputs and Side Effects

- **`fit()`**: returns `self`; sets `self.time_idx`, `self.entity_idx`, `self.entity_ids`, `self.pools`, `self.params` (per cell `{"zero_rate", "pos"}`). No external side effects.
- **`predict()`**: returns `dict[str, PredictionFrame]`; raises `ValueError` (via `require_entities`) if no entities have params; emits `WARNING` on entity drops, `clamp_log` clamping, or `clamp_floor` flooring. Stochastic but reproducible given `seed`. `views_frames` imported lazily in the seam.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| Native-zero `family` (e.g. `"nb"`) | `ValueError` at `__init__` | Fail-loud; native-zero families are no-hurdle-only. |
| Unknown `family` / illegal `transform` | `ValueError` | `validate_family_transform` / `fit_family`. |
| `gumbel` left-tail / `log1p` `expm1∈(-1,0)` draw | Floored to 0 + `WARNING` | `clamp_floor` guarantees non-negativity (C-26). |
| Heavy-tail log-space draw | Capped at `EMIT_LOG_CEIL` + `WARNING` | `clamp_log` before `expm1` (overflow guard). |
| All-zero window | `zero_rate = 1.0`, `pos = None` | Point mass at 0; `is_positive` always false. |
| `window_months <= 0` | `ValueError` at **fit** (crash) | `window_pool` fails loud with `"window_months must be >= 1, got ..."` (ADR-019). |
| Target column missing / DataFrame index names not matching `loa` / `FeatureFrame` level or target mismatch | `ValueError` (crash) | `to_feature_frame` validates targets and level at the input boundary (ADR-019). |
| `partition_dict` missing `"test"` | `KeyError` (crash) | No validation (consistent with the other baselines). |
| No entities have params | `ValueError` (crash) | `require_entities`, fail-loud. |

---

## Boundaries and Interactions

- **Depends on:** `numpy`; `views_baseline.model.distributions` (`fit_family`/`sample_family`, `CONTINUOUS_FAMILIES`, `TRANSFORMS`, `clamp_log`, `clamp_floor`, `validate_family_transform`); `frames.input` (`to_feature_frame` — the single pandas boundary, ADR-019); `frames.pooling.window_pool` (numpy windowing); `spatial` (`resolve_level`, reached via `to_feature_frame`); `grid` (`build_time_grid`, `filter_entities`, `require_entities`, `build_identifier_arrays`); `frames.output` (`sample_prediction_grid` / `to_prediction_frames`). pandas is not a runtime dependency — confined to `to_feature_frame`, imported only under `TYPE_CHECKING`.
- **Output construction (ADR-020):** single `to_prediction_frames` seam; pandas confined to the `to_feature_frame` boundary.
- **Predict scaffold (ADR-011):** `predict()` delegates the shared distributional shell (entity→time→target fill, seeded RNG, entity drop/`require_entities`, seam construction) to `frames.output.sample_prediction_grid(..., draw_cell)`, supplying only the per-cell hurdle draw (Bernoulli zero-gate + positive-part `sample_family` → per-sample detransform → `clamp_floor`) as `draw_cell`, plus `level=ff.index.level` (PR-2 changed `sample_prediction_grid`'s signature to take `level`, not `(loa, index_names)`).
- **Precision (C-32):** `views_frames.FeatureFrame` storage is `float32` by design, so the pooled window magnitudes (and thus the fitted zero-rate and positive-part params) are `float32`-rounded relative to raw `float64` (accepted trade-off). The entity/tail/RNG-advance **order** is preserved exactly (ADR-011), and a DataFrame and the `FeatureFrame` built from it produce identical output — so reproducibility holds bit-for-bit for both input types.
- **Instantiated by:** `BaselineModelCatalog._get_parametric_hurdle()`.
- **Genome:** `ALGORITHM_GENOMES["ParametricHurdleConflictology"] = ["window_months", "n_samples", "seed", "family", "transform"]`.

---

## Examples of Correct Usage

```python
from views_baseline.model.models.distributional import ParametricHurdleConflictology

model = ParametricHurdleConflictology(
    targets=["y1"], window_months=36, partition_dict={"test": (457, 504)},
    loa="pgm", n_samples=1000, family="gamma", transform="none", seed=42,
)
model.fit(df)                # model.fit(ff) also works — dual input (pd.DataFrame | FeatureFrame)
pf = model.predict(df=df, sequence_number=0, output_length=1)["y1"]
assert (pf.values >= 0).all()                # clamp_floor guarantee
# sampled zero fraction ≈ the window's empirical zero-rate per cell

# Vesco path: gumbel positive part with log1p — still non-negative (clamp_floor closes gumbel's ℝ tail)
ParametricHurdleConflictology(targets=["y1"], window_months=36, partition_dict={"test": (457, 504)},
                              loa="pgm", n_samples=1000, family="gumbel", transform="log1p", seed=42)
```

---

## Examples of Incorrect Usage

```python
# Native-zero family on the hurdle model — fails loud (use ParametricConflictology)
ParametricHurdleConflictology(targets=["y1"], window_months=36, partition_dict=p,
                              loa="pgm", n_samples=64, family="nb", transform="none")
# ValueError: ... supports continuous positive-part families ['gamma','gumbel','lognormal'] ...

# Defaulting to log1p because "counts are skewed" — S8 shows log1p is strictly worse on active
# cells (per-sample expm1 inflates the heavy right tail). Prefer transform="none".
```

---

## Test Alignment

Files: `tests/test_parametric.py`, `tests/test_distributions.py`, `tests/test_catalog.py`, `tests/test_reproducibility_gate.py`, `tests/test_protocol.py`.

| Test area | What it verifies (Green/Beige/Red) |
|---|---|
| Shape/type, reproducibility (`test_parametric.py`) | Green — `(N, n_samples)` PF, identical output under equal `seed`. |
| Sampled zero-rate ≈ empirical zero-rate | Green — the Bernoulli gate matches the window. |
| `log1p` round-trip → raw non-negative scale | Green — per-sample `expm1`, `clamp_floor`. |
| `gumbel` left tail floored to non-negative (deterministic) | Red — heavy-outlier window forces `loc<0`; `clamp_floor` engages. |
| `nb` rejected by the hurdle | Red — native-zero family fails loud. |
| `clamp_floor` unit behaviour (`test_distributions.py`) | Green/Red — floors negatives, WARNs, leaves non-negatives untouched. |
| Catalog forwarding + genome audit + protocol | Beige — `family`/`transform`/`seed` forwarded & audited; satisfies `DistributionalBaselineModel`. |

---

## Evolution Notes

- Positive-part family set is a registry entry (`CONTINUOUS_FAMILIES`); adding one is a `distributions/families.py` change, not a change to this class.
- A zero-truncated `nb`/`tweedie` positive part is a possible future extension (ADR-022 open question).
- Non-negativity (`EMIT_FLOOR`) and overflow (`EMIT_LOG_CEIL`) are single-sourced policy constants; changing them is a deliberate, documented decision.
- **Evolving** stability (ADR-004): the `views_frames` schema is upstream-owned.
