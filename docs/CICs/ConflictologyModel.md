# Class Intent Contract: ConflictologyModel

**Date:** 2026-03-17
**Owner:** Project maintainers
**Status:** Active
**Related ADRs:** ADR-001, ADR-003, ADR-005, ADR-006, ADR-008, ADR-009, ADR-011, ADR-019, ADR-020

---

## Purpose

`ConflictologyModel` is a distributional climatology baseline. For each entity, it treats the last `window_months` of training observations as an empirical distribution and produces `n_samples` i.i.d. bootstrap resamples (with replacement) per prediction cell. This gives a full predictive distribution that reflects the historical variability of conflict intensity for each location, without pooling across entities or weighting by recency. It serves as the distributional counterpart to `AverageModel`.

The class attribute `distributional = True` is a semantic marker (it satisfies the `DistributionalBaselineModel` protocol per ADR-012), **not** an `isinstance` dispatch discriminator: `BaselineForecastingModelManager` has a single type-uniform path since ADR-017, because every model returns `dict[str, PredictionFrame]`.

---

## Non-Goals

- Does not mix pools across entities. Each entity's samples come exclusively from its own local history.
- Does not weight observations by recency within the window.
- Does not produce a point prediction. The output is a `dict[str, PredictionFrame]`, not a `pd.DataFrame`.
- Does not validate `window_months` or `n_samples` at construction time.

---

## Responsibilities and Guarantees

- `fit(df)` normalizes the input via `to_feature_frame(df, loa, targets)` (the ADR-019 pandas boundary; `pd.DataFrame` or `FeatureFrame`), records `time_idx`/`entity_idx` from `ff.index.level.index_names`, and extracts the last `window_months` rows per entity up to `train_end` via `window_pool(ff, targets, window_months, train_end)` on the frame's numpy panel — a stable `(entity, time)` `np.lexsort`, no pandas. Stores the result as `self.hist_per_entity`: a `dict[entity_id, dict[target, np.ndarray]]`, plus `self.entity_ids` (entities present at `train_end`, ascending). Returns `self`.
- `predict(df, sequence_number, output_length)` returns `dict[str, PredictionFrame]` — one `PredictionFrame` per target. Each PF has:
  - `y_pred` of shape `(N, n_samples)` where `N = len(entities_with_history) * output_length`.
  - `identifiers` dict with `"time"` and `"unit"` arrays of length `N`.
- Samples are drawn using `np.random.default_rng(self.seed)`. The RNG is re-initialised fresh on each `predict()` call, making results deterministic given the same `seed` and `sequence_number`. RNG state is advanced in entity→time→target order to ensure consistent ordering.
- Entities with no history (absent from `hist_per_entity`) are dropped with a `WARNING` log.
- If no entities have history, raises a descriptive `ValueError` via `require_entities` (fail-loud — consistent with all baseline models).

---

## Inputs and Assumptions

| Parameter | Type | Notes |
|---|---|---|
| `targets` | `List[str]` | Column names in `df`. Must exist at fit time. |
| `window_months` | `int` | Number of trailing training months per entity to retain. Must be `>= 1` — a non-positive value fails loud at fit (`window_pool`). |
| `partition_dict` | `dict` | Must contain `"test"` key with tuple `(test_start, test_end)`. |
| `loa` | `str` | Declared level of analysis (`"cm"` or `"pgm"`). Now **used**, not just stored: `to_feature_frame` resolves it to a `SpatialLevel`, validates the input's spatial level against it, and carries it into every output `PredictionFrame` (ADR-003/ADR-020). |
| `n_samples` | `int` | Number of bootstrap draws per prediction cell. Required — no default. |
| `seed` | `int` | Base seed for `np.random.default_rng`. **Required, audited genome key** (ADR-021 / C-10): the catalog forwards `config["seed"]` and it must be declared in config. `DEFAULT_SEED` (42) is a single-sourced sentinel for direct/test construction only, never the production path. |
| `df` (fit/predict) | `pd.DataFrame \| FeatureFrame` | Normalized at entry via `to_feature_frame`. A DataFrame needs a 2-level MultiIndex (level 0 = time, level 1 = entity) matching `loa`; a `FeatureFrame` must carry the matching `level` and the `targets` features. |
| `sequence_number` | `int` | Offset from `test_start` for the prediction window start. |
| `output_length` | `int` | Number of forecast timesteps. Required — no default; passed from `config["time_steps"]` by the manager. |

Note: `fit()` uses `time_idx <= train_end` (inclusive) for the training filter, whereas the point models use `< test_start`. These are equivalent since `train_end = test_start - 1`, but the inclusive bound is explicit here.

---

## Outputs and Side Effects

- **`fit()`**: Returns `self`. Sets `self.time_idx`, `self.entity_idx`, `self.hist_per_entity`, `self.entity_ids`. No external side effects.
- **`predict()`**: Returns `dict[str, PredictionFrame]`. Raises `ValueError` (via `require_entities`) if no entities have history. Emits `WARNING` on entity drops. No external side effects. `PredictionFrame`/`SpatioTemporalIndex` are lazy-imported from `views_frames` inside the `to_prediction_frames` seam at call time.

---

## Failure Modes and Loudness

| Failure | Loudness | Notes |
|---|---|---|
| Target column missing / feature absent from a `FeatureFrame` | `ValueError` (crash) | `to_feature_frame` checks target presence at the input boundary (ADR-019). |
| DataFrame index names do not match `loa` (incl. non-2-level index) | `ValueError` (crash) | `to_feature_frame` → `resolve_level` raises at entry. |
| `partition_dict` missing `"test"` | `KeyError` (crash) | No validation. |
| `window_months <= 0` | `ValueError` at **fit** (crash) | `window_pool` fails loud with `"window_months must be >= 1, got ..."`. Previously this was an accidental empty-group `KeyError` — now caught deliberately (ADR-019). |
| `n_samples <= 0` | `ValueError` from numpy | `rng.choice(..., size=0)` succeeds but `size < 0` raises during `predict()`. |
| Entities with empty history at fit time | Silently excluded from `hist_per_entity` | `window_pool` skips entities whose windowed slice is empty. |
| Entities absent from `hist_per_entity` at predict time | `WARNING` log | Dropped from output. |
| No entities have history | `ValueError` (crash) | `require_entities` raises a descriptive error. Fail-loud — consistent with all baseline models; an empty result would otherwise surface as a `StopIteration` deep in pipeline-core's evaluation path. |
| `views_frames` not installed | `ImportError` (crash at predict time) | Lazy import in the `to_prediction_frames` seam defers this to `predict()`. |

---

## Boundaries and Interactions

- **Depends on:** `numpy` (`np.random.default_rng`, `rng.choice`); `views_baseline.model.frames.input` (`to_feature_frame` — the single pandas boundary, ADR-019); `views_baseline.model.frames.pooling.window_pool` (numpy windowing); `views_baseline.model.frames.output.sample_prediction_grid`; `views_baseline.model.spatial` (via the resolved `ff.index.level`). pandas is not a runtime dependency — confined to `to_feature_frame`, imported only under `TYPE_CHECKING`.
- **Output construction (ADR-020):** routes through the single seam `to_prediction_frames` in `model/frames/output.py`, which lazy-imports the `views_frames` leaf (`PredictionFrame`, `SpatioTemporalIndex`) inside the function — not at module load. No inline construction; `build_prediction_grid` is deleted.
- **Predict scaffold (ADR-011):** `predict()` delegates the shared distributional shell (entity→time→target fill, seeded RNG, entity drop/`require_entities`, seam construction) to `frames.output.sample_prediction_grid(..., draw_cell)`, supplying only the per-cell resample as `draw_cell` (`rng.choice(hist_per_entity[cid][t], size=n_samples, replace=True)`). The spatial `level` is resolved once at the input boundary and passed in as `level=ff.index.level` (since PR-2 `sample_prediction_grid`'s signature takes `level`, not `(loa, index_names)`).
- **Instantiated by:** `BaselineModelCatalog._get_conflictology_model()`.
- **Dispatched by:** `BaselineForecastingModelManager._generate_predictions()` — a single type-uniform path since ADR-017; `distributional = True` is a semantic marker, not an `isinstance` dispatch discriminator.
- **Pool equivalence:** `hist_per_entity` is constructed identically to `MixtureBaseline.local_pool`. This equivalence is verified by `test_conflictology_matches_mixture_lambda_zero`.

---

## Examples of Correct Usage

```python
from views_baseline.model.models.distributional import ConflictologyModel

partition_dict = {"test": (493, 528)}
model = ConflictologyModel(
    targets=["y1", "y2"],
    window_months=4,
    partition_dict=partition_dict,
    loa="pgm",
    n_samples=64,
    seed=42,
)
model.fit(df)
result = model.predict(df=df, sequence_number=0, output_length=5)

# result is dict[str, PredictionFrame]
pf = result["y1"]
assert pf.y_pred.shape == (n_entities * 5, 64)

# All samples come from local history
for i in range(pf.y_pred.shape[0]):
    uid = pf.identifiers["unit"][i]
    history_vals = set(model.hist_per_entity[uid]["y1"].tolist())
    assert set(pf.y_pred[i].tolist()).issubset(history_vals)
```

---

## Examples of Incorrect Usage

```python
# window_months=0 now fails loud at fit (previously an accidental empty-group KeyError)
model = ConflictologyModel(targets=["y1"], window_months=0, partition_dict=partition_dict, loa="pgm")
model.fit(df)   # ValueError: window_months must be >= 1, got 0.

# Expecting a DataFrame instead of dict
result = model.predict(df=df, sequence_number=0, output_length=5)
result.values   # AttributeError: dict has no .values

# Assuming RNG state carries over between predict calls
r1 = model.predict(df=df, sequence_number=0, output_length=5)
r2 = model.predict(df=df, sequence_number=0, output_length=5)
# r1 and r2 ARE identical (RNG re-seeded each call) — but this is a feature, not a bug.
# Do NOT assume r1 != r2 for different sequence_numbers; they can coincidentally match
# if the drawn samples happen to be the same from the same history.
```

---

## Test Alignment

File: `tests/test_baseline.py`

| Test | What it verifies |
|---|---|
| `test_conflictology_model_resamples_from_history` | Output is `dict[str, PredictionFrame]`, shape is `(n_entities * output_length, n_samples)`, and all sampled values are elements of the entity's local history window. |
| `test_conflictology_model_respects_sequence_number` | Time identifiers in the PredictionFrame span `[test_start + seq_num, test_start + seq_num + output_length - 1]`. |
| `test_conflictology_matches_mixture_lambda_zero` | `hist_per_entity` pools are element-wise equal (after sorting) to `MixtureBaseline.local_pool` when built on the same data, confirming pool construction parity. |

---

## Evolution Notes

- If temporal weighting within the window is required, a new class (e.g., `WeightedClimatologyModel`) is preferable to adding optional parameters here. This preserves the "pure i.i.d. resample from flat history" semantics.
- The `views_frames` import lives in the single `to_prediction_frames` seam (ADR-020); `model/` stays importable without the frame library, the dependency incurred only when `predict()` runs.
- Adding `sequence_number` as an input to the RNG seed (i.e., `default_rng(seed + sequence_number)`) would make samples vary across sequence steps. This is a design choice currently left to the caller to work around via different seeds.

---

## Known Deviations

- `window_months <= 0` now fails loud with a descriptive `ValueError` at **fit** (via `window_pool`), no longer an accidental empty-group `KeyError` at predict. `n_samples > 0` is still unvalidated and surfaces as a numpy error during `predict()`.
- `PredictionFrame`/`SpatioTemporalIndex` are imported lazily inside the `to_prediction_frames` seam at predict time, not at module load or construction. Import failures (e.g., `views_frames` not installed) are deferred to prediction time.
- `FeatureFrame` storage is `float32` by design (C-32), so the resampled window values are `float32`-rounded relative to raw `float64`. The entity order, per-entity tail order, and RNG-advance order are preserved exactly (the byte-identity contract, ADR-011), and a DataFrame and the `FeatureFrame` built from it produce identical output — an accepted precision trade-off. This also preserves the byte-identical pool equivalence with `MixtureBaseline.local_pool`.
- `fit()` windows on `time <= train_end` (inclusive upper bound on training data) while the point models frame it as `time < test_start`. These are semantically equivalent (`train_end = test_start - 1`) but the phrasing difference makes code comparison slightly harder.
