# views-baseline

Baseline forecasting models for the VIEWS pipeline.

This package provides **simple, transparent baseline models** that can be used for benchmarking more complex forecasting models and sanity checks. The baselines are intentionally minimal and deterministic (or reproducibly stochastic for distributional models).

---

## Overview

The package implements several common baseline strategies for **panel time-series data** with a multi-index of:

* **time** (e.g. `month_id`)
* **entity** (e.g. `priogrid_id`, `country_id`)

All models follow a common interface:

```python
model.fit(df)                       # df: a pandas DataFrame OR a views_frames FeatureFrame
predictions = model.predict(df, sequence_number)
```

`fit()` / `predict()` accept **either** a pandas `DataFrame` **or** a `views_frames.FeatureFrame` (ADR-019): the input is normalized to a `FeatureFrame` once at the boundary (`to_feature_frame` in `model/frames/input.py`, the only place pandas is read), and all windowing/aggregation runs on its numpy arrays. Models are fitted automatically via `fit()` before generating predictions. Fitted model artifacts are pickled for ensemble compatibility.

---

## Implemented Models

All baselines operate on panel data indexed by **(time `t`, unit `i`)** — `month_id` × spatial unit (`priogrid_id` at `pgm`, `country_id` at `cm`) — and forecast one or more target columns over a horizon of `output_length` steps starting at `test_start + sequence_number`. ("unit" is the term used in a `PredictionFrame`'s `SpatioTemporalIndex`; the input index calls the same axis the *entity*.)

**Causal split (no leakage).** Every model is fit using only observations strictly before the test period. The last training month is `train_end = test_start − 1`; no value at or after `test_start` enters any fitted quantity. (The point models and `MixtureBaseline` filter `t < test_start`; `ConflictologyModel` filters `t ≤ train_end` — the same boundary, written two ways.)

Two output families:

- **Point forecasts** — one deterministic value per (unit, time, target). Returned as `dict[str, PredictionFrame]` with `y_pred` of shape `(N, 1)`.
- **Distributional forecasts** — `n_samples` Monte-Carlo draws per (unit, time, target). Returned as `dict[str, PredictionFrame]` with `y_pred` of shape `(N, n_samples)`.

In all cases `N = (number of units) × output_length`, and each `PredictionFrame` carries a `SpatioTemporalIndex` with the `time`, `unit`, and spatial `level` (CM/PGM) of every row. All frames are built through a single construction seam, `to_prediction_frames` in `model/frames/output.py` (ADR-020), and the `level` is derived from the declared `loa` and validated against the input index (ADR-003).

### Point Forecast Models

#### ZeroModel

Predicts exactly **0** for every target, unit, and forecast step. The lower-bound reference; performs no fitting.

```python
ZeroModel(targets, partition_dict, loa)
```

#### LocfModel — Last Observation Carried Forward

For each unit and target, carries the **last observed value at `train_end`** forward unchanged across the entire horizon. A persistence baseline ("the most recent observation is the best guess"), strong for highly autocorrelated targets.

```python
LocfModel(targets, partition_dict, loa)
```

#### AverageModel

For each unit and target, forecasts the **arithmetic mean of that unit's last `window_months` observations** before `test_start`, held constant across the horizon. A smoothed-persistence baseline, more robust than LOCF when individual months are noisy.

```python
AverageModel(targets, window_months, partition_dict, loa)
```

* Means are computed per unit; units with no history before `test_start` are skipped (logged).

### Distributional Models

Both draw `n_samples` i.i.d. samples per cell from a fresh, seeded generator (`numpy.random.default_rng(seed)`), so a given (data, configuration, seed) reproduces bit-for-bit. The RNG is consumed in a fixed unit → time → target order to guarantee reproducibility.

#### ConflictologyModel — empirical climatology

For each unit `i`, collects that unit's **last `window_months` observed values** up to `train_end`, then draws `n_samples` samples **with replacement** from that per-unit history for every forecast cell. The predictive distribution for a cell is the recent empirical distribution of that same unit — a conflict "climatology." It uses only the unit's own recent history: no pooling across units, no older history.

**The window is fixed at `train_end` for every forecast origin, on purpose.** It mirrors the production regime, where models are trained once a year and the saved artifacts are used for inference for the rest of the year. Note that this is stricter than what the learned models get: they freeze their *parameters* but still read the latest observed data at each origin, whereas for a climatology the data *is* the model, so freezing it also freezes its inputs. `RollingConflictology` below is the version whose information matches the learned models' (#82).

```python
ConflictologyModel(targets, window_months, partition_dict, loa, n_samples, seed=42)
```

#### MixtureBaseline — mixture of local and global empirical pools

Combines two empirical sources to avoid the **zero-probability trap** (a unit whose recent history is entirely zero being structurally unable to predict a nonzero outcome):

- **Local pool** — the unit's last `window_months` observed values (as in `ConflictologyModel`).
- **Global pool** — **all strictly-positive** observed values, pooled across **every unit** and the **entire training span**.

Each of the `n_samples` draws is taken from the **global** pool with probability `lambda_mix`, otherwise from the **local** pool (probability `1 − lambda_mix`). At `lambda_mix = 0` it reduces to a local-only empirical baseline (same source as `ConflictologyModel`); larger `lambda_mix` injects more cross-unit, full-history positive mass.

```python
MixtureBaseline(targets, window_months, lambda_mix, n_samples, partition_dict, loa, seed=42)
```

#### ParametricConflictology — no-hurdle parametric climatology

The parametric counterpart of `ConflictologyModel`: instead of resampling each unit's window empirically, it **fits a single native-zero distribution** (`family`, e.g. `nb`) to that same window and draws `n_samples` per cell from the fitted law. `family`, `transform`, and `seed` are required, audited genome keys (ADR-021/ADR-022); `transform="log1p"` is illegal for count families and fails loud. In the closeness study (see `reports/closeness_experiment/`), `nb` is the family closest to conflictology on the C2ST indistinguishability metric.

```python
ParametricConflictology(targets, window_months, partition_dict, loa, n_samples, family, transform="none", seed=42)
```

#### ParametricHurdleConflictology — hurdle parametric climatology

A two-part law per unit: a **zero-spike** (empirical zero-rate, Bernoulli) plus a **continuous positive-part family** (`lognormal`/`gumbel`/`gamma`) fit to the positive window values (mirrors Vesco et al. 2026's RVI mixture). `transform` (`none`/`log1p`) applies to the positive part and is inverted **per sample**; a non-negativity floor (`EMIT_FLOOR`) guarantees no negative magnitudes even for `gumbel`. Closeness study: `gamma`/`none` is closest on magnitude fidelity (Wasserstein/energy); `log1p` is worse on active cells. (Tweedie was evaluated and excluded — ADR-022.)

```python
ParametricHurdleConflictology(targets, window_months, partition_dict, loa, n_samples, family, transform="none", seed=42)
```

#### RollingConflictology — conflictology with a moving window

`ConflictologyModel` whose window moves with each rolling forecast origin (#126; the control for ADR-024). `ConflictologyModel` draws every origin's forecast from one window ending at the last training month. `RollingConflictology` draws origin `k`'s forecast from the window ending at `train_end + k`: the month just before that origin's first forecast month, the same information rule views-stepshifter uses. At origin 0 it is identical to `ConflictologyModel`. It stops with an error if the data does not reach `train_end + k` for every cell, rather than quietly using an older window.

```python
RollingConflictology(targets, window_months, partition_dict, loa, n_samples, seed=42)
```

#### MatryoshkaConflictology — conflictology that also draws from a cell's surroundings

Conflictology that chooses *which past observations to draw from* (ADR-024). Each draw is a value that was really observed, picked in three steps:
1. a doll: the cell, its neighbourhood (within 5 cells, closer ones more likely), the rest of its country, or the rest of the map;
2. a month, recent ones more likely;
3. a cell in that doll.

Each doll has a weight and a memory. The further ahead a forecast looks, the more it draws from the dolls with the longest memories. The fitted memories decide which those are, and they are not always the outer dolls. A quiet cell is never forecast as *certainly* quiet unless its neighbourhood, its country and the world give no reason to expect conflict.

The 8 knobs per target (3 weights, 4 memories and the neighbourhood's length scale) are **fitted** by backtest on data up to `train_end`, not configured. The data moves with each rolling origin, as for `RollingConflictology`. It needs a complete grid-month panel with a country column (views-models' pgm input carries `c_id`), at least 120 months before the earliest backtest cutoff, and pgm level.

```python
MatryoshkaConflictology(targets, partition_dict, loa, n_samples, country_column, backtest_cutoffs, backtest_horizon, stability_refits, seed=42)
```

The notebook [`notebooks/matryoshka_conflictology.ipynb`](notebooks/matryoshka_conflictology.ipynb) explains what it does and how, and draws the plots when it is run (Jupyter; the synthetic part runs in seconds, the real-data part needs the calibration input pull).

#### InterpolatedQuantileConflictology — smoothed climatology B

`ConflictologyModel` with the positive part smoothed (ADR-023). The zero rate is the window's, exactly as in `ConflictologyModel`. A positive draw comes from the **linearly interpolated quantile function** of the window's positive values, rounded to a whole count. So it can draw the counts between the observed ones, but never beyond the smallest or largest. Cells with fewer than two distinct positive values are drawn as `ConflictologyModel` draws them, and counted in one warning.

```python
InterpolatedQuantileConflictology(targets, window_months, partition_dict, loa, n_samples, seed=42)
```

#### KDEConflictology — smoothed climatology C

`ConflictologyModel` with the positive part drawn from a **kernel density estimate** of the window's positive values (ADR-023). The zero rate is the window's. Each positive draw picks an observed positive value, adds Gaussian noise of width `bandwidth`, mirrors anything below 0.5 back up, and rounds to a whole count, so it never produces a zero and can go above the largest observed value. `bandwidth` is required: a number (in deaths, the same for every cell), `"sj"` (Sheather–Jones, chosen per cell) or `"silverman"` (per cell; known to oversmooth). Cells it cannot smooth are drawn as `ConflictologyModel` draws them and counted in one warning. Known bias: it moves some mass from 1 to larger counts, more for larger `bandwidth`.

```python
KDEConflictology(targets, window_months, partition_dict, loa, n_samples, bandwidth, seed=42)
```

#### EmpiricalKDEMixtureConflictology — smoothed climatology E

A mixture of `ConflictologyModel` and `KDEConflictology` (ADR-023): each draw is a `ConflictologyModel` draw with probability `1 − alpha` and a `KDEConflictology` draw with probability `alpha`. `alpha` therefore dials the spread from the empirical model (0) to the smoothed one (1). The random numbers it uses don't depend on `alpha`, so at a fixed seed, raising `alpha` only swaps individual draws from one model to the other: comparisons across `alpha` are paired and low-noise.

```python
EmpiricalKDEMixtureConflictology(targets, window_months, partition_dict, loa, n_samples, alpha, bandwidth, seed=42)
```

---

## Model Catalog

The `BaselineModelCatalog` provides a factory for instantiating models based on config:

```python
from views_baseline.model.catalog import BaselineModelCatalog

catalog = BaselineModelCatalog(config, partition_dict, loa)
model = catalog.get_model("LocfModel")
```

Available models:

```python
catalog.list_models()
# ['ZeroModel', 'LocfModel', 'AverageModel', 'ConflictologyModel', 'MixtureBaseline',
#  'ParametricConflictology', 'ParametricHurdleConflictology',
#  'InterpolatedQuantileConflictology', 'KDEConflictology',
#  'EmpiricalKDEMixtureConflictology', 'RollingConflictology', 'MatryoshkaConflictology']
```

---

## Integration with the VIEWS Pipeline

The package integrates with the core pipeline via:

```python
BaselineForecastingModelManager
```

Key characteristics:

* Models are fitted automatically via `fit()` and artifacts are pickled
* Predictions are generated per evaluation sequence
* Supports both evaluation and forecasting modes
* Distributional models are dispatched automatically via the `DistributionalBaselineModel` protocol

---

## Data Assumptions

Input DataFrame:

* Must be indexed by `(time, entity)` as a MultiIndex
* Must contain target columns specified in `config['regression_targets']` (the `targets` key was retired upstream in views-pipeline-core #380 and is now *rejected* — do not add it back)

Example index:

```text
MultiIndex(levels=[month_id, priogrid_id])
```

---

## Notes & Caveats

* Entities without sufficient history are skipped (with a warning)
* No imputation beyond what the baseline logic implies
* No clipping or post-processing is applied by default

---

## Installation

Clone the repository and install with pip:

```bash
pip install -e .
```

---

## License / Usage

Internal VIEWS package. Intended for research and forecasting pipelines, not as a general-purpose forecasting library.
