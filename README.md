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
model.fit(df)
predictions = model.predict(df, sequence_number)
```

Models are fitted automatically via `fit()` before generating predictions. Fitted model artifacts are pickled for ensemble compatibility.

---

## Implemented Models

All baselines operate on panel data indexed by **(time `t`, unit `i`)** — `month_id` × spatial unit (`priogrid_id` at `pgm`, `country_id` at `cm`) — and forecast one or more target columns over a horizon of `output_length` steps starting at `test_start + sequence_number`. ("unit" is the term used in a `PredictionFrame`'s `SpatioTemporalIndex`; the input index calls the same axis the *entity*.)

**Causal split (no leakage).** Every model is fit using only observations strictly before the test period. The last training month is `train_end = test_start − 1`; no value at or after `test_start` enters any fitted quantity. (The point models and `MixtureBaseline` filter `t < test_start`; `ConflictologyModel` filters `t ≤ train_end` — the same boundary, written two ways.)

Two output families:

- **Point forecasts** — one deterministic value per (unit, time, target). Returned as `dict[str, PredictionFrame]` with `y_pred` of shape `(N, 1)`.
- **Distributional forecasts** — `n_samples` Monte-Carlo draws per (unit, time, target). Returned as `dict[str, PredictionFrame]` with `y_pred` of shape `(N, n_samples)`.

In all cases `N = (number of units) × output_length`, and each `PredictionFrame` carries a `SpatioTemporalIndex` with the `time`, `unit`, and spatial `level` (CM/PGM) of every row. All frames are built through a single construction seam, `to_prediction_frames` in `model/helpers.py` (ADR-020), and the `level` is derived from the declared `loa` and validated against the input index (ADR-003).

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
# ['ZeroModel', 'LocfModel', 'AverageModel', 'ConflictologyModel', 'MixtureBaseline']
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
* Must contain target columns specified in `config['targets']`

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
