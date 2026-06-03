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

### Point Forecast Models

These models return `dict[str, PredictionFrame]` — one `PredictionFrame` per target with `y_pred` shape `(N, 1)` (single deterministic value).

#### ZeroModel

Predicts **0 for all targets**, entities, and forecast horizons.

```python
ZeroModel(targets, partition_dict, loa)
```

#### LocfModel (Last Observation Carried Forward)

Repeats the **last observed value before the test period** for each entity and target across the forecast horizon.

```python
LocfModel(targets, partition_dict, loa)
```

#### AverageModel

Forecasts the **mean of the last `window_months` months** before the test period for each entity and target.

```python
AverageModel(targets, window_months, partition_dict, loa)
```

* Averages are computed per entity
* Entities without sufficient history are skipped

### Distributional Models

These models return `dict[str, PredictionFrame]` — one `PredictionFrame` per target, each containing `n_samples` draws per cell.

#### ConflictologyModel

Climatology baseline that **resamples with replacement** from the last `window_months` of data for each entity, producing `n_samples` i.i.d. draws per cell.

```python
ConflictologyModel(targets, window_months, partition_dict, loa, n_samples=256, seed=42)
```

#### MixtureBaseline

Mixture empirical baseline that **combines local history with a global positive pool** to avoid the zero-probability trap. Each sample is drawn from the local pool with probability `1 - lambda_mix` or the global positive pool with probability `lambda_mix`.

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
