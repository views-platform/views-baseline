# views-baseline

**views-baseline** is a package providing simple, interpretable baseline models for the VIEWS forecasting pipeline. These models are designed for benchmarking and sanity-checking more complex machine learning models. Baseline models in this package do not require training and include approaches such as always predicting zero or repeating the last observed value.

## Features

- **ZeroModel**: predicts zero for all targets and all forecast horizons.
- **LOCFModel**: repeats the last observed value for each target into the future.
- **AverageModel**: predicts the average of the last n months (default 18 months)
- **Plug-and-play**: Fully compatible with the VIEWS pipeline and model manager interfaces.
- **No training required**: Baseline models are stateless and require no fitting.

The current set up does not support ensembling as no model artifact is saved. 

## Installation

Clone the repository and install with pip:

```bash
pip install -e .
```

# views-baseline

Baseline forecasting models for the VIEWS pipeline.

This package provides **simple, transparent baseline models** that can be used for:

* benchmarking more complex forecasting models
* sanity checks

The baselines are intentionally minimal and deterministic.

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

Predictions are returned as a `pd.DataFrame` indexed by `(time, entity)` with columns named:

```
pred_<target>
```

---

## Implemented Models

### 1. ZeroModel

Predicts **0 for all targets**, entities, and forecast horizons.

```python
ZeroModel(targets, partition_dict, loa)
```

---

### 2. LocfModel (Last Observation Carried Forward)

Repeats the **last observed value before the test period** for each entity and target across the forecast horizon.


```python
LocfModel(targets, partition_dict, loa)
```

---

### 3. AverageModel

Forecasts the **mean of the last `m` months** before the test period for each entity and target.

The window length `m` is configurable.

```python
AverageModel(targets, months, partition_dict, loa)
```

**Notes**:

* averages are computed per entity
* entities without sufficient history are skipped

---

### 4. ConflictologyModel

A special baseline used in the VIEWS context.

Instead of producing point forecasts, this model returns uncertainty forecasts, e.g. forecasting samples (last `m` months) as the prediction for each future time step.

```python
ConflictologyModel(targets, months, partition_dict, loa)
```

Each `pred_<target>` column contains a **list of length `m`**.

---

## Model Catalog

The `BaselineModelCatalog` provides a simple factory for instantiating models based on config:

```python
from views_baseline.model.catalog import BaselineModelCatalog

catalog = BaselineModelCatalog(config, partition_dict, loa)
model = catalog.get_model("LocfModel")
```

Available models:

```python
catalog.list_models()
# ['ZeroModel', 'LocfModel', 'AverageModel', 'ConflictologyModel']
```

---

## Integration with the VIEWS Pipeline

The package integrates with the core pipeline via:

```python
BaselineForecastingModelManager
```

Key characteristics:

* **no training step** (training is skipped)
* predictions are generated per evaluation sequence
* supports both evaluation and forecasting modes

---

## Data Assumptions

Input DataFrame:

* must be indexed by `(time, entity)`
* must contain target columns specified in `config['targets']`

Example index:

```text
MultiIndex(levels=[month_id, priogrid_id])
```

---

## Notes & Caveats

* Entities without sufficient history are skipped
* No imputation beyond what the baseline logic implies
* No clipping or post-processing is applied by default

---

## License / Usage

Internal VIEWS package. Intended for research and forecasting pipelines, not as a general-purpose forecasting library.

