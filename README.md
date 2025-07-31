# views-baseline

**views-baseline** is a package providing simple, interpretable baseline models for the VIEWS forecasting pipeline. These models are designed for benchmarking and sanity-checking more complex machine learning models. Baseline models in this package do not require training and include approaches such as always predicting zero or repeating the last observed value.

## Features

- **ZeroModel**: Predicts zero for all targets and all forecast horizons.
- **LastValueModel**: not implemented yet - Repeats the last observed value for each target into the future.
- **Plug-and-play**: Fully compatible with the VIEWS pipeline and model manager interfaces.
- **No training required**: Baseline models are stateless and require no fitting.

## Installation

Clone the repository and install with pip:

```bash
pip install .
```