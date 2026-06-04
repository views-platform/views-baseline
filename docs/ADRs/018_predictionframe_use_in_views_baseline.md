# ADR-018: views-baseline's Use of PredictionFrame (Output)

**Status:** Accepted
**Date:** 2026-06-04
**Deciders:** Simon, VIEWS platform team

---

## Implementation status (read this first)

> **IMPLEMENTED and in production for output.** Every views-baseline model returns `dict[str, PredictionFrame]` today; this is merged, tested (98 tests), and runs in the integration suite.
>
> **NOT yet resolved (cross-repo, out of this repo's control):** the platform contracts that make a point `(N, 1)` PredictionFrame fully safe in *downstream consumption* — the point/stochastic content descriptor (pipeline-core #159), the ensemble aggregation guard (pipeline-core #160 / risk C-13), and the point-aware readiness check (views-models #81). These are tracked elsewhere; baseline cannot close them.
>
> This ADR describes **only the output side**. Input is still pandas DataFrame — see ADR-019 (FeatureFrame), which is **not implemented**.

---

## Context

views-baseline produces forecasts that the VIEWS pipeline persists and ensembles. The platform's output container is `PredictionFrame` (owned by `views-pipeline-core`, `data/prediction_frame.py`): a numpy-backed `y_pred` of shape `(N, n_samples)` plus an `identifiers` dict (`time`, `unit`). ADR-017 records the platform direction (frames replace DataFrames); ADR-010 records the output-type decision. **This ADR records how views-baseline actually uses PredictionFrame.**

---

## Decision (what views-baseline does today)

All five baseline models return `dict[str, PredictionFrame]` from `predict()` — one PredictionFrame per target.

- **Point models** (`ZeroModel`, `LocfModel`, `AverageModel`): `y_pred` shape `(N, 1)` — one deterministic value per cell — built by `build_prediction_frame()` in `model/helpers.py`. `N = (units at train_end) × output_length`.
- **Distributional models** (`ConflictologyModel`, `MixtureBaseline`): `y_pred` shape `(N, n_samples)`, built inline in `predict()`.
- **Identifiers:** every PredictionFrame carries `{"time", "unit"}` arrays of length `N`, in unit→time order (ADR-011).
- **Manager:** `BaselineForecastingModelManager._generate_predictions` accumulates `dict[str, list[PredictionFrame]]` across sequence numbers — a single code path, no `isinstance` dispatch (ADR-012, ADR-017).
- **Config contract:** `prediction_format` is in `ReproducibilityGate.CORE_GENOME`; deployed baseline configs declare `prediction_format: "prediction_frame"` (+ `skip_predictions_delivery: True`).
- **Empty-entity:** `require_entities()` fails loud with a descriptive `ValueError` rather than emitting an unusable empty/degenerate frame.
- **Coupling:** `PredictionFrame` is lazy-imported at three sites (the two distributional `predict()` methods and `build_prediction_frame`); `model/` stays importable without pipeline-core, but model `predict()` now requires it (ADR-013).

There is **no DataFrame output path**. `build_prediction_grid()` is retained in `helpers.py` for reference but is called by nothing.

---

## What is explicitly NOT done here

- **No content descriptor.** A point frame is distinguished only by `y_pred` width `(N,1)` and the model's `distributional` marker — there is no first-class "point | stochastic | parametric" tag on the frame. That belongs to pipeline-core (#159).
- **Point models are not yet safe in PFE concat ensembles.** Mixing `(N,1)` with `(N,n_samples)` in `PredictionFrameEnsembleManager` silently corrupts (risk C-13 / pipeline-core #160). Baseline point models are currently used as evaluation benchmarks, not ensemble constituents, so this is latent — but unguarded.
- **The readiness unit test** (`views-models` `TestPFModelConfigReadiness`) still assumes PredictionFrame ⇒ sampled; runtime works for point models, but that check needs point-awareness (views-models #81).

## Consequences

**Positive:** one output container; uniform manager path; point baselines are first-class PredictionFrame producers.

**Negative / honest costs:** model `predict()` tests now require pipeline-core (ADR-013); the safety of point frames downstream depends on cross-repo work not yet done (#159/#160/#81); the input side is unchanged (still DataFrame — ADR-019).

## References

- ADR-010 (output-type decision), ADR-017 (frames-replace-dataframes direction)
- ADR-012 (model addition), ADR-013 (coupling / lazy import), ADR-011 (RNG order)
- ADR-019 — FeatureFrame use (input; **not implemented**)
- pipeline-core #159 / #160, views-models #81; risk register C-13
