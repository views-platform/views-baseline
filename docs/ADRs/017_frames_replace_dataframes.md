# ADR-017: Frames Replace DataFrames as the Operational Data Containers

**Status:** Accepted
**Date:** 2026-06-04
**Deciders:** Simon, VIEWS platform team

---

## Context

The VIEWS pipeline historically passed data as pandas DataFrames on both the input and output sides. DataFrames proved computationally infeasible as a *distributional* container: representing `n_samples` draws per cell required list-in-cell DataFrames, which are slow and memory-hungry. `PredictionFrame` (in `views-pipeline-core`, `data/prediction_frame.py`) was created to solve this — a numpy-backed container (`y_pred` shape `(N, n_samples)` + `identifiers`) that holds samples efficiently.

The platform direction is to **purge DataFrames from the central operational path** and lean on two purpose-built "frame" containers:

- **PredictionFrame** — the **output** container (model forecasts).
- **FeatureFrame** — the **input** container (model features).

This ADR records the decision for views-baseline and names the platform-level direction it participates in. The output half is being adopted now (this repo's baselines moved onto PredictionFrame — see ADR-010); the input half is maturing elsewhere and is not yet in the core.

---

## Decision

**PredictionFrame is the universal output container for views-baseline models, including deterministic point forecasts represented as `(N, 1)`.** A PredictionFrame with a single sample (`n_samples == 1`) is a legitimate, first-class deterministic output — not a degenerate special case to be avoided.

### Rationale for allowing `n_samples == 1`

- **It already exists in production.** views-hydranet emits `(N, 1)` PredictionFrames in point mode: `evaluation_mode == "point"` triggers `VolumeHandler.collapse_to_point()`, and the assembler reshapes to `(N, 1)` (`views_hydranet/utils/prediction_frame_assembler.py`, `config_initializer.py` documents "`y_pred.shape == (N, 1)` regardless of `n_posterior_samples`"). A `(N,1)` PredictionFrame is established platform behaviour.
- **DataFrames are being retired.** Once DataFrames leave the operational core, PredictionFrame must carry *every* forecast shape, including point estimates. Refusing `n_samples == 1` would force point models to keep a DataFrame path alive indefinitely — defeating the purge.
- **Point forecasts will not disappear.** We cannot assume the model zoo will be exclusively distributional. Baselines (Zero/LOCF/Average) are inherently deterministic.
- **It does not weaken the container.** A point estimate is a valid degenerate distribution; `(N, 1)` is its honest representation.

### What this means concretely (views-baseline)

- Every baseline model returns `dict[str, PredictionFrame]` from `predict()` (ADR-010). Point models produce `(N, 1)`; distributional models produce `(N, n_samples)`.
- The manager has a single, type-uniform path (no `isinstance` dispatch) — ADR-012, ADR-001, ADR-004.

---

## The input half: FeatureFrame (status, not owned here)

Phasing out DataFrames requires the **input** side too. As of 2026-06-04:

- **FeatureFrame exists in `views-datafactory`** (`src/datafactory_adapters/feature_frame.py`): `y_features` shape `(N, D)` or `(N, D, S)` (observations × features × samples), `identifiers`, validation. It is the input-side analogue of PredictionFrame and already supports a sample dimension, though no current model has input-uncertainty features (so `S` is unused in practice).
- **`views-bayesian` is an early consumer** (`feature_frame_loader.py`).
- **It is NOT yet integrated into `views-pipeline-core`.** The operational dataloaders still produce DataFrames (cached as `{run_type}_{source}_df{format}`).

**Therefore the DataFrame phase-out is deliberately incomplete:** the output half (PredictionFrame) is in the core now; the input half (FeatureFrame) is built and proven in datafactory/bayesian but pending integration into the central pipeline. DataFrames remain on the input path until FeatureFrame lands there. This ADR does **not** claim DataFrames are gone.

---

## Open platform-level work (tracked cross-repo, not in this repo)

Allowing `n_samples == 1` as universal output surfaces contracts elsewhere that still assume "PredictionFrame ⇒ sampled/distributional." These are owned at the platform level and tracked as issues, with solutions intentionally left open:

- **views-pipeline-core #159** — formalise PredictionFrame as the universal container with an explicit point/stochastic (and future *parametric*) content descriptor, rather than inferring meaning from `y_pred.shape[1]`.
- **views-pipeline-core #160** — guard `PredictionFrameEnsembleManager` concat aggregation against mixing heterogeneous sample counts (a point `(N,1)` silently swamped by `(N,64)` constituents). Tracked locally as risk **C-13**.
- **views-models #81** — make `TestPFModelConfigReadiness` point-aware so deterministic models are not required to declare `n_posterior_samples`.

A future PredictionFrame upgrade may carry **distribution parameters** (e.g. a zero-inflation ratio plus heavy-tailed log-normal parameters) instead of samples. That possibility is the reason the content descriptor in #159 should be explicit and extensible rather than shape-inferred.

---

## Consequences

**Positive**
- One output container for all models; the manager's dispatch is uniform (ADR-012).
- Point baselines participate in the PredictionFrame ecosystem without a DataFrame fallback.
- Forward-compatible with a parametric PredictionFrame.

**Negative / costs**
- `views-baseline` model tests now require `views-pipeline-core` at predict time (PredictionFrame is lazy-imported by `build_prediction_frame`), narrowing the "model/ testable without pipeline-core" guarantee (ADR-013).
- Downstream contracts that equated PredictionFrame with sampled output must become point-aware (#81, #159), and aggregation must be guarded (#160, C-13) before point models can safely join distributional ensembles.
- DataFrame retirement is gated on FeatureFrame reaching the core — out of this repo's control.

---

## References

- ADR-010 — PredictionFrame as Universal Output Format (output-type decision for this repo)
- ADR-012 — Model Addition Protocol (single output path)
- ADR-013 — views-pipeline-core Coupling (lazy-import / testability impact)
- `views-pipeline-core` #159, #160; `views-models` #81 (platform-level follow-up)
- `views-datafactory` `src/datafactory_adapters/feature_frame.py` (input-half container)
- risk register C-13 (aggregation guard)
