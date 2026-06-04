# ADR-019: views-baseline's Use of FeatureFrame (Input)

**Status:** Proposed — **NOT IMPLEMENTED**
**Date:** 2026-06-04
**Deciders:** Simon, VIEWS platform team

---

## Implementation status (read this first)

> **NOTHING IN THIS ADR IS IMPLEMENTED.** views-baseline does **not** use FeatureFrame anywhere. It is not imported, not consumed, not referenced in any code path.
>
> **What baseline actually does today:** every model receives a pandas **`DataFrame`** in `fit(self, df: pd.DataFrame)` and `predict(self, df, ...)`. It reads the time/entity axes from `df.index.names[0]` and `df.index.names[1]`, slices targets with `df[self.targets]`, and filters by `df.index.get_level_values(...)`. The input path is **100% DataFrame**.
>
> This ADR is a **proposal** describing how baseline *would* consume FeatureFrame **if and when** `views-pipeline-core` integrates it into the data-loading path. That integration **has not started** (see Dependencies). Until it does, this document is aspirational and must not be read as describing current behaviour.

---

## Context

ADR-017 records that the platform intends to phase out DataFrames using two containers: PredictionFrame for output (implemented in baseline — ADR-018) and **FeatureFrame for input**. FeatureFrame exists today in `views-datafactory` (`src/datafactory_adapters/feature_frame.py`: `y_features` shape `(N, D)` or `(N, D, S)`, `identifiers`, `feature_names`, validation) and is consumed experimentally by `views-bayesian` and `views-lab00`. It is the input-side analogue of PredictionFrame.

**Crucially, `views-pipeline-core` does not use FeatureFrame.** Its dataloaders hand models a `pd.DataFrame` (cached as `*_df.parquet`); `_LOA_TO_OUTPUT_FORMAT` offers only `dataframe` / `country_month`. Because baseline receives whatever the pipeline hands it, **baseline cannot adopt FeatureFrame unilaterally** — the input contract is set upstream.

---

## Proposed decision (conditional, not in effect)

**If** pipeline-core begins handing models a FeatureFrame on input, views-baseline models **should** consume it directly instead of a DataFrame. Anticipated shape of that change (to be designed when the dependency lands — not prescribed here):

- `fit()` / `predict()` would accept a `FeatureFrame` (or be fronted by a thin adapter), reading:
  - time/unit from `FeatureFrame.identifiers["time"|"unit"]` instead of `df.index.names`;
  - target/feature columns from `feature_names` instead of `df[targets]`.
- The deterministic point baselines have **no input uncertainty**, so the FeatureFrame sample axis `S` would be `1` / absent (`(N, D)`); baseline would not exercise `(N, D, S)`.
- FeatureFrame's built-in validation could **subsume** several current input-contract gaps that exist precisely because the DataFrame input is unvalidated — risks C-05 (MultiIndex assumption), C-06 (partition_dict), C-07 (targets-in-columns). This is a potential benefit to flag, not a committed design.

## Dependencies / blockers (why this is not implemented)

1. **pipeline-core has no FeatureFrame input path** — dataloaders produce DataFrames; no PR exists to make FeatureFrame the core's input (the datafactory-fetch roadmap PRs explicitly use `output_format="dataframe"`). This is the hard blocker.
2. **Cross-repo format contract is fragile** — output-format string literals are duplicated between pipeline-core and datafactory (risk C-62), with no contract test (C-30). These should be resolved before/with any FeatureFrame input wiring.
3. **Ownership** — the decision to emit FeatureFrame on input is pipeline-core's; baseline's role is to consume what it's given. A companion platform ADR in pipeline-core is the prerequisite for this one to become Accepted.

## Consequences

- **Until the dependency lands, no change to baseline.** This ADR exists to (a) record the intended direction and (b) make explicit that the *input* half of the DataFrame phase-out is **not done** in baseline — unlike the output half (ADR-018).
- When implemented, it would complete the DataFrame phase-out for baseline (input + output both frame-based) and potentially close C-05/C-06/C-07 via FeatureFrame validation.

## Status transition

This ADR moves from **Proposed** to **Accepted** only when: (1) pipeline-core ships a FeatureFrame input path, and (2) a concrete baseline adapter/signature change is designed and merged. Revisit then.

## References

- ADR-017 (frames-replace-dataframes direction), ADR-018 (PredictionFrame use — output, implemented)
- `views-datafactory` `src/datafactory_adapters/feature_frame.py` (the container)
- `views-bayesian` `feature_frame_loader.py`, `views-lab00` (experimental consumers)
- risk register C-05, C-06, C-07 (input-validation gaps a FeatureFrame contract could subsume); C-62, C-30 (cross-repo format contract)
