# ADR-019: views-baseline's Use of FeatureFrame (Input)

**Status:** Accepted (implemented PR-2, issues #52–#58, 2026-07-20)
**Date:** 2026-06-04 (proposed); 2026-07-20 (accepted)
**Deciders:** Simon, VIEWS platform team

---

## Implementation status (read this first)

> **IMPLEMENTED as a dual-input boundary.** Every model now accepts **either** a pandas
> `DataFrame` **or** a `views_frames.FeatureFrame`, normalized once at entry by the single
> adapter `to_feature_frame` (`views_baseline/model/frames/input.py`). Model internals —
> windowing (`window_pool`), point/Mixture aggregations, and the distributional fits —
> run on the FeatureFrame's numpy panel; **pandas is confined to the adapter** (a DIP
> boundary; the model layer no longer imports pandas at runtime — guarded by
> `tests/test_falsification_reorg_principles.py::test_dip_model_layer_has_no_runtime_pandas_import`).
>
> **Why dual-input, not FeatureFrame-only:** pipeline-core still hands models a `DataFrame`
> (blocker #1 below is unchanged — no FeatureFrame *producer* exists upstream). Rather than
> wait, baseline **decouples its readiness**: it converts df→FeatureFrame at its own boundary,
> so the pure-FeatureFrame path is live and tested now, and a future pipeline-core FeatureFrame
> producer becomes a pass-through with no model change. In production the df→FF conversion runs;
> the FF passthrough has no live upstream producer yet (forward-looking readiness).
>
> **Precision (C-32):** `views_frames.FeatureFrame` is a **float32** container by design.
> The chosen direction is FeatureFrame-canonical, so pooled magnitudes are float32-rounded.
> The distributional golden tests use integer data (float32-exact) and stayed byte-identical;
> the entity/tail/RNG **order** is preserved exactly (float64 guard in `test_pooling`). A
> df and the FeatureFrame built from it converge to the *same* float32 frame inside the
> boundary, so the two input routes are **exactly** equivalent (`tests/test_dual_input.py`).

---

## Context

ADR-017 records that the platform intends to phase out DataFrames using two containers: PredictionFrame for output (implemented in baseline — ADR-018) and **FeatureFrame for input**. FeatureFrame exists today in `views-datafactory` (`src/datafactory_adapters/feature_frame.py`: `y_features` shape `(N, D)` or `(N, D, S)`, `identifiers`, `feature_names`, validation) and is consumed experimentally by `views-bayesian` and `views-lab00`. It is the input-side analogue of PredictionFrame.

**Crucially, `views-pipeline-core` does not use FeatureFrame.** Its dataloaders hand models a `pd.DataFrame` (cached as `*_df.parquet`); `_LOA_TO_OUTPUT_FORMAT` offers only `dataframe` / `country_month`. Because baseline receives whatever the pipeline hands it, **baseline cannot adopt FeatureFrame unilaterally** — the input contract is set upstream.

---

## Decision (in effect)

views-baseline models accept **`pd.DataFrame | views_frames.FeatureFrame`** and normalize to a
`FeatureFrame` at entry. The implemented shape:

- **One boundary adapter** — `to_feature_frame(x, *, loa, targets) -> FeatureFrame`
  (`model/frames/input.py`): a DataFrame is lifted via `FeatureFrame.from_2d` +
  `SpatioTemporalIndex` (pandas imported lazily inside this one function); a FeatureFrame passes
  through after a declared-`loa` ↔ frame-`level` agreement check (ADR-003). `panel(ff, targets)`
  is the numpy view (`(time, unit, {target -> (N,) values})`) the internals read.
- **Internals are numpy-on-FeatureFrame** — `window_pool` consumes a FeatureFrame; the point
  models (`Locf`=window-of-1, `Average`=window-mean) and `MixtureBaseline` reuse it; the
  distributional fits/samples run on the numpy pools. No model reads `df.index` / `groupby` /
  `.xs` any more.
- **Protocol widened** — `BaselineModel` / `DistributionalBaselineModel` `fit`/`predict` hints are
  `pd.DataFrame | FeatureFrame` (ISP-minimal).
- **Sample axis** — baseline input is observed data, so `S == 1`; `panel` takes the `S=0` slice.
- **Input validation** — `to_feature_frame` + `resolve_level` validate the declared level against
  the index (DataFrame) or the frame's own `level` (FeatureFrame), partially subsuming C-05
  (MultiIndex assumption) and C-07 (targets-in-columns) at the boundary.

## Dependencies / blockers (status)

1. **pipeline-core has no FeatureFrame *producer*** — dataloaders still produce DataFrames. This
   is **no longer a blocker** for baseline: the dual-input boundary converts df→FeatureFrame
   itself, so baseline is FeatureFrame-native internally regardless of upstream. The pure-FF
   passthrough simply has no live producer yet (forward-looking).
2. **Cross-repo format contract is fragile** — output-format string literals duplicated between
   pipeline-core and datafactory (C-62). Unchanged; orthogonal to baseline's internal boundary.
3. **Ownership** — emitting FeatureFrame on input remains pipeline-core's decision; baseline now
   *accepts* it whenever it arrives, at zero further cost.

## Consequences

- Completes the DataFrame phase-out for baseline **on baseline's own terms**: input + output are
  both frame-based (ADR-018 output; this ADR input), with pandas confined to one adapter.
- **New risk accepted (C-32):** FeatureFrame float32 vs the historical float64 outputs — a
  one-time, deliberate precision boundary; ordering/logic preserved and guarded.
- The pure-FeatureFrame path is exercised by tests but has no upstream producer, so its
  production value is architectural readiness until pipeline-core emits FeatureFrames.

## Status transition

**Proposed → Accepted (2026-07-20).** Condition (2) — a concrete baseline adapter/signature change
— is met (`to_feature_frame` + numpy internals + widened protocol, merged in PR-2). Condition (1)
— a pipeline-core FeatureFrame producer — was **superseded** by the dual-input design: baseline no
longer needs upstream to change in order to be FeatureFrame-native. Revisit if/when pipeline-core
ships a producer (the passthrough branch then carries live data).

## References

- ADR-017 (frames-replace-dataframes direction), ADR-018 (PredictionFrame use — output, implemented)
- `views-datafactory` `src/datafactory_adapters/feature_frame.py` (the container)
- `views-bayesian` `feature_frame_loader.py`, `views-lab00` (experimental consumers)
- risk register C-05, C-06, C-07 (input-validation gaps a FeatureFrame contract could subsume); C-62, C-30 (cross-repo format contract)
