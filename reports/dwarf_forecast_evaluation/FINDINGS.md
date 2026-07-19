# Parametric baselines vs conflictology — full forecast evaluation (the seven dwarfs)

**Epic #33 follow-up · date 2026-07-19**
Platform-native evaluation of the parametric distributional baselines against `white_ranger`
(`ConflictologyModel`) in **views-models**, with rigorous out-of-sample, calibration, tail,
and seed-robustness testing driven by an expert-method-review.

## Models

Seven white_ranger clones (identical targets `lr_{sb,ns,os}_best`, pgm, `window_months=36`,
`n_samples=64`, `seed=42`, calibration + validation partitions) — only the model varies:

| dwarf | algorithm | family / transform |
|---|---|---|
| doctorish_dwarf | ParametricConflictology | nb / none |
| grumpy_dwarf | ParametricHurdleConflictology | lognormal / none |
| happy_dwarf | ParametricHurdleConflictology | lognormal / log1p |
| sleepy_dwarf | ParametricHurdleConflictology | gamma / none |
| bashful_dwarf | ParametricHurdleConflictology | gamma / log1p |
| sneezy_dwarf | ParametricHurdleConflictology | gumbel / none |
| dopey_dwarf | ParametricHurdleConflictology | gumbel / log1p |

Run conventionally: `WANDB_MODE=offline python main.py -r <calibration|validation> -t -e -sa`
(offline, cached data). Evaluation = the platform's `hydranet_ucdp` profile (CRPS, QS_sample,
MCR_sample, Brier_rgs_sample) vs ground truth. nb/gamma/lognormal/gumbel need no code change —
S7 already wired the algorithms into the catalog + genome.

## Headline

> **The plain parametric baselines (nb, gamma, lognormal) match conflictology but don't robustly
> beat it once out-of-sample + seed noise are accounted for.** BUT the follow-up **zero-inflated
> NB (ZINB)** — added after this analysis — is the **strongest family tested**: it robustly beats
> plain nb on both partitions, and robustly beats conflictology on the two low-volume targets
> (`lr_ns`, `lr_os`); on the high-volume `lr_sb` it edges conflictology at 2/3 seeds but is within
> seed noise. Robust negatives throughout: `log1p` catastrophic, `gumbel` worse. All models —
> conflictology included — under-disperse on active cells.

## ZINB (added 2026-07-19, model `stumpy_dwarf`) — the one change that moved the needle

`ParametricConflictology(family="zinb")`: a structural-zero spike `π` on top of NB; fixes the NB
dispersion from the window, solves the inflated NB mean so mean **and** empirical zero-rate match
exactly (falls back to plain nb when no excess zeros — no magic π). Motivated by the recommendation
to escalate `nb → ZINB → hurdle` on 99%-zero data.

| partition | ZINB avg CRPS | nb | conflictology | best-plain |
|---|---|---|---|---|
| calibration | **0.1077** | 0.1101 | 0.1094 | lognormal 0.1071 |
| validation (seed 42) | **0.1773 (best)** | 0.1795 | 0.1799 | gamma 0.1782 |

Seed-robustness (validation, seeds 42/123/456):
- **ZINB > conflictology robustly on `lr_ns`, `lr_os`** (lower at every seed; tiny seed noise there).
- **`lr_sb`: wash** — ZINB beats conflictology at seeds 42/123, loses at 456 (within the ~0.006
  seed-noise band). The seed-42 "significant −0.0065" was partly a favourable draw.
- **ZINB > plain nb robustly** — lower avg CRPS + lower on `ns`/`os` at every seed, both partitions.

**Verdict: ZINB is the best parametric baseline — it genuinely (if modestly) improves on both
conflictology and plain NB, robustly except on the single hardest target.** The zero-inflation is
the ingredient that helped; plain NB under-modelled the zeros.

The rigorous testing *changed the conclusion*: a single-partition read said "lognormal beats
conflictology by 1.4% (13/13 windows)"; out-of-sample + seed testing dissolved that to a tie.

## Evidence trail

### 1. Calibration partition (in-sample-ish) — CRPS vs truth
`lognormal/none` best (avg CRPS 0.1071, −1.4% vs conflictology 0.1094, better in **13/13**
rolling-origin windows on all 3 targets); gamma tied; nb slightly worse; gumbel worse; **log1p
catastrophic** (lognormal/log1p 67× worse — per-sample `expm1` inflates the tail).

### 2. Calibration PIT + sharpness (Gneiting)
- **Pooled PIT is uninformative** — every model looks perfectly uniform (TV≈0.002) because the
  99% zeros dominate. **Must condition on active cells.**
- **On active cells (y>0), every model is under-dispersed** (U-shaped PIT, TV 0.5–0.68) —
  including conflictology. When conflict happens, the observed toll routinely exceeds the
  model's high quantiles. The parametric models inherit this; none fixes it. lognormal is
  marginally least-bad (widest sensible intervals); log1p pathological (sharpness 300–1500 from
  monster draws, yet still under-covering).

### 3. Active-cell CRPS reorders the calibration ranking
The pooled winner was partly a zero-mass artifact. On active cells: **nb is the most consistent**
(≥ conflictology on all 3 targets); lognormal wins big on `lr_sb` only; **gamma is actually weak
on active `lr_sb`/`lr_ns`** (its pooled "tie" came from the zeros); gumbel worst.

### 4. Honest significance (Diebold)
Wilcoxon-on-13-overlapping-windows is not valid inference. Origin-block bootstrap of the per-cell
CRPS differential (13 rolling origins as the exchangeable unit) confirms the calibration ranking
is *consistent across origins* — but see §6/§7: consistency across origins ≠ robustness across
partitions or seeds.

### 5. Validation partition (2022–2025, genuinely out-of-sample) — the real test (Hegre)
Fetched held-out UCDP data (months 505–552). **The calibration winner did NOT replicate.**

| model | calibration avg CRPS | validation avg CRPS |
|---|---|---|
| gamma / none | 0.1097 (tie) | **0.1782 (best)** |
| nb / none | 0.1101 | 0.1795 |
| conflictology (ref) | 0.1094 | 0.1799 |
| lognormal / none | **0.1071 (was best)** | 0.1811 (now 4th, worse) |
| gumbel / none | 0.1150 | 0.1865 (worse) |
| log1p (any) | catastrophic | catastrophic |

`lognormal`'s edge evaporated/reversed out-of-sample — textbook winner's curse. No `/none`
family robustly beats conflictology on `lr_sb` (CIs span 0). Caveat: newest UCDP months (~550+)
are incomplete (reporting lag).

### 6. Tail-weighted scoring — twCRPS on validation (Lerch)
Threshold-weighted CRPS `twCRPS_t(F,y)=CRPS(max(X−t,0),max(y−t,0))`, t∈{0,10,100}. As the focus
moves to the extreme tail (t=100), **gamma/none and nb/none get better** relative to conflictology
(both ≥ on all 3 targets), while **lognormal/none gets *worse*** (its heavy tail over-disperses;
the proper score punishes it) — **refuting Lerch's own hypothesis** that lognormal's tail would
win. gumbel worse throughout. Still: nothing robustly beats conflictology on `lr_sb` even in the tail.

### 7. Seed sweep (Sculley) — the decisive null
4 models × seeds {42,123,456} on validation. On `lr_sb`, seed SD ≈ 0.002–0.003 (range ≈ 0.006)
is **as large as or larger than every between-model gap**. The ranking *reshuffles by seed*
(at seed 456, conflictology is best). So the `lr_sb` differences among conflictology / nb / gamma /
lognormal are **Monte-Carlo noise at n_samples=64**. lognormal's "worse on validation" was partly
a seed-42 unlucky draw. (Low-volume `lr_ns`/`lr_os`: seed noise tiny, gaps tiny — real but negligible.)

### 8. Mechanism — why nb and gamma differ on active cells
By construction both have the same overall predictive mean (moment-matched climatology). But:
- **nb (all-cells)** emits positives *more often* and *smaller* (dilutes its small mean across
  frequent small counts). Median-if-positive systematically **below** gamma.
- **gamma (hurdle)** reserves a zero-spike at the true zero-rate, then puts positive mass at the
  *actual positive magnitude scale* → fewer but larger positives → closer to the (large) active
  outcomes → its slight tail edge (§6).
- Caveat: on single-event windows (~40% of active cells) gamma degenerates to a **point mass** at
  the one observed value (over-confident, no spread); nb keeps genuine spread. A frequency-vs-
  magnitude, spread-vs-concentration trade — neither uniformly better.

## Recommendation

- **For a parametric baseline that matches conflictology:** `nb/none` (most consistent, native-zero,
  simplest) or `gamma/none` (slight tail edge, hurdle). Both tie conflictology at real forecasting.
- **Do not use `log1p`** (catastrophic) or `gumbel` (consistently worse).
- **Do not claim a skill improvement over conflictology** — the data don't support it.
- The parametric models earn their place on **extensibility** (covariates, pooling) and being a
  proper generative object, not on out-of-the-box skill.

## Open / remaining (methodological risks from the expert-method-review)
- M-2 out-of-sample: **addressed** (validation run). M-3 calibration: **addressed** (PIT — all
  under-disperse). M-4 tail metric: **addressed** (twCRPS). M-5 winner's curse: **addressed**
  (seed sweep — dissolved the effect). Remaining: a true forecast partition; a decision-relevant
  metric (TADDA/AP); larger `n_samples` to shrink Monte-Carlo noise; `window_months` sensitivity.

## Reproduce
Model dirs under `../views-models/models/{doctorish,grumpy,happy,sleepy,bashful,sneezy,dopey}_dwarf`.
Analysis scripts (session scratchpad): `compare_partition.py`, `calib_analysis.py`+`report_calib.py`
(PIT/CRPS stratified), `twcrps.py`, `seed_sweep.py`. Raw per-cell prediction dirs were deleted to
reclaim disk after metrics were extracted; re-run the model dirs to regenerate.
