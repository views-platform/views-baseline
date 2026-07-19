# ADR-022: Parametric Climatology Baselines — Distribution/Transform Design and Measurement Protocol

**Status:** Accepted
**Date:** 2026-07-18
**Deciders:** Simon, VIEWS platform team
**Consulted:** expert-code-review (2026-07-18); Vesco et al. 2026 (`vesco2026`)
**Informed:** views-models maintainers

---

## Context

`ConflictologyModel` is an *empirical* climatology: for each pgm cell it resamples the entity's last `window_months` observed values — its predictive law is the discrete empirical distribution of that window, drawn **independently** across cell/time/target. We want two **parametric** distributional baselines that fit a distribution to the *same* per-entity window and sample from it, designed to get **as close to conflictology as possible** (indistinguishable at ~1000 draws/cell) while being proper, extensible generative models.

Two things make this tractable and worth doing: (1) because conflictology's joint law *factorizes*, being indistinguishable reduces to matching each per-(cell, target) 1-D marginal — *if* the new models are also independent across those axes; (2) a parametric family that provably matches the empirical baseline is a validated foundation for later covariate/hierarchical extension. The design is motivated in part by **Vesco et al. 2026** (JCR), whose reported-value-inflated (RVI) **Gumbel mixture** for reporting uncertainty validates both the mixture/hurdle structure and Gumbel's "right-skew, lighter-than-lognormal upper tail" for conflict magnitudes.

A decision is needed now because the implementation touches ADR-007 "sensitive areas" (extracting shared logic from `ConflictologyModel`; new genome keys) and must fix, up front, *how similarity to conflictology is measured* — otherwise "as close as possible" is unfalsifiable.

---

## Decision

### 1. Two models (ontology cat. 2 — Distributional; stability: Evolving)

- **`ParametricConflictology`** — no-hurdle, native-zero family: `family` ∈ {`nb`, `zinb`}. One distribution fit to the full window (zeros included); `zinb` (zero-inflated NB) adds a structural zero-spike for windows with more zeros than plain NB predicts. (Tweedie was implemented, measured, and **excluded** in S9; `zinb` was **added** post-hoc and is the best-performing family — see the records below.)
- **`ParametricHurdleConflictology`** — hurdle: a zero-spike (`w` = empirical zero-rate, Bernoulli) + a positive-part family fit to the positive values: `family` ∈ {`lognormal`, `gumbel`, `gamma`} (and, later, zero-truncated `nb`/`tweedie`).

Both are `distributional = True`, return `dict[str, PredictionFrame]` `(N, n_samples)` via the **single `to_prediction_frames` seam** (ADR-020), with the spatial `level` from `resolve_level` and the ADR-011 entity→time→target RNG order (fresh `default_rng(seed)`).

### 2. Decomplected core (shared, reused by both)

- A **`window_pool()`** helper extracted from `ConflictologyModel` and shared, so the pool is **byte-identical** across all three — the precondition for a valid comparison.
- A **distribution strategy registry** (`model/distributions.py`) keyed by family: `fit(values) -> params` (method-of-moments) and `sample(params, size, rng) -> ndarray`.
- A **`TRANSFORMS`** registry: `none` and `log1p` as `(forward, inverse)` pairs (inverse = `expm1`).
- An **input-adapter seam** that lifts the incoming pandas `DataFrame` to numpy + `SpatioTemporalIndex`/`resolve_level` **once**, confining pandas to that boundary.

### 3. Rules (binding)

- **Transform is scoped and validated per family.** `log1p` applies only to continuous positive-part families. `nb`/`tweedie` + `log1p` is a **contract violation and fails loud** (ADR-021 Zero-Magic — no silent no-op).
- **Detransform per sample.** Apply `expm1` to *each drawn sample*, never to a summary mean (Jensen: `E[expm1(Y)] ≠ expm1(E[Y])`). Sampling makes the empirical detransformed distribution correct without an analytic `σ²/2` correction.
- **Numeric fallbacks fail safe and loud-ish.** Underdispersed windows (`var ≤ mean`, breaks NB moment-matching) → Poisson or point-mass; degenerate windows (all-zero / single value) → **point-mass** (matches conflictology exactly); positive-only families require positive input (guaranteed by the hurdle). Fallbacks emit a WARN (ADR-008 data-driven signal), never `NaN`/crash.
- **`expm1` overflow clamp (ceiling).** A single-sourced `EMIT_LOG_CEIL` constant caps log-space samples before `expm1`, with a WARN on clamp — mirroring hydranet's C-113/C-142 guard (a heavy-tail draw otherwise → astronomical counts / float overflow).
- **Non-negativity floor.** A single-sourced `EMIT_FLOOR` (= 0) floors the *raw emitted scale* after any `expm1`, with a WARN on floor — the ceiling's mirror image. Some positive-part families (`gumbel_r`) have support on all of ℝ, so their left tail yields negative draws (directly for `transform="none"`, or via `expm1(x)∈(-1,0)` for `x<0`); the floor guarantees the models never emit a negative conflict magnitude, which would also distort the closeness metrics against (non-negative) conflictology.
- **Genome (ADR-021).** `family`, `transform`, `seed` are **required, audited** `ALGORITHM_GENOMES` keys for both models — no magic defaults; `DEFAULT_SEED` remains the direct-construction sentinel only.

### 4. Measurement protocol (how "close to conflictology" is decided)

Operationalize "I couldn't tell which model the 1000 draws came from" rigorously:

- **Metrics (per cell, per target):** 1-Wasserstein (interpretable, target units) + energy distance + a **Classifier Two-Sample Test (C2ST)** accuracy/AUC (0.5 = indistinguishable).
- **Same-model null (the load-bearing device):** compute every metric for conflictology(seed A) vs conflictology(seed B) to get the finite-N noise floor. A candidate is "indistinguishable" only if its metric lands **inside** that null band.
- **Stratify** by activity regime (all-zero / low / high), target, and level; report the *distribution* of per-cell discrepancies (median, 90th/99th), never a single pooled number (which the zero-majority would dominate).
- **Equivalence, not significance.** Confirm similarity via a TOST/ROPE-style test against a **pre-registered negligibility margin δ**, not a p-value (which is over-powered at N=1000×many-cells). Working default δ (finalized in the S8 experiment): stratified C2ST accuracy ≤ **~0.55** above null, per-cell Wasserstein within the null band. Optionally report the **detectability-vs-N** curve.

### Scope

**In:** the two models, the registries, fallbacks/clamp, the input seam, catalog/genome wiring, the measurement harness and ranking experiment, governance.
**Out:** covariate/hierarchical extensions; FeatureFrame *input* adoption (ADR-019 not implemented — pandas-in stays, lifted internally); the OCP registration-table refactor (C-19, deferred); any change to `ConflictologyModel`/`MixtureBaseline` *behaviour* (only a pure extraction).

---

## Rationale

- Conflictology's factorized empirical law means marginal-matching suffices — so the models keep its independence structure and are compared 1-D per cell. The same-model null makes the comparison *valid* at finite N; equivalence testing makes it a *claim*, not a vibe.
- The hurdle mirrors Vesco's RVI mixture (point mass at a distinguished value + continuous part); their distinguished value is the reported count, ours is 0 (zero-inflation). Their empirical win for Gumbel (lighter upper tail than lognormal) is a well-motivated, in-ecosystem hypothesis for our positive-part — to be *tested*, not assumed.
- Decomplecting pooling / family / zero-handling / scale keeps the two classes thin and additive (a new family is a registry entry), and confines pandas so the ADR-019 future is a one-seam swap.

---

## Considered Alternatives

- **One parameterized class vs two classes.** One engine keyed by (family, hurdle, transform) is DRYer; two named classes read better and each gets a clean genome. **Chosen:** two *thin* classes over one shared core — the split is contract-level, not logic-level.
- **MLE vs method-of-moments fits.** MLE is more accurate but heavier and needs per-family optimizers. **Chosen:** moment-matching for the MVP (closed-form, transparent); MLE is a later ablation.
- **Smoothed-bootstrap / KDE as the "closest" family.** With bandwidth→0 it *is* conflictology, so it would trivially win but isn't a parametric model. **Deferred:** valuable as a *ceiling reference* in the harness, not as a shipped family.
- **ZINB (zero-inflated negative binomial) as a native-zero family — ADDED 2026-07-19.** A structural zero-spike `π` on top of NB: the NB dispersion is fixed from the window and the inflated NB mean solved so that **both** the sample mean and the empirical zero-rate match exactly, falling back to plain NB when the NB already explains the zeros (no magic `π` — ADR-021). Motivated by escalating `nb → ZINB → hurdle` on 99%-zero data. In the platform-native dwarf evaluation it is the **best parametric baseline** — robustly beats plain `nb` on both partitions and beats conflictology on the low-volume targets (`lr_ns`/`lr_os`); on the high-volume `lr_sb` it is within seed noise. Evidence: `reports/dwarf_forecast_evaluation/FINDINGS.md`.
- **Tweedie (compound Poisson-Gamma) as a native-zero family — evaluated in S9 and EXCLUDED.** Tweedie was implemented (index `p` derived from the window's mean/var/zero-rate, so the zero mass matches by construction) and measured against conflictology. It was competitive on *magnitude* (best/near-best Wasserstein & energy at high activity) but lost C2ST to `nb`, and — decisively — the family-feasibility condition `m² < var·(−ln zero_rate)` is **violated on ~40% of real active pgm cells** (windows with 1–2 events in 36 months, which a continuous family cannot represent). The catastrophic `p≥2` / overflow regime never occurred on conflict data (that theoretical risk, formerly C-27, is **withdrawn**). The only ways past the feasibility wall were to *clamp `p`* (silently approximate a non-existent fit — violates ADR-021 "no silent magic") or *empirically resample the window* (become conflictology on 40% of cells — no longer a parametric generative model). **Chosen:** exclude Tweedie from this baseline; it stays viable for a future data-rich (pooled/covariate) model where each fit sees enough observations that the feasibility wall dissolves. Evidence: `reports/closeness_experiment/FINDINGS.md` §S9.

---

## Consequences

**Positive:** a validated parametric baseline with a rigorous fidelity measurement; extensible; the harness is reusable for any future distributional baseline.
**Negative / accepted:** no smooth family will be *perfectly* indistinguishable on small discrete windows (conflictology emits only the window's values) — expected; the harness *localizes* this. The shipped families are five method-of-moments fits (`nb` and `zinb` no-hurdle native-zero; `lognormal`/`gumbel`/`gamma` hurdle positive-parts), sampled via numpy generators; **Tweedie was evaluated and excluded** and **`zinb` was added** (see Considered Alternatives). The hurdle adds a positive-part fit; the transform round-trip adds Jensen/overflow subtleties (handled by the rules above).

---

## Implementation Notes

Delivered as epic #33, stories #34–#43 (tracking #44), in order:
S1 this ADR → S2 `window_pool` + input seam → S3 measurement harness (null-calibrated on conflictology) → S4 distributions/transforms registry (nb/lognormal/gumbel/gamma) → S5 `ParametricConflictology` → S6 `ParametricHurdleConflictology` → S7 catalog/genome wiring + illegal-combo validation → S8 closeness experiment + findings → S9 Tweedie evaluated + **excluded** (feasibility) → S10 governance close-out.

Reuse: `to_prediction_frames`/`resolve_level`/`DEFAULT_SEED` (ADR-020/021); `scipy.stats` (`nbinom`,`lognorm`,`gumbel_r`,`gamma`); the hydranet `EMIT_LOG_CEIL` clamp *pattern* (mirror, do not import).

---

## Validation & Monitoring

- Pool identity: a test asserts the extracted `window_pool` reproduces conflictology's `hist_per_entity` byte-for-byte.
- Per-family: param recovery on synthetic data; reproducibility under seed; transform round-trip (raw in ⇒ raw out); fallbacks fire; clamp caps overflow; floor caps the ℝ-support left tail (gumbel emits no negative magnitude).
- Harness: same-model null band computed; parametric families ranked, stratified, with an equivalence verdict vs δ on real pgm data (`white_ranger` parquet).
- Failure mode that reopens this: a family added with a silent default/transform, or a pool divergence from conflictology.

---

## Open Questions

- Final value of δ (pre-register in S8).
- MLE vs moments; whether the hurdle should offer zero-truncated `nb`/`tweedie` positive parts.
- Whether to add a smoothed-bootstrap ceiling reference to the harness.

---

## References

- ADR-001 (ontology), ADR-004 (stability), ADR-007 (sensitive-area gate), ADR-008 (fail loud / WARN), ADR-010 (PredictionFrame output), ADR-011 (RNG determinism), ADR-019 (FeatureFrame input — not implemented; input-seam constraint), ADR-020 (`to_prediction_frames`/`resolve_level`), ADR-021 (Zero-Magic / audited genome keys).
- Vesco, Randahl, Hegre, Högbladh & Yilmaz (2026), "The underreported death toll of wars," *Journal of Conflict Resolution* (`vesco2026`) — RVI Gumbel mixture.
- `views-hydranet` `utils/hurdle_nb.py` + the C-113/C-142 `expm1` overflow guard (pattern to mirror).
- Risk register C-05 (positional index — mitigated by the input seam).
- Epic #33; stories #34–#43; tracking #44.
