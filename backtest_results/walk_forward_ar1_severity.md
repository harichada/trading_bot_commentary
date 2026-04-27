# Walk-Forward AR(1) Rejection Severity — W1 vs W2 vs W3

**Verdict — MODERATE severity (neither near-miss nor far-miss).** Both
held-out windows breach the median-gate threshold, but neither breaches
the max-gate. The window-to-window difference is roughly one fold's
worth of label autocorrelation drift; the base-rate sweep in Phase 2
will tell us whether W1's pass is typical or fortunate.

## Per-Fold AR(1) Values (absolute)

Source: `backtest_results/walk_w{1,2,3}_training.json` →
`iid_diagnostics.per_fold_ar1`. The trainer's gate evaluates absolute
values; sign flips are not informative for AR(1) magnitude.

| Fold | W1 (PASS) |W2 (REJECT) | W3 (REJECT) |
|:----:|----------:|-----------:|------------:|
| 1    | 0.084     |  -0.194    |   0.059     |
| 2    | 0.169     |   0.173    |   0.160     |
| 3    | 0.073     |   0.084    |   0.169     |
| 4    | 0.085     |  -0.073    |   0.161     |
| 5    | 0.188     |   0.173    |   0.097     |

## Window-Level Gate Diagnostics

| Window | median \|AR(1)\| | max \|AR(1)\| | median ≤ 0.10? | max ≤ 0.20? | Outcome |
|--------|-----------------:|--------------:|:--------------:|:-----------:|---------|
| W1     | **0.085**        | 0.188         | YES (Δ=0.015)  | YES (Δ=0.012)| PASS    |
| W2     | **0.173**        | 0.194         | NO  (over by 0.073) | YES (Δ=0.006)| FAIL — median |
| W3     | **0.160**        | 0.169         | NO  (over by 0.060) | YES (Δ=0.031)| FAIL — median |

Both rejections fired on the **median** gate alone. The **max** gate
was satisfied in every window, including the rejected ones.

## Per-Fold Severity Classification

Each fold's |AR(1)| classified against the median-gate bins specified
for this diagnostic (near-miss = 0.10–0.15, moderate = 0.15–0.25,
far-miss ≥ 0.25). Folds with |AR(1)| < 0.10 individually meet the
gate's median criterion.

| Window | would-pass (<0.10) | near-miss (0.10–0.15) | moderate (0.15–0.25) | far-miss (≥0.25) |
|--------|:----------------:|:---------------------:|:--------------------:|:----------------:|
| W1     | 3 (folds 1,3,4)  | 0                     | 2 (folds 2,5)        | 0                |
| W2     | 2 (folds 3,4)    | 0                     | 3 (folds 1,2,5)      | 0                |
| W3     | 2 (folds 1,5)    | 0                     | 3 (folds 2,3,4)      | 0                |

## Window-Level Severity (median gate)

The window's median |AR(1)| classified directly:

| Window | median \|AR(1)\| | Bin |
|--------|-----------------:|-----|
| W1     | 0.085            | would-pass |
| W2     | 0.173            | **moderate** |
| W3     | 0.160            | **moderate** |

Neither held-out window is in the **near-miss** band (the gate is not
borderline-strict on this data: clearing 0.10 by ±0.005 would not
change the verdict for either). Neither is in the **far-miss** band
(the data regime is not catastrophically different — a single fold
swap of moderate→would-pass would have rescued either window).

## Interpretation

The cross-window pattern is striking and worth flagging:

* **W1 has 2 moderate folds; W2 has 3; W3 has 3.** The gate threshold
  (median ≤ 0.10) acts as a hard cutoff on a quantity that drifts
  smoothly from window to window. With 5 folds, having 3+ moderate
  folds *guarantees* the median lands in the moderate band; having 2
  moderate folds (W1) lets the median be pulled down by the other 3
  would-pass folds.
* **The gap between W1 and W2/W3 is one fold.** Not a regime change —
  one additional fold drifting from |AR(1)|<0.10 into the 0.15–0.20
  range. That is well within what we would expect from re-sampling
  noise on a small (~1000-sample) labeled dataset.
* **The gate is fragile-by-construction at 5 folds.** With an even
  number of "moderate" folds determining the median, the experiment
  exhibits high sensitivity to which specific 60-day window we look
  at. This is a separate concern from whether the *underlying* alpha
  generalizes.

That said, "moderate" severity is consistent with two distinct
hypotheses we cannot yet distinguish:

1. **Gate is appropriately strict; W2/W3 labels genuinely have higher
   autocorrelation than W1's.** Sequential bootstrap (AFML §4.5.3) is
   meaningfully degraded at |AR(1)|≈0.17, and the retune's
   hyperparameters were tuned under the IID-friendlier W1 conditions.
   The gate is correctly catching that.

2. **Gate is too strict for the regime; W1 narrowly cleared and W2/W3
   narrowly missed.** The raw AR(1) magnitudes (0.16–0.19) are within
   the "weak-but-nonzero" band where sequential bootstrap is degraded
   but not collapsed. AFML §4.5.3 does not prescribe an exact
   threshold; 0.10 is a project choice, not a literature constant.

The base-rate sweep (Phase 2) discriminates between these: if many
windows pass the gate, hypothesis 1 holds (W2/W3 are unlucky draws);
if few pass, hypothesis 2 is more credible. We will not relax the
gate during the sweep — the gate is the experimental control variable.
