# 20-Window AR(1) Base-Rate Sweep — `baseline-P4-retune`

**Verdict — MARGINAL with original window unusual.** The trainer's
gate stack admits 60% of 60-day windows (12/20), so W2/W3 from the
prior 3-window study were not unlucky draws. But among the 12 windows
where the gate admits training, the retune's median PF is **0.764**,
below the 0.78 W4-spec floor. **W1 (PF 0.838) is tied for #2 of 12 —
the original training window is in the top quartile of what the
retune can achieve when the gate admits the data.**

S2 (median passing PF ≥ 0.78) FAILS by 0.016. S3 (median passing exp
≥ -0.15R) PASSES at -0.131R but is uniformly negative (every passing
window has expectancy below zero). S4 (released bundle SHA unchanged)
PASSES.

## Methodology

López de Prado, AFML §11 — combinatorial walk-forward CV / base-rate
analysis. 20 strictly disjoint 60-day windows spanning the data
history (2021-01-04 → 2026-04-17). Min gap between consecutive
windows: 11 calendar days. The trainer's gate stack (cost-drag rc=2,
P4 hard-reject rc=3 covering AR(1) / uniqueness / labeling) is
applied as-is — gate thresholds are the experimental control variable
and are NOT tampered with during this sweep.

W01–W03 reuse the prior W1/W2/W3 results from
`walk_forward_validation.json` verbatim (same per-window primary
bundles `ml_model_v2_walk_w{1,2,3}.pkl`); W04–W20 are 17 new
trainings producing `ml_model_v2_baserate_w{04..20}.pkl` (gitignored).
The released `ml_model_v2_retune.pkl` was never written and SHA256 is
byte-identical pre/post the entire run.

## Headline

| Metric | Value |
|--------|------:|
| n windows | 20 |
| n pass (trainer rc=0) | **12** |
| n fail | 8 |
| pass-rate | **0.60** |
| classification | **robust** (≥0.50) |
| median passing PF | **0.764** |
| median passing exp (R) | **-0.131** |
| released-bundle SHA unchanged | **YES** |

## Acceptance Gates (S1–S4)

| Gate | Threshold | Observed | Result |
|------|-----------|----------|--------|
| S1 — pass-rate recorded & classified | informational | 0.60 → "robust" | **PASS (informational)** |
| S2 — median passing PF ≥ 0.78 | hard | 0.764 (Δ −0.016) | **FAIL** |
| S3 — median passing exp ≥ −0.15R | hard | −0.131 (Δ +0.019) | **PASS** |
| S4 — released bundle SHA256 unchanged | hard | unchanged | **PASS** |

`pytest tests/test_walk_forward_base_rate.py -v` → 4 passed, 1 failed.
The failure (S2) records the experimental finding honestly.

## Per-Window Detail

Sorted by window-end ascending. Median and max are absolute values.

| Window | end (UTC) | trainer | reject_kind | med \|AR1\| | max \|AR1\| | PF | exp (R) |
|--------|-----------|---------|-------------|------------:|------------:|---:|--------:|
| W02 | 2021-03-05 | rejected | p4_hard (MED)      | 0.173 | 0.194 |   —   |   —    |
| W04 | 2021-05-15 | rejected | cost_drag          |   —   |   —   |   —   |   —    |
| W05 | 2021-08-15 | rejected | p4_hard (MAX)      | 0.066 | 0.261 |   —   |   —    |
| W06 | 2021-11-15 | ok       | —                  | 0.048 | 0.066 | 0.668 | −0.199 |
| W07 | 2022-02-15 | rejected | p4_hard (MAX)      | 0.048 | 0.279 |   —   |   —    |
| W03 | 2022-07-31 | rejected | p4_hard (MED)      | 0.160 | 0.169 |   —   |   —    |
| W08 | 2022-11-01 | rejected | p4_hard (MED+MAX)  | 0.174 | 0.247 |   —   |   —    |
| W09 | 2023-01-31 | ok       | —                  | 0.070 | 0.087 | 0.806 | −0.107 |
| W10 | 2023-04-30 | ok       | —                  | 0.057 | 0.083 | 0.523 | −0.321 |
| W11 | 2023-07-31 | ok       | —                  | 0.024 | 0.175 | 0.685 | −0.188 |
| W12 | 2023-10-31 | ok       | —                  | 0.059 | 0.087 | **0.962** | **−0.019** |
| W13 | 2024-01-31 | ok       | —                  | 0.040 | 0.095 | 0.724 | −0.160 |
| W14 | 2024-04-30 | ok       | —                  | 0.031 | 0.060 | 0.755 | −0.139 |
| W15 | 2024-07-31 | ok       | —                  | 0.016 | 0.094 | 0.663 | −0.202 |
| W16 | 2024-10-31 | ok       | —                  | 0.036 | 0.077 | 0.809 | −0.104 |
| W17 | 2025-01-31 | rejected | p4_hard (MAX)      | 0.088 | 0.332 |   —   |   —    |
| W18 | 2025-04-30 | rejected | p4_hard (MED+MAX)  | 0.183 | 0.344 |   —   |   —    |
| W19 | 2025-07-31 | ok       | —                  | 0.090 | 0.133 | 0.773 | −0.123 |
| W20 | 2025-10-31 | ok       | —                  | 0.036 | 0.128 | 0.838 | −0.087 |
| W01 | 2026-04-17 | ok       | —                  | 0.085 | 0.188 | **0.838** | −0.083 |

## Rejection Breakdown

```
reject_kind:    p4_hard = 7 (5 involve MAX gate)
                cost_drag = 1
which gate fired (among 7 p4_hard):
                MED only       : 2 (W02, W03)
                MAX only       : 3 (W05, W07, W17)
                MED + MAX both : 2 (W08, W18)
```

**Important correction to the prior 3-window severity diagnostic:**
the prior study saw only MED-gate rejections (W2, W3). The 20-window
sweep reveals that 5 of 7 P4 rejections involve the MAX gate, which
fires on individual outlier folds (|AR(1)| > 0.20). The MAX-gate
failure mode is the dominant rejection mechanism on this dataset —
not the MED gate the Phase 1 sidecar focused on. This means the
median-only "near-miss / moderate / far-miss" classification under-
states the rejection mechanism for W05, W07, and W17; their MAX
values reach 0.261, 0.279, and 0.332 respectively, well into what
would be classified "moderate" or "far-miss" if the bins were
applied to the max instead.

## Histogram of |AR(1)| — All Folds, All Windows (n=95)

5 folds × 19 windows that produced AR(1) values (W04 cost-drag short-
circuited before AR(1) was computed).

```
  [0.00, 0.05)   37  #####################################
  [0.05, 0.10)   30  ##############################
  [0.10, 0.15)    8  ########
  [0.15, 0.20)   13  #############
  [0.20, 0.25)    3  ###
  [0.25, 0.30)    2  ##
  [0.30, 0.35)    2  ##
  [0.35, 1.00)    0
```

* **67/95 (71%)** of fold values are below the median gate (0.10) —
  the bulk of the data is well-behaved.
* **21/95 (22%)** sit in the awkward 0.10–0.20 band — would pass max,
  may fail median if 3+ folds in a window land here.
* **7/95 (7%)** breach the max gate (≥0.20) — these are the outlier
  folds that drive MAX-gate rejections.
* The tail extends to 0.344. No fold reaches 0.35+ on this data.

## Among the 12 Passing Windows

* PFs sorted: `[0.523, 0.663, 0.668, 0.685, 0.724, 0.755, 0.773, 0.806, 0.809, 0.838, 0.838, 0.962]`
* Expectancies sorted (R): `[-0.321, -0.202, -0.199, -0.188, -0.160, -0.139, -0.123, -0.107, -0.104, -0.087, -0.083, -0.019]`
* 6 of 12 PFs below 0.78 (would fail prior W2/W3 floor of 0.70 in 3 cases)
* **All 12 expectancies are negative**. The retune's "edge" is barely-
  not-as-bad-as-cost; it does not beat costs on a single passing
  window in this sweep.
* **W1's 0.838 is tied for #2/12** — original window is favorable
  but not anomalously so. The single best passing window (W12,
  2023-Q4) reaches PF 0.962 / exp −0.019R, suggesting the retune's
  ceiling is not yet plumbed; what's missing is consistent generation,
  not absolute capacity.

## Among the 8 Failing Windows

* P4 hard-rejects: 7 (87.5%); cost-drag: 1 (12.5%)
* Median |AR(1)| of MED-failed windows: 0.173 (W02), 0.160 (W03), 0.174 (W08), 0.183 (W18)
* MAX |AR(1)| of MAX-failed windows: 0.261, 0.279, 0.247, 0.332, 0.344
* Median-only severity bins (per Phase 1 sidecar): near-miss=3,
  moderate=4, far-miss=0. **This metric is misleading** because the
  near-miss windows (W05, W07, W17) actually failed on MAX gate, not
  MED — see correction above.

## Plain-English Verdict

The 60% pass-rate puts the gate stack in the "robust" bucket — most
60-day windows on this data history pass it. So the prior W2/W3
rejections were not unlucky draws. **But that's not the headline.**

The headline is that the retune's PF does not generalize: median
passing PF is 0.764, which fails the 0.78 floor by 0.016. W1 (the
training window) sits at 0.838, tied for #2 of 12 in the passing
distribution. The retune was tuned on a window in the top quartile
of what its own methodology can achieve when the gate admits the
data.

Two specific findings sharpen this:

1. **The MAX-gate is the dominant rejection mechanism, not MED.** The
   prior 3-window study was unrepresentative — both W2 and W3 happened
   to trip only the MED gate, but in the 20-window sweep 5 of 7 P4
   rejections involve outlier folds breaching the MAX gate (|AR(1)|
   ≥ 0.20). Tightening fold-level robustness, not just average IID-
   ness, is where the brittleness lives.

2. **Every passing window has negative expectancy.** Across 12 windows
   spanning four years, the retune does not beat costs in any single
   one. The "edge" the retune optimized for is a relative
   improvement on a uniformly-loss-making baseline; it has not
   demonstrated absolute alpha.

## Decision Map (per the prompt's pre-stated rules)

| Rule | Applies? |
|------|---------|
| pass-rate ≥ 50% AND median passing PF ≥ 0.78 → "foundation robust" | NO — PF fails by 0.016 |
| pass-rate 20–50% → "foundation marginal" | NO — pass-rate is 60% |
| pass-rate < 20% AND severity far-miss → "foundation fragile" | NO |
| pass-rate < 20% AND severity near-miss → "gate too strict" | NO |
| **W1 at high end of passing PF distribution → "original window unusual"** | **YES** (W1 is tied #2 of 12) |
| Phase 1 severity far-miss but base-rate high → "regime outliers" | NO — severity was MODERATE not far-miss |

The result occupies a corner of the decision space the prompt
anticipated only obliquely: **pass-rate is high enough to call the
gate stack "robust", but the apparent alpha sits in the upper tail of
what the retune can produce.** The most defensible characterization
is **"marginal — gate admits, but edge is window-specific"**.

## Recommendation

The user's pre-stated decision rules pointed two ways for the marginal
band. With pass-rate at 60% (above the marginal band's 50% upper) and
S2 narrowly failing (PF 0.764 vs floor 0.78, Δ 0.016), my
recommendation, in order of preference:

1. **Pivot to extraction work (P9–P11) on the W1 baseline as the best
   honest reference.** The 20-window sweep has answered the
   methodology-generalization question we needed it to answer: the
   retune's measured edge is not generalizable in a way that survives
   the W4-spec floor. Continuing alpha work on top of a non-
   generalizing primary will compound the fragility. P9–P11 can use
   the W1 result as the legitimate-but-rare reference it is.

2. **Run the gate-relaxation stress test (Option 2)** *only if* the
   user judges the 0.016-PF gap to be small enough to attribute to
   sample noise on 12 passers. The MAX gate's frequent firing
   suggests this is not a noise issue — but a single experiment
   with the MED gate held at 0.10 and the MAX gate temporarily
   relaxed to (say) 0.30 would isolate which gate is driving the
   PF distribution shift.

I am NOT recommending P7 LLM-news work in the next session. The
foundation is too uncertain to layer additional features on.

## Artifacts

* `backtest_results/walk_forward_base_rate.json` — full per-window report
* `backtest_results/baserate_w{NN}_training.json` — per-window trainer reports (W04–W20)
* `backtest_results/baserate_w{NN}_p3.json` — P3 evaluation reports for the 9 newly-passing windows
* `tests/test_walk_forward_base_rate.py` — S1–S4 acceptance gates
* `validate_walk_forward_base_rate.py` — orchestrator
