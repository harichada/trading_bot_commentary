# Walk-Forward Validation — `baseline-P4-retune`

**Verdict — METHODOLOGY-BRITTLE / TIME-WINDOW ARTIFACT (mixed-leaning-negative).**

The retune's own P4 acceptance gates reject **both** held-out windows
before P3 routing can even score them. W1 replicates byte-for-byte; the
retune's apparent edge does not generalize to the two disjoint windows
this study tested.

## Methodology

López de Prado, AFML §11 (combinatorial walk-forward CV) and §12
(backtest overfitting). Single-window 5-fold purged CV does not control
for **time-window selection bias** — the choice of *which* 60 days to
train on is itself a hyperparameter. To detect that, we re-run the
exact `train_ml_model_v2` + `train_p3_regime_routing` pipeline on three
disjoint 60-day windows and compare aggregate metrics.

The orchestrator (`validate_walk_forward.py`) monkey-patches
`PostgresDataProvider.get_market_data` to inject `end=window_end`,
exploiting the provider's existing `end` kwarg
(`data_providers/postgres.py:43`). **No production code modified.**
`ml_model_v2_retune.pkl` SHA256 verified byte-identical pre/post the
entire run (W5 invariant).

## Per-Window Results

| Window | Period (UTC) | Trainer | Agg PF | Agg Exp (R) | n | Note |
|--------|-------------|---------|-------:|------------:|--:|------|
| W1     | 2026-02-16 → 2026-04-17 | OK | **0.838** | **-0.083** | 818 | Exact replication of the original training window. |
| W2     | 2021-01-04 → 2021-03-05 | **REJECTED** | — | — | 0 | Per-fold AR(1) gate failed: median=0.173 > 0.10. |
| W3     | 2022-06-01 → 2022-07-31 | **REJECTED** | — | — | 0 | Per-fold AR(1) gate failed: median=0.160 > 0.10. |

W2 per-fold AR(1) values: `[-0.194, 0.173, 0.084, -0.073, 0.173]`
W3 per-fold AR(1) values: `[ 0.059, 0.160, 0.169, 0.161, 0.097]`

## Acceptance Gates (W1-W5)

| Gate | Threshold | Observed | Result |
|------|-----------|----------|--------|
| W1 — replication | PF within ±0.05 of 0.838; exp within ±0.020R of -0.083R | PF 0.838 (Δ 0.000), exp -0.083R (Δ 0.000) | **PASS** |
| W2 — oldest window | PF ≥ 0.70 AND exp ≥ -0.20R | undefined (trainer-rejected) | **FAIL** |
| W3 — alternate window | PF ≥ 0.70 AND exp ≥ -0.20R | undefined (trainer-rejected) | **FAIL** |
| W4 — median across 3 windows | PF ≥ 0.78 AND exp ≥ -0.15R | undefined (only 1/3 windows produced metrics) | **FAIL** |
| W5 — released bundle SHA256 unchanged | byte-identical pre/post | unchanged | **PASS** |

**3/6 pytest cases pass** (W1 replication, W5 invariant, structural
3-window presence). **3/6 fail** (W2 floor, W3 floor, W4 median) because
the per-window aggregate metrics are `null` — the trainer never produced
them.

## What This Means

The held-out windows do not fail because the retune's *edge* collapses
on them — they fail because the retune's *labeling pipeline* itself
declares them unsuitable. The per-fold AR(1) gate (P4 hard-reject)
checks that triple-barrier labels are approximately IID across folds;
both held-out windows show 1st-order autocorrelation roughly twice the
0.10 threshold the trainer requires. The sequential bootstrap (AFML
§4.5.3) the trainer relies on assumes weak label dependence, and that
assumption breaks on these windows.

Two readings, both worth flagging:

1. **Methodology brittleness.** The AR(1) threshold (0.10 median, 0.20
   max) was tuned on the original window. If most other 60-day windows
   on this data are auto-correlated above 0.10, the gate is effectively
   a window-selection filter — the retune is only "valid" on data the
   gate happens to admit, and the original window may be the unusual
   case rather than the representative one.

2. **Time-window artifact.** The original window's IID-friendly label
   structure is itself a feature of that window. The retune's
   measured PF=0.838 is conditioned on a labeling regime the trainer's
   own gate considers exceptional. Even if we relaxed the gate
   (cannot — production has no bypass), there is no a priori reason to
   expect the retune's hyperparameters to remain optimal under the
   different label-dependence structure of W2/W3.

This is **not** evidence the retune is *wrong*; it is evidence the
study did not produce out-of-window confirmation that it is *right*.
The single-window 5-fold purged CV that motivated the retune cannot
distinguish between (a) genuine edge and (b) an edge specific to a
window the trainer's IID-checks happen to admit. We have not closed
that question.

## Recommendation

Do **not** treat `baseline-P4-retune` as out-of-window-validated. Two
defensible follow-ups:

* **Expand the candidate window set.** Sweep ~20 disjoint 60-day
  windows across the full data history; tabulate the AR(1) gate
  pass-rate. If the gate passes on <10% of windows, the original
  window's eligibility is itself the prior, and the retune's PF should
  be quoted conditional on that.
* **Stress the AR(1) gate as a separate experiment.** Re-train (in
  research-only mode, with the gate temporarily relaxed) on a window
  where it fails, evaluate via P3 with full event load, and compare PF
  to W1's. This isolates "the gate gates real things" from "the gate
  is too tight."

W5 invariant held: `ml_model_v2_retune.pkl` was not touched. The
released bundle is unchanged. Per-window training bundles
(`ml_model_v2_walk_w1.pkl` etc.) are the only new pkl artifacts.

## Artifacts

* `backtest_results/walk_forward_validation.json` — full per-window report (incl. per-fold AR(1) values, regime distribution where defined)
* `backtest_results/walk_w1_p3.json` — W1 P3 report (replication; agg PF 0.838)
* `backtest_results/walk_w{2,3}_training.json` — trainer rejection reports for W2/W3
* `tests/test_walk_forward_validation.py` — W1-W5 acceptance gates
