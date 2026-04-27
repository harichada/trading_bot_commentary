# P1 Diff Table — Baseline P4 | Leaky P1 | Nested-CV P1

Generated: 2026-04-27.

**Files**: `backtest_results/P4_after.json`, `backtest_results/P1_after.json` (leaky — kept for reference), `backtest_results/P1_nested_after.json` (AFML §7.4 nested CV — honest).

**Primary bundle SHA256 verification (nested run)**: 
  before = `70737ae7a5324f80ba63c038f30846c3793324cc044fd81c73f5f7ee00739784`
  after  = `70737ae7a5324f80ba63c038f30846c3793324cc044fd81c73f5f7ee00739784`
  unchanged = **True** → `ml_model_v2.pkl` on disk is byte-identical pre/post run; the released primary was not modified.

## Headline metrics

| Metric | P4 baseline | P1 leaky (in-sample primary) | P1 nested (OOS primary) |
| --- | ---: | ---: | ---: |
| Profit factor | 0.738 | 9.760093908508729 | 0.600986475440749 |
| Expectancy (R/trade, net) | -0.196 | 0.31114800069358584 | -0.24123056843845647 |
| Take-rate | ~0.70 (no meta) | 0.939906103286385 | 0.04656319290465632 |
| Win rate (post-filter) | n/a | 0.985014985014985 | 0.5714285714285714 |
| Per-fold AR(1) median | 0.092 | 0.025 | 0.076 |
| Per-fold AR(1) max | 0.168 | 0.171 | 0.150 |
| Brier OOS | n/a | 0.0290 | 0.2818 |
| Calibration max-dev | n/a | 0.0924 | 0.3354 |
| Calibration n_qualifying_bins | n/a | 2/10 | 5/10 |
| Sample base win rate | 0.7375625443883556 (PF only) | 0.9559 (95.6% — leak) | 0.5211 (52.1% — honest) |
| n samples | 2123 events | 2130 primary fires | 451 OOS-meta samples (from 2129 candidates) |

## Acceptance gates — leaky vs nested

| # | Gate | Threshold | Leaky | Nested | Honest verdict |
| ---: | --- | --- | ---: | ---: | :---: |
| 1 | Brier OOS | < 0.22 | 0.0290 ✓ | 0.2818 ✗ | **✗** |
| 2 | Calibration max-dev (n≥30) | ≤ 0.05 | 0.0924 ✗ | 0.3354 ✗ | **✗** |
| 3 | Take-rate ∈ [0.20, 0.40] | inclusive | 0.939906103286385 ✗ | 0.04656319290465632 ✗ | **✗** |
| 4 | Profit factor (post-filter) | ≥ 1.30 | 9.760093908508729 ✓ | 0.600986475440749 ✗ | **✗** |
| 5 | Expectancy (post-filter) | > 0 | 0.31114800069358584 ✓ | -0.24123056843845647 ✗ | **✗** |
| 6 | Per-fold AR(1) | med<0.10, max<0.20 | med=0.025 max=0.171 ✓ | med=0.076 max=0.150 ✓ | **✓** |

**Leaky overall**: FAIL (invalid — primary was in-sample on its own training window).
**Nested overall**: FAIL — failed gates: brier, calibration, take_rate, profit_factor, expectancy

## Per-fold detail (nested CV)

| Fold | Primary train | Test | OOS fires | Inner train/test | AR(1) | Tail thr | Inner win-rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1661 | 426 | 360 | 270/90 | +0.076 | 0.8500 | 0.6000 |
| 2 | 1656 | 426 | 355 | 266/89 | +0.083 | 0.8500 | 0.5955 |
| 3 | 1657 | 426 | 367 | 275/92 | -0.011 | 0.8500 | 0.5217 |
| 4 | 1649 | 426 | 353 | 265/88 | -0.034 | 0.8000 | 0.4659 |
| 5 | 1698 | 425 | 369 | 277/92 | +0.150 | 0.5000 | 0.4239 |

Threshold agreement: median=0.8500, min=0.5000, max=0.8500, spread=0.3500, flagged_disagreement=✓ (gate ±0.2).

## Honest reading

1. **Brier 0.282 vs 0.22 gate**: the calibrated meta proba on truly OOS primary fires has roughly a coin-flip Brier — the meta-classifier cannot distinguish winners from losers among the primary's OOS calls.
2. **Calibration deciles populate (5/10 vs leaky's 2/10) but max-dev 0.335 ≫ 0.05**: probabilities span the range now, but the empirical win frequency in each decile is far from the bin midpoint. The model is *miscalibrated*, not *over-confident*.
3. **Take-rate at threshold 0.85 is 0.047 (4.7%)** — below the [0.20, 0.40] band. The full sweep shows no threshold lands in-band with PF≥1.3 AND expectancy>0; PF tops out at 0.77 and expectancy is negative across every threshold tested. The threshold band failure is symptomatic, not the root cause.
4. **PF 0.60 / expectancy −0.24R**: at the median per-fold threshold of 0.85, post-filter trades still lose money on net. The primary's OOS edge (P4 reports −0.196R) is fundamentally negative on this universe; the meta-filter's job is impossible until that root number turns positive.
5. **AR(1) gate passes** (median 0.076, max 0.150). The CV math is clean — the failure is in the primary's edge, not in label IID-ness.

## What this confirms

- The leaky-run gate failures (calibration + take-rate) were a *symptom* of in-sample primary leakage; the nested run shows the *disease* — the primary itself does not have positive OOS edge on this universe at this configuration. Per-fold OOS-meta inner win-rate of ~52% is near coin-flip, exactly consistent with P4's −0.196R/trade expectancy after costs.
- The released `ml_model_v2.pkl` is byte-identical pre and post run (SHA256 verified). The harness refit primaries are evaluation artefacts only, never persisted.
- AR(1) passes cleanly in both runs — confirms P4's labeling pipeline is sound; the issue is model edge, not label IID-ness.

## Decision branch (yours, not mine)

- **Skip P1, move to P3 (regime conditioning)** — meta-labeling adds no value when the underlying primary edge is negative; regime gating may either unlock edge in select regimes or confirm there is none on this universe.
- **Revisit P4's primary edge** — investigate whether the universe, feature set, or label horizons need to change. P1 is downstream of P4 and cannot create edge that does not exist upstream.
- **Do NOT relax P1 gates** — they performed exactly as designed: they surfaced the leakage in the first run and the absence of edge in the second.