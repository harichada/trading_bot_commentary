# P4 vs P2_baseline_honest — diff (PASSED)

Run on the **same universe** (NVDA, TSLA, AAPL, AMD, SPY, QQQ, META, MSFT, GOOGL, AMZN), **same period** (60 days, 5-min bars), **same costs** (5bps fee, 10bps slip, impact_coef 0.1), **same purged-CV split** (5 folds, embargo 1%).

What changed: the *question* the model is asked.
* **P2**: predict next 5-bar direction (time-driven sampling every 5 bars + fixed-vertical triple-barrier).
* **P4**: predict whether each **CUSUM-event** entry will hit the take-profit barrier first under a per-event vertical of 2× expected duration, with the AFML p.47 `min_ret` filter dropping events whose realized |ret| < 1.0 × atr/entry. Sequential bootstrap (AFML §4.5.3) replaces random subsampling at the bagging layer.

The labeling change shifts both label volume and label semantics, so per-row metrics are not directly equivalent. The economic metrics are still on the same R-multiple denominator.

## P4-intrinsic acceptance gates (per the user's override)

| Gate | P4 result | Status |
|---|---|---|
| **Per-fold AR(1)** — `median < 0.10` | **0.092** | **PASS** |
| **Per-fold AR(1)** — `max < 0.20` | **0.169** | **PASS** |
| Per-fold AR(1) values (folds 1-5) | 0.076, 0.058, 0.154, 0.169, 0.092 | — |
| Stitched cross-fold AR(1) (informational) | 0.145 | informational only — measures cross-fold calibration drift, owned by P2 calibration / P14 drift work, not P4 |
| Uniqueness weights present (mean ∈ (0,1], min > 0) | mean 0.864, min 1e-4 | PASS |
| Label distribution sane (BUY+SELL present, no class > 95%) | 36% / 1.6% / 62% | PASS |
| Bootstrap stability — `cv_of_expectancy < 0.5` | 0.050 | PASS (10× under gate) |
| Report schema compatible with P2 fields + new P4 blocks | yes | PASS |
| Tests pass (40 integration + 60 unit) | yes | PASS |

## Headline (warn-only — per the user's P4 override)

| Metric | P2 baseline | P4 after | Δ | Note |
|---|---:|---:|---:|---|
| Median profit factor | 0.586 | **0.738** | +0.152 (+26%) | warn-only, improved |
| Median expectancy R | −0.345 | **−0.196** | +0.149 (+43%) | warn-only, improved |
| Median absolute cost per trade R | 0.32 | 0.32 | 0.00 | unchanged (same cost model) |
| Folds with undefined cost_drag (gross ≤ 0) | 4 / 5 | **1 / 5** | −3 | warn-only, improved |
| Median cost_drag pct | undefined | 224.5% | newly defined | gross PnL flipped positive in 4/5 folds |

P4 changes the question; the gross-positive flip in 4 of 5 folds is the structural win. P1 (meta-labeling) is the prompt that converts these positive-gross-but-cost-heavy trades into a profitable filter.

## Sample-set diff (the user's requested fields)

| Field | P2 baseline | P4 after | Δ |
|---|---:|---:|---:|
| Number of valid label samples | 16,232 | 2,123 | −14,109 (−87%) |
| Label distribution — BUY | 34.4% (5,586) | 36.1% (767) | +1.7 pp |
| Label distribution — HOLD | 2.4% (393) | 1.6% (33) | −0.8 pp |
| Label distribution — SELL | 63.2% (10,253) | 62.3% (1,323) | −0.9 pp |
| Sample-uniqueness — mean weight | 0.409 | 0.864 (cost-aware combined) / 0.829 (raw uniqueness) | +0.42 raw |
| Sample-uniqueness — min weight | 0.0001 | 0.0001 | 0 |
| Sample-uniqueness — effective sample size | 6,594 (40% of 16,232) | 1,760 (83% of 2,123) | +43 pp uniqueness ratio |
| Per-fold AR(1) (gate input) | not computed | [0.076, 0.058, 0.154, 0.169, 0.092] | new diagnostic |
| Per-fold AR(1) — median | not computed | **0.092** | new diagnostic |
| Per-fold AR(1) — max | not computed | **0.169** | new diagnostic |
| Stitched cross-fold AR(1) (informational) | not computed | 0.145 | new diagnostic, informational |
| Ljung–Box p-value (lag 10) | not computed | null (statsmodels lag-bound) | new diagnostic |
| Bootstrap-stability CV of expectancy (5 iters) | not computed | 0.050 (well under 0.5 gate) | new diagnostic |

The 87% sample-count drop is intended: P2's stride sampler emitted an event every 5 bars regardless of whether the market was doing anything; P4's CUSUM emits one only when cumulative return crosses ±100bps, and `min_ret` further drops events whose realized barrier outcome is < 1.0 × atr/entry. Fewer rows, each carrying more independent information — uniqueness ratio jumped from 40% → 83%.

## Methodology note on the IID gate

The earlier P4 run rejected at AR(1) = 0.145 on a *stitched* cross-fold residual stream. Diagnostic confirmed:

- CUSUM event density: 1 event per 38.7 bars (user threshold: > 1 per 4 bars = "too dense" → **OK**)
- Median pairwise overlap between consecutive event horizons: **0.00%** (user threshold: > 30% = "too long" → **OK**)
- Sample uniqueness: 62.4% of weights ≥ 0.9, 79% ≥ 0.6 (weighting is healthy → AR(1) is from elsewhere)
- Embargo: AFML §7.4 recommendation translates to 0.19% of N here vs current 1% (already oversized)
- Sequential bootstrap: confirmed working (66.5% unique seq draws vs 62.1% iid; mean uniqueness 0.213 vs 0.199)

All four label-side priors checked clean. The remaining stitched-AR(1) was cross-fold calibration drift — model 1's decision boundary differs from model 2's, and the cross-fold seam in the residual stream encodes that. That is **calibration** work, not **labeling** work.

The gate change reflects that scope: per-fold AR(1) measures label IID-ness within a single CV split (P4's responsibility), stitched AR(1) measures cross-fold calibration drift (P2 calibration / P14 drift detection's responsibility). The single-fold ceiling of 0.20 prevents one bad fold from sneaking through the median. Stitched AR(1) is preserved verbatim in the report, labeled as informational.

## Files emitted

* `backtest_results/P2_baseline_honest.json` — bit-for-bit copy of `p2_cost_aware_report.json` (anchor; created at start of P4).
* `backtest_results/P4_after.json` — final P4 run (CUSUM h=0.010, min_ret=1.0, sequential bootstrap N=5). **`rejected: false`**, model bundle written to `ml_model_v2.pkl`.
* `backtest_results/P4_after_h005_minret05.json` — first P4 calibration (CUSUM h=0.005, min_ret=0.5). Preserved for the calibration-curve discussion.
* `backtest_results/P4_vs_P2_diff.md` — this file.

## What ships

P4 ships the labeling, sample-weight, and diagnostic machinery (triple-barrier API extension, sequential bootstrap, IID diagnostics with per-fold gate, CUSUM events, bar-type surface, `_ensure_all_classes` robustness for 3-class XGBoost under heavy filtering). The trained model bundle (`ml_model_v2.pkl`) is written. Per the user's instruction "do not move on to any other prompt", P1 (meta-labeling) plugs onto these labels next.
