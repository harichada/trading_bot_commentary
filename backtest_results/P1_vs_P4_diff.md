# P1 vs P4 — Diff Table

Generated: 2026-04-27 (P4 baseline backtest_results/P4_after.json).

## Headline metrics — P1 deployment threshold = 0.85 (median across folds)

| Metric | P4 baseline | P1 after | Delta |
| --- | ---: | ---: | ---: |
| Profit factor | 0.738 | 9.760 | +9.023 |
| Expectancy (R/trade, net of costs) | -0.196 | 0.311 | +0.508 |
| Take-rate | ~0.70 (no meta-filter) | 0.940 | +0.240 |
| Per-fold AR(1) median | 0.092 | 0.025 | -0.067 |
| Per-fold AR(1) max | 0.168 | 0.171 | +0.003 |
| Sample count | 2123 events | 2130 primary fires of 2182 candidate events | — |
| Cost anchor (R/trade) | 0.32 | 0.32 (P2/P4 source-of-truth via ml.costs) | 0.0 |

## Acceptance gates (P1 spec)

| # | Gate | Threshold | P1 actual | Pass |
| ---: | --- | --- | ---: | :---: |
| 1 | Brier OOS | < 0.22 | 0.0290 | ✓ |
| 2 | Calibration max-dev across deciles (n≥30) | ≤ 0.05 | 0.0924 | ✗ |
| 3 | Take-rate ∈ [0.20, 0.40] | inclusive | 0.940 | ✗ |
| 4 | Profit factor (post-filter) | ≥ 1.30 | 9.760 | ✓ |
| 5 | Expectancy (post-filter, after costs) | > 0 | +0.3111 | ✓ |
| 6 | Per-fold AR(1) | median<0.10, max<0.20 | med=0.025, max=0.171 | ✓ |

**Overall: FAIL** — failed gates: calibration, take_rate

## Per-fold thresholds (calibration-tail picks)

| Fold | Threshold | AR(1) | Tail size | Rationale |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0.8500 | +0.105 | 416 | selected threshold=0.85 sortino=-0.387 (best=-0.387) take_ra |
| 2 | 0.8500 | -0.011 | 414 | selected threshold=0.85 sortino=-0.402 (best=-0.402) take_ra |
| 3 | 0.8500 | -0.003 | 415 | selected threshold=0.85 sortino=-0.404 (best=-0.402) take_ra |
| 4 | 0.8500 | -0.025 | 413 | selected threshold=0.85 sortino=-0.202 (best=-0.202) take_ra |
| 5 | 0.5500 | +0.171 | 426 | selected threshold=0.55 sortino=0.092 (best=0.096) take_rate |

Threshold agreement: median=0.8500, min=0.5500, max=0.8500, spread=0.3000, flagged_disagreement=✓ (gate ±0.2).

## Why the gates failed (root cause)

- **Calibration (gate 2)**: max deviation 0.092 vs. ≤0.05 gate, with only 2/10 deciles having n≥30. The OOS calibrated probability mass is **collapsed against the upper end** (label_distribution: {'wins': 2036, 'losses': 94, 'base_win_rate': 0.9559}) — wins dominate every decile, so empirical_p = 0.96+ across the populated bins.
- **Take-rate (gate 3)**: 0.94 vs. [0.20, 0.40] band. Even at threshold 0.85 the meta-classifier still takes 94% of fires. The full sweep shows take-rate barely moves (0.98 at thr=0.40 → 0.94 at thr=0.85) because nearly every primary fire IS a winner in this dataset.

**Underlying mechanism**: the P4 primary (`ml_model_v2.pkl`) was trained on the same 60-day window the P1 collection uses. Running the trained primary back over its own training window produces **in-sample** predictions whose accuracy is wildly inflated (95.6% wins) compared to P4's purged-CV OOS expectancy of −0.196 R/trade. The meta-classifier sees an overwhelmingly easy task (always predict win), the calibrated probability piles up at 1.0, and the take-rate band is structurally unreachable.

## What this means and what to do next

This is a methodological issue, **not** a model-quality issue, and it matches AFML §7.4: meta-labeling on a primary requires the primary's predictions to be **out-of-sample with respect to the primary's training data**. Two principled fixes, in ascending order of effort:

1. Run the P1 trainer on a strictly later time window than the P4 primary's training window. Requires data the current Postgres ingest does not have past 2026-04-27.
2. Re-do purged-CV at the primary level inside P1: for each meta-fold's train rows, re-fit a fresh primary on that fold's train side (using the same XGBoost config from `ml_model_v2.pkl.config`), predict on the meta-test rows. This is the AFML §7 recipe and is **not** a change to `ml_model_v2.pkl` itself — it is an OOS evaluation harness for the primary's *predictions*.

Before either is implemented, the planner-level constraint *"Changes to the primary model (ml_model_v2.pkl)"* needs interpretation. Refitting a fresh primary inside the meta-CV does not modify the persisted bundle, but it does add a primary training step to the P1 pipeline.