# Strategy Search Batch 2 — Selection Control, Limit Execution, Factor Sleeve (2026-09-26)

Universe 539 symbols, 2021–2026, walk-forward 12m/3m, costs on, entry-time regime labels, **corrected split
adjustment** (Task E: the detector no longer compounds false splits; 35 symbols had been distorted, which had
inflated rank-based families — `rs_rotation` fell from +0.107 R to +0.03 R once fixed), **holdout: final 6 months
withheld**. Two controls per pooled row: the *path shuffle* (permutes each session's intrabar path; tests timing)
and the new *selection* control (replays the real trade schedule on seeded random peer symbols with the same
relative stop/target, hold and entry style; tests stock selection). Two pinned execution arms: `market` fills
(spread + slippage) and `limit_open` (resting limit 10 bp inside the open, filled only if the bar trades through,
commission only — an upper bound). Roots `data/strategy_hunt_batch2` (market) and `_batch2L` (limit); machine
reports `2026-09-26-strategy-search-batch2-{market,limit}-machine.{md,json}`.

## Verdict

**0 PASS in both arms** (market: 3,970 rows, 0 / 58 WATCH; limit: 3,779 rows, 0 / 43 WATCH; pooled 0 PASS).
No family met the holdout-confirmation trigger, so the withheld 6 months stay closed.

Two questions answered:

1. **Execution is not the gap.** Switching every slow family to maker limit entries changes pooled expectancy by
   +0.00 to +0.02 R per trade while filling only 60–75 % of the signals. Flat stays flat.
2. **Selection skill is small and unproven.** Against random-peer selection, the best rows are
   `mr_20d long / risk_off` (+0.237 R vs +0.108, excess +0.129, n 302, t_session 1.8),
   `mom_12_1 long / mixed` (+0.060 vs +0.017, excess +0.043, n 1,302, t_session 1.4),
   `mr_20d long / mixed` (+0.055 vs +0.013, excess +0.042, n 641) and `rs_rotation long` (+0.036 vs +0.008,
   excess +0.028). All positive, none significant, none near the floors (PF ≥ 1.2, t_session ≥ 3, recovery ≥ 3).
   Momentum and mean-reversion **shorts** are worse than random selection (−0.07 to −0.09 R excess).

## Results — market arm, pooled rows with n ≥ 200 (exp_R per trade after costs; control = selection)

| family | side | regime | n | exp_R | control | excess | t_session | replayed | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| low_vol_trend | long | mixed | 1,487 | −0.076 | −0.080 | +0.004 | +0.23 | 0.95 | FAIL |
| mom_12_1 | long | mixed | 1,302 | +0.060 | +0.017 | +0.043 | +1.37 | 0.94 | WATCH |
| mom_12_1 | short | mixed | 1,366 | −0.091 | −0.005 | −0.086 | −2.02 | 0.95 | FAIL |
| mr_20d | long | mixed | 641 | +0.055 | +0.013 | +0.042 | −0.66 | 0.95 | WATCH |
| mr_20d | long | risk_off | 302 | +0.237 | +0.108 | +0.129 | +1.80 | 0.89 | WATCH |
| mr_20d | short | mixed | 1,051 | −0.102 | −0.030 | −0.073 | −1.49 | 0.93 | FAIL |
| mr_short_term | long | mixed | 3,490 | −0.005 | +0.034 | −0.039 | +0.09 | 0.93 | FAIL |
| mr_short_term | long | risk_off | 812 | +0.001 | −0.038 | +0.039 | +0.01 | 0.93 | FAIL |
| mr_short_term | short | mixed | 5,181 | +0.009 | −0.008 | +0.016 | −1.42 | 0.96 | FAIL |
| mr_short_term | short | risk_on | 324 | +0.107 | +0.068 | +0.040 | −1.02 | 0.97 | WATCH |
| overnight_hold | long | mixed | 14,807 | −0.006 | −0.008 | +0.001 | −0.03 | 0.96 | FAIL |
| overnight_hold | long | risk_off | 6,485 | −0.032 | −0.030 | −0.002 | −1.46 | 0.97 | FAIL |
| overnight_hold | long | risk_on | 4,122 | +0.012 | +0.005 | +0.007 | −0.16 | 0.96 | WATCH |
| overnight_hold | short | mixed | 19,180 | −0.020 | −0.030 | +0.009 | −4.54 | 0.95 | FAIL |
| overnight_hold | short | risk_off | 735 | +0.002 | +0.022 | −0.020 | −1.63 | 0.96 | FAIL |
| overnight_hold | short | risk_on | 9,989 | +0.000 | −0.003 | +0.004 | −1.92 | 0.97 | FAIL |
| rs_rotation | long | mixed | 1,470 | +0.036 | +0.008 | +0.028 | +0.61 | 0.95 | WATCH |

## Limit-entry arm — what changed

| family / side / regime | market exp_R (n) | limit exp_R (n) | fill share |
|---|---:|---:|---:|
| mom_12_1 long mixed | +0.060 (1,302) | +0.063 (1,047) | 80 % |
| mr_20d long mixed / risk_off | +0.055 (641) / +0.237 (302) | +0.055 (530) / +0.215 (248) | 83 % |
| mr_short_term long mixed / short mixed | −0.005 (3,490) / +0.009 (5,181) | +0.004 (2,582) / +0.011 (3,836) | 74 % |
| overnight_hold long mixed / short mixed | −0.006 (14,807) / −0.020 (19,180) | +0.001 (8,894) / −0.009 (10,908) | 60 % |
| rs_rotation long mixed | +0.036 (1,470) | +0.021 (1,176) | 80 % |

Maker fills are worth roughly one to two cents of R per trade and cost a quarter to a third of the signals. Because
the limit only fills on bars that trade into it, this is the *favourable* bound; real queue position would be worse.

## Search accounting

Market arm: 6 families, 62 grid points × sides (after pinning), 3,970 rows scored, 0 PASS, 58 WATCH (all
`overnight_hold`). Limit arm: 62 grid points × sides, 3,779 rows, 0 PASS, 43 WATCH. Cumulative across Batches 1–2
and the 13 original families: **20 families, ≈4,000 symbol-lanes × grid points, 0 PASS.**

## Caveats

- **`baseline_stale` on every market-arm pooled row** is an artifact, not a finding: one real multiday trade entered
  before the holdout cut exits 2026-04-09, thirteen days after the selection control's last exit (03-27), and the
  guard compares last *exits*. The controls share the entry schedule by construction; the guard should compare last
  *entries* for the selection kind (fix queued). No row was otherwise near PASS.
- Selection-control replays cover 89–97 % of each real group (`replayed` column); rows under 90 % carry
  `baseline_underreplayed` and cannot PASS.
- `mom_12_1` needs 252 sessions of history, so its sample starts in 2022 and is thin; the two factor families share
  entry dates, so their controls are correlated.
- Live-tape R for the playbook is still derived from the closing stop until the ledger-integrity migration is applied.

## What this means

After twenty families, two execution styles, three controls and a corrected data pipeline, the only thing that
survives costs in this universe is a small, statistically weak selection effect in slow long mean-reversion and
12-1 momentum — on the order of +0.04 to +0.13 R per trade on a few hundred to a thousand trades, with session-level
t-statistics below 2. That is consistent with the academic literature (factor premia exist but are small and slow)
and inconsistent with the bot's design (intraday signals, many trades, market fills).

Recommendation: stop searching for entry rules. If the account is to be run systematically, the candidate is a
low-turnover long-only sleeve (12-1 momentum decile and/or 20-day mean reversion in risk-off, monthly, wide stops)
sized so that a +0.05 R average over ~20 trades a month is worth having — which is a portfolio decision about
capital and risk budget, not a strategy-search decision. The desk and playbook keep running in shadow to collect
live evidence either way.
