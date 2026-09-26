# Strategy Search Batch 1 — Slower, Fewer, Cheaper Trades (2026-09-25)

Universe 539 symbols, 5-minute bars 2021–2026, walk-forward 12m/3m, costs on, regime labels known at entry,
in-root shuffled baseline, **holdout: the final 6 months withheld** (windows end 2026-03-25). Root
`data/strategy_hunt_batch1`; machine report `2026-09-25-strategy-search-batch1-machine.md`.
Code: `strategies/families/swing2.py` (5 families), simulator `overnight` hold kind (commits 328107b, 47e0bba),
holdout window + search accounting (7b4d3f7).

## Verdict

**0 PASS.** 3,467 unconditional rows: 0 PASS / 58 WATCH / 3,409 FAIL. 24 pooled rows: 0 PASS / 4 WATCH.
No family met the holdout-confirmation trigger (a PASS, or ≥ 3 WATCH rows with excess ≥ +0.05 R over the
baseline), so the withheld 6 months were not opened.

The slower families are **not losers** the way the intraday families were, but they are **indistinguishable from
their shuffled baselines**: every pooled row with n ≥ 200 sits within ±0.05 R of the baseline that keeps each
session's open-to-close return and scrambles the intraday path. Whatever these rules earn is the drift of the
stocks they select, not the timing they add.

## Results (pooled rows, n ≥ 200; exp_R per trade after costs; baseline = same rule on shuffled bars)

| family | side | regime | n | exp_R | baseline | excess | t_session | PF | verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| mr_short_term | long | mixed | 3,516 | +0.000 | +0.004 | −0.003 | +0.62 | 1.00 | FAIL |
| mr_short_term | long | risk_off | 817 | +0.001 | −0.001 | +0.003 | −0.06 | 1.01 | FAIL |
| mr_short_term | short | mixed | 5,157 | +0.013 | +0.009 | +0.003 | −1.28 | 1.07 | WATCH |
| mr_short_term | short | risk_on | 312 | +0.132 | +0.086 | +0.046 | −0.53 | 1.90 | WATCH |
| multi_day_drop | long | mixed | 2,166 | +0.012 | +0.018 | −0.006 | +0.59 | 1.04 | FAIL |
| multi_day_drop | long | risk_off | 329 | −0.031 | −0.006 | −0.024 | +0.06 | 0.90 | FAIL |
| overnight_hold | long | mixed | 14,813 | −0.006 | −0.005 | −0.001 | −0.03 | 0.91 | FAIL |
| overnight_hold | long | risk_off | 6,488 | −0.029 | −0.028 | −0.001 | −1.15 | 0.68 | FAIL |
| overnight_hold | long | risk_on | 4,122 | +0.012 | +0.009 | +0.003 | −0.16 | 1.20 | WATCH |
| overnight_hold | short | mixed | 19,177 | −0.020 | −0.019 | −0.001 | −4.60 | 0.71 | FAIL |
| overnight_hold | short | risk_off | 734 | +0.002 | −0.003 | +0.005 | −1.62 | 1.04 | FAIL |
| overnight_hold | short | risk_on | 9,989 | +0.000 | +0.002 | −0.002 | −1.92 | 1.00 | FAIL |
| pead_wide | long | mixed | 512 | −0.062 | −0.054 | −0.008 | −0.90 | 0.88 | FAIL |
| pead_wide | short | mixed | 284 | +0.022 | +0.003 | +0.019 | +0.61 | 1.04 | FAIL |
| rs_rotation | long | mixed | 1,467 | +0.090 | +0.086 | +0.004 | +1.06 | 1.25 | WATCH |

Trade-weighted unconditional exp_R by family and side: overnight_hold −0.009 (long, 25,423 trades) / −0.013 (short,
29,900); mr_short_term +0.001 / +0.020; multi_day_drop +0.007; pead_wide −0.023 / −0.144 (thin); rs_rotation
+0.107 (PF 1.25, 1,682 trades) — and its baseline is +0.086.

## Reading

- **`rs_rotation` is beta, not alpha.** +0.09 R pooled in `mixed` with a +0.086 R baseline: buying the top RS decile
  above its 50-day average earns what those stocks drift; the weekly timing adds +0.004 R. A cheaper way to own that
  drift is to hold it, not to trade it.
- **The overnight premium did not show up here.** 55k overnight trades net ≈ −0.01 R after a spread each way; the
  shuffled baseline is the same, so the sign is the cost, not a missing signal. The family buys the session *after* a
  down day (all inputs are prior-session), which is the conservative, look-ahead-free reading; a close-of-the-down-day
  version would need intraday context at 15:50 and is a Batch 2 candidate.
- **Short-term reversion is flat.** RSI(2) extremes with a 200-day trend filter: +0.00 R long, +0.01 to +0.02 R short,
  baselines the same. The only sub-cell with lift (`short/risk_on` +0.132 vs +0.086, n 312) fails the sample floors.
- **PEAD is unmeasurable in this universe even wide open** (657 long, 345 short trades over five years); a proper test
  needs the full earnings calendar, not the yfinance cache.

## Search accounting

5 families, 54 grid points × sides evaluated, 3,467 unconditional rows scored, 0 PASS, 58 WATCH (all `overnight_hold`,
the family with the most trades per symbol). Together with the 13 earlier families (B1.6: 9,227 rows, 0 PASS) the
search stands at **18 families, ~3,700 symbol-lanes × grid points, 0 PASS** under floors that a shuffled tape cannot
pass.

## What this means for the bot

Across 18 families spanning intraday momentum and reversion, news, breakouts, multi-day reversion, overnight holds,
PEAD and relative-strength rotation, nothing beats a baseline that already owns each day's return. The edge that
survives costs in this data is the **drift of selected stocks over multi-day horizons** (rs_rotation, mr_20d,
overnight long in risk_on), which is a low-turnover portfolio question, not a signal-timing question.

Recommended Batch 2 is therefore not "more entry rules". It is (a) execution variants on the two flat-but-positive
families (limit entries at the open / VWAP for `mr_short_term` short and `overnight_hold` long) to see whether the
cost model is the whole gap, and (b) a low-turnover systematic sleeve tested with the same pipeline at weekly/monthly
rebalance (momentum/quality tilts), where the literature's evidence is strongest and costs are an order of magnitude
smaller. Both are decisions for the operator; neither changes the live posture, which stays shadow-only.

## Artifacts

- `research/reports/strategy_hunt/2026-09-25-strategy-search-batch1-machine.{md,json}` (from the batch root)
- Main-root combined snapshot with all 18 families (holdout 0), scored 22:58 ET: `2026-09-25T2258-strategy-hunt.{md,json}`
  — 12,764 rows, 0 PASS / 195 WATCH; this is what the live gate reads (shadow).
