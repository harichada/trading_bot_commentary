# Playbook Desk B1.5 — the B1 run re-scored under hardened pooled rules

**Run:** 2026-09-25, branch `feature/newsbus-autonomy-decision-card`, code at `2c03596`.
Machine report: `research/reports/strategy_hunt/2026-09-25T1433-strategy-hunt.md` (+ `.json` companion).
Prior run this supersedes: `research/reports/strategy_hunt/2026-09-25-strategy-hunt-b1.md`.

## Headline

**0 PASS rows anywhere, and the hardened floors say *why* — which is the point of B1.5.**

| | B1 (2026-09-25 13:28, FinBERT news, old floors) | B1.5 (2026-09-25 14:33, VADER news, hardened floors) |
|---|---|---|
| Unconditional rows | 9,505 · PASS 0 / WATCH 127 / FAIL 9,378 | **9,231 · PASS 0 / WATCH 127 / FAIL 9,104** |
| Symbol-regime rows | 1,390 over 396 symbols · PASS 0 / WATCH 159 | **1,390 over 396 symbols · PASS 0 / WATCH 159** |
| Universe-pooled rows | 69 · PASS 0 / WATCH 15 / FAIL 54 | **69 · PASS 0 / WATCH 14 / FAIL 55** |
| Pooled rows with a baseline | — (control lived in a separate root, 50 symbols) | **69 of 69** (in-root `trades_shuffle/`, 539 symbols) |
| Mean unconditional exp_R | −0.124 | **−0.126** |
| Out-of-sample trades | 340,211 | **335,499** (real) · 479,354 (shuffled baseline) |

B1's single closest row — `dt_momentum short/risk_off`, which failed *only* `max_dd_r` — does not
survive. Dropping the drawdown cap for `min_recovery` clears it (recovery 3.60 ≥ 3.0, so the 639 R
drawdown really was a scale artefact), but the two new tests it had never faced both fail it: its
t-statistic collapses from **+3.85 to +0.56** once trades are clustered by session, and **83 % of its
expectancy is reproduced by shuffled bars** (+0.172 real vs +0.142 baseline → excess **+0.030**,
under the +0.05 floor). The B1 caveat was right on both counts.

The best pooled row in this run is now `gap_play short/risk_off` — the only one of 69 that fails a
single floor. It clears excess (+0.073), session-clustered t (+3.52) and recovery (4.20), and dies on
`min_decay` alone: its last four quarters average **−0.336 R** against a +0.181 R full-sample
expectancy (decay −1.86). Whatever that lane was, it is not working now.

## What changed between B1 and B1.5

Three scoring changes plus one data change, all upstream of this run:

1. **`FitMetrics` gained `t_stat_session` and `recovery`** (`7cf467f`). `t_stat_session` clusters by
   `entry_ts.normalize()` — the independent unit, since hundreds of symbols trade the same session —
   and `recovery = net_R / max_dd_R` replaces the flat `max_dd_r ≤ 60 R` pooled cap, which was not
   scale-invariant.
2. **The shuffled control became an in-root baseline variant** (`5258242`). `run --shuffle-seed 7`
   writes `trades_shuffle/` beside `trades/` over the *full* 539-symbol universe; `score` pools both
   tapes in one pass, so every pooled row carries `baseline_exp_r`, `baseline_n` and
   `excess_r = exp_r − baseline_exp_r`, with `min_excess_r 0.05` required for PASS and
   `no_baseline` / `baseline_mismatch` / `baseline_stale` when the control is missing, partial or old
   (`2c03596`).
3. **Machine reports are run-stamped** `<date>T<HHMM>-strategy-hunt.md`, so a second run in a day no
   longer overwrites the first one's evidence.
4. **The research `sentiment` column is back on VADER** (parity with the live news loop, which never
   moved). `sentiment_finbert` is kept in the cache for the parked FinBERT threshold study.

## Steps

Env: `set -a; . ./.env; set +a`; `POSTGRES_DSN` from `backtest.engine.DEFAULT_DSN` (never printed).
Steps 1–4 were run by the controller before this task; 5–8 are this task's. All logs under
`data/strategy_hunt/`.

| # | Step | Command | Wall-clock | Outcome |
|---|---|---|---|---|
| 1 | news rescore | `SENTIMENT_BACKEND=vader … news --rescore` | ~5 min | 0 errors; `sentiment` == `sentiment_vader`, `sentiment_finbert` retained (`news_rescore_vader.log`) |
| 2 | context | `… context` | 212 s | universe context rebuilt with VADER `ctx_news_sent_24h` (`context_b15.log`) |
| 3 | run (real) | `… run --workers 12 --force` | 298 s | 539/539 done, 0 errors, 335,499 trades, 13 families (`full_run_b15.log`) |
| 4 | run (baseline) | `… run --shuffle-seed 7 --workers 12 --force` | 362 s | 539/539 done, 0 errors, 479,354 trades → `trades_shuffle/` (`baseline_run_b15.log`) |
| 5 | control cost 2× | `… run --cost-mult 2.0 --root data/strategy_hunt_cost2 --symbols @…/sample50.txt --workers 12 --force` | 33 s | 50/50 done, 0 errors (`control_cost2_run_b15.log`) |
| 6 | control cost 2× score | `… score --root data/strategy_hunt_cost2 --no-coldstart` | 5 s | 873 rows, PASS 0 / WATCH 3, mean exp_R −0.256 (`control_cost2_score_b15.log`) |
| 7 | score | `… score --postgres` | 75 s | 9,231 unconditional + 1,390 symbol-regime + 69 pooled + **69 baseline** rows; snapshot + Parquet + Postgres upsert + machine report (`score_b15.log`) |
| 8 | report / promote | `… report`, `… promote` | 2 s / 1 s | run-stamped report rewritten; 13-family promotion table (`report_b15.log`, `promote_b15.log`) |

Steps 5–8 total 1 m 57 s (`data/strategy_hunt/b15_pipeline.log`). Nothing failed, so no step was
skipped and no fix was needed. The 0.5× cost control and the separate-root shuffle control from B1
were deliberately not re-run: the in-root baseline supersedes the shuffle control, and the B1.5 plan
keeps only the 2× cost control. Their stale B1 snapshots were removed from
`data/strategy_hunt/fit/` rather than left for the report to quote as if they were this run's.

**The baseline guard passed.** `score` reported 69 pooled rows and 69 baseline rows with no
`baseline_mismatch` / `baseline_stale` / `no_baseline` marker on any row: the control covers all 539
real symbols (0 missing) and its last exit is the same session as the real tape's.

## Pooled rows vs the shuffled baseline (the headline table)

`excess = exp_R − baseline exp_R`, where the baseline is the same `(family, side, regime)` group on
5-minute returns permuted inside each session (each session's open-to-close return preserved).
Failed floors are abbreviated to the metric name.

### A. every pooled row with n >= 500, sorted by excess

| family | side | regime | n | sessions | exp_R | baseline exp_R | excess | t | t_session | recovery | PF | WR | stab | decay | verdict | failed floors |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| orb_inplay | long | mixed | 2,494 | 558 | -0.310 | -1.869 | **+1.558** | -5.14 | -4.36 | 0.00 | 0.77 | 0.23 | 0.11 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb |
| orb_inplay | long | risk_off | 678 | 164 | -0.473 | -2.023 | **+1.550** | -4.64 | -3.82 | 0.00 | 0.65 | 0.20 | 0.26 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb |
| orb_inplay | short | risk_off | 876 | 177 | +0.200 | -1.297 | **+1.497** | +1.67 | +1.23 | 1.89 | 1.16 | 0.32 | 0.58 | +0.92 | WATCH | pf, wr, recovery, stability, t, t_session |
| orb_inplay | long | risk_on | 1,912 | 419 | +0.128 | -1.316 | **+1.444** | +1.84 | +1.02 | 1.28 | 1.11 | 0.29 | 0.58 | -0.00 | WATCH | pf, wr, recovery, stability, decay, t, t_session |
| orb_inplay | short | mixed | 2,488 | 561 | -0.172 | -1.496 | **+1.323** | -2.53 | -1.00 | 0.00 | 0.87 | 0.26 | 0.26 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb |
| orb_inplay | short | risk_on | 1,627 | 409 | -0.539 | -1.475 | **+0.936** | -7.88 | -6.50 | 0.00 | 0.62 | 0.21 | 0.05 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb |
| orb | short | mixed | 1,153 | 401 | -0.221 | -0.377 | **+0.156** | -4.95 | -5.32 | 0.00 | 0.74 | 0.39 | 0.21 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb |
| dt_momentum | long | mixed | 35,247 | 579 | -0.174 | -0.285 | **+0.111** | -26.93 | -12.60 | 0.00 | 0.72 | 0.38 | 0.00 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb |
| gap_play | short | risk_on | 7,560 | 422 | -0.160 | -0.260 | **+0.100** | -10.96 | -8.27 | 0.00 | 0.76 | 0.35 | 0.11 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb |
| gap_play | long | risk_off | 3,861 | 174 | -0.233 | -0.324 | **+0.091** | -11.69 | -6.78 | 0.00 | 0.66 | 0.34 | 0.21 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb |
| gap_play | short | risk_off | 5,292 | 177 | +0.181 | +0.107 | **+0.073** | +9.87 | +3.52 | 4.20 | 1.35 | 0.47 | 0.63 | -1.86 | WATCH | decay |
| orb | short | risk_on | 658 | 283 | -0.478 | -0.541 | **+0.062** | -8.59 | -8.25 | 0.00 | 0.49 | 0.29 | 0.11 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb |
| dt_momentum | short | mixed | 40,690 | 580 | -0.136 | -0.189 | **+0.052** | -22.33 | -11.23 | 0.00 | 0.78 | 0.39 | 0.05 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb |
| gap_play | long | mixed | 8,614 | 570 | -0.120 | -0.168 | **+0.048** | -8.46 | -7.07 | 0.00 | 0.82 | 0.36 | 0.16 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb, excess |
| dt_momentum | long | risk_on | 24,229 | 426 | +0.048 | +0.003 | **+0.045** | +5.96 | -0.07 | 2.11 | 1.09 | 0.46 | 0.58 | +0.08 | WATCH | pf, exp_r, recovery, stability, decay, t_session, excess |
| mean_rev_intraday | long | risk_on | 2,051 | 377 | +0.267 | +0.224 | **+0.043** | +7.93 | +3.37 | 6.88 | 1.45 | 0.49 | 0.79 | -1.12 | WATCH | decay, excess |
| dt_momentum | long | risk_off | 12,004 | 180 | -0.483 | -0.525 | **+0.042** | -49.75 | -19.93 | 0.00 | 0.37 | 0.25 | 0.00 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb, excess |
| orb | short | risk_off | 520 | 141 | -0.068 | -0.101 | **+0.033** | -1.01 | -3.19 | 0.00 | 0.91 | 0.41 | 0.41 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb, excess |
| dt_momentum | short | risk_off | 13,370 | 179 | +0.172 | +0.142 | **+0.030** | +3.85 | +0.56 | 3.60 | 1.33 | 0.52 | 0.84 | +0.71 | WATCH | t_session, excess |
| gap_play | long | risk_on | 8,771 | 423 | +0.060 | +0.035 | **+0.025** | +4.20 | +1.63 | 1.19 | 1.10 | 0.43 | 0.58 | +1.47 | WATCH | pf, wr, recovery, stability, t_session, excess |
| dt_momentum | short | risk_on | 30,551 | 426 | -0.457 | -0.482 | **+0.025** | -19.47 | -17.77 | 0.00 | 0.41 | 0.28 | 0.05 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb, excess |
| gap_play | short | mixed | 9,017 | 571 | -0.060 | -0.083 | **+0.022** | -4.30 | -1.68 | 0.00 | 0.91 | 0.39 | 0.32 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb, excess |
| news_catalyst | long | risk_on | 556 | 255 | +0.118 | +0.098 | **+0.020** | +2.34 | +1.13 | 4.19 | 1.25 | 0.50 | 0.79 | -0.19 | WATCH | decay, t, t_session, excess |
| mr_20d | short | mixed | 626 | 282 | -0.150 | -0.158 | **+0.008** | -3.11 | -1.97 | 0.00 | 0.73 | 0.40 | 0.33 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb, excess |
| vwap_reversion | short | risk_on | 26,722 | 426 | -0.376 | -0.381 | **+0.005** | -54.51 | -17.32 | 0.00 | 0.46 | 0.32 | 0.00 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb, excess |
| mr_20d | long | mixed | 512 | 195 | +0.108 | +0.118 | **-0.010** | +2.30 | -0.29 | 2.25 | 1.29 | 0.50 | 0.47 | +2.20 | WATCH | recovery, stability, t, t_session, excess |
| orb | long | mixed | 1,074 | 389 | -0.333 | -0.313 | **-0.020** | -7.31 | -8.61 | 0.00 | 0.63 | 0.34 | 0.17 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb, excess |
| vwap_reversion | long | risk_on | 10,594 | 416 | +0.073 | +0.098 | **-0.025** | +5.71 | +2.83 | 3.68 | 1.14 | 0.48 | 0.58 | -1.15 | WATCH | pf, stability, decay, t_session, excess |
| vwap_reversion | short | mixed | 21,229 | 573 | -0.149 | -0.117 | **-0.032** | -16.91 | -5.93 | 0.00 | 0.76 | 0.40 | 0.11 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb, excess |
| orb | long | risk_on | 981 | 313 | -0.165 | -0.129 | **-0.036** | -3.47 | -2.69 | 0.00 | 0.79 | 0.38 | 0.33 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb, excess |
| vwap_reversion | long | mixed | 27,896 | 577 | -0.196 | -0.147 | **-0.050** | -27.04 | -8.06 | 0.00 | 0.69 | 0.38 | 0.00 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb, excess |
| vwap_reversion | long | risk_off | 14,675 | 179 | -0.440 | -0.386 | **-0.054** | -50.06 | -11.00 | 0.00 | 0.40 | 0.29 | 0.05 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb, excess |
| mean_rev_intraday | long | mixed | 5,319 | 547 | -0.129 | -0.037 | **-0.092** | -6.00 | -4.65 | 0.00 | 0.83 | 0.37 | 0.26 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb, excess |
| vwap_reversion | short | risk_off | 3,385 | 175 | +0.076 | +0.200 | **-0.123** | +3.32 | +1.44 | 1.26 | 1.14 | 0.48 | 0.58 | -0.22 | WATCH | pf, recovery, stability, decay, t_session, excess |
| news_catalyst | long | mixed | 595 | 302 | -0.160 | -0.004 | **-0.156** | -3.39 | -2.67 | 0.00 | 0.73 | 0.36 | 0.21 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb, excess |
| mean_rev_intraday | long | risk_off | 2,724 | 177 | -0.590 | -0.429 | **-0.160** | -23.19 | -8.79 | 0.00 | 0.38 | 0.21 | 0.05 | +0.00 | FAIL | pf, wr, exp_r, recovery, stability, decay, t, t_session, exp_r_lb, excess |

**Reading the table.** Across all 69 pooled rows, 43 have positive excess and 17 clear the +0.05
floor — but 12 of those 17 are rows with *negative* expectancy whose shuffled twin is merely more
negative still. Restricted to the 36 rows with n ≥ 500 shown above, 13 clear the floor and 10 of
those are negative-expectancy.
That is the excess test behaving as designed: `min_excess_r` is a necessary condition layered on top
of `min_exp_r`, `min_pf`, `min_wr` and the rest, not a substitute for them. The six `orb_inplay` rows
make the point loudly — permuting intraday returns destroys the opening range the family trades, so
its baseline collapses to −1.3 … −2.0 R and every `orb_inplay` excess lands above +0.9 R. An excess
that large means "the rule depends on intraday structure", not "the rule makes money".

Only five rows are simultaneously positive-expectancy *and* clear of the excess floor, and only
three of them have n ≥ 500: `orb_inplay short/risk_off` (+0.200, excess +1.497), `orb_inplay
long/risk_on` (+0.128, excess +1.444) and `gap_play short/risk_off` (+0.181, excess +0.073). The two
`orb_inplay` rows fail on PF, WR, recovery, stability and both t-statistics; `gap_play short/risk_off`
fails only `min_decay`. The other two are tiny-n FAIL rows that no floor set would admit:
`pead_smallcap short/mixed` (n 39, +0.234, excess +0.160) and `vol_breakout long/risk_off` (n 26,
+0.412, excess +0.149).

## Do B1's WATCH rows survive the excess test?

B1 produced 15 pooled WATCH rows. **14 of the 15 are still WATCH in B1.5**; the one that dropped out
is `news_catalyst short/risk_off`, which fell below the WATCH `min_n` of 200 when the sentiment
column went back to VADER (n 688 → 135). Of the 14 survivors:

| | count | rows |
|---|---|---|
| positive excess over the shuffled baseline | **10 of 14** | orb_inplay short/risk_off +1.497, orb_inplay long/risk_on +1.444, gap_play short/risk_off +0.073, dt_momentum long/risk_on +0.045, mean_rev_intraday long/risk_on +0.043, dt_momentum short/risk_off +0.030, gap_play long/risk_on +0.025, news_catalyst long/risk_on +0.020, post_news_drift short/mixed +0.013, mr_20d long/risk_on +0.001 |
| negative excess (worse than shuffled bars) | **4 of 14** | vwap_reversion short/risk_off −0.123, vwap_reversion long/risk_on −0.025, pullback_trend long/risk_on −0.015, mr_20d long/mixed −0.010 |
| clear the `min_excess_r` +0.05 floor | **3 of 14** | orb_inplay short/risk_off, orb_inplay long/risk_on, gap_play short/risk_off |

So the excess test alone removes 11 of B1's 15 pooled WATCH candidates from any PASS path, and four
of them (two of them `vwap_reversion` lanes B1 listed among its best) are literally worse than noise
with the same drift.

| family | side | regime | n | sessions | exp_R | baseline exp_R | excess | t | t_session | recovery | PF | WR | stab | decay | verdict | failed floors |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| mr_20d | long | mixed | 512 | 195 | +0.108 | +0.118 | **-0.010** | +2.30 | -0.29 | 2.25 | 1.29 | 0.50 | 0.47 | +2.20 | WATCH | recovery, stability, t, t_session, excess |
| post_news_drift | short | mixed | 222 | 163 | +0.022 | +0.009 | **+0.013** | +0.41 | +0.83 | 0.27 | 1.07 | 0.53 | 0.53 | +10.51 | WATCH | n, pf, exp_r, recovery, stability, t, t_session, exp_r_lb, excess |
| orb_inplay | short | risk_off | 876 | 177 | +0.200 | -1.297 | **+1.497** | +1.67 | +1.23 | 1.89 | 1.16 | 0.32 | 0.58 | +0.92 | WATCH | pf, wr, recovery, stability, t, t_session |
| gap_play | short | risk_off | 5,292 | 177 | +0.181 | +0.107 | **+0.073** | +9.87 | +3.52 | 4.20 | 1.35 | 0.47 | 0.63 | -1.86 | WATCH | decay |
| dt_momentum | short | risk_off | 13,370 | 179 | +0.172 | +0.142 | **+0.030** | +3.85 | +0.56 | 3.60 | 1.33 | 0.52 | 0.84 | +0.71 | WATCH | t_session, excess |
| vwap_reversion | short | risk_off | 3,385 | 175 | +0.076 | +0.200 | **-0.123** | +3.32 | +1.44 | 1.26 | 1.14 | 0.48 | 0.58 | -0.22 | WATCH | pf, recovery, stability, decay, t_session, excess |
| mean_rev_intraday | long | risk_on | 2,051 | 377 | +0.267 | +0.224 | **+0.043** | +7.93 | +3.37 | 6.88 | 1.45 | 0.49 | 0.79 | -1.12 | WATCH | decay, excess |
| mr_20d | long | risk_on | 325 | 155 | +0.179 | +0.178 | **+0.001** | +3.22 | +2.14 | 4.91 | 1.53 | 0.58 | 0.72 | +0.52 | WATCH | n, t_session, excess |
| orb_inplay | long | risk_on | 1,912 | 419 | +0.128 | -1.316 | **+1.444** | +1.84 | +1.02 | 1.28 | 1.11 | 0.29 | 0.58 | -0.00 | WATCH | pf, wr, recovery, stability, decay, t, t_session |
| pullback_trend | long | risk_on | 328 | 193 | +0.119 | +0.134 | **-0.015** | +2.17 | +2.79 | 1.82 | 1.32 | 0.54 | 0.82 | +2.02 | WATCH | n, recovery, t, t_session, excess |
| news_catalyst | long | risk_on | 556 | 255 | +0.118 | +0.098 | **+0.020** | +2.34 | +1.13 | 4.19 | 1.25 | 0.50 | 0.79 | -0.19 | WATCH | decay, t, t_session, excess |
| vwap_reversion | long | risk_on | 10,594 | 416 | +0.073 | +0.098 | **-0.025** | +5.71 | +2.83 | 3.68 | 1.14 | 0.48 | 0.58 | -1.15 | WATCH | pf, stability, decay, t_session, excess |
| gap_play | long | risk_on | 8,771 | 423 | +0.060 | +0.035 | **+0.025** | +4.20 | +1.63 | 1.19 | 1.10 | 0.43 | 0.58 | +1.47 | WATCH | pf, wr, recovery, stability, t_session, excess |
| dt_momentum | long | risk_on | 24,229 | 426 | +0.048 | +0.003 | **+0.045** | +5.96 | -0.07 | 2.11 | 1.09 | 0.46 | 0.58 | +0.08 | WATCH | pf, exp_r, recovery, stability, decay, t_session, excess |

## `dt_momentum short / risk_off` — the row B1 came closest on

| metric | B1 (old floors) | B1.5 (hardened) | floor | verdict |
|---|---|---|---|---|
| n / sessions | 13,370 / 179 | 13,370 / 179 | ≥ 500 / ≥ 60 | ok |
| exp_R | +0.172 | +0.172 | ≥ +0.05 | ok |
| PF / WR | 1.33 / 0.52 | 1.33 / 0.52 | ≥ 1.2 / ≥ 0.45 | ok |
| stability / decay | 0.84 / 0.71 | 0.84 / 0.71 | ≥ 0.6 / ≥ 0.5 | ok |
| max_dd_R | 639 | (floor removed) | ≤ 60 R | **the only B1 failure** |
| recovery (net R / max DD R) | — | **3.60** | ≥ 3.0 | ok — the drawdown was a scale artefact |
| t_stat (iid) | +3.85 | +3.85 | ≥ 3.0 | ok |
| **t_stat_session (clustered)** | — | **+0.56** | ≥ 3.0 | **FAIL** |
| baseline exp_R | — (separate-root control, 50 symbols) | **+0.142** on 539 symbols | — | — |
| **excess_R** | — | **+0.030** | ≥ +0.05 | **FAIL** |

Both new tests fail it, and they fail it for the same underlying reason: `risk_off` sessions are few
(179 of ~580) and highly correlated across symbols, so 13,370 trades are nowhere near 13,370
independent observations, and most of what the lane earns in them is the regime's own downward drift,
which the shuffled tape reproduces (+0.142 of the +0.172). Its quarterly profile is also carried by a
single window — 2025 Q3 averages −3.35 R on 188 trades while 2025 Q1/Q2/Q4 average +0.31/+0.36/+0.43 —
which `stability` 0.84 does not show.

`failed_floors` for the row is exactly `["min_t_stat_session", "min_excess_r"]`.

## FinBERT (B1) → VADER (B1.5): the news families

Only `news_catalyst` reads the sentiment column; `post_news_drift` (969 trades) and `pead_smallcap`
(283 trades) are unchanged to the trade.

| | B1 (FinBERT) | B1.5 (VADER) |
|---|---|---|
| `news_catalyst` trades | 6,639 | **1,927** |
| pooled long/risk_on | +0.090 on n 1,367 · WATCH | **+0.118 on n 556 · WATCH** (excess +0.020) |
| pooled short/risk_off | +0.066 on n 688 · WATCH | **+0.105 on n 135 · FAIL** (below WATCH `min_n` 200; excess −0.176) |
| pooled by side, all regimes | long −0.077 · short −0.073 | **long −0.064 (n 1,326) · short −0.006 (n 601)** |
| family mean exp_R | −0.048 | **−0.046** |
| unconditional `news_catalyst` rows | 881 | **607** (long 351 + short 256) |

FinBERT fired the family 3.4× as often at materially the same (slightly worse) expectancy, which is
what B1 flagged: the families' `min_abs_sentiment` grid is tuned to VADER's compound score and
FinBERT's more polarised distribution walks straight through it. Going back to VADER restores
research/live parity — the live news loop never left VADER — and concentrates the family into a
smaller, mildly better set of trades, at the cost of the one pooled row that had enough sample to
WATCH on the short side. The parked follow-up stands: a FinBERT variant needs its own threshold grid
against `sentiment_finbert`, not the VADER-calibrated one.

Total unconditional rows fell 9,505 → 9,231 for the same reason (fewer `news_catalyst` parameter sets
clear the row floor), with the WATCH count unchanged at 127.

## Controls

| Control | Scope | Unconditional rows | PASS | WATCH | mean exp_R | Pooled PASS | Pooled WATCH |
|---|---|---|---|---|---|---|---|
| **in-root shuffled baseline** (seed 7) | **539 symbols, full universe** | — (scored as baseline rows, not lanes) | — | — | — | — | — |
| cost 2× | 50-symbol sample, separate root | 873 | **0** | 3 | **−0.256** | **0** | 1 |
| real run | 539 symbols | 9,231 | **0** | 127 | −0.126 | **0** | 14 |

- **Cost sensitivity behaves.** Doubling costs on the 50-symbol sample drops mean exp_R to −0.256 and
  leaves 3 unconditional WATCH rows and one pooled WATCH (`gap_play short/risk_off` again, n 527,
  exp_R +0.168). No PASS at any cost level.
- **The cost-2× root has no baseline of its own**, and its pooled rows say so: every one carries
  `no_baseline` in `failed_floors`, so the control cannot mint a pooled PASS either. That is the
  guard from task 10 doing its job on a real root.
- **The shuffle control is no longer a separate root.** B1's shuffle control ran on 50 symbols and
  produced a pooled PASS on structure-destroyed bars — the finding that motivated B1.5. The in-root
  baseline replaces the binary "did the control PASS?" question with a per-row excess, over the full
  universe, and the B1 embarrassment cannot recur: the baseline's rows are stamped
  `source = "baseline"`, stored under their own snapshot key, and never read as lanes by the live gate.
- B1's 0.5× cost control was not re-run (dropped by the B1.5 plan); its stale B1 snapshot was removed
  so the machine report cannot quote FinBERT-era numbers as this run's.

Caveats carried forward unchanged: survivorship (universe selected by today's data availability),
costs estimated from bars, multiple-comparisons pressure from 13 families × up to 27 parameter sets ×
539 symbols, and research using closed bars where live indicators use the in-progress bar.

## Portfolio replay, calibration, promotion

- **Portfolio replay:** empty. No PASS lane exists, so no trades are taken and equity is unchanged at
  $43,500 (`2026-09-25T1433-equity-curve.csv` has a header and no rows).
- **Calibration:** unchanged — n=108 realized fills, median slippage fraction 0.000127, usable.
- **`promote`:** all 13 families `pass_symbols 0`, `pooled_pass_regimes 0`, blocker
  `no_pass_symbols`, recommendation `collect_shadow`; "no changes recommended", nothing applied.
  `docs/research/strategy_hunt_promotion.md` regenerated (byte-identical to B1 — the table is all
  zeroes in both runs).

## Live posture (verified, unchanged)

`ENABLE_STRATEGY_FIT_GATE`, `STRATEGY_FIT_SHADOW`, `FIT_LANES_ENABLED` and
`FIT_LANE_LIVE_ENTRIES_ENABLED` are absent from both `.env` and `Config().yaml`, so the gate is off,
shadow-only when enabled, and no fit lane can place a live entry. `SENTIMENT_BACKEND` is also unset,
so the live news loop scores with VADER — which this run now matches. Nothing in `.env` or
`Config().yaml` was touched and the live bot was not restarted.

## B2 go/no-go

The gate, from `docs/superpowers/specs/2026-09-25-playbook-desk-design.md` § 9 (B1), as amended by
B1.5 (`docs/superpowers/plans/2026-09-25-playbook-desk-b1-5.md`):

> Gate: at least one family reaches PASS or a pooled regime row reaches PASS before B2's live wiring
> is enabled beyond shadow — and a pooled PASS counts only with positive excess over the shuffled
> baseline.

**Verdict: NO-GO, with more confidence than B1's.** 0 PASS rows of any kind: 0 of 9,231 unconditional,
0 of 1,390 symbol-regime, 0 of 69 pooled. The amendment is not what decides it — nothing reaches PASS
even before the excess test — but it does remove the path by which B1's gate could have been met by
noise. B2 live wiring stays shadow-only; B2's shadow-only work proceeds regardless.

What the run does establish, and what B2/B3 should build on:

1. **The regime split is real and large, and it is the only robust finding.** `dt_momentum short`:
   +0.172 R in `risk_off` vs −0.457 R in `risk_on`; `dt_momentum long`: +0.048 in `risk_on` vs −0.483
   in `risk_off`. The same asymmetry appears in `gap_play`, `vwap_reversion`, `orb_inplay` and
   `mean_rev_intraday`. Regime conditioning is worth keeping even though no conditioned row passes.
2. **Most of that asymmetry is drift, not edge.** For the large rows the baseline reproduces the bulk
   of the expectancy (`dt_momentum short/risk_off` +0.142 of +0.172; `dt_momentum long/risk_on` +0.003
   of +0.048 is the exception, not the rule). A lane that only harvests its regime's drift is a
   directional bet on the regime classifier, sized as if it were a rule.
3. **Session clustering is the binding constraint now.** 34 of the 36 pooled rows with n ≥ 500 fail
   `min_t_stat_session`, versus 29 failing the iid `min_t_stat`. Adding symbols no longer buys
   significance; only more independent *sessions* do, and `risk_off` has only ~179 of them in this
   sample. Any future pooled PASS needs either a longer history or a lane whose trades are not
   simultaneous across the universe.
4. **The one row that clears everything but decay is a warning, not a candidate.**
   `gap_play short/risk_off` earns +0.181 R full-sample and −0.336 R over the last four quarters. A
   floors set without `min_decay` would have passed a dead lane.

## Known limitations of this run

- Pooled `stability` and `decay` bucket by calendar quarter, and the sample's tail quarters are thin
  (2026 Q2 has 26 `gap_play short/risk_off` trades, 2026 Q3 has 41). `decay` is a four-window mean, so
  a thin recent quarter moves it a lot in either direction.
- The shuffled baseline preserves each session's open-to-close return by construction, so it is a
  control for *intraday structure*, not for the regime's drift itself. `excess_r` therefore answers
  "is there a rule here beyond the day's direction?" and not "is the regime label itself predictive?"
- `pead_smallcap` (283 trades) and `rs_leader` (169 trades) remain unmeasurable at this sample size;
  their pooled rows are all below `min_n` and their verdicts carry no information.
- `orb_inplay`'s numbers are a lower bound — the simulator has no trailing stop, so `trail_r` is
  applied as a take-profit.
- The shared run manifest `research/reports/strategy_hunt/<date>.json` is keyed by variant but not by
  root, so a control run in a separate root overwrites the real run's `run` record. It was restored
  from a pre-control copy for this report; see "Concerns" in the task report.
