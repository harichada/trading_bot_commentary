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

---

# B1.6 — regime label fix (rerun under entry-time labels)

**Run:** 2026-09-25 15:45–15:59, branch `feature/newsbus-autonomy-decision-card`, code at `7d394d5`.
Machine report: `research/reports/strategy_hunt/2026-09-25T1557-strategy-hunt.md` (+ `.json`,
`2026-09-25T1557-equity-curve.csv`). Supersedes the B1.5 numbers above wherever they disagree.

Why the rerun: the whole-branch review found that `research/strategy_hunt/data/regime.py` labelled
every trade with the regime computed from the entry **day's** SPY and VIX **closes**. A 10:00 entry
was therefore stamped with a regime decided six hours later. That is a look-ahead of exactly the
kind a regime-conditioned row is most vulnerable to: a day that *ended* down 1% with VIX up was
labelled `risk_off`, so "shorts work in `risk_off`" was partly the tautology "shorts work on days
that ended down". The label is now reconstructed from what was knowable at the entry bar — SPY's
day-to-date move through the last 5-minute bar that closed at or before the entry bar's open, plus
the **prior** session's VIX close and close-to-close change — through the same
`classify_regime` thresholds as `core/market_context.py`. Full description in
`docs/research/strategy_hunt.md` § "Regime and pooled rows" → "How a trade's regime label is
computed".

## Headline

**The regime split was mostly the look-ahead.** Nothing reaches PASS under either labelling, so the
B2 gate is unchanged — but the single finding B1/B1.5 offered as "real and large" does not survive
honest labels.

| | B1.5 (close-of-day labels) | **B1.6 (entry-time labels)** |
|---|---|---|
| Unconditional rows | 9,231 · PASS 0 / WATCH 127 / FAIL 9,104 | **9,227 · PASS 0 / WATCH 127 / FAIL 9,100** |
| Symbol-regime rows | 1,390 over 396 symbols · PASS 0 / WATCH 159 | **1,625 over 496 symbols · PASS 0 / WATCH 107 / FAIL 1,518** |
| Universe-pooled rows | 69 · PASS 0 / WATCH 14 / FAIL 55 | **67 · PASS 0 / WATCH 6 / FAIL 61** |
| Pooled rows with a baseline | 69 of 69 | **67 of 67** (9 flagged `baseline_population_mismatch`) |
| Mean unconditional exp_R | −0.126 | **−0.1265** |
| Out-of-sample trades | 335,499 real · 479,354 shuffled | **335,495 real · 479,357 shuffled** |
| Trades by regime | mixed 47.2% · risk_on 35.2% · risk_off 17.6% | **mixed 72.7% · risk_on 16.1% · risk_off 11.2%** |
| Sessions by regime | risk_off ~179 | **mixed 1,079 · risk_on 409 · risk_off 158** |

The trade count barely moves (−4 trades, from the `PEAD_ENTRY_BAR 6 → 5` correction in the same
batch), so this is the *same tape* re-labelled. What changes is which regime each trade is assigned
to: at 10:00 most sessions have not yet moved ±0.3%, so **72.7% of trades are now `mixed`** against
47.2% before. The old labels were, in effect, sorting trades by the day's eventual outcome.

## The two risk_off pooled rows, before and after

These are the two rows B1.5 named as its best candidates. Both flip sign.

| row | | n | exp_R | baseline exp_R | excess | t_session | recovery | decay | verdict | failed floors |
|---|---|---|---|---|---|---|---|---|---|---|
| `dt_momentum short/risk_off` | B1.5 | 13,370 | **+0.172** | +0.142 | +0.030 | +0.56 | 3.60 | — | WATCH | `min_t_stat_session`, `min_excess_r` |
| | **B1.6** | 8,868 | **−0.275** | −0.230 | **−0.045** | −3.97 | 0.00 | 0.00 | **FAIL** | 10 floors incl. `min_exp_r`, `min_excess_r` |
| `gap_play short/risk_off` | B1.5 | 5,292 | **+0.181** | +0.107 | +0.073 | +3.52 | 4.20 | −1.86 | WATCH | `min_decay` only |
| | **B1.6** | 2,477 | **−0.152** | −0.173 | +0.021 | −2.21 | 0.00 | 0.00 | **FAIL** | 10 floors incl. `min_exp_r`, `min_excess_r` |

`gap_play short/risk_off` — the row B1.5 described as "the only one of 69 that fails a single floor"
— loses 33 basis points of R per trade and now fails ten. `dt_momentum short/risk_off`, B1.5's
closest row, loses 45. Neither was a lane that worked when the tape was risk-off; both were lanes
that worked on days that *closed* risk-off, which is not a fact available at 10:00.

## The regime split headline, before and after

| family / side | regime | B1.5 exp_R (n) | **B1.6 exp_R (n)** |
|---|---|---|---|
| `dt_momentum` short | risk_off | **+0.172** (13,370) | **−0.275** (8,868) |
| | risk_on | −0.457 (30,551) | **−0.276** (14,625) |
| | mixed | −0.136 (40,690) | −0.176 (61,118) |
| `dt_momentum` long | risk_on | **+0.048** (24,229) | **−0.130** (12,463) |
| | risk_off | −0.483 (12,004) | **−0.140** (8,020) |
| | mixed | −0.174 (35,247) | −0.157 (50,997) |
| `gap_play` long | risk_on | +0.060 (8,771) | **+0.114** (2,916) |
| | risk_off | −0.233 (3,861) | **+0.060** (2,466) |
| | mixed | −0.120 (8,614) | −0.119 (15,864) |
| `gap_play` short | risk_off | **+0.181** (5,292) | **−0.152** (2,477) |
| | risk_on | −0.160 (7,560) | −0.191 (1,643) |
| | mixed | −0.060 (9,017) | −0.006 (17,749) |

B1.5's item 1 ("the regime split is real and large, and it is the only robust finding") claimed a
**+0.63 R** spread for `dt_momentum short` between `risk_off` and `risk_on`, and **+0.53 R** for
`dt_momentum long`. Under entry-time labels those spreads are **+0.000 R** and **−0.010 R** — the
two regimes are indistinguishable. The same collapse happens to `orb_inplay short` (+0.200 /
−0.539 → +0.003 / −0.176) and `mean_rev_intraday long` (+0.267 in `risk_on` → −0.159). **That
finding must be withdrawn.**

One conditional pattern does survive, and it is the source of both surviving WATCH rows of any
substance: `gap_play long` earns **+0.114 R in `risk_on` and +0.060 R in `risk_off` against −0.119
in `mixed`** — i.e. it wants a market that is *already* moving in either direction by 10:00 and
loses money when it is not. That is a statement about known-at-entry conditions, so it is the first
regime claim in this programme that is not circular. It is still not a PASS: both rows fail
`min_recovery`, `min_stability` and `min_t_stat_session`.

## Pooled verdicts under the corrected labels

**PASS 0 / WATCH 6 / FAIL 61** of 67 rows (B1.5: 0 / 14 / 55 of 69). Two rows disappear and none
appear: `rs_leader long/risk_on` and `vol_breakout long/risk_on` have **no** trades left in `risk_on`
once the labels stop counting days that merely *closed* up (both families are multiday and thin —
169 and 371 trades universe-wide; `rs_leader` now has 160 `mixed` + 9 `risk_off` and nothing else). The six WATCH rows:

| family | side | regime | n | exp_R | excess | baseline n | failed floors |
|---|---|---|---|---|---|---|---|
| `gap_play` | long | risk_off | 2,466 | +0.060 | **+0.154** | 2,666 | `min_pf`, `min_recovery`, `min_stability`, `min_t_stat`, `min_t_stat_session` |
| `gap_play` | long | risk_on | 2,916 | +0.114 | **+0.098** | 2,298 | `min_wr`, `min_recovery`, `min_stability`, `min_t_stat_session` |
| `mr_20d` | long | mixed | 701 | +0.027 | +0.008 | 701 | 9 floors incl. `min_excess_r` |
| `mr_20d` | long | risk_off | 331 | +0.227 | −0.031 | 331 | `min_n`, `min_recovery`, `min_stability`, `min_t_stat_session`, `min_excess_r` |
| `news_catalyst` | short | mixed | 424 | +0.044 | −0.026 | 438 | 10 floors incl. `min_excess_r` |
| `post_news_drift` | short | mixed | 394 | +0.052 | −0.000 | 394 | 9 floors incl. `min_excess_r` |

Only the two `gap_play long` rows clear the excess floor; the other four are WATCH on
`POOLED_WATCH_FLOORS` (n ≥ 200, exp_R > 0, PF ≥ 1.05) alone and fail `min_excess_r`. **Every one of
the 67 rows still fails `min_t_stat_session`, `min_recovery` or both** — B1.5's item 3 (session
clustering is the binding constraint) survives the relabelling intact.

## The new pooled guard (`baseline_population_mismatch`)

Nine of 67 rows carry the new key, meaning the shuffled control group's trade count is outside
[0.5×, 2×] the real row's `n`:

| row | real n | baseline n | ratio |
|---|---|---|---|
| `orb short/risk_on` | 166 | 1,027 | 6.19 |
| `orb short/mixed` | 1,826 | 9,353 | 5.12 |
| `orb long/mixed` | 1,750 | 9,286 | 5.31 |
| `orb long/risk_off` | 195 | 913 | 4.68 |
| `orb long/risk_on` | 478 | 1,374 | 2.87 |
| `orb short/risk_off` | 339 | 935 | 2.76 |
| `vwap_reversion long/risk_on` | 3,397 | 8,592 | 2.53 |
| `dt_momentum short/risk_on` | 14,625 | 31,834 | 2.18 |
| `mean_rev_intraday long/risk_off` | 1,299 | 599 | 0.46 |

All nine were already FAIL on eight or more other floors, so **the key changed no verdict this
run** — it is currently diagnostic. What it diagnoses is real: `orb` fires 3–6× as often on
structure-destroyed bars as on the real tape (its contraction filter keys off a *lack* of range
expansion, which a within-session permutation manufactures), so its `excess_r` was never comparing
like with like. Any future `orb` pooled row must be read with that in mind, and the earlier
"`orb` excess" numbers in the B1/B1.5 tables should be treated as uninterpretable rather than
merely weak.

## Steps and durations

Environment `set -a; . ./.env; set +a` with `POSTGRES_DSN` from `backtest.engine.DEFAULT_DSN`.
Logs under `data/strategy_hunt/` (not committed).

| # | Step | Command | Wall-clock | Outcome |
|---|---|---|---|---|
| 1 | run (real) | `… run --workers 12 --force` | **303 s** | 539/539 done, 0 errors, 335,495 trades (`run_real_b16.log`) |
| 2 | run (baseline) | `… run --shuffle-seed 7 --workers 12 --force` | **366 s** | 539/539 done, 0 errors, 479,357 trades → `trades_shuffle/` (`run_baseline_b16.log`) |
| 3 | score | `… score --postgres` | **71 s** | 9,227 unconditional + 1,625 symbol-regime + 67 pooled + 67 baseline; snapshot + Parquet + Postgres upsert (`score_b16.log`) |
| 4 | report | `… report` | 2 s | `2026-09-25T1557-strategy-hunt.md` + `.json` + equity CSV (`report_b16.log`) |
| 5 | promote | `… promote` | 1 s | 13-family table → `research/reports/strategy_hunt/strategy_hunt_promotion.md` (`promote_b16.log`) |

No step failed and no step was skipped. `export`/`news`/`earnings`/`context`/`calibrate` were not
re-run: this batch changed no input, only how trades are labelled and scored. The 2× cost control
was not re-run either — its root's trades are unaffected by the label fix, and its snapshot is
quoted above as B1.5's.

**The baseline guard passed.** `score` logged no `baseline_mismatch` / `baseline_stale` warning and
no row carries `no_baseline`: the control covers all 539 real symbols and ends on the same session.

Side effects of the same fix batch, visible in these numbers:

- **Symbol-regime rows rose 1,390 → 1,625 over 396 → 496 symbols** while their WATCH count fell
  159 → 107. More symbols now clear `REGIME_ROW_MIN_N` (= 60) in *some* regime because the labels
  concentrate into `mixed`, which is the largest bucket; and fewer of those rows are WATCH because
  the flattering `risk_off` rows are gone.
- `promote` writes to `research/reports/strategy_hunt/strategy_hunt_promotion.md` now (was
  `docs/research/`). All 13 families: `pass_symbols 0`, `pooled_pass_regimes 0`, blocker
  `no_pass_symbols`, recommendation `collect_shadow`. Nothing applied; `.env` and `Config().yaml`
  untouched, live bot not restarted.
- **Portfolio replay is still empty** (no PASS lane → no trades; `2026-09-25T1557-equity-curve.csv`
  is a header row). Calibration unchanged (n=108 fills, median slippage fraction 0.000127, usable).
- 0 cold-start rows: every (symbol, family, side) lane has ≥ `COLDSTART_MAX_N` walk-forward trades.

## B2 go/no-go, restated

The gate, unchanged:

> Gate: at least one family reaches PASS or a pooled regime row reaches PASS before B2's live
> wiring is enabled beyond shadow — and a pooled PASS counts only with positive excess over the
> shuffled baseline.

**Verdict: NO-GO — and B1.5's NO-GO should now be read as *more* firmly established, not less.**
0 PASS rows of any kind: 0 of 9,227 unconditional, 0 of 1,625 symbol-regime, 0 of 67 pooled. B2
live wiring stays shadow-only; B2's shadow-only work proceeds.

What changes in the *reasoning* behind the verdict:

1. **Withdraw B1.5's item 1.** "The regime split is real and large" was a look-ahead artefact. The
   `dt_momentum short` risk_off/risk_on spread goes from +0.63 R to 0.00 R. Any downstream design
   that leaned on regime-conditioned sizing for `dt_momentum`, `orb_inplay` or `mean_rev_intraday`
   has no evidence behind it.
2. **B1.5's item 2 stands, and hardens.** "Most of that asymmetry is drift, not edge" was the right
   instinct; the honest version is stronger — most of it was not even drift, it was the outcome
   variable leaking into the conditioning variable.
3. **B1.5's items 3 and 4 stand unchanged.** Session clustering is still the binding constraint
   (all 67 rows fail `min_t_stat_session` and/or `min_recovery`); `min_decay` still earns its place.
4. **One new, non-circular lead:** `gap_play long` prefers a market already moving by 10:00
   (+0.114 `risk_on`, +0.060 `risk_off`, −0.119 `mixed`, excess +0.098 / +0.154 over the shuffled
   control). It is the only regime claim in this programme measured on known-at-entry information.
   It is not a PASS and should be treated as a shadow-ledger hypothesis, not a lane.
5. **A methodological lesson worth carrying:** the leak was invisible in every summary statistic —
   B1 and B1.5 both looked internally consistent, and the shuffle baseline did not catch it because
   the baseline shares the same labels. Only reading the label's construction found it. Every
   future conditioning variable needs an explicit "what did we know, and when" note before its rows
   are believed.

## Known limitations added by B1.6

- **VIX is lagged one session.** The research store holds daily VIX only, so `vix_level` and
  `vix_change_pct` are the *prior* session's. Live, `vix_change_pct` is today's move. Lagging can
  never smuggle in the future, but on a day when VIX turns intraday the research label is blunter
  than the live one — so a research `mixed` row may be governing live signals the gate sees as
  `risk_off`. Closing this needs intraday VIX history in `export`.
- **`spy_change_pct` is 0.0 on the session's opening bar**, so every bar-0 entry lands in `mixed`
  unless the prior session's VIX alone decides it. That is correct (nothing has closed yet) but it
  means opening-bar families are structurally under-represented in `risk_on`/`risk_off`.
- **The `mixed` bucket now holds 72.7% of trades**, which makes it a weak conditioning variable:
  it is closer to "the market has not moved yet" than to a regime. A future iteration should
  consider finer intraday thresholds rather than reusing the live gate's daily-scale ±0.3%/±0.5%
  cuts at 10:00.
