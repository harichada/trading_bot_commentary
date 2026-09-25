# Strategy hunt — Playbook Desk B1 run (2026-09-25)

Full-universe re-run of the offline strategy-fit pipeline after Phase B1 (`v-playbook-desk-b1-2026-09-25`):
universe context, three in-play families (13 total), regime-conditioned and universe-pooled fit rows, and a
FinBERT-scored news corpus. Pipeline version 0.3.0, so every checkpoint from the morning run was invalid and
the whole universe was re-simulated (`run --force`).

Machine-generated companion report (same run, regenerated in place):
`research/reports/strategy_hunt/2026-09-25-strategy-hunt.md`; manifest `2026-09-25.json`. The pre-B1 (morning,
ten-family, VADER) version of that file is preserved in git at commit `b3be6d5` and is referred to below as
**the A run**.

## Headline

| | A run (10 families, VADER) | **B1 run (13 families, FinBERT)** |
|---|---|---|
| Symbols | 539 | 539 |
| Out-of-sample trades | 324,972 | **340,211** |
| Unconditional rows | 7,756 | **9,505** |
| Unconditional PASS / WATCH / FAIL | 0 / 127 / 7,629 | **0 / 127 / 9,378** |
| Symbol-regime rows | — (feature did not exist) | **1,390 over 396 symbols; PASS 0, WATCH 159, FAIL 1,231** |
| Universe-pooled regime rows | — | **69; PASS 0, WATCH 15, FAIL 54** |
| Mean exp_R over unconditional rows | not reported | **−0.124** |

**Zero PASS rows anywhere** — not one of the 9,505 unconditional rows, none of the 1,390 symbol-regime rows,
none of the 69 pooled regime rows. The three new families added 1,749 rows and 10,527 trades and produced no
WATCH and no PASS; the 127 WATCH rows are the same lanes the A run found (dt_momentum 48, gap_play 51,
vwap_reversion 25, mean_rev_intraday 3).

**B2 verdict: NO-GO for live wiring beyond shadow.** See "B2 go/no-go" below.

## What ran

Environment: `set -a; . ./.env; set +a`, `POSTGRES_DSN` from `backtest.engine.DEFAULT_DSN`,
`$PY = /home/nvidia/anaconda3/envs/trading-bot/bin/python`, `--workers 12` on a 12-core host.
`export --min-days 250` and `earnings` were run by the controller beforehand (log
`data/strategy_hunt/prep_b1.log`), as was the FinBERT news prep (`data/strategy_hunt/news_b1.log`,
`news_rescore.log`).

| Step | Command | Wall-clock | Outcome | Log |
|---|---|---|---|---|
| export | `export --min-days 250` | 138 s | 539 symbols, incremental (+1 bar each) | `prep_b1.log` |
| earnings | `earnings` | 51 s | 539 symbols, now with EPS surprises | `prep_b1.log` |
| news (fetch) | `SENTIMENT_BACKEND=finbert news` | 25 s | ~309 new rows, FinBERT-scored at fetch | `news_b1.log` |
| news (rescore) | `SENTIMENT_BACKEND=finbert news --rescore` | 761 s (12 m 41 s) | **725,484 rows re-scored on GPU, 0 errors** | `news_rescore.log` |
| context | `context` | 215 s | splits + universe context; exit 0 | `context_b1.log` |
| calibrate | `calibrate` | 29 s | n=108 realized fills, median_frac 0.000127, usable | `calibrate_b1.log` |
| run | `run --workers 12 --force` | 317 s (elapsed 315 s) | **539 done, 0 skipped, 0 errors**, 340,211 trades | `full_run_b1.log` |
| control: shuffle | `run --shuffle-seed 7 --root data/strategy_hunt_shuffle --symbols @sample50.txt --force` + `score --no-coldstart` | 42 s + 5 s | exit 0 | `control_shuffle_*_b1.log` |
| control: cost 0.5× | `run --cost-mult 0.5 --root data/strategy_hunt_cost05 …` + `score` | 36 s + 5 s | exit 0 | `control_cost05_*_b1.log` |
| control: cost 2× | `run --cost-mult 2.0 --root data/strategy_hunt_cost2 …` + `score` | 36 s + 5 s | exit 0 | `control_cost2_*_b1.log` |
| score | `score --postgres` | 59 s | rows + regime rows + pooled rows + Postgres upsert + report | `score_b1.log` |
| report | `report` | 2 s | rebuilt from the snapshot | `report_b1.log` |
| promote | `promote` | 2 s | 13-family table, `docs/research/strategy_hunt_promotion.md` | `promote_b1.log` |

Pipeline wall-clock from `context` to `promote`: **13 m 45 s** (13:15:59 → 13:28:32 ET, plus the 12 m 41 s
FinBERT rescore before it). Every step exited 0; nothing was skipped or patched.

Each control root symlinks `bars5`, `daily`, `news`, `earnings`, `splits` **and `context`** from the main root
(the last one is new in B1 — without it the in-play families see all-NaN `ctx_*` and propose nothing) and its
`trades/`, `choices/`, `checkpoints/` were wiped before the run so no pre-B1 output could leak in.

## Inputs

**Universe context** — `data/strategy_hunt/context/universe_daily.parquet`: 458,182 rows, 539 symbols,
2021-01-04 → 2026-09-24, 36 MB. Non-null coverage: `rvol_open5`/`rvol_rank5` 99.4%, `gap_pct`/`atr14_pct` 99.9%,
`rs_60`/`rs_rank_60` 92.8%, `rs_120` 85.8%, `pct_from_52w_high` 85.2%, `adv_dollars_20` 98.8%,
`news_sent_24h` 35.1% (a session only gets one if a headline landed in the prior 24 h),
`eps_surprise_pct` 4,183 of the 4,243 `earnings_prev` sessions (98.6%). ADV buckets: mega 57,924,
large 181,209, mid 183,090, small 35,959 sessions. No symbol failed the build.

**Sentiment** — the research news cache was scored by **FinBERT** (`ProsusAI/finbert` from the local
safetensors checkpoint `models/finbert`, GPU device 0): **725,793 rows across 536 symbol files, 100.0 % with a
`sentiment_finbert` score and 100.0 % with `sentiment_vader`**. Headline-level agreement between the two is low
(per-symbol Pearson r 0.09–0.36 on the symbols sampled), i.e. this was a real change of input, not a rescale.

That change is visible in the news families: `news_catalyst` went from 1,927 pooled trades in the A run to
**6,639** here (long exp_R −0.064 → −0.077, short −0.006 → −0.073) — FinBERT crosses the families'
`min_abs_sentiment` thresholds far more often, and the extra trades are not better ones. `post_news_drift`
kept the same trade count (969) with slightly worse expectancy (short +0.039 → +0.030). Every other old
family's pooled numbers are unchanged to within ±0.005 exp_R, which is the expected no-op (they do not read sentiment).

## Results — pooled regime rows (the headline)

69 pooled rows (one per family/side/regime that traded), symbol `*`, source `pooled`, scored against
`POOLED_FLOORS` (n ≥ 500, sessions ≥ 60, PF ≥ 1.2, WR ≥ 45 %, exp_R ≥ +0.05, max DD ≤ 60 R, stability ≥ 0.6,
decay ≥ 0.5, t_stat ≥ 3.0, exp_r_lb ≥ 0). **PASS 0, WATCH 15, FAIL 54.**

Best rows by regime (exp_R order; all of the positive-expectancy ones):

**risk_on**

| family | side | verdict | n | exp_R | lb | PF | WR | t | stab | decay | DD(R) | failed floors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mean_rev_intraday | long | WATCH | 2,051 | +0.267 | +0.215 | 1.45 | 0.49 | +7.93 | 0.79 | −1.12 | 80 | max_dd_r, min_decay |
| mr_20d | long | WATCH | 325 | +0.179 | +0.082 | 1.53 | 0.58 | +3.22 | 0.72 | 0.52 | 12 | min_n |
| orb_inplay | long | WATCH | 1,912 | +0.128 | +0.007 | 1.11 | 0.29 | +1.84 | 0.58 | −0.00 | 190 | pf, wr, dd, stability, decay, t |
| pullback_trend | long | WATCH | 328 | +0.119 | +0.041 | 1.32 | 0.54 | +2.17 | 0.82 | 2.02 | 21 | min_n, min_t_stat |
| news_catalyst | long | WATCH | 1,367 | +0.090 | +0.033 | 1.18 | 0.47 | +2.72 | 0.63 | 1.43 | 19 | pf, t |
| vwap_reversion | long | WATCH | 10,594 | +0.073 | +0.052 | 1.14 | 0.48 | +5.71 | 0.58 | −1.15 | 210 | pf, dd, stability, decay |
| gap_play | long | WATCH | 8,771 | +0.060 | +0.036 | 1.10 | 0.43 | +4.20 | 0.58 | 1.47 | 441 | pf, wr, dd, stability |
| dt_momentum | long | WATCH | 24,229 | +0.048 | +0.035 | 1.09 | 0.46 | +5.96 | 0.58 | 0.08 | 553 | pf, exp_r, dd, stability, decay |

**risk_off**

| family | side | verdict | n | exp_R | lb | PF | WR | t | stab | decay | DD(R) | failed floors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| post_news_drift | short | FAIL | 79 | +0.240 | +0.072 | 1.86 | 0.58 | +2.13 | 0.67 | −0.22 | 3 | n, sessions, decay, t |
| orb_inplay | short | WATCH | 876 | +0.200 | +0.020 | 1.16 | 0.32 | +1.67 | 0.58 | 0.92 | 92 | pf, wr, dd, stability, t |
| gap_play | short | WATCH | 5,292 | +0.181 | +0.151 | 1.35 | 0.47 | +9.87 | 0.63 | −1.86 | 228 | max_dd_r, min_decay |
| **dt_momentum** | **short** | **WATCH** | **13,370** | **+0.172** | **+0.077** | **1.33** | **0.52** | **+3.85** | **0.84** | **0.71** | **639** | **max_dd_r only** |
| vwap_reversion | short | WATCH | 3,385 | +0.076 | +0.037 | 1.14 | 0.48 | +3.32 | 0.58 | −0.22 | 204 | pf, dd, stability, decay |
| news_catalyst | short | WATCH | 688 | +0.066 | −0.009 | 1.12 | 0.46 | +1.39 | 0.58 | 1.88 | 23 | pf, stability, t, exp_r_lb |
| vol_breakout | long | FAIL | 26 | +0.412 | +0.124 | 2.55 | 0.56 | +1.85 | 0.00 | 0.00 | 3 | n, sessions, stability, decay, t |

**mixed**

| family | side | verdict | n | exp_R | lb | PF | WR | t | stab | decay | DD(R) | failed floors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pead_smallcap | short | FAIL | 39 | +0.234 | −0.201 | 1.50 | 0.51 | +1.07 | 0.50 | −0.14 | 4 | n, sessions, stability, decay, t, exp_r_lb |
| mr_20d | long | WATCH | 512 | +0.108 | +0.037 | 1.29 | 0.50 | +2.30 | 0.47 | 2.20 | 25 | stability, t |
| pead_smallcap | long | FAIL | 70 | +0.025 | −0.231 | 1.05 | 0.41 | +0.16 | 0.44 | 11.11 | 14 | most floors |
| post_news_drift | short | WATCH | 222 | +0.022 | −0.059 | 1.07 | 0.53 | +0.41 | 0.53 | 10.51 | 18 | n, pf, exp_r, stability, t, exp_r_lb |

`mixed` is the largest regime bucket and is negative nearly everywhere else: every high-n family is between
−0.06 and −0.20 exp_R in `mixed` (dt_momentum long −0.174 on 35,247 trades, vwap_reversion long −0.196 on
27,896). The regime split is the single strongest signal in this run: **long momentum/breakout lanes are
positive only in `risk_on`, short lanes only in `risk_off`, and everything is negative in `mixed`** —
e.g. dt_momentum long +0.048 (risk_on) / −0.174 (mixed) / −0.483 (risk_off); dt_momentum short +0.172
(risk_off) / −0.136 (mixed) / −0.457 (risk_on). That is exactly the asymmetry `lookup_regime` exists to
exploit, and it survives in-sample-to-out-of-sample here at scale (13,370–35,247 trades per cell).

### The pooled `max_dd_r` floor is the binding constraint, and it is not scale-invariant

40 of the 69 pooled rows fail `max_dd_r`, and exactly **one row fails only `max_dd_r`**: dt_momentum short
risk_off (n 13,370, exp_R +0.172, PF 1.33, t +3.85, stability 0.84, decay 0.71, DD 639 R vs the 60 R cap).
Of the 39 pooled rows that clear the pooled `min_n` of 500, only 3 have DD ≤ 60 R. A pooled equity curve
concatenates every symbol's trades, so its drawdown grows roughly with n — the floors therefore pull in
opposite directions (`min_n` 500 up, `max_dd_r` 60 R down) and a high-n pooled row is effectively barred from
PASS by arithmetic rather than by evidence. This is a floors-design issue to settle before pooled PASS can
mean anything: either normalize the pooled drawdown (per-trade, or per-symbol-average R), or drop the floor
for pooled rows and rely on t_stat/stability/decay. Recorded here, **not** changed in this run.

## Results — symbol-regime rows

1,390 rows over 396 symbols (mixed 870, risk_on 460, risk_off 60): **PASS 0, WATCH 159, FAIL 1,231**.
Regime rows only exist where a symbol traded that regime ≥ 60 times, which is why `risk_off` is thin.

Every WATCH row misses on `min_n` (the 150-trade PASS floor) — the regime split cuts each symbol's sample by
construction. The strongest ones are worth naming because they are the shortlist B2's shadow work should
watch, not because they are evidence:

| symbol | family | side | regime | n | exp_R | PF | t | failed |
|---|---|---|---|---|---|---|---|---|
| TQQQ | dt_momentum | short | risk_off | 73 | +0.699 | 2.64 | +3.59 | min_n |
| SOXL | dt_momentum | short | risk_off | 80 | +0.548 | 2.47 | +3.31 | min_n |
| CZR | dt_momentum | short | mixed | 73 | +0.540 | 1.90 | +2.50 | min_n, min_wr |
| SOXL | dt_momentum | long | risk_on | 101 | +0.487 | 2.47 | +3.69 | min_n |
| HIVE | dt_momentum | long | risk_on | 63 | +0.455 | 2.02 | +2.47 | min_n |
| SOFI | dt_momentum | short | risk_off | 73 | +0.452 | 2.12 | +2.74 | min_n |
| SHOP | dt_momentum | long | risk_on | 113 | +0.439 | 2.06 | +3.29 | min_n |
| PTON | dt_momentum | short | risk_off | 80 | +0.445 | 2.03 | +2.77 | min_n |

Leveraged ETFs and high-beta names dominate, in the expected direction (short them in `risk_off`, long them in
`risk_on`). No symbol-regime row for any of the three new families reaches WATCH.

## Results — the three new families vs the ten old ones

| family | rows | PASS | WATCH | trades | pooled best (regime) | mean exp_R |
|---|---|---|---|---|---|---|
| **orb_inplay** (new) | 1,023 | 0 | 0 | 10,075 | WATCH +0.200 short/risk_off; WATCH +0.128 long/risk_on | −0.206 |
| **pead_smallcap** (new) | 283 | 0 | 0 | 283 | FAIL +0.234 short/mixed (n=39) | −0.025 |
| **rs_leader** (new) | 169 | 0 | 0 | 169 | FAIL −0.037 long/risk_on (n=66) | −0.085 |
| dt_momentum | 1,075 | 0 | 48 | 156,091 | WATCH +0.172 short/risk_off | −0.194 |
| gap_play | 1,039 | 0 | 51 | 43,115 | WATCH +0.181 short/risk_off | −0.024 |
| vwap_reversion | 1,067 | 0 | 25 | 104,501 | WATCH +0.076 short/risk_off | −0.234 |
| mean_rev_intraday | 530 | 0 | 3 | 10,094 | WATCH +0.267 long/risk_on | −0.195 |
| mr_20d | 1,010 | 0 | 0 | 2,274 | WATCH +0.179 long/risk_on | −0.050 |
| news_catalyst | 881 | 0 | 0 | 6,639 | WATCH +0.090 long/risk_on | −0.048 |
| orb | 1,007 | 0 | 0 | 4,754 | FAIL −0.068 short/risk_off | −0.211 |
| post_news_drift | 645 | 0 | 0 | 969 | FAIL +0.240 short/risk_off (n=79); WATCH +0.022 short/mixed | +0.009 |
| pullback_trend | 481 | 0 | 0 | 876 | WATCH +0.119 long/risk_on | −0.032 |
| vol_breakout | 295 | 0 | 0 | 371 | FAIL +0.412 long/risk_off (n=26) | −0.141 |

**Best of the three new families is `orb_inplay`**, and only at the pooled level: short in `risk_off`
(n 876, exp_R +0.200, PF 1.16, t +1.67) and long in `risk_on` (n 1,912, exp_R +0.128, PF 1.11, t +1.84).
Both are WATCH, both fail on win rate (0.29–0.32) and profit factor as well as drawdown — the shape of a
lane that wins rarely and large, which the simulator's take-profit-instead-of-trailing-stop approximation
(`trail_r`) systematically under-measures. `orb_inplay` is also the family with the highest single-symbol
expectancies in the whole run (RIVN long exp_R +1.07 on n=48, t +2.36; NBIS +0.98 on n=37), all far below the
n ≥ 150 floor. In `mixed` it is badly negative (−0.310 long on 2,494 trades), consistent with everything else.

`pead_smallcap` and `rs_leader` produced almost no trades — 283 and 169 across 539 symbols and 19 quarters —
because their gates are narrow by design (a small/mid-cap ADV bucket plus a same-direction EPS surprise plus a
gap; a top-decile RS leader pulling back to its EMA with a positive prior day). Their numbers are unusable at
this sample size in either direction: `pead_smallcap` short/mixed is +0.234 exp_R on 39 trades with an
exp_r_lb of −0.201. Nothing here says the ideas are wrong; it says the pipeline cannot measure them yet.
If they matter for B2, they need either looser gates or a longer history, and that is a decision, not a result.

## Controls

All three controls ran on the deterministic 50-symbol sample (`data/strategy_hunt/sample50.txt`, every 10th
symbol of the sorted universe), with the same bars/context/news/earnings inputs as the base run.

| Control | Unconditional rows | PASS | WATCH | mean exp_R | Pooled PASS | Pooled WATCH |
|---|---|---|---|---|---|---|
| base (539 symbols) | 9,505 | 0 | 127 | −0.124 | 0 | 15 |
| base restricted to the same 50 symbols | 898 | 0 | 17 | −0.135 | **0** | 6 |
| shuffle (`--shuffle-seed 7`) | 903 | 0 | 10 | −0.298 | **1** | 4 |
| cost 0.5× | 898 | 0 | 28 | −0.080 | 0 | 7 |
| cost 2× | 898 | 0 | 3 | −0.245 | 0 | 1 |

**Cost sensitivity behaves correctly.** Halving costs lifts mean exp_R from −0.124 to −0.080 and WATCH rows
from 3 to 28; doubling them drops mean exp_R to −0.245 and leaves 3 WATCH rows and a single pooled WATCH
(gap_play short risk_off, exp_R +0.168 — the one pooled lane that survives 2× costs). No control produces an
unconditional PASS, so the run is not cost-threshold-fragile in the PASS sense: it has nothing to be fragile
about.

**The shuffle control produced a pooled PASS, and that is the most important number in this report.** On
structure-destroyed bars (5-minute returns permuted within each session, seed 7) the pooled row
`dt_momentum:short:risk_off` clears **every** pooled floor: n 1,917, sessions 179, exp_R +0.147,
exp_r_lb +0.094, PF 1.25, WR 0.47, t_stat **+4.25**, stability 0.74, decay 0.85, DD 39.5 R, `failed_floors []`.
On the *same 50 symbols with real bars* that lane is FAIL at exp_R −0.176. So:

- the false-discovery estimate for pooled rows on this sample is **1 shuffled pooled PASS vs 0 real pooled
  PASS** — the pooled floors admit a PASS on pure noise before they admit one on the real tape;
- part of the mechanism is the `max_dd_r` scale problem above: a 50-symbol root concatenates far fewer trades,
  so its pooled drawdown (39.5 R) fits under a floor that the 539-symbol version of the same lane (639 R)
  cannot;
- the consequence for B2 is concrete: **a pooled PASS must not by itself authorize live wiring.** The spec's
  go/no-go gate is satisfiable by noise as the floors stand today.

The machine report's Controls section reports "shuffle: PASS rows 0 … false-discovery estimate 0 / 0" — that
counts *unconditional* rows only and does not see pooled rows at all. Read it together with this section.

### How to read these numbers

- **Pooled `t_stat` overstates significance.** It is a trade-level mean-R t-statistic that treats every trade
  as independent, but pooled rows contain many same-session entries across cross-sectionally correlated
  symbols. The effective sample is materially smaller than n, so `min_t_stat 3.0` is a weaker filter than it
  reads (the shuffle PASS at t +4.25 is the demonstration).
- **A pooled PASS needs ≥ 4 calendar quarters with trades**, because pooled stability/decay bucket by quarter
  rather than by the per-symbol walk-forward window index. That is not binding in this run: all 69 pooled
  groups span 7–19 quarters (median 19), out-of-sample trades run 2022-01-03 → 2026-09-24.
- **Symbol-regime rows exist only above 60 trades in that regime**; below that the scorer deliberately writes
  no row, so a thin regime cannot manufacture a FAIL that outranks the pooled or unconditional evidence. 396
  of 539 symbols have at least one regime row; `risk_off` has only 60 rows across the universe.
- **Unconditional rows still govern by default in the live gate** (`STRATEGY_FIT_REGIME_ROWS` only changes
  which row is read when a regime is known), and the gate remains off and shadow-only.
- The `orb_inplay` `trail_r` take-profit approximation means that family's numbers are a lower bound on what a
  real trailing stop would produce; a trailing-stop simulator option is the obvious follow-up.

## Portfolio replay and calibration

No PASS lane exists, so the portfolio replay is empty (0 trades taken, equity unchanged at $43,500) and the
equity-curve CSV has no data — same as the A run. Cost calibration is unchanged: n=108 realized fills, median
slippage fraction 0.000127, p75 0.00104, usable (used as a fixed slippage override).

`promote` (`docs/research/strategy_hunt_promotion.md`) now lists all 13 families with a new
`pooled_pass_regimes` column: 0 for every family, blocker `no_pass_symbols`, recommendation `collect_shadow`
for all 13. No `.env` change is recommended and none was made.

## B2 go/no-go

The spec's gate (`docs/superpowers/specs/2026-09-25-playbook-desk-design.md` § 9, B1):

> Gate: at least one family reaches PASS or a pooled regime row reaches PASS before B2's live wiring is
> enabled beyond shadow.

**Verdict: NO-GO — the gate is not met.** This run produced 0 PASS rows of any kind: 0 of 9,505 unconditional
rows, 0 of 1,390 symbol-regime rows, 0 of 69 pooled regime rows. **No B2 live wiring may be enabled beyond
shadow.** B2's shadow-only work (DailyPlan, InPlayRanker, AdaptivePlaybook, journal, endpoints, engine wiring
in shadow, nightly playbook update) proceeds regardless, as does the strategy-fit gate's current posture:
`ENABLE_STRATEGY_FIT_GATE=False`, `STRATEGY_FIT_SHADOW=True`, `FIT_LANES_ENABLED` empty — all defaults,
unchanged by this run, verified absent from both `.env` and `Config().yaml`.

Two recommendations attached to the gate itself, for the controller to rule on:

1. **Fix the pooled floors before a pooled PASS is allowed to satisfy the gate.** As shown above, the shuffle
   control reaches a pooled PASS while the real tape does not, and the only near-miss real row misses solely on
   a drawdown floor that scales with universe size. A gate that noise can pass is not a gate.
2. **The regime asymmetry is the finding worth building B2 on.** No single (symbol, family, side) lane clears
   the PASS floors, but the regime conditioning separates +0.17 R from −0.46 R on the same family with tens of
   thousands of trades on each side. That is a plan-level edge (which side to take, and when to stand down),
   not a lane-level one — which is precisely what a DailyPlan/playbook layer can act on in shadow.

## Artifacts

- This report: `research/reports/strategy_hunt/2026-09-25-strategy-hunt-b1.md`
- Machine report (regenerated by this run): `research/reports/strategy_hunt/2026-09-25-strategy-hunt.md`
- Manifest: `research/reports/strategy_hunt/2026-09-25.json`; equity curve CSV: unchanged (no PASS lanes)
- Promotion table: `docs/research/strategy_hunt_promotion.md`
- Snapshot (git-ignored): `data/strategy_hunt/fit/strategy_fit.json` (+ `fit_2026-09-25.parquet`,
  `fit_regime_2026-09-25.parquet`, `fit_pooled_2026-09-25.parquet`), Postgres table `strategy_fit` upserted
  with the 9,505 unconditional rows only (regime/pooled rows are deliberately kept out of it — their key would
  collide across regimes)
- Control snapshots: `data/strategy_hunt/fit/strategy_fit_{shuffle,cost_0.5,cost_2.0}.json`
- Logs: `data/strategy_hunt/{prep_b1,news_b1,news_rescore,context_b1,calibrate_b1,full_run_b1,control_*_b1,score_b1,report_b1,promote_b1,b1_pipeline}.log`
