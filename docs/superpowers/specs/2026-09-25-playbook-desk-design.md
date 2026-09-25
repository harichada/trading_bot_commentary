# Playbook Desk — Design

Date: 2026-09-25
Branch: `feature/newsbus-autonomy-decision-card`
Status: approved by delegation (user: "take necessary decisions to make this bot working professionally and profitably")
Builds on: `docs/superpowers/specs/2026-09-24-strategy-hunt-design.md` (fit pipeline, gate, fit lanes) and the four research reports under `research/reports/strategy_hunt/2026-09-25-*.md`.

## 1. Goal

Make the bot reason like a skilled discretionary trader and trade only setups with demonstrated, regime-conditional edge:

- Widen the strategy vocabulary with the families the evidence supports: opening-range breakout restricted to "stocks in play", post-earnings drift in small and mid caps, relative-strength leader continuation.
- Give every family a cross-sectional context (relative volume rank, relative strength rank, catalysts) so selectivity, which the evidence says is the edge, is testable and enforceable.
- Condition verdicts on market regime, the strongest effect in our own data.
- Add the front half of a trader's day: a pre-market plan, an in-play ranking, an explicit sit-out decision, a daily risk budget.
- Close the feedback loop: expectancy per setup and regime from shadow and live results suspends losers and upsizes winners.
- Write a daily journal and, optionally, a GPU-hosted briefing and post-trade critique that is advisory only.

Non-goals: LLM-generated trade signals (evidence says leakage); new order types; changing bracket/OCO handling; any live promotion without Stage A evidence.

## 2. Evidence this design rests on

| Claim | Source (see reports) | Consequence |
|---|---|---|
| ORB Sharpe 0.48 unfiltered → 2.81 on top-RVOL "stocks in play" | Zarattini, Barbon, Aziz 2024 (survey §ORB) | Cross-sectional RVOL rank is a first-class input; hold-to-close exit variant |
| Small/mid-cap low-coverage PEAD still ~5%/yr net gross edge; large-cap PEAD dead | academic §PEAD | PEAD family restricted by ADV bucket, needs surprise magnitude |
| Longs earn only in risk-on; shorts only in risk-off (our data: mean-rev long +0.27R risk-on vs −0.59R risk-off, n=2k/2.7k) | first full run report | Fit rows and gate keyed by regime |
| Every profitable practitioner: fixed R sizing, confirmed triggers, regime-gated aggression, pre-committed management, daily loss limits, expectancy per setup | human-trader report §1–8 | Desk components below; most risk pieces already exist in the engine |
| Backtests decay ~70% live; LLM-agent alpha is mostly leakage | profitable-bots report | Shadow-first stays mandatory; LLM only for briefing/critique |

## 3. Architecture

```
research/strategy_hunt/context.py      UniverseContext: daily cross-sectional table for all symbols
strategies/families/context.py         pure accessors: in_play_rank, rs_rank, near_52w_high, catalyst flags
strategies/families/inplay.py          orb_inplay, pead_smallcap, rs_leader
research/strategy_hunt/score/regime_rows.py   (symbol, family, side, regime) and pooled (family, side, regime) rows
core/strategy_fit.py                   regime-aware lookup (falls back to unconditional row)
core/strategy_fit_gate.py              consults live regime; pooled-lane fallback
core/desk/daily_plan.py                DailyPlan: regime, breadth, VIX, calendar, risk budget, bias, sit-out
core/desk/inplay_ranker.py             live in-play ranking from screener candidates + RVOL + catalysts + RS
core/desk/playbook.py                  AdaptivePlaybook: per (family, side, regime) status/size from expectancy
core/desk/journal.py                   daily markdown journal; optional LLM briefing/critique (advisory)
core/engine.py                         wiring: plan at 09:00/09:35, ranker feeds watchlist when enabled, gate reads plan+playbook
research/strategy_hunt/playbook_update.py   nightly: resolve ledgers + live trades → playbook.json
api/routes.py                          GET /api/desk/plan, GET /api/desk/playbook, GET /api/desk/journal
```

Import boundary unchanged: `core/`, `api/`, `strategies/` never import `research/`. Cross-sectional context is a data file the research pipeline writes and the live desk computes for the live watchlist.

## 4. Universe context (research)

`research/strategy_hunt/context.py` builds `data/strategy_hunt/context/universe_daily.parquet` from every `bars5/*.parquet` and `daily/*.parquet`:

| Column | Definition |
|---|---|
| `date`, `symbol` | key |
| `rvol_open5` | first 5-min bar volume ÷ mean first-5-min volume over the prior 14 sessions |
| `rvol_open15` | same over the first three bars |
| `rvol_rank5` | cross-sectional rank of `rvol_open5` among symbols that day (1 = highest) |
| `gap_pct` | session open ÷ prior close − 1 |
| `rs_20`, `rs_60`, `rs_120` | symbol return over 20/60/120 sessions minus SPY's, through the prior session |
| `rs_rank_60` | cross-sectional percentile of `rs_60` (0–1) |
| `pct_from_52w_high` | prior-session close ÷ 252-session high − 1 |
| `adv_dollars_20` | mean close×volume over 20 sessions; `adv_bucket` mega/large/mid/small (same cut-offs as calibration) |
| `atr14_pct` | prior-session ATR14 ÷ close |
| `earnings_today`, `earnings_prev` | from the earnings cache: report date == date, == prior session |
| `eps_surprise_pct` | from yfinance `earnings_dates` (reported − estimate) ÷ abs(estimate), on `earnings_prev` days; NaN otherwise |
| `news_count_24h`, `news_sent_24h` | Alpaca news cache: items in prior 24 h and mean sentiment |

Built incrementally per date; rebuilt fully weekly. The runner passes the symbol's slice (indexed by date, reindexed onto bars) to families as a new `context` argument; existing families ignore it. `strategies/families/context.py` exposes pure helpers so families read named columns with NaN-safe defaults.

Live equivalent (`core/desk/inplay_ranker.py`): computes `rvol_open5/15`, `gap_pct`, `rs_20/60`, `pct_from_52w_high`, `earnings_today/prev`, news counts for the screener's candidate set plus the pinned list, from the merged bars history the fit lanes already use, and ranks them. Same column names, so a family behaves identically live and in research.

## 5. New families (`strategies/families/inplay.py`)

| Family | Side | Hold | Entry | Stop / exit | Grid (≤ 27) |
|---|---|---|---|---|---|
| `orb_inplay` | long, short | intraday, hold to close | at the close of the first 5-min bar: `rvol_rank5 <= top_n`, price ≥ $5, `atr14_pct ≥ 0.5%`, long if first bar green and next bar breaks its high, short if red and breaks its low; entry at bar 2 open on break | stop `max(0.10×ATR14, first-bar low)` (short mirror); exit at 15:55 close, or trail at `trail_r` R once ≥ `trail_r` | `top_n {10, 20, 30}`, `stop_atr {0.05, 0.10, 0.20}`, `trail_r {0 (none), 5, 10}` |
| `pead_smallcap` | long, short | multiday 5–20 sessions | on `earnings_prev` with `eps_surprise_pct ≥ +s` (long) or ≤ −s (short), `adv_bucket in {mid, small}`, and the first 30 minutes holding the gap direction (close at 10:00 above/below open); entry at the 10:00 bar open | stop 1.5×ATR14 beyond entry; time stop `hold_days`; earnings-eve flatten | `s {5, 10, 20}` percent, `hold_days {5, 10, 20}`, `min_gap {0, 2, 4}` percent |
| `rs_leader` | long | multiday 10–20 sessions | session open: `rs_rank_60 ≥ 0.9`, `pct_from_52w_high ≥ −15%`, prior session was a pullback low touching the 10- or 20-day EMA with `d_ret1 > 0` (resume) | stop 2×ATR14; target 3R; time stop 20; earnings-eve flatten | `ema {10, 20}`, `rs_floor {0.85, 0.90, 0.95}`, `atr_stop {1.5, 2.0, 2.5}` |

All three ship through the same pipeline, scorer, fit table, and shadow-first lane adapter. `orb_inplay` and `rs_leader` are session-open or first-bars families, which the adapter now supports.

## 6. Regime-conditioned fit

- The scorer emits, in addition to the existing per-(symbol, family, side) rows, rows keyed `(symbol, family, side, regime)` for regimes `risk_on`, `risk_off`, `mixed`, and pooled rows `(*, family, side, regime)` across the universe with pooled floors (`n ≥ 500`, `sessions ≥ 60`, `t_stat ≥ 3.0`, `exp_r ≥ 0.05`, `pf ≥ 1.2`, `stability ≥ 0.6` over windows).
- JSON snapshot gains `"regime_rows"` and `"pooled_rows"`; `core/strategy_fit.py` gets `lookup_regime(symbol, strategy, side, regime)` returning the symbol-regime row, else the pooled regime row (flagged `pooled=True`), else `None`.
- Gate: when a live regime is known, `evaluate_fit_gate` uses `lookup_regime`; a PASS pooled row lets a lane trade any symbol in the in-play list for that regime (still shadow-first through the lane's own live list). Unconditional rows remain the fallback when the regime is `unknown`.

## 7. The desk (live)

### 7.1 DailyPlan (`core/desk/daily_plan.py`)

Built at 09:00 ET from cached data and refreshed at 09:35 with opening RVOL:

```
DailyPlan(date, regime, spy_change_pct, vix, vix_change_pct, breadth_score, calendar_flags,
          direction_bias: {long_ok, short_ok}, risk_budget_pct, max_trades, sit_out: bool, sit_out_reason,
          in_play: [InPlayRow], notes)
```

Rules (all pure, tested):
- `sit_out` when a high-impact calendar event window is active before 10:30, when `vix_change_pct > 15`, after two consecutive losing days (from `trade_history`), or when the fit snapshot is stale and the gate is enforcing.
- `risk_budget_pct` = `MAX_DAILY_LOSS` × {1.0 risk_on, 0.6 mixed, 0.4 risk_off}, halved after a losing day.
- `max_trades` = 6 (risk_on), 4 (mixed), 2 (risk_off); a trade-rate limiter in the router audits `desk_limit skip max_trades_reached`.
- `direction_bias` from `allows_long/allows_short` plus breadth.
Persisted to `data/desk/plan_<date>.json` (atomic) and served by `GET /api/desk/plan`.

### 7.2 InPlayRanker (`core/desk/inplay_ranker.py`)

Score = 0.4×min(rvol_open15/3, 1) + 0.2×|gap_pct|/5 capped + 0.2×catalyst (earnings today/prev or ≥ 3 news items in 24 h) + 0.2×|rs_20| rank. Top `DESK_IN_PLAY_N` (default 20) plus pinned names become the active watchlist when `ENABLE_DESK_WATCHLIST` is on; the screener loop keeps producing candidates as today. Audit line per refresh: `engine_decision component=desk_inplay action=rank ...`.

### 7.3 AdaptivePlaybook (`core/desk/playbook.py`, nightly updater in research)

`playbook.json`: per `(family, side, regime)`: `status` (`enabled`, `suspended`, `probation`), `size_mult` (0.5–1.5), `exp_r_recent`, `n_recent`, `updated_at`. Rules: suspend when the last 30 resolved trades have `exp_r < 0` and `t_stat < −1`; probation (size 0.5) when `exp_r < 0` with insufficient n; upsize to 1.25 when `exp_r_recent ≥ 0.15` over ≥ 40 trades and a PASS/WATCH fit row exists. Inputs: resolved shadow ledgers (`shadow_resolver`) plus live `bot_trades` rows tagged with `fit_family`/regime. The gate reads `playbook.status` after the fit verdict; sizing multiplies by `size_mult` through `signal.reasoning["strategy_size_multiplier"]`.

### 7.4 Journal (`core/desk/journal.py`)

After the close: `docs/journal/<date>.md` with the plan, every governed decision (accepted, skipped and why), trades with R and exit reason, expectancy by setup×regime for the day and trailing 20 days, playbook changes, and a "what to change" list derived from the rules above. Optional `DESK_LLM_BRIEFING` (default off): a local model on the RTX 3090 (transformers, e.g. a 7–8B instruct model) writes a 200-word morning briefing from the plan and a 200-word post-close critique from the journal; both stored, never used as signals.

### 7.5 FinBERT sentiment (from the earlier ruling)

`research/strategy_hunt/data/sentiment.py` and `core/sentiment_backend.py`: `score(headlines) -> list[float]` with backend `vader` (default) or `finbert` (`ProsusAI/finbert` on CUDA when available), selected by `SENTIMENT_BACKEND`; the news cache stores both columns so families can choose (`sentiment_vader`, `sentiment_finbert`); the live NewsBus uses the same backend so research and live agree.

## 8. Engine wiring (all shadow-first, flag-gated)

| Flag | Default | Effect |
|---|---|---|
| `ENABLE_DESK` | False | build DailyPlan, journal |
| `ENABLE_DESK_WATCHLIST` | False | in-play ranker feeds the dynamic watchlist |
| `ENABLE_DESK_LIMITS` | False | sit-out, risk budget, max-trades enforced (else audited as shadow) |
| `ENABLE_PLAYBOOK_GATE` | False | playbook status/size applied (else audited) |
| `STRATEGY_FIT_REGIME_ROWS` | True | gate uses regime rows when the regime is known |
| `DESK_LLM_BRIEFING` | False | GPU briefing/critique |
| `SENTIMENT_BACKEND` | vader | vader or finbert |

Every desk decision emits `engine_decision component=desk_* ...` and, in shadow, never blocks.

## 9. Phases

- **B1 (research first):** universe context, three families, regime rows, pooled rows, regime-aware reader/gate, sentiment backend. Run the full pipeline; report. Gate: at least one family reaches PASS or a pooled regime row reaches PASS before B2's live wiring is enabled beyond shadow.
- **B2 (desk, shadow):** DailyPlan, InPlayRanker, AdaptivePlaybook, journal, endpoints, engine wiring, nightly playbook update in cron.
- **B3 (optional, GPU):** LLM briefing/critique.

## 10. Testing

Pure modules tested with synthetic frames (context columns, ranking, plan rules, playbook rules, regime lookup precedence, gate with regime, family golden sets). Engine wiring via source-level tests plus import smoke. Determinism test extended to the context builder. Endpoint tests offline. Existing failing-test baseline must not grow.

## 11. Risks

- Cross-sectional context needs the nightly export to be complete; a missing day degrades to NaN and families skip.
- `eps_surprise_pct` depends on yfinance data quality; PEAD family fails closed when NaN.
- Regime labels are daily; intraday regime flips are not captured (acceptable for v1).
- More lanes and a desk mean more audit volume; the monitor summarizes rather than lists.
