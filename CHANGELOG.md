# Changelog

## 2026-09-28

### Go-live with a $5,000 allocation
- feat(risk): `BOT_CAPITAL_ALLOCATION` (`core/config.py` § `v-go-live-5k-2026-09-28`, default `0.0` = off)
  caps `RiskManagerWithCommentary.effective_balance()`/`.effective_buying_power()`
  (`risk/manager.py:185-190`, `risk/allocation.py`) at a hard dollar slice of the
  brokerage account, applied at read time so no live write site (signal router,
  buying-power check, `/ws` refresh) can restore the whole-account numbers.
  Sizing, the buying-power cap, and the daily-loss circuit all read the capped
  value; a new allocation-headroom check trims or zeroes a position's notional
  (`allocation_cap` / `allocation_exhausted`) rather than letting bot-managed
  open notional exceed the allocation. `MAX_DAILY_LOSS` and `RISK_PER_TRADE_PCT`
  gained env overrides restricted to `(0, 1]` (`_env_fraction`) specifically so
  a legacy dollar-valued `.env` line (`MAX_DAILY_LOSS="500"`) could not silently
  disable the circuit. Commits `c3ee3c2`, `658a0b0`.
- feat(fit_lane): `FIT_LANE_LIVE_ALLOWLIST` (`core/config.py`, same section) lets
  named `(family, side[:regimes])` lanes trade live without a walk-forward PASS
  row (`strategies/fit_lane_admission.py`, new). A row blocks the allowlist path
  only when it is *measured and losing* (`n ≥ 60`, `exp_r < 0`); absent or thin
  rows — most of the universe today — are admitted, sized on
  `FIT_LANE_ALLOWLIST_STRENGTH` (default `0.6`) against `RISK_PER_TRADE_PCT`
  rather than a score-derived Kelly haircut. `core/strategy_fit_bars.py::load_context_slice`
  + `strategies/fit_lane_frames.py::align_context` (`CTX_MAX_CARRY_SESSIONS=3`)
  give factor families (`mom_12_1`) their first live context feed, carrying a
  session forward at most 3 business days before standing down. Every
  downstream gate (desk limits, playbook, swing overnight, risk sizing, the
  allocation cap) still applies. Commits `b41c5e9`, `758c28f`, `81a9e60`,
  `af8043c`, `60235f4`.
- chore(strategy-hunt): `scripts/nightly_strategy_hunt.sh` (commit `e609a52`)
  runs the full `export → news → earnings → context → calibrate → run →
  run(shuffle) → score --postgres → playbook` pipeline after the close;
  installed via crontab at 22:30 ET Monday–Friday, superseding the earlier
  22:00 line that invoked `research.strategy_hunt all` directly.
- ops: ledger-integrity migration (`scripts/migrate_ledger_integrity.py`)
  applied to Postgres `bot_trades` 2026-09-28 09:30 ET during a quiet window;
  `LEDGER_INTEGRITY=1` set at the go-live restart, switching on broker-orphan
  reconciliation, immutable initial-stop/TP tracking, and `pnl_r`-derived R
  instead of the closing-stop approximation.
- go-live `.env`: `BOT_CAPITAL_ALLOCATION=5000`, `RISK_PER_TRADE_PCT=0.01`,
  `MAX_DAILY_LOSS=0.03`,
  `FIT_LANES_ENABLED=dt_momentum,gap_play,vwap_reversion,mean_rev_intraday,mr_20d,mom_12_1`,
  `FIT_LANE_LIVE_ENTRIES_ENABLED=mr_20d,mom_12_1`,
  `FIT_LANE_LIVE_ALLOWLIST="mr_20d:long,mom_12_1:long:risk_on|mixed"` (quoted —
  an unquoted `|` breaks any shell that sources `.env`),
  `FIT_LANE_ALLOWLIST_STRENGTH=0.8`, `SWING_ALLOW_OVERNIGHT=1`,
  `SWING_MAX_HOLD_SESSIONS=20`, `ENABLE_DESK_LIMITS=1`, `ENABLE_PLAYBOOK_GATE=1`,
  `LEDGER_INTEGRITY=1`; `STRATEGY_FIT_SHADOW` stays on (the allowlist is the
  sleeve's gate, not the enforcing walk-forward gate). Docs:
  `docs/BOT_OPERATIONS_MANUAL.md` § "Live allocation and the sleeve lanes",
  `docs/desk.md` (flags table + day-one checklist),
  `docs/research/strategy_hunt.md` § "Scheduling" and § "Universe context".
- **Go-live executed.** Live since 11:52 ET (after-close gate lifted at the
  user's request; no bot-managed positions open at the time). Restarted again
  13:13 ET onto `d66380d` once the engine-wedge fix below cleared review; pid
  1300568, live mode, sleeve lanes `live=True`, allocation active, watchdog
  health ok. Backups: `.env.bak_pre_golive_20260928`,
  `Config().yaml.bak_pre_golive_20260928`.
- **Losing-day streak reset.** `DESK_STREAK_SINCE=2026-09-28` (commit
  `bd54992`) floors the desk's `two_losing_days` sit-out lookback at go-live
  day, at the user's request — sessions before it never count toward the
  streak. Today's plan was rebuilt with streak 0 / prior-day R 0. Docs:
  `docs/desk.md`.
- **Nightly cron.** `scripts/nightly_strategy_hunt.sh` installed at 22:30 ET
  Monday–Friday (see the "chore(strategy-hunt)" entry above); a 09:49–10:06 ET
  test run on 2026-09-28 completed clean (rc 0, 17 min).
- **fix(engine): the engine-wedge fix landed.** Bounded every synchronous
  provider call reachable from the engine's event loop
  (`core/provider_calls.py`: `provider_call`/`provider_call_sync` for market
  data, `broker_call`/`broker_call_sync` for the 20 Schwab order/account call
  sites, raising `BrokerTimeoutError` on timeout instead of returning `None`);
  moved all yfinance usage into a spawn-context subprocess pool that is
  killed and rebuilt on a timeout (`core/isolated_calls.py`); added a
  per-loop heartbeat that logs `loop_lag` when a loop stalls
  (`core/loop_heartbeat.py`); and added an external restarter,
  `scripts/bot_watchdog.sh`, now cron-installed live (not `--dry-run`) every
  5 minutes, 06:00–19:55 ET Monday–Friday, which restarts the bot on two
  consecutive unhealthy `/api/status` checks (capped at 2 restarts/day,
  otherwise `OPERATOR ACTION REQUIRED`). The screener's quality gate now
  fails closed on missing/short/erroring data instead of passing symbols
  through, and the close-timeout audit
  (`zombie_pending`/`broker_timeout_close`) says plainly that a position
  whose close call timed out has no broker-side protection and needs
  operator action. Diagnosed after the live process wedged on 2026-09-28 (247
  of 251 threads parked in `futex_wait`, `/api/status` unresponsive 60+ s) —
  root cause a GIL-holding yfinance/curl_cffi hang. Commits `066174e`,
  `feba6ca`, `56bf7fd`, `a5382bc` (tests pinned at `d66380d`). Full
  round-by-round facts and RED/GREEN evidence:
  `.superpowers/sdd/2026-09-28-go-live-5k/task-4-report.md`. Docs:
  `docs/BOT_OPERATIONS_MANUAL.md` § 9.8.

## 2026-09-26

### Strategy search batch 2 — selection control, limit execution, factor sleeve (2026-09-26)
- Selection baseline (random-peer replay of the real trade schedule; tests stock selection rather than timing) with per-group pairing counts and `baseline_underreplayed`; limit-at-open execution style (maker fills, upper bound); grid pins for pinned A/B arms (unknown keys rejected); factor context columns (`mom_12_1`, `mom_rank_12_1`, `vol_rank_60`) and monthly factor families (`mom_12_1`, `low_vol_trend`; registry 20); `time_stop` on trade records; `promote` blocks non-market entry styles; `splits-audit` subcommand.
- Data fix: split adjustment now trusts the yfinance cache and requires persistence before applying a detected split — false detections had compounded (LCID ÷1e8, LRCX 10× off) and inflated rank-based families (`rs_rotation` +0.107 → +0.03 R). Contexts and all roots re-run.
- Result: 0 PASS in both arms. Limit entries add +0.00–0.02 R while filling 60–75 % of signals (execution is not the gap). Against random-peer selection the best rows are `mr_20d long/risk_off` (+0.129 R excess, n 302) and `mom_12_1 long` (+0.043, n 1,302), neither significant; momentum/mean-reversion shorts are worse than random. Report: `research/reports/strategy_hunt/2026-09-26-strategy-search-batch2.md`.

## 2026-09-25

### Strategy search batch 1 — slower families (2026-09-25)
- Five new pure families (`strategies/families/swing2.py`: overnight_hold, mr_short_term, multi_day_drop, pead_wide, rs_rotation), a simulator `overnight` hold kind (exit at the next session's open; earnings-eve overnight entries skipped), `--holdout-months` (OOS windows end early; the withheld period is opened only for confirmation) and a "Search accounting" report section (families × grid points × sides vs PASS).
- Result: 0 PASS (3,467 rows, 58 WATCH; pooled 0/4/20). The slower families are flat, not negative, but indistinguishable from their shuffled baselines; `rs_rotation` (+0.107 R) matches its baseline (+0.086 R) — stock drift, not timing. Report: `research/reports/strategy_hunt/2026-09-25-strategy-search-batch1.md`. Registry now 18 families; the main-root snapshot (all 18, holdout 0) was re-scored at 22:58 ET (0 PASS / 195 WATCH).
- Guards: `promote`, `core/swing_exits` and the fit-lane adapter treat `overnight` like `multiday` for `SWING_ALLOW_OVERNIGHT`; no new lane is live.

- feat(desk): Playbook Desk B2 (shadow) — a human trader's daily routine, running shadow-only
  behind four flags that all default off (`ENABLE_DESK`, `ENABLE_DESK_WATCHLIST`,
  `ENABLE_DESK_LIMITS`, `ENABLE_PLAYBOOK_GATE`; `core/config.py` § `v-playbook-desk-b2-2026-09-25`).
  Pure decision modules under `core/desk/`: `daily_plan.py` builds a pre-market `DailyPlan` (sit-out
  day, risk budget, max trades, long/short bias, all keyed by regime) and its router rule
  `evaluate_desk_gate`; `inplay_ranker.py`/`inplay_features.py` score a 09:35 in-play list from
  opening relative volume, gap, catalyst and 20-day relative strength; `playbook.py` is the adaptive
  playbook — `update_entry` transitions each `(family, side, regime)` key between `suspended`
  (0.0x), `probation` (0.5x) and `enabled` (1.0x, or 1.25x once `exp_R >= 0.15` over `n >= 40` trades
  *and* a PASS/WATCH fit verdict agree), `carry_forward` lapses a 30-day-silent suspension back to
  probation, and `evaluate_playbook_gate` is its router rule; `journal.py`/`journal_data.py` render
  a post-close markdown journal (plan, decisions, trades, expectancy by family x regime, playbook
  changes, a derived "what to change" section) to `data/desk/journal/<date>.md`. `core/loops/desk_loop.py`
  (+`desk_steps.py`, `desk_trade_stats.py`) drives the three ET crossings — 09:00 plan, 09:35
  in-play refresh, 16:05 journal — as a supervised loop registered only under `ENABLE_DESK`, every
  step wrapped so a failure logs once a day rather than killing the loop or retrying every 30s. Two
  router hooks bracket the existing strategy-fit gate in `core/engine.py`: `desk_limit` (component
  `desk_limit`) immediately before it, `playbook` (component `playbook`) immediately after, writing
  `signal.reasoning["strategy_size_multiplier"]` only when `ENABLE_PLAYBOOK_GATE` is on
  (`risk/manager.py::playbook_size_multiplier`, clamped `[0.0, 1.5]`); with their own flags off,
  every blocking rule audits `action=shadow` and blocks nothing. `core/loops/screener_loop.py` stops
  overwriting `engine.dynamic_watchlist` once `ENABLE_DESK_WATCHLIST` is on and today's plan has
  in-play rows. A nightly `research.strategy_hunt playbook` subcommand (`research/strategy_hunt/playbook_update.py`,
  `playbook_live.py`, `playbook_cli.py`) rewrites `data/desk/playbook.json` from resolved shadow
  ledgers plus, with `--postgres`, the live `bot_trades` tape; `all` now ends with it
  (`export -> news -> earnings -> context -> calibrate -> run -> run(shuffle) -> score -> playbook`).
  Three read-only endpoints (`GET /api/desk/plan`, `/api/desk/playbook`, `/api/desk/journal`, never
  500) and a "## Desk" `tools/bot_monitor.py` section (before "## Strategy fit"). Docs:
  `docs/desk.md` (new), `docs/research/strategy_hunt.md` § "The `playbook` subcommand",
  `docs/BOT_OPERATIONS_MANUAL.md` § "Desk (shadow)" and the flags table.

  **Research posture: shadow-only, deliberately.** No family has a walk-forward PASS in any regime
  as of the latest full-universe run (`research/reports/strategy_hunt/2026-09-25-strategy-hunt-b1-5.md`
  § B1.6) — B1/B1.5's "regime split is real and large" finding was withdrawn once the regime label
  was corrected to entry-time information (see the B1.6 entry below). The desk's regime-keyed
  `risk_budget_pct`/`max_trades` rules are therefore defensive risk rules, not evidence of an edge;
  the playbook learns per `(family, side, regime)` from shadow and live results precisely because no
  research result yet favors one key over another. `ENABLE_DESK_LIMITS`/`ENABLE_PLAYBOOK_GATE`
  remain off; Task 8 of the plan turns on `ENABLE_DESK` alone, after the close, so the desk starts
  building plans and writing journals in shadow without governing any signal.
- fix(strategy-hunt): Playbook Desk B1.6 — the research regime label is now the one the desk could
  have known at entry. `research/strategy_hunt/data/regime.py` built its label from the entry
  *day's* SPY and VIX **closes**, so a 10:00 entry was stamped with a regime decided six hours
  later and every regime-conditioned/pooled row (and the live gate reading them) was fitted to the
  future. It is replaced by a per-run `RegimeLookup` over the root's 5-minute SPY bars plus daily
  VIX: `spy_change_pct` is SPY's day-to-date move through the last bar that closed at or before the
  entry bar's open, `vix_level`/`vix_change_pct` are the prior session's close and close-to-close
  change (a documented lagged proxy — the store holds daily VIX only), classified through the same
  thresholds as `core/market_context.py::_classify_regime` with a parity test pinning the two
  together. The shuffle control shares the same lookup, since SPY is never shuffled. Alongside it:
  a `baseline_population_mismatch` pooled failure key when the shuffled control group's trade count
  is outside 0.5x–2x the real row's `n`, with `baseline_n` rendered in the report's pooled table and
  counted in the report manifest; `StrategyFit.lookup_regime` now lets a universe-pooled row fill a
  gap or stand in front of a WATCH but **never** override the symbol's own measured FAIL for that
  lane; `all` rejects `--shuffle-seed` with a parser error (it runs the fixed-seed baseline itself,
  and a user seed meant `trades/` was never written); `promote` writes its markdown under
  `reports_dir(root)` (`research/reports/strategy_hunt/strategy_hunt_promotion.md` for the default
  root) instead of unconditionally into `docs/research/`; PEAD's entry bar is corrected to
  `bar_of_day == 5` so the signal is the bar closing 10:00 and the fill is the 10:00 bar's open,
  with `PEAD_TARGET_R` documented as an addition to the spec; and `indicator_frame`'s context join
  NaNs `ctx_rvol_open5`/`ctx_rvol_rank5` at `bar_of_day == 0` and `ctx_rvol_open15` at
  `bar_of_day < 3`, making the known-at-open contract mechanical. Docs:
  `docs/research/strategy_hunt.md` § "Regime and pooled rows",
  `docs/BOT_OPERATIONS_MANUAL.md` § "Strategy fit gate".
- docs(strategy-hunt): B1.6 rerun of the full universe under the corrected labels
  (`research/reports/strategy_hunt/2026-09-25T1557-strategy-hunt.md` + json + equity CSV; narrated
  in `research/reports/strategy_hunt/2026-09-25-strategy-hunt-b1-5.md` § "B1.6 — regime label fix").
  Same tape (335,495 real / 479,357 shuffled trades), re-labelled: **0 PASS rows anywhere** (0 of
  9,227 unconditional, 0 of 1,625 symbol-regime, 0 of 67 pooled — pooled PASS 0 / WATCH 6 / FAIL 61,
  was 0 / 14 / 55). B2 stays **NO-GO**. The substantive result is that B1/B1.5's headline finding is
  **withdrawn**: the "regime split is real and large" was the look-ahead. `dt_momentum short`'s
  risk_off-minus-risk_on spread collapses from +0.63 R to 0.00 R; the two rows B1.5 named as its
  best candidates flip sign (`dt_momentum short/risk_off` +0.172 → −0.275 R,
  `gap_play short/risk_off` +0.181 → −0.152 R, the latter going from "fails one floor" to ten).
  `mixed` now holds 72.7% of trades (was 47.2%), since at 10:00 most sessions have not yet moved
  ±0.3%. One non-circular lead survives: `gap_play long` earns +0.114 R in risk_on and +0.060 in
  risk_off against −0.119 in mixed, with excess +0.098 / +0.154 over the shuffled control — WATCH,
  not PASS. Nine of 67 pooled rows carry the new `baseline_population_mismatch` (all already FAIL on
  other floors); it shows `orb` firing 3–6x as often on structure-destroyed bars, which makes its
  earlier excess numbers uninterpretable.
- feat(strategy-hunt): full strategy-fit system — shared candidate-family library
  (`strategies/families/`: ten intraday and swing families, common entry/grid/de-dup helpers) used by
  both the offline research pipeline and the live engine; the offline walk-forward pipeline
  (`research/strategy_hunt/`, subcommands export/context/news/earnings/calibrate/run/score/report/all)
  with a realistic cost model (Corwin-Schultz spread, volume-scaled slippage with a calibration
  override, sell fees, 3% short borrow), split-adjusted bars, per-symbol fit table with t-stat/CI
  floors, cold-start classifier fallback, and a portfolio replay of the PASS lane against the bot's
  real sizing rules; a strategy-fit gate in the live engine (`core/strategy_fit_gate.py`, wired after
  the regime gate) that is off by default and shadow-only (`STRATEGY_FIT_SHADOW=True`) when enabled;
  swing-lane multi-session exit support (`SWING_ALLOW_OVERNIGHT`, `SWING_MAX_HOLD_SESSIONS`); fit-lane
  adapters (`strategies/fit_lane.py`) that are shadow-first per family via `FIT_LANES_ENABLED` /
  `FIT_LANE_LIVE_ENTRIES_ENABLED` and emit a live signal only for an entry whose own
  (symbol, family, side) fit row is a walk-forward `PASS` — a `WATCH`, `FAIL`, or cold-start row
  stays shadow (ledger line plus a `fit_lane_shadow_verdict` decision) even for a family listed in
  `FIT_LANE_LIVE_ENTRIES_ENABLED`; a "Strategy fit"
  section in `tools/bot_monitor.py`'s report; and a read-only `GET /api/strategy-fit` endpoint. Nightly
  cron scheduling and full docs in `docs/research/strategy_hunt.md` and
  `docs/BOT_OPERATIONS_MANUAL.md` § "Strategy fit gate". Plan:
  `docs/superpowers/plans/2026-09-25-strategy-hunt-full-system.md`; spec:
  `docs/superpowers/specs/2026-09-24-strategy-hunt-design.md`.
- feat(strategy-hunt): Phase 3 — Stage A shadow scorecards (`research/strategy_hunt/shadow_resolver.py`,
  `cli.py scorecard`) that resolve a family's live shadow ledger (`data/shadow_fit_<family>.ndjson`)
  into simulated trades and a GO/NO_GO verdict; and a promotion command
  (`research/strategy_hunt/promote.py`, `cli.py promote`) that combines the fit snapshot's per-family
  PASS counts, each family's latest scorecard, and the live-facing flags into a promotion table with a
  blocker/recommendation per family (`collect_shadow`/`register_shadow`/`promote_live`/`hold`), printing
  the exact `.env` lines to apply and writing the promotion markdown (moved to
  `research/reports/strategy_hunt/strategy_hunt_promotion.md` in B1.6) — never editing
  `.env` or `Config().yaml` itself. Operator flow (nightly `all` → `scorecard` → `promote` → apply →
  restart via `./deploy.sh` → rollback by unsetting the flag) documented in
  `docs/research/strategy_hunt.md` § "Operator flow"; fit-lane row and the eight strategy-fit flags
  added to `docs/BOT_OPERATIONS_MANUAL.md` §§ 2.4 and 6.4.
- feat(strategy-hunt): Playbook Desk B1 — trade the tape the way a desk does, offline first.
  A **universe context builder** (`research/strategy_hunt/context.py`, `context` /
  `universe-context` subcommands) writes `data/strategy_hunt/context/universe_daily.parquet`: per
  (symbol, session) RVOL, gap, relative strength, distance from the 52-week high, ADV bucket, ATR%,
  earnings/EPS-surprise and 24-hour news features plus their same-date cross-sectional ranks, joined
  onto each symbol's indicator frame as `ctx_*` columns. Everything but the opening-bar RVOL/gap
  columns is lagged to the prior close, and those are only read from `bar_of_day >= 1`, so the whole
  set is known at open. Three **in-play families** (`strategies/families/inplay.py`; 13 families
  total): `orb_inplay` (opening-range breakout on the session's top-RVOL names, `trail_r`
  approximated as a take-profit since the simulator has no trailing stop), `pead_smallcap`
  (post-earnings drift for small/mid caps, gated on the new EPS surprise column) and `rs_leader`
  (relative-strength leader pullback-resume). **Regime-conditioned fit rows** and
  **universe-pooled (family, side, regime) rows** (`research/strategy_hunt/score/regime_rows.py`)
  are written into the snapshot by `score` and read by `core/strategy_fit.py::lookup_regime` in the
  precedence symbol-regime → pooled → unconditional; a regime needs ≥ 60 trades to get a row at all
  (a thin sample could only manufacture a FAIL that outranks better evidence), pooled rows use
  stricter floors (`POOLED_FLOORS`: n ≥ 500, PF ≥ 1.2, t_stat ≥ 3.0) and bucket stability/decay by
  calendar quarter. The live gate consults them when `STRATEGY_FIT_REGIME_ROWS` (new, default True)
  is on and the market-context regime is known, adding `regime=`/`pooled=` to its audit line; the
  gate itself stays off and shadow-only by default. A **pluggable sentiment backend**
  (`core/sentiment_backend.py`, `SENTIMENT_BACKEND=vader|finbert`, `SENTIMENT_FINBERT_PATH`,
  `scripts/prepare_finbert.py`) scores research and live news through one code path, with
  `news --rescore` to backfill `sentiment_vader`/`sentiment_finbert` on the cached corpus, and
  earnings surprises (`eps_estimate`, `reported_eps`, `surprise_pct`) added to the earnings cache.
  `all` now orders `export → news → earnings → context → calibrate → run → score` so the context
  sees tonight's news and earnings.
  **Full-universe B1 run (2026-09-25):** 539 symbols, 13 families, 340,211 out-of-sample trades over
  2022-01 → 2026-09, news corpus 100% FinBERT-scored (725,793 rows), ~27 min wall-clock. **0 PASS rows
  anywhere** — 0 of 9,505 unconditional rows (127 WATCH), 0 of 1,390 symbol-regime rows (159 WATCH),
  0 of 69 pooled regime rows (15 WATCH) — so the spec's B1 gate ("at least one family reaches PASS or a
  pooled regime row reaches PASS") is **not met and B2 live wiring stays shadow-only**. The regime split is
  the run's real finding (dt_momentum short: +0.172 exp_R in `risk_off` vs −0.457 in `risk_on` on ~13k/31k
  trades); two floor caveats are recorded in the report: pooled `max_dd_r` is not scale-invariant (40 of 69
  pooled rows fail it) and the structure-destroying shuffle control produced one pooled PASS where the real
  tape produced none. Run report:
  `research/reports/strategy_hunt/2026-09-25-strategy-hunt-b1.md`; docs:
  `docs/research/strategy_hunt.md`; spec:
  `docs/superpowers/specs/2026-09-25-playbook-desk-design.md`; plan:
  `docs/superpowers/plans/2026-09-25-playbook-desk-b1.md`.
- feat(strategy-hunt): Playbook Desk B1.5 — make a pooled regime PASS mean "edge beyond the regime's own
  drift". `FitMetrics` gains **`t_stat_session`** (trades clustered by entry session, the independent unit,
  instead of treating hundreds of same-session trades across symbols as iid) and **`recovery`**
  (`net_R / max_dd_R`); `POOLED_FLOORS` drops the absolute `max_dd_r` cap — not scale-invariant, a 13k-trade
  pooled row reaches a 639 R drawdown by arithmetic alone — for `min_recovery 3.0`, and adds
  `min_t_stat_session 3.0`. The **shuffled control becomes an in-root baseline variant**: `run --shuffle-seed N`
  writes `trades_shuffle/` / `choices_shuffle/` / `checkpoints_shuffle/` beside the real tape over the full
  universe, `all` runs it with seed 7 right after `run` (`--no-baseline` skips it), and `score` pools both
  tapes in one pass so every pooled row carries `baseline_exp_r`, `baseline_n` and
  `excess_r = exp_r − baseline_exp_r`, with a new **`min_excess_r 0.05`** floor. A row with no baseline records
  `no_baseline`, and a baseline that is present but covers < 95 % of the real symbols or ends more than 7 days
  early is flagged `baseline_mismatch` / `baseline_stale`; all three permit WATCH and forbid PASS, so a stale
  or missing control can never mint a pooled PASS. The report gains a "Pooled rows vs shuffled baseline"
  section, counts pooled verdicts for each cost control, and machine reports are **run-stamped**
  `<date>T<HHMM>-strategy-hunt.md` so two runs in a day no longer overwrite each other; the run manifest keys
  the baseline run under `run_shuffle` / `errors_shuffle`. Research news scoring went **back to VADER** for
  parity with the live news loop (`sentiment_finbert` is kept in the cache for a future FinBERT-specific
  threshold grid).
  **B1 run re-scored (2026-09-25):** 539 symbols, 13 families, 335,499 real + 479,354 shuffled out-of-sample
  trades, ~19 min wall-clock, baseline guard clean (69 of 69 pooled rows benchmarked). **Still 0 PASS rows
  anywhere** — 0 of 9,231 unconditional (127 WATCH), 0 of 1,390 symbol-regime (159 WATCH), 0 of 69 pooled
  (14 WATCH) — so the B1 gate, now amended to "a pooled PASS counts only with positive excess over the
  shuffled baseline", remains **not met and B2 live wiring stays shadow-only**. B1's closest row
  (`dt_momentum short/risk_off`, which failed only `max_dd_r`) clears the new recovery floor at 3.60 but fails
  both new tests: session-clustered t +0.56 against an iid t of +3.85, and excess +0.030 because the shuffled
  tape reproduces +0.142 of its +0.172 expectancy. Of B1's 15 pooled WATCH rows 14 survive, 10 keep positive
  excess, 3 clear the +0.05 floor. The best row in the run, `gap_play short/risk_off`, fails `min_decay` alone
  — its last four quarters average −0.336 R against +0.181 R full-sample. Run report:
  `research/reports/strategy_hunt/2026-09-25-strategy-hunt-b1-5.md`; docs: `docs/research/strategy_hunt.md`;
  plan: `docs/superpowers/plans/2026-09-25-playbook-desk-b1-5.md`.

## 2026-09-21 — fix: Day-trade RSI missing from reasoning + breakout hard veto (v-rsi-breakout-veto-2026-09-21)

### Problem

P0 LIVE churn incident observed 2026-09-21 on META. Day-trade breakout entry at RSI 86.09 filled, then ~4s later closed by `open_desk_rsi_extreme_overbought` (scratch churn). Same class as XE RCA 2026-09-17.

The engine's `DAY_TRADE_RSI_HIGH_ENTRY_VETO_ENABLED=True` gate did NOT fire despite being enabled.

### Root Cause Analysis

**Missing RSI in reasoning**: Engine veto does `_reasoning.get('rsi', 0)` but `DayTradeMomentumStrategy.TradingSignal.reasoning` did NOT include the `'rsi'` key. Missing RSI defaulted to 0, so the veto condition `_signal_rsi >= 70` was never true.

**Breakout pattern not covered**: Strategy-level hard veto (`ENABLE_HARD_VETO_CONTINUATION_RSI70`) only checked `_entry_pattern == "continuation"`, NOT `"breakout"`. So breakout@RSI86 bypassed both strategy veto AND engine veto.

### Added

- **`'rsi': rsi`** in `TradingSignal.reasoning` for both:
  - `DayTradeMomentumStrategy` (long)
  - `DayTradeMomentumShortStrategy` (short)
  
  Now engine veto can read the actual RSI value instead of defaulting to 0.

- **`ENABLE_HARD_VETO_BREAKOUT_RSI70`** config flag (default **True**)
  - When True: if entry_pattern==breakout AND rsi >= 70, return None (skip order)
  - When False: legacy (engine veto must catch it — now possible since RSI in reasoning)
  - Override: `ENABLE_HARD_VETO_BREAKOUT_RSI70=0` env or `trading.enable_hard_veto_breakout_rsi70: false`

- **Hard veto block** for breakout + RSI>=70 in ALL regimes
  - Logs strategy_decision action=hard_veto with reason `breakout_overbought_all_regimes`
  - Commentary: "🚫 Hard Veto: BREAKOUT + RSI≥70 (all regimes)"
  - Modular: separate flag from continuation veto for fine-grained control

- **Tests** in `tests/test_day_trade_momentum.py`:
  - `TestDayTradeMomentumRsiInReasoning`: verifies RSI key in long/short reasoning
  - `TestHardVetoBreakoutRSI70Config`: config flag defaults and env override
  - `TestHardVetoBreakoutRSI70Strategy`: breakout+RSI>=70 blocked, flag off = legacy, RSI<70 passes

### Changed

- `strategies/builtin.py`:
  - `DayTradeMomentumStrategy.generate_signal_with_commentary`: added RSI to reasoning + breakout hard veto
  - `DayTradeMomentumShortStrategy.generate_signal_with_commentary`: added RSI to reasoning

- `core/config.py`:
  - Added `ENABLE_HARD_VETO_BREAKOUT_RSI70` property

### Unchanged

- LIVE entry flags untouched (DAY_TRADE_LIVE still paused per CoS)
- Mean-rev/ORB/short LIVE flags unchanged
- HANDS_OFF_DENYLIST (MU/HQGE/SPCX) unchanged
- `feature/trading_bot_v1` and PR #14 untouched
- Engine `DAY_TRADE_RSI_HIGH_ENTRY_VETO_ENABLED` logic unchanged (now works correctly since RSI present)
- Existing `ENABLE_HARD_VETO_CONTINUATION_RSI70` unchanged

### Flags Summary

| Flag | Default | Effect |
|------|---------|--------|
| `ENABLE_HARD_VETO_BREAKOUT_RSI70` | True | Strategy blocks breakout+RSI>=70 |
| `ENABLE_HARD_VETO_CONTINUATION_RSI70` | True | Strategy blocks continuation+RSI>=70 |
| `DAY_TRADE_RSI_HIGH_ENTRY_VETO_ENABLED` | True | Engine blocks breakout/continuation+RSI>=70 |

With all defaults, both strategy AND engine block overbought day-trade entries (defense in depth).

---

## 2026-09-21 — fix: Day-trade RS premarket mismatch + off_hours hard-skip (v-same-basis-rs-2026-09-21, v-daytrade-offhours-skip-2026-09-21)

### Problem

P1 reliability issue observed 2026-09-21 pre-market/morning. Two related bugs in day-trade momentum:

**Bug 1: RS basis mismatch**
- LIVE day-trade longs almost all skipped `weak_relative_strength` pre-open
- Example: symbol_change ~0 to ±0.1% while spy_change ~+0.6% → rs_vs_spy ~-0.6%
- Symbols like META/TSLA/NVDA logged many `signal_buy` with high rs_vs_spy earlier same morning
- As SPY day% rose, those flipped to weak_RS skips despite price barely moving

**Bug 2: off_hours signal noise**
- Strategy generated `signal_buy` logs with `time_of_day=off_hours` during premarket
- Engine's market_hours gate blocks LIVE orders, but signals still created noise with corrupt data

### Root Cause Analysis

**RS mismatch**: RS calculation compared mismatched bases:
- SPY day% used `netPercentChange` from Schwab (day vs prior close via MarketIndicesCache)
- Symbol day% used `(bar_close - bar_open) / bar_open` which is **bar % change**, not **day % change**

When SPY moved +0.6% from yesterday's close but a symbol's current 5-min bar only moved +0.1%, the RS calculation showed -0.5% (weak RS) — false skip.

**off_hours signals**: Strategy generated signals during off_hours (premarket 4-9:30 ET) when indicator data is unreliable. While engine gate blocked LIVE orders, the signals created false RS skip logs and noisy data.

### Added

- **`ENABLE_SAME_BASIS_RS`** config flag (default **True**)
  - When True: RS uses symbol's `netPercentChange` from Schwab (same basis as SPY)
  - When False: legacy bar-based estimate `(close - open) / open`
  - Override: `ENABLE_SAME_BASIS_RS=0` env or `trading.enable_same_basis_rs=false`

- **`DAY_TRADE_HARD_SKIP_OFF_HOURS`** config flag (default **True**)
  - When True: day_trade_momentum returns None during `time_of_day=off_hours`
  - When False: legacy (signals generated, engine gate still blocks LIVE)
  - Override: `DAY_TRADE_HARD_SKIP_OFF_HOURS=0` env or yaml

- **`get_symbol_day_change_pct()`** helper in `core/market_context.py`
  - Fetches symbol day % from Schwab `netPercentChange` (same source as SPY)
  - 30-second module-level cache to avoid excessive Schwab calls
  - Falls back to bar-based estimate if Schwab unavailable
  - Returns `(day_change_pct, source)` where source is one of:
    `schwab_quote`, `cache`, `bar_estimate`, `unavailable`

- **weak_relative_strength log** now includes `rs_source` and `same_basis_enabled` fields
  for audit trail showing which RS method was used

- **Tests** in `tests/test_day_trade_momentum.py`:
  - `TestSameBasisRS`: config flag default, helper function, strategy integration
  - `TestDayTradeOffHoursHardSkip`: off_hours hard-skip for long and short

- **Tests** in `tests/core/test_market_context.py`:
  - `TestGetSymbolDayChangePct`: schwab_quote source, bar_estimate fallback, caching

### Changed

- `strategies/builtin.py`:
  - `DayTradeMomentumStrategy.generate_signal_with_commentary`: same-basis RS + off_hours skip
  - `DayTradeMomentumShortStrategy.generate_signal_with_commentary`: same-basis RS + off_hours skip

- `data_providers/schwab.py`:
  - `get_quote()` now returns `netPercentChange` field for same-basis RS

- `core/config.py`:
  - Added `ENABLE_SAME_BASIS_RS` property
  - Added `DAY_TRADE_HARD_SKIP_OFF_HOURS` property

### Unchanged

- LIVE entry flags untouched (DAY_TRADE_LIVE, MEAN_REV, ORB, SHORT)
- HANDS_OFF_DENYLIST (MU/HQGE/SPCX) unchanged
- `feature/trading_bot_v1` and PR #14 untouched

### Residual Risks

- **opening_30 window**: Still allows reduced-size signals during first 30 min (existing behavior)
- **Same-basis cache TTL**: 30-second cache means brief lag in symbol day% updates
- **Schwab API failure**: Falls back to bar-estimate if Schwab quote fails (logged as `fallback_bar` source)

---

## 2026-09-18 — fix: IREN hard_stop race — rebracket guard when exiting (v-rebracket-exiting-guard-2026-09-18)

### Problem

P1 reliability issue observed 2026-09-18 ~14:43 ET on IREN. Race condition during hard_stop close path:

1. `hard_stop_breached` → FSM transitions to EXITING
2. `_close_real_position` cancels working OCO bracket
3. Close SELL fills (position is now flat)
4. `order_monitor` sees `bracket_canceled` → calls `re_bracket_position`
5. `re_bracket` places NEW OCO while flat → REJECTED by Schwab

On Schwab the new OCO was rejected (never WORKING), but even without rejection, placing protection for a closed trade is incorrect.

### Root Cause

`handle_bracket_canceled` blindly called `re_bracket_position` on any cancel without checking if the cancel was owned by an intentional close path (hard_stop, take_profit, proactive exit).

### Added

- **`REBRACKET_SKIP_IF_FLAT_OR_EXITING`** config flag (default **True**)
  - Env override: `REBRACKET_SKIP_IF_FLAT_OR_EXITING=0` to disable
  - YAML: `trading.rebracket_skip_if_flat_or_exiting: false`

- **Exit-in-flight guard** in `handle_bracket_canceled`:
  - Skips `re_bracket_position` if FSM state is EXITING / CLOSED / ZOMBIE
  - Skips if local `position.quantity <= 0`
  - Skips if broker confirms qty == 0 (LIVE mode authoritative check)
  - Logs audit reason `rebracket_skipped_exit_in_flight` with skip_reason detail

- **12 new tests** in `tests/test_order_monitor.py`:
  - `TestRebracketExitingGuard`: config flag existence, defaults, env override, code inspection
  - `TestRebracketExitingGuardFunctional`: async integration tests
    - `bracket_canceled during hard_stop close → no re_bracket`
    - `bracket_canceled with qty=0 → no re_bracket`
    - `genuine orphan cancel with live qty → still re_brackets`
    - `flag disabled → always re_brackets (legacy behavior)`

### Changed

- `core/order_monitor/brackets.py`: `handle_bracket_canceled` now checks FSM state and qty before re_bracket
- `core/config.py`: new `REBRACKET_SKIP_IF_FLAT_OR_EXITING` property

### Unchanged

- LIVE entry flags untouched (DAY_TRADE_LIVE, MEAN_REV, ORB, SHORT, PAPER)
- Genuine orphan bracket cancels (live qty, not exiting) still re-bracket correctly
- HANDS_OFF_DENYLIST (MU/HQGE/SPCX) unchanged

---

## 2026-09-18 — fix: Correlation guard ignores unmanaged/hands-off positions (v-correlation-ignore-unmanaged-2026-09-18)

### Problem

Day-trade momentum emitted `signal_buy` on QCOM at ~13:09 ET (pullback, rs_vs_spy=1.13, volume_ratio=2.24). Engine received it in live mode, then `correlation_guard` skipped with:
`correlated_with_semis_and_chip_adjacent existing=['NVDA', 'MU']`

NVDA is an unmanaged/external hold (`managed_by_bot=false`). MU is a permanent hands-off long-term holding. Neither represents active bot risk, but correlation_guard was treating them as blocking positions.

### Added

- **`CORRELATION_IGNORE_UNMANAGED`** config flag (default **True**)
  - Excludes positions with `managed_by_bot=False` from correlation cluster checks
  - External/operator-managed holds don't block bot entries into same sector
  - Override: `CORRELATION_IGNORE_UNMANAGED=0` env var or `trading.correlation_ignore_unmanaged=false` yaml

- **`CORRELATION_IGNORE_HANDS_OFF`** config flag (default **True**)
  - Excludes HANDS_OFF_DENYLIST symbols (MU, HQGE, SPCX) from correlation cluster checks
  - Permanent hands-off positions don't block bot entries
  - Override: `CORRELATION_IGNORE_HANDS_OFF=0` env var or `trading.correlation_ignore_hands_off=false` yaml

- **Correlation filter logic** in `core/engine.py` correlation guard section
  - Constructs `_correlation_positions` set by filtering `active_positions`
  - Checks `managed_by_bot` and `HANDS_OFF_DENYLIST` when flags enabled
  - Uses filtered set for cluster overlap checks

- **10 new tests** in `tests/test_recent_fixes.py::TestCorrelationIgnoreUnmanaged*`
  - Config flag existence and defaults
  - Code inspection for filter logic
  - Functional tests: unmanaged NVDA + hands-off MU does NOT block QCOM
  - Functional tests: managed AMD STILL blocks QCOM
  - Flags disabled falls back to legacy all-positions behavior

### Changed

- `core/engine.py`: correlation guard now filters positions before cluster check
- `core/config.py`: two new config properties with env override support

### Unchanged

- LIVE entry flags untouched (DAY_TRADE_LIVE, MEAN_REV, ORB, SHORT, PAPER)
- HANDS_OFF_DENYLIST (MU/HQGE/SPCX) still protected for exit/management
- `feature/trading_bot_v1` and PR #14 untouched

---

## 2026-09-17 — fix: Book B contamination FP lexicon (v-book-b-contam-lexicon-2026-09-17)

### Problem

Book B Stage A contam_fp = 15% (3/20) FAIL vs floor ≤5%. Three FPs leaked because MiniLM flat softmax tipped `ai_compute` on Corning/oil wraps with `contamination_risk=0.0`:
- f59b23116fcb: "Why Corning Plunged Today"
- 9579cf0d1b4f: "The Oil Crisis Has Reached Costco's Motor Oil Aisle"
- 2d83a5133335: "Corning Rides on Expanding Partner Base: Will it Boost Prospects?"

### Added

- **Config-gated contamination lexicon** in `GPUNewsCriticModel._compute_contamination_risk`:
  - `GPU_CRITIC_CONTAM_LEXICON_ENABLE` (default False, safe off-path)
  - `GPU_CRITIC_CONTAM_LEXICON` list: corning, glw, nyse:glw, motor oil, oil crisis, crude oil, petroleum, opec, wti, brent, cohr, cien, aaoi
  - On headline/summary text hit: `contamination_risk = max(risk, 0.65)` UNLESS headline also has AI_COMPUTE_TRIGGERS (anthropic, openai, gpu demand, data center, ...)
  - Matches headline/summary text only (NOT symbols_seen dump) to preserve TP bf0c664141dc Nasdaq/AI Leaders

- **FakeCritic lexicon mirror** for CI honesty: same triggers in `FakeCritic.CONTAMINATION_TRIGGERS`

- **8 new unit tests** in `tests/test_gpu_news_critic.py::TestBookBContamLexicon`:
  - 3 FP eids → contamination_risk ≥ 0.6 + WOULD_SUPPRESS_HARD_SKIP when lexicon enabled
  - Anthropic/AI headline TPs unchanged
  - GLW with AI headline trigger not suppressed (TP protection)

### Unchanged

- `THEME_HARD_SKIP_REQUIRE_GPU=False` (not flipped)
- No LIVE knobs changed
- HANDS_OFF MU/HQGE/SPCX unchanged

---

## 2026-09-17 — fix: name failing pretrade checklist gates

- Verdict strip lists failing gate labels (not count-only) and appends last snapshot reason/gate.
- Mean-rev / ORB intentionally OFF no longer count as fail (skip) so they don't fake BLOCKED.

---

## 2026-09-17 — Day-Trade Momentum SHORT Strategy (SHADOW-FIRST)

### Problem

The desk can only go LONG. On weak/breakdown days, profitable short setups pass by while the bot sits idle waiting for long entries that won't work in a down-tape. Operator wants a real shorting algorithm, not discretionary freestyle.

### Added

- **DayTradeMomentumShortStrategy** (v-tag `v-day-trade-short-2026-09-17`)
  - Modular short path mirroring the long day-trade momentum logic with inverse filters.
  - Entry patterns:
    - **Breakdown**: price < 20-bar low with volume + ADX > 20
    - **Continuation-down**: RSI 30-50, price below SMA20, MACD bearish
    - **Rejection**: near 20-bar high but failing (weak close, MACD bearish)
  - Gates (inverse of long logic):
    - Weak RS vs SPY (symbol UNDERPERFORMING by >= 0.5%, configurable)
    - Direction reader bearish + non-exhausted phase
    - RSI floor 30 (don't short oversold — bounce risk)
    - RSI ceiling 85 (optional, blocks extreme overbought)
    - risk_on regime HARD BLOCK (don't short bullish tape)
    - Flatten-hour entry gate (same as longs)
  - ATR-based stop/target: stop ABOVE entry, target BELOW entry (1.5x ATR / 2:1 R:R).

- **SHADOW-FIRST deployment mode**
  - `DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED=False` (default) blocks LIVE short orders.
  - `ENABLE_DAY_TRADE_SHORT_SHADOW=True` (default) logs shadow entries to `data/day_trade_short_shadow.ndjson`.
  - Signal emitted for sim/commentary/shadow analysis without placing broker short orders.
  - Flip to LIVE only after Stage-A soak passes (n>=150, PF>=1.30, WR>=48%, etc.).

- **HANDS_OFF_DENYLIST protection**
  - MU, HQGE, SPCX are NEVER shorted — permanent hands-off regardless of setup quality.
  - Check runs at Gate 0 before any other processing.

### Config Flags

| Flag | Default | Description |
|------|---------|-------------|
| `ENABLE_DAY_TRADE_SHORT` | True | Enable short strategy for shadow soak |
| `DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED` | False | Master switch for LIVE short orders |
| `ENABLE_DAY_TRADE_SHORT_SHADOW` | True | Log shadow entries when LIVE disabled |
| `DAY_TRADE_SHORT_MIN_WEAK_RS_VS_SPY` | 0.5 | Min negative RS required (underperformance) |
| `DAY_TRADE_SHORT_RSI_FLOOR` | 30 | Don't short when RSI <= floor (oversold) |
| `DAY_TRADE_SHORT_RSI_CEILING` | 85 | Don't short when RSI >= ceiling |
| `DAY_TRADE_SHORT_MAX_CONCURRENT` | 2 | Max concurrent short positions |

### Changed

- `core/engine.py` — registers `DayTradeMomentumShortStrategy` when `ENABLE_DAY_TRADE_SHORT=True`.
- `strategies/builtin.py` — new `DayTradeMomentumShortStrategy` class (~350 lines).
- `core/config.py` — 8 new config properties for short strategy.

### Tests

- 16 new tests in `tests/test_day_trade_short.py` covering:
  - Config flag defaults (LIVE=False, SHADOW=True)
  - HANDS_OFF_DENYLIST protection (MU/HQGE/SPCX never shorted)
  - risk_on hard-block
  - RSI floor gate (oversold protection)
  - Weak RS filter (underperformance required)
  - Breakdown/continuation-down pattern detection
  - Shadow logging when LIVE disabled
  - LIVE signal generation when enabled
  - Direction reader gates

### Operational notes

- **SHADOW-FIRST**: No LIVE shorts by default. Desk can short on paper during Stage-A soak.
- **How to enable LIVE later**: Set `DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED=1` via env or `trading.day_trade_short_live_entries_enabled: true` in Config.yaml after Stage-A validation passes.
- **Does NOT change**: Long day-trade LIVE default, mean-rev LIVE default, ORB LIVE default. All existing live knobs unchanged.
- **HANDS_OFF unchanged**: MU, HQGE, SPCX remain permanently hands-off for both longs and shorts.
- Shadow ledger location: `data/day_trade_short_shadow.ndjson` (same format as `shadow_short_log.ndjson`).

---

Notable changes to the trading bot from project genesis (2025-07-08) to present. Each entry lists the date range, the user-facing impact, and where applicable the v-tag (greppable code anchor) or commit SHA.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) with added `Verified` (empirical outcomes from backtests/live) and `Operational notes` sections specific to trading-bot concerns.

**194 commits over 11 months.** Project paused 2025-09 through 2025-12 inclusive, resumed full-time 2026-01.

---

## 2026-09-17 — Universe Redesign: Pinned Watchlist + Dollar-Volume Demote

### Problem

WATCHLIST_SIZE=20 slots filled by Yahoo movers (~60 pass relaxed filter), but the final 20 was dominated by micro-cap/lottery names (PURR, AEMD, IOVA, NVAX-class) while quality filter rejected profitable liquid names (AMZN, GOOG, AVGO, SHOP, UBER). TSLA/NVDA often missed day-trade entries due to `weak_relative_strength` AND weren't even in the final 20. XE (junk RS winner) got in and lost.

### Added

- **PINNED_WATCHLIST** (v-tag `v-pinned-watchlist-2026-09-17`)
  - Config: `trading.pinned_watchlist` (list) or env `PINNED_WATCHLIST` (comma-separated).
  - Default: NVDA, TSLA, META, AMZN, MSFT, GOOGL, AVGO, AMD.
  - Pinned symbols always reserve slots in `WATCHLIST_SIZE` before Yahoo/Schwab movers fill remaining.
  - Respects `HANDS_OFF_DENYLIST` — pinned symbols on denylist appear for analysis but never auto-trade.
  - Enable/disable via `ENABLE_PINNED_WATCHLIST` (default True).

- **MIN_DOLLAR_VOLUME floor** (v-tag `v-dollar-volume-floor-2026-09-17`)
  - Config: `trading.min_dollar_volume` (default $10M).
  - Non-pinned movers MUST clear `price × avg_volume > $10M` to occupy a watchlist slot.
  - Demotes micro-cap junk (PURR $2×1M = $2M) while admitting mid-cap movers with genuine flow.
  - Movers sorted by dollar-volume descending for slot allocation.

- **RS soften for pinned liquid core** (v-tag `v-pinned-rs-soften-2026-09-17`)
  - Config: `ENABLE_PINNED_RS_SOFTEN` (default False — conservative).
  - Config: `PINNED_MIN_RS_VS_SPY` (default 0.25 vs standard 0.5).
  - When enabled, pinned symbols use softer RS threshold for day-trade momentum entry.
  - Non-pinned movers always use standard `MOMENTUM_MIN_RS_VS_SPY`.

- **Gap commentary time gate** (v-tag `v-gap-commentary-time-gate-2026-09-17`)
  - "Gap Detected" commentary only emitted during first 30 minutes (09:30–10:00 ET).
  - Gap data still tracked internally; just no commentary spam after 10:00 ET.

### Changed

- `core/loops/screener_loop.py::_build_pinned_watchlist()` — new method orchestrates pinned-first, dollar-vol-ranked slot allocation.
- `strategies/builtin.py::DayTradeMomentumStrategy` — RS threshold selection considers pinned status.
- `core/engine.py::_analyze_premarket_gaps()` — time-gated commentary.

### Config Flags

| Flag | Default | Description |
|------|---------|-------------|
| `ENABLE_PINNED_WATCHLIST` | True | Reserve slots for pinned symbols |
| `PINNED_WATCHLIST` | NVDA,TSLA,META,AMZN,MSFT,GOOGL,AVGO,AMD | Symbols to pin |
| `MIN_DOLLAR_VOLUME` | 10,000,000 | Dollar-vol floor for non-pinned movers |
| `ENABLE_PINNED_RS_SOFTEN` | False | Use softer RS for pinned symbols |
| `PINNED_MIN_RS_VS_SPY` | 0.25 | RS threshold when RS soften enabled |

### Tests

- 20 new tests in `tests/test_pinned_watchlist.py` covering config parsing, watchlist building, dollar-vol demote, and RS threshold selection.

### Operational notes

- All flags have safe off-path defaults. Set `ENABLE_PINNED_WATCHLIST=0` to revert to legacy mover-driven watchlist.
- `ENABLE_PINNED_RS_SOFTEN` starts False; enable after observing pinned symbols being rejected on `weak_relative_strength` with otherwise valid setups.
- DAY_TRADE/MEAN_REV/ORB live knobs unchanged. HANDS_OFF unchanged.

---

## 2026-06-09 — Live verification of close-path + circuit fixes

### Operational notes

- Bot restarted at 09:29 ET into fresh live session (PID 664706).
- First production observation of `Bot-only P&L Check` log line at 09:24:10: bot P&L $0.00, Schwab P&L −$161.86, circuit threshold $1,657.91. **#31 working as designed** — externals' overnight drift no longer consumes circuit headroom.
- Watchlist re-discovered 22 names; falling-knife filter immediately validated on PLTR (RSI 26.9 oversold but `close $135 < SMA50 $135.77` + bearish MACD → blocked).
- Bot remained in live mode across day, ready for first signal at the open.

---

## 2026-06-08 / 2026-06-09 — Close-path hardening + market-context awareness + SHORT verdict

A single trading session's work triggered by the **2026-06-08 11:12 ET manual close incident** (Schwab rejected 3 closes for `oversold position` because the bot's close path did not cancel the OCO bracket first; operator recovered $533 of profit via a 6-cancel Schwab dance).

### Added

- **MarketContext service** (`core/market_context.py`, commits `d2728fd` + `d4399fb`)
  - Pure-function `read_market_context(symbol)` returning regime, sector strength, VIX state, time-of-day, and 0.5..1.5 conviction multiplier.
  - 11 SPDR sector ETFs added to cache (XLK/XLF/XLE/XLV/XLY/XLP/XLI/XLB/XLU/XLRE/XLC).
  - Wired into mean-reversion, breakout, and news strategies as `allows_long`/`allows_short` gate.
  - Wired into `risk/manager.py` as conviction multiplier (clamped 0.4..1.6).
  - Gated by `ENABLE_MARKET_CONTEXT_GATE` + `ENABLE_MARKET_CONTEXT_SIZING` (both default True).
  - 37 unit tests.

- **Bot-only P&L circuit** (commit `df7ac9b`, v-tag `v-bot-only-pnl-circuit-2026-06-08`)
  - Daily-loss circuit uses bot-managed P&L (realized today + unrealized open) instead of account-wide Schwab P&L.
  - Stops external (HQGE/PINS/COIN) gap-downs from pausing the bot.
  - Error message shows both numbers: `"P&L: $-X, schwab_pnl: $-Y"`.

- **Rising-peak filter for mean-rev SHORT** (commit `df7ac9b`, v-tag `v-rising-peak-filter-2026-06-08`)
  - Symmetric to LONG-side falling-knife filter.
  - Originally `close<SMA50 AND MACD<MACD_signal`; loosened to `close<SMA50` only after backtest evidence (commit `7112d8e`).
  - Does NOT enable SHORT — filter is precondition only.

- **Deploy runbook** (`docs/DEPLOY_2026-06-08_EVENING.md`) — line-by-line checklist.
- **SHORT backtest script** (`research/backtest_short_filter_2026-06-08.py`) — 3-run comparison with automated decision gate.

### Changed

- **Close path cancels OCO before sell** (v-tag `v-cancel-bracket-before-close-2026-06-08`)
  - `_close_real_position` now calls `_cancel_existing_orders(position.symbol)` before placing the close.
  - Fixes: manual close API works first-try instead of being rejected for `oversold position`.

- **Close path verifies fill before state cleanup** (v-tag `v-verify-close-fill-2026-06-08`)
  - `_close_real_position` awaits `_verify_order_fill` before returning True.
  - Schwab's 201 ("accepted") no longer treated as filled.
  - Failures → ZOMBIE state instead of silent local-state pop.

### Verified

- **mean-rev SHORT not currently viable** (commit `7112d8e`)
  - 60d: 1,176 candidates, 0 pass filter. Unfiltered: PF 1.04, max DD −52%, max 20 simultaneous shorts.
  - 360d: 4,965 candidates, 0 pass filter. Unfiltered: PF 1.05, max DD −167%.
  - Geometric diagnosis: strategy entry (`close > SMA20+2σ`) and filter (`close < SMA50`) target near-disjoint setups.
  - **Verdict**: ENABLE_MEAN_REV_SHORT stays False. Filter retained for any future SHORT variant.

### Tests

- 22 new in `tests/test_recent_fixes.py`, 37 in `tests/core/test_market_context.py`.

### Operational notes

- All four fixes deployed atomically at 16:03 ET 2026-06-08 (post-close).
- Day P&L: **+$666.21 realized** across 5 winning bot trades (NOK/GLW/ARM/ONDS/OWL).
- Account P&L +$1,007 (includes externals).
- MarketContext conviction multipliers live-observed: NOK 0.95×, GLW 1.25×, ARM 1.25×, OWL 1.10×, ONDS 1.25×.

---

## 2026-06-01 to 2026-06-03 — Smart target + monitor + sizing adjustments

### Added

- **Destination-aware take_profit selection** (commit `6884155`)
  - `core/smart_target.py` picks chart-based destinations (resistance, prior swing high) instead of fixed % targets.
  - Validated 2026-06-08: ARM hit ABOVE its smart_target of $361.61 (filled at $362.79).

- **Local bot monitor agent** (commit `2169619`, v-tag `v-bot-monitor-2026-06-01`)
  - `tools/bot_monitor.py` — pure-observer agent, writes to `docs/monitor/latest.md` every ~60s.
  - Tracks P&L window, positions, errors, market regime, token TTL.

- **R:R audit at signal router** (commit `0e483b7`)
  - Logs realized risk/reward ratio for every signal; surfaces low-R:R trades for review.

### Changed

- **Trade sizing graduated** (commit `0e483b7`)
  - Progressive sizing scale based on consecutive wins.

- **max_daily_loss bumped 0.01 → 0.03** (commit `71ed6ec`)
  - Original 1% (≈$290) was too tight; mid-day intraday drift would trip it.

- **atr_reward_risk_ratio 2.5 → 2.0** (commit `e3de906`)
  - Mid-day adjustment after observing target-too-far rejections.

### Fixed

- **rr_target fallback capped at half-20-bar-range** (commit `aebaf24`)
  - Prevents aspirational fantasy targets when smart_target has no chart anchor.

- **tail_log monitor bug** (commit `37912c1`)
  - Was returning 0 entries on busy logs.

---

## 2026-05 — Launch infrastructure + parallel loops + side classifier + dashboard hardening

### Added

- **Quarter-size live launch** (commit `1dafd3e`, 2026-05-26)
  - `live_size_multiplier = 0.25` for safety.
  - Strategy-specific multipliers via `strategy_size_multipliers.mean_reversion = 0.5`.
  - Effective sizes: news 25% of normal, mean-rev 12.5% of normal.

- **Rollback script** (commit `c4ab0b2`)
  - `rollback_launch.sh` — one-command revert of the 2026-05-26 launch config.

- **Parallel-loops phase 1 + stream-watchdog rewrite + day-pnl fix** (commit `11911b2`)
  - Multiple analysis loops can run in parallel without stomping shared state.
  - Stream watchdog detects silent stream death (heartbeat threshold-based).
  - day_pnl computed per-position from netChange × qty (vs unreliable `currentDayProfitLoss`).

- **Side classifier** (commit `11911b2`, design phase)
  - Long/short scoring service (designed; live-integration deferred).

- **Per-strategy P&L attribution backtest** (commit `8b26300`)
  - `backtest/engine.py` + CLI + WebSocket progress.
  - Replaced inline backtest with proper package structure.

- **Direction reader algorithm** (commits `81e2fa2` + `76dc980`)
  - `core/direction_reader.py` — 5-component score for price direction.
  - Wired into 3 strategy gates.

- **News strategy technical confirmation gate** (commit `f1ae93b`)
  - News becomes consulting input, not entry driver.

- **Dashboard market regime strip** (commit `f2ccc17`)
  - Strip above ticker tape showing SPX / Nasdaq / VIX / breadth.

- **Mean-rev uptrend-pullback entry** (commit `393b980`)
  - New entry path: RSI 30–55 + close ≥ SMA50 + close near BB lower.

- **Periodic _save_state loop** (commit `bb437bf`)
  - `trading_state.json` now tracks live state on a timer, not just after closes.

### Changed

- **Token TTL warning at startup** — `deploy.sh` aborts if Schwab refresh-token has <8h life.
- **max_daily_loss bumped 0.01 → 0.03** (2026-05-27 calibration, commit `5aaa5e0`).
- **Mean-rev take_profit no longer capped at bb_middle** (commit `a99dd45`) — targets were too tight.
- **Mean-rev uptrend_pullback bar gate relaxed** (commit `fc48a00`).
- **Screener Gate 3 rel-strength filter loosened** (commit `486b161`) — was rejecting mega-cap rotations.

### Fixed

- **Silent auto-cancel logging** (commit `8cd7a14`, v-tag `v-autocancel-logging-2026-05-26`)
  - Prior version silently cancelled unfilled limit orders after 20s with no log line.
  - QBTS 2026-05-26 09:37 was the trigger incident.

- **Conda env pinned for deploy** (commit `ea45ef5`)
  - `deploy.sh` refuses silent Python fallback.

### Operational notes

- **2026-05-11 incident**: Re-enabling `ENABLE_MEAN_REV_SHORT` produced 5 simultaneous shorts at −$665 unrealized. Reverted same day. Triggered design of rising-peak filter (shipped 2026-06-08 as #35).
- **2026-05-26 launch**: First quarter-size live session. `max_daily_loss=0.01` ($290) tripped on a transient externals drawdown; bumped to 0.03 next day.

---

## 2026-04 — The "make it actually work" month (largest single-month effort: 65 commits)

This is the month where the bot transitioned from "prototype with features" to "operational system with discipline." Multiple major themes shipped in parallel.

### Added — risk + sizing

- **Kelly bet sizing + sector correlation guard + weekly retrain** (commit `44b2fa8`)
- **ATR-scaled stops + position sizing across all strategies** (commit `b1e37ca`) — replaces fixed 3%/6% stops.
- **Pre-trade shortable check via Alpaca assets API** (commit `a8f7b20`)
- **Reject penny stocks below $5** (commit `9fa4508`)
- **Momentum strategy: ATR-scaled stops/targets** (commit `0ea0d9a`)

### Added — exit discipline

- **Breakeven-stop ratchet** (commit `8d36529`) — *winners can no longer become losers*. At +0.5R, stop moves to entry.
- **1R partial exit + ATR trailing stop** (commit `66dbc07`)
- **Thesis re-validation on stale trades** (commit `a426b89`) — re-check news + indicators if held >N minutes.
- **Proactive exit when raw_data builds correct indicator view** (commit `04afc4d`)

### Added — market hours + audit

- **Pre-open warmup window** (commit `1844f8a`) — bot is ready at 09:30 instead of cold-starting.
- **Full pause loop when markets closed** (commit `1547462`) — no wasted Schwab API calls overnight.
- **Block new entries outside RTH in sim mode too** (commit `f1dd215`).
- **Postgres logging for all decisions + trades** (commit `1a5e26a`) — every gate decision audit-logged.
- **Per-position `managed_by_bot` flag with dashboard checkbox** (commit `a1943f1`) — distinguishes bot trades from operator's external holdings.
- **`bot_positions` table + sync open positions to DB** (commit `c903bb3`)
- **Structured per-tick decision logs** (commit `3c5e79c`)
- **Structured audit log for trade-decision gates** (commit `e27cb39`)
- **Persist simulated positions across restarts** (commit `1417754`)

### Added — ML pipeline

- **López de Prado meta-labeling pipeline + shadow mode** (commit `2df4148`) — proper out-of-sample backtest discipline.
- **Vectorized feature extraction, ETF filter, expanded training universe** (commit `c3cfa26`)
- **Walk-forward CV + per-class precision/recall report** (commit `084613a`)
- **Train IntegratedMLModel from rudra_dev minute_bars** (commit `dc98ed1`)
- **ML signal as actual veto, not advisory** (commit `aed933d`)

### Added — news + sentiment

- **Fresh-news verification gate via Alpaca + Yahoo** (commit `c789d5c`)
- **Shadow-track every news veto + post-hoc evaluation** (commit `95f5fe8`)
- **Block news-BUY in confirmed downtrends** (commit `902b2ea`)
- **Mean-rev: block long entries in confirmed downtrends** (commit `9a6589e`) — falling-knife filter (the LONG side mirror of yesterday's #35).
- **News sentiment widget rebuilt** (commit `0f22839`)

### Added — data pipeline

- **Alpaca 1-min bar backfill into Postgres `minute_bars`** (commit `9993eb0`)
- **Per-strategy P&L attribution backtest harness** (commit `add5e4e`)

### Added — UI

- **Full dashboard overhaul** (5-commit series `cbb5c30` → `48906be`)
  - Design tokens, glass utilities, fonts, ambient blobs
  - Unified top bar (logo, nav, toggles, status, uptime)
  - KPI cards + market status strip + watchlist heatmap
  - Modernized positions tables, commentary feed, news, trades
  - Chart.js + Portfolio Donut + Performance Metrics
- **Per-symbol logo display in positions tables** (commit `bea734d`)
- **WebSocket keepalive + auto-reconnect** (commit `28cdae8`)

### Changed

- **Default port 8000 → 9000** (commit `5804c0e`)
- **Watchlist cap raised 5-8 → configurable 20** (commit `f884736`)
- **Screener candidate pool 20 → 50, pre-market filters relaxed** (commit `1914e33`)
- **News freshness window 30 min → 4 hours** (commit `21365c1`)

### Fixed

- **Sim parity** (commit `156cf0e`) — dropped Simulation Override on ML disagreement; sim and live now use identical decision logic.
- **Manual_close_only + confirmations respected regardless of mode** (commit `5ea64e7`)
- **Ticker uses Schwab `netPercentChange`** (commit `0f3dd11`)
- **ML model validated as fitted before trusting loaded artifact** (commit `e16ddc4`)
- **Breakout volume gate scoped to actual breakouts** (commit `5663d16`)
- **News verifier toggle-able** (commit `81b6c67`) — default OFF.
- **News calls wrapped in `run_in_executor`** (commit `8425bf0`) — was blocking the event loop.

---

## 2026-03 — Security + structural refactor sprint (P1–P8)

A single-week disciplined refactor sprint, each commit numbered P1–P8.

### Changed

- **P1: Break up 14K-line core file into modular packages** (commit `efa279a`) — `core/`, `risk/`, `strategies/`, `analysis/`, `api/` directories created.
- **P3: Eliminate code duplication in config and circuit breakers** (commit `c7e24d1`)
- **P5: Harden error handling + graceful shutdown** (commit `97e8fdd`)
- **P6: Structured JSON logging + trade event logging** (commit `548f9a9`)
- **P7: Consolidate configuration + extract magic numbers** (commit `6565440`)
- **P8: Basic test coverage + fix pre-existing bugs** (commit `ef0ddf7`)

### Security

- **P2: Restrict CORS, add `.env.example`, harden `.gitignore`** (commit `a565144`)
- **API authentication + remove command injection vectors** (commit `07cf267`)
- **JSON.dumps for API key injection** (commit `53692f5`) — prevent script breakage from special chars.
- **Code review findings on security implementation** (commit `cbd2d5e`)

### Fixed

- **P4: Placeholder implementations + undefined references** (commit `b995bf0`)

---

## 2026-01 — Resume from hiatus, professional modules, Schwab API switch

After a 4-month pause (Sept–Dec 2025), the project resumes with a 20-commit January push.

### Added

- **Professional trading modules** (commit `ecb26d5`, integrated `64700bf`) — strategy_system.py, ml_model_manager_safe.py, advanced_orders.py, backtesting_engine.py, risk_management.py.
- **Real-time news sentiment widget** (commit `789cf5f`)
- **Tabbed interface with Sentiment + Backtesting views** (commit `7148c34`)
- **Settings management UI + API endpoints** (commit `db28d99`) — manage Schwab credentials, risk limits, ML toggles via web UI.
- **Real-time account stats + live trading toggle** (commit `5f96645`)
- **Real-time market data for simulation mode via yfinance** (commit `a47d83c`)
- **Comprehensive analytics logging system** (commit `563b87f`)
- **`get_win_rate` on PerformanceAnalyzer** (commit `4036c38`)

### Changed

- **Backtesting switched from Yahoo to Schwab API** (commit `7fc2e30`) — more authoritative data source.
- **Settings UI ports configurable via CLI** (commit `24fdcb2`)
- **Settings UI connects to main bot on port 9000** (commit `27c515e`)
- **External positions protected from bot management when manual mode ON** (commit `fb221ce`)

### Fixed

- **Schwab API 400 in backtesting** (commit `e85adb3`) — removed conflicting parameters.
- **Sentiment widget field names + score scaling** (commit `16e6c65`)
- **Sentiment widget integrated into main dashboard** (commit `e4fed`)
- **schwab-py compatibility** (commits `d18c38a`, `e4fcab1`, `76ccfe9`) — token format, interactive prompt skip, parameter compat.
- **Noisy commentary queue error logging** (commit `813168e`)
- **Improved error messages when main bot not running** (commit `fda3ac5`)
- **RobustScaler "not fitted" error** (commit `2603e54`) — set `is_trained=False` initially instead of forcing True.
- **Forced `is_trained=True` removed** (commit `00baf03`) — was bypassing model training.
- **NoneType comparison errors for config values** (commit `e3a39df`)
- **NoneType format error for MAX_RISK_PER_TRADE** (commit `ed23240`)

### Operational notes

- January 2026 represents the project's "second start" — operator returned with substantially clearer requirements and a working baseline from July 2025.

---

## 2025-08 — Backtesting dashboard

### Added

- **Backtesting-performance dashboard** (commit `91e27e4`) — initially shipped, then reverted (`e916611`) due to instability.
- **Major trading bot enhancements and fixes** (commit `bb9f6b2`) — catch-all consolidation commit.

---

## 2025-07 — Project genesis

The original bot, built over 4 days in mid-July 2025.

### Added

- **First commit** (`761b781`, 2025-07-08) — initial bot scaffold.
- **Trading bot with backtesting, UI, and ML** (commit `883e785`) — first feature-complete prototype.
- **Real-time ticker tape via Schwab API** (commits `5727031`, `35b78be`, plus 4 follow-up fixes).
- **ML prediction enable/disable toggle** (commits `80d29dc`, `9fedb0c`)
- **VotingClassifier with multi-timeframe features + news sentiment** (commit `4020cac`)
- **OCO order placement after trade execution** (commits `3bb449b`, `5b16a57`)

### Fixed

- **Critical buying power issue** (commit `f64b121`)
- **Buying power updated periodically** (commit `42be237`)
- **Config.yaml filename / format updates** (commits `055e82c`, `b0c8c11`)

### Operational notes

- Initial development sprint completed in 4 days (Jul 8 → Jul 12).
- Project then paused for 4 months (no commits Sept–Dec 2025) before resuming Jan 2026.

---

## How to read this file

Each release entry follows the same structure:

- **Added** — new features, files, configs
- **Changed** — modifications to existing behavior
- **Fixed** — bug fixes (called out separately when distinct from changed behavior)
- **Removed** — deleted features
- **Security** — security-relevant changes
- **Verified** — empirical outcomes from backtests / live observation (not code changes, but evidence that informs strategy decisions)
- **Tests** — regression coverage added
- **Operational notes** — deploy timing, live evidence, incidents, gotchas

v-tags (e.g., `v-cancel-bracket-before-close-2026-06-08`) are greppable anchors in the source. They live inline as `# v-…` comments at the patch site; future regressions can be diagnosed by grepping the codebase for the tag.

Commit SHAs (e.g., `df7ac9b`) reference the git history. `git show <sha>` reconstructs any single change in full.
