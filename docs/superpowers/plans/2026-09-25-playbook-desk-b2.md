# Playbook Desk B2 — The Desk (shadow) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the bot a human trader's daily routine — a pre-market plan, an in-play stock ranking at 09:35, an adaptive playbook per (family, side, regime), and a post-close journal — all running in shadow (audited, never blocking) until the research gate is met.

**Architecture:** Pure, tested decision modules under `core/desk/` (plan rules, in-play scoring, playbook transitions, journal rendering) with thin I/O adapters; one supervised loop (`core/loops/desk_loop.py`) that drives them on ET wall-clock crossings; two router hooks (desk limits before the fit gate, playbook after it) that audit in shadow; a nightly playbook updater in `research/strategy_hunt/playbook_update.py` driven by resolved shadow ledgers and `bot_trades`; three read-only API endpoints and a monitor section. Every desk decision emits `engine_decision component=desk_* …`.

**Tech Stack:** existing engine (`core/engine.py` router and `_build_supervisor`), `core/market_context.py` + `MarketIndicesCache`, `core/strategy_fit_bars.py` (bars history merge), `data_providers/db_logger.py` (`bot_trades`, `bot_decisions`), `risk/manager.py` sizing chain, `strategies/fit_lane.py` LANE_MAP, FastAPI routes in `api/routes.py`, `tools/bot_monitor.py`.

**Spec:** `docs/superpowers/specs/2026-09-25-playbook-desk-design.md` §7 (desk), §8 (flags), §9 (B2 is shadow-only until a PASS exists; B1.5 result: NO-GO beyond shadow).

## Global Constraints

- Same as B1/B1.5: explicit-path commits with the trailer, never `git add -A`, never touch runtime files, `data/`, or `models/`; tests offline; `core/api/strategies` never import `research`; files under 400 lines; immutable style (rules return new objects); explicit error handling; the engine is not constructible in tests — wiring is verified with source-level assertions (`tests/core/test_strategy_fit_wiring.py` pattern) and logic with pure-function tests.
- Live safety: all new flags default off (`ENABLE_DESK`, `ENABLE_DESK_WATCHLIST`, `ENABLE_DESK_LIMITS`, `ENABLE_PLAYBOOK_GATE`); with `ENABLE_DESK` off nothing new runs; with it on, limits and playbook only audit unless their own flags are on. No lane gets live entries. The live bot is restarted only after the close.
- Regime labels `risk_on`, `risk_off`, `mixed`, `unknown`. Times are ET (`America/New_York`). Persisted desk files live under `Config().DESK_DATA_DIR` (default `data/desk`, git-ignored). Journal files are runtime artifacts: `data/desk/journal/<date>.md` (ruling: not `docs/journal`, to keep runtime output out of the repo).
- Family naming: `strategies/fit_lane.py::LANE_MAP` maps live strategy names to family names; the playbook is keyed by family.
- Commit trailer:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BT1pvoczNh7y1pBxVN53R9
```

---

### Task 1: Desk models, config flags, DailyPlan rules and persistence

**Files:** create `core/desk/__init__.py`, `core/desk/models.py`, `core/desk/daily_plan.py`, `core/desk/store.py`; modify `core/config.py` (new section `# v-playbook-desk-b2-2026-09-25` right after the sentiment-backend block); tests `tests/core/desk/test_daily_plan.py`, `tests/core/desk/test_store.py`, `tests/core/test_desk_config.py`.

**Config (env + yaml fallback, same pattern as `ENABLE_STRATEGY_FIT_GATE`):** `ENABLE_DESK` (False), `ENABLE_DESK_WATCHLIST` (False), `ENABLE_DESK_LIMITS` (False), `ENABLE_PLAYBOOK_GATE` (False), `DESK_IN_PLAY_N` (20), `DESK_SCAN_MAX_SYMBOLS` (60), `DESK_DATA_DIR` ("data/desk"), `DESK_PLAN_TIME_ET` ("09:00"), `DESK_REFRESH_TIME_ET` ("09:35"), `DESK_JOURNAL_TIME_ET` ("16:05"), `DESK_CALENDAR_PATH` ("data/desk/calendar.json").

**Models (`core/desk/models.py`, frozen dataclasses with `to_dict`/`from_dict`, unknown keys ignored, non-finite floats → None):**
- `InPlayRow(symbol, score, rvol_open15, gap_pct, catalyst, rs_20, reasons: tuple[str, ...])`
- `DailyPlan(date, regime, spy_change_pct, vix, vix_change_pct, breadth_score, calendar_flags: tuple[str, ...], long_ok, short_ok, risk_budget_pct, max_trades, sit_out, sit_out_reason, in_play: tuple[InPlayRow, ...], notes: tuple[str, ...], built_at, refreshed_at)`
- `PlanInputs(date, regime, spy_change_pct, vix, vix_change_pct, breadth_score, calendar_flags, losing_day_streak: int, prev_day_pnl_r: float, fit_stale: bool, gate_enforcing: bool, max_daily_loss_pct: float, allows_long: bool, allows_short: bool)`

**Rules (`core/desk/daily_plan.py`, pure):** `build_daily_plan(inputs, now_et) -> DailyPlan` and `refresh_plan(plan, in_play, now_et) -> DailyPlan`.
- `sit_out` (with reason, first match wins): `"calendar_high_impact_before_1030"` when any calendar flag starts with `high:` and its time is < 10:30; `"vix_spike"` when `vix_change_pct > 15`; `"two_losing_days"` when `losing_day_streak >= 2`; `"fit_stale_gate_enforcing"` when `fit_stale and gate_enforcing`.
- `risk_budget_pct = max_daily_loss_pct × {risk_on 1.0, mixed 0.6, risk_off 0.4, unknown 0.4}`, halved when `prev_day_pnl_r < 0`.
- `max_trades = {risk_on 6, mixed 4, risk_off 2, unknown 2}`.
- `long_ok = allows_long and breadth_score >= 0.4` (breadth None → uses `allows_long` alone); `short_ok = allows_short and breadth_score <= 0.6` (None → `allows_short`).
- `notes` list the fired rules in words.
- `refresh_plan` replaces `in_play` and sets `refreshed_at`; everything else immutable.

**Calendar:** `load_calendar_flags(path, date) -> tuple[str, ...]` reads a JSON list of `{date, time_et, event, impact}` and returns flags formatted `"<impact>:<HH:MM>:<event>"`; missing file → empty tuple; malformed rows skipped with a warning.

**Store (`core/desk/store.py`):** `plan_path(dir, date)`, `save_plan(plan, dir)` atomic (tmp + replace), `load_plan(dir, date) -> DailyPlan | None` never raises (logs and returns None on malformed JSON).

**Tests (exact):** each sit-out rule fires alone and by precedence; budget table incl. halving; max_trades table; long/short with breadth None; refresh keeps other fields; calendar parsing incl. missing/malformed; store round-trip, atomicity (no tmp left), malformed → None; config defaults and env overrides for every flag.

- [ ] TDD; commit `feat(desk): models, flags, DailyPlan rules, plan store`.

---

### Task 2: InPlayRanker — pure scoring and opening-feature computation

**Files:** create `core/desk/inplay_ranker.py`, `core/desk/inplay_features.py`; tests `tests/core/desk/test_inplay_ranker.py`, `tests/core/desk/test_inplay_features.py`.

**Scoring (`rank_in_play(candidates: list[InPlayRow], pinned: list[str], top_n) -> tuple[InPlayRow, ...]`):** score = 0.4×min(rvol_open15/3, 1) + 0.2×min(|gap_pct|/5, 1) + 0.2×catalyst + 0.2×rs_component, where catalyst ∈ {0, 1} and rs_component = min(|rs_20|/10, 1) (rs_20 in percent). NaN inputs count as 0 with a reason `"missing:<field>"`. Pinned symbols are always included (appended after the ranked top_n if not already there, flagged `"pinned"` in reasons). Deterministic order: score desc, then symbol.

**Features (`opening_features(symbol, today_bars5, history_bars5, prior_ctx: dict | None, now_et) -> InPlayRow`):** `rvol_open15` = today's first three 5-minute bars' volume ÷ median of the same window over the prior 14 sessions in `history_bars5` (NaN if fewer than 5 prior sessions or fewer than 3 bars today); `gap_pct` = today's open ÷ prior session close − 1 (×100); `catalyst` = 1 if `prior_ctx.get("earnings_today")` or `prior_ctx.get("earnings_prev")` or `prior_ctx.get("news_count_24h", 0) >= 3`; `rs_20` from `prior_ctx.get("rs_20")`. Uses the same session-bar conventions as `strategies/families/indicators.py` (bar_of_day from 09:30, 5-minute). `prior_ctx` comes from the last row of `data/strategy_hunt/context/universe_daily.parquet` for that symbol: add `load_prior_context(root, symbol) -> dict | None` in `core/desk/inplay_features.py` reading with pyarrow filters (never raises; None on any failure) — note `core` may read the research *data* tree but must not import `research`.

**Tests:** score formula by hand on three rows; caps; pinned appended; NaN handling; opening features on a constructed two-week frame (rvol exactly 2.0, gap exactly +3%); insufficient history → NaN; catalyst rules; `load_prior_context` on a tmp parquet and on a missing file.

- [ ] TDD; commit `feat(desk): in-play ranker and opening features`.

---

### Task 3: AdaptivePlaybook — rules, store, nightly updater, gate function

**Files:** create `core/desk/playbook.py` (rules + reader), `research/strategy_hunt/playbook_update.py` (nightly updater), CLI subcommand `playbook` in `research/strategy_hunt/cli.py`; tests `tests/core/desk/test_playbook.py`, `tests/research/strategy_hunt/test_playbook_update.py`, `test_cli.py` (extend).

**Model:** `PlaybookEntry(family, side, regime, status: "enabled"|"suspended"|"probation", size_mult: float, exp_r_recent, t_stat_recent, n_recent, updated_at, reason)`; `Playbook(entries: dict[(family, side, regime) → PlaybookEntry], updated_at)` with `to_dict`/`from_dict` (keys serialized as `"family|side|regime"`), `load_playbook(path) -> Playbook` never raises (empty playbook on missing/malformed), `save_playbook` atomic.

**Rules (`update_entry(prev: PlaybookEntry | None, recent: list[ResolvedTrade], fit_verdict: str | None, now) -> PlaybookEntry`, pure):** `recent` = last 30 resolved trades for the key, newest last; `n_recent = len(recent)`; `exp_r_recent = mean(r)`; `t_stat_recent` = trade-level t (0 if n < 2). `suspended` (size 0.0) when `n_recent >= 10 and exp_r_recent < 0 and t_stat_recent < -1`; `probation` (size 0.5) when `exp_r_recent < 0` and not suspended, or when `n_recent < 10`; `enabled` with `size_mult 1.25` when `exp_r_recent >= 0.15 and n_recent >= 40 and fit_verdict in ("PASS", "WATCH")`; otherwise `enabled` with `size_mult 1.0`. `size_mult` clamped to [0.0, 1.5]. `reason` names the rule.

**Updater (`research/strategy_hunt/playbook_update.py`):** `resolved_trades_by_key(root, families, engine=None) -> dict[key, list[ResolvedTrade]]` merges (a) shadow ledgers resolved by `shadow_resolver.resolve` (regime from the resolved `SimTrade.regime`) and (b) live `bot_trades` rows with `pnl_r` not null and a strategy mapped through a research-side copy of the family map (`FAMILY_BY_STRATEGY` in `research/strategy_hunt/playbook_update.py`; test asserts it equals `strategies.fit_lane.LANE_MAP` — research may import strategies, not the reverse), regime from `bot_decisions` audit fields when present else `"unknown"`; `update_playbook(root, snapshot, now) -> Playbook` applies `update_entry` per key using the snapshot's pooled/symbol verdicts (`fit_verdict` = best verdict for that family/side/regime, PASS > WATCH > FAIL, None if no row) and writes `<DESK_DATA_DIR>/playbook.json`. CLI `playbook --root … [--desk-dir …] [--postgres]`; `cmd_all` appends it after `score`.

**Gate (`core/desk/playbook.py::evaluate_playbook_gate(playbook, family, side, regime, enabled) -> GateDecision`):** reuse `core/strategy_fit_gate.GateDecision`; no entry → `action="allow", reason="no_playbook_entry"`, governed False; `suspended` → `skip` when enabled else `shadow`, reason `playbook_suspended`; `probation`/`enabled` → `allow` with `size_mult` in `audit_fields()`; regime `unknown` falls back to the entry for `mixed` if present.

**Tests:** each transition; clamp; serialization round-trip; updater on a tmp root with a fake ledger + a fake `bot_trades` frame (inject the SQL reader as a callable); CLI recorder for `cmd_all` order (…score → playbook); gate decisions incl. unknown-regime fallback.

- [ ] TDD; commit `feat(desk): adaptive playbook rules, nightly updater, playbook gate`.

---

### Task 4: Journal — rendering and data adapters

**Files:** create `core/desk/journal.py`, `core/desk/journal_data.py`; tests `tests/core/desk/test_journal.py`.

**Rendering (`render_journal(plan, decisions, trades, playbook_changes, trailing) -> str`, pure):** sections in order: Plan (regime, bias, budget, max trades, sit-out, in-play top 10), Decisions (count by component/action; every `skip` with reason; every desk shadow audit that would have blocked), Trades (symbol, side, family, R, exit reason, hold), Expectancy by setup×regime (today and trailing 20 sessions: n, exp_R, PF), Playbook changes, What to change (derived: families with trailing exp_R < 0 and n ≥ 20 → "suspend candidate"; sit-out days with positive would-have-been trades → "review sit-out rule"; max_trades hits → "raise/lower max_trades").

**Adapters (`core/desk/journal_data.py`):** `decisions_for(date, db_logger) -> list[dict]` and `trades_for(date_range, db_logger) -> list[dict]` using `DbLogger.get_todays_bot_decision_entries` / `get_todays_bot_entries` and a new `DbLogger.get_bot_trades_between(start, end)` (add it in `data_providers/db_logger.py` with the same parameterized-query style; never raise, return []). `journal_path(dir, date)`, `write_journal(text, dir, date)` atomic.

**Tests:** render on fixtures covers every section and the three "what to change" rules; adapters with a fake db_logger; missing data renders "none".

- [ ] TDD; commit `feat(desk): journal rendering and data adapters`.

---

### Task 5: Desk loop and engine wiring (shadow)

**Files:** create `core/loops/desk_loop.py`; modify `core/engine.py` (`_build_supervisor` registration under `ENABLE_DESK`; router hooks), `core/loops/screener_loop.py` (skip overwriting `dynamic_watchlist` when `ENABLE_DESK_WATCHLIST` and a plan with in-play rows exists today), `risk/manager.py` (size step reading `signal.reasoning["strategy_size_multiplier"]` clamped 0.0–1.5, applied only when `ENABLE_PLAYBOOK_GATE`); tests `tests/core/desk/test_desk_loop.py`, `tests/core/test_desk_wiring.py` (source-level assertions), `tests/risk/test_desk_size_multiplier.py` (or the existing risk test module).

**DeskLoop (`DeskLoop(engine).run()`):** every 30 s compute `now_et`; on first crossing of `DESK_PLAN_TIME_ET` (per date) build the plan from `read_market_context("SPY")` (regime, spy change, vix, vix change, `allows_long/short`), `engine.market_state.get("breadth")`, calendar flags, losing-day streak and prev-day R from `bot_trades` (via `journal_data`), `fit_stale` from `engine._strategy_fit.is_stale()` (or the reader's equivalent), `gate_enforcing = ENABLE_STRATEGY_FIT_GATE and not STRATEGY_FIT_SHADOW`; save; audit `desk_plan action=build`. On `DESK_REFRESH_TIME_ET`: candidates = pinned ∪ `engine.dynamic_watchlist` ∪ top `DESK_SCAN_MAX_SYMBOLS` symbols by yesterday's `rvol_rank5`/ADV from the context parquet; for each, today's 5-minute bars via the engine's provider (`asyncio.to_thread`, bounded concurrency 4, per-symbol try/except) and history via `core/strategy_fit_bars.load_fit_history`; rank; `refresh_plan`; save; audit `desk_inplay action=rank n=…`; when `ENABLE_DESK_WATCHLIST`, set `engine.dynamic_watchlist = [r.symbol for r in plan.in_play]` (pinned preserved). On `DESK_JOURNAL_TIME_ET`: render and write the journal; audit `desk_journal action=write`. All steps wrapped: an exception logs `desk_loop step failed` and never kills the loop. Weekend/holiday: skip when the market calendar (existing helper if any, else weekday check) says closed.

**Router hooks (in the signal router, immediately before the strategy-fit gate block):** `evaluate_desk_gate(plan, trades_today: int, side, enabled) -> GateDecision` (pure, in `core/desk/daily_plan.py`): `sit_out` → skip/shadow `desk_sit_out:<reason>`; `trades_today >= plan.max_trades` → `desk_max_trades`; `side == long and not plan.long_ok` → `desk_direction_long_off` (short likewise). Audited as `component=desk_limit`. Immediately after the fit gate: `evaluate_playbook_gate(...)` with family from `LANE_MAP.get(strategy_name)` and regime from `signal.reasoning`; audited `component=playbook`; `size_mult` written to `signal.reasoning["strategy_size_multiplier"]` only when `ENABLE_PLAYBOOK_GATE`. `trades_today` from the engine's own count of today's entries (existing counter if present, else `len(get_todays_bot_entries())` cached per minute).

**Tests:** wiring assertions (desk gate text precedes the fit-gate audit string; playbook audit follows it; supervisor registers `desk_loop` under the flag; screener guard present); loop step functions unit-tested with fakes (a fake engine with `market_state`, `dynamic_watchlist`, provider stub); risk multiplier step clamps and is inert when the flag is off.

- [ ] TDD; commit `feat(desk): desk loop, router hooks (shadow), playbook size step`.

---

### Task 6: API endpoints and monitor section

**Files:** modify `api/routes.py` (`GET /api/desk/plan?date=`, `GET /api/desk/playbook`, `GET /api/desk/journal?date=` — all read-only, never 500: return `{ok: false, error}` with 200 on failure like `/api/strategy-fit`), `tools/bot_monitor.py` ("## Desk" section before "## Strategy fit": plan summary, sit-out, top 5 in-play, playbook suspended/probation counts); tests `tests/api/test_desk_routes.py` (TestClient with the engine attribute stubbed as the existing strategy-fit route tests do), `tests/test_bot_monitor.py` (order assertion `## Desk` < `## Strategy fit`).

- [ ] TDD; commit `feat(desk): read-only desk endpoints and monitor section`.

---

### Task 7: Docs, nightly cron, changelog

**Files:** `docs/research/strategy_hunt.md` (playbook step in `all`, `playbook` subcommand), new `docs/desk.md` (what the desk does, flags, files, endpoints, how to read the journal, how to promote from shadow), `docs/BOT_OPERATIONS_MANUAL.md` (desk section + flags table), `CHANGELOG.md`, the crontab line documentation (nightly `all` now ends with `playbook`).

- [ ] Commit `docs(desk): desk layer documentation and nightly playbook step`.

---

### Task 8: Enable the desk in shadow after the close (execution)

- [ ] Run `python -m research.strategy_hunt playbook --postgres` once (produces `data/desk/playbook.json`; expect every entry `probation` or `enabled 1.0` given the shadow ledgers so far).
- [ ] After 16:00 ET only: back up `.env`, append `ENABLE_DESK=1` (all other desk flags absent = off), restart the bot the way it is normally launched, verify `GET /api/status`, `GET /api/desk/plan` (404-style `ok:false` until 09:00 next session is acceptable), `GET /api/desk/playbook`, and that the supervisor log shows `desk_loop` registered. Record in the ledger and memory. Never enable `ENABLE_DESK_LIMITS`, `ENABLE_PLAYBOOK_GATE`, `ENABLE_DESK_WATCHLIST`, or any live entries.

---

## Parked
- B3: `DESK_LLM_BRIEFING` (local 7–8B instruct model on the RTX 3090) for the morning briefing and post-close critique.
- `bot_trades` has no `fit_family`/`regime` columns; the playbook updater derives them (LANE_MAP, bot_decisions). Adding the columns is a later schema task.
- Lanes that do not publish `market_context_regime` into `reasoning` (day_trade_momentum, shorts, ORB) get `unknown` regime in the playbook and fall back to `mixed`.
