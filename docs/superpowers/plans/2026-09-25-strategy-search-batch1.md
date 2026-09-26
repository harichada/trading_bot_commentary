# Strategy Search Batch 1 — Slower, Fewer, Cheaper Trades

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Test whether slower strategies (overnight holds, 2–20 day mean reversion, wide-universe PEAD, daily volatility-contraction breakouts, weekly relative-strength rotation) survive costs where the intraday families did not, with a held-out final period and honest multiple-testing accounting.

**Architecture:** Five new pure families in `strategies/families/swing2.py` (same `Family` contract), one simulator addition (`hold_kind="overnight"` exits at the next session's open), a `holdout_months` option in the walk-forward windows and CLI (default 0; the batch runs with 6 and unlocks the holdout only for the final scoring), and a "search accounting" section in the report (families × grid points × sides tested vs PASS/WATCH). Everything else (costs, floors, shuffle baseline, regime labels, reports) is reused unchanged.

**Spec:** `docs/superpowers/specs/2026-09-24-strategy-hunt-design.md` (families contract, floors) and `research/reports/strategy_hunt/2026-09-25-strategy-hunt-b1-5.md` §B1.6 (why slower families: the only positive pooled unconditional family after costs was `mr_20d` long).

## Global Constraints
- Same as B1/B1.5/B2: explicit-path commits with the trailer; never `git add -A`; never touch runtime files, `data/`, `models/`; tests offline; `strategies/families` pure (no core/api/research imports); `core/api/strategies` never import `research`; files under 400 lines; immutable style; explicit error handling.
- Known-at-entry: every signal reads only bars closed before the fill bar; context columns only from `bar_of_day ≥ 1` (≥ 3 for `ctx_rvol_open15`); daily features are lagged one session (`d_*`).
- Multiday exits: `time_stop` counts sessions, exit at the last bar of the final session (existing); `overnight` exits at the first bar (open) of the next session.
- Live bot untouched (no flags, no restart).
- Commit trailer:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BT1pvoczNh7y1pBxVN53R9
```

---

### Task 1: Overnight hold kind and five swing2 families

**Files:** modify `research/strategy_hunt/sim/simulator.py` (`_limit`: `hold_kind == "overnight"` → exit position `tb.first_pos[s0+1]` with reason `next_open`; `data_end` if no next session; earnings-eve rule still applies), `strategies/families/common.py` (`make_entry` accepts `hold_kind` in {`intraday`,`multiday`,`overnight`}); create `strategies/families/swing2.py` (+ register `FAMILIES_SWING2` in `registry.py`, expected registry size 18); tests `tests/research/strategy_hunt/test_simulator_overnight.py`, `tests/strategies/families/test_swing2.py`, `test_registry.py`.

**Families (exact; all signals evaluated on bar `i`, fill at bar `i+1` open via `make_entry`; `d_*` columns are prior-session daily features from `indicator_frame`; `ctx_*` via `ctx(ind, name)`):**
1. `overnight_hold` (sides long, short; hold_kind `overnight`): signal at `bar_of_day == 76` (fill at the 15:55 bar's open), exit next open. Long when `d_ret1 <= -down_pct` (a down day) and `ctx_rs_20 >= rs_min`; short when `d_ret1 >= up_pct` and `ctx_rs_20 <= -rs_min`. Stop `stop_atr × d_atr14` (protective only), target 20 R (effectively none). Grid: `down_pct/up_pct ∈ {1.0, 2.0}`, `rs_min ∈ {0, 5}`, `stop_atr ∈ {1.5, 3.0}`.
2. `mr_short_term` (long, short; multiday): long when `d_rsi2 <= rsi_lo` (add `d_rsi2` to `daily_features` if absent: 2-period RSI on close) and `d_close > d_sma200` (uptrend filter; add `d_sma200`); short when `d_rsi2 >= rsi_hi` and `d_close < d_sma200`. Signal at `bar_of_day == 0` (uses prior-session values), fill at bar 1. Exit `time_stop_days` sessions or when the daily close crosses `d_sma5` (approximate with the time stop; document). Stop `2.5 × d_atr14`, target 20 R. Grid: `rsi_lo ∈ {5, 10}`, `rsi_hi ∈ {90, 95}`, `time_stop_days ∈ {2, 3, 5}`.
3. `multi_day_drop` (long; multiday): three consecutive down closes (`d_ret1 < 0` for the last three sessions — add `d_down_streak`) and `d_close > d_sma200`; fill at bar 1; time stop `{3, 5}` sessions; stop `2.0 × d_atr14`; target `{1.0, 2.0}` × `d_atr14` above entry. Grid as listed.
4. `pead_wide` (long, short; multiday): `ctx_earnings_prev` true and `ctx_eps_surprise_pct ≥ surprise_min` (long) / `≤ −surprise_min` (short), no ADV-bucket restriction, signal at `bar_of_day == 5`, fill at bar 6; stop `1.5 × d_atr14`, target `3 R`, time stop `{3, 5, 10}` sessions. Grid: `surprise_min ∈ {2, 5, 10}`.
5. `rs_rotation` (long; multiday): signal on the first session of each week (`ind` needs a `d_week_start` flag: first bar of the first session whose weekday ≤ previous session's weekday); `ctx_rs_rank_60 ≥ rank_min` and `d_close > d_sma50`; fill at bar 1; time stop `{5, 10}` sessions; stop `2.5 × d_atr14`; target 20 R. Grid: `rank_min ∈ {0.9, 0.95}`.
Keep grids ≤ 24 combos per family. Each family: `spaced_by_sessions` one entry per symbol per hold.

**Tests:** simulator `overnight` exits at the next session's first bar open with reason `next_open` (and `data_end` on the last session; earnings-eve exit still fires); each family fires on a constructed two-week frame and not when its filter fails; no context → nothing (except `mr_short_term`, `multi_day_drop`, which need no ctx); registry size 18.

- [ ] TDD; commit `feat(families): overnight hold kind and five slower swing families (swing2)`.

---

### Task 2: Holdout window and search accounting

**Files:** modify `research/strategy_hunt/walkforward/windows.py` (`build_windows(..., holdout_months=0)`: OOS windows end at `last_bar − holdout_months`), `research/strategy_hunt/run.py` (`process_symbol`/`run_symbols` pass `holdout_months` through; `_config_key` includes it), `research/strategy_hunt/cli_args.py` (`--holdout-months`, default 0), `research/strategy_hunt/report.py` (new "Search accounting" section: families run, grid combos × sides evaluated, unconditional rows scored, PASS/WATCH counts, and the ratio; read families/grids from the registry and the run manifest), `research/strategy_hunt/run.py::write_manifest` (record `holdout_months` and `families`); tests `test_windows.py`, `test_run.py`, `test_report.py`, `test_cli_guards.py`.

- [ ] TDD; commit `feat(strategy-hunt): holdout window and search accounting`.

---

### Task 3: Run Batch 1 (execution)
- [ ] Env as before. `run --families overnight_hold,mr_short_term,multi_day_drop,pead_wide,rs_rotation --holdout-months 6 --workers 12 --force` (real) and `run --shuffle-seed 7 …` with the same flags (baseline), `score --postgres`, `report`. Read the report: pooled and unconditional PASS/WATCH per new family, exp_R, PF, t_session, recovery, excess over baseline.
- [ ] Only for families with a pooled or symbol PASS (or ≥ 3 WATCH rows with positive excess): rerun those families with `--holdout-months 0` and report holdout-period results separately (the last 6 months) — this is the confirmation test, run once.
- [ ] Report `research/reports/strategy_hunt/2026-09-25-strategy-search-batch1.md`: results table per family, search accounting, holdout confirmation, verdict and what Batch 2 (execution variants) should test. CHANGELOG. Commit artifacts and docs.
