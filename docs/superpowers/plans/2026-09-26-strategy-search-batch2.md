# Strategy Search Batch 2 — Selection Control, Limit Execution, Factor Sleeve

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Answer two questions Batch 1 could not: (1) is the cost model the whole gap for the flat-but-positive slow families (limit entries instead of market fills), and (2) does a low-turnover factor sleeve have *selection* skill — measured against a control that keeps each family's entry schedule and reassigns the trades to random symbols, because the existing path-shuffle keeps each day's return and therefore cannot test selection.

**Architecture:** Three independent additions to `research/strategy_hunt` and `strategies/families`, then one run in its own root. (A) A *selection baseline* variant: replay every real trade on a seeded random peer symbol at the same entry time with the same relative stop/target and hold, simulate with the same cost model, write `trades_selection/`; scoring uses the selection baseline for `multiday`/`overnight` families and the path-shuffle baseline for `intraday`. (B) An execution style on `Entry` (`market` | `limit_open`): a limit at the fill bar's open minus/plus a fixed offset, filled only if the bar trades through it, at maker cost (no spread, no slippage); a grid axis on the four slow families. (C) Two factor families with monthly signals (12-1 momentum decile, low-volatility trend) fed by three new context columns.

**Spec:** Batch 1 report `research/reports/strategy_hunt/2026-09-25-strategy-search-batch1.md` ("What this means for the bot").

## Global Constraints
- Same as Batch 1 (explicit-path commits with the trailer; tests offline; `strategies/families` pure; `core/api/strategies` never import `research`; files under 400 lines; immutable style; known-at-entry; live bot untouched; subsets of families always run in a separate root).
- Interfaces between tasks (fixed here so the three tasks can run in parallel):
  - `Entry` gains `entry_style: str = "market"` (Task B); `SimTrade`/`TRADE_COLUMNS` gain `entry_style` (Task B). Task A reads `entry_style` from the trades parquet with default `"market"` when the column is absent and copies it onto the replayed entry.
  - New paths variant name `"selection"` → `trades_selection/`, `checkpoints_selection/`, `choices_selection/` (Task A, in `paths.py`).
  - New context columns `mom_12_1`, `mom_rank_12_1`, `vol_rank_60` appended to `CTX_COLUMNS` (Task C) — the builder must emit them; older context files lacking them yield NaN through `ctx()` (already the contract).
- Commit trailer:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BT1pvoczNh7y1pBxVN53R9
```

---

### Task A: Selection baseline (random-peer replay)

**Files:** create `research/strategy_hunt/selection_baseline.py`; modify `research/strategy_hunt/paths.py` (variant `selection`), `research/strategy_hunt/cli_args.py` (`--shuffle-mode {path,selection}`, default `path`), `research/strategy_hunt/cli.py::cmd_run` (when `--shuffle-mode selection`: run the replay instead of the family simulation; `cmd_all` runs both baselines), `research/strategy_hunt/run.py::score_regime_root` (baseline rows: for pooled groups whose trades are `intraday` use `trades_shuffle/`, for `multiday`/`overnight` use `trades_selection/` when present; each pooled row records `baseline_kind` ∈ {`path`,`selection`,`none`}; `no_baseline` when the needed kind is missing), `score/regime_rows.py` (`baseline_kind` field, `ROW_KEYS`), `report.py` (show `baseline_kind`); tests.

**Replay algorithm (`replay_selection(root, seed, workers)`):** load every `trades/<S>.parquet` (real run); for each trade t (symbol S, `entry_ts`, side, `entry_px`, `stop_px`, `target_px`, `time_stop`, `hold_kind`, `entry_style`, family, params) draw a peer P ≠ S uniformly (seeded by `(seed, S, entry_ts)`) from the symbols whose `bars5` contain a bar at `entry_ts`; build an `Entry` on P: signal bar = P's bar at `entry_ts` minus one bar (so `make_entry` fills at P's bar at `entry_ts`), `stop_px = P_fill_open × (1 − sgn·|S.entry − S.stop|/S.entry)`, `target_px` likewise, same `time_stop`, `hold_kind`, `entry_style`, tag `sel:<family>`; group entries by P and call `simulate` once per P with P's cost model, earnings set, the regime lookup (same as the real run) and `family` = the original family name; write `trades_selection/<P>.parquet` with the same `TRADE_COLUMNS` (symbol = P). Trades whose peer has no bar at `entry_ts` after 3 redraws are dropped and counted in the manifest (`selection_dropped`). Determinism test: same seed → identical output.

**Tests:** paths; replay on a 3-symbol toy root (every real trade replayed on a different symbol at the same timestamp, relative stop preserved to 1e-9, `entry_style` copied, dropped-count when no peer has the bar); `score_regime_root` picks the baseline by hold kind and sets `baseline_kind`; CLI flag and `cmd_all` order (real → shuffle → selection → score → playbook).

- [ ] TDD; commit `feat(strategy-hunt): selection baseline — random-peer replay of the real trade schedule`.

---

### Task B: Limit-at-open execution style

**Files:** modify `strategies/families/models.py` (`Entry.entry_style: str = "market"`, `ENTRY_STYLES = ("market", "limit_open")`), `strategies/families/common.py` (`make_entry(..., entry_style="market")`, validated), `research/strategy_hunt/models.py` (`SimTrade.entry_style`, `TRADE_COLUMNS`), `research/strategy_hunt/sim/simulator.py` (fill logic), `research/strategy_hunt/sim/costs.py` (`entry_fill(..., style)`: `limit_open` pays commission only, no spread/slippage/calibrated override), families `swing2.py` (`overnight_hold`, `mr_short_term`) and `swing.py` (`mr_20d`) plus `rs_rotation`: grid axis `entry_style ∈ {market, limit_open}` (keep each grid ≤ 24); tests.

**Fill rule (`limit_open`, offset `LIMIT_OFFSET_BPS = 10`):** limit = fill bar open × (1 − 0.0010) for longs, × (1 + 0.0010) for shorts; filled at the limit price if the fill bar's low ≤ limit (long) / high ≥ limit (short); otherwise no trade (counted as `unfilled` in the family's choice record if cheap, else ignored). Everything after the fill is unchanged. Document the assumption (a resting limit at the open is a maker order; no queue-position model) in the module docstring.

**Tests:** long/short fill and no-fill cases; cost model applies commission only for `limit_open`; `market` byte-identical to before (regression on an existing simulator test); `TRADE_COLUMNS` round-trip through the writer; grids still ≤ 24 combos; registry still 18.

- [ ] TDD; commit `feat(strategy-hunt): limit-at-open execution style for slow families`.

---

### Task C: Factor context columns and two factor families

**Files:** modify `research/strategy_hunt/context.py` (`mom_12_1` = close.shift(21)/close.shift(251) − 1 minus SPY's same quantity, ×100; `mom_rank_12_1` = same-date percentile; `vol_rank_60` = same-date percentile of the 60-session std of daily log returns, lagged one session), `strategies/families/context.py` (`CTX_COLUMNS` + 3), `strategies/families/indicators.py` (`d_month_start` flag: first session of a calendar month, date property, not lagged); create `strategies/families/factor.py` (`FAMILIES_FACTOR`, registry → 20):
- `mom_12_1` (long, short; multiday): signal at `bar_of_day == 0` on `d_month_start` sessions; long when `ctx_mom_rank_12_1 ≥ rank_min` and `d_close > d_sma200`; short when `ctx_mom_rank_12_1 ≤ 1 − rank_min` and `d_close < d_sma200`; fill bar 1; stop `3.0 × d_atr14`; target 20 R; time stop `{20}` sessions; grid `rank_min ∈ {0.8, 0.9}`, `entry_style ∈ {market, limit_open}` (if Task B has landed; else `market` only — check `ENTRY_STYLES` at implementation time and say which).
- `low_vol_trend` (long; multiday): month-start signal; `ctx_vol_rank_60 ≤ vol_max` and `ctx_rs_120 > 0` and `d_close > d_sma200`; stop `2.5 × d_atr14`; target 20 R; time stop 20; grid `vol_max ∈ {0.2, 0.3}`.
Tests: new columns lagged and ranked within date; `d_month_start` flags exactly the first session of each month; each family fires/does not fire; registry 20.

- [ ] TDD; commit `feat(families): factor context columns and monthly factor families (mom_12_1, low_vol_trend)`.

---

### Task D: Run Batch 2 (execution)
- [ ] Root `data/strategy_hunt_batch2`: symlink `bars5 daily news earnings splits`, copy `calibration.json`, **real directory** `context/` rebuilt with `universe-context --root data/strategy_hunt_batch2` (new columns). Families: `mr_short_term,overnight_hold,mr_20d,rs_rotation,mom_12_1,low_vol_trend`; `--holdout-months 6`; real run, `--shuffle-seed 7` (path) run, `--shuffle-seed 7 --shuffle-mode selection` run; `score`; `report`.
- [ ] Report `research/reports/strategy_hunt/2026-09-26-strategy-search-batch2.md`: per family × entry_style: exp_R, PF, n, path-baseline excess (intraday-timing) and selection-baseline excess (selection skill), t_session, verdicts; the limit-vs-market delta; search accounting; holdout confirmation only if triggered; verdict. CHANGELOG; commit artifacts.
