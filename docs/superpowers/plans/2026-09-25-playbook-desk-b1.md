# Playbook Desk Phase B1 (Research First) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add cross-sectional "in play" context, three evidence-backed families, regime-conditioned fit rows with pooled lanes, a regime-aware reader and gate, and a FinBERT sentiment backend; then rerun the full-universe research and report.

**Architecture:** The research pipeline builds a daily cross-sectional context table for the whole universe; the runner joins each symbol's slice onto the indicator frame as `ctx_*` columns (known-at-open semantics), so families keep their `(bars5, ind, daily, news, params)` signature. The scorer emits regime-keyed rows per symbol and pooled per family; `core/strategy_fit.py` resolves symbol-regime → pooled-regime → unconditional; the gate passes the signal's `market_context_regime`. Sentiment gets a pluggable backend used identically by the research news cache and the live NewsBus.

**Tech Stack:** as before (pandas, numpy, ta, sklearn, yfinance, transformers + torch CUDA for FinBERT, pytest); `$PY=/home/nvidia/anaconda3/envs/trading-bot/bin/python`.

**Spec:** `docs/superpowers/specs/2026-09-25-playbook-desk-design.md` (§4–6, §7.5, §9 B1). Prior system: `docs/superpowers/plans/2026-09-25-strategy-hunt-full-system.md`.

## Global Constraints

- Same as the previous plan: explicit-path commits with the trailer, never `git add -A`, never touch runtime files or `data/`, tests offline, `core/api/strategies` never import `research`, `strategies/families` pure, files under 400 lines, immutable style. Branch `feature/newsbus-autonomy-decision-card`, no push.
- Context semantics: every `ctx_*` value on a bar must be knowable at that bar's open. Columns derived from prior sessions are lagged one session; `ctx_rvol_open5`/`ctx_rvol_open15`/`ctx_gap_pct` are computed from the current session's first bars and are NaN on bars 0 (and 0–2 for open15), so families may only read them from `bar_of_day ≥ 1` (≥ 3).
- Regime labels: `risk_on`, `risk_off`, `mixed`, `unknown` (from `data/regime.py`).
- Commit trailer:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BT1pvoczNh7y1pBxVN53R9
```

---

## File Structure

```
research/strategy_hunt/data/earnings.py     + surprise columns, load_earnings_table
research/strategy_hunt/context.py           symbol_daily_features, rank_cross_section, build_universe_context, load_context_slice
strategies/families/context.py              CTX_COLUMNS, ctx(ind, name), in_play(ind, top_n)
strategies/families/indicators.py           indicator_frame(..., context=None) joins ctx_* by session date
strategies/families/inplay.py               orb_inplay, pead_smallcap, rs_leader, FAMILIES_INPLAY
strategies/families/registry.py             + FAMILIES_INPLAY
research/strategy_hunt/walkforward/runner.py  run_symbol_family(..., context=None)
research/strategy_hunt/run.py               loads context slice; context CLI step
research/strategy_hunt/score/regime_rows.py regime_rows(), pooled_rows(), POOLED_FLOORS
research/strategy_hunt/fit/writer.py        snapshot gains regime_rows, pooled_rows
core/strategy_fit.py                        lookup_regime, RegimeRow parsing
core/strategy_fit_gate.py                   evaluate_fit_gate(..., regime=None)
core/engine.py                              gate call passes signal.reasoning["market_context_regime"]
research/strategy_hunt/data/sentiment.py    score_headlines(backend)
core/sentiment_backend.py                   shared backend selection (vader | finbert)
core/loops/news_loop.py                     uses core.sentiment_backend
research/strategy_hunt/data/news.py         stores sentiment_vader and sentiment_finbert (when available)
research/strategy_hunt/cli.py               `context` step builds the universe context after export
tests/...                                   one module per source module
```

---

### Task 1: Earnings surprise in the earnings cache

**Files:** modify `research/strategy_hunt/data/earnings.py`; test `tests/research/strategy_hunt/test_earnings.py` (extend).

**Interfaces:**
- `update_earnings(root, symbol, fetcher)` now accepts a fetcher returning either a `DatetimeIndex` (legacy) or a DataFrame indexed by date with columns `EPS Estimate`, `Reported EPS`, `Surprise(%)`; the cache Parquet stores `date, eps_estimate, reported_eps, surprise_pct` (NaN when absent). Return value unchanged (row count).
- `load_earnings_table(root, symbol) -> DataFrame` indexed by normalized `date` with those three float columns (empty frame with columns when missing). `load_earnings` (set of dates) unchanged.
- `yfinance_fetcher(symbol)` returns the DataFrame (rows with `Event Type == "Earnings"` only, tz dropped), empty DataFrame on failure.

- [ ] Tests (add): fetcher returning a DataFrame with two rows (one with surprise 5.5, one NaN) → `load_earnings_table` has both rows with the right values and `load_earnings` still returns the two dates; legacy DatetimeIndex fetcher still works (NaN surprise columns).
- [ ] Implement; run `$PY -m pytest tests/research/strategy_hunt/test_earnings.py -q -p no:cacheprovider`; commit `feat(strategy-hunt): earnings cache stores EPS surprise`.

---

### Task 2: Universe context builder

**Files:** create `research/strategy_hunt/context.py`; test `tests/research/strategy_hunt/test_context_builder.py`.

**Interfaces:**
- `symbol_daily_features(symbol, bars5, daily, spy_daily, earnings_table, news) -> DataFrame` indexed by session `date` with columns: `rvol_open5`, `rvol_open15` (first-bar / first-three-bars volume ÷ mean of the same over the prior 14 sessions; NaN when < 5 prior sessions), `gap_pct` (session open ÷ prior close − 1, ×100), `rs_20`, `rs_60`, `rs_120` (symbol close return over N sessions minus SPY's, through the prior session, ×100), `pct_from_52w_high` (prior close ÷ rolling 252-session high of closes − 1, ×100), `adv_dollars_20` (prior 20 sessions mean close×volume), `atr14_pct` (prior-session ATR14 ÷ prior close ×100), `earnings_today` (bool), `earnings_prev` (bool: an earnings date equals the prior session, or a date between the prior session's close and this session's open), `eps_surprise_pct` (from the table on `earnings_prev` days), `news_count_24h`, `news_sent_24h` (items with `created_at` in the 24 h before the session open; mean sentiment).
- `rank_cross_section(frames: dict[str, DataFrame]) -> DataFrame` concatenating all symbols (`symbol` column) and adding per-date `rvol_rank5` (1 = highest `rvol_open5`, NaN rows unranked), `rs_rank_60` (percentile 0–1 of `rs_60`), `adv_bucket` (mega ≥ 1e9, large ≥ 2e8, mid ≥ 2e7, else small).
- `build_universe_context(root, spy_symbol="SPY") -> Path` writes `<root>/context/universe_daily.parquet` (columns `date, symbol, ...`) using `load_adjusted`, `load_earnings_table`, `load_news` for every symbol under `bars5/` except SPY/VIX; never raises per symbol (log and skip).
- `load_context_slice(root, symbol) -> DataFrame` indexed by `date` for one symbol (empty with columns when missing). Reads the Parquet with a `symbol ==` filter (pyarrow filters).

- [ ] Tests: synthetic 40-session bars for three symbols with distinct opening volumes → `rvol_open5` correct by hand on day 20, `rvol_rank5` orders them, `gap_pct` correct, `rs_20` = symbol return − SPY return, `pct_from_52w_high ≤ 0`, `earnings_prev` and `eps_surprise_pct` set on the session after an earnings date, `news_count_24h` counts only the prior 24 h, `adv_bucket` thresholds; `build_universe_context` on a tmp root writes the file and `load_context_slice` returns one symbol's frame.
- [ ] Implement (keep the module under 400 lines; split ranking into `context_rank.py` if needed); run tests; commit `feat(strategy-hunt): universe context builder (in-play RVOL, RS ranks, catalysts)`.

---

### Task 3: Context accessors and indicator-frame join

**Files:** create `strategies/families/context.py`; modify `strategies/families/indicators.py`; tests `tests/strategies/families/test_context.py`, extend `test_indicators.py`.

**Interfaces:**
- `strategies/families/context.py`: `CTX_COLUMNS = ("rvol_open5", "rvol_open15", "gap_pct", "rs_20", "rs_60", "rs_120", "rs_rank_60", "rvol_rank5", "pct_from_52w_high", "adv_dollars_20", "adv_bucket", "atr14_pct", "earnings_today", "earnings_prev", "eps_surprise_pct", "news_count_24h", "news_sent_24h")`; `ctx(ind, name) -> np.ndarray` returns `ind["ctx_"+name]` as float array (bools as 0/1, `adv_bucket` mapped mega=3, large=2, mid=1, small=0), all-NaN if missing; `in_play(ind, top_n) -> np.ndarray[bool]` = `ctx_rvol_rank5 <= top_n`.
- `indicator_frame(bars5, daily, spy5=None, context=None)`: when `context` (indexed by date) is given, join its columns prefixed `ctx_` onto every bar of the matching session date (`reindex(bars5.index.normalize())`). Existing behavior unchanged when `context is None`.

- [ ] Tests: `ctx` mapping and NaN default; `in_play`; `indicator_frame` with a two-day context joins the right values to each session's bars and leaves other columns unchanged.
- [ ] Implement; run; commit `feat(families): context accessors and ctx_* join in indicator_frame`.

---

### Task 4: In-play families

**Files:** create `strategies/families/inplay.py`; modify `strategies/families/registry.py`; test `tests/strategies/families/test_inplay.py`, update `test_registry.py` (13 families).

**Interfaces (all mask-based, pure, `(bars5, ind, daily, news, params)`):**
- `orb_inplay_entries`: at bar `i` with `bar_of_day == 1` (the second bar, after the first 5-min bar closed): require `ctx_rvol_rank5 <= top_n`, `close_of_bar0 >= 5`, `ctx_atr14_pct >= 0.5`; let `h0,l0,o0,c0` be bar 0's values; long if `c0 > o0` and `high[i] > h0`; short if `c0 < o0` and `low[i] < l0`; entry via `make_entry(bars5, i, ...)` (fills at bar i+1 open); stop long `= max(entry − stop_atr×ATR14$, l0)` where ATR14$ = `ctx_atr14_pct/100 × entry`; short mirror; target: if `trail_r == 0` a far target (entry + 20×risk) with `time_stop = bars_to_1500(bar_of_day+1)` … no: hold to close means `time_stop = 77 − (i+1)` bars (the 15:55 bar), so use `time_stop = max(1, 77 − (i + 1))`; if `trail_r > 0` the simulator has no trailing stop, so emulate with target `= entry ± trail_r×risk` (document: research approximates "trail once ≥ trail_r R" by a take-profit at trail_r R). Tag `orb_inplay_long/short`. One per session per side.
- `pead_smallcap_entries`: at bar `i` with `bar_of_day == 6` (10:00 bar closed): require `ctx_earnings_prev == 1`, `ctx_adv_bucket in {mid, small}` (i.e. ≤ 1), `abs(ctx_gap_pct) >= min_gap`, long if `ctx_eps_surprise_pct >= s` and `close[i] > session_open` (holding), short if `<= −s` and `close[i] < session_open`; entry at bar i+1 open; stop 1.5×ATR14$ (from `ctx_atr14_pct`); target 3R; `time_stop = hold_days`, `hold_kind = multiday`; tags `pead_long/short`; spaced by `hold_days`.
- `rs_leader_entries`: at each session's first bar `j` (`bar_of_day == 0`) via `make_entry(bars5, j−1, ...)`: `ctx_rs_rank_60 >= rs_floor`, `ctx_pct_from_52w_high >= −15`, `d_low <= d_emaX <= d_close` (X from `ema`: `d_ema10`/`d_ema20`), `d_ret1 > 0`; stop `atr_stop × d_atr14`; target 3R; `time_stop = 20`, multiday; tag `rs_leader`; spaced by 20.
- `FAMILIES_INPLAY`: `orb_inplay` grid `{"top_n": [10, 20, 30], "stop_atr": [0.05, 0.10, 0.20], "trail_r": [0, 5, 10]}` intraday both sides; `pead_smallcap` grid `{"s": [5, 10, 20], "hold_days": [5, 10, 20], "min_gap": [0, 2, 4]}` multiday both sides; `rs_leader` grid `{"ema": [10, 20], "rs_floor": [0.85, 0.90, 0.95], "atr_stop": [1.5, 2.0, 2.5]}` multiday long.

- [ ] Tests: one golden test per family (synthetic bars + ind with the needed `ctx_*`/`d_*` columns; entry timestamp, side, stop/target arithmetic, and the negative case for the key filter), plus registry = 13 names and grids ≤ 27.
- [ ] Implement; run `tests/strategies/families`; commit `feat(families): orb_inplay, pead_smallcap, rs_leader`.

---

### Task 5: Runner and orchestration pass context

**Files:** modify `research/strategy_hunt/walkforward/runner.py` (`run_symbol_family(..., context=None)` → `indicator_frame(bars5, daily, spy5, context)`), `research/strategy_hunt/run.py` (load `load_context_slice(root, symbol)` in `process_symbol` and pass it; `__version__` bump to `0.3.0` so checkpoints invalidate), `research/strategy_hunt/cli.py` (`context` subcommand also calls `build_universe_context` after `ensure_context`; new `universe-context` subcommand to rebuild only that); tests `test_runner.py`, `test_run.py` (extend: a toy family that reads `ctx_rvol_rank5` fires only when the context slice is present).
- [ ] Implement; run; commit `feat(strategy-hunt): pipeline passes universe context to families`.

---

### Task 6: Regime rows, pooled rows, snapshot, reader, gate, engine

**Files:** create `research/strategy_hunt/score/regime_rows.py`; modify `research/strategy_hunt/fit/writer.py`, `research/strategy_hunt/run.py` (`score_root` adds them), `core/strategy_fit.py`, `core/strategy_fit_gate.py`, `core/engine.py` (gate call), `research/strategy_hunt/promote.py` (pooled PASS counts as evidence: `pass_symbols` unchanged, add `pooled_pass_regimes` column and treat a pooled PASS as clearing `no_pass_symbols`); tests `test_regime_rows.py`, `test_writer.py`, `test_strategy_fit.py`, `test_strategy_fit_gate.py`, `test_strategy_fit_wiring.py`, `test_promote.py`.

**Interfaces:**
- `regime_rows(symbol, family, side, trades, n_windows, computed_at) -> list[dict]`: one `build_row`-shaped dict per regime present with ≥ 1 trade, plus key `regime`; verdict via the same floors.
- `POOLED_FLOORS = {"min_n": 500, "min_sessions": 60, "min_pf": 1.2, "min_wr": 0.45, "min_exp_r": 0.05, "max_dd_r": 60.0, "min_stability": 0.6, "min_decay": 0.5, "min_t_stat": 3.0, "min_exp_r_lb": 0.0}`; `pooled_rows(all_trades: DataFrame, computed_at) -> list[dict]` per `(family, side, regime)` with `symbol="*"`, `source="pooled"`, metrics from `compute_metrics` on the concatenated trades (windows = distinct `window_id`), verdict against `POOLED_FLOORS`.
- Snapshot JSON gains `"regime_rows": {symbol: {"strategy:side:regime": row}}` and `"pooled_rows": {"strategy:side:regime": row}`; Parquet gets `fit_regime_<date>.parquet` and `fit_pooled_<date>.parquet`.
- `StrategyFit.lookup_regime(symbol, strategy, side, regime) -> FitRow | None`: symbol-regime row → pooled regime row (a `FitRow` with `symbol="*"`, `source="pooled"`) → `lookup(symbol, strategy, side)` → None. `summary()` gains `n_regime_rows`, `n_pooled_pass`.
- `evaluate_fit_gate(..., regime: str | None = None)`: when `regime` in (risk_on, risk_off, mixed) use `lookup_regime`, else `lookup`; audit fields add `regime` and `pooled`. Engine passes `signal.reasoning.get("market_context_regime")` when `Config().STRATEGY_FIT_REGIME_ROWS` (new flag, default True).
- [ ] Tests for each piece (precedence order in `lookup_regime`; pooled floors; snapshot shape; gate with regime; wiring test for the new kwarg; promote treats pooled PASS as evidence).
- [ ] Implement; run all affected suites; commit `feat(strategy-hunt): regime-conditioned fit rows, pooled lanes, regime-aware gate`.

---

### Task 7: Sentiment backend (VADER | FinBERT on GPU)

**Files:** create `core/sentiment_backend.py`, `research/strategy_hunt/data/sentiment.py` (thin re-export for research), modify `research/strategy_hunt/data/news.py` (store `sentiment_vader` always and `sentiment_finbert` when the backend is available; `sentiment` column = the selected backend), `core/loops/news_loop.py` (use `core.sentiment_backend.score`), `core/config.py` (`SENTIMENT_BACKEND` env/yaml default `vader`); tests `tests/core/test_sentiment_backend.py`, `test_news.py` (extend).

**Interfaces:** `score(headlines: list[str], backend: str | None = None) -> list[float]` in [−1, 1]; `finbert` maps class probabilities to `p_pos − p_neg` using `ProsusAI/finbert` via transformers on CUDA if available (lazy-loaded singleton, batch size 32, max length 64); falls back to VADER with a one-time warning if transformers/torch/model are unavailable; `available_backends()`; tests use a fake pipeline injected via a module-level factory so no model download occurs in tests.
- [ ] Implement; run; commit `feat(sentiment): pluggable VADER/FinBERT backend shared by research and live`.

---

### Task 8: Full B1 research run, report, and docs (execution)

- [ ] `set -a; . ./.env; set +a; export POSTGRES_DSN=$(...DEFAULT_DSN...)`; `$PY -m research.strategy_hunt export --min-days 250` (incremental), `context` (rebuilds universe context), `earnings` (now with surprises), `news` (adds finbert column if `SENTIMENT_BACKEND=finbert` for the research cache: run once with `SENTIMENT_BACKEND=finbert` to populate), `run --workers 12 --force` (version bump invalidates checkpoints anyway), controls on the 50-symbol sample as before, `score --postgres`, `promote`.
- [ ] Read the report: pooled regime rows are the headline now; compare the three new families with the old ten; note any PASS.
- [ ] Update `docs/research/strategy_hunt.md` (context, new families, regime rows, sentiment backend) and `CHANGELOG.md`; commit report artifacts and docs.
- [ ] Deliver: ranked pooled lanes by regime, any PASS rows, and the B2 go/no-go per spec §9.

---

## Self-review notes

- Interfaces reconciled: families keep the 5-argument signature; context reaches them through `ind` (`ctx_*`), so the live adapter needs only to compute the same columns for the watchlist (B2). `lookup_regime` precedence is fixed here and consumed by the gate; `promote` reads pooled PASS from the snapshot's new section.
- The simulator has no trailing stop; `orb_inplay`'s `trail_r` is approximated by a take-profit at `trail_r` R and documented as such; a trailing-stop simulator option is a follow-up.
- `orb_inplay` reads `ctx_rvol_rank5` at `bar_of_day == 1`; the context builder computes `rvol_open5` from bar 0 only, so this respects known-at-open semantics.
