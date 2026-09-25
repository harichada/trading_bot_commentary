# Playbook Desk B1.5 — Pooled Verdict Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a pooled regime PASS mean "edge beyond the regime's own drift", so the B1 go/no-go gate cannot be satisfied by noise, then re-score the B1 run under the hardened rules.

**Architecture:** Three changes to the offline scorer, nothing in the live path beyond reader tolerance. (1) Pooled metrics gain a session-clustered t-statistic and a recovery factor; the pooled floors drop the absolute drawdown cap in favour of those. (2) The shuffled control becomes a first-class *baseline variant* inside the same data root (`trades_shuffle/`), computed by `run --shuffle-seed`, and `score` subtracts the baseline's pooled expectancy from every pooled row; a pooled PASS requires positive excess over the baseline. (3) The report shows pooled rows against their baseline and counts pooled verdicts in the controls; machine reports get a run-stamped filename.

**Tech Stack:** existing `research/strategy_hunt` (pandas, numpy, pyarrow), `core/strategy_fit_rows.py` reader.

**Spec:** `docs/superpowers/specs/2026-09-25-playbook-desk-design.md` §6 (regime-conditioned fit) as amended by the B1 run report `research/reports/strategy_hunt/2026-09-25-strategy-hunt-b1.md` ("Controls" and "B2 go/no-go"): the shuffle control keeps each session's open-to-close return, so pooled regime rows measure regime drift plus noise unless benchmark-adjusted.

## Global Constraints

- Same as B1: explicit-path commits with the trailer, never `git add -A`, never touch runtime files, `data/`, or `models/`; tests offline and never loading a real model; `core/api/strategies` never import `research`; files under 400 lines; immutable style; explicit error handling.
- Live safety: no change to gate defaults (`ENABLE_STRATEGY_FIT_GATE` off, `STRATEGY_FIT_SHADOW` on); `core/strategy_fit_rows.py::FitRow.from_dict` must ignore unknown keys so new metric fields never break the live reader.
- Regime labels: `risk_on`, `risk_off`, `mixed`, `unknown`.
- Commit trailer:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BT1pvoczNh7y1pBxVN53R9
```

---

### Task 9: Pooled metrics — session-clustered t-stat and recovery factor

**Files:** modify `research/strategy_hunt/models.py` (`FitMetrics` gains `t_stat_session: float = 0.0` and `recovery: float = 0.0` as the last two fields, with defaults so positional construction elsewhere keeps working), `research/strategy_hunt/score/metrics.py` (`compute_metrics` fills both), `research/strategy_hunt/score/regime_rows.py` (`POOLED_FLOORS`, `_pooled_failures`), `core/strategy_fit_rows.py` only if `FitRow.from_dict` does not already ignore unknown keys; tests `tests/research/strategy_hunt/test_metrics.py` (extend or create), `test_regime_rows.py`, `tests/core/test_strategy_fit.py` (one test: a row dict with extra keys `t_stat_session`, `recovery`, `excess_r` parses).

**Definitions (exact):**
- `t_stat_session`: group trades by `entry_ts.normalize()`; `m_s` = mean `r_multiple` per session; `t_stat_session = mean(m_s) / (std(m_s, ddof=1) / sqrt(S))` where `S` = number of sessions; `0.0` when `S < 2` or the std is 0. Rationale: same-session trades across symbols are correlated; sessions are the independent unit.
- `recovery`: `net_r = sum(r_multiple)`; `recovery = net_r / max_dd_r` when `net_r > 0 and max_dd_r > 0`; `float("inf")` when `net_r > 0 and max_dd_r == 0`; `0.0` when `net_r <= 0`.
- `POOLED_FLOORS` becomes `{"min_n": 500, "min_sessions": 60, "min_pf": 1.2, "min_wr": 0.45, "min_exp_r": 0.05, "min_recovery": 3.0, "min_stability": 0.6, "min_decay": 0.5, "min_t_stat": 3.0, "min_t_stat_session": 3.0, "min_exp_r_lb": 0.0}` (drop `max_dd_r`). `_pooled_failures` keys follow the dict. Per-symbol `FLOORS` in `score/verdict.py` are unchanged.
- Update the comment block above `POOLED_FLOORS` (currently explains the 60R cap) to explain recovery + clustered t.

**Tests (exact cases):**
- `test_t_stat_session_is_lower_than_iid_t_for_correlated_sessions`: 5 sessions × 40 trades where every trade in a session has the same `r_multiple` (session values `[+1, +1, +1, −0.5, +1]`); assert `t_stat > t_stat_session` and `t_stat_session == pytest.approx(mean/std*sqrt(5))` computed by hand.
- `test_t_stat_session_single_session_is_zero`.
- `test_recovery_factor`: trades `[+2, −1, +2, −1, +2]` → net 4, max_dd 1 → recovery 4.0; all-negative → 0.0; no drawdown → inf.
- `test_pooled_floors_use_recovery_not_max_dd`: a metrics object with `max_dd_r=639`, `recovery=3.6`, all other floors met → no failures; `recovery=2.0` → `["min_recovery"]`; `t_stat_session=2.0` → `["min_t_stat_session"]`.
- `test_fit_row_from_dict_ignores_unknown_metric_keys` in `tests/core/test_strategy_fit.py`.

- [ ] Write the failing tests; run; implement; run; commit `feat(strategy-hunt): pooled floors use session-clustered t-stat and recovery factor`.

---

### Task 10: Shuffled baseline inside the root; pooled verdicts require excess over baseline; report

**Files:** modify `research/strategy_hunt/run.py` (variant-aware paths and `run_symbols(..., variant=None)`), `research/strategy_hunt/score/regime_rows.py` (`pooled_rows(all_trades, computed_at, baseline=None)`, `verdict_pooled(m, excess_r)`), `research/strategy_hunt/fit/writer.py` (snapshot section `baseline_pooled_rows`), `research/strategy_hunt/report.py` (pooled-vs-baseline table, pooled counts in controls, run-stamped filename), `research/strategy_hunt/cli.py` (`run --shuffle-seed` → variant `shuffle`; `cmd_all` runs the baseline; `--no-baseline`), docs strings; tests `test_run.py`, `test_regime_rows.py`, `test_writer.py`, `test_report.py`, `test_cli.py`.

**Interfaces:**
- Paths: `trades_path(root, symbol, variant=None)`, `checkpoint_path(...)`, `choices_path(...)`: `variant=None` → `<root>/trades/…` (unchanged); `variant="shuffle"` → `<root>/trades_shuffle/…`, `<root>/checkpoints_shuffle/…`, `<root>/choices_shuffle/…`. `process_symbol(..., variant=None)` and `run_symbols(..., variant=None)` thread it through; `cmd_run` sets `variant="shuffle"` whenever `--shuffle-seed` is given (a separate root is no longer needed for the shuffle control; `--cost-mult` controls keep using separate roots).
- `score_regime_root(root, families, computed_at)` additionally loads `<root>/trades_shuffle/*.parquet` when that directory exists, computes baseline pooled rows from them (same grouping, quarter buckets, `source="baseline"`), and passes them to `pooled_rows(all_trades, computed_at, baseline=baseline_rows)`. Returns `(regime_rows, pooled_rows, baseline_rows)`; `cmd_score` passes `baseline_pooled_rows=` to `write_fit`, which stores them under snapshot key `baseline_pooled_rows` (list of dicts; missing key tolerated by the reader — it never reads it).
- Each pooled row gains `baseline_exp_r` (float or NaN), `baseline_n` (int, 0 if none), `excess_r` (= `exp_r − baseline_exp_r`, NaN if no baseline). `verdict_pooled(m, excess_r)`: `POOLED_FLOORS` gains `"min_excess_r": 0.05`; failure key `min_excess_r` when `excess_r < 0.05`; failure key `no_baseline` when `excess_r` is NaN. A row with `no_baseline` can be WATCH but never PASS.
- Report: new section "Pooled rows vs shuffled baseline" (family, side, regime, n, exp_R, baseline exp_R, excess, t_session, recovery, verdict, failed floors) sorted by excess; `_controls_section` reports pooled PASS/WATCH counts for each control in addition to unconditional; `write_report` names the machine report `<date>T<HHMM>-strategy-hunt.md` (from `computed_at`) and keeps writing the manifest JSON next to it with the same stem, so consecutive runs no longer overwrite each other.
- `cmd_all`: after `cmd_run`, run the shuffle baseline (`shuffle_seed=7`, same symbols/families/workers, `force=True`) unless `--no-baseline`; then `cmd_score`. Register `--no-baseline` on the shared `common(sp)` flags.

**Tests (exact cases):**
- Paths: `trades_path(root, "AAA", "shuffle") == root/"trades_shuffle"/"AAA.parquet"`; `process_symbol` with `variant="shuffle"` writes only under the `_shuffle` dirs and leaves `trades/` untouched.
- `pooled_rows` with a baseline whose matching `(family, side, regime)` has `exp_r 0.10` and a real row `exp_r 0.12` → `excess_r 0.02`, failed contains `min_excess_r`; real `0.20` → excess `0.10`, no `min_excess_r`; no matching baseline group → `excess_r` NaN and `no_baseline` in failed, verdict at most WATCH.
- `score_regime_root` on a root with `trades/` and `trades_shuffle/` returns baseline rows with `source == "baseline"` and pooled rows carrying `baseline_exp_r`.
- Writer round-trip: `baseline_pooled_rows` persisted and `load_snapshot` returns it; `core.strategy_fit.StrategyFit` loads a snapshot containing the key without error (reuse the malformed-sections test pattern).
- Report: the baseline table renders the excess column; controls section prints `pooled PASS` counts; two `write_report` calls with different `computed_at` produce two files.
- CLI: `cmd_all` order recorder asserts `run` (variant None) → `run` (variant shuffle) → `score`; `--no-baseline` skips the second.

- [ ] Write the failing tests; run; implement; run; commit `feat(strategy-hunt): shuffled baseline in-root; pooled PASS requires excess over baseline; report`.

---

### Task 11: Re-score the B1 run under the hardened rules (execution)

- [ ] `SENTIMENT_BACKEND=vader $PY -m research.strategy_hunt news --rescore` (flips the research `sentiment` column back to VADER for parity with the live loop; keeps `sentiment_finbert`).
- [ ] Env as in B1 (`set -a; . ./.env; set +a`; `POSTGRES_DSN` from `backtest/engine.py:DEFAULT_DSN`). `context` (rebuild; `news_sent_24h` now VADER), `run --workers 12 --force` (13 families), `run --shuffle-seed 7 --workers 12 --force` (baseline, full universe, in-root), `score --postgres`, `report`, `promote`. Cost controls on `sample50` as before (separate roots, 2×). Log every step under `data/strategy_hunt/`.
- [ ] Report `research/reports/strategy_hunt/2026-09-25-strategy-hunt-b1-5.md`: pooled rows vs baseline (the headline), how many former WATCH rows survive the excess test, any PASS, the dt_momentum short risk_off row's excess and clustered t, the FinBERT→VADER effect on the news families, the go/no-go under the hardened gate (the spec's rule now reads: a pooled PASS counts only with positive excess over the shuffled baseline). Update `docs/research/strategy_hunt.md` (baseline variant, `--no-baseline`, new floors and fields, run-stamped reports), `CHANGELOG.md`, `docs/research/strategy_hunt_promotion.md`; commit `docs(strategy-hunt): B1.5 hardened pooled verdicts — run, report, docs`.

---

## Parked follow-ups (not in this plan)
- News-family thresholds are tuned to VADER's scale; a FinBERT variant needs its own threshold grid (read `sentiment_finbert` directly).
- Trailing-stop option in the simulator for `orb_inplay`.
- `pead_smallcap` / `rs_leader` are too thin to measure as specified; revisit the filters.
- `day_trade_momentum` and the short/ORB lanes do not publish `market_context_regime` into `reasoning` (Phase B2).
