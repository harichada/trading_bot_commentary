# Plan — Per-Domain Parallel Loops

**Status**: DRAFT for operator review (2026-05-12).
**Author**: senior dev (Claude).
**Trigger**: yesterday's `pnl_to_check` NameError fired inside the emergency-stop block of `_analysis_loop_body` and aborted *every* downstream step (screener, signals, state save) on every iteration. Operator critique: "each item on the UI should have its own routine running in parallel."

This doc lays out the smallest architectural change that delivers that property without rewriting the engine.

---

## 1. What's already correct (don't touch)

`core/task_supervisor.py` (208 lines, v-task-supervisor-2026-04-30) already provides:
- Per-task `asyncio.create_task` with rolling-window crash counter
- Exponential backoff (`5s → 30s → 60s` NORMAL, `2s → 5s → 15s` CRITICAL)
- Circuit-break after N crashes in T sec → operator must reset
- Commentary card on every restart (so the dashboard surfaces failures)

`core/engine.py:2892-2906` registers four tasks:
- `schwab_stream` — WS tick ingestion (NORMAL)
- `quote_streamer` — polling fallback (NORMAL)
- `position_loop` — exits / FSM, CRITICAL, **per-position try/except** (`engine.py:2796`)
- `analysis_loop` — everything else, NORMAL

`position_loop` is already the model for what the rest should look like: tight cadence, per-unit try/except, never wedges siblings.

## 2. What's wrong (the actual gap)

`_analysis_loop_body` (`engine.py:3507`, ~300 lines) is a god-function with a single outer try/except. Inside, **serially**:

1. Schwab account sync (line ~3397)
2. Position-price update
3. Market-condition assessment + pre-market gap analysis
4. `should_trade` gate
5. Market-analysis-with-commentary → **screener** (line 3636 → 4036)
6. Position management with commentary
7. **Emergency-stop check** (line 3650 — yesterday's crash site)
8. Order-status check (live mode)
9. ML retrain (every 100 cycles)
10. State save + ML buffer save (every 10 cycles)
11. Performance snapshot

Any exception in any of these aborts the rest of the iteration. The exception handler at line 3796 logs `Trading loop error: …` and sleeps **60s** (longer than the normal 30s RTH cadence — crash makes things worse). Nothing in this list except (5) has a real data dependency on (4); (7) is unconditional and independent of everything else; (9) and (10) are pure side-effects.

## 3. Target topology

Replace the single `analysis_loop` with **seven** supervised tasks. Each owns one domain, has its own cadence, and a per-iteration try/except so siblings are unaffected.

| New task | Priority | Cadence (RTH) | Produces / writes | Reads |
|---|---|---|---|---|
| `screener_loop` | NORMAL | 120s | `screener.top_movers`, `dynamic_watchlist`, `price_book` watchlist scope | (none) |
| `breadth_loop` | NORMAL | 300s | VIX, advance/decline, market regime | (none) |
| `account_sync_loop` | NORMAL | 60s | `schwab_daily_pnl`, account balance, BP, schwab positions cache | Schwab REST |
| `emergency_stop_loop` | CRITICAL | 30s | `is_running` flag flip + commentary on breach | `schwab_daily_pnl` |
| `state_save_loop` | NORMAL | 300s | `trading_state.json`, `trading_brain.json`, ML buffer | engine state (read-only) |
| `ml_retrain_loop` | NORMAL | 1h | ML model files | trade history |
| `signal_loop` (residual analysis_loop) | NORMAL | 30s | `signals` → orders | positions, breadth, watchlist, top_movers, news |

**Why `emergency_stop_loop` is CRITICAL**: a 5%-daily-loss breach must trip even if `signal_loop` and `screener_loop` are circuit-broken. Today it's coupled to the screener — exactly backwards.

**Why `signal_loop` keeps internal serial order**: read positions → correlation gate → strategies → risk gate → emit is a real data dependency chain. Decomposing that further is a separate, larger refactor.

## 4. Shared state — don't over-engineer

State already lives as instance attributes (`self.positions`, `self.dynamic_watchlist`, `screener.top_movers`, `risk_manager.schwab_daily_pnl`, `price_book`). **Keep that.** Add only what's needed for safe cross-task access:

- Each producer task is the **sole writer** of its slice.
- Readers use `asyncio.Lock` only where a write is non-atomic (the dict-key updates we already do are atomic; replacing the dict whole-cloth is too — we'd need a lock for in-place multi-field updates).
- No new "AppState" class, no event bus, no Redis. This is a single-process Python bot; the simplest thing that satisfies fault-isolation is what we want.

## 5. WebSocket broadcast pattern

Today the connection_manager broadcasts state snapshots. After this refactor:

- Each producer task, on successful update, calls a slim helper `broadcast_slice(slice_name, payload)`.
- `slice_name` ∈ `{positions, watchlist, screener, breadth, account, signals, commentary}`.
- Dashboard merges per-slice into local state — no re-render of the whole tab on a screener refresh.

**Out of scope for phase 1**: frontend changes. Phase 1 keeps the existing full-state broadcast and only changes the *producers*. Frontend optimization is a separate, scoped phase 2.

## 6. Migration sequence (each step independently shippable + reversible)

Each step is one PR-sized change, behind a v-tag, with one regression test. Operator approves each before the next.

1. **`screener_loop`** (smallest, highest immediate ROI). Cut today's wedge pattern: screener kept refreshing all day if it had been its own task. Effort: ~50 lines + test. Risk: low.
2. **`emergency_stop_loop`** — moves the file's currently-crashing code into its own try/except boundary. Defense-in-depth for the next pnl_to_check-class bug. Effort: ~80 lines + test.
3. **`account_sync_loop`** — owns Schwab REST polling for account info + positions reconciliation. Effort: ~120 lines + test. Bigger because it's the highest-traffic Schwab caller.
4. **`state_save_loop`** + **`ml_retrain_loop`** — both are pure side-effects, easy to extract together. Effort: ~60 lines + test.
5. **`breadth_loop`** — currently inside `_assess_market_conditions` / `_analyze_premarket_gaps`. Effort: ~80 lines + test.
6. **Rename what's left** of `_analysis_loop` → `_signal_loop`, update task registration. Effort: ~30 lines, mostly grep-replace.

After step 6, `_analysis_loop_body` should be ~80 lines (down from ~300) and contain only the strategy pipeline.

## 7. Rollback

Each task is registered conditionally on a config flag (default `true` once shipped, but operator can disable via yaml to revert to the monolithic loop). Flag names: `parallel.screener_loop`, `parallel.emergency_stop_loop`, etc.

## 8. Definition of done — phase 1

- Six new tasks visible in `TaskSupervisor._tasks` at startup
- Each tile on the dashboard refreshes on its own cadence even if `signal_loop` is circuit-broken
- Regression test: a synthetic exception injected into `signal_loop` does not delay the next `screener_loop` tick
- `_analysis_loop_body` deleted; `_signal_loop` body ≤100 lines
- All 61 existing regression tests still green + 6 new ones (one per task)

## 9. Open questions for operator

1. **Cadence values** — table above is my best guess. Want to override anything? (Screener 120s, account sync 60s, emergency stop 30s feel like the load-bearing ones.)
2. **Frontend phase 2** — do you want it scoped now (per-slice WS broadcasts + per-tile merge) or after phase 1 lands?
3. **Strict-mode test coverage** — should every supervised task have a fault-injection test, or only the CRITICAL ones?
4. **Circuit-break behavior** — today, circuit-broken tasks require a bot restart. Should we instead expose a `/api/restart-task/{name}` endpoint so you can recover without a full restart? (Recommended yes.)

---

## Appendix: file-level changes (phase 1, step 1 only)

**New file** `core/loops/screener_loop.py` (~50 lines):
- Owns `screener.screen_stocks()` invocation + `dynamic_watchlist` update + `price_book.register_interest('watchlist', …)`
- Cadence: `Config().SCREENER_LOOP_SEC` (new, default 120)
- Per-iteration try/except, never raises out of `run()`

**Edit** `core/engine.py`:
- Remove screener block (lines 4030-4093 in current revision)
- `sup.register("screener_loop", screener_loop.run, TaskPriority.NORMAL)` next to the existing four (line 2906)
- One v-tag: `v-parallel-screener-loop-2026-05-12`

**Edit** `core/config.py`:
- Add `SCREENER_LOOP_SEC` property (default 120s)

**New test** `tests/test_recent_fixes.py::TestParallelScreenerLoop`:
- Asserts `screener_loop` is registered with the supervisor at startup
- Asserts the loop's `run()` does not raise when `screen_stocks()` raises
- Asserts `dynamic_watchlist` is updated after one successful tick
