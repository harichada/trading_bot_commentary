# Changelog

Notable changes to the trading bot. Each entry lists the commit, the user-facing impact, and the v-tag (greppable code anchor) where applicable.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) with v-tag anchors added.

---

## 2026-06-08 / 2026-06-09 — Close-path hardening + market-context awareness + SHORT filter verdict

A single trading session's worth of work, triggered by the **2026-06-08 11:12 ET manual close incident** (Schwab rejected 3 closes for `oversold position` because the bot's close path did not cancel the OCO bracket first, costing the operator a 6-cancel Schwab dance to recover $533 of profit).

**Result**: 4 architectural fixes shipped, MarketContext awareness added across the strategy + risk layers, SHORT capability empirically verified as not-currently-viable. Bot deployed and verified running on the new code.

### Added

- **MarketContext service** (`core/market_context.py`, commits `d2728fd` + `d4399fb`)
  - Pure-function `read_market_context(symbol)` returning regime (risk_on / risk_off / mixed), sector strength, VIX state, time-of-day, and a 0.5..1.5 conviction multiplier.
  - 11 SPDR sector ETFs added to `core/market_indices.py` cache (XLK / XLF / XLE / XLV / XLY / XLP / XLI / XLB / XLU / XLRE / XLC) with `get_quote(symbol)` accessor.
  - Symbol→sector mapping for ~150 names.
  - Wired into mean-reversion, breakout, and news strategies as a gate (`allows_long` / `allows_short`).
  - Wired into `risk/manager.py` as a sizing multiplier (clamped 0.4..1.6, applied after strategy_mult before live_mult).
  - Gated by `Config.ENABLE_MARKET_CONTEXT_GATE` (default True) and `Config.ENABLE_MARKET_CONTEXT_SIZING` (default True).
  - 37 unit tests in `tests/core/test_market_context.py`.

- **Bot-only P&L circuit** (commit `df7ac9b`, v-tag `v-bot-only-pnl-circuit-2026-06-08`)
  - Daily-loss circuit now uses bot-managed P&L instead of account-wide Schwab P&L.
  - `RiskManager.bot_daily_pnl` field, synced alongside `schwab_daily_pnl` on the same 5-cycle cadence.
  - `_compute_bot_daily_pnl()` in `core/engine.py` — pure local-state read of today's realized bot trades + open bot positions' unrealized P&L.
  - `check_trading_allowed()` reads bot_daily_pnl when `Config.ENABLE_BOT_ONLY_PNL_CIRCUIT=True` (default), falls back to schwab_daily_pnl when False.
  - Error message shows BOTH numbers for transparency: `"Bot daily loss limit exceeded (P&L: $-X, schwab_pnl: $-Y)"`.
  - **Fixes**: external holdings (HQGE/PINS/COIN) gap-downs no longer pause the bot for losses that aren't its responsibility.

- **Rising-peak filter for mean-rev SHORT** (commits `df7ac9b` + `7112d8e`, v-tag `v-rising-peak-filter-2026-06-08`)
  - Mean-rev SHORT branch in `strategies/builtin.py` now requires `close < SMA50` before firing (initially also required MACD < signal, loosened in 7112d8e after backtest evidence — see Verified below).
  - Skip with audit reason `rising_peak_uptrend` when blocked, `rising_peak_no_data` when SMA50 unavailable.
  - `Config.ENABLE_RISING_PEAK_FILTER` (default True).
  - **Does NOT enable SHORT** — `ENABLE_MEAN_REV_SHORT` stays False. Filter is structural precondition only.

- **Deploy runbook** (`docs/DEPLOY_2026-06-08_EVENING.md`) — line-by-line checklist for shipping the 4 fixes after market close.

- **SHORT strategy backtest script** (`research/backtest_short_filter_2026-06-08.py`) — three-run comparison (filter ON / filter OFF baseline / SHORT off production) with automated decision gate (PF ≥ 1.3, n ≥ 5, max_simultaneous_shorts < 4).

- **CHANGELOG.md** — this file.

### Changed

- **Close path cancels OCO before sell** (commit `df7ac9b`, v-tag `v-cancel-bracket-before-close-2026-06-08`)
  - `_close_real_position` in `core/engine.py` now calls `_cancel_existing_orders(position.symbol)` before placing the close order (when `exit_portion >= 1.0`).
  - Reuses the existing helper that the entry path uses at line 5241.
  - **Fixes**: manual close API (`/api/request-close-position`) now works first-try instead of being rejected by Schwab with "oversold position" because OCO legs still pledge the shares.

- **Close path verifies fill before state cleanup** (commit `df7ac9b`, v-tag `v-verify-close-fill-2026-06-08`)
  - `_close_real_position` awaits `_verify_order_fill(order_id, position)` before returning True.
  - Schwab's HTTP 201 ("accepted for validation") no longer treated as filled — only the polled `FILLED` status counts.
  - On REJECTED / CANCELED / timeout → returns False, caller promotes position to ZOMBIE state.
  - **Fixes**: state corruption when Schwab asynchronously rejects an accepted-but-not-filled close order. Previously the position was popped from local state on the 201; now it's only removed on confirmed fill.

### Verified (empirical, not code)

- **mean-rev SHORT capability is not currently viable** (backtest evidence, commit `7112d8e`)
  - 60-day backtest: 1,176 SHORT candidates, **0 pass the rising-peak filter** (PF on unfiltered: 1.04, max DD −52%, max simultaneous shorts 20).
  - 360-day backtest: 4,965 SHORT candidates, **0 pass the rising-peak filter** (PF on unfiltered: 1.05, max DD −167%, max simultaneous shorts 19).
  - Geometric diagnosis: mean-rev SHORT entry requires close > SMA20+2σ; rising-peak filter requires close < SMA50. Joint condition (SMA50 > SMA20+2σ) is geometrically rare.
  - **Verdict**: the strategy and the filter target near-disjoint setups. Even unfiltered, PF is marginal with catastrophic drawdowns.
  - **Action**: `ENABLE_MEAN_REV_SHORT` stays False. Filter code remains for any future SHORT strategy variant. The capability stays low-priority.

### Tests

- **22 new regression tests** in `tests/test_recent_fixes.py` across 4 classes:
  - `TestCancelBracketBeforeClose` (3 tests)
  - `TestVerifyCloseFill` (5 tests)
  - `TestBotOnlyPnLCircuit` (7 tests)
  - `TestRisingPeakFilter` (7 tests)
- 37 unit tests in `tests/core/test_market_context.py` (commits `d2728fd` + `d4399fb`).
- Style: static source-marker assertions (consistent with the file's prior pattern); no Schwab/PriceBook mocking.

### Operational notes

- **All four `df7ac9b` fixes activate on bot restart.** The bot was restarted at 16:03 ET 2026-06-08 after market close; new PID 624375 verified running on the patched code.
- **MarketContext sizing was active during the morning session** (commits landed at 09:10 + 09:24 ET); conviction multipliers visible in the 5 trades placed 09:41–09:47 ET. Live evidence: NOK 0.95×, GLW 1.25×, ARM 1.25×, OWL 1.10×, ONDS 1.25×.
- **Day P&L outcome**: +$666.21 realized across 5 closed bot trades (NOK +$124, GLW +$76, ARM +$263, ONDS +$70, OWL +$132). All winners. Account-wide day P&L +$1,007 (includes externals).
- **Pre-existing test failures** (8 in test_recent_fixes.py — DirectionGate defaults, NewsVerifier, MaxPositions, etc.) are unrelated to this session and have been failing since before today's work started.

### Next priorities (informed by this session's data)

1. **Breakout strategy retune** — backtests blocked it from re-enable in May; today's risk-on tape (Nasdaq +2.38%) would have been textbook breakout territory.
2. **Live size multiplier 0.25 → 0.5** — after 2-3 more profitable sessions. Today's $666 at quarter size = ~$2,100 at full size.
3. **Bot-only P&L circuit live observation** — tomorrow (2026-06-09) is the first day with `ENABLE_BOT_ONLY_PNL_CIRCUIT=True` in production. Watch for divergence between `Bot-only P&L Check` log line and `Schwab P&L Check`.
4. **SHORT capability** — stays low-priority. Would require a *different strategy* (e.g., breakdown-momentum SHORT) rather than filtering mean-rev SHORT.

---

## How to read this file

Each release entry follows the same structure:

- **Added** — new features, files, configs
- **Changed** — modifications to existing behavior
- **Fixed** — bug fixes (called out separately when distinct from changed behavior)
- **Removed** — deleted features
- **Verified** — empirical outcomes from backtests / live observation (not code changes, but evidence that informs strategy decisions)
- **Tests** — regression coverage added
- **Operational notes** — deploy timing, live evidence, gotchas
- **Next priorities** — what the data points to next

v-tags (e.g., `v-cancel-bracket-before-close-2026-06-08`) are greppable anchors in the source. They live inline as `# v-…` comments at the patch site; future regressions can be diagnosed by grepping the codebase for the tag.
