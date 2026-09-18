# Changelog

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
