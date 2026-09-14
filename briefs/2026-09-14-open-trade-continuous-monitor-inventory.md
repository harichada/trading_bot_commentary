# Open-Trade Continuous Monitor Inventory & Gap Analysis

**Date:** 2026-09-14  
**Context:** FTFT hard-stopped (~12:49 @~5.23 after entry ~12:28 @~5.66 continuation RSI~74 mixed) then price recovered. Hari requires: the bot must NOT just set stop/TP and idle — it must continuously monitor ALL open trades in parallel for sentiment/regime/commentary signals and take immediate action (tighten/trail/exit/add context).

---

## 1. Inventory of Existing Open-Position Continuous Monitors

### 1.1 DynamicExitManager

| Attribute | Value |
|-----------|-------|
| **File** | `trading_bot_commentary_updated.py` |
| **Lines** | 2241–2440 |
| **Key Functions** | `initialize_position_tracking()`, `evaluate_exit()` |
| **What It Does** | Evaluates exits based on price action: quick scalps (>0.5% in <5min), progressive profit taking (>1.5%, >3%), momentum failure (MACD/RSI), time-based trailing stop activation, ATR-based dynamic trailing, exhaustion detection (RSI>75), time decay exit (>30min with <0.3% P&L), patience limit |
| **Cadence/Trigger** | Called from `_manage_positions_with_commentary()` every 30 seconds |
| **Parallel Across Positions?** | **No** — sequential `for` loop over positions |
| **Sentiment/Regime/Commentary Aware?** | **No** — only checks price, RSI, MACD, ATR technical indicators |
| **Can Tighten/Trail/Exit?** | Yes — updates `current_stop`, returns `should_exit`, `reason`, `exit_portion` for partial/full exits |

### 1.2 AdvancedExitManager

| Attribute | Value |
|-----------|-------|
| **File** | `trading_bot_commentary_updated.py` |
| **Lines** | 450–545 |
| **Key Functions** | `initialize_trailing_stop()`, `update_trailing_stop()`, `should_partial_exit()`, `time_based_exit()`, `volatility_based_exit()` |
| **What It Does** | Manages trailing stops (percent-based), partial exits (RSI overbought/oversold >80/<20), time-based exits (max hold days), volatility change exits (>50% change) |
| **Cadence/Trigger** | Methods available but **not actively called** in the main loop — `DynamicExitManager` is the active manager |
| **Parallel Across Positions?** | N/A (not integrated into main loop) |
| **Sentiment/Regime/Commentary Aware?** | **No** — pure technical/time/volatility based |
| **Can Tighten/Trail/Exit?** | Yes — designed for it, but currently unused |

### 1.3 Main Trading Loop Position Management

| Attribute | Value |
|-----------|-------|
| **File** | `trading_bot_commentary_updated.py` |
| **Lines** | 6060–6155 (main loop), 6744–6870 (`_manage_positions_with_commentary`) |
| **Key Functions** | `_manage_positions_with_commentary()`, `_evaluate_exit_conditions()` |
| **What It Does** | Iterates through all positions, updates prices, checks exit conditions via `DynamicExitManager.evaluate_exit()` |
| **Cadence/Trigger** | Every 30 seconds (`await asyncio.sleep(30)`) |
| **Parallel Across Positions?** | **No** — sequential `for symbol, position in list(positions_to_check.items())` |
| **Sentiment/Regime/Commentary Aware?** | **No** — only price/technical checks |
| **Can Tighten/Trail/Exit?** | Yes — calls `_close_position_with_commentary()` |

### 1.4 OCO Bracket Orders (`_place_bracket_orders`)

| Attribute | Value |
|-----------|-------|
| **File** | `trading_bot_commentary_updated.py` |
| **Lines** | 5543–5612 |
| **Key Functions** | `_place_bracket_orders()`, `_validate_oco_prices()` |
| **What It Does** | Places stop loss and take profit as Schwab OCO order after entry fill |
| **Cadence/Trigger** | Once after order fill verification |
| **Parallel Across Positions?** | N/A (one-time placement) |
| **Sentiment/Regime/Commentary Aware?** | **No** — fire-and-forget static brackets |
| **Can Tighten/Trail/Exit?** | **No** — no dynamic adjustment after placement |

### 1.5 Order Status Monitor (`_check_order_status`)

| Attribute | Value |
|-----------|-------|
| **File** | `trading_bot_commentary_updated.py` |
| **Lines** | 5714–5798 |
| **Key Functions** | `_check_order_status()`, `_check_specific_order_status()`, `_verify_order_fill()` |
| **What It Does** | Checks pending order fills, handles FILLED/REJECTED/CANCELED status |
| **Cadence/Trigger** | Every loop iteration in LIVE mode |
| **Parallel Across Positions?** | **No** — sequential iteration over `pending_orders` |
| **Sentiment/Regime/Commentary Aware?** | **No** — pure order status checking |
| **Can Tighten/Trail/Exit?** | **No** — order fill verification only |

### 1.6 MarketRegimeDetector

| Attribute | Value |
|-----------|-------|
| **File** | `trading_bot_commentary_updated.py` |
| **Lines** | 3189–3217 |
| **Key Functions** | `detect_regime()` |
| **What It Does** | Detects market regime (high_volatility, low_volatility, trending, normal) based on VIX and trend strength |
| **Cadence/Trigger** | Called during ML model operations |
| **Parallel Across Positions?** | N/A (market-level, not position-level) |
| **Sentiment/Regime/Commentary Aware?** | Yes — detects regime |
| **Can Tighten/Trail/Exit?** | **No** — only used for ML retraining intervals and position sizing notes, **NOT** for exit decisions |

### 1.7 FreeNewsSignalStrategy

| Attribute | Value |
|-----------|-------|
| **File** | `trading_bot_commentary_updated.py` |
| **Lines** | 7738–7838 |
| **Key Functions** | `generate_signal_with_commentary()` |
| **What It Does** | Fetches news from free sources (RSS, Google News), analyzes sentiment with VADER, generates entry signals |
| **Cadence/Trigger** | Called during market analysis for new entries |
| **Parallel Across Positions?** | N/A (entry signal generation, not position monitoring) |
| **Sentiment/Regime/Commentary Aware?** | Yes — for entries |
| **Can Tighten/Trail/Exit?** | **No** — **does NOT monitor open positions for sentiment changes** |

### 1.8 Close Confirmation System

| Attribute | Value |
|-----------|-------|
| **File** | `trading_bot_commentary_updated.py` |
| **Lines** | 6874–7094 |
| **Key Functions** | `_close_position_with_commentary()`, `_get_close_confirmation()` |
| **What It Does** | Requests user confirmation before closing positions in LIVE mode (configurable thresholds) |
| **Cadence/Trigger** | Triggered on exit decision |
| **Parallel Across Positions?** | N/A (per-position at exit time) |
| **Sentiment/Regime/Commentary Aware?** | **No** — just confirmation workflow |
| **Can Tighten/Trail/Exit?** | Yes — gates the exit action |

---

## 2. Gap Analysis: FTFT-Class Trades

### 2.1 What Happened on FTFT (2026-09-14)

| Time | Event | Price | Notes |
|------|-------|-------|-------|
| ~12:28 | Entry | ~$5.66 | Continuation setup, RSI ~74 (mixed/overbought territory) |
| Post-entry | OCO Placed | - | Stop loss + take profit brackets set |
| ~12:49 | Hard Stop Hit | ~$5.23 | Price dropped, stop triggered |
| Post-stop | Recovery | >$5.23 | Price recovered after stop hit |

### 2.2 What Signals Existed But Did NOT Drive Action

| Signal Source | Signal Present? | Bot Awareness? | Action Taken? |
|---------------|-----------------|----------------|---------------|
| RSI ~74 at entry (overbought territory) | Yes | Partial (DynamicExitManager checks RSI>75 for exhaustion) | **No** — threshold is 75, was 74 |
| Momentum fade (MACD cross) | Unknown | Yes (DynamicExitManager checks) | Depends on MACD state |
| Regime change / volatility spike | Possibly | **No** — MarketRegimeDetector not connected to exit logic | **No** |
| Negative news/sentiment shift | Unknown | **No** — FreeNewsSignalStrategy only checks for entries | **No** |
| Commentary/context signals | Unknown | **No** — commentary is output-only, not decision input | **No** |

### 2.3 What Would Have Happened With Today's Stack

1. **Entry at ~12:28**: Bot places OCO brackets (stop + TP)
2. **OCO brackets are fire-and-forget**: No dynamic adjustment based on changing conditions
3. **30-second loop runs**: Only checks price vs static stop/TP, technical indicators
4. **RSI at 74**: Below exhaustion threshold (75), no action
5. **No sentiment monitoring**: Even if negative news emerged, bot wouldn't see it for open positions
6. **No regime monitoring**: Even if volatility spiked, no automatic tightening
7. **Stop hit at ~12:49**: Static OCO stop triggered by Schwab
8. **Recovery**: Bot has no position, cannot benefit from recovery

### 2.4 Key Gaps

| Gap | Impact |
|-----|--------|
| **No parallel position monitoring** | Positions checked sequentially every 30s; slow reaction |
| **No sentiment monitoring for open positions** | News/sentiment only used for entries, not exits/management |
| **No regime-driven exit logic** | Regime detection exists but not connected to position management |
| **Static OCO brackets** | No dynamic stop tightening based on changing conditions |
| **No early warning / proactive tightening** | Bot waits for stop hit instead of proactively protecting profits |
| **Commentary is output-only** | Rich commentary generated but not used as decision input |

---

## 3. Modular P0 Proposal: Smallest Actionable PR Slices

### 3.1 PR Slice #1: Shadow Open-Trade Sentiment/Exit Desk (Log-Only)

**Boundary:** New `OpenTradeMonitorDesk` class that runs parallel to existing loop, monitors all open positions for sentiment/regime/technical signals, and logs would-be actions without executing them.

**Key Files to Add/Modify:**
- New: `analysis/open_trade_desk.py` (or inline class)
- Modify: `trading_bot_commentary_updated.py` — integrate desk into main loop

**Config Flags:**
```python
'open_trade_desk': {
    'enabled': False,  # Safe off-path default
    'shadow_mode': True,  # Log-only, no actions
    'check_interval_seconds': 5,  # Faster than 30s main loop
    'sentiment_enabled': True,
    'regime_enabled': True,
    'technical_enabled': True,
}
```

**What It Enables:**
- Parallel `asyncio.gather()` monitoring of all open positions
- Per-position sentiment fetch (using existing `FreeNewsAggregator`)
- Per-position regime check (using existing `MarketRegimeDetector`)
- Shadow log: "WOULD_TIGHTEN: FTFT stop from $5.20 to $5.40 due to negative sentiment shift"
- Shadow log: "WOULD_EXIT: FTFT due to regime change to high_volatility"
- No actual order modifications

**HANDS_OFF Respect:**
```python
HANDS_OFF_SYMBOLS = {'MU', 'HQGE', 'SPCX'}
if symbol in HANDS_OFF_SYMBOLS:
    logger.info(f"OpenTradeDesk: Skipping {symbol} (HANDS_OFF)")
    continue
```

**Safe Off-Path:**
- `enabled: False` by default
- `shadow_mode: True` — even if enabled, only logs
- Does not touch `DAY_TRADE_LIVE` or execute any orders

---

### 3.2 PR Slice #2: Action-Capable Desk (Tighten/Trail/Exit Behind Second Flag)

**Boundary:** Extends PR #1 with actual order modification capability, gated behind additional flag.

**Config Flags (extends PR #1):**
```python
'open_trade_desk': {
    'enabled': True,
    'shadow_mode': False,  # <-- New: allows actions
    'action_mode': 'tighten_only',  # Options: 'tighten_only', 'trail', 'full_exit'
    'tighten_threshold_percent': 1.0,  # Min profit before tightening allowed
    'max_tighten_per_check': 0.5,  # Max % to tighten per check
    'require_confirmation': True,  # Use existing confirmation system
}
```

**What It Enables:**
- Actual OCO modification via Schwab API (cancel old, place new tighter)
- Sentiment-driven tightening: "News turned negative, tightening stop from $5.20 to $5.35"
- Regime-driven tightening: "Volatility spiking, moving stop to breakeven"
- Still respects HANDS_OFF list
- Still respects existing confirmation workflow for exits

**Depends On:** PR #1 merged and validated in shadow mode

---

### 3.3 PR Slice #3: Async Parallel Position Monitor Infrastructure

**Boundary:** Refactors `_manage_positions_with_commentary()` to use `asyncio.gather()` for true parallel checking.

**What It Changes:**
```python
# BEFORE (sequential)
for symbol, position in list(positions_to_check.items()):
    await self._check_single_position(position)

# AFTER (parallel)
tasks = [
    self._check_single_position(position) 
    for position in positions_to_check.values()
    if position.symbol not in HANDS_OFF_SYMBOLS
]
await asyncio.gather(*tasks, return_exceptions=True)
```

**Config Flag:**
```python
'position_management': {
    'parallel_check_enabled': False,  # Safe off-path
    'max_concurrent_checks': 5,  # Limit for API rate limiting
}
```

**What It Enables:**
- All positions checked simultaneously instead of sequentially
- Faster reaction time (not waiting for N*30s worst case)
- Better for high-position-count scenarios

**Depends On:** Can be done independently or after PR #1

---

## 4. Recommended First PR

**Recommendation: Start with PR Slice #1 (Shadow Open-Trade Sentiment/Exit Desk)**

**Rationale:**
1. **Zero production risk** — shadow/log-only mode
2. **Validates the data pipeline** — confirms sentiment/regime signals are available and meaningful for open positions
3. **Provides evidence** — logs show exactly what actions would have been taken on FTFT
4. **Modular** — clearly bounded new class, minimal changes to existing code
5. **Reviewable** — Hari can review shadow logs and decide if action logic is sound before enabling

**Estimated Scope:**
- New `OpenTradeMonitorDesk` class (~200-300 lines)
- Config additions (~20 lines)
- Integration into main loop (~30 lines)
- HANDS_OFF enforcement (~10 lines)

**Not In Scope (Phase C blocked):**
- No extraction of exit logic into separate module
- No architectural refactor of main trading engine
- No changes to live trading flow

---

## 5. Constraints Checklist

| Constraint | Addressed? |
|------------|------------|
| Modular: boundary + config flag + safe off-path | ✅ Each PR has clear boundary, flag, and off-by-default |
| No LIVE day-trade flag changes | ✅ Does not touch `DAY_TRADE_LIVE` |
| Phase C extract blocked | ✅ No extraction refactoring proposed |
| Prefer inventory + plan PR only | ✅ This brief is the deliverable |
| HANDS_OFF respect (MU/HQGE/SPCX) | ✅ Explicit skip logic in all proposals |
