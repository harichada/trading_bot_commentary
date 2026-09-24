"""v-ledger-integrity-2026-09-24 (PR1): Trade ledger integrity module.

Provides comprehensive trade ledger integrity for accurate R-multiple computation:
  1. Broker-orphan reconciliation: fills that happened at the broker but aren't
     in bot_trades (broker stops/TPs fired without bot knowledge)
  2. TRUE initial stop/TP/risk tracking: immutable fields set at entry
  3. Accurate R computation: uses initial_stop, fallback to 1.5*ATR from reasoning
  4. Hands-off/external exclusion: MU/HQGE/SPCX never counted in bot stats

Trade Sources:
  - 'bot': trade opened and closed by the bot
  - 'broker_orphan': broker fill discovered during reconciliation (stop/TP hit
    externally, or manual close at broker)
  - 'external': external/unmanaged position closed

Exit Reasons (standardized):
  - 'take_profit': hit TP target
  - 'stop_loss': hit stop loss
  - 'proactive_macd': MACD flipped bearish (early exit)
  - 'proactive_rsi': RSI below 50 (early exit)
  - 'open_desk': Active Open Desk RSI extreme exit
  - 'flatten_hour': day-trade flatten at market close
  - 'external_reconcile': broker fill detected during sync
  - 'time_stop': thesis time-out
  - 'emergency_stop': circuit breaker / emergency exit

Setup Types:
  - 'breakout': 20-bar high breakout
  - 'pullback': pullback to VWAP/trend continuation
  - 'continuation': trend continuation entry
  - 'mean_reversion': oversold bounce / mean-rev
  - 'orb': opening range breakout
  - 'unknown': pattern not recorded

Config flag: LEDGER_INTEGRITY (default True)
Safe off-path: set LEDGER_INTEGRITY=0 to preserve pre-PR1 behavior.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("TradingBot")

# Standard exit reason vocabulary
EXIT_REASONS = frozenset({
    "take_profit",
    "stop_loss", 
    "proactive_macd",
    "proactive_rsi",
    "open_desk",
    "flatten_hour",
    "external_reconcile",
    "time_stop",
    "emergency_stop",
    "external_close",
    "broker_stop",
    "broker_tp",
    "manual_override",
})

# Setup type vocabulary
SETUP_TYPES = frozenset({
    "breakout",
    "pullback",
    "continuation",
    "mean_reversion",
    "orb",
    "unknown",
})

# Trade source vocabulary
TRADE_SOURCES = frozenset({
    "bot",
    "broker_orphan",
    "external",
})


@dataclass
class TradeLedgerEntry:
    """A standardized trade ledger entry with full R-tracking.
    
    This is the canonical representation for closed trades that
    enables accurate statistics and R-multiple computation.
    """
    symbol: str
    side: str
    strategy: Optional[str]
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    pnl_r: Optional[float]  # P&L in R-multiples
    exit_reason: str
    setup_type: str
    source: str  # 'bot', 'broker_orphan', 'external'
    hold_time_seconds: int
    initial_stop: Optional[float]
    initial_tp: Optional[float]
    initial_risk_per_share: Optional[float]
    atr_at_entry: Optional[float]
    stop_loss: Optional[float]  # Final stop (may differ from initial)
    take_profit: Optional[float]
    confidence: Optional[float]
    meta_proba: Optional[float]
    kelly_fraction: Optional[float]
    scaled_out: bool
    mode: str
    is_hands_off: bool  # MU/HQGE/SPCX
    is_external: bool   # External/unmanaged position
    reasoning: Dict[str, Any]

    def to_db_params(self) -> Dict[str, Any]:
        """Convert to parameters for db_logger.log_trade_v2."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "strategy": self.strategy,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
            "pnl_r": self.pnl_r,
            "exit_reason": self.exit_reason,
            "setup_type": self.setup_type,
            "source": self.source,
            "hold_time_seconds": self.hold_time_seconds,
            "initial_stop": self.initial_stop,
            "initial_tp": self.initial_tp,
            "initial_risk_per_share": self.initial_risk_per_share,
            "atr_at_entry": self.atr_at_entry,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "confidence": self.confidence,
            "meta_proba": self.meta_proba,
            "kelly_fraction": self.kelly_fraction,
            "scaled_out": self.scaled_out,
            "mode": self.mode,
            "is_hands_off": self.is_hands_off,
            "is_external": self.is_external,
            "reasoning": self.reasoning,
        }


def compute_r_multiple(
    side: str,
    entry_price: float,
    exit_price: float,
    initial_stop: Optional[float] = None,
    stop_loss: Optional[float] = None,
    atr: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute R-multiple and initial risk per share.
    
    R = pnl_per_share / risk_per_share
    
    Risk is determined by (in priority order):
      1. initial_stop if set (immutable stop at entry)
      2. stop_loss if different from entry_price
      3. 1.5 * ATR fallback if ATR is available
      4. None (R undefined)
    
    Args:
        side: 'long' or 'short'
        entry_price: Entry price
        exit_price: Exit price
        initial_stop: Immutable stop set at entry (preferred)
        stop_loss: Current/final stop loss (may have been modified)
        atr: ATR at entry for 1.5*ATR fallback
    
    Returns:
        (pnl_r, initial_risk_per_share) tuple. Both may be None if risk undefined.
    """
    if entry_price <= 0:
        return None, None
    
    # Determine risk per share
    risk_per_share = None
    
    # Priority 1: Use initial_stop if set
    if initial_stop is not None and initial_stop > 0:
        if side == "long":
            risk_per_share = entry_price - initial_stop
        else:
            risk_per_share = initial_stop - entry_price
    
    # Priority 2: Use stop_loss if it's meaningful (not equal to entry)
    if risk_per_share is None or risk_per_share <= 0:
        if stop_loss is not None and stop_loss > 0:
            if side == "long":
                candidate = entry_price - stop_loss
            else:
                candidate = stop_loss - entry_price
            if candidate > 0.001:  # Must be > 0.1 cents
                risk_per_share = candidate
    
    # Priority 3: 1.5*ATR fallback
    if risk_per_share is None or risk_per_share <= 0:
        if atr is not None and atr > 0:
            risk_per_share = 1.5 * atr
    
    if risk_per_share is None or risk_per_share <= 0:
        return None, None
    
    # Compute PnL per share
    if side == "long":
        pnl_per_share = exit_price - entry_price
    else:
        pnl_per_share = entry_price - exit_price
    
    pnl_r = pnl_per_share / risk_per_share
    return round(pnl_r, 4), round(risk_per_share, 6)


def compute_hold_time_seconds(entry_time: datetime, exit_time: datetime) -> int:
    """Compute hold time in seconds between entry and exit."""
    if entry_time is None or exit_time is None:
        return 0
    try:
        delta = exit_time - entry_time
        return max(0, int(delta.total_seconds()))
    except Exception:
        return 0


def infer_setup_type(reasoning: Optional[Dict[str, Any]]) -> str:
    """Infer setup type from position reasoning.
    
    Checks reasoning fields: entry_pattern, setup_type, pattern, strategy
    """
    if not reasoning or not isinstance(reasoning, dict):
        return "unknown"
    
    # Direct setup_type field
    if "setup_type" in reasoning:
        st = str(reasoning["setup_type"]).lower()
        if st in SETUP_TYPES:
            return st
    
    # entry_pattern field
    entry_pattern = reasoning.get("entry_pattern", "")
    if isinstance(entry_pattern, str):
        entry_pattern = entry_pattern.lower()
        if "breakout" in entry_pattern:
            return "breakout"
        if "pullback" in entry_pattern:
            return "pullback"
        if "continuation" in entry_pattern:
            return "continuation"
        if "orb" in entry_pattern or "opening_range" in entry_pattern:
            return "orb"
        if "mean_rev" in entry_pattern or "oversold" in entry_pattern:
            return "mean_reversion"
    
    # pattern field
    pattern = reasoning.get("pattern", "")
    if isinstance(pattern, str):
        pattern = pattern.lower()
        if "breakout" in pattern:
            return "breakout"
        if "pullback" in pattern:
            return "pullback"
        if "continuation" in pattern:
            return "continuation"
    
    # strategy field as last resort
    strategy = reasoning.get("strategy", "")
    if isinstance(strategy, str):
        strategy = strategy.lower()
        if "mean_rev" in strategy or "oversold" in strategy:
            return "mean_reversion"
        if "orb" in strategy:
            return "orb"
        if "momentum" in strategy:
            return "continuation"
    
    return "unknown"


def normalize_exit_reason(raw_reason: Optional[str]) -> str:
    """Normalize exit reason to standard vocabulary.
    
    Maps various exit reason strings to canonical forms.
    """
    if not raw_reason:
        return "unknown"
    
    reason = str(raw_reason).lower().strip()
    
    # Direct matches
    if reason in EXIT_REASONS:
        return reason
    
    # Mappings
    if "take_profit" in reason or "tp" in reason or "target" in reason:
        return "take_profit"
    if "stop" in reason and "broker" in reason:
        return "broker_stop"
    if "stop" in reason or "sl" in reason:
        return "stop_loss"
    if "macd" in reason:
        return "proactive_macd"
    if "rsi" in reason and ("proactive" in reason or "below" in reason):
        return "proactive_rsi"
    if "desk" in reason or "open_desk" in reason:
        return "open_desk"
    if "flatten" in reason or "day_trade_flatten" in reason:
        return "flatten_hour"
    if "external" in reason:
        return "external_close"
    if "time" in reason and ("decay" in reason or "stop" in reason):
        return "time_stop"
    if "emergency" in reason or "circuit" in reason:
        return "emergency_stop"
    if "manual" in reason or "override" in reason:
        return "manual_override"
    if "reconcile" in reason:
        return "external_reconcile"
    
    return reason


def extract_initial_values_from_position(
    position: Any,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract initial_stop, initial_tp, and initial_risk from position.
    
    Checks position attributes and reasoning dict for initial values.
    Falls back to current stop_loss/take_profit if initial values not set.
    
    Returns:
        (initial_stop, initial_tp, initial_risk_per_share)
    """
    initial_stop = None
    initial_tp = None
    initial_risk = None
    
    # Try position attributes first (v-ledger-integrity fields)
    if hasattr(position, "initial_stop"):
        initial_stop = getattr(position, "initial_stop", None)
    if hasattr(position, "initial_tp"):
        initial_tp = getattr(position, "initial_tp", None)
    if hasattr(position, "initial_risk_per_share"):
        initial_risk = getattr(position, "initial_risk_per_share", None)
    
    # Fallback: check original_stop (older field)
    if initial_stop is None and hasattr(position, "original_stop"):
        initial_stop = getattr(position, "original_stop", None)
    
    # Fallback: check reasoning dict
    reasoning = getattr(position, "reasoning", {}) or {}
    if isinstance(reasoning, dict):
        if initial_stop is None:
            # Check initial_stop first (absolute price), then stop_distance (relative)
            initial_stop_val = reasoning.get("initial_stop")
            stop_distance_val = reasoning.get("stop_distance")
            
            if initial_stop_val and isinstance(initial_stop_val, (int, float)):
                # initial_stop is already an absolute price
                initial_stop = initial_stop_val
            elif stop_distance_val and isinstance(stop_distance_val, (int, float)):
                # stop_distance is relative to entry - convert to absolute
                entry = getattr(position, "entry_price", 0) or 0
                side = getattr(position, "side", "long")
                if entry > 0:
                    if side == "long":
                        initial_stop = entry - stop_distance_val
                    else:
                        initial_stop = entry + stop_distance_val
        
        if initial_tp is None:
            initial_tp = reasoning.get("initial_tp") or reasoning.get("take_profit")
    
    # Final fallback: use current values if initial not set
    if initial_stop is None and hasattr(position, "stop_loss"):
        stop_loss = getattr(position, "stop_loss", None)
        entry = getattr(position, "entry_price", 0) or 0
        if stop_loss and stop_loss != entry:  # Avoid stop==entry bug
            initial_stop = stop_loss
    
    if initial_tp is None and hasattr(position, "take_profit"):
        initial_tp = getattr(position, "take_profit", None)
    
    # Compute initial risk if we have initial_stop
    if initial_risk is None and initial_stop is not None:
        entry = getattr(position, "entry_price", 0) or 0
        side = getattr(position, "side", "long")
        if entry > 0 and initial_stop > 0:
            if side == "long":
                initial_risk = entry - initial_stop
            else:
                initial_risk = initial_stop - entry
            if initial_risk <= 0:
                initial_risk = None
    
    return initial_stop, initial_tp, initial_risk


def build_trade_ledger_entry(
    position: Any,
    exit_price: float,
    exit_time: datetime,
    exit_reason: str,
    pnl: float,
    pnl_pct: float,
    mode: str = "live",
    source: str = "bot",
    hands_off_denylist: Optional[frozenset] = None,
) -> TradeLedgerEntry:
    """Build a complete TradeLedgerEntry from a closed position.
    
    This is the main entry point for creating a standardized ledger entry
    that includes all R-tracking and metadata fields.
    
    Args:
        position: The Position object being closed
        exit_price: Fill price at exit
        exit_time: Timestamp of exit
        exit_reason: Raw exit reason string
        pnl: Realized P&L in dollars
        pnl_pct: Realized P&L percentage
        mode: 'live' or 'simulation'
        source: 'bot', 'broker_orphan', or 'external'
        hands_off_denylist: Set of hands-off symbols (MU, HQGE, SPCX)
    
    Returns:
        TradeLedgerEntry with all fields populated
    """
    if hands_off_denylist is None:
        hands_off_denylist = frozenset({"MU", "HQGE", "SPCX"})
    
    symbol = getattr(position, "symbol", "")
    side = getattr(position, "side", "long")
    entry_price = getattr(position, "entry_price", 0) or 0
    quantity = getattr(position, "quantity", 0) or 0
    entry_time = getattr(position, "entry_time", None)
    reasoning = getattr(position, "reasoning", {}) or {}
    
    # Extract initial values
    initial_stop, initial_tp, initial_risk = extract_initial_values_from_position(position)
    
    # Get ATR from reasoning for R fallback
    atr = None
    if isinstance(reasoning, dict):
        atr = reasoning.get("atr")
    
    # Compute R-multiple
    pnl_r, computed_risk = compute_r_multiple(
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        initial_stop=initial_stop,
        stop_loss=getattr(position, "stop_loss", None),
        atr=atr,
    )
    
    # Use computed risk if initial_risk not set
    if initial_risk is None:
        initial_risk = computed_risk
    
    # Hold time
    hold_time = compute_hold_time_seconds(entry_time, exit_time)
    
    # Setup type
    setup_type = infer_setup_type(reasoning)
    
    # Normalize exit reason
    normalized_reason = normalize_exit_reason(exit_reason)
    
    # Strategy
    strategy = reasoning.get("strategy") if isinstance(reasoning, dict) else None
    
    # Hands-off and external flags
    is_hands_off = symbol.upper() in hands_off_denylist
    is_external = (
        reasoning.get("source") == "external"
        or reasoning.get("is_external", False)
        or not getattr(position, "managed_by_bot", True)
    ) if isinstance(reasoning, dict) else not getattr(position, "managed_by_bot", True)
    
    return TradeLedgerEntry(
        symbol=symbol,
        side=side,
        strategy=strategy,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        pnl=pnl,
        pnl_pct=pnl_pct,
        pnl_r=pnl_r,
        exit_reason=normalized_reason,
        setup_type=setup_type,
        source=source,
        hold_time_seconds=hold_time,
        initial_stop=initial_stop,
        initial_tp=initial_tp,
        initial_risk_per_share=initial_risk,
        atr_at_entry=atr,
        stop_loss=getattr(position, "stop_loss", None),
        take_profit=getattr(position, "take_profit", None),
        confidence=getattr(position, "confidence", None) or (
            reasoning.get("confidence") if isinstance(reasoning, dict) else None
        ),
        meta_proba=(
            reasoning.get("meta_proba") if isinstance(reasoning, dict) else None
        ),
        kelly_fraction=(
            reasoning.get("kelly_fraction") if isinstance(reasoning, dict) else None
        ),
        scaled_out=getattr(position, "scaled_out", False),
        mode=mode,
        is_hands_off=is_hands_off,
        is_external=is_external,
        reasoning=reasoning if isinstance(reasoning, dict) else {},
    )


def build_broker_orphan_entry(
    symbol: str,
    side: str,
    entry_price: float,
    exit_price: float,
    quantity: int,
    entry_time: datetime,
    exit_time: datetime,
    exit_reason: str = "external_reconcile",
    hands_off_denylist: Optional[frozenset] = None,
    strategy: Optional[str] = None,
    atr: Optional[float] = None,
) -> TradeLedgerEntry:
    """Build a TradeLedgerEntry for a broker orphan fill.
    
    Broker orphans are fills that happened at the broker (stop/TP hit,
    manual close) but weren't recorded through the normal bot exit path.
    These are discovered during position reconciliation.
    
    Args:
        symbol: Stock symbol
        side: 'long' or 'short'
        entry_price: Entry price (from prior position or estimate)
        exit_price: Fill price from broker
        quantity: Shares filled
        entry_time: Entry timestamp (from prior position or estimate)
        exit_time: Fill timestamp from broker
        exit_reason: Why the fill happened (default 'external_reconcile')
        hands_off_denylist: Set of hands-off symbols
        strategy: Strategy that opened the position (if known)
        atr: ATR at entry for R computation
    
    Returns:
        TradeLedgerEntry with source='broker_orphan'
    """
    if hands_off_denylist is None:
        hands_off_denylist = frozenset({"MU", "HQGE", "SPCX"})
    
    # Compute P&L
    if side == "long":
        pnl_per_share = exit_price - entry_price
    else:
        pnl_per_share = entry_price - exit_price
    pnl = pnl_per_share * quantity
    pnl_pct = (pnl_per_share / entry_price * 100) if entry_price > 0 else 0
    
    # Compute R with ATR fallback (no initial_stop available for orphans)
    pnl_r, initial_risk = compute_r_multiple(
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        initial_stop=None,
        stop_loss=None,
        atr=atr,
    )
    
    # For broker orphans where we hit what looks like a stop, assume R=-1
    # This is a convention since we don't have the exact stop level
    normalized_reason = normalize_exit_reason(exit_reason)
    if pnl_r is None and pnl < 0:
        pnl_r = -1.0  # Convention for stop hit without known stop level
    
    hold_time = compute_hold_time_seconds(entry_time, exit_time)
    is_hands_off = symbol.upper() in hands_off_denylist
    
    return TradeLedgerEntry(
        symbol=symbol,
        side=side,
        strategy=strategy,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=quantity,
        pnl=pnl,
        pnl_pct=pnl_pct,
        pnl_r=pnl_r,
        exit_reason=normalized_reason,
        setup_type="unknown",  # No reasoning for orphans
        source="broker_orphan",
        hold_time_seconds=hold_time,
        initial_stop=None,
        initial_tp=None,
        initial_risk_per_share=initial_risk,
        atr_at_entry=atr,
        stop_loss=None,
        take_profit=None,
        confidence=None,
        meta_proba=None,
        kelly_fraction=None,
        scaled_out=False,
        mode="live",  # Broker orphans are always live
        is_hands_off=is_hands_off,
        is_external=False,  # Not external, just orphaned from bot tracking
        reasoning={
            "source": "broker_orphan",
            "reconcile_reason": exit_reason,
            "atr": atr,
        },
    )


def is_bot_managed_trade(entry: TradeLedgerEntry) -> bool:
    """Check if a trade should be counted in bot statistics.
    
    Excludes:
      - Hands-off symbols (MU, HQGE, SPCX)
      - External/unmanaged positions
    
    Includes:
      - Bot trades
      - Broker orphans (these WERE bot trades, just lost tracking)
    """
    if entry.is_hands_off:
        return False
    if entry.is_external:
        return False
    return True


def compute_trade_stats(entries: List[TradeLedgerEntry]) -> Dict[str, Any]:
    """Compute aggregate statistics for a list of trades.
    
    Only includes bot-managed trades (excludes hands-off and external).
    
    Returns:
        Dict with: n, winners, losers, scratches, win_rate, exp_r, total_r,
        profit_factor, total_pnl, avg_hold_seconds, by_setup_type, by_exit_reason
    """
    # Filter to bot-managed only
    managed = [e for e in entries if is_bot_managed_trade(e)]
    
    if not managed:
        return {
            "n": 0,
            "winners": 0,
            "losers": 0,
            "scratches": 0,
            "win_rate": 0.0,
            "exp_r": 0.0,
            "total_r": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
            "avg_hold_seconds": 0,
            "by_setup_type": {},
            "by_exit_reason": {},
        }
    
    n = len(managed)
    
    # Win/loss classification (scratch if |R| <= 0.05)
    winners = sum(1 for e in managed if e.pnl_r is not None and e.pnl_r > 0.05)
    losers = sum(1 for e in managed if e.pnl_r is not None and e.pnl_r < -0.05)
    scratches = sum(1 for e in managed if e.pnl_r is not None and abs(e.pnl_r) <= 0.05)
    
    # Win rate (decisive trades only)
    decisive = winners + losers
    win_rate = (winners / decisive * 100) if decisive > 0 else 0.0
    
    # R stats
    r_values = [e.pnl_r for e in managed if e.pnl_r is not None]
    total_r = sum(r_values) if r_values else 0.0
    exp_r = (total_r / len(r_values)) if r_values else 0.0
    
    # Profit factor (sum of winning R / sum of losing R)
    win_r = sum(r for r in r_values if r > 0)
    loss_r = abs(sum(r for r in r_values if r < 0))
    profit_factor = (win_r / loss_r) if loss_r > 0 else (float("inf") if win_r > 0 else 0.0)
    
    # Dollar P&L
    total_pnl = sum(e.pnl for e in managed)
    
    # Hold time
    total_hold = sum(e.hold_time_seconds for e in managed)
    avg_hold = total_hold // n if n > 0 else 0
    
    # Breakdown by setup type
    by_setup = {}
    for e in managed:
        st = e.setup_type or "unknown"
        if st not in by_setup:
            by_setup[st] = {"n": 0, "total_r": 0.0, "pnl": 0.0}
        by_setup[st]["n"] += 1
        if e.pnl_r is not None:
            by_setup[st]["total_r"] += e.pnl_r
        by_setup[st]["pnl"] += e.pnl
    
    # Breakdown by exit reason
    by_exit = {}
    for e in managed:
        er = e.exit_reason or "unknown"
        if er not in by_exit:
            by_exit[er] = {"n": 0, "total_r": 0.0, "pnl": 0.0}
        by_exit[er]["n"] += 1
        if e.pnl_r is not None:
            by_exit[er]["total_r"] += e.pnl_r
        by_exit[er]["pnl"] += e.pnl
    
    return {
        "n": n,
        "winners": winners,
        "losers": losers,
        "scratches": scratches,
        "win_rate": round(win_rate, 1),
        "exp_r": round(exp_r, 4),
        "total_r": round(total_r, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "total_pnl": round(total_pnl, 2),
        "avg_hold_seconds": avg_hold,
        "by_setup_type": by_setup,
        "by_exit_reason": by_exit,
    }
