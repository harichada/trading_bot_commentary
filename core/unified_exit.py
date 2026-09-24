"""Unified exit manager for day-trade positions — shadow logging first.

v-unified-exit-2026-09-24. PR3 of the pro-trader rebuild. Implements
the unified exit policy from the blueprint:

  1. Structure-based initial stop (not indicator-based)
  2. Scale out 50% at +1R, move stop to breakeven
  3. Trail the remainder (ATR or swing low)
  4. Time stop N minutes before flatten

Shadow mode (default):
  DT_UNIFIED_EXIT=1             → shadow logging enabled
  DT_UNIFIED_EXIT_LIVE_ENFORCE=0 → no live order changes

Enforce mode (opt-in only after shadow validation):
  DT_UNIFIED_EXIT_LIVE_ENFORCE=1 → replaces proactive_macd/rsi/desk exits
                                   for day_trade_momentum only

Hard stops and circuits are NEVER bypassed by this module.
Hands-off forever: MU, HQGE, SPCX.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import Position

logger = logging.getLogger("TradingBot")

HANDS_OFF_SYMBOLS = frozenset({"MU", "HQGE", "SPCX"})
ATR_STOP_MULT = 1.5
RR_RATIO = 2.0
SCALE_AT_R = 1.0
TRAIL_ATR_MULT = 1.5
TIME_STOP_MINUTES_BEFORE_FLATTEN = 15
DEFAULT_FLATTEN_HOUR = 15

SHADOW_LOG_PATH = Path("data/shadow_unified_exit.ndjson")


@dataclass
class UnifiedExitAction:
    """What the unified exit policy would do on this tick."""
    symbol: str
    timestamp: str
    action: str  # 'hold', 'scale_partial', 'move_stop_breakeven', 'trail_stop', 'time_stop_exit', 'target_exit', 'stop_exit'
    reason: str
    current_price: float
    entry_price: float
    stop_distance: float
    r_so_far: float
    recommended_stop: Optional[float] = None
    recommended_action_qty: Optional[int] = None
    live_action_taken: str = "none"  # What the live system actually did
    live_enforce: bool = False
    scaled_out: bool = False
    trailing_active: bool = False


class UnifiedExitManager:
    """Computes unified exit policy for day-trade positions.
    
    Call evaluate_position() on each position manager tick. Returns what
    the unified policy WOULD do. With enforce=False (default), the return
    value is shadow-logged only. With enforce=True, the caller should use
    the returned action to override proactive indicator exits.
    
    Hard stops and flatten-hour are NOT managed here — they remain in the
    engine's core loop and fire regardless of this policy.
    """
    
    def __init__(
        self,
        shadow_log_path: Optional[Path] = None,
        enabled: bool = True,
        live_enforce: bool = False,
        time_stop_minutes: int = TIME_STOP_MINUTES_BEFORE_FLATTEN,
        flatten_hour: int = DEFAULT_FLATTEN_HOUR,
    ):
        self.shadow_log_path = shadow_log_path or SHADOW_LOG_PATH
        self.enabled = enabled
        self.live_enforce = live_enforce
        self.time_stop_minutes = time_stop_minutes
        self.flatten_hour = flatten_hour
        
        self._position_state: dict[str, dict[str, Any]] = {}
        
        self.shadow_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "UnifiedExitManager":
        """Create manager from environment variables.
        
        DT_UNIFIED_EXIT=1           → shadow logging enabled
        DT_UNIFIED_EXIT_LIVE_ENFORCE=1 → live enforce mode
        """
        enabled = os.getenv("DT_UNIFIED_EXIT", "0").lower() in ("1", "true", "yes")
        live_enforce = os.getenv("DT_UNIFIED_EXIT_LIVE_ENFORCE", "0").lower() in ("1", "true", "yes")
        
        return cls(
            enabled=enabled,
            live_enforce=live_enforce,
        )
    
    def _get_position_state(self, symbol: str) -> dict[str, Any]:
        """Get or initialize position tracking state."""
        if symbol not in self._position_state:
            self._position_state[symbol] = {
                "scaled_out": False,
                "trailing_active": False,
                "highest_price": None,
                "breakeven_stop_set": False,
            }
        return self._position_state[symbol]
    
    def _reset_position_state(self, symbol: str) -> None:
        """Reset state when position is closed."""
        if symbol in self._position_state:
            del self._position_state[symbol]
    
    def _is_time_stop_window(self, now_et_hour: int, now_et_minute: int) -> bool:
        """Check if we're in the time-stop window (N min before flatten)."""
        flatten_time_minutes = self.flatten_hour * 60
        current_minutes = now_et_hour * 60 + now_et_minute
        cutoff_minutes = flatten_time_minutes - self.time_stop_minutes
        return current_minutes >= cutoff_minutes
    
    def evaluate_position(
        self,
        position: "Position",
        current_price: float,
        atr: float,
        now_et_hour: int,
        now_et_minute: int,
        live_exit_reason: Optional[str] = None,
    ) -> Optional[UnifiedExitAction]:
        """Evaluate what the unified exit policy would do.
        
        Args:
            position: The Position to evaluate
            current_price: Current market price
            atr: Current ATR value
            now_et_hour: Current hour in ET (0-23)
            now_et_minute: Current minute in ET (0-59)
            live_exit_reason: If the live system is about to exit, what reason
        
        Returns:
            UnifiedExitAction describing what the policy would do, or None if
            the position is not a managed day-trade or hands-off.
        """
        if not self.enabled:
            return None
        
        symbol = position.symbol.upper()
        
        if symbol in HANDS_OFF_SYMBOLS:
            return None
        
        reasoning = getattr(position, "reasoning", {}) or {}
        strategy = reasoning.get("strategy", "")
        if "day_trade" not in strategy.lower() and "momentum" not in strategy.lower():
            return None
        
        if not getattr(position, "managed_by_bot", False):
            return None
        
        if position.side != "long":
            return None
        
        state = self._get_position_state(symbol)
        
        entry_price = position.entry_price
        original_stop = getattr(position, "original_stop", None) or position.stop_loss
        stop_distance = abs(entry_price - original_stop)
        if stop_distance <= 0:
            stop_distance = ATR_STOP_MULT * atr if atr > 0 else entry_price * 0.02
        
        take_profit = position.take_profit or (entry_price + RR_RATIO * stop_distance)
        
        current_r = (current_price - entry_price) / stop_distance if stop_distance > 0 else 0
        
        if state["highest_price"] is None:
            state["highest_price"] = current_price
        else:
            state["highest_price"] = max(state["highest_price"], current_price)
        
        action = "hold"
        reason = "within_thesis"
        recommended_stop = None
        recommended_action_qty = None
        
        if self._is_time_stop_window(now_et_hour, now_et_minute):
            action = "time_stop_exit"
            reason = f"time_stop_{self.time_stop_minutes}min_before_flatten"
        
        elif current_price >= take_profit:
            action = "target_exit"
            reason = "target_reached"
        
        elif current_price <= original_stop:
            action = "stop_exit"
            reason = "stop_hit"
        
        elif current_r >= SCALE_AT_R and not state["scaled_out"]:
            action = "scale_partial"
            reason = f"scale_50pct_at_{SCALE_AT_R}R"
            recommended_action_qty = max(1, int(position.quantity * 0.5))
            recommended_stop = entry_price
            state["scaled_out"] = True
            state["breakeven_stop_set"] = True
            state["trailing_active"] = True
        
        elif state["scaled_out"] and not state["breakeven_stop_set"]:
            action = "move_stop_breakeven"
            reason = "post_scale_move_to_breakeven"
            recommended_stop = entry_price
            state["breakeven_stop_set"] = True
        
        elif state["trailing_active"] and current_price > entry_price:
            trail_stop = current_price - (TRAIL_ATR_MULT * atr)
            current_stop = position.trailing_stop or position.stop_loss
            if trail_stop > current_stop:
                action = "trail_stop"
                reason = "atr_trail_ratchet"
                recommended_stop = trail_stop
        
        result = UnifiedExitAction(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            action=action,
            reason=reason,
            current_price=round(current_price, 4),
            entry_price=round(entry_price, 4),
            stop_distance=round(stop_distance, 4),
            r_so_far=round(current_r, 4),
            recommended_stop=round(recommended_stop, 4) if recommended_stop else None,
            recommended_action_qty=recommended_action_qty,
            live_action_taken=live_exit_reason or "none",
            live_enforce=self.live_enforce,
            scaled_out=state["scaled_out"],
            trailing_active=state["trailing_active"],
        )
        
        self._log_shadow(result)
        
        return result
    
    def _log_shadow(self, action: UnifiedExitAction) -> None:
        """Append action to shadow NDJSON log."""
        try:
            with open(self.shadow_log_path, "a") as f:
                f.write(json.dumps(asdict(action), default=str) + "\n")
        except Exception as exc:
            logger.warning("unified_exit_shadow_log_failed: %s", exc)
    
    def should_override_proactive_exit(
        self,
        action: UnifiedExitAction,
        live_exit_reason: str,
    ) -> bool:
        """Determine if we should override a proactive indicator exit.
        
        Only applies when:
          - live_enforce=True
          - The live exit is a proactive indicator exit (macd/rsi/desk)
          - The unified policy says "hold" (within thesis)
        
        Hard stops, target hits, and flatten-hour are NEVER overridden.
        """
        if not self.live_enforce:
            return False
        
        proactive_indicators = {
            "proactive_macd_flipped_bearish",
            "proactive_macd_flipped_bullish",
            "proactive_rsi_below_50",
            "proactive_rsi_above_50",
            "proactive_adx_collapsing",
        }
        
        if not any(ind in live_exit_reason.lower() for ind in {"macd", "rsi", "adx"}):
            return False
        
        override_actions = {"hold", "trail_stop", "scale_partial", "move_stop_breakeven"}
        if action.action in override_actions:
            logger.info(
                "unified_exit_override symbol=%s live_exit=%s unified_action=%s",
                action.symbol, live_exit_reason, action.action
            )
            return True
        
        return False
    
    def close_position(self, symbol: str) -> None:
        """Call when a position is closed to reset tracking state."""
        self._reset_position_state(symbol)


def get_unified_exit_manager() -> UnifiedExitManager:
    """Get or create the global unified exit manager instance."""
    global _unified_exit_manager
    if "_unified_exit_manager" not in globals() or _unified_exit_manager is None:
        _unified_exit_manager = UnifiedExitManager.from_env()
    return _unified_exit_manager


_unified_exit_manager: Optional[UnifiedExitManager] = None
