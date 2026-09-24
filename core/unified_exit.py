"""Unified exit manager for day-trade positions — SHADOW LOGGING ONLY.

v-unified-exit-2026-09-24-r2. PR3 of the pro-trader rebuild. Implements
the unified exit policy from the blueprint:

  1. Structure-based initial stop (not indicator-based)
  2. Scale out 50% at +1R, move stop to breakeven
  3. Trail the remainder (ATR or swing low)
  4. Time stop N minutes before flatten

SHADOW-ONLY MODE:
  DT_UNIFIED_EXIT=1 → shadow logging enabled, NO live enforcement
  
  Live enforce has been removed entirely. This module only logs what
  the unified policy WOULD do. The shadow log can be analyzed offline
  to validate the policy before any live integration is considered.

Hard stops and circuits are NEVER bypassed by this module.
Uses Config().HANDS_OFF_DENYLIST for hands-off symbols.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from core.models import Position

logger = logging.getLogger("TradingBot")

ATR_STOP_MULT = 1.5
RR_RATIO = 2.0
SCALE_AT_R = 1.0
TRAIL_ATR_MULT = 1.5
TIME_STOP_MINUTES_BEFORE_FLATTEN = 15


def _get_repo_root() -> Path:
    """Get repository root for absolute data paths."""
    return Path(__file__).resolve().parent.parent


def _get_shadow_log_path() -> Path:
    """Return absolute path for shadow log file."""
    return _get_repo_root() / "data" / "shadow_unified_exit.ndjson"


def _get_hands_off_denylist() -> frozenset:
    """Get hands-off symbols from Config, with fallback."""
    try:
        from core.config import Config
        return Config().HANDS_OFF_DENYLIST
    except Exception:
        return frozenset({"MU", "HQGE", "SPCX"})


def _get_flatten_hour() -> int:
    """Get flatten hour from Config, with fallback."""
    try:
        from core.config import Config
        return Config().DAY_TRADE_FLATTEN_HOUR
    except Exception:
        return 15


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
    scaled_out: bool = False
    trailing_active: bool = False
    entry_time: Optional[str] = None  # For position keying


class UnifiedExitManager:
    """Computes unified exit policy for day-trade positions — SHADOW ONLY.
    
    Call evaluate_position() on each position manager tick. Returns what
    the unified policy WOULD do. Shadow-logged only — NO live enforcement.
    
    Hard stops and flatten-hour are NOT managed here — they remain in the
    engine's core loop and fire regardless of this policy.
    
    Position state is keyed by (symbol, entry_time) to correctly handle
    re-entries in the same symbol on the same day.
    """
    
    _last_warning_time: float = 0.0
    _WARNING_INTERVAL: float = 60.0  # Rate limit warnings to 1/minute
    
    def __init__(
        self,
        shadow_log_path: Optional[Path] = None,
        enabled: bool = True,
        time_stop_minutes: int = TIME_STOP_MINUTES_BEFORE_FLATTEN,
        flatten_hour: Optional[int] = None,
    ):
        self.shadow_log_path = shadow_log_path or _get_shadow_log_path()
        self.enabled = enabled
        self.time_stop_minutes = time_stop_minutes
        self.flatten_hour = flatten_hour if flatten_hour is not None else _get_flatten_hour()
        
        self._position_state: dict[tuple[str, str], dict[str, Any]] = {}
        self._last_logged_action: dict[tuple[str, str], str] = {}
        
        self.shadow_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_config(cls) -> "UnifiedExitManager":
        """Create manager from Config (env + YAML merged)."""
        try:
            from core.config import Config
            cfg = Config()
            enabled = cfg.DT_UNIFIED_EXIT
            flatten_hour = cfg.DAY_TRADE_FLATTEN_HOUR
        except Exception:
            enabled = False
            flatten_hour = 15
        
        return cls(
            enabled=enabled,
            flatten_hour=flatten_hour,
        )
    
    @classmethod
    def from_env(cls) -> "UnifiedExitManager":
        """DEPRECATED: Use from_config() instead. Kept for test compatibility."""
        return cls.from_config()
    
    def _get_position_key(self, symbol: str, entry_time: Optional[datetime]) -> tuple[str, str]:
        """Get the key for position state tracking."""
        symbol_upper = symbol.upper()
        entry_str = entry_time.isoformat() if entry_time else "unknown"
        return (symbol_upper, entry_str)
    
    def _get_position_state(self, symbol: str, entry_time: Optional[datetime] = None) -> dict[str, Any]:
        """Get or initialize position tracking state."""
        key = self._get_position_key(symbol, entry_time)
        if key not in self._position_state:
            self._position_state[key] = {
                "scaled_out": False,
                "trailing_active": False,
                "highest_price": None,
                "breakeven_stop_set": False,
            }
        return self._position_state[key]
    
    def _reset_position_state(self, symbol: str, entry_time: Optional[datetime] = None) -> None:
        """Reset state when position is closed."""
        key = self._get_position_key(symbol, entry_time)
        if key in self._position_state:
            del self._position_state[key]
        if key in self._last_logged_action:
            del self._last_logged_action[key]
    
    def _is_time_stop_window(self, now_et_hour: int, now_et_minute: int) -> bool:
        """Check if we're in the time-stop window (N min before flatten)."""
        flatten_time_minutes = self.flatten_hour * 60
        current_minutes = now_et_hour * 60 + now_et_minute
        cutoff_minutes = flatten_time_minutes - self.time_stop_minutes
        return current_minutes >= cutoff_minutes
    
    def _warn_rate_limited(self, msg: str, *args) -> None:
        """Log a warning at most once per minute."""
        now = time.time()
        if now - self._last_warning_time >= self._WARNING_INTERVAL:
            logger.warning(msg, *args)
            self._last_warning_time = now
    
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
            atr: Current ATR value (pass current_atr from engine, NOT proactive_indicators)
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
        
        hands_off = _get_hands_off_denylist()
        if symbol in hands_off:
            return None
        
        reasoning = getattr(position, "reasoning", {}) or {}
        strategy = reasoning.get("strategy", "")
        if "day_trade" not in strategy.lower() and "momentum" not in strategy.lower():
            return None
        
        if not getattr(position, "managed_by_bot", False):
            return None
        
        if position.side != "long":
            return None
        
        entry_time = getattr(position, "entry_time", None)
        state = self._get_position_state(symbol, entry_time)
        
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
        
        entry_time_str = entry_time.isoformat() if entry_time else None
        
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
            scaled_out=state["scaled_out"],
            trailing_active=state["trailing_active"],
            entry_time=entry_time_str,
        )
        
        self._log_shadow_if_changed(result, entry_time)
        
        return result
    
    def _log_shadow_if_changed(self, action: UnifiedExitAction, entry_time: Optional[datetime]) -> None:
        """Append action to shadow NDJSON log only if action changed from last tick."""
        key = self._get_position_key(action.symbol, entry_time)
        current_action_key = f"{action.action}:{action.reason}"
        
        if self._last_logged_action.get(key) == current_action_key:
            return
        
        self._last_logged_action[key] = current_action_key
        
        try:
            with open(self.shadow_log_path, "a") as f:
                f.write(json.dumps(asdict(action), default=str) + "\n")
        except Exception as exc:
            self._warn_rate_limited("unified_exit_shadow_log_failed: %s", exc)
    
    def _log_shadow(self, action: UnifiedExitAction) -> None:
        """DEPRECATED: Use _log_shadow_if_changed. Kept for test compatibility."""
        try:
            with open(self.shadow_log_path, "a") as f:
                f.write(json.dumps(asdict(action), default=str) + "\n")
        except Exception as exc:
            self._warn_rate_limited("unified_exit_shadow_log_failed: %s", exc)
    
    def should_override_proactive_exit(
        self,
        action: UnifiedExitAction,
        live_exit_reason: str,
    ) -> bool:
        """SHADOW-ONLY: Always returns False.
        
        Live enforcement has been removed entirely from this module.
        This method exists only for API compatibility and logging.
        The caller should NOT use this to skip exits — it's shadow-only.
        """
        return False
    
    def close_position(self, symbol: str, entry_time: Optional[datetime] = None) -> None:
        """Call when a position is closed to reset tracking state."""
        self._reset_position_state(symbol, entry_time)


def get_unified_exit_manager() -> UnifiedExitManager:
    """Get or create the global unified exit manager instance."""
    global _unified_exit_manager
    if "_unified_exit_manager" not in globals() or _unified_exit_manager is None:
        _unified_exit_manager = UnifiedExitManager.from_config()
    return _unified_exit_manager


def reset_unified_exit_manager() -> None:
    """Reset the global manager instance (for testing)."""
    global _unified_exit_manager
    _unified_exit_manager = None


_unified_exit_manager: Optional[UnifiedExitManager] = None


HANDS_OFF_SYMBOLS = _get_hands_off_denylist()
