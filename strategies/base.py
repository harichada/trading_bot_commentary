"""Base class for trading strategies."""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from core.models import TradingSignal


_logger = logging.getLogger("TradingBot")


class TradingStrategyWithCommentary(ABC):
    """Base class for strategies with commentary."""

    name: str = "strategy"  # subclasses override

    def __init__(self, commentary_system):
        self.commentary = commentary_system
        # v-feature-snapshot-2026-09-09: engine_ref set by engine for snapshot writes
        self._engine_ref = None
        # v-feature-snapshot-2026-09-09: track last snapshot for commentary metadata
        self._last_snapshot_id: Optional[str] = None
        self._last_snapshot_ts: Optional[str] = None

    @abstractmethod
    async def generate_signal_with_commentary(self, market_data) -> Optional[TradingSignal]:
        ...

    def _log_decision(
        self,
        market_data,
        action: str,
        reason: str,
        gate_name: Optional[str] = None,
        confidence: float = 0.0,
        would_entry_price: Optional[float] = None,
        would_stop_loss: Optional[float] = None,
        would_take_profit: Optional[float] = None,
        would_size_shares: Optional[int] = None,
        would_size_mult: Optional[float] = None,
        news_aggregate: Optional[Dict[str, Any]] = None,
        news_gate_result: Optional[Any] = None,
        regime_context: Optional[Any] = None,
        **details: Any,
    ) -> None:
        """Structured per-tick decision log + snapshot capture.

        Every strategy call should emit exactly one of these so the log file
        is a complete audit trail even when the UI commentary stays quiet.
        action: 'signal_buy' | 'signal_sell' | 'skip' | 'error'
        reason: short snake_case label
        
        v-feature-snapshot-2026-09-09: also emits DecisionSnapshot for ML training.
        """
        symbol = getattr(market_data, "symbol", "?")
        close = getattr(market_data, "close", None)
        kv = " ".join(f"{k}={v}" for k, v in details.items())
        _logger.info(
            "strategy_decision strategy=%s symbol=%s action=%s reason=%s close=%s %s",
            self.name, symbol, action, reason, close, kv,
        )
        
        # v-feature-snapshot-2026-09-09: emit DecisionSnapshot for ML training
        self._emit_snapshot(
            market_data=market_data,
            action=action,
            reason=reason,
            gate_name=gate_name,
            confidence=confidence,
            would_entry_price=would_entry_price,
            would_stop_loss=would_stop_loss,
            would_take_profit=would_take_profit,
            would_size_shares=would_size_shares,
            would_size_mult=would_size_mult,
            news_aggregate=news_aggregate,
            news_gate_result=news_gate_result,
            regime_context=regime_context,
            extra=details,
        )

    def _emit_snapshot(
        self,
        market_data,
        action: str,
        reason: str,
        gate_name: Optional[str] = None,
        confidence: float = 0.0,
        would_entry_price: Optional[float] = None,
        would_stop_loss: Optional[float] = None,
        would_take_profit: Optional[float] = None,
        would_size_shares: Optional[int] = None,
        would_size_mult: Optional[float] = None,
        news_aggregate: Optional[Dict[str, Any]] = None,
        news_gate_result: Optional[Any] = None,
        regime_context: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """v-feature-snapshot-2026-09-09: build and persist DecisionSnapshot.
        
        Fire-and-forget via asyncio task. Never blocks the analysis loop.
        """
        try:
            from core.decision_snapshot import (
                DecisionAction, build_snapshot, is_snapshot_logging_enabled
            )
            if not is_snapshot_logging_enabled():
                return
            
            # Get engine reference for db_logger and mode
            engine = self._engine_ref
            if engine is None:
                # Try getting via commentary.engine_ref (legacy path)
                engine = getattr(self.commentary, "engine_ref", None)
            if engine is None or getattr(engine, "db_logger", None) is None:
                return
            
            symbol = getattr(market_data, "symbol", "UNKNOWN")
            price = float(getattr(market_data, "close", 0) or 0)
            indicators = getattr(market_data, "indicators", {}) or {}
            mode = getattr(engine, "mode", None)
            mode_str = mode.value if hasattr(mode, "value") else str(mode or "simulation")
            
            # Map action string to DecisionAction enum
            action_map = {
                "signal_buy": DecisionAction.SIGNAL_BUY,
                "signal_sell": DecisionAction.SIGNAL_SELL,
                "skip": DecisionAction.SKIP,
                "veto": DecisionAction.VETO,
                "error": DecisionAction.ERROR,
            }
            action_enum = action_map.get(action.lower(), DecisionAction.SKIP)
            
            snapshot = build_snapshot(
                symbol=symbol,
                strategy_id=self.name,
                action=action_enum,
                reason=reason,
                mode=mode_str,
                gate_name=gate_name,
                confidence=confidence,
                indicators=indicators,
                price=price,
                news_aggregate=news_aggregate,
                news_gate_result=news_gate_result,
                regime_context=regime_context,
                would_entry_price=would_entry_price,
                would_stop_loss=would_stop_loss,
                would_take_profit=would_take_profit,
                would_size_shares=would_size_shares,
                would_size_mult=would_size_mult,
                extra=extra,
            )
            
            # Track for commentary metadata attachment
            self._last_snapshot_id = snapshot.snapshot_id
            self._last_snapshot_ts = snapshot.ts.isoformat()
            
            # Fire-and-forget async write
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(engine.db_logger.log_decision_snapshot(snapshot))
            except RuntimeError:
                # No event loop running — skip snapshot (shouldn't happen in normal operation)
                pass
                
        except Exception as exc:
            _logger.debug("snapshot_emit_error strategy=%s err=%s", self.name, exc)
    
    def get_snapshot_metadata(self) -> Dict[str, Any]:
        """v-feature-snapshot-2026-09-09: get metadata for last emitted snapshot.
        
        Use this to attach snapshot reference to Commentary.data for
        Decision Card correlation.
        """
        if self._last_snapshot_id:
            return {
                "snapshot_id": self._last_snapshot_id,
                "snapshot_ts": self._last_snapshot_ts,
                "snapshot_strategy": self.name,
            }
        return {}
