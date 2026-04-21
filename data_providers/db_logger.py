"""Asynchronous Postgres logger for bot decisions and trades.

Every engine decision and completed trade is written to Postgres so
the full audit trail is queryable with SQL — no more grepping log
files or parsing JSON state.

Tables: bot_decisions, bot_trades (created by migration in engine init).

Thread-safe: uses a dedicated connection pool. Non-blocking: writes
are fire-and-forget via asyncio tasks. A failed DB write logs a
warning but never crashes the trading loop.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger("TradingBot")

DEFAULT_DSN = "postgresql+asyncpg://rudra:rudra_dev_2024@localhost:5432/rudra_dev"


class DbLogger:
    """Fire-and-forget Postgres writer for decisions + trades."""

    def __init__(self, dsn: str | None = None) -> None:
        raw_dsn = dsn or os.environ.get("POSTGRES_DSN_ASYNC", DEFAULT_DSN)
        # Ensure async driver
        if "asyncpg" not in raw_dsn:
            raw_dsn = raw_dsn.replace("postgresql://", "postgresql+asyncpg://")
        self._engine: AsyncEngine = create_async_engine(
            raw_dsn, pool_size=2, max_overflow=2, pool_pre_ping=True,
        )
        self._enabled = True
        # Serialise sync_positions so two fire-and-forget tasks can't race
        # into overlapping DELETE+INSERT transactions (symptom: repeated
        # UniqueViolationError on bot_positions_pkey when _save_state
        # fires twice in quick succession).
        self._sync_positions_lock = asyncio.Lock()

    async def close(self) -> None:
        await self._engine.dispose()

    async def log_decision(
        self,
        component: str,
        symbol: str | None,
        action: str,
        reason: str | None = None,
        mode: str | None = None,
        signal_type: int | None = None,
        confidence: float | None = None,
        strength: float | None = None,
        meta_proba: float | None = None,
        atr: float | None = None,
        stop_distance: float | None = None,
        price: float | None = None,
        **extra: Any,
    ) -> None:
        """Insert one row into bot_decisions. Never raises."""
        if not self._enabled:
            return
        try:
            details = {k: _safe_json(v) for k, v in extra.items()} if extra else {}
            async with self._engine.begin() as conn:
                await conn.execute(
                    text("""
                        INSERT INTO bot_decisions
                            (ts, component, symbol, action, reason, mode,
                             signal_type, confidence, strength, meta_proba,
                             atr, stop_distance, price, details_json)
                        VALUES
                            (NOW(), :component, :symbol, :action, :reason, :mode,
                             :signal_type, :confidence, :strength, :meta_proba,
                             :atr, :stop_distance, :price, :details_json)
                    """),
                    {
                        "component": component,
                        "symbol": symbol,
                        "action": action,
                        "reason": reason,
                        "mode": mode,
                        "signal_type": signal_type,
                        "confidence": confidence,
                        "strength": strength,
                        "meta_proba": meta_proba,
                        "atr": atr,
                        "stop_distance": stop_distance,
                        "price": price,
                        "details_json": json.dumps(details) if details else "{}",
                    },
                )
        except Exception as exc:
            logger.warning("db_logger_decision_error err=%s", exc)

    async def log_trade(
        self,
        symbol: str,
        side: str,
        strategy: str | None,
        entry_time: datetime,
        exit_time: datetime,
        entry_price: float,
        exit_price: float,
        quantity: int,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
        atr_at_entry: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        confidence: float | None = None,
        meta_proba: float | None = None,
        kelly_fraction: float | None = None,
        scaled_out: bool = False,
        mode: str | None = None,
        reasoning: dict | None = None,
    ) -> None:
        """Insert one row into bot_trades. Never raises."""
        if not self._enabled:
            return
        try:
            async with self._engine.begin() as conn:
                await conn.execute(
                    text("""
                        INSERT INTO bot_trades
                            (symbol, side, strategy, entry_time, exit_time,
                             entry_price, exit_price, quantity, pnl, pnl_pct,
                             exit_reason, atr_at_entry, stop_loss, take_profit,
                             confidence, meta_proba, kelly_fraction, scaled_out,
                             mode, reasoning_json)
                        VALUES
                            (:symbol, :side, :strategy, :entry_time, :exit_time,
                             :entry_price, :exit_price, :quantity, :pnl, :pnl_pct,
                             :exit_reason, :atr_at_entry, :stop_loss, :take_profit,
                             :confidence, :meta_proba, :kelly_fraction, :scaled_out,
                             :mode, :reasoning_json)
                    """),
                    {
                        "symbol": symbol,
                        "side": side,
                        "strategy": strategy,
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "quantity": quantity,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "exit_reason": exit_reason,
                        "atr_at_entry": atr_at_entry,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "confidence": confidence,
                        "meta_proba": meta_proba,
                        "kelly_fraction": kelly_fraction,
                        "scaled_out": scaled_out,
                        "mode": mode,
                        "reasoning_json": json.dumps(reasoning or {}),
                    },
                )
        except Exception as exc:
            logger.warning("db_logger_trade_error err=%s", exc)

    async def sync_positions(self, positions: dict) -> None:
        """Upsert all open positions and delete closed ones.

        Called from _save_state() every loop tick so the DB always
        reflects the current portfolio. Dashboard/queries can read
        bot_positions instead of parsing JSON.

        Concurrency: guarded by self._sync_positions_lock. _save_state is
        called both periodically (every 10 cycles) and after each trade,
        and it uses asyncio.create_task to dispatch this coroutine without
        awaiting, so two sync_positions can otherwise run in parallel and
        race inside the DELETE+INSERT transaction. The lock serialises
        them; combined with ON CONFLICT DO UPDATE on the INSERT, the
        operation is now both safe under concurrency and idempotent.
        """
        if not self._enabled:
            return
        async with self._sync_positions_lock:
            try:
                # Compute the set of symbols we want to keep so we can
                # delete only those that are no longer open — this avoids
                # DELETE-all-then-INSERT patterns that briefly empty the
                # table for any concurrent reader.
                keep_symbols = [
                    sym for sym, pos in positions.items()
                    if pos is not None and getattr(pos, 'quantity', 0) > 0
                ]
                async with self._engine.begin() as conn:
                    if keep_symbols:
                        # <>ALL(:keep) matches against a pg array param —
                        # avoids expanding-bindparam complexity of NOT IN.
                        await conn.execute(
                            text("DELETE FROM bot_positions "
                                 "WHERE symbol <> ALL(:keep)"),
                            {"keep": list(keep_symbols)},
                        )
                    else:
                        await conn.execute(text("DELETE FROM bot_positions"))
                    for symbol, pos in positions.items():
                        if pos is None or getattr(pos, 'quantity', 0) <= 0:
                            continue
                        await conn.execute(
                            text("""
                                INSERT INTO bot_positions
                                    (symbol, side, strategy, entry_time, entry_price,
                                     current_price, quantity, stop_loss, take_profit,
                                     trailing_stop, original_stop, scaled_out,
                                     unrealized_pnl, atr_at_entry, confidence, mode,
                                     updated_at)
                                VALUES
                                    (:symbol, :side, :strategy, :entry_time, :entry_price,
                                     :current_price, :quantity, :stop_loss, :take_profit,
                                     :trailing_stop, :original_stop, :scaled_out,
                                     :unrealized_pnl, :atr_at_entry, :confidence, :mode,
                                     NOW())
                                ON CONFLICT (symbol) DO UPDATE SET
                                    side             = EXCLUDED.side,
                                    strategy         = EXCLUDED.strategy,
                                    entry_time       = EXCLUDED.entry_time,
                                    entry_price      = EXCLUDED.entry_price,
                                    current_price    = EXCLUDED.current_price,
                                    quantity         = EXCLUDED.quantity,
                                    stop_loss        = EXCLUDED.stop_loss,
                                    take_profit      = EXCLUDED.take_profit,
                                    trailing_stop    = EXCLUDED.trailing_stop,
                                    original_stop    = EXCLUDED.original_stop,
                                    scaled_out       = EXCLUDED.scaled_out,
                                    unrealized_pnl   = EXCLUDED.unrealized_pnl,
                                    atr_at_entry     = EXCLUDED.atr_at_entry,
                                    confidence       = EXCLUDED.confidence,
                                    mode             = EXCLUDED.mode,
                                    updated_at       = NOW()
                            """),
                            {
                                "symbol": pos.symbol,
                                "side": getattr(pos, "side", "long"),
                                "strategy": (getattr(pos, "reasoning", {}) or {}).get("strategy"),
                                "entry_time": pos.entry_time,
                                "entry_price": pos.entry_price,
                                "current_price": pos.current_price,
                                "quantity": pos.quantity,
                                "stop_loss": pos.stop_loss,
                                "take_profit": pos.take_profit,
                                "trailing_stop": getattr(pos, "trailing_stop", None),
                                "original_stop": getattr(pos, "original_stop", None),
                                "scaled_out": getattr(pos, "scaled_out", False),
                                "unrealized_pnl": getattr(pos, "unrealized_pnl", 0),
                                "atr_at_entry": (getattr(pos, "reasoning", {}) or {}).get("atr"),
                                "confidence": getattr(pos, "confidence", None),
                                # v-mode-field-2026-04-20: was hardcoded "simulation";
                                # now reads attribute-based tag set at Position
                                # construction. Fallback "simulation" is the safe
                                # default for objects created before the upgrade.
                                "mode": getattr(pos, "mode", "simulation"),
                            },
                        )
            except Exception as exc:
                logger.warning("db_logger_positions_error err=%s", exc)

    async def delete_position(self, symbol: str) -> None:
        """Remove a closed position from bot_positions."""
        if not self._enabled:
            return
        try:
            async with self._engine.begin() as conn:
                await conn.execute(
                    text("DELETE FROM bot_positions WHERE symbol = :symbol"),
                    {"symbol": symbol},
                )
        except Exception as exc:
            logger.warning("db_logger_delete_position_error err=%s", exc)


def _safe_json(value: Any) -> Any:
    """Coerce a value to JSON-serializable form."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, set):
        return list(value)
    return str(value)
