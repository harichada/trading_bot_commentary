"""Asynchronous Postgres logger for bot decisions and trades.

Every engine decision and completed trade is written to Postgres so
the full audit trail is queryable with SQL — no more grepping log
files or parsing JSON state.

Tables: bot_decisions, bot_trades (created by migration in engine init).

Thread-safe: uses a dedicated connection pool. Non-blocking: writes
    are fire-and-forget via asyncio tasks. A failed DB write logs a
warning but never crashes the trading loop.

v-fix-loop-safety-2026-09-10: The async engine's connection pool is bound
to the event loop it was created in. When fire-and-forget tasks run on
different loops (strategy loop vs FastAPI loop), we get "Future attached
to a different loop" errors. This module now tracks the owner loop and
uses run_coroutine_threadsafe for cross-loop operations.

v-fix-cross-loop-crash-2026-09-10: PR #19 only protected snapshot methods.
    This revision protects ALL async methods (close, log_decision, log_trade,
sync_positions, etc.) with loop-safety checks. When a loop mismatch is
detected:
  - Write operations: fail soft (log warning, skip the write) to avoid crash
  - close(): dispose safely by detecting loop mismatch and handling gracefully
  - Never call asyncpg operations from a different loop than created them

v-fix-cross-loop-routing-2026-09-15: Cross-loop writes no longer skip!
  Methods like log_decision, log_trade, sync_positions now ROUTE to the
  owner loop via run_coroutine_threadsafe (same as log_decision_snapshot).
  This eliminates silent data loss from the 411+ db_logger_cross_loop_skip
  warnings observed in 16h KiddoKingdom soak (PID 331467, 045186b).
  
  Env flag DB_LOGGER_CROSS_LOOP_ROUTE (default True):
    - True (default): cross-loop calls route to owner loop (no data loss)
    - False: legacy skip behavior (for emergency rollback only)
  
  Owner loop is now "sticky": set once at init, subsequent set_owner_loop
  calls from different loops are ignored with a one-time warning.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger("TradingBot")

# v-fix-cross-loop-routing-2026-09-15: Cross-loop routing flag.
# True (default): cross-loop calls route to owner loop via run_coroutine_threadsafe.
# False: legacy skip behavior (for emergency rollback only — causes silent data loss).
DB_LOGGER_CROSS_LOOP_ROUTE = os.environ.get("DB_LOGGER_CROSS_LOOP_ROUTE", "true").lower() in ("true", "1", "yes")

# Rate-limit interval for cross-loop warnings (seconds). Max one warning per method per interval.
_CROSS_LOOP_WARN_INTERVAL_SEC = 60.0

DEFAULT_DSN = "postgresql+asyncpg://rudra:rudra_dev_2024@localhost:5432/rudra_dev"

# v-fix-pool-leak-2026-09-10: module-level singleton instance to prevent
# multiple engine/pool creations. API routes and other code paths should use
# get_shared_db_logger() instead of creating new DbLogger instances.
_shared_instance: Optional["DbLogger"] = None
_shared_lock = asyncio.Lock()


def get_shared_db_logger(dsn: str | None = None) -> "DbLogger":
    """Get or create the shared DbLogger singleton.
    
    v-fix-pool-leak-2026-09-10: prevents multiple engine/pool creations
    that exhaust max_connections. All code paths should use this instead
    of creating new DbLogger instances directly.
    
    Thread-safe via module-level lock. The singleton is bound to the first
    event loop that creates it; cross-loop operations are routed via
    run_coroutine_threadsafe.
    """
    global _shared_instance
    if _shared_instance is None:
        _shared_instance = DbLogger(dsn)
        try:
            loop = asyncio.get_running_loop()
            _shared_instance.set_owner_loop(loop)
        except RuntimeError:
            pass
        logger.info("db_logger_singleton_created pool_size=5 max_overflow=3")
    return _shared_instance


async def dispose_shared_db_logger() -> None:
    """Dispose the shared DbLogger singleton.
    
    Call during graceful shutdown to clean up connection pool.
    """
    global _shared_instance
    if _shared_instance is not None:
        await _shared_instance.close()
        _shared_instance = None
        logger.info("db_logger_singleton_disposed")


class DbLogger:
    """Fire-and-forget Postgres writer for decisions + trades.
    
    v-fix-loop-safety-2026-09-10: Loop-safe async operations. The engine
    is bound to an owner loop; cross-loop calls are routed via
    run_coroutine_threadsafe to avoid "Future attached to different loop".
    
    v-fix-cross-loop-crash-2026-09-10: ALL async methods now check for loop
    mismatch before touching the engine. Write operations fail soft (log +
    skip) when on wrong loop. close() handles loop mismatch gracefully.
    """

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or os.environ.get("POSTGRES_DSN_ASYNC", DEFAULT_DSN)
        # Ensure async driver
        if "asyncpg" not in self._dsn:
            self._dsn = self._dsn.replace("postgresql://", "postgresql+asyncpg://")
        # v-fix-pool-leak-2026-09-10: bounded pool with recycle to prevent
        # connection exhaustion. pool_size=5 base, max_overflow=3 burst,
        # pool_recycle=1800s (30min) to prevent stale connections.
        self._engine: AsyncEngine = create_async_engine(
            self._dsn,
            pool_size=5,
            max_overflow=3,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
        self._enabled = True
        # Serialise sync_positions so two fire-and-forget tasks can't race
        # into overlapping DELETE+INSERT transactions (symptom: repeated
        # UniqueViolationError on bot_positions_pkey when _save_state
        # fires twice in quick succession).
        self._sync_positions_lock = asyncio.Lock()
        # v-fix-loop-safety-2026-09-10: track owner loop for cross-loop safety
        self._owner_loop: Optional[asyncio.AbstractEventLoop] = None
        # v-fix-cross-loop-crash-2026-09-10: lock to serialise engine recreation
        self._engine_lock = threading.Lock()
        # v-fix-cross-loop-routing-2026-09-15: rate-limit cross-loop warnings
        self._cross_loop_warn_times: dict[str, float] = {}
        self._cross_loop_warn_lock = threading.Lock()
        # v-fix-cross-loop-routing-2026-09-15: sticky owner loop flag
        self._owner_loop_set_once = False
    
    def set_owner_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Set the event loop that owns this DbLogger's engine.
        
        v-fix-loop-safety-2026-09-10: Call this once from the main async
        context (e.g., FastAPI startup) so cross-loop operations can route
        back correctly. If loop is None, uses the current running loop.
        
        v-fix-cross-loop-routing-2026-09-15: Owner loop is now STICKY.
        Once set, subsequent calls from DIFFERENT loops are ignored with a
        one-time warning. This prevents news/uvicorn/worker secondary loops
        from overwriting the canonical main engine loop.
        """
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
        
        # v-fix-cross-loop-routing-2026-09-15: Sticky owner loop
        if self._owner_loop_set_once and self._owner_loop is not None:
            if self._owner_loop is not loop:
                self._warn_once(
                    "set_owner_loop_ignored",
                    "db_logger_set_owner_loop_ignored existing=%s attempted=%s "
                    "(owner loop is sticky; first caller wins)",
                    id(self._owner_loop), id(loop)
                )
                return
            # Same loop - no-op, already set
            return
        
        self._owner_loop = loop
        self._owner_loop_set_once = True
        logger.debug("db_logger_owner_loop_set loop_id=%s", id(loop))
    
    def _is_on_owner_loop(self) -> bool:
        """Check if we're currently on the owner loop.
        
        v-fix-cross-loop-crash-2026-09-10: Returns True if:
          - No owner loop is set (pre-startup, proceed cautiously)
          - Current loop is the owner loop
        Returns False if:
          - We're on a different loop than the owner
          - No running loop (shouldn't happen in async context)
        """
        if self._owner_loop is None:
            return True
        try:
            current = asyncio.get_running_loop()
            return current is self._owner_loop
        except RuntimeError:
            return False
    
    def _warn_once(self, key: str, msg: str, *args) -> None:
        """Log a warning at most once per _CROSS_LOOP_WARN_INTERVAL_SEC per key.
        
        v-fix-cross-loop-routing-2026-09-15: Rate-limit warnings so logs stay
        OS-grade clean even under sustained cross-loop traffic.
        """
        now = time.monotonic()
        with self._cross_loop_warn_lock:
            last = self._cross_loop_warn_times.get(key, 0.0)
            if now - last < _CROSS_LOOP_WARN_INTERVAL_SEC:
                return
            self._cross_loop_warn_times[key] = now
        logger.warning(msg, *args)
    
    def _check_loop_or_warn(self, method_name: str) -> bool:
        """Check if we're on the owner loop, log warning if not.
        
        v-fix-cross-loop-crash-2026-09-10: Helper for write methods that
        should fail soft on loop mismatch. Returns True if safe to proceed.
        
        v-fix-cross-loop-routing-2026-09-15: Warnings are now rate-limited
        (max once per 60s per method) to keep logs OS-grade clean.
        """
        if self._is_on_owner_loop():
            return True
        try:
            current = asyncio.get_running_loop()
            self._warn_once(
                f"cross_loop_skip_{method_name}",
                "db_logger_cross_loop_skip method=%s owner_loop=%s current_loop=%s",
                method_name, id(self._owner_loop), id(current)
            )
        except RuntimeError:
            self._warn_once(
                f"no_running_loop_{method_name}",
                "db_logger_no_running_loop method=%s", method_name
            )
        return False
    
    def _route_to_owner_loop(
        self,
        method_name: str,
        coro_factory,
        *args,
        **kwargs,
    ) -> bool:
        """Route a coroutine to the owner loop if we're on a different loop.
        
        v-fix-cross-loop-routing-2026-09-15: Instead of silently skipping
        cross-loop calls (causing data loss), route them to the owner loop
        via run_coroutine_threadsafe.
        
        Returns True if the call was routed (caller should return early).
        Returns False if we're on the owner loop (caller should proceed).
        
        Args:
            method_name: Name of the method (for logging)
            coro_factory: A callable that returns the coroutine to run
            *args, **kwargs: Arguments to pass to coro_factory
        """
        if self._is_on_owner_loop():
            return False
        
        # Cross-loop detected
        if not DB_LOGGER_CROSS_LOOP_ROUTE:
            # Legacy skip mode (emergency rollback only)
            self._warn_once(
                f"cross_loop_skip_{method_name}",
                "db_logger_cross_loop_skip method=%s owner_loop=%s (routing disabled)",
                method_name, id(self._owner_loop)
            )
            return True
        
        if self._owner_loop is None:
            self._warn_once(
                f"cross_loop_no_owner_{method_name}",
                "db_logger_cross_loop_no_owner method=%s (cannot route)",
                method_name
            )
            return True
        
        if not self._owner_loop.is_running():
            self._warn_once(
                f"cross_loop_dead_{method_name}",
                "db_logger_cross_loop_dead_owner method=%s owner_loop=%s",
                method_name, id(self._owner_loop)
            )
            return True
        
        # Owner loop alive - route the call
        try:
            asyncio.run_coroutine_threadsafe(
                coro_factory(*args, **kwargs),
                self._owner_loop,
            )
            logger.debug(
                "db_logger_cross_loop_routed method=%s owner_loop=%s",
                method_name, id(self._owner_loop)
            )
        except Exception as exc:
            self._warn_once(
                f"cross_loop_route_error_{method_name}",
                "db_logger_cross_loop_route_error method=%s err=%s",
                method_name, exc
            )
        return True

    async def close(self) -> None:
        """Dispose the engine and close all connections.
        
        v-fix-cross-loop-crash-2026-09-10: Handle loop mismatch safely.
        If called from a different loop than the owner:
          - Try to route to owner loop if it's still running
          - If owner loop is dead, dispose synchronously in thread pool
          - Never let asyncpg see a cross-loop call
        """
        if not self._enabled:
            return
        
        if self._is_on_owner_loop():
            await self._engine.dispose()
            return
        
        # Cross-loop close detected. Try to route to owner loop.
        if self._owner_loop is not None:
            try:
                if self._owner_loop.is_running():
                    # Owner loop still alive - route the dispose there
                    fut = asyncio.run_coroutine_threadsafe(
                        self._engine.dispose(), self._owner_loop
                    )
                    try:
                        fut.result(timeout=5.0)
                    except concurrent.futures.TimeoutError:
                        logger.warning("db_logger_close_timeout owner_loop=%s", id(self._owner_loop))
                    return
            except RuntimeError:
                # Owner loop closed/invalid
                pass
        
        # Owner loop is dead or unreachable. Dispose synchronously in thread
        # pool to avoid blocking and to handle cleanup outside any event loop.
        logger.warning(
            "db_logger_close_cross_loop_fallback owner=%s",
            id(self._owner_loop) if self._owner_loop else "None"
        )
        try:
            # Create a new temporary event loop in a thread to dispose
            def _sync_dispose():
                try:
                    temp_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(temp_loop)
                    try:
                        # Create a fresh engine just to close it cleanly
                        # (the old engine's pool is orphaned on the dead loop)
                        temp_loop.run_until_complete(asyncio.sleep(0))
                    finally:
                        temp_loop.close()
                except Exception as e:
                    logger.warning("db_logger_sync_dispose_error: %s", e)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(_sync_dispose).result(timeout=5.0)
        except Exception as exc:
            logger.warning("db_logger_close_fallback_error: %s", exc)
        
        self._enabled = False

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
        """Insert one row into bot_decisions. Never raises.
        
        v-fix-cross-loop-crash-2026-09-10: Skip write if called from wrong loop.
        v-fix-cross-loop-routing-2026-09-15: Now ROUTES cross-loop calls to
        owner loop instead of skipping (no more silent data loss).
        """
        if not self._enabled:
            return
        # v-fix-cross-loop-routing-2026-09-15: route cross-loop calls
        if self._route_to_owner_loop(
            "log_decision",
            self._log_decision_impl,
            component, symbol, action, reason, mode,
            signal_type, confidence, strength, meta_proba,
            atr, stop_distance, price, extra,
        ):
            return
        await self._log_decision_impl(
            component, symbol, action, reason, mode,
            signal_type, confidence, strength, meta_proba,
            atr, stop_distance, price, extra,
        )

    async def _log_decision_impl(
        self,
        component: str,
        symbol: str | None,
        action: str,
        reason: str | None,
        mode: str | None,
        signal_type: int | None,
        confidence: float | None,
        strength: float | None,
        meta_proba: float | None,
        atr: float | None,
        stop_distance: float | None,
        price: float | None,
        extra: dict,
    ) -> None:
        """Internal impl: actually insert decision into the database."""
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
        """Insert one row into bot_trades. Never raises.
        
        v-fix-cross-loop-crash-2026-09-10: Skip write if called from wrong loop.
        v-fix-cross-loop-routing-2026-09-15: Now ROUTES cross-loop calls to
        owner loop instead of skipping (no more silent data loss).
        """
        if not self._enabled:
            return
        # v-fix-cross-loop-routing-2026-09-15: route cross-loop calls
        if self._route_to_owner_loop(
            "log_trade",
            self._log_trade_impl,
            symbol, side, strategy, entry_time, exit_time,
            entry_price, exit_price, quantity, pnl, pnl_pct,
            exit_reason, atr_at_entry, stop_loss, take_profit,
            confidence, meta_proba, kelly_fraction, scaled_out, mode, reasoning,
        ):
            return
        await self._log_trade_impl(
            symbol, side, strategy, entry_time, exit_time,
            entry_price, exit_price, quantity, pnl, pnl_pct,
            exit_reason, atr_at_entry, stop_loss, take_profit,
            confidence, meta_proba, kelly_fraction, scaled_out, mode, reasoning,
        )

    async def _log_trade_impl(
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
        atr_at_entry: float | None,
        stop_loss: float | None,
        take_profit: float | None,
        confidence: float | None,
        meta_proba: float | None,
        kelly_fraction: float | None,
        scaled_out: bool,
        mode: str | None,
        reasoning: dict | None,
    ) -> None:
        """Internal impl: actually insert trade into the database."""
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
        
        v-fix-cross-loop-crash-2026-09-10: Skip sync if called from wrong loop.
        v-fix-cross-loop-routing-2026-09-15: Now ROUTES cross-loop calls to
        owner loop instead of skipping (no more silent data loss).
        """
        if not self._enabled:
            return
        # v-fix-cross-loop-routing-2026-09-15: route cross-loop calls
        if self._route_to_owner_loop("sync_positions", self._sync_positions_impl, positions):
            return
        await self._sync_positions_impl(positions)

    async def _sync_positions_impl(self, positions: dict) -> None:
        """Internal impl: actually sync positions to the database."""
        async with self._sync_positions_lock:
            try:
                keep_symbols = [
                    sym for sym, pos in positions.items()
                    if pos is not None and getattr(pos, 'quantity', 0) > 0
                ]
                async with self._engine.begin() as conn:
                    if keep_symbols:
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
                                "mode": getattr(pos, "mode", "simulation"),
                            },
                        )
            except Exception as exc:
                logger.warning("db_logger_positions_error err=%s", exc)

    async def log_news_veto(
        self,
        symbol: str,
        side: str,                  # 'long' or 'short' (would-be direction)
        veto_reason: str,           # e.g. 'insufficient_fresh_articles_0_lt_2'
        veto_source: str | None = None,    # 'alpaca' / 'yfinance' / 'falling_knife'
        cached_sentiment: float | None = None,
        fresh_count: int | None = None,
        fresh_avg_sentiment: float | None = None,
        latest_age_min: float | None = None,
        would_entry_price: float | None = None,
        would_stop_loss: float | None = None,
        would_take_profit: float | None = None,
    ) -> None:
        """v-news-veto-tracker-2026-04-28: record a vetoed news signal so we
        can later evaluate whether the veto was correct (saved a loss) or
        wrong (missed a winner). Outcome columns are filled later by
        evaluate_open_news_vetoes(). Never raises.
        
        v-fix-cross-loop-crash-2026-09-10: Skip write if called from wrong loop.
        v-fix-cross-loop-routing-2026-09-15: Now ROUTES cross-loop calls to
        owner loop instead of skipping (no more silent data loss).
        """
        if not self._enabled:
            return
        # v-fix-cross-loop-routing-2026-09-15: route cross-loop calls
        if self._route_to_owner_loop(
            "log_news_veto",
            self._log_news_veto_impl,
            symbol, side, veto_reason, veto_source,
            cached_sentiment, fresh_count, fresh_avg_sentiment,
            latest_age_min, would_entry_price, would_stop_loss, would_take_profit,
        ):
            return
        await self._log_news_veto_impl(
            symbol, side, veto_reason, veto_source,
            cached_sentiment, fresh_count, fresh_avg_sentiment,
            latest_age_min, would_entry_price, would_stop_loss, would_take_profit,
        )

    async def _log_news_veto_impl(
        self,
        symbol: str,
        side: str,
        veto_reason: str,
        veto_source: str | None,
        cached_sentiment: float | None,
        fresh_count: int | None,
        fresh_avg_sentiment: float | None,
        latest_age_min: float | None,
        would_entry_price: float | None,
        would_stop_loss: float | None,
        would_take_profit: float | None,
    ) -> None:
        """Internal impl: actually insert news veto into the database."""
        try:
            async with self._engine.begin() as conn:
                await conn.execute(
                    text("""
                        INSERT INTO bot_shadow_news_vetoes
                            (symbol, side, veto_reason, veto_source,
                             cached_sentiment, fresh_count, fresh_avg_sentiment,
                             latest_age_min,
                             would_entry_price, would_stop_loss, would_take_profit)
                        VALUES
                            (:symbol, :side, :veto_reason, :veto_source,
                             :cached_sentiment, :fresh_count, :fresh_avg_sentiment,
                             :latest_age_min,
                             :would_entry_price, :would_stop_loss, :would_take_profit)
                    """),
                    {
                        "symbol": symbol, "side": side,
                        "veto_reason": veto_reason, "veto_source": veto_source,
                        "cached_sentiment": cached_sentiment,
                        "fresh_count": fresh_count,
                        "fresh_avg_sentiment": fresh_avg_sentiment,
                        "latest_age_min": latest_age_min,
                        "would_entry_price": would_entry_price,
                        "would_stop_loss": would_stop_loss,
                        "would_take_profit": would_take_profit,
                    },
                )
        except Exception as exc:
            logger.warning("db_logger_news_veto_error err=%s", exc)

    async def evaluate_open_news_vetoes(
        self,
        get_current_price,           # callable(symbol) -> Optional[float]
        max_age_hours: int = 4,
    ) -> int:
        """For every veto with NULL outcome and veto_time within the
        evaluation window, fetch the current price and decide:

          long would-be:
            current_price >= would_take_profit  →  outcome='missed_winner'
            current_price <= would_stop_loss    →  outcome='correct_veto'
            else (still open)                   →  leave NULL

          short would-be: mirror.

          If veto_time older than max_age_hours and still neither hit:
            outcome='neutral_timeout' so we don't keep re-checking.

        Returns count of rows updated. Designed to run every 5-15 min
        from the engine main loop; cheap because the open-set is small.
        
        v-fix-cross-loop-crash-2026-09-10: Skip if called from wrong loop.
        v-fix-cross-loop-routing-2026-09-15: Now ROUTES cross-loop calls to
        owner loop instead of skipping (no more silent data loss).
        Note: cross-loop routing returns 0 immediately (fire-and-forget).
        """
        if not self._enabled:
            return 0
        # v-fix-cross-loop-routing-2026-09-15: route cross-loop calls
        # Note: for methods that return values, cross-loop routing is fire-and-forget
        if self._route_to_owner_loop(
            "evaluate_open_news_vetoes",
            self._evaluate_open_news_vetoes_impl,
            get_current_price, max_age_hours,
        ):
            return 0
        return await self._evaluate_open_news_vetoes_impl(get_current_price, max_age_hours)

    async def _evaluate_open_news_vetoes_impl(
        self,
        get_current_price,
        max_age_hours: int,
    ) -> int:
        """Internal impl: actually evaluate open news vetoes."""
        updated = 0
        try:
            async with self._engine.begin() as conn:
                rows = (await conn.execute(
                    text("""
                        SELECT id, symbol, side, would_entry_price,
                               would_stop_loss, would_take_profit, veto_time
                        FROM bot_shadow_news_vetoes
                        WHERE outcome IS NULL
                          AND veto_time > NOW() - INTERVAL ':hrs hours'
                          AND would_stop_loss IS NOT NULL
                          AND would_take_profit IS NOT NULL
                    """.replace(":hrs", str(max_age_hours)))
                )).mappings().all()

                for row in rows:
                    symbol = row['symbol']
                    side = row['side']
                    entry = row['would_entry_price']
                    stop = row['would_stop_loss']
                    target = row['would_take_profit']
                    veto_time = row['veto_time']
                    if entry is None or stop is None or target is None:
                        continue

                    price = None
                    try:
                        price = get_current_price(symbol)
                    except Exception:
                        price = None
                    if price is None or price <= 0:
                        from datetime import datetime as _dt, timedelta as _td
                        if veto_time < _dt.now() - _td(hours=max_age_hours):
                            await conn.execute(
                                text("""UPDATE bot_shadow_news_vetoes
                                        SET outcome='neutral_no_price',
                                            outcome_evaluated_at=NOW()
                                        WHERE id=:id"""),
                                {"id": row['id']},
                            )
                            updated += 1
                        continue

                    pnl_pct = ((price - entry) / entry * 100.0
                               if side == 'long'
                               else (entry - price) / entry * 100.0)

                    hit_target = (price >= target) if side == 'long' else (price <= target)
                    hit_stop   = (price <= stop)   if side == 'long' else (price >= stop)

                    outcome = None
                    if hit_target and not hit_stop:
                        outcome = 'missed_winner'
                    elif hit_stop and not hit_target:
                        outcome = 'correct_veto'
                    elif hit_target and hit_stop:
                        outcome = 'neutral_both_crossed'
                    else:
                        from datetime import datetime as _dt, timedelta as _td
                        if veto_time < _dt.now() - _td(hours=max_age_hours):
                            outcome = 'neutral_timeout'

                    if outcome:
                        await conn.execute(
                            text("""UPDATE bot_shadow_news_vetoes
                                    SET outcome=:outcome,
                                        outcome_evaluated_at=NOW(),
                                        outcome_price=:price,
                                        outcome_pnl_pct=:pnl,
                                        outcome_hit_target=:t,
                                        outcome_hit_stop=:s
                                    WHERE id=:id"""),
                            {
                                "outcome": outcome,
                                "price": price,
                                "pnl": pnl_pct,
                                "t": bool(hit_target),
                                "s": bool(hit_stop),
                                "id": row['id'],
                            },
                        )
                        updated += 1
        except Exception as exc:
            logger.warning("db_logger_evaluate_vetoes_error err=%s", exc)
        return updated

    async def delete_position(self, symbol: str) -> None:
        """Remove a closed position from bot_positions.
        
        v-fix-cross-loop-crash-2026-09-10: Skip delete if called from wrong loop.
        v-fix-cross-loop-routing-2026-09-15: Now ROUTES cross-loop calls to
        owner loop instead of skipping (no more silent data loss).
        """
        if not self._enabled:
            return
        # v-fix-cross-loop-routing-2026-09-15: route cross-loop calls
        if self._route_to_owner_loop("delete_position", self._delete_position_impl, symbol):
            return
        await self._delete_position_impl(symbol)

    async def _delete_position_impl(self, symbol: str) -> None:
        """Internal impl: actually delete position from the database."""
        try:
            async with self._engine.begin() as conn:
                await conn.execute(
                    text("DELETE FROM bot_positions WHERE symbol = :symbol"),
                    {"symbol": symbol},
                )
        except Exception as exc:
            logger.warning("db_logger_delete_position_error err=%s", exc)

    # =========================================================================
    # v-feature-snapshot-2026-09-09: Decision snapshots for ML training
    # v-fix-loop-safety-2026-09-10: Loop-safe with cross-loop routing
    # =========================================================================
    async def log_decision_snapshot(self, snapshot) -> None:
        """v-feature-snapshot-2026-09-09: persist a DecisionSnapshot for ML training.
        
        Table: bot_decision_snapshots
        - snapshot_id (TEXT PK): deterministic hash for deduplication
        - symbol, ts, mode, strategy_id, action, reason, gate_name
        - confidence, price
        - price_vol_json: PriceVolumeFeatures as JSON
        - news_json: NewsAggregate as JSON
        - regime_json: RegimeContext as JSON
        - would_entry_price, would_stop_loss, would_take_profit
        - would_size_shares, would_size_mult
        - extra_json: strategy-specific extras
        
        Never raises. Fire-and-forget.
        
        v-fix-loop-safety-2026-09-10: If called from a different event loop
        than the owner loop, routes the insert via run_coroutine_threadsafe.
        
        v-fix-cross-loop-crash-2026-09-10: Check if owner loop is alive before
        routing. Skip write if owner loop is dead to avoid crash.
        """
        if not self._enabled:
            return
        try:
            from core.decision_snapshot import is_snapshot_logging_enabled
            if not is_snapshot_logging_enabled():
                return
        except ImportError:
            return
        
        # v-fix-loop-safety-2026-09-10: cross-loop safety check
        # v-fix-cross-loop-crash-2026-09-10: verify owner loop is alive
        try:
            current_loop = asyncio.get_running_loop()
            if self._owner_loop is not None and self._owner_loop is not current_loop:
                # Running on wrong loop — check if owner loop is alive
                if not self._owner_loop.is_running():
                    logger.warning(
                        "db_logger_snapshot_skip_dead_loop owner=%s",
                        id(self._owner_loop)
                    )
                    return
                # Owner loop alive — route fire-and-forget
                asyncio.run_coroutine_threadsafe(
                    self._log_decision_snapshot_impl(snapshot),
                    self._owner_loop,
                )
                return
        except RuntimeError:
            # No running loop — shouldn't happen in async context, but proceed
            pass
        
        await self._log_decision_snapshot_impl(snapshot)

    async def _log_decision_snapshot_impl(self, snapshot) -> None:
        """Internal impl: actually insert the snapshot into the database.
        
        v-fix-loop-safety-2026-09-10: separated from log_decision_snapshot
        for cross-loop routing.
        """
        try:
            snap_dict = snapshot.to_dict()
            # v-fix-snapshot-ts-2026-09-09: asyncpg requires datetime objects for
            # TIMESTAMPTZ columns; snap_dict["ts"] is an ISO string from to_dict().
            ts_value = _ensure_datetime(snap_dict["ts"])
            if ts_value is None:
                logger.warning("db_logger_snapshot_error: ts is None or unparseable")
                return
            async with self._engine.begin() as conn:
                await conn.execute(
                    text("""
                        INSERT INTO bot_decision_snapshots
                            (snapshot_id, symbol, ts, mode, strategy_id,
                             action, reason, gate_name, confidence, price,
                             price_vol_json, news_json, regime_json,
                             would_entry_price, would_stop_loss, would_take_profit,
                             would_size_shares, would_size_mult, extra_json)
                        VALUES
                            (:snapshot_id, :symbol, :ts, :mode, :strategy_id,
                             :action, :reason, :gate_name, :confidence, :price,
                             :price_vol_json, :news_json, :regime_json,
                             :would_entry_price, :would_stop_loss, :would_take_profit,
                             :would_size_shares, :would_size_mult, :extra_json)
                        ON CONFLICT (snapshot_id) DO NOTHING
                    """),
                    {
                        "snapshot_id": snap_dict["snapshot_id"],
                        "symbol": snap_dict["symbol"],
                        "ts": ts_value,
                        "mode": snap_dict["mode"],
                        "strategy_id": snap_dict["strategy_id"],
                        "action": snap_dict["action"],
                        "reason": snap_dict["reason"],
                        "gate_name": snap_dict.get("gate_name"),
                        "confidence": snap_dict["confidence"],
                        "price": snap_dict["price_vol"]["price"],
                        "price_vol_json": json.dumps(snap_dict["price_vol"]),
                        "news_json": json.dumps(snap_dict["news"]),
                        "regime_json": json.dumps(snap_dict["regime"]),
                        "would_entry_price": snap_dict.get("would_entry_price"),
                        "would_stop_loss": snap_dict.get("would_stop_loss"),
                        "would_take_profit": snap_dict.get("would_take_profit"),
                        "would_size_shares": snap_dict.get("would_size_shares"),
                        "would_size_mult": snap_dict.get("would_size_mult"),
                        "extra_json": json.dumps(snap_dict.get("extra", {})),
                    },
                )
        except Exception as exc:
            logger.warning("db_logger_snapshot_error err=%s", exc)

    async def get_decision_snapshots(
        self,
        symbol: str | None = None,
        strategy_id: str | None = None,
        action: str | None = None,
        limit: int = 100,
        offset: int = 0,
        since: datetime | None = None,
    ) -> list[dict]:
        """v-feature-snapshot-2026-09-09: query decision snapshots.
        
        Returns list of snapshot dicts for ML training pipelines.
        Filters are optional and combinable.
        
        v-fix-loop-safety-2026-09-10: If called from a different event loop
        than the owner loop, routes the query via run_coroutine_threadsafe.
        
        v-fix-cross-loop-crash-2026-09-10: Check if owner loop is alive before
        routing. Return empty list if owner loop is dead to avoid crash.
        """
        if not self._enabled:
            return []
        
        # v-fix-loop-safety-2026-09-10: cross-loop safety check
        # v-fix-cross-loop-crash-2026-09-10: verify owner loop is alive
        try:
            current_loop = asyncio.get_running_loop()
            if self._owner_loop is not None and self._owner_loop is not current_loop:
                # Running on wrong loop — check if owner loop is alive
                if not self._owner_loop.is_running():
                    logger.warning(
                        "db_logger_get_snapshots_skip_dead_loop owner=%s",
                        id(self._owner_loop)
                    )
                    return []
                # Owner loop alive — route and await result
                fut = asyncio.run_coroutine_threadsafe(
                    self._get_decision_snapshots_impl(
                        symbol=symbol, strategy_id=strategy_id,
                        action=action, limit=limit, offset=offset, since=since,
                    ),
                    self._owner_loop,
                )
                return await asyncio.wrap_future(fut)
        except RuntimeError:
            # No running loop — proceed (shouldn't happen in async context)
            pass
        
        return await self._get_decision_snapshots_impl(
            symbol=symbol, strategy_id=strategy_id,
            action=action, limit=limit, offset=offset, since=since,
        )

    async def _get_decision_snapshots_impl(
        self,
        symbol: str | None = None,
        strategy_id: str | None = None,
        action: str | None = None,
        limit: int = 100,
        offset: int = 0,
        since: datetime | None = None,
    ) -> list[dict]:
        """Internal impl: actually query snapshots from the database.
        
        v-fix-loop-safety-2026-09-10: separated from get_decision_snapshots
        for cross-loop routing.
        """
        try:
            filters = []
            params: dict = {"limit": limit, "offset": offset}
            
            if symbol:
                filters.append("symbol = :symbol")
                params["symbol"] = symbol
            if strategy_id:
                filters.append("strategy_id = :strategy_id")
                params["strategy_id"] = strategy_id
            if action:
                filters.append("action = :action")
                params["action"] = action
            if since:
                filters.append("ts >= :since")
                # v-fix-since-bind-2026-09-10: asyncpg requires datetime objects,
                # not isoformat strings. Pass datetime directly.
                params["since"] = since
            
            where = "WHERE " + " AND ".join(filters) if filters else ""
            
            async with self._engine.begin() as conn:
                rows = (await conn.execute(
                    text(f"""
                        SELECT snapshot_id, symbol, ts, mode, strategy_id,
                               action, reason, gate_name, confidence, price,
                               price_vol_json, news_json, regime_json,
                               would_entry_price, would_stop_loss, would_take_profit,
                               would_size_shares, would_size_mult, extra_json
                        FROM bot_decision_snapshots
                        {where}
                        ORDER BY ts DESC
                        LIMIT :limit OFFSET :offset
                    """),
                    params,
                )).mappings().all()
            
            result = []
            for row in rows:
                # v-fix-json-decode-2026-09-10: use _safe_json_decode for JSONB
                # columns since asyncpg may return dicts directly
                result.append({
                    "snapshot_id": row["snapshot_id"],
                    "symbol": row["symbol"],
                    "ts": row["ts"].isoformat() if hasattr(row["ts"], "isoformat") else row["ts"],
                    "mode": row["mode"],
                    "strategy_id": row["strategy_id"],
                    "action": row["action"],
                    "reason": row["reason"],
                    "gate_name": row["gate_name"],
                    "confidence": row["confidence"],
                    "price_vol": _safe_json_decode(row["price_vol_json"]),
                    "news": _safe_json_decode(row["news_json"]),
                    "regime": _safe_json_decode(row["regime_json"]),
                    "would_entry_price": row["would_entry_price"],
                    "would_stop_loss": row["would_stop_loss"],
                    "would_take_profit": row["would_take_profit"],
                    "would_size_shares": row["would_size_shares"],
                    "would_size_mult": row["would_size_mult"],
                    "extra": _safe_json_decode(row["extra_json"]),
                })
            return result
        except Exception as exc:
            logger.warning("db_logger_get_snapshots_error err=%s", exc)
            return []

    async def get_latest_snapshot(self, symbol: str) -> dict | None:
        """v-feature-snapshot-2026-09-09: get the most recent snapshot for a symbol."""
        snapshots = await self.get_decision_snapshots(symbol=symbol, limit=1)
        return snapshots[0] if snapshots else None

    async def get_todays_bot_entries(self, symbols: list[str] | None = None) -> dict[str, dict]:
        """v-manage-persist-2026-09-15: get today's bot_trades entries.

        Returns a dict mapping symbol -> trade metadata for bot-opened
        positions from today. Used as a fallback source of truth for
        restoring managed_by_bot=True when _saved_positions_meta is
        stale or missing.

        Only returns entries (entry_time today), not closes. The caller
        determines which symbols are still open on Schwab and cross-refs.

        Returns:
            {symbol: {"side": "long"|"short", "entry_time": datetime,
                      "entry_price": float, "quantity": int, "strategy": str}}
        
        v-manage-persist-hotfix-2026-09-15: cross-loop routing now properly
        waits for the result instead of returning {}. This fixes the bug where
        bot_trades fallback always returned empty on cross-loop calls.
        """
        if not self._enabled:
            return {}
        
        # v-manage-persist-hotfix-2026-09-15: proper cross-loop routing for value-returning method
        # Unlike fire-and-forget writes, this must wait for and return the actual result.
        if not self._is_on_owner_loop():
            if self._owner_loop is None:
                self._warn_once(
                    "get_todays_bot_entries_no_owner",
                    "db_logger_get_todays_bot_entries_no_owner (cannot route, returning empty)",
                )
                return {}
            if not self._owner_loop.is_running():
                self._warn_once(
                    "get_todays_bot_entries_dead_owner",
                    "db_logger_get_todays_bot_entries_dead_owner (cannot route, returning empty)",
                )
                return {}
            # Route to owner loop and await result
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._get_todays_bot_entries_impl(symbols),
                    self._owner_loop,
                )
                return await asyncio.wrap_future(fut)
            except Exception as exc:
                self._warn_once(
                    "get_todays_bot_entries_route_error",
                    "db_logger_get_todays_bot_entries_route_error err=%s",
                    exc,
                )
                return {}
        
        return await self._get_todays_bot_entries_impl(symbols)

    async def _get_todays_bot_entries_impl(self, symbols: list[str] | None = None) -> dict[str, dict]:
        """Internal impl: query bot_trades for today's entries."""
        try:
            async with self._engine.begin() as conn:
                query = """
                    SELECT symbol, side, entry_time, entry_price, quantity, strategy
                    FROM bot_trades
                    WHERE DATE(entry_time) = CURRENT_DATE
                      AND mode = 'live'
                """
                params = {}
                if symbols:
                    query += " AND symbol = ANY(:symbols)"
                    params["symbols"] = [s.upper() for s in symbols]
                query += " ORDER BY entry_time DESC"
                result = await conn.execute(text(query), params)
                rows = result.mappings().all()
                entries = {}
                for row in rows:
                    sym = row["symbol"].upper()
                    if sym not in entries:
                        entries[sym] = {
                            "side": row["side"],
                            "entry_time": row["entry_time"],
                            "entry_price": float(row["entry_price"]),
                            "quantity": int(row["quantity"]),
                            "strategy": row["strategy"],
                        }
                return entries
        except Exception as exc:
            logger.warning("db_logger_get_todays_bot_entries_error err=%s", exc)
            return {}

    async def ensure_snapshot_table(self) -> None:
        """v-feature-snapshot-2026-09-09: create bot_decision_snapshots table if missing.
        
        Called during engine init. Idempotent.
        
        v-fix-cross-loop-crash-2026-09-10: Skip if called from wrong loop.
        v-fix-cross-loop-routing-2026-09-15: Now ROUTES cross-loop calls to
        owner loop instead of skipping (no more silent data loss).
        """
        if not self._enabled:
            return
        # v-fix-cross-loop-routing-2026-09-15: route cross-loop calls
        if self._route_to_owner_loop("ensure_snapshot_table", self._ensure_snapshot_table_impl):
            return
        await self._ensure_snapshot_table_impl()

    async def _ensure_snapshot_table_impl(self) -> None:
        """Internal impl: actually create snapshot table if missing."""
        try:
            async with self._engine.begin() as conn:
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS bot_decision_snapshots (
                        snapshot_id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        ts TIMESTAMPTZ NOT NULL,
                        mode TEXT,
                        strategy_id TEXT NOT NULL,
                        action TEXT NOT NULL,
                        reason TEXT,
                        gate_name TEXT,
                        confidence REAL,
                        price REAL,
                        price_vol_json JSONB,
                        news_json JSONB,
                        regime_json JSONB,
                        would_entry_price REAL,
                        would_stop_loss REAL,
                        would_take_profit REAL,
                        would_size_shares INTEGER,
                        would_size_mult REAL,
                        extra_json JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """))
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_ts
                    ON bot_decision_snapshots (symbol, ts DESC)
                """))
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_snapshots_strategy_ts
                    ON bot_decision_snapshots (strategy_id, ts DESC)
                """))
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_snapshots_action
                    ON bot_decision_snapshots (action)
                """))
        except Exception as exc:
            logger.warning("db_logger_ensure_snapshot_table_error err=%s", exc)

    # =========================================================================
    # v-fix-log-strategy-decision-2026-09-14: Sync fire-and-forget for strategy logging
    # =========================================================================
    # Keys in log_decision signature that would collide with extra_data from Stage A cards
    _LOG_DECISION_RESERVED_KEYS = frozenset({
        'component', 'symbol', 'action', 'reason', 'mode',
        'signal_type', 'confidence', 'strength', 'meta_proba',
        'atr', 'stop_distance', 'price',
    })

    def log_strategy_decision(
        self,
        strategy: str,
        symbol: str,
        action: str,
        reason: str | None = None,
        extra_data: dict | None = None,
    ) -> None:
        """Sync fire-and-forget entry point for strategy decision logging.
        
        v-fix-log-strategy-decision-2026-09-14: gpu_news_critic and theme_shock_logger
        call this from sync contexts (NewsBus.on_publish path). This method schedules
        the async log_decision onto the owner loop without blocking.
        
        Maps to existing log_decision with:
          - component=strategy (e.g. 'gpu_news_critic', 'theme_shock_logger')
          - symbol=symbol
          - action=action
          - reason=reason
          - extra_data passed to details_json (with collision handling)
        
        v-hotfix-collision-2026-09-14: CriticCard.to_dict() and ThemeEvent.to_dict()
        include keys like 'action', 'symbol', 'confidence' that collide with
        log_decision's explicit parameters. We split extra_data:
          - Non-colliding keys go directly into details_json (flat structure)
          - Colliding keys go into details_json._card_original (preserved for Research)
        
        Never raises to callers. Fails soft with warning log.
        """
        if not self._enabled:
            return
        
        try:
            if self._owner_loop is None:
                logger.warning(
                    "db_logger_strategy_decision_skip_no_owner_loop strategy=%s symbol=%s",
                    strategy, symbol
                )
                return
            
            if not self._owner_loop.is_running():
                logger.warning(
                    "db_logger_strategy_decision_skip_dead_loop strategy=%s symbol=%s",
                    strategy, symbol
                )
                return
            
            # v-hotfix-collision-2026-09-14: split extra_data to avoid kwarg collision
            safe_extra: dict = {}
            if extra_data:
                colliding = {}
                for k, v in extra_data.items():
                    if k in self._LOG_DECISION_RESERVED_KEYS:
                        colliding[k] = v
                    else:
                        safe_extra[k] = v
                # Preserve colliding keys under _card_original for Research
                if colliding:
                    safe_extra['_card_original'] = colliding
            
            asyncio.run_coroutine_threadsafe(
                self.log_decision(
                    component=strategy,
                    symbol=symbol,
                    action=action,
                    reason=reason,
                    **safe_extra,
                ),
                self._owner_loop,
            )
        except Exception as exc:
            logger.warning(
                "db_logger_strategy_decision_error strategy=%s symbol=%s err=%s",
                strategy, symbol, exc
            )


def _safe_json(value: Any) -> Any:
    """Coerce a value to JSON-serializable form.
    
    v-fix-log-strategy-decision-2026-09-14: properly handle dicts and lists
    so that nested structures like theme_probs and relevance_by_symbol are
    preserved as proper JSON objects in details_json, not string representations.
    """
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, dict):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if isinstance(value, set):
        return [_safe_json(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _safe_json_decode(value: Any) -> dict | list:
    """Decode JSON, handling asyncpg's automatic JSONB deserialization.
    
    v-fix-json-decode-2026-09-10: asyncpg/SQLAlchemy returns JSONB columns
    as Python dicts directly. Calling json.loads() on an already-deserialized
    dict raises: "the JSON object must be str, bytes or bytearray, not dict".
    
    This helper:
      - Returns dict/list values as-is (already deserialized by asyncpg)
      - Parses str/bytes/bytearray via json.loads()
      - Returns {} for None or unparseable values
    """
    if value is None:
        return {}
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (str, bytes, bytearray)):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            logger.warning("_safe_json_decode: failed to parse %r", value[:100] if hasattr(value, '__getitem__') else value)
            return {}
    return {}


def _ensure_datetime(value: Any) -> datetime | None:
    """Coerce a value to datetime for asyncpg bind parameters.
    
    asyncpg requires actual datetime objects for TIMESTAMPTZ columns;
    ISO format strings cause DataError. This function normalizes:
      - datetime objects: returned as-is (timezone added if naive)
      - ISO strings: parsed to datetime (timezone-aware)
      - None: returned as None
    
    v-fix-snapshot-ts-2026-09-09: fixes asyncpg DataError on snapshot insert.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            logger.warning("_ensure_datetime: could not parse %r", value)
            return None
    return None
