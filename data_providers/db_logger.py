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
from datetime import datetime, timezone
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
        evaluate_open_news_vetoes(). Never raises."""
        if not self._enabled:
            return
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
        """
        if not self._enabled:
            return 0
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
                        # Time out positions older than max_age_hours that
                        # we never managed to price — mark neutral so we
                        # don't loop on them forever.
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
                        outcome = 'missed_winner'    # we should have entered
                    elif hit_stop and not hit_target:
                        outcome = 'correct_veto'     # we saved a loss
                    elif hit_target and hit_stop:
                        # Both crossed inside the same window — ambiguous,
                        # call it neutral and inspect manually.
                        outcome = 'neutral_both_crossed'
                    else:
                        # Still open inside window. Only resolve if past
                        # max_age_hours; otherwise wait.
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

    # =========================================================================
    # v-feature-snapshot-2026-09-09: Decision snapshots for ML training
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
        """
        if not self._enabled:
            return
        try:
            from core.decision_snapshot import is_snapshot_logging_enabled
            if not is_snapshot_logging_enabled():
                return
        except ImportError:
            return
        
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
        """
        if not self._enabled:
            return []
        
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
                params["since"] = since.isoformat()
            
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
                    "price_vol": json.loads(row["price_vol_json"]) if row["price_vol_json"] else {},
                    "news": json.loads(row["news_json"]) if row["news_json"] else {},
                    "regime": json.loads(row["regime_json"]) if row["regime_json"] else {},
                    "would_entry_price": row["would_entry_price"],
                    "would_stop_loss": row["would_stop_loss"],
                    "would_take_profit": row["would_take_profit"],
                    "would_size_shares": row["would_size_shares"],
                    "would_size_mult": row["would_size_mult"],
                    "extra": json.loads(row["extra_json"]) if row["extra_json"] else {},
                })
            return result
        except Exception as exc:
            logger.warning("db_logger_get_snapshots_error err=%s", exc)
            return []

    async def get_latest_snapshot(self, symbol: str) -> dict | None:
        """v-feature-snapshot-2026-09-09: get the most recent snapshot for a symbol."""
        snapshots = await self.get_decision_snapshots(symbol=symbol, limit=1)
        return snapshots[0] if snapshots else None

    async def ensure_snapshot_table(self) -> None:
        """v-feature-snapshot-2026-09-09: create bot_decision_snapshots table if missing.
        
        Called during engine init. Idempotent.
        """
        if not self._enabled:
            return
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


def _safe_json(value: Any) -> Any:
    """Coerce a value to JSON-serializable form."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, set):
        return list(value)
    return str(value)


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
