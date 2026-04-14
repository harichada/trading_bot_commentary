"""Alpaca 1-minute bar ingestion into the rudra_dev Postgres minute_bars table.

Design goals (production-grade):
  * Async HTTP with bounded concurrency (asyncio.Semaphore)
  * Idempotent upserts: INSERT ... ON CONFLICT (symbol, ts) DO NOTHING
  * Retries with exponential backoff on 429 / 5xx / network errors
  * Checkpointed resume: progress written after each (batch, window) pair
  * Timezone-correct: Alpaca emits UTC; minute_bars uses naive
    America/New_York -- all writes are converted
  * Structured logging: every batch emits one audit line
  * No secrets in logs, no shell interpolation anywhere
  * Typed public API + pydantic-free dataclasses for portability

Public entry point:
    python -m data_providers.alpaca_ingest ...
or:
    from data_providers.alpaca_ingest import IngestJob, IngestConfig
    await IngestJob(cfg).run(symbols, start, end)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

import httpx
import psycopg2
from psycopg2.extras import execute_values


__all__ = ["IngestConfig", "IngestJob", "AlpacaBarClient", "PostgresBarSink"]


_ET = ZoneInfo("America/New_York")
_logger = logging.getLogger("alpaca_ingest")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IngestConfig:
    """Runtime configuration. Construct once, pass to IngestJob."""

    # Postgres
    dsn: str
    table: str = "minute_bars"

    # Alpaca
    api_key: str = ""
    api_secret: str = ""
    feed: str = "sip"  # 'sip' (all exchanges) or 'iex' (free tier)
    base_url: str = "https://data.alpaca.markets/v2"

    # Tuning
    symbols_per_request: int = 50    # Alpaca caps at 200; 50 balances memory
    max_bars_per_page: int = 10_000  # Alpaca hard cap
    concurrent_requests: int = 4     # safe for both free (200/min) and paid
    request_timeout_s: float = 30.0
    max_retries: int = 5
    backoff_base_s: float = 1.0
    backoff_cap_s: float = 30.0

    # Checkpoint file (idempotent resume)
    checkpoint_path: Path = Path("alpaca_ingest_checkpoint.json")

    @classmethod
    def from_env(cls, **overrides: Any) -> "IngestConfig":
        env_map = {
            "dsn": os.environ.get(
                "POSTGRES_DSN",
                "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev",
            ),
            "api_key": os.environ.get("ALPACA_API_KEY", ""),
            "api_secret": os.environ.get("ALPACA_SECRET_KEY", ""),
            "feed": os.environ.get("ALPACA_DATA_FEED", "sip"),
        }
        env_map.update(overrides)
        cfg = cls(**env_map)
        if not (cfg.api_key and cfg.api_secret):
            raise RuntimeError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in the "
                "environment (see .env)."
            )
        return cfg


# ---------------------------------------------------------------------------
# Alpaca client
# ---------------------------------------------------------------------------

@dataclass
class _BarBatch:
    """One page of bars returned by Alpaca for one or more symbols."""
    bars: dict[str, list[dict[str, Any]]]
    next_page_token: Optional[str]


class AlpacaBarClient:
    """Thin async wrapper around Alpaca's /v2/stocks/bars endpoint.

    Handles retry + rate limiting. Does not transform data — callers are
    responsible for any projection / timezone conversion. This keeps the
    HTTP layer easily mockable for tests.
    """

    def __init__(self, cfg: IngestConfig):
        self._cfg = cfg
        self._sem = asyncio.Semaphore(cfg.concurrent_requests)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "AlpacaBarClient":
        self._client = httpx.AsyncClient(
            base_url=self._cfg.base_url,
            headers={
                "APCA-API-KEY-ID": self._cfg.api_key,
                "APCA-API-SECRET-KEY": self._cfg.api_secret,
                "User-Agent": "trading_bot_commentary/alpaca_ingest",
            },
            timeout=self._cfg.request_timeout_s,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_bars(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        timeframe: str = "1Min",
        page_token: Optional[str] = None,
    ) -> _BarBatch:
        """Single paginated call. Times must be timezone-aware UTC."""
        assert self._client is not None, "Use 'async with AlpacaBarClient(cfg)'"
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be tz-aware UTC datetimes")

        params: dict[str, Any] = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "end": end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "feed": self._cfg.feed,
            "limit": self._cfg.max_bars_per_page,
            "adjustment": "raw",  # use 'raw' to match unadjusted stream data
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token

        async with self._sem:
            for attempt in range(self._cfg.max_retries):
                try:
                    resp = await self._client.get("/stocks/bars", params=params)
                except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                    self._log_retry(attempt, f"network_error:{type(e).__name__}")
                    await self._backoff(attempt)
                    continue

                if resp.status_code == 200:
                    data = resp.json()
                    return _BarBatch(
                        bars=data.get("bars") or {},
                        next_page_token=data.get("next_page_token"),
                    )

                # Rate limited -> honour Retry-After when present
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("retry-after", "1"))
                    self._log_retry(attempt, "rate_limited", retry_after=retry_after)
                    await asyncio.sleep(min(retry_after, self._cfg.backoff_cap_s))
                    continue

                # 5xx -> retry with backoff
                if 500 <= resp.status_code < 600:
                    self._log_retry(attempt, f"server_error:{resp.status_code}")
                    await self._backoff(attempt)
                    continue

                # Anything else is a hard error (4xx other than 429)
                raise RuntimeError(
                    f"Alpaca API {resp.status_code}: {resp.text[:400]}"
                )

            raise RuntimeError(
                f"Exhausted {self._cfg.max_retries} retries fetching "
                f"{len(symbols)} symbols {start.date()}..{end.date()}"
            )

    async def _backoff(self, attempt: int) -> None:
        delay = min(
            self._cfg.backoff_base_s * (2 ** attempt),
            self._cfg.backoff_cap_s,
        )
        await asyncio.sleep(delay)

    @staticmethod
    def _log_retry(attempt: int, reason: str, **fields: Any) -> None:
        _logger.warning(
            "alpaca_retry attempt=%d reason=%s %s",
            attempt + 1, reason,
            " ".join(f"{k}={v}" for k, v in fields.items()),
        )


# ---------------------------------------------------------------------------
# Postgres sink
# ---------------------------------------------------------------------------

class PostgresBarSink:
    """Idempotent bulk inserter for minute_bars.

    Uses psycopg2.execute_values for efficient multi-row INSERT and
    ON CONFLICT DO NOTHING so reruns are safe.
    """

    def __init__(self, cfg: IngestConfig):
        self._cfg = cfg
        self._conn: Optional[psycopg2.extensions.connection] = None

    def __enter__(self) -> "PostgresBarSink":
        self._conn = psycopg2.connect(self._cfg.dsn)
        return self

    def __exit__(self, *_: Any) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def upsert_bars(
        self,
        symbol: str,
        bars: Iterable[dict[str, Any]],
    ) -> int:
        """Convert Alpaca UTC bars to ET-naive rows and insert."""
        assert self._conn is not None, "Use 'with PostgresBarSink(cfg)'"

        rows: list[tuple] = []
        for b in bars:
            # Alpaca timestamp is ISO8601 with 'Z' suffix (UTC)
            ts_utc = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
            ts_et = ts_utc.astimezone(_ET).replace(tzinfo=None)
            rows.append((
                symbol,
                ts_et,
                float(b.get("o", 0.0)),
                float(b.get("h", 0.0)),
                float(b.get("l", 0.0)),
                float(b.get("c", 0.0)),
                int(b.get("v", 0)),
            ))
        if not rows:
            return 0

        sql = (
            f"INSERT INTO {self._cfg.table} "
            "(symbol, ts, open, high, low, close, volume) VALUES %s "
            "ON CONFLICT (symbol, ts) DO NOTHING"
        )
        with self._conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=1000)
            inserted = cur.rowcount
        self._conn.commit()
        return inserted


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

@dataclass
class _Checkpoint:
    """Persisted per-batch progress. Keyed by (batch_index, window_start)."""
    completed: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "_Checkpoint":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(completed=set(data.get("completed", [])))
        except (json.JSONDecodeError, OSError) as e:
            _logger.warning("checkpoint_load_failed path=%s err=%s", path, e)
            return cls()

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({"completed": sorted(self.completed)}))
        tmp.replace(path)

    def mark(self, key: str) -> None:
        self.completed.add(key)

    def has(self, key: str) -> bool:
        return key in self.completed


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class IngestJob:
    """High-level backfill orchestrator."""

    def __init__(self, cfg: IngestConfig):
        self._cfg = cfg

    async def run(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> dict[str, Any]:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("start and end must be tz-aware datetimes")

        checkpoint = _Checkpoint.load(self._cfg.checkpoint_path)
        batches = _batched(symbols, self._cfg.symbols_per_request)
        total_batches = len(batches)

        _logger.info(
            "alpaca_ingest_start symbols=%d batches=%d start=%s end=%s feed=%s",
            len(symbols), total_batches,
            start.isoformat(), end.isoformat(), self._cfg.feed,
        )

        stats = {"requests": 0, "rows_inserted": 0, "batches_skipped": 0,
                 "symbols_empty": 0}
        t0 = datetime.now(timezone.utc)

        async with AlpacaBarClient(self._cfg) as client:
            with PostgresBarSink(self._cfg) as sink:
                for batch_idx, batch in enumerate(batches):
                    key = f"{batch_idx}:{start.date()}:{end.date()}"
                    if checkpoint.has(key):
                        stats["batches_skipped"] += 1
                        continue

                    await self._ingest_batch(
                        client, sink, batch, start, end, stats,
                    )
                    checkpoint.mark(key)
                    checkpoint.save(self._cfg.checkpoint_path)

                    _logger.info(
                        "alpaca_ingest_batch idx=%d/%d size=%d "
                        "cumulative_rows=%d cumulative_requests=%d",
                        batch_idx + 1, total_batches, len(batch),
                        stats["rows_inserted"], stats["requests"],
                    )

        elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
        _logger.info(
            "alpaca_ingest_done elapsed_s=%.1f rows=%d requests=%d "
            "skipped=%d symbols_empty=%d",
            elapsed, stats["rows_inserted"], stats["requests"],
            stats["batches_skipped"], stats["symbols_empty"],
        )
        return {**stats, "elapsed_s": elapsed}

    async def _ingest_batch(
        self,
        client: AlpacaBarClient,
        sink: PostgresBarSink,
        symbols: list[str],
        start: datetime,
        end: datetime,
        stats: dict[str, Any],
    ) -> None:
        page_token: Optional[str] = None
        while True:
            batch = await client.fetch_bars(symbols, start, end, page_token=page_token)
            stats["requests"] += 1

            for symbol, bars in batch.bars.items():
                if not bars:
                    stats["symbols_empty"] += 1
                    continue
                inserted = sink.upsert_bars(symbol, bars)
                stats["rows_inserted"] += inserted

            if not batch.next_page_token:
                break
            page_token = batch.next_page_token


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _batched(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def top_symbols_by_volume(dsn: str, n: int, days: int = 30) -> list[str]:
    """Pick the N most-traded symbols over the last `days` from minute_bars."""
    import sqlalchemy  # imported lazily so this module works without SA
    engine = sqlalchemy.create_engine(dsn)
    with engine.connect() as conn:
        rows = conn.execute(
            sqlalchemy.text(
                "SELECT symbol, SUM(volume) AS v "
                "FROM minute_bars "
                "WHERE ts >= NOW() - (:days || ' days')::interval "
                "GROUP BY symbol ORDER BY v DESC LIMIT :n"
            ),
            {"days": days, "n": n},
        ).fetchall()
    return [r[0] for r in rows]
