"""v-wedge-fix-2026-09-28 (Task 4, ruling 2): yfinance entry points for the
isolated subprocess pool.

Everything in here runs in a CHILD process launched by
`core.isolated_calls.run_isolated`. Three hard rules follow from that:

1. **Module-level functions only.** A spawn child unpickles a callable by
   qualified name; bound methods, closures and lambdas cannot cross the
   boundary.
2. **Picklable returns only.** DataFrames and plain dicts/lists are fine;
   yfinance `Ticker` / `FastInfo` objects are not — every entry point
   converts before returning.
3. **This module must not import `core`.** Importing `core` pulls
   `core/__init__` → `core/config` → `setup_logging()`, which attaches a
   RotatingFileHandler to `trading_bot.log`. Two processes rotating the
   same file corrupts the live bot's log. `child_init()` below strips that
   handler in the child anyway (the spawn bootstrap re-imports the main
   module, which does drag `core` in), belt and braces.

Every function degrades instead of raising: the parent treats None / empty
as "no data", which is exactly the fallback each call site already had.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("yf_isolated")

#: Fields the parent actually reads off a Yahoo news item. Normalising here
#: keeps the pickled payload small and guarantees the keys exist.
_NEWS_FIELDS = ("title", "summary", "link", "publisher", "providerPublishTime")


def child_init() -> None:
    """Pool initializer — runs once per child, after the spawn bootstrap.

    Detaches file handlers from the `TradingBot` logger so the child can
    never rotate or interleave into the live bot's log, and installs a
    socket default timeout so a silent Yahoo socket cannot hang the child
    forever (the parent's timeout kills it regardless; this just lets a
    healthy child recycle itself).
    """
    try:
        bot_logger = logging.getLogger("TradingBot")
        for handler in list(bot_logger.handlers):
            if isinstance(handler, logging.FileHandler):
                bot_logger.removeHandler(handler)
    except Exception:                                   # pragma: no cover
        pass
    try:
        import socket

        socket.setdefaulttimeout(15.0)
    except Exception:                                   # pragma: no cover
        pass


def _yf() -> Optional[Any]:
    """Import yfinance lazily, inside the child. Returns None if absent."""
    try:
        import yfinance as yf

        return yf
    except Exception as exc:
        logger.warning("yf_isolated: yfinance import failed: %s", exc)
        return None


# ── history / bars ────────────────────────────────────────────────────

def yf_download_daily(
    tickers: List[str],
    period: str = "90d",
    interval: str = "1d",
) -> Any:
    """Batch daily bars (screener quality filter). Returns the raw
    DataFrame — MultiIndex columns for multi-ticker requests, flat for
    one — or None. The parent normalises, exactly as it does today."""
    yf = _yf()
    if yf is None:
        return None
    try:
        return yf.download(
            tickers=list(tickers),
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.warning("yf_isolated: batch download failed: %s", exc)
        return None


def yf_history(symbol: str, period: str = "1d", interval: str = "5m") -> Any:
    """Single-symbol OHLCV with title-cased columns and `Adj Close`
    dropped — the shape `RealTimeDataProvider.get_market_data` returns."""
    yf = _yf()
    if yf is None:
        return None
    try:
        data = yf.Ticker(symbol).history(period=period, interval=interval)
        if data is None or data.empty:
            return None
        data.columns = [str(col).title() for col in data.columns]
        if "Adj Close" in data.columns:
            data = data.drop("Adj Close", axis=1, errors="ignore")
        return data
    except Exception as exc:
        logger.warning("yf_isolated: history failed for %s: %s", symbol, exc)
        return None


# ── news ──────────────────────────────────────────────────────────────

def yf_ticker_news(symbol: str) -> List[Dict[str, Any]]:
    """Yahoo news items, normalised to plain dicts with the five fields
    the callers read. Empty list on any failure."""
    yf = _yf()
    if yf is None:
        return []
    try:
        raw = yf.Ticker(symbol).news or []
    except Exception as exc:
        logger.warning("yf_isolated: news failed for %s: %s", symbol, exc)
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            # Types are pinned here, not at the call sites: the callers do
            # `article['link'].encode()` and `fromtimestamp(...)` straight
            # off these fields, so a None would be a crash rather than a
            # missing article.
            out.append({
                "title": str(item.get("title") or ""),
                "summary": str(item.get("summary") or ""),
                "link": str(item.get("link") or ""),
                "publisher": str(item.get("publisher") or ""),
                "providerPublishTime": int(item.get("providerPublishTime") or 0),
            })
        except (TypeError, ValueError) as exc:
            logger.debug("yf_isolated: news item skipped for %s: %s", symbol, exc)
    return out


# ── quotes ────────────────────────────────────────────────────────────

def yf_fast_quote(symbol: str) -> Optional[Dict[str, float]]:
    """`fast_info` collapsed into the quote dict shape the engine expects."""
    yf = _yf()
    if yf is None:
        return None
    try:
        info = yf.Ticker(symbol).fast_info
        last = info.last_price
        if not last:
            return None
        return {
            "bid": getattr(info, "bid", 0) or last * 0.9999,
            "ask": getattr(info, "ask", 0) or last * 1.0001,
            "last": last,
            "volume": info.last_volume or 0,
            "high": info.day_high or last,
            "low": info.day_low or last,
            "open": info.open or last,
            "close": info.previous_close or last,
            "market_cap": getattr(info, "market_cap", 0) or 0,
        }
    except Exception as exc:
        logger.warning("yf_isolated: quote failed for %s: %s", symbol, exc)
        return None


def yf_multi_quotes(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """Batch quotes; symbols that fail are simply absent from the result."""
    yf = _yf()
    if yf is None:
        return {}
    quotes: Dict[str, Dict[str, float]] = {}
    try:
        tickers = yf.Tickers(" ".join(symbols))
    except Exception as exc:
        logger.warning("yf_isolated: multi-quote setup failed: %s", exc)
        return {}
    for symbol in symbols:
        try:
            ticker = tickers.tickers.get(symbol)
            if ticker is None:
                continue
            info = ticker.fast_info
            prev = info.previous_close
            quotes[symbol] = {
                "last": info.last_price,
                "volume": info.last_volume or 0,
                "high": info.day_high or info.last_price,
                "low": info.day_low or info.last_price,
                "open": info.open or info.last_price,
                "change_pct": (
                    (info.last_price - prev) / prev * 100 if prev else 0
                ),
            }
        except Exception as exc:
            logger.debug("yf_isolated: quote skipped for %s: %s", symbol, exc)
    return quotes


def yf_market_breadth(symbols: List[str]) -> Optional[Dict[str, float]]:
    """Advance/decline breadth over the index+sector ETF list."""
    yf = _yf()
    if yf is None:
        return None
    advances = 0
    declines = 0
    try:
        tickers = yf.Tickers(" ".join(symbols))
    except Exception as exc:
        logger.warning("yf_isolated: breadth setup failed: %s", exc)
        return None
    for symbol in symbols:
        try:
            ticker = tickers.tickers.get(symbol)
            if ticker is None:
                continue
            info = ticker.fast_info
            if info.last_price > info.previous_close:
                advances += 1
            else:
                declines += 1
        except Exception as exc:
            logger.debug("yf_isolated: breadth skipped for %s: %s", symbol, exc)
    total = advances + declines
    if total == 0:
        return None
    return {
        "advance_decline_ratio": advances / max(declines, 1),
        "advancing_percent": advances / total * 100,
        "declining_percent": declines / total * 100,
        "advances": advances,
        "declines": declines,
    }


def probe() -> str:
    """Readiness probe. Submitted once when the pool is created so the
    parent can tell a booted child from a cold one WITHOUT importing
    `core` in the child (that is the whole point of living in this
    module). Deliberately trivial — it measures spawn, not Yahoo."""
    return "ready"
