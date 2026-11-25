"""
Direction quality gate for scalp re-entry after losses.

After a symbol hits the daily-loss caution threshold, every new signal
runs through this gate before an order is placed.  The gate checks five
independent signals and requires at least `min_score` to agree with the
intended trade direction.

Signals
-------
1. VWAP position     — price on correct side of today's session VWAP
2. RSI direction     — RSI trending toward the signal direction (last 4 bars)
3. Order flow proxy  — net candle-body direction over last 5 bars
4. L2 bid/ask ratio  — Alpaca latest quote bid_size vs ask_size (cached 30 s)
5. News sentiment    — Alpaca news headlines (cached 5 min), TextBlob polarity

Returns
-------
(allowed: bool, reason: str)
  allowed  — True when confirmed_signals >= min_score
  reason   — human-readable summary, e.g. "3/5 confirmed: vwap rsi flow"
"""
from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd

logger = logging.getLogger("TradingBot")

# ── per-process cache (symbol → {l2_ts, l2_ratio, news_ts, news_sent}) ────
_cache: dict[str, dict] = {}
_L2_TTL   = 30.0        # seconds between Alpaca quote fetches
_NEWS_TTL = 300.0       # seconds between Alpaca news fetches


# ── helpers ────────────────────────────────────────────────────────────────

_VWAP_MAX_GAP = 0.025   # allow up to 2.5% below VWAP for a long reversal snap-back

def _vwap_check(history: pd.DataFrame, side: str) -> tuple[bool, float]:
    """VWAP distance check tuned for reversal entries.

    Long:  price is within _VWAP_MAX_GAP below VWAP (close enough to snap back).
           Price far below VWAP means the stock is in free-fall — skip.
    Short: price within _VWAP_MAX_GAP above VWAP (close enough to fade).
    """
    if history.empty or len(history) < 2:
        return True, float("nan")
    typical = (history["High"] + history["Low"] + history["Close"]) / 3.0
    cum_tpv = (typical * history["Volume"]).cumsum()
    cum_vol = history["Volume"].cumsum().replace(0, float("nan"))
    vwap    = float((cum_tpv / cum_vol).iloc[-1])
    close   = float(history["Close"].iloc[-1])
    gap     = (close - vwap) / vwap   # negative = below VWAP
    if side == "long":
        ok = gap >= -_VWAP_MAX_GAP    # not too far below (or already above)
    else:
        ok = gap <= _VWAP_MAX_GAP     # not too far above
    return bool(ok), round(vwap, 4)


def _rsi_direction_check(ind: pd.DataFrame, side: str) -> bool:
    """RSI micro-turn: did RSI turn the right direction on the LAST bar?

    For a reversal entry we don't require a 4-bar trend (RSI is falling at
    the bottom by definition).  A single bar of RSI turning up is enough
    to confirm the micro-reversal has started.
    """
    if len(ind) < 2:
        return True
    rsi_now  = float(ind["rsi"].iloc[-1])
    rsi_prev = float(ind["rsi"].iloc[-2])
    turning  = rsi_now > rsi_prev
    return turning if side == "long" else not turning


def _order_flow_check(history: pd.DataFrame, side: str) -> bool:
    """Volume exhaustion check: is there capitulation volume on recent bars?

    A volume spike (vol_ratio ≥ 1.5) on the last 1-3 bars indicates
    climactic selling (for longs) or buying (for shorts) — a classic
    exhaustion / reversal signal.  We use the max vol_ratio over the
    last 3 bars so we catch spikes that peaked one bar before the signal.
    """
    recent = history.tail(3)
    if recent.empty:
        return True
    avg_vol = history["Volume"].mean()
    if avg_vol <= 0:
        return True
    max_ratio = float(recent["Volume"].max() / avg_vol)
    return max_ratio >= 1.5


def _l2_check(symbol: str, side: str, data_client: Any) -> bool | None:
    """Alpaca latest-quote bid/ask size ratio.
    Long: bid_size / (bid+ask) > 0.45  (more buyers than sellers).
    Short: bid_size / (bid+ask) < 0.55 (sellers dominating).
    Returns None if unavailable (skipped, not scored)."""
    if data_client is None:
        return None
    now    = time.time()
    cached = _cache.setdefault(symbol, {})
    if now - cached.get("l2_ts", 0) < _L2_TTL:
        ratio = cached.get("l2_ratio")
        if ratio is None:
            return None
    else:
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            resp  = data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            )
            q        = resp[symbol]
            bid_sz   = max(float(q.bid_size or 0), 1.0)
            ask_sz   = max(float(q.ask_size or 0), 1.0)
            ratio    = bid_sz / (bid_sz + ask_sz)
            cached["l2_ts"]    = now
            cached["l2_ratio"] = ratio
        except Exception as exc:
            logger.debug("direction_gate L2 fetch %s: %s", symbol, exc)
            cached["l2_ts"]    = now
            cached["l2_ratio"] = None
            return None
    return (ratio > 0.45) if side == "long" else (ratio < 0.55)


def _news_sentiment_check(symbol: str, side: str, data_client: Any) -> bool | None:
    """Alpaca news headlines → TextBlob polarity.
    Long: avg polarity >= -0.25 (neutral/positive; strongly negative = news-driven fall).
    Short: avg polarity <=  0.25 (neutral/negative; strong good-news = don't short).
    Returns None if unavailable (skipped)."""
    if data_client is None:
        return None
    now    = time.time()
    cached = _cache.setdefault(symbol, {})
    if now - cached.get("news_ts", 0) < _NEWS_TTL:
        sent = cached.get("news_sent")
        if sent is None:
            return None
    else:
        try:
            from datetime import datetime, timedelta, timezone
            from alpaca.data.requests import NewsRequest
            start_dt = datetime.now(timezone.utc) - timedelta(hours=4)
            news     = data_client.get_news(
                NewsRequest(symbols=[symbol], limit=5, start=start_dt)
            )
            items  = news.get(symbol, []) if isinstance(news, dict) else list(news)
            if not items:
                cached["news_ts"]   = now
                cached["news_sent"] = None
                return None
            try:
                from textblob import TextBlob
                headlines = [getattr(n, "headline", str(n)) for n in items]
                polars    = [TextBlob(h).sentiment.polarity for h in headlines]
                sent      = sum(polars) / len(polars)
            except ImportError:
                sent = 0.0   # textblob not available — neutral
            cached["news_ts"]   = now
            cached["news_sent"] = sent
        except Exception as exc:
            logger.debug("direction_gate news fetch %s: %s", symbol, exc)
            cached["news_ts"]   = now
            cached["news_sent"] = None
            return None
    return (sent >= -0.25) if side == "long" else (sent <= 0.25)


# ── public entry point ─────────────────────────────────────────────────────

def direction_gate(
    symbol: str,
    side: str,
    history: pd.DataFrame,
    ind: pd.DataFrame,
    data_client: Any = None,
    min_score: int = 3,
) -> tuple[bool, str]:
    """Run the five-signal direction gate.

    Parameters
    ----------
    symbol      Ticker symbol (for L2/news caching).
    side        "long" or "short".
    history     Today's RTH bar DataFrame — columns Open/High/Low/Close/Volume.
    ind         compute_indicators() output aligned with history.
    data_client Alpaca StockHistoricalDataClient (or None — skips L2 and news).
    min_score   Required confirmed signals (default 3).

    Returns
    -------
    (allowed, reason_str)
    """
    results: dict[str, bool | None] = {}

    results["vwap"], vwap_val = _vwap_check(history, side)
    results["rsi"]            = _rsi_direction_check(ind, side)
    results["flow"]           = _order_flow_check(history, side)
    results["l2"]             = _l2_check(symbol, side, data_client)
    results["news"]           = _news_sentiment_check(symbol, side, data_client)

    scored    = {k: v for k, v in results.items() if v is not None}
    confirmed = [k for k, v in scored.items() if v]
    failed    = [k for k, v in scored.items() if not v]
    skipped   = [k for k, v in results.items() if v is None]

    score     = len(confirmed)
    available = len(scored)
    # Adjust threshold if fewer signals are available
    threshold = max(2, min(min_score, available))
    allowed   = score >= threshold

    parts = []
    if confirmed: parts.append("OK:" + "+".join(confirmed))
    if failed:    parts.append("FAIL:" + "+".join(failed))
    if skipped:   parts.append("skip:" + "+".join(skipped))
    reason = f"{score}/{available} {'PASS' if allowed else 'FAIL'} [{' '.join(parts)}]"
    if not allowed and vwap_val == vwap_val:   # not NaN
        reason += f" vwap={vwap_val}"

    return allowed, reason
