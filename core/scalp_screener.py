"""
Daily symbol screener for the scalp reversal engines.

Pulls today's top movers (gainers + losers) from Alpaca's market screener,
scores them by absolute move % and relative volume, and returns the top N
symbols most likely to produce RSI extreme + BB reversal setups.

Sources (tried in order):
  1. Alpaca ScreenerClient  — `/v2/screener/stocks/movers` (requires data subscription)
  2. yfinance day_gainers   — scrapes Yahoo Finance gainers/losers

Filters:
  - min_price: $2.00 (need shares with $10k slot)
  - max_price: $600.00
  - min_avg_volume: 500,000 (liquid enough for 1-min scalps)
  - min_move_pct: 3.0% absolute move from prior close (ensures RSI extremes)

Score = 0.5 × abs_move_score + 0.5 × rvol_score   (both normalised 0-1)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("TradingBot")

# ── filter defaults ────────────────────────────────────────────────────────
MIN_PRICE       = 2.0
MAX_PRICE       = 600.0
MIN_AVG_VOLUME  = 500_000
MIN_MOVE_PCT    = 3.0      # absolute % from prior close
MAX_MOVE_PCT    = 60.0     # ignore halted/circuit-breaker spikes


@dataclass
class SymbolScore:
    symbol:     str
    move_pct:   float    # positive = gainer, negative = loser
    rvol:       float    # today's vol / avg_vol
    price:      float
    composite:  float    # 0–1 ranking score


# ── Alpaca screener ────────────────────────────────────────────────────────

def _fetch_alpaca_movers(key: str, secret: str, top: int = 50) -> list[dict]:
    """Call Alpaca /v2/screener/stocks/movers and return raw dicts."""
    try:
        from alpaca.data.historical.screener import ScreenerClient
        from alpaca.data.requests import MostActivesRequest
        sc  = ScreenerClient(api_key=key, secret_key=secret)
        # get_stock_movers returns top gainers + losers
        resp = sc.get_stock_movers(top=top)
        rows = []
        for item in list(getattr(resp, "gainers", [])) + list(getattr(resp, "losers", [])):
            rows.append({
                "symbol":    item.symbol,
                "move_pct":  float(getattr(item, "percent_change", 0.0)),
                "price":     float(getattr(item, "price", 0.0)),
                "volume":    int(getattr(item, "volume", 0)),
                "avg_volume": 0,   # not provided — will fallback to rvol=1
            })
        logger.info("scalp_screener: Alpaca movers returned %d symbols", len(rows))
        return rows
    except Exception as exc:
        logger.debug("scalp_screener: Alpaca movers failed: %s", exc)
        return []


# ── yfinance screener fallback ─────────────────────────────────────────────

def _fetch_yfinance_movers(top: int = 50) -> list[dict]:
    """Fetch Yahoo Finance gainers + losers via yfinance."""
    try:
        import yfinance as yf
        rows = []
        for screen_name in ("day_gainers", "day_losers"):
            try:
                result = yf.screen(screen_name, count=top)
                quotes = result.get("quotes", [])
                for q in quotes:
                    sym = q.get("symbol", "")
                    if not sym:
                        continue
                    pc = float(q.get("regularMarketChangePercent", 0.0))
                    rows.append({
                        "symbol":    sym,
                        "move_pct":  pc,
                        "price":     float(q.get("regularMarketPrice", 0.0)),
                        "volume":    int(q.get("regularMarketVolume", 0)),
                        "avg_volume": int(q.get("averageDailyVolume3Month", 0)),
                    })
            except Exception as exc:
                logger.debug("scalp_screener: yfinance screen %s: %s", screen_name, exc)
        logger.info("scalp_screener: yfinance movers returned %d symbols", len(rows))
        return rows
    except Exception as exc:
        logger.debug("scalp_screener: yfinance unavailable: %s", exc)
        return []


# ── scoring + filtering ────────────────────────────────────────────────────

def _score(rows: list[dict]) -> list[SymbolScore]:
    scored = []
    for r in rows:
        sym  = r.get("symbol", "")
        if not sym or "." in sym or len(sym) > 5:
            continue   # skip ETF sub-classes, warrants, etc.
        price     = float(r.get("price", 0.0))
        move_pct  = float(r.get("move_pct", 0.0))
        volume    = int(r.get("volume", 0))
        avg_vol   = int(r.get("avg_volume", 0))

        if price < MIN_PRICE or price > MAX_PRICE:
            continue
        if abs(move_pct) < MIN_MOVE_PCT or abs(move_pct) > MAX_MOVE_PCT:
            continue
        if avg_vol > 0 and avg_vol < MIN_AVG_VOLUME:
            continue

        rvol = (volume / avg_vol) if avg_vol > 1000 else 1.0
        move_score = min(abs(move_pct) / 15.0, 1.0)   # normalise: 15%+ = max
        rvol_score = min(rvol / 5.0, 1.0)              # normalise: 5×+ = max
        composite  = move_score * 0.5 + rvol_score * 0.5

        scored.append(SymbolScore(
            symbol=sym, move_pct=move_pct, rvol=rvol,
            price=price, composite=composite,
        ))
    scored.sort(key=lambda s: s.composite, reverse=True)
    return scored


# ── public API ─────────────────────────────────────────────────────────────

def select_symbols(
    top_n: int = 15,
    key: str | None = None,
    secret: str | None = None,
) -> list[SymbolScore]:
    """Return top-N scored symbols for today's session.

    Tries Alpaca ScreenerClient first, falls back to yfinance.
    Returns an empty list (not an error) if both sources are unavailable.
    """
    key    = key    or os.getenv("ALPACA_PAPER_KEY") or os.getenv("ALPACA_API_KEY") or ""
    secret = secret or os.getenv("ALPACA_PAPER_SECRET") or os.getenv("ALPACA_SECRET_KEY") or ""

    rows = _fetch_alpaca_movers(key, secret, top=top_n * 3)
    if not rows:
        rows = _fetch_yfinance_movers(top=top_n * 3)
    if not rows:
        logger.warning("scalp_screener: both sources empty — caller should use fallback")
        return []

    scored = _score(rows)
    top    = scored[:top_n]
    logger.info(
        "scalp_screener: top %d symbols: %s",
        len(top),
        [(s.symbol, f"{s.move_pct:+.1f}%", f"rvol={s.rvol:.1f}x") for s in top],
    )
    return top


def select_symbol_names(top_n: int = 15, **kw) -> list[str]:
    """Convenience wrapper — returns just the symbol strings."""
    return [s.symbol for s in select_symbols(top_n=top_n, **kw)]
