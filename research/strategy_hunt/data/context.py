"""Market context series: SPY 5-min (Alpaca gap-fill) and VIX daily (yfinance)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from research.strategy_hunt.data.bars import _write, bars5_path, daily_path, last_ts, load_bars5, to_daily

logger = logging.getLogger("strategy_hunt.context")
BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"
ET = "America/New_York"
COLS = ["open", "high", "low", "close", "volume"]
START_DEFAULT = "2021-01-01"


def _iso(ts: pd.Timestamp) -> str:
    t = ts if ts.tzinfo else ts.tz_localize(ET)
    return t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def alpaca_bars(symbol: str, start: pd.Timestamp, end: pd.Timestamp, session: Any, key: str, secret: str,
                timeframe: str = "5Min", feed: str = "sip") -> pd.DataFrame:
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params: dict[str, Any] = {"symbols": symbol, "timeframe": timeframe, "start": _iso(start), "end": _iso(end),
                              "limit": 10000, "adjustment": "split", "feed": feed, "sort": "asc"}
    rows: list[dict] = []
    while True:
        resp = session.get(BARS_URL, headers=headers, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(payload.get("bars", {}).get(symbol, []))
        token = payload.get("next_page_token")
        if not token:
            break
        params = {**params, "page_token": token}
    if not rows:
        return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], name="ts"))
    df = pd.DataFrame(rows).rename(columns={"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
    df = df.set_index("ts")[COLS].sort_index()
    df = df[(df.index.time >= pd.Timestamp("09:30").time()) & (df.index.time <= pd.Timestamp("15:55").time())]
    return df.astype({"volume": "int64"})


def yfinance_daily(symbol: str, start: str = "2020-12-01") -> pd.DataFrame:
    try:
        import yfinance as yf

        df = yf.download(symbol, start=start, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], name="date"))
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        out = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})[COLS]
        out.index = pd.DatetimeIndex(pd.to_datetime(out.index).tz_localize(None).normalize(), name="date")
        return out.fillna({"volume": 0}).astype({"volume": "int64"})
    except Exception as exc:
        logger.warning("yfinance daily failed for %s: %s", symbol, exc)
        return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], name="date"))


def ensure_spy(root: Path, session: Any, key: str, secret: str, now: Optional[pd.Timestamp] = None) -> int:
    lt = last_ts(root, "SPY")
    start = (lt + pd.Timedelta(minutes=5)) if lt is not None else pd.Timestamp(START_DEFAULT)
    end = now if now is not None else pd.Timestamp.now(tz=ET).tz_localize(None)
    if start >= end:
        return 0
    new = alpaca_bars("SPY", start, end, session, key, secret)
    if new.empty:
        return 0
    merged = pd.concat([load_bars5(root, "SPY"), new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    _write(bars5_path(root, "SPY"), merged, "ts")
    _write(daily_path(root, "SPY"), to_daily(merged), "date")
    return len(new)


def ensure_vix(root: Path, fetcher: Callable[[str], pd.DataFrame] = yfinance_daily) -> int:
    df = fetcher("^VIX")
    if df.empty:
        return 0
    _write(daily_path(root, "VIX"), df, "date")
    return len(df)


def ensure_context(root: Path, session: Any, key: str, secret: str,
                   vix_fetcher: Callable[[str], pd.DataFrame] = yfinance_daily) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        out["spy"] = ensure_spy(root, session, key, secret) if session is not None and key and secret else -1
    except Exception as exc:
        logger.warning("ensure_spy failed: %s", exc)
        out["spy"] = -1
    try:
        out["vix"] = ensure_vix(root, vix_fetcher)
    except Exception as exc:
        logger.warning("ensure_vix failed: %s", exc)
        out["vix"] = -1
    return out
