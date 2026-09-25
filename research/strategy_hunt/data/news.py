"""Alpaca historical news -> per-symbol Parquet with VADER sentiment. Incremental and cached."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger("strategy_hunt.news")

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
COLUMNS = ["created_at", "headline", "source", "sentiment"]
PAGE_LIMIT = 50
ET = "America/New_York"

_analyzer = None


def _vader():
    global _analyzer
    if _analyzer is None:
        from nltk.sentiment import SentimentIntensityAnalyzer

        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def score_headlines(headlines: list[str]) -> list[float]:
    an = _vader()
    return [float(an.polarity_scores(h or "")["compound"]) for h in headlines]


def news_path(root: Path, symbol: str) -> Path:
    return Path(root) / "news" / f"{symbol.upper()}.parquet"


def _iso(ts: pd.Timestamp) -> str:
    t = ts if ts.tzinfo else ts.tz_localize("UTC")
    return t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_news(symbol: str, start: pd.Timestamp, end: pd.Timestamp, session: Any, key: str, secret: str) -> pd.DataFrame:
    """Fetch all news for `symbol` in [start, end] (UTC), paginating. Timestamps converted to naive ET."""
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params: dict[str, Any] = {"symbols": symbol.upper(), "start": _iso(start), "end": _iso(end),
                              "limit": PAGE_LIMIT, "sort": "asc", "include_content": "false"}
    rows: list[dict[str, Any]] = []
    while True:
        resp = session.get(NEWS_URL, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("news", []):
            rows.append({"created_at": item["created_at"], "headline": item.get("headline", ""),
                         "source": item.get("source", "")})
        token = payload.get("next_page_token")
        if not token:
            break
        params = {**params, "page_token": token}
    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
    df["sentiment"] = score_headlines(df["headline"].tolist())
    return df[COLUMNS].sort_values("created_at").reset_index(drop=True)


def load_news(root: Path, symbol: str) -> pd.DataFrame:
    p = news_path(root, symbol)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_parquet(p)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df[COLUMNS].sort_values("created_at").reset_index(drop=True)


def update_news(root: Path, symbol: str, session: Any, key: str, secret: str,
                start_default: str = "2021-01-01", now: Optional[pd.Timestamp] = None) -> int:
    """Fetch news after the cached max timestamp and append. Returns rows added."""
    existing = load_news(root, symbol)
    if existing.empty:
        start = pd.Timestamp(start_default).tz_localize("UTC")
    else:
        start = existing["created_at"].max().tz_localize(ET).tz_convert("UTC") + pd.Timedelta(seconds=1)
    end = (now if now is not None else pd.Timestamp.utcnow())
    end = end if end.tzinfo else end.tz_localize("UTC")
    new = fetch_news(symbol, start, end, session, key, secret)
    if new.empty:
        return 0
    merged = pd.concat([existing, new]).drop_duplicates(subset=["created_at", "headline"])
    merged = merged.sort_values("created_at").reset_index(drop=True)
    p = news_path(root, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.parquet")
    merged.to_parquet(tmp, index=False)
    tmp.replace(p)
    return len(new)
