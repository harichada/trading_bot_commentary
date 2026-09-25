"""Alpaca historical news -> per-symbol Parquet with pluggable sentiment scoring. Incremental and cached."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from research.strategy_hunt.data.sentiment import available_backends, score

logger = logging.getLogger("strategy_hunt.news")

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
COLUMNS = ["created_at", "headline", "source", "sentiment"]
# v-playbook-desk-b1-2026-09-25: per-backend sentiment columns stored alongside
# the legacy `sentiment` column so families can pick either (§7.5).
EXTRA_COLUMNS = ["sentiment_vader", "sentiment_finbert"]
PAGE_LIMIT = 50
ET = "America/New_York"


def score_headlines(headlines: list[str]) -> list[float]:
    """VADER sentiment for `headlines` (backward-compatible helper; always VADER)."""
    return score(headlines, backend="vader")


def news_path(root: Path, symbol: str) -> Path:
    return Path(root) / "news" / f"{symbol.upper()}.parquet"


def _iso(ts: pd.Timestamp) -> str:
    t = ts if ts.tzinfo else ts.tz_localize("UTC")
    return t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _configured_backend() -> str:
    from core.config import Config

    return Config().SENTIMENT_BACKEND


def _score_columns(headlines: list[str]) -> dict[str, list[float]]:
    """sentiment_vader always; sentiment_finbert only when the configured
    backend is finbert (and it is actually available), else NaN.

    v-playbook-desk-b1-2026-09-25 (Task 7 fix round 1, Critical finding):
    `available_backends()` (which triggers the lazy FinBERT load) is only
    called when the configured backend is "finbert" — under the default
    "vader" backend, no FinBERT load is attempted at all.
    """
    cols: dict[str, list[float]] = {"sentiment_vader": score(headlines, backend="vader")}
    if _configured_backend() == "finbert" and "finbert" in available_backends():
        cols["sentiment_finbert"] = score(headlines, backend="finbert")
    else:
        cols["sentiment_finbert"] = [float("nan")] * len(headlines)
    return cols


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
        return pd.DataFrame(columns=COLUMNS + EXTRA_COLUMNS)
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(ET).dt.tz_localize(None)
    headlines = df["headline"].tolist()
    for col, values in _score_columns(headlines).items():
        df[col] = values
    backend = _configured_backend()
    sentiment_col = f"sentiment_{backend}"
    # v-playbook-desk-b1-2026-09-25 (Task 7 fix round 1, Minor finding):
    # fall back to sentiment_vader per-row wherever the configured backend's
    # column is NaN (e.g. finbert configured but unavailable), rather than
    # poisoning `sentiment` with NaN for the whole batch.
    if sentiment_col in df.columns:
        df["sentiment"] = df[sentiment_col].fillna(df["sentiment_vader"])
    else:
        df["sentiment"] = df["sentiment_vader"]
    return df[COLUMNS + EXTRA_COLUMNS].sort_values("created_at").reset_index(drop=True)


def load_news(root: Path, symbol: str) -> pd.DataFrame:
    p = news_path(root, symbol)
    if not p.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_parquet(p)
    df["created_at"] = pd.to_datetime(df["created_at"])
    extra = [c for c in EXTRA_COLUMNS if c in df.columns]
    return df[COLUMNS + extra].sort_values("created_at").reset_index(drop=True)


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
