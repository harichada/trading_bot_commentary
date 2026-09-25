"""Split detection, yfinance split cache, and back-adjustment of bars."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger("strategy_hunt.splits")
SPLIT_RATIOS = (2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 40, 50, 100)
MERGE_SESSIONS = 3


def _match(ratio: float, tol: float) -> float | None:
    for r in SPLIT_RATIOS:
        if abs(ratio / r - 1) <= tol:
            return 1.0 / r          # forward split: prior prices shrink
        if abs(ratio * r - 1) <= tol:
            return float(r)         # reverse split: prior prices grow
    return None


def detect_splits(daily: pd.DataFrame, tol: float = 0.03) -> list[tuple[pd.Timestamp, float]]:
    if daily.empty or len(daily) < 2:
        return []
    prev_close = daily["close"].shift(1)
    ratio = prev_close / daily["open"]
    intraday = (daily["close"] / daily["open"] - 1).abs()
    out = []
    for d, r, im in zip(daily.index, ratio, intraday):
        if np.isnan(r) or r <= 0 or im >= 0.25:
            continue
        m = _match(float(r), tol)
        if m is not None:
            out.append((pd.Timestamp(d), m))
    return out


def splits_path(root: Path, symbol: str) -> Path:
    return Path(root) / "splits" / f"{symbol.upper()}.parquet"


def update_splits(root: Path, symbol: str, fetcher: Callable[[str], pd.Series]) -> int:
    s = fetcher(symbol)
    rows = [(pd.Timestamp(d).tz_localize(None).normalize() if getattr(d, "tzinfo", None) else pd.Timestamp(d).normalize(), 1.0 / float(v))
            for d, v in s.items() if float(v) > 0]
    p = splits_path(root, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.parquet")
    pd.DataFrame(rows, columns=["date", "prior_multiplier"]).to_parquet(tmp, index=False)
    tmp.replace(p)
    return len(rows)


def load_splits(root: Path, symbol: str) -> list[tuple[pd.Timestamp, float]]:
    p = splits_path(root, symbol)
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    return [(pd.Timestamp(d), float(m)) for d, m in zip(pd.to_datetime(df["date"]), df["prior_multiplier"])]


def all_splits(root: Path, symbol: str, daily: pd.DataFrame) -> list[tuple[pd.Timestamp, float]]:
    cached = load_splits(root, symbol)
    detected = detect_splits(daily)
    out = list(cached)
    for d, m in detected:
        near = [c for c in cached if abs((c[0] - d).days) <= MERGE_SESSIONS * 2]
        if not near:
            out.append((d, m))
    return sorted(out)


def adjust_for_splits(bars: pd.DataFrame, splits: list[tuple[pd.Timestamp, float]]) -> pd.DataFrame:
    if not splits or bars.empty:
        return bars.copy()
    out = bars.copy()
    for d, m in splits:
        mask = out.index < pd.Timestamp(d)
        for col in ("open", "high", "low", "close"):
            out.loc[mask, col] = out.loc[mask, col] * m
        out.loc[mask, "volume"] = (out.loc[mask, "volume"] / m).round().astype(out["volume"].dtype)
    return out


def yfinance_splits_fetcher(symbol: str) -> pd.Series:
    try:
        import yfinance as yf

        s = yf.Ticker(symbol).splits
        return s if s is not None else pd.Series(dtype=float)
    except Exception as exc:
        logger.warning("splits fetch failed for %s: %s", symbol, exc)
        return pd.Series(dtype=float)


def load_adjusted(root: Path, symbol: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from research.strategy_hunt.data.bars import load_bars5, load_daily

    b5, d = load_bars5(root, symbol), load_daily(root, symbol)
    sp = all_splits(root, symbol, d)
    return adjust_for_splits(b5, sp), adjust_for_splits(d, sp)
