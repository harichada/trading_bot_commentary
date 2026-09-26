"""Split detection, yfinance split cache, and back-adjustment of bars."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger("strategy_hunt.splits")
SPLIT_RATIOS = (2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 25, 30, 40, 50, 100)
PERSIST_SESSIONS = 5     # window (sessions) on each side of the candidate date used to confirm
                          # the price move actually held, rather than a gap that reverted.
PERSIST_TOL = 0.05


def _match(ratio: float, tol: float) -> float | None:
    for r in SPLIT_RATIOS:
        if abs(ratio / r - 1) <= tol:
            return 1.0 / r          # forward split: prior prices shrink
        if abs(ratio * r - 1) <= tol:
            return float(r)         # reverse split: prior prices grow
    return None


def _persists(closes: np.ndarray, i: int, m: float, tol: float) -> bool:
    """Ruling 2: a candidate at position `i` is only genuine if the price level actually held --
    median(close[i : i+PERSIST_SESSIONS]) / median(close[i-PERSIST_SESSIONS : i]) must match the
    candidate multiplier `m` within `tol`. A gap that reverts within a few sessions (a bad print,
    a one-day crash) fails this and is dropped."""
    before = closes[max(0, i - PERSIST_SESSIONS):i]
    after = closes[i:i + PERSIST_SESSIONS]
    if len(before) == 0 or len(after) == 0:
        return False
    med_before, med_after = np.median(before), np.median(after)
    if not np.isfinite(med_before) or med_before <= 0 or not np.isfinite(med_after):
        return False
    return abs((med_after / med_before) / m - 1) <= tol


def detect_splits(daily: pd.DataFrame, tol: float = 0.03) -> list[tuple[pd.Timestamp, float]]:
    """Candidate split dates from prev_close/open ratio matching a known split ratio within `tol`,
    with the intraday move < 25% filter (Finding), an explicit reject of a non-positive/NaN open
    (Ruling 2), and a persistence check (Ruling 2, `_persists`) so a gap that reverts is not
    mistaken for a split."""
    if daily.empty or len(daily) < 2:
        return []
    prev_close = daily["close"].shift(1)
    open_ = daily["open"]
    ratio = prev_close / open_
    intraday = (daily["close"] / open_ - 1).abs()
    closes = daily["close"].to_numpy(dtype=float)
    out = []
    for i, (d, r, im, o) in enumerate(zip(daily.index, ratio, intraday, open_)):
        if pd.isna(o) or o <= 0 or np.isnan(r) or r <= 0 or im >= 0.25:
            continue
        m = _match(float(r), tol)
        if m is None or not _persists(closes, i, m, PERSIST_TOL):
            continue
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
    """Ruling 1: a splits cache file (`splits/<SYM>.parquet`) is authoritative, even when it is
    empty -- an empty cache means yfinance was checked and found no splits, not "unknown". The
    detector (`detect_splits`) only ever runs for a symbol that has no cache file at all, and any
    split it finds is logged at WARNING (Ruling 3) so the nightly log shows what got applied
    without a cached record backing it."""
    if splits_path(root, symbol).exists():
        return sorted(load_splits(root, symbol))
    detected = sorted(detect_splits(daily))
    for d, m in detected:
        logger.warning("detected split applied for %s: date=%s multiplier=%s (no cache file)",
                       symbol, d.date(), m)
    return detected


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
