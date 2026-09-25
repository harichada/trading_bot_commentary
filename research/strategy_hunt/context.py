"""Universe context: per-symbol RVOL/RS/catalyst features and their cross-sectional ranks.

`symbol_daily_features` computes the known-at-open feature set for one symbol from already
split-adjusted bars (see `research.strategy_hunt.data.splits.load_adjusted`). Everything except
`rvol_open5`/`rvol_open15`/`gap_pct` is lagged one session (computed through the prior close) so
it is genuinely known before today's open; the RVOL/gap columns use today's opening bars and are
only consumed by families from bar_of_day >= 1 (see strategies/families/context.py).
`rank_cross_section` adds the per-date cross-sectional columns once every symbol's frame is
concatenated: `rvol_rank5` is a same-date rank of `rvol_open5` and so inherits that same
bar_of_day >= 1 restriction; `rs_rank_60` and `adv_bucket` rank/bucket same-date peers of already-
lagged inputs (`rs_60`, `adv_dollars_20`) and so carry no additional look-ahead of their own.
`build_universe_context`/`load_context_slice` are the on-disk read/write side.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import ta

from strategies.families.context import CTX_COLUMNS

logger = logging.getLogger("strategy_hunt.context")

RVOL_LOOKBACK = 14
RVOL_MIN_PRIOR = 5
RS_WINDOWS = (20, 60, 120)
HIGH_WINDOW = 252
HIGH_MIN_PERIODS = HIGH_WINDOW // 2  # 126: half the lookback before trusting the 52w high
ADV_WINDOW = 20
ADV_MIN_PERIODS = ADV_WINDOW // 2  # 10: half the lookback before trusting the 20-session ADV
ATR_WINDOW = 14
SESSION_OPEN = pd.Timedelta(hours=9, minutes=30)
ADV_BUCKETS = (("mega", 1e9), ("large", 2e8), ("mid", 2e7))

_RANK_COLUMNS = ("rvol_rank5", "rs_rank_60", "adv_bucket")
_PER_SYMBOL_COLUMNS = tuple(c for c in CTX_COLUMNS if c not in _RANK_COLUMNS)


def _safe_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Matches strategies/families/indicators.py: ta's ATR indexes out of bounds when the
    frame is shorter than the window, so degrade to all-NaN instead of crashing."""
    if len(close) < window:
        return pd.Series(np.nan, index=close.index)
    return ta.volatility.average_true_range(high, low, close, window=window)


def _session_first_bar_volumes(bars5: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(first-bar volume, first-three-bar volume) per session, indexed by normalized date.
    first-three-bar volume is NaN for a session with fewer than 3 recorded bars — a partial
    session can't be compared to a full opening-15-minutes baseline. Sorted defensively since
    `.iloc[:3]` within each session depends on chronological order."""
    if bars5.empty:
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], name="date"))
        return empty, empty.copy()
    bars5 = bars5.sort_index()
    grp = bars5.groupby(bars5.index.normalize())
    first5 = grp["volume"].first().astype(float)
    first15 = grp["volume"].apply(lambda s: s.iloc[:3].sum() if len(s) >= 3 else np.nan).astype(float)
    first5.index.name = "date"
    first15.index.name = "date"
    return first5, first15


def _rolling_rvol(session_values: pd.Series) -> pd.Series:
    """value / mean(prior RVOL_LOOKBACK sessions); NaN when fewer than RVOL_MIN_PRIOR of those
    prior sessions have a value."""
    prior_mean = session_values.shift(1).rolling(RVOL_LOOKBACK, min_periods=RVOL_MIN_PRIOR).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        return session_values / prior_mean


def _relative_strength(close: pd.Series, spy_close: pd.Series, n: int) -> pd.Series:
    """Symbol return over the prior n sessions (through the prior close) minus SPY's, in %."""
    sym_ret = close.shift(1) / close.shift(1 + n) - 1
    spy_ret = (spy_close.shift(1) / spy_close.shift(1 + n) - 1).reindex(sym_ret.index)
    return (sym_ret - spy_ret) * 100


def _earnings_features(idx: pd.DatetimeIndex, earnings_table: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """earnings_today (date == an earnings date), earnings_prev (an earnings date equal to the
    prior session or strictly between the prior session and today), eps_surprise_pct on those
    earnings_prev days (the latest such date's surprise_pct when more than one qualifies)."""
    earnings_dates = sorted(earnings_table.index)
    today = pd.Series(idx.isin(earnings_dates), index=idx)
    prev = pd.Series(False, index=idx)
    surprise = pd.Series(np.nan, index=idx)
    prior = pd.Series(idx, index=idx).shift(1)
    for d in idx:
        prior_d = prior.loc[d]
        if pd.isna(prior_d):
            continue
        window = [e for e in earnings_dates if prior_d <= e < d]
        if window:
            prev.loc[d] = True
            surprise.loc[d] = float(earnings_table.loc[max(window), "surprise_pct"])
    return today, prev, surprise


def _news_features(idx: pd.DatetimeIndex, news: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Count and mean sentiment of news items with created_at in [session_open - 24h, session_open)."""
    counts = pd.Series(0.0, index=idx)
    sent = pd.Series(np.nan, index=idx)
    if news.empty:
        return counts, sent
    created, sentiment = news["created_at"], news["sentiment"]
    for d in idx:
        open_ts = pd.Timestamp(d) + SESSION_OPEN
        mask = (created >= open_ts - pd.Timedelta(hours=24)) & (created < open_ts)
        if mask.any():
            counts.loc[d] = float(mask.sum())
            sent.loc[d] = float(sentiment[mask].mean())
    return counts, sent


def _adv_bucket(adv: float) -> str:
    """mega >= 1e9, large >= 2e8, mid >= 2e7, else small. NaN/inf is explicitly small, matching
    research/strategy_hunt/sim/calibrate.py's `_bucket` convention."""
    if not np.isfinite(adv):
        return "small"
    for name, floor in ADV_BUCKETS:
        if adv >= floor:
            return name
    return "small"


def symbol_daily_features(
    symbol: str,
    bars5: pd.DataFrame,
    daily: pd.DataFrame,
    spy_daily: pd.DataFrame,
    earnings_table: pd.DataFrame,
    news: pd.DataFrame,
) -> pd.DataFrame:
    """Per-session features for one symbol, indexed by session date. Columns are CTX_COLUMNS minus
    the cross-sectional ones (rvol_rank5, rs_rank_60, adv_bucket), which rank_cross_section adds.

    `symbol` isn't read by the computation itself (every column derives from `bars5`/`daily`/
    `spy_daily`/`earnings_table`/`news`); it's kept in the signature as part of the public
    interface so callers — notably `build_universe_context`'s per-symbol try/except — have it on
    hand for log context."""
    idx = daily.index
    out = pd.DataFrame(index=idx)
    out.index.name = "date"

    close, open_, high, low = daily["close"], daily["open"], daily["high"], daily["low"]
    volume = daily["volume"].astype(float)
    prior_close = close.shift(1)

    first5, first15 = _session_first_bar_volumes(bars5)
    out["rvol_open5"] = _rolling_rvol(first5.reindex(idx))
    out["rvol_open15"] = _rolling_rvol(first15.reindex(idx))
    out["gap_pct"] = (open_ / prior_close - 1) * 100

    for n in RS_WINDOWS:
        out[f"rs_{n}"] = _relative_strength(close, spy_daily["close"], n)

    rolling_high = prior_close.rolling(HIGH_WINDOW, min_periods=HIGH_MIN_PERIODS).max()
    out["pct_from_52w_high"] = (prior_close / rolling_high - 1) * 100
    out["adv_dollars_20"] = (close * volume).shift(1).rolling(ADV_WINDOW, min_periods=ADV_MIN_PERIODS).mean()

    atr = _safe_atr(high, low, close, window=ATR_WINDOW)
    out["atr14_pct"] = atr.shift(1) / prior_close * 100

    out["earnings_today"], out["earnings_prev"], out["eps_surprise_pct"] = _earnings_features(idx, earnings_table)
    out["news_count_24h"], out["news_sent_24h"] = _news_features(idx, news)

    return out[list(_PER_SYMBOL_COLUMNS)]


def rank_cross_section(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-symbol frames (as from symbol_daily_features) with a symbol column and add
    the cross-sectional columns: rvol_rank5 (1 = highest rvol_open5 that date, NaN unranked),
    rs_rank_60 (percentile 0-1 of rs_60 that date), adv_bucket (from adv_dollars_20)."""
    parts = []
    for symbol, frame in frames.items():
        if frame is None or frame.empty:
            continue
        part = frame.reset_index()
        part.insert(1, "symbol", symbol)
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["date", "symbol", *CTX_COLUMNS])

    combined = pd.concat(parts, ignore_index=True)
    combined["rvol_rank5"] = combined.groupby("date")["rvol_open5"].rank(ascending=False, method="min")
    combined["rs_rank_60"] = combined.groupby("date")["rs_60"].rank(pct=True)
    combined["adv_bucket"] = combined["adv_dollars_20"].map(_adv_bucket)
    return combined[["date", "symbol", *CTX_COLUMNS]]


def _universe_symbols(root: Path) -> list[str]:
    bars5_dir = Path(root) / "bars5"
    if not bars5_dir.exists():
        return []
    return sorted({p.stem.upper() for p in bars5_dir.glob("*.parquet")})


def build_universe_context(root: Path, spy_symbol: str = "SPY") -> Path:
    """Writes <root>/context/universe_daily.parquet from every bars5/*.parquet symbol except SPY
    and VIX. A single symbol's failure (or a symbol that produces no rows, e.g. bars5 present but
    no daily file) is logged and skipped, never raised. SPY itself is loaded through the same
    split-adjustment path as every other symbol (`load_adjusted`), since rs_* compares each
    symbol's close series against SPY's directly; a missing/empty SPY frame is a build-wide
    precondition failure (every symbol's rs_* would silently be all-NaN otherwise), so it raises
    rather than being swallowed like a per-symbol failure."""
    from research.strategy_hunt.data.earnings import load_earnings_table
    from research.strategy_hunt.data.news import load_news
    from research.strategy_hunt.data.splits import load_adjusted

    root = Path(root)
    spy_symbol = spy_symbol.upper()
    _, spy_daily = load_adjusted(root, spy_symbol)
    if spy_daily.empty:
        raise ValueError(
            f"SPY daily data missing/empty for {spy_symbol!r} under {root} "
            "(every symbol's rs_* columns depend on it)"
        )
    skip = {spy_symbol, "VIX"}

    frames: dict[str, pd.DataFrame] = {}
    for symbol in _universe_symbols(root):
        if symbol in skip:
            continue
        try:
            bars5, daily = load_adjusted(root, symbol)
            earnings_table = load_earnings_table(root, symbol)
            news = load_news(root, symbol)
            frame = symbol_daily_features(symbol, bars5, daily, spy_daily, earnings_table, news)
        except Exception as exc:  # per-symbol failures must never abort the build
            logger.warning("context build failed for %s: %s", symbol, exc)
            continue
        if frame.empty:
            logger.warning("context build produced no rows for %s (missing/empty daily data?)", symbol)
            continue
        frames[symbol] = frame

    combined = rank_cross_section(frames)
    out_path = root / "context" / "universe_daily.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.parquet")
    combined.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return out_path


def load_context_slice(root: Path, symbol: str) -> pd.DataFrame:
    """One symbol's context, indexed by date. Empty (but columned) frame when the file or the
    symbol is missing."""
    path = Path(root) / "context" / "universe_daily.parquet"
    if not path.exists():
        empty = pd.DataFrame(columns=list(CTX_COLUMNS))
        empty.index = pd.DatetimeIndex([], name="date")
        return empty
    df = pd.read_parquet(path, engine="pyarrow", filters=[("symbol", "==", symbol.upper())])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df[list(CTX_COLUMNS)]
