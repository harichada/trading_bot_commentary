"""v-turn-engine-2026-10-05: microstructure features for research.

Two producers feed the same 5-minute `mf_*` columns that `rsi_turn` reads off
the indicator frame:

* the trade archive (`trade_ticks`: ts, price, size, no quotes) — aggressor by
  the TICK rule, tagged `aggressor_rule = "tick"`;
* the live book's persisted minute bins (`data/micro/<SYM>/<date>.parquet`,
  written by `core.micro_book`) — QUOTE-rule aggressor, tagged "quote".

A model fitted on tick-rule features must say so; the tag travels with every
row so research and the forward ledger are never silently mixed.

Pure functions on DataFrames; the only I/O is `load_ticks` / `load_minute_bins`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

MF_COLUMNS = ("mf_buy_vol_15", "mf_sell_vol_15", "mf_vol_15", "mf_cvd_15", "mf_cvd_60", "mf_vol_60",
              "mf_sell_share_15", "mf_large_buy_15", "mf_large_sell_15", "mf_large_vol_15",
              "mf_n_trades_15", "mf_aggressor_rule")
LARGE_PRINT_PCTL = 95.0


def tag_aggressor_tick_rule(price: pd.Series) -> pd.Series:
    """+1 uptick, −1 downtick, flat inherits the previous side (vectorised:
    forward-fill of the non-zero signs)."""
    d = np.sign(price.diff().fillna(0.0).to_numpy())
    s = pd.Series(d, index=price.index).replace(0.0, np.nan).ffill().fillna(0.0)
    return s.astype(int)


def minute_bins_from_ticks(ticks: pd.DataFrame) -> pd.DataFrame:
    """One symbol's trade prints (`ts`, `price`, `size`, sorted) → one row per
    minute with the same fields the live book persists (`core.micro_book`
    flush), aggressor by the tick rule."""
    if ticks.empty:
        return pd.DataFrame(columns=["t_open", "buy_vol", "sell_vol", "vol", "n_buy", "n_sell", "n_trades",
                                     "vwap", "high", "low", "close", "large_vol", "large_buy", "large_sell",
                                     "aggressor_rule"])
    t = ticks.sort_values("ts").reset_index(drop=True)
    price = t["price"].astype(float)
    size = t["size"].astype(float)
    side = tag_aggressor_tick_rule(price)
    floor = float(np.percentile(size, LARGE_PRINT_PCTL)) if len(size) >= 20 else np.inf
    large = size.where(size >= floor, 0.0)
    minute = pd.to_datetime(t["ts"]).dt.floor("min")
    g = pd.DataFrame({
        "t_open": minute, "price": price, "size": size,
        "buy_vol": size.where(side > 0, 0.0), "sell_vol": size.where(side < 0, 0.0),
        "n_buy": (side > 0).astype(int), "n_sell": (side < 0).astype(int),
        "pv": price * size, "large_vol": large,
        "large_buy": large.where(side > 0, 0.0), "large_sell": large.where(side < 0, 0.0),
    }).groupby("t_open")
    out = g.agg(buy_vol=("buy_vol", "sum"), sell_vol=("sell_vol", "sum"), vol=("size", "sum"),
                n_buy=("n_buy", "sum"), n_sell=("n_sell", "sum"), n_trades=("size", "size"),
                pv=("pv", "sum"), high=("price", "max"), low=("price", "min"), close=("price", "last"),
                large_vol=("large_vol", "sum"), large_buy=("large_buy", "sum"), large_sell=("large_sell", "sum"))
    out["vwap"] = out["pv"] / out["vol"].replace(0, np.nan)
    out = out.drop(columns="pv").reset_index()
    out["aggressor_rule"] = "tick"
    return out


def features_5m(minute_bins: pd.DataFrame) -> pd.DataFrame:
    """Minute bins (either producer) → the `mf_*` columns at every 5-minute
    bar close, using trailing 15- and 60-minute sums. Index: the 5-minute bar
    timestamp (bar open, matching the research bars5 index)."""
    if minute_bins.empty:
        return pd.DataFrame(columns=list(MF_COLUMNS))
    m = minute_bins.copy()
    m["t_open"] = pd.to_datetime(m["t_open"])
    m = m.set_index("t_open").sort_index()
    for col in ("large_buy", "large_sell"):
        if col not in m.columns:
            m[col] = np.nan
    num_cols = ["buy_vol", "sell_vol", "vol", "n_trades", "large_vol", "large_buy", "large_sell"]
    m = m[num_cols].astype(float)
    full = m.reindex(pd.date_range(m.index.min(), m.index.max(), freq="min")).fillna(0.0)
    r15 = full.rolling("15min").sum()
    r60 = full.rolling("60min").sum()
    f = pd.DataFrame(index=full.index)
    f["mf_buy_vol_15"] = r15["buy_vol"]
    f["mf_sell_vol_15"] = r15["sell_vol"]
    f["mf_vol_15"] = r15["vol"]
    f["mf_cvd_15"] = r15["buy_vol"] - r15["sell_vol"]
    f["mf_cvd_60"] = r60["buy_vol"] - r60["sell_vol"]
    f["mf_vol_60"] = r60["vol"]
    sides = r15["buy_vol"] + r15["sell_vol"]
    f["mf_sell_share_15"] = (r15["sell_vol"] / sides.replace(0, np.nan)).fillna(0.5)
    f["mf_large_buy_15"] = r15["large_buy"]
    f["mf_large_sell_15"] = r15["large_sell"]
    f["mf_large_vol_15"] = r15["large_vol"]
    f["mf_n_trades_15"] = r15["n_trades"]
    rule = str(minute_bins["aggressor_rule"].iloc[0]) if "aggressor_rule" in minute_bins.columns else "tick"
    f["mf_aggressor_rule"] = rule
    # keep the value at the END of each 5-minute bar, indexed by the bar's open
    ends = f[(f.index.minute % 5) == 4]
    ends.index = ends.index - pd.Timedelta(minutes=4)
    return ends[list(MF_COLUMNS)]


def join_micro(ind: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Left-join the `mf_*` columns onto an indicator frame (bars5 index). Bars
    without micro data get NaN numeric columns and an empty rule, which the
    family treats as "no tape" (the A component contributes nothing)."""
    out = ind.copy()
    if feats is None or feats.empty:
        for c in MF_COLUMNS:
            out[c] = "" if c == "mf_aggressor_rule" else np.nan
        return out
    joined = out.join(feats.reindex(out.index), how="left")
    joined["mf_aggressor_rule"] = joined["mf_aggressor_rule"].fillna("")
    return joined


# ── I/O ───────────────────────────────────────────────────────────────

def load_ticks(engine: Any, symbol: str, day: str) -> pd.DataFrame:
    """One symbol-day of trade prints from `trade_ticks` (regular session,
    exchange timestamps as stored). The table is unindexed; callers batch by
    day and run this off-hours."""
    from sqlalchemy import text
    q = text("SELECT ts, price, size FROM trade_ticks WHERE symbol = :s AND ts >= :d0 AND ts < :d1 ORDER BY ts")
    d0 = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
    d1 = pd.Timestamp(day).tz_localize("America/New_York") + pd.Timedelta(hours=16)
    with engine.connect() as c:
        df = pd.read_sql(q, c, params={"s": symbol.upper(), "d0": d0, "d1": d1})
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(float)
    return df


def load_minute_bins(root: Path, symbol: str) -> pd.DataFrame:
    """Every persisted session of the live book for `symbol`, concatenated."""
    d = Path(root) / symbol.upper()
    if not d.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in sorted(d.glob("*.parquet"))]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["t_open"] = pd.to_datetime(out["t_open"])
    if getattr(out["t_open"].dt, "tz", None) is not None:
        out["t_open"] = out["t_open"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    return out.sort_values("t_open").reset_index(drop=True)
