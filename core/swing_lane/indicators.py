"""SPEC §3 indicators (ported verbatim from review/swing_system/swing_lib.py)."""
from __future__ import annotations

import pandas as pd


def compute(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df.close, df.high, df.low
    o = pd.DataFrame(index=df.index)
    o["sma50"] = c.rolling(50).mean()
    o["sma200"] = c.rolling(200).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    o["atr"] = tr.rolling(14).mean()
    o["vol50"] = df.volume.rolling(50).mean().shift(1)
    o["dvol50"] = (c * df.volume).rolling(50).mean()
    o["hh20"] = h.rolling(20).max().shift(1)
    o["rng20"] = (h.rolling(20).max() - l.rolling(20).min()).shift(1)
    o["low5"] = l.rolling(5).min()
    o["rvol"] = df.volume / o.vol50
    o["gap"] = df.open / c.shift() - 1
    o["clv"] = (c - l) / (h - l).where(h != l)
    return o


def regime_on(spy: pd.DataFrame) -> bool:
    sma = spy.close.rolling(200).mean().iat[-1]
    return bool(pd.notna(sma) and spy.close.iat[-1] > sma)
