"""Daily regime labels using the same thresholds as core/market_context.py so research and live agree."""
from __future__ import annotations

import pandas as pd


def classify_regime(spy_change_pct: float, vix_level: float, vix_change_pct: float) -> str:
    if spy_change_pct > 0.3 and (vix_change_pct < 0 or vix_level < 16):
        return "risk_on"
    if spy_change_pct < -0.5 and vix_change_pct > 5:
        return "risk_off"
    if vix_change_pct > 10:
        return "risk_off"
    return "mixed"


def regime_series(spy_daily: pd.DataFrame, vix_daily: pd.DataFrame) -> pd.Series:
    spy = spy_daily["close"].astype(float)
    vix = vix_daily["close"].astype(float).reindex(spy.index).ffill()
    spy_chg = spy.pct_change() * 100
    vix_chg = vix.pct_change() * 100
    labels = []
    for d in spy.index:
        if pd.isna(spy_chg[d]) or pd.isna(vix_chg[d]) or pd.isna(vix[d]):
            labels.append("unknown")
        else:
            labels.append(classify_regime(float(spy_chg[d]), float(vix[d]), float(vix_chg[d])))
    out = pd.Series(labels, index=spy.index.normalize(), name="regime")
    out.index.name = "date"
    return out


def regime_at(series: pd.Series, ts: pd.Timestamp) -> str:
    key = pd.Timestamp(ts).normalize()
    return str(series[key]) if key in series.index else "unknown"
