"""`rsi_turn` family (v-turn-engine-2026-10-05): the Adaptive RSI Turn score as
a strategy family, so research (walk-forward / pooled fixed-rule) and the live
fit lane run the same code.

Spec: docs/superpowers/specs/2026-10-05-adaptive-rsi-turn-design.md §3–4.

Per 5-minute bar, for symbols whose lagged daily RSI(14) is below the symbol's
own trailing 5th percentile (`strategies.families.turn_score.zone_component`):
  Z  zone depth,  M  15-minute RSI momentum/divergence,  A  tape absorption
  (from the `mf_*` columns on `ind`, NaN → 0),  B  Level 1 book pressure (from
  `mf_imb_ema` etc. when the live book provides them; the archive has no
  quotes, so B = 0 there and the liquidity veto is off).
Entry when the composite ≥ `theta` and no veto; stop `stop_atr` × daily ATR14,
target the 20-day mean (≥ 1R away), `time_stop_days` sessions. Weights come
from `params["weights"]` (a dict) or the prior.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from strategies.families import turn_score as ts
from strategies.families.common import Family, make_entry, spaced_by_sessions
from strategies.families.models import Entry

SLOPE_SIGMA_SESSIONS = 20
BARS_15M = 3          # 5-minute bars per 15-minute bar
DIV_LOOKBACK = 20


def _rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder RSI, vectorised (EMA with alpha = 1/n), NaN until warm."""
    c = pd.Series(close, dtype=float)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    out = (100 - 100 / (1 + rs)).to_numpy()
    out[:n] = np.nan
    return out


def bars_15m(bars5: pd.DataFrame) -> pd.DataFrame:
    """15-minute bars from 5-minute bars, regular session, each labelled by its
    open; the RSI and ATR series the M component reads."""
    b = bars5[["open", "high", "low", "close"]]
    o = b["open"].resample("15min", origin="start_day", offset="9h30min").first()
    h = b["high"].resample("15min", origin="start_day", offset="9h30min").max()
    lo = b["low"].resample("15min", origin="start_day", offset="9h30min").min()
    c = b["close"].resample("15min", origin="start_day", offset="9h30min").last()
    out = pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}).dropna(subset=["close"])
    out["rsi"] = _rsi(out["close"].to_numpy())
    tr = np.maximum(out["high"] - out["low"],
                    np.maximum((out["high"] - out["close"].shift()).abs(), (out["low"] - out["close"].shift()).abs()))
    out["atr_frac"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / out["close"]
    out["ret_15"] = out["close"].pct_change()
    return out


def _slope_sigma(rsi15: np.ndarray, sessions: int = SLOPE_SIGMA_SESSIONS) -> float:
    s = pd.Series(rsi15).diff(BARS_15M).dropna()
    s = s.tail(sessions * 26)      # ~26 fifteen-minute bars per session
    return float(max(s.std(), ts.SLOPE_SIGMA_FLOOR)) if len(s) > 5 else ts.SLOPE_SIGMA_FLOOR


def score_bar(ind_row: Mapping[str, Any], zone: ts.ZoneStats, rsi15: np.ndarray, lows15: np.ndarray,
              slope_sigma: float, ret_15: float, atr_frac: float, weights: Optional[Mapping[str, float]]
              ) -> tuple[bool, float, dict, tuple[str, ...]]:
    """One bar → (in_zone, score, components+raw, vetoes). Pure."""
    in_zone, z = ts.zone_component(ind_row.get("d_rsi14"), zone)
    if not in_zone:
        return False, 0.0, {"Z": 0.0}, ()
    m, mparts = ts.momentum_component(rsi15, lows15, slope_sigma, DIV_LOOKBACK)
    g = lambda k: ind_row.get(k, np.nan)
    a, aparts = ts.absorption_component(g("mf_buy_vol_15"), g("mf_sell_vol_15"), ret_15, atr_frac,
                                        g("mf_cvd_15"), g("mf_cvd_60"), g("mf_vol_15"),
                                        g("mf_large_buy_15"), g("mf_large_sell_15"), g("mf_large_vol_15"))
    has_quotes = ts._finite(g("mf_imb_ema")) is not None
    if has_quotes:
        b, bparts, liq_ok = ts.book_component(g("mf_imb_ema"), g("mf_spread_ema"), g("mf_spread_baseline"),
                                              g("mf_spread_mad"), g("mf_bid_size"), g("mf_ask_size"),
                                              g("mf_median_bbo_size"))
    else:
        b, bparts, liq_ok = 0.0, {"imb": 0.0, "spread_z": 0.0, "bbo_size": 0.0, "liq_ok": 1.0}, True
    comps = {"Z": z, "M": m, "A": a, "B": b}
    score = ts.composite_score(comps, weights)
    vetoes = () if liq_ok else ("liquidity",)
    detail = {**comps, **{f"m_{k}": v for k, v in mparts.items()}, **{f"a_{k}": v for k, v in aparts.items()},
              **{f"b_{k}": v for k, v in bparts.items()}, "rule": str(ind_row.get("mf_aggressor_rule", "") or "")}
    return True, score, detail, vetoes


ROW_COLUMNS = ("d_rsi14", "d_atr14", "d_bb_mid", "mf_buy_vol_15", "mf_sell_vol_15", "mf_vol_15", "mf_cvd_15",
               "mf_cvd_60", "mf_large_buy_15", "mf_large_sell_15", "mf_large_vol_15", "mf_imb_ema", "mf_spread_ema",
               "mf_spread_baseline", "mf_spread_mad", "mf_bid_size", "mf_ask_size", "mf_median_bbo_size",
               "mf_aggressor_rule")
MIN_RSI_HISTORY = 60


def _row_columns(ind: pd.DataFrame) -> dict[str, np.ndarray]:
    """The columns `score_bar` reads, as arrays (absent columns stay absent so
    `row.get` falls back exactly as it did on a pandas row); `d_rsi14` is
    always present (NaN when missing) because the zone gate reads it."""
    cols = {c: ind[c].to_numpy() for c in ROW_COLUMNS if c in ind.columns}
    if "d_rsi14" not in cols:
        cols["d_rsi14"] = np.full(len(ind), np.nan)
    else:
        cols["d_rsi14"] = pd.to_numeric(ind["d_rsi14"], errors="coerce").to_numpy(dtype=float)
    return cols


def _zone_by_day(daily_rsi: pd.Series, bars_index: pd.DatetimeIndex) -> dict:
    """Zone stats per session: the trailing-252 percentiles of the daily RSI
    values BEFORE that day (so today's own RSI never gates today). Days with
    fewer than MIN_RSI_HISTORY prior values are left out (no zone)."""
    s = daily_rsi.sort_index()
    didx, vals = s.index, s.to_numpy(dtype=float)
    zones = {}
    for day in pd.DatetimeIndex(bars_index.normalize().unique()):
        n = int(didx.searchsorted(day, side="left"))
        if n < MIN_RSI_HISTORY:
            continue
        zones[day] = ts.zone_stats(vals[:n], 252)
    return zones


def rsi_turn_entries(bars5, ind, daily, news, params) -> list[Entry]:
    theta = float(params.get("theta", 60))
    stop_atr = float(params.get("stop_atr", 1.5))
    hold = int(params.get("time_stop_days", 10))
    weights = params.get("weights") or None
    if bars5.empty or ind.empty or daily is None or daily.empty or "close" not in daily.columns:
        return []
    daily_rsi = pd.Series(_rsi(daily["close"].to_numpy()), index=daily.index)
    b15 = bars_15m(bars5)
    if b15.empty:
        return []
    rsi15_all = b15["rsi"].to_numpy()
    lows15_all = b15["low"].to_numpy()
    ret15_all = b15["ret_15"].to_numpy()
    atrf_all = b15["atr_frac"].to_numpy()
    b15_pos = b15.index.get_indexer(bars5.index.floor("15min") - pd.Timedelta(0))
    sigma = _slope_sigma(rsi15_all)
    out: list[Entry] = []
    closes5 = bars5["close"].to_numpy()
    # v-turn-perf-2026-10-07: the live lane hands this family ~100k bars (the
    # research history spliced under the provider window), so the per-bar work
    # must be O(in-zone bars), not O(bars). Zone stats are per session (one
    # percentile pass per day, not per bar), the zone gate is vectorised, and
    # `ind` is read as column arrays — `ind.iloc[i]` per bar cost ~60 s per
    # symbol and stalled the whole process (GIL) while it ran.
    cols = _row_columns(ind)
    d_rsi = cols["d_rsi14"]
    zones = _zone_by_day(daily_rsi, bars5.index)
    q05 = np.array([zones[d].q05 if d in zones else np.nan for d in bars5.index.normalize()], dtype=float)
    candidates = np.flatnonzero((b15_pos >= 0) & np.isfinite(d_rsi) & np.isfinite(q05) & (d_rsi <= q05))
    for i in candidates:
        j = b15_pos[i]
        if i + 1 >= len(bars5):
            continue
        row = {k: v[i] for k, v in cols.items()}
        zone = zones[bars5.index[i].normalize()]
        in_zone, score, detail, vetoes = score_bar(
            row, zone, rsi15_all[max(0, j - DIV_LOOKBACK - 1):j + 1], lows15_all[max(0, j - DIV_LOOKBACK - 1):j + 1],
            sigma, float(ret15_all[j]) if not np.isnan(ret15_all[j]) else 0.0,
            float(atrf_all[j]) if not np.isnan(atrf_all[j]) else 0.0, weights)
        if in_zone:
            # Side channel for the live adapter (ledger factor block, exit
            # verdict cache). bars5.attrs is already how the symbol travels.
            # Only the CURRENT bar's verdict is kept: pandas deep-copies attrs on
            # every column access, so a growing dict would make a research run
            # quadratic (review round 1). The live adapter reads only now_ts.
            bars5.attrs["rsi_turn_verdicts"] = {bars5.index[i]: {
                "score": score, "theta": theta, "vetoes": list(vetoes),
                "cvd_60": ts._finite(row.get("mf_cvd_60")) or 0.0,
                "has_quotes": ts._finite(row.get("mf_imb_ema")) is not None, **detail}}
        if not in_zone or vetoes or score < theta:
            continue
        atr = ts._finite(row.get("d_atr14"))
        mean20 = ts._finite(row.get("d_bb_mid"))
        if atr is None or mean20 is None or atr <= 0:
            continue
        px = float(closes5[i])
        dist = stop_atr * atr
        if mean20 - px < dist:
            continue
        e = make_entry(bars5, i, "long", px - dist, mean20, hold, "multiday", "rsi_turn")
        if e:
            out.append(e)
    return spaced_by_sessions(out, hold)


FAMILIES_RSI_TURN: tuple[Family, ...] = (
    # 3 x 3 x 3 = 27 combos; median = theta 60 / stop 1.5 / 10 sessions.
    Family("rsi_turn", rsi_turn_entries,
           {"theta": [50, 60, 70], "stop_atr": [1.0, 1.5, 2.5], "time_stop_days": [5, 10, 15]},
           "multiday", ("long",)),
)
