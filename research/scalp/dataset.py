"""Adaptive Scalp Engine, S1: per-minute features and triple-barrier labels.

Spec: docs/superpowers/specs/2026-10-06-adaptive-scalp-engine-design.md §3–4.

Input: one symbol-session of minute bins (the archive's tick-rule bins or the
live book's flush) with columns t_open, buy_vol, sell_vol, vol, n_trades,
high, low, close, vwap, large_vol, large_buy, large_sell.

`features(bins)` → one row per minute, all causal (nothing from minute t+1).
`labels(bins, tau, sigma, horizon, cost)` → for each minute and side the
first-touch outcome within the horizon and the realised R of the fixed rule,
with the cost charged inside the label. `remaining_r(...)` → for the exit
policy: at every minute of a hypothetical open position, the R still to come.
Pure numpy; no I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "cvd_1", "cvd_3", "cvd_5", "cvd_15", "sell_share_5", "sell_share_15", "large_bias_5", "large_bias_15",
    "print_rate_rel", "ret_1", "ret_3", "ret_5", "ret_15", "rvol_5", "vwap15_dist", "svwap_dist", "range_pos_15",
    "minute_of_day", "has_book", "imb_ema", "spread_z",
)
GRID_TAU = (0.002, 0.003, 0.005)
GRID_H = (10, 20, 30)
SIGMA_FRAC = 2.0 / 3.0
DEFAULT_COST = 0.0005      # per side: half-spread proxy + slippage, as a fraction of price


def _roll_sum(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = c[w:] - c[:-w]
    return np.concatenate([np.full(w - 1, np.nan), out])


def _roll_max(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=1).max().to_numpy()


def _roll_min(x: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(x).rolling(w, min_periods=1).min().to_numpy()


def features(bins: pd.DataFrame, print_rate_median: float | None = None) -> pd.DataFrame:
    """One causal feature row per minute (NaN where the window is not full)."""
    b = bins.sort_values("t_open").reset_index(drop=True)
    close = b["close"].astype(float).to_numpy()
    buy, sell = b["buy_vol"].astype(float).to_numpy(), b["sell_vol"].astype(float).to_numpy()
    vol = b["vol"].astype(float).to_numpy()
    delta = buy - sell
    pv = (b["vwap"].astype(float).fillna(b["close"]).to_numpy() * vol)
    f = pd.DataFrame(index=b.index)
    for w in (1, 3, 5, 15):
        f[f"cvd_{w}"] = _roll_sum(delta, w) / np.maximum(_roll_sum(vol, w), 1.0)
    for w in (5, 15):
        s, bb = _roll_sum(sell, w), _roll_sum(buy, w)
        f[f"sell_share_{w}"] = s / np.maximum(s + bb, 1.0)
        lb, ls, lv = _roll_sum(b["large_buy"].astype(float).to_numpy(), w), _roll_sum(b["large_sell"].astype(float).to_numpy(), w), _roll_sum(b["large_vol"].astype(float).to_numpy(), w)
        f[f"large_bias_{w}"] = (lb - ls) / np.maximum(lv, 1.0)
    n = b["n_trades"].astype(float).to_numpy()
    med = print_rate_median if print_rate_median and print_rate_median > 0 else max(np.nanmedian(n), 1.0)
    f["print_rate_rel"] = np.log1p(_roll_sum(n, 5) / 5.0) - np.log1p(med)
    lc = np.log(np.maximum(close, 1e-9))
    for w in (1, 3, 5, 15):
        f[f"ret_{w}"] = np.concatenate([np.full(w, np.nan), lc[w:] - lc[:-w]])
    r1 = np.diff(lc, prepend=np.nan)
    f["rvol_5"] = pd.Series(r1).rolling(5).std().to_numpy()
    vwap15 = _roll_sum(pv, 15) / np.maximum(_roll_sum(vol, 15), 1.0)
    f["vwap15_dist"] = close / np.where(vwap15 > 0, vwap15, np.nan) - 1.0
    svwap = np.cumsum(pv) / np.maximum(np.cumsum(vol), 1.0)
    f["svwap_dist"] = close / np.where(svwap > 0, svwap, np.nan) - 1.0
    hi15, lo15 = _roll_max(b["high"].astype(float).to_numpy(), 15), _roll_min(b["low"].astype(float).to_numpy(), 15)
    f["range_pos_15"] = (close - lo15) / np.where(hi15 > lo15, hi15 - lo15, np.nan)
    t = pd.to_datetime(b["t_open"])
    f["minute_of_day"] = (t.dt.hour * 60 + t.dt.minute - (9 * 60 + 30)).to_numpy()
    f["has_book"] = 1.0 if "imb_ema" in b.columns else 0.0
    f["imb_ema"] = b["imb_ema"].astype(float).to_numpy() if "imb_ema" in b.columns else np.nan
    f["spread_z"] = b["spread_z"].astype(float).to_numpy() if "spread_z" in b.columns else np.nan
    f.insert(0, "t_open", t.to_numpy())
    f["close"] = close
    return f


@dataclass(frozen=True)
class BarrierSpec:
    tau: float
    horizon: int
    cost: float = DEFAULT_COST

    @property
    def sigma(self) -> float:
        return self.tau * SIGMA_FRAC

    @property
    def name(self) -> str:
        return f"t{int(round(self.tau * 1e4))}_h{self.horizon}"


def first_touch(high: np.ndarray, low: np.ndarray, close: np.ndarray, i: int, side: int,
                spec: BarrierSpec) -> Tuple[int, int, float]:
    """Outcome of a position opened at minute i's close (crossing the spread)
    and judged on minutes i+1..i+H: (y, exit_index, realised R). y = +1 target
    first, −1 stop first (a minute that touches both counts as the stop), 0
    neither (exit at the horizon close). R is net of cost, in units of the
    stop distance."""
    entry = close[i] * (1 + side * spec.cost)
    target = entry * (1 + side * spec.tau)
    stop = entry * (1 - side * spec.sigma)
    n = len(close)
    end = min(n - 1, i + spec.horizon)
    for k in range(i + 1, end + 1):
        hit_stop = low[k] <= stop if side > 0 else high[k] >= stop
        hit_target = high[k] >= target if side > 0 else low[k] <= target
        if hit_stop:
            return -1, k, _r(entry, stop, side, spec)
        if hit_target:
            return 1, k, _r(entry, target, side, spec)
    exit_px = close[end]
    return 0, end, _r(entry, exit_px, side, spec)


def _r(entry: float, exit_px: float, side: int, spec: BarrierSpec) -> float:
    exit_net = exit_px * (1 - side * spec.cost)
    return side * (exit_net - entry) / (entry * spec.sigma)


def labels(bins: pd.DataFrame, spec: BarrierSpec) -> pd.DataFrame:
    """Per minute, both sides: y_long, r_long, exit_long, y_short, r_short,
    exit_short. Vectorised: an (n x H) matrix of barrier touches per side."""
    b = bins.sort_values("t_open").reset_index(drop=True)
    high, low, close = (b[c].astype(float).to_numpy() for c in ("high", "low", "close"))
    n, H = len(b), spec.horizon
    out = {}
    for side, tag in ((1, "long"), (-1, "short")):
        entry = close * (1 + side * spec.cost)
        target = entry * (1 + side * spec.tau)
        stop = entry * (1 - side * spec.sigma)
        idx = np.arange(n)[:, None] + np.arange(1, H + 1)[None, :]          # (n, H) future minute indices
        valid = idx < n
        idx_c = np.minimum(idx, n - 1)
        hi, lo = high[idx_c], low[idx_c]
        if side > 0:
            hit_stop, hit_target = (lo <= stop[:, None]) & valid, (hi >= target[:, None]) & valid
        else:
            hit_stop, hit_target = (hi >= stop[:, None]) & valid, (lo <= target[:, None]) & valid
        first_stop = np.where(hit_stop.any(1), hit_stop.argmax(1), H)        # offset-1 of first touch, H = never
        first_target = np.where(hit_target.any(1), hit_target.argmax(1), H)
        last_valid = np.minimum(np.arange(n) + H, n - 1)                      # horizon close index
        y = np.where(first_stop <= first_target, np.where(first_stop < H, -1, 0), 1)
        k = np.where(y == -1, np.arange(n) + first_stop + 1, np.where(y == 1, np.arange(n) + first_target + 1, last_valid))
        k = np.minimum(k, n - 1)
        exit_px = np.where(y == -1, stop, np.where(y == 1, target, close[k]))
        r = side * (exit_px * (1 - side * spec.cost) - entry) / (entry * spec.sigma)
        y = y.astype(float); r = r.astype(float); kf = k.astype(float)
        y[-1] = r[-1] = kf[-1] = np.nan                                       # no future bar after the last minute
        out[f"y_{tag}"], out[f"r_{tag}"], out[f"exit_{tag}"] = y, r, kf
    return pd.DataFrame(out)


def remaining_r(bins: pd.DataFrame, spec: BarrierSpec, side: int, entry_i: int) -> pd.DataFrame:
    """For a position opened at `entry_i` under `spec`: at each later minute m
    until the fixed rule's exit, the R still to come (exit R minus the
    mark-to-market R at m) plus the position-state features."""
    b = bins.sort_values("t_open").reset_index(drop=True)
    high, low, close = (b[c].astype(float).to_numpy() for c in ("high", "low", "close"))
    y, k_exit, r_exit = first_touch(high, low, close, entry_i, side, spec)
    entry = close[entry_i] * (1 + side * spec.cost)
    rows = []
    mfe = mae = 0.0
    for m in range(entry_i + 1, k_exit + 1):
        mtm = side * (close[m] * (1 - side * spec.cost) - entry) / (entry * spec.sigma)
        exc_hi = side * (high[m] - entry) / (entry * spec.sigma) if side > 0 else side * (low[m] - entry) / (entry * spec.sigma)
        exc_lo = side * (low[m] - entry) / (entry * spec.sigma) if side > 0 else side * (high[m] - entry) / (entry * spec.sigma)
        mfe, mae = max(mfe, exc_hi), min(mae, exc_lo)
        rows.append({"m": m, "minutes_held": m - entry_i, "mtm_r": mtm, "mfe_r": mfe, "mae_r": mae,
                     "remaining_r": r_exit - mtm, "y": y, "exit_m": k_exit})
    return pd.DataFrame(rows)


def build_session(bins: pd.DataFrame, symbol: str, specs: Iterable[BarrierSpec] | None = None,
                  print_rate_median: float | None = None) -> pd.DataFrame:
    """Features + labels for every spec, one frame per session."""
    specs = list(specs) if specs is not None else [BarrierSpec(t, h) for t in GRID_TAU for h in GRID_H]
    f = features(bins, print_rate_median)
    f.insert(1, "symbol", symbol)
    for spec in specs:
        lab = labels(bins, spec)
        for c in lab.columns:
            f[f"{c}@{spec.name}"] = lab[c].to_numpy()
    return f
