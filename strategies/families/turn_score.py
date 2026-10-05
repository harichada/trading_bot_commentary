"""v-turn-score-2026-10-05: the Adaptive RSI Turn score — pure functions.

Spec: docs/superpowers/specs/2026-10-05-adaptive-rsi-turn-design.md §3.

Four explainable components, each in [0, 1], combined by convex weights into a
0–100 score. No I/O, no state: research and live call the same functions, so
the two cannot drift. Every function clamps its output, so the score is in
range for any finite input; NaN inputs degrade to 0 for that component.

    Z  dynamic oversold zone   — how far the daily RSI sits below the symbol's
                                 own trailing 5th percentile (0 outside the zone)
    M  RSI momentum/divergence — one-sided: rising 15-minute RSI, an inflection,
                                 a bullish divergence
    A  tape absorption         — sellers dominate the tape but price holds;
                                 cumulative volume delta turning positive
    B  Level 1 book pressure   — EMA bid/ask-size imbalance, zero when the
                                 book is thin or the spread is wide
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

PRIOR_WEIGHTS: Mapping[str, float] = {"Z": 0.30, "M": 0.25, "A": 0.25, "B": 0.20}
ZONE_PERCENTILE = 5.0
ZONE_MIN_SESSIONS = 120
ZONE_FALLBACK_Q05 = 25.0
DEPTH_CAP_IQR = 1.5
SLOPE_SIGMA_FLOOR = 2.0
SPREAD_Z_VETO = 3.0
BBO_SIZE_MULT = 2.0


def _finite(x) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def clip01(x) -> float:
    v = _finite(x)
    return 0.0 if v is None else min(1.0, max(0.0, v))


def clip11(x) -> float:
    v = _finite(x)
    return 0.0 if v is None else min(1.0, max(-1.0, v))


# ── Z: dynamic oversold zone ───────────────────────────────────────────

@dataclass(frozen=True)
class ZoneStats:
    q05: float
    q25: float
    q50: float
    q75: float
    n: int

    @property
    def iqr(self) -> float:
        return max(self.q75 - self.q25, 1e-6)


def zone_stats(rsi_history: Sequence[float], window: int = 252) -> ZoneStats:
    """Percentiles of the trailing `window` daily RSI values (the ones BEFORE
    today). Fewer than ZONE_MIN_SESSIONS usable values → the fixed fallback
    bar (q05 = 25) with a wide IQR, so depth stays modest."""
    arr = np.asarray([v for v in rsi_history if _finite(v) is not None], dtype=float)
    arr = arr[-window:]
    if arr.size < ZONE_MIN_SESSIONS:
        return ZoneStats(ZONE_FALLBACK_Q05, 40.0, 50.0, 60.0, int(arr.size))
    q05, q25, q50, q75 = np.percentile(arr, [ZONE_PERCENTILE, 25, 50, 75])
    return ZoneStats(float(q05), float(q25), float(q50), float(q75), int(arr.size))


def zone_component(rsi_daily, stats: ZoneStats) -> Tuple[bool, float]:
    """(in_zone, Z). Z = clip((q05 − r) / IQR, 0, 1.5) / 1.5."""
    r = _finite(rsi_daily)
    if r is None or r > stats.q05:
        return False, 0.0
    depth = min(DEPTH_CAP_IQR, max(0.0, (stats.q05 - r) / stats.iqr)) / DEPTH_CAP_IQR
    return True, depth


# ── M: RSI momentum and divergence ────────────────────────────────────

def momentum_component(rsi_15m: Sequence[float], lows_15m: Sequence[float],
                       slope_sigma: float, lookback: int = 20) -> Tuple[float, Mapping[str, float]]:
    """15-minute RSI series and bar lows, oldest first; the last element is now.
    slope = ρ_t − ρ_{t−3}; m_slope = clip(slope / (2σ), −1, 1), one-sided;
    inflect = slope_t > 0 and slope_{t−1} ≤ 0; divergence = price lower low
    over `lookback` bars while RSI made a higher low.
    M = clip(0.5·max(m_slope,0) + 0.3·inflect + 0.2·div, 0, 1)."""
    rho = np.asarray([_finite(v) if _finite(v) is not None else np.nan for v in rsi_15m], dtype=float)
    lows = np.asarray([_finite(v) if _finite(v) is not None else np.nan for v in lows_15m], dtype=float)
    parts = {"slope": 0.0, "m_slope": 0.0, "inflect": 0.0, "div": 0.0}
    if rho.size < 5 or np.isnan(rho[-4:]).any():
        return 0.0, parts
    sigma = max(_finite(slope_sigma) or 0.0, SLOPE_SIGMA_FLOOR)
    slope = float(rho[-1] - rho[-4])
    slope_prev = float(rho[-2] - rho[-5]) if rho.size >= 5 and not np.isnan(rho[-5]) else 0.0
    m_slope = clip11(slope / (2.0 * sigma))
    inflect = 1.0 if slope > 0 and slope_prev <= 0 else 0.0
    div = 0.0
    n = min(lookback, rho.size, lows.size)
    if n >= 12 and not np.isnan(lows[-n:]).any() and not np.isnan(rho[-n:]).any():
        p_now, p_prev = float(np.min(lows[-6:])), float(np.min(lows[-n:-6]))
        r_now, r_prev = float(np.min(rho[-6:])), float(np.min(rho[-n:-6]))
        div = 1.0 if (p_now < p_prev and r_now > r_prev) else 0.0
    parts.update(slope=slope, m_slope=m_slope, inflect=inflect, div=div)
    return clip01(0.5 * max(m_slope, 0.0) + 0.3 * inflect + 0.2 * div), parts


# ── A: tape absorption ────────────────────────────────────────────────

def absorption_component(buy_vol_15, sell_vol_15, ret_15, atr_15_frac, cvd_15, cvd_60, vol_15,
                         large_buy_15, large_sell_15, large_vol_15) -> Tuple[float, Mapping[str, float]]:
    """sell_share = sell/(buy+sell); price_hold = 1 − clip(−ret/ATR, 0, 1);
    absorb = clip((sell_share − 0.5)/0.3, 0, 1)·price_hold;
    cvd_turn = 1[CVD_15 > 0 and CVD_60 < 0]; cvd_slope = clip(CVD_15/vol_15, −1, 1);
    large_bias = clip((large_buy − large_sell)/large_vol, −1, 1);
    A = clip(0.5·absorb + 0.3·cvd_turn + 0.2·max(cvd_slope,0) + 0.1·max(large_bias,0), 0, 1)."""
    b, s = _finite(buy_vol_15) or 0.0, _finite(sell_vol_15) or 0.0
    sides = b + s
    sell_share = s / sides if sides > 0 else 0.5
    ret = _finite(ret_15) or 0.0
    atr = _finite(atr_15_frac) or 0.0
    price_hold = 1.0 - clip01(-ret / atr) if atr > 0 else (1.0 if ret >= 0 else 0.0)
    absorb = clip01((sell_share - 0.5) / 0.3) * price_hold
    c15, c60 = _finite(cvd_15) or 0.0, _finite(cvd_60) or 0.0
    cvd_turn = 1.0 if (c15 > 0 and c60 < 0) else 0.0
    v15 = _finite(vol_15) or 0.0
    cvd_slope = clip11(c15 / v15) if v15 > 0 else 0.0
    lv = _finite(large_vol_15) or 0.0
    large_bias = clip11(((_finite(large_buy_15) or 0.0) - (_finite(large_sell_15) or 0.0)) / lv) if lv > 0 else 0.0
    a = clip01(0.5 * absorb + 0.3 * cvd_turn + 0.2 * max(cvd_slope, 0.0) + 0.1 * max(large_bias, 0.0))
    return a, {"sell_share_15": sell_share, "price_hold": price_hold, "absorb": absorb,
               "cvd_turn": cvd_turn, "cvd_slope": cvd_slope, "large_bias": large_bias}


# ── B: Level 1 book pressure ──────────────────────────────────────────

def book_component(imb_ema, spread_ema, spread_baseline, spread_mad, bid_size, ask_size,
                   median_bbo_size) -> Tuple[float, Mapping[str, float], bool]:
    """B = liq_ok · (IMB_ema + 1)/2, with liq_ok = spread_z ≤ 3 and
    bid_size + ask_size ≥ 2·median BBO size. Returns (B, parts, liq_ok);
    liq_ok False is also a veto for the caller."""
    imb = clip11(imb_ema)
    spr = _finite(spread_ema)
    base = _finite(spread_baseline)
    mad = _finite(spread_mad)
    if spr is None or base is None or mad is None or mad <= 0:
        spread_z = 0.0 if spr is not None else float("inf")
    else:
        spread_z = (spr - base) / mad
    bbo = (_finite(bid_size) or 0.0) + (_finite(ask_size) or 0.0)
    med = _finite(median_bbo_size) or 0.0
    liq_ok = bool(spread_z <= SPREAD_Z_VETO and (med <= 0 or bbo >= BBO_SIZE_MULT * med) and bbo > 0)
    b = clip01((imb + 1.0) / 2.0) if liq_ok else 0.0
    return b, {"imb": imb, "spread_z": spread_z, "bbo_size": bbo, "liq_ok": float(liq_ok)}, liq_ok


# ── composite ─────────────────────────────────────────────────────────

def normalise_weights(weights: Mapping[str, float]) -> Mapping[str, float]:
    """Non-negative, sum to one over Z, M, A, B; falls back to the prior when
    the input is unusable (negative, NaN, all zero)."""
    vals = {k: max(0.0, _finite(weights.get(k, 0.0)) or 0.0) for k in ("Z", "M", "A", "B")}
    tot = sum(vals.values())
    if tot <= 0:
        return dict(PRIOR_WEIGHTS)
    return {k: v / tot for k, v in vals.items()}


def composite_score(components: Mapping[str, float], weights: Optional[Mapping[str, float]] = None) -> float:
    """S = 100·Σ w_i·c_i ∈ [0, 100]."""
    w = normalise_weights(weights or PRIOR_WEIGHTS)
    s = sum(w[k] * clip01(components.get(k, 0.0)) for k in ("Z", "M", "A", "B"))
    return float(min(100.0, max(0.0, 100.0 * s)))


def size_scale(score: float, theta: float) -> float:
    """0.5 at the threshold, 1.0 at 100; 0 below the threshold."""
    s, t = _finite(score) or 0.0, _finite(theta) or 0.0
    if s < t:
        return 0.0
    if t >= 100.0:
        return 1.0
    return float(min(1.0, 0.5 + 0.5 * (s - t) / (100.0 - t)))


def thesis_failed(score: float, theta: float, cvd_60: float, sessions_held: int) -> bool:
    """Early invalidation: within the first 2 sessions, S < θ − 20 and CVD_60 < 0."""
    s, t = _finite(score), _finite(theta)
    if s is None or t is None:
        return False
    return sessions_held <= 2 and s < t - 20.0 and (_finite(cvd_60) or 0.0) < 0
