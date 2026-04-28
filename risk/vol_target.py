"""Volatility-targeted position sizing per López de Prado, AFML §10.1.

Pure module — sizing math only. No I/O, no FastAPI, no Schwab. The
walk-forward harness (P9 runner) calls ``size_event`` per routed event
during per-event accumulation.

Methodology
-----------
* AFML §10.1 — constant-portfolio-vol bet sizing. Per-position size is
  ``w_i = (target_vol / N) / realized_vol_i``. With N concurrent
  positions each contributing ``target_vol / N``, the portfolio
  realized vol equals ``target_vol`` under independence.
* AFML §10.4 — caps prevent runaway sizing on stale or collapsing vol;
  a 5x parity cap is the spec floor.

Granularity bridge
------------------
Bars are 5-minute; the spec is in DAILY terms (0.75% daily ≈ 12%
annualized). We resample 5-min closes to daily-last-close and compute
log-returns on the daily series. AFML §10.1 works in daily-return
space natively; this avoids the sqrt(78) intraday-to-daily scaling
fudge that conflates microstructure noise with realized session vol.

Hooks
-----
``regime_size_multiplier`` and ``meta_p_win_multiplier`` are pure
pass-through scalars (default 1.0). They exist for future composability
with P1/P3 meta-models. With no passing meta-model in the current
stack, they default to the identity. The caller is responsible for
sourcing them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

DEFAULT_TARGET_DAILY_VOL: float = 0.0075        # 0.75% daily ≈ 11.9% annualized
DEFAULT_LOOKBACK_DAYS: int = 20                 # AFML §10.1 reference span
DEFAULT_FLOOR_PERCENTILE: float = 0.10          # 10th percentile of lookback
DEFAULT_CAP_MULT: float = 5.0                   # 5x vol-parity allocation cap


@dataclass(frozen=True)
class SizingDecision:
    """Single-event vol-target sizing output. Frozen per coding-style.md."""
    symbol: str
    timestamp: pd.Timestamp
    realized_vol_daily: float
    realized_vol_floor: float
    n_positions: int
    target_vol_per_position: float
    vol_parity_size: float
    regime_size_multiplier: float
    meta_p_win_multiplier: float
    pre_cap_size: float
    applied_size: float
    cap_hit: bool
    floor_hit: bool


def daily_log_returns(bars_5min: pd.DataFrame) -> pd.Series:
    """Resample a 5-min OHLCV frame to daily last-close and return log-diffs.

    Parameters
    ----------
    bars_5min:
        DataFrame indexed by timestamp with at least a ``close`` column
        (case-insensitive: ``Close`` also accepted). Must be
        chronologically sorted (groupby relies on this).

    Returns
    -------
    pd.Series of log-returns indexed by daily Timestamp (midnight). The
    first session has no return (no prior close), so it is dropped.
    """
    cols = {c.lower(): c for c in bars_5min.columns}
    if "close" not in cols:
        raise KeyError(
            f"bars_5min must have a 'close' column; got {list(bars_5min.columns)}"
        )
    close = bars_5min[cols["close"]].astype(float)
    daily_close = close.groupby(close.index.date).last()
    daily_close.index = pd.DatetimeIndex(daily_close.index)
    log_ret = np.log(daily_close / daily_close.shift(1)).dropna()
    return log_ret


def realized_daily_vol_ewma(
    daily_log_rets: pd.Series,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> pd.Series:
    """sqrt of EWMA(span=lookback_days) of squared daily log-returns.

    Uses pandas' ``span`` rather than halflife to match the literal
    "EWMA-20" spec wording. Output may have leading NaN during warmup
    (``min_periods=lookback_days``); caller drops NaN explicitly.
    """
    if lookback_days < 1:
        raise ValueError(f"lookback_days must be >= 1, got {lookback_days}")
    var_ewm = (daily_log_rets ** 2).ewm(
        span=lookback_days, adjust=False, min_periods=lookback_days,
    ).mean()
    return np.sqrt(var_ewm)


def vol_floor(
    realized_vol_series: pd.Series,
    floor_percentile: float = DEFAULT_FLOOR_PERCENTILE,
) -> float:
    """Compute the absolute floor as ``quantile(rv_series, p)``.

    Linear-interpolation quantile (numpy default). AFML §10.4 treats
    the lower tail of the lookback-window's vol distribution as the
    floor — when realized vol collapses below this, we cap rather than
    over-leverage.
    """
    if not 0.0 < floor_percentile < 1.0:
        raise ValueError(
            f"floor_percentile must be in (0,1), got {floor_percentile}"
        )
    s = realized_vol_series.dropna()
    if len(s) == 0:
        raise ValueError(
            "realized_vol_series is empty after dropna; cannot compute floor"
        )
    return float(np.quantile(s.values, floor_percentile))


def apply_floor(rv: float, floor: float) -> Tuple[float, bool]:
    """Return ``(max(rv, floor), did_floor_bind)``."""
    if rv < floor:
        return floor, True
    return rv, False


def size_event(
    *,
    symbol: str,
    event_ts: pd.Timestamp,
    bars_5min: pd.DataFrame,
    n_active_positions: int,
    target_daily_vol: float = DEFAULT_TARGET_DAILY_VOL,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    floor_percentile: float = DEFAULT_FLOOR_PERCENTILE,
    cap_mult: float = DEFAULT_CAP_MULT,
    regime_size_multiplier: float = 1.0,
    meta_p_win_multiplier: float = 1.0,
) -> SizingDecision:
    """Compute the vol-target size for one event.

    Math (AFML §10.1)::

        rv_series = sqrt(EWMA_span=L( daily_log_ret(bars[<event_ts])^2 ))
        floor     = quantile(rv_series, floor_percentile)
        rv_i      = max(rv_series.iloc[-1], floor)
        n         = max(n_active_positions, 1)
        w_parity  = (target_daily_vol / n) / rv_i
        w_pre_cap = w_parity * regime_size_multiplier * meta_p_win_multiplier
        w_applied = min(w_pre_cap, cap_mult * w_parity)

    Pre-conditions (raise ValueError):
      * ``target_daily_vol > 0``
      * ``cap_mult > 0``
      * ``0 < floor_percentile < 1``
      * ``len(daily_log_returns) >= lookback_days``
      * realized vol after floor must be > 0 (else "degenerate" raised)

    Edge cases:
      * ``n_active_positions == 0`` → treated as n=1 (the event being
        sized becomes the first position; no division by zero).
    """
    if target_daily_vol <= 0:
        raise ValueError(
            f"target_daily_vol must be > 0, got {target_daily_vol}"
        )
    if cap_mult <= 0:
        raise ValueError(f"cap_mult must be > 0, got {cap_mult}")
    if not 0.0 < floor_percentile < 1.0:
        raise ValueError(
            f"floor_percentile must be in (0,1), got {floor_percentile}"
        )

    past_bars = bars_5min[bars_5min.index < event_ts]
    daily_rets = daily_log_returns(past_bars)
    if len(daily_rets) < lookback_days:
        raise ValueError(
            f"lookback too short: need >= {lookback_days} daily returns, "
            f"have {len(daily_rets)} (symbol={symbol}, event={event_ts})"
        )

    rv_series = realized_daily_vol_ewma(
        daily_rets, lookback_days=lookback_days,
    )
    rv_clean = rv_series.dropna()
    if len(rv_clean) == 0:
        raise ValueError(
            f"realized_vol series empty after EWMA warmup "
            f"(symbol={symbol}, event={event_ts})"
        )

    floor = vol_floor(rv_clean, floor_percentile=floor_percentile)
    rv_raw = float(rv_clean.iloc[-1])
    rv_floored, floor_hit = apply_floor(rv_raw, floor)
    if rv_floored <= 0:
        raise ValueError(
            f"degenerate lookback: realized vol is zero "
            f"(symbol={symbol}, event={event_ts})"
        )

    n_eff = max(n_active_positions, 1)
    target_vol_per_position = target_daily_vol / n_eff
    vol_parity_size = target_vol_per_position / rv_floored
    pre_cap_size = (
        vol_parity_size * regime_size_multiplier * meta_p_win_multiplier
    )
    cap = cap_mult * vol_parity_size
    if pre_cap_size > cap:
        applied_size = cap
        cap_hit = True
    else:
        applied_size = pre_cap_size
        cap_hit = False

    return SizingDecision(
        symbol=symbol,
        timestamp=event_ts,
        realized_vol_daily=rv_raw,
        realized_vol_floor=rv_floored,
        n_positions=n_eff,
        target_vol_per_position=target_vol_per_position,
        vol_parity_size=vol_parity_size,
        regime_size_multiplier=regime_size_multiplier,
        meta_p_win_multiplier=meta_p_win_multiplier,
        pre_cap_size=pre_cap_size,
        applied_size=applied_size,
        cap_hit=cap_hit,
        floor_hit=floor_hit,
    )
