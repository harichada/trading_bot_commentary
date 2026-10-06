"""v-turn-engine-2026-10-05: the fit lane's adapter for the `rsi_turn` family —
the live Level 1 + tape snapshot as `mf_*` columns, the family's per-bar
verdict side channel, and the process-wide verdict cache the swing exit reads.
Kept out of `strategies/fit_lane.py` to respect its line budget.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("TradingBot")


MICRO_FIELDS = ("mf_buy_vol_15", "mf_sell_vol_15", "mf_vol_15", "mf_cvd_15", "mf_cvd_60", "mf_vol_60",
                "mf_sell_share_15", "mf_large_buy_15", "mf_large_sell_15", "mf_large_vol_15", "mf_n_trades_15",
                "mf_imb_ema", "mf_spread_ema", "mf_spread_baseline", "mf_spread_mad", "mf_bid_size", "mf_ask_size",
                "mf_median_bbo_size", "mf_aggressor_rule", "mf_age_s")
MICRO_MAX_AGE_S = 120.0


def _default_micro_provider(symbol: str):
    from core.micro_book import MicroBook
    book = MicroBook.instance()
    return book.snapshot(symbol) if book is not None else None


def attach_micro(ind: pd.DataFrame, snap: Any) -> pd.DataFrame:
    """v-turn-engine-2026-10-05: put the live book's snapshot on the LAST bar
    of the indicator frame as the `mf_*` columns `rsi_turn` reads (earlier
    bars stay NaN: only the current bar is scored live). A stale or missing
    snapshot leaves the tape columns NaN (A contributes nothing) and the
    quote columns absent (B = 0, no liquidity veto). The spread baseline is
    not calibrated yet (Phase 1): z = 0 by construction, and the median BBO
    size is unknown (0 → the size check is skipped); the ledger says so via
    `mf_aggressor_rule`."""
    out = ind.copy()
    for c in MICRO_FIELDS:
        if c not in out.columns:
            out[c] = "" if c == "mf_aggressor_rule" else np.nan
    if snap is None or out.empty:
        return out
    age = float(getattr(snap, "age_s", float("inf")))
    if not (age <= MICRO_MAX_AGE_S) or getattr(snap, "vol_60", 0.0) <= 0:
        return out
    last = out.index[-1]
    vals = {
        "mf_buy_vol_15": snap.buy_vol_15, "mf_sell_vol_15": snap.sell_vol_15, "mf_vol_15": snap.vol_15,
        "mf_cvd_15": snap.cvd_15, "mf_cvd_60": snap.cvd_60, "mf_vol_60": snap.vol_60,
        "mf_sell_share_15": snap.sell_share_15, "mf_large_buy_15": snap.large_buy_15,
        "mf_large_sell_15": snap.large_sell_15, "mf_large_vol_15": snap.large_vol_15,
        "mf_n_trades_15": snap.n_updates_15, "mf_age_s": age,
        "mf_aggressor_rule": "quote_conflated",
    }
    if snap.bid is not None and snap.ask is not None and snap.bid_size > 0 and snap.ask_size > 0:
        vals.update({"mf_imb_ema": snap.imb_ema, "mf_spread_ema": snap.spread_ema,
                     "mf_spread_baseline": snap.spread_ema, "mf_spread_mad": max(snap.spread_ema * 0.5, 1e-6),
                     "mf_bid_size": snap.bid_size, "mf_ask_size": snap.ask_size, "mf_median_bbo_size": 0.0})
    for k, v in vals.items():
        out.loc[last, k] = v
    return out


def verdict_from_attrs(bars: pd.DataFrame, now_ts) -> Optional[dict]:
    """The rsi_turn family's per-bar verdict for the current bar, if it scored one."""
    try:
        v = (bars.attrs.get("rsi_turn_verdicts") or {}).get(now_ts)
    except Exception:
        return None
    if not v:
        return None
    return {k: (float(x) if isinstance(x, (int, float, np.floating)) and not isinstance(x, bool) else x)
            for k, x in v.items()}


def record_turn_verdict(symbol: str, now_ts, thesis: dict) -> None:
    try:
        from core.turn_verdicts import TurnVerdict, TurnVerdicts
        TurnVerdicts.instance().record(TurnVerdict(
            symbol=symbol, ts=pd.Timestamp(now_ts).to_pydatetime(), score=float(thesis.get("score", 0.0)),
            theta=float(thesis.get("theta", 0.0)), cvd_60=float(thesis.get("cvd_60", 0.0) or 0.0),
            vetoes=tuple(thesis.get("vetoes") or ()), detail=thesis))
    except Exception as exc:
        logger.debug("fit_lane: turn verdict not recorded for %s: %s", symbol, exc)


def shadow_norow_sides(family: str, sides, shadow_norow, live_families) -> tuple:
    """Sides a shadow-only family may ledger when the fit snapshot has no row
    for the symbol (a brand-new family collecting its first occurrences).
    Always () for a live family, so no live entry can come from here."""
    try:
        if family in shadow_norow() and family not in live_families():
            return tuple(sides)
    except Exception:
        pass
    return ()
