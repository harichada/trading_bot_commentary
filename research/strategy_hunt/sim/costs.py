"""Transaction costs: Corwin-Schultz spread, volume-scaled slippage (or calibrated override), sell fees, borrow."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd

SELL_FEE_PCT = 2.946e-5
SPREAD_FLOOR = 0.0002
SPREAD_CAP = 0.005
SLIPPAGE_FLOOR = 0.0001
BORROW_RATE_ANNUAL = 0.03
SESSIONS_PER_YEAR = 252
_K = 3 - 2 * np.sqrt(2)


def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    h, l = high.astype(float), low.astype(float)
    hl2 = np.log(h / l) ** 2
    beta = hl2 + hl2.shift(1)
    h2 = pd.concat([h, h.shift(1)], axis=1).max(axis=1)
    l2 = pd.concat([l, l.shift(1)], axis=1).min(axis=1)
    gamma = np.log(h2 / l2) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / _K - np.sqrt(gamma / _K)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0.0)


def monthly_half_spread(bars5: pd.DataFrame, floor: float = SPREAD_FLOOR, cap: float = SPREAD_CAP) -> pd.Series:
    s = corwin_schultz_spread(bars5["high"], bars5["low"]).dropna()
    if s.empty:
        return pd.Series(dtype=float, index=pd.PeriodIndex([], freq="M"))
    return (s.groupby(s.index.to_period("M")).median() / 2.0).clip(lower=floor, upper=cap)


@dataclass(frozen=True)
class CostModel:
    half_spread_by_month: pd.Series
    notional: float = 6000.0
    sell_fee_pct: float = SELL_FEE_PCT
    slippage_floor: float = SLIPPAGE_FLOOR
    borrow_rate_annual: float = BORROW_RATE_ANNUAL
    slippage_override: Optional[float] = None

    @classmethod
    def from_bars(cls, bars5: pd.DataFrame, notional: float = 6000.0, slippage_override: Optional[float] = None) -> "CostModel":
        return cls(half_spread_by_month=monthly_half_spread(bars5), notional=notional, slippage_override=slippage_override)

    def scaled(self, mult: float) -> "CostModel":
        return replace(self, half_spread_by_month=self.half_spread_by_month * mult, sell_fee_pct=self.sell_fee_pct * mult,
                       slippage_floor=self.slippage_floor * mult, borrow_rate_annual=self.borrow_rate_annual * mult,
                       slippage_override=None if self.slippage_override is None else self.slippage_override * mult)

    def half_spread(self, ts: pd.Timestamp) -> float:
        s = self.half_spread_by_month
        if s.empty:
            return SPREAD_FLOOR
        key = pd.Timestamp(ts).to_period("M")
        return float(s[key]) if key in s.index else float(s.median())

    def slippage(self, bar_volume: float, price: float) -> float:
        if self.slippage_override is not None:
            return self.slippage_override
        if self.slippage_floor == 0.0 and self.notional == 0.0:
            return 0.0
        participation_pct = 100.0 * (self.notional / max(price, 1e-9)) / max(float(bar_volume), 1.0)
        return max(self.slippage_floor, 0.0001 * participation_pct)

    def borrow_cost(self, side: str, hold_sessions: int) -> float:
        return self.borrow_rate_annual * max(int(hold_sessions), 0) / SESSIONS_PER_YEAR if side == "short" else 0.0

    def _friction(self, ts: pd.Timestamp, bar_volume: float, price: float) -> float:
        return self.half_spread(ts) + self.slippage(bar_volume, price)

    def entry_fill(self, side: str, px: float, ts: pd.Timestamp, bar_volume: float) -> float:
        f = self._friction(ts, bar_volume, px)
        return px * (1 + f) if side == "long" else px * (1 - f - self.sell_fee_pct)

    def exit_fill(self, side: str, px: float, ts: pd.Timestamp, bar_volume: float, hold_sessions: int = 0) -> float:
        f = self._friction(ts, bar_volume, px)
        if side == "long":
            return px * (1 - f - self.sell_fee_pct)
        return px * (1 + f + self.borrow_cost(side, hold_sessions))


ZERO_COSTS = CostModel(half_spread_by_month=pd.Series([0.0], index=pd.PeriodIndex(["2000-01"], freq="M")),
                       notional=0.0, sell_fee_pct=0.0, slippage_floor=0.0, borrow_rate_annual=0.0)
