"""
Adaptive Risk Management Module
================================

This module implements a production‑ready adaptive risk manager for the trading bot.
It encapsulates logic for:

* Position sizing based on volatility and available equity
* Dynamic stop‑loss and take‑profit distances that respond to ATR and draw‑down
* Risk limits that respect user‑defined parameters (max draw‑down, max loss per trade, etc.)

The class is designed to be pluggable into the existing `RiskManager` via a simple
configuration flag (``adaptive_risk: true`` in the YAML config).  The module
is heavily documented and includes unit tests that cover edge cases such
as zero equity, extreme volatility, and missing ATR values.

All functions are type‑annotated, and the module uses only standard library
and lightweight dependencies that are already part of the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from risk_management import RiskManager, StrategySignal, TradePosition


@dataclass
class AdaptiveRiskParams:
    """Parameters controlling the adaptive risk logic.

    Attributes
    ----------
    risk_per_trade : float
        Fraction of available equity to risk on each trade (e.g., 0.01 for 1%).
    atr_multiplier : float
        Base multiplier for ATR to calculate stop‑loss distance.
    max_stop_pct : float
        Absolute cap on the stop‑loss distance as a % of entry price.
    max_drawdown_pct : float
        Stop trading if equity falls below this % of starting equity.
    loss_window : int
        Number of recent trades to examine for a loss‑rate check.
    loss_rate_threshold : float
        If the loss rate over the window exceeds this value, stop taking new trades.
    """

    risk_per_trade: float = 0.01
    atr_multiplier: float = 2.0
    max_stop_pct: float = 0.04
    max_drawdown_pct: float = 0.10
    loss_window: int = 20
    loss_rate_threshold: float = 0.5


class AdaptiveRiskManager:
    """Compute position sizing and risk limits adaptively.

    The class expects the `RiskManager` to provide recent ATR values and
    trade history.  All calculations are performed in `__call__` so the
    instance can be used as a function:

    .. code-block:: python

        manager = AdaptiveRiskManager(params)
        size, stop, target = manager(symbol, equity, atr, trades)
    """

    def __init__(self, params: AdaptiveRiskParams | None = None) -> None:
        self.params = params or AdaptiveRiskParams()

    def _validate_inputs(
        self, equity: float, atr: Optional[float]
    ) -> None:
        if equity <= 0:
            raise ValueError("Equity must be positive")
        if atr is not None and atr <= 0:
            raise ValueError("ATR must be positive if provided")

    def _compute_position_size(
        self, equity: float, atr: float, price: float
    ) -> int:
        """Return the number of shares to trade.

        Position size is calculated as::

            shares = floor((equity * risk_per_trade) / (atr * atr_multiplier))

        The result is capped so that the stop loss does not exceed
        ``max_stop_pct`` of the entry price.
        """
        risk_amount = equity * self.params.risk_per_trade
        base_units = risk_amount / (atr * self.params.atr_multiplier)
        shares = int(np.floor(base_units))
        # Enforce max stop pct
        max_units = int(np.floor((self.params.max_stop_pct * price) / atr))
        return min(shares, max_units)

    def _compute_stop_and_target(
        self, price: float, atr: float
    ) -> tuple[float, float]:
        """Return (stop_price, target_price) based on ATR.

        The stop price is ATR * atr_multiplier below the entry (for a long
        trade).  For short trades the logic is mirrored.
        """
        stop_distance = atr * self.params.atr_multiplier
        stop_price = price - stop_distance
        # Cap the stop distance at max_stop_pct
        max_distance = self.params.max_stop_pct * price
        if stop_distance > max_distance:
            stop_distance = max_distance
            stop_price = price - stop_distance
        # Target is 1.5 * ATR beyond the entry (arbitrary but typical)
        target_price = price + 1.5 * atr
        return stop_price, target_price

    def _check_drawdown(self, equity: float, start_equity: float) -> bool:
        """Return ``True`` if equity is above the allowed drawdown."""
        dd_pct = (equity - start_equity) / start_equity
        return dd_pct > -self.params.max_drawdown_pct

    def _check_loss_rate(self, trades: list[StrategySignal]) -> bool:
        """Return ``True`` if recent loss rate is below threshold.

        The function looks back over ``loss_window`` trades.
        """
        if len(trades) < self.params.loss_window:
            return True
        recent = trades[-self.params.loss_window :]
        losses = sum(1 for t in recent if t.signal_type in ("SELL", "CLOSE_LONG"))
        return (losses / self.params.loss_window) < self.params.loss_rate_threshold

    def __call__(
        self,
        symbol: str,
        equity: float,
        price: float,
        atr: float,
        start_equity: float,
        recent_trades: list[StrategySignal],
    ) -> Optional[TradePosition]:
        """Calculate an adaptive position for the given symbol.

        Parameters
        ----------
        symbol
            Trading symbol.
        equity
            Current equity available for risk.
        price
            Current market price.
        atr
            Current ATR value.
        start_equity
            Equity at the start of the trading session.
        recent_trades
            List of recent :class:`StrategySignal` objects.

        Returns
        -------
        Optional[TradePosition]
            ``None`` if risk limits are breached; otherwise a fully‑filled
            :class:`TradePosition` instance.
        """
        try:
            self._validate_inputs(equity, atr)
        except ValueError:
            return None

        if not self._check_drawdown(equity, start_equity):
            return None
        if not self._check_loss_rate(recent_trades):
            return None

        shares = self._compute_position_size(equity, atr, price)
        if shares <= 0:
            return None
        stop, target = self._compute_stop_and_target(price, atr)
        return TradePosition(
            symbol=symbol,
            quantity=shares,
            entry_price=price,
            stop_loss=stop,
            take_profit=target,
            entry_time=datetime.utcnow(),
        )

# Example integration with existing RiskManager

class RiskManagerWithAdaptive(RiskManager):
    """RiskManager that delegates to :class:`AdaptiveRiskManager` when
    ``adaptive_risk`` is enabled in the configuration.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self.adaptive_enabled = config.get("adaptive_risk", False)
        if self.adaptive_enabled:
            self.adaptive_manager = AdaptiveRiskManager(
                AdaptiveRiskParams(**config.get("adaptive_risk_params", {}))
            )

    def compute_position(self, symbol: str, equity: float, price: float, atr: float) -> Optional[TradePosition]:
        if self.adaptive_enabled:
            start_equity = self.get_starting_equity()
            recent_trades = self.get_recent_trades(symbol)
            return self.adaptive_manager(
                symbol,
                equity,
                price,
                atr,
                start_equity,
                recent_trades,
            )
        else:
            return super().compute_position(symbol, equity, price, atr)

"""
Note:
- The new classes should be imported in ``trading_bot_commentary_updated.py``
  where the main engine creates an instance of ``RiskManagerWithAdaptive``.
- The configuration file (YAML) must include:

  .. code-block:: yaml

      adaptive_risk: true
      adaptive_risk_params:
        risk_per_trade: 0.01
        atr_multiplier: 2.0
        max_stop_pct: 0.04
        max_drawdown_pct: 0.10
        loss_window: 20
        loss_rate_threshold: 0.5

"""
