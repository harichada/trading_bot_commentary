"""
Position Sizing Module

Various position sizing methods for risk management.
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing methods"""
    FIXED_AMOUNT = "fixed_amount"
    FIXED_PERCENTAGE = "fixed_percentage"
    KELLY_CRITERION = "kelly"
    VOLATILITY_ADJUSTED = "volatility"
    RISK_PARITY = "risk_parity"


@dataclass
class SizeResult:
    """Position sizing calculation result"""
    shares: int
    dollar_amount: float
    risk_amount: float
    method: SizingMethod
    details: Dict[str, Any]


class PositionSizer:
    """
    Calculates appropriate position sizes based on various methods.

    Methods:
    - Fixed Amount: Fixed dollar amount per trade
    - Fixed Percentage: Percentage of portfolio
    - Kelly Criterion: Optimal betting based on win rate/payoff
    - Volatility Adjusted: Size inversely proportional to volatility
    - Risk Parity: Equal risk contribution

    Usage:
        sizer = PositionSizer(method=SizingMethod.FIXED_PERCENTAGE)
        result = sizer.calculate(
            price=150.0,
            portfolio_value=100000.0,
            volatility=0.02
        )
    """

    def __init__(
        self,
        method: SizingMethod = SizingMethod.FIXED_PERCENTAGE,
        config: Optional[Dict[str, Any]] = None
    ):
        self.method = method
        self.config = config or {}

        # Default parameters
        self.fixed_amount = self.config.get('fixed_amount', 5000.0)
        self.fixed_percentage = self.config.get('fixed_percentage', 0.05)  # 5%
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02)  # 2%
        self.min_shares = self.config.get('min_shares', 1)
        self.max_shares = self.config.get('max_shares', 10000)

        # Kelly parameters
        self.kelly_fraction = self.config.get('kelly_fraction', 0.25)  # Quarter Kelly

    def calculate(
        self,
        price: float,
        portfolio_value: float,
        stop_loss: Optional[float] = None,
        volatility: Optional[float] = None,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None
    ) -> SizeResult:
        """
        Calculate position size.

        Args:
            price: Current stock price
            portfolio_value: Total portfolio value
            stop_loss: Stop loss price (for risk-based sizing)
            volatility: Stock volatility (for vol-adjusted sizing)
            win_rate: Historical win rate (for Kelly)
            avg_win: Average winning trade return (for Kelly)
            avg_loss: Average losing trade return (for Kelly)

        Returns:
            SizeResult with calculated position size
        """
        if price <= 0 or portfolio_value <= 0:
            return SizeResult(
                shares=0,
                dollar_amount=0,
                risk_amount=0,
                method=self.method,
                details={'error': 'Invalid price or portfolio value'}
            )

        if self.method == SizingMethod.FIXED_AMOUNT:
            return self._fixed_amount(price, portfolio_value)

        elif self.method == SizingMethod.FIXED_PERCENTAGE:
            return self._fixed_percentage(price, portfolio_value)

        elif self.method == SizingMethod.KELLY_CRITERION:
            return self._kelly_criterion(
                price, portfolio_value, win_rate, avg_win, avg_loss
            )

        elif self.method == SizingMethod.VOLATILITY_ADJUSTED:
            return self._volatility_adjusted(price, portfolio_value, volatility)

        elif self.method == SizingMethod.RISK_PARITY:
            return self._risk_parity(price, portfolio_value, stop_loss, volatility)

        else:
            return self._fixed_percentage(price, portfolio_value)

    def _fixed_amount(self, price: float, portfolio_value: float) -> SizeResult:
        """Fixed dollar amount sizing"""
        amount = min(self.fixed_amount, portfolio_value * 0.25)
        shares = self._constrain_shares(int(amount / price))

        return SizeResult(
            shares=shares,
            dollar_amount=shares * price,
            risk_amount=shares * price * self.max_risk_per_trade,
            method=SizingMethod.FIXED_AMOUNT,
            details={
                'target_amount': self.fixed_amount,
                'actual_amount': shares * price
            }
        )

    def _fixed_percentage(self, price: float, portfolio_value: float) -> SizeResult:
        """Fixed percentage of portfolio"""
        amount = portfolio_value * self.fixed_percentage
        shares = self._constrain_shares(int(amount / price))

        return SizeResult(
            shares=shares,
            dollar_amount=shares * price,
            risk_amount=shares * price * self.max_risk_per_trade,
            method=SizingMethod.FIXED_PERCENTAGE,
            details={
                'percentage': self.fixed_percentage,
                'target_amount': amount,
                'actual_amount': shares * price,
                'actual_percentage': (shares * price) / portfolio_value
            }
        )

    def _kelly_criterion(
        self,
        price: float,
        portfolio_value: float,
        win_rate: Optional[float],
        avg_win: Optional[float],
        avg_loss: Optional[float]
    ) -> SizeResult:
        """Kelly criterion position sizing"""
        # Default values if not provided
        win_rate = win_rate or 0.5
        avg_win = avg_win or 0.02
        avg_loss = avg_loss or 0.01

        # Ensure positive loss
        avg_loss = abs(avg_loss)

        if avg_loss == 0:
            avg_loss = 0.01

        # Kelly formula: f* = (p * b - q) / b
        # where p = win prob, q = loss prob, b = win/loss ratio
        b = avg_win / avg_loss
        q = 1 - win_rate

        kelly_pct = (win_rate * b - q) / b

        # Apply fraction (quarter Kelly is common)
        kelly_pct = kelly_pct * self.kelly_fraction

        # Constrain to reasonable range
        kelly_pct = max(0, min(kelly_pct, 0.25))

        amount = portfolio_value * kelly_pct
        shares = self._constrain_shares(int(amount / price))

        return SizeResult(
            shares=shares,
            dollar_amount=shares * price,
            risk_amount=shares * price * avg_loss,
            method=SizingMethod.KELLY_CRITERION,
            details={
                'kelly_pct': kelly_pct,
                'full_kelly': kelly_pct / self.kelly_fraction,
                'kelly_fraction': self.kelly_fraction,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': b
            }
        )

    def _volatility_adjusted(
        self,
        price: float,
        portfolio_value: float,
        volatility: Optional[float]
    ) -> SizeResult:
        """Volatility-adjusted position sizing"""
        # Default volatility
        volatility = volatility or 0.02

        # Target volatility contribution
        target_vol = self.config.get('target_volatility', 0.01)  # 1% portfolio vol

        # Calculate size inversely proportional to volatility
        if volatility > 0:
            vol_factor = target_vol / volatility
        else:
            vol_factor = 1.0

        # Base amount adjusted by volatility
        base_pct = self.fixed_percentage
        adjusted_pct = base_pct * vol_factor

        # Cap at reasonable level
        adjusted_pct = min(adjusted_pct, 0.20)

        amount = portfolio_value * adjusted_pct
        shares = self._constrain_shares(int(amount / price))

        return SizeResult(
            shares=shares,
            dollar_amount=shares * price,
            risk_amount=shares * price * volatility,
            method=SizingMethod.VOLATILITY_ADJUSTED,
            details={
                'volatility': volatility,
                'target_vol': target_vol,
                'vol_factor': vol_factor,
                'base_pct': base_pct,
                'adjusted_pct': adjusted_pct
            }
        )

    def _risk_parity(
        self,
        price: float,
        portfolio_value: float,
        stop_loss: Optional[float],
        volatility: Optional[float]
    ) -> SizeResult:
        """Risk parity - equal risk contribution"""
        # Calculate risk per share
        if stop_loss and stop_loss > 0:
            risk_per_share = abs(price - stop_loss)
        elif volatility:
            risk_per_share = price * volatility * 2  # 2x daily vol as proxy
        else:
            risk_per_share = price * 0.02  # Default 2%

        # Target risk amount
        risk_budget = portfolio_value * self.max_risk_per_trade

        # Calculate shares based on risk
        if risk_per_share > 0:
            shares = int(risk_budget / risk_per_share)
        else:
            shares = int(portfolio_value * self.fixed_percentage / price)

        shares = self._constrain_shares(shares)

        return SizeResult(
            shares=shares,
            dollar_amount=shares * price,
            risk_amount=shares * risk_per_share,
            method=SizingMethod.RISK_PARITY,
            details={
                'risk_per_share': risk_per_share,
                'risk_budget': risk_budget,
                'stop_loss': stop_loss,
                'volatility': volatility
            }
        )

    def _constrain_shares(self, shares: int) -> int:
        """Constrain shares to valid range"""
        return max(self.min_shares, min(shares, self.max_shares))

    def calculate_with_stop(
        self,
        price: float,
        stop_loss: float,
        portfolio_value: float,
        risk_percentage: Optional[float] = None
    ) -> SizeResult:
        """
        Calculate position size based on stop loss.

        This is a risk-first approach where you define max loss first.

        Args:
            price: Entry price
            stop_loss: Stop loss price
            portfolio_value: Total portfolio value
            risk_percentage: Max risk as percentage of portfolio

        Returns:
            SizeResult with calculated size
        """
        risk_pct = risk_percentage or self.max_risk_per_trade
        risk_amount = portfolio_value * risk_pct

        # Risk per share
        risk_per_share = abs(price - stop_loss)

        if risk_per_share <= 0:
            return SizeResult(
                shares=0,
                dollar_amount=0,
                risk_amount=0,
                method=self.method,
                details={'error': 'Invalid stop loss'}
            )

        shares = self._constrain_shares(int(risk_amount / risk_per_share))

        return SizeResult(
            shares=shares,
            dollar_amount=shares * price,
            risk_amount=shares * risk_per_share,
            method=SizingMethod.RISK_PARITY,
            details={
                'risk_percentage': risk_pct,
                'risk_amount': risk_amount,
                'risk_per_share': risk_per_share,
                'stop_loss': stop_loss,
                'stop_percentage': risk_per_share / price
            }
        )
