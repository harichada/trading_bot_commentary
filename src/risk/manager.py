"""
Risk Manager

Central risk management coordinator.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from .position_sizer import PositionSizer, SizingMethod, SizeResult
from .limits import RiskLimits, RiskLimitChecker, RiskCheck

logger = logging.getLogger(__name__)


@dataclass
class RiskAssessment:
    """Complete risk assessment for a trade"""
    approved: bool
    position_size: int
    dollar_amount: float
    risk_amount: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    checks: RiskCheck
    sizing: SizeResult
    reason: str = ""


class RiskManager:
    """
    Central risk management for the trading system.

    Coordinates:
    - Position sizing
    - Risk limit checking
    - Stop loss / take profit calculation
    - Portfolio risk monitoring

    Usage:
        manager = RiskManager(limits, sizing_method=SizingMethod.VOLATILITY_ADJUSTED)

        assessment = await manager.assess_trade(
            symbol='AAPL',
            action='buy',
            price=150.0,
            confidence=0.75,
            portfolio_state=portfolio
        )

        if assessment.approved:
            # Execute trade with assessment.position_size
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        sizing_method: SizingMethod = SizingMethod.FIXED_PERCENTAGE,
        sizing_config: Optional[Dict[str, Any]] = None
    ):
        self.limits = limits or RiskLimits()
        self.sizer = PositionSizer(method=sizing_method, config=sizing_config)
        self.checker = RiskLimitChecker(self.limits)

        # Risk parameters
        self.stop_loss_atr_multiplier = 2.0
        self.take_profit_atr_multiplier = 3.0
        self.default_stop_loss_pct = 0.02  # 2%
        self.default_take_profit_pct = 0.04  # 4%

        # Performance tracking
        self._trade_history: List[Dict[str, Any]] = []
        self._win_rate = 0.5
        self._avg_win = 0.02
        self._avg_loss = 0.01

    async def check_signal(self, signal: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Quick check if a signal passes basic risk checks.

        Args:
            signal: Signal dictionary with symbol, action, confidence

        Returns:
            Tuple of (approved, reason)
        """
        confidence = signal.get('confidence', 0)

        # Minimum confidence
        if confidence < 0.5:
            return False, f"Confidence too low: {confidence:.1%}"

        # Check circuit breaker
        if self.checker._is_circuit_breaker_active():
            return False, "Circuit breaker active"

        return True, "Signal passed initial checks"

    async def calculate_position_size(
        self,
        symbol: str,
        buying_power: float,
        price: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> int:
        """
        Calculate position size for a symbol.

        Args:
            symbol: Stock symbol
            buying_power: Available buying power
            price: Current price (if known)
            volatility: Stock volatility (if known)

        Returns:
            Number of shares
        """
        if price is None or price <= 0:
            return 0

        result = self.sizer.calculate(
            price=price,
            portfolio_value=buying_power,
            volatility=volatility,
            win_rate=self._win_rate,
            avg_win=self._avg_win,
            avg_loss=self._avg_loss
        )

        return result.shares

    async def assess_trade(
        self,
        symbol: str,
        action: str,  # 'buy' or 'sell'
        price: float,
        confidence: float,
        portfolio_value: float,
        buying_power: float,
        current_positions: Dict[str, Any],
        volatility: Optional[float] = None,
        atr: Optional[float] = None,
        sector: Optional[str] = None
    ) -> RiskAssessment:
        """
        Perform complete risk assessment for a trade.

        Args:
            symbol: Stock symbol
            action: Trade action
            price: Current price
            confidence: Signal confidence
            portfolio_value: Total portfolio value
            buying_power: Available buying power
            current_positions: Current positions dict
            volatility: Stock volatility
            atr: Average True Range
            sector: Stock sector

        Returns:
            RiskAssessment with decision and parameters
        """
        # Calculate position size
        sizing = self.sizer.calculate(
            price=price,
            portfolio_value=portfolio_value,
            volatility=volatility,
            win_rate=self._win_rate,
            avg_win=self._avg_win,
            avg_loss=self._avg_loss
        )

        # Adjust size based on confidence
        adjusted_shares = int(sizing.shares * min(1.0, confidence + 0.25))

        # Check risk limits
        checks = self.checker.check_new_position(
            symbol=symbol,
            quantity=adjusted_shares,
            price=price,
            portfolio_value=portfolio_value,
            buying_power=buying_power,
            current_positions=current_positions,
            volatility=volatility,
            sector=sector
        )

        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_exits(
            price=price,
            action=action,
            atr=atr,
            volatility=volatility
        )

        # Determine approval
        approved = checks.passed and adjusted_shares > 0 and confidence >= 0.5

        reason = ""
        if not checks.passed:
            reason = f"Risk violations: {[v.value for v in checks.violations]}"
        elif adjusted_shares <= 0:
            reason = "Position size too small"
        elif confidence < 0.5:
            reason = f"Confidence too low: {confidence:.1%}"

        return RiskAssessment(
            approved=approved,
            position_size=adjusted_shares if approved else 0,
            dollar_amount=adjusted_shares * price if approved else 0,
            risk_amount=sizing.risk_amount * (adjusted_shares / max(sizing.shares, 1)),
            stop_loss=stop_loss,
            take_profit=take_profit,
            checks=checks,
            sizing=sizing,
            reason=reason if not approved else "Approved"
        )

    def _calculate_exits(
        self,
        price: float,
        action: str,
        atr: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels"""
        if atr and atr > 0:
            # ATR-based stops
            if action == 'buy':
                stop_loss = price - (atr * self.stop_loss_atr_multiplier)
                take_profit = price + (atr * self.take_profit_atr_multiplier)
            else:
                stop_loss = price + (atr * self.stop_loss_atr_multiplier)
                take_profit = price - (atr * self.take_profit_atr_multiplier)
        else:
            # Percentage-based stops
            if action == 'buy':
                stop_loss = price * (1 - self.default_stop_loss_pct)
                take_profit = price * (1 + self.default_take_profit_pct)
            else:
                stop_loss = price * (1 + self.default_stop_loss_pct)
                take_profit = price * (1 - self.default_take_profit_pct)

        return stop_loss, take_profit

    def record_trade(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        exit_price: Optional[float] = None,
        quantity: int = 0,
        pnl: float = 0.0
    ):
        """
        Record a trade for performance tracking.

        Args:
            symbol: Stock symbol
            action: Trade action
            entry_price: Entry price
            exit_price: Exit price (if closed)
            quantity: Number of shares
            pnl: Profit/loss
        """
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'quantity': quantity,
            'pnl': pnl,
            'return_pct': (pnl / (entry_price * quantity)) if entry_price * quantity > 0 else 0
        }

        self._trade_history.append(trade)

        # Update circuit breaker
        if exit_price is not None:
            self.checker.record_trade_result(pnl)

        # Update performance metrics
        self._update_performance()

    def _update_performance(self):
        """Update performance metrics from trade history"""
        if not self._trade_history:
            return

        # Only count closed trades
        closed_trades = [t for t in self._trade_history if t.get('exit_price')]

        if not closed_trades:
            return

        wins = [t for t in closed_trades if t['pnl'] > 0]
        losses = [t for t in closed_trades if t['pnl'] < 0]

        # Win rate
        self._win_rate = len(wins) / len(closed_trades) if closed_trades else 0.5

        # Average win/loss
        if wins:
            self._avg_win = sum(t['return_pct'] for t in wins) / len(wins)
        if losses:
            self._avg_loss = abs(sum(t['return_pct'] for t in losses) / len(losses))

    async def check_portfolio_risk(
        self,
        portfolio_value: float,
        daily_pnl: float,
        positions: Dict[str, Any]
    ) -> RiskCheck:
        """
        Check overall portfolio risk.

        Args:
            portfolio_value: Current portfolio value
            daily_pnl: Today's P&L
            positions: Current positions

        Returns:
            RiskCheck with results
        """
        # Check daily loss
        daily_check = self.checker.check_daily_loss(daily_pnl, portfolio_value)

        # Check drawdown
        drawdown_check = self.checker.check_drawdown(portfolio_value)

        # Combine results
        result = RiskCheck(passed=daily_check.passed and drawdown_check.passed)
        result.violations.extend(daily_check.violations)
        result.violations.extend(drawdown_check.violations)
        result.warnings.extend(daily_check.warnings)
        result.warnings.extend(drawdown_check.warnings)
        result.details.update(daily_check.details)
        result.details.update(drawdown_check.details)

        # Calculate total exposure
        total_value = sum(
            p.get('market_value', 0) for p in positions.values()
        )
        exposure_pct = total_value / portfolio_value if portfolio_value > 0 else 0
        result.details['total_exposure'] = exposure_pct

        return result

    def get_risk_status(self) -> Dict[str, Any]:
        """Get current risk status"""
        return {
            'checker_status': self.checker.get_status(),
            'performance': {
                'win_rate': self._win_rate,
                'avg_win': self._avg_win,
                'avg_loss': self._avg_loss,
                'total_trades': len(self._trade_history)
            },
            'limits': self.limits.to_dict(),
            'sizing_method': self.sizer.method.value
        }

    def reset_daily(self):
        """Reset daily counters"""
        self.checker.reset_daily()

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker"""
        self.checker.reset_circuit_breaker()
