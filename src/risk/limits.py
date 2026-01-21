"""
Risk Limits Module

Defines and checks various risk limits.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RiskViolation(Enum):
    """Types of risk violations"""
    MAX_POSITION_SIZE = "max_position_size"
    MAX_POSITIONS = "max_positions"
    MAX_DAILY_LOSS = "max_daily_loss"
    MAX_DRAWDOWN = "max_drawdown"
    MAX_EXPOSURE = "max_exposure"
    MAX_CONCENTRATION = "max_concentration"
    INSUFFICIENT_BUYING_POWER = "insufficient_buying_power"
    CIRCUIT_BREAKER = "circuit_breaker"
    CORRELATION_LIMIT = "correlation_limit"
    VOLATILITY_LIMIT = "volatility_limit"


@dataclass
class RiskCheck:
    """Result of a risk check"""
    passed: bool
    violations: List[RiskViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_violations(self) -> bool:
        return len(self.violations) > 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def add_violation(self, violation: RiskViolation, detail: str = ""):
        """Add a violation"""
        self.violations.append(violation)
        self.passed = False
        if detail:
            self.details[violation.value] = detail

    def add_warning(self, warning: str):
        """Add a warning"""
        self.warnings.append(warning)


@dataclass
class RiskLimits:
    """
    Risk limit configuration.

    All percentages are decimals (e.g., 0.02 = 2%)
    """
    # Position limits
    max_position_size: float = 0.10  # 10% of portfolio per position
    max_position_value: float = 50000.0  # Max dollar value per position
    max_positions: int = 10  # Max number of concurrent positions
    min_position_size: int = 1  # Minimum shares

    # Portfolio limits
    max_portfolio_risk: float = 0.02  # 2% total risk
    max_daily_loss: float = 0.03  # 3% daily loss limit
    max_weekly_loss: float = 0.05  # 5% weekly loss limit
    max_drawdown: float = 0.15  # 15% max drawdown
    max_exposure: float = 0.80  # 80% max total exposure

    # Concentration limits
    max_sector_exposure: float = 0.30  # 30% per sector
    max_correlation: float = 0.70  # Max correlation between positions

    # Volatility limits
    max_position_volatility: float = 0.05  # 5% daily vol
    max_portfolio_volatility: float = 0.02  # 2% portfolio daily vol

    # Circuit breaker
    max_consecutive_losses: int = 3
    circuit_breaker_cooldown_minutes: int = 60

    # Buying power
    min_buying_power: float = 1000.0  # Minimum buying power required

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'max_position_size': self.max_position_size,
            'max_position_value': self.max_position_value,
            'max_positions': self.max_positions,
            'min_position_size': self.min_position_size,
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_daily_loss': self.max_daily_loss,
            'max_weekly_loss': self.max_weekly_loss,
            'max_drawdown': self.max_drawdown,
            'max_exposure': self.max_exposure,
            'max_sector_exposure': self.max_sector_exposure,
            'max_correlation': self.max_correlation,
            'max_position_volatility': self.max_position_volatility,
            'max_portfolio_volatility': self.max_portfolio_volatility,
            'max_consecutive_losses': self.max_consecutive_losses,
            'circuit_breaker_cooldown_minutes': self.circuit_breaker_cooldown_minutes,
            'min_buying_power': self.min_buying_power
        }


class RiskLimitChecker:
    """
    Checks various risk limits and returns violations.

    Usage:
        checker = RiskLimitChecker(limits)
        result = checker.check_new_position(
            symbol='AAPL',
            quantity=100,
            price=150.0,
            portfolio_value=100000.0,
            current_positions=current_positions
        )

        if not result.passed:
            print(f"Violations: {result.violations}")
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()

        # State tracking
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._peak_value = 0.0
        self._consecutive_losses = 0
        self._circuit_breaker_until: Optional[datetime] = None

    def check_new_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        portfolio_value: float,
        buying_power: float,
        current_positions: Dict[str, Any],
        volatility: Optional[float] = None,
        sector: Optional[str] = None
    ) -> RiskCheck:
        """
        Check if a new position meets all risk limits.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            portfolio_value: Total portfolio value
            buying_power: Available buying power
            current_positions: Dict of current positions
            volatility: Stock volatility
            sector: Stock sector

        Returns:
            RiskCheck with results
        """
        result = RiskCheck(passed=True)
        position_value = quantity * price

        # Check circuit breaker
        if self._is_circuit_breaker_active():
            result.add_violation(
                RiskViolation.CIRCUIT_BREAKER,
                f"Circuit breaker active until {self._circuit_breaker_until}"
            )
            return result

        # Check buying power
        if position_value > buying_power:
            result.add_violation(
                RiskViolation.INSUFFICIENT_BUYING_POWER,
                f"Position ${position_value:.2f} > buying power ${buying_power:.2f}"
            )

        if buying_power < self.limits.min_buying_power:
            result.add_violation(
                RiskViolation.INSUFFICIENT_BUYING_POWER,
                f"Buying power ${buying_power:.2f} below minimum ${self.limits.min_buying_power:.2f}"
            )

        # Check position size
        position_pct = position_value / portfolio_value if portfolio_value > 0 else 1.0
        if position_pct > self.limits.max_position_size:
            result.add_violation(
                RiskViolation.MAX_POSITION_SIZE,
                f"Position {position_pct:.1%} > limit {self.limits.max_position_size:.1%}"
            )

        if position_value > self.limits.max_position_value:
            result.add_violation(
                RiskViolation.MAX_POSITION_SIZE,
                f"Position ${position_value:.2f} > max ${self.limits.max_position_value:.2f}"
            )

        # Check number of positions
        num_positions = len(current_positions)
        if symbol not in current_positions and num_positions >= self.limits.max_positions:
            result.add_violation(
                RiskViolation.MAX_POSITIONS,
                f"At max positions ({num_positions}/{self.limits.max_positions})"
            )

        # Check total exposure
        current_exposure = sum(
            p.get('market_value', 0) for p in current_positions.values()
        )
        new_exposure = current_exposure + position_value
        exposure_pct = new_exposure / portfolio_value if portfolio_value > 0 else 1.0

        if exposure_pct > self.limits.max_exposure:
            result.add_violation(
                RiskViolation.MAX_EXPOSURE,
                f"Total exposure {exposure_pct:.1%} > limit {self.limits.max_exposure:.1%}"
            )

        # Check volatility
        if volatility and volatility > self.limits.max_position_volatility:
            result.add_warning(
                f"High volatility: {volatility:.1%} > {self.limits.max_position_volatility:.1%}"
            )

        # Check concentration (sector)
        if sector:
            sector_exposure = sum(
                p.get('market_value', 0)
                for p in current_positions.values()
                if p.get('sector') == sector
            )
            sector_pct = (sector_exposure + position_value) / portfolio_value
            if sector_pct > self.limits.max_sector_exposure:
                result.add_violation(
                    RiskViolation.MAX_CONCENTRATION,
                    f"Sector exposure {sector_pct:.1%} > limit {self.limits.max_sector_exposure:.1%}"
                )

        # Add details
        result.details.update({
            'position_value': position_value,
            'position_pct': position_pct,
            'current_positions': num_positions,
            'total_exposure': new_exposure,
            'exposure_pct': exposure_pct
        })

        return result

    def check_daily_loss(self, daily_pnl: float, portfolio_value: float) -> RiskCheck:
        """Check daily loss limit"""
        result = RiskCheck(passed=True)

        self._daily_pnl = daily_pnl
        loss_pct = abs(min(0, daily_pnl)) / portfolio_value if portfolio_value > 0 else 0

        if loss_pct > self.limits.max_daily_loss:
            result.add_violation(
                RiskViolation.MAX_DAILY_LOSS,
                f"Daily loss {loss_pct:.1%} > limit {self.limits.max_daily_loss:.1%}"
            )

        # Warning at 80% of limit
        elif loss_pct > self.limits.max_daily_loss * 0.8:
            result.add_warning(
                f"Approaching daily loss limit: {loss_pct:.1%} / {self.limits.max_daily_loss:.1%}"
            )

        result.details['daily_pnl'] = daily_pnl
        result.details['loss_pct'] = loss_pct

        return result

    def check_drawdown(self, portfolio_value: float) -> RiskCheck:
        """Check drawdown limit"""
        result = RiskCheck(passed=True)

        # Update peak
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

        drawdown = (self._peak_value - portfolio_value) / self._peak_value if self._peak_value > 0 else 0

        if drawdown > self.limits.max_drawdown:
            result.add_violation(
                RiskViolation.MAX_DRAWDOWN,
                f"Drawdown {drawdown:.1%} > limit {self.limits.max_drawdown:.1%}"
            )

        # Warning at 80% of limit
        elif drawdown > self.limits.max_drawdown * 0.8:
            result.add_warning(
                f"Approaching drawdown limit: {drawdown:.1%} / {self.limits.max_drawdown:.1%}"
            )

        result.details['drawdown'] = drawdown
        result.details['peak_value'] = self._peak_value
        result.details['current_value'] = portfolio_value

        return result

    def record_trade_result(self, pnl: float):
        """Record a trade result for circuit breaker tracking"""
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Trigger circuit breaker
        if self._consecutive_losses >= self.limits.max_consecutive_losses:
            from datetime import timedelta
            self._circuit_breaker_until = (
                datetime.now() +
                timedelta(minutes=self.limits.circuit_breaker_cooldown_minutes)
            )
            logger.warning(
                f"Circuit breaker triggered: {self._consecutive_losses} consecutive losses"
            )

    def _is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is active"""
        if self._circuit_breaker_until is None:
            return False
        return datetime.now() < self._circuit_breaker_until

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker"""
        self._circuit_breaker_until = None
        self._consecutive_losses = 0

    def reset_daily(self):
        """Reset daily counters"""
        self._daily_pnl = 0.0

    def reset_weekly(self):
        """Reset weekly counters"""
        self._weekly_pnl = 0.0

    def get_status(self) -> Dict[str, Any]:
        """Get current risk status"""
        return {
            'daily_pnl': self._daily_pnl,
            'weekly_pnl': self._weekly_pnl,
            'peak_value': self._peak_value,
            'consecutive_losses': self._consecutive_losses,
            'circuit_breaker_active': self._is_circuit_breaker_active(),
            'circuit_breaker_until': (
                self._circuit_breaker_until.isoformat()
                if self._circuit_breaker_until else None
            )
        }
