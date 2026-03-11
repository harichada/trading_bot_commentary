#!/usr/bin/env python3
"""
Professional Risk Management and Position Sizing System
Implements various position sizing algorithms and risk controls
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger('RiskManagement')

class PositionSizingMethod(Enum):
    FIXED_AMOUNT = "FIXED_AMOUNT"
    FIXED_PERCENTAGE = "FIXED_PERCENTAGE"
    KELLY_CRITERION = "KELLY_CRITERION"
    VOLATILITY_BASED = "VOLATILITY_BASED"
    ATR_BASED = "ATR_BASED"
    RISK_PARITY = "RISK_PARITY"
    VAR_BASED = "VAR_BASED"
    OPTIMAL_F = "OPTIMAL_F"

@dataclass
class RiskMetrics:
    """Current risk metrics for the portfolio"""
    total_exposure: float
    var_95: float  # 95% Value at Risk
    cvar_95: float  # Conditional VaR
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    correlation_risk: float
    concentration_risk: float
    leverage: float
    margin_usage: float
    
@dataclass
class PositionRiskProfile:
    """Risk profile for a single position"""
    symbol: str
    current_risk: float  # Current $ at risk
    max_risk: float  # Maximum $ at risk
    volatility: float  # Asset volatility
    beta: float  # Market beta
    correlation_to_portfolio: float
    var_contribution: float  # Contribution to portfolio VaR
    weight: float  # Portfolio weight
    
@dataclass
class RiskLimits:
    """Risk limits for the portfolio"""
    max_portfolio_risk: float = 0.02  # 2% max portfolio risk
    max_position_risk: float = 0.01  # 1% max per position
    max_sector_exposure: float = 0.30  # 30% max sector exposure
    max_correlation: float = 0.80  # Max correlation between positions
    max_leverage: float = 1.0  # No leverage by default
    max_var_95: float = 0.03  # 3% VaR limit
    max_daily_loss: float = 0.02  # 2% daily loss limit
    max_weekly_loss: float = 0.05  # 5% weekly loss limit
    max_monthly_loss: float = 0.10  # 10% monthly loss limit
    max_positions: int = 10
    min_position_size: float = 100  # Minimum $100 per position
    max_position_size_pct: float = 0.10  # 10% max position size

class BasePositionSizer(ABC):
    """Base class for position sizing algorithms"""
    
    def __init__(self, capital: float, risk_limits: RiskLimits):
        self.capital = capital
        self.risk_limits = risk_limits
        
    @abstractmethod
    def calculate_position_size(self, signal_data: Dict[str, Any]) -> float:
        """Calculate position size based on the algorithm"""
        pass
    
    def apply_risk_limits(self, size: float, price: float) -> float:
        """Apply risk limits to position size"""
        # Maximum position value
        max_value = self.capital * self.risk_limits.max_position_size_pct
        
        # Minimum position value
        min_value = self.risk_limits.min_position_size
        
        # Constrain position value
        position_value = size * price
        position_value = max(min_value, min(max_value, position_value))
        
        return position_value / price

class FixedPercentageSizer(BasePositionSizer):
    """Fixed percentage of capital position sizing"""
    
    def __init__(self, capital: float, risk_limits: RiskLimits, percentage: float = 0.02):
        super().__init__(capital, risk_limits)
        self.percentage = percentage
        
    def calculate_position_size(self, signal_data: Dict[str, Any]) -> float:
        price = signal_data['price']
        position_value = self.capital * self.percentage
        size = position_value / price
        
        return self.apply_risk_limits(size, price)

class VolatilityBasedSizer(BasePositionSizer):
    """Position sizing based on asset volatility"""
    
    def __init__(self, capital: float, risk_limits: RiskLimits, target_risk: float = 0.01):
        super().__init__(capital, risk_limits)
        self.target_risk = target_risk  # Target risk per position
        
    def calculate_position_size(self, signal_data: Dict[str, Any]) -> float:
        price = signal_data['price']
        volatility = signal_data.get('volatility', 0.02)  # Default 2% volatility
        
        # Calculate position size to achieve target risk
        # Position Value = Target Risk / Volatility
        target_value = (self.capital * self.target_risk) / volatility
        size = target_value / price
        
        return self.apply_risk_limits(size, price)

class ATRBasedSizer(BasePositionSizer):
    """Position sizing based on Average True Range"""
    
    def __init__(self, capital: float, risk_limits: RiskLimits, risk_per_trade: float = 0.01):
        super().__init__(capital, risk_limits)
        self.risk_per_trade = risk_per_trade
        
    def calculate_position_size(self, signal_data: Dict[str, Any]) -> float:
        price = signal_data['price']
        atr = signal_data.get('atr', price * 0.02)  # Default 2% of price
        stop_loss_distance = signal_data.get('stop_loss_distance', atr * 2)
        
        # Calculate position size based on stop loss
        # Position Size = Risk Amount / Stop Loss Distance
        risk_amount = self.capital * self.risk_per_trade
        size = risk_amount / stop_loss_distance
        
        return self.apply_risk_limits(size, price)

class KellyCriterionSizer(BasePositionSizer):
    """Kelly Criterion position sizing"""
    
    def __init__(self, capital: float, risk_limits: RiskLimits, kelly_fraction: float = 0.25):
        super().__init__(capital, risk_limits)
        self.kelly_fraction = kelly_fraction  # Fraction of Kelly to use (for safety)
        
    def calculate_position_size(self, signal_data: Dict[str, Any]) -> float:
        price = signal_data['price']
        win_rate = signal_data.get('win_rate', 0.5)
        avg_win = signal_data.get('avg_win', 0.02)
        avg_loss = signal_data.get('avg_loss', 0.01)
        
        if avg_loss == 0:
            return 0
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        win_loss_ratio = avg_win / avg_loss
        kelly_percentage = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
        
        # Apply Kelly fraction for safety
        kelly_percentage = max(0, kelly_percentage * self.kelly_fraction)
        
        # Limit to maximum position size
        kelly_percentage = min(kelly_percentage, self.risk_limits.max_position_size_pct)
        
        position_value = self.capital * kelly_percentage
        size = position_value / price
        
        return self.apply_risk_limits(size, price)

class RiskParitySizer(BasePositionSizer):
    """Risk parity position sizing - equal risk contribution"""
    
    def __init__(self, capital: float, risk_limits: RiskLimits, 
                 existing_positions: Dict[str, Dict] = None):
        super().__init__(capital, risk_limits)
        self.existing_positions = existing_positions or {}
        
    def calculate_position_size(self, signal_data: Dict[str, Any]) -> float:
        price = signal_data['price']
        volatility = signal_data.get('volatility', 0.02)
        
        # Calculate current portfolio risk
        total_risk = 0
        for pos in self.existing_positions.values():
            pos_volatility = pos.get('volatility', 0.02)
            pos_value = pos['quantity'] * pos['current_price']
            total_risk += (pos_value / self.capital) * pos_volatility
        
        # Target equal risk contribution
        num_positions = len(self.existing_positions) + 1
        target_risk_contribution = self.risk_limits.max_portfolio_risk / num_positions
        
        # Calculate position size for target risk
        position_weight = target_risk_contribution / volatility
        position_value = self.capital * position_weight
        size = position_value / price
        
        return self.apply_risk_limits(size, price)

class OptimalFSizer(BasePositionSizer):
    """Optimal F position sizing based on historical trade results"""
    
    def __init__(self, capital: float, risk_limits: RiskLimits, 
                 trade_history: List[float] = None):
        super().__init__(capital, risk_limits)
        self.trade_history = trade_history or []
        self.optimal_f = self._calculate_optimal_f()
        
    def _calculate_optimal_f(self) -> float:
        """Calculate Optimal F from trade history"""
        if len(self.trade_history) < 10:
            return 0.02  # Default to 2% if insufficient history
        
        # Find the f that maximizes TWR (Terminal Wealth Relative)
        best_f = 0.01
        best_twr = 0
        
        # Test different f values
        for f in np.arange(0.01, 0.50, 0.01):
            twr = 1.0
            largest_loss = abs(min(self.trade_history))
            
            for trade in self.trade_history:
                hpp = trade / -largest_loss  # Holding Period Return
                twr *= (1 + f * hpp)
            
            if twr > best_twr:
                best_twr = twr
                best_f = f
        
        # Apply safety factor
        return min(best_f * 0.5, self.risk_limits.max_position_size_pct)
    
    def calculate_position_size(self, signal_data: Dict[str, Any]) -> float:
        price = signal_data['price']
        position_value = self.capital * self.optimal_f
        size = position_value / price
        
        return self.apply_risk_limits(size, price)

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self, initial_capital: float, risk_limits: RiskLimits = None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_limits = risk_limits or RiskLimits()
        
        # Position sizers
        self.position_sizers = {
            PositionSizingMethod.FIXED_PERCENTAGE: FixedPercentageSizer(
                self.current_capital, self.risk_limits
            ),
            PositionSizingMethod.VOLATILITY_BASED: VolatilityBasedSizer(
                self.current_capital, self.risk_limits
            ),
            PositionSizingMethod.ATR_BASED: ATRBasedSizer(
                self.current_capital, self.risk_limits
            ),
            PositionSizingMethod.KELLY_CRITERION: KellyCriterionSizer(
                self.current_capital, self.risk_limits
            ),
        }
        
        # Risk tracking
        self.positions: Dict[str, PositionRiskProfile] = {}
        self.daily_pnl: List[float] = []
        self.trade_history: List[float] = []
        self.high_water_mark = initial_capital
        
        # Circuit breakers
        self.trading_halted = False
        self.halt_reason = ""
        self.halt_until = None
        
    def calculate_position_size(self, method: PositionSizingMethod, 
                              signal_data: Dict[str, Any]) -> float:
        """Calculate position size using specified method"""
        if self.trading_halted:
            logger.warning(f"Trading halted: {self.halt_reason}")
            return 0
        
        # Update capital for sizers
        for sizer in self.position_sizers.values():
            sizer.capital = self.current_capital
        
        # Get position size from selected method
        if method in self.position_sizers:
            size = self.position_sizers[method].calculate_position_size(signal_data)
        else:
            # Default to fixed percentage
            size = self.position_sizers[PositionSizingMethod.FIXED_PERCENTAGE].calculate_position_size(signal_data)
        
        # Apply portfolio-level constraints
        size = self._apply_portfolio_constraints(size, signal_data)
        
        return size
    
    def _apply_portfolio_constraints(self, size: float, signal_data: Dict[str, Any]) -> float:
        """Apply portfolio-level risk constraints"""
        symbol = signal_data['symbol']
        price = signal_data['price']
        
        # Check number of positions
        if len(self.positions) >= self.risk_limits.max_positions:
            if symbol not in self.positions:
                logger.warning(f"Maximum positions ({self.risk_limits.max_positions}) reached")
                return 0
        
        # Check correlation risk
        if self._check_correlation_risk(symbol, signal_data):
            logger.warning(f"High correlation risk for {symbol}")
            size *= 0.5  # Reduce position size
        
        # Check concentration risk
        position_value = size * price
        portfolio_weight = position_value / self.current_capital
        
        if portfolio_weight > self.risk_limits.max_position_size_pct:
            size = (self.current_capital * self.risk_limits.max_position_size_pct) / price
        
        # Check VaR limits
        if self._would_exceed_var_limit(symbol, size, signal_data):
            logger.warning(f"Position would exceed VaR limit")
            size *= 0.5
        
        return size
    
    def update_position(self, symbol: str, quantity: float, current_price: float,
                       entry_price: float, market_data: pd.DataFrame = None):
        """Update position risk profile"""
        # Calculate metrics
        position_value = quantity * current_price
        weight = position_value / self.current_capital
        
        # Calculate volatility
        if market_data is not None and len(market_data) > 20:
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate beta (simplified - would need market data)
            beta = 1.0
        else:
            volatility = 0.02  # Default 2%
            beta = 1.0
        
        # Update or create position profile
        self.positions[symbol] = PositionRiskProfile(
            symbol=symbol,
            current_risk=position_value * volatility,
            max_risk=position_value * self.risk_limits.max_position_risk,
            volatility=volatility,
            beta=beta,
            correlation_to_portfolio=self._calculate_correlation(symbol, market_data),
            var_contribution=self._calculate_var_contribution(symbol, position_value, volatility),
            weight=weight
        )
    
    def check_risk_limits(self) -> Tuple[bool, List[str]]:
        """Check if any risk limits are breached"""
        violations = []
        
        # Calculate current metrics
        metrics = self.calculate_risk_metrics()
        
        # Check VaR limit
        if metrics.var_95 > self.risk_limits.max_var_95:
            violations.append(f"VaR exceeds limit: {metrics.var_95:.2%} > {self.risk_limits.max_var_95:.2%}")
        
        # Check drawdown
        if metrics.current_drawdown > self.risk_limits.max_daily_loss:
            violations.append(f"Daily loss exceeds limit: {metrics.current_drawdown:.2%}")
        
        # Check leverage
        if metrics.leverage > self.risk_limits.max_leverage:
            violations.append(f"Leverage exceeds limit: {metrics.leverage:.2f}x")
        
        # Check concentration
        max_weight = max((p.weight for p in self.positions.values()), default=0)
        if max_weight > self.risk_limits.max_position_size_pct:
            violations.append(f"Position concentration too high: {max_weight:.2%}")
        
        return len(violations) == 0, violations
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        # Portfolio value
        positions_value = sum(p.current_risk / p.volatility for p in self.positions.values())
        total_value = self.current_capital
        
        # VaR calculation (simplified)
        portfolio_volatility = self._calculate_portfolio_volatility()
        var_95 = 1.645 * portfolio_volatility  # 95% VaR
        cvar_95 = 2.063 * portfolio_volatility  # Approximate CVaR
        
        # Performance metrics
        if len(self.daily_pnl) > 1:
            returns = pd.Series(self.daily_pnl).pct_change().dropna()
            sharpe = self._calculate_sharpe_ratio(returns)
            sortino = self._calculate_sortino_ratio(returns)
        else:
            sharpe = sortino = 0
        
        # Drawdown
        current_drawdown = (self.high_water_mark - self.current_capital) / self.high_water_mark
        max_drawdown = self._calculate_max_drawdown()
        
        # Risk concentration
        concentration_risk = self._calculate_concentration_risk()
        correlation_risk = self._calculate_correlation_risk_score()
        
        return RiskMetrics(
            total_exposure=positions_value / total_value,
            var_95=var_95,
            cvar_95=cvar_95,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            current_drawdown=current_drawdown,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            leverage=positions_value / self.current_capital,
            margin_usage=0  # Placeholder
        )
    
    def apply_circuit_breakers(self):
        """Check and apply circuit breakers"""
        # Daily loss circuit breaker
        if len(self.daily_pnl) > 0:
            daily_return = (self.current_capital - self.daily_pnl[-1]) / self.daily_pnl[-1]
            if daily_return < -self.risk_limits.max_daily_loss:
                self._halt_trading("Daily loss limit breached", hours=24)
        
        # Drawdown circuit breaker
        metrics = self.calculate_risk_metrics()
        if metrics.current_drawdown > self.risk_limits.max_monthly_loss:
            self._halt_trading("Maximum drawdown breached", hours=72)
        
        # Consecutive losses circuit breaker
        if self._check_consecutive_losses(5):
            self._halt_trading("5 consecutive losses", hours=12)
    
    def _halt_trading(self, reason: str, hours: int):
        """Halt trading for specified hours"""
        self.trading_halted = True
        self.halt_reason = reason
        self.halt_until = datetime.now() + timedelta(hours=hours)
        logger.warning(f"Trading halted: {reason}. Resume at {self.halt_until}")
    
    def _check_consecutive_losses(self, threshold: int) -> bool:
        """Check for consecutive losses"""
        if len(self.trade_history) < threshold:
            return False
        
        return all(trade < 0 for trade in self.trade_history[-threshold:])
    
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility considering correlations"""
        if not self.positions:
            return 0
        
        # Simplified calculation - assumes equal correlation
        avg_volatility = np.mean([p.volatility for p in self.positions.values()])
        num_positions = len(self.positions)
        
        # Diversification benefit (simplified)
        correlation = 0.5  # Assume average correlation
        portfolio_vol = avg_volatility * np.sqrt(
            (1 + (num_positions - 1) * correlation) / num_positions
        )
        
        return portfolio_vol
    
    def _calculate_var_contribution(self, symbol: str, position_value: float, 
                                   volatility: float) -> float:
        """Calculate position's contribution to portfolio VaR"""
        weight = position_value / self.current_capital
        return weight * volatility * 1.645  # 95% VaR
    
    def _check_correlation_risk(self, symbol: str, signal_data: Dict[str, Any]) -> bool:
        """Check if adding position increases correlation risk"""
        # Simplified check - in practice would use actual correlation matrix
        sector = signal_data.get('sector', 'unknown')
        
        # Count positions in same sector
        same_sector_count = sum(
            1 for pos in self.positions.values()
            if pos.symbol != symbol and pos.symbol.startswith(sector[:2])
        )
        
        return same_sector_count >= 3
    
    def _would_exceed_var_limit(self, symbol: str, size: float, 
                               signal_data: Dict[str, Any]) -> bool:
        """Check if position would exceed VaR limit"""
        # Estimate new VaR
        position_value = size * signal_data['price']
        volatility = signal_data.get('volatility', 0.02)
        
        new_var_contribution = self._calculate_var_contribution(symbol, position_value, volatility)
        current_var = sum(p.var_contribution for p in self.positions.values())
        
        return (current_var + new_var_contribution) > self.risk_limits.max_var_95
    
    def _calculate_correlation(self, symbol: str, market_data: pd.DataFrame) -> float:
        """Calculate correlation of symbol returns to existing portfolio returns"""
        if not self.positions or market_data is None or len(market_data) < 20:
            return 0.0  # No correlation data available

        try:
            symbol_returns = market_data['close'].pct_change().dropna()
            if len(symbol_returns) < 10:
                return 0.0

            # Average correlation across existing positions
            correlations = []
            for pos_symbol, pos in self.positions.items():
                if hasattr(pos, 'returns_history') and len(pos.returns_history) >= 10:
                    common_len = min(len(symbol_returns), len(pos.returns_history))
                    corr = np.corrcoef(
                        symbol_returns.values[-common_len:],
                        pos.returns_history[-common_len:]
                    )[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))

            return float(np.mean(correlations)) if correlations else 0.0
        except Exception:
            return 0.0
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate Herfindahl index for concentration"""
        if not self.positions:
            return 0
        
        weights = [p.weight for p in self.positions.values()]
        herfindahl = sum(w**2 for w in weights)
        
        return herfindahl
    
    def _calculate_correlation_risk_score(self) -> float:
        """Calculate overall correlation risk score based on position concentration"""
        if len(self.positions) < 2:
            return 0.0

        # Use concentration as a proxy when full correlation matrix isn't available
        concentration = self._calculate_concentration_risk()
        # Herfindahl of 1.0 = single position (max risk), 1/n = equal weight (min risk)
        n = len(self.positions)
        min_herfindahl = 1.0 / n if n > 0 else 1.0
        # Normalize to 0-1 range
        if min_herfindahl >= 1.0:
            return 0.0
        return min(1.0, (concentration - min_herfindahl) / (1.0 - min_herfindahl))
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        
        return np.sqrt(252) * returns.mean() / returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0
        
        return np.sqrt(252) * returns.mean() / downside_returns.std()
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from PnL history"""
        if not self.daily_pnl:
            return 0
        
        cumulative = np.cumsum(self.daily_pnl)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))
    
    def update_capital(self, new_capital: float):
        """Update current capital"""
        self.current_capital = new_capital
        if new_capital > self.high_water_mark:
            self.high_water_mark = new_capital
    
    def record_trade(self, pnl: float):
        """Record trade result"""
        self.trade_history.append(pnl)
        
        # Update Optimal F sizer if available
        if PositionSizingMethod.OPTIMAL_F in self.position_sizers:
            self.position_sizers[PositionSizingMethod.OPTIMAL_F] = OptimalFSizer(
                self.current_capital, self.risk_limits, self.trade_history
            )
    
    def get_position_sizing_recommendation(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive position sizing recommendation"""
        recommendations = {}
        
        # Calculate size for each method
        for method in PositionSizingMethod:
            try:
                size = self.calculate_position_size(method, signal_data)
                recommendations[method.value] = {
                    'size': size,
                    'value': size * signal_data['price'],
                    'risk': size * signal_data['price'] * signal_data.get('volatility', 0.02)
                }
            except Exception as e:
                logger.error(f"Error calculating {method.value}: {e}")
                recommendations[method.value] = {'size': 0, 'value': 0, 'risk': 0}
        
        # Add risk metrics
        metrics = self.calculate_risk_metrics()
        
        return {
            'recommendations': recommendations,
            'current_risk_metrics': {
                'var_95': metrics.var_95,
                'sharpe_ratio': metrics.sharpe_ratio,
                'current_drawdown': metrics.current_drawdown,
                'positions_count': len(self.positions)
            },
            'risk_capacity': {
                'remaining_positions': self.risk_limits.max_positions - len(self.positions),
                'remaining_var': self.risk_limits.max_var_95 - metrics.var_95,
                'can_trade': not self.trading_halted
            }
        }