#!/usr/bin/env python3
"""
Risk Intelligence Engine
========================

A comprehensive risk management and analysis system for the trading bot.

Features:
- Real-time VaR Calculation (Historical, Parametric, Monte Carlo)
- Portfolio Correlation Monitoring with regime change detection
- Market Regime Detection (volatility, trend, microstructure)
- Strategy Performance Attribution and quarantine system
- Risk Limits Enforcement with automatic position reduction

Author: Trading Bot System
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import logging
import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import comb

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger('RiskIntelligence')


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class VaRMethod(Enum):
    """Value at Risk calculation methods."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


class VolatilityRegime(Enum):
    """Market volatility regimes."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


class TrendRegime(Enum):
    """Market trend regimes."""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    RANGING = "ranging"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    CHOPPY = "choppy"


class LiquidityRegime(Enum):
    """Market liquidity regimes."""
    HIGHLY_LIQUID = "highly_liquid"
    LIQUID = "liquid"
    NORMAL = "normal"
    ILLIQUID = "illiquid"
    HIGHLY_ILLIQUID = "highly_illiquid"


class RiskLimitType(Enum):
    """Types of risk limits."""
    DAILY_LOSS = "daily_loss"
    WEEKLY_LOSS = "weekly_loss"
    MONTHLY_LOSS = "monthly_loss"
    MAX_DRAWDOWN = "max_drawdown"
    POSITION_CONCENTRATION = "position_concentration"
    SECTOR_EXPOSURE = "sector_exposure"
    LEVERAGE = "leverage"
    VAR_LIMIT = "var_limit"
    CORRELATION = "correlation"


class StrategyStatus(Enum):
    """Strategy operational status."""
    ACTIVE = "active"
    MONITORING = "monitoring"
    QUARANTINED = "quarantined"
    DISABLED = "disabled"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VaRResult:
    """Result of VaR calculation."""
    method: VaRMethod
    confidence_level: float
    var_value: float  # As a percentage of portfolio
    var_dollar: float  # In dollar terms
    cvar_value: float  # Conditional VaR (Expected Shortfall)
    cvar_dollar: float
    time_horizon_days: int
    calculation_timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'method': self.method.value,
            'confidence_level': self.confidence_level,
            'var_value': self.var_value,
            'var_dollar': self.var_dollar,
            'cvar_value': self.cvar_value,
            'cvar_dollar': self.cvar_dollar,
            'time_horizon_days': self.time_horizon_days,
            'timestamp': self.calculation_timestamp.isoformat()
        }


@dataclass
class CorrelationAnalysis:
    """Portfolio correlation analysis result."""
    correlation_matrix: np.ndarray
    symbols: List[str]
    avg_correlation: float
    max_correlation: float
    min_correlation: float
    diversification_score: float  # 0-1, higher is better
    correlation_clusters: List[List[str]]  # Groups of highly correlated assets
    eigenvalues: np.ndarray  # For PCA analysis
    effective_assets: float  # Number of effective independent assets
    regime_change_detected: bool = False
    regime_change_details: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbols': self.symbols,
            'avg_correlation': self.avg_correlation,
            'max_correlation': self.max_correlation,
            'min_correlation': self.min_correlation,
            'diversification_score': self.diversification_score,
            'correlation_clusters': self.correlation_clusters,
            'effective_assets': self.effective_assets,
            'regime_change_detected': self.regime_change_detected,
            'regime_change_details': self.regime_change_details,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RegimeState:
    """Current market regime state."""
    volatility_regime: VolatilityRegime
    trend_regime: TrendRegime
    liquidity_regime: LiquidityRegime
    volatility_percentile: float  # 0-100
    trend_strength: float  # -1 to 1
    liquidity_score: float  # 0-1
    regime_confidence: float  # 0-1
    hmm_state: Optional[int] = None  # Hidden Markov Model state
    transition_probability: Optional[float] = None  # Probability of regime change
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'volatility_regime': self.volatility_regime.value,
            'trend_regime': self.trend_regime.value,
            'liquidity_regime': self.liquidity_regime.value,
            'volatility_percentile': self.volatility_percentile,
            'trend_strength': self.trend_strength,
            'liquidity_score': self.liquidity_score,
            'regime_confidence': self.regime_confidence,
            'hmm_state': self.hmm_state,
            'transition_probability': self.transition_probability,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class StrategyPerformance:
    """Performance metrics for a single strategy."""
    strategy_name: str
    total_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    calmar_ratio: float
    contribution_to_portfolio: float  # Percentage of total P&L
    correlation_to_portfolio: float
    status: StrategyStatus = StrategyStatus.ACTIVE
    quarantine_reason: Optional[str] = None
    quarantine_until: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'profit_factor': self.profit_factor,
            'calmar_ratio': self.calmar_ratio,
            'contribution_to_portfolio': self.contribution_to_portfolio,
            'status': self.status.value,
            'quarantine_reason': self.quarantine_reason,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class RiskLimitViolation:
    """Represents a risk limit violation."""
    limit_type: RiskLimitType
    current_value: float
    limit_value: float
    violation_severity: float  # How much over the limit (percentage)
    timestamp: datetime = field(default_factory=datetime.now)
    auto_action_taken: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'limit_type': self.limit_type.value,
            'current_value': self.current_value,
            'limit_value': self.limit_value,
            'violation_severity': self.violation_severity,
            'timestamp': self.timestamp.isoformat(),
            'auto_action_taken': self.auto_action_taken
        }


@dataclass
class RiskLimitConfig:
    """Configuration for risk limits."""
    # Daily limits
    max_daily_loss_pct: float = 0.02  # 2%
    max_daily_loss_abs: float = float('inf')  # Absolute dollar amount

    # Weekly/Monthly limits
    max_weekly_loss_pct: float = 0.05  # 5%
    max_monthly_loss_pct: float = 0.10  # 10%

    # Drawdown limits
    max_drawdown_pct: float = 0.15  # 15%
    drawdown_reduction_threshold: float = 0.10  # Start reducing at 10%

    # Position limits
    max_position_concentration: float = 0.15  # 15% in single position
    max_sector_exposure: float = 0.30  # 30% in single sector
    max_correlation_exposure: float = 0.50  # 50% in correlated assets

    # VaR limits
    max_var_95: float = 0.03  # 3% VaR at 95%
    max_var_99: float = 0.05  # 5% VaR at 99%

    # Leverage
    max_leverage: float = 1.0  # No leverage by default

    # Strategy quarantine thresholds
    strategy_max_consecutive_losses: int = 5
    strategy_max_drawdown: float = 0.15  # 15% drawdown triggers quarantine
    strategy_min_sharpe: float = -0.5  # Negative Sharpe triggers quarantine
    quarantine_duration_hours: int = 24


@dataclass
class RiskIntelligenceReport:
    """Comprehensive risk intelligence report."""
    timestamp: datetime
    portfolio_value: float

    # VaR metrics
    var_results: Dict[str, VaRResult]

    # Correlation analysis
    correlation_analysis: CorrelationAnalysis

    # Regime state
    regime_state: RegimeState

    # Strategy performance
    strategy_performances: Dict[str, StrategyPerformance]

    # Risk limit status
    risk_limit_violations: List[RiskLimitViolation]
    all_limits_ok: bool

    # Recommended actions
    recommended_actions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': self.portfolio_value,
            'var_results': {k: v.to_dict() for k, v in self.var_results.items()},
            'correlation_analysis': self.correlation_analysis.to_dict(),
            'regime_state': self.regime_state.to_dict(),
            'strategy_performances': {k: v.to_dict() for k, v in self.strategy_performances.items()},
            'risk_limit_violations': [v.to_dict() for v in self.risk_limit_violations],
            'all_limits_ok': self.all_limits_ok,
            'recommended_actions': self.recommended_actions
        }


# =============================================================================
# VAR CALCULATOR
# =============================================================================

class VaRCalculator:
    """
    Value at Risk Calculator.

    Implements three VaR calculation methods:
    - Historical VaR: Based on historical return distribution
    - Parametric VaR: Assumes normal distribution
    - Monte Carlo VaR: Simulation-based approach
    """

    def __init__(
        self,
        confidence_levels: List[float] = None,
        rolling_window: int = 252,
        monte_carlo_simulations: int = 10000
    ):
        """
        Initialize VaR Calculator.

        Args:
            confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
            rolling_window: Number of days for historical calculation
            monte_carlo_simulations: Number of Monte Carlo simulations
        """
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.rolling_window = rolling_window
        self.monte_carlo_simulations = monte_carlo_simulations

        # Cache for performance
        self._returns_cache: Dict[str, pd.Series] = {}
        self._cache_timestamp: Optional[datetime] = None

    def calculate_all(
        self,
        returns: Union[pd.Series, np.ndarray],
        portfolio_value: float,
        time_horizon_days: int = 1
    ) -> Dict[str, VaRResult]:
        """
        Calculate VaR using all methods.

        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            time_horizon_days: Time horizon for VaR calculation

        Returns:
            Dictionary of VaR results by method
        """
        results = {}

        for method in VaRMethod:
            for confidence in self.confidence_levels:
                key = f"{method.value}_{int(confidence * 100)}"
                try:
                    if method == VaRMethod.HISTORICAL:
                        var_result = self._calculate_historical_var(
                            returns, portfolio_value, confidence, time_horizon_days
                        )
                    elif method == VaRMethod.PARAMETRIC:
                        var_result = self._calculate_parametric_var(
                            returns, portfolio_value, confidence, time_horizon_days
                        )
                    else:  # MONTE_CARLO
                        var_result = self._calculate_monte_carlo_var(
                            returns, portfolio_value, confidence, time_horizon_days
                        )
                    results[key] = var_result
                except Exception as e:
                    logger.error(f"Error calculating {key} VaR: {e}")

        return results

    def _calculate_historical_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        portfolio_value: float,
        confidence: float,
        time_horizon_days: int
    ) -> VaRResult:
        """
        Calculate Historical VaR.

        Uses the actual historical return distribution without assuming
        any particular distribution shape.
        """
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) < self.rolling_window:
            returns_window = returns
        else:
            returns_window = returns[-self.rolling_window:]

        # Scale returns for time horizon
        if time_horizon_days > 1:
            # Use square root of time scaling
            scaled_returns = returns_window * np.sqrt(time_horizon_days)
        else:
            scaled_returns = returns_window

        # Calculate VaR as the percentile of losses
        var_percentile = (1 - confidence) * 100
        var_value = np.percentile(scaled_returns, var_percentile)

        # VaR is typically reported as positive number for losses
        var_value = -var_value if var_value < 0 else 0
        var_dollar = var_value * portfolio_value

        # Calculate CVaR (Expected Shortfall)
        threshold = np.percentile(scaled_returns, var_percentile)
        cvar_returns = scaled_returns[scaled_returns <= threshold]
        cvar_value = -np.mean(cvar_returns) if len(cvar_returns) > 0 else var_value
        cvar_dollar = cvar_value * portfolio_value

        return VaRResult(
            method=VaRMethod.HISTORICAL,
            confidence_level=confidence,
            var_value=var_value,
            var_dollar=var_dollar,
            cvar_value=cvar_value,
            cvar_dollar=cvar_dollar,
            time_horizon_days=time_horizon_days
        )

    def _calculate_parametric_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        portfolio_value: float,
        confidence: float,
        time_horizon_days: int
    ) -> VaRResult:
        """
        Calculate Parametric (Variance-Covariance) VaR.

        Assumes returns follow a normal distribution.
        """
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            raise ValueError("Insufficient data for parametric VaR")

        # Calculate mean and standard deviation
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        # Scale for time horizon
        mu_scaled = mu * time_horizon_days
        sigma_scaled = sigma * np.sqrt(time_horizon_days)

        # Calculate VaR using inverse normal distribution
        z_score = stats.norm.ppf(1 - confidence)
        var_value = -(mu_scaled + z_score * sigma_scaled)
        var_value = max(0, var_value)  # VaR should be non-negative
        var_dollar = var_value * portfolio_value

        # Calculate CVaR for normal distribution
        # CVaR = mu + sigma * phi(z) / (1 - alpha)
        # where phi is the standard normal PDF
        phi_z = stats.norm.pdf(z_score)
        cvar_value = -(mu_scaled - sigma_scaled * phi_z / (1 - confidence))
        cvar_value = max(var_value, cvar_value)
        cvar_dollar = cvar_value * portfolio_value

        return VaRResult(
            method=VaRMethod.PARAMETRIC,
            confidence_level=confidence,
            var_value=var_value,
            var_dollar=var_dollar,
            cvar_value=cvar_value,
            cvar_dollar=cvar_dollar,
            time_horizon_days=time_horizon_days
        )

    def _calculate_monte_carlo_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        portfolio_value: float,
        confidence: float,
        time_horizon_days: int
    ) -> VaRResult:
        """
        Calculate Monte Carlo VaR.

        Simulates future returns using Geometric Brownian Motion.
        """
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            raise ValueError("Insufficient data for Monte Carlo VaR")

        # Estimate parameters from historical data
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)

        # Simulate future portfolio values
        np.random.seed(42)  # For reproducibility

        # Generate random returns using GBM-inspired model
        z = np.random.standard_normal((self.monte_carlo_simulations, time_horizon_days))

        # Calculate cumulative returns
        simulated_returns = mu - 0.5 * sigma**2 + sigma * z
        cumulative_returns = np.sum(simulated_returns, axis=1)

        # Calculate VaR from simulated distribution
        var_percentile = (1 - confidence) * 100
        var_value = -np.percentile(cumulative_returns, var_percentile)
        var_value = max(0, var_value)
        var_dollar = var_value * portfolio_value

        # Calculate CVaR
        threshold = np.percentile(cumulative_returns, var_percentile)
        cvar_returns = cumulative_returns[cumulative_returns <= threshold]
        cvar_value = -np.mean(cvar_returns) if len(cvar_returns) > 0 else var_value
        cvar_dollar = cvar_value * portfolio_value

        return VaRResult(
            method=VaRMethod.MONTE_CARLO,
            confidence_level=confidence,
            var_value=var_value,
            var_dollar=var_dollar,
            cvar_value=cvar_value,
            cvar_dollar=cvar_dollar,
            time_horizon_days=time_horizon_days
        )

    def calculate_component_var(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        portfolio_var: float
    ) -> np.ndarray:
        """
        Calculate component VaR (contribution of each asset to portfolio VaR).

        Args:
            weights: Portfolio weights
            cov_matrix: Covariance matrix
            portfolio_var: Total portfolio VaR

        Returns:
            Array of component VaR values
        """
        # Marginal VaR = partial derivative of VaR with respect to weight
        marginal_var = cov_matrix @ weights / np.sqrt(weights @ cov_matrix @ weights)

        # Component VaR = weight * marginal VaR
        component_var = weights * marginal_var

        # Scale to match portfolio VaR
        component_var = component_var * portfolio_var / np.sum(component_var)

        return component_var

    def calculate_incremental_var(
        self,
        current_returns: np.ndarray,
        new_position_returns: np.ndarray,
        current_weights: np.ndarray,
        new_weight: float,
        portfolio_value: float,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate incremental VaR from adding a new position.

        Args:
            current_returns: Returns matrix for current positions
            new_position_returns: Returns for the new position
            current_weights: Current portfolio weights
            new_weight: Weight of new position
            portfolio_value: Portfolio value
            confidence: Confidence level

        Returns:
            Incremental VaR in dollar terms
        """
        # Calculate current portfolio VaR
        current_portfolio_returns = current_returns @ current_weights
        current_var = self._calculate_historical_var(
            current_portfolio_returns, portfolio_value, confidence, 1
        ).var_dollar

        # Rebalance weights for new position
        adjusted_weights = current_weights * (1 - new_weight)
        new_weights = np.append(adjusted_weights, new_weight)

        # Calculate new portfolio returns
        all_returns = np.column_stack([current_returns, new_position_returns])
        new_portfolio_returns = all_returns @ new_weights

        new_var = self._calculate_historical_var(
            new_portfolio_returns, portfolio_value, confidence, 1
        ).var_dollar

        return new_var - current_var


# =============================================================================
# CORRELATION MONITOR
# =============================================================================

class CorrelationMonitor:
    """
    Portfolio Correlation Monitoring System.

    Monitors real-time correlations, detects regime changes,
    and calculates diversification metrics.
    """

    def __init__(
        self,
        lookback_window: int = 60,
        regime_change_threshold: float = 0.20,
        correlation_update_frequency: int = 1  # Days
    ):
        """
        Initialize Correlation Monitor.

        Args:
            lookback_window: Days of data for correlation calculation
            regime_change_threshold: Threshold for detecting regime changes
            correlation_update_frequency: How often to update correlations
        """
        self.lookback_window = lookback_window
        self.regime_change_threshold = regime_change_threshold
        self.correlation_update_frequency = correlation_update_frequency

        # Historical correlation storage
        self._correlation_history: List[np.ndarray] = []
        self._correlation_timestamps: List[datetime] = []
        self._previous_correlation: Optional[np.ndarray] = None

    def analyze(
        self,
        returns_data: pd.DataFrame,
        symbols: List[str]
    ) -> CorrelationAnalysis:
        """
        Perform comprehensive correlation analysis.

        Args:
            returns_data: DataFrame with returns for each symbol
            symbols: List of symbols in the portfolio

        Returns:
            CorrelationAnalysis object
        """
        # Calculate correlation matrix
        correlation_matrix = returns_data.corr().values

        # Handle NaN values
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)

        # Calculate statistics
        upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]

        avg_correlation = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0
        max_correlation = np.max(upper_triangle) if len(upper_triangle) > 0 else 0
        min_correlation = np.min(upper_triangle) if len(upper_triangle) > 0 else 0

        # Calculate diversification score
        diversification_score = self._calculate_diversification_score(correlation_matrix)

        # Find correlation clusters
        correlation_clusters = self._find_correlation_clusters(
            correlation_matrix, symbols, threshold=0.7
        )

        # PCA analysis
        eigenvalues, effective_assets = self._calculate_effective_assets(correlation_matrix)

        # Detect regime change
        regime_change_detected, regime_change_details = self._detect_regime_change(
            correlation_matrix
        )

        # Store for history
        self._update_history(correlation_matrix)

        return CorrelationAnalysis(
            correlation_matrix=correlation_matrix,
            symbols=symbols,
            avg_correlation=avg_correlation,
            max_correlation=max_correlation,
            min_correlation=min_correlation,
            diversification_score=diversification_score,
            correlation_clusters=correlation_clusters,
            eigenvalues=eigenvalues,
            effective_assets=effective_assets,
            regime_change_detected=regime_change_detected,
            regime_change_details=regime_change_details
        )

    def _calculate_diversification_score(self, correlation_matrix: np.ndarray) -> float:
        """
        Calculate diversification score (0-1).

        Higher score means better diversification.
        Uses the formula: 1 - avg_correlation
        """
        n = correlation_matrix.shape[0]
        if n <= 1:
            return 1.0

        # Get average pairwise correlation (excluding diagonal)
        total = np.sum(correlation_matrix) - n  # Subtract diagonal sum
        count = n * (n - 1)
        avg_correlation = total / count if count > 0 else 0

        # Score is inverse of average correlation
        score = 1 - abs(avg_correlation)
        return max(0, min(1, score))

    def _find_correlation_clusters(
        self,
        correlation_matrix: np.ndarray,
        symbols: List[str],
        threshold: float = 0.7
    ) -> List[List[str]]:
        """
        Find groups of highly correlated assets using simple clustering.
        """
        n = len(symbols)
        if n <= 1:
            return [symbols] if symbols else []

        # Simple agglomerative approach
        clusters = [[s] for s in symbols]
        visited = set()

        for i in range(n):
            if i in visited:
                continue

            cluster = [symbols[i]]
            for j in range(i + 1, n):
                if j not in visited and correlation_matrix[i, j] >= threshold:
                    cluster.append(symbols[j])
                    visited.add(j)

            if len(cluster) > 1:
                # Merge into existing cluster or create new
                merged = False
                for existing in clusters:
                    if any(s in existing for s in cluster):
                        for s in cluster:
                            if s not in existing:
                                existing.append(s)
                        merged = True
                        break

                if not merged:
                    clusters.append(cluster)

        # Filter clusters with more than 1 member
        return [c for c in clusters if len(c) > 1]

    def _calculate_effective_assets(
        self,
        correlation_matrix: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate effective number of independent assets using PCA.

        Returns:
            Tuple of (eigenvalues, effective_assets)
        """
        if correlation_matrix.shape[0] <= 1:
            return np.array([1.0]), 1.0

        try:
            eigenvalues = np.linalg.eigvalsh(correlation_matrix)
            eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

            # Effective number of assets (using Participation Ratio)
            if np.sum(eigenvalues**2) > 0:
                effective_assets = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)
            else:
                effective_assets = 1.0

            return eigenvalues, effective_assets

        except Exception as e:
            logger.warning(f"Error in PCA calculation: {e}")
            n = correlation_matrix.shape[0]
            return np.ones(n), float(n)

    def _detect_regime_change(
        self,
        current_correlation: np.ndarray
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect correlation regime changes.

        Compares current correlation matrix to historical average.
        """
        if self._previous_correlation is None:
            self._previous_correlation = current_correlation
            return False, None

        # Compare current to previous
        if current_correlation.shape != self._previous_correlation.shape:
            self._previous_correlation = current_correlation
            return False, "Portfolio composition changed"

        # Calculate Frobenius norm of difference
        diff = current_correlation - self._previous_correlation
        frobenius_distance = np.sqrt(np.sum(diff**2))

        # Normalize by matrix size
        n = current_correlation.shape[0]
        normalized_distance = frobenius_distance / n

        self._previous_correlation = current_correlation.copy()

        if normalized_distance > self.regime_change_threshold:
            # Determine direction of change
            current_avg = np.mean(current_correlation[np.triu_indices_from(current_correlation, k=1)])
            previous_avg = np.mean(self._previous_correlation[np.triu_indices_from(self._previous_correlation, k=1)])

            if current_avg > previous_avg:
                details = f"Correlations increasing (avg change: {current_avg - previous_avg:.3f})"
            else:
                details = f"Correlations decreasing (avg change: {current_avg - previous_avg:.3f})"

            return True, details

        return False, None

    def _update_history(self, correlation_matrix: np.ndarray):
        """Update correlation history."""
        self._correlation_history.append(correlation_matrix.copy())
        self._correlation_timestamps.append(datetime.now())

        # Keep only recent history
        max_history = 100
        if len(self._correlation_history) > max_history:
            self._correlation_history = self._correlation_history[-max_history:]
            self._correlation_timestamps = self._correlation_timestamps[-max_history:]

    def get_rolling_correlation(
        self,
        symbol1: str,
        symbol2: str,
        returns_data: pd.DataFrame,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate rolling correlation between two symbols.

        Args:
            symbol1: First symbol
            symbol2: Second symbol
            returns_data: DataFrame with returns
            window: Rolling window size

        Returns:
            Series of rolling correlations
        """
        if symbol1 not in returns_data.columns or symbol2 not in returns_data.columns:
            raise ValueError(f"Symbols not found in returns data")

        return returns_data[symbol1].rolling(window).corr(returns_data[symbol2])


# =============================================================================
# REGIME DETECTOR
# =============================================================================

class RegimeDetector:
    """
    Market Regime Detection System.

    Detects volatility, trend, and liquidity regimes using
    statistical methods and Hidden Markov Models.
    """

    def __init__(
        self,
        volatility_lookback: int = 60,
        trend_lookback: int = 20,
        hmm_states: int = 3
    ):
        """
        Initialize Regime Detector.

        Args:
            volatility_lookback: Days for volatility calculation
            trend_lookback: Days for trend detection
            hmm_states: Number of HMM states
        """
        self.volatility_lookback = volatility_lookback
        self.trend_lookback = trend_lookback
        self.hmm_states = hmm_states

        # Historical volatility for percentile calculation
        self._volatility_history: List[float] = []

        # HMM parameters (simple implementation)
        self._hmm_transition_matrix: Optional[np.ndarray] = None
        self._hmm_emission_means: Optional[np.ndarray] = None
        self._hmm_emission_stds: Optional[np.ndarray] = None

    def detect_regime(
        self,
        price_data: pd.DataFrame,
        volume_data: Optional[pd.Series] = None
    ) -> RegimeState:
        """
        Detect current market regime.

        Args:
            price_data: DataFrame with OHLCV data
            volume_data: Optional volume data

        Returns:
            RegimeState object
        """
        # Extract close prices
        if 'close' in price_data.columns:
            close = price_data['close']
        else:
            close = price_data.iloc[:, 0]  # Assume first column is price

        # Calculate returns
        returns = close.pct_change().dropna()

        # Detect volatility regime
        volatility_regime, vol_percentile = self._detect_volatility_regime(returns)

        # Detect trend regime
        trend_regime, trend_strength = self._detect_trend_regime(close)

        # Detect liquidity regime
        if volume_data is not None or 'volume' in price_data.columns:
            vol = volume_data if volume_data is not None else price_data['volume']
            liquidity_regime, liquidity_score = self._detect_liquidity_regime(vol, close)
        else:
            liquidity_regime = LiquidityRegime.NORMAL
            liquidity_score = 0.5

        # Calculate regime confidence
        regime_confidence = self._calculate_regime_confidence(
            returns, volatility_regime, trend_regime
        )

        # HMM state detection (if enough data)
        hmm_state, transition_prob = self._detect_hmm_state(returns)

        return RegimeState(
            volatility_regime=volatility_regime,
            trend_regime=trend_regime,
            liquidity_regime=liquidity_regime,
            volatility_percentile=vol_percentile,
            trend_strength=trend_strength,
            liquidity_score=liquidity_score,
            regime_confidence=regime_confidence,
            hmm_state=hmm_state,
            transition_probability=transition_prob
        )

    def _detect_volatility_regime(
        self,
        returns: pd.Series
    ) -> Tuple[VolatilityRegime, float]:
        """
        Detect volatility regime using rolling volatility.
        """
        # Calculate current volatility (annualized)
        current_vol = returns.tail(self.volatility_lookback).std() * np.sqrt(252)

        # Update history
        self._volatility_history.append(current_vol)
        if len(self._volatility_history) > 1000:
            self._volatility_history = self._volatility_history[-1000:]

        # Calculate percentile
        if len(self._volatility_history) > 20:
            vol_percentile = stats.percentileofscore(
                self._volatility_history, current_vol
            )
        else:
            # Use typical market volatility distribution
            # Average market vol ~15-20%, std ~5%
            vol_percentile = stats.norm.cdf(current_vol, loc=0.17, scale=0.05) * 100

        # Classify regime
        if vol_percentile < 20:
            regime = VolatilityRegime.LOW
        elif vol_percentile < 60:
            regime = VolatilityRegime.NORMAL
        elif vol_percentile < 90:
            regime = VolatilityRegime.HIGH
        else:
            regime = VolatilityRegime.EXTREME

        return regime, vol_percentile

    def _detect_trend_regime(
        self,
        prices: pd.Series
    ) -> Tuple[TrendRegime, float]:
        """
        Detect trend regime using multiple indicators.
        """
        if len(prices) < self.trend_lookback * 2:
            return TrendRegime.RANGING, 0.0

        # Calculate trend metrics
        short_ma = prices.rolling(self.trend_lookback).mean()
        long_ma = prices.rolling(self.trend_lookback * 2).mean()

        # Linear regression for trend strength
        x = np.arange(self.trend_lookback)
        y = prices.tail(self.trend_lookback).values

        if len(y) < self.trend_lookback:
            return TrendRegime.RANGING, 0.0

        # Calculate slope
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Normalize slope by price level
        normalized_slope = slope / np.mean(y) * self.trend_lookback

        # Calculate ADX-like trend strength (simplified)
        price_changes = prices.diff().tail(self.trend_lookback)
        up_moves = price_changes.where(price_changes > 0, 0)
        down_moves = (-price_changes).where(price_changes < 0, 0)

        plus_di = up_moves.sum() / (up_moves.sum() + down_moves.sum() + 1e-10)
        minus_di = down_moves.sum() / (up_moves.sum() + down_moves.sum() + 1e-10)

        dx = abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_proxy = dx * r_value**2  # Scale by trend fit quality

        # Choppiness indicator
        highest = prices.tail(self.trend_lookback).max()
        lowest = prices.tail(self.trend_lookback).min()
        atr_sum = abs(prices.diff()).tail(self.trend_lookback).sum()

        if highest != lowest:
            choppiness = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(self.trend_lookback)
        else:
            choppiness = 50

        # Classify regime
        trend_strength = normalized_slope

        if choppiness > 61.8:  # Fibonacci level
            regime = TrendRegime.CHOPPY
        elif abs(normalized_slope) < 0.02 and adx_proxy < 0.25:
            regime = TrendRegime.RANGING
        elif normalized_slope > 0.05:
            regime = TrendRegime.STRONG_UPTREND if adx_proxy > 0.4 else TrendRegime.UPTREND
        elif normalized_slope < -0.05:
            regime = TrendRegime.STRONG_DOWNTREND if adx_proxy > 0.4 else TrendRegime.DOWNTREND
        elif normalized_slope > 0:
            regime = TrendRegime.UPTREND
        else:
            regime = TrendRegime.DOWNTREND

        return regime, trend_strength

    def _detect_liquidity_regime(
        self,
        volume: pd.Series,
        prices: pd.Series
    ) -> Tuple[LiquidityRegime, float]:
        """
        Detect liquidity regime using volume and spread proxies.
        """
        # Calculate relative volume
        avg_volume = volume.rolling(20).mean()
        relative_volume = (volume / avg_volume).tail(5).mean()

        # Calculate Amihud illiquidity ratio (simplified)
        returns = prices.pct_change().abs()
        dollar_volume = prices * volume

        illiquidity = (returns / dollar_volume).tail(20).mean() * 1e9

        # Normalize to 0-1 score
        # Lower illiquidity = higher liquidity
        liquidity_score = 1 / (1 + illiquidity * 1000)

        # Adjust by relative volume
        liquidity_score = liquidity_score * min(relative_volume, 2) / 2
        liquidity_score = max(0, min(1, liquidity_score))

        # Classify regime
        if liquidity_score > 0.8:
            regime = LiquidityRegime.HIGHLY_LIQUID
        elif liquidity_score > 0.6:
            regime = LiquidityRegime.LIQUID
        elif liquidity_score > 0.4:
            regime = LiquidityRegime.NORMAL
        elif liquidity_score > 0.2:
            regime = LiquidityRegime.ILLIQUID
        else:
            regime = LiquidityRegime.HIGHLY_ILLIQUID

        return regime, liquidity_score

    def _calculate_regime_confidence(
        self,
        returns: pd.Series,
        volatility_regime: VolatilityRegime,
        trend_regime: TrendRegime
    ) -> float:
        """
        Calculate confidence in regime detection.
        """
        # Use stability of recent regime indicators
        if len(returns) < 20:
            return 0.5

        # Calculate rolling volatility stability
        rolling_vol = returns.rolling(10).std()
        vol_stability = 1 - rolling_vol.tail(10).std() / (rolling_vol.tail(10).mean() + 1e-10)

        # Calculate trend consistency
        signs = np.sign(returns.tail(20))
        trend_consistency = abs(signs.sum()) / 20

        # Combine metrics
        confidence = 0.5 * vol_stability + 0.5 * trend_consistency
        return max(0, min(1, confidence))

    def _detect_hmm_state(
        self,
        returns: pd.Series
    ) -> Tuple[Optional[int], Optional[float]]:
        """
        Detect Hidden Markov Model state.

        Simple implementation without external HMM library.
        """
        if len(returns) < 100:
            return None, None

        # Initialize HMM if needed
        if self._hmm_transition_matrix is None:
            self._initialize_hmm(returns)

        if self._hmm_emission_means is None:
            return None, None

        # Calculate state probabilities based on current observation
        current_return = returns.iloc[-1]

        state_probs = np.zeros(self.hmm_states)
        for s in range(self.hmm_states):
            state_probs[s] = stats.norm.pdf(
                current_return,
                loc=self._hmm_emission_means[s],
                scale=self._hmm_emission_stds[s]
            )

        state_probs = state_probs / (np.sum(state_probs) + 1e-10)

        # Most likely state
        current_state = np.argmax(state_probs)

        # Transition probability (probability of leaving current state)
        transition_prob = 1 - self._hmm_transition_matrix[current_state, current_state]

        return int(current_state), float(transition_prob)

    def _initialize_hmm(self, returns: pd.Series):
        """
        Initialize HMM parameters using simple K-means-like clustering.
        """
        returns_array = returns.values

        # Initialize states by percentiles
        percentiles = [100 * (i + 1) / (self.hmm_states + 1)
                      for i in range(self.hmm_states)]

        self._hmm_emission_means = np.array([
            np.percentile(returns_array, p) for p in percentiles
        ])

        # Estimate standard deviations for each state
        self._hmm_emission_stds = np.full(
            self.hmm_states,
            np.std(returns_array) / np.sqrt(self.hmm_states)
        )

        # Initialize transition matrix (favor staying in current state)
        self._hmm_transition_matrix = np.full(
            (self.hmm_states, self.hmm_states),
            0.1 / (self.hmm_states - 1)
        )
        np.fill_diagonal(self._hmm_transition_matrix, 0.9)


# =============================================================================
# STRATEGY PERFORMANCE TRACKER
# =============================================================================

class StrategyPerformanceTracker:
    """
    Tracks and analyzes performance of individual trading strategies.

    Provides per-strategy P&L tracking, risk metrics, and automatic
    quarantine functionality for underperforming strategies.
    """

    def __init__(self, config: RiskLimitConfig = None):
        """
        Initialize Strategy Performance Tracker.

        Args:
            config: Risk limit configuration
        """
        self.config = config or RiskLimitConfig()

        # Strategy tracking
        self._strategies: Dict[str, StrategyPerformance] = {}
        self._trade_history: Dict[str, List[Dict[str, Any]]] = {}
        self._daily_returns: Dict[str, List[float]] = {}
        self._portfolio_returns: List[float] = []

    def register_strategy(self, strategy_name: str):
        """Register a new strategy for tracking."""
        if strategy_name not in self._strategies:
            self._strategies[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name,
                total_pnl=0.0,
                realized_pnl=0.0,
                unrealized_pnl=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                current_drawdown=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                calmar_ratio=0.0,
                contribution_to_portfolio=0.0,
                correlation_to_portfolio=0.0
            )
            self._trade_history[strategy_name] = []
            self._daily_returns[strategy_name] = []

    def record_trade(
        self,
        strategy_name: str,
        pnl: float,
        trade_details: Optional[Dict[str, Any]] = None
    ):
        """
        Record a completed trade for a strategy.

        Args:
            strategy_name: Name of the strategy
            pnl: Profit/loss of the trade
            trade_details: Optional additional trade information
        """
        if strategy_name not in self._strategies:
            self.register_strategy(strategy_name)

        # Record trade
        trade_record = {
            'pnl': pnl,
            'timestamp': datetime.now(),
            **(trade_details or {})
        }
        self._trade_history[strategy_name].append(trade_record)

        # Update strategy metrics
        self._update_strategy_metrics(strategy_name)

        # Check for quarantine conditions
        self._check_quarantine_conditions(strategy_name)

    def update_unrealized_pnl(self, strategy_name: str, unrealized_pnl: float):
        """Update unrealized P&L for a strategy."""
        if strategy_name in self._strategies:
            self._strategies[strategy_name].unrealized_pnl = unrealized_pnl
            self._strategies[strategy_name].total_pnl = (
                self._strategies[strategy_name].realized_pnl + unrealized_pnl
            )

    def record_daily_return(self, strategy_name: str, daily_return: float):
        """Record daily return for a strategy."""
        if strategy_name not in self._daily_returns:
            self._daily_returns[strategy_name] = []

        self._daily_returns[strategy_name].append(daily_return)

        # Keep last 252 days (1 year)
        if len(self._daily_returns[strategy_name]) > 252:
            self._daily_returns[strategy_name] = self._daily_returns[strategy_name][-252:]

    def record_portfolio_return(self, portfolio_return: float):
        """Record portfolio-level daily return."""
        self._portfolio_returns.append(portfolio_return)

        if len(self._portfolio_returns) > 252:
            self._portfolio_returns = self._portfolio_returns[-252:]

    def _update_strategy_metrics(self, strategy_name: str):
        """Update all metrics for a strategy."""
        trades = self._trade_history[strategy_name]
        if not trades:
            return

        strategy = self._strategies[strategy_name]

        # Basic trade statistics
        pnls = [t['pnl'] for t in trades]
        strategy.total_trades = len(trades)
        strategy.winning_trades = sum(1 for p in pnls if p > 0)
        strategy.losing_trades = sum(1 for p in pnls if p < 0)
        strategy.win_rate = strategy.winning_trades / strategy.total_trades

        # P&L statistics
        strategy.realized_pnl = sum(pnls)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        strategy.avg_win = np.mean(wins) if wins else 0.0
        strategy.avg_loss = np.mean(losses) if losses else 0.0

        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        strategy.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Performance ratios from daily returns
        if strategy_name in self._daily_returns and len(self._daily_returns[strategy_name]) > 20:
            returns = np.array(self._daily_returns[strategy_name])

            # Sharpe Ratio (annualized)
            if np.std(returns) > 0:
                strategy.sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)

            # Sortino Ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                strategy.sortino_ratio = np.sqrt(252) * np.mean(returns) / np.std(downside_returns)

            # Drawdown calculations
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = cumulative - running_max

            strategy.max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
            strategy.current_drawdown = abs(drawdowns[-1]) if len(drawdowns) > 0 else 0

            # Calmar Ratio
            if strategy.max_drawdown > 0:
                annual_return = np.mean(returns) * 252
                strategy.calmar_ratio = annual_return / strategy.max_drawdown

        # Correlation to portfolio
        if len(self._portfolio_returns) > 20 and strategy_name in self._daily_returns:
            strategy_returns = self._daily_returns[strategy_name]
            min_len = min(len(strategy_returns), len(self._portfolio_returns))

            if min_len > 10:
                corr = np.corrcoef(
                    strategy_returns[-min_len:],
                    self._portfolio_returns[-min_len:]
                )[0, 1]
                strategy.correlation_to_portfolio = corr if not np.isnan(corr) else 0

        # Update contribution
        self._update_contributions()

        strategy.last_updated = datetime.now()

    def _update_contributions(self):
        """Update each strategy's contribution to portfolio."""
        total_pnl = sum(s.realized_pnl for s in self._strategies.values())

        for strategy in self._strategies.values():
            if total_pnl != 0:
                strategy.contribution_to_portfolio = strategy.realized_pnl / abs(total_pnl)
            else:
                strategy.contribution_to_portfolio = 0

    def _check_quarantine_conditions(self, strategy_name: str):
        """Check if strategy should be quarantined."""
        strategy = self._strategies[strategy_name]
        trades = self._trade_history[strategy_name]

        if strategy.status == StrategyStatus.QUARANTINED:
            # Check if quarantine period is over
            if (strategy.quarantine_until and
                datetime.now() > strategy.quarantine_until):
                strategy.status = StrategyStatus.MONITORING
                strategy.quarantine_reason = None
                strategy.quarantine_until = None
                logger.info(f"Strategy {strategy_name} released from quarantine")
            return

        quarantine_reason = None

        # Check consecutive losses
        if len(trades) >= self.config.strategy_max_consecutive_losses:
            recent_trades = trades[-self.config.strategy_max_consecutive_losses:]
            if all(t['pnl'] < 0 for t in recent_trades):
                quarantine_reason = f"{self.config.strategy_max_consecutive_losses} consecutive losses"

        # Check max drawdown
        if strategy.max_drawdown > self.config.strategy_max_drawdown:
            quarantine_reason = f"Max drawdown exceeded ({strategy.max_drawdown:.2%})"

        # Check Sharpe ratio
        if (strategy.total_trades >= 20 and
            strategy.sharpe_ratio < self.config.strategy_min_sharpe):
            quarantine_reason = f"Poor risk-adjusted returns (Sharpe: {strategy.sharpe_ratio:.2f})"

        # Apply quarantine if needed
        if quarantine_reason:
            strategy.status = StrategyStatus.QUARANTINED
            strategy.quarantine_reason = quarantine_reason
            strategy.quarantine_until = datetime.now() + timedelta(
                hours=self.config.quarantine_duration_hours
            )
            logger.warning(
                f"Strategy {strategy_name} quarantined: {quarantine_reason}. "
                f"Until: {strategy.quarantine_until}"
            )

    def get_strategy_performance(self, strategy_name: str) -> Optional[StrategyPerformance]:
        """Get performance metrics for a strategy."""
        return self._strategies.get(strategy_name)

    def get_all_performances(self) -> Dict[str, StrategyPerformance]:
        """Get performance metrics for all strategies."""
        return self._strategies.copy()

    def get_active_strategies(self) -> List[str]:
        """Get list of strategies that are not quarantined."""
        return [
            name for name, perf in self._strategies.items()
            if perf.status not in [StrategyStatus.QUARANTINED, StrategyStatus.DISABLED]
        ]

    def get_quarantined_strategies(self) -> List[Tuple[str, str, Optional[datetime]]]:
        """Get list of quarantined strategies with reasons."""
        return [
            (name, perf.quarantine_reason, perf.quarantine_until)
            for name, perf in self._strategies.items()
            if perf.status == StrategyStatus.QUARANTINED
        ]

    def force_quarantine(
        self,
        strategy_name: str,
        reason: str,
        duration_hours: int = 24
    ):
        """Manually quarantine a strategy."""
        if strategy_name in self._strategies:
            strategy = self._strategies[strategy_name]
            strategy.status = StrategyStatus.QUARANTINED
            strategy.quarantine_reason = f"Manual: {reason}"
            strategy.quarantine_until = datetime.now() + timedelta(hours=duration_hours)

    def release_from_quarantine(self, strategy_name: str):
        """Manually release a strategy from quarantine."""
        if strategy_name in self._strategies:
            strategy = self._strategies[strategy_name]
            strategy.status = StrategyStatus.MONITORING
            strategy.quarantine_reason = None
            strategy.quarantine_until = None


# =============================================================================
# RISK LIMITS ENFORCER
# =============================================================================

class RiskLimitsEnforcer:
    """
    Enforces risk limits and triggers automatic actions when limits are breached.

    Features:
    - Multiple limit types with configurable thresholds
    - Automatic position reduction on drawdown
    - Real-time monitoring and alerts
    """

    def __init__(
        self,
        config: RiskLimitConfig = None,
        on_violation_callback: Optional[Callable[[RiskLimitViolation], None]] = None
    ):
        """
        Initialize Risk Limits Enforcer.

        Args:
            config: Risk limit configuration
            on_violation_callback: Callback function for violations
        """
        self.config = config or RiskLimitConfig()
        self.on_violation_callback = on_violation_callback

        # Tracking
        self._violations_today: List[RiskLimitViolation] = []
        self._daily_start_value: Optional[float] = None
        self._weekly_start_value: Optional[float] = None
        self._monthly_start_value: Optional[float] = None
        self._high_water_mark: float = 0.0
        self._last_check_date: Optional[datetime] = None

        # Position tracking for automatic reduction
        self._reduction_in_progress: bool = False
        self._target_reduction_pct: float = 0.0

    def set_starting_values(
        self,
        daily: float,
        weekly: Optional[float] = None,
        monthly: Optional[float] = None
    ):
        """Set starting portfolio values for loss calculations."""
        self._daily_start_value = daily
        self._weekly_start_value = weekly or daily
        self._monthly_start_value = monthly or daily
        self._high_water_mark = max(self._high_water_mark, daily)

    def check_all_limits(
        self,
        portfolio_value: float,
        positions: Dict[str, Dict[str, Any]],
        sector_exposure: Dict[str, float],
        var_95: float,
        var_99: float,
        correlation_exposure: float = 0.0
    ) -> Tuple[bool, List[RiskLimitViolation]]:
        """
        Check all risk limits.

        Args:
            portfolio_value: Current portfolio value
            positions: Dictionary of positions with values
            sector_exposure: Dictionary of sector exposures
            var_95: 95% VaR value
            var_99: 99% VaR value
            correlation_exposure: Exposure to correlated assets

        Returns:
            Tuple of (all_ok, list of violations)
        """
        violations = []

        # Reset daily tracking if new day
        self._check_new_period()

        # Update high water mark
        self._high_water_mark = max(self._high_water_mark, portfolio_value)

        # Check daily loss limit
        if self._daily_start_value:
            daily_loss_pct = (self._daily_start_value - portfolio_value) / self._daily_start_value
            daily_loss_abs = self._daily_start_value - portfolio_value

            if daily_loss_pct > self.config.max_daily_loss_pct:
                violations.append(self._create_violation(
                    RiskLimitType.DAILY_LOSS,
                    daily_loss_pct,
                    self.config.max_daily_loss_pct
                ))

            if daily_loss_abs > self.config.max_daily_loss_abs:
                violations.append(self._create_violation(
                    RiskLimitType.DAILY_LOSS,
                    daily_loss_abs,
                    self.config.max_daily_loss_abs
                ))

        # Check weekly loss limit
        if self._weekly_start_value:
            weekly_loss_pct = (self._weekly_start_value - portfolio_value) / self._weekly_start_value
            if weekly_loss_pct > self.config.max_weekly_loss_pct:
                violations.append(self._create_violation(
                    RiskLimitType.WEEKLY_LOSS,
                    weekly_loss_pct,
                    self.config.max_weekly_loss_pct
                ))

        # Check monthly loss limit
        if self._monthly_start_value:
            monthly_loss_pct = (self._monthly_start_value - portfolio_value) / self._monthly_start_value
            if monthly_loss_pct > self.config.max_monthly_loss_pct:
                violations.append(self._create_violation(
                    RiskLimitType.MONTHLY_LOSS,
                    monthly_loss_pct,
                    self.config.max_monthly_loss_pct
                ))

        # Check drawdown
        drawdown = (self._high_water_mark - portfolio_value) / self._high_water_mark
        if drawdown > self.config.max_drawdown_pct:
            violation = self._create_violation(
                RiskLimitType.MAX_DRAWDOWN,
                drawdown,
                self.config.max_drawdown_pct
            )
            # Trigger automatic position reduction
            reduction_pct = self._calculate_reduction_percentage(drawdown)
            violation.auto_action_taken = f"Position reduction: {reduction_pct:.1%}"
            self._reduction_in_progress = True
            self._target_reduction_pct = reduction_pct
            violations.append(violation)
        elif drawdown > self.config.drawdown_reduction_threshold:
            # Start gradual reduction
            reduction_pct = self._calculate_reduction_percentage(drawdown)
            self._reduction_in_progress = True
            self._target_reduction_pct = reduction_pct

        # Check position concentration
        for symbol, position in positions.items():
            position_value = position.get('value', 0)
            concentration = position_value / portfolio_value if portfolio_value > 0 else 0

            if concentration > self.config.max_position_concentration:
                violations.append(self._create_violation(
                    RiskLimitType.POSITION_CONCENTRATION,
                    concentration,
                    self.config.max_position_concentration
                ))

        # Check sector exposure
        for sector, exposure in sector_exposure.items():
            if exposure > self.config.max_sector_exposure:
                violations.append(self._create_violation(
                    RiskLimitType.SECTOR_EXPOSURE,
                    exposure,
                    self.config.max_sector_exposure
                ))

        # Check VaR limits
        if var_95 > self.config.max_var_95:
            violations.append(self._create_violation(
                RiskLimitType.VAR_LIMIT,
                var_95,
                self.config.max_var_95
            ))

        if var_99 > self.config.max_var_99:
            violations.append(self._create_violation(
                RiskLimitType.VAR_LIMIT,
                var_99,
                self.config.max_var_99
            ))

        # Check correlation exposure
        if correlation_exposure > self.config.max_correlation_exposure:
            violations.append(self._create_violation(
                RiskLimitType.CORRELATION,
                correlation_exposure,
                self.config.max_correlation_exposure
            ))

        # Check leverage
        total_exposure = sum(pos.get('value', 0) for pos in positions.values())
        leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0

        if leverage > self.config.max_leverage:
            violations.append(self._create_violation(
                RiskLimitType.LEVERAGE,
                leverage,
                self.config.max_leverage
            ))

        # Process violations
        for violation in violations:
            self._violations_today.append(violation)
            if self.on_violation_callback:
                self.on_violation_callback(violation)

            logger.warning(
                f"Risk limit violation: {violation.limit_type.value} - "
                f"Current: {violation.current_value:.4f}, Limit: {violation.limit_value:.4f}"
            )

        return len(violations) == 0, violations

    def _create_violation(
        self,
        limit_type: RiskLimitType,
        current_value: float,
        limit_value: float
    ) -> RiskLimitViolation:
        """Create a risk limit violation object."""
        severity = (current_value - limit_value) / limit_value if limit_value > 0 else float('inf')

        return RiskLimitViolation(
            limit_type=limit_type,
            current_value=current_value,
            limit_value=limit_value,
            violation_severity=severity
        )

    def _check_new_period(self):
        """Check if we've entered a new period and reset tracking."""
        now = datetime.now()

        if self._last_check_date is None:
            self._last_check_date = now
            return

        # New day
        if now.date() > self._last_check_date.date():
            self._violations_today = []
            self._daily_start_value = None  # Will be set by caller

        # New week (Monday)
        if now.weekday() == 0 and self._last_check_date.weekday() != 0:
            self._weekly_start_value = None

        # New month
        if now.month != self._last_check_date.month:
            self._monthly_start_value = None

        self._last_check_date = now

    def _calculate_reduction_percentage(self, drawdown: float) -> float:
        """Calculate position reduction percentage based on drawdown."""
        if drawdown <= self.config.drawdown_reduction_threshold:
            return 0.0

        # Linear scaling from threshold to max drawdown
        threshold = self.config.drawdown_reduction_threshold
        max_dd = self.config.max_drawdown_pct

        if max_dd <= threshold:
            return 0.5  # Default 50% reduction

        # Scale from 0 at threshold to 50% at max drawdown
        reduction = 0.5 * (drawdown - threshold) / (max_dd - threshold)
        return min(0.5, max(0, reduction))

    def get_position_reduction_target(self) -> float:
        """Get target position reduction percentage."""
        if self._reduction_in_progress:
            return self._target_reduction_pct
        return 0.0

    def clear_reduction_target(self):
        """Clear position reduction target after execution."""
        self._reduction_in_progress = False
        self._target_reduction_pct = 0.0

    def get_violations_today(self) -> List[RiskLimitViolation]:
        """Get all violations recorded today."""
        return self._violations_today.copy()

    def get_current_drawdown(self, portfolio_value: float) -> float:
        """Calculate current drawdown from high water mark."""
        if self._high_water_mark <= 0:
            return 0.0
        return (self._high_water_mark - portfolio_value) / self._high_water_mark


# =============================================================================
# RISK INTELLIGENCE ENGINE
# =============================================================================

class RiskIntelligenceEngine:
    """
    Main Risk Intelligence Engine.

    Integrates all risk analysis components:
    - VaR Calculator
    - Correlation Monitor
    - Regime Detector
    - Strategy Performance Tracker
    - Risk Limits Enforcer

    Provides comprehensive risk intelligence reports and recommendations.
    """

    def __init__(
        self,
        config: RiskLimitConfig = None,
        var_confidence_levels: List[float] = None,
        var_rolling_window: int = 252,
        correlation_lookback: int = 60,
        hmm_states: int = 3
    ):
        """
        Initialize Risk Intelligence Engine.

        Args:
            config: Risk limit configuration
            var_confidence_levels: Confidence levels for VaR
            var_rolling_window: Rolling window for VaR calculation
            correlation_lookback: Lookback for correlation analysis
            hmm_states: Number of HMM states for regime detection
        """
        self.config = config or RiskLimitConfig()

        # Initialize components
        self.var_calculator = VaRCalculator(
            confidence_levels=var_confidence_levels or [0.95, 0.99],
            rolling_window=var_rolling_window
        )

        self.correlation_monitor = CorrelationMonitor(
            lookback_window=correlation_lookback
        )

        self.regime_detector = RegimeDetector(
            hmm_states=hmm_states
        )

        self.strategy_tracker = StrategyPerformanceTracker(
            config=self.config
        )

        self.risk_enforcer = RiskLimitsEnforcer(
            config=self.config
        )

        # State
        self._last_report: Optional[RiskIntelligenceReport] = None
        self._report_history: List[RiskIntelligenceReport] = []

        logger.info("Risk Intelligence Engine initialized")

    async def generate_report(
        self,
        portfolio_value: float,
        positions: Dict[str, Dict[str, Any]],
        returns_data: pd.DataFrame,
        price_data: pd.DataFrame,
        sector_mapping: Optional[Dict[str, str]] = None,
        volume_data: Optional[pd.Series] = None
    ) -> RiskIntelligenceReport:
        """
        Generate comprehensive risk intelligence report.

        Args:
            portfolio_value: Current portfolio value
            positions: Dictionary of positions
            returns_data: DataFrame with returns for each asset
            price_data: DataFrame with price data
            sector_mapping: Optional mapping of symbols to sectors
            volume_data: Optional volume data

        Returns:
            RiskIntelligenceReport
        """
        timestamp = datetime.now()

        # Set starting values for risk enforcer if not set
        if self.risk_enforcer._daily_start_value is None:
            self.risk_enforcer.set_starting_values(portfolio_value)

        # Get symbols
        symbols = list(returns_data.columns) if isinstance(returns_data, pd.DataFrame) else []

        # 1. Calculate VaR
        portfolio_returns = self._calculate_portfolio_returns(returns_data, positions, portfolio_value)
        var_results = self.var_calculator.calculate_all(
            portfolio_returns, portfolio_value, time_horizon_days=1
        )

        # 2. Analyze correlations
        if len(symbols) > 1:
            correlation_analysis = self.correlation_monitor.analyze(
                returns_data, symbols
            )
        else:
            correlation_analysis = CorrelationAnalysis(
                correlation_matrix=np.array([[1.0]]),
                symbols=symbols,
                avg_correlation=0.0,
                max_correlation=0.0,
                min_correlation=0.0,
                diversification_score=1.0,
                correlation_clusters=[],
                eigenvalues=np.array([1.0]),
                effective_assets=1.0
            )

        # 3. Detect regime
        regime_state = self.regime_detector.detect_regime(price_data, volume_data)

        # 4. Get strategy performances
        strategy_performances = self.strategy_tracker.get_all_performances()

        # 5. Calculate sector exposure
        sector_exposure = self._calculate_sector_exposure(
            positions, portfolio_value, sector_mapping
        )

        # 6. Check risk limits
        var_95 = var_results.get('historical_95', VaRResult(
            method=VaRMethod.HISTORICAL, confidence_level=0.95,
            var_value=0, var_dollar=0, cvar_value=0, cvar_dollar=0,
            time_horizon_days=1
        )).var_value

        var_99 = var_results.get('historical_99', VaRResult(
            method=VaRMethod.HISTORICAL, confidence_level=0.99,
            var_value=0, var_dollar=0, cvar_value=0, cvar_dollar=0,
            time_horizon_days=1
        )).var_value

        # Calculate correlation exposure
        correlation_exposure = self._calculate_correlation_exposure(
            correlation_analysis, positions, portfolio_value
        )

        all_ok, violations = self.risk_enforcer.check_all_limits(
            portfolio_value=portfolio_value,
            positions=positions,
            sector_exposure=sector_exposure,
            var_95=var_95,
            var_99=var_99,
            correlation_exposure=correlation_exposure
        )

        # 7. Generate recommendations
        recommendations = self._generate_recommendations(
            var_results=var_results,
            correlation_analysis=correlation_analysis,
            regime_state=regime_state,
            strategy_performances=strategy_performances,
            violations=violations,
            portfolio_value=portfolio_value
        )

        # Create report
        report = RiskIntelligenceReport(
            timestamp=timestamp,
            portfolio_value=portfolio_value,
            var_results=var_results,
            correlation_analysis=correlation_analysis,
            regime_state=regime_state,
            strategy_performances=strategy_performances,
            risk_limit_violations=violations,
            all_limits_ok=all_ok,
            recommended_actions=recommendations
        )

        # Store report
        self._last_report = report
        self._report_history.append(report)

        # Keep last 100 reports
        if len(self._report_history) > 100:
            self._report_history = self._report_history[-100:]

        return report

    def _calculate_portfolio_returns(
        self,
        returns_data: pd.DataFrame,
        positions: Dict[str, Dict[str, Any]],
        portfolio_value: float
    ) -> pd.Series:
        """Calculate weighted portfolio returns."""
        if returns_data.empty or not positions:
            return pd.Series([0.0])

        # Calculate weights
        weights = {}
        for symbol, position in positions.items():
            if symbol in returns_data.columns:
                value = position.get('value', 0)
                weights[symbol] = value / portfolio_value if portfolio_value > 0 else 0

        # Calculate weighted returns
        portfolio_returns = pd.Series(0.0, index=returns_data.index)
        for symbol, weight in weights.items():
            if symbol in returns_data.columns:
                portfolio_returns += returns_data[symbol] * weight

        return portfolio_returns

    def _calculate_sector_exposure(
        self,
        positions: Dict[str, Dict[str, Any]],
        portfolio_value: float,
        sector_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Calculate sector exposure."""
        if not sector_mapping:
            return {}

        sector_values: Dict[str, float] = {}

        for symbol, position in positions.items():
            sector = sector_mapping.get(symbol, 'Unknown')
            value = position.get('value', 0)
            sector_values[sector] = sector_values.get(sector, 0) + value

        # Convert to percentages
        return {
            sector: value / portfolio_value
            for sector, value in sector_values.items()
        } if portfolio_value > 0 else {}

    def _calculate_correlation_exposure(
        self,
        correlation_analysis: CorrelationAnalysis,
        positions: Dict[str, Dict[str, Any]],
        portfolio_value: float
    ) -> float:
        """Calculate exposure to highly correlated assets."""
        if not correlation_analysis.correlation_clusters:
            return 0.0

        # Sum exposure in each correlation cluster
        max_cluster_exposure = 0.0

        for cluster in correlation_analysis.correlation_clusters:
            cluster_value = sum(
                positions.get(symbol, {}).get('value', 0)
                for symbol in cluster
            )
            cluster_exposure = cluster_value / portfolio_value if portfolio_value > 0 else 0
            max_cluster_exposure = max(max_cluster_exposure, cluster_exposure)

        return max_cluster_exposure

    def _generate_recommendations(
        self,
        var_results: Dict[str, VaRResult],
        correlation_analysis: CorrelationAnalysis,
        regime_state: RegimeState,
        strategy_performances: Dict[str, StrategyPerformance],
        violations: List[RiskLimitViolation],
        portfolio_value: float
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # High VaR recommendations
        for key, var_result in var_results.items():
            if var_result.var_value > self.config.max_var_95:
                recommendations.append(
                    f"REDUCE RISK: VaR ({key}) is {var_result.var_value:.2%}, "
                    f"exceeding limit of {self.config.max_var_95:.2%}. "
                    f"Consider reducing position sizes or hedging."
                )

        # Correlation recommendations
        if correlation_analysis.avg_correlation > 0.6:
            recommendations.append(
                f"DIVERSIFY: Portfolio correlation is high ({correlation_analysis.avg_correlation:.2f}). "
                f"Consider adding uncorrelated assets. "
                f"Effective assets: {correlation_analysis.effective_assets:.1f}"
            )

        if correlation_analysis.regime_change_detected:
            recommendations.append(
                f"ALERT: Correlation regime change detected. "
                f"{correlation_analysis.regime_change_details}"
            )

        # Regime-based recommendations
        if regime_state.volatility_regime == VolatilityRegime.EXTREME:
            recommendations.append(
                f"HIGH VOLATILITY ALERT: Volatility is at {regime_state.volatility_percentile:.0f}th percentile. "
                f"Consider reducing position sizes and tightening stops."
            )

        if regime_state.liquidity_regime in [LiquidityRegime.ILLIQUID, LiquidityRegime.HIGHLY_ILLIQUID]:
            recommendations.append(
                f"LOW LIQUIDITY ALERT: Market liquidity is low (score: {regime_state.liquidity_score:.2f}). "
                f"Use limit orders and be cautious with large positions."
            )

        if regime_state.trend_regime == TrendRegime.CHOPPY:
            recommendations.append(
                "CHOPPY MARKET: Trend-following strategies may underperform. "
                "Consider mean-reversion or reducing trading frequency."
            )

        # Strategy recommendations
        quarantined = [
            (name, perf.quarantine_reason)
            for name, perf in strategy_performances.items()
            if perf.status == StrategyStatus.QUARANTINED
        ]

        for name, reason in quarantined:
            recommendations.append(
                f"STRATEGY QUARANTINED: {name} - {reason}. "
                f"Review and adjust parameters before re-enabling."
            )

        # Violation-based recommendations
        for violation in violations:
            if violation.limit_type == RiskLimitType.MAX_DRAWDOWN:
                recommendations.append(
                    f"DRAWDOWN LIMIT BREACHED: Current drawdown is {violation.current_value:.2%}. "
                    f"Automatic position reduction initiated. "
                    f"Review risk management and consider reducing exposure."
                )
            elif violation.limit_type == RiskLimitType.POSITION_CONCENTRATION:
                recommendations.append(
                    f"CONCENTRATION RISK: Position too large ({violation.current_value:.2%}). "
                    f"Consider trimming to meet {violation.limit_value:.2%} limit."
                )

        # General recommendations based on position reduction
        if self.risk_enforcer.get_position_reduction_target() > 0:
            target = self.risk_enforcer.get_position_reduction_target()
            recommendations.append(
                f"ACTION REQUIRED: Reduce overall position size by {target:.1%} "
                f"to manage drawdown risk."
            )

        return recommendations

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def register_strategy(self, strategy_name: str):
        """Register a strategy for performance tracking."""
        self.strategy_tracker.register_strategy(strategy_name)

    def record_trade(
        self,
        strategy_name: str,
        pnl: float,
        trade_details: Optional[Dict[str, Any]] = None
    ):
        """Record a trade for a strategy."""
        self.strategy_tracker.record_trade(strategy_name, pnl, trade_details)

    def get_strategy_status(self, strategy_name: str) -> Optional[StrategyStatus]:
        """Get the current status of a strategy."""
        perf = self.strategy_tracker.get_strategy_performance(strategy_name)
        return perf.status if perf else None

    def is_strategy_active(self, strategy_name: str) -> bool:
        """Check if a strategy is active (not quarantined or disabled)."""
        status = self.get_strategy_status(strategy_name)
        return status not in [StrategyStatus.QUARANTINED, StrategyStatus.DISABLED]

    def get_last_report(self) -> Optional[RiskIntelligenceReport]:
        """Get the most recent risk intelligence report."""
        return self._last_report

    def get_current_var(self) -> Optional[float]:
        """Get current 95% Historical VaR."""
        if self._last_report:
            var_result = self._last_report.var_results.get('historical_95')
            return var_result.var_value if var_result else None
        return None

    def get_current_regime(self) -> Optional[RegimeState]:
        """Get current market regime state."""
        if self._last_report:
            return self._last_report.regime_state
        return None

    def get_position_reduction_target(self) -> float:
        """Get target position reduction percentage."""
        return self.risk_enforcer.get_position_reduction_target()

    def clear_position_reduction(self):
        """Clear position reduction target after execution."""
        self.risk_enforcer.clear_reduction_target()

    def export_report_json(self, filepath: str):
        """Export last report to JSON file."""
        if self._last_report:
            with open(filepath, 'w') as f:
                json.dump(self._last_report.to_dict(), f, indent=2, default=str)
            logger.info(f"Report exported to {filepath}")

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get a concise risk summary."""
        if not self._last_report:
            return {'status': 'No report available'}

        report = self._last_report

        var_95 = report.var_results.get('historical_95')

        return {
            'timestamp': report.timestamp.isoformat(),
            'portfolio_value': report.portfolio_value,
            'var_95': var_95.var_value if var_95 else None,
            'var_95_dollar': var_95.var_dollar if var_95 else None,
            'volatility_regime': report.regime_state.volatility_regime.value,
            'trend_regime': report.regime_state.trend_regime.value,
            'avg_correlation': report.correlation_analysis.avg_correlation,
            'diversification_score': report.correlation_analysis.diversification_score,
            'all_limits_ok': report.all_limits_ok,
            'violation_count': len(report.risk_limit_violations),
            'active_strategies': len([
                s for s in report.strategy_performances.values()
                if s.status == StrategyStatus.ACTIVE
            ]),
            'quarantined_strategies': len([
                s for s in report.strategy_performances.values()
                if s.status == StrategyStatus.QUARANTINED
            ]),
            'recommendation_count': len(report.recommended_actions)
        }


# =============================================================================
# ASYNC WRAPPER FOR REAL-TIME MONITORING
# =============================================================================

class RiskMonitor:
    """
    Async wrapper for continuous risk monitoring.

    Provides real-time risk updates and alerts.
    """

    def __init__(
        self,
        engine: RiskIntelligenceEngine,
        update_interval_seconds: int = 60,
        alert_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize Risk Monitor.

        Args:
            engine: RiskIntelligenceEngine instance
            update_interval_seconds: Seconds between updates
            alert_callback: Callback for risk alerts
        """
        self.engine = engine
        self.update_interval = update_interval_seconds
        self.alert_callback = alert_callback

        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(
        self,
        portfolio_value_getter: Callable[[], float],
        positions_getter: Callable[[], Dict[str, Dict[str, Any]]],
        returns_getter: Callable[[], pd.DataFrame],
        price_getter: Callable[[], pd.DataFrame]
    ):
        """
        Start continuous risk monitoring.

        Args:
            portfolio_value_getter: Function to get current portfolio value
            positions_getter: Function to get current positions
            returns_getter: Function to get returns data
            price_getter: Function to get price data
        """
        self._running = True

        while self._running:
            try:
                # Get current data
                portfolio_value = portfolio_value_getter()
                positions = positions_getter()
                returns_data = returns_getter()
                price_data = price_getter()

                # Generate report
                report = await self.engine.generate_report(
                    portfolio_value=portfolio_value,
                    positions=positions,
                    returns_data=returns_data,
                    price_data=price_data
                )

                # Check for alerts
                if not report.all_limits_ok and self.alert_callback:
                    for violation in report.risk_limit_violations:
                        self.alert_callback(
                            f"Risk Limit Violation: {violation.limit_type.value} - "
                            f"{violation.current_value:.4f} exceeds {violation.limit_value:.4f}"
                        )

                # Check for critical recommendations
                for rec in report.recommended_actions:
                    if rec.startswith("ALERT") or rec.startswith("ACTION REQUIRED"):
                        if self.alert_callback:
                            self.alert_callback(rec)

            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")

            await asyncio.sleep(self.update_interval)

    def stop(self):
        """Stop continuous monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_risk_intelligence_engine(
    max_daily_loss_pct: float = 0.02,
    max_drawdown_pct: float = 0.15,
    max_var_95: float = 0.03,
    max_position_concentration: float = 0.15,
    strategy_quarantine_hours: int = 24
) -> RiskIntelligenceEngine:
    """
    Factory function to create a configured Risk Intelligence Engine.

    Args:
        max_daily_loss_pct: Maximum daily loss percentage
        max_drawdown_pct: Maximum drawdown percentage
        max_var_95: Maximum 95% VaR
        max_position_concentration: Maximum single position concentration
        strategy_quarantine_hours: Hours to quarantine underperforming strategies

    Returns:
        Configured RiskIntelligenceEngine instance
    """
    config = RiskLimitConfig(
        max_daily_loss_pct=max_daily_loss_pct,
        max_drawdown_pct=max_drawdown_pct,
        max_var_95=max_var_95,
        max_position_concentration=max_position_concentration,
        quarantine_duration_hours=strategy_quarantine_hours
    )

    return RiskIntelligenceEngine(config=config)


# =============================================================================
# MAIN / TESTING
# =============================================================================

if __name__ == "__main__":
    # Example usage
    import asyncio

    async def main():
        # Create engine
        engine = create_risk_intelligence_engine()

        # Register strategies
        engine.register_strategy("MA_Cross")
        engine.register_strategy("RSI_Momentum")
        engine.register_strategy("Bollinger_Bands")

        # Simulate some data
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')

        # Generate synthetic returns
        returns_data = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 252),
            'GOOGL': np.random.normal(0.0008, 0.022, 252),
            'MSFT': np.random.normal(0.0012, 0.018, 252),
            'AMZN': np.random.normal(0.0009, 0.025, 252)
        }, index=dates)

        # Generate price data
        price_data = pd.DataFrame({
            'close': 100 * np.exp(np.cumsum(returns_data['AAPL'])),
            'high': 100 * np.exp(np.cumsum(returns_data['AAPL'])) * 1.01,
            'low': 100 * np.exp(np.cumsum(returns_data['AAPL'])) * 0.99,
            'volume': np.random.randint(1000000, 10000000, 252)
        }, index=dates)

        # Sample positions
        positions = {
            'AAPL': {'value': 25000, 'quantity': 100},
            'GOOGL': {'value': 20000, 'quantity': 50},
            'MSFT': {'value': 30000, 'quantity': 150},
            'AMZN': {'value': 25000, 'quantity': 75}
        }

        portfolio_value = 100000

        # Generate report
        report = await engine.generate_report(
            portfolio_value=portfolio_value,
            positions=positions,
            returns_data=returns_data,
            price_data=price_data,
            sector_mapping={
                'AAPL': 'Technology',
                'GOOGL': 'Technology',
                'MSFT': 'Technology',
                'AMZN': 'Consumer'
            }
        )

        # Print summary
        print("\n" + "="*60)
        print("RISK INTELLIGENCE REPORT")
        print("="*60)

        summary = engine.get_risk_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")

        print("\n" + "-"*60)
        print("RECOMMENDATIONS:")
        print("-"*60)
        for i, rec in enumerate(report.recommended_actions, 1):
            print(f"{i}. {rec}")

        print("\n" + "-"*60)
        print("REGIME STATE:")
        print("-"*60)
        regime = report.regime_state
        print(f"Volatility: {regime.volatility_regime.value} ({regime.volatility_percentile:.1f}th percentile)")
        print(f"Trend: {regime.trend_regime.value} (strength: {regime.trend_strength:.3f})")
        print(f"Liquidity: {regime.liquidity_regime.value} (score: {regime.liquidity_score:.2f})")

        print("\n" + "-"*60)
        print("VAR RESULTS:")
        print("-"*60)
        for key, var_result in report.var_results.items():
            print(f"{key}: {var_result.var_value:.4f} (${var_result.var_dollar:,.2f})")

    # Run
    asyncio.run(main())
