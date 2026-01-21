"""
Tests for Risk Management Module
"""

import pytest
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from risk.position_sizer import PositionSizer, SizingMethod, SizeResult
from risk.limits import RiskLimits, RiskLimitChecker, RiskCheck, RiskViolation
from risk.manager import RiskManager, RiskAssessment


class TestPositionSizer:
    """Tests for Position Sizer"""

    def test_fixed_amount_sizing(self):
        """Test fixed amount position sizing"""
        sizer = PositionSizer(
            method=SizingMethod.FIXED_AMOUNT,
            config={'fixed_amount': 5000}
        )

        result = sizer.calculate(
            price=100.0,
            portfolio_value=100000.0
        )

        assert result.shares == 50  # 5000 / 100
        assert result.dollar_amount == 5000.0
        assert result.method == SizingMethod.FIXED_AMOUNT

    def test_fixed_percentage_sizing(self):
        """Test fixed percentage position sizing"""
        sizer = PositionSizer(
            method=SizingMethod.FIXED_PERCENTAGE,
            config={'fixed_percentage': 0.05}
        )

        result = sizer.calculate(
            price=100.0,
            portfolio_value=100000.0
        )

        assert result.shares == 50  # 5% of 100k / 100
        assert result.dollar_amount == 5000.0
        assert result.method == SizingMethod.FIXED_PERCENTAGE

    def test_kelly_criterion_sizing(self):
        """Test Kelly criterion position sizing"""
        sizer = PositionSizer(
            method=SizingMethod.KELLY_CRITERION,
            config={'kelly_fraction': 0.25}
        )

        result = sizer.calculate(
            price=100.0,
            portfolio_value=100000.0,
            win_rate=0.6,
            avg_win=0.04,
            avg_loss=0.02
        )

        assert result.shares > 0
        assert result.method == SizingMethod.KELLY_CRITERION
        assert 'kelly_pct' in result.details

    def test_volatility_adjusted_sizing(self):
        """Test volatility-adjusted position sizing"""
        sizer = PositionSizer(
            method=SizingMethod.VOLATILITY_ADJUSTED
        )

        # High volatility should reduce position size
        high_vol_result = sizer.calculate(
            price=100.0,
            portfolio_value=100000.0,
            volatility=0.04
        )

        low_vol_result = sizer.calculate(
            price=100.0,
            portfolio_value=100000.0,
            volatility=0.01
        )

        # Lower volatility should result in larger position
        assert low_vol_result.shares >= high_vol_result.shares

    def test_risk_parity_sizing(self):
        """Test risk parity position sizing"""
        sizer = PositionSizer(
            method=SizingMethod.RISK_PARITY,
            config={'max_risk_per_trade': 0.02}
        )

        result = sizer.calculate(
            price=100.0,
            portfolio_value=100000.0,
            stop_loss=95.0  # 5% stop
        )

        # Risk per share is $5, risk budget is $2000 (2% of 100k)
        # So max shares = 2000 / 5 = 400
        assert result.shares <= 400
        assert result.method == SizingMethod.RISK_PARITY

    def test_calculate_with_stop(self):
        """Test position sizing with stop loss"""
        sizer = PositionSizer()

        result = sizer.calculate_with_stop(
            price=100.0,
            stop_loss=95.0,
            portfolio_value=100000.0,
            risk_percentage=0.01  # 1% risk
        )

        # Risk is $1000, risk per share is $5, so 200 shares
        assert result.shares == 200
        assert result.risk_amount == 1000.0

    def test_invalid_inputs(self):
        """Test handling of invalid inputs"""
        sizer = PositionSizer()

        result = sizer.calculate(price=0, portfolio_value=100000.0)
        assert result.shares == 0

        result = sizer.calculate(price=100.0, portfolio_value=0)
        assert result.shares == 0

    def test_share_constraints(self):
        """Test min/max share constraints"""
        sizer = PositionSizer(config={
            'min_shares': 10,
            'max_shares': 100
        })

        # Should hit max
        large_result = sizer.calculate(price=1.0, portfolio_value=100000.0)
        assert large_result.shares == 100

        # Should hit min (with very small position)
        sizer2 = PositionSizer(
            method=SizingMethod.FIXED_AMOUNT,
            config={'fixed_amount': 5, 'min_shares': 10}
        )
        small_result = sizer2.calculate(price=100.0, portfolio_value=1000.0)
        assert small_result.shares >= 10


class TestRiskLimits:
    """Tests for Risk Limits"""

    def test_default_limits(self):
        """Test default limit values"""
        limits = RiskLimits()

        assert limits.max_position_size == 0.10
        assert limits.max_positions == 10
        assert limits.max_daily_loss == 0.03
        assert limits.max_drawdown == 0.15

    def test_custom_limits(self):
        """Test custom limit values"""
        limits = RiskLimits(
            max_position_size=0.05,
            max_positions=5,
            max_daily_loss=0.02
        )

        assert limits.max_position_size == 0.05
        assert limits.max_positions == 5
        assert limits.max_daily_loss == 0.02

    def test_to_dict(self):
        """Test serialization"""
        limits = RiskLimits()
        d = limits.to_dict()

        assert 'max_position_size' in d
        assert 'max_positions' in d
        assert 'max_daily_loss' in d


class TestRiskLimitChecker:
    """Tests for Risk Limit Checker"""

    def test_check_new_position_approved(self, empty_portfolio):
        """Test position that passes all checks"""
        checker = RiskLimitChecker()

        result = checker.check_new_position(
            symbol='AAPL',
            quantity=50,
            price=100.0,
            portfolio_value=100000.0,
            buying_power=50000.0,
            current_positions={}
        )

        assert result.passed is True
        assert len(result.violations) == 0

    def test_check_position_size_violation(self):
        """Test position size limit violation"""
        limits = RiskLimits(max_position_size=0.05)
        checker = RiskLimitChecker(limits)

        result = checker.check_new_position(
            symbol='AAPL',
            quantity=100,
            price=100.0,  # $10,000 = 10% of portfolio
            portfolio_value=100000.0,
            buying_power=50000.0,
            current_positions={}
        )

        assert result.passed is False
        assert RiskViolation.MAX_POSITION_SIZE in result.violations

    def test_check_max_positions_violation(self):
        """Test max positions limit violation"""
        limits = RiskLimits(max_positions=2)
        checker = RiskLimitChecker(limits)

        current_positions = {
            'AAPL': {'market_value': 5000},
            'GOOGL': {'market_value': 5000}
        }

        result = checker.check_new_position(
            symbol='NVDA',
            quantity=10,
            price=100.0,
            portfolio_value=100000.0,
            buying_power=50000.0,
            current_positions=current_positions
        )

        assert result.passed is False
        assert RiskViolation.MAX_POSITIONS in result.violations

    def test_check_buying_power_violation(self):
        """Test buying power violation"""
        checker = RiskLimitChecker()

        result = checker.check_new_position(
            symbol='AAPL',
            quantity=100,
            price=100.0,  # $10,000
            portfolio_value=100000.0,
            buying_power=5000.0,  # Only $5,000 available
            current_positions={}
        )

        assert result.passed is False
        assert RiskViolation.INSUFFICIENT_BUYING_POWER in result.violations

    def test_check_daily_loss_limit(self):
        """Test daily loss limit check"""
        limits = RiskLimits(max_daily_loss=0.02)
        checker = RiskLimitChecker(limits)

        result = checker.check_daily_loss(
            daily_pnl=-3000.0,  # 3% loss
            portfolio_value=100000.0
        )

        assert result.passed is False
        assert RiskViolation.MAX_DAILY_LOSS in result.violations

    def test_check_drawdown_limit(self):
        """Test drawdown limit check"""
        limits = RiskLimits(max_drawdown=0.10)
        checker = RiskLimitChecker(limits)

        # Set peak value
        checker.check_drawdown(100000.0)  # Peak

        # Check with lower value (15% drawdown)
        result = checker.check_drawdown(85000.0)

        assert result.passed is False
        assert RiskViolation.MAX_DRAWDOWN in result.violations

    def test_circuit_breaker(self):
        """Test circuit breaker triggering"""
        limits = RiskLimits(max_consecutive_losses=3)
        checker = RiskLimitChecker(limits)

        # Record 3 consecutive losses
        checker.record_trade_result(-100)
        checker.record_trade_result(-100)
        checker.record_trade_result(-100)

        # Circuit breaker should be active
        result = checker.check_new_position(
            symbol='AAPL',
            quantity=10,
            price=100.0,
            portfolio_value=100000.0,
            buying_power=50000.0,
            current_positions={}
        )

        assert result.passed is False
        assert RiskViolation.CIRCUIT_BREAKER in result.violations

    def test_circuit_breaker_reset_on_win(self):
        """Test circuit breaker resets on winning trade"""
        limits = RiskLimits(max_consecutive_losses=3)
        checker = RiskLimitChecker(limits)

        # Record 2 losses, then a win
        checker.record_trade_result(-100)
        checker.record_trade_result(-100)
        checker.record_trade_result(100)  # Win resets counter

        # Should not trigger circuit breaker
        assert checker._consecutive_losses == 0

    def test_warnings(self):
        """Test warning generation"""
        limits = RiskLimits(max_daily_loss=0.05)
        checker = RiskLimitChecker(limits)

        # 4.5% loss (90% of 5% limit)
        result = checker.check_daily_loss(
            daily_pnl=-4500.0,
            portfolio_value=100000.0
        )

        assert result.passed is True  # Not violated yet
        assert len(result.warnings) > 0  # But should have warning


class TestRiskManager:
    """Tests for Risk Manager"""

    @pytest.mark.asyncio
    async def test_check_signal_approved(self, sample_signal):
        """Test signal that passes checks"""
        manager = RiskManager()

        approved, reason = await manager.check_signal(sample_signal)

        assert approved is True

    @pytest.mark.asyncio
    async def test_check_signal_low_confidence(self, low_confidence_signal):
        """Test low confidence signal rejection"""
        manager = RiskManager()

        approved, reason = await manager.check_signal(low_confidence_signal)

        assert approved is False
        assert 'confidence' in reason.lower()

    @pytest.mark.asyncio
    async def test_calculate_position_size(self):
        """Test position size calculation"""
        manager = RiskManager(sizing_method=SizingMethod.FIXED_PERCENTAGE)

        size = await manager.calculate_position_size(
            symbol='AAPL',
            buying_power=100000.0,
            price=150.0,
            volatility=0.02
        )

        assert size > 0
        assert size <= 10000  # Reasonable max

    @pytest.mark.asyncio
    async def test_assess_trade_approved(self, empty_portfolio):
        """Test trade assessment approval"""
        manager = RiskManager()

        assessment = await manager.assess_trade(
            symbol='AAPL',
            action='buy',
            price=150.0,
            confidence=0.75,
            portfolio_value=100000.0,
            buying_power=50000.0,
            current_positions={}
        )

        assert assessment.approved is True
        assert assessment.position_size > 0
        assert assessment.stop_loss is not None
        assert assessment.take_profit is not None

    @pytest.mark.asyncio
    async def test_assess_trade_rejected_low_confidence(self, empty_portfolio):
        """Test trade rejection for low confidence"""
        manager = RiskManager()

        assessment = await manager.assess_trade(
            symbol='AAPL',
            action='buy',
            price=150.0,
            confidence=0.3,
            portfolio_value=100000.0,
            buying_power=50000.0,
            current_positions={}
        )

        assert assessment.approved is False
        assert 'confidence' in assessment.reason.lower()

    @pytest.mark.asyncio
    async def test_assess_trade_rejected_risk_violation(self, sample_portfolio):
        """Test trade rejection for risk violation"""
        limits = RiskLimits(max_positions=2)
        manager = RiskManager(limits=limits)

        assessment = await manager.assess_trade(
            symbol='NVDA',
            action='buy',
            price=500.0,
            confidence=0.8,
            portfolio_value=100000.0,
            buying_power=50000.0,
            current_positions=sample_portfolio['positions']
        )

        assert assessment.approved is False

    def test_record_trade(self):
        """Test trade recording"""
        manager = RiskManager()

        manager.record_trade(
            symbol='AAPL',
            action='buy',
            entry_price=150.0,
            exit_price=160.0,
            quantity=100,
            pnl=1000.0
        )

        assert len(manager._trade_history) == 1
        assert manager._trade_history[0]['pnl'] == 1000.0

    def test_performance_tracking(self):
        """Test performance metric updates"""
        manager = RiskManager()

        # Record some trades
        manager.record_trade('AAPL', 'buy', 150, 160, 100, 1000)
        manager.record_trade('GOOGL', 'buy', 140, 135, 50, -250)
        manager.record_trade('NVDA', 'buy', 500, 520, 20, 400)

        assert manager._win_rate > 0.5  # 2 wins, 1 loss

    @pytest.mark.asyncio
    async def test_check_portfolio_risk(self, sample_portfolio):
        """Test portfolio risk check"""
        manager = RiskManager()

        result = await manager.check_portfolio_risk(
            portfolio_value=100000.0,
            daily_pnl=sample_portfolio['daily_pnl'],
            positions=sample_portfolio['positions']
        )

        assert result.passed is True
        assert 'total_exposure' in result.details

    def test_get_risk_status(self):
        """Test risk status retrieval"""
        manager = RiskManager()

        status = manager.get_risk_status()

        assert 'checker_status' in status
        assert 'performance' in status
        assert 'limits' in status
        assert 'sizing_method' in status
