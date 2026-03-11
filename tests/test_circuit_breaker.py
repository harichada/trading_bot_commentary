"""Tests for circuit breaker implementations."""
import time
import pytest

from circuit_breaker import CircuitBreaker, CircuitState
from trading_exceptions import CircuitBreakerException, NetworkException


class TestCircuitBreaker:
    """Test the API/order circuit breaker."""

    def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
        assert cb.state == CircuitState.CLOSED

    def test_stays_closed_under_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10,
                            expected_exception=ValueError)

        @cb
        def failing_func():
            raise ValueError("test")

        for _ in range(2):
            with pytest.raises(ValueError):
                failing_func()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 2

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60,
                            expected_exception=ValueError)

        @cb
        def failing_func():
            raise ValueError("test")

        for _ in range(3):
            with pytest.raises(ValueError):
                failing_func()

        assert cb.state == CircuitState.OPEN

    def test_open_breaker_raises_circuit_breaker_exception(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60,
                            expected_exception=ValueError)

        @cb
        def failing_func():
            raise ValueError("test")

        with pytest.raises(ValueError):
            failing_func()

        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerException):
            failing_func()

    def test_resets_on_success(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10,
                            expected_exception=ValueError)

        call_count = [0]

        @cb
        def maybe_fail():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("fail")
            return "ok"

        with pytest.raises(ValueError):
            maybe_fail()
        with pytest.raises(ValueError):
            maybe_fail()

        assert cb.failure_count == 2
        result = maybe_fail()
        assert result == "ok"
        # Failure count stays at 2 because _on_success only resets in HALF_OPEN
        # But the breaker is still CLOSED since we didn't hit threshold
        assert cb.state == CircuitState.CLOSED

    def test_manual_reset(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=60,
                            expected_exception=ValueError)

        @cb
        def failing_func():
            raise ValueError("test")

        with pytest.raises(ValueError):
            failing_func()
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_half_open_recovery(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0,
                            expected_exception=ValueError)

        call_count = [0]

        @cb
        def maybe_fail():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("fail")
            return "recovered"

        with pytest.raises(ValueError):
            maybe_fail()
        assert cb.state == CircuitState.OPEN

        # With recovery_timeout=0, should transition to HALF_OPEN immediately
        result = maybe_fail()
        assert result == "recovered"
        assert cb.state == CircuitState.CLOSED


class TestTradingLossBreaker:
    """Test the trading loss circuit breaker."""

    def test_allows_trading_by_default(self):
        from core.config import TradingLossBreaker
        breaker = TradingLossBreaker()
        assert breaker.can_trade() is True

    def test_trips_on_max_daily_loss(self):
        from core.config import TradingLossBreaker
        breaker = TradingLossBreaker(max_daily_loss=0.10, emergency_stop=0.15)
        breaker.update_daily_pnl(-11000, account_balance=100000)
        assert breaker.is_tripped is True
        assert breaker.can_trade() is False

    def test_does_not_trip_on_gains(self):
        from core.config import TradingLossBreaker
        breaker = TradingLossBreaker(max_daily_loss=0.10)
        breaker.update_daily_pnl(5000, account_balance=100000)
        assert breaker.is_tripped is False
        assert breaker.can_trade() is True

    def test_does_not_trip_below_threshold(self):
        from core.config import TradingLossBreaker
        breaker = TradingLossBreaker(max_daily_loss=0.10)
        breaker.update_daily_pnl(-5000, account_balance=100000)  # 5% < 10%
        assert breaker.is_tripped is False

    def test_manual_reset(self):
        from core.config import TradingLossBreaker
        breaker = TradingLossBreaker(max_daily_loss=0.10)
        breaker.update_daily_pnl(-15000, account_balance=100000)
        assert breaker.can_trade() is False
        breaker.reset()
        assert breaker.can_trade() is True
