#!/usr/bin/env python3
"""
Test Suite for Institutional Trading Core
=========================================
Comprehensive tests for all institutional-grade components.

Run with: python -m pytest test_institutional_core.py -v
"""

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import numpy as np

# Import components to test
from institutional_core import (
    # State Machine
    TradingState,
    TradingStateMachine,
    StateTransition,

    # State Persistence
    AtomicStateManager,

    # Order Management
    IdempotentOrderManager,
    OrderAttempt,

    # Position Reconciliation
    PositionReconciler,
    PositionDiscrepancy,
    ReconciliationReport,

    # Health Monitoring
    HealthMonitor,
    HealthCheck,
    create_memory_health_check,
    create_disk_health_check,

    # Circuit Breakers
    EnhancedCircuitBreaker,
    CircuitState,
    CircuitBreakerOpenError,

    # Validators
    PositionSizeValidator,
    FeatureValidator,

    # Core
    InstitutionalTradingCore,
)


# ============================================================================
# STATE MACHINE TESTS
# ============================================================================

class TestTradingStateMachine:
    """Tests for the trading state machine"""

    def test_initial_state(self):
        """Test that state machine starts in INITIALIZING state"""
        sm = TradingStateMachine()
        assert sm.state == TradingState.INITIALIZING

    def test_custom_initial_state(self):
        """Test custom initial state"""
        sm = TradingStateMachine(initial_state=TradingState.CONNECTED)
        assert sm.state == TradingState.CONNECTED

    def test_valid_transition(self):
        """Test valid state transition"""
        sm = TradingStateMachine()

        # INITIALIZING -> CONNECTED is valid
        result = sm.transition_to(TradingState.CONNECTED, trigger="test")
        assert result is True
        assert sm.state == TradingState.CONNECTED

    def test_invalid_transition(self):
        """Test invalid state transition is blocked"""
        sm = TradingStateMachine()

        # INITIALIZING -> TRADING is not directly valid
        result = sm.transition_to(TradingState.TRADING, trigger="test")
        assert result is False
        assert sm.state == TradingState.INITIALIZING

    def test_transition_history(self):
        """Test transition history is recorded"""
        sm = TradingStateMachine()

        sm.transition_to(TradingState.CONNECTED, trigger="init_complete")
        sm.transition_to(TradingState.TRADING, trigger="market_open")

        history = sm.get_transition_history()
        assert len(history) >= 2

    def test_operation_allowed(self):
        """Test operation permission checking"""
        sm = TradingStateMachine(initial_state=TradingState.TRADING)

        assert sm.is_operation_allowed('place_order') is True
        assert sm.is_operation_allowed('market_sell_all') is False

    def test_operation_not_allowed_in_shutdown(self):
        """Test no operations allowed in shutdown state"""
        sm = TradingStateMachine(initial_state=TradingState.SHUTDOWN)

        assert sm.is_operation_allowed('place_order') is False
        assert sm.is_operation_allowed('health_check') is False

    def test_state_duration(self):
        """Test state duration tracking"""
        sm = TradingStateMachine()
        time.sleep(0.1)

        duration = sm.get_state_duration()
        assert duration.total_seconds() >= 0.1

    def test_serialization(self):
        """Test state machine serialization and deserialization"""
        sm = TradingStateMachine()
        sm.transition_to(TradingState.CONNECTED, trigger="test")

        # Serialize
        data = sm.serialize()
        assert 'current_state' in data
        assert data['current_state'] == 'CONNECTED'

        # Deserialize
        sm2 = TradingStateMachine.deserialize(data)
        assert sm2.state == TradingState.CONNECTED

    def test_guard_registration(self):
        """Test guard condition registration"""
        sm = TradingStateMachine()

        # Register a guard that always fails
        sm.register_guard(
            TradingState.INITIALIZING,
            TradingState.CONNECTED,
            lambda: False
        )

        # Transition should now be blocked
        result = sm.transition_to(TradingState.CONNECTED, trigger="test")
        assert result is False

    def test_state_callback(self):
        """Test state entry callbacks"""
        sm = TradingStateMachine()
        callback_called = {'value': False}

        def on_connected(old_state, new_state):
            callback_called['value'] = True

        sm.register_state_callback(TradingState.CONNECTED, on_connected)
        sm.transition_to(TradingState.CONNECTED, trigger="test")

        assert callback_called['value'] is True


# ============================================================================
# ATOMIC STATE MANAGER TESTS
# ============================================================================

class TestAtomicStateManager:
    """Tests for atomic state persistence"""

    def test_save_and_load_state(self):
        """Test basic save and load functionality"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "test_state.json")
            manager = AtomicStateManager(state_file)

            # Save state
            state = {'key': 'value', 'number': 42}
            result = manager.save_state(state)
            assert result is True

            # Load state
            loaded = manager.load_state()
            assert loaded == state

    def test_checksum_verification(self):
        """Test checksum verification on load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "test_state.json")
            manager = AtomicStateManager(state_file)

            # Save state
            state = {'test': 'data'}
            manager.save_state(state)

            # Corrupt the state file
            with open(state_file, 'w') as f:
                f.write('{"corrupted": true}')

            # Load should detect corruption and try backup
            loaded = manager.load_state()
            # Since backup exists, it should recover
            assert loaded is not None

    def test_backup_creation(self):
        """Test backup files are created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "test_state.json")
            manager = AtomicStateManager(state_file, backup_count=3)

            # Save multiple times
            for i in range(5):
                manager.save_state({'version': i})
                time.sleep(0.01)  # Ensure different timestamps

            # Check backups exist
            backup_dir = Path(tmpdir) / ".test_state_backups"
            backups = list(backup_dir.glob("state_*.json"))

            # Should have at most 3 backups
            assert len(backups) <= 3

    def test_wal_writing(self):
        """Test write-ahead log is created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "test_state.json")
            manager = AtomicStateManager(state_file)

            manager.save_state({'test': 'data'})

            # WAL file should exist
            wal_file = Path(state_file).with_suffix('.wal')
            assert wal_file.exists()

    def test_nonexistent_file(self):
        """Test loading nonexistent file returns None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "nonexistent.json")
            manager = AtomicStateManager(state_file)

            loaded = manager.load_state()
            assert loaded is None


# ============================================================================
# IDEMPOTENT ORDER MANAGER TESTS
# ============================================================================

class TestIdempotentOrderManager:
    """Tests for order idempotency"""

    @pytest.mark.asyncio
    async def test_unique_order_succeeds(self):
        """Test that unique orders are processed"""
        manager = IdempotentOrderManager()

        async def mock_execute():
            return {'order_id': 'ORD123', 'status': 'FILLED'}

        result = await manager.check_or_submit(
            symbol='AAPL',
            side='BUY',
            quantity=100,
            price=150.0,
            timestamp=datetime.now(),
            execute_fn=mock_execute
        )

        assert result['status'] == 'SUCCESS'
        assert result['result']['order_id'] == 'ORD123'

    @pytest.mark.asyncio
    async def test_duplicate_order_blocked(self):
        """Test that duplicate orders are blocked"""
        manager = IdempotentOrderManager()

        async def mock_execute():
            return {'order_id': 'ORD123', 'status': 'FILLED'}

        # Same timestamp (rounded to minute)
        timestamp = datetime.now().replace(second=0, microsecond=0)

        # First order succeeds
        result1 = await manager.check_or_submit(
            symbol='AAPL',
            side='BUY',
            quantity=100,
            price=150.0,
            timestamp=timestamp,
            execute_fn=mock_execute
        )
        assert result1['status'] == 'SUCCESS'

        # Duplicate order blocked
        result2 = await manager.check_or_submit(
            symbol='AAPL',
            side='BUY',
            quantity=100,
            price=150.0,
            timestamp=timestamp,
            execute_fn=mock_execute
        )
        assert result2['status'] == 'DUPLICATE'

    @pytest.mark.asyncio
    async def test_different_orders_both_succeed(self):
        """Test that different orders both succeed"""
        manager = IdempotentOrderManager()

        call_count = {'value': 0}

        async def mock_execute():
            call_count['value'] += 1
            return {'order_id': f'ORD{call_count["value"]}', 'status': 'FILLED'}

        timestamp = datetime.now()

        # Order 1
        result1 = await manager.check_or_submit(
            symbol='AAPL',
            side='BUY',
            quantity=100,
            price=150.0,
            timestamp=timestamp,
            execute_fn=mock_execute
        )

        # Order 2 (different symbol)
        result2 = await manager.check_or_submit(
            symbol='GOOGL',
            side='BUY',
            quantity=50,
            price=2800.0,
            timestamp=timestamp,
            execute_fn=mock_execute
        )

        assert result1['status'] == 'SUCCESS'
        assert result2['status'] == 'SUCCESS'
        assert call_count['value'] == 2

    @pytest.mark.asyncio
    async def test_failed_order_can_be_retried(self):
        """Test that failed orders can be retried"""
        manager = IdempotentOrderManager()

        attempt = {'count': 0}

        async def mock_execute():
            attempt['count'] += 1
            if attempt['count'] == 1:
                raise Exception("Network error")
            return {'order_id': 'ORD123', 'status': 'FILLED'}

        timestamp = datetime.now().replace(second=0, microsecond=0)

        # First attempt fails
        result1 = await manager.check_or_submit(
            symbol='AAPL',
            side='BUY',
            quantity=100,
            price=150.0,
            timestamp=timestamp,
            execute_fn=mock_execute
        )
        assert result1['status'] == 'FAILED'

        # Retry succeeds
        result2 = await manager.check_or_submit(
            symbol='AAPL',
            side='BUY',
            quantity=100,
            price=150.0,
            timestamp=timestamp,
            execute_fn=mock_execute
        )
        assert result2['status'] == 'SUCCESS'


# ============================================================================
# POSITION RECONCILER TESTS
# ============================================================================

class TestPositionReconciler:
    """Tests for position reconciliation"""

    @pytest.mark.asyncio
    async def test_matching_positions(self):
        """Test reconciliation with matching positions"""
        reconciler = PositionReconciler()

        local = {
            'AAPL': {'quantity': 100, 'entry_price': 150.0},
            'GOOGL': {'quantity': 50, 'entry_price': 2800.0}
        }

        async def fetch_broker():
            return {
                'AAPL': {'quantity': 100, 'avg_price': 150.0},
                'GOOGL': {'quantity': 50, 'avg_price': 2800.0}
            }

        report = await reconciler.reconcile(local, fetch_broker)

        assert report.success is True
        assert len(report.discrepancies) == 0

    @pytest.mark.asyncio
    async def test_phantom_position_detected(self):
        """Test detection of phantom positions"""
        reconciler = PositionReconciler()

        local = {
            'AAPL': {'quantity': 100, 'entry_price': 150.0},
            'PHANTOM': {'quantity': 50, 'entry_price': 100.0}  # Not at broker
        }

        async def fetch_broker():
            return {
                'AAPL': {'quantity': 100, 'avg_price': 150.0}
            }

        report = await reconciler.reconcile(local, fetch_broker)

        assert len(report.discrepancies) == 1
        assert report.discrepancies[0].discrepancy_type == 'PHANTOM'
        assert report.discrepancies[0].symbol == 'PHANTOM'

    @pytest.mark.asyncio
    async def test_missing_position_detected(self):
        """Test detection of missing positions"""
        reconciler = PositionReconciler()

        local = {
            'AAPL': {'quantity': 100, 'entry_price': 150.0}
        }

        async def fetch_broker():
            return {
                'AAPL': {'quantity': 100, 'avg_price': 150.0},
                'MISSING': {'quantity': 75, 'avg_price': 200.0}  # Not in local
            }

        report = await reconciler.reconcile(local, fetch_broker)

        assert len(report.discrepancies) == 1
        assert report.discrepancies[0].discrepancy_type == 'MISSING'
        assert report.discrepancies[0].symbol == 'MISSING'

    @pytest.mark.asyncio
    async def test_quantity_mismatch_detected(self):
        """Test detection of quantity mismatches"""
        reconciler = PositionReconciler()

        local = {
            'AAPL': {'quantity': 100, 'entry_price': 150.0}
        }

        async def fetch_broker():
            return {
                'AAPL': {'quantity': 150, 'avg_price': 150.0}  # Different quantity
            }

        report = await reconciler.reconcile(local, fetch_broker)

        assert len(report.discrepancies) == 1
        assert report.discrepancies[0].discrepancy_type == 'QUANTITY_MISMATCH'


# ============================================================================
# HEALTH MONITOR TESTS
# ============================================================================

class TestHealthMonitor:
    """Tests for health monitoring"""

    def test_register_and_run_check(self):
        """Test registering and running health checks"""
        monitor = HealthMonitor()

        def simple_check():
            return HealthCheck(
                component='test',
                status='OK',
                message='Test passed',
                checked_at=datetime.now()
            )

        monitor.register_check('test', simple_check)

        # Run check
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(monitor.run_check('test'))
        loop.close()

        assert result.status == 'OK'
        assert result.component == 'test'

    def test_overall_status_ok(self):
        """Test overall status when all checks pass"""
        monitor = HealthMonitor()

        def ok_check():
            return HealthCheck(
                component='test',
                status='OK',
                message='OK',
                checked_at=datetime.now()
            )

        monitor.register_check('check1', ok_check)
        monitor.register_check('check2', ok_check)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(monitor.run_all_checks())
        loop.close()

        assert monitor.get_overall_status() == 'OK'

    def test_overall_status_critical(self):
        """Test overall status when any check is critical"""
        monitor = HealthMonitor()

        def ok_check():
            return HealthCheck(
                component='ok',
                status='OK',
                message='OK',
                checked_at=datetime.now()
            )

        def critical_check():
            return HealthCheck(
                component='critical',
                status='CRITICAL',
                message='Failed',
                checked_at=datetime.now()
            )

        monitor.register_check('ok', ok_check)
        monitor.register_check('critical', critical_check)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(monitor.run_all_checks())
        loop.close()

        assert monitor.get_overall_status() == 'CRITICAL'
        assert monitor.is_healthy() is False

    def test_memory_health_check(self):
        """Test memory health check factory"""
        check_fn = create_memory_health_check(threshold_mb=10000)
        result = check_fn()

        assert result.component == 'memory'
        assert result.status in ['OK', 'WARNING', 'CRITICAL', 'UNKNOWN']

    def test_disk_health_check(self):
        """Test disk health check factory"""
        check_fn = create_disk_health_check(threshold_percent=99)
        result = check_fn()

        assert result.component == 'disk'
        assert result.status in ['OK', 'WARNING', 'CRITICAL', 'UNKNOWN']


# ============================================================================
# CIRCUIT BREAKER TESTS
# ============================================================================

class TestEnhancedCircuitBreaker:
    """Tests for enhanced circuit breaker"""

    def test_initial_state_closed(self):
        """Test circuit breaker starts in closed state"""
        cb = EnhancedCircuitBreaker(name='test', failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        """Test circuit breaker opens after failure threshold"""
        cb = EnhancedCircuitBreaker(name='test', failure_threshold=3)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_blocks_calls_when_open(self):
        """Test circuit breaker blocks calls when open"""
        cb = EnhancedCircuitBreaker(name='test', failure_threshold=3)

        # Open the circuit
        for _ in range(3):
            cb.record_failure()

        assert cb.can_execute() is False

    def test_success_resets_in_half_open(self):
        """Test successful call resets circuit from half-open"""
        cb = EnhancedCircuitBreaker(
            name='test',
            failure_threshold=3,
            recovery_timeout=0.1,
            half_open_max_calls=1
        )

        # Open the circuit
        for _ in range(3):
            cb.record_failure()

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should transition to half-open and allow execution
        assert cb.can_execute() is True
        cb.record_success()

        # Should now be closed
        assert cb.state == CircuitState.CLOSED

    def test_failure_in_half_open_reopens(self):
        """Test failure in half-open state reopens circuit"""
        cb = EnhancedCircuitBreaker(
            name='test',
            failure_threshold=3,
            recovery_timeout=0.1
        )

        # Open the circuit
        for _ in range(3):
            cb.record_failure()

        # Wait for recovery timeout
        time.sleep(0.15)

        # Transition to half-open
        _ = cb.state  # This triggers the transition check

        # Record failure
        cb.record_failure()

        # Should be back to open
        assert cb.state == CircuitState.OPEN

    def test_metrics(self):
        """Test circuit breaker metrics"""
        cb = EnhancedCircuitBreaker(name='test', failure_threshold=5)

        cb.record_success()
        cb.record_success()
        cb.record_failure()

        metrics = cb.get_metrics()

        assert metrics['total_calls'] == 3
        assert metrics['total_failures'] == 1
        assert metrics['failure_rate'] == 1/3

    def test_decorator(self):
        """Test circuit breaker as decorator"""
        cb = EnhancedCircuitBreaker(name='test', failure_threshold=3)

        @cb
        def protected_function():
            return "success"

        result = protected_function()
        assert result == "success"

    def test_decorator_catches_exceptions(self):
        """Test circuit breaker decorator catches exceptions"""
        cb = EnhancedCircuitBreaker(name='test', failure_threshold=3)

        @cb
        def failing_function():
            raise ValueError("Test error")

        for _ in range(3):
            try:
                failing_function()
            except ValueError:
                pass

        assert cb.state == CircuitState.OPEN

    def test_manual_reset(self):
        """Test manual circuit breaker reset"""
        cb = EnhancedCircuitBreaker(name='test', failure_threshold=3)

        # Open the circuit
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Manual reset
        cb.reset()
        assert cb.state == CircuitState.CLOSED


# ============================================================================
# VALIDATOR TESTS
# ============================================================================

class TestPositionSizeValidator:
    """Tests for position size validation"""

    def test_valid_position(self):
        """Test valid position passes validation"""
        validator = PositionSizeValidator(
            max_position_value=10000,
            min_position_value=100,
            max_portfolio_percent=0.20
        )

        is_valid, message = validator.validate(
            symbol='AAPL',
            quantity=50,
            price=150.0,
            portfolio_value=100000
        )

        assert is_valid is True

    def test_exceeds_max_value(self):
        """Test position exceeding max value is rejected"""
        validator = PositionSizeValidator(max_position_value=5000)

        is_valid, message = validator.validate(
            symbol='AAPL',
            quantity=100,
            price=100.0,  # $10,000 position
            portfolio_value=100000
        )

        assert is_valid is False
        assert 'exceeds max' in message

    def test_below_min_value(self):
        """Test position below min value is rejected"""
        validator = PositionSizeValidator(min_position_value=1000)

        is_valid, message = validator.validate(
            symbol='AAPL',
            quantity=5,
            price=100.0,  # $500 position
            portfolio_value=100000
        )

        assert is_valid is False
        assert 'below min' in message

    def test_exceeds_portfolio_percent(self):
        """Test position exceeding portfolio percent is rejected"""
        validator = PositionSizeValidator(
            max_position_value=100000,
            max_portfolio_percent=0.10
        )

        is_valid, message = validator.validate(
            symbol='AAPL',
            quantity=200,
            price=100.0,  # $20,000 = 20% of $100,000
            portfolio_value=100000
        )

        assert is_valid is False
        assert 'exceeds max' in message

    def test_nan_quantity_rejected(self):
        """Test NaN quantity is rejected"""
        validator = PositionSizeValidator()

        is_valid, message = validator.validate(
            symbol='AAPL',
            quantity=float('nan'),
            price=150.0,
            portfolio_value=100000
        )

        assert is_valid is False
        assert 'Invalid quantity' in message

    def test_inf_price_rejected(self):
        """Test infinite price is rejected"""
        validator = PositionSizeValidator()

        is_valid, message = validator.validate(
            symbol='AAPL',
            quantity=100,
            price=float('inf'),
            portfolio_value=100000
        )

        assert is_valid is False
        assert 'Invalid price' in message


class TestFeatureValidator:
    """Tests for ML feature validation"""

    def test_valid_features(self):
        """Test valid features pass validation"""
        validator = FeatureValidator()

        features = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        is_valid, issues, cleaned = validator.validate(features)

        assert is_valid is True
        assert len(issues) == 0
        assert np.array_equal(features, cleaned)

    def test_nan_detected(self):
        """Test NaN values are detected"""
        validator = FeatureValidator()

        features = np.array([1.0, float('nan'), 3.0, 4.0, 5.0])
        is_valid, issues, cleaned = validator.validate(features)

        assert is_valid is False
        assert any('NaN' in issue for issue in issues)
        assert not np.isnan(cleaned).any()

    def test_inf_detected(self):
        """Test infinite values are detected"""
        validator = FeatureValidator()

        features = np.array([1.0, float('inf'), 3.0, float('-inf'), 5.0])
        is_valid, issues, cleaned = validator.validate(features)

        assert is_valid is False
        assert any('Inf' in issue for issue in issues)
        assert not np.isinf(cleaned).any()

    def test_feature_names_in_issues(self):
        """Test feature names are included in issues"""
        validator = FeatureValidator()

        features = np.array([1.0, float('nan'), 3.0])
        names = ['rsi', 'macd', 'volume']
        is_valid, issues, cleaned = validator.validate(features, names)

        assert 'macd' in issues[0]

    def test_statistics_tracking(self):
        """Test validation statistics are tracked"""
        validator = FeatureValidator()

        # First validation with NaN
        validator.validate(np.array([1.0, float('nan')]))

        # Second validation clean
        validator.validate(np.array([1.0, 2.0]))

        stats = validator.get_stats()
        assert stats['total_validations'] == 2
        assert stats['nan_detected'] == 1


# ============================================================================
# INSTITUTIONAL CORE INTEGRATION TESTS
# ============================================================================

class TestInstitutionalTradingCore:
    """Integration tests for the full institutional core"""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test core initialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                'state_file': os.path.join(tmpdir, 'state.json'),
                'max_position_value': 10000
            }
            core = InstitutionalTradingCore(config)

            result = await core.initialize()
            assert result is True
            assert core.state_machine.state == TradingState.CONNECTED

    @pytest.mark.asyncio
    async def test_state_persistence(self):
        """Test state is persisted and can be recovered"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {'state_file': os.path.join(tmpdir, 'state.json')}

            # Create first core and save state
            core1 = InstitutionalTradingCore(config)
            await core1.initialize()
            await core1.save_state({'custom_data': 'test_value'})

            # Create second core and verify state recovery
            core2 = InstitutionalTradingCore(config)
            await core2.initialize()

            assert core2.state_machine.state == TradingState.CONNECTED

    def test_get_status(self):
        """Test status reporting"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {'state_file': os.path.join(tmpdir, 'state.json')}
            core = InstitutionalTradingCore(config)

            status = core.get_status()

            assert 'state' in status
            assert 'health' in status
            assert 'circuit_breakers' in status


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    # Run with pytest for async support
    pytest.main([__file__, '-v', '--tb=short'])
