"""Tests for v-fix-cross-loop-routing-2026-09-15: Cross-loop DB write routing.

Root Cause Analysis (RCA):
  KiddoKingdom soak test (PID 331467, tip 045186b, ~16h) observed 411+
  db_logger_cross_loop_skip warnings (still firing 08:54 ET 2026-09-15).
  Methods affected: log_decision, sync_positions, etc.
  Pattern: owner_loop id ≠ current_loop id — shared DbLogger touched from
  multiple asyncio loops (main engine loop vs news/uvicorn/worker loops).
  
  Before this fix: _check_loop_or_warn → return (silent data loss except warning).
  After this fix: cross-loop calls ROUTE to owner loop via run_coroutine_threadsafe.

Tests cover:
  1. Same-loop write OK (no routing needed)
  2. Cross-loop routes to owner loop (not skip)
  3. DB_LOGGER_CROSS_LOOP_ROUTE=false restores legacy skip behavior
  4. Owner loop is sticky (refuses overwrite from secondary loops)
  5. Rate-limiting of warnings (max once per 60s per method)
"""
import asyncio
import logging
import os
import pytest
import time
from unittest.mock import MagicMock, AsyncMock, patch


def _load_db_logger_module():
    """Load db_logger module directly without triggering data_providers/__init__.py."""
    import sys
    sys.path.insert(0, '/workspace')
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "db_logger_direct", "/workspace/data_providers/db_logger.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCrossLoopRouting:
    """Tests for cross-loop routing behavior."""

    @pytest.fixture
    def mock_db_logger(self):
        """Create a DbLogger with mocked engine for testing."""
        db_logger_module = _load_db_logger_module()
        DbLogger = db_logger_module.DbLogger
        
        with patch.object(db_logger_module, 'create_async_engine') as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine
            db_logger = DbLogger(dsn="postgresql+asyncpg://test:test@localhost/test")
        
        return db_logger

    @pytest.mark.asyncio
    async def test_same_loop_write_ok(self, mock_db_logger):
        """Write from same loop proceeds directly without routing."""
        current_loop = asyncio.get_running_loop()
        mock_db_logger.set_owner_loop(current_loop)
        
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_db_logger._engine.begin = MagicMock(return_value=mock_cm)
        
        with patch('asyncio.run_coroutine_threadsafe') as mock_route:
            await mock_db_logger.log_decision("test", "TSLA", "buy", "test_reason")
        
        mock_route.assert_not_called()
        mock_db_logger._engine.begin.assert_called_once()
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_cross_loop_routes_to_owner(self, mock_db_logger):
        """Write from different loop routes to owner loop via run_coroutine_threadsafe."""
        owner_loop = asyncio.new_event_loop()
        mock_db_logger.set_owner_loop(owner_loop)
        
        routed_coros = []
        
        def capture_route(coro, loop):
            routed_coros.append(coro)
            future = MagicMock()
            return future
        
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_db_logger._engine.begin = MagicMock(return_value=mock_cm)
        
        with patch.object(owner_loop, 'is_running', return_value=True):
            with patch('asyncio.run_coroutine_threadsafe', side_effect=capture_route):
                await mock_db_logger.log_decision("test", "TSLA", "buy", "test_reason")
        
        assert len(routed_coros) == 1
        mock_db_logger._engine.begin.assert_not_called()
        
        owner_loop.close()

    @pytest.mark.asyncio
    async def test_cross_loop_routes_log_trade(self, mock_db_logger):
        """log_trade routes cross-loop calls to owner loop."""
        from datetime import datetime, timezone
        
        owner_loop = asyncio.new_event_loop()
        mock_db_logger.set_owner_loop(owner_loop)
        
        routed_coros = []
        
        def capture_route(coro, loop):
            routed_coros.append(coro)
            return MagicMock()
        
        with patch.object(owner_loop, 'is_running', return_value=True):
            with patch('asyncio.run_coroutine_threadsafe', side_effect=capture_route):
                now = datetime.now(timezone.utc)
                await mock_db_logger.log_trade(
                    symbol="TSLA", side="long", strategy="test",
                    entry_time=now, exit_time=now,
                    entry_price=100.0, exit_price=105.0,
                    quantity=10, pnl=50.0, pnl_pct=5.0,
                    exit_reason="take_profit"
                )
        
        assert len(routed_coros) == 1
        owner_loop.close()

    @pytest.mark.asyncio
    async def test_cross_loop_routes_sync_positions(self, mock_db_logger):
        """sync_positions routes cross-loop calls to owner loop."""
        owner_loop = asyncio.new_event_loop()
        mock_db_logger.set_owner_loop(owner_loop)
        
        routed_coros = []
        
        def capture_route(coro, loop):
            routed_coros.append(coro)
            return MagicMock()
        
        mock_pos = MagicMock()
        mock_pos.quantity = 10
        mock_pos.symbol = "TSLA"
        
        with patch.object(owner_loop, 'is_running', return_value=True):
            with patch('asyncio.run_coroutine_threadsafe', side_effect=capture_route):
                await mock_db_logger.sync_positions({"TSLA": mock_pos})
        
        assert len(routed_coros) == 1
        owner_loop.close()

    @pytest.mark.asyncio
    async def test_cross_loop_routes_delete_position(self, mock_db_logger):
        """delete_position routes cross-loop calls to owner loop."""
        owner_loop = asyncio.new_event_loop()
        mock_db_logger.set_owner_loop(owner_loop)
        
        routed_coros = []
        
        def capture_route(coro, loop):
            routed_coros.append(coro)
            return MagicMock()
        
        with patch.object(owner_loop, 'is_running', return_value=True):
            with patch('asyncio.run_coroutine_threadsafe', side_effect=capture_route):
                await mock_db_logger.delete_position("TSLA")
        
        assert len(routed_coros) == 1
        owner_loop.close()

    @pytest.mark.asyncio
    async def test_cross_loop_routes_log_news_veto(self, mock_db_logger):
        """log_news_veto routes cross-loop calls to owner loop."""
        owner_loop = asyncio.new_event_loop()
        mock_db_logger.set_owner_loop(owner_loop)
        
        routed_coros = []
        
        def capture_route(coro, loop):
            routed_coros.append(coro)
            return MagicMock()
        
        with patch.object(owner_loop, 'is_running', return_value=True):
            with patch('asyncio.run_coroutine_threadsafe', side_effect=capture_route):
                await mock_db_logger.log_news_veto(
                    symbol="TSLA", side="long",
                    veto_reason="test_veto"
                )
        
        assert len(routed_coros) == 1
        owner_loop.close()


class TestCrossLoopRouteFlag:
    """Tests for DB_LOGGER_CROSS_LOOP_ROUTE flag behavior."""

    @pytest.fixture
    def mock_db_logger_route_disabled(self):
        """Create a DbLogger with routing disabled via flag."""
        db_logger_module = _load_db_logger_module()
        
        original_flag = db_logger_module.DB_LOGGER_CROSS_LOOP_ROUTE
        db_logger_module.DB_LOGGER_CROSS_LOOP_ROUTE = False
        
        DbLogger = db_logger_module.DbLogger
        
        with patch.object(db_logger_module, 'create_async_engine') as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine
            db_logger = DbLogger(dsn="postgresql+asyncpg://test:test@localhost/test")
        
        yield db_logger, db_logger_module
        
        db_logger_module.DB_LOGGER_CROSS_LOOP_ROUTE = original_flag

    @pytest.mark.asyncio
    async def test_flag_off_restores_skip_behavior(self, mock_db_logger_route_disabled):
        """When DB_LOGGER_CROSS_LOOP_ROUTE=false, cross-loop calls skip (legacy behavior)."""
        db_logger, db_logger_module = mock_db_logger_route_disabled
        
        assert db_logger_module.DB_LOGGER_CROSS_LOOP_ROUTE is False
        
        owner_loop = asyncio.new_event_loop()
        db_logger.set_owner_loop(owner_loop)
        
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        db_logger._engine.begin = MagicMock(return_value=mock_cm)
        
        with patch('asyncio.run_coroutine_threadsafe') as mock_route:
            await db_logger.log_decision("test", "TSLA", "buy", "test_reason")
        
        mock_route.assert_not_called()
        db_logger._engine.begin.assert_not_called()
        
        owner_loop.close()


class TestOwnerLoopSticky:
    """Tests for sticky owner loop behavior."""

    @pytest.fixture
    def mock_db_logger(self):
        """Create a DbLogger with mocked engine for testing."""
        db_logger_module = _load_db_logger_module()
        DbLogger = db_logger_module.DbLogger
        
        with patch.object(db_logger_module, 'create_async_engine') as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine
            db_logger = DbLogger(dsn="postgresql+asyncpg://test:test@localhost/test")
        
        return db_logger

    def test_owner_loop_sticky_first_call_sets(self, mock_db_logger):
        """First set_owner_loop call sets the owner loop."""
        loop1 = asyncio.new_event_loop()
        
        assert mock_db_logger._owner_loop is None
        assert mock_db_logger._owner_loop_set_once is False
        
        mock_db_logger.set_owner_loop(loop1)
        
        assert mock_db_logger._owner_loop is loop1
        assert mock_db_logger._owner_loop_set_once is True
        
        loop1.close()

    def test_owner_loop_sticky_refuses_overwrite(self, mock_db_logger, caplog):
        """Second set_owner_loop from different loop is ignored with warning."""
        loop1 = asyncio.new_event_loop()
        loop2 = asyncio.new_event_loop()
        
        mock_db_logger.set_owner_loop(loop1)
        
        assert mock_db_logger._owner_loop is loop1
        
        with caplog.at_level(logging.WARNING):
            mock_db_logger.set_owner_loop(loop2)
        
        assert mock_db_logger._owner_loop is loop1
        assert "set_owner_loop_ignored" in caplog.text
        assert "owner loop is sticky" in caplog.text
        
        loop1.close()
        loop2.close()

    def test_owner_loop_sticky_same_loop_noop(self, mock_db_logger, caplog):
        """Re-setting same loop is a no-op (no warning)."""
        loop1 = asyncio.new_event_loop()
        
        mock_db_logger.set_owner_loop(loop1)
        
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            mock_db_logger.set_owner_loop(loop1)
        
        assert mock_db_logger._owner_loop is loop1
        assert "set_owner_loop_ignored" not in caplog.text
        
        loop1.close()


class TestRateLimitWarnings:
    """Tests for rate-limited warnings."""

    @pytest.fixture
    def mock_db_logger(self):
        """Create a DbLogger with mocked engine for testing."""
        db_logger_module = _load_db_logger_module()
        DbLogger = db_logger_module.DbLogger
        
        with patch.object(db_logger_module, 'create_async_engine') as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine
            db_logger = DbLogger(dsn="postgresql+asyncpg://test:test@localhost/test")
        
        return db_logger

    def test_warn_once_logs_first_call(self, mock_db_logger, caplog):
        """_warn_once logs on first call."""
        with caplog.at_level(logging.WARNING):
            mock_db_logger._warn_once("test_key", "test warning message %s", "arg1")
        
        assert "test warning message arg1" in caplog.text

    def test_warn_once_suppresses_repeated_calls(self, mock_db_logger, caplog):
        """_warn_once suppresses warnings within rate limit interval."""
        with caplog.at_level(logging.WARNING):
            mock_db_logger._warn_once("test_key", "first warning")
            caplog.clear()
            mock_db_logger._warn_once("test_key", "second warning")
        
        assert "second warning" not in caplog.text

    def test_warn_once_different_keys_independent(self, mock_db_logger, caplog):
        """_warn_once allows different keys to warn independently."""
        with caplog.at_level(logging.WARNING):
            mock_db_logger._warn_once("key_a", "warning A")
            mock_db_logger._warn_once("key_b", "warning B")
        
        assert "warning A" in caplog.text
        assert "warning B" in caplog.text


class TestRouteToOwnerLoopEdgeCases:
    """Tests for _route_to_owner_loop edge cases."""

    @pytest.fixture
    def mock_db_logger(self):
        """Create a DbLogger with mocked engine for testing."""
        db_logger_module = _load_db_logger_module()
        DbLogger = db_logger_module.DbLogger
        
        with patch.object(db_logger_module, 'create_async_engine') as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine
            db_logger = DbLogger(dsn="postgresql+asyncpg://test:test@localhost/test")
        
        return db_logger

    @pytest.mark.asyncio
    async def test_no_owner_loop_skips_with_warning(self, mock_db_logger, caplog):
        """Cross-loop call with no owner loop set skips with warning."""
        assert mock_db_logger._owner_loop is None
        
        async def dummy_coro():
            pass
        
        with caplog.at_level(logging.WARNING):
            result = mock_db_logger._route_to_owner_loop(
                "test_method", lambda: dummy_coro()
            )
        
        assert result is False

    @pytest.mark.asyncio
    async def test_dead_owner_loop_skips_with_warning(self, mock_db_logger, caplog):
        """Cross-loop call with dead owner loop skips with warning."""
        dead_loop = asyncio.new_event_loop()
        mock_db_logger.set_owner_loop(dead_loop)
        dead_loop.close()
        
        async def dummy_coro():
            pass
        
        with caplog.at_level(logging.WARNING):
            result = mock_db_logger._route_to_owner_loop(
                "test_method", lambda: dummy_coro()
            )
        
        assert result is True
        assert "cross_loop_dead" in caplog.text


class TestImplMethodsWork:
    """Tests that _impl methods work correctly when called directly."""

    @pytest.fixture
    def mock_db_logger(self):
        """Create a DbLogger with mocked engine for testing."""
        db_logger_module = _load_db_logger_module()
        DbLogger = db_logger_module.DbLogger
        
        with patch.object(db_logger_module, 'create_async_engine') as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine
            db_logger = DbLogger(dsn="postgresql+asyncpg://test:test@localhost/test")
        
        return db_logger

    @pytest.mark.asyncio
    async def test_log_decision_impl_executes_insert(self, mock_db_logger):
        """_log_decision_impl inserts row into bot_decisions."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_db_logger._engine.begin = MagicMock(return_value=mock_cm)
        
        await mock_db_logger._log_decision_impl(
            component="test",
            symbol="TSLA",
            action="buy",
            reason="test_reason",
            mode=None,
            signal_type=None,
            confidence=0.8,
            strength=None,
            meta_proba=None,
            atr=None,
            stop_distance=None,
            price=100.0,
            extra={"custom_field": "value"},
        )
        
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        
        assert params['component'] == "test"
        assert params['symbol'] == "TSLA"
        assert params['action'] == "buy"
        assert params['confidence'] == 0.8

    @pytest.mark.asyncio
    async def test_sync_positions_impl_uses_lock(self, mock_db_logger):
        """_sync_positions_impl acquires the sync lock."""
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        mock_db_logger._engine.begin = MagicMock(return_value=mock_cm)
        
        await mock_db_logger._sync_positions_impl({})
        
        mock_db_logger._engine.begin.assert_called_once()


class TestCheckLoopOrWarnRateLimited:
    """Tests for _check_loop_or_warn rate limiting."""

    @pytest.fixture
    def mock_db_logger(self):
        """Create a DbLogger with mocked engine for testing."""
        db_logger_module = _load_db_logger_module()
        DbLogger = db_logger_module.DbLogger
        
        with patch.object(db_logger_module, 'create_async_engine') as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine
            db_logger = DbLogger(dsn="postgresql+asyncpg://test:test@localhost/test")
        
        return db_logger

    def test_check_loop_or_warn_rate_limited(self, mock_db_logger, caplog):
        """_check_loop_or_warn rate-limits repeated warnings for same method."""
        other_loop = asyncio.new_event_loop()
        mock_db_logger.set_owner_loop(other_loop)
        
        current_loop = asyncio.new_event_loop()
        
        async def check_from_current():
            return mock_db_logger._check_loop_or_warn("test_method")
        
        with caplog.at_level(logging.WARNING):
            result1 = current_loop.run_until_complete(check_from_current())
            assert result1 is False
            assert "db_logger_cross_loop_skip" in caplog.text
            
            caplog.clear()
            result2 = current_loop.run_until_complete(check_from_current())
            assert result2 is False
            assert "db_logger_cross_loop_skip" not in caplog.text
        
        other_loop.close()
        current_loop.close()
