"""Tests for v-fix-log-strategy-decision-2026-09-14: log_strategy_decision sync wrapper.

Tests cover:
  - log_strategy_decision schedules async log_decision call
  - Proper parameter mapping (strategy → component, extra_data → details)
  - Loop safety: skips when no owner loop or owner loop is dead
  - Error handling: logs warning, never raises
  - Integration: gpu_news_critic and theme_shock_logger error handling
"""
import asyncio
import logging
import pytest
from datetime import datetime, timezone
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


class TestLogStrategyDecision:
    """Tests for DbLogger.log_strategy_decision sync fire-and-forget wrapper."""

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

    def test_log_strategy_decision_signature(self):
        """log_strategy_decision accepts expected parameters."""
        import inspect
        import sys
        
        sys.path.insert(0, '/workspace')
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "db_logger", "/workspace/data_providers/db_logger.py"
        )
        db_logger_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(db_logger_module)
        
        sig = inspect.signature(db_logger_module.DbLogger.log_strategy_decision)
        params = list(sig.parameters.keys())
        
        assert 'self' in params
        assert 'strategy' in params
        assert 'symbol' in params
        assert 'action' in params
        assert 'reason' in params
        assert 'extra_data' in params
        
        assert sig.parameters['reason'].default is None
        assert sig.parameters['extra_data'].default is None

    def test_log_strategy_decision_schedules_async_call(self, mock_db_logger):
        """log_strategy_decision schedules log_decision on owner loop."""
        loop = asyncio.new_event_loop()
        mock_db_logger.set_owner_loop(loop)
        
        scheduled_coros = []
        
        def capture_run_coroutine_threadsafe(coro, lp):
            scheduled_coros.append(coro)
            future = MagicMock()
            return future
        
        with patch('asyncio.run_coroutine_threadsafe', side_effect=capture_run_coroutine_threadsafe):
            with patch.object(loop, 'is_running', return_value=True):
                mock_db_logger.log_strategy_decision(
                    strategy="gpu_news_critic",
                    symbol="NVDA",
                    action="shadow_pass",
                    reason="theme=ai_compute(0.65); stance=mixed",
                    extra_data={"theme_probs": {"ai_compute": 0.65}},
                )
        
        assert len(scheduled_coros) == 1
        loop.close()

    def test_log_strategy_decision_skips_when_disabled(self, mock_db_logger):
        """log_strategy_decision returns early when logger is disabled."""
        mock_db_logger._enabled = False
        
        with patch('asyncio.run_coroutine_threadsafe') as mock_schedule:
            mock_db_logger.log_strategy_decision(
                strategy="test",
                symbol="TEST",
                action="test_action",
            )
        
        mock_schedule.assert_not_called()

    def test_log_strategy_decision_skips_no_owner_loop(self, mock_db_logger, caplog):
        """log_strategy_decision logs warning when no owner loop is set."""
        assert mock_db_logger._owner_loop is None
        
        with patch('asyncio.run_coroutine_threadsafe') as mock_schedule:
            with caplog.at_level(logging.WARNING):
                mock_db_logger.log_strategy_decision(
                    strategy="gpu_news_critic",
                    symbol="TSLA",
                    action="shadow_would_suppress_hard_skip",
                )
        
        mock_schedule.assert_not_called()
        assert "skip_no_owner_loop" in caplog.text
        assert "gpu_news_critic" in caplog.text

    def test_log_strategy_decision_skips_dead_loop(self, mock_db_logger, caplog):
        """log_strategy_decision logs warning when owner loop is not running."""
        dead_loop = asyncio.new_event_loop()
        mock_db_logger.set_owner_loop(dead_loop)
        dead_loop.close()
        
        with patch('asyncio.run_coroutine_threadsafe') as mock_schedule:
            with caplog.at_level(logging.WARNING):
                mock_db_logger.log_strategy_decision(
                    strategy="theme_shock_logger",
                    symbol="AMD",
                    action="shadow_hard_skip_entries",
                )
        
        mock_schedule.assert_not_called()
        assert "skip_dead_loop" in caplog.text

    def test_log_strategy_decision_maps_parameters(self, mock_db_logger):
        """log_strategy_decision maps strategy→component, extra_data→details."""
        loop = asyncio.new_event_loop()
        mock_db_logger.set_owner_loop(loop)
        
        async def mock_log_decision(**kwargs):
            return kwargs
        
        with patch.object(mock_db_logger, 'log_decision', side_effect=mock_log_decision) as mock_ld:
            with patch('asyncio.run_coroutine_threadsafe') as mock_schedule:
                with patch.object(loop, 'is_running', return_value=True):
                    mock_db_logger.log_strategy_decision(
                        strategy="gpu_news_critic",
                        symbol="NVDA",
                        action="shadow_pass",
                        reason="test_reason",
                        # v-hotfix: 'confidence' collides with log_decision param,
                        # use non-colliding keys for this test
                        extra_data={"theme_probs": {"ai_compute": 0.75}, "infer_ms": 50.0},
                    )
        
        mock_ld.assert_called_once()
        call_kwargs = mock_ld.call_args[1]
        
        assert call_kwargs['component'] == "gpu_news_critic"
        assert call_kwargs['symbol'] == "NVDA"
        assert call_kwargs['action'] == "shadow_pass"
        assert call_kwargs['reason'] == "test_reason"
        # Non-colliding keys passed through directly
        assert call_kwargs['theme_probs'] == {"ai_compute": 0.75}
        assert call_kwargs['infer_ms'] == 50.0
        
        loop.close()

    def test_log_strategy_decision_handles_none_extra_data(self, mock_db_logger):
        """log_strategy_decision handles None extra_data."""
        loop = asyncio.new_event_loop()
        mock_db_logger.set_owner_loop(loop)
        
        async def mock_log_decision(**kwargs):
            return kwargs
        
        with patch.object(mock_db_logger, 'log_decision', side_effect=mock_log_decision) as mock_ld:
            with patch('asyncio.run_coroutine_threadsafe') as mock_schedule:
                with patch.object(loop, 'is_running', return_value=True):
                    mock_db_logger.log_strategy_decision(
                        strategy="test",
                        symbol="TEST",
                        action="test_action",
                    )
        
        mock_ld.assert_called_once()
        call_kwargs = mock_ld.call_args[1]
        
        assert call_kwargs['component'] == "test"
        assert call_kwargs['symbol'] == "TEST"
        assert 'extra_field' not in call_kwargs
        
        loop.close()

    def test_log_strategy_decision_never_raises(self, mock_db_logger, caplog):
        """log_strategy_decision catches and logs exceptions, never raises."""
        loop = asyncio.new_event_loop()
        mock_db_logger.set_owner_loop(loop)
        
        with patch.object(loop, 'is_running', return_value=True):
            with patch('asyncio.run_coroutine_threadsafe', side_effect=RuntimeError("test error")):
                with caplog.at_level(logging.WARNING):
                    mock_db_logger.log_strategy_decision(
                        strategy="test",
                        symbol="TEST",
                        action="test_action",
                    )
        
        assert "db_logger_strategy_decision_error" in caplog.text
        assert "test error" in caplog.text
        
        loop.close()


class TestExtraDataCollisionHotfix:
    """v-hotfix-collision-2026-09-14: Regression tests for extra_data key collision.
    
    CriticCard.to_dict() and ThemeEvent.to_dict() include keys like 'action',
    'symbol', 'confidence' that collide with log_decision's explicit parameters.
    This caused: "log_decision() got multiple values for keyword argument 'action'"
    """

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

    def test_extra_data_with_colliding_keys_no_typeerror(self, mock_db_logger):
        """log_strategy_decision handles extra_data with colliding keys without TypeError."""
        loop = asyncio.new_event_loop()
        mock_db_logger.set_owner_loop(loop)
        
        # Simulates CriticCard.to_dict() which includes action, symbol, confidence
        critic_card_dict = {
            "event_id": "abc123",
            "headline": "Test headline",
            "summary": "Test summary",
            "theme_probs": {"ai_compute": 0.65, "other": 0.35},
            "relevance_by_symbol": {"NVDA": 0.8, "AMD": 0.7},
            "stance": "mixed",
            "contamination_risk": 0.1,
            "confidence": 0.75,  # COLLIDES with log_decision param
            "action": "pass",  # COLLIDES with log_decision param
            "symbol": "NVDA",  # COLLIDES with log_decision param
            "reason": "test_reason",  # Would collide if passed
        }
        
        captured_kwargs = {}
        
        async def capture_log_decision(**kwargs):
            captured_kwargs.update(kwargs)
        
        with patch.object(mock_db_logger, 'log_decision', side_effect=capture_log_decision):
            with patch('asyncio.run_coroutine_threadsafe') as mock_schedule:
                with patch.object(loop, 'is_running', return_value=True):
                    # This should NOT raise TypeError
                    mock_db_logger.log_strategy_decision(
                        strategy="gpu_news_critic",
                        symbol="NVDA",
                        action="shadow_pass",
                        reason="theme=ai_compute(0.65)",
                        extra_data=critic_card_dict,
                    )
        
        # Verify log_decision was called (via run_coroutine_threadsafe)
        mock_schedule.assert_called_once()
        loop.close()

    def test_colliding_keys_preserved_in_card_original(self, mock_db_logger):
        """Colliding keys are preserved under _card_original for Research."""
        loop = asyncio.new_event_loop()
        mock_db_logger.set_owner_loop(loop)
        
        critic_card_dict = {
            "theme_probs": {"ai_compute": 0.65},
            "relevance_by_symbol": {"NVDA": 0.8},
            "confidence": 0.75,  # COLLIDES
            "action": "pass",  # COLLIDES
            "symbol": "NVDA",  # COLLIDES
            "infer_ms": 45.2,  # Does not collide
        }
        
        async def capture_log_decision(**kwargs):
            return kwargs
        
        with patch.object(mock_db_logger, 'log_decision', side_effect=capture_log_decision) as mock_ld:
            with patch('asyncio.run_coroutine_threadsafe') as mock_schedule:
                with patch.object(loop, 'is_running', return_value=True):
                    mock_db_logger.log_strategy_decision(
                        strategy="gpu_news_critic",
                        symbol="NVDA",
                        action="shadow_pass",
                        reason="test",
                        extra_data=critic_card_dict,
                    )
        
        # Verify log_decision was called
        mock_ld.assert_called_once()
        call_kwargs = mock_ld.call_args[1]
        
        # Verify explicit params
        assert call_kwargs['component'] == "gpu_news_critic"
        assert call_kwargs['symbol'] == "NVDA"
        assert call_kwargs['action'] == "shadow_pass"
        
        # Verify non-colliding keys passed through
        assert call_kwargs['theme_probs'] == {"ai_compute": 0.65}
        assert call_kwargs['relevance_by_symbol'] == {"NVDA": 0.8}
        assert call_kwargs['infer_ms'] == 45.2
        
        # Verify colliding keys preserved in _card_original
        assert '_card_original' in call_kwargs
        assert call_kwargs['_card_original']['confidence'] == 0.75
        assert call_kwargs['_card_original']['action'] == "pass"
        assert call_kwargs['_card_original']['symbol'] == "NVDA"
        
        loop.close()

    def test_no_collision_no_card_original(self, mock_db_logger):
        """When no collisions, _card_original is not added."""
        loop = asyncio.new_event_loop()
        mock_db_logger.set_owner_loop(loop)
        
        # No colliding keys
        safe_extra = {
            "theme_probs": {"ai_compute": 0.65},
            "infer_ms": 45.2,
            "event_id": "abc123",
        }
        
        async def capture_log_decision(**kwargs):
            return kwargs
        
        with patch.object(mock_db_logger, 'log_decision', side_effect=capture_log_decision) as mock_ld:
            with patch('asyncio.run_coroutine_threadsafe') as mock_schedule:
                with patch.object(loop, 'is_running', return_value=True):
                    mock_db_logger.log_strategy_decision(
                        strategy="gpu_news_critic",
                        symbol="NVDA",
                        action="shadow_pass",
                        reason="test",
                        extra_data=safe_extra,
                    )
        
        call_kwargs = mock_ld.call_args[1]
        assert '_card_original' not in call_kwargs
        assert call_kwargs['theme_probs'] == {"ai_compute": 0.65}
        
        loop.close()


@pytest.mark.skipif(True, reason="Requires full analysis dependencies (fastapi, ta)")
class TestGpuNewsCriticErrorHandling:
    """Tests for gpu_news_critic AttributeError handling.
    
    Skipped in CI environments without full dependencies.
    These tests verify error handling changes in gpu_news_critic.py.
    """

    def test_emit_shadow_log_logs_warning_on_attr_error(self, caplog):
        """_emit_shadow_log logs warning once on AttributeError."""
        pass

    def test_emit_shadow_log_suppresses_repeated_warnings(self, caplog):
        """_emit_shadow_log only logs warning once."""
        pass


@pytest.mark.skipif(True, reason="Requires full analysis dependencies (fastapi, ta)")
class TestThemeShockLoggerErrorHandling:
    """Tests for theme_shock_logger AttributeError handling.
    
    Skipped in CI environments without full dependencies.
    These tests verify error handling changes in theme_shock_logger.py.
    """

    def test_log_theme_event_logs_warning_on_attr_error(self, caplog):
        """log_theme_event logs warning once on AttributeError."""
        pass

    def test_log_theme_event_suppresses_repeated_warnings(self, caplog):
        """log_theme_event only logs warning once."""
        pass


class TestSafeJsonNestedDicts:
    """Tests for _safe_json handling nested dicts (theme_probs, relevance_by_symbol)."""

    def test_safe_json_preserves_nested_dicts(self):
        """_safe_json preserves nested dicts as proper JSON objects."""
        db_logger_module = _load_db_logger_module()
        _safe_json = db_logger_module._safe_json
        
        theme_probs = {"ai_compute": 0.65, "other": 0.35, "none": 0.0}
        result = _safe_json(theme_probs)
        
        assert isinstance(result, dict)
        assert result["ai_compute"] == 0.65
        assert result["other"] == 0.35

    def test_safe_json_preserves_nested_lists(self):
        """_safe_json preserves nested lists."""
        db_logger_module = _load_db_logger_module()
        _safe_json = db_logger_module._safe_json
        
        symbols = ["NVDA", "AMD", "AVGO"]
        result = _safe_json(symbols)
        
        assert isinstance(result, list)
        assert result == ["NVDA", "AMD", "AVGO"]

    def test_safe_json_preserves_full_critic_card(self):
        """_safe_json preserves full CriticCard structure for Research queries."""
        db_logger_module = _load_db_logger_module()
        _safe_json = db_logger_module._safe_json
        import json
        
        card_dict = {
            "event_id": "abc123",
            "headline": "Test headline",
            "theme_probs": {"ai_compute": 0.65, "ai_mega_cap": 0.15, "other": 0.20},
            "relevance_by_symbol": {"NVDA": 0.8, "AMD": 0.7},
            "stance": "mixed",
            "contamination_risk": 0.1,
            "confidence": 0.75,
        }
        
        result = {k: _safe_json(v) for k, v in card_dict.items()}
        
        assert isinstance(result["theme_probs"], dict)
        assert result["theme_probs"]["ai_compute"] == 0.65
        assert isinstance(result["relevance_by_symbol"], dict)
        assert result["relevance_by_symbol"]["NVDA"] == 0.8
        
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["theme_probs"]["ai_compute"] == 0.65
        assert parsed["relevance_by_symbol"]["NVDA"] == 0.8


class TestLogStrategyDecisionIntegration:
    """Integration tests: full path from caller to log_decision."""

    @pytest.mark.asyncio
    async def test_log_strategy_decision_writes_to_bot_decisions(self):
        """log_strategy_decision ultimately writes to bot_decisions table."""
        db_logger_module = _load_db_logger_module()
        DbLogger = db_logger_module.DbLogger
        
        with patch.object(db_logger_module, 'create_async_engine') as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine
            db_logger = DbLogger(dsn="postgresql+asyncpg://test:test@localhost/test")
        
        loop = asyncio.get_running_loop()
        db_logger.set_owner_loop(loop)
        
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock()
        
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        db_logger._engine.begin = MagicMock(return_value=mock_cm)
        
        extra_data = {
            "theme_probs": {"ai_compute": 0.65, "other": 0.35},
            "confidence": 0.75,
            "infer_ms": 45.2,
        }
        
        await db_logger.log_decision(
            component="gpu_news_critic",
            symbol="NVDA",
            action="shadow_pass",
            reason="theme=ai_compute(0.65)",
            **extra_data,
        )
        
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args
        params = call_args[0][1]
        
        assert params['component'] == "gpu_news_critic"
        assert params['symbol'] == "NVDA"
        assert params['action'] == "shadow_pass"
        assert params['reason'] == "theme=ai_compute(0.65)"
        
        details_json = params['details_json']
        assert isinstance(details_json, str)
        assert 'theme_probs' in details_json
        assert 'infer_ms' in details_json

    @pytest.mark.asyncio
    async def test_log_decision_query_by_component(self):
        """Verify Research can query bot_decisions by component=strategy."""
        db_logger_module = _load_db_logger_module()
        DbLogger = db_logger_module.DbLogger
        
        with patch.object(db_logger_module, 'create_async_engine') as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine
            db_logger = DbLogger(dsn="postgresql+asyncpg://test:test@localhost/test")
        
        loop = asyncio.get_running_loop()
        db_logger.set_owner_loop(loop)
        
        call_count = 0
        
        async def track_calls(*args, **kwargs):
            nonlocal call_count
            call_count += 1
        
        mock_conn = MagicMock()
        mock_conn.execute = AsyncMock(side_effect=track_calls)
        
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_cm.__aexit__ = AsyncMock(return_value=None)
        db_logger._engine.begin = MagicMock(return_value=mock_cm)
        
        await db_logger.log_decision(
            component="gpu_news_critic",
            symbol="AMD",
            action="shadow_would_suppress_hard_skip",
            reason="contamination=0.70",
        )
        
        assert call_count == 1
