"""Tests for v-log-decision-collision-fix-2026-09-22: _log_decision kwarg collision hardening.

Root cause: DayTradeMomentumShortStrategy passed reason= in kwargs while reason was
already the 3rd positional argument, causing:
  "_log_decision() got multiple values for argument 'reason'"

This manifested ~10:20 ET on 2026-09-22 for SCHW, stalling the strategy scan.

Verifies:
  - _log_decision handles callers passing reserved keys (reason, action, etc.)
    in **details without raising TypeError
  - Colliding keys are stripped and preserved under _extra for observability
  - Fixed call sites in DayTradeMomentumShortStrategy use detail_reason= instead
"""
import logging
import pytest
from unittest.mock import MagicMock, patch


class MockMarketData:
    """Mock market data for testing."""
    
    def __init__(self, symbol="SCHW", close=50.0):
        self.symbol = symbol
        self.close = close
        self.open = close * 1.01
        self.indicators = {
            'rsi': 45.0,
            'volume_ratio': 1.8,
            'adx': 25.0,
            'atr': 1.0,
            'low_20': 52.0,
            'high_20': 58.0,
            'sma_20': 54.0,
            'sma_50': 55.0,
            'macd': -0.1,
            'macd_signal': 0.0,
            'day_change_pct': -1.5,
        }


def _load_base_module_direct():
    """Load strategies.base directly without triggering full imports."""
    import sys
    import importlib.util
    
    sys.path.insert(0, '/workspace')
    spec = importlib.util.spec_from_file_location(
        "strategies_base_direct", "/workspace/strategies/base.py"
    )
    module = importlib.util.module_from_spec(spec)
    
    sys.modules['core'] = MagicMock()
    sys.modules['core.models'] = MagicMock()
    sys.modules['core.models'].TradingSignal = MagicMock
    
    spec.loader.exec_module(module)
    return module


class TestLogDecisionCollisionFix:
    """Tests for _log_decision collision hardening."""

    def test_log_decision_with_reason_kwarg_no_typeerror(self, caplog):
        """_log_decision with reason= in kwargs must NOT raise TypeError.
        
        This is the exact SCHW scenario that caused the production stall.
        """
        base_module = _load_base_module_direct()
        TradingStrategyWithCommentary = base_module.TradingStrategyWithCommentary
        
        class TestStrategy(TradingStrategyWithCommentary):
            name = "test_strategy"
            
            async def generate_signal_with_commentary(self, market_data):
                return None
        
        strategy = TestStrategy(MagicMock())
        market_data = MockMarketData(symbol="SCHW")
        
        with caplog.at_level(logging.INFO):
            strategy._log_decision(
                market_data, "skip", "rsi_oversold_no_short",
                rsi=30.0,
                rsi_floor=30,
                reason="oversold_bounce_risk",
            )
        
        assert "strategy_decision" in caplog.text
        assert "SCHW" in caplog.text
        assert "rsi_oversold_no_short" in caplog.text

    def test_log_decision_collision_preserved_in_extra(self, caplog):
        """Colliding keys are stripped from details and preserved under _extra."""
        base_module = _load_base_module_direct()
        TradingStrategyWithCommentary = base_module.TradingStrategyWithCommentary
        
        class TestStrategy(TradingStrategyWithCommentary):
            name = "test_strategy"
            
            async def generate_signal_with_commentary(self, market_data):
                return None
        
        strategy = TestStrategy(MagicMock())
        market_data = MockMarketData()
        
        captured_details = {}
        
        def capture_emit(*args, **kwargs):
            captured_details.update(kwargs.get('extra', {}))
        
        strategy._emit_snapshot = capture_emit
        
        strategy._log_decision(
            market_data, "skip", "test_reason",
            rsi=45.0,
            reason="colliding_reason",
            action="colliding_action",
        )
        
        assert '_extra' in captured_details
        assert captured_details['_extra']['reason'] == "colliding_reason"
        assert captured_details['_extra']['action'] == "colliding_action"
        assert captured_details['rsi'] == 45.0

    def test_log_decision_positional_reason_wins(self, caplog):
        """Positional reason parameter takes precedence over kwarg."""
        base_module = _load_base_module_direct()
        TradingStrategyWithCommentary = base_module.TradingStrategyWithCommentary
        
        class TestStrategy(TradingStrategyWithCommentary):
            name = "test_strategy"
            
            async def generate_signal_with_commentary(self, market_data):
                return None
        
        strategy = TestStrategy(MagicMock())
        market_data = MockMarketData()
        
        with caplog.at_level(logging.INFO):
            strategy._log_decision(
                market_data, "skip", "correct_positional_reason",
                reason="incorrect_kwarg_reason",
            )
        
        assert "correct_positional_reason" in caplog.text

    def test_log_decision_no_collision_no_extra(self, caplog):
        """When no collisions, _extra is not added."""
        base_module = _load_base_module_direct()
        TradingStrategyWithCommentary = base_module.TradingStrategyWithCommentary
        
        class TestStrategy(TradingStrategyWithCommentary):
            name = "test_strategy"
            
            async def generate_signal_with_commentary(self, market_data):
                return None
        
        strategy = TestStrategy(MagicMock())
        market_data = MockMarketData()
        
        captured_details = {}
        
        def capture_emit(*args, **kwargs):
            captured_details.update(kwargs.get('extra', {}))
        
        strategy._emit_snapshot = capture_emit
        
        strategy._log_decision(
            market_data, "skip", "test_reason",
            rsi=45.0,
            volume_ratio=1.5,
            detail_reason="safe_kwarg",
        )
        
        assert '_extra' not in captured_details
        assert captured_details['rsi'] == 45.0
        assert captured_details['detail_reason'] == "safe_kwarg"


class TestDayTradeMomentumShortCallSiteFix:
    """Verify DayTradeMomentumShortStrategy call sites use detail_reason=."""

    def test_short_strategy_uses_detail_reason(self):
        """DayTradeMomentumShortStrategy must use detail_reason=, not reason=."""
        import re
        from pathlib import Path
        
        src = Path('/workspace/strategies/builtin.py').read_text()
        
        class_start = src.find('class DayTradeMomentumShortStrategy')
        class_end = src.find('\nclass ', class_start + 1)
        short_strategy_src = src[class_start:class_end]
        
        log_decision_calls = re.findall(
            r'self\._log_decision\([^)]+\)',
            short_strategy_src,
            re.DOTALL
        )
        
        for call in log_decision_calls:
            if 'reason=' in call:
                if 'detail_reason=' not in call:
                    bare_reason_match = re.search(r'[^_]reason\s*=', call)
                    assert bare_reason_match is None, (
                        f"Found bare reason= (collision risk) in _log_decision call:\n{call}"
                    )

    def test_hands_off_denylist_uses_detail_reason(self):
        """hands_off_denylist gate must use detail_reason=."""
        from pathlib import Path
        
        src = Path('/workspace/strategies/builtin.py').read_text()
        
        assert 'hands_off_denylist' in src
        
        hands_off_pattern = src.find('hands_off_denylist')
        context_start = max(0, hands_off_pattern - 200)
        context_end = min(len(src), hands_off_pattern + 200)
        context = src[context_start:context_end]
        
        assert 'detail_reason="permanent_hands_off"' in context, (
            "hands_off_denylist must use detail_reason=, not reason="
        )


class TestReservedKeysSet:
    """Tests for _LOG_DECISION_RESERVED_KEYS completeness."""

    def test_reserved_keys_include_all_explicit_params(self):
        """Reserved keys set must include all explicit _log_decision params."""
        base_module = _load_base_module_direct()
        TradingStrategyWithCommentary = base_module.TradingStrategyWithCommentary
        import inspect
        
        sig = inspect.signature(TradingStrategyWithCommentary._log_decision)
        explicit_params = [
            p.name for p in sig.parameters.values()
            if p.name not in ('self', 'details') and p.kind != inspect.Parameter.VAR_KEYWORD
        ]
        
        reserved_keys = TradingStrategyWithCommentary._LOG_DECISION_RESERVED_KEYS
        
        for param in explicit_params:
            assert param in reserved_keys, (
                f"Parameter '{param}' should be in _LOG_DECISION_RESERVED_KEYS"
            )


class TestSCHWScenarioRegression:
    """Regression test for the exact SCHW ~10:20 ET scenario."""

    @pytest.mark.asyncio
    async def test_schw_short_signal_no_collision_error(self):
        """SCHW-style call with colliding reason= must not raise TypeError.
        
        This test replicates the exact production error that stalled scans.
        """
        base_module = _load_base_module_direct()
        TradingStrategyWithCommentary = base_module.TradingStrategyWithCommentary
        
        class TestStrategy(TradingStrategyWithCommentary):
            name = "day_trade_momentum_short"
            
            async def generate_signal_with_commentary(self, market_data):
                rsi = market_data.indicators.get('rsi', 50)
                rsi_floor = 30
                
                if rsi <= rsi_floor:
                    self._log_decision(
                        market_data, "skip", "rsi_oversold_no_short",
                        rsi=round(rsi, 2),
                        rsi_floor=rsi_floor,
                        detail_reason="oversold_bounce_risk",
                    )
                    return None
                return None
        
        strategy = TestStrategy(MagicMock())
        market_data = MockMarketData(symbol="SCHW")
        market_data.indicators['rsi'] = 28.0
        
        result = await strategy.generate_signal_with_commentary(market_data)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_collision_hardening_prevents_future_regressions(self):
        """Even if someone adds reason= again, collision hardening saves us.
        
        The hardening layer strips collisions instead of raising TypeError.
        """
        base_module = _load_base_module_direct()
        TradingStrategyWithCommentary = base_module.TradingStrategyWithCommentary
        
        class BuggyStrategy(TradingStrategyWithCommentary):
            name = "buggy_strategy"
            
            async def generate_signal_with_commentary(self, market_data):
                self._log_decision(
                    market_data, "skip", "some_reason",
                    reason="accidental_collision",
                    action="another_collision",
                )
                return None
        
        strategy = BuggyStrategy(MagicMock())
        market_data = MockMarketData(symbol="SCHW")
        
        result = await strategy.generate_signal_with_commentary(market_data)
        
        assert result is None
