"""Tests for unified exit manager.

v-unified-exit-tests-2026-09-24-r2. Tests the unified exit policy for
day-trade positions (SHADOW-ONLY mode):
  - Scale 50% at +1R, move stop to breakeven
  - Trail remainder by ATR
  - Time stop N minutes before flatten
  - Position state keyed by (symbol, entry_time)
  - NO live enforcement (shadow logging only)
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.unified_exit import (
    UnifiedExitManager,
    UnifiedExitAction,
    SCALE_AT_R,
    TRAIL_ATR_MULT,
    _get_hands_off_denylist,
    reset_unified_exit_manager,
)


def _mock_position(
    symbol: str = "TEST",
    entry_price: float = 100.0,
    stop_loss: float = 97.0,
    take_profit: float = 106.0,
    quantity: int = 100,
    side: str = "long",
    managed_by_bot: bool = True,
    strategy: str = "day_trade_momentum",
    original_stop: float = None,
    trailing_stop: float = None,
    entry_time: datetime = None,
) -> MagicMock:
    """Create a mock Position for testing."""
    pos = MagicMock()
    pos.symbol = symbol
    pos.entry_price = entry_price
    pos.stop_loss = stop_loss
    pos.take_profit = take_profit
    pos.quantity = quantity
    pos.side = side
    pos.managed_by_bot = managed_by_bot
    pos.reasoning = {"strategy": strategy}
    pos.original_stop = original_stop or stop_loss
    pos.trailing_stop = trailing_stop
    pos.entry_time = entry_time or datetime(2026, 9, 24, 10, 0, 0)
    return pos


class TestUnifiedExitManagerInit:
    """Tests for UnifiedExitManager initialization."""
    
    def test_from_config_defaults(self):
        """Default initialization via from_config."""
        with patch.dict("os.environ", {}, clear=True):
            reset_unified_exit_manager()
            mgr = UnifiedExitManager.from_config()
            assert mgr.enabled is False
    
    def test_from_config_enabled(self):
        """DT_UNIFIED_EXIT=1 enables shadow logging."""
        with patch.dict("os.environ", {"DT_UNIFIED_EXIT": "1"}):
            reset_unified_exit_manager()
            mgr = UnifiedExitManager.from_config()
            assert mgr.enabled is True
    
    def test_from_env_calls_from_config(self):
        """from_env is deprecated alias for from_config."""
        with patch.dict("os.environ", {"DT_UNIFIED_EXIT": "1"}):
            reset_unified_exit_manager()
            mgr = UnifiedExitManager.from_env()
            assert mgr.enabled is True
    
    def test_no_live_enforce_parameter(self):
        """Constructor no longer accepts live_enforce parameter."""
        with pytest.raises(TypeError):
            UnifiedExitManager(live_enforce=True)


class TestUnifiedExitEvaluate:
    """Tests for evaluate_position method."""
    
    def setup_method(self):
        self.tmpdir = TemporaryDirectory()
        self.log_path = Path(self.tmpdir.name) / "shadow.ndjson"
        self.mgr = UnifiedExitManager(
            shadow_log_path=self.log_path,
            enabled=True,
        )
    
    def teardown_method(self):
        self.tmpdir.cleanup()
    
    def test_disabled_returns_none(self):
        """When disabled, returns None."""
        mgr = UnifiedExitManager(enabled=False)
        pos = _mock_position()
        
        result = mgr.evaluate_position(pos, 100.0, 2.0, 10, 0)
        
        assert result is None
    
    def test_hands_off_symbol_returns_none(self):
        """Hands-off symbols return None."""
        hands_off = _get_hands_off_denylist()
        for symbol in hands_off:
            pos = _mock_position(symbol=symbol)
            result = self.mgr.evaluate_position(pos, 100.0, 2.0, 10, 0)
            assert result is None
    
    def test_non_day_trade_returns_none(self):
        """Non-day-trade strategies return None."""
        pos = _mock_position(strategy="swing_trade")
        result = self.mgr.evaluate_position(pos, 100.0, 2.0, 10, 0)
        assert result is None
    
    def test_unmanaged_position_returns_none(self):
        """Unmanaged positions return None."""
        pos = _mock_position(managed_by_bot=False)
        result = self.mgr.evaluate_position(pos, 100.0, 2.0, 10, 0)
        assert result is None
    
    def test_short_position_returns_none(self):
        """Short positions return None (long-only for now)."""
        pos = _mock_position(side="short")
        result = self.mgr.evaluate_position(pos, 100.0, 2.0, 10, 0)
        assert result is None
    
    def test_hold_within_thesis(self):
        """Returns 'hold' when price is within thesis bounds."""
        pos = _mock_position(entry_price=100, stop_loss=97, take_profit=106)
        
        result = self.mgr.evaluate_position(pos, 101.0, 2.0, 10, 0)
        
        assert result is not None
        assert result.action == "hold"
        assert result.reason == "within_thesis"
    
    def test_scale_at_1r(self):
        """Returns 'scale_partial' at +1R."""
        pos = _mock_position(entry_price=100, stop_loss=97, take_profit=106)
        
        result = self.mgr.evaluate_position(pos, 103.5, 2.0, 10, 0)
        
        assert result is not None
        assert result.action == "scale_partial"
        assert "scale_50pct" in result.reason
        assert result.recommended_stop == 100.0  # breakeven
        assert result.recommended_action_qty == 50
    
    def test_scale_only_once(self):
        """Scale happens only once, then switches to trail."""
        pos = _mock_position(entry_price=100, stop_loss=97, take_profit=106)
        
        self.mgr.evaluate_position(pos, 103.5, 2.0, 10, 0)
        result = self.mgr.evaluate_position(pos, 104.0, 2.0, 10, 5)
        
        assert result.action != "scale_partial"
        assert result.scaled_out is True
    
    def test_trail_after_scale(self):
        """Trail stop ratchets after scale."""
        pos = _mock_position(entry_price=100, stop_loss=97, take_profit=106)
        pos.trailing_stop = 100.0
        
        self.mgr.evaluate_position(pos, 103.5, 2.0, 10, 0)
        result = self.mgr.evaluate_position(pos, 105.0, 2.0, 10, 5)
        
        assert result.trailing_active is True
        if result.action == "trail_stop":
            expected_trail = 105.0 - (TRAIL_ATR_MULT * 2.0)
            assert result.recommended_stop == pytest.approx(expected_trail, rel=0.01)
    
    def test_target_exit(self):
        """Returns 'target_exit' when price hits target."""
        pos = _mock_position(entry_price=100, stop_loss=97, take_profit=106)
        
        result = self.mgr.evaluate_position(pos, 107.0, 2.0, 10, 0)
        
        assert result.action == "target_exit"
        assert result.reason == "target_reached"
    
    def test_stop_exit(self):
        """Returns 'stop_exit' when price hits stop."""
        pos = _mock_position(entry_price=100, stop_loss=97, take_profit=106)
        
        result = self.mgr.evaluate_position(pos, 96.5, 2.0, 10, 0)
        
        assert result.action == "stop_exit"
        assert result.reason == "stop_hit"
    
    def test_time_stop_exit(self):
        """Returns 'time_stop_exit' near flatten time."""
        pos = _mock_position(entry_price=100, stop_loss=97, take_profit=106)
        
        result = self.mgr.evaluate_position(pos, 101.0, 2.0, 14, 50)
        
        assert result.action == "time_stop_exit"
        assert "time_stop" in result.reason
    
    def test_shadow_log_written_on_action_change(self):
        """Shadow log is written when action changes."""
        pos = _mock_position()
        
        self.mgr.evaluate_position(pos, 101.0, 2.0, 10, 0)
        
        assert self.log_path.exists()
        with open(self.log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1
    
    def test_shadow_log_not_written_on_same_action(self):
        """Shadow log is NOT written when action stays the same."""
        pos = _mock_position()
        
        self.mgr.evaluate_position(pos, 101.0, 2.0, 10, 0)
        self.mgr.evaluate_position(pos, 101.1, 2.0, 10, 1)
        self.mgr.evaluate_position(pos, 101.2, 2.0, 10, 2)
        
        with open(self.log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1, "Should only log once when action doesn't change"
    
    def test_entry_time_in_action(self):
        """Action includes entry_time for state keying."""
        entry_time = datetime(2026, 9, 24, 10, 30, 0)
        pos = _mock_position(entry_time=entry_time)
        
        result = self.mgr.evaluate_position(pos, 101.0, 2.0, 10, 45)
        
        assert result.entry_time == entry_time.isoformat()


class TestUnifiedExitOverride:
    """Tests for proactive exit override logic (now always returns False)."""
    
    def setup_method(self):
        self.tmpdir = TemporaryDirectory()
        self.log_path = Path(self.tmpdir.name) / "shadow.ndjson"
    
    def teardown_method(self):
        self.tmpdir.cleanup()
    
    def test_override_always_false(self):
        """Override is ALWAYS False — shadow-only mode."""
        mgr = UnifiedExitManager(
            shadow_log_path=self.log_path,
            enabled=True,
        )
        
        action = UnifiedExitAction(
            symbol="TEST",
            timestamp=datetime.now(timezone.utc).isoformat(),
            action="hold",
            reason="within_thesis",
            current_price=101.0,
            entry_price=100.0,
            stop_distance=3.0,
            r_so_far=0.33,
        )
        
        result = mgr.should_override_proactive_exit(action, "proactive_macd_flipped_bearish")
        
        assert result is False, "Override must always be False in shadow-only mode"
    
    def test_override_false_for_all_actions(self):
        """Override is False regardless of action type."""
        mgr = UnifiedExitManager(
            shadow_log_path=self.log_path,
            enabled=True,
        )
        
        for action_type in ["hold", "trail_stop", "scale_partial", "target_exit", "stop_exit"]:
            action = UnifiedExitAction(
                symbol="TEST",
                timestamp=datetime.now(timezone.utc).isoformat(),
                action=action_type,
                reason="test_reason",
                current_price=101.0,
                entry_price=100.0,
                stop_distance=3.0,
                r_so_far=0.33,
            )
            
            result = mgr.should_override_proactive_exit(action, "proactive_macd_flipped_bearish")
            
            assert result is False


class TestPositionStateManagement:
    """Tests for position state tracking (keyed by symbol + entry_time)."""
    
    def setup_method(self):
        self.tmpdir = TemporaryDirectory()
        self.log_path = Path(self.tmpdir.name) / "shadow.ndjson"
        self.mgr = UnifiedExitManager(
            shadow_log_path=self.log_path,
            enabled=True,
        )
    
    def teardown_method(self):
        self.tmpdir.cleanup()
    
    def test_state_persists_between_evaluations(self):
        """Position state persists between evaluation calls."""
        pos = _mock_position(entry_price=100, stop_loss=97, take_profit=106)
        
        self.mgr.evaluate_position(pos, 103.5, 2.0, 10, 0)
        result = self.mgr.evaluate_position(pos, 104.0, 2.0, 10, 5)
        
        assert result.scaled_out is True
    
    def test_close_position_resets_state(self):
        """close_position resets tracking state."""
        entry_time = datetime(2026, 9, 24, 10, 0, 0)
        pos = _mock_position(symbol="RESET_TEST", entry_price=100, stop_loss=97, entry_time=entry_time)
        
        self.mgr.evaluate_position(pos, 103.5, 2.0, 10, 0)
        key = self.mgr._get_position_key("RESET_TEST", entry_time)
        assert key in self.mgr._position_state
        
        self.mgr.close_position("RESET_TEST", entry_time)
        
        assert key not in self.mgr._position_state
    
    def test_different_entry_times_tracked_separately(self):
        """Same symbol with different entry times tracked separately."""
        entry_time_1 = datetime(2026, 9, 24, 9, 35, 0)
        entry_time_2 = datetime(2026, 9, 24, 11, 0, 0)
        
        pos1 = _mock_position(symbol="AAPL", entry_price=100, stop_loss=97, entry_time=entry_time_1)
        pos2 = _mock_position(symbol="AAPL", entry_price=105, stop_loss=102, entry_time=entry_time_2)
        
        self.mgr.evaluate_position(pos1, 103.5, 2.0, 10, 0)
        result2 = self.mgr.evaluate_position(pos2, 106.0, 2.0, 11, 5)
        
        key1 = self.mgr._get_position_key("AAPL", entry_time_1)
        key2 = self.mgr._get_position_key("AAPL", entry_time_2)
        
        assert key1 in self.mgr._position_state
        assert key2 in self.mgr._position_state
        assert key1 != key2
        
        state1 = self.mgr._position_state[key1]
        state2 = self.mgr._position_state[key2]
        assert state1["scaled_out"] is True
        assert state2["scaled_out"] is False
    
    def test_highest_price_tracked(self):
        """Highest price is tracked for trailing stop."""
        pos = _mock_position(entry_price=100, stop_loss=97)
        
        self.mgr.evaluate_position(pos, 102.0, 2.0, 10, 0)
        self.mgr.evaluate_position(pos, 105.0, 2.0, 10, 5)
        self.mgr.evaluate_position(pos, 103.0, 2.0, 10, 10)
        
        key = self.mgr._get_position_key("TEST", pos.entry_time)
        state = self.mgr._position_state.get(key)
        assert state is not None
        assert state["highest_price"] == 105.0


class TestConfigIntegration:
    """Tests for Config integration."""
    
    def test_config_flags_exist(self):
        """Config properties for unified exit exist."""
        try:
            from core.config import Config
            cfg = Config()
            
            assert hasattr(cfg, "DT_UNIFIED_EXIT")
            assert hasattr(cfg, "DT_UNIFIED_EXIT_LIVE_ENFORCE")
        except ImportError:
            pytest.skip("Config import requires full dependencies")
    
    def test_live_enforce_always_false(self):
        """DT_UNIFIED_EXIT_LIVE_ENFORCE always returns False (shadow-only)."""
        try:
            from core.config import Config
            cfg = Config()
            
            assert cfg.DT_UNIFIED_EXIT_LIVE_ENFORCE is False
        except ImportError:
            pytest.skip("Config import requires full dependencies")
    
    def test_config_defaults(self):
        """Config defaults are False (shadow-only)."""
        try:
            with patch.dict("os.environ", {}, clear=True):
                from core.config import Config
                cfg = Config()
                
                assert cfg.DT_UNIFIED_EXIT is False
        except ImportError:
            pytest.skip("Config import requires full dependencies")


class TestEngineHooks:
    """Tests for engine integration hooks."""
    
    def setup_method(self):
        self.tmpdir = TemporaryDirectory()
        self.log_path = Path(self.tmpdir.name) / "shadow.ndjson"
        self.mgr = UnifiedExitManager(
            shadow_log_path=self.log_path,
            enabled=True,
        )
    
    def teardown_method(self):
        self.tmpdir.cleanup()
    
    def test_evaluate_swallows_exceptions(self):
        """evaluate_position handles exceptions gracefully."""
        bad_pos = MagicMock()
        bad_pos.symbol = "TEST"
        bad_pos.reasoning = None  # Will cause AttributeError on .get()
        bad_pos.managed_by_bot = True
        bad_pos.side = "long"
        bad_pos.entry_time = datetime(2026, 9, 24, 10, 0, 0)
        
        result = self.mgr.evaluate_position(bad_pos, 100.0, 2.0, 10, 0)
        
        assert result is None
    
    def test_no_close_or_transition_calls(self):
        """Manager never directly closes positions or transitions state."""
        pos = _mock_position()
        
        result = self.mgr.evaluate_position(pos, 96.0, 2.0, 10, 0)
        
        pos.close.assert_not_called() if hasattr(pos, 'close') else None
        assert hasattr(pos, 'close') is False or not pos.close.called
    
    def test_evaluate_returns_action_not_order(self):
        """evaluate_position returns UnifiedExitAction, not an order."""
        pos = _mock_position()
        
        result = self.mgr.evaluate_position(pos, 103.5, 2.0, 10, 0)
        
        assert isinstance(result, UnifiedExitAction)
        assert not hasattr(result, 'execute')
        assert not hasattr(result, 'submit')


class TestAbsolutePaths:
    """Tests for absolute path handling."""
    
    def test_default_shadow_log_is_absolute(self):
        """Default shadow log path is absolute."""
        from core.unified_exit import _get_shadow_log_path
        
        path = _get_shadow_log_path()
        assert path.is_absolute()
    
    def test_shadow_log_under_data_dir(self):
        """Shadow log is under data/ directory."""
        from core.unified_exit import _get_shadow_log_path
        
        path = _get_shadow_log_path()
        assert "data" in str(path)
        assert path.name == "shadow_unified_exit.ndjson"
