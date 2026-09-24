"""Tests for unified exit manager.

v-unified-exit-tests-2026-09-24. Tests the unified exit policy for
day-trade positions:
  - Scale 50% at +1R, move stop to breakeven
  - Trail remainder by ATR
  - Time stop N minutes before flatten
  - Override of proactive indicator exits
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root is in path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from core.unified_exit import (
    UnifiedExitManager,
    UnifiedExitAction,
    HANDS_OFF_SYMBOLS,
    SCALE_AT_R,
    TRAIL_ATR_MULT,
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
    return pos


class TestUnifiedExitManagerInit:
    """Tests for UnifiedExitManager initialization."""
    
    def test_from_env_defaults(self):
        """Default initialization has shadow disabled."""
        with patch.dict("os.environ", {}, clear=True):
            mgr = UnifiedExitManager.from_env()
            assert mgr.enabled is False
            assert mgr.live_enforce is False
    
    def test_from_env_enabled(self):
        """DT_UNIFIED_EXIT=1 enables shadow logging."""
        with patch.dict("os.environ", {"DT_UNIFIED_EXIT": "1"}):
            mgr = UnifiedExitManager.from_env()
            assert mgr.enabled is True
            assert mgr.live_enforce is False
    
    def test_from_env_live_enforce(self):
        """DT_UNIFIED_EXIT_LIVE_ENFORCE=1 enables live enforcement."""
        with patch.dict("os.environ", {
            "DT_UNIFIED_EXIT": "1",
            "DT_UNIFIED_EXIT_LIVE_ENFORCE": "1",
        }):
            mgr = UnifiedExitManager.from_env()
            assert mgr.enabled is True
            assert mgr.live_enforce is True


class TestUnifiedExitEvaluate:
    """Tests for evaluate_position method."""
    
    def setup_method(self):
        self.tmpdir = TemporaryDirectory()
        self.log_path = Path(self.tmpdir.name) / "shadow.ndjson"
        self.mgr = UnifiedExitManager(
            shadow_log_path=self.log_path,
            enabled=True,
            live_enforce=False,
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
        for symbol in HANDS_OFF_SYMBOLS:
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
    
    def test_shadow_log_written(self):
        """Shadow log is written on each evaluation."""
        pos = _mock_position()
        
        self.mgr.evaluate_position(pos, 101.0, 2.0, 10, 0)
        
        assert self.log_path.exists()
        with open(self.log_path) as f:
            lines = f.readlines()
        assert len(lines) == 1


class TestUnifiedExitOverride:
    """Tests for proactive exit override logic."""
    
    def setup_method(self):
        self.tmpdir = TemporaryDirectory()
        self.log_path = Path(self.tmpdir.name) / "shadow.ndjson"
    
    def teardown_method(self):
        self.tmpdir.cleanup()
    
    def test_no_override_without_enforce(self):
        """Override is False when live_enforce is False."""
        mgr = UnifiedExitManager(
            shadow_log_path=self.log_path,
            enabled=True,
            live_enforce=False,
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
        
        assert result is False
    
    def test_override_with_enforce_and_hold(self):
        """Override is True when enforce=True and action='hold'."""
        mgr = UnifiedExitManager(
            shadow_log_path=self.log_path,
            enabled=True,
            live_enforce=True,
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
        
        assert result is True
    
    def test_no_override_for_target_exit(self):
        """No override when unified says exit at target."""
        mgr = UnifiedExitManager(
            shadow_log_path=self.log_path,
            enabled=True,
            live_enforce=True,
        )
        
        action = UnifiedExitAction(
            symbol="TEST",
            timestamp=datetime.now(timezone.utc).isoformat(),
            action="target_exit",
            reason="target_reached",
            current_price=107.0,
            entry_price=100.0,
            stop_distance=3.0,
            r_so_far=2.33,
        )
        
        result = mgr.should_override_proactive_exit(action, "proactive_macd_flipped_bearish")
        
        assert result is False
    
    def test_no_override_for_non_indicator_exits(self):
        """No override for non-indicator exit reasons."""
        mgr = UnifiedExitManager(
            shadow_log_path=self.log_path,
            enabled=True,
            live_enforce=True,
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
        
        result = mgr.should_override_proactive_exit(action, "stop_loss")
        
        assert result is False
    
    def test_override_allows_trail_action(self):
        """Override allows trail_stop action (still holding position)."""
        mgr = UnifiedExitManager(
            shadow_log_path=self.log_path,
            enabled=True,
            live_enforce=True,
        )
        
        action = UnifiedExitAction(
            symbol="TEST",
            timestamp=datetime.now(timezone.utc).isoformat(),
            action="trail_stop",
            reason="atr_trail_ratchet",
            current_price=105.0,
            entry_price=100.0,
            stop_distance=3.0,
            r_so_far=1.67,
            recommended_stop=102.0,
        )
        
        result = mgr.should_override_proactive_exit(action, "proactive_rsi_below_50")
        
        assert result is True


class TestPositionStateManagement:
    """Tests for position state tracking."""
    
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
        pos = _mock_position(symbol="RESET_TEST", entry_price=100, stop_loss=97)
        
        self.mgr.evaluate_position(pos, 103.5, 2.0, 10, 0)
        state_before = self.mgr._position_state.get("RESET_TEST", {})
        
        self.mgr.close_position("RESET_TEST")
        
        assert "RESET_TEST" not in self.mgr._position_state
    
    def test_highest_price_tracked(self):
        """Highest price is tracked for trailing stop."""
        pos = _mock_position(entry_price=100, stop_loss=97)
        
        self.mgr.evaluate_position(pos, 102.0, 2.0, 10, 0)
        self.mgr.evaluate_position(pos, 105.0, 2.0, 10, 5)
        self.mgr.evaluate_position(pos, 103.0, 2.0, 10, 10)
        
        state = self.mgr._position_state.get("TEST")
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
    
    def test_config_defaults(self):
        """Config defaults are False (shadow-only)."""
        try:
            with patch.dict("os.environ", {}, clear=True):
                from core.config import Config
                cfg = Config()
                
                assert cfg.DT_UNIFIED_EXIT is False
                assert cfg.DT_UNIFIED_EXIT_LIVE_ENFORCE is False
        except ImportError:
            pytest.skip("Config import requires full dependencies")
