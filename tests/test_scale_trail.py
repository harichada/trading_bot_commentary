"""Tests for 1R partial exit + ATR trailing stop."""
from __future__ import annotations

from datetime import datetime

import pytest

from core.models import Position
from analysis.scale_trail_manager import ScaleTrailManager


def _pos(
    entry: float = 100.0,
    stop: float = 97.0,
    tp: float = 106.0,
    qty: int = 100,
    side: str = "long",
    scaled_out: bool = False,
    trailing_stop: float | None = None,
) -> Position:
    return Position(
        symbol="TEST",
        entry_price=entry,
        current_price=entry,
        quantity=qty,
        side=side,
        stop_loss=stop,
        take_profit=tp,
        entry_time=datetime.now(),
        original_stop=stop,
        scaled_out=scaled_out,
        trailing_stop=trailing_stop,
    )


class TestPartialExit:
    mgr = ScaleTrailManager()

    def test_long_partial_fires_at_1R(self):
        """Entry 100, stop 97 → stop_dist=3. At 103 (1R), partial fires."""
        pos = _pos(entry=100, stop=97)
        action = self.mgr.check_partial_exit(pos, current_price=103.0)

        assert action is not None
        assert action.exit_qty == 50
        assert action.new_stop == 100.0  # breakeven
        assert action.exit_fraction == 0.5

    def test_long_no_partial_below_1R(self):
        pos = _pos(entry=100, stop=97)
        assert self.mgr.check_partial_exit(pos, current_price=102.9) is None

    def test_short_partial_fires_at_1R(self):
        """Entry 100, stop 103 → stop_dist=3. At 97 (1R down), partial fires."""
        pos = _pos(entry=100, stop=103, tp=94, side="short")
        action = self.mgr.check_partial_exit(pos, current_price=97.0)

        assert action is not None
        assert action.exit_qty == 50
        assert action.new_stop == 100.0  # breakeven

    def test_short_no_partial_above_1R(self):
        pos = _pos(entry=100, stop=103, tp=94, side="short")
        assert self.mgr.check_partial_exit(pos, current_price=97.1) is None

    def test_double_fire_blocked(self):
        """scaled_out=True must prevent a second partial."""
        pos = _pos(scaled_out=True)
        assert self.mgr.check_partial_exit(pos, current_price=999.0) is None

    def test_single_share_cannot_split(self):
        """int(1 * 0.5) = 0 → no partial possible."""
        pos = _pos(qty=1)
        assert self.mgr.check_partial_exit(pos, current_price=999.0) is None

    def test_zero_stop_distance_returns_none(self):
        """If stop == entry, stop_distance is 0 → skip."""
        pos = _pos(entry=100, stop=100)
        assert self.mgr.check_partial_exit(pos, current_price=105.0) is None

    def test_original_stop_fallback(self):
        """When original_stop is None, falls back to stop_loss."""
        pos = _pos(entry=100, stop=97)
        pos.original_stop = None
        action = self.mgr.check_partial_exit(pos, current_price=103.0)
        assert action is not None


class TestTrailingStop:
    mgr = ScaleTrailManager()

    def test_long_not_activated_below_1_atr(self):
        pos = _pos(entry=100)
        assert self.mgr.update_trailing_stop(pos, current_price=102.9, atr=3.0) is None

    def test_long_activates_at_1_atr(self):
        """Entry 100, ATR=3. At 103 (1×ATR), trail activates at 103-3=100."""
        pos = _pos(entry=100)
        trail = self.mgr.update_trailing_stop(pos, current_price=103.0, atr=3.0)
        assert trail == pytest.approx(100.0)

    def test_long_trail_ratchets_up(self):
        """Trail at 100, price moves to 106 → new trail 103 (better)."""
        pos = _pos(entry=100, trailing_stop=100.0)
        trail = self.mgr.update_trailing_stop(pos, current_price=106.0, atr=3.0)
        assert trail == pytest.approx(103.0)

    def test_long_trail_never_moves_down(self):
        """Trail at 103, price dips to 104 → new trail 101 (worse) → None."""
        pos = _pos(entry=100, trailing_stop=103.0)
        assert self.mgr.update_trailing_stop(pos, current_price=104.0, atr=3.0) is None

    def test_short_activates_at_1_atr(self):
        pos = _pos(entry=100, stop=103, side="short")
        trail = self.mgr.update_trailing_stop(pos, current_price=97.0, atr=3.0)
        assert trail == pytest.approx(100.0)

    def test_short_trail_ratchets_down(self):
        pos = _pos(entry=100, stop=103, side="short", trailing_stop=100.0)
        trail = self.mgr.update_trailing_stop(pos, current_price=94.0, atr=3.0)
        assert trail == pytest.approx(97.0)

    def test_short_trail_never_moves_up(self):
        pos = _pos(entry=100, stop=103, side="short", trailing_stop=97.0)
        assert self.mgr.update_trailing_stop(pos, current_price=96.0, atr=3.0) is None

    def test_zero_atr_returns_none(self):
        pos = _pos(entry=100)
        assert self.mgr.update_trailing_stop(pos, current_price=110.0, atr=0.0) is None


class TestTrailingStopHit:
    mgr = ScaleTrailManager()

    def test_long_trail_hit(self):
        pos = _pos(trailing_stop=102.0)
        assert self.mgr.is_trailing_stop_hit(pos, current_price=101.5) is True

    def test_long_trail_not_hit(self):
        pos = _pos(trailing_stop=102.0)
        assert self.mgr.is_trailing_stop_hit(pos, current_price=103.0) is False

    def test_short_trail_hit(self):
        pos = _pos(side="short", trailing_stop=98.0)
        assert self.mgr.is_trailing_stop_hit(pos, current_price=98.5) is True

    def test_no_trail_set_returns_false(self):
        pos = _pos()
        assert self.mgr.is_trailing_stop_hit(pos, current_price=50.0) is False


class TestPositionDefaults:
    def test_new_fields_default_correctly(self):
        pos = Position(
            symbol="X", entry_price=10, current_price=10,
            quantity=1, side="long", stop_loss=9, take_profit=12,
            entry_time=datetime.now(),
        )
        assert pos.scaled_out is False
        assert pos.original_stop is None
        assert pos.trailing_stop is None
