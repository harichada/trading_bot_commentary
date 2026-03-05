"""Tests for Classic Gap Fade Strategy — Profit Protection (v1.1).

Covers:
- Breakeven stop (short + long)
- Trailing stop (short + long)
- Profit drawdown exit (short + long)
- Edge cases (zero entry, zero shares, boundary values)
- Config customization
- Engine defaults passthrough (unchanged behavior)
- Full lifecycle scenarios (multi-tick simulations)
"""

import pytest
from datetime import datetime

from gap_fade_strategies import GapFadeStrategyRegistry
from gap_fade_strategies.classic_gap_fade import ClassicGapFadeStrategy, _DEFAULTS
from gap_fade_strategies.base import ExitSignal


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def strategy():
    """Fresh classic strategy with default config."""
    return GapFadeStrategyRegistry.create_strategy('classic_gap_fade')


@pytest.fixture
def custom_strategy():
    """Strategy with custom config for tighter thresholds."""
    return GapFadeStrategyRegistry.create_strategy('classic_gap_fade', config={
        'breakeven_trigger_pct': 0.003,
        'trail_trigger_pct': 0.008,
        'trail_offset_pct': 0.002,
        'profit_protect_trigger_pct': 0.01,
        'profit_drawdown_pct': 0.40,
    })


def _short_pos(entry=100.0, stop=101.5, hwm_pnl=0.0, hwm_price=0.0,
               remaining=100, partial=False):
    """Helper to build a short position dict."""
    return {
        'direction': 'short',
        'entry_price': entry,
        'stop_price': stop,
        'high_water_pnl_pct': hwm_pnl,
        'high_water_price': hwm_price,
        'remaining_shares': remaining,
        'partial_filled': partial,
    }


def _long_pos(entry=100.0, stop=98.5, hwm_pnl=0.0, hwm_price=0.0,
              remaining=100, partial=False):
    """Helper to build a long position dict."""
    return {
        'direction': 'long',
        'entry_price': entry,
        'stop_price': stop,
        'high_water_pnl_pct': hwm_pnl,
        'high_water_price': hwm_price,
        'remaining_shares': remaining,
        'partial_filled': partial,
    }


# ═══════════════════════════════════════════════════════════════════
# Registration & Config
# ═══════════════════════════════════════════════════════════════════

class TestRegistration:
    def test_registered_in_registry(self):
        strategies = GapFadeStrategyRegistry.list_strategies()
        ids = [s['id'] for s in strategies]
        assert 'classic_gap_fade' in ids

    def test_create_returns_correct_class(self):
        s = GapFadeStrategyRegistry.create_strategy('classic_gap_fade')
        assert isinstance(s, ClassicGapFadeStrategy)

    def test_version_upgraded(self, strategy):
        assert strategy.version == '1.1'

    def test_name(self, strategy):
        assert strategy.name == 'Classic Gap Fade'


class TestDefaultConfig:
    def test_has_all_keys(self, strategy):
        for key in _DEFAULTS:
            assert key in strategy.config, f'Missing config key: {key}'

    def test_default_values(self, strategy):
        assert strategy.config['breakeven_trigger_pct'] == 0.005
        assert strategy.config['trail_trigger_pct'] == 0.01
        assert strategy.config['trail_offset_pct'] == 0.004
        assert strategy.config['profit_protect_trigger_pct'] == 0.015
        assert strategy.config['profit_drawdown_pct'] == 0.50

    def test_custom_config_overrides(self, custom_strategy):
        assert custom_strategy.config['breakeven_trigger_pct'] == 0.003
        assert custom_strategy.config['trail_trigger_pct'] == 0.008
        assert custom_strategy.config['profit_drawdown_pct'] == 0.40

    def test_update_config(self, strategy):
        strategy.update_config({'trail_offset_pct': 0.01})
        assert strategy.config['trail_offset_pct'] == 0.01
        # Other values unchanged
        assert strategy.config['breakeven_trigger_pct'] == 0.005


class TestParameterSchema:
    def test_has_all_params(self, strategy):
        schema = strategy.get_parameter_schema()
        assert set(schema.keys()) == set(_DEFAULTS.keys())

    def test_schema_fields(self, strategy):
        schema = strategy.get_parameter_schema()
        for key, spec in schema.items():
            assert 'type' in spec
            assert 'label' in spec
            assert 'default' in spec
            assert 'description' in spec
            assert spec['type'] == 'float'
            assert spec['default'] == _DEFAULTS[key]


# ═══════════════════════════════════════════════════════════════════
# Engine Defaults Passthrough (unchanged behavior)
# ═══════════════════════════════════════════════════════════════════

class TestPassthrough:
    """Verify all entry/target/filter methods still return defaults."""

    def test_filter_candidate(self, strategy):
        assert strategy.filter_candidate({}) == (True, '')

    def test_score_candidate(self, strategy):
        assert strategy.score_candidate({}) is None

    def test_get_entry_window(self, strategy):
        assert strategy.get_entry_window() is None

    def test_should_enter_now(self, strategy):
        assert strategy.should_enter_now({}, 100.0) == (True, '')

    def test_compute_stop_price(self, strategy):
        assert strategy.compute_stop_price(100.0, {}) is None

    def test_compute_targets(self, strategy):
        assert strategy.compute_targets(100.0, {}) is None

    def test_get_required_indicators(self, strategy):
        assert strategy.get_required_indicators() == []

    def test_get_llm_system_prompt(self, strategy):
        assert strategy.get_llm_system_prompt() is None

    def test_get_llm_profit_prompt(self, strategy):
        assert strategy.get_llm_profit_prompt() is None


# ═══════════════════════════════════════════════════════════════════
# Breakeven Stop — Shorts
# ═══════════════════════════════════════════════════════════════════

class TestBreakevenShort:
    def test_no_trigger_below_threshold(self, strategy):
        """Stop should NOT move when profit hasn't reached breakeven trigger."""
        pos = _short_pos(hwm_pnl=0.004)  # 0.4% < 0.5% trigger
        assert strategy.update_trailing_stop(pos, 99.6) is None

    def test_trigger_at_exact_threshold(self, strategy):
        """Stop should move when profit exactly equals threshold."""
        pos = _short_pos(hwm_pnl=0.005, hwm_price=99.5)  # exactly 0.5%
        result = strategy.update_trailing_stop(pos, 99.5)
        assert result == 100.0  # breakeven = entry price

    def test_trigger_above_threshold(self, strategy):
        """Stop should move to breakeven when profit exceeds threshold."""
        pos = _short_pos(hwm_pnl=0.008, hwm_price=99.2)  # 0.8% > 0.5%
        result = strategy.update_trailing_stop(pos, 99.5)
        assert result == 100.0

    def test_stop_already_at_breakeven(self, strategy):
        """No change when stop is already at entry price."""
        pos = _short_pos(stop=100.0, hwm_pnl=0.008, hwm_price=99.2)
        result = strategy.update_trailing_stop(pos, 99.5)
        assert result is None

    def test_stop_already_tighter_than_breakeven(self, strategy):
        """No change when stop is already below entry (tighter for shorts)."""
        pos = _short_pos(stop=99.8, hwm_pnl=0.008, hwm_price=99.2)
        result = strategy.update_trailing_stop(pos, 99.5)
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# Breakeven Stop — Longs
# ═══════════════════════════════════════════════════════════════════

class TestBreakevenLong:
    def test_no_trigger_below_threshold(self, strategy):
        pos = _long_pos(hwm_pnl=0.004)
        assert strategy.update_trailing_stop(pos, 100.4) is None

    def test_trigger_moves_stop_to_entry(self, strategy):
        pos = _long_pos(hwm_pnl=0.008, hwm_price=100.8)
        result = strategy.update_trailing_stop(pos, 100.5)
        assert result == 100.0  # breakeven = entry

    def test_stop_already_at_breakeven(self, strategy):
        pos = _long_pos(stop=100.0, hwm_pnl=0.008, hwm_price=100.8)
        result = strategy.update_trailing_stop(pos, 100.5)
        assert result is None

    def test_stop_already_tighter(self, strategy):
        """Stop already above entry (tighter for longs)."""
        pos = _long_pos(stop=100.3, hwm_pnl=0.008, hwm_price=100.8)
        result = strategy.update_trailing_stop(pos, 100.5)
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# Trailing Stop — Shorts
# ═══════════════════════════════════════════════════════════════════

class TestTrailingShort:
    def test_no_trail_below_trigger(self, strategy):
        """Trail should not activate when profit below trail_trigger_pct."""
        pos = _short_pos(hwm_pnl=0.008, hwm_price=99.2)  # above breakeven but below trail
        result = strategy.update_trailing_stop(pos, 99.5)
        assert result == 100.0  # only breakeven, not trail

    def test_trail_activates_at_trigger(self, strategy):
        """Trail stop should kick in at 1.0% profit."""
        pos = _short_pos(hwm_pnl=0.01, hwm_price=99.0)
        result = strategy.update_trailing_stop(pos, 99.3)
        expected = round(99.0 * 1.004, 2)  # 99.40
        assert result == expected

    def test_trail_follows_deeper_profit(self, strategy):
        """Trail stop moves with high water as price drops further."""
        pos = _short_pos(hwm_pnl=0.03, hwm_price=97.0)
        result = strategy.update_trailing_stop(pos, 97.5)
        expected = round(97.0 * 1.004, 2)  # 97.39
        assert result == expected

    def test_trail_does_not_widen(self, strategy):
        """Trail stop should NOT move away from price (no widening)."""
        pos = _short_pos(stop=97.39, hwm_pnl=0.03, hwm_price=97.0)
        result = strategy.update_trailing_stop(pos, 97.5)
        assert result is None  # 97.39 not < 97.39

    def test_trail_tighter_than_breakeven(self, strategy):
        """Trail stop should be tighter (lower) than breakeven for shorts."""
        pos = _short_pos(hwm_pnl=0.02, hwm_price=98.0)
        result = strategy.update_trailing_stop(pos, 98.5)
        trail_val = round(98.0 * 1.004, 2)  # 98.39
        assert result == trail_val
        assert result < 100.0  # tighter than breakeven

    def test_trail_with_large_profit(self, strategy):
        """Large profits should trail close to the low."""
        pos = _short_pos(hwm_pnl=0.10, hwm_price=90.0)  # 10% profit
        result = strategy.update_trailing_stop(pos, 91.0)
        expected = round(90.0 * 1.004, 2)  # 90.36
        assert result == expected


# ═══════════════════════════════════════════════════════════════════
# Trailing Stop — Longs
# ═══════════════════════════════════════════════════════════════════

class TestTrailingLong:
    def test_trail_activates_at_trigger(self, strategy):
        pos = _long_pos(hwm_pnl=0.02, hwm_price=102.0)
        result = strategy.update_trailing_stop(pos, 101.5)
        expected = round(102.0 * 0.996, 2)  # 101.59
        assert result == expected

    def test_trail_follows_higher_prices(self, strategy):
        pos = _long_pos(hwm_pnl=0.05, hwm_price=105.0)
        result = strategy.update_trailing_stop(pos, 104.0)
        expected = round(105.0 * 0.996, 2)  # 104.58
        assert result == expected

    def test_trail_does_not_widen(self, strategy):
        """Stop should not move down (wider) for longs."""
        pos = _long_pos(stop=104.58, hwm_pnl=0.05, hwm_price=105.0)
        result = strategy.update_trailing_stop(pos, 104.0)
        assert result is None

    def test_trail_tighter_than_breakeven(self, strategy):
        pos = _long_pos(hwm_pnl=0.03, hwm_price=103.0)
        result = strategy.update_trailing_stop(pos, 102.0)
        trail_val = round(103.0 * 0.996, 2)  # 102.59
        assert result == trail_val
        assert result > 100.0  # tighter than breakeven for longs


# ═══════════════════════════════════════════════════════════════════
# Profit Drawdown Exit — Shorts
# ═══════════════════════════════════════════════════════════════════

class TestDrawdownShort:
    def test_no_exit_below_protect_trigger(self, strategy):
        """No drawdown exit when peak profit is below threshold."""
        pos = _short_pos(hwm_pnl=0.01)  # 1.0% < 1.5% trigger
        assert strategy.evaluate_exit(pos, 99.5, 99.5) is None

    def test_no_exit_when_drawdown_small(self, strategy):
        """No exit when position still has most of its profit."""
        pos = _short_pos(hwm_pnl=0.03)  # peaked at 3%
        # Price = 97.5 -> current pnl = 2.5%, gave back 17%
        assert strategy.evaluate_exit(pos, 97.5, 97.5) is None

    def test_exit_at_50pct_drawdown(self, strategy):
        """Exit when 50% of peak profit is given back."""
        pos = _short_pos(hwm_pnl=0.03)  # peaked at 3%
        # Price = 98.5 -> current pnl = 1.5%, gave back 50%
        result = strategy.evaluate_exit(pos, 98.5, 98.5)
        assert result is not None
        assert isinstance(result, ExitSignal)
        assert result.action == 'close'
        assert 'profit_drawdown' in result.reason
        assert result.shares == 100

    def test_exit_when_fully_reversed(self, strategy):
        """Exit when all profit is gone."""
        pos = _short_pos(hwm_pnl=0.02)  # peaked at 2%
        # Price = 100.0 -> current pnl = 0%, gave back 100%
        result = strategy.evaluate_exit(pos, 100.0, 100.0)
        assert result is not None
        assert result.shares == 100

    def test_exit_when_gone_negative(self, strategy):
        """Exit when position has reversed past entry into a loss."""
        pos = _short_pos(hwm_pnl=0.02)  # peaked at 2%
        # Price = 101.0 -> current pnl = -1%, gave back >100%
        result = strategy.evaluate_exit(pos, 101.0, 101.0)
        assert result is not None
        assert 'profit_drawdown' in result.reason

    def test_reason_contains_percentages(self, strategy):
        """Exit reason should include peak, current, and drawdown %."""
        pos = _short_pos(hwm_pnl=0.04)  # 4% peak
        # Price = 98.0 -> current pnl = 2%, gave back 50%
        result = strategy.evaluate_exit(pos, 98.0, 98.0)
        assert result is not None
        assert 'peak' in result.reason
        assert 'now' in result.reason
        assert 'gave back' in result.reason

    def test_correct_shares_in_signal(self, strategy):
        pos = _short_pos(hwm_pnl=0.03, remaining=42)
        result = strategy.evaluate_exit(pos, 98.5, 98.5)
        assert result is not None
        assert result.shares == 42


# ═══════════════════════════════════════════════════════════════════
# Profit Drawdown Exit — Longs
# ═══════════════════════════════════════════════════════════════════

class TestDrawdownLong:
    def test_no_exit_below_protect_trigger(self, strategy):
        pos = _long_pos(hwm_pnl=0.01)
        assert strategy.evaluate_exit(pos, 100.5, 100.5) is None

    def test_no_exit_when_drawdown_small(self, strategy):
        pos = _long_pos(hwm_pnl=0.03)
        # Price = 102.5 -> current pnl = 2.5%, gave back 17%
        assert strategy.evaluate_exit(pos, 102.5, 102.5) is None

    def test_exit_at_50pct_drawdown(self, strategy):
        pos = _long_pos(hwm_pnl=0.03)
        # Price = 101.5 -> current pnl = 1.5%, gave back 50%
        result = strategy.evaluate_exit(pos, 101.5, 101.5)
        assert result is not None
        assert result.action == 'close'
        assert 'profit_drawdown' in result.reason
        assert result.shares == 100

    def test_exit_when_fully_reversed(self, strategy):
        pos = _long_pos(hwm_pnl=0.02)
        # Price = 100.0 -> current pnl = 0%, gave back 100%
        result = strategy.evaluate_exit(pos, 100.0, 100.0)
        assert result is not None

    def test_exit_when_gone_negative(self, strategy):
        pos = _long_pos(hwm_pnl=0.02)
        # Price = 99.0 -> current pnl = -1%, gave back >100%
        result = strategy.evaluate_exit(pos, 99.0, 99.0)
        assert result is not None


# ═══════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_zero_entry_price_trailing(self, strategy):
        pos = _short_pos(entry=0.0, hwm_pnl=0.01)
        assert strategy.update_trailing_stop(pos, 99.0) is None

    def test_zero_entry_price_exit(self, strategy):
        pos = _short_pos(entry=0.0, hwm_pnl=0.03)
        assert strategy.evaluate_exit(pos, 99.0, 99.0) is None

    def test_zero_remaining_shares(self, strategy):
        pos = _short_pos(hwm_pnl=0.03, remaining=0)
        assert strategy.evaluate_exit(pos, 98.5, 98.5) is None

    def test_zero_high_water_pnl(self, strategy):
        pos = _short_pos(hwm_pnl=0.0)
        assert strategy.update_trailing_stop(pos, 99.0) is None
        assert strategy.evaluate_exit(pos, 99.0, 99.0) is None

    def test_negative_high_water_pnl(self, strategy):
        """Position never went positive — no protection should fire."""
        pos = _short_pos(hwm_pnl=-0.01)
        assert strategy.update_trailing_stop(pos, 101.0) is None
        assert strategy.evaluate_exit(pos, 101.0, 101.0) is None

    def test_zero_high_water_price_trailing(self, strategy):
        """HWM pnl met but HWM price is 0 — breakeven only, no trail."""
        pos = _short_pos(hwm_pnl=0.02, hwm_price=0.0)
        result = strategy.update_trailing_stop(pos, 99.0)
        assert result == 100.0  # breakeven only, trail skipped

    def test_tick_data_ignored(self, strategy):
        """Strategy should work with any tick_data value since it uses position only."""
        pos = _short_pos(hwm_pnl=0.008, hwm_price=99.2)
        r1 = strategy.update_trailing_stop(pos, 99.5, tick_data=None)
        r2 = strategy.update_trailing_stop(pos, 99.5, tick_data={})
        r3 = strategy.update_trailing_stop(pos, 99.5, tick_data={'vwap': 99.0})
        assert r1 == r2 == r3

    def test_now_ignored(self, strategy):
        """Strategy doesn't use time for stop decisions."""
        pos = _short_pos(hwm_pnl=0.008, hwm_price=99.2)
        r1 = strategy.update_trailing_stop(pos, 99.5, now=None)
        r2 = strategy.update_trailing_stop(pos, 99.5,
                                           now=datetime(2026, 2, 27, 10, 30))
        assert r1 == r2

    def test_missing_position_keys_use_defaults(self, strategy):
        """Sparse position dict should not crash."""
        pos = {'entry_price': 100.0}  # missing direction, stop, hwm, etc.
        # Should not raise; defaults to direction='short', hwm_pnl=0
        assert strategy.update_trailing_stop(pos, 99.0) is None

    def test_penny_stock(self, strategy):
        """Low-priced stock: verify rounding works correctly."""
        pos = _short_pos(entry=2.50, stop=2.54, hwm_pnl=0.02, hwm_price=2.45)
        result = strategy.update_trailing_stop(pos, 2.46)
        expected = round(2.45 * 1.004, 2)  # 2.46
        assert result == expected

    def test_high_priced_stock(self, strategy):
        """High-priced stock: verify math works at large values."""
        pos = _short_pos(entry=5000.0, stop=5075.0, hwm_pnl=0.02,
                         hwm_price=4900.0)
        result = strategy.update_trailing_stop(pos, 4920.0)
        expected = round(4900.0 * 1.004, 2)  # 4919.60
        assert result == expected


# ═══════════════════════════════════════════════════════════════════
# Custom Config Thresholds
# ═══════════════════════════════════════════════════════════════════

class TestCustomConfig:
    def test_tighter_breakeven_trigger(self, custom_strategy):
        """Custom 0.3% breakeven trigger should fire earlier."""
        pos = _short_pos(hwm_pnl=0.004, hwm_price=99.6)
        # Default would not trigger (0.4% < 0.5%), but custom 0.3% should
        result = custom_strategy.update_trailing_stop(pos, 99.6)
        assert result == 100.0

    def test_tighter_trail_trigger(self, custom_strategy):
        """Custom 0.8% trail trigger should start trailing earlier."""
        pos = _short_pos(hwm_pnl=0.009, hwm_price=99.1)
        result = custom_strategy.update_trailing_stop(pos, 99.3)
        # Trail offset 0.2%: 99.1 * 1.002 = 99.30
        expected = round(99.1 * 1.002, 2)
        assert result == expected

    def test_tighter_drawdown_exit(self, custom_strategy):
        """Custom 40% drawdown should exit sooner."""
        pos = _short_pos(hwm_pnl=0.02)  # 2% peak (above 1% protect trigger)
        # Current pnl ~1.1%, gave back ~45% (clearly above 40% threshold)
        result = custom_strategy.evaluate_exit(pos, 98.9, 98.9)
        assert result is not None
        assert 'profit_drawdown' in result.reason

    def test_lower_protect_trigger(self, custom_strategy):
        """Custom 1.0% protect trigger should activate earlier."""
        pos = _short_pos(hwm_pnl=0.012)  # 1.2% peak (above 1% custom trigger)
        # give back 50% -> current pnl = 0.6% -> price = 99.4
        result = custom_strategy.evaluate_exit(pos, 99.4, 99.4)
        assert result is not None


# ═══════════════════════════════════════════════════════════════════
# Boundary Values
# ═══════════════════════════════════════════════════════════════════

class TestBoundaries:
    def test_drawdown_exactly_at_threshold(self, strategy):
        """Exactly 50% drawdown should trigger exit."""
        pos = _short_pos(hwm_pnl=0.02)  # 2% peak
        # Need current_pnl = 1.0% (50% given back)
        # Short: pnl = (entry - price) / entry = (100 - 99) / 100 = 0.01
        result = strategy.evaluate_exit(pos, 99.0, 99.0)
        assert result is not None

    def test_drawdown_just_below_threshold(self, strategy):
        """49% drawdown should NOT trigger exit."""
        pos = _short_pos(hwm_pnl=0.02)  # 2% peak
        # Need current_pnl = 1.02% (49% given back)
        # Short: price = 100 * (1 - 0.0102) = 98.98
        result = strategy.evaluate_exit(pos, 98.98, 98.98)
        assert result is None

    def test_breakeven_just_below_trigger(self, strategy):
        """0.499% profit should NOT trigger breakeven."""
        pos = _short_pos(hwm_pnl=0.00499)
        assert strategy.update_trailing_stop(pos, 99.5) is None

    def test_trail_just_below_trigger(self, strategy):
        """0.999% profit should activate breakeven but NOT trail."""
        pos = _short_pos(hwm_pnl=0.00999, hwm_price=99.0)
        result = strategy.update_trailing_stop(pos, 99.3)
        assert result == 100.0  # breakeven only

    def test_protect_trigger_just_below(self, strategy):
        """1.499% peak profit should NOT activate drawdown exit."""
        pos = _short_pos(hwm_pnl=0.01499)
        # Even with 100% drawdown, should not exit
        result = strategy.evaluate_exit(pos, 100.0, 100.0)
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# Full Lifecycle — Multi-Tick Simulation
# ═══════════════════════════════════════════════════════════════════

class TestLifecycleShort:
    """Simulate a short position through its entire lifecycle."""

    def test_short_lifecycle(self, strategy):
        """Short $100, drops to $97, reverses to $99 — protection should fire."""
        pos = _short_pos(entry=100.0, stop=101.5)

        # Tick 1: Price at 100.0 — no profit yet
        pos['high_water_pnl_pct'] = 0.0
        pos['high_water_price'] = 100.0
        assert strategy.update_trailing_stop(pos, 100.0) is None
        assert strategy.evaluate_exit(pos, 100.0, 100.0) is None

        # Tick 2: Price drops to 99.4 — 0.6% profit -> breakeven triggers
        pos['high_water_pnl_pct'] = 0.006
        pos['high_water_price'] = 99.4
        result = strategy.update_trailing_stop(pos, 99.4)
        assert result == 100.0  # breakeven
        pos['stop_price'] = 100.0  # simulate engine applying the stop

        # Tick 3: Price drops to 98.8 — 1.2% profit -> trail triggers
        pos['high_water_pnl_pct'] = 0.012
        pos['high_water_price'] = 98.8
        result = strategy.update_trailing_stop(pos, 98.8)
        expected = round(98.8 * 1.004, 2)  # 99.20
        assert result == expected
        pos['stop_price'] = expected

        # Tick 4: Price drops to 97.0 — 3% profit -> trail tightens
        pos['high_water_pnl_pct'] = 0.03
        pos['high_water_price'] = 97.0
        result = strategy.update_trailing_stop(pos, 97.0)
        expected = round(97.0 * 1.004, 2)  # 97.39
        assert result == expected
        pos['stop_price'] = expected

        # Tick 5: Price still at 97.0 — no change
        assert strategy.update_trailing_stop(pos, 97.0) is None

        # Tick 6: Price reverses to 98.0 — profit now 2%, gave back 33%
        # HWM stays at 3%/97.0 since price is going against us
        result = strategy.evaluate_exit(pos, 98.0, 98.0)
        assert result is None  # only 33% given back < 50%

        # Tick 7: Price reverses to 98.5 — profit now 1.5%, gave back 50%
        result = strategy.evaluate_exit(pos, 98.5, 98.5)
        assert result is not None
        assert 'profit_drawdown' in result.reason

    def test_short_small_profit_no_protection(self, strategy):
        """Short with small profit (0.3%) — no protection should fire."""
        pos = _short_pos(entry=100.0, stop=101.5,
                         hwm_pnl=0.003, hwm_price=99.7)

        # No breakeven, no trail, no drawdown exit
        assert strategy.update_trailing_stop(pos, 99.7) is None
        assert strategy.evaluate_exit(pos, 100.0, 100.0) is None


class TestLifecycleLong:
    """Simulate a long position through its lifecycle."""

    def test_long_lifecycle(self, strategy):
        """Long $100, rises to $103, reverses to $101 — protection should fire."""
        pos = _long_pos(entry=100.0, stop=98.5)

        # Tick 1: Price at 100.0 — no profit
        pos['high_water_pnl_pct'] = 0.0
        pos['high_water_price'] = 100.0
        assert strategy.update_trailing_stop(pos, 100.0) is None

        # Tick 2: Price rises to 100.6 — 0.6% profit -> breakeven
        pos['high_water_pnl_pct'] = 0.006
        pos['high_water_price'] = 100.6
        result = strategy.update_trailing_stop(pos, 100.6)
        assert result == 100.0  # breakeven
        pos['stop_price'] = 100.0

        # Tick 3: Price rises to 101.5 — 1.5% -> trail kicks in
        pos['high_water_pnl_pct'] = 0.015
        pos['high_water_price'] = 101.5
        result = strategy.update_trailing_stop(pos, 101.5)
        expected = round(101.5 * 0.996, 2)  # 101.09
        assert result == expected
        pos['stop_price'] = expected

        # Tick 4: Price rises to 103.0 — 3% -> trail tightens further
        pos['high_water_pnl_pct'] = 0.03
        pos['high_water_price'] = 103.0
        result = strategy.update_trailing_stop(pos, 103.0)
        expected = round(103.0 * 0.996, 2)  # 102.59
        assert result == expected
        pos['stop_price'] = expected

        # Tick 5: Price drops to 101.5 — profit 1.5%, gave back 50%
        result = strategy.evaluate_exit(pos, 101.5, 101.5)
        assert result is not None
        assert 'profit_drawdown' in result.reason


# ═══════════════════════════════════════════════════════════════════
# Interaction Between Trailing Stop and Drawdown Exit
# ═══════════════════════════════════════════════════════════════════

class TestInteraction:
    def test_trailing_stop_fires_before_drawdown(self, strategy):
        """When trailing stop is tight, engine stop-out fires before drawdown exit."""
        pos = _short_pos(entry=100.0, stop=97.39, hwm_pnl=0.03,
                         hwm_price=97.0)
        # Price reverses to 97.4 -> just above trailing stop
        # Engine would catch stop at 97.39
        # Drawdown: current pnl = 2.6%, gave back 13% -> no drawdown exit
        result = strategy.evaluate_exit(pos, 97.4, 97.4)
        assert result is None  # trailing stop (engine) catches this, not drawdown

    def test_drawdown_exit_when_trail_not_tight_enough(self, strategy):
        """Drawdown exit fires when profit reverses too fast for trailing to catch."""
        # Scenario: trail hasn't updated because high_water_price is stale
        pos = _short_pos(entry=100.0, stop=99.20,  # trail from 98.8 high water
                         hwm_pnl=0.04, hwm_price=96.0)  # but actual HWM was 96
        # Price jumps to 98.0 -> current pnl = 2%, gave back 50%
        result = strategy.evaluate_exit(pos, 98.0, 98.0)
        assert result is not None
        assert 'profit_drawdown' in result.reason

    def test_both_methods_none_when_no_profit(self, strategy):
        """When position is flat/losing, both methods return None."""
        pos = _short_pos(entry=100.0, stop=101.5, hwm_pnl=0.0)
        assert strategy.update_trailing_stop(pos, 100.5) is None
        assert strategy.evaluate_exit(pos, 100.5, 100.5) is None
