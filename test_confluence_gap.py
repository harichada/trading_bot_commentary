#!/usr/bin/env python3
"""
Test Suite for Confluence Gap Strategy
=======================================
Comprehensive tests for all confluence gap strategy components:
key level detection, Bollinger Bands, Fibonacci, ATR, confidence scoring,
trend confirmation, entry/exit logic, stops, targets, and indicator engine.

Run with: python -m pytest test_confluence_gap.py -v
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from gap_fade_strategies.confluence_gap import (
    ConfluenceGapStrategy,
    ConfluenceAnalysis,
    KeyLevel,
    PendingSignal,
)
from gap_fade_strategies.base import ExitSignal
from gap_fade_strategies.indicators import FiveMinBar, TickIndicatorEngine
from gap_fade_strategies import GapFadeStrategyRegistry


# ============================================================================
# FIXTURES
# ============================================================================

def _make_df(closes, n=60, base_price=100.0):
    """Build a DataFrame with OHLCV from a list of closes or auto-generated.

    If closes is None, generates `n` bars of smooth trending data around base_price.
    """
    if closes is not None:
        n = len(closes)
    else:
        np.random.seed(42)
        closes = (np.random.randn(n).cumsum() * 0.5 + base_price).tolist()

    dates = pd.bdate_range(end='2025-06-15', periods=n)
    highs = [c + abs(np.random.randn() * 0.5) + 0.1 for c in closes]
    lows = [c - abs(np.random.randn() * 0.5) - 0.1 for c in closes]
    opens = [(h + l) / 2 for h, l in zip(highs, lows)]
    volumes = [int(1e6 + np.random.randint(0, 5e5)) for _ in closes]

    df = pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows,
        'close': closes, 'volume': volumes,
    }, index=dates)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(np.float64)
    return df


def _make_swing_df():
    """Build a 80-bar DataFrame with clear swing highs and lows for key level testing.

    Creates a pattern: up-down-up-down so swing detection finds definite peaks/troughs.
    """
    np.random.seed(123)
    n = 80
    # Zigzag: 20 bars up, 20 down, 20 up, 20 down
    segment = 20
    closes = []
    base = 100.0
    for cycle in range(4):
        direction = 1 if cycle % 2 == 0 else -1
        for i in range(segment):
            base += direction * 0.3
            closes.append(base)

    return _make_df(closes)


def _make_bars(highs_lows):
    """Build a list of FiveMinBar objects from (high, low) tuples."""
    bars = []
    for i, (h, l) in enumerate(highs_lows):
        bars.append(FiveMinBar(
            open=(h + l) / 2, high=h, low=l,
            close=(h + l) / 2, volume=1000,
            bar_time=f'9:{35 + i * 5:02d}',
        ))
    return bars


def _make_analysis(key_levels=None, bb_upper=110.0, bb_middle=100.0,
                   bb_lower=90.0, fib_levels=None, atr=2.0):
    """Build a ConfluenceAnalysis for testing confidence scoring."""
    if key_levels is None:
        key_levels = [
            KeyLevel(price=102.0, level_type='resistance', touch_count=3,
                     last_touch_date='2025-06-10', strength=0.75),
            KeyLevel(price=95.0, level_type='support', touch_count=4,
                     last_touch_date='2025-06-08', strength=1.0),
        ]
    if fib_levels is None:
        fib_levels = {
            '0.236': 108.82,
            '0.382': 106.18,
            '0.5': 104.0,
            '0.618': 101.82,
        }
    return ConfluenceAnalysis(
        symbol='TEST',
        computed_date='2025-06-15',
        key_levels=key_levels,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        bb_middle=bb_middle,
        fib_levels=fib_levels,
        fib_swing_high=112.0,
        fib_swing_low=96.0,
        atr_14=atr,
    )


# ============================================================================
# REGISTRY & INSTANTIATION
# ============================================================================

class TestRegistration:
    """Verify the strategy registers and instantiates correctly."""

    def test_registered_in_registry(self):
        assert GapFadeStrategyRegistry.has_strategy('confluence_gap')

    def test_create_via_registry(self):
        s = GapFadeStrategyRegistry.create_strategy('confluence_gap')
        assert s.name == 'Confluence Gap'
        assert s.version == '1.0'

    def test_default_config_has_all_keys(self):
        s = ConfluenceGapStrategy()
        cfg = s.config
        expected_keys = [
            'lookback_days', 'swing_window', 'zone_merge_pct', 'min_touches',
            'max_key_levels', 'zone_proximity_pct', 'bb_period', 'bb_std_dev',
            'fib_lookback_days', 'min_confidence', 'premium_confidence',
            'require_trend_confirmation', 'trend_confirm_bars',
            'entry_after_minute', 'entry_cutoff_hour', 'entry_cutoff_minute',
            'stop_atr_multiple', 'stop_key_level_buffer_pct',
            'ema_trailing_period', 'ema_trailing_buffer_pct',
        ]
        for key in expected_keys:
            assert key in cfg, f"Missing config key: {key}"

    def test_custom_config_override(self):
        s = ConfluenceGapStrategy(config={'min_confidence': 0.60, 'bb_period': 30})
        assert s.config['min_confidence'] == 0.60
        assert s.config['bb_period'] == 30

    def test_required_indicators(self):
        s = ConfluenceGapStrategy()
        inds = s.get_required_indicators()
        assert 'vwap' in inds
        assert 'ema' in inds
        assert 'day_low' in inds
        assert 'bar_history' in inds

    def test_entry_window(self):
        s = ConfluenceGapStrategy()
        window = s.get_entry_window()
        assert window == (9, 40, 11, 30)

    def test_entry_window_custom(self):
        s = ConfluenceGapStrategy(config={
            **ConfluenceGapStrategy().get_default_config(),
            'entry_after_minute': 50,
            'entry_cutoff_hour': 12,
            'entry_cutoff_minute': 0,
        })
        assert s.get_entry_window() == (9, 50, 12, 0)

    def test_parameter_schema_completeness(self):
        s = ConfluenceGapStrategy()
        schema = s.get_parameter_schema()
        assert len(schema) >= 15
        for key, spec in schema.items():
            assert 'type' in spec
            assert 'label' in spec
            assert 'default' in spec

    def test_llm_prompts_not_none(self):
        s = ConfluenceGapStrategy()
        assert s.get_llm_system_prompt() is not None
        assert s.get_llm_profit_prompt() is not None


# ============================================================================
# BOLLINGER BANDS
# ============================================================================

class TestBollingerBands:

    def test_basic_computation(self):
        closes = [100.0] * 20
        upper, middle, lower = ConfluenceGapStrategy._compute_bollinger(closes, period=20, std_dev=2.0)
        assert middle == pytest.approx(100.0, abs=0.01)
        # With constant data, std dev = 0, so upper == lower == middle
        assert upper == pytest.approx(100.0, abs=0.01)
        assert lower == pytest.approx(100.0, abs=0.01)

    def test_with_variance(self):
        # alternating 99/101 gives mean=100, std=1
        closes = [99.0, 101.0] * 10
        upper, middle, lower = ConfluenceGapStrategy._compute_bollinger(closes, period=20, std_dev=2.0)
        assert middle == pytest.approx(100.0, abs=0.01)
        assert upper == pytest.approx(102.0, abs=0.01)
        assert lower == pytest.approx(98.0, abs=0.01)

    def test_insufficient_data(self):
        closes = [100.0] * 5
        upper, middle, lower = ConfluenceGapStrategy._compute_bollinger(closes, period=20)
        assert upper == 0.0
        assert middle == 0.0
        assert lower == 0.0

    def test_uses_last_n_bars(self):
        # 30 bars, first 10 at 50, last 20 at 100 — should use last 20
        closes = [50.0] * 10 + [100.0] * 20
        upper, middle, lower = ConfluenceGapStrategy._compute_bollinger(closes, period=20, std_dev=2.0)
        assert middle == pytest.approx(100.0, abs=0.01)


# ============================================================================
# FIBONACCI
# ============================================================================

class TestFibonacci:

    def test_basic_levels(self):
        # swing high=200, swing low=100, diff=100
        df = _make_df(None, n=60, base_price=150.0)
        # Override to have clean swing range
        df['high'] = 200.0
        df['low'] = 100.0
        fib, sh, sl = ConfluenceGapStrategy._compute_fibonacci(df, lookback=60)
        assert sh == 200.0
        assert sl == 100.0
        assert fib['0.382'] == pytest.approx(200.0 - 38.2, abs=0.01)
        assert fib['0.5'] == pytest.approx(150.0, abs=0.01)
        assert fib['0.618'] == pytest.approx(200.0 - 61.8, abs=0.01)

    def test_insufficient_data(self):
        df = _make_df([100.0, 101.0, 100.5])  # only 3 bars
        fib, sh, sl = ConfluenceGapStrategy._compute_fibonacci(df, lookback=60)
        assert fib == {}

    def test_flat_range_returns_empty(self):
        df = _make_df([100.0] * 20)
        df['high'] = 100.005
        df['low'] = 100.0
        fib, sh, sl = ConfluenceGapStrategy._compute_fibonacci(df, lookback=60)
        assert fib == {}

    def test_lookback_respects_window(self):
        # 100 bars; first 50 have wild range, last 30 are tight
        closes = list(np.linspace(50, 150, 70)) + [100.0] * 30
        df = _make_df(closes)
        df.loc[df.index[:70], 'high'] = 155.0
        df.loc[df.index[:70], 'low'] = 45.0
        df.loc[df.index[70:], 'high'] = 101.0
        df.loc[df.index[70:], 'low'] = 99.0
        fib, sh, sl = ConfluenceGapStrategy._compute_fibonacci(df, lookback=30)
        # Should only see the tight range
        assert sh == pytest.approx(101.0, abs=0.1)
        assert sl == pytest.approx(99.0, abs=0.1)


# ============================================================================
# ATR
# ============================================================================

class TestATR:

    def test_basic_atr(self):
        # Constant bars: high=101, low=99, close=100 → TR=2 each bar
        closes = [100.0] * 20
        df = _make_df(closes)
        df['high'] = 101.0
        df['low'] = 99.0
        df['close'] = 100.0
        atr = ConfluenceGapStrategy._compute_atr(df, period=14)
        assert atr == pytest.approx(2.0, abs=0.01)

    def test_insufficient_data(self):
        # 5 bars < period+1 (15) → returns 0.0 early
        df = _make_df([100.0] * 5)
        atr = ConfluenceGapStrategy._compute_atr(df, period=14)
        assert atr == 0.0

    def test_partial_data_uses_available(self):
        # 10 bars → 9 true ranges, fewer than period=14, uses average of available
        df = _make_df([100.0] * 16)
        df['high'] = 101.0
        df['low'] = 99.0
        df['close'] = 100.0
        atr = ConfluenceGapStrategy._compute_atr(df, period=14)
        assert atr > 0

    def test_very_short_data(self):
        df = _make_df([100.0])
        atr = ConfluenceGapStrategy._compute_atr(df, period=14)
        assert atr == 0.0

    def test_gap_increases_atr(self):
        # Day 2 gaps up: prev close=100, today high=110, low=108
        closes = [100.0] * 15 + [109.0] * 5
        df = _make_df(closes)
        df['high'] = 101.0
        df['low'] = 99.0
        df['close'] = 100.0
        # Create a gap at bar 15
        df.iloc[15, df.columns.get_loc('high')] = 110.0
        df.iloc[15, df.columns.get_loc('low')] = 108.0
        df.iloc[15, df.columns.get_loc('close')] = 109.0
        atr = ConfluenceGapStrategy._compute_atr(df, period=14)
        assert atr > 2.0  # Bigger than the normal TR of 2


# ============================================================================
# KEY LEVEL DETECTION
# ============================================================================

class TestKeyLevels:

    def test_finds_swing_highs_and_lows(self):
        s = ConfluenceGapStrategy()
        df = _make_swing_df()
        levels = s._find_key_levels(df)
        assert len(levels) > 0
        types = {lv.level_type for lv in levels}
        assert 'support' in types or 'resistance' in types

    def test_respects_min_touches(self):
        s = ConfluenceGapStrategy(config={
            **ConfluenceGapStrategy().get_default_config(),
            'min_touches': 5,
        })
        df = _make_swing_df()
        levels = s._find_key_levels(df)
        for lv in levels:
            assert lv.touch_count >= 5

    def test_respects_max_levels(self):
        s = ConfluenceGapStrategy(config={
            **ConfluenceGapStrategy().get_default_config(),
            'max_key_levels': 3,
            'min_touches': 1,
        })
        df = _make_swing_df()
        levels = s._find_key_levels(df)
        assert len(levels) <= 3

    def test_strength_normalized(self):
        s = ConfluenceGapStrategy()
        df = _make_swing_df()
        levels = s._find_key_levels(df)
        if levels:
            strengths = [lv.strength for lv in levels]
            assert max(strengths) <= 1.0
            assert min(strengths) >= 0.0

    def test_zone_merge_clusters_nearby_prices(self):
        """Two swing highs at 100.0 and 100.5 (0.5% apart) should merge
        when zone_merge_pct=1.5%."""
        s = ConfluenceGapStrategy(config={
            **ConfluenceGapStrategy().get_default_config(),
            'zone_merge_pct': 0.015,
            'min_touches': 1,
            'swing_window': 2,
        })
        # Build data with two peaks near 100 and a trough
        closes = [95.0] * 5 + [100.0] + [95.0] * 5 + [100.5] + [95.0] * 5
        df = _make_df(closes)
        # Make the peaks actual swing highs
        for i in [5, 11]:
            df.iloc[i, df.columns.get_loc('high')] = df.iloc[i]['close'] + 1.0
        levels = s._find_key_levels(df)
        # The two close peaks should be merged into 1 level
        resistance_levels = [lv for lv in levels if lv.level_type == 'resistance']
        # With merging, there should be at most 1 resistance near 100
        near_100 = [lv for lv in resistance_levels if 99 < lv.price < 102]
        assert len(near_100) <= 1

    def test_empty_data(self):
        s = ConfluenceGapStrategy()
        df = _make_df([100.0] * 3)  # too short for window=5
        levels = s._find_key_levels(df)
        assert levels == []


# ============================================================================
# CONFIDENCE SCORING
# ============================================================================

class TestConfidenceScoring:

    def test_gap_at_key_level_scores_high(self):
        s = ConfluenceGapStrategy()
        analysis = _make_analysis()
        # Open at 102.5 — within 2% of resistance at 102.0
        candidate = {
            'symbol': 'TEST', 'open': 102.5, 'prev_close': 100.0,
            'gap_pct': 0.025, 'direction': 'short',
        }
        score, factors, stype = s._compute_confidence(candidate, analysis)
        assert 'key_level' in factors
        assert factors['key_level'] == 0.35
        assert score >= 0.50

    def test_gap_far_from_levels_rejected(self):
        s = ConfluenceGapStrategy()
        analysis = _make_analysis()
        # Open at 80.0 — far from any level
        candidate = {
            'symbol': 'TEST', 'open': 80.0, 'prev_close': 78.0,
            'gap_pct': 0.025, 'direction': 'short',
        }
        score, factors, stype = s._compute_confidence(candidate, analysis)
        assert 'key_level' not in factors
        # Without key_level, score should be low
        assert score < 0.50 or stype != 'key_level'

    def test_gap_width_tiers(self):
        s = ConfluenceGapStrategy()
        analysis = _make_analysis()
        base = {'symbol': 'TEST', 'open': 102.0, 'prev_close': 100.0, 'direction': 'short'}

        # Small gap
        c1 = {**base, 'gap_pct': 0.007}
        _, f1, _ = s._compute_confidence(c1, analysis)
        assert f1.get('gap_width', 0) == 0.05

        # Medium gap
        c2 = {**base, 'gap_pct': 0.015}
        _, f2, _ = s._compute_confidence(c2, analysis)
        assert f2.get('gap_width', 0) == 0.10

        # Wide gap
        c3 = {**base, 'gap_pct': 0.03}
        _, f3, _ = s._compute_confidence(c3, analysis)
        assert f3.get('gap_width', 0) == 0.15

        # Very wide gap
        c4 = {**base, 'gap_pct': 0.05}
        _, f4, _ = s._compute_confidence(c4, analysis)
        assert f4.get('gap_width', 0) == 0.20

    def test_level_strength_tiers(self):
        s = ConfluenceGapStrategy()

        # 2-touch level
        analysis2 = _make_analysis(key_levels=[
            KeyLevel(price=102.0, level_type='resistance', touch_count=2,
                     last_touch_date='2025-06-10', strength=0.5),
        ])
        c = {'symbol': 'TEST', 'open': 102.0, 'prev_close': 100.0,
             'gap_pct': 0.02, 'direction': 'short'}
        _, f2, _ = s._compute_confidence(c, analysis2)
        assert f2.get('level_strength') == 0.04

        # 3-touch level
        analysis3 = _make_analysis(key_levels=[
            KeyLevel(price=102.0, level_type='resistance', touch_count=3,
                     last_touch_date='2025-06-10', strength=0.75),
        ])
        _, f3, _ = s._compute_confidence(c, analysis3)
        assert f3.get('level_strength') == 0.07

        # 4+-touch level
        analysis4 = _make_analysis(key_levels=[
            KeyLevel(price=102.0, level_type='resistance', touch_count=5,
                     last_touch_date='2025-06-10', strength=1.0),
        ])
        _, f4, _ = s._compute_confidence(c, analysis4)
        assert f4.get('level_strength') == 0.10

    def test_bb_break_short(self):
        s = ConfluenceGapStrategy()
        # BB upper at 110, open at 111 → above upper band
        analysis = _make_analysis(bb_upper=110.0)
        candidate = {
            'symbol': 'TEST', 'open': 111.0, 'prev_close': 105.0,
            'gap_pct': 0.057, 'direction': 'short',
        }
        _, factors, _ = s._compute_confidence(candidate, analysis)
        assert 'bb_break' in factors
        assert factors['bb_break'] == 0.15

    def test_bb_break_long(self):
        s = ConfluenceGapStrategy()
        analysis = _make_analysis(bb_lower=90.0)
        candidate = {
            'symbol': 'TEST', 'open': 89.0, 'prev_close': 92.0,
            'gap_pct': 0.033, 'direction': 'long',
        }
        _, factors, _ = s._compute_confidence(candidate, analysis)
        assert 'bb_break' in factors

    def test_fib_golden_zone(self):
        s = ConfluenceGapStrategy()
        # Fib levels: 0.5=104.0, 0.618=101.82
        # Open at 103.0 → inside golden zone [101.82, 104.0]
        analysis = _make_analysis(key_levels=[
            KeyLevel(price=103.0, level_type='resistance', touch_count=3,
                     last_touch_date='2025-06-10', strength=0.75),
        ])
        candidate = {
            'symbol': 'TEST', 'open': 103.0, 'prev_close': 100.0,
            'gap_pct': 0.03, 'direction': 'short',
        }
        _, factors, _ = s._compute_confidence(candidate, analysis)
        assert 'fib_golden_zone' in factors
        assert factors['fib_golden_zone'] == 0.15

    def test_fib_zone_near_382(self):
        s = ConfluenceGapStrategy()
        # Fib 0.382 = 106.18. Open at 106.0 → within 2% of 106.18
        analysis = _make_analysis(key_levels=[
            KeyLevel(price=106.0, level_type='resistance', touch_count=3,
                     last_touch_date='2025-06-10', strength=0.75),
        ])
        candidate = {
            'symbol': 'TEST', 'open': 106.0, 'prev_close': 103.0,
            'gap_pct': 0.029, 'direction': 'short',
        }
        _, factors, _ = s._compute_confidence(candidate, analysis)
        assert 'fib_zone' in factors or 'fib_golden_zone' in factors

    def test_false_breakout_short(self):
        s = ConfluenceGapStrategy()
        # Resistance at 102, gap opens at 103 (above resistance) → false breakout
        analysis = _make_analysis(key_levels=[
            KeyLevel(price=102.0, level_type='resistance', touch_count=3,
                     last_touch_date='2025-06-10', strength=0.75),
        ])
        candidate = {
            'symbol': 'TEST', 'open': 103.0, 'prev_close': 100.0,
            'gap_pct': 0.03, 'direction': 'short',
        }
        _, factors, _ = s._compute_confidence(candidate, analysis)
        assert 'false_breakout' in factors

    def test_false_breakout_long(self):
        s = ConfluenceGapStrategy()
        # Support at 95, gap opens at 94 (below support) → false breakdown
        analysis = _make_analysis(key_levels=[
            KeyLevel(price=95.0, level_type='support', touch_count=4,
                     last_touch_date='2025-06-08', strength=1.0),
        ])
        candidate = {
            'symbol': 'TEST', 'open': 94.0, 'prev_close': 96.0,
            'gap_pct': 0.021, 'direction': 'long',
        }
        _, factors, _ = s._compute_confidence(candidate, analysis)
        assert 'false_breakout' in factors

    def test_score_capped_at_095(self):
        """Even with all factors maxed, score should not exceed 0.95."""
        s = ConfluenceGapStrategy()
        # Craft a setup that hits every factor
        analysis = _make_analysis(
            key_levels=[
                KeyLevel(price=102.0, level_type='resistance', touch_count=5,
                         last_touch_date='2025-06-10', strength=1.0),
            ],
            bb_upper=101.0,  # open above BB
            fib_levels={
                '0.236': 108.82, '0.382': 106.18,
                '0.5': 104.0, '0.618': 101.82,
            },
        )
        candidate = {
            'symbol': 'TEST', 'open': 103.0, 'prev_close': 98.0,
            'gap_pct': 0.051, 'direction': 'short',
        }
        score, factors, stype = s._compute_confidence(candidate, analysis)
        assert score <= 0.95

    def test_premium_strategy_type(self):
        """Premium requires confidence >= 0.70 AND has BB break AND Fib."""
        s = ConfluenceGapStrategy()
        analysis = _make_analysis(
            key_levels=[
                KeyLevel(price=102.0, level_type='resistance', touch_count=4,
                         last_touch_date='2025-06-10', strength=1.0),
            ],
            bb_upper=101.0,
            fib_levels={
                '0.236': 108.82, '0.382': 106.18,
                '0.5': 104.0, '0.618': 101.82,
            },
        )
        candidate = {
            'symbol': 'TEST', 'open': 103.0, 'prev_close': 98.0,
            'gap_pct': 0.051, 'direction': 'short',
        }
        score, factors, stype = s._compute_confidence(candidate, analysis)
        assert stype == 'premium'
        assert score >= 0.70
        assert 'bb_break' in factors
        assert 'fib_golden_zone' in factors or 'fib_zone' in factors

    def test_key_level_strategy_type(self):
        s = ConfluenceGapStrategy()
        analysis = _make_analysis(bb_upper=200.0)  # BB way above, won't trigger
        candidate = {
            'symbol': 'TEST', 'open': 102.0, 'prev_close': 100.0,
            'gap_pct': 0.02, 'direction': 'short',
        }
        score, factors, stype = s._compute_confidence(candidate, analysis)
        if score >= 0.50 and 'key_level' in factors:
            assert stype == 'key_level'

    def test_none_strategy_below_threshold(self):
        s = ConfluenceGapStrategy()
        analysis = _make_analysis(
            key_levels=[],  # no key levels
            bb_upper=200.0, bb_lower=10.0,
            fib_levels={},
        )
        candidate = {
            'symbol': 'TEST', 'open': 50.0, 'prev_close': 49.5,
            'gap_pct': 0.003, 'direction': 'short',
        }
        score, factors, stype = s._compute_confidence(candidate, analysis)
        assert stype == 'none'

    def test_zero_ref_price(self):
        s = ConfluenceGapStrategy()
        analysis = _make_analysis()
        candidate = {'symbol': 'TEST', 'open': 0, 'prev_close': 0,
                     'gap_pct': 0.02, 'direction': 'short'}
        score, factors, stype = s._compute_confidence(candidate, analysis)
        assert score == 0.0
        assert stype == 'none'


# ============================================================================
# TREND CONFIRMATION
# ============================================================================

class TestTrendConfirmation:

    def test_bearish_lower_highs_lower_lows(self):
        s = ConfluenceGapStrategy()
        bars = _make_bars([(105, 100), (104, 99), (103, 98)])
        confirmed, reason = s._check_trend_confirmation('short', bars, 97.0)
        assert confirmed is True
        assert 'bearish' in reason

    def test_bearish_lower_highs_price_below_first_low(self):
        s = ConfluenceGapStrategy()
        bars = _make_bars([(105, 100), (104, 101), (103, 100.5)])
        # lower highs but NOT lower lows; price at 99 is below first bar low (100)
        confirmed, reason = s._check_trend_confirmation('short', bars, 99.0)
        assert confirmed is True
        assert 'below first bar low' in reason

    def test_bearish_no_structure(self):
        s = ConfluenceGapStrategy()
        # Higher highs — no bearish structure
        bars = _make_bars([(100, 95), (101, 96), (102, 97)])
        confirmed, reason = s._check_trend_confirmation('short', bars, 97.0)
        assert confirmed is False

    def test_bullish_higher_highs_higher_lows(self):
        s = ConfluenceGapStrategy()
        bars = _make_bars([(100, 95), (101, 96), (102, 97)])
        confirmed, reason = s._check_trend_confirmation('long', bars, 103.0)
        assert confirmed is True
        assert 'bullish' in reason

    def test_bullish_higher_lows_price_above_first_high(self):
        s = ConfluenceGapStrategy()
        bars = _make_bars([(100, 95), (99.5, 96), (100.2, 97)])
        # higher lows but NOT higher highs; price at 101 is above first bar high (100)
        confirmed, reason = s._check_trend_confirmation('long', bars, 101.0)
        assert confirmed is True
        assert 'above first bar high' in reason

    def test_bullish_no_structure(self):
        s = ConfluenceGapStrategy()
        bars = _make_bars([(105, 100), (104, 99), (103, 98)])
        confirmed, reason = s._check_trend_confirmation('long', bars, 97.0)
        assert confirmed is False

    def test_insufficient_bars(self):
        s = ConfluenceGapStrategy()
        bars = _make_bars([(105, 100)])
        confirmed, reason = s._check_trend_confirmation('short', bars, 99.0)
        assert confirmed is False
        assert 'need 3' in reason

    def test_custom_bar_count(self):
        s = ConfluenceGapStrategy(config={
            **ConfluenceGapStrategy().get_default_config(),
            'trend_confirm_bars': 4,
        })
        # 3 bars not enough for 4-bar requirement
        bars = _make_bars([(105, 100), (104, 99), (103, 98)])
        confirmed, reason = s._check_trend_confirmation('short', bars, 97.0)
        assert confirmed is False
        assert 'need 4' in reason


# ============================================================================
# FILTER CANDIDATE (with mocked _compute_analysis)
# ============================================================================

class TestFilterCandidate:

    def _strategy_with_analysis(self, analysis):
        """Return a strategy whose _compute_analysis is mocked."""
        s = ConfluenceGapStrategy()
        s._compute_analysis = MagicMock(return_value=analysis)
        return s

    def test_accepts_good_candidate(self):
        analysis = _make_analysis()
        s = self._strategy_with_analysis(analysis)
        candidate = {
            'symbol': 'TEST', 'date': '2025-06-15', 'open': 102.0,
            'prev_close': 100.0, 'gap_pct': 0.02, 'direction': 'short',
        }
        passed, reason = s.filter_candidate(candidate)
        assert passed is True
        assert 'confidence' in reason

    def test_rejects_no_symbol(self):
        s = ConfluenceGapStrategy()
        passed, reason = s.filter_candidate({'date': '2025-06-15'})
        assert passed is False
        assert 'no symbol' in reason

    def test_rejects_insufficient_data(self):
        s = self._strategy_with_analysis(None)
        candidate = {
            'symbol': 'TEST', 'date': '2025-06-15', 'open': 102.0,
            'prev_close': 100.0, 'gap_pct': 0.02, 'direction': 'short',
        }
        passed, reason = s.filter_candidate(candidate)
        assert passed is False
        assert 'insufficient' in reason

    def test_rejects_low_confidence(self):
        # No key levels, no BB, no fib → low confidence
        analysis = _make_analysis(
            key_levels=[], bb_upper=200.0, bb_lower=10.0, fib_levels={},
        )
        s = self._strategy_with_analysis(analysis)
        candidate = {
            'symbol': 'TEST', 'date': '2025-06-15', 'open': 50.0,
            'prev_close': 49.5, 'gap_pct': 0.003, 'direction': 'short',
        }
        passed, reason = s.filter_candidate(candidate)
        assert passed is False
        assert 'confidence' in reason

    def test_creates_pending_signal(self):
        analysis = _make_analysis()
        s = self._strategy_with_analysis(analysis)
        candidate = {
            'symbol': 'AAPL', 'date': '2025-06-15', 'open': 102.0,
            'prev_close': 100.0, 'gap_pct': 0.02, 'direction': 'short',
        }
        passed, _ = s.filter_candidate(candidate)
        if passed:
            assert 'AAPL' in s._pending
            pending = s._pending['AAPL']
            assert pending.confidence >= 0.50
            assert pending.direction == 'short'


# ============================================================================
# SCORE CANDIDATE
# ============================================================================

class TestScoreCandidate:

    def test_boost_scales_with_confidence(self):
        s = ConfluenceGapStrategy()
        # Install a pending signal at 0.60 confidence
        s._pending['TEST'] = PendingSignal(
            symbol='TEST', confidence=0.60, strategy_type='key_level',
            factors={'key_level': 0.35}, direction='short',
            nearest_level=KeyLevel(102.0, 'resistance', 3, '2025-06-10', 0.75),
            created_at=datetime.now(),
        )
        candidate = {'symbol': 'TEST', 'score': 50.0}
        result = s.score_candidate(candidate)
        assert result is not None
        assert result > 50.0
        boost = result - 50.0
        assert 10.0 <= boost <= 40.0

    def test_premium_gets_extra_boost(self):
        s = ConfluenceGapStrategy()
        level = KeyLevel(102.0, 'resistance', 3, '2025-06-10', 0.75)

        s._pending['A'] = PendingSignal(
            symbol='A', confidence=0.75, strategy_type='key_level',
            factors={}, direction='short', nearest_level=level,
            created_at=datetime.now(),
        )
        s._pending['B'] = PendingSignal(
            symbol='B', confidence=0.75, strategy_type='premium',
            factors={}, direction='short', nearest_level=level,
            created_at=datetime.now(),
        )
        score_key = s.score_candidate({'symbol': 'A', 'score': 50.0})
        score_prem = s.score_candidate({'symbol': 'B', 'score': 50.0})
        assert score_prem > score_key
        assert score_prem - score_key == pytest.approx(15.0, abs=0.1)

    def test_no_pending_returns_none(self):
        s = ConfluenceGapStrategy()
        assert s.score_candidate({'symbol': 'UNKNOWN', 'score': 50.0}) is None


# ============================================================================
# SHOULD ENTER NOW
# ============================================================================

class TestShouldEnterNow:

    def _setup_with_pending(self, direction='short', confirmed=False,
                            require_trend=True):
        s = ConfluenceGapStrategy(config={
            **ConfluenceGapStrategy().get_default_config(),
            'require_trend_confirmation': require_trend,
        })
        s._pending['TEST'] = PendingSignal(
            symbol='TEST', confidence=0.60, strategy_type='key_level',
            factors={'key_level': 0.35}, direction=direction,
            nearest_level=KeyLevel(102.0, 'resistance', 3, '2025-06-10', 0.75),
            created_at=datetime.now(), confirmed=confirmed,
        )
        return s

    def test_no_pending_signal_rejected(self):
        s = ConfluenceGapStrategy()
        ok, reason = s.should_enter_now({'symbol': 'TEST'}, 102.0)
        assert ok is False
        assert 'no pending' in reason

    def test_already_confirmed(self):
        s = self._setup_with_pending(confirmed=True)
        s._pending['TEST'].confirm_reason = 'already done'
        ok, reason = s.should_enter_now({'symbol': 'TEST'}, 102.0)
        assert ok is True

    def test_trend_confirmation_disabled_passes(self):
        s = self._setup_with_pending(require_trend=False)
        ok, reason = s.should_enter_now({'symbol': 'TEST'}, 102.0)
        assert ok is True
        assert 'disabled' in reason

    def test_no_tick_data_backtest_passthrough(self):
        s = self._setup_with_pending()
        ok, reason = s.should_enter_now({'symbol': 'TEST'}, 102.0, tick_data=None)
        assert ok is True
        assert 'backtest' in reason

    def test_empty_bar_history_waits(self):
        s = self._setup_with_pending()
        ok, reason = s.should_enter_now(
            {'symbol': 'TEST'}, 102.0,
            tick_data={'bar_history': []},
        )
        assert ok is False
        assert 'waiting' in reason

    def test_bearish_trend_confirms_short(self):
        s = self._setup_with_pending(direction='short')
        bars = _make_bars([(105, 100), (104, 99), (103, 98)])
        ok, reason = s.should_enter_now(
            {'symbol': 'TEST'}, 97.0,
            tick_data={'bar_history': bars},
        )
        assert ok is True
        assert 'confirmed' in reason.lower()
        # Confidence should have gotten the +0.15 boost
        assert s._pending['TEST'].confidence == pytest.approx(0.75, abs=0.01)

    def test_no_trend_structure_waits(self):
        s = self._setup_with_pending(direction='short')
        # Higher highs → no bearish trend
        bars = _make_bars([(100, 95), (101, 96), (102, 97)])
        ok, reason = s.should_enter_now(
            {'symbol': 'TEST'}, 103.0,
            tick_data={'bar_history': bars},
        )
        assert ok is False
        assert 'awaiting' in reason

    def test_trend_confirm_boost_applied_once(self):
        s = self._setup_with_pending(direction='short')
        bars = _make_bars([(105, 100), (104, 99), (103, 98)])
        tick_data = {'bar_history': bars}
        # First call confirms and boosts
        s.should_enter_now({'symbol': 'TEST'}, 97.0, tick_data=tick_data)
        conf_after_first = s._pending['TEST'].confidence
        # Second call uses cached confirmation, no double-boost
        s._pending['TEST'].confirmed = False  # reset to re-trigger
        s.should_enter_now({'symbol': 'TEST'}, 97.0, tick_data=tick_data)
        assert s._pending['TEST'].confidence == pytest.approx(conf_after_first, abs=0.01)


# ============================================================================
# COMPUTE STOP PRICE
# ============================================================================

class TestComputeStop:

    def _setup_strategy(self, atr=2.0, level_price=102.0, direction='short'):
        s = ConfluenceGapStrategy()
        analysis = _make_analysis(atr=atr)
        s._cache['TEST'] = analysis
        s._pending['TEST'] = PendingSignal(
            symbol='TEST', confidence=0.60, strategy_type='key_level',
            factors={}, direction=direction,
            nearest_level=KeyLevel(level_price, 'resistance', 3, '2025-06-10', 0.75),
            created_at=datetime.now(),
        )
        return s

    def test_atr_stop_short(self):
        s = self._setup_strategy(atr=2.0, level_price=0)  # no level
        s._pending['TEST'].nearest_level.price = 0  # disable level stop
        candidate = {'symbol': 'TEST', 'direction': 'short', 'date': '2025-06-15'}
        stop = s.compute_stop_price(100.0, candidate)
        # ATR=2.0, mult=1.5 → stop = 100 + 3 = 103
        assert stop == pytest.approx(103.0, abs=0.01)

    def test_atr_stop_long(self):
        s = self._setup_strategy(atr=2.0, level_price=0, direction='long')
        s._pending['TEST'].nearest_level.price = 0
        s._pending['TEST'].direction = 'long'
        candidate = {'symbol': 'TEST', 'direction': 'long', 'date': '2025-06-15'}
        stop = s.compute_stop_price(100.0, candidate)
        # ATR=2.0, mult=1.5 → stop = 100 - 3 = 97
        assert stop == pytest.approx(97.0, abs=0.01)

    def test_key_level_stop_tighter_for_short(self):
        # Resistance at 101.5, entry at 100, ATR stop = 103
        # Level stop = 101.5 * 1.003 ≈ 101.80 → tighter
        s = self._setup_strategy(atr=2.0, level_price=101.5)
        candidate = {'symbol': 'TEST', 'direction': 'short', 'date': '2025-06-15'}
        stop = s.compute_stop_price(100.0, candidate)
        assert stop < 103.0  # tighter than pure ATR
        assert stop == pytest.approx(101.80, abs=0.1)

    def test_no_analysis_returns_none(self):
        s = ConfluenceGapStrategy()
        candidate = {'symbol': 'UNKNOWN', 'direction': 'short', 'date': '2025-06-15'}
        stop = s.compute_stop_price(100.0, candidate)
        assert stop is None


# ============================================================================
# COMPUTE TARGETS
# ============================================================================

class TestComputeTargets:

    def _setup_strategy(self, atr=2.0, support_at=95.0):
        s = ConfluenceGapStrategy()
        analysis = _make_analysis(
            atr=atr,
            key_levels=[
                KeyLevel(price=102.0, level_type='resistance', touch_count=3,
                         last_touch_date='2025-06-10', strength=0.75),
                KeyLevel(price=support_at, level_type='support', touch_count=4,
                         last_touch_date='2025-06-08', strength=1.0),
            ],
        )
        s._cache['TEST'] = analysis
        s._pending['TEST'] = PendingSignal(
            symbol='TEST', confidence=0.60, strategy_type='key_level',
            factors={}, direction='short',
            nearest_level=KeyLevel(102.0, 'resistance', 3, '2025-06-10', 0.75),
            created_at=datetime.now(),
        )
        return s

    def test_short_targets_ordering(self):
        s = self._setup_strategy()
        candidate = {
            'symbol': 'TEST', 'direction': 'short',
            'prev_close': 98.0, 'date': '2025-06-15',
        }
        targets = s.compute_targets(100.0, candidate)
        assert targets is not None
        half, full = targets
        # For shorts: half_target >= full_target (closer to entry first)
        assert half >= full

    def test_long_targets_ordering(self):
        s = ConfluenceGapStrategy()
        analysis = _make_analysis(
            atr=2.0,
            key_levels=[
                KeyLevel(price=105.0, level_type='resistance', touch_count=3,
                         last_touch_date='2025-06-10', strength=0.75),
                KeyLevel(price=95.0, level_type='support', touch_count=4,
                         last_touch_date='2025-06-08', strength=1.0),
            ],
        )
        s._cache['TEST'] = analysis
        s._pending['TEST'] = PendingSignal(
            symbol='TEST', confidence=0.60, strategy_type='key_level',
            factors={}, direction='long',
            nearest_level=KeyLevel(95.0, 'support', 4, '2025-06-08', 1.0),
            created_at=datetime.now(),
        )
        candidate = {
            'symbol': 'TEST', 'direction': 'long',
            'prev_close': 98.0, 'date': '2025-06-15',
        }
        targets = s.compute_targets(96.0, candidate)
        assert targets is not None
        half, full = targets
        # For longs: half_target <= full_target
        assert half <= full

    def test_no_stop_returns_none(self):
        s = ConfluenceGapStrategy()
        candidate = {'symbol': 'UNKNOWN', 'direction': 'short',
                     'prev_close': 100.0, 'date': '2025-06-15'}
        assert s.compute_targets(100.0, candidate) is None


# ============================================================================
# EVALUATE EXIT
# ============================================================================

class TestEvaluateExit:

    def _setup_with_position(self, direction='short', level_price=102.0,
                             remaining=100):
        s = ConfluenceGapStrategy()
        s._pending['TEST'] = PendingSignal(
            symbol='TEST', confidence=0.70, strategy_type='key_level',
            factors={}, direction=direction,
            nearest_level=KeyLevel(level_price, 'resistance', 3, '2025-06-10', 0.75),
            created_at=datetime.now(),
        )
        position = {
            'symbol': 'TEST', 'direction': direction,
            'remaining_shares': remaining, 'partial_filled': False,
        }
        return s, position

    def test_key_level_invalidation_short(self):
        s, position = self._setup_with_position('short', level_price=102.0)
        # Price reclaims above resistance + buffer (102 * 1.003 = 102.306)
        signal = s.evaluate_exit(position, 103.0, 103.0)
        assert signal is not None
        assert signal.action == 'close'
        assert 'invalidation' in signal.reason

    def test_key_level_invalidation_long(self):
        s, position = self._setup_with_position('long', level_price=95.0)
        s._pending['TEST'].nearest_level = KeyLevel(
            95.0, 'support', 4, '2025-06-08', 1.0)
        # Price drops below support - buffer (95 * 0.997 = 94.715)
        signal = s.evaluate_exit(position, 94.0, 95.0)
        assert signal is not None
        assert signal.action == 'close'
        assert 'invalidation' in signal.reason

    def test_no_invalidation_within_zone(self):
        s, position = self._setup_with_position('short', level_price=102.0)
        # Price at 101.5 — still below the invalidation price
        signal = s.evaluate_exit(position, 101.5, 102.0)
        assert signal is None

    def test_ema_cross_exit_after_partial(self):
        s, position = self._setup_with_position('short', level_price=110.0)
        position['partial_filled'] = True
        tick_data = {'ema9': 99.0, 'ema_initialized': True}
        # Price above EMA → exit
        signal = s.evaluate_exit(position, 100.0, 100.0, tick_data=tick_data)
        assert signal is not None
        assert 'ema_cross' in signal.reason

    def test_ema_cross_not_triggered_before_partial(self):
        s, position = self._setup_with_position('short', level_price=110.0)
        position['partial_filled'] = False
        tick_data = {'ema9': 99.0, 'ema_initialized': True}
        signal = s.evaluate_exit(position, 100.0, 100.0, tick_data=tick_data)
        # Should NOT trigger EMA exit before partial fill
        assert signal is None

    def test_no_pending_returns_none(self):
        s = ConfluenceGapStrategy()
        position = {'symbol': 'UNKNOWN', 'direction': 'short',
                    'remaining_shares': 100}
        assert s.evaluate_exit(position, 100.0, 100.0) is None

    def test_zero_remaining_returns_none(self):
        s, position = self._setup_with_position('short')
        position['remaining_shares'] = 0
        assert s.evaluate_exit(position, 200.0, 200.0) is None


# ============================================================================
# TRAILING STOP
# ============================================================================

class TestTrailingStop:

    def test_tightens_short_stop(self):
        s = ConfluenceGapStrategy()
        position = {
            'direction': 'short', 'partial_filled': True,
            'stop_price': 105.0,
        }
        tick_data = {
            'ema_initialized': True,
            'ema_stop_short': 103.0,  # tighter than 105
        }
        new_stop = s.update_trailing_stop(position, 100.0, tick_data)
        assert new_stop == 103.0

    def test_does_not_widen_short_stop(self):
        s = ConfluenceGapStrategy()
        position = {
            'direction': 'short', 'partial_filled': True,
            'stop_price': 101.0,
        }
        tick_data = {
            'ema_initialized': True,
            'ema_stop_short': 103.0,  # wider than 101
        }
        new_stop = s.update_trailing_stop(position, 100.0, tick_data)
        assert new_stop is None

    def test_tightens_long_stop(self):
        s = ConfluenceGapStrategy()
        position = {
            'direction': 'long', 'partial_filled': True,
            'stop_price': 95.0,
        }
        tick_data = {
            'ema_initialized': True,
            'ema_stop_long': 97.0,  # tighter than 95
        }
        new_stop = s.update_trailing_stop(position, 100.0, tick_data)
        assert new_stop == 97.0

    def test_no_trail_before_partial(self):
        s = ConfluenceGapStrategy()
        position = {
            'direction': 'short', 'partial_filled': False,
            'stop_price': 105.0,
        }
        tick_data = {'ema_initialized': True, 'ema_stop_short': 103.0}
        assert s.update_trailing_stop(position, 100.0, tick_data) is None

    def test_no_trail_without_ema(self):
        s = ConfluenceGapStrategy()
        position = {
            'direction': 'short', 'partial_filled': True,
            'stop_price': 105.0,
        }
        tick_data = {'ema_initialized': False}
        assert s.update_trailing_stop(position, 100.0, tick_data) is None

    def test_no_tick_data(self):
        s = ConfluenceGapStrategy()
        position = {'direction': 'short', 'partial_filled': True, 'stop_price': 105.0}
        assert s.update_trailing_stop(position, 100.0, None) is None


# ============================================================================
# INDICATOR ENGINE EXTENSIONS (day_low, bar_history)
# ============================================================================

class TestIndicatorExtensions:

    def test_day_low_tracked(self):
        engine = TickIndicatorEngine(
            indicators=['day_low'],
        )
        ts = datetime(2025, 6, 15, 9, 35)
        engine.on_tick('TEST', 100.0, size=100, timestamp=ts)
        engine.on_tick('TEST', 98.0, size=100, timestamp=ts)
        engine.on_tick('TEST', 99.0, size=100, timestamp=ts)
        data = engine.get_data('TEST')
        assert data['day_low'] == 98.0

    def test_day_low_none_if_no_ticks(self):
        engine = TickIndicatorEngine(indicators=['day_low'])
        data = engine.get_data('TEST')
        assert data == {}

    def test_bar_history_builds(self):
        engine = TickIndicatorEngine(
            indicators=['bar_history'],
        )
        # Feed ticks across two 5-min bars
        ts1 = datetime(2025, 6, 15, 9, 31)
        ts2 = datetime(2025, 6, 15, 9, 36)  # next 5-min bar
        engine.on_tick('TEST', 100.0, size=100, timestamp=ts1)
        engine.on_tick('TEST', 101.0, size=100, timestamp=ts1)
        # New bar triggers completion of first
        engine.on_tick('TEST', 102.0, size=100, timestamp=ts2)
        data = engine.get_data('TEST')
        assert 'bar_history' in data
        assert data['bar_count'] >= 1
        assert len(data['bar_history']) >= 1

    def test_bar_history_without_ema(self):
        """bar_history should work even without 'ema' in indicators."""
        engine = TickIndicatorEngine(
            indicators=['bar_history'],  # no 'ema'
        )
        ts1 = datetime(2025, 6, 15, 9, 31)
        ts2 = datetime(2025, 6, 15, 9, 36)
        engine.on_tick('TEST', 100.0, size=100, timestamp=ts1)
        engine.on_tick('TEST', 102.0, size=100, timestamp=ts2)
        data = engine.get_data('TEST')
        assert 'bar_history' in data

    def test_day_high_still_works(self):
        """Ensure existing day_high indicator wasn't broken."""
        engine = TickIndicatorEngine(indicators=['day_high'])
        ts = datetime(2025, 6, 15, 9, 35)
        engine.on_tick('TEST', 100.0, size=100, timestamp=ts)
        engine.on_tick('TEST', 105.0, size=100, timestamp=ts)
        engine.on_tick('TEST', 103.0, size=100, timestamp=ts)
        data = engine.get_data('TEST')
        assert data['day_high'] == 105.0

    def test_combined_indicators(self):
        """All indicators for confluence_gap work together."""
        engine = TickIndicatorEngine(
            indicators=['vwap', 'ema', 'opening_range', 'day_high', 'day_low', 'bar_history'],
            ema_period=9,
        )
        ts = datetime(2025, 6, 15, 9, 31)
        for i in range(5):
            engine.on_tick('TEST', 100.0 + i, size=100,
                           timestamp=ts + timedelta(seconds=i * 10))
        data = engine.get_data('TEST')
        assert 'vwap' in data
        assert 'day_high' in data
        assert 'day_low' in data


# ============================================================================
# FIND NEXT LEVEL TARGET
# ============================================================================

class TestFindNextLevelTarget:

    def test_finds_support_below_for_short(self):
        s = ConfluenceGapStrategy()
        s._cache['TEST'] = _make_analysis(key_levels=[
            KeyLevel(110.0, 'resistance', 3, '2025-06-10', 0.75),
            KeyLevel(95.0, 'support', 4, '2025-06-08', 1.0),
            KeyLevel(90.0, 'support', 2, '2025-06-05', 0.5),
        ])
        target = s._find_next_level_target('TEST', 100.0, 'short')
        # Should find highest support below entry = 95.0
        assert target == 95.0

    def test_finds_resistance_above_for_long(self):
        s = ConfluenceGapStrategy()
        s._cache['TEST'] = _make_analysis(key_levels=[
            KeyLevel(110.0, 'resistance', 3, '2025-06-10', 0.75),
            KeyLevel(105.0, 'resistance', 2, '2025-06-09', 0.5),
            KeyLevel(90.0, 'support', 4, '2025-06-08', 1.0),
        ])
        target = s._find_next_level_target('TEST', 100.0, 'long')
        # Should find lowest resistance above entry = 105.0
        assert target == 105.0

    def test_no_level_in_direction(self):
        s = ConfluenceGapStrategy()
        s._cache['TEST'] = _make_analysis(key_levels=[
            KeyLevel(110.0, 'resistance', 3, '2025-06-10', 0.75),
        ])
        # No support below 100
        target = s._find_next_level_target('TEST', 100.0, 'short')
        assert target is None

    def test_no_cache(self):
        s = ConfluenceGapStrategy()
        assert s._find_next_level_target('UNKNOWN', 100.0, 'short') is None


# ============================================================================
# CONFIG UPDATE
# ============================================================================

class TestConfigUpdate:

    def test_update_config(self):
        s = ConfluenceGapStrategy()
        original_min = s.config['min_confidence']
        s.update_config({'min_confidence': 0.65})
        assert s.config['min_confidence'] == 0.65
        assert s.config['min_confidence'] != original_min

    def test_update_preserves_other_keys(self):
        s = ConfluenceGapStrategy()
        original_bb = s.config['bb_period']
        s.update_config({'min_confidence': 0.65})
        assert s.config['bb_period'] == original_bb
