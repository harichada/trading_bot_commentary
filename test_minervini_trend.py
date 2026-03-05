#!/usr/bin/env python3
"""
Test Suite for Minervini Trend Template Strategy
=================================================
Comprehensive tests for Stage 2 trend analysis, 8-criterion evaluation,
direction-aware filtering, scoring, stops, targets, exits, and trailing stops.

Run with: python -m pytest test_minervini_trend.py -v
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from gap_fade_strategies.minervini_trend import (
    MinerviniTrendStrategy,
    MinerviniAnalysis,
    MinerviniCriteria,
)
from gap_fade_strategies.base import ExitSignal
from gap_fade_strategies import GapFadeStrategyRegistry


# ============================================================================
# FIXTURES
# ============================================================================

def _make_stage2_df(n=252):
    """Build a DataFrame simulating a steady Stage 2 uptrend.

    Price rises from ~100 to ~150 over n bars. All SMAs aligned:
    price > sma_50 > sma_150 > sma_200.
    """
    np.random.seed(42)
    # Smooth uptrend with small noise
    t = np.linspace(0, 1, n)
    closes = 100.0 + 50.0 * t + np.random.randn(n) * 0.5
    closes = closes.tolist()

    dates = pd.bdate_range(end='2025-06-15', periods=n)
    highs = [c + abs(np.random.randn() * 0.3) + 0.2 for c in closes]
    lows = [c - abs(np.random.randn() * 0.3) - 0.2 for c in closes]
    opens = [(h + l) / 2 for h, l in zip(highs, lows)]
    volumes = [int(1e6 + np.random.randint(0, 5e5)) for _ in closes]

    df = pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows,
        'close': closes, 'volume': volumes,
    }, index=dates)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(np.float64)
    return df


def _make_stage4_df(n=252):
    """Build a DataFrame simulating a Stage 4 downtrend.

    Price drops from ~100 to ~60 over n bars.
    """
    np.random.seed(99)
    t = np.linspace(0, 1, n)
    closes = 100.0 - 40.0 * t + np.random.randn(n) * 0.5
    closes = closes.tolist()

    dates = pd.bdate_range(end='2025-06-15', periods=n)
    highs = [c + abs(np.random.randn() * 0.3) + 0.2 for c in closes]
    lows = [c - abs(np.random.randn() * 0.3) - 0.2 for c in closes]
    opens = [(h + l) / 2 for h, l in zip(highs, lows)]
    volumes = [int(1e6 + np.random.randint(0, 5e5)) for _ in closes]

    df = pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows,
        'close': closes, 'volume': volumes,
    }, index=dates)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(np.float64)
    return df


def _make_spy_df(n=252, start=100.0, end=120.0):
    """Build a SPY DataFrame for relative strength calculation."""
    np.random.seed(77)
    t = np.linspace(0, 1, n)
    closes = start + (end - start) * t + np.random.randn(n) * 0.3
    closes = closes.tolist()

    dates = pd.bdate_range(end='2025-06-15', periods=n)
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    opens = [(h + l) / 2 for h, l in zip(highs, lows)]
    volumes = [int(5e7) for _ in closes]

    df = pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows,
        'close': closes, 'volume': volumes,
    }, index=dates)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(np.float64)
    return df


def _make_analysis(stage=2, criteria_count=7, confidence=None,
                   rs=1.2, sma_50=145.0, sma_150=130.0, sma_200=120.0,
                   price=148.0, low_52w=100.0, high_52w=152.0,
                   low_50d=140.0, atr=2.5):
    """Build a MinerviniAnalysis directly for unit testing."""
    # Build criteria to match the requested count
    all_flags = [True] * min(criteria_count, 8) + [False] * max(0, 8 - criteria_count)
    criteria = MinerviniCriteria(
        price_above_150sma=all_flags[0],
        price_above_200sma=all_flags[1],
        sma150_above_200sma=all_flags[2],
        sma200_trending_up=all_flags[3],
        sma50_above_150_and_200=all_flags[4],
        price_above_50sma=all_flags[5],
        pct_above_52w_low=all_flags[6],
        within_25pct_52w_high=all_flags[7],
    )
    if confidence is None:
        confidence = criteria.count / 8.0

    return MinerviniAnalysis(
        symbol='TEST',
        computed_date='2025-06-15',
        criteria=criteria,
        stage=stage,
        confidence=confidence,
        relative_strength=rs,
        sma_50=sma_50,
        sma_150=sma_150,
        sma_200=sma_200,
        current_price=price,
        low_52w=low_52w,
        high_52w=high_52w,
        low_50d=low_50d,
        atr_14=atr,
    )


# ============================================================================
# REGISTRY & INSTANTIATION
# ============================================================================

class TestRegistration:
    """Verify the strategy registers and instantiates correctly."""

    def test_registered_in_registry(self):
        assert GapFadeStrategyRegistry.has_strategy('minervini_trend')

    def test_create_via_registry(self):
        s = GapFadeStrategyRegistry.create_strategy('minervini_trend')
        assert s.name == 'Minervini Trend Template'
        assert s.version == '1.0'

    def test_default_config_has_all_keys(self):
        s = MinerviniTrendStrategy()
        cfg = s.config
        expected_keys = [
            'lookback_days', 'sma_50_period', 'sma_150_period', 'sma_200_period',
            'sma_200_uptrend_days', 'pct_above_52w_low', 'pct_within_52w_high',
            'min_criteria', 'min_confidence', 'reject_stage2_shorts',
            'score_boost_max', 'score_boost_min',
            'entry_after_minute', 'entry_cutoff_hour', 'entry_cutoff_minute',
            'require_volume_confirmation', 'volume_confirm_ratio',
            'stage_deterioration_exit', 'profit_drawdown_pct', 'profit_protect_trigger_pct',
        ]
        for key in expected_keys:
            assert key in cfg, f"Missing config key: {key}"

    def test_required_indicators(self):
        s = MinerviniTrendStrategy()
        inds = s.get_required_indicators()
        assert 'vwap' in inds
        assert 'day_high' in inds
        assert 'day_low' in inds

    def test_entry_window(self):
        s = MinerviniTrendStrategy()
        window = s.get_entry_window()
        assert window == (9, 35, 11, 0)

    def test_parameter_schema_completeness(self):
        s = MinerviniTrendStrategy()
        schema = s.get_parameter_schema()
        assert len(schema) >= 10
        for key, spec in schema.items():
            assert 'type' in spec
            assert 'label' in spec
            assert 'default' in spec

    def test_llm_prompts_not_none(self):
        s = MinerviniTrendStrategy()
        assert s.get_llm_system_prompt() is not None
        assert s.get_llm_profit_prompt() is not None
        assert 'Minervini' in s.get_llm_system_prompt()
        assert 'Stage 2' in s.get_llm_system_prompt()


# ============================================================================
# CRITERIA EVALUATION
# ============================================================================

class TestCriteriaEvaluation:

    def test_all_8_criteria_met(self):
        """Stage 2 uptrend should have all 8 criteria True."""
        criteria = MinerviniCriteria(
            price_above_150sma=True,
            price_above_200sma=True,
            sma150_above_200sma=True,
            sma200_trending_up=True,
            sma50_above_150_and_200=True,
            price_above_50sma=True,
            pct_above_52w_low=True,
            within_25pct_52w_high=True,
        )
        assert criteria.count == 8

    def test_no_criteria_met(self):
        criteria = MinerviniCriteria()
        assert criteria.count == 0

    def test_partial_criteria(self):
        criteria = MinerviniCriteria(
            price_above_150sma=True,
            price_above_200sma=True,
            sma150_above_200sma=True,
            sma200_trending_up=False,
            sma50_above_150_and_200=True,
            price_above_50sma=False,
            pct_above_52w_low=True,
            within_25pct_52w_high=False,
        )
        assert criteria.count == 5

    @patch('gap_fade_app.get_price_db')
    def test_compute_analysis_stage2(self, mock_get_db):
        """Full _compute_analysis on Stage 2 data should identify Stage 2."""
        df = _make_stage2_df(252)
        spy_df = _make_spy_df(252)
        mock_db = MagicMock()
        mock_db.get_bars.side_effect = lambda sym, s, e: df if sym != 'SPY' else spy_df
        mock_get_db.return_value = mock_db

        s = MinerviniTrendStrategy()
        analysis = s._compute_analysis('AAPL', '2025-06-15')

        assert analysis is not None
        assert analysis.stage == 2
        assert analysis.criteria.count >= 6
        assert analysis.confidence >= 0.75

    @patch('gap_fade_app.get_price_db')
    def test_compute_analysis_stage4(self, mock_get_db):
        """Full _compute_analysis on Stage 4 data should identify Stage 4."""
        df = _make_stage4_df(252)
        spy_df = _make_spy_df(252)
        mock_db = MagicMock()
        mock_db.get_bars.side_effect = lambda sym, s, e: df if sym != 'SPY' else spy_df
        mock_get_db.return_value = mock_db

        s = MinerviniTrendStrategy()
        analysis = s._compute_analysis('WEAK', '2025-06-15')

        assert analysis is not None
        assert analysis.stage == 4
        assert analysis.criteria.count <= 3

    @patch('gap_fade_app.get_price_db')
    def test_compute_analysis_insufficient_data(self, mock_get_db):
        """Should return None if fewer than 200 bars."""
        df = _make_stage2_df(252).tail(100)  # Only 100 bars
        mock_db = MagicMock()
        mock_db.get_bars.return_value = df
        mock_get_db.return_value = mock_db

        s = MinerviniTrendStrategy()
        analysis = s._compute_analysis('SHORT', '2025-06-15')
        assert analysis is None


# ============================================================================
# STAGE CLASSIFICATION
# ============================================================================

class TestStageClassification:

    def test_stage2_full_alignment(self):
        criteria = MinerviniCriteria(
            price_above_150sma=True, price_above_200sma=True,
            sma150_above_200sma=True, sma200_trending_up=True,
            sma50_above_150_and_200=True, price_above_50sma=True,
            pct_above_52w_low=True, within_25pct_52w_high=True,
        )
        stage = MinerviniTrendStrategy._classify_stage(
            criteria, sma_50=145, sma_150=130, sma_200=120, price=150
        )
        assert stage == 2

    def test_stage4_downtrend(self):
        criteria = MinerviniCriteria(
            price_above_150sma=False, price_above_200sma=False,
            sma150_above_200sma=False, sma200_trending_up=False,
            sma50_above_150_and_200=False, price_above_50sma=False,
            pct_above_52w_low=False, within_25pct_52w_high=False,
        )
        # price < sma_50 AND sma_50 < sma_150
        stage = MinerviniTrendStrategy._classify_stage(
            criteria, sma_50=70, sma_150=80, sma_200=90, price=60
        )
        assert stage == 4

    def test_stage3_distribution(self):
        criteria = MinerviniCriteria(
            price_above_150sma=True, price_above_200sma=True,
            sma150_above_200sma=True, sma200_trending_up=False,
            sma50_above_150_and_200=False, price_above_50sma=False,
            pct_above_52w_low=True, within_25pct_52w_high=False,
        )
        # 4 criteria, MA alignment broken (price < sma_50)
        stage = MinerviniTrendStrategy._classify_stage(
            criteria, sma_50=135, sma_150=125, sma_200=120, price=130
        )
        assert stage == 3

    def test_stage1_basing(self):
        criteria = MinerviniCriteria(
            price_above_150sma=True, price_above_200sma=True,
            sma150_above_200sma=False, sma200_trending_up=False,
            sma50_above_150_and_200=False, price_above_50sma=False,
            pct_above_52w_low=False, within_25pct_52w_high=False,
        )
        # 2 criteria — doesn't match stage 3 (need 3-5), not stage 4 pattern
        stage = MinerviniTrendStrategy._classify_stage(
            criteria, sma_50=99, sma_150=100, sma_200=101, price=100.5
        )
        assert stage == 1


# ============================================================================
# RELATIVE STRENGTH
# ============================================================================

class TestRelativeStrength:

    def test_outperforming_spy(self):
        # Stock: 100 → 150 (50% gain), SPY: 100 → 120 (20% gain)
        sym_df = _make_stage2_df(252)
        spy_df = _make_spy_df(252, start=100.0, end=120.0)

        rs = MinerviniTrendStrategy._compute_relative_strength(sym_df, spy_df, lookback=63)
        assert rs > 1.0

    def test_no_spy_data_fallback(self):
        sym_df = _make_stage2_df(252)
        rs = MinerviniTrendStrategy._compute_relative_strength(sym_df, None, lookback=63)
        assert rs == 1.0

    def test_insufficient_spy_data(self):
        sym_df = _make_stage2_df(252)
        spy_df = _make_spy_df(30)  # Only 30 bars, need 63
        rs = MinerviniTrendStrategy._compute_relative_strength(sym_df, spy_df, lookback=63)
        assert rs == 1.0


# ============================================================================
# FILTER CANDIDATE
# ============================================================================

class TestFilterCandidate:

    def test_long_stage2_passes(self):
        """Gap-down long on Stage 2 stock should pass."""
        s = MinerviniTrendStrategy()
        analysis = _make_analysis(stage=2, criteria_count=7)
        s._cache['AAPL'] = analysis

        with patch.object(s, '_compute_analysis', return_value=analysis):
            passed, reason = s.filter_candidate({
                'symbol': 'AAPL', 'direction': 'long', 'date': '2025-06-15',
            })
        assert passed is True
        assert 'Stage 2 pullback' in reason

    def test_long_stage4_rejected(self):
        """Gap-down long on Stage 4 stock should be rejected."""
        s = MinerviniTrendStrategy()
        analysis = _make_analysis(stage=4, criteria_count=1)
        s._cache['WEAK'] = analysis

        with patch.object(s, '_compute_analysis', return_value=analysis):
            passed, reason = s.filter_candidate({
                'symbol': 'WEAK', 'direction': 'long', 'date': '2025-06-15',
            })
        assert passed is False
        assert 'not Stage 2' in reason

    def test_short_stage2_rejected(self):
        """Gap-up short on Stage 2 stock should be REJECTED (don't short uptrends)."""
        s = MinerviniTrendStrategy()
        analysis = _make_analysis(stage=2, criteria_count=7)

        with patch.object(s, '_compute_analysis', return_value=analysis):
            passed, reason = s.filter_candidate({
                'symbol': 'AAPL', 'direction': 'short', 'date': '2025-06-15',
            })
        assert passed is False
        assert 'REJECT short' in reason
        assert 'Stage 2' in reason

    def test_short_stage4_passes(self):
        """Gap-up short on Stage 4 stock should pass (safe to short)."""
        s = MinerviniTrendStrategy()
        analysis = _make_analysis(stage=4, criteria_count=1)

        with patch.object(s, '_compute_analysis', return_value=analysis):
            passed, reason = s.filter_candidate({
                'symbol': 'WEAK', 'direction': 'short', 'date': '2025-06-15',
            })
        assert passed is True
        assert 'short allowed' in reason

    def test_no_symbol_fails(self):
        s = MinerviniTrendStrategy()
        passed, reason = s.filter_candidate({'direction': 'long'})
        assert passed is False
        assert 'no symbol' in reason

    def test_insufficient_data_passes_through(self):
        """If analysis returns None (not enough bars), pass through."""
        s = MinerviniTrendStrategy()
        with patch.object(s, '_compute_analysis', return_value=None):
            passed, reason = s.filter_candidate({
                'symbol': 'NEW', 'direction': 'long', 'date': '2025-06-15',
            })
        assert passed is True
        assert 'insufficient data' in reason


# ============================================================================
# SCORE CANDIDATE
# ============================================================================

class TestScoreCandidate:

    def test_long_gets_boost(self):
        """Long candidates in Stage 2 get a score boost."""
        s = MinerviniTrendStrategy()
        analysis = _make_analysis(stage=2, criteria_count=7, rs=1.2)
        s._cache['AAPL'] = analysis

        score = s.score_candidate({
            'symbol': 'AAPL', 'direction': 'long', 'score': 50.0,
        })
        assert score is not None
        assert score > 50.0
        assert score <= 90.0  # base 50 + max boost 40

    def test_short_returns_none(self):
        """Short candidates should return None (engine default)."""
        s = MinerviniTrendStrategy()
        analysis = _make_analysis(stage=4, criteria_count=1)
        s._cache['WEAK'] = analysis

        score = s.score_candidate({
            'symbol': 'WEAK', 'direction': 'short', 'score': 50.0,
        })
        assert score is None

    def test_8_of_8_criteria_max_boost(self):
        """8/8 criteria with high RS should approach max boost."""
        s = MinerviniTrendStrategy()
        analysis = _make_analysis(stage=2, criteria_count=8, rs=1.5)
        s._cache['BEST'] = analysis

        score = s.score_candidate({
            'symbol': 'BEST', 'direction': 'long', 'score': 50.0,
        })
        assert score is not None
        # Should get close to max boost (40)
        assert score >= 80.0


# ============================================================================
# ENTRY GATING
# ============================================================================

class TestEntryGating:

    def test_short_passthrough(self):
        s = MinerviniTrendStrategy()
        enter, reason = s.should_enter_now(
            {'direction': 'short'}, price=100.0
        )
        assert enter is True
        assert 'passthrough' in reason

    def test_long_volume_confirmed(self):
        s = MinerviniTrendStrategy()
        enter, reason = s.should_enter_now(
            {'direction': 'long'}, price=100.0,
            tick_data={'volume_ratio': 1.5},
        )
        assert enter is True
        assert 'volume confirmed' in reason

    def test_long_volume_insufficient(self):
        s = MinerviniTrendStrategy()
        enter, reason = s.should_enter_now(
            {'direction': 'long'}, price=100.0,
            tick_data={'volume_ratio': 0.5},
        )
        assert enter is False
        assert 'waiting for volume' in reason

    def test_long_no_tick_data_passes(self):
        s = MinerviniTrendStrategy()
        enter, reason = s.should_enter_now(
            {'direction': 'long'}, price=100.0, tick_data=None,
        )
        assert enter is True

    def test_volume_confirmation_disabled(self):
        s = MinerviniTrendStrategy(config={
            **MinerviniTrendStrategy().get_default_config(),
            'require_volume_confirmation': False,
        })
        enter, reason = s.should_enter_now(
            {'direction': 'long'}, price=100.0,
            tick_data={'volume_ratio': 0.1},
        )
        assert enter is True


# ============================================================================
# STOPS & TARGETS (engine-managed)
# ============================================================================

class TestStopsAndTargets:
    """Stops and targets are handled by the engine — strategy returns None."""

    def test_stop_returns_none_long(self):
        s = MinerviniTrendStrategy()
        stop = s.compute_stop_price(148.0, {'symbol': 'AAPL', 'direction': 'long'})
        assert stop is None

    def test_stop_returns_none_short(self):
        s = MinerviniTrendStrategy()
        stop = s.compute_stop_price(148.0, {'symbol': 'AAPL', 'direction': 'short'})
        assert stop is None

    def test_targets_returns_none_long(self):
        s = MinerviniTrendStrategy()
        targets = s.compute_targets(148.0, {'symbol': 'AAPL', 'direction': 'long'})
        assert targets is None

    def test_targets_returns_none_short(self):
        s = MinerviniTrendStrategy()
        targets = s.compute_targets(148.0, {'symbol': 'AAPL', 'direction': 'short'})
        assert targets is None


# ============================================================================
# EXIT EVALUATION
# ============================================================================

class TestExitEvaluation:

    def test_stage4_deterioration_exit(self):
        """Should exit long if stock drops to Stage 4."""
        s = MinerviniTrendStrategy()
        analysis = _make_analysis(stage=4, criteria_count=1)
        s._cache['AAPL'] = analysis

        signal = s.evaluate_exit(
            {'symbol': 'AAPL', 'direction': 'long', 'remaining_shares': 100,
             'entry_price': 148.0, 'peak_price': 155.0},
            price=130.0, high=132.0,
        )
        assert signal is not None
        assert signal.action == 'close'
        assert 'stage_deterioration' in signal.reason
        assert signal.shares == 100

    def test_profit_drawdown_exit(self):
        """Should exit if giving back 40%+ of 5%+ peak profit."""
        s = MinerviniTrendStrategy()
        # Cache a Stage 2 analysis (so stage deterioration doesn't trigger)
        analysis = _make_analysis(stage=2, criteria_count=7)
        s._cache['AAPL'] = analysis

        signal = s.evaluate_exit(
            {'symbol': 'AAPL', 'direction': 'long', 'remaining_shares': 100,
             'entry_price': 100.0, 'peak_price': 110.0},  # 10% peak profit
            price=104.0, high=105.0,   # only 4% now, gave back 60% of peak
        )
        assert signal is not None
        assert signal.action == 'close'
        assert 'profit_drawdown' in signal.reason

    def test_no_exit_in_healthy_uptrend(self):
        """Should NOT exit in healthy Stage 2 with growing profit."""
        s = MinerviniTrendStrategy()
        analysis = _make_analysis(stage=2, criteria_count=8)
        s._cache['AAPL'] = analysis

        signal = s.evaluate_exit(
            {'symbol': 'AAPL', 'direction': 'long', 'remaining_shares': 100,
             'entry_price': 100.0, 'peak_price': 108.0},
            price=107.0, high=108.0,   # 7% profit, gave back only 12.5% of peak
        )
        assert signal is None

    def test_short_returns_none(self):
        """Short positions should return None (engine default)."""
        s = MinerviniTrendStrategy()
        signal = s.evaluate_exit(
            {'symbol': 'AAPL', 'direction': 'short', 'remaining_shares': 100,
             'entry_price': 100.0, 'peak_price': 95.0},
            price=96.0, high=97.0,
        )
        assert signal is None

    def test_no_exit_below_trigger(self):
        """Should NOT trigger profit drawdown if profit < trigger threshold."""
        s = MinerviniTrendStrategy()
        analysis = _make_analysis(stage=2, criteria_count=7)
        s._cache['AAPL'] = analysis

        signal = s.evaluate_exit(
            {'symbol': 'AAPL', 'direction': 'long', 'remaining_shares': 100,
             'entry_price': 100.0, 'peak_price': 103.0},  # only 3% peak
            price=101.0, high=102.0,   # gave back 67% but peak was only 3%
        )
        assert signal is None


# ============================================================================
# TRAILING STOP (engine-managed)
# ============================================================================

class TestTrailingStop:
    """Trailing stop is handled by the engine — strategy returns None."""

    def test_trailing_returns_none(self):
        s = MinerviniTrendStrategy()
        new_stop = s.update_trailing_stop(
            {'symbol': 'AAPL', 'direction': 'long', 'stop_price': 137.0},
            price=155.0,
        )
        assert new_stop is None


# ============================================================================
# ATR COMPUTATION
# ============================================================================

class TestATR:

    def test_atr_basic(self):
        df = _make_stage2_df(50)
        atr = MinerviniTrendStrategy._compute_atr(df, period=14)
        assert atr > 0

    def test_atr_insufficient_data(self):
        df = _make_stage2_df(252).head(2)
        atr = MinerviniTrendStrategy._compute_atr(df, period=14)
        # With only 2 bars, should still return something (1 true range)
        assert atr >= 0
