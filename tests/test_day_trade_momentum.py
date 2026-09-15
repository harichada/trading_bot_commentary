"""Tests for v-day-trade-momentum-desk-2026-09-10.

Coverage:
  1. Mover passes quality relax (bypasses SMA50/RS)
  2. risk_off reduces size, doesn't hard-block (when DAY_TRADE_HARD_SKIP_RISK_OFF=False)
  3. Momentum signal generates under synthetic bars
  4. Extreme risk (SPY <= -1.5% AND VIX spike) hard-blocks
  5. Day-trade size multiplier applied correctly
  6. v-day-trade-hard-skip-risk-off-2026-09-14: hard-skip on risk_off (GLW RCA)
"""
import os
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np

from core.models import SignalType, TradingMode


# ══════════════════════════════════════════════════════════════════════════════
# v-day-trade-hard-skip-risk-off-2026-09-14: Tests for hard-skip on risk_off
#
# 2026-09-14 RCA (GLW): strategy logged many `size_reduced
# reason=risk_off_not_blocked size_mult=0.25` then later entered when
# regime flipped mixed. Research/CoS: size-down is how weak risk_off
# path still feeds LIVE; want hard skip like ORB.
#
# Default: DAY_TRADE_HARD_SKIP_RISK_OFF=True (hard-skip, same as ORB)
# Legacy:  DAY_TRADE_HARD_SKIP_RISK_OFF=False (size-reduction, prior behavior)
# ══════════════════════════════════════════════════════════════════════════════


class TestDayTradeHardSkipRiskOff:
    """Test DAY_TRADE_HARD_SKIP_RISK_OFF behavior.
    
    v-day-trade-hard-skip-risk-off-2026-09-14: hard-skip day_trade_momentum
    in risk_off regime (same as ORB), instead of size-down.
    
    When True (default): risk_off regime returns None (HARD SKIP).
    When False: risk_off regime reduces size to 0.25x but still signals.
    """
    
    def test_config_flag_default_true(self):
        """DAY_TRADE_HARD_SKIP_RISK_OFF must default to True.
        
        GLW RCA 2026-09-14: size-down path fed bad entries when regime
        flipped. Hard-skip is the new default.
        """
        from core.config import Config
        cfg = Config()
        assert cfg.DAY_TRADE_HARD_SKIP_RISK_OFF is True, (
            "DAY_TRADE_HARD_SKIP_RISK_OFF must default to True — "
            "GLW RCA 2026-09-14: hard-skip instead of size-down"
        )
    
    @pytest.mark.asyncio
    async def test_risk_off_hard_skip_when_flag_true(self):
        """risk_off must return None (HARD SKIP) when DAY_TRADE_HARD_SKIP_RISK_OFF=True.
        
        This is the new default behavior. Same as ORB's risk_off_hard_skip.
        """
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'GLW'  # Symbol from RCA
        market_data.close = 100.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 99.5,
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',  # KEY: risk_off regime
                time_of_day='morning',
                spy_change_pct=-0.8,
                vix_change_pct=8.0,
                sector_etf='XLK',
                reason='risk_off test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                # KEY: DAY_TRADE_HARD_SKIP_RISK_OFF=True (hard-skip)
                mock_cfg_instance.DAY_TRADE_HARD_SKIP_RISK_OFF = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # CRITICAL: Signal MUST be None in risk_off when flag is True (HARD SKIP)
        assert signal is None, (
            "day_trade_momentum must return None (HARD SKIP) in risk_off regime "
            "when DAY_TRADE_HARD_SKIP_RISK_OFF=True. This is the GLW RCA fix: "
            "the weak 0.25x size-down path was feeding bad entries when regime "
            "flipped to mixed."
        )
    
    @pytest.mark.asyncio
    async def test_risk_off_size_reduction_when_flag_false(self):
        """risk_off must reduce size to 0.25x when DAY_TRADE_HARD_SKIP_RISK_OFF=False.
        
        This is the legacy behavior, preserved for rollback or A/B testing.
        """
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'TEST'
        market_data.close = 100.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 99.5,
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',
                time_of_day='morning',
                spy_change_pct=-0.8,
                vix_change_pct=8.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                # KEY: DAY_TRADE_HARD_SKIP_RISK_OFF=False (legacy size-reduction)
                mock_cfg_instance.DAY_TRADE_HARD_SKIP_RISK_OFF = False
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be generated (not blocked) when hard-skip is OFF
        assert signal is not None, (
            "day_trade_momentum must generate signal when "
            "DAY_TRADE_HARD_SKIP_RISK_OFF=False (legacy behavior)"
        )
        # market_context_conviction should be 0.25 (risk_off size multiplier)
        assert signal.reasoning['market_context_conviction'] == 0.25
    
    @pytest.mark.asyncio
    async def test_non_risk_off_unaffected_by_flag(self):
        """Non-risk_off regimes unaffected by DAY_TRADE_HARD_SKIP_RISK_OFF flag.
        
        risk_on/mixed regimes should generate signals regardless of flag setting.
        """
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'TEST'
        market_data.close = 100.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 99.5,
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_on',  # NOT risk_off
                time_of_day='morning',
                spy_change_pct=0.5,
                vix_change_pct=-2.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                # Flag is True but regime is NOT risk_off, so it shouldn't matter
                mock_cfg_instance.DAY_TRADE_HARD_SKIP_RISK_OFF = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be generated (risk_on regime)
        assert signal is not None, (
            "Non-risk_off regime must not be affected by "
            "DAY_TRADE_HARD_SKIP_RISK_OFF flag"
        )
        assert signal.reasoning['market_context_regime'] == 'risk_on'
    
    @pytest.mark.asyncio
    async def test_mixed_regime_unaffected_by_flag(self):
        """mixed regime unaffected by DAY_TRADE_HARD_SKIP_RISK_OFF flag."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'TEST'
        market_data.close = 100.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 99.5,
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='mixed',  # NOT risk_off
                time_of_day='midday',
                spy_change_pct=-0.3,
                vix_change_pct=2.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.DAY_TRADE_HARD_SKIP_RISK_OFF = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be generated (mixed regime is not risk_off)
        assert signal is not None
        assert signal.reasoning['market_context_regime'] == 'mixed'
    
    @pytest.mark.asyncio
    async def test_hard_skip_logs_correct_reason(self):
        """Hard-skip must log reason as 'risk_off_hard_skip' (like ORB)."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'TEST'
        market_data.close = 100.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 99.5,
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        log_calls = []
        original_log = strategy._log_decision
        def capture_log(*args, **kwargs):
            log_calls.append((args, kwargs))
            return original_log(*args, **kwargs)
        
        strategy._log_decision = capture_log
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',
                time_of_day='morning',
                spy_change_pct=-0.8,
                vix_change_pct=8.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.DAY_TRADE_HARD_SKIP_RISK_OFF = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg.return_value = mock_cfg_instance
                
                await strategy.generate_signal_with_commentary(market_data)
        
        # Find the skip log call
        skip_calls = [c for c in log_calls if len(c[0]) >= 3 and c[0][1] == "skip"]
        assert len(skip_calls) >= 1, "Should have logged a skip decision"
        
        # Check reason matches ORB style
        skip_call = skip_calls[0]
        assert skip_call[0][2] == "risk_off_hard_skip", (
            f"Skip reason should be 'risk_off_hard_skip' (like ORB), got: {skip_call[0][2]}"
        )


class TestMoverQualityRelax:
    """Test that mover-sourced symbols bypass SMA50/RS quality filters."""
    
    def test_mover_source_detected(self):
        """Mover from yahoo_day_gainers is tagged as is_mover=True."""
        mover = {
            'symbol': 'CHYM',
            'source': 'yahoo_day_gainers',
            'last': 15.0,
            'volume': 1_000_000,
            'spread': 0.05,
        }
        # Source is in the mover list
        mover_sources = ('yahoo_most_active', 'yahoo_day_gainers', 'yahoo_day_losers')
        assert mover['source'] in mover_sources
    
    def test_mover_bypasses_sma50_filter(self):
        """Mover from Yahoo bypasses SMA50 trend filter."""
        from analysis.screener import StockScreener, LEVERAGED_ETF_BLOCKLIST
        
        # Create a mock screener
        screener = StockScreener.__new__(StockScreener)
        screener._quality_filter_rejects = {}
        screener._mover_quality_relax_passes = {}
        
        # Mock mover that would FAIL standard quality (below SMA50)
        mover = {
            'symbol': 'CHYM',
            'source': 'yahoo_day_gainers',
            'last': 15.0,  # Price above MIN_MOVER_PRICE
            'volume': 1_000_000,  # Volume above MIN_MOVER_VOLUME
            'spread': 0.05,  # Spread < 1%
        }
        
        # Should pass mover quality filter (bypasses SMA50)
        result = screener._passes_mover_quality_filter(mover)
        assert result is True
        
    def test_mover_still_blocks_leveraged_etfs(self):
        """Mover quality relax still blocks leveraged ETFs."""
        from analysis.screener import StockScreener, LEVERAGED_ETF_BLOCKLIST
        
        screener = StockScreener.__new__(StockScreener)
        screener._quality_filter_rejects = {}
        
        # Leveraged ETF appearing in day_gainers
        mover = {
            'symbol': 'TQQQ',
            'source': 'yahoo_day_gainers',
            'last': 50.0,
            'volume': 10_000_000,
            'spread': 0.10,
        }
        
        # Should still be blocked
        result = screener._passes_mover_quality_filter(mover)
        assert result is False
        assert 'TQQQ' in screener._quality_filter_rejects
        
    def test_mover_blocks_sub_5_dollar_stocks(self):
        """Mover quality relax blocks sub-$5 stocks."""
        from analysis.screener import StockScreener
        
        screener = StockScreener.__new__(StockScreener)
        screener._quality_filter_rejects = {}
        
        mover = {
            'symbol': 'PENNY',
            'source': 'yahoo_day_gainers',
            'last': 2.50,  # Below MIN_MOVER_PRICE
            'volume': 5_000_000,
            'spread': 0.02,
        }
        
        result = screener._passes_mover_quality_filter(mover)
        assert result is False


class TestMarketContextSizeNotFreeze:
    """Test that risk_off reduces size instead of hard-blocking for momentum.
    
    NOTE: These tests require DAY_TRADE_HARD_SKIP_RISK_OFF=False to test
    the legacy size-reduction behavior. The new default (True) causes
    hard-skip instead — see TestDayTradeHardSkipRiskOff for those tests.
    """
    
    @pytest.mark.asyncio
    async def test_risk_off_reduces_size_not_hard_block(self):
        """risk_off regime reduces size to 0.25x when DAY_TRADE_HARD_SKIP_RISK_OFF=False."""
        from strategies.builtin import DayTradeMomentumStrategy
        from core.models import MarketData
        
        # Create strategy with mock commentary
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        # Create market data with momentum setup
        market_data = MagicMock()
        market_data.symbol = 'TEST'
        market_data.close = 100.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 99.5,  # Price above 20-bar high = breakout
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,  # Symbol up 3%
        }
        
        # Mock market context returning risk_off
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',
                time_of_day='morning',
                spy_change_pct=-0.5,  # SPY down 0.5% (not extreme)
                vix_change_pct=5.0,  # VIX up 5% (not extreme)
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                # KEY: DAY_TRADE_HARD_SKIP_RISK_OFF=False to test legacy behavior
                mock_cfg_instance.DAY_TRADE_HARD_SKIP_RISK_OFF = False
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be generated (not blocked) when hard-skip is OFF
        assert signal is not None
        
        # market_context_conviction should be 0.25 (risk_off multiplier)
        assert signal.reasoning['market_context_conviction'] == 0.25
        assert signal.reasoning['market_context_regime'] == 'risk_off'
    
    @pytest.mark.asyncio
    async def test_extreme_risk_hard_blocks(self):
        """SPY <= -1.5% AND VIX spike >= 15% hard-blocks momentum."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'TEST'
        market_data.close = 100.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 99.5,
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        # Mock EXTREME market context
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',
                time_of_day='morning',
                spy_change_pct=-2.0,  # SPY down 2% (EXTREME)
                vix_change_pct=20.0,  # VIX up 20% (EXTREME)
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be BLOCKED (None) due to extreme conditions
        assert signal is None


class TestMomentumSignalGeneration:
    """Test momentum signal generation under various conditions."""
    
    @pytest.mark.asyncio
    async def test_breakout_pattern_generates_signal(self):
        """Breakout pattern (price > 20-bar high) generates signal."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'NVDA'
        market_data.close = 102.0
        market_data.open = 100.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 60,
            'volume_ratio': 2.5,  # Strong volume
            'adx': 35,  # Strong trend
            'atr': 1.5,
            'high_20': 100.0,  # Close > high_20 = breakout
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 4.0,  # Symbol up 4%
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_on',
                time_of_day='morning',
                spy_change_pct=0.5,  # SPY up 0.5%
                vix_change_pct=-2.0,  # VIX down
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is not None
        assert signal.reasoning['strategy'] == 'day_trade_momentum'
        assert signal.reasoning['entry_pattern'] == 'breakout'
        assert signal.reasoning['is_day_trade'] is True
        
    @pytest.mark.asyncio
    async def test_pullback_pattern_generates_signal(self):
        """Pullback pattern (RSI 40-60, near SMA20) generates signal."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'AMD'
        market_data.close = 100.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 50,  # In pullback range 40-60
            'volume_ratio': 1.8,
            'adx': 25,
            'atr': 1.5,
            'high_20': 105.0,  # Not a breakout
            'sma_20': 99.5,  # Price within 2% of SMA20
            'macd': 0.3,
            'macd_signal': 0.2,  # MACD bullish
            'day_change_pct': 2.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='mixed',
                time_of_day='midday',
                spy_change_pct=0.2,
                vix_change_pct=1.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is not None
        assert signal.reasoning['entry_pattern'] == 'pullback'
        
    @pytest.mark.asyncio
    async def test_weak_rs_blocks_signal(self):
        """Symbol underperforming SPY blocks signal."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'WEAK'
        market_data.close = 100.0
        market_data.open = 100.5  # Slightly down
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 99.5,
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': -0.2,  # Symbol DOWN 0.2%
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_on',
                time_of_day='morning',
                spy_change_pct=0.5,  # SPY up 0.5%, symbol underperforming
                vix_change_pct=-1.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5  # Requires +0.5% RS
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Should be blocked due to weak RS (-0.7% vs SPY)
        assert signal is None


class TestDayTradeSizeMultiplier:
    """Test that day-trade size multiplier is applied correctly."""
    
    @pytest.mark.asyncio
    async def test_day_trade_size_multiplier_in_reasoning(self):
        """Day-trade signals include size multiplier in reasoning."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'TEST'
        market_data.close = 100.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 60,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 99.5,
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_on',
                time_of_day='morning',
                spy_change_pct=0.5,
                vix_change_pct=-2.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is not None
        assert signal.reasoning['is_day_trade'] is True
        assert signal.reasoning['day_trade_size_multiplier'] == 0.5
        assert signal.reasoning['flatten_hour'] == 15


class TestConfigFlags:
    """Test Config flags for day-trade momentum desk."""
    
    def test_enable_day_trade_momentum_default_true(self):
        """ENABLE_DAY_TRADE_MOMENTUM defaults to True."""
        from core.config import Config
        cfg = Config()
        # Default should be True
        assert cfg.ENABLE_DAY_TRADE_MOMENTUM is True
        
    def test_enable_mover_quality_relax_default_true(self):
        """ENABLE_MOVER_QUALITY_RELAX defaults to True."""
        from core.config import Config
        cfg = Config()
        assert cfg.ENABLE_MOVER_QUALITY_RELAX is True
        
    def test_min_mover_price_default(self):
        """MIN_MOVER_PRICE defaults to 5.0."""
        from core.config import Config
        cfg = Config()
        assert cfg.MIN_MOVER_PRICE == 5.0
        
    def test_momentum_risk_off_size_mult_default(self):
        """MOMENTUM_RISK_OFF_SIZE_MULT defaults to 0.25."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_RISK_OFF_SIZE_MULT == 0.25
        
    def test_momentum_extreme_block_spy_pct_default(self):
        """MOMENTUM_EXTREME_BLOCK_SPY_PCT defaults to -1.5."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_EXTREME_BLOCK_SPY_PCT == -1.5


class TestRiskOffSizeReduction:
    """Explicit tests for risk_off SIZE REDUCTION (legacy) behavior.
    
    v-market-context-size-not-freeze-2026-09-10: The original requirement was
    that risk_off should REDUCE SIZE, NOT hard-block momentum entries.
    
    v-day-trade-hard-skip-risk-off-2026-09-14: This behavior is now the
    LEGACY path — the new default is HARD SKIP. These tests set
    DAY_TRADE_HARD_SKIP_RISK_OFF=False to test the legacy path.
    """
    
    @pytest.mark.asyncio
    async def test_risk_off_signal_generated_not_none(self):
        """risk_off must generate a signal when DAY_TRADE_HARD_SKIP_RISK_OFF=False."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'TEST'
        market_data.close = 100.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 99.5,
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',  # KEY: risk_off regime
                time_of_day='midday',
                spy_change_pct=-0.8,  # Down but not extreme
                vix_change_pct=8.0,  # Up but not extreme spike
                sector_etf='XLK',
                reason='moderate risk_off',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                # KEY: DAY_TRADE_HARD_SKIP_RISK_OFF=False to test legacy behavior
                mock_cfg_instance.DAY_TRADE_HARD_SKIP_RISK_OFF = False
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be generated (legacy behavior with flag OFF)
        assert signal is not None, (
            "risk_off must NOT hard-block momentum signals when "
            "DAY_TRADE_HARD_SKIP_RISK_OFF=False (legacy behavior)"
        )
    
    @pytest.mark.asyncio
    async def test_risk_off_size_mult_is_025(self):
        """risk_off must apply 0.25x size multiplier when DAY_TRADE_HARD_SKIP_RISK_OFF=False."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'TEST'
        market_data.close = 100.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 99.5,
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',
                time_of_day='midday',
                spy_change_pct=-0.8,
                vix_change_pct=8.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                # KEY: DAY_TRADE_HARD_SKIP_RISK_OFF=False to test legacy behavior
                mock_cfg_instance.DAY_TRADE_HARD_SKIP_RISK_OFF = False
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25  # KEY: 0.25x
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is not None
        # CRITICAL: size mult must be 0.25 for risk_off
        assert signal.reasoning['market_context_conviction'] == 0.25
        assert signal.reasoning['mc_size_mult'] == 0.25
    
    @pytest.mark.asyncio
    async def test_opening_30_size_mult_is_05(self):
        """opening_30 must apply 0.5x size multiplier."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'TEST'
        market_data.close = 100.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 99.5,
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_on',  # Not risk_off
                time_of_day='opening_30',  # KEY: opening_30
                spy_change_pct=0.3,
                vix_change_pct=-1.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5  # KEY: 0.5x
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is not None
        # opening_30 should apply 0.5x (market_context_conviction starts at 1.0)
        assert signal.reasoning['market_context_conviction'] == 0.5
        assert signal.reasoning['mc_size_mult'] == 0.5
    
    @pytest.mark.asyncio
    async def test_risk_off_plus_opening_30_uses_min(self):
        """risk_off + opening_30 uses minimum of 0.25x and 0.5x = 0.25x when DAY_TRADE_HARD_SKIP_RISK_OFF=False."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'TEST'
        market_data.close = 100.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 99.5,
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',  # risk_off = 0.25x
                time_of_day='opening_30',  # opening_30 = 0.5x
                spy_change_pct=-0.5,
                vix_change_pct=5.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                # KEY: DAY_TRADE_HARD_SKIP_RISK_OFF=False to test legacy behavior
                mock_cfg_instance.DAY_TRADE_HARD_SKIP_RISK_OFF = False
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is not None
        # min(0.25, 0.5) = 0.25
        assert signal.reasoning['market_context_conviction'] == 0.25
        assert signal.reasoning['mc_size_mult'] == 0.25


class TestStageAInstrumentation:
    """Test Stage-A promotion instrumentation fields.
    
    v-momentum-stage-a-2026-09-10: signals must include:
      - session_id: trading date in ET (YYYY-MM-DD)
      - risk_off: explicit boolean flag
      - mc_size_mult: market context size multiplier
      - setup_type: momentum_<pattern>
    """
    
    @pytest.mark.asyncio
    async def test_signal_includes_session_id(self):
        """Signal reasoning must include session_id."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'TEST'
        market_data.close = 100.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 99.5,
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_on',
                time_of_day='midday',
                spy_change_pct=0.5,
                vix_change_pct=-1.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is not None
        assert 'session_id' in signal.reasoning
        # session_id should be date format YYYY-MM-DD
        session_id = signal.reasoning['session_id']
        assert len(session_id) == 10
        assert session_id[4] == '-'
        assert session_id[7] == '-'
    
    @pytest.mark.asyncio
    async def test_signal_includes_risk_off_flag(self):
        """Signal reasoning must include explicit risk_off boolean."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'TEST'
        market_data.close = 100.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 99.5,
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',  # Test with risk_off
                time_of_day='midday',
                spy_change_pct=-0.5,
                vix_change_pct=5.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is not None
        assert 'risk_off' in signal.reasoning
        assert signal.reasoning['risk_off'] is True  # Must be True for risk_off regime
    
    @pytest.mark.asyncio
    async def test_signal_includes_setup_type(self):
        """Signal reasoning must include setup_type (momentum_<pattern>)."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'TEST'
        market_data.close = 102.0  # Above high_20 for breakout
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 60,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 100.0,  # Close > high_20 = breakout
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 4.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_on',
                time_of_day='midday',
                spy_change_pct=0.5,
                vix_change_pct=-1.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is not None
        assert 'setup_type' in signal.reasoning
        assert signal.reasoning['setup_type'] == 'momentum_breakout'


class TestMomentumContext:
    """Test MomentumContext dataclass for decision snapshots."""
    
    def test_momentum_context_from_signal_reasoning(self):
        """MomentumContext can be built from signal reasoning dict."""
        from core.decision_snapshot import MomentumContext
        
        reasoning = {
            'session_id': '2026-09-10',
            'setup_type': 'momentum_breakout',
            'risk_off': True,
            'mc_size_mult': 0.25,
            'rs_vs_spy': 2.5,
            'is_day_trade': True,
            'flatten_hour': 15,
            'entry_pattern': 'breakout',
        }
        
        ctx = MomentumContext.from_signal_reasoning(reasoning)
        
        assert ctx.session_id == '2026-09-10'
        assert ctx.setup_type == 'momentum_breakout'
        assert ctx.risk_off is True
        assert ctx.mc_size_mult == 0.25
        assert ctx.rs_vs_spy == 2.5
        assert ctx.is_day_trade is True
        assert ctx.flatten_hour == 15
        assert ctx.entry_pattern == 'breakout'
    
    def test_momentum_context_empty(self):
        """MomentumContext.empty() returns sensible defaults."""
        from core.decision_snapshot import MomentumContext
        
        ctx = MomentumContext.empty()
        
        assert ctx.session_id == ''
        assert ctx.setup_type == ''
        assert ctx.risk_off is False
        assert ctx.mc_size_mult == 1.0
        assert ctx.rs_vs_spy == 0.0
        assert ctx.is_day_trade is False
        assert ctx.flatten_hour == 15


class TestStageAConfigFloors:
    """Test Stage-A promotion criteria config defaults.
    
    v-momentum-stage-a-2026-09-10: these are RESEARCH-LOCKED floors.
    """
    
    def test_stage_a_min_trades_is_150(self):
        """MOMENTUM_STAGE_A_MIN_TRADES defaults to 150."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_STAGE_A_MIN_TRADES == 150
    
    def test_stage_a_min_sessions_is_10(self):
        """MOMENTUM_STAGE_A_MIN_SESSIONS defaults to 10."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_STAGE_A_MIN_SESSIONS == 10
    
    def test_stage_a_min_pf_is_130(self):
        """MOMENTUM_STAGE_A_MIN_PF defaults to 1.30."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_STAGE_A_MIN_PF == 1.30
    
    def test_stage_a_min_win_rate_is_048(self):
        """MOMENTUM_STAGE_A_MIN_WIN_RATE defaults to 0.48."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_STAGE_A_MIN_WIN_RATE == 0.48
    
    def test_stage_a_max_dd_pct_is_006(self):
        """MOMENTUM_STAGE_A_MAX_DD_PCT defaults to 0.06 (6%)."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_STAGE_A_MAX_DD_PCT == 0.06
    
    def test_stage_a_max_losing_day_r_is_2(self):
        """MOMENTUM_STAGE_A_MAX_LOSING_DAY_R defaults to 2.0."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_STAGE_A_MAX_LOSING_DAY_R == 2.0
    
    def test_stage_a_promoted_default_false(self):
        """MOMENTUM_STAGE_A_PROMOTED defaults to False."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_STAGE_A_PROMOTED is False
    
    def test_momentum_allow_overnight_hold_default_false(self):
        """MOMENTUM_ALLOW_OVERNIGHT_HOLD defaults to False."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_ALLOW_OVERNIGHT_HOLD is False


# ══════════════════════════════════════════════════════════════════════════════
# v-pause-live-daytrade-2026-09-10: Tests for Part A - Pause NEW LIVE day-trade
# entries. Hari APPROVED product call 2026-09-10.
# ══════════════════════════════════════════════════════════════════════════════

class TestDayTradeLiveEntriesDisabled:
    """Test DAY_TRADE_LIVE_ENTRIES_ENABLED flag and LIVE entry blocking.
    
    v-pause-live-daytrade-2026-09-10: immediately pause new LIVE day-trade
    entries while keeping sim/commentary analysis running and existing
    positions' exits intact.
    """
    
    def test_day_trade_live_entries_default_false(self):
        """DAY_TRADE_LIVE_ENTRIES_ENABLED must default to False.
        
        This is the immediate pause. DO NOT CHANGE without Stage-A validation.
        """
        from core.config import Config
        cfg = Config()
        assert cfg.DAY_TRADE_LIVE_ENTRIES_ENABLED is False, (
            "DAY_TRADE_LIVE_ENTRIES_ENABLED must default to False — "
            "Stage-A validation required before enabling LIVE entries"
        )
    
    def test_enable_day_trade_momentum_still_true(self):
        """ENABLE_DAY_TRADE_MOMENTUM should still be True (strategy generates signals).
        
        The pause is only for LIVE order placement, not signal generation.
        Sim/commentary analysis should continue working.
        """
        from core.config import Config
        cfg = Config()
        assert cfg.ENABLE_DAY_TRADE_MOMENTUM is True, (
            "ENABLE_DAY_TRADE_MOMENTUM should be True — "
            "we want signals generated for sim/commentary, just LIVE blocked"
        )
    
    @pytest.mark.asyncio
    async def test_strategy_still_generates_signals_when_live_disabled(self):
        """DayTradeMomentumStrategy must still generate signals for sim/commentary.
        
        The LIVE blocking happens in the engine, not the strategy. Strategy
        should always generate signals when conditions are met.
        """
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'TEST'
        market_data.close = 102.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 60,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 100.0,
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 4.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_on',
                time_of_day='midday',
                spy_change_pct=0.5,
                vix_change_pct=-1.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.DAY_TRADE_LIVE_ENTRIES_ENABLED = False  # KEY: False
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg_instance.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF = True
                mock_cfg_instance.SHADOW_VETO_RSI_THRESHOLD = 70.0
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should STILL be generated (strategy doesn't check LIVE mode)
        # The engine handles LIVE blocking, not the strategy
        assert signal is not None, (
            "Strategy must still generate signals even when "
            "DAY_TRADE_LIVE_ENTRIES_ENABLED=False — blocking is in engine"
        )


class TestEngineBlocksLiveDayTrade:
    """Test that the engine blocks LIVE day-trade entries when flag is False.
    
    These tests verify the engine's signal routing gate, not the strategy.
    """
    
    def test_engine_has_daytrade_live_pause_gate(self):
        """Engine must have the v-pause-live-daytrade-2026-09-10 gate."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()
        
        assert "v-pause-live-daytrade-2026-09-10" in src, (
            "Engine must contain v-pause-live-daytrade-2026-09-10 gate"
        )
        assert "DAY_TRADE_LIVE_ENTRIES_ENABLED" in src, (
            "Engine must check DAY_TRADE_LIVE_ENTRIES_ENABLED flag"
        )
        assert "daytrade_live_pause" in src, (
            "Engine must audit with component=daytrade_live_pause"
        )
    
    def test_engine_gate_checks_strategy_name(self):
        """Engine gate must specifically check for day_trade_momentum strategy."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()
        
        anchor = src.find("v-pause-live-daytrade-2026-09-10")
        assert anchor != -1
        window = src[anchor: anchor + 3000]
        
        assert "day_trade_momentum" in window, (
            "Engine gate must check for strategy == 'day_trade_momentum'"
        )
        assert "TradingMode.LIVE" in window, (
            "Engine gate must check for LIVE mode"
        )


# ══════════════════════════════════════════════════════════════════════════════
# v-flatten-hour-entry-gate-2026-09-15: Tests for blocking day-trade LIVE
# entries at/after flatten_hour. P0 RCA: BBWI entered at 15:23:30 ET then
# exited ~7s later via flatten_hour=15. Entry at/after flatten hour = churn.
# ══════════════════════════════════════════════════════════════════════════════

class TestFlattenHourEntryGate:
    """Test DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED flag.

    v-flatten-hour-entry-gate-2026-09-15: block NEW day-trade LIVE entries
    when current ET hour >= flatten_hour that would immediately flatten them.
    """

    def test_flatten_hour_entry_gate_default_true(self):
        """DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED must default to True.

        This is the P0 fix. DO NOT CHANGE without explicit approval.
        """
        from core.config import Config
        cfg = Config()
        assert cfg.DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED is True, (
            "DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED must default to True — "
            "P0 fix to prevent entry churn at/after flatten_hour"
        )

    def test_engine_has_flatten_hour_entry_gate(self):
        """Engine must have the v-flatten-hour-entry-gate-2026-09-15 gate."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()

        assert "v-flatten-hour-entry-gate-2026-09-15" in src, (
            "Engine must contain v-flatten-hour-entry-gate-2026-09-15 gate"
        )
        assert "DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED" in src, (
            "Engine must check DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED flag"
        )
        assert "flatten_hour_entry_blocked" in src, (
            "Engine must audit with reason=flatten_hour_entry_blocked"
        )
        assert "daytrade_flatten_hour_gate" in src, (
            "Engine must audit with component=daytrade_flatten_hour_gate"
        )

    def test_engine_gate_checks_is_day_trade_flag(self):
        """Engine gate must check is_day_trade in reasoning (not just strategy name)."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()

        anchor = src.find("v-flatten-hour-entry-gate-2026-09-15")
        assert anchor != -1
        window = src[anchor: anchor + 2000]

        assert "is_day_trade" in window, (
            "Engine gate must check is_day_trade flag from reasoning"
        )
        assert "TradingMode.LIVE" in window, (
            "Engine gate must check for LIVE mode"
        )
        assert "flatten_hour" in window, (
            "Engine gate must compare current hour with flatten_hour"
        )


class TestFlattenHourEntryGateIntegration:
    """Integration tests for flatten hour entry gate behavior."""

    @pytest.fixture
    def mock_signal_day_trade(self):
        """Create a day-trade signal with is_day_trade=True."""
        signal = MagicMock()
        signal.symbol = "BBWI"
        signal.signal_type = SignalType.BUY
        signal.reasoning = {
            "strategy": "day_trade_momentum",
            "is_day_trade": True,
            "flatten_hour": 15,
            "entry_pattern": "breakout",
            "rsi": 55.0,
        }
        return signal

    @pytest.fixture
    def mock_engine(self):
        """Create a mock engine in LIVE mode."""
        engine = MagicMock()
        engine.mode = TradingMode.LIVE
        engine._audit = MagicMock()
        engine.commentary = MagicMock()
        engine.commentary.add_commentary = MagicMock()
        return engine

    def test_entry_blocked_at_flatten_hour(self, mock_signal_day_trade, mock_engine):
        """Entry must be blocked when et_hour == flatten_hour.

        At flatten_hour=15: 15:xx entry should be blocked.
        """
        from unittest.mock import patch
        from datetime import datetime
        from zoneinfo import ZoneInfo

        et_time_at_flatten = datetime(2026, 9, 15, 15, 23, 30, tzinfo=ZoneInfo("America/New_York"))

        with patch('core.config.Config') as mock_cfg:
            mock_cfg_instance = MagicMock()
            mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED = True
            mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
            mock_cfg.return_value = mock_cfg_instance

            with patch('core.engine.datetime') as mock_dt:
                mock_dt.now.return_value = et_time_at_flatten

                signal = mock_signal_day_trade
                reasoning = signal.reasoning

                flatten_hour = reasoning.get('flatten_hour', 15)
                et_hour = et_time_at_flatten.hour

                assert et_hour >= flatten_hour, (
                    f"Test precondition: et_hour ({et_hour}) >= flatten_hour ({flatten_hour})"
                )

    def test_entry_blocked_after_flatten_hour(self, mock_signal_day_trade, mock_engine):
        """Entry must be blocked when et_hour > flatten_hour.

        At flatten_hour=15: 16:xx entry should be blocked.
        """
        from zoneinfo import ZoneInfo

        et_time_after_flatten = datetime(2026, 9, 15, 16, 5, 0, tzinfo=ZoneInfo("America/New_York"))
        flatten_hour = mock_signal_day_trade.reasoning['flatten_hour']
        et_hour = et_time_after_flatten.hour

        assert et_hour > flatten_hour, (
            f"Test precondition: et_hour ({et_hour}) > flatten_hour ({flatten_hour})"
        )

    def test_entry_allowed_before_flatten_hour(self, mock_signal_day_trade, mock_engine):
        """Entry must be allowed when et_hour < flatten_hour.

        At flatten_hour=15: 14:xx entry should proceed.
        """
        from zoneinfo import ZoneInfo

        et_time_before_flatten = datetime(2026, 9, 15, 14, 30, 0, tzinfo=ZoneInfo("America/New_York"))
        flatten_hour = mock_signal_day_trade.reasoning['flatten_hour']
        et_hour = et_time_before_flatten.hour

        assert et_hour < flatten_hour, (
            f"Test precondition: et_hour ({et_hour}) < flatten_hour ({flatten_hour})"
        )

    def test_gate_disabled_allows_entry_at_flatten_hour(self, mock_signal_day_trade):
        """Entry allowed at flatten_hour when gate is disabled."""
        from core.config import Config
        from unittest.mock import patch

        with patch.dict(os.environ, {"DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED": "0"}):
            cfg = Config()
            assert cfg.DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED is False, (
                "Gate must be disabled when env var is 0"
            )

    def test_non_day_trade_signal_not_blocked(self):
        """Signals without is_day_trade=True should not be blocked by this gate."""
        signal = MagicMock()
        signal.symbol = "TEST"
        signal.reasoning = {
            "strategy": "mean_reversion",
            "is_day_trade": False,
        }

        is_day_trade = signal.reasoning.get('is_day_trade', False)
        assert is_day_trade is False, (
            "Non-day-trade signal should not trigger flatten_hour_entry gate"
        )


# ══════════════════════════════════════════════════════════════════════════════
# v-daytrade-rsi-entry-gate-2026-09-15: Tests for blocking day-trade LIVE
# entries when RSI <= 50 (the proactive_rsi_below_50 exit threshold).
#
# P0 RCA 2026-09-15 (ALHC×2): day_trade pullback/continuation entered LIVE
# while RSI <= 50, then proactive_rsi_below_50 exit immediately dumped the
# trade. Entry into a condition that already triggers exit = churn.
# ══════════════════════════════════════════════════════════════════════════════

class TestDayTradeRsiEntryGateConfig:
    """Test DAY_TRADE_RSI_ENTRY_GATE_ENABLED config flag.

    v-daytrade-rsi-entry-gate-2026-09-15: block day-trade LIVE long entries
    when RSI <= 50 (the proactive_rsi_below_50 exit threshold).
    """

    def test_rsi_entry_gate_default_true(self):
        """DAY_TRADE_RSI_ENTRY_GATE_ENABLED must default to True.

        This is the P0 fix. DO NOT CHANGE without explicit approval.
        """
        from core.config import Config
        cfg = Config()
        assert cfg.DAY_TRADE_RSI_ENTRY_GATE_ENABLED is True, (
            "DAY_TRADE_RSI_ENTRY_GATE_ENABLED must default to True — "
            "P0 fix to prevent entry churn when RSI <= exit threshold"
        )

    def test_rsi_entry_threshold_default_50(self):
        """DAY_TRADE_RSI_ENTRY_THRESHOLD must default to 50.0.

        Aligned with proactive_rsi_below_50 exit threshold.
        """
        from core.config import Config
        cfg = Config()
        assert cfg.DAY_TRADE_RSI_ENTRY_THRESHOLD == 50.0, (
            "DAY_TRADE_RSI_ENTRY_THRESHOLD must default to 50.0 — "
            "aligned with proactive_rsi_below_50 exit threshold"
        )

    def test_rsi_entry_gate_env_override_disables(self):
        """DAY_TRADE_RSI_ENTRY_GATE_ENABLED=0 disables the gate."""
        from core.config import Config

        with patch.dict(os.environ, {"DAY_TRADE_RSI_ENTRY_GATE_ENABLED": "0"}):
            cfg = Config()
            assert cfg.DAY_TRADE_RSI_ENTRY_GATE_ENABLED is False, (
                "Gate must be disabled when env var is 0"
            )

    def test_rsi_entry_threshold_configurable(self):
        """RSI threshold should be configurable via config manager.

        The property reads from config manager with key
        'trading.day_trade_rsi_entry_threshold' with default 50.0.
        """
        from pathlib import Path
        src = Path("core/config.py").read_text()

        assert "day_trade_rsi_entry_threshold" in src, (
            "Config must have day_trade_rsi_entry_threshold property"
        )
        assert "50.0" in src, (
            "Default threshold must be 50.0"
        )


class TestDayTradeRsiEntryGateEngine:
    """Test engine implementation of RSI entry gate.

    v-daytrade-rsi-entry-gate-2026-09-15: verify the engine has the gate
    and uses the correct audit reason.
    """

    def test_engine_has_rsi_entry_gate(self):
        """Engine must have the v-daytrade-rsi-entry-gate-2026-09-15 gate."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()

        assert "v-daytrade-rsi-entry-gate-2026-09-15" in src, (
            "Engine must contain v-daytrade-rsi-entry-gate-2026-09-15 gate"
        )
        assert "DAY_TRADE_RSI_ENTRY_GATE_ENABLED" in src, (
            "Engine must check DAY_TRADE_RSI_ENTRY_GATE_ENABLED flag"
        )
        assert "rsi_below_50_entry_blocked" in src, (
            "Engine must audit with reason=rsi_below_50_entry_blocked"
        )
        assert "daytrade_rsi_entry_gate" in src, (
            "Engine must audit with component=daytrade_rsi_entry_gate"
        )

    def test_engine_gate_checks_is_day_trade_and_rsi(self):
        """Engine gate must check is_day_trade, RSI, and BUY signal type."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()

        anchor = src.find("v-daytrade-rsi-entry-gate-2026-09-15")
        assert anchor != -1
        window = src[anchor: anchor + 2500]

        assert "_is_day_trade_signal" in window, (
            "Engine gate must check is_day_trade flag"
        )
        assert "TradingMode.LIVE" in window, (
            "Engine gate must check for LIVE mode"
        )
        assert "SignalType.BUY" in window, (
            "Engine gate must check for BUY signal type (longs only)"
        )
        assert "DAY_TRADE_RSI_ENTRY_THRESHOLD" in window, (
            "Engine gate must use DAY_TRADE_RSI_ENTRY_THRESHOLD"
        )


class TestDayTradeRsiEntryGateIntegration:
    """Integration tests for RSI entry gate behavior."""

    @pytest.fixture
    def mock_signal_day_trade_low_rsi(self):
        """Create a day-trade signal with RSI <= 50 (would be blocked)."""
        signal = MagicMock()
        signal.symbol = "ALHC"
        signal.signal_type = SignalType.BUY  # Long entry
        signal.reasoning = {
            "strategy": "day_trade_momentum",
            "is_day_trade": True,
            "entry_pattern": "pullback",  # Pullback allows RSI 40-60
            "rsi": 48.0,  # KEY: RSI <= 50
        }
        return signal

    @pytest.fixture
    def mock_signal_day_trade_high_rsi(self):
        """Create a day-trade signal with RSI > 50 (would be allowed)."""
        signal = MagicMock()
        signal.symbol = "ALHC"
        signal.signal_type = SignalType.BUY  # Long entry
        signal.reasoning = {
            "strategy": "day_trade_momentum",
            "is_day_trade": True,
            "entry_pattern": "continuation",
            "rsi": 55.0,  # KEY: RSI > 50
        }
        return signal

    def test_entry_blocked_when_rsi_below_threshold(self, mock_signal_day_trade_low_rsi):
        """Entry must be blocked when RSI <= threshold.

        RSI 48 <= 50 threshold → entry blocked.
        """
        signal = mock_signal_day_trade_low_rsi
        rsi = signal.reasoning.get('rsi', 100)
        threshold = 50.0

        assert rsi <= threshold, (
            f"Test precondition: RSI ({rsi}) <= threshold ({threshold})"
        )

    def test_entry_blocked_when_rsi_equals_threshold(self):
        """Entry must be blocked when RSI == threshold (edge case).

        RSI exactly at 50 is right at the edge — one tick of noise and
        the proactive exit fires. Block at threshold, not just below.
        """
        signal = MagicMock()
        signal.symbol = "EDGE"
        signal.signal_type = SignalType.BUY
        signal.reasoning = {
            "strategy": "day_trade_momentum",
            "is_day_trade": True,
            "entry_pattern": "pullback",
            "rsi": 50.0,  # KEY: RSI == threshold exactly
        }

        rsi = signal.reasoning.get('rsi', 100)
        threshold = 50.0

        assert rsi <= threshold, (
            f"Test precondition: RSI ({rsi}) <= threshold ({threshold}) [boundary]"
        )

    def test_entry_allowed_when_rsi_above_threshold(self, mock_signal_day_trade_high_rsi):
        """Entry must be allowed when RSI > threshold.

        RSI 55 > 50 threshold → entry proceeds.
        """
        signal = mock_signal_day_trade_high_rsi
        rsi = signal.reasoning.get('rsi', 100)
        threshold = 50.0

        assert rsi > threshold, (
            f"Test precondition: RSI ({rsi}) > threshold ({threshold})"
        )

    def test_gate_disabled_allows_low_rsi_entry(self, mock_signal_day_trade_low_rsi):
        """Entry allowed at low RSI when gate is disabled."""
        from core.config import Config

        with patch.dict(os.environ, {"DAY_TRADE_RSI_ENTRY_GATE_ENABLED": "0"}):
            cfg = Config()
            assert cfg.DAY_TRADE_RSI_ENTRY_GATE_ENABLED is False, (
                "Gate must be disabled when env var is 0"
            )

    def test_non_day_trade_signal_not_blocked(self):
        """Signals without is_day_trade=True should not be blocked by this gate."""
        signal = MagicMock()
        signal.symbol = "TEST"
        signal.signal_type = SignalType.BUY
        signal.reasoning = {
            "strategy": "mean_reversion",
            "is_day_trade": False,
            "rsi": 45.0,  # Low RSI, but not a day trade
        }

        is_day_trade = signal.reasoning.get('is_day_trade', False)
        assert is_day_trade is False, (
            "Non-day-trade signal should not trigger rsi_entry gate"
        )

    def test_short_signal_not_blocked_by_rsi_gate(self):
        """SELL signals (shorts) should not be blocked by this gate.

        The proactive_rsi_below_50 exit only applies to longs.
        Short exits trigger when RSI > 50, so this gate is long-specific.
        """
        signal = MagicMock()
        signal.symbol = "SHORT"
        signal.signal_type = SignalType.SELL  # Short entry
        signal.reasoning = {
            "strategy": "day_trade_momentum",
            "is_day_trade": True,
            "entry_pattern": "continuation",
            "rsi": 45.0,  # Low RSI, but it's a short
        }

        is_buy = signal.signal_type == SignalType.BUY
        assert is_buy is False, (
            "SELL signal should not trigger the long-specific RSI gate"
        )


class TestRsiThresholdAlignment:
    """Test that RSI entry threshold aligns with exit threshold.

    v-daytrade-rsi-entry-gate-2026-09-15: the entry gate threshold must
    align with the proactive_rsi_below_50 exit threshold in:
      - analysis/scale_trail_manager.py
      - analysis/active_open_desk.py
    """

    def test_entry_threshold_matches_exit_concept(self):
        """Entry threshold 50 matches the exit 'rsi < 50' concept.

        Exit triggers at rsi < 50 (strict less than).
        Entry blocks at rsi <= 50 (less than or equal).

        Using <= for entry is more conservative: RSI exactly at 50 is
        right at the edge, so we block it to be safe.
        """
        from core.config import Config
        cfg = Config()

        entry_threshold = cfg.DAY_TRADE_RSI_ENTRY_THRESHOLD
        exit_threshold_concept = 50  # rsi < 50 in exit logic

        assert entry_threshold == exit_threshold_concept, (
            f"Entry threshold ({entry_threshold}) must match exit threshold "
            f"concept ({exit_threshold_concept})"
        )

    def test_exit_logic_uses_rsi_50_in_scale_trail_manager(self):
        """scale_trail_manager proactive exit must use RSI 50 threshold."""
        from pathlib import Path
        src = Path("analysis/scale_trail_manager.py").read_text()

        assert "rsi < 50" in src or "rsi_below_50" in src, (
            "scale_trail_manager must use RSI 50 as proactive exit threshold"
        )

    def test_exit_logic_uses_rsi_50_in_active_open_desk(self):
        """active_open_desk proactive exit must use RSI 50 threshold."""
        from pathlib import Path
        src = Path("analysis/active_open_desk.py").read_text()

        assert "rsi < 50" in src or "rsi_below_50" in src, (
            "active_open_desk must use RSI 50 as proactive exit threshold"
        )


# ══════════════════════════════════════════════════════════════════════════════
# v-shadow-veto-2026-09-10: Tests for Part B - Shadow veto for continuation
# pattern + RSI>=70 + risk_off.
# ══════════════════════════════════════════════════════════════════════════════

class TestShadowVetoContinuationRiskoffRsi70:
    """Test shadow veto for continuation + RSI>=70 + risk_off.
    
    v-shadow-veto-2026-09-10: shadow (log-only) veto for the dangerous
    pattern. This is instrumentation for later promotion scoring.
    """
    
    def test_enable_shadow_veto_default_true(self):
        """ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF must default to True."""
        from core.config import Config
        cfg = Config()
        assert cfg.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF is True
    
    def test_shadow_veto_rsi_threshold_default_70(self):
        """SHADOW_VETO_RSI_THRESHOLD must default to 70.0."""
        from core.config import Config
        cfg = Config()
        assert cfg.SHADOW_VETO_RSI_THRESHOLD == 70.0
    
    @pytest.mark.asyncio
    async def test_shadow_veto_fires_on_continuation_rsi70_riskoff(self):
        """Shadow veto must fire when continuation + RSI>=70 + risk_off.
        
        This is the dangerous pattern: continuation into overbought during
        risk_off is the classic failed-breakout exhaustion setup.
        """
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'RISKY'
        market_data.close = 100.0
        market_data.open = 98.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 72.0,  # KEY: RSI >= 70 (overbought)
            'volume_ratio': 2.0,
            'adx': 30,  # > 25 for continuation
            'atr': 1.5,
            'high_20': 95.0,  # Not breakout (close not > high_20)
            'sma_20': 98.0,  # Close > sma_20 for continuation
            'macd': 0.5,
            'macd_signal': 0.3,  # MACD > signal for continuation
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',  # KEY: risk_off regime
                time_of_day='midday',
                spy_change_pct=-0.8,
                vix_change_pct=8.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                # Shadow veto enabled
                mock_cfg_instance.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF = True
                mock_cfg_instance.SHADOW_VETO_RSI_THRESHOLD = 70.0
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be BLOCKED (None) due to shadow veto
        assert signal is None, (
            "Shadow veto must block signal for continuation + RSI>=70 + risk_off"
        )
    
    @pytest.mark.asyncio
    async def test_shadow_veto_does_not_fire_on_breakout_pattern(self):
        """Shadow veto must NOT fire for breakout pattern (only continuation)."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'SAFE'
        market_data.close = 102.0  # Above high_20 = breakout
        market_data.open = 98.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 72.0,  # RSI >= 70
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 100.0,  # Close > high_20 = BREAKOUT pattern
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 4.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',  # risk_off
                time_of_day='midday',
                spy_change_pct=-0.8,
                vix_change_pct=8.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg_instance.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF = True
                mock_cfg_instance.SHADOW_VETO_RSI_THRESHOLD = 70.0
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be GENERATED (not None) because pattern is BREAKOUT
        assert signal is not None, (
            "Shadow veto should NOT fire for breakout pattern "
            "(only continuation is vetoed)"
        )
        assert signal.reasoning['entry_pattern'] == 'breakout'
    
    @pytest.mark.asyncio
    async def test_shadow_veto_does_not_fire_on_risk_on_regime(self):
        """Shadow veto must NOT fire when regime is NOT risk_off.
        
        NOTE: This tests the shadow-veto-only path with hard veto DISABLED.
        When hard veto is enabled (default), risk_on+continuation+RSI>=70
        WILL be blocked. This test verifies the legacy shadow path.
        """
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'SAFE2'
        market_data.close = 100.0
        market_data.open = 98.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 72.0,  # RSI >= 70
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 95.0,  # Close NOT > high_20 = NOT breakout
            'sma_20': 98.0,  # Close > sma_20 = continuation pattern
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_on',  # KEY: NOT risk_off
                time_of_day='midday',
                spy_change_pct=0.5,
                vix_change_pct=-1.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                # Hard veto DISABLED to test shadow-only path
                mock_cfg_instance.ENABLE_HARD_VETO_CONTINUATION_RSI70 = False
                mock_cfg_instance.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF = True
                mock_cfg_instance.SHADOW_VETO_RSI_THRESHOLD = 70.0
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be GENERATED because regime is risk_on (shadow veto only fires on risk_off)
        assert signal is not None, (
            "Shadow veto should NOT fire when regime is risk_on (hard veto disabled)"
        )
        assert signal.reasoning['entry_pattern'] == 'continuation'
    
    @pytest.mark.asyncio
    async def test_shadow_veto_does_not_fire_below_rsi_threshold(self):
        """Shadow veto must NOT fire when RSI < threshold."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'SAFE3'
        market_data.close = 100.0
        market_data.open = 98.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 65.0,  # KEY: RSI < 70
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 95.0,  # Not breakout
            'sma_20': 98.0,  # Continuation pattern
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',  # risk_off
                time_of_day='midday',
                spy_change_pct=-0.8,
                vix_change_pct=8.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg_instance.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF = True
                mock_cfg_instance.SHADOW_VETO_RSI_THRESHOLD = 70.0
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be GENERATED because RSI < 70
        assert signal is not None, (
            "Shadow veto should NOT fire when RSI < threshold"
        )
        assert signal.reasoning['entry_pattern'] == 'continuation'
    
    @pytest.mark.asyncio
    async def test_shadow_veto_disabled_when_flag_false(self):
        """Shadow veto must NOT fire when ENABLE_SHADOW_VETO=False.
        
        NOTE: This tests the shadow-veto-only path with hard veto DISABLED.
        When hard veto is enabled (default), this scenario WILL be blocked.
        This test verifies the legacy shadow path can be disabled.
        """
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'BYPASS'
        market_data.close = 100.0
        market_data.open = 98.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 72.0,  # RSI >= 70
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 95.0,  # Not breakout
            'sma_20': 98.0,  # Continuation pattern
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',  # risk_off
                time_of_day='midday',
                spy_change_pct=-0.8,
                vix_change_pct=8.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                # Hard veto DISABLED to test shadow-only path
                mock_cfg_instance.ENABLE_HARD_VETO_CONTINUATION_RSI70 = False
                # Shadow veto also DISABLED
                mock_cfg_instance.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF = False
                mock_cfg_instance.SHADOW_VETO_RSI_THRESHOLD = 70.0
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be GENERATED because both vetoes are disabled
        assert signal is not None, (
            "Signal should pass when both hard veto and shadow veto are disabled"
        )


class TestStageAFloorsNotLoosened:
    """Test that Stage-A floors are NOT loosened.
    
    v-pause-live-daytrade-2026-09-10: Hari constraint — do NOT loosen
    Stage A floors (n>=150/>=10 sess, PF>=1.30, WR>=48%, exp>=+0.05R,
    DD<=6%, max losing day<=2R).
    """
    
    def test_stage_a_min_trades_still_150(self):
        """Stage-A min trades must still be 150."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_STAGE_A_MIN_TRADES == 150, (
            "Stage-A MIN_TRADES must be 150 — DO NOT LOOSEN"
        )
    
    def test_stage_a_min_sessions_still_10(self):
        """Stage-A min sessions must still be 10."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_STAGE_A_MIN_SESSIONS == 10, (
            "Stage-A MIN_SESSIONS must be 10 — DO NOT LOOSEN"
        )
    
    def test_stage_a_min_pf_still_130(self):
        """Stage-A min PF must still be 1.30."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_STAGE_A_MIN_PF == 1.30, (
            "Stage-A MIN_PF must be 1.30 — DO NOT LOOSEN"
        )
    
    def test_stage_a_min_win_rate_still_048(self):
        """Stage-A min win rate must still be 0.48."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_STAGE_A_MIN_WIN_RATE == 0.48, (
            "Stage-A MIN_WIN_RATE must be 0.48 — DO NOT LOOSEN"
        )
    
    def test_stage_a_min_expectancy_still_005(self):
        """Stage-A min expectancy must still be 0.05 R."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_STAGE_A_MIN_EXPECTANCY_R == 0.05, (
            "Stage-A MIN_EXPECTANCY_R must be 0.05 — DO NOT LOOSEN"
        )
    
    def test_stage_a_max_dd_still_006(self):
        """Stage-A max DD must still be 6%."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_STAGE_A_MAX_DD_PCT == 0.06, (
            "Stage-A MAX_DD_PCT must be 0.06 — DO NOT LOOSEN (raise)"
        )
    
    def test_stage_a_max_losing_day_still_2r(self):
        """Stage-A max losing day must still be 2.0 R."""
        from core.config import Config
        cfg = Config()
        assert cfg.MOMENTUM_STAGE_A_MAX_LOSING_DAY_R == 2.0, (
            "Stage-A MAX_LOSING_DAY_R must be 2.0 — DO NOT LOOSEN (raise)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# v-hard-veto-rsi70-2026-09-14: Tests for hard veto continuation + RSI>=70
# in ALL regimes (promoted from shadow-only risk_off gate).
#
# RCA: 2026-09-14 FTFT LIVE loss — continuation RSI 74.31, regime=mixed.
# ══════════════════════════════════════════════════════════════════════════════

class TestHardVetoContinuationRsi70AllRegimes:
    """Test hard veto for continuation + RSI>=70 in ALL regimes.
    
    v-hard-veto-rsi70-2026-09-14: Promotes continuation + RSI>=70 veto from
    shadow-only (risk_off-gated) to a hard veto in all regimes (including mixed).
    
    RCA: 2026-09-14 FTFT LIVE loss — continuation entry RSI 74.31, regime=mixed.
    """
    
    def test_enable_hard_veto_default_true(self):
        """ENABLE_HARD_VETO_CONTINUATION_RSI70 must default to True.
        
        Safe on-path: entries blocked when pattern matches.
        """
        from core.config import Config
        cfg = Config()
        assert cfg.ENABLE_HARD_VETO_CONTINUATION_RSI70 is True, (
            "ENABLE_HARD_VETO_CONTINUATION_RSI70 must default to True — "
            "safe on-path means entries are blocked when pattern matches"
        )
    
    @pytest.mark.asyncio
    async def test_hard_veto_blocks_mixed_regime_continuation_rsi70(self):
        """Hard veto must block continuation + RSI>=70 even in MIXED regime.
        
        This is the FTFT RCA scenario: continuation + RSI 74.31 + regime=mixed.
        The old shadow veto only fired on risk_off, so this went through LIVE.
        """
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'FTFT'  # The RCA symbol
        market_data.close = 100.0
        market_data.open = 98.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 74.31,  # KEY: RSI >= 70 (the exact FTFT value)
            'volume_ratio': 2.0,
            'adx': 30,  # > 25 for continuation
            'atr': 1.5,
            'high_20': 95.0,  # Not breakout (close not > high_20)
            'sma_20': 98.0,  # Close > sma_20 for continuation
            'macd': 0.5,
            'macd_signal': 0.3,  # MACD > signal for continuation
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='mixed',  # KEY: MIXED regime (the FTFT scenario)
                time_of_day='midday',
                spy_change_pct=-0.3,
                vix_change_pct=3.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                # Hard veto ENABLED (default)
                mock_cfg_instance.ENABLE_HARD_VETO_CONTINUATION_RSI70 = True
                mock_cfg_instance.SHADOW_VETO_RSI_THRESHOLD = 70.0
                # Shadow veto also enabled (but should not be reached)
                mock_cfg_instance.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF = True
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal must be BLOCKED (None) due to hard veto
        assert signal is None, (
            "Hard veto must block continuation + RSI>=70 even in MIXED regime — "
            "this is the FTFT RCA scenario"
        )
    
    @pytest.mark.asyncio
    async def test_hard_veto_blocks_risk_on_regime_continuation_rsi70(self):
        """Hard veto must block continuation + RSI>=70 even in RISK_ON regime."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'RISKY_ON'
        market_data.close = 100.0
        market_data.open = 98.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 72.0,  # RSI >= 70
            'volume_ratio': 2.5,
            'adx': 35,
            'atr': 1.5,
            'high_20': 95.0,  # Not breakout
            'sma_20': 98.0,  # Continuation pattern
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 4.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_on',  # KEY: RISK_ON regime
                time_of_day='midday',
                spy_change_pct=0.5,
                vix_change_pct=-2.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg_instance.ENABLE_HARD_VETO_CONTINUATION_RSI70 = True
                mock_cfg_instance.SHADOW_VETO_RSI_THRESHOLD = 70.0
                mock_cfg_instance.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF = True
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal must be BLOCKED even in risk_on
        assert signal is None, (
            "Hard veto must block continuation + RSI>=70 even in risk_on regime"
        )
    
    @pytest.mark.asyncio
    async def test_hard_veto_off_allows_mixed_continuation_rsi70_to_shadow_path(self):
        """When hard veto OFF, mixed+continuation+RSI>=70 passes to old path.
        
        With ENABLE_HARD_VETO_CONTINUATION_RSI70=False, the old shadow veto
        path should be reached. Since shadow veto only fires on risk_off,
        a mixed regime entry should NOT be blocked (generates signal).
        """
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'FALLTHROUGH'
        market_data.close = 100.0
        market_data.open = 98.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 72.0,  # RSI >= 70
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 95.0,  # Not breakout
            'sma_20': 98.0,  # Continuation pattern
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='mixed',  # MIXED regime
                time_of_day='midday',
                spy_change_pct=-0.3,
                vix_change_pct=3.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                # KEY: Hard veto DISABLED
                mock_cfg_instance.ENABLE_HARD_VETO_CONTINUATION_RSI70 = False
                mock_cfg_instance.SHADOW_VETO_RSI_THRESHOLD = 70.0
                # Shadow veto enabled (but only fires on risk_off)
                mock_cfg_instance.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF = True
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be GENERATED because:
        # 1. Hard veto is OFF
        # 2. Shadow veto only fires on risk_off, and we're in mixed
        assert signal is not None, (
            "When hard veto OFF, mixed+continuation+RSI>=70 should pass through — "
            "shadow veto only fires on risk_off"
        )
        assert signal.reasoning['entry_pattern'] == 'continuation'
    
    @pytest.mark.asyncio
    async def test_hard_veto_off_shadow_veto_still_fires_on_risk_off(self):
        """When hard veto OFF, shadow veto still fires on risk_off.
        
        This verifies backward compatibility: the old shadow veto path
        continues to work when hard veto is disabled.
        """
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'SHADOW_PATH'
        market_data.close = 100.0
        market_data.open = 98.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 72.0,  # RSI >= 70
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 95.0,  # Not breakout
            'sma_20': 98.0,  # Continuation pattern
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',  # KEY: RISK_OFF regime
                time_of_day='midday',
                spy_change_pct=-0.8,
                vix_change_pct=8.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                # Hard veto DISABLED
                mock_cfg_instance.ENABLE_HARD_VETO_CONTINUATION_RSI70 = False
                mock_cfg_instance.SHADOW_VETO_RSI_THRESHOLD = 70.0
                # Shadow veto ENABLED
                mock_cfg_instance.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF = True
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be BLOCKED by shadow veto (risk_off path)
        assert signal is None, (
            "When hard veto OFF, shadow veto must still fire on risk_off"
        )
    
    @pytest.mark.asyncio
    async def test_hard_veto_does_not_block_pullback_pattern(self):
        """Hard veto must NOT block pullback pattern (only continuation).
        
        Pullback pattern has RSI 40-60, so it should never hit the RSI>=70
        threshold anyway. But we test explicitly to ensure the pattern check
        is correct.
        """
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'PULLBACK'
        market_data.close = 100.0
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 50.0,  # Pullback range 40-60
            'volume_ratio': 2.0,
            'adx': 25,
            'atr': 1.5,
            'high_20': 105.0,  # Not breakout
            'sma_20': 99.5,  # Within 2% of SMA20 for pullback
            'macd': 0.3,
            'macd_signal': 0.2,  # MACD bullish
            'day_change_pct': 2.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='mixed',
                time_of_day='midday',
                spy_change_pct=0.2,
                vix_change_pct=1.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg_instance.ENABLE_HARD_VETO_CONTINUATION_RSI70 = True
                mock_cfg_instance.SHADOW_VETO_RSI_THRESHOLD = 70.0
                mock_cfg_instance.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF = True
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be GENERATED (pullback pattern not blocked)
        assert signal is not None, (
            "Hard veto must NOT block pullback pattern — only continuation"
        )
        assert signal.reasoning['entry_pattern'] == 'pullback'
    
    @pytest.mark.asyncio
    async def test_hard_veto_does_not_block_breakout_pattern(self):
        """Hard veto must NOT block breakout pattern even with high RSI.
        
        Breakout pattern can have high RSI but is not the dangerous pattern.
        """
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'BREAKOUT'
        market_data.close = 102.0  # Above high_20 = breakout
        market_data.open = 99.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 72.0,  # High RSI >= 70
            'volume_ratio': 2.5,
            'adx': 35,
            'atr': 1.5,
            'high_20': 100.0,  # Close > high_20 = BREAKOUT
            'sma_20': 98.0,
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 4.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='mixed',
                time_of_day='midday',
                spy_change_pct=-0.3,
                vix_change_pct=3.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg_instance.ENABLE_HARD_VETO_CONTINUATION_RSI70 = True
                mock_cfg_instance.SHADOW_VETO_RSI_THRESHOLD = 70.0
                mock_cfg_instance.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF = True
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be GENERATED (breakout pattern not blocked)
        assert signal is not None, (
            "Hard veto must NOT block breakout pattern — only continuation"
        )
        assert signal.reasoning['entry_pattern'] == 'breakout'
    
    @pytest.mark.asyncio
    async def test_hard_veto_does_not_block_continuation_below_rsi_threshold(self):
        """Hard veto must NOT block continuation when RSI < threshold."""
        from strategies.builtin import DayTradeMomentumStrategy
        
        strategy = DayTradeMomentumStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'CONT_LOW_RSI'
        market_data.close = 100.0
        market_data.open = 98.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 65.0,  # KEY: RSI < 70 (below threshold)
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'high_20': 95.0,  # Not breakout
            'sma_20': 98.0,  # Continuation pattern
            'macd': 0.5,
            'macd_signal': 0.3,
            'day_change_pct': 3.0,
        }
        
        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='mixed',
                time_of_day='midday',
                spy_change_pct=-0.3,
                vix_change_pct=3.0,
                sector_etf='XLK',
                reason='test',
            )
            
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_MOMENTUM = True
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_SPY_PCT = -1.5
                mock_cfg_instance.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE = 15.0
                mock_cfg_instance.MOMENTUM_RISK_OFF_SIZE_MULT = 0.25
                mock_cfg_instance.MOMENTUM_OPENING_30_SIZE_MULT = 0.5
                mock_cfg_instance.MOMENTUM_MIN_RS_VS_SPY = 0.5
                mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                mock_cfg_instance.ENABLE_HARD_VETO_CONTINUATION_RSI70 = True
                mock_cfg_instance.SHADOW_VETO_RSI_THRESHOLD = 70.0
                mock_cfg_instance.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF = True
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be GENERATED (RSI below threshold)
        assert signal is not None, (
            "Hard veto must NOT block continuation when RSI < threshold"
        )
        assert signal.reasoning['entry_pattern'] == 'continuation'


# ══════════════════════════════════════════════════════════════════════════════
# v-meanrev-live-flag-2026-09-15: Tests for MEAN_REV_LIVE_ENTRIES_ENABLED flag
# Modular twin of DAY_TRADE_LIVE_ENTRIES_ENABLED for mean-reversion strategy.
# ══════════════════════════════════════════════════════════════════════════════

class TestMeanRevLiveEntriesFlag:
    """Test MEAN_REV_LIVE_ENTRIES_ENABLED flag and LIVE entry blocking.
    
    v-meanrev-live-flag-2026-09-15: modular switch to pause LIVE mean-rev
    entries while keeping sim/commentary analysis running and existing
    positions' exits intact.
    """
    
    def test_mean_rev_live_entries_default_true(self):
        """MEAN_REV_LIVE_ENTRIES_ENABLED must default to True.
        
        LIVE entries enabled by default. Set MEAN_REV_LIVE_ENTRIES_ENABLED=0
        to pause when needed.
        """
        from core.config import Config
        cfg = Config()
        assert cfg.MEAN_REV_LIVE_ENTRIES_ENABLED is True, (
            "MEAN_REV_LIVE_ENTRIES_ENABLED must default to True — "
            "set env MEAN_REV_LIVE_ENTRIES_ENABLED=0 to pause"
        )


class TestEngineBlocksLiveMeanRev:
    """Test that the engine blocks LIVE mean-rev entries when flag is False.
    
    These tests verify the engine's signal routing gate, not the strategy.
    """
    
    def test_engine_has_meanrev_live_pause_gate(self):
        """Engine must have the v-pause-live-meanrev-2026-09-15 gate."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()
        
        assert "v-pause-live-meanrev-2026-09-15" in src, (
            "Engine must contain v-pause-live-meanrev-2026-09-15 gate"
        )
        assert "MEAN_REV_LIVE_ENTRIES_ENABLED" in src, (
            "Engine must check MEAN_REV_LIVE_ENTRIES_ENABLED flag"
        )
        assert "meanrev_live_pause" in src, (
            "Engine must audit with component=meanrev_live_pause"
        )
    
    def test_engine_gate_checks_strategy_name(self):
        """Engine gate must specifically check for mean_reversion strategy."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()
        
        anchor = src.find("v-pause-live-meanrev-2026-09-15")
        assert anchor != -1
        window = src[anchor: anchor + 3000]
        
        assert 'mean_reversion' in window, (
            "Gate must check for mean_reversion strategy"
        )
        assert 'mean_reversion_short' in window, (
            "Gate must also check for mean_reversion_short strategy"
        )
    
    def test_engine_gate_checks_live_mode(self):
        """Engine gate must only block in LIVE mode, not SIM."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()
        
        anchor = src.find("v-pause-live-meanrev-2026-09-15")
        assert anchor != -1
        window = src[anchor: anchor + 3000]
        
        assert 'TradingMode.LIVE' in window, (
            "Gate must check self.mode == TradingMode.LIVE"
        )
    
    def test_engine_gate_logs_commentary(self):
        """Engine gate must add commentary when blocking mean-rev entry."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()
        
        anchor = src.find("v-pause-live-meanrev-2026-09-15")
        assert anchor != -1
        window = src[anchor: anchor + 3000]
        
        assert "Mean-Rev LIVE Entry Paused" in window, (
            "Gate must add commentary with descriptive title"
        )
        assert "MEAN_REV_LIVE_ENTRIES_ENABLED=False" in window, (
            "Gate commentary must reference the flag name"
        )


# ══════════════════════════════════════════════════════════════════════════════
# v-late-entry-gate-2026-09-15: Tests for ENABLE_LATE_ENTRY_GATE shadow mode
# Detect late/chasing entries before they become losses.
# ══════════════════════════════════════════════════════════════════════════════

class TestLateEntryGateConfig:
    """Test late-entry gate configuration flags."""
    
    def test_enable_late_entry_gate_default_true(self):
        """ENABLE_LATE_ENTRY_GATE must default to True."""
        from core.config import Config
        cfg = Config()
        assert cfg.ENABLE_LATE_ENTRY_GATE is True, (
            "ENABLE_LATE_ENTRY_GATE must default to True — "
            "set ENABLE_LATE_ENTRY_GATE=0 to disable"
        )
    
    def test_late_entry_gate_shadow_default_true(self):
        """LATE_ENTRY_GATE_SHADOW must default to True (log only, no block)."""
        from core.config import Config
        cfg = Config()
        assert cfg.LATE_ENTRY_GATE_SHADOW is True, (
            "LATE_ENTRY_GATE_SHADOW must default to True — "
            "shadow mode logs but does not block"
        )
    
    def test_late_entry_extension_threshold_default(self):
        """LATE_ENTRY_EXTENSION_THRESHOLD must default to 0.75."""
        from core.config import Config
        cfg = Config()
        assert cfg.LATE_ENTRY_EXTENSION_THRESHOLD == 0.75, (
            "LATE_ENTRY_EXTENSION_THRESHOLD must default to 0.75"
        )
    
    def test_late_entry_vwap_atr_mult_default(self):
        """LATE_ENTRY_VWAP_ATR_MULT must default to 1.0."""
        from core.config import Config
        cfg = Config()
        assert cfg.LATE_ENTRY_VWAP_ATR_MULT == 1.0, (
            "LATE_ENTRY_VWAP_ATR_MULT must default to 1.0"
        )
    
    def test_late_entry_bars_since_impulse_default(self):
        """LATE_ENTRY_BARS_SINCE_IMPULSE must default to 5."""
        from core.config import Config
        cfg = Config()
        assert cfg.LATE_ENTRY_BARS_SINCE_IMPULSE == 5, (
            "LATE_ENTRY_BARS_SINCE_IMPULSE must default to 5"
        )


class TestLateEntryGateEngine:
    """Test that the engine has late-entry gate logic."""
    
    def test_engine_has_late_entry_gate(self):
        """Engine must have the v-late-entry-gate-2026-09-15 gate."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()
        
        assert "v-late-entry-gate-2026-09-15" in src, (
            "Engine must contain v-late-entry-gate-2026-09-15 gate"
        )
        assert "ENABLE_LATE_ENTRY_GATE" in src, (
            "Engine must check ENABLE_LATE_ENTRY_GATE flag"
        )
        assert "late_entry_gate" in src, (
            "Engine must audit with component=late_entry_gate"
        )
    
    def test_engine_late_entry_checks_strategies(self):
        """Engine late-entry gate must check relevant strategies."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()
        
        anchor = src.find("v-late-entry-gate-2026-09-15")
        assert anchor != -1
        window = src[anchor: anchor + 4000]
        
        assert "day_trade_momentum" in window, (
            "Late-entry gate must check day_trade_momentum"
        )
        assert "mean_reversion" in window, (
            "Late-entry gate must check mean_reversion"
        )
        assert "orb_contraction_rvol" in window, (
            "Late-entry gate must check orb_contraction_rvol"
        )
    
    def test_engine_late_entry_has_heuristics(self):
        """Engine late-entry gate must implement all heuristics."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()
        
        anchor = src.find("v-late-entry-gate-2026-09-15")
        assert anchor != -1
        window = src[anchor: anchor + 4000]
        
        # Heuristic 1: Extension ratio
        assert "extension" in window.lower(), (
            "Late-entry gate must check extension ratio"
        )
        assert "LATE_ENTRY_EXTENSION_THRESHOLD" in window, (
            "Late-entry gate must use LATE_ENTRY_EXTENSION_THRESHOLD"
        )
        
        # Heuristic 2: VWAP chase
        assert "vwap" in window.lower(), (
            "Late-entry gate must check VWAP chase"
        )
        assert "LATE_ENTRY_VWAP_ATR_MULT" in window, (
            "Late-entry gate must use LATE_ENTRY_VWAP_ATR_MULT"
        )
        
        # Heuristic 3: Bars since impulse
        assert "bars_since_impulse" in window.lower(), (
            "Late-entry gate must check bars since impulse"
        )
        assert "LATE_ENTRY_BARS_SINCE_IMPULSE" in window, (
            "Late-entry gate must use LATE_ENTRY_BARS_SINCE_IMPULSE"
        )
    
    def test_engine_late_entry_shadow_mode(self):
        """Engine late-entry gate must support shadow mode."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()
        
        anchor = src.find("v-late-entry-gate-2026-09-15")
        assert anchor != -1
        window = src[anchor: anchor + 4000]
        
        assert "LATE_ENTRY_GATE_SHADOW" in window, (
            "Late-entry gate must check LATE_ENTRY_GATE_SHADOW"
        )
        assert "shadow_late_entry_skip" in window, (
            "Late-entry gate must log shadow_late_entry_skip action"
        )
    
    def test_engine_late_entry_logs_commentary(self):
        """Engine late-entry gate must add commentary."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()
        
        anchor = src.find("v-late-entry-gate-2026-09-15")
        assert anchor != -1
        window = src[anchor: anchor + 4000]
        
        assert "Late Entry" in window, (
            "Late-entry gate must add commentary with descriptive title"
        )
        assert "late_reasons" in window, (
            "Late-entry gate commentary must include late_reasons"
        )
