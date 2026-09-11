"""Tests for v-day-trade-momentum-desk-2026-09-10.

Coverage:
  1. Mover passes quality relax (bypasses SMA50/RS)
  2. risk_off reduces size, doesn't hard-block
  3. Momentum signal generates under synthetic bars
  4. Extreme risk (SPY <= -1.5% AND VIX spike) hard-blocks
  5. Day-trade size multiplier applied correctly
"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np


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
    """Test that risk_off reduces size instead of hard-blocking for momentum."""
    
    @pytest.mark.asyncio
    async def test_risk_off_reduces_size_not_hard_block(self):
        """risk_off regime reduces size to 0.25x, doesn't veto."""
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
        
        # Signal should be generated (not blocked)
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
    """Explicit tests for risk_off SIZE REDUCTION (not freeze) behavior.
    
    v-market-context-size-not-freeze-2026-09-10: The user requirement is
    that risk_off should REDUCE SIZE, NOT hard-block momentum entries.
    These tests verify that invariant explicitly.
    """
    
    @pytest.mark.asyncio
    async def test_risk_off_signal_generated_not_none(self):
        """risk_off must generate a signal (not None), proving no hard-block."""
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
        
        # CRITICAL: Signal must NOT be None under risk_off
        assert signal is not None, "risk_off must NOT hard-block momentum signals"
    
    @pytest.mark.asyncio
    async def test_risk_off_size_mult_is_025(self):
        """risk_off must apply 0.25x size multiplier."""
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
        """risk_off + opening_30 uses minimum of 0.25x and 0.5x = 0.25x."""
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
        """Shadow veto must NOT fire when regime is NOT risk_off."""
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
                mock_cfg_instance.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF = True
                mock_cfg_instance.SHADOW_VETO_RSI_THRESHOLD = 70.0
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be GENERATED because regime is risk_on
        assert signal is not None, (
            "Shadow veto should NOT fire when regime is risk_on"
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
        """Shadow veto must NOT fire when ENABLE_SHADOW_VETO=False."""
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
                # KEY: Shadow veto DISABLED
                mock_cfg_instance.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF = False
                mock_cfg_instance.SHADOW_VETO_RSI_THRESHOLD = 70.0
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        # Signal should be GENERATED because shadow veto is disabled
        assert signal is not None, (
            "Shadow veto should NOT fire when ENABLE_SHADOW_VETO=False"
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
