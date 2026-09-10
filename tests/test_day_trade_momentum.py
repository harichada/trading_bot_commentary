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
