"""Tests for v-orb-prototype-2026-09-10: ORB + contraction + RVOL strategy.

Coverage:
  1. CRITICAL: risk_off HARD SKIP (NOT size-down) - Research+CoS lock
  2. Config flag defaults
  3. Signal generation for bullish/bearish breakouts
  4. LIVE entry blocking
  5. Time-based gates (opening range window, entry cutoff)
  6. Volatility contraction check
  7. RVOL check
  8. Stage-A floors not loosened

Hari APPROVED order: #1 ORB+contraction+RVOL → then #2 RVOL continuation
→ #3 Gao late-day. This PR is ONLY #1.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, AsyncMock


class TestORBRiskOffHardSkip:
    """CRITICAL: Test that ORB HARD-SKIPS in risk_off regime (NOT size-down).
    
    v-orb-prototype-2026-09-10: Research+CoS lock: skip ORB entirely in
    risk_off (NOT size-down). If regime is risk_off, do not signal/enter
    ORB at all.
    
    This is different from DayTradeMomentumStrategy which REDUCES SIZE
    on risk_off. ORB is an opening-range directional bet that doesn't
    make sense when the market is in panic mode.
    """
    
    @pytest.mark.asyncio
    async def test_risk_off_returns_none_not_signal(self):
        """risk_off must return None (NO signal), not a size-reduced signal."""
        from strategies.builtin import ORBContractionRVOLStrategy
        
        strategy = ORBContractionRVOLStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'SPY'
        market_data.close = 450.0
        market_data.open = 448.0
        market_data.timestamp = datetime.now(timezone.utc)
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 1.5,  # Above min RVOL
            'adx': 30,
            'atr': 2.0,
            'high_20': 449.0,  # close > orb_high = bullish breakout
            'low_20': 447.0,
            'orb_high': 449.0,
            'orb_low': 447.0,
        }
        
        with patch('strategies.builtin._get_minutes_since_open', return_value=30):
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
                    mock_cfg_instance.ENABLE_ORB_STRATEGY = True
                    mock_cfg_instance.ORB_LIVE_ENTRIES_ENABLED = True
                    mock_cfg_instance.ORB_SIM_SHADOW_ENABLED = True
                    mock_cfg_instance.ORB_OPENING_RANGE_MINUTES = 15
                    mock_cfg_instance.ORB_MIN_CONTRACTION_PCT = 0.20
                    mock_cfg_instance.ORB_MIN_RVOL = 1.2
                    mock_cfg_instance.ORB_PRIMARY_SYMBOLS = ['SPY', 'QQQ']
                    mock_cfg_instance.ORB_ATR_STOP_MULTIPLIER = 1.0
                    mock_cfg_instance.ORB_REWARD_RISK_RATIO = 2.0
                    mock_cfg_instance.ORB_MAX_ENTRY_MINUTES_AFTER_OPEN = 60
                    mock_cfg_instance.ORB_FLATTEN_BY_HOUR = 15
                    mock_cfg_instance.ORB_SIZE_MULTIPLIER = 0.5
                    mock_cfg.return_value = mock_cfg_instance
                    
                    signal = await strategy.generate_signal_with_commentary(market_data)
        
        # CRITICAL: Signal MUST be None in risk_off (HARD SKIP)
        assert signal is None, (
            "ORB must return None (HARD SKIP) in risk_off regime, NOT a "
            "size-reduced signal. This is different from momentum which "
            "reduces size. ORB doesn't make sense in panic conditions."
        )
    
    @pytest.mark.asyncio
    async def test_risk_off_does_not_apply_size_multiplier(self):
        """Verify risk_off doesn't just reduce size — it completely skips.
        
        The code path should never reach the point where it would apply
        a size multiplier in risk_off — it should return None before that.
        """
        from strategies.builtin import ORBContractionRVOLStrategy
        
        strategy = ORBContractionRVOLStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'QQQ'
        market_data.close = 380.0
        market_data.open = 378.0
        market_data.timestamp = datetime.now(timezone.utc)
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 2.0,
            'adx': 35,
            'atr': 2.5,
            'high_20': 379.0,
            'low_20': 377.0,
            'orb_high': 379.0,
            'orb_low': 377.0,
        }
        
        with patch('strategies.builtin._get_minutes_since_open', return_value=30):
            with patch('core.market_context.read_market_context') as mock_mc:
                mock_mc.return_value = MagicMock(
                    regime='risk_off',
                    time_of_day='morning',
                    spy_change_pct=-1.0,
                    vix_change_pct=10.0,
                    sector_etf='XLK',
                    reason='strong risk_off',
                )
                
                with patch('core.config.Config') as mock_cfg:
                    mock_cfg_instance = MagicMock()
                    mock_cfg_instance.ENABLE_ORB_STRATEGY = True
                    mock_cfg_instance.ORB_LIVE_ENTRIES_ENABLED = True
                    mock_cfg_instance.ORB_SIM_SHADOW_ENABLED = True
                    mock_cfg_instance.ORB_OPENING_RANGE_MINUTES = 15
                    mock_cfg_instance.ORB_MIN_CONTRACTION_PCT = 0.20
                    mock_cfg_instance.ORB_MIN_RVOL = 1.2
                    mock_cfg_instance.ORB_PRIMARY_SYMBOLS = ['SPY', 'QQQ']
                    mock_cfg_instance.ORB_ATR_STOP_MULTIPLIER = 1.0
                    mock_cfg_instance.ORB_REWARD_RISK_RATIO = 2.0
                    mock_cfg_instance.ORB_MAX_ENTRY_MINUTES_AFTER_OPEN = 60
                    mock_cfg_instance.ORB_FLATTEN_BY_HOUR = 15
                    mock_cfg_instance.ORB_SIZE_MULTIPLIER = 0.5
                    mock_cfg.return_value = mock_cfg_instance
                    
                    signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is None
    
    @pytest.mark.asyncio
    async def test_risk_on_generates_signal(self):
        """Verify signal IS generated when regime is NOT risk_off."""
        from strategies.builtin import ORBContractionRVOLStrategy
        
        strategy = ORBContractionRVOLStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'SPY'
        market_data.close = 450.5
        market_data.open = 448.0
        market_data.timestamp = datetime.now(timezone.utc)
        market_data.indicators = {
            'rsi': 55,
            'volume_ratio': 1.5,
            'adx': 30,
            'atr': 2.0,
            'high_20': 449.0,
            'low_20': 447.0,
            'orb_high': 449.0,
            'orb_low': 447.0,
        }
        
        with patch('strategies.builtin._get_minutes_since_open', return_value=30):
            with patch('core.market_context.read_market_context') as mock_mc:
                mock_mc.return_value = MagicMock(
                    regime='risk_on',  # NOT risk_off
                    time_of_day='morning',
                    spy_change_pct=0.5,
                    vix_change_pct=-2.0,
                    sector_etf='XLK',
                    reason='risk_on test',
                )
                
                with patch('core.config.Config') as mock_cfg:
                    mock_cfg_instance = MagicMock()
                    mock_cfg_instance.ENABLE_ORB_STRATEGY = True
                    mock_cfg_instance.ORB_LIVE_ENTRIES_ENABLED = True
                    mock_cfg_instance.ORB_SIM_SHADOW_ENABLED = True
                    mock_cfg_instance.ORB_OPENING_RANGE_MINUTES = 15
                    mock_cfg_instance.ORB_MIN_CONTRACTION_PCT = 0.10
                    mock_cfg_instance.ORB_MIN_RVOL = 1.2
                    mock_cfg_instance.ORB_PRIMARY_SYMBOLS = ['SPY', 'QQQ']
                    mock_cfg_instance.ORB_ATR_STOP_MULTIPLIER = 1.0
                    mock_cfg_instance.ORB_REWARD_RISK_RATIO = 2.0
                    mock_cfg_instance.ORB_MAX_ENTRY_MINUTES_AFTER_OPEN = 60
                    mock_cfg_instance.ORB_FLATTEN_BY_HOUR = 15
                    mock_cfg_instance.ORB_SIZE_MULTIPLIER = 0.5
                    mock_cfg.return_value = mock_cfg_instance
                    
                    signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is not None, "Signal should be generated when regime is risk_on"
        assert signal.reasoning['market_context_regime'] == 'risk_on'


class TestORBConfigDefaults:
    """Test ORB config flag defaults."""
    
    def test_enable_orb_strategy_default_false(self):
        """ENABLE_ORB_STRATEGY must default to False (feature flag off)."""
        from core.config import Config
        cfg = Config()
        assert cfg.ENABLE_ORB_STRATEGY is False, (
            "ENABLE_ORB_STRATEGY must default to False — "
            "prototype strategy should be opt-in"
        )
    
    def test_orb_live_entries_default_false(self):
        """ORB_LIVE_ENTRIES_ENABLED must default to False."""
        from core.config import Config
        cfg = Config()
        assert cfg.ORB_LIVE_ENTRIES_ENABLED is False, (
            "ORB_LIVE_ENTRIES_ENABLED must default to False — "
            "Stage-A validation required before LIVE entries"
        )
    
    def test_orb_sim_shadow_enabled_default_true(self):
        """ORB_SIM_SHADOW_ENABLED should default to True for testing."""
        from core.config import Config
        cfg = Config()
        assert cfg.ORB_SIM_SHADOW_ENABLED is True, (
            "ORB_SIM_SHADOW_ENABLED should default to True — "
            "allow paper testing / data collection"
        )
    
    def test_orb_primary_symbols_includes_spy_qqq(self):
        """ORB_PRIMARY_SYMBOLS must include SPY and QQQ (Hari instruction)."""
        from core.config import Config
        cfg = Config()
        primary = cfg.ORB_PRIMARY_SYMBOLS
        assert 'SPY' in primary, "SPY must be in ORB_PRIMARY_SYMBOLS"
        assert 'QQQ' in primary, "QQQ must be in ORB_PRIMARY_SYMBOLS"
    
    def test_orb_opening_range_minutes_default(self):
        """ORB_OPENING_RANGE_MINUTES defaults to 15."""
        from core.config import Config
        cfg = Config()
        assert cfg.ORB_OPENING_RANGE_MINUTES == 15
    
    def test_orb_min_contraction_pct_default(self):
        """ORB_MIN_CONTRACTION_PCT defaults to 0.20."""
        from core.config import Config
        cfg = Config()
        assert cfg.ORB_MIN_CONTRACTION_PCT == 0.20
    
    def test_orb_min_rvol_default(self):
        """ORB_MIN_RVOL defaults to 1.2."""
        from core.config import Config
        cfg = Config()
        assert cfg.ORB_MIN_RVOL == 1.2
    
    def test_orb_size_multiplier_default(self):
        """ORB_SIZE_MULTIPLIER defaults to 0.5 (half-size for prototype)."""
        from core.config import Config
        cfg = Config()
        assert cfg.ORB_SIZE_MULTIPLIER == 0.5


class TestORBSignalGeneration:
    """Test ORB signal generation under various conditions."""
    
    @pytest.mark.asyncio
    async def test_bullish_breakout_generates_buy_signal(self):
        """Bullish breakout (close > ORB high) generates BUY signal."""
        from strategies.builtin import ORBContractionRVOLStrategy
        from core.models import SignalType
        
        strategy = ORBContractionRVOLStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'SPY'
        market_data.close = 450.5  # Above orb_high (449.0)
        market_data.open = 448.0
        market_data.timestamp = datetime.now(timezone.utc)
        market_data.indicators = {
            'volume_ratio': 1.5,
            'atr': 2.0,
            'high_20': 449.0,
            'low_20': 447.0,
            'orb_high': 449.0,
            'orb_low': 447.0,
        }
        
        with patch('strategies.builtin._get_minutes_since_open', return_value=30):
            with patch('core.market_context.read_market_context') as mock_mc:
                mock_mc.return_value = MagicMock(
                    regime='risk_on',
                    time_of_day='morning',
                    spy_change_pct=0.5,
                    vix_change_pct=-1.0,
                    sector_etf='XLK',
                    reason='test',
                )
                
                with patch('core.config.Config') as mock_cfg:
                    mock_cfg_instance = MagicMock()
                    mock_cfg_instance.ENABLE_ORB_STRATEGY = True
                    mock_cfg_instance.ORB_LIVE_ENTRIES_ENABLED = True
                    mock_cfg_instance.ORB_SIM_SHADOW_ENABLED = True
                    mock_cfg_instance.ORB_OPENING_RANGE_MINUTES = 15
                    mock_cfg_instance.ORB_MIN_CONTRACTION_PCT = 0.10
                    mock_cfg_instance.ORB_MIN_RVOL = 1.2
                    mock_cfg_instance.ORB_PRIMARY_SYMBOLS = ['SPY', 'QQQ']
                    mock_cfg_instance.ORB_ATR_STOP_MULTIPLIER = 1.0
                    mock_cfg_instance.ORB_REWARD_RISK_RATIO = 2.0
                    mock_cfg_instance.ORB_MAX_ENTRY_MINUTES_AFTER_OPEN = 60
                    mock_cfg_instance.ORB_FLATTEN_BY_HOUR = 15
                    mock_cfg_instance.ORB_SIZE_MULTIPLIER = 0.5
                    mock_cfg.return_value = mock_cfg_instance
                    
                    signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is not None
        assert signal.signal_type == SignalType.BUY
        assert signal.reasoning['breakout_type'] == 'bullish'
        assert signal.reasoning['is_orb'] is True
    
    @pytest.mark.asyncio
    async def test_bearish_breakout_generates_sell_signal(self):
        """Bearish breakout (close < ORB low) generates SELL signal."""
        from strategies.builtin import ORBContractionRVOLStrategy
        from core.models import SignalType
        
        strategy = ORBContractionRVOLStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'SPY'
        market_data.close = 446.5  # Below orb_low (447.0)
        market_data.open = 448.0
        market_data.timestamp = datetime.now(timezone.utc)
        market_data.indicators = {
            'volume_ratio': 1.5,
            'atr': 2.0,
            'high_20': 449.0,
            'low_20': 447.0,
            'orb_high': 449.0,
            'orb_low': 447.0,
        }
        
        with patch('strategies.builtin._get_minutes_since_open', return_value=30):
            with patch('core.market_context.read_market_context') as mock_mc:
                mock_mc.return_value = MagicMock(
                    regime='mixed',  # Not risk_off
                    time_of_day='morning',
                    spy_change_pct=-0.3,
                    vix_change_pct=2.0,
                    sector_etf='XLK',
                    reason='test',
                )
                
                with patch('core.config.Config') as mock_cfg:
                    mock_cfg_instance = MagicMock()
                    mock_cfg_instance.ENABLE_ORB_STRATEGY = True
                    mock_cfg_instance.ORB_LIVE_ENTRIES_ENABLED = True
                    mock_cfg_instance.ORB_SIM_SHADOW_ENABLED = True
                    mock_cfg_instance.ORB_OPENING_RANGE_MINUTES = 15
                    mock_cfg_instance.ORB_MIN_CONTRACTION_PCT = 0.10
                    mock_cfg_instance.ORB_MIN_RVOL = 1.2
                    mock_cfg_instance.ORB_PRIMARY_SYMBOLS = ['SPY', 'QQQ']
                    mock_cfg_instance.ORB_ATR_STOP_MULTIPLIER = 1.0
                    mock_cfg_instance.ORB_REWARD_RISK_RATIO = 2.0
                    mock_cfg_instance.ORB_MAX_ENTRY_MINUTES_AFTER_OPEN = 60
                    mock_cfg_instance.ORB_FLATTEN_BY_HOUR = 15
                    mock_cfg_instance.ORB_SIZE_MULTIPLIER = 0.5
                    mock_cfg.return_value = mock_cfg_instance
                    
                    signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is not None
        assert signal.signal_type == SignalType.SELL
        assert signal.reasoning['breakout_type'] == 'bearish'
    
    @pytest.mark.asyncio
    async def test_no_breakout_returns_none(self):
        """Price within ORB range returns None (no signal)."""
        from strategies.builtin import ORBContractionRVOLStrategy
        
        strategy = ORBContractionRVOLStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'SPY'
        market_data.close = 448.0  # Within range [447, 449]
        market_data.open = 448.0
        market_data.timestamp = datetime.now(timezone.utc)
        market_data.indicators = {
            'volume_ratio': 1.5,
            'atr': 2.0,
            'high_20': 449.0,
            'low_20': 447.0,
            'orb_high': 449.0,
            'orb_low': 447.0,
        }
        
        with patch('strategies.builtin._get_minutes_since_open', return_value=30):
            with patch('core.market_context.read_market_context') as mock_mc:
                mock_mc.return_value = MagicMock(
                    regime='risk_on',
                    time_of_day='morning',
                    spy_change_pct=0.3,
                    vix_change_pct=-1.0,
                    sector_etf='XLK',
                    reason='test',
                )
                
                with patch('core.config.Config') as mock_cfg:
                    mock_cfg_instance = MagicMock()
                    mock_cfg_instance.ENABLE_ORB_STRATEGY = True
                    mock_cfg_instance.ORB_OPENING_RANGE_MINUTES = 15
                    mock_cfg_instance.ORB_MIN_CONTRACTION_PCT = 0.10
                    mock_cfg_instance.ORB_MIN_RVOL = 1.2
                    mock_cfg_instance.ORB_PRIMARY_SYMBOLS = ['SPY', 'QQQ']
                    mock_cfg_instance.ORB_MAX_ENTRY_MINUTES_AFTER_OPEN = 60
                    mock_cfg.return_value = mock_cfg_instance
                    
                    signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is None


class TestORBTimeGates:
    """Test time-based gates for ORB strategy."""
    
    @pytest.mark.asyncio
    async def test_before_opening_range_established_returns_none(self):
        """Before opening range is established, return None."""
        from strategies.builtin import ORBContractionRVOLStrategy
        
        strategy = ORBContractionRVOLStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'SPY'
        market_data.close = 450.0
        market_data.indicators = {
            'volume_ratio': 1.5,
            'atr': 2.0,
        }
        
        # 10 minutes after open, but ORB requires 15 minutes
        with patch('strategies.builtin._get_minutes_since_open', return_value=10):
            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_ORB_STRATEGY = True
                mock_cfg_instance.ORB_OPENING_RANGE_MINUTES = 15
                mock_cfg.return_value = mock_cfg_instance
                
                signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is None
    
    @pytest.mark.asyncio
    async def test_after_entry_window_returns_none(self):
        """After entry window closes, return None."""
        from strategies.builtin import ORBContractionRVOLStrategy
        
        strategy = ORBContractionRVOLStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'SPY'
        market_data.close = 450.0
        market_data.indicators = {
            'volume_ratio': 1.5,
            'atr': 2.0,
            'high_20': 449.0,
            'low_20': 447.0,
        }
        
        # 90 minutes after open, but max entry is 60 minutes
        with patch('strategies.builtin._get_minutes_since_open', return_value=90):
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
                    mock_cfg_instance.ENABLE_ORB_STRATEGY = True
                    mock_cfg_instance.ORB_OPENING_RANGE_MINUTES = 15
                    mock_cfg_instance.ORB_MAX_ENTRY_MINUTES_AFTER_OPEN = 60
                    mock_cfg.return_value = mock_cfg_instance
                    
                    signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is None


class TestORBRVOLCheck:
    """Test RVOL (Relative Volume) check."""
    
    @pytest.mark.asyncio
    async def test_insufficient_rvol_returns_none(self):
        """Volume below ORB_MIN_RVOL threshold returns None."""
        from strategies.builtin import ORBContractionRVOLStrategy
        
        strategy = ORBContractionRVOLStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'SPY'
        market_data.close = 450.5
        market_data.open = 448.0
        market_data.indicators = {
            'volume_ratio': 1.0,  # Below min (1.2)
            'atr': 2.0,
            'high_20': 449.0,
            'low_20': 447.0,
            'orb_high': 449.0,
            'orb_low': 447.0,
        }
        
        with patch('strategies.builtin._get_minutes_since_open', return_value=30):
            with patch('core.market_context.read_market_context') as mock_mc:
                mock_mc.return_value = MagicMock(
                    regime='risk_on',
                    time_of_day='morning',
                    spy_change_pct=0.5,
                    vix_change_pct=-1.0,
                    sector_etf='XLK',
                    reason='test',
                )
                
                with patch('core.config.Config') as mock_cfg:
                    mock_cfg_instance = MagicMock()
                    mock_cfg_instance.ENABLE_ORB_STRATEGY = True
                    mock_cfg_instance.ORB_OPENING_RANGE_MINUTES = 15
                    mock_cfg_instance.ORB_MIN_CONTRACTION_PCT = 0.10
                    mock_cfg_instance.ORB_MIN_RVOL = 1.2  # Requires 1.2
                    mock_cfg_instance.ORB_PRIMARY_SYMBOLS = ['SPY', 'QQQ']
                    mock_cfg_instance.ORB_MAX_ENTRY_MINUTES_AFTER_OPEN = 60
                    mock_cfg.return_value = mock_cfg_instance
                    
                    signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is None


class TestORBLiveEntriesBlocked:
    """Test that LIVE ORB entries are blocked when flag is False."""
    
    def test_engine_has_orb_live_pause_gate(self):
        """Engine must have the v-orb-prototype-2026-09-10 LIVE gate."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()
        
        assert "v-orb-prototype-2026-09-10" in src, (
            "Engine must contain v-orb-prototype-2026-09-10 gate"
        )
        assert "ORB_LIVE_ENTRIES_ENABLED" in src, (
            "Engine must check ORB_LIVE_ENTRIES_ENABLED flag"
        )
        assert "orb_live_pause" in src, (
            "Engine must audit with component=orb_live_pause"
        )
    
    def test_engine_gate_checks_strategy_name(self):
        """Engine gate must specifically check for orb_contraction_rvol strategy."""
        from pathlib import Path
        src = Path("core/engine.py").read_text()
        
        anchor = src.find("orb_live_pause")
        assert anchor != -1
        window = src[max(0, anchor - 1000): anchor + 2000]
        
        assert "orb_contraction_rvol" in window, (
            "Engine gate must check for strategy == 'orb_contraction_rvol'"
        )
        assert "TradingMode.LIVE" in window, (
            "Engine gate must check for LIVE mode"
        )


class TestORBSignalContents:
    """Test that ORB signals contain required fields."""
    
    @pytest.mark.asyncio
    async def test_signal_includes_orb_fields(self):
        """Signal must include ORB-specific fields in reasoning."""
        from strategies.builtin import ORBContractionRVOLStrategy
        
        strategy = ORBContractionRVOLStrategy(MagicMock())
        
        market_data = MagicMock()
        market_data.symbol = 'SPY'
        market_data.close = 450.5
        market_data.open = 448.0
        market_data.timestamp = datetime.now(timezone.utc)
        market_data.indicators = {
            'volume_ratio': 1.5,
            'atr': 2.0,
            'high_20': 449.0,
            'low_20': 447.0,
            'orb_high': 449.0,
            'orb_low': 447.0,
        }
        
        with patch('strategies.builtin._get_minutes_since_open', return_value=30):
            with patch('core.market_context.read_market_context') as mock_mc:
                mock_mc.return_value = MagicMock(
                    regime='risk_on',
                    time_of_day='morning',
                    spy_change_pct=0.5,
                    vix_change_pct=-1.0,
                    sector_etf='XLK',
                    reason='test',
                )
                
                with patch('core.config.Config') as mock_cfg:
                    mock_cfg_instance = MagicMock()
                    mock_cfg_instance.ENABLE_ORB_STRATEGY = True
                    mock_cfg_instance.ORB_LIVE_ENTRIES_ENABLED = True
                    mock_cfg_instance.ORB_SIM_SHADOW_ENABLED = True
                    mock_cfg_instance.ORB_OPENING_RANGE_MINUTES = 15
                    mock_cfg_instance.ORB_MIN_CONTRACTION_PCT = 0.10
                    mock_cfg_instance.ORB_MIN_RVOL = 1.2
                    mock_cfg_instance.ORB_PRIMARY_SYMBOLS = ['SPY', 'QQQ']
                    mock_cfg_instance.ORB_ATR_STOP_MULTIPLIER = 1.0
                    mock_cfg_instance.ORB_REWARD_RISK_RATIO = 2.0
                    mock_cfg_instance.ORB_MAX_ENTRY_MINUTES_AFTER_OPEN = 60
                    mock_cfg_instance.ORB_FLATTEN_BY_HOUR = 15
                    mock_cfg_instance.ORB_SIZE_MULTIPLIER = 0.5
                    mock_cfg.return_value = mock_cfg_instance
                    
                    signal = await strategy.generate_signal_with_commentary(market_data)
        
        assert signal is not None
        reasoning = signal.reasoning
        
        assert reasoning['strategy'] == 'orb_contraction_rvol'
        assert 'breakout_type' in reasoning
        assert 'orb_high' in reasoning
        assert 'orb_low' in reasoning
        assert 'orb_range' in reasoning
        assert 'contraction_pct' in reasoning
        assert 'volume_ratio' in reasoning
        assert reasoning['is_orb'] is True
        assert 'is_primary_symbol' in reasoning
        assert 'orb_size_multiplier' in reasoning
        assert 'flatten_by_hour' in reasoning
        assert 'session_id' in reasoning
        assert 'setup_type' in reasoning


class TestStageAFloorsNotLoosened:
    """Test that Stage-A floors are preserved.
    
    v-orb-prototype-2026-09-10: Stage-A floors (LOCKED — do NOT loosen):
      n>=150 trades, >=10 sessions, PF>=1.30, WR>=48%, exp>=+0.05R,
      DD<=6%, max losing day<=2R.
    """
    
    def test_momentum_stage_a_floors_still_locked(self):
        """Verify momentum Stage-A floors still exist (not accidentally removed)."""
        from core.config import Config
        cfg = Config()
        
        # Momentum Stage-A floors (from existing code)
        assert cfg.MOMENTUM_STAGE_A_MIN_TRADES == 150
        assert cfg.MOMENTUM_STAGE_A_MIN_SESSIONS == 10
        assert cfg.MOMENTUM_STAGE_A_MIN_PF == 1.30
        assert cfg.MOMENTUM_STAGE_A_MIN_WIN_RATE == 0.48
        assert cfg.MOMENTUM_STAGE_A_MIN_EXPECTANCY_R == 0.05
        assert cfg.MOMENTUM_STAGE_A_MAX_DD_PCT == 0.06
        assert cfg.MOMENTUM_STAGE_A_MAX_LOSING_DAY_R == 2.0


class TestLTHandsOffSymbols:
    """Test that LT hands-off forever symbols are not affected.
    
    MU, SNAP, HQGE, SPCX are marked is_long_term=True and should never
    be touched by new strategies.
    """
    
    def test_orb_does_not_affect_lt_symbols_mechanism(self):
        """Verify the ORB strategy doesn't interfere with LT hands-off.
        
        The engine's is_long_term check happens in the position management
        layer, not in the strategy layer, so ORB cannot accidentally
        override it.
        """
        from pathlib import Path
        
        engine_src = Path("core/engine.py").read_text()
        config_src = Path("core/config.py").read_text()
        
        assert "is_long_term" in engine_src, "Engine must respect is_long_term flag"
        assert "managed_by_bot" in engine_src, "Engine must respect managed_by_bot flag"
