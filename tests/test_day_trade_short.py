"""Tests for v-day-trade-short-2026-09-17: Day-trade momentum SHORT strategy.

Coverage:
  1. Config flags default correctly (LIVE=False, SHADOW=True)
  2. HANDS_OFF_DENYLIST symbols are never shorted
  3. risk_on regime hard-blocks short signals
  4. Direction reader bearish gate
  5. Weak RS vs SPY filter (symbol must underperform)
  6. RSI floor (don't short oversold)
  7. Breakdown/continuation-down pattern detection
  8. Shadow logging when LIVE disabled
  9. LIVE signal generation when enabled
"""
import os
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np

from core.models import SignalType, TradingMode


class TestDayTradeShortConfigFlags:
    """Test config flag defaults for day-trade short strategy."""

    def test_enable_day_trade_short_default_true(self):
        """ENABLE_DAY_TRADE_SHORT must default to True for shadow soak."""
        from core.config import Config
        cfg = Config()
        assert cfg.ENABLE_DAY_TRADE_SHORT is True, (
            "ENABLE_DAY_TRADE_SHORT must default to True to allow shadow soak"
        )

    def test_day_trade_short_live_entries_enabled_default_false(self):
        """DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED must default to False (SHADOW MODE).

        This is the critical SHADOW-FIRST deployment mode: strategy generates
        signals for shadow analysis but blocks LIVE short orders by default.
        """
        from core.config import Config
        cfg = Config()
        assert cfg.DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED is False, (
            "DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED must default to False — "
            "SHADOW-FIRST deployment: no LIVE shorts until Stage-A passes"
        )

    def test_enable_day_trade_short_shadow_default_true(self):
        """ENABLE_DAY_TRADE_SHORT_SHADOW must default to True for shadow logging."""
        from core.config import Config
        cfg = Config()
        assert cfg.ENABLE_DAY_TRADE_SHORT_SHADOW is True, (
            "ENABLE_DAY_TRADE_SHORT_SHADOW must default to True to collect shadow data"
        )

    def test_day_trade_short_rsi_floor_default_30(self):
        """DAY_TRADE_SHORT_RSI_FLOOR must default to 30 (avoid shorting oversold)."""
        from core.config import Config
        cfg = Config()
        assert cfg.DAY_TRADE_SHORT_RSI_FLOOR == 30.0, (
            "DAY_TRADE_SHORT_RSI_FLOOR must default to 30 — don't short into oversold"
        )

    def test_day_trade_short_min_weak_rs_default(self):
        """DAY_TRADE_SHORT_MIN_WEAK_RS_VS_SPY must default to 0.5."""
        from core.config import Config
        cfg = Config()
        assert cfg.DAY_TRADE_SHORT_MIN_WEAK_RS_VS_SPY == 0.5, (
            "DAY_TRADE_SHORT_MIN_WEAK_RS_VS_SPY must default to 0.5 — "
            "symbol must underperform SPY by at least 0.5%"
        )


class TestDayTradeShortHandsOffDenylist:
    """Test HANDS_OFF_DENYLIST protection for shorts."""

    @pytest.mark.asyncio
    async def test_mu_never_shorted(self):
        """MU must never be shorted — it's in HANDS_OFF_DENYLIST."""
        from strategies.builtin import DayTradeMomentumShortStrategy

        strategy = DayTradeMomentumShortStrategy(MagicMock())

        market_data = MagicMock()
        market_data.symbol = 'MU'  # HANDS_OFF symbol
        market_data.close = 100.0
        market_data.open = 101.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 45,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'low_20': 101.0,
            'sma_20': 102.0,
            'macd': -0.5,
            'macd_signal': -0.3,
            'day_change_pct': -3.0,
        }

        with patch('core.config.Config') as mock_cfg:
            mock_cfg_instance = MagicMock()
            mock_cfg_instance.ENABLE_DAY_TRADE_SHORT = True
            mock_cfg_instance.HANDS_OFF_DENYLIST = frozenset({'MU', 'HQGE', 'SPCX'})
            mock_cfg.return_value = mock_cfg_instance

            signal = await strategy.generate_signal_with_commentary(market_data)

        assert signal is None, (
            "MU must NEVER be shorted — it's in HANDS_OFF_DENYLIST. "
            "These are permanent hands-off positions."
        )

    @pytest.mark.asyncio
    async def test_hqge_never_shorted(self):
        """HQGE must never be shorted — it's in HANDS_OFF_DENYLIST."""
        from strategies.builtin import DayTradeMomentumShortStrategy

        strategy = DayTradeMomentumShortStrategy(MagicMock())

        market_data = MagicMock()
        market_data.symbol = 'HQGE'  # HANDS_OFF symbol
        market_data.close = 50.0
        market_data.open = 52.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 45,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 0.75,
            'low_20': 51.0,
            'sma_20': 52.0,
            'macd': -0.5,
            'macd_signal': -0.3,
            'day_change_pct': -3.0,
        }

        with patch('core.config.Config') as mock_cfg:
            mock_cfg_instance = MagicMock()
            mock_cfg_instance.ENABLE_DAY_TRADE_SHORT = True
            mock_cfg_instance.HANDS_OFF_DENYLIST = frozenset({'MU', 'HQGE', 'SPCX'})
            mock_cfg.return_value = mock_cfg_instance

            signal = await strategy.generate_signal_with_commentary(market_data)

        assert signal is None, (
            "HQGE must NEVER be shorted — it's in HANDS_OFF_DENYLIST"
        )

    @pytest.mark.asyncio
    async def test_spcx_never_shorted(self):
        """SPCX must never be shorted — it's in HANDS_OFF_DENYLIST."""
        from strategies.builtin import DayTradeMomentumShortStrategy

        strategy = DayTradeMomentumShortStrategy(MagicMock())

        market_data = MagicMock()
        market_data.symbol = 'SPCX'  # HANDS_OFF symbol
        market_data.close = 30.0
        market_data.open = 32.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 45,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 0.5,
            'low_20': 31.0,
            'sma_20': 32.0,
            'macd': -0.5,
            'macd_signal': -0.3,
            'day_change_pct': -3.0,
        }

        with patch('core.config.Config') as mock_cfg:
            mock_cfg_instance = MagicMock()
            mock_cfg_instance.ENABLE_DAY_TRADE_SHORT = True
            mock_cfg_instance.HANDS_OFF_DENYLIST = frozenset({'MU', 'HQGE', 'SPCX'})
            mock_cfg.return_value = mock_cfg_instance

            signal = await strategy.generate_signal_with_commentary(market_data)

        assert signal is None, (
            "SPCX must NEVER be shorted — it's in HANDS_OFF_DENYLIST"
        )


class TestDayTradeShortRiskOnBlock:
    """Test risk_on regime hard-block for shorts."""

    @pytest.mark.asyncio
    async def test_risk_on_hard_blocks_short(self):
        """risk_on regime must hard-block short signals — don't short bullish tape."""
        from strategies.builtin import DayTradeMomentumShortStrategy

        strategy = DayTradeMomentumShortStrategy(MagicMock())

        market_data = MagicMock()
        market_data.symbol = 'NVDA'
        market_data.close = 100.0
        market_data.open = 101.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 45,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 1.5,
            'low_20': 101.0,
            'sma_20': 102.0,
            'macd': -0.5,
            'macd_signal': -0.3,
            'day_change_pct': -2.0,
        }

        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_on',  # KEY: risk_on regime blocks shorts
                time_of_day='morning',
                spy_change_pct=1.5,  # SPY bullish
                vix_change_pct=-5.0,  # VIX falling
                sector_etf='XLK',
                reason='risk_on test',
            )

            with patch('core.config.Config') as mock_cfg:
                mock_cfg_instance = MagicMock()
                mock_cfg_instance.ENABLE_DAY_TRADE_SHORT = True
                mock_cfg_instance.HANDS_OFF_DENYLIST = frozenset({'MU', 'HQGE', 'SPCX'})
                mock_cfg.return_value = mock_cfg_instance

                signal = await strategy.generate_signal_with_commentary(market_data)

        assert signal is None, (
            "day_trade_momentum_short must return None in risk_on regime — "
            "shorting into a bullish tape is counter-trend and loses"
        )


class TestDayTradeShortRSIFloor:
    """Test RSI floor gate — don't short into oversold."""

    @pytest.mark.asyncio
    async def test_rsi_oversold_blocks_short(self):
        """RSI <= 30 must block short entry — oversold bounce risk."""
        from strategies.builtin import DayTradeMomentumShortStrategy

        strategy = DayTradeMomentumShortStrategy(MagicMock())

        market_data = MagicMock()
        market_data.symbol = 'TSLA'
        market_data.close = 200.0
        market_data.open = 210.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 25,  # KEY: RSI below floor (30)
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 3.0,
            'low_20': 205.0,
            'sma_20': 210.0,
            'sma_50': 220.0,
            'macd': -1.0,
            'macd_signal': -0.5,
            'day_change_pct': -5.0,
        }

        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',
                time_of_day='morning',
                spy_change_pct=-1.0,
                vix_change_pct=5.0,
                sector_etf='XLY',
                reason='risk_off test',
            )

            with patch('core.direction_reader.read_direction') as mock_dr:
                mock_dr.return_value = MagicMock(
                    direction=-5.0,
                    phase='middle',
                    ema_stack='aligned_down',
                    allows_short_entry=True,
                    reason='strong downtrend',
                )

                with patch('core.config.Config') as mock_cfg:
                    mock_cfg_instance = MagicMock()
                    mock_cfg_instance.ENABLE_DAY_TRADE_SHORT = True
                    mock_cfg_instance.HANDS_OFF_DENYLIST = frozenset({'MU', 'HQGE', 'SPCX'})
                    mock_cfg_instance.DAY_TRADE_SHORT_RSI_FLOOR = 30.0
                    mock_cfg_instance.DAY_TRADE_SHORT_RSI_CEILING = 85.0
                    mock_cfg_instance.DAY_TRADE_SHORT_MIN_WEAK_RS_VS_SPY = 0.5
                    mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                    mock_cfg.return_value = mock_cfg_instance

                    signal = await strategy.generate_signal_with_commentary(market_data)

        assert signal is None, (
            "day_trade_momentum_short must block when RSI <= 30 (oversold) — "
            "shorting into capitulation has high bounce risk"
        )


class TestDayTradeShortWeakRS:
    """Test weak RS vs SPY filter — symbol must underperform."""

    @pytest.mark.asyncio
    async def test_strong_rs_blocks_short(self):
        """Symbol outperforming SPY must NOT generate short signal."""
        from strategies.builtin import DayTradeMomentumShortStrategy

        strategy = DayTradeMomentumShortStrategy(MagicMock())

        market_data = MagicMock()
        market_data.symbol = 'META'
        market_data.close = 500.0
        market_data.open = 495.0  # Symbol is UP
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 45,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 7.5,
            'low_20': 495.0,
            'sma_20': 498.0,
            'sma_50': 490.0,
            'macd': -0.5,
            'macd_signal': -0.3,
            'day_change_pct': 1.0,  # Symbol UP 1%
        }

        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='mixed',
                time_of_day='morning',
                spy_change_pct=-0.5,  # SPY DOWN 0.5%
                vix_change_pct=2.0,
                sector_etf='XLC',
                reason='mixed test',
            )

            with patch('core.direction_reader.read_direction') as mock_dr:
                mock_dr.return_value = MagicMock(
                    direction=-3.0,
                    phase='middle',
                    ema_stack='mixed',
                    allows_short_entry=True,
                    reason='mild downtrend',
                )

                with patch('core.config.Config') as mock_cfg:
                    mock_cfg_instance = MagicMock()
                    mock_cfg_instance.ENABLE_DAY_TRADE_SHORT = True
                    mock_cfg_instance.HANDS_OFF_DENYLIST = frozenset({'MU', 'HQGE', 'SPCX'})
                    mock_cfg_instance.DAY_TRADE_SHORT_MIN_WEAK_RS_VS_SPY = 0.5
                    mock_cfg.return_value = mock_cfg_instance

                    signal = await strategy.generate_signal_with_commentary(market_data)

        # RS = 1.0 - (-0.5) = +1.5% (outperforming SPY)
        # Required: <= -0.5% (underperforming)
        assert signal is None, (
            "day_trade_momentum_short must block when symbol outperforms SPY — "
            "shorts require weak relative strength"
        )


class TestDayTradeShortShadowFallthrough:
    """Test shadow fallthrough bug fix (v-shadow-fallthrough-fix-2026-09-17).

    CRITICAL: When LIVE=False and SHADOW=False, the code must NOT fall
    through to emit a LIVE TradingSignal(SELL). It must return None.
    """

    @pytest.mark.asyncio
    async def test_live_false_shadow_false_returns_none(self):
        """LIVE=False + SHADOW=False must return None, NOT a TradingSignal.

        This test guards against the shadow fallthrough bug where the code
        would fall through to the LIVE path when both flags were False.
        """
        from strategies.builtin import DayTradeMomentumShortStrategy

        strategy = DayTradeMomentumShortStrategy(MagicMock())
        strategy._log_decision = MagicMock()

        market_data = MagicMock()
        market_data.symbol = 'NVDA'
        market_data.close = 98.0  # Below low_20 = breakdown pattern
        market_data.open = 105.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 40,
            'volume_ratio': 2.0,
            'adx': 25,
            'atr': 1.5,
            'low_20': 100.0,
            'high_20': 110.0,
            'sma_20': 105.0,
            'sma_50': 108.0,
            'macd': -0.5,
            'macd_signal': -0.3,
            'day_change_pct': -3.0,
        }

        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',
                time_of_day='morning',
                spy_change_pct=0.5,
                vix_change_pct=5.0,
                sector_etf='XLK',
                reason='risk_off test',
            )

            with patch('core.direction_reader.read_direction') as mock_dr:
                mock_dr.return_value = MagicMock(
                    direction=-4.0,
                    phase='middle',
                    ema_stack='aligned_down',
                    allows_short_entry=True,
                    reason='strong downtrend',
                )

                with patch('core.config.Config') as mock_cfg:
                    mock_cfg_instance = MagicMock()
                    mock_cfg_instance.ENABLE_DAY_TRADE_SHORT = True
                    mock_cfg_instance.HANDS_OFF_DENYLIST = frozenset({'MU', 'HQGE', 'SPCX'})
                    # KEY: Both LIVE and SHADOW disabled
                    mock_cfg_instance.DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED = False
                    mock_cfg_instance.ENABLE_DAY_TRADE_SHORT_SHADOW = False
                    mock_cfg_instance.DAY_TRADE_SHORT_MIN_WEAK_RS_VS_SPY = 0.5
                    mock_cfg_instance.DAY_TRADE_SHORT_RSI_FLOOR = 30.0
                    mock_cfg_instance.DAY_TRADE_SHORT_RSI_CEILING = 85.0
                    mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                    mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED = False
                    mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                    mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                    mock_cfg.return_value = mock_cfg_instance

                    signal = await strategy.generate_signal_with_commentary(market_data)

        # CRITICAL: Must return None when LIVE=False, even if SHADOW=False
        assert signal is None, (
            "CRITICAL BUG: LIVE=False + SHADOW=False must return None, "
            "but code fell through to LIVE path and returned a TradingSignal. "
            "This would place actual short orders when operator explicitly "
            "disabled LIVE shorts."
        )

        # Verify the skip was logged
        skip_calls = [c for c in strategy._log_decision.call_args_list
                      if c[0][1] == 'skip' and 'live_short_disabled' in str(c[0][2])]
        assert len(skip_calls) > 0, (
            "When LIVE=False and SHADOW=False, should log skip with reason "
            "indicating both flags are disabled"
        )


class TestDayTradeShortBreakdownPattern:
    """Test breakdown pattern detection for shorts."""

    @pytest.mark.asyncio
    async def test_breakdown_pattern_shadow_logged(self):
        """Breakdown pattern with all gates passing must shadow-log when LIVE disabled."""
        from strategies.builtin import DayTradeMomentumShortStrategy

        strategy = DayTradeMomentumShortStrategy(MagicMock())
        strategy._log_decision = MagicMock()

        market_data = MagicMock()
        market_data.symbol = 'AMD'
        market_data.close = 98.0  # Below low_20
        market_data.open = 105.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 40,  # Between floor (30) and ceiling (85)
            'volume_ratio': 2.0,  # Above 1.5x
            'adx': 25,  # Above 20
            'atr': 1.5,
            'low_20': 100.0,  # Close BELOW this = breakdown
            'high_20': 110.0,
            'sma_20': 105.0,
            'sma_50': 108.0,
            'macd': -0.5,
            'macd_signal': -0.3,
            'day_change_pct': -3.0,  # Symbol down 3%
        }

        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',  # Allows shorts
                time_of_day='morning',
                spy_change_pct=0.5,  # SPY up 0.5%
                vix_change_pct=5.0,
                sector_etf='XLK',
                reason='risk_off test',
            )

            with patch('core.direction_reader.read_direction') as mock_dr:
                mock_dr.return_value = MagicMock(
                    direction=-4.0,  # Bearish
                    phase='middle',  # Not exhausted
                    ema_stack='aligned_down',
                    allows_short_entry=True,
                    reason='strong downtrend',
                )

                with patch('core.config.Config') as mock_cfg:
                    mock_cfg_instance = MagicMock()
                    mock_cfg_instance.ENABLE_DAY_TRADE_SHORT = True
                    mock_cfg_instance.HANDS_OFF_DENYLIST = frozenset({'MU', 'HQGE', 'SPCX'})
                    mock_cfg_instance.DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED = False  # SHADOW MODE
                    mock_cfg_instance.ENABLE_DAY_TRADE_SHORT_SHADOW = True
                    mock_cfg_instance.DAY_TRADE_SHORT_MIN_WEAK_RS_VS_SPY = 0.5
                    mock_cfg_instance.DAY_TRADE_SHORT_RSI_FLOOR = 30.0
                    mock_cfg_instance.DAY_TRADE_SHORT_RSI_CEILING = 85.0
                    mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                    mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED = False
                    mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                    mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                    mock_cfg.return_value = mock_cfg_instance

                    # Patch _shadow_short_should_log to return True
                    with patch('strategies.builtin._shadow_short_should_log', return_value=True):
                        # Patch file operations
                        with patch('builtins.open', MagicMock()):
                            signal = await strategy.generate_signal_with_commentary(market_data)

        # RS = -3.0 - 0.5 = -3.5% (weak, passes filter)
        # Signal should be None (shadow mode) but shadow log should be written
        assert signal is None, (
            "In SHADOW mode (LIVE=False), signal should be None but shadow logged"
        )

        # Verify shadow logging was called
        shadow_calls = [c for c in strategy._log_decision.call_args_list
                        if c[0][1] == 'shadow']
        assert len(shadow_calls) > 0, "Shadow decision should be logged"


class TestDayTradeShortLiveSignal:
    """Test LIVE signal generation when enabled."""

    @pytest.mark.asyncio
    async def test_live_short_signal_generated(self):
        """When LIVE enabled, breakdown pattern must generate SELL signal."""
        from strategies.builtin import DayTradeMomentumShortStrategy

        strategy = DayTradeMomentumShortStrategy(MagicMock())

        market_data = MagicMock()
        market_data.symbol = 'AMD'
        market_data.close = 98.0  # Below low_20
        market_data.open = 105.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 40,
            'volume_ratio': 2.0,
            'adx': 25,
            'atr': 1.5,
            'low_20': 100.0,
            'high_20': 110.0,
            'sma_20': 105.0,
            'sma_50': 108.0,
            'macd': -0.5,
            'macd_signal': -0.3,
            'day_change_pct': -3.0,
        }

        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',
                time_of_day='morning',
                spy_change_pct=0.5,
                vix_change_pct=5.0,
                sector_etf='XLK',
                reason='risk_off test',
            )

            with patch('core.direction_reader.read_direction') as mock_dr:
                mock_dr.return_value = MagicMock(
                    direction=-4.0,
                    phase='middle',
                    ema_stack='aligned_down',
                    allows_short_entry=True,
                    reason='strong downtrend',
                )

                with patch('core.config.Config') as mock_cfg:
                    mock_cfg_instance = MagicMock()
                    mock_cfg_instance.ENABLE_DAY_TRADE_SHORT = True
                    mock_cfg_instance.HANDS_OFF_DENYLIST = frozenset({'MU', 'HQGE', 'SPCX'})
                    mock_cfg_instance.DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED = True  # LIVE MODE
                    mock_cfg_instance.ENABLE_DAY_TRADE_SHORT_SHADOW = True
                    mock_cfg_instance.DAY_TRADE_SHORT_MIN_WEAK_RS_VS_SPY = 0.5
                    mock_cfg_instance.DAY_TRADE_SHORT_RSI_FLOOR = 30.0
                    mock_cfg_instance.DAY_TRADE_SHORT_RSI_CEILING = 85.0
                    mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                    mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED = False
                    mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                    mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                    mock_cfg.return_value = mock_cfg_instance

                    signal = await strategy.generate_signal_with_commentary(market_data)

        assert signal is not None, (
            "When LIVE enabled, valid breakdown pattern should generate signal"
        )
        assert signal.signal_type == SignalType.SELL, (
            "Short signal must have signal_type=SELL"
        )
        assert signal.reasoning.get('is_short') is True, (
            "Short signal must have is_short=True in reasoning"
        )
        assert signal.reasoning.get('strategy') == 'day_trade_momentum_short', (
            "Short signal must identify strategy as day_trade_momentum_short"
        )
        assert signal.stop_loss > signal.entry_price, (
            "Short stop_loss must be ABOVE entry_price"
        )
        assert signal.take_profit < signal.entry_price, (
            "Short take_profit must be BELOW entry_price"
        )


class TestDayTradeShortDirectionGate:
    """Test direction reader gate for shorts."""

    @pytest.mark.asyncio
    async def test_bullish_direction_blocks_short(self):
        """Direction reader bullish must block short signal."""
        from strategies.builtin import DayTradeMomentumShortStrategy

        strategy = DayTradeMomentumShortStrategy(MagicMock())

        market_data = MagicMock()
        market_data.symbol = 'GOOGL'
        market_data.close = 175.0
        market_data.open = 178.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 45,
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 2.5,
            'low_20': 176.0,
            'sma_20': 178.0,
            'macd': -0.5,
            'macd_signal': -0.3,
            'day_change_pct': -2.0,
        }

        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='mixed',
                time_of_day='morning',
                spy_change_pct=0.0,
                vix_change_pct=2.0,
                sector_etf='XLC',
                reason='mixed test',
            )

            with patch('core.direction_reader.read_direction') as mock_dr:
                mock_dr.return_value = MagicMock(
                    direction=3.0,  # KEY: Bullish direction (positive)
                    phase='middle',
                    ema_stack='aligned_up',
                    allows_short_entry=False,  # KEY: Direction reader blocks shorts
                    reason='mild uptrend',
                )

                with patch('core.config.Config') as mock_cfg:
                    mock_cfg_instance = MagicMock()
                    mock_cfg_instance.ENABLE_DAY_TRADE_SHORT = True
                    mock_cfg_instance.HANDS_OFF_DENYLIST = frozenset({'MU', 'HQGE', 'SPCX'})
                    mock_cfg.return_value = mock_cfg_instance

                    signal = await strategy.generate_signal_with_commentary(market_data)

        assert signal is None, (
            "Direction reader allows_short_entry=False must block short signal — "
            "don't short when direction is bullish"
        )

    @pytest.mark.asyncio
    async def test_exhausted_phase_blocks_short(self):
        """Direction reader exhausted phase must block short — bounce risk."""
        from strategies.builtin import DayTradeMomentumShortStrategy

        strategy = DayTradeMomentumShortStrategy(MagicMock())

        market_data = MagicMock()
        market_data.symbol = 'MSFT'
        market_data.close = 400.0
        market_data.open = 420.0
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 22,  # Oversold
            'volume_ratio': 2.0,
            'adx': 30,
            'atr': 6.0,
            'low_20': 405.0,
            'sma_20': 415.0,
            'macd': -2.0,
            'macd_signal': -1.5,
            'day_change_pct': -5.0,
        }

        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',
                time_of_day='morning',
                spy_change_pct=-1.0,
                vix_change_pct=10.0,
                sector_etf='XLK',
                reason='risk_off test',
            )

            with patch('core.direction_reader.read_direction') as mock_dr:
                mock_dr.return_value = MagicMock(
                    direction=-6.0,  # Bearish
                    phase='exhausted',  # KEY: Exhausted phase
                    ema_stack='aligned_down',
                    allows_short_entry=True,  # Would allow, but phase check blocks
                    reason='strong downtrend at exhaustion',
                )

                with patch('core.config.Config') as mock_cfg:
                    mock_cfg_instance = MagicMock()
                    mock_cfg_instance.ENABLE_DAY_TRADE_SHORT = True
                    mock_cfg_instance.HANDS_OFF_DENYLIST = frozenset({'MU', 'HQGE', 'SPCX'})
                    mock_cfg.return_value = mock_cfg_instance

                    signal = await strategy.generate_signal_with_commentary(market_data)

        assert signal is None, (
            "Direction reader phase=exhausted must block short signal — "
            "oversold bounce risk is too high"
        )


class TestDayTradeShortContinuationPattern:
    """Test continuation-down pattern for shorts."""

    @pytest.mark.asyncio
    async def test_continuation_down_pattern_detected(self):
        """Continuation-down pattern: RSI 30-50, below SMA20, MACD bearish."""
        from strategies.builtin import DayTradeMomentumShortStrategy

        strategy = DayTradeMomentumShortStrategy(MagicMock())

        market_data = MagicMock()
        market_data.symbol = 'INTC'
        market_data.close = 25.0  # Not below low_20 (so not breakdown)
        market_data.open = 26.5
        market_data.timestamp = datetime.now()
        market_data.indicators = {
            'rsi': 42,  # KEY: 30 <= RSI <= 50 for continuation
            'volume_ratio': 2.0,
            'adx': 25,
            'atr': 0.4,
            'low_20': 24.5,  # Close NOT below low_20
            'high_20': 28.0,
            'sma_20': 26.0,  # Close BELOW sma_20
            'sma_50': 27.0,
            'macd': -0.3,  # KEY: MACD < signal (bearish)
            'macd_signal': -0.1,
            'day_change_pct': -4.0,
        }

        with patch('core.market_context.read_market_context') as mock_mc:
            mock_mc.return_value = MagicMock(
                regime='risk_off',
                time_of_day='morning',
                spy_change_pct=0.2,
                vix_change_pct=5.0,
                sector_etf='XLK',
                reason='risk_off test',
            )

            with patch('core.direction_reader.read_direction') as mock_dr:
                mock_dr.return_value = MagicMock(
                    direction=-4.0,
                    phase='middle',
                    ema_stack='aligned_down',
                    allows_short_entry=True,
                    reason='downtrend',
                )

                with patch('core.config.Config') as mock_cfg:
                    mock_cfg_instance = MagicMock()
                    mock_cfg_instance.ENABLE_DAY_TRADE_SHORT = True
                    mock_cfg_instance.HANDS_OFF_DENYLIST = frozenset({'MU', 'HQGE', 'SPCX'})
                    mock_cfg_instance.DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED = True
                    mock_cfg_instance.ENABLE_DAY_TRADE_SHORT_SHADOW = True
                    mock_cfg_instance.DAY_TRADE_SHORT_MIN_WEAK_RS_VS_SPY = 0.5
                    mock_cfg_instance.DAY_TRADE_SHORT_RSI_FLOOR = 30.0
                    mock_cfg_instance.DAY_TRADE_SHORT_RSI_CEILING = 85.0
                    mock_cfg_instance.MOMENTUM_MIN_VOLUME_RATIO = 1.5
                    mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED = False
                    mock_cfg_instance.DAY_TRADE_SIZE_MULTIPLIER = 0.5
                    mock_cfg_instance.DAY_TRADE_FLATTEN_HOUR = 15
                    mock_cfg.return_value = mock_cfg_instance

                    signal = await strategy.generate_signal_with_commentary(market_data)

        assert signal is not None, "Continuation-down pattern should generate signal"
        assert signal.reasoning.get('entry_pattern') == 'continuation_down', (
            "Pattern should be identified as continuation_down"
        )
