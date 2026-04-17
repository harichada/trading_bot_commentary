import logging
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np

from core.models import CommentaryType, SignalType, TradingSignal, MarketData
from core.commentary import TradingCommentary
from strategies.base import TradingStrategyWithCommentary

logger = logging.getLogger('TradingBot')


class NewsSignalStrategy(TradingStrategyWithCommentary):
    """Trading strategy based on news sentiment"""

    def __init__(self, commentary_system, news_aggregator, sentiment_analyzer):
        super().__init__(commentary_system)
        self.news_aggregator = news_aggregator
        self.sentiment_analyzer = sentiment_analyzer

    async def generate_signal_with_commentary(self, market_data) -> Optional[TradingSignal]:
        """Generate signal based on news sentiment"""
        news_items = await self.news_aggregator.fetch_news(market_data.symbol)

        if not news_items:
            return None

        # Analyze sentiment
        sentiments = []
        for item in news_items:
            sentiment = self.sentiment_analyzer.analyze(item.headline + " " + item.summary)
            sentiments.append(sentiment['compound'])

        avg_sentiment = np.mean(sentiments)

        # Generate signal if sentiment is strong
        if abs(avg_sentiment) > 0.3:
            signal_type = SignalType.BUY if avg_sentiment > 0 else SignalType.SELL

            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.FUNDAMENTAL,
                symbol=market_data.symbol,
                title=f"News Signal: {signal_type.name}",
                message=f"Strong {'positive' if avg_sentiment > 0 else 'negative'} news sentiment detected.",
                data={'avg_sentiment': avg_sentiment, 'news_count': len(news_items)},
                importance=7
            ))

            # ATR-scaled stops
            from core.config import Config
            atr = float(getattr(market_data, 'indicators', {}).get('atr', market_data.close * 0.02))
            atr_mult = Config().ATR_STOP_MULTIPLIER
            rr_ratio = Config().ATR_REWARD_RISK_RATIO
            stop_distance = atr_mult * atr
            if signal_type == SignalType.BUY:
                stop_loss = market_data.close - stop_distance
                take_profit = market_data.close + (rr_ratio * stop_distance)
            else:
                stop_loss = market_data.close + stop_distance
                take_profit = market_data.close - (rr_ratio * stop_distance)

            return TradingSignal(
                symbol=market_data.symbol,
                signal_type=signal_type,
                strength=abs(avg_sentiment),
                entry_price=market_data.close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=0,
                reasoning={'strategy': 'news_sentiment', 'sentiment': avg_sentiment,
                           'atr': atr, 'atr_mult': atr_mult, 'stop_distance': stop_distance},
                confidence=min(abs(avg_sentiment) * 1.5, 0.8)
            )

        return None

class BreakoutStrategyWithCommentary(TradingStrategyWithCommentary):
    """Breakout strategy with detailed explanations"""

    name = "breakout"

    async def generate_signal_with_commentary(self, market_data) -> Optional[TradingSignal]:
        indicators = market_data.indicators

        try:
            # Check for breakout with proper type conversion
            resistance_1 = float(indicators.get('resistance_1', 0))
            support_1 = float(indicators.get('support_1', 0))
            volume_ratio = float(indicators.get('volume_ratio', 1))

            # Validate values
            if np.isnan(resistance_1) or resistance_1 <= 0 or np.isnan(support_1) or support_1 <= 0:
                self._log_decision(market_data, "skip", "invalid_indicators",
                                   resistance=resistance_1, support=support_1)
                return None

            if not (resistance_1 > 0 and market_data.close > resistance_1):
                self._log_decision(market_data, "skip", "no_breakout",
                                   resistance=resistance_1, volume_ratio=round(volume_ratio, 2))
                return None

            # Breakout detected — check volume confirmation
            volume_surge = market_data.volume > volume_ratio * 1.5

            if not volume_surge:
                self._log_decision(market_data, "skip", "low_volume",
                                   resistance=resistance_1, volume_ratio=round(volume_ratio, 2))
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.RISK_ASSESSMENT,
                    symbol=market_data.symbol,
                    title=f"⛔ Breakout Skipped — Low Volume",
                    message=(f"Price broke above ${resistance_1:.2f} but volume_ratio "
                             f"{volume_ratio:.2f} < 1.5x. Breakouts without volume "
                             "typically fail."),
                    data={'breakout_level': resistance_1, 'volume_ratio': volume_ratio},
                    importance=6,
                ))
                return None

            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.OPPORTUNITY,
                symbol=market_data.symbol,
                title=f"🚀 Breakout Detected!",
                message=f"Price broke above resistance at ${resistance_1:.2f}. "
                       f"Volume confirms breakout!",
                data={
                    'breakout_level': resistance_1,
                    'current_price': market_data.close,
                    'volume_surge': True,
                    'distance_from_resistance': ((market_data.close - resistance_1) / resistance_1) * 100
                },
                confidence=0.8,
                importance=8
            ))

            # ATR-scaled stops: wider stops for volatile stocks, tighter for calm
            from core.config import Config
            atr = float(indicators.get('atr', market_data.close * 0.02))
            atr_mult = Config().ATR_STOP_MULTIPLIER
            rr_ratio = Config().ATR_REWARD_RISK_RATIO
            stop_distance = atr_mult * atr
            stop_loss = market_data.close - stop_distance
            take_profit = market_data.close + (rr_ratio * stop_distance)

            self._log_decision(market_data, "signal_buy", "breakout_with_volume",
                               resistance=resistance_1, volume_ratio=round(volume_ratio, 2),
                               stop=round(stop_loss, 2), target=round(take_profit, 2),
                               atr=round(atr, 3), atr_mult=atr_mult,
                               stop_dist=round(stop_distance, 2))
            return TradingSignal(
                symbol=market_data.symbol,
                signal_type=SignalType.BUY,
                strength=0.8,
                entry_price=market_data.close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=0,
                reasoning={
                    'strategy': 'breakout',
                    'breakout_level': resistance_1,
                    'volume_confirmation': True,
                    'atr': atr,
                    'atr_mult': atr_mult,
                    'stop_distance': stop_distance,
                },
                confidence=0.75
            )
        except Exception as e:
            self._log_decision(market_data, "error", "exception", err=str(e))
            logger.debug(f"Breakout strategy error for {market_data.symbol}: {e}")
            return None

class MeanReversionStrategyWithCommentary(TradingStrategyWithCommentary):
    """Mean reversion strategy with explanations and tunable parameters"""

    name = "mean_reversion"

    def __init__(self, commentary_system, rsi_threshold=30, bb_window=20, stop_loss_mult=0.98, take_profit_mult=1.0):
        super().__init__(commentary_system)
        self.rsi_threshold = rsi_threshold
        self.bb_window = bb_window
        self.stop_loss_mult = stop_loss_mult
        self.take_profit_mult = take_profit_mult

    async def generate_signal_with_commentary(self, market_data) -> Optional[TradingSignal]:
        indicators = market_data.indicators
        try:
            rsi = float(indicators.get('rsi', 50))
            bb_lower = float(indicators.get('bb_lower', 0))
            bb_middle = float(indicators.get('bb_middle', market_data.close))
            bb_upper = float(indicators.get('bb_upper', market_data.close))
            if np.isnan(rsi) or np.isnan(bb_lower) or bb_lower <= 0:
                self._log_decision(market_data, "skip", "invalid_indicators",
                                   rsi=rsi, bb_lower=bb_lower)
                return None
            if rsi < self.rsi_threshold and market_data.close < bb_lower:
                # Trend filter: don't catch a falling knife.
                # Skip the long when price is below MA50 AND momentum is bearish.
                # This is the LCID 2026-04-14 setup: oversold inside a downtrend
                # rarely mean-reverts cleanly; it usually keeps falling.
                sma_50 = float(indicators.get('sma_50', 0))
                macd_val = float(indicators.get('macd', 0))
                macd_signal_val = float(indicators.get('macd_signal', 0))
                if (sma_50 > 0 and market_data.close < sma_50
                        and macd_val < macd_signal_val):
                    self._log_decision(market_data, "skip", "falling_knife",
                                       rsi=round(rsi, 2), sma_50=round(sma_50, 2),
                                       macd=round(macd_val, 4))
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=market_data.symbol,
                        title=f"⛔ Mean Reversion Skipped — Falling Knife",
                        message=(f"RSI {rsi:.1f} oversold but price below MA50 "
                                 f"and MACD bearish. Avoiding catch-the-knife setup."),
                        data={'rsi': rsi, 'close': market_data.close,
                              'sma_50': sma_50, 'macd': macd_val,
                              'macd_signal': macd_signal_val},
                        importance=6
                    ))
                    return None

                distance_from_mean = ((bb_middle - market_data.close) / market_data.close) * 100
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.OPPORTUNITY,
                    symbol=market_data.symbol,
                    title=f"🔄 Mean Reversion Setup",
                    message=f"Oversold conditions detected - RSI at {rsi:.1f} and price below Bollinger Band. "
                            f"Price is {distance_from_mean:.1f}% below the mean.",
                    data={
                        'rsi': rsi,
                        'bollinger_position': 'below_lower_band',
                        'distance_from_mean': distance_from_mean,
                        'bb_lower': bb_lower,
                        'bb_middle': bb_middle
                    },
                    confidence=0.65,
                    importance=7
                ))
                from core.config import Config
                atr = float(indicators.get('atr', market_data.close * 0.02))
                atr_mult = Config().ATR_STOP_MULTIPLIER
                rr_ratio = Config().ATR_REWARD_RISK_RATIO
                stop_distance = atr_mult * atr
                stop_loss = market_data.close - stop_distance
                rr_target = market_data.close + (rr_ratio * stop_distance)
                take_profit = min(bb_middle * self.take_profit_mult, rr_target)
                self._log_decision(market_data, "signal_buy", "oversold_bounce",
                                   rsi=round(rsi, 2), distance_pct=round(distance_from_mean, 2),
                                   stop=round(stop_loss, 2), target=round(take_profit, 2),
                                   atr=round(atr, 3), stop_dist=round(stop_distance, 2))
                return TradingSignal(
                    symbol=market_data.symbol,
                    signal_type=SignalType.BUY,
                    strength=0.7,
                    entry_price=market_data.close,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=0,
                    reasoning={
                        'strategy': 'mean_reversion',
                        'rsi': rsi,
                        'bb_position': 'below_lower_band',
                        'distance_from_mean': distance_from_mean,
                        'atr': atr, 'atr_mult': atr_mult,
                        'stop_distance': stop_distance,
                    },
                    confidence=0.65
                )
            elif rsi > 70 and market_data.close > bb_upper:
                distance_from_mean = ((market_data.close - bb_middle) / market_data.close) * 100
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.OPPORTUNITY,
                    symbol=market_data.symbol,
                    title=f"🔻 Short Setup - Mean Reversion",
                    message=f"Overbought conditions - RSI at {rsi:.1f} and price above upper Bollinger Band. "
                            f"Price is {distance_from_mean:.1f}% above the mean.",
                    data={
                        'rsi': rsi,
                        'bollinger_position': 'above_upper_band',
                        'distance_from_mean': distance_from_mean
                    },
                    confidence=0.65,
                    importance=7
                ))
                from core.config import Config
                atr = float(indicators.get('atr', market_data.close * 0.02))
                atr_mult = Config().ATR_STOP_MULTIPLIER
                rr_ratio = Config().ATR_REWARD_RISK_RATIO
                stop_distance = atr_mult * atr
                stop_loss = market_data.close + stop_distance
                rr_target = market_data.close - (rr_ratio * stop_distance)
                take_profit = max(bb_middle, rr_target)
                self._log_decision(market_data, "signal_sell", "overbought_fade",
                                   rsi=round(rsi, 2), distance_pct=round(distance_from_mean, 2),
                                   stop=round(stop_loss, 2), target=round(take_profit, 2),
                                   atr=round(atr, 3), stop_dist=round(stop_distance, 2))
                return TradingSignal(
                    symbol=market_data.symbol,
                    signal_type=SignalType.SELL,
                    strength=0.7,
                    entry_price=market_data.close,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=0,
                    reasoning={
                        'strategy': 'mean_reversion_short',
                        'rsi': rsi,
                        'bb_position': 'above_upper_band',
                        'distance_from_mean': distance_from_mean,
                        'atr': atr, 'atr_mult': atr_mult,
                        'stop_distance': stop_distance,
                    },
                    confidence=0.65
                )
        except Exception as e:
            self._log_decision(market_data, "error", "exception", err=str(e))
            logger.debug(f"Mean reversion strategy error for {market_data.symbol}: {e}")
            return None
        self._log_decision(market_data, "skip", "no_setup",
                           rsi=round(rsi, 2), bb_lower=round(bb_lower, 2),
                           close=round(market_data.close, 2))
        return None

class MomentumStrategyWithCommentary(TradingStrategyWithCommentary):
    """Momentum strategy with explanations"""

    name = "momentum"

    async def generate_signal_with_commentary(self, market_data) -> Optional[TradingSignal]:
        indicators = market_data.indicators

        try:
            # Check for momentum with proper type conversion
            macd = float(indicators.get('macd', 0))
            macd_signal = float(indicators.get('macd_signal', 0))
            rsi = float(indicators.get('rsi', 50))
            adx = float(indicators.get('adx', 0))

            # Validate values
            if np.isnan(macd) or np.isnan(macd_signal) or np.isnan(rsi) or np.isnan(adx):
                self._log_decision(market_data, "skip", "invalid_indicators",
                                   macd=macd, rsi=rsi, adx=adx)
                return None

            if macd > macd_signal and 50 < rsi < 70 and adx > 25:

                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.OPPORTUNITY,
                    symbol=market_data.symbol,
                    title=f"📈 Momentum Building",
                    message=f"Strong momentum detected: MACD bullish crossover, RSI at {rsi:.1f} (healthy), "
                           f"ADX at {adx:.1f} (strong trend)",
                    data={
                        'macd_crossover': True,
                        'macd': macd,
                        'macd_signal': macd_signal,
                        'rsi': rsi,
                        'adx': adx,
                        'trend_strength': 'strong' if adx > 30 else 'moderate'
                    },
                    confidence=0.7,
                    importance=7
                ))

                # ATR-scaled stops/targets. Previous fixed 3% stop / 6% target
                # was a daily-bar swing setup; on 5-min bars only 12.5% of trades
                # ever hit the 6% target while 47% timed out — backtest 2026-04-14.
                # 1.5x ATR stop / 3x ATR target = 2:1 R:R, achievable in ~10-20
                # bars given typical intraday volatility.
                atr = float(indicators.get('atr', market_data.close * 0.005))
                stop_loss = market_data.close - 1.5 * atr
                take_profit = market_data.close + 3.0 * atr

                self._log_decision(market_data, "signal_buy", "macd_rsi_adx_aligned",
                                   macd=round(macd, 4), rsi=round(rsi, 2), adx=round(adx, 2),
                                   atr=round(atr, 3),
                                   stop=round(stop_loss, 2), target=round(take_profit, 2))
                return TradingSignal(
                    symbol=market_data.symbol,
                    signal_type=SignalType.BUY,
                    strength=0.75,
                    entry_price=market_data.close,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=0,
                    reasoning={
                        'strategy': 'momentum',
                        'macd_bullish': True,
                        'rsi_healthy': True,
                        'trend_strong': adx > 25
                    },
                    confidence=0.7
                )
        except Exception as e:
            self._log_decision(market_data, "error", "exception", err=str(e))
            logger.debug(f"Momentum strategy error for {market_data.symbol}: {e}")
            return None

        self._log_decision(market_data, "skip", "no_setup",
                           macd=round(macd, 4), rsi=round(rsi, 2), adx=round(adx, 2))
        return None
