import logging
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import ta

from core.models import CommentaryType
from core.commentary import TradingCommentary

logger = logging.getLogger('TradingBot')


class TechnicalAnalyzerWithCommentary:
    """Technical analyzer that explains its analysis"""

    def __init__(self, commentary_system):
        self.commentary = commentary_system

    async def analyze_with_commentary(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Perform technical analysis with commentary"""
        if data is None or data.empty:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title=f"\u26a0\ufe0f No Data Available: {symbol}",
                message="No price data available for technical analysis",
                importance=5
            ))
            return {}

        # Check if we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title=f"\u26a0\ufe0f Missing Data Columns: {symbol}",
                message=f"Missing required columns: {', '.join(missing_columns)}",
                importance=5
            ))
            return {}

        if len(data) < 50:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title=f"\u26a0\ufe0f Insufficient Data: {symbol}",
                message=f"Only {len(data)} candles available, need at least 50 for analysis",
                importance=5
            ))
            return {}

        indicators = {}

        try:
            # Ensure data is numeric and handle any non-numeric values
            numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in numeric_columns:
                if col in data.columns:
                    # Convert to numeric, replacing any non-numeric with NaN
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    # Fill NaN values with forward fill, then backward fill
                    data[col] = data[col].ffill().bfill()

            # Drop any remaining rows with NaN values
            data = data.dropna()

            if len(data) < 50:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"\u26a0\ufe0f Data Quality Issue: {symbol}",
                    message=f"After cleaning, only {len(data)} valid candles remain",
                    importance=5
                ))
                return {}

            # Convert to numpy arrays with proper type
            close = data['Close'].values.astype(np.float64)
            high = data['High'].values.astype(np.float64)
            low = data['Low'].values.astype(np.float64)
            volume = data['Volume'].values.astype(np.float64)

            # Moving averages
            indicators['sma_20'] = ta.trend.sma_indicator(data['Close'], window=20).iloc[-1]
            indicators['sma_50'] = ta.trend.sma_indicator(data['Close'], window=50).iloc[-1]
            indicators['ema_20'] = ta.trend.ema_indicator(data['Close'], window=20).iloc[-1]

            # RSI
            indicators['rsi'] = ta.momentum.rsi(data['Close'], window=14).iloc[-1]

            # MACD
            macd = ta.trend.MACD(data['Close'])
            indicators['macd'] = macd.macd().iloc[-1]
            indicators['macd_signal'] = macd.macd_signal().iloc[-1]
            indicators['macd_histogram'] = macd.macd_diff().iloc[-1]

            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(data['Close'], window=20, window_dev=2)
            indicators['bb_upper'] = bollinger.bollinger_hband().iloc[-1]
            indicators['bb_middle'] = bollinger.bollinger_mavg().iloc[-1]
            indicators['bb_lower'] = bollinger.bollinger_lband().iloc[-1]

            # ATR
            indicators['atr'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14).iloc[-1]

            # Volume indicators
            indicators['obv'] = ta.volume.on_balance_volume(data['Close'], data['Volume']).iloc[-1]

            # ADX
            adx = ta.trend.ADXIndicator(data['High'], data['Low'], data['Close'], window=14)
            indicators['adx'] = adx.adx().iloc[-1]

            # Support/Resistance — single-bar pivot (legacy, kept for ML features)
            pivot = (high[-1] + low[-1] + close[-1]) / 3
            indicators['pivot'] = float(pivot)
            indicators['resistance_1'] = float(2 * pivot - low[-1])
            indicators['support_1'] = float(2 * pivot - high[-1])

            # Multi-bar resistance/support — 20-bar high/low (real breakout levels)
            if len(high) >= 20:
                indicators['high_20'] = float(np.max(high[-20:]))
                indicators['low_20'] = float(np.min(low[-20:]))
            else:
                indicators['high_20'] = float(np.max(high))
                indicators['low_20'] = float(np.min(low))

            # v-oversold-v2-indicators-2026-05-19: prev_low, lows_50 for OversoldBounceV2Strategy
            if len(low) >= 2:
                indicators['prev_low'] = float(low[-2])
            indicators['lows_50'] = low[-50:].tolist() if len(low) >= 50 else (low.tolist() if len(low) >= 10 else [])

            # Add calculated metrics for ML
            indicators['returns'] = float((close[-1] - close[-2]) / close[-2]) if len(close) > 1 else 0.0
            indicators['volume_ratio'] = float(volume[-1] / np.mean(volume[-20:])) if len(volume) > 20 and np.mean(volume[-20:]) > 0 else 1.0
            indicators['high_low_ratio'] = float((high[-1] - low[-1]) / close[-1]) if close[-1] > 0 else 0.02

            # v-direction-reader-features-2026-05-28: slope-over-time
            # features required by the direction reader. Snapshot
            # indicators (rsi, ema_20, macd) tell you WHERE the stock is.
            # Slopes tell you WHICH WAY it's heading. Direction reading
            # without slopes is a still photo; with slopes it's a movie.
            #
            # All slopes are computed as % change over a window of bars,
            # which normalizes across stocks (a $5 stock and a $500 stock
            # both express their trend strength in the same units).
            try:
                # EMA-20 slope over last 5 bars: trend direction + steepness.
                ema_20_series = ta.trend.ema_indicator(
                    data['Close'], window=20
                )
                if len(ema_20_series) >= 6 and ema_20_series.iloc[-6] > 0:
                    indicators['ema_20_slope_pct'] = float(
                        (ema_20_series.iloc[-1] - ema_20_series.iloc[-6])
                        / ema_20_series.iloc[-6] * 100
                    )
                else:
                    indicators['ema_20_slope_pct'] = 0.0

                # SMA-50 slope over last 10 bars: dominant trend direction.
                sma_50_series = ta.trend.sma_indicator(
                    data['Close'], window=50
                )
                if len(sma_50_series) >= 11 and sma_50_series.iloc[-11] > 0:
                    indicators['sma_50_slope_pct'] = float(
                        (sma_50_series.iloc[-1] - sma_50_series.iloc[-11])
                        / sma_50_series.iloc[-11] * 100
                    )
                else:
                    indicators['sma_50_slope_pct'] = 0.0

                # RSI slope over last 3 bars: momentum building or fading.
                rsi_series = ta.momentum.rsi(data['Close'], window=14)
                if len(rsi_series) >= 4:
                    indicators['rsi_slope'] = float(
                        rsi_series.iloc[-1] - rsi_series.iloc[-4]
                    )
                else:
                    indicators['rsi_slope'] = 0.0

                # OBV slope over last 10 bars: smart-money direction
                # via volume. Rising OBV in flat-price = accumulation;
                # falling OBV in flat-price = distribution.
                obv_series = ta.volume.on_balance_volume(
                    data['Close'], data['Volume']
                )
                if len(obv_series) >= 11 and abs(obv_series.iloc[-11]) > 0:
                    indicators['obv_slope_pct'] = float(
                        (obv_series.iloc[-1] - obv_series.iloc[-11])
                        / abs(obv_series.iloc[-11]) * 100
                    )
                else:
                    indicators['obv_slope_pct'] = 0.0

                # MACD histogram direction over last 3 bars: trend
                # acceleration/deceleration. Rising histogram = trend
                # gaining force; falling histogram = trend losing steam.
                macd_hist_series = macd.macd_diff()
                if len(macd_hist_series) >= 4:
                    indicators['macd_hist_slope'] = float(
                        macd_hist_series.iloc[-1] - macd_hist_series.iloc[-4]
                    )
                else:
                    indicators['macd_hist_slope'] = 0.0

                # Close-vs-SMA50 percentage: location within the trend
                # cycle. Far above SMA50 = late-trend (extended);
                # near SMA50 = pulling back; far below = downtrend.
                if indicators.get('sma_50', 0) > 0:
                    indicators['close_vs_sma50_pct'] = float(
                        (close[-1] - indicators['sma_50'])
                        / indicators['sma_50'] * 100
                    )
                else:
                    indicators['close_vs_sma50_pct'] = 0.0
            except Exception as exc:
                # Slope features are nice-to-have, not load-bearing.
                # If anything fails, default to neutral (0.0) so the
                # direction reader treats this as "no trend signal".
                logger.debug("direction-reader slope features failed: %s", exc)
                for k in (
                    'ema_20_slope_pct', 'sma_50_slope_pct', 'rsi_slope',
                    'obv_slope_pct', 'macd_hist_slope', 'close_vs_sma50_pct',
                ):
                    indicators.setdefault(k, 0.0)

            # Ensure all indicators are float type
            # v-oversold-v2-indicators-2026-05-19: skip non-scalar entries
            # (e.g. lows_50 is a list) so the float-coercion loop doesn't
            # blow up on container indicators.
            for key, value in indicators.items():
                if isinstance(value, (list, tuple)):
                    continue
                if isinstance(value, (np.floating, np.integer)):
                    indicators[key] = float(value)
                elif np.isnan(value) or np.isinf(value):
                    indicators[key] = 0.0

            # Interpret indicators
            interpretations = []

            # Trend analysis
            if close[-1] > indicators['sma_50']:
                interpretations.append("\U0001f4c8 Price above 50 SMA - Uptrend")
            else:
                interpretations.append("\U0001f4c9 Price below 50 SMA - Downtrend")

            # RSI analysis
            if indicators['rsi'] > 70:
                interpretations.append(f"\U0001f525 RSI at {indicators['rsi']:.1f} - Overbought")
            elif indicators['rsi'] < 30:
                interpretations.append(f"\u2744\ufe0f RSI at {indicators['rsi']:.1f} - Oversold")
            else:
                interpretations.append(f"\u2796 RSI at {indicators['rsi']:.1f} - Neutral")

            # MACD analysis
            if indicators['macd'] > indicators['macd_signal']:
                interpretations.append("\U0001f7e2 MACD above signal - Bullish momentum")
            else:
                interpretations.append("\U0001f534 MACD below signal - Bearish momentum")

            # Bollinger Bands
            if close[-1] > indicators['bb_upper']:
                interpretations.append("\U0001f4ca Price above upper Bollinger Band - Overbought")
            elif close[-1] < indicators['bb_lower']:
                interpretations.append("\U0001f4ca Price below lower Bollinger Band - Oversold")

            # ADX trend strength
            if indicators['adx'] > 25:
                interpretations.append(f"\U0001f4aa ADX at {indicators['adx']:.1f} - Strong trend")
            else:
                interpretations.append(f"\U0001f634 ADX at {indicators['adx']:.1f} - Weak trend")

            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.TECHNICAL,
                symbol=symbol,
                title=f"\U0001f4ca Technical Analysis: {symbol}",
                message="\n".join(interpretations),
                data={
                    'price': float(close[-1]),
                    'rsi': indicators['rsi'],
                    'sma_20': indicators['sma_20'],
                    'sma_50': indicators['sma_50'],
                    'volume_ratio': indicators['volume_ratio'],
                    'atr': indicators['atr'],
                    'adx': indicators['adx']
                },
                importance=5
            ))

        except Exception as e:
            logger.error(f"Technical analysis failed for {symbol}: {e}", exc_info=True)
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=symbol,
                title=f"\u26a0\ufe0f Technical Analysis Error: {symbol}",
                message=f"Could not complete technical analysis: {str(e)}",
                importance=7
            ))

            # Return basic indicators as fallback
            try:
                close_price = float(data['Close'].iloc[-1])
                indicators = {
                    'sma_20': close_price,
                    'sma_50': close_price,
                    'ema_20': close_price,
                    'rsi': 50.0,
                    'macd': 0.0,
                    'macd_signal': 0.0,
                    'macd_histogram': 0.0,
                    'bb_upper': close_price * 1.02,
                    'bb_middle': close_price,
                    'bb_lower': close_price * 0.98,
                    'atr': close_price * 0.01,
                    'obv': 0.0,
                    'adx': 25.0,
                    'pivot': close_price,
                    'resistance_1': close_price * 1.01,
                    'support_1': close_price * 0.99,
                    'returns': 0.0,
                    'volume_ratio': 1.0,
                    'high_low_ratio': 0.02
                }
            except Exception as e:
                logger.debug(f"Error computing default indicators: {e}")
                indicators = {}

        return indicators
