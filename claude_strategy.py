#!/usr/bin/env python3
"""
Claude AI Trading Strategy
Uses the Anthropic Claude API to make trading decisions based on configurable
data factors (price action, technical indicators, news/sentiment, market regime).
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import aiohttp
except ImportError:
    aiohttp = None

from strategy_system import (
    BaseStrategy, StrategyConfig, StrategySignal, SignalType,
    MarketRegimeDetector, RegimeAnalysis
)

logger = logging.getLogger('ClaudeStrategy')


# =============================================================================
# FACTOR MODULES - Each prepares a text section for the prompt
# =============================================================================

class PriceActionFactor:
    """Formats recent OHLCV candles, price structure, volume, and momentum."""

    def format(self, df: pd.DataFrame, symbol: str) -> str:
        if df is None or len(df) < 5:
            return ""

        recent = df.tail(20)
        lines = [f"## Price Action ({symbol})"]

        # Recent candles (last 10)
        lines.append("Recent candles (newest first):")
        candles = recent.tail(10)
        for i in range(len(candles) - 1, max(len(candles) - 10, -1), -1):
            row = candles.iloc[i]
            o, h, l, c = row.get('open', 0), row.get('high', 0), row.get('low', 0), row.get('close', 0)
            vol = row.get('volume', 0)
            change_pct = ((c - o) / o * 100) if o != 0 else 0
            body_pct = abs(c - o) / (h - l) * 100 if (h - l) > 0 else 0
            direction = "UP" if c >= o else "DN"
            lines.append(f"  {direction} O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{c:.2f} V:{vol:,.0f} ({change_pct:+.2f}%) body:{body_pct:.0f}%")

        # Today's session summary (group by date for intraday data)
        closes = df['close'].values
        current = closes[-1]
        try:
            if hasattr(df.index, 'date'):
                today = df.index[-1].date()
                today_bars = df[df.index.date == today]
                if len(today_bars) >= 1:
                    today_open = float(today_bars['open'].iloc[0])
                    today_high = float(today_bars['high'].max())
                    today_low = float(today_bars['low'].min())
                    today_vol = int(today_bars['volume'].sum()) if 'volume' in today_bars.columns else 0
                    today_change = (current / today_open - 1) * 100
                    today_range_pos = ((current - today_low) / (today_high - today_low) * 100) if (today_high - today_low) > 0 else 50
                    lines.append(f"\nToday's session ({today}, {len(today_bars)} bars so far):")
                    lines.append(f"  Open: ${today_open:.2f} | High: ${today_high:.2f} | Low: ${today_low:.2f} | Current: ${current:.2f}")
                    lines.append(f"  Day change: {today_change:+.2f}% | Position in today's range: {today_range_pos:.0f}%")
                    lines.append(f"  Day volume: {today_vol:,}")

                    # Previous day's close (for gap context)
                    prev_days = df[df.index.date < today]
                    if len(prev_days) >= 1:
                        prev_close = float(prev_days['close'].iloc[-1])
                        prev_high = float(prev_days[prev_days.index.date == prev_days.index[-1].date()]['high'].max())
                        prev_low = float(prev_days[prev_days.index.date == prev_days.index[-1].date()]['low'].min())
                        gap = (today_open / prev_close - 1) * 100
                        gap_type = "GAP UP" if gap > 0.1 else "GAP DOWN" if gap < -0.1 else "FLAT OPEN"
                        lines.append(f"  Previous close: ${prev_close:.2f} | {gap_type} ({gap:+.2f}%)")
                        lines.append(f"  Previous day range: ${prev_low:.2f} - ${prev_high:.2f}")
        except Exception:
            pass  # Fall through if index isn't datetime

        # Bar-to-bar price changes for momentum context
        if len(closes) >= 2:
            last_change = (closes[-1] / closes[-2] - 1) * 100
            lines.append(f"\nLast bar change: {last_change:+.2f}%")
        if len(closes) >= 6:
            change_5 = (closes[-1] / closes[-5] - 1) * 100
            lines.append(f"Change over last 5 bars: {change_5:+.2f}%")
        if len(closes) >= 11:
            change_10 = (closes[-1] / closes[-10] - 1) * 100
            lines.append(f"Change over last 10 bars: {change_10:+.2f}%")
        if len(closes) >= 21:
            change_20 = (closes[-1] / closes[-20] - 1) * 100
            lines.append(f"Change over last 20 bars: {change_20:+.2f}%")

        # Volume analysis
        volumes = df['volume'].tail(20).values
        if len(volumes) >= 10:
            avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
            current_vol = volumes[-1]
            vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
            vol_trend = "HIGH" if vol_ratio > 1.5 else "LOW" if vol_ratio < 0.5 else "NORMAL"
            lines.append(f"\nVolume: current={current_vol:,.0f} avg={avg_vol:,.0f} ratio={vol_ratio:.1f}x [{vol_trend}]")
            # Volume trend (are recent bars seeing more volume?)
            if len(volumes) >= 10:
                recent_avg_vol = np.mean(volumes[-5:])
                earlier_avg_vol = np.mean(volumes[-10:-5])
                vol_momentum = "INCREASING" if recent_avg_vol > earlier_avg_vol * 1.2 else "DECREASING" if recent_avg_vol < earlier_avg_vol * 0.8 else "STEADY"
                lines.append(f"Volume trend: {vol_momentum}")

        # Price structure analysis
        high_5 = df['high'].tail(5).max()
        low_5 = df['low'].tail(5).min()
        high_20 = df['high'].tail(20).max()
        low_20 = df['low'].tail(20).min()

        # Higher highs / higher lows detection
        recent_highs = df['high'].tail(10).values
        recent_lows = df['low'].tail(10).values
        mid = len(recent_highs) // 2
        hh = recent_highs[mid:].max() > recent_highs[:mid].max()
        hl = recent_lows[mid:].min() > recent_lows[:mid].min()

        structure = "BULLISH (HH+HL)" if (hh and hl) else "BEARISH (LH+LL)" if (not hh and not hl) else "MIXED"
        lines.append(f"\nPrice structure: {structure}")
        lines.append(f"Current: {current:.2f} | 5-bar range: {low_5:.2f}-{high_5:.2f} | 20-bar range: {low_20:.2f}-{high_20:.2f}")

        range_pos = ((current - low_20) / (high_20 - low_20) * 100) if (high_20 - low_20) > 0 else 50
        lines.append(f"Position in 20-bar range: {range_pos:.0f}% (0=bottom, 100=top)")

        # Key levels (recent pivot points)
        if len(df) >= 10:
            yesterday_high = df['high'].iloc[-2] if len(df) >= 2 else None
            yesterday_low = df['low'].iloc[-2] if len(df) >= 2 else None
            if yesterday_high and yesterday_low:
                lines.append(f"Previous bar: H:{yesterday_high:.2f} L:{yesterday_low:.2f}")

        return "\n".join(lines)


class TechnicalIndicatorsFactor:
    """Formats RSI, MACD, Bollinger Bands, ADX, ATR, EMAs, Stochastic, VWAP with interpretation."""

    def format(self, df: pd.DataFrame, symbol: str) -> str:
        if df is None or len(df) < 26:
            return ""

        lines = [f"## Technical Indicators ({symbol})"]
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume'] if 'volume' in df.columns else None

        try:
            import ta as ta_lib

            # RSI with divergence hint
            rsi_series = ta_lib.momentum.RSIIndicator(close, window=14).rsi()
            rsi = rsi_series.iloc[-1]
            rsi_prev = rsi_series.iloc[-2] if len(rsi_series) >= 2 else rsi
            rsi_interp = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
            rsi_dir = "rising" if rsi > rsi_prev else "falling"
            # Check for divergence
            price_rising = close.iloc[-1] > close.iloc[-3] if len(close) >= 3 else True
            rsi_rising = rsi > rsi_series.iloc[-3] if len(rsi_series) >= 3 else True
            divergence = ""
            if price_rising and not rsi_rising:
                divergence = " ⚠ BEARISH DIVERGENCE (price up, RSI down)"
            elif not price_rising and rsi_rising:
                divergence = " ⚠ BULLISH DIVERGENCE (price down, RSI up)"
            lines.append(f"RSI(14): {rsi:.1f} ({rsi_dir}) [{rsi_interp}]{divergence}")

            # Stochastic
            stoch = ta_lib.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
            stoch_k = stoch.stoch().iloc[-1]
            stoch_d = stoch.stoch_signal().iloc[-1]
            stoch_interp = "OVERSOLD" if stoch_k < 20 else "OVERBOUGHT" if stoch_k > 80 else "NEUTRAL"
            stoch_cross = ""
            if stoch_k > stoch_d and stoch.stoch().iloc[-2] <= stoch.stoch_signal().iloc[-2]:
                stoch_cross = " BULLISH CROSS"
            elif stoch_k < stoch_d and stoch.stoch().iloc[-2] >= stoch.stoch_signal().iloc[-2]:
                stoch_cross = " BEARISH CROSS"
            lines.append(f"Stochastic(14,3): %K={stoch_k:.1f} %D={stoch_d:.1f} [{stoch_interp}]{stoch_cross}")

            # MACD with histogram direction
            macd_ind = ta_lib.trend.MACD(close)
            macd_val = macd_ind.macd().iloc[-1]
            macd_signal = macd_ind.macd_signal().iloc[-1]
            macd_hist = macd_ind.macd_diff().iloc[-1]
            macd_hist_prev = macd_ind.macd_diff().iloc[-2] if len(macd_ind.macd_diff()) >= 2 else macd_hist
            macd_interp = "BULLISH" if macd_val > macd_signal else "BEARISH"
            hist_dir = "expanding" if abs(macd_hist) > abs(macd_hist_prev) else "contracting"
            macd_cross = ""
            if macd_val > macd_signal and macd_ind.macd().iloc[-2] <= macd_ind.macd_signal().iloc[-2]:
                macd_cross = " ★ BULLISH CROSSOVER"
            elif macd_val < macd_signal and macd_ind.macd().iloc[-2] >= macd_ind.macd_signal().iloc[-2]:
                macd_cross = " ★ BEARISH CROSSOVER"
            lines.append(f"MACD: {macd_val:.3f} Signal: {macd_signal:.3f} Hist: {macd_hist:.3f} ({hist_dir}) [{macd_interp}]{macd_cross}")

            # Bollinger Bands with squeeze detection
            bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            bb_mid = bb.bollinger_mavg().iloc[-1]
            bb_pct = bb.bollinger_pband().iloc[-1]
            bb_width = (bb_upper - bb_lower) / bb_mid * 100 if bb_mid > 0 else 0
            bb_interp = "ABOVE UPPER (overbought)" if bb_pct > 1 else "BELOW LOWER (oversold)" if bb_pct < 0 else f"{bb_pct:.0%} position"
            squeeze = " [SQUEEZE - low volatility, breakout likely]" if bb_width < 3 else ""
            lines.append(f"Bollinger Bands: Upper:{bb_upper:.2f} Mid:{bb_mid:.2f} Lower:{bb_lower:.2f} Width:{bb_width:.1f}% [{bb_interp}]{squeeze}")

            # ADX with DI+ and DI-
            adx_ind = ta_lib.trend.ADXIndicator(high, low, close, window=14)
            adx_val = adx_ind.adx().iloc[-1]
            di_plus = adx_ind.adx_pos().iloc[-1]
            di_minus = adx_ind.adx_neg().iloc[-1]
            adx_interp = "STRONG TREND" if adx_val > 25 else "WEAK/RANGING"
            trend_dir = "BULLISH (DI+ > DI-)" if di_plus > di_minus else "BEARISH (DI- > DI+)"
            lines.append(f"ADX(14): {adx_val:.1f} DI+:{di_plus:.1f} DI-:{di_minus:.1f} [{adx_interp}, {trend_dir}]")

            # ATR for position sizing context
            atr = ta_lib.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
            atr_pct = (atr / close.iloc[-1]) * 100
            lines.append(f"ATR(14): {atr:.3f} ({atr_pct:.2f}% of price) — suggested stop distance: {atr*1.5:.2f}")

            # EMAs with price relationship
            ema9 = ta_lib.trend.EMAIndicator(close, window=9).ema_indicator().iloc[-1]
            ema21 = ta_lib.trend.EMAIndicator(close, window=21).ema_indicator().iloc[-1]
            ema50 = ta_lib.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1] if len(df) >= 50 else None
            ema_interp = "BULLISH ALIGNMENT (9>21)" if ema9 > ema21 else "BEARISH ALIGNMENT (9<21)"
            price_vs_ema = "above" if close.iloc[-1] > ema9 else "below"
            lines.append(f"EMA9: {ema9:.2f} EMA21: {ema21:.2f}" +
                         (f" EMA50: {ema50:.2f}" if ema50 else "") +
                         f" [price {price_vs_ema} EMA9, {ema_interp}]")

            # VWAP (intraday key level)
            if volume is not None and len(df) >= 5:
                typical_price = (high + low + close) / 3
                cumvol = volume.cumsum()
                vwap = (typical_price * volume).cumsum() / cumvol
                vwap_val = vwap.iloc[-1]
                price_vs_vwap = "ABOVE" if close.iloc[-1] > vwap_val else "BELOW"
                vwap_dist = (close.iloc[-1] / vwap_val - 1) * 100
                lines.append(f"VWAP: {vwap_val:.2f} [price {price_vs_vwap} by {abs(vwap_dist):.2f}%] — key intraday level")

            # Signal summary — count bullish vs bearish signals
            bullish_count = 0
            bearish_count = 0
            if rsi < 30: bullish_count += 1
            elif rsi > 70: bearish_count += 1
            if macd_val > macd_signal: bullish_count += 1
            else: bearish_count += 1
            if ema9 > ema21: bullish_count += 1
            else: bearish_count += 1
            if stoch_k < 20: bullish_count += 1
            elif stoch_k > 80: bearish_count += 1
            if bb_pct < 0: bullish_count += 1
            elif bb_pct > 1: bearish_count += 1
            if di_plus > di_minus: bullish_count += 1
            else: bearish_count += 1

            lines.append(f"\nIndicator consensus: {bullish_count} BULLISH vs {bearish_count} BEARISH signals")

        except Exception as e:
            lines.append(f"(Indicator calculation error: {e})")

        return "\n".join(lines)


class NewsSentimentFactor:
    """Uses NewsSentimentEngine for headlines + sentiment scores."""

    def __init__(self):
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from news_sentiment_widget import get_sentiment_engine
                self._engine = get_sentiment_engine()
            except Exception as e:
                logger.warning(f"Could not initialize news sentiment engine: {e}")
        return self._engine

    async def format_async(self, symbol: str) -> str:
        engine = self._get_engine()
        if engine is None:
            return ""

        lines = [f"## News & Sentiment ({symbol})"]
        try:
            state = await engine.update()

            # Market-wide sentiment
            market_sent = state.get('market_sentiment', {})
            if isinstance(market_sent, dict):
                lines.append(f"Market sentiment: {market_sent.get('overall', 'N/A')}")

            # Symbol-specific sentiment
            symbol_sents = state.get('symbol_sentiments', {})
            if symbol in symbol_sents:
                s = symbol_sents[symbol]
                score = s.get('score', 0) if isinstance(s, dict) else getattr(s, 'score', 0)
                lines.append(f"{symbol} sentiment score: {score:.1f}")

            # Recent headlines
            headlines = state.get('headlines', [])
            relevant = [h for h in headlines
                        if symbol in getattr(h, 'symbols_mentioned', getattr(h, 'keywords', []))][:5]
            if relevant:
                lines.append("\nRecent headlines:")
                for h in relevant:
                    headline_text = getattr(h, 'headline', str(h))
                    score = getattr(h, 'sentiment_score', 0)
                    lines.append(f"  [{score:+.0f}] {headline_text}")
            else:
                lines.append("No recent symbol-specific headlines found.")

            # Confluence modifier
            modifier = engine.get_confluence_modifier(symbol)
            lines.append(f"\nSentiment confluence modifier: {modifier:+.2f}")

        except Exception as e:
            lines.append(f"(Sentiment data unavailable: {e})")

        return "\n".join(lines)

    def format(self, symbol: str) -> str:
        """Sync wrapper for async format."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context - can't use run_until_complete
                return f"## News & Sentiment ({symbol})\n(Async context - sentiment loading deferred)"
            return loop.run_until_complete(self.format_async(symbol))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.format_async(symbol))
            finally:
                loop.close()


class MarketRegimeFactor:
    """Uses MarketRegimeDetector for regime, trend, volatility, S/R levels."""

    def __init__(self):
        self.detector = MarketRegimeDetector()

    def format(self, df: pd.DataFrame, symbol: str) -> str:
        if df is None or len(df) < 20:
            return ""

        lines = [f"## Market Regime ({symbol})"]
        try:
            regime: RegimeAnalysis = self.detector.analyze(df)
            lines.append(f"Regime: {regime.regime.value}")
            lines.append(f"Trend strength (ADX): {regime.trend_strength:.1f}")
            lines.append(f"Trend direction: {regime.trend_direction:+.2f} (-1=bearish, +1=bullish)")
            lines.append(f"Volatility percentile: {regime.volatility_percentile:.0f}%")
            lines.append(f"Momentum: {regime.momentum:+.3f}")
            lines.append(f"Support: {regime.support_level:.2f} | Resistance: {regime.resistance_level:.2f}")
            lines.append(f"Position in range: {regime.position_in_range:.0f}%")
            lines.append(f"Recommended approach: {regime.recommended_strategy}")
            lines.append(f"Confidence: {regime.confidence:.2f}")
        except Exception as e:
            lines.append(f"(Regime analysis error: {e})")

        return "\n".join(lines)


# =============================================================================
# PROMPT BUILDER
# =============================================================================

class PromptBuilder:
    """Assembles structured prompts from enabled factors."""

    SYSTEM_PROMPT = """You are an expert intraday day trader. Your job is to make profitable short-term trades. Analyze the data and return a trading decision as strict JSON.

You MUST return ONLY a JSON object with these exact fields:
{
  "signal": "BUY" | "SELL" | "HOLD",
  "strength": 0.0 to 1.0,
  "reasoning": "brief explanation of your analysis",
  "entry_price": number or null,
  "stop_loss": number or null,
  "take_profit": number or null,
  "risk_notes": "any risk warnings or caveats"
}

Signal meanings:
- BUY = go long (if flat) or cover/exit short (if short)
- SELL = go short (if flat) or close/exit long (if long)
- HOLD = keep current state, no action

CRITICAL DECISION FRAMEWORK:
1. TREND IDENTIFICATION: Use EMA alignment, ADX, and price structure to identify the trend. Trade WITH the trend, not against it.
2. ENTRY TIMING: Wait for pullbacks within a trend. Use RSI, Stochastic, and Bollinger Band position for entry timing.
3. SIGNAL CONFLUENCE: Only enter when 3+ indicators agree. The "indicator consensus" line tells you the count.
4. VOLUME CONFIRMATION: High volume on breakouts confirms the move. Low volume on reversals means the reversal is weak.
5. VWAP: Price above VWAP = bullish bias, below = bearish bias. VWAP acts as dynamic support/resistance.
6. EXIT RULES:
   - Take profits when price reaches the opposite Bollinger Band
   - Take profits when RSI becomes overbought (>70 for longs) or oversold (<30 for shorts)
   - Cut losses when the trade moves against you by more than 1x ATR
   - EXIT immediately if a MACD crossover goes against your position

KEY RULES:
- Only signal BUY or SELL when you have HIGH confidence (strength >= 0.6)
- HOLD is the default when signals are mixed or unclear
- This is DAY TRADING: aim for quick 0.5%-2% moves, not big swings
- When you have a profitable position (+0.5% or more), strongly consider taking profits
- When a position is losing (-0.5% or worse), cut losses immediately, don't hope
- NEVER ignore the CURRENT POSITION section — read it carefully for side, P&L, and exit instructions
- Capital preservation is priority #1 — it's better to miss a trade than to take a bad one
- Return ONLY the JSON object, no other text"""

    def build(self, symbol: str, current_positions: Dict, factor_sections: List[str]) -> Tuple[str, str]:
        """Build system prompt and user prompt.

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        parts = [f"# Trading Analysis Request for {symbol}"]
        parts.append(f"Timestamp: {datetime.now().isoformat()}")

        # Position context - clearly communicate long/short/flat with P&L %
        if symbol in current_positions:
            pos = current_positions[symbol]
            if isinstance(pos, dict):
                side = pos.get('side', 'long').upper()
                entry = pos.get('entry_price', 0)
                size = pos.get('size', 0)
                pnl = pos.get('pnl', 0)
                pnl_str = f"${pnl:.2f}" if isinstance(pnl, (int, float)) else str(pnl)
                # Calculate P&L percentage
                entry_val = entry if isinstance(entry, (int, float)) else 0
                pnl_pct = (pnl / (entry_val * size) * 100) if entry_val > 0 and size > 0 and isinstance(pnl, (int, float)) else 0
                pnl_status = ""
                if pnl_pct >= 1.0:
                    pnl_status = "STRONGLY PROFITABLE — consider taking profits!"
                elif pnl_pct >= 0.3:
                    pnl_status = "Profitable — monitor for exit signals"
                elif pnl_pct <= -1.0:
                    pnl_status = "SIGNIFICANT LOSS — strongly consider cutting losses!"
                elif pnl_pct <= -0.3:
                    pnl_status = "Losing — watch closely, cut if worsening"
                else:
                    pnl_status = "Near breakeven"

                parts.append(f"\n## CURRENT POSITION: {side} {symbol}")
                parts.append(f"  Side: {side}")
                parts.append(f"  Entry price: ${entry_val:.2f}")
                parts.append(f"  Size: {size} shares")
                parts.append(f"  Unrealized P&L: {pnl_str} ({pnl_pct:+.2f}%)")
                parts.append(f"  Status: {pnl_status}")
                if side == 'LONG':
                    parts.append(f"  → To EXIT this long, signal SELL")
                    parts.append(f"  → To HOLD this long, signal HOLD")
                elif side == 'SHORT':
                    parts.append(f"  → To EXIT this short, signal BUY")
                    parts.append(f"  → To HOLD this short, signal HOLD")
            else:
                parts.append(f"\nCurrent position: {pos}")
        else:
            parts.append(f"\n## CURRENT POSITION: FLAT (no position in {symbol})")
            parts.append(f"  → Signal BUY to go long, SELL to go short")
            parts.append(f"  → Signal HOLD to stay flat and wait for a better setup")

        # Add enabled factor sections
        for section in factor_sections:
            if section:
                parts.append(f"\n{section}")

        user_prompt = "\n".join(parts)
        return self.SYSTEM_PROMPT, user_prompt


# =============================================================================
# RESPONSE PARSER
# =============================================================================

class ResponseParser:
    """Parses Claude's JSON response into a StrategySignal."""

    @staticmethod
    def parse(response_text: str, symbol: str, strategy_name: str = "claude_ai") -> Optional[StrategySignal]:
        """Parse Claude's response text into a StrategySignal.

        Returns StrategySignal on success, None on parse failure (caller should HOLD).
        """
        try:
            # Strip markdown code fences
            text = response_text.strip()
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)

            # Try direct JSON parse
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Fallback: extract JSON object via regex
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                else:
                    logger.warning(f"Could not extract JSON from response: {text[:200]}")
                    return None

            # Map signal string to SignalType
            signal_str = data.get('signal', 'HOLD').upper()
            signal_map = {
                'BUY': SignalType.BUY,
                'SELL': SignalType.SELL,
                'HOLD': SignalType.HOLD,
                'CLOSE_LONG': SignalType.CLOSE_LONG,
                'CLOSE_SHORT': SignalType.CLOSE_SHORT,
            }
            signal_type = signal_map.get(signal_str, SignalType.HOLD)

            # Validate and clamp strength
            strength = float(data.get('strength', 0.5))
            strength = max(0.0, min(1.0, strength))

            signal = StrategySignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                strategy_name=strategy_name,
                timestamp=datetime.now(),
                entry_price=data.get('entry_price'),
                stop_loss=data.get('stop_loss'),
                take_profit=data.get('take_profit'),
                metadata={
                    'reasoning': data.get('reasoning', ''),
                    'risk_notes': data.get('risk_notes', ''),
                    'raw_response': data,
                    'source': 'claude_api'
                }
            )
            return signal

        except Exception as e:
            logger.error(f"Failed to parse Claude response: {e}")
            return None


# =============================================================================
# CLAUDE API CLIENT
# =============================================================================

class ClaudeAPIClient:
    """Handles API calls with rate limiting, caching, and circuit breaker."""

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        self.api_key = config.get('api_key') or os.environ.get('ANTHROPIC_API_KEY', '')
        self.model = config.get('model', 'claude-3-5-haiku-latest')
        self.max_tokens = config.get('max_tokens', 500)

        # Rate limiting
        self.max_calls_per_hour = config.get('max_calls_per_hour', 5000)
        self.daily_cost_cap = config.get('daily_cost_cap', 5.0)
        self._call_timestamps: List[float] = []
        self._daily_calls = 0
        self._daily_reset_time = time.time()

        # Cache: key -> (timestamp, signal_data)
        self.cache_ttl = config.get('cache_ttl_seconds', 300)
        self._cache: Dict[str, Tuple[float, dict]] = {}
        self._price_change_threshold = config.get('price_change_threshold', 0.005)  # 0.5%

        # Circuit breaker (lenient for backtesting - retries handle transient errors)
        self._failure_count = 0
        self._circuit_open_until = 0
        self._max_failures = 10
        self._circuit_reset_seconds = 60  # 1 minute cooldown

    @property
    def is_available(self) -> bool:
        """Check if API key is set."""
        return bool(self.api_key)

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()

        # Reset daily counter
        if now - self._daily_reset_time > 86400:
            self._daily_calls = 0
            self._daily_reset_time = now

        # Prune old timestamps
        self._call_timestamps = [t for t in self._call_timestamps if now - t < 3600]

        return len(self._call_timestamps) < self.max_calls_per_hour

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows a call."""
        if self._failure_count >= self._max_failures:
            if time.time() < self._circuit_open_until:
                return False
            # Reset after cooldown
            self._failure_count = 0
        return True

    def _get_cache_key(self, symbol: str, current_price: float) -> str:
        """Generate cache key from symbol and quantized price."""
        # Quantize price to reduce cache misses on small moves
        quantized = round(current_price, 1)
        return f"{symbol}:{quantized}"

    def get_cached(self, symbol: str, current_price: float) -> Optional[dict]:
        """Check cache for a recent result."""
        key = self._get_cache_key(symbol, current_price)
        if key in self._cache:
            ts, data = self._cache[key]
            if time.time() - ts < self.cache_ttl:
                logger.info(f"Cache hit for {symbol}")
                return data
            else:
                del self._cache[key]
        return None

    def set_cached(self, symbol: str, current_price: float, data: dict):
        """Store result in cache."""
        key = self._get_cache_key(symbol, current_price)
        self._cache[key] = (time.time(), data)

    async def call_api(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call Claude API and return response text.

        Returns None if rate limited, circuit broken, or API error.
        """
        if not self.is_available:
            logger.warning("Claude API key not set")
            return None

        if not self._check_circuit_breaker():
            logger.warning("Circuit breaker open - skipping API call")
            return None

        if not self._check_rate_limit():
            logger.warning("Rate limit reached - skipping API call")
            return None

        if aiohttp is None:
            logger.error("aiohttp not installed - cannot call Claude API")
            return None

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        'https://api.anthropic.com/v1/messages',
                        headers={
                            'x-api-key': self.api_key,
                            'content-type': 'application/json',
                            'anthropic-version': '2023-06-01'
                        },
                        json={
                            'model': self.model,
                            'max_tokens': self.max_tokens,
                            'system': system_prompt,
                            'messages': [{'role': 'user', 'content': user_prompt}]
                        },
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as resp:
                        self._call_timestamps.append(time.time())
                        self._daily_calls += 1

                        if resp.status == 200:
                            data = await resp.json()
                            content = data['content'][0]['text']
                            self._failure_count = 0
                            return content
                        elif resp.status == 429:
                            # Rate limited - wait and retry
                            retry_after = float(resp.headers.get('retry-after', 5))
                            logger.warning(f"Rate limited (429), waiting {retry_after}s (attempt {attempt+1}/{max_retries})")
                            await asyncio.sleep(retry_after)
                            continue
                        elif resp.status == 529:
                            # Overloaded - wait and retry
                            logger.warning(f"API overloaded (529), waiting 10s (attempt {attempt+1}/{max_retries})")
                            await asyncio.sleep(10)
                            continue
                        else:
                            error_text = await resp.text()
                            logger.error(f"Claude API error {resp.status}: {error_text[:200]}")
                            self._failure_count += 1
                            if self._failure_count >= self._max_failures:
                                self._circuit_open_until = time.time() + self._circuit_reset_seconds
                            return None

            except Exception as e:
                logger.error(f"Claude API call failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                self._failure_count += 1
                if self._failure_count >= self._max_failures:
                    self._circuit_open_until = time.time() + self._circuit_reset_seconds
                return None
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        return {
            'calls_this_hour': len(self._call_timestamps),
            'max_calls_per_hour': self.max_calls_per_hour,
            'daily_calls': self._daily_calls,
            'cache_size': len(self._cache),
            'circuit_breaker_failures': self._failure_count,
            'circuit_open': self._failure_count >= self._max_failures and time.time() < self._circuit_open_until,
            'api_available': self.is_available,
            'model': self.model,
        }


# =============================================================================
# CLAUDE STRATEGY (BaseStrategy implementation)
# =============================================================================

class ClaudeStrategy(BaseStrategy):
    """Trading strategy that uses Claude AI for decision-making.

    Supports configurable factors and two backtest modes:
    - 'fallback': Delegates to AdaptiveStrategy (no API calls)
    - 'live_api': Calls Claude API per bar (with rate limiting)
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

        # Factor configuration
        factor_config = self.parameters.get('factors', {})
        self.factors_enabled = {
            'price_action': factor_config.get('price_action', True),
            'technical_indicators': factor_config.get('technical_indicators', True),
            'news_sentiment': factor_config.get('news_sentiment', False),
            'market_regime': factor_config.get('market_regime', True),
        }

        # Initialize factor modules
        self.price_action_factor = PriceActionFactor()
        self.technical_factor = TechnicalIndicatorsFactor()
        self.news_factor = NewsSentimentFactor()
        self.regime_factor = MarketRegimeFactor()

        # API client
        api_config = {
            'api_key': self.parameters.get('api_key', ''),
            'model': self.parameters.get('model', 'claude-3-5-haiku-latest'),
            'max_calls_per_hour': self.parameters.get('max_calls_per_hour', 5000),
            'cache_ttl_seconds': self.parameters.get('cache_ttl_seconds', 300),
        }
        self.api_client = ClaudeAPIClient(api_config)

        # Prompt builder and parser
        self.prompt_builder = PromptBuilder()
        self.response_parser = ResponseParser()

        # Backtest mode: 'fallback' or 'live_api'
        self.backtest_mode = self.parameters.get('backtesting_mode', 'fallback')
        self._is_backtesting = False

        # Fallback strategy for backtest mode
        self._fallback_strategy = None

    def set_backtesting(self, enabled: bool, mode: str = 'fallback'):
        """Enable/disable backtesting mode."""
        self._is_backtesting = enabled
        self.backtest_mode = mode
        if enabled:
            # Disable time-based cache during backtesting - each bar is unique
            self.api_client._cache.clear()
            self.api_client.cache_ttl = 0

    def _get_fallback_strategy(self):
        """Lazy-init the fallback strategy."""
        if self._fallback_strategy is None:
            try:
                from strategy_system import AdaptiveStrategy
                fallback_config = StrategyConfig(
                    name='Claude Fallback (Adaptive)',
                    enabled=True,
                    weight=1.0,
                    parameters={}
                )
                self._fallback_strategy = AdaptiveStrategy(fallback_config)
            except Exception as e:
                logger.error(f"Could not create fallback strategy: {e}")
        return self._fallback_strategy

    def analyze(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        """Analyze market data and generate signals (sync interface for BaseStrategy)."""
        if self._is_backtesting and self.backtest_mode == 'fallback':
            return self._backtest_analyze_fallback(market_data, current_positions)
        if self._is_backtesting and self.backtest_mode == 'rules':
            return self._rules_analyze(market_data, current_positions)

        # Live analysis (or live_api backtest mode)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context - use thread to avoid blocking event loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(self._run_async_analyze, market_data, current_positions)
                    return future.result(timeout=45)
            else:
                return loop.run_until_complete(self._live_analyze(market_data, current_positions))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._live_analyze(market_data, current_positions))
            finally:
                loop.close()

    async def analyze_async(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        """Async analyze - use this from async callers (like the backtest runner).

        Avoids blocking the event loop with ThreadPoolExecutor.
        """
        if self._is_backtesting and self.backtest_mode == 'fallback':
            return self._backtest_analyze_fallback(market_data, current_positions)
        if self._is_backtesting and self.backtest_mode == 'rules':
            return self._rules_analyze(market_data, current_positions)

        return await self._live_analyze(market_data, current_positions)

    def _run_async_analyze(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        """Run async analyze in a new event loop (for thread pool)."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._live_analyze(market_data, current_positions))
        finally:
            loop.close()

    async def _live_analyze(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        """Live analysis using Claude API."""
        if market_data is None or len(market_data) < 10:
            return []

        # Determine symbol
        symbol = market_data.attrs.get('symbol', 'UNKNOWN') if hasattr(market_data, 'attrs') else 'UNKNOWN'
        if symbol == 'UNKNOWN' and hasattr(market_data, 'name'):
            symbol = str(market_data.name) if market_data.name else 'UNKNOWN'
        if symbol == 'UNKNOWN':
            # Try to find from index name
            symbol = market_data.index.name if market_data.index.name else 'UNKNOWN'

        current_price = float(market_data['close'].iloc[-1])

        # Check cache
        cached = self.api_client.get_cached(symbol, current_price)
        if cached:
            signal = self.response_parser.parse(json.dumps(cached), symbol, self.name)
            if signal:
                signal.metadata['from_cache'] = True
                return [signal]

        # Assemble factor sections
        factor_sections = []
        if self.factors_enabled.get('price_action'):
            factor_sections.append(self.price_action_factor.format(market_data, symbol))
        if self.factors_enabled.get('technical_indicators'):
            factor_sections.append(self.technical_factor.format(market_data, symbol))
        if self.factors_enabled.get('news_sentiment'):
            section = await self.news_factor.format_async(symbol)
            factor_sections.append(section)
        if self.factors_enabled.get('market_regime'):
            factor_sections.append(self.regime_factor.format(market_data, symbol))

        # Filter empty sections
        factor_sections = [s for s in factor_sections if s]

        if not factor_sections:
            return []

        # Build prompt
        system_prompt, user_prompt = self.prompt_builder.build(symbol, current_positions, factor_sections)

        # Call API
        response_text = await self.api_client.call_api(system_prompt, user_prompt)
        if response_text is None:
            # API unavailable - return HOLD
            logger.info("Claude API unavailable, returning HOLD")
            return []

        # Parse response
        signal = self.response_parser.parse(response_text, symbol, self.name)
        if signal is None:
            return []

        # Cache the raw response data
        self.api_client.set_cached(symbol, current_price, signal.metadata.get('raw_response', {}))

        # Store the prompt in metadata for debugging
        signal.metadata['prompt_sent'] = user_prompt
        signal.metadata['factors_used'] = [k for k, v in self.factors_enabled.items() if v]

        return [signal]

    def _backtest_analyze_fallback(self, market_data: pd.DataFrame, current_positions: Dict) -> List[StrategySignal]:
        """Backtest using fallback (Adaptive) strategy."""
        fallback = self._get_fallback_strategy()
        if fallback is None:
            return []

        signals = fallback.analyze(market_data, current_positions)
        # Re-tag signals as coming from claude_ai strategy
        for s in signals:
            s.strategy_name = self.name
            s.metadata['backtest_mode'] = 'fallback'
            s.metadata['source'] = 'adaptive_strategy_fallback'
        return signals

    def _rules_analyze(self, market_data: pd.DataFrame, current_positions: Dict, params: Dict = None) -> List[StrategySignal]:
        """V3 Setup-based trading engine — trades like an experienced human.

        Context → Setup → Trigger → Confirmation framework:
        1. Assess trend (triple EMA + ADX + slope)
        2. Find key S/R levels (swing highs/lows)
        3. Wait for setup at logical price (pullback, breakout, bounce)
        4. Require trigger confirmation (candle, momentum turn, volume)
        5. Manage with ATR-based trailing stops
        """
        if market_data is None or len(market_data) < 50:
            return []

        # Lazy init for trade state (persists across bars within one backtest run)
        if not hasattr(self, '_trade_state'):
            self._trade_state = {}

        p = params or self.parameters.get('rules_params', {})

        # --- Parameters ---
        atr_stop_mult = p.get('atr_stop_mult', 2.0)       # Stop = entry ± ATR * this
        atr_target_mult = p.get('atr_target_mult', 3.0)    # Target = entry ± ATR * this
        trail_after_r = p.get('trail_after_r', 1.5)        # Start trailing after this many R
        trail_atr_mult = p.get('trail_atr_mult', 2.5)      # Trail at highest - ATR * this
        breakeven_r = p.get('breakeven_r', 1.0)             # Move stop to breakeven at this R
        pullback_zone_atr = p.get('pullback_zone_atr', 1.0) # How close to EMA = pullback (ATR units)
        min_trend_adx = p.get('min_trend_adx', 20)          # Min ADX for confirmed trend
        rsi_oversold = p.get('rsi_oversold', 30)
        rsi_overbought = p.get('rsi_overbought', 70)
        min_vol_ratio = p.get('min_vol_ratio', 0.8)
        min_triggers = p.get('min_triggers', 2)             # Min trigger confirmations for entry
        # Backward compat: fixed % overrides ATR-based if set
        tp_pct_override = p.get('take_profit_pct', None)
        sl_pct_override = p.get('stop_loss_pct', None)

        symbol = market_data.index.name if market_data.index.name else 'UNKNOWN'
        close = market_data['close']
        high = market_data['high']
        low = market_data['low']
        open_price = market_data['open'] if 'open' in market_data.columns else close
        volume = market_data['volume'] if 'volume' in market_data.columns else None
        current_price = float(close.iloc[-1])
        current_high = float(high.iloc[-1])
        current_low = float(low.iloc[-1])
        current_open = float(open_price.iloc[-1])

        try:
            import ta as ta_lib
        except ImportError:
            return []

        try:
            # ============================================================
            # INDICATOR EXTRACTION (precomputed or on-the-fly)
            # ============================================================
            precomputed = '_rsi' in market_data.columns

            if precomputed:
                row = market_data.iloc[-1]
                prev = market_data.iloc[-2] if len(market_data) >= 2 else row

                def _v(r, col, default=0.0):
                    v = r.get(col, default)
                    return float(v) if pd.notna(v) else default

                rsi = _v(row, '_rsi', 50.0)
                rsi_prev = _v(prev, '_rsi', rsi)
                rsi_series = market_data['_rsi']

                stoch_k = _v(row, '_stoch_k', 50.0)
                stoch_d = _v(row, '_stoch_d', 50.0)
                stoch_k_prev = _v(prev, '_stoch_k', stoch_k)
                stoch_d_prev = _v(prev, '_stoch_d', stoch_d)

                macd_val = _v(row, '_macd')
                macd_signal = _v(row, '_macd_signal')
                macd_hist = _v(row, '_macd_hist')
                macd_hist_prev = _v(prev, '_macd_hist', macd_hist)
                macd_val_prev = _v(prev, '_macd')
                macd_sig_prev = _v(prev, '_macd_signal')

                bb_pct = _v(row, '_bb_pct', 0.5)

                adx_val = _v(row, '_adx')
                di_plus = _v(row, '_di_plus')
                di_minus = _v(row, '_di_minus')

                atr = _v(row, '_atr')

                ema9 = _v(row, '_ema9', current_price)
                ema21 = _v(row, '_ema21', current_price)
                ema50 = _v(row, '_ema50', ema21)

                ema21_slope = _v(row, '_ema21_slope')

                vwap_val = _v(row, '_vwap') if pd.notna(row.get('_vwap')) else None
                vol_ratio = _v(row, '_vol_ratio', 1.0)

            else:
                # Slow path: calculate from scratch
                rsi_ind = ta_lib.momentum.RSIIndicator(close, window=14)
                rsi_series = rsi_ind.rsi()
                rsi = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else 50.0
                rsi_prev = float(rsi_series.iloc[-2]) if len(rsi_series) >= 2 and pd.notna(rsi_series.iloc[-2]) else rsi

                stoch = ta_lib.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
                stoch_k = float(stoch.stoch().iloc[-1]) if pd.notna(stoch.stoch().iloc[-1]) else 50.0
                stoch_d = float(stoch.stoch_signal().iloc[-1]) if pd.notna(stoch.stoch_signal().iloc[-1]) else 50.0
                stoch_k_prev = float(stoch.stoch().iloc[-2]) if len(stoch.stoch()) >= 2 else stoch_k
                stoch_d_prev = float(stoch.stoch_signal().iloc[-2]) if len(stoch.stoch_signal()) >= 2 else stoch_d

                macd_ind = ta_lib.trend.MACD(close)
                macd_val = float(macd_ind.macd().iloc[-1]) if pd.notna(macd_ind.macd().iloc[-1]) else 0.0
                macd_signal = float(macd_ind.macd_signal().iloc[-1]) if pd.notna(macd_ind.macd_signal().iloc[-1]) else 0.0
                macd_hist = float(macd_ind.macd_diff().iloc[-1]) if pd.notna(macd_ind.macd_diff().iloc[-1]) else 0.0
                macd_hist_prev = float(macd_ind.macd_diff().iloc[-2]) if len(macd_ind.macd_diff()) >= 2 else macd_hist
                macd_val_prev = float(macd_ind.macd().iloc[-2]) if len(macd_ind.macd()) >= 2 else macd_val
                macd_sig_prev = float(macd_ind.macd_signal().iloc[-2]) if len(macd_ind.macd_signal()) >= 2 else macd_signal

                bb = ta_lib.volatility.BollingerBands(close, window=20, window_dev=2)
                bb_pct = float(bb.bollinger_pband().iloc[-1]) if pd.notna(bb.bollinger_pband().iloc[-1]) else 0.5

                adx_ind = ta_lib.trend.ADXIndicator(high, low, close, window=14)
                adx_val = float(adx_ind.adx().iloc[-1]) if pd.notna(adx_ind.adx().iloc[-1]) else 0.0
                di_plus = float(adx_ind.adx_pos().iloc[-1]) if pd.notna(adx_ind.adx_pos().iloc[-1]) else 0.0
                di_minus = float(adx_ind.adx_neg().iloc[-1]) if pd.notna(adx_ind.adx_neg().iloc[-1]) else 0.0

                atr = float(ta_lib.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1])

                ema9_s = ta_lib.trend.EMAIndicator(close, window=9).ema_indicator()
                ema21_s = ta_lib.trend.EMAIndicator(close, window=21).ema_indicator()
                ema50_s = ta_lib.trend.EMAIndicator(close, window=50).ema_indicator()
                ema9 = float(ema9_s.iloc[-1])
                ema21 = float(ema21_s.iloc[-1])
                ema50 = float(ema50_s.iloc[-1]) if pd.notna(ema50_s.iloc[-1]) else ema21

                ema21_slope = 0.0
                if len(ema21_s) >= 6:
                    ema21_slope = (float(ema21_s.iloc[-1]) - float(ema21_s.iloc[-5])) / float(ema21_s.iloc[-5]) * 100

                vwap_val = None
                if volume is not None and len(market_data) >= 5:
                    typical = (high + low + close) / 3
                    cumvol = volume.cumsum()
                    if cumvol.iloc[-1] > 0:
                        vwap_val = float((typical * volume).cumsum().iloc[-1] / cumvol.iloc[-1])

                vol_ratio = 1.0
                if volume is not None and len(volume) >= 20:
                    avg_vol = volume.tail(20).mean()
                    vol_ratio = float(volume.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0

            # Ensure ATR is valid
            if atr <= 0 or pd.isna(atr):
                atr = current_price * 0.01

            # ============================================================
            # PHASE 1: MARKET CONTEXT
            # ============================================================

            # --- Trend (triple EMA alignment + slope + ADX) ---
            slope_bull = ema21_slope > 0.02
            slope_bear = ema21_slope < -0.02
            has_adx_trend = adx_val > min_trend_adx

            if ema9 > ema21 > ema50 and slope_bull:
                trend = 'STRONG_BULL'
                trend_strength = 3 if (has_adx_trend and di_plus > di_minus) else 2
            elif ema9 > ema21 and slope_bull:
                trend = 'BULL'
                trend_strength = 2 if (has_adx_trend and di_plus > di_minus) else 1
            elif ema9 < ema21 < ema50 and slope_bear:
                trend = 'STRONG_BEAR'
                trend_strength = 3 if (has_adx_trend and di_minus > di_plus) else 2
            elif ema9 < ema21 and slope_bear:
                trend = 'BEAR'
                trend_strength = 2 if (has_adx_trend and di_minus > di_plus) else 1
            else:
                trend = 'CHOPPY'
                trend_strength = 0

            # --- Support / Resistance from swing highs/lows ---
            sr_lookback = min(50, len(close) - 1)
            sw = 5  # Swing confirmed when bar is extreme in ±5-bar window

            swing_highs = []
            swing_lows = []
            for j in range(sw, sr_lookback - sw):
                idx = len(close) - sr_lookback + j
                if idx < sw or idx >= len(high) - sw:
                    continue
                h_val = float(high.iloc[idx])
                l_val = float(low.iloc[idx])
                is_swing_high = all(h_val >= float(high.iloc[idx + k]) for k in range(-sw, sw + 1) if k != 0)
                is_swing_low = all(l_val <= float(low.iloc[idx + k]) for k in range(-sw, sw + 1) if k != 0)
                if is_swing_high:
                    swing_highs.append(h_val)
                if is_swing_low:
                    swing_lows.append(l_val)

            # Nearest resistance (closest swing high above price)
            res_candidates = sorted([h for h in swing_highs if h > current_price * 1.001])
            resistance = res_candidates[0] if res_candidates else None

            # Nearest support (closest swing low below price)
            sup_candidates = sorted([l for l in swing_lows if l < current_price * 0.999], reverse=True)
            support = sup_candidates[0] if sup_candidates else None

            # Distance to S/R in ATR units
            dist_to_res = (resistance - current_price) / atr if resistance else float('inf')
            dist_to_sup = (current_price - support) / atr if support else float('inf')

            # --- Volume context ---
            vol_ok = vol_ratio >= min_vol_ratio
            vol_surge = vol_ratio > 1.5

            # --- Candle analysis ---
            body = abs(current_price - current_open)
            upper_wick = current_high - max(current_price, current_open)
            lower_wick = min(current_price, current_open) - current_low

            candle_bullish = current_price > current_open
            candle_bearish = current_price < current_open
            hammer = lower_wick > body * 2 and upper_wick < body * 0.5 and body > 0
            shooting_star = upper_wick > body * 2 and lower_wick < body * 0.5 and body > 0
            strong_bull_bar = candle_bullish and body > atr * 0.7
            strong_bear_bar = candle_bearish and body > atr * 0.7

            # --- Momentum signals ---
            macd_bull_cross = macd_val > macd_signal and macd_val_prev <= macd_sig_prev
            macd_bear_cross = macd_val < macd_signal and macd_val_prev >= macd_sig_prev
            macd_hist_accel_up = macd_hist > macd_hist_prev and macd_hist < 0
            macd_hist_accel_down = macd_hist < macd_hist_prev and macd_hist > 0

            stoch_bull_cross = stoch_k > stoch_d and stoch_k_prev <= stoch_d_prev
            stoch_bear_cross = stoch_k < stoch_d and stoch_k_prev >= stoch_d_prev

            rsi_turning_up = rsi > rsi_prev and rsi_prev < 45
            rsi_turning_down = rsi < rsi_prev and rsi_prev > 55

            # Divergence (8-bar lookback)
            bull_divergence = False
            bear_divergence = False
            if len(close) >= 10 and len(rsi_series) >= 10:
                p8 = float(rsi_series.iloc[-8]) if pd.notna(rsi_series.iloc[-8]) else rsi
                if float(close.iloc[-1]) < float(close.iloc[-8]) and rsi > p8:
                    bull_divergence = True
                if float(close.iloc[-1]) > float(close.iloc[-8]) and rsi < p8:
                    bear_divergence = True

            # --- Pullback detection ---
            dist_to_ema21 = abs(current_price - ema21) / atr
            near_ema21 = dist_to_ema21 < pullback_zone_atr
            price_above_ema21 = current_price > ema21
            price_below_ema21 = current_price < ema21

            # ============================================================
            # PHASE 2: POSITION MANAGEMENT (if in a trade)
            # ============================================================
            pos = current_positions.get(symbol, {})
            pos_side = pos.get('side', None) if isinstance(pos, dict) else None
            pos_entry = pos.get('entry_price', 0) if isinstance(pos, dict) else 0
            pos_size = pos.get('size', 0) if isinstance(pos, dict) else 0

            state = self._trade_state.get(symbol, {})

            signal_type = SignalType.HOLD
            strength = 0.5
            reasons = []
            risk_notes = ""
            stop_loss_price = None
            take_profit_price = None

            if pos_side in ('long', 'short'):
                # --- Initialize or update trade tracking state ---
                if state.get('entry_price') != pos_entry or not state:
                    r_value = atr * atr_stop_mult
                    if sl_pct_override:
                        r_value = pos_entry * sl_pct_override / 100
                    state = {
                        'entry_price': pos_entry,
                        'highest_high': current_high,
                        'lowest_low': current_low,
                        'r_value': r_value,
                        'bars_in_trade': 0,
                    }
                    if pos_side == 'long':
                        state['trailing_stop'] = pos_entry - r_value
                    else:
                        state['trailing_stop'] = pos_entry + r_value

                state['bars_in_trade'] = state.get('bars_in_trade', 0) + 1
                state['highest_high'] = max(state.get('highest_high', current_high), current_high)
                state['lowest_low'] = min(state.get('lowest_low', current_low), current_low)

                r_value = state['r_value']
                if r_value <= 0:
                    r_value = atr * atr_stop_mult

                # Current R-multiple of profit
                if pos_side == 'long':
                    current_r = (current_price - pos_entry) / r_value if r_value > 0 else 0
                else:
                    current_r = (pos_entry - current_price) / r_value if r_value > 0 else 0

                # --- Update trailing stop ---
                if pos_side == 'long':
                    if current_r >= breakeven_r:
                        be = pos_entry + atr * 0.1
                        state['trailing_stop'] = max(state['trailing_stop'], be)
                    if current_r >= trail_after_r:
                        trail = state['highest_high'] - atr * trail_atr_mult
                        state['trailing_stop'] = max(state['trailing_stop'], trail)
                else:
                    if current_r >= breakeven_r:
                        be = pos_entry - atr * 0.1
                        state['trailing_stop'] = min(state['trailing_stop'], be)
                    if current_r >= trail_after_r:
                        trail = state['lowest_low'] + atr * trail_atr_mult
                        state['trailing_stop'] = min(state['trailing_stop'], trail)

                self._trade_state[symbol] = state

                # --- Check exit conditions ---
                exit_reasons = []
                is_stop = False

                if pos_side == 'long':
                    if current_low <= state['trailing_stop']:
                        if current_r < 0:
                            exit_reasons.append(f"Stop loss at {state['trailing_stop']:.2f}")
                            is_stop = True
                        elif current_r < trail_after_r:
                            exit_reasons.append(f"Breakeven stop at {state['trailing_stop']:.2f}")
                            is_stop = True
                        else:
                            exit_reasons.append(f"Trailing stop at {state['trailing_stop']:.2f} (+{current_r:.1f}R)")

                    if tp_pct_override:
                        if ((current_price - pos_entry) / pos_entry * 100) >= tp_pct_override:
                            exit_reasons.append(f"Take profit {tp_pct_override}%")
                    elif current_r >= atr_target_mult:
                        exit_reasons.append(f"ATR target ({current_r:.1f}R)")

                    if resistance and current_high >= resistance * 0.998 and current_price < resistance * 0.995 and current_r > 0.5:
                        exit_reasons.append(f"Rejected at resistance {resistance:.2f}")

                    if current_r > 1 and bear_divergence and rsi > 65:
                        exit_reasons.append("Bearish divergence + elevated RSI")

                    if current_r > 0.5 and ema9 < ema21 and trend in ('BEAR', 'STRONG_BEAR', 'CHOPPY'):
                        exit_reasons.append("Trend turned bearish")

                else:  # short
                    if current_high >= state['trailing_stop']:
                        if current_r < 0:
                            exit_reasons.append(f"Stop loss at {state['trailing_stop']:.2f}")
                            is_stop = True
                        elif current_r < trail_after_r:
                            exit_reasons.append(f"Breakeven stop at {state['trailing_stop']:.2f}")
                            is_stop = True
                        else:
                            exit_reasons.append(f"Trailing stop at {state['trailing_stop']:.2f} (+{current_r:.1f}R)")

                    if tp_pct_override:
                        if ((pos_entry - current_price) / pos_entry * 100) >= tp_pct_override:
                            exit_reasons.append(f"Take profit {tp_pct_override}%")
                    elif current_r >= atr_target_mult:
                        exit_reasons.append(f"ATR target ({current_r:.1f}R)")

                    if support and current_low <= support * 1.002 and current_price > support * 1.005 and current_r > 0.5:
                        exit_reasons.append(f"Rejected at support {support:.2f}")

                    if current_r > 1 and bull_divergence and rsi < 35:
                        exit_reasons.append("Bullish divergence + low RSI")

                    if current_r > 0.5 and ema9 > ema21 and trend in ('BULL', 'STRONG_BULL', 'CHOPPY'):
                        exit_reasons.append("Trend turned bullish")

                # --- Exit decision ---
                should_exit = is_stop or len(exit_reasons) >= 1
                if should_exit and exit_reasons:
                    signal_type = SignalType.SELL if pos_side == 'long' else SignalType.BUY
                    strength = min(0.6 + len(exit_reasons) * 0.15, 0.95)
                    pnl_pct = ((current_price - pos_entry) / pos_entry * 100) if pos_side == 'long' else ((pos_entry - current_price) / pos_entry * 100)
                    risk_notes = f"EXIT {pos_side.upper()} ({pnl_pct:+.1f}%, {current_r:.1f}R): {'; '.join(exit_reasons)}"
                    reasons = exit_reasons
                else:
                    pnl_pct = ((current_price - pos_entry) / pos_entry * 100) if pos_side == 'long' else ((pos_entry - current_price) / pos_entry * 100)
                    risk_notes = f"Holding {pos_side} ({pnl_pct:+.1f}%, {current_r:.1f}R, stop={state['trailing_stop']:.2f}, bars={state['bars_in_trade']})"
                    reasons = [risk_notes]

            else:
                # ============================================================
                # PHASE 3: SETUP DETECTION (no position)
                # ============================================================
                if symbol in self._trade_state:
                    del self._trade_state[symbol]

                # --- Long setups ---
                long_setup = None
                long_score = 0
                long_reasons = []

                # A. Pullback to EMA in uptrend (most reliable setup)
                if trend in ('STRONG_BULL', 'BULL') and near_ema21 and price_above_ema21:
                    if candle_bullish or hammer:
                        long_setup = 'PULLBACK'
                        long_score = 1
                        long_reasons = [f"Pullback to EMA21 in {trend}"]
                        if rsi_turning_up:
                            long_score += 1; long_reasons.append("RSI turning up")
                        if stoch_bull_cross or (stoch_k < 50 and stoch_k > stoch_d):
                            long_score += 1; long_reasons.append("Stochastic bullish")
                        if macd_hist_accel_up or macd_bull_cross:
                            long_score += 1; long_reasons.append("MACD momentum turning")
                        if vol_surge:
                            long_score += 1; long_reasons.append(f"Volume surge {vol_ratio:.1f}x")
                        if support and dist_to_sup < 2:
                            long_score += 1; long_reasons.append(f"Near support {support:.2f}")

                # B. Breakout above resistance with volume
                if resistance and current_price > resistance and vol_surge and strong_bull_bar:
                    if not long_setup or long_score < 2:
                        long_setup = 'BREAKOUT'
                        long_score = 1
                        long_reasons = [f"Breakout above {resistance:.2f}"]
                        if trend in ('STRONG_BULL', 'BULL'):
                            long_score += 1; long_reasons.append("With-trend breakout")
                        if macd_val > macd_signal:
                            long_score += 1; long_reasons.append("MACD confirms")
                        if adx_val > 25:
                            long_score += 1; long_reasons.append(f"Strong trend ADX={adx_val:.0f}")

                # C. Oversold bounce at support
                if support and dist_to_sup < 1.5 and rsi < rsi_oversold:
                    if (candle_bullish or hammer) and (not long_setup or long_score < 2):
                        long_setup = 'BOUNCE'
                        long_score = 1
                        long_reasons = [f"Oversold bounce near support {support:.2f}"]
                        if stoch_bull_cross:
                            long_score += 1; long_reasons.append("Stochastic bullish cross")
                        if bull_divergence:
                            long_score += 2; long_reasons.append("BULLISH DIVERGENCE")
                        if vol_surge:
                            long_score += 1; long_reasons.append(f"Volume spike {vol_ratio:.1f}x")

                # D. MACD crossover in confirmed trend
                if macd_bull_cross and trend in ('STRONG_BULL', 'BULL') and candle_bullish:
                    if not long_setup or long_score < 2:
                        long_setup = 'MOMENTUM'
                        long_score = 1
                        long_reasons = ["MACD bullish cross in uptrend"]
                        if vol_ok:
                            long_score += 1; long_reasons.append("Volume confirms")
                        if rsi < 60:
                            long_score += 1; long_reasons.append(f"RSI room ({rsi:.0f})")

                # --- Short setups ---
                short_setup = None
                short_score = 0
                short_reasons = []

                # A. Pullback to EMA in downtrend
                if trend in ('STRONG_BEAR', 'BEAR') and near_ema21 and price_below_ema21:
                    if candle_bearish or shooting_star:
                        short_setup = 'PULLBACK'
                        short_score = 1
                        short_reasons = [f"Pullback to EMA21 in {trend}"]
                        if rsi_turning_down:
                            short_score += 1; short_reasons.append("RSI turning down")
                        if stoch_bear_cross or (stoch_k > 50 and stoch_k < stoch_d):
                            short_score += 1; short_reasons.append("Stochastic bearish")
                        if macd_hist_accel_down or macd_bear_cross:
                            short_score += 1; short_reasons.append("MACD momentum turning")
                        if vol_surge:
                            short_score += 1; short_reasons.append(f"Volume surge {vol_ratio:.1f}x")
                        if resistance and dist_to_res < 2:
                            short_score += 1; short_reasons.append(f"Near resistance {resistance:.2f}")

                # B. Breakdown below support with volume
                if support and current_price < support and vol_surge and strong_bear_bar:
                    if not short_setup or short_score < 2:
                        short_setup = 'BREAKDOWN'
                        short_score = 1
                        short_reasons = [f"Breakdown below {support:.2f}"]
                        if trend in ('STRONG_BEAR', 'BEAR'):
                            short_score += 1; short_reasons.append("With-trend breakdown")
                        if macd_val < macd_signal:
                            short_score += 1; short_reasons.append("MACD confirms")
                        if adx_val > 25:
                            short_score += 1; short_reasons.append(f"Strong trend ADX={adx_val:.0f}")

                # C. Overbought rejection at resistance
                if resistance and dist_to_res < 1.5 and rsi > rsi_overbought:
                    if (candle_bearish or shooting_star) and (not short_setup or short_score < 2):
                        short_setup = 'REJECTION'
                        short_score = 1
                        short_reasons = [f"Overbought rejection near resistance {resistance:.2f}"]
                        if stoch_bear_cross:
                            short_score += 1; short_reasons.append("Stochastic bearish cross")
                        if bear_divergence:
                            short_score += 2; short_reasons.append("BEARISH DIVERGENCE")
                        if vol_surge:
                            short_score += 1; short_reasons.append(f"Volume spike {vol_ratio:.1f}x")

                # D. MACD crossover in confirmed trend
                if macd_bear_cross and trend in ('STRONG_BEAR', 'BEAR') and candle_bearish:
                    if not short_setup or short_score < 2:
                        short_setup = 'MOMENTUM'
                        short_score = 1
                        short_reasons = ["MACD bearish cross in downtrend"]
                        if vol_ok:
                            short_score += 1; short_reasons.append("Volume confirms")
                        if rsi > 40:
                            short_score += 1; short_reasons.append(f"RSI room ({rsi:.0f})")

                # --- Entry decision ---
                long_blocked = dist_to_res < 0.5 and long_setup != 'BREAKOUT'
                short_blocked = dist_to_sup < 0.5 and short_setup != 'BREAKDOWN'

                if long_setup and long_score >= min_triggers and not long_blocked and vol_ok:
                    signal_type = SignalType.BUY
                    strength = min(0.5 + long_score * 0.1, 0.95)
                    stop_loss_price = current_price - atr * atr_stop_mult
                    take_profit_price = current_price + atr * atr_target_mult
                    if resistance and resistance < take_profit_price:
                        take_profit_price = resistance - atr * 0.2
                    risk_notes = (f"LONG {long_setup}: {long_score} triggers | trend={trend} | "
                                  f"stop={stop_loss_price:.2f} target={take_profit_price:.2f} | "
                                  f"R:R={atr_target_mult / atr_stop_mult:.1f}")
                    reasons = long_reasons

                elif short_setup and short_score >= min_triggers and not short_blocked and vol_ok:
                    signal_type = SignalType.SELL
                    strength = min(0.5 + short_score * 0.1, 0.95)
                    stop_loss_price = current_price + atr * atr_stop_mult
                    take_profit_price = current_price - atr * atr_target_mult
                    if support and support > take_profit_price:
                        take_profit_price = support + atr * 0.2
                    risk_notes = (f"SHORT {short_setup}: {short_score} triggers | trend={trend} | "
                                  f"stop={stop_loss_price:.2f} target={take_profit_price:.2f} | "
                                  f"R:R={atr_target_mult / atr_stop_mult:.1f}")
                    reasons = short_reasons

                else:
                    signal_type = SignalType.HOLD
                    strength = 0.3
                    ctx = f"trend={trend}({trend_strength})"
                    if long_setup:
                        ctx += f" | L:{long_setup}({long_score}/{min_triggers})"
                        if long_blocked:
                            ctx += "[near R]"
                    if short_setup:
                        ctx += f" | S:{short_setup}({short_score}/{min_triggers})"
                        if short_blocked:
                            ctx += "[near S]"
                    sr = ""
                    if support:
                        sr += f" S={support:.2f}({dist_to_sup:.1f}A)"
                    if resistance:
                        sr += f" R={resistance:.2f}({dist_to_res:.1f}A)"
                    risk_notes = f"WAIT: {ctx}{sr}"
                    reasons = [risk_notes]

            # ============================================================
            # BUILD SIGNAL
            # ============================================================
            reasoning = "; ".join(reasons[:6])

            signal = StrategySignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                strategy_name=self.name,
                timestamp=datetime.now(),
                entry_price=current_price if signal_type in (SignalType.BUY, SignalType.SELL) else None,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                metadata={
                    'reasoning': reasoning,
                    'risk_notes': risk_notes,
                    'source': 'rules_engine_v3',
                    'backtest_mode': 'rules',
                    'indicators': {
                        'rsi': round(rsi, 1), 'stoch_k': round(stoch_k, 1),
                        'macd_hist': round(macd_hist, 3), 'bb_pct': round(bb_pct * 100, 0),
                        'adx': round(adx_val, 1), 'di_plus': round(di_plus, 1),
                        'di_minus': round(di_minus, 1), 'atr': round(atr, 3),
                        'ema9': round(ema9, 2), 'ema21': round(ema21, 2),
                        'ema50': round(ema50, 2),
                        'vwap': round(vwap_val, 2) if vwap_val else None,
                        'vol_ratio': round(vol_ratio, 1),
                        'trend': trend, 'trend_strength': trend_strength,
                        'ema21_slope': round(ema21_slope, 3),
                        'candle_dir': 'bullish' if candle_bullish else 'bearish',
                        'support': round(support, 2) if support else None,
                        'resistance': round(resistance, 2) if resistance else None,
                    },
                    'raw_response': {
                        'signal': signal_type.value, 'strength': strength,
                        'reasoning': reasoning, 'risk_notes': risk_notes,
                    },
                    'factors_used': [k for k, v in self.factors_enabled.items() if v],
                }
            )
            return [signal]

        except Exception as e:
            logger.error(f"Rules engine error: {e}")
            return []

    def get_required_indicators(self) -> List[str]:
        """Return union of indicators needed by enabled factors."""
        indicators = []
        if self.factors_enabled.get('price_action'):
            indicators.extend(['open', 'high', 'low', 'close', 'volume'])
        if self.factors_enabled.get('technical_indicators'):
            indicators.extend(['rsi', 'macd', 'bollinger_bands', 'adx', 'atr', 'ema_9', 'ema_21', 'ema_50'])
        if self.factors_enabled.get('market_regime'):
            indicators.extend(['adx', 'atr', 'sma_20', 'sma_50'])
        return list(set(indicators))

    def get_required_lookback(self) -> int:
        """Return required lookback period."""
        return 50

    def get_factor_status(self) -> Dict[str, bool]:
        """Return which factors are currently enabled."""
        return dict(self.factors_enabled)

    def set_factors(self, factors: Dict[str, bool]):
        """Update which factors are enabled."""
        for key, value in factors.items():
            if key in self.factors_enabled:
                self.factors_enabled[key] = value

    def get_api_stats(self) -> Dict[str, Any]:
        """Get API usage stats."""
        return self.api_client.get_stats()
