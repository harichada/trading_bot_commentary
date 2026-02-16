#!/usr/bin/env python3
"""
LLM Prediction System for Trading Bot Backtesting.

Four production classes:
  - DataPrep: Builds rich market context from PrecomputedArrays
  - LLMPromptBuilder: Constructs structured prompts for LLM trading analysis
  - Predictor: Calls Ollama/Anthropic API and parses JSON predictions
  - BacktestTracker: Tracks prediction accuracy with per-pattern/structure stats
"""

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

from strategy_system import StrategySignal, SignalType

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependency with claude_backtest_app
_backtest_imports = {}


def _get_backtest_imports():
    """Lazy-import functions from claude_backtest_app to avoid circular imports."""
    if not _backtest_imports:
        from claude_backtest_app import (
            _snapshot_active_patterns,
            _market_structure_bias,
        )
        _backtest_imports['_snapshot_active_patterns'] = _snapshot_active_patterns
        _backtest_imports['_market_structure_bias'] = _market_structure_bias
    return _backtest_imports


def _nan(v):
    """Replace NaN with 0."""
    try:
        return 0.0 if (v is None or np.isnan(v)) else float(v)
    except (TypeError, ValueError):
        return 0.0


# =============================================================================
# DataPrep — Rich market context from PrecomputedArrays
# =============================================================================

class DataPrep:
    """Extracts rich market context from PrecomputedArrays at a specific bar.

    Reuses all 75+ pre-computed numpy arrays (RSI, MACD, ADX, ATR, EMAs,
    S/R, patterns, divergences, etc.) — zero recomputation. Only computes
    a few additional values: 52-week range, recent candle descriptions,
    MACD cross detection.
    """

    def __init__(self, df: pd.DataFrame, arrays, bar_index: int,
                 symbol: str = 'UNKNOWN', interval: str = '5m',
                 market_context=None):
        """
        Args:
            df: Full OHLCV DataFrame (same one used to build arrays).
            arrays: PrecomputedArrays from precompute_arrays().
            bar_index: Current bar index (0-based into arrays).
            symbol: Ticker symbol.
            interval: Bar interval ('1m', '5m', '1d', etc.).
            market_context: Optional MarketContext (SPY/VIX).
        """
        self.df = df
        self.arrays = arrays
        self.i = bar_index
        self.symbol = symbol
        self.interval = interval
        self.mkt = market_context

    def build(self) -> dict:
        """Extract all data and return a rich context dict."""
        a = self.arrays
        i = self.i

        # Current bar
        price = _nan(a.close[i])
        bar_open = _nan(a.open[i])
        bar_high = _nan(a.high[i])
        bar_low = _nan(a.low[i])
        volume = _nan(a.volume[i])

        # 52-week range (use full available history, up to 252 bars for daily)
        lookback_52w = min(i + 1, 252)
        start_idx = max(0, i - lookback_52w + 1)
        high_52w = float(np.nanmax(a.high[start_idx:i + 1]))
        low_52w = float(np.nanmin(a.low[start_idx:i + 1]))
        dist_from_high = ((price - high_52w) / high_52w * 100) if high_52w > 0 else 0
        dist_from_low = ((price - low_52w) / low_52w * 100) if low_52w > 0 else 0

        # Trend
        ema9 = _nan(a.ema9[i])
        ema21 = _nan(a.ema21[i])
        ema50 = _nan(a.ema50[i])
        ema21_slope = _nan(a.ema21_slope[i])
        adx = _nan(a.adx[i])
        di_plus = _nan(a.di_plus[i])
        di_minus = _nan(a.di_minus[i])

        # Trend classification
        if ema9 > ema21 > ema50 and ema21_slope > 0.02:
            trend_dir = 'bullish'
            trend_str = 'strong' if adx > 25 else 'moderate'
        elif ema9 > ema21 and ema21_slope > 0:
            trend_dir = 'bullish'
            trend_str = 'weak'
        elif ema9 < ema21 < ema50 and ema21_slope < -0.02:
            trend_dir = 'bearish'
            trend_str = 'strong' if adx > 25 else 'moderate'
        elif ema9 < ema21 and ema21_slope < 0:
            trend_dir = 'bearish'
            trend_str = 'weak'
        else:
            trend_dir = 'neutral'
            trend_str = 'none'

        # EMA alignment
        if ema9 > ema21 > ema50:
            ema_alignment = 'bullish_stacked'
        elif ema9 < ema21 < ema50:
            ema_alignment = 'bearish_stacked'
        else:
            ema_alignment = 'mixed'

        # S/R
        nearest_r = _nan(a.nearest_r[i])
        nearest_s = _nan(a.nearest_s[i])
        atr = _nan(a.atr[i])
        if atr <= 0:
            atr = price * 0.01
        dist_r_atr = ((nearest_r - price) / atr) if nearest_r > 0 and atr > 0 else 0
        dist_s_atr = ((price - nearest_s) / atr) if nearest_s > 0 and atr > 0 else 0

        # Momentum
        rsi = _nan(a.rsi[i])
        stoch_k = _nan(a.stoch_k[i])
        stoch_d = _nan(a.stoch_d[i])
        macd = _nan(a.macd[i])
        macd_signal = _nan(a.macd_signal[i])
        macd_hist = _nan(a.macd_hist[i])
        bb_pct = _nan(a.bb_pct[i])
        bb_squeeze = bool(a.bb_squeeze[i] > 0.5) if hasattr(a, 'bb_squeeze') else False

        # RSI zone
        if rsi < 30:
            rsi_zone = 'oversold'
        elif rsi > 70:
            rsi_zone = 'overbought'
        elif rsi < 40:
            rsi_zone = 'approaching_oversold'
        elif rsi > 60:
            rsi_zone = 'approaching_overbought'
        else:
            rsi_zone = 'neutral'

        # MACD cross detection
        macd_cross = 'none'
        if i > 0:
            prev_macd = _nan(a.macd[i - 1])
            prev_signal = _nan(a.macd_signal[i - 1])
            if prev_macd <= prev_signal and macd > macd_signal:
                macd_cross = 'bullish_cross'
            elif prev_macd >= prev_signal and macd < macd_signal:
                macd_cross = 'bearish_cross'

        # Volume
        vol_ratio = _nan(a.vol_ratio[i])
        obv_slope = _nan(a.obv_slope[i])
        if vol_ratio > 1.5:
            vol_trend = 'surging'
        elif vol_ratio > 1.1:
            vol_trend = 'above_average'
        elif vol_ratio < 0.7:
            vol_trend = 'below_average'
        else:
            vol_trend = 'average'

        # Divergences
        bull_div = bool(a.div_bull[i] > 0.5) if hasattr(a, 'div_bull') else False
        bear_div = bool(a.div_bear[i] > 0.5) if hasattr(a, 'div_bear') else False

        # Patterns
        imports = _get_backtest_imports()
        active_patterns = imports['_snapshot_active_patterns'](a, i)

        # Market structure
        struct_bias, struct_type = imports['_market_structure_bias'](a, i)

        # Recent price action (last 5 bars)
        recent_bars = []
        for j in range(max(0, i - 4), i + 1):
            bar_change = ((a.close[j] - a.open[j]) / a.open[j] * 100) if a.open[j] > 0 else 0
            bar_range = a.high[j] - a.low[j]
            body = abs(a.close[j] - a.open[j])
            bar_type = 'bullish' if a.close[j] > a.open[j] else ('bearish' if a.close[j] < a.open[j] else 'doji')
            vr = _nan(a.vol_ratio[j]) if j < len(a.vol_ratio) else 1.0
            recent_bars.append({
                'change_pct': round(float(bar_change), 2),
                'type': bar_type,
                'vol_ratio': round(float(vr), 1),
                'range_atr': round(float(bar_range / atr), 2) if atr > 0 else 0,
            })

        # Market context
        spy_trend = None
        vix_level = None
        if self.mkt is not None:
            try:
                spy_trend = int(self.mkt.spy_trend[i])
            except (IndexError, TypeError):
                pass
            try:
                vix_level = float(self.mkt.vix_level[i])
            except (IndexError, TypeError):
                pass

        return {
            'symbol': self.symbol,
            'interval': self.interval,
            'timestamp': str(self.df.index[self.i]) if self.i < len(self.df) else '',
            'price': round(price, 2),
            'open': round(bar_open, 2),
            'high': round(bar_high, 2),
            'low': round(bar_low, 2),
            'volume': int(volume),
            'high_52w': round(high_52w, 2),
            'low_52w': round(low_52w, 2),
            'dist_from_52w_high_pct': round(dist_from_high, 2),
            'dist_from_52w_low_pct': round(dist_from_low, 2),
            'ema9': round(ema9, 2),
            'ema21': round(ema21, 2),
            'ema50': round(ema50, 2),
            'ema21_slope': round(ema21_slope * 100, 2),
            'ema_alignment': ema_alignment,
            'trend_direction': trend_dir,
            'trend_strength': trend_str,
            'adx': round(adx, 1),
            'di_plus': round(di_plus, 1),
            'di_minus': round(di_minus, 1),
            'nearest_resistance': round(nearest_r, 2) if nearest_r > 0 else None,
            'nearest_support': round(nearest_s, 2) if nearest_s > 0 else None,
            'dist_to_resistance_atr': round(dist_r_atr, 1),
            'dist_to_support_atr': round(dist_s_atr, 1),
            'atr': round(atr, 2),
            'rsi': round(rsi, 1),
            'rsi_zone': rsi_zone,
            'stoch_k': round(stoch_k, 1),
            'stoch_d': round(stoch_d, 1),
            'macd': round(macd, 3),
            'macd_signal': round(macd_signal, 3),
            'macd_hist': round(macd_hist, 3),
            'macd_cross': macd_cross,
            'bb_pct': round(bb_pct, 1),
            'bb_squeeze': bb_squeeze,
            'vol_ratio': round(vol_ratio, 1),
            'vol_trend': vol_trend,
            'obv_slope': round(obv_slope, 3),
            'bull_divergence': bull_div,
            'bear_divergence': bear_div,
            'active_patterns': active_patterns,
            'structure_bias': struct_bias,
            'structure_type': struct_type,
            'recent_bars': recent_bars,
            'spy_trend': spy_trend,
            'vix_level': round(vix_level, 1) if vix_level is not None else None,
        }

    def to_text(self) -> str:
        """Build natural language market summary for LLM prompt."""
        ctx = self.build()
        sections = []

        # 1. PRICE OVERVIEW
        price_lines = [f"## PRICE OVERVIEW",
                        f"{ctx['symbol']} is trading at ${ctx['price']:.2f}."]
        price_lines.append(
            f"Today's bar: Open ${ctx['open']:.2f}, High ${ctx['high']:.2f}, "
            f"Low ${ctx['low']:.2f}, Volume {ctx['volume']:,}.")
        if ctx['high_52w'] > 0:
            price_lines.append(
                f"52-week range: ${ctx['low_52w']:.2f} — ${ctx['high_52w']:.2f}. "
                f"Price is {abs(ctx['dist_from_52w_high_pct']):.1f}% below the high "
                f"and {ctx['dist_from_52w_low_pct']:.1f}% above the low.")
        sections.append('\n'.join(price_lines))

        # 2. TREND
        trend_lines = [f"## TREND"]
        trend_lines.append(
            f"Direction: {ctx['trend_direction'].upper()} ({ctx['trend_strength']}). "
            f"EMA alignment: {ctx['ema_alignment'].replace('_', ' ')} "
            f"(EMA9={ctx['ema9']:.2f}, EMA21={ctx['ema21']:.2f}, EMA50={ctx['ema50']:.2f}).")
        trend_lines.append(
            f"EMA21 slope: {ctx['ema21_slope']:+.2f}%. "
            f"ADX: {ctx['adx']:.1f} (DI+={ctx['di_plus']:.1f}, DI-={ctx['di_minus']:.1f}).")
        price_vs = []
        if ctx['price'] > ctx['ema9']:
            price_vs.append('above EMA9')
        else:
            price_vs.append('below EMA9')
        if ctx['price'] > ctx['ema21']:
            price_vs.append('above EMA21')
        else:
            price_vs.append('below EMA21')
        if ctx['price'] > ctx['ema50']:
            price_vs.append('above EMA50')
        else:
            price_vs.append('below EMA50')
        trend_lines.append(f"Price is {', '.join(price_vs)}.")
        sections.append('\n'.join(trend_lines))

        # 3. KEY LEVELS
        levels_lines = [f"## KEY LEVELS"]
        if ctx['nearest_resistance']:
            levels_lines.append(
                f"Nearest resistance: ${ctx['nearest_resistance']:.2f} "
                f"({ctx['dist_to_resistance_atr']:.1f} ATR away).")
        else:
            levels_lines.append("No clear resistance detected nearby.")
        if ctx['nearest_support']:
            levels_lines.append(
                f"Nearest support: ${ctx['nearest_support']:.2f} "
                f"({ctx['dist_to_support_atr']:.1f} ATR away).")
        else:
            levels_lines.append("No clear support detected nearby.")
        levels_lines.append(f"ATR(14): ${ctx['atr']:.2f} ({ctx['atr']/ctx['price']*100:.1f}% of price).")
        sections.append('\n'.join(levels_lines))

        # 4. MOMENTUM
        mom_lines = [f"## MOMENTUM"]
        mom_lines.append(
            f"RSI(14): {ctx['rsi']:.1f} ({ctx['rsi_zone'].replace('_', ' ')}). "
            f"Stochastic: %K={ctx['stoch_k']:.1f}, %D={ctx['stoch_d']:.1f}.")
        macd_desc = f"MACD: {ctx['macd']:.3f}, Signal: {ctx['macd_signal']:.3f}, Histogram: {ctx['macd_hist']:.3f}"
        if ctx['macd_cross'] != 'none':
            macd_desc += f" — {ctx['macd_cross'].replace('_', ' ').upper()}"
        elif ctx['macd_hist'] > 0:
            macd_desc += " (bullish momentum)"
        else:
            macd_desc += " (bearish momentum)"
        mom_lines.append(macd_desc + '.')
        bb_desc = f"Bollinger Band position: {ctx['bb_pct']:.0f}%"
        if ctx['bb_squeeze']:
            bb_desc += " — SQUEEZE ACTIVE (low volatility, breakout expected)"
        mom_lines.append(bb_desc + '.')
        sections.append('\n'.join(mom_lines))

        # 5. VOLUME
        vol_lines = [f"## VOLUME"]
        vol_lines.append(
            f"Volume ratio: {ctx['vol_ratio']:.1f}x average ({ctx['vol_trend'].replace('_', ' ')}).")
        if ctx['obv_slope'] > 0.1:
            vol_lines.append(f"OBV slope: {ctx['obv_slope']:.3f} (accumulation — money flowing in).")
        elif ctx['obv_slope'] < -0.1:
            vol_lines.append(f"OBV slope: {ctx['obv_slope']:.3f} (distribution — money flowing out).")
        else:
            vol_lines.append(f"OBV slope: {ctx['obv_slope']:.3f} (neutral flow).")
        sections.append('\n'.join(vol_lines))

        # 6. PATTERNS
        pat_lines = [f"## DETECTED PATTERNS"]
        if ctx['active_patterns']:
            bullish_pats = [p for p in ctx['active_patterns'] if any(
                b in p for b in ['BULL', 'HAMMER', 'MORNING', 'PIERCING', 'THREE_WHITE',
                                  'DOJI_DRAGON', 'BOTTOM', 'ASC_TRIANGLE', 'FALLING_WEDGE',
                                  'CUP_HANDLE', 'INV_HS', 'VOL_SURGE'])]
            bearish_pats = [p for p in ctx['active_patterns'] if any(
                b in p for b in ['BEAR', 'SHOOTING', 'EVENING', 'DARK_CLOUD', 'THREE_BLACK',
                                  'DOJI_GRAVE', 'TOP', 'DESC_TRIANGLE', 'RISING_WEDGE',
                                  'HEAD_SHOULDERS', 'HANGING_MAN'])]
            neutral_pats = [p for p in ctx['active_patterns']
                           if p not in bullish_pats and p not in bearish_pats]
            if bullish_pats:
                pat_lines.append(f"Bullish: {', '.join(bullish_pats)}.")
            if bearish_pats:
                pat_lines.append(f"Bearish: {', '.join(bearish_pats)}.")
            if neutral_pats:
                pat_lines.append(f"Other: {', '.join(neutral_pats)}.")
        else:
            pat_lines.append("No significant patterns detected.")
        sections.append('\n'.join(pat_lines))

        # 7. MARKET STRUCTURE
        struct_lines = [f"## MARKET STRUCTURE"]
        struct_labels = {
            'uptrend': 'UPTREND (higher highs + higher lows)',
            'downtrend': 'DOWNTREND (lower highs + lower lows)',
            'accumulation': 'ACCUMULATION (higher lows forming)',
            'distribution': 'DISTRIBUTION (lower highs forming)',
            'w_bottom_confirmed': 'W-BOTTOM CONFIRMED (double bottom breakout)',
            'w_bottom_forming': 'W-BOTTOM FORMING (potential reversal)',
            'm_top_confirmed': 'M-TOP CONFIRMED (double top breakdown)',
            'm_top_forming': 'M-TOP FORMING (potential reversal)',
            'triangle': 'TRIANGLE (converging price range)',
            'broadening': 'BROADENING (expanding price range)',
            'range': 'RANGE-BOUND',
        }
        label = struct_labels.get(ctx['structure_type'], ctx['structure_type'].upper())
        bias_str = {-2: 'strongly bearish', -1: 'bearish', 0: 'neutral',
                    1: 'bullish', 2: 'strongly bullish'}.get(ctx['structure_bias'], 'neutral')
        struct_lines.append(f"Structure: {label}. Bias: {bias_str} ({ctx['structure_bias']:+d}).")
        sections.append('\n'.join(struct_lines))

        # 8. DIVERGENCES
        if ctx['bull_divergence'] or ctx['bear_divergence']:
            div_lines = [f"## DIVERGENCES"]
            if ctx['bull_divergence']:
                div_lines.append("BULLISH DIVERGENCE detected: price making lower lows but RSI/MACD making higher lows.")
            if ctx['bear_divergence']:
                div_lines.append("BEARISH DIVERGENCE detected: price making higher highs but RSI/MACD making lower highs.")
            sections.append('\n'.join(div_lines))

        # 9. MARKET CONTEXT
        if ctx['spy_trend'] is not None or ctx['vix_level'] is not None:
            mkt_lines = [f"## MARKET CONTEXT"]
            if ctx['spy_trend'] is not None:
                spy_label = {1: 'BULLISH (EMA9>21>50)', -1: 'BEARISH (EMA9<21<50)',
                             0: 'NEUTRAL'}.get(ctx['spy_trend'], 'NEUTRAL')
                mkt_lines.append(f"SPY trend: {spy_label}.")
            if ctx['vix_level'] is not None:
                vix_desc = 'elevated fear' if ctx['vix_level'] > 25 else (
                    'high fear' if ctx['vix_level'] > 30 else 'normal')
                mkt_lines.append(f"VIX: {ctx['vix_level']:.1f} ({vix_desc}).")
            sections.append('\n'.join(mkt_lines))

        # 10. RECENT PRICE ACTION
        if ctx['recent_bars']:
            recent_lines = [f"## RECENT PRICE ACTION (last {len(ctx['recent_bars'])} bars)"]
            for j, bar in enumerate(ctx['recent_bars']):
                bar_num = len(ctx['recent_bars']) - j
                vol_desc = ''
                if bar['vol_ratio'] >= 1.5:
                    vol_desc = ', HIGH volume'
                elif bar['vol_ratio'] <= 0.7:
                    vol_desc = ', low volume'
                recent_lines.append(
                    f"  Bar -{bar_num}: {bar['change_pct']:+.2f}% {bar['type']} candle "
                    f"(range {bar['range_atr']:.1f}x ATR{vol_desc})")
            sections.append('\n'.join(recent_lines))

        return '\n\n'.join(sections)


# =============================================================================
# LLMPromptBuilder — Structured prompts with trading-style awareness
# =============================================================================

class LLMPromptBuilder:
    """Constructs detailed LLM prompts for trading predictions.

    Unlike the old PromptBuilder in claude_strategy.py which was hardcoded as
    "intraday day trader", this builder adapts to trading style (day/swing/position)
    and requires richer structured output including bull/bear cases and invalidation.
    """

    STYLE_CONFIG = {
        'day': {
            'role': 'expert intraday trader',
            'horizon': 'within today\'s session (minutes to hours)',
            'target': '0.3%-1.5% moves',
            'holding': 'minutes to hours, never overnight',
        },
        'swing': {
            'role': 'expert swing trader and technical analyst',
            'horizon': '2-10 trading days',
            'target': '2%-8% moves',
            'holding': 'days to 2 weeks',
        },
        'position': {
            'role': 'expert position trader and macro analyst',
            'horizon': 'weeks to months',
            'target': '5%-25% moves',
            'holding': 'weeks to several months',
        },
    }

    RESPONSE_SCHEMA = '''{
  "analysis": "2-3 sentence overall market assessment",
  "bull_case": "strongest bullish argument with specific price levels",
  "bear_case": "strongest bearish argument with specific price levels",
  "prediction": "BUY" | "SELL" | "HOLD",
  "confidence": 0.0 to 1.0,
  "entry_price": number or null,
  "stop_loss": number or null,
  "targets": [target_price_1, target_price_2],
  "risk_reward": "1:2.5",
  "timeframe": "expected holding period",
  "key_levels": {"support": [price1, price2], "resistance": [price1, price2]},
  "invalidation": "specific condition that would invalidate this thesis"
}'''

    def __init__(self, trading_style: str = 'swing'):
        self.trading_style = trading_style
        self.style = self.STYLE_CONFIG.get(trading_style, self.STYLE_CONFIG['swing'])

    def build(self, context_text: str, current_position: Optional[dict] = None) -> Tuple[str, str]:
        """Build (system_prompt, user_prompt) tuple.

        Args:
            context_text: Natural language market summary from DataPrep.to_text().
            current_position: Dict with 'side', 'entry_price', 'size', 'pnl' or None.

        Returns:
            (system_prompt, user_prompt) tuple.
        """
        return self._build_system_prompt(), self._build_user_prompt(context_text, current_position)

    def _build_system_prompt(self) -> str:
        return f"""You are an {self.style['role']} with deep expertise in technical analysis, \
price action, and market structure. Your job is to analyze market data and make \
precise trading decisions.

TRADING STYLE: {self.trading_style.upper()}
- Time horizon: {self.style['horizon']}
- Target moves: {self.style['target']}
- Typical holding period: {self.style['holding']}

You MUST return ONLY a JSON object with these exact fields:
{self.RESPONSE_SCHEMA}

CRITICAL RULES:
1. TREND FIRST: Identify the dominant trend from EMA alignment, ADX, and market structure. \
Trade WITH the trend unless you see strong reversal evidence.
2. CONFLUENCE: Only signal BUY or SELL when 3+ indicators agree. Mixed signals = HOLD.
3. RISK/REWARD: Every trade must have a clear stop loss and at least 1:2 risk/reward ratio.
4. VOLUME CONFIRMS: Strong volume confirms breakouts and reversals. Low volume = suspect.
5. PATTERNS MATTER: Candlestick and chart patterns are leading indicators. Weight them heavily.
6. DIVERGENCES WARN: RSI/MACD divergences often precede reversals. Don't ignore them.
7. MARKET CONTEXT: SPY trend and VIX level set the macro backdrop. Don't fight the market.
8. INVALIDATION: Every thesis has a failure point. State it clearly.
9. HOLD is the default when signals are mixed, unclear, or risk/reward is poor.
10. Only signal with confidence >= 0.6. Below that, signal HOLD.

When a position exists:
- BUY = go long (if flat) or cover short (if short)
- SELL = go short (if flat) or close long (if long)
- HOLD = maintain current position

Capital preservation is priority #1. Return ONLY the JSON object, no other text."""

    def _build_user_prompt(self, context_text: str, current_position: Optional[dict]) -> str:
        parts = []

        # Position context
        if current_position and current_position.get('side'):
            side = current_position['side']
            entry = current_position.get('entry_price', 0)
            size = current_position.get('size', 0)
            pnl = current_position.get('pnl', 0)
            pnl_pct = current_position.get('pnl_pct', 0)

            parts.append(f"## CURRENT POSITION")
            parts.append(f"Side: {side.upper()}, Entry: ${entry:.2f}, Size: {size} shares")
            parts.append(f"Unrealized P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")

            if pnl_pct >= 1.0:
                parts.append("STATUS: STRONGLY PROFITABLE — consider taking profits!")
            elif pnl_pct >= 0.3:
                parts.append("STATUS: Profitable — monitor for exit signals.")
            elif pnl_pct <= -1.0:
                parts.append("STATUS: SIGNIFICANT LOSS — strongly consider cutting losses!")
            elif pnl_pct <= -0.3:
                parts.append("STATUS: Losing — watch closely for exit.")
            else:
                parts.append("STATUS: Near breakeven.")

            if side.lower() == 'long':
                parts.append("To EXIT this long position, signal SELL.")
            else:
                parts.append("To EXIT this short position, signal BUY.")
            parts.append("")
        else:
            parts.append("## CURRENT POSITION: FLAT (no open position)")
            parts.append("Signal BUY to go long, SELL to go short, HOLD to stay flat.")
            parts.append("")

        # Market data
        parts.append(context_text)

        return '\n'.join(parts)


# =============================================================================
# LLMPrediction — Structured prediction result
# =============================================================================

@dataclass
class LLMPrediction:
    """Structured prediction from the LLM."""
    timestamp: str
    symbol: str
    bar_index: int
    price_at_prediction: float

    # LLM response fields
    analysis: str
    bull_case: str
    bear_case: str
    prediction: str              # 'BUY', 'SELL', 'HOLD'
    confidence: float
    entry_price: Optional[float]
    stop_loss: Optional[float]
    targets: List[float]
    risk_reward: str
    timeframe: str
    key_levels: dict
    invalidation: str

    # Metadata
    model_used: str = ''
    provider: str = ''
    response_time_ms: float = 0.0
    raw_response: str = ''

    # Context at prediction time (for BacktestTracker)
    active_patterns: List[str] = field(default_factory=list)
    structure_type: str = ''
    structure_bias: int = 0


# =============================================================================
# Predictor — LLM API calls with robust parsing
# =============================================================================

class Predictor:
    """Calls LLM (Ollama or Anthropic) and parses trading predictions.

    Reuses the proven patterns from ClaudeAPIClient in claude_strategy.py:
    - Circuit breaker for API failures
    - <think> tag stripping for local models
    - Robust JSON extraction with fallback regex
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.llm_provider = config.get('llm_provider', 'ollama')
        self.ollama_url = config.get('ollama_url', 'http://localhost:11434')
        self.ollama_model = config.get('ollama_model', 'qwen3-coder:30b')
        self.api_key = config.get('api_key', os.environ.get('ANTHROPIC_API_KEY', ''))
        self.model = config.get('model', 'claude-3-5-haiku-latest')
        self.max_tokens = config.get('max_tokens', 1024)
        self.temperature = config.get('temperature', 0.3)
        self.timeout = config.get('timeout', 120)

        # Circuit breaker
        self._failure_count = 0
        self._circuit_open_until = 0.0
        self._max_failures = 10
        self._circuit_reset_seconds = 60

        # Prediction log
        self._log_dir = os.path.dirname(os.path.abspath(__file__))
        self._log_file = os.path.join(self._log_dir, 'prediction_log.jsonl')

    @property
    def is_available(self) -> bool:
        if self.llm_provider == 'ollama':
            return True
        return bool(self.api_key)

    def _check_circuit_breaker(self) -> bool:
        """Returns True if circuit is open (should NOT call API)."""
        if self._failure_count >= self._max_failures:
            if time.time() < self._circuit_open_until:
                return True
            # Reset after cooldown
            self._failure_count = 0
        return False

    async def predict(self, system_prompt: str, user_prompt: str,
                      context_dict: dict, bar_index: int) -> Optional[LLMPrediction]:
        """Call LLM and parse response into LLMPrediction.

        Args:
            system_prompt: From LLMPromptBuilder.build().
            user_prompt: From LLMPromptBuilder.build().
            context_dict: From DataPrep.build() — for metadata.
            bar_index: Current bar index for logging.

        Returns:
            LLMPrediction on success, None on failure.
        """
        if not self.is_available:
            logger.warning(f"Predictor not available (provider={self.llm_provider})")
            return None

        if self._check_circuit_breaker():
            logger.warning("Predictor circuit breaker open — skipping call")
            return None

        start_time = time.time()

        # Call LLM
        if self.llm_provider == 'ollama':
            raw_response = await self._call_ollama(system_prompt, user_prompt)
        else:
            raw_response = await self._call_anthropic(system_prompt, user_prompt)

        elapsed_ms = (time.time() - start_time) * 1000

        if raw_response is None:
            return None

        # Parse response
        parsed = self._parse_response(raw_response)
        if parsed is None:
            logger.error(f"Failed to parse LLM response: {raw_response[:200]}")
            return None

        # Build prediction
        symbol = context_dict.get('symbol', 'UNKNOWN')
        prediction = LLMPrediction(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            bar_index=bar_index,
            price_at_prediction=context_dict.get('price', 0),
            analysis=parsed.get('analysis', ''),
            bull_case=parsed.get('bull_case', ''),
            bear_case=parsed.get('bear_case', ''),
            prediction=parsed.get('prediction', 'HOLD').upper(),
            confidence=max(0.0, min(1.0, float(parsed.get('confidence', 0.5)))),
            entry_price=parsed.get('entry_price'),
            stop_loss=parsed.get('stop_loss'),
            targets=parsed.get('targets', []),
            risk_reward=str(parsed.get('risk_reward', 'N/A')),
            timeframe=str(parsed.get('timeframe', 'N/A')),
            key_levels=parsed.get('key_levels', {}),
            invalidation=str(parsed.get('invalidation', '')),
            model_used=self.ollama_model if self.llm_provider == 'ollama' else self.model,
            provider=self.llm_provider,
            response_time_ms=round(elapsed_ms, 1),
            raw_response=raw_response,
            active_patterns=context_dict.get('active_patterns', []),
            structure_type=context_dict.get('structure_type', ''),
            structure_bias=context_dict.get('structure_bias', 0),
        )

        # Validate prediction field
        if prediction.prediction not in ('BUY', 'SELL', 'HOLD'):
            prediction.prediction = 'HOLD'

        # Ensure targets is a list of floats
        clean_targets = []
        for t in prediction.targets:
            try:
                clean_targets.append(float(t))
            except (TypeError, ValueError):
                pass
        prediction.targets = clean_targets

        # Log prediction
        self._log_prediction(prediction)

        return prediction

    async def _call_ollama(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call local Ollama API."""
        if aiohttp is None:
            logger.error("aiohttp not available for Ollama call")
            return None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f'{self.ollama_url}/api/chat',
                    json={
                        'model': self.ollama_model,
                        'messages': [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': user_prompt},
                        ],
                        'stream': False,
                        'options': {
                            'temperature': self.temperature,
                            'num_predict': self.max_tokens,
                        },
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data.get('message', {}).get('content', '')
                        # Strip <think>...</think> tags (qwen3-coder thinking mode)
                        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                        self._failure_count = 0
                        return content
                    else:
                        error_text = await resp.text()
                        logger.error(f"Ollama error {resp.status}: {error_text[:200]}")
                        self._failure_count += 1
                        if self._failure_count >= self._max_failures:
                            self._circuit_open_until = time.time() + self._circuit_reset_seconds
                        return None
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            self._failure_count += 1
            if self._failure_count >= self._max_failures:
                self._circuit_open_until = time.time() + self._circuit_reset_seconds
            return None

    async def _call_anthropic(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call Anthropic Claude API."""
        if aiohttp is None:
            logger.error("aiohttp not available for Anthropic call")
            return None
        if not self.api_key:
            logger.error("No Anthropic API key")
            return None
        try:
            headers = {
                'x-api-key': self.api_key,
                'content-type': 'application/json',
                'anthropic-version': '2023-06-01',
            }
            payload = {
                'model': self.model,
                'max_tokens': self.max_tokens,
                'system': system_prompt,
                'messages': [{'role': 'user', 'content': user_prompt}],
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://api.anthropic.com/v1/messages',
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data['content'][0]['text']
                        self._failure_count = 0
                        return content
                    elif resp.status in (429, 529):
                        logger.warning(f"Anthropic rate limited ({resp.status}), will retry")
                        self._failure_count += 1
                        return None
                    else:
                        error_text = await resp.text()
                        logger.error(f"Anthropic error {resp.status}: {error_text[:200]}")
                        self._failure_count += 1
                        if self._failure_count >= self._max_failures:
                            self._circuit_open_until = time.time() + self._circuit_reset_seconds
                        return None
        except Exception as e:
            logger.error(f"Anthropic call failed: {e}")
            self._failure_count += 1
            if self._failure_count >= self._max_failures:
                self._circuit_open_until = time.time() + self._circuit_reset_seconds
            return None

    def _parse_response(self, response_text: str) -> Optional[dict]:
        """Parse LLM response into a dict. Robust with multiple fallbacks."""
        if not response_text:
            return None

        text = response_text.strip()

        # Strip <think> tags (belt-and-suspenders)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Strip markdown code fences
        text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        # Try direct JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return self._fill_defaults(data)
        except json.JSONDecodeError:
            pass

        # Regex fallback: extract first JSON object
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, dict):
                    return self._fill_defaults(data)
            except json.JSONDecodeError:
                pass

        # Last resort: try to find any JSON-like content
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, dict):
                    return self._fill_defaults(data)
            except json.JSONDecodeError:
                pass

        return None

    def _fill_defaults(self, data: dict) -> dict:
        """Fill missing fields with defaults."""
        data.setdefault('analysis', '')
        data.setdefault('bull_case', '')
        data.setdefault('bear_case', '')
        data.setdefault('prediction', 'HOLD')
        data.setdefault('confidence', 0.5)
        data.setdefault('entry_price', None)
        data.setdefault('stop_loss', None)
        data.setdefault('targets', [])
        data.setdefault('risk_reward', 'N/A')
        data.setdefault('timeframe', 'N/A')
        data.setdefault('key_levels', {})
        data.setdefault('invalidation', '')

        # Handle old-format responses (signal/strength/reasoning)
        if 'signal' in data and 'prediction' not in data:
            data['prediction'] = data.pop('signal', 'HOLD')
        if 'strength' in data and 'confidence' not in data:
            data['confidence'] = data.pop('strength', 0.5)
        if 'reasoning' in data and 'analysis' not in data:
            data['analysis'] = data.pop('reasoning', '')
        if 'take_profit' in data and not data.get('targets'):
            tp = data.pop('take_profit', None)
            if tp is not None:
                data['targets'] = [tp]

        return data

    def _log_prediction(self, prediction: LLMPrediction):
        """Append prediction to JSONL log file."""
        entry = {
            'timestamp': prediction.timestamp,
            'symbol': prediction.symbol,
            'bar_index': prediction.bar_index,
            'price': prediction.price_at_prediction,
            'prediction': prediction.prediction,
            'confidence': prediction.confidence,
            'entry_price': prediction.entry_price,
            'stop_loss': prediction.stop_loss,
            'targets': prediction.targets,
            'risk_reward': prediction.risk_reward,
            'invalidation': prediction.invalidation,
            'analysis': prediction.analysis,
            'bull_case': prediction.bull_case,
            'bear_case': prediction.bear_case,
            'model': prediction.model_used,
            'provider': prediction.provider,
            'response_time_ms': prediction.response_time_ms,
            'patterns': prediction.active_patterns,
            'structure': prediction.structure_type,
        }
        try:
            with open(self._log_file, 'a') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write prediction log: {e}")

    def to_strategy_signal(self, prediction: LLMPrediction) -> StrategySignal:
        """Convert LLMPrediction to StrategySignal for existing trade execution."""
        signal_map = {
            'BUY': SignalType.BUY,
            'SELL': SignalType.SELL,
            'HOLD': SignalType.HOLD,
        }
        return StrategySignal(
            symbol=prediction.symbol,
            signal_type=signal_map.get(prediction.prediction, SignalType.HOLD),
            strength=prediction.confidence,
            strategy_name='llm_predictor',
            timestamp=datetime.now(),
            entry_price=prediction.entry_price,
            stop_loss=prediction.stop_loss,
            take_profit=prediction.targets[0] if prediction.targets else None,
            metadata={
                'reasoning': prediction.analysis,
                'risk_notes': (
                    f"Bull: {prediction.bull_case} | "
                    f"Bear: {prediction.bear_case} | "
                    f"Invalidation: {prediction.invalidation}"
                ),
                'source': f'llm_{prediction.provider}',
                'factors_used': ['price_action', 'technical_indicators', 'market_regime'],
                'raw_response': {
                    'analysis': prediction.analysis,
                    'bull_case': prediction.bull_case,
                    'bear_case': prediction.bear_case,
                    'prediction': prediction.prediction,
                    'confidence': prediction.confidence,
                    'entry_price': prediction.entry_price,
                    'stop_loss': prediction.stop_loss,
                    'targets': prediction.targets,
                    'risk_reward': prediction.risk_reward,
                    'timeframe': prediction.timeframe,
                    'key_levels': prediction.key_levels,
                    'invalidation': prediction.invalidation,
                },
                'backtest_mode': 'live_api',
            },
        )


# =============================================================================
# BacktestTracker — Prediction accuracy tracking
# =============================================================================

@dataclass
class PredictionOutcome:
    """Links a prediction to its actual result."""
    prediction: LLMPrediction
    exit_price: float
    exit_bar: int
    pnl: float
    return_pct: float
    r_multiple: float
    holding_bars: int
    hit_target: bool
    hit_stop: bool
    max_favorable: float       # max favorable excursion %
    max_adverse: float         # max adverse excursion %


class BacktestTracker:
    """Tracks LLM prediction accuracy with per-pattern and per-structure stats."""

    def __init__(self):
        self._outcomes: List[PredictionOutcome] = []
        self._pending: Dict[str, LLMPrediction] = {}  # symbol -> open prediction
        self._pattern_stats: Dict[str, dict] = {}
        self._structure_stats: Dict[str, dict] = {}
        self._confidence_buckets: Dict[str, dict] = {}

    def record_prediction(self, prediction: LLMPrediction):
        """Record a new prediction (pending outcome)."""
        if prediction.prediction in ('BUY', 'SELL'):
            self._pending[prediction.symbol] = prediction

    def record_outcome(self, symbol: str, exit_price: float, exit_bar: int,
                       pnl: float, return_pct: float,
                       arrays=None, entry_bar: Optional[int] = None):
        """Record the actual outcome of a prediction.

        Called when a trade closes in BacktestRunner.
        """
        prediction = self._pending.pop(symbol, None)
        if prediction is None:
            return

        # R-multiple
        r_multiple = 0.0
        if prediction.stop_loss and prediction.entry_price:
            risk = abs(prediction.entry_price - prediction.stop_loss)
            if risk > 0:
                r_multiple = pnl / risk if prediction.prediction == 'BUY' else (
                    -pnl / risk if prediction.prediction == 'SELL' else 0)
                # For shorts, positive pnl means winning, r_multiple should be positive
                # Actually pnl is already signed correctly from BacktestRunner
                r_multiple = pnl / risk if risk > 0 else 0

        # Max favorable/adverse excursion
        max_favorable = 0.0
        max_adverse = 0.0
        hit_target = False
        hit_stop = pnl < 0

        if arrays is not None and entry_bar is not None and entry_bar < exit_bar:
            entry_p = prediction.entry_price or float(arrays.close[entry_bar])
            for j in range(entry_bar, min(exit_bar + 1, arrays.n)):
                if prediction.prediction == 'BUY':
                    fav = (float(arrays.high[j]) - entry_p) / entry_p * 100
                    adv = (entry_p - float(arrays.low[j])) / entry_p * 100
                else:  # SELL/short
                    fav = (entry_p - float(arrays.low[j])) / entry_p * 100
                    adv = (float(arrays.high[j]) - entry_p) / entry_p * 100
                max_favorable = max(max_favorable, fav)
                max_adverse = max(max_adverse, adv)

            # Check if any target was hit
            for t in (prediction.targets or []):
                try:
                    target = float(t)
                except (TypeError, ValueError):
                    continue
                for j in range(entry_bar, min(exit_bar + 1, arrays.n)):
                    if prediction.prediction == 'BUY' and float(arrays.high[j]) >= target:
                        hit_target = True
                        break
                    elif prediction.prediction == 'SELL' and float(arrays.low[j]) <= target:
                        hit_target = True
                        break
                if hit_target:
                    break

        outcome = PredictionOutcome(
            prediction=prediction,
            exit_price=exit_price,
            exit_bar=exit_bar,
            pnl=pnl,
            return_pct=return_pct,
            r_multiple=r_multiple,
            holding_bars=exit_bar - prediction.bar_index,
            hit_target=hit_target,
            hit_stop=hit_stop,
            max_favorable=round(max_favorable, 2),
            max_adverse=round(max_adverse, 2),
        )
        self._outcomes.append(outcome)

        # Update stats
        is_win = pnl > 0
        self._update_pattern_stats(prediction, is_win, r_multiple)
        self._update_structure_stats(prediction, is_win, r_multiple)
        self._update_confidence_stats(prediction, is_win)

    def _update_pattern_stats(self, prediction: LLMPrediction, is_win: bool, r_multiple: float):
        for pattern in prediction.active_patterns:
            if pattern not in self._pattern_stats:
                self._pattern_stats[pattern] = {'wins': 0, 'losses': 0, 'total_r': 0.0, 'count': 0}
            s = self._pattern_stats[pattern]
            s['count'] += 1
            s['wins' if is_win else 'losses'] += 1
            s['total_r'] += r_multiple

    def _update_structure_stats(self, prediction: LLMPrediction, is_win: bool, r_multiple: float):
        st = prediction.structure_type or 'unknown'
        if st not in self._structure_stats:
            self._structure_stats[st] = {'wins': 0, 'losses': 0, 'total_r': 0.0, 'count': 0}
        s = self._structure_stats[st]
        s['count'] += 1
        s['wins' if is_win else 'losses'] += 1
        s['total_r'] += r_multiple

    def _update_confidence_stats(self, prediction: LLMPrediction, is_win: bool):
        bucket_low = int(prediction.confidence * 10) / 10
        bucket = f"{bucket_low:.1f}-{bucket_low + 0.1:.1f}"
        if bucket not in self._confidence_buckets:
            self._confidence_buckets[bucket] = {'wins': 0, 'losses': 0, 'count': 0}
        b = self._confidence_buckets[bucket]
        b['count'] += 1
        b['wins' if is_win else 'losses'] += 1

    def report(self) -> dict:
        """Generate comprehensive accuracy report."""
        if not self._outcomes:
            return {
                'total_predictions': len(self._pending),
                'total_trades': 0,
                'outcomes': [],
            }

        wins = [o for o in self._outcomes if o.pnl > 0]
        r_values = [o.r_multiple for o in self._outcomes]

        best = max(self._outcomes, key=lambda o: o.r_multiple)
        worst = min(self._outcomes, key=lambda o: o.r_multiple)

        # Best/worst conditions (min 3 trades for significance)
        all_stats = {}
        all_stats.update(self._pattern_stats)
        all_stats.update(self._structure_stats)
        conditions = []
        for name, stats in all_stats.items():
            if stats['count'] >= 3:
                conditions.append({
                    'name': name,
                    'win_rate': round(stats['wins'] / stats['count'], 2),
                    'avg_r': round(stats['total_r'] / stats['count'], 2),
                    'count': stats['count'],
                })
        conditions.sort(key=lambda x: x['avg_r'], reverse=True)

        target_hits = sum(1 for o in self._outcomes if o.hit_target)

        return {
            'total_predictions': len(self._outcomes) + len(self._pending),
            'total_trades': len(self._outcomes),
            'win_rate': round(len(wins) / len(self._outcomes), 3) if self._outcomes else 0,
            'avg_r_multiple': round(sum(r_values) / len(r_values), 3) if r_values else 0,
            'total_r': round(sum(r_values), 2),
            'target_hit_rate': round(target_hits / len(self._outcomes), 3) if self._outcomes else 0,
            'avg_response_time_ms': round(
                sum(o.prediction.response_time_ms for o in self._outcomes) / len(self._outcomes), 0
            ),
            'avg_holding_bars': round(
                sum(o.holding_bars for o in self._outcomes) / len(self._outcomes), 1
            ),
            'avg_max_favorable_pct': round(
                sum(o.max_favorable for o in self._outcomes) / len(self._outcomes), 2
            ),
            'avg_max_adverse_pct': round(
                sum(o.max_adverse for o in self._outcomes) / len(self._outcomes), 2
            ),
            'best_prediction': {
                'symbol': best.prediction.symbol,
                'bar': best.prediction.bar_index,
                'prediction': best.prediction.prediction,
                'confidence': best.prediction.confidence,
                'r_multiple': round(best.r_multiple, 2),
                'return_pct': round(best.return_pct, 2),
                'patterns': best.prediction.active_patterns,
                'analysis': best.prediction.analysis[:200],
            },
            'worst_prediction': {
                'symbol': worst.prediction.symbol,
                'bar': worst.prediction.bar_index,
                'prediction': worst.prediction.prediction,
                'confidence': worst.prediction.confidence,
                'r_multiple': round(worst.r_multiple, 2),
                'return_pct': round(worst.return_pct, 2),
                'patterns': worst.prediction.active_patterns,
                'analysis': worst.prediction.analysis[:200],
            },
            'per_pattern_accuracy': {
                name: {
                    'win_rate': round(s['wins'] / s['count'], 2) if s['count'] > 0 else 0,
                    'avg_r': round(s['total_r'] / s['count'], 2) if s['count'] > 0 else 0,
                    'count': s['count'],
                }
                for name, s in sorted(self._pattern_stats.items(),
                                       key=lambda x: x[1]['total_r'] / max(1, x[1]['count']),
                                       reverse=True)
            },
            'per_structure_accuracy': {
                name: {
                    'win_rate': round(s['wins'] / s['count'], 2) if s['count'] > 0 else 0,
                    'avg_r': round(s['total_r'] / s['count'], 2) if s['count'] > 0 else 0,
                    'count': s['count'],
                }
                for name, s in self._structure_stats.items()
            },
            'per_confidence_accuracy': {
                bucket: {
                    'win_rate': round(b['wins'] / b['count'], 2) if b['count'] > 0 else 0,
                    'count': b['count'],
                }
                for bucket, b in sorted(self._confidence_buckets.items())
            },
            'best_conditions': conditions[:5],
            'worst_conditions': conditions[-5:][::-1] if len(conditions) >= 5 else [],
        }
