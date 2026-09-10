import logging
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np

from core.models import CommentaryType, SignalType, TradingSignal, MarketData
from core.commentary import TradingCommentary
from strategies.base import TradingStrategyWithCommentary

logger = logging.getLogger('TradingBot')

# ATR floor: 1% of price. Prevents microscopically small stops when
# ATR is computed on too few bars (e.g., first 15 min after market open).
# NIO at $6.70 had ATR=$0.028 → stop $0.04 from entry → 7,052 shares.
# With floor: ATR = max(0.028, 6.70*0.01=0.067) → stop $0.10 → ~3,300 shares.
_ATR_FLOOR_PCT = 0.01


def _floored_atr(raw_atr: float, price: float) -> float:
    """Ensure ATR is at least _ATR_FLOOR_PCT of price."""
    return max(float(raw_atr), price * _ATR_FLOOR_PCT)


# v-shadow-log-gating-2026-06-10: session + throttle gate for the SHORT
# shadow ledger. Day-1 soak wrote 81,970 rows of which ~1 was usable:
# after-close evaluations carried corrupt prices (NVDA at half its real
# price), the pre-open/warm-up window showed every watchlist symbol
# RSI>=70 at once (indicators on too few fresh bars), and each symbol
# re-logged every analysis pass. Window starts 10:00 ET, not 09:30 —
# the 2026-06-10 ledger shows artifacts persisting until ~10:00.
_SHADOW_LOG_THROTTLE: Dict[str, datetime] = {}
_SHADOW_THROTTLE_MIN = 30.0  # minutes between entries per symbol


def _shadow_short_should_log(
    symbol: str,
    now_utc: Optional[datetime] = None,
    _throttle: Optional[Dict[str, datetime]] = None,
) -> bool:
    """True when a shadow SHORT entry for `symbol` is worth recording:
    weekday, 10:00-16:00 ET, and not logged within the last 30 min.
    `now_utc`/`_throttle` are injectable for tests."""
    from datetime import timezone
    from zoneinfo import ZoneInfo
    throttle = _SHADOW_LOG_THROTTLE if _throttle is None else _throttle
    now = now_utc or datetime.now(timezone.utc)
    et = now.astimezone(ZoneInfo("America/New_York"))
    if et.weekday() >= 5:
        return False
    minutes = et.hour * 60 + et.minute
    if not (10 * 60 <= minutes < 16 * 60):
        return False
    last = throttle.get(symbol)
    if last is not None and (now - last).total_seconds() < _SHADOW_THROTTLE_MIN * 60:
        return False
    throttle[symbol] = now
    return True


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
            atr = _floored_atr(getattr(market_data, 'indicators', {}).get('atr', market_data.close * 0.02), market_data.close)
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

        # v-profitability-pass-2026-04-20: disabled by default on 5-min
        # bars. Backtest PF 0.79-0.83 after strict gates. Flip
        # ENABLE_BREAKOUT_LONG True in Config.yaml to reactivate.
        # Short branch still runs (gated separately by ENABLE_SHORT_MIRRORS).
        from core.config import Config as _CfgGate
        _cfg_gate = _CfgGate()
        _long_enabled = _cfg_gate.ENABLE_BREAKOUT_LONG
        _short_enabled = _cfg_gate.ENABLE_SHORT_MIRRORS
        if not _long_enabled and not _short_enabled:
            return None

        try:
            # ---- Real resistance: 20-bar high (not single-bar pivot) ----
            # A breakout means price exceeds the highest point of the last
            # 20 bars — a genuine new high, not single-bar pivot noise.
            high_20 = float(indicators.get('high_20', 0))
            volume_ratio = float(indicators.get('volume_ratio', 1))
            adx = float(indicators.get('adx', 0))
            sma_20 = float(indicators.get('sma_20', 0))

            if high_20 <= 0 or np.isnan(high_20):
                self._log_decision(market_data, "skip", "invalid_indicators",
                                   high_20=high_20)
                return None

            # v-short-mirrors-2026-04-20: added breakdown short path below.
            # Existing long path (Gates 1-4 + BUY signal) is preserved
            # bit-for-bit inside this `if market_data.close > high_20:` block.
            # Any state where close <= high_20 now falls through to the
            # breakdown check instead of an unconditional `return None`.
            # v-profitability-pass-2026-04-20: thresholds come from config
            # so we can A/B test stricter gates. Defaults to stricter values
            # (1.003x break, ADX 25, volume 2.0x) based on 60-day backtest
            # showing loose gates produced PF 0.83.
            from core.config import Config as _CfgBO
            _strict_bo = _CfgBO().ENABLE_STRICT_LONG_GATES
            _break_mult = 1.003 if _strict_bo else 1.0
            _adx_min = 25.0 if _strict_bo else 20.0
            _vol_min = 2.0 if _strict_bo else 1.5

            if _long_enabled and market_data.close > high_20 * _break_mult:
                # ---- LONG: breakout above 20-bar high -----------------------

                # Gate 2: Trend confirmation — ADX
                if adx < _adx_min:
                    self._log_decision(market_data, "skip", "weak_trend",
                                       adx=round(adx, 1), high_20=round(high_20, 2),
                                       adx_min=_adx_min)
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=market_data.symbol,
                        title=f"⛔ Breakout Skipped — Weak Trend",
                        message=(f"Price broke 20-bar high ${high_20:.2f} but ADX "
                                 f"only {adx:.1f} (<{_adx_min:.0f}). Breakouts in chop fail."),
                        importance=6,
                    ))
                    return None

                # Gate 3: Price must be above SMA20 (uptrend context)
                if sma_20 > 0 and market_data.close < sma_20:
                    self._log_decision(market_data, "skip", "below_sma20",
                                       close=round(market_data.close, 2),
                                       sma_20=round(sma_20, 2))
                    return None

                # Gate 4: Volume confirmation — volume_ratio threshold
                if volume_ratio < _vol_min:
                    self._log_decision(market_data, "skip", "low_volume",
                                       high_20=round(high_20, 2),
                                       volume_ratio=round(volume_ratio, 2),
                                       vol_min=_vol_min)
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=market_data.symbol,
                        title=f"⛔ Breakout Skipped — Low Volume",
                        message=(f"Price broke 20-bar high ${high_20:.2f} but volume "
                                 f"only {volume_ratio:.1f}x average. Real breakouts "
                                 f"need at least {_vol_min:.1f}x."),
                        importance=6,
                    ))
                    return None

                # v-direction-gate-breakout-2026-05-29: late breakouts
                # systematically underperform. The 20-bar-high break +
                # ADX + volume gates verify "right now this is a clear
                # break"; the direction reader verifies "this break is
                # not the climax of an already-extended trend." A break
                # firing on a stock already 10% above SMA50 with RSI 75+
                # is the textbook failed breakout setup (exhaustion).
                #
                # Require direction >= +3 (strong uptrend) AND phase
                # in {early, middle} (not late or exhausted). This is
                # stricter than mean-rev's gate because breakout
                # explicitly commits to "buy strength" — the strength
                # better be confirmed.
                #
                # Gated by Config.ENABLE_DIRECTION_GATE_BREAKOUT
                # (default True).
                try:
                    from core.config import Config as _CfgDirBO
                    if _CfgDirBO().ENABLE_DIRECTION_GATE_BREAKOUT:
                        from core.direction_reader import read_direction
                        _dr = read_direction(market_data.close, indicators)
                        if not (_dr.direction >= 3.0 and _dr.phase in ("early", "middle")):
                            self._log_decision(
                                market_data, "skip",
                                "direction_gate_rejected_breakout",
                                direction=_dr.direction,
                                phase=_dr.phase,
                                ema_stack=_dr.ema_stack,
                                adx=round(adx, 1),
                                high_20=round(high_20, 2),
                                reason_text=_dr.reason,
                            )
                            self.commentary.add_commentary(TradingCommentary(
                                timestamp=datetime.now(),
                                type=CommentaryType.RISK_ASSESSMENT,
                                symbol=market_data.symbol,
                                title=f"⛔ Breakout Skipped — Direction/Phase Gate",
                                message=(
                                    f"20-bar high break confirmed but direction "
                                    f"reader says {_dr.reason} (phase: {_dr.phase}, "
                                    f"score {_dr.direction:+.1f}). Need direction>=3 "
                                    f"and phase early/middle — late breakouts fail."
                                ),
                                importance=6,
                            ))
                            return None
                except Exception:
                    pass

                # v-market-context-gate-breakout-2026-06-08: even a
                # technically-clean breakout can fail when the broader
                # market is panic-selling. Apply the same regime gate
                # as mean-rev. Conviction multiplier flows through
                # risk_manager via signal.reasoning.
                _mc_conviction_bo = 1.0
                _mc_regime_bo = None
                _mc_sector_bo = None
                try:
                    from core.config import Config as _CfgMCBO
                    cfg_mc_bo = _CfgMCBO()
                    if cfg_mc_bo.ENABLE_MARKET_CONTEXT_GATE or cfg_mc_bo.ENABLE_MARKET_CONTEXT_SIZING:
                        from core.market_context import read_market_context
                        _mc_ctx_bo = read_market_context(market_data.symbol)
                        if cfg_mc_bo.ENABLE_MARKET_CONTEXT_GATE and not _mc_ctx_bo.allows_long:
                            self._log_decision(
                                market_data, "skip", "market_context_blocks_breakout",
                                regime=_mc_ctx_bo.regime,
                                spy_change_pct=_mc_ctx_bo.spy_change_pct,
                                sector_etf=_mc_ctx_bo.sector_etf,
                                sector_strength_pct=_mc_ctx_bo.sector_strength_pct,
                                reason_text=_mc_ctx_bo.reason,
                            )
                            return None
                        _mc_conviction_bo = _mc_ctx_bo.conviction_multiplier
                        _mc_regime_bo = _mc_ctx_bo.regime
                        _mc_sector_bo = _mc_ctx_bo.sector_etf
                except Exception:
                    pass

                # All gates passed — genuine breakout
                breakout_distance_pct = ((market_data.close - high_20) / high_20) * 100
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.OPPORTUNITY,
                    symbol=market_data.symbol,
                    title=f"🚀 Breakout: New 20-Bar High!",
                    message=(f"Price ${market_data.close:.2f} broke above 20-bar "
                             f"resistance ${high_20:.2f} (+{breakout_distance_pct:.1f}%). "
                             f"ADX {adx:.0f} confirms trend. Volume {volume_ratio:.1f}x."),
                    data={
                        'breakout_level': high_20,
                        'current_price': market_data.close,
                        'adx': adx,
                        'volume_ratio': volume_ratio,
                        'distance_pct': breakout_distance_pct,
                    },
                    confidence=0.8,
                    importance=8,
                ))

                # ATR-scaled stops
                from core.config import Config
                atr = _floored_atr(indicators.get('atr', market_data.close * 0.02), market_data.close)
                atr_mult = Config().ATR_STOP_MULTIPLIER
                rr_ratio = Config().ATR_REWARD_RISK_RATIO
                stop_distance = atr_mult * atr
                stop_loss = market_data.close - stop_distance

                # v-smart-target-breakout-2026-06-02: pick a reachable
                # take_profit. For breakout, the entry is above
                # high_20 so smart_target's high_20_projection
                # candidate fires (5% above the broken level).
                _bo_smart_target = None
                try:
                    if Config().ENABLE_SMART_TAKE_PROFIT:
                        from core.smart_target import compute_smart_target
                        _st_bo = compute_smart_target(
                            entry=market_data.close,
                            stop_distance=stop_distance,
                            indicators=indicators,
                            rr_ratio=rr_ratio,
                        )
                        if _st_bo is None:
                            self._log_decision(
                                market_data, "skip", "no_reachable_target",
                                strategy="breakout",
                                stop_dist=round(stop_distance, 2),
                            )
                            return None
                        _bo_smart_target = _st_bo.target
                        self._log_decision(
                            market_data, "smart_target_picked",
                            "smart_take_profit_breakout",
                            source=_st_bo.source,
                            target=round(_st_bo.target, 4),
                            R=_st_bo.R,
                        )
                except Exception:
                    _bo_smart_target = None
                take_profit = (
                    _bo_smart_target if _bo_smart_target is not None
                    else market_data.close + (rr_ratio * stop_distance)
                )

                self._log_decision(market_data, "signal_buy", "breakout_new_high",
                                   high_20=round(high_20, 2),
                                   volume_ratio=round(volume_ratio, 2),
                                   adx=round(adx, 1),
                                   stop=round(stop_loss, 2), target=round(take_profit, 2),
                                   atr=round(atr, 3), stop_dist=round(stop_distance, 2))
                return TradingSignal(
                    symbol=market_data.symbol,
                    signal_type=SignalType.BUY,
                    strength=min(0.6 + (adx / 100), 0.95),  # stronger ADX → higher strength
                    entry_price=market_data.close,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    position_size=0,
                    reasoning={
                        'strategy': 'breakout',
                        'breakout_level': high_20,
                        'adx': adx,
                        'volume_ratio': volume_ratio,
                        'atr': atr,
                        'atr_mult': atr_mult,
                        'stop_distance': stop_distance,
                        # v-market-context-2026-06-08
                        'market_context_conviction': _mc_conviction_bo,
                        'market_context_regime': _mc_regime_bo,
                        'market_context_sector': _mc_sector_bo,
                    },
                    confidence=min(0.6 + (adx / 200) + (volume_ratio - 1.5) * 0.1, 0.9),
                )

            # ---- SHORT: breakdown below 20-bar low ----------------------
            # Mirror of the long gates, activated when whole watchlist is
            # red and no long trigger can fire. Same gate logic for symmetry:
            #   Gate 1: close < low_20 (new 20-bar low)
            #   Gate 2: ADX > 20 (trending, not chop)
            #   Gate 3: close < SMA20 (downtrend context)
            #   Gate 4: volume_ratio > 1.5 (real selling pressure)
            # Gated behind ENABLE_SHORT_MIRRORS config flag (default False) —
            # 2026-04-20 backtest showed PF 0.69, symmetric shorts fight
            # the oversold bounce. Code kept for config-based paper testing.
            from core.config import Config as _Cfg
            if not _Cfg().ENABLE_SHORT_MIRRORS:
                self._log_decision(market_data, "skip", "no_breakout",
                                   close=round(market_data.close, 2),
                                   high_20=round(high_20, 2),
                                   volume_ratio=round(volume_ratio, 2))
                return None
            low_20 = float(indicators.get('low_20', 0))
            if low_20 <= 0 or np.isnan(low_20):
                self._log_decision(market_data, "skip", "no_breakout",
                                   close=round(market_data.close, 2),
                                   high_20=round(high_20, 2),
                                   low_20=round(low_20, 2),
                                   volume_ratio=round(volume_ratio, 2))
                return None

            if market_data.close >= low_20:
                self._log_decision(market_data, "skip", "no_breakout",
                                   close=round(market_data.close, 2),
                                   high_20=round(high_20, 2),
                                   low_20=round(low_20, 2),
                                   volume_ratio=round(volume_ratio, 2))
                return None

            if adx < 20:
                self._log_decision(market_data, "skip", "weak_trend_short",
                                   adx=round(adx, 1), low_20=round(low_20, 2))
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.RISK_ASSESSMENT,
                    symbol=market_data.symbol,
                    title=f"⛔ Breakdown Skipped — Weak Trend",
                    message=(f"Price broke 20-bar low ${low_20:.2f} but ADX "
                             f"only {adx:.1f} (<20). Breakdowns in chop fail."),
                    importance=6,
                ))
                return None

            if sma_20 > 0 and market_data.close > sma_20:
                self._log_decision(market_data, "skip", "above_sma20_short",
                                   close=round(market_data.close, 2),
                                   sma_20=round(sma_20, 2))
                return None

            if volume_ratio < 1.5:
                self._log_decision(market_data, "skip", "low_volume_short",
                                   low_20=round(low_20, 2),
                                   volume_ratio=round(volume_ratio, 2))
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.RISK_ASSESSMENT,
                    symbol=market_data.symbol,
                    title=f"⛔ Breakdown Skipped — Low Volume",
                    message=(f"Price broke 20-bar low ${low_20:.2f} but volume "
                             f"only {volume_ratio:.1f}x average. Real breakdowns "
                             "need at least 1.5x."),
                    importance=6,
                ))
                return None

            breakdown_distance_pct = ((low_20 - market_data.close) / low_20) * 100
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.OPPORTUNITY,
                symbol=market_data.symbol,
                title=f"🔻 Breakdown: New 20-Bar Low",
                message=(f"Price ${market_data.close:.2f} broke below 20-bar "
                         f"support ${low_20:.2f} (-{breakdown_distance_pct:.1f}%). "
                         f"ADX {adx:.0f} confirms trend. Volume {volume_ratio:.1f}x."),
                data={
                    'breakdown_level': low_20,
                    'current_price': market_data.close,
                    'adx': adx,
                    'volume_ratio': volume_ratio,
                    'distance_pct': breakdown_distance_pct,
                },
                confidence=0.8,
                importance=8,
            ))

            from core.config import Config
            atr_s = _floored_atr(indicators.get('atr', market_data.close * 0.02), market_data.close)
            atr_mult_s = Config().ATR_STOP_MULTIPLIER
            rr_ratio_s = Config().ATR_REWARD_RISK_RATIO
            stop_distance_s = atr_mult_s * atr_s
            stop_loss_s = market_data.close + stop_distance_s
            take_profit_s = market_data.close - (rr_ratio_s * stop_distance_s)

            self._log_decision(market_data, "signal_sell", "breakdown_new_low",
                               low_20=round(low_20, 2),
                               volume_ratio=round(volume_ratio, 2),
                               adx=round(adx, 1),
                               stop=round(stop_loss_s, 2), target=round(take_profit_s, 2),
                               atr=round(atr_s, 3), stop_dist=round(stop_distance_s, 2))
            return TradingSignal(
                symbol=market_data.symbol,
                signal_type=SignalType.SELL,
                strength=min(0.6 + (adx / 100), 0.95),
                entry_price=market_data.close,
                stop_loss=stop_loss_s,
                take_profit=take_profit_s,
                position_size=0,
                reasoning={
                    'strategy': 'breakdown',
                    'breakdown_level': low_20,
                    'adx': adx,
                    'volume_ratio': volume_ratio,
                    'atr': atr_s,
                    'atr_mult': atr_mult_s,
                    'stop_distance': stop_distance_s,
                },
                confidence=min(0.6 + (adx / 200) + (volume_ratio - 1.5) * 0.1, 0.9),
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
            # v-mean-rev-uptrend-pullback-2026-05-28: alternative entry
            # path. Operator complaint 2026-05-28 mid-morning: bot has
            # been silent for 3 sessions while specific stocks rise on
            # momentum. Mean-rev currently fires only on oversold
            # extremes (RSI<30 + below BB lower) — never catches the
            # "stock in uptrend pulls back to support" pattern that's
            # been the regime this week.
            #
            # New trigger: RSI in 30-45 (mildly soft, not extreme) AND
            # close >= SMA50 (uptrend confirmation) AND close within 2%
            # of BB lower (still pulling back to support). The existing
            # falling-knife gate doesn't trigger for this path since
            # it requires close<SMA50. The existing price-direction
            # gate (green bar + vol>=1.5x + rejection of lows) DOES
            # still run — confirms a bounce at the bar level before
            # entry. Same ATR-based stop/target as the classic path.
            #
            # Gated by Config.ENABLE_MEAN_REV_UPTREND_PULLBACK (default
            # True). Toggle False via API if it proves too noisy.
            # Audit grep: `engine_decision .* signal_buy .* uptrend_pullback`.
            from core.config import Config as _CfgUP
            _classic_oversold = rsi < self.rsi_threshold and market_data.close < bb_lower
            _sma_50_for_uptrend = float(indicators.get('sma_50', 0))
            _bb_lower_dist_pct = (
                (market_data.close - bb_lower) / bb_lower * 100
                if bb_lower > 0 else 999.0
            )
            # v-mean-rev-uptrend-pullback-widen-2026-05-28: aggressive
            # widening after the initial deploy produced 0 fires in 90
            # minutes of RTH evaluation. The original RSI 30-45 +
            # within 2% of BB lower band was too narrow for today's
            # regime — market is showing momentum (RSI 45-70 widely),
            # not classic pullbacks. Widened to:
            #   RSI 30-55  (was 30-45) — catches mid-pullbacks
            #   close vs BB lower in (-1%, +8%) (was (-1%, +2%)) —
            #     captures pullbacks that paused above support
            # Safety remains intact: still requires close >= SMA50
            # (real uptrend) so falling-knife pattern stays excluded.
            _uptrend_pullback = (
                _CfgUP().ENABLE_MEAN_REV_UPTREND_PULLBACK
                and 30 <= rsi <= 55
                and _sma_50_for_uptrend > 0
                and market_data.close >= _sma_50_for_uptrend
                and bb_lower > 0
                and -1.0 <= _bb_lower_dist_pct <= 8.0
            )
            _entry_pattern = (
                "oversold_bounce" if _classic_oversold
                else ("uptrend_pullback" if _uptrend_pullback else None)
            )

            # v-direction-gate-mean-rev-uptrend-pullback-2026-05-29: the
            # uptrend_pullback path explicitly claims "in uptrend." If
            # the direction reader says we're NOT in an uptrend, the
            # path should not fire — the SMA50 check is a snapshot,
            # the direction reader is a multi-feature read over time.
            # Classic oversold_bounce path is NOT gated because its
            # edge is buying bounces in downtrends; the falling-knife
            # filter already handles direction risk there.
            #
            # Gated by Config.ENABLE_DIRECTION_GATE_MEAN_REV (default
            # True). Toggle False via API if the reader produces too
            # many false rejections in practice.
            if _entry_pattern == "uptrend_pullback":
                try:
                    from core.config import Config as _CfgDir
                    if _CfgDir().ENABLE_DIRECTION_GATE_MEAN_REV:
                        from core.direction_reader import read_direction
                        _dr = read_direction(market_data.close, indicators)
                        # Require mild bullish direction (>= +2) AND
                        # phase not exhausted. allows_long_entry alone
                        # would pass mild ranges (direction 0..2) but
                        # for the explicit uptrend_pullback path we
                        # want a CLEAR uptrend reading.
                        if not (_dr.direction >= 2.0 and _dr.phase != "exhausted"):
                            self._log_decision(
                                market_data, "skip",
                                "direction_gate_rejected_uptrend_pullback",
                                direction=_dr.direction,
                                phase=_dr.phase,
                                ema_stack=_dr.ema_stack,
                                rsi=round(rsi, 2),
                                reason_text=_dr.reason,
                            )
                            return None
                except Exception:
                    # Direction reader failure should not block trading.
                    # The strategy's existing gates still run.
                    pass

            # v-market-context-gate-mean-rev-2026-06-08: pro-level
            # context check. Mean-rev long entries should respect
            # broader market regime + sector tailwind. If risk_off,
            # don't catch a "pullback" that's actually capitulation.
            # The conviction multiplier flows through risk_manager
            # (read from signal.reasoning).
            _mc_conviction = 1.0
            _mc_regime = None
            _mc_sector = None
            try:
                from core.config import Config as _CfgMC
                cfg_mc = _CfgMC()
                if cfg_mc.ENABLE_MARKET_CONTEXT_GATE or cfg_mc.ENABLE_MARKET_CONTEXT_SIZING:
                    from core.market_context import read_market_context
                    _mc_ctx = read_market_context(market_data.symbol)
                    if cfg_mc.ENABLE_MARKET_CONTEXT_GATE and not _mc_ctx.allows_long:
                        self._log_decision(
                            market_data, "skip", "market_context_blocks_long",
                            regime=_mc_ctx.regime,
                            spy_change_pct=_mc_ctx.spy_change_pct,
                            sector_etf=_mc_ctx.sector_etf,
                            sector_strength_pct=_mc_ctx.sector_strength_pct,
                            vix_change_pct=_mc_ctx.vix_change_pct,
                            reason_text=_mc_ctx.reason,
                        )
                        return None
                    _mc_conviction = _mc_ctx.conviction_multiplier
                    _mc_regime = _mc_ctx.regime
                    _mc_sector = _mc_ctx.sector_etf
            except Exception:
                pass  # fail-open

            if _entry_pattern is not None:
                # Trend filter: don't catch a falling knife.
                # Skip the long when price is below MA50 AND momentum is bearish.
                # This is the LCID 2026-04-14 setup: oversold inside a downtrend
                # rarely mean-reverts cleanly; it usually keeps falling.
                #
                # v-profitability-pass-2026-04-20: the old filter was too strict
                # — backtest showed only 57 trades in 60 days × 20 symbols at
                # PF 2.43. The setup IS profitable; we just need more of them.
                # New override (gated by ENABLE_RELAXED_MEAN_REV_LONG, default
                # True): if the oversold bar is already green (close >= open),
                # the reversal is starting — take the trade even inside the
                # downtrend. Still skip when the bar is red (knife still falling).
                from core.config import Config as _CfgMR
                sma_50 = float(indicators.get('sma_50', 0))
                macd_val = float(indicators.get('macd', 0))
                macd_signal_val = float(indicators.get('macd_signal', 0))
                _knife_still_falling = (
                    sma_50 > 0 and market_data.close < sma_50
                    and macd_val < macd_signal_val
                )
                if _CfgMR().ENABLE_RELAXED_MEAN_REV_LONG:
                    # Relaxed: require a red bar (close<open) to confirm
                    # the knife is still falling at THIS bar, not just
                    # that we're inside a downtrend.
                    _knife_still_falling = (
                        _knife_still_falling
                        and market_data.close <= market_data.open
                    )
                if _knife_still_falling:
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

                # v-mean-rev-price-direction-gate-2026-05-13: require the
                # ENTRY bar itself to confirm reversal — close > open AND
                # volume_ratio >= 1.0. Operator incident 2026-05-12: ORCL
                # entered at 12:31 after a 46-minute slide (184.50 → 182.44),
                # CRWV at 12:47 after a 63-minute slide. Both had RSI < 30
                # and close < BB_lower (the setup), AND the existing
                # falling-knife guard was bypassed because the entry bar
                # itself happened to close green. But there was no positive
                # confirmation that the slide had ended — the bar was
                # simply a brief uptick inside continued weakness.
                #
                # Mirror of v-news-price-direction-gate-2026-05-08, which
                # the news strategy uses to refuse low-conviction buys.
                # Mean-rev LONG by definition is a contrarian entry; the
                # entry bar's price-action confirmation is the difference
                # between "buying the bottom" and "catching the knife".
                # v-bounce-confirmation-2026-05-20: strengthened the
                # price-direction gate after the 5/20 FIG -$247 incident.
                # The OLD check only required `close > bar_open AND vol_ratio >= 1.0`
                # — which can pass on a marginally green bar inside a continuing
                # slide. FIG met the old check (5-min bar happened to be green
                # at the open tick) but stopped out in 3 minutes as the slide
                # resumed. We now require:
                #   1. Current bar green (close > open)
                #   2. Volume conviction (vol_ratio >= 1.5, was 1.0)
                #   3. Visible rejection of the low: lower_wick > body
                #      (a "hammer-like" pattern — seller exhaustion).
                # All three must be true. This is the "wait for the sign of
                # bouncing" rule, not "RSI is low so jump in".
                _bar_open = float(getattr(market_data, "open", market_data.close) or market_data.close)
                _bar_low = float(getattr(market_data, "low", market_data.close) or market_data.close)
                _vol_ratio_now = float(indicators.get("volume_ratio", 1.0) or 1.0)
                _lower_wick = min(_bar_open, market_data.close) - _bar_low
                _body = abs(market_data.close - _bar_open)
                _is_green = market_data.close > _bar_open
                _has_volume = _vol_ratio_now >= 1.5
                _has_rejection = _lower_wick > _body  # lower wick > body = rejection of lows

                # v-uptrend-pullback-gate-relax-2026-05-28: relax the
                # bar-level confirmation for the uptrend_pullback path.
                # For classic falling-knife oversold (rsi<30 + below BB),
                # the hammer-pattern bar (green + vol>=1.5x + lower_wick
                # > body) is the difference between "buying the bottom"
                # and "catching the knife." For uptrend_pullback, the
                # close>=SMA50 requirement IS the safety net — the
                # stock is already in a defined uptrend. Requiring a
                # full hammer ON TOP OF that double-counts caution and
                # makes the path fire ~never in practice (0 fires in
                # 90 min on the original strict gate).
                # New rule: uptrend_pullback only requires GREEN BAR
                # (close > bar_open). Classic oversold keeps all three.
                if _entry_pattern == "uptrend_pullback":
                    _bar_ok = _is_green
                else:
                    _bar_ok = _is_green and _has_volume and _has_rejection
                if not _bar_ok:
                    self._log_decision(
                        market_data, "skip", "mean_rev_price_direction_disagrees",
                        rsi=round(rsi, 2),
                        close=round(market_data.close, 2),
                        bar_open=round(_bar_open, 2),
                        bar_low=round(_bar_low, 2),
                        volume_ratio=round(_vol_ratio_now, 2),
                        lower_wick=round(_lower_wick, 3),
                        body=round(_body, 3),
                        is_green=_is_green,
                        has_volume=_has_volume,
                        has_rejection=_has_rejection,
                    )
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=market_data.symbol,
                        title=f"⛔ Mean Reversion Skipped — No Bar-Level Reversal",
                        message=(
                            f"RSI {rsi:.1f} oversold but the entry bar isn't "
                            f"confirming reversal: close={market_data.close:.2f} "
                            f"vs bar_open={_bar_open:.2f}, vol_ratio={_vol_ratio_now:.2f}. "
                            f"Waiting for a confirming green bar with real volume."
                        ),
                        data={'rsi': rsi, 'close': market_data.close,
                              'bar_open': _bar_open,
                              'volume_ratio': _vol_ratio_now},
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
                atr = _floored_atr(indicators.get('atr', market_data.close * 0.02), market_data.close)
                # v-mean-rev-wider-stop-2026-05-20: use mean-rev-specific
                # multiplier (default 2.5× ATR vs the global 1.5×). Pairs
                # with the bounce-confirmation filter — higher-quality
                # entries deserve room to breathe through intraday noise.
                atr_mult = Config().MEAN_REV_ATR_STOP_MULTIPLIER
                rr_ratio = Config().ATR_REWARD_RISK_RATIO
                stop_distance = atr_mult * atr
                stop_loss = market_data.close - stop_distance
                rr_target = market_data.close + (rr_ratio * stop_distance)

                # v-smart-target-2026-06-02: try to pick a reachable
                # take_profit from price structure (bb_upper, high_20,
                # recent range) instead of blindly setting rr_target.
                # If the nearest meaningful resistance doesn't give
                # at least 1.5R, SKIP the trade — don't place an OCO
                # at an unreachable level. Operator complaint
                # 2026-06-01: only 5/100 historical bot trades hit
                # take_profit because targets were aspirational.
                _smart_tp_target = None
                try:
                    from core.config import Config as _CfgSTP
                    if _CfgSTP().ENABLE_SMART_TAKE_PROFIT:
                        from core.smart_target import compute_smart_target
                        _st = compute_smart_target(
                            entry=market_data.close,
                            stop_distance=stop_distance,
                            indicators=indicators,
                            rr_ratio=rr_ratio,
                        )
                        if _st is None:
                            # No reachable target gives 1.5R — skip.
                            self._log_decision(
                                market_data, "skip", "no_reachable_target",
                                pattern=_entry_pattern,
                                rsi=round(rsi, 2),
                                rr_ratio=rr_ratio,
                                stop_dist=round(stop_distance, 2),
                            )
                            return None
                        _smart_tp_target = _st.target
                        self._log_decision(
                            market_data, "smart_target_picked",
                            "smart_take_profit",
                            pattern=_entry_pattern,
                            source=_st.source,
                            target=round(_st.target, 4),
                            R=_st.R,
                        )
                except Exception as _exc:
                    # Smart-target failure must not block trades.
                    # Fall through to the legacy rr_target.
                    _smart_tp_target = None

                # v-mean-rev-target-uncap-2026-05-28: drop the
                # min(bb_middle * take_profit_mult, rr_target) cap.
                # Operator reported 2026-05-28 OCO targets sitting only
                # $2 from entry on real fills today. Inspection of the
                # log lines confirmed: KEEL target dist $0.125 vs stop
                # dist $0.135 (R:R 0.93), SNAP R:R 0.72, MRVL 1.20,
                # RDW 1.37 — all far below the configured 2.0 R:R.
                # Root cause: when bb_middle sits just above entry
                # (common on a shallow pullback or a stock that's
                # already mean-reverted partway), `min(bb_middle,
                # rr_target)` collapsed to bb_middle and capped the
                # winner before it could run.
                #
                # New rule: take_profit is the ATR-based rr_target,
                # unconditionally. Preserves the configured 2.0 R:R.
                # Earlier "revert to mean" logic was correct in theory
                # but in practice killed winners — the trailing stop
                # in the exit manager handles "revert to mean" exits
                # better than a hard cap on take_profit ever did.
                # v-smart-target-2026-06-02: when smart-target is on
                # and found a destination, use it; otherwise fall
                # through to the rr_target fallback.
                take_profit = _smart_tp_target if _smart_tp_target is not None else rr_target
                self._log_decision(market_data, "signal_buy", _entry_pattern,
                                   rsi=round(rsi, 2), distance_pct=round(distance_from_mean, 2),
                                   stop=round(stop_loss, 2), target=round(take_profit, 2),
                                   atr=round(atr, 3), stop_dist=round(stop_distance, 2),
                                   bb_lower_dist_pct=round(_bb_lower_dist_pct, 2))
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
                        # v-market-context-2026-06-08: include
                        # context so risk_manager can size by
                        # conviction and the trade record carries
                        # the regime for post-trade analysis.
                        'market_context_conviction': _mc_conviction,
                        'market_context_regime': _mc_regime,
                        'market_context_sector': _mc_sector,
                    },
                    confidence=0.65
                )
            elif rsi > 70 and market_data.close > bb_upper:
                # v-disable-shorts-2026-04-22: mean-reversion SHORT gated
                # behind ENABLE_MEAN_REV_SHORT (default False). Live 5/5
                # trades today: PF 0.25, realized -$115. Missing rising-peak
                # filter equivalent to the long side's falling-knife filter.
                # Existing open shorts keep running — this only blocks NEW
                # entries. Re-enable via trading.enable_mean_rev_short=true.
                from core.config import Config as _CfgMS
                _cfg_ms = _CfgMS()
                if not _cfg_ms.ENABLE_MEAN_REV_SHORT:
                    # v-mean-rev-short-shadow-2026-06-09: before returning
                    # None, log the would-be signal to shadow ledger if
                    # shadow mode is on. Pure observation — no order
                    # placed. Computes full hypothetical OCO bracket
                    # (stop/target) using the same ATR-based logic the
                    # live SHORT branch uses below, so the shadow data
                    # is comparable to a real signal had it fired.
                    if (_cfg_ms.ENABLE_MEAN_REV_SHORT_SHADOW
                            and _shadow_short_should_log(market_data.symbol)):
                        try:
                            from core.config import Config as _CfgShadow
                            _cfg_sh = _CfgShadow()
                            _atr_sh = _floored_atr(
                                indicators.get('atr', market_data.close * 0.02),
                                market_data.close
                            )
                            _atr_mult_sh = _cfg_sh.MEAN_REV_ATR_STOP_MULTIPLIER
                            _rr_ratio_sh = _cfg_sh.ATR_REWARD_RISK_RATIO
                            _stop_dist_sh = _atr_mult_sh * _atr_sh
                            _hyp_stop = market_data.close + _stop_dist_sh
                            _hyp_target = market_data.close - (_rr_ratio_sh * _stop_dist_sh)
                            from datetime import timezone as _tz_sh
                            _shadow_entry = {
                                'timestamp': datetime.now(_tz_sh.utc).isoformat(),
                                'symbol': market_data.symbol,
                                'signal_type': 'SHORT',
                                'reason': 'overbought_fade',
                                'signal_close': float(market_data.close),
                                'rsi': float(rsi),
                                'bb_upper': float(bb_upper),
                                'bb_middle': float(bb_middle),
                                'sma_50': float(indicators.get('sma_50', 0) or 0),
                                'macd': float(indicators.get('macd', 0) or 0),
                                'macd_signal': float(indicators.get('macd_signal', 0) or 0),
                                'atr': float(_atr_sh),
                                'hypothetical_stop': float(_hyp_stop),
                                'hypothetical_target': float(_hyp_target),
                                'rr_ratio': float(_rr_ratio_sh),
                                'stop_dist': float(_stop_dist_sh),
                            }
                            import json as _json_sh
                            # NDJSON append-only (one JSON object per
                            # line). The original read-modify-rewrite
                            # had two failure modes: lost updates when
                            # two captures interleave, and a zero-byte
                            # file if the process dies between truncate
                            # and dump (the bot WAS hard-killed today).
                            # O_APPEND writes of < 4 KiB are atomic on
                            # Linux, so concurrent captures can't tear.
                            _ledger_path = 'shadow_short_log.ndjson'
                            try:
                                with open(_ledger_path, 'a') as _f:
                                    _f.write(
                                        _json_sh.dumps(
                                            _shadow_entry, default=str
                                        ) + "\n"
                                    )
                            except Exception as _shadow_io_exc:
                                logger.warning(
                                    "shadow_short_log write failed for %s: %s",
                                    market_data.symbol, _shadow_io_exc,
                                )
                            # Emit audit line so the shadow event is
                            # also visible in the structured log.
                            self._log_decision(
                                market_data, "shadow", "short_shadow_logged",
                                rsi=round(rsi, 2),
                                close=round(market_data.close, 2),
                                bb_upper=round(bb_upper, 2),
                                hyp_stop=round(_hyp_stop, 2),
                                hyp_target=round(_hyp_target, 2),
                                sma_50=round(float(indicators.get('sma_50', 0) or 0), 2),
                                macd=round(float(indicators.get('macd', 0) or 0), 4),
                            )
                        except Exception as _shadow_exc:
                            logger.debug(
                                "short_shadow capture failed for %s: %s",
                                market_data.symbol, _shadow_exc,
                            )
                    self._log_decision(market_data, "skip", "short_disabled",
                                       rsi=round(rsi, 2),
                                       close=round(market_data.close, 2),
                                       bb_upper=round(bb_upper, 2))
                    return None

                # v-rising-peak-filter-2026-06-08: symmetric to the
                # falling-knife filter on the LONG side. Mean-rev SHORT
                # must NOT fire when the broader trend is up — RSI>70
                # is the *norm* during a sustained rally, not a reversal
                # signal. May 11 2026 incident: re-enabling SHORT without
                # this filter produced 5 simultaneous shorts at -$665
                # unrealized inside one session because every rally peak
                # read as "overbought" and the strategy piled in.
                #
                # Gates (BOTH required when the filter is on):
                #   * close < SMA50          (price below the 50-period
                #                             MA — established downtrend)
                #   * MACD < MACD signal     (bearish momentum confirmed)
                #
                # Indicator availability: sma_50 / macd / macd_signal are
                # computed by analysis/technical.py for every bar with
                # sufficient history. sma_50 == 0 means insufficient bars
                # or a freshly-listed name → fail-closed with audit reason
                # 'rising_peak_no_data'. Block path uses audit reason
                # 'rising_peak_uptrend' for grep-discoverability.
                #
                # Gated by Config.ENABLE_RISING_PEAK_FILTER (default
                # True). Flip False ONLY for backtest comparison.
                from core.config import Config as _CfgRP
                if _CfgRP().ENABLE_RISING_PEAK_FILTER:
                    _sma_50_rp = float(indicators.get('sma_50', 0))
                    _macd_rp = float(indicators.get('macd', 0))
                    _macd_signal_rp = float(indicators.get('macd_signal', 0))
                    if _sma_50_rp <= 0:
                        self._log_decision(
                            market_data, "skip", "rising_peak_no_data",
                            rsi=round(rsi, 2),
                            close=round(market_data.close, 2),
                            sma_50=round(_sma_50_rp, 2),
                        )
                        return None
                    _in_downtrend = market_data.close < _sma_50_rp
                    _bearish_momentum = _macd_rp < _macd_signal_rp
                    # v-rising-peak-filter-loosen-2026-06-08-evening:
                    # 60d AND 360d backtests showed `_in_downtrend AND
                    # _bearish_momentum` rejected 100% of the 1,176-trade
                    # (60d) and 4,965-trade (360d) candidate pool — too
                    # strict to ever fire. Loosened to require ONLY
                    # `_in_downtrend` (close < SMA50). Rationale: the
                    # May 11 incident was 5 simultaneous shorts on a
                    # confirmed-uptrend tape — every name had close >
                    # SMA50. The MACD-bearish check was redundant for
                    # that failure mode (it would have failed too).
                    # close<SMA50 alone is the necessary-and-sufficient
                    # structural signal. _bearish_momentum is kept in
                    # the audit log for diagnostics on rare borderline
                    # cases (price below SMA50 but MACD just turned up).
                    if not _in_downtrend:
                        self._log_decision(
                            market_data, "skip", "rising_peak_uptrend",
                            rsi=round(rsi, 2),
                            close=round(market_data.close, 2),
                            sma_50=round(_sma_50_rp, 2),
                            in_downtrend=_in_downtrend,
                            macd=round(_macd_rp, 4),
                            macd_signal=round(_macd_signal_rp, 4),
                            bearish_momentum=_bearish_momentum,
                        )
                        return None

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
                atr = _floored_atr(indicators.get('atr', market_data.close * 0.02), market_data.close)
                # v-mean-rev-wider-stop-2026-05-20: mean-rev-specific multiplier
                # (mirror of the long-side change above).
                atr_mult = Config().MEAN_REV_ATR_STOP_MULTIPLIER
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

        # v-profitability-pass-2026-04-20: disabled by default on 5-min
        # bars. Backtest PF 0.84-0.85 even with close>SMA50 + RSI 55-65
        # tighter gates. Flip ENABLE_MOMENTUM_LONG True to reactivate.
        # Short branch still runs (gated separately by ENABLE_SHORT_MIRRORS).
        from core.config import Config as _CfgMGate
        _cfg_mgate = _CfgMGate()
        _mom_long_enabled = _cfg_mgate.ENABLE_MOMENTUM_LONG
        _mom_short_enabled = _cfg_mgate.ENABLE_SHORT_MIRRORS
        if not _mom_long_enabled and not _mom_short_enabled:
            return None

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

            # v-profitability-pass-2026-04-20: tighter RSI window (55-65 vs
            # 50-70) plus close>SMA50 trend-confirmation filter, defaulting
            # to strict via ENABLE_STRICT_LONG_GATES. Loose gates produced
            # PF 0.85 over 60 days — too many entries against the larger
            # trend. close>SMA50 is symmetric to the short-mirror gate
            # (close<SMA50) we added for bearish momentum.
            from core.config import Config as _CfgMo
            _strict_mo = _CfgMo().ENABLE_STRICT_LONG_GATES
            _rsi_min, _rsi_max = (55.0, 65.0) if _strict_mo else (50.0, 70.0)
            sma_50_long = float(indicators.get('sma_50', 0))
            _trend_ok = (not _strict_mo) or (sma_50_long > 0 and market_data.close > sma_50_long)

            if (_mom_long_enabled
                    and macd > macd_signal and _rsi_min < rsi < _rsi_max and adx > 25
                    and _trend_ok):

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
                atr = _floored_atr(indicators.get('atr', market_data.close * 0.005), market_data.close)
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

            # v-short-mirrors-2026-04-20: bearish-momentum mirror of the
            # long branch above. Conditions mirror long exactly:
            #   BUY : macd>signal, 50<rsi<70, adx>25   (bullish crossover)
            #   SELL: macd<signal, 30<rsi<50, adx>25   (bearish crossover)
            # Extra gate: close < sma_50 to confirm established downtrend
            # (symmetric to the falling-knife filter on the long side).
            # Gated behind ENABLE_SHORT_MIRRORS config flag (default False) —
            # 2026-04-20 backtest showed PF 0.69, same as breakdown short.
            from core.config import Config as _Cfg
            sma_50_m = float(indicators.get('sma_50', 0))
            if (_Cfg().ENABLE_SHORT_MIRRORS
                    and macd < macd_signal and 30 < rsi < 50 and adx > 25
                    and sma_50_m > 0 and market_data.close < sma_50_m):
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.OPPORTUNITY,
                    symbol=market_data.symbol,
                    title=f"📉 Bearish Momentum Building",
                    message=(f"Bearish momentum: MACD bearish crossover, RSI at "
                             f"{rsi:.1f} (weak), ADX at {adx:.1f} (strong trend), "
                             f"price below 50-period MA."),
                    data={
                        'macd_crossover': False,
                        'macd': macd,
                        'macd_signal': macd_signal,
                        'rsi': rsi,
                        'adx': adx,
                        'sma_50': sma_50_m,
                        'trend_strength': 'strong' if adx > 30 else 'moderate',
                    },
                    confidence=0.7,
                    importance=7,
                ))

                atr_m = _floored_atr(indicators.get('atr', market_data.close * 0.005), market_data.close)
                stop_loss_m = market_data.close + 1.5 * atr_m
                take_profit_m = market_data.close - 3.0 * atr_m

                self._log_decision(market_data, "signal_sell", "macd_rsi_adx_bearish",
                                   macd=round(macd, 4), rsi=round(rsi, 2), adx=round(adx, 2),
                                   sma_50=round(sma_50_m, 2), atr=round(atr_m, 3),
                                   stop=round(stop_loss_m, 2), target=round(take_profit_m, 2))
                return TradingSignal(
                    symbol=market_data.symbol,
                    signal_type=SignalType.SELL,
                    strength=0.75,
                    entry_price=market_data.close,
                    stop_loss=stop_loss_m,
                    take_profit=take_profit_m,
                    position_size=0,
                    reasoning={
                        'strategy': 'momentum_short',
                        'macd_bearish': True,
                        'rsi_weak': True,
                        'below_sma50': True,
                        'trend_strong': adx > 25,
                    },
                    confidence=0.7,
                )
        except Exception as e:
            self._log_decision(market_data, "error", "exception", err=str(e))
            logger.debug(f"Momentum strategy error for {market_data.symbol}: {e}")
            return None

        self._log_decision(market_data, "skip", "no_setup",
                           macd=round(macd, 4), rsi=round(rsi, 2), adx=round(adx, 2))
        return None


def _get_session_id() -> str:
    """Get current trading session ID (date in ET timezone).
    
    v-momentum-stage-a-2026-09-10: session_id groups trades for Stage-A
    daily rollup. Format: YYYY-MM-DD in America/New_York timezone.
    """
    try:
        from datetime import timezone
        from zoneinfo import ZoneInfo
        now_utc = datetime.now(timezone.utc)
        et = now_utc.astimezone(ZoneInfo("America/New_York"))
        return et.strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


class DayTradeMomentumStrategy(TradingStrategyWithCommentary):
    """Day-trade momentum strategy for intraday movers.
    
    v-day-trade-momentum-desk-2026-09-10: supervised day-trade momentum
    for Yahoo day_gainers/losers/most-active movers.
    
    Entry logic (like a human scalper/day trader):
      1. Relative strength vs SPY (symbol outperforming SPY on the day)
      2. Volume surge (volume_ratio >= 1.5x)
      3. Pullback-or-breakout confirmation:
         - Pullback: RSI 40-60 after move, price near VWAP/EMA support
         - Breakout: Price above 20-bar high with volume
      4. Defined ATR stop (1.5-2x ATR, tighter than swing)
      5. Defined target (2-3x ATR) with trail option
      6. Time stop: flatten by late_entry / EOD
    
    Market context:
      - risk_off: REDUCE SIZE (0.25x), don't hard-block
      - opening_30: REDUCE SIZE (0.5x), don't hard-block  
      - extreme conditions (SPY <= -1.5% AND VIX spike): HARD BLOCK
    
    News: OPTIONAL confirmation / size bump, NOT required
    
    Stage-A Instrumentation (v-momentum-stage-a-2026-09-10):
      - session_id: trading date in ET (YYYY-MM-DD)
      - risk_off: explicit boolean flag
      - mc_size_mult: market context size multiplier
      - setup_type: entry_pattern (breakout/pullback/continuation)
    """
    
    name = "day_trade_momentum"
    
    async def generate_signal_with_commentary(self, market_data) -> Optional[TradingSignal]:
        """Generate day-trade momentum signal."""
        from core.config import Config as _Cfg
        cfg = _Cfg()
        
        if not cfg.ENABLE_DAY_TRADE_MOMENTUM:
            return None
        
        indicators = market_data.indicators or {}
        symbol = market_data.symbol
        
        try:
            rsi = float(indicators.get('rsi', 50))
            volume_ratio = float(indicators.get('volume_ratio', 1.0))
            adx = float(indicators.get('adx', 0))
            atr = float(indicators.get('atr', market_data.close * 0.02))
            high_20 = float(indicators.get('high_20', 0))
            sma_20 = float(indicators.get('sma_20', 0))
            macd = float(indicators.get('macd', 0))
            macd_signal = float(indicators.get('macd_signal', 0))
            
            if np.isnan(rsi) or np.isnan(volume_ratio) or high_20 <= 0:
                self._log_decision(market_data, "skip", "invalid_indicators",
                                   rsi=rsi, volume_ratio=volume_ratio, high_20=high_20)
                return None
            
            # ────────────────────────────────────────────────────────────────
            # Gate 1: Market Context Check (size-based, not hard-block)
            # ────────────────────────────────────────────────────────────────
            _mc_size_mult = 1.0
            _mc_regime = None
            _mc_time_of_day = None
            _mc_spy_change = 0.0
            _mc_vix_change = 0.0
            _mc_sector = None
            _mc_reason = ""
            
            try:
                from core.market_context import read_market_context
                _mc = read_market_context(symbol)
                _mc_regime = _mc.regime
                _mc_time_of_day = _mc.time_of_day
                _mc_spy_change = _mc.spy_change_pct
                _mc_vix_change = _mc.vix_change_pct
                _mc_sector = _mc.sector_etf
                _mc_reason = _mc.reason
                
                # Extreme condition: HARD BLOCK
                _extreme_spy = cfg.MOMENTUM_EXTREME_BLOCK_SPY_PCT
                _extreme_vix = cfg.MOMENTUM_EXTREME_BLOCK_VIX_SPIKE
                if _mc_spy_change <= _extreme_spy and _mc_vix_change >= _extreme_vix:
                    self._log_decision(
                        market_data, "skip", "extreme_risk_hard_block",
                        regime=_mc_regime,
                        spy_change=round(_mc_spy_change, 2),
                        vix_change=round(_mc_vix_change, 2),
                        threshold_spy=_extreme_spy,
                        threshold_vix=_extreme_vix,
                    )
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=symbol,
                        title=f"⛔ Day-Trade Momentum BLOCKED — Extreme Risk",
                        message=(
                            f"SPY {_mc_spy_change:+.2f}% AND VIX +{_mc_vix_change:.1f}% "
                            f"exceeds hard-block thresholds ({_extreme_spy}% / +{_extreme_vix}%). "
                            f"This is a circuit-breaker condition; no new momentum longs."
                        ),
                        importance=9,
                    ))
                    return None
                
                # risk_off: reduce size, don't hard-block
                if _mc_regime == "risk_off":
                    _mc_size_mult = cfg.MOMENTUM_RISK_OFF_SIZE_MULT
                    # v-fix-double-emit-2026-09-10: skip_snapshot=True for informational
                    # size reduction logs (not decision points, just context)
                    self._log_decision(
                        market_data, "size_reduced", "risk_off_not_blocked",
                        regime=_mc_regime,
                        size_mult=_mc_size_mult,
                        spy_change=round(_mc_spy_change, 2),
                        skip_snapshot=True,
                    )
                
                # opening_30: reduce size, don't hard-block
                if _mc_time_of_day == "opening_30":
                    _opening_mult = cfg.MOMENTUM_OPENING_30_SIZE_MULT
                    _mc_size_mult = min(_mc_size_mult, _opening_mult)
                    # v-fix-double-emit-2026-09-10: skip_snapshot=True for informational
                    # size reduction logs (not decision points, just context)
                    self._log_decision(
                        market_data, "size_reduced", "opening_30_not_blocked",
                        time_of_day=_mc_time_of_day,
                        size_mult=_mc_size_mult,
                        skip_snapshot=True,
                    )
                    
            except Exception as _mc_exc:
                logger.debug("day_trade_momentum: market_context failed: %s", _mc_exc)
            
            # ────────────────────────────────────────────────────────────────
            # Gate 2: Relative Strength vs SPY
            # ────────────────────────────────────────────────────────────────
            # Symbol must be outperforming SPY on the day
            _symbol_change = float(indicators.get('day_change_pct', 0))
            if _symbol_change == 0:
                # Fallback: estimate from price vs open
                _bar_open = float(getattr(market_data, 'open', market_data.close) or market_data.close)
                if _bar_open > 0:
                    _symbol_change = ((market_data.close - _bar_open) / _bar_open) * 100
            
            _rs_vs_spy = _symbol_change - _mc_spy_change
            _min_rs = cfg.MOMENTUM_MIN_RS_VS_SPY
            
            if _rs_vs_spy < _min_rs:
                self._log_decision(
                    market_data, "skip", "weak_relative_strength",
                    symbol_change=round(_symbol_change, 2),
                    spy_change=round(_mc_spy_change, 2),
                    rs_vs_spy=round(_rs_vs_spy, 2),
                    min_rs=_min_rs,
                )
                return None
            
            # ────────────────────────────────────────────────────────────────
            # Gate 3: Volume Surge Confirmation
            # ────────────────────────────────────────────────────────────────
            _min_vol_ratio = cfg.MOMENTUM_MIN_VOLUME_RATIO
            if volume_ratio < _min_vol_ratio:
                self._log_decision(
                    market_data, "skip", "insufficient_volume",
                    volume_ratio=round(volume_ratio, 2),
                    min_vol_ratio=_min_vol_ratio,
                )
                return None
            
            # ────────────────────────────────────────────────────────────────
            # Gate 4: Entry Pattern (Pullback or Breakout)
            # ────────────────────────────────────────────────────────────────
            _entry_pattern = None
            _entry_confidence = 0.0
            
            # Breakout pattern: price > 20-bar high, ADX > 20
            if market_data.close > high_20 and adx > 20:
                _breakout_dist_pct = ((market_data.close - high_20) / high_20) * 100
                if _breakout_dist_pct < 3.0:  # Not chasing if <3% above
                    _entry_pattern = "breakout"
                    _entry_confidence = 0.70 + (adx / 100) * 0.2
                    
            # Pullback pattern: RSI 40-60, price near SMA20, MACD bullish
            if _entry_pattern is None:
                if (40 <= rsi <= 60 and
                    sma_20 > 0 and
                    abs((market_data.close - sma_20) / sma_20) < 0.02 and  # within 2% of SMA20
                    macd > macd_signal):
                    _entry_pattern = "pullback"
                    _entry_confidence = 0.65 + (volume_ratio - 1.0) * 0.1
            
            # Continuation pattern: strong momentum (RSI 55-75, price trending)
            if _entry_pattern is None:
                if (55 <= rsi <= 75 and
                    adx > 25 and
                    macd > macd_signal and
                    market_data.close > sma_20 if sma_20 > 0 else True):
                    _entry_pattern = "continuation"
                    _entry_confidence = 0.60 + (adx / 100) * 0.2
            
            if _entry_pattern is None:
                self._log_decision(
                    market_data, "skip", "no_entry_pattern",
                    rsi=round(rsi, 2),
                    adx=round(adx, 2),
                    high_20=round(high_20, 2),
                    close=round(market_data.close, 2),
                    sma_20=round(sma_20, 2) if sma_20 > 0 else 0,
                    macd=round(macd, 4),
                    macd_signal=round(macd_signal, 4),
                )
                return None
            
            # ────────────────────────────────────────────────────────────────
            # v-shadow-veto-2026-09-10: Shadow veto for continuation + RSI>=70 + risk_off
            #
            # Pattern: continuation pattern into overbought (RSI>=70) during
            # risk_off regime is the classic failed breakout setup (exhaustion
            # gap). This is instrumentation for later promotion scoring.
            #
            # Shadow mode (log-only): we log the would-be entry with full
            # instrumentation but return None (don't place the trade). LIVE
            # Schwab fills will count for scorecard later when promoted to
            # a hard veto.
            #
            # Gated by Config.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF (default True)
            # ────────────────────────────────────────────────────────────────
            _shadow_veto_rsi_threshold = cfg.SHADOW_VETO_RSI_THRESHOLD
            _shadow_veto_triggered = (
                cfg.ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF
                and _entry_pattern == "continuation"
                and rsi >= _shadow_veto_rsi_threshold
                and _mc_regime == "risk_off"
            )
            
            if _shadow_veto_triggered:
                # Log the shadow veto with full instrumentation for later scoring
                self._log_decision(
                    market_data, "shadow_veto", "continuation_riskoff_rsi70",
                    pattern=_entry_pattern,
                    rsi=round(rsi, 2),
                    rsi_threshold=_shadow_veto_rsi_threshold,
                    regime=_mc_regime,
                    rs_vs_spy=round(_rs_vs_spy, 2),
                    volume_ratio=round(volume_ratio, 2),
                    adx=round(adx, 2),
                    high_20=round(high_20, 2),
                    close=round(market_data.close, 2),
                    sma_20=round(sma_20, 2) if sma_20 > 0 else 0,
                    macd=round(macd, 4),
                    macd_signal=round(macd_signal, 4),
                    spy_change=round(_mc_spy_change, 2),
                    vix_change=round(_mc_vix_change, 2),
                    session_id=_get_session_id(),
                )
                
                # Log to shadow veto file for later resolution
                try:
                    from datetime import timezone as _tz_sv
                    import json as _json_sv
                    _shadow_veto_entry = {
                        'timestamp': datetime.now(_tz_sv.utc).isoformat(),
                        'symbol': symbol,
                        'veto_type': 'continuation_riskoff_rsi70',
                        'entry_pattern': _entry_pattern,
                        'rsi': float(rsi),
                        'rsi_threshold': float(_shadow_veto_rsi_threshold),
                        'regime': _mc_regime,
                        'signal_close': float(market_data.close),
                        'rs_vs_spy': float(_rs_vs_spy),
                        'volume_ratio': float(volume_ratio),
                        'adx': float(adx),
                        'high_20': float(high_20),
                        'sma_20': float(sma_20) if sma_20 > 0 else 0,
                        'macd': float(macd),
                        'macd_signal': float(macd_signal),
                        'spy_change_pct': float(_mc_spy_change),
                        'vix_change_pct': float(_mc_vix_change),
                        'session_id': _get_session_id(),
                        # Include hypothetical stop/target for later outcome tracking
                        'hypothetical_atr': float(_floored_atr(atr, market_data.close)),
                        'hypothetical_stop': float(market_data.close - 1.5 * _floored_atr(atr, market_data.close)),
                        'hypothetical_target': float(market_data.close + 2.0 * 1.5 * _floored_atr(atr, market_data.close)),
                    }
                    _shadow_veto_ledger = 'shadow_veto_log.ndjson'
                    with open(_shadow_veto_ledger, 'a') as _f:
                        _f.write(_json_sv.dumps(_shadow_veto_entry, default=str) + "\n")
                except Exception as _sv_io_exc:
                    logger.warning(
                        "shadow_veto_log write failed for %s: %s",
                        symbol, _sv_io_exc,
                    )
                
                # Emit commentary about the shadow veto
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.RISK_ASSESSMENT,
                    symbol=symbol,
                    title=f"👁️ Shadow Veto: {_entry_pattern.upper()} + RSI≥70 + risk_off",
                    message=(
                        f"WOULD HAVE entered {symbol} via continuation pattern but:\n"
                        f"  RSI {rsi:.1f} >= {_shadow_veto_rsi_threshold} (overbought)\n"
                        f"  Regime: {_mc_regime} (risk off)\n\n"
                        f"This combination is the classic failed-breakout exhaustion setup.\n"
                        f"Shadow logged for later scoring; no order placed.\n\n"
                        f"Signal details: RS vs SPY {_rs_vs_spy:+.2f}%, Vol {volume_ratio:.1f}x"
                    ),
                    data={
                        'veto_type': 'shadow',
                        'pattern': _entry_pattern,
                        'rsi': rsi,
                        'rsi_threshold': _shadow_veto_rsi_threshold,
                        'regime': _mc_regime,
                        'rs_vs_spy': _rs_vs_spy,
                        'volume_ratio': volume_ratio,
                    },
                    importance=7,
                ))
                
                # Return None to prevent order placement
                return None
            
            # ────────────────────────────────────────────────────────────────
            # Calculate Stop/Target (ATR-based, tighter for day-trade)
            # ────────────────────────────────────────────────────────────────
            atr = _floored_atr(atr, market_data.close)
            
            # Day-trade uses tighter stops than swing (1.5x ATR vs 2.5x)
            _atr_stop_mult = 1.5
            _rr_ratio = 2.0  # 2:1 R:R for day trades
            
            stop_distance = _atr_stop_mult * atr
            stop_loss = market_data.close - stop_distance
            take_profit = market_data.close + (_rr_ratio * stop_distance)
            
            # ────────────────────────────────────────────────────────────────
            # Calculate signal strength and confidence
            # ────────────────────────────────────────────────────────────────
            # Strength combines: RS vs SPY, volume surge, ADX
            _strength = min(0.95, (
                0.3 +  # base
                min(0.25, _rs_vs_spy / 5.0) +  # RS contribution (up to +0.25 at 5% RS)
                min(0.25, (volume_ratio - 1.0) / 2.0) +  # Volume contribution
                min(0.15, adx / 100)  # ADX contribution
            ))
            
            _confidence = min(0.85, _entry_confidence * (1.0 + _rs_vs_spy / 10.0))
            
            # ────────────────────────────────────────────────────────────────
            # Emit commentary and signal
            # ────────────────────────────────────────────────────────────────
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.OPPORTUNITY,
                symbol=symbol,
                title=f"🚀 Day-Trade Momentum: {_entry_pattern.upper()}",
                message=(
                    f"Intraday momentum setup detected.\n"
                    f"  Pattern: {_entry_pattern}\n"
                    f"  RS vs SPY: {_rs_vs_spy:+.2f}%\n"
                    f"  Volume: {volume_ratio:.1f}x average\n"
                    f"  RSI: {rsi:.1f}  ADX: {adx:.1f}\n"
                    f"  Market: {_mc_regime or 'unknown'} | {_mc_time_of_day or 'unknown'}\n"
                    f"  Size mult: {_mc_size_mult:.2f}x"
                ),
                data={
                    'pattern': _entry_pattern,
                    'rs_vs_spy': _rs_vs_spy,
                    'volume_ratio': volume_ratio,
                    'rsi': rsi,
                    'adx': adx,
                    'market_regime': _mc_regime,
                    'size_mult': _mc_size_mult,
                },
                confidence=_confidence,
                importance=8,
            ))
            
            self._log_decision(
                market_data, "signal_buy", f"day_trade_{_entry_pattern}",
                pattern=_entry_pattern,
                rs_vs_spy=round(_rs_vs_spy, 2),
                volume_ratio=round(volume_ratio, 2),
                rsi=round(rsi, 2),
                adx=round(adx, 2),
                regime=_mc_regime,
                time_of_day=_mc_time_of_day,
                mc_size_mult=_mc_size_mult,
                stop=round(stop_loss, 2),
                target=round(take_profit, 2),
                atr=round(atr, 3),
            )
            
            # ────────────────────────────────────────────────────────────────
            # Stage-A Instrumentation: session_id, risk_off flag, setup_type
            # v-momentum-stage-a-2026-09-10
            # ────────────────────────────────────────────────────────────────
            _session_id = _get_session_id()
            _risk_off = (_mc_regime == "risk_off")
            
            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=_strength,
                entry_price=market_data.close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=0,  # Sized by risk manager
                reasoning={
                    'strategy': 'day_trade_momentum',
                    'entry_pattern': _entry_pattern,
                    'rs_vs_spy': _rs_vs_spy,
                    'volume_ratio': volume_ratio,
                    'atr': atr,
                    'atr_mult': _atr_stop_mult,
                    'stop_distance': stop_distance,
                    # Market context for sizing
                    'market_context_conviction': _mc_size_mult,
                    'market_context_regime': _mc_regime,
                    'market_context_sector': _mc_sector,
                    'market_context_time_of_day': _mc_time_of_day,
                    # Day-trade specific
                    'is_day_trade': True,
                    'day_trade_size_multiplier': cfg.DAY_TRADE_SIZE_MULTIPLIER,
                    'flatten_hour': cfg.DAY_TRADE_FLATTEN_HOUR,
                    # Stage-A instrumentation (v-momentum-stage-a-2026-09-10)
                    'session_id': _session_id,
                    'risk_off': _risk_off,
                    'mc_size_mult': _mc_size_mult,
                    'setup_type': f"momentum_{_entry_pattern}",
                },
                confidence=_confidence,
            )
            
        except Exception as e:
            self._log_decision(market_data, "error", "exception", err=str(e))
            logger.debug(f"DayTradeMomentumStrategy error for {symbol}: {e}")
            return None
