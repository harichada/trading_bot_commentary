"""v-oversold-v2-2026-05-19: OversoldBounceV2 strategy.

Six-gate long-only mean-reversion strategy that fires when a symbol is
statistically extreme AND showing real capitulation AND the broader market
isn't in free-fall AND there's no fresh negative news. Built on the lessons
from MeanReversionStrategyWithCommentary — which was getting chopped up by
falling-knife setups — and adds:

  Gate A — Statistical extreme  (RSI < 30 AND close < BB_lower)
  Gate B — Capitulation volume  (volume_ratio > 2.0)
  Gate C — Bullish bar structure (long lower wick OR close >= prev_low)
  Gate D — Support level nearby (a swing low within 0.5 ATR)
  Gate E — Market regime         (SPY EMA-slope not strongly down)
  Gate F — Adverse news check    (no fresh negative-sentiment news)

ALL six must pass. The first failure logs an audit line with the gate name
and the function returns ``None``. Exit-management hints are passed back
through the ``reasoning`` dict so the engine's scale-out / trail / time-stop
modules can apply per-strategy overrides.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from core.models import MarketData, SignalType, TradingSignal
from strategies.base import TradingStrategyWithCommentary


logger = logging.getLogger("TradingBot")


# ----------------------------------------------------------------------
# Module-level helpers (pure functions — easy to unit-test in isolation).
# ----------------------------------------------------------------------
def bar_has_capitulation_structure(
    open_: float,
    high: float,
    low: float,
    close: float,
    prev_low: Optional[float],
) -> bool:
    """Return True if the current bar shows a bullish reversal structure.

    Two acceptance shapes:
      1. A long lower wick (lower_wick > 1.5 * body) — classic hammer.
      2. ``close >= prev_low`` — the bar reclaimed the previous low,
         signalling that sellers exhausted at the prior support.

    Either shape passes. ``prev_low`` may be ``None`` (insufficient
    history) — in that case only the wick shape can satisfy the gate.
    """
    try:
        o = float(open_)
        h = float(high)
        lo = float(low)
        c = float(close)
    except (TypeError, ValueError):
        return False

    lower_wick = min(o, c) - lo
    body = abs(c - o)

    # Long lower wick relative to the candle's body.
    if body > 0 and lower_wick > 1.5 * body:
        return True
    # Doji with any lower wick still wins the structure check only if the
    # wick is meaningfully long vs. the bar's total range.
    if body == 0 and lower_wick > 0 and (h - lo) > 0 and lower_wick / (h - lo) > 0.6:
        return True

    if prev_low is not None:
        try:
            if c >= float(prev_low):
                return True
        except (TypeError, ValueError):
            pass

    return False


def find_nearest_swing_low(
    lows: List[float],
    current_price: float,
    atr: float,
) -> Optional[float]:
    """Scan ``lows`` for the most recent local minimum within 0.5 * ATR.

    A swing low at index ``i`` is ``lows[i] < lows[i-1]`` and
    ``lows[i] < lows[i+1]`` (i.e. a strict local minimum, ignoring the
    first/last entries which can't be evaluated). We walk from most-recent
    to oldest and return the first swing low whose distance from
    ``current_price`` is at most 0.5 * ``atr``. Returns ``None`` if no
    such level exists or inputs are degenerate.
    """
    if atr <= 0:
        return None
    if lows is None:
        return None

    arr = np.asarray(lows, dtype=float)
    if arr.size < 3:
        return None

    threshold = 0.5 * float(atr)
    # i ranges over interior indices [1, len-2]; walk most-recent first.
    for i in range(arr.size - 2, 0, -1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            if abs(float(current_price) - float(arr[i])) <= threshold:
                return float(arr[i])
    return None


# ----------------------------------------------------------------------
# Strategy class
# ----------------------------------------------------------------------
class OversoldBounceV2Strategy(TradingStrategyWithCommentary):
    """Six-gate oversold-bounce strategy — long-only.

    See module docstring for the gate list. The constructor accepts an
    optional ``adverse_news_verifier`` so tests can inject a mock. When
    omitted, a real :class:`NewsVerifier` is lazy-constructed with a tight
    30-min freshness window and a 0.3 sentiment-match threshold.
    """

    name = "oversold_bounce_v2"

    def __init__(self, commentary_system, adverse_news_verifier=None) -> None:
        super().__init__(commentary_system)
        self._adverse_news_verifier = adverse_news_verifier  # may be None

    # ------------------------------------------------------------------
    # Lazy verifier construction
    # ------------------------------------------------------------------
    def _get_news_verifier(self):
        """Return the injected verifier, building one on first use if needed."""
        if self._adverse_news_verifier is not None:
            return self._adverse_news_verifier
        try:
            from analysis.news_verifier import NewsVerifier
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._adverse_news_verifier = NewsVerifier(
                sentiment_analyzer=SentimentIntensityAnalyzer(),
                freshness_minutes=30,
                min_match_strength=0.3,
            )
        except Exception as exc:
            # If we can't even import the verifier, leave the slot empty.
            # Gate F treats missing verifier as 'api_unavailable' → pass.
            logger.debug("oversold_bounce_v2: news verifier unavailable: %s", exc)
            self._adverse_news_verifier = None
        return self._adverse_news_verifier

    # ------------------------------------------------------------------
    # Sizing
    # ------------------------------------------------------------------
    def _compute_sizing(self, market_data: MarketData, atr: float) -> Tuple[float, float, float]:
        """Return (stop_distance, stop_loss, take_profit).

        stop_distance = max(2 * ATR, 0.5 * bar_range)
        stop_loss     = close - stop_distance
        take_profit   = close + 3 * stop_distance   (3:1 reward:risk)
        """
        bar_range = max(0.0, float(market_data.high) - float(market_data.low))
        stop_distance = max(2.0 * float(atr), 0.5 * bar_range)
        stop_loss = float(market_data.close) - stop_distance
        take_profit = float(market_data.close) + 3.0 * stop_distance
        return stop_distance, stop_loss, take_profit

    # ------------------------------------------------------------------
    # Gates A–F evaluated sequentially
    # ------------------------------------------------------------------
    async def generate_signal_with_commentary(
        self, market_data: MarketData
    ) -> Optional[TradingSignal]:
        try:
            indicators = market_data.indicators or {}
            symbol = market_data.symbol
            close = float(market_data.close)
            open_ = float(getattr(market_data, "open", close) or close)
            high = float(market_data.high)
            low = float(market_data.low)

            # --- Gate A: statistical extreme ----------------------------------
            rsi = float(indicators.get("rsi", 100))
            bb_lower = float(indicators.get("bb_lower", 0))
            if not (rsi < 30 and close < bb_lower):
                self._log_decision(
                    market_data, "skip", "gate_a_not_extreme",
                    rsi=round(rsi, 2), close=close, bb_lower=bb_lower,
                )
                return None

            # --- Gate B: capitulation volume ----------------------------------
            volume_ratio = float(indicators.get("volume_ratio", 0))
            if not (volume_ratio > 2.0):
                self._log_decision(
                    market_data, "skip", "gate_b_no_capitulation",
                    volume_ratio=round(volume_ratio, 2),
                )
                return None

            # --- Gate C: bullish bar structure --------------------------------
            prev_low = indicators.get("prev_low")
            if prev_low is not None:
                try:
                    prev_low = float(prev_low)
                except (TypeError, ValueError):
                    prev_low = None
            if not bar_has_capitulation_structure(open_, high, low, close, prev_low):
                self._log_decision(
                    market_data, "skip", "gate_c_no_capitulation_structure",
                    open=open_, high=high, low=low, close=close,
                    prev_low=prev_low,
                )
                return None

            # --- Gate D: support level nearby ---------------------------------
            atr = float(indicators.get("atr", 0))
            if atr <= 0:
                self._log_decision(
                    market_data, "skip", "gate_d_invalid_atr", atr=atr,
                )
                return None
            lows_50 = indicators.get("lows_50") or []
            nearest_swing = find_nearest_swing_low(lows_50, close, atr)
            if nearest_swing is None:
                self._log_decision(
                    market_data, "skip", "gate_d_no_nearby_support",
                    atr=round(atr, 3), lows_seen=len(lows_50),
                )
                return None

            # --- Gate E: market regime ----------------------------------------
            slope = 0.0
            try:
                from core.spy_regime import SpyRegimeCache
                slope = float(SpyRegimeCache.instance().ema_slope_pct(period=9))
            except Exception as exc:
                # SPY cache not wired yet — fall back to a flat regime.
                logger.debug("oversold_bounce_v2: SPY regime unavailable: %s", exc)
                slope = 0.0

            if slope >= -0.05:
                regime_factor = 1.0
            elif slope > -0.10:  # -0.10 < slope < -0.05
                regime_factor = 0.5
            else:
                regime_factor = 0.0

            if regime_factor == 0.0:
                self._log_decision(
                    market_data, "skip", "gate_e_strong_down_market",
                    spy_slope_pct=round(slope, 3),
                )
                return None

            # --- Gate F: adverse news check (async) ---------------------------
            news_gate_status = "ok"
            news_fresh_count = 0
            news_avg_sent = 0.0
            verifier = self._get_news_verifier()
            if verifier is None:
                news_gate_status = "api_unavailable"
            else:
                try:
                    result = await verifier.verify(symbol=symbol, expected_direction=+1)
                    news_fresh_count = int(getattr(result, "fresh_count", 0) or 0)
                    news_avg_sent = float(getattr(result, "avg_fresh_sentiment", 0.0) or 0.0)
                    if news_fresh_count > 0 and news_avg_sent < -0.3:
                        self._log_decision(
                            market_data, "skip", "gate_f_adverse_news",
                            fresh_count=news_fresh_count,
                            avg_sent=round(news_avg_sent, 3),
                        )
                        return None
                except Exception as exc:
                    # Conservative posture: API down should NOT block trades.
                    logger.debug(
                        "oversold_bounce_v2: news verifier raised for %s: %s",
                        symbol, exc,
                    )
                    news_gate_status = "api_unavailable"

            # --- All gates passed → build the signal --------------------------
            stop_distance, stop_loss, take_profit = self._compute_sizing(
                market_data, atr,
            )
            bar_range = max(0.0, high - low)

            reasoning = {
                "strategy": "oversold_bounce_v2",
                "primary_reason": "oversold_bounce_v2",
                "atr": round(atr, 4),
                "rsi": round(rsi, 2),
                "volume_ratio": round(volume_ratio, 2),
                "bar_range": round(bar_range, 4),
                "stop_distance": round(stop_distance, 4),
                "regime_factor": regime_factor,
                "nearest_swing_low": round(nearest_swing, 4),
                "spy_slope_pct": round(slope, 3),
                "news_gate_status": news_gate_status,
                "news_fresh_count": news_fresh_count,
                "news_avg_sentiment": round(news_avg_sent, 3),
                # Engine-side exit-management hints. Scale-out at +1×ATR
                # (== 0.5R when stop_distance == 2*ATR), lift stop to BE
                # after +1.5×ATR favorable excursion, and force-exit a
                # stagnant trade after 30 minutes.
                "scale_out_r_override": 0.5,
                "breakeven_atr_mult": 1.5,
                "time_stop_minutes": 30,
            }

            signal = TradingSignal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                entry_price=close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=0.65,
                strength=0.65,
                position_size=0,
                timestamp=market_data.timestamp,
                reasoning=reasoning,
            )

            self._log_decision(
                market_data, "signal_buy", "oversold_bounce_v2_pass",
                rsi=round(rsi, 2),
                volume_ratio=round(volume_ratio, 2),
                atr=round(atr, 4),
                stop_distance=round(stop_distance, 4),
                stop_loss=round(stop_loss, 4),
                take_profit=round(take_profit, 4),
                regime_factor=regime_factor,
                spy_slope_pct=round(slope, 3),
                nearest_swing_low=round(nearest_swing, 4),
                news_gate_status=news_gate_status,
            )
            return signal

        except Exception as exc:
            self._log_decision(market_data, "error", "exception", err=str(exc))
            logger.debug(
                "oversold_bounce_v2 error for %s: %s",
                getattr(market_data, "symbol", "?"), exc,
            )
            return None
