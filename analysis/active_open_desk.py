"""v-active-open-desk-2026-09-14: continuous monitor for open trades.

RCA FTFT 2026-09-14: bot set OCO bracket + hard stop then idled. Hari:
must continuously monitor ALL managed open trades for sentiment/regime/
indicators and take proactive action (not fire-and-forget brackets).

PR1 SHADOW ONLY: logs WOULD_TIGHTEN / WOULD_TRAIL / WOULD_EXIT with
reason + symbol + suggested levels. No broker calls, no order mutations,
no cancel-replace, no market exit.

PR2 v-proactive-exit-daytrade-2026-09-14: Proactive exit age/R gates.
  - PROACTIVE_EXIT_MIN_AGE_DAYTRADE: day trades use shorter min-age (3min
    default vs 15min) because intraday momentum breaks faster than swing.
  - PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD: when pnl_r <= -0.3R (default),
    bypass min-age entirely — thesis is likely broken regardless of age.
  - FTFT fix: held ~20min, proactive_exit suppressed by below_min_age the
    whole time, then hit hard_stop. Now: day trade would fire at 3min or
    immediately if R-override triggers.
  - Shadow decisions include age/R metadata: age_minutes, pnl_r, is_day_trade,
    age_gate_met, r_override_used.
  - HANDS_OFF via Config.HANDS_OFF_DENYLIST only — never hardcodes symbols.
  - LT/HANDS_OFF unchanged: is_long_term positions skip desk entirely.

Modular boundary:
  - Config flags: ENABLE_ACTIVE_OPEN_DESK (master switch, default True),
    ACTIVE_OPEN_DESK_SHADOW (default True), ACTIVE_OPEN_DESK_INTERVAL_SEC.
  - Default True + shadow=True: evidence soak mode, structured logs only.
  - SAFE OFF-PATH: Set ENABLE_ACTIVE_OPEN_DESK=0/false to disable entirely
    and preserve today's bracket + hard-stop behavior unchanged.
  - Respects HANDS_OFF_DENYLIST from Config — never hardcodes symbols.

Design integrates with live modular paths (not monolith-only):
  - analysis/exit_managers.py: existing exit logic patterns
  - core/order_monitor/: fill-path bracket monitoring (NOT this desk)
  - core/news_bus.py + news_loop: sentiment source

Inputs:
  - NewsBus: per-symbol sentiment from core/news_bus.py
  - MarketContext: regime + sector + VIX from core/market_context.py
  - Indicators: RSI/MACD/ADX from data_provider via engine

Outputs (shadow mode):
  - strategy_decision-style structured logs:
      WOULD_TIGHTEN: reason, symbol, current_stop, suggested_stop
      WOULD_TRAIL: reason, symbol, suggested_trail_stop
      WOULD_EXIT: reason, symbol, suggested_exit_price, age, pnl_r

Future PR3+: LIVE cancel-replace / market exit actuators behind
ACTIVE_OPEN_DESK_SHADOW=False.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from core.engine import TradingEngineWithCommentary
    from core.models import Position

logger = logging.getLogger("TradingBot")


class DeskAction(str, Enum):
    """Action types the Active Open Desk can recommend."""
    WOULD_TIGHTEN = "WOULD_TIGHTEN"
    WOULD_TRAIL = "WOULD_TRAIL"
    WOULD_CANCEL_REPLACE = "WOULD_CANCEL_REPLACE"
    WOULD_EXIT = "WOULD_EXIT"
    NO_ACTION = "NO_ACTION"


@dataclass
class DeskDecision:
    """A decision from the Active Open Desk for one position."""
    symbol: str
    action: DeskAction
    reason: str
    current_stop: Optional[float] = None
    suggested_stop: Optional[float] = None
    suggested_target: Optional[float] = None
    suggested_exit_price: Optional[float] = None
    current_price: Optional[float] = None
    regime: Optional[str] = None
    sentiment_score: Optional[float] = None
    rsi: Optional[float] = None
    macd_signal: Optional[str] = None
    timestamp: datetime = None
    # v-proactive-exit-daytrade-2026-09-14: age/R gate tracking
    age_minutes: Optional[float] = None
    pnl_r: Optional[float] = None
    is_day_trade: bool = False
    age_gate_met: bool = False
    r_override_used: bool = False

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Serialize for structured logging."""
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "reason": self.reason,
            "current_stop": self.current_stop,
            "suggested_stop": self.suggested_stop,
            "suggested_target": self.suggested_target,
            "suggested_exit_price": self.suggested_exit_price,
            "current_price": self.current_price,
            "regime": self.regime,
            "sentiment_score": self.sentiment_score,
            "rsi": self.rsi,
            "macd_signal": self.macd_signal,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "age_minutes": self.age_minutes,
            "pnl_r": self.pnl_r,
            "is_day_trade": self.is_day_trade,
            "age_gate_met": self.age_gate_met,
            "r_override_used": self.r_override_used,
        }


class ActiveOpenDesk:
    """Continuous monitor for all bot-managed open positions.

    v-active-open-desk-2026-09-14: spawned by the engine's task supervisor
    when ENABLE_ACTIVE_OPEN_DESK=True. Iterates managed_by_bot positions
    in parallel via asyncio.gather, evaluating sentiment, regime, and
    technical indicators for proactive exit/tighten recommendations.

    Shadow mode (ACTIVE_OPEN_DESK_SHADOW=True, the default): logs
    structured strategy_decision-style events without executing any
    broker calls. Allows validation of the logic before enabling
    live actuators.

    Respects HANDS_OFF_DENYLIST (MU, HQGE, SPCX) — never evaluates
    these positions.
    """

    def __init__(self, engine: "TradingEngineWithCommentary") -> None:
        self.engine = engine
        self._last_decisions: Dict[str, DeskDecision] = {}

    async def run(self) -> None:
        """Main loop — poll positions at configured interval."""
        from core.config import Config

        cfg = Config()
        interval = cfg.ACTIVE_OPEN_DESK_INTERVAL_SEC

        logger.info(
            "active_open_desk: started (shadow=%s, interval=%.1fs)",
            cfg.ACTIVE_OPEN_DESK_SHADOW,
            interval,
        )

        while self.engine.is_running:
            try:
                await self._tick()
            except Exception as exc:
                logger.error(
                    "active_open_desk: tick error: %s",
                    exc,
                    exc_info=True,
                )
            await asyncio.sleep(interval)

        logger.info("active_open_desk: stopped")

    async def _tick(self) -> None:
        """One evaluation cycle across all monitored positions."""
        positions = self.get_monitored_positions()
        if not positions:
            return

        tasks = [
            self._evaluate_position(symbol, position)
            for symbol, position in positions
        ]
        decisions = await asyncio.gather(*tasks, return_exceptions=True)

        from core.config import Config
        is_shadow = Config().ACTIVE_OPEN_DESK_SHADOW

        for decision in decisions:
            if isinstance(decision, Exception):
                logger.warning(
                    "active_open_desk: position evaluation failed: %s",
                    decision,
                )
                continue

            if decision is None or decision.action == DeskAction.NO_ACTION:
                continue

            self._last_decisions[decision.symbol] = decision

            if is_shadow:
                self._log_shadow_decision(decision)
            else:
                pass

    def get_monitored_positions(self) -> List[Tuple[str, "Position"]]:
        """Return (symbol, position) pairs for positions to monitor.

        Criteria:
          - In engine.positions (LIVE mode) OR engine.simulated_positions (SIM mode)
          - managed_by_bot=True
          - NOT in HANDS_OFF_DENYLIST (MU, HQGE, SPCX)
          - NOT is_external / is_manually_managed / is_long_term
        """
        from core.config import Config

        denylist = Config().HANDS_OFF_DENYLIST
        result = []

        containers = [
            self.engine.positions,
            getattr(self.engine, 'simulated_positions', {}) or {},
        ]

        for container in containers:
            for symbol, pos in list(container.items()):
                if pos is None:
                    continue
                if not getattr(pos, 'managed_by_bot', False):
                    continue
                if symbol.upper() in denylist:
                    continue
                if getattr(pos, 'is_external', False):
                    continue
                if getattr(pos, 'is_manually_managed', False):
                    continue
                if getattr(pos, 'is_long_term', False):
                    continue
                result.append((symbol, pos))

        return result

    async def _evaluate_position(
        self,
        symbol: str,
        position: "Position",
    ) -> Optional[DeskDecision]:
        """Evaluate a single position for proactive action.

        v-proactive-exit-daytrade-2026-09-14 (PR2): added proactive exit
        evaluation with age-gate and R-override for day trades.

        Checks (in priority order):
          0. Proactive exit: age+R gate, indicator confirmation (FTFT fix)
          1. Regime: if risk_off and position is long, consider exit/tighten
          2. Sentiment: if news sentiment turned negative for a long position
          3. Indicators: RSI overbought (exit) or MACD cross against position
        """
        try:
            context = self._get_market_context(symbol)
            news_sentiment = await self._get_news_sentiment(symbol)
            indicators = await self._get_indicators(symbol)
        except Exception as exc:
            logger.debug(
                "active_open_desk: failed to fetch context for %s: %s",
                symbol,
                exc,
            )
            return None

        current_price = getattr(position, 'current_price', 0.0) or position.entry_price
        current_stop = position.stop_loss
        side = position.side

        # v-proactive-exit-daytrade-2026-09-14: check proactive exit FIRST
        # This is the FTFT RCA fix — don't let min-age suppress early losers
        proactive_decision = self._check_proactive_exit_gate(
            position, current_price, indicators
        )
        if proactive_decision is not None:
            proactive_decision.regime = context.regime if context else "unknown"
            proactive_decision.sentiment_score = news_sentiment
            proactive_decision.rsi = indicators.get("rsi", 50.0) if indicators else 50.0
            macd = indicators.get("macd", 0.0) if indicators else 0.0
            macd_sig = indicators.get("macd_signal", 0.0) if indicators else 0.0
            proactive_decision.macd_signal = (
                "bullish" if macd > macd_sig else "bearish" if macd < macd_sig else "neutral"
            )
            return proactive_decision

        regime = context.regime if context else "unknown"
        sentiment_score = news_sentiment
        rsi = indicators.get("rsi", 50.0) if indicators else 50.0
        macd = indicators.get("macd", 0.0) if indicators else 0.0
        macd_sig = indicators.get("macd_signal", 0.0) if indicators else 0.0

        macd_signal_str = "neutral"
        if macd > macd_sig:
            macd_signal_str = "bullish"
        elif macd < macd_sig:
            macd_signal_str = "bearish"

        action = DeskAction.NO_ACTION
        reason = ""
        suggested_stop = None
        suggested_exit_price = None

        # Compute age and R for all decisions (for logging context)
        age_minutes = (datetime.now() - position.entry_time).total_seconds() / 60.0
        pnl_r = self._compute_pnl_r(position, current_price)
        is_day_trade = self._is_day_trade(position)

        if side == "long":
            if regime == "risk_off":
                action = DeskAction.WOULD_TIGHTEN
                reason = "regime_risk_off_tighten"
                suggested_stop = self._compute_tighter_stop(position, current_price)

            elif sentiment_score is not None and sentiment_score < -0.3:
                action = DeskAction.WOULD_TIGHTEN
                reason = f"negative_sentiment_{sentiment_score:.2f}"
                suggested_stop = self._compute_tighter_stop(position, current_price)

            elif rsi >= 80:
                action = DeskAction.WOULD_EXIT
                reason = f"rsi_extreme_overbought_{rsi:.1f}"
                suggested_exit_price = current_price

            elif macd_signal_str == "bearish" and rsi > 60:
                action = DeskAction.WOULD_TIGHTEN
                reason = f"macd_bearish_cross_rsi_{rsi:.1f}"
                suggested_stop = self._compute_tighter_stop(position, current_price)

        else:
            if regime == "risk_on":
                action = DeskAction.WOULD_TIGHTEN
                reason = "regime_risk_on_tighten"
                suggested_stop = self._compute_tighter_stop(position, current_price)

            elif sentiment_score is not None and sentiment_score > 0.3:
                action = DeskAction.WOULD_TIGHTEN
                reason = f"positive_sentiment_{sentiment_score:.2f}"
                suggested_stop = self._compute_tighter_stop(position, current_price)

            elif rsi <= 20:
                action = DeskAction.WOULD_EXIT
                reason = f"rsi_extreme_oversold_{rsi:.1f}"
                suggested_exit_price = current_price

            elif macd_signal_str == "bullish" and rsi < 40:
                action = DeskAction.WOULD_TIGHTEN
                reason = f"macd_bullish_cross_rsi_{rsi:.1f}"
                suggested_stop = self._compute_tighter_stop(position, current_price)

        return DeskDecision(
            symbol=symbol,
            action=action,
            reason=reason,
            current_stop=current_stop,
            suggested_stop=suggested_stop,
            suggested_exit_price=suggested_exit_price,
            current_price=current_price,
            regime=regime,
            sentiment_score=sentiment_score,
            rsi=rsi,
            macd_signal=macd_signal_str,
            age_minutes=age_minutes,
            pnl_r=pnl_r,
            is_day_trade=is_day_trade,
        )

    def _get_market_context(self, symbol: str):
        """Fetch market context (regime, sector, VIX)."""
        try:
            from core.market_context import read_market_context
            return read_market_context(symbol)
        except Exception:
            return None

    async def _get_news_sentiment(self, symbol: str) -> Optional[float]:
        """Fetch latest news sentiment score for symbol from NewsBus."""
        try:
            from core.news_bus import get_news_bus
            bus = get_news_bus()
            agg = await bus.get_aggregate_sentiment(symbol, max_age_sec=1800)
            if agg and agg.get("article_count", 0) > 0:
                return agg.get("avg_sentiment", 0.0)
        except Exception:
            pass
        return None

    async def _get_indicators(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch current technical indicators for symbol."""
        try:
            if self.engine.data_provider is None:
                return None
            data = self.engine.data_provider.get_market_data(symbol)
            if data is None or data.empty:
                return None
            indicators = await self.engine.technical_analyzer.analyze_with_commentary(
                data, symbol
            )
            return indicators
        except Exception:
            pass
        return None

    def _compute_tighter_stop(
        self,
        position: "Position",
        current_price: float,
    ) -> float:
        """Compute a tighter stop level based on position side and current price.

        For a long: move stop up toward current price (reduce risk).
        For a short: move stop down toward current price (reduce risk).

        Uses a simple 50% of distance rule for shadow mode demonstration.
        """
        entry = position.entry_price
        current_stop = position.stop_loss

        if position.side == "long":
            if current_price > entry:
                new_stop = entry + (current_price - entry) * 0.5
                return max(new_stop, current_stop)
            return current_stop
        else:
            if current_price < entry:
                new_stop = entry - (entry - current_price) * 0.5
                return min(new_stop, current_stop)
            return current_stop

    def _is_day_trade(self, position: "Position") -> bool:
        """Check if a position is a day trade based on reasoning.

        v-proactive-exit-daytrade-2026-09-14: day trades use shorter min-age
        for proactive exit since intraday momentum breaks faster than swing theses.
        """
        reasoning = getattr(position, 'reasoning', {}) or {}
        strategy = reasoning.get('strategy', '')
        is_day_trade_flag = reasoning.get('is_day_trade', False)
        return (
            is_day_trade_flag
            or strategy == 'day_trade_momentum'
            or 'day_trade' in strategy.lower()
        )

    def _compute_pnl_r(
        self,
        position: "Position",
        current_price: float,
    ) -> float:
        """Compute unrealized P&L in R-multiples.

        Returns negative values when the trade is adverse (losing).
        """
        original_stop = position.original_stop or position.stop_loss
        stop_distance = abs(position.entry_price - original_stop)
        if stop_distance <= 0:
            return 0.0

        if position.side == "long":
            return (current_price - position.entry_price) / stop_distance
        else:
            return (position.entry_price - current_price) / stop_distance

    def _get_min_age_for_position(self, position: "Position") -> int:
        """Get the minimum age (minutes) before proactive exit can fire.

        v-proactive-exit-daytrade-2026-09-14: day trades use shorter min-age
        (default 3 min) vs swing/news (15-30 min) because intraday momentum
        breaks faster.
        """
        from core.config import Config
        cfg = Config()

        if self._is_day_trade(position):
            return cfg.PROACTIVE_EXIT_MIN_AGE_DAYTRADE

        reasoning = getattr(position, 'reasoning', {}) or {}
        strategy = reasoning.get('strategy', '')

        is_news = 'news' in strategy.lower() or strategy == 'free_news_sentiment'
        is_meanrev = 'mean_reversion' in strategy.lower()

        if is_news:
            return cfg.PROACTIVE_EXIT_MIN_AGE_NEWS
        elif is_meanrev:
            return cfg.PROACTIVE_EXIT_MIN_AGE_MEANREV
        else:
            return cfg.PROACTIVE_EXIT_MIN_AGE_DEFAULT

    def _check_proactive_exit_gate(
        self,
        position: "Position",
        current_price: float,
        indicators: Optional[Dict[str, Any]],
    ) -> Optional[DeskDecision]:
        """Evaluate proactive exit conditions with age-gate and R-override.

        v-proactive-exit-daytrade-2026-09-14 (PR2):
        RCA FTFT: held ~20min, proactive_exit suppressed by below_min_age (15m)
        the entire time, then hit hard_stop. Fix:
          1. Day trades use shorter min-age (3min default vs 15min)
          2. R-override: if pnl_r <= -0.3R, bypass min-age entirely

        Returns a DeskDecision if proactive exit/tighten should fire, else None.
        The decision includes age/R metadata for shadow logging.
        """
        from core.config import Config
        cfg = Config()

        if not cfg.ENABLE_PROACTIVE_EXIT:
            return None

        age_minutes = (datetime.now() - position.entry_time).total_seconds() / 60.0
        pnl_r = self._compute_pnl_r(position, current_price)
        is_day_trade = self._is_day_trade(position)
        min_age = self._get_min_age_for_position(position)
        r_override_threshold = cfg.PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD

        age_gate_met = age_minutes >= min_age
        r_override_used = pnl_r <= r_override_threshold

        proactive_allowed = age_gate_met or r_override_used

        if not proactive_allowed:
            logger.debug(
                "active_open_desk: proactive_exit suppressed symbol=%s "
                "age=%.1fm min_age=%dm pnl_r=%.2f r_override=%.2f is_day_trade=%s",
                position.symbol,
                age_minutes,
                min_age,
                pnl_r,
                r_override_threshold,
                is_day_trade,
            )
            return None

        if indicators is None:
            return None

        macd = float(indicators.get("macd", 0) or 0)
        macd_sig = float(indicators.get("macd_signal", 0) or 0)
        rsi = float(indicators.get("rsi", 50) or 50)
        adx = float(indicators.get("adx", 0) or 0)
        adx_prev = float(indicators.get("adx_prev", adx) or adx)

        reasoning = getattr(position, 'reasoning', {}) or {}
        strategy = reasoning.get('strategy', '')
        is_news = 'news' in strategy.lower() or strategy == 'free_news_sentiment'

        pnl_threshold = -0.70 if is_news else -0.50
        if pnl_r > pnl_threshold:
            return None

        proactive_reason = None
        if position.side == "long":
            if macd < macd_sig:
                proactive_reason = "macd_flipped_bearish"
            elif rsi < 50:
                proactive_reason = "rsi_below_50"
        else:
            if macd > macd_sig:
                proactive_reason = "macd_flipped_bullish"
            elif rsi > 50:
                proactive_reason = "rsi_above_50"

        if proactive_reason is None:
            if adx < 20 and adx < adx_prev:
                proactive_reason = "adx_collapsing"

        if proactive_reason is None:
            return None

        reason_detail = (
            f"{proactive_reason}_age={age_minutes:.1f}m_pnl_r={pnl_r:.2f}"
        )
        if r_override_used and not age_gate_met:
            reason_detail += f"_r_override_bypass"
        if is_day_trade:
            reason_detail += "_daytrade"

        return DeskDecision(
            symbol=position.symbol,
            action=DeskAction.WOULD_EXIT,
            reason=reason_detail,
            current_stop=position.stop_loss,
            suggested_exit_price=current_price,
            current_price=current_price,
            age_minutes=age_minutes,
            pnl_r=pnl_r,
            is_day_trade=is_day_trade,
            age_gate_met=age_gate_met,
            r_override_used=r_override_used,
        )

    def _log_shadow_decision(self, decision: DeskDecision) -> None:
        """Log a shadow decision in strategy_decision style.

        v-proactive-exit-daytrade-2026-09-14: includes age_minutes, pnl_r,
        is_day_trade, age_gate_met, r_override_used for FTFT RCA observability.
        """
        logger.info(
            "active_open_desk action=%s symbol=%s reason=%s "
            "current_stop=%.4f suggested_stop=%s suggested_exit=%s "
            "price=%.4f regime=%s sentiment=%s rsi=%.1f macd=%s "
            "age=%.1fm pnl_r=%.2f is_day_trade=%s age_gate=%s r_override=%s",
            decision.action.value,
            decision.symbol,
            decision.reason,
            decision.current_stop or 0,
            f"{decision.suggested_stop:.4f}" if decision.suggested_stop else "N/A",
            f"{decision.suggested_exit_price:.4f}" if decision.suggested_exit_price else "N/A",
            decision.current_price or 0,
            decision.regime or "unknown",
            f"{decision.sentiment_score:.2f}" if decision.sentiment_score is not None else "N/A",
            decision.rsi or 0,
            decision.macd_signal or "unknown",
            decision.age_minutes or 0,
            decision.pnl_r or 0,
            decision.is_day_trade,
            decision.age_gate_met,
            decision.r_override_used,
        )

        if self.engine.db_logger:
            try:
                self.engine.db_logger.log_strategy_decision(
                    strategy="active_open_desk",
                    symbol=decision.symbol,
                    action=f"shadow_{decision.action.value.lower()}",
                    reason=decision.reason,
                    extra_data=decision.to_dict(),
                )
            except Exception:
                pass

    def get_last_decisions(self) -> Dict[str, DeskDecision]:
        """Return the last decision for each symbol (for testing/observability)."""
        return dict(self._last_decisions)


def should_start_desk() -> bool:
    """Check if the Active Open Desk should be started.

    Used by the engine to gate task registration. Returns True only when
    ENABLE_ACTIVE_OPEN_DESK=True.
    """
    from core.config import Config
    return Config().ENABLE_ACTIVE_OPEN_DESK
