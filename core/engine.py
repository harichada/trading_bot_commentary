"""Trading Engine with Commentary - Core orchestrator for the trading bot."""

import asyncio
import json
import logging
import os
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from core.models import (TradingMode, CommentaryType, SignalType, NewsImpact,
                         TradingSignal, Position, MarketData, NewsItem)
from core.config import Config, config, config_manager, logger, TradingLossBreaker
from core.commentary import TradingCommentary, CommentarySystem
from core.brain import TradingBrain
from core.websocket_manager import ConnectionManager
from core.position_state import (
    PositionState, ExitRule, ALLOWED_EXITS,
    is_exit_rule_allowed, can_transition, try_transition,
    infer_state_from_legacy_flags,
)
from core.quote_cache import QuoteCache, CachedQuote
from core.task_supervisor import TaskSupervisor, TaskPriority, SupervisorState
from core.schwab_stream import SchwabQuoteStream, STREAMING_AVAILABLE
from core.price_book import PriceBook, PriceObject
from core.calculations import CalculationEngine, PnLResult
from analysis.anomaly import AnomalyDetector, DataValidator
from analysis.behavioral import BehavioralAnalyzer
from analysis.exit_managers import AdvancedExitManager, DynamicExitManager
from analysis.alternative_data import AlternativeDataIntegrator, MarketNeutralStrategies
from analysis.technical import TechnicalAnalyzerWithCommentary
from analysis.screener import StockScreener
from ml.predictor import MLPredictorWithCommentary
from risk.manager import RiskManagerWithCommentary
from risk.backtest import PerformanceAnalyzer
from strategies.builtin import (BreakoutStrategyWithCommentary,
                                MeanReversionStrategyWithCommentary,
                                MomentumStrategyWithCommentary)
from strategies.news_strategy import FreeNewsSignalStrategy
from data_providers.realtime import RealTimeDataProvider, DummyDataProvider
from data_providers.schwab import SchwabDataProvider

from trading_exceptions import *
from circuit_breaker import api_circuit_breaker, order_circuit_breaker
from error_recovery import ErrorRecoveryManager


def log_signal(*args, **kwargs): pass
def log_decision(*args, **kwargs): pass
def log_trade(*args, **kwargs): pass
def log_system_event(*args, **kwargs): pass
def log_pnl(*args, **kwargs): pass
def log_error(*args, **kwargs): pass
def log_performance(*args, **kwargs): pass

# Schwab SDK imports
try:
    from schwab import auth, client as schwab_client_module
    from schwab.orders.equities import equity_buy_market, equity_sell_market, equity_buy_limit, equity_sell_limit
    from schwab.orders.common import Duration, Session
    SCHWAB_AVAILABLE = True
except ImportError:
    SCHWAB_AVAILABLE = False


class TradingEngineWithCommentary:
    """Trading engine that explains its decisions in real-time"""
    
    def __init__(self, mode: TradingMode = TradingMode.SIMULATION_WITH_COMMENTARY, connection_manager=None):
        self.mode = mode
        self.connection_manager = connection_manager  # Store connection manager for broadcasting
        self.schwab_client = None
        self.data_provider = None
        self.positions = {}
        
        # Trading hours configuration from config file
        config = Config()
        self.allow_premarket = config.manager.get('trading.extended_hours.allow_premarket', False)
        self.allow_afterhours = config.manager.get('trading.extended_hours.allow_afterhours', False)
        self.use_limit_orders_extended_hours = config.manager.get('trading.extended_hours.use_limit_orders', True)
        self.simulated_positions = {}
        self.trade_history = []
        self.is_running = False
        # v-schwab-positions-cache-2026-04-30: per-position P&L on the
        # dashboard for LIVE trades comes straight from Schwab (their
        # numbers are authoritative — broker is the source of truth for
        # real money). The engine refreshes this cache from the periodic
        # Schwab sync; the WS endpoint reads it when building the live
        # positions payload, no recalculation.
        self._schwab_positions_cache: list[dict] = []
        # v-loop-decoupling-2026-04-30 (Phase 2): three-task model.
        # quote_cache holds last-observed price per symbol; position_loop
        # reads from it. _supervisor manages the three task lifecycles.
        # _quote_fetch_sem caps concurrent broker quote calls.
        self.quote_cache: QuoteCache = QuoteCache()
        # v-pricebook-2026-05-01: canonical mark store. Singleton-
        # installed so Position.current_price / unrealized_pnl
        # properties find it via PriceBook.instance().
        # v-stream-only-2026-05-01: stale_threshold raised to 600s
        # (10 min). With REST fallback removed, "stale" no longer means
        # "this symbol needs a top-up." It means "the stream has been
        # silent so long that we should treat the mark as suspect."
        # 10 minutes is conservative — if Schwab streamed AAPL at
        # 14:00 and 14:08, that's still a meaningful price for our
        # 5-min-bar bot. Anything older than 10 min is genuinely
        # questionable (held position over a session boundary or stream
        # disconnect we missed).
        self.price_book: PriceBook = PriceBook(stale_threshold_sec=600.0)
        PriceBook.install(self.price_book)
        self._supervisor: Optional[TaskSupervisor] = None
        self._quote_fetch_sem: Optional[asyncio.Semaphore] = None
        # Set by quote_streamer when it observes itself stale > HARD threshold.
        # analysis_loop checks this gate before signalling new entries.
        self._quote_streamer_healthy: bool = True
        # v-schwab-stream-2026-05-01 (Phase 4a): real-time tick stream.
        # When healthy, ticks land in QuoteCache via websocket and the
        # polling streamer (Phase 2) becomes a no-op fallback. When the
        # stream goes unhealthy (5 consecutive connect failures or 60s
        # tick silence during RTH), the polling streamer auto-resumes.
        self._schwab_quote_stream: Optional[SchwabQuoteStream] = None
        # Optional UI loop owner for cross-thread websocket broadcasts.
        self._ws_broadcast_loop = None
        self.account_id = None
        self.account_hash = None
        self._last_bp_check = datetime.now() - timedelta(minutes=5)  # Force initial check
        # Commentary system
        self.commentary = CommentarySystem()
        # v-news-veto-tracker-2026-04-28: back-reference so strategies can
        # reach engine.db_logger via their commentary instance for shadow-
        # tracking writes (avoid passing engine into every strategy ctor).
        self.commentary.engine_ref = self
        # Initialize brain BEFORE ml_predictor
        self.brain = TradingBrain()
        # Error recovery manager
        self.error_recovery = ErrorRecoveryManager()
        # Initialize components
        self._init_schwab_client()
        # Order tracking
        self.pending_orders = {}
        self.order_id_to_symbol = {}
        self.last_order_check = datetime.now()

        # v-classifier-runtime-2026-05-13: side-classifier shadow hook.
        # Lazy-built — the import lives inside this conditional so a
        # torch-less environment can start the bot without dragging in
        # the trained classifier subpackage. When the flag is False
        # (default), ``self._classifier_runtime`` is None and the
        # signal_router skips the shadow_evaluate call entirely.
        self._classifier_runtime = None
        try:
            if Config().SIDE_CLASSIFIER_SHADOW_MODE:
                from core.classifier.runtime import ClassifierRuntime
                self._classifier_runtime = ClassifierRuntime(
                    enabled=True,
                    model_path=Config().SIDE_CLASSIFIER_MODEL_PATH,
                    long_threshold=Config().SIDE_CLASSIFIER_LONG_THRESHOLD,
                    short_threshold=Config().SIDE_CLASSIFIER_SHORT_THRESHOLD,
                )
                logger.info(
                    "side_classifier: shadow mode ENABLED (model_path=%s) — "
                    "decisions logged, no gating applied.",
                    Config().SIDE_CLASSIFIER_MODEL_PATH,
                )
        except Exception as _cl_exc:
            logger.warning(
                "side_classifier: failed to initialize shadow runtime: %s. "
                "Continuing without classifier.", _cl_exc,
            )
            self._classifier_runtime = None

        # v-regime-allocator-shadow-2026-06-10 (roadmap P1): regime
        # allocator shadow. Logs which strategy the SPY-trend-efficiency
        # allocator WOULD permit for each routed signal. Pure logging —
        # gates nothing. Built lazily; SPY daily closes come from the
        # data provider, fetched at most every 4h (allocator caches).
        self._regime_allocator = None
        try:
            if Config().ENABLE_REGIME_ALLOCATOR_SHADOW:
                from allocators.regime_allocator import RegimeAllocatorShadow

                def _spy_daily_closes():
                    df = self.data_provider.get_market_data(
                        "SPY", period_type="month", period=2,
                        frequency_type="daily", frequency=1,
                    )
                    col = "Close" if "Close" in df.columns else "close"
                    return df[col].dropna().tolist()

                self._regime_allocator = RegimeAllocatorShadow(
                    fetch_daily_closes=_spy_daily_closes,
                    threshold=Config().REGIME_ALLOCATOR_ER_THRESHOLD,
                )
                logger.info(
                    "regime_allocator: shadow mode ENABLED (threshold=%.2f)"
                    " — allocations logged, no gating applied.",
                    Config().REGIME_ALLOCATOR_ER_THRESHOLD,
                )
        except Exception as _ra_exc:
            logger.warning(
                "regime_allocator: failed to initialize shadow: %s. "
                "Continuing without it.", _ra_exc,
            )
            self._regime_allocator = None

        # v-conviction-sizer-shadow-2026-06-11: conviction-weighted
        # sizing shadow (operator directive: "real solid confirmation
        # → more risk"). Historical validation: meta-proba buckets ran
        # PF 0.44 / 3.01 / 5.02 across 89 trades. Logs would-be size
        # multipliers only — sizing is unchanged until the live shadow
        # reproduces the gradient (promotion gate in the roadmap).
        self._conviction_sizer = None
        try:
            from sizing.conviction_sizer import ConvictionSizerShadow
            self._conviction_sizer = ConvictionSizerShadow()
            logger.info("conviction_sizer: shadow enabled — multipliers "
                        "logged, sizing unchanged.")
        except Exception as _cs_exc:
            logger.warning("conviction_sizer: failed to init: %s", _cs_exc)

        # v-symbol-intel-2026-06-10: per-symbol realtime intelligence
        # hub. The analysis loop pushes a composite view per evaluated
        # symbol; the dashboard WS payload reads snapshot(); the NDJSON
        # ledger doubles as Composite View Phase 1 evidence. Read-only
        # everywhere — affects no trading decision.
        self._symbol_intel_hub = None
        try:
            from core.symbol_intel import SymbolIntelHub
            self._symbol_intel_hub = SymbolIntelHub()
            logger.info("symbol_intel: hub enabled (panel + Phase 1 ledger)")
        except Exception as _si_exc:
            logger.warning("symbol_intel: failed to initialize: %s", _si_exc)
        
        # Risk manager
        self.risk_manager = RiskManagerWithCommentary(
            account_balance=100000,
            commentary_system=self.commentary
        )
        
        # Technical analyzer
        self.technical_analyzer = TechnicalAnalyzerWithCommentary(
            commentary_system=self.commentary
        )
        # ML predictor
        from ml.predictor import MLPredictorWithCommentary
        self.ml_predictor = MLPredictorWithCommentary(
            commentary_system=self.commentary
        )
        self.ml_predictor.set_brain(self.brain)
        # Meta-model (shadow mode): evaluates mean-reversion signals and
        # logs a trust score without affecting trade decisions. Absent
        # model file → feature silently disabled.
        self.meta_inference = None
        try:
            from ml.meta_inference import MetaInference
            meta_path = Path("ml_meta_model.pkl")
            if meta_path.exists():
                self.meta_inference = MetaInference.load(meta_path)
                logger.info(
                    "meta_shadow_enabled primary=%s features=%d",
                    self.meta_inference.primary_strategy,
                    len(self.meta_inference.feature_names),
                )
        except (FileNotFoundError, KeyError, ImportError) as exc:
            # Expected failure modes: missing file, missing bundle keys,
            # missing dependency. Log once and continue — bot still trades.
            logger.warning("meta_shadow_disabled reason=%s", exc)
        except Exception:
            # Anything else (corrupt bundle, import failure of a subclass,
            # etc.) logs with traceback so a silent corruption is loud.
            logger.exception("meta_shadow_disabled_unexpected")
        # Scale-out + trailing stop manager
        from analysis.scale_trail_manager import ScaleTrailManager
        self.scale_trail = ScaleTrailManager()
        # Postgres audit logger — every decision + trade persisted to DB
        self.db_logger = None
        try:
            from data_providers.db_logger import DbLogger
            self.db_logger = DbLogger()
            logger.info("db_logger_enabled")
        except Exception as exc:
            logger.warning("db_logger_disabled reason=%s", exc)
        # Stock screener
        self.screener = StockScreener(
            self.schwab_client,
            self.commentary
        ) if self.schwab_client else None
        
        # Dynamic watchlist
        self.dynamic_watchlist = ['NVDA', 'TSLA', 'PLTR']  # Default symbols
        self.last_screener_run = None
        # Trading strategies
        # v-oversold-v2-2026-05-19: conditional swap of the legacy
        # MeanReversionStrategyWithCommentary for the new
        # OversoldBounceV2Strategy. Controlled by
        # Config().USE_OVERSOLD_BOUNCE_V2 (yaml: trading.use_oversold_bounce_v2,
        # default True). Old class preserved for one-flag rollback.
        if Config().USE_OVERSOLD_BOUNCE_V2:
            from strategies.oversold_bounce_v2 import OversoldBounceV2Strategy
            _mean_rev = OversoldBounceV2Strategy(self.commentary)
            logger.info("strategy_swap: using OversoldBounceV2Strategy (v2)")
        else:
            _mean_rev = MeanReversionStrategyWithCommentary(self.commentary)
            logger.info("strategy_swap: using legacy MeanReversionStrategyWithCommentary")
        self.strategies = [
            BreakoutStrategyWithCommentary(self.commentary),
            _mean_rev,
        ]
        # Momentum disabled 2026-04-15: backtest PF 0.78–0.83 across 1/5/15-min
        # timeframes — consistently losing. Re-enable via ENABLE_MOMENTUM=1.
        if os.environ.get("ENABLE_MOMENTUM", "0") == "1":
            self.strategies.append(MomentumStrategyWithCommentary(self.commentary))
        self.strategies.append(FreeNewsSignalStrategy(self.commentary))
        
        # v-feature-snapshot-2026-09-09: wire engine ref on all strategies
        # so they can emit DecisionSnapshots via db_logger
        for strat in self.strategies:
            strat._engine_ref = self
        
	    # Initialize brain and exit manager
        self.brain = TradingBrain()
        # v-brain-session-reset-2026-05-19: pull greed_level/fear_level 60%
        # toward neutral on each startup so yesterday's accumulated state
        # doesn't lock the bot into all-day veto mode. Fixes the 5/19
        # incident where greed_level was stuck at 0.8 after morning
        # RKLB win and vetoed every news signal for the rest of the day.
        try:
            self.brain.reset_for_new_session()
        except Exception as _ex:
            logger.warning("brain.reset_for_new_session failed: %s", _ex)
        self.exit_manager = DynamicExitManager(self.brain, self.commentary)

	    # Load previous state
        # v-ownership-survives-restart-2026-06-10: default before
        # _load_state so a missing/corrupt state file leaves an empty
        # dict, not an AttributeError in the Schwab sync.
        self._saved_positions_meta = {}
        self._load_state()

        # Connection manager for WebSocket
        self.connection_manager = connection_manager or ConnectionManager()
        # Confirmation settings
        self.require_confirmations = Config().REQUIRE_CLOSE_CONFIRMATION
        self.confirm_only_losses = Config().CONFIRM_ONLY_LOSSES
        self.confirm_threshold_percent = Config().CONFIRM_THRESHOLD_PERCENT
        # v-live-exit-ladder-2026-06-11: was hardcoded True since the
        # manual-supervision era, silently disabling breakeven/trail/
        # tp/stop enforcement for ALL live positions — they only ever
        # had their static OCO. Now config-driven, default False.
        self.manual_close_only = Config().MANUAL_CLOSE_ONLY

        # Market state tracking
        self.market_state = {
            'vix': 0,
            'market_trend': 'neutral',
            'breadth': {},
            'economic_events': []
        }
        
        # Enhanced error recovery and circuit breakers
        self.circuit_breaker = TradingLossBreaker()
        self.error_counts = {}
        self.last_error_time = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5
        
        # Performance tracking
        self.performance_analyzer = PerformanceAnalyzer(config_manager.config)
        
        # Advanced components integration
        self.behavioral_analyzer = BehavioralAnalyzer(self.commentary)
        self.data_validator = DataValidator(self.commentary)
        self.advanced_exit_manager = AdvancedExitManager(self.commentary)
        self.alternative_data_integrator = AlternativeDataIntegrator(self.commentary)
        self.market_neutral_strategies = MarketNeutralStrategies(self.commentary)

        self.pro_trading_wrapper = None


        # Add initial commentary
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.MARKET_ANALYSIS,
            symbol=None,
            title="🚀 Trading Bot Initialized",
            message=f"Starting in {mode.value} mode with enhanced error recovery and circuit breakers. I'll explain my thinking process as I analyze markets.",
            importance=10
        ))
        self._cached_buying_power = 0
        
        # Track loss alerts to avoid spamming
        self.loss_alert_tracker = {}  # {symbol: {'last_alert_time': datetime, 'last_loss_pct': float}}
        self._last_bp_check = datetime.now()
        self.day_trades_count = 0
        self.performance_metrics = {
            'slippage': [],
            'fill_rate': {'attempts': 0, 'fills': 0}
        }

    @property
    def auto_close_disabled(self) -> bool:
        """True when the bot must not auto-close any positions.

        Simulation always auto-closes on stop-loss / take-profit —
        otherwise the simulated P&L stops reflecting what live
        execution would produce, which defeats the purpose of sim.
        The manual-close-only safety applies only to LIVE mode,
        where it prevents the bot from touching real broker positions.
        """
        if self.mode != TradingMode.LIVE:
            return False
        return bool(self.manual_close_only)

    def _indicator_thesis_supportive(self, position, indicators: dict) -> tuple[bool, str]:
        """v-thesis-revalidate-2026-04-28: is the indicator picture still
        supportive of the position's direction? Returns (ok, reason).

        For a LONG: MACD>=signal AND RSI>=45 AND close>=SMA50
        For a SHORT: MACD<=signal AND RSI<=55 AND close<=SMA50
        Missing indicators are treated as 'supportive' (don't close on
        absence of data — could be a new symbol or stale feed).
        """
        try:
            macd = float(indicators.get('macd', 0) or 0)
            macd_sig = float(indicators.get('macd_signal', 0) or 0)
            rsi = float(indicators.get('rsi', 50) or 50)
            sma_50 = float(indicators.get('sma_50', 0) or 0)
            close = float(indicators.get('close', position.current_price) or position.current_price)
        except (TypeError, ValueError):
            return True, "indicators_unparseable_treat_as_ok"

        if position.side == 'long':
            if macd < macd_sig:
                return False, f"macd_below_signal({macd:.4f}<{macd_sig:.4f})"
            if rsi < 45:
                return False, f"rsi_weak({rsi:.1f}<45)"
            if sma_50 > 0 and close < sma_50:
                return False, f"price_below_sma50({close:.2f}<{sma_50:.2f})"
            return True, "long_indicators_ok"
        else:  # short
            if macd > macd_sig:
                return False, f"macd_above_signal({macd:.4f}>{macd_sig:.4f})"
            if rsi > 55:
                return False, f"rsi_strong({rsi:.1f}>55)"
            if sma_50 > 0 and close > sma_50:
                return False, f"price_above_sma50({close:.2f}>{sma_50:.2f})"
            return True, "short_indicators_ok"

    async def _revalidate_thesis(
        self, symbol: str, position, current_price: float, indicators: dict
    ) -> None:
        """v-thesis-revalidate-2026-04-28: re-verify news + indicators on
        positions in the limbo zone after their first 30 min. Closes only
        when BOTH theses have broken; logs every decision for diagnostics.

        Gates checked BEFORE this is called (in _manage_positions_with_commentary):
          - position is auto-managed
          - breakeven_stop / 1R partial / trail / proactive_exit all skipped this tick
        Gates checked HERE:
          - thesis revalidation enabled
          - position age >= THESIS_REVALIDATION_AGE_MIN
          - last_revalidation_at >= THESIS_REVALIDATION_INTERVAL_MIN ago
          - |current_R| < THESIS_REVALIDATION_LIMBO_R
          - breakeven_lifted == False (winners already protected, leave alone)
        """
        cfg = Config()
        if not cfg.ENABLE_THESIS_REVALIDATION:
            return
        if getattr(position, 'breakeven_lifted', False):
            return  # winner already protected; don't second-guess

        # Age + cadence
        age_min = (datetime.now() - position.entry_time).total_seconds() / 60.0
        if age_min < cfg.THESIS_REVALIDATION_AGE_MIN:
            return
        last_str = getattr(position, 'last_revalidation_at', None)
        if last_str:
            try:
                last = datetime.fromisoformat(last_str)
                if (datetime.now() - last).total_seconds() / 60.0 < cfg.THESIS_REVALIDATION_INTERVAL_MIN:
                    return
            except (TypeError, ValueError):
                pass  # bad timestamp → re-check now

        # Limbo zone: outside ±limbo_R, existing exits handle it
        original_stop = position.original_stop or position.stop_loss
        stop_distance = abs(position.entry_price - original_stop)
        if stop_distance <= 0:
            return
        if position.side == 'long':
            current_r = (current_price - position.entry_price) / stop_distance
        else:
            current_r = (position.entry_price - current_price) / stop_distance
        if abs(current_r) > cfg.THESIS_REVALIDATION_LIMBO_R:
            return

        # Stamp last_revalidation_at NOW so failed paths (verifier exception
        # etc.) still respect the cadence.
        position.last_revalidation_at = datetime.now().isoformat()

        # --- News thesis ---
        strategy = (getattr(position, 'reasoning', {}) or {}).get('strategy', '')
        is_news_trade = 'news' in strategy.lower()
        news_ok = True
        news_detail = "non_news_strategy_skip"
        if is_news_trade:
            try:
                # Reuse the news_strategy verifier — it's already initialised
                # in the news_strategy instance. Walk the strategies list to
                # find it. Cheap; happens at most once per position per 15 min.
                verifier = None
                for s in (getattr(self, 'strategies', None) or []):
                    if hasattr(s, 'verifier'):
                        verifier = s.verifier
                        break
                if verifier is None:
                    news_detail = "verifier_unavailable"
                else:
                    expected_dir = 1 if position.side == 'long' else -1
                    v = await verifier.verify(symbol, expected_dir)
                    news_ok = v.is_verified
                    news_detail = (
                        f"src={v.source} fresh={v.fresh_count} "
                        f"avg={v.avg_fresh_sentiment:+.3f} reason={v.reason}"
                    )
            except Exception as exc:
                # Verifier failure is non-fatal — treat as ok (don't close
                # on inability to confirm; would over-trigger on outages).
                news_ok = True
                news_detail = f"verifier_error_treat_as_ok:{str(exc)[:60]}"

        # --- Indicator thesis ---
        ind_ok, ind_detail = self._indicator_thesis_supportive(position, indicators)

        # --- Decision ---
        thesis_broken = (not news_ok) and (not ind_ok)
        action_str = "would_close" if thesis_broken else "keep_open"
        if thesis_broken and not cfg.THESIS_REVALIDATION_SHADOW_MODE:
            action_str = "closed"

        # Always audit
        self._audit("thesis_revalidate", symbol, action_str,
                    "thesis_broken" if thesis_broken else "thesis_intact",
                    age_min=round(age_min, 1),
                    current_r=round(current_r, 3),
                    side=position.side,
                    is_news_trade=is_news_trade,
                    news_ok=news_ok, news=news_detail,
                    ind_ok=ind_ok, ind=ind_detail,
                    shadow=cfg.THESIS_REVALIDATION_SHADOW_MODE)

        if thesis_broken:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING if cfg.THESIS_REVALIDATION_SHADOW_MODE
                else CommentaryType.DECISION,
                symbol=symbol,
                title=("🔍 [SHADOW] Thesis Broken — Would Close"
                       if cfg.THESIS_REVALIDATION_SHADOW_MODE
                       else "🚪 Thesis Broken — Closing"),
                message=(
                    f"At {age_min:.0f} min ({current_r:+.2f}R), both news and "
                    f"indicators turned against the position.\n"
                    f"  News:       {news_detail}\n"
                    f"  Indicators: {ind_detail}\n"
                    + ("(shadow mode — position kept open for now)"
                       if cfg.THESIS_REVALIDATION_SHADOW_MODE else "")
                ),
                importance=8,
            ))
            if not cfg.THESIS_REVALIDATION_SHADOW_MODE:
                await self._close_position_with_commentary(
                    position, "thesis_revalidation_broken"
                )

    async def _check_news_thesis_flip(
        self, symbol: str, position, current_price: float
    ) -> bool:
        """v-newsbus-gates-2026-09-09: check if fresh news has flipped against position.
        
        Uses the NewsBus directly (no external API call) to detect when
        sentiment has reversed against an open news-driven position.
        
        Returns True if the position should be flagged for exit (thesis flipped).
        Returns False otherwise (no flip, non-news trade, or feature disabled).
        
        Gates:
          - ENABLE_NEWS_THESIS_EXIT must be True
          - Position must be from a news strategy
          - Fresh news (< NEWS_GATE_MAX_AGE_SEC) must exist
          - Sentiment must have flipped past NEWS_THESIS_EXIT_SENTIMENT_FLIP
        """
        cfg = Config()
        if not cfg.ENABLE_NEWS_THESIS_EXIT:
            return False
        
        # Only applies to news-driven positions
        strategy = (getattr(position, 'reasoning', {}) or {}).get('strategy', '')
        if 'news' not in strategy.lower():
            return False
        
        # Need a NewsBus to check
        if self._news_bus is None:
            return False
        
        try:
            # Get aggregate sentiment from fresh news
            agg = await self._news_bus.get_aggregate_sentiment(
                symbol,
                max_age_sec=cfg.NEWS_GATE_MAX_AGE_SEC,
            )
            
            if agg['article_count'] == 0:
                return False  # No fresh news to evaluate
            
            avg_sentiment = agg['avg_sentiment']
            flip_threshold = cfg.NEWS_THESIS_EXIT_SENTIMENT_FLIP
            
            # Detect flip: long needs bearish news, short needs bullish news
            if position.side == 'long' and avg_sentiment < -flip_threshold:
                self._audit(
                    "news_thesis_flip", symbol, "flip_detected",
                    "bearish_news_on_long",
                    side=position.side,
                    avg_sentiment=round(avg_sentiment, 3),
                    flip_threshold=flip_threshold,
                    article_count=agg['article_count'],
                )
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"⚠️ News Thesis Flip — {symbol} LONG",
                    message=(
                        f"Fresh news sentiment has turned bearish ({avg_sentiment:+.2f}) "
                        f"against your LONG position.\n"
                        f"  Articles: {agg['article_count']}\n"
                        f"  Flip threshold: {-flip_threshold:+.2f}\n"
                        "Position flagged for early exit."
                    ),
                    importance=8,
                ))
                return True
            
            if position.side == 'short' and avg_sentiment > flip_threshold:
                self._audit(
                    "news_thesis_flip", symbol, "flip_detected",
                    "bullish_news_on_short",
                    side=position.side,
                    avg_sentiment=round(avg_sentiment, 3),
                    flip_threshold=flip_threshold,
                    article_count=agg['article_count'],
                )
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"⚠️ News Thesis Flip — {symbol} SHORT",
                    message=(
                        f"Fresh news sentiment has turned bullish ({avg_sentiment:+.2f}) "
                        f"against your SHORT position.\n"
                        f"  Articles: {agg['article_count']}\n"
                        f"  Flip threshold: {flip_threshold:+.2f}\n"
                        "Position flagged for early exit."
                    ),
                    importance=8,
                ))
                return True
            
            return False
        except Exception as exc:
            logger.debug(f"news_thesis_flip check error for {symbol}: {exc}")
            return False

    def _is_auto_managed(self, position) -> bool:
        """v-managed-by-bot-2026-04-28: per-position auto-management gate.

        Returns True if the bot may auto-close / auto-manage this position
        (apply trailing stops, breakeven lift, proactive exit, take profit,
        hard stop). Returns False to leave the position untouched.

        Logic:
          SIM mode  → honour position.managed_by_bot (default True for new
                      bot-opened entries; user can toggle off via dashboard
                      to pause management on a single trade).
          LIVE mode → require BOTH the global manual_close_only flag to be
                      OFF (no global pause) AND position.managed_by_bot=True
                      (this specific trade is bot-managed, not external).

        Long-term flag still wins: any position with is_long_term=True is
        excluded from automation regardless of mode or managed_by_bot.
        """
        if getattr(position, 'is_long_term', False):
            return False
        if not getattr(position, 'managed_by_bot', False):
            return False
        if self.mode == TradingMode.LIVE and self.manual_close_only:
            return False
        return True

    def _audit(self, component: str, symbol, action: str, reason: str, **details) -> None:
        """Structured audit log for engine-level decisions.

        Format (one line per event, shell-greppable):
          engine_decision component=NAME symbol=SYM action=ACTION reason=R k=v k=v

        Every decision gate that accepts, skips, or modifies a trade must
        emit one of these so trading_bot.log is a complete audit trail.
        Also writes to Postgres bot_decisions for SQL queryability.
        """
        kv = " ".join(f"{k}={v}" for k, v in details.items())
        logger.info(
            "engine_decision component=%s symbol=%s action=%s reason=%s mode=%s %s",
            component, symbol or "-", action, reason, self.mode.value, kv,
        )
        # Async DB write — fire-and-forget so it never blocks the loop
        if hasattr(self, 'db_logger') and self.db_logger is not None:
            try:
                asyncio.get_event_loop().create_task(
                    self.db_logger.log_decision(
                        component=component,
                        symbol=symbol or None,
                        action=action,
                        reason=reason,
                        mode=self.mode.value,
                        signal_type=details.get("signal_type"),
                        confidence=details.get("confidence"),
                        strength=details.get("strength"),
                        meta_proba=details.get("proba"),
                        atr=details.get("atr"),
                        stop_distance=details.get("stop_dist") or details.get("stop_distance"),
                        price=details.get("price"),
                        **{k: v for k, v in details.items()
                           if k not in ("signal_type", "confidence", "strength",
                                        "proba", "atr", "stop_dist", "stop_distance", "price")},
                    )
                )
            except Exception:
                pass  # never break trading loop for DB

    def _emit_veto_snapshot(
        self,
        signal,
        strategy_id: str,
        reason: str,
        gate_name: str,
        extra: dict | None = None,
    ) -> None:
        """v-feature-snapshot-2026-09-09: emit DecisionSnapshot on engine-level veto.
        
        Called when a signal is vetoed by the signal router (ML veto, regime gate,
        conviction floor, etc.). Fire-and-forget async write.
        """
        if self.db_logger is None:
            return
        try:
            from core.decision_snapshot import (
                DecisionAction, build_snapshot, is_snapshot_logging_enabled
            )
            if not is_snapshot_logging_enabled():
                return
            
            # Extract what we can from the signal
            indicators = {}
            price = float(getattr(signal, "entry_price", 0) or 0)
            if hasattr(signal, "reasoning") and signal.reasoning:
                indicators = signal.reasoning.copy()
            
            snapshot = build_snapshot(
                symbol=signal.symbol,
                strategy_id=strategy_id,
                action=DecisionAction.VETO,
                reason=reason,
                mode=self.mode.value,
                gate_name=gate_name,
                confidence=float(getattr(signal, "confidence", 0) or 0),
                indicators=indicators,
                price=price,
                would_entry_price=float(getattr(signal, "entry_price", 0) or 0),
                would_stop_loss=float(getattr(signal, "stop_loss", 0) or 0),
                would_take_profit=float(getattr(signal, "take_profit", 0) or 0),
                would_size_shares=int(getattr(signal, "position_size", 0) or 0),
                extra=extra,
            )
            
            try:
                asyncio.get_event_loop().create_task(
                    self.db_logger.log_decision_snapshot(snapshot)
                )
            except RuntimeError:
                pass
        except Exception as exc:
            logger.debug("veto_snapshot_emit_error: %s", exc)

    # Add to rest brain - caution state:
    def reset_brain_state(self):
        """Reset brain to neutral confident state"""
        self.brain.emotional_state = {
            'confidence': 0.7,
            'risk_appetite': 0.6,
            'patience': 0.8,
            'fear_level': 0.2,
            'greed_level': 0.3,
            'last_update': datetime.now().isoformat()
        }
        # Clear recent bad memories
        self.brain.memories = self.brain.memories[:-10] if len(self.brain.memories) > 10 else []
        self.brain.consecutive_losses = 0
        self.brain.save_memories()
        
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.PSYCHOLOGY,
            symbol=None,
            title="🧠 Brain Reset",
            message="Cleared recent negative memories. Fresh start!",
            importance=8
        ))

    def _load_state(self):
        """Load previous trading state including simulated positions."""
        state_file = Path("trading_state.json")
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.trade_history = state.get('trade_history', [])
                    # v-consec-loss-daily-reset-2026-05-21: reset
                    # consecutive_losses counter at session boundary.
                    # Previously it persisted across days, so a single
                    # losing day's count would carry forward and could
                    # trip the MAX_CONSECUTIVE_LOSSES gate without giving
                    # the next session a fresh start. Compare last_save's
                    # calendar date (ET) to today; if different, reset.
                    _loaded_cl = state.get('consecutive_losses', 0)
                    _last_save = state.get('last_save', '')
                    _should_reset = False
                    try:
                        from datetime import datetime as _dt
                        _today_date = _dt.now().date()
                        if _last_save:
                            _save_dt = _dt.fromisoformat(str(_last_save).replace('Z', ''))
                            _save_date = _save_dt.date()
                            if _save_date < _today_date:
                                _should_reset = True
                    except Exception as _ex:
                        logger.debug("consec_loss daily-reset date parse failed: %s", _ex)
                    if _should_reset and _loaded_cl > 0:
                        logger.info(
                            "consecutive_losses session_reset: %d -> 0 (last_save=%s)",
                            _loaded_cl, _last_save,
                        )
                        self.risk_manager.consecutive_losses = 0
                    else:
                        self.risk_manager.consecutive_losses = _loaded_cl
                    # v-consec-loss-daily-reset-2026-05-21: also load
                    # last_loss_date so the runtime daily-rollover check
                    # in risk/manager.py works after restart.
                    _lld = state.get('last_loss_date')
                    if _lld:
                        try:
                            from datetime import date as _date
                            self.risk_manager.last_loss_date = _date.fromisoformat(str(_lld))
                        except Exception:
                            self.risk_manager.last_loss_date = None

                    # Load real position long-term flags
                    positions_data = state.get('positions_data', {})
                    for symbol, pos_data in positions_data.items():
                        if symbol in self.positions:
                            self.positions[symbol].is_long_term = pos_data.get('is_long_term', False)
                    # v-ownership-survives-restart-2026-06-10: stash the
                    # full saved records. At startup self.positions is
                    # empty, so the Schwab sync re-discovers every
                    # position — it consults this dict to restore
                    # bot ownership instead of defaulting to external.
                    self._saved_positions_meta = dict(positions_data)

                    # Restore simulated positions so they survive restarts
                    sim_data = state.get('simulated_positions', {})
                    for symbol, pd in sim_data.items():
                        if symbol in self.simulated_positions:
                            continue  # don't overwrite if already loaded
                        try:
                            entry_time = datetime.fromisoformat(pd['entry_time'])
                        except (KeyError, ValueError):
                            entry_time = datetime.now()
                        pos = Position(
                            symbol=pd.get('symbol', symbol),
                            entry_price=pd['entry_price'],
                            current_price=pd.get('current_price', pd['entry_price']),
                            quantity=pd['quantity'],
                            side=pd.get('side', 'long'),
                            stop_loss=pd['stop_loss'],
                            take_profit=pd['take_profit'],
                            entry_time=entry_time,
                            unrealized_pnl=pd.get('unrealized_pnl', 0),
                            reasoning=pd.get('reasoning', {}),
                            is_long_term=pd.get('is_long_term', False),
                            scaled_out=pd.get('scaled_out', False),
                            original_stop=pd.get('original_stop'),
                            trailing_stop=pd.get('trailing_stop'),
                            mode=pd.get('mode', 'simulation'),
                            peak_favorable_r=pd.get('peak_favorable_r', 0.0),
                            breakeven_lifted=pd.get('breakeven_lifted', False),
                            managed_by_bot=pd.get('managed_by_bot', True),  # restored sim positions: bot-managed by default
                            last_revalidation_at=pd.get('last_revalidation_at'),
                            # v-fsm-state-migration-2026-04-30: infer the
                            # right FSM state from existing flags. A
                            # position with breakeven_lifted=True must
                            # land in AT_BREAKEVEN, not LIVE — otherwise
                            # the trailing stop never engages and we
                            # leave money on the table (MRVL bug today).
                            state=infer_state_from_legacy_flags(
                                pd.get('state'),
                                pd.get('breakeven_lifted', False),
                                pd.get('scaled_out', False),
                                pd.get('trailing_stop'),
                            ).value,
                            state_reason=pd.get('state_reason') or 'restored_with_state_inferred',
                            state_changed_at=pd.get('state_changed_at'),
                            exiting_started_at=pd.get('exiting_started_at'),
                            zombie_reason=pd.get('zombie_reason'),
                        )
                        self.simulated_positions[symbol] = pos
                    if sim_data:
                        logger.info(
                            "restored_sim_positions count=%d symbols=%s",
                            len(sim_data), list(sim_data.keys()),
                        )

                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.PSYCHOLOGY,
                        symbol=None,
                        title="☕ Good Morning!",
                        message=f"Back at the desk. Yesterday's P&L: ${state.get('daily_pnl', 0):.2f}. "
                               f"Let's see what the market has for us today.",
                        importance=8
                    ))
            except Exception as e:
                logger.error(f"Error loading state: {e}")
    
    @order_circuit_breaker
    async def _execute_real_trade(self, signal) -> bool:
        """Execute real trade through Schwab with proper error handling"""
        
        if not self.schwab_client or not self.account_id:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"❌ Trade Execution Failed",
                message="Schwab client not properly initialized",
                importance=10
            ))
            raise AuthenticationException("Schwab client not initialized - check credentials")
        
        try:
            # Validate OCO prices first
            # Check if this is a short position (SELL signal without existing position)
            is_short = signal.signal_type == SignalType.SELL and signal.symbol not in self.positions
            if not self._validate_oco_prices(signal.symbol, signal.stop_loss, signal.take_profit, is_short):
                return False
            
            # Check buying power
            quote = self.data_provider.get_quote(signal.symbol)
            if not quote:
                raise StaleDataException(60, 60)
                
            price = quote.get('ask', signal.entry_price) if signal.signal_type == SignalType.BUY else quote.get('bid', signal.entry_price)
            required = signal.position_size * price
            
            # Check available funds
            if self.risk_manager.buying_power < required:
                raise InsufficientFundsException(required, self.risk_manager.buying_power)
                                    
            from schwab.orders.equities import (equity_buy_market, equity_sell_market, 
                                                equity_sell_short_market, equity_buy_limit, 
                                                equity_sell_limit, equity_sell_short_limit)
            from schwab.orders.common import Duration, Session
            
            # Check if we should use limit orders
            use_limit, reason = self.should_use_limit_order()
            is_regular, session = self.is_market_hours()
            
            # Block trading if market is closed and extended hours not allowed
            if session == 'closed':
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"🚫 Market Closed",
                    message=f"Cannot place order - market is closed. Will retry during market hours.",
                    importance=8
                ))
                return None
            elif session == 'premarket' and not self.allow_premarket:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"🚫 Premarket Trading Disabled",
                    message=f"Order blocked - premarket trading is disabled for safety. Enable in settings if needed.",
                    importance=8
                ))
                return None
            elif session == 'afterhours' and not self.allow_afterhours:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"🚫 After-Hours Trading Disabled",
                    message=f"Order blocked - after-hours trading is disabled for safety. Enable in settings if needed.",
                    importance=8
                ))
                return None
            
            # Get current price for limit orders
            current_price = signal.entry_price
            
            # Build order based on type and market hours
            if use_limit or self.use_limit_orders_extended_hours:
                # Use limit orders during extended hours
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.INFO,
                    symbol=signal.symbol,
                    title=f"⚠️ Using Limit Order",
                    message=f"{reason}. Placing limit order at ${current_price:.2f}",
                    importance=7
                ))
                
                slippage = Config().LIMIT_ORDER_SLIPPAGE
                if signal.signal_type == SignalType.BUY:
                    limit_price = current_price * (1 + slippage)
                    order_builder = equity_buy_limit(signal.symbol, signal.position_size, limit_price)
                else:  # SELL/SHORT
                    limit_price = current_price * (1 - slippage)
                    try:
                        order_builder = equity_sell_short_limit(signal.symbol, signal.position_size, limit_price)
                    except AttributeError:
                        order_builder = equity_sell_limit(signal.symbol, signal.position_size, limit_price)
                        order_builder.set_instruction('SELL_SHORT')
            else:
                # Use market orders during regular hours
                if signal.signal_type == SignalType.BUY:
                    order_builder = equity_buy_market(signal.symbol, signal.position_size)
                else:  # SHORT
                    try:
                        order_builder = equity_sell_short_market(signal.symbol, signal.position_size)
                    except AttributeError:
                        order_builder = equity_sell_market(signal.symbol, signal.position_size)
                        order_builder.set_instruction('SELL_SHORT')
            
            # Set order parameters
            order_builder.set_duration(Duration.DAY)
            
            # Set session based on market hours
            if session == 'premarket' or session == 'afterhours':
                order_builder.set_session(Session.EXTENDED)
            else:
                order_builder.set_session(Session.NORMAL)
            
            # Build the order
            order = order_builder.build()
            
            # Place the order
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=signal.symbol,
                title=f"📤 Placing Live Order",
                message=f"Sending {'BUY' if signal.signal_type == SignalType.BUY else 'SELL'} order for {signal.position_size} shares",
                importance=9
            ))
            
            response = self.schwab_client.place_order(self.account_hash, order)

            if response.status_code in [200, 201]:
                # Extract order ID from Location header
                order_id = response.headers.get('Location', '').split('/')[-1]

                logger.info(
                    f"Order placed: {signal.signal_type.value} {signal.position_size} {signal.symbol}",
                    extra={'symbol': signal.symbol, 'action': signal.signal_type.value,
                           'quantity': signal.position_size, 'order_id': order_id,
                           'price': getattr(signal, 'price', None)}
                )

                # Track pending order
                self.pending_orders[order_id] = {
                    'symbol': signal.symbol,
                    'signal': signal,
                    'status': 'PENDING',
                    'placed_time': datetime.now()
                }
                self.order_id_to_symbol[order_id] = signal.symbol
                
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"✅ Order Submitted",
                    message=f"Order ID: {order_id} - Waiting for fill confirmation",
                    data={
                        'order_id': order_id,
                        'quantity': signal.position_size,
                        'expected_price': signal.entry_price
                    },
                    importance=9
                ))

                # Wait a moment and check order status immediately
                await asyncio.sleep(2)
                # Wait for fill verification
                if await self._verify_order_fill(order_id, signal):
                    # v-fill-creates-position-2026-05-08: ROOT CAUSE FIX for
                    # the multi-day `managed_by_bot=False` mystery. Prior
                    # code only created the Position object inside the
                    # _check_order_status sweep path (line ~1561), and that
                    # path also had an UnboundLocalError on `signal` so it
                    # never fired. Net result: every bot-opened live trade
                    # was first registered to self.positions by
                    # sync_positions_with_schwab as an "external discovery"
                    # with managed_by_bot=False, leaving stops/targets
                    # unmanaged. Create the bot-managed Position here in
                    # the immediate fill path, attach exit-manager
                    # tracking, and remove the order from pending_orders
                    # so the sweep doesn't double-process.
                    fill_price = float(getattr(signal, 'entry_price', 0) or 0)
                    try:
                        position = Position(
                            symbol=signal.symbol,
                            entry_price=fill_price,
                            current_price=fill_price,
                            quantity=int(signal.position_size),
                            side=('long' if signal.signal_type == SignalType.BUY else 'short'),
                            stop_loss=float(getattr(signal, 'stop_loss', 0) or 0),
                            take_profit=float(getattr(signal, 'take_profit', 0) or 0),
                            entry_time=datetime.now(),
                            reasoning=signal.reasoning or {},
                            mode="live",
                            managed_by_bot=True,
                        )
                        self.positions[signal.symbol] = position
                        if hasattr(self, 'exit_manager') and self.exit_manager is not None:
                            try:
                                self.exit_manager.initialize_position_tracking(
                                    signal.symbol,
                                    fill_price,
                                    position.stop_loss,
                                    position.take_profit,
                                )
                            except Exception as _ex:
                                logger.warning(
                                    "exit_manager init failed for %s: %s",
                                    signal.symbol, _ex,
                                )
                        # Remove from pending so the sweep does not try to
                        # create a duplicate Position later.
                        self.pending_orders.pop(order_id, None)
                        self.order_id_to_symbol.pop(order_id, None)
                        self._audit(
                            "order_fill", signal.symbol, "position_created",
                            "managed_by_bot",
                            order_id=order_id,
                            fill_price=round(fill_price, 4),
                            qty=int(signal.position_size),
                            stop=round(float(position.stop_loss or 0), 4),
                            target=round(float(position.take_profit or 0), 4),
                        )
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.DECISION,
                            symbol=signal.symbol,
                            title=f"💰 Order Filled — Bot Managing",
                            message=(
                                f"{signal.symbol} {position.side} "
                                f"{position.quantity} @ ${fill_price:.2f} "
                                f"— stop ${position.stop_loss:.2f}, "
                                f"target ${position.take_profit:.2f}. "
                                "Bot will manage exits."
                            ),
                            data={
                                'order_id': order_id,
                                'fill_price': fill_price,
                                'quantity': int(signal.position_size),
                                'managed_by_bot': True,
                            },
                            importance=9,
                        ))
                    except Exception as _ex:
                        logger.error(
                            "fill-path Position creation failed for %s: %s",
                            signal.symbol, _ex,
                        )

                    # v-bracket-call-fix-2026-05-08: signature is
                    # _place_bracket_orders(signal, parent_order_id) and
                    # parent_order_id is required. Prior call passed only
                    # `signal` → silent TypeError → no broker-side bracket
                    # orders ever placed. Pass the parent order_id we
                    # have in scope.
                    try:
                        await self._place_bracket_orders(signal, order_id)
                    except Exception as _ex:
                        logger.warning(
                            "bracket order placement failed for %s: %s",
                            signal.symbol, _ex,
                        )

                    # Log the trade for analytics
                    log_trade(
                        symbol=signal.symbol,
                        action='BUY' if signal.signal_type == SignalType.BUY else 'SELL',
                        quantity=signal.position_size,
                        price=signal.entry_price,
                        order_type='LIMIT' if use_limit else 'MARKET',
                        side='long' if signal.signal_type == SignalType.BUY else 'short',
                        reason=signal.reasoning.get('primary_reason', 'strategy_signal'),
                        strategy=signal.reasoning.get('strategy', 'unknown'),
                        confidence=signal.confidence,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        account_balance=self.risk_manager.account_balance,
                        mode=self.mode.value,
                        order_id=order_id
                    )
                    log_system_event('TRADE_EXECUTED', f"Executed {signal.signal_type.value} for {signal.symbol}", 'INFO', {
                        'symbol': signal.symbol,
                        'quantity': signal.position_size,
                        'price': signal.entry_price,
                        'order_id': order_id
                    })

                    return True
                else:
                    return False                
            else:
                # Parse rejection reason
                rejection = self._parse_order_rejection(response)
                
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"❌ Order Rejected: {rejection['reason']}",
                    message=rejection['message'],
                    data={'details': rejection['details']},
                    importance=10
                ))
                return False
                
        except InsufficientFundsException as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"💸 Insufficient Funds",
                message=e.args[0],
                data=e.details,
                importance=9
            ))
            logger.warning(f"Insufficient funds for {signal.symbol}: {e}")
            self.error_recovery.increment_error_count('InsufficientFunds', signal.symbol)
            action = self.error_recovery.suggest_recovery_action(e)
            if action:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.INFO,
                    symbol=signal.symbol,
                    title="💡 Suggested Action",
                    message=action,
                    importance=7
                ))
            return False
            
        except RateLimitException as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"⏳ Rate Limited",
                message=f"API rate limit hit. Retry after {e.retry_after}s",
                importance=8
            ))
            logger.warning(f"Rate limited: {e}")
            # Schedule retry
            asyncio.create_task(self._retry_order_after_delay(signal, e.retry_after))
            return False
            
        except (NetworkException, requests.exceptions.RequestException) as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"🌐 Network Error",
                message="Connection issue - will retry",
                importance=7
            ))
            logger.error(f"Network error: {e}")
            # Retry with exponential backoff
            asyncio.create_task(self._retry_order_with_backoff(signal))
            return False
            
        except StaleDataException as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"📊 Stale Data",
                message="Market data too old - refreshing",
                importance=6
            ))
            logger.warning(f"Stale data for {signal.symbol}")
            return False
            
        except CircuitBreakerException as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"🔌 Circuit Breaker Activated",
                message=f"Too many failures - pausing for {e.cooldown_minutes} minutes",
                importance=9
            ))
            logger.warning(f"Circuit breaker activated: {e}")
            self.error_recovery.record_circuit_breaker_open('order_placement', str(e))
            return False
            
        except AuthenticationException as e:
            # Fatal - stop trading
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"🔐 Authentication Failed",
                message="Invalid credentials - stopping bot",
                importance=10
            ))
            logger.critical(f"Authentication failed: {e}")
            self.stop_trading = True
            raise
            
        except Exception as e:
            # Unknown error - log details for debugging
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"❌ Unexpected Error",
                message=f"Unknown error: {type(e).__name__}",
                data={'error': str(e), 'traceback': traceback.format_exc()},
                importance=10
            ))
            logger.error(f"Unexpected order error: {e}", exc_info=True)
            return False
    async def _retry_order_after_delay(self, signal, delay: float):
        """Retry order after specified delay"""
        await asyncio.sleep(delay)
        await self._execute_real_trade(signal)
    
    async def _retry_order_with_backoff(self, signal, attempt: int = 0, max_attempts: int = 3):
        """Retry order with exponential backoff"""
        if attempt >= max_attempts:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"❌ Max Retries Exceeded",
                message=f"Failed after {max_attempts} attempts",
                importance=9
            ))
            return
        
        delay = get_retry_delay(NetworkException("Network error"), attempt)
        await asyncio.sleep(delay)
        
        try:
            await self._execute_real_trade(signal)
        except (NetworkException, requests.exceptions.RequestException):
            await self._retry_order_with_backoff(signal, attempt + 1, max_attempts)
    
    # Add these methods after _execute_real_trade in TradingEngineWithCommentary class:

    async def _check_existing_orders(self, symbol: str) -> Dict[str, Any]:
        """Check for existing orders on a symbol"""
        try:
            response = self.schwab_client.get_orders_for_account(
                self.account_hash,
                from_entered_datetime=datetime.now() - timedelta(days=1),
                status=self.schwab_client.Order.Status.WORKING  # Only active orders
            )
            
            if response.status_code == 200:
                orders = response.json()
                symbol_orders = []
                
                for order in orders:
                    # Check main order
                    order_symbol = self._extract_symbol_from_order(order)
                    if order_symbol == symbol:
                        symbol_orders.append(order)
                    
                    # Check child orders (OCO)
                    for child in order.get('childOrderStrategies', []):
                        child_symbol = self._extract_symbol_from_order(child)
                        if child_symbol == symbol:
                            symbol_orders.append(order)
                            break
                
                return {
                    'has_orders': len(symbol_orders) > 0,
                    'orders': symbol_orders,
                    'count': len(symbol_orders)
                }
        except Exception as e:
            logger.error(f"Error checking orders: {e}")
        
        return {'has_orders': False, 'orders': [], 'count': 0}

    def _extract_symbol_from_order(self, order: Dict) -> Optional[str]:
        """Extract symbol from order structure"""
        try:
            legs = order.get('orderLegCollection', [])
            if legs:
                return legs[0].get('instrument', {}).get('symbol')
        except (KeyError, IndexError, TypeError) as e:
            logger.debug(f"Could not extract symbol from order: {e}")
        return None

    async def _cancel_existing_orders(self, symbol: str) -> bool:
        """Cancel existing orders for a symbol"""
        try:
            existing = await self._check_existing_orders(symbol)
            
            if existing['has_orders']:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=symbol,
                    title=f"🔄 Canceling Existing Orders",
                    message=f"Found {existing['count']} existing orders for {symbol}",
                    importance=7
                ))
                
                for order in existing['orders']:
                    order_id = order.get('orderId')
                    if order_id:
                        try:
                            # v-schwab-arg-order-2026-05-18: cancel_order signature
                            # is (order_id, account_hash). Was passing
                            # (account_id, order_id) — wrong arg order AND
                            # account_id instead of account_hash.
                            response = self.schwab_client.cancel_order(order_id, self.account_hash)
                            if response.status_code in [200, 201, 202]:
                                logger.info(f"Canceled order {order_id}")
                        except Exception as e:
                            logger.error(f"Failed to cancel order {order_id}: {e}")
                
                # Wait for cancellations to process
                await asyncio.sleep(1)
                return True
        
        except Exception as e:
            logger.error(f"Error canceling orders: {e}")
        
        return False
    '''
    async def _check_buying_power(self, required_amount: float) -> Tuple[bool, float]:
        """Check if we have sufficient buying power"""
        try:
            if hasattr(self, '_cached_buying_power'):
                # Use cached value if recent (within 30 seconds)
                if (datetime.now() - self._last_bp_check).total_seconds() < 30:
                    return self._cached_buying_power >= required_amount, self._cached_buying_power
            
            response = self.schwab_client.get_account(self.account_hash)
            logger.info(f" Checking Buying Power: {response.json()}")
            if response.status_code == 200:
                data = response.json()
                balances = data.get('securitiesAccount', {}).get('currentBalances', {})
                
                # Get appropriate buying power based on account type
                buying_power = balances.get('buyingPower', 0)
                
                # For margin accounts
                if 'dayTradingBuyingPower' in balances:
                    buying_power = min(buying_power, balances['dayTradingBuyingPower'])
                
                # Cache the result
                self._cached_buying_power = buying_power
                self._last_bp_check = datetime.now()
                
                # Update risk manager
                self.risk_manager.buying_power = buying_power
                
                return buying_power >= required_amount, buying_power
                
        except Exception as e:
            logger.error(f"Error checking buying power: {e}")
        
        return False, 0
    '''
    async def _check_buying_power(self, required_amount: float) -> Tuple[bool, float]:
        """Check if we have sufficient buying power"""
        try:
            # Check if we should use cache
            cache_age = (datetime.now() - self._last_bp_check).total_seconds()
        
            # Only use cache if we have a valid cached value AND it's recent
            if self._cached_buying_power > 0 and cache_age < 30:
                logger.debug(f"Using cached buying power: ${self._cached_buying_power:,.2f} (age: {cache_age:.1f}s)")
                return self._cached_buying_power >= required_amount, self._cached_buying_power
            
            # Fetch fresh data
            response = self.schwab_client.get_account(self.account_hash)
            logger.info(f" Buying power check: {response.json()}")
            if response.status_code != 200:
                logger.error(f"Failed to get account data: {response.status_code}")
                return False, 0
            
            data = response.json()
            
            # Log the structure for debugging
            logger.debug(f"Account data keys: {list(data.keys())}")
            
            # Navigate to the correct location
            account_data = data.get('securitiesAccount', {})
            
            # Try multiple possible locations for buying power
            buying_power = 0
            
            # Method 1: Check currentBalances
            if 'currentBalances' in account_data:
                balances = account_data['currentBalances']
                logger.debug(f"Balance fields available: {list(balances.keys())}")
                
                # Try different field names
                buying_power = (
                    balances.get('buyingPower', 0) or
                    balances.get('availableFunds', 0) or
                    balances.get('availableFundsNonMarginableTrade', 0) or
                    balances.get('buyingPowerNonMarginableTrade', 0)
                )
                
                # For margin accounts, check day trading buying power
                if 'dayTradingBuyingPower' in balances:
                    day_trading_bp = balances.get('dayTradingBuyingPower', 0)
                    if day_trading_bp > 0:
                        buying_power = min(buying_power, day_trading_bp)
            
            # Method 2: Check projectedBalances if currentBalances didn't work
            if buying_power == 0 and 'projectedBalances' in account_data:
                projected = account_data['projectedBalances']
                buying_power = (
                    projected.get('buyingPower', 0) or
                    projected.get('availableFunds', 0) or
                    projected.get('availableFundsNonMarginableTrade', 0)
                )
            
            # Method 3: Calculate from cash and margin if still 0
            if buying_power == 0 and 'currentBalances' in account_data:
                balances = account_data['currentBalances']
                cash = balances.get('cashBalance', 0) or balances.get('totalCash', 0)
                margin = balances.get('marginBalance', 0)
                
                # For cash accounts, buying power is usually just cash
                # For margin accounts, it could be cash + margin available
                account_type = account_data.get('type', 'CASH')
                if account_type == 'MARGIN':
                    buying_power = cash + margin
                else:
                    buying_power = cash
            
            # Log what we found
            logger.info(f"Buying power found: ${buying_power:,.2f} (required: ${required_amount:,.2f})")
            
            # Cache the result
            self._cached_buying_power = buying_power
            self._last_bp_check = datetime.now()
            
            # Update risk manager
            self.risk_manager.buying_power = buying_power
            
            return buying_power >= required_amount, buying_power
            
        except Exception as e:
            logger.error(f"Error checking buying power: {e}", exc_info=True)
            # Return cached value if available
            if hasattr(self, '_cached_buying_power') and self._cached_buying_power > 0:
                logger.warning(f"Using last known buying power: ${self._cached_buying_power:,.2f}")
                return self._cached_buying_power >= required_amount, self._cached_buying_power
        
        return False, 0
    def _parse_order_rejection(self, response) -> Dict[str, Any]:
        """Parse Schwab order rejection reasons"""
        try:
            if hasattr(response, 'text'):
                error_data = json.loads(response.text)
                error_message = str(error_data).lower()
                
                # Common rejection reasons
                if 'buying power' in error_message or 'insufficient funds' in error_message:
                    return {
                        'reason': 'INSUFFICIENT_FUNDS',
                        'message': 'Not enough buying power for this trade',
                        'details': error_data
                    }
                elif 'duplicate' in error_message:
                    return {
                        'reason': 'DUPLICATE_ORDER',
                        'message': 'A similar order already exists for this symbol',
                        'details': error_data
                    }
                elif 'market closed' in error_message or 'outside hours' in error_message:
                    return {
                        'reason': 'MARKET_CLOSED',
                        'message': 'Cannot place order - market is closed',
                        'details': error_data
                    }
                elif 'position' in error_message and 'exist' in error_message:
                    return {
                        'reason': 'POSITION_EXISTS',
                        'message': 'Position already exists in this symbol',
                        'details': error_data
                    }
                elif 'halted' in error_message:
                    return {
                        'reason': 'SYMBOL_HALTED',
                        'message': 'Trading is halted for this symbol',
                        'details': error_data
                    }
                elif 'invalid' in error_message and 'price' in error_message:
                    return {
                        'reason': 'INVALID_PRICE',
                        'message': 'Order price is invalid (check stop/limit prices)',
                        'details': error_data
                    }
        except Exception as e:
            logger.debug(f"Could not parse order rejection details: {e}")

        return {
            'reason': 'UNKNOWN',
            'message': f'Order rejected - Status: {response.status_code}',
            'details': response.text if hasattr(response, 'text') else None
        }

    def _validate_oco_prices(self, symbol: str, stop_price: float, take_profit: float, is_short: bool = False) -> bool:
        """Validate OCO order prices before submission for both long and short positions"""
        try:
            # Get current quote
            quote = self.data_provider.get_quote(symbol)
            if not quote:
                return True  # Skip validation if no quote
            
            current_price = quote.get('last', 0)
            bid = quote.get('bid', current_price)
            ask = quote.get('ask', current_price)
            
            if is_short:
                # For short positions (sell to open):
                # - Stop loss must be ABOVE current ask (we're buying back at a loss)
                # - Take profit must be BELOW current bid (we're buying back at a profit)
                
                if stop_price <= ask:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Invalid Stop Price for Short",
                        message=f"Stop ${stop_price:.2f} must be above ask ${ask:.2f} for short position",
                        importance=8
                    ))
                    return False
                
                if take_profit >= bid:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Invalid Target Price for Short",
                        message=f"Target ${take_profit:.2f} must be below bid ${bid:.2f} for short position",
                        importance=8
                    ))
                    return False
            else:
                # For long positions (existing logic):
                # - Stop loss must be below current bid
                # - Take profit must be above current ask
                
                if stop_price >= bid:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Invalid Stop Price",
                        message=f"Stop ${stop_price:.2f} must be below bid ${bid:.2f}",
                        importance=8
                    ))
                    return False
                
                if take_profit <= ask:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Invalid Target Price",
                        message=f"Target ${take_profit:.2f} should be above ask ${ask:.2f}",
                        importance=8
                    ))
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Price validation error: {e}")
            return True  # Allow order if validation fails
    
    async def _place_bracket_orders(self, signal, parent_order_id: str):
        """Place stop loss and take profit orders as OCO"""
        # v-bracket-diagnostic-2026-05-18: log every attempt so silent
        # failures surface. Before this, an exception or non-2xx
        # response would either get swallowed by the caller's try/except
        # (engine.py:920) or quietly logged once and never seen again.
        logger.info(
            "bracket_attempt symbol=%s stop=%s target=%s qty=%s parent_order=%s",
            signal.symbol,
            getattr(signal, 'stop_loss', None),
            getattr(signal, 'take_profit', None),
            getattr(signal, 'position_size', None),
            parent_order_id,
        )
        try:
            from schwab.orders.common import one_cancels_other, Duration, Session, OrderType
            from schwab.orders.equities import equity_sell_limit

            # Create take profit order (limit sell)
            take_profit_order = equity_sell_limit(
                signal.symbol,
                signal.position_size,
                signal.take_profit
            ).set_duration(Duration.GOOD_TILL_CANCEL).set_session(Session.NORMAL)

            # Create stop loss order (stop limit - using stop price slightly below limit)
            # For stop limit, we set both stop price and limit price
            stop_loss_order = (equity_sell_limit(
                signal.symbol,
                signal.position_size,
                signal.stop_loss * 0.995  # Limit price slightly below stop
            ).set_order_type(OrderType.STOP_LIMIT)
            .set_stop_price(signal.stop_loss)
            .set_duration(Duration.GOOD_TILL_CANCEL)
            .set_session(Session.NORMAL))

            # Create OCO order using helper function
            oco_order = one_cancels_other(take_profit_order, stop_loss_order)

            # Build and place the OCO order
            response = self.schwab_client.place_order(self.account_hash, oco_order.build())
            logger.info(
                "bracket_response symbol=%s http=%s body=%s",
                signal.symbol, response.status_code, response.text[:200],
            )

            if response.status_code in [200, 201]:
                # Extract order ID from response
                order_id = response.headers.get('Location', '').split('/')[-1]

                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.RISK_ASSESSMENT,
                    symbol=signal.symbol,
                    title=f"✅ OCO Bracket Order Placed",
                    message=f"Stop Loss: ${signal.stop_loss:.2f} | Take Profit: ${signal.take_profit:.2f}",
                    data={
                        'stop_loss': signal.stop_loss,
                        'take_profit': signal.take_profit,
                        'order_type': 'OCO',
                        'order_id': order_id
                    },
                    importance=7
                ))
            else:
                rejection = self._parse_order_rejection(response)
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"⚠️ OCO Order Failed: {rejection['reason']}",
                    message=rejection['message'],
                    data={'details': rejection['details']},
                    importance=7
                ))

        except Exception as e:
            logger.error(f"Bracket order error: {e}", exc_info=True)
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"⚠️ Bracket Order Error",
                message=f"Could not place stop/target orders: {str(e)}",
                importance=7
            ))
    def _validate_oco_prices(self, symbol: str, stop_price: float, take_profit: float, is_short: bool = False) -> bool:
        """Validate OCO order prices before submission for both long and short positions"""
        try:
            # Get current quote
            quote = self.data_provider.get_quote(symbol)
            if not quote:
                return True  # Skip validation if no quote
            
            current_price = quote.get('last', 0)
            bid = quote.get('bid', current_price)
            ask = quote.get('ask', current_price)
            
            if is_short:
                # For short positions (sell to open):
                # - Stop loss must be ABOVE current ask (we're buying back at a loss)
                # - Take profit must be BELOW current bid (we're buying back at a profit)
                
                if stop_price <= ask:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Invalid Stop Price for Short",
                        message=f"Stop ${stop_price:.2f} must be above ask ${ask:.2f} for short position",
                        importance=8
                    ))
                    return False
                
                if take_profit >= bid:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Invalid Target Price for Short",
                        message=f"Target ${take_profit:.2f} must be below bid ${bid:.2f} for short position",
                        importance=8
                    ))
                    return False
            else:
                # For long positions (existing logic):
                # - Stop loss must be below current bid
                # - Take profit must be above current ask
                
                if stop_price >= bid:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Invalid Stop Price",
                        message=f"Stop ${stop_price:.2f} must be below bid ${bid:.2f}",
                        importance=8
                    ))
                    return False
                
                if take_profit <= ask:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=symbol,
                        title=f"⚠️ Invalid Target Price",
                        message=f"Target ${take_profit:.2f} should be above ask ${ask:.2f}",
                        importance=8
                    ))
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Price validation error: {e}")
            return True  # Allow order if validation fails
    async def _check_specific_order_status(self, order_id: str) -> str:
        """Check status of a specific order immediately"""
        try:
            # v-schwab-arg-order-2026-05-18: get_order signature is
            # (order_id, account_hash). Was swapped — Schwab returned 400
            # "not a valid orderId" on every call, so _verify_order_fill
            # never saw FILLED status, never created managed_by_bot
            # Position, never called _place_bracket_orders. Every trade
            # since the bot was written exited via external_close because
            # of this one swap.
            response = self.schwab_client.get_order(order_id, self.account_hash)

            if response.status_code == 200:
                order_info = response.json()
                status = order_info.get('status', 'UNKNOWN')
                
                if order_id in self.pending_orders:
                    self.pending_orders[order_id]['status'] = status
                    
                    if status == 'REJECTED':
                        symbol = self.pending_orders[order_id]['symbol']
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=symbol,
                            title=f"❌ Order Rejected by Broker",
                            message=f"Order {order_id} was rejected. Not proceeding with position.",
                            data={'order_info': order_info},
                            importance=10
                        ))
                        # Remove from tracking
                        del self.pending_orders[order_id]
                        if order_id in self.order_id_to_symbol:
                            del self.order_id_to_symbol[order_id]
                    
                    elif status == 'FILLED':
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.DECISION,
                            symbol=self.pending_orders[order_id]['symbol'],
                            title=f"✅ Order Filled",
                            message=f"Order {order_id} has been filled",
                            importance=8
                        ))
                
                return status
        except Exception as e:
            logger.error(f"Error checking order {order_id}: {e}")
            return 'UNKNOWN'
        
    def _clear_recent_attempt_for_order(self, order_id: str, reason: str) -> None:
        """v-autocancel-fix-2026-05-26: clear ghost entry attempt.

        anti-pyramid blocks re-entry for 4 hours after _recent_entry_attempts
        is set. That dict is set BEFORE the order goes out so two parallel
        signals don't race past the cooldown. If the order then fails to
        fill (auto-cancel, Schwab REJECTED/CANCELED), no position was opened
        and the cooldown is a false positive. QBTS 2026-05-26 09:37 was
        auto-cancelled at 09:37:47; the 10:45 news signal was then anti-
        pyramid-blocked based on the ghost attempt. Clear the entry so
        re-entry is permitted.
        """
        sym = self.pending_orders.get(order_id, {}).get('symbol')
        if sym and hasattr(self, '_recent_entry_attempts'):
            if self._recent_entry_attempts.pop(sym, None) is not None:
                logger.debug(
                    "cleared_recent_entry_attempt symbol=%s order_id=%s reason=%s",
                    sym, order_id, reason,
                )

    async def _verify_order_fill(self, order_id: str, signal, max_retries: int = 30) -> bool:
        """Verify order filled with retries.

        v-verify-timeout-extend-2026-05-11: 3 → 10 retries (6s → 20s).
        Original 6s window was too tight for Schwab's order-status API
        lag on busy days. NVDA 09:54, RDW/RGTI/SMR/LUNR/FCEL 10:03–10:05
        all filled on Schwab but verify timed out → no Position with
        managed_by_bot=True → no `_place_bracket_orders` → naked broker
        positions. 20s covers >99% of fills based on observation.

        v-verify-timeout-extend-2026-05-26: 10 → 30 retries (20s → 60s).
        QBTS 09:37:27 was a limit BUY @ $27.45 that didn't fill in 20s and
        was silently auto-cancelled at 09:37:47. Limit orders on fast-
        moving names need more patience; 60s reduces the false-cancel rate.

        Also: after timeout, do ONE MORE status check before cancelling.
        If status is FILLED, return True so the caller's Position-
        creation path runs and bracket orders get placed. If we cancel
        without that check, we can cancel an already-filled order in
        Schwab's eyes (the cancel will fail but the position is still
        open with no bot tracking).
        """
        for attempt in range(max_retries):
            status = await self._check_specific_order_status(order_id)
            if status == 'FILLED':
                return True
            elif status in ['REJECTED', 'CANCELED']:
                # v-autocancel-fix-2026-05-26: Schwab reported terminal-
                # not-filled status. Clear the ghost attempt so subsequent
                # signals for this symbol aren't blocked by anti-pyramid.
                self._clear_recent_attempt_for_order(order_id, reason=f"schwab_{status.lower()}")
                return False
            await asyncio.sleep(2)

        # v-verify-timeout-extend-2026-05-11: final FILLED check before
        # giving up. Catches the case where verify ran while Schwab was
        # still processing the fill.
        try:
            final_status = await self._check_specific_order_status(order_id)
            if final_status == 'FILLED':
                logger.info(
                    "order %s reached FILLED on final post-retry check (took >%ds)",
                    order_id, max_retries * 2,
                )
                return True
            if final_status in ('REJECTED', 'CANCELED'):
                # v-autocancel-fix-2026-05-26: same ghost-attempt cleanup.
                self._clear_recent_attempt_for_order(order_id, reason=f"schwab_final_{final_status.lower()}")
                return False
        except Exception as exc:
            logger.warning(
                "order %s final status check failed: %s", order_id, exc,
            )

        # Cancel unfilled order. After the final check above, if we got
        # here the order is genuinely stuck — cancel it. If cancel fails
        # because it actually JUST filled (race), check one more time and
        # return True so the Position gets created.
        try:
            # v-schwab-arg-order-2026-05-18: cancel_order(order_id, account_hash)
            self.schwab_client.cancel_order(order_id, self.account_hash)
            # v-autocancel-logging-2026-05-26: prior version silently
            # cancelled unfilled limit orders after the verify window
            # with NO log line on the success path. QBTS 2026-05-26 09:37
            # was the trigger incident. Log the cancel + clear the ghost
            # attempt so anti-pyramid doesn't block follow-up signals.
            cancelled_symbol = self.pending_orders.get(order_id, {}).get('symbol')
            logger.info(
                "auto_cancelled_unfilled_order order_id=%s symbol=%s "
                "after_seconds=%d reason=verify_timeout",
                order_id, cancelled_symbol, max_retries * 2,
            )
            self._clear_recent_attempt_for_order(order_id, reason="auto_cancel_verify_timeout")
            del self.pending_orders[order_id]
            self.order_id_to_symbol.pop(order_id, None)
        except Exception as e:
            logger.warning(f"Failed to cancel unfilled order {order_id}: {e}")
            try:
                race_status = await self._check_specific_order_status(order_id)
                if race_status == 'FILLED':
                    logger.info(
                        "order %s actually filled during cancel attempt — "
                        "promoting to FILLED so Position gets created",
                        order_id,
                    )
                    return True
            except Exception:
                pass
        return False
    
    async def _check_order_status(self):
        """Check status of pending orders"""
        if not self.pending_orders or not self.schwab_client:
            return
        
        # Check orders every 10 seconds
        if (datetime.now() - self.last_order_check).total_seconds() < 10:
            return
        
        self.last_order_check = datetime.now()
        
        for order_id, order_data in list(self.pending_orders.items()):
            try:
                # v-schwab-arg-order-2026-05-18: get_order(order_id, account_hash)
                response = self.schwab_client.get_order(
                    order_id,
                    self.account_hash,
                    #fields=[schwab_client.Orders.Fields.EXECUTION_LEGS]
                )
                
                if response.status_code == 200:
                    order_info = response.json()
                    logger.info(f"Existing Orders : {order_info}")
                    status = order_info.get('status', 'UNKNOWN')
                    
                    if status == 'FILLED':
                        # v-sweep-signal-scope-fix-2026-05-08: prior code
                        # read `signal.entry_price` BEFORE assigning
                        # `signal = order_data['signal']` → UnboundLocalError
                        # on every FILLED status check. The except handler
                        # at the bottom of this loop swallowed it silently,
                        # so the sweep path's Position-creation never ran.
                        # Hoist the assignment to the top of the branch.
                        signal = order_data['signal']
                        # Get fill price
                        fill_price = signal.entry_price  # Default
                        if 'orderActivityCollection' in order_info:
                            for activity in order_info['orderActivityCollection']:
                                if activity.get('executionType') == 'FILL':
                                    legs = activity.get('executionLegs', [])
                                    if legs:
                                        fill_price = legs[0].get('price', signal.entry_price)
                                        expected = order_data['signal'].entry_price
                                        slippage = abs(fill_price - expected) / expected
                                        self.performance_metrics['slippage'].append(slippage)
                        # v-sweep-skip-if-already-managed-2026-05-08: the
                        # immediate fill path already creates the Position
                        # and pops the order_id from pending_orders. If a
                        # stale entry slips in here for a symbol we are
                        # already tracking as bot-managed, skip duplicate
                        # creation but still run the rest of the cleanup.
                        if (
                            signal.symbol in self.positions
                            and bool(getattr(self.positions[signal.symbol], 'managed_by_bot', False))
                        ):
                            logger.debug(
                                "sweep: %s already managed_by_bot, skipping duplicate Position creation",
                                signal.symbol,
                            )
                            self.pending_orders.pop(order_id, None)
                            self.order_id_to_symbol.pop(order_id, None)
                            continue
                        # Create position
                        position = Position(
                            symbol=signal.symbol,
                            entry_price=fill_price,
                            current_price=fill_price,
                            quantity=signal.position_size,
                            side='long' if signal.signal_type == SignalType.BUY else 'short',
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            entry_time=datetime.now(),
                            reasoning=signal.reasoning,
                            mode="live",
                            managed_by_bot=True,  # bot-opened live trade
                        )
                        self.positions[signal.symbol] = position
                        
                        # Initialize exit tracking
                        self.exit_manager.initialize_position_tracking(
                            signal.symbol,
                            fill_price,
                            signal.stop_loss,
                            signal.take_profit
                        )
                        
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.DECISION,
                            symbol=signal.symbol,
                            title=f"💰 Order Filled!",
                            message=f"Position opened at ${fill_price:.2f}",
                            data={
                                'order_id': order_id,
                                'fill_price': fill_price,
                                'quantity': signal.position_size
                            },
                            importance=9
                        ))
                        
                        del self.pending_orders[order_id]
                        
                    elif status in ['CANCELED', 'REJECTED', 'EXPIRED']:
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=order_data['symbol'],
                            title=f"❌ Order {status}",
                            message=f"Order {order_id} was {status.lower()}",
                            importance=8
                        ))
                        del self.pending_orders[order_id]
                        
            except Exception as e:
                logger.error(f"Error checking order {order_id}: {e}")
    
    async def _close_real_position(self, position, exit_portion: float = 1.0):
        """Close real position through Schwab"""
        if not self.schwab_client:
            return False

        try:
            from schwab.orders.equities import (equity_sell_market, equity_buy_market,
                                                equity_buy_to_cover_market, equity_sell_limit,
                                                equity_buy_limit, equity_buy_to_cover_limit)
            from schwab.orders.common import Duration, Session

            quantity_to_sell = int(position.quantity * exit_portion)

            # v-cancel-bracket-before-close-2026-06-08: cancel the
            # active OCO bracket BEFORE placing the close order.
            # Bug observed 2026-06-08 11:12 ET (NOK/GLW/OWL manual
            # closes): Schwab rejected every sell with `oversold
            # position` because the OCO's stop+target legs still
            # pledged the shares. The 4 reconciled trades (~$533
            # realized) were only saved by the user closing in the
            # Schwab app after cancelling the bracket manually —
            # a 6-cancel workflow that should be one API call.
            #
            # `_cancel_existing_orders` is the same helper used in the
            # entry path (engine.py:5241) and includes a 1s settle
            # sleep so Schwab releases the shares before the sell
            # hits. If no bracket exists (e.g., partial fill, external
            # position), the helper is a no-op and returns False —
            # safe to call unconditionally.
            # v-live-exit-ladder-2026-06-11: cancel for ANY portion,
            # not only full closes — a partial sell against pledged
            # shares is the same June-8 "oversold position" reject.
            # (Re-bracketing the remainder after a partial is part of
            # the ENABLE_LIVE_SCALE_OUT work, not yet armed.)
            await self._cancel_existing_orders(position.symbol)

            # Check market hours for order type
            use_limit, reason = self.should_use_limit_order()
            is_regular, session = self.is_market_hours()
            
            # Get current price for limit orders
            current_price = position.current_price if hasattr(position, 'current_price') else position.entry_price
            
            # Create appropriate order based on position side and market hours
            if use_limit or self.use_limit_orders_extended_hours:
                # Use limit orders during extended hours
                logger.info(f"Using limit order to close position: {reason}")
                
                if position.side == 'long':
                    # Sell with limit slightly below current for quick fill
                    limit_price = current_price * 0.999
                    order_builder = equity_sell_limit(position.symbol, quantity_to_sell, limit_price)
                else:  # SHORT position - need to buy to cover
                    # Buy to cover with limit slightly above current for quick fill
                    limit_price = current_price * 1.001
                    try:
                        order_builder = equity_buy_to_cover_limit(position.symbol, quantity_to_sell, limit_price)
                    except AttributeError:
                        order_builder = equity_buy_limit(position.symbol, quantity_to_sell, limit_price)
                        order_builder.set_instruction('BUY_TO_COVER')
            else:
                # Use market orders during regular hours
                if position.side == 'long':
                    order_builder = equity_sell_market(position.symbol, quantity_to_sell)
                else:  # SHORT position - need to buy to cover
                    try:
                        order_builder = equity_buy_to_cover_market(position.symbol, quantity_to_sell)
                    except AttributeError:
                        order_builder = equity_buy_market(position.symbol, quantity_to_sell)
                        order_builder.set_instruction('BUY_TO_COVER')

            order_builder.set_duration(Duration.DAY)
            
            # Set session based on market hours
            if session == 'premarket' or session == 'afterhours':
                order_builder.set_session(Session.EXTENDED)
            else:
                order_builder.set_session(Session.NORMAL)
            
            order = order_builder.build()
            
            response = self.schwab_client.place_order(self.account_hash, order)
            
            if response.status_code in [200, 201]:
                order_id = response.headers.get('Location', '').split('/')[-1]

                logger.info(
                    f"Close order placed: SELL {quantity_to_sell} {position.symbol}",
                    extra={'symbol': position.symbol, 'action': 'SELL',
                           'quantity': quantity_to_sell, 'order_id': order_id,
                           'reason': reason}
                )

                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=position.symbol,
                    title=f"📤 Closing Position",
                    message=f"Selling {quantity_to_sell} shares at market",
                    importance=9
                ))

                # Track this as a closing order
                self.pending_orders[order_id] = {
                    'symbol': position.symbol,
                    'type': 'CLOSE',
                    'quantity': quantity_to_sell,
                    'status': 'PENDING',
                    'placed_time': datetime.now()
                }

                # v-verify-close-fill-2026-06-08: Schwab's 201 means the
                # order was accepted for validation, NOT filled. Without
                # this verify the caller (`_close_position_with_commentary`)
                # pops the position from local state on any 201 — even if
                # Schwab asynchronously rejects (the 2026-06-08 NOK/GLW/OWL
                # incident: 3 closes rejected for "oversold position", but
                # local state was already cleaned up → bot blind to live
                # positions still at the broker).
                #
                # Reuse the entry path's `_verify_order_fill` helper. It
                # polls Schwab's order-status API for up to 60s (30 × 2s
                # backoff), returns True only on FILLED, and on
                # REJECTED/CANCELED/timeout it self-cleans (cancels stuck
                # orders, clears anti-pyramid attempt tracking). The
                # `signal` parameter is unused inside the body — passing
                # `position` is safe and self-documenting at the call site.
                #
                # When verify returns False, we return False so the caller
                # promotes the position to ZOMBIE state (see the
                # `broker_close_returned_false` reason in
                # `_close_position_with_commentary`) — the position stays
                # in local state, the operator gets a dashboard alert,
                # and no other exit rule second-guesses the stuck close.
                filled = await self._verify_order_fill(order_id, position)
                if not filled:
                    logger.warning(
                        "close_verify_failed symbol=%s order_id=%s — "
                        "caller will promote to ZOMBIE",
                        position.symbol, order_id,
                    )
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=position.symbol,
                        title=f"⚠️ Close Did Not Fill",
                        message=(
                            f"Close order for {position.symbol} was accepted "
                            f"by Schwab but did not reach FILLED status. "
                            f"Position will be marked ZOMBIE for operator review."
                        ),
                        importance=10,
                    ))
                    return False

                return True
            else:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=position.symbol,
                    title=f"❌ Close Order Failed",
                    message=f"Could not close position: {response.status_code}",
                    importance=10
                ))
                return False
                
        except Exception as e:
            logger.error(f"Position close error: {e}")
            return False
    async def _update_real_positions(self):
        """Update real positions from account - syncs ALL Schwab positions"""
        if not self.schwab_client or self.mode != TradingMode.LIVE:
            return

        try:
            # Get all Schwab positions
            schwab_positions = await self.get_schwab_positions()

            # Create a set of symbols from Schwab
            schwab_symbols = {pos['symbol'] for pos in schwab_positions}

            # Add or update positions from Schwab
            for pos_data in schwab_positions:
                symbol = pos_data['symbol']

                if symbol not in self.positions:
                    # Create new position for externally opened position
                    # IMPORTANT: External positions have NO automatic stop loss/take profit
                    # They are flagged as manually managed and the bot won't take actions on them
                    position = Position(
                        symbol=symbol,
                        quantity=abs(pos_data['quantity']),
                        entry_price=pos_data['average_price'],
                        current_price=pos_data['current_price'],
                        stop_loss=0,  # No automatic stop loss for external positions
                        take_profit=float('inf'),  # No automatic take profit
                        entry_time=datetime.now(),
                        side='long' if pos_data['quantity'] > 0 else 'short',
                        reasoning={'source': 'external', 'strategy': 'manual_entry'},
                        mode="live",
                        managed_by_bot=False,  # external/manual → hands off
                    )
                    position.unrealized_pnl = pos_data['total_pnl']
                    position.is_external = True  # Flag as externally created
                    position.is_manually_managed = True  # Bot won't auto-manage this
                    self.positions[symbol] = position

                    # Log that we found an external position
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.MARKET_ANALYSIS,
                        symbol=symbol,
                        title=f"📥 External Position Detected (Manual Mode)",
                        message=f"Found {abs(pos_data['quantity'])} shares of {symbol} "
                                f"({'long' if pos_data['quantity'] > 0 else 'short'})\n"
                                f"Entry: ${pos_data['average_price']:.2f}, "
                                f"Current: ${pos_data['current_price']:.2f}\n"
                                f"⚠️ This position is MANUALLY MANAGED - bot will NOT auto-close",
                        data=pos_data,
                        importance=7
                    ))
                else:
                    # Update existing position prices only
                    position = self.positions[symbol]
                    position.current_price = pos_data['current_price']
                    position.unrealized_pnl = pos_data['total_pnl']
            
            # Remove positions that no longer exist in Schwab
            positions_to_remove = []
            for symbol in self.positions:
                if symbol not in schwab_symbols:
                    positions_to_remove.append(symbol)
            
            for symbol in positions_to_remove:
                position = self.positions[symbol]
                
                # Try to get current market price for more accurate P&L
                exit_price = position.current_price
                try:
                    if self.data_provider:
                        quote = self.data_provider.get_quote(symbol)
                        if quote and 'last' in quote:
                            exit_price = quote['last']
                except Exception as e:
                    logger.debug(f"Could not get quote for {symbol}: {e}")
                
                # Calculate final P&L
                if position.side == 'short':
                    final_pnl = (position.entry_price - exit_price) * position.quantity
                else:  # long
                    final_pnl = (exit_price - position.entry_price) * position.quantity
                
                # v-consecutive-losses-live-only-2026-05-11: see twin
                # site in _close_position_with_commentary. Sibling
                # increment site — same rule: only count live closes
                # against the live circuit breaker.
                _is_live_close_sweep = (getattr(position, "mode", "live") == "live")
                if _is_live_close_sweep:
                    if final_pnl < 0:
                        # v-consec-loss-daily-reset-2026-05-21: also stamp date.
                        self.risk_manager.consecutive_losses += 1
                        self.risk_manager.last_loss_date = datetime.now().date()
                    else:
                        self.risk_manager.consecutive_losses = 0
                
                # Record in trade history
                self.trade_history.append({
                    'symbol': position.symbol,
                    'entry_time': position.entry_time.isoformat(),
                    'exit_time': datetime.now().isoformat(),
                    'entry_price': position.entry_price,
                    'exit_price': exit_price,
                    'quantity': position.quantity,
                    'pnl': final_pnl,
                    'exit_reason': 'external_close',
                    'reasoning': position.reasoning
                })
                
                # Remove from positions
                del self.positions[symbol]
                
                # Add commentary
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=symbol,
                    title=f"📤 Position Closed Externally",
                    message=f"{symbol} closed outside bot - P&L: ${final_pnl:.2f} ({'profit' if final_pnl > 0 else 'loss'})",
                    importance=6
                ))
                        
        except Exception as e:
            logger.error(f"Position update error: {e}")
    def _in_warmup_window(self) -> bool:
        """True if we're within WARMUP_MINUTES_BEFORE_OPEN of regular-hours
        open on a weekday. Lets the bot pre-scan the watchlist so by 09:30
        ET it already has indicator state and a candidate list ready."""
        warmup_minutes = Config().WARMUP_MINUTES_BEFORE_OPEN
        if warmup_minutes <= 0:
            return False
        try:
            import pytz
            from datetime import time, timedelta
            now_et = datetime.now(pytz.timezone("US/Eastern"))
            if now_et.weekday() >= 5:  # weekend
                return False
            open_dt = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            warmup_start = open_dt - timedelta(minutes=warmup_minutes)
            return warmup_start <= now_et < open_dt
        except Exception:
            return False

    def is_market_hours(self) -> tuple[bool, str]:
        """Check if we're in regular market hours
        
        Returns:
            (is_regular_hours, session_type)
            session_type: 'regular', 'premarket', 'afterhours', or 'closed'
        """
        from datetime import time
        import pytz
        
        # Get current time in Eastern timezone
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        current_time = now.time()
        weekday = now.weekday()
        
        # Skip weekends
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return False, 'closed'
        
        # Define market sessions (Eastern Time)
        premarket_start = time(4, 0)    # 4:00 AM ET
        regular_start = time(9, 30)     # 9:30 AM ET
        regular_end = time(16, 0)       # 4:00 PM ET
        afterhours_end = time(20, 0)    # 8:00 PM ET
        
        # Check which session we're in
        if current_time < premarket_start:
            return False, 'closed'
        elif current_time < regular_start:
            return False, 'premarket'
        elif current_time < regular_end:
            return True, 'regular'
        elif current_time < afterhours_end:
            return False, 'afterhours'
        else:
            return False, 'closed'
    
    def should_use_limit_order(self) -> tuple[bool, str]:
        """Determine if we should use limit orders based on market hours
        
        Returns:
            (use_limit, reason)
        """
        is_regular, session = self.is_market_hours()
        
        if is_regular:
            return False, "Regular hours - market orders OK"
        
        if session == 'closed':
            return True, "Market closed - orders will queue"
        elif session == 'premarket':
            if not self.allow_premarket:
                return True, "Premarket trading disabled"
            return True, "Premarket - using limit orders for safety"
        elif session == 'afterhours':
            if not self.allow_afterhours:
                return True, "After-hours trading disabled"
            return True, "After-hours - using limit orders for safety"
        
        return True, "Extended hours - using limit orders"
    
    def get_todays_trades(self):
        """Get actual trades from Schwab (both open positions and closed trades from today).

        v-orders-cache-2026-05-04: this function is called from the dashboard
        WS broadcast at 4 Hz. Without a cache it makes 4–5 Schwab REST calls
        per second, hammering the orders endpoint and risking rate-limit while
        live exposure is open. Cache for 5 seconds — orders state changes on
        the order of seconds, so freshness is unaffected.
        """
        if self.mode == TradingMode.LIVE and self.schwab_client:
            now = time.time()
            cached = getattr(self, "_schwab_trades_cache", None)
            cached_at = getattr(self, "_schwab_trades_cache_at", 0.0)
            if cached is not None and (now - cached_at) < 5.0:
                return cached
            try:
                fresh = self.get_schwab_trades_today()
            except Exception:
                # On REST failure, return last-known cache if we have one,
                # rather than letting the dashboard payload break.
                return cached if cached is not None else []
            self._schwab_trades_cache = fresh
            self._schwab_trades_cache_at = now
            return fresh
        else:
            # Fallback to internal trade history for simulation mode
            return self.get_internal_trades_today()
    
    def get_internal_trades_today(self):
        """Get trades from internal history (used for simulation)"""
        today = datetime.now().date()
        todays_trades = []
        
        for trade in self.trade_history:
            # Parse the entry time
            try:
                if isinstance(trade.get('entry_time'), str):
                    entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                else:
                    entry_time = trade.get('entry_time', datetime.now())
                
                # Check if trade is from today
                if entry_time.date() == today:
                    todays_trades.append(trade)
            except Exception as e:
                logger.debug(f"Error parsing trade time: {e}")
                # If we can't parse the time, include it if it's recent
                if len(self.trade_history) <= 50 and trade in self.trade_history[-50:]:
                    todays_trades.append(trade)
        
        # Sort by entry time (most recent first)
        todays_trades.sort(key=lambda x: x.get('entry_time', ''), reverse=True)
        
        return todays_trades
    
    def get_schwab_trades_today(self):
        """Get actual trades from Schwab API for today"""
        try:
            today = datetime.now().date()
            todays_trades = []
            
            # Get today's filled orders from Schwab
            from_time = datetime.combine(today, datetime.min.time())
            to_time = datetime.now()
            
            # Get orders for today
            response = self.schwab_client.get_orders_for_account(
                self.account_hash,
                from_entered_datetime=from_time,
                to_entered_datetime=to_time
            )
            
            if response.status_code == 200:
                orders = response.json()
                
                for order in orders:
                    # Process each order
                    status = order.get('status', '')
                    
                    # Include both FILLED orders (closed trades) and WORKING orders (open positions)
                    if status in ['FILLED', 'WORKING', 'QUEUED', 'ACCEPTED']:
                        # Extract order details
                        order_activities = order.get('orderActivityCollection', [])
                        order_legs = order.get('orderLegCollection', [])
                        
                        for leg in order_legs:
                            symbol = leg.get('instrument', {}).get('symbol', 'Unknown')
                            quantity = leg.get('quantity', 0)
                            instruction = leg.get('instruction', '')  # BUY, SELL, etc.
                            
                            # Get execution details if filled
                            if order_activities:
                                for activity in order_activities:
                                    execution_legs = activity.get('executionLegs', [])
                                    for exec_leg in execution_legs:
                                        exec_price = exec_leg.get('price', 0)
                                        exec_quantity = exec_leg.get('quantity', 0)
                                        exec_time = exec_leg.get('time', '')
                                        
                                        # Create trade record
                                        trade = {
                                            'symbol': symbol,
                                            'entry_time': order.get('enteredTime', ''),
                                            'exit_time': exec_time if status == 'FILLED' else None,
                                            'entry_price': exec_price if instruction in ['BUY', 'BUY_TO_OPEN'] else 0,
                                            'exit_price': exec_price if instruction in ['SELL', 'SELL_TO_CLOSE'] else 0,
                                            'quantity': exec_quantity,
                                            'status': status,
                                            'instruction': instruction,
                                            'pnl': 0,  # Will calculate below
                                            'reason': 'Schwab Order'
                                        }
                                        todays_trades.append(trade)
                            else:
                                # For working orders without execution yet
                                if status == 'WORKING':
                                    # Get current position info to show unrealized P&L
                                    positions = self.positions.get(symbol)
                                    current_pnl = positions.unrealized_pnl if positions else 0
                                    
                                    trade = {
                                        'symbol': symbol,
                                        'entry_time': order.get('enteredTime', ''),
                                        'exit_time': None,  # Still open
                                        'entry_price': order.get('price', 0),
                                        'exit_price': 0,
                                        'quantity': quantity,
                                        'status': 'OPEN',
                                        'instruction': instruction,
                                        'pnl': current_pnl,
                                        'reason': 'Open Position'
                                    }
                                    todays_trades.append(trade)
            
            # Also add current open positions that might not have orders today
            current_positions = self.positions
            for symbol, position in current_positions.items():
                # Check if we already have this position in trades
                if not any(t['symbol'] == symbol and t['status'] == 'OPEN' for t in todays_trades):
                    trade = {
                        'symbol': symbol,
                        'entry_time': position.entry_time.isoformat() if hasattr(position.entry_time, 'isoformat') else str(position.entry_time),
                        'exit_time': None,
                        'entry_price': position.entry_price,
                        'exit_price': 0,
                        'quantity': position.quantity,
                        'status': 'OPEN',
                        'pnl': position.unrealized_pnl,
                        'reason': 'Open Position'
                    }
                    todays_trades.append(trade)
            
            # v-trades-fifo-pairing-2026-06-10: SELL fills used to ship
            # entry_price=0 / pnl=0 (the operator's +$1,089 MU scalp
            # rendered as a zero-P&L stub). Pair same-day SELL fills
            # against BUY fills FIFO per symbol so closed round-trips
            # show their realized P&L. Cross-day basis is left at 0 —
            # Schwab's cost-basis endpoint is the authority there.
            _buys: dict = {}
            for t in sorted(todays_trades, key=lambda x: x.get('exit_time')
                            or x.get('entry_time') or ''):
                if t.get('instruction') in ('BUY', 'BUY_TO_OPEN') and t.get('entry_price'):
                    _buys.setdefault(t['symbol'], []).append(
                        [t['quantity'], t['entry_price']])
                elif (t.get('instruction') in ('SELL', 'SELL_TO_CLOSE')
                      and t.get('exit_price') and not t.get('entry_price')):
                    lots = _buys.get(t['symbol'], [])
                    remaining = t['quantity']
                    cost = 0.0
                    matched = 0.0
                    while lots and remaining > 0:
                        lot_qty, lot_px = lots[0]
                        take = min(lot_qty, remaining)
                        cost += take * lot_px
                        matched += take
                        remaining -= take
                        lot_qty -= take
                        if lot_qty <= 0:
                            lots.pop(0)
                        else:
                            lots[0][0] = lot_qty
                    if matched > 0:
                        avg_cost = cost / matched
                        t['entry_price'] = avg_cost
                        t['pnl'] = round(
                            (t['exit_price'] - avg_cost) * matched, 2)
                        t['status'] = 'CLOSED'
                        t['reason'] = 'Round trip (same day)'

            # Sort by time (most recent first)
            todays_trades.sort(key=lambda x: x.get('entry_time', ''), reverse=True)

            return todays_trades

        except Exception as e:
            logger.error(f"Error getting Schwab trades: {e}")
            # Fallback to internal history
            return self.get_internal_trades_today()
    
    async def _broadcast_trade_update(self, trade_record):
        """Broadcast a single trade update to all connected WebSocket clients"""
        if hasattr(self, 'connection_manager') and self.connection_manager:
            try:
                await self._broadcast_ui({
                    'type': 'trade_update',
                    'data': {
                        'new_trade': trade_record,
                        'all_trades': self.get_todays_trades()
                    }
                })
            except Exception as e:
                logger.debug(f"Error broadcasting trade update: {e}")
    
    def _save_state(self):
        """Save current trading state including simulated positions."""
        # Save real position metadata (long-term flags etc.)
        # v-ownership-survives-restart-2026-06-10: persist the ownership
        # and bracket fields too. Their absence is why the 11:03 restart
        # demoted the bot's own NET trade to external — the rediscovery
        # path had nothing to restore from.
        positions_data = {}
        for symbol, pos in self.positions.items():
            positions_data[symbol] = {
                'is_long_term': getattr(pos, 'is_long_term', False),
                'entry_price': pos.entry_price,
                'quantity': pos.quantity,
                'side': pos.side,
                'entry_time': pos.entry_time.isoformat(),
                'managed_by_bot': getattr(pos, 'managed_by_bot', False),
                'stop_loss': getattr(pos, 'stop_loss', 0),
                # inf (external positions' "no target") is not valid
                # strict JSON — store None and restore as inf.
                'take_profit': (
                    None if getattr(pos, 'take_profit', 0) == float('inf')
                    else getattr(pos, 'take_profit', 0)
                ),
                'state': getattr(pos, 'state', None),
                # v-trade-record-ml-columns-2026-09-02: without this, a
                # position surviving a restart loses confidence/
                # meta_proba/kelly_fraction and its close writes NULL
                # ML columns to bot_trades (sim_data already saves it).
                'reasoning': getattr(pos, 'reasoning', None) or {},
            }

        # Save simulated positions in full so they survive restarts.
        # Without this, every restart wipes sim positions → duplicate
        # entries, lost P&L tracking, and broken trailing stops.
        sim_data = {}
        for symbol, pos in self.simulated_positions.items():
            if pos is None or pos.quantity <= 0:
                continue
            sim_data[symbol] = {
                'symbol': pos.symbol,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'quantity': pos.quantity,
                'side': pos.side,
                'stop_loss': pos.stop_loss,
                'take_profit': pos.take_profit,
                'entry_time': pos.entry_time.isoformat(),
                'unrealized_pnl': pos.unrealized_pnl,
                'reasoning': pos.reasoning,
                'is_long_term': getattr(pos, 'is_long_term', False),
                'scaled_out': getattr(pos, 'scaled_out', False),
                'original_stop': getattr(pos, 'original_stop', None),
                'trailing_stop': getattr(pos, 'trailing_stop', None),
                'mode': getattr(pos, 'mode', 'simulation'),
                'peak_favorable_r': getattr(pos, 'peak_favorable_r', 0.0),
                'breakeven_lifted': getattr(pos, 'breakeven_lifted', False),
                'managed_by_bot': getattr(pos, 'managed_by_bot', True),
                'last_revalidation_at': getattr(pos, 'last_revalidation_at', None),
                # v-position-fsm-2026-04-30 (Phase 1): persist FSM fields
                # so a restart can recover an in-flight close intent.
                'state': getattr(pos, 'state', PositionState.LIVE.value),
                'state_reason': getattr(pos, 'state_reason', 'initial'),
                'state_changed_at': getattr(pos, 'state_changed_at', None),
                'exiting_started_at': getattr(pos, 'exiting_started_at', None),
                'zombie_reason': getattr(pos, 'zombie_reason', None),
            }

        state = {
            'trade_history': self.trade_history[-100:],
            'schwab_pnl': self.risk_manager.schwab_daily_pnl,
            'consecutive_losses': self.risk_manager.consecutive_losses,
            # v-consec-loss-daily-reset-2026-05-21: persist last_loss_date.
            'last_loss_date': (
                self.risk_manager.last_loss_date.isoformat()
                if self.risk_manager.last_loss_date is not None else None
            ),
            'positions_data': positions_data,
            'simulated_positions': sim_data,
            'last_save': datetime.now().isoformat()
        }

        with open("trading_state.json", 'w') as f:
            json.dump(state, f, indent=2, default=str)

        # Sync open positions to Postgres for SQL queryability.
        # v-live-wins-merge-2026-04-30: order matters — when the same
        # symbol exists in both dicts (e.g. user has real PLTR at Schwab
        # AND the bot has a sim PLTR trade), the LIVE record must win
        # because that's the user's actual money. Previously the merge
        # was {**positions, **simulated_positions} and sim overwrote live.
        if self.db_logger is not None:
            all_positions = {**self.simulated_positions, **self.positions}
            try:
                asyncio.get_event_loop().create_task(
                    self.db_logger.sync_positions(all_positions)
                )
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────
    # v-position-fsm-2026-04-30 (Phase 1): FSM helpers.
    # ──────────────────────────────────────────────────────────────────
    EXITING_ZOMBIE_SEC: int = 60  # if EXITING held longer than this → ZOMBIE

    def _ensure_position_lock(self, position) -> None:
        """Lazily attach an asyncio.Lock to a Position.

        Locks are PER-POSITION (not engine-global) so two different
        symbols can transition concurrently. Same-symbol transitions
        are serialised through the same lock — that's what makes the
        check-and-swap in try_transition() race-free.
        """
        if position._state_lock is None:
            try:
                position._state_lock = asyncio.Lock()
            except RuntimeError:
                # No running event loop — let try_transition handle it.
                pass

    async def _recover_inflight_exits(self) -> None:
        """Restart-safety check: positions persisted in EXITING state
        were mid-close when the bot stopped. We do NOT re-trigger a
        new close (that would risk a double-fill in live mode). Two
        cases:

          (a) Live position whose broker order may have actually filled
              — we need to verify with Schwab. If the position is gone
              from Schwab's side, mark CLOSED. If still present, the
              prior close did not stick: promote to ZOMBIE for operator
              review (could be partial fill, rejected order, etc.).
          (b) Sim position — there's no broker. The previous EXITING
              was definitionally an in-memory bookkeeping intent that
              didn't complete. Auto-recover: revert state to LIVE so
              the engine can decide afresh.

        This method runs once at engine.start() before the main loop.
        """
        all_positions = list(self.positions.items()) + list(self.simulated_positions.items())
        for sym, pos in all_positions:
            if pos is None or pos.state != PositionState.EXITING.value:
                continue
            self._ensure_position_lock(pos)
            self._audit(
                "position_fsm", sym, "recovery_inspect",
                "found_in_exiting_state",
                exiting_started_at=pos.exiting_started_at,
                mode=pos.mode,
            )
            if getattr(pos, "mode", "simulation") == "live" and self.schwab_client:
                # Verify with broker
                try:
                    schwab_positions = await self.get_schwab_positions()
                    still_held = next((p for p in schwab_positions if p.get("symbol") == sym), None)
                    if still_held is None:
                        # Schwab no longer holds it — close DID complete
                        pos.state = PositionState.CLOSED.value
                        pos.state_reason = "recovery_close_confirmed"
                        pos.state_changed_at = datetime.now(timezone.utc).isoformat()
                        self._audit("position_fsm", sym, "recovery_resolved",
                                    "close_confirmed_at_broker")
                    else:
                        # Schwab still has it; close did not stick → ZOMBIE
                        pos.state = PositionState.ZOMBIE.value
                        pos.zombie_reason = "exiting_persisted_but_broker_still_holds_position"
                        pos.state_changed_at = datetime.now(timezone.utc).isoformat()
                        self._audit("position_fsm", sym, "recovery_zombie",
                                    "broker_still_holds")
                except Exception as exc:
                    pos.state = PositionState.ZOMBIE.value
                    pos.zombie_reason = f"recovery_failed: {exc}"
                    pos.state_changed_at = datetime.now(timezone.utc).isoformat()
                    self._audit("position_fsm", sym, "recovery_zombie",
                                "schwab_check_failed", err=str(exc)[:80])
            else:
                # Sim path — no broker to verify against. Revert to LIVE
                # so the engine reconsiders the exit cleanly. Audit it so
                # the operator can see the recovery happened.
                pos.state = PositionState.LIVE.value
                pos.state_reason = "recovery_sim_revert"
                pos.state_changed_at = datetime.now(timezone.utc).isoformat()
                pos.exiting_started_at = None
                self._audit("position_fsm", sym, "recovery_resolved",
                            "sim_reverted_to_live")

    async def _promote_zombie_if_stalled(self, position) -> bool:
        """Live-loop safeguard: if a position has been in EXITING for
        longer than EXITING_ZOMBIE_SEC, the close clearly didn't go
        through. Promote to ZOMBIE so:
          (a) the engine stops trying to manage it
          (b) the dashboard surfaces it for operator action
        Returns True if promoted (caller should skip further exit logic).
        """
        if position.state != PositionState.EXITING.value:
            return False
        try:
            started = position.exiting_started_at
            if not started:
                return False
            t0 = datetime.fromisoformat(started)
            age = (datetime.now(timezone.utc) - t0).total_seconds()
        except Exception:
            return False
        if age < self.EXITING_ZOMBIE_SEC:
            return False
        # Promote
        self._ensure_position_lock(position)
        async with position._state_lock:
            if position.state == PositionState.EXITING.value:
                position.state = PositionState.ZOMBIE.value
                position.zombie_reason = f"exiting_stalled_{int(age)}s"
                position.state_changed_at = datetime.now(timezone.utc).isoformat()
                self._audit(
                    "position_fsm", position.symbol, "promoted_to_zombie",
                    "exiting_stalled",
                    age_sec=int(age),
                )
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=position.symbol,
                    title=f"🧟 Position Stuck in EXITING — Promoted to ZOMBIE",
                    message=(
                        f"{position.symbol} stayed in EXITING for {int(age)}s "
                        f"(limit {self.EXITING_ZOMBIE_SEC}s). The close did "
                        "not complete. Bot is hands-off until you resolve "
                        "this from the dashboard."
                    ),
                    importance=10,
                ))
        return True

    # ──────────────────────────────────────────────────────────────────
    # v-loop-decoupling-2026-04-30 (Phase 2): three-task implementation.
    #   _quote_streamer_loop  — refresh QuoteCache at QUOTE_REFRESH_SEC
    #   _position_loop        — fast FSM dispatcher at POSITION_LOOP_SEC
    #   _analysis_loop        — slow signal generation at ANALYSIS_LOOP_SEC
    # All three are run under TaskSupervisor for crash isolation.
    # ──────────────────────────────────────────────────────────────────

    def _symbols_with_open_positions(self) -> List[str]:
        """Snapshot of every symbol the bot has skin in. Used by the
        streamer (knows which symbols to fetch) and the position loop
        (knows which positions to evaluate). Snapshotting via list()
        avoids 'dict changed size during iteration' under concurrent
        opens/closes from analysis_loop."""
        syms = set()
        syms.update(self.positions.keys())
        syms.update(self.simulated_positions.keys())
        return list(syms)

    async def _fetch_one_quote(self, symbol: str) -> None:
        """Single-symbol fetch + cache write. Bounded by the semaphore.
        Network call wrapped in run_in_executor with a hard timeout so
        a hanging Schwab response can't lock up the streamer."""
        if self._quote_fetch_sem is None:
            return
        async with self._quote_fetch_sem:
            if self.data_provider is None:
                return
            try:
                loop = asyncio.get_event_loop()
                quote = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, self.data_provider.get_quote, symbol,
                    ),
                    timeout=Config().QUOTE_FETCH_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                await self.quote_cache.mark_stale(symbol, "fetch_timeout")
                return
            except Exception as exc:
                await self.quote_cache.mark_stale(symbol, f"fetch_err:{type(exc).__name__}")
                return
            if not quote:
                return
            last = float(quote.get("last") or quote.get("price") or 0)
            bid = float(quote.get("bid") or 0)
            ask = float(quote.get("ask") or 0)
            if last <= 0 and (bid <= 0 or ask <= 0):
                return
            # v-per-symbol-fallback-2026-05-01: write to canonical
            # PriceBook (not just legacy QuoteCache). Same partial-merge
            # semantics as the stream path.
            try:
                self.price_book.apply_tick_sync(
                    symbol, last=last if last > 0 else None,
                    bid=bid if bid > 0 else None,
                    ask=ask if ask > 0 else None,
                    source="rest",
                )
            except Exception:
                pass
            await self.quote_cache.put(CachedQuote(
                symbol=symbol,
                last=last,
                bid=float(quote.get("bid") or 0),
                ask=float(quote.get("ask") or 0),
                source="schwab",
            ))

    async def _state_persistence_loop(self) -> None:
        """v-state-persistence-loop-2026-05-27: periodic trading_state.json
        save.

        Background. sync_positions_with_schwab updates self.positions in
        memory when external closes are detected (line 3577), but
        trading_state.json was only written by ad-hoc _save_state calls
        scattered through the codebase. When a user manually closed a
        bot-tracked position outside an active save path, the on-disk
        state stayed stale indefinitely — until the bot's next restart
        forced a sync.

        Trigger: 2026-05-27 user closed QCOM at ~15:30. Bot's in-memory
        view caught up via sync_positions_with_schwab (QCOM correctly
        removed from self.positions, API /api/positions/db returned 6
        positions without QCOM), but trading_state.json mtime stayed
        15:32:47 with stale QCOM entry intact for 38+ minutes.

        Fix: a dedicated NORMAL-priority loop that calls _save_state
        every _SAVE_INTERVAL_SEC. Writes are cheap (~50KB JSON file).
        Errors are caught and logged so a transient write failure
        doesn't kill the loop. Loop checks self.is_running so it
        exits cleanly during shutdown.
        """
        _SAVE_INTERVAL_SEC = 30
        # Don't save immediately on startup — give the bot a moment
        # to finish initial position sync and brain load so we don't
        # write a half-built state file.
        await asyncio.sleep(_SAVE_INTERVAL_SEC)
        while self.is_running:
            try:
                self._save_state()
            except Exception as exc:
                logger.warning(
                    "state_persistence_loop: save failed: %s", exc,
                )
            await asyncio.sleep(_SAVE_INTERVAL_SEC)

    async def _market_indices_loop(self) -> None:
        """v-market-indices-strip-2026-05-27: refresh dashboard regime strip.

        Calls ``MarketIndicesCache.instance().refresh(schwab_client)``
        every ``_REFRESH_SEC`` seconds. Single Schwab batched call per
        cycle, which gates against per-tab amplification we'd get with
        on-demand fetching. Failures are logged at DEBUG; the strip
        renders a "stale" badge after 30s without an update.
        """
        _REFRESH_SEC = 10
        # Light initial sleep so we don't hammer Schwab in the first
        # second of startup while other loops are still initializing.
        await asyncio.sleep(2)
        from core.market_indices import MarketIndicesCache
        cache = MarketIndicesCache.instance()
        while self.is_running:
            try:
                if self.schwab_client is not None:
                    cache.refresh(self.schwab_client)
            except Exception as exc:
                logger.debug("market_indices_loop iteration failed: %s", exc)
            await asyncio.sleep(_REFRESH_SEC)

    async def _quote_streamer_loop(self) -> None:
        """Position-interest reconciler — stream-only, no REST fallback.

        v-stream-only-2026-05-01: this loop is ONLY the position-scope
        register_interest emitter. It MUST NOT call
        schwab_stream.update_subscriptions() directly — doing so clobbers
        the PriceBook union (positions ∪ watchlist ∪ ticker_tape) with
        positions-only and triggers a 20-symbol unsub burst that Schwab
        punishes as churn. Position interest flows through PriceBook the
        same way watchlist and ticker_tape do.

        v-no-rest-2026-05-01: REST safety-belt removed. If the stream
        goes unhealthy, the dashboard surfaces is_stale via the
        PriceObject; we do NOT pull REST quotes. Per directive: a
        symbol with no recent stream tick has no recent price change —
        do not synthesize one from REST.
        """
        while self.is_running:
            symbols = self._symbols_with_open_positions()
            try:
                await self.price_book.register_interest("positions", set(symbols))
            except Exception:
                pass
            self._quote_streamer_healthy = (
                self._schwab_quote_stream is not None
                and self._schwab_quote_stream.is_healthy()
            )
            await asyncio.sleep(Config().QUOTE_REFRESH_SEC)

    async def _evaluate_exits_with_cached_quote(
        self, position, quote=None,
    ) -> None:
        """The FSM dispatcher driven by the canonical PriceBook mark.

        v-pricebook-2026-05-01: source-of-truth migration. The `quote`
        argument is now optional (legacy callers may still pass a
        CachedQuote, which we ignore in favour of PriceBook). The
        canonical mark for THIS exit decision comes from
        price_book.get_mark(symbol). If the PriceObject is stale or
        unpriced, we skip the tick — matching the prior behaviour of
        "don't close on yesterday's price."

        Phase 1 invariants preserved verbatim:
          - each exit rule's invocation is gated by is_exit_rule_allowed()
          - the decision-to-close goes through try_transition() to EXITING
          - the FSM advances on transition triggers (BREAKEVEN_LIFT, SCALE_OUT_1R)
        """
        if position is None or position.quantity <= 0:
            return
        if not self._is_auto_managed(position):
            return

        symbol = position.symbol
        # Reactive-pull mark. Single source of truth.
        price_obj = self.price_book.get_mark(symbol)
        if price_obj.is_stale or not price_obj.has_price:
            # No safe mark → skip exit evaluation. The dashboard's
            # "is_stale" indicator will surface this to the user.
            return
        current_price = price_obj.price
        # P&L is now derived (Position has no stored mark/pnl); we
        # don't write it anywhere. Anyone needing it computes from
        # PriceBook + CalculationEngine. The Position class's @property
        # setters for current_price / unrealized_pnl are silent no-ops
        # during the migration (they used to be stored fields); writing
        # them here would just be dead code. Removed.
        self._ensure_position_lock(position)

        # Watchdog: stuck-EXITING auto-promotion.
        if await self._promote_zombie_if_stalled(position):
            return

        try:
            _state = PositionState(position.state)
        except ValueError:
            position.state = PositionState.ZOMBIE.value
            position.zombie_reason = f"unknown_state_{position.state!r}"
            return
        if _state in (PositionState.EXITING, PositionState.CLOSED, PositionState.ZOMBIE):
            return

        # ── HARD STOP ── (always evaluated when allowed; the safety belt)
        if is_exit_rule_allowed(_state, ExitRule.HARD_STOP):
            stop = position.stop_loss
            if stop and stop > 0:
                breached = (
                    (position.side == "long"  and current_price <= stop) or
                    (position.side == "short" and current_price >= stop)
                )
                if breached:
                    if not await try_transition(
                        position, PositionState.EXITING,
                        "hard_stop_breached",
                        audit_fn=self._audit,
                    ):
                        return
                    self._audit("hard_stop", symbol, "exit",
                                "stop_breached",
                                price=round(current_price, 4),
                                stop=round(stop, 4))
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.DECISION,
                        symbol=symbol,
                        title=f"🛑 Stop Hit: {symbol}",
                        message=f"Price ${current_price:.4f} crossed stop ${stop:.4f}.",
                        importance=9,
                    ))
                    await self._close_position_with_commentary(position, "stop_loss")
                    return

        # ── TAKE PROFIT ──
        if is_exit_rule_allowed(_state, ExitRule.TAKE_PROFIT):
            tp = position.take_profit
            if tp and tp > 0:
                hit = (
                    (position.side == "long"  and current_price >= tp) or
                    (position.side == "short" and current_price <= tp)
                )
                if hit:
                    if not await try_transition(
                        position, PositionState.EXITING,
                        "take_profit_reached",
                        audit_fn=self._audit,
                    ):
                        return
                    self._audit("take_profit", symbol, "exit",
                                "tp_reached",
                                price=round(current_price, 4),
                                target=round(tp, 4))
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.DECISION,
                        symbol=symbol,
                        title=f"🎯 Take-Profit Hit: {symbol}",
                        message=f"Target ${tp:.4f} reached at ${current_price:.4f}.",
                        importance=9,
                    ))
                    await self._close_position_with_commentary(position, "take_profit")
                    return

        # ── BREAKEVEN LIFT ── (LIVE → AT_BREAKEVEN)
        if is_exit_rule_allowed(_state, ExitRule.BREAKEVEN_LIFT):
            # v-oversold-v2-2026-05-19: per-position activation override
            # from reasoning. OversoldBounceV2 stores breakeven_atr_mult
            # (e.g. 1.5 = lift at +1.5×ATR); engine converts to R-multiple
            # using stop_distance. Other strategies' positions don't carry
            # this key → falls back to Config().BREAKEVEN_ACTIVATION_R.
            _r_dict = getattr(position, 'reasoning', {}) or {}
            _be_mult = _r_dict.get('breakeven_atr_mult')
            _be_atr = _r_dict.get('atr')
            _be_sd = _r_dict.get('stop_distance')
            if _be_mult and _be_atr and _be_sd and _be_sd > 0:
                _be_activation_r = _be_mult * _be_atr / _be_sd
            else:
                _be_activation_r = Config().BREAKEVEN_ACTIVATION_R
            be_new_stop = self.scale_trail.update_peak_and_breakeven(
                position, current_price, _be_activation_r,
            )
            if be_new_stop is not None:
                old_stop = position.stop_loss
                position.stop_loss = be_new_stop
                position.breakeven_lifted = True
                self._audit("breakeven_stop", symbol, "lifted",
                            "peak_above_activation",
                            peak_r=round(position.peak_favorable_r, 3),
                            old_stop=round(old_stop, 2),
                            new_stop=round(be_new_stop, 2))
                await try_transition(
                    position, PositionState.AT_BREAKEVEN,
                    "+0.5R reached, stop ratcheted to entry",
                    audit_fn=self._audit,
                )
                _state = PositionState(position.state)

        # ── SCALE OUT 1R ── (AT_BREAKEVEN → AT_1R)
        if is_exit_rule_allowed(_state, ExitRule.SCALE_OUT_1R):
            # v-live-exit-ladder-2026-06-11: this block is BOOKKEEPING-
            # ONLY (no real order placed). Fine in sim — in live it
            # would desync share counts from Schwab while the OCO
            # still pledges the full size. Gated until partial closes
            # are implemented end-to-end (cancel → sell → verify →
            # re-bracket). Audited so the gap stays visible.
            if (self.mode == TradingMode.LIVE
                    and not Config().ENABLE_LIVE_SCALE_OUT):
                self._audit("scale_out", symbol, "scale_out_skipped",
                            "live_partials_not_armed")
                _so_override = None
            else:
                # v-oversold-v2-2026-05-19: per-position activation
                # override. OversoldBounceV2 stores
                # scale_out_r_override=0.5 (its stop_distance = 2×ATR,
                # so +0.5R = +1×ATR for the partial). Other strategies
                # fall back to 1.0R (legacy behavior).
                _so_override = (getattr(position, 'reasoning', {}) or {}).get(
                    'scale_out_r_override', 1.0,
                )
            partial = (
                self.scale_trail.check_partial_exit_at_r(
                    position, current_price, activation_r=_so_override,
                ) if _so_override is not None else None
            )
            if partial is not None and partial.exit_qty > 0:
                position.quantity -= partial.exit_qty
                position.scaled_out = True
                position.stop_loss = partial.new_stop
                self._audit("scale_out", symbol, "partial_exit_1R",
                            "1R_reached",
                            exit_qty=partial.exit_qty,
                            remaining=position.quantity,
                            new_stop=round(partial.new_stop, 2))
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=symbol,
                    title=f"🎯 1R Reached — Scaled Out 50%",
                    message=(f"Closed {partial.exit_qty} shares; keeping "
                             f"{position.quantity}. Stop now ${partial.new_stop:.2f}."),
                    importance=9,
                ))
                await try_transition(
                    position, PositionState.AT_1R,
                    "+1R reached, 50% scaled out",
                    audit_fn=self._audit,
                )
                _state = PositionState(position.state)

        # ── TRAILING STOP ──
        # Allowed in AT_BREAKEVEN, AT_1R, TRAILING (per Phase 1 + the
        # v-trail-from-breakeven fix that closed the AT_BREAKEVEN gap).
        if is_exit_rule_allowed(_state, ExitRule.TRAILING_STOP_ATR):
            # ATR is computed in the analysis loop and stamped on the
            # position via reasoning. If absent, skip — analysis loop
            # will populate it on its next tick.
            atr = (getattr(position, "reasoning", {}) or {}).get("atr")
            if atr is not None and atr > 0:
                new_trail = self.scale_trail.update_trailing_stop(
                    position, current_price, float(atr),
                )
                if new_trail is not None:
                    old_trail = position.trailing_stop
                    position.trailing_stop = new_trail
                    if _state == PositionState.AT_1R:
                        await try_transition(
                            position, PositionState.TRAILING,
                            "trailing stop engaged",
                            audit_fn=self._audit,
                        )
                        _state = PositionState(position.state)
                    self._audit("trailing_stop", symbol, "update",
                                "atr_trail",
                                old=round(old_trail or 0, 2),
                                new=round(new_trail, 2),
                                atr=round(float(atr), 4))

                if self.scale_trail.is_trailing_stop_hit(position, current_price):
                    if not await try_transition(
                        position, PositionState.EXITING,
                        "trailing_stop_hit",
                        audit_fn=self._audit,
                    ):
                        return
                    self._audit("trailing_stop", symbol, "exit",
                                "atr_trail_hit",
                                price=round(current_price, 4),
                                trail=round(position.trailing_stop, 4))
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.DECISION,
                        symbol=symbol,
                        title=f"📉 Trailing Stop Hit: {symbol}",
                        message=(f"Trail ${position.trailing_stop:.4f} breached "
                                 f"at ${current_price:.4f}. Locking gains."),
                        importance=9,
                    ))
                    await self._close_position_with_commentary(position, "trailing_stop_atr")
                    return

    async def _position_loop(self) -> None:
        """Fast FSM dispatcher. Reads cached quotes only — never calls
        the broker. Each tick iterates all open positions and evaluates
        the FSM-permitted exit rules for the current state."""
        cadence = Config().POSITION_LOOP_SEC
        while self.is_running:
            try:
                # Snapshot first — list() so a concurrent open/close in
                # analysis_loop doesn't raise dict-changed-size.
                positions = []
                for sym in self._symbols_with_open_positions():
                    pos = self.simulated_positions.get(sym) or self.positions.get(sym)
                    if pos is not None:
                        positions.append((sym, pos))

                # v-pricebook-2026-05-01: tell the PriceBook which
                # symbols positions care about right now. The book
                # reconciles the union (positions ∪ watchlist ∪ chart)
                # and asks the Schwab stream to subscribe diff-only.
                try:
                    await self.price_book.register_interest(
                        "positions", set(s for s, _ in positions),
                    )
                except Exception:
                    pass

                for sym, pos in positions:
                    # PriceBook is consulted INSIDE the evaluator now;
                    # we don't need to fetch a quote at this layer.
                    try:
                        await self._evaluate_exits_with_cached_quote(pos)
                    except Exception as exc:
                        # Per-position failure should NOT crash the loop.
                        logger.error(
                            "position_loop: error evaluating %s: %s",
                            sym, exc, exc_info=True,
                        )
            except Exception:
                # Outer guard: re-raise so supervisor can restart this loop.
                # We caught per-position; anything reaching here is structural.
                raise
            await asyncio.sleep(cadence)

    async def _analysis_loop(self) -> None:
        """Phase 2: thin wrapper around the existing master-loop body.

        The master loop's content (market-hours pause/resume, schwab
        sync, screener, indicator computation, ML predict, strategy
        signals, signal routing) lives in `_analysis_loop_body()`.
        This wrapper exists so the supervisor can manage it as one
        coroutine. The `while self.is_running:` is INSIDE the body so
        a supervisor restart correctly resumes from the top of the
        loop.

        The `_quote_streamer_healthy` gate is checked inside the body
        before any new-entry signal-routing call.
        """
        await self._analysis_loop_body()

    async def _broadcast_ui(self, payload: dict) -> None:
        """Broadcast to UI safely when engine runs on a worker thread."""
        cm = getattr(self, "connection_manager", None)
        if cm is None:
            return
        target_loop = getattr(self, "_ws_broadcast_loop", None)
        current_loop = asyncio.get_running_loop()
        if target_loop is not None and target_loop is not current_loop:
            fut = asyncio.run_coroutine_threadsafe(
                cm.broadcast(payload),
                target_loop,
            )
            await asyncio.wrap_future(fut)
            return
        await cm.broadcast(payload)

    def _build_supervisor(self) -> TaskSupervisor:
        """Construct the supervisor with the engine's tasks registered.
        Critical-priority for position_loop because exit safety depends
        on it.

        v-schwab-stream-2026-05-01 (Phase 4a): when streaming is
        available AND we have a Schwab client, register the websocket
        stream as a NORMAL-priority task. The polling streamer stays
        registered too — when the stream is healthy, polling becomes a
        no-op (auto-detected via .is_healthy()); when the stream
        disconnects, polling resumes automatically.
        """
        sup = TaskSupervisor(
            max_crashes=Config().TASK_RESTART_MAX_CRASHES,
            window_sec=Config().TASK_RESTART_WINDOW_SEC,
            on_crash=self._on_task_crash,
            on_circuit_broken=self._on_task_circuit_broken,
        )

        # Build the stream object iff library + client are present.
        if STREAMING_AVAILABLE and self.schwab_client and self.account_hash:
            # v-stream-raw-2026-05-01: forward each raw Schwab message
            # to all WS clients as 'schwab_raw' so the /stream debug
            # view can render exactly what the broker is sending.
            def _publish_raw(msg):
                try:
                    cm = self.connection_manager
                    if cm is None:
                        return
                    target_loop = self._ws_broadcast_loop
                    if target_loop is None:
                        target_loop = asyncio.get_event_loop()
                    asyncio.run_coroutine_threadsafe(
                        cm.broadcast({"type": "schwab_raw", "data": msg}),
                        target_loop,
                    )
                except Exception as exc:
                    logger.debug(f"_publish_raw failed: {exc}")
            self._schwab_quote_stream = SchwabQuoteStream(
                schwab_client=self.schwab_client,
                account_hash=self.account_hash,
                quote_cache=self.quote_cache,
                on_raw_message=_publish_raw,
            )
            # v-pricebook-2026-05-01: bridge PriceBook subscription
            # changes to the stream. When position_loop / watchlist /
            # chart-picker register a new symbol of interest, the book
            # fires this callback with the new union and the stream
            # subscribes/unsubscribes diff-only.
            self.price_book.set_subscription_callback(
                self._schwab_quote_stream.update_subscriptions,
            )
            sup.register("schwab_stream",
                         self._schwab_quote_stream.run,
                         TaskPriority.NORMAL)
            logger.info("schwab_stream registered (Phase 4a real-time quotes)")
        else:
            logger.info("schwab_stream NOT registered "
                        "(streaming_available=%s, has_client=%s, has_account=%s) — "
                        "REST polling only",
                        STREAMING_AVAILABLE,
                        self.schwab_client is not None,
                        bool(self.account_hash))

        sup.register("quote_streamer", self._quote_streamer_loop, TaskPriority.NORMAL)
        sup.register("position_loop",  self._position_loop,        TaskPriority.CRITICAL)
        sup.register("analysis_loop",  self._analysis_loop,        TaskPriority.NORMAL)
        # v-market-indices-strip-2026-05-27: regime strip refresh loop.
        # NORMAL priority — purely cosmetic, must never starve trading
        # decisions. Failures degrade gracefully (strip shows "stale").
        sup.register("market_indices", self._market_indices_loop,  TaskPriority.NORMAL)
        # v-state-persistence-loop-2026-05-27: every 30s, write the
        # current in-memory positions/trade_history/risk state to
        # trading_state.json. Without this loop trading_state.json
        # would only update on ad-hoc save calls and a user's manual
        # close of a bot-tracked position would leave the file stale
        # for minutes-to-hours. NORMAL priority.
        sup.register("state_persistence", self._state_persistence_loop, TaskPriority.NORMAL)

        # v-parallel-screener-loop-2026-05-12: extracted from
        # `_analyze_markets_with_commentary` so a crash in the signal
        # pipeline can no longer freeze the watchlist refresh.
        from core.loops.screener_loop import ScreenerLoop
        self._screener_loop = ScreenerLoop(self)
        sup.register("screener_loop", self._screener_loop.run, TaskPriority.NORMAL)

        # v-newsbus-2026-09-08: news_loop refreshes the NewsBus from
        # FreeNewsAggregator on a ~20s cadence. Strategies read from
        # the bus instead of fetching independently. High-impact items
        # trigger wake events for the analysis loop.
        from core.loops.news_loop import NewsLoop
        from core.news_bus import get_news_bus
        self._news_bus = get_news_bus(
            on_high_impact=self._on_news_high_impact,
        )
        self._news_loop = NewsLoop(self, bus=self._news_bus)
        sup.register("news_loop", self._news_loop.run, TaskPriority.NORMAL)

        return sup

    def _on_news_high_impact(self, symbol: str, item) -> None:
        """Callback for high-impact news events from NewsBus.
        
        Sets the wake event so the analysis loop can evaluate entry
        opportunities early instead of waiting for the next cadence tick.
        """
        try:
            logger.info(
                "news_high_impact_wake symbol=%s headline=%s sentiment=%.3f",
                symbol, item.headline[:60], item.sentiment_score,
            )
        except Exception:
            pass

    def _on_task_crash(self, st, exc) -> None:
        """Surface task crashes on the dashboard. NORMAL — we don't pause
        trading; supervisor handles restart."""
        try:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=None,
                title=f"⚠️ Background Task Crashed: {st.name}",
                message=(
                    f"Task `{st.name}` raised {type(exc).__name__}: {exc}. "
                    "Supervisor will restart with backoff."
                ),
                importance=7,
            ))
        except Exception:
            pass

    def _on_task_circuit_broken(self, st) -> None:
        """A task has tripped its breaker. Stop entries (analysis_loop)
        or alert loudly (position_loop). Operator must reset via API."""
        importance = 10 if st.priority == TaskPriority.CRITICAL else 9
        try:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=None,
                title=f"🚨 Task Circuit Broken: {st.name}",
                message=(
                    f"`{st.name}` crashed {len(st.crash_times)} times. "
                    "Auto-restart disabled. Operator action required: "
                    "POST /api/supervisor/reset/{name} when fixed."
                ),
                importance=importance,
            ))
        except Exception:
            pass
        # Critical: if position_loop is broken, halt new entries too.
        if st.priority == TaskPriority.CRITICAL:
            self._quote_streamer_healthy = False  # acts as analysis_loop gate

    def _init_schwab_client(self):
        """Initialize Schwab client with commentary"""
        try:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.TECHNICAL,
                symbol=None,
                title="🔌 Connecting to Schwab API",
                message="Attempting to establish connection with broker...",
                importance=7
            ))
            
            # Check credentials
            if not Config().SCHWAB_API_KEY:
                raise ValueError("SCHWAB_API_KEY not set")
            if not Config().SCHWAB_APP_SECRET:
                raise ValueError("SCHWAB_SECRET not set")
            
            # Check if token exists
            if Config().SCHWAB_TOKEN_PATH.exists():
                try:
                    # Fixed token loading
                    with open(Config().SCHWAB_TOKEN_PATH, 'r') as f:
                        token_file_data = json.load(f)
                        # Handle both formats
                        if 'token' in token_file_data:
                            token_data = token_file_data['token']
                        else:
                            token_data = token_file_data
                    
                    # v-token-health-2026-05-23: warn the operator if the
                    # Schwab refresh token is close to expiring (7-day TTL
                    # from creation). Non-fatal — just logs a warning so
                    # the user knows to re-auth before the bot starts
                    # failing mid-session.
                    try:
                        from core.token_health import check_token_health
                        check_token_health(Path(Config().SCHWAB_TOKEN_PATH))
                    except Exception as _tok_exc:
                        logger.debug("token_health check failed: %s", _tok_exc)

                    # Create client from token
                    self.schwab_client = auth.client_from_token_file(
                        Config().SCHWAB_TOKEN_PATH,
                        Config().SCHWAB_API_KEY,
                        Config().SCHWAB_APP_SECRET
                    )
                    
                    # Test if token is valid
                    response_hash = self.schwab_client.get_account_numbers()
                    response = self.schwab_client.get_accounts()
                    if response.status_code == 200:
                        self.data_provider = SchwabDataProvider(self.schwab_client, self.commentary)
                        accounts = response.json()
                        accounts_hash = response_hash.json()
                        #logger.info(f"Account Info: {accounts}")
                        if accounts:
                            self.account_id = Config().SCHWAB_ACCOUNT_NUMBER or accounts[0].get('securitiesAccount', {}).get('accountNumber')
                            self.account_hash = accounts_hash[0].get('hashValue')
                                                        
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.TECHNICAL,
                            symbol=None,
                            title="✅ Schwab Connected",
                            message="Successfully connected to Schwab API. Ready to trade!",
                            importance=8
                        ))
                        return
                    else:
                        logger.warning("Token expired or invalid")
                except Exception as e:
                    logger.warning(f"Token loading failed: {e}")
            
            # If we get here, need to authenticate
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=None,
                title="⚠️ Authentication Required",
                message="Schwab token not found or invalid. Running in simulation mode with REAL market data from yfinance.",
                importance=8
            ))

            # Use RealTimeDataProvider for real prices even in simulation mode
            self.data_provider = RealTimeDataProvider(self.commentary)

        except Exception as e:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=None,
                title="⚠️ Broker Connection Issue",
                message=f"Could not connect to Schwab: {str(e)}. Running in simulation mode with REAL market data.",
                importance=8
            ))

            # Use RealTimeDataProvider for real prices even in simulation mode
            self.data_provider = RealTimeDataProvider(self.commentary)

    async def _reconcile_external_closes(self, closed_symbols, prior_positions):
        """Write bot_trades rows for positions closed outside the bot.

        v-external-close-reconcile-2026-05-07: when a symbol disappears
        from Schwab between syncs, the user closed it manually (or a
        broker-side stop fired). The bot's exit pipeline never ran, so
        log_trade was never called. Reconcile against Schwab's filled
        orders and write a synthetic bot_trades row so:
          * the dashboard's "Recent Bot Trades" tile shows what really
            happened to the position,
          * /api/trades?date= queries return correct realized P&L,
          * TradingBrain has real-money outcomes to learn from instead
            of only sim closes.

        Strategy:
          1. Pull today's filled orders from Schwab (one REST call,
             reused via the existing 5s order cache).
          2. For each closed symbol, find the most recent SELL execution
             leg that matches the prior position's quantity and side.
          3. Compute P&L from the prior Position's entry vs the fill.
          4. Best-effort log_trade with exit_reason='external_close'.

        Robustness: every step is best-effort and isolated; any failure
        on a single symbol is logged and skipped, not raised.
        """
        if not closed_symbols or self.schwab_client is None or self.db_logger is None:
            return

        # Pull today's filled orders. Use existing helper which already
        # handles auth + 5s cache; falls back gracefully if the API errors.
        try:
            schwab_orders = await asyncio.to_thread(
                self.get_schwab_trades_today
            )
        except Exception as exc:
            logger.debug("reconcile_close: order fetch failed: %s", exc)
            return

        if not schwab_orders:
            logger.debug(
                "reconcile_close: %d symbols closed but no Schwab orders today (%s)",
                len(closed_symbols), closed_symbols,
            )
            return

        # Index SELL fills per symbol, most recent first.
        sells_by_symbol = {}
        for order in schwab_orders:
            try:
                sym = (order.get("symbol") or "").upper()
                instr = order.get("instruction") or ""
                if (
                    not sym
                    or sym not in closed_symbols
                    or instr not in ("SELL", "SELL_TO_CLOSE", "SELL_SHORT", "BUY_TO_COVER")
                    or order.get("status") not in ("FILLED",)
                ):
                    continue
                exit_price = float(order.get("exit_price") or 0)
                if exit_price <= 0:
                    continue
                sells_by_symbol.setdefault(sym, []).append(order)
            except Exception:
                continue

        for sym in closed_symbols:
            try:
                prior_pos = prior_positions.get(sym)
                if prior_pos is None:
                    continue
                fills = sells_by_symbol.get(sym, [])
                if not fills:
                    logger.info(
                        "reconcile_close: %s closed externally but no Schwab "
                        "SELL fill found in today's orders — skipping bot_trades log",
                        sym,
                    )
                    continue

                # Best fill: most recent SELL whose qty matches the prior
                # position. If none matches exactly, take the largest qty.
                prior_qty = int(getattr(prior_pos, "quantity", 0) or 0)
                exact = [f for f in fills if int(f.get("quantity") or 0) == prior_qty]
                fill = (exact or sorted(
                    fills, key=lambda f: int(f.get("quantity") or 0), reverse=True
                ))[0]

                exit_price = float(fill.get("exit_price") or 0)
                exit_time_raw = fill.get("exit_time") or datetime.now().isoformat()
                # Schwab times are ISO with Z suffix; parse defensively
                try:
                    exit_time = datetime.fromisoformat(
                        str(exit_time_raw).replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except Exception:
                    exit_time = datetime.now()

                entry_price = float(getattr(prior_pos, "entry_price", 0) or 0)
                quantity = prior_qty or int(fill.get("quantity") or 0)
                side = getattr(prior_pos, "side", "long") or "long"
                entry_time = getattr(prior_pos, "entry_time", None) or datetime.now()
                if isinstance(entry_time, str):
                    try:
                        entry_time = datetime.fromisoformat(entry_time)
                    except Exception:
                        entry_time = datetime.now()

                if side == "short":
                    pnl = (entry_price - exit_price) * quantity
                else:
                    pnl = (exit_price - entry_price) * quantity
                pnl_pct = (
                    (pnl / (entry_price * quantity)) * 100.0
                    if entry_price > 0 and quantity > 0
                    else 0.0
                )

                reasoning = getattr(prior_pos, "reasoning", None) or {}
                strategy = (
                    (reasoning.get("strategy") if isinstance(reasoning, dict) else None)
                    or "external"
                )

                await self.db_logger.log_trade(
                    symbol=sym,
                    side=side,
                    strategy=strategy,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=int(quantity),
                    pnl=float(pnl),
                    pnl_pct=float(pnl_pct),
                    exit_reason="external_close",
                    atr_at_entry=(
                        reasoning.get("atr") if isinstance(reasoning, dict) else None
                    ),
                    stop_loss=float(getattr(prior_pos, "stop_loss", 0) or 0) or None,
                    take_profit=(
                        float(getattr(prior_pos, "take_profit", 0) or 0)
                        if getattr(prior_pos, "take_profit", 0) not in (float("inf"), 0, None)
                        else None
                    ),
                    confidence=(
                        reasoning.get("confidence") if isinstance(reasoning, dict) else None
                    ),
                    meta_proba=(
                        reasoning.get("meta_proba") if isinstance(reasoning, dict) else None
                    ),
                    kelly_fraction=(
                        reasoning.get("kelly_fraction") if isinstance(reasoning, dict) else None
                    ),
                    scaled_out=bool(getattr(prior_pos, "scaled_out", False)),
                    mode=getattr(prior_pos, "mode", "live"),
                    reasoning=(
                        {**reasoning, "external_close": True}
                        if isinstance(reasoning, dict)
                        else {"external_close": True}
                    ),
                )

                self._audit(
                    "external_close_reconcile", sym, "logged",
                    "manual_close",
                    qty=int(quantity),
                    entry=round(entry_price, 4),
                    exit=round(exit_price, 4),
                    pnl=round(float(pnl), 2),
                    pnl_pct=round(float(pnl_pct), 2),
                    side=side,
                )
                logger.info(
                    "reconcile_close: %s %s qty=%d entry=%.4f exit=%.4f pnl=%+.2f (%+.2f%%) — bot_trades row written",
                    sym, side, int(quantity), entry_price, exit_price, pnl, pnl_pct,
                )

                # v-brain-learns-from-manual-closes-2026-05-08: feed the
                # reconciled close into TradingBrain so the learning loop
                # sees real-money outcomes, not just sim closes. Same
                # call signature as the bot-driven close path. Best-effort
                # — a brain failure must not roll back the bot_trades row.
                try:
                    brain = getattr(self, "brain", None)
                    if brain is not None and hasattr(brain, "remember_trade"):
                        try:
                            holding_min = (
                                exit_time - entry_time
                            ).total_seconds() / 60.0
                        except Exception:
                            holding_min = 0.0
                        brain.remember_trade(
                            symbol=sym,
                            pattern=strategy,
                            outcome=("win" if pnl > 0 else "loss"),
                            pnl_percent=float(pnl_pct),
                            context={
                                "exit_reason": "external_close",
                                "holding_time": holding_min,
                                "max_profit": float(pnl),
                                "market_conditions": getattr(self, "market_state", {}),
                                "external_close": True,
                            },
                        )
                        logger.debug(
                            "reconcile_close: %s fed to brain (%s, %+.2f%%)",
                            sym, ("win" if pnl > 0 else "loss"), pnl_pct,
                        )
                except Exception as brain_exc:
                    logger.debug(
                        "reconcile_close: %s brain.remember_trade failed: %s",
                        sym, brain_exc,
                    )
            except Exception as exc:
                logger.warning(
                    "reconcile_close: %s reconciliation failed: %s",
                    sym, exc,
                )

    async def sync_positions_with_schwab(self):
        """Sync internal position tracking with actual Schwab positions.

        v-sync-preserve-bot-state-2026-05-05: prior implementation called
        ``self.positions.clear()`` every cycle and rebuilt every Position
        with ``managed_by_bot=False`` plus ``is_external=True``. That
        destroyed:
          * the ``managed_by_bot=True`` flag set when the bot itself opens
            a live trade (engine.py:1559),
          * any user toggle via /api/toggle-managed-by-bot (routes.py:2315),
          * the bot's planned ``stop_loss`` / ``take_profit`` / ``entry_time``,
          * trailing-stop / peak-favorable-R / scaled-out state.
        Sync now preserves the per-symbol bot state and only updates the
        fields Schwab is authoritative for (qty, average_price, side,
        current_price, total_pnl).
        """
        if not self.schwab_client or self.mode != TradingMode.LIVE:
            return

        try:
            schwab_positions = await self.get_schwab_positions()

            # Snapshot existing tracked state per symbol before the merge.
            # We preserve everything except the Schwab-authoritative fields
            # (qty, entry/avg price, current price, side, unrealized_pnl).
            prior = dict(self.positions)
            schwab_symbols = {pd['symbol'] for pd in schwab_positions}

            for pos_data in schwab_positions:
                symbol = pos_data['symbol']
                qty_signed = pos_data['quantity']
                side = 'long' if qty_signed > 0 else 'short'
                qty_abs = abs(qty_signed)

                existing = prior.get(symbol)
                if existing is not None:
                    # Merge: keep bot-managed flags, refresh Schwab fields.
                    existing.entry_price = pos_data['average_price']
                    existing.current_price = pos_data['current_price']
                    existing.quantity = qty_abs
                    existing.side = side
                    existing.unrealized_pnl = pos_data.get('total_pnl', 0)
                    self.positions[symbol] = existing
                else:
                    # Newly discovered position (pre-existing on Schwab,
                    # or opened outside the bot). Default to hands-off —
                    # UNLESS the saved state proves the bot owned it.
                    #
                    # v-ownership-survives-restart-2026-06-10: after a
                    # restart self.positions is empty, so every position
                    # lands here and used to be demoted to external —
                    # the bot disowned its own NET trade on 2026-06-10.
                    # Restore ownership only on strict identity match:
                    # saved record says managed_by_bot=True AND side
                    # matches AND quantity matches (drift means the
                    # operator intervened — stay hands-off).
                    saved = self._saved_positions_meta.get(symbol) or {}
                    _restore = (
                        saved.get('managed_by_bot') is True
                        and saved.get('side') == side
                        and abs(float(saved.get('quantity', -1)) - qty_abs) < 1e-6
                    )
                    if _restore:
                        try:
                            _entry_time = datetime.fromisoformat(
                                saved['entry_time'])
                        except (KeyError, ValueError):
                            _entry_time = datetime.now() - timedelta(hours=1)
                        _tp = saved.get('take_profit')
                        position = Position(
                            symbol=symbol,
                            entry_price=pos_data['average_price'],
                            current_price=pos_data['current_price'],
                            quantity=qty_abs,
                            side=side,
                            stop_loss=saved.get('stop_loss', 0) or 0,
                            take_profit=float('inf') if _tp is None else _tp,
                            entry_time=_entry_time,
                            mode="live",
                            managed_by_bot=True,
                            # v-trade-record-ml-columns-2026-09-02:
                            # rehydrate reasoning so ML columns survive
                            # a restart-then-close.
                            reasoning=saved.get('reasoning') or {},
                        )
                        position.is_external = False
                        position.is_manually_managed = False
                        self._audit(
                            "position_sync", symbol,
                            "position_ownership_restored",
                            "saved_state_identity_match",
                            side=side, quantity=qty_abs,
                            stop=saved.get('stop_loss', 0),
                            target=_tp,
                        )
                    else:
                        position = Position(
                            symbol=symbol,
                            entry_price=pos_data['average_price'],
                            current_price=pos_data['current_price'],
                            quantity=qty_abs,
                            side=side,
                            stop_loss=0,
                            take_profit=float('inf'),
                            entry_time=datetime.now() - timedelta(hours=1),
                            mode="live",
                            managed_by_bot=False,
                        )
                        position.is_external = True
                        position.is_manually_managed = True
                    position.unrealized_pnl = pos_data.get('total_pnl', 0)
                    self.positions[symbol] = position

            # Drop tracked positions that no longer exist on Schwab
            # (closed externally, or shares all sold). Do NOT touch any
            # bot-opened position that's mid-flight if the cache is empty
            # or stale — Schwab's REST view IS authoritative when we got
            # a non-empty response.
            if schwab_positions:
                stale = [s for s in self.positions if s not in schwab_symbols]
                # v-external-close-reconcile-2026-05-07: when a symbol
                # disappears from Schwab between syncs, the user (or a
                # broker stop) closed it outside the bot's exit pipeline.
                # Bot's bot_trades table never sees that close, so the
                # brain learns from zero real-money outcomes. Reconcile
                # against Schwab fill history and write the missing trade
                # rows BEFORE we drop the prior Position state.
                if stale:
                    try:
                        await self._reconcile_external_closes(stale, prior)
                    except Exception as exc:
                        logger.warning(
                            "external_close_reconcile_error err=%s",
                            exc,
                        )
                for s in stale:
                    self.positions.pop(s, None)

            managed_count = sum(
                1 for p in self.positions.values()
                if bool(getattr(p, 'managed_by_bot', False))
            )
            logger.info(
                "Synced %d positions from Schwab (%d bot-managed, %d hands-off)",
                len(self.positions), managed_count,
                len(self.positions) - managed_count,
            )

            # Update commentary
            if self.positions:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.ACCOUNT_UPDATE,
                    symbol=None,
                    title="📊 Position Sync Complete",
                    message=f"Synced {len(self.positions)} positions from Schwab: {', '.join(self.positions.keys())}\n"
                           f"⚠️ All positions marked as MANUALLY MANAGED - bot will not auto-close",
                    importance=5
                ))
                
        except Exception as e:
            logger.error(f"Failed to sync positions with Schwab: {e}")
    
    async def start(self):
        """Start the trading engine with commentary"""
        self.is_running = True
        # v-uptime-2026-06-10: /api/status reads this; it was never set,
        # so the dashboard showed uptime 0:00:00 forever.
        self.start_time = datetime.now()

        # v-feature-snapshot-2026-09-09: ensure snapshot table exists
        if self.db_logger is not None:
            try:
                await self.db_logger.ensure_snapshot_table()
                logger.info("snapshot_table_ready")
            except Exception as exc:
                logger.warning("snapshot_table_create_failed: %s", exc)

        # v-startup-schwab-sync-2026-04-30: previously this branch only
        # ran in LIVE mode, so SIM-mode dashboards saw the default
        # $100,000 / $50,000 placeholder values for the entire 5-cycle
        # warm-up before the periodic sync first fired (~5 min). Now the
        # initial Schwab account+positions sync runs whenever the
        # schwab_client exists, regardless of mode — the user wants real
        # account numbers visible immediately even when the engine is
        # only paper-trading. Wrapped in try/except so a Schwab outage
        # doesn't block startup.
        if self.schwab_client:
            try:
                acct = await self._get_real_account_info()
                if acct:
                    self.risk_manager.sync_with_schwab_data(acct)
                    logger.info(
                        f"startup_schwab_sync balance=${acct.get('balance', 0):.2f} "
                        f"buying_power=${acct.get('buying_power', 0):.2f} "
                        f"day_pnl=${acct.get('day_pnl', 0):.2f}"
                    )
            except Exception as exc:
                logger.warning(f"startup Schwab account sync failed: {exc}")
            try:
                # _update_and_track_real_positions pulls live Schwab
                # holdings INTO self.positions. _save_state then mirrors
                # them to the bot_positions Postgres table so /api/positions/db
                # (which reads from Postgres, not memory) sees them.
                # Both are needed at startup; previously _save_state only
                # ran periodically inside the main loop, so the dashboard
                # showed 0 live positions until the first save tick.
                await self._update_and_track_real_positions()
                # v-schwab-positions-cache-2026-04-30: also seed the
                # per-position Schwab cache so the dashboard's per-trade
                # P&L is correct from the first WS tick instead of
                # waiting 5 cycles for the periodic refresh.
                try:
                    self._schwab_positions_cache = await self.get_schwab_positions()
                except Exception as exc:
                    logger.debug(f"startup schwab_positions_cache seed failed: {exc}")
                self._save_state()
            except Exception as exc:
                logger.warning(f"startup live positions sync failed: {exc}")

        # v-position-fsm-2026-04-30 (Phase 1): recover any positions
        # persisted in EXITING. Either confirm with broker, revert (sim),
        # or promote to ZOMBIE for operator review. Runs BEFORE the main
        # loop so the first management tick sees a clean FSM.
        try:
            await self._recover_inflight_exits()
        except Exception as exc:
            logger.warning(f"_recover_inflight_exits failed: {exc}")

        # Lazy-init per-position locks for everything currently in tracking.
        # The locks couldn't be created in __init__ (no event loop yet);
        # creating them here once means try_transition's lazy fallback
        # never has to fire under normal flow.
        for _pos in list(self.positions.values()) + list(self.simulated_positions.values()):
            if _pos is not None:
                self._ensure_position_lock(_pos)

        risk_per_trade = Config().MAX_RISK_PER_TRADE
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.MARKET_ANALYSIS,
            symbol=None,
            title="🎯 Trading Session Started",
            message="Beginning market analysis. I'll explain each step of my decision-making process.",
            data={
                'mode': self.mode.value,
                'account_balance': self.risk_manager.account_balance,
                'risk_per_trade': f"{risk_per_trade:.1%}"
            },
            importance=9
        ))
        
        # In start() method, after self.is_running = True
        if not self.ml_predictor.model.is_trained:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="🎯 Initial Model Training Required",
                message="Collecting historical data for first-time model training...",
                importance=9
            ))
            # Train on initial symbols
            self.ml_predictor.retrain_model(
                self.data_provider,
                ['PLTR', 'NVDA', 'TSLA']
            )
        
        # Sync with existing Schwab positions on startup
        if self.mode == TradingMode.LIVE and self.schwab_client:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="🔄 Syncing with Schwab Account",
                message="Checking for existing positions in your account...",
                importance=8
            ))
            
            await self._update_and_track_real_positions()
            
            if self.positions:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.MARKET_ANALYSIS,
                    symbol=None,
                    title=f"📊 Found {len(self.positions)} Existing Positions",
                    message=f"Tracking positions: {', '.join(self.positions.keys())}",
                    importance=8
                ))

        # v-loop-decoupling-2026-04-30 (Phase 2): replace single master
        # loop with supervised three-task model. The master loop body
        # below moved into _analysis_loop_body(). _position_loop and
        # _quote_streamer_loop run alongside under the supervisor.
        # _quote_fetch_sem must be created here, NOT in __init__, so it's
        # bound to this event loop.
        self._quote_fetch_sem = asyncio.Semaphore(Config().QUOTE_FETCH_CONCURRENCY)
        self._supervisor = self._build_supervisor()
        self._supervisor.start_all()

        # The supervisor's tasks own the work; this coroutine just waits
        # for is_running to flip to False (via /api/stop or signal).
        # When that happens, stop_all() cancels the tasks gracefully.
        try:
            while self.is_running:
                await asyncio.sleep(0.5)
        finally:
            await self._supervisor.stop_all()

    async def _analysis_loop_body(self) -> None:
        """The legacy master loop, lifted verbatim from the old start()
        body. Now driven by the TaskSupervisor instead of being the
        engine's only thread of control. Position management is
        deliberately removed from here (handled by _position_loop)."""
        analysis_count = 0
        self._paused_for_session: Optional[str] = None  # track session to avoid log spam

        while self.is_running:
            try:
                # v-off-hours-analysis-2026-05-01: analysis NEVER pauses now.
                # Previously the loop did `continue` outside tradable hours,
                # which silenced screener / indicator / news / ML work. The
                # user wants the bot to keep THINKING off-hours and only
                # stop ORDERING. The signal_router's market_hours gate
                # (engine.py: _process_signal_with_commentary) already
                # blocks new entries during off-hours — that's the right
                # place for "no orders," not the analysis loop.
                #
                # We do still slow the cadence off-hours: data sources
                # don't update meaningfully overnight, and burning a full
                # screener every 60s is wasted Schwab budget. _tradable
                # is computed and stamped on `self._is_tradable_now` so
                # the cadence at the end of this iteration knows which
                # sleep to apply.
                is_regular, session = self.is_market_hours()
                in_warmup = self._in_warmup_window()
                tradable = (
                    is_regular
                    or (session == "premarket" and self.allow_premarket)
                    or (session == "afterhours" and self.allow_afterhours)
                    or in_warmup  # pre-open analysis window — loop runs but
                                  # signal-router gate still blocks entries
                )
                self._is_tradable_now = tradable

                # One-shot session-change announcements so the dashboard
                # shows when the market opens/closes, but no pausing.
                if not tradable and self._paused_for_session != session:
                    next_open = self._get_next_market_open()
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.MARKET_ANALYSIS,
                        symbol=None,
                        title=f"🌙 Market {session.title()} — Analysis Continues, Orders Held",
                        message=(f"Markets are currently {session}. "
                                 f"Bot is still analysing the market and tracking "
                                 f"existing positions; new orders are held until "
                                 f"the next regular session at {next_open}."),
                        data={"session": session, "next_open": next_open,
                              "allow_premarket": self.allow_premarket,
                              "allow_afterhours": self.allow_afterhours},
                        importance=4,
                    ))
                    self._audit("market_hours", None, "orders_held",
                                f"session_{session}", next_open=next_open)
                    self._paused_for_session = session

                if tradable and self._paused_for_session is not None:
                    if in_warmup:
                        title = "🔎 Pre-Market Warmup — Scanning Watchlist"
                        msg = (f"{Config().WARMUP_MINUTES_BEFORE_OPEN} minutes "
                               "before regular open. Building indicator state "
                               "and ranking candidates. New entries still blocked "
                               "until 09:30 ET.")
                        resume_reason = "warmup"
                    else:
                        title = "☀️ Market Open — Orders Resumed"
                        msg = f"Entering {session} session. New orders allowed."
                        resume_reason = f"session_{session}"
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.MARKET_ANALYSIS,
                        symbol=None,
                        title=title,
                        message=msg,
                        importance=6,
                    ))
                    self._audit("market_hours", None, "resume", resume_reason)
                    self._paused_for_session = None

                analysis_count += 1
                
                # Sync with Schwab account data periodically (every 5 analyses)
                if analysis_count % 5 == 0 and self.schwab_client:
                    try:
                        # Sync account info
                        account_info = await self._get_real_account_info()
                        if account_info:
                            self.risk_manager.sync_with_schwab_data(account_info)
                            
                            # Log the sync to commentary
                            self.commentary.add_commentary(TradingCommentary(
                                timestamp=datetime.now(),
                                type=CommentaryType.ACCOUNT_UPDATE,
                                symbol=None,
                                title="📊 Account Sync",
                                message=f"Synced with Schwab: Daily P&L=${account_info.get('day_pnl', 0):.2f}",
                                data={'schwab_pnl': account_info.get('day_pnl', 0),
                                      'buying_power': account_info.get('buying_power', 0)},
                                importance=4
                            ))
                        
                        # Also sync positions every 5 analyses
                        await self.sync_positions_with_schwab()

                        # v-schwab-positions-cache-2026-04-30: refresh
                        # per-position cache so the dashboard shows
                        # Schwab's authoritative P&L instead of the
                        # bot's recomputed number.
                        try:
                            self._schwab_positions_cache = await self.get_schwab_positions()
                        except Exception as exc:
                            logger.debug(f"schwab_positions_cache refresh failed: {exc}")

                    except Exception as e:
                        logger.error(f"Failed to sync with Schwab: {e}")
                
                # Update position prices before analysis
                await self._update_position_prices()
                
                # Risk assessment with detailed explanation
                await self._assess_market_conditions()
                await self._analyze_premarket_gaps() 
                # Check if we should trade (with explanation)
                should_trade, reason = await self._evaluate_trading_conditions()
                
                if should_trade:
                    # Analyze markets with commentary
                    await self._analyze_markets_with_commentary()
                else:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=None,
                        title="🛑 Trading Paused",
                        message=f"Not trading right now: {reason}",
                        importance=6
                    ))
                
                # Manage existing positions with commentary
                await self._manage_positions_with_commentary()
                
                # Emergency stop check - ONLY use Schwab P&L
                # Get fresh P&L from Schwab for accuracy
                schwab_pnl = 0
                if self.schwab_client and analysis_count % 5 == 0:  # Check every 5 cycles
                    fresh_info = await self._get_real_account_info()
                    if fresh_info:
                        schwab_pnl = fresh_info.get('day_pnl', 0)
                        # Update risk manager with Schwab data
                        self.risk_manager.schwab_daily_pnl = schwab_pnl
                        # v-bot-only-pnl-circuit-2026-06-08: also update
                        # bot_daily_pnl on the same cadence so the
                        # circuit reads fresh bot-only P&L (used by
                        # can_trade when ENABLE_BOT_ONLY_PNL_CIRCUIT
                        # is True). Failure here must NOT raise — the
                        # circuit falls back to the prior bot_daily_pnl
                        # value (initialized 0). Computation is pure
                        # local state — never blocks on network.
                        try:
                            self.risk_manager.bot_daily_pnl = self._compute_bot_daily_pnl()
                            logger.info(
                                "Bot-only P&L Check: $%.2f (Schwab P&L: $%.2f, "
                                "Limit: $%.2f)",
                                self.risk_manager.bot_daily_pnl,
                                schwab_pnl,
                                self.risk_manager.account_balance * 0.05,
                            )
                        except Exception as _bp_exc:
                            logger.warning(
                                "compute_bot_daily_pnl failed: %s — "
                                "circuit will use stale bot P&L",
                                _bp_exc,
                            )
                        logger.info(f"Schwab P&L Check: ${schwab_pnl:.2f} (Limit: ${self.risk_manager.account_balance * 0.05:.2f})")
                
                # v-emergency-stop-bot-only-2026-06-17: this emergency
                # stop used to trip on ACCOUNT-WIDE schwab_pnl, which
                # includes external/manual holdings. On 2026-06-17 a
                # manual SPCX position bled past 5% of equity, fired
                # this stop, set is_running=False, and silently zombied
                # the bot (0 bot trades that day). Same flaw the #31
                # circuit fix addressed in risk/manager — but this is a
                # SEPARATE path. Honour the same flag: under
                # ENABLE_BOT_ONLY_PNL_CIRCUIT the emergency stop reads
                # bot-only P&L, so external positions can never kill
                # the engine. Fall back to account-wide only when the
                # circuit is explicitly off.
                _emrg_cfg = Config()
                if _emrg_cfg.ENABLE_BOT_ONLY_PNL_CIRCUIT:
                    _emrg_pnl = self.risk_manager.bot_daily_pnl
                else:
                    _emrg_pnl = schwab_pnl
                # Only check emergency stop with the circuit-selected P&L
                if _emrg_pnl < 0 and abs(_emrg_pnl) > self.risk_manager.account_balance * 0.05:
                    # Double-check before halting. Under the bot-only
                    # circuit, recompute bot P&L from local state (it's
                    # the authoritative, network-free value) — do NOT
                    # abort on account-wide fresh_pnl, which would let
                    # external profit cancel a legitimate bot stop (and
                    # was half of the 2026-06-17 confusion). Account-wide
                    # mode keeps the original fresh-Schwab re-check.
                    if _emrg_cfg.ENABLE_BOT_ONLY_PNL_CIRCUIT:
                        try:
                            _recheck = self._compute_bot_daily_pnl()
                            self.risk_manager.bot_daily_pnl = _recheck
                        except Exception:
                            _recheck = _emrg_pnl
                        if _recheck >= 0 or abs(_recheck) <= self.risk_manager.account_balance * 0.05:
                            logger.warning(
                                "Emergency stop ABORTED - bot-only P&L "
                                "recheck within limit: $%.2f", _recheck)
                            continue
                    elif self.schwab_client:
                        fresh_account_info = await self._get_real_account_info()
                        if fresh_account_info:
                            fresh_pnl = fresh_account_info.get('day_pnl', 0)
                            logger.warning(f"Emergency stop check - Fresh P&L: ${fresh_pnl:.2f}")

                            # If fresh data shows we're not in loss, abort emergency stop
                            if fresh_pnl >= 0:
                                logger.warning(f"Emergency stop ABORTED - Fresh data shows profit: ${fresh_pnl:.2f}")
                                self.commentary.add_commentary(TradingCommentary(
                                    timestamp=datetime.now(),
                                    type=CommentaryType.INFO,
                                    symbol=None,
                                    title="✅ False Alarm",
                                    message=f"Emergency stop cancelled - Account is actually in profit: ${fresh_pnl:.2f}",
                                    importance=7
                                ))
                                continue  # Skip emergency stop
                    
                    # Check if manual close only is enabled
                    if self.manual_close_only:
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=None,
                            title="🚨 EMERGENCY STOP (Manual Mode)",
                            # v-emergency-stop-pnl-var-2026-05-11: var was
                            # renamed schwab_pnl ↑ but f-strings still read
                            # the old `pnl_to_check`. NameError fired in
                            # the emergency-stop branch and propagated up
                            # the trading loop, which silently froze the
                            # screener/watchlist refresh for the rest of
                            # the session. Restore the correct var.
                            message=f"Daily loss of ${abs(schwab_pnl):.2f} exceeded 5% limit (${self.risk_manager.account_balance * 0.05:.2f}). "
                                   f"Manual close only is ON - YOU must close positions manually!",
                            importance=10
                        ))
                        
                        # Stop trading but DON'T close positions
                        self.is_running = False
                        break
                    else:
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=None,
                            title="🚨 EMERGENCY STOP",
                            # v-emergency-stop-pnl-var-2026-05-11: see above.
                            message=f"Daily loss of ${abs(schwab_pnl):.2f} exceeded 5% limit (${self.risk_manager.account_balance * 0.05:.2f}). Closing all positions.",
                            importance=10
                        ))
                        
                        # Close all positions
                        for symbol in list(self.positions.keys()):
                            position = self.positions[symbol]
                            await self._close_position_with_commentary(position, "emergency_stop")
                    
                    self.is_running = False
                    break
                # Check order status for real trades
                if self.mode == TradingMode.LIVE:
                    await self._check_order_status()
                    await self._update_real_positions()
                    if (datetime.now() - self._last_bp_check).total_seconds() > 300:
                        await self._check_buying_power(0)
                
                # After the position management section - retrain ML model periodically
                if analysis_count % 100 == 0:  # Every 100 cycles
                    # ScalpingMLModel doesn't have async retrain_model, just train()
                    # The training happens automatically via update_with_outcome
                    logger.info(f"ML model analysis cycle {analysis_count} - model is self-retraining via online learning")

		        # Save state periodically
                if analysis_count % 10 == 0:  # Every 10 analysis cycles
                    self._save_state()
                    self.brain.save_memories()

                    # v-news-veto-tracker-2026-04-28: evaluate open vetoes
                    # against current prices. Cheap; small open-set bounded
                    # by 4-hour eval window. Fire-and-forget — failure of
                    # this task must never block the management loop.
                    if self.db_logger is not None and self.data_provider is not None:
                        def _quote_lookup(_sym):
                            try:
                                q = self.data_provider.get_quote(_sym)
                                if q:
                                    return q.get('last') or q.get('bid') or q.get('ask')
                            except Exception:
                                return None
                            return None
                        try:
                            asyncio.get_event_loop().create_task(
                                self.db_logger.evaluate_open_news_vetoes(
                                    get_current_price=_quote_lookup,
                                    max_age_hours=4,
                                )
                            )
                        except Exception:
                            pass

                    # Log performance snapshot for analytics
                    positions_count = len(self.positions) + len(self.simulated_positions)
                    total_unrealized = sum(getattr(p, 'unrealized_pnl', 0) for p in self.positions.values())
                    total_unrealized += sum(getattr(p, 'unrealized_pnl', 0) for p in self.simulated_positions.values())

                    log_performance(
                        account_balance=self.risk_manager.account_balance,
                        daily_pnl=self.risk_manager.schwab_daily_pnl if hasattr(self.risk_manager, 'schwab_daily_pnl') else 0,
                        total_pnl=total_unrealized,
                        win_rate=self.performance_analyzer.get_win_rate() if hasattr(self, 'performance_analyzer') else 0,
                        total_trades=len(self.trade_history),
                        open_positions=positions_count,
                        mode=self.mode.value,
                        unrealized_pnl=total_unrealized,
                        buying_power=self.risk_manager.buying_power
                    )

                    # Also save ML model periodically to preserve buffer
                    if hasattr(self, 'ml_predictor') and self.ml_predictor:
                        if hasattr(self.ml_predictor, 'save_model'):
                            self.ml_predictor.save_model(create_version=False)  # Don't create version every time
                # v-off-hours-analysis-2026-05-01: cadence is RTH-aware.
                # Tradable hours: 30s (frequent screener / signal scans).
                # Off-hours: 5 min by default (data barely changes; news
                # arrives episodically). Both knobs are config-tunable.
                #
                # v-newsbus-wake-2026-09-08: analysis loop can wake early
                # on high-impact news alerts. The NewsBus sets the wake
                # event when a HIGH impact item arrives; we check it with
                # wait_for so we either wake early or complete the full
                # cadence sleep. After handling, clear the event so we
                # don't spin. This is the "event-driven entry wake" from
                # the P0 spec — 30s backstop with early wake on alerts.
                cadence = (
                    Config().ANALYSIS_LOOP_RTH_SEC
                    if getattr(self, "_is_tradable_now", True)
                    else Config().ANALYSIS_LOOP_OFF_HOURS_SEC
                )
                wake_reason = "poll"
                try:
                    bus = getattr(self, "_news_bus", None)
                    if bus is not None:
                        wake_event = bus.get_wake_event()
                        try:
                            await asyncio.wait_for(wake_event.wait(), timeout=cadence)
                            wake_reason = "news_alert"
                            bus.clear_wake_event()
                        except asyncio.TimeoutError:
                            pass  # Normal cadence completed
                    else:
                        await asyncio.sleep(cadence)
                except Exception as _wake_exc:
                    logger.debug("analysis_loop wake failed: %s", _wake_exc)
                    await asyncio.sleep(cadence)
                
                self._last_wake_reason = wake_reason
                
            except Exception as e:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=None,
                    title="❌ Error in Trading Loop",
                    message=f"Encountered an error: {str(e)}",
                    importance=9
                ))
                logger.error(f"Trading loop error: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _update_position_prices(self):
        """Update current prices for all positions"""
        try:
            if self.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
                # Update simulated positions
                for symbol, position in self.simulated_positions.items():
                    # Skip positions that are being closed or partially-
                    # constructed (zero quantity or zero entry price).
                    # Overwriting their stored P&L with a nonsense value
                    # can corrupt trading_state.json — especially now that
                    # this refresh also runs during the paused loop.
                    if not position or position.quantity == 0 or position.entry_price == 0:
                        continue
                    try:
                        # Get current quote
                        quote = await self.data_provider.get_current_quote(symbol)
                        if quote and 'price' in quote:
                            old_price = position.current_price
                            position.current_price = quote['price']
                            # v-pnl-sign-fix-2026-04-22: side-aware P&L.
                            # Short profit = entry above current; long the
                            # reverse. Without the branch, a short PLTR
                            # short ends up with long-style P&L (+ when
                            # price rises) overwriting side-aware writers.
                            if position.side == 'short':
                                position.unrealized_pnl = (position.entry_price - position.current_price) * position.quantity
                            else:
                                position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                            # Log significant price movements (guard /0)
                            if old_price:
                                price_change = abs(position.current_price - old_price) / old_price
                                if price_change > 0.01:
                                    logger.debug(f"Updated {symbol} price: ${old_price:.2f} -> ${position.current_price:.2f}")
                    except Exception as e:
                        logger.debug(f"Error updating price for {symbol}: {e}")
            
            elif self.mode == TradingMode.LIVE and self.schwab_client:
                # Update real positions
                loop = asyncio.get_running_loop()
                for symbol, position in self.positions.items():
                    if position:
                        try:
                            response = await asyncio.wait_for(
                                loop.run_in_executor(
                                    None,
                                    self.schwab_client.get_quote,
                                    symbol,
                                ),
                                timeout=Config().QUOTE_FETCH_TIMEOUT_SEC,
                            )
                            if response.status_code == 200:
                                quote_data = response.json()
                                if symbol in quote_data:
                                    old_price = position.current_price
                                    position.current_price = quote_data[symbol]['quote']['lastPrice']
                                    # v-pnl-sign-fix-2026-04-22: side-aware P&L (see matching fix above)
                                    if position.side == 'short':
                                        position.unrealized_pnl = (position.entry_price - position.current_price) * position.quantity
                                    else:
                                        position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                                    
                                    # Log significant price movements
                                    price_change = abs(position.current_price - old_price) / old_price
                                    if price_change > 0.01:  # More than 1% change
                                        logger.debug(f"Updated {symbol} price: ${old_price:.2f} -> ${position.current_price:.2f}")
                        except Exception as e:
                            logger.debug(f"Error updating price for {symbol}: {e}")
                            
        except Exception as e:
            logger.error(f"Error in _update_position_prices: {e}")
    
    async def _assess_market_conditions(self):
        """Assess overall market conditions with detailed commentary"""
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.MARKET_ANALYSIS,
            symbol=None,
            title="🔍 Assessing Market Conditions",
            message="Let me check the overall market health before making any trading decisions...",
            importance=7
        ))
        
        # Get market breadth
        if self.data_provider:
            loop = asyncio.get_running_loop()
            try:
                breadth = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        self.data_provider.calculate_market_breadth,
                    ),
                    timeout=4.0,
                )
            except Exception as exc:
                logger.debug(f"calculate_market_breadth failed: {exc}")
                breadth = {
                    'advance_decline': 0,
                    'new_highs': 0,
                    'new_lows': 0,
                    'vix': 20,
                    'put_call_ratio': 1.0,
                }
            self.market_state['breadth'] = breadth
            self.market_state['vix'] = breadth.get('vix', 20)
            
            # Interpret VIX
            vix_interpretation = ""
            vix_value = self.market_state['vix']
            if vix_value < 15:
                vix_interpretation = "Low volatility - Markets are calm, good for trend following"
            elif vix_value < 25:
                vix_interpretation = "Normal volatility - Standard trading conditions"
            elif vix_value < 35:
                vix_interpretation = "Elevated volatility - Higher risk, but also opportunities"
            else:
                vix_interpretation = "High volatility - Extreme caution needed, reduce position sizes"
            
            # Interpret breadth
            advance_decline = breadth.get('advance_decline', 0)
            breadth_interpretation = ""
            if advance_decline > 1.5:
                breadth_interpretation = "Strong bullish breadth - More stocks advancing"
                self.market_state['market_trend'] = 'bullish'
            elif advance_decline < -1.5:
                breadth_interpretation = "Strong bearish breadth - More stocks declining"
                self.market_state['market_trend'] = 'bearish'
            else:
                breadth_interpretation = "Neutral market breadth"
                self.market_state['market_trend'] = 'neutral'
            
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="📊 Market Breadth Analysis",
                message=f"{vix_interpretation}\n{breadth_interpretation}",
                data={
                    'vix': vix_value,
                    'advance_decline_ratio': advance_decline,
                    'new_highs': breadth.get('new_highs', 0),
                    'new_lows': breadth.get('new_lows', 0),
                    'put_call_ratio': breadth.get('put_call_ratio', 1.0)
                },
                confidence=0.8,
                importance=8
            ))
    
    async def _analyze_premarket_gaps(self):
        """Analyze pre-market gaps for fade opportunities"""
        if 'gaps' not in self.market_state:
            self.market_state['gaps'] = {}
        
        for symbol in self.dynamic_watchlist:
            try:
                quote = self.data_provider.get_quote(symbol)
                if not quote:
                    continue
                    
                # Get today's open and yesterday's close
                open_price = quote.get('open', 0)
                prev_close = quote.get('close', 0)  # This is previous day's close
                
                # Skip if prices are invalid
                if open_price <= 0 or prev_close <= 0:
                    continue
                    
                gap_percent = ((open_price - prev_close) / prev_close) * 100
                
                if abs(gap_percent) > 2:
                    self.market_state['gaps'][symbol] = {
                        'percent': gap_percent,
                        'direction': 'up' if gap_percent > 0 else 'down',
                        'fade_probability': 0.65 if abs(gap_percent) > 5 else 0.45,
                        'open': open_price,
                        'prev_close': prev_close
                    }
                    
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.MARKET_ANALYSIS,
                        symbol=symbol,
                        title=f"🌅 Gap Detected: {symbol}",
                        message=f"{gap_percent:.1f}% gap {self.market_state['gaps'][symbol]['direction']} (Open: ${open_price:.2f}, Prev Close: ${prev_close:.2f})",
                        importance=7
                    ))
            except Exception as e:
                logger.debug(f"Gap analysis error for {symbol}: {e}")

    async def _evaluate_trading_conditions(self) -> Tuple[bool, str]:
        """Evaluate if we should trade with detailed reasoning"""
        # In commentary mode, we always "trade" but don't execute real orders
        if self.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
            if self.risk_manager.margin_call:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=None,
                    title="⚠️ Margin Call Active",
                    message="Account has a margin call! In real trading, I would stop here. "
                            "But in commentary mode, I'll continue analyzing to show you my thought process.",
                    data={
                        'margin_call': True,
                        'buying_power': self.risk_manager.buying_power
                    },
                    importance=10
                ))
            return True, "Simulation mode - always analyze"
        
        if self._is_news_blackout():
            return False, "Major economic news event"
        # For other modes, check actual conditions
        allowed, reason = self.risk_manager.check_trading_allowed()
        
        if not allowed:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=None,
                title="🚫 Trading Conditions Not Met",
                message=f"I'm not trading because: {reason}",
                data={
                    'daily_pnl': self.risk_manager.schwab_daily_pnl,  # Use Schwab P&L
                    'consecutive_losses': self.risk_manager.consecutive_losses,
                    'buying_power': self.risk_manager.buying_power
                },
                importance=8
            ))
        
        return allowed, reason
    
    async def _analyze_markets_with_commentary(self):
        """Analyze markets with detailed commentary.

        v-parallel-screener-loop-2026-05-12: the screener cadence + watchlist
        refresh used to live here. Extracted to `core/loops/screener_loop.py`
        as its own supervised task so a crash anywhere in the signal pipeline
        (see v-emergency-stop-pnl-var-2026-05-11 for the precipitating
        incident) can no longer freeze the watchlist refresh. We now just
        consume `self.dynamic_watchlist` populated by that loop.
        """
        # Use dynamic watchlist
        watchlist = self.dynamic_watchlist

        # v-early-session-soft-2026-04-30: the early-session guard used to
        # `return` here — aborting the entire watchlist scan during 09:30-
        # 09:45 ET. That meant zero analysis, zero strategy_decision logs,
        # zero indicator computation. Looked like the bot had frozen.
        # The veto is now a SOFT one applied later at the entry-acceptance
        # gate (see _process_signal_with_commentary) — analysis runs every
        # tick, only the final entry is blocked. User can watch the bot
        # think, decisions stream live, and at 09:45 the same signals get
        # acted on instead of skipped.

        # Get current positions (both tracked and from Schwab)
        positions_held = set(self.positions.keys())
        
        # Add Schwab positions if in live mode
        if self.mode == TradingMode.LIVE and self.schwab_client:
            schwab_positions = await self.get_schwab_positions()
            for pos in schwab_positions:
                positions_held.add(pos['symbol'])
        
        # Filter watchlist to exclude symbols we already have positions in
        symbols_to_analyze = [s for s in watchlist if s not in positions_held]
        
        if not symbols_to_analyze:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="📊 All Watchlist Symbols Have Positions",
                message=f"Currently holding positions in all watchlist symbols: {', '.join(positions_held)}",
                importance=5
            ))
            return
        
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.MARKET_ANALYSIS,
            symbol=None,
            title="📈 Scanning Watchlist",
            message=f"Analyzing {len(symbols_to_analyze)} symbols for trading opportunities. "
                f"Market trend: {self.market_state['market_trend']}\n"
                f"Skipping {', '.join(positions_held)} (already have positions)",
            importance=6
        ))
        
        for symbol in symbols_to_analyze:
            try:
                # Rate limiting check
                await asyncio.sleep(0.5)
                
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.MARKET_ANALYSIS,
                    symbol=symbol,
                    title=f"🔎 Analyzing {symbol}",
                    message="Fetching price data and calculating technical indicators...",
                    importance=4
                ))
                
                # v-loop-yield-2026-04-30: data_provider.get_market_data and
                # .get_quote are SYNCHRONOUS calls to Schwab. Each one blocks
                # the entire asyncio event loop for ~1-2s while Schwab responds.
                # With 20 watchlist symbols, that's 20-40s per cycle of total
                # blocked time, during which the FastAPI HTTP handlers can't
                # serve a single request. Browser sees the dashboard freeze
                # even though the trading loop itself is healthy.
                # Fix: run both calls in a thread executor with timeouts so
                # the event loop stays responsive throughout.
                if self.data_provider:
                    _loop = asyncio.get_event_loop()
                    try:
                        data = await asyncio.wait_for(
                            _loop.run_in_executor(
                                None, self.data_provider.get_market_data, symbol
                            ),
                            timeout=8.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"market_data timeout for {symbol} — skipping this cycle")
                        continue
                    except Exception as exc:
                        logger.debug(f"market_data error for {symbol}: {exc}")
                        continue
                    if data is None or data.empty:
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=symbol,
                            title=f"⚠️ No Data: {symbol}",
                            message="Could not retrieve price data for this symbol",
                            importance=5
                        ))
                        continue

                    # Yield to the event loop so HTTP/WebSocket handlers can
                    # interleave between the heavy fetch and the heavy analysis.
                    await asyncio.sleep(0)

                    try:
                        quote = await asyncio.wait_for(
                            _loop.run_in_executor(
                                None, self.data_provider.get_quote, symbol
                            ),
                            timeout=4.0,
                        )
                    except (asyncio.TimeoutError, Exception):
                        quote = None
                    if not quote:
                        quote = {'last': data['Close'].iloc[-1]}

                    # Minimum price filter — penny stocks have spreads
                    # wider than their ATR, making stops meaningless.
                    # HQGE ($0.019) cost -$4,681 across 49 repeat trades.
                    current_quote_price = float(quote.get('last', data['Close'].iloc[-1]))
                    if current_quote_price < 5.0:
                        self._audit("price_filter", symbol, "skip",
                                    "below_min_price", price=round(current_quote_price, 4))
                        continue

                    # v-loop-yield-2026-04-30: technical_analyzer is async-
                    # named but runs synchronously (20+ indicators × pandas
                    # rolling on 200 rows = ~100-500ms per symbol). Wrap so
                    # the event loop can serve HTTP/WS during the compute.
                    # We can't call its async fn directly from a thread, so
                    # build a thin sync shim that drives the coroutine to
                    # completion using asyncio.run on a private loop.
                    def _ta_sync():
                        import asyncio as _aio
                        try:
                            return _aio.run(
                                self.technical_analyzer.analyze_with_commentary(data, symbol)
                            )
                        except RuntimeError:
                            # Already-running loop edge case — fall back
                            loop2 = _aio.new_event_loop()
                            try:
                                return loop2.run_until_complete(
                                    self.technical_analyzer.analyze_with_commentary(data, symbol)
                                )
                            finally:
                                loop2.close()
                    try:
                        indicators = await asyncio.wait_for(
                            _loop.run_in_executor(None, _ta_sync),
                            timeout=6.0,
                        )
                    except (asyncio.TimeoutError, Exception) as exc:
                        logger.debug(f"technical_analyzer error for {symbol}: {exc}")
                        indicators = {}

                    await asyncio.sleep(0)  # yield before ML predict
                    
                    # Create market data object with proper type conversion
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        open=float(data['Open'].iloc[-1]),
                        high=float(data['High'].iloc[-1]),
                        low=float(data['Low'].iloc[-1]),
                        close=float(quote.get('last', data['Close'].iloc[-1])),
                        volume=int(quote.get('volume', data['Volume'].iloc[-1])),
                        timeframe='5min',
                        indicators=indicators
                    )

                    # v-symbol-intel-2026-06-10: publish this symbol's
                    # composite read to the intelligence hub (UI panel +
                    # Phase 1 ledger). Pure observation; never raises.
                    if self._symbol_intel_hub is not None and indicators:
                        try:
                            from core.symbol_intel import build_symbol_view
                            self._symbol_intel_hub.update(build_symbol_view(
                                symbol, market_data.close, indicators,
                                tape_er=getattr(self._regime_allocator,
                                                '_cached_er', None),
                            ))
                        except Exception as _si_exc:
                            logger.debug("symbol_intel publish error: %s",
                                         _si_exc)

                    # Validate market data
                    if np.isnan(market_data.close) or market_data.close <= 0:
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.WARNING,
                            symbol=symbol,
                            title=f"⚠️ Invalid Price Data: {symbol}",
                            message=f"Price data is invalid (close=${market_data.close}), skipping analysis",
                            importance=6
                        ))
                        continue
                    
                    # v-loop-yield-2026-04-30: ML predict is sync sklearn —
                    # 200-800ms per call × 20 symbols = up to 16s of blocked
                    # event loop per cycle. Wrap in executor so HTTP and
                    # WebSocket handlers can interleave.
                    ml_signal, ml_explanation = (0, {})
                    if Config().ML_PREDICTION_ENABLED:
                        try:
                            ml_signal, ml_explanation = await asyncio.wait_for(
                                _loop.run_in_executor(
                                    None,
                                    self.ml_predictor.predict_with_commentary,
                                    indicators, symbol, data,
                                ),
                                timeout=4.0,
                            )
                        except (asyncio.TimeoutError, Exception) as exc:
                            logger.debug(f"ml_predict skipped for {symbol}: {exc}")
                            ml_signal, ml_explanation = (0, {})

                    await asyncio.sleep(0)  # yield before strategy iteration

                    # Check each strategy
                    for strategy in self.strategies:
                        signal = await strategy.generate_signal_with_commentary(market_data)
                        await asyncio.sleep(0)  # yield between strategies
                        if signal:
                            # Process the signal with full explanation. Pass
                            # market_data through so the classifier shadow
                            # hook can build SymbolFeatures from indicators
                            # (signal itself doesn't carry the MarketData).
                            await self._process_signal_with_commentary(
                                signal, ml_signal, ml_explanation,
                                market_data=market_data,
                            )
                
            except Exception as e:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"⚠️ Error Analyzing {symbol}",
                    message=f"Encountered error: {str(e)}",
                    importance=5
                ))
                logger.error(f"Error analyzing {symbol}: {e}", exc_info=True)
    
    def _shadow_meta_evaluate_sync(self, signal) -> dict | None:
        """Synchronous core — runs in a thread, returns result for async wrapper.

        Returns a dict with audit fields if evaluation succeeds, None otherwise.
        The async wrapper emits the _audit() call on the event loop so the
        DB write (which requires an async context) works correctly.
        """
        if self.meta_inference is None:
            return None
        strategy = signal.reasoning.get("strategy", "")
        if not strategy.startswith(self.meta_inference.primary_strategy):
            return None
        try:
            df = self.data_provider.get_market_data(
                signal.symbol, frequency_type="minute", frequency=5,
            )
            if df is None or df.empty or len(df) < 50:
                return None
            feat_df = self.ml_predictor.model.feature_extractor.batch_extract(df)
            if feat_df.empty:
                return None
            side = 1 if signal.signal_type == SignalType.BUY else -1
            features = feat_df.iloc[-1].to_dict()
            result = self.meta_inference.predict(features, side=side)
            return {
                "symbol": signal.symbol,
                "strategy": strategy,
                "side": side,
                "proba": round(result.win_probability, 4),
                "thr_065": result.would_trade_at_065,
                "thr_070": result.would_trade_at_070,
                "thr_075": result.would_trade_at_075,
            }
        except Exception as exc:
            logger.debug("meta_shadow_error symbol=%s err=%s", signal.symbol, exc)
            return None

    async def _shadow_meta_evaluate(self, signal) -> None:
        """Async wrapper ��� offloads sync body to a thread, emits audit on event loop."""
        if self.meta_inference is None:
            return
        try:
            result = await asyncio.to_thread(self._shadow_meta_evaluate_sync, signal)
            if result is not None:
                side_label = "long" if result["side"] == 1 else "short"
                # v-meta-proba-on-signal-2026-05-05: stash proba on the
                # signal so downstream gates (early_session bypass etc.)
                # can read ML conviction without re-running inference.
                try:
                    if signal.reasoning is None:
                        signal.reasoning = {}
                    signal.reasoning["meta_proba"] = float(result["proba"])
                except Exception:
                    pass
                self._audit(
                    "meta_shadow", result["symbol"], "evaluated",
                    f"{result['strategy']}_{side_label}",
                    proba=result["proba"],
                    thr_065=result["thr_065"],
                    thr_070=result["thr_070"],
                    thr_075=result["thr_075"],
                )
        except Exception as exc:
            logger.debug("meta_shadow_dispatch_error err=%s", exc)

    async def _process_signal_with_commentary(self, signal, ml_signal,
                                                ml_explanation,
                                                market_data=None):
        """Process trading signal with detailed explanation.

        ``market_data`` is the MarketData that the strategy used to
        generate ``signal``. Threaded through so the classifier shadow
        hook can read indicators (RSI, SMA, ATR, volume_ratio) — the
        TradingSignal dataclass itself doesn't carry them.
        """
        self._audit("signal_router", signal.symbol, "received",
                    signal.reasoning.get("strategy", "unknown"),
                    signal_type=signal.signal_type.value,
                    confidence=round(float(getattr(signal, "confidence", 0)), 3),
                    strength=round(float(getattr(signal, "strength", 0)), 3),
                    ml_signal=ml_signal)

        # Shadow meta-model evaluation — logs only, no decision impact.
        # Dispatched via asyncio.to_thread so the sync sklearn/xgb call
        # does not block the event loop.
        await self._shadow_meta_evaluate(signal)

        # v-classifier-runtime-2026-05-13: side-classifier shadow hook.
        # When SIDE_CLASSIFIER_SHADOW_MODE is True, we ask the rule+
        # trained composer for a verdict on this signal's symbol and
        # log it as an audit line. The verdict does NOT affect
        # gating — that comes later, per rollout flavor B (news_strategy
        # first, then mean_reversion). Today this is pure data
        # collection: after N sessions of shadow data, we'll have
        # empirical numbers on whether the classifier blocks the bad
        # trades it claims to.
        if self._classifier_runtime is not None and market_data is not None:
            try:
                decision = self._classifier_runtime.shadow_evaluate(market_data)
                if decision is not None:
                    self._audit(
                        "side_classifier", signal.symbol, "shadow",
                        decision.primary_reason,
                        long_score=round(decision.long_score, 3),
                        short_score=round(decision.short_score, 3),
                        allows_long=decision.allows_long(),
                        allows_short=decision.allows_short(),
                        signal_side=signal.signal_type.value,
                    )
            except Exception as _cl_exc:
                # Pure logging path — never let it crash signal flow.
                logger.debug("side_classifier shadow evaluate error: %s", _cl_exc)

        # v-regime-allocator-shadow-2026-06-10 (roadmap P1): log what
        # the regime allocator would have decided for this signal.
        # Audit line + NDJSON ledger; affects nothing. After 2 weeks,
        # research compares would_allow=False signals' outcomes to
        # the walk-forward prediction before any gating wire-up.
        _alloc = None  # also read by the conviction-sizer hook below
        if self._regime_allocator is not None:
            try:
                _ra_strategy = signal.reasoning.get("strategy", "unknown")
                _alloc = await self._regime_allocator.evaluate(
                    strategy=_ra_strategy,
                    symbol=signal.symbol,
                    signal_side=signal.signal_type.value,
                )
                if _alloc is not None:
                    self._audit(
                        "regime_allocator", signal.symbol, "shadow",
                        _alloc.tape,
                        er=None if _alloc.er is None
                           else round(_alloc.er, 3),
                        strategy=_ra_strategy,
                        would_allow=_alloc.allows(_ra_strategy),
                    )
            except Exception as _ra_exc:
                logger.debug("regime_allocator shadow error: %s", _ra_exc)

        # v-regime-gate-live-meanrev-2026-06-16: the allocator's FIRST
        # live veto. Mean-reversion is a chop strategy — in the
        # walk-forward it bled -104.7% in a trending window it never
        # should have traded. Gating it to choppy tape removed that
        # disaster (-> +1.4%) and lifted pooled PF 1.06 -> 1.12
        # (research/meanrev_gated_report_2026-06-15.json). Scope is the
        # mean-rev family ONLY; breakout/news stay in shadow. Fail-open
        # is inherited from RegimeAllocation.allows() — unknown ER
        # returns True, so a Schwab data hiccup never blocks trading.
        _ra_strategy = signal.reasoning.get("strategy", "unknown") \
            if signal.reasoning else "unknown"
        if (Config().REGIME_GATE_LIVE_MEANREV
                and _alloc is not None
                and _ra_strategy in ("mean_reversion", "oversold_v2")
                and not _alloc.allows(_ra_strategy)):
            self._audit(
                "regime_gate", signal.symbol, "blocked",
                "regime_gate_meanrev",
                strategy=_ra_strategy, tape=_alloc.tape,
                er=None if _alloc.er is None else round(_alloc.er, 3),
                threshold=_alloc.threshold,
            )
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=signal.symbol,
                title=f"⛔ Regime Gate — {signal.symbol} mean-rev blocked",
                message=(
                    f"Tape is trending (SPY efficiency "
                    f"{_alloc.er:.2f} ≥ {_alloc.threshold:.2f}). "
                    f"Mean-reversion sits out trending tapes — this is "
                    f"the gate that removed the −104% walk-forward "
                    f"window. Waiting for chop."
                    if _alloc.er is not None else
                    f"Mean-reversion blocked by regime gate ({_alloc.tape})."
                ),
                importance=7,
            ))
            # v-feature-snapshot-2026-09-09: emit snapshot on regime veto
            self._emit_veto_snapshot(
                signal=signal,
                strategy_id=_ra_strategy,
                reason="regime_gate_meanrev",
                gate_name="regime_gate",
                extra={"tape": _alloc.tape, "er": _alloc.er, "threshold": _alloc.threshold},
            )
            return  # abort the signal — no order placed

        # v-conviction-floor-meanrev-2026-06-17: skip mean-rev signals
        # below the meta-model conviction floor. The meta<0.60 bucket
        # ran PF 0.44 across 89 trades; 2026-06-16 live, an all-medium-
        # conviction basket netted -$268. Scope = mean-rev family only.
        # Fail-open: a signal whose meta_proba is None is never blocked
        # (the meta model is shadow — can't floor what wasn't measured).
        # Coerce meta_proba defensively — the only current writer stores
        # a float, but a future path writing a non-numeric must fail
        # OPEN (allow the trade), never raise into the signal router.
        _cf_meta = signal.reasoning.get("meta_proba") if signal.reasoning else None
        try:
            _cf_meta = float(_cf_meta) if _cf_meta is not None else None
        except (TypeError, ValueError):
            _cf_meta = None
        if (Config().ENABLE_CONVICTION_FLOOR_MEANREV
                and _ra_strategy in ("mean_reversion", "oversold_v2")
                and _cf_meta is not None
                and _cf_meta < Config().CONVICTION_FLOOR_META):
            _cf_floor = Config().CONVICTION_FLOOR_META
            self._audit(
                "conviction_floor", signal.symbol, "blocked",
                "below_meta_floor",
                strategy=_ra_strategy, meta_proba=round(float(_cf_meta), 3),
                floor=_cf_floor,
            )
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=signal.symbol,
                title=f"⛔ Conviction Floor — {signal.symbol} skipped",
                message=(
                    f"Meta-model conviction {float(_cf_meta):.2f} below "
                    f"floor {_cf_floor:.2f}. Sub-floor trades historically "
                    f"lose (meta<0.60: PF 0.44 over 89 trades; "
                    f"0.60–0.65: -$247 over 9). Skipping "
                    f"low-conviction mean-rev entry."
                ),
                importance=6,
            ))
            return  # abort the signal — no order placed

        # v-conviction-sizer-shadow-2026-06-11: log the would-be size
        # multiplier for this signal from the confluence of independent
        # confirmations. Pure observation — actual sizing unchanged.
        if self._conviction_sizer is not None:
            try:
                _ind = (getattr(market_data, "indicators", None) or {}) \
                    if market_data is not None else {}
                _dir_score = None
                try:
                    from core.direction_reader import read_direction
                    if market_data is not None and _ind:
                        _dir_score = read_direction(
                            market_data.close, _ind).direction
                except Exception:
                    pass
                _conv = self._conviction_sizer.evaluate(
                    strategy=signal.reasoning.get("strategy", "unknown"),
                    symbol=signal.symbol,
                    meta_proba=signal.reasoning.get("meta_proba"),
                    ml_agrees=(
                        None if ml_signal is None
                        else ml_signal == signal.signal_type.value
                    ),
                    allocator_allows=(
                        _alloc.allows(signal.reasoning.get("strategy", ""))
                        if _alloc is not None else None
                    ),
                    direction=_dir_score,
                    volume_ratio=_ind.get("volume_ratio"),
                )
                if _conv is not None:
                    self._audit(
                        "conviction_sizer", signal.symbol, "shadow",
                        "confluence",
                        score=_conv.score,
                        would_be_mult=_conv.multiplier,
                        meta=signal.reasoning.get("meta_proba"),
                    )
            except Exception as _cs_exc:
                logger.debug("conviction_sizer shadow error: %s", _cs_exc)

        # v-health-gate-2026-05-08: fail-closed guard on new live entries.
        # Two conditions block placement:
        #   (a) stream_dead — Schwab quote stream is unhealthy (no recent
        #       ticks). Trading on stale REST quotes was the proximate
        #       cause of the May 5–7 stale-data entries (INTC/PINS bursts
        #       at 1–5 minute lagged prices). The watchdog (T2) restarts
        #       the stream automatically; this gate refuses to open new
        #       positions until ticks are flowing again.
        #   (b) management_unavailable — there is at least one bot-opened
        #       live position from the last 5 minutes that DID NOT receive
        #       managed_by_bot=True. T1 fixed the known cause of this
        #       symptom; the gate is paranoia in case another path slips
        #       through. If the engine cannot manage what it just opened,
        #       the engine should NOT open more.
        # Both checks apply only in LIVE mode; sim mode runs unrestricted
        # so the operator can keep exercising the strategy logic.
        if self.mode == TradingMode.LIVE:
            stream = getattr(self, "_schwab_quote_stream", None)
            stream_unhealthy = (
                stream is not None
                and hasattr(stream, "is_healthy")
                and not stream.is_healthy()
            )
            if stream_unhealthy:
                last_tick = getattr(stream, "_last_tick_at", None)
                age_s = None
                if last_tick is not None:
                    try:
                        from datetime import timezone as _tz
                        age_s = (
                            datetime.now(_tz.utc) - last_tick
                        ).total_seconds()
                    except Exception:
                        age_s = None
                self._audit(
                    "health_gate", signal.symbol, "skip", "stream_dead",
                    last_tick_age_s=int(age_s) if age_s is not None else None,
                )
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.RISK_ASSESSMENT,
                    symbol=signal.symbol,
                    title=f"🛑 Stream Unhealthy — Refusing Entry",
                    message=(
                        f"Schwab quote stream is not healthy "
                        f"(last tick {age_s if age_s is not None else 'never'}s ago). "
                        "Refusing to place new live orders on stale REST data. "
                        "Watchdog will restart the stream automatically."
                    ),
                    importance=8,
                ))
                return

            # Management-unavailable guard: scan for any bot-opened
            # position in the last 5 minutes whose managed_by_bot flag is
            # False. T1's fixes should make this impossible, but a stuck
            # flag here is a reliable canary that something has regressed.
            try:
                _now = datetime.now()
                _recent_unmanaged = []
                for _sym, _pos in list(self.positions.items()):
                    _et = getattr(_pos, "entry_time", None)
                    if _et is None:
                        continue
                    if isinstance(_et, str):
                        try:
                            _et = datetime.fromisoformat(_et)
                        except Exception:
                            continue
                    if (_now - _et).total_seconds() > 300:
                        continue
                    _is_external = bool(getattr(_pos, "is_external", False))
                    _is_managed = bool(getattr(_pos, "managed_by_bot", False))
                    # Only flag positions that should have been bot-managed:
                    # NOT marked external, AND missing managed_by_bot=True.
                    if not _is_external and not _is_managed:
                        _recent_unmanaged.append(_sym)
                if _recent_unmanaged:
                    self._audit(
                        "health_gate", signal.symbol, "skip",
                        "management_unavailable",
                        unmanaged=_recent_unmanaged,
                    )
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=signal.symbol,
                        title=f"🛑 Recent Bot Trade Unmanaged — Refusing Entry",
                        message=(
                            f"Recently opened bot trades show "
                            f"managed_by_bot=False: {_recent_unmanaged}. "
                            "Refusing to open more positions until the "
                            "fill→management path is confirmed healthy."
                        ),
                        importance=9,
                    ))
                    return
            except Exception as _exc:
                logger.debug("health_gate management check error: %s", _exc)

        # Market-hours gate — applies to BOTH sim and live so simulation
        # accurately previews live behaviour (sim used to enter trades
        # 24/7 which gave misleading fills on afterhours noise).
        is_regular, session = self.is_market_hours()
        if not is_regular:
            if session == "premarket" and self.allow_premarket:
                pass
            elif session == "afterhours" and self.allow_afterhours:
                pass
            else:
                self._audit("market_hours", signal.symbol, "skip",
                            f"session_{session}",
                            allow_premarket=self.allow_premarket,
                            allow_afterhours=self.allow_afterhours)
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"🌙 Market {session.title()} — Skipping Entry",
                    message=(f"Signal received during {session} session. "
                             "New trades only during regular hours (09:30-16:00 ET). "
                             "Toggle trading.extended_hours.allow_* to override."),
                    importance=5,
                ))
                return

        # v-early-session-soft-2026-04-30: soft veto on the first 15 minutes
        # after open. Indicators computed on <15 bars of post-open data are
        # unreliable (ATR is microscopic, volume ratios skewed by opening
        # auction). Block entries only — analysis upstream still runs and
        # logs every strategy_decision so the user can watch the bot think.
        #
        # v-early-session-narrow-2026-05-05: narrowed 15min → 5min, AND
        # exempt signals with strong meta-shadow conviction (proba>=0.65).
        #
        # v-early-session-hard-2026-05-23: REMOVED meta_proba bypass.
        # Weekly report 2026-05-22 (docs/weekly_report_2026-05-22.md
        # Section 4) showed first_5min slot was -$247 on FIG mean_reversion
        # (3-min stop-out). The bypass case (ORCL 09:40) cited when adding
        # the escape was actually outside the 9:30-9:34 window, so the
        # bypass never had real positive evidence — only a defensive
        # "what if" that data now contradicts. Hard block until 09:35 ET.
        # We still record meta_proba in the audit line so we can later
        # measure whether hard-blocking left money on the table.
        import pytz as _pytz
        _et = datetime.now(_pytz.timezone("America/New_York"))
        meta_proba = float((signal.reasoning or {}).get("meta_proba") or 0.0)
        in_first_5 = (_et.hour == 9 and 30 <= _et.minute < 35)
        if in_first_5:
            self._audit("early_session", signal.symbol, "skip", "first_5min_hard_block",
                        time=_et.strftime("%H:%M"),
                        strategy=signal.reasoning.get("strategy", "unknown"),
                        meta_proba=round(meta_proba, 4))
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=signal.symbol,
                title=f"⏳ Opening Auction — Holding Off",
                message=(f"Signal seen at {_et.strftime('%H:%M')} ET. "
                         "Hard block until 09:35 ET (no bypass). "
                         "Backed by weekly report: first_5min slot lost "
                         "$247 on FIG (3-min stop-out)."),
                importance=5,
            ))
            return

        # v-late-entry-cutoff-2026-05-23: no new entries after the
        # configured cutoff (default 15:30 ET). Bot is in manual_close_only
        # mode globally (line ~280), meaning all exits are delegated to
        # Schwab OCO brackets. Without an EOD flatten, a late-day entry
        # can become an unintended overnight hold (see ASTS 2026-05-21:
        # entered 15:30, OCO target hit next morning, 17-hour hold,
        # +$303 only because of a favorable gap). The trade was profitable
        # this time but the exposure was asymmetric overnight gap risk.
        # Until EOD-flatten exists, refuse new entries close to the bell.
        cutoff_hour = Config().LATE_ENTRY_CUTOFF_HOUR
        cutoff_minute = Config().LATE_ENTRY_CUTOFF_MINUTE
        past_cutoff = (
            _et.hour > cutoff_hour
            or (_et.hour == cutoff_hour and _et.minute >= cutoff_minute)
        )
        before_close = _et.hour < 16  # 16:00 ET = market close
        if past_cutoff and before_close:
            self._audit("late_entry", signal.symbol, "skip", "after_entry_cutoff",
                        time=_et.strftime("%H:%M"),
                        strategy=signal.reasoning.get("strategy", "unknown"),
                        cutoff=f"{cutoff_hour:02d}:{cutoff_minute:02d}")
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=signal.symbol,
                title=f"🌅 Late-Day Entry — Skipping",
                message=(f"Signal at {_et.strftime('%H:%M')} ET is past "
                         f"the {cutoff_hour:02d}:{cutoff_minute:02d} entry "
                         f"cutoff. Bot delegates exits to Schwab OCO and "
                         f"has no EOD flatten, so late entries risk "
                         f"becoming overnight holds (ASTS 5/21 incident). "
                         f"Re-enable after EOD-flatten is implemented."),
                importance=6,
            ))
            return

        # CRITICAL: Check real Schwab positions FIRST before internal tracking
        if self.mode == TradingMode.LIVE and self.schwab_client:
            schwab_positions = await self.get_schwab_positions()
            existing_position = next((pos for pos in schwab_positions if pos['symbol'] == signal.symbol), None)
            
            if existing_position:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"🚫 Existing Schwab Position Found",
                    message=f"Found existing position: {existing_position['quantity']} shares at ${existing_position['average_price']:.2f}\n"
                        f"Current P&L: ${existing_position['total_pnl']:.2f} ({existing_position['pnl_percent']:.1f}%)",
                    data=existing_position,
                    importance=9
                ))
                
                # Add to bot tracking if not already tracked
                if signal.symbol not in self.positions:
                    position = Position(
                        symbol=signal.symbol,
                        entry_price=existing_position['average_price'],
                        current_price=existing_position['current_price'],
                        quantity=existing_position['quantity'],
                        side='long' if existing_position['quantity'] > 0 else 'short',
                        stop_loss=existing_position['average_price'] * (1 - Config().DEFAULT_STOP_LOSS_PCT),
                        take_profit=existing_position['average_price'] * (1 + Config().DEFAULT_TAKE_PROFIT_PCT),
                        entry_time=datetime.now(),
                        unrealized_pnl=existing_position['total_pnl'],
                        reasoning={'source': 'existing_schwab_position'},
                        mode="live",
                        managed_by_bot=False,  # pre-existing Schwab position
                    )
                    self.positions[signal.symbol] = position
                    
                    # Initialize exit tracking
                    if hasattr(self, 'exit_manager'):
                        self.exit_manager.initialize_position_tracking(
                            signal.symbol,
                            position.entry_price,
                            position.stop_loss,
                            position.take_profit
                        )

                self._audit("signal_router", signal.symbol, "skip",
                            "existing_schwab_position",
                            qty=existing_position["quantity"],
                            avg=existing_position["average_price"])
                return

        # Check internal tracking (for simulation mode or as fallback)
        if signal.symbol in self.positions or signal.symbol in self.simulated_positions:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=signal.symbol,
                title=f"📍 Already Tracking Position",
                message=f"Already tracking position in {signal.symbol}, skipping this signal.",
                importance=3
            ))
            self._audit("signal_router", signal.symbol, "skip", "already_tracking")
            return

        # v-anti-pyramiding-2026-05-13: defense-in-depth against the
        # FCEL × 3 pattern observed on 2026-05-12 (35 and 21 minutes
        # between three buys, far past any verify-fill window).
        # Two checks:
        #   (a) Pending order in flight for this symbol — order_id
        #       exists in self.pending_orders but the Position hasn't
        #       materialized yet.
        #   (b) Recent entry-attempt cooldown — within 30 minutes of
        #       the last accepted entry for this symbol, refuse a new
        #       one regardless of strategy or position-state.
        # Audit grep: `engine_decision .* action=skip .* reason=anti_pyramid`
        for _pending in self.pending_orders.values():
            if _pending.get('symbol') == signal.symbol:
                self._audit(
                    "signal_router", signal.symbol, "skip",
                    "anti_pyramid_pending_order_for_symbol",
                    order_status=_pending.get('status'),
                )
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"🛑 Anti-Pyramiding (Order In-Flight)",
                    message=(
                        f"Order already in flight for {signal.symbol}; "
                        f"refusing duplicate entry."
                    ),
                    importance=5,
                ))
                return

        if not hasattr(self, '_recent_entry_attempts'):
            self._recent_entry_attempts = {}
        _last_attempt = self._recent_entry_attempts.get(signal.symbol)
        if _last_attempt is not None:
            _age_min = (datetime.now() - _last_attempt).total_seconds() / 60
            # v-anti-pyramid-widen-2026-05-18: 30→240 min. On 5/15 RKLB
            # doubled-up at 63 min gap (loaded $17.7K = 59% of equity in
            # one name). Same-direction same-day adds during the same
            # news cycle compound the same bet; 4-hour window keeps the
            # gap > a typical news half-life.
            if _age_min < 240:
                self._audit(
                    "signal_router", signal.symbol, "skip",
                    "anti_pyramid_recent_attempt",
                    age_min=round(_age_min, 1),
                )
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"🛑 Anti-Pyramiding (Recent Entry)",
                    message=(
                        f"Entered {signal.symbol} {_age_min:.0f} min ago; "
                        f"4-hour anti-pyramid window not elapsed."
                    ),
                    importance=5,
                ))
                return
        # Record this attempt — set BEFORE the order goes out so two
        # parallel signals don't race past the cooldown.
        self._recent_entry_attempts[signal.symbol] = datetime.now()

        # Per-symbol loss cooldown: if we just stopped out on this symbol,
        # don't re-enter for 60 minutes. LCID re-entered 24 min after a
        # -$81 stop-loss and lost again. The setup rarely improves that fast.
        if not hasattr(self, '_symbol_loss_cooldown'):
            self._symbol_loss_cooldown = {}
        cooldown_end = self._symbol_loss_cooldown.get(signal.symbol)
        if cooldown_end and datetime.now() < cooldown_end:
            remaining = (cooldown_end - datetime.now()).total_seconds() / 60
            self._audit("loss_cooldown", signal.symbol, "skip",
                        "recent_loss", cooldown_remaining_min=round(remaining, 1))
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=signal.symbol,
                title=f"🧊 Loss Cooldown — {signal.symbol}",
                message=(
                    f"Skipping {signal.symbol} — lost on it recently. "
                    f"Cooldown expires in {remaining:.0f} min."
                ),
                importance=6,
            ))
            return

        # v-short-exit-fix-2026-04-21: generic re-entry cooldown after
        # ANY exit (not just stop-loss). Belt-and-suspenders after today's
        # churn loop where a long-only exit manager fired "trailing_stop"
        # on every short entry, re-entered within 30-60s, and burned
        # 37 round-trips in 90 min. Capped at 5 min so genuine re-setups
        # aren't locked out — this guard only fires within the fast-churn
        # window a real re-entry shouldn't be in anyway.
        if not hasattr(self, '_symbol_reentry_cooldown'):
            self._symbol_reentry_cooldown = {}
        reentry_end = self._symbol_reentry_cooldown.get(signal.symbol)
        if reentry_end and datetime.now() < reentry_end:
            remaining = (reentry_end - datetime.now()).total_seconds() / 60
            self._audit("reentry_cooldown", signal.symbol, "skip",
                        "recent_exit", cooldown_remaining_min=round(remaining, 2))
            return

        # v-shortable-check-2026-04-29: pre-trade shortable gate.
        # SELL signals only — verify the broker will actually accept the
        # short before we submit. Avoids HTB rejection at order placement
        # time (Schwab returns generic "order rejected" without indicating
        # shortable status, so the gate must be upstream).
        # Skips this check when this is a closing SELL on an existing LONG
        # position (signal.symbol already in self.positions/simulated_positions).
        if signal.signal_type == SignalType.SELL:
            already_long = (
                signal.symbol in self.positions and
                getattr(self.positions.get(signal.symbol), 'side', None) == 'long'
            ) or (
                signal.symbol in self.simulated_positions and
                getattr(self.simulated_positions.get(signal.symbol), 'side', None) == 'long'
            )
            if not already_long:
                # Lazy-init the checker once per engine — tiny memory cost.
                if not hasattr(self, '_shortable_checker'):
                    from analysis.shortable_check import ShortableChecker
                    self._shortable_checker = ShortableChecker(
                        cache_ttl_sec=6 * 3600,
                        assume_shortable_on_error=False,  # fail-closed: if
                        # we can't confirm shortable, don't risk an HTB
                        # rejection mid-order. User can flip to True for
                        # more aggressive sim runs.
                    )
                try:
                    info = await self._shortable_checker.is_shortable(signal.symbol)
                except Exception as _exc:
                    info = None
                if info is not None and not info.shortable:
                    self._audit(
                        "shortable_check", signal.symbol, "skip",
                        "not_shortable",
                        side="short",
                        source=info.source,
                        easy_to_borrow=info.easy_to_borrow,
                    )
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=signal.symbol,
                        title=f"⛔ Short Skipped — Not Shortable",
                        message=(
                            f"{signal.symbol} is not shortable per Alpaca "
                            f"({info.source}). Avoiding HTB rejection at "
                            f"order placement."
                        ),
                        importance=7,
                    ))
                    return
                # Note ETB status for audit (still proceeds; HTB borrow
                # fees are usually small but worth tracking).
                if info is not None and not info.easy_to_borrow:
                    self._audit(
                        "shortable_check", signal.symbol, "warn",
                        "hard_to_borrow",
                        side="short", source=info.source,
                    )

        # Sector correlation guard — limit to 1 open position per
        # correlated group to prevent cluster stop-outs (e.g., MARA +
        # RIOT + COIN all dropping together when BTC falls).
        # v-correlation-groups-expanded-2026-05-08: 2026-05-06 produced
        # a 6-entry burst across SMCI, ARM, INTC, NVDA, AVGO + ORCL in
        # 2 minutes. Old `semiconductors` group only caught NVDA, AMD,
        # INTC, MU, AVGO, QCOM — missing ARM (chip-design adjacent),
        # SMCI (server hardware, semi-cycle correlated), MRVL, TSM, LRCX
        # (semi-equipment). Expanded to a single broader `semis_and_chip_adjacent`
        # set so the cluster guard fires on the actual correlation, not
        # just a narrow pure-foundry list. New groups for fintech and
        # AI-software are added because COIN+SOFI+HOOD and
        # PLTR+SNOW+AI-related names trade together too.
        _CORRELATED_GROUPS = {
            "crypto_miners": {"MARA", "RIOT", "COIN", "CLSK", "BTBT", "CIFR", "WULF", "HUT"},
            "ev_makers": {"TSLA", "RIVN", "NIO", "LCID", "XPEV"},
            "china_tech": {"BABA", "JD", "PDD", "BIDU", "NIO", "XPEV"},
            "meme_retail": {"AMC", "GME", "BBBY", "FUBO"},
            "semis_and_chip_adjacent": {
                "NVDA", "AMD", "INTC", "MU", "AVGO", "QCOM",
                "ARM", "SMCI", "MRVL", "TSM", "LRCX", "AMAT", "KLAC",
                "ASML", "ON", "MCHP", "NXPI",
            },
            "fintech_payments": {"COIN", "SOFI", "HOOD", "PYPL", "SQ", "AFRM"},
            "ai_software": {"PLTR", "SNOW", "AI", "PATH", "ASAN"},
            "social_media": {"META", "PINS", "SNAP", "GOOGL", "GOOG"},
            "streaming_media": {"NFLX", "DIS", "ROKU", "SPOT"},
        }
        # v-correlation-mode-isolated-2026-05-05: only consider positions
        # in the SAME trading mode as the incoming signal. Previously this
        # unioned live + simulated, so a leftover sim INTC from yesterday
        # blocked today's live QCOM/AMD/MU entries (and vice-versa). The
        # correlation risk is real ONLY within a real-money portfolio or a
        # paper portfolio — they aren't sharing capital.
        if self.mode == TradingMode.LIVE:
            active_positions = set(self.positions.keys())
            active_pos_objs = self.positions.values()
        else:
            active_positions = set(self.simulated_positions.keys())
            active_pos_objs = self.simulated_positions.values()

        # v-max-positions-gate-2026-05-06: hard cap on concurrent
        # bot-managed trades. Counts only positions where the bot is
        # actively managing exits (managed_by_bot=True) — pre-existing
        # Schwab holdings tagged hands-off do NOT consume a slot. This
        # config was defined since day one but never enforced; live
        # account had grown to 9 bot-opened positions before this gate.
        try:
            _max_positions = int(Config().MAX_POSITIONS or 10)
        except Exception:
            _max_positions = 10
        # v-max-positions-pending-2026-05-18: also count pending entry
        # orders. On 5/18 the bot took 6 mean-rev oversold_bounce trades
        # in 8 min vs a cap of 5 because new orders hadn't reconciled to
        # managed_by_bot=True yet — bot_managed_count was 0 throughout
        # the burst. Counting distinct symbols across (managed positions)
        # ∪ (pending orders) closes the burst-window bypass.
        _managed_syms = {
            getattr(p, 'symbol', None) for p in active_pos_objs
            if bool(getattr(p, 'managed_by_bot', False))
            and int(getattr(p, 'quantity', 0) or 0) > 0
        }
        _pending_syms = {
            info.get('symbol') for info in self.pending_orders.values()
            if isinstance(info, dict) and info.get('symbol')
            and info.get('status') == 'PENDING'
        }
        bot_managed_count = len(_managed_syms | _pending_syms)
        already_in = signal.symbol in active_positions
        if bot_managed_count >= _max_positions and not already_in:
            self._audit("max_positions_gate", signal.symbol, "skip",
                        "limit_reached",
                        bot_managed=bot_managed_count,
                        cap=_max_positions)
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=signal.symbol,
                title=f"🛑 Max Concurrent Trades — Slot Cap Hit",
                message=(
                    f"Already managing {bot_managed_count} bot trades "
                    f"(cap {_max_positions}). Skipping {signal.symbol} "
                    "until a slot frees. Adjust trading.max_positions "
                    "to change the cap."
                ),
                importance=7,
            ))
            return
        for group_name, members in _CORRELATED_GROUPS.items():
            if signal.symbol in members:
                overlap = active_positions & members
                if overlap:
                    self._audit("correlation_guard", signal.symbol, "skip",
                                f"correlated_with_{group_name}",
                                existing=list(overlap))
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=signal.symbol,
                        title=f"🔗 Correlated Position Blocked",
                        message=(
                            f"Already holding {', '.join(overlap)} in the "
                            f"{group_name.replace('_', ' ')} group. "
                            f"Skipping {signal.symbol} to avoid cluster risk."
                        ),
                        importance=7,
                    ))
                    return

        # EARLY BUYING POWER CHECK for LIVE mode
        if self.mode == TradingMode.LIVE:
            # First check if we have ANY buying power before doing calculations
            min_buying_power = Config().MIN_BUYING_POWER or 100  # Default $100
            has_power, available_bp = await self._check_buying_power(min_buying_power)
            logger.info(f"Has buying power or not: {has_power} {available_bp}")
            if not has_power or (available_bp is not None and available_bp < min_buying_power):
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"❌ No Buying Power",
                    message=f"Cannot trade - buying power is ${available_bp or 0:.2f} (need at least ${min_buying_power})",
                    importance=9
                ))
                self._audit("risk", signal.symbol, "skip", "no_buying_power",
                            available_bp=round(float(available_bp or 0), 2),
                            min_required=min_buying_power)
                return
            
            # Update risk manager with current buying power
            self.risk_manager.buying_power = available_bp
            # Check for existing orders
            if self.mode == TradingMode.LIVE:
                existing_orders = await self._check_existing_orders(signal.symbol)
                if existing_orders['has_orders']:
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.WARNING,
                        symbol=signal.symbol,
                        title=f"⚠️ Existing Orders Found",
                        message=f"Found {existing_orders['count']} existing orders for {signal.symbol}",
                        importance=7
                    ))
                    
                    # Cancel existing orders if configured
                    if Config().AUTO_CANCEL_EXISTING_ORDERS:
                        self._audit("signal_router", signal.symbol, "cancel_and_retry",
                                    "existing_orders",
                                    existing_count=existing_orders["count"])
                        await self._cancel_existing_orders(signal.symbol)
                    else:
                        self._audit("signal_router", signal.symbol, "skip",
                                    "existing_orders",
                                    existing_count=existing_orders["count"])
                        return
        # Check with brain if we should take this trade
        should_trade, brain_reason = self.brain.should_take_trade(
            signal.symbol, 
            signal.reasoning.get('strategy', 'unknown'),
            signal.confidence
        )
        if not should_trade:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.PSYCHOLOGY,
                symbol=signal.symbol,
                title=f"🧠 Experience Says No",
                message=brain_reason,
                data={
                    'pattern_history': self.brain.get_pattern_confidence(signal.reasoning.get('strategy', 'unknown')),
                    'emotional_state': self.brain.emotional_state
                },
                importance=7
            ))
            self._audit("brain", signal.symbol, "skip", "experience_says_no",
                        strategy=signal.reasoning.get("strategy", "unknown"),
                        brain_reason=brain_reason[:60].replace(" ", "_"))
            return
	
	# Add more human-like pre-trade thoughts
        pre_trade_thoughts = [
            f"Alright, {signal.symbol} is setting up nicely. Let me check the level 2...",
            f"Interesting setup on {signal.symbol}. This reminds me of last week's trade.",
            f"{signal.symbol} looking ready. Volume confirms - let's do this.",
            f"Been watching {signal.symbol} all morning. Finally getting the entry I wanted."
        ]
        
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.PSYCHOLOGY,
            symbol=signal.symbol,
            title=f"💭 Pre-Trade Thoughts",
            message=np.random.choice(pre_trade_thoughts),
            importance=5
        ))

        # Evaluate the signal
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.SIGNAL_GENERATION,
            symbol=signal.symbol,
            title=f"🎯 Trading Signal Generated!",
            message=f"Strategy '{signal.reasoning.get('strategy', 'unknown')}' found a potential {signal.signal_type.name} opportunity",
            data={
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'risk_reward_ratio': (signal.take_profit - signal.entry_price) / (signal.entry_price - signal.stop_loss),
                'signal_strength': signal.strength,
                'ml_confirmation': ml_signal == 1
            },
            confidence=signal.confidence,
            importance=8
        ))



        # Check ML confirmation
        # Check ML confirmation (1 for BUY, -1 for SELL)
        expected_ml_signal = 1 if signal.signal_type == SignalType.BUY else -1
        if ml_signal != expected_ml_signal and Config().ML_PREDICTION_ENABLED:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=signal.symbol,
                title=f"🤖 ML Model Disagrees",
                message="The machine learning model doesn't confirm this signal. "
                       "I typically wait for ML confirmation to increase win probability.",
                data=ml_explanation,
                importance=7
            ))

            # Log the decision to skip
            log_decision(
                symbol=signal.symbol,
                decision='SKIP',
                reason='ML model disagrees with strategy signal',
                factors={'ml_signal': ml_signal, 'expected': expected_ml_signal},
                signals={'strategy': signal.strength, 'ml': ml_signal},
                indicators=ml_explanation,
                confidence=signal.confidence,
                mode=self.mode.value
            )

            # Veto policy: when ML disagrees with high confidence, skip the trade
            # in ALL modes (not just live). Previously sim mode ignored the veto,
            # which let strategies enter against trained-model signals (LCID 2026-04-14).
            ml_confidence = float(ml_explanation.get("confidence", 0.0)) if isinstance(ml_explanation, dict) else 0.0
            ml_veto_threshold = Config().ML_VETO_CONFIDENCE
            if ml_confidence >= ml_veto_threshold:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"🛑 ML Veto",
                    message=f"Skipping trade — ML confidence {ml_confidence:.0%} ≥ veto threshold "
                            f"{ml_veto_threshold:.0%}. The model strongly disagrees with the strategy.",
                    importance=8
                ))
                self._audit("ml", signal.symbol, "skip", "ml_veto_high_confidence",
                            ml_signal=ml_signal, ml_conf=round(ml_confidence, 3),
                            threshold=round(ml_veto_threshold, 3))
                # v-feature-snapshot-2026-09-09: emit snapshot on ML veto
                self._emit_veto_snapshot(
                    signal=signal,
                    strategy_id=signal.reasoning.get("strategy", "unknown") if signal.reasoning else "unknown",
                    reason="ml_veto_high_confidence",
                    gate_name="ml_veto",
                    extra={"ml_signal": ml_signal, "ml_conf": ml_confidence, "threshold": ml_veto_threshold},
                )
                return

            # v-ml-advisor-2026-04-20: below the veto threshold, ML is an
            # ADVISOR, not a gatekeeper. The prior behaviour was to skip the
            # trade on ANY disagreement regardless of ML confidence — which
            # silently blocked every mean-reversion short (ML is long-biased
            # by training distribution, will always predict BUY at high RSI)
            # and every other contrarian signal. Backtest PF 2.31 on mean-rev
            # long was measured with ML off; running with "any disagreement
            # blocks" produced zero trades today.
            #
            # New behaviour: log the disagreement and REDUCE strategy
            # confidence by 0.7x (same pattern the against-daily-trend check
            # uses at line 3052). Position sizing will down-weight the trade
            # but it still gets a chance to fire.
            signal.confidence *= 0.7
            self._audit("ml", signal.symbol, "proceed", "ml_disagreement_low_conf",
                        ml_signal=ml_signal, ml_conf=round(ml_confidence, 3),
                        expected=expected_ml_signal,
                        strategy_conf_after=round(signal.confidence, 3))
        # Multi-timeframe confirmation
        try:
            daily_data = self.data_provider.get_market_data(signal.symbol, frequency_type='daily', frequency=1)
            if not daily_data.empty and len(daily_data) > 20:
                daily_trend = 'up' if daily_data['Close'].iloc[-1] > daily_data['Close'].iloc[-20] else 'down'
                
                if (signal.signal_type == SignalType.BUY and daily_trend == 'down') or \
                (signal.signal_type == SignalType.SELL and daily_trend == 'up'):
                    signal.confidence *= 0.7
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=signal.symbol,
                        title=f"⚠️ Against Daily Trend",
                        message=f"Signal is against daily trend ({daily_trend}), reducing confidence",
                        importance=6
                    ))
        except Exception as e:
            logger.error(f"Multi-timeframe check failed: {e}")

        # ================================================================
        # PROFESSIONAL TRADING FILTER - Institutional Quality Gate
        # This is where we filter out low-quality setups that retail loses on
        # ================================================================
        if self.pro_trading_wrapper is not None:
            try:
                # Get market data for regime/context analysis
                market_df = None
                if self.data_provider:
                    raw_data = self.data_provider.get_market_data(signal.symbol)
                    if not raw_data.empty:
                        # Rename columns to lowercase for pro modules
                        market_df = raw_data.rename(columns={
                            'Open': 'open', 'High': 'high', 'Low': 'low',
                            'Close': 'close', 'Volume': 'volume'
                        })

                # Build additional signals for confluence
                additional_signals = {
                    'trend_aligned': ml_signal == (1 if signal.signal_type == SignalType.BUY else -1),
                    'ml_confirmed': ml_signal == (1 if signal.signal_type == SignalType.BUY else -1),
                    'volume_confirmed': signal.reasoning.get('volume_confirmation', False),
                    'momentum_confirmed': signal.reasoning.get('momentum_confirmed', False),
                }

                # Evaluate through professional filter
                pro_decision = self.pro_trading_wrapper.evaluate_trade_signal(
                    symbol=signal.symbol,
                    direction='long' if signal.signal_type == SignalType.BUY else 'short',
                    strategy=signal.reasoning.get('strategy', 'unknown'),
                    entry_price=signal.entry_price,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size=signal.position_size if signal.position_size else 100,
                    confidence=signal.confidence,
                    market_data=market_df,
                    additional_signals=additional_signals
                )

                if not pro_decision.allowed:
                    # Trade blocked by professional filters
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.DECISION,
                        symbol=signal.symbol,
                        title=f"🏛️ Professional Filter: BLOCKED",
                        message=pro_decision.final_recommendation,
                        data={
                            'reasons_against': pro_decision.reasons_against,
                            'regime': pro_decision.regime,
                            'session': pro_decision.session,
                            'quality_score': pro_decision.quality_score,
                            'confluence_score': pro_decision.confluence_score
                        },
                        importance=8
                    ))

                    # Log the blocked trade for analysis
                    log_decision(
                        symbol=signal.symbol,
                        decision='BLOCKED_BY_PRO_FILTER',
                        reason='; '.join(pro_decision.reasons_against[:3]),
                        factors={
                            'regime': pro_decision.regime,
                            'session': pro_decision.session,
                            'quality': pro_decision.quality_score,
                            'confluence': pro_decision.confluence_score
                        },
                        signals={'strategy': signal.strength, 'ml': ml_signal},
                        indicators=signal.reasoning.get('indicators', {}),
                        confidence=signal.confidence,
                        mode=self.mode.value
                    )
                    return  # Don't take this trade

                # Trade approved - use adjusted parameters from pro filter
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"🏛️ Professional Filter: APPROVED",
                    message=f"Quality: {pro_decision.quality_score:.0f}/100, "
                           f"Confluence: {pro_decision.confluence_score}, "
                           f"R-Target: {pro_decision.r_multiple_target:.1f}",
                    data={
                        'reasons_for': pro_decision.reasons_for,
                        'regime': pro_decision.regime,
                        'session': pro_decision.session,
                        'adjusted_stop': pro_decision.stop_loss,
                        'adjusted_size': pro_decision.position_size
                    },
                    importance=7
                ))

                # Apply adjustments from pro filter
                signal.stop_loss = pro_decision.stop_loss
                signal.position_size = int(pro_decision.position_size) if pro_decision.position_size else signal.position_size

            except Exception as e:
                logger.error(f"Professional filter error (continuing with trade): {e}")
                # On error, continue with original signal - fail open

        # Check portfolio concentration
        # v-parity-fix-2026-04-20: include simulated_positions so sim
        # concentration cap matches live 1:1. Previously only live
        # positions counted, letting sim oversize the book.
        all_open_positions = list(self.positions.values()) + list(self.simulated_positions.values())
        if all_open_positions:
            total_value = sum(p.current_price * p.quantity for p in all_open_positions)
            max_position_value = total_value * Config().MAX_POSITION_VALUE_PCT

            if signal.position_size * signal.entry_price > max_position_value:
                signal.position_size = int(max_position_value / signal.entry_price)
        # Calculate position size with explanation
        position_size = await self.risk_manager.calculate_position_size_with_commentary(
            signal, signal.entry_price
        )
        
        if position_size == 0:
            return
        
        signal.position_size = position_size

	# After creating position, initialize exit tracking
        if self.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
            self.exit_manager.initialize_position_tracking(
                signal.symbol,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit
            )
        
        # Final decision
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.DECISION,
            symbol=signal.symbol,
            title=f"✅ Trade Decision: {signal.signal_type.name}",
            message=f"After careful analysis, I've decided to {'BUY' if signal.signal_type.value == 1 else 'SELL'} {position_size} shares of {signal.symbol}",
            data={
                'total_cost': position_size * signal.entry_price,
                'max_risk': position_size * (signal.entry_price - signal.stop_loss),
                'max_reward': position_size * (signal.take_profit - signal.entry_price),
                'reasoning': signal.reasoning
            },
            confidence=signal.confidence,
            importance=10
        ))
        self._audit("signal_router", signal.symbol, "accepted",
                    signal.reasoning.get("strategy", "unknown"),
                    side=signal.signal_type.name,
                    qty=position_size,
                    entry=round(signal.entry_price, 2),
                    stop=round(signal.stop_loss, 2),
                    target=round(signal.take_profit, 2),
                    total_cost=round(position_size * signal.entry_price, 2))

        # v-rr-audit-2026-06-01: warn if the accepted signal's
        # risk:reward is below the configured floor. Would have
        # caught the 2026-05-28 bb_middle bug in minutes: that bug
        # produced R:R 0.7-1.4 across mean-rev signals because the
        # strategy capped take_profit at bb_middle when bb_middle
        # sat just above entry. The audit fires at the router (one
        # place) so it covers every strategy without per-strategy
        # plumbing.
        try:
            _stop_dist = abs(signal.entry_price - signal.stop_loss)
            _target_dist = abs(signal.take_profit - signal.entry_price)
            if _stop_dist > 0:
                _actual_rr = _target_dist / _stop_dist
                # Tolerance: most signals target 2.0 R:R or higher.
                # Anything below 1.8 is suspect — either a tight-
                # target bug (like bb_middle) or an unusual setup
                # the strategy author didn't intend.
                _RR_FLOOR = 1.8
                if _actual_rr < _RR_FLOOR:
                    logger.warning(
                        "rr_audit_low symbol=%s strategy=%s "
                        "entry=%.4f stop=%.4f target=%.4f "
                        "stop_dist=%.4f target_dist=%.4f actual_rr=%.2f "
                        "floor=%.2f — strategy may be capping target",
                        signal.symbol,
                        signal.reasoning.get("strategy", "unknown"),
                        signal.entry_price, signal.stop_loss, signal.take_profit,
                        _stop_dist, _target_dist, _actual_rr, _RR_FLOOR,
                    )
        except Exception as _exc:
            # Audit must never block the trade. Swallow.
            logger.debug("rr_audit calc failed: %s", _exc)
        
        # Execute the trade based on mode
        position = None
        if self.mode == TradingMode.SIMULATION_WITH_COMMENTARY:
            # v-parity-fix-2026-04-20: OCO validation now runs in sim too
            # (previously live-only at line ~386). Sim must reject the same
            # impossible stop/target configurations live would reject so
            # sim decisions match live 1:1.
            is_short_sim = (signal.signal_type == SignalType.SELL
                            and signal.symbol not in self.simulated_positions)
            if not self._validate_oco_prices(signal.symbol, signal.stop_loss,
                                              signal.take_profit, is_short_sim):
                self._audit("signal_router", signal.symbol, "reject_oco_invalid_sim",
                            signal.reasoning.get("strategy", "unknown"),
                            side=signal.signal_type.name,
                            stop=round(signal.stop_loss, 2),
                            target=round(signal.take_profit, 2))
                return
            # Create simulated position
            position = Position(
                symbol=signal.symbol,
                entry_price=signal.entry_price,
                current_price=signal.entry_price,
                quantity=signal.position_size,
                side='long' if signal.signal_type == SignalType.BUY else 'short',
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                entry_time=datetime.now(),
                reasoning=signal.reasoning,
                original_stop=signal.stop_loss,
                mode="simulation",
                managed_by_bot=True,  # bot-opened sim trade
            )
            self.simulated_positions[signal.symbol] = position

            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=signal.symbol,
                title=f"📋 Simulated Position Opened",
                message=f"Position tracked in simulation. I'll monitor it and explain my management decisions.",
                importance=7
            ))
            
        elif self.mode == TradingMode.LIVE:
            # Execute real trade
            success = await self._execute_real_trade(signal)
            if success:
                # Don't create position here - wait for order fill confirmation
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"📋 Order Placed Successfully",
                    message=f"Waiting for order fill confirmation before tracking position",
                    importance=8
                ))
            else:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=signal.symbol,
                    title=f"❌ Trade Execution Failed",
                    message=f"Could not execute trade for {signal.symbol}",
                    importance=10
                ))
        # Only initialize exit tracking if position was created
        if position and hasattr(self, 'exit_manager'):
            self.exit_manager.initialize_position_tracking(
                signal.symbol,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit
            )

        # Register position with professional exit manager for R-multiple tracking
        if position and self.pro_trading_wrapper is not None:
            try:
                # Get ATR for the symbol
                atr = None
                if self.data_provider:
                    raw_data = self.data_provider.get_market_data(signal.symbol)
                    if not raw_data.empty and len(raw_data) >= 14:
                        high = raw_data['High'].values
                        low = raw_data['Low'].values
                        close = raw_data['Close'].values
                        tr = np.maximum(
                            high[1:] - low[1:],
                            np.maximum(
                                np.abs(high[1:] - close[:-1]),
                                np.abs(low[1:] - close[:-1])
                            )
                        )
                        atr = float(np.mean(tr[-14:]))

                # Open position in professional exit manager
                self.pro_trading_wrapper.exit_manager.open_position(
                    symbol=signal.symbol,
                    direction='long' if signal.signal_type == SignalType.BUY else 'short',
                    entry_price=signal.entry_price,
                    position_size=signal.position_size,
                    stop_loss=signal.stop_loss,
                    atr=atr
                )

                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=signal.symbol,
                    title=f"🏛️ Pro Exit Manager: Tracking",
                    message=f"Position registered for R-multiple management. "
                           f"Scale-out targets: 1R (50%), 2R (25%), 3R (25%)",
                    importance=6
                ))
            except Exception as e:
                logger.error(f"Failed to register position with pro exit manager: {e}")

    async def close_position_manually(self, symbol: str, position_type: str):
        """Manually close a position"""
        position = None
        if position_type == 'real':
            position = self.positions.get(symbol)
        elif position_type == 'simulated':
            position = self.simulated_positions.get(symbol)

        if position:
            await self._close_position_with_commentary(position, "manual_override")
   
    async def _manage_positions_with_commentary(self):
        """Manage positions with detailed commentary"""
        positions_to_check = self.simulated_positions if self.mode == TradingMode.SIMULATION_WITH_COMMENTARY else self.positions
        
        if not positions_to_check:
            return
        
        for symbol, position in list(positions_to_check.items()):
            try:
                # Check if position is None
                if position is None:
                    logger.warning(f"None position found for {symbol}, removing")
                    del positions_to_check[symbol]
                    continue

                # CRITICAL: Skip external/manually managed positions when manual_close_only is ON
                # These positions should only be managed by the user, not the bot
                is_external = getattr(position, 'is_external', False)
                is_manually_managed = getattr(position, 'is_manually_managed', False)

                if not self._is_auto_managed(position):
                    self._audit("position_manager", symbol, "hold",
                                "not_auto_managed",
                                managed_by_bot=getattr(position, 'managed_by_bot', False),
                                external=is_external, manually_managed=is_manually_managed,
                                manual_close_only=self.manual_close_only)
                    # Only update the price for display purposes, but don't take any action
                    if self.data_provider:
                        quote = self.data_provider.get_quote(symbol)
                        if quote:
                            position.current_price = quote.get('last', position.current_price)
                            if position.side == 'short':
                                position.unrealized_pnl = (position.entry_price - position.current_price) * position.quantity
                            else:
                                position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                    continue  # Skip all automatic management for this position

                # Get current price - always use real data when available
                if self.data_provider:
                    quote = self.data_provider.get_quote(symbol)
                    if quote and quote.get('last'):
                        current_price = quote.get('last')
                    else:
                        # Fallback: keep current price if real data unavailable
                        logger.warning(f"Could not get real price for {symbol}, using last known price")
                        current_price = position.current_price
                    
                    # Update position
                    old_price = position.current_price
                    position.current_price = current_price
                    if position.side == 'short':
                        position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                    else:  # long
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    
                    # Calculate P&L percentage
                    if position.side == 'short':
                        pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100
                    else:
                        pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                    
                    # Alert on significant losses
                    if position.unrealized_pnl < 0:
                        loss_pct = abs(pnl_pct)
                        
                        # Check if we should alert (avoid spamming)
                        should_alert = False
                        alert_info = self.loss_alert_tracker.get(symbol, {})
                        last_alert_time = alert_info.get('last_alert_time')
                        last_loss_pct = alert_info.get('last_loss_pct', 0)
                        
                        # Alert if: first time, or loss increased by 0.5%, or 5 minutes passed
                        if (not last_alert_time or 
                            loss_pct >= last_loss_pct + 0.5 or
                            (datetime.now() - last_alert_time).total_seconds() > 300):
                            should_alert = True
                        
                        if should_alert:
                            # Critical alert for large losses
                            if loss_pct >= 3.0:  # 3% or more loss
                                self.commentary.add_commentary(TradingCommentary(
                                    timestamp=datetime.now(),
                                    type=CommentaryType.WARNING,
                                    symbol=symbol,
                                    title=f"🚨 CRITICAL LOSS: {symbol}",
                                    message=f"Position down {loss_pct:.1f}% (${abs(position.unrealized_pnl):.2f})\n"
                                            f"IMMEDIATE ACTION REQUIRED!\n"
                                            f"Stop loss: ${position.stop_loss:.2f}",
                                    data={
                                        'current_price': current_price,
                                        'entry_price': position.entry_price,
                                        'unrealized_pnl': position.unrealized_pnl,
                                        'pnl_percentage': pnl_pct,
                                        'distance_to_stop': abs((current_price - position.stop_loss) / current_price * 100)
                                    },
                                    importance=10
                                ))
                            elif loss_pct >= 1.0:  # 1% or more loss
                                self.commentary.add_commentary(TradingCommentary(
                                    timestamp=datetime.now(),
                                    type=CommentaryType.WARNING,
                                    symbol=symbol,
                                    title=f"⚠️ Position Losing: {symbol}",
                                    message=f"Down {loss_pct:.1f}% (${abs(position.unrealized_pnl):.2f})\n"
                                            f"Stop loss at ${position.stop_loss:.2f}",
                                    data={
                                        'current_price': current_price,
                                        'entry_price': position.entry_price,
                                        'unrealized_pnl': position.unrealized_pnl,
                                        'pnl_percentage': pnl_pct,
                                        'distance_to_stop': abs((current_price - position.stop_loss) / current_price * 100)
                                    },
                                    importance=7 if loss_pct < 2 else 9
                                ))
                            
                            # Update tracker
                            self.loss_alert_tracker[symbol] = {
                                'last_alert_time': datetime.now(),
                                'last_loss_pct': loss_pct
                            }
                    
                    # Regular position updates for smaller moves
                    price_change_pct = ((current_price - old_price) / old_price) * 100
                    if abs(price_change_pct) > 0.05 and abs(pnl_pct) >= 0.5:  # Lower threshold
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.MARKET_ANALYSIS,
                            symbol=symbol,
                            title=f"📊 Position Update: {symbol}",
                            message=f"Position {'profitable' if position.unrealized_pnl > 0 else 'losing'} "
                                f"by {abs(pnl_pct):.1f}%",
                            data={
                                'current_price': current_price,
                                'entry_price': position.entry_price,
                                'unrealized_pnl': position.unrealized_pnl,
                                'pnl_percentage': pnl_pct,
                                'distance_to_stop': ((current_price - position.stop_loss) / current_price) * 100,
                                'distance_to_target': ((position.take_profit - current_price) / current_price) * 100
                            },
                            importance=4
                        ))
                    
                    # ================================================================
                    # ================================================================
                    # ATR CALCULATION (used by scale-trail + pro exit manager)
                    # ================================================================
                    current_atr = None
                    market_df = None
                    # v-proactive-exit-fix-2026-04-27: build indicator view from
                    # raw_data alongside ATR so the proactive-exit check can read
                    # MACD/RSI/ADX without depending on a `market_data` variable
                    # that doesn't exist in this scope.
                    proactive_indicators: dict = {}
                    if self.data_provider and not getattr(position, 'is_long_term', False):
                        try:
                            raw_data = self.data_provider.get_market_data(symbol)
                            if not raw_data.empty and len(raw_data) >= 14:
                                high = raw_data['High'].values
                                low = raw_data['Low'].values
                                close = raw_data['Close'].values
                                tr = np.maximum(
                                    high[1:] - low[1:],
                                    np.maximum(
                                        np.abs(high[1:] - close[:-1]),
                                        np.abs(low[1:] - close[:-1])
                                    )
                                )
                                current_atr = float(np.mean(tr[-14:]))
                                market_df = raw_data.rename(columns={
                                    'Open': 'open', 'High': 'high', 'Low': 'low',
                                    'Close': 'close', 'Volume': 'volume'
                                })
                                # Compute the three indicators proactive-exit needs.
                                # ta.* funcs are vectorised; cheap on 5-min bars.
                                try:
                                    import ta as _ta
                                    close_s = raw_data['Close']
                                    high_s = raw_data['High']
                                    low_s = raw_data['Low']
                                    macd_obj = _ta.trend.MACD(close_s)
                                    macd_series = macd_obj.macd()
                                    macd_sig_series = macd_obj.macd_signal()
                                    rsi_series = _ta.momentum.rsi(close_s, window=14)
                                    adx_series = _ta.trend.ADXIndicator(high_s, low_s, close_s, window=14).adx()
                                    proactive_indicators = {
                                        'macd':        float(macd_series.iloc[-1])    if not macd_series.empty    and not np.isnan(macd_series.iloc[-1])    else 0.0,
                                        'macd_signal': float(macd_sig_series.iloc[-1])if not macd_sig_series.empty and not np.isnan(macd_sig_series.iloc[-1])else 0.0,
                                        'rsi':         float(rsi_series.iloc[-1])     if not rsi_series.empty     and not np.isnan(rsi_series.iloc[-1])     else 50.0,
                                        'adx':         float(adx_series.iloc[-1])     if not adx_series.empty     and not np.isnan(adx_series.iloc[-1])     else 0.0,
                                        'adx_prev':    float(adx_series.iloc[-2])     if len(adx_series) >= 2     and not np.isnan(adx_series.iloc[-2])     else 0.0,
                                    }
                                except Exception as ind_err:
                                    logger.debug(f"proactive indicators calc skipped for {symbol}: {ind_err}")
                                    proactive_indicators = {}
                        except Exception as e:
                            logger.debug(f"ATR calc error for {symbol}: {e}")

                    # ================================================================
                    # SCALE-OUT AT 1R + ATR TRAILING STOP
                    # ================================================================
                    if self._is_auto_managed(position):
                        # v-position-fsm-2026-04-30 (Phase 1): every exit
                        # rule below is now gated by the FSM dispatcher.
                        # The rule is only ALLOWED to evaluate if the
                        # current PositionState explicitly permits it
                        # (see core/position_state.py:ALLOWED_EXITS).
                        # Each rule is also responsible for transitioning
                        # the position when its post-condition fires
                        # (e.g. breakeven_lift triggers LIVE → AT_BREAKEVEN).
                        #
                        # Ensure the per-position lock exists before any
                        # transition can be attempted.
                        self._ensure_position_lock(position)

                        # Zombie auto-promotion: if a previous tick left the
                        # position in EXITING longer than the timeout, mark
                        # it ZOMBIE and skip all further exit logic.
                        if await self._promote_zombie_if_stalled(position):
                            continue

                        try:
                            _state = PositionState(position.state)
                        except ValueError:
                            # Defensive: any unrecognized state value is
                            # treated as ZOMBIE so the engine stays hands-off.
                            position.state = PositionState.ZOMBIE.value
                            position.zombie_reason = f"unknown_state_{position.state!r}"
                            continue
                        if _state in (PositionState.EXITING, PositionState.CLOSED, PositionState.ZOMBIE):
                            # Terminal/in-flight states have no allowed rules.
                            continue

                        # --- Breakeven Stop Lift (fires before everything else) ---
                        # v-breakeven-stop-2026-04-28: once the trade has moved
                        # +BREAKEVEN_ACTIVATION_R (default 0.5R) in our favor,
                        # ratchet stop_loss to entry. Prevents the "winner
                        # turned loser" pattern (PLTR/TSLA today: both went
                        # positive then reversed to full -0.5R/-1.5R losses).
                        be_new_stop = None
                        if is_exit_rule_allowed(_state, ExitRule.BREAKEVEN_LIFT):
                            # v-oversold-v2-2026-05-19: per-position override
                            # (sim-path mirror of the live-path block above).
                            _r_dict_s = getattr(position, 'reasoning', {}) or {}
                            _be_mult_s = _r_dict_s.get('breakeven_atr_mult')
                            _be_atr_s = _r_dict_s.get('atr')
                            _be_sd_s = _r_dict_s.get('stop_distance')
                            if _be_mult_s and _be_atr_s and _be_sd_s and _be_sd_s > 0:
                                _be_activation_r_s = _be_mult_s * _be_atr_s / _be_sd_s
                            else:
                                _be_activation_r_s = Config().BREAKEVEN_ACTIVATION_R
                            be_new_stop = self.scale_trail.update_peak_and_breakeven(
                                position, current_price, _be_activation_r_s,
                            )
                        if be_new_stop is not None:
                            old_stop = position.stop_loss
                            position.stop_loss = be_new_stop
                            position.breakeven_lifted = True
                            # FSM transition LIVE → AT_BREAKEVEN. Atomic.
                            await try_transition(
                                position, PositionState.AT_BREAKEVEN,
                                "+0.5R reached, stop ratcheted to entry",
                                audit_fn=self._audit,
                            )
                            # Re-read state for the rest of this tick — the
                            # rules permitted AFTER this transition are
                            # different (no longer in LIVE's set).
                            _state = PositionState(position.state)
                            self._audit("breakeven_stop", symbol, "lifted",
                                        "peak_above_activation",
                                        peak_r=round(position.peak_favorable_r, 3),
                                        old_stop=round(old_stop, 2),
                                        new_stop=round(be_new_stop, 2))
                            self.commentary.add_commentary(TradingCommentary(
                                timestamp=datetime.now(),
                                type=CommentaryType.DECISION,
                                symbol=symbol,
                                title=f"🔒 Breakeven Stop Locked",
                                message=(
                                    f"Trade reached {position.peak_favorable_r:+.2f}R favor. "
                                    f"Stop lifted from ${old_stop:.2f} to ${be_new_stop:.2f} "
                                    f"(entry). Cannot become a loser from here."
                                ),
                                importance=8,
                            ))

                        # --- 1R Partial Exit ---
                        partial = None
                        if is_exit_rule_allowed(_state, ExitRule.SCALE_OUT_1R):
                            # v-oversold-v2-2026-05-19: per-position override
                            # (sim-path mirror of the live-path block above).
                            _so_override_s = (getattr(position, 'reasoning', {}) or {}).get(
                                'scale_out_r_override', 1.0,
                            )
                            partial = self.scale_trail.check_partial_exit_at_r(
                                position, current_price, activation_r=_so_override_s,
                            )
                        if partial is not None and partial.exit_qty > 0:
                            position.quantity -= partial.exit_qty
                            position.scaled_out = True
                            position.stop_loss = partial.new_stop
                            # FSM transition AT_BREAKEVEN → AT_1R. Proactive_exit
                            # is disconnected after this transition (see ALLOWED_EXITS
                            # for AT_1R — it has no PROACTIVE_EXIT).
                            await try_transition(
                                position, PositionState.AT_1R,
                                "+1R reached, 50% scaled out",
                                audit_fn=self._audit,
                            )
                            _state = PositionState(position.state)
                            self._audit("scale_out", symbol, "partial_exit_1R",
                                        "1R_reached",
                                        exit_qty=partial.exit_qty,
                                        remaining=position.quantity,
                                        new_stop=round(partial.new_stop, 2),
                                        pnl_locked=round(partial.exit_qty * abs(current_price - position.entry_price), 2))
                            self.commentary.add_commentary(TradingCommentary(
                                timestamp=datetime.now(),
                                type=CommentaryType.DECISION,
                                symbol=symbol,
                                title=f"🎯 1R Reached — Scaling Out 50%",
                                message=(
                                    f"Price hit 1R at ${current_price:.2f}. "
                                    f"Closed {partial.exit_qty} shares, keeping {position.quantity}. "
                                    f"Stop moved to breakeven at ${partial.new_stop:.2f}."
                                ),
                                importance=9,
                            ))

                        # --- ATR Trailing Stop ---
                        # Only allowed in AT_1R and TRAILING per ALLOWED_EXITS.
                        if current_atr is not None and is_exit_rule_allowed(_state, ExitRule.TRAILING_STOP_ATR):
                            new_trail = self.scale_trail.update_trailing_stop(
                                position, current_price, current_atr,
                            )
                            if new_trail is not None:
                                old_trail = position.trailing_stop
                                position.trailing_stop = new_trail
                                # AT_1R → TRAILING on first valid trail set.
                                if _state == PositionState.AT_1R:
                                    await try_transition(
                                        position, PositionState.TRAILING,
                                        "trailing stop engaged",
                                        audit_fn=self._audit,
                                    )
                                    _state = PositionState(position.state)
                                self._audit("trailing_stop", symbol, "update",
                                            "atr_trail",
                                            old=round(old_trail or 0, 2),
                                            new=round(new_trail, 2),
                                            atr=round(current_atr, 3))

                            if self.scale_trail.is_trailing_stop_hit(position, current_price):
                                # Atomic intent-to-close: try to acquire EXITING.
                                # If another concurrent rule already did, we
                                # short-circuit harmlessly (transition_to returns False).
                                if not await try_transition(
                                    position, PositionState.EXITING,
                                    "trailing_stop_hit",
                                    audit_fn=self._audit,
                                ):
                                    continue
                                self._audit("trailing_stop", symbol, "exit",
                                            "atr_trail_hit",
                                            price=round(current_price, 2),
                                            trail=round(position.trailing_stop, 2))
                                self.commentary.add_commentary(TradingCommentary(
                                    timestamp=datetime.now(),
                                    type=CommentaryType.DECISION,
                                    symbol=symbol,
                                    title=f"📉 Trailing Stop Hit",
                                    message=(
                                        f"ATR trail at ${position.trailing_stop:.2f} breached "
                                        f"at ${current_price:.2f}. Locking in gains."
                                    ),
                                    importance=9,
                                ))
                                await self._close_position_with_commentary(
                                    position, "trailing_stop_atr"
                                )
                                continue

                        # --- Proactive exit: bail when thesis breaks at -0.5R ---
                        # v-proactive-exit-2026-04-27: avoid riding losers all
                        # the way to the full stop when indicators have already
                        # turned against the position. RIOT short 04-27 took
                        # -$236 full stop with 75-min hold; this fires earlier.
                        # Indicators are computed from raw_data above (no
                        # dependency on a `market_data` variable in this scope).
                        #
                        # v-proactive-time-floor-2026-04-30 (Fix A): the
                        # whipsaw analysis on 2026-04-30 showed 100% of
                        # MACD-flip-bullish exits and 100% of RSI-below-50
                        # exits became winners if held; ALL whipsaws
                        # exited within 5 min of entry. We need to give
                        # the trade time to develop before letting a
                        # 5-min indicator wiggle kick it out. News theses
                        # operate on hours, not bars. Exception: hard
                        # stop-loss is never gated — that's the genuine
                        # broken-thesis exit. The time floor only
                        # protects the "soft" proactive triggers.
                        _strategy = (getattr(position, 'reasoning', {}) or {}).get('strategy', '')
                        _hold_min = (datetime.now() - position.entry_time).total_seconds() / 60.0
                        _is_news_trade = 'news' in _strategy.lower() or _strategy == 'free_news_sentiment'
                        _is_mean_rev = 'mean_reversion' in _strategy.lower()
                        # Per-strategy minimum age before proactive exit fires.
                        # News: 30 min — institutional re-rate plays out over
                        # hours. MeanRev: 10 min — bounces are faster. Other
                        # (momentum/breakout): 15 min default.
                        if _is_news_trade:
                            _min_age = Config().PROACTIVE_EXIT_MIN_AGE_NEWS
                        elif _is_mean_rev:
                            _min_age = Config().PROACTIVE_EXIT_MIN_AGE_MEANREV
                        else:
                            _min_age = Config().PROACTIVE_EXIT_MIN_AGE_DEFAULT

                        _proactive_allowed = _hold_min >= _min_age
                        # FSM gate: PROACTIVE_EXIT is in ALLOWED_EXITS only
                        # for LIVE and AT_BREAKEVEN. Past 1R / TRAILING the
                        # rule is physically disconnected — the trade is in
                        # profit-protection territory, not thesis-break
                        # territory. This is what the user asked for: the
                        # rule is not behind an `if`, it's not in the set.
                        _fsm_allows_proactive = is_exit_rule_allowed(_state, ExitRule.PROACTIVE_EXIT)
                        if Config().ENABLE_PROACTIVE_EXIT and proactive_indicators and _fsm_allows_proactive and not _proactive_allowed:
                            # Suppress and audit so we can measure how often
                            # the time-floor saved us a whipsaw.
                            self._audit(
                                "proactive_exit", symbol, "suppressed", "below_min_age",
                                hold_min=round(_hold_min, 1),
                                min_age=_min_age,
                                strategy=_strategy or "unknown",
                                state=_state.value,
                            )
                        if (Config().ENABLE_PROACTIVE_EXIT and proactive_indicators
                                and _fsm_allows_proactive and _proactive_allowed):
                            proactive_reason = self.scale_trail.check_proactive_exit(
                                position, current_price, proactive_indicators,
                            )
                            if proactive_reason is not None:
                                # R-multiple at fire time
                                stop_dist = abs(position.entry_price - (position.original_stop or position.stop_loss))
                                if position.side == 'long':
                                    pnl_r = (current_price - position.entry_price) / stop_dist if stop_dist > 0 else 0
                                else:
                                    pnl_r = (position.entry_price - current_price) / stop_dist if stop_dist > 0 else 0
                                # Atomic intent-to-close. If another rule
                                # already grabbed EXITING, this returns False
                                # and we skip the duplicate close.
                                if not await try_transition(
                                    position, PositionState.EXITING,
                                    f"proactive_{proactive_reason}",
                                    audit_fn=self._audit,
                                ):
                                    continue
                                self._audit("proactive_exit", symbol, "exit",
                                            proactive_reason,
                                            pnl_r=round(pnl_r, 3),
                                            price=round(current_price, 2),
                                            entry=round(position.entry_price, 2))
                                self.commentary.add_commentary(TradingCommentary(
                                    timestamp=datetime.now(),
                                    type=CommentaryType.DECISION,
                                    symbol=symbol,
                                    title=f"🚪 Proactive Exit — Thesis Broken",
                                    message=(
                                        f"At {pnl_r:+.2f}R, indicators turned against the "
                                        f"position ({proactive_reason}). Exiting at "
                                        f"${current_price:.2f} instead of riding to full stop."
                                    ),
                                    importance=9,
                                ))
                                await self._close_position_with_commentary(
                                    position, f"proactive_{proactive_reason}"
                                )
                                continue

                        # --- Thesis re-validation (shadow-mode by default) ---
                        # v-thesis-revalidate-2026-04-28: catches the "trade
                        # sat flat for 30 min while the underlying thesis
                        # broke" case. Only fires when no other exit triggered
                        # this tick. Closes only when BOTH news AND indicators
                        # are against the position. Logs every decision; in
                        # shadow mode (default), does NOT actually close.
                        try:
                            # Pass the same proactive_indicators dict already
                            # built from raw_data above; add 'close' so the
                            # SMA50 check has a price to compare to.
                            tv_indicators = dict(proactive_indicators) if proactive_indicators else {}
                            if market_df is not None and len(market_df) > 50:
                                try:
                                    tv_indicators['sma_50'] = float(market_df['close'].rolling(50).mean().iloc[-1])
                                except Exception:
                                    pass
                            tv_indicators['close'] = current_price
                            await self._revalidate_thesis(
                                symbol, position, current_price, tv_indicators,
                            )
                        except Exception as exc:
                            logger.debug(f"thesis_revalidate error for {symbol}: {exc}")

                        # --- News thesis flip check (v-newsbus-gates-2026-09-09) ---
                        # Quick NewsBus-based sentiment flip detection. Fires when
                        # fresh news sentiment has flipped against the position
                        # direction (e.g., bullish news on a short position).
                        # Gated by ENABLE_NEWS_THESIS_EXIT (default False).
                        try:
                            if await self._check_news_thesis_flip(symbol, position, current_price):
                                await self._close_position_with_commentary(
                                    position, "news_thesis_flip"
                                )
                                continue
                        except Exception as exc:
                            logger.debug(f"news_thesis_flip error for {symbol}: {exc}")

                    # ================================================================
                    # PROFESSIONAL EXIT MANAGER
                    # ================================================================
                    if self.pro_trading_wrapper is not None and not getattr(position, 'is_long_term', False):
                        try:

                            # Check for exit signals from professional exit manager
                            pro_exit_signals = self.pro_trading_wrapper.update_positions(
                                current_prices={symbol: current_price},
                                market_data={symbol: market_df} if market_df is not None else None,
                                atrs={symbol: current_atr} if current_atr else None
                            )

                            # Process any exit signals
                            for exit_sig in pro_exit_signals:
                                if exit_sig.symbol == symbol:
                                    self.commentary.add_commentary(TradingCommentary(
                                        timestamp=datetime.now(),
                                        type=CommentaryType.DECISION,
                                        symbol=symbol,
                                        title=f"🏛️ Pro Exit: {exit_sig.exit_reason.value.upper()}",
                                        message=f"{exit_sig.message}\n"
                                               f"Exiting {exit_sig.exit_size_pct:.0%} @ ${exit_sig.exit_price:.2f}",
                                        data={
                                            'r_multiple': exit_sig.r_multiple,
                                            'exit_reason': exit_sig.exit_reason.value,
                                            'exit_size_pct': exit_sig.exit_size_pct
                                        },
                                        importance=9
                                    ))

                                    # Handle partial exit vs full exit
                                    if exit_sig.exit_size_pct < 1.0:
                                        # Partial exit - scale out
                                        exit_quantity = int(position.quantity * exit_sig.exit_size_pct)
                                        if exit_quantity > 0 and self._is_auto_managed(position):
                                            # Execute partial close
                                            position.quantity -= exit_quantity
                                            self.commentary.add_commentary(TradingCommentary(
                                                timestamp=datetime.now(),
                                                type=CommentaryType.DECISION,
                                                symbol=symbol,
                                                title=f"📊 Scaled Out {exit_sig.exit_size_pct:.0%}",
                                                message=f"Locked in profit at {exit_sig.r_multiple:.1f}R. "
                                                       f"Remaining: {position.quantity} shares",
                                                importance=8
                                            ))
                                    else:
                                        # Full exit
                                        if self._is_auto_managed(position):
                                            await self._close_position_with_commentary(
                                                position,
                                                f"pro_exit_{exit_sig.exit_reason.value}"
                                            )
                                            continue  # Position closed, move to next

                        except Exception as e:
                            logger.error(f"Professional exit manager error for {symbol}: {e}")
                            # Continue with normal exit logic on error

                    # Check exit conditions with reasoning (skip if long-term)
                    if not getattr(position, 'is_long_term', False):
                        should_exit, exit_reason = await self._evaluate_exit_conditions(position, current_price)
                        
                        if should_exit:
                            # Check if manual close only is enabled
                            if self.auto_close_disabled:
                                self.commentary.add_commentary(TradingCommentary(
                                    timestamp=datetime.now(),
                                    type=CommentaryType.INFO,
                                    symbol=position.symbol,
                                    title=f"📍 Exit Signal (Manual Mode)",
                                    message=f"Exit condition met ({exit_reason}) but manual close only is ON. "
                                           f"Current P&L: ${position.unrealized_pnl:.2f}. YOU must close manually.",
                                    importance=7
                                ))
                            else:
                                await self._close_position_with_commentary(position, exit_reason)
                    else:
                        # Log that we're skipping long-term position
                        if not hasattr(self, '_long_term_logged') or symbol not in self._long_term_logged:
                            if not hasattr(self, '_long_term_logged'):
                                self._long_term_logged = set()
                            self._long_term_logged.add(symbol)
                            
                            self.commentary.add_commentary(TradingCommentary(
                                timestamp=datetime.now(),
                                type=CommentaryType.DECISION,
                                symbol=symbol,
                                title=f"🔒 Long-Term Hold Protected",
                                message=f"{symbol} is marked as long-term hold - automatic exit conditions disabled",
                                importance=5
                            ))
                
            except Exception as e:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=symbol,
                    title=f"⚠️ Error Managing {symbol}",
                    message=str(e),
                    importance=6
                ))
    async def _evaluate_exit_conditions(self, position, current_price: float) -> Tuple[bool, str]:
        """Evaluate exit conditions with detailed reasoning"""
        # CRITICAL: Never auto-exit external/manually managed positions
        is_external = getattr(position, 'is_external', False)
        is_manually_managed = getattr(position, 'is_manually_managed', False)

        if is_external or is_manually_managed:
            # External positions are NEVER auto-managed
            return False, ""

        # Check if stop loss / take profit are hit based on position side.
        # Both use the same simple price comparison — no indicator logic.
        # The dynamic exit manager below still runs for softer/partial exits
        # but hard SL/TP must fire regardless of regime.
        stop_loss_hit = False
        take_profit_hit = False
        if position.side == 'short':
            stop_loss_hit = current_price >= position.stop_loss
            take_profit_hit = (
                position.take_profit is not None
                and current_price <= position.take_profit
            )
        else:  # long
            stop_loss_hit = current_price <= position.stop_loss
            take_profit_hit = (
                position.take_profit is not None
                and current_price >= position.take_profit
            )

        # If manual close only mode, only check hard stops
        if self.auto_close_disabled:
            # Still check stop loss for safety
            if stop_loss_hit:
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=position.symbol,
                    title=f"⚠️ Stop Loss Triggered",
                    message=f"Price hit stop loss at ${current_price:.2f}. Manual close required.",
                    importance=9
                ))
                # Don't auto-close, just warn
                return False, ""
            return False, ""
        # First check hard stops
        if stop_loss_hit:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=position.symbol,
                title=f"🛑 Stop Loss Hit",
                message=f"Stopped out at ${current_price:.2f}. Part of the game - on to the next one.",
                importance=9
            ))
            # Set 60-min cooldown on this symbol to prevent immediate re-entry
            if not hasattr(self, '_symbol_loss_cooldown'):
                self._symbol_loss_cooldown = {}
            from datetime import timedelta
            self._symbol_loss_cooldown[position.symbol] = datetime.now() + timedelta(minutes=60)
            return True, "stop_loss"

        # Hard take-profit check — fires on the next bar whose price
        # reaches the target, independent of indicators. Previous code
        # let the dynamic exit manager swallow this case and sometimes
        # held past the target.
        if take_profit_hit:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=position.symbol,
                title=f"🎯 Take-Profit Hit",
                message=(
                    f"Target reached at ${current_price:.2f} "
                    f"(target ${position.take_profit:.2f}). Closing."
                ),
                importance=9,
            ))
            return True, "take_profit"
        
        # Get current indicators for dynamic exit
        if self.data_provider:
            data = self.data_provider.get_market_data(position.symbol)
            if not data.empty:
                indicators = await self.technical_analyzer.analyze_with_commentary(data, position.symbol)
                
                # Use dynamic exit manager
                should_exit, reason, exit_portion = await self.exit_manager.evaluate_exit(
                    position, current_price, indicators
                )
                
                if should_exit:
                    if exit_portion < 1.0:
                        # Partial exit - reduce position
                        position.quantity = int(position.quantity * (1 - exit_portion))
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.DECISION,
                            symbol=position.symbol,
                            title=f"📊 Partial Exit",
                            message=f"Taking {exit_portion*100:.0f}% off the table. Letting the rest ride.",
                            importance=8
                        ))
                        if position.quantity > 0:
                            return False, ""  # Keep remaining position
                    
                    return True, reason
        
        return False, ""
    
    async def _close_position_with_commentary(self, position, reason: str):
        """Close position with detailed commentary"""
        # CRITICAL: Never auto-close external/manually managed positions (except for explicit manual_override)
        is_external = getattr(position, 'is_external', False)
        is_manually_managed = getattr(position, 'is_manually_managed', False)

        if (is_external or is_manually_managed) and reason != "manual_override":
            logger.warning(f"Blocked auto-close of EXTERNAL position {position.symbol}. Reason: {reason}")
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=position.symbol,
                title=f"🚫 External Position Protected",
                message=f"Cannot auto-close {position.symbol} - this is an EXTERNAL position.\n"
                       f"Current P&L: ${position.unrealized_pnl:.2f}\n"
                       f"Use Schwab directly or the manual close button.",
                importance=9
            ))
            return  # NEVER auto-close external positions

        # CRITICAL: Check if manual close only is enabled (except for manual_override)
        if self.auto_close_disabled and reason != "manual_override":
            self._audit("position_manager", position.symbol, "skip_close",
                        "auto_close_disabled",
                        requested_reason=reason,
                        pnl=round(float(getattr(position, "unrealized_pnl", 0)), 2))
            logger.warning(f"Attempted to auto-close {position.symbol} but manual_close_only is ON. Reason: {reason}")
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=position.symbol,
                title=f"⚠️ Auto-Close Blocked",
                message=f"Attempted to close {position.symbol} ({reason}) but manual close only is ON. "
                       f"Current P&L: ${position.unrealized_pnl:.2f}. Use manual close button.",
                importance=8
            ))
            return  # Don't close the position
        
        # Debug log
        logger.info(f"Attempting to close position: {position.symbol}, mode: {self.mode}")

        # v-close-path-robust-2026-04-29: snapshot which container the
        # position lives in BEFORE any intermediate work. Previously the
        # function inferred the container from self.mode at the very end,
        # so if (a) brain.remember_trade or _save_state raised mid-flow,
        # or (b) self.mode was toggled during an await, the del at the
        # bottom never fired and the position became "undead" — proactive
        # exits kept screaming "EXIT" while the position bled out (NVDA
        # 7× over 80 min on 04-29 lost ~$110 this way).
        if position.symbol in self.simulated_positions and \
           self.simulated_positions.get(position.symbol) is position:
            _close_container = self.simulated_positions
        elif position.symbol in self.positions and \
             self.positions.get(position.symbol) is position:
            _close_container = self.positions
        else:
            # Position already removed by another path; nothing to do
            logger.info(f"close_position: {position.symbol} not in any container — already closed")
            return
        # ADD CONFIRMATION HERE
        # Check if confirmation is needed
        if self.mode == TradingMode.LIVE and self.require_confirmations:
            # Calculate P&L percentage
            pnl = position.unrealized_pnl
            pnl_percent = abs((pnl / (position.entry_price * position.quantity)) * 100)
            
            # Determine if confirmation is needed based on settings
            needs_confirmation = True
            
            # Skip confirmation for profits if only confirming losses
            if self.confirm_only_losses and pnl > 0:
                needs_confirmation = False
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=position.symbol,
                    title=f"✅ Auto-closing Profit",
                    message=f"Confirmations only required for losses. Closing profitable position.",
                    importance=6
                ))
            
            # Skip confirmation if below threshold
            elif self.confirm_threshold_percent > 0 and pnl_percent < self.confirm_threshold_percent:
                needs_confirmation = False
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=position.symbol,
                    title=f"✅ Auto-closing Small Position",
                    message=f"P&L below {self.confirm_threshold_percent}% threshold. No confirmation needed.",
                    importance=6
                ))
            
            # Request confirmation if needed
            if needs_confirmation:
                if not await self._get_close_confirmation(position, reason):
                    # v-exiting-revert-on-deny-2026-06-11 (review
                    # finding on the armed exit ladder): the call site
                    # already transitioned the position to EXITING.
                    # Returning here used to strand it there — the FSM
                    # allows no exit rules in EXITING, so the 60s
                    # watchdog promoted it to ZOMBIE and management
                    # stopped. Revert to the ladder state implied by
                    # the position's own flags so the FSM resumes.
                    self._ensure_position_lock(position)
                    async with position._state_lock:
                        if position.state == PositionState.EXITING.value:
                            if getattr(position, 'trailing_stop', None):
                                _revert = PositionState.TRAILING
                            elif getattr(position, 'scaled_out', False):
                                _revert = PositionState.AT_1R
                            elif getattr(position, 'breakeven_lifted', False):
                                _revert = PositionState.AT_BREAKEVEN
                            else:
                                _revert = PositionState.LIVE
                            position.state = _revert.value
                            position.state_reason = "confirmation_denied"
                            self._audit(
                                "position_manager", position.symbol,
                                "state_reverted", "confirmation_denied",
                                reverted_to=_revert.value,
                                requested_reason=reason,
                            )
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.DECISION,
                        symbol=position.symbol,
                        title=f"🚫 Close Cancelled",
                        message=f"User cancelled position close for {position.symbol}",
                        importance=7
                    ))
                    return
        
        # Get the most current price for accurate P&L calculation.
        # v-close-quote-async-2026-04-30: data_provider.get_quote is a
        # blocking network call. On 04-30 09:15 it stalled mid-close on
        # FCEL and froze the entire asyncio event loop — every dashboard
        # WebSocket, every API call, every other position check went silent
        # for ~60 sec until the bot was killed manually. The accept queue
        # backed up to 9 pending connections.
        # Fix: run the sync quote call in an executor with a hard 3s timeout.
        # On timeout, fall back to position.current_price (which we already
        # have from the prior tick — at most 3-5 min stale, still vastly
        # better than freezing the bot).
        exit_price = position.current_price
        if self.data_provider:
            try:
                loop = asyncio.get_event_loop()
                quote = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, self.data_provider.get_quote, position.symbol
                    ),
                    timeout=3.0,
                )
                if quote and 'last' in quote:
                    exit_price = quote['last']
            except asyncio.TimeoutError:
                logger.warning(
                    f"close_position: quote timeout for {position.symbol} — "
                    f"using last known price ${exit_price:.2f}"
                )
            except Exception as e:
                logger.debug(f"Could not get current quote for {position.symbol}: {e}")
        
        # Calculate P&L using the most current price
        if position.side == 'short':
            pnl = (position.entry_price - exit_price) * position.quantity
        else:  # long
            pnl = (exit_price - position.entry_price) * position.quantity
        
        roi = (pnl / (position.entry_price * position.quantity)) * 100
        
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.DECISION,
            symbol=position.symbol,
            title=f"{'💰' if pnl > 0 else '💸'} Position Closed: {position.symbol}",
            message=f"Closing position with {'profit' if pnl > 0 else 'loss'} of ${abs(pnl):.2f} ({abs(roi):.1f}%)",
            data={
                'entry_price': position.entry_price,
                'exit_price': exit_price,
                'quantity': position.quantity,
            'pnl': pnl,
                'roi_percentage': roi,
                'exit_reason': reason,
                'holding_period': (datetime.now() - position.entry_time).total_seconds() / 3600
            },
            importance=9
        ))
        
        # v-close-path-robust-2026-04-29: brain learning is best-effort.
        # If embedding/lesson generation fails, the close must still proceed.
        try:
            memory = self.brain.remember_trade(
                symbol=position.symbol,
                pattern=position.reasoning.get('strategy', 'unknown'),
                outcome='win' if pnl > 0 else 'loss',
                pnl_percent=roi,
                context={
                    'exit_reason': reason,
                    'holding_time': (datetime.now() - position.entry_time).total_seconds() / 60,
                    'max_profit': getattr(position, 'max_unrealized_pnl', pnl),
                    'market_conditions': self.market_state
                }
            )
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.PSYCHOLOGY,
                symbol=position.symbol,
                title=f"📚 Lesson Learned",
                message=memory.lesson,
                importance=7
            ))
        except Exception as exc:
            logger.warning(f"brain.remember_trade failed for {position.symbol}: {exc}")
        
        # Track strategy performance
        strategy = position.reasoning.get('strategy', 'unknown')
        if not hasattr(self, 'strategy_performance'):
            self.strategy_performance = {}

        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {'wins': 0, 'losses': 0, 'total': 0}

        if pnl > 0:
            self.strategy_performance[strategy]['wins'] += 1
        else:
            self.strategy_performance[strategy]['losses'] += 1
        self.strategy_performance[strategy]['total'] += 1

        # Disable underperforming strategies
        if self.strategy_performance[strategy]['total'] > 20:
            win_rate = self.strategy_performance[strategy]['wins'] / self.strategy_performance[strategy]['total']
            if win_rate < 0.35:
                logger.warning(f"Disabling {strategy} - win rate {win_rate:.1%}")
        # v-short-exit-fix-2026-04-21: generic 5-min re-entry cooldown
        # on every exit. Caller-agnostic belt for the churn class of bug.
        # The 60-min loss cooldown at engine.py:3750 still applies on top
        # for stop-loss exits specifically.
        if not hasattr(self, '_symbol_reentry_cooldown'):
            self._symbol_reentry_cooldown = {}
        from datetime import timedelta as _td
        self._symbol_reentry_cooldown[position.symbol] = datetime.now() + _td(minutes=5)

        # Save state after each trade — best-effort; an I/O hiccup must
        # not strand the position. v-close-path-robust-2026-04-29.
        try:
            self._save_state()
        except Exception as exc:
            logger.warning(f"_save_state failed during close of {position.symbol}: {exc}")

        try:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.PSYCHOLOGY,
                symbol=position.symbol,
                title=f"📝 Post-Trade Analysis",
                message=self._generate_post_trade_analysis(position, reason, pnl),
                importance=6
            ))
        except Exception as exc:
            logger.debug(f"post-trade analysis failed for {position.symbol}: {exc}")
        
        # v-close-path-robust-2026-04-29: live broker submission only when
        # the position is in self.positions AND the engine is in LIVE mode.
        # A sim position holds no broker state, so calling Schwab on it
        # would fail and (under the old logic) prevent the del — the root
        # cause of the "undead position" bug.
        # v-position-fsm-2026-04-30 (Phase 1): mark intent-to-close on
        # the FSM. If a parent already transitioned the position to
        # EXITING (via try_transition at the rule's site), this is a
        # no-op idempotent call. If the close was reached via a path
        # that DIDN'T call try_transition (e.g. take_profit, stop_loss
        # in legacy code), this ensures the FSM is in EXITING before
        # any broker work begins.
        self._ensure_position_lock(position)
        await try_transition(
            position, PositionState.EXITING,
            f"close_invoked: {reason}",
            audit_fn=self._audit,
        )

        if self.mode == TradingMode.LIVE and _close_container is self.positions:
            success = await self._close_real_position(position)
            if not success:
                # Live broker close failed. The position is now in
                # EXITING but the broker still holds it. Per the
                # zombie-state contract, escalate so:
                #   - the engine stops applying any other exit rule
                #   - the dashboard surfaces it for operator action
                #   - the EXITING_ZOMBIE_SEC watchdog won't double-fire
                async with position._state_lock:
                    if position.state == PositionState.EXITING.value:
                        position.state = PositionState.ZOMBIE.value
                        position.zombie_reason = "broker_close_returned_false"
                        position.state_changed_at = datetime.now(timezone.utc).isoformat()
                self._audit(
                    "position_fsm", position.symbol, "promoted_to_zombie",
                    "broker_close_failed",
                )
                logger.warning(
                    f"Live close failed for {position.symbol} — promoted to ZOMBIE; "
                    f"operator must resolve from dashboard."
                )
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.WARNING,
                    symbol=position.symbol,
                    title=f"🧟 Broker Close Failed — ZOMBIE",
                    message=(
                        f"{position.symbol}: close order returned failure. "
                        "The position is in ZOMBIE state and bot will not "
                        "manage it until manually resolved."
                    ),
                    importance=10,
                ))
                return

        # v-consecutive-losses-live-only-2026-05-11: only count CLOSED
        # LIVE trades against the live-mode circuit breaker. Prior code
        # mixed sim and live losses, so a sim-mode losing streak (paper
        # money) could trip the live circuit breaker and pause real
        # trading. Operator observed `consecutive_losses=11` while the
        # brain's actual recent history was 14 wins / 11 losses with the
        # most recent being a WIN — the counter was wildly out of sync
        # with live reality because sim closes were polluting it.
        _is_live_close = (getattr(position, "mode", "live") == "live")
        if _is_live_close:
            if pnl < 0:
                # v-consec-loss-daily-reset-2026-05-21: stamp date for
                # the runtime daily-rollover check in risk/manager.py.
                self.risk_manager.consecutive_losses += 1
                self.risk_manager.last_loss_date = datetime.now().date()
            else:
                self.risk_manager.consecutive_losses = 0

        # FSM transition EXITING → CLOSED. After this the position is
        # popped from tracking; the state field is mostly diagnostic
        # but kept consistent for any audit replay.
        async with position._state_lock:
            if position.state == PositionState.EXITING.value:
                position.state = PositionState.CLOSED.value
                position.state_reason = f"closed: {reason}"
                position.state_changed_at = datetime.now(timezone.utc).isoformat()

        # Atomic removal — popped, not deleted-by-key, so a parallel close
        # attempt on the same symbol can't double-remove or KeyError.
        _close_container.pop(position.symbol, None)
        
        # Update trade history
        trade_record = {
            'symbol': position.symbol,
            'entry_time': position.entry_time.isoformat(),
            'exit_time': datetime.now().isoformat(),
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.quantity,
            'pnl': pnl,
            'reason': reason,  # Changed from exit_reason to match frontend
            'reasoning': position.reasoning
        }
        self.trade_history.append(trade_record)

        # Persist to Postgres for SQL queryability
        if self.db_logger is not None:
            pnl_pct = ((exit_price - position.entry_price) / position.entry_price * 100
                        if position.side == "long"
                        else (position.entry_price - exit_price) / position.entry_price * 100)
            try:
                asyncio.get_event_loop().create_task(
                    self.db_logger.log_trade(
                        symbol=position.symbol,
                        side=position.side or "long",
                        strategy=(position.reasoning or {}).get("strategy"),
                        entry_time=position.entry_time,
                        exit_time=datetime.now(),
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        quantity=position.quantity,
                        pnl=pnl,
                        pnl_pct=round(pnl_pct, 4),
                        exit_reason=reason,
                        atr_at_entry=(position.reasoning or {}).get("atr"),
                        stop_loss=position.stop_loss,
                        take_profit=position.take_profit,
                        # v-trade-record-ml-columns-2026-09-02: Position has
                        # no `confidence` attribute — the old getattr always
                        # wrote NULL. Pull confidence/meta_proba/kelly from
                        # reasoning, same as the external-close reconcile path.
                        confidence=(position.reasoning or {}).get("confidence"),
                        meta_proba=(position.reasoning or {}).get("meta_proba"),
                        kelly_fraction=(position.reasoning or {}).get("kelly_fraction"),
                        scaled_out=getattr(position, "scaled_out", False),
                        mode=self.mode.value,
                        reasoning=position.reasoning,
                    )
                )
            except Exception as exc:
                # v-trade-record-ml-columns-2026-09-02: was a bare pass —
                # a failed bot_trades write vanished without a trace.
                logger.warning(
                    "log_trade task creation failed for %s: %s",
                    position.symbol, exc,
                )

        # Broadcast trade update immediately if we have a connection manager
        if hasattr(self, 'connection_manager') and self.connection_manager:
            try:
                asyncio.create_task(self._broadcast_trade_update(trade_record))
            except Exception as e:
                logger.debug(f"Could not broadcast trade update: {e}")
    
    async def _get_close_confirmation(self, position, reason: str) -> bool:
        """Get user confirmation before closing position.
        
        v-autonomy-profile-2026-09-08: respects autonomy profile settings.
        - supervised profile: timeout defaults to deny (position stays open)
        - autonomous_live profile: timeout defaults to execute (close position)
        
        The CONFIRMATION_TIMEOUT_ACTION config controls this behavior.
        """
        # Initialize pending requests dict if it doesn't exist
        if not hasattr(self, 'pending_close_requests'):
            self.pending_close_requests = {}
        # Store pending close request
        close_request_id = f"close_{position.symbol}_{int(time.time())}"
        self.pending_close_requests = getattr(self, 'pending_close_requests', {})
        
        # v-autonomy-profile-2026-09-08: read timeout settings from config
        cfg = Config()
        timeout = cfg.CONFIRMATION_TIMEOUT_SEC
        timeout_action = cfg.CONFIRMATION_TIMEOUT_ACTION
        profile = cfg.TRADING_PROFILE
        
        # Create confirmation request
        self.pending_close_requests[close_request_id] = {
            'position': position,
            'reason': reason,
            'timestamp': datetime.now(),
            'confirmed': None
        }
        
        # Send confirmation request to UI
        pnl = position.unrealized_pnl
        roi = (pnl / (position.entry_price * position.quantity)) * 100
        # Log that we're requesting confirmation
        logger.info(
            "close_confirmation_request symbol=%s reason=%s profile=%s timeout_sec=%.0f timeout_action=%s",
            position.symbol, reason, profile, timeout, timeout_action,
        )
        # Send confirmation request to UI
        try:
            await self._broadcast_ui({
                'type': 'close_confirmation_request',
                'data': {
                    'request_id': close_request_id,
                    'symbol': position.symbol,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'pnl': pnl,
                    'roi': roi,
                    'reason': reason,
                    'profile': profile,
                    'timeout_sec': timeout,
                    'timeout_action': timeout_action,
                    'message': f"Close {position.symbol} with {'profit' if pnl > 0 else 'loss'} of ${abs(pnl):.2f} ({abs(roi):.1f}%)?"
                }
            })
        except Exception as e:
            logger.error(f"Error broadcasting close confirmation: {e}")
            # v-autonomy-profile-2026-09-08: on broadcast failure, follow
            # the profile's default behavior. autonomous_live should still
            # execute risk exits (stops) even if UI is unreachable.
            if timeout_action == 'execute':
                logger.warning(
                    "close_confirmation broadcast failed, fail-open executing close for %s",
                    position.symbol,
                )
                return True
            return False
        
        # Wait for user response (with timeout)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if close_request_id in self.pending_close_requests:
                if self.pending_close_requests[close_request_id]['confirmed'] is not None:
                    confirmed = self.pending_close_requests[close_request_id]['confirmed']
                    del self.pending_close_requests[close_request_id]
                    logger.info(
                        "close_confirmation_response symbol=%s confirmed=%s profile=%s",
                        position.symbol, confirmed, profile,
                    )
                    return confirmed
            await asyncio.sleep(0.1)
        
        # v-autonomy-profile-2026-09-08: timeout behavior depends on profile
        should_execute = (timeout_action == 'execute')
        
        if should_execute:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=position.symbol,
                title=f"⏱️ Close Confirmation Timeout — Executing (Fail-Open)",
                message=(
                    f"No response received for closing {position.symbol}. "
                    f"Profile '{profile}' with timeout_action='{timeout_action}' "
                    f"— executing close to protect against unattended risk."
                ),
                data={
                    'profile': profile,
                    'timeout_action': timeout_action,
                    'confirmation_timeout_action': 'execute',
                },
                importance=9
            ))
            logger.warning(
                "close_confirmation_timeout symbol=%s profile=%s action=execute reason=%s",
                position.symbol, profile, reason,
            )
        else:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=position.symbol,
                title=f"⏱️ Close Confirmation Timeout",
                message=f"No response received for closing {position.symbol}. Position remains open.",
                data={
                    'profile': profile,
                    'timeout_action': timeout_action,
                    'confirmation_timeout_action': 'deny',
                },
                importance=8
            ))
            logger.info(
                "close_confirmation_timeout symbol=%s profile=%s action=deny reason=%s",
                position.symbol, profile, reason,
            )
        
        if close_request_id in self.pending_close_requests:
            del self.pending_close_requests[close_request_id]
        
        return should_execute
    def _generate_post_trade_analysis(self, position, exit_reason: str, pnl: float) -> str:
        """Generate insightful post-trade analysis"""
        analysis = []
        
        if pnl > 0:
            analysis.append("✅ What went right:")
            analysis.append(f"- The {position.reasoning.get('strategy', 'strategy')} correctly identified the opportunity")
            analysis.append(f"- Entry timing was good at ${position.entry_price:.2f}")
            if exit_reason == "take_profit":
                analysis.append("- The trade reached its full potential as planned")
        else:
            analysis.append("❌ What could be improved:")
            if exit_reason == "stop_loss":
                analysis.append("- The market moved against our analysis")
                analysis.append("- Consider if the stop was too tight or entry was premature")
            analysis.append("- Review if market conditions changed after entry")
        
        analysis.append("\n💡 Key lessons:")
        analysis.append("- Risk management worked as designed")
        analysis.append(f"- The {abs((position.current_price - position.entry_price) / position.entry_price * 100):.1f}% move took {(datetime.now() - position.entry_time).total_seconds() / 3600:.1f} hours")
        
        return "\n".join(analysis)
    
    @staticmethod
    def _compute_position_day_pnl(position: Dict[str, Any]) -> float:
        """v-day-pnl-intraday-entry-2026-05-12: see ``core.pnl`` docstring.
        Thin wrapper that routes outlier events through the engine's
        WARNING logger so the operator sees the PLTR-style fallback
        on the dashboard."""
        from core.pnl import compute_position_day_pnl

        def _on_outlier(symbol: str, broker: float, mv: float, fallback: float) -> None:
            logger.warning(
                "day_pnl: %s currentDayProfitLoss=%.2f exceeds 50%% of "
                "market_value=%.2f — falling back to netChange*qty=%.2f "
                "(PLTR-style outlier, see v-day-pnl-intraday-entry).",
                symbol, broker, mv, fallback,
            )

        return compute_position_day_pnl(position, on_outlier=_on_outlier)

    def _compute_bot_daily_pnl(self) -> float:
        """v-bot-only-pnl-circuit-2026-06-08: realized + unrealized
        P&L of BOT-managed positions only.

        Used by `risk/manager.py:can_trade` when
        `Config.ENABLE_BOT_ONLY_PNL_CIRCUIT` is True. Excludes
        external holdings (HQGE/PINS/COIN etc.) so the daily-loss
        circuit doesn't pause the bot for losses on positions the
        operator opened outside the bot.

        Components:
          * realized — sum of `pnl` on trade_history entries closed
            today AND flagged managed_by_bot (default True for entries
            the bot created via _execute_real_trade or reconciled via
            _reconcile_external_closes with bot-tagged context).
          * unrealized — sum of `unrealized_pnl` on currently-open
            `self.positions` where `position.managed_by_bot` is True.

        Robustness: every entry/position is wrapped in a per-item
        try/except. A single malformed record (missing field, bad
        datetime) is skipped, not raised — the circuit cannot block
        on a data hygiene issue. Returns 0.0 if everything fails.

        Pure local-state read; no network calls. Safe to invoke from
        the analysis loop on every cycle.
        """
        today = datetime.now().date()

        realized = 0.0
        try:
            for trade in self.trade_history:
                try:
                    exit_time = trade.get('exit_time')
                    if not exit_time:
                        continue
                    if isinstance(exit_time, str):
                        try:
                            exit_dt = datetime.fromisoformat(exit_time)
                        except (TypeError, ValueError):
                            continue
                    else:
                        exit_dt = exit_time
                    if exit_dt.date() != today:
                        continue
                    # Default True: trade_history entries that predate
                    # the managed_by_bot field are assumed to be bot
                    # trades (the field was added only for distinguishing
                    # externally-discovered closes from bot-driven ones).
                    if not trade.get('managed_by_bot', True):
                        continue
                    pnl_val = trade.get('pnl') or trade.get('realized_pnl') or 0
                    realized += float(pnl_val)
                except Exception:
                    continue
        except Exception:
            pass

        unrealized = 0.0
        try:
            for symbol, position in self.positions.items():
                try:
                    if not getattr(position, 'managed_by_bot', False):
                        continue
                    unrealized += float(getattr(position, 'unrealized_pnl', 0) or 0)
                except Exception:
                    continue
        except Exception:
            pass

        return realized + unrealized

    async def _get_real_account_info(self) -> Dict[str, float]:
        """Get real account information directly from Schwab - NO CALCULATIONS"""
        if not self.schwab_client or not self.account_id:
            return {}
    
        try:
            # Get account with positions to ensure we have all data
            from schwab.client import Client
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.schwab_client.get_account(
                        self.account_hash,
                        fields=[Client.Account.Fields.POSITIONS],
                    ),
                ),
                timeout=6.0,
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to get account info from Schwab. Status: {response.status_code}")
                return {}
            
            data = response.json()
            account = data.get('securitiesAccount', {})
            current_balances = account.get('currentBalances', {})
            
            # ONLY use Schwab's reported values - NO CALCULATIONS
            current_value = current_balances.get('liquidationValue', 0)
            current_cash = current_balances.get('cashBalance', 0)
            buying_power = current_balances.get('buyingPower', 0)
            
            # v-schwab-pnl-fix-2026-04-21: original switched to
            # netChange × quantity because currentDayProfitLoss was
            # unreliable on same-day-lot-transferred positions (PLTR
            # +$21,932 on -$3,393 position).
            # v-day-pnl-intraday-entry-2026-05-12: that switch broke
            # day P&L for intraday entries — netChange is the move
            # from prev close, not from entry price, so positions
            # opened intraday at entry ≠ today's open had hundreds of
            # dollars of pre-entry move attributed to them. New
            # approach: trust currentDayProfitLoss UNLESS its
            # magnitude is implausible vs market value (PLTR-style
            # outlier), in which case fall back to netChange × qty.
            day_pnl = 0
            if 'positions' in account:
                for position in account.get('positions', []):
                    day_pnl += self._compute_position_day_pnl(position)
            
            logger.debug(f"Schwab Direct Values - Balance: ${current_value:.2f}, P&L: ${day_pnl:.2f}, Cash: ${current_cash:.2f}")
            
            return {
                'balance': current_value,
                'buying_power': buying_power,
                'day_trades_remaining': account.get('roundTrips', 3),
                'cash': current_cash,
                'day_pnl': day_pnl  # Direct from Schwab positions
            }
        except Exception as e:
            logger.error(f"Account info error: {e}")
    
        return {}

    def _is_market_open(self) -> bool:
        """Check if market is open"""
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        
        # Pre-market: 4:00 AM - 9:30 AM EST
        pre_market_start = now.replace(hour=4, minute=0, second=0)
    
        # After-hours end: 8:00 PM EST
        after_hours_end = now.replace(hour=20, minute=0, second=0)
    
        return pre_market_start <= now < after_hours_end
    
    def _get_next_market_open(self) -> str:
        """Get next market open time"""
        now = datetime.now()
        
        if now.weekday() < 5 and now.hour < 9 or (now.hour == 9 and now.minute < 30):
            next_open = now.replace(hour=9, minute=30, second=0)
        elif now.weekday() >= 4:
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            next_open = now + timedelta(days=days_until_monday)
            next_open = next_open.replace(hour=9, minute=30, second=0)
        else:
            next_open = now + timedelta(days=1)
            next_open = next_open.replace(hour=9, minute=30, second=0)
        
        return next_open.strftime("%Y-%m-%d %H:%M:%S")

    # Add these methods inside the TradingEngineWithCommentary class:

    async def get_schwab_positions(self) -> List[Dict]:
        """Get real positions from Schwab account"""
        if not self.schwab_client or not self.account_id:
            return []
        
        try:
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.schwab_client.get_account(
                        self.account_hash,
                        fields=[self.schwab_client.Account.Fields.POSITIONS],
                    ),
                ),
                timeout=6.0,
            )
            
            if response.status_code == 200:
                data = response.json()
                account_data = data.get('securitiesAccount', {})
                positions_data = account_data.get('positions', [])
                
                schwab_positions = []
                
                for pos in positions_data:
                    instrument = pos.get('instrument', {})
                    symbol = instrument.get('symbol', 'UNKNOWN')
                    
                    # Extract position details
                    long_quantity = pos.get('longQuantity', 0)
                    short_quantity = pos.get('shortQuantity', 0)
                    quantity = long_quantity - short_quantity
                    
                    if quantity == 0:
                        continue
                    
                    # Get prices
                    average_price = pos.get('averagePrice', 0)
                    market_value = pos.get('marketValue', 0)

                    # Calculate current price
                    current_price = market_value / quantity if quantity != 0 else 0

                    # v-schwab-pnl-fix-2026-04-21: Schwab payload field names
                    # differ from what the old code assumed.
                    #   unrealizedProfitLoss   — NOT present in responses. The
                    #       old .get(default=0) silently returned 0, which is
                    #       why every position showed total_pnl=0.
                    #   longOpenProfitLoss /   — Schwab's authoritative total
                    #   shortOpenProfitLoss      unrealized P&L per position.
                    #   currentDayProfitLoss   — unreliable. Same-day lot
                    #       transfers can inflate this dramatically (e.g.
                    #       observed PLTR value $21,932 on a -$3,393 position,
                    #       paired with currentDayCost -$21,823 so the account
                    #       total nets out — but the per-position number is
                    #       broken for our purposes).
                    #   netChange              — per-share $ move today,
                    #       reliable for any open position.
                    if quantity > 0:
                        total_pnl = pos.get('longOpenProfitLoss', 0)
                    else:
                        total_pnl = pos.get('shortOpenProfitLoss', 0)

                    net_change = (instrument or {}).get('netChange', 0)
                    if net_change:
                        day_pnl = net_change * quantity
                    else:
                        day_pnl = pos.get('currentDayProfitLoss', 0)

                    # Calculate percentages
                    pnl_percent = (total_pnl / (average_price * abs(quantity))) * 100 if average_price > 0 and quantity != 0 else 0
                    
                    position_info = {
                        'symbol': symbol,
                        'quantity': quantity,
                        'side': 'long' if quantity > 0 else 'short',
                        'average_price': average_price,
                        'current_price': current_price,
                        'market_value': market_value,
                        'day_pnl': day_pnl,
                        'total_pnl': total_pnl,
                        'pnl_percent': pnl_percent,
                        'asset_type': instrument.get('assetType', 'EQUITY')
                    }
                    
                    schwab_positions.append(position_info)
                    
                    # Add commentary about the position
                    # self.commentary.add_commentary(TradingCommentary(
                    #     timestamp=datetime.now(),
                    #     type=CommentaryType.MARKET_ANALYSIS,
                    #     symbol=symbol,
                    #     title=f"📊 Existing Position: {symbol}",
                    #     message=f"Found {abs(quantity)} shares {'long' if quantity > 0 else 'short'}\n"
                    #         f"Avg Price: ${average_price:.2f}, Current: ${current_price:.2f}\n"
                    #         f"P&L: ${total_pnl:.2f} ({pnl_percent:.1f}%)",
                    #     data=position_info,
                    #     importance=6
                    # ))
                
                return schwab_positions
                
        except Exception as e:
            logger.error(f"Error fetching Schwab positions: {e}")
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=None,
                title="⚠️ Error Fetching Positions",
                message=f"Could not retrieve positions: {str(e)}",
                importance=7
            ))
        
        return []

    async def _update_and_track_real_positions(self):
        """Update and track real positions from Schwab account.

        v-track-live-in-sim-2026-04-30: previously returned early unless
        mode==LIVE. That meant SIM-mode dashboards never saw the user's
        actual Schwab holdings — only the in-bot simulated trades. The
        user wants their real holdings visible alongside sim trades for
        a complete account picture, so the LIVE-mode gate is removed.
        Discovered Schwab positions are tagged managed_by_bot=False so
        the engine doesn't try to manage them with sim-mode logic.
        """
        if not self.schwab_client:
            return
        
        try:
            schwab_positions = await self.get_schwab_positions()
            
            # Update existing tracked positions
            for pos_data in schwab_positions:
                symbol = pos_data['symbol']
                
                # Check if we're already tracking this position
                if symbol in self.positions:
                    # Update the tracked position with latest data
                    position = self.positions[symbol]
                    position.current_price = pos_data['current_price']
                    position.unrealized_pnl = pos_data['total_pnl']
                else:
                    # v-discovered-position-tagging-2026-05-11: this
                    # sibling discovery path was creating external
                    # positions with `entry_time=now` and NO
                    # `is_external=True`. Result: the T3 health gate
                    # (which flags "recent bot-opened positions still
                    # unmanaged" as a regression canary) was tripping
                    # on every discovered Schwab position and blocking
                    # all live entries. The sibling `_update_real_positions`
                    # at line 1848 already does this correctly — mirror
                    # that behavior here. Back-date entry_time so the
                    # health gate's 5-minute "recent" filter skips it.
                    #
                    # v-discovered-claim-bot-orders-2026-05-11: if this
                    # discovered symbol matches a recently-placed bot
                    # order still in pending_orders, claim it as
                    # managed_by_bot=True and use the signal's stop/
                    # target. Solves the case where `_verify_order_fill`
                    # timed out (causing the immediate-path Position
                    # creation to never run), but the order did fill
                    # downstream and now shows up on Schwab. Without
                    # this rescue, the bot opens trades it can't manage.
                    _matching_order_id = None
                    _matching_signal = None
                    for _oid, _od in list(self.pending_orders.items()):
                        try:
                            if _od.get('symbol') == symbol:
                                _matching_order_id = _oid
                                _matching_signal = _od.get('signal')
                                break
                        except Exception:
                            continue

                    if _matching_signal is not None:
                        # Bot-opened — claim as managed.
                        position = Position(
                            symbol=symbol,
                            entry_price=pos_data['average_price'],
                            current_price=pos_data['current_price'],
                            quantity=abs(pos_data['quantity']),
                            side='long' if pos_data['quantity'] > 0 else 'short',
                            stop_loss=float(getattr(_matching_signal, 'stop_loss', 0) or 0),
                            take_profit=float(getattr(_matching_signal, 'take_profit', 0) or 0),
                            entry_time=datetime.now(),
                            unrealized_pnl=pos_data['total_pnl'],
                            reasoning=(getattr(_matching_signal, 'reasoning', None) or {
                                'source': 'rescued_from_pending', 'order_id': _matching_order_id,
                            }),
                            mode="live",
                            managed_by_bot=True,
                        )
                        # NOT external; bot owns it.
                        self.positions[symbol] = position
                        # Initialize exit tracking if we have an exit_manager
                        if hasattr(self, 'exit_manager') and self.exit_manager is not None:
                            try:
                                self.exit_manager.initialize_position_tracking(
                                    symbol,
                                    position.entry_price,
                                    position.stop_loss,
                                    position.take_profit,
                                )
                            except Exception as _ex:
                                logger.warning(
                                    "exit_manager init failed in rescue path for %s: %s",
                                    symbol, _ex,
                                )
                        # Pop the rescued order so the sweep doesn't double-process
                        self.pending_orders.pop(_matching_order_id, None)
                        self.order_id_to_symbol.pop(_matching_order_id, None)
                        if hasattr(self, '_audit'):
                            self._audit(
                                "discovery_rescue", symbol, "position_created",
                                "claimed_pending_order",
                                order_id=_matching_order_id,
                                fill_price=round(float(pos_data['average_price'] or 0), 4),
                                qty=int(abs(pos_data['quantity'])),
                                stop=round(float(position.stop_loss or 0), 4),
                                target=round(float(position.take_profit or 0), 4),
                                side=position.side,
                            )
                        logger.info(
                            "discovery_rescue: %s %s qty=%d entry=%.4f stop=%.4f target=%.4f — claimed as managed_by_bot=True from pending order %s",
                            symbol, position.side, int(abs(pos_data['quantity'])),
                            float(pos_data['average_price'] or 0),
                            float(position.stop_loss or 0),
                            float(position.take_profit or 0),
                            _matching_order_id,
                        )
                    else:
                        # Truly external — pre-existing or operator-opened.
                        position = Position(
                            symbol=symbol,
                            entry_price=pos_data['average_price'],
                            current_price=pos_data['current_price'],
                            quantity=pos_data['quantity'],
                            side='long' if pos_data['quantity'] > 0 else 'short',
                            stop_loss=pos_data['average_price'] * (1 - Config().DEFAULT_STOP_LOSS_PCT),
                            take_profit=pos_data['average_price'] * (1 + Config().DEFAULT_TAKE_PROFIT_PCT),
                            entry_time=datetime.now() - timedelta(hours=1),
                            unrealized_pnl=pos_data['total_pnl'],
                            reasoning={'source': 'existing_position', 'tracked_from': datetime.now().isoformat()},
                            mode="live",
                            managed_by_bot=False,
                        )
                        position.is_external = True
                        position.is_manually_managed = True
                        self.positions[symbol] = position
                    
                    # Initialize exit tracking for existing positions
                    if hasattr(self, 'exit_manager'):
                        self.exit_manager.initialize_position_tracking(
                            symbol,
                            position.entry_price,
                            position.stop_loss,
                            position.take_profit
                        )
                    
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.MARKET_ANALYSIS,
                        symbol=symbol,
                        title=f"📌 Tracking Existing Position",
                        message=f"Now tracking {symbol} position from your account",
                        data={
                            'shares': position.quantity,
                            'avg_price': position.entry_price,
                            'current_pnl': position.unrealized_pnl
                        },
                        importance=7
                    ))
                    
        except Exception as e:
            logger.error(f"Error updating real positions: {e}")
    
    def _is_news_blackout(self) -> bool:
        """Check if we're in news blackout period.
        
        v-econ-calendar-2026-09-09: delegates to EconCalendarProvider.
        Provider selection (in order):
          1. API provider (if ECON_CALENDAR_API_URL env var is set)
          2. Config provider (if trading.econ_calendar_events in yaml)
          3. Static provider (hardcoded CPI/Jobs/FOMC/Fed times)
        
        To customize blackout windows without code changes, add events
        to Config.yaml under trading.econ_calendar_events.
        """
        from core.econ_calendar import get_econ_calendar
        return get_econ_calendar().is_blackout()
