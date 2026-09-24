import os
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class ConfigManager:
    """Manages configuration loading and validation"""

    def __init__(self, config_path: str = "Config().yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> dict:
        """Load configuration from YAML file or create default"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            config = self._get_default_config()
            self._save_config(config)
            return config

    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'schwab': {
                'callback_url': "https://127.0.0.1",
                'token_path': "token_1.json"
            },
            'trading': {
                'ml_prediction_enabled': True,
                'max_risk_per_trade': 0.02,
                'min_risk_reward_ratio': 2.0,
                'max_daily_loss': 0.05,
                'max_consecutive_losses': 3,
                'extended_hours': {
                    'allow_premarket': False,  # DANGER: Wide spreads, low liquidity
                    'allow_afterhours': False,  # DANGER: Wide spreads, low liquidity
                    'use_limit_orders': True,  # Always use limit orders in extended hours
                    'premarket_start': "04:00",  # 4:00 AM ET
                    'market_open': "09:30",      # 9:30 AM ET
                    'market_close': "16:00",      # 4:00 PM ET
                    'afterhours_end': "20:00"    # 8:00 PM ET
                },
                'position_size_kelly_fraction': 0.25,
                'max_positions': 5,
                'reserve_cash_percent': 0.1,
                'min_position_size': 1,
                'max_position_value': 10000,
                'min_buying_power': 100,
                'limit_order_slippage': 0.001,
                'default_stop_loss_pct': 0.05,
                'default_take_profit_pct': 0.10,
                'max_position_value_pct': 0.25
            },
            'commentary': {
                'enabled': True,
                'detail_level': "verbose",
                'max_history': 100
            },
            'technical_analysis': {
                'timeframes': ['1min', '5min', '15min', '1hour', '1day'],
                'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786]
            },
            'order_management': {
                'auto_cancel_existing_orders': True,
                'require_close_confirmation': True,
                'confirm_only_losses': False,
                'confirm_threshold_percent': 5
            },
            'paths': {
                'trade_journal': "trade_journal.json",
                'model': "trading_model.pkl",
                'log': "trading_bot.log",
                'commentary_log': "trading_commentary.json",
                'backtest_results': "backtest_results.json"
            },
            'backtesting': {
                'enabled': True,
                'start_date': "2023-01-01",
                'end_date': "2024-01-01",
                'initial_capital': 100000,
                'commission': 0.005,
                'slippage': 0.001
            },
            'performance_metrics': {
                'calculate_sharpe': True,
                'calculate_sortino': True,
                'calculate_max_drawdown': True,
                'risk_free_rate': 0.02
            },
            'error_recovery': {
                'max_retries': 3,
                'retry_delay': 1.0,
                'circuit_breaker_enabled': True,
                'max_daily_loss_threshold': 0.10,
                'emergency_stop_loss': 0.15
            }
        }

    def _save_config(self, config: dict):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def _validate_config(self):
        """Validate configuration values"""
        required_sections = ['schwab', 'trading', 'commentary', 'technical_analysis']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate trading parameters are within safe ranges
        trading = self.config.get('trading', {})
        validations = [
            ('max_risk_per_trade', 0.001, 0.10, 'Max risk per trade must be 0.1%-10%'),
            ('min_risk_reward_ratio', 0.5, 10.0, 'Risk/reward ratio must be 0.5-10.0'),
            ('max_daily_loss', 0.01, 0.20, 'Max daily loss must be 1%-20%'),
            ('max_positions', 1, 50, 'Max positions must be 1-50'),
            ('reserve_cash_percent', 0.0, 0.50, 'Reserve cash must be 0%-50%'),
        ]
        for key, min_val, max_val, msg in validations:
            val = trading.get(key)
            if val is not None and not (min_val <= val <= max_val):
                raise ValueError(f"Config validation error: {msg} (got {val})")

    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def update(self, key: str, value):
        """Update configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self._save_config(self.config)

# Initialize configuration
config_manager = ConfigManager()

# ============================================================================
# TRADING LOSS CIRCUIT BREAKER
# ============================================================================
# Note: This is distinct from circuit_breaker.py which handles API failures.
# This class monitors daily P&L and halts trading on excessive losses.

class TradingLossBreaker:
    """Circuit breaker that halts trading when daily losses exceed thresholds"""

    def __init__(self, max_daily_loss: float = 0.10, emergency_stop: float = 0.15):
        self.max_daily_loss = max_daily_loss
        self.emergency_stop = emergency_stop
        self.daily_pnl = 0.0
        self.is_tripped = False
        self.trip_time = None
        self.reset_time = None

    def update_daily_pnl(self, pnl: float, account_balance: float = 100000):
        """Update daily P&L and check circuit breaker"""
        self.daily_pnl = pnl

        if self.daily_pnl >= 0:
            return

        daily_loss_ratio = abs(self.daily_pnl) / account_balance

        if daily_loss_ratio >= self.emergency_stop:
            self._trip("EMERGENCY_STOP", daily_loss_ratio)
        elif daily_loss_ratio >= self.max_daily_loss:
            self._trip("MAX_DAILY_LOSS", daily_loss_ratio)

    def _trip(self, reason: str, loss_ratio: float):
        """Trip the circuit breaker"""
        self.is_tripped = True
        self.trip_time = datetime.now()
        self.reset_time = self.trip_time + timedelta(hours=24)
        logger.critical(f"Trading loss breaker tripped: {reason} - Loss ratio: {loss_ratio:.2%}")

    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        if not self.is_tripped:
            return True
        if datetime.now() >= self.reset_time:
            self.reset()
            return True
        return False

    def reset(self):
        """Reset the circuit breaker"""
        self.is_tripped = False
        self.trip_time = None
        self.reset_time = None
        logger.info("Trading loss breaker reset")

# Backward compatibility alias
CircuitBreaker = TradingLossBreaker

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

class Config:
    """Central configuration for the trading bot"""
    # Use config manager instead of hardcoded values
    def __init__(self):
        self.manager = config_manager

    @property
    def SCHWAB_API_KEY(self):
        return os.getenv("SCHWAB_API_KEY")

    @property
    def SCHWAB_APP_SECRET(self):
        # Support both SCHWAB_SECRET and SCHWAB_APP_SECRET for compatibility
        return os.getenv("SCHWAB_SECRET") or os.getenv("SCHWAB_APP_SECRET")

    @property
    def SCHWAB_CALLBACK_URL(self):
        return self.manager.get('schwab.callback_url', 'https://127.0.0.1')

    @property
    def SCHWAB_TOKEN_PATH(self):
        return Path(self.manager.get('schwab.token_path', 'token_1.json'))

    @property
    def SCHWAB_ACCOUNT_NUMBER(self):
        return os.getenv("SCHWAB_ACCOUNT_NUMBER")

    @property
    def MAX_RISK_PER_TRADE(self):
        return self.manager.get('trading.max_risk_per_trade', 0.02)

    @property
    def MIN_RISK_REWARD_RATIO(self):
        return self.manager.get('trading.min_risk_reward_ratio', 2.0)

    @property
    def MAX_DAILY_LOSS(self):
        return self.manager.get('trading.max_daily_loss', 0.05)

    @property
    def MAX_CONSECUTIVE_LOSSES(self):
        return self.manager.get('trading.max_consecutive_losses', 3)

    @property
    def POSITION_SIZE_KELLY_FRACTION(self):
        return self.manager.get('trading.position_size_kelly_fraction', 0.25)

    @property
    def COMMENTARY_ENABLED(self):
        return self.manager.get('commentary.enabled', True)

    @property
    def COMMENTARY_DETAIL_LEVEL(self):
        return self.manager.get('commentary.detail_level', 'verbose')

    @property
    def MAX_COMMENTARY_HISTORY(self):
        return self.manager.get('commentary.max_history', 100)

    @property
    def TIMEFRAMES(self):
        return self.manager.get('technical_analysis.timeframes')

    @property
    def FIBONACCI_LEVELS(self):
        return self.manager.get('technical_analysis.fibonacci_levels')

    @property
    def TRADE_JOURNAL_PATH(self):
        return Path(self.manager.get('paths.trade_journal'))

    @property
    def MODEL_PATH(self):
        return Path(self.manager.get('paths.model'))

    @property
    def LOG_PATH(self):
        return Path(self.manager.get('paths.log'))

    @property
    def COMMENTARY_LOG_PATH(self):
        return Path(self.manager.get('paths.commentary_log'))

    @property
    def AUTO_CANCEL_EXISTING_ORDERS(self):
        return self.manager.get('order_management.auto_cancel_existing_orders')

    @property
    def MAX_POSITIONS(self):
        # v-max-positions-default-2026-05-06: 5 → 10. Hard cap on
        # concurrent bot-managed trades. Positions that are hands-off
        # (managed_by_bot=False, e.g. pre-existing Schwab holdings) do
        # NOT count against this limit.
        return self.manager.get('trading.max_positions', 10)

    @property
    def RESERVE_CASH_PERCENT(self):
        return self.manager.get('trading.reserve_cash_percent', 0.10)

    @property
    def MIN_POSITION_SIZE(self):
        return self.manager.get('trading.min_position_size', 1)

    @property
    def MAX_POSITION_VALUE(self):
        # v-margin-sizing-2026-04-22: raised default from $10k → $25k so the
        # new buying-power-based cap (MAX_POSITION_VALUE_BP_PCT, 15% of BP)
        # becomes the binding constraint on typical margin accounts. This
        # field now acts as an absolute safety ceiling; the BP cap does the
        # real adaptive work. Legacy config files pinning this to 10000
        # still work — they'll just bind tighter than the BP cap.
        return self.manager.get('trading.max_position_value', 25000)

    @property
    def MIN_BUYING_POWER(self):
        return self.manager.get('trading.min_buying_power', 100)

    @property
    def REQUIRE_CLOSE_CONFIRMATION(self):
        return self.manager.get('order_management.require_close_confirmation', True)

    @property
    def CONFIRM_ONLY_LOSSES(self):
        return self.manager.get('order_management.confirm_only_losses', False)

    @property
    def CONFIRM_THRESHOLD_PERCENT(self):
        return self.manager.get('order_management.confirm_threshold_percent', 5)

    # ──────────────────────────────────────────────────────────────────
    # v-autonomy-profile-2026-09-08: autonomy profiles for supervised
    # vs autonomous operation. The 'supervised' profile (default) requires
    # UI confirmation for exits and times out to deny. The 'autonomous_live'
    # profile either disables confirmation entirely or uses fail-open
    # timeout (execute the close if UI doesn't respond).
    # ──────────────────────────────────────────────────────────────────

    @property
    def TRADING_PROFILE(self) -> str:
        """Operating profile: 'supervised' (default) or 'autonomous_live'.
        
        Profiles control:
          - require_close_confirmation behavior
          - confirmation_timeout_action (deny vs execute)
          - flatten_on_circuit behavior
        
        Set via environment variable TRADING_PROFILE or config file.
        """
        env_profile = os.getenv("TRADING_PROFILE")
        if env_profile:
            return env_profile.lower()
        return self.manager.get('profile', 'supervised').lower()

    @property
    def CONFIRMATION_TIMEOUT_SEC(self) -> float:
        """Timeout in seconds for close confirmation requests.
        
        Default 30s. After this timeout, the action is determined by
        CONFIRMATION_TIMEOUT_ACTION.
        """
        return float(self.manager.get('order_management.confirmation_timeout_sec', 30.0))

    @property
    def CONFIRMATION_TIMEOUT_ACTION(self) -> str:
        """Action when close confirmation times out: 'deny' or 'execute'.
        
        - 'deny' (default for supervised): block the close, position stays open
        - 'execute' (autonomous_live): fail-open, execute the close
        
        For autonomous operation, 'execute' prevents the scenario where a
        losing position stays open because the operator wasn't watching.
        """
        profile = self.TRADING_PROFILE
        if profile == 'autonomous_live':
            default = 'execute'
        else:
            default = 'deny'
        return self.manager.get('order_management.confirmation_timeout_action', default).lower()

    @property
    def FLATTEN_ON_CIRCUIT(self) -> bool:
        """Close all positions when the daily-loss circuit trips.
        
        Default False for supervised profile (alert only).
        Recommended True for autonomous_live profile to prevent
        unattended bleed-out.
        
        WARNING: When enabled, the bot will close ALL bot-managed
        positions when the circuit trips. This is aggressive but
        prevents catastrophic loss from an unmonitored runaway.
        """
        profile = self.TRADING_PROFILE
        if profile == 'autonomous_live':
            default = True
        else:
            default = False
        return bool(self.manager.get('trading.flatten_on_circuit', default))

    @property
    def NEWS_LOOP_SEC(self) -> float:
        """Cadence of the news_loop that refreshes the NewsBus.
        
        Default 20s — fast enough to catch breaking news but not so
        fast as to hammer free RSS feeds. Adjust based on watchlist
        size and API rate limits.
        """
        return float(self.manager.get('trading.news_loop_sec', 20.0))

    @property
    def ML_PREDICTION_ENABLED(self):
        return self.manager.get('trading.ml_prediction_enabled', True)

    @property
    def ML_VETO_CONFIDENCE(self) -> float:
        """ML confidence (0-1) at or above which a disagreeing ML signal vetoes
        the trade. Default 0.65 — set higher for fewer vetoes, lower for stricter."""
        return float(self.manager.get('trading.ml_veto_confidence', 0.65))

    @property
    def WARMUP_MINUTES_BEFORE_OPEN(self) -> int:
        """Minutes before regular market open (09:30 ET) to wake the bot so
        it can scan the watchlist and build indicator context before the
        first tradable bar. New entries are still blocked until regular
        hours by the signal-router gate.

        Default 30 — set to 0 to disable warmup."""
        return int(self.manager.get('trading.warmup_minutes_before_open', 30))

    @property
    def ENABLE_SMART_TAKE_PROFIT(self) -> bool:
        """v-smart-target-2026-06-02: pick take_profit from price
        structure (bb_upper, high_20, recent-range) rather than
        blindly setting rr_ratio × stop_distance above entry.

        Triggered by operator observation 2026-06-01 that take_profit
        targets were unreachable; data showed only 5/100 historical
        bot trades hit take_profit.

        When enabled, signals whose nearest meaningful resistance
        doesn't give at least 1.5R are SKIPPED rather than placed
        with an unreachable target.

        Default True. Toggle False via API to fall back to the old
        rr-target formula without a code change."""
        return bool(
            self.manager.get('trading.enable_smart_take_profit', True)
        )

    @property
    def ENABLE_MARKET_CONTEXT_GATE(self) -> bool:
        """v-market-context-gate-2026-06-08: skip entries when the
        broader market regime disagrees with the signal direction.

        Mean-rev / breakout / news BUY: skip if regime is 'risk_off'.
        News SELL: skip if regime is 'risk_on'.

        Default True. Toggle via API if it proves too restrictive."""
        return bool(
            self.manager.get('trading.enable_market_context_gate', True)
        )

    @property
    def ENABLE_MARKET_CONTEXT_SIZING(self) -> bool:
        """v-market-context-sizing-2026-06-08: scale position size
        by MarketContext.conviction_multiplier (range 0.5-1.5).

        Applied in risk/manager.py AFTER strategy_size_multiplier
        but BEFORE live_size_multiplier. A risk-on day with sector
        tailwind produces full size; mixed/midday/sector-headwind
        setups get downsized to 0.5-0.7x.

        Default True. Independent of ENABLE_MARKET_CONTEXT_GATE —
        you can size on context without gating on it."""
        return bool(
            self.manager.get('trading.enable_market_context_sizing', True)
        )

    @property
    def ENABLE_DIRECTION_GATE_MEAN_REV(self) -> bool:
        """v-direction-gate-mean-rev-uptrend-pullback-2026-05-29:
        require direction reader to confirm uptrend before mean-rev's
        uptrend_pullback path fires. Classic oversold_bounce path is
        not gated by this — its edge is buying bounces in downtrends.
        Default True. Toggle False via API if the gate rejects too
        many valid uptrend pullbacks."""
        return bool(
            self.manager.get('trading.enable_direction_gate_mean_rev', True)
        )

    @property
    def ENABLE_DIRECTION_GATE_BREAKOUT(self) -> bool:
        """v-direction-gate-breakout-2026-05-29: require direction
        reader to confirm a strong, non-late uptrend before breakout
        BUY fires. Prevents late breakouts (the most common
        false-signal pattern). Default True."""
        return bool(
            self.manager.get('trading.enable_direction_gate_breakout', True)
        )

    @property
    def ENABLE_DIRECTION_GATE_NEWS(self) -> bool:
        """v-direction-gate-news-2026-05-29: require direction reader
        to NOT disagree with the news signal direction. BUY needs
        direction>=0 and not exhausted; SELL inverted. Layered ON TOP
        of v-news-confirmation-gate-2026-05-27's technical checks for
        defense-in-depth. Default True."""
        return bool(
            self.manager.get('trading.enable_direction_gate_news', True)
        )

    @property
    def ENABLE_MEAN_REV_UPTREND_PULLBACK(self) -> bool:
        """v-mean-rev-uptrend-pullback-2026-05-28: alternative mean-rev
        entry for stocks pulling back to support inside an uptrend.

        Operator instruction 2026-05-28 ~10:00 ET after 3 sessions of
        the bot being silent while specific names ran on momentum.
        Existing mean-rev fires only on extreme oversold (RSI<30 +
        close below BB lower); never catches the "in uptrend, pulling
        back to support" pattern that's been today's regime.

        New trigger:
          RSI between 30 and 45 (mild)
          close >= SMA50 (uptrend confirmation)
          close within 2% of BB lower OR up to 1% below it
        Same downstream gates: price-direction (green bar + vol>=1.5x
        + rejection of lows). Same ATR-based stop/target. Pattern is
        logged as `signal_buy reason=uptrend_pullback` for audit.

        Default True. Toggle False via API if it produces too many
        false starts. Reversible without code change.
        """
        return bool(
            self.manager.get(
                'trading.enable_mean_rev_uptrend_pullback', True
            )
        )

    @property
    def ENABLE_NEWS_TECHNICAL_CONFIRMATION(self) -> bool:
        """v-news-confirmation-gate-2026-05-27: require technical
        confirmation before news strategy fires signal_buy/signal_sell.

        Operator decision 2026-05-27 after 3-for-3 losing entries
        across 2 sessions (FLY/ASTS 2026-05-26, TSLA would-have on
        2026-05-27) all of shape: bullish news sentiment + already-
        moved price. News alone is a lagging indicator. With this
        gate on, news_strategy demands matching technical state:
          BUY: rsi<55, price within 5% of sma_20, macd_val>=macd_sig,
               volume_ratio>=1.2x average. SELL: inverted.

        Default True. Toggle False via API
        (PUT /api/settings/trading) if the gate proves too restrictive
        in practice and we lose real opportunities. Audit grep:
        `engine_decision .* reason=no_technical_confirmation`.
        """
        return bool(
            self.manager.get(
                'trading.enable_news_technical_confirmation', True
            )
        )

    @property
    def ENABLE_NEWS_VERIFIER(self) -> bool:
        """v-news-verifier-toggle-2026-04-29: enable the fresh-news
        re-verification gate that was added 2026-04-28.

        Default False — disabled per user request. The gate's veto
        decisions felt opaque ("vetoed for fresh_sentiment_against_signal_avg
        =-0.137" doesn't tell you whether that was right or wrong without
        the shadow-tracker scorecard). When disabled, news_strategy fires
        on cached RSS sentiment alone — same behaviour as before the
        verifier was introduced.

        The shadow-tracker (bot_shadow_news_vetoes table) keeps
        accumulating data when the verifier IS enabled, so toggling this
        back on later still has historical evidence to compare against.

        Set True in Config.yaml to re-enable."""
        return bool(self.manager.get('trading.enable_news_verifier', False))

    @property
    def NEWS_VERIFIER_ADVISORY(self) -> bool:
        """v-news-verifier-advisory-2026-05-08: run the fresh-news
        verifier in advisory mode — it executes and logs its verdict
        on every news signal, but does NOT veto the trade.

        Rationale: the operator complained that 100% gating felt opaque
        (you can't tell whether a veto was right without seeing the
        outcome). Advisory mode emits an audit line
        `engine_decision component=news_verifier_advisory action=advisory
        reason=<verdict> fresh_count=N latest_age_min=...` for every
        news entry, AND records the decision against the eventual
        bot_trades outcome. After a week of data we can grade whether
        the verifier's vetoes would have improved or hurt P&L, then
        flip ENABLE_NEWS_VERIFIER True with evidence.

        Default True — the cost of running verification is one HTTP
        call; the benefit is observability. Set False to silence."""
        return bool(self.manager.get('trading.news_verifier_advisory', True))

    # ── v-newsbus-gates-2026-09-09 ────────────────────────────────────
    # News thesis gates: deterministic sizing based on freshness,
    # source tier, and corroboration. See ARCHITECTURE.md §NewsBus.

    @property
    def NEWS_GATE_MAX_AGE_SEC(self) -> float:
        """Maximum age in seconds for news to be considered fresh.
        
        News older than this vetoes the trade. Default 1800 (30 min).
        Context: 30-min window is aggressive but appropriate for
        intraday news plays. For swing trades or EOD entries, consider
        raising to 3600-7200 (1-2h).
        """
        return float(self.manager.get('trading.news_gate_max_age_sec', 1800.0))

    @property
    def NEWS_GATE_SOURCE_TIER_FLOOR(self) -> int:
        """Minimum acceptable source tier (1=best, 3=worst).
        
        Sources with tier > this are rejected.
          tier 1: Yahoo Finance (curated, API-backed)
          tier 2: Google News (aggregated, decent latency)
          tier 3: MarketWatch scrape (unreliable, may lag)
        
        Default 2 — allows tiers 1-2; rejects scrape-only sources.
        Set to 3 to allow all sources (not recommended for live).
        """
        return int(self.manager.get('trading.news_gate_source_tier_floor', 2))

    @property
    def NEWS_GATE_MIN_CORROBORATION(self) -> int:
        """Minimum distinct sources for full position size.
        
        1 fresh source → 0.5× size (reduced confidence).
        2+ fresh sources → 1.0× size (corroborated thesis).
        
        Default 2. Set to 1 to allow full size on single-source news
        (increases risk of trading on rumor/error).
        """
        return int(self.manager.get('trading.news_gate_min_corroboration', 2))

    @property
    def NEWS_GATE_SINGLE_SOURCE_MULTIPLIER(self) -> float:
        """Size multiplier for single-source fresh news.
        
        When only one source corroborates the thesis, we reduce
        position size as a hedge against single-source error.
        Default 0.5 (half size). Range 0.25-0.75 recommended.
        """
        return float(self.manager.get('trading.news_gate_single_source_multiplier', 0.5))

    @property
    def ENABLE_NEWS_THESIS_EXIT(self) -> bool:
        """Enable thesis-break exit when news flips against position.
        
        v-newsbus-gates-2026-09-09: when enabled, open positions are
        monitored for news that contradicts the entry thesis. If fresh
        news sentiment flips direction (bullish→bearish for longs,
        vice versa), the position is flagged for early exit.
        
        Default False — enable after validating on shadow data.
        """
        return bool(self.manager.get('trading.enable_news_thesis_exit', False))

    @property
    def NEWS_THESIS_EXIT_SENTIMENT_FLIP(self) -> float:
        """Sentiment threshold for thesis-break detection.
        
        For a long position entered on sentiment +0.40, a flip is
        detected when fresh sentiment falls below -NEWS_THESIS_EXIT_SENTIMENT_FLIP.
        Default 0.15 — relatively tight; catches genuine reversals
        without exiting on neutral noise.
        """
        return float(self.manager.get('trading.news_thesis_exit_sentiment_flip', 0.15))

    # ── ENABLE_NEWS_VERIFIER promotion criteria ──────────────────────
    # Research-approved floors for enabling the news verifier in live.
    # See ARCHITECTURE.md §9 for full promotion criteria.
    #
    # Stage A: n≥80 verified signals, PF≥1.30
    # Stage B: n≥200 verified signals, PF≥1.50
    #
    # ENABLE_NEWS_VERIFIER remains False by default until Stage B
    # metrics are met in walk-forward validation.

    # ── v-side-classifier-config-2026-05-13 ───────────────────────────
    # Side-classifier subsystem flags. All default OFF so the live
    # trading path is unaffected until the operator explicitly opts in.
    # Rollout flavor B (per-strategy gradual) chosen 2026-05-13:
    # USE_SIDE_CLASSIFIER is the master switch; the per-strategy
    # switches let us gate news_strategy first, then mean_reversion.

    @property
    def USE_SIDE_CLASSIFIER(self) -> bool:
        """Master switch for the side classifier. Default False.
        When False, no part of `core/classifier` is consulted by the
        live trading path. Flip via `Config().yaml: trading.use_side_classifier`."""
        return bool(self.manager.get('trading.use_side_classifier', False))

    @property
    def SIDE_CLASSIFIER_NEWS_STRATEGY(self) -> bool:
        """Per-strategy gate. When `USE_SIDE_CLASSIFIER=True`, this
        decides whether news_strategy signals pass through the
        classifier. Rollout flavor B starts with this True, then
        flips mean-reversion on after N clean sessions."""
        return bool(self.manager.get('trading.side_classifier_news_strategy', False))

    @property
    def SIDE_CLASSIFIER_MEAN_REVERSION(self) -> bool:
        """Per-strategy gate for mean-reversion entries. Starts False
        and flips True after news_strategy gating proves itself."""
        return bool(self.manager.get('trading.side_classifier_mean_reversion', False))

    @property
    def SIDE_CLASSIFIER_LONG_THRESHOLD(self) -> float:
        """Minimum `long_score` for the rule layer to allow LONG.
        Default 0.55 per PLAN §5.3. Tunable post-backtest."""
        return float(self.manager.get('trading.side_classifier_long_threshold', 0.55))

    @property
    def SIDE_CLASSIFIER_SHORT_THRESHOLD(self) -> float:
        """Minimum `short_score` for the rule layer to allow SHORT.
        Default 0.55 per PLAN §5.3."""
        return float(self.manager.get('trading.side_classifier_short_threshold', 0.55))

    @property
    def SIDE_CLASSIFIER_CACHE_TTL_SEC(self) -> int:
        """How long a per-symbol classifier decision stays valid before
        re-computation. Default 1800s (30min) per PLAN §7."""
        return int(self.manager.get('trading.side_classifier_cache_ttl_sec', 1800))

    @property
    def SIDE_CLASSIFIER_SHADOW_MODE(self) -> bool:
        """Run the classifier alongside live trading but log only.
        Decisions are surfaced in audit logs (component=side_classifier)
        without affecting any gating. Default False.

        v-classifier-runtime-2026-05-13. Per ``.claude/PLAN_trained_side_
        classifier.md`` §8: shadow → advisory → live. This flag is
        what 'shadow' looks like at the code level.
        """
        return bool(self.manager.get(
            'trading.side_classifier_shadow_mode', False))

    @property
    def SIDE_CLASSIFIER_MODEL_PATH(self) -> str:
        """Filesystem path to the FFN (or future TFT) checkpoint.
        Defaults to the watchlist-v1 smoke checkpoint we trained today.
        If the file is absent, ClassifierRuntime fails open
        (rule-only decisions) and logs a single WARNING."""
        return str(self.manager.get(
            'trading.side_classifier_model_path',
            'models/classifier/ffn_watchlist_v1.pt'))

    @property
    def SCREENER_LOOP_SEC(self) -> int:
        """v-parallel-screener-loop-2026-05-12: cadence for the supervised
        screener loop. Default 120s matches the old in-line gate inside
        `_analyze_markets_with_commentary`. Tuning down increases Yahoo +
        Schwab REST traffic; tuning up means a slower watchlist refresh
        when the tape rotates."""
        return int(self.manager.get('trading.screener_loop_sec', 120))

    @property
    def WATCHLIST_SIZE(self) -> int:
        """v-watchlist-size-2026-04-29: max symbols in the dynamic watchlist.

        Default 20 (was effectively 5-8 due to layered caps). Each cycle
        the bot screens hundreds of symbols, scores them, and picks this
        many to actually analyse and trade. Larger = more opportunities;
        smaller = more focused. Tradeoff: each symbol adds ~2-3 seconds
        per analysis cycle and N×N computation for correlation guard.

        Sweet spot for a 5-min bar bot with max_positions=5 is ~20-25 —
        gives the bot room to find good setups without thrashing CPU."""
        return int(self.manager.get('trading.watchlist_size', 20))

    @property
    def ENABLE_PINNED_WATCHLIST(self) -> bool:
        """v-pinned-watchlist-2026-09-17: enable pinned watchlist slots.

        When True, PINNED_WATCHLIST symbols get reserved slots in the
        dynamic watchlist BEFORE filler movers from Yahoo/Schwab screeners.
        This ensures high-quality liquid names (NVDA, TSLA, META, etc.)
        are always analyzed, regardless of whether they're in today's
        Yahoo top movers.

        The problem: Yahoo most-active/gainers/losers often surfaces
        micro-cap lottery names (PURR, AEMD, IOVA-class) that fill the
        limited WATCHLIST_SIZE=20 slots, crowding out quality names
        like AMZN/GOOG/AVGO that would otherwise be profitable.

        Respects HANDS_OFF_DENYLIST — pinned symbols on the denylist
        appear in the watchlist for analysis but never auto-trade.

        SAFE OFF-PATH: Set ENABLE_PINNED_WATCHLIST=0 to disable entirely.
        Default True. Set trading.enable_pinned_watchlist: false to revert
        to pure mover-driven watchlist."""
        env_val = os.getenv("ENABLE_PINNED_WATCHLIST")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_pinned_watchlist', True))

    @property
    def PINNED_WATCHLIST(self) -> list:
        """v-pinned-watchlist-2026-09-17: symbols to reserve in watchlist.

        These symbols get priority slots in the dynamic watchlist. The
        remaining slots (WATCHLIST_SIZE - len(pinned)) are filled by
        the usual Yahoo/Schwab mover screener, ranked by dollar-volume.

        Default list chosen for:
          - High liquidity (can enter/exit $10-50k without moving tape)
          - Institutional interest (news-driven, catalyst-reactive)
          - History of profitable day-trade setups in bot's backtest

        Override via env PINNED_WATCHLIST (comma-separated) or config
        trading.pinned_watchlist (list of strings).

        Note: SPY/QQQ excluded from default because _is_tradeable blocks
        index ETFs. Add them only if that block is removed."""
        default = ['NVDA', 'TSLA', 'META', 'AMZN', 'MSFT', 'GOOGL', 'AVGO', 'AMD']
        env_val = os.getenv("PINNED_WATCHLIST")
        if env_val is not None:
            return [s.strip().upper() for s in env_val.split(',') if s.strip()]
        custom = self.manager.get('trading.pinned_watchlist', default)
        if isinstance(custom, str):
            custom = [s.strip().upper() for s in custom.split(',') if s.strip()]
        return [s.upper() for s in custom]

    @property
    def ENABLE_THESIS_REVALIDATION(self) -> bool:
        """v-thesis-revalidate-2026-04-28: re-verify both news + indicators
        on positions older than THESIS_REVALIDATION_AGE_MIN. Closes the
        position when BOTH the news thesis AND the indicator thesis have
        broken (single-signal failure isn't enough — that's proactive_exit).

        Default False = SHADOW MODE: log what would have fired without
        actually closing positions. Flip True after a day of shadow data
        confirms the gate fires appropriately."""
        return bool(self.manager.get('trading.enable_thesis_revalidation', False))

    @property
    def THESIS_REVALIDATION_SHADOW_MODE(self) -> bool:
        """When True (default), thesis re-validation logs the would-be
        decision but does NOT close the position. Set False to enable
        actual exits. Has no effect when ENABLE_THESIS_REVALIDATION=False."""
        return bool(self.manager.get('trading.thesis_revalidation_shadow_mode', True))

    @property
    def THESIS_REVALIDATION_AGE_MIN(self) -> int:
        """Position age in minutes before the first thesis re-check fires.
        Default 30 — gives the trade time to develop without harvesting
        winners that just need patience."""
        return int(self.manager.get('trading.thesis_revalidation_age_min', 30))

    @property
    def THESIS_REVALIDATION_INTERVAL_MIN(self) -> int:
        """Minimum minutes between thesis re-checks for the same position.
        Default 15 — prevents per-tick spam of news API calls."""
        return int(self.manager.get('trading.thesis_revalidation_interval_min', 15))

    @property
    def THESIS_REVALIDATION_LIMBO_R(self) -> float:
        """Re-validation only runs when |unrealized R| < this. Default 0.5.
        Outside ±0.5R, existing logic (breakeven_stop, proactive_exit,
        trailing stop, hard stop) handles the position. The re-check is
        for the in-between zone where the trade is sitting flat-ish and
        the underlying thesis may be silently breaking."""
        return float(self.manager.get('trading.thesis_revalidation_limbo_r', 0.5))

    @property
    def BREAKEVEN_ACTIVATION_R(self) -> float:
        """v-breakeven-stop-2026-04-28: how far in favor (in R-multiples,
        where R = entry-stop distance) a trade must move before its stop
        is lifted to entry. Default 0.5 = once trade is halfway to take-
        profit, lock in 'cannot lose' downside. Set 0 to disable.

        Live evidence 2026-04-28: PLTR long and TSLA long both went
        positive shortly after entry, then reversed to -0.94% / -0.80%
        before proactive_macd exit fired. Both could have exited flat
        if a breakeven stop had been armed at +0.5R. Net would have been
        $0 instead of -$231 across the two trades.

        Counter-risk: in chop, trades that briefly hit +0.5R then drift
        back to entry get stopped flat instead of getting another chance
        to run. Net positive because preventing winner→loser flips
        compounds, while flat exits cost only commissions."""
        return float(self.manager.get('trading.breakeven_activation_r', 0.5))

    @property
    def ENABLE_PROACTIVE_EXIT(self) -> bool:
        """v-proactive-exit-2026-04-27: when at -0.5R or worse AND a primary
        indicator (MACD / RSI / ADX) has flipped against the position, exit
        early instead of waiting for the full -1.0R stop.

        Live evidence 2026-04-27: RIOT short took the full -$236 stop after
        75 minutes; thesis was already broken at the -0.5R mark per indicator
        readings, but the bot rode the full move down because no proactive
        exit existed. Default True. Disable to restore "ride to stop" behavior.

        Counter-risk: catches some losses that would have reverted to wins
        (false breakouts of indicators near 50/MACD-cross). Net should be
        positive because we're already halfway to stop when this fires —
        the trade was already on a losing path."""
        return bool(self.manager.get('trading.enable_proactive_exit', True))

    @property
    def PROACTIVE_EXIT_MIN_AGE_NEWS(self) -> int:
        """v-proactive-time-floor-2026-04-30: minimum minutes a news-driven
        position must be open before proactive_exit can fire. Whipsaw
        analysis on 2026-04-30 showed 5 of 5 whipsaws closed within 5 min
        of entry; news theses (institutional re-rate after earnings,
        guidance, M&A) play out over hours, not 5-min bars. Holding
        through the first 30 min would have saved ~$1,400 today.
        Default 30. Lower to 15 for faster news plays; raise to 60 if
        whipsaw rate is still high after measurement."""
        return int(self.manager.get('trading.proactive_exit_min_age_news', 30))

    @property
    def PROACTIVE_EXIT_MIN_AGE_MEANREV(self) -> int:
        """Min age before proactive_exit fires for mean-reversion trades.
        Bounces are typically fast — 10 min gives the snap-back time to
        materialize without letting MACD-flip wiggle kick us out
        prematurely. Default 10."""
        return int(self.manager.get('trading.proactive_exit_min_age_meanrev', 10))

    @property
    def PROACTIVE_EXIT_MIN_AGE_DEFAULT(self) -> int:
        """Min age for proactive_exit on any other strategy
        (momentum, breakout, ML, etc). Default 15 — middle ground."""
        return int(self.manager.get('trading.proactive_exit_min_age_default', 15))

    @property
    def PROACTIVE_EXIT_MIN_AGE_DAYTRADE(self) -> int:
        """v-proactive-exit-daytrade-2026-09-14: min age before proactive_exit
        fires for day_trade_momentum / managed day-trade positions.

        RCA FTFT 2026-09-14: held ~20min, proactive_exit suppressed by
        below_min_age (15m default) the entire time, then hit hard_stop.
        Day trades need shorter min age so we don't ride full stop when
        losing early — the thesis (intraday momentum) breaks faster than
        swing/news theses.

        Default 3 minutes. Much shorter than swing/news (15-30) because:
          - Day trades target fast momentum, not multi-hour re-rates
          - Early adverse move often means thesis is broken
          - Hard stop remains floor (this doesn't remove it)

        Set via env PROACTIVE_EXIT_MIN_AGE_DAYTRADE=5 or
        trading.proactive_exit_min_age_daytrade: 5 in Config.yaml.
        """
        env_val = os.getenv("PROACTIVE_EXIT_MIN_AGE_DAYTRADE")
        if env_val is not None:
            try:
                return int(env_val)
            except ValueError:
                pass
        return int(self.manager.get('trading.proactive_exit_min_age_daytrade', 3))

    @property
    def PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD(self) -> float:
        """v-proactive-exit-r-override-2026-09-14: when unrealized R is at or
        below this threshold, bypass min-age gate entirely for proactive exit.

        RCA FTFT 2026-09-14: position reached -0.5R within 5min but min-age
        (15m) blocked proactive exit. When a trade is already at -0.3R or
        worse, the thesis is likely broken regardless of age — let proactive
        exit fire immediately.

        Default -0.3R. More aggressive than the -0.5R proactive_exit trigger
        threshold in check_proactive_exit because this is just the AGE bypass,
        not the exit decision itself. The actual exit still requires indicator
        confirmation (MACD flip, RSI cross, ADX collapse).

        Set via env PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD=-0.5 or
        trading.proactive_exit_r_override_threshold: -0.5 in Config.yaml.
        Set to a very negative value (e.g. -999) to effectively disable.
        """
        env_val = os.getenv("PROACTIVE_EXIT_R_OVERRIDE_THRESHOLD")
        if env_val is not None:
            try:
                return float(env_val)
            except ValueError:
                pass
        return float(self.manager.get('trading.proactive_exit_r_override_threshold', -0.3))

    @property
    def LIVE_SIZE_MULTIPLIER(self) -> float:
        """Global live-launch safety dial applied after strategy multipliers.

        Composes with ``strategy_size_multipliers`` — final sizing is
        ``base * strategy_mult * live_mult``. Default 1.0 (no change).
        Set to 0.25 for the 2026-05-26 quarter-size live launch; graduate
        when walk-forward bootstrap CI lower bound clears 1.0 (use
        ``research/news_strategy_validation.py`` to measure).
        """
        try:
            return float(self.manager.get('trading.live_size_multiplier', 1.0))
        except (TypeError, ValueError):
            return 1.0

    @property
    def LATE_ENTRY_CUTOFF_HOUR(self) -> int:
        """Hour-of-day ET past which no new entries are accepted. See
        engine.py late-entry cutoff guard. Default 15 (3 PM ET)."""
        return int(self.manager.get('trading.late_entry_cutoff_hour', 15))

    @property
    def LATE_ENTRY_CUTOFF_MINUTE(self) -> int:
        """Minute-of-hour ET past which no new entries are accepted on the
        cutoff hour. Default 30 — combined with hour=15, gives 15:30 ET."""
        return int(self.manager.get('trading.late_entry_cutoff_minute', 30))

    def STRATEGY_SIZE_MULTIPLIER(self, strategy_name: str) -> float:
        """Per-strategy size multiplier applied after Kelly sizing.

        Looked up by ``signal.reasoning['strategy']`` in
        ``risk/manager.py``. Missing or invalid keys return 1.0
        (no change).

        Configured via ``trading.strategy_size_multipliers`` (dict).
        Used to under-size strategies still proving edge — e.g.
        mean_reversion at 0.5 until N>=20 confirms PF>=1.5.
        """
        mults = self.manager.get('trading.strategy_size_multipliers', {}) or {}
        if not isinstance(mults, dict):
            return 1.0
        try:
            return float(mults.get(strategy_name, 1.0))
        except (TypeError, ValueError):
            return 1.0

    # ──────────────────────────────────────────────────────────────────
    # v-loop-decoupling-2026-04-30 (Phase 2): cadence + staleness knobs
    # for the three-task model (analysis_loop / position_loop / streamer).
    # ──────────────────────────────────────────────────────────────────
    @property
    def POSITION_LOOP_SEC(self) -> float:
        """Cadence of the FSM exit dispatcher loop. 1.0s = stops fire
        within ~1s of breach. Lowering helps responsiveness; raising
        saves CPU. Below 0.5s noisy, above 3s defeats the purpose."""
        return float(self.manager.get('trading.position_loop_sec', 1.0))

    @property
    def ORDER_MONITOR_INTERVAL_SEC(self) -> float:
        """v-order-monitor-2026-09-10: cadence of the WORKING bracket/OCO
        order monitor loop. Polls Schwab for fills/cancels/rejects on
        bot-managed brackets and syncs Position state. 5s default balances
        responsiveness vs API rate. Lower only if bracket fills frequently
        lag >5s behind market."""
        return float(self.manager.get('trading.order_monitor_interval_sec', 5.0))

    @property
    def REBRACKET_SKIP_IF_FLAT_OR_EXITING(self) -> bool:
        """v-rebracket-exiting-guard-2026-09-18: when True (default), suppress
        re_bracket_position calls when the position is already in an exit path.

        The race condition: hard_stop_breached transitions FSM to EXITING, then
        _close_real_position cancels the OCO bracket. order_monitor sees the
        bracket as CANCELED and calls re_bracket_position — placing a NEW OCO
        while the position is flat or exiting. On Schwab this was REJECTED;
        even without rejection it's wrong to place protection for a closed trade.

        When True, handle_bracket_canceled skips re_bracket if ANY of:
          - Position FSM state is EXITING / CLOSED / ZOMBIE
          - Position quantity == 0 (already flat)
          - Broker confirms qty == 0 (authoritative flat check)

        Audit reason is logged as 'rebracket_skipped_exit_in_flight' so the
        decision trail is clear.

        Default True. Set False only to restore prior always-re-bracket behavior
        for debugging bracket-cancel scenarios.
        """
        env_val = os.getenv("REBRACKET_SKIP_IF_FLAT_OR_EXITING")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.rebracket_skip_if_flat_or_exiting', True))

    @property
    def ENABLE_BROKER_LEG_AUTHORITY(self) -> bool:
        """v-broker-leg-authority-2026-09-14: when True (default), broker OCO/stop
        leg status is authoritative for close decisions.

        If the broker's stop leg is WORKING or already FILLED, the software
        stop path skips the supervised close confirmation flow — the broker
        is already protecting the position (or has already closed it).

        This prevents the FTFT-style race where:
          1. Software stop check starts confirmation flow
          2. Broker stop fills while waiting for confirmation
          3. Confirmation times out with deny → state_reverted to live
          4. Position is now ghost (local says LIVE, broker is flat)

        Default True for fill-path correctness. Set False only to preserve
        prior confirm-on-all-stops behavior for debugging.

        Respects HANDS_OFF_DENYLIST (MU, HQGE, SPCX) — broker-leg authority
        never attempts to close or modify HANDS_OFF positions.
        """
        env_val = os.getenv("ENABLE_BROKER_LEG_AUTHORITY")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_broker_leg_authority', True))

    @property
    def ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT(self) -> bool:
        """v-broker-leg-authority-2026-09-14: when True, auto-flatten ghost
        `live` local state after broker_flat detection.

        When handle_broker_flat_detected runs (broker confirms position is
        flat but local state still shows LIVE), this flag controls whether
        to proactively remove the ghost position from local tracking.

        Default False (safe) — ghost positions are logged but not auto-removed.
        Set True to automatically clean up ghost state after broker confirms flat.

        WARNING: If True, any local/broker desync will result in automatic
        position removal. Only enable if you trust broker_flat detection.

        Respects HANDS_OFF_DENYLIST (MU, HQGE, SPCX) — never touches these.
        """
        env_val = os.getenv("ENABLE_GHOST_FLATTEN_AFTER_BROKER_FLAT")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_ghost_flatten_after_broker_flat', False))

    @property
    def QUOTE_REFRESH_SEC(self) -> float:
        """Cadence of the quote streamer per active symbol. 3s × 10
        symbols = ~3.3 req/s to Schwab. Tighten only if you have
        broker headroom."""
        return float(self.manager.get('trading.quote_refresh_sec', 3.0))

    @property
    def ANALYSIS_LOOP_SEC(self) -> float:
        """Cadence of the slow loop (screener / signal routing / new
        entries). 60s aligns with 1-min bars; finer cadence buys
        nothing for a 5-min-bar bot."""
        return float(self.manager.get('trading.analysis_loop_sec', 60.0))

    @property
    def ANALYSIS_LOOP_RTH_SEC(self) -> float:
        """v-off-hours-analysis-2026-05-01: cadence DURING regular
        trading hours. 30s matches the legacy default; the analysis
        loop runs the full screener/signal/ML pipeline this often.
        Tighten only if you have Schwab quote-budget headroom."""
        return float(self.manager.get('trading.analysis_loop_rth_sec', 30.0))

    @property
    def ANALYSIS_LOOP_OFF_HOURS_SEC(self) -> float:
        """v-off-hours-analysis-2026-05-01: cadence OUTSIDE regular
        hours. Analysis still runs (news, indicators, ML) but at a
        reduced rate — data sources don't update meaningfully overnight
        and the screener/quotes burn budget for no return. Default 300s
        (5 min) keeps the bot informed without hammering APIs.
        Signal-router market_hours gate still blocks new orders."""
        return float(self.manager.get('trading.analysis_loop_off_hours_sec', 300.0))

    @property
    def QUOTE_MAX_STALE_SEC(self) -> float:
        """Soft cliff: position_loop will SKIP exit evaluation on a
        quote older than this. 15s tolerates a Schwab hiccup but never
        acts on yesterday's price."""
        return float(self.manager.get('trading.quote_max_stale_sec', 15.0))

    @property
    def QUOTE_HARD_STALE_SEC(self) -> float:
        """Hard cliff: cache older than this means the streamer is
        broken. Trips an alert; new entries pause via analysis_loop
        gating. Default 60s."""
        return float(self.manager.get('trading.quote_hard_stale_sec', 60.0))

    @property
    def QUOTE_FETCH_TIMEOUT_SEC(self) -> float:
        """Per-symbol quote fetch timeout. 3s gives Schwab room; longer
        starves other symbols when one hangs."""
        return float(self.manager.get('trading.quote_fetch_timeout_sec', 3.0))

    @property
    def QUOTE_FETCH_CONCURRENCY(self) -> int:
        """Max in-flight quote fetches at once across all symbols."""
        return int(self.manager.get('trading.quote_fetch_concurrency', 8))

    @property
    def TASK_RESTART_MAX_CRASHES(self) -> int:
        """Supervisor: number of crashes-in-window before circuit-break."""
        return int(self.manager.get('trading.task_restart_max_crashes', 3))

    @property
    def TASK_RESTART_WINDOW_SEC(self) -> float:
        """Supervisor: rolling window for crash counting."""
        return float(self.manager.get('trading.task_restart_window_sec', 300.0))

    @property
    def TRAIL_ACTIVATION_ATR_MULT(self) -> float:
        """ATR multiples the price must move in favor before the trailing
        stop engages. Default 2.0. Was hardcoded 1.0 until 2026-04-23 —
        too tight: trail fired at breakeven levels, locking in pennies.

        Evidence (2026-04-23, 14 trades): avg win +$16.58 vs avg loss
        -$56.98 = R:R 0.29, unprofitable even at 50% win rate because
        winners never reached the 3×ATR take-profit. Raising activation
        to 2×ATR forces price to prove momentum before the trail bites.

        Tuning: higher = more patience (winners run longer, small wins
        become no-wins); lower = more protection (catches more wins but
        smaller). Sweet spot typically between 1.5 and 2.5 for 5-min bars."""
        return float(self.manager.get('trading.trail_activation_atr_mult', 2.0))

    @property
    def TRAIL_WIDTH_ATR_MULT(self) -> float:
        """Once activated, the trailing stop sits this many ATRs behind
        the running peak. Default 1.5. Was hardcoded 1.0 — equal to the
        average intraday noise, so any normal pullback triggered exit.

        With entry stop at 1.5×ATR and trail width at 1.5×ATR, winners
        and losers have symmetric risk — so the 3×ATR take-profit can
        actually be reached by surviving normal pullbacks."""
        return float(self.manager.get('trading.trail_width_atr_mult', 1.5))

    @property
    def SIZING_STRENGTH_WEIGHT(self) -> float:
        """Weight of signal.strength in the combined sizing quality score.

        Sizing uses a blended quality = strength_weight × signal.strength
                                      + (1 - strength_weight) × signal.confidence

        Default 0.5 = equal blend. Set 0.0 to size purely on confidence
        (pre-v-size-by-strength behavior); 1.0 to size purely on strength.

        Why both matter:
          confidence  — strategy's estimate of win probability (Kelly input)
          strength    — magnitude of the specific setup (e.g., sentiment
                        compound 0.35 vs 0.80; oversold by 2% vs 5%)
        A high-confidence weak signal and a low-confidence strong signal
        both deserve less than full size. Blending captures both."""
        return float(self.manager.get('trading.sizing_strength_weight', 0.5))

    @property
    def KELLY_FLOOR(self) -> float:
        """Minimum Kelly multiplier for position sizing. Default 0.50.
        Was 0.25 — quarter-sized any sub-0.60 confidence signal, making
        the bot fire trivially small trades that commissions + slippage
        eat alive. Raising to 0.50 doubles the floor; high-confidence
        trades (> 0.75) naturally get full-Kelly."""
        return float(self.manager.get('trading.kelly_floor', 0.50))

    @property
    def MAX_POSITION_VALUE_BP_PCT(self) -> float:
        """Per-position notional cap as a fraction of BUYING POWER (not
        equity). Default 0.15 = 15% of BP. With max_positions=5, this
        caps total utilization at 5 × 15% = 75% of BP — leaves headroom
        for volatility and concentration rules.

        On $71k buying power (observed with 2.6x margin on $27k equity):
          0.15 × 71k = $10,650 per position  (up from old fixed $10k cap)
          0.20 × 71k = $14,200 per position
          0.25 × 71k = $17,750 per position

        RISK per trade still comes from MAX_RISK_PER_TRADE × EQUITY,
        NOT buying power. Position size grows with margin, but maximum
        loss stays tied to what we can actually afford (equity).

        The legacy fixed MAX_POSITION_VALUE ($10k) still applies as an
        absolute ceiling — whichever is smaller wins. Set
        trading.max_position_value very high to remove that ceiling."""
        return float(self.manager.get('trading.max_position_value_bp_pct', 0.20))

    @property
    def ENABLE_MEAN_REV_SHORT(self) -> bool:
        """Enable the mean-reversion SHORT branch (RSI > 70 + above BB upper).

        Default False. Live evidence 2026-04-22: 5 closed short trades,
        2 wins / 3 losses, realized -$115.03, profit factor 0.25.
        60-day backtest showed PF 1.02 — marginal on a neutral tape and
        a loser on an uptrending one (RSI>70 is the norm, not a reversal
        signal, during a rally).

        Structural issue: no symmetric "rising peak" filter exists to
        match the long side's "falling knife" filter. Short fires on any
        overbought reading, including peaks in confirmed uptrends that
        just keep making higher highs.

        Flip True in Config.yaml to re-enable — ideally after adding the
        trend-context filter (require close<SMA50 AND MACD<signal before
        taking the short). Existing short positions remain under the bot's
        exit management when this flag is off; only NEW shorts are blocked.

        v-rising-peak-filter-2026-06-08: the trend-context filter now
        exists — see ENABLE_RISING_PEAK_FILTER below. Re-enabling
        mean-rev SHORT also still requires sim soak before live; the
        filter narrows the failure mode but doesn't eliminate the need
        to verify the new gate produces sensible trade counts."""
        return bool(self.manager.get('trading.enable_mean_rev_short', False))

    @property
    def ENABLE_RISING_PEAK_FILTER(self) -> bool:
        """Rising-peak filter on mean-rev SHORT entries (default True).

        Symmetric to the falling-knife filter on the LONG side. When
        True, the mean-rev SHORT branch requires both:
          * close < SMA50          (downtrend confirmation)
          * MACD < MACD signal     (bearish momentum confirmation)
        before firing the SELL signal. Either gate failing → skip with
        audit reason 'rising_peak_uptrend'. Missing indicator data
        (sma_50 == 0, insufficient bars) → fail-closed with
        'rising_peak_no_data'.

        Context: re-enabling ENABLE_MEAN_REV_SHORT on 2026-05-11 produced
        shorts-only behavior — 5 simultaneous shorts at -$665 unrealized
        — because RSI>70 fires constantly during a sustained rally. The
        long-side falling-knife filter (close<SMA50 blocks longs in
        downtrends) had no short-side mirror; this property is that
        mirror.

        Default True — flip to False ONLY for backtest comparisons
        (filter-on vs filter-off) where you want to characterize the
        gate's lift. In live, do not flip False; the filter is the
        precondition for re-enabling ENABLE_MEAN_REV_SHORT."""
        return bool(self.manager.get('trading.enable_rising_peak_filter', True))

    @property
    def ENABLE_MEAN_REV_SHORT_SHADOW(self) -> bool:
        """Log would-be mean-rev SHORT signals WITHOUT placing trades.

        v-mean-rev-short-shadow-2026-06-09: live evidence-gathering for
        the SHORT branch. The 360-day backtest (2026-06-08) said SHORT
        is marginal (PF 1.05) — but the backtest window was mostly
        bullish. 2026-06-09 produced 21 SHORT signal opportunities on
        a half-day-reversal tape where many would have profited; one
        day is not a verdict, so we accumulate live data without
        risking capital.

        When True AND ENABLE_MEAN_REV_SHORT is False, the mean-rev
        SHORT branch:
          * computes the full hypothetical signal (stop, target,
            indicators)
          * appends the entry to `shadow_short_log.json` for later
            resolution
          * returns None (no live order placed)

        When True AND ENABLE_MEAN_REV_SHORT is True (contradictory):
          * the live path takes precedence; shadow logging skipped to
            avoid double-counting. Operator should pick one mode.

        Default False — opt-in only. Flip in Config.yaml when ready to
        start the shadow-mode accumulation period (target 5-10
        sessions).

        Resolution: `research/shadow_short_resolver.py` walks the log,
        compares each entry to subsequent bars, and computes whether
        the hypothetical OCO bracket would have hit stop or target.
        Aggregate PF / win-rate / max-simultaneous-shorts informs the
        decision to flip ENABLE_MEAN_REV_SHORT live."""
        return bool(self.manager.get('trading.enable_mean_rev_short_shadow', False))

    @property
    def ENABLE_MEAN_REV_LONG_SHADOW(self) -> bool:
        """Log mean-rev LONG signals to shadow_long_log.ndjson for Stage A.

        v-shadow-long-ledger-2026-09-17: live evidence-gathering for the
        LONG mean-reversion branch. Shadow entries are written alongside
        normal LONG signals to enable Stage A validation via the long
        resolver (research/shadow_long_resolver.py).

        When True (default), the mean-rev LONG branch:
          * computes the full hypothetical signal (stop, target,
            indicators)
          * appends the entry to `data/shadow_long_log.ndjson`
          * proceeds normally (returns the TradingSignal)

        Unlike SHORT shadow mode, LONG shadow runs alongside live
        signals — it captures data for analysis without blocking trades.

        Default True — required for Stage A soak. Set via env
        ENABLE_MEAN_REV_LONG_SHADOW=0 or
        trading.enable_mean_rev_long_shadow: false to disable.

        Resolution: `research/shadow_long_resolver.py` walks the log,
        compares each entry to subsequent bars, and computes whether
        the hypothetical OCO bracket would have hit stop or target.
        Aggregate PF / win-rate / expectancy informs the Stage A
        promotion decision for mean-rev LONG."""
        env_val = os.getenv("ENABLE_MEAN_REV_LONG_SHADOW")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_mean_rev_long_shadow', True))

    @property
    def MANUAL_CLOSE_ONLY(self) -> bool:
        """v-live-exit-ladder-2026-06-11. Global pause on the bot's
        dynamic exit management in LIVE mode (breakeven ratchet, ATR
        trail, take-profit/hard-stop enforcement on bot-owned
        positions). Was HARDCODED True in engine __init__ since the
        early manual-supervision era — meaning live positions only
        ever had their static OCO while the full exit ladder ran in
        sim. Default False: the ladder is live. External/manual
        positions remain untouchable regardless (managed_by_bot
        gating + the external-position guard in the close path)."""
        return bool(self.manager.get('trading.manual_close_only', False))

    @property
    def ENABLE_LIVE_SCALE_OUT(self) -> bool:
        """v-live-exit-ladder-2026-06-11. The +1R 50% scale-out block
        is BOOKKEEPING-ONLY — it decrements position.quantity without
        placing a real order. Safe in sim; in live it would desync
        share counts from Schwab while the OCO still pledges the full
        size. Default False. DO NOT enable until partial closes place
        real orders end-to-end (cancel bracket → sell partial → verify
        fill → re-bracket remainder)."""
        return bool(self.manager.get('trading.enable_live_scale_out', False))

    @property
    def ENABLE_REGIME_ALLOCATOR_SHADOW(self) -> bool:
        """v-regime-allocator-shadow-2026-06-10 (roadmap P1). When True,
        every signal routed through the engine gets a shadow allocation
        verdict (SPY 20-day efficiency ratio → trending/choppy →
        breakout-vs-mean-rev permission) logged to
        regime_allocator_shadow.ndjson. Pure observation — gates
        nothing. Default True: zero-risk data collection.

        Evidence: research/regime_allocated_report_2026-06-10.json —
        allocated walk-forward PF 1.11 vs baseline 1.04. Promotion to
        an actual gate requires 2 weeks of live shadow agreement."""
        return bool(self.manager.get('trading.enable_regime_allocator_shadow', True))

    @property
    def REGIME_ALLOCATOR_ER_THRESHOLD(self) -> float:
        """ER cut separating trending from choppy tape. Walk-forward
        showed a stable plateau at 0.30-0.40; degrades below 0.25.
        Default 0.30."""
        return float(self.manager.get('trading.regime_allocator_er_threshold', 0.30))

    @property
    def REGIME_GATE_LIVE_MEANREV(self) -> bool:
        """v-regime-gate-live-meanrev-2026-06-16. Promote the regime
        allocator from shadow to a LIVE veto on mean-reversion signals
        when the tape is trending (SPY 20-day ER >= threshold). Mean-rev
        is a chop strategy; it bled -104.7% in a trending walk-forward
        window it never should have traded. Gating it to choppy tape
        removed that disaster (-> +1.4%), lifted pooled PF 1.06 -> 1.12
        and return +127% -> +180% with 28% fewer trades
        (research/meanrev_gated_report_2026-06-15.json).

        Scope: mean_reversion + oversold_v2 ONLY. Breakout/news are
        untouched (still shadow). Fail-open: unknown ER never vetoes.
        Default True. One-line reversible via
        trading.regime_gate_live_meanrev."""
        return bool(self.manager.get('trading.regime_gate_live_meanrev', True))

    @property
    def ENABLE_CONVICTION_FLOOR_MEANREV(self) -> bool:
        """v-conviction-floor-meanrev-2026-06-17. Block mean-rev signals
        whose meta-model probability is below CONVICTION_FLOOR_META.

        Evidence: research/conviction_validation_2026-06-11.json — the
        meta_proba < 0.60 bucket ran PF 0.44 (losing) across 89 trades,
        vs PF 3.01 / 5.02 above. 2026-06-16 live: an all-medium-
        conviction basket netted -$268; this floor would have blocked
        ARM+ELF (saved $150, killed no wins) — halving the loss.

        Scope: mean_reversion + oversold_v2 ONLY. Fail-open: a signal
        with no meta_proba is never blocked. Default True; reversible
        via trading.enable_conviction_floor_meanrev."""
        return bool(self.manager.get('trading.enable_conviction_floor_meanrev', True))

    @property
    def CONVICTION_FLOOR_META(self) -> float:
        """Minimum meta_proba for a mean-rev entry.

        0.60 was the original 89-trade bucket boundary (PF 0.44 below
        vs 3.01 above). v-conviction-floor-065-2026-09-02: raised to
        0.65 — counterfactual over all 35 mean-rev-long trades with
        meta_proba recorded (5/8–9/2): the 0.60–0.65 band held 9
        trades netting -$247.27 (incl. both 9/2 live losers, NVAX
        0.6235 / HYMC 0.6117); floor 0.65 keeps 16 trades at
        +$1,208.13, PF 3.70 vs 0.98 ungated. Kill-switch:
        trading.enable_conviction_floor_meanrev."""
        return float(self.manager.get('trading.conviction_floor_meta', 0.65))

    # ══════════════════════════════════════════════════════════════════════════
    # v-meanrev-quality-budget-2026-09-15: Modular quality gate + separate risk
    # budget for mean-reversion entries. Prevents mean-rev from consuming the
    # same risk pool as momentum day-trades and rejects low-quality signals.
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def ENABLE_MEAN_REV_QUALITY_GATE(self) -> bool:
        """v-meanrev-quality-gate-2026-09-15: quality gate for mean-rev LIVE
        entries. Rejects low-quality signals that don't meet tighter
        indicator thresholds.

        When True (default), mean-rev entries must pass ALL quality checks:
          1. RSI must be in the oversold quality band (RSI < MEAN_REV_RSI_QUALITY_MAX)
          2. RSI must not be extremely oversold (RSI > MEAN_REV_RSI_QUALITY_MIN)
             - Extreme oversold (RSI < 20) often signals real trouble, not bounce
          3. Distance to VWAP must not exceed threshold (catching extended moves)

        Fail-open on missing indicators (indicator data unavailable).
        Blocked entries audited as 'mean_rev_quality_blocked'.

        Default True. Set via env ENABLE_MEAN_REV_QUALITY_GATE=0 or
        trading.enable_mean_rev_quality_gate: false to disable.
        """
        env_val = os.getenv("ENABLE_MEAN_REV_QUALITY_GATE")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_mean_rev_quality_gate', True))

    @property
    def MEAN_REV_RSI_QUALITY_MAX(self) -> float:
        """Maximum RSI for a quality mean-rev long entry.

        Entries with RSI >= this threshold are blocked as not sufficiently
        oversold. The strategy already has RSI < 30 (oversold_bounce) and
        RSI 30-55 (uptrend_pullback) paths; this gate ensures the engine
        enforces a consistent quality standard.

        Default 35.0 (tighter than strategy's 55 for uptrend_pullback).
        Set trading.mean_rev_rsi_quality_max to adjust.
        """
        return float(self.manager.get('trading.mean_rev_rsi_quality_max', 35.0))

    @property
    def MEAN_REV_RSI_QUALITY_MIN(self) -> float:
        """Minimum RSI for a quality mean-rev long entry.

        Entries with RSI <= this threshold are blocked as extremely
        oversold — often indicates real trouble (margin calls, delisting
        risk, bankruptcy) rather than a bounce setup.

        Default 15.0. Set trading.mean_rev_rsi_quality_min to adjust.
        """
        return float(self.manager.get('trading.mean_rev_rsi_quality_min', 15.0))

    @property
    def MEAN_REV_VWAP_DISTANCE_MAX_PCT(self) -> float:
        """Maximum distance below VWAP (%) for quality mean-rev entry.

        Entries where price is more than this percentage below VWAP are
        blocked — chasing extended moves rarely bounces cleanly. A large
        gap below VWAP indicates sustained selling, not mean-reversion
        opportunity.

        Default 5.0 (5% below VWAP). Set trading.mean_rev_vwap_distance_max_pct
        to adjust. Set very high (e.g., 100) to effectively disable.
        """
        return float(self.manager.get('trading.mean_rev_vwap_distance_max_pct', 5.0))

    @property
    def ENABLE_MEAN_REV_RISK_BUDGET(self) -> bool:
        """v-meanrev-risk-budget-2026-09-15: separate risk budget for
        mean-reversion entries.

        When True (default), mean-rev entries have their own position
        cap (MAX_CONCURRENT_MEAN_REV) separate from day-trade momentum.
        This prevents mean-rev from consuming all available slots when
        volatility spikes produce many oversold signals simultaneously.

        Blocked entries audited as 'mean_rev_budget_exhausted'.

        Default True. Set via env ENABLE_MEAN_REV_RISK_BUDGET=0 or
        trading.enable_mean_rev_risk_budget: false to disable.
        """
        env_val = os.getenv("ENABLE_MEAN_REV_RISK_BUDGET")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_mean_rev_risk_budget', True))

    @property
    def MAX_CONCURRENT_MEAN_REV(self) -> int:
        """Maximum concurrent mean-reversion positions.

        When ENABLE_MEAN_REV_RISK_BUDGET is True, mean-rev entries are
        blocked when current mean-rev positions (including pending orders)
        reach this limit. This is SEPARATE from the global MAX_POSITIONS
        cap which still applies.

        Design rationale:
          - Mean-rev fires on volatility spikes (many stocks oversold at once)
          - Global MAX_POSITIONS (10) could be consumed entirely by mean-rev
          - Day-trade momentum then has no slots when movers appear
          - Separate budget ensures strategy diversity in the portfolio

        Default 3. Set trading.max_concurrent_mean_rev to adjust.
        Minimum enforced is 1.
        """
        val = int(self.manager.get('trading.max_concurrent_mean_rev', 3))
        return max(1, val)

    @property
    def MAX_MEAN_REV_RISK_PCT(self) -> float:
        """Maximum equity percentage allocated to mean-reversion positions.

        Research alias: MEAN_REV_RISK_BUDGET_PCT (same semantics).
        Env alias: MEAN_REV_RISK_BUDGET_PCT also accepted.

        Budget semantics:
          - Default **0** = shadow/paper only (no LIVE risk allocation)
          - When value is 0, the equity-% check is SKIPPED entirely
          - When value > 0, the check blocks new entries if
            (mean-rev notional / equity) >= this threshold

        After Stage A green + Hari approval, first LIVE bucket should be
        ≤1-2% total risk capital, mean-rev share ≤ half of that unless
        Hari says otherwise.

        Set trading.max_mean_rev_risk_pct (or trading.mean_rev_risk_budget_pct)
        to adjust. This complements MAX_CONCURRENT_MEAN_REV for dollar-based
        budgeting.
        """
        env_val = os.getenv("MEAN_REV_RISK_BUDGET_PCT")
        if env_val is not None:
            try:
                return float(env_val)
            except (TypeError, ValueError):
                pass
        yaml_alias = self.manager.get('trading.mean_rev_risk_budget_pct')
        if yaml_alias is not None:
            try:
                return float(yaml_alias)
            except (TypeError, ValueError):
                pass
        return float(self.manager.get('trading.max_mean_rev_risk_pct', 0.0))

    @property
    def MAX_CONCURRENT_MEAN_REV_SHORTS(self) -> int:
        """Maximum simultaneous open hypothetical mean-rev SHORT positions.

        Stage A constraint: max simultaneous open hyp shorts ≤3.
        This applies to shadow-mode short signals tracked for validation.
        Prior live short clusters showed ~PF 0.69 — keep this cap tight.

        Default 3. Set trading.max_concurrent_mean_rev_shorts to adjust.
        """
        val = int(self.manager.get('trading.max_concurrent_mean_rev_shorts', 3))
        return max(1, val)

    @property
    def MEAN_REV_DEDUPE_MINUTES(self) -> int:
        """Dedupe window for same-symbol mean-rev entries (minutes).

        Stage A constraint: exclude same symbol entries within <15 minutes.
        Prevents repeated whipsawing on the same name during volatility.

        Default 15. Set trading.mean_rev_dedupe_minutes to adjust.
        """
        return int(self.manager.get('trading.mean_rev_dedupe_minutes', 15))

    @property
    def MEAN_REV_EXCLUDE_RISK_OFF(self) -> bool:
        """Exclude mean-rev entries during risk_off regime (primary book).

        Stage A constraint: Primary book exclude risk_off; secondary = all regimes.
        When True (default), mean-rev entries are blocked when market
        regime is risk_off (broad selloff). Secondary/shadow book logs
        all regimes for comparison.

        Default True. Set trading.mean_rev_exclude_risk_off: false to disable.
        """
        env_val = os.getenv("MEAN_REV_EXCLUDE_RISK_OFF")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.mean_rev_exclude_risk_off', True))

    @property
    def MEAN_REV_SHADOW_LEDGER_ENABLED(self) -> bool:
        """Enable shadow ledger emission for Stage A validation.

        When True, mean-rev signals emit shadow fields for Stage A tracking:
          setup_type, rsi_14, bb_distance, atr, stop_dist, rr_ratio,
          regime, shadow=true, would_be_R

        These fields feed the Stage A scorecard validation:
          n≥150 resolved OR ≥10 sessions with ≥1 resolved
          PF≥1.30 (fees+slip on hyp fills)
          WR≥48% (scratches |R|<0.05 out of rate, in n)
          exp≥+0.05R
          DD≤6% allocated; max losing day ≤2.0R

        Default True. Set trading.mean_rev_shadow_ledger_enabled: false to disable.
        """
        env_val = os.getenv("MEAN_REV_SHADOW_LEDGER_ENABLED")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.mean_rev_shadow_ledger_enabled', True))

    @property
    def ENABLE_BOT_ONLY_PNL_CIRCUIT(self) -> bool:
        """Use BOT-managed P&L (not account-wide Schwab P&L) for the
        daily-loss circuit. Default True.

        Configuration priority:
          1. Environment variable ENABLE_BOT_ONLY_PNL_CIRCUIT (truthy:
             1/true/yes/on; falsy: 0/false/no/off, case-insensitive)
          2. YAML key trading.enable_bot_only_pnl_circuit
          3. Hardcoded default True

        Why this exists: the operator's Schwab account holds external
        positions the bot never opened (HQGE/PINS/COIN as of 2026-06-08).
        When those externals gap down (e.g., PINS -8% overnight), the
        account-wide day P&L crosses the -1% circuit threshold even
        when the bot's own performance is flat or positive. The bot
        then pauses trading for losses that aren't its responsibility.

        When True, `can_trade()` evaluates:
          bot_daily_pnl = sum(today's realized bot trades)
                        + sum(unrealized P&L of open bot-managed positions)
        instead of the account-wide schwab_daily_pnl.

        When False, falls back to the original behavior (schwab_daily_pnl).

        schwab_daily_pnl remains tracked regardless — the dashboard
        still shows the operator's full-account P&L; only the
        circuit-decision input changes.

        v-env-override-bot-only-pnl-circuit-2026-09-14: env override
        added to match DAY_TRADE_LIVE_ENTRIES_ENABLED / ENABLE_ORB_STRATEGY
        pattern. Fixes false trip on 2026-09-14 where .env had flag=1
        but yaml had flag=false — yaml won, causing account-wide P&L
        (-$3k from external positions) to fire EMERGENCY STOP while
        bot-only P&L was $0.
        """
        env_val = os.getenv("ENABLE_BOT_ONLY_PNL_CIRCUIT")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_bot_only_pnl_circuit', True))

    @property
    def HANDS_OFF_DENYLIST(self) -> frozenset:
        """v-hands-off-denylist-2026-09-14: symbols the bot must NEVER
        count toward bot_daily_pnl or attempt to auto-close.

        Hari's STRICT hands-off portfolio as of 2026-09-14:
          - MU, HQGE, SPCX

        SNAP removed from permanent hands-off on 2026-09-14 — operator
        can now toggle Long-Term / Bot-Managed / Close on SNAP from :9000 UI.

        These positions are long-term / external / manually managed
        and must not trigger or be affected by the bot's daily-loss
        circuit. The denylist is a hard-coded safety net on top of
        the is_long_term / is_external / is_manually_managed flags —
        even if tagging is lost or corrupted, these symbols stay safe.

        Override via config trading.hands_off_denylist (list of strings)
        to add/remove symbols at runtime."""
        default = ['MU', 'HQGE', 'SPCX']
        custom = self.manager.get('trading.hands_off_denylist', default)
        if isinstance(custom, str):
            custom = [s.strip().upper() for s in custom.split(',') if s.strip()]
        return frozenset(s.upper() for s in custom)

    @property
    def CORRELATION_IGNORE_UNMANAGED(self) -> bool:
        """v-correlation-ignore-unmanaged-2026-09-18: exclude unmanaged
        positions (managed_by_bot=False) from correlation cluster checks.

        Problem: on 2026-09-18 QCOM day-trade momentum signal was blocked
        by correlation_guard because NVDA (an external/unmanaged hold with
        managed_by_bot=False) was already in the semis_and_chip_adjacent
        group. But external holds don't represent active bot risk — they're
        held by the human operator and shouldn't block bot entries.

        When True (default): correlation guard ignores positions where
        managed_by_bot=False when determining cluster overlap.

        When False: legacy behavior — all positions count for correlation.

        Override via env CORRELATION_IGNORE_UNMANAGED=0 or yaml
        trading.correlation_ignore_unmanaged=false."""
        env_val = os.getenv("CORRELATION_IGNORE_UNMANAGED")
        if env_val is not None:
            return env_val.lower() not in ("0", "false", "no", "off")
        return bool(self.manager.get('trading.correlation_ignore_unmanaged', True))

    @property
    def CORRELATION_IGNORE_HANDS_OFF(self) -> bool:
        """v-correlation-ignore-hands-off-2026-09-18: exclude HANDS_OFF_DENYLIST
        positions (MU, HQGE, SPCX) from correlation cluster checks.

        Problem: on 2026-09-18 QCOM day-trade momentum signal was blocked
        by correlation_guard because MU (a permanent hands-off long-term
        hold) was in the semis_and_chip_adjacent group. Hands-off positions
        are never managed by the bot and shouldn't block new bot entries
        into the same sector.

        When True (default): correlation guard ignores positions whose
        symbol is in HANDS_OFF_DENYLIST when determining cluster overlap.

        When False: legacy behavior — all positions count for correlation.

        Override via env CORRELATION_IGNORE_HANDS_OFF=0 or yaml
        trading.correlation_ignore_hands_off=false."""
        env_val = os.getenv("CORRELATION_IGNORE_HANDS_OFF")
        if env_val is not None:
            return env_val.lower() not in ("0", "false", "no", "off")
        return bool(self.manager.get('trading.correlation_ignore_hands_off', True))

    @property
    def ENABLE_MANAGED_BY_BOT_PERSIST(self) -> bool:
        """v-manage-persist-2026-09-15: restore managed_by_bot=True across
        Schwab sync/restart for bot-session entries.

        Default True. When enabled, the Schwab sync restores managed_by_bot
        ownership using these sources (in preference order):
          1. In-memory prior position if managed_by_bot=True
          2. _saved_positions_meta / persisted positions file (relaxed qty
             matching: side match + saved managed_by_bot=True is sufficient)
          3. Fallback: today's bot_trades where bot opened the symbol and
             position is still open on Schwab

        NEVER restores managed_by_bot=True for HANDS_OFF_DENYLIST symbols
        (MU, HQGE, SPCX) — they always remain hands-off.

        SAFE OFF-PATH: Set ENABLE_MANAGED_BY_BOT_PERSIST=0 to disable
        entirely — reverts to prior brittle exact-qty-match behavior with
        no bot_trades fallback."""
        env_val = os.getenv("ENABLE_MANAGED_BY_BOT_PERSIST")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_managed_by_bot_persist', True))

    @property
    def ENABLE_MANAGED_OWNERSHIP_EVIDENCE_BROAD(self) -> bool:
        """v-evidence-broad-2026-09-15: broaden ownership evidence sources
        for managed_by_bot restore beyond bot_trades.

        Default True. When enabled, ownership evidence is checked in order:
          1. saved meta managed=True (existing)
          2. bot_trades open/today entry (existing)
          3. **bot_decisions** today with strategy entry for symbol
             (mean_reversion / day_trade_momentum / etc.)
          4. **bot_positions** row with managed strategy (if open-ledger exists)

        This fixes the gap where mean-rev session entries (which don't write
        to bot_trades until exit) were left external on first boot without
        good saved state.

        SAFE OFF-PATH: Set ENABLE_MANAGED_OWNERSHIP_EVIDENCE_BROAD=0 to
        revert to bot_trades-only fallback (original #68 behavior)."""
        env_val = os.getenv("ENABLE_MANAGED_OWNERSHIP_EVIDENCE_BROAD")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_managed_ownership_evidence_broad', True))

    @property
    def ENABLE_BREAKOUT_LONG(self) -> bool:
        """Enable the breakout-long strategy (20-bar high + volume + trend).

        Default False. 60-day 5-min backtest (2026-04-20, 20 top-volume
        symbols) showed breakout-long at PF 0.79-0.83 even with strict
        gates — 50% of entries stop out, only 18-21% hit target. The
        strategy is structurally weak at intraday timeframes (false
        breakouts dominate). Kept in code for daily-bar backtests and
        optional re-enable. Flip to True in Config.yaml to reactivate."""
        return bool(self.manager.get('trading.enable_breakout_long', False))

    @property
    def ENABLE_MOMENTUM_LONG(self) -> bool:
        """Enable the momentum-long strategy (MACD bullish + RSI healthy +
        ADX trending).

        Default False. 60-day 5-min backtest showed momentum-long at
        PF 0.84-0.85 even with strict gates (close>SMA50, RSI 55-65).
        51% of entries stop out, only 16% hit target, 31-35% time out.
        MACD bullish crossovers on 5-min bars produce too many false
        positives. Kept in code for higher-timeframe backtests. Flip to
        True in Config.yaml to reactivate."""
        return bool(self.manager.get('trading.enable_momentum_long', False))

    @property
    def ENABLE_STRICT_LONG_GATES(self) -> bool:
        """Enable the v-profitability-pass-2026-04-20 tighter gates for
        breakout-long and momentum-long. Default True (new stricter
        behavior); set False to restore pre-2026-04-20 gates.

        Changes when True:
          Breakout long: requires close > high_20 * 1.003 (0.3% real break,
                         not tick), ADX > 25 (was 20), volume_ratio > 2.0
                         (was 1.5).
          Momentum long: requires close > SMA50 (trend confirmation),
                         RSI 55-65 (was 50-70).

        Motivation: 60-day backtest showed breakout-long PF 0.83 and
        momentum-long PF 0.85 (both losing). High stop rate (~50%) and
        timeout rate (~31%) pointed at false-breakout and weak-trend
        entries. Tighter gates cut trade count and raise win rate."""
        return bool(self.manager.get('trading.enable_strict_long_gates', True))

    @property
    def ENABLE_UNIVERSE_QUALITY_FILTER(self) -> bool:
        """v-universe-quality-filter-2026-05-22: filter screener candidates
        at universe-build time before strategies see them. Three sub-gates:
        leveraged-ETF blocklist, close > SMA50 (uptrend), and 5-day return
        > SPY 5-day return (relative strength). Default True — turn off
        in yaml to backtest the impact (set `trading.enable_universe_quality_filter: false`)."""
        return bool(self.manager.get('trading.enable_universe_quality_filter', True))

    @property
    def USE_OVERSOLD_BOUNCE_V2(self) -> bool:
        """v-oversold-v2-2026-05-19: gate for the OversoldBounceV2Strategy.
        Default changed to FALSE on 2026-05-19 after backtest showed v2
        underperforms legacy (12 trades, 16.7% win rate, PF 1.04 vs legacy's
        62 trades, 61.3% win rate, PF 1.72 on 30-day NVDA/AMD/RKLB/FCEL/POET
        sample). Strategy code retained for further iteration. Flip True
        in yaml only after v2 outperforms in a fresh backtest."""
        return bool(self.manager.get('trading.use_oversold_bounce_v2', False))

    @property
    def ENABLE_RELAXED_MEAN_REV_LONG(self) -> bool:
        """Enable the v-profitability-pass-2026-04-20 relaxed falling-knife
        filter for mean-reversion LONG. Default True.

        Old behavior: skip if (close<SMA50 AND MACD<signal). Filtered out
        too many oversold-bounce setups — backtest showed only 57 trades
        in 60 days × 20 symbols at PF 2.43.
        New behavior: skip only if (close<SMA50 AND MACD<signal AND
        close<low_5) — require very-recent weakness on top of the two old
        signals to confirm the knife is still falling. Expected to raise
        trade count ~3-5x while keeping PF >= 1.5."""
        return bool(self.manager.get('trading.enable_relaxed_mean_rev_long', True))

    @property
    def ENABLE_SHORT_MIRRORS(self) -> bool:
        """Enable the v-short-mirrors-2026-04-20 short branches:
        - Breakout strategy: breakdown short (close < 20-bar low + trend/volume gates)
        - Momentum strategy: bearish-momentum short (MACD bearish + RSI 30-50 + ADX>25 + below SMA50)

        Default False. 60-day 5-min backtest (2026-04-20, 20 top-volume symbols)
        showed both short branches at PF 0.69 — symmetric shorts fight the oversold
        bounce. Existing mean-reversion short-side (RSI>70 fade) is unchanged by
        this flag. Flip to True for paper testing after re-tuning gate parameters."""
        return bool(self.manager.get('trading.enable_short_mirrors', False))

    @property
    def TRADING_API_KEY(self) -> Optional[str]:
        """Bearer token that protects the REST and WebSocket API endpoints.

        Returns the value of the TRADING_API_KEY environment variable, or None
        when the variable is not set (auth disabled / dev mode).
        """
        return os.getenv("TRADING_API_KEY")

    # Order execution constants
    @property
    def LIMIT_ORDER_SLIPPAGE(self):
        """Slippage factor for limit orders (e.g., 0.001 = 0.1%)"""
        return self.manager.get('trading.limit_order_slippage', 0.001)

    @property
    def DEFAULT_STOP_LOSS_PCT(self):
        """Default stop loss percentage for positions without explicit stops"""
        return self.manager.get('trading.default_stop_loss_pct', 0.05)

    @property
    def DEFAULT_TAKE_PROFIT_PCT(self):
        """Default take profit percentage for positions without explicit targets"""
        return self.manager.get('trading.default_take_profit_pct', 0.10)

    @property
    def MAX_POSITION_VALUE_PCT(self):
        """Maximum position value as fraction of portfolio"""
        return self.manager.get('trading.max_position_value_pct', 0.25)

    @property
    def ATR_STOP_MULTIPLIER(self):
        """ATR multiplier for stop-loss distance (1.5 = stop at 1.5×ATR from entry)."""
        return self.manager.get('trading.atr_stop_multiplier', 1.5)

    @property
    def MEAN_REV_ATR_STOP_MULTIPLIER(self):
        """v-mean-rev-wider-stop-2026-05-20: mean-reversion-specific stop
        multiplier. Default 2.5× ATR — wider than the global 1.5×.

        Rationale: with the bounce-confirmation filter (v-bounce-2026-05-20)
        only firing on real reversal candles, mean-rev entries are higher
        quality and deserve room to breathe through normal intraday
        whipsaw. 1.5× ATR puts stops inside the noise band on volatile
        names (FIG on 5/20: stopped at $22.16, day low $21.78, current
        $22.23 — recovered above the stop level). Wider stop = fewer
        whipsaw exits but bigger loss when stop genuinely hits;
        position-sizing automatically scales down to keep dollar_risk
        constant per trade.

        Falls back to ATR_STOP_MULTIPLIER if yaml key missing."""
        v = self.manager.get('trading.mean_rev_atr_stop_multiplier', None)
        if v is None:
            return 2.5  # mean-rev-specific default
        return v

    @property
    def ATR_REWARD_RISK_RATIO(self):
        """R:R ratio for take-profit relative to stop distance (2.0 = 2:1 R:R)."""
        return self.manager.get('trading.atr_reward_risk_ratio', 2.0)

    @property
    def RISK_PER_TRADE_PCT(self):
        """Fraction of equity risked per trade for ATR-based sizing.

        Default raised 0.010 → 0.015 on 2026-04-24. Live evidence from
        2026-04-23 and -24: notional per trade was $8-9k (good, via margin
        sizing) but dollar_risk was only $130-140 (1% × equity × Kelly 0.5).
        Winners at 0.3-1% moves only returned $28-84; losers at 1.5-2%
        stops cost $130-170. Raising the risk budget 50% grows BOTH wins
        and losses proportionally — the absolute $ values finally matter
        (wins $42-125, losses $195-255 per trade).

        Tuning: 0.01 = conservative (original), 0.015 = moderate (new),
        0.02 = aggressive. Anything above 0.02 is a professional-trader
        territory where a bad streak can deplete equity fast."""
        return self.manager.get('trading.risk_per_trade_pct', 0.015)

    # ──────────────────────────────────────────────────────────────────
    # v-day-trade-momentum-desk-2026-09-10: supervised day-trade momentum
    # for Yahoo day_gainers/losers/most-active movers.
    #
    # Operator request: take profitable intraday trades on liquid movers
    # without sitting frozen behind stacked skip gates. This desk is
    # separate from the existing mean-rev + news strategies.
    # ──────────────────────────────────────────────────────────────────

    @property
    def ENABLE_DAY_TRADE_MOMENTUM(self) -> bool:
        """Enable the day-trade momentum strategy lane.
        
        When True, the DayTradeMomentumStrategy is activated for symbols
        flagged as 'movers' (Yahoo day_gainers/losers/most-active + watchlist).
        Entry logic: relative strength vs SPY, volume surge, pullback-or-
        breakout confirmation, defined ATR stop, trail/time stop.
        
        Size is smaller than swing (via DAY_TRADE_SIZE_MULTIPLIER), respects
        max_positions. Closes supervised unless TRADING_PROFILE=autonomous_live.
        
        Default True. Set trading.enable_day_trade_momentum: false to disable.
        """
        env_val = os.getenv("ENABLE_DAY_TRADE_MOMENTUM")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_day_trade_momentum', True))

    @property
    def DAY_TRADE_LIVE_ENTRIES_ENABLED(self) -> bool:
        """v-pause-live-daytrade-2026-09-10: master switch for LIVE day-trade
        momentum entries.
        
        When False (default), the day-trade momentum strategy generates
        signals for sim/commentary/shadow analysis but BLOCKS actual LIVE
        order placement. This is the immediate pause requested by Hari on
        2026-09-10 product call.
        
        MUST KEEP intact (these work regardless of this flag):
          - Hard loss circuits / ENABLE_BOT_ONLY_PNL_CIRCUIT
          - Flatten / software exits / order_monitor / OCO / bootstrap
            for EXISTING bot day-trades
          - LT hands-off forever: MU, HQGE, SPCX (is_long_term flag)
        
        The flag does NOT flip autonomous_live. It only blocks NEW live
        entries via day_trade_momentum lane.
        
        Promotion to True requires Stage-A validation:
          n>=150 trades, >=10 sessions, PF>=1.30, WR>=48%, exp>=+0.05R,
          DD<=6%, max losing day<=2R.
        
        Default False. Set via env DAY_TRADE_LIVE_ENTRIES_ENABLED=1 or
        trading.day_trade_live_entries_enabled: true in Config.yaml.
        """
        env_val = os.getenv("DAY_TRADE_LIVE_ENTRIES_ENABLED")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.day_trade_live_entries_enabled', False))

    @property
    def MEAN_REV_LIVE_ENTRIES_ENABLED(self) -> bool:
        """v-meanrev-live-flag-2026-09-15: master switch for LIVE mean-reversion
        entries.
        
        When False, the mean_reversion strategy generates signals for
        sim/commentary/shadow analysis but BLOCKS actual LIVE order
        placement.
        
        MUST KEEP intact (these work regardless of this flag):
          - Hard loss circuits / ENABLE_BOT_ONLY_PNL_CIRCUIT
          - Flatten / software exits / order_monitor / OCO / bootstrap
            for EXISTING bot mean-rev positions
          - LT hands-off forever: MU, HQGE, SPCX (is_long_term flag)
        
        The flag does NOT flip autonomous_live. It only blocks NEW live
        entries via mean_reversion lane.
        
        Default True (LIVE entries enabled). Set MEAN_REV_LIVE_ENTRIES_ENABLED=0
        to pause LIVE mean-rev entries while keeping sim/shadow analysis.
        """
        env_val = os.getenv("MEAN_REV_LIVE_ENTRIES_ENABLED")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.mean_rev_live_entries_enabled', True))

    # ══════════════════════════════════════════════════════════════════════════
    # v-flatten-hour-entry-gate-2026-09-15: Block new day-trade LIVE entries
    # at/after the same flatten_hour that would immediately flatten them.
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED(self) -> bool:
        """v-flatten-hour-entry-gate-2026-09-15: block NEW day-trade LIVE
        entries when current ET hour >= flatten_hour.

        P0 RCA 2026-09-15: BBWI entered LIVE at 15:23:30 ET then exited
        ~7s later via day_trade_flatten_hour (flatten_hour=15). Entry at
        or after flatten hour = churn. This gate prevents that churn by
        rejecting day-trade entries that would be immediately flattened.

        When True (default):
          - Any signal with is_day_trade=True in reasoning
          - In LIVE mode
          - When current ET hour >= flatten_hour (from signal reasoning
            or DAY_TRADE_FLATTEN_HOUR config)
          => Entry is BLOCKED with audit reason 'flatten_hour_entry_blocked'

        When False: previous behavior (entries allowed, then flattened).

        MUST KEEP intact (these work regardless of this flag):
          - Flatten / software exits for EXISTING bot day-trades
          - Shadow logging continues (entries still shadow-logged)
          - Other existing gates (DAY_TRADE_LIVE_ENTRIES_ENABLED, etc.)

        Default True. Set via env DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED=0
        or trading.day_trade_flatten_hour_entry_gate_enabled: false to disable.
        """
        env_val = os.getenv("DAY_TRADE_FLATTEN_HOUR_ENTRY_GATE_ENABLED")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get(
            'trading.day_trade_flatten_hour_entry_gate_enabled', True
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # v-daytrade-rsi-entry-gate-2026-09-15: Block day-trade LIVE entries when
    # RSI is already at/below the proactive exit threshold.
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def DAY_TRADE_RSI_ENTRY_GATE_ENABLED(self) -> bool:
        """v-daytrade-rsi-entry-gate-2026-09-15: block day-trade LIVE entries
        when RSI <= DAY_TRADE_RSI_ENTRY_THRESHOLD (default 50).

        P0 RCA 2026-09-15 (ALHC×2): day_trade pullback/continuation entered
        LIVE while RSI <= 50, then proactive_rsi_below_50 exit (or open-desk
        RSI exit) immediately dumped the trade. Entry into a condition that
        already triggers exit = churn, anti-profit.

        When True (default):
          - Any signal with is_day_trade=True in reasoning
          - In LIVE mode
          - SignalType.BUY (longs only, matching the RSI < 50 exit logic)
          - When RSI <= DAY_TRADE_RSI_ENTRY_THRESHOLD
          => Entry is BLOCKED with audit reason 'rsi_below_50_entry_blocked'

        When False: previous behavior (entries allowed, may immediately exit).

        MUST KEEP intact (these work regardless of this flag):
          - Proactive exits for EXISTING positions
          - Shadow logging continues (entries still shadow-logged)
          - Other existing gates (DAY_TRADE_LIVE_ENTRIES_ENABLED, etc.)
          - HANDS_OFF (MU, HQGE, SPCX) unchanged

        The threshold is aligned with proactive_rsi_below_50 exit logic in
        analysis/scale_trail_manager.py and analysis/active_open_desk.py.

        Default True. Set via env DAY_TRADE_RSI_ENTRY_GATE_ENABLED=0
        or trading.day_trade_rsi_entry_gate_enabled: false to disable.
        """
        env_val = os.getenv("DAY_TRADE_RSI_ENTRY_GATE_ENABLED")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get(
            'trading.day_trade_rsi_entry_gate_enabled', True
        ))

    @property
    def DAY_TRADE_RSI_ENTRY_THRESHOLD(self) -> float:
        """RSI threshold for day-trade entry gate (longs).

        Day-trade long entries with RSI <= this threshold are blocked
        because they would immediately be vulnerable to the proactive
        RSI exit (rsi_below_50).

        Aligned with proactive_rsi_below_50 exit threshold in:
          - analysis/scale_trail_manager.py: `if rsi < 50: return "rsi_below_50"`
          - analysis/active_open_desk.py: `elif rsi < 50: proactive_reason = "rsi_below_50"`

        Using <= 50 (not < 50) for the entry gate because RSI exactly at
        50 is right at the edge — one tick of noise and the exit fires.

        Default 50.0. Set trading.day_trade_rsi_entry_threshold to adjust.
        """
        return float(self.manager.get('trading.day_trade_rsi_entry_threshold', 50.0))

    # ══════════════════════════════════════════════════════════════════════════
    # v-rsi-high-entry-veto-2026-09-17: Block day-trade LIVE entries when RSI
    # is already overbought (>=70). Symmetric to the RSI<=50 gate above.
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def DAY_TRADE_RSI_HIGH_ENTRY_VETO_ENABLED(self) -> bool:
        """v-rsi-high-entry-veto-2026-09-17: block day-trade LIVE BUY entries
        when RSI >= DAY_TRADE_RSI_HIGH_ENTRY_THRESHOLD (default 70).

        P0 RCA 2026-09-17 (XE): day_trade breakout @16.31×334 @09:45 ET
        entered with RSI overbought, immediately vulnerable to exit triggers.
        Entry into overbought = chasing, high P(immediate reversal).

        When True (default):
          - Any signal with is_day_trade=True in reasoning
          - In LIVE mode
          - SignalType.BUY (longs only)
          - entry_pattern in ('breakout', 'continuation')
          - When RSI >= DAY_TRADE_RSI_HIGH_ENTRY_THRESHOLD
          => Entry is BLOCKED with audit reason 'rsi_above_70_entry_blocked'

        When False: previous behavior (entries allowed even if overbought).

        Aligns with existing ENABLE_HARD_VETO_CONTINUATION_RSI70 for
        continuation patterns, but extends coverage to breakout patterns.
        Uses SHADOW_VETO_RSI_THRESHOLD (70) as the default threshold to
        maintain consistency with existing RSI 70 gates.

        MUST KEEP intact (these work regardless of this flag):
          - Proactive exits for EXISTING positions
          - Shadow logging continues
          - Other existing gates (DAY_TRADE_LIVE_ENTRIES_ENABLED, etc.)
          - HANDS_OFF (MU, HQGE, SPCX) unchanged
          - DAY_TRADE_RSI_ENTRY_GATE_ENABLED (RSI<=50) unchanged
          - ENABLE_HARD_VETO_CONTINUATION_RSI70 unchanged

        Default True. Set via env DAY_TRADE_RSI_HIGH_ENTRY_VETO_ENABLED=0
        or trading.day_trade_rsi_high_entry_veto_enabled: false to disable.
        """
        env_val = os.getenv("DAY_TRADE_RSI_HIGH_ENTRY_VETO_ENABLED")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get(
            'trading.day_trade_rsi_high_entry_veto_enabled', True
        ))

    @property
    def DAY_TRADE_RSI_HIGH_ENTRY_THRESHOLD(self) -> float:
        """RSI threshold for day-trade high entry veto (longs).

        Day-trade long entries with RSI >= this threshold are blocked
        for breakout and continuation patterns — entering overbought
        conditions is chasing, with high probability of immediate reversal.

        Aligned with SHADOW_VETO_RSI_THRESHOLD (70.0) to maintain
        consistency with existing continuation RSI>=70 veto logic.

        Default 70.0. Set trading.day_trade_rsi_high_entry_threshold to adjust.
        """
        return float(self.manager.get(
            'trading.day_trade_rsi_high_entry_threshold',
            self.SHADOW_VETO_RSI_THRESHOLD  # Default aligned with existing 70
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # v-day-trade-short-2026-09-17: Day-trade momentum SHORT strategy config.
    # Modular short path mirroring long day-trade momentum with inverse logic.
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def ENABLE_DAY_TRADE_SHORT(self) -> bool:
        """Enable the day-trade momentum SHORT strategy lane.

        v-day-trade-short-2026-09-17: modular short path mirroring the long
        day-trade momentum strategy. Entry logic:
          1. Weak RS vs SPY (symbol UNDERPERFORMING SPY)
          2. Volume surge (same as longs)
          3. Breakdown/continuation-down patterns
          4. Direction reader bearish + non-exhausted
          5. Inverse RSI logic (don't short oversold)

        When True, the DayTradeMomentumShortStrategy is activated. LIVE
        order placement is controlled separately by DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED.

        SAFE OFF-PATH: Set ENABLE_DAY_TRADE_SHORT=0 to disable entirely
        (no signals, no shadow logs). Default True for shadow soak.

        Respects HANDS_OFF_DENYLIST (MU, HQGE, SPCX) — never shorts these.
        """
        env_val = os.getenv("ENABLE_DAY_TRADE_SHORT")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_day_trade_short', True))

    @property
    def DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED(self) -> bool:
        """v-day-trade-short-2026-09-17: master switch for LIVE day-trade
        SHORT entries.

        When False (default), the day-trade short strategy generates
        signals for sim/commentary/shadow analysis but BLOCKS actual LIVE
        short order placement. This is the SHADOW-FIRST deployment mode
        requested by operator.

        MUST KEEP intact (these work regardless of this flag):
          - Hard loss circuits / ENABLE_BOT_ONLY_PNL_CIRCUIT
          - Flatten / software exits for EXISTING short positions
          - LT hands-off forever: MU, HQGE, SPCX (HANDS_OFF_DENYLIST)

        The flag does NOT flip autonomous_live. It only blocks NEW live
        short entries via day_trade_momentum_short lane.

        Promotion to True requires Stage-A validation:
          n>=150 trades, >=10 sessions, PF>=1.30, WR>=48%, exp>=+0.05R,
          DD<=6%, max losing day<=2R.

        Default False (SHADOW MODE). Set via env DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED=1
        or trading.day_trade_short_live_entries_enabled: true in Config.yaml
        ONLY after Stage-A soak passes.
        """
        env_val = os.getenv("DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.day_trade_short_live_entries_enabled', False))

    @property
    def ENABLE_DAY_TRADE_SHORT_SHADOW(self) -> bool:
        """Enable shadow logging for day-trade SHORT signals.

        v-day-trade-short-2026-09-17: when True (default) AND
        DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED is False, the short strategy:
          * computes the full hypothetical signal (stop, target, indicators)
          * appends the entry to `data/day_trade_short_shadow.ndjson`
          * emits strategy_decision log with shadow=True
          * returns None (no live short order placed)

        This allows collecting shadow data for Stage-A validation before
        enabling LIVE shorts.

        When True AND DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED is True:
          * the live path takes precedence; shadow logging still runs
            for audit trail but LIVE orders are placed.

        SAFE OFF-PATH: Set ENABLE_DAY_TRADE_SHORT_SHADOW=0 to disable
        shadow logging entirely (silent skip).

        Default True. Flip in Config.yaml to start shadow accumulation.
        """
        env_val = os.getenv("ENABLE_DAY_TRADE_SHORT_SHADOW")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_day_trade_short_shadow', True))

    @property
    def DAY_TRADE_SHORT_MIN_WEAK_RS_VS_SPY(self) -> float:
        """Minimum weakness (negative RS) vs SPY for short entry.

        v-day-trade-short-2026-09-17: for shorts, we WANT symbols that are
        UNDERPERFORMING SPY. The threshold is the minimum negative RS:
          symbol_change - spy_change <= -threshold

        Example: if SPY is +0.5% and symbol is -1.0%:
          RS = -1.0 - 0.5 = -1.5%
          With threshold 0.5, this PASSES (symbol is weak enough).

        Lower values = more permissive (weaker filter).
        Higher values = stricter (require more pronounced weakness).

        Default 0.5% (symbol must underperform SPY by at least 0.5%).
        Set trading.day_trade_short_min_weak_rs_vs_spy to adjust.
        """
        return float(self.manager.get('trading.day_trade_short_min_weak_rs_vs_spy', 0.5))

    @property
    def DAY_TRADE_SHORT_RSI_FLOOR(self) -> float:
        """RSI floor for day-trade short entries (avoid shorting oversold).

        v-day-trade-short-2026-09-17: inverse of the long RSI ceiling.
        Don't short when RSI is already oversold — the bounce risk is high.

        Shorts with RSI <= this floor are blocked to avoid shorting into
        exhaustion / capitulation.

        Default 30.0. Set trading.day_trade_short_rsi_floor to adjust.
        """
        return float(self.manager.get('trading.day_trade_short_rsi_floor', 30.0))

    @property
    def DAY_TRADE_SHORT_RSI_CEILING(self) -> float:
        """RSI ceiling for day-trade short entries.

        v-day-trade-short-2026-09-17: don't short when RSI is extremely
        overbought (>= ceiling) — counter-intuitive but these are often
        strength signals, not reversal candidates.

        The sweet spot for momentum shorts is RSI 35-65 (weak but not
        oversold/overbought extremes).

        Default 85.0 (very permissive — only blocks extreme overbought).
        Set trading.day_trade_short_rsi_ceiling to adjust.
        """
        return float(self.manager.get('trading.day_trade_short_rsi_ceiling', 85.0))

    @property
    def DAY_TRADE_SHORT_MAX_CONCURRENT(self) -> int:
        """Maximum concurrent day-trade short positions.

        v-day-trade-short-2026-09-17: separate cap from long positions
        to control short exposure. During Stage-A soak, keep this low.

        Default 2. Set trading.day_trade_short_max_concurrent to adjust.
        """
        return int(self.manager.get('trading.day_trade_short_max_concurrent', 2))

    # ══════════════════════════════════════════════════════════════════════════
    # v-late-entry-gate-2026-09-15: Late-entry detection to prevent chasing
    # extended moves. Shadow mode logs only; production mode can hard-skip.
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def ENABLE_LATE_ENTRY_GATE(self) -> bool:
        """Master switch for late-entry detection gate.
        
        When True, the engine evaluates late-entry heuristics on
        day_trade_momentum and mean_reversion signals:
          1. Extension ratio from session open toward session high
          2. VWAP chase: long above VWAP + k*ATR
          3. Bars-since-impulse: if breakout age >= N bars
        
        Action depends on LATE_ENTRY_GATE_SHADOW:
          - Shadow=True (default): log LATE_ENTRY_SKIP, no block
          - Shadow=False: hard-skip the entry
        
        Safe off: ENABLE_LATE_ENTRY_GATE=0 disables all late-entry checks.
        
        Default True. Set via env ENABLE_LATE_ENTRY_GATE=0 to disable.
        """
        env_val = os.getenv("ENABLE_LATE_ENTRY_GATE")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_late_entry_gate', True))

    @property
    def LATE_ENTRY_GATE_SHADOW(self) -> bool:
        """Shadow mode for late-entry gate (log only, no block).
        
        When True (default), late-entry detection logs LATE_ENTRY_SKIP
        with action=shadow_late_entry_skip but does NOT block the entry.
        This allows collecting data before enabling hard-skips.
        
        When False, late entries are hard-skipped (entry blocked).
        
        Default True (shadow mode). Set LATE_ENTRY_GATE_SHADOW=0 to enable
        hard-skips after soak testing.
        
        See docs/late_entry_promote_checklist.md for promotion criteria.
        """
        env_val = os.getenv("LATE_ENTRY_GATE_SHADOW")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.late_entry_gate_shadow', True))

    @property
    def LATE_ENTRY_EXTENSION_THRESHOLD(self) -> float:
        """Extension ratio threshold for late-entry detection (longs).
        
        Measures how far price has extended from session open toward
        session high: (price - open) / (high - open).
        
        If extension >= threshold, the entry is flagged as late (chasing
        an already-extended move). Lower = more conservative.
        
        Default 0.75 (price 75%+ of the way from open to high).
        Set trading.late_entry_extension_threshold to adjust.
        """
        return float(self.manager.get('trading.late_entry_extension_threshold', 0.75))

    @property
    def LATE_ENTRY_VWAP_ATR_MULT(self) -> float:
        """ATR multiplier for VWAP chase detection (longs).
        
        A long entry above VWAP + k*ATR is flagged as late (chasing
        above fair value). Higher = more permissive.
        
        Default 1.0 (entry > VWAP + 1.0*ATR is late).
        Set trading.late_entry_vwap_atr_mult to adjust.
        """
        return float(self.manager.get('trading.late_entry_vwap_atr_mult', 1.0))

    @property
    def LATE_ENTRY_BARS_SINCE_IMPULSE(self) -> int:
        """Max bars since impulse/breakout for late-entry detection.
        
        If the signal's breakout/impulse occurred >= N bars ago, the
        entry is flagged as late (move has already played out).
        
        Default 5 bars. Set trading.late_entry_bars_since_impulse to adjust.
        """
        return int(self.manager.get('trading.late_entry_bars_since_impulse', 5))

    @property
    def ENABLE_MOVER_QUALITY_RELAX(self) -> bool:
        """Relax quality filters for mover-sourced symbols.
        
        When True, symbols from Yahoo day_gainers/day_losers/most_active
        bypass the SMA50/RS quality filters. They still respect:
          - Leveraged ETF blocklist (hard block)
          - MIN_MOVER_PRICE floor ($5 default)
          - MIN_MOVER_VOLUME floor (500k default)
          - Spread filter (1% max)
        
        Rationale: day-gainers ARE the momentum names; requiring them to
        also be above SMA50 with positive RS is circular — they're moving
        BECAUSE something changed today. Quality-for-swing is wrong for
        intraday momentum.
        
        Default True. Set trading.enable_mover_quality_relax: false to
        apply full quality filters to movers.
        """
        env_val = os.getenv("ENABLE_MOVER_QUALITY_RELAX")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_mover_quality_relax', True))

    @property
    def MIN_MOVER_PRICE(self) -> float:
        """Minimum price floor for mover-sourced symbols.
        
        Even with quality relax, reject sub-$5 names to avoid illiquid
        penny stocks that appear in Yahoo screeners. The bot's ATR-based
        sizing breaks down below $5 (1.5×ATR can be 20%+ of price).
        
        Default 5.0. Set trading.min_mover_price to adjust.
        """
        return float(self.manager.get('trading.min_mover_price', 5.0))

    @property
    def MIN_MOVER_VOLUME(self) -> int:
        """Minimum volume floor for mover-sourced symbols.
        
        Movers with <500k daily volume are too thin for intraday trades.
        The bot's typical position ($5-15k notional) needs liquidity to
        enter/exit without moving the tape.
        
        Default 500000. Set trading.min_mover_volume to adjust.
        """
        return int(self.manager.get('trading.min_mover_volume', 500000))

    @property
    def MIN_DOLLAR_VOLUME(self) -> float:
        """v-dollar-volume-floor-2026-09-17: minimum daily dollar-volume.

        Dollar-volume = price × average daily volume. This gate demotes
        micro-cap / lottery names (PURR, AEMD, IOVA-class) that clear
        price and volume floors individually but have thin dollar-volume.

        Example: $3 stock × 1M shares = $3M dollar-vol (may pass).
                 $50 stock × 2M shares = $100M dollar-vol (definitely pass).
                 $0.80 stock × 5M shares = $4M dollar-vol (borderline).

        Non-pinned symbols MUST clear this floor to occupy a watchlist
        slot. Pinned symbols bypass this gate (they're manually curated).

        Default $10M. This excludes most sub-$1B market-cap penny names
        while admitting mid-cap movers with genuine institutional flow.
        Set trading.min_dollar_volume to adjust."""
        return float(self.manager.get('trading.min_dollar_volume', 10_000_000))

    @property
    def DAY_TRADE_SIZE_MULTIPLIER(self) -> float:
        """Size multiplier for day-trade momentum entries.
        
        Applied on top of Kelly sizing. Day trades have shorter holding
        periods and more frequent trades; smaller size per trade keeps
        the portfolio heat manageable.
        
        Default 0.5 (half-size vs swing). Set trading.day_trade_size_multiplier
        to adjust.
        """
        return float(self.manager.get('trading.day_trade_size_multiplier', 0.5))

    @property
    def DAY_TRADE_FLATTEN_HOUR(self) -> int:
        """Hour (ET) by which day-trade positions should be flattened.
        
        Day-trade momentum is intraday by definition. Positions entered
        via this lane should close by EOD to avoid overnight gap risk.
        The exit manager will start trailing more aggressively after
        this hour and force-close by LATE_ENTRY_CUTOFF.
        
        Default 15 (3 PM ET). Set trading.day_trade_flatten_hour to adjust.
        """
        return int(self.manager.get('trading.day_trade_flatten_hour', 15))

    @property
    def MOMENTUM_RISK_OFF_SIZE_MULT(self) -> float:
        """Size multiplier for momentum lane when market context is risk_off.
        
        v-market-context-size-not-freeze-2026-09-10: instead of hard-blocking
        momentum longs on risk_off, reduce size. This lets the desk take
        high-quality setups even in adverse conditions, just smaller.
        
        Default 0.25 (quarter-size). Set trading.momentum_risk_off_size_mult.
        
        NOTE: When DAY_TRADE_HARD_SKIP_RISK_OFF is True (new default),
        this multiplier is NOT used — the strategy hard-skips instead.
        This multiplier only applies when DAY_TRADE_HARD_SKIP_RISK_OFF=False.
        """
        return float(self.manager.get('trading.momentum_risk_off_size_mult', 0.25))

    @property
    def DAY_TRADE_HARD_SKIP_RISK_OFF(self) -> bool:
        """v-day-trade-hard-skip-risk-off-2026-09-14: hard-skip day_trade_momentum
        in risk_off regime (same as ORB), instead of size-down.
        
        2026-09-14 RCA: strategy logged many `action=size_reduced
        reason=risk_off_not_blocked size_mult=0.25` then later entered when
        regime flipped mixed (GLW). Research/CoS: size-down is how weak
        risk_off path still feeds LIVE; want hard skip like ORB.
        
        When True (default): risk_off regime causes day_trade_momentum to
        return None (no signal, no entry), same as ORB's `risk_off_hard_skip`.
        The weak signal path that could later enter is eliminated.
        
        When False: preserve prior behavior — reduce size to
        MOMENTUM_RISK_OFF_SIZE_MULT (0.25x) but still generate signal.
        Use for A/B testing or rollback if hard-skip proves too restrictive.
        
        Default True. Set via env DAY_TRADE_HARD_SKIP_RISK_OFF=0 or
        trading.day_trade_hard_skip_risk_off: false in Config.yaml.
        """
        env_val = os.getenv("DAY_TRADE_HARD_SKIP_RISK_OFF")
        if env_val is not None:
            return env_val.lower() not in ("0", "false", "no", "off")
        return bool(self.manager.get('trading.day_trade_hard_skip_risk_off', True))

    @property
    def MOMENTUM_OPENING_30_SIZE_MULT(self) -> float:
        """Size multiplier for momentum lane during opening 30 minutes.
        
        Opening 30 is volatile chop. For momentum/breakout, we reduce size
        rather than blocking entirely — first 30 min breakouts CAN be valid,
        just riskier.
        
        Default 0.5 (half-size). Set trading.momentum_opening_30_size_mult.
        """
        return float(self.manager.get('trading.momentum_opening_30_size_mult', 0.5))

    @property
    def MOMENTUM_EXTREME_BLOCK_SPY_PCT(self) -> float:
        """SPY down % threshold for hard-blocking momentum longs.
        
        When SPY is down MORE than this AND VIX spikes (see below), the
        desk hard-blocks new longs entirely. This is the circuit breaker
        for extreme risk-off conditions.
        
        Default -1.5 (SPY down 1.5%+). Set trading.momentum_extreme_block_spy_pct.
        """
        return float(self.manager.get('trading.momentum_extreme_block_spy_pct', -1.5))

    @property
    def MOMENTUM_EXTREME_BLOCK_VIX_SPIKE(self) -> float:
        """VIX spike % threshold for hard-blocking momentum longs.
        
        When VIX is UP more than this AND SPY is down past the threshold,
        hard-block new momentum longs. This catches panic days.
        
        Default 15.0 (VIX up 15%+). Set trading.momentum_extreme_block_vix_spike.
        """
        return float(self.manager.get('trading.momentum_extreme_block_vix_spike', 15.0))

    @property
    def MOMENTUM_MIN_RS_VS_SPY(self) -> float:
        """Minimum relative strength vs SPY for momentum entries.
        
        Momentum entries require the symbol to be outperforming SPY on the
        day. This is the minimum delta (symbol_change_pct - spy_change_pct).
        
        Default 0.5 (symbol at least 0.5% above SPY). Set trading.momentum_min_rs_vs_spy.
        """
        return float(self.manager.get('trading.momentum_min_rs_vs_spy', 0.5))

    @property
    def ENABLE_PINNED_RS_SOFTEN(self) -> bool:
        """v-pinned-rs-soften-2026-09-17: softer RS threshold for pinned symbols.

        When True, symbols in PINNED_WATCHLIST use PINNED_MIN_RS_VS_SPY
        instead of MOMENTUM_MIN_RS_VS_SPY for day-trade momentum entry.

        Rationale: NVDA/TSLA/META-class liquid names often lag SPY intraday
        on rotation days but still produce high-quality setups. A 0.5% RS
        floor rejects them on weak_relative_strength even when the setup
        is valid. Pinned symbols have been manually curated for quality,
        so a softer RS floor (e.g. 0.25%) lets more of their setups through.

        Non-pinned movers (Yahoo day_gainers/losers fills) still use the
        standard MOMENTUM_MIN_RS_VS_SPY. This prevents junk names from
        sneaking in with weak momentum.

        SAFE OFF-PATH: Set ENABLE_PINNED_RS_SOFTEN=0 to disable.
        Default False (conservative). Set trading.enable_pinned_rs_soften: true
        to enable softer RS for pinned symbols only."""
        env_val = os.getenv("ENABLE_PINNED_RS_SOFTEN")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_pinned_rs_soften', False))

    @property
    def PINNED_MIN_RS_VS_SPY(self) -> float:
        """Softer RS threshold for PINNED_WATCHLIST symbols.

        Only used when ENABLE_PINNED_RS_SOFTEN is True. Pinned symbols
        use this (default 0.25) instead of MOMENTUM_MIN_RS_VS_SPY (0.5).

        The lower threshold lets mega-cap liquid names enter even when
        slightly lagging SPY on a rotation day. Combined with the volume
        and pattern gates, this admits ~20-30% more pinned setups without
        materially degrading quality (walk-forward validated on 60-day
        backtest with NVDA/TSLA/META/AMZN).

        Default 0.25. Set trading.pinned_min_rs_vs_spy to adjust."""
        return float(self.manager.get('trading.pinned_min_rs_vs_spy', 0.25))

    @property
    def ENABLE_SAME_BASIS_RS(self) -> bool:
        """v-same-basis-rs-2026-09-21: use same-basis RS calculation.

        When True (default), RS calculation uses symbol's netPercentChange
        from Schwab quotes — the same source as SPY uses via MarketIndicesCache.
        This ensures apples-to-apples comparison: both are day % change vs
        prior close.

        Prior bug: symbol used (bar_close - bar_open) / bar_open which is
        current bar % change, NOT day % change. SPY used netPercentChange
        (day vs prior close). This mismatch caused false weak_relative_strength
        skips in premarket when SPY had already moved +0.6% from prior close
        but the symbol's current bar was flat.

        When False: preserve legacy behavior (bar-based estimate). Use for
        A/B testing or rollback.

        SAFE OFF-PATH: Set ENABLE_SAME_BASIS_RS=0 to disable.
        Default True. Set trading.enable_same_basis_rs: false to disable."""
        env_val = os.getenv("ENABLE_SAME_BASIS_RS")
        if env_val is not None:
            return env_val.lower() not in ("0", "false", "no", "off")
        return bool(self.manager.get('trading.enable_same_basis_rs', True))

    @property
    def DAY_TRADE_HARD_SKIP_OFF_HOURS(self) -> bool:
        """v-daytrade-offhours-skip-2026-09-21: hard-skip day_trade_momentum
        and day_trade_momentum_short signals during off_hours.

        2026-09-21 RCA: during premarket/off_hours, the strategy was still
        generating signal_buy logs with time_of_day=off_hours. While the
        engine's market_hours gate blocks actual LIVE orders during off_hours,
        the signals themselves created noise (false weak_RS skips logged,
        RS drift as SPY moves pre-market, etc.).

        When True (default): day_trade_momentum returns None (no signal)
        if time_of_day == 'off_hours'. Shadow logging still works during
        RTH. This eliminates noisy pre-market signals with corrupt data.

        When False: preserve prior behavior — signals generated during
        off_hours (engine gate still blocks LIVE orders). Use for debugging
        or if pre-market shadow data is desired.

        SAFE OFF-PATH: Set DAY_TRADE_HARD_SKIP_OFF_HOURS=0 to disable.
        Default True. Set trading.day_trade_hard_skip_off_hours: false."""
        env_val = os.getenv("DAY_TRADE_HARD_SKIP_OFF_HOURS")
        if env_val is not None:
            return env_val.lower() not in ("0", "false", "no", "off")
        return bool(self.manager.get('trading.day_trade_hard_skip_off_hours', True))

    # ══════════════════════════════════════════════════════════════════════════
    # v-open30-cont-index-confirm-2026-09-24: Index confirm gate for opening_30
    # continuation longs. Research brief 2026-09-24 diagnosis.
    #
    # Predicate: Skip day_trade_momentum CONTINUATION longs when
    #   time_of_day == opening_30 unless:
    #   SPY change vs prior close ≥ 0 AND SPY last ≥ session VWAP
    #
    # Scope: continuation longs only in opening_30. Does NOT apply to:
    #   - pullback pattern
    #   - breakout pattern
    #   - post-opening_30 time windows (morning, midday, afternoon)
    #   - ORB, mean_rev, news, shorts
    #   - existing RSI≥70 veto (unchanged)
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def DT_OPEN30_CONT_INDEX_CONFIRM(self) -> bool:
        """v-open30-cont-index-confirm-2026-09-24: enable index confirm gate
        for opening_30 continuation longs.

        When True, the day_trade_momentum strategy evaluates the index confirm
        predicate for continuation longs during opening_30:
          - SPY change vs prior close ≥ 0 (SPY day-green)
          - SPY last ≥ session VWAP

        When the predicate FAILS:
          - If DT_OPEN30_CONT_INDEX_CONFIRM_LIVE_ENFORCE=False (default):
            Shadow mode — log would_skip but allow the trade.
          - If DT_OPEN30_CONT_INDEX_CONFIRM_LIVE_ENFORCE=True:
            Live enforcement — actually skip/block the entry.

        Shadow mode emits decision log with fields for counterfactual PF:
          would_skip, spy_chg_vs_prior_close, spy_last, spy_vwap, spy_vs_vwap,
          symbol, pattern, time_of_day, rsi, regime, entry_price

        SCOPE LIMITS (does NOT touch):
          - DAY_TRADE_LIVE_ENTRIES_ENABLED master flip
          - MU / HQGE / SPCX hands-off denylist
          - pullback pattern entries
          - time_of_day in {morning, midday, afternoon, late} after opening_30
          - ORB / mean_rev / news / day_trade_short lanes
          - existing RSI≥70 continuation hard-veto (unchanged)
          - opening_30 0.5× size_mult for signals that still pass

        SAFE OFF-PATH: Set DT_OPEN30_CONT_INDEX_CONFIRM=0 to disable.
        Default False (OFF). Set via env DT_OPEN30_CONT_INDEX_CONFIRM=1
        or trading.dt_open30_cont_index_confirm: true in Config.yaml."""
        env_val = os.getenv("DT_OPEN30_CONT_INDEX_CONFIRM")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.dt_open30_cont_index_confirm', False))

    @property
    def DT_OPEN30_CONT_INDEX_CONFIRM_LIVE_ENFORCE(self) -> bool:
        """v-open30-cont-index-confirm-2026-09-24: live enforcement for
        the opening_30 continuation index confirm gate.

        When True AND DT_OPEN30_CONT_INDEX_CONFIRM=True:
          - If the predicate FAILS (SPY day-red OR SPY < VWAP):
            The entry is BLOCKED with reason 'open30_cont_index_confirm_fail'.

        When False (default) AND DT_OPEN30_CONT_INDEX_CONFIRM=True:
          - Shadow mode: log would_skip but ALLOW the trade.
            This enables Research to compute counterfactual PF before
            promoting to LIVE enforcement.

        Requires DT_OPEN30_CONT_INDEX_CONFIRM=True to have any effect.

        SAFE OFF-PATH: keep False (default) until Research confirms
        shadow metrics pass Stage-A floors.
        Default False. Set via env DT_OPEN30_CONT_INDEX_CONFIRM_LIVE_ENFORCE=1
        or trading.dt_open30_cont_index_confirm_live_enforce: true."""
        env_val = os.getenv("DT_OPEN30_CONT_INDEX_CONFIRM_LIVE_ENFORCE")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get(
            'trading.dt_open30_cont_index_confirm_live_enforce', False
        ))

    # ══════════════════════════════════════════════════════════════════════════
    # v-open30-cont-rs-from-open-2026-09-24: RTH-open RS floor gate for
    # opening_30 continuation longs. Kiddo Alpaca tape reconstruction RCA.
    #
    # Prior-close RS can mask intraday deterioration:
    #   Symbol +3% vs prior close, +1% vs today's open
    #   SPY +2% vs today's open
    #   → Prior-close RS shows +1.5% (outperforming)
    #   → RTH-open RS shows -1.0% (actually underperforming today)
    #
    # This gate applies the SAME MOMENTUM_MIN_RS_VS_SPY threshold but
    # using RTH-open as the baseline instead of prior close.
    #
    # Scope: continuation + opening_30 only. Stacks AFTER index confirm.
    # ══════════════════════════════════════════════════════════════════════════

    @property
    def DT_OPEN30_CONT_RS_FROM_OPEN(self) -> bool:
        """v-open30-cont-rs-from-open-2026-09-24: enable RTH-open RS floor gate
        for opening_30 continuation longs.

        When True, after the index confirm gate passes, the day_trade_momentum
        strategy also checks RTH-open RS:
          - Calculate: (sym_last/sym_rth_open - 1) - (spy_last/spy_rth_open - 1)
          - Require: rs_from_open >= MOMENTUM_MIN_RS_VS_SPY

        This catches intraday deterioration that prior-close RS misses.

        When the predicate FAILS:
          - If DT_OPEN30_CONT_RS_FROM_OPEN_LIVE_ENFORCE=False (default):
            Shadow mode — log would_skip but allow the trade.
          - If DT_OPEN30_CONT_RS_FROM_OPEN_LIVE_ENFORCE=True:
            Live enforcement — actually skip/block the entry.

        SCOPE LIMITS: same as DT_OPEN30_CONT_INDEX_CONFIRM (continuation +
        opening_30 only). Does NOT replace prior-close RS gate (both run).

        SAFE OFF-PATH: Set DT_OPEN30_CONT_RS_FROM_OPEN=0 to disable.
        Default False (OFF). Set via env DT_OPEN30_CONT_RS_FROM_OPEN=1
        or trading.dt_open30_cont_rs_from_open: true in Config.yaml."""
        env_val = os.getenv("DT_OPEN30_CONT_RS_FROM_OPEN")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.dt_open30_cont_rs_from_open', False))

    @property
    def DT_OPEN30_CONT_RS_FROM_OPEN_LIVE_ENFORCE(self) -> bool:
        """v-open30-cont-rs-from-open-2026-09-24: live enforcement for
        the RTH-open RS floor gate.

        When True AND DT_OPEN30_CONT_RS_FROM_OPEN=True:
          - If rs_from_open < MOMENTUM_MIN_RS_VS_SPY:
            The entry is BLOCKED with reason 'open30_cont_rs_from_open_fail'.

        When False (default) AND DT_OPEN30_CONT_RS_FROM_OPEN=True:
          - Shadow mode: log would_skip but ALLOW the trade.

        Requires DT_OPEN30_CONT_RS_FROM_OPEN=True to have any effect.

        SAFE OFF-PATH: keep False (default) until Research confirms
        shadow metrics pass Stage-A floors.
        Default False. Set via env DT_OPEN30_CONT_RS_FROM_OPEN_LIVE_ENFORCE=1
        or trading.dt_open30_cont_rs_from_open_live_enforce: true."""
        env_val = os.getenv("DT_OPEN30_CONT_RS_FROM_OPEN_LIVE_ENFORCE")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get(
            'trading.dt_open30_cont_rs_from_open_live_enforce', False
        ))

    @property
    def MOMENTUM_MIN_VOLUME_RATIO(self) -> float:
        """Minimum volume ratio (vs 20-bar avg) for momentum entries.
        
        Volume surge confirms institutional interest. Breakouts without
        volume are more likely to fail.
        
        Default 1.5 (50% above average). Set trading.momentum_min_volume_ratio.
        """
        return float(self.manager.get('trading.momentum_min_volume_ratio', 1.5))

    @property
    def MOMENTUM_NEWS_OPTIONAL(self) -> bool:
        """Whether news is optional for momentum entries.
        
        When True (default), the momentum lane does NOT require fresh news
        articles. News is an optional confirmation that can bump size, not
        a gate that vetoes entries.
        
        This differs from the news strategy which requires fresh articles.
        Momentum is price-action-driven, not news-driven.
        
        Default True. Set trading.momentum_news_optional: false to require news.
        """
        return bool(self.manager.get('trading.momentum_news_optional', True))

    # ──────────────────────────────────────────────────────────────────
    # v-momentum-stage-a-2026-09-10: Stage-A promotion criteria for the
    # day-trade momentum desk. These are RESEARCH-LOCKED floors — do NOT
    # relax without walk-forward validation evidence.
    #
    # Promote to full capital allocation ONLY when ALL pass:
    #   n ≥ 150 closed day-trades across ≥ 10 sessions
    #   PF ≥ 1.30 after fees + slippage
    #   Win rate ≥ 48% (scratches out of rate but counted in n)
    #   Expectancy ≥ +0.05 R AND positive small $/day edge
    #   Max DD ≤ 6% of allocated
    #   Max losing day ≤ 2.0 R
    #   Throughput 3-12/session is product report only (not a gate)
    #
    # HARD VETO for promotion:
    #   - Overnight holds without explicit flag
    #   - Capital scale before Stage A green
    # ──────────────────────────────────────────────────────────────────

    @property
    def MOMENTUM_STAGE_A_MIN_TRADES(self) -> int:
        """Minimum closed day-trades for Stage-A promotion.
        
        RESEARCH-LOCKED: n ≥ 150 closed trades required before promoting
        to full capital. This ensures statistical significance.
        
        Default 150. DO NOT LOWER without walk-forward evidence.
        """
        return int(self.manager.get('trading.momentum_stage_a_min_trades', 150))

    @property
    def MOMENTUM_STAGE_A_MIN_SESSIONS(self) -> int:
        """Minimum trading sessions for Stage-A promotion.
        
        RESEARCH-LOCKED: ≥ 10 sessions ensures the strategy has been
        tested across different market conditions, not just one regime.
        
        Default 10. DO NOT LOWER without walk-forward evidence.
        """
        return int(self.manager.get('trading.momentum_stage_a_min_sessions', 10))

    @property
    def MOMENTUM_STAGE_A_MIN_PF(self) -> float:
        """Minimum Profit Factor for Stage-A promotion.
        
        RESEARCH-LOCKED: PF ≥ 1.30 after fees + slippage. A PF below this
        means the strategy is barely profitable and risky to scale.
        
        Default 1.30. DO NOT LOWER without walk-forward evidence.
        """
        return float(self.manager.get('trading.momentum_stage_a_min_pf', 1.30))

    @property
    def MOMENTUM_STAGE_A_MIN_WIN_RATE(self) -> float:
        """Minimum win rate for Stage-A promotion.
        
        RESEARCH-LOCKED: Win rate ≥ 48%. Scratches (breakeven exits) are
        counted OUT of win rate but IN the trade count n.
        
        At 2:1 R:R, 48% win rate gives expectancy ~0.44 R/trade.
        
        Default 0.48. DO NOT LOWER without walk-forward evidence.
        """
        return float(self.manager.get('trading.momentum_stage_a_min_win_rate', 0.48))

    @property
    def MOMENTUM_STAGE_A_MIN_EXPECTANCY_R(self) -> float:
        """Minimum expectancy in R-multiples for Stage-A promotion.
        
        RESEARCH-LOCKED: Expectancy ≥ +0.05 R per trade AND positive
        small $/day edge. This ensures the strategy has edge even after
        accounting for variance.
        
        Default 0.05. DO NOT LOWER without walk-forward evidence.
        """
        return float(self.manager.get('trading.momentum_stage_a_min_expectancy_r', 0.05))

    @property
    def MOMENTUM_STAGE_A_MAX_DD_PCT(self) -> float:
        """Maximum drawdown % for Stage-A promotion.
        
        RESEARCH-LOCKED: Max DD ≤ 6% of allocated capital during the
        Stage-A evaluation period. Higher DD indicates poor risk management
        or strategy flaws.
        
        Default 0.06 (6%). DO NOT RAISE without walk-forward evidence.
        """
        return float(self.manager.get('trading.momentum_stage_a_max_dd_pct', 0.06))

    @property
    def MOMENTUM_STAGE_A_MAX_LOSING_DAY_R(self) -> float:
        """Maximum losing day in R-multiples for Stage-A promotion.
        
        RESEARCH-LOCKED: Max losing day ≤ 2.0 R. A single day losing more
        than 2R indicates poor position sizing or lack of circuit breakers.
        
        Default 2.0. DO NOT RAISE without walk-forward evidence.
        """
        return float(self.manager.get('trading.momentum_stage_a_max_losing_day_r', 2.0))

    @property
    def MOMENTUM_STAGE_A_THROUGHPUT_MIN(self) -> int:
        """Minimum trades per session for Stage-A (PRODUCT REPORT ONLY).
        
        NOT A PROMOTION GATE — throughput 3-12/session is informational.
        Too few trades = desk is too selective; too many = overtrading.
        
        Default 3. This is for reporting, not gating.
        """
        return int(self.manager.get('trading.momentum_stage_a_throughput_min', 3))

    @property
    def MOMENTUM_STAGE_A_THROUGHPUT_MAX(self) -> int:
        """Maximum trades per session for Stage-A (PRODUCT REPORT ONLY).
        
        NOT A PROMOTION GATE — throughput 3-12/session is informational.
        
        Default 12. This is for reporting, not gating.
        """
        return int(self.manager.get('trading.momentum_stage_a_throughput_max', 12))

    @property
    def MOMENTUM_ALLOW_OVERNIGHT_HOLD(self) -> bool:
        """Allow overnight holds for day-trade momentum positions.
        
        HARD VETO FOR PROMOTION: Overnight holds without this flag
        explicitly set to True will block Stage-A promotion.
        
        Default False. Day-trade positions should flatten by EOD.
        Set True ONLY if you explicitly want swing-style holds.
        """
        return bool(self.manager.get('trading.momentum_allow_overnight_hold', False))

    @property
    def MOMENTUM_STAGE_A_PROMOTED(self) -> bool:
        """Whether the momentum desk has passed Stage-A promotion.
        
        HARD VETO: Capital scale before Stage A green is forbidden.
        This flag should only be set True AFTER all promotion criteria
        are met and validated by research.
        
        Default False. Flip to True only after Stage-A validation.
        """
        return bool(self.manager.get('trading.momentum_stage_a_promoted', False))

    # ──────────────────────────────────────────────────────────────────
    # v-shadow-veto-2026-09-10: shadow (log-only) veto for risky setups.
    # Instrumentation for later promotion scoring. LIVE Schwab fills only
    # count for scorecard later — this is shadow first.
    # ──────────────────────────────────────────────────────────────────

    @property
    def ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF(self) -> bool:
        """v-shadow-veto-2026-09-10: shadow veto for continuation pattern
        when RSI >= 70 AND risk_off regime.
        
        When True, the day-trade momentum strategy logs would-be entries
        that match the dangerous pattern (continuation + RSI>=70 + risk_off)
        without placing orders. This is instrumentation for later promotion
        scoring — LIVE Schwab fills only count for scorecard later.
        
        Pattern rationale: continuation into overbought on a risk-off day
        is the classic failed breakout setup (exhaustion gap). Shadow-first
        to gather evidence before promoting to a hard veto.
        
        Default True. Set trading.enable_shadow_veto_continuation_riskoff: false
        to disable shadow logging.
        """
        env_val = os.getenv("ENABLE_SHADOW_VETO_CONTINUATION_RISKOFF")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_shadow_veto_continuation_riskoff', True))

    @property
    def SHADOW_VETO_RSI_THRESHOLD(self) -> float:
        """RSI threshold for shadow veto on continuation + risk_off.
        
        Continuation entries with RSI >= this AND risk_off regime get
        shadow-logged (not placed). 70 is the classic overbought level.
        
        Default 70.0. Set trading.shadow_veto_rsi_threshold to adjust.
        """
        return float(self.manager.get('trading.shadow_veto_rsi_threshold', 70.0))

    @property
    def ENABLE_HARD_VETO_CONTINUATION_RSI70(self) -> bool:
        """v-hard-veto-rsi70-2026-09-14: hard veto for continuation + RSI>=70
        in ALL regimes (not just risk_off).
        
        RCA 2026-09-14 FTFT LIVE loss: continuation entry RSI 74.31,
        regime=mixed. The existing shadow veto only fires on risk_off,
        so mixed overbought continuation still placed LIVE and lost.
        
        When True (default): if entry_pattern==continuation AND
        rsi >= SHADOW_VETO_RSI_THRESHOLD, return None (skip order)
        regardless of regime. Logs strategy_decision action=hard_veto
        with reason continuation_overbought_all_regimes.
        
        When False: preserves prior behavior (shadow veto risk_off-only
        path continues to work as before).
        
        Default True (safe on-path = entries blocked when pattern matches).
        Set trading.enable_hard_veto_continuation_rsi70: false or
        ENABLE_HARD_VETO_CONTINUATION_RSI70=0 to disable.
        """
        env_val = os.getenv("ENABLE_HARD_VETO_CONTINUATION_RSI70")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_hard_veto_continuation_rsi70', True))

    @property
    def ENABLE_HARD_VETO_BREAKOUT_RSI70(self) -> bool:
        """v-rsi-breakout-veto-2026-09-21: hard veto for breakout + RSI>=70
        in ALL regimes.
        
        RCA 2026-09-21 META LIVE churn: breakout entry RSI 86.09 filled
        then immediately closed by open_desk_rsi_extreme_overbought.
        Entering overbought breakout = chasing exhaustion gap.
        
        When True (default): if entry_pattern==breakout AND
        rsi >= SHADOW_VETO_RSI_THRESHOLD, return None (skip order)
        regardless of regime. Logs strategy_decision action=hard_veto
        with reason breakout_overbought_all_regimes.
        
        When False: legacy (engine veto must catch it via
        DAY_TRADE_RSI_HIGH_ENTRY_VETO_ENABLED, but that requires
        reasoning to include 'rsi' key — fixed in same commit).
        
        Default True (safe on-path = entries blocked when pattern matches).
        Set trading.enable_hard_veto_breakout_rsi70: false or
        ENABLE_HARD_VETO_BREAKOUT_RSI70=0 to disable.
        """
        env_val = os.getenv("ENABLE_HARD_VETO_BREAKOUT_RSI70")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_hard_veto_breakout_rsi70', True))

    # ──────────────────────────────────────────────────────────────────
    # v-feature-snapshot-config-2026-09-09: decision snapshot feature flags.
    # Moved from module constants in core/decision_snapshot.py to Config
    # for runtime configurability via env vars or Config.yaml.
    # ──────────────────────────────────────────────────────────────────

    @property
    def FEATURE_SNAPSHOT_LOGGING(self) -> bool:
        """Enable snapshot logging for ML training data collection.
        
        When True, DecisionSnapshot objects are built and persisted to
        the bot_decision_snapshots table on every strategy decision.
        Zero inference cost — just data collection.
        
        Default True. Set via env FEATURE_SNAPSHOT_LOGGING=0 or
        trading.feature_snapshot_logging: false in Config.yaml.
        """
        env_val = os.getenv("FEATURE_SNAPSHOT_LOGGING")
        if env_val is not None:
            return env_val.lower() not in ("0", "false", "no", "off")
        return bool(self.manager.get('trading.feature_snapshot_logging', True))

    @property
    def FEATURE_SNAPSHOT_INFERENCE(self) -> bool:
        """Enable snapshot-based inference for trade decisions.
        
        When True, the bot uses trained models on DecisionSnapshot
        features to influence sizing or gate decisions. Requires a
        trained model checkpoint at SNAPSHOT_MODEL_PATH.
        
        DANGER: Only enable after sufficient training data and
        walk-forward validation. Default False — logging is on,
        inference is off.
        
        Set via env FEATURE_SNAPSHOT_INFERENCE=1 or
        trading.feature_snapshot_inference: true in Config.yaml.
        """
        env_val = os.getenv("FEATURE_SNAPSHOT_INFERENCE")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.feature_snapshot_inference', False))

    # ──────────────────────────────────────────────────────────────────
    # v-orb-prototype-2026-09-10: ORB (Opening Range Breakout) + volatility
    # contraction + relative volume (RVOL) prototype strategy.
    #
    # Hari APPROVED order: #1 ORB+contraction+RVOL → then #2 RVOL continuation
    # → #3 Gao late-day. This is ONLY #1.
    #
    # CRITICAL Research+CoS lock: skip ORB ENTIRELY in risk_off (NOT size-down).
    # If regime is risk_off, do not signal/enter ORB at all.
    #
    # Stage-A promotion floors (LOCKED — do NOT loosen):
    #   n>=150 trades, >=10 sessions, PF>=1.30, WR>=48%, exp>=+0.05R,
    #   DD<=6%, max losing day<=2R.
    # ──────────────────────────────────────────────────────────────────

    @property
    def ENABLE_ORB_STRATEGY(self) -> bool:
        """Enable the ORB + contraction + RVOL prototype strategy.
        
        v-orb-prototype-2026-09-10: Hari APPROVED product call.
        
        When True, the ORBContractionRVOLStrategy is activated and can
        generate signals for sim/shadow analysis. LIVE order placement
        is controlled separately by ORB_LIVE_ENTRIES_ENABLED.
        
        CRITICAL: This strategy HARD-SKIPS when regime is risk_off.
        Unlike momentum which reduces size on risk_off, ORB does NOT
        signal at all — ORB is an opening-range directional bet that
        doesn't make sense when the market is in panic mode.
        
        Default False. Enable via env ENABLE_ORB_STRATEGY=1 or
        trading.enable_orb_strategy: true in Config.yaml.
        """
        env_val = os.getenv("ENABLE_ORB_STRATEGY")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_orb_strategy', False))

    @property
    def ORB_LIVE_ENTRIES_ENABLED(self) -> bool:
        """Master switch for LIVE ORB entries.
        
        When False (default), the ORB strategy generates signals for
        sim/shadow analysis but BLOCKS actual LIVE order placement.
        
        CRITICAL: Keep default False until Stage-A validation:
          n>=150 trades, >=10 sessions, PF>=1.30, WR>=48%, exp>=+0.05R,
          DD<=6%, max losing day<=2R.
        
        Default False. Set via env ORB_LIVE_ENTRIES_ENABLED=1 or
        trading.orb_live_entries_enabled: true in Config.yaml.
        """
        env_val = os.getenv("ORB_LIVE_ENTRIES_ENABLED")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.orb_live_entries_enabled', False))

    @property
    def ORB_SIM_SHADOW_ENABLED(self) -> bool:
        """Enable ORB strategy for sim/shadow soak testing.
        
        When True (default), ORB signals are generated and logged for
        sim/shadow analysis even when ORB_LIVE_ENTRIES_ENABLED=False.
        This allows paper testing and collecting performance data
        before promoting to LIVE.
        
        Default True. Set trading.orb_sim_shadow_enabled: false to
        disable completely (including sim/shadow).
        """
        env_val = os.getenv("ORB_SIM_SHADOW_ENABLED")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.orb_sim_shadow_enabled', True))

    @property
    def ORB_OPENING_RANGE_MINUTES(self) -> int:
        """Duration of the opening range window in minutes.
        
        The opening range is the high/low of the first N minutes after
        market open (09:30 ET). Classic ORB uses 15 or 30 minutes.
        
        Default 15 minutes. Set trading.orb_opening_range_minutes to
        adjust (common values: 5, 15, 30).
        """
        return int(self.manager.get('trading.orb_opening_range_minutes', 15))

    @property
    def ORB_MIN_CONTRACTION_PCT(self) -> float:
        """Minimum volatility contraction percentage for ORB entry.
        
        ORB works best after a volatility squeeze (contraction) followed
        by an expansion (breakout). This is the minimum contraction %
        vs the N-bar ATR before the strategy considers a setup valid.
        
        Calculation: (ATR_N - current_range) / ATR_N >= this threshold
        
        Default 0.20 (20% contraction). Higher = more selective, fewer
        signals; lower = more signals, less filtered.
        """
        return float(self.manager.get('trading.orb_min_contraction_pct', 0.20))

    @property
    def ORB_MIN_RVOL(self) -> float:
        """Minimum relative volume (RVOL) for ORB entry.
        
        RVOL = current_volume / average_volume. ORB breakouts need volume
        confirmation to distinguish real moves from fakeouts.
        
        Default 1.2 (20% above average). Set trading.orb_min_rvol to adjust.
        """
        return float(self.manager.get('trading.orb_min_rvol', 1.2))

    @property
    def ORB_PRIMARY_SYMBOLS(self) -> list:
        """Primary symbols for ORB strategy (SPY/QQQ first per Hari).
        
        Hari instruction: prefer SPY/QQQ first as primary symbols.
        Liquid index ETFs have clean ORB setups with minimal spread/slippage.
        
        Default ['SPY', 'QQQ']. Extend via trading.orb_primary_symbols list.
        """
        default = ['SPY', 'QQQ']
        configured = self.manager.get('trading.orb_primary_symbols', None)
        if configured and isinstance(configured, list):
            return configured
        return default

    @property
    def ORB_ATR_STOP_MULTIPLIER(self) -> float:
        """ATR multiplier for ORB stop-loss distance.
        
        ORB stops are typically tighter than swing trades — the opening
        range itself provides a natural stop level (entry near the range
        boundary, stop at the opposite boundary or slightly beyond).
        
        Default 1.0 (stop at 1.0x ATR from entry). Set trading.orb_atr_stop_multiplier.
        """
        return float(self.manager.get('trading.orb_atr_stop_multiplier', 1.0))

    @property
    def ORB_REWARD_RISK_RATIO(self) -> float:
        """Reward:Risk ratio for ORB take-profit.
        
        Default 2.0 (2:1 R:R). Tighter than swing trades because ORB is
        an intraday pattern that plays out quickly.
        """
        return float(self.manager.get('trading.orb_reward_risk_ratio', 2.0))

    @property
    def ORB_MAX_ENTRY_MINUTES_AFTER_OPEN(self) -> int:
        """Maximum minutes after market open to take ORB entries.
        
        ORB is an opening-range strategy — entries taken too late lose
        the edge (the range has already resolved). This gate prevents
        chasing late setups.
        
        Default 60 (1 hour after open, i.e., by 10:30 ET). Set
        trading.orb_max_entry_minutes_after_open to adjust.
        """
        return int(self.manager.get('trading.orb_max_entry_minutes_after_open', 60))

    @property
    def ORB_FLATTEN_BY_HOUR(self) -> int:
        """Hour (ET) by which ORB positions should be flattened.
        
        ORB is an intraday pattern. Positions should close by EOD to
        avoid overnight gap risk. The exit manager will trail more
        aggressively after this hour.
        
        Default 15 (3 PM ET). Set trading.orb_flatten_by_hour to adjust.
        """
        return int(self.manager.get('trading.orb_flatten_by_hour', 15))

    @property
    def ORB_SIZE_MULTIPLIER(self) -> float:
        """Size multiplier for ORB entries.
        
        Applied on top of Kelly sizing. During prototype phase, use
        smaller size until Stage-A validation completes.
        
        Default 0.5 (half-size vs swing). Set trading.orb_size_multiplier.
        """
        return float(self.manager.get('trading.orb_size_multiplier', 0.5))

    # ──────────────────────────────────────────────────────────────────
    # v-active-open-desk-2026-09-14: continuous monitor for open trades.
    # RCA FTFT 2026-09-14: bot set bracket + hard stop then idled. Hari:
    # must continuously monitor ALL managed open trades for sentiment/
    # regime/indicators and take proactive action (not fire-and-forget).
    #
    # PR1 SHADOW ONLY: logs WOULD_TIGHTEN / WOULD_CANCEL_REPLACE /
    # WOULD_EXIT with reason + symbol + suggested levels. No broker
    # calls, no order mutations.
    # ──────────────────────────────────────────────────────────────────

    @property
    def ENABLE_ACTIVE_OPEN_DESK(self) -> bool:
        """Master switch for the Active Open Desk continuous position monitor.

        v-active-open-desk-2026-09-14: when True (default), spawns a supervised
        task that iterates managed_by_bot open positions in parallel, evaluating
        sentiment, regime, and indicator signals for proactive exit/tighten
        decisions.

        Default True + ACTIVE_OPEN_DESK_SHADOW=True enables evidence collection
        (shadow logs WOULD_TIGHTEN / WOULD_EXIT without broker calls). This is
        the recommended soak configuration.

        SAFE OFF-PATH: To disable entirely and preserve today's bracket +
        hard-stop behavior unchanged, set:
          - env: ENABLE_ACTIVE_OPEN_DESK=0
          - yaml: trading.enable_active_open_desk: false

        When disabled, the desk task is not started at all — zero behavior
        change from pre-PR1 baseline.
        """
        env_val = os.getenv("ENABLE_ACTIVE_OPEN_DESK")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_active_open_desk', True))

    @property
    def ACTIVE_OPEN_DESK_SHADOW(self) -> bool:
        """Shadow mode for the Active Open Desk (log-only, no order mutations).

        v-active-open-desk-2026-09-14: when True AND ENABLE_ACTIVE_OPEN_DESK
        is True, the desk logs structured strategy_decision-style events:
          - WOULD_TIGHTEN: suggests tighter stop level
          - WOULD_CANCEL_REPLACE: suggests replacing bracket
          - WOULD_EXIT: suggests immediate exit

        All with reason + symbol + suggested levels. **No broker calls,
        no cancel-replace, no market exit.**

        When False: desk can execute actual order modifications (PR2+).
        
        Default True (shadow mode = safe). Set via env
        ACTIVE_OPEN_DESK_SHADOW=0 or trading.active_open_desk_shadow: false
        to enable live actuators (future PR).
        """
        env_val = os.getenv("ACTIVE_OPEN_DESK_SHADOW")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.active_open_desk_shadow', True))

    @property
    def ACTIVE_OPEN_DESK_INTERVAL_SEC(self) -> float:
        """Interval (seconds) between Active Open Desk evaluation cycles.

        v-active-open-desk-2026-09-14: the desk polls all managed positions
        on this cadence. Faster than the 30s main loop for quicker reaction.

        Default 5.0 seconds. Set via env ACTIVE_OPEN_DESK_INTERVAL_SEC=10
        or trading.active_open_desk_interval_sec: 10.0 in Config.yaml.
        """
        env_val = os.getenv("ACTIVE_OPEN_DESK_INTERVAL_SEC")
        if env_val is not None:
            try:
                return float(env_val)
            except ValueError:
                pass
        return float(self.manager.get('trading.active_open_desk_interval_sec', 5.0))

    @property
    def ENABLE_OPEN_DESK_RSI_EXTREME_EXIT(self) -> bool:
        """v-open-desk-rsi-exit-2026-09-17: execute REAL exit for day-trade
        RSI extreme overbought/oversold conditions from the Active Open Desk.

        P0 RCA 2026-09-17 (XE): day_trade breakout @16.31×334 @09:45 ET →
        open-desk spam WOULD_EXIT rsi_extreme_overbought_90+ from ~09:51
        (pnl_r −0.5→−0.9) but suggest-only → stop fill 234@16.05 @09:55 →
        broker_flat ghost on remaining 100.

        When True (default): for day-trade positions (is_day_trade=True in
        reasoning), when Active Open Desk detects rsi_extreme_overbought (RSI
        >= 90 for longs) or rsi_extreme_oversold (RSI <= 20 for shorts), the
        desk calls supervised close immediately instead of logging WOULD_EXIT.

        This is scoped to day trades ONLY. LT/hands-off positions are never
        touched — HANDS_OFF_DENYLIST (MU, HQGE, SPCX) and is_long_term
        positions skip the desk entirely.

        SAFE OFF-PATH: Set ENABLE_OPEN_DESK_RSI_EXTREME_EXIT=0 to disable
        real exits and preserve shadow-only WOULD_EXIT logging for RSI
        extreme conditions. Other open desk shadow actions unchanged.

        Default True. Set via env ENABLE_OPEN_DESK_RSI_EXTREME_EXIT=0 or
        trading.enable_open_desk_rsi_extreme_exit: false to disable.
        """
        env_val = os.getenv("ENABLE_OPEN_DESK_RSI_EXTREME_EXIT")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_open_desk_rsi_extreme_exit', True))

    # ──────────────────────────────────────────────────────────────────
    # v-theme-shock-logger-2026-09-14: ThemeShock shadow logger
    # Structured logging for multi-sentiment desk theme matching.
    # See docs/research/2026-09-14-multisenti-stage-a.md for design.
    # ──────────────────────────────────────────────────────────────────

    @property
    def ENABLE_THEME_SHOCK_LOGGER(self) -> bool:
        """Master switch for the ThemeShock shadow logger.

        v-theme-shock-logger-2026-09-14: when True (default), enables
        theme-based news classification and shadow action logging for
        multi-sentiment desk (Fed/earnings/news/CEO-AI shocks).

        Shadow-only behavior: logs would-hard-skip / would-size-down /
        would-exit / alert-only events WITHOUT executing broker calls
        or mutating orders. Collects evidence for Stage A promotion.

        SAFE OFF-PATH: Set ENABLE_THEME_SHOCK_LOGGER=0 to disable
        entirely. No theme matching, no shadow logs.

        Respects HANDS_OFF_DENYLIST (MU, HQGE, SPCX) — theme hits on
        these symbols only emit alert_only, never action shadows.
        """
        env_val = os.getenv("ENABLE_THEME_SHOCK_LOGGER")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_theme_shock_logger', True))

    @property
    def ENABLE_THEME_HARD_SKIP_HEADLINE_ONLY(self) -> bool:
        """Stage A false-positive reduction: headline-only matching for hard_skip.
        
        v-themeshock-hygiene-2026-09-14: when True (default), theme matches
        for hard_skip_entries action only fire if the keyword hits the
        headline, NOT summary. This prevents FPs like:
        
          Headline: "Crude Oil Jumps 3%; Corning Shares Move Lower"
          Summary: "...anthropic research suggests AI slowdown..."
          → FP: ai_compute theme triggers hard_skip for NVDA
        
        With headline-only=True, the summary "anthropic" hit is ignored
        for hard_skip actions. Other actions (size_down, thesis_exit,
        alert_only) still match headline+summary.
        
        CoS greenlight Stage A: reduces noise while collecting evidence.
        No LIVE actuators; shadow-only logging.
        
        SAFE OFF-PATH: Set ENABLE_THEME_HARD_SKIP_HEADLINE_ONLY=0 to
        restore original headline+summary matching for all actions.
        """
        env_val = os.getenv("ENABLE_THEME_HARD_SKIP_HEADLINE_ONLY")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_theme_hard_skip_headline_only', True))

    @property
    def ENABLE_THEME_FANOUT_REQUIRE_SYMBOL_OVERLAP(self) -> bool:
        """Fanout symbol overlap: only emit for symbols in news item.
        
        v-themeshock-hygiene-2026-09-14: when True, process_news_item_from_publish
        only emits theme events for symbols that appear in BOTH:
          1. The theme's basket/watch_only list
          2. The news item's symbols (item.symbol or item.symbols)
        
        Example: An "anthropic" headline for symbols=["AAPL"] would NOT
        emit hard_skip for NVDA, even though NVDA is in ai_compute basket.
        
        When False (default), after a keyword match we emit for ALL symbols
        in the theme basket/watch_only, regardless of the news item's symbols.
        This is the original behavior.
        
        Stage A: default False to collect baseline data. Enable True only
        after evidence shows symbol-overlap reduces FPs without missing TPs.
        
        SAFE OFF-PATH: Default False preserves original fan-out behavior.
        """
        env_val = os.getenv("ENABLE_THEME_FANOUT_REQUIRE_SYMBOL_OVERLAP")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_theme_fanout_require_symbol_overlap', False))

    @property
    def THEME_CONFIG_PATH(self) -> str:
        """Path to theme configuration directory.

        v-theme-shock-logger-2026-09-14: directory containing theme
        YAML/JSON files. Research owns these files; Engine loads them.
        Default: data/themes/
        """
        return self.manager.get('trading.theme_config_path', 'data/themes')

    # ──────────────────────────────────────────────────────────────────
    # v-gpu-news-critic-2026-09-14: GPU News Critic shadow logger
    # GPU-accelerated news scoring for theme classification and
    # contamination detection. See docs/research/2026-09-14-gpu-sense-stage-a.md
    # ──────────────────────────────────────────────────────────────────

    @property
    def ENABLE_GPU_NEWS_CRITIC(self) -> bool:
        """Master switch for the GPU News Critic.

        v-gpu-news-critic-2026-09-14: when True (default), enables GPU-
        accelerated news scoring using sentence-transformers/all-MiniLM-L6-v2
        for theme classification and contamination detection.

        Shadow-only behavior: logs WOULD_SUPPRESS_HARD_SKIP when contamination
        detected or low theme_prob for keyword-matched theme. Does NOT execute
        broker calls or mutate orders.

        Model: ~22M parameters, ~90MB VRAM, p95 inference <100ms on RTX 3090.
        Leaves >23GB headroom for existing trading FFN.

        Graceful degradation:
        - CUDA unavailable: falls back to CPU inference (slower but functional)
        - Model download fails: returns no-op cards with skip_reason
        - Inference error: catches exception, logs, returns no-op card

        SAFE OFF-PATH: Set ENABLE_GPU_NEWS_CRITIC=0 to disable entirely.
        No model loading, no GPU inference, no shadow logs.

        Respects HANDS_OFF_DENYLIST (MU, HQGE, SPCX) — no actions for these.
        """
        env_val = os.getenv("ENABLE_GPU_NEWS_CRITIC")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_gpu_news_critic', True))

    @property
    def GPU_NEWS_CRITIC_SHADOW(self) -> bool:
        """Shadow mode for the GPU News Critic (log-only, no live actions).

        v-gpu-news-critic-2026-09-14: when True (default) AND ENABLE_GPU_NEWS_CRITIC
        is True, the critic logs structured events with action=WOULD_SUPPRESS_HARD_SKIP
        but does NOT actually suppress the hard_skip action.

        When False: critic can gate keyword hard_skip (requires THEME_HARD_SKIP_REQUIRE_GPU).

        Default True (shadow mode = safe). Set via env GPU_NEWS_CRITIC_SHADOW=0
        or trading.gpu_news_critic_shadow: false to enable live gating (future PR).
        """
        env_val = os.getenv("GPU_NEWS_CRITIC_SHADOW")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.gpu_news_critic_shadow', True))

    @property
    def THEME_HARD_SKIP_REQUIRE_GPU(self) -> bool:
        """Require GPU critic approval for keyword hard_skip actions.

        v-gpu-news-critic-2026-09-14: when True, keyword ThemeShock hard_skip
        actions are gated by GPU critic approval. If GPU critic says
        WOULD_SUPPRESS_HARD_SKIP, the hard_skip is converted to alert_only.

        Default False (Stage A data collection). Do NOT flip to True until
        Stage A promotion gates are met:
        - n >= 80 gated decisions
        - Precision >= 65%
        - FP rate on contamination set <= 5%
        - Recall on true positives >= 80%

        SAFE OFF-PATH: Keep False to preserve current keyword-only hard_skip
        behavior. Set True only after Stage A metrics are green.
        """
        env_val = os.getenv("THEME_HARD_SKIP_REQUIRE_GPU")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.theme_hard_skip_require_gpu', False))

    @property
    def GPU_CRITIC_CONTAM_LEXICON_ENABLE(self) -> bool:
        """Enable lexicon-based contamination detection in GPU News Critic.

        v-book-b-contam-lexicon-2026-09-17: when True, applies expanded
        contamination triggers (Corning/GLW, oil crisis, commodity wraps)
        inside _compute_contamination_risk. Bumps contamination_risk to >=0.65
        on headline/summary text match UNLESS headline also has AI_COMPUTE_TRIGGERS.

        Fixes Book B contam_fp (f59b23116fcb, 9579cf0d1b4f, 2d83a5133335) where
        MiniLM flat softmax tipped ai_compute on Corning/oil wraps with contam=0.

        Default False (shadow-only, safe off-path). Enable via env
        GPU_CRITIC_CONTAM_LEXICON_ENABLE=1 or trading.gpu_critic_contam_lexicon_enable: true.
        """
        env_val = os.getenv("GPU_CRITIC_CONTAM_LEXICON_ENABLE")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.gpu_critic_contam_lexicon_enable', False))

    @property
    def GPU_CRITIC_CONTAM_LEXICON(self) -> list:
        """Lexicon tokens for contamination detection in GPU News Critic.

        v-book-b-contam-lexicon-2026-09-17: list of tokens that trigger
        contamination_risk bump when found in headline/summary text.
        Match is case-insensitive on headline/summary text only (NOT symbols_seen).

        Default includes: corning, glw, nyse:glw, motor oil, oil crisis, crude oil,
        petroleum, opec, wti, brent, and peer tokens (cohr, cien, aaoi) when
        co-mentioned with glass/optical without AI headline triggers.

        Override via env GPU_CRITIC_CONTAM_LEXICON (comma-separated) or
        trading.gpu_critic_contam_lexicon (list) in Config.yaml.
        """
        env_val = os.getenv("GPU_CRITIC_CONTAM_LEXICON")
        if env_val is not None:
            return [t.strip().lower() for t in env_val.split(",") if t.strip()]
        default_lexicon = [
            "corning", "glw", "nyse:glw",
            "motor oil", "oil crisis", "crude oil", "petroleum", "opec", "wti", "brent",
            "cohr", "cien", "aaoi",
        ]
        return self.manager.get('trading.gpu_critic_contam_lexicon', default_lexicon)

    @property
    def ENABLE_ALPACA_NEWS_BUS(self) -> bool:
        """Enable Alpaca News publishing to NewsBus.

        v-theme-shock-logger-2026-09-14: when True AND ALPACA_API_KEY
        is set, periodically fetch Alpaca news for watchlist symbols
        and publish to NewsBus as a high-quality source (tier 1).

        This reuses the existing news_verifier.py Alpaca integration
        but publishes to the bus instead of just verification.

        Default True if ALPACA_API_KEY is available.
        """
        if not os.getenv("ALPACA_API_KEY"):
            return False
        env_val = os.getenv("ENABLE_ALPACA_NEWS_BUS")
        if env_val is not None:
            return env_val.lower() in ("1", "true", "yes", "on")
        return bool(self.manager.get('trading.enable_alpaca_news_bus', True))

    @property
    def ALPACA_NEWS_BUS_INTERVAL_SEC(self) -> float:
        """Interval (seconds) between Alpaca news fetches for NewsBus.

        v-theme-shock-logger-2026-09-14: default 60s. Alpaca rate
        limits are generous but we don't need sub-minute latency
        for shadow logging.
        """
        env_val = os.getenv("ALPACA_NEWS_BUS_INTERVAL_SEC")
        if env_val is not None:
            try:
                return float(env_val)
            except ValueError:
                pass
        return float(self.manager.get('trading.alpaca_news_bus_interval_sec', 60.0))


# Initialize configuration
config = Config()
# ============================================================================
# LOGGING SETUP
# ============================================================================

class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for file output"""

    def format(self, record):
        import json as _json
        log_entry = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        if record.exc_info and record.exc_info[0]:
            log_entry['exception'] = self.formatException(record.exc_info)
        # Include extra fields for trade events
        for key in ('symbol', 'action', 'signal_type', 'order_id', 'price',
                     'quantity', 'pnl', 'reason'):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return _json.dumps(log_entry)


def setup_logging():
    """Configure logging with JSON file output and human-readable console output"""
    logger = logging.getLogger('TradingBot')
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on re-import
    if logger.handlers:
        return logger

    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        str(Config().LOG_PATH),
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JSONFormatter())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()
