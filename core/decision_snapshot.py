"""v-feature-snapshot-2026-09-09: Structured decision snapshots for ML training.

This module defines compact, machine-readable snapshots that capture the bot's
complete decision context at every analyze/skip/veto/entry attempt. These
snapshots feed offline training of a GPU-based policy/regime model.

Architecture:
  - Strategies and engine emit DecisionSnapshot on every decision point
  - DbLogger persists snapshots to Postgres (bot_decision_snapshots table)
  - Read API exposes snapshots for training pipelines and tooling
  - Online inference remains gated behind FEATURE_SNAPSHOT_INFERENCE_ENABLED

Thread-safe: snapshots are immutable dataclasses, fire-and-forget writes.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("TradingBot")


class DecisionAction(str, Enum):
    """Action taken at a decision point."""
    SIGNAL_BUY = "signal_buy"
    SIGNAL_SELL = "signal_sell"
    SKIP = "skip"
    VETO = "veto"
    ERROR = "error"


@dataclass(frozen=True)
class PriceVolumeFeatures:
    """Compact price/volume features already computed by the analysis loop.
    
    These match the MLFeatureExtractor feature set for consistency.
    All fields are normalized/ratio-based for model stability.
    """
    price: float  # Current close price
    returns_1: float  # 1-bar return
    returns_5: float  # 5-bar return
    returns_20: float  # 20-bar return
    price_vs_sma20: float  # Price relative to 20-bar SMA
    price_vs_sma50: float  # Price relative to 50-bar SMA
    rsi: float  # RSI (0-100 normalized to 0-1)
    macd: float  # MACD (normalized by price)
    macd_signal: float  # MACD signal line
    macd_hist: float  # MACD histogram
    bb_position: float  # Position within Bollinger Bands (0-1)
    bb_width: float  # Bollinger Band width (normalized)
    volume_ratio: float  # Volume vs 20-bar average
    volume_std_ratio: float  # Volume std vs mean
    volume_trend: float  # 5-bar vs 20-bar volume trend
    atr_ratio: float  # ATR as percentage of price
    high_low_ratio: float  # Bar range as percentage of price
    std_dev_ratio: float  # 20-bar price std vs mean
    
    @classmethod
    def from_indicators(cls, indicators: Dict[str, Any], price: float) -> "PriceVolumeFeatures":
        """Build from indicator dict (as strategies receive)."""
        safe_price = float(price) if price else 1.0
        return cls(
            price=safe_price,
            returns_1=float(indicators.get("returns_1", 0)),
            returns_5=float(indicators.get("returns_5", 0)),
            returns_20=float(indicators.get("returns_20", 0)),
            price_vs_sma20=float(indicators.get("price_vs_sma20", 0)),
            price_vs_sma50=float(indicators.get("price_vs_sma50", 0)),
            rsi=float(indicators.get("rsi", 50)) / 100.0,
            macd=float(indicators.get("macd", 0)) / safe_price,
            macd_signal=float(indicators.get("macd_signal", 0)) / safe_price,
            macd_hist=float(indicators.get("macd_histogram", indicators.get("macd_hist", 0))) / safe_price,
            bb_position=float(indicators.get("bb_position", 0.5)),
            bb_width=float(indicators.get("bb_width", 0)),
            volume_ratio=float(indicators.get("volume_ratio", 1.0)),
            volume_std_ratio=float(indicators.get("volume_std_ratio", 0)),
            volume_trend=float(indicators.get("volume_trend", 0)),
            atr_ratio=float(indicators.get("atr", 0)) / safe_price,
            high_low_ratio=float(indicators.get("high_low_ratio", 0)),
            std_dev_ratio=float(indicators.get("std_dev_ratio", 0)),
        )
    
    @classmethod
    def empty(cls, price: float = 0.0) -> "PriceVolumeFeatures":
        """Return empty features (for error/missing data cases)."""
        return cls(
            price=price,
            returns_1=0.0, returns_5=0.0, returns_20=0.0,
            price_vs_sma20=0.0, price_vs_sma50=0.0,
            rsi=0.5, macd=0.0, macd_signal=0.0, macd_hist=0.0,
            bb_position=0.5, bb_width=0.0,
            volume_ratio=1.0, volume_std_ratio=0.0, volume_trend=0.0,
            atr_ratio=0.0, high_low_ratio=0.0, std_dev_ratio=0.0,
        )
    
    def to_vector(self) -> List[float]:
        """Return features as a flat vector for model input."""
        return [
            self.returns_1, self.returns_5, self.returns_20,
            self.price_vs_sma20, self.price_vs_sma50,
            self.rsi, self.macd, self.macd_signal, self.macd_hist,
            self.bb_position, self.bb_width,
            self.volume_ratio, self.volume_std_ratio, self.volume_trend,
            self.atr_ratio, self.high_low_ratio, self.std_dev_ratio,
        ]


@dataclass(frozen=True)
class NewsAggregate:
    """Aggregate news context at decision time.
    
    Captures what the bot knew about news when it made a decision,
    matching the NewsBus evaluate_gate inputs.
    """
    article_count: int  # Total articles in TTL window
    fresh_count: int  # Articles within freshness window
    avg_sentiment: float  # Average sentiment score (-1 to +1)
    freshest_age_sec: Optional[float]  # Age of most recent article
    source_tier_min: Optional[int]  # Best source tier (1=primary)
    high_impact_count: int  # Number of high-impact articles
    corroboration_n: int  # Number of distinct sources
    gate_action: str  # NewsGateAction result (full_size, reduced, veto_*)
    gate_size_mult: float  # Size multiplier from gate (0.0-1.0)
    
    @classmethod
    def from_gate_result(
        cls,
        gate_result,  # NewsGateResult
        aggregate: Dict[str, Any],
    ) -> "NewsAggregate":
        """Build from NewsBus.evaluate_gate result + get_aggregate_sentiment."""
        return cls(
            article_count=int(aggregate.get("article_count", 0)),
            fresh_count=gate_result.corroboration_n if gate_result else 0,
            avg_sentiment=float(aggregate.get("avg_sentiment", 0)),
            freshest_age_sec=gate_result.news_age_sec if gate_result else None,
            source_tier_min=gate_result.source_tier_min if gate_result else None,
            high_impact_count=int(aggregate.get("high_impact_count", 0)),
            corroboration_n=gate_result.corroboration_n if gate_result else 0,
            gate_action=gate_result.action.value if gate_result else "unknown",
            gate_size_mult=gate_result.size_multiplier if gate_result else 0.0,
        )
    
    @classmethod
    def empty(cls) -> "NewsAggregate":
        """Return empty news aggregate (no news context)."""
        return cls(
            article_count=0, fresh_count=0, avg_sentiment=0.0,
            freshest_age_sec=None, source_tier_min=None,
            high_impact_count=0, corroboration_n=0,
            gate_action="no_news", gate_size_mult=0.0,
        )


@dataclass(frozen=True)
class RegimeContext:
    """Market regime context at decision time."""
    regime: str  # Current regime label (trending, choppy, volatile, etc.)
    spy_slope_pct: float  # SPY EMA slope
    vix: Optional[float]  # VIX if available
    tape: Optional[str]  # RegimeAllocator tape classification
    er: Optional[float]  # Efficiency ratio
    
    @classmethod
    def from_context(
        cls,
        regime: str = "unknown",
        spy_slope_pct: float = 0.0,
        vix: Optional[float] = None,
        allocator_result: Optional[Any] = None,
    ) -> "RegimeContext":
        return cls(
            regime=regime,
            spy_slope_pct=spy_slope_pct,
            vix=vix,
            tape=getattr(allocator_result, "tape", None) if allocator_result else None,
            er=getattr(allocator_result, "er", None) if allocator_result else None,
        )
    
    @classmethod
    def empty(cls) -> "RegimeContext":
        return cls(regime="unknown", spy_slope_pct=0.0, vix=None, tape=None, er=None)


@dataclass(frozen=True)
class MomentumContext:
    """Momentum-specific context for day-trade desk Stage-A instrumentation.
    
    v-momentum-stage-a-2026-09-10: captures fields needed for Stage-A
    promotion tracking and daily rollup analysis.
    """
    session_id: str  # Trading date in ET (YYYY-MM-DD)
    setup_type: str  # Entry setup type (momentum_breakout/momentum_pullback/momentum_continuation)
    risk_off: bool  # Whether market context is risk_off
    mc_size_mult: float  # Market context size multiplier applied
    rs_vs_spy: float  # Relative strength vs SPY
    is_day_trade: bool  # True for day-trade momentum entries
    flatten_hour: int  # Hour by which position should flatten (15 = 3PM ET)
    entry_pattern: str  # Raw pattern (breakout/pullback/continuation)
    
    @classmethod
    def from_signal_reasoning(cls, reasoning: Dict[str, Any]) -> "MomentumContext":
        """Build from signal.reasoning dict."""
        return cls(
            session_id=reasoning.get("session_id", ""),
            setup_type=reasoning.get("setup_type", ""),
            risk_off=bool(reasoning.get("risk_off", False)),
            mc_size_mult=float(reasoning.get("mc_size_mult", 1.0)),
            rs_vs_spy=float(reasoning.get("rs_vs_spy", 0.0)),
            is_day_trade=bool(reasoning.get("is_day_trade", False)),
            flatten_hour=int(reasoning.get("flatten_hour", 15)),
            entry_pattern=reasoning.get("entry_pattern", ""),
        )
    
    @classmethod
    def empty(cls) -> "MomentumContext":
        """Return empty momentum context (for non-momentum strategies)."""
        return cls(
            session_id="",
            setup_type="",
            risk_off=False,
            mc_size_mult=1.0,
            rs_vs_spy=0.0,
            is_day_trade=False,
            flatten_hour=15,
            entry_pattern="",
        )


@dataclass(frozen=True)
class DecisionSnapshot:
    """Complete snapshot of decision context for ML training.
    
    This is the primary artifact for offline policy learning. Each snapshot
    captures everything the bot knew when it made a decision, enabling
    supervised learning of "what would a human do?" or reinforcement
    learning of "what maximizes risk-adjusted returns?".
    
    Fields are deliberately compact and typed for efficient serialization.
    The snapshot_id is a deterministic hash of key fields for deduplication.
    """
    # Identity
    snapshot_id: str  # Deterministic hash for deduplication
    symbol: str  # Ticker symbol
    ts: datetime  # Timestamp of decision
    
    # Trading context
    mode: str  # Trading mode (live, simulation, paper)
    strategy_id: str  # Strategy that produced this decision
    
    # Decision outcome
    action: DecisionAction  # What the bot decided
    reason: str  # Human-readable reason (snake_case)
    gate_name: Optional[str]  # Which gate blocked (if skip/veto)
    confidence: float  # Strategy confidence (0-1)
    
    # Features
    price_vol: PriceVolumeFeatures  # Price/volume features
    news: NewsAggregate  # News context
    regime: RegimeContext  # Market regime context
    
    # Momentum-specific context (v-momentum-stage-a-2026-09-10)
    # Optional for non-momentum strategies, populated for day_trade_momentum
    momentum: Optional[MomentumContext] = None
    
    # Sizing (if applicable)
    would_entry_price: Optional[float] = None  # Intended entry
    would_stop_loss: Optional[float] = None  # Intended stop
    would_take_profit: Optional[float] = None  # Intended target
    would_size_shares: Optional[int] = None  # Intended position size
    would_size_mult: Optional[float] = None  # Size multiplier applied
    
    # Metadata
    extra: Dict[str, Any] = field(default_factory=dict)  # Strategy-specific extras
    
    @staticmethod
    def compute_id(symbol: str, ts: datetime, strategy_id: str, action: str) -> str:
        """Compute deterministic snapshot ID for deduplication."""
        key = f"{symbol}:{ts.isoformat()}:{strategy_id}:{action}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON/DB storage."""
        result = {
            "snapshot_id": self.snapshot_id,
            "symbol": self.symbol,
            "ts": self.ts.isoformat(),
            "mode": self.mode,
            "strategy_id": self.strategy_id,
            "action": self.action.value,
            "reason": self.reason,
            "gate_name": self.gate_name,
            "confidence": self.confidence,
            "price_vol": asdict(self.price_vol),
            "news": asdict(self.news),
            "regime": asdict(self.regime),
            "momentum": asdict(self.momentum) if self.momentum else None,
            "would_entry_price": self.would_entry_price,
            "would_stop_loss": self.would_stop_loss,
            "would_take_profit": self.would_take_profit,
            "would_size_shares": self.would_size_shares,
            "would_size_mult": self.would_size_mult,
            "extra": self.extra,
        }
        return result
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionSnapshot":
        """Deserialize from dict."""
        momentum_data = data.get("momentum")
        momentum = MomentumContext(**momentum_data) if momentum_data else None
        return cls(
            snapshot_id=data["snapshot_id"],
            symbol=data["symbol"],
            ts=datetime.fromisoformat(data["ts"]) if isinstance(data["ts"], str) else data["ts"],
            mode=data["mode"],
            strategy_id=data["strategy_id"],
            action=DecisionAction(data["action"]),
            reason=data["reason"],
            gate_name=data.get("gate_name"),
            confidence=float(data.get("confidence", 0)),
            price_vol=PriceVolumeFeatures(**data["price_vol"]),
            news=NewsAggregate(**data["news"]),
            regime=RegimeContext(**data["regime"]),
            momentum=momentum,
            would_entry_price=data.get("would_entry_price"),
            would_stop_loss=data.get("would_stop_loss"),
            would_take_profit=data.get("would_take_profit"),
            would_size_shares=data.get("would_size_shares"),
            would_size_mult=data.get("would_size_mult"),
            extra=data.get("extra", {}),
        )


def build_snapshot(
    symbol: str,
    strategy_id: str,
    action: DecisionAction,
    reason: str,
    mode: str = "simulation",
    gate_name: Optional[str] = None,
    confidence: float = 0.0,
    indicators: Optional[Dict[str, Any]] = None,
    price: float = 0.0,
    news_aggregate: Optional[Dict[str, Any]] = None,
    news_gate_result: Optional[Any] = None,
    regime_context: Optional[RegimeContext] = None,
    momentum_context: Optional[MomentumContext] = None,
    signal_reasoning: Optional[Dict[str, Any]] = None,
    would_entry_price: Optional[float] = None,
    would_stop_loss: Optional[float] = None,
    would_take_profit: Optional[float] = None,
    would_size_shares: Optional[int] = None,
    would_size_mult: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
    ts: Optional[datetime] = None,
) -> DecisionSnapshot:
    """Factory function to build a DecisionSnapshot with sensible defaults.
    
    Use this instead of the dataclass constructor for convenience.
    
    v-momentum-stage-a-2026-09-10: added momentum_context and signal_reasoning
    parameters for Stage-A instrumentation. If signal_reasoning is provided
    and contains momentum-specific fields (is_day_trade=True), a MomentumContext
    is automatically built from it.
    """
    now = ts or datetime.now()
    snapshot_id = DecisionSnapshot.compute_id(symbol, now, strategy_id, action.value)
    
    price_vol = (
        PriceVolumeFeatures.from_indicators(indicators, price)
        if indicators
        else PriceVolumeFeatures.empty(price)
    )
    
    news = (
        NewsAggregate.from_gate_result(news_gate_result, news_aggregate or {})
        if news_gate_result or news_aggregate
        else NewsAggregate.empty()
    )
    
    regime = regime_context or RegimeContext.empty()
    
    # Build momentum context if provided directly or extract from signal reasoning
    momentum = momentum_context
    if momentum is None and signal_reasoning:
        if signal_reasoning.get("is_day_trade") or signal_reasoning.get("strategy") == "day_trade_momentum":
            momentum = MomentumContext.from_signal_reasoning(signal_reasoning)
    
    return DecisionSnapshot(
        snapshot_id=snapshot_id,
        symbol=symbol,
        ts=now,
        mode=mode,
        strategy_id=strategy_id,
        action=action,
        reason=reason,
        gate_name=gate_name,
        confidence=confidence,
        price_vol=price_vol,
        news=news,
        regime=regime,
        momentum=momentum,
        would_entry_price=would_entry_price,
        would_stop_loss=would_stop_loss,
        would_take_profit=would_take_profit,
        would_size_shares=would_size_shares,
        would_size_mult=would_size_mult,
        extra=extra or {},
    )


# v-feature-snapshot-config-2026-09-09: module constants kept for backward
# compat; actual flags now live in Config with env override.
FEATURE_SNAPSHOT_LOGGING_ENABLED = True   # Deprecated: use Config().FEATURE_SNAPSHOT_LOGGING
FEATURE_SNAPSHOT_INFERENCE_ENABLED = False  # Deprecated: use Config().FEATURE_SNAPSHOT_INFERENCE


def is_snapshot_logging_enabled() -> bool:
    """Check if snapshot logging is enabled.
    
    v-feature-snapshot-config-2026-09-09: reads from Config with env override.
    """
    try:
        from core.config import Config
        return Config().FEATURE_SNAPSHOT_LOGGING
    except Exception:
        return FEATURE_SNAPSHOT_LOGGING_ENABLED


def is_snapshot_inference_enabled() -> bool:
    """Check if snapshot-based inference is enabled.
    
    This is separate from logging — we can log snapshots for training
    without using them for live inference. Inference requires explicit
    opt-in via Config or environment variable.
    
    v-feature-snapshot-config-2026-09-09: reads from Config with env override.
    """
    try:
        from core.config import Config
        return Config().FEATURE_SNAPSHOT_INFERENCE
    except Exception:
        return FEATURE_SNAPSHOT_INFERENCE_ENABLED
