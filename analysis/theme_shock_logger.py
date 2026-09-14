"""v-theme-shock-logger-2026-09-14: ThemeShock shadow logger.

Multi-sentiment desk theme matching and shadow action logging for
Fed/earnings/news/CEO-AI shocks. Shadow-only: logs would-hard-skip /
would-size-down / would-exit / alert-only events WITHOUT executing
broker calls or mutating orders.

See docs/research/2026-09-14-multisenti-stage-a.md for design.

Schema (aligned with research doc):
  event_id, theme_id, source, source_published_ts, bus_ingest_ts,
  news_age_sec, symbols_tradable[], symbols_watch_only[], action_shadow,
  managed_flags{}, forward_ret_1/5/15/60m, half_move_before_ingest

Modular boundary:
  - Config flags: ENABLE_THEME_SHOCK_LOGGER (master), THEME_CONFIG_PATH
  - Theme config: data/themes/themes.yaml (Research-owned)
  - Respects HANDS_OFF_DENYLIST (MU, HQGE, SPCX) — alert_only only

Shadow action values:
  - hard_skip_entries: would block new entries
  - size_down_open: would reduce size on open positions
  - thesis_exit: would exit managed open positions
  - alert_only: only log alert, no action
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

import yaml

if TYPE_CHECKING:
    from core.news_bus import ScoredNewsItem

logger = logging.getLogger("TradingBot")


class ShadowAction(str, Enum):
    """Shadow action types for theme matches."""
    HARD_SKIP_ENTRIES = "hard_skip_entries"
    SIZE_DOWN_OPEN = "size_down_open"
    THESIS_EXIT = "thesis_exit"
    ALERT_ONLY = "alert_only"


@dataclass
class ThemeConfig:
    """Configuration for a single theme."""
    theme_id: str
    description: str
    match: List[str]
    basket: List[str]
    watch_only: List[str]
    actions_shadow: List[ShadowAction]

    @classmethod
    def from_dict(cls, data: dict) -> "ThemeConfig":
        """Create ThemeConfig from YAML dict."""
        actions = []
        for action_str in data.get("actions_shadow", []):
            try:
                actions.append(ShadowAction(action_str))
            except ValueError:
                logger.warning(f"Unknown shadow action: {action_str}")
        return cls(
            theme_id=data.get("theme_id", "unknown"),
            description=data.get("description", ""),
            match=[m.lower() for m in data.get("match", [])],
            basket=[s.upper() for s in data.get("basket", [])],
            watch_only=[s.upper() for s in data.get("watch_only", [])],
            actions_shadow=actions,
        )


@dataclass
class ThemeMatch:
    """A matched theme with affected symbols."""
    theme_id: str
    matched_text: str
    symbols_tradable: List[str]
    symbols_watch_only: List[str]
    actions_shadow: List[ShadowAction]


@dataclass
class ThemeEvent:
    """A logged theme event (aligned with research schema).
    
    Schema: event_id, theme_id, source, source_published_ts, bus_ingest_ts,
    news_age_sec, symbols_tradable[], symbols_watch_only[], action_shadow,
    managed_flags{}, forward_ret_1/5/15/60m, half_move_before_ingest
    """
    event_id: str
    theme_id: str
    source: str
    source_published_ts: Optional[datetime]
    bus_ingest_ts: datetime
    news_age_sec: float
    symbols_tradable: List[str]
    symbols_watch_only: List[str]
    action_shadow: ShadowAction
    ticker_basket: List[str]
    managed_flags: Dict[str, Any] = field(default_factory=dict)
    forward_ret_1m: Optional[float] = None
    forward_ret_5m: Optional[float] = None
    forward_ret_15m: Optional[float] = None
    forward_ret_60m: Optional[float] = None
    half_move_before_ingest: Optional[bool] = None
    headline: str = ""
    summary: str = ""

    def to_dict(self) -> dict:
        """Serialize for structured logging."""
        return {
            "event_id": self.event_id,
            "theme_id": self.theme_id,
            "source": self.source,
            "source_published_ts": (
                self.source_published_ts.isoformat()
                if self.source_published_ts else None
            ),
            "bus_ingest_ts": self.bus_ingest_ts.isoformat(),
            "news_age_sec": self.news_age_sec,
            "symbols_tradable": self.symbols_tradable,
            "symbols_watch_only": self.symbols_watch_only,
            "ticker_basket": self.ticker_basket,
            "action_shadow": self.action_shadow.value,
            "managed_flags": self.managed_flags,
            "forward_ret_1m": self.forward_ret_1m,
            "forward_ret_5m": self.forward_ret_5m,
            "forward_ret_15m": self.forward_ret_15m,
            "forward_ret_60m": self.forward_ret_60m,
            "half_move_before_ingest": self.half_move_before_ingest,
            "headline": self.headline[:100] if self.headline else "",
        }


class ThemeConfigLoader:
    """Loads theme configurations from YAML files."""

    def __init__(self, config_path: Optional[str] = None):
        from core.config import Config
        self._config_path = config_path or Config().THEME_CONFIG_PATH
        self._themes: List[ThemeConfig] = []
        self._loaded_at: Optional[datetime] = None
        self._load()

    def _load(self) -> None:
        """Load theme configs from YAML files."""
        path = Path(self._config_path)
        if not path.exists():
            logger.warning(f"Theme config path does not exist: {path}")
            return

        yaml_files = list(path.glob("*.yaml")) + list(path.glob("*.yml"))
        if not yaml_files:
            logger.warning(f"No theme YAML files in: {path}")
            return

        self._themes = []
        for yaml_file in yaml_files:
            try:
                with open(yaml_file, "r") as f:
                    data = yaml.safe_load(f)
                if data and "themes" in data:
                    for theme_data in data["themes"]:
                        self._themes.append(ThemeConfig.from_dict(theme_data))
            except Exception as e:
                logger.error(f"Failed to load theme config {yaml_file}: {e}")

        self._loaded_at = datetime.now(timezone.utc)
        logger.info(
            f"theme_shock_logger: loaded {len(self._themes)} themes from {len(yaml_files)} files"
        )

    def reload(self) -> None:
        """Reload theme configs."""
        self._load()

    @property
    def themes(self) -> List[ThemeConfig]:
        """Get loaded themes."""
        return self._themes


class ThemeTagger:
    """Tags news items with matching themes.
    
    Matches headline + summary against theme match patterns (case-insensitive).
    Returns list of ThemeMatch for all matching themes.
    """

    def __init__(self, loader: ThemeConfigLoader):
        self._loader = loader

    def tag(self, headline: str, summary: str = "") -> List[ThemeMatch]:
        """Match headline+summary against themes, return matches."""
        text = f"{headline} {summary}".lower()
        matches = []

        for theme in self._loader.themes:
            for pattern in theme.match:
                if pattern in text:
                    matches.append(ThemeMatch(
                        theme_id=theme.theme_id,
                        matched_text=pattern,
                        symbols_tradable=theme.basket[:],
                        symbols_watch_only=theme.watch_only[:],
                        actions_shadow=theme.actions_shadow[:],
                    ))
                    break

        return matches


class ThemeShockLogger:
    """Shadow logger for theme-based news events.
    
    Logs structured theme events for Stage A evidence collection.
    Does NOT execute any broker calls or order mutations.
    
    Respects HANDS_OFF_DENYLIST (MU, HQGE, SPCX) — theme hits on these
    symbols only emit alert_only, never action shadows.
    """

    def __init__(self, engine=None):
        self._engine = engine
        self._loader = ThemeConfigLoader()
        self._tagger = ThemeTagger(self._loader)
        self._recent_events: Dict[str, datetime] = {}
        self._dedupe_window_sec = 900.0

    def is_enabled(self) -> bool:
        """Check if theme shock logger is enabled."""
        from core.config import Config
        return Config().ENABLE_THEME_SHOCK_LOGGER

    def get_hands_off_denylist(self) -> frozenset:
        """Get the HANDS_OFF_DENYLIST from config."""
        from core.config import Config
        return Config().HANDS_OFF_DENYLIST

    def tag_news_item(self, item: "ScoredNewsItem") -> List[ThemeMatch]:
        """Tag a news item with matching themes."""
        if not self.is_enabled():
            return []
        return self._tagger.tag(item.headline, getattr(item, "summary", ""))

    def _make_event_id(self, theme_id: str, headline: str) -> str:
        """Generate a stable event ID for deduplication."""
        content = f"{theme_id}:{headline[:50]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _should_dedupe(self, event_id: str, theme_id: str) -> bool:
        """Check if this event was recently logged (deduplication)."""
        key = f"{theme_id}:{event_id}"
        now = datetime.now(timezone.utc)
        if key in self._recent_events:
            elapsed = (now - self._recent_events[key]).total_seconds()
            if elapsed < self._dedupe_window_sec:
                return True
        self._recent_events[key] = now
        self._clean_stale_events()
        return False

    def _clean_stale_events(self) -> None:
        """Remove stale entries from recent events cache."""
        now = datetime.now(timezone.utc)
        stale_keys = [
            k for k, ts in self._recent_events.items()
            if (now - ts).total_seconds() > self._dedupe_window_sec * 2
        ]
        for k in stale_keys:
            del self._recent_events[k]

    def _determine_action_for_symbol(
        self,
        symbol: str,
        match: ThemeMatch,
        is_managed_open: bool = False,
    ) -> ShadowAction:
        """Determine the shadow action for a specific symbol.
        
        Rules:
          - HANDS_OFF_DENYLIST symbols → alert_only only
          - watch_only symbols → alert_only only
          - Managed open positions → thesis_exit or size_down_open
          - Entry candidates → hard_skip_entries
        """
        denylist = self.get_hands_off_denylist()
        symbol_upper = symbol.upper()

        if symbol_upper in denylist:
            return ShadowAction.ALERT_ONLY

        if symbol_upper in match.symbols_watch_only:
            return ShadowAction.ALERT_ONLY

        if is_managed_open:
            if ShadowAction.THESIS_EXIT in match.actions_shadow:
                return ShadowAction.THESIS_EXIT
            elif ShadowAction.SIZE_DOWN_OPEN in match.actions_shadow:
                return ShadowAction.SIZE_DOWN_OPEN

        if ShadowAction.HARD_SKIP_ENTRIES in match.actions_shadow:
            return ShadowAction.HARD_SKIP_ENTRIES

        return ShadowAction.ALERT_ONLY

    def _get_forward_returns_stub(
        self,
        symbol: str,
        event_ts: datetime,
    ) -> Dict[str, Optional[float]]:
        """Stub for forward returns computation.
        
        TODO: Implement using minute_bars/Alpaca bars when available.
        For now, returns None for all forward returns with a TODO marker.
        """
        return {
            "forward_ret_1m": None,
            "forward_ret_5m": None,
            "forward_ret_15m": None,
            "forward_ret_60m": None,
            "half_move_before_ingest": None,
        }

    def log_theme_event(
        self,
        item: "ScoredNewsItem",
        match: ThemeMatch,
        action: ShadowAction,
        symbol: str,
        managed_flags: Optional[Dict[str, Any]] = None,
    ) -> Optional[ThemeEvent]:
        """Log a theme event with structured data.
        
        Returns the ThemeEvent if logged, None if deduped/skipped.
        """
        if not self.is_enabled():
            return None

        event_id = self._make_event_id(match.theme_id, item.headline)
        if self._should_dedupe(event_id, match.theme_id):
            return None

        now = datetime.now(timezone.utc)
        pub_ts = getattr(item, "published_time", None)
        if pub_ts and pub_ts.tzinfo is None:
            pub_ts = pub_ts.replace(tzinfo=timezone.utc)

        news_age_sec = item.age_sec() if hasattr(item, "age_sec") else 0.0

        fwd_rets = self._get_forward_returns_stub(symbol, now)

        event = ThemeEvent(
            event_id=event_id,
            theme_id=match.theme_id,
            source=item.source,
            source_published_ts=pub_ts,
            bus_ingest_ts=now,
            news_age_sec=news_age_sec,
            symbols_tradable=match.symbols_tradable,
            symbols_watch_only=match.symbols_watch_only,
            ticker_basket=match.symbols_tradable + match.symbols_watch_only,
            action_shadow=action,
            managed_flags=managed_flags or {},
            headline=item.headline,
            summary=getattr(item, "summary", ""),
            **fwd_rets,
        )

        self._emit_log(event, symbol)

        if self._engine and hasattr(self._engine, "db_logger") and self._engine.db_logger:
            try:
                self._engine.db_logger.log_strategy_decision(
                    strategy="theme_shock_logger",
                    symbol=symbol,
                    action=f"shadow_{action.value}",
                    reason=f"theme_{match.theme_id}",
                    extra_data=event.to_dict(),
                )
            except Exception:
                pass

        return event

    def _emit_log(self, event: ThemeEvent, symbol: str) -> None:
        """Emit structured log for theme event."""
        logger.info(
            "theme_shock_logger "
            "action=%s theme_id=%s symbol=%s "
            "event_id=%s news_age_sec=%.1f source=%s "
            "basket=%s watch_only=%s "
            "headline=%s",
            event.action_shadow.value,
            event.theme_id,
            symbol,
            event.event_id,
            event.news_age_sec,
            event.source,
            ",".join(event.symbols_tradable),
            ",".join(event.symbols_watch_only),
            event.headline[:60],
        )

    def process_news_item_for_entry(
        self,
        item: "ScoredNewsItem",
        candidate_symbol: str,
    ) -> Optional[ThemeEvent]:
        """Process a news item for entry gate (would-hard-skip shadow).
        
        Called by entry strategies to check if a candidate entry should
        be shadow-skipped due to a theme match.
        
        Returns ThemeEvent if a hard_skip_entries shadow was logged,
        None otherwise.
        """
        if not self.is_enabled():
            return None

        matches = self.tag_news_item(item)
        if not matches:
            return None

        for match in matches:
            all_affected = set(match.symbols_tradable + match.symbols_watch_only)
            if candidate_symbol.upper() not in all_affected:
                continue

            action = self._determine_action_for_symbol(
                candidate_symbol, match, is_managed_open=False
            )

            event = self.log_theme_event(
                item=item,
                match=match,
                action=action,
                symbol=candidate_symbol,
                managed_flags={"entry_candidate": True},
            )

            if event and action == ShadowAction.HARD_SKIP_ENTRIES:
                return event

        return None

    def process_news_item_for_open_position(
        self,
        item: "ScoredNewsItem",
        position_symbol: str,
        position_side: str = "long",
        is_managed_by_bot: bool = True,
    ) -> Optional[ThemeEvent]:
        """Process a news item for open position (WOULD_TIGHTEN/WOULD_EXIT shadow).
        
        Called by Active Open Desk to check if an open position should
        be shadow-exited or shadow-tightened due to a theme match.
        
        Returns ThemeEvent if an action shadow was logged, None otherwise.
        """
        if not self.is_enabled():
            return None

        if not is_managed_by_bot:
            return None

        matches = self.tag_news_item(item)
        if not matches:
            return None

        for match in matches:
            all_affected = set(match.symbols_tradable + match.symbols_watch_only)
            if position_symbol.upper() not in all_affected:
                continue

            action = self._determine_action_for_symbol(
                position_symbol, match, is_managed_open=True
            )

            event = self.log_theme_event(
                item=item,
                match=match,
                action=action,
                symbol=position_symbol,
                managed_flags={
                    "managed_by_bot": is_managed_by_bot,
                    "position_side": position_side,
                    "open_position": True,
                },
            )

            if event and action in (
                ShadowAction.THESIS_EXIT,
                ShadowAction.SIZE_DOWN_OPEN,
            ):
                return event

        return None

    def get_themes_for_symbol(self, symbol: str) -> List[ThemeConfig]:
        """Get all themes that affect a symbol (basket or watch_only)."""
        symbol_upper = symbol.upper()
        result = []
        for theme in self._loader.themes:
            if symbol_upper in theme.basket or symbol_upper in theme.watch_only:
                result.append(theme)
        return result

    def reload_themes(self) -> None:
        """Reload theme configurations."""
        self._loader.reload()


_logger_singleton: Optional[ThemeShockLogger] = None


def get_theme_shock_logger(engine=None) -> ThemeShockLogger:
    """Get or create the singleton ThemeShockLogger instance."""
    global _logger_singleton
    if _logger_singleton is None:
        _logger_singleton = ThemeShockLogger(engine=engine)
    elif engine is not None and _logger_singleton._engine is None:
        _logger_singleton._engine = engine
    return _logger_singleton


def reset_theme_shock_logger() -> None:
    """Reset the singleton (for testing)."""
    global _logger_singleton
    _logger_singleton = None
