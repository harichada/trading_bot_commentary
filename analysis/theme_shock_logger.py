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
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

import yaml

if TYPE_CHECKING:
    from core.news_bus import ScoredNewsItem

logger = logging.getLogger("TradingBot")

_DB_PERSIST_ERROR_LOGGED = False


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
    """A matched theme with affected symbols.
    
    v-themeshock-hygiene-2026-09-14: added matched_field for Stage A FP reduction.
    matched_field indicates whether the keyword hit the headline or summary,
    enabling headline-only hard_skip filtering.
    """
    theme_id: str
    matched_text: str
    matched_field: str  # "headline" | "summary"
    symbols_tradable: List[str]
    symbols_watch_only: List[str]
    actions_shadow: List[ShadowAction]


@dataclass
class ThemeEvent:
    """A logged theme event (aligned with research schema).
    
    Schema: event_id, theme_id, source, source_published_ts, bus_ingest_ts,
    news_age_sec, symbols_tradable[], symbols_watch_only[], action_shadow,
    managed_flags{}, forward_ret_1/5/15/60m, half_move_before_ingest
    
    v-themeshock-hygiene-2026-09-14: added matched_text/matched_field for
    Stage A FP reduction (Crude Oil → ai_compute via contaminated summary).
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
    matched_text: str = ""
    matched_field: str = ""  # "headline" | "summary"
    managed_flags: Dict[str, Any] = field(default_factory=dict)
    forward_ret_1m: Optional[float] = None
    forward_ret_5m: Optional[float] = None
    forward_ret_15m: Optional[float] = None
    forward_ret_60m: Optional[float] = None
    half_move_before_ingest: Optional[bool] = None
    headline: str = ""
    summary: str = ""

    def to_dict(self) -> dict:
        """Serialize for structured logging.
        
        v-themeshock-hygiene-2026-09-14: added matched_text/matched_field
        for Stage A FP analysis (which pattern, in which field).
        """
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
            "matched_text": self.matched_text,
            "matched_field": self.matched_field,
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
    
    v-themeshock-hygiene-2026-09-14: now tracks matched_field ("headline"|"summary")
    to enable headline-only filtering for hard_skip actions (Stage A FP reduction).
    """

    def __init__(self, loader: ThemeConfigLoader):
        self._loader = loader

    def tag(
        self,
        headline: str,
        summary: str = "",
        headline_only_for_hard_skip: bool = False,
    ) -> List[ThemeMatch]:
        """Match headline+summary against themes, return matches.
        
        Args:
            headline: News headline text.
            summary: News summary text (optional).
            headline_only_for_hard_skip: When True, for themes with hard_skip_entries
                action, only match against headline (ignore summary-only hits).
                This reduces false positives like Crude Oil → ai_compute via
                contaminated summary containing "anthropic".
        
        Returns:
            List of ThemeMatch with matched_field indicating where the hit was.
        """
        headline_lower = headline.lower()
        summary_lower = summary.lower() if summary else ""
        matches = []

        for theme in self._loader.themes:
            for pattern in theme.match:
                matched_field = None
                
                if pattern in headline_lower:
                    matched_field = "headline"
                elif pattern in summary_lower:
                    matched_field = "summary"
                
                if matched_field is None:
                    continue
                
                has_hard_skip = ShadowAction.HARD_SKIP_ENTRIES in theme.actions_shadow
                if (
                    headline_only_for_hard_skip
                    and has_hard_skip
                    and matched_field == "summary"
                ):
                    continue
                
                matches.append(ThemeMatch(
                    theme_id=theme.theme_id,
                    matched_text=pattern,
                    matched_field=matched_field,
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

    def is_headline_only_for_hard_skip(self) -> bool:
        """Check if hard_skip should only match on headline (not summary).
        
        v-themeshock-hygiene-2026-09-14: Stage A false-positive reduction.
        When True (default), hard_skip_entries action only fires if the
        keyword hits the headline, not summary. Prevents FPs like
        Crude Oil headline → ai_compute via summary containing "anthropic".
        """
        from core.config import Config
        return Config().ENABLE_THEME_HARD_SKIP_HEADLINE_ONLY

    def is_fanout_require_symbol_overlap(self) -> bool:
        """Check if fanout should require symbol overlap.
        
        v-themeshock-hygiene-2026-09-14: when True, process_news_item_from_publish
        only emits for symbols in (item.symbols ∩ basket/watch_only).
        When False (default), emits for ALL basket/watch_only symbols after
        any keyword match.
        """
        from core.config import Config
        return Config().ENABLE_THEME_FANOUT_REQUIRE_SYMBOL_OVERLAP

    def tag_news_item(self, item: "ScoredNewsItem") -> List[ThemeMatch]:
        """Tag a news item with matching themes.
        
        v-themeshock-hygiene-2026-09-14: respects ENABLE_THEME_HARD_SKIP_HEADLINE_ONLY.
        """
        if not self.is_enabled():
            return []
        return self._tagger.tag(
            headline=item.headline,
            summary=getattr(item, "summary", ""),
            headline_only_for_hard_skip=self.is_headline_only_for_hard_skip(),
        )

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
            matched_text=match.matched_text,
            matched_field=match.matched_field,
            managed_flags=managed_flags or {},
            headline=item.headline,
            summary=getattr(item, "summary", ""),
            **fwd_rets,
        )

        self._emit_log(event, symbol)

        if self._engine and hasattr(self._engine, "db_logger") and self._engine.db_logger:
            global _DB_PERSIST_ERROR_LOGGED
            try:
                self._engine.db_logger.log_strategy_decision(
                    strategy="theme_shock_logger",
                    symbol=symbol,
                    action=f"shadow_{action.value}",
                    reason=f"theme_{match.theme_id}",
                    extra_data=event.to_dict(),
                )
            except AttributeError as exc:
                if not _DB_PERSIST_ERROR_LOGGED:
                    logger.warning(
                        "theme_shock_logger db_persist_error symbol=%s err=%s "
                        "(log_strategy_decision missing? suppressing future warnings)",
                        symbol, exc,
                    )
                    _DB_PERSIST_ERROR_LOGGED = True
            except Exception as exc:
                if not _DB_PERSIST_ERROR_LOGGED:
                    logger.warning(
                        "theme_shock_logger db_persist_error symbol=%s err=%s "
                        "(suppressing future warnings)",
                        symbol, exc,
                    )
                    _DB_PERSIST_ERROR_LOGGED = True

        return event

    def _emit_log(self, event: ThemeEvent, symbol: str) -> None:
        """Emit structured log for theme event.
        
        v-themeshock-hygiene-2026-09-14: added matched_text/matched_field
        for Stage A FP analysis.
        """
        logger.info(
            "theme_shock_logger "
            "action=%s theme_id=%s symbol=%s "
            "event_id=%s news_age_sec=%.1f source=%s "
            "matched_text=%s matched_field=%s "
            "basket=%s watch_only=%s "
            "headline=%s",
            event.action_shadow.value,
            event.theme_id,
            symbol,
            event.event_id,
            event.news_age_sec,
            event.source,
            event.matched_text,
            event.matched_field,
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

    def _get_item_symbols(self, item: "ScoredNewsItem") -> Set[str]:
        """Extract symbols from a news item.
        
        v-themeshock-hygiene-2026-09-14: helper for symbol overlap fanout.
        Returns set of uppercase symbols from item.symbol and item.symbols.
        """
        result: Set[str] = set()
        if hasattr(item, "symbol") and item.symbol:
            result.add(item.symbol.upper())
        if hasattr(item, "symbols") and item.symbols:
            for s in item.symbols:
                if s:
                    result.add(s.upper())
        return result

    def process_news_item_from_publish(
        self,
        item: "ScoredNewsItem",
        get_open_positions_fn: Optional[callable] = None,
    ) -> List[ThemeEvent]:
        """Process a news item from NewsBus publish path.
        
        v-theme-shock-hotfix-2026-09-14: This is the CRITICAL fix. Previously
        ThemeShock was only invoked on entry path (process_news_item_for_entry)
        which only fires when evaluating a specific entry signal. Anthropic
        headlines from Alpaca went into the bus but ThemeShock was never
        notified, resulting in ZERO shadow_* emits.
        
        Now this method is called from NewsBus.on_publish for EVERY new item.
        It:
          1. Tags the item for theme matches
          2. For each match, emits shadow logs for basket symbols (entry path)
          3. Checks if any affected symbols have managed open positions and
             emits shadow logs for those too (open-position path)
        
        v-themeshock-hygiene-2026-09-14: ENABLE_THEME_FANOUT_REQUIRE_SYMBOL_OVERLAP
        controls whether we emit for ALL basket symbols (False, default) or only
        for symbols that overlap with item.symbols (True).
        
        Args:
            item: The ScoredNewsItem from NewsBus publish.
            get_open_positions_fn: Optional callable that returns dict of
                {symbol: position} for managed_by_bot positions. Used to
                check if themes hit open positions.
        
        Returns:
            List of ThemeEvents that were logged.
        """
        if not self.is_enabled():
            return []

        matches = self.tag_news_item(item)
        if not matches:
            return []

        events = []
        open_positions = {}
        if get_open_positions_fn is not None:
            try:
                open_positions = get_open_positions_fn()
            except Exception as exc:
                logger.debug(
                    "theme_shock process_from_publish get_open_positions error: %s",
                    exc,
                )
        
        require_overlap = self.is_fanout_require_symbol_overlap()
        item_symbols = self._get_item_symbols(item) if require_overlap else set()
        
        for match in matches:
            all_affected = set(match.symbols_tradable + match.symbols_watch_only)
            
            if require_overlap and item_symbols:
                all_affected = all_affected & item_symbols
                if not all_affected:
                    continue
            
            for symbol in all_affected:
                symbol_upper = symbol.upper()
                
                is_open_position = (
                    symbol_upper in open_positions
                    and getattr(open_positions[symbol_upper], 'managed_by_bot', False)
                )
                
                if is_open_position:
                    position = open_positions[symbol_upper]
                    action = self._determine_action_for_symbol(
                        symbol_upper, match, is_managed_open=True
                    )
                    managed_flags = {
                        "publish_path": True,
                        "open_position": True,
                        "managed_by_bot": True,
                        "position_side": getattr(position, 'side', 'unknown'),
                    }
                else:
                    action = self._determine_action_for_symbol(
                        symbol_upper, match, is_managed_open=False
                    )
                    managed_flags = {"publish_path": True, "entry_gate": True}
                
                event = self.log_theme_event(
                    item=item,
                    match=match,
                    action=action,
                    symbol=symbol_upper,
                    managed_flags=managed_flags,
                )
                if event:
                    events.append(event)
        
        return events


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
