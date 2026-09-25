"""FitLaneStrategy: generic live adapter over strategies/families entry
functions, gated by per-symbol strategy-fit verdicts.

Shadow-first: a family registered here always runs its entry logic and
writes what it would have done to an ndjson ledger. Only families listed
in Config().FIT_LANE_LIVE_ENTRIES_ENABLED actually emit a live
TradingSignal; every other registered family stays shadow-only (returns
None, ledger append only).

v-strategy-fit-2026-09-25 (Task 31): live fit-lane adapter and engine
registration. May import core and strategies; must never import research
(core/research separation — see core/strategy_fit.py docstring).

v-strategy-fit-review-2026-09-25 (Task 31 review fixes):
  1. entries now carry the real symbol through to the emitted signal
     (families/attrs don't reliably set it themselves).
  2. the body after the fit_no_row check is exception-contained — a bad
     family name fails fast at construction, everything else degrades to
     a logged "error" skip rather than propagating.
  3. a staleness guard rejects post-close / pre-open bar snapshots so the
     shadow ledger doesn't get polluted with entries computed on stale data.
  4. the bar-count floor is checked on *closed* bars, not raw bars.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import time as dt_time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Tuple

import pandas as pd

from core.models import MarketData, SignalType, TradingSignal
from strategies.base import TradingStrategyWithCommentary
from strategies.families.common import Family, median_params
from strategies.families.indicators import closed_bars, indicator_frame
from strategies.families.models import Entry
from strategies.families.registry import get_family

logger = logging.getLogger("TradingBot")

LEDGER_DIR = Path("data")

_NEWS_FRAME_COLUMNS = ["created_at", "headline", "source", "sentiment"]

# v-strategy-fit-review-2026-09-25: bar snapshot must be no older than this,
# and the synthetic "now" bar must not extend past the RTH close, or the
# adapter skips rather than computing entries on stale data.
_FRESHNESS_LIMIT = pd.Timedelta(minutes=15)
_RTH_CLOSE = dt_time(15, 55)


def ledger_path(family: str, ledger_dir: Path = LEDGER_DIR) -> Path:
    """Path to the shadow ledger for `family` under `ledger_dir` (default `data/`)."""
    return Path(ledger_dir) / f"shadow_fit_{family}.ndjson"


def _to_naive_et(ts: pd.Timestamp) -> pd.Timestamp:
    """Tz-aware timestamps are converted to ET then stripped; naive ones
    are assumed to already be ET."""
    if ts.tzinfo is not None:
        return ts.tz_convert("America/New_York").tz_localize(None)
    return ts


def news_frame_from_items(items: Iterable[Any]) -> pd.DataFrame:
    """Pure conversion of NewsBus items (ScoredNewsItem, or anything with
    similar attributes) into the DataFrame shape families expect:
    created_at (naive ET), headline, source, sentiment. Empty frame for
    no items, read defensively via getattr for attribute variance.
    """
    rows = []
    for item in items or []:
        pub = getattr(item, "published_time", None)
        if pub is None:
            pub = getattr(item, "created_at", None)
        ts = _to_naive_et(pd.Timestamp(pub)) if pub is not None else pd.NaT
        sentiment = getattr(item, "sentiment_score", None)
        if sentiment is None:
            sentiment = getattr(item, "sentiment", 0.0)
        rows.append({
            "created_at": ts,
            "headline": getattr(item, "headline", ""),
            "source": getattr(item, "source", ""),
            "sentiment": float(sentiment or 0.0),
        })
    return pd.DataFrame(rows, columns=_NEWS_FRAME_COLUMNS)


def with_synthetic_next_bar(bars5: pd.DataFrame, now_px: float) -> pd.DataFrame:
    """Append one synthetic bar at `bars5.index[-1] + 5min` representing
    "right now": open=high=low=close=now_px, volume=0. Families produce
    entries at bar i+1's open, so an entry landing on this synthetic
    timestamp means "enter now at the current price".

    v-strategy-fit-review-2026-09-25: pd.concat drops DataFrame.attrs, so
    attrs are copied through explicitly (the adapter relies on
    bars.attrs["symbol"] surviving this call).
    """
    if bars5.empty:
        return bars5
    next_ts = bars5.index[-1] + pd.Timedelta(minutes=5)
    row = pd.DataFrame(
        {"open": [now_px], "high": [now_px], "low": [now_px], "close": [now_px], "volume": [0]},
        index=pd.Index([next_ts], name=bars5.index.name),
    )
    out = pd.concat([bars5, row])
    out.attrs = dict(bars5.attrs)
    return out


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def entry_to_signal(entry: Entry, family: str, params: dict, verdict: str, score: float, symbol: str) -> TradingSignal:
    """`entry.symbol` falls back to "" when the family's make_entry() call
    didn't pass one explicitly (the common case) — `symbol` is the
    market-data symbol the adapter is actually evaluating, and always wins
    when entry.symbol is empty."""
    strength = _clamp(float(score) * 2.0, 0.2, 1.0)
    is_day_trade = entry.hold_kind == "intraday"
    reasoning = {
        "strategy": f"fit:{family}",
        "fit_family": family,
        "fit_params": params,
        "fit_verdict": verdict,
        "hold_kind": entry.hold_kind,
        "entry_pattern": entry.tag,
        "max_hold_sessions": None if is_day_trade else entry.time_stop,
        "is_day_trade": is_day_trade,
        "day_trade_size_multiplier": 0.5 if is_day_trade else 1.0,
        "flatten_hour": 15 if is_day_trade else None,
        "strategy_size_multiplier": 1.0,
    }
    return TradingSignal(
        symbol=entry.symbol or symbol,
        signal_type=SignalType.BUY if entry.side == "long" else SignalType.SELL,
        strength=strength,
        entry_price=entry.entry_px,
        stop_loss=entry.stop_px,
        take_profit=entry.target_px,
        position_size=0,
        reasoning=reasoning,
        confidence=strength,
    )


class FitLaneStrategy(TradingStrategyWithCommentary):
    """Live adapter: runs one `strategies/families` entry function per
    tick, gated by `StrategyFit` verdicts, shadow-first per family."""

    def __init__(
        self,
        commentary,
        family_name: str,
        fit,
        bars_source: Callable[[str], tuple],
        live_families: Callable[[], tuple],
        ledger_dir: Path = LEDGER_DIR,
        clock: Optional[Callable[[], pd.Timestamp]] = None,
    ) -> None:
        super().__init__(commentary)
        # v-strategy-fit-review-2026-09-25: resolved eagerly so a typo'd
        # family name in FIT_LANES_ENABLED raises KeyError at registration
        # time (caught by the engine's registration loop) instead of on
        # every analysis tick.
        self._fam: Family = get_family(family_name)
        self.name = f"fit:{family_name}"
        self._family_name = family_name
        self._fit = fit
        self._bars_source = bars_source
        self._live_families = live_families
        self._ledger_dir = Path(ledger_dir)
        self._clock = clock or (lambda: pd.Timestamp.now(tz="America/New_York").tz_localize(None))
        self._news_bus = None

    def set_news_bus(self, bus) -> None:
        self._news_bus = bus

    async def generate_signal_with_commentary(self, market_data: MarketData) -> Optional[TradingSignal]:
        symbol = market_data.symbol
        family = self._family_name

        row_long = self._fit.lookup(symbol, family, "long")
        row_short = self._fit.lookup(symbol, family, "short")
        if row_long is None and row_short is None:
            self._log_decision(market_data, "skip", "fit_no_row", skip_snapshot=True)
            return None

        # v-strategy-fit-review-2026-09-25: everything past this point can
        # raise (bad/missing bars, indicator math, a broken family fn) —
        # contain it so one symbol's bad data can't take the strategy down.
        try:
            return await self._generate(market_data, symbol, row_long, row_short)
        except Exception as exc:
            logger.warning("fit_lane: %s errored for %s: %s", family, symbol, exc)
            self._log_decision(market_data, "error", "fit_lane_error", skip_snapshot=True)
            return None

    async def _generate(self, market_data: MarketData, symbol: str, row_long, row_short) -> Optional[TradingSignal]:
        family = self._family_name

        bars5, daily = await asyncio.to_thread(self._bars_source, symbol)
        if bars5 is None:
            bars5 = pd.DataFrame()

        now = self._clock()
        closed = closed_bars(bars5, now)
        if len(closed) < 60:
            self._log_decision(market_data, "skip", "insufficient_bars", skip_snapshot=True)
            return None

        # v-strategy-fit-review-2026-09-25: staleness guard. Post-close the
        # synthetic bar would land after 15:55 ET; pre-open the last closed
        # bar is from the prior session. Either way the snapshot no longer
        # reflects "now" — skip instead of computing (and ledger-logging)
        # entries against it. The market-hours gate downstream would block
        # the order anyway; this just keeps the shadow ledger clean.
        last_closed_ts = closed.index[-1]
        now_ts_val = pd.Timestamp(now)
        if (now_ts_val - last_closed_ts) > _FRESHNESS_LIMIT:
            self._log_decision(market_data, "skip", "stale_bars", skip_snapshot=True)
            return None
        if (last_closed_ts + pd.Timedelta(minutes=5)).time() > _RTH_CLOSE:
            self._log_decision(market_data, "skip", "stale_bars", skip_snapshot=True)
            return None

        params = (
            self._fit.params_for(symbol, family, "long")
            or self._fit.params_for(symbol, family, "short")
            or median_params(self._fam.grid)
        )

        bars = with_synthetic_next_bar(closed, market_data.close)
        # v-strategy-fit-review-2026-09-25: make_entry() falls back to
        # bars5.attrs.get("symbol", "") when a family doesn't pass symbol
        # explicitly (the common case) — set it so the emitted entry (and
        # the resulting signal) carries the real symbol.
        bars.attrs["symbol"] = symbol

        spy5 = None
        try:
            spy_bars, _spy_daily = await asyncio.to_thread(self._bars_source, "SPY")
            if spy_bars is not None and not spy_bars.empty:
                spy5 = spy_bars
        except Exception:
            spy5 = None

        try:
            news_items = await self._news_bus.get_items(symbol) if self._news_bus is not None else []
        except Exception:
            news_items = []
        news = news_frame_from_items(news_items)

        ind = indicator_frame(bars, daily, spy5)
        entries = self._fam.fn(bars, ind, daily, news, params)

        now_ts = bars.index[-1]
        rows_by_side = {"long": row_long, "short": row_short}
        kept = [e for e in entries if e.entry_ts == now_ts and rows_by_side.get(e.side) is not None]
        if not kept:
            self._log_decision(market_data, "skip", "no_entry_now", skip_snapshot=True)
            return None

        best_entry, best_row = self._append_ledger(now_ts, symbol, kept, rows_by_side, params)

        if family not in self._live_families():
            self._log_decision(
                market_data, "shadow", "fit_lane_shadow",
                would_entry_price=best_entry.entry_px,
                would_stop_loss=best_entry.stop_px,
                would_take_profit=best_entry.target_px,
            )
            return None

        action = "signal_buy" if best_entry.side == "long" else "signal_sell"
        self._log_decision(market_data, action, "fit_lane_live")
        return entry_to_signal(best_entry, family, params, best_row.verdict, best_row.score, symbol)

    def _append_ledger(self, now_ts, symbol: str, kept: list, rows_by_side: dict, params: dict) -> Tuple[Entry, Any]:
        """Picks the best-scoring kept entry and appends one ledger line per
        kept entry. The ledger write is best-effort: a bad ledger_dir (e.g.
        unwritable, or a file where a directory is expected) must not stop
        a live/shadow decision from being made — it's just logged."""
        best_entry: Optional[Entry] = None
        best_row = None
        for entry in kept:
            row = rows_by_side[entry.side]
            if best_row is None or row.score > best_row.score:
                best_entry, best_row = entry, row

        try:
            self._ledger_dir.mkdir(parents=True, exist_ok=True)
            with open(ledger_path(self._family_name, self._ledger_dir), "a") as fh:
                for entry in kept:
                    row = rows_by_side[entry.side]
                    fh.write(json.dumps({
                        "ts": now_ts.isoformat(),
                        "symbol": symbol,
                        "side": entry.side,
                        "entry_px": entry.entry_px,
                        "stop_px": entry.stop_px,
                        "target_px": entry.target_px,
                        "time_stop": entry.time_stop,
                        "hold_kind": entry.hold_kind,
                        "tag": entry.tag,
                        "params": params,
                        "verdict": row.verdict,
                        "score": row.score,
                    }) + "\n")
        except Exception as exc:
            logger.warning("fit_lane: ledger write failed for %s/%s: %s", self._family_name, symbol, exc)

        return best_entry, best_row
