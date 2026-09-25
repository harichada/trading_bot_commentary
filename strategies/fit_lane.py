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

v-strategy-fit-review2-2026-09-25 (whole-branch review fixes):
  Critical 1. a live family only emits a TradingSignal when the entry's own
     side carries a walk-forward PASS row (`live_eligible`); WATCH, FAIL and
     cold-start rows stay shadow ("fit_lane_shadow_verdict").
  Important 4. session-open mode: the synthetic bar is `next_rth_bar(...)`
     (15:55 -> next weekday 09:30), the freshness window is that bar's
     15-minute lead-in, and the daily frame gets a placeholder row for a new
     session date so `bar_of_day == 0` families are reachable live.
  Minor. `_clamp` and the best-row comparison both guard NaN.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
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


def ledger_path(family: str, ledger_dir: Path = LEDGER_DIR) -> Path:
    """Path to the shadow ledger for `family` under `ledger_dir` (default `data/`)."""
    return Path(ledger_dir) / f"shadow_fit_{family}.ndjson"


# v-strategy-fit-review2-2026-09-25: the pure frame/verdict helpers live in
# strategies/fit_lane_frames.py (400-line file budget); re-exported here so
# existing `from strategies.fit_lane import ...` call sites keep working.
from strategies.fit_lane_frames import (  # noqa: E402
    _FRESHNESS_LIMIT,
    _RTH_CLOSE,
    _clamp,
    _to_naive_et,
    live_eligible,
    news_frame_from_items,
    next_rth_bar,
    with_daily_placeholder,
    with_synthetic_next_bar,
)

__all__ = [
    "FitLaneStrategy", "entry_to_signal", "ledger_path", "live_eligible",
    "news_frame_from_items", "next_rth_bar", "with_daily_placeholder",
    "with_synthetic_next_bar",
]


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
        # v-strategy-fit-review-2026-09-25: closed_bars() compares its
        # index to a Timestamp and raises TypeError on an empty frame's
        # default RangeIndex — check emptiness before calling it rather
        # than coercing None to an empty frame and passing it through.
        if bars5 is None or bars5.empty:
            self._log_decision(market_data, "skip", "insufficient_bars", skip_snapshot=True)
            return None

        now = self._clock()
        closed = closed_bars(bars5, now)
        if len(closed) < 60:
            self._log_decision(market_data, "skip", "insufficient_bars", skip_snapshot=True)
            return None

        # v-strategy-fit-review-2026-09-25: staleness guard, reworked by
        # v-strategy-fit-review2-2026-09-25 (Important 4) for session opens.
        # The synthetic bar is the bar that *follows* the last closed one on
        # the RTH grid — +5min intraday, or the next weekday's 09:30 when the
        # last closed bar is the 15:55 session finale. The snapshot is fresh
        # only while the clock sits inside that bar's 15-minute lead-in
        # window, which covers both the intraday case and the 09:30-09:45
        # opening window session-open families need. Outside it the snapshot
        # no longer reflects "now" — skip instead of computing (and
        # ledger-logging) entries against it. The market-hours gate
        # downstream would block the order anyway; this just keeps the
        # shadow ledger clean.
        last_closed_ts = closed.index[-1]
        synthetic_ts = next_rth_bar(last_closed_ts)
        now_ts_val = pd.Timestamp(now)
        if not (synthetic_ts <= now_ts_val <= synthetic_ts + _FRESHNESS_LIMIT):
            self._log_decision(market_data, "skip", "stale_bars", skip_snapshot=True)
            return None
        # Retained 15:55 RTH cap: next_rth_bar() can only land past the close
        # if the incoming frame itself carries extended-hours bars.
        if synthetic_ts.time() > _RTH_CLOSE:
            self._log_decision(market_data, "skip", "stale_bars", skip_snapshot=True)
            return None

        params = (
            self._fit.params_for(symbol, family, "long")
            or self._fit.params_for(symbol, family, "short")
            or median_params(self._fam.grid)
        )

        bars = with_synthetic_next_bar(closed, market_data.close, synthetic_ts)
        # Session-open synthetic bar: the new session's date isn't in the
        # provider's daily frame yet, so daily_features' one-session lag has
        # nothing to resolve against. A placeholder row fixes that without
        # inventing price data (Important 4).
        daily = with_daily_placeholder(daily, synthetic_ts)
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

        # v-strategy-fit-review2-2026-09-25 (Critical 1): being listed in
        # FIT_LANE_LIVE_ENTRIES_ENABLED is necessary but not sufficient — the
        # entry's own side must have a walk-forward PASS row. A WATCH, FAIL,
        # or cold-start row keeps the entry in shadow (ledger line already
        # written above) instead of passing the verdict through as metadata
        # on a live signal.
        live_kept = [e for e in kept if live_eligible(rows_by_side.get(e.side))]
        if not live_kept:
            self._log_decision(
                market_data, "shadow", "fit_lane_shadow_verdict",
                fit_verdict=getattr(best_row, "verdict", ""),
                fit_source=getattr(best_row, "source", ""),
                would_entry_price=best_entry.entry_px,
                would_stop_loss=best_entry.stop_px,
                would_take_profit=best_entry.target_px,
            )
            return None

        live_entry, live_row = self._best_of(live_kept, rows_by_side)
        action = "signal_buy" if live_entry.side == "long" else "signal_sell"
        self._log_decision(market_data, action, "fit_lane_live")
        return entry_to_signal(live_entry, family, params, live_row.verdict, live_row.score, symbol)

    def _append_ledger(self, now_ts, symbol: str, kept: list, rows_by_side: dict, params: dict) -> Tuple[Entry, Any]:
        """Picks the best-scoring kept entry and appends one ledger line per
        kept entry. The ledger write is best-effort: a bad ledger_dir (e.g.
        unwritable, or a file where a directory is expected) must not stop
        a live/shadow decision from being made — it's just logged."""
        best_entry, best_row = self._best_of(kept, rows_by_side)

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

    @staticmethod
    def _best_of(entries: list, rows_by_side: dict) -> Tuple[Entry, Any]:
        """Highest-scoring entry among `entries` (first one wins ties).

        v-strategy-fit-review2-2026-09-25 (Minor): a NaN score must never
        read as "higher" than a real one, so the comparison is on a
        NaN-to--inf coercion rather than raw `row.score > best.score`.
        """
        def _score(row: Any) -> float:
            try:
                v = float(getattr(row, "score", float("nan")))
            except (TypeError, ValueError):
                return float("-inf")
            return v if math.isfinite(v) else float("-inf")

        best_entry: Optional[Entry] = None
        best_row = None
        for entry in entries:
            row = rows_by_side[entry.side]
            if best_row is None or _score(row) > _score(best_row):
                best_entry, best_row = entry, row
        return best_entry, best_row
