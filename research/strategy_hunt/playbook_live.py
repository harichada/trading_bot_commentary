"""v-playbook-desk-b2-2026-09-25 (Task 3): the playbook's live `bot_trades` tape, plus the
value coercions both tapes share.

Split out of `playbook_update.py` (fix r2) when the entry-time regime lookup pushed that module
past the 400-line budget. This is the lower layer: it knows nothing about playbook rules, and
`playbook_update` re-exports everything here so existing imports keep working.

The only Postgres touch point is `sql_reader(engine)`; every other entry point takes an injected
`reader(query, params) -> DataFrame`, which is what makes the updater testable offline.
"""
from __future__ import annotations

import bisect
import json
import logging
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from core.desk.playbook import ResolvedTrade

logger = logging.getLogger("strategy_hunt.playbook_update")

__all__ = ["FAMILY_BY_STRATEGY", "LIVE_LOOKBACK_DAYS", "LIVE_REGIME_SQL", "LIVE_TRADES_SQL",
           "REGIME_MATCH_WINDOW", "REGIME_VALUES", "UNKNOWN_REGIME", "family_for",
           "live_resolved_trades", "resolved_from_live", "sql_reader"]


# Research-side copy of the live strategy→family map (`core/strategy_fit_gate.LANE_MAP`; the B2
# plan cites it as `strategies.fit_lane.LANE_MAP`, which exports nothing of the sort). Copied
# rather than imported so the nightly research path never depends on the live gate module, and
# pinned equal to it by tests/research/strategy_hunt/test_playbook_update.py.
FAMILY_BY_STRATEGY = {
    "mean_reversion": "mean_rev_intraday", "mean_reversion_short": "mean_rev_intraday",
    "oversold_bounce_v2": "mean_rev_intraday",
    "day_trade_momentum": "dt_momentum", "day_trade_momentum_short": "dt_momentum",
    "orb_contraction_rvol": "orb",
    "news_sentiment": "news_catalyst", "free_news_sentiment": "news_catalyst",
}
FIT_PREFIX = "fit:"
UNKNOWN_REGIME = "unknown"
# The regime labels core/market_context.py emits and the scorer conditions on; anything
# else in an audit row is not a regime we can key on.
REGIME_VALUES = ("risk_on", "risk_off", "mixed")

# Both reads are bounded and parameterized (fix r1, finding 7): the rules only ever look back
# RECENT_DAYS, so dragging years of bot_trades through pandas every night buys nothing.
LIVE_LOOKBACK_DAYS = 120
LIVE_TRADES_SQL = ("SELECT symbol, side, strategy, entry_time, exit_time, pnl_r "
                   "FROM bot_trades WHERE pnl_r IS NOT NULL AND exit_time >= :since")
LIVE_REGIME_SQL = ("SELECT symbol, ts, details_json FROM bot_decisions "
                   "WHERE component IN ('strategy_fit', 'playbook') AND details_json IS NOT NULL "
                   "AND ts >= :since")
# How far before an entry an audit row may sit and still describe that entry's regime.
REGIME_MATCH_WINDOW = pd.Timedelta(minutes=30)
_SIDE_ALIASES = {"long": "long", "buy": "long", "short": "short", "sell": "short"}


# ── Live tape ──

def sql_reader(engine: Any) -> Callable[..., pd.DataFrame]:
    """A `reader(query, params) -> DataFrame` over `engine` (the only Postgres touch point)."""
    def _read(query: str, params: Optional[dict] = None) -> pd.DataFrame:
        from sqlalchemy import text

        with engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params or {})

    return _read


def _regime_index(decisions: Optional[pd.DataFrame]) -> Dict[str, tuple]:
    """symbol → (sorted timestamps, regimes) from the `bot_decisions` audit rows."""
    if decisions is None or not isinstance(decisions, pd.DataFrame) or decisions.empty:
        return {}
    rows: Dict[str, List[tuple]] = {}
    for rec in decisions.to_dict("records"):
        regime = _regime_from_details(rec.get("details_json"))
        if regime is None:
            continue
        ts = _ts(rec.get("ts"))
        symbol = str(rec.get("symbol") or "").upper()
        if not symbol or pd.isna(ts):
            continue
        rows.setdefault(symbol, []).append((ts, regime))
    # Two plain lists per symbol so the per-trade bisect below is O(log n), not O(n).
    return {sym: tuple(map(list, zip(*sorted(vals)))) for sym, vals in rows.items()}


def _regime_at(index: Dict[str, tuple], symbol: str, when: Any) -> str:
    """The regime of the latest audit row at or before `when`, within
    `REGIME_MATCH_WINDOW`; "unknown" when there is none."""
    pair = index.get(str(symbol or "").upper())
    ts = _ts(when)
    if not pair or pd.isna(ts):
        return UNKNOWN_REGIME
    stamps, regimes = pair
    pos = bisect.bisect_right(stamps, ts) - 1
    if pos < 0 or ts - stamps[pos] > REGIME_MATCH_WINDOW:
        return UNKNOWN_REGIME
    return regimes[pos]


def resolved_from_live(trades: Optional[pd.DataFrame],
                       decisions: Optional[pd.DataFrame] = None) -> List[ResolvedTrade]:
    """`bot_trades` rows → `ResolvedTrade`s, keeping only rows with a readable R, a
    readable side and a strategy this research map knows."""
    if trades is None or not isinstance(trades, pd.DataFrame) or trades.empty:
        return []
    index = _regime_index(decisions)
    out: List[ResolvedTrade] = []
    for rec in trades.to_dict("records"):
        family = family_for(rec.get("strategy"))
        side = _norm_side(rec.get("side"))
        if family is None or side is None:
            continue
        try:
            r = float(rec.get("pnl_r"))
        except (TypeError, ValueError):
            continue
        if not pd.notna(r):
            continue
        exit_ts = _iso(rec.get("exit_time"))
        if not exit_ts:
            logger.warning("playbook: skipping a %s live trade with an unreadable exit_time (%r)",
                           family, rec.get("exit_time"))
            continue
        out.append(ResolvedTrade(family=family, side=side,
                                 regime=_regime_at(index, rec.get("symbol"), rec.get("entry_time")),
                                 r=r, exit_ts=exit_ts, source="live"))
    return out


def live_resolved_trades(reader: Optional[Callable[..., pd.DataFrame]],
                         now: Any = None) -> List[ResolvedTrade]:
    """The live tape through `reader(query, params)`, bounded to the last
    `LIVE_LOOKBACK_DAYS`. Never raises: the nightly run must still rewrite the playbook
    from the shadow tape when Postgres is unreachable."""
    if reader is None:
        return []
    since = _ts(now if now is not None else pd.Timestamp.utcnow())
    if pd.isna(since):
        since = pd.Timestamp.utcnow().tz_localize(None)
    params = {"since": (since - pd.Timedelta(days=LIVE_LOOKBACK_DAYS)).to_pydatetime()}
    try:
        trades = reader(LIVE_TRADES_SQL, params)
        decisions = reader(LIVE_REGIME_SQL, params)
    except Exception as exc:
        logger.warning("playbook: live bot_trades read failed (%s); shadow tape only", exc)
        return []
    try:
        return resolved_from_live(trades, decisions)
    except Exception as exc:
        logger.warning("playbook: live bot_trades frame unusable (%s); shadow tape only", exc)
        return []


# ── Shared coercions ──

def family_for(strategy: Any) -> Optional[str]:
    """Family for a live `bot_trades.strategy` value: the research copy of the lane map,
    plus the `fit:<family>` names the fit lane itself writes."""
    if not isinstance(strategy, str) or not strategy:
        return None
    if strategy in FAMILY_BY_STRATEGY:
        return FAMILY_BY_STRATEGY[strategy]
    if strategy.startswith(FIT_PREFIX) and len(strategy) > len(FIT_PREFIX):
        return strategy[len(FIT_PREFIX):]
    return None


def _norm_side(side: Any) -> Optional[str]:
    if not isinstance(side, str):
        return None
    return _SIDE_ALIASES.get(side.strip().lower())


def _norm_regime(regime: Any) -> str:
    return regime if isinstance(regime, str) and regime in REGIME_VALUES else UNKNOWN_REGIME


def _regime_from_details(raw: Any) -> Optional[str]:
    """The regime an audit row's `details_json` recorded, or None. A malformed blob is
    simply no evidence."""
    details = raw
    if isinstance(raw, str):
        try:
            details = json.loads(raw)
        except ValueError:
            return None
    if not isinstance(details, dict):
        return None
    regime = details.get("regime")
    return regime if isinstance(regime, str) and regime in REGIME_VALUES else None


def _ts(value: Any) -> Any:
    """`value` as a tz-naive UTC Timestamp (or NaT): Postgres hands back tz-aware stamps
    and resolved shadow trades are naive, and comparing the two raises — which would cost
    the whole live tape."""
    ts = pd.to_datetime(value, errors="coerce")
    if not pd.isna(ts) and getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def _iso(value: Any) -> str:
    ts = _ts(value)
    return "" if pd.isna(ts) else pd.Timestamp(ts).isoformat()
