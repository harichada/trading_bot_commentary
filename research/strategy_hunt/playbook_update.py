"""v-playbook-desk-b2-2026-09-25 (Task 3): the nightly playbook updater.

Reads two tapes of *closed, R-measured* trades and rewrites
`<desk_dir>/playbook.json` through the pure rules in `core/desk/playbook.py`:

* shadow — `strategies/fit_lane.py` ledgers resolved by `shadow_resolver.resolve`
  (regime comes from the resolved `SimTrade.regime`), and
* live — `bot_trades` rows with a non-null `pnl_r` whose `strategy` maps to a
  family, with the regime joined from the `bot_decisions` audit row that
  preceded the entry (else "unknown").

Import direction: `research` may import `core` and `strategies`; `core` must
never import `research`. The value types and the transition rules therefore live
in `core/desk/playbook.py` and this module only feeds them.

Offline-testable by construction: the Postgres side arrives through an injected
`reader(query) -> DataFrame` callable (`sql_reader(engine)` builds the real one),
so no test needs a database.
"""
from __future__ import annotations

import bisect
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from core.desk.playbook import (
    RECENT_WINDOW,
    SUSPENSION_LAPSE_DAYS,
    Playbook,
    PlaybookKey,
    ResolvedTrade,
    carry_forward,
    load_playbook,
    playbook_path,
    save_playbook,
    update_entry,
)
from core.desk.playbook import DEFAULT_DESK_DIR as DEFAULT_DESK_DIR
from research.strategy_hunt.shadow_resolver import DEFAULT_LEDGER_DIR, default_ledger_path, read_ledger, resolve
from strategies.families import family_names

logger = logging.getLogger("strategy_hunt.playbook_update")

__all__ = ["FAMILY_BY_STRATEGY", "LIVE_LOOKBACK_DAYS", "LIVE_REGIME_SQL", "LIVE_TRADES_SQL",
           "RECENT_DAYS", "best_verdict", "group_by_key", "live_resolved_trades", "resolved_from_live",
           "resolved_from_sim", "resolved_trades_by_key", "sql_reader", "update_playbook"]

# Research-side copy of the live strategy→family map (`core/strategy_fit_gate.LANE_MAP`;
# the B2 plan cites it as `strategies.fit_lane.LANE_MAP`, which re-exports nothing of the
# sort). Copied rather than imported so the nightly research path never depends on the
# live gate module, and pinned equal to it by
# tests/research/strategy_hunt/test_playbook_update.py.
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
VERDICT_RANK = {"FAIL": 0, "WATCH": 1, "PASS": 2}

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
# The rules' sample is time-bounded before the last-RECENT_WINDOW slice (fix r1, finding 3), so a
# stale losing streak cannot re-suspend a key every night for ever.
RECENT_DAYS = 90

_SIDE_ALIASES = {"long": "long", "buy": "long", "short": "short", "sell": "short"}


# ── Shadow tape ──

def resolved_from_sim(sim_trades: Iterable[Any], family: str) -> List[ResolvedTrade]:
    """`SimTrade`s → `ResolvedTrade`s. A non-finite R is dropped (not a measurement)."""
    out: List[ResolvedTrade] = []
    for t in sim_trades:
        try:
            r = float(t.r_multiple)
        except (TypeError, ValueError):
            continue
        if not pd.notna(r):
            continue
        side = _norm_side(getattr(t, "side", ""))
        if side is None:
            continue
        exit_ts = _iso(getattr(t, "exit_ts", None))
        if not exit_ts:
            logger.warning("playbook: skipping a %s shadow trade with an unreadable exit_ts", family)
            continue
        out.append(ResolvedTrade(family=getattr(t, "family", "") or family, side=side,
                                 regime=_norm_regime(getattr(t, "regime", "")),
                                 r=r, exit_ts=exit_ts, source="shadow"))
    return out


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


# ── Merge ──

def group_by_key(trades: Iterable[ResolvedTrade]) -> Dict[PlaybookKey, List[ResolvedTrade]]:
    """(family, side, regime) → trades, each list oldest first (newest last), which is
    the ordering `update_entry`'s window contract expects."""
    grouped: Dict[PlaybookKey, List[ResolvedTrade]] = {}
    for t in trades:
        grouped.setdefault(t.key, []).append(t)
    return {k: sorted(v, key=lambda t: t.exit_ts) for k, v in grouped.items()}


def resolved_trades_by_key(root: Any, families: Sequence[str], engine: Any = None,
                           reader: Optional[Callable[..., pd.DataFrame]] = None,
                           ledger_dir: Any = DEFAULT_LEDGER_DIR,
                           now: Any = None) -> Dict[PlaybookKey, List[ResolvedTrade]]:
    """Every resolved trade this root knows about, keyed by (family, side, regime).

    One family failing to resolve (a truncated or hand-edited ledger) is logged and
    skipped: the nightly playbook must still be rebuilt from the families that work."""
    trades: List[ResolvedTrade] = []
    for family in families or ():
        path = default_ledger_path(family, Path(ledger_dir))
        try:
            records = read_ledger(path)
            if not records:
                continue
            trades.extend(resolved_from_sim(resolve(records, Path(root), family), family))
        except Exception as exc:
            logger.warning("playbook: could not resolve the shadow ledger for %s (%s); skipping",
                           family, exc)
    trades.extend(live_resolved_trades(reader if reader is not None else
                                       (sql_reader(engine) if engine is not None else None), now=now))
    return group_by_key(trades)


# ── Fit verdicts from the snapshot ──

def best_verdict(snapshot: Any, family: str, side: str, regime: str) -> Optional[str]:
    """The best fit verdict for this key (PASS > WATCH > FAIL), or None if research has
    no row for it.

    Candidates are the universe-pooled row for that (family, side, regime) and every
    per-symbol regime row for it. With no regime evidence ("unknown") the unconditional
    per-symbol rows are used instead — the same fallback the live fit gate makes.
    Malformed snapshot sections degrade to "no row", never an exception."""
    if not isinstance(snapshot, dict):
        return None
    regime_key = f"{family}:{side}:{regime}"
    verdicts: List[Optional[str]] = []

    pooled = snapshot.get("pooled_rows")
    if isinstance(pooled, dict):
        verdicts.append(_verdict_of(pooled.get(regime_key)))

    per_symbol = snapshot.get("regime_rows")
    if isinstance(per_symbol, dict):
        for rows in per_symbol.values():
            if isinstance(rows, dict):
                verdicts.append(_verdict_of(rows.get(regime_key)))

    if regime not in REGIME_VALUES:
        plain = snapshot.get("rows")
        if isinstance(plain, dict):
            for rows in plain.values():
                if isinstance(rows, dict):
                    verdicts.append(_verdict_of(rows.get(f"{family}:{side}")))

    known = [v for v in verdicts if v in VERDICT_RANK]
    return max(known, key=lambda v: VERDICT_RANK[v]) if known else None


# ── The nightly entry point ──

def _cutoffs(now: Any) -> Tuple[str, str]:
    """`(window_cutoff, fresh_cutoff)` as ISO strings comparable against `ResolvedTrade.exit_ts`.

    Both are "" on an unreadable clock: the window then stays unbounded and nothing lapses,
    which is the safe direction — a broken clock must not silently un-suspend a losing key."""
    ts = _ts(now)
    if pd.isna(ts):
        logger.warning("playbook: unreadable now=%r; leaving the window unbounded", now)
        return "", ""
    return (_iso(ts - pd.Timedelta(days=RECENT_DAYS)), _iso(ts - pd.Timedelta(days=SUSPENSION_LAPSE_DAYS)))


def update_playbook(root: Any, snapshot: Any, now: Any, desk_dir: Any = DEFAULT_DESK_DIR,
                    families: Optional[Sequence[str]] = None, engine: Any = None,
                    reader: Optional[Callable[..., pd.DataFrame]] = None,
                    ledger_dir: Any = DEFAULT_LEDGER_DIR) -> Playbook:
    """Rewrite `<desk_dir>/playbook.json` from tonight's resolved trades and return it.

    Per key: the sample is every resolved trade newer than `RECENT_DAYS`, then the last
    `RECENT_WINDOW` of those. A key with no sample keeps its previous entry. Either way the
    result goes through `carry_forward`, so a suspension that has gone `SUSPENSION_LAPSE_DAYS`
    without any resolved trade lapses back to probation (fix r1, finding 3) — a suspended key
    places no live orders, so silence is not evidence against it."""
    fams = list(families) if families is not None else family_names()
    grouped = resolved_trades_by_key(root, fams, engine=engine, reader=reader,
                                     ledger_dir=ledger_dir, now=now)
    previous = load_playbook(playbook_path(desk_dir))
    window_cutoff, fresh_cutoff = _cutoffs(now)

    entries: Dict[PlaybookKey, Any] = {}
    for key in sorted(set(grouped) | set(previous.entries)):
        trades = grouped.get(key, ())
        window = [t for t in trades if t.exit_ts >= window_cutoff][-RECENT_WINDOW:]
        has_recent = (not fresh_cutoff) or any(t.exit_ts >= fresh_cutoff for t in trades)
        prev = previous.entries.get(key)
        if window:
            entry = update_entry(prev, window, best_verdict(snapshot, *key), now)
        elif prev is not None:
            entry = prev
        else:
            continue
        entries[key] = carry_forward(entry, now, has_recent)

    playbook = Playbook(entries=entries, updated_at=str(now))
    save_playbook(playbook, playbook_path(desk_dir))
    return playbook


# ── Small helpers ──

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


def _verdict_of(row: Any) -> Optional[str]:
    if not isinstance(row, dict):
        return None
    verdict = row.get("verdict")
    return verdict if isinstance(verdict, str) else None


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
