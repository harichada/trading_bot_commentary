"""v-playbook-desk-b2-2026-09-25 (Task 3): the nightly playbook updater.

Reads two tapes of *closed, R-measured* trades and rewrites
`<desk_dir>/playbook.json` through the pure rules in `core/desk/playbook.py`:

* shadow — `strategies/fit_lane.py` ledgers resolved by `shadow_resolver.resolve`,
  labelled with the entry-time regime (`data/regime.py`) because the resolver leaves
  every trade `unknown`, and
* live — `bot_trades` rows with a non-null `pnl_r` whose `strategy` maps to a
  family, with the regime joined from the `bot_decisions` audit row that
  preceded the entry (else "unknown").

Import direction: `research` may import `core` and `strategies`; `core` must
never import `research`. The value types and the transition rules therefore live
in `core/desk/playbook.py` and this module only feeds them.

Offline-testable by construction: the Postgres side arrives through an injected
`reader(query, params) -> DataFrame` callable (`sql_reader(engine)` builds the real
one), so no test needs a database.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
from research.strategy_hunt.run import load_context
from research.strategy_hunt.shadow_resolver import DEFAULT_LEDGER_DIR, default_ledger_path, read_ledger, resolve
from strategies.families import family_names

# The live tape and the coercions both tapes share live one layer down (fix r2: this module hit
# the 400-line budget). Re-exported here so `from research.strategy_hunt.playbook_update import
# LIVE_TRADES_SQL, resolved_from_live, ...` keeps working for every existing caller and test.
from research.strategy_hunt.playbook_live import (  # noqa: F401
    FAMILY_BY_STRATEGY,
    LIVE_LOOKBACK_DAYS,
    LIVE_REGIME_SQL,
    LIVE_TRADES_SQL,
    REGIME_MATCH_WINDOW,
    REGIME_VALUES,
    UNKNOWN_REGIME,
    _iso,
    _norm_regime,
    _norm_side,
    _ts,
    family_for,
    live_resolved_trades,
    resolved_from_live,
    sql_reader,
)

logger = logging.getLogger("strategy_hunt.playbook_update")

__all__ = ["FAMILY_BY_STRATEGY", "LIVE_LOOKBACK_DAYS", "LIVE_REGIME_SQL", "LIVE_TRADES_SQL",
           "RECENT_DAYS", "best_verdict", "group_by_key", "live_resolved_trades", "regime_lookup_for",
           "resolved_from_live", "resolved_from_sim", "resolved_trades_by_key", "sql_reader",
           "update_playbook"]

VERDICT_RANK = {"FAIL": 0, "WATCH": 1, "PASS": 2}
# The rules' sample is time-bounded before the last-RECENT_WINDOW slice (fix r1, finding 3), so a
# stale losing streak cannot re-suspend a key every night for ever.
RECENT_DAYS = 90


# ── Shadow tape ──

def regime_lookup_for(root: Any) -> Optional[Callable[[Any], str]]:
    """The entry-time regime lookup for `root`, or None when there is nothing to label with.

    Built once per updater run from the loader `run.process_symbol` uses — SPY 5-minute bars
    plus daily VIX, thresholds in `research/strategy_hunt/data/regime.py` — because
    `shadow_resolver.resolve` stamps every trade `unknown` (fix r2, finding 1a) and an
    `unknown` key is invisible to the live gate's exact-regime lookup. A root with no SPY/VIX
    context gives an empty lookup, which is the same as none: every trade stays `unknown`."""
    try:
        lookup = load_context(Path(root)).get("regime_lookup")
    except Exception as exc:
        logger.warning("playbook: no entry-time regime lookup for %s (%s); labels stay unknown",
                       root, exc)
        return None
    if lookup is None or len(lookup) == 0:
        logger.info("playbook: %s has no SPY/VIX context; shadow trades stay unknown", root)
        return None
    return lookup


def resolved_from_sim(sim_trades: Iterable[Any], family: str,
                      regime_lookup: Optional[Callable[[Any], str]] = None) -> List[ResolvedTrade]:
    """`SimTrade`s → `ResolvedTrade`s. A non-finite R is dropped (not a measurement).

    `regime_lookup` labels a trade the resolver left `unknown`, from data known at its entry
    bar; a trade the resolver already labelled keeps that label, and a failing lookup degrades
    to `unknown` (warned once per family, not once per trade)."""
    out: List[ResolvedTrade] = []
    failed = 0
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
        regime = _norm_regime(getattr(t, "regime", ""))
        entry_ts = _ts(getattr(t, "entry_ts", None))
        if regime == UNKNOWN_REGIME and regime_lookup is not None and not pd.isna(entry_ts):
            try:
                regime = _norm_regime(regime_lookup(entry_ts))
            except Exception as exc:
                failed += 1
                if failed == 1:
                    logger.warning("playbook: entry-time regime lookup failed for %s (%s); "
                                   "those trades stay unknown", family, exc)
        out.append(ResolvedTrade(family=getattr(t, "family", "") or family, side=side,
                                 regime=regime, r=r, exit_ts=exit_ts, source="shadow"))
    return out


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
    skipped: the nightly playbook must still be rebuilt from the families that work. The
    entry-time regime lookup is built once here and shared by every family."""
    trades: List[ResolvedTrade] = []
    lookup = regime_lookup_for(root) if families else None
    for family in families or ():
        path = default_ledger_path(family, Path(ledger_dir))
        try:
            records = read_ledger(path)
            if not records:
                continue
            trades.extend(resolved_from_sim(resolve(records, Path(root), family), family,
                                            regime_lookup=lookup))
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

def _verdict_of(row: Any) -> Optional[str]:
    if not isinstance(row, dict):
        return None
    verdict = row.get("verdict")
    return verdict if isinstance(verdict, str) else None
