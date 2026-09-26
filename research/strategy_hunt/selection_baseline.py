"""The selection baseline: replay the real trade schedule on seeded random peer symbols.

Batch-2 Task A. The path-shuffle control (`run.shuffle_bars`, `trades_shuffle/`) permutes 5-minute
returns *inside* each session, so every session's open-to-close return survives it. For an intraday
hold that is a real control -- it destroys the path the entry was timed on. For a multi-day or
overnight hold it destroys almost nothing: the claim under test is which symbol was held across those
sessions, and the shuffled tape holds the same symbol across the same sessions. This control asks
that question instead -- keep the real *schedule* and reassign each trade to a random peer symbol
that traded at the same timestamp -- so a family with selection skill beats it while one harvesting
the average stock's drift over those sessions does not.

Preserved: the entry timestamp (the peer is entered on *its* bar at that stamp, so the control trades
the same calendar), the side, the stop/target as the same *fractions* of the entry price (an absolute
distance is meaningless across price levels; the fraction is what the family chose), the `hold_kind`
and time stop, and the peer's own cost model, earnings dates and the run's real regime labels.
Redrawn: selection alone -- uniformly over the symbols with a bar at that stamp, seeded by
`(seed, source symbol, entry_ts)` so a rerun reproduces the tape trade-for-trade regardless of worker
count or file order. `selection_manifest` records how much of each pooled group was paired.

Two honest limitations, because they bound what the control proves. (1) The horizon is the trade's
own `time_stop` column (Batch-2 Task F); a tape written before it falls back to `params_json`
(`TIME_STOP_PARAM_KEYS`) and then to the *realized* hold, which can only shorten such a control's
horizon. (2) Peers come from `<root>/bars5/`, the universe as it exists now, so the control inherits
the real tape's survivorship bias: both tapes must be drawn from the same population to be compared.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from research.strategy_hunt.data.bars import load_bars5
from research.strategy_hunt.data.earnings import earnings_days
from research.strategy_hunt.data.splits import load_adjusted
from research.strategy_hunt.models import TRADE_COLUMNS
from research.strategy_hunt.paths import SELECTION_VARIANT, trades_dir, trades_path
from research.strategy_hunt.run import load_context
from research.strategy_hunt.selection_manifest import (group_key, pairing_counts, real_group_counts,
                                                       replayed_group_counts, underreplayed_groups)
# `run --shuffle-mode selection`'s command lives with the manifest record it writes; re-exported
# because `cli.cmd_run` imports it from here (and it reads as this module's entry point).
from research.strategy_hunt.selection_manifest import run_selection_command  # noqa: F401
from research.strategy_hunt.sim.calibrate import slippage_override_from
from research.strategy_hunt.sim.costs import CostModel
from research.strategy_hunt.sim.simulator import build_session_table, simulate
from strategies.families import get_family
from strategies.families.common import make_entry

logger = logging.getLogger("strategy_hunt.selection_baseline")

CONTEXT_SYMBOLS = ("SPY", "VIX")   # market context, not tradeable candidates (as in `cli._symbols_arg`)
TAG_PREFIX = "sel:"
# The first draw plus three redraws. A peer is redrawn only when it has no bar at that timestamp;
# four misses means essentially nothing traded then, so the trade is dropped and counted rather than
# forced onto a symbol that was not trading.
PEER_DRAWS = 4
# `hold_kind` -> params_json keys carrying that kind's time stop, in priority order. Keyed by hold
# kind because the unit differs (bars vs sessions): `time_stop_days` read into an intraday entry
# would silently mean 5 *bars*.
TIME_STOP_PARAM_KEYS = {"intraday": ("time_stop_bars", "time_stop"),
                        "multiday": ("time_stop_days", "hold_days", "hold_sessions", "time_stop"),
                        "overnight": ()}
OVERNIGHT_TIME_STOP = 1   # unused by the overnight exit rule; `Entry` only requires it positive

DROP_NO_PEER, DROP_NO_ENTRY = "no_peer_bar", "no_entry"
DROP_UNKNOWN_FAMILY, DROP_BAD_GEOMETRY, DROP_BAD_RECORD = "unknown_family", "bad_geometry", "bad_record"
DROP_TZ_AWARE = "tz_aware_trades"

DEFAULT_ENTRY_STYLE = "market"   # a tape written before Task B's `entry_style` column

# Set once per pool worker by `_init_worker`: the context and cost settings are the same for every
# peer, and pickling them per job would copy the SPY tape hundreds of times.
_WORKER: dict[str, Any] = {}


@dataclass(frozen=True)
class ReplayPlan:
    """One real trade's replay assignment: everything needed to build the peer's `Entry`."""

    peer: str
    source_symbol: str
    family: str
    side: str
    entry_ts: pd.Timestamp
    rel_stop: float
    rel_target: float
    time_stop: int
    hold_kind: str
    entry_style: str
    window_id: int
    params_json: str
    # The source trade's regime label: the pooled group this replay is paired into (`selection_manifest`).
    regime: str = "unknown"


def selection_universe(root: Path) -> tuple[str, ...]:
    """Every symbol a trade can be replayed onto: the root's `bars5/` minus the context symbols."""
    return tuple(sorted(p.stem.upper() for p in (Path(root) / "bars5").glob("*.parquet")
                        if p.stem.upper() not in CONTEXT_SYMBOLS))


def real_trades(root: Path) -> pd.DataFrame:
    """Every readable real trade in `<root>/trades/`, concatenated. Per-file isolation matches the
    scorers': an unreadable Parquet is logged and skipped -- a control built from 99% of the tape is
    still a control."""
    frames: list[pd.DataFrame] = []
    for tp in sorted(trades_dir(root).glob("*.parquet")):
        try:
            df = pd.read_parquet(tp)
        except Exception:
            logger.warning("selection: skipping unreadable trades file %s", tp.name, exc_info=True)
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(TRADE_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def bar_presence(root: Path, symbols: tuple[str, ...], stamps: np.ndarray) -> dict[str, np.ndarray]:
    """`symbol -> bool array over `stamps`: has this symbol a 5-minute bar at that stamp? Only the
    timestamps the schedule asks about are materialised (one bool per symbol per distinct entry
    stamp), so each draw's test is O(1) without holding every symbol's bars in memory."""
    out: dict[str, np.ndarray] = {}
    for sym in symbols:
        out[sym] = np.isin(stamps, load_bars5(root, sym).index.values)
    return out


def _seeded_rng(seed: int, symbol: str, entry_ts) -> np.random.Generator:
    """A generator depending only on `(seed, symbol, entry_ts)`. blake2b, not Python's `hash()`
    (salted per process): the draw must be identical in a pool, on a rerun, and on another machine."""
    key = f"{int(seed)}|{symbol}|{pd.Timestamp(entry_ts).isoformat()}".encode()
    return np.random.default_rng(int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big"))


def draw_peer(seed: int, symbol: str, entry_ts, peers: tuple[str, ...],
              has_bar: Callable[[str], bool]) -> Optional[str]:
    """A peer `!= symbol` that has a bar at `entry_ts`, or None after `PEER_DRAWS` misses."""
    if not peers:
        return None
    rng = _seeded_rng(seed, symbol, entry_ts)
    for _ in range(PEER_DRAWS):
        peer = peers[int(rng.integers(len(peers)))]
        if has_bar(peer):
            return peer
    return None


def relative_offsets(entry_px: float, stop_px: float, target_px: float) -> tuple[float, float]:
    """`(|entry - stop| / entry, |target - entry| / entry)` -- the family's choice, scale-free."""
    entry = float(entry_px)
    if not entry > 0:
        return 0.0, 0.0
    return abs(entry - float(stop_px)) / entry, abs(float(target_px) - entry) / entry


def replay_brackets(side: str, fill_open: float, rel_stop: float, rel_target: float) -> tuple[float, float]:
    """The peer's `(stop_px, target_px)`: the same fractions of *its* fill price."""
    sgn = 1.0 if side == "long" else -1.0
    return fill_open * (1.0 - sgn * rel_stop), fill_open * (1.0 + sgn * rel_target)


def time_stop_for(hold_kind: str, params: dict, hold_bars: int, hold_sessions: int,
                  trade_time_stop: Any = 0) -> int:
    """The horizon for the replayed entry. The trade's own `time_stop` column (Batch-2 Task F) wins
    outright -- it *is* `Entry.time_stop`, in the unit `hold_kind` implies; 0 or NaN means a tape
    written before that column and falls through to the recovery (docstring limitation 1)."""
    try:
        if not isinstance(trade_time_stop, bool) and int(trade_time_stop) > 0:
            return int(trade_time_stop)
    except (TypeError, ValueError):               # NaN, None, or a non-numeric record
        pass
    if hold_kind == "overnight":
        return OVERNIGHT_TIME_STOP
    for key in TIME_STOP_PARAM_KEYS.get(hold_kind, ()):
        value = params.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and int(value) > 0:
            return int(value)
    realized = hold_bars if hold_kind == "intraday" else hold_sessions
    try:
        return max(1, int(realized))
    except (TypeError, ValueError):
        return 1


def hold_kind_of(family: str) -> Optional[str]:
    """The family's hold kind, or None when this root's trades name a family the registry lost."""
    try:
        return get_family(family).hold_kind
    except KeyError:
        return None


def _params(params_json: Any) -> dict:
    try:
        out = json.loads(params_json) if isinstance(params_json, str) else {}
    except ValueError:
        return {}
    return out if isinstance(out, dict) else {}


def _entry_style(record: dict) -> str:
    """The trade's entry style, defaulting to `"market"` for a tape written before the column."""
    value = record.get("entry_style")
    return DEFAULT_ENTRY_STYLE if value is None or value != value else str(value)   # NaN: value != value


def build_plans(trades: pd.DataFrame, root: Path, universe: tuple[str, ...], seed: int,
                presence: Optional[dict[str, np.ndarray]] = None,
                kind_of: Callable[[str], Optional[str]] = hold_kind_of
                ) -> tuple[list[ReplayPlan], dict[str, int]]:
    """One `ReplayPlan` per replayable real trade, plus the counts of what could not be replayed."""
    if trades is None or trades.empty:
        return [], {}
    entry_stamps = pd.to_datetime(trades["entry_ts"])
    if getattr(entry_stamps.dt, "tz", None) is not None:
        logger.warning("selection: trades/ has a tz-aware entry_ts column (%s) but bars5 is tz-naive; "
                       "no trade can be matched to a bar. Rewrite the tape tz-naive and re-run.",
                       entry_stamps.dt.tz)
        return [], {DROP_TZ_AWARE: int(len(trades))}
    stamps = np.unique(entry_stamps.to_numpy())
    presence = presence if presence is not None else bar_presence(root, universe, stamps)
    column_of = {pd.Timestamp(ts): i for i, ts in enumerate(stamps)}
    peers_of = {sym: tuple(p for p in universe if p != sym)
                for sym in {str(s).upper() for s in trades["symbol"]}}
    plans: list[ReplayPlan] = []
    dropped: Counter = Counter()
    for record in trades.to_dict("records"):
        try:                    # a single unusable row must not cost the whole control tape
            symbol, family = str(record["symbol"]).upper(), str(record["family"])
            kind = kind_of(family)
            if kind is None:
                dropped[DROP_UNKNOWN_FAMILY] += 1
                continue
            entry_ts = pd.Timestamp(record["entry_ts"])
            column = column_of[entry_ts]
            peer = draw_peer(seed, symbol, entry_ts, peers_of[symbol],
                             lambda p, c=column: bool(presence[p][c]))
            if peer is None:
                dropped[DROP_NO_PEER] += 1
                continue
            rel_stop, rel_target = relative_offsets(record["entry_px"], record["stop_px"], record["target_px"])
            if not (rel_stop > 0 and rel_target > 0):
                dropped[DROP_BAD_GEOMETRY] += 1
                continue
            params = _params(record.get("params_json"))
            plans.append(ReplayPlan(
                peer=peer, source_symbol=symbol, family=family, side=str(record["side"]), entry_ts=entry_ts,
                rel_stop=rel_stop, rel_target=rel_target,
                time_stop=time_stop_for(kind, params, record.get("hold_bars", 1), record.get("hold_sessions", 1),
                                        record.get("time_stop", 0)),
                hold_kind=kind, entry_style=_entry_style(record), window_id=int(record.get("window_id", -1)),
                params_json=str(record.get("params_json") or "{}"), regime=str(record.get("regime", "unknown"))))
        except Exception:
            logger.warning("selection: skipping unusable trade record %s", record.get("symbol"), exc_info=True)
            dropped[DROP_BAD_RECORD] += 1
    return plans, dict(dropped)


def plan_entry(plan: ReplayPlan, bars5: pd.DataFrame):
    """The peer's `Entry` for one plan, or None when the peer cannot take it. The signal bar is the
    bar *before* `entry_ts`, because `make_entry` fills at the next bar's open -- so the fill is the
    peer's own bar at the real trade's entry timestamp."""
    if bars5 is None or bars5.empty:
        return None
    try:
        pos = int(bars5.index.get_loc(plan.entry_ts))
    except (KeyError, TypeError):
        return None
    if pos < 1:
        return None
    fill_open = float(bars5["open"].iloc[pos])
    if not fill_open > 0:
        return None
    stop_px, target_px = replay_brackets(plan.side, fill_open, plan.rel_stop, plan.rel_target)
    return make_entry(bars5, pos - 1, plan.side, stop_px, target_px, plan.time_stop, plan.hold_kind,
                      f"{TAG_PREFIX}{plan.family}", plan.peer, entry_style=plan.entry_style)


def replay_peer(peer: str, plans: list[ReplayPlan], root: Path, context: dict, notional: float,
                cost_mult: float, override: Optional[float]) -> dict[str, Any]:
    """Simulate one peer's replayed entries and write `trades_selection/<peer>.parquet`. One
    `simulate` call per group, because `simulate` stamps one family/window/params on every trade and
    those are the source trade's own provenance."""
    bars5, daily = load_adjusted(root, peer)
    if bars5.empty:
        return {"symbol": peer, "status": "empty", "replayed": 0, "dropped": len(plans), "groups": {}}
    cost_model = CostModel.from_bars(bars5, notional, slippage_override=override).scaled(cost_mult)
    earnings = earnings_days(root, peer, daily)
    table = build_session_table(bars5)
    # The group key carries the plan's pooled group (family, side, regime) as well as the provenance
    # `simulate` stamps, so each call's output size *is* that pooled group's replayed count (fix r1).
    groups: dict[tuple[str, str, str, int, str], list] = {}
    for plan in plans:
        entry = plan_entry(plan, bars5)
        if entry is not None:
            groups.setdefault((plan.family, plan.side, plan.regime, plan.window_id, plan.params_json),
                              []).append(entry)
    trades, replayed_groups = [], Counter()
    for (family, side, regime, window_id, params_json), entries in sorted(groups.items()):
        out = simulate(entries, bars5, cost_model, earnings, family, window_id=window_id,
                       params_json=params_json, regime_lookup=context.get("regime_lookup") or None,
                       table=table)
        replayed_groups[group_key(family, side, regime)] += len(out)
        trades.extend(out)
    # No zero-row file: an empty Parquet is not a control tape, only flattering coverage (fix r1).
    if trades:
        tp = trades_path(root, peer, SELECTION_VARIANT)
        tp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([t.to_dict() for t in trades], columns=list(TRADE_COLUMNS)).to_parquet(tp, index=False)
    return {"symbol": peer, "status": "done", "replayed": len(trades),
            "dropped": len(plans) - len(trades), "groups": dict(replayed_groups)}


def _safe_replay_peer(peer: str, plans: list[ReplayPlan], root: Path, context: dict, notional: float,
                      cost_mult: float, override: Optional[float]) -> dict[str, Any]:
    """`replay_peer` with per-peer isolation: one bad symbol must not lose the whole control tape."""
    try:
        return replay_peer(peer, plans, root, context, notional, cost_mult, override)
    except Exception as exc:
        logger.exception("selection: replay failed for peer %s", peer)
        return {"symbol": peer, "status": "error", "replayed": 0, "dropped": len(plans), "groups": {},
                "error": f"{type(exc).__name__}: {exc}"}


def _init_worker(root: Path, notional: float, cost_mult: float, override: Optional[float]) -> None:
    _WORKER.update({"root": Path(root), "context": load_context(Path(root)), "notional": notional,
                    "cost_mult": cost_mult, "override": override})


def _peer_worker(job: tuple[str, list[ReplayPlan]]) -> dict[str, Any]:
    peer, plans = job
    return _safe_replay_peer(peer, plans, _WORKER["root"], _WORKER["context"], _WORKER["notional"],
                             _WORKER["cost_mult"], _WORKER["override"])


def _run_peers(root: Path, by_peer: dict[str, list[ReplayPlan]], workers: int, notional: float,
               cost_mult: float, override: Optional[float]) -> list[dict[str, Any]]:
    jobs = sorted(by_peer.items())
    if not jobs:
        return []
    if workers <= 1 or len(jobs) <= 1:
        context = load_context(root)
        return [_safe_replay_peer(peer, plans, root, context, notional, cost_mult, override)
                for peer, plans in jobs]
    with Pool(processes=workers, initializer=_init_worker,
              initargs=(root, notional, cost_mult, override)) as pool:
        records = list(pool.imap_unordered(_peer_worker, jobs, chunksize=1))
    return sorted(records, key=lambda r: r["symbol"])


def _clear_selection_tape(root: Path) -> None:
    """Remove the previous selection tape: the control is regenerated wholesale for one (tape, seed),
    and a leftover peer file from an earlier seed would be scored as part of this run's control --
    the "leftover baseline is worse than none" failure `regime_rows.baseline_guard` exists for."""
    for p in sorted(trades_dir(root, SELECTION_VARIANT).glob("*.parquet")):
        try:
            p.unlink()
        except OSError:
            logger.warning("selection: could not remove stale control file %s", p, exc_info=True)


def replay_selection(root: Path, seed: int, workers: int = 1, notional: float = 6000.0,
                     cost_mult: float = 1.0) -> dict[str, Any]:
    """Replay every real trade in `<root>/trades/` on a seeded random peer into `trades_selection/`.
    The summary is a pure function of `(root contents, seed)` -- no timings -- so the control's
    determinism can be asserted on it directly."""
    root = Path(root)
    trades = real_trades(root)
    universe = selection_universe(root)
    plans, dropped = build_plans(trades, root, universe, seed)
    by_peer: dict[str, list[ReplayPlan]] = {}
    for plan in plans:
        by_peer.setdefault(plan.peer, []).append(plan)
    _clear_selection_tape(root)
    records = _run_peers(root, by_peer, workers, notional, cost_mult, slippage_override_from(root))
    replayed = sum(int(r["replayed"]) for r in records)
    no_entry = sum(int(r["dropped"]) for r in records)
    counts = dict(Counter(dropped) + Counter({DROP_NO_ENTRY: no_entry} if no_entry else {}))
    pairing = pairing_counts(real_group_counts(trades), replayed_group_counts(records))
    summary = {"seed": int(seed), "symbols": len(universe),
               "peers": sum(1 for r in records if r["replayed"]), "peers_total": len(records),
               "trades": int(len(trades)), "replayed": replayed,
               "selection_dropped": int(len(trades)) - replayed, "dropped": counts,
               "selection_groups": pairing, "underreplayed_groups": underreplayed_groups(pairing),
               "errors": [r for r in records if r["status"] == "error"]}
    logger.info("selection: %d/%d trades replayed onto %d peers (dropped %s; under-replayed groups %s)",
                replayed, len(trades), summary["peers"], counts or "none",
                ", ".join(summary["underreplayed_groups"]) or "none")
    return summary
