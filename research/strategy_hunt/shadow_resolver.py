# research/strategy_hunt/shadow_resolver.py
"""Resolves `strategies/fit_lane.py` shadow ledgers (data/shadow_fit_<family>.ndjson)
into simulated trades and a Stage A scorecard.

Ledger lines are written by `FitLaneStrategy._generate_signal_with_commentary`
(see strategies/fit_lane.py) with keys: ts, symbol, side, entry_px, stop_px,
target_px, time_stop, hold_kind, tag, params, verdict, score. `ts` is the live
clock timestamp at write time and is generally NOT on the historical 5-min
bar grid used here for research-side resolution, so entries are snapped
forward to the next available bar (see `_snap_entry`).

May import `backtest` and `strategies.families`; must never import
`strategies.fit_lane` or anything outside `strategies.families` (core/research
separation).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from backtest.stage_a_scorer import StageAMetrics, check_floors
from research.strategy_hunt.data.earnings import earnings_days
from research.strategy_hunt.data.splits import load_adjusted
from research.strategy_hunt.models import SimTrade
from research.strategy_hunt.run import REPORTS_DIR
from research.strategy_hunt.sim.costs import CostModel
from research.strategy_hunt.sim.simulator import build_session_table, simulate
from strategies.families.models import SIDES, Entry

logger = logging.getLogger("strategy_hunt.shadow_resolver")

# Mirrors strategies/fit_lane.py's `ledger_path` convention (data/shadow_fit_<family>.ndjson)
# without importing that module: research may depend on strategies.families only.
DEFAULT_LEDGER_DIR = Path("data")


def default_ledger_path(family: str, ledger_dir: Path = DEFAULT_LEDGER_DIR) -> Path:
    return Path(ledger_dir) / f"shadow_fit_{family}.ndjson"


def read_ledger(path: Path) -> list[dict]:
    """Parse NDJSON records from `path`, skipping (and counting) malformed lines."""
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict] = []
    skipped = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    if skipped:
        logger.warning("read_ledger: skipped %d malformed line(s) in %s", skipped, path)
    return records


def ledger_entries(records: list[dict]) -> list[Entry]:
    """Raw (unsnapped) Entry objects, one per ledger record."""
    return [
        Entry(
            symbol=r["symbol"],
            side=r["side"],
            entry_ts=pd.Timestamp(r["ts"]),
            entry_px=float(r["entry_px"]),
            stop_px=float(r["stop_px"]),
            target_px=float(r["target_px"]),
            time_stop=int(r["time_stop"]),
            hold_kind=r["hold_kind"],
            tag=r.get("tag", ""),
        )
        for r in records
    ]


def _snap_entry(entry: Entry, bars5: pd.DataFrame) -> Optional[Entry]:
    """Snap `entry.entry_ts` forward to the next 5-min bar and re-price at
    that bar's open, shifting stop/target by the same delta so R stays
    meaningful. Returns None (dropped) if no bar exists at or after `ts`."""
    idx = bars5.index
    if entry.entry_ts in idx:
        return entry
    pos = int(idx.searchsorted(entry.entry_ts))
    if pos >= len(idx):
        return None
    new_ts = idx[pos]
    new_entry_px = float(bars5["open"].iloc[pos])
    delta = new_entry_px - entry.entry_px
    return Entry(
        symbol=entry.symbol,
        side=entry.side,
        entry_ts=new_ts,
        entry_px=new_entry_px,
        stop_px=entry.stop_px + delta,
        target_px=entry.target_px + delta,
        time_stop=entry.time_stop,
        hold_kind=entry.hold_kind,
        tag=entry.tag,
    )


def resolve(records: list[dict], root: Path, family: str, cost_mult: float = 1.0) -> list[SimTrade]:
    """Group ledger records by symbol, snap entries onto the 5-min grid, and
    simulate each against adjusted bars. Trades keep window_id=-1 and
    params_json from their own ledger record."""
    root = Path(root)
    by_symbol: dict[str, list[dict]] = {}
    for r in records:
        by_symbol.setdefault(r["symbol"], []).append(r)

    trades: list[SimTrade] = []
    dropped = 0
    for symbol, recs in by_symbol.items():
        bars5, daily = load_adjusted(root, symbol)
        if bars5.empty:
            dropped += len(recs)
            continue
        cost_model = CostModel.from_bars(bars5).scaled(cost_mult)
        earnings = earnings_days(root, symbol, daily)
        table = build_session_table(bars5)
        for r, raw_entry in zip(recs, ledger_entries(recs)):
            entry = _snap_entry(raw_entry, bars5)
            if entry is None:
                dropped += 1
                continue
            params_json = json.dumps(r.get("params", {}))
            trades.extend(simulate([entry], bars5, cost_model, earnings, family,
                                   window_id=-1, params_json=params_json, table=table))
    if dropped:
        logger.warning("resolve: dropped %d ledger entr(y/ies) with no bar at/after entry_ts", dropped)
    return trades


def _side_metrics(trades: list[SimTrade], family: str, side: str) -> StageAMetrics:
    side_trades = [t for t in trades if t.side == side]
    n = len(side_trades)
    if n == 0:
        m = StageAMetrics(lane=family, direction=side, n=0, sessions=0, pf=0.0, wr=0.0, exp_r=0.0,
                          max_dd_pct=0.0, max_losing_day_r=0.0, verdict="FAIL", recommendation="LIVE off",
                          notes="no trades", risk_off_excluded=False)
        verdict, failures = check_floors(m)
        m.verdict, m.floor_failures = verdict, failures
        return m

    df = pd.DataFrame({
        "r": [float(t.r_multiple) for t in side_trades],
        "entry_date": [pd.Timestamp(t.entry_ts).normalize() for t in side_trades],
        "exit_ts": [pd.Timestamp(t.exit_ts) for t in side_trades],
    }).sort_values("exit_ts")

    sessions = int(df["entry_date"].nunique())
    non_scratch = df[df["r"].abs() >= 0.05]
    wins = non_scratch[non_scratch["r"] > 0]
    losses = non_scratch[non_scratch["r"] <= 0]
    wr = float(len(wins) / len(non_scratch)) if len(non_scratch) else 0.0
    gross_wins = float(wins["r"].sum()) if len(wins) else 0.0
    gross_losses = float(abs(losses["r"].sum())) if len(losses) else 0.0
    pf = (gross_wins / gross_losses) if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)
    exp_r = float(df["r"].mean())

    # Max drawdown of cumulative R, expressed as a fraction of a nominal 100R
    # book (not the trading account's actual capital) -- documented convention
    # per Task 32 brief; simpler than backtest.stage_a_scorer's legacy *6.0 scale.
    cum_r = df["r"].cumsum()
    drawdown = cum_r - cum_r.cummax()
    max_dd_r = float(abs(drawdown.min())) if len(drawdown) else 0.0
    max_dd_pct = max_dd_r / 100.0

    daily_r = df.groupby("entry_date")["r"].sum()
    max_losing_day_r = float(abs(daily_r.min())) if len(daily_r) and daily_r.min() < 0 else 0.0

    m = StageAMetrics(lane=family, direction=side, n=n, sessions=sessions, pf=round(pf, 3), wr=round(wr, 4),
                      exp_r=round(exp_r, 4), max_dd_pct=round(max_dd_pct, 4),
                      max_losing_day_r=round(max_losing_day_r, 2), verdict="FAIL", recommendation="LIVE off",
                      risk_off_excluded=False, sample_start=str(df["entry_date"].min().date()),
                      sample_end=str(df["entry_date"].max().date()))
    verdict, failures = check_floors(m)
    m.verdict, m.floor_failures = verdict, failures
    m.recommendation = "LIVE on" if verdict == "PASS" else "LIVE off"
    return m


def stage_a_scorecard(trades: list[SimTrade], family: str) -> dict:
    """Per-side Stage A metrics plus an overall GO/NO_GO verdict. GO requires
    at least one side with n>0 and every side with n>0 to PASS its floors."""
    card: dict = {"family": family}
    passing = []
    for side in SIDES:
        m = _side_metrics(trades, family, side)
        card[side] = m.to_dict()
        if m.n > 0:
            passing.append(m.verdict == "PASS")
    card["verdict"] = "GO" if passing and all(passing) else "NO_GO"
    return card


def write_scorecard(root: Path, family: str, card: dict, computed_at: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    p = REPORTS_DIR / f"scorecard-{family}-{computed_at[:10]}.md"
    lines = [
        f"# Stage A scorecard: {family}",
        "",
        f"Computed at: {computed_at}",
        f"Verdict: **{card['verdict']}**",
        "",
        "| side | n | sessions | pf | wr | exp_r | max_dd_pct | max_losing_day_r | verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for side in SIDES:
        m = card.get(side)
        if not m:
            continue
        lines.append(f"| {side} | {m['n']} | {m['sessions']} | {m['pf']:.3f} | {m['wr']:.3f} | "
                     f"{m['exp_r']:+.4f} | {m['max_dd_pct']:.4f} | {m['max_losing_day_r']:.2f} | {m['verdict']} |")
    p.write_text("\n".join(lines) + "\n")
    return p
