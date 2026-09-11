"""Shadow SHORT trade resolver — replays hypothetical entries against bars.

v-shadow-short-resolver-2026-09-11. Loads the shadow_short_log.ndjson
ledger (written by strategies/builtin.py when ENABLE_MEAN_REV_SHORT_SHADOW
is True but ENABLE_MEAN_REV_SHORT is False), fetches subsequent minute
bars from Postgres, and resolves each entry to stop/target/timeout/flatten.

RESEARCH-LOCKED promotion floors (Stage A):
  n >= 150 trades OR >= 10 sessions
  PF >= 1.30
  WR >= 48%
  exp >= +0.05 R
  DD <= 6%
  max losing day <= 2 R
  max simultaneous <= 3 (target)

Usage:
    python -m research.shadow_short_resolver \\
        --log path/to/shadow_short_log.ndjson \\
        [--dsn postgresql://user:pass@host:port/db] \\
        --out /tmp/shadow_results/

Outputs:
    <out>/resolved_trades.ndjson    — one JSON row per resolved trade
    <out>/resolved_trades.csv       — same, tabular
    <out>/stage_a_summary.json      — aggregate metrics with promotion gates

Hands-off forever exclusions (hard-coded, do NOT trade these):
    MU, SNAP, SPCX, HQGE

Deduplication:
    Same symbol within 15 minutes → keep the first entry only.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone, date, time
from pathlib import Path
from typing import Any, Optional, Callable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("shadow_short_resolver")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
HANDS_OFF_SYMBOLS = frozenset({"MU", "SNAP", "SPCX", "HQGE"})
DEDUPE_WINDOW_MIN = 15
MAX_HOLD_BARS = 60
FLATTEN_TIME_ET = time(15, 55)  # 15:55 ET — flatten before close
DEFAULT_SLIPPAGE_PCT = 0.001    # 0.1% conservative slip model

# Stage A promotion floors — RESEARCH-LOCKED, do NOT loosen
STAGE_A_MIN_TRADES = 150
STAGE_A_MIN_SESSIONS = 10
STAGE_A_MIN_PF = 1.30
STAGE_A_MIN_WR = 0.48
STAGE_A_MIN_EXPECTANCY_R = 0.05
STAGE_A_MAX_DD_PCT = 0.06
STAGE_A_MAX_LOSING_DAY_R = 2.0
STAGE_A_MAX_SIMULTANEOUS = 3

DEFAULT_DSN = "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev"


@dataclass
class ShadowEntry:
    """Parsed row from shadow_short_log.ndjson."""
    timestamp: datetime
    symbol: str
    signal_type: str
    reason: str
    signal_close: float
    rsi: float
    bb_upper: float
    bb_middle: float
    sma_50: float
    macd: float
    macd_signal: float
    atr: float
    hypothetical_stop: float
    hypothetical_target: float
    rr_ratio: float
    stop_dist: float
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "ShadowEntry":
        ts = d.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                ts = datetime.now(timezone.utc)
        return cls(
            timestamp=ts,
            symbol=str(d.get("symbol", "")).upper(),
            signal_type=str(d.get("signal_type", "SHORT")).upper(),
            reason=str(d.get("reason", "")),
            signal_close=float(d.get("signal_close", 0)),
            rsi=float(d.get("rsi", 0)),
            bb_upper=float(d.get("bb_upper", 0)),
            bb_middle=float(d.get("bb_middle", 0)),
            sma_50=float(d.get("sma_50", 0)),
            macd=float(d.get("macd", 0)),
            macd_signal=float(d.get("macd_signal", 0)),
            atr=float(d.get("atr", 0)),
            hypothetical_stop=float(d.get("hypothetical_stop", 0)),
            hypothetical_target=float(d.get("hypothetical_target", 0)),
            rr_ratio=float(d.get("rr_ratio", 2.0)),
            stop_dist=float(d.get("stop_dist", 0)),
            raw=d,
        )


@dataclass
class ResolvedTrade:
    """Result of resolving a shadow entry against subsequent bars."""
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime]
    exit_price: Optional[float]
    exit_reason: str  # "stop" | "target" | "timeout" | "flatten" | "no_bars"
    stop: float
    target: float
    stop_dist: float
    hold_bars: int
    r_multiple: float  # (entry - exit) / stop_dist for SHORT
    return_pct: float
    regime: str        # from shadow entry context or "unknown"
    rising_peak_pass: bool
    raw_entry: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
def _default_log_paths() -> list[Path]:
    """Return candidate paths in priority order."""
    return [
        REPO_ROOT / "shadow_short_log.ndjson",
        REPO_ROOT / "data" / "shadow_short_log.ndjson",
    ]


def load_shadow_log(path: Optional[Path] = None) -> list[ShadowEntry]:
    """Load and parse the shadow_short_log.ndjson file."""
    if path is not None:
        candidates = [Path(path)]
    else:
        candidates = _default_log_paths()

    log_path: Optional[Path] = None
    for p in candidates:
        if p.exists():
            log_path = p
            break

    if log_path is None:
        logger.warning("No shadow_short_log.ndjson found at: %s", candidates)
        return []

    logger.info("Loading shadow log from: %s", log_path)
    entries: list[ShadowEntry] = []
    with open(log_path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                entries.append(ShadowEntry.from_dict(d))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logger.debug("Line %d parse error: %s", line_no, exc)
                continue

    logger.info("Loaded %d raw shadow entries", len(entries))
    return entries


def filter_and_dedupe(entries: list[ShadowEntry]) -> list[ShadowEntry]:
    """Apply hands-off exclusions and 15-min deduplication."""
    # 1. Exclude hands-off symbols
    filtered = [e for e in entries if e.symbol not in HANDS_OFF_SYMBOLS]
    excluded = len(entries) - len(filtered)
    if excluded > 0:
        logger.info("Excluded %d hands-off symbols (MU/SNAP/SPCX/HQGE)", excluded)

    # 2. Sort by timestamp
    filtered.sort(key=lambda e: e.timestamp)

    # 3. Dedupe: same symbol within 15 minutes → keep first
    last_seen: dict[str, datetime] = {}
    deduped: list[ShadowEntry] = []
    dupe_count = 0
    for e in filtered:
        prev = last_seen.get(e.symbol)
        if prev is not None:
            delta = (e.timestamp - prev).total_seconds() / 60.0
            if delta < DEDUPE_WINDOW_MIN:
                dupe_count += 1
                continue
        last_seen[e.symbol] = e.timestamp
        deduped.append(e)

    if dupe_count > 0:
        logger.info("Deduped %d entries (same symbol within 15m)", dupe_count)

    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# Bars Loading
# ─────────────────────────────────────────────────────────────────────────────
def make_postgres_bars_loader(dsn: str) -> Callable[[str, datetime, int], pd.DataFrame]:
    """Return a bars loader using PostgresDataProvider."""
    from data_providers.postgres import PostgresDataProvider
    provider = PostgresDataProvider(dsn)

    def _load(symbol: str, start: datetime, lookforward_days: int = 5) -> pd.DataFrame:
        """Load minute bars for symbol starting at `start` for `lookforward_days`."""
        end = start + timedelta(days=lookforward_days)
        try:
            df = provider.get_market_data(
                symbol,
                period_type="day",
                period=lookforward_days + 2,  # extra buffer
                frequency_type="minute",
                frequency=5,  # 5-min bars
                end=end,
            )
            if df.empty:
                return df
            # Filter to bars AFTER entry time
            df = df[df.index > start]
            return df
        except Exception as exc:
            logger.debug("Bars load failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    return _load


# ─────────────────────────────────────────────────────────────────────────────
# Resolution Logic
# ─────────────────────────────────────────────────────────────────────────────
def _is_flatten_time(ts: datetime) -> bool:
    """Check if timestamp is at or past flatten time (15:55 ET)."""
    try:
        import pytz
        et = pytz.timezone("America/New_York")
        ts_et = ts.astimezone(et) if ts.tzinfo else et.localize(ts)
        return ts_et.time() >= FLATTEN_TIME_ET
    except ImportError:
        # Fallback: assume UTC-4 (EDT) or UTC-5 (EST)
        hour_utc = ts.hour if ts.tzinfo is None else ts.utctimetuple().tm_hour
        # 15:55 ET is 19:55 or 20:55 UTC depending on DST
        return hour_utc >= 19 and ts.minute >= 55


def resolve_entry(
    entry: ShadowEntry,
    bars_loader: Callable[[str, datetime, int], pd.DataFrame],
) -> ResolvedTrade:
    """Resolve a single shadow entry against subsequent bars.
    
    Entry price = signal_close + conservative slip (0.1% adverse for SHORT).
    Barriers: hypothetical_stop (above), hypothetical_target (below).
    Time stop: 60 bars OR flatten at 15:55 ET — whichever comes first.
    R = (entry - exit) / stop_dist for SHORT (positive R = profit).
    """
    stop = entry.hypothetical_stop
    target = entry.hypothetical_target
    stop_dist = entry.stop_dist or abs(stop - entry.signal_close)

    # Conservative slip: for SHORT, slip UPWARD (worse entry)
    slip = entry.signal_close * DEFAULT_SLIPPAGE_PCT
    entry_price = entry.signal_close + slip

    # Regime from raw entry if available (market_context_regime field)
    regime = entry.raw.get("market_context_regime", "unknown")
    if regime is None:
        regime = "unknown"

    # v-shadow-rising-peak-2026-09-11: read rising_peak_pass from the shadow
    # entry. True = entry would pass the filter (close < SMA50 = downtrend).
    # Entries without this field (historical data) default to True.
    rising_peak_pass = entry.raw.get("rising_peak_pass", True)

    # Load bars
    df = bars_loader(entry.symbol, entry.timestamp, lookforward_days=5)
    if df.empty or len(df) == 0:
        return ResolvedTrade(
            symbol=entry.symbol,
            entry_time=entry.timestamp,
            entry_price=entry_price,
            exit_time=None,
            exit_price=None,
            exit_reason="no_bars",
            stop=stop,
            target=target,
            stop_dist=stop_dist,
            hold_bars=0,
            r_multiple=0.0,
            return_pct=0.0,
            regime=regime,
            rising_peak_pass=rising_peak_pass,
            raw_entry=entry.raw,
        )

    # Walk through bars
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason = "timeout"
    hold_bars = 0

    for i in range(min(len(df), MAX_HOLD_BARS)):
        bar = df.iloc[i]
        hold_bars = i + 1
        bar_ts = df.index[i]

        # Check flatten time first
        if _is_flatten_time(bar_ts):
            exit_time = bar_ts
            exit_price = float(bar["Close"])
            exit_reason = "flatten"
            break

        bar_high = float(bar["High"])
        bar_low = float(bar["Low"])

        # For SHORT: stop is ABOVE entry, target is BELOW entry
        stop_hit = bar_high >= stop
        target_hit = bar_low <= target

        if stop_hit and target_hit:
            # Both hit same bar — use close direction as tiebreaker
            exit_time = bar_ts
            if bar["Close"] < bar["Open"]:  # Down bar → target wins
                exit_price = target
                exit_reason = "target"
            else:  # Up bar → stop wins
                exit_price = stop
                exit_reason = "stop"
            break
        elif stop_hit:
            exit_time = bar_ts
            exit_price = stop
            exit_reason = "stop"
            break
        elif target_hit:
            exit_time = bar_ts
            exit_price = target
            exit_reason = "target"
            break

    # If we exhausted bars or hit MAX_HOLD_BARS without a barrier
    if exit_price is None:
        if len(df) > 0:
            last_bar = df.iloc[min(len(df) - 1, MAX_HOLD_BARS - 1)]
            exit_time = df.index[min(len(df) - 1, MAX_HOLD_BARS - 1)]
            exit_price = float(last_bar["Close"])
            hold_bars = min(len(df), MAX_HOLD_BARS)
        else:
            exit_price = entry_price  # No data fallback
            hold_bars = 0

    # Compute R-multiple for SHORT: profit when price goes DOWN
    # R = (entry - exit) / stop_dist
    if stop_dist > 0 and exit_price is not None:
        r_multiple = (entry_price - exit_price) / stop_dist
    else:
        r_multiple = 0.0

    # Return percentage for SHORT
    if entry_price > 0 and exit_price is not None:
        return_pct = (entry_price - exit_price) / entry_price * 100
    else:
        return_pct = 0.0

    return ResolvedTrade(
        symbol=entry.symbol,
        entry_time=entry.timestamp,
        entry_price=entry_price,
        exit_time=exit_time,
        exit_price=exit_price,
        exit_reason=exit_reason,
        stop=stop,
        target=target,
        stop_dist=stop_dist,
        hold_bars=hold_bars,
        r_multiple=r_multiple,
        return_pct=return_pct,
        regime=regime,
        rising_peak_pass=rising_peak_pass,
        raw_entry=entry.raw,
    )


def resolve_all(
    entries: list[ShadowEntry],
    bars_loader: Callable[[str, datetime, int], pd.DataFrame],
) -> list[ResolvedTrade]:
    """Resolve all shadow entries."""
    trades: list[ResolvedTrade] = []
    for i, entry in enumerate(entries):
        if (i + 1) % 10 == 0 or i == len(entries) - 1:
            logger.info("Resolving %d/%d: %s", i + 1, len(entries), entry.symbol)
        trade = resolve_entry(entry, bars_loader)
        trades.append(trade)
    return trades


# ─────────────────────────────────────────────────────────────────────────────
# Metrics / Stage A Summary
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StageASummary:
    """Stage A promotion metrics."""
    n_trades: int
    n_sessions: int
    n_wins: int
    n_losses: int
    n_scratches: int
    win_rate: float
    profit_factor: float
    total_r: float
    expectancy_r: float
    max_dd_r: float
    max_dd_pct: float
    max_losing_day_r: float
    max_simultaneous: int
    avg_hold_bars: float
    by_regime: dict[str, dict[str, Any]]
    by_exit_reason: dict[str, int]
    promotion_gates: dict[str, bool]
    all_gates_pass: bool


def compute_metrics(
    trades: list[ResolvedTrade],
    exclude_regimes: Optional[set[str]] = None,
) -> StageASummary:
    """Compute Stage A metrics from resolved trades."""
    if exclude_regimes is None:
        exclude_regimes = set()

    # Filter trades for promotion book (exclude risk_off by default)
    book_trades = [t for t in trades if t.regime not in exclude_regimes]

    # Count unique sessions (trading days)
    sessions: set[date] = set()
    for t in book_trades:
        if t.entry_time:
            d = t.entry_time.date() if hasattr(t.entry_time, "date") else date.today()
            sessions.add(d)

    # Win/loss classification
    wins = [t for t in book_trades if t.r_multiple > 0.05]
    losses = [t for t in book_trades if t.r_multiple < -0.05]
    scratches = [t for t in book_trades if -0.05 <= t.r_multiple <= 0.05]

    n_trades = len(book_trades)
    n_sessions = len(sessions)
    n_wins = len(wins)
    n_losses = len(losses)
    n_scratches = len(scratches)

    # Win rate (exclude scratches from denominator)
    win_loss_count = n_wins + n_losses
    win_rate = n_wins / win_loss_count if win_loss_count > 0 else 0.0

    # Profit factor
    gross_profit = sum(t.r_multiple for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.r_multiple for t in losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )

    # Total R and expectancy
    total_r = sum(t.r_multiple for t in book_trades)
    expectancy_r = total_r / n_trades if n_trades > 0 else 0.0

    # Max drawdown (simple cumulative R curve)
    peak_r = 0.0
    max_dd_r = 0.0
    cumulative_r = 0.0
    for t in sorted(book_trades, key=lambda x: x.entry_time or datetime.min):
        cumulative_r += t.r_multiple
        if cumulative_r > peak_r:
            peak_r = cumulative_r
        dd = peak_r - cumulative_r
        if dd > max_dd_r:
            max_dd_r = dd

    # Max DD as percentage of peak (rough proxy)
    max_dd_pct = max_dd_r / peak_r if peak_r > 0 else 0.0

    # Max losing day in R
    daily_r: dict[date, float] = defaultdict(float)
    for t in book_trades:
        if t.entry_time:
            d = t.entry_time.date() if hasattr(t.entry_time, "date") else date.today()
            daily_r[d] += t.r_multiple
    max_losing_day_r = abs(min(daily_r.values())) if daily_r else 0.0

    # Max simultaneous positions (approximate: count overlapping by day)
    daily_counts: dict[date, int] = defaultdict(int)
    for t in book_trades:
        if t.entry_time:
            d = t.entry_time.date() if hasattr(t.entry_time, "date") else date.today()
            daily_counts[d] += 1
    max_simultaneous = max(daily_counts.values()) if daily_counts else 0

    # Average hold bars
    avg_hold = sum(t.hold_bars for t in book_trades) / n_trades if n_trades > 0 else 0.0

    # By regime breakdown
    regime_groups: dict[str, list[ResolvedTrade]] = defaultdict(list)
    for t in trades:  # Use ALL trades for regime breakdown
        regime_groups[t.regime].append(t)

    by_regime: dict[str, dict[str, Any]] = {}
    for regime, group in regime_groups.items():
        group_wins = [t for t in group if t.r_multiple > 0.05]
        group_losses = [t for t in group if t.r_multiple < -0.05]
        group_r = sum(t.r_multiple for t in group)
        group_gross_profit = sum(t.r_multiple for t in group_wins)
        group_gross_loss = abs(sum(t.r_multiple for t in group_losses))
        by_regime[regime] = {
            "n": len(group),
            "wins": len(group_wins),
            "losses": len(group_losses),
            "total_r": round(group_r, 3),
            "pf": round(group_gross_profit / group_gross_loss, 2) if group_gross_loss > 0 else None,
            "wr": round(len(group_wins) / (len(group_wins) + len(group_losses)), 3) if (len(group_wins) + len(group_losses)) > 0 else 0.0,
        }

    # By exit reason
    by_exit: dict[str, int] = defaultdict(int)
    for t in book_trades:
        by_exit[t.exit_reason] += 1

    # Promotion gates
    # n>=150 OR >=10 sessions (either is acceptable)
    n_gate = n_trades >= STAGE_A_MIN_TRADES or n_sessions >= STAGE_A_MIN_SESSIONS
    pf_gate = profit_factor >= STAGE_A_MIN_PF
    wr_gate = win_rate >= STAGE_A_MIN_WR
    exp_gate = expectancy_r >= STAGE_A_MIN_EXPECTANCY_R
    dd_gate = max_dd_pct <= STAGE_A_MAX_DD_PCT
    losing_day_gate = max_losing_day_r <= STAGE_A_MAX_LOSING_DAY_R
    simultaneous_gate = max_simultaneous <= STAGE_A_MAX_SIMULTANEOUS

    gates = {
        f"n>={STAGE_A_MIN_TRADES}_or_sessions>={STAGE_A_MIN_SESSIONS}": n_gate,
        f"pf>={STAGE_A_MIN_PF}": pf_gate,
        f"wr>={STAGE_A_MIN_WR}": wr_gate,
        f"exp>={STAGE_A_MIN_EXPECTANCY_R}R": exp_gate,
        f"dd<={STAGE_A_MAX_DD_PCT*100}%": dd_gate,
        f"max_losing_day<={STAGE_A_MAX_LOSING_DAY_R}R": losing_day_gate,
        f"max_simultaneous<={STAGE_A_MAX_SIMULTANEOUS}": simultaneous_gate,
    }

    return StageASummary(
        n_trades=n_trades,
        n_sessions=n_sessions,
        n_wins=n_wins,
        n_losses=n_losses,
        n_scratches=n_scratches,
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 3) if profit_factor != float("inf") else None,
        total_r=round(total_r, 3),
        expectancy_r=round(expectancy_r, 4),
        max_dd_r=round(max_dd_r, 3),
        max_dd_pct=round(max_dd_pct, 4),
        max_losing_day_r=round(max_losing_day_r, 3),
        max_simultaneous=max_simultaneous,
        avg_hold_bars=round(avg_hold, 1),
        by_regime=by_regime,
        by_exit_reason=dict(by_exit),
        promotion_gates=gates,
        all_gates_pass=all(gates.values()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output Writers
# ─────────────────────────────────────────────────────────────────────────────
def write_trades_ndjson(trades: list[ResolvedTrade], path: Path) -> None:
    """Write resolved trades as NDJSON."""
    with open(path, "w") as f:
        for t in trades:
            row = {
                "symbol": t.symbol,
                "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                "entry_price": round(t.entry_price, 4),
                "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                "exit_price": round(t.exit_price, 4) if t.exit_price else None,
                "exit_reason": t.exit_reason,
                "stop": round(t.stop, 4),
                "target": round(t.target, 4),
                "stop_dist": round(t.stop_dist, 4),
                "hold_bars": t.hold_bars,
                "r_multiple": round(t.r_multiple, 4),
                "return_pct": round(t.return_pct, 4),
                "regime": t.regime,
                "rising_peak_pass": t.rising_peak_pass,
            }
            f.write(json.dumps(row, default=str) + "\n")
    logger.info("Wrote %d trades to %s", len(trades), path)


def write_trades_csv(trades: list[ResolvedTrade], path: Path) -> None:
    """Write resolved trades as CSV."""
    rows = []
    for t in trades:
        rows.append({
            "symbol": t.symbol,
            "entry_time": t.entry_time.isoformat() if t.entry_time else "",
            "entry_price": round(t.entry_price, 4),
            "exit_time": t.exit_time.isoformat() if t.exit_time else "",
            "exit_price": round(t.exit_price, 4) if t.exit_price else "",
            "exit_reason": t.exit_reason,
            "stop": round(t.stop, 4),
            "target": round(t.target, 4),
            "stop_dist": round(t.stop_dist, 4),
            "hold_bars": t.hold_bars,
            "r_multiple": round(t.r_multiple, 4),
            "return_pct": round(t.return_pct, 4),
            "regime": t.regime,
            "rising_peak_pass": t.rising_peak_pass,
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("Wrote %d trades to %s", len(trades), path)


def write_summary_json(
    summary: StageASummary,
    all_regime_summary: StageASummary,
    path: Path,
) -> None:
    """Write Stage A summary as JSON."""
    output = {
        "promotion_book": {
            "description": "risk_off excluded",
            **asdict(summary),
        },
        "all_regime": {
            "description": "all regimes included (secondary)",
            **asdict(all_regime_summary),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Wrote summary to %s", path)


def print_summary(summary: StageASummary, label: str = "Promotion Book") -> None:
    """Print human-readable summary."""
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"STAGE A SUMMARY — {label}")
    print(bar)
    print(f"  Trades:          {summary.n_trades}")
    print(f"  Sessions:        {summary.n_sessions}")
    print(f"  Wins/Losses:     {summary.n_wins}/{summary.n_losses} "
          f"(scratches: {summary.n_scratches})")
    print(f"  Win Rate:        {summary.win_rate*100:.1f}%")
    print(f"  Profit Factor:   {summary.profit_factor}")
    print(f"  Total R:         {summary.total_r:+.2f}")
    print(f"  Expectancy:      {summary.expectancy_r:+.4f} R/trade")
    print(f"  Max DD:          {summary.max_dd_r:.2f} R ({summary.max_dd_pct*100:.1f}%)")
    print(f"  Max Losing Day:  {summary.max_losing_day_r:.2f} R")
    print(f"  Max Simultaneous:{summary.max_simultaneous}")
    print(f"  Avg Hold Bars:   {summary.avg_hold_bars:.1f}")
    print(f"\n  By Exit Reason:  {dict(summary.by_exit_reason)}")
    print(f"\n  By Regime:")
    for regime, stats in summary.by_regime.items():
        print(f"    {regime}: n={stats['n']}, wins={stats['wins']}, "
              f"R={stats['total_r']:+.2f}, PF={stats['pf']}, WR={stats['wr']:.1%}")

    print(f"\n{bar}")
    print("PROMOTION GATES:")
    for gate, passed in summary.promotion_gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {gate}")
    if summary.all_gates_pass:
        print(f"\n  → ALL GATES PASS. Shadow data supports enabling MEAN_REV_SHORT.")
    else:
        print(f"\n  → GATE(S) FAILED. Do NOT enable live SHORT yet.")
    print(bar)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log", "-l",
        type=Path,
        default=None,
        help="Path to shadow_short_log.ndjson. Default: repo-root then data/",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help=f"Postgres DSN. Default: POSTGRES_DSN env or {DEFAULT_DSN}",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and filter entries without resolving (for quick checks)",
    )
    args = parser.parse_args(argv)

    # Load entries
    entries = load_shadow_log(args.log)
    if not entries:
        logger.error("No shadow entries to process")
        return 1

    # Filter and dedupe
    entries = filter_and_dedupe(entries)
    logger.info("After filtering: %d entries", len(entries))

    if args.dry_run:
        print(f"\nDry run: {len(entries)} entries would be resolved.")
        for e in entries[:10]:
            print(f"  {e.timestamp} {e.symbol} @ {e.signal_close:.2f}")
        if len(entries) > 10:
            print(f"  ... and {len(entries) - 10} more")
        return 0

    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)

    # Create bars loader
    dsn = args.dsn or os.environ.get("POSTGRES_DSN", DEFAULT_DSN)
    bars_loader = make_postgres_bars_loader(dsn)

    # Resolve all entries
    logger.info("Resolving %d entries...", len(entries))
    trades = resolve_all(entries, bars_loader)

    # Filter out no_bars trades for metrics
    valid_trades = [t for t in trades if t.exit_reason != "no_bars"]
    no_bars_count = len(trades) - len(valid_trades)
    if no_bars_count > 0:
        logger.warning("%d trades had no bars data (excluded from metrics)", no_bars_count)

    # Compute metrics
    # Promotion book: exclude risk_off
    promo_summary = compute_metrics(valid_trades, exclude_regimes={"risk_off"})
    # All-regime secondary
    all_summary = compute_metrics(valid_trades, exclude_regimes=set())

    # Write outputs
    write_trades_ndjson(trades, args.out / "resolved_trades.ndjson")
    write_trades_csv(trades, args.out / "resolved_trades.csv")
    write_summary_json(promo_summary, all_summary, args.out / "stage_a_summary.json")

    # Print summaries
    print_summary(promo_summary, "Promotion Book (risk_off excluded)")
    print_summary(all_summary, "All Regimes (secondary)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
