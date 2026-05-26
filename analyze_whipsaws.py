"""v-whipsaw-analysis-2026-04-30: Phase 1 diagnostic — measure how many of
today's closed losers have since recovered to profit.

Read-only. Takes today's trades from trading_state.json, fetches the
current price + intraday bars from the running bot's /api/historical-data
endpoint, then for each trade computes:

  - Was the original take-profit hit since the exit?
  - Was the original stop-loss hit since the exit?
  - What's the unrealized P&L NOW if the trade had been held?
  - Whipsaw classification: did a -loss exit recover to >= breakeven?

Then aggregates the data three ways the user asked for:
  1. By strategy
  2. By exit reason
  3. By holding-time bucket at exit

The output tells us which exit rules are killing future winners and where
the targeted fix should land. No code changes — pure measurement.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

API_BASE = "http://localhost:9000"
API_KEY = os.environ.get(
    "TRADING_API_KEY",
    "X-682hk4CHr7niQqHge0L5ZWQl5PMf5AzLNGihvxrfg",
)
TODAY = datetime.now().strftime("%Y-%m-%d")


def fetch_bars(symbol: str, frequency: int = 5, period: int = 1) -> list[dict]:
    """Pull intraday bars from the bot's historical-data endpoint."""
    url = (
        f"{API_BASE}/api/historical-data/{urllib.parse.quote(symbol)}"
        f"?frequency={frequency}&period={period}"
    )
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())
        return data.get("bars", []) or []
    except Exception as exc:
        print(f"  ! fetch error for {symbol}: {exc}", file=sys.stderr)
        return []


@dataclass
class TradeReplay:
    symbol: str
    side: str
    strategy: str
    exit_reason: str
    entry_price: float
    exit_price: float
    quantity: float
    realized_pnl: float
    hold_minutes: float
    entry_time: str
    exit_time: str
    # Computed:
    current_price: Optional[float] = None
    target_hit: bool = False         # would-be take-profit
    stop_hit: bool = False           # would-be stop-loss
    pnl_if_held: Optional[float] = None
    whipsaw: bool = False            # closed at loss, since recovered to >= 0
    recovery_pct: Optional[float] = None  # how far past breakeven we'd be
    notes: list[str] = field(default_factory=list)


def replay(trade: dict, bars_after_exit: list[dict]) -> TradeReplay:
    """Walk forward from the exit and find what would have happened."""
    sym = trade.get("symbol", "?")
    side = trade.get("side") or "long"
    strat = trade.get("strategy") or (trade.get("reasoning") or {}).get("strategy") or "?"
    reason = trade.get("exit_reason") or trade.get("reason") or "?"
    entry = float(trade.get("entry_price") or 0)
    exit_p = float(trade.get("exit_price") or 0)
    qty = float(trade.get("quantity") or 0)
    pnl = float(trade.get("pnl") or 0)
    held = float(trade.get("hold_minutes") or trade.get("holding_period") or 0)
    entry_ts = trade.get("entry_time") or ""
    exit_ts = trade.get("exit_time") or trade.get("timestamp") or ""

    rep = TradeReplay(
        symbol=sym, side=side, strategy=strat, exit_reason=reason,
        entry_price=entry, exit_price=exit_p, quantity=qty,
        realized_pnl=pnl, hold_minutes=held,
        entry_time=entry_ts, exit_time=exit_ts,
    )

    # Reconstruct the original stop / target the trade was sized with.
    # Most strategies use ATR_STOP_MULTIPLIER=1.5 with R:R=2.0, so
    # stop_dist = abs(entry - exit_when_stopped). For a -1R stop-out
    # trade we already know stop_dist; for proactive-exit trades it's
    # the entry's original_stop. trading_state.json doesn't store these
    # cleanly, so we approximate using R-multiple of the realized loss.
    # If we know entry+exit, the realized R is implicit: bigger losses
    # mean bigger stops. We'll just walk forward and look for any 2:1
    # favorable move relative to the entry.
    if not bars_after_exit:
        rep.notes.append("no_bars")
        return rep

    rep.current_price = float(bars_after_exit[-1].get("close") or 0)

    # Approximate the trade's risk-per-share using the loss/qty (only
    # works when realized was negative; for winners we use 1% of entry).
    if pnl < 0 and qty > 0:
        risk_per_share = abs(pnl) / qty
    else:
        risk_per_share = entry * 0.01  # 1% fallback

    if side == "short":
        target_price = entry - 2.0 * risk_per_share
        stop_price   = entry + 1.0 * risk_per_share
        # P&L if held
        rep.pnl_if_held = (entry - rep.current_price) * qty
    else:
        target_price = entry + 2.0 * risk_per_share
        stop_price   = entry - 1.0 * risk_per_share
        rep.pnl_if_held = (rep.current_price - entry) * qty

    # Walk forward from the exit time looking for either bound.
    exit_dt = exit_ts
    saw_target = saw_stop = False
    for b in bars_after_exit:
        b_time = b.get("time") or ""
        if b_time < exit_dt:
            continue
        hi = float(b.get("high") or 0)
        lo = float(b.get("low") or 0)
        if side == "short":
            if lo <= target_price:
                saw_target = True
                break  # target hit first → whipsaw
            if hi >= stop_price:
                saw_stop = True
                break  # original stop also broken
        else:
            if hi >= target_price:
                saw_target = True
                break
            if lo <= stop_price:
                saw_stop = True
                break

    rep.target_hit = saw_target
    rep.stop_hit = saw_stop and not saw_target
    if pnl < 0 and rep.pnl_if_held is not None and rep.pnl_if_held >= 0:
        rep.whipsaw = True
        if entry > 0:
            rep.recovery_pct = (rep.pnl_if_held - pnl) / abs(pnl) * 100 if pnl else 0
    return rep


def bucket_hold(minutes: float) -> str:
    if minutes < 5:    return "<5m"
    if minutes < 15:   return "5-15m"
    if minutes < 30:   return "15-30m"
    if minutes < 60:   return "30-60m"
    if minutes < 120:  return "1-2h"
    return "2h+"


def main():
    state_path = "/home/nvidia/claude/trading_bot_commentary/trading_state.json"
    with open(state_path) as f:
        state = json.load(f)
    history = state.get("trade_history", [])

    today_trades = [
        t for t in history
        if str(t.get("exit_time") or t.get("timestamp") or "").startswith(TODAY)
    ]
    if not today_trades:
        print(f"No trades closed today ({TODAY}).")
        return

    print(f"=== Whipsaw analysis for {TODAY}: {len(today_trades)} closed trades ===\n")

    # Cache bars per symbol so we only fetch each once
    bars_cache: dict[str, list[dict]] = {}
    replays: list[TradeReplay] = []
    for t in today_trades:
        sym = t.get("symbol", "?")
        if sym not in bars_cache:
            bars_cache[sym] = fetch_bars(sym, frequency=5, period=1)
        rep = replay(t, bars_cache[sym])
        replays.append(rep)

    # =================== TABLE: trade-by-trade ===================
    print(f"{'Sym':<5} {'Side':<5} {'Strat':<14} {'Exit Reason':<28} "
          f"{'Held':>6} {'PnL':>9} {'PnL@now':>9} {'Whip?':>6}")
    print("-" * 95)
    for r in sorted(replays, key=lambda x: (x.whipsaw, x.realized_pnl)):
        whip = "YES" if r.whipsaw else "no"
        held = f"{r.hold_minutes:.0f}m" if r.hold_minutes else "?"
        pnl_held = f"${r.pnl_if_held:+.0f}" if r.pnl_if_held is not None else "n/a"
        print(f"{r.symbol:<5} {r.side:<5} {r.strategy[:14]:<14} "
              f"{r.exit_reason[:28]:<28} {held:>6} "
              f"${r.realized_pnl:+8.0f} {pnl_held:>9} {whip:>6}")

    # =================== AGGREGATES ===================
    print()
    print(f"=== Headline ===")
    realized = sum(r.realized_pnl for r in replays)
    if_held  = sum((r.pnl_if_held or 0) for r in replays)
    whipsaw_count = sum(1 for r in replays if r.whipsaw)
    whipsaw_loss  = sum(abs(r.realized_pnl) for r in replays if r.whipsaw)
    whipsaw_recov = sum((r.pnl_if_held or 0) for r in replays if r.whipsaw)

    print(f"  Realized today (actual):   ${realized:+,.0f}")
    print(f"  P&L if all held to now:    ${if_held:+,.0f}")
    print(f"  Whipsaw lift if held:      ${if_held - realized:+,.0f}")
    print(f"  Whipsaw count:             {whipsaw_count}/{len(replays)} "
          f"({whipsaw_count/len(replays)*100:.0f}%)")
    print(f"  $ left on table by whipsaws: ${whipsaw_recov - (-whipsaw_loss):+,.0f}")

    # By exit reason
    print(f"\n=== By exit reason ===")
    print(f"{'Reason':<35} {'N':>3} {'Whip':>5} {'Realized':>11} {'IfHeld':>11} {'Lift':>11}")
    by_reason: dict[str, list[TradeReplay]] = defaultdict(list)
    for r in replays:
        by_reason[r.exit_reason].append(r)
    for reason, items in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        n = len(items)
        whip = sum(1 for x in items if x.whipsaw)
        rsum = sum(x.realized_pnl for x in items)
        hsum = sum((x.pnl_if_held or 0) for x in items)
        print(f"{reason[:35]:<35} {n:>3} {whip:>5} ${rsum:>+9.0f} ${hsum:>+9.0f} ${hsum-rsum:>+9.0f}")

    # By strategy
    print(f"\n=== By strategy ===")
    print(f"{'Strategy':<25} {'N':>3} {'Whip':>5} {'Realized':>11} {'IfHeld':>11} {'Lift':>11}")
    by_strategy: dict[str, list[TradeReplay]] = defaultdict(list)
    for r in replays:
        by_strategy[r.strategy].append(r)
    for s, items in sorted(by_strategy.items(), key=lambda x: -len(x[1])):
        n = len(items)
        whip = sum(1 for x in items if x.whipsaw)
        rsum = sum(x.realized_pnl for x in items)
        hsum = sum((x.pnl_if_held or 0) for x in items)
        print(f"{s[:25]:<25} {n:>3} {whip:>5} ${rsum:>+9.0f} ${hsum:>+9.0f} ${hsum-rsum:>+9.0f}")

    # By holding-time bucket at exit
    print(f"\n=== By holding-time bucket at exit ===")
    print(f"{'Bucket':<8} {'N':>3} {'Whip':>5} {'Realized':>11} {'IfHeld':>11} {'Lift':>11}")
    by_hold: dict[str, list[TradeReplay]] = defaultdict(list)
    for r in replays:
        by_hold[bucket_hold(r.hold_minutes)].append(r)
    bucket_order = ["<5m", "5-15m", "15-30m", "30-60m", "1-2h", "2h+"]
    for b in bucket_order:
        items = by_hold.get(b, [])
        if not items:
            continue
        n = len(items)
        whip = sum(1 for x in items if x.whipsaw)
        rsum = sum(x.realized_pnl for x in items)
        hsum = sum((x.pnl_if_held or 0) for x in items)
        print(f"{b:<8} {n:>3} {whip:>5} ${rsum:>+9.0f} ${hsum:>+9.0f} ${hsum-rsum:>+9.0f}")

    # By exit reason × side
    print(f"\n=== Whipsaw rate by (exit_reason, side) ===")
    cross: dict[tuple[str, str], list[TradeReplay]] = defaultdict(list)
    for r in replays:
        cross[(r.exit_reason[:25], r.side)].append(r)
    print(f"{'Exit Reason':<25} {'Side':<5} {'N':>3} {'Whip%':>6} {'Avg lift/whip':>15}")
    for (reason, side), items in sorted(cross.items(), key=lambda x: -len(x[1])):
        n = len(items)
        whips = [x for x in items if x.whipsaw]
        whip_pct = len(whips) / n * 100 if n else 0
        avg_lift = sum((x.pnl_if_held or 0) - x.realized_pnl for x in whips) / len(whips) if whips else 0
        print(f"{reason:<25} {side:<5} {n:>3} {whip_pct:>5.0f}% ${avg_lift:>+12.0f}")


if __name__ == "__main__":
    main()
