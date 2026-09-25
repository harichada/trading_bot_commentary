"""Capacity-constrained replay of PASS-lane trades with the bot's sizing rules -> dollar equity curve."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SizingRules:
    equity0: float = 43500.0
    risk_pct: float = 0.015
    kelly: float = 0.5
    day_trade_mult: float = 0.5
    max_notional_abs: float = 10000.0
    max_notional_bp_pct: float = 0.20
    max_positions: int = 5
    bp_ratio: float = 0.69


def shares_for(rules: SizingRules, equity: float, entry_px: float, stop_px: float, is_day_trade: bool) -> int:
    """Risk-based share count, capped by notional limits; 0 if below one share."""
    risk_per_share = abs(entry_px - stop_px)
    if risk_per_share <= 0 or entry_px <= 0:
        return 0
    risk = equity * rules.risk_pct * rules.kelly * (rules.day_trade_mult if is_day_trade else 1.0)
    shares = math.floor(risk / risk_per_share)
    cap = min(rules.max_notional_abs, equity * rules.bp_ratio * rules.max_notional_bp_pct)
    shares = min(shares, math.floor(cap / entry_px))
    return int(shares) if shares >= 1 else 0


def pass_lane_trades(trades: pd.DataFrame, fit_rows: list[dict]) -> pd.DataFrame:
    """Filter trades to (symbol, family, side) combos with a walkforward PASS verdict."""
    keys = {(r["symbol"], r["strategy"], r["side"]) for r in fit_rows
            if r.get("verdict") == "PASS" and r.get("source") == "walkforward"}
    if trades.empty:
        return trades
    mask = [(s, f, sd) in keys for s, f, sd in zip(trades["symbol"], trades["family"], trades["side"])]
    return trades[mask].reset_index(drop=True)


def _hold_kind(row) -> str:
    same_day = pd.Timestamp(row["entry_ts"]).date() == pd.Timestamp(row["exit_ts"]).date()
    return "intraday" if row["hold_sessions"] == 1 and same_day else "multiday"


def _empty_result(rules: SizingRules) -> dict:
    return {"taken": 0, "skipped_capacity": 0, "skipped_symbol": 0, "skipped_size": 0, "final_equity": rules.equity0,
            "total_return_pct": 0.0, "max_drawdown_pct": 0.0, "daily_sharpe": 0.0,
            "equity_curve": pd.Series(dtype=float), "by_family": {}}


def _run_acceptance(df: pd.DataFrame, rules: SizingRules) -> tuple[dict, dict, list[tuple[pd.Timestamp, float]]]:
    """Single pass over sorted trades: applies capacity/symbol/size rules, tracks running
    equity for sizing, and collects every accepted (exit_ts, pnl) for the equity curve."""
    equity = rules.equity0
    open_pos: list[tuple[pd.Timestamp, str]] = []      # (exit_ts, symbol) currently open
    pending: list[tuple[pd.Timestamp, float]] = []      # (exit_ts, pnl) not yet applied to equity
    exits: list[tuple[pd.Timestamp, float]] = []        # every accepted (exit_ts, pnl)
    counters = {"taken": 0, "skipped_capacity": 0, "skipped_symbol": 0, "skipped_size": 0}
    by_family: dict[str, dict] = {}
    for _, r in df.iterrows():
        ts = pd.Timestamp(r["entry_ts"])
        while pending and pending[0][0] <= ts:
            equity += pending.pop(0)[1]
        open_pos = [p for p in open_pos if p[0] > ts]
        if any(p[1] == r["symbol"] for p in open_pos):
            counters["skipped_symbol"] += 1
            continue
        if len(open_pos) >= rules.max_positions:
            counters["skipped_capacity"] += 1
            continue
        is_day_trade = _hold_kind(r) == "intraday"
        sh = shares_for(rules, equity, float(r["entry_fill"]), float(r["stop_px"]), is_day_trade)
        if sh == 0:
            counters["skipped_size"] += 1
            continue
        sign = 1.0 if r["side"] == "long" else -1.0
        pnl = sh * (float(r["exit_fill"]) - float(r["entry_fill"])) * sign
        exit_ts = pd.Timestamp(r["exit_ts"])
        open_pos.append((exit_ts, r["symbol"]))
        pending.append((exit_ts, pnl))
        pending.sort(key=lambda x: x[0])
        exits.append((exit_ts, pnl))
        counters["taken"] += 1
        fam = by_family.setdefault(r["family"], {"taken": 0, "pnl": 0.0})
        fam["taken"] += 1
        fam["pnl"] += pnl
    return counters, by_family, exits


def _build_curve(df: pd.DataFrame, exits: list[tuple[pd.Timestamp, float]], equity0: float) -> pd.Series:
    dates = sorted(set(pd.Timestamp(t).normalize() for t in df["entry_ts"]) |
                    {pd.Timestamp(e).normalize() for e, _ in exits})
    daily_pnl: dict[pd.Timestamp, float] = {}
    for exit_ts, pnl in exits:
        d = pd.Timestamp(exit_ts).normalize()
        daily_pnl[d] = daily_pnl.get(d, 0.0) + pnl
    curve = pd.Series(index=dates, dtype=float)
    running = equity0
    for d in dates:
        running += daily_pnl.get(d, 0.0)
        curve[d] = running
    return curve


def replay(trades: pd.DataFrame, rules: SizingRules = SizingRules()) -> dict:
    if trades.empty:
        return _empty_result(rules)
    df = trades.sort_values(["entry_ts", "symbol"]).reset_index(drop=True)
    counters, by_family, exits = _run_acceptance(df, rules)
    curve = _build_curve(df, exits, rules.equity0)
    peak = curve.cummax()
    dd = float(((peak - curve) / peak * 100).max()) if len(curve) else 0.0
    rets = curve.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(252)) if len(rets) > 1 and rets.std(ddof=1) > 0 else 0.0
    final = float(curve.iloc[-1]) if len(curve) else rules.equity0
    return {**counters, "final_equity": final, "total_return_pct": (final / rules.equity0 - 1) * 100,
            "max_drawdown_pct": dd, "daily_sharpe": sharpe, "equity_curve": curve, "by_family": by_family}
