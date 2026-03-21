#!/usr/bin/env python3
"""
Automated backtest parameter sweep — find optimal gap fade configuration.

Runs multiple backtests varying key parameters, evaluates each against
success criteria, and saves the best configuration.

Usage:
    python backtest_sweep.py
    python backtest_sweep.py --quick        # Shorter date range (1 year)
    python backtest_sweep.py --resume N     # Resume from iteration N
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
import jwt
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.environ.get("GAP_FADE_URL", "http://localhost:8003")
REPORT_DIR = Path("backtest_reports/sweep")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Generate auth token for API calls
JWT_SECRET = os.environ.get("AUTH_JWT_SECRET", "")
AUTH_EMAIL = os.environ.get("AUTH_ALLOWED_EMAILS", "").split(",")[0].strip()


def get_headers():
    h = {"Content-Type": "application/json"}
    if JWT_SECRET and AUTH_EMAIL:
        token = jwt.encode(
            {"sub": AUTH_EMAIL, "name": "sweep", "iat": int(time.time()),
             "exp": int(time.time()) + 7200},
            JWT_SECRET, algorithm="HS256")
        h["Cookie"] = f"gf_session={token}"
    return h


# ── Base config (shared across all iterations) ──
BASE_CONFIG = {
    "start_date": "2023-06-01",
    "end_date": "2026-03-20",
    "universe": "alpaca",
    "initial_capital": 25000,
    "reentry_enabled": True,
    "reentry_cooldown_minutes": 30,
    "reentry_max_per_symbol": 1,
    "reentry_stop_pct": 0.01,
    "limit_orders_only": True,
    "limit_offset_pct": 0.001,
    "slippage_pct": 0.0015,
    "dd_circuit_breaker": True,
    "dd_tier1_threshold": 0.06,
    "dd_tier1_scale": 0.5,
    "dd_tier2_threshold": 0.08,
    "dd_tier2_scale": 0.25,
    "dd_tier2_max_positions": 1,
    "entry_cutoff_hour": 11,
    "entry_cutoff_min": 30,
    "time_exit_hour": 15,
    "time_exit_min": 0,
    "eod_exit_hour": 15,
    "eod_exit_min": 50,
}


# ── Parameter sweep configurations ──
ITERATIONS = [
    # === Group 1: Gap threshold and direction ===
    {
        "label": "Baseline: 7% gap, shorts only, adaptive stops",
        "overrides": {
            "gap_threshold": 0.07, "max_gap_pct": 0.12, "vol_ratio_max": 1.0,
            "max_positions": 3, "risk_pct": 0.008,
            "trade_gap_downs": False, "regime_filter": True,
            "adaptive_stops": True, "stop_gap_fraction": 0.25,
            "stop_min_pct": 0.015, "stop_max_pct": 0.025,
            "trailing_stop_enabled": True, "trailing_activation_pct": 0.01,
            "trailing_distance_pct": 0.005,
        },
    },
    {
        "label": "Lower threshold: 5% gap, shorts only",
        "overrides": {
            "gap_threshold": 0.05, "max_gap_pct": 0.12, "vol_ratio_max": 1.0,
            "max_positions": 3, "risk_pct": 0.008,
            "trade_gap_downs": False, "regime_filter": True,
            "adaptive_stops": True, "stop_gap_fraction": 0.25,
            "stop_min_pct": 0.015, "stop_max_pct": 0.025,
            "trailing_stop_enabled": True, "trailing_activation_pct": 0.01,
            "trailing_distance_pct": 0.005,
        },
    },
    {
        "label": "Wide range: 5-15% gap, shorts only",
        "overrides": {
            "gap_threshold": 0.05, "max_gap_pct": 0.15, "vol_ratio_max": 1.5,
            "max_positions": 3, "risk_pct": 0.008,
            "trade_gap_downs": False, "regime_filter": True,
            "adaptive_stops": True, "stop_gap_fraction": 0.25,
            "stop_min_pct": 0.015, "stop_max_pct": 0.025,
            "trailing_stop_enabled": True, "trailing_activation_pct": 0.01,
            "trailing_distance_pct": 0.005,
        },
    },
    {
        "label": "Both directions: 7% gap, shorts + longs",
        "overrides": {
            "gap_threshold": 0.07, "max_gap_pct": 0.12, "vol_ratio_max": 1.0,
            "max_positions": 3, "risk_pct": 0.008,
            "trade_gap_downs": True, "regime_filter": True,
            "adaptive_stops": True, "stop_gap_fraction": 0.25,
            "stop_min_pct": 0.015, "stop_max_pct": 0.025,
            "trailing_stop_enabled": True, "trailing_activation_pct": 0.01,
            "trailing_distance_pct": 0.005,
        },
    },

    # === Group 2: Trailing stop tuning ===
    {
        "label": "Tight trail: 0.3% distance, 0.5% activation",
        "overrides": {
            "gap_threshold": 0.05, "max_gap_pct": 0.12, "vol_ratio_max": 1.0,
            "max_positions": 3, "risk_pct": 0.008,
            "trade_gap_downs": False, "regime_filter": True,
            "adaptive_stops": True, "stop_gap_fraction": 0.25,
            "stop_min_pct": 0.015, "stop_max_pct": 0.025,
            "trailing_stop_enabled": True, "trailing_activation_pct": 0.005,
            "trailing_distance_pct": 0.003,
        },
    },
    {
        "label": "Wide trail: 1.0% distance, 1.5% activation",
        "overrides": {
            "gap_threshold": 0.05, "max_gap_pct": 0.12, "vol_ratio_max": 1.0,
            "max_positions": 3, "risk_pct": 0.008,
            "trade_gap_downs": False, "regime_filter": True,
            "adaptive_stops": True, "stop_gap_fraction": 0.25,
            "stop_min_pct": 0.015, "stop_max_pct": 0.025,
            "trailing_stop_enabled": True, "trailing_activation_pct": 0.015,
            "trailing_distance_pct": 0.01,
        },
    },
    {
        "label": "No trailing stop (control)",
        "overrides": {
            "gap_threshold": 0.05, "max_gap_pct": 0.12, "vol_ratio_max": 1.0,
            "max_positions": 3, "risk_pct": 0.008,
            "trade_gap_downs": False, "regime_filter": True,
            "adaptive_stops": True, "stop_gap_fraction": 0.25,
            "stop_min_pct": 0.015, "stop_max_pct": 0.025,
            "trailing_stop_enabled": False,
        },
    },

    # === Group 3: Stop sizing ===
    {
        "label": "Wider stops: gap_fraction 0.35, min 2%, max 3.5%",
        "overrides": {
            "gap_threshold": 0.05, "max_gap_pct": 0.12, "vol_ratio_max": 1.0,
            "max_positions": 3, "risk_pct": 0.008,
            "trade_gap_downs": False, "regime_filter": True,
            "adaptive_stops": True, "stop_gap_fraction": 0.35,
            "stop_min_pct": 0.02, "stop_max_pct": 0.035,
            "trailing_stop_enabled": True, "trailing_activation_pct": 0.01,
            "trailing_distance_pct": 0.005,
        },
    },
    {
        "label": "Tighter stops: gap_fraction 0.15, min 1%, max 2%",
        "overrides": {
            "gap_threshold": 0.05, "max_gap_pct": 0.12, "vol_ratio_max": 1.0,
            "max_positions": 3, "risk_pct": 0.008,
            "trade_gap_downs": False, "regime_filter": True,
            "adaptive_stops": True, "stop_gap_fraction": 0.15,
            "stop_min_pct": 0.01, "stop_max_pct": 0.02,
            "trailing_stop_enabled": True, "trailing_activation_pct": 0.01,
            "trailing_distance_pct": 0.005,
        },
    },

    # === Group 4: Position sizing and risk ===
    {
        "label": "Higher risk: 1.2% per trade, 4 max positions",
        "overrides": {
            "gap_threshold": 0.05, "max_gap_pct": 0.12, "vol_ratio_max": 1.0,
            "max_positions": 4, "risk_pct": 0.012,
            "trade_gap_downs": False, "regime_filter": True,
            "adaptive_stops": True, "stop_gap_fraction": 0.25,
            "stop_min_pct": 0.015, "stop_max_pct": 0.025,
            "trailing_stop_enabled": True, "trailing_activation_pct": 0.01,
            "trailing_distance_pct": 0.005,
        },
    },
    {
        "label": "Conservative: 0.5% risk, 2 max positions",
        "overrides": {
            "gap_threshold": 0.05, "max_gap_pct": 0.12, "vol_ratio_max": 1.0,
            "max_positions": 2, "risk_pct": 0.005,
            "trade_gap_downs": False, "regime_filter": True,
            "adaptive_stops": True, "stop_gap_fraction": 0.25,
            "stop_min_pct": 0.015, "stop_max_pct": 0.025,
            "trailing_stop_enabled": True, "trailing_activation_pct": 0.01,
            "trailing_distance_pct": 0.005,
        },
    },

    # === Group 5: Volume and regime filters ===
    {
        "label": "Relaxed volume: vol_ratio_max 2.0",
        "overrides": {
            "gap_threshold": 0.05, "max_gap_pct": 0.12, "vol_ratio_max": 2.0,
            "max_positions": 3, "risk_pct": 0.008,
            "trade_gap_downs": False, "regime_filter": True,
            "adaptive_stops": True, "stop_gap_fraction": 0.25,
            "stop_min_pct": 0.015, "stop_max_pct": 0.025,
            "trailing_stop_enabled": True, "trailing_activation_pct": 0.01,
            "trailing_distance_pct": 0.005,
        },
    },
    {
        "label": "No regime filter",
        "overrides": {
            "gap_threshold": 0.05, "max_gap_pct": 0.12, "vol_ratio_max": 1.0,
            "max_positions": 3, "risk_pct": 0.008,
            "trade_gap_downs": False, "regime_filter": False,
            "adaptive_stops": True, "stop_gap_fraction": 0.25,
            "stop_min_pct": 0.015, "stop_max_pct": 0.025,
            "trailing_stop_enabled": True, "trailing_activation_pct": 0.01,
            "trailing_distance_pct": 0.005,
        },
    },
]

# ── Success criteria ──
CRITERIA = {
    "win_rate":       {"target": 0.53, "op": ">=", "label": "Win Rate >= 53%"},
    "profit_factor":  {"target": 1.3,  "op": ">=", "label": "PF >= 1.3"},
    "sharpe":         {"target": 1.5,  "op": ">=", "label": "Sharpe >= 1.5"},
    "max_drawdown_pct": {"target": 10.0, "op": "<=", "label": "Max DD <= 10%"},
}


async def run_backtest(config: dict, label: str = "") -> dict:
    """Run a backtest via the API, poll for completion."""
    headers = get_headers()
    async with httpx.AsyncClient(timeout=600) as client:
        print(f"\n  Starting: {label}")
        resp = await client.post(f"{BASE_URL}/api/backtest", json=config, headers=headers)
        if resp.status_code != 200:
            print(f"  ERROR: {resp.status_code} {resp.text[:300]}")
            return {}

        t0 = time.time()
        last_pct = -1
        while True:
            await asyncio.sleep(5)
            status = await client.get(f"{BASE_URL}/api/backtest/status", headers=headers)
            data = status.json()
            pct = data.get("progress", 0)
            st = data.get("status", "unknown")

            if int(pct) > last_pct:
                elapsed = time.time() - t0
                print(f"  [{elapsed:5.0f}s] {pct:5.1f}% — {st}")
                last_pct = int(pct)

            if st in ("done", "error", "cancelled"):
                break
            if time.time() - t0 > 900:
                print("  TIMEOUT after 15 minutes")
                await client.post(f"{BASE_URL}/api/backtest/cancel", headers=headers)
                return {}

        result_resp = await client.get(f"{BASE_URL}/api/backtest/results", headers=headers)
        if result_resp.status_code != 200:
            print(f"  ERROR getting results: {result_resp.status_code}")
            return {}
        return result_resp.json()


def score_result(metrics: dict) -> float:
    """Composite score: higher is better. Balances PF, WR, Sharpe, drawdown, and trade count."""
    pf = metrics.get("profit_factor", 0)
    wr = metrics.get("win_rate", 0)
    sharpe = metrics.get("sharpe", 0)
    dd = metrics.get("max_drawdown_pct", 100)
    trades = metrics.get("total_trades", 0)
    pnl = metrics.get("total_pnl", 0)

    if trades < 50 or pf <= 0:
        return 0.0

    # Weighted composite (normalize each to ~0-10 range)
    score = (
        pf * 3.0                          # profit factor (most important)
        + wr * 10.0                       # win rate
        + min(sharpe, 5) * 1.0            # sharpe (capped)
        + max(0, (15 - dd)) * 0.5         # drawdown penalty
        + min(trades / 100, 3) * 1.0      # trade count bonus (up to 300)
        + (1 if pnl > 0 else -5)          # profitability bonus/penalty
    )
    return round(score, 2)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="1-year backtest")
    parser.add_argument("--resume", type=int, default=0, help="Resume from iteration N")
    args = parser.parse_args()

    if args.quick:
        BASE_CONFIG["start_date"] = "2025-03-20"

    results_all = []
    best_score = 0
    best_idx = -1

    for i, iteration in enumerate(ITERATIONS):
        if i < args.resume:
            continue

        config = {**BASE_CONFIG, **iteration["overrides"]}
        label = f"[{i+1}/{len(ITERATIONS)}] {iteration['label']}"

        result = await run_backtest(config, label)
        if not result:
            results_all.append({"iteration": i + 1, "label": iteration["label"], "error": True})
            continue

        metrics = {
            "total_trades": result.get("total_trades", 0),
            "wins": result.get("wins", 0),
            "losses": result.get("losses", 0),
            "win_rate": result.get("win_rate", 0),
            "total_pnl": result.get("total_pnl", 0),
            "return_pct": result.get("return_pct", 0),
            "profit_factor": result.get("profit_factor", 0),
            "sharpe": result.get("sharpe", 0),
            "max_drawdown_pct": result.get("max_drawdown_pct", 0),
            "avg_win": result.get("avg_win", 0),
            "avg_loss": result.get("avg_loss", 0),
            "final_equity": result.get("final_equity", 0),
        }

        # Check criteria
        criteria_results = {}
        all_pass = True
        for key, crit in CRITERIA.items():
            val = metrics.get(key, 0)
            passed = val >= crit["target"] if crit["op"] == ">=" else val <= crit["target"]
            criteria_results[key] = {"pass": passed, "value": val, "target": crit["target"]}
            if not passed:
                all_pass = False

        score = score_result(metrics)

        entry = {
            "iteration": i + 1,
            "label": iteration["label"],
            "config_overrides": iteration["overrides"],
            "metrics": metrics,
            "criteria": criteria_results,
            "all_criteria_pass": all_pass,
            "score": score,
        }
        results_all.append(entry)

        # Print summary
        pass_str = "✓ ALL PASS" if all_pass else "✗ FAIL"
        print(f"\n  Results: {pass_str} | Score: {score:.1f}")
        print(f"  Trades: {metrics['total_trades']} | WR: {metrics['win_rate']:.1%} | "
              f"PF: {metrics['profit_factor']:.2f} | Sharpe: {metrics['sharpe']:.2f}")
        print(f"  P&L: ${metrics['total_pnl']:+,.2f} | DD: {metrics['max_drawdown_pct']:.1f}% | "
              f"Equity: ${metrics['final_equity']:,.2f}")

        if score > best_score:
            best_score = score
            best_idx = i

        # Save individual result
        report_file = REPORT_DIR / f"sweep_{i+1:02d}.json"
        with open(report_file, "w") as f:
            json.dump(entry, f, indent=2, default=str)

    # ── Final summary ──
    print("\n" + "=" * 80)
    print("SWEEP COMPLETE — RESULTS RANKED BY SCORE")
    print("=" * 80)

    ranked = sorted([r for r in results_all if not r.get("error")],
                    key=lambda x: x.get("score", 0), reverse=True)

    print(f"\n{'#':>3} {'Score':>6} {'Pass':>5} {'Trades':>7} {'WR':>6} {'PF':>6} "
          f"{'Sharpe':>7} {'DD%':>6} {'P&L':>10} Label")
    print("-" * 95)
    for r in ranked:
        m = r["metrics"]
        p = "✓" if r.get("all_criteria_pass") else "✗"
        print(f"{r['iteration']:3d} {r['score']:6.1f} {p:>5} {m['total_trades']:7d} "
              f"{m['win_rate']:5.1%} {m['profit_factor']:6.2f} {m['sharpe']:7.2f} "
              f"{m['max_drawdown_pct']:5.1f}% {m['total_pnl']:+10,.0f} {r['label']}")

    if best_idx >= 0:
        best = ranked[0]
        print(f"\n🏆 BEST: #{best['iteration']} — {best['label']}")
        print(f"   Config: {json.dumps(best['config_overrides'], indent=2)}")

        # Save best config
        best_file = REPORT_DIR / "best_config.json"
        with open(best_file, "w") as f:
            json.dump(best, f, indent=2, default=str)
        print(f"   Saved to: {best_file}")

    # Save full summary
    summary_file = REPORT_DIR / "sweep_summary.json"
    with open(summary_file, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": ranked}, f, indent=2, default=str)


if __name__ == "__main__":
    asyncio.run(main())
