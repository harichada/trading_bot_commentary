#!/usr/bin/env python3
"""
Backtest Iteration Runner — automated backtest + analysis cycle.

Usage:
    python backtest_iterate.py                    # Run baseline (Iteration 1)
    python backtest_iterate.py --iteration 2      # Run iteration N
    python backtest_iterate.py --compare 1 2      # Compare two iterations
    python backtest_iterate.py --summary           # Show all iterations

Saves results to backtest_reports/iteration_N.json
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx

BASE_URL = os.environ.get("GAP_FADE_URL", "http://localhost:8003")
API_KEY = os.environ.get("GAP_FADE_API_KEY", "")
REPORT_DIR = Path("backtest_reports/iterations")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Success Criteria ──
CRITERIA = {
    "win_rate":       {"target": 0.55, "op": ">=", "label": "Win Rate >= 55%"},
    "profit_factor":  {"target": 1.3,  "op": ">=", "label": "PF >= 1.3"},
    "sharpe":         {"target": 1.5,  "op": ">=", "label": "Sharpe >= 1.5"},
    "max_drawdown_pct": {"target": 8.0, "op": "<=", "label": "Max DD <= 8%"},  # API returns as % (e.g. 19.57 = 19.57%)
}


def headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


async def run_backtest(config: dict, label: str = "") -> dict:
    """Run a backtest via the API, poll for completion, return results."""
    async with httpx.AsyncClient(timeout=600) as client:
        # Start backtest
        print(f"  Starting backtest{f' ({label})' if label else ''}...")
        resp = await client.post(f"{BASE_URL}/api/backtest", json=config, headers=headers())
        if resp.status_code != 200:
            print(f"  ERROR starting backtest: {resp.status_code} {resp.text[:200]}")
            return {}

        # Poll for completion
        t0 = time.time()
        last_pct = -1
        while True:
            await asyncio.sleep(3)
            status = await client.get(f"{BASE_URL}/api/backtest/status", headers=headers())
            data = status.json()
            pct = data.get("progress", 0)
            st = data.get("status", "unknown")

            if int(pct) > last_pct:
                elapsed = time.time() - t0
                print(f"  [{elapsed:5.0f}s] {pct:5.1f}% — {st}")
                last_pct = int(pct)

            if st in ("done", "error", "cancelled"):
                break
            if time.time() - t0 > 600:
                print("  TIMEOUT after 10 minutes")
                await client.post(f"{BASE_URL}/api/backtest/cancel", headers=headers())
                return {}

        # Get results
        result_resp = await client.get(f"{BASE_URL}/api/backtest/results", headers=headers())
        if result_resp.status_code != 200:
            print(f"  ERROR getting results: {result_resp.status_code}")
            return {}
        return result_resp.json()


def extract_metrics(result: dict) -> dict:
    """Pull key metrics from backtest result."""
    return {
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
        "avg_holding_min": result.get("avg_holding_min", 0),
        "gap_days_found": result.get("gap_days_found", 0),
        "gap_days_traded": result.get("gap_days_traded", 0),
        "final_equity": result.get("final_equity", 0),
    }


def check_criteria(metrics: dict) -> dict:
    """Check each success criterion. Returns dict of {name: {pass, value, target}}."""
    results = {}
    for key, crit in CRITERIA.items():
        val = metrics.get(key, 0)
        if crit["op"] == ">=":
            passed = val >= crit["target"]
        else:
            passed = val <= crit["target"]
        results[key] = {
            "pass": passed,
            "value": val,
            "target": crit["target"],
            "label": crit["label"],
        }
    return results


def analyze_trades(result: dict) -> dict:
    """Analyze trade-level data for problems."""
    trades = result.get("all_trades", result.get("trades", []))
    if not trades:
        return {"error": "no trades"}

    # Win/loss streaks
    streaks = []
    current_streak = 0
    for t in trades:
        pnl = t.get("pnl", 0)
        if pnl >= 0:
            current_streak = max(1, current_streak + 1) if current_streak > 0 else 1
        else:
            current_streak = min(-1, current_streak - 1) if current_streak < 0 else -1
        streaks.append(current_streak)
    max_win_streak = max(streaks) if streaks else 0
    max_loss_streak = abs(min(streaks)) if streaks else 0

    # By exit reason
    reasons = {}
    for t in trades:
        r = t.get("reason", "unknown")
        if r not in reasons:
            reasons[r] = {"count": 0, "pnl": 0, "wins": 0}
        reasons[r]["count"] += 1
        reasons[r]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) >= 0:
            reasons[r]["wins"] += 1

    # By direction
    directions = {}
    for t in trades:
        d = t.get("direction", t.get("side", "short"))
        if d not in directions:
            directions[d] = {"count": 0, "pnl": 0, "wins": 0}
        directions[d]["count"] += 1
        directions[d]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) >= 0:
            directions[d]["wins"] += 1

    # Worst symbols
    by_symbol = {}
    for t in trades:
        s = t.get("symbol", "?")
        if s not in by_symbol:
            by_symbol[s] = {"count": 0, "pnl": 0, "wins": 0}
        by_symbol[s]["count"] += 1
        by_symbol[s]["pnl"] += t.get("pnl", 0)
        if t.get("pnl", 0) >= 0:
            by_symbol[s]["wins"] += 1

    worst_symbols = sorted(by_symbol.items(), key=lambda x: x[1]["pnl"])[:10]
    best_symbols = sorted(by_symbol.items(), key=lambda x: x[1]["pnl"], reverse=True)[:10]

    # Average P&L by gap size bucket
    gap_buckets = {"7-10%": [], "10-15%": [], "15-20%": [], "20%+": []}
    for t in trades:
        gp = abs(t.get("gap_pct", 0)) if "gap_pct" in t else 0
        if gp == 0:
            continue
        if gp < 0.10:
            gap_buckets["7-10%"].append(t.get("pnl", 0))
        elif gp < 0.15:
            gap_buckets["10-15%"].append(t.get("pnl", 0))
        elif gp < 0.20:
            gap_buckets["15-20%"].append(t.get("pnl", 0))
        else:
            gap_buckets["20%+"].append(t.get("pnl", 0))

    gap_analysis = {}
    for bucket, pnls in gap_buckets.items():
        if pnls:
            gap_analysis[bucket] = {
                "count": len(pnls),
                "total_pnl": round(sum(pnls), 2),
                "avg_pnl": round(sum(pnls) / len(pnls), 2),
                "win_rate": round(sum(1 for p in pnls if p >= 0) / len(pnls), 3),
            }

    return {
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        "by_exit_reason": reasons,
        "by_direction": directions,
        "worst_symbols": worst_symbols,
        "best_symbols": best_symbols,
        "by_gap_size": gap_analysis,
    }


def identify_problems(metrics: dict, analysis: dict) -> list:
    """Auto-detect top problems from results."""
    problems = []

    # Win rate too low
    wr = metrics["win_rate"]
    wr_display = wr * 100 if wr <= 1 else wr  # handle both 0.55 and 55.0
    if wr_display < 55:
        problems.append({
            "severity": "high",
            "problem": f"Win rate {wr_display:.1f}% < 55% target",
            "suggestion": "Tighten gap_threshold, improve entry timing, or reduce vol_ratio_max",
        })

    # Profit factor
    if metrics["profit_factor"] < 1.3:
        problems.append({
            "severity": "high",
            "problem": f"Profit factor {metrics['profit_factor']:.2f} < 1.3 target",
            "suggestion": "Increase avg win (better targets) or decrease avg loss (tighter stops)",
        })

    # Max DD (API returns as %, e.g. 19.57 = 19.57%)
    dd = metrics["max_drawdown_pct"]
    if dd > 8.0:
        problems.append({
            "severity": "high",
            "problem": f"Max DD {dd:.1f}% > 8% target",
            "suggestion": "Enable dd_circuit_breaker, reduce max_positions, or reduce risk_pct",
        })

    # Loss streaks
    if analysis.get("max_loss_streak", 0) >= 5:
        problems.append({
            "severity": "medium",
            "problem": f"Max loss streak: {analysis['max_loss_streak']}",
            "suggestion": "Add max_consec_losses circuit breaker or regime filter",
        })

    # Stop-outs dominating
    reasons = analysis.get("by_exit_reason", {})
    stop_count = reasons.get("stop", {}).get("count", 0) + reasons.get("stop_adverse", {}).get("count", 0)
    total = metrics["total_trades"]
    if total > 0 and stop_count / total > 0.40:
        problems.append({
            "severity": "medium",
            "problem": f"Stop-outs are {stop_count}/{total} ({stop_count/total:.0%}) of trades",
            "suggestion": "Widen stops (increase stop_pct/stop_max_pct) or use adaptive stops",
        })

    # Direction imbalance
    dirs = analysis.get("by_direction", {})
    for d, stats in dirs.items():
        if stats["count"] > 10:
            wr = stats["wins"] / stats["count"]
            if wr < 0.35:
                problems.append({
                    "severity": "medium",
                    "problem": f"{d} direction win rate only {wr:.0%} ({stats['count']} trades, P&L ${stats['pnl']:.0f})",
                    "suggestion": f"Consider disabling {d} trades or adjusting thresholds",
                })

    # Worst symbols dragging P&L
    worst = analysis.get("worst_symbols", [])
    if worst and worst[0][1]["pnl"] < -500:
        top3 = [(s, d["pnl"], d["count"]) for s, d in worst[:3]]
        problems.append({
            "severity": "low",
            "problem": f"Worst symbols: {', '.join(f'{s} (${p:.0f}, {c}t)' for s,p,c in top3)}",
            "suggestion": "Consider blacklisting or the high-volatility filter",
        })

    # Gap size buckets
    gap_data = analysis.get("by_gap_size", {})
    for bucket, stats in gap_data.items():
        if stats["count"] >= 5 and stats["win_rate"] < 0.30:
            problems.append({
                "severity": "medium",
                "problem": f"Gap {bucket}: {stats['win_rate']:.0%} win rate, ${stats['total_pnl']:.0f} P&L ({stats['count']} trades)",
                "suggestion": f"Adjust gap_threshold or max_gap_pct to exclude this bucket",
            })

    problems.sort(key=lambda p: {"high": 0, "medium": 1, "low": 2}[p["severity"]])
    return problems


def print_report(iteration: int, metrics: dict, criteria: dict, analysis: dict, problems: list, config: dict):
    """Print formatted iteration report."""
    print("\n" + "=" * 70)
    print(f"  ITERATION {iteration} RESULTS")
    print("=" * 70)

    # Metrics summary
    print(f"\n  Trades: {metrics['total_trades']:,}  |  "
          f"Win Rate: {metrics['win_rate']:.1%}  |  "
          f"P&L: ${metrics['total_pnl']:,.0f}  |  "
          f"Return: {metrics['return_pct']:.1f}%")
    print(f"  PF: {metrics['profit_factor']:.2f}  |  "
          f"Sharpe: {metrics['sharpe']:.2f}  |  "
          f"Max DD: {metrics['max_drawdown_pct']:.1f}%  |  "
          f"Avg W/L: ${metrics['avg_win']:.0f}/${metrics['avg_loss']:.0f}")

    # Success criteria
    print(f"\n  {'CRITERION':<25} {'VALUE':>10} {'TARGET':>10} {'STATUS':>8}")
    print(f"  {'-'*55}")
    all_pass = True
    for key, c in criteria.items():
        status = "PASS" if c["pass"] else "FAIL"
        mark = "+" if c["pass"] else "X"
        if not c["pass"]:
            all_pass = False
        if key == "max_drawdown_pct":
            print(f"  {c['label']:<25} {c['value']:>8.1f}% {c['target']:>8.1f}% [{mark}] {status}")
        elif key == "win_rate":
            print(f"  {c['label']:<25} {c['value']:>9.1%} {c['target']:>9.1%} [{mark}] {status}")
        else:
            print(f"  {c['label']:<25} {c['value']:>10.2f} {c['target']:>10.2f} [{mark}] {status}")

    if all_pass:
        print(f"\n  >>> ALL CRITERIA MET <<<")

    # Exit reasons
    print(f"\n  Exit Reasons:")
    for reason, stats in sorted(analysis.get("by_exit_reason", {}).items(), key=lambda x: -x[1]["count"]):
        wr = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
        print(f"    {reason:<20} {stats['count']:>4} trades  WR {wr:.0%}  P&L ${stats['pnl']:>8,.0f}")

    # Direction breakdown
    print(f"\n  By Direction:")
    for d, stats in analysis.get("by_direction", {}).items():
        wr = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
        print(f"    {d:<10} {stats['count']:>4} trades  WR {wr:.0%}  P&L ${stats['pnl']:>8,.0f}")

    # Gap size analysis
    gap_data = analysis.get("by_gap_size", {})
    if gap_data:
        print(f"\n  By Gap Size:")
        for bucket in ["7-10%", "10-15%", "15-20%", "20%+"]:
            if bucket in gap_data:
                s = gap_data[bucket]
                print(f"    {bucket:<8} {s['count']:>4} trades  WR {s['win_rate']:.0%}  "
                      f"Avg ${s['avg_pnl']:>6,.0f}  Total ${s['total_pnl']:>8,.0f}")

    # Problems
    if problems:
        print(f"\n  Top Problems:")
        for i, p in enumerate(problems[:5], 1):
            sev = {"high": "!!!", "medium": " ! ", "low": " . "}[p["severity"]]
            print(f"    {sev} {i}. {p['problem']}")
            print(f"        -> {p['suggestion']}")

    # Config used
    print(f"\n  Config Overrides: {json.dumps(config, indent=None)}")
    print("=" * 70)


def print_comparison(iter_a: int, iter_b: int):
    """Compare two iterations side by side."""
    file_a = REPORT_DIR / f"iteration_{iter_a}.json"
    file_b = REPORT_DIR / f"iteration_{iter_b}.json"
    if not file_a.exists() or not file_b.exists():
        print(f"Missing iteration file(s)")
        return

    a = json.loads(file_a.read_text())
    b = json.loads(file_b.read_text())
    ma, mb = a["metrics"], b["metrics"]

    print(f"\n{'=' * 60}")
    print(f"  COMPARISON: Iteration {iter_a} vs {iter_b}")
    print(f"{'=' * 60}")
    print(f"\n  {'METRIC':<20} {'Iter '+str(iter_a):>12} {'Iter '+str(iter_b):>12} {'DELTA':>12}")
    print(f"  {'-'*58}")

    for key, fmt in [
        ("total_trades", "{:.0f}"),
        ("win_rate", "{:.1%}"),
        ("total_pnl", "${:,.0f}"),
        ("return_pct", "{:.1f}%"),
        ("profit_factor", "{:.2f}"),
        ("sharpe", "{:.2f}"),
        ("max_drawdown_pct", "{:.1f}%"),
        ("avg_win", "${:.0f}"),
        ("avg_loss", "${:.0f}"),
    ]:
        va = ma.get(key, 0)
        vb = mb.get(key, 0)
        delta = vb - va
        sign = "+" if delta > 0 else ""
        # Format values
        va_s = fmt.format(va)
        vb_s = fmt.format(vb)
        if "%" in fmt and key != "return_pct":
            delta_s = f"{sign}{delta:.1%}"
        elif "$" in fmt:
            delta_s = f"{sign}${delta:,.0f}"
        else:
            delta_s = f"{sign}{delta:.2f}"
        print(f"  {key:<20} {va_s:>12} {vb_s:>12} {delta_s:>12}")

    # Config diff
    ca = a.get("config_overrides", {})
    cb = b.get("config_overrides", {})
    all_keys = set(list(ca.keys()) + list(cb.keys()))
    if all_keys:
        print(f"\n  Config Changes:")
        for k in sorted(all_keys):
            va = ca.get(k, "(default)")
            vb = cb.get(k, "(default)")
            if va != vb:
                print(f"    {k}: {va} -> {vb}")

    # Problem changes
    pa = [p["problem"] for p in a.get("problems", [])]
    pb = [p["problem"] for p in b.get("problems", [])]
    fixed = [p for p in pa if p not in pb]
    new = [p for p in pb if p not in pa]
    if fixed:
        print(f"\n  Fixed Problems:")
        for p in fixed:
            print(f"    [+] {p}")
    if new:
        print(f"\n  New Problems:")
        for p in new:
            print(f"    [-] {p}")

    print(f"{'=' * 60}")


def print_summary():
    """Show all iterations summary."""
    files = sorted(REPORT_DIR.glob("iteration_*.json"))
    if not files:
        print("No iterations found.")
        return

    print(f"\n{'=' * 90}")
    print(f"  ALL ITERATIONS SUMMARY")
    print(f"{'=' * 90}")
    print(f"  {'#':>3} {'Date':<12} {'Trades':>7} {'WR':>7} {'P&L':>10} {'PF':>6} {'Sharpe':>7} {'MaxDD':>7} {'Pass':>5}")
    print(f"  {'-'*72}")

    for f in files:
        data = json.loads(f.read_text())
        m = data["metrics"]
        n_pass = sum(1 for c in data.get("criteria", {}).values() if c.get("pass"))
        total_crit = len(CRITERIA)
        it = data.get("iteration", "?")
        dt = data.get("timestamp", "")[:10]
        print(f"  {it:>3} {dt:<12} {m['total_trades']:>7,} {m['win_rate']:>6.1%} "
              f"${m['total_pnl']:>9,.0f} {m['profit_factor']:>5.2f} {m['sharpe']:>6.2f} "
              f"{m['max_drawdown_pct']:>5.1f}% {n_pass}/{total_crit}")

    print(f"{'=' * 90}")


# ── Iteration configs ──
# Each iteration builds on the previous, targeting one problem at a time.
ITERATION_CONFIGS = {
    1: {
        "label": "Baseline — defaults + re-entry + gap-down longs",
        "config": {
            # Use broad universe, 3yr lookback
            "start_date": "2023-03-08",
            "end_date": "2026-03-08",
            "universe": "alpaca",
            "reentry_enabled": True,
            "trade_gap_downs": True,
        },
    },
    # Future iterations will be added as we analyze results
}


async def run_iteration(iteration: int, custom_config: dict = None):
    """Run a single backtest iteration."""
    if custom_config:
        config = custom_config
        label = f"Custom iteration {iteration}"
    elif iteration in ITERATION_CONFIGS:
        config = ITERATION_CONFIGS[iteration]["config"]
        label = ITERATION_CONFIGS[iteration]["label"]
    else:
        print(f"No config for iteration {iteration}. Use --config to provide one.")
        return

    print(f"\n{'#' * 60}")
    print(f"  ITERATION {iteration}: {label}")
    print(f"{'#' * 60}")

    result = await run_backtest(config, label)
    if not result or result.get("error"):
        print(f"  Backtest failed: {result.get('error', 'unknown')}")
        return

    metrics = extract_metrics(result)
    criteria = check_criteria(metrics)
    analysis = analyze_trades(result)
    problems = identify_problems(metrics, analysis)

    # Save report
    report = {
        "iteration": iteration,
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "config_overrides": config,
        "metrics": metrics,
        "criteria": criteria,
        "analysis": analysis,
        "problems": problems,
    }
    report_file = REPORT_DIR / f"iteration_{iteration}.json"
    report_file.write_text(json.dumps(report, indent=2, default=str))

    print_report(iteration, metrics, criteria, analysis, problems, config)
    print(f"\n  Report saved: {report_file}")

    # Compare with previous if exists
    if iteration > 1:
        prev = REPORT_DIR / f"iteration_{iteration - 1}.json"
        if prev.exists():
            print_comparison(iteration - 1, iteration)

    return report


def main():
    parser = argparse.ArgumentParser(description="Backtest Iteration Runner")
    parser.add_argument("--iteration", "-i", type=int, default=1, help="Iteration number")
    parser.add_argument("--compare", nargs=2, type=int, metavar=("A", "B"), help="Compare two iterations")
    parser.add_argument("--summary", "-s", action="store_true", help="Show all iterations")
    parser.add_argument("--config", type=str, help="JSON config override string")
    parser.add_argument("--start", type=str, help="Start date override (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date override (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    if args.compare:
        print_comparison(args.compare[0], args.compare[1])
        return

    config = None
    if args.config:
        config = json.loads(args.config)
    if args.start or args.end:
        if config is None and args.iteration in ITERATION_CONFIGS:
            config = dict(ITERATION_CONFIGS[args.iteration]["config"])
        elif config is None:
            config = {}
        if args.start:
            config["start_date"] = args.start
        if args.end:
            config["end_date"] = args.end

    asyncio.run(run_iteration(args.iteration, config))


if __name__ == "__main__":
    main()
