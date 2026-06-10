#!/usr/bin/env python3
"""v-bot-monitor-2026-06-01: local agent that monitors the trading
bot and produces structured Markdown reports for Claude to consume
on demand.

Architecture
------------
PURE OBSERVER. Never writes to any bot-controlled file or API. Only
reads:
  - trading_bot.log         (tails for recent events)
  - trading_state.json      (positions, trade history)
  - token_1.json            (Schwab token TTL)
  - GET /api/status         (bot self-report)
  - GET /api/positions/db   (in-memory positions)
  - GET /api/market-indices (regime snapshot)

Writes ONLY to docs/monitor/latest.md (+ timestamped history in
docs/monitor/<YYYY-MM-DD>/<HH-MM>.md). The bot itself never sees
these files.

Usage
-----
  python tools/bot_monitor.py                  # daemon, 60s poll
  python tools/bot_monitor.py --interval 30    # custom interval
  python tools/bot_monitor.py --once           # single report + exit
  python tools/bot_monitor.py --once --quiet   # report file only, no stdout
  python tools/bot_monitor.py --no-history     # skip timestamped history

How Claude consumes it
----------------------
Operator says "check status" or "ask the monitor"; Claude reads
docs/monitor/latest.md. The report is written in Markdown optimized
for Claude consumption (tables, ordered headings, no fluff).

For long-running background operation:
  nohup python tools/bot_monitor.py > /tmp/monitor.out 2>&1 &
  disown
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# Resolve repo root relative to this file
REPO = Path(__file__).resolve().parent.parent
LOG_FILE = REPO / "trading_bot.log"
STATE_FILE = REPO / "trading_state.json"
TOKEN_FILE = REPO / "token_1.json"
ENV_FILE = REPO / ".env"
REPORT_DIR = REPO / "docs" / "monitor"
LATEST_REPORT = REPORT_DIR / "latest.md"


# ────────────────────────────────────────────────────────────────────
# Data collection — each function is a single, testable read
# ────────────────────────────────────────────────────────────────────

def get_bot_process() -> Optional[dict]:
    """Returns {pid, uptime_sec, command} or None if bot not running."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", "trading_bot_commentary_updated"],
            capture_output=True, text=True, timeout=3,
        )
        pids = [p for p in out.stdout.strip().split("\n") if p.strip()]
        if not pids:
            return None
        pid = int(pids[0])
        # Read uptime from /proc/<pid>/stat — field 22 is start_time in
        # clock ticks since boot.
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        # Format: "pid (comm) state ppid ..."
        # Comm can contain spaces; parse from the LAST closing paren.
        rparen = stat.rindex(")")
        rest = stat[rparen + 2:].split()
        start_ticks = int(rest[19])  # field 22 - 3 (offset from 1-indexed)
        clk = os.sysconf("SC_CLK_TCK")
        boot_time = float((Path("/proc") / "stat").read_text().split("btime ")[1].split()[0])
        start_epoch = boot_time + start_ticks / clk
        uptime = time.time() - start_epoch
        return {
            "pid": pid,
            "uptime_sec": uptime,
            "uptime_human": _fmt_duration(uptime),
        }
    except Exception:
        return None


def get_token_ttl_hours() -> Optional[float]:
    try:
        t = json.loads(TOKEN_FILE.read_text())
        created = datetime.fromtimestamp(t.get("creation_timestamp", 0), tz=timezone.utc)
        expires = created + timedelta(days=7)
        return (expires - datetime.now(timezone.utc)).total_seconds() / 3600.0
    except Exception:
        return None


def get_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def get_api_key() -> Optional[str]:
    try:
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith("TRADING_API_KEY="):
                val = line.split("=", 1)[1].strip()
                return val.strip('"').strip("'")
    except Exception:
        return None
    return None


def fetch_api(path: str, port: int = 9000, timeout: float = 3.0) -> Optional[dict]:
    """Fetch JSON from the bot's REST API with auth header."""
    key = get_api_key()
    if not key:
        return None
    try:
        import urllib.request
        req = urllib.request.Request(
            f"http://localhost:{port}{path}",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def tail_log(seconds_back: int = 3600) -> list[dict]:
    """Parse log entries from the last N seconds. Each entry is a dict
    with keys: timestamp, level, message, module, function, line.

    The bot logs in JSON Lines format; we read backwards efficiently
    by limiting how much of the file we scan.
    """
    cutoff = datetime.now() - timedelta(seconds=seconds_back)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    entries: list[dict] = []
    try:
        # Read last ~3MB of the log — enough for an hour or so on a busy day.
        size = LOG_FILE.stat().st_size
        read_back = min(size, 3_000_000)
        with LOG_FILE.open("rb") as f:
            f.seek(size - read_back)
            data = f.read().decode("utf-8", errors="ignore")
        # v-bot-monitor-tail-fix-2026-06-01: previous code did
        # `data.split("\n", 1)[1:]` which yielded a SINGLE string
        # (the rest of the file after the first newline) rather
        # than a list of lines. The loop then called json.loads()
        # on the whole multi-entry blob and failed silently, so
        # the monitor produced empty reports despite a busy log.
        # splitlines() properly returns one entry per line; the
        # [1:] skips the (likely partial) first line.
        lines = data.splitlines()[1:] if "\n" in data else []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts = entry.get("timestamp", "")
                # Compare lexicographically — ISO-ish format is sortable.
                if ts >= cutoff_str:
                    entries.append(entry)
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    return entries


# ────────────────────────────────────────────────────────────────────
# Analysis — turn raw events into summaries
# ────────────────────────────────────────────────────────────────────

def parse_signal_activity(entries: list[dict]) -> dict:
    """Count signals fired, accepted, blocked (with reasons)."""
    fired_by_strategy: Counter = Counter()
    accepted_by_strategy: Counter = Counter()
    block_reasons: Counter = Counter()
    rr_audit_low: list[dict] = []
    direction_gate_skips: list[dict] = []
    fills: list[dict] = []
    errors: list[dict] = []

    for e in entries:
        msg = e.get("message", "")

        # Signals fired
        m = re.search(r"strategy=(\w+).*action=signal_(buy|sell)", msg)
        if m:
            fired_by_strategy[m.group(1)] += 1

        # Signals accepted by router
        if "signal_router" in msg and "action=accepted" in msg:
            sm = re.search(r"strategy=(\w+)|reason=(\w+)", msg)
            strategy = "unknown"
            if sm:
                strategy = sm.group(1) or sm.group(2) or "unknown"
            accepted_by_strategy[strategy] += 1

        # Block reasons (signal_router skips)
        if "signal_router" in msg and "action=skip" in msg:
            rm = re.search(r"reason=([a-z0-9_]+)", msg)
            if rm:
                block_reasons[rm.group(1)] += 1

        # R:R audit warnings
        if "rr_audit_low" in msg:
            symbol_m = re.search(r"symbol=([A-Z]+)", msg)
            rr_m = re.search(r"actual_rr=([0-9.]+)", msg)
            rr_audit_low.append({
                "ts": e.get("timestamp", ""),
                "symbol": symbol_m.group(1) if symbol_m else "?",
                "rr": float(rr_m.group(1)) if rr_m else None,
            })

        # Direction gate skips
        if "direction_gate_rejected" in msg:
            symbol_m = re.search(r"symbol=([A-Z]+)", msg)
            phase_m = re.search(r"phase=(\w+)", msg)
            dir_m = re.search(r"direction=([+\-0-9.]+)", msg)
            direction_gate_skips.append({
                "ts": e.get("timestamp", ""),
                "symbol": symbol_m.group(1) if symbol_m else "?",
                "phase": phase_m.group(1) if phase_m else "?",
                "direction": float(dir_m.group(1)) if dir_m else None,
            })

        # Order placed
        if msg.startswith("Order placed:"):
            fills.append({"ts": e.get("timestamp", ""), "raw": msg})

        # Errors
        if e.get("level") in ("ERROR", "CRITICAL"):
            errors.append({"ts": e.get("timestamp", ""), "msg": msg[:200]})

    return {
        "fired_by_strategy": dict(fired_by_strategy),
        "accepted_by_strategy": dict(accepted_by_strategy),
        "block_reasons": dict(block_reasons),
        "rr_audit_low": rr_audit_low,
        "direction_gate_skips": direction_gate_skips,
        "fills": fills,
        "errors": errors,
    }


def parse_pnl_progression(entries: list[dict]) -> dict:
    """Extract day P&L observations from Risk Manager sync events."""
    pnls: list[tuple[str, float, float]] = []  # (ts, pnl, balance)
    for e in entries:
        m = re.search(
            r"Risk Manager synced with Schwab: P&L=\$(-?[0-9.]+).*Balance=\$([0-9.]+)",
            e.get("message", ""),
        )
        if m:
            pnls.append((e.get("timestamp", ""), float(m.group(1)), float(m.group(2))))
    if not pnls:
        return {}
    return {
        "first": pnls[0],
        "last": pnls[-1],
        "peak_pnl": max(pnls, key=lambda x: x[1]),
        "trough_pnl": min(pnls, key=lambda x: x[1]),
        "samples": len(pnls),
    }


def parse_stream_health(entries: list[dict]) -> dict:
    """Recent watchdog heartbeats + tick rate."""
    heartbeats: list[tuple[str, int, bool]] = []  # (ts, age, rth)
    tick_counts: list[tuple[str, int]] = []
    for e in entries:
        msg = e.get("message", "")
        hm = re.search(r"watchdog: heartbeat age=(\d+)s.*rth=(\w+)", msg)
        if hm:
            heartbeats.append((
                e.get("timestamp", ""),
                int(hm.group(1)),
                hm.group(2) == "True",
            ))
        tm = re.search(r"schwab_stream: (\d+) ticks total", msg)
        if tm:
            tick_counts.append((e.get("timestamp", ""), int(tm.group(1))))
    return {
        "latest_heartbeat": heartbeats[-1] if heartbeats else None,
        "tick_count": tick_counts[-1][1] if tick_counts else None,
        "rth": heartbeats[-1][2] if heartbeats else None,
    }


# ────────────────────────────────────────────────────────────────────
# Report formatting — markdown optimized for Claude consumption
# ────────────────────────────────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h {m}m"


def _fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"${x:,.2f}" if x >= 0 else f"-${abs(x):,.2f}"


def build_report(window_seconds: int = 3600) -> str:
    """Assemble all sources into a structured Markdown report."""
    now_local = datetime.now()
    proc = get_bot_process()
    ttl = get_token_ttl_hours()
    state = get_state()
    status = fetch_api("/api/status")
    positions = fetch_api("/api/positions/db")
    indices = fetch_api("/api/market-indices")

    entries = tail_log(seconds_back=window_seconds)
    activity = parse_signal_activity(entries)
    pnl = parse_pnl_progression(entries)
    stream = parse_stream_health(entries)

    lines: list[str] = []
    lines.append(f"# Bot Monitor — {now_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append("")
    lines.append(f"*Window: last {window_seconds // 60} min. Pure-observer agent, never writes to bot state.*")
    lines.append("")

    # ── Process health ──────────────────────────────────────────────
    lines.append("## Process health")
    lines.append("")
    if proc:
        lines.append(f"- **Bot:** running PID {proc['pid']} ({proc['uptime_human']} uptime)")
    else:
        lines.append("- **Bot:** ⛔ NOT RUNNING")
    if ttl is not None:
        if ttl < 12:
            lines.append(f"- **Schwab token:** ⚠️ {ttl:.1f}h remaining (re-auth soon)")
        elif ttl < 0:
            lines.append(f"- **Schwab token:** ⛔ EXPIRED")
        else:
            lines.append(f"- **Schwab token:** {ttl:.1f}h remaining")
    else:
        lines.append("- **Schwab token:** unknown")
    if stream.get("latest_heartbeat"):
        hb_ts, hb_age, hb_rth = stream["latest_heartbeat"]
        rth_label = "RTH" if hb_rth else "off-hours"
        lines.append(f"- **Stream:** heartbeat {hb_age}s ago ({rth_label}, {stream.get('tick_count', '?'):,} ticks)")
    if status:
        lines.append(f"- **Mode:** {status.get('mode', '?')}")
    lines.append("")

    # ── P&L ──────────────────────────────────────────────────────────
    lines.append("## P&L")
    lines.append("")
    if pnl:
        first_ts, first_pnl, first_bal = pnl["first"]
        last_ts, last_pnl, last_bal = pnl["last"]
        peak_ts, peak_pnl, _ = pnl["peak_pnl"]
        trough_ts, trough_pnl, _ = pnl["trough_pnl"]
        lines.append(f"- **Current:** {_fmt_money(last_pnl)} (balance {_fmt_money(last_bal)})")
        lines.append(f"- **First in window:** {_fmt_money(first_pnl)}")
        lines.append(f"- **Peak:** {_fmt_money(peak_pnl)} at {peak_ts.split()[1] if ' ' in peak_ts else peak_ts}")
        lines.append(f"- **Trough:** {_fmt_money(trough_pnl)} at {trough_ts.split()[1] if ' ' in trough_ts else trough_ts}")
        # Distance to daily-loss circuit
        # max_daily_loss=0.01 of balance → ~$330 currently
        if last_bal > 0:
            circuit_at = last_bal * -0.01
            lines.append(f"- **Distance to circuit:** {_fmt_money(last_pnl - circuit_at)} until -1% cap")
    else:
        lines.append("- (no Schwab sync events in window)")
    lines.append("")

    # ── Positions ───────────────────────────────────────────────────
    lines.append("## Positions")
    lines.append("")
    positions_data = state.get("positions_data", {})
    if positions_data:
        lines.append("| Symbol | Qty | Entry | Source |")
        lines.append("|---|---|---|---|")
        for sym, p in positions_data.items():
            qty = p.get("quantity", "?")
            entry = p.get("entry_price", "?")
            etime = p.get("entry_time", "")
            # Heuristic: round entry_time (xxx.xx) often = bot fill;
            # off-the-second entry_time = external. Not authoritative.
            lines.append(f"| {sym} | {qty} | ${entry} | external/pre-existing |")
    else:
        lines.append("- (no open positions)")
    lines.append("")

    # ── Today's signal activity ─────────────────────────────────────
    lines.append("## Signal activity in window")
    lines.append("")
    lines.append(f"- **Fired:** {sum(activity['fired_by_strategy'].values())}")
    if activity["fired_by_strategy"]:
        for s, n in sorted(activity["fired_by_strategy"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {s}: {n}")
    lines.append(f"- **Accepted by router:** {sum(activity['accepted_by_strategy'].values())}")
    if activity["accepted_by_strategy"]:
        for s, n in sorted(activity["accepted_by_strategy"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  - {s}: {n}")
    if activity["block_reasons"]:
        lines.append(f"- **Blocked at router:** {sum(activity['block_reasons'].values())}")
        for r, n in sorted(activity["block_reasons"].items(), key=lambda kv: -kv[1])[:10]:
            lines.append(f"  - {r}: {n}")
    lines.append(f"- **Orders placed:** {len(activity['fills'])}")
    for f in activity["fills"][-5:]:
        lines.append(f"  - {f['ts'].split()[1] if ' ' in f['ts'] else f['ts']}: {f['raw']}")
    lines.append("")

    # ── Direction gates (signal of how the new wiring is performing)
    if activity["direction_gate_skips"]:
        lines.append("## Direction-gate skips")
        lines.append("")
        lines.append(f"Total: **{len(activity['direction_gate_skips'])}** signals blocked by direction reader.")
        lines.append("")
        lines.append("| Time | Symbol | Phase | Direction |")
        lines.append("|---|---|---|---|")
        for s in activity["direction_gate_skips"][-10:]:
            ts_only = s["ts"].split()[1] if " " in s["ts"] else s["ts"]
            lines.append(f"| {ts_only} | {s['symbol']} | {s['phase']} | {s['direction']:+.1f} |")
        lines.append("")

    # ── R:R audit warnings (critical — any indicates a bug) ─────────
    if activity["rr_audit_low"]:
        lines.append("## ⚠️ R:R audit warnings")
        lines.append("")
        lines.append("**Any entry here indicates a strategy producing tight-target signals — investigate.**")
        lines.append("")
        for w in activity["rr_audit_low"][-10:]:
            ts_only = w["ts"].split()[1] if " " in w["ts"] else w["ts"]
            lines.append(f"- {ts_only} {w['symbol']}: R:R {w['rr']:.2f} (floor 1.80)")
        lines.append("")

    # ── Errors ──────────────────────────────────────────────────────
    if activity["errors"]:
        lines.append("## Errors")
        lines.append("")
        for err in activity["errors"][-10:]:
            ts_only = err["ts"].split()[1] if " " in err["ts"] else err["ts"]
            lines.append(f"- {ts_only}: {err['msg']}")
        lines.append("")

    # ── Shadow ledgers ──────────────────────────────────────────────
    # v-monitor-shadows-2026-06-10: the two evidence-collection shadows
    # (regime allocator, SHORT shadow) write NDJSON ledgers. Surface
    # today's row counts + last entry so the morning soak review is a
    # single file read. Pure read — same observer contract.
    lines.append("## Shadow ledgers")
    lines.append("")
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for label, fname in (("Regime allocator", "regime_allocator_shadow.ndjson"),
                         ("SHORT shadow", "shadow_short_log.ndjson")):
        path = REPO / fname
        if not path.exists():
            lines.append(f"- **{label}:** no ledger yet")
            continue
        rows_today = 0
        last_entry = None
        try:
            with open(path) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    last_entry = row
                    if str(row.get("timestamp", "")).startswith(today_str):
                        rows_today += 1
        except OSError as exc:
            lines.append(f"- **{label}:** unreadable ({exc})")
            continue
        if last_entry is None:
            lines.append(f"- **{label}:** empty")
        else:
            sym = last_entry.get("symbol", "?")
            ts = str(last_entry.get("timestamp", ""))[:16]
            extra = (f"tape={last_entry.get('tape')}"
                     if "tape" in last_entry
                     else f"rsi={last_entry.get('rsi', '?')}")
            lines.append(
                f"- **{label}:** {rows_today} rows today (UTC) — "
                f"last: {sym} at {ts} ({extra})")
    lines.append("")

    # ── Market regime ───────────────────────────────────────────────
    if indices and indices.get("indices"):
        lines.append("## Market regime")
        lines.append("")
        lines.append("| Index | Last | Change |")
        lines.append("|---|---|---|")
        for ix in indices["indices"]:
            arrow = "↗" if ix["change_pct"] >= 0 else "↘"
            lines.append(f"| {ix['name']} | {ix['last']:,.2f} | {arrow} {ix['change_pct']:+.2f}% |")
        lines.append("")

    # ── Footer ──────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append(f"*Generated by tools/bot_monitor.py. Window {window_seconds // 60} min. Next report in ~60s.*")

    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────

def write_report(content: str, also_history: bool = True) -> None:
    """Write to docs/monitor/latest.md and an optional timestamped copy."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_REPORT.write_text(content)
    if also_history:
        now = datetime.now()
        day_dir = REPORT_DIR / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        ts_file = day_dir / now.strftime("%H-%M.md")
        ts_file.write_text(content)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=60,
                        help="Poll interval in seconds (default 60).")
    parser.add_argument("--window", type=int, default=3600,
                        help="Log window in seconds (default 3600 = 1h).")
    parser.add_argument("--once", action="store_true",
                        help="Generate one report and exit.")
    parser.add_argument("--quiet", action="store_true",
                        help="Don't print the report to stdout.")
    parser.add_argument("--no-history", action="store_true",
                        help="Don't keep timestamped history files.")
    args = parser.parse_args()

    while True:
        try:
            report = build_report(window_seconds=args.window)
            write_report(report, also_history=not args.no_history)
            if not args.quiet:
                print(report)
                print()  # blank line between reports
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] report → {LATEST_REPORT}")
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            print(f"[monitor] error: {exc}", file=sys.stderr)

        if args.once:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    sys.exit(main())
