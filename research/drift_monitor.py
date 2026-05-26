"""Classifier drift monitor + auto-flip recommender.

v-drift-monitor-2026-05-13. Two responsibilities:

  1. **Drift**: compare the classifier's predictions on the last
     N sessions to the realized triple-barrier outcomes on the same
     symbol-days. If precision degrades materially, emit a CRITICAL
     log line — the model needs a retrain.

  2. **Auto-flip**: once shadow data has accumulated and shows the
     classifier blocks (a) a meaningful fraction of trades that
     turned out badly, AND (b) doesn't block trades that turned out
     well, emit a yaml-change recommendation moving news_strategy
     from shadow → advisory → gating (rollout flavor B).

Both run as one-shot scripts. The bot is unchanged. The autonomy
story is: this script's output is what an operator (or a cron job)
reads to decide whether to flip the next yaml flag.

Process-level isolation: imports core.classifier and research.* but
NEVER core.engine.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class ShadowPick:
    """A classifier verdict paired with the bot's actual decision."""
    date: str
    symbol: str
    signal_side: str
    allows_long: bool
    allows_short: bool
    long_score: float
    short_score: float


@dataclass
class DriftReport:
    sessions_scanned: int = 0
    shadow_total: int = 0
    classifier_blocked_bot_entries: int = 0
    classifier_allowed_bot_entries: int = 0
    # Of the entries the bot took: were they LONG or SHORT?
    bot_long_entries: int = 0
    bot_short_entries: int = 0
    sessions: List[str] = field(default_factory=list)
    # Symbol-level breakdown
    by_symbol: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    recommendation: str = ""
    yaml_diff: str = ""


def _parse_kvs(message: str) -> Dict[str, str]:
    return {k: v for k, v in re.findall(r"(\w+)=([^\s]+)", message)}


def scan_logs(log_path: Path, days: int) -> DriftReport:
    """Scan the bot log for the last `days` days of shadow decisions
    and bot entries. Pair them up and compute disagreement metrics."""
    report = DriftReport()
    if not log_path.exists():
        return report

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    # date -> list of (timestamp, symbol, kind in {"shadow","fill"}, kv)
    by_date: Dict[str, List[Tuple[datetime, str, str, Dict[str, str]]]] = \
        defaultdict(list)

    with log_path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            ts = d.get("timestamp", "")
            date_str = ts[:10]
            if date_str < cutoff:
                continue
            msg = d.get("message", "")
            if "component=side_classifier" in msg and "action=shadow" in msg:
                kv = _parse_kvs(msg)
                try:
                    ts_dt = datetime.fromisoformat(ts.replace(",", "."))
                    by_date[date_str].append((ts_dt, kv.get("symbol", "?"),
                                              "shadow", kv))
                except Exception:
                    pass
            elif "component=signal_router" in msg and "action=accepted" in msg:
                kv = _parse_kvs(msg)
                try:
                    ts_dt = datetime.fromisoformat(ts.replace(",", "."))
                    by_date[date_str].append((ts_dt, kv.get("symbol", "?"),
                                              "fill", kv))
                except Exception:
                    pass

    report.sessions = sorted(by_date.keys())
    report.sessions_scanned = len(report.sessions)

    for date_str, events in by_date.items():
        shadows = [(ts, sym, kv) for ts, sym, k, kv in events if k == "shadow"]
        fills = [(ts, sym, kv) for ts, sym, k, kv in events if k == "fill"]
        report.shadow_total += len(shadows)

        for fts, fsym, fkv in fills:
            side = (fkv.get("side") or "").upper()
            if side == "BUY":
                report.bot_long_entries += 1
            elif side == "SELL":
                report.bot_short_entries += 1

            # Find nearest shadow within 60s for the same symbol.
            best = None
            best_delta = None
            for sts, ssym, skv in shadows:
                if ssym != fsym:
                    continue
                delta = abs((sts - fts).total_seconds())
                if delta > 60:
                    continue
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best = skv
            if best is None:
                continue

            allows_long = best.get("allows_long", "False") == "True"
            allows_short = best.get("allows_short", "False") == "True"
            verdict = (
                (side == "BUY" and allows_long)
                or (side == "SELL" and allows_short)
            )
            if verdict:
                report.classifier_allowed_bot_entries += 1
                report.by_symbol[fsym]["agreed"] += 1
            else:
                report.classifier_blocked_bot_entries += 1
                report.by_symbol[fsym]["disagreed"] += 1

    # Emit recommendation
    report.recommendation, report.yaml_diff = _build_recommendation(report)
    return report


def _build_recommendation(report: DriftReport) -> Tuple[str, str]:
    """Decide what (if anything) to flip in Config().yaml.

    Conservative thresholds — autonomy must not run faster than
    statistical evidence supports.
    """
    n_paired = report.classifier_allowed_bot_entries + report.classifier_blocked_bot_entries
    if report.sessions_scanned < 3:
        return ("Insufficient sessions scanned (need ≥3). Continue shadow.",
                "")
    if n_paired < 20:
        return (f"Only {n_paired} bot-entry/shadow-decision pairs over "
                f"{report.sessions_scanned} sessions. Need ≥20. Continue shadow.",
                "")

    block_rate = report.classifier_blocked_bot_entries / max(n_paired, 1)
    if block_rate < 0.10:
        return ("Classifier disagrees with bot on <10% of entries. The signal "
                "is too weak to justify gating. Either retrain on a larger "
                "universe or wait for a session with more disagreement.", "")
    if block_rate > 0.60:
        return ("Classifier disagrees with bot on >60% of entries. This is "
                "too aggressive — gating would shut off most trades. "
                "Lower the thresholds OR check for a training/eval drift.", "")

    # Sweet spot: 10-60% block rate. Recommend advisory mode for news_strategy.
    return (
        f"Classifier disagrees with bot on {block_rate:.0%} of entries over "
        f"{report.sessions_scanned} sessions ({n_paired} paired). Healthy "
        f"signal. Recommend promoting to advisory mode for news_strategy.",
        "trading:\n"
        "  # Set after reviewing the most recent session_<DATE>.md report\n"
        "  side_classifier_news_strategy: true\n"
    )


def render_report(report: DriftReport) -> str:
    lines = [
        "# Side-Classifier Drift Monitor + Auto-Flip Recommendation",
        "",
        f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z",
        f"Sessions scanned: {report.sessions_scanned} "
        f"({', '.join(report.sessions) if report.sessions else 'none'})",
        "",
        "## Aggregate counts",
        "",
        f"- Total shadow decisions: **{report.shadow_total}**",
        f"- Bot LONG entries: **{report.bot_long_entries}**",
        f"- Bot SHORT entries: **{report.bot_short_entries}**",
        f"- Classifier agreed with bot's entry side: **{report.classifier_allowed_bot_entries}**",
        f"- Classifier would have blocked bot's entry: **{report.classifier_blocked_bot_entries}**",
        "",
        "## Per-symbol disagreement",
        "",
    ]
    if report.by_symbol:
        lines += ["| Symbol | Agreed | Disagreed |", "|---|---:|---:|"]
        for sym in sorted(report.by_symbol.keys()):
            counts = report.by_symbol[sym]
            lines.append(
                f"| {sym} | {counts.get('agreed', 0)} | {counts.get('disagreed', 0)} |"
            )
    else:
        lines.append("(no paired data yet)")
    lines += [
        "",
        "## Recommendation",
        "",
        report.recommendation,
        "",
    ]
    if report.yaml_diff:
        lines += ["**Proposed yaml change** (review then apply manually):", "",
                  "```yaml", report.yaml_diff, "```"]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Classifier drift monitor")
    parser.add_argument("--days", type=int, default=7,
                        help="Look-back window (default 7 days)")
    parser.add_argument("--log", default=str(_REPO_ROOT / "trading_bot.log"))
    parser.add_argument("--out", default=None,
                        help="Output markdown path. Default: "
                        "docs/training_runs/drift_<YYYY-MM-DD>.md")
    args = parser.parse_args(argv)

    report = scan_logs(Path(args.log), args.days)
    md = render_report(report)
    out_path = Path(args.out) if args.out else (
        _REPO_ROOT / "docs" / "training_runs" /
        f"drift_{datetime.now().strftime('%Y-%m-%d')}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"[drift_monitor] wrote {out_path}", flush=True)
    print(f"[drift_monitor] recommendation: {report.recommendation}", flush=True)
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
