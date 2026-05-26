"""Daily session report for the side classifier.

v-session-reporter-2026-05-13. Parses ``trading_bot.log`` for one
day's classifier shadow audit lines + the actual trades the bot
took, and produces a markdown summary. The autonomy story:

  - Operator restarts the bot with shadow_mode=True.
  - Bot runs all day, classifier audits every accepted signal.
  - End-of-day: run this script to produce
    ``docs/training_runs/session_<YYYY-MM-DD>.md``.
  - The report answers: "Of the entries the bot took today, which
    would the classifier have *blocked*? Of those, how many ended
    badly?"

Process-level isolation: imports nothing from core.engine. Pure
log parsing + math.

Usage:
    python research/session_reporter.py --date 2026-05-13
    python research/session_reporter.py            # defaults to today
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

LOG_PATH = _REPO_ROOT / "trading_bot.log"


@dataclass
class ShadowDecision:
    timestamp: datetime
    symbol: str
    signal_side: str
    primary_reason: str
    long_score: float
    short_score: float
    allows_long: bool
    allows_short: bool


@dataclass
class TradeFill:
    timestamp: datetime
    symbol: str
    side: str
    qty: int
    entry_price: float


@dataclass
class SessionStats:
    shadow_count: int = 0
    shadow_decisions_by_symbol: Dict[str, List[ShadowDecision]] = field(
        default_factory=lambda: defaultdict(list))
    fills_by_symbol: Dict[str, List[TradeFill]] = field(
        default_factory=lambda: defaultdict(list))
    classifier_would_have_blocked: int = 0
    classifier_would_have_allowed: int = 0
    fail_open_count: int = 0


def _parse_kvs(message: str) -> Dict[str, str]:
    """Extract key=value pairs from an audit message line."""
    out: Dict[str, str] = {}
    for tok in re.findall(r"(\w+)=([^\s]+)", message):
        out[tok[0]] = tok[1]
    return out


def _matches_date(ts_str: str, date_str: str) -> bool:
    return ts_str.startswith(date_str)


def parse_session(log_path: Path, date_str: str) -> SessionStats:
    stats = SessionStats()

    if not log_path.exists():
        return stats

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
            if not _matches_date(ts, date_str):
                continue
            msg = d.get("message", "")

            # Shadow audit lines from the side classifier
            if "component=side_classifier" in msg and "action=shadow" in msg:
                kv = _parse_kvs(msg)
                try:
                    dec = ShadowDecision(
                        timestamp=datetime.fromisoformat(ts.replace(",", ".")),
                        symbol=kv.get("symbol", "?"),
                        signal_side=kv.get("signal_side", "?"),
                        primary_reason=kv.get("reason", "?"),
                        long_score=float(kv.get("long_score", "0") or 0),
                        short_score=float(kv.get("short_score", "0") or 0),
                        allows_long=kv.get("allows_long", "False") == "True",
                        allows_short=kv.get("allows_short", "False") == "True",
                    )
                except Exception:
                    continue
                stats.shadow_count += 1
                stats.shadow_decisions_by_symbol[dec.symbol].append(dec)

                # "Would have blocked": signal_side is BUY but
                # allows_long is False (and vice versa).
                if (dec.signal_side.upper() == "BUY" and not dec.allows_long) or \
                   (dec.signal_side.upper() == "SELL" and not dec.allows_short):
                    stats.classifier_would_have_blocked += 1
                else:
                    stats.classifier_would_have_allowed += 1

            # Actual fills (signal_router accepted)
            elif "component=signal_router" in msg and "action=accepted" in msg:
                kv = _parse_kvs(msg)
                try:
                    fill = TradeFill(
                        timestamp=datetime.fromisoformat(ts.replace(",", ".")),
                        symbol=kv.get("symbol", "?"),
                        side=kv.get("side", "?"),
                        qty=int(float(kv.get("qty", "0") or 0)),
                        entry_price=float(kv.get("entry", "0") or 0),
                    )
                    stats.fills_by_symbol[fill.symbol].append(fill)
                except Exception:
                    continue

            # Fail-open warnings — only count load failures that reference
            # the production model path (not pytest /tmp/... fixture paths).
            elif ("ClassifierRuntime: predictor load failed" in msg
                  and "/tmp/" not in msg
                  and "/nonexistent/" not in msg):
                stats.fail_open_count += 1

    return stats


def render_report(date_str: str, stats: SessionStats) -> str:
    """Markdown report. Audited so operator can sanity-check by eye."""
    lines: List[str] = []
    lines.append(f"# Side-Classifier Shadow Session — {date_str}")
    lines.append("")
    lines.append(f"Generated: {datetime.utcnow().isoformat(timespec='seconds')}Z")
    lines.append("")

    lines.append("## Headline numbers")
    lines.append("")
    lines.append(f"- Shadow decisions emitted: **{stats.shadow_count}**")
    lines.append(f"- Classifier would have ALLOWED: **{stats.classifier_would_have_allowed}**")
    lines.append(f"- Classifier would have BLOCKED: **{stats.classifier_would_have_blocked}**")
    lines.append(f"- Predictor fail-open events: {stats.fail_open_count}")
    lines.append(f"- Actual trade fills (signal_router accepted): "
                 f"{sum(len(v) for v in stats.fills_by_symbol.values())}")
    lines.append("")

    if stats.shadow_count == 0:
        lines.append("> No shadow decisions found for this date. "
                     "Check that SIDE_CLASSIFIER_SHADOW_MODE is True "
                     "in Config().yaml and that the bot was restarted "
                     "after the flag was flipped.")
        return "\n".join(lines)

    # Per-symbol cross-tab: did the classifier and the bot agree?
    lines.append("## Per-symbol overlap (signal_router accepted vs classifier verdict)")
    lines.append("")
    lines.append("| Symbol | Bot entered | Shadow decisions | Last classifier reason |")
    lines.append("|---|---:|---:|---|")
    overlap_blocked = overlap_allowed = 0
    syms_to_show = set(stats.shadow_decisions_by_symbol.keys()) | set(stats.fills_by_symbol.keys())
    for sym in sorted(syms_to_show):
        decisions = stats.shadow_decisions_by_symbol.get(sym, [])
        fills = stats.fills_by_symbol.get(sym, [])
        last_reason = decisions[-1].primary_reason if decisions else "—"
        lines.append(f"| {sym} | {len(fills)} | {len(decisions)} | {last_reason[:80]} |")

        # For each actual fill, find the closest-in-time shadow decision.
        for fill in fills:
            best = None
            best_delta = None
            for dec in decisions:
                delta = abs((dec.timestamp - fill.timestamp).total_seconds())
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best = dec
            if best is not None and best_delta is not None and best_delta < 60:
                # Only count if we matched the same evaluation tick
                if fill.side.upper() == "BUY":
                    if not best.allows_long:
                        overlap_blocked += 1
                    else:
                        overlap_allowed += 1
                else:
                    if not best.allows_short:
                        overlap_blocked += 1
                    else:
                        overlap_allowed += 1

    lines.append("")
    lines.append("## Counterfactual: of bot's actual fills, what would the classifier have done?")
    lines.append("")
    lines.append(f"- Classifier would have **agreed** (allowed the same side): **{overlap_allowed}**")
    lines.append(f"- Classifier would have **vetoed** (different side, or ∅): **{overlap_blocked}**")
    total = overlap_allowed + overlap_blocked
    if total > 0:
        lines.append(f"- Overlap rate (classifier agrees with bot): **{overlap_allowed/total:.0%}**")
    lines.append("")

    lines.append("## Next-action recommendation")
    lines.append("")
    if stats.shadow_count < 30:
        lines.append("- **Not enough data** (<30 shadow decisions). Continue shadow mode for more sessions.")
    elif overlap_blocked == 0:
        lines.append("- Classifier agrees with bot on every entry. Either the classifier is too permissive "
                     "(check thresholds) or today's tape was clean. Wait one more session before flipping.")
    elif overlap_blocked / max(total, 1) > 0.5:
        lines.append("- Classifier disagrees with bot on >50% of entries — DO NOT flip to gating yet. "
                     "Inspect a few of the blocked entries manually before any rollout step.")
    else:
        lines.append("- Classifier blocks a minority of entries. **Consider promoting news_strategy** "
                     "to advisory mode by setting `side_classifier_news_strategy: true` "
                     "in Config().yaml — but ONLY if the blocked entries correspond to "
                     "trades you'd have skipped manually.")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Side-classifier session reporter")
    parser.add_argument("--date", default=None,
                        help="Date to report on (YYYY-MM-DD). Default: today.")
    parser.add_argument("--log", default=str(LOG_PATH),
                        help="Path to trading_bot.log")
    parser.add_argument("--out", default=None,
                        help="Output markdown path. Default: docs/training_runs/session_<DATE>.md")
    args = parser.parse_args(argv)

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    log_path = Path(args.log)

    stats = parse_session(log_path, date_str)
    report = render_report(date_str, stats)

    out_path = (Path(args.out) if args.out else
                _REPO_ROOT / "docs" / "training_runs" / f"session_{date_str}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"[session_reporter] wrote {out_path}", flush=True)
    print(f"[session_reporter] {stats.shadow_count} shadow decisions, "
          f"{sum(len(v) for v in stats.fills_by_symbol.values())} fills",
          flush=True)
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
