#!/usr/bin/env python3
"""Backfill shadow_long_log.ndjson from existing audit logs.

v-shadow-long-ledger-2026-09-17: best-effort backfill script.

Attempts to reconstruct shadow_long_log.ndjson from:
  1. bot_decisions.ndjson (structured decision log)
  2. Engine audit logs (JSON-per-line)

LIMITATIONS (fields may be incomplete):
  - Historical audit logs may not include all resolver fields
  - bb_lower, bb_middle, sma_50, macd, macd_signal often missing
  - falling_knife_pass defaults to True for historical entries
  - market_context_regime may be null

Usage:
    python tools/backfill_shadow_long.py \\
        --input data/bot_decisions.ndjson \\
        --output data/shadow_long_log.ndjson \\
        [--audit-dir logs/]

The script appends to the output file (does not overwrite).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("backfill_shadow_long")

HANDS_OFF_SYMBOLS = frozenset({"MU", "HQGE", "SPCX"})


def parse_bot_decision_line(line: str) -> Optional[dict]:
    """Parse a bot_decisions.ndjson line and extract LONG mean-rev entries."""
    try:
        d = json.loads(line.strip())
    except json.JSONDecodeError:
        return None

    if d.get("signal_type") not in ("BUY", "buy"):
        return None

    strategy = d.get("strategy") or d.get("reasoning", {}).get("strategy")
    if strategy not in ("mean_reversion", "oversold_v2"):
        return None

    symbol = d.get("symbol", "").upper()
    if not symbol or symbol in HANDS_OFF_SYMBOLS:
        return None

    reasoning = d.get("reasoning", {})
    entry = {
        "timestamp": d.get("timestamp") or datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "signal_type": "LONG",
        "reason": reasoning.get("entry_pattern", "oversold_bounce"),
        "signal_close": float(d.get("entry_price", 0)),
        "entry_price": float(d.get("entry_price", 0)),
        "rsi": float(reasoning.get("rsi", 0)),
        "bb_lower": float(reasoning.get("bb_lower", 0)),
        "bb_middle": float(reasoning.get("bb_middle", 0)),
        "sma_50": float(reasoning.get("sma_50", 0)),
        "macd": float(reasoning.get("macd", 0)),
        "macd_signal": float(reasoning.get("macd_signal", 0)),
        "atr": float(reasoning.get("atr", 0)),
        "hypothetical_stop": float(d.get("stop_loss", 0)),
        "hypothetical_target": float(d.get("take_profit", 0)),
        "rr_ratio": 2.0,
        "stop_dist": float(reasoning.get("stop_distance", 0)),
        "market_context_regime": reasoning.get("market_context_regime"),
        "falling_knife_pass": True,
        "_backfill_source": "bot_decisions",
    }

    if entry["stop_dist"] == 0 and entry["hypothetical_stop"] > 0 and entry["entry_price"] > 0:
        entry["stop_dist"] = abs(entry["entry_price"] - entry["hypothetical_stop"])

    return entry


def parse_audit_log_line(line: str) -> Optional[dict]:
    """Parse an audit log line and extract LONG mean-rev entries."""
    try:
        d = json.loads(line.strip())
    except json.JSONDecodeError:
        return None

    component = d.get("component")
    action = d.get("action")

    if component == "mean_rev_shadow_ledger" and action == "stage_a_entry":
        setup_type = d.get("setup_type", "")
        if "buy" not in setup_type.lower() and "long" not in setup_type.lower():
            return None

        symbol = d.get("symbol", "").upper()
        if not symbol or symbol in HANDS_OFF_SYMBOLS:
            return None

        entry = {
            "timestamp": d.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "signal_type": "LONG",
            "reason": d.get("entry_pattern", "oversold_bounce"),
            "signal_close": float(d.get("entry_price", 0)),
            "entry_price": float(d.get("entry_price", 0)),
            "rsi": float(d.get("rsi_14", 0)),
            "bb_lower": 0.0,
            "bb_middle": 0.0,
            "sma_50": 0.0,
            "macd": 0.0,
            "macd_signal": 0.0,
            "atr": float(d.get("atr", 0)),
            "hypothetical_stop": float(d.get("stop_loss", 0)),
            "hypothetical_target": float(d.get("take_profit", 0)),
            "rr_ratio": float(d.get("rr_ratio", 2.0)),
            "stop_dist": float(d.get("stop_dist", 0)),
            "market_context_regime": d.get("regime"),
            "falling_knife_pass": True,
            "_backfill_source": "audit_log",
            "_incomplete_fields": ["bb_lower", "bb_middle", "sma_50", "macd", "macd_signal"],
        }
        return entry

    if component in ("engine_decision", "strategy_decision"):
        if action not in ("signal_buy", "shadow"):
            return None

        symbol = d.get("symbol", "").upper()
        if not symbol or symbol in HANDS_OFF_SYMBOLS:
            return None

        entry_pattern = d.get("entry_pattern") or d.get("reason", "")
        if entry_pattern not in ("oversold_bounce", "uptrend_pullback", "long_shadow_logged"):
            return None

        entry = {
            "timestamp": d.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "signal_type": "LONG",
            "reason": entry_pattern if entry_pattern != "long_shadow_logged" else "oversold_bounce",
            "signal_close": float(d.get("close", 0)),
            "entry_price": float(d.get("close", 0)),
            "rsi": float(d.get("rsi", 0)),
            "bb_lower": float(d.get("bb_lower", 0)),
            "bb_middle": 0.0,
            "sma_50": float(d.get("sma_50", 0)),
            "macd": float(d.get("macd", 0)),
            "macd_signal": 0.0,
            "atr": float(d.get("atr", 0)),
            "hypothetical_stop": float(d.get("stop", d.get("hyp_stop", 0))),
            "hypothetical_target": float(d.get("target", d.get("hyp_target", 0))),
            "rr_ratio": 2.0,
            "stop_dist": float(d.get("stop_dist", 0)),
            "market_context_regime": d.get("market_context_regime"),
            "falling_knife_pass": d.get("falling_knife_pass", True),
            "_backfill_source": "audit_log",
        }

        if entry["stop_dist"] == 0 and entry["hypothetical_stop"] > 0 and entry["entry_price"] > 0:
            entry["stop_dist"] = abs(entry["entry_price"] - entry["hypothetical_stop"])

        return entry

    return None


def backfill_from_file(input_path: Path, output_path: Path, parser_fn) -> int:
    """Read input file, extract entries, append to output."""
    if not input_path.exists():
        logger.warning("Input file not found: %s", input_path)
        return 0

    count = 0
    seen = set()
    with open(input_path, "r") as fin, open(output_path, "a") as fout:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue
            entry = parser_fn(line)
            if entry is None:
                continue
            key = (entry["symbol"], entry["timestamp"][:16])
            if key in seen:
                continue
            seen.add(key)
            fout.write(json.dumps(entry, default=str) + "\n")
            count += 1
            if count % 100 == 0:
                logger.info("Processed %d entries from %s", count, input_path.name)

    logger.info("Appended %d entries from %s to %s", count, input_path, output_path)
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Path to bot_decisions.ndjson or similar decision log",
    )
    parser.add_argument(
        "--audit-dir", "-a",
        type=Path,
        default=None,
        help="Directory containing audit logs (*.log, *.ndjson)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/shadow_long_log.ndjson"),
        help="Output path (default: data/shadow_long_log.ndjson)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and count entries without writing",
    )
    args = parser.parse_args(argv)

    if args.input is None and args.audit_dir is None:
        logger.error("Provide --input or --audit-dir (or both)")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0

    if args.input is not None and args.input.exists():
        if args.dry_run:
            logger.info("[DRY RUN] Would process %s", args.input)
        else:
            total += backfill_from_file(args.input, args.output, parse_bot_decision_line)

    if args.audit_dir is not None and args.audit_dir.exists():
        for log_file in sorted(args.audit_dir.glob("*.log")):
            if args.dry_run:
                logger.info("[DRY RUN] Would process %s", log_file)
            else:
                total += backfill_from_file(log_file, args.output, parse_audit_log_line)
        for log_file in sorted(args.audit_dir.glob("*.ndjson")):
            if args.dry_run:
                logger.info("[DRY RUN] Would process %s", log_file)
            else:
                total += backfill_from_file(log_file, args.output, parse_audit_log_line)

    logger.info("Total backfilled: %d entries", total)
    if total > 0:
        logger.info("Output written to: %s", args.output)
        logger.warning(
            "NOTE: Backfilled entries may have incomplete fields (bb_lower, "
            "bb_middle, sma_50, macd, macd_signal). The resolver handles "
            "missing values gracefully but Stage A metrics may be less accurate."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
