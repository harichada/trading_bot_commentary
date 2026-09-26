"""Diagnostic: symbols whose split-adjusted daily history is still implausible after the
cache-trust + persistence fix in `research.strategy_hunt.data.splits` (Batch-2 Task E, Ruling 4).

Two independent checks flag a symbol:
  * adjusted daily close min/max ratio > MAX_MINMAX_RATIO -- the signature of a bad split
    multiplier compounding into the back-adjusted history (e.g. LCID divided by 1e8 before the
    fix).
  * |rs_120| > MAX_ABS_RS_120 anywhere in <root>/context/universe_daily.parquet -- the
    120-session relative-strength column a bad multiplier corrupts downstream, which is what the
    rank-based families (rs_leader, rs_rotation, mom_12_1) select on.

Intended as a one-off check after `context` has rebuilt universe_daily.parquet, e.g.:
    $PY -c "from pathlib import Path; from research.strategy_hunt.splits_audit import audit_splits
             print(audit_splits(Path('data/strategy_hunt')))"
`run_splits_audit` is the thin CLI entry point (kept here rather than in `cli.py` per the task
brief); see the Task E report for why the `cli.py` subparser registration itself is deferred.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MAX_MINMAX_RATIO = 1e4
MAX_ABS_RS_120 = 500.0
CONTEXT_SYMBOLS = ("SPY", "VIX")


def _symbols_arg(value: str | None, root: Path) -> list[str]:
    """Mirrors `cli._symbols_arg` (kept independent to avoid a splits_audit -> cli import)."""
    root = Path(root)
    if not value:
        return sorted(p.stem for p in (root / "bars5").glob("*.parquet") if p.stem not in CONTEXT_SYMBOLS)
    if value.startswith("@"):
        return [s.strip().upper() for s in Path(value[1:]).read_text().splitlines() if s.strip()]
    return [s.strip().upper() for s in value.split(",") if s.strip()]


def _minmax_ratio(root: Path, symbol: str) -> float | None:
    from research.strategy_hunt.data.splits import load_adjusted

    _, daily = load_adjusted(root, symbol)
    if daily.empty:
        return None
    lo, hi = float(daily["close"].min()), float(daily["close"].max())
    if lo <= 0:
        return float("inf")
    return hi / lo


def _rs_120_abs_max(root: Path) -> dict[str, float]:
    path = Path(root) / "context" / "universe_daily.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path, columns=["symbol", "rs_120"]).dropna(subset=["rs_120"])
    if df.empty:
        return {}
    return {sym: float(vals.abs().max()) for sym, vals in df.groupby("symbol")["rs_120"]}


def audit_splits(root: Path, symbols_arg: str | None = None) -> list[dict]:
    """One row per flagged symbol: {"symbol", "reasons"} where `reasons` lists which of the two
    checks tripped and its value. Never raises on a single symbol's data problem -- a symbol
    whose bars/context can't be read is skipped, not fatal to the audit."""
    root = Path(root)
    rs_max = _rs_120_abs_max(root)
    flagged: list[dict] = []
    for symbol in _symbols_arg(symbols_arg, root):
        reasons = []
        try:
            ratio = _minmax_ratio(root, symbol)
        except Exception as exc:
            reasons.append(f"minmax_ratio: ERROR {exc}")
            ratio = None
        if ratio is not None and ratio > MAX_MINMAX_RATIO:
            reasons.append(f"minmax_ratio={ratio:.3g}")
        rs = rs_max.get(symbol.upper())
        if rs is not None and rs > MAX_ABS_RS_120:
            reasons.append(f"|rs_120|={rs:.3g}")
        if reasons:
            flagged.append({"symbol": symbol, "reasons": reasons})
    return flagged


def run_splits_audit(a: argparse.Namespace) -> int:
    flagged = audit_splits(a.root, getattr(a, "symbols", None))
    for row in flagged:
        print(f"splits-audit {row['symbol']}: {', '.join(row['reasons'])}")
    print(f"splits-audit: {len(flagged)} symbol(s) flagged")
    return 0
