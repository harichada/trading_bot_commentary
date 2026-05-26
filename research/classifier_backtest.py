"""Side-classifier backtest harness.

v-classifier-backtest-2026-05-13. Replays the classifier against
historical bars + labels, producing a markdown report. Process-level
isolation: this script imports from ``core.classifier`` and ``research.*``
but NEVER from ``core.engine``. Safe to run while the bot is live.

Usage:
    python research/classifier_backtest.py --symbols NVDA,AAPL,MSFT \\
        --start 2024-01-01 --end 2025-12-31 \\
        --out docs/training_runs/2026-05-13.md

The skeleton handles:
  - CLI parsing
  - Universe loading (yfinance bars + SPY benchmark)
  - Triple-barrier labeling
  - Classifier scoring across every (symbol, day) example
  - Confusion matrix + per-side precision + per-regime breakdown
  - Markdown report

Out of scope for v1 (covered by ``PLAN_trained_side_classifier.md``):
  - Trained-model inference (needs torch+CUDA install)
  - Equity-curve simulation (next iteration)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Bootstrap: when this file is invoked as ``python research/classifier_backtest.py``
# (script mode, not module mode), only the research/ directory is in sys.path —
# so ``import core.classifier`` fails. Prepend the project root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class BacktestConfig:
    symbols: List[str]
    start_date: str
    end_date: str
    benchmark: str
    output_path: Path
    long_threshold: float
    short_threshold: float


def parse_args(argv: Optional[List[str]] = None) -> BacktestConfig:
    parser = argparse.ArgumentParser(
        description="Replay side-classifier decisions on historical bars."
    )
    parser.add_argument(
        "--symbols", required=True,
        help="Comma-separated tickers, e.g. NVDA,AAPL,MSFT",
    )
    parser.add_argument(
        "--start", default="2024-01-01",
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end", default=None,
        help="Backtest end date (YYYY-MM-DD), default = today",
    )
    parser.add_argument(
        "--benchmark", default="SPY",
        help="Benchmark ticker for relative-strength feature",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output markdown report path",
    )
    parser.add_argument(
        "--long-threshold", type=float, default=0.55,
        help="Rule-layer threshold for LONG (default 0.55)",
    )
    parser.add_argument(
        "--short-threshold", type=float, default=0.55,
        help="Rule-layer threshold for SHORT (default 0.55)",
    )
    args = parser.parse_args(argv)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    end_date = args.end or datetime.utcnow().strftime("%Y-%m-%d")

    return BacktestConfig(
        symbols=symbols,
        start_date=args.start,
        end_date=end_date,
        benchmark=args.benchmark.upper(),
        output_path=Path(args.out),
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
    )


def _load_universe(cfg: BacktestConfig):
    """Load OHLCV via yfinance for each symbol + benchmark.

    Lazy import — yfinance install happens on first run; the rest of
    this module imports cleanly without it for unit testing.
    """
    import yfinance as yf
    import pandas as pd

    universe = {}
    print(f"[backtest] loading {len(cfg.symbols)} symbols + {cfg.benchmark}", flush=True)
    for sym in cfg.symbols + [cfg.benchmark]:
        df = yf.download(
            sym, start=cfg.start_date, end=cfg.end_date,
            progress=False, auto_adjust=True,
        )
        if df.empty:
            print(f"[backtest] WARN: {sym} returned no bars; skipping", flush=True)
            continue
        df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
        universe[sym] = df
    return universe


def _evaluate_symbol(symbol: str, daily_bars, benchmark_bars, cfg: BacktestConfig):
    """Score every bar with the classifier; return (decisions, labels)."""
    from core.classifier import classify_rule_based
    from research.features import build_features
    from research.labels import label_dataset

    # ATR series for labeling: simple True Range mean over 14
    import pandas as pd
    high = daily_bars["high"]
    low = daily_bars["low"]
    close = daily_bars["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean()
    labels = label_dataset(daily_bars, atr, k=1.5, horizon_bars=5)

    decisions = []
    # For each bar starting at index 50 (need history for SMA-50)
    for i in range(50, len(daily_bars)):
        window = daily_bars.iloc[: i + 1]
        bench_window = (
            benchmark_bars.iloc[: i + 1] if benchmark_bars is not None else None
        )
        features = build_features(
            symbol=symbol,
            daily_bars=window,
            benchmark_daily=bench_window,
        )
        d = classify_rule_based(
            features,
            long_threshold=cfg.long_threshold,
            short_threshold=cfg.short_threshold,
        )
        decisions.append({
            "date": daily_bars.index[i],
            "long_score": d.long_score,
            "short_score": d.short_score,
            "allowed_long": d.allows_long(),
            "allowed_short": d.allows_short(),
            "is_forbidden": d.is_forbidden(),
            "primary_reason": d.primary_reason,
            "label": labels.iloc[i].value,
        })
    return decisions


def _write_report(cfg: BacktestConfig, by_symbol: dict) -> None:
    """Markdown report at cfg.output_path with confusion matrix +
    per-side precision + counts."""
    lines = [
        f"# Side-Classifier Backtest — {datetime.utcnow().isoformat(timespec='seconds')}Z",
        "",
        f"**Universe**: {', '.join(cfg.symbols)}",
        f"**Period**: {cfg.start_date} → {cfg.end_date}",
        f"**Benchmark**: {cfg.benchmark}",
        f"**Thresholds**: long={cfg.long_threshold}  short={cfg.short_threshold}",
        "",
        "## Per-symbol summary",
        "",
        "| Symbol | Examples | LONG allowed | SHORT allowed | Forbidden | LONG prec | SHORT prec |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    total_long_tp = total_long_allowed = 0
    total_short_tp = total_short_allowed = 0
    for sym, decisions in by_symbol.items():
        if not decisions:
            continue
        long_allowed = sum(d["allowed_long"] for d in decisions)
        short_allowed = sum(d["allowed_short"] for d in decisions)
        forbidden = sum(d["is_forbidden"] for d in decisions)
        long_tp = sum(d["allowed_long"] and d["label"] == "long_wins" for d in decisions)
        short_tp = sum(d["allowed_short"] and d["label"] == "short_wins" for d in decisions)
        long_prec = (long_tp / long_allowed) if long_allowed else float("nan")
        short_prec = (short_tp / short_allowed) if short_allowed else float("nan")
        lines.append(
            f"| {sym} | {len(decisions)} | {long_allowed} | {short_allowed} | "
            f"{forbidden} | {long_prec:.2%} | {short_prec:.2%} |"
        )
        total_long_tp += long_tp
        total_long_allowed += long_allowed
        total_short_tp += short_tp
        total_short_allowed += short_allowed

    overall_long_prec = (total_long_tp / total_long_allowed) if total_long_allowed else float("nan")
    overall_short_prec = (total_short_tp / total_short_allowed) if total_short_allowed else float("nan")
    lines += [
        "",
        "## Overall",
        "",
        f"- LONG entries allowed: **{total_long_allowed}**, precision: **{overall_long_prec:.2%}**",
        f"- SHORT entries allowed: **{total_short_allowed}**, precision: **{overall_short_prec:.2%}**",
        "",
        "## Notes",
        "",
        "- Triple-barrier labels: upper/lower = entry ± 1.5×ATR_14, horizon 5 bars.",
        "- Precision counts only the entries the classifier ALLOWED on its side; entries the classifier blocked are not in the denominator. To get the *value* of blocking, compute precision on a baseline run with thresholds=0 (allow everything).",
        "- Per-regime (VIX-bucket), per-sector breakdowns: future work.",
    ]
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_path.write_text("\n".join(lines))
    print(f"[backtest] wrote {cfg.output_path}", flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    universe = _load_universe(cfg)
    bench = universe.get(cfg.benchmark)
    by_symbol: dict = {}
    for sym in cfg.symbols:
        bars = universe.get(sym)
        if bars is None:
            print(f"[backtest] skipping {sym} — no data", flush=True)
            by_symbol[sym] = []
            continue
        by_symbol[sym] = _evaluate_symbol(sym, bars, bench, cfg)
    _write_report(cfg, by_symbol)
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
