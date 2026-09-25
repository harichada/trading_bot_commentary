"""Command line for the strategy-hunt pipeline: export, context, news, earnings, calibrate, run, score, report, all."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from research.strategy_hunt.run import DEFAULT_ROOT, run_symbols, score_root, write_manifest
from strategies.families import family_names

logger = logging.getLogger("strategy_hunt.cli")
CONTEXT_SYMBOLS = ("SPY", "VIX")


def _symbols_arg(value: str | None, root: Path) -> list[str]:
    if not value:
        return sorted(p.stem for p in (root / "bars5").glob("*.parquet") if p.stem not in CONTEXT_SYMBOLS)
    if value.startswith("@"):
        return [s.strip().upper() for s in Path(value[1:]).read_text().splitlines() if s.strip()]
    return [s.strip().upper() for s in value.split(",") if s.strip()]


def _families_arg(value: str | None) -> list[str]:
    return [f.strip() for f in value.split(",")] if value else family_names()


def cmd_export(a: argparse.Namespace) -> int:
    from sqlalchemy import create_engine
    from research.strategy_hunt.data.bars import export_symbol, select_universe

    engine = create_engine(os.environ["POSTGRES_DSN"])
    symbols = _symbols_arg(a.symbols, a.root) if a.symbols else select_universe(engine, a.min_days)
    for s in [*CONTEXT_SYMBOLS, *symbols]:
        try:
            n = export_symbol(engine, s, a.root)
            print(f"export {s}: +{n} bars")
        except Exception as exc:
            print(f"export {s}: ERROR {exc}")
    return 0


def cmd_context(a: argparse.Namespace) -> int:
    import requests
    from research.strategy_hunt.data.context import ensure_context
    from research.strategy_hunt.data.splits import update_splits, yfinance_splits_fetcher

    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    sess = requests.Session()
    out = ensure_context(a.root, sess, key, secret)
    print(f"context: spy+{out['spy']} vix+{out['vix']}")
    for s in _symbols_arg(a.symbols, a.root):
        try:
            n = update_splits(a.root, s, yfinance_splits_fetcher)
            print(f"splits {s}: {n}")
        except Exception as exc:
            print(f"splits {s}: ERROR {exc}")
    return 0


def cmd_news(a: argparse.Namespace) -> int:
    import requests
    from research.strategy_hunt.data.news import update_news

    key, secret = os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        print("news: ALPACA_API_KEY / ALPACA_SECRET_KEY not set; skipping")
        return 0
    sess = requests.Session()
    for s in _symbols_arg(a.symbols, a.root):
        try:
            print(f"news {s}: +{update_news(a.root, s, sess, key, secret)}")
        except Exception as exc:
            print(f"news {s}: ERROR {exc}")
    return 0


def cmd_earnings(a: argparse.Namespace) -> int:
    from research.strategy_hunt.data.earnings import update_earnings, yfinance_fetcher

    for s in _symbols_arg(a.symbols, a.root):
        print(f"earnings {s}: {update_earnings(a.root, s, yfinance_fetcher)} dates")
    return 0


def cmd_calibrate(a: argparse.Namespace) -> int:
    from sqlalchemy import create_engine
    from research.strategy_hunt.sim.calibrate import (calibration_summary, load_fill_pairs, realized_slippage,
                                                       write_calibration)

    engine = create_engine(os.environ["POSTGRES_DSN"])
    fills = load_fill_pairs(engine)
    slips = realized_slippage(fills)
    summary = calibration_summary(slips)
    write_calibration(a.root, summary)
    print(f"calibrate: n={summary['n']} median_frac={summary['median_frac']} usable={summary['usable']}")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    symbols, fams = _symbols_arg(a.symbols, a.root), _families_arg(a.families)
    t0 = time.time()
    recs = run_symbols(symbols, a.root, fams, workers=a.workers, force=a.force, cost_mult=a.cost_mult,
                       shuffle_seed=a.shuffle_seed, notional=a.notional)
    counts = {}
    for r in recs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        print(f"run {r['symbol']}: {r['status']} trades={r['n_trades']} {r.get('error', '')}")
    print(f"run summary: {counts} elapsed={time.time() - t0:.0f}s")
    write_manifest(a.root, {"date": time.strftime("%Y-%m-%d"), "run": {"symbols": len(symbols), "families": fams,
                            "counts": counts, "cost_mult": a.cost_mult, "shuffle_seed": a.shuffle_seed,
                            "notional": a.notional, "elapsed_sec": round(time.time() - t0)},
                            "errors": [r for r in recs if r["status"] == "error"]})
    return 0 if counts.get("error", 0) <= 0.1 * max(1, len(recs)) else 1


def cmd_score(a: argparse.Namespace) -> int:
    from research.strategy_hunt.fit.writer import write_fit
    from research.strategy_hunt.report import write_report

    computed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows = score_root(a.root, _families_arg(a.families), computed_at, coldstart=not a.no_coldstart)
    engine = None
    if a.postgres:
        from sqlalchemy import create_engine
        engine = create_engine(os.environ["POSTGRES_DSN"])
    out = write_fit(rows, a.root, computed_at, engine=engine)
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    for r in sorted(rows, key=lambda r: -r["score"])[:20]:
        print(f"{r['symbol']:6} {r['strategy']:18} {r['side']:5} {r['verdict']:5} n={r['n']:4} exp_r={r['exp_r']:+.3f} score={r['score']:.3f}")
    print(f"score summary: {counts} -> {out['json']}")
    rp = write_report(a.root, rows, computed_at)
    print(f"report: {rp}")
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    from research.strategy_hunt.fit.writer import load_snapshot
    from research.strategy_hunt.report import write_report

    snap = load_snapshot(a.root)
    rows = [row for sym_rows in snap["rows"].values() for row in sym_rows.values()]
    computed_at = snap.get("computed_at") or time.strftime("%Y-%m-%dT%H:%M:%S")
    rp = write_report(a.root, rows, computed_at)
    print(f"report: {rp}")
    return 0


def cmd_scorecard(a: argparse.Namespace) -> int:
    from research.strategy_hunt.shadow_resolver import (default_ledger_path, read_ledger, resolve,
                                                         stage_a_scorecard, write_scorecard)

    ledger = a.ledger or default_ledger_path(a.family)
    records = read_ledger(ledger)
    trades = resolve(records, a.root, a.family, cost_mult=a.cost_mult)
    card = stage_a_scorecard(trades, a.family)
    computed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    for side in ("long", "short"):
        m = card.get(side, {})
        print(f"{a.family:14} {side:5} n={m.get('n', 0):4} pf={m.get('pf', 0):.3f} wr={m.get('wr', 0):.3f} "
              f"exp_r={m.get('exp_r', 0):+.4f} max_dd_pct={m.get('max_dd_pct', 0):.4f} verdict={m.get('verdict', '')}")
    print(f"scorecard {a.family}: {card['verdict']} (ledger={ledger})")
    rp = write_scorecard(a.root, a.family, card, computed_at)
    print(f"scorecard: {rp}")
    return 0


def cmd_promote(a: argparse.Namespace) -> int:
    from core.config import Config
    from research.strategy_hunt.fit.writer import load_snapshot
    from research.strategy_hunt.promote import promotion_table, render_promotion
    from research.strategy_hunt.shadow_resolver import default_ledger_path, read_ledger, resolve, stage_a_scorecard

    cfg = Config()
    snapshot = load_snapshot(a.root)
    scorecards: dict[str, object] = {}
    for name in family_names():
        ledger = default_ledger_path(name)
        if not ledger.exists():
            scorecards[name] = "none"
            continue
        records = read_ledger(ledger)
        trades = resolve(records, a.root, name)
        scorecards[name] = stage_a_scorecard(trades, name)

    rows = promotion_table(snapshot, scorecards, cfg.FIT_LANES_ENABLED, cfg.FIT_LANE_LIVE_ENTRIES_ENABLED,
                           cfg.SWING_ALLOW_OVERNIGHT)
    md = render_promotion(rows)
    print(md)

    out = Path("docs/research/strategy_hunt_promotion.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"promote: {out}")
    return 0


def cmd_all(a: argparse.Namespace) -> int:
    for fn in (cmd_export, cmd_context, cmd_news, cmd_earnings, cmd_calibrate, cmd_run, cmd_score):
        rc = fn(a)
        if rc:
            return rc
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m research.strategy_hunt")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--root", type=Path, default=DEFAULT_ROOT)
        sp.add_argument("--symbols", default=None, help="comma list or @file")
        sp.add_argument("--families", default=None, help="comma list; default all")
        sp.add_argument("--min-days", type=int, default=250)
        sp.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
        sp.add_argument("--force", action="store_true")
        sp.add_argument("--cost-mult", type=float, default=1.0)
        sp.add_argument("--shuffle-seed", type=int, default=None)
        sp.add_argument("--notional", type=float, default=6000.0)
        sp.add_argument("--no-coldstart", action="store_true")
        sp.add_argument("--postgres", action="store_true")

    for name, fn in (("export", cmd_export), ("context", cmd_context), ("news", cmd_news), ("earnings", cmd_earnings),
                     ("calibrate", cmd_calibrate), ("run", cmd_run), ("score", cmd_score), ("report", cmd_report),
                     ("all", cmd_all)):
        sp = sub.add_parser(name)
        common(sp)
        sp.set_defaults(fn=fn)

    sp = sub.add_parser("scorecard")
    sp.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sp.add_argument("--family", required=True)
    sp.add_argument("--ledger", type=Path, default=None, help="default: data/shadow_fit_<family>.ndjson")
    sp.add_argument("--cost-mult", type=float, default=1.0)
    sp.set_defaults(fn=cmd_scorecard)

    sp = sub.add_parser("promote")
    sp.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sp.set_defaults(fn=cmd_promote)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    a = build_parser().parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
