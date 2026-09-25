"""Command line for the strategy-hunt pipeline: export, context, universe-context, news, earnings,
calibrate, run, score, report, playbook, all."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from core.desk.playbook import DEFAULT_DESK_DIR as PLAYBOOK_DESK_DIR
from research.strategy_hunt.fit.writer import write_fit
from research.strategy_hunt.paths import SHUFFLE_VARIANT
from research.strategy_hunt.run import DEFAULT_ROOT, run_symbols, score_regime_root, score_root, write_manifest
from strategies.families import family_names

logger = logging.getLogger("strategy_hunt.cli")
CONTEXT_SYMBOLS = ("SPY", "VIX")
# v-playbook-desk-b1-5 task 10: `all` runs the shuffled baseline with this fixed seed, so the
# control is reproducible and the same across nightly runs.
BASELINE_SHUFFLE_SEED = 7


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
    from research.strategy_hunt.context import build_universe_context
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
    # v-playbook-desk-b1-2026-09-25 (fix round 1, finding 2): a failed universe-context build is
    # fatal, not a swallowed warning. Every ctx_* column downstream families read
    # (strategies/families/inplay.py) depends on this file; letting the pipeline limp on with a
    # missing/stale one means those families silently propose nothing for the rest of the run,
    # which is worse than stopping loudly here.
    try:
        uc = build_universe_context(a.root)
    except Exception as exc:
        print(f"context: universe ERROR {exc}")
        return 1
    print(f"context: universe -> {uc}")
    return 0


def cmd_universe_context(a: argparse.Namespace) -> int:
    """Rebuild only <root>/context/universe_daily.parquet from whatever bars5/earnings/news are
    already on disk -- no SPY/VIX/splits fetch. Useful to refresh the cross-sectional context
    after `news`/`earnings` land, without re-running the whole `context` subcommand."""
    from research.strategy_hunt.context import build_universe_context

    try:
        uc = build_universe_context(a.root)
    except Exception as exc:
        print(f"universe-context ERROR {exc}")
        return 1
    print(f"universe-context: {uc}")
    return 0


def cmd_news(a: argparse.Namespace) -> int:
    from research.strategy_hunt.data.news import rescore_news, update_news

    # v-playbook-desk-b1-2026-09-25 (Task 7b): --rescore re-scores the existing cache with the
    # configured sentiment backend instead of fetching new rows, so it needs no Alpaca creds.
    # `cmd_all`'s "all" subparser doesn't register --rescore, hence getattr with a False default.
    if getattr(a, "rescore", False):
        for s in _symbols_arg(a.symbols, a.root):
            try:
                print(f"news {s}: rescored {rescore_news(a.root, s)}")
            except Exception as exc:
                print(f"news {s}: ERROR {exc}")
        return 0

    import requests

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
    # v-playbook-desk-b1-5 task 10: a shuffled run is a *variant* of this root, not a root of
    # its own -- it writes trades_shuffle/, checkpoints_shuffle/, choices_shuffle/ and shares
    # the root's bars/news/context, so `score` can read both tapes in one pass.
    variant = SHUFFLE_VARIANT if a.shuffle_seed is not None else None
    t0 = time.time()
    recs = run_symbols(symbols, a.root, fams, workers=a.workers, variant=variant, force=a.force,
                       cost_mult=a.cost_mult, shuffle_seed=a.shuffle_seed, notional=a.notional)
    counts = {}
    for r in recs:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        print(f"run {r['symbol']}: {r['status']} trades={r['n_trades']} {r.get('error', '')}")
    print(f"run{'' if variant is None else ' (' + variant + ')'} summary: {counts} elapsed={time.time() - t0:.0f}s")
    # Keyed by variant -- both the run record and its error list. `write_manifest` merges with
    # dict.update, so a shared "errors" key meant the baseline run wiped the real run's failures
    # off the record for that date (fix r1, finding 1).
    suffix = "" if variant is None else f"_{variant}"
    write_manifest(a.root, {"date": time.strftime("%Y-%m-%d"),
                            f"run{suffix}": {"symbols": len(symbols), "families": fams,
                                             "counts": counts, "cost_mult": a.cost_mult,
                                             "shuffle_seed": a.shuffle_seed, "variant": variant,
                                             "notional": a.notional, "elapsed_sec": round(time.time() - t0)},
                            f"errors{suffix}": [r for r in recs if r["status"] == "error"]})
    return 0 if counts.get("error", 0) <= 0.1 * max(1, len(recs)) else 1


def cmd_score(a: argparse.Namespace) -> int:
    from research.strategy_hunt.report import write_report

    computed_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows = score_root(a.root, _families_arg(a.families), computed_at, coldstart=not a.no_coldstart)
    # v-playbook-desk-b1-2026-09-25: regime-conditioned per-symbol rows and the universe-pooled
    # (family, side, regime) rows -- without this the gate's lookup_regime always falls back to
    # the unconditional lookup(), since write_fit never sees any regime/pooled rows to persist.
    # v-playbook-desk-b1-5 task 10: the third list is the shuffled baseline pooled from
    # <root>/trades_shuffle/; pooled verdicts are already scored against the excess over it.
    reg_rows, pool_rows, base_rows = score_regime_root(a.root, _families_arg(a.families), computed_at)
    engine = None
    if a.postgres:
        from sqlalchemy import create_engine
        engine = create_engine(os.environ["POSTGRES_DSN"])
    out = write_fit(rows, a.root, computed_at, engine=engine, regime_rows=reg_rows, pooled_rows=pool_rows,
                    baseline_pooled_rows=base_rows)
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    for r in sorted(rows, key=lambda r: -r["score"])[:20]:
        print(f"{r['symbol']:6} {r['strategy']:18} {r['side']:5} {r['verdict']:5} n={r['n']:4} exp_r={r['exp_r']:+.3f} score={r['score']:.3f}")
    print(f"score summary: {counts} -> {out['json']}")
    print(f"pooled rows: {len(pool_rows)} (baseline rows: {len(base_rows)})")
    rp = write_report(a.root, rows, computed_at, pooled=pool_rows)
    print(f"report: {rp}")
    return 0


def cmd_playbook(a: argparse.Namespace) -> int:
    """v-playbook-desk-b2-2026-09-25 (Task 3): rewrite <desk-dir>/playbook.json from tonight's
    resolved shadow ledgers plus the live `bot_trades` tape. The body lives next to the updater
    (`playbook_update.run_playbook_command`) to keep this module inside its line budget, and is
    imported lazily so `--help` on any other subcommand still costs nothing."""
    from research.strategy_hunt.playbook_update import run_playbook_command

    return run_playbook_command(a, _families_arg(getattr(a, "families", None)))


def cmd_report(a: argparse.Namespace) -> int:
    from research.strategy_hunt.fit.writer import load_snapshot
    from research.strategy_hunt.report import write_report

    snap = load_snapshot(a.root)
    rows = [row for sym_rows in snap["rows"].values() for row in sym_rows.values()]
    # A malformed section must degrade to "no pooled rows", never raise -- the same contract
    # core/strategy_fit.py holds on the live path (fix r1, finding 6).
    raw_pooled = snap.get("pooled_rows")
    if isinstance(raw_pooled, dict):
        pooled = [r for r in raw_pooled.values() if isinstance(r, dict)]
    else:
        pooled = []
        if raw_pooled:
            logger.warning("report: ignoring malformed pooled_rows section (%s)", type(raw_pooled).__name__)
    computed_at = snap.get("computed_at") or time.strftime("%Y-%m-%dT%H:%M:%S")
    rp = write_report(a.root, rows, computed_at, pooled=pooled)
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
        try:
            records = read_ledger(ledger)
            trades = resolve(records, a.root, name)
            scorecards[name] = stage_a_scorecard(trades, name)
        except Exception:
            logger.warning("promote: failed to resolve scorecard for family %s; treating as none", name,
                           exc_info=True)
            scorecards[name] = "none"

    # v-strategy-fit-review2-2026-09-25 (Critical 1): the gate's own two flags feed the
    # gate_not_enforcing blocker -- a lane must not go live while the engine-side
    # per-symbol check is disabled or still in shadow.
    rows = promotion_table(snapshot, scorecards, cfg.FIT_LANES_ENABLED, cfg.FIT_LANE_LIVE_ENTRIES_ENABLED,
                           cfg.SWING_ALLOW_OVERNIGHT, cfg.ENABLE_STRATEGY_FIT_GATE, cfg.STRATEGY_FIT_SHADOW)
    md = render_promotion(rows)
    print(md)

    out = Path("docs/research/strategy_hunt_promotion.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"promote: {out}")
    return 0


def _baseline_namespace(a: argparse.Namespace) -> argparse.Namespace:
    """A copy of `a` configured for the shuffled baseline run. A copy, not a mutation: the
    later steps must still see the real run's `shuffle_seed=None` and the user's `--force`."""
    return argparse.Namespace(**{**vars(a), "shuffle_seed": BASELINE_SHUFFLE_SEED, "force": True})


def cmd_all(a: argparse.Namespace) -> int:
    # v-playbook-desk-b1-2026-09-25 (fix round 1, finding 1): context must build the universe
    # context AFTER news/earnings land, so ctx_news_count_24h/ctx_earnings_prev/ctx_eps_surprise_pct
    # reflect tonight's caches by the time `run` consumes them via process_symbol -> ctx_* columns.
    for fn in (cmd_export, cmd_news, cmd_earnings, cmd_context, cmd_calibrate, cmd_run):
        rc = fn(a)
        if rc:
            return rc
    # v-playbook-desk-b1-5 task 10: the shuffled baseline is part of the nightly pipeline, not a
    # separate manual run -- without it every pooled row scores `no_baseline` and can only WATCH.
    # force=True because its checkpoints are only valid for one tape+seed and must not be reused
    # from a night whose real trades have since changed.
    if not getattr(a, "no_baseline", False):
        rc = cmd_run(_baseline_namespace(a))
        if rc:
            return rc
    rc = cmd_score(a)
    if rc:
        return rc
    # v-playbook-desk-b2-2026-09-25 (Task 3): the playbook is updated last -- it reads the
    # snapshot `score` just wrote for its fit verdicts.
    return cmd_playbook(a)


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
        sp.add_argument("--no-baseline", action="store_true",
                        help="`all` only: skip the shuffled baseline run. Whatever baseline is "
                             "already in the root is reused; scoring flags it baseline_mismatch / "
                             "baseline_stale if it is partial or old, and pooled rows fall back to "
                             "no_baseline only when the root has none at all")
        sp.add_argument("--postgres", action="store_true")

    for name, fn in (("export", cmd_export), ("context", cmd_context),
                     ("earnings", cmd_earnings), ("calibrate", cmd_calibrate), ("run", cmd_run),
                     ("score", cmd_score), ("report", cmd_report), ("all", cmd_all)):
        sp = sub.add_parser(name)
        common(sp)
        sp.set_defaults(fn=fn)

    # v-playbook-desk-b1-2026-09-25 (Task 7b): `news` gets its own subparser (like
    # universe-context below) so --rescore only exists there, not on every other subcommand.
    sp = sub.add_parser("news")
    common(sp)
    sp.add_argument("--rescore", action="store_true",
                    help="re-score the existing cache with the configured sentiment backend "
                         "instead of fetching new rows (no Alpaca creds required)")
    sp.set_defaults(fn=cmd_news)

    # v-playbook-desk-b1-2026-09-25 (fix round 1, finding 5): rebuild-only-the-context-file takes
    # no symbols/families/cost-mult/etc -- just --root, like `promote`.
    sp = sub.add_parser("universe-context")
    sp.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sp.set_defaults(fn=cmd_universe_context)

    sp = sub.add_parser("scorecard")
    sp.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sp.add_argument("--family", required=True)
    sp.add_argument("--ledger", type=Path, default=None, help="default: data/shadow_fit_<family>.ndjson")
    sp.add_argument("--cost-mult", type=float, default=1.0)
    sp.set_defaults(fn=cmd_scorecard)

    # v-playbook-desk-b2-2026-09-25 (Task 3): --desk-dir lives only here (like `promote`'s
    # --root), so no other subcommand grows a flag it ignores.
    sp = sub.add_parser("playbook")
    sp.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sp.add_argument("--families", default=None, help="comma list; default all")
    sp.add_argument("--desk-dir", type=Path, default=PLAYBOOK_DESK_DIR,
                    help="where playbook.json is written (default data/desk)")
    sp.add_argument("--postgres", action="store_true",
                    help="also read the live bot_trades tape (needs POSTGRES_DSN)")
    sp.set_defaults(fn=cmd_playbook)

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
