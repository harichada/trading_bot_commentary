"""Per-symbol orchestration with checkpoints, multiprocessing, shuffle control, and scoring."""
from __future__ import annotations

import json
import logging
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from research.strategy_hunt import __version__
from research.strategy_hunt.context import load_context_slice
from research.strategy_hunt.data.bars import load_bars5, load_daily
from research.strategy_hunt.data.earnings import earnings_days
from research.strategy_hunt.data.news import load_news
from research.strategy_hunt.data.regime import regime_at, regime_series
from research.strategy_hunt.data.splits import load_adjusted
from research.strategy_hunt.fit.coldstart import FEATURES, predict_coldstart, symbol_characteristics, train_coldstart
from research.strategy_hunt.fit.writer import build_row
from research.strategy_hunt.models import TRADE_COLUMNS
from research.strategy_hunt.paths import (SHUFFLE_VARIANT, checkpoint_path, choices_path, trades_dir,
                                          trades_path)
from research.strategy_hunt.score.regime_rows import (baseline_pooled_rows, pooled_rows, regime_rows,
                                                      trades_from_frame)
from research.strategy_hunt.sim.calibrate import slippage_override_from
from research.strategy_hunt.sim.costs import CostModel
from research.strategy_hunt.sim.simulator import build_session_table
from research.strategy_hunt.walkforward.runner import run_symbol_family
from research.strategy_hunt.walkforward.windows import build_windows
from strategies.families import get_family
from strategies.families.indicators import indicator_frame

logger = logging.getLogger("strategy_hunt.run")
DEFAULT_ROOT = Path("data/strategy_hunt")
REPORTS_DIR = Path("research/reports/strategy_hunt")
COLDSTART_MAX_N = 60


# trades_path / checkpoint_path / choices_path / trades_dir are imported above and re-exported:
# `from research.strategy_hunt.run import trades_path` stays the public path, while
# research/strategy_hunt/paths.py owns the variant-aware definitions (v-playbook-desk-b1-5 task 10).


def shuffle_bars(bars5: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute 5-min returns within each session (structure-destroying control). Deterministic for a seed."""
    rng = np.random.default_rng(seed)
    parts = []
    for day, g in bars5.groupby(bars5.index.normalize(), sort=False):
        closes = g["close"].to_numpy()
        base = float(g["open"].iloc[0])
        lr = np.diff(np.log(np.concatenate([[base], closes])))
        perm = rng.permutation(len(lr))
        new_close = base * np.exp(np.cumsum(lr[perm]))
        off_o = (g["open"] - g["close"]).to_numpy()[perm]
        off_h = (g["high"] - g["close"]).to_numpy()[perm]
        off_l = (g["low"] - g["close"]).to_numpy()[perm]
        new_open = new_close + off_o
        new_open[0] = base  # session open is the fixed real boundary, not itself permuted
        parts.append(pd.DataFrame({"open": new_open, "high": np.maximum(new_close + off_h, new_close),
                                   "low": np.minimum(new_close + off_l, new_close), "close": new_close,
                                   "volume": g["volume"].to_numpy()[perm]}, index=g.index))
    return pd.concat(parts)


def load_context(root: Path, spy_symbol: str = "SPY", vix_symbol: str = "VIX") -> dict[str, Any]:
    spy5, spy_d, vix_d = load_bars5(root, spy_symbol), load_daily(root, spy_symbol), load_daily(root, vix_symbol)
    regime = regime_series(spy_d, vix_d) if not spy_d.empty and not vix_d.empty else pd.Series(dtype=object)
    return {"spy5": spy5, "spy_daily": spy_d, "regime": regime}


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return "unknown"


def _config_key(family_names: list[str], cost_mult: float, shuffle_seed: Optional[int], notional: float,
                override: Optional[float]) -> dict[str, Any]:
    return {"families": sorted(family_names), "cost_mult": float(cost_mult), "shuffle_seed": shuffle_seed,
            "notional": float(notional), "slippage_override": override, "version": __version__}


def process_symbol(symbol: str, root: Path, family_names: list[str], notional: float = 6000.0, cost_mult: float = 1.0,
                   shuffle_seed: Optional[int] = None, force: bool = False, context: Optional[dict] = None,
                   variant: Optional[str] = None) -> dict[str, Any]:
    """One symbol's out-of-sample trades, written under `root` (see `paths.py` for `variant`).

    `variant="shuffle"` sends trades/choices/checkpoint to the `_shuffle` siblings so the
    shuffled baseline shares the root's inputs without overwriting the real tape's outputs."""
    root = Path(root)
    t0 = time.time()
    raw = load_bars5(root, symbol)
    if raw.empty:
        return {"symbol": symbol, "status": "empty", "families": family_names, "n_trades": 0, "last_bar": None, "elapsed_sec": 0.0}
    last_bar = raw.index[-1].isoformat()
    override = slippage_override_from(root)
    cfg = _config_key(family_names, cost_mult, shuffle_seed, notional, override)
    cp = checkpoint_path(root, symbol, variant)
    if cp.exists() and not force:
        prev = json.loads(cp.read_text())
        if prev.get("last_bar") == last_bar and prev.get("config") == cfg:
            return {"symbol": symbol, "status": "skipped", "families": family_names, "n_trades": prev.get("n_trades", 0),
                    "last_bar": last_bar, "elapsed_sec": 0.0}
    try:
        bars5, daily = load_adjusted(root, symbol)
        if shuffle_seed is not None:
            bars5 = shuffle_bars(bars5, shuffle_seed)
        news = load_news(root, symbol)
        earnings = earnings_days(root, symbol, daily)
        ctx = context or load_context(root)
        regime = ctx["regime"]
        lookup = (lambda ts: regime_at(regime, ts)) if len(regime) else None
        cost_model = CostModel.from_bars(bars5, notional, slippage_override=override).scaled(cost_mult)
        # v-playbook-desk-b1-2026-09-25: the symbol's cross-sectional universe-context slice
        # (rvol rank, RS rank, catalysts, ...), joined onto `ind` as `ctx_*` columns so the
        # in-play families (strategies/families/inplay.py) have something to read. Empty
        # (all-NaN after the join) when the universe context hasn't been built yet, in which
        # case those families propose nothing -- never an error.
        universe_ctx = load_context_slice(root, symbol)
        ind = indicator_frame(bars5, daily, ctx["spy5"], context=universe_ctx)
        table = build_session_table(bars5)
        windows = build_windows(bars5.index[0], bars5.index[-1])
        all_trades, choices = [], {}
        for name in family_names:
            fam = get_family(name)
            trades, ch = run_symbol_family(fam, symbol, bars5, daily, news, cost_model, earnings, windows,
                                           regime_lookup=lookup, spy5=ctx["spy5"], ind=ind, table=table)
            all_trades.extend(t.to_dict() for t in trades)
            choices[name] = ch
        tp, chp = trades_path(root, symbol, variant), choices_path(root, symbol, variant)
        tp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_trades, columns=list(TRADE_COLUMNS)).to_parquet(tp, index=False)
        chp.parent.mkdir(parents=True, exist_ok=True)
        chp.write_text(json.dumps(choices, indent=1, default=str))
        rec = {"symbol": symbol, "status": "done", "families": family_names, "n_trades": len(all_trades), "last_bar": last_bar,
               "elapsed_sec": round(time.time() - t0, 2), "n_windows": len(windows)}
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({**rec, "config": cfg, "commit": _git_commit()}, indent=1))
        return rec
    except Exception as exc:  # per-symbol isolation: record and continue
        logger.exception("process_symbol failed for %s", symbol)
        return {"symbol": symbol, "status": "error", "families": family_names, "n_trades": 0, "last_bar": last_bar,
                "elapsed_sec": round(time.time() - t0, 2), "error": f"{type(exc).__name__}: {exc}"}


def _worker(args: tuple) -> dict[str, Any]:
    symbol, root, families, kw = args
    return process_symbol(symbol, root, families, **kw)


def run_symbols(symbols: list[str], root: Path, family_names: list[str], workers: int = 1,
                variant: Optional[str] = None, **kw) -> list[dict[str, Any]]:
    jobs = [(s, Path(root), list(family_names), {**kw, "variant": variant}) for s in symbols]
    if workers <= 1 or len(jobs) <= 1:
        recs = [_worker(j) for j in jobs]
    else:
        with Pool(processes=workers) as pool:
            recs = list(pool.imap_unordered(_worker, jobs, chunksize=1))
    return sorted(recs, key=lambda r: r["symbol"])


def _load_symbol_trades(tp: Path, root: Path, variant: Optional[str] = None) -> tuple[str, pd.DataFrame, dict]:
    """(symbol, trades frame, choices) for one trades Parquet file. Raises on a bad file;
    both scorers contain that per symbol so one unreadable file cannot end the pass."""
    symbol = tp.stem
    df = pd.read_parquet(tp)
    chp = choices_path(root, symbol, variant)
    choices = json.loads(chp.read_text()) if chp.exists() else {}
    return symbol, df, choices


def _variant_frames(root: Path, family_names: list[str], variant: Optional[str], log_prefix: str) -> list[pd.DataFrame]:
    """Every readable trades frame for one variant, restricted to `family_names`.

    Per-file isolation matches the scorers': one unreadable Parquet is logged and skipped
    rather than ending the pass."""
    frames: list[pd.DataFrame] = []
    for tp in sorted(trades_dir(root, variant).glob("*.parquet")):
        try:
            _symbol, df, _choices = _load_symbol_trades(tp, root, variant)
        except Exception:
            logger.warning("%s: skipping unreadable trades file %s", log_prefix, tp.name, exc_info=True)
            continue
        if not df.empty:
            frames.append(df[df["family"].isin(family_names)])
    return frames


def _walkforward_rows(root: Path, family_names: list[str], computed_at: str) -> list[dict]:
    rows: list[dict] = []
    for tp in sorted(trades_dir(root).glob("*.parquet")):
        try:
            symbol, df, choices = _load_symbol_trades(tp, root)
        except Exception:  # per-symbol isolation: a corrupt Parquet must not end the pass
            logger.warning("score: skipping unreadable trades file %s", tp.name, exc_info=True)
            continue
        for fam in family_names:
            sub = df[df["family"] == fam]
            ch = choices.get(fam, [])
            n_windows = len(ch)
            latest = ch[-1]["params"] if ch else {}
            window_end = str(sub["exit_ts"].max())[:10] if len(sub) else ""
            for side in sorted(sub["side"].unique()) if len(sub) else []:
                trades = trades_from_frame(sub[sub["side"] == side])
                rows.append(build_row(symbol, fam, side, trades, n_windows, latest, computed_at, window_end))
    return rows


def _coldstart_rows(root: Path, wf_rows: list[dict], computed_at: str) -> list[dict]:
    ctx = load_context(root)
    feats: dict[str, dict] = {}
    for bp in sorted((Path(root) / "bars5").glob("*.parquet")):
        sym = bp.stem
        b, d = load_bars5(root, sym), load_daily(root, sym)
        if d.empty:
            continue
        feats[sym] = symbol_characteristics(b, d, load_news(root, sym), ctx["spy_daily"])
    if not feats:
        return []
    X = pd.DataFrame.from_dict(feats, orient="index")[list(FEATURES)]
    models = train_coldstart(X, wf_rows)
    have = {(r["symbol"], r["strategy"], r["side"]): r["n"] for r in wf_rows}
    preds = predict_coldstart(models, X, computed_at)
    return [p for p in preds if have.get((p["symbol"], p["strategy"], p["side"]), 0) < COLDSTART_MAX_N
            and (p["symbol"], p["strategy"], p["side"]) not in {k for k, n in have.items() if n >= COLDSTART_MAX_N}]


def score_regime_root(root: Path, family_names: list[str],
                      computed_at: str) -> tuple[list[dict], list[dict], list[dict]]:
    """v-playbook-desk-b1-2026-09-25: the regime-conditioned per-symbol rows and the
    universe-pooled (family, side, regime) rows, for `write_fit(..., regime_rows=,
    pooled_rows=, baseline_pooled_rows=)`.

    Kept out of `score_root`'s list on purpose: that list feeds the cold-start model and
    the markdown report, both of which count one row per (symbol, family, side) lane and
    would double-count a regime row.

    v-playbook-desk-b1-5 task 10 adds the third return value: when `<root>/trades_shuffle/`
    exists, the shuffled control's trades are pooled the same way into baseline rows, and
    the real pooled rows are scored against the excess over their matching baseline group.
    With no `trades_shuffle/` the baseline list is empty and every pooled row reports
    `no_baseline` — which is a WATCH ceiling, not a PASS.
    """
    per_symbol: list[dict] = []
    frames: list[pd.DataFrame] = []
    base_frames = _variant_frames(Path(root), family_names, SHUFFLE_VARIANT, "score_baseline")
    for tp in sorted(trades_dir(root).glob("*.parquet")):
        try:
            symbol, df, choices = _load_symbol_trades(tp, root)
        except Exception:  # per-symbol isolation: a corrupt Parquet must not end the pass
            logger.warning("score_regime: skipping unreadable trades file %s", tp.name, exc_info=True)
            continue
        if df.empty:
            continue
        frames.append(df[df["family"].isin(family_names)])
        for fam in family_names:
            sub = df[df["family"] == fam]
            if sub.empty:
                continue
            ch = choices.get(fam, [])
            latest = ch[-1]["params"] if ch else {}
            for side in sorted(sub["side"].unique()):
                trades = trades_from_frame(sub[sub["side"] == side])
                # window_end is left to regime_rows: each row ends where *that regime*
                # last traded, which is the honest answer to "how current is this row?".
                per_symbol.extend(regime_rows(symbol, fam, side, trades, len(ch), computed_at, latest))
    baseline = baseline_pooled_rows(pd.concat(base_frames, ignore_index=True), computed_at) if base_frames else []
    pooled = pooled_rows(pd.concat(frames, ignore_index=True), computed_at, baseline=baseline) if frames else []
    return per_symbol, pooled, baseline


def score_root(root: Path, family_names: list[str], computed_at: str, coldstart: bool = True) -> list[dict]:
    rows = _walkforward_rows(root, family_names, computed_at)
    if coldstart:
        keys = {(r["symbol"], r["strategy"], r["side"]) for r in rows}
        extra = [r for r in _coldstart_rows(root, rows, computed_at) if (r["symbol"], r["strategy"], r["side"]) not in keys]
        rows = rows + extra
    return rows


def write_manifest(root: Path, record: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    p = REPORTS_DIR / f"{record.get('date', time.strftime('%Y-%m-%d'))}.json"
    existing = json.loads(p.read_text()) if p.exists() else {}
    existing.update(record)
    p.write_text(json.dumps(existing, indent=1, default=str))
    return p
