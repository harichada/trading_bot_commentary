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
from research.strategy_hunt.data.bars import load_bars5, load_daily
from research.strategy_hunt.data.earnings import earnings_days
from research.strategy_hunt.data.news import load_news
from research.strategy_hunt.data.regime import regime_at, regime_series
from research.strategy_hunt.data.splits import load_adjusted
from research.strategy_hunt.fit.coldstart import FEATURES, predict_coldstart, symbol_characteristics, train_coldstart
from research.strategy_hunt.fit.writer import build_row
from research.strategy_hunt.models import TRADE_COLUMNS, SimTrade
from research.strategy_hunt.score.regime_rows import pooled_rows, regime_rows, trades_from_frame
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


def trades_path(root: Path, symbol: str) -> Path:
    return Path(root) / "trades" / f"{symbol.upper()}.parquet"


def checkpoint_path(root: Path, symbol: str) -> Path:
    return Path(root) / "checkpoints" / f"{symbol.upper()}.json"


def choices_path(root: Path, symbol: str) -> Path:
    return Path(root) / "choices" / f"{symbol.upper()}.json"


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
                   shuffle_seed: Optional[int] = None, force: bool = False, context: Optional[dict] = None) -> dict[str, Any]:
    root = Path(root)
    t0 = time.time()
    raw = load_bars5(root, symbol)
    if raw.empty:
        return {"symbol": symbol, "status": "empty", "families": family_names, "n_trades": 0, "last_bar": None, "elapsed_sec": 0.0}
    last_bar = raw.index[-1].isoformat()
    override = slippage_override_from(root)
    cfg = _config_key(family_names, cost_mult, shuffle_seed, notional, override)
    cp = checkpoint_path(root, symbol)
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
        ind = indicator_frame(bars5, daily, ctx["spy5"])
        table = build_session_table(bars5)
        windows = build_windows(bars5.index[0], bars5.index[-1])
        all_trades, choices = [], {}
        for name in family_names:
            fam = get_family(name)
            trades, ch = run_symbol_family(fam, symbol, bars5, daily, news, cost_model, earnings, windows,
                                           regime_lookup=lookup, spy5=ctx["spy5"], ind=ind, table=table)
            all_trades.extend(t.to_dict() for t in trades)
            choices[name] = ch
        tp = trades_path(root, symbol)
        tp.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_trades, columns=list(TRADE_COLUMNS)).to_parquet(tp, index=False)
        choices_path(root, symbol).parent.mkdir(parents=True, exist_ok=True)
        choices_path(root, symbol).write_text(json.dumps(choices, indent=1, default=str))
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


def run_symbols(symbols: list[str], root: Path, family_names: list[str], workers: int = 1, **kw) -> list[dict[str, Any]]:
    jobs = [(s, Path(root), list(family_names), dict(kw)) for s in symbols]
    if workers <= 1 or len(jobs) <= 1:
        recs = [_worker(j) for j in jobs]
    else:
        with Pool(processes=workers) as pool:
            recs = list(pool.imap_unordered(_worker, jobs, chunksize=1))
    return sorted(recs, key=lambda r: r["symbol"])


def _walkforward_rows(root: Path, family_names: list[str], computed_at: str) -> list[dict]:
    rows: list[dict] = []
    for tp in sorted((Path(root) / "trades").glob("*.parquet")):
        df = pd.read_parquet(tp)
        symbol = tp.stem
        choices = json.loads(choices_path(root, symbol).read_text()) if choices_path(root, symbol).exists() else {}
        for fam in family_names:
            sub = df[df["family"] == fam]
            ch = choices.get(fam, [])
            n_windows = len(ch)
            latest = ch[-1]["params"] if ch else {}
            window_end = str(sub["exit_ts"].max())[:10] if len(sub) else ""
            for side in sorted(sub["side"].unique()) if len(sub) else []:
                s2 = sub[sub["side"] == side]
                trades = [SimTrade(**{**r, "entry_ts": pd.Timestamp(r["entry_ts"]), "exit_ts": pd.Timestamp(r["exit_ts"])})
                          for r in s2.to_dict("records")]
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


def score_regime_root(root: Path, family_names: list[str], computed_at: str) -> tuple[list[dict], list[dict]]:
    """v-playbook-desk-b1-2026-09-25: the regime-conditioned per-symbol rows and the
    universe-pooled (family, side, regime) rows, for `write_fit(..., regime_rows=,
    pooled_rows=)`.

    Kept out of `score_root`'s list on purpose: that list feeds the cold-start model and
    the markdown report, both of which count one row per (symbol, family, side) lane and
    would double-count a regime row.
    """
    per_symbol: list[dict] = []
    frames: list[pd.DataFrame] = []
    for tp in sorted((Path(root) / "trades").glob("*.parquet")):
        df = pd.read_parquet(tp)
        if df.empty:
            continue
        symbol = tp.stem
        choices = json.loads(choices_path(root, symbol).read_text()) if choices_path(root, symbol).exists() else {}
        frames.append(df[df["family"].isin(family_names)])
        for fam in family_names:
            sub = df[df["family"] == fam]
            if sub.empty:
                continue
            ch = choices.get(fam, [])
            latest = ch[-1]["params"] if ch else {}
            window_end = str(sub["exit_ts"].max())[:10]
            for side in sorted(sub["side"].unique()):
                trades = trades_from_frame(sub[sub["side"] == side])
                per_symbol.extend(regime_rows(symbol, fam, side, trades, len(ch), computed_at, latest, window_end))
    pooled = pooled_rows(pd.concat(frames, ignore_index=True), computed_at) if frames else []
    return per_symbol, pooled


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
