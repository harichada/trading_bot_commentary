"""Build fit rows from out-of-sample trades and persist them (Parquet + atomic JSON + SQL upsert)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from research.strategy_hunt.models import FitMetrics, SimTrade
from research.strategy_hunt.score.metrics import compute_metrics, per_regime
from research.strategy_hunt.score.verdict import FLOORS, verdict

ROW_KEYS = ("symbol", "strategy", "side", "computed_at", "window_end", "n", "sessions", "exp_r", "exp_r_lb",
            "t_stat", "pf", "wr", "max_dd_r", "stability", "decay", "score", "verdict", "failed_floors",
            "params_json", "regime_exp_json", "source")
PF_INF = 999.0
TABLE = "strategy_fit"


def build_row(symbol: str, strategy: str, side: str, trades: list[SimTrade], n_windows: int,
              latest_params: dict, computed_at: str, window_end: str, source: str = "walkforward",
              verdict_fn: Optional[Callable[[FitMetrics], tuple[str, list[str]]]] = None) -> dict[str, Any]:
    """One fit row. `source`/`verdict_fn` let the pooled lanes reuse this builder with
    their own floors (`score.regime_rows.verdict_pooled`) without duplicating it."""
    m = compute_metrics(trades, n_windows)
    v, failed = (verdict_fn or verdict)(m)
    score = m.exp_r * math.sqrt(m.stability) if m.exp_r > 0 else 0.0
    row = {
        "symbol": symbol, "strategy": strategy, "side": side, "computed_at": computed_at, "window_end": window_end,
        "n": m.n, "sessions": m.sessions, "exp_r": m.exp_r, "exp_r_lb": m.exp_r_lb, "t_stat": m.t_stat,
        "pf": PF_INF if math.isinf(m.pf) else m.pf, "wr": m.wr,
        "max_dd_r": m.max_dd_r, "stability": m.stability, "decay": m.decay, "score": float(score), "verdict": v,
        "failed_floors": json.dumps(failed), "params_json": json.dumps(latest_params, sort_keys=True),
        "regime_exp_json": json.dumps(per_regime(trades), sort_keys=True), "source": source,
    }
    return {k: row[k] for k in ROW_KEYS}


def fit_dir(root: Path) -> Path:
    return Path(root) / "fit"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_fit(rows: list[dict], root: Path, computed_at: str, engine: Any = None,
              regime_rows: Optional[list[dict]] = None, pooled_rows: Optional[list[dict]] = None) -> dict[str, str]:
    """Persist the fit rows: Parquet per row family plus one atomic JSON snapshot.

    `regime_rows` / `pooled_rows` (v-playbook-desk-b1-2026-09-25) are additive and
    optional — a caller that passes neither writes exactly the snapshot shape older
    readers expect, with the two new keys present but empty. They are kept out of
    `fit_<date>.parquet` and out of the SQL table on purpose: that table's primary key
    is (symbol, strategy, side, computed_at), which pooled rows would collide on
    across regimes.
    """
    d = fit_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    date = str(computed_at)[:10]
    pq = d / f"fit_{date}.parquet"
    pd.DataFrame(rows, columns=list(ROW_KEYS)).to_parquet(pq, index=False)
    nested: dict[str, dict[str, dict]] = {}
    for r in rows:
        nested.setdefault(r["symbol"], {})[f"{r['strategy']}:{r['side']}"] = r

    regime_nested: dict[str, dict[str, dict]] = {}
    for r in regime_rows or []:
        regime_nested.setdefault(r["symbol"], {})[_regime_key(r)] = r
    pooled_flat = {_regime_key(r): r for r in pooled_rows or []}

    # v-playbook-desk-b1-2026-09-25: the regime/pooled row shape is defined next to the code
    # that builds it. Imported here rather than at module scope because score/regime_rows.py
    # imports build_row from this module — a top-level import either way would be a cycle.
    from research.strategy_hunt.score.regime_rows import REGIME_ROW_KEYS

    out = {"parquet": str(pq), "json": ""}
    if regime_rows:
        rpq = d / f"fit_regime_{date}.parquet"
        pd.DataFrame(regime_rows, columns=list(REGIME_ROW_KEYS)).to_parquet(rpq, index=False)
        out["regime_parquet"] = str(rpq)
    if pooled_rows:
        ppq = d / f"fit_pooled_{date}.parquet"
        pd.DataFrame(pooled_rows, columns=list(REGIME_ROW_KEYS)).to_parquet(ppq, index=False)
        out["pooled_parquet"] = str(ppq)

    js = d / "strategy_fit.json"
    _atomic_write_text(js, json.dumps({"computed_at": computed_at, "floors": FLOORS, "rows": nested,
                                       "regime_rows": regime_nested, "pooled_rows": pooled_flat}, indent=1))
    out["json"] = str(js)
    if engine is not None:
        out["db_rows"] = str(upsert_postgres(rows, engine))
    return out


def _regime_key(row: dict) -> str:
    return f"{row['strategy']}:{row['side']}:{row.get('regime', '')}"


def load_snapshot(root: Path) -> dict:
    js = fit_dir(root) / "strategy_fit.json"
    empty = {"computed_at": None, "floors": {}, "rows": {}, "regime_rows": {}, "pooled_rows": {}}
    if not js.exists():
        return empty
    snap = json.loads(js.read_text())
    if not isinstance(snap, dict):
        return empty
    # Snapshots written before the regime lanes existed have neither key.
    return {**{"regime_rows": {}, "pooled_rows": {}}, **snap}


_DDL = f"""CREATE TABLE IF NOT EXISTS {TABLE} (
  symbol TEXT NOT NULL, strategy TEXT NOT NULL, side TEXT NOT NULL, computed_at TEXT NOT NULL, window_end TEXT,
  n INTEGER, sessions INTEGER, exp_r REAL, exp_r_lb REAL, t_stat REAL, pf REAL, wr REAL, max_dd_r REAL,
  stability REAL, decay REAL, score REAL, verdict TEXT, failed_floors TEXT, params_json TEXT,
  regime_exp_json TEXT, source TEXT,
  PRIMARY KEY (symbol, strategy, side, computed_at))"""
_UPSERT = f"""INSERT INTO {TABLE} ({", ".join(ROW_KEYS)}) VALUES ({", ".join(":" + k for k in ROW_KEYS)})
ON CONFLICT (symbol, strategy, side, computed_at) DO UPDATE SET
{", ".join(f"{k} = excluded.{k}" for k in ROW_KEYS if k not in ("symbol", "strategy", "side", "computed_at"))}"""


def upsert_postgres(rows: list[dict], engine: Any) -> int:
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text(_DDL))
        for r in rows:
            conn.execute(text(_UPSERT), {k: r.get(k) for k in ROW_KEYS})
    return len(rows)
