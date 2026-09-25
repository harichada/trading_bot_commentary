"""Build fit rows from out-of-sample trades and persist them (Parquet + atomic JSON + SQL upsert)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from research.strategy_hunt.models import SimTrade
from research.strategy_hunt.score.metrics import compute_metrics, per_regime
from research.strategy_hunt.score.verdict import FLOORS, verdict

ROW_KEYS = ("symbol", "strategy", "side", "computed_at", "window_end", "n", "sessions", "exp_r", "exp_r_lb",
            "t_stat", "pf", "wr", "max_dd_r", "stability", "decay", "score", "verdict", "failed_floors",
            "params_json", "regime_exp_json", "source")
PF_INF = 999.0
TABLE = "strategy_fit"


def build_row(symbol: str, strategy: str, side: str, trades: list[SimTrade], n_windows: int,
              latest_params: dict, computed_at: str, window_end: str) -> dict[str, Any]:
    m = compute_metrics(trades, n_windows)
    v, failed = verdict(m)
    score = m.exp_r * math.sqrt(m.stability) if m.exp_r > 0 else 0.0
    row = {
        "symbol": symbol, "strategy": strategy, "side": side, "computed_at": computed_at, "window_end": window_end,
        "n": m.n, "sessions": m.sessions, "exp_r": m.exp_r, "exp_r_lb": m.exp_r_lb, "t_stat": m.t_stat,
        "pf": PF_INF if math.isinf(m.pf) else m.pf, "wr": m.wr,
        "max_dd_r": m.max_dd_r, "stability": m.stability, "decay": m.decay, "score": float(score), "verdict": v,
        "failed_floors": json.dumps(failed), "params_json": json.dumps(latest_params, sort_keys=True),
        "regime_exp_json": json.dumps(per_regime(trades), sort_keys=True), "source": "walkforward",
    }
    return {k: row[k] for k in ROW_KEYS}


def fit_dir(root: Path) -> Path:
    return Path(root) / "fit"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_fit(rows: list[dict], root: Path, computed_at: str, engine: Any = None) -> dict[str, str]:
    d = fit_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    date = str(computed_at)[:10]
    pq = d / f"fit_{date}.parquet"
    pd.DataFrame(rows, columns=list(ROW_KEYS)).to_parquet(pq, index=False)
    nested: dict[str, dict[str, dict]] = {}
    for r in rows:
        nested.setdefault(r["symbol"], {})[f"{r['strategy']}:{r['side']}"] = r
    js = d / "strategy_fit.json"
    _atomic_write_text(js, json.dumps({"computed_at": computed_at, "floors": FLOORS, "rows": nested}, indent=1))
    out = {"parquet": str(pq), "json": str(js)}
    if engine is not None:
        out["db_rows"] = str(upsert_postgres(rows, engine))
    return out


def load_snapshot(root: Path) -> dict:
    js = fit_dir(root) / "strategy_fit.json"
    if not js.exists():
        return {"computed_at": None, "floors": {}, "rows": {}}
    return json.loads(js.read_text())


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
