"""Build fit rows from out-of-sample trades and persist them (Parquet + atomic JSON + SQL upsert)."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from research.strategy_hunt.models import FitMetrics, SimTrade
from research.strategy_hunt.score.metrics import compute_metrics, per_regime
from research.strategy_hunt.score.verdict import FLOORS, verdict

logger = logging.getLogger("strategy_hunt.fit.writer")

ROW_KEYS = ("symbol", "strategy", "side", "computed_at", "window_end", "n", "sessions", "exp_r", "exp_r_lb",
            "t_stat", "pf", "wr", "max_dd_r", "stability", "decay", "score", "verdict", "failed_floors",
            "params_json", "regime_exp_json", "source",
            # v-playbook-desk-b1-5: task 9's two pooled metrics and task 10's baseline columns.
            # Appended, never interleaved, so an older Parquet/SQL column order still lines up.
            "t_stat_session", "recovery", "baseline_exp_r", "baseline_n", "excess_r")
PF_INF = 999.0
TABLE = "strategy_fit"
# The task-10 columns, with the SQL types an ALTER has to use when an existing strategy_fit
# table predates them (see `_add_missing_columns`).
NEW_SQL_COLUMNS = (("t_stat_session", "REAL"), ("recovery", "REAL"), ("baseline_exp_r", "REAL"),
                   ("baseline_n", "INTEGER"), ("excess_r", "REAL"))


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
        "t_stat_session": m.t_stat_session, "recovery": m.recovery,
        # No baseline is measured for an unconditional or per-symbol regime row: NaN and 0,
        # never a silent 0.0 excess that would read as "matched the shuffled control exactly".
        # `score.regime_rows.pooled_rows` overwrites these three for pooled rows.
        "baseline_exp_r": float("nan"), "baseline_n": 0, "excess_r": float("nan"),
    }
    return {k: row[k] for k in ROW_KEYS}


def fit_dir(root: Path) -> Path:
    return Path(root) / "fit"


def json_safe(obj: Any) -> Any:
    """`obj` with every non-finite float replaced by None.

    `recovery` is +inf for a lane that never drew down and `excess_r` is NaN with no
    baseline; `json.dumps` writes those as bare `Infinity`/`NaN`, which is not valid JSON
    (FastAPI's encoder and every non-Python reader reject it). `None` is the honest wire
    value for "not measured" — the same choice `core.strategy_fit_rows.to_json_dict` makes
    on the read side."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def _dumps(payload: Any) -> str:
    """Strict JSON: `allow_nan=False` makes a float `json_safe` somehow missed a loud
    failure here rather than a snapshot no other reader can parse."""
    return json.dumps(json_safe(payload), indent=1, allow_nan=False)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_fit(rows: list[dict], root: Path, computed_at: str, engine: Any = None,
              regime_rows: Optional[list[dict]] = None, pooled_rows: Optional[list[dict]] = None,
              baseline_pooled_rows: Optional[list[dict]] = None) -> dict[str, str]:
    """Persist the fit rows: Parquet per row family plus one atomic JSON snapshot.

    `regime_rows` / `pooled_rows` (v-playbook-desk-b1-2026-09-25) and
    `baseline_pooled_rows` (v-playbook-desk-b1-5 task 10) are additive and optional — a
    caller that passes none writes exactly the snapshot shape older readers expect, with
    the new keys present but empty. They are kept out of `fit_<date>.parquet` and out of
    the SQL table on purpose: that table's primary key is (symbol, strategy, side,
    computed_at), which pooled rows would collide on across regimes.

    `baseline_pooled_rows` are the *shuffled control's* pooled rows (`source="baseline"`).
    They live only under their own snapshot key: the live gate reads `pooled_rows`, and a
    baseline row appearing there would let the control's drift authorise a live lane.
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
    baseline_flat = {_regime_key(r): r for r in baseline_pooled_rows or []}

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
    if baseline_pooled_rows:
        bpq = d / f"fit_baseline_{date}.parquet"
        pd.DataFrame(baseline_pooled_rows, columns=list(REGIME_ROW_KEYS)).to_parquet(bpq, index=False)
        out["baseline_parquet"] = str(bpq)

    js = d / "strategy_fit.json"
    _atomic_write_text(js, _dumps({"computed_at": computed_at, "floors": FLOORS, "rows": nested,
                                   "regime_rows": regime_nested, "pooled_rows": pooled_flat,
                                   "baseline_pooled_rows": baseline_flat}))
    out["json"] = str(js)
    if engine is not None:
        out["db_rows"] = str(upsert_postgres(rows, engine))
    return out


def _regime_key(row: dict) -> str:
    return f"{row['strategy']}:{row['side']}:{row.get('regime', '')}"


def load_snapshot(root: Path) -> dict:
    js = fit_dir(root) / "strategy_fit.json"
    empty = {"computed_at": None, "floors": {}, "rows": {}, "regime_rows": {}, "pooled_rows": {},
             "baseline_pooled_rows": {}}
    if not js.exists():
        return empty
    snap = json.loads(js.read_text())
    if not isinstance(snap, dict):
        return empty
    # Snapshots written before the regime/baseline lanes existed have none of those keys.
    return {**{"regime_rows": {}, "pooled_rows": {}, "baseline_pooled_rows": {}}, **snap}


_DDL = f"""CREATE TABLE IF NOT EXISTS {TABLE} (
  symbol TEXT NOT NULL, strategy TEXT NOT NULL, side TEXT NOT NULL, computed_at TEXT NOT NULL, window_end TEXT,
  n INTEGER, sessions INTEGER, exp_r REAL, exp_r_lb REAL, t_stat REAL, pf REAL, wr REAL, max_dd_r REAL,
  stability REAL, decay REAL, score REAL, verdict TEXT, failed_floors TEXT, params_json TEXT,
  regime_exp_json TEXT, source TEXT, t_stat_session REAL, recovery REAL, baseline_exp_r REAL,
  baseline_n INTEGER, excess_r REAL,
  PRIMARY KEY (symbol, strategy, side, computed_at))"""
_UPSERT = f"""INSERT INTO {TABLE} ({", ".join(ROW_KEYS)}) VALUES ({", ".join(":" + k for k in ROW_KEYS)})
ON CONFLICT (symbol, strategy, side, computed_at) DO UPDATE SET
{", ".join(f"{k} = excluded.{k}" for k in ROW_KEYS if k not in ("symbol", "strategy", "side", "computed_at"))}"""


def _add_missing_columns(conn: Any) -> list[str]:
    """Add whichever of `NEW_SQL_COLUMNS` an already-existing `strategy_fit` table lacks.

    `CREATE TABLE IF NOT EXISTS` is a no-op on a table created before v-playbook-desk-b1-5,
    so the upsert would fail on unknown columns. Introspecting and issuing a plain
    `ALTER TABLE ... ADD COLUMN` (rather than Postgres-only `ADD COLUMN IF NOT EXISTS`)
    keeps this working on both Postgres and the SQLite engines the tests use."""
    from sqlalchemy import inspect, text

    have = {c["name"] for c in inspect(conn).get_columns(TABLE)}
    added = []
    for name, sql_type in NEW_SQL_COLUMNS:
        if name not in have:
            conn.execute(text(f"ALTER TABLE {TABLE} ADD COLUMN {name} {sql_type}"))
            added.append(name)
    return added


def upsert_postgres(rows: list[dict], engine: Any) -> int:
    from sqlalchemy import text

    with engine.begin() as conn:
        conn.execute(text(_DDL))
        added = _add_missing_columns(conn)
        if added:
            logger.info("strategy_fit: added columns %s to the existing table", ", ".join(added))
        for r in rows:
            conn.execute(text(_UPSERT), {k: r.get(k) for k in ROW_KEYS})
    return len(rows)
