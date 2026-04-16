"""Retrospective precision tracker for meta_shadow evaluations.

Reads every ``component=meta_shadow`` audit line from the trading bot
log, looks up the subsequent price path for each eval from PostgreSQL,
applies the same triple-barrier rule the meta-model was trained against,
and reports live win-rate per probability threshold so we can compare
against the backtest:

    backtest thr=0.50: 54% win  thr=0.65: 62%  thr=0.70: 68%  thr=0.75: 72%

A meta_shadow eval is considered "resolved" once enough bars exist after
its timestamp to evaluate the time barrier (default 60 bars on 5-min).
Unresolved evals are listed but excluded from the precision summary.

Usage:
    python analyze_shadow_outcomes.py
    python analyze_shadow_outcomes.py --log /path/to/trading_bot.log
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ml.meta_labels import apply_meta_triple_barrier
from train_ml_model import DEFAULT_DSN
from train_ml_model_v2 import _compute_raw_atr


load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("analyze_shadow")

DEFAULT_LOG = "trading_bot.log"
DEFAULT_MODEL = "ml_meta_model.pkl"
DEFAULT_OUTPUT = "ml_shadow_outcomes.json"

_AUDIT_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*"
    r"component=meta_shadow.*"
    r"symbol=(?P<symbol>\S+).*"
    r"reason=(?P<reason>\S+).*"
    r"proba=(?P<proba>[\d.]+)"
)


def parse_shadow_log(log_path: str) -> pd.DataFrame:
    """Extract every meta_shadow eval into a dataframe."""
    rows = []
    with open(log_path, "r") as fh:
        for line in fh:
            m = _AUDIT_RE.search(line)
            if not m:
                continue
            d = m.groupdict()
            side = 1 if d["reason"].endswith("_long") else -1
            rows.append({
                "ts": pd.Timestamp(d["ts"]),
                "symbol": d["symbol"],
                "side": side,
                "proba": float(d["proba"]),
            })
    return pd.DataFrame(rows)


def resolve_outcomes(
    evals: pd.DataFrame,
    dsn: str,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
    frequency: int,
) -> pd.DataFrame:
    """For each eval, label it via the same triple-barrier rule used in training."""
    from data_providers.postgres import PostgresDataProvider

    if evals.empty:
        return evals.assign(bin=pd.NA, ret=pd.NA, resolved=False)

    provider = PostgresDataProvider(dsn)
    out_rows = []

    for symbol, group in evals.groupby("symbol"):
        try:
            df = provider.get_market_data(
                symbol, period_type="month", period=2,
                frequency_type="minute", frequency=frequency,
            )
        except Exception as exc:
            logger.error("load_failed symbol=%s err=%s", symbol, exc)
            for _, row in group.iterrows():
                out_rows.append({**row.to_dict(), "bin": None,
                                 "ret": None, "resolved": False})
            continue

        if df.empty:
            for _, row in group.iterrows():
                out_rows.append({**row.to_dict(), "bin": None,
                                 "ret": None, "resolved": False})
            continue

        atr = _compute_raw_atr(df, window=14)
        last_bar = df.index.max()

        for _, row in group.iterrows():
            try:
                snap_loc = df.index.get_indexer([row["ts"]], method="nearest")[0]
            except Exception:
                out_rows.append({**row.to_dict(), "bin": None,
                                 "ret": None, "resolved": False})
                continue

            entry_ts = df.index[snap_loc]
            if snap_loc + max_holding >= len(df):
                out_rows.append({**row.to_dict(), "bin": None,
                                 "ret": None, "resolved": False})
                continue

            atr_at_entry = atr.iloc[snap_loc]
            if atr_at_entry <= 0 or pd.isna(atr_at_entry):
                out_rows.append({**row.to_dict(), "bin": None,
                                 "ret": None, "resolved": False})
                continue

            single_event = pd.DataFrame(
                {"side": [int(row["side"])]}, index=[entry_ts],
            )
            label = apply_meta_triple_barrier(
                prices=df["Close"], events=single_event, atr=atr,
                pt_mult=pt_mult, sl_mult=sl_mult, max_holding=max_holding,
            )
            out_rows.append({
                **row.to_dict(),
                "bin": int(label.iloc[0]["bin"]),
                "ret": float(label.iloc[0]["ret"]),
                "resolved": True,
                "last_bar_available": last_bar,
            })

    return pd.DataFrame(out_rows)


def threshold_summary(resolved: pd.DataFrame, thresholds: list[float]) -> list[dict[str, Any]]:
    rows = []
    for thr in thresholds:
        gate = resolved[resolved["proba"] >= thr]
        if gate.empty:
            rows.append({"threshold": thr, "n": 0, "win_rate": None,
                         "avg_return": None})
            continue
        rows.append({
            "threshold": thr,
            "n": int(len(gate)),
            "win_rate": round(float((gate["bin"] == 1).mean()), 4),
            "avg_return": round(float(gate["ret"].mean()), 5),
        })
    return rows


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log", default=DEFAULT_LOG)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Meta-model bundle path (used to read barrier config)")
    p.add_argument("--dsn", default=os.environ.get("POSTGRES_DSN", DEFAULT_DSN))
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--pt-mult", type=float)
    p.add_argument("--sl-mult", type=float)
    p.add_argument("--max-holding", type=int)
    p.add_argument("--frequency", type=int)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    bundle = joblib.load(args.model)
    cfg = bundle["config"]
    pt_mult = args.pt_mult if args.pt_mult is not None else cfg["pt_mult"]
    sl_mult = args.sl_mult if args.sl_mult is not None else cfg["sl_mult"]
    max_holding = args.max_holding if args.max_holding is not None else cfg["max_holding"]
    frequency = args.frequency if args.frequency is not None else cfg["frequency"]
    logger.info("config pt=%.2f sl=%.2f mh=%d freq=%dmin",
                pt_mult, sl_mult, max_holding, frequency)

    evals = parse_shadow_log(args.log)
    logger.info("parsed_evals n=%d symbols=%s",
                len(evals), evals["symbol"].unique().tolist() if not evals.empty else [])
    if evals.empty:
        logger.warning("no_evals_found in %s", args.log)
        return 0

    resolved = resolve_outcomes(
        evals, args.dsn, pt_mult, sl_mult, max_holding, frequency,
    )
    n_resolved = int(resolved["resolved"].sum())
    n_pending = len(resolved) - n_resolved
    logger.info("resolution_done resolved=%d pending=%d", n_resolved, n_pending)

    finished = resolved[resolved["resolved"]].copy()
    if finished.empty:
        logger.warning("no resolved evals yet — wait for time-barrier to elapse")
        Path(args.output).write_text(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "n_evals": int(len(resolved)),
            "n_resolved": 0,
            "n_pending": int(len(resolved)),
            "message": "no resolved evals — wait for time-barrier",
        }, indent=2, default=str))
        return 0

    base_win = float((finished["bin"] == 1).mean())
    base_avg_ret = float(finished["ret"].mean())
    sweep = threshold_summary(
        finished, thresholds=[0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    )

    for row in sweep:
        if row["win_rate"] is not None:
            logger.info(
                "thr=%.2f n=%d live_win=%.3f avg_ret=%+.4f",
                row["threshold"], row["n"], row["win_rate"], row["avg_return"],
            )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {"pt_mult": pt_mult, "sl_mult": sl_mult,
                   "max_holding": max_holding, "frequency": frequency},
        "n_evals_total": int(len(resolved)),
        "n_resolved": n_resolved,
        "n_pending": n_pending,
        "base_live_win_rate": round(base_win, 4),
        "base_avg_return": round(base_avg_ret, 5),
        "by_symbol": (
            finished.groupby("symbol")["bin"].agg(["count", "mean"]).to_dict("index")
        ),
        "by_side": (
            finished.groupby("side")["bin"].agg(["count", "mean"]).to_dict("index")
        ),
        "threshold_sweep": sweep,
        "backtest_reference": {
            "thr_0.50": {"precision": 0.542, "n": 44960},
            "thr_0.65": {"precision": 0.616, "n": 6841},
            "thr_0.70": {"precision": 0.681, "n": 2829},
            "thr_0.75": {"precision": 0.723, "n": 1344},
        },
    }
    Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
    logger.info("written %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
