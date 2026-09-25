"""Calibrate slippage from the bot's real fills (bot_trades vs accepted bot_decisions)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("strategy_hunt.calibrate")
BUCKETS = (("mega", 1e9), ("large", 2e8), ("mid", 2e7))


def _bucket(adv: float) -> str:
    for name, floor in BUCKETS:
        if adv >= floor:
            return name
    return "small"


def _extract_entry_price(details_json_str: str) -> float:
    """Extract entry price from details_json JSON string. Primary: JSON 'entry' key. Fallback: None. Returns NaN on failure."""
    if not isinstance(details_json_str, str) or not details_json_str.strip():
        return float("nan")
    try:
        obj = json.loads(details_json_str)
        if isinstance(obj, dict) and "entry" in obj:
            val = obj["entry"]
            return float(val) if val is not None else float("nan")
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return float("nan")


def realized_slippage(fills: pd.DataFrame) -> pd.DataFrame:
    sign = np.where(fills["side"].str.lower() == "long", 1.0, -1.0)
    slip = sign * (fills["fill_price"] - fills["signal_price"]) / fills["signal_price"]
    return fills.assign(slip_frac=slip.astype(float), bucket=[_bucket(float(a)) for a in fills["adv_dollars"]])


def calibration_summary(slips: pd.DataFrame, min_n: int = 20) -> dict[str, Any]:
    if slips.empty:
        return {"n": 0, "median_frac": float("nan"), "p75_frac": float("nan"), "by_bucket": {}, "usable": False}
    adverse = slips["slip_frac"].clip(lower=0.0)
    by = {b: {"n": int(len(g)), "median_frac": float(g["slip_frac"].clip(lower=0.0).median())} for b, g in slips.groupby("bucket")}
    return {"n": int(len(slips)), "median_frac": float(adverse.median()), "p75_frac": float(adverse.quantile(0.75)),
            "by_bucket": by, "usable": bool(len(slips) >= min_n)}


def load_fill_pairs(engine: Any) -> pd.DataFrame:
    """Join fills to the accepted decision that produced them. Empty frame on any failure."""
    from sqlalchemy import text

    cols = ["symbol", "side", "signal_price", "fill_price", "adv_dollars"]
    try:
        with engine.connect() as conn:
            trades = pd.read_sql(text("SELECT symbol, side, entry_time, entry_price FROM bot_trades WHERE entry_price > 0"), conn)
            decisions = pd.read_sql(text("SELECT symbol, ts, details_json FROM bot_decisions WHERE action = 'accepted'"), conn)
            adv = pd.read_sql(text("""SELECT symbol, avg(close*volume) AS adv FROM (
                                        SELECT symbol, close, volume, row_number() OVER (PARTITION BY symbol ORDER BY date DESC) rn
                                        FROM daily_bars) t WHERE rn <= 60 GROUP BY symbol"""), conn)
    except Exception as exc:
        logger.warning("load_fill_pairs failed: %s", exc)
        return pd.DataFrame(columns=cols)
    if trades.empty or decisions.empty:
        return pd.DataFrame(columns=cols)
    decisions = decisions.assign(signal_price=decisions["details_json"].apply(_extract_entry_price))
    decisions = decisions.dropna(subset=["signal_price"])
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    decisions["ts"] = pd.to_datetime(decisions["ts"])
    merged = pd.merge_asof(trades.sort_values("entry_time"), decisions.sort_values("ts"), left_on="entry_time", right_on="ts",
                           by="symbol", direction="backward", tolerance=pd.Timedelta(minutes=10)).dropna(subset=["signal_price"])
    merged = merged.merge(adv, on="symbol", how="left").fillna({"adv": 0.0})
    return merged.rename(columns={"entry_price": "fill_price", "adv": "adv_dollars"})[cols]


def write_calibration(root: Path, summary: dict) -> Path:
    p = Path(root) / "calibration.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=1))
    return p


def load_calibration(root: Path) -> Optional[dict]:
    p = Path(root) / "calibration.json"
    return json.loads(p.read_text()) if p.exists() else None


def slippage_override_from(root: Path) -> Optional[float]:
    c = load_calibration(root)
    return float(c["median_frac"]) if c and c.get("usable") else None
