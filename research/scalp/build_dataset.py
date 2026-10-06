"""S1: build the scalp dataset from the tick-rule archive minute bins.
One parquet per symbol under data/scalp/<SYM>.parquet (features + all
label cells), plus a manifest. Causal, per-session; the print-rate baseline
is the symbol's median minute print count over the archive."""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.scalp.dataset import build_session  # noqa: E402
from research.strategy_hunt.data.bars import load_bars5, load_daily  # noqa: E402
from research.strategy_hunt.data.regime import build_regime_lookup  # noqa: E402

ARCHIVE = ROOT / "data" / "micro_archive"
OUT = ROOT / "data" / "scalp"
PRICE_COLS = ("open", "high", "low", "close")


def clean_bars(symbol: str) -> pd.DataFrame:
    """Official 1-minute OHLC (Postgres `minute_bars`, naive ET timestamps) for
    the archive window. The archive's print-derived highs/lows/closes are far
    too noisy to label scalps (2026-10-06 control: median minute range 99 bp
    vs 18 bp official; the model learned stray prints). Prices and labels come
    from here; only the FLOW columns come from the archive bins."""
    from sqlalchemy import create_engine, text
    from backtest.engine import DEFAULT_DSN
    e = create_engine(os.environ.get("POSTGRES_DSN", DEFAULT_DSN))
    with e.connect() as c:
        c.execute(text("set statement_timeout = 300000"))
        df = pd.read_sql(text("select ts, open, high, low, close, volume from minute_bars where symbol = :s "
                              "and ts >= '2024-10-01' and ts < '2025-03-12' order by ts"), c, params={"s": symbol})
    df["ts"] = pd.to_datetime(df["ts"])
    df = df[(df["ts"].dt.hour * 60 + df["ts"].dt.minute).between(9 * 60 + 30, 15 * 60 + 59)]
    for col in PRICE_COLS:
        df[col] = df[col].astype(float)
    return df.rename(columns={"ts": "t_open"})


def merge_flow(bars: pd.DataFrame, bins: pd.DataFrame) -> pd.DataFrame:
    """Clean bars + archive flow on the minute; minutes without prints get zero flow."""
    flow_cols = ["buy_vol", "sell_vol", "vol", "n_trades", "large_vol", "large_buy", "large_sell"]
    f = bins[["t_open"] + flow_cols].copy()
    f["t_open"] = pd.to_datetime(f["t_open"])
    out = bars.merge(f, on="t_open", how="left")
    out[flow_cols] = out[flow_cols].fillna(0.0)
    out["vwap"] = out["close"]          # the official bars carry no VWAP; the print VWAP is noisy
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    regime_of = build_regime_lookup(load_bars5(ROOT / "data/strategy_hunt", "SPY"), load_daily(ROOT / "data/strategy_hunt", "VIX"))
    manifest = {}
    for sym_dir in sorted(p for p in ARCHIVE.iterdir() if p.is_dir()):
        sym = sym_dir.name
        files = sorted(f for f in sym_dir.glob("*.parquet") if os.path.getsize(f) > 2000)
        if not files:
            continue
        t0 = time.time()
        sessions = []
        rates = []
        for f in files:
            b = pd.read_parquet(f)
            if b.empty or "t_open" not in b.columns or len(b) < 60:
                continue
            rates.append(b["n_trades"].median())
        med_rate = float(np.median(rates)) if rates else None
        try:
            bars_all = clean_bars(sym)
        except Exception as exc:
            print(f"{sym}: clean bars unavailable: {exc}", file=sys.stderr)
            continue
        bars_all["day"] = bars_all["t_open"].dt.strftime("%Y-%m-%d")
        for f in files:
            b = pd.read_parquet(f)
            if b.empty or "t_open" not in b.columns or len(b) < 60:
                continue
            day = f.stem
            bars = bars_all[bars_all["day"] == day].drop(columns="day")
            if len(bars) < 200:
                continue
            b = merge_flow(bars, b)
            s = build_session(b, sym, print_rate_median=med_rate)
            s["session"] = day
            try:
                s["regime"] = s["t_open"].map(lambda t: regime_of(pd.Timestamp(t)))
            except Exception:
                s["regime"] = "unknown"
            sessions.append(s)
        if not sessions:
            continue
        df = pd.concat(sessions, ignore_index=True)
        df.to_parquet(OUT / f"{sym}.parquet", index=False)
        manifest[sym] = {"rows": int(len(df)), "sessions": len(sessions), "secs": round(time.time() - t0, 1)}
        print(f"{sym}: rows={len(df)} sessions={len(sessions)} {time.time()-t0:.0f}s", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("DONE", sum(m["rows"] for m in manifest.values()), "rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
