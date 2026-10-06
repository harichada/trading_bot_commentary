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
        for f in files:
            b = pd.read_parquet(f)
            if b.empty or "t_open" not in b.columns or len(b) < 60:
                continue
            b = b[(b["t_open"].dt.hour * 60 + b["t_open"].dt.minute) >= 9 * 60 + 30]
            s = build_session(b, sym, print_rate_median=med_rate)
            s["session"] = f.stem
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
