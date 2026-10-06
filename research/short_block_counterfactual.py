"""Would the blocked day-trade shorts have made money? (2026-10-06)

The live short lane blocks every evaluation in a `risk_on` regime before any
setup is detected (`risk_on_hard_block_short`), so blocked days leave no
would-be trades behind. The nearest evidence is:

1. the lane's own shadow ledger (`data/day_trade_short_shadow.ndjson`): the
   setups it logged on days it was allowed to look (mixed / risk_off), each
   with its hypothetical stop and target;
2. the legacy mean-reversion short shadow (`data/shadow_short_log.ndjson`,
   `overbought_fade`), which keeps logging in every regime, including risk_on;
3. the five-year walk-forward result for the same family by entry-time regime
   (research/reports/strategy_hunt/2026-09-25-strategy-hunt-b1-5.md §B1.6).

This script resolves 1 and 2 against 5-minute bars (research cache, yfinance
for anything after the cache's last day), labels each entry with the
pipeline's entry-time regime, and prints the counterfactual by regime.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.day_trade_short_resolver import ShadowEntry, make_file_bars_loader, resolve_entry  # noqa: E402
from research.strategy_hunt.data.bars import load_bars5, load_daily  # noqa: E402
from research.strategy_hunt.data.regime import build_regime_lookup  # noqa: E402

CACHE = ROOT / "data" / "strategy_hunt"
LEDGERS = {"day_trade_short_shadow": ROOT / "data" / "day_trade_short_shadow.ndjson",
           "mean_rev_short_shadow": ROOT / "data" / "shadow_short_log.ndjson"}


def load_entries(path: Path, source: str) -> list[ShadowEntry]:
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        d.setdefault("strategy", source)
        d.setdefault("entry_pattern", d.get("reason", source))
        d.setdefault("signal_type", "SHORT")
        for k in ("rs_vs_spy", "volume_ratio", "adx", "low_20", "high_20", "sma_20", "sma_50", "macd", "macd_signal",
                  "atr", "rsi", "rr_ratio"):
            d.setdefault(k, 0.0)
        try:
            out.append(ShadowEntry.from_dict(d))
        except Exception as exc:
            print(f"skip row ({source}): {exc}", file=sys.stderr)
    return out


_YF_CACHE: dict = {}


def yf_bars(symbol: str, since: datetime, lookforward_days: int = 5) -> pd.DataFrame:
    import yfinance as yf
    if symbol not in _YF_CACHE:
        h = yf.Ticker(symbol).history(period="60d", interval="5m")
        if h is None or h.empty:
            _YF_CACHE[symbol] = pd.DataFrame()
        else:
            h = h.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            h.index = pd.DatetimeIndex(h.index).tz_convert("America/New_York").tz_localize(None)
            _YF_CACHE[symbol] = h
    h = _YF_CACHE[symbol]
    if h.empty:
        return h
    s = pd.Timestamp(since).tz_localize(None) if pd.Timestamp(since).tzinfo else pd.Timestamp(since)
    out = h[(h.index > s) & (h.index <= s + pd.Timedelta(days=lookforward_days))].copy()
    out.columns = [c.capitalize() for c in out.columns]      # the resolver reads Open/High/Low/Close
    return out


def main() -> int:
    file_loader = make_file_bars_loader(CACHE / "bars5")
    spy5, vix_d = load_bars5(CACHE, "SPY"), load_daily(CACHE, "VIX")
    regime_of = build_regime_lookup(spy5, vix_d)
    cache_end = spy5.index.max()

    def loader(symbol, since, lookforward_days=5):
        since_naive = pd.Timestamp(since).tz_convert("America/New_York").tz_localize(None) if pd.Timestamp(since).tzinfo else pd.Timestamp(since)
        if since_naive <= cache_end - pd.Timedelta(days=lookforward_days):
            df = file_loader(symbol, since, lookforward_days)
            if df is not None and not df.empty:
                return df
        return yf_bars(symbol, since_naive, lookforward_days)

    rows = []
    for source, path in LEDGERS.items():
        if not path.exists():
            continue
        for e in load_entries(path, source):
            ts_et = pd.Timestamp(e.timestamp).tz_convert("America/New_York").tz_localize(None) if pd.Timestamp(e.timestamp).tzinfo else pd.Timestamp(e.timestamp)
            try:
                reg = regime_of(ts_et) if ts_et <= cache_end else "unknown"
            except Exception:
                reg = "unknown"
            try:
                from dataclasses import replace as _replace
                t = resolve_entry(_replace(e, timestamp=ts_et.to_pydatetime()), loader)
            except Exception as exc:
                print(f"resolve failed {e.symbol} {ts_et}: {exc}", file=sys.stderr)
                continue
            if t.exit_time is None:
                continue
            rows.append({"source": source, "symbol": e.symbol, "ts": ts_et, "day": ts_et.strftime("%Y-%m-%d"),
                         "regime": reg, "r": t.r_multiple, "ret_pct": t.return_pct, "exit": t.exit_reason,
                         "hold_bars": t.hold_bars, "pattern": e.entry_pattern})
    df = pd.DataFrame(rows)
    if df.empty:
        print("no resolved trades")
        return 0

    def stats(g):
        r = g["r"].to_numpy(float)
        wins, losses = r[r > 0], r[r <= 0]
        pf = wins.sum() / -losses.sum() if losses.size and losses.sum() < 0 else float("inf")
        return pd.Series({"n": len(r), "win": (r > 0).mean(), "avg_r": r.mean(), "sum_r": r.sum(), "pf": pf,
                          "avg_ret_pct": g["ret_pct"].mean()})

    pd.set_option("display.width", 160)
    print("=== resolved short signals by source x regime")
    print(df.groupby(["source", "regime"]).apply(stats).round(3).to_string())
    print("\n=== by regime (all sources)")
    print(df.groupby("regime").apply(stats).round(3).to_string())
    print("\n=== by day (last 15)")
    print(df.groupby("day").apply(stats).round(3).tail(15).to_string())
    print("\n=== exit reasons")
    print(df.groupby(["regime", "exit"]).size().unstack(fill_value=0).to_string())
    out = ROOT / "research" / "reports" / "rsi" / "2026-10-06-short-block-counterfactual.parquet"
    df.to_parquet(out, index=False)
    print("saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
