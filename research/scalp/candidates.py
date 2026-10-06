"""Which names are scalpable? A structural screen, not a signal.

For every symbol in the research cache (5-minute bars, last N sessions) and,
where available, the official 1-minute bars: price, dollar volume, how much a
stock moves in 5 minutes relative to what a round trip costs (1 tick + 2 bp
slippage per side, the floor; measured spreads come from the live book later),
consistency of trading, tail risk, and short-horizon return autocorrelation
(negative = mean-reverting at the scalp horizon, positive = momentum).

Score = room (median 5-min range / round-trip cost) x liquidity gate x
consistency, with tail and price floors. Output: research/reports/scalp/<date>-candidates.md
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.strategy_hunt.data.bars import load_bars5  # noqa: E402

CACHE = ROOT / "data" / "strategy_hunt"
SESSIONS = 60
SLIP_BP = 2.0          # per side, on top of the half-spread proxy
MIN_PRICE = 15.0
MIN_DOLLAR_VOL = 150e6   # per session


def metrics(symbol: str) -> dict | None:
    b = load_bars5(CACHE, symbol)
    if b.empty:
        return None
    mod = b.index.hour * 60 + b.index.minute
    b = b[(mod >= 9 * 60 + 30) & (mod <= 15 * 60 + 55)]
    days = sorted(set(b.index.date))[-SESSIONS:]
    b = b[np.isin(b.index.date, days)]
    if len(days) < 40 or len(b) < 40 * 70:
        return None
    close = b["close"].to_numpy(float)
    rng_bp = ((b["high"] - b["low"]) / b["close"] * 1e4).to_numpy(float)
    ret = np.diff(np.log(close))
    px = float(np.median(close[-78 * 5:]))
    tick_bp = 0.01 / px * 1e4
    cost_rt_bp = 2 * (tick_bp / 2 + SLIP_BP) + tick_bp      # cross half a spread twice + slippage + one tick of adverse
    dvol = (b["close"] * b["volume"]).groupby(b.index.date).sum()
    bars_per_day = b.groupby(b.index.date).size()
    active = (b["volume"] > 0).groupby(b.index.date).mean()
    rng_med = float(np.nanmedian(rng_bp))
    rng_p25 = float(np.nanpercentile(rng_bp, 25))
    ac1 = float(pd.Series(ret).autocorr(1)) if len(ret) > 100 else np.nan
    ac3 = float(pd.Series(ret).autocorr(3)) if len(ret) > 100 else np.nan
    tail = float(np.nanpercentile(np.abs(ret) * 1e4, 99.5))
    first_hour = b[(b.index.hour * 60 + b.index.minute) < 10 * 60 + 30]
    rng_open = float(np.nanmedian(((first_hour["high"] - first_hour["low"]) / first_hour["close"] * 1e4))) if len(first_hour) else np.nan
    return {"symbol": symbol, "price": px, "dollar_vol_m": float(dvol.median() / 1e6), "range5_bp": rng_med,
            "range5_p25_bp": rng_p25, "range5_open_bp": rng_open, "tick_bp": tick_bp, "cost_rt_bp": cost_rt_bp,
            "room": rng_med / cost_rt_bp, "room_p25": rng_p25 / cost_rt_bp, "active": float(active.median()),
            "bars_per_day": float(bars_per_day.median()), "tail_bp": tail, "ac1": ac1, "ac3": ac3, "sessions": len(days)}


def main() -> int:
    syms = sorted(p.stem for p in (CACHE / "bars5").glob("*.parquet"))
    rows = [m for s in syms if (m := metrics(s)) is not None]
    df = pd.DataFrame(rows)
    liq = (df["price"] >= MIN_PRICE) & (df["dollar_vol_m"] * 1e6 >= MIN_DOLLAR_VOL) & (df["active"] >= 0.97)
    df["score"] = np.where(liq, df["room_p25"] * np.sqrt(df["dollar_vol_m"].clip(upper=5000) / 1000.0), 0.0)
    df["liquid"] = liq
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    out = ROOT / "research" / "reports" / "scalp" / f"{datetime.now():%Y-%m-%d}-candidates.md"
    cols = ["symbol", "price", "dollar_vol_m", "range5_bp", "range5_p25_bp", "range5_open_bp", "cost_rt_bp", "room", "room_p25", "tail_bp", "ac1", "ac3", "score"]
    def table(d):
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in d.iterrows():
            lines.append("| " + " | ".join(
                f"{r[c]:.0f}" if c in ("price", "dollar_vol_m", "range5_bp", "range5_p25_bp", "range5_open_bp", "tail_bp") else
                f"{r[c]:.2f}" if c in ("cost_rt_bp", "room", "room_p25", "ac1", "ac3", "score") else str(r[c]) for c in cols) + " |")
        return "\n".join(lines)
    md = [f"# Scalp candidates — structural screen — {datetime.now():%Y-%m-%d %H:%M}", "",
          f"{len(df)} symbols with ≥40 of the last {SESSIONS} sessions in the research cache (5-minute bars to {pd.Timestamp.now().date()}).",
          "Round-trip cost floor = cross half a spread twice (1 tick proxy) + 2 bp slippage per side + 1 tick adverse.",
          "`room` = median 5-minute range / that cost; `room_p25` the same on a quiet bar (25th percentile): a scalp",
          "needs room > ~3 on quiet bars to pay for itself. `ac1`/`ac3` = 5-minute return autocorrelation (negative =",
          "mean-reverting at the scalp horizon). Liquidity gate: price ≥ $15, ≥ $150M/day, trades in ≥ 97% of bars.", "",
          "## Top 30 by score", "", table(df.head(30)), "",
          "## The bot's current scan names", "",
          table(df[df["symbol"].isin(["NVDA", "TSLA", "META", "AMZN", "MSFT", "GOOGL", "AVGO", "AMD", "MU", "PLTR", "SMCI", "COIN", "HOOD", "INTC", "SPCX", "AAPL"])]), "",
          "## Bottom of the liquid set (worst room)", "", table(df[df["liquid"]].tail(10)), ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    df.to_parquet(out.with_suffix(".parquet"), index=False)
    print(f"symbols={len(df)} liquid={int(liq.sum())} report={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
