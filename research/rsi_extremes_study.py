"""RSI-extreme bounce / fade study (2026-10-05).

Question from the operator: how often, and in what pattern, do SPCX, MU, TSLA,
META, PLTR, SMCI and other top names bounce after a low RSI and fall back after
a high RSI — and can a buy/sell rule be built on it?

Method (pure event study, no optimisation):
  * Bars: the research cache (`data/strategy_hunt/{bars5,daily}`), 2021 → now;
    SPCX is not cached and is pulled from yfinance (daily + 60m, shorter).
  * Timeframes: daily, 60-minute and 15-minute (resampled from 5-minute bars,
    regular session only).
  * RSI(14) on closes. Two event kinds per threshold:
      - `cross_below T` : first bar whose RSI closes below T after being >= T
        (the "it's oversold" moment), T in {20, 25, 30, 35};
      - `turn_up T`     : RSI was below T within the last 3 bars and now closes
        back above T (the bounce confirmation).
    Mirror for overbought: `cross_above T` and `turn_down T`, T in {65, 70, 75, 80}.
  * For each event: forward close-to-close return after 1, 3, 5, 10 bars, the
    hit rate (return > 0 for longs, < 0 for shorts), max favourable / adverse
    excursion over 10 bars, and bars until RSI crosses back through 50.
  * Baseline: the unconditional forward-return distribution of the same symbol
    and horizon, so every table shows EXCESS over "just being long".

Run:  python research/rsi_extremes_study.py [--out research/reports/rsi/<date>.md]
Pure read-only on the cache; writes one markdown report.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.strategy_hunt.data.bars import load_bars5, load_daily  # noqa: E402

CACHE = ROOT / "data" / "strategy_hunt"
SYMBOLS = ["SPCX", "MU", "TSLA", "META", "PLTR", "SMCI", "NVDA", "AMD", "AVGO",
           "AMZN", "MSFT", "GOOGL", "HOOD", "COIN", "INTC"]
LOW_T = (20, 25, 30, 35)
HIGH_T = (65, 70, 75, 80)
HORIZONS = (1, 3, 5, 10)
LOOKBACK_TURN = 3


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def session_bars(bars5: pd.DataFrame) -> pd.DataFrame:
    idx = bars5.index
    t = idx.time if hasattr(idx, "time") else pd.DatetimeIndex(idx).time
    mask = np.array([(h.hour, h.minute) >= (9, 30) and (h.hour, h.minute) < (16, 0) for h in t])
    return bars5[mask]


def resample(bars5: pd.DataFrame, rule: str) -> pd.DataFrame:
    b = session_bars(bars5)
    o = b["open"].resample(rule, origin="start_day", offset="9h30min").first()
    h = b["high"].resample(rule, origin="start_day", offset="9h30min").max()
    l = b["low"].resample(rule, origin="start_day", offset="9h30min").min()
    c = b["close"].resample(rule, origin="start_day", offset="9h30min").last()
    v = b["volume"].resample(rule, origin="start_day", offset="9h30min").sum()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}).dropna(subset=["close"])
    return out


def load_frames(symbol: str) -> dict[str, pd.DataFrame]:
    daily = load_daily(CACHE, symbol)
    bars5 = load_bars5(CACHE, symbol)
    frames: dict[str, pd.DataFrame] = {}
    if not daily.empty:
        frames["daily"] = daily
    if not bars5.empty:
        frames["60m"] = resample(bars5, "60min")
        frames["15m"] = resample(bars5, "15min")
    if not frames:
        try:
            import yfinance as yf
            d = yf.Ticker(symbol).history(period="5y", interval="1d")
            h = yf.Ticker(symbol).history(period="730d", interval="60m")
            for k, df in (("daily", d), ("60m", h)):
                if df is not None and len(df) > 60:
                    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
                    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
                    frames[k] = df
        except Exception as exc:  # pragma: no cover - network
            print(f"{symbol}: yfinance fallback failed: {exc}", file=sys.stderr)
    return frames


@dataclass
class EventStats:
    kind: str
    threshold: int
    side: str
    n: int
    rows: list


def forward_returns(close: np.ndarray, i: int) -> dict[int, float]:
    out = {}
    for h in HORIZONS:
        out[h] = close[i + h] / close[i] - 1 if i + h < len(close) else np.nan
    return out


def excursions(high: np.ndarray, low: np.ndarray, close: np.ndarray, i: int, side: str) -> tuple[float, float]:
    j = min(len(close) - 1, i + 10)
    if j <= i:
        return np.nan, np.nan
    px = close[i]
    if side == "long":
        return high[i + 1:j + 1].max() / px - 1, low[i + 1:j + 1].min() / px - 1
    return px / low[i + 1:j + 1].min() - 1, px / high[i + 1:j + 1].max() - 1


def bars_to_50(r: np.ndarray, i: int, side: str) -> float:
    for k in range(i + 1, min(len(r), i + 60)):
        if (side == "long" and r[k] >= 50) or (side == "short" and r[k] <= 50):
            return k - i
    return np.nan


def events_for(df: pd.DataFrame, symbol: str, tf: str) -> list[dict]:
    c = df["close"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    r = rsi(df["close"]).to_numpy(float)
    rows: list[dict] = []
    n = len(c)
    for i in range(15, n - 1):
        for t in LOW_T:
            if r[i] < t <= r[i - 1]:
                rows.append(_row(symbol, tf, "cross_below", t, "long", i, c, h, lo, r))
            if r[i] > t and (r[i - LOOKBACK_TURN:i] < t).any() and r[i - 1] <= t:
                rows.append(_row(symbol, tf, "turn_up", t, "long", i, c, h, lo, r))
        for t in HIGH_T:
            if r[i] > t >= r[i - 1]:
                rows.append(_row(symbol, tf, "cross_above", t, "short", i, c, h, lo, r))
            if r[i] < t and (r[i - LOOKBACK_TURN:i] > t).any() and r[i - 1] >= t:
                rows.append(_row(symbol, tf, "turn_down", t, "short", i, c, h, lo, r))
    # unconditional baseline: every bar
    base = {hz: np.nanmean(c[hz:] / c[:-hz] - 1) for hz in HORIZONS}
    for row in rows:
        for hz in HORIZONS:
            row[f"base_{hz}"] = base[hz]
    return rows


def _row(symbol, tf, kind, t, side, i, c, h, lo, r) -> dict:
    fr = forward_returns(c, i)
    sign = 1 if side == "long" else -1
    mfe, mae = excursions(h, lo, c, i, side)
    return {"symbol": symbol, "tf": tf, "kind": kind, "threshold": t, "side": side,
            **{f"ret_{hz}": sign * fr[hz] for hz in HORIZONS}, "mfe": mfe, "mae": mae,
            "bars_to_50": bars_to_50(r, i, side), "rsi": r[i]}


def summarise(ev: pd.DataFrame) -> pd.DataFrame:
    g = ev.groupby(["tf", "kind", "threshold", "side"])
    out = g.agg(n=("ret_5", "size"),
                hit1=("ret_1", lambda s: np.nanmean(s > 0)),
                hit5=("ret_5", lambda s: np.nanmean(s > 0)),
                hit10=("ret_10", lambda s: np.nanmean(s > 0)),
                mean1=("ret_1", "mean"), mean5=("ret_5", "mean"), mean10=("ret_10", "mean"),
                med5=("ret_5", "median"),
                mfe=("mfe", "median"), mae=("mae", "median"),
                to50=("bars_to_50", "median"),
                base5=("base_5", "mean"), base10=("base_10", "mean")).reset_index()
    sign = np.where(out["side"] == "long", 1, -1)
    out["excess5"] = out["mean5"] - sign * out["base5"]
    out["excess10"] = out["mean10"] - sign * out["base10"]
    return out


def per_symbol(ev: pd.DataFrame, tf: str, kind: str, t: int) -> pd.DataFrame:
    sub = ev[(ev.tf == tf) & (ev.kind == kind) & (ev.threshold == t)]
    g = sub.groupby("symbol").agg(n=("ret_5", "size"), hit5=("ret_5", lambda s: np.nanmean(s > 0)),
                                  mean5=("ret_5", "mean"), mean10=("ret_10", "mean"),
                                  mae=("mae", "median"), base5=("base_5", "mean"))
    sign = 1 if sub["side"].iloc[0] == "long" else -1
    g["excess5"] = g["mean5"] - sign * g["base5"]
    return g.sort_values("excess5", ascending=False)


def fmt_pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x * 100:+.2f}%"


def table(df: pd.DataFrame, cols: list[str], pct: set[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in pct:
                cells.append(fmt_pct(v))
            elif isinstance(v, float):
                cells.append("" if pd.isna(v) else (f"{v:.0%}" if c.startswith("hit") else f"{v:.1f}"))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    args = ap.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    rows: list[dict] = []
    coverage: list[str] = []
    for sym in symbols:
        frames = load_frames(sym)
        if not frames:
            coverage.append(f"| {sym} | no data | | |")
            continue
        for tf, df in frames.items():
            rows.extend(events_for(df, sym, tf))
        d = frames.get("daily")
        coverage.append(f"| {sym} | {', '.join(frames)} | {d.index.min().date() if d is not None else ''} | "
                        f"{d.index.max().date() if d is not None else ''} |")
    ev = pd.DataFrame(rows)
    summ = summarise(ev)
    cols = ["tf", "kind", "threshold", "n", "hit1", "hit5", "hit10", "mean1", "mean5", "mean10", "med5",
            "excess5", "excess10", "mfe", "mae", "to50"]
    pct = {"mean1", "mean5", "mean10", "med5", "excess5", "excess10", "mfe", "mae"}
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [f"# RSI-extreme bounce / fade study — {stamp}", "",
          "Event study on RSI(14) extremes, 2021 → now, regular session bars. Returns are signed in the",
          "direction of the trade (long after a low RSI, short after a high RSI); `excess` is the mean",
          "trade return minus the unconditional drift of the same symbol over the same horizon. No costs",
          "deducted: a round trip costs roughly 0.05–0.10% intraday and 0.10% daily on these names.", "",
          "## Coverage", "", "| symbol | timeframes | daily from | daily to |", "|---|---|---|---|", *coverage, ""]
    for tf in ("daily", "60m", "15m"):
        sub = summ[summ.tf == tf]
        if sub.empty:
            continue
        md += [f"## {tf}", "", "### Longs after a low RSI", "",
               table(sub[sub.side == "long"].sort_values(["kind", "threshold"]), cols, pct), "",
               "### Shorts after a high RSI", "",
               table(sub[sub.side == "short"].sort_values(["kind", "threshold"]), cols, pct), ""]
    md += ["## Per symbol — daily turn_up 30 (the bounce confirmation)", "",
           table(per_symbol(ev, "daily", "turn_up", 30).reset_index(),
                 ["symbol", "n", "hit5", "mean5", "mean10", "excess5", "mae"], {"mean5", "mean10", "excess5", "mae"}), "",
           "## Per symbol — daily turn_down 70 (the fade confirmation)", "",
           table(per_symbol(ev, "daily", "turn_down", 70).reset_index(),
                 ["symbol", "n", "hit5", "mean5", "mean10", "excess5", "mae"], {"mean5", "mean10", "excess5", "mae"}), "",
           "## Per symbol — 60m turn_up 30", "",
           table(per_symbol(ev, "60m", "turn_up", 30).reset_index(),
                 ["symbol", "n", "hit5", "mean5", "mean10", "excess5", "mae"], {"mean5", "mean10", "excess5", "mae"}), ""]
    out = args.out or ROOT / "research" / "reports" / "rsi" / f"{datetime.now():%Y-%m-%d}-rsi-extremes.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    ev.to_parquet(out.with_suffix(".parquet"), index=False)
    print(f"events={len(ev)} report={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
