"""Pooled, fixed-rule evaluation of the RSI-extreme families (2026-10-05).

Why this exists: the walk-forward pipeline selects a parameter set per symbol and
per 12-month window only when that set produced >= 30 in-sample trades, else it
falls back to the grid's median parameters. A daily RSI(14) < 25 print happens a
handful of times a year per name, so `rsi_dip` never cleared the floor and the
pipeline simulated the median fallback (which includes the SMA200 filter and
fires almost never): 3 out-of-sample trades across 14 symbols. That is a
property of the search machinery, not a verdict on the rule.

Here every grid combination is simulated on every symbol with the pipeline's own
simulator and cost model (fills, slippage from calibration, time stops, regime
labels), then POOLED across symbols with no selection at all. Because the rule
came from an event study on the same tape, the whole period is in-sample for
the hypothesis: read the per-year split as the robustness check.

Writes `research/reports/rsi/<date>-rsi-fixed-rule.md`.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.strategy_hunt.context import load_context_slice  # noqa: E402
from research.strategy_hunt.data.earnings import earnings_days  # noqa: E402
from research.strategy_hunt.data.news import load_news  # noqa: E402
from research.strategy_hunt.data.splits import load_adjusted  # noqa: E402
from research.strategy_hunt.run import load_context  # noqa: E402
from research.strategy_hunt.sim.calibrate import slippage_override_from  # noqa: E402
from research.strategy_hunt.sim.costs import CostModel  # noqa: E402
from research.strategy_hunt.sim.simulator import build_session_table  # noqa: E402
from research.strategy_hunt.walkforward.runner import precompute  # noqa: E402
from strategies.families import get_family  # noqa: E402
from strategies.families.indicators import indicator_frame  # noqa: E402

CACHE = ROOT / "data" / "strategy_hunt"
SYMBOLS = ["MU", "TSLA", "META", "PLTR", "SMCI", "NVDA", "AMD", "AVGO", "AMZN", "MSFT", "GOOGL", "HOOD", "COIN", "INTC"]
FAMILIES = ("rsi_dip", "rsi_fade")
START = pd.Timestamp("2022-01-01")


def simulate_symbol(symbol: str, ctx: dict) -> list[dict]:
    bars5, daily = load_adjusted(CACHE, symbol)
    if bars5.empty:
        return []
    news = load_news(CACHE, symbol)
    earnings = earnings_days(CACHE, symbol, daily)
    cost_model = CostModel.from_bars(bars5, 6000.0, slippage_override=slippage_override_from(CACHE))
    ind = indicator_frame(bars5, daily, ctx["spy5"], context=load_context_slice(CACHE, symbol))
    table = build_session_table(bars5)
    rows: list[dict] = []
    for fam_name in FAMILIES:
        fam = get_family(fam_name)
        pre = precompute(fam, symbol, bars5, ind, daily, news, cost_model, earnings, ctx["regime_lookup"], table)
        for key, trades in pre.items():
            params = json.loads(key)
            for t in trades:
                if t.entry_ts < START:
                    continue
                rows.append({"symbol": symbol, "family": fam_name, "params": key, "year": t.entry_ts.year,
                             "regime": t.regime, "r": t.r_multiple, "pnl_pct": t.pnl_pct, "cost_pct": t.cost_pct,
                             "exit": t.exit_reason, "hold": t.hold_sessions, **{f"p_{k}": v for k, v in params.items()}})
    return rows


def stats(g: pd.DataFrame) -> pd.Series:
    r = g["r"].to_numpy(float)
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / -losses.sum() if losses.size and losses.sum() < 0 else np.inf
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 2 and r.std(ddof=1) > 0 else np.nan
    return pd.Series({"n": len(r), "exp_r": r.mean(), "win": (r > 0).mean(), "pf": pf, "t": t,
                      "symbols": g["symbol"].nunique(), "cost_pct": g["cost_pct"].mean()})


def by(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Group stats as a plain DataFrame indexed by `cols` (groupby.apply returns a
    stacked Series for single-column groups under pandas 2)."""
    out = {}
    for key, g in df.groupby(cols):
        out[key if isinstance(key, tuple) else (key,)] = stats(g)
    res = pd.DataFrame(out).T
    res.index = pd.MultiIndex.from_tuples(res.index, names=cols) if len(cols) > 1 else pd.Index([k[0] for k in res.index], name=cols[0])
    return res


def table(df: pd.DataFrame, index_cols: list[str]) -> str:
    df = df.reset_index()
    cols = index_cols + ["n", "exp_r", "win", "pf", "t", "symbols", "cost_pct"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if c in ("exp_r",):
                cells.append(f"{v:+.3f}")
            elif c == "win":
                cells.append(f"{v:.0%}")
            elif c in ("pf", "t"):
                cells.append("inf" if v == np.inf else ("" if pd.isna(v) else f"{v:.2f}"))
            elif c == "cost_pct":
                cells.append(f"{v:.2f}%")
            elif isinstance(v, float):
                cells.append(f"{v:g}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> int:
    ctx = load_context(CACHE)
    rows: list[dict] = []
    for sym in SYMBOLS:
        rows.extend(simulate_symbol(sym, ctx))
        print(f"{sym}: {sum(1 for r in rows if r['symbol'] == sym)} simulated trades", file=sys.stderr)
    ev = pd.DataFrame(rows)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [f"# RSI families — pooled fixed-rule evaluation — {stamp}", "",
          "Every grid combination simulated on every symbol with the strategy-hunt simulator (fills,",
          "calibrated slippage, stops, time stops) from 2022-01-01, pooled across the 14 symbols with",
          "NO parameter selection. `exp_r` is the mean R-multiple net of costs; `t` is the plain",
          "t-statistic (trades treated as independent, so optimistic); `pf` the profit factor.",
          "The rule was formed on this same tape: treat the per-year split as the robustness check.", ""]
    for fam in FAMILIES:
        sub = ev[ev.family == fam]
        if sub.empty:
            md += [f"## {fam}", "", "no trades", ""]
            continue
        pcols = sorted(c for c in sub.columns if c.startswith("p_") and sub[c].notna().any())
        by_params = by(sub, pcols).sort_values("exp_r", ascending=False)
        md += [f"## {fam} — by parameter set (pooled)", "", table(by_params, pcols), ""]
        # Best set among those with at least 15 pooled trades (a 3-trade set is noise).
        per = sub.groupby("params")["r"].agg(["mean", "size"])
        eligible = per[per["size"] >= 15]
        best_key = (eligible if not eligible.empty else per)["mean"].idxmax()
        best = sub[sub.params == best_key]
        md += [f"### {fam} — best pooled set `{best_key}`", "",
               "By year:", "", table(by(best, ["year"]), ["year"]), "",
               "By regime (entry-time label):", "", table(by(best, ["regime"]), ["regime"]), "",
               "By symbol:", "", table(by(best, ["symbol"]).sort_values("exp_r", ascending=False), ["symbol"]), "",
               "Exit reasons:", "",
               "| exit | n |", "|---|---|",
               *[f"| {k} | {v} |" for k, v in best["exit"].value_counts().items()], ""]
    out = ROOT / "research" / "reports" / "rsi" / f"{datetime.now():%Y-%m-%d}-rsi-fixed-rule.md"
    out.write_text("\n".join(md))
    ev.to_parquet(out.with_suffix(".parquet"), index=False)
    print(f"trades={len(ev)} report={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
