"""v-learning-loop-2026-10-22: weekly loss-attribution report (ENABLE_LOSS_ATTRIBUTION_REPORT).

For each closed live bot trade in the window (hands-off/external excluded),
attribute P&L to five buckets:
  entry    first-5-min adverse move, MAE reached before any MFE (R)
  exit     MFE given back; early (non-stop) losing exits vs the counterfactual
           hold-to-stop/target/flatten replay (research/learning/exit_replay.py)
  cost     fill vs the signal_router `accepted` entry price ($, bps); fees when recorded
  size     $ risk vs the 1R cap (RISK_CAP_PER_TRADE_USD, $150): loss in excess of the cap
  cluster  same-session, same-side entries within 5 minutes (correlated bets)
Report only. CLI: --csv/--bars (offline export) or --dsn (read-only Postgres).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from research.learning.data import (DAY_TRADE_STRATEGIES, Trade, bars_for_db, load_bars_csv, load_csv,
                                    load_db, sql_reader)
from research.learning.exit_replay import replay, summarize

CLUSTER_WINDOW = pd.Timedelta(minutes=5)
DEFAULT_RISK_CAP_USD = 150.0


def excursions(t: Trade, bars: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """MFE/MAE in R over the actual hold, plus 5-min post-entry return."""
    out: Dict[str, Any] = {"has_bars": False}
    if bars is None or bars.empty or not t.rps:
        return out
    en = t.entry_ts.floor("min")
    hold = bars[(bars["ts"] >= en) & (bars["ts"] <= t.exit_ts)]
    if hold.empty:
        return out
    if t.side == "long":
        fav, adv = hold["h"] - t.entry, t.entry - hold["l"]
    else:
        fav, adv = t.entry - hold["l"], hold["h"] - t.entry
    mfe, mae = max(0.0, float(fav.max())) / t.rps, max(0.0, float(adv.max())) / t.rps
    first_fav = hold["ts"][fav >= 0.25 * t.rps]
    first_adv = hold["ts"][adv >= 0.25 * t.rps]
    p5 = bars[bars["ts"] == en + pd.Timedelta(minutes=5)]
    ret5 = (t.sign * (float(p5["c"].iloc[0]) - t.entry) / t.rps) if not p5.empty else None
    out.update(has_bars=True, mfe_r=round(mfe, 3), mae_r=round(mae, 3),
               adverse_first=bool(len(first_adv) and (not len(first_fav) or first_adv.iloc[0] < first_fav.iloc[0])),
               ret5m_r=None if ret5 is None else round(ret5, 3))
    return out


def attribute(trades: Sequence[Trade], bars: Dict[str, pd.DataFrame],
              risk_cap_usd: float = DEFAULT_RISK_CAP_USD) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for t in sorted(trades, key=lambda x: x.entry_ts):
        ex = excursions(t, bars.get(str(t.id)))
        slip = None if t.sig_entry is None else t.sign * (t.entry - t.sig_entry) * t.qty
        slip_bps = None if t.sig_entry is None else t.sign * (t.entry - t.sig_entry) / t.sig_entry * 1e4
        risk_usd = t.rps * t.qty if t.rps else None
        size_excess = (t.pnl - t.pnl * risk_cap_usd / risk_usd) if (risk_usd and risk_usd > risk_cap_usd and t.pnl < 0) else 0.0
        rows.append(dict(id=str(t.id), symbol=t.symbol, side=t.side, strategy=t.strategy, setup=t.setup_type,
                         session=t.session, entry_et=t.entry_ts.strftime("%H:%M"), pnl=round(t.pnl, 2),
                         r=None if t.r is None else round(t.r, 3), r_source=t.r_source, exit_reason=t.exit_reason,
                         notional=round(t.entry * t.qty, 0), risk_usd=None if risk_usd is None else round(risk_usd, 2),
                         size_excess_loss=round(size_excess, 2),
                         slip_usd=None if slip is None else round(slip, 2),
                         slip_bps=None if slip_bps is None else round(slip_bps, 1),
                         fees=t.fees, **ex))
    # clusters
    by = defaultdict(list)
    for row, t in zip(rows, sorted(trades, key=lambda x: x.entry_ts)):
        by[(t.session, t.side)].append((t, row))
    clusters = []
    for (sess, side), items in by.items():
        grp = [items[0]] if items else []
        for prev, cur in zip(items, items[1:]):
            if cur[0].entry_ts - prev[0].entry_ts <= CLUSTER_WINDOW:
                grp.append(cur)
            else:
                if len(grp) >= 2: clusters.append(grp)
                grp = [cur]
        if len(grp) >= 2: clusters.append(grp)
    cl_out = []
    for g in clusters:
        ids = [r["id"] for _, r in g]
        for _, r in g: r["cluster"] = "+".join(ids)
        cl_out.append(dict(session=g[0][0].session, side=g[0][0].side, ids=ids, symbols=[t.symbol for t, _ in g],
                           pnl=round(sum(t.pnl for t, _ in g), 2),
                           r=round(sum((t.r or 0) for t, _ in g), 3)))
    rep = replay(trades, bars)
    rep_by = {r.id: r for r in rep}
    for row in rows:
        rr = rep_by.get(row["id"])
        row["cf_pnl"] = rr.cf_pnl if rr and rr.status == "ok" else None
        row["cf_status"] = rr.status if rr else "n/a"
    losers = [r for r in rows if r["pnl"] < 0]
    early = [r for r in losers if not r["exit_reason"].startswith("stop")]
    giveback = [r for r in rows if r.get("mfe_r") is not None and r["r"] is not None and r["mfe_r"] >= 0.5 and r["r"] < r["mfe_r"] - 0.4]
    slip_known = [r["slip_usd"] for r in rows if r["slip_usd"] is not None]
    by_exit = defaultdict(lambda: [0, 0.0]); by_sym = defaultdict(float); by_strat = defaultdict(lambda: [0, 0.0, 0.0])
    for r in rows:
        k = r["exit_reason"].split("_flipped")[0] if r["exit_reason"] else "?"
        by_exit[k][0] += 1; by_exit[k][1] += r["pnl"]; by_sym[r["symbol"]] += r["pnl"]
        by_strat[r["strategy"]][0] += 1; by_strat[r["strategy"]][1] += r["pnl"]; by_strat[r["strategy"]][2] += (r["r"] or 0)
    total = sum(r["pnl"] for r in rows)
    summary = dict(
        n=len(rows), pnl=round(total, 2), n_with_bars=sum(r["has_bars"] for r in rows),
        n_r_stored=sum(r["r_source"] == "pnl_r" for r in rows), n_r_derived=sum(str(r["r_source"]).startswith("derived") for r in rows),
        entry=dict(adverse_first=sum(bool(r.get("adverse_first")) for r in rows),
                   neg_5m=sum(1 for r in rows if (r.get("ret5m_r") or 0) < 0 and r.get("ret5m_r") is not None),
                   with_5m=sum(1 for r in rows if r.get("ret5m_r") is not None),
                   mean_mae_r_losers=round(sum(r["mae_r"] for r in losers if r.get("mae_r") is not None) / max(1, sum(1 for r in losers if r.get("mae_r") is not None)), 3)),
        exit=dict(early_losing_exits=len(early), early_losing_pnl=round(sum(r["pnl"] for r in early), 2),
                  early_losing_cf_pnl=round(sum(r["cf_pnl"] for r in early if r["cf_pnl"] is not None), 2),
                  early_losing_cf_n=sum(1 for r in early if r["cf_pnl"] is not None),
                  early_losing_actual_on_cf_n=round(sum(r["pnl"] for r in early if r["cf_pnl"] is not None), 2),
                  giveback_trades=[(r["id"], r["symbol"], r["mfe_r"], r["r"]) for r in giveback],
                  replay=summarize(rep)),
        cost=dict(slip_usd_total=round(sum(slip_known), 2), n_with_signal=len(slip_known),
                  fees_recorded=sum(r["fees"] is not None for r in rows)),
        size=dict(n_over_cap=sum(1 for r in rows if (r["risk_usd"] or 0) > DEFAULT_RISK_CAP_USD),
                  excess_loss_over_cap=round(sum(r["size_excess_loss"] for r in rows), 2),
                  max_risk_usd=max((r["risk_usd"] or 0) for r in rows) if rows else 0),
        cluster=dict(n_clusters=len(cl_out), clusters=cl_out),
        by_exit={k: dict(n=v[0], pnl=round(v[1], 2)) for k, v in sorted(by_exit.items(), key=lambda kv: kv[1][1])},
        by_symbol={k: round(v, 2) for k, v in sorted(by_sym.items(), key=lambda kv: kv[1])},
        by_strategy={k: dict(n=v[0], pnl=round(v[1], 2), sum_r=round(v[2], 3)) for k, v in by_strat.items()},
    )
    return {"summary": summary, "trades": rows}


def render(res: Dict[str, Any], since: str, until: str) -> str:
    s = res["summary"]; e = s["exit"]; rp = e["replay"]
    L = [f"# Loss attribution {since} .. {until} (ET, live bot trades, hands-off/external excluded)", "",
         f"n={s['n']}  P&L=${s['pnl']:+.2f}  bars for {s['n_with_bars']}/{s['n']}  "
         f"R: stored {s['n_r_stored']}, derived from router stop {s['n_r_derived']}", "",
         "## Entry", f"- MAE (>=0.25R) before MFE: {s['entry']['adverse_first']}/{s['n_with_bars']}",
         f"- negative 5-min post-entry: {s['entry']['neg_5m']}/{s['entry']['with_5m']}",
         f"- mean MAE of losers: {s['entry']['mean_mae_r_losers']}R", "",
         "## Exit", f"- early (non-stop) losing exits: {e['early_losing_exits']} for ${e['early_losing_pnl']:+.2f}; "
         f"of those with bars ({e['early_losing_cf_n']}) actual ${e['early_losing_actual_on_cf_n']:+.2f} vs hold-to-stop/target/flatten ${e['early_losing_cf_pnl']:+.2f}",
         f"- MFE >=0.5R given back (>0.4R): {e['giveback_trades']}",
         f"- full exit replay (all replayable trades): actual ${rp['replayed_actual_pnl']:+.2f} / {rp['replayed_actual_r']:+.2f}R vs counterfactual ${rp['replayed_cf_pnl']:+.2f} / {rp['replayed_cf_r']:+.2f}R (n={rp['n_replayed']}, no bars {rp['n_no_bars']})", "",
         "## Cost", f"- fill vs signal: ${s['cost']['slip_usd_total']:+.2f} over {s['cost']['n_with_signal']} trades (+ = paid)",
         f"- fees recorded on {s['cost']['fees_recorded']} trades", "",
         "## Size", f"- trades with $risk > $150 (1R cap): {s['size']['n_over_cap']}; loss in excess of cap ${s['size']['excess_loss_over_cap']:+.2f}; max $risk ${s['size']['max_risk_usd']}", "",
         "## Cluster"] + [f"- {c['session']} {c['side']} {c['symbols']}: ${c['pnl']:+.2f} / {c['r']:+.2f}R" for c in s["cluster"]["clusters"]] + [
         "", "## By exit reason"] + [f"- {k}: n={v['n']} ${v['pnl']:+.2f}" for k, v in s["by_exit"].items()] + [
         "", "## By strategy"] + [f"- {k}: n={v['n']} ${v['pnl']:+.2f} sumR={v['sum_r']:+.2f}" for k, v in s["by_strategy"].items()] + [
         "", "## Worst symbols"] + [f"- {k}: ${v:+.2f}" for k, v in list(s["by_symbol"].items())[:8]] + [
         "", "## Trades", "", "| id | sym | side | session | entry | $ | R | exit | MFE R | MAE R | 5m R | slip $ | $risk | cf $ | cluster |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["trades"]:
        f = lambda v: "" if v is None else v
        L.append(f"| {r['id']} | {r['symbol']} | {r['side']} | {r['session']} | {r['entry_et']} | {r['pnl']:+.2f} | {f(r['r'])} | "
                 f"{r['exit_reason']} | {f(r.get('mfe_r'))} | {f(r.get('mae_r'))} | {f(r.get('ret5m_r'))} | {f(r['slip_usd'])} | "
                 f"{f(r['risk_usd'])} | {f(r['cf_pnl'])} | {r.get('cluster', '')} |")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Weekly loss attribution (report only).")
    ap.add_argument("--csv"); ap.add_argument("--bars"); ap.add_argument("--dsn")
    ap.add_argument("--since"); ap.add_argument("--until"); ap.add_argument("--out", required=True)
    ap.add_argument("--all-strategies", action="store_true"); ap.add_argument("--denylist", default="HQGE,SPCX")
    ap.add_argument("--force", action="store_true", help="run even if ENABLE_LOSS_ATTRIBUTION_REPORT is off")
    a = ap.parse_args(argv)
    if not a.force:
        try:
            from core.config import Config
            if Config().ENABLE_LOSS_ATTRIBUTION_REPORT is not True:
                print("ENABLE_LOSS_ATTRIBUTION_REPORT off; nothing to do"); return 0
        except Exception as exc:  # pragma: no cover
            print(f"config unavailable ({exc}); use --force"); return 0
    until = pd.Timestamp(a.until) if a.until else pd.Timestamp(datetime.now().date()) + pd.Timedelta(days=1)
    since = pd.Timestamp(a.since) if a.since else until - pd.Timedelta(days=7)
    deny = [s for s in a.denylist.split(",") if s]
    if a.csv:
        trades = load_csv(a.csv, deny); bars = load_bars_csv(a.bars) if a.bars else {}
    else:
        rd = sql_reader(a.dsn); trades = load_db(rd, since.to_pydatetime(), until.to_pydatetime(), deny); bars = bars_for_db(rd, trades)
    trades = [t for t in trades if since <= t.exit_ts < until]
    if not a.all_strategies:
        trades = [t for t in trades if t.strategy in DAY_TRADE_STRATEGIES]
    res = attribute(trades, bars)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(res, str(since.date()), str((until - pd.Timedelta(days=1)).date())))
    out.with_suffix(".json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res["summary"][k] for k in ("n", "pnl")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
