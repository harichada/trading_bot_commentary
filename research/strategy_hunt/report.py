"""Markdown research report from fit rows, pooled trades, control runs, portfolio replay, and cost calibration."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd

from research.strategy_hunt.paths import trades_dir
from research.strategy_hunt.run import REPORTS_DIR
from research.strategy_hunt.score.regime_rows import POOLED_SOURCE

# v-playbook-desk-b1-5 task 10 fix r1: only the cost controls live in separate roots now. The
# shuffled control moved in-root (<root>/trades_shuffle/) and is reported as the per-pooled-row
# `excess` in "Pooled rows vs shuffled baseline", which supersedes the old separate-root
# strategy_fit_shuffle.json and the whole-run false-discovery ratio it fed.
CONTROL_FILES = {"cost_0.5": "strategy_fit_cost_0.5.json", "cost_2.0": "strategy_fit_cost_2.0.json"}
CAVEATS = (
    "- Universe is selected by today's data availability (survivorship); results are stated for symbols that exist now.\n"
    "- Spread and slippage are estimated from bars; see the cost controls for sensitivity.\n"
    "- Ten families times up to 27 parameter sets across hundreds of symbols will produce spurious PASS rows; "
    "the in-root shuffled baseline estimates how much of a pooled lane's expectancy is the regime's drift "
    "(see \"Pooled rows vs shuffled baseline\").\n"
    "- Live indicators are computed on the in-progress bar; research uses closed bars. New lanes must use closed bars.\n"
)


def _f(x, nd=3) -> str:
    return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _verdict_table(rows: list[dict]) -> str:
    cell = defaultdict(lambda: {"PASS": 0, "WATCH": 0, "FAIL": 0, "coldstart_watch": 0})
    for r in rows:
        k = (r["strategy"], r["side"])
        if r["source"] == "coldstart":
            if r["verdict"] == "WATCH":
                cell[k]["coldstart_watch"] += 1
        else:
            cell[k][r["verdict"]] += 1
    lines = ["| family | side | PASS | WATCH | FAIL | coldstart WATCH |", "|---|---|---|---|---|---|"]
    for (fam, side), c in sorted(cell.items()):
        lines.append(f"| {fam} | {side} | {c['PASS']} | {c['WATCH']} | {c['FAIL']} | {c['coldstart_watch']} |")
    return "\n".join(lines)


def _top_table(rows: list[dict], k: int = 30) -> str:
    wf = [r for r in rows if r["source"] == "walkforward"]
    lines = ["| symbol | family | side | verdict | n | exp_r | exp_r_lb | t_stat | pf | wr | max_dd_r | stability | decay | score |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(wf, key=lambda r: -r["score"])[:k]:
        lines.append(f"| {r['symbol']} | {r['strategy']} | {r['side']} | {r['verdict']} | {r['n']} | {_f(r['exp_r'])} | "
                     f"{_f(r.get('exp_r_lb'))} | {_f(r.get('t_stat'))} | {_f(r['pf'], 2)} | {_f(r['wr'], 2)} | "
                     f"{_f(r['max_dd_r'], 1)} | {_f(r['stability'], 2)} | {_f(r['decay'], 2)} | {_f(r['score'])} |")
    return "\n".join(lines)


def _pooled_table(trades: Optional[pd.DataFrame]) -> str:
    if trades is None or trades.empty:
        return "_no trades loaded_"
    lines = ["| family | side | n | exp_r | pf | wr |", "|---|---|---|---|---|---|"]
    for (fam, side), g in trades.groupby(["family", "side"]):
        r = g["r_multiple"].astype(float)
        wins, losses = r[r > 0].sum(), -r[r < 0].sum()
        pf = wins / losses if losses > 0 else float("inf")
        decided = r[r.abs() >= 0.05]
        wr = (decided > 0).mean() if len(decided) else float("nan")
        lines.append(f"| {fam} | {side} | {len(r)} | {_f(r.mean())} | {_f(pf, 2)} | {_f(wr, 2)} |")
    return "\n".join(lines)


def _regime_table(rows: list[dict]) -> str:
    acc: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for r in rows:
        if r["source"] != "walkforward":
            continue
        for reg, v in json.loads(r["regime_exp_json"] or "{}").items():
            a = acc[(r["strategy"], r["side"])][reg]
            a[0] += v["exp_r"] * v["n"]
            a[1] += v["n"]
    regs = sorted({reg for d in acc.values() for reg in d})
    lines = ["| family | side | " + " | ".join(regs) + " |", "|---|---|" + "---|" * len(regs)]
    for (fam, side), d in sorted(acc.items()):
        cells = [_f(d[reg][0] / d[reg][1]) + f" (n={d[reg][1]})" if reg in d and d[reg][1] else "" for reg in regs]
        lines.append(f"| {fam} | {side} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _excess_key(row: dict) -> float:
    """Sort key for the baseline table: a NaN excess ("no baseline measured") sorts last."""
    v = row.get("excess_r")
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return float("-inf")
    return float(v)


def _pooled_baseline_table(pooled: Optional[list[dict]]) -> str:
    """Every pooled row against its same-regime shuffled baseline, biggest excess first.

    This is the table the B1 run needed: a pooled `exp_R` alone cannot say whether a lane
    beats the regime's drift, and `excess` is exactly that difference."""
    if not pooled:
        return "_no pooled rows_"
    lines = ["| family | side | regime | n | exp_R | baseline exp_R | excess | t_session | recovery | "
             "verdict | failed floors |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(pooled, key=_excess_key, reverse=True):
        try:
            failed = ", ".join(json.loads(r.get("failed_floors") or "[]"))
        except (TypeError, ValueError):       # a hand-edited snapshot must not break the report
            failed = str(r.get("failed_floors"))
        lines.append(f"| {r.get('strategy', '')} | {r.get('side', '')} | {r.get('regime', '')} | "
                     f"{r.get('n', 0)} | {_f(r.get('exp_r'))} | {_f(r.get('baseline_exp_r'))} | "
                     f"{_f(r.get('excess_r'))} | {_f(r.get('t_stat_session'), 2)} | "
                     f"{_f(r.get('recovery'), 2)} | {r.get('verdict', '')} | {failed} |")
    return "\n".join(lines)


def _pooled_counts(rows: list[dict]) -> dict[str, int]:
    out = {"PASS": 0, "WATCH": 0}
    for r in rows:
        if r.get("source") == POOLED_SOURCE and r.get("verdict") in out:
            out[r["verdict"]] += 1
    return out


def _controls_section(rows: list[dict], controls: Optional[dict[str, list[dict]]]) -> str:
    real_pass = sum(1 for r in rows if r["source"] == "walkforward" and r["verdict"] == "PASS")
    if not controls:
        return f"_no control runs found_ (real PASS rows: {real_pass})"
    lines = []
    for name, crows in sorted(controls.items()):
        p = sum(1 for r in crows if r.get("source", "walkforward") == "walkforward" and r["verdict"] == "PASS")
        exp = [r["exp_r"] for r in crows if r.get("source", "walkforward") == "walkforward" and not (isinstance(r["exp_r"], float) and math.isnan(r["exp_r"]))]
        mean_exp = sum(exp) / len(exp) if exp else float("nan")
        pc = _pooled_counts(crows)
        lines.append(f"- **{name}**: PASS rows {p}, mean exp_r {_f(mean_exp)}, "
                     f"pooled PASS {pc['PASS']}, pooled WATCH {pc['WATCH']}")
    lines.append(f"- real PASS rows: {real_pass}")
    return "\n".join(lines)


def _no_pass_section(rows: list[dict]) -> str:
    by_sym: dict[str, bool] = defaultdict(bool)
    for r in rows:
        if r["source"] == "walkforward":
            by_sym[r["symbol"]] = by_sym[r["symbol"]] or r["verdict"] == "PASS"
    none = sorted(s for s, ok in by_sym.items() if not ok)
    return f"{len(none)} of {len(by_sym)} symbols have no PASS lane.\n\n" + ", ".join(none[:50])


def _monthly_curve_table(curve: Optional[pd.Series]) -> str:
    if curve is None or curve.empty:
        return "_no data_"
    monthly = curve.groupby(pd.PeriodIndex(pd.to_datetime(curve.index), freq="M")).last()
    lines = ["| month | equity |", "|---|---|"]
    for period, val in monthly.items():
        lines.append(f"| {period} | {val:.2f} |")
    return "\n".join(lines)


def _portfolio_section(portfolio: Optional[dict]) -> str:
    if portfolio is None:
        return "_not available_"
    fam_lines = ["| family | taken | pnl |", "|---|---|---|"]
    for fam, v in sorted(portfolio.get("by_family", {}).items()):
        fam_lines.append(f"| {fam} | {v['taken']} | {v['pnl']:.2f} |")
    fam_table = "\n".join(fam_lines) if len(fam_lines) > 2 else "_no trades taken_"
    curve_table = _monthly_curve_table(portfolio.get("equity_curve"))
    return (
        f"- taken: {portfolio['taken']}\n"
        f"- skipped (capacity): {portfolio['skipped_capacity']}\n"
        f"- skipped (symbol already open): {portfolio['skipped_symbol']}\n"
        f"- skipped (size too small): {portfolio['skipped_size']}\n"
        f"- final equity: {portfolio['final_equity']:.2f}\n"
        f"- total return: {portfolio['total_return_pct']:.2f}%\n"
        f"- max drawdown: {portfolio['max_drawdown_pct']:.1f}%\n"
        f"- daily Sharpe: {portfolio['daily_sharpe']:.2f}\n\n"
        f"### By family\n\n{fam_table}\n\n"
        f"### Equity curve (monthly)\n\n{curve_table}"
    )


def _calibration_section(calibration: Optional[dict]) -> str:
    if calibration is None:
        return "_not available_"
    bucket_lines = ["| bucket | n | median_frac |", "|---|---|---|"]
    for name, v in sorted(calibration.get("by_bucket", {}).items()):
        bucket_lines.append(f"| {name} | {v['n']} | {_f(v['median_frac'], 4)} |")
    bucket_table = "\n".join(bucket_lines) if len(bucket_lines) > 2 else "_no buckets_"
    return (
        f"- n: {calibration['n']}\n"
        f"- median slippage fraction: {_f(calibration['median_frac'], 4)}\n"
        f"- p75 slippage fraction: {_f(calibration['p75_frac'], 4)}\n"
        f"- usable: {calibration['usable']}\n\n"
        f"### By bucket\n\n{bucket_table}"
    )


def render_report(rows: list[dict], computed_at: str, trades: Optional[pd.DataFrame] = None,
                  controls: Optional[dict[str, list[dict]]] = None, portfolio: Optional[dict] = None,
                  calibration: Optional[dict] = None, pooled: Optional[list[dict]] = None) -> str:
    syms = {r["symbol"] for r in rows}
    parts = [f"# Strategy hunt report\n\nComputed: {computed_at}. Rows: {len(rows)}. Symbols: {len(syms)}.\n",
             "## Verdicts by family\n\n" + _verdict_table(rows),
             "## Top lanes by score\n\n" + _top_table(rows),
             "## Pooled out-of-sample by family\n\n" + _pooled_table(trades),
             "## Per-regime expectancy by family\n\n" + _regime_table(rows),
             "## Pooled rows vs shuffled baseline\n\n" + _pooled_baseline_table(pooled),
             "## Controls\n\n" + _controls_section(rows, controls),
             "## Symbols with no passing lane\n\n" + _no_pass_section(rows),
             "## Portfolio replay of PASS lanes\n\n" + _portfolio_section(portfolio),
             "## Cost calibration\n\n" + _calibration_section(calibration),
             "## Caveats\n\n" + CAVEATS]
    return "\n\n".join(parts) + "\n"


def _load_controls(root: Path) -> dict[str, list[dict]]:
    """Each control snapshot's rows: the unconditional lanes plus its pooled rows.

    Pooled rows are kept in the same list and told apart by `source`, so
    `_controls_section` can report both counts without a second container shape.
    Baseline rows are never loaded — a control's shuffled baseline is its own business."""
    out = {}
    for name, fname in CONTROL_FILES.items():
        p = Path(root) / "fit" / fname
        if p.exists():
            snap = json.loads(p.read_text())
            lanes = [r for lanes in snap.get("rows", {}).values() for r in lanes.values()]
            pooled = list(snap.get("pooled_rows", {}).values())
            out[name] = lanes + pooled
    return out


def _write_equity_curve_csv(curve: pd.Series, path: Path) -> None:
    lines = ["date,equity"]
    for ts, val in curve.items():
        lines.append(f"{pd.Timestamp(ts).date()},{val}")
    path.write_text("\n".join(lines) + "\n")


def report_stem(computed_at: str) -> str:
    """`<date>T<HHMM>` from an ISO `computed_at`.

    v-playbook-desk-b1-5 task 10: the report used to be named by date alone, so a second
    run on the same day silently overwrote the first one's evidence. A `computed_at` with
    no time part stamps `T0000`."""
    ts = str(computed_at)
    if len(ts) < 10 or ts[4] != "-" or ts[7] != "-":
        raise ValueError(f"computed_at must start with an ISO date (YYYY-MM-DD), got {computed_at!r}")
    hhmm = ts[11:16].replace(":", "") if len(ts) >= 16 and ts[10] in "T " else ""
    return f"{ts[:10]}T{hhmm or '0000'}"


def _report_manifest(rows: list[dict], pooled: Optional[list[dict]], computed_at: str, report_name: str) -> dict:
    """The machine-readable companion to the markdown report, same stem so the pair stays
    together and neither is overwritten by the next run."""
    verdicts: dict[str, int] = {}
    for r in rows:
        if r.get("source") == "walkforward":
            v = r.get("verdict", "")
            verdicts[v] = verdicts.get(v, 0) + 1
    pooled_verdicts: dict[str, int] = {}
    for r in pooled or []:
        v = r.get("verdict", "")
        pooled_verdicts[v] = pooled_verdicts.get(v, 0) + 1
    return {"computed_at": computed_at, "report": report_name, "n_rows": len(rows),
            "n_symbols": len({r.get("symbol", "") for r in rows}), "verdicts": verdicts,
            "pooled_verdicts": pooled_verdicts,
            "pooled_with_baseline": sum(1 for r in pooled or [] if int(r.get("baseline_n") or 0) > 0)}


def write_report(root: Path, rows: list[dict], computed_at: str, pooled: Optional[list[dict]] = None) -> Path:
    from research.strategy_hunt.portfolio import pass_lane_trades, replay
    from research.strategy_hunt.sim.calibrate import load_calibration

    files = sorted(trades_dir(root).glob("*.parquet"))
    trades = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else None
    controls = _load_controls(root)
    calibration = load_calibration(root)
    portfolio = replay(pass_lane_trades(trades, rows)) if trades is not None else None
    md = render_report(rows, computed_at, trades, controls, portfolio, calibration, pooled=pooled)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = report_stem(computed_at)
    p = REPORTS_DIR / f"{stem}-strategy-hunt.md"
    p.write_text(md)
    p.with_suffix(".json").write_text(
        json.dumps(_report_manifest(rows, pooled, computed_at, p.name), indent=1, allow_nan=False))
    if portfolio is not None:
        _write_equity_curve_csv(portfolio["equity_curve"], REPORTS_DIR / f"{stem}-equity-curve.csv")
    return p
