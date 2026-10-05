"""Adaptive RSI Turn — archive evaluation and weight fit (Phase 1, spec §3.7).

Inputs: the research cache (5-minute and daily bars) and the tick-rule minute
bins built from `trade_ticks` into `data/micro_archive/<SYM>/<date>.parquet`
(2024-10-01 → 2025-03-11, ~20 names). B (Level 1 book) is absent in the
archive, so only Z, M and A are fitted here; B keeps its prior weight.

Two outputs, written to `research/reports/rsi/<date>-rsi-turn-archive.md`:

1. Pooled fixed-rule simulation of the `rsi_turn` family over its 27-point
   grid with the strategy-hunt simulator (fills, calibrated slippage, stops,
   time stops), exactly as `rsi_fixed_rule_eval.py` did for `rsi_dip`.
2. The weight fit: every in-zone 5-minute bar is an occurrence; label y = 1
   when +1R (1.5 x daily ATR above the next bar's open) is touched before the
   stop within 10 sessions. Logistic regression on the standardised
   components with an L2 penalty and non-negative coefficients (bounded
   L-BFGS), λ chosen by grouped 5-fold cross-validation (groups = symbol x
   month), weights shrunk to the prior by η chosen on out-of-fold expected R,
   θ per regime at ≥ 30 trades. Model written to `data/micro/turn_model.json`.

The archive is a single bull regime (the directive's warning): the per-month
table and the shrinkage η are the honesty controls.
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

from research.rsi_fixed_rule_eval import by, table  # noqa: E402
from research.strategy_hunt.context import load_context_slice  # noqa: E402
from research.strategy_hunt.data.earnings import earnings_days  # noqa: E402
from research.strategy_hunt.data.micro import features_5m, join_micro, load_minute_bins  # noqa: E402
from research.strategy_hunt.data.news import load_news  # noqa: E402
from research.strategy_hunt.data.splits import load_adjusted  # noqa: E402
from research.strategy_hunt.run import load_context  # noqa: E402
from research.strategy_hunt.sim.calibrate import slippage_override_from  # noqa: E402
from research.strategy_hunt.sim.costs import CostModel  # noqa: E402
from research.strategy_hunt.sim.simulator import build_session_table  # noqa: E402
from research.strategy_hunt.walkforward.runner import precompute  # noqa: E402
from strategies.families import get_family  # noqa: E402
from strategies.families import turn_score as ts  # noqa: E402
from strategies.families.indicators import indicator_frame  # noqa: E402
from strategies.families.rsi_turn import DIV_LOOKBACK, _rsi, _slope_sigma, bars_15m, score_bar  # noqa: E402

CACHE = ROOT / "data" / "strategy_hunt"
ARCHIVE = ROOT / "data" / "micro_archive"
MODEL_PATH = ROOT / "data" / "micro" / "turn_model.json"
START, END = pd.Timestamp("2024-10-01"), pd.Timestamp("2025-03-12")
STOP_ATR, HOLD_SESSIONS = 1.5, 10


def symbols_with_archive() -> list[str]:
    return sorted(p.name for p in ARCHIVE.iterdir() if p.is_dir() and any(p.glob("*.parquet")))


def prepare(symbol: str, ctx: dict):
    bars5, daily = load_adjusted(CACHE, symbol)
    if bars5.empty:
        return None
    bars5 = bars5[(bars5.index >= START - pd.Timedelta(days=90)) & (bars5.index < END)]
    bins = load_minute_bins(ARCHIVE, symbol)
    bins = bins[bins["n_trades"] > 0] if not bins.empty else bins
    feats = features_5m(bins) if not bins.empty else pd.DataFrame()
    ind = indicator_frame(bars5, daily, ctx["spy5"], context=load_context_slice(CACHE, symbol))
    ind = join_micro(ind, feats)
    return bars5, daily, ind


def simulate(symbol: str, bars5, daily, ind, ctx: dict) -> list[dict]:
    news = load_news(CACHE, symbol)
    earnings = earnings_days(CACHE, symbol, daily)
    cost_model = CostModel.from_bars(bars5, 6000.0, slippage_override=slippage_override_from(CACHE))
    table_ = build_session_table(bars5)
    fam = get_family("rsi_turn")
    pre = precompute(fam, symbol, bars5, ind, daily, news, cost_model, earnings, ctx["regime_lookup"], table_)
    rows = []
    for key, trades in pre.items():
        params = json.loads(key)
        for t in trades:
            if t.entry_ts < START:
                continue
            rows.append({"symbol": symbol, "family": "rsi_turn", "params": key, "year": t.entry_ts.year,
                         "month": t.entry_ts.strftime("%Y-%m"), "regime": t.regime, "r": t.r_multiple,
                         "pnl_pct": t.pnl_pct, "cost_pct": t.cost_pct, "exit": t.exit_reason, "hold": t.hold_sessions,
                         **{f"p_{k}": v for k, v in params.items()}})
    return rows


def occurrences(symbol: str, bars5, daily, ind, regime_lookup) -> list[dict]:
    """Every in-zone 5-minute bar with its components and the forward outcome."""
    daily_rsi = pd.Series(_rsi(daily["close"].to_numpy()), index=daily.index)
    b15 = bars_15m(bars5)
    rsi15, lows15 = b15["rsi"].to_numpy(), b15["low"].to_numpy()
    pos = b15.index.get_indexer(bars5.index.floor("15min"))
    sigma = _slope_sigma(rsi15)
    opens, highs, lows, closes = (bars5[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    sess = bars5.index.normalize()
    out = []
    zone_cache: dict = {}
    for i in range(len(bars5) - 1):
        if bars5.index[i] < START:
            continue
        row = ind.iloc[i]
        day = sess[i]
        if day not in zone_cache:
            hist = daily_rsi[daily_rsi.index < day]
            zone_cache[day] = ts.zone_stats(hist.to_numpy(), 252) if len(hist) >= 60 else None
        zone = zone_cache[day]
        if zone is None:
            continue
        j = pos[i]
        if j < 0:
            continue
        in_zone, score, detail, vetoes = score_bar(
            row, zone, rsi15[max(0, j - DIV_LOOKBACK - 1):j + 1], lows15[max(0, j - DIV_LOOKBACK - 1):j + 1], sigma,
            float(np.nan_to_num(b15["ret_15"].iloc[j])), float(np.nan_to_num(b15["atr_frac"].iloc[j])), None)
        if not in_zone:
            continue
        atr = ts._finite(row.get("d_atr14"))
        if atr is None or atr <= 0:
            continue
        entry = opens[i + 1]
        risk = STOP_ATR * atr
        stop, target = entry - risk, entry + risk
        y, r_realised, k_end = 0, np.nan, None
        last_sess = sess[i + 1]
        n_sess = 0
        for k in range(i + 1, len(bars5)):
            if sess[k] != last_sess:
                n_sess += 1
                last_sess = sess[k]
                if n_sess >= HOLD_SESSIONS:
                    r_realised = (opens[k] - entry) / risk
                    k_end = k
                    break
            if lows[k] <= stop:
                y, r_realised, k_end = 0, -1.0, k
                break
            if highs[k] >= target:
                y, r_realised, k_end = 1, 1.0, k
                break
        if k_end is None:
            continue
        out.append({"symbol": symbol, "ts": bars5.index[i], "month": bars5.index[i].strftime("%Y-%m"),
                    "regime": regime_lookup(bars5.index[i]) if regime_lookup else "unknown",
                    "Z": detail["Z"], "M": detail["M"], "A": detail["A"], "B": detail["B"], "score": score,
                    "has_tape": str(row.get("mf_aggressor_rule", "") or "") != "",
                    "y": y, "r": r_realised})
    return out


# ── weight fit ────────────────────────────────────────────────────────

def _nll(beta, X, y, lam):
    z = beta[0] + X @ beta[1:]
    p = 1.0 / (1.0 + np.exp(-z))
    eps = 1e-9
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + lam * np.sum(beta[1:] ** 2)


def fit_logit(X, y, lam):
    from scipy.optimize import minimize
    b0 = np.zeros(X.shape[1] + 1)
    bounds = [(None, None)] + [(0.0, None)] * X.shape[1]
    res = minimize(_nll, b0, args=(X, y, lam), method="L-BFGS-B", bounds=bounds)
    return res.x


def grouped_folds(groups: np.ndarray, k: int = 5, seed: int = 0):
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    for f in range(k):
        test_groups = set(uniq[f::k])
        test = np.array([g in test_groups for g in groups])
        yield ~test, test


def fit_weights(occ: pd.DataFrame) -> dict:
    comps = ["Z", "M", "A"]
    X = occ[comps].to_numpy(float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    y = occ["y"].to_numpy(float)
    groups = (occ["symbol"] + ":" + occ["month"]).to_numpy()
    best = None
    for lam in (0.001, 0.01, 0.1, 1.0):
        oof = np.zeros(len(y))
        for tr, te in grouped_folds(groups):
            b = fit_logit(Xs[tr], y[tr], lam)
            oof[te] = 1 / (1 + np.exp(-(b[0] + Xs[te] @ b[1:])))
        ll = -np.mean(y * np.log(oof + 1e-9) + (1 - y) * np.log(1 - oof + 1e-9))
        if best is None or ll < best[0]:
            best = (ll, lam)
    lam = best[1]
    beta = fit_logit(Xs, y, lam)
    raw = {c: float(max(0.0, beta[1 + i])) for i, c in enumerate(comps)}
    tot = sum(raw.values())
    w_fit = {c: (raw[c] / tot if tot > 0 else 0.0) for c in comps}
    prior = {c: ts.PRIOR_WEIGHTS[c] for c in comps}
    ptot = sum(prior.values())
    prior = {c: v / ptot for c, v in prior.items()}
    # shrinkage η on out-of-fold expected R of "enter when score >= θ"
    best_eta = None
    for eta in (0.0, 0.25, 0.5, 0.75, 1.0):
        w = {c: (1 - eta) * w_fit[c] + eta * prior[c] for c in comps}
        s = 100 * (occ[comps].to_numpy(float) @ np.array([w[c] for c in comps]))
        oof_r = []
        for tr, te in grouped_folds(groups):
            theta = _best_theta(s[tr], occ["r"].to_numpy()[tr])
            sel = te & (s >= theta)
            oof_r.append(occ["r"].to_numpy()[sel].mean() if sel.sum() >= 5 else 0.0)
        val = float(np.mean(oof_r))
        if best_eta is None or val > best_eta[0]:
            best_eta = (val, eta)
    eta = best_eta[1]
    w = {c: (1 - eta) * w_fit[c] + eta * prior[c] for c in comps}
    # fold B back in at its prior share
    wb = ts.PRIOR_WEIGHTS["B"]
    weights = {c: w[c] * (1 - wb) for c in comps}
    weights["B"] = wb
    s_all = 100 * (occ[comps].to_numpy(float) @ np.array([w[c] for c in comps]))
    theta_pooled = _best_theta(s_all, occ["r"].to_numpy())
    thetas = {"pooled": theta_pooled}
    for reg, g in occ.assign(s=s_all).groupby("regime"):
        thetas[reg] = _best_theta(g["s"].to_numpy(), g["r"].to_numpy(), min_n=30, fallback=theta_pooled)
    return {"weights": weights, "w_fit": w_fit, "prior": prior, "eta": eta, "lambda": lam, "beta": beta.tolist(),
            "standardise": {"mu": mu.tolist(), "sd": sd.tolist()}, "thetas": thetas,
            "n_occurrences": int(len(occ)), "base_rate": float(y.mean()), "oof_logloss": float(best[0]),
            "oof_expected_r": float(best_eta[0]), "components_fitted": comps,
            "note": "B (Level 1 book) not fitted: absent in the archive; keeps its prior weight.",
            "archive": "trade_ticks 2024-10-01..2025-03-11, tick-rule aggressor",
            "fitted_at": datetime.now().isoformat(timespec="seconds")}


def _best_theta(s: np.ndarray, r: np.ndarray, min_n: int = 30, fallback: float = 60.0) -> float:
    best = None
    for theta in range(30, 95, 5):
        sel = s >= theta
        if sel.sum() < min_n:
            continue
        val = r[sel].mean()
        if best is None or val > best[0]:
            best = (val, float(theta))
    return best[1] if best else fallback


def main() -> int:
    ctx = load_context(CACHE)
    syms = symbols_with_archive()
    sim_rows, occ_rows = [], []
    for sym in syms:
        prep = prepare(sym, ctx)
        if prep is None:
            continue
        bars5, daily, ind = prep
        sim_rows.extend(simulate(sym, bars5, daily, ind, ctx))
        occ_rows.extend(occurrences(sym, bars5, daily, ind, ctx["regime_lookup"]))
        print(f"{sym}: sim={sum(1 for r in sim_rows if r['symbol'] == sym)} occurrences={sum(1 for o in occ_rows if o['symbol'] == sym)}",
              file=sys.stderr)
    sim = pd.DataFrame(sim_rows)
    occ = pd.DataFrame(occ_rows)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [f"# Adaptive RSI Turn — archive evaluation — {stamp}", "",
          f"Symbols with tick-rule minute bins: {', '.join(syms)}. Window {START.date()} → {END.date()} (one bull regime).", ""]
    if not sim.empty:
        pcols = sorted(c for c in sim.columns if c.startswith("p_") and sim[c].notna().any())
        md += ["## rsi_turn — pooled fixed-rule simulation by parameter set", "",
               table(by(sim, pcols).sort_values("exp_r", ascending=False), pcols), ""]
        per = sim.groupby("params")["r"].agg(["mean", "size"])
        elig = per[per["size"] >= 15]
        key = (elig if not elig.empty else per)["mean"].idxmax()
        best = sim[sim.params == key]
        md += [f"### best set `{key}` by month / regime / symbol", "",
               table(by(best, ["month"]), ["month"]), "", table(by(best, ["regime"]), ["regime"]), "",
               table(by(best, ["symbol"]).sort_values("exp_r", ascending=False), ["symbol"]), ""]
    else:
        md += ["## rsi_turn — no simulated trades", ""]
    if len(occ) >= 50:
        model = fit_weights(occ)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_text(json.dumps(model, indent=2))
        md += ["## Weight fit (Z, M, A; B at prior)", "",
               f"occurrences n = {model['n_occurrences']}, base rate P(+1R before stop) = {model['base_rate']:.3f}, "
               f"λ = {model['lambda']}, out-of-fold logloss = {model['oof_logloss']:.4f}", "",
               "| component | fitted w | prior | final w |", "|---|---|---|---|",
               *[f"| {c} | {model['w_fit'].get(c, 0):.3f} | {model['prior'].get(c, 0):.3f} | {model['weights'][c]:.3f} |"
                 for c in ("Z", "M", "A")],
               f"| B | — | {ts.PRIOR_WEIGHTS['B']:.3f} | {model['weights']['B']:.3f} |", "",
               f"shrinkage η = {model['eta']} (0 = pure fit, 1 = prior) chosen on out-of-fold expected R "
               f"= {model['oof_expected_r']:+.3f}; θ: {model['thetas']}", "",
               "By tape availability (occurrences with mf_* data vs without):", "",
               occ.groupby("has_tape")["y"].agg(["size", "mean"]).to_string(), "",
               "Base rate by month:", "", occ.groupby("month")["y"].agg(["size", "mean"]).to_string(), ""]
    else:
        md += [f"## Weight fit skipped: only {len(occ)} occurrences", ""]
    out = ROOT / "research" / "reports" / "rsi" / f"{datetime.now():%Y-%m-%d}-rsi-turn-archive.md"
    out.write_text("\n".join(md))
    if not occ.empty:
        occ.to_parquet(out.with_suffix(".occurrences.parquet"), index=False)
    print(f"sim_trades={len(sim)} occurrences={len(occ)} report={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
