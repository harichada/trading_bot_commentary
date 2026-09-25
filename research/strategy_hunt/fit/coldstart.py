"""Cold-start fit: symbol characteristics -> per-lane PASS probability. Emits WATCH or FAIL only, never PASS."""
from __future__ import annotations

import json
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from research.strategy_hunt.fit.writer import ROW_KEYS
from research.strategy_hunt.sim.costs import monthly_half_spread

FEATURES = ("atr_pct", "vr_5m", "vr_daily", "hurst", "trend_eff", "gap_freq", "half_spread", "adv_dollars",
            "beta_spy", "news_per_month")


def _variance_ratio(logret: np.ndarray, q: int) -> float:
    if logret.size < 5 * q:
        return math.nan
    v1 = np.var(logret, ddof=1)
    if v1 <= 0:
        return math.nan
    agg = np.convolve(logret, np.ones(q), mode="valid")[::q]
    return float(np.var(agg, ddof=1) / (q * v1))


def _hurst(close: np.ndarray) -> float:
    if close.size < 20:
        return math.nan
    lr = np.diff(np.log(close))
    ks, sds = [], []
    for k in (1, 2, 4, 8):
        if lr.size < 3 * k:
            break
        agg = np.convolve(lr, np.ones(k), mode="valid")[::k]
        if agg.size >= 3 and np.std(agg) > 0:
            ks.append(math.log(k))
            sds.append(math.log(np.std(agg, ddof=1)))
    return float(np.polyfit(ks, sds, 1)[0]) if len(ks) >= 3 else math.nan


def symbol_characteristics(bars5: pd.DataFrame, daily: pd.DataFrame, news: pd.DataFrame,
                           spy_daily: Optional[pd.DataFrame] = None, sessions: int = 60) -> dict[str, float]:
    d = daily.tail(sessions)
    out: dict[str, float] = {k: math.nan for k in FEATURES}
    if d.empty:
        return out
    b = bars5[bars5.index >= d.index[0]]
    prev_close = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev_close).abs(), (d["low"] - prev_close).abs()], axis=1).max(axis=1)
    out["atr_pct"] = float((tr / d["close"]).mean() * 100)
    lr5 = np.diff(np.log(b["close"].to_numpy())) if len(b) > 1 else np.array([])
    out["vr_5m"] = _variance_ratio(lr5, 6)
    lrd = np.diff(np.log(d["close"].to_numpy()))
    out["vr_daily"] = _variance_ratio(lrd, 5)
    out["hurst"] = _hurst(d["close"].to_numpy())
    c = d["close"].to_numpy()
    if c.size >= 21:
        path = np.abs(np.diff(c[-21:])).sum()
        out["trend_eff"] = float(abs(c[-1] - c[-21]) / path) if path > 0 else math.nan
    gaps = (d["open"] / prev_close - 1).abs().dropna()
    out["gap_freq"] = float((gaps >= 0.02).mean()) if len(gaps) else math.nan
    hs = monthly_half_spread(b) if len(b) > 2 else pd.Series(dtype=float)
    out["half_spread"] = float(hs.median()) if len(hs) else math.nan
    out["adv_dollars"] = float((d["close"] * d["volume"]).mean())
    if spy_daily is not None and not spy_daily.empty:
        joined = pd.concat([d["close"].pct_change(), spy_daily["close"].pct_change().reindex(d.index)], axis=1).dropna()
        if len(joined) >= 10 and joined.iloc[:, 1].var() > 0:
            out["beta_spy"] = float(np.polyfit(joined.iloc[:, 1], joined.iloc[:, 0], 1)[0])
    else:
        out["beta_spy"] = 0.0
    if news is not None and not news.empty:
        n_items = int(((news["created_at"] >= d.index[0]) & (news["created_at"] < d.index[-1] + pd.Timedelta(days=1))).sum())
    else:
        n_items = 0
    out["news_per_month"] = float(n_items / (len(d) / 21.0))
    return {k: out[k] for k in FEATURES}


def train_coldstart(features: pd.DataFrame, fit_rows: list[dict], min_rows: int = 30, min_pos: int = 8) -> dict[str, dict]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    lanes: dict[str, dict[str, int]] = {}
    for r in fit_rows:
        lanes.setdefault(f"{r['strategy']}:{r['side']}", {})[r["symbol"]] = int(r["verdict"] == "PASS")
    models: dict[str, dict] = {}
    for lane, labels in lanes.items():
        syms = [s for s in labels if s in features.index]
        y = np.array([labels[s] for s in syms])
        if len(syms) < min_rows or y.sum() < min_pos or y.sum() == len(y):
            continue
        X = features.loc[syms, list(FEATURES)].to_numpy(dtype=float)
        model = HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, random_state=0)
        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            auc = float(roc_auc_score(y, p))
        except ValueError:
            auc = math.nan
        model.fit(X, y)
        models[lane] = {"model": model, "cv_auc": auc, "n": int(len(y)), "n_pos": int(y.sum())}
    return models


def predict_coldstart(models: dict[str, dict], features: pd.DataFrame, computed_at: str, threshold: float = 0.6) -> list[dict]:
    rows: list[dict[str, Any]] = []
    if features.empty:
        return rows
    X = features[list(FEATURES)].to_numpy(dtype=float)
    for lane, info in sorted(models.items()):
        strategy, side = lane.split(":")
        probs = info["model"].predict_proba(X)[:, 1]
        for sym, p in zip(features.index, probs):
            row = {k: math.nan for k in ROW_KEYS}
            row.update({"symbol": str(sym), "strategy": strategy, "side": side, "computed_at": computed_at, "window_end": "",
                        "n": 0, "sessions": 0, "score": 0.0, "verdict": "WATCH" if p >= threshold else "FAIL",
                        "failed_floors": json.dumps(["coldstart"]), "params_json": "{}", "regime_exp_json": "{}",
                        "source": "coldstart"})
            rows.append({k: row[k] for k in ROW_KEYS})
    return rows
