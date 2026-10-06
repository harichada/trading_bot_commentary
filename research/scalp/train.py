"""S1: walk-forward baseline for the Adaptive Scalp Engine (spec §5–6).

For every barrier cell (tau x horizon) and side:
  * entry model — LightGBM multiclass over y in {-1, 0, +1} on the causal
    minute features (+ symbol, regime); score = P(target first) − P(stop
    first); the entry threshold π is chosen on the TRAINING months to
    maximise mean net R with at least MIN_TRADES trades;
  * trades are non-overlapping per symbol (one open position at a time);
  * exit policy — LightGBM regressor of the R still to come at every minute of
    the training trades (features at that minute + position state); on the
    test trades the policy exits the first minute the prediction is < 0;
  * walk-forward by calendar month: train on months < m, test on m.
Everything is net of the cost baked into the labels. The 3090 is used when
LightGBM's GPU build is available; otherwise CPU (same numbers, slower).

Output: research/reports/scalp/<date>-s1-baseline.md and a JSON with the per-
cell numbers; data/scalp/models/ is not written (S1 is an answer, not a model).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.scalp.dataset import FEATURE_COLUMNS, GRID_H, GRID_TAU, BarrierSpec  # noqa: E402

DATA = ROOT / "data" / "scalp"
MIN_TRADES = 100
STATE_COLS = ("minutes_held", "mtm_r", "mfe_r", "mae_r")
warnings.filterwarnings("ignore")


def _cuda_available() -> bool:
    try:
        import xgboost as xgb
        xgb.train({"device": "cuda", "tree_method": "hist", "verbosity": 0},
                  xgb.DMatrix(np.random.default_rng(0).normal(size=(64, 3)), np.arange(64) % 2), 1)
        return True
    except Exception:
        return False


class Model:
    """XGBoost on the 3090 (device=cuda) when it works, LightGBM on CPU otherwise.
    `kind` = "multiclass" (3 classes, predict → (n, 3) probabilities) or "regression"."""
    USE_CUDA = None

    def __init__(self, kind: str) -> None:
        self.kind = kind
        if Model.USE_CUDA is None:
            Model.USE_CUDA = _cuda_available()
        self.cuda = Model.USE_CUDA
        self._m = None

    def fit(self, X, y, rounds: int) -> "Model":
        if self.cuda:
            import xgboost as xgb
            params = {"device": "cuda", "tree_method": "hist", "max_depth": 6, "eta": 0.05, "subsample": 0.8,
                      "colsample_bytree": 0.8, "min_child_weight": 50, "lambda": 10.0, "verbosity": 0, "max_bin": 128}
            if self.kind == "multiclass":
                params.update(objective="multi:softprob", num_class=3)
                yy = (y + 1).astype(int)
            else:
                params.update(objective="reg:squarederror")
                yy = y
            self._m = xgb.train(params, xgb.DMatrix(X, yy), num_boost_round=rounds)
        else:
            import lightgbm as lgb
            params = {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200, "feature_fraction": 0.8,
                      "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "num_threads": 8}
            if self.kind == "multiclass":
                params.update(objective="multiclass", num_class=3)
                yy = (y + 1).astype(int)
            else:
                params.update(objective="regression_l2")
                yy = y
            self._m = lgb.train(params, lgb.Dataset(X, yy), num_boost_round=rounds)
        return self

    def predict(self, X) -> np.ndarray:
        if self.cuda:
            import xgboost as xgb
            return self._m.predict(xgb.DMatrix(X))
        return self._m.predict(X)


def load() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in sorted(DATA.glob("*.parquet")) if p.name != "manifest.json"]
    df = pd.concat(frames, ignore_index=True)
    df["month"] = pd.to_datetime(df["t_open"]).dt.strftime("%Y-%m")
    df["symbol_code"] = df["symbol"].astype("category").cat.codes
    df["regime_code"] = df["regime"].astype("category").cat.codes
    # label exit indices are session-local; the policy code needs global row indices
    sess = df["symbol"] + "|" + df["session"]
    df["session_start"] = df.groupby(sess).cumcount().rsub(np.arange(len(df)))
    return df


def feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in FEATURE_COLUMNS if c not in ("imb_ema", "spread_z", "has_book")] + ["symbol_code", "regime_code"]
    return df[cols].to_numpy(float)


def nonoverlap(df: pd.DataFrame, selected: np.ndarray, exit_col: str) -> np.ndarray:
    """Greedy: within each symbol-session, keep a selected minute only if the
    previous kept trade has exited. Returns a boolean mask."""
    keep = np.zeros(len(df), dtype=bool)
    sess = (df["symbol"] + "|" + df["session"]).to_numpy()
    exits = df[exit_col].to_numpy()
    idx_in_session = df.groupby(sess).cumcount().to_numpy()
    order = np.argsort(sess, kind="stable")
    open_until: dict = {}
    for j in order:
        if not selected[j]:
            continue
        s = sess[j]
        if idx_in_session[j] <= open_until.get(s, -1):
            continue
        keep[j] = True
        open_until[s] = exits[j]
    return keep


def train_entry(X, y, params=None):
    return Model("multiclass").fit(X, y, 300)


def choose_pi(score: np.ndarray, r: np.ndarray, min_trades: int) -> float:
    best = (None, -1e9)
    for q in np.linspace(0.5, 0.995, 40):
        pi = float(np.quantile(score, q))
        sel = score >= pi
        if sel.sum() < min_trades:
            break
        m = float(r[sel].mean())
        if m > best[1]:
            best = (pi, m)
    return best[0] if best[0] is not None else float(np.quantile(score, 0.9))


def state_rows(df_sess: pd.DataFrame, trades_idx: np.ndarray, spec: BarrierSpec, side: int, Xf: np.ndarray,
               tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position-state rows for the exit regressor: for each trade, every
    minute until the fixed-rule exit, with the remaining R as the target."""
    close = df_sess["close"].to_numpy(float)
    exit_idx = df_sess[f"exit_{tag}@{spec.name}"].to_numpy() + df_sess["session_start"].to_numpy()   # global
    r_final = df_sess[f"r_{tag}@{spec.name}"].to_numpy(float)
    X_rows, y_rows, trade_id = [], [], []
    for t in trades_idx:
        k_exit = int(exit_idx[t])
        entry = close[t] * (1 + side * spec.cost)
        mfe = mae = 0.0
        for m in range(t + 1, k_exit + 1):
            mtm = side * (close[m] * (1 - side * spec.cost) - entry) / (entry * spec.sigma)
            mfe, mae = max(mfe, mtm), min(mae, mtm)
            X_rows.append(np.concatenate([Xf[m], [m - t, mtm, mfe, mae]]))
            y_rows.append(r_final[t] - mtm)
            trade_id.append(t)
    if not X_rows:
        return np.zeros((0, Xf.shape[1] + 4)), np.zeros(0), np.zeros(0, dtype=int)
    return np.asarray(X_rows), np.asarray(y_rows), np.asarray(trade_id)


def apply_exit_policy(df_sess: pd.DataFrame, trades_idx: np.ndarray, spec: BarrierSpec, side: int, Xf: np.ndarray,
                      tag: str, model) -> np.ndarray:
    """Realised R per trade when exiting the first minute the predicted
    remaining R is negative (else the fixed rule's R)."""
    close = df_sess["close"].to_numpy(float)
    exit_idx = df_sess[f"exit_{tag}@{spec.name}"].to_numpy() + df_sess["session_start"].to_numpy()   # global
    r_final = df_sess[f"r_{tag}@{spec.name}"].to_numpy(float)
    out = np.zeros(len(trades_idx))
    for n, t in enumerate(trades_idx):
        k_exit = int(exit_idx[t])
        entry = close[t] * (1 + side * spec.cost)
        mfe = mae = 0.0
        realised = r_final[t]
        for m in range(t + 1, k_exit + 1):
            mtm = side * (close[m] * (1 - side * spec.cost) - entry) / (entry * spec.sigma)
            mfe, mae = max(mfe, mtm), min(mae, mtm)
            if m == k_exit:
                break
            pred = model.predict(np.concatenate([Xf[m], [m - t, mtm, mfe, mae]])[None, :])[0]
            if pred < 0:
                realised = mtm
                break
        out[n] = realised
    return out


def run_cell(df: pd.DataFrame, spec: BarrierSpec, side: int, params_cls: dict = None, params_reg: dict = None) -> dict:
    tag = "long" if side > 0 else "short"
    ycol, rcol, ecol = f"y_{tag}@{spec.name}", f"r_{tag}@{spec.name}", f"exit_{tag}@{spec.name}"
    months = sorted(df["month"].unique())
    X_all = feature_matrix(df)
    res = {"cell": spec.name, "side": tag, "months": []}
    for m in months[1:]:
        tr, te = (df["month"] < m).to_numpy(), (df["month"] == m).to_numpy()
        ok_tr = tr & np.isfinite(X_all).all(1) & df[ycol].notna().to_numpy()
        ok_te = te & np.isfinite(X_all).all(1) & df[ycol].notna().to_numpy()
        if ok_tr.sum() < 5000 or ok_te.sum() < 1000:
            continue
        model = train_entry(X_all[ok_tr], df.loc[ok_tr, ycol].to_numpy(), params_cls)
        p_tr = model.predict(X_all[ok_tr]); p_te = model.predict(X_all[ok_te])
        s_tr, s_te = p_tr[:, 2] - p_tr[:, 0], p_te[:, 2] - p_te[:, 0]
        r_tr, r_te = df.loc[ok_tr, rcol].to_numpy(float), df.loc[ok_te, rcol].to_numpy(float)
        pi = choose_pi(s_tr, r_tr, MIN_TRADES * 3)
        # non-overlapping selected trades in the test month
        sel_full = np.zeros(len(df), dtype=bool); sel_full[np.where(ok_te)[0][s_te >= pi]] = True
        keep = nonoverlap(df, sel_full, ecol)
        trades = np.where(keep)[0]
        base_full = np.zeros(len(df), dtype=bool); base_full[np.where(ok_te)[0]] = True
        base_keep = nonoverlap(df, base_full, ecol)
        r_sel = df.loc[keep, rcol].to_numpy(float)
        r_base = df.loc[base_keep, rcol].to_numpy(float)
        row = {"month": m, "n_train": int(ok_tr.sum()), "n_test_minutes": int(ok_te.sum()), "pi": round(pi, 4),
               "n_trades": int(len(r_sel)), "win": float((r_sel > 0).mean()) if len(r_sel) else np.nan,
               "mean_r": float(r_sel.mean()) if len(r_sel) else np.nan, "sum_r": float(r_sel.sum()),
               "base_n": int(len(r_base)), "base_mean_r": float(r_base.mean()) if len(r_base) else np.nan}
        # AUC of target-first vs rest on the test month
        try:
            from sklearn.metrics import roc_auc_score
            yte = df.loc[ok_te, ycol].to_numpy()
            row["auc_target"] = float(roc_auc_score((yte == 1).astype(int), p_te[:, 2]))
        except Exception:
            row["auc_target"] = np.nan
        # exit policy: fit on the training months' trades (selected by the same π), test on this month's trades
        tr_sel = np.zeros(len(df), dtype=bool); tr_sel[np.where(ok_tr)[0][s_tr >= pi]] = True
        tr_keep = np.where(nonoverlap(df, tr_sel, ecol))[0]
        if len(tr_keep) >= 50 and len(trades) >= 10:
            rng = np.random.default_rng(0)
            tr_keep = rng.choice(tr_keep, size=min(len(tr_keep), 3000), replace=False)
            Xs, ys, _ = state_rows(df, tr_keep, spec, side, X_all, tag)
            if len(ys) < 200:
                res["months"].append(row); continue
            reg = Model("regression").fit(Xs, ys, 200)
            r_policy = apply_exit_policy(df, trades, spec, side, X_all, tag, reg)
            row["policy_mean_r"] = float(r_policy.mean()); row["policy_sum_r"] = float(r_policy.sum())
            row["policy_win"] = float((r_policy > 0).mean())
        res["months"].append(row)
    if res["months"]:
        mm = pd.DataFrame(res["months"])
        w = mm["n_trades"].clip(lower=0)
        res["summary"] = {"n_trades": int(w.sum()), "mean_r": float((mm["mean_r"] * w).sum() / max(w.sum(), 1)),
                          "sum_r": float(mm["sum_r"].sum()), "base_mean_r": float((mm["base_mean_r"] * mm["base_n"]).sum() / max(mm["base_n"].sum(), 1)),
                          "months_positive": int((mm["mean_r"] > 0).sum()), "months": int(len(mm)),
                          "policy_sum_r": float(mm.get("policy_sum_r", pd.Series(dtype=float)).sum()),
                          "auc_target": float(mm["auc_target"].mean())}
    return res


def main() -> int:
    t0 = time.time()
    df = load()
    print(f"dataset rows={len(df)} symbols={df['symbol'].nunique()} months={sorted(df['month'].unique())}", file=sys.stderr)
    params_cls = params_reg = None
    print("device:", "cuda (xgboost)" if _cuda_available() else "cpu (lightgbm)", file=sys.stderr)
    results = []
    for tau in GRID_TAU:
        for h in GRID_H:
            spec = BarrierSpec(tau, h)
            for side in (1, -1):
                r = run_cell(df, spec, side, params_cls, params_reg)
                results.append(r)
                s = r.get("summary", {})
                print(f"{spec.name} {r['side']:5} trades={s.get('n_trades', 0):5} mean_r={s.get('mean_r', float('nan')):+.3f} "
                      f"base={s.get('base_mean_r', float('nan')):+.3f} policy_sum={s.get('policy_sum_r', float('nan')):+.1f} "
                      f"auc={s.get('auc_target', float('nan')):.3f} pos_months={s.get('months_positive', 0)}/{s.get('months', 0)} "
                      f"[{time.time() - t0:.0f}s]", file=sys.stderr, flush=True)
    out_dir = ROOT / "research" / "reports" / "scalp"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now()
    (out_dir / f"{stamp:%Y-%m-%d}-s1-baseline.json").write_text(json.dumps(results, indent=1, default=float))
    md = [f"# Adaptive Scalp Engine — S1 baseline — {stamp:%Y-%m-%d %H:%M}", "",
          f"Dataset: {len(df):,} minute rows, {df['symbol'].nunique()} symbols, months {sorted(df['month'].unique())}.",
          "Tick-rule archive tape (no book). Costs inside the labels: 0.05% per side. Walk-forward by month,",
          "train on earlier months only. Trades are non-overlapping per symbol. `base` = every minute taken",
          "(non-overlapping), i.e. the fixed rule with no model. `policy` = the learned stay-or-exit rule", "",
          "| cell | side | trades | win | mean R | sum R | base mean R | policy sum R | AUC(target) | +months |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        s = r.get("summary")
        if not s:
            md.append(f"| {r['cell']} | {r['side']} | 0 | | | | | | | |"); continue
        mm = pd.DataFrame(r["months"])
        md.append(f"| {r['cell']} | {r['side']} | {s['n_trades']} | {(mm['win'] * mm['n_trades']).sum() / max(mm['n_trades'].sum(), 1):.0%} | "
                  f"{s['mean_r']:+.3f} | {s['sum_r']:+.1f} | {s['base_mean_r']:+.3f} | {s['policy_sum_r']:+.1f} | "
                  f"{s['auc_target']:.3f} | {s['months_positive']}/{s['months']} |")
    md += ["", "## Per month (cells with the best pooled mean R)", ""]
    best = sorted([r for r in results if r.get("summary")], key=lambda r: -r["summary"]["mean_r"])[:3]
    for r in best:
        md += [f"### {r['cell']} {r['side']}", "", pd.DataFrame(r["months"]).round(3).to_string(index=False), ""]
    (out_dir / f"{stamp:%Y-%m-%d}-s1-baseline.md").write_text("\n".join(md))
    print(f"report={out_dir / f'{stamp:%Y-%m-%d}-s1-baseline.md'} elapsed={time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
