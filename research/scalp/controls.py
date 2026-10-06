"""S1 controls on one cell (t30_h10 long, test month 2025-02): base, real labels,
shuffled labels within symbol-session, features lagged one minute, top gains,
and the exit-policy state-row count. A real edge survives the lag; a
microstructure artefact does not."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from research.scalp.train import load, feature_matrix, choose_pi, nonoverlap, state_rows, Model
from research.scalp.dataset import BarrierSpec, FEATURE_COLUMNS

df = load(); spec = BarrierSpec(0.003, 10); tag = "long"
ycol, rcol, ecol = f"y_{tag}@{spec.name}", f"r_{tag}@{spec.name}", f"exit_{tag}@{spec.name}"
X = feature_matrix(df); m = "2025-02"
tr = (df.month < m).to_numpy() & np.isfinite(X).all(1) & df[ycol].notna().to_numpy()
te = (df.month == m).to_numpy() & np.isfinite(X).all(1) & df[ycol].notna().to_numpy()

def run(Xm, trm, tem, y_tr, r_tr, label):
    mod = Model("multiclass").fit(Xm[trm], y_tr, 300)
    p_tr, p_te = mod.predict(Xm[trm]), mod.predict(Xm[tem])
    s_tr, s_te = p_tr[:, 2] - p_tr[:, 0], p_te[:, 2] - p_te[:, 0]
    pi = choose_pi(s_tr, r_tr, 300)
    sel = np.zeros(len(df), bool); sel[np.where(tem)[0][s_te >= pi]] = True
    keep = nonoverlap(df, sel, ecol); r = df.loc[keep, rcol].to_numpy(float)
    print(f"{label:28s} n={len(r)} mean_r={r.mean():+.3f} win={(r > 0).mean():.2f}", flush=True); return mod

base = df.loc[te, rcol].to_numpy(float); print(f"{'base (every minute)':28s} n={len(base)} mean_r={base.mean():+.3f}", flush=True)
mod = run(X, tr, te, df.loc[tr, ycol].to_numpy(), df.loc[tr, rcol].to_numpy(float), "real labels")
g = (df.loc[tr, "symbol"] + "|" + df.loc[tr, "session"]).to_numpy(); rng = np.random.default_rng(1)
y_s = df.loc[tr, ycol].to_numpy().copy(); r_s = df.loc[tr, rcol].to_numpy(float).copy()
for s in np.unique(g):
    idx = np.where(g == s)[0]; perm = rng.permutation(idx); y_s[idx] = y_s[perm]; r_s[idx] = r_s[perm]
run(X, tr, te, y_s, r_s, "shuffled labels (train)")
Xl = np.vstack([np.full((1, X.shape[1]), np.nan), X[:-1]]); same = (df["symbol"].to_numpy()[1:] == df["symbol"].to_numpy()[:-1]); Xl[1:][~same] = np.nan
tr2, te2 = tr & np.isfinite(Xl).all(1), te & np.isfinite(Xl).all(1)
run(Xl, tr2, te2, df.loc[tr2, ycol].to_numpy(), df.loc[tr2, rcol].to_numpy(float), "features lagged 1 min")
cols = [c for c in FEATURE_COLUMNS if c not in ("imb_ema", "spread_z", "has_book")] + ["symbol_code", "regime_code"]
imp = mod._m.get_score(importance_type="gain"); top = sorted(((float(v), cols[int(k[1:])]) for k, v in imp.items()), reverse=True)[:8]
print("top gain:", ", ".join(f"{n}={v:.0f}" for v, n in top), flush=True)
p_tr = mod.predict(X[tr]); s_tr = p_tr[:, 2] - p_tr[:, 0]; pi = choose_pi(s_tr, df.loc[tr, rcol].to_numpy(float), 300)
tr_sel = np.zeros(len(df), bool); tr_sel[np.where(tr)[0][s_tr >= pi]] = True; tr_keep = np.where(nonoverlap(df, tr_sel, ecol))[0]
Xs, ys, _ = state_rows(df, tr_keep[:500], spec, 1, X, tag); print("policy: state rows from 500 trades:", len(ys), flush=True)
