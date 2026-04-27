"""Meta-labeling filter (P1) — secondary classifier on primary signals.

López de Prado, *Advances in Financial Machine Learning* (2018), §3.6-3.7.
The primary model decides *side* (long/short). The meta-model decides
*take-or-pass* per-event, using the same triple-barrier outcome as the
ground truth (1 = primary's bet hit profit-target before stop-loss).

Pure helpers live here; the orchestrator (``train_meta_model_p1.py``)
wires them into a purged-CV training loop with isotonic calibration on
a held-out fold tail. No I/O in this module.

User-bound clarifications baked in:
1. The operating threshold is selected on the **same chronological tail**
   used for isotonic calibration — never on training, never on OOS.
   Objective: cost-adjusted Sortino on the tail with the P2/P4 anchor
   cost (0.32 R/trade). Tie-break in favor of the higher threshold when
   within ``tiebreak_within_pct`` of the maximum.
2. Sample uniqueness for the meta-set is computed by callers on the
   post-fire ``(event_times, touch_times, groups)`` — this module does
   not reuse any P4 weights.
3. The cost model is imported from ``ml.costs`` (single source of
   truth for P2/P4); cost figures passed in here are R-multiples
   already produced by that module's :func:`cost_in_r`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


# ---- Acceptance gate thresholds (P1 spec) -----------------------------------

P1_BRIER_GATE = 0.22
P1_CALIBRATION_GATE = 0.05
P1_TAKE_RATE_MIN = 0.20
P1_TAKE_RATE_MAX = 0.40
P1_PF_GATE = 1.3
P1_EXPECTANCY_GATE = 0.0
P1_AR1_MEDIAN_GATE = 0.10
P1_AR1_MAX_GATE = 0.20

# P2/P4 cost anchor: 0.32 R/trade (5 bps fee × 2 + 10 bps slip × 2 + 0.1 ×
# √(0.01) × 100 bps impact, normalized to a 1% stop_pct → ≈ 0.32R).
P2_P4_COST_ANCHOR_R = 0.32


class MetaThresholdRejection(RuntimeError):
    """Raised when no threshold yields a usable trade set on the calibration
    tail (e.g. zero candidates above the minimum threshold)."""


@dataclass(frozen=True)
class MetaFoldArtefacts:
    """Public contract for what one purged-CV fold produces."""
    base_clf: Any
    isotonic: Any | None
    scaler: Any
    oos_calibrated_proba: np.ndarray
    tail_proba_calibrated: np.ndarray
    tail_y: np.ndarray
    tail_indices: np.ndarray


# ---- Brier + decile calibration --------------------------------------------


def compute_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Brier score for binary outcomes.

    ``Brier = mean((p_i - y_i)^2)``. AFML p. 78: the canonical proper
    scoring rule for calibrated binary classifiers — its quadratic loss
    penalizes both miscalibration and refinement, unlike accuracy.
    """
    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(y_proba, dtype="float64")
    if y.shape != p.shape:
        raise ValueError(
            f"y_true and y_proba must align; got {y.shape} vs {p.shape}"
        )
    if y.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def calibration_deciles(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    n_bins: int = 10,
    min_count: int = 30,
) -> dict[str, Any]:
    """Decile-by-decile empirical frequency vs. predicted probability.

    For each of ``n_bins`` equal-width bins on [0, 1], reports n,
    ``empirical_p``, ``bin_midpoint``, ``deviation`` (= empirical_p −
    midpoint). ``max_abs_deviation`` aggregates across bins where
    ``n >= min_count``; if no bin qualifies, it is None and the P1
    gate fails (we cannot demonstrate calibration).
    """
    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(y_proba, dtype="float64")
    if y.shape != p.shape:
        raise ValueError("y_true and y_proba must align")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed-on-1 so a proba of exactly 1.0 lands in the last bin.
    bin_idx = np.clip(np.digitize(p, edges, right=False) - 1, 0, n_bins - 1)

    bins: list[dict[str, Any]] = []
    qualifying_devs: list[float] = []
    for i in range(n_bins):
        mask = bin_idx == i
        n = int(mask.sum())
        midpoint = float(0.5 * (edges[i] + edges[i + 1]))
        if n == 0:
            bins.append({
                "bin": i, "n": 0, "midpoint": midpoint,
                "empirical_p": None, "deviation": None,
            })
            continue
        emp = float(y[mask].mean())
        dev = emp - midpoint
        bins.append({
            "bin": i, "n": n, "midpoint": midpoint,
            "empirical_p": emp, "deviation": dev,
        })
        if n >= min_count:
            qualifying_devs.append(abs(dev))

    return {
        "bins": bins,
        "n_qualifying_bins": len(qualifying_devs),
        "max_abs_deviation": (
            float(max(qualifying_devs)) if qualifying_devs else None
        ),
    }


# ---- Cost-adjusted Sortino (per-trade R cost subtracted) -------------------


def cost_adjusted_sortino(
    *,
    net_r: np.ndarray,
    cost_per_trade_r: float,
) -> float:
    """Sortino of the *per-trade* R series after subtracting a flat cost.

    Used by the calibration-tail threshold picker as the objective
    (clarification 1). Annualization is intentionally **not** applied:
    we are comparing thresholds within a single fold's tail, so the
    constant √(bars_per_year) factor would only scale every candidate
    equally and is irrelevant for the argmax.

    Returns 0.0 on empty input (no trades passed the threshold).
    Returns +inf when there are wins and zero downside variance — that
    is correctly the most attractive case.
    """
    r = np.asarray(net_r, dtype="float64")
    if r.size == 0:
        return 0.0
    after_cost = r - float(cost_per_trade_r)
    mean_r = float(after_cost.mean())
    downside = np.minimum(after_cost, 0.0)
    downside_mean_sq = float((downside ** 2).mean())
    if downside_mean_sq == 0.0:
        return math.inf if mean_r > 0 else 0.0
    return mean_r / math.sqrt(downside_mean_sq)


# ---- Threshold sweep + calibration-tail picker ------------------------------


def threshold_sweep_meta(
    *,
    proba: np.ndarray,
    y_true: np.ndarray,
    gross_r: np.ndarray,
    cost_r: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, Any]]:
    """Compute (take-rate, expectancy_net, PF, Sortino) per threshold.

    ``proba`` is the calibrated meta probability. ``gross_r`` is the
    primary's signed R-multiple realization (positive for primary's
    winners). ``cost_r`` is the per-trade cost in R from
    :mod:`ml.costs`. Net R = gross_r − cost_r when meta takes; zero
    otherwise. Returns one row per threshold; rows where nothing is
    taken carry None for PF/expectancy/sortino (never 0.0, which would
    falsely suggest "bad" rather than "absent").
    """
    p = np.asarray(proba, dtype="float64")
    y = np.asarray(y_true, dtype="int64")
    g = np.asarray(gross_r, dtype="float64")
    c = np.asarray(cost_r, dtype="float64")
    if not (p.shape == y.shape == g.shape == c.shape):
        raise ValueError("proba/y/gross_r/cost_r must align")

    rows: list[dict[str, Any]] = []
    n_total = max(p.size, 1)
    for thr in thresholds:
        take = p >= thr
        n_taken = int(take.sum())
        take_rate = n_taken / n_total
        if n_taken == 0:
            rows.append({
                "threshold": float(thr),
                "n_taken": 0,
                "take_rate": float(take_rate),
                "expectancy_net_r": None,
                "profit_factor": None,
                "win_rate": None,
                "sortino_net": None,
            })
            continue

        net_r = g[take] - c[take]
        wins = float(net_r[net_r > 0].sum())
        losses = float(-net_r[net_r < 0].sum())
        pf = (wins / losses) if losses > 0 else float("inf")
        # Sortino on net (no extra cost subtracted — cost_r is already in net_r).
        downside = np.minimum(net_r, 0.0)
        ds_ms = float((downside ** 2).mean())
        if ds_ms == 0.0:
            srt = math.inf if float(net_r.mean()) > 0 else 0.0
        else:
            srt = float(net_r.mean()) / math.sqrt(ds_ms)
        rows.append({
            "threshold": float(thr),
            "n_taken": n_taken,
            "take_rate": float(take_rate),
            "expectancy_net_r": float(net_r.mean()),
            "profit_factor": float(pf) if np.isfinite(pf) else None,
            "win_rate": float(y[take].mean()),
            "sortino_net": float(srt) if np.isfinite(srt) else None,
        })
    return rows


def pick_threshold_on_calibration_tail(
    *,
    tail_proba: np.ndarray,
    tail_y: np.ndarray,
    tail_gross_r: np.ndarray,
    tail_cost_r: np.ndarray,
    thresholds: list[float],
    cost_per_trade_r: float = P2_P4_COST_ANCHOR_R,
    tiebreak_within_pct: float = 0.05,
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    """Pick the operating threshold on the chronological calibration tail.

    Clarification 1 (binding): threshold is picked on the **same**
    chronological tail used for isotonic calibration. The picker never
    sees the OOS test fold and never reads training data.

    Objective: maximize cost-adjusted Sortino with cost = ``cost_per_trade_r``
    (default = P2/P4 anchor 0.32 R/trade). Among candidates whose Sortino
    is within ``tiebreak_within_pct`` of the maximum, prefer the
    **highest threshold** — bias toward selectivity (fewer trades).

    Returns
    -------
    (chosen_row, rationale, sweep)
        ``chosen_row`` is the picked entry from the sweep.
        ``rationale`` is a one-line human-readable explanation.
        ``sweep`` is the full per-threshold table (returned for the report).
    """
    sweep = threshold_sweep_meta(
        proba=tail_proba, y_true=tail_y,
        gross_r=tail_gross_r, cost_r=tail_cost_r,
        thresholds=thresholds,
    )

    # Score each row by cost-adjusted Sortino on the tail. Use
    # ``cost_adjusted_sortino`` over the same per-take-net-R the sweep
    # used, but with the P2/P4 anchor cost subtracted on top of any
    # per-row cost — the spec says the tail-side picker uses the anchor.
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in sweep:
        if row["n_taken"] == 0:
            continue
        thr = row["threshold"]
        take = tail_proba >= thr
        net_r = tail_gross_r[take] - tail_cost_r[take]
        s = cost_adjusted_sortino(net_r=net_r, cost_per_trade_r=cost_per_trade_r)
        candidates.append((s, row))

    if not candidates:
        raise MetaThresholdRejection(
            f"no threshold yields any trades on calibration tail "
            f"(thresholds={thresholds}, n_tail={len(tail_proba)})"
        )

    # Best Sortino. Treat +inf as a real maximum (no losses on the tail
    # is genuinely the best outcome).
    best_score = max(s for s, _ in candidates)

    if not math.isfinite(best_score):
        # All finite candidates are tied at -inf? Pick highest threshold.
        finite = [(s, r) for s, r in candidates if math.isfinite(s)]
        within_band = candidates if not finite else [
            (s, r) for s, r in candidates if math.isinf(s)
        ]
    else:
        # Within-tiebreak band: any candidate whose score is at least
        # (1 - tiebreak_within_pct) × best (handles best>0); for
        # best≤0, fall back to absolute distance.
        if best_score > 0:
            band_floor = best_score * (1.0 - tiebreak_within_pct)
            within_band = [(s, r) for s, r in candidates if s >= band_floor]
        else:
            slack = abs(best_score) * tiebreak_within_pct
            within_band = [(s, r) for s, r in candidates if s >= best_score - slack]

    # Tiebreak: highest threshold wins.
    chosen_score, chosen_row = max(
        within_band, key=lambda sr: (sr[1]["threshold"], sr[0]),
    )

    return chosen_row, (
        f"selected threshold={chosen_row['threshold']:.2f} "
        f"sortino={chosen_score:.3f} (best={best_score:.3f}) "
        f"take_rate={chosen_row['take_rate']:.3f} "
        f"PF={chosen_row['profit_factor']} "
        f"expectancy={chosen_row['expectancy_net_r']}"
    ), sweep


# ---- Acceptance-gate evaluator ---------------------------------------------


def evaluate_p1_gates(
    *,
    brier: float,
    calibration_max_dev: float | None,
    chosen_threshold_row: dict[str, Any],
    per_fold_ar1_median: float,
    per_fold_ar1_max: float,
) -> dict[str, Any]:
    """Apply the six P1 acceptance gates.

    Gates (in spec order):
      1. Brier < 0.22
      2. max |empirical_p − bin_midpoint| ≤ 0.05 (qualifying deciles only)
      3. take_rate ∈ [0.20, 0.40]
      4. profit_factor ≥ 1.3
      5. expectancy_net_r > 0
      6. per-fold AR(1): median < 0.10 AND max < 0.20
    """
    brier_pass = brier < P1_BRIER_GATE
    cal_pass = (
        calibration_max_dev is not None
        and calibration_max_dev <= P1_CALIBRATION_GATE
    )
    tr = chosen_threshold_row.get("take_rate", 0.0)
    take_pass = P1_TAKE_RATE_MIN <= tr <= P1_TAKE_RATE_MAX
    pf = chosen_threshold_row.get("profit_factor")
    pf_pass = (pf is not None) and (pf >= P1_PF_GATE)
    exp = chosen_threshold_row.get("expectancy_net_r")
    exp_pass = (exp is not None) and (exp > P1_EXPECTANCY_GATE)
    ar1_pass = (
        per_fold_ar1_median < P1_AR1_MEDIAN_GATE
        and per_fold_ar1_max < P1_AR1_MAX_GATE
    )

    verdicts: dict[str, Any] = {
        "brier": {"value": brier, "threshold": P1_BRIER_GATE, "pass": bool(brier_pass)},
        "calibration": {
            "value": calibration_max_dev, "threshold": P1_CALIBRATION_GATE,
            "pass": bool(cal_pass),
        },
        "take_rate": {
            "value": tr, "min": P1_TAKE_RATE_MIN, "max": P1_TAKE_RATE_MAX,
            "pass": bool(take_pass),
        },
        "profit_factor": {
            "value": pf, "threshold": P1_PF_GATE, "pass": bool(pf_pass),
        },
        "expectancy": {
            "value": exp, "threshold": P1_EXPECTANCY_GATE, "pass": bool(exp_pass),
        },
        "ar1": {
            "median": per_fold_ar1_median, "max": per_fold_ar1_max,
            "median_threshold": P1_AR1_MEDIAN_GATE,
            "max_threshold": P1_AR1_MAX_GATE,
            "pass": bool(ar1_pass),
        },
    }
    verdicts["all_pass"] = all(
        v["pass"] for k, v in verdicts.items()
        if isinstance(v, dict) and "pass" in v
    )
    return verdicts


# ---- Primary-fire selector -------------------------------------------------


def select_primary_fires(
    *,
    primary_proba: np.ndarray,
    primary_decision_threshold: float,
    label_map: dict[int, str],
) -> dict[str, np.ndarray]:
    """Identify rows where the P4 primary fires BUY or SELL ≥ threshold.

    Returns a dict with mask (boolean per row), sides (+1/−1 per fire),
    max_proba (primary confidence per fire), class_idx (argmax class id).
    """
    proba = np.asarray(primary_proba, dtype="float64")
    if proba.ndim != 2 or proba.shape[1] < 3:
        raise ValueError(
            f"primary_proba must be (n, n_classes>=3); got {proba.shape}"
        )

    inv_map = {v: k for k, v in label_map.items()}
    if "BUY" not in inv_map or "SELL" not in inv_map or "HOLD" not in inv_map:
        raise KeyError("label_map must contain BUY, HOLD, SELL")
    buy_idx = inv_map["BUY"]
    sell_idx = inv_map["SELL"]

    argmax = proba.argmax(axis=1)
    max_p = proba.max(axis=1)
    is_directional = (argmax == buy_idx) | (argmax == sell_idx)
    mask = is_directional & (max_p >= primary_decision_threshold)

    sides = np.where(argmax[mask] == buy_idx, 1, -1).astype(np.int8)
    return {
        "mask": mask,
        "sides": sides,
        "max_proba": max_p[mask],
        "class_idx": argmax[mask],
    }


# ---- Per-fold trainer (isotonic on chronological tail) ---------------------


def train_meta_fold(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray,
    X_test: np.ndarray,
    calibration_tail_frac: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Train one fold's binary meta-classifier with isotonic calibration.

    Pipeline:
      1. ``StandardScaler`` on X_train.
      2. Slice the *chronological tail* of training (``calibration_tail_frac``)
         as the calibration set; fit base XGBoost on the head.
      3. Predict base proba on the calibration tail; fit
         ``IsotonicRegression`` on (proba, y_tail).
      4. Apply isotonic to outer-test base proba → ``oos_calibrated_proba``.
      5. Apply isotonic to tail base proba → ``tail_proba_calibrated``;
         expose alongside ``tail_y`` so the caller's threshold picker can
         operate on the same tail without leaking OOS info.

    The chronological tail (vs. random hold-out) preserves time-order,
    matching purged-CV semantics: calibration cannot leak future
    test-block info because the tail is still strictly earlier than the
    outer-test fold.

    ``calibration_tail_frac == 0`` skips calibration (returns base proba
    on test); tail artefacts are empty in that case.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    if calibration_tail_frac < 0.0 or calibration_tail_frac >= 1.0:
        raise ValueError("calibration_tail_frac must be in [0, 1)")

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    n = len(X_train)
    cal_n = int(round(calibration_tail_frac * n))
    use_cal = (calibration_tail_frac > 0.0 and cal_n >= 30)

    if use_cal:
        head_slice = slice(0, n - cal_n)
        tail_slice = slice(n - cal_n, n)
        X_head = X_tr_s[head_slice]
        y_head = y_train[head_slice]
        w_head = w_train[head_slice]
        X_tail = X_tr_s[tail_slice]
        y_tail = y_train[tail_slice]
        tail_indices = np.arange(n - cal_n, n)
    else:
        X_head, y_head, w_head = X_tr_s, y_train, w_train
        X_tail = None
        y_tail = None
        tail_indices = np.array([], dtype=np.int64)

    # XGBoost requires both classes present. If the head somehow only
    # has one class (rare on real data; common on tiny synthetic), pad
    # with a single zero-weight row.
    if len(np.unique(y_head)) < 2 and len(y_head) > 0:
        missing = 1 - int(y_head[0])
        X_head = np.vstack([X_head, X_head[0:1]])
        y_head = np.concatenate(
            [y_head, np.array([missing], dtype=y_head.dtype)]
        )
        w_head = np.concatenate(
            [w_head, np.array([0.0], dtype=w_head.dtype)]
        )

    base = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=seed,
        eval_metric="logloss",
        objective="binary:logistic", n_jobs=-1,
    )
    base.fit(X_head, y_head, sample_weight=w_head, verbose=False)

    raw_test_proba = base.predict_proba(X_te_s)[:, 1]

    if use_cal and y_tail is not None and len(np.unique(y_tail)) == 2:
        tail_raw = base.predict_proba(X_tail)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(tail_raw, y_tail)
        oos_proba = iso.transform(raw_test_proba)
        tail_proba_cal = iso.transform(tail_raw)
    else:
        iso = None
        oos_proba = raw_test_proba
        # If calibration is disabled or tail is single-class, expose raw
        # tail proba (still correct: caller can still pick a threshold,
        # the calibration just does nothing).
        if use_cal and X_tail is not None:
            tail_proba_cal = base.predict_proba(X_tail)[:, 1]
        else:
            tail_proba_cal = np.array([], dtype="float64")
            y_tail = np.array([], dtype=y_train.dtype)

    return {
        "base_clf": base,
        "isotonic": iso,
        "scaler": scaler,
        "oos_calibrated_proba": oos_proba,
        "tail_proba_calibrated": tail_proba_cal,
        "tail_y": (y_tail if y_tail is not None else np.array([], dtype=y_train.dtype)),
        "tail_indices": tail_indices,
    }
