import numpy as np
import pandas as pd
import pytest


def _bars(sessions=70, vol_scale=1.0, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2025-01-02", periods=sessions)
    frames, px = [], 100.0
    for d in days:
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", name="ts")
        close = px + rng.normal(0, 0.05 * vol_scale, 78).cumsum()
        frames.append(pd.DataFrame({"open": close - 0.01, "high": close + 0.2 * vol_scale, "low": close - 0.2 * vol_scale,
                                    "close": close, "volume": 5000}, index=idx))
        px = float(close[-1])
    return pd.concat(frames)


def test_characteristics_keys_and_ranges():
    from research.strategy_hunt.fit.coldstart import symbol_characteristics, FEATURES
    from research.strategy_hunt.data.bars import to_daily
    b = _bars()
    d = to_daily(b)
    news = pd.DataFrame({"created_at": pd.to_datetime(["2025-02-03 10:00", "2025-03-03 10:00"]), "headline": ["a", "b"],
                         "source": ["x", "x"], "sentiment": [0.1, -0.1]})
    f = symbol_characteristics(b, d, news, spy_daily=d)
    assert tuple(f) == FEATURES
    assert f["atr_pct"] > 0 and f["adv_dollars"] > 0 and 0 <= f["gap_freq"] <= 1
    assert f["beta_spy"] == pytest.approx(1.0, abs=1e-6)          # regressed on itself
    assert f["news_per_month"] == pytest.approx(2 / (60 / 21))
    assert 0.0002 <= f["half_spread"] <= 0.005
    assert not np.isnan(f["hurst"]) and not np.isnan(f["vr_5m"])


def test_characteristics_scale_with_volatility():
    from research.strategy_hunt.fit.coldstart import symbol_characteristics
    from research.strategy_hunt.data.bars import to_daily
    lo, hi = _bars(vol_scale=1.0), _bars(vol_scale=4.0, seed=1)
    a = symbol_characteristics(lo, to_daily(lo), pd.DataFrame())
    b = symbol_characteristics(hi, to_daily(hi), pd.DataFrame())
    assert b["atr_pct"] > a["atr_pct"] and b["beta_spy"] == 0.0


def test_train_and_predict_coldstart_only_emits_watch_or_fail():
    from research.strategy_hunt.fit.coldstart import train_coldstart, predict_coldstart, FEATURES
    rng = np.random.default_rng(0)
    syms = [f"S{i}" for i in range(80)]
    X = pd.DataFrame(rng.normal(size=(80, len(FEATURES))), index=syms, columns=FEATURES)
    label = (X["trend_eff"] > 0).astype(int)                       # learnable rule
    rows = [{"symbol": s, "strategy": "orb", "side": "long", "verdict": "PASS" if label[s] else "FAIL"} for s in syms]
    rows += [{"symbol": s, "strategy": "mr_20d", "side": "long", "verdict": "FAIL"} for s in syms]   # no positives -> skipped
    models = train_coldstart(X, rows)
    assert set(models) == {"orb:long"} and models["orb:long"]["cv_auc"] > 0.7
    newX = pd.DataFrame({f: [2.0, -2.0] for f in FEATURES}, index=["NEW1", "NEW2"])
    preds = predict_coldstart(models, newX, "2025-09-25T00:00:00")
    assert len(preds) == 2 and {p["verdict"] for p in preds} <= {"WATCH", "FAIL"}
    assert all(p["source"] == "coldstart" and p["n"] == 0 and p["score"] == 0.0 for p in preds)
    by = {p["symbol"]: p["verdict"] for p in preds}
    assert by["NEW1"] == "WATCH" and by["NEW2"] == "FAIL"
