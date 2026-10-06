"""Scalp S1 dataset: causal features, triple-barrier labels (vectorised == scalar), remaining R."""
import numpy as np
import pandas as pd
import pytest

from research.scalp.dataset import (BarrierSpec, FEATURE_COLUMNS, build_session, features, first_touch, labels,
                                    remaining_r)


def _bins(n=120, seed=0, start="2025-02-03 09:30"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.0015, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.0008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0008, n)))
    vol = rng.integers(1000, 50000, n).astype(float)
    buy = vol * rng.uniform(0.2, 0.8, n)
    return pd.DataFrame({"t_open": pd.date_range(start, periods=n, freq="min"), "buy_vol": buy, "sell_vol": vol - buy,
                         "vol": vol, "n_trades": rng.integers(50, 500, n), "high": high, "low": low, "close": close,
                         "vwap": close, "large_vol": vol * 0.1, "large_buy": vol * 0.06, "large_sell": vol * 0.04})


def test_features_shape_causality_and_bounds():
    b = _bins()
    f = features(b)
    assert len(f) == len(b) and set(FEATURE_COLUMNS) <= set(f.columns)
    assert f["minute_of_day"].iloc[0] == 0 and f["minute_of_day"].iloc[-1] == 119
    assert f["sell_share_5"].dropna().between(0, 1).all() and f["cvd_15"].dropna().between(-1, 1).all()
    assert f["range_pos_15"].dropna().between(0, 1).all() and f["has_book"].eq(0).all()
    # causal: changing minute 100 cannot change any feature before it
    b2 = b.copy(); b2.loc[100, ["close", "high", "low", "buy_vol"]] *= 1.05
    f2 = features(b2)
    cols = [c for c in FEATURE_COLUMNS if c not in ("print_rate_rel",)]
    pd.testing.assert_frame_equal(f.loc[:99, cols], f2.loc[:99, cols])


@pytest.mark.parametrize("tau,h", [(0.002, 10), (0.003, 20), (0.005, 30)])
def test_vectorised_labels_match_scalar_first_touch(tau, h):
    b = _bins(n=200, seed=3)
    spec = BarrierSpec(tau, h)
    lab = labels(b, spec)
    high, low, close = (b[c].to_numpy(float) for c in ("high", "low", "close"))
    for i in range(0, 199, 7):
        for side, tag in ((1, "long"), (-1, "short")):
            y, k, r = first_touch(high, low, close, i, side, spec)
            assert lab[f"y_{tag}"][i] == y and lab[f"exit_{tag}"][i] == k, (i, tag)
            assert lab[f"r_{tag}"][i] == pytest.approx(r, abs=1e-9), (i, tag)
    assert np.isnan(lab["y_long"].iloc[-1])


def test_label_economics():
    """A winner pays tau/sigma minus costs; a loser pays about -1 minus costs; costs are charged both ways."""
    spec = BarrierSpec(0.003, 10, cost=0.0005)
    n = 12
    close = np.full(n, 100.0); high = close.copy(); low = close.copy()
    high[3] = 100.5                                     # target hit at minute 3
    b = _bins(n); b["close"], b["high"], b["low"] = close, high, low
    lab = labels(b, spec)
    assert lab["y_long"][0] == 1 and lab["exit_long"][0] == 3
    # barriers are set from the PAID price (entry cost moves them away), the exit crosses the spread again
    assert lab["r_long"][0] == pytest.approx(((1 + spec.tau) * (1 - spec.cost) - 1) / spec.sigma, rel=1e-6)
    low2 = close.copy(); low2[2] = 99.5
    b2 = _bins(n); b2["close"], b2["high"], b2["low"] = close, close.copy(), low2
    lab2 = labels(b2, spec)
    assert lab2["y_long"][0] == -1 and lab2["r_long"][0] < -1.0
    # a minute touching both barriers counts as the stop
    hi3 = close.copy(); hi3[1] = 100.6; lo3 = close.copy(); lo3[1] = 99.6
    b3 = _bins(n); b3["close"], b3["high"], b3["low"] = close, hi3, lo3
    assert labels(b3, spec)["y_long"][0] == -1


def test_remaining_r_walks_to_the_exit():
    b = _bins(n=80, seed=5)
    spec = BarrierSpec(0.003, 20)
    rr = remaining_r(b, spec, side=1, entry_i=10)
    assert len(rr) >= 1 and (rr["minutes_held"] == np.arange(1, len(rr) + 1)).all()
    assert rr["mfe_r"].is_monotonic_increasing and rr["mae_r"].is_monotonic_decreasing
    # remaining + mark-to-market == final R at every minute
    final = rr["mtm_r"].iloc[-1] + rr["remaining_r"].iloc[-1]
    assert np.allclose(rr["mtm_r"] + rr["remaining_r"], final)


def test_build_session_has_all_label_columns():
    df = build_session(_bins(), "MU")
    assert df["symbol"].eq("MU").all()
    assert "y_long@t20_h10" in df.columns and "r_short@t50_h30" in df.columns
    assert df.shape[0] == 120
