"""v-turn-score-2026-10-05: spec §3 — every component in [0, 1], the score in
[0, 100], for any finite or non-finite input; the worked formulas."""
from __future__ import annotations

import math

import numpy as np
import pytest

from strategies.families import turn_score as ts


# ── Z ──────────────────────────────────────────────────────────────────

def test_zone_stats_fallback_and_percentiles():
    st = ts.zone_stats([50.0] * 50)
    assert st.q05 == 25.0 and st.n == 50
    hist = np.linspace(10, 90, 300)
    st = ts.zone_stats(hist)
    assert st.n == 252 and 15 < st.q05 < 30 and st.iqr > 10
    st2 = ts.zone_stats(list(hist) + [float("nan"), None])
    assert st2.q05 == st.q05


def test_zone_component_depth_and_edges():
    st = ts.ZoneStats(q05=24.0, q25=40.0, q50=50.0, q75=60.0, n=252)   # IQR 20
    assert ts.zone_component(30.0, st) == (False, 0.0)
    in_zone, z = ts.zone_component(24.0, st)
    assert in_zone and z == 0.0
    in_zone, z = ts.zone_component(14.0, st)                            # 10 below = 0.5 IQR → 0.5/1.5
    assert in_zone and z == pytest.approx(0.5 / 1.5)
    in_zone, z = ts.zone_component(-50.0, st)
    assert in_zone and z == 1.0
    assert ts.zone_component(float("nan"), st) == (False, 0.0)


# ── M ──────────────────────────────────────────────────────────────────

def test_momentum_rising_inflection_and_divergence():
    # RSI falling then turning up over the last three bars; price made a lower low, RSI a higher low
    rho = [40, 38, 36, 34, 32, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 22, 24, 26, 28, 30]
    lows = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 85.5, 86, 86.5, 87]
    # make the recent window's low lower than the earlier window but RSI low higher
    lows[-6:] = [84, 84.5, 85, 85.5, 86, 86.5]
    rho[-6:] = [23, 24, 26, 28, 30, 32]
    m, parts = ts.momentum_component(rho, lows, slope_sigma=4.0)
    assert parts["slope"] == 32 - 26 and parts["m_slope"] == pytest.approx(6 / 8)
    assert parts["div"] == 1.0
    assert 0.0 < m <= 1.0 and m == pytest.approx(0.5 * 0.75 + 0.3 * parts["inflect"] + 0.2 * 1.0)


def test_momentum_is_one_sided_and_tolerates_short_or_nan_series():
    falling = list(range(40, 20, -1))
    m, parts = ts.momentum_component(falling, [100] * 20, slope_sigma=2.0)
    assert m == 0.0 and parts["m_slope"] < 0 and parts["inflect"] == 0.0
    assert ts.momentum_component([30, 31], [1, 1], 2.0)[0] == 0.0
    assert ts.momentum_component([30, 31, float("nan"), 33, 34], [1] * 5, 2.0)[0] == 0.0


# ── A ──────────────────────────────────────────────────────────────────

def test_absorption_worked_example():
    # sellers 80% of the tape, price flat → absorb = clip(0.3/0.3)=1 * price_hold 1
    a, p = ts.absorption_component(buy_vol_15=200, sell_vol_15=800, ret_15=0.0, atr_15_frac=0.01,
                                   cvd_15=100, cvd_60=-500, vol_15=1000, large_buy_15=300, large_sell_15=100, large_vol_15=400)
    assert p["sell_share_15"] == 0.8 and p["absorb"] == 1.0 and p["cvd_turn"] == 1.0
    assert p["cvd_slope"] == pytest.approx(0.1) and p["large_bias"] == pytest.approx(0.5)
    assert a == pytest.approx(min(1.0, 0.5 + 0.3 + 0.02 + 0.05))
    # price fell a full ATR → absorption credit gone
    a2, p2 = ts.absorption_component(200, 800, -0.01, 0.01, -100, -500, 1000, 0, 0, 0)
    assert p2["price_hold"] == 0.0 and a2 == 0.0
    # empty tape
    a3, p3 = ts.absorption_component(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert a3 == 0.0 and p3["sell_share_15"] == 0.5


# ── B ──────────────────────────────────────────────────────────────────

def test_book_component_and_liquidity_veto():
    b, p, ok = ts.book_component(imb_ema=0.5, spread_ema=0.0005, spread_baseline=0.0004, spread_mad=0.0001,
                                 bid_size=600, ask_size=600, median_bbo_size=400)
    assert ok and b == pytest.approx(0.75) and p["spread_z"] == pytest.approx(1.0)
    b, p, ok = ts.book_component(0.9, 0.0010, 0.0004, 0.0001, 600, 600, 400)      # spread_z 6 → veto
    assert not ok and b == 0.0
    b, p, ok = ts.book_component(0.9, 0.0004, 0.0004, 0.0001, 100, 100, 400)      # thin book → veto
    assert not ok and b == 0.0
    b, p, ok = ts.book_component(0.0, None, None, None, 100, 100, 0)              # no spread stats → vetoed
    assert not ok and math.isinf(p["spread_z"])


# ── composite ──────────────────────────────────────────────────────────

def test_weights_normalise_and_fallback():
    assert ts.normalise_weights({"Z": 1, "M": 1, "A": 1, "B": 1}) == {"Z": 0.25, "M": 0.25, "A": 0.25, "B": 0.25}
    assert ts.normalise_weights({"Z": -1, "M": 0, "A": 0, "B": 0}) == dict(ts.PRIOR_WEIGHTS)
    w = ts.normalise_weights({"Z": 3, "M": 1, "A": float("nan")})
    assert w["Z"] == 0.75 and w["M"] == 0.25 and w["A"] == 0.0 and w["B"] == 0.0


def test_composite_examples_and_scale():
    assert ts.composite_score({"Z": 1, "M": 1, "A": 1, "B": 1}) == 100.0
    assert ts.composite_score({"Z": 0, "M": 0, "A": 0, "B": 0}) == 0.0
    assert ts.composite_score({"Z": 0.8, "M": 0.4, "A": 0.7, "B": 0.3}) == pytest.approx(100 * (0.24 + 0.10 + 0.175 + 0.06))
    assert ts.size_scale(60, 60) == 0.5 and ts.size_scale(100, 60) == 1.0 and ts.size_scale(59, 60) == 0.0
    assert ts.size_scale(80, 60) == pytest.approx(0.75)
    assert ts.thesis_failed(score=30, theta=60, cvd_60=-1, sessions_held=1)
    assert not ts.thesis_failed(score=30, theta=60, cvd_60=-1, sessions_held=3)
    assert not ts.thesis_failed(score=45, theta=60, cvd_60=-1, sessions_held=1)
    assert not ts.thesis_failed(score=30, theta=60, cvd_60=+1, sessions_held=1)


def test_property_bounds_for_random_and_hostile_inputs():
    rng = np.random.default_rng(7)
    hostile = [float("nan"), float("inf"), -float("inf"), None, "x", 1e300, -1e300, 0]
    for _ in range(2000):
        pick = lambda: hostile[rng.integers(len(hostile))] if rng.random() < 0.2 else float(rng.normal(0, 1e3))
        st = ts.ZoneStats(q05=25.0, q25=40.0, q50=50.0, q75=60.0, n=252)
        _, z = ts.zone_component(pick(), st)
        m, _ = ts.momentum_component([pick() for _ in range(25)], [pick() for _ in range(25)], pick())
        a, _ = ts.absorption_component(*[pick() for _ in range(10)])
        b, _, _ = ts.book_component(*[pick() for _ in range(7)])
        for c in (z, m, a, b):
            assert 0.0 <= c <= 1.0 and math.isfinite(c)
        s = ts.composite_score({"Z": z, "M": m, "A": a, "B": b}, {"Z": pick(), "M": pick(), "A": pick(), "B": pick()})
        assert 0.0 <= s <= 100.0 and math.isfinite(s)
