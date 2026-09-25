import numpy as np
import pandas as pd
import pytest


def test_corwin_schultz_wider_range_means_wider_spread():
    from research.strategy_hunt.sim.costs import corwin_schultz_spread
    idx = pd.date_range("2025-01-02 09:30", periods=4, freq="5min")
    tight = corwin_schultz_spread(pd.Series([100.2] * 4, index=idx), pd.Series([100.0] * 4, index=idx))
    wide = corwin_schultz_spread(pd.Series([102.0] * 4, index=idx), pd.Series([100.0] * 4, index=idx))
    assert np.isnan(tight.iloc[0]) and (wide.iloc[1:] >= tight.iloc[1:]).all() and (tight.iloc[1:] >= 0).all()


def test_monthly_half_spread_floor_and_cap():
    from research.strategy_hunt.sim.costs import monthly_half_spread
    idx = pd.date_range("2025-01-02 09:30", periods=100, freq="5min")
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000}, index=idx)
    assert monthly_half_spread(flat).iloc[0] == pytest.approx(0.0002)
    assert monthly_half_spread(flat.assign(high=150.0, low=50.0)).iloc[0] == pytest.approx(0.005)


def test_fills_slippage_borrow_and_override():
    from research.strategy_hunt.sim.costs import CostModel, SELL_FEE_PCT
    hs = pd.Series([0.001], index=pd.PeriodIndex(["2025-01"], freq="M"))
    cm = CostModel(half_spread_by_month=hs, notional=6000.0)
    ts = pd.Timestamp("2025-01-15 10:00")
    assert cm.slippage(60_000, 100.0) == pytest.approx(0.0001) and cm.slippage(600, 100.0) == pytest.approx(0.001)
    assert cm.entry_fill("long", 100.0, ts, 60_000) == pytest.approx(100.0 * (1 + 0.001 + 0.0001))
    assert cm.exit_fill("long", 100.0, ts, 60_000) == pytest.approx(100.0 * (1 - 0.001 - 0.0001 - SELL_FEE_PCT))
    assert cm.entry_fill("short", 100.0, ts, 60_000) == pytest.approx(100.0 * (1 - 0.001 - 0.0001 - SELL_FEE_PCT))
    assert cm.exit_fill("short", 100.0, ts, 60_000) == pytest.approx(100.0 * (1 + 0.001 + 0.0001))
    assert cm.borrow_cost("short", 5) == pytest.approx(0.03 * 5 / 252) and cm.borrow_cost("long", 5) == 0.0
    assert cm.exit_fill("short", 100.0, ts, 60_000, hold_sessions=5) == pytest.approx(100.0 * (1 + 0.001 + 0.0001 + 0.03 * 5 / 252))
    assert cm.half_spread(pd.Timestamp("2030-06-01")) == pytest.approx(0.001)
    cal = CostModel(half_spread_by_month=hs, notional=6000.0, slippage_override=0.0005)
    assert cal.slippage(600, 100.0) == pytest.approx(0.0005)


def test_scaled_and_zero_costs():
    from research.strategy_hunt.sim.costs import CostModel, ZERO_COSTS
    hs = pd.Series([0.001], index=pd.PeriodIndex(["2025-01"], freq="M"))
    cm = CostModel(half_spread_by_month=hs, notional=6000.0).scaled(2.0)
    assert cm.half_spread(pd.Timestamp("2025-01-15")) == pytest.approx(0.002) and cm.notional == 6000.0
    assert cm.borrow_rate_annual == pytest.approx(0.06) and cm.slippage_floor == pytest.approx(0.0002)
    ts = pd.Timestamp("2025-01-15 10:00")
    assert ZERO_COSTS.entry_fill("long", 50.0, ts, 10) == 50.0 and ZERO_COSTS.exit_fill("short", 50.0, ts, 10, 9) == 50.0


def test_from_bars_builds_monthly_series():
    from research.strategy_hunt.sim.costs import CostModel
    idx = pd.date_range("2025-01-02 09:30", periods=200, freq="5min")
    df = pd.DataFrame({"open": 100.0, "high": 100.3, "low": 99.7, "close": 100.0, "volume": 5000}, index=idx)
    cm = CostModel.from_bars(df)
    assert isinstance(cm.half_spread_by_month.index, pd.PeriodIndex) and 0.0002 <= cm.half_spread(idx[0]) <= 0.005
