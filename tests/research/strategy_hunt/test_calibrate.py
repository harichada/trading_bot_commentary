import pandas as pd
import pytest


def test_realized_slippage_sign_and_buckets():
    from research.strategy_hunt.sim.calibrate import realized_slippage
    fills = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "side": ["long", "short", "long", "short"],
                          "signal_price": [100.0, 100.0, 50.0, 50.0], "fill_price": [100.1, 99.9, 49.99, 50.02],
                          "adv_dollars": [2e9, 3e8, 5e7, 1e6]})
    s = realized_slippage(fills)
    assert list(s["slip_frac"].round(6)) == [0.001, 0.001, -0.0002, -0.0004]
    assert list(s["bucket"]) == ["mega", "large", "mid", "small"]


def test_summary_usable_threshold_and_adverse_only():
    from research.strategy_hunt.sim.calibrate import calibration_summary, realized_slippage
    fills = pd.DataFrame({"symbol": ["A"] * 25, "side": ["long"] * 25, "signal_price": [100.0] * 25,
                          "fill_price": [100.05] * 20 + [99.9] * 5, "adv_dollars": [5e8] * 25})
    summ = calibration_summary(realized_slippage(fills))
    assert summ["usable"] and summ["n"] == 25 and summ["median_frac"] == pytest.approx(0.0005)
    assert summ["by_bucket"]["large"]["n"] == 25
    assert not calibration_summary(realized_slippage(fills.head(5)))["usable"]


def test_write_load_and_override(tmp_path):
    from research.strategy_hunt.sim.calibrate import write_calibration, load_calibration, slippage_override_from
    assert slippage_override_from(tmp_path) is None
    write_calibration(tmp_path, {"n": 30, "median_frac": 0.0004, "p75_frac": 0.001, "by_bucket": {}, "usable": True})
    assert load_calibration(tmp_path)["n"] == 30 and slippage_override_from(tmp_path) == pytest.approx(0.0004)
    write_calibration(tmp_path, {"n": 3, "median_frac": 0.0004, "p75_frac": 0.001, "by_bucket": {}, "usable": False})
    assert slippage_override_from(tmp_path) is None


def test_load_fill_pairs_returns_empty_on_sql_error():
    from research.strategy_hunt.sim.calibrate import load_fill_pairs

    class _Engine:
        def connect(self): raise RuntimeError("no db")

    assert load_fill_pairs(_Engine()).empty
