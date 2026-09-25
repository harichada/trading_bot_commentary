import pandas as pd
import pytest


def test_simtrade_columns_and_dict():
    from research.strategy_hunt.models import SimTrade, TRADE_COLUMNS
    t = SimTrade("NVDA", "orb", "long", pd.Timestamp("2025-01-02 09:35"), pd.Timestamp("2025-01-02 10:00"),
                 100.0, 98.0, 104.0, 102.0, 100.02, 101.98, 0.98, 1.96, 0.04, "target", 5, 1, "orb_long", "risk_on", 3, "{}")
    d = t.to_dict()
    assert tuple(d) == TRADE_COLUMNS and d["stop_px"] == 98.0 and d["hold_sessions"] == 1


def test_fitmetrics_and_verdicts():
    from research.strategy_hunt.models import FitMetrics, VERDICTS, Window
    m = FitMetrics(n=10, sessions=5, exp_r=0.1, exp_r_lb=0.02, t_stat=2.1, pf=1.4, wr=0.5, max_dd_r=3.0, stability=0.7, decay=0.9)
    assert m.to_dict()["t_stat"] == 2.1 and VERDICTS == ("PASS", "WATCH", "FAIL")
    w = Window(0, pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-25"), pd.Timestamp("2022-01-01"), pd.Timestamp("2022-04-01"))
    assert w.oos_end > w.oos_start > w.is_end > w.is_start
