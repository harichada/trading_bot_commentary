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


def test_entry_style_is_an_appended_trade_column_and_defaults_to_market():
    """Batch-2 Task B: appended, never interleaved, so an older trades Parquet's column order
    still lines up -- and defaulted, so every existing positional `SimTrade(...)` keeps working.
    Task F appended `time_stop` after it, which is why this asserts a position, not last-ness."""
    from research.strategy_hunt.models import SimTrade, TRADE_COLUMNS
    assert TRADE_COLUMNS.index("entry_style") == 21 and TRADE_COLUMNS.count("entry_style") == 1
    t = SimTrade("NVDA", "orb", "long", pd.Timestamp("2025-01-02 09:35"), pd.Timestamp("2025-01-02 10:00"),
                 100.0, 98.0, 104.0, 102.0, 100.02, 101.98, 0.98, 1.96, 0.04, "target", 5, 1, "orb_long",
                 "risk_on", 3, "{}")
    assert t.entry_style == "market" and tuple(t.to_dict()) == TRADE_COLUMNS


def test_time_stop_is_an_optional_trailing_trade_column():
    """Batch-2 Task F item 4: the entry's horizon (sessions or bars, per `hold_kind`) is now on the
    record, so the selection replay no longer has to guess it from `params_json`. Appended after
    `entry_style` and defaulted, so an older trades Parquet still lines up positionally."""
    from research.strategy_hunt.models import OPTIONAL_TRADE_COLUMNS, SimTrade, TRADE_COLUMNS
    assert TRADE_COLUMNS[-1] == "time_stop" and TRADE_COLUMNS.count("time_stop") == 1
    assert set(OPTIONAL_TRADE_COLUMNS) == {"entry_style", "time_stop"}
    t = SimTrade("NVDA", "orb", "long", pd.Timestamp("2025-01-02 09:35"), pd.Timestamp("2025-01-02 10:00"),
                 100.0, 98.0, 104.0, 102.0, 100.02, 101.98, 0.98, 1.96, 0.04, "target", 5, 1, "orb_long",
                 "risk_on", 3, "{}")
    assert t.time_stop == 0 and tuple(t.to_dict()) == TRADE_COLUMNS


def test_trades_from_frame_tolerates_a_file_without_time_stop():
    import pandas as pd

    from research.strategy_hunt.models import TRADE_COLUMNS
    from research.strategy_hunt.score.regime_rows import trades_from_frame
    row = {"symbol": "NVDA", "family": "orb", "side": "long", "entry_ts": pd.Timestamp("2025-01-02 09:35"),
           "exit_ts": pd.Timestamp("2025-01-02 10:00"), "entry_px": 100.0, "stop_px": 98.0,
           "target_px": 104.0, "exit_px": 102.0, "entry_fill": 100.0, "exit_fill": 102.0,
           "r_multiple": 1.0, "pnl_pct": 2.0, "cost_pct": 0.0, "exit_reason": "target", "hold_bars": 5,
           "hold_sessions": 1, "tag": "t", "regime": "risk_on", "window_id": 3, "params_json": "{}"}
    df = pd.DataFrame([row])[[c for c in TRADE_COLUMNS if c in row]]
    trades = trades_from_frame(df)
    assert len(trades) == 1 and trades[0].time_stop == 0 and trades[0].entry_style == "market"
