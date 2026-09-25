import pandas as pd


def test_build_windows_shapes_and_embargo():
    from research.strategy_hunt.walkforward.windows import build_windows
    w = build_windows(pd.Timestamp("2021-01-04 09:30"), pd.Timestamp("2023-06-15 15:55"))
    assert w[0].window_id == 0 and w[0].oos_start == pd.Timestamp("2022-01-01")
    assert w[0].is_start == pd.Timestamp("2021-01-01") and w[0].is_end == pd.Timestamp("2021-12-25")
    assert w[0].oos_end == pd.Timestamp("2022-04-01")
    assert w[1].oos_start == pd.Timestamp("2022-04-01")
    assert w[-1].oos_end == pd.Timestamp("2023-06-15 15:55") and w[-1].oos_start == pd.Timestamp("2023-04-01")
    assert all(x.is_end < x.oos_start for x in w)
    assert [x.window_id for x in w] == list(range(len(w)))


def test_build_windows_respects_short_history():
    from research.strategy_hunt.walkforward.windows import build_windows
    w = build_windows(pd.Timestamp("2024-03-15 09:30"), pd.Timestamp("2025-09-24 15:55"))
    assert w[0].oos_start == pd.Timestamp("2025-04-01")    # first month >= first_bar + 12 months
    assert len(w) == 2 and w[-1].oos_start == pd.Timestamp("2025-07-01")
    assert build_windows(pd.Timestamp("2025-01-02"), pd.Timestamp("2025-06-01")) == []


def test_build_windows_grace_days_boundary():
    from research.strategy_hunt.walkforward.windows import build_windows
    last_bar = pd.Timestamp("2023-06-15 15:55")
    w_in_grace = build_windows(pd.Timestamp("2021-01-08"), last_bar)
    assert w_in_grace[0].oos_start == pd.Timestamp("2022-01-01")
    w_past_grace = build_windows(pd.Timestamp("2021-01-09"), last_bar)
    assert w_past_grace[0].oos_start == pd.Timestamp("2022-02-01")
