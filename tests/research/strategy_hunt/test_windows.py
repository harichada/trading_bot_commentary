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


# ──────────────────────
# strategy-search-batch1 task 2 — holdout_months truncates the OOS range only
# ──────────────────────

def test_build_windows_holdout_months_zero_is_unchanged():
    from research.strategy_hunt.walkforward.windows import build_windows
    first_bar, last_bar = pd.Timestamp("2021-01-04 09:30"), pd.Timestamp("2023-06-15 15:55")
    assert build_windows(first_bar, last_bar, holdout_months=0) == build_windows(first_bar, last_bar)


def test_build_windows_holdout_months_truncates_oos_and_drops_windows_past_cutoff():
    from research.strategy_hunt.walkforward.windows import build_windows
    first_bar, last_bar = pd.Timestamp("2021-01-04 09:30"), pd.Timestamp("2023-06-15 15:55")
    unheld = build_windows(first_bar, last_bar)
    held = build_windows(first_bar, last_bar, holdout_months=3)
    last_bar_eff = last_bar - pd.DateOffset(months=3)
    assert len(held) == len(unheld) - 1                    # the last (2023-04-01) window is dropped
    assert held[-1].oos_start == pd.Timestamp("2023-01-01") and held[-1].oos_end == last_bar_eff
    assert all(x.oos_start < last_bar_eff for x in held)
    # in-sample dates are untouched by the holdout -- only the OOS range is truncated
    assert held[-1].is_start == unheld[-2].is_start and held[-1].is_end == unheld[-2].is_end


def test_build_windows_holdout_months_can_drop_every_window():
    from research.strategy_hunt.walkforward.windows import build_windows
    w = build_windows(pd.Timestamp("2024-03-15 09:30"), pd.Timestamp("2025-09-24 15:55"), holdout_months=18)
    assert w == []
