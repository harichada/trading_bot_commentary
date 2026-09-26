"""Batch-2 Task E: split adjustment must trust the cache and require persistence before
applying a detected split (see .superpowers/sdd/2026-09-26-strategy-search-batch2/task-E-brief.md).

Covers the four scenarios in the brief:
  1. LCID-like bad opens, no real split, empty cache file -> no adjustment (Ruling 1).
  2. LRCX-like: cache has the real split; a far-away false detection candidate in the daily
     frame is ignored entirely because a cache file exists (Ruling 1).
  3. No cache file + a genuine persisting 10:1 -> detected once (Ruling 2).
  4. A reverting gap (no real split) -> not detected (Ruling 2).
Plus the explicit open<=0/NaN guard and the WARNING log for an applied detected split (Rulings
2 and 3).
"""
import logging

import numpy as np
import pandas as pd


def _flat_daily(n=40, price=400.0, start="2025-01-02"):
    idx = pd.bdate_range(start, periods=n, name="date")
    close = np.full(n, price)
    return pd.DataFrame(
        {"open": close.copy(), "high": close + 1.0, "low": close - 1.0, "close": close.copy(),
         "volume": np.full(n, 1_000_000.0)},
        index=idx,
    )


def _step(d: pd.DataFrame, from_pos: int, price: float) -> pd.DataFrame:
    """A persisting price step from `from_pos` to the end of the frame."""
    out = d.copy()
    out.loc[out.index[from_pos]:, ["open", "close"]] = price
    out.loc[out.index[from_pos]:, "high"] = price + 1.0
    out.loc[out.index[from_pos]:, "low"] = price - 1.0
    return out


def test_empty_cache_file_suppresses_a_genuinely_persisting_candidate(tmp_path):
    """LCID-like: an empty cache file means "checked, no splits" -- Ruling 1 says the detector
    must not run at all, even though the daily frame here has a candidate that would otherwise
    persist and be detected."""
    from research.strategy_hunt.data.splits import all_splits, detect_splits, load_splits, splits_path, update_splits

    d = _step(_flat_daily(), 20, 40.0)                     # ratio 10 -> would-be m=0.1, persists
    assert detect_splits(d) == [(d.index[20], 0.1)]        # sanity: the detector alone would find it
    n = update_splits(tmp_path, "LCIDX", lambda s: pd.Series(dtype=float))
    assert n == 0 and splits_path(tmp_path, "LCIDX").exists()
    assert load_splits(tmp_path, "LCIDX") == []
    assert all_splits(tmp_path, "LCIDX", d) == []


def test_cache_present_ignores_a_far_away_false_candidate(tmp_path):
    """LRCX-like: the cache holds the real split; a persisting false candidate far from the
    cached date must not be merged in -- Ruling 1 trusts the cache exclusively once a cache
    file exists, regardless of proximity to the cached date(s)."""
    from research.strategy_hunt.data.splits import all_splits, load_splits, update_splits

    d = _step(_flat_daily(), 20, 100.0)                    # false ratio-4 candidate, far from date[5]
    update_splits(tmp_path, "LRCXX", lambda s: pd.Series([10.0], index=pd.DatetimeIndex([d.index[5]])))
    cached = load_splits(tmp_path, "LRCXX")
    assert cached == [(d.index[5], 0.1)]
    assert all_splits(tmp_path, "LRCXX", d) == cached       # the false candidate at index 20 is dropped


def test_no_cache_file_and_genuine_persisting_split_detected_once(tmp_path):
    from research.strategy_hunt.data.splits import all_splits, detect_splits, splits_path

    d = _step(_flat_daily(), 20, 40.0)                     # genuine, persists to the end
    assert not splits_path(tmp_path, "NOCACHEX").exists()
    assert detect_splits(d) == [(d.index[20], 0.1)]
    assert all_splits(tmp_path, "NOCACHEX", d) == [(d.index[20], 0.1)]


def test_reverting_gap_is_not_detected():
    from research.strategy_hunt.data.splits import detect_splits

    d = _flat_daily()
    d.loc[d.index[20], ["open", "close"]] = 40.0            # one bad day matching the ratio-10 pattern
    d.loc[d.index[20], ["high", "low"]] = [41.0, 39.0]
    # every session before and after (including 21 onward) stays at the original level -> reverts
    assert detect_splits(d) == []


def test_invalid_open_is_rejected_even_if_it_would_otherwise_persist():
    from research.strategy_hunt.data.splits import detect_splits

    zero_open = _step(_flat_daily(), 20, 40.0)
    zero_open.loc[zero_open.index[20], "open"] = 0.0
    assert detect_splits(zero_open) == []

    nan_open = _step(_flat_daily(), 20, 40.0)
    nan_open.loc[nan_open.index[20], "open"] = np.nan
    assert detect_splits(nan_open) == []


def test_applied_detected_split_logs_a_warning(tmp_path, caplog):
    from research.strategy_hunt.data.splits import all_splits

    d = _step(_flat_daily(), 20, 40.0)
    with caplog.at_level(logging.WARNING, logger="strategy_hunt.splits"):
        out = all_splits(tmp_path, "WARNSYM", d)
    assert out == [(d.index[20], 0.1)]
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("WARNSYM" in w and "0.1" in w for w in warnings)
