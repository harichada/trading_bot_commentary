"""Tests for core/strategy_fit_bars.py.

v-strategy-fit-review2-2026-09-25 (Important 3): the engine's
`_fit_bars_source` could only fetch ~7 sessions of 5-minute bars (Schwab
caps minute history at 10 days), while `rvol` needs 20 prior sessions and
`or*_range_avg20` needs 10 — so `orb` and `news_catalyst` could never fire
live. The pure merge helpers are tested here directly; `_fit_bars_source`
just calls them.

Offline: Parquet files are written under tmp_path, no network, no Postgres.
"""
import pandas as pd
import pytest


def _bars5(start, sessions, close=100.0, volume=1000):
    frames = []
    for d in pd.bdate_range(start, periods=sessions):
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", name="ts")
        frames.append(pd.DataFrame({"open": close, "high": close + 0.5, "low": close - 0.5,
                                    "close": close, "volume": volume}, index=idx))
    return pd.concat(frames)


def _daily(start, sessions, close=100.0):
    idx = pd.bdate_range(start, periods=sessions)
    d = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close,
                      "volume": 1_000_000}, index=idx)
    d.index.name = "date"
    return d


# ──────────────────────────────────────────────────────────────────────
# merge_history
# ──────────────────────────────────────────────────────────────────────

def test_merge_history_concatenates_deduplicates_and_sorts():
    from core.strategy_fit_bars import merge_history

    hist = _bars5("2025-07-01", 20, close=90.0)              # 20 sessions, 07-01..07-28
    recent = _bars5("2025-09-01", 7, close=100.0)            # provider's ~7-session window
    out = merge_history(hist, recent)

    assert out.index.is_monotonic_increasing
    assert not out.index.has_duplicates
    assert out.index.name == "ts"
    assert len(out.index.normalize().unique()) == 27          # 20 + 7, disjoint
    assert out.index[0] == hist.index[0] and out.index[-1] == recent.index[-1]
    assert list(out.columns) == list(recent.columns)


def test_merge_history_recent_wins_overlapping_timestamps():
    """The broker's view of a bar is the live one; the nightly export may
    predate an adjustment — so keep="last" with recent concatenated last."""
    from core.strategy_fit_bars import merge_history

    hist = _bars5("2025-09-01", 10, close=90.0)
    recent = _bars5("2025-09-08", 5, close=100.0)             # overlaps 09-08..09-12
    out = merge_history(hist, recent)

    overlap_ts = recent.index[0]
    assert out.loc[overlap_ts, "close"] == 100.0              # recent, not 90.0
    assert out.loc[hist.index[0], "close"] == 90.0            # history kept where unique
    assert len(out) == len(hist) + len(recent) - len(recent.index.intersection(hist.index))


def test_merge_history_handles_missing_and_empty_sides():
    from core.strategy_fit_bars import merge_history

    recent = _bars5("2025-09-01", 3)
    assert merge_history(None, recent).equals(recent)
    assert merge_history(pd.DataFrame(), recent).equals(recent)

    hist = _bars5("2025-08-01", 3)
    assert len(merge_history(hist, None)) == len(hist)
    assert len(merge_history(hist, pd.DataFrame())) == len(hist)

    assert merge_history(None, None).empty
    assert merge_history(pd.DataFrame(), pd.DataFrame()).empty


def test_merge_history_strips_timezone_to_naive_et():
    from core.strategy_fit_bars import merge_history

    hist = _bars5("2025-08-01", 2)
    recent = _bars5("2025-09-01", 2)
    recent = recent.tz_localize("America/New_York")
    out = merge_history(hist, recent)
    assert out.index.tz is None
    assert pd.Timestamp("2025-09-01 09:30") in out.index


def test_merge_history_preserves_attrs_from_recent():
    from core.strategy_fit_bars import merge_history

    hist = _bars5("2025-08-01", 2)
    recent = _bars5("2025-09-01", 2)
    recent.attrs["symbol"] = "NVDA"
    assert merge_history(hist, recent).attrs.get("symbol") == "NVDA"


# ──────────────────────────────────────────────────────────────────────
# merge_daily
# ──────────────────────────────────────────────────────────────────────

def test_merge_daily_normalizes_index_and_dedupes():
    from core.strategy_fit_bars import merge_daily

    hist = _daily("2025-01-02", 200, close=80.0)
    recent = _daily("2025-09-01", 20, close=100.0)
    out = merge_daily(hist, recent)

    assert out.index.name == "date"
    assert out.index.is_monotonic_increasing and not out.index.has_duplicates
    assert (out.index == out.index.normalize()).all()
    assert out.loc[pd.Timestamp("2025-09-01"), "close"] == 100.0
    assert out.loc[pd.Timestamp("2025-01-02"), "close"] == 80.0


def test_merge_daily_handles_missing_sides():
    from core.strategy_fit_bars import merge_daily

    recent = _daily("2025-09-01", 5)
    assert len(merge_daily(None, recent)) == 5
    assert merge_daily(None, None).empty


# ──────────────────────────────────────────────────────────────────────
# load_history
# ──────────────────────────────────────────────────────────────────────

def test_load_history_reads_research_parquet_tree(tmp_path):
    from core.strategy_fit_bars import load_history

    (tmp_path / "bars5").mkdir()
    (tmp_path / "daily").mkdir()
    b = _bars5("2025-08-01", 25)
    d = _daily("2025-01-02", 200)
    # The research exporter writes the timestamp as a column, not an index.
    b.reset_index().to_parquet(tmp_path / "bars5" / "NVDA.parquet", index=False)
    d.reset_index().to_parquet(tmp_path / "daily" / "NVDA.parquet", index=False)

    bars5, daily = load_history(str(tmp_path), "NVDA")
    assert bars5 is not None and daily is not None
    assert bars5.index.name == "ts" and list(bars5.columns) == ["open", "high", "low", "close", "volume"]
    assert len(bars5.index.normalize().unique()) == 25
    assert daily.index.name == "date" and len(daily) == 200


def test_load_history_uppercases_symbol(tmp_path):
    from core.strategy_fit_bars import load_history

    (tmp_path / "bars5").mkdir()
    _bars5("2025-08-01", 2).reset_index().to_parquet(tmp_path / "bars5" / "NVDA.parquet", index=False)
    bars5, _ = load_history(str(tmp_path), "nvda")
    assert bars5 is not None and not bars5.empty


def test_load_history_missing_files_yield_none(tmp_path):
    from core.strategy_fit_bars import load_history

    assert load_history(str(tmp_path), "NVDA") == (None, None)
    assert load_history(str(tmp_path / "no_such_root"), "NVDA") == (None, None)


def test_load_history_never_raises_on_garbage(tmp_path):
    """A corrupt or non-Parquet file must degrade to "no history", not take
    the trading path down."""
    from core.strategy_fit_bars import load_history

    (tmp_path / "bars5").mkdir()
    (tmp_path / "daily").mkdir()
    (tmp_path / "bars5" / "NVDA.parquet").write_text("this is not parquet")
    (tmp_path / "daily" / "NVDA.parquet").write_bytes(b"\x00\x01\x02")
    assert load_history(str(tmp_path), "NVDA") == (None, None)


def test_load_history_parquet_without_bar_columns_yields_none(tmp_path):
    from core.strategy_fit_bars import load_history

    (tmp_path / "bars5").mkdir()
    pd.DataFrame({"ts": pd.date_range("2025-08-01", periods=3), "junk": [1, 2, 3]}).to_parquet(
        tmp_path / "bars5" / "NVDA.parquet", index=False)
    bars5, _ = load_history(str(tmp_path), "NVDA")
    assert bars5 is None


# ──────────────────────────────────────────────────────────────────────
# The point of the exercise: rvol / or*_range_avg20 resolve after the merge
# ──────────────────────────────────────────────────────────────────────

def test_merged_window_makes_rvol_and_or_range_resolve():
    """Important 3's actual symptom: with only the provider's ~7-session
    window, `rvol` (20 prior sessions) and `or30_range_avg20` (10) are all
    NaN, so `orb` and `news_catalyst` can never fire. After the merge they
    have real values."""
    from core.strategy_fit_bars import merge_history
    from strategies.families.indicators import indicator_frame

    recent = _bars5("2025-09-01", 7)
    hist = _bars5("2025-07-01", 40)

    recent_only = indicator_frame(recent, pd.DataFrame())
    assert recent_only["rvol"].isna().all()
    assert recent_only["or30_range_avg20"].isna().all()

    merged = indicator_frame(merge_history(hist, recent), pd.DataFrame())
    last_session = merged.loc[merged.index.normalize() == pd.Timestamp("2025-09-09")]
    assert last_session["rvol"].notna().any()
    assert last_session["or30_range_avg20"].notna().any()
