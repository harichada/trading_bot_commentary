"""Tests for per-day dynamic symbol ranking (_rank_symbols_for_day)."""
import pandas as pd
import pytest

from backtest.scalp_reversal import _rank_symbols_for_day


def _make_day_bars(symbol: str, date: str, gap_pct: float, rvol: float) -> pd.DataFrame:
    """Minimal synthetic bars: one prev-close bar + today's bars with open gap + volume spike.

    The prev-close bar is placed at 19:55 UTC the day before ``date`` so that
    ``_rank_symbols_for_day`` can recover gap_pct from (open - prev_close).
    Volume on today's bars is ``avg_vol * rvol``; the prev-close bar uses
    ``avg_vol`` as the baseline, making mean_volume a mix and rvol > 1 for
    high-volume days.
    """
    prev_close = 100.0
    open_px = prev_close * (1 + gap_pct / 100)
    avg_vol = 1_000_000

    # Previous session's last bar — provides prev_close for gap computation
    prev_date = (pd.Timestamp(date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prev_idx = pd.to_datetime([f"{prev_date} 19:55"]).tz_localize("UTC")
    prev_df = pd.DataFrame(
        {
            "Open": prev_close,
            "High": prev_close + 0.1,
            "Low": prev_close - 0.1,
            "Close": prev_close,
            "Volume": avg_vol,
        },
        index=prev_idx,
    )

    # Today's bars — open at the gap price, volume at rvol × baseline
    idx = pd.date_range(f"{date} 14:30", periods=5, freq="1min", tz="UTC")
    today_df = pd.DataFrame(
        {
            "Open": open_px,
            "High": open_px + 0.1,
            "Low": open_px - 0.1,
            "Close": open_px,
            "Volume": int(avg_vol * rvol),
        },
        index=idx,
    )

    return pd.concat([prev_df, today_df])


def test_rank_sorts_by_composite_score():
    """MARA (gap=5%, rvol=4×) should rank above SNAP (3%, 2×) above AMD (1%, 1×)."""
    day_bars = {
        "MARA": _make_day_bars("MARA", "2026-10-01", gap_pct=5.0, rvol=4.0),
        "AMD": _make_day_bars("AMD", "2026-10-01", gap_pct=1.0, rvol=1.0),
        "SNAP": _make_day_bars("SNAP", "2026-10-01", gap_pct=3.0, rvol=2.0),
    }
    ranked = _rank_symbols_for_day(day_bars)
    assert ranked[0] == "MARA"
    assert ranked[-1] == "AMD"


def test_rank_top_n_truncates():
    """5 equal symbols, top_n=3 → ranked list has exactly 3 entries."""
    day_bars = {s: _make_day_bars(s, "2026-10-01", 2.0, 2.0) for s in ["A", "B", "C", "D", "E"]}
    ranked = _rank_symbols_for_day(day_bars, top_n=3)
    assert len(ranked) == 3
