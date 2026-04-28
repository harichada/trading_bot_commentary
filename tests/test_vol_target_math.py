"""Pure-math unit tests for risk/vol_target.py.

These tests exercise the vol-target sizing math in isolation — no
data-provider, no walk-forward harness, no XGBoost. The module under
test (``risk.vol_target``) implements López de Prado, AFML §10.1
(constant-portfolio-vol position sizing).

Methodology references
----------------------
* AFML §10.1 — vol-targeted bet sizing. Per-position size is
  ``w_i = (target_vol / N_positions) / realized_vol_i``.
* AFML §10.4 — caps prevent runaway sizing on stale/collapsing vol.

Test coverage
-------------
14 cases per the P9 self-approval plan:
  1. ``daily_log_returns`` resamples 5-min bars correctly
  2. EWMA realized vol on constant returns
  3. EWMA realized vol on zero returns raises (degenerate)
  4. Floor at the 10th percentile of the lookback window
  5. ``apply_floor`` below / above
  7. ``size_event`` vol-parity formula
  8. cap binds when pre-cap would exceed 5x parity
  9. cap not binding under default multipliers
 10. multipliers compose (regime * meta)
 11. n_positions == 0 treated as 1 (no DivisionByZero)
 12. lookback too short raises
 13. negative target raises
 14. default multipliers are exactly 1.0
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk.vol_target import (
    DEFAULT_CAP_MULT,
    DEFAULT_FLOOR_PERCENTILE,
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_TARGET_DAILY_VOL,
    SizingDecision,
    apply_floor,
    daily_log_returns,
    realized_daily_vol_ewma,
    size_event,
    vol_floor,
)


EPS = 1e-9


def _synthetic_5min_bars(
    *,
    start: str,
    n_sessions: int,
    bars_per_session: int = 78,
    daily_close_path: list[float] | None = None,
) -> pd.DataFrame:
    """Build a synthetic 5-min OHLCV frame spanning ``n_sessions`` calendar
    days. Each session has ``bars_per_session`` 5-min bars (78 ≈ 6.5h US
    equity session). If ``daily_close_path`` is given, each session's
    last-close is set to that value so the daily resample produces a
    deterministic series.
    """
    rows: list[dict] = []
    base = pd.Timestamp(start)
    for d in range(n_sessions):
        day_open = base + pd.Timedelta(days=d, hours=14, minutes=30)  # 14:30 UTC ≈ 09:30 ET
        for b in range(bars_per_session):
            ts = day_open + pd.Timedelta(minutes=5 * b)
            close = (
                daily_close_path[d]
                if daily_close_path is not None and b == bars_per_session - 1
                else 100.0 + 0.001 * (d * bars_per_session + b)
            )
            rows.append({
                "timestamp": ts,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1000,
            })
    df = pd.DataFrame(rows).set_index("timestamp")
    return df


# ---------------------------------------------------------------------------
# 1. daily_log_returns
# ---------------------------------------------------------------------------

class TestDailyLogReturns:

    def test_resamples_three_sessions_to_two_returns(self) -> None:
        """Three calendar days of 5-min bars → 3 daily closes → 2 log-returns."""
        bars = _synthetic_5min_bars(
            start="2025-01-02 14:30:00", n_sessions=3,
            daily_close_path=[100.0, 101.0, 102.01],
        )
        rets = daily_log_returns(bars)
        assert len(rets) == 2, f"expected 2 daily returns, got {len(rets)}"
        # Day-1 return = ln(101/100), day-2 = ln(102.01/101)
        assert abs(float(rets.iloc[0]) - np.log(101.0 / 100.0)) < EPS
        assert abs(float(rets.iloc[1]) - np.log(102.01 / 101.0)) < EPS

    def test_index_is_daily(self) -> None:
        """Returned series must have one entry per calendar day after the first."""
        bars = _synthetic_5min_bars(
            start="2025-01-02 14:30:00", n_sessions=4,
            daily_close_path=[100.0, 100.0, 100.0, 100.0],
        )
        rets = daily_log_returns(bars)
        assert len(rets) == 3
        # Same close every day → all log-returns are zero (within fp).
        assert all(abs(float(r)) < EPS for r in rets)


# ---------------------------------------------------------------------------
# 2 + 3. realized_daily_vol_ewma
# ---------------------------------------------------------------------------

class TestRealizedDailyVolEWMA:

    def test_constant_return_yields_constant_vol(self) -> None:
        """Series of constant log-returns r → EWMA(r^2) == r^2 → vol == |r|.

        Mathematical fact: for a constant series, EWMA equals that constant
        regardless of span. Tested with span=20 over 25 observations so we
        are past the min_periods=span warmup.
        """
        r = 0.01
        rets = pd.Series([r] * 25, index=pd.date_range("2025-01-02", periods=25, freq="D"))
        vol = realized_daily_vol_ewma(rets, lookback_days=20)
        # Last value must equal r within fp; series may carry NaNs in warmup.
        last = float(vol.dropna().iloc[-1])
        assert abs(last - r) < 1e-12, f"realized vol {last} != r={r}"

    def test_zero_returns_realized_vol_is_zero(self) -> None:
        """All-zero returns produce realized_vol == 0. The downstream
        ``size_event`` must catch that as degenerate; this test only
        verifies the math primitive itself doesn't blow up."""
        rets = pd.Series(
            [0.0] * 25,
            index=pd.date_range("2025-01-02", periods=25, freq="D"),
        )
        vol = realized_daily_vol_ewma(rets, lookback_days=20)
        last = float(vol.dropna().iloc[-1])
        assert last == 0.0


# ---------------------------------------------------------------------------
# 4 + 5. vol_floor + apply_floor
# ---------------------------------------------------------------------------

class TestVolFloor:

    def test_floor_at_10th_percentile_linear(self) -> None:
        """Floor = quantile(rv_series, 0.10) (numpy linear interpolation)."""
        rv = pd.Series(np.linspace(0.001, 0.020, 20))
        floor = vol_floor(rv, floor_percentile=0.10)
        expected = float(np.quantile(rv.values, 0.10))
        assert abs(float(floor) - expected) < EPS

    def test_apply_floor_below_returns_floor_and_flag(self) -> None:
        out, hit = apply_floor(0.005, 0.010)
        assert out == 0.010
        assert hit is True

    def test_apply_floor_above_returns_value_and_no_flag(self) -> None:
        out, hit = apply_floor(0.020, 0.010)
        assert out == 0.020
        assert hit is False


# ---------------------------------------------------------------------------
# 7-14. size_event
# ---------------------------------------------------------------------------

def _bars_with_known_daily_vol(
    *, daily_log_ret: float, n_sessions: int = 30,
) -> pd.DataFrame:
    """Build 5-min bars whose daily-resampled log-returns equal the given
    constant. Achieved by setting each session's last-close to
    ``prev_close * exp(daily_log_ret)``.
    """
    closes = [100.0]
    for _ in range(n_sessions - 1):
        closes.append(closes[-1] * np.exp(daily_log_ret))
    return _synthetic_5min_bars(
        start="2025-01-02 14:30:00",
        n_sessions=n_sessions, daily_close_path=closes,
    )


class TestSizeEventCore:

    def test_default_multipliers_are_one(self) -> None:
        """When called with no multiplier kwargs, both defaults must be 1.0
        and the pre-cap size must equal the vol-parity size."""
        bars = _bars_with_known_daily_vol(daily_log_ret=0.01, n_sessions=30)
        decision = size_event(
            symbol="TEST", event_ts=bars.index[-1] + pd.Timedelta(minutes=5),
            bars_5min=bars, n_active_positions=3,
            target_daily_vol=0.0075,
            lookback_days=20, floor_percentile=0.10, cap_mult=5.0,
        )
        assert decision.regime_size_multiplier == 1.0
        assert decision.meta_p_win_multiplier == 1.0
        # No multipliers, cap shouldn't bind (vol_parity itself is the
        # pre-cap value, which is < 5*vol_parity).
        assert abs(decision.applied_size - decision.vol_parity_size) < EPS

    def test_vol_parity_formula(self) -> None:
        """rv=0.01, target=0.0075, n=3 → vol_parity = (0.0075/3) / 0.01 = 0.25."""
        bars = _bars_with_known_daily_vol(daily_log_ret=0.01, n_sessions=30)
        decision = size_event(
            symbol="TEST", event_ts=bars.index[-1] + pd.Timedelta(minutes=5),
            bars_5min=bars, n_active_positions=3,
            target_daily_vol=0.0075,
            lookback_days=20, floor_percentile=0.10, cap_mult=5.0,
        )
        # rv_floor close to 0.01 — not exactly 0.01 because the EWMA has
        # warmup transient on the first few bars, and the 10th-pct floor
        # is applied. Both effects are small here. Tolerance generous:
        # the formula is the assertion, not the exact rv reading.
        expected_parity = (0.0075 / 3) / decision.realized_vol_floor
        assert abs(decision.vol_parity_size - expected_parity) < EPS
        # Sanity: the magnitude is in the right ballpark (0.20-0.30 for
        # rv≈0.01).
        assert 0.20 <= decision.vol_parity_size <= 0.30

    def test_cap_binds_when_multipliers_inflate(self) -> None:
        """regime_mult=10, meta_mult=1 → pre_cap = 10*parity → cap (5x) binds."""
        bars = _bars_with_known_daily_vol(daily_log_ret=0.01, n_sessions=30)
        decision = size_event(
            symbol="TEST", event_ts=bars.index[-1] + pd.Timedelta(minutes=5),
            bars_5min=bars, n_active_positions=3,
            target_daily_vol=0.0075,
            regime_size_multiplier=10.0, meta_p_win_multiplier=1.0,
            lookback_days=20, floor_percentile=0.10, cap_mult=5.0,
        )
        assert decision.cap_hit is True
        assert abs(decision.applied_size - 5.0 * decision.vol_parity_size) < EPS
        assert decision.applied_size < decision.pre_cap_size

    def test_cap_not_binding_under_default_multipliers(self) -> None:
        bars = _bars_with_known_daily_vol(daily_log_ret=0.01, n_sessions=30)
        decision = size_event(
            symbol="TEST", event_ts=bars.index[-1] + pd.Timedelta(minutes=5),
            bars_5min=bars, n_active_positions=3,
            target_daily_vol=0.0075,
            lookback_days=20, floor_percentile=0.10, cap_mult=5.0,
        )
        assert decision.cap_hit is False
        assert abs(decision.applied_size - decision.pre_cap_size) < EPS

    def test_multipliers_compose(self) -> None:
        """regime_mult=1.5, meta_mult=2.0 → pre_cap = 3.0 * parity (cap=5 not binding)."""
        bars = _bars_with_known_daily_vol(daily_log_ret=0.01, n_sessions=30)
        decision = size_event(
            symbol="TEST", event_ts=bars.index[-1] + pd.Timedelta(minutes=5),
            bars_5min=bars, n_active_positions=3,
            target_daily_vol=0.0075,
            regime_size_multiplier=1.5, meta_p_win_multiplier=2.0,
            lookback_days=20, floor_percentile=0.10, cap_mult=5.0,
        )
        assert decision.cap_hit is False
        assert abs(decision.applied_size - 3.0 * decision.vol_parity_size) < EPS

    def test_n_positions_zero_treated_as_one(self) -> None:
        """Empty book → first event being sized → n=1 in the formula. No
        DivisionByZero. The vol_parity = (target/1)/rv."""
        bars = _bars_with_known_daily_vol(daily_log_ret=0.01, n_sessions=30)
        decision = size_event(
            symbol="TEST", event_ts=bars.index[-1] + pd.Timedelta(minutes=5),
            bars_5min=bars, n_active_positions=0,
            target_daily_vol=0.0075,
            lookback_days=20, floor_percentile=0.10, cap_mult=5.0,
        )
        # n=1 effective → parity = 0.0075 / rv_floor
        expected = 0.0075 / decision.realized_vol_floor
        assert abs(decision.vol_parity_size - expected) < EPS

    def test_lookback_too_short_raises(self) -> None:
        """Need at least lookback_days daily returns; 5-day window with
        lookback=20 must raise."""
        bars = _bars_with_known_daily_vol(daily_log_ret=0.01, n_sessions=5)
        with pytest.raises(ValueError, match="lookback"):
            size_event(
                symbol="TEST",
                event_ts=bars.index[-1] + pd.Timedelta(minutes=5),
                bars_5min=bars, n_active_positions=1,
                target_daily_vol=0.0075,
                lookback_days=20, floor_percentile=0.10, cap_mult=5.0,
            )

    def test_negative_target_raises(self) -> None:
        bars = _bars_with_known_daily_vol(daily_log_ret=0.01, n_sessions=30)
        with pytest.raises(ValueError, match="target"):
            size_event(
                symbol="TEST",
                event_ts=bars.index[-1] + pd.Timedelta(minutes=5),
                bars_5min=bars, n_active_positions=1,
                target_daily_vol=-0.01,
                lookback_days=20, floor_percentile=0.10, cap_mult=5.0,
            )

    def test_zero_vol_window_raises(self) -> None:
        """Lookback window with all-zero returns → degenerate (rv == 0).
        ``size_event`` must raise rather than silently return inf."""
        bars = _bars_with_known_daily_vol(daily_log_ret=0.0, n_sessions=30)
        with pytest.raises(ValueError, match="degenerate"):
            size_event(
                symbol="TEST",
                event_ts=bars.index[-1] + pd.Timedelta(minutes=5),
                bars_5min=bars, n_active_positions=1,
                target_daily_vol=0.0075,
                lookback_days=20, floor_percentile=0.10, cap_mult=5.0,
            )


class TestModuleConstants:

    def test_default_target_is_75bps_daily(self) -> None:
        assert DEFAULT_TARGET_DAILY_VOL == 0.0075

    def test_default_lookback_is_20_days(self) -> None:
        assert DEFAULT_LOOKBACK_DAYS == 20

    def test_default_floor_percentile_is_10pct(self) -> None:
        assert DEFAULT_FLOOR_PERCENTILE == 0.10

    def test_default_cap_mult_is_5x(self) -> None:
        assert DEFAULT_CAP_MULT == 5.0

    def test_sizing_decision_is_frozen(self) -> None:
        """Per coding-style.md immutability rule — SizingDecision must be
        a frozen dataclass."""
        bars = _bars_with_known_daily_vol(daily_log_ret=0.01, n_sessions=30)
        d = size_event(
            symbol="TEST", event_ts=bars.index[-1] + pd.Timedelta(minutes=5),
            bars_5min=bars, n_active_positions=1,
            target_daily_vol=0.0075,
            lookback_days=20, floor_percentile=0.10, cap_mult=5.0,
        )
        with pytest.raises((AttributeError, Exception)):
            d.applied_size = 999.0  # type: ignore[misc]
