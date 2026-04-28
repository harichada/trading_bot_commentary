"""Smoke tests for the P9 vol-target runner — exercise the per-event
sizing-walk function with synthetic inputs (no DB, no XGBoost, no
provider patching).

The full 20-window run lives in the integration phase (Phase 3 of the
P9 prompt). These tests guard:

  * the runner module is importable
  * its per-event walk function applies vol-target sizing per AFML §10.1
  * the equity-curve construction produces a sensible max-DD
  * the baseline construction (applied_size = 1.0 for every traded event)
    is computed identically on the same event sequence
  * P9.B cap invariant holds on the synthetic input
  * the per-window record schema matches what
    ``tests/test_p9_acceptance_gates.py`` expects

Strict TDD: this test file goes RED before ``train_p9_vol_target.py``
exists, GREEN once the runner exposes the documented helper API.
"""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def runner_module():
    return importlib.import_module("train_p9_vol_target")


def _synthetic_5min_bars(
    *,
    start: str,
    n_sessions: int,
    bars_per_session: int = 78,
    daily_log_ret: float = 0.01,
) -> pd.DataFrame:
    """Synthetic 5-min bars whose daily-resampled log-returns equal a
    constant. Deterministic input for sizing math."""
    closes = [100.0]
    for _ in range(n_sessions - 1):
        closes.append(closes[-1] * np.exp(daily_log_ret))
    rows: list[dict] = []
    base = pd.Timestamp(start)
    for d in range(n_sessions):
        day_open = base + pd.Timedelta(days=d, hours=14, minutes=30)
        for b in range(bars_per_session):
            ts = day_open + pd.Timedelta(minutes=5 * b)
            close = (
                closes[d]
                if b == bars_per_session - 1
                else 100.0 + 0.001 * (d * bars_per_session + b)
            )
            rows.append({
                "timestamp": ts,
                "open": close, "high": close, "low": close,
                "close": close, "volume": 1000,
            })
    return pd.DataFrame(rows).set_index("timestamp")


def _synthetic_event(
    *, symbol: str, entry_offset_min: int, exit_offset_min: int,
    R: float, regime: str = "trend_up_low_vol", fold: int = 1,
    decision: str = "trade",
) -> dict:
    base = pd.Timestamp("2025-01-31 14:30:00")
    return {
        "symbol": symbol,
        "entry_ts": (base + pd.Timedelta(minutes=entry_offset_min)).isoformat(),
        "exit_ts": (base + pd.Timedelta(minutes=exit_offset_min)).isoformat(),
        "regime": regime,
        "decision": decision,
        "R": R,
        "fold": fold,
    }


class TestRunnerImport:

    def test_module_importable(self, runner_module) -> None:
        """train_p9_vol_target is importable as a module."""
        assert runner_module is not None

    def test_main_callable_exists(self, runner_module) -> None:
        assert callable(getattr(runner_module, "main", None)), (
            "train_p9_vol_target.main must be a callable function"
        )

    def test_apply_sizing_helper_exists(self, runner_module) -> None:
        assert callable(getattr(runner_module, "_apply_sizing_to_window", None)), (
            "train_p9_vol_target._apply_sizing_to_window must exist"
        )

    def test_baseline_helper_exists(self, runner_module) -> None:
        assert callable(getattr(runner_module, "_baseline_window_record", None)), (
            "train_p9_vol_target._baseline_window_record must exist"
        )


class TestApplySizingToWindowSynthetic:
    """Drive ``_apply_sizing_to_window`` with synthetic events + bars,
    no provider patch needed."""

    @pytest.fixture
    def fixture(self, runner_module):
        bars = {
            "AAPL": _synthetic_5min_bars(
                start="2025-01-02 14:30:00", n_sessions=30, daily_log_ret=0.01,
            ),
            "TSLA": _synthetic_5min_bars(
                start="2025-01-02 14:30:00", n_sessions=30, daily_log_ret=0.015,
            ),
        }
        # Three trades, chronological. Mix of winners and losers.
        events = [
            _synthetic_event(
                symbol="AAPL", entry_offset_min=0, exit_offset_min=120, R=1.5,
            ),
            _synthetic_event(
                symbol="TSLA", entry_offset_min=30, exit_offset_min=180, R=-1.0,
            ),
            _synthetic_event(
                symbol="AAPL", entry_offset_min=240, exit_offset_min=480, R=0.8,
            ),
        ]
        return runner_module, bars, events

    def test_produces_per_event_records_with_required_keys(self, fixture) -> None:
        runner_module, bars, events = fixture
        rec = runner_module._apply_sizing_to_window(
            label="WSYNTH", events=events, bars=bars,
        )
        assert "per_event" in rec
        assert len(rec["per_event"]) == len(events)
        for ev in rec["per_event"]:
            for k in (
                "symbol", "entry_ts", "exit_ts", "regime", "decision",
                "R", "applied_size", "vol_parity_size", "cap_hit",
                "floor_hit", "portfolio_r",
            ):
                assert k in ev, f"per_event record missing {k!r}"

    def test_cap_invariant_holds(self, fixture) -> None:
        """Per-event applied_size / vol_parity_size <= 5 + ε for every
        traded event in the synthetic batch."""
        runner_module, bars, events = fixture
        rec = runner_module._apply_sizing_to_window(
            label="WSYNTH", events=events, bars=bars,
        )
        for ev in rec["per_event"]:
            if ev["decision"] != "trade":
                continue
            assert ev["vol_parity_size"] > 0
            ratio = ev["applied_size"] / ev["vol_parity_size"]
            assert ratio <= 5.0 + 1e-9, (
                f"cap violated on synthetic event {ev}: ratio={ratio:.6f}"
            )

    def test_default_multipliers_no_cap_hit(self, fixture) -> None:
        """With default multipliers (1.0, 1.0), no event should hit the
        cap on this benign synthetic input."""
        runner_module, bars, events = fixture
        rec = runner_module._apply_sizing_to_window(
            label="WSYNTH", events=events, bars=bars,
        )
        cap_hits = sum(
            1 for ev in rec["per_event"]
            if ev["decision"] == "trade" and ev["cap_hit"]
        )
        assert cap_hits == 0, (
            f"unexpected cap hits with default multipliers: {cap_hits}"
        )

    def test_equity_curve_and_max_dd_present(self, fixture) -> None:
        runner_module, bars, events = fixture
        rec = runner_module._apply_sizing_to_window(
            label="WSYNTH", events=events, bars=bars,
        )
        assert "equity_curve" in rec
        assert "max_dd" in rec
        # max_dd is non-positive (drawdown is negative or zero)
        assert rec["max_dd"] <= 0, f"max_dd must be <= 0, got {rec['max_dd']}"

    def test_realized_vol_field_present(self, fixture) -> None:
        runner_module, bars, events = fixture
        rec = runner_module._apply_sizing_to_window(
            label="WSYNTH", events=events, bars=bars,
        )
        assert "realized_portfolio_vol_daily" in rec
        # Realized portfolio vol must be a finite, non-negative float when
        # the equity curve has at least one daily return; with only 3
        # events on day 1, daily resample produces 1 daily point and
        # std is 0 or NaN — accept None in that degenerate case.
        v = rec["realized_portfolio_vol_daily"]
        assert v is None or (np.isfinite(v) and v >= 0)


class TestBaselineConstruction:

    def test_baseline_uses_unit_size_per_event(self, runner_module) -> None:
        """``_baseline_window_record`` produces a record where every
        traded event has applied_size == 1.0 (fixed-fractional reference)."""
        bars = {
            "AAPL": _synthetic_5min_bars(
                start="2025-01-02 14:30:00", n_sessions=30, daily_log_ret=0.01,
            ),
        }
        events = [
            _synthetic_event(
                symbol="AAPL", entry_offset_min=0, exit_offset_min=120, R=1.0,
            ),
            _synthetic_event(
                symbol="AAPL", entry_offset_min=240, exit_offset_min=360,
                R=-0.5,
            ),
        ]
        rec = runner_module._baseline_window_record(
            label="WSYNTH", events=events, bars=bars,
        )
        for ev in rec["per_event"]:
            if ev["decision"] != "trade":
                continue
            assert ev["applied_size"] == 1.0, (
                f"baseline must use applied_size=1.0; got {ev['applied_size']}"
            )

    def test_baseline_max_dd_present(self, runner_module) -> None:
        bars = {
            "AAPL": _synthetic_5min_bars(
                start="2025-01-02 14:30:00", n_sessions=30, daily_log_ret=0.01,
            ),
        }
        events = [
            _synthetic_event(
                symbol="AAPL", entry_offset_min=0, exit_offset_min=120, R=-1.0,
            ),
        ]
        rec = runner_module._baseline_window_record(
            label="WSYNTH", events=events, bars=bars,
        )
        assert "max_dd" in rec
        assert rec["max_dd"] <= 0
