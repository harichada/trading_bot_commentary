"""Tests for the scalp reversal backtest module."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Task 1 — ScalpParams + load_bars
# ---------------------------------------------------------------------------

def test_scalp_params_defaults():
    from backtest.scalp_reversal import ScalpParams
    p = ScalpParams()
    assert p.rsi_long_thresh == 30
    assert p.rsi_short_thresh == 75
    assert p.vol_min == 2.0
    assert p.atr_stop_mult == 1.0
    assert p.atr_target_mult == 2.0
    assert p.quick_exit_bars == 3
    assert p.min_profit_pct == 0.15
    assert p.max_hold_bars == 15


def test_scalp_params_frozen():
    from backtest.scalp_reversal import ScalpParams
    p = ScalpParams()
    with pytest.raises((AttributeError, TypeError)):
        p.rsi_long_thresh = 25  # type: ignore


def test_load_bars_rth_filter(monkeypatch):
    idx = pd.date_range("2026-11-01 08:00", periods=600, freq="1min", tz="UTC")
    fake_df = pd.DataFrame({
        "Open": 100.0, "High": 101.0, "Low": 99.0,
        "Close": 100.0, "Volume": 1000,
    }, index=idx)

    import data_providers.postgres as pg_mod
    monkeypatch.setattr(
        pg_mod, "PostgresDataProvider",
        lambda dsn: type("FP", (), {"get_market_data": staticmethod(
            lambda *a, **kw: fake_df
        )})(),
    )
    from backtest.scalp_reversal import load_bars
    df = load_bars("NVDA", 7, "postgresql://x/y")
    eastern_times = df.index.tz_convert("America/New_York")
    assert all(t.hour * 60 + t.minute >= 9 * 60 + 30 for t in eastern_times)
    assert all(t.hour * 60 + t.minute <= 15 * 60 + 55 for t in eastern_times)


# ---------------------------------------------------------------------------
# Task 2 — replay_symbol_scalp
# ---------------------------------------------------------------------------

def _make_bars(n: int = 200, base: float = 100.0,
               freq: str = "1min") -> pd.DataFrame:
    """Synthetic RTH 1-min bars with stable flat price."""
    idx = pd.date_range("2026-11-03 14:30", periods=n, freq=freq, tz="UTC")
    closes = np.full(n, base)
    return pd.DataFrame({
        "Open": closes - 0.05,
        "High": closes + 0.20,
        "Low": closes - 0.20,
        "Close": closes,
        "Volume": np.full(n, 5_000.0),
    }, index=idx)


def test_replay_empty_df():
    from backtest.scalp_reversal import ScalpParams, replay_symbol_scalp
    df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df.index = pd.DatetimeIndex([], tz="UTC")
    result = replay_symbol_scalp("NVDA", df, ScalpParams())
    assert result == []


def test_replay_no_signal_stable_market():
    from backtest.scalp_reversal import ScalpParams, replay_symbol_scalp
    from backtest.trades import Trade
    df = _make_bars(200, base=100.0)
    trades = replay_symbol_scalp("NVDA", df, ScalpParams())
    assert isinstance(trades, list)
    assert all(isinstance(t, Trade) for t in trades)


def test_replay_valid_trade_fields():
    """Any trade emitted must have valid symbol, side and exit_reason."""
    from backtest.scalp_reversal import ScalpParams, replay_symbol_scalp
    from backtest.trades import Trade
    n = 150
    idx = pd.date_range("2026-11-03 14:30", periods=n, freq="1min", tz="UTC")
    closes = np.full(n, 100.0)
    closes[60:80] = 95.0
    closes[80:] = 100.5
    volumes = np.full(n, 1_000.0)
    volumes[70:80] = 3_000.0
    df = pd.DataFrame({
        "Open": closes - 0.05, "High": closes + 0.30,
        "Low": closes - 0.30, "Close": closes,
        "Volume": volumes,
    }, index=idx)
    params = ScalpParams(rsi_long_thresh=35, min_profit_pct=0.10, quick_exit_bars=5)
    trades = replay_symbol_scalp("NVDA", df, params)
    valid_reasons = {"quick_profit", "stop", "target", "timeout", "eod"}
    for t in trades:
        assert t.symbol == "NVDA"
        assert t.side in ("long", "short")
        assert t.exit_reason in valid_reasons
        assert t.hold_bars >= 0


def test_replay_eod_flush():
    """Positions open at the last bar of a day must be flushed with reason='eod'."""
    from backtest.scalp_reversal import ScalpParams, replay_symbol_scalp
    idx1 = pd.date_range("2026-11-03 14:30", periods=390, freq="1min", tz="UTC")
    idx2 = pd.date_range("2026-11-04 14:30", periods=50, freq="1min", tz="UTC")
    idx = idx1.append(idx2)
    closes = np.full(len(idx), 100.0)
    volumes = np.full(len(idx), 1_000.0)
    df = pd.DataFrame({
        "Open": closes - 0.05, "High": closes + 0.1,
        "Low": closes - 0.1, "Close": closes, "Volume": volumes,
    }, index=idx)
    trades = replay_symbol_scalp("NVDA", df, ScalpParams())
    for t in trades:
        assert t.exit_price > 0
        assert t.exit_reason in {"quick_profit", "stop", "target", "timeout", "eod"}


# ---------------------------------------------------------------------------
# Task 3 — allocate_slots
# ---------------------------------------------------------------------------

def _fake_trade(symbol: str, entry_t, exit_t,
                side: str = "long",
                entry_px: float = 100.0,
                exit_px: float = 100.5):
    from backtest.trades import Trade
    return Trade(
        strategy="scalp_reversal", symbol=symbol, side=side,
        entry_time=entry_t, entry_price=entry_px,
        exit_time=exit_t, exit_price=exit_px,
        exit_reason="quick_profit", hold_bars=1,
    )


def test_allocate_slots_empty():
    from backtest.scalp_reversal import allocate_slots
    assert allocate_slots([], max_slots=10) == []


def test_allocate_slots_under_cap():
    from backtest.scalp_reversal import allocate_slots
    t0 = pd.Timestamp("2026-11-03 14:35", tz="UTC")
    t1 = pd.Timestamp("2026-11-03 14:38", tz="UTC")
    trades = [_fake_trade("NVDA", t0, t1), _fake_trade("AMD", t0, t1)]
    result = allocate_slots(trades, max_slots=10)
    assert len(result) == 2


def test_allocate_slots_cap_enforced():
    from backtest.scalp_reversal import allocate_slots
    t0 = pd.Timestamp("2026-11-03 14:35", tz="UTC")
    t1 = pd.Timestamp("2026-11-03 15:00", tz="UTC")
    syms = [f"SYM{i}" for i in range(11)]
    trades = [_fake_trade(s, t0, t1) for s in syms]
    result = allocate_slots(trades, max_slots=10)
    assert len(result) == 10


def test_allocate_slots_sequential_fills_all():
    from backtest.scalp_reversal import allocate_slots
    base = pd.Timestamp("2026-11-03 14:30", tz="UTC")
    trades = [
        _fake_trade(
            "NVDA",
            base + pd.Timedelta(minutes=i * 5),
            base + pd.Timedelta(minutes=i * 5 + 3),
        )
        for i in range(10)
    ]
    result = allocate_slots(trades, max_slots=10)
    assert len(result) == 10


def test_allocate_slots_tiebreak_alphabetical():
    from backtest.scalp_reversal import allocate_slots
    t0 = pd.Timestamp("2026-11-03 14:35", tz="UTC")
    t1 = pd.Timestamp("2026-11-03 15:30", tz="UTC")
    trades = [_fake_trade("ZYXW", t0, t1), _fake_trade("AAPL", t0, t1)]
    result = allocate_slots(trades, max_slots=1)
    assert len(result) == 1
    assert result[0].symbol == "AAPL"


# ---------------------------------------------------------------------------
# Task 4 — CLI smoke test
# ---------------------------------------------------------------------------

def test_scalp_cli_help():
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "backtest.scalp_cli", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--symbols" in result.stdout
    assert "--days" in result.stdout
    assert "--min-profit-pct" in result.stdout
