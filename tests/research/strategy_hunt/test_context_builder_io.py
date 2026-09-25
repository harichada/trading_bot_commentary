"""`build_universe_context` / `load_context_slice`: the on-disk read/write side of the universe
context.

Split out of `test_context_builder.py` (v-playbook-desk-b1-6, final review Minor 3): that module
had grown to 520 lines covering both the per-symbol feature maths and this I/O surface. The
feature tests stay there; everything that writes a root and reads it back lives here.
"""
import numpy as np
import pandas as pd
import pytest


def _write_symbol(tmp_path, symbol: str, n: int = 15, daily_volume: float = 1_000_000.0) -> None:
    from research.strategy_hunt.data.bars import _write, bars5_path, daily_path

    dates = pd.bdate_range("2025-02-03", periods=n, name="date")
    close = np.full(n, 50.0)
    daily = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": np.full(n, daily_volume)},
        index=dates,
    )
    idx, rows = [], []
    for d in dates:
        for j in range(3):
            idx.append(d + pd.Timedelta(minutes=5 * j))
            rows.append({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1000.0})
    bars5 = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="ts"))
    _write(bars5_path(tmp_path, symbol), bars5, "ts")
    _write(daily_path(tmp_path, symbol), daily, "date")


def test_build_universe_context_writes_file_and_load_context_slice_reads_it(tmp_path):
    from research.strategy_hunt.context import build_universe_context, load_context_slice
    from strategies.families.context import CTX_COLUMNS

    _write_symbol(tmp_path, "SPY", daily_volume=1_000_000.0)
    _write_symbol(tmp_path, "VIX", daily_volume=1_000_000.0)
    _write_symbol(tmp_path, "TEST1", daily_volume=5_000_000.0)
    _write_symbol(tmp_path, "TEST2", daily_volume=1_000_000.0)

    out_path = build_universe_context(tmp_path)
    assert out_path == tmp_path / "context" / "universe_daily.parquet"
    assert out_path.exists()

    written = pd.read_parquet(out_path)
    assert set(written["symbol"].unique()) == {"TEST1", "TEST2"}  # SPY/VIX excluded
    assert list(written.columns) == ["date", "symbol", *CTX_COLUMNS]

    slice1 = load_context_slice(tmp_path, "TEST1")
    assert not slice1.empty
    assert slice1.index.name == "date"
    assert list(slice1.columns) == list(CTX_COLUMNS)

    empty = load_context_slice(tmp_path, "NOSUCHSYMBOL")
    assert empty.empty and list(empty.columns) == list(CTX_COLUMNS)


def test_load_context_slice_missing_root_is_empty(tmp_path):
    from research.strategy_hunt.context import load_context_slice
    from strategies.families.context import CTX_COLUMNS

    out = load_context_slice(tmp_path, "AAPL")
    assert out.empty
    assert list(out.columns) == list(CTX_COLUMNS)
    assert isinstance(out.index, pd.DatetimeIndex) and out.index.name == "date"


def test_build_universe_context_skips_symbol_on_failure(tmp_path, monkeypatch):
    from research.strategy_hunt import context as C

    _write_symbol(tmp_path, "SPY")
    _write_symbol(tmp_path, "TEST1")

    def boom(symbol, bars5, daily, spy_daily, earnings_table, news):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(C, "symbol_daily_features", boom)
    out_path = C.build_universe_context(tmp_path)
    written = pd.read_parquet(out_path)
    assert written.empty  # the only non-SPY symbol failed and was skipped, not raised


def test_build_universe_context_loads_spy_via_load_adjusted(tmp_path, monkeypatch):
    """SPY must go through the same split-adjustment path as every other symbol, not the raw
    load_daily -- otherwise rs_* compares an adjusted symbol close against an unadjusted SPY."""
    from research.strategy_hunt import context as C
    from research.strategy_hunt.data import splits as S

    _write_symbol(tmp_path, "SPY")
    _write_symbol(tmp_path, "TEST1")

    calls = []
    real_load_adjusted = S.load_adjusted

    def spy_load_adjusted(root, symbol):
        calls.append(symbol)
        return real_load_adjusted(root, symbol)

    monkeypatch.setattr(S, "load_adjusted", spy_load_adjusted)
    C.build_universe_context(tmp_path)
    assert "SPY" in calls


def test_build_universe_context_raises_when_spy_missing(tmp_path):
    from research.strategy_hunt.context import build_universe_context

    _write_symbol(tmp_path, "TEST1")  # no SPY bars5/daily written at all
    with pytest.raises(ValueError, match="SPY"):
        build_universe_context(tmp_path)


def _write_bars5_only(tmp_path, symbol: str, n: int = 10) -> None:
    from research.strategy_hunt.data.bars import _write, bars5_path

    dates = pd.bdate_range("2025-03-03", periods=n, name="date")
    idx, rows = [], []
    for d in dates:
        for j in range(3):
            idx.append(d + pd.Timedelta(minutes=5 * j))
            rows.append({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1000.0})
    bars5 = pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="ts"))
    _write(bars5_path(tmp_path, symbol), bars5, "ts")


def test_build_universe_context_warns_when_symbol_has_no_daily_data(tmp_path, caplog):
    from research.strategy_hunt.context import build_universe_context

    _write_symbol(tmp_path, "SPY")
    _write_bars5_only(tmp_path, "NODAILY")  # bars5 exists, daily/ does not

    with caplog.at_level("WARNING", logger="strategy_hunt.context"):
        out_path = build_universe_context(tmp_path)

    assert any("NODAILY" in rec.message for rec in caplog.records)
    written = pd.read_parquet(out_path)
    assert "NODAILY" not in set(written["symbol"].unique())
