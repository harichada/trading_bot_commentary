"""Batch-2 Task E, Ruling 4: `splits_audit.audit_splits` flags a symbol whose adjusted daily
close min/max ratio is implausible, or whose context rs_120 is implausible, and flags neither
for an ordinary symbol."""
import numpy as np
import pandas as pd


def _write_daily(tmp_path, symbol, close):
    from research.strategy_hunt.data.bars import _write, daily_path

    idx = pd.bdate_range("2025-01-02", periods=len(close), name="date")
    d = pd.DataFrame({"open": close, "high": np.array(close) + 1.0, "low": np.array(close) - 1.0,
                      "close": close, "volume": np.full(len(close), 1_000_000.0)}, index=idx)
    _write(daily_path(tmp_path, symbol), d, "date")
    return d


def _write_bars5(tmp_path, symbol, n_sessions):
    from research.strategy_hunt.data.bars import _write, bars5_path

    b5 = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1},
                      index=pd.date_range("2025-01-02 09:30", periods=78 * n_sessions, freq="5min", name="ts"))
    b5 = b5[b5.index.time <= pd.Timestamp("15:55").time()]
    _write(bars5_path(tmp_path, symbol), b5, "ts")


def test_flags_symbol_with_implausible_minmax_ratio(tmp_path):
    from research.strategy_hunt.splits_audit import MAX_MINMAX_RATIO, audit_splits

    close = [1e-6] * 20 + [400.0] * 20                       # ratio ~4e8, no cache -> no splits applied
    _write_bars5(tmp_path, "BADX", 40)
    _write_daily(tmp_path, "BADX", close)
    flagged = audit_splits(tmp_path, "BADX")
    assert len(flagged) == 1 and flagged[0]["symbol"] == "BADX"
    assert any("minmax_ratio" in r for r in flagged[0]["reasons"])
    ratio = 400.0 / 1e-6
    assert ratio > MAX_MINMAX_RATIO


def test_does_not_flag_an_ordinary_symbol(tmp_path):
    from research.strategy_hunt.splits_audit import audit_splits

    close = [100.0 + i * 0.1 for i in range(40)]
    _write_bars5(tmp_path, "OKX", 40)
    _write_daily(tmp_path, "OKX", close)
    assert audit_splits(tmp_path, "OKX") == []


def test_flags_symbol_with_implausible_rs_120(tmp_path):
    from research.strategy_hunt.splits_audit import audit_splits

    close = [100.0] * 40
    _write_bars5(tmp_path, "RSX", 40)
    _write_daily(tmp_path, "RSX", close)
    ctx_dir = tmp_path / "context"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    ctx = pd.DataFrame({"date": pd.bdate_range("2025-01-02", periods=3), "symbol": ["RSX"] * 3,
                        "rs_120": [10.0, 900.0, -5.0]})
    ctx.to_parquet(ctx_dir / "universe_daily.parquet", index=False)
    flagged = audit_splits(tmp_path, "RSX")
    assert len(flagged) == 1
    assert any("rs_120" in r for r in flagged[0]["reasons"])
