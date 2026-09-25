import numpy as np
import pandas as pd
import pytest


def _trade(sym, fam, entry, exit_, entry_fill, exit_fill, side="long", stop=None):
    entry, exit_ = pd.Timestamp(entry), pd.Timestamp(exit_)
    stop = stop if stop is not None else (entry_fill * 0.98 if side == "long" else entry_fill * 1.02)
    return {"symbol": sym, "family": fam, "side": side, "entry_ts": entry, "exit_ts": exit_, "entry_px": entry_fill,
            "stop_px": stop, "target_px": entry_fill * 1.04, "exit_px": exit_fill, "entry_fill": entry_fill, "exit_fill": exit_fill,
            "r_multiple": 1.0, "pnl_pct": 1.0, "cost_pct": 0.0, "exit_reason": "target", "hold_bars": 5,
            "hold_sessions": 1 if entry.date() == exit_.date() else 2, "tag": "", "regime": "mixed", "window_id": 0, "params_json": "{}"}


def test_shares_for_caps():
    from research.strategy_hunt.portfolio import SizingRules, shares_for
    r = SizingRules()
    # risk = 43500*0.015*0.5*0.5 = 163.1 ; stop 2 away -> 81 shares ; notional 8100 > cap min(10000, 43500*0.69*0.2=6003) -> 60
    assert shares_for(r, 43500.0, 100.0, 98.0, True) == 60
    assert shares_for(r, 43500.0, 100.0, 98.0, False) == 60
    assert shares_for(r, 43500.0, 1000.0, 999.0, True) == 6
    assert shares_for(r, 1000.0, 100.0, 50.0, True) == 0


def test_replay_capacity_and_symbol_rules():
    from research.strategy_hunt.portfolio import replay, SizingRules
    rules = SizingRules(max_positions=2)
    rows = [_trade("A", "f", "2025-01-02 10:00", "2025-01-02 11:00", 100, 101),
            _trade("B", "f", "2025-01-02 10:05", "2025-01-02 11:00", 100, 101),
            _trade("C", "f", "2025-01-02 10:10", "2025-01-02 11:00", 100, 101),      # capacity full -> skipped
            _trade("A", "g", "2025-01-02 10:30", "2025-01-02 12:00", 100, 99),       # same symbol open -> skipped
            _trade("D", "f", "2025-01-02 11:30", "2025-01-02 12:00", 100, 102)]      # slots free again -> taken
    out = replay(pd.DataFrame(rows), rules)
    assert out["taken"] == 3 and out["skipped_capacity"] == 1 and out["skipped_symbol"] == 1
    assert out["final_equity"] > 43500.0 and out["total_return_pct"] > 0 and out["max_drawdown_pct"] >= 0
    assert set(out["by_family"]) == {"f"} and out["by_family"]["f"]["taken"] == 3
    assert isinstance(out["equity_curve"], pd.Series) and out["equity_curve"].index[0] == pd.Timestamp("2025-01-02")


def test_replay_drawdown_and_sharpe_shapes():
    from research.strategy_hunt.portfolio import replay
    rows = [_trade("A", "f", f"2025-01-{d:02d} 10:00", f"2025-01-{d:02d} 11:00", 100, 100 + (1 if d % 2 else -1)) for d in range(2, 30) if pd.Timestamp(f"2025-01-{d:02d}").weekday() < 5]
    out = replay(pd.DataFrame(rows))
    assert out["taken"] == len(rows) and out["max_drawdown_pct"] > 0 and np.isfinite(out["daily_sharpe"])


def test_pass_lane_filter():
    from research.strategy_hunt.portfolio import pass_lane_trades
    df = pd.DataFrame([_trade("A", "f", "2025-01-02 10:00", "2025-01-02 11:00", 100, 101),
                       _trade("B", "f", "2025-01-02 10:00", "2025-01-02 11:00", 100, 101)])
    rows = [{"symbol": "A", "strategy": "f", "side": "long", "verdict": "PASS", "source": "walkforward"},
            {"symbol": "B", "strategy": "f", "side": "long", "verdict": "WATCH", "source": "walkforward"}]
    assert list(pass_lane_trades(df, rows)["symbol"]) == ["A"]


def test_pass_lane_filter_excludes_missing_source():
    from research.strategy_hunt.portfolio import pass_lane_trades
    df = pd.DataFrame([_trade("A", "f", "2025-01-02 10:00", "2025-01-02 11:00", 100, 101)])
    rows = [{"symbol": "A", "strategy": "f", "side": "long", "verdict": "PASS"}]  # no "source" key
    assert list(pass_lane_trades(df, rows)["symbol"]) == []
