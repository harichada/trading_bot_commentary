"""Session-clustered t-stat and recovery factor (Playbook Desk B1.5 task 9).

Same-session trades across symbols are correlated (they share the day's tape), so a
naive per-trade t-stat overstates significance on a pooled row. `t_stat_session`
groups by `entry_ts.normalize()` and treats the session as the independent unit.
`recovery` replaces the non-scale-invariant `max_dd_r <= 60R` pooled floor with a
net-R-over-drawdown ratio that stays meaningful regardless of trade count.
"""
import numpy as np
import pandas as pd
import pytest

from research.strategy_hunt.models import SimTrade
from research.strategy_hunt.score.metrics import compute_metrics


def _trade(r, day, symbol="NVDA", family="orb", side="long", window_id=0):
    ts = pd.Timestamp(f"{day} 10:00")
    return SimTrade(symbol, family, side, ts, ts + pd.Timedelta(minutes=30), 100.0, 99.0, 102.0, 100.0 + r,
                    100.0, 100.0 + r, r, r, 0.0, "target" if r > 0 else "stop", 6, 1, f"{family}_{side}",
                    "risk_on", window_id, "{}")


def _session(r_value, day, count=40):
    return [_trade(r_value, day) for _ in range(count)]


# ──────────────────────────────────────────────────────────────────────
# t_stat_session
# ──────────────────────────────────────────────────────────────────────

def test_t_stat_session_is_lower_than_iid_t_for_correlated_sessions():
    session_values = [1.0, 1.0, 1.0, -0.5, 1.0]
    days = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
    trades = [t for r, day in zip(session_values, days) for t in _session(r, day, count=40)]

    m = compute_metrics(trades, n_windows=1)

    arr = np.array(session_values)
    hand = float(arr.mean() / (arr.std(ddof=1) / np.sqrt(len(session_values))))
    assert m.t_stat_session == pytest.approx(hand)
    assert m.t_stat > m.t_stat_session


def test_t_stat_session_single_session_is_zero():
    trades = _session(1.0, "2025-01-02", count=40)
    m = compute_metrics(trades, n_windows=1)
    assert m.t_stat_session == 0.0


# ──────────────────────────────────────────────────────────────────────
# recovery
# ──────────────────────────────────────────────────────────────────────

def test_recovery_factor():
    trades = [_trade(2.0, "2025-01-02"), _trade(-1.0, "2025-01-03"), _trade(2.0, "2025-01-06"),
              _trade(-1.0, "2025-01-07"), _trade(2.0, "2025-01-08")]
    m = compute_metrics(trades, n_windows=1)
    assert m.recovery == pytest.approx(4.0)          # net 4, max_dd 1

    losing = [_trade(-1.0, "2025-01-02"), _trade(-2.0, "2025-01-03")]
    assert compute_metrics(losing, n_windows=1).recovery == 0.0

    winning_only = [_trade(1.0, "2025-01-02"), _trade(2.0, "2025-01-03")]
    assert compute_metrics(winning_only, n_windows=1).recovery == float("inf")
