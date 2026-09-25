import math
import numpy as np
import pandas as pd
import pytest

from research.strategy_hunt.models import SimTrade


def _t(r, window_id=0, day="2025-01-02", regime="mixed"):
    ts = pd.Timestamp(f"{day} 10:00")
    return SimTrade("X", "f", "long", ts, ts + pd.Timedelta(minutes=30), 100, 99.0, 102.0, 100 + r, 100, 100 + r,
                     r, r, 0.0, "target" if r > 0 else "stop", 6, 1, "", regime, window_id, "{}")


def test_metrics_basic():
    from research.strategy_hunt.score.metrics import compute_metrics
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-01-02", periods=12)]
    trades = [_t(2.0, 0, days[0]), _t(-1.0, 0, days[1]), _t(0.02, 0, days[2]),      # scratch excluded from wr
              _t(1.0, 1, days[3]), _t(1.0, 1, days[4]), _t(-1.0, 1, days[5]),
              _t(-1.0, 2, days[6]), _t(-1.0, 2, days[7]), _t(0.5, 2, days[8]),
              _t(1.5, 3, days[9]), _t(1.5, 3, days[10]), _t(-0.5, 3, days[11])]
    m = compute_metrics(trades, n_windows=4)
    assert m.n == 12 and m.sessions == 12
    assert m.exp_r == pytest.approx(sum(t.r_multiple for t in trades) / 12)
    assert m.wr == pytest.approx(6 / 11)
    assert m.pf == pytest.approx((2 + 1 + 1 + 0.5 + 1.5 + 1.5 + 0.02) / (1 + 1 + 1 + 1 + 0.5))
    # peak 3.02 (after the two +1.0s); cum drops to 0.02 by the third -1.0 => dd of 3.0 before recovering
    assert m.max_dd_r == pytest.approx(3.0)
    assert m.stability == pytest.approx(3 / 4)
    assert m.decay > 0


def test_stability_and_decay_need_four_qualifying_windows():
    from research.strategy_hunt.score.metrics import compute_metrics
    trades = [_t(1.0, w, f"2025-01-{2 + w:02d}") for w in range(3)] * 3
    m = compute_metrics(trades, n_windows=3)
    assert m.stability == 0.0 and m.decay == 0.0


def test_empty_and_no_loss_edge_cases():
    from research.strategy_hunt.score.metrics import compute_metrics
    m = compute_metrics([], n_windows=5)
    assert m.n == 0 and m.pf == 0.0 and m.wr == 0.0 and m.max_dd_r == 0.0
    m2 = compute_metrics([_t(1.0), _t(2.0)], n_windows=1)
    assert math.isinf(m2.pf) and m2.max_dd_r == 0.0


def test_per_regime():
    from research.strategy_hunt.score.metrics import per_regime
    out = per_regime([_t(1.0, regime="risk_on"), _t(-1.0, regime="risk_on"), _t(2.0, regime="risk_off")])
    assert out["risk_on"] == {"n": 2, "exp_r": 0.0} and out["risk_off"]["exp_r"] == 2.0


def test_verdicts():
    from research.strategy_hunt.models import FitMetrics
    from research.strategy_hunt.score.verdict import verdict, FLOORS
    good = FitMetrics(n=200, sessions=50, exp_r=0.2, exp_r_lb=0.05, t_stat=2.5, pf=1.5, wr=0.52, max_dd_r=8.0,
                       stability=0.7, decay=0.9)
    assert verdict(good) == ("PASS", [])
    watch = FitMetrics(n=80, sessions=30, exp_r=0.1, exp_r_lb=0.05, t_stat=2.5, pf=1.2, wr=0.5, max_dd_r=8.0,
                        stability=0.5, decay=0.9)
    v, failed = verdict(watch)
    assert v == "WATCH" and "min_n" in failed and "min_pf" in failed and "min_stability" in failed
    fail = FitMetrics(n=80, sessions=30, exp_r=-0.1, exp_r_lb=0.05, t_stat=2.5, pf=0.9, wr=0.4, max_dd_r=8.0,
                       stability=0.5, decay=0.9)
    assert verdict(fail)[0] == "FAIL"
    dd = FitMetrics(n=200, sessions=50, exp_r=0.2, exp_r_lb=0.05, t_stat=2.5, pf=1.5, wr=0.52, max_dd_r=25.0,
                     stability=0.7, decay=0.9)
    assert verdict(dd) == ("WATCH", ["max_dd_r"])
    assert FLOORS["min_n"] == 150


def test_t_stat_and_bootstrap_lb():
    from research.strategy_hunt.score.metrics import compute_metrics
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2025-01-02", periods=40)]
    trades = [_t(0.5 if i % 4 else -0.5, i // 10, days[i]) for i in range(40)]
    m = compute_metrics(trades, n_windows=4)
    r = np.array([t.r_multiple for t in trades])
    assert m.t_stat == pytest.approx(r.mean() / (r.std(ddof=1) / np.sqrt(40)))
    assert m.exp_r_lb < m.exp_r and m.exp_r_lb > 0
    assert compute_metrics(trades, 4).exp_r_lb == compute_metrics(trades, 4).exp_r_lb     # seeded
    assert compute_metrics(trades[:5], 1).exp_r_lb == 0.0 and compute_metrics([trades[0]], 1).t_stat == 0.0


def test_verdict_requires_t_stat():
    from research.strategy_hunt.models import FitMetrics
    from research.strategy_hunt.score.verdict import verdict
    weak = FitMetrics(n=200, sessions=50, exp_r=0.2, exp_r_lb=0.01, t_stat=1.2, pf=1.5, wr=0.52, max_dd_r=8.0,
                       stability=0.7, decay=0.9)
    assert verdict(weak) == ("WATCH", ["min_t_stat"])
