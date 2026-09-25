"""Regime-conditioned fit rows and universe-pooled regime lanes (spec §6)."""
import json

import pandas as pd
import pytest

from research.strategy_hunt.models import TRADE_COLUMNS, FitMetrics, SimTrade


def _t(r, day, w, regime, symbol="NVDA", family="orb", side="long"):
    ts = pd.Timestamp(f"{day} 10:00")
    return SimTrade(symbol, family, side, ts, ts + pd.Timedelta(minutes=30), 100.0, 99.0, 102.0, 100.0 + r,
                    100.0, 100.0 + r, r, r, 0.0, "target" if r > 0 else "stop", 6, 1, f"{family}_{side}",
                    regime, w, '{"k": 1}')


def _many(regime="risk_on", per_day=10, sessions=60, symbol="NVDA", family="orb", side="long", start="2025-01-02"):
    """`per_day * sessions` trades, two winners per loser, 10 sessions per window."""
    out = []
    for di, d in enumerate(pd.bdate_range(start, periods=sessions)):
        for j in range(per_day):
            r = 1.0 if (di * per_day + j) % 3 else -1.0
            out.append(_t(r, d.strftime("%Y-%m-%d"), di // 10, regime, symbol, family, side))
    return out


def _frame(trades):
    return pd.DataFrame([t.to_dict() for t in trades], columns=list(TRADE_COLUMNS))


def _metrics(**kw):
    base = dict(n=600, sessions=60, exp_r=0.33, exp_r_lb=0.2, t_stat=8.0, pf=2.0, wr=0.66, max_dd_r=10.0,
                stability=1.0, decay=1.0)
    base.update(kw)
    return FitMetrics(**base)


# ──────────────────────────────────────────────────────────────────────
# regime_rows
# ──────────────────────────────────────────────────────────────────────

def test_regime_rows_one_row_per_known_regime_present():
    from research.strategy_hunt.fit.writer import ROW_KEYS
    from research.strategy_hunt.score.regime_rows import REGIME_ROW_KEYS, REGIMES, regime_rows

    trades = _many(regime="risk_on", sessions=20) + _many(regime="mixed", sessions=20, start="2025-04-01")
    trades += [_t(0.5, "2025-07-01", 9, "unknown")]
    rows = regime_rows("NVDA", "orb", "long", trades, 4, "2025-09-25T00:00:00", {"k": 1}, "2025-09-24")

    assert {r["regime"] for r in rows} == {"risk_on", "mixed"}           # no row for "unknown"
    assert REGIME_ROW_KEYS == ROW_KEYS + ("regime",)
    for r in rows:
        assert tuple(r) == REGIME_ROW_KEYS
        assert r["symbol"] == "NVDA" and r["strategy"] == "orb" and r["side"] == "long"
        assert r["source"] == "walkforward" and r["regime"] in REGIMES
        assert r["computed_at"] == "2025-09-25T00:00:00" and r["window_end"] == "2025-09-24"
        assert json.loads(r["params_json"]) == {"k": 1}
    assert sum(r["n"] for r in rows) == 400


def test_regime_rows_use_the_unconditional_floors():
    from research.strategy_hunt.score.regime_rows import regime_rows

    trades = _many(regime="risk_on", sessions=60) + [_t(-1.0, "2025-01-02", 0, "risk_off")]
    by_regime = {r["regime"]: r for r in regime_rows("NVDA", "orb", "long", trades, 6, "2025-09-25T00:00:00")}
    assert by_regime["risk_on"]["verdict"] == "PASS"                     # 600 trades, pf 2.0, 60 sessions
    assert by_regime["risk_off"]["verdict"] == "FAIL" and by_regime["risk_off"]["n"] == 1


def test_regime_rows_empty_for_no_trades():
    from research.strategy_hunt.score.regime_rows import regime_rows

    assert regime_rows("NVDA", "orb", "long", [], 0, "2025-09-25T00:00:00") == []
    assert regime_rows("NVDA", "orb", "long", [_t(1.0, "2025-01-02", 0, "unknown")], 1, "x") == []


# ──────────────────────────────────────────────────────────────────────
# verdict_pooled
# ──────────────────────────────────────────────────────────────────────

def test_pooled_floors_are_stricter_than_the_per_symbol_floors():
    from research.strategy_hunt.score.regime_rows import POOLED_FLOORS
    from research.strategy_hunt.score.verdict import FLOORS

    assert POOLED_FLOORS == {"min_n": 500, "min_sessions": 60, "min_pf": 1.2, "min_wr": 0.45, "min_exp_r": 0.05,
                             "max_dd_r": 60.0, "min_stability": 0.6, "min_decay": 0.5, "min_t_stat": 3.0,
                             "min_exp_r_lb": 0.0}
    assert POOLED_FLOORS["min_n"] > FLOORS["min_n"] and POOLED_FLOORS["min_t_stat"] > FLOORS["min_t_stat"]


def test_verdict_pooled_pass_watch_fail():
    from research.strategy_hunt.score.regime_rows import verdict_pooled

    assert verdict_pooled(_metrics()) == ("PASS", [])
    v, failed = verdict_pooled(_metrics(n=300, sessions=40))
    assert v == "WATCH" and set(failed) == {"min_n", "min_sessions"}      # n >= 200, exp_r > 0, pf >= 1.05
    v2, failed2 = verdict_pooled(_metrics(n=150))
    assert v2 == "FAIL" and "min_n" in failed2
    assert verdict_pooled(_metrics(exp_r=-0.1))[0] == "FAIL"              # exp_r <= 0 is never WATCH
    assert verdict_pooled(_metrics(n=300, sessions=40, pf=1.04))[0] == "FAIL"
    # t_stat 2.5 clears the per-symbol floor (2.0) but not the pooled one (3.0)
    assert verdict_pooled(_metrics(t_stat=2.5))[0] == "WATCH"


# ──────────────────────────────────────────────────────────────────────
# pooled_rows
# ──────────────────────────────────────────────────────────────────────

def test_pooled_rows_concatenate_the_universe_per_family_side_regime():
    from research.strategy_hunt.score.regime_rows import POOLED_SOURCE, POOLED_SYMBOL, pooled_rows

    trades = _many(symbol="NVDA", regime="risk_on") + _many(symbol="TSLA", regime="risk_on")
    trades += _many(symbol="NVDA", regime="mixed", sessions=30)
    trades += _many(symbol="NVDA", regime="risk_on", side="short", sessions=10)
    rows = pooled_rows(_frame(trades), "2025-09-25T00:00:00")

    keyed = {(r["strategy"], r["side"], r["regime"]): r for r in rows}
    assert set(keyed) == {("orb", "long", "risk_on"), ("orb", "long", "mixed"), ("orb", "short", "risk_on")}
    pooled = keyed[("orb", "long", "risk_on")]
    assert pooled["symbol"] == POOLED_SYMBOL == "*" and pooled["source"] == POOLED_SOURCE == "pooled"
    assert pooled["n"] == 1200 and pooled["sessions"] == 60 and pooled["verdict"] == "PASS"
    assert pooled["window_end"] == str(max(t.exit_ts for t in trades if t.regime == "risk_on"))[:10]
    assert keyed[("orb", "long", "mixed")]["n"] == 300 and keyed[("orb", "long", "mixed")]["verdict"] == "WATCH"
    assert keyed[("orb", "short", "risk_on")]["verdict"] == "FAIL"        # 100 trades


def test_pooled_rows_skip_unknown_regime_and_empty_frame():
    from research.strategy_hunt.models import TRADE_COLUMNS
    from research.strategy_hunt.score.regime_rows import pooled_rows

    trades = _many(regime="unknown", sessions=60)
    assert pooled_rows(_frame(trades), "2025-09-25T00:00:00") == []
    assert pooled_rows(pd.DataFrame(columns=list(TRADE_COLUMNS)), "2025-09-25T00:00:00") == []
    assert pooled_rows(None, "2025-09-25T00:00:00") == []


def test_pooled_rows_count_windows_by_distinct_window_id():
    from research.strategy_hunt.score.regime_rows import pooled_rows, trades_from_frame

    trades = _many(regime="risk_on")
    df = _frame(trades)
    assert len(trades_from_frame(df)) == 600 and trades_from_frame(df)[0].regime == "risk_on"
    # stability is measured over windows: 6 distinct window_ids, all positive -> 1.0
    row = pooled_rows(df, "2025-09-25T00:00:00")[0]
    assert row["stability"] == pytest.approx(1.0)
