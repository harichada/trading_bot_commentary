"""Regime-conditioned fit rows and universe-pooled regime lanes (spec §6)."""
import json
import math

import pandas as pd
import pytest

from research.strategy_hunt.models import TRADE_COLUMNS, FitMetrics, SimTrade


def _t(r, day, w, regime, symbol="NVDA", family="orb", side="long"):
    ts = pd.Timestamp(f"{day} 10:00")
    return SimTrade(symbol, family, side, ts, ts + pd.Timedelta(minutes=30), 100.0, 99.0, 102.0, 100.0 + r,
                    100.0, 100.0 + r, r, r, 0.0, "target" if r > 0 else "stop", 6, 1, f"{family}_{side}",
                    regime, w, '{"k": 1}')


def _many(regime="risk_on", per_day=10, sessions=60, symbol="NVDA", family="orb", side="long",
          start="2025-01-02", step=5):
    """`per_day * sessions` trades, two winners per loser, 10 sessions per walk-forward window.

    Sessions are spaced `step` business days apart, so the default 60 sessions span five
    calendar quarters — pooled rows bucket their windows by quarter and need at least the
    four qualifying buckets `metrics.MIN_QUALIFYING_WINDOWS` asks for."""
    out = []
    for di, d in enumerate(pd.bdate_range(start, periods=sessions * step)[::step]):
        for j in range(per_day):
            r = 1.0 if (di * per_day + j) % 3 else -1.0
            out.append(_t(r, d.strftime("%Y-%m-%d"), di // 10, regime, symbol, family, side))
    return out


def _frame(trades):
    return pd.DataFrame([t.to_dict() for t in trades], columns=list(TRADE_COLUMNS))


GOOD_EXCESS = 0.20


def _metrics(**kw):
    base = dict(n=600, sessions=60, exp_r=0.33, exp_r_lb=0.2, t_stat=8.0, pf=2.0, wr=0.66, max_dd_r=10.0,
                stability=1.0, decay=1.0, t_stat_session=10.0, recovery=10.0)
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
    rows = regime_rows("NVDA", "orb", "long", trades, 4, "2025-09-25T00:00:00", {"k": 1})

    assert {r["regime"] for r in rows} == {"risk_on", "mixed"}           # no row for "unknown"
    assert REGIME_ROW_KEYS == ROW_KEYS + ("regime",)
    by_regime = {r["regime"]: r for r in rows}
    for r in rows:
        assert tuple(r) == REGIME_ROW_KEYS
        assert r["symbol"] == "NVDA" and r["strategy"] == "orb" and r["side"] == "long"
        assert r["source"] == "walkforward" and r["regime"] in REGIMES
        assert r["computed_at"] == "2025-09-25T00:00:00"
        assert json.loads(r["params_json"]) == {"k": 1}
    assert sum(r["n"] for r in rows) == 400
    # each row's window_end is its own regime's last exit, never the caller's or the family's
    assert by_regime["risk_on"]["window_end"] == str(max(t.exit_ts for t in trades
                                                        if t.regime == "risk_on"))[:10]
    assert by_regime["risk_on"]["window_end"] < by_regime["mixed"]["window_end"]


def test_regime_rows_use_the_unconditional_floors():
    from research.strategy_hunt.score.regime_rows import regime_rows

    trades = _many(regime="risk_on", sessions=60) + [_t(-1.0, "2025-01-02", 0, "risk_off")]
    by_regime = {r["regime"]: r for r in regime_rows("NVDA", "orb", "long", trades, 6, "2025-09-25T00:00:00")}
    assert by_regime["risk_on"]["verdict"] == "PASS"                     # 600 trades, pf 2.0, 60 sessions
    # a single risk_off trade is no evidence, not a FAIL verdict: no row at all (finding 1)
    assert "risk_off" not in by_regime


def test_regime_rows_need_a_minimum_sample():
    """v-playbook-desk-b1-2026-09-25 (review r1, Important 1): below REGIME_ROW_MIN_N a
    regime row could only ever be FAIL (stability/decay are structurally 0 under four
    windows, and min_n is 150), and `lookup_regime` would let that manufactured FAIL
    outrank both the pooled PASS and the symbol's own unconditional PASS. "No evidence"
    must not read as "no edge" — such regimes get no row and fall through."""
    from research.strategy_hunt.score.regime_rows import REGIME_ROW_MIN_N, regime_rows
    from research.strategy_hunt.score.verdict import WATCH_FLOORS

    assert REGIME_ROW_MIN_N == WATCH_FLOORS["min_n"] == 60

    def _n(count):
        trades = _many(regime="risk_on", per_day=1, sessions=count, step=1)
        return {r["regime"] for r in regime_rows("NVDA", "orb", "long", trades, 6, "x")}

    assert _n(REGIME_ROW_MIN_N - 1) == set()          # 59 trades -> no row
    assert _n(REGIME_ROW_MIN_N) == {"risk_on"}        # 60 trades -> row


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
                             "min_recovery": 3.0, "min_stability": 0.6, "min_decay": 0.5, "min_t_stat": 3.0,
                             "min_t_stat_session": 3.0, "min_exp_r_lb": 0.0, "min_excess_r": 0.05}
    assert POOLED_FLOORS["min_n"] > FLOORS["min_n"] and POOLED_FLOORS["min_t_stat"] > FLOORS["min_t_stat"]
    assert "max_dd_r" not in POOLED_FLOORS          # not scale-invariant across pooled trade counts


def test_pooled_floors_use_recovery_not_max_dd():
    from research.strategy_hunt.score.regime_rows import _pooled_failures

    ok = _metrics(max_dd_r=639.0, recovery=3.6, t_stat_session=10.0)
    assert _pooled_failures(ok, GOOD_EXCESS) == []    # a 639R pooled drawdown no longer fails on its own

    low_recovery = _metrics(max_dd_r=639.0, recovery=2.0, t_stat_session=10.0)
    assert _pooled_failures(low_recovery, GOOD_EXCESS) == ["min_recovery"]

    low_t_session = _metrics(max_dd_r=639.0, recovery=3.6, t_stat_session=2.0)
    assert _pooled_failures(low_t_session, GOOD_EXCESS) == ["min_t_stat_session"]


def test_verdict_pooled_pass_watch_fail():
    from research.strategy_hunt.score.regime_rows import verdict_pooled

    assert verdict_pooled(_metrics(), GOOD_EXCESS) == ("PASS", [])
    v, failed = verdict_pooled(_metrics(n=300, sessions=40), GOOD_EXCESS)
    assert v == "WATCH" and set(failed) == {"min_n", "min_sessions"}      # n >= 200, exp_r > 0, pf >= 1.05
    v2, failed2 = verdict_pooled(_metrics(n=150), GOOD_EXCESS)
    assert v2 == "FAIL" and "min_n" in failed2
    assert verdict_pooled(_metrics(exp_r=-0.1), GOOD_EXCESS)[0] == "FAIL"   # exp_r <= 0 is never WATCH
    assert verdict_pooled(_metrics(n=300, sessions=40, pf=1.04), GOOD_EXCESS)[0] == "FAIL"
    # t_stat 2.5 clears the per-symbol floor (2.0) but not the pooled one (3.0)
    assert verdict_pooled(_metrics(t_stat=2.5), GOOD_EXCESS)[0] == "WATCH"


# ──────────────────────────────────────────────────────────────────────
# v-playbook-desk-b1-5 task 10 — excess over the shuffled baseline
# ──────────────────────────────────────────────────────────────────────

def test_verdict_pooled_requires_excess_over_the_shuffled_baseline():
    """The B1 run's shuffle control reached a pooled PASS while the real tape did not: a
    pooled row over a drifting regime measures the drift, not the rule. A pooled PASS now
    has to beat the same-regime shuffled baseline by at least POOLED_FLOORS["min_excess_r"]."""
    from research.strategy_hunt.score.regime_rows import POOLED_FLOORS, verdict_pooled

    assert POOLED_FLOORS["min_excess_r"] == 0.05
    thin, failed = verdict_pooled(_metrics(), 0.02)
    assert thin == "WATCH" and failed == ["min_excess_r"]
    assert verdict_pooled(_metrics(), 0.05) == ("PASS", [])

    none_v, none_failed = verdict_pooled(_metrics(), float("nan"))
    assert none_failed == ["no_baseline"] and none_v == "WATCH"           # never PASS without a baseline
    assert verdict_pooled(_metrics(n=10), float("nan"))[0] == "FAIL"


def test_verdict_pooled_defaults_to_no_baseline():
    """`build_row(verdict_fn=verdict_pooled)` passes the metrics only, so the default has to be
    the honest "no baseline measured" one rather than a free pass."""
    from research.strategy_hunt.score.regime_rows import verdict_pooled

    assert verdict_pooled(_metrics())[1] == ["no_baseline"]


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
    assert pooled["n"] == 1200 and pooled["sessions"] == 60
    # every floor but the new baseline one is cleared; with no baseline passed it can only WATCH
    assert pooled["verdict"] == "WATCH" and json.loads(pooled["failed_floors"]) == ["no_baseline"]
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


def test_pooled_rows_measure_stability_over_quarters_not_window_ids():
    """v-playbook-desk-b1-2026-09-25 (review r1, Minor 2): `window_id` is a per-symbol
    walk-forward index, so pooling by it mixes non-contemporaneous windows. Pooled rows
    bucket by calendar quarter of `entry_ts` instead."""
    from research.strategy_hunt.score.regime_rows import bucket_by_quarter, pooled_rows, quarter_index

    a = _many(symbol="AAA", sessions=60, step=1, start="2025-01-02")      # Q1 2025
    b = _many(symbol="BBB", sessions=60, step=1, start="2025-04-01")      # Q2 2025
    trades = a + b
    # both symbols reuse window_ids 0..5, and those ids are not contemporaneous
    assert {t.window_id for t in trades} == {0, 1, 2, 3, 4, 5}
    quarters = {str(pd.Timestamp(t.entry_ts).to_period("Q")) for t in trades}
    assert len({t.window_id for t in bucket_by_quarter(trades)}) == len(quarters) == 2
    assert [t.regime for t in bucket_by_quarter(trades)] == [t.regime for t in trades]   # only ids change
    # two quarters is under metrics.MIN_QUALIFYING_WINDOWS, so no stability is claimed --
    # bucketing by the six window_ids would have reported a spurious 1.0.
    assert pooled_rows(_frame(trades), "2025-09-25T00:00:00")[0]["stability"] == 0.0
    # ...while the same trade count spread over five quarters does clear it
    assert pooled_rows(_frame(_many(sessions=60)), "2025-09-25T00:00:00")[0]["stability"] == pytest.approx(1.0)
    # the bucket index is chronological across a year boundary
    assert quarter_index(pd.Timestamp("2025-12-31")) < quarter_index(pd.Timestamp("2026-01-01"))


def test_trades_from_frame_round_trips_a_trades_parquet_frame():
    from research.strategy_hunt.score.regime_rows import trades_from_frame

    df = _frame(_many(regime="risk_on"))
    trades = trades_from_frame(df)
    assert len(trades) == 600 and trades[0].regime == "risk_on" and trades[0].symbol == "NVDA"
    assert trades[0].window_id == 0 and trades[-1].window_id == 5
    assert trades_from_frame(df.assign(extra=1))[0].symbol == "NVDA"     # extra columns ignored


def _flat(r, symbol="NVDA", regime="risk_on", side="long", sessions=60, per_day=10, step=5,
          start="2025-01-02"):
    """Constant-`r` trades over five calendar quarters, so `exp_r` is exactly `r`."""
    out = []
    for di, d in enumerate(pd.bdate_range(start, periods=sessions * step)[::step]):
        for _ in range(per_day):
            out.append(_t(r, d.strftime("%Y-%m-%d"), di // 10, regime, symbol, "orb", side))
    return out


def _baseline_row(exp_r, regime="risk_on", side="long", n=600):
    from research.strategy_hunt.score.regime_rows import BASELINE_SOURCE
    return {"symbol": "*", "strategy": "orb", "side": side, "regime": regime, "n": n, "exp_r": exp_r,
            "source": BASELINE_SOURCE}


def test_pooled_rows_carry_excess_over_a_matching_baseline_group():
    from research.strategy_hunt.score.regime_rows import pooled_rows

    base = [_baseline_row(0.10)]
    thin = pooled_rows(_frame(_flat(0.12)), "2025-09-25T00:00:00", baseline=base)[0]
    assert thin["exp_r"] == pytest.approx(0.12)
    assert thin["baseline_exp_r"] == pytest.approx(0.10) and thin["baseline_n"] == 600
    assert thin["excess_r"] == pytest.approx(0.02)
    assert "min_excess_r" in json.loads(thin["failed_floors"])

    fat = pooled_rows(_frame(_flat(0.20)), "2025-09-25T00:00:00", baseline=base)[0]
    assert fat["excess_r"] == pytest.approx(0.10)
    assert "min_excess_r" not in json.loads(fat["failed_floors"])
    assert "no_baseline" not in json.loads(fat["failed_floors"])


def test_pooled_rows_without_a_matching_baseline_group_report_no_baseline():
    from research.strategy_hunt.score.regime_rows import pooled_rows

    # the only baseline group is a different regime, so this row has no baseline of its own
    row = pooled_rows(_frame(_flat(0.20)), "2025-09-25T00:00:00", baseline=[_baseline_row(0.10, regime="mixed")])[0]
    assert math.isnan(row["baseline_exp_r"]) and row["baseline_n"] == 0 and math.isnan(row["excess_r"])
    assert "no_baseline" in json.loads(row["failed_floors"])
    assert row["verdict"] in ("WATCH", "FAIL") and row["verdict"] != "PASS"

    # ...and with no baseline at all
    bare = pooled_rows(_frame(_flat(0.20)), "2025-09-25T00:00:00")[0]
    assert math.isnan(bare["excess_r"]) and "no_baseline" in json.loads(bare["failed_floors"])


def test_pooled_rows_keep_the_regime_row_shape():
    from research.strategy_hunt.score.regime_rows import REGIME_ROW_KEYS, pooled_rows

    row = pooled_rows(_frame(_flat(0.20)), "2025-09-25T00:00:00", baseline=[_baseline_row(0.10)])[0]
    assert tuple(row) == REGIME_ROW_KEYS


def test_baseline_pooled_rows_are_pooled_rows_stamped_baseline():
    from research.strategy_hunt.score.regime_rows import BASELINE_SOURCE, baseline_pooled_rows

    rows = baseline_pooled_rows(_frame(_flat(0.10)), "2025-09-25T00:00:00")
    assert len(rows) == 1 and rows[0]["source"] == BASELINE_SOURCE == "baseline"
    assert rows[0]["symbol"] == "*" and rows[0]["regime"] == "risk_on"
    assert rows[0]["exp_r"] == pytest.approx(0.10) and rows[0]["n"] == 600
    # a baseline row has no baseline of its own
    assert math.isnan(rows[0]["excess_r"]) and rows[0]["baseline_n"] == 0
    assert baseline_pooled_rows(None, "2025-09-25T00:00:00") == []
