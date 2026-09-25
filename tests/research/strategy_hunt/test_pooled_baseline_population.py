"""v-playbook-desk-b1-6 (final review, Important 2) — baseline_population_mismatch.

A shuffled-control group whose trade count is nowhere near the real group's is not a
yardstick for it: the two rows measure different populations, so their expectancy difference
is not the "excess over drift" the pooled verdict claims it is. The pooled row says so on
itself, and a mismatched baseline can never mint a PASS.
"""
import json

import pandas as pd
import pytest

from research.strategy_hunt.models import TRADE_COLUMNS, FitMetrics, SimTrade

MISMATCH = "baseline_population_mismatch"


def _t(r, day, w, regime="risk_on", symbol="NVDA", family="orb", side="long"):
    ts = pd.Timestamp(f"{day} 10:00")
    return SimTrade(symbol, family, side, ts, ts + pd.Timedelta(minutes=30), 100.0, 99.0, 102.0, 100.0 + r,
                    100.0, 100.0 + r, r, r, 0.0, "target" if r > 0 else "stop", 6, 1, f"{family}_{side}",
                    regime, w, '{"k": 1}')


def _flat(r, sessions=60, per_day=10, step=5, start="2025-01-02"):
    out = []
    for di, d in enumerate(pd.bdate_range(start, periods=sessions * step)[::step]):
        for _ in range(per_day):
            out.append(_t(r, d.strftime("%Y-%m-%d"), di // 10))
    return out


def _frame(trades):
    return pd.DataFrame([t.to_dict() for t in trades], columns=list(TRADE_COLUMNS))


def _baseline_row(exp_r, n, regime="risk_on", side="long"):
    from research.strategy_hunt.score.regime_rows import BASELINE_SOURCE
    return {"symbol": "*", "strategy": "orb", "side": side, "regime": regime, "n": n, "exp_r": exp_r,
            "source": BASELINE_SOURCE}


def _metrics(**kw):
    base = dict(n=600, sessions=60, exp_r=0.33, exp_r_lb=0.2, t_stat=8.0, pf=2.0, wr=0.66, max_dd_r=10.0,
                stability=1.0, decay=1.0, t_stat_session=10.0, recovery=10.0)
    base.update(kw)
    return FitMetrics(**base)


def test_constants_are_the_ruling_bounds():
    from research.strategy_hunt.score.regime_rows import (BASELINE_POPULATION_MAX_RATIO,
                                                          BASELINE_POPULATION_MIN_RATIO,
                                                          BASELINE_POPULATION_MISMATCH)
    assert BASELINE_POPULATION_MISMATCH == MISMATCH
    assert (BASELINE_POPULATION_MIN_RATIO, BASELINE_POPULATION_MAX_RATIO) == (0.5, 2.0)


@pytest.mark.parametrize("baseline_n,flagged", [(299, True), (300, False), (600, False),
                                               (1200, False), (1201, True), (5, True)])
def test_verdict_pooled_flags_a_baseline_population_far_from_the_row(baseline_n, flagged):
    from research.strategy_hunt.score.regime_rows import verdict_pooled
    v, failed = verdict_pooled(_metrics(n=600), 0.20, baseline_n=baseline_n)
    assert (MISMATCH in failed) is flagged
    assert (v == "PASS") is not flagged


def test_population_check_is_skipped_when_there_is_no_baseline_at_all():
    """`no_baseline` already says what happened; saying it twice only obscures which it was."""
    from research.strategy_hunt.score.regime_rows import verdict_pooled
    v, failed = verdict_pooled(_metrics(), float("nan"), baseline_n=0)
    assert failed == ["no_baseline"] and v == "WATCH"


def test_unsupplied_baseline_n_leaves_the_existing_contract_untouched():
    from research.strategy_hunt.score.regime_rows import verdict_pooled
    assert verdict_pooled(_metrics(), 0.20) == ("PASS", [])


def test_pooled_rows_always_supply_baseline_n_and_flag_the_mismatch():
    from research.strategy_hunt.score.regime_rows import pooled_rows
    ok = pooled_rows(_frame(_flat(0.20)), "2025-09-25T00:00:00", baseline=[_baseline_row(0.10, 600)])[0]
    assert ok["baseline_n"] == 600 and MISMATCH not in json.loads(ok["failed_floors"])

    thin = pooled_rows(_frame(_flat(0.20)), "2025-09-25T00:00:00", baseline=[_baseline_row(0.10, 50)])[0]
    assert thin["baseline_n"] == 50 and MISMATCH in json.loads(thin["failed_floors"])
    assert thin["verdict"] != "PASS"
    # the mismatch is the *only* difference between the two rows' floors
    assert set(json.loads(thin["failed_floors"])) - set(json.loads(ok["failed_floors"])) == {MISMATCH}

    fat = pooled_rows(_frame(_flat(0.20)), "2025-09-25T00:00:00", baseline=[_baseline_row(0.10, 9000)])[0]
    assert MISMATCH in json.loads(fat["failed_floors"]) and fat["verdict"] != "PASS"


def test_report_pooled_table_renders_baseline_n():
    from research.strategy_hunt.report import _pooled_baseline_table
    rows = [{"strategy": "orb", "side": "long", "regime": "risk_on", "n": 600, "exp_r": 0.2,
             "baseline_exp_r": 0.1, "baseline_n": 50, "excess_r": 0.1, "t_stat_session": 4.0,
             "recovery": 5.0, "verdict": "WATCH", "failed_floors": json.dumps([MISMATCH])}]
    md = _pooled_baseline_table(rows)
    header = md.splitlines()[0]
    assert "baseline n" in header
    assert header.index("baseline n") > header.index("baseline exp_R")
    assert "| 50 |" in md and MISMATCH in md


def test_report_manifest_counts_population_mismatches():
    from research.strategy_hunt.report import _report_manifest
    pooled = [{"verdict": "WATCH", "baseline_n": 50, "failed_floors": json.dumps([MISMATCH])},
              {"verdict": "PASS", "baseline_n": 600, "failed_floors": json.dumps([])},
              {"verdict": "FAIL", "baseline_n": 0, "failed_floors": json.dumps(["no_baseline"])},
              {"verdict": "WATCH", "baseline_n": 9000, "failed_floors": "not-json"}]
    man = _report_manifest([], pooled, "2026-09-25T14:33:00", "x.md")
    assert man["pooled_baseline_population_mismatch"] == 1
    assert man["pooled_with_baseline"] == 3
