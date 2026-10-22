"""v-learning-loop-2026-10-22: shared Stage A floors; n AND sessions both required."""
from core.learning import stage_a


def _ok(**over):
    kw = dict(n_trades=200, n_sessions=20, profit_factor=1.5, win_rate=0.5, expectancy_r=0.1,
              max_dd_pct=0.03, max_losing_day_r=1.0)
    kw.update(over)
    return stage_a.evaluate(**kw)


def test_all_floors_pass():
    assert _ok().verdict == stage_a.PASS


def test_sample_gate_requires_both_n_and_sessions():
    assert not stage_a.sample_gate(200, 9)
    assert not stage_a.sample_gate(149, 50)
    assert stage_a.sample_gate(150, 10)
    assert _ok(n_sessions=9).verdict == stage_a.FAIL
    assert _ok(n_trades=12).verdict == stage_a.FAIL


def test_each_floor_fails_on_its_own():
    for over in (dict(profit_factor=1.29), dict(win_rate=0.47), dict(expectancy_r=0.049),
                 dict(max_dd_pct=0.061), dict(max_losing_day_r=2.01), dict(max_losing_day_r=-2.5)):
        r = _ok(**over)
        assert r.verdict == stage_a.FAIL and len(r.failures) == 1, over


def test_values_are_research_locked():
    assert (stage_a.MIN_TRADES, stage_a.MIN_SESSIONS, stage_a.MIN_PF, stage_a.MIN_WR,
            stage_a.MIN_EXPECTANCY_R, stage_a.MAX_DD_PCT, stage_a.MAX_LOSING_DAY_R) == (150, 10, 1.30, 0.48, 0.05, 0.06, 2.0)


def test_resolvers_share_the_constants_and_the_and_gate():
    import inspect
    import research.day_trade_short_resolver as d
    import research.shadow_long_resolver as l
    import research.shadow_short_resolver as s
    for m in (s, d, l):
        assert m.STAGE_A_MIN_TRADES == 150 and m.STAGE_A_MIN_SESSIONS == 10
        src = inspect.getsource(m)
        assert "n_trades >= STAGE_A_MIN_TRADES or n_sessions" not in src
        assert "_stage_a.sample_gate(n_trades, n_sessions)" in src
