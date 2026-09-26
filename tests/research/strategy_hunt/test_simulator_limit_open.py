"""Batch-2 Task B: the `limit_open` execution style and the pin that `market` did not move.

The limit is placed at the fill bar's open minus (long) / plus (short) `LIMIT_OFFSET_BPS`; it fills
at that price only if the fill bar trades through it, and an unfilled limit produces no trade at all.
"""
import pandas as pd
import pytest

from research.strategy_hunt.sim.costs import ZERO_COSTS, CostModel
from strategies.families.models import Entry

OFFSET = 0.0010


def _bars(opens, highs, lows, closes, day="2025-01-02", volume=10_000):
    idx = pd.date_range(f"{day} 09:30", periods=len(opens), freq="5min", name="ts")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": [volume] * len(opens)}, index=idx)


def _long(bars, style="limit_open", stop=98.0, target=104.0, hold=20, kind="intraday"):
    return Entry("X", "long", bars.index[0], 100.0, stop, target, hold, kind, "t", style)


def _short(bars, style="limit_open", stop=102.0, target=96.0, hold=20, kind="intraday"):
    return Entry("X", "short", bars.index[0], 100.0, stop, target, hold, kind, "t", style)


# ---- market is unchanged ----------------------------------------------------------------------

MARKET_PIN = {
    "symbol": "X", "family": "fam", "side": "long",
    "entry_ts": pd.Timestamp("2025-01-02 09:30:00"), "exit_ts": pd.Timestamp("2025-01-02 09:40:00"),
    "entry_px": 100.0, "stop_px": 98.0, "target_px": 104.0, "exit_px": 104.0,
    "entry_fill": 100.11000000000001, "exit_fill": 103.88253616,
    "r_multiple": 1.8862680799999936, "pnl_pct": 3.768390929977012, "cost_pct": 0.22746384000001282,
    "exit_reason": "target", "hold_bars": 2, "hold_sessions": 1, "tag": "pin", "regime": "risk_on",
    "window_id": 7, "params_json": '{"a":1}',
}


def test_market_style_trade_is_byte_identical_to_the_pre_batch2_simulator():
    """Every number a `market` trade carried before `entry_style` existed, pinned exactly (not
    approximately): the whole Batch-2 limit experiment is a *comparison* against this lane, so a
    market fill that drifted by a basis point would invalidate the delta the report reads."""
    from research.strategy_hunt.sim.simulator import simulate
    bars = _bars([100, 100, 101], [100.5, 100.5, 104.5], [99.5, 99.5, 100.5], [100, 100, 104])
    cm = CostModel(half_spread_by_month=pd.Series([0.001], index=pd.PeriodIndex(["2025-01"], freq="M")),
                   notional=6000.0)
    e = Entry("X", "long", bars.index[0], 100.0, 98.0, 104.0, 20, "intraday", "pin")
    t = simulate([e], bars, cm, set(), "fam", window_id=7, params_json='{"a":1}',
                 regime_lookup=lambda ts: "risk_on")[0]
    d = t.to_dict()
    # the two Batch-2 provenance columns are new *records*, not new numbers: they say what was
    # asked for (Task B's style, Task F's horizon), and are asserted separately from the pin so
    # this test keeps meaning "no computed value moved".
    assert d["entry_style"] == "market" and d["time_stop"] == 20
    assert {k: v for k, v in d.items() if k not in ("entry_style", "time_stop")} == MARKET_PIN


# ---- limit_open fills ------------------------------------------------------------------------

def test_long_limit_fills_at_the_limit_price_when_the_bar_trades_through_it():
    from research.strategy_hunt.sim.simulator import LIMIT_OFFSET_BPS, simulate
    assert LIMIT_OFFSET_BPS == 10
    bars = _bars([100, 100, 101], [100.5, 100.5, 104.5], [99.8, 99.5, 100.5], [100, 100, 104])
    t = simulate([_long(bars)], bars, ZERO_COSTS, set(), "fam")[0]
    assert t.entry_style == "limit_open"
    assert t.entry_fill == pytest.approx(100.0 * (1 - OFFSET))      # better than the open by 10 bps
    assert t.entry_px == 100.0 and t.exit_reason == "target" and t.exit_fill == 104.0
    assert t.r_multiple == pytest.approx((104.0 - 100.0 * (1 - OFFSET)) / 2.0)


def test_long_limit_that_the_bar_never_reaches_produces_no_trade():
    from research.strategy_hunt.sim.simulator import simulate
    bars = _bars([100, 100, 101], [100.5, 100.5, 104.5], [99.95, 99.5, 100.5], [100, 100, 104])
    assert simulate([_long(bars)], bars, ZERO_COSTS, set(), "fam") == []
    # ... while the same entry as a market order still trades.
    assert len(simulate([_long(bars, style="market")], bars, ZERO_COSTS, set(), "fam")) == 1


def test_short_limit_mirrors_the_long_rule_on_the_high():
    from research.strategy_hunt.sim.simulator import simulate
    hit = _bars([100, 100, 100], [100.2, 100.5, 100.5], [99.8, 99.5, 95.5], [100, 100, 96])
    t = simulate([_short(hit)], hit, ZERO_COSTS, set(), "fam")[0]
    assert t.entry_fill == pytest.approx(100.0 * (1 + OFFSET)) and t.exit_reason == "target"
    miss = _bars([100, 100, 100], [100.05, 100.5, 100.5], [99.8, 99.5, 95.5], [100, 100, 96])
    assert simulate([_short(miss)], miss, ZERO_COSTS, set(), "fam") == []


def test_limit_fill_is_exact_at_the_touch():
    """low == limit is a fill: the bar printed the price. `<`-only would silently drop the fills
    that matter most (a limit resting exactly at the session's low)."""
    from research.strategy_hunt.sim.simulator import simulate
    bars = _bars([100, 100, 101], [100.5, 100.5, 104.5], [100.0 * (1 - OFFSET), 99.5, 100.5], [100, 100, 104])
    assert len(simulate([_long(bars)], bars, ZERO_COSTS, set(), "fam")) == 1


def test_limit_uses_the_fill_bars_own_open_not_the_entry_price():
    """The rule is written against the bar, so an Entry whose `entry_px` drifted from the bar's
    open (a stale frame, a rebuilt replay) still gets the limit the bar would have supported."""
    from research.strategy_hunt.sim.simulator import simulate
    bars = _bars([50, 50, 60], [50.2, 50.2, 60.2], [49.9, 49.5, 50.5], [50, 50, 60])
    e = Entry("X", "long", bars.index[0], 100.0, 98.0, 104.0, 20, "intraday", "t", "limit_open")
    t = simulate([e], bars, ZERO_COSTS, set(), "fam")[0]
    assert t.entry_fill == pytest.approx(50.0 * (1 - OFFSET))


def test_stop_scan_still_starts_after_the_fill_bar():
    """Exactly as for a market fill: the fill bar's own range is not available to the position, so a
    fill bar that pierces the stop does not stop the trade out on the bar it was opened on."""
    from research.strategy_hunt.sim.simulator import simulate
    bars = _bars([100, 100, 101], [100.5, 100.5, 104.5], [97.0, 99.5, 100.5], [100, 100, 104])
    lim = simulate([_long(bars)], bars, ZERO_COSTS, set(), "fam")[0]
    mkt = simulate([_long(bars, style="market")], bars, ZERO_COSTS, set(), "fam")[0]
    assert lim.exit_reason == mkt.exit_reason == "target"
    assert lim.exit_ts == mkt.exit_ts and lim.hold_bars == mkt.hold_bars


def test_limit_entry_pays_no_spread_or_slippage_but_the_exit_still_does():
    from research.strategy_hunt.sim.simulator import simulate
    from research.strategy_hunt.sim.costs import SELL_FEE_PCT
    bars = _bars([100, 100, 101], [100.5, 100.5, 104.5], [99.8, 99.5, 100.5], [100, 100, 104])
    cm = CostModel(half_spread_by_month=pd.Series([0.001], index=pd.PeriodIndex(["2025-01"], freq="M")),
                   notional=6000.0, slippage_override=0.002)
    t = simulate([_long(bars)], bars, cm, set(), "fam")[0]
    assert t.entry_fill == pytest.approx(100.0 * (1 - OFFSET))
    assert t.exit_fill == pytest.approx(104.0 * (1 - 0.001 - 0.002 - SELL_FEE_PCT))
    # The 10 bps the maker *gained* on the entry net against the exit's friction, so `cost_pct`
    # is the exit cost minus the entry rebate -- signed, not an absolute charge.
    exit_cost = 104.0 * (0.001 + 0.002 + SELL_FEE_PCT)
    assert t.cost_pct == pytest.approx((exit_cost - 100.0 * OFFSET) / 100.0 * 100.0)


def test_overnight_limit_entry_keeps_the_next_open_exit():
    """Nothing after the fill changes: the exit vocabulary, the scan window and the hold accounting
    are the market lane's."""
    from research.strategy_hunt.sim.simulator import simulate
    sess = [pd.DataFrame({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1000},
                         index=pd.date_range(f"{d} 09:30", periods=78, freq="5min", name="ts"))
            for d in ("2025-01-02", "2025-01-03")]
    bars = pd.concat(sess)
    e = Entry("X", "long", bars.index[77], 100.0, 97.0, 160.0, 1, "overnight", "on", "limit_open")
    t = simulate([e], bars, ZERO_COSTS, set(), "fam")[0]
    assert t.exit_reason == "next_open" and t.exit_ts == pd.Timestamp("2025-01-03 09:30")
    assert t.entry_fill == pytest.approx(100.0 * (1 - OFFSET)) and t.hold_sessions == 2
