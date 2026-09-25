"""Shared builders for the playbook-updater tests.

Not a test module (no `test_` prefix, so pytest never collects it): fix r2 finding "split the
oversized test modules" moved these here so `test_playbook_update.py` (converters and readers)
and `test_playbook_nightly.py` (verdicts and the nightly rewrite) can share one set of fixtures
without either file passing 400 lines.
"""
import json

import numpy as np
import pandas as pd
import pytest

from research.strategy_hunt.playbook_update import LIVE_REGIME_SQL, LIVE_TRADES_SQL

NOW = "2026-09-25T20:00:00"


def _bars(start="2026-07-01", sessions=30, seed=0):
    rng = np.random.default_rng(seed)
    frames, px = [], 100.0
    for d in pd.bdate_range(start, periods=sessions):
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", name="ts")
        close = px + rng.normal(0, 0.05, 78).cumsum()
        frames.append(pd.DataFrame({"open": close - 0.02, "high": close + 0.3, "low": close - 0.3,
                                    "close": close, "volume": 1000}, index=idx))
        px = float(close[-1])
    return pd.concat(frames)


@pytest.fixture
def root(tmp_path):
    from research.strategy_hunt.data.bars import _write, bars5_path, daily_path, to_daily

    b = _bars()
    _write(bars5_path(tmp_path, "AAA"), b, "ts")
    _write(daily_path(tmp_path, "AAA"), to_daily(b), "date")
    return tmp_path


def _ledger(ledger_dir, family="toy", timestamps=("2026-07-06T10:00:00", "2026-07-09T10:00:00")):
    from research.strategy_hunt.shadow_resolver import default_ledger_path

    p = default_ledger_path(family, ledger_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps({
        "ts": ts, "symbol": "AAA", "side": "long", "entry_px": 100.0, "stop_px": 99.0,
        "target_px": 100.5, "time_stop": 20, "hold_kind": "intraday", "tag": "t",
        "params": {"w": 1.0}, "verdict": "PASS", "score": 0.5,
    }) for ts in timestamps) + "\n")
    return p


def _sim_trade(**kw):
    from research.strategy_hunt.models import SimTrade

    base = dict(symbol="AAA", family="orb", side="long", entry_ts=pd.Timestamp("2026-09-01 10:00"),
                exit_ts=pd.Timestamp("2026-09-01 11:00"), entry_px=100.0, stop_px=99.0, target_px=102.0,
                exit_px=102.0, entry_fill=100.0, exit_fill=102.0, r_multiple=2.0, pnl_pct=0.02,
                cost_pct=0.0005, exit_reason="target", hold_bars=12, hold_sessions=1, tag="t",
                regime="risk_on", window_id=-1, params_json="{}")
    base.update(kw)
    return SimTrade(**base)


def _live_frames(rows, decisions=()):
    trades = pd.DataFrame(list(rows), columns=["symbol", "side", "strategy", "entry_time", "exit_time", "pnl_r"])
    dec = pd.DataFrame(list(decisions), columns=["symbol", "ts", "details_json"])
    return {LIVE_TRADES_SQL: trades, LIVE_REGIME_SQL: dec}


def _reader(frames, seen=None):
    """A `reader(query, params)` over pre-built frames (the injected SQL reader contract)."""
    def _read(query, params=None):
        if seen is not None:
            seen.append((query, params))
        return frames[query]
    return _read


def _snapshot(pooled=None, regime=None, rows=None):
    return {"computed_at": NOW, "floors": {}, "rows": rows or {}, "regime_rows": regime or {},
            "pooled_rows": pooled or {}, "baseline_pooled_rows": {}}


def _live_row(day_offset, r, strategy="orb_contraction_rvol", hour=10, symbol="AAA", side="long"):
    """One bot_trades row `day_offset` days before NOW (entry and exit on the same day)."""
    day = (pd.Timestamp(NOW) - pd.Timedelta(days=day_offset)).normalize()
    return (symbol, side, strategy, (day + pd.Timedelta(hours=hour - 1)).isoformat(),
            (day + pd.Timedelta(hours=hour)).isoformat(), r)


def _seed(desk, *entries):
    from core.desk.playbook import Playbook, playbook_path, save_playbook

    pb = Playbook(entries={}, updated_at="2026-06-01T00:00:00")
    for e in entries:
        pb = pb.with_entry(e)
    save_playbook(pb, playbook_path(desk))


def _prev_entry(**kw):
    from core.desk.playbook import PlaybookEntry

    base = dict(family="news_catalyst", side="short", regime="mixed", status="suspended",
                size_mult=0.0, exp_r_recent=-0.4, t_stat_recent=-2.5, n_recent=20,
                updated_at="2026-06-01T00:00:00", reason="suspended_negative_expectancy")
    base.update(kw)
    return PlaybookEntry(**base)



# ── entry-time regime (fix r2, finding 1a) ──

def spy_bars(sessions=("2026-07-06", "2026-07-07", "2026-07-08", "2026-07-09"),
             morning_pct=0.5, close_pct=-1.0):
    """SPY 5-minute bars whose sessions are up `morning_pct` by 09:55 and end `close_pct`.

    The two must disagree: a label taken from the *close* would read risk_off, so a test that
    asserts risk_on proves the label came from bars closed before the entry.
    """
    frames = []
    for day in sessions:
        idx = pd.date_range(f"{day} 09:30", periods=78, freq="5min", name="ts")
        close = np.empty(78, dtype=float)
        close[:5] = 100.0 * (1.0 + morning_pct / 100.0)      # 09:30..09:50 already up
        close[5:] = np.linspace(100.0 * (1.0 + morning_pct / 100.0),
                                100.0 * (1.0 + close_pct / 100.0), 73)
        frames.append(pd.DataFrame({"open": np.r_[100.0, close[:-1]], "high": close + 0.2,
                                    "low": close - 0.2, "close": close, "volume": 1000}, index=idx))
    return pd.concat(frames)


def vix_daily(dates=("2026-07-01", "2026-07-02", "2026-07-03", "2026-07-06", "2026-07-07",
                     "2026-07-08", "2026-07-09"), level=15.0):
    """Flat daily VIX below 16 — with SPY up 0.5% that classifies risk_on."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates], name="date")
    return pd.DataFrame({"open": level, "high": level, "low": level, "close": level,
                         "volume": 0}, index=idx)


def write_regime_context(root, spy=None, vix=None):
    """Write SPY 5-minute/daily bars and daily VIX into `root`, the way `load_context` reads them."""
    from research.strategy_hunt.data.bars import _write, bars5_path, daily_path, to_daily

    spy = spy_bars() if spy is None else spy
    _write(bars5_path(root, "SPY"), spy, "ts")
    _write(daily_path(root, "SPY"), to_daily(spy), "date")
    _write(daily_path(root, "VIX"), vix_daily() if vix is None else vix, "date")
    return root
