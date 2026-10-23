"""v-swing-lane-2026-10-23 tests: flags off, off-path no-op, denylist, earnings fail-closed,
scan window, setup B logic, no order path, shadow log schema."""
import ast
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "core" / "swing_lane"
ET = ZoneInfo("America/New_York")
FLAGS = ("ENABLE_SWING_SHADOW", "ENABLE_SWING_LIVE", "SWING_SETUP_PULLBACK",
         "SWING_SETUP_BREAKOUT", "SWING_SETUP_GAPDRIFT", "SWING_SETUP_MEANREV")


class _M:
    def __init__(self, d=None):
        self.d = d or {}

    def get(self, k, default=None):
        return self.d.get(k, default)


def _cfg(d=None):
    from core.config import Config
    c = Config()
    c.manager = _M(d)
    return c


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for f in FLAGS + ("SWING_EARNINGS_BLACKOUT",):
        monkeypatch.delenv(f, raising=False)


# ── flags ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("flag", FLAGS)
def test_flag_default_off(flag):
    assert getattr(_cfg(), flag) is False


@pytest.mark.parametrize("flag", FLAGS)
def test_flag_env_on(monkeypatch, flag):
    monkeypatch.setenv(flag, "1")
    assert getattr(_cfg(), flag) is True


def test_flag_yaml_on_and_blackout_default():
    assert _cfg({"trading.enable_swing_shadow": True}).ENABLE_SWING_SHADOW is True
    assert _cfg().SWING_EARNINGS_BLACKOUT is True


def test_real_repo_yaml_leaves_flags_off():
    from core.config import Config
    c = Config()
    for f in FLAGS:
        assert getattr(c, f) is False, f


# ── off-path ─────────────────────────────────────────────────────────────
def test_off_path_imports_nothing():
    code = ("import sys\nfrom core.config import Config\nimport core.swing_lane_gate as g\n"
            "c=Config()\nassert g.maybe_run_eod(c, None)==[]\nassert g.maybe_record_fills(c, None)==[]\n"
            "bad=[m for m in sys.modules if m.startswith('core.swing_lane.') or m=='core.swing_lane']\n"
            "assert not bad, bad\nprint('OK')")
    env = {k: v for k, v in __import__("os").environ.items() if k not in FLAGS}
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-2000:]


def test_engine_does_not_import_swing_lane():
    for p in [ROOT / "core" / "engine.py", ROOT / "trading_bot_commentary_updated.py"]:
        if p.exists():
            assert "swing_lane" not in p.read_text()


def test_live_flag_refuses(monkeypatch):
    from core.swing_lane_gate import SwingLiveNotSupported, check_startup, maybe_run_eod
    monkeypatch.setenv("ENABLE_SWING_LIVE", "1")
    with pytest.raises(SwingLiveNotSupported):
        check_startup(_cfg())
    monkeypatch.setenv("ENABLE_SWING_SHADOW", "1")
    with pytest.raises(SwingLiveNotSupported):
        maybe_run_eod(_cfg(), datetime(2026, 10, 22, 16, 25, tzinfo=ET))


# ── no order path ────────────────────────────────────────────────────────
BANNED_IMPORTS = ("data_providers.schwab", "schwab", "alpaca", "core.order_monitor", "core.engine",
                  "requests", "httpx", "aiohttp", "urllib", "core.provider_calls", "core.isolated_calls")
BANNED_NAMES = ("place_order", "submit_order", "cancel_order", "replace_order", "order_payload",
                "buy_market", "sell_market", "OrderBuilder", "execute_trade", "open_position")


def _py_files():
    return list(PKG.rglob("*.py")) + [ROOT / "core" / "swing_lane_gate.py"]


def test_no_order_or_broker_code():
    for p in _py_files():
        tree = ast.parse(p.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                mods = [n.module or ""]
            else:
                mods = []
            for m in mods:
                assert not any(m == b or m.startswith(b + ".") for b in BANNED_IMPORTS), (p, m)
            if isinstance(n, (ast.Name, ast.Attribute, ast.FunctionDef)):
                ident = getattr(n, "id", None) or getattr(n, "attr", None) or getattr(n, "name", None)
                assert ident not in BANNED_NAMES, (p, ident)


# ── synthetic data ───────────────────────────────────────────────────────
ASOF = pd.Timestamp("2026-10-22")


def _bars(n=300, breakout=True, seed=0, price=50.0):
    idx = pd.bdate_range(end=ASOF, periods=n)
    rng = np.random.default_rng(seed)
    close = price * np.cumprod(1 + 0.0015 + rng.normal(0, 0.004, n))  # gentle uptrend
    df = pd.DataFrame({"open": close * 0.998, "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": np.full(n, 2_000_000.0)}, index=idx)
    if breakout:
        hh = df.high.iloc[-21:-1].max()
        df.iloc[-1, df.columns.get_loc("close")] = hh * 1.02
        df.iloc[-1, df.columns.get_loc("high")] = hh * 1.03
        df.iloc[-1, df.columns.get_loc("open")] = hh * 0.995
        df.iloc[-1, df.columns.get_loc("volume")] = 4_000_000.0
    return df


P = json.loads((PKG / "best_params.json").read_text())


def test_setup_b_fires_on_breakout():
    from core.swing_lane import indicators
    from core.swing_lane.setups import breakout
    df = _bars()
    sigs = breakout.scan(df, indicators.compute(df), P["B"], sym="X")
    assert len(sigs) == 1
    s = sigs[0]
    atr = indicators.compute(df).atr.iat[-1]
    assert s.planned_stop == pytest.approx(s.close - 2 * atr)
    assert s.planned_target is None and s.score == pytest.approx(2.0, rel=0.05)


@pytest.mark.parametrize("mut", ["low_volume", "no_breakout", "below_sma50", "wide_base", "illiquid"])
def test_setup_b_rejects(mut):
    from core.swing_lane import indicators
    from core.swing_lane.setups import breakout
    df = _bars()
    if mut == "low_volume":
        df.iloc[-1, df.columns.get_loc("volume")] = 2_500_000.0
    elif mut == "no_breakout":
        df = _bars(breakout=False)
        df.iloc[-1, df.columns.get_loc("volume")] = 4_000_000.0
    elif mut == "below_sma50":
        df = _bars(); df["close"] = df["close"][::-1].values; df["high"] = df.close * 1.01; df["low"] = df.close * .99
        df.iloc[-1, df.columns.get_loc("volume")] = 4_000_000.0
    elif mut == "wide_base":
        p = dict(P["B"], k=0.5)
        assert breakout.scan(df, indicators.compute(df), p) == []
        return
    elif mut == "illiquid":
        df["volume"] = df["volume"] / 100
    assert breakout.scan(df, indicators.compute(df), P["B"]) == []


RESEARCH = Path("/workspace/review/swing_system")


@pytest.mark.skipif(not (RESEARCH / "swing_lib.py").exists(), reason="research artefacts not on this box")
def test_setup_b_parity_with_backtest_lib():
    sys.path.insert(0, str(RESEARCH))
    import swing_lib
    from core.swing_lane import indicators
    from core.swing_lane.setups import breakout
    checked = 0
    for sym in ("RKLB", "IONQ", "NVDA", "BE"):
        f = RESEARCH / "data" / f"{sym}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).set_index("date")[["open", "high", "low", "close", "volume"]].iloc[-700:]
        ref, _ = swing_lib.signals(df, swing_lib.ind(df), "B", P["B"])
        ind = indicators.compute(df)
        mine = [breakout.is_signal(df, ind, P["B"], i) for i in range(-400, 0)]
        assert mine == list(ref.iloc[-400:].astype(bool)), sym
        checked += 1
    assert checked


def test_scan_window():
    from core.swing_lane.scheduler import in_scan_window
    d = lambda h, m, day=22: datetime(2026, 10, day, h, m, tzinfo=ET)
    assert in_scan_window(d(16, 20)) and in_scan_window(d(16, 25)) and in_scan_window(d(16, 30))
    assert not in_scan_window(d(16, 19)) and not in_scan_window(d(16, 31)) and not in_scan_window(d(9, 30))
    assert not in_scan_window(d(16, 25, day=24))  # Saturday
    assert in_scan_window(datetime(2026, 10, 22, 20, 25, tzinfo=ZoneInfo("UTC")))  # 16:25 EDT


# ── end-to-end shadow scan ───────────────────────────────────────────────
def _settings(tmp_path, deny=frozenset({"HQGE", "SPCX"}), **kw):
    from core.swing_lane.config import LaneSettings
    s = LaneSettings(shadow=True, setups={"A": False, "B": True, "C": False, "D": False},
                     denylist=deny, log_dir=str(tmp_path))
    for k, v in kw.items():
        setattr(s, k, v)
    return s


NOW = datetime(2026, 10, 22, 16, 25, tzinfo=ET)


def _run(tmp_path, universe, earnings=None, settings=None, **kw):
    from core.swing_lane.scheduler import run_eod_scan
    bars = {"SPY": _bars(breakout=False, seed=9, price=500)}
    for i, s in enumerate(universe):
        bars[s] = _bars(seed=i)
    if earnings is None:
        earnings = {s: {date(2026, 12, 1)} for s in universe}
    return run_eod_scan(settings or _settings(tmp_path), NOW, bars_loader=bars.get,
                        earnings_loader=lambda s: earnings.get(s), universe=universe, params=P, **kw)


def test_denylist_from_config_excluded(tmp_path):
    from core.swing_lane.config import LaneSettings
    cfg = _cfg({"trading.hands_off_denylist": ["HQGE", "SPCX"]})
    ls = LaneSettings.from_config(cfg)
    assert ls.denylist == frozenset({"HQGE", "SPCX"})
    ls.shadow = True; ls.setups["B"] = True; ls.log_dir = str(tmp_path)
    rows = _run(tmp_path, ["HQGE", "SPCX", "AAA"], settings=ls)
    sig_syms = {r["sym"] for r in rows if r["event"] == "signal"}
    assert sig_syms == {"AAA"}
    assert {r["sym"] for r in rows if r["skipped_reason"] == "hands_off_denylist"} == {"HQGE", "SPCX"}


def test_denylist_not_hardcoded(tmp_path):
    rows = _run(tmp_path, ["HQGE"], settings=_settings(tmp_path, deny=frozenset()))
    assert [r["sym"] for r in rows if r["event"] == "signal"] == ["HQGE"]
    src = "".join(p.read_text() for p in _py_files())
    assert "HQGE" not in src and "SPCX" not in src


def test_earnings_missing_blocks_entry(tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO, logger="swing_lane")
    rows = _run(tmp_path, ["AAA", "BBB"], earnings={"BBB": {date(2026, 12, 1)}})
    by = {r["sym"]: r for r in rows if r["event"] == "signal"}
    assert by["AAA"]["skipped_reason"] == "earnings_data_missing" and by["AAA"]["blackout_skip"] is True
    assert by["BBB"]["skipped_reason"] is None
    assert "earnings_data_missing" in caplog.text


def test_earnings_blackout_within_5_days(tmp_path):
    rows = _run(tmp_path, ["AAA"], earnings={"AAA": {date(2026, 10, 27)}})
    r = [r for r in rows if r["event"] == "signal"][0]
    assert r["skipped_reason"] == "earnings_blackout" and r["earnings_next"] == "2026-10-27"


def test_outside_window_noop(tmp_path):
    from core.swing_lane.scheduler import run_eod_scan
    assert run_eod_scan(_settings(tmp_path), datetime(2026, 10, 22, 15, 0, tzinfo=ET)) == []
    assert not (tmp_path / "swing_shadow.ndjson").exists()


def test_disabled_setups_noop(tmp_path):
    s = _settings(tmp_path); s.setups = {k: False for k in "ABCD"}
    assert _run(tmp_path, ["AAA"], settings=s) == []


def test_caps_and_existing_position(tmp_path):
    rows = _run(tmp_path, ["S1", "S2", "S3", "S4", "S5", "S6"], held_symbols=["S6"])
    sig = [r for r in rows if r["event"] == "signal"]
    accepted = [r for r in sig if r["skipped_reason"] is None]
    assert len(accepted) == 4
    assert sum(r["planned_risk_usd"] for r in accepted) <= 600
    assert any(r["skipped_reason"] == "existing_position" and r["sym"] == "S6" for r in rows)


def test_daily_loss_blocks(tmp_path):
    rows = _run(tmp_path, ["AAA"], realized_today=-500.0)
    assert [r["skipped_reason"] for r in rows if r["event"] == "signal"] == ["daily_loss_limit"]


# ── shadow log schema ────────────────────────────────────────────────────
SPEC_FIELDS = ("ts, run_id, setup, sym, signal_date, regime_on, close, atr14, rvol, gap, rng20_atr, "
               "earnings_next, blackout_skip, rank, slot_available, planned_entry, planned_stop, "
               "planned_target, planned_shares, planned_risk_usd, fill_date, fill_price, slippage_bps, "
               "stop_updates, exit_date, exit_price, exit_reason, gap_through, pnl_usd, R, mae_R, mfe_R, "
               "hold_days, skipped_reason, code_version, params_hash").split(", ")


def test_shadow_schema_matches_spec(tmp_path):
    from core.swing_lane.shadow_log import FIELDS, make_row
    assert list(FIELDS) == SPEC_FIELDS
    _run(tmp_path, ["AAA"])
    lines = (tmp_path / "swing_shadow.ndjson").read_text().splitlines()
    assert lines
    for ln in lines:
        row = json.loads(ln)
        assert set(SPEC_FIELDS) <= set(row), set(SPEC_FIELDS) - set(row)
        assert row["code_version"] and row["params_hash"] and row["run_id"]
    with pytest.raises(KeyError):
        make_row(bogus=1)
    with pytest.raises(ValueError):
        make_row(exit_reason="vibes")


def test_fill_and_exit_lifecycle(tmp_path):
    from core.swing_lane import scheduler
    s = _settings(tmp_path)
    _run(tmp_path, ["AAA"])
    df = _bars(seed=0)
    nxt = pd.bdate_range(ASOF + pd.Timedelta(days=1), periods=12)
    last = df.close.iat[-1]
    fut = pd.DataFrame({"open": last, "high": last * 1.005, "low": last * 0.995, "close": last,
                        "volume": 2e6}, index=nxt)
    full = pd.concat([df, fut])
    fills = scheduler.record_fills(s, nxt[0].date(), bars_loader=lambda sym: full)
    assert len(fills) == 1 and fills[0]["fill_price"] == pytest.approx(last)
    exits = scheduler.update_exits(s, bars_loader=lambda sym: full, earnings_loader=lambda sym: set())
    assert len(exits) == 1 and exits[0]["exit_reason"] == "time" and exits[0]["hold_days"] == 10
    assert scheduler.update_exits(s, bars_loader=lambda sym: full, earnings_loader=lambda sym: set()) == []


def test_gate_isolates_exceptions(monkeypatch):
    import core.swing_lane_gate as g
    monkeypatch.setenv("ENABLE_SWING_SHADOW", "1")
    from core.swing_lane import scheduler
    monkeypatch.setattr(scheduler, "run_eod_scan", lambda *a, **k: 1 / 0)
    assert g.maybe_run_eod(_cfg(), NOW) == []
