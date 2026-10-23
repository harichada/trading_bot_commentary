"""v-swing-lane-2026-10-23: earnings provider tests. Recorded fixtures only — no network.
Fixtures were captured live 2026-10-23 from api.nasdaq.com and Yahoo quoteSummary."""
import asyncio
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from core.swing_lane import earnings_provider as EP

FIX = Path(__file__).parent / "fixtures" / "earnings"
ET = ZoneInfo("America/New_York")
TODAY = date(2026, 10, 23)
TEN = ["IBM", "NVDA", "MU", "PLTR", "TSLA", "RKLB", "IONQ", "BE", "META", "AMD"]
EMPTY_DAY = json.loads((FIX / "nasdaq_2026-10-05.json").read_text())   # real 'rows: null' shape


def _nasdaq_fixture_days():
    return {p.stem.split("_")[1]: json.loads(p.read_text()) for p in FIX.glob("nasdaq_*.json")}


class FakeNet:
    """Records every URL; serves fixtures. Any unexpected URL raises."""

    def __init__(self, nasdaq=True, yahoo=True, fail_days=(), crumb="abcDEF12345"):
        self.urls = []
        self.days = _nasdaq_fixture_days()
        self.nasdaq, self.yahoo, self.fail_days, self.crumb = nasdaq, yahoo, set(fail_days), crumb

    def text(self, url):
        self.urls.append(url)
        if url == EP.YAHOO_COOKIE_URL:
            return ""
        if url == EP.YAHOO_CRUMB_URL:
            if not self.yahoo:
                raise OSError("yahoo down")
            return self.crumb
        raise AssertionError(url)

    def json(self, url):
        self.urls.append(url)
        if url.startswith("https://api.nasdaq.com/"):
            if not self.nasdaq:
                raise OSError("nasdaq down")
            d = url.rsplit("=", 1)[1]
            if d in self.fail_days:
                return {"data": None, "status": {"rCode": 400}}
            return self.days.get(d, EMPTY_DAY)
        if url.startswith("https://query2.finance.yahoo.com/"):
            sym = url.split("/quoteSummary/")[1].split("?")[0]
            if "crumb=" + self.crumb not in url:
                return json.loads((FIX / "yahoo_invalid_crumb.json").read_text())
            return json.loads((FIX / f"yahoo_{sym}.json").read_text())
        raise AssertionError(url)


def _prov(tmp_path, kind, net, syms=TEN):
    return EP.EarningsProvider(kind, str(tmp_path), syms, fetch_json=net.json, fetch_text=net.text,
                               sleep=lambda s: None)


# ── parsers ──────────────────────────────────────────────────────────────
def test_parse_nasdaq_day_real_fixture():
    d = EP.parse_nasdaq_day(json.loads((FIX / "nasdaq_2026-10-21.json").read_text()))
    assert {"IBM", "TSLA", "META"} <= set(d)


def test_parse_nasdaq_empty_and_error():
    assert EP.parse_nasdaq_day(EMPTY_DAY) == {}
    assert EP.parse_nasdaq_day({"data": None, "status": {"rCode": 400}}) is None
    assert EP.parse_nasdaq_day("garbage") is None


def test_parse_nasdaq_timing_codes():
    d = EP.parse_nasdaq_day(json.loads((FIX / "nasdaq_2026-11-02.json").read_text()))
    assert set(d.values()) <= {"bmo", "amc", "unknown"}


def test_parse_yahoo_real_fixtures():
    want = {"IBM": "2026-10-21", "NVDA": "2026-11-17", "PLTR": "2026-11-02", "AMD": "2026-11-03",
            "IONQ": "2026-11-05", "RKLB": "2026-11-09", "MU": "2026-12-23"}
    for s, d in want.items():
        got = EP.parse_yahoo(json.loads((FIX / f"yahoo_{s}.json").read_text()))
        assert got[0][0] == d, s
    assert EP.parse_yahoo(json.loads((FIX / "yahoo_MU.json").read_text()))[0][1] is True  # estimate


def test_parse_yahoo_invalid_crumb_is_unknown():
    assert EP.parse_yahoo(json.loads((FIX / "yahoo_invalid_crumb.json").read_text())) is None


# ── flag ─────────────────────────────────────────────────────────────────
def test_flag_default_none(monkeypatch):
    from core.config import Config
    monkeypatch.delenv("SWING_EARNINGS_PROVIDER", raising=False)
    c = Config()

    class M:
        def get(self, k, d=None):
            return d
    c.manager = M()
    assert c.SWING_EARNINGS_PROVIDER == "none"
    monkeypatch.setenv("SWING_EARNINGS_PROVIDER", "AUTO")
    assert c.SWING_EARNINGS_PROVIDER == "auto"
    monkeypatch.setenv("SWING_EARNINGS_PROVIDER", "finnhub")   # unsupported => none
    assert c.SWING_EARNINGS_PROVIDER == "none"


def test_none_provider_never_fetches(tmp_path):
    net = FakeNet()
    p = _prov(tmp_path, "none", net)
    assert p.refresh(TODAY) is False and net.urls == []
    assert p.dates("IBM", TODAY) is None


# ── refresh / cache / fail-closed ────────────────────────────────────────
def test_nasdaq_refresh_and_dates(tmp_path):
    net = FakeNet()
    p = _prov(tmp_path, "nasdaq", net)
    assert p.refresh(TODAY)
    assert all(u.startswith("https://api.nasdaq.com/") for u in net.urls)
    assert p.dates("IBM", TODAY) == {date(2026, 10, 21)}
    assert p.dates("PLTR", TODAY) == {date(2026, 11, 2)}
    assert p.dates("ZZZZ", TODAY) == set()       # full-market coverage: no report in window
    # daily refresh: second call the same day is a no-op
    n = len(net.urls)
    assert p.refresh(TODAY) is False and len(net.urls) == n


def test_nasdaq_partial_window_is_unusable(tmp_path):
    p = _prov(tmp_path, "nasdaq", FakeNet(fail_days={"2026-10-29"}))
    assert p.refresh(TODAY) is False
    assert p.dates("IBM", TODAY) is None          # fail closed


def test_auto_unions_sources(tmp_path):
    p = _prov(tmp_path, "auto", FakeNet())
    assert p.refresh(TODAY)
    # Nasdaq (Zacks) says 11-18, NVDA confirmed 11-17 (Yahoo): both kept => wider blackout
    assert p.dates("NVDA", TODAY) == {date(2026, 11, 17), date(2026, 11, 18)}
    c = json.loads((tmp_path / EP.CACHE_NAME).read_text())
    assert c["sources"] == {"nasdaq": True, "yahoo": True}


def test_auto_falls_back_to_yahoo(tmp_path):
    p = _prov(tmp_path, "auto", FakeNet(nasdaq=False))
    assert p.refresh(TODAY)
    assert p.dates("IONQ", TODAY) == {date(2026, 11, 5)}
    assert p.dates("ZZZZ", TODAY) is None          # yahoo covers only requested symbols


def test_yahoo_bad_crumb_fails_closed(tmp_path):
    net = FakeNet(nasdaq=False)
    net.crumb = "<html>"
    p = _prov(tmp_path, "auto", net)
    assert p.refresh(TODAY) is False
    assert p.dates("IBM", TODAY) is None


def test_stale_cache_fails_closed(tmp_path):
    p = _prov(tmp_path, "nasdaq", FakeNet())
    assert p.refresh(TODAY)
    assert p.dates("IBM", date(2026, 10, 26)) is not None   # weekend: 3 days ok
    assert p.dates("IBM", date(2026, 10, 27)) is None       # 4 days: unknown


def test_failed_refresh_keeps_previous_cache(tmp_path):
    assert _prov(tmp_path, "nasdaq", FakeNet()).refresh(TODAY)
    p2 = _prov(tmp_path, "nasdaq", FakeNet(nasdaq=False))
    assert p2.refresh(date(2026, 10, 26)) is False
    assert p2.dates("IBM", date(2026, 10, 26)) == {date(2026, 10, 21)}


def test_provider_mismatch_is_unknown(tmp_path):
    assert _prov(tmp_path, "nasdaq", FakeNet()).refresh(TODAY)
    assert _prov(tmp_path, "yahoo", FakeNet()).dates("IBM", TODAY) is None


def test_corrupt_cache_is_unknown(tmp_path):
    (tmp_path / EP.CACHE_NAME).write_text("{not json")
    assert _prov(tmp_path, "nasdaq", FakeNet()).dates("IBM", TODAY) is None


def test_blackout_with_provider_dates(tmp_path):
    from core.swing_lane import risk as R
    p = _prov(tmp_path, "auto", FakeNet())
    p.refresh(TODAY)
    blocked, why, nxt = R.earnings_check(date(2026, 10, 28), p.dates("PLTR", TODAY), True)
    assert blocked and why == "earnings_blackout" and nxt == date(2026, 11, 2)
    blocked, why, _ = R.earnings_check(TODAY, p.dates("ZZZZ", TODAY), True)
    assert not blocked
    blocked, why, _ = R.earnings_check(TODAY, _prov(tmp_path, "yahoo", FakeNet()).dates("IBM", TODAY), True)
    assert blocked and why == "earnings_data_missing"


def test_http_allowlist():
    from core.swing_lane.earnings_http import Http, HostNotAllowed
    for bad in ("https://api.schwabapi.com/x", "http://api.nasdaq.com/x", "https://paper-api.alpaca.markets/v2"):
        with pytest.raises(HostNotAllowed):
            Http().get_text(bad)


def test_make_loader_none_keeps_legacy():
    from core.swing_lane.config import LaneSettings
    assert EP.make_loader(LaneSettings()) is None


# ── engine scheduler loop ────────────────────────────────────────────────
class _Cfg:
    def __init__(self, on):
        self.ENABLE_SWING_SHADOW = on
        self.HANDS_OFF_DENYLIST = ["HQGE", "SPCX"]


class _Eng:
    is_running = True
    trade_history = []

    def _symbols_with_open_positions(self):
        return ["AAPL"]


def _loop(monkeypatch, on, when):
    from core import swing_lane_gate as gate
    from core.loops.swing_shadow_loop import SwingShadowLoop
    calls = []
    monkeypatch.setattr(gate, "maybe_record_fills", lambda c, d: calls.append(("fills", d)) or [])
    monkeypatch.setattr(gate, "maybe_refresh_earnings", lambda c, d: calls.append(("earn", d)) or True)
    monkeypatch.setattr(gate, "maybe_run_eod", lambda c, n, **kw: calls.append(("eod", n, kw)) or [])
    lp = SwingShadowLoop(_Eng(), cfg_factory=lambda: _Cfg(on), clock=lambda: when[0])
    return lp, calls


def test_loop_flag_off_does_nothing(monkeypatch):
    when = [datetime(2026, 10, 23, 16, 25, tzinfo=ET)]
    lp, calls = _loop(monkeypatch, False, when)
    assert asyncio.run(lp.tick()) == [] and calls == []


def test_loop_schedule(monkeypatch):
    when = [datetime(2026, 10, 23, 9, 0, tzinfo=ET)]
    lp, calls = _loop(monkeypatch, True, when)
    assert asyncio.run(lp.tick()) == []                       # pre-open
    when[0] = datetime(2026, 10, 23, 9, 40, tzinfo=ET)
    assert asyncio.run(lp.tick()) == ["fills"]
    assert asyncio.run(lp.tick()) == []                       # once per day
    when[0] = datetime(2026, 10, 23, 16, 10, tzinfo=ET)
    assert asyncio.run(lp.tick()) == ["earnings"]
    when[0] = datetime(2026, 10, 23, 16, 21, tzinfo=ET)
    assert asyncio.run(lp.tick()) == ["eod"]
    assert asyncio.run(lp.tick()) == []
    eod = [c for c in calls if c[0] == "eod"][0]
    assert eod[2]["held_symbols"] == {"AAPL"} and eod[2]["realized_today"] == 0.0


def test_loop_late_start_refreshes_before_eod(monkeypatch):
    when = [datetime(2026, 10, 23, 16, 29, tzinfo=ET)]
    lp, calls = _loop(monkeypatch, True, when)
    assert asyncio.run(lp.tick()) == ["fills", "earnings", "eod"]
    when[0] = datetime(2026, 10, 23, 16, 31, tzinfo=ET)
    lp2, calls2 = _loop(monkeypatch, True, when)
    assert "eod" not in asyncio.run(lp2.tick())


def test_loop_weekend_noop(monkeypatch):
    lp, calls = _loop(monkeypatch, True, [datetime(2026, 10, 24, 16, 25, tzinfo=ET)])
    assert asyncio.run(lp.tick()) == [] and calls == []


def test_engine_registers_loop_only_behind_flag():
    src = (Path(__file__).resolve().parents[2] / "core" / "engine.py").read_text()
    i = src.index("from core.loops.swing_shadow_loop import SwingShadowLoop")
    assert "if Config().ENABLE_SWING_SHADOW is True:" in src[i - 200:i]
    assert "swing_lane" not in src   # engine never names the lane package directly


def test_gate_refresh_off_path_imports_nothing():
    import os
    import subprocess
    import sys
    root = Path(__file__).resolve().parents[2]
    code = ("import sys\nfrom core.config import Config\nimport core.swing_lane_gate as g\n"
            "assert g.maybe_refresh_earnings(Config(), None) is False\n"
            "bad=[m for m in sys.modules if m.startswith('core.swing_lane') and m!='core.swing_lane_gate'"
            " or m=='core.loops.swing_shadow_loop']\nassert not bad, bad\nprint('OK')")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ENABLE_SWING", "SWING_"))}
    r = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-2000:]
