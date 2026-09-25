import pandas as pd
import pytest


class _Resp:
    def __init__(self, payload): self._p = payload
    def json(self): return self._p
    def raise_for_status(self): pass


class _Session:
    def __init__(self, pages): self.pages = list(pages); self.calls = []
    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(dict(params)); return _Resp(self.pages.pop(0))


def test_alpaca_bars_paginates_filters_rth_and_converts_et():
    from research.strategy_hunt.data.context import alpaca_bars
    sess = _Session([
        {"bars": {"SPY": [{"t": "2025-06-02T13:30:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
                          {"t": "2025-06-02T13:25:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10}]}, "next_page_token": "tok"},
        {"bars": {"SPY": [{"t": "2025-06-02T19:55:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10},
                          {"t": "2025-06-02T20:00:00Z", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10}]}, "next_page_token": None},
    ])
    df = alpaca_bars("SPY", pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-03"), sess, "k", "s")
    assert list(df.index) == [pd.Timestamp("2025-06-02 09:30"), pd.Timestamp("2025-06-02 15:55")]   # EDT = UTC-4
    assert sess.calls[1]["page_token"] == "tok" and df.index.name == "ts" and list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_ensure_spy_appends_after_last_ts(tmp_path, monkeypatch):
    from research.strategy_hunt.data import context as C
    from research.strategy_hunt.data.bars import _write, bars5_path, load_bars5, load_daily
    idx = pd.date_range("2025-06-02 09:30", periods=78, freq="5min", name="ts")
    _write(bars5_path(tmp_path, "SPY"), pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}, index=idx), "ts")
    seen = {}

    def fake_alpaca(symbol, start, end, session, key, secret, timeframe="5Min", feed="sip"):
        seen["start"] = start
        idx2 = pd.date_range("2025-06-03 09:30", periods=78, freq="5min", name="ts")
        return pd.DataFrame({"open": 2.0, "high": 2.0, "low": 2.0, "close": 2.0, "volume": 1}, index=idx2)

    monkeypatch.setattr(C, "alpaca_bars", fake_alpaca)
    n = C.ensure_spy(tmp_path, object(), "k", "s", now=pd.Timestamp("2025-06-04"))
    assert n == 78 and seen["start"] == pd.Timestamp("2025-06-02 16:00")
    assert len(load_bars5(tmp_path, "SPY")) == 156 and len(load_daily(tmp_path, "SPY")) == 2


def test_ensure_vix_and_context_never_raise(tmp_path):
    from research.strategy_hunt.data import context as C
    from research.strategy_hunt.data.bars import load_daily
    good = pd.DataFrame({"open": 20.0, "high": 21.0, "low": 19.0, "close": 20.5, "volume": 0},
                        index=pd.DatetimeIndex(pd.bdate_range("2025-06-02", periods=3), name="date"))
    assert C.ensure_vix(tmp_path, fetcher=lambda s: good) == 3 and len(load_daily(tmp_path, "VIX")) == 3

    def boom(symbol): raise RuntimeError("network")
    out = C.ensure_context(tmp_path, None, "", "", vix_fetcher=boom)
    assert out["vix"] == -1 and out["spy"] == -1        # no session/keys -> spy fails too; reported, not raised
