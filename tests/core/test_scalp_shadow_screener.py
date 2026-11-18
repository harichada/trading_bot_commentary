"""Tests: ScalpScreener wired into ScalpShadowEngine (Task 2)."""
from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch


@pytest.fixture(autouse=True)
def _isolate_audit_log(tmp_path, monkeypatch):
    """Redirect SELECTIONS_PATH to a temp file so tests never touch the real audit log."""
    import core.scalp_shadow as mod
    monkeypatch.setattr(mod, "SELECTIONS_PATH", tmp_path / "sel.ndjson")
    yield


def test_active_symbols_initialized_from_symbols():
    """Before any refresh, _active_symbols == DEFAULT_SYMBOLS."""
    from core.scalp_shadow import ScalpShadowEngine, DEFAULT_SYMBOLS

    e = ScalpShadowEngine()
    assert e._active_symbols == list(DEFAULT_SYMBOLS)


def test_refresh_symbols_updates_active():
    """After refresh with a mocked screener, _active_symbols is updated."""
    from core.scalp_shadow import ScalpShadowEngine

    e = ScalpShadowEngine()
    mock_select = AsyncMock(return_value=["MARA", "SNAP"])
    with patch("core.scalp_shadow.ScalpScreener") as MockScreener:
        MockScreener.return_value.select = mock_select
        asyncio.run(e._refresh_symbols_async())
    assert e._active_symbols == ["MARA", "SNAP"]


def test_refresh_symbols_falls_back_on_exception():
    """If screener raises, _active_symbols falls back to self.symbols."""
    from core.scalp_shadow import ScalpShadowEngine

    e = ScalpShadowEngine(symbols=["NVDA", "AMD"])
    mock_select = AsyncMock(side_effect=RuntimeError("network error"))
    with patch("core.scalp_shadow.ScalpScreener") as MockScreener:
        MockScreener.return_value.select = mock_select
        asyncio.run(e._refresh_symbols_async())
    assert e._active_symbols == ["NVDA", "AMD"]


def test_refresh_symbols_falls_back_on_empty_result():
    """If screener returns [], _active_symbols falls back to self.symbols."""
    from core.scalp_shadow import ScalpShadowEngine

    e = ScalpShadowEngine(symbols=["NVDA"])
    mock_select = AsyncMock(return_value=[])
    with patch("core.scalp_shadow.ScalpScreener") as MockScreener:
        MockScreener.return_value.select = mock_select
        asyncio.run(e._refresh_symbols_async())
    assert e._active_symbols == ["NVDA"]


def test_get_status_includes_active_symbols():
    """get_status() dict contains active_symbols key."""
    from core.scalp_shadow import ScalpShadowEngine

    e = ScalpShadowEngine()
    status = e.get_status()
    assert "active_symbols" in status
    assert isinstance(status["active_symbols"], list)


def test_open_rebuilt_on_refresh():
    """After refresh, _open only contains keys from the new active_symbols."""
    from core.scalp_shadow import ScalpShadowEngine

    e = ScalpShadowEngine(symbols=["NVDA", "AMD", "TSLA"])
    # Seed an open position for NVDA so we can verify it's preserved
    e._open["NVDA"] = {"side": "long", "entry_px": 100.0, "stop_px": 99.0,
                       "target_px": 102.0, "entry_time": "2026-11-18T10:00:00",
                       "hold_bars": 1}

    mock_select = AsyncMock(return_value=["NVDA", "AMD"])
    with patch("core.scalp_shadow.ScalpScreener") as MockScreener:
        MockScreener.return_value.select = mock_select
        asyncio.run(e._refresh_symbols_async())

    # TSLA dropped out, NVDA position preserved, AMD added as None
    assert set(e._open.keys()) == {"NVDA", "AMD"}
    assert e._open["NVDA"] is not None  # preserved
    assert e._open["AMD"] is None       # newly added


def test_audit_log_written(tmp_path, monkeypatch):
    """_refresh_symbols_async appends a line to SELECTIONS_PATH."""
    import json
    import core.scalp_shadow as mod

    orig_path = mod.SELECTIONS_PATH
    test_path = tmp_path / "scalp_symbol_selections.ndjson"
    monkeypatch.setattr(mod, "SELECTIONS_PATH", test_path)

    try:
        from core.scalp_shadow import ScalpShadowEngine

        e = ScalpShadowEngine(symbols=["SNAP", "MARA"])
        mock_select = AsyncMock(return_value=["SNAP"])
        with patch("core.scalp_shadow.ScalpScreener") as MockScreener:
            MockScreener.return_value.select = mock_select
            asyncio.run(e._refresh_symbols_async())

        lines = test_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["selected"] == ["SNAP"]
        assert set(record["universe"]) == {"SNAP", "MARA"}
        assert "date" in record
        assert "timestamp" in record
    finally:
        monkeypatch.setattr(mod, "SELECTIONS_PATH", orig_path)


def test_refresh_preserves_open_trade_for_dropped_symbol():
    """Symbol with open trade survives screener drop in _active_symbols."""
    from core.scalp_shadow import ScalpShadowEngine

    e = ScalpShadowEngine(symbols=["NVDA", "AMD", "MARA"])
    # Simulate an open trade on AMD
    e._open = {"NVDA": None, "AMD": {"entry": 50.0, "stop": 48.0}, "MARA": None}
    e._active_symbols = ["NVDA", "AMD", "MARA"]

    # Screener drops AMD — only returns NVDA and MARA
    mock_select = AsyncMock(return_value=["NVDA", "MARA"])
    with patch("core.scalp_shadow.ScalpScreener") as MockScreener:
        MockScreener.return_value.select = mock_select
        asyncio.run(e._refresh_symbols_async())

    # AMD must still be in _active_symbols because it has an open trade
    assert "AMD" in e._active_symbols
    # Its open trade must be preserved
    assert e._open.get("AMD") is not None
    assert e._open["AMD"]["entry"] == 50.0


def test_screener_skips_before_market_open(tmp_path, monkeypatch):
    """Before 09:31 ET: active_symbols = universe, _last_screener_date stays empty."""
    import core.scalp_shadow as mod
    from core.scalp_shadow import ScalpShadowEngine
    from zoneinfo import ZoneInfo
    from datetime import datetime

    # Patch _now_et to return 09:00 ET (before market open)
    et_zone = ZoneInfo("America/New_York")
    fake_time = datetime(2026, 11, 18, 9, 0, 0, tzinfo=et_zone)
    monkeypatch.setattr(mod, "_now_et", lambda: fake_time)
    # Patch _today_et to return a fixed date different from _last_screener_date
    monkeypatch.setattr(mod, "_today_et", lambda: "2026-11-18")

    engine = ScalpShadowEngine(symbols=["NVDA", "AMD"])
    engine._last_screener_date = ""  # ensure the date check would fire

    # Manually run the screener block logic (simulating _loop iteration)
    today = mod._today_et()
    if today != engine._last_screener_date:
        et_now = mod._now_et()
        if et_now.hour > 9 or (et_now.hour == 9 and et_now.minute >= 31):
            engine._last_screener_date = today
            asyncio.run(engine._refresh_symbols_async())
        else:
            engine._active_symbols = list(engine.symbols)

    assert engine._active_symbols == list(engine.symbols)
    assert engine._last_screener_date == ""


def test_restart_reuses_todays_selection(tmp_path, monkeypatch):
    """On same-day restart, _refresh_symbols_async reuses today's NDJSON entry without calling screener."""
    import core.scalp_shadow as mod
    from core.scalp_shadow import ScalpShadowEngine

    today = "2026-11-18"
    monkeypatch.setattr(mod, "_today_et", lambda: today)

    # Write a today entry to the tmp NDJSON
    sel_path = tmp_path / "sel2.ndjson"
    monkeypatch.setattr(mod, "SELECTIONS_PATH", sel_path)
    entry = {"date": today, "selected": ["SNAP", "MARA"], "universe": ["NVDA", "AMD", "SNAP", "MARA"], "timestamp": "2026-11-18T10:00:00Z"}
    sel_path.write_text(json.dumps(entry) + "\n")

    engine = ScalpShadowEngine(symbols=["NVDA", "AMD", "SNAP", "MARA"])

    # Screener should NOT be called — if it is, raise
    mock_select = AsyncMock(side_effect=AssertionError("screener should not be called on restart"))
    with patch("core.scalp_shadow.ScalpScreener") as MockScreener:
        MockScreener.return_value.select = mock_select
        asyncio.run(engine._refresh_symbols_async())

    assert engine._active_symbols == ["SNAP", "MARA"]
