"""Tests: ScalpScreener wired into ScalpShadowEngine (Task 2)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch


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
