"""Tests for core/symbol_intel.py — the per-symbol realtime
intelligence hub (operator request 2026-06-09/10: "analyse each symbol
in realtime — its direction, price movement, news sentiment,
indicators and everything which determines the trade quality").

One build, two consumers: the dashboard's Symbol Intelligence panel
(via WS dashboard_update) and the Composite View Phase 1 evidence
ledger (symbol_intel_log.ndjson)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = REPO_ROOT / "core" / "engine.py"
ROUTES_PATH = REPO_ROOT / "api" / "routes.py"

INDICATORS = {
    "rsi": 62.0, "adx": 31.0, "macd": 1.2, "macd_signal": 0.8,
    "macd_histogram": 0.4, "atr": 2.1, "volume_ratio": 1.8,
    "sma_20": 99.0, "sma_50": 95.0, "ema_20": 99.5,
    "bb_upper": 104.0, "bb_middle": 99.0, "bb_lower": 94.0,
    "ema_20_slope_pct": 0.6, "close_vs_sma50_pct": 5.3,
    "obv_slope_pct": 12.0,
}


class TestBuildSymbolView:
    def test_view_has_required_fields(self):
        from core.symbol_intel import build_symbol_view
        v = build_symbol_view("NVDA", 100.5, INDICATORS)
        for key in ("symbol", "price", "direction", "phase", "tape_er",
                    "rsi", "macd_hist", "volume_ratio", "atr_pct",
                    "quality", "updated_at"):
            assert key in v, f"missing {key}"
        assert v["symbol"] == "NVDA"
        assert v["price"] == 100.5

    def test_uptrend_scores_positive_direction_and_quality(self):
        from core.symbol_intel import build_symbol_view
        v = build_symbol_view("NVDA", 100.5, INDICATORS)
        assert v["direction"] > 0
        assert 0 <= v["quality"] <= 100

    def test_empty_indicators_safe(self):
        from core.symbol_intel import build_symbol_view
        v = build_symbol_view("XXXX", 10.0, {})
        assert v["quality"] >= 0  # no crash, degraded scores

    def test_tape_er_passthrough(self):
        from core.symbol_intel import build_symbol_view
        v = build_symbol_view("NVDA", 100.5, INDICATORS, tape_er=0.42)
        assert v["tape_er"] == 0.42


class TestSymbolIntelHub:
    def test_update_and_snapshot_sorted_by_quality(self, tmp_path):
        from core.symbol_intel import SymbolIntelHub, build_symbol_view
        hub = SymbolIntelHub(ledger_path=tmp_path / "intel.ndjson")
        weak = dict(INDICATORS, adx=10.0, volume_ratio=0.5,
                    ema_20_slope_pct=0.0, close_vs_sma50_pct=0.0,
                    obv_slope_pct=0.0, macd_histogram=0.0)
        hub.update(build_symbol_view("WEAK", 10.0, weak))
        hub.update(build_symbol_view("STRONG", 100.0, INDICATORS))
        snap = hub.snapshot()
        assert [s["symbol"] for s in snap][0] == "STRONG"
        assert len(snap) == 2

    def test_update_replaces_not_appends(self, tmp_path):
        from core.symbol_intel import SymbolIntelHub, build_symbol_view
        hub = SymbolIntelHub(ledger_path=tmp_path / "intel.ndjson")
        hub.update(build_symbol_view("NVDA", 100.0, INDICATORS))
        hub.update(build_symbol_view("NVDA", 101.0, INDICATORS))
        snap = hub.snapshot()
        assert len(snap) == 1 and snap[0]["price"] == 101.0

    def test_ledger_throttled_per_symbol(self, tmp_path):
        from core.symbol_intel import SymbolIntelHub, build_symbol_view
        ledger = tmp_path / "intel.ndjson"
        # force-open the session gate so the test is time-independent
        hub = SymbolIntelHub(ledger_path=ledger, session_gate=lambda: True)
        hub.update(build_symbol_view("NVDA", 100.0, INDICATORS))
        hub.update(build_symbol_view("NVDA", 101.0, INDICATORS))
        lines = ledger.read_text().splitlines()
        assert len(lines) == 1, "second update within throttle must not write"
        row = json.loads(lines[0])
        assert row["symbol"] == "NVDA"

    def test_ledger_respects_session_gate(self, tmp_path):
        from core.symbol_intel import SymbolIntelHub, build_symbol_view
        ledger = tmp_path / "intel.ndjson"
        hub = SymbolIntelHub(ledger_path=ledger, session_gate=lambda: False)
        hub.update(build_symbol_view("NVDA", 100.0, INDICATORS))
        assert not ledger.exists() or ledger.read_text() == ""
        # UI snapshot still updates even when ledger is gated
        assert len(hub.snapshot()) == 1


class TestWiring:
    def test_engine_updates_hub_in_analysis_path(self):
        src = ENGINE_PATH.read_text()
        assert "_symbol_intel_hub" in src
        anchor = src.find("timeframe='5min',\n                        indicators=indicators")
        assert anchor != -1, "market_data assembly anchor missing"
        window = src[anchor: anchor + 2500]
        assert "_symbol_intel_hub" in window, (
            "hub.update must run right after market_data assembly"
        )

    def test_ws_payload_includes_symbol_intel(self):
        src = ROUTES_PATH.read_text()
        assert "'symbol_intel'" in src, (
            "dashboard_update payload must carry symbol_intel"
        )
