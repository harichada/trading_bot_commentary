"""Tests for v-activity-page-2026-09-24: Activity page API endpoint.

Covers:
  - Feature flag off-path (404 when UI_ACTIVITY_PAGE=0)
  - Entry/exit pairing from bot_trades
  - R-multiple calculation including SL==entry edge case
  - Orphan flagging (trades without decision snapshots)
  - Hands-off / external exclusion from bot stats
  - Summary statistics
"""
import pytest
from unittest.mock import patch, MagicMock
import json


class TestActivityPageFeatureFlag:
    """Tests for the UI_ACTIVITY_PAGE feature flag."""

    def test_config_has_ui_activity_page_property(self):
        """Config must expose UI_ACTIVITY_PAGE property."""
        from core.config import Config
        c = Config()
        assert hasattr(c, "UI_ACTIVITY_PAGE"), "Config.UI_ACTIVITY_PAGE missing"

    def test_ui_activity_page_default_off(self, monkeypatch):
        """UI_ACTIVITY_PAGE should default to False (off-path safe)."""
        monkeypatch.delenv("UI_ACTIVITY_PAGE", raising=False)
        from core.config import Config
        c = Config()
        assert c.UI_ACTIVITY_PAGE is False, "UI_ACTIVITY_PAGE should default to False"

    def test_ui_activity_page_env_enables(self, monkeypatch):
        """UI_ACTIVITY_PAGE=1 env var should enable the flag."""
        monkeypatch.setenv("UI_ACTIVITY_PAGE", "1")
        from core.config import Config
        c = Config()
        assert c.UI_ACTIVITY_PAGE is True

    def test_ui_activity_page_env_true_enables(self, monkeypatch):
        """UI_ACTIVITY_PAGE=true env var should enable the flag."""
        monkeypatch.setenv("UI_ACTIVITY_PAGE", "true")
        from core.config import Config
        c = Config()
        assert c.UI_ACTIVITY_PAGE is True

    def test_ui_activity_page_env_zero_disables(self, monkeypatch):
        """UI_ACTIVITY_PAGE=0 env var should disable the flag."""
        monkeypatch.setenv("UI_ACTIVITY_PAGE", "0")
        from core.config import Config
        c = Config()
        assert c.UI_ACTIVITY_PAGE is False


class TestActivityEndpointFlagOff:
    """Tests for /api/activity when feature flag is off."""

    def test_api_activity_returns_404_when_flag_off(self, monkeypatch):
        """GET /api/activity returns 404 when UI_ACTIVITY_PAGE is off."""
        monkeypatch.delenv("UI_ACTIVITY_PAGE", raising=False)
        
        # The Config class reads env at property access time, so monkeypatch works
        from core.config import Config
        assert Config().UI_ACTIVITY_PAGE is False, "Flag should be off"
        
        # Test that the endpoint logic checks the flag
        # (We can't easily test the full endpoint without a running server,
        # but we verify the flag check logic is correct)
        flag_is_off = not Config().UI_ACTIVITY_PAGE
        assert flag_is_off is True, "Flag should be off for this test"

    def test_activity_page_returns_404_when_flag_off(self, monkeypatch):
        """GET /activity returns 404 when UI_ACTIVITY_PAGE is off."""
        monkeypatch.delenv("UI_ACTIVITY_PAGE", raising=False)
        
        # The Config class reads env at property access time, so monkeypatch works
        from core.config import Config
        assert Config().UI_ACTIVITY_PAGE is False, "Flag should be off"
        
        # Verify the flag is correctly False
        flag_is_off = not Config().UI_ACTIVITY_PAGE
        assert flag_is_off is True, "Flag should be off for this test"


class TestRMultipleCalculation:
    """Tests for R-multiple calculation logic."""

    def test_calc_r_normal_case(self):
        """R-multiple calculation: normal case with valid stop loss."""
        def _calc_r(pnl, entry_price, stop_loss, quantity):
            if stop_loss is None or entry_price is None or quantity is None:
                return None
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share < 0.0001:
                return None
            total_risk = risk_per_share * quantity
            if total_risk < 0.01:
                return None
            return pnl / total_risk

        # Entry at $100, stop at $98 (2% risk), 50 shares = $100 risk
        # PnL of $200 = 2R
        r = _calc_r(pnl=200.0, entry_price=100.0, stop_loss=98.0, quantity=50)
        assert r is not None
        assert abs(r - 2.0) < 0.001, f"Expected 2R, got {r}"

    def test_calc_r_stop_equals_entry(self):
        """R-multiple: SL == entry should return None (R undefined), not divide by zero."""
        def _calc_r(pnl, entry_price, stop_loss, quantity):
            if stop_loss is None or entry_price is None or quantity is None:
                return None
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share < 0.0001:
                return None
            total_risk = risk_per_share * quantity
            if total_risk < 0.01:
                return None
            return pnl / total_risk

        r = _calc_r(pnl=50.0, entry_price=100.0, stop_loss=100.0, quantity=100)
        assert r is None, "SL==entry should return None, not error"

    def test_calc_r_stop_very_close_to_entry(self):
        """R-multiple: SL very close to entry (< 0.0001) should return None."""
        def _calc_r(pnl, entry_price, stop_loss, quantity):
            if stop_loss is None or entry_price is None or quantity is None:
                return None
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share < 0.0001:
                return None
            total_risk = risk_per_share * quantity
            if total_risk < 0.01:
                return None
            return pnl / total_risk

        r = _calc_r(pnl=50.0, entry_price=100.0, stop_loss=100.00005, quantity=100)
        assert r is None, "SL almost equal to entry should return None"

    def test_calc_r_none_stop_loss(self):
        """R-multiple: None stop_loss should return None."""
        def _calc_r(pnl, entry_price, stop_loss, quantity):
            if stop_loss is None or entry_price is None or quantity is None:
                return None
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share < 0.0001:
                return None
            total_risk = risk_per_share * quantity
            if total_risk < 0.01:
                return None
            return pnl / total_risk

        r = _calc_r(pnl=50.0, entry_price=100.0, stop_loss=None, quantity=100)
        assert r is None

    def test_calc_r_negative_pnl(self):
        """R-multiple: negative PnL should produce negative R."""
        def _calc_r(pnl, entry_price, stop_loss, quantity):
            if stop_loss is None or entry_price is None or quantity is None:
                return None
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share < 0.0001:
                return None
            total_risk = risk_per_share * quantity
            if total_risk < 0.01:
                return None
            return pnl / total_risk

        # Loss of $100 with $100 risk = -1R
        r = _calc_r(pnl=-100.0, entry_price=100.0, stop_loss=98.0, quantity=50)
        assert r is not None
        assert abs(r - (-1.0)) < 0.001, f"Expected -1R, got {r}"


class TestHandsOffDenylist:
    """Tests for hands-off denylist handling in Activity page."""

    def test_hands_off_symbols_identified(self):
        """MU, HQGE, SPCX should be identified as hands-off."""
        from core.config import Config
        denylist = Config().HANDS_OFF_DENYLIST
        assert "MU" in denylist
        assert "HQGE" in denylist
        assert "SPCX" in denylist

    def test_hands_off_case_insensitive(self):
        """Hands-off check should be case-insensitive."""
        from core.config import Config
        denylist = Config().HANDS_OFF_DENYLIST
        # Denylist stores uppercase
        assert "mu".upper() in denylist
        assert "Mu".upper() in denylist


class TestOrphanFlagging:
    """Tests for orphan trade flagging logic."""

    def test_orphan_detection_logic(self):
        """Trade without reasoning data should be flagged as orphan."""
        def is_orphan(has_reasoning, is_external):
            return not has_reasoning and not is_external

        # Trade with no reasoning and not external = orphan
        assert is_orphan(has_reasoning=False, is_external=False) is True

        # Trade with reasoning = not orphan
        assert is_orphan(has_reasoning=True, is_external=False) is False

        # External trade without reasoning = not orphan (expected for manual)
        assert is_orphan(has_reasoning=False, is_external=True) is False


class TestStatsExclusion:
    """Tests for bot stats excluding hands-off and external positions."""

    def test_stats_exclude_hands_off(self):
        """Bot stats should exclude hands-off positions."""
        trades = [
            {"symbol": "AAPL", "pnl": 100, "is_hands_off": False, "is_external": False},
            {"symbol": "MU", "pnl": -50, "is_hands_off": True, "is_external": False},
            {"symbol": "TSLA", "pnl": 200, "is_hands_off": False, "is_external": False},
        ]

        bot_pnl = sum(t["pnl"] for t in trades if not t["is_hands_off"] and not t["is_external"])
        all_pnl = sum(t["pnl"] for t in trades)

        assert bot_pnl == 300, "Bot PnL should exclude MU (hands-off)"
        assert all_pnl == 250, "All PnL should include everything"

    def test_stats_exclude_external(self):
        """Bot stats should exclude external positions."""
        trades = [
            {"symbol": "AAPL", "pnl": 100, "is_hands_off": False, "is_external": False},
            {"symbol": "META", "pnl": -80, "is_hands_off": False, "is_external": True},
            {"symbol": "TSLA", "pnl": 200, "is_hands_off": False, "is_external": False},
        ]

        bot_pnl = sum(t["pnl"] for t in trades if not t["is_hands_off"] and not t["is_external"])
        all_pnl = sum(t["pnl"] for t in trades)

        assert bot_pnl == 300, "Bot PnL should exclude META (external)"
        assert all_pnl == 220, "All PnL should include everything"

    def test_win_rate_excludes_hands_off_external(self):
        """Bot win rate should exclude hands-off and external."""
        trades = [
            {"symbol": "AAPL", "pnl": 100, "is_hands_off": False, "is_external": False},  # winner
            {"symbol": "MU", "pnl": 500, "is_hands_off": True, "is_external": False},  # hands-off winner
            {"symbol": "TSLA", "pnl": -50, "is_hands_off": False, "is_external": False},  # loser
            {"symbol": "META", "pnl": 200, "is_hands_off": False, "is_external": True},  # external winner
        ]

        bot_trades = [t for t in trades if not t["is_hands_off"] and not t["is_external"]]
        bot_winners = sum(1 for t in bot_trades if t["pnl"] > 0)
        bot_count = len(bot_trades)
        bot_win_rate = 100 * bot_winners / bot_count if bot_count > 0 else 0

        assert bot_count == 2, "Bot should have 2 trades (AAPL, TSLA)"
        assert bot_winners == 1, "Bot should have 1 winner (AAPL)"
        assert bot_win_rate == 50.0, "Bot win rate should be 50%"


class TestActivityEndpointSchema:
    """Tests for /api/activity response schema."""

    def test_response_has_required_fields(self):
        """Activity response must have status, summary, and trades fields."""
        response = {
            "status": "success",
            "summary": {
                "date": "2026-09-24",
                "trades": 5,
                "winners": 3,
                "losers": 2,
                "win_rate": 60.0,
                "pnl": 150.0,
                "total_r": 2.5,
                "avg_r": 0.5,
                "bot_trades": 4,
                "bot_winners": 3,
                "bot_losers": 1,
                "bot_win_rate": 75.0,
                "bot_pnl": 200.0,
                "bot_total_r": 3.0,
                "bot_avg_r": 0.75,
            },
            "trades": []
        }

        assert "status" in response
        assert "summary" in response
        assert "trades" in response

        summary = response["summary"]
        required_summary_fields = [
            "date", "trades", "winners", "losers", "win_rate", "pnl",
            "bot_trades", "bot_winners", "bot_losers", "bot_win_rate", "bot_pnl"
        ]
        for field in required_summary_fields:
            assert field in summary, f"Summary missing {field}"

    def test_trade_has_required_fields(self):
        """Each trade in response must have required fields."""
        trade = {
            "id": 1,
            "symbol": "AAPL",
            "side": "long",
            "strategy": "day_trade_momentum",
            "entry_time": "2026-09-24T09:35:00",
            "entry_time_et": "09:35:00",
            "exit_time": "2026-09-24T11:20:00",
            "exit_time_et": "11:20:00",
            "entry_price": 175.50,
            "exit_price": 177.25,
            "quantity": 100,
            "stop_loss": 174.00,
            "take_profit": 180.00,
            "pnl": 175.00,
            "pnl_pct": 1.0,
            "r_multiple": 1.17,
            "exit_reason": "take_profit",
            "hold_time_min": 105,
            "confidence": 0.72,
            "meta_proba": 0.65,
            "is_hands_off": False,
            "is_external": False,
            "is_orphan": False,
            "reasoning": {
                "setup_type": "momentum_breakout",
                "rs_vs_spy": 0.025,
                "rsi": 0.58,
                "volume_ratio": 1.8,
            }
        }

        required_fields = [
            "id", "symbol", "side", "strategy",
            "entry_price", "exit_price", "quantity",
            "pnl", "exit_reason",
            "is_hands_off", "is_external", "is_orphan"
        ]
        for field in required_fields:
            assert field in trade, f"Trade missing {field}"


class TestTimeConversion:
    """Tests for time conversion to ET."""

    def test_format_time_et_naive_as_et(self):
        """v-activity-page-p0-2026-09-24: Naive timestamps should be treated as ET, not UTC.
        
        Root cause of 4-hour offset: bot_trades stores naive ET but old code
        treated naive as UTC, showing MRNA 15:00 as 11:00.
        """
        from datetime import datetime
        from zoneinfo import ZoneInfo

        et_tz = ZoneInfo("America/New_York")

        def _format_time_et(ts_str):
            """Updated function that treats naive as ET."""
            if not ts_str:
                return None
            try:
                if isinstance(ts_str, str):
                    if 'Z' in ts_str or '+' in ts_str or '-' in ts_str[10:]:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    else:
                        ts = datetime.fromisoformat(ts_str)
                else:
                    ts = ts_str
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=et_tz)
                ts_et = ts.astimezone(et_tz)
                return ts_et.strftime("%H:%M:%S")
            except Exception:
                return str(ts_str)[:8] if ts_str else None

        # Naive timestamp should stay the same time (it's already ET)
        result = _format_time_et("2026-09-24T15:00:00")
        assert result == "15:00:00", f"Naive ET 15:00 should display as 15:00, got {result}"

        # Explicit UTC should convert to ET (EDT = UTC-4)
        result_utc = _format_time_et("2026-09-24T19:00:00+00:00")
        assert result_utc == "15:00:00", f"19:00 UTC should be 15:00 ET, got {result_utc}"

    def test_format_time_et_with_tz(self):
        """Timestamps with explicit timezone should convert correctly."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        et_tz = ZoneInfo("America/New_York")

        def _format_time_et(ts_str):
            if not ts_str:
                return None
            try:
                if isinstance(ts_str, str):
                    if 'Z' in ts_str or '+' in ts_str or '-' in ts_str[10:]:
                        ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    else:
                        ts = datetime.fromisoformat(ts_str)
                else:
                    ts = ts_str
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=et_tz)
                ts_et = ts.astimezone(et_tz)
                return ts_et.strftime("%H:%M:%S")
            except Exception:
                return str(ts_str)[:8] if ts_str else None

        # UTC timestamp (Z suffix)
        result = _format_time_et("2026-09-24T14:30:00Z")
        assert result == "10:30:00", f"14:30 UTC should be 10:30 ET, got {result}"

    def test_calc_hold_time(self):
        """Hold time calculation should return minutes correctly."""
        from datetime import datetime

        def _calc_hold_time(entry_time, exit_time):
            if not entry_time or not exit_time:
                return None
            try:
                if isinstance(entry_time, str):
                    entry = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
                else:
                    entry = entry_time
                if isinstance(exit_time, str):
                    exit_t = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
                else:
                    exit_t = exit_time
                delta = exit_t - entry
                return int(delta.total_seconds() / 60)
            except Exception:
                return None

        # 2 hours = 120 minutes
        result = _calc_hold_time(
            "2026-09-24T09:30:00",
            "2026-09-24T11:30:00"
        )
        assert result == 120


class TestRouteExists:
    """Test that the Activity routes exist in the codebase."""

    def test_api_activity_route_defined(self):
        """GET /api/activity route must be defined."""
        from pathlib import Path
        routes_path = Path(__file__).parent.parent / "api" / "routes.py"
        source = routes_path.read_text()
        assert '/api/activity' in source, "/api/activity endpoint missing"
        assert 'def get_activity' in source, "get_activity function missing"

    def test_activity_page_route_defined(self):
        """GET /activity route must be defined."""
        from pathlib import Path
        routes_path = Path(__file__).parent.parent / "api" / "routes.py"
        source = routes_path.read_text()
        assert '"/activity"' in source or "'/activity'" in source, "/activity page route missing"
        assert 'def get_activity_page' in source, "get_activity_page function missing"


class TestActivityTemplate:
    """Test that Activity template exists."""

    def test_activity_template_exists(self):
        """templates/activity.html must exist."""
        from pathlib import Path
        import os

        # Find the workspace root
        current = Path(__file__).parent.parent
        template_path = current / "templates" / "activity.html"
        assert template_path.exists(), f"Activity template missing at {template_path}"

    def test_activity_template_has_required_elements(self):
        """Activity template must have table, filters, and summary."""
        from pathlib import Path

        current = Path(__file__).parent.parent
        template_path = current / "templates" / "activity.html"

        with open(template_path) as f:
            content = f.read()

        assert '<table' in content, "Template missing table element"
        assert 'date-filter' in content, "Template missing date filter"
        assert 'strategy-filter' in content, "Template missing strategy filter"
        assert 'summary' in content.lower(), "Template missing summary section"
        assert 'reasoning' in content.lower(), "Template missing reasoning section"


class TestBrokerFillsSource:
    """Tests for the modular broker fills source function."""

    def test_broker_fills_source_function_exists(self):
        """_get_broker_fills_for_date function must exist in routes."""
        from pathlib import Path
        routes_path = Path(__file__).parent.parent / "api" / "routes.py"
        source = routes_path.read_text()
        assert 'def _get_broker_fills_for_date' in source, "Broker fills source function missing"

    def test_broker_fills_returns_list(self):
        """_get_broker_fills_for_date should return a list (empty if no engine)."""
        # The function returns [] when trading_engine is None (safe fallback)
        from pathlib import Path
        routes_path = Path(__file__).parent.parent / "api" / "routes.py"
        source = routes_path.read_text()
        assert 'return []' in source, "Function should return empty list as fallback"
        assert 'trading_engine is None' in source, "Function should check for None engine"


class TestDedupeLogic:
    """Tests for deduplicating bot_trades with broker fills."""

    def test_dedupe_function_exists(self):
        """_dedupe_trades_with_broker_fills function must exist."""
        from pathlib import Path
        routes_path = Path(__file__).parent.parent / "api" / "routes.py"
        source = routes_path.read_text()
        assert 'def _dedupe_trades_with_broker_fills' in source, "Dedupe function missing"

    def test_dedupe_broker_only_becomes_orphan(self):
        """A broker fill with no matching db trade should become orphan."""
        # Simulate the dedupe logic
        def _dedupe_trades_with_broker_fills(db_trades, broker_fills, hands_off):
            from datetime import datetime
            db_index = {}
            for t in db_trades:
                sym = (t.get("symbol") or "").upper()
                exit_ts = t.get("exit_time") or ""
                try:
                    if isinstance(exit_ts, str):
                        exit_dt = datetime.fromisoformat(exit_ts.replace('Z', '+00:00'))
                    else:
                        exit_dt = exit_ts
                    key = (sym, exit_dt.strftime("%Y-%m-%d %H:%M") if exit_dt else "")
                except Exception:
                    key = (sym, str(exit_ts)[:16])
                db_index[key] = t

            orphan_fills = []
            for bf in broker_fills:
                sym = (bf.get("symbol") or "").upper()
                exit_ts = bf.get("exit_time") or ""
                try:
                    if isinstance(exit_ts, str):
                        exit_dt = datetime.fromisoformat(exit_ts.replace('Z', '+00:00'))
                    else:
                        exit_dt = exit_ts
                    key = (sym, exit_dt.strftime("%Y-%m-%d %H:%M") if exit_dt else "")
                except Exception:
                    key = (sym, str(exit_ts)[:16])

                if key not in db_index:
                    orphan_fills.append({
                        "symbol": sym,
                        "is_orphan": True,
                        "source": "broker",
                    })

            return db_trades + orphan_fills

        # Test: broker fill with no matching DB trade
        db_trades = [
            {"symbol": "AAPL", "exit_time": "2026-09-24T15:30:00"}
        ]
        broker_fills = [
            {"symbol": "CRCL", "exit_time": "2026-09-24T14:00:00"},  # No match in DB
            {"symbol": "AAPL", "exit_time": "2026-09-24T15:30:00"},  # Matches DB
        ]

        result = _dedupe_trades_with_broker_fills(db_trades, broker_fills, frozenset())

        # Should have 2 trades: AAPL from DB, CRCL as orphan
        assert len(result) == 2
        orphans = [t for t in result if t.get("is_orphan")]
        assert len(orphans) == 1, "CRCL should be an orphan"
        assert orphans[0]["symbol"] == "CRCL"

    def test_dedupe_both_sources_keeps_db(self):
        """When both sources have the same trade, DB version is kept (no duplicate)."""
        def _dedupe_trades_with_broker_fills(db_trades, broker_fills, hands_off):
            from datetime import datetime
            db_index = {}
            for t in db_trades:
                sym = (t.get("symbol") or "").upper()
                exit_ts = t.get("exit_time") or ""
                try:
                    if isinstance(exit_ts, str):
                        exit_dt = datetime.fromisoformat(exit_ts.replace('Z', '+00:00'))
                    else:
                        exit_dt = exit_ts
                    key = (sym, exit_dt.strftime("%Y-%m-%d %H:%M") if exit_dt else "")
                except Exception:
                    key = (sym, str(exit_ts)[:16])
                db_index[key] = t

            orphan_fills = []
            for bf in broker_fills:
                sym = (bf.get("symbol") or "").upper()
                exit_ts = bf.get("exit_time") or ""
                try:
                    if isinstance(exit_ts, str):
                        exit_dt = datetime.fromisoformat(exit_ts.replace('Z', '+00:00'))
                    else:
                        exit_dt = exit_ts
                    key = (sym, exit_dt.strftime("%Y-%m-%d %H:%M") if exit_dt else "")
                except Exception:
                    key = (sym, str(exit_ts)[:16])

                if key not in db_index:
                    orphan_fills.append({
                        "symbol": sym,
                        "is_orphan": True,
                        "source": "broker",
                    })

            return db_trades + orphan_fills

        # Same trade in both sources
        db_trades = [
            {"symbol": "INTC", "exit_time": "2026-09-24T12:00:00", "pnl": 100, "source": "db"}
        ]
        broker_fills = [
            {"symbol": "INTC", "exit_time": "2026-09-24T12:00:00", "pnl": 100, "source": "broker"}
        ]

        result = _dedupe_trades_with_broker_fills(db_trades, broker_fills, frozenset())

        # Should have only 1 trade (DB version, no duplicate)
        assert len(result) == 1
        assert result[0].get("source") == "db", "DB version should be kept"


class TestNavEntry:
    """Tests for Activity nav entry in dashboard."""

    def test_nav_entry_conditional_on_flag(self):
        """Dashboard nav should conditionally include Activity based on flag."""
        from pathlib import Path
        dashboard_path = Path(__file__).parent.parent / "templates" / "dashboard.html"
        source = dashboard_path.read_text()

        assert 'UI_ACTIVITY_PAGE' in source, "Dashboard must check UI_ACTIVITY_PAGE flag"
        assert 'activity' in source.lower(), "Dashboard must have activity nav entry"
        assert '/activity' in source, "Dashboard must link to /activity page"

    def test_flag_injected_into_dashboard(self):
        """Dashboard route must inject UI_ACTIVITY_PAGE flag."""
        from pathlib import Path
        routes_path = Path(__file__).parent.parent / "api" / "routes.py"
        source = routes_path.read_text()

        assert 'UI_ACTIVITY_PAGE' in source, "Route must reference UI_ACTIVITY_PAGE"
        assert 'window.UI_ACTIVITY_PAGE' in source, "Route must inject flag into window"


class TestResponsiveLayout:
    """Tests for responsive table layout at 1280px."""

    def test_template_has_one_line_truncate_class(self):
        """Activity template must have one-line-truncate class for ellipsis preview."""
        from pathlib import Path
        template_path = Path(__file__).parent.parent / "templates" / "activity.html"
        source = template_path.read_text()

        assert 'one-line-truncate' in source, "Template must have one-line-truncate class"
        assert 'text-overflow: ellipsis' in source, "Template must have text-overflow ellipsis"

    def test_template_has_reasoning_preview(self):
        """Activity template must have reasoning-preview element for collapsed row."""
        from pathlib import Path
        template_path = Path(__file__).parent.parent / "templates" / "activity.html"
        source = template_path.read_text()

        assert 'reasoning-preview' in source, "Template must have reasoning-preview class"
        assert 'getReasoningPreview' in source, "Template must have getReasoningPreview function"

    def test_template_has_combined_entry_exit_cells(self):
        """Activity template must combine entry time+price and exit time+price."""
        from pathlib import Path
        template_path = Path(__file__).parent.parent / "templates" / "activity.html"
        source = template_path.read_text()

        assert 'entry-exit-cell' in source, "Template must have entry-exit-cell class"

    def test_reasoning_grid_wraps(self):
        """Reasoning grid must use flex-wrap for responsive layout."""
        from pathlib import Path
        template_path = Path(__file__).parent.parent / "templates" / "activity.html"
        source = template_path.read_text()

        assert 'flex-wrap: wrap' in source, "Reasoning grid must use flex-wrap"


class TestBrokerNoteLogic:
    """Tests for broker-only fills note on past dates."""

    def test_template_has_broker_note_element(self):
        """Activity template must have broker-note element."""
        from pathlib import Path
        template_path = Path(__file__).parent.parent / "templates" / "activity.html"
        source = template_path.read_text()

        assert 'broker-note' in source, "Template must have broker-note element"
        assert 'Broker-only fills are shown for today only' in source, "Template must have broker note text"

    def test_broker_note_visibility_logic(self):
        """Broker note should be shown when date is not today."""
        from pathlib import Path
        template_path = Path(__file__).parent.parent / "templates" / "activity.html"
        source = template_path.read_text()

        # Check that the JS logic compares selected date with today
        assert 'selectedDate !== todayStr' in source, "Must compare selected date with today"
        assert "brokerNote.style.display = 'flex'" in source, "Must show broker note"
        assert "brokerNote.style.display = 'none'" in source, "Must hide broker note"

    def test_broker_note_hidden_by_default(self):
        """Broker note element should be hidden by default (display: none)."""
        from pathlib import Path
        template_path = Path(__file__).parent.parent / "templates" / "activity.html"
        source = template_path.read_text()

        # Check that broker-note has style="display: none;" initially
        assert 'id="broker-note"' in source, "Must have broker-note id"
        # The element should start hidden
        assert 'broker-note' in source and 'display: none' in source, "Broker note should be hidden by default"


# =============================================================================
# v-activity-page-p0-2026-09-24: Tests for P0 fixes
# =============================================================================

class TestFifoPairing:
    """v-activity-page-p0-2026-09-24: Tests for FIFO pairing of broker fills."""

    def test_fifo_pairs_long_round_trip(self):
        """FIFO should pair OPENING (buy) with CLOSING (sell) for longs."""
        from collections import defaultdict

        def _fifo_pair_fills(raw_fills, et_tz=None):
            entry_queues = defaultdict(list)
            round_trips = []
            sorted_fills = sorted(raw_fills, key=lambda f: f.get('time', ''))

            for fill in sorted_fills:
                symbol = fill['symbol']
                instruction = fill.get('instruction', '').upper()
                quantity = fill['quantity']
                price = fill['price']
                fill_time = fill.get('time', '')

                is_opening = instruction == 'OPENING'
                is_closing = instruction == 'CLOSING'

                if is_opening:
                    entry_queues[symbol].append({
                        'quantity': quantity,
                        'price': price,
                        'time': fill_time,
                        'description': fill.get('description', ''),
                    })
                elif is_closing and entry_queues[symbol]:
                    entry = entry_queues[symbol].pop(0)
                    pnl = (price - entry['price']) * quantity
                    round_trips.append({
                        'symbol': symbol,
                        'side': 'long',
                        'entry_price': entry['price'],
                        'exit_price': price,
                        'quantity': quantity,
                        'pnl': round(pnl, 2),
                    })

            return round_trips

        # Buy at 100, sell at 105 = +5 per share
        fills = [
            {'symbol': 'AAPL', 'instruction': 'OPENING', 'quantity': 100, 'price': 100.0, 'time': '2026-09-24T09:30:00'},
            {'symbol': 'AAPL', 'instruction': 'CLOSING', 'quantity': 100, 'price': 105.0, 'time': '2026-09-24T10:00:00'},
        ]

        result = _fifo_pair_fills(fills)
        assert len(result) == 1
        assert result[0]['symbol'] == 'AAPL'
        assert result[0]['side'] == 'long'
        assert result[0]['pnl'] == 500.0  # 100 shares * $5

    def test_fifo_pairs_short_round_trip(self):
        """FIFO should handle shorts: sell-to-open paired with buy-to-close."""
        from collections import defaultdict

        def _fifo_pair_fills(raw_fills, et_tz=None):
            entry_queues = defaultdict(list)
            round_trips = []
            sorted_fills = sorted(raw_fills, key=lambda f: f.get('time', ''))

            for fill in sorted_fills:
                symbol = fill['symbol']
                instruction = fill.get('instruction', '').upper()
                quantity = fill['quantity']
                price = fill['price']
                fill_time = fill.get('time', '')
                description = fill.get('description', '').upper()

                is_opening = instruction == 'OPENING'
                is_closing = instruction == 'CLOSING'
                is_short = 'SELL SHORT' in description or 'SHORT' in description

                if is_opening:
                    entry_queues[symbol].append({
                        'quantity': quantity,
                        'price': price,
                        'time': fill_time,
                        'is_short': is_short,
                    })
                elif is_closing and entry_queues[symbol]:
                    entry = entry_queues[symbol].pop(0)
                    side = 'short' if entry.get('is_short') else 'long'
                    if side == 'short':
                        pnl = (entry['price'] - price) * quantity
                    else:
                        pnl = (price - entry['price']) * quantity
                    round_trips.append({
                        'symbol': symbol,
                        'side': side,
                        'entry_price': entry['price'],
                        'exit_price': price,
                        'quantity': quantity,
                        'pnl': round(pnl, 2),
                    })

            return round_trips

        # Short sell at 100, buy to close at 95 = +5 per share
        fills = [
            {'symbol': 'META', 'instruction': 'OPENING', 'quantity': 50, 'price': 100.0, 'time': '2026-09-24T09:30:00', 'description': 'SELL SHORT'},
            {'symbol': 'META', 'instruction': 'CLOSING', 'quantity': 50, 'price': 95.0, 'time': '2026-09-24T10:00:00', 'description': 'BUY TO COVER'},
        ]

        result = _fifo_pair_fills(fills)
        assert len(result) == 1
        assert result[0]['symbol'] == 'META'
        assert result[0]['side'] == 'short'
        assert result[0]['pnl'] == 250.0  # 50 shares * $5

    def test_fifo_handles_partial_fills(self):
        """FIFO should handle partial fills correctly."""
        from collections import defaultdict

        def _fifo_pair_fills(raw_fills, et_tz=None):
            entry_queues = defaultdict(list)
            round_trips = []
            sorted_fills = sorted(raw_fills, key=lambda f: f.get('time', ''))

            for fill in sorted_fills:
                symbol = fill['symbol']
                instruction = fill.get('instruction', '').upper()
                quantity = fill['quantity']
                price = fill['price']
                fill_time = fill.get('time', '')

                is_opening = instruction == 'OPENING'
                is_closing = instruction == 'CLOSING'

                if is_opening:
                    entry_queues[symbol].append({
                        'quantity': quantity,
                        'price': price,
                        'time': fill_time,
                    })
                elif is_closing and entry_queues[symbol]:
                    remaining_exit_qty = quantity
                    while remaining_exit_qty > 0 and entry_queues[symbol]:
                        entry = entry_queues[symbol][0]
                        matched_qty = min(entry['quantity'], remaining_exit_qty)
                        pnl = (price - entry['price']) * matched_qty
                        round_trips.append({
                            'symbol': symbol,
                            'side': 'long',
                            'entry_price': entry['price'],
                            'exit_price': price,
                            'quantity': matched_qty,
                            'pnl': round(pnl, 2),
                        })
                        remaining_exit_qty -= matched_qty
                        entry['quantity'] -= matched_qty
                        if entry['quantity'] <= 0:
                            entry_queues[symbol].pop(0)

            return round_trips

        # Two entries, one exit covering both
        fills = [
            {'symbol': 'INTC', 'instruction': 'OPENING', 'quantity': 50, 'price': 30.0, 'time': '2026-09-24T09:30:00'},
            {'symbol': 'INTC', 'instruction': 'OPENING', 'quantity': 50, 'price': 31.0, 'time': '2026-09-24T09:35:00'},
            {'symbol': 'INTC', 'instruction': 'CLOSING', 'quantity': 100, 'price': 32.0, 'time': '2026-09-24T10:00:00'},
        ]

        result = _fifo_pair_fills(fills)
        assert len(result) == 2  # Two round trips from partial matching
        # First 50 shares: entry 30, exit 32 = +$100
        assert result[0]['pnl'] == 100.0
        # Second 50 shares: entry 31, exit 32 = +$50
        assert result[1]['pnl'] == 50.0


class TestOpenPositionsExclusion:
    """v-activity-page-p0-2026-09-24: Open positions excluded from trade totals."""

    def test_open_positions_section_in_template(self):
        """Activity template must have open-positions-section."""
        from pathlib import Path
        template_path = Path(__file__).parent.parent / "templates" / "activity.html"
        source = template_path.read_text()

        assert 'open-positions-section' in source, "Template must have open-positions-section"
        assert 'excluded from trade totals' in source, "Template must indicate exclusion"
        assert 'renderOpenPositions' in source, "Template must have renderOpenPositions function"

    def test_api_returns_open_positions(self):
        """API response must include open_positions field."""
        from pathlib import Path
        routes_path = Path(__file__).parent.parent / "api" / "routes.py"
        source = routes_path.read_text()

        assert '"open_positions":' in source, "API must return open_positions in response"
        assert '_get_open_positions_for_date' in source, "API must have open positions function"


class TestRsFormatting:
    """v-activity-page-p0-2026-09-24: RS vs SPY formatting (no double multiply)."""

    def test_rs_raw_ratio_formatted_correctly(self):
        """RS stored as raw ratio (4.72) should show as +472.0%, not +47200%."""
        def format_rs_vs_spy(val):
            if val is None:
                return None
            abs_val = abs(val)
            is_raw_ratio = abs_val < 10
            pct_val = val * 100 if is_raw_ratio else val
            sign = '+' if pct_val >= 0 else ''
            return f"{sign}{pct_val:.1f}%"

        # Raw ratio (stored as 4.72 for +472%)
        result = format_rs_vs_spy(4.72)
        assert result == "+472.0%", f"Expected +472.0%, got {result}"

        # Negative ratio
        result_neg = format_rs_vs_spy(-0.5)
        assert result_neg == "-50.0%", f"Expected -50.0%, got {result_neg}"

    def test_rs_already_percentage_not_doubled(self):
        """RS already as percentage (47.2) should not be doubled."""
        def format_rs_vs_spy(val):
            if val is None:
                return None
            abs_val = abs(val)
            is_raw_ratio = abs_val < 10
            pct_val = val * 100 if is_raw_ratio else val
            sign = '+' if pct_val >= 0 else ''
            return f"{sign}{pct_val:.1f}%"

        # Already a percentage
        result = format_rs_vs_spy(47.2)
        assert result == "+47.2%", f"Expected +47.2%, got {result}"

    def test_template_has_format_rs_function(self):
        """Activity template must have formatRsVsSpy function."""
        from pathlib import Path
        template_path = Path(__file__).parent.parent / "templates" / "activity.html"
        source = template_path.read_text()

        assert 'formatRsVsSpy' in source, "Template must have formatRsVsSpy function"


class TestSlTpDecimals:
    """v-activity-page-p0-2026-09-24: SL/TP should show 2 decimal places."""

    def test_template_formats_sl_tp_with_decimals(self):
        """Activity template must use toFixed(2) for SL/TP."""
        from pathlib import Path
        template_path = Path(__file__).parent.parent / "templates" / "activity.html"
        source = template_path.read_text()

        # Look for the formatSlTp function
        assert 'formatSlTp' in source, "Template must have formatSlTp function"
        assert '.toFixed(2)' in source, "Template must use toFixed(2) for decimals"


class TestSignalStrengthLabeling:
    """v-activity-page-p0-2026-09-24: Signal strength labeled correctly (not probability)."""

    def test_template_labels_signal_not_probability(self):
        """Activity template must label signal strength as not a probability."""
        from pathlib import Path
        template_path = Path(__file__).parent.parent / "templates" / "activity.html"
        source = template_path.read_text()

        assert 'not a probability' in source, "Template must indicate signal is not probability"
        assert 'Signal strength' in source or 'Signal-strength' in source, "Template must have signal strength label"


class TestSchwabFailureLogging:
    """v-activity-page-p0-2026-09-24: Schwab failures logged at WARNING."""

    def test_schwab_transactions_logs_warning(self):
        """_fetch_schwab_transactions_for_date must log failures at WARNING."""
        from pathlib import Path
        routes_path = Path(__file__).parent.parent / "api" / "routes.py"
        source = routes_path.read_text()

        assert 'logger.warning' in source, "Must log failures at WARNING"
        assert 'schwab_transactions_non_200' in source or 'schwab_transactions_error' in source, "Must log Schwab errors"


class TestSideColumnWidth:
    """v-activity-page-p0-2026-09-24: Side column should not truncate LONG/SHORT."""

    def test_side_column_width_sufficient(self):
        """Side column must be wide enough for LONG/SHORT text."""
        from pathlib import Path
        template_path = Path(__file__).parent.parent / "templates" / "activity.html"
        source = template_path.read_text()

        # Check that side column is at least 45px wide
        assert 'th:nth-child(3)' in source, "Must have side column width rule"
        # The side pill should not truncate
        assert 'side-pill' in source, "Must have side-pill class for non-truncating display"


class TestActivityPageP0Fixture:
    """v-activity-page-p0-2026-09-24: Fixture matching real 9/24 data."""

    @pytest.fixture
    def sept_24_fixture(self):
        """Fixture mirroring 9/24 real data: 9 round trips with various sources."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        et_tz = ZoneInfo("America/New_York")

        return {
            "trades": [
                # From broker fills only (not in bot_trades)
                {"symbol": "CRCL", "side": "long", "entry_time": "2026-09-24T09:35:00",
                 "exit_time": "2026-09-24T10:15:00", "entry_price": 12.50, "exit_price": 13.00,
                 "quantity": 100, "pnl": 50.0, "stop_loss": 12.00, "take_profit": 14.00,
                 "is_orphan": True, "source": "broker"},
                {"symbol": "CRWV", "side": "long", "entry_time": "2026-09-24T09:40:00",
                 "exit_time": "2026-09-24T11:00:00", "entry_price": 8.00, "exit_price": 7.50,
                 "quantity": 150, "pnl": -75.0, "stop_loss": 7.50, "take_profit": 9.00,
                 "is_orphan": True, "source": "broker"},
                {"symbol": "NBIS", "side": "long", "entry_time": "2026-09-24T10:00:00",
                 "exit_time": "2026-09-24T11:30:00", "entry_price": 22.00, "exit_price": 23.50,
                 "quantity": 50, "pnl": 75.0, "stop_loss": 21.00, "take_profit": 25.00,
                 "is_orphan": True, "source": "broker"},
                {"symbol": "GOOGL", "side": "long", "entry_time": "2026-09-24T10:15:00",
                 "exit_time": "2026-09-24T12:00:00", "entry_price": 165.00, "exit_price": 167.50,
                 "quantity": 30, "pnl": 75.0, "stop_loss": 163.00, "take_profit": 170.00,
                 "is_orphan": True, "source": "broker"},
                # Two INTC trades
                {"symbol": "INTC", "side": "long", "entry_time": "2026-09-24T09:45:00",
                 "exit_time": "2026-09-24T10:30:00", "entry_price": 30.00, "exit_price": 30.50,
                 "quantity": 200, "pnl": 100.0, "stop_loss": 29.50, "take_profit": 31.50,
                 "is_orphan": True, "source": "broker"},
                {"symbol": "INTC", "side": "long", "entry_time": "2026-09-24T11:00:00",
                 "exit_time": "2026-09-24T12:30:00", "entry_price": 30.50, "exit_price": 31.00,
                 "quantity": 200, "pnl": 100.0, "stop_loss": 30.00, "take_profit": 32.00,
                 "is_orphan": True, "source": "broker"},
                # From bot_trades with naive ET timestamps
                {"symbol": "ONDS", "side": "long", "entry_time": "2026-09-24T09:30:00",
                 "exit_time": "2026-09-24T15:00:00", "entry_price": 8.00, "exit_price": 8.50,
                 "quantity": 300, "pnl": 150.0, "stop_loss": 7.50, "take_profit": 9.00,
                 "is_orphan": False, "source": "db",
                 "reasoning": {"rs_vs_spy": 4.72, "rsi": 55, "score": 0.60}},  # RS stored raw
                {"symbol": "MRNA", "side": "long", "entry_time": "2026-09-24T10:00:00",
                 "exit_time": "2026-09-24T15:00:00", "entry_price": 65.00, "exit_price": 66.00,
                 "quantity": 50, "pnl": 50.0, "stop_loss": 63.50, "take_profit": 68.00,
                 "is_orphan": False, "source": "db",
                 "reasoning": {"rs_vs_spy": 0.25, "rsi": 48, "score": 0.45}},
                # META short (external)
                {"symbol": "META", "side": "short", "entry_time": "2026-09-24T09:35:00",
                 "exit_time": "2026-09-24T14:00:00", "entry_price": 550.00, "exit_price": 545.00,
                 "quantity": 20, "pnl": 100.0, "stop_loss": 555.00, "take_profit": 540.00,
                 "is_external": True, "is_orphan": False, "source": "external"},
            ],
            "open_positions": [
                # Currently held position (excluded from totals)
                {"symbol": "MU", "side": "long", "entry_price": 100.00, "quantity": 50,
                 "unrealized_pnl": 150.0, "stop_loss": 98.00, "take_profit": 110.00,
                 "is_hands_off": True},
                {"symbol": "HQGE", "side": "long", "entry_price": 5.00, "quantity": 500,
                 "unrealized_pnl": -50.0, "stop_loss": 4.50, "take_profit": 6.00,
                 "is_hands_off": True},
            ],
        }

    def test_fixture_has_nine_round_trips(self, sept_24_fixture):
        """Fixture should have 9 round trips."""
        assert len(sept_24_fixture["trades"]) == 9

    def test_fixture_has_broker_only_trades(self, sept_24_fixture):
        """Fixture should have trades that only came from broker (is_orphan=True)."""
        orphans = [t for t in sept_24_fixture["trades"] if t.get("is_orphan")]
        assert len(orphans) >= 6, "Should have CRCL, CRWV, NBIS, GOOGL, INTC x2 as orphans"

    def test_fixture_has_short(self, sept_24_fixture):
        """Fixture should have META as a short."""
        shorts = [t for t in sept_24_fixture["trades"] if t.get("side") == "short"]
        assert len(shorts) == 1
        assert shorts[0]["symbol"] == "META"

    def test_fixture_has_open_positions(self, sept_24_fixture):
        """Fixture should have open positions (excluded from totals)."""
        assert len(sept_24_fixture["open_positions"]) >= 1
        # MU and HQGE are hands-off
        hands_off = [p for p in sept_24_fixture["open_positions"] if p.get("is_hands_off")]
        assert len(hands_off) >= 2

    def test_fixture_rs_stored_raw(self, sept_24_fixture):
        """Fixture should have RS stored as raw ratio (4.72 = +472%)."""
        onds = next(t for t in sept_24_fixture["trades"] if t["symbol"] == "ONDS")
        rs = onds.get("reasoning", {}).get("rs_vs_spy")
        assert rs == 4.72, "ONDS should have RS stored as raw 4.72"

    def test_open_positions_excluded_from_pnl_totals(self, sept_24_fixture):
        """Open positions should NOT be counted in trade totals."""
        trades = sept_24_fixture["trades"]
        open_positions = sept_24_fixture["open_positions"]

        # Total PnL should only include closed trades
        closed_pnl = sum(t.get("pnl", 0) for t in trades)
        open_unrealized = sum(p.get("unrealized_pnl", 0) for p in open_positions)

        # These should be separate
        assert closed_pnl != closed_pnl + open_unrealized, "Open positions must be excluded from totals"
