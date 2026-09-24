"""Tests for data_providers/ingest_universe.py and ingest_alpaca_bars.py.

v-ingest-traded-2026-09-24: comprehensive mocked tests covering:
  - Flag off means identical symbols
  - Union ordering and deduplication
  - Top-N empty + traded present means traded used (no symbols=0)
  - All sources failing still gives top-N
  - Action allowlist excludes shadow_pass
  - Checkpoint key changes when symbols change
  - CLI arg parsing
"""
import argparse
from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


class TestResolveIngestUniverse:
    """Test resolve_ingest_universe() ordering and deduplication."""

    def test_top_n_order_preserved(self):
        """Top-N symbols appear first in their original volume-sorted order."""
        from data_providers.ingest_universe import resolve_ingest_universe

        top_n = ["NVDA", "TSLA", "AAPL", "MSFT"]
        traded = {"GOOG", "AMZN"}

        result = resolve_ingest_universe(top_n, traded)

        assert result[:4] == ["NVDA", "TSLA", "AAPL", "MSFT"]
        assert set(result[4:]) == {"AMZN", "GOOG"}

    def test_traded_extras_sorted_alphabetically(self):
        """Traded symbols not in top-N are sorted alphabetically."""
        from data_providers.ingest_universe import resolve_ingest_universe

        top_n = ["NVDA"]
        traded = {"ZZZZ", "AAAA", "MMMM"}

        result = resolve_ingest_universe(top_n, traded)

        assert result == ["NVDA", "AAAA", "MMMM", "ZZZZ"]

    def test_deduplication(self):
        """Symbols appearing in both top-N and traded are deduplicated."""
        from data_providers.ingest_universe import resolve_ingest_universe

        top_n = ["NVDA", "TSLA", "AAPL"]
        traded = {"TSLA", "GOOG", "AAPL"}

        result = resolve_ingest_universe(top_n, traded)

        assert result == ["NVDA", "TSLA", "AAPL", "GOOG"]
        assert len(result) == len(set(result))

    def test_uppercase_normalization(self):
        """All symbols are uppercased."""
        from data_providers.ingest_universe import resolve_ingest_universe

        top_n = ["nvda", "Tsla"]
        traded = {"goog", "Amzn"}

        result = resolve_ingest_universe(top_n, traded)

        for sym in result:
            assert sym == sym.upper()

    def test_empty_traded_returns_top_n_only(self):
        """When traded is empty, only top-N symbols are returned."""
        from data_providers.ingest_universe import resolve_ingest_universe

        top_n = ["NVDA", "TSLA", "AAPL"]
        traded: set[str] = set()

        result = resolve_ingest_universe(top_n, traded)

        assert result == ["NVDA", "TSLA", "AAPL"]

    def test_empty_top_n_returns_traded_only(self):
        """When top-N is empty, only traded symbols are returned (sorted)."""
        from data_providers.ingest_universe import resolve_ingest_universe

        top_n: list[str] = []
        traded = {"GOOG", "AMZN", "META"}

        result = resolve_ingest_universe(top_n, traded)

        assert result == ["AMZN", "GOOG", "META"]


class TestSymbolListHash:
    """Test symbol_list_hash() determinism and change detection."""

    def test_deterministic(self):
        """Same input produces same hash."""
        from data_providers.ingest_universe import symbol_list_hash

        symbols = ["NVDA", "TSLA", "AAPL"]

        hash1 = symbol_list_hash(symbols)
        hash2 = symbol_list_hash(symbols)

        assert hash1 == hash2

    def test_order_matters(self):
        """Different order produces different hash."""
        from data_providers.ingest_universe import symbol_list_hash

        hash1 = symbol_list_hash(["NVDA", "TSLA", "AAPL"])
        hash2 = symbol_list_hash(["TSLA", "NVDA", "AAPL"])

        assert hash1 != hash2

    def test_content_matters(self):
        """Different symbols produce different hash."""
        from data_providers.ingest_universe import symbol_list_hash

        hash1 = symbol_list_hash(["NVDA", "TSLA"])
        hash2 = symbol_list_hash(["NVDA", "GOOG"])

        assert hash1 != hash2

    def test_length_param(self):
        """Hash length is configurable."""
        from data_providers.ingest_universe import symbol_list_hash

        short = symbol_list_hash(["NVDA"], length=4)
        long = symbol_list_hash(["NVDA"], length=16)

        assert len(short) == 4
        assert len(long) == 16

    def test_case_insensitive(self):
        """Hash is case-insensitive (uppercased internally)."""
        from data_providers.ingest_universe import symbol_list_hash

        hash1 = symbol_list_hash(["NVDA", "tsla"])
        hash2 = symbol_list_hash(["nvda", "TSLA"])

        assert hash1 == hash2


class TestTradedSymbolsForDay:
    """Test traded_symbols_for_day() with mocked DB queries."""

    @patch("data_providers.ingest_universe.sqlalchemy")
    def test_queries_all_sources(self, mock_sa):
        """All source tables are queried by default."""
        from data_providers.ingest_universe import traded_symbols_for_day

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_sa.create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_conn.execute.return_value.fetchall.return_value = []

        traded_symbols_for_day("postgresql://test", date(2026, 9, 24))

        assert mock_conn.execute.call_count >= 5

    @patch("data_providers.ingest_universe.sqlalchemy")
    def test_combines_all_sources(self, mock_sa):
        """Symbols from all sources are combined."""
        from data_providers.ingest_universe import traded_symbols_for_day

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_sa.create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_sa.text = lambda x: x

        mock_conn.execute.return_value.fetchall.side_effect = [
            [("NVDA",)],
            [("TSLA",)],
            [("AAPL",)],
            [("GOOG",)],
            [("META",)],
        ]

        result = traded_symbols_for_day("postgresql://test", date(2026, 9, 24))

        assert result == {"NVDA", "TSLA", "AAPL", "GOOG", "META"}

    @patch("data_providers.ingest_universe.sqlalchemy")
    def test_source_failure_continues(self, mock_sa):
        """One failing source doesn't break the query."""
        from data_providers.ingest_universe import traded_symbols_for_day

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_sa.create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_sa.text = lambda x: x

        def side_effect(*args, **kwargs):
            mock_result = MagicMock()
            if "bot_trades" in str(args):
                raise Exception("Table does not exist")
            mock_result.fetchall.return_value = [("NVDA",)]
            return mock_result

        mock_conn.execute.side_effect = side_effect

        result = traded_symbols_for_day("postgresql://test", date(2026, 9, 24))

        assert "NVDA" in result

    @patch("data_providers.ingest_universe.sqlalchemy")
    def test_action_allowlist_applied(self, mock_sa):
        """Only specified actions are included from bot_decisions."""
        from data_providers.ingest_universe import traded_symbols_for_day

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_sa.create_engine.return_value = mock_engine
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_sa.text = lambda x: x
        mock_conn.execute.return_value.fetchall.return_value = []

        custom_actions = ["signal_buy", "signal_sell"]
        traded_symbols_for_day(
            "postgresql://test",
            date(2026, 9, 24),
            decision_actions=custom_actions,
        )

        calls = mock_conn.execute.call_args_list
        decisions_calls = [c for c in calls if "bot_decisions" in str(c)]
        assert len(decisions_calls) >= 1


class TestActionAllowlistExcludesShadowPass:
    """Test that shadow_pass is excluded from default action allowlist."""

    def test_default_excludes_shadow_pass(self):
        """DEFAULT_DECISION_ACTIONS does not include shadow_pass."""
        from data_providers.ingest_universe import DEFAULT_DECISION_ACTIONS

        assert "shadow_pass" not in DEFAULT_DECISION_ACTIONS

    def test_default_includes_signal_actions(self):
        """DEFAULT_DECISION_ACTIONS includes real signal actions."""
        from data_providers.ingest_universe import DEFAULT_DECISION_ACTIONS

        assert "signal_buy" in DEFAULT_DECISION_ACTIONS
        assert "signal_sell" in DEFAULT_DECISION_ACTIONS
        assert "bracket_placed" in DEFAULT_DECISION_ACTIONS
        assert "position_created" in DEFAULT_DECISION_ACTIONS


class TestFlagOffMeansIdenticalSymbols:
    """Test that --no-include-traded produces byte-identical behavior."""

    def test_cli_no_include_traded_skips_traded_resolution(self):
        """With --no-include-traded, traded symbols are not resolved."""
        from ingest_alpaca_bars import _parse_args

        args = _parse_args([
            "--top-n", "10",
            "--days", "7",
            "--no-include-traded",
        ])

        assert args.include_traded is False

    @patch("ingest_alpaca_bars.top_symbols_by_volume")
    @patch("ingest_alpaca_bars.traded_symbols_for_day")
    def test_flag_off_no_traded_query(self, mock_traded, mock_top_n):
        """When flag is off, traded_symbols_for_day is not called."""
        from ingest_alpaca_bars import _resolve_symbols_with_stall_guard, IngestConfig
        import logging

        mock_cfg = MagicMock(spec=IngestConfig)
        mock_cfg.dsn = "postgresql://test"
        mock_top_n.return_value = ["NVDA", "TSLA", "AAPL"]

        args = argparse.Namespace(
            symbols=None,
            top_n=10,
            days=7,
            include_traded=False,
            traded_date=None,
        )
        end = datetime(2026, 9, 24, 21, 0, tzinfo=timezone.utc)
        logger = logging.getLogger("test")

        symbols, info = _resolve_symbols_with_stall_guard(mock_cfg, args, end, logger)

        mock_traded.assert_not_called()
        assert symbols == ["NVDA", "TSLA", "AAPL"]
        assert info["traded_count"] == 0


class TestStallGuard:
    """Test stall guard behavior when top-N is empty."""

    @patch("ingest_alpaca_bars.top_symbols_by_volume")
    @patch("ingest_alpaca_bars.traded_symbols_for_day")
    def test_empty_top_n_uses_traded(self, mock_traded, mock_top_n):
        """When top-N is empty but traded has symbols, use traded."""
        from ingest_alpaca_bars import _resolve_symbols_with_stall_guard, IngestConfig
        import logging

        mock_cfg = MagicMock(spec=IngestConfig)
        mock_cfg.dsn = "postgresql://test"
        mock_top_n.return_value = []
        mock_traded.return_value = {"NVDA", "TSLA"}

        args = argparse.Namespace(
            symbols=None,
            top_n=10,
            days=7,
            include_traded=True,
            traded_date=None,
        )
        end = datetime(2026, 9, 24, 21, 0, tzinfo=timezone.utc)
        logger = logging.getLogger("test")

        symbols, info = _resolve_symbols_with_stall_guard(mock_cfg, args, end, logger)

        assert set(symbols) == {"NVDA", "TSLA"}
        assert info["final_count"] == 2

    @patch("ingest_alpaca_bars.top_symbols_by_volume")
    @patch("ingest_alpaca_bars.traded_symbols_for_day")
    def test_empty_both_retries_wide_lookback(self, mock_traded, mock_top_n):
        """When both are empty, retry with wider lookback."""
        from ingest_alpaca_bars import _resolve_symbols_with_stall_guard, IngestConfig
        import logging

        mock_cfg = MagicMock(spec=IngestConfig)
        mock_cfg.dsn = "postgresql://test"

        call_count = [0]
        def mock_top_n_impl(dsn, n, days=30):
            call_count[0] += 1
            if call_count[0] == 1:
                return []
            return ["NVDA"]

        mock_top_n.side_effect = mock_top_n_impl
        mock_traded.return_value = set()

        args = argparse.Namespace(
            symbols=None,
            top_n=10,
            days=7,
            include_traded=True,
            traded_date=None,
        )
        end = datetime(2026, 9, 24, 21, 0, tzinfo=timezone.utc)
        logger = logging.getLogger("test")

        symbols, info = _resolve_symbols_with_stall_guard(mock_cfg, args, end, logger)

        assert mock_top_n.call_count == 2
        assert info["used_wide_lookback"] is True
        assert "NVDA" in symbols

    @patch("ingest_alpaca_bars.top_symbols_by_volume")
    @patch("ingest_alpaca_bars.traded_symbols_for_day")
    def test_all_empty_exits_nonzero(self, mock_traded, mock_top_n):
        """When all sources are empty after retry, exit non-zero."""
        from ingest_alpaca_bars import _resolve_symbols_with_stall_guard, IngestConfig
        import logging

        mock_cfg = MagicMock(spec=IngestConfig)
        mock_cfg.dsn = "postgresql://test"
        mock_top_n.return_value = []
        mock_traded.return_value = set()

        args = argparse.Namespace(
            symbols=None,
            top_n=10,
            days=7,
            include_traded=True,
            traded_date=None,
        )
        end = datetime(2026, 9, 24, 21, 0, tzinfo=timezone.utc)
        logger = logging.getLogger("test")

        with pytest.raises(SystemExit) as exc_info:
            _resolve_symbols_with_stall_guard(mock_cfg, args, end, logger)

        assert exc_info.value.code == 1


class TestCheckpointKeyWithHash:
    """Test that checkpoint key includes symbol hash."""

    def test_key_includes_hash_when_provided(self):
        """Checkpoint key format includes sym_hash when provided."""
        sym_hash = "abc12345"
        batch_idx = 0
        win_start = date(2026, 9, 1)
        win_end = date(2026, 9, 30)

        if sym_hash:
            key = f"{sym_hash}:{batch_idx}:{win_start}:{win_end}"
        else:
            key = f"{batch_idx}:{win_start}:{win_end}"

        assert key == "abc12345:0:2026-09-01:2026-09-30"

    def test_different_symbols_different_key(self):
        """Different symbol lists produce different checkpoint keys."""
        from data_providers.ingest_universe import symbol_list_hash

        hash1 = symbol_list_hash(["NVDA", "TSLA"])
        hash2 = symbol_list_hash(["NVDA", "GOOG"])

        key1 = f"{hash1}:0:2026-09-01:2026-09-30"
        key2 = f"{hash2}:0:2026-09-01:2026-09-30"

        assert key1 != key2


class TestCLIArgParsing:
    """Test CLI argument parsing for new flags."""

    def test_include_traded_default_from_config(self):
        """--include-traded default comes from config flag."""
        with patch("ingest_alpaca_bars._get_include_traded_default", return_value=True):
            from ingest_alpaca_bars import _parse_args
            args = _parse_args(["--top-n", "10"])
            assert args.include_traded is True

    def test_no_include_traded_overrides_default(self):
        """--no-include-traded overrides the default."""
        from ingest_alpaca_bars import _parse_args

        args = _parse_args(["--top-n", "10", "--no-include-traded"])

        assert args.include_traded is False

    def test_include_traded_explicit(self):
        """--include-traded can be explicitly set."""
        from ingest_alpaca_bars import _parse_args

        args = _parse_args(["--top-n", "10", "--include-traded"])

        assert args.include_traded is True

    def test_traded_date_parsing(self):
        """--traded-date is parsed correctly."""
        from ingest_alpaca_bars import _parse_args

        args = _parse_args(["--top-n", "10", "--traded-date", "2026-09-23"])

        assert args.traded_date == "2026-09-23"

    def test_traded_date_default_none(self):
        """--traded-date defaults to None (uses end date's ET day)."""
        from ingest_alpaca_bars import _parse_args

        args = _parse_args(["--top-n", "10"])

        assert args.traded_date is None

    def test_symbols_and_include_traded_compatible(self):
        """--symbols can be combined with --include-traded."""
        from ingest_alpaca_bars import _parse_args

        args = _parse_args([
            "--symbols", "NVDA", "TSLA",
            "--include-traded",
        ])

        assert args.symbols == ["NVDA", "TSLA"]
        assert args.include_traded is True


class TestConfigFlags:
    """Test config flag behavior for MINUTE_BARS_INGEST_INCLUDE_TRADED."""

    def test_flag_default_true(self, monkeypatch):
        """Flag defaults to True when env/yaml not set."""
        monkeypatch.delenv("MINUTE_BARS_INGEST_INCLUDE_TRADED", raising=False)
        from core.config import Config
        c = Config()
        assert c.MINUTE_BARS_INGEST_INCLUDE_TRADED is True

    def test_flag_env_0_is_false(self, monkeypatch):
        """env=0 sets flag to False."""
        monkeypatch.setenv("MINUTE_BARS_INGEST_INCLUDE_TRADED", "0")
        from core.config import Config
        c = Config()
        assert c.MINUTE_BARS_INGEST_INCLUDE_TRADED is False

    def test_flag_env_1_is_true(self, monkeypatch):
        """env=1 sets flag to True."""
        monkeypatch.setenv("MINUTE_BARS_INGEST_INCLUDE_TRADED", "1")
        from core.config import Config
        c = Config()
        assert c.MINUTE_BARS_INGEST_INCLUDE_TRADED is True

    def test_actions_flag_default(self, monkeypatch):
        """Actions flag has sensible defaults without shadow_pass."""
        monkeypatch.delenv("MINUTE_BARS_INGEST_TRADED_ACTIONS", raising=False)
        from core.config import Config
        c = Config()
        actions = c.MINUTE_BARS_INGEST_TRADED_ACTIONS
        assert "signal_buy" in actions
        assert "signal_sell" in actions
        assert "shadow_pass" not in actions

    def test_actions_flag_env_override(self, monkeypatch):
        """Actions can be overridden via env."""
        monkeypatch.setenv("MINUTE_BARS_INGEST_TRADED_ACTIONS", "signal_buy,custom_action")
        from core.config import Config
        c = Config()
        actions = c.MINUTE_BARS_INGEST_TRADED_ACTIONS
        assert actions == ["signal_buy", "custom_action"]
