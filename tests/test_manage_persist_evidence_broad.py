"""Tests for v-evidence-broad-2026-09-15: broadened ownership evidence.

Root Cause Analysis (RCA):
  After #68 merge, CRCL/SLS/FPS (mean-rev session entries) were left external
  on first boot without good saved state because:
    - bot_trades fallback was empty for mean-rev sessions (no bot_trades row
      until exit/close)
    - update_track path lacked INFO skip_no_evidence log (sync had it; 
      update_track did not)

Fix (this PR):
  1. INFO `update_track_skip_no_evidence` logged when non-denylist symbol left
     external after all evidence sources failed. Includes tried_sources.
  2. Broadened ownership evidence via `_ownership_evidence()` helper:
     - saved meta managed=True → bot_trades → bot_decisions → bot_positions
  3. ENABLE_MANAGED_OWNERSHIP_EVIDENCE_BROAD config flag (default True)
  4. HANDS_OFF_DENYLIST (MU, HQGE, SPCX) NEVER auto-managed
  5. Respect managed_source='operator' toggle off

Test coverage:
  1. skip_no_evidence logged in update_track path when evidence fails
  2. skip_no_evidence logged in sync path when evidence fails
  3. bot_decisions evidence restores mean-rev-like entries
  4. bot_positions evidence restores open-ledger symbols
  5. denylist symbols never auto-managed (regardless of evidence)
  6. operator toggle off respected (no evidence check)
  7. _ownership_evidence helper returns correct tried_sources
  8. Config flag ENABLE_MANAGED_OWNERSHIP_EVIDENCE_BROAD works
"""
import pytest
from pathlib import Path

ENGINE_PATH = Path('/workspace/core/engine.py')
CONFIG_PATH = Path('/workspace/core/config.py')
DB_LOGGER_PATH = Path('/workspace/data_providers/db_logger.py')


class TestSkipNoEvidenceLogging:
    """Tests for skip_no_evidence logging in both sync paths."""

    def test_sync_skip_no_evidence_log_present(self):
        """sync_positions_with_schwab must log sync_skip_no_evidence with tried_sources."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def sync_positions_with_schwab")
        assert anchor != -1
        body_end = src.find("\n    async def ", anchor + 1)
        body = src[anchor:body_end]
        
        assert "sync_skip_no_evidence" in body, (
            "sync_positions_with_schwab must log sync_skip_no_evidence"
        )
        assert "tried_sources" in body, (
            "sync_skip_no_evidence must include tried_sources"
        )

    def test_update_track_skip_no_evidence_log_present(self):
        """_update_and_track_real_positions must log update_track_skip_no_evidence."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _update_and_track_real_positions")
        assert anchor != -1
        body_end = src.find("\n    async def ", anchor + 1)
        if body_end == -1:
            body_end = src.find("\n    def ", anchor + 1)
        body = src[anchor:body_end]
        
        assert "update_track_skip_no_evidence" in body, (
            "_update_and_track_real_positions must log update_track_skip_no_evidence"
        )
        assert "tried_sources" in body, (
            "update_track_skip_no_evidence must include tried_sources"
        )

    def test_skip_no_evidence_logs_are_info_level(self):
        """skip_no_evidence logs must be INFO level."""
        src = ENGINE_PATH.read_text()
        
        # Check sync path
        sync_anchor = src.find("sync_skip_no_evidence")
        assert sync_anchor != -1
        sync_context = src[sync_anchor - 100:sync_anchor + 50]
        assert "logger.info" in sync_context, (
            "sync_skip_no_evidence must use logger.info"
        )
        
        # Check update_track path
        track_anchor = src.find("update_track_skip_no_evidence")
        assert track_anchor != -1
        track_context = src[track_anchor - 100:track_anchor + 50]
        assert "logger.info" in track_context, (
            "update_track_skip_no_evidence must use logger.info"
        )


class TestOwnershipEvidenceHelper:
    """Tests for _ownership_evidence helper method."""

    def test_ownership_evidence_helper_exists(self):
        """_ownership_evidence helper method must exist."""
        src = ENGINE_PATH.read_text()
        assert "async def _ownership_evidence" in src, (
            "_ownership_evidence helper method not found"
        )

    def test_ownership_evidence_returns_tried_sources(self):
        """_ownership_evidence must return tried_sources list."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _ownership_evidence")
        assert anchor != -1
        
        body_end = src.find("\n    async def ", anchor + 1)
        body = src[anchor:body_end]
        
        assert "tried_sources" in body, (
            "_ownership_evidence must track tried_sources"
        )
        assert "return" in body and "tried_sources" in body, (
            "_ownership_evidence must return tried_sources in tuple"
        )

    def test_ownership_evidence_checks_bot_decisions(self):
        """_ownership_evidence must check bot_decisions table."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _ownership_evidence")
        assert anchor != -1
        body_end = src.find("\n    async def ", anchor + 1)
        body = src[anchor:body_end]
        
        assert "bot_decisions" in body, (
            "_ownership_evidence must check bot_decisions"
        )
        assert "get_todays_bot_decision_entries" in body, (
            "_ownership_evidence must call get_todays_bot_decision_entries"
        )

    def test_ownership_evidence_checks_bot_positions(self):
        """_ownership_evidence must check bot_positions table."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _ownership_evidence")
        assert anchor != -1
        body_end = src.find("\n    async def ", anchor + 1)
        body = src[anchor:body_end]
        
        assert "bot_positions" in body, (
            "_ownership_evidence must check bot_positions"
        )
        assert "get_managed_bot_position" in body, (
            "_ownership_evidence must call get_managed_bot_position"
        )

    def test_ownership_evidence_order(self):
        """_ownership_evidence must check sources in correct order."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _ownership_evidence")
        assert anchor != -1
        body_end = src.find("\n    async def ", anchor + 1)
        body = src[anchor:body_end]
        
        # Find positions of each check
        saved_idx = body.find("saved.get('managed_by_bot')")
        bot_trades_idx = body.find("get_todays_bot_entries")
        bot_decisions_idx = body.find("get_todays_bot_decision_entries")
        bot_positions_idx = body.find("get_managed_bot_position")
        
        assert saved_idx != -1, "saved meta check not found"
        assert bot_trades_idx != -1, "bot_trades check not found"
        assert bot_decisions_idx != -1, "bot_decisions check not found"
        assert bot_positions_idx != -1, "bot_positions check not found"
        
        # Verify order: saved → bot_trades → bot_decisions → bot_positions
        assert saved_idx < bot_trades_idx < bot_decisions_idx < bot_positions_idx, (
            "Evidence sources must be checked in order: "
            "saved → bot_trades → bot_decisions → bot_positions"
        )


class TestBotDecisionsEvidence:
    """Tests for bot_decisions as ownership evidence source."""

    def test_db_logger_has_get_todays_bot_decision_entries(self):
        """DbLogger must have get_todays_bot_decision_entries method."""
        src = DB_LOGGER_PATH.read_text()
        
        assert "async def get_todays_bot_decision_entries" in src, (
            "DbLogger.get_todays_bot_decision_entries method not found"
        )

    def test_bot_decision_entries_queries_strategy_signals(self):
        """get_todays_bot_decision_entries must query for strategy entry signals."""
        src = DB_LOGGER_PATH.read_text()
        
        anchor = src.find("async def _get_todays_bot_decision_entries_impl")
        assert anchor != -1
        body_end = src.find("\n    async def ", anchor + 1)
        body = src[anchor:body_end]
        
        # Must query bot_decisions table
        assert "bot_decisions" in body, (
            "Must query bot_decisions table"
        )
        # Must filter for entry/open_position actions
        assert "entry" in body.lower() and "open_position" in body.lower(), (
            "Must filter for entry/open_position actions"
        )
        # Must filter for managed strategies
        assert "mean_reversion" in body, (
            "Must filter for mean_reversion strategy"
        )
        assert "day_trade_momentum" in body, (
            "Must filter for day_trade_momentum strategy"
        )


class TestBotPositionsEvidence:
    """Tests for bot_positions as ownership evidence source."""

    def test_db_logger_has_get_managed_bot_position(self):
        """DbLogger must have get_managed_bot_position method."""
        src = DB_LOGGER_PATH.read_text()
        
        assert "async def get_managed_bot_position" in src, (
            "DbLogger.get_managed_bot_position method not found"
        )

    def test_managed_bot_position_queries_with_strategy(self):
        """get_managed_bot_position must query for positions with strategy."""
        src = DB_LOGGER_PATH.read_text()
        
        anchor = src.find("async def _get_managed_bot_position_impl")
        assert anchor != -1
        body_end = src.find("\n    async def ", anchor + 1)
        body = src[anchor:body_end]
        
        # Must query bot_positions table
        assert "bot_positions" in body, (
            "Must query bot_positions table"
        )
        # Must filter for positions with strategy
        assert "strategy IS NOT NULL" in body, (
            "Must filter for positions with strategy"
        )


class TestDenylistNeverAutoManaged:
    """Tests for HANDS_OFF_DENYLIST never being auto-managed."""

    def test_denylist_checked_before_evidence(self):
        """Denylist must be checked before evidence lookup."""
        src = ENGINE_PATH.read_text()
        
        # Check sync_positions_with_schwab
        sync_anchor = src.find("async def sync_positions_with_schwab")
        assert sync_anchor != -1
        sync_body = src[sync_anchor:sync_anchor + 5000]
        
        # in_denylist must be checked
        assert "in_denylist" in sync_body, (
            "sync_positions_with_schwab must check in_denylist"
        )
        # persist_enabled and not in_denylist gates evidence check
        assert "persist_enabled and not in_denylist" in sync_body, (
            "Evidence check must be gated on not in_denylist"
        )

    def test_update_track_denylist_check(self):
        """_update_and_track_real_positions must check denylist."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _update_and_track_real_positions")
        assert anchor != -1
        body = src[anchor:anchor + 10000]
        
        assert "in_denylist" in body, (
            "_update_and_track_real_positions must check in_denylist"
        )
        assert "HANDS_OFF_DENYLIST" in body, (
            "_update_and_track_real_positions must reference HANDS_OFF_DENYLIST"
        )

    def test_sync_skip_denylist_log_present(self):
        """sync_positions_with_schwab must log sync_skip_denylist for denylist symbols."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def sync_positions_with_schwab")
        assert anchor != -1
        body_end = src.find("\n    async def ", anchor + 1)
        body = src[anchor:body_end]
        
        assert "sync_skip_denylist" in body, (
            "sync_positions_with_schwab must log sync_skip_denylist"
        )


class TestOperatorToggleRespected:
    """Tests for respecting operator toggle off."""

    def test_operator_toggle_off_skips_evidence(self):
        """managed_source='operator' with managed_by_bot=False must skip evidence check."""
        src = ENGINE_PATH.read_text()
        
        # Check sync_positions_with_schwab
        sync_anchor = src.find("async def sync_positions_with_schwab")
        assert sync_anchor != -1
        sync_body_end = src.find("\n    async def ", sync_anchor + 1)
        sync_body = src[sync_anchor:sync_body_end]
        
        assert "_operator_toggled_off" in sync_body, (
            "sync_positions_with_schwab must check _operator_toggled_off"
        )
        assert "_saved_source == 'operator'" in sync_body, (
            "Operator toggle check must verify _saved_source='operator'"
        )

    def test_update_track_operator_toggle(self):
        """_update_and_track_real_positions must respect operator toggle off."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _update_and_track_real_positions")
        assert anchor != -1
        body_end = src.find("\n    async def ", anchor + 1)
        if body_end == -1:
            body_end = src.find("\n    def ", anchor + 1)
        body = src[anchor:body_end]
        
        assert "_operator_toggled_off" in body, (
            "_update_and_track_real_positions must check _operator_toggled_off"
        )
        assert "update_track_skip_operator_toggle" in body, (
            "_update_and_track_real_positions must log update_track_skip_operator_toggle"
        )


class TestConfigFlag:
    """Tests for ENABLE_MANAGED_OWNERSHIP_EVIDENCE_BROAD config flag."""

    def test_config_flag_exists(self):
        """ENABLE_MANAGED_OWNERSHIP_EVIDENCE_BROAD config property must exist."""
        src = CONFIG_PATH.read_text()
        
        assert "ENABLE_MANAGED_OWNERSHIP_EVIDENCE_BROAD" in src, (
            "ENABLE_MANAGED_OWNERSHIP_EVIDENCE_BROAD config not found"
        )

    def test_config_flag_default_true(self):
        """ENABLE_MANAGED_OWNERSHIP_EVIDENCE_BROAD must default to True."""
        src = CONFIG_PATH.read_text()
        
        # Find the property definition
        anchor = src.find("def ENABLE_MANAGED_OWNERSHIP_EVIDENCE_BROAD")
        assert anchor != -1
        
        # Get the property body
        body_end = src.find("\n    @property", anchor + 1)
        if body_end == -1:
            body_end = src.find("\n    def ", anchor + 1)
        body = src[anchor:body_end]
        
        # Default must be True
        assert "True" in body and "enable_managed_ownership_evidence_broad" in body, (
            "ENABLE_MANAGED_OWNERSHIP_EVIDENCE_BROAD must default to True"
        )

    def test_ownership_evidence_uses_config_flag(self):
        """_ownership_evidence must respect ENABLE_MANAGED_OWNERSHIP_EVIDENCE_BROAD."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _ownership_evidence")
        assert anchor != -1
        body_end = src.find("\n    async def ", anchor + 1)
        body = src[anchor:body_end]
        
        assert "ENABLE_MANAGED_OWNERSHIP_EVIDENCE_BROAD" in body, (
            "_ownership_evidence must check ENABLE_MANAGED_OWNERSHIP_EVIDENCE_BROAD"
        )

    def test_broad_evidence_gated_by_flag(self):
        """bot_decisions/bot_positions checks must be gated by broad_enabled flag."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _ownership_evidence")
        assert anchor != -1
        body_end = src.find("\n    async def ", anchor + 1)
        body = src[anchor:body_end]
        
        # broad_enabled or similar flag must gate the extended checks
        assert "broad_enabled" in body, (
            "_ownership_evidence must use broad_enabled flag"
        )
        
        # get_todays_bot_decision_entries must be inside broad_enabled block
        broad_idx = body.find("broad_enabled")
        decisions_idx = body.find("get_todays_bot_decision_entries")
        positions_idx = body.find("get_managed_bot_position")
        
        assert decisions_idx > broad_idx, (
            "bot_decisions check must come after broad_enabled check"
        )
        assert positions_idx > broad_idx, (
            "bot_positions check must come after broad_enabled check"
        )


class TestSyncPathUsesOwnershipEvidence:
    """Tests for sync path using _ownership_evidence helper."""

    def test_sync_uses_ownership_evidence(self):
        """sync_positions_with_schwab must use _ownership_evidence helper."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def sync_positions_with_schwab")
        assert anchor != -1
        body_end = src.find("\n    async def ", anchor + 1)
        body = src[anchor:body_end]
        
        assert "_ownership_evidence" in body, (
            "sync_positions_with_schwab must call _ownership_evidence"
        )

    def test_update_track_uses_ownership_evidence(self):
        """_update_and_track_real_positions must use _ownership_evidence helper."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _update_and_track_real_positions")
        assert anchor != -1
        body_end = src.find("\n    async def ", anchor + 1)
        if body_end == -1:
            body_end = src.find("\n    def ", anchor + 1)
        body = src[anchor:body_end]
        
        assert "_ownership_evidence" in body, (
            "_update_and_track_real_positions must call _ownership_evidence"
        )


class TestDocstringsUpdated:
    """Tests for updated docstrings documenting new evidence sources."""

    def test_sync_docstring_mentions_broad_evidence(self):
        """sync_positions_with_schwab docstring must mention broad evidence sources."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def sync_positions_with_schwab")
        assert anchor != -1
        
        # Get docstring
        doc_start = src.find('"""', anchor)
        doc_end = src.find('"""', doc_start + 3)
        docstring = src[doc_start:doc_end + 3]
        
        assert "bot_decisions" in docstring or "evidence-broad" in docstring, (
            "sync_positions_with_schwab docstring must mention bot_decisions or evidence-broad"
        )

    def test_ownership_evidence_docstring(self):
        """_ownership_evidence must have docstring documenting sources."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _ownership_evidence")
        assert anchor != -1
        
        doc_start = src.find('"""', anchor)
        doc_end = src.find('"""', doc_start + 3)
        docstring = src[doc_start:doc_end + 3]
        
        assert "saved" in docstring.lower(), (
            "_ownership_evidence docstring must document saved meta source"
        )
        assert "bot_trades" in docstring, (
            "_ownership_evidence docstring must document bot_trades source"
        )
        assert "bot_decisions" in docstring, (
            "_ownership_evidence docstring must document bot_decisions source"
        )
        assert "bot_positions" in docstring, (
            "_ownership_evidence docstring must document bot_positions source"
        )
