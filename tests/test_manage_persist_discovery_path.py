"""Tests for v-manage-persist-discovery-path-2026-09-15.

Root Cause Analysis (RCA):
  After #67 merge, zero `sync_restore_success` / `update_track_restore` logs.
  
  api/routes.py mode toggle to LIVE (~line 157) called:
    await trading_engine._update_real_positions()
  
  core/engine.py `_update_real_positions` (~2416) created every missing Schwab
  symbol as managed_by_bot=False, is_external=True, is_manually_managed=True
  WITHOUT:
    - ENABLE_MANAGED_BY_BOT_PERSIST restore
    - HANDS_OFF denylist handling beyond default false
    - bot_trades / saved meta lookup
  
  Also called from analysis loop ~5010.
  
  Meanwhile restore-aware logic lived only in:
    - sync_positions_with_schwab
    - _update_and_track_real_positions
  
  Fix: Make _update_real_positions delegate to _update_and_track_real_positions
  so all discovery paths use restore-aware logic.

Test coverage:
  1. _update_real_positions delegates to _update_and_track_real_positions
  2. API mode-toggle path uses restore-aware function
  3. INFO log position_discovery_begin present in _update_and_track_real_positions
  4. Toggle API sets managed_source='operator_on' when enabling, 'operator' when disabling
  5. Denylist symbols still hands-off (HANDS_OFF_DENYLIST)
"""
import pytest
from pathlib import Path

ENGINE_PATH = Path('/workspace/core/engine.py')
ROUTES_PATH = Path('/workspace/api/routes.py')


class TestUpdateRealPositionsDelegation:
    """Tests for _update_real_positions delegating to _update_and_track_real_positions."""

    def test_update_real_positions_calls_update_and_track(self):
        """_update_real_positions must delegate to _update_and_track_real_positions."""
        src = ENGINE_PATH.read_text()
        
        # Find _update_real_positions method
        anchor = src.find("async def _update_real_positions(self):")
        assert anchor != -1, "_update_real_positions not found"
        
        # Get the method body (up to next method)
        body_start = anchor
        body_end = src.find("\n    async def ", anchor + 1)
        if body_end == -1:
            body_end = src.find("\n    def ", anchor + 1)
        body = src[body_start:body_end]
        
        # Must call _update_and_track_real_positions
        assert "await self._update_and_track_real_positions()" in body, (
            "_update_real_positions must delegate to _update_and_track_real_positions "
            "for restore-aware position discovery"
        )
    
    def test_update_real_positions_no_blind_external_stamp(self):
        """_update_real_positions must NOT blindly stamp positions as external."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _update_real_positions(self):")
        assert anchor != -1
        
        body_end = src.find("\n    async def ", anchor + 1)
        if body_end == -1:
            body_end = src.find("\n    def ", anchor + 1)
        body = src[anchor:body_end]
        
        # Should NOT have blind external stamping
        assert "is_external = True  # Flag as externally created" not in body, (
            "_update_real_positions should NOT blindly stamp is_external=True"
        )
        assert "External Position Detected (Manual Mode)" not in body, (
            "_update_real_positions should NOT emit 'External Position Detected' commentary "
            "for all discovered positions — that bypasses restore logic"
        )

    def test_update_real_positions_keeps_removal_logic(self):
        """_update_real_positions must still remove positions closed externally."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _update_real_positions(self):")
        assert anchor != -1
        
        body_end = src.find("\n    async def ", anchor + 1)
        if body_end == -1:
            body_end = src.find("\n    def ", anchor + 1)
        body = src[anchor:body_end]
        
        # Must still have removal logic
        assert "Remove positions that no longer exist in Schwab" in body or "positions_to_remove" in body, (
            "_update_real_positions must retain removal logic for externally closed positions"
        )
        assert "external_close" in body, (
            "_update_real_positions should record external_close reason for removed positions"
        )


class TestModeToggleUsesRestoreAwarePath:
    """Tests for API mode toggle using restore-aware discovery."""

    def test_mode_toggle_calls_update_and_track(self):
        """Mode toggle to LIVE should use restore-aware position discovery."""
        routes_src = ROUTES_PATH.read_text()
        engine_src = ENGINE_PATH.read_text()
        
        # Find the mode toggle code
        toggle_idx = routes_src.find("if new_mode == 'live':")
        assert toggle_idx != -1, "Mode toggle block not found"
        
        toggle_block = routes_src[toggle_idx:toggle_idx + 500]
        
        # Mode toggle calls _update_real_positions
        assert "_update_real_positions()" in toggle_block, (
            "Mode toggle should call _update_real_positions"
        )
        
        # And _update_real_positions now delegates to _update_and_track_real_positions
        anchor = engine_src.find("async def _update_real_positions(self):")
        body_end = engine_src.find("\n    async def ", anchor + 1)
        body = engine_src[anchor:body_end]
        
        assert "await self._update_and_track_real_positions()" in body, (
            "_update_real_positions must delegate to _update_and_track_real_positions"
        )


class TestDiscoveryInfoLog:
    """Tests for position_discovery_begin INFO log."""

    def test_update_and_track_has_discovery_begin_log(self):
        """_update_and_track_real_positions must log position_discovery_begin at start."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _update_and_track_real_positions(self):")
        assert anchor != -1
        
        body_end = src.find("\n    async def ", anchor + 1)
        if body_end == -1:
            body_end = src.find("\n    def ", anchor + 1)
        body = src[anchor:body_end]
        
        assert "position_discovery_begin" in body, (
            "_update_and_track_real_positions must log position_discovery_begin at start"
        )
        assert "n_schwab" in body, (
            "position_discovery_begin log must include n_schwab count"
        )
        assert "logger.info" in body and "position_discovery_begin" in body, (
            "position_discovery_begin must be an INFO-level log"
        )


class TestToggleAPIManagedSourceDirection:
    """Tests for toggle API setting managed_source based on toggle direction."""

    def test_toggle_off_sets_operator_source(self):
        """Toggle API sets managed_source='operator' when disabling."""
        src = ROUTES_PATH.read_text()
        
        # Find toggle-managed-by-bot endpoint
        toggle_idx = src.find('@app.post("/api/toggle-managed-by-bot")')
        assert toggle_idx != -1
        
        # Get endpoint body
        next_endpoint = src.find("@app.post", toggle_idx + 1)
        toggle_body = src[toggle_idx:next_endpoint]
        
        # Should set managed_source conditionally
        assert "managed_source = 'operator' if not new_state else 'operator_on'" in toggle_body, (
            "Toggle API should set managed_source='operator' when disabling (not new_state)"
        )

    def test_toggle_on_sets_operator_on_source(self):
        """Toggle API sets managed_source='operator_on' when enabling."""
        src = ROUTES_PATH.read_text()
        
        toggle_idx = src.find('@app.post("/api/toggle-managed-by-bot")')
        assert toggle_idx != -1
        
        next_endpoint = src.find("@app.post", toggle_idx + 1)
        toggle_body = src[toggle_idx:next_endpoint]
        
        # Should have operator_on for enabling
        assert "'operator_on'" in toggle_body, (
            "Toggle API should use 'operator_on' when enabling managed_by_bot"
        )

    def test_only_operator_blocks_restore(self):
        """Only managed_source='operator' (not 'operator_on') should block restore."""
        src = ENGINE_PATH.read_text()
        
        # Find restore logic in sync_positions_with_schwab
        sync_anchor = src.find("async def sync_positions_with_schwab")
        assert sync_anchor != -1
        sync_block = src[sync_anchor:sync_anchor + 10000]
        
        # operator_toggled_off only matches 'operator', not 'operator_on'
        assert "_saved_source == 'operator'" in sync_block, (
            "Restore logic should check for exact 'operator' source, not 'operator_on'"
        )
        
        # Find restore logic in _update_and_track_real_positions
        track_anchor = src.find("async def _update_and_track_real_positions")
        assert track_anchor != -1
        track_block = src[track_anchor:track_anchor + 15000]
        
        assert "_saved_source == 'operator'" in track_block, (
            "Restore logic should check for exact 'operator' source, not 'operator_on'"
        )


class TestDenylistStillHandsOff:
    """Tests for HANDS_OFF_DENYLIST still preventing auto-management."""

    def test_update_and_track_checks_denylist(self):
        """_update_and_track_real_positions must check HANDS_OFF_DENYLIST."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _update_and_track_real_positions")
        assert anchor != -1
        
        body_end = src.find("\n    async def ", anchor + 1)
        if body_end == -1:
            body_end = src.find("\n    def ", anchor + 1)
        body = src[anchor:body_end]
        
        assert "HANDS_OFF_DENYLIST" in body, (
            "_update_and_track_real_positions must check HANDS_OFF_DENYLIST"
        )
        assert "in_denylist" in body, (
            "_update_and_track_real_positions must have denylist check variable"
        )

    def test_denylist_symbols_never_restore(self):
        """Denylist symbols should NEVER get managed_by_bot=True from restore."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _update_and_track_real_positions")
        assert anchor != -1
        body = src[anchor:anchor + 15000]
        
        # Denylist check must come BEFORE restore check
        denylist_check_idx = body.find("in_denylist = _symbol_upper in denylist")
        restore_check_idx = body.find("if persist_enabled and not in_denylist")
        
        assert denylist_check_idx != -1, "Denylist check not found"
        assert restore_check_idx != -1, "Restore check not found"
        assert denylist_check_idx < restore_check_idx, (
            "Denylist check must come before restore logic"
        )
        
        # Restore logic gated on not in_denylist
        assert "persist_enabled and not in_denylist" in body, (
            "Restore logic must be gated on not in_denylist"
        )


class TestRestoreLogicCoverage:
    """Integration tests for restore logic in the discovery path."""

    def test_saved_positions_meta_checked(self):
        """_update_and_track_real_positions must check _saved_positions_meta."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _update_and_track_real_positions")
        assert anchor != -1
        body = src[anchor:anchor + 15000]
        
        assert "_saved_positions_meta" in body, (
            "_update_and_track_real_positions must check saved positions metadata"
        )

    def test_bot_trades_fallback_present(self):
        """_update_and_track_real_positions must have bot_trades fallback."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _update_and_track_real_positions")
        assert anchor != -1
        body = src[anchor:anchor + 15000]
        
        assert "bot_trades" in body.lower(), (
            "_update_and_track_real_positions must have bot_trades fallback"
        )
        assert "_restore_source = \"bot_trades\"" in body, (
            "_update_and_track_real_positions must set _restore_source for bot_trades path"
        )

    def test_restore_logs_present(self):
        """_update_and_track_real_positions must log successful restores."""
        src = ENGINE_PATH.read_text()
        
        anchor = src.find("async def _update_and_track_real_positions")
        assert anchor != -1
        # Function is large; read more
        body_end = src.find("\n    async def ", anchor + 1)
        if body_end == -1:
            body_end = src.find("\n    def ", anchor + 1)
        body = src[anchor:body_end]
        
        assert "update_track_restore" in body, (
            "_update_and_track_real_positions must log restore with 'update_track_restore'"
        )


class TestAnalysisLoopPath:
    """Tests for analysis loop calling restore-aware path."""

    def test_analysis_loop_uses_update_real_positions(self):
        """Analysis loop at ~5010 should use _update_real_positions (which now delegates)."""
        src = ENGINE_PATH.read_text()
        
        # Find the analysis loop section
        loop_marker = "await self._check_order_status()"
        loop_idx = src.find(loop_marker)
        assert loop_idx != -1, "Analysis loop section not found"
        
        # Check what's called after order status check
        loop_section = src[loop_idx:loop_idx + 200]
        
        assert "_update_real_positions()" in loop_section, (
            "Analysis loop should call _update_real_positions"
        )
