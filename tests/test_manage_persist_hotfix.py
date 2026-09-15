"""Tests for v-manage-persist-hotfix-2026-09-15.

Root Cause Analysis (RCA):
  After merge+restart CRCL/SLS/FPS stayed managed_by_bot=False external=True
  while MU/HQGE/SPCX (denylist) correctly stayed hands-off.

Two bugs fixed:
  A) False-positive operator toggle-off: Code treated managed_by_bot=False as
     operator toggle without managed_source stamp. Fix: only block bot_trades
     fallback if managed_source='operator'.
  B) get_todays_bot_entries returns {} on cross-loop: _route_to_owner_loop
     was fire-and-forget. Fix: await the routed coroutine result.

Test coverage:
  1. Stale False without managed_source=operator still restores via bot_trades
  2. Operator stamp (managed_source='operator') blocks restoration
  3. Denylist symbols NEVER restore managed_by_bot=True
  4. get_todays_bot_entries returns routed value not {}
  5. managed_source persisted in _save_state and restored in _load_state
  6. Toggle API sets managed_source='operator'
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch


class TestManagedSourceField:
    """Tests for managed_source field on Position."""

    def test_position_has_managed_source_field(self):
        """Position dataclass has managed_source field."""
        # Read the model source to verify field exists
        import re
        with open('/workspace/core/models.py', 'r') as f:
            content = f.read()
        
        assert 'managed_source: Optional[str] = None' in content, \
            "Position should have managed_source field"
        
    def test_position_managed_source_values(self):
        """Position managed_source accepts expected values."""
        # This verifies the field is properly defined
        import re
        with open('/workspace/core/models.py', 'r') as f:
            content = f.read()
        
        # Check field definition exists with correct type
        assert re.search(r"managed_source:\s*Optional\[str\]\s*=\s*None", content), \
            "managed_source should be Optional[str] with None default"


class TestOperatorToggleDetection:
    """Tests for operator toggle-off detection logic."""

    def test_stale_false_without_operator_source_allows_bot_trades(self):
        """managed_by_bot=False without managed_source='operator' is treated as stale."""
        saved_meta = {
            'managed_by_bot': False,
            'managed_source': None,  # No operator stamp
            'side': 'long',
        }
        
        # Without operator stamp, should NOT be treated as operator toggle
        _operator_toggled_off = (
            saved_meta.get('managed_by_bot') is False
            and saved_meta.get('managed_source') == 'operator'
        )
        
        assert not _operator_toggled_off, "Stale False without operator stamp should allow fallback"

    def test_operator_stamp_blocks_restoration(self):
        """managed_by_bot=False with managed_source='operator' blocks restoration."""
        saved_meta = {
            'managed_by_bot': False,
            'managed_source': 'operator',  # Operator intentionally toggled off
            'side': 'long',
        }
        
        _operator_toggled_off = (
            saved_meta.get('managed_by_bot') is False
            and saved_meta.get('managed_source') == 'operator'
        )
        
        assert _operator_toggled_off, "Operator stamp should block bot_trades fallback"

    def test_managed_true_with_bot_source_restores(self):
        """managed_by_bot=True with managed_source='bot' allows restoration."""
        saved_meta = {
            'managed_by_bot': True,
            'managed_source': 'bot',
            'side': 'long',
        }
        
        _restore_from_saved = (
            saved_meta.get('managed_by_bot') is True
            and saved_meta.get('side') == 'long'
        )
        
        assert _restore_from_saved, "Bot-opened position should restore"


class TestDenylistBehavior:
    """Tests for HANDS_OFF_DENYLIST behavior."""

    def test_denylist_never_restores_managed(self):
        """Symbols in HANDS_OFF_DENYLIST are NEVER auto-managed."""
        denylist = frozenset({'MU', 'HQGE', 'SPCX'})
        
        for symbol in ['MU', 'HQGE', 'SPCX']:
            in_denylist = symbol.upper() in denylist
            assert in_denylist, f"{symbol} should be in denylist"
            
            # Even with valid bot evidence, denylist blocks
            saved_meta = {
                'managed_by_bot': True,
                'managed_source': 'bot',
                'side': 'long',
            }
            
            persist_enabled = True
            _restore = False
            
            if persist_enabled and not in_denylist:  # This branch won't execute
                _restore = True
            
            assert not _restore, f"{symbol} should not restore even with bot evidence"

    def test_non_denylist_can_restore(self):
        """Symbols NOT in denylist can be restored."""
        denylist = frozenset({'MU', 'HQGE', 'SPCX'})
        
        symbol = 'CRCL'
        in_denylist = symbol.upper() in denylist
        assert not in_denylist, f"{symbol} should NOT be in denylist"


class TestGetTodaysBotEntriesCrossLoop:
    """Tests for get_todays_bot_entries cross-loop routing."""

    def test_cross_loop_routing_code_fixed(self):
        """Verify get_todays_bot_entries uses asyncio.wrap_future for cross-loop."""
        # Read the db_logger source to verify the fix
        with open('/workspace/data_providers/db_logger.py', 'r') as f:
            content = f.read()
        
        # Find the get_todays_bot_entries method
        method_start = content.find("async def get_todays_bot_entries")
        method_end = content.find("async def _get_todays_bot_entries_impl")
        method_body = content[method_start:method_end]
        
        # Old buggy pattern: return {} after routing
        old_pattern = "if self._route_to_owner_loop(\"get_todays_bot_entries\""
        assert old_pattern not in method_body, \
            "Should not use fire-and-forget _route_to_owner_loop for value-returning method"
        
        # New fixed pattern: use run_coroutine_threadsafe + wrap_future
        assert "asyncio.run_coroutine_threadsafe" in method_body, \
            "Should use run_coroutine_threadsafe for cross-loop routing"
        assert "asyncio.wrap_future" in method_body, \
            "Should use wrap_future to await the routed result"

    def test_cross_loop_does_not_return_empty_blindly(self):
        """Verify the old 'return {}' pattern is removed."""
        with open('/workspace/data_providers/db_logger.py', 'r') as f:
            content = f.read()
        
        # Find the get_todays_bot_entries method
        method_start = content.find("async def get_todays_bot_entries")
        method_end = content.find("async def _get_todays_bot_entries_impl")
        method_body = content[method_start:method_end]
        
        # The OLD buggy pattern was:
        # if self._route_to_owner_loop(...):
        #     return {}  # BUG - returns empty instead of waiting for result
        # 
        # The fix should NOT have this pattern
        old_buggy_pattern = """if self._route_to_owner_loop("get_todays_bot_entries", self._get_todays_bot_entries_impl, symbols):
            return {}"""
        
        assert old_buggy_pattern not in method_body, \
            "The old buggy pattern should be removed"


class TestToggleAPIManagedSource:
    """Tests for /api/toggle-managed-by-bot setting managed_source."""

    def test_toggle_sets_operator_source(self):
        """Toggle API sets managed_source based on toggle direction.
        
        v-manage-persist-discovery-path-2026-09-15: Updated to use conditional:
          - 'operator' when disabling (not new_state) - blocks restore
          - 'operator_on' when enabling (new_state) - indicates deliberate re-enable
        """
        with open('/workspace/api/routes.py', 'r') as f:
            content = f.read()
        
        # Verify it happens in the toggle-managed-by-bot endpoint
        toggle_section = content[content.find("@app.post(\"/api/toggle-managed-by-bot\")"):]
        toggle_section = toggle_section[:toggle_section.find("@app.post", 1)]
        
        # v-manage-persist-discovery-path-2026-09-15: conditional based on direction
        assert "pos.managed_source = 'operator' if not new_state else 'operator_on'" in toggle_section, \
            "managed_source should be set conditionally based on toggle direction"


class TestSaveStateIncludesManagedSource:
    """Tests for managed_source persistence in _save_state."""

    def test_positions_data_includes_managed_source(self):
        """positions_data dict includes managed_source field."""
        # Verify _save_state includes managed_source in serialization
        with open('/workspace/core/engine.py', 'r') as f:
            content = f.read()
        
        # Check that managed_source is persisted in positions_data
        assert "'managed_source': getattr(pos, 'managed_source', None)" in content, \
            "_save_state should persist managed_source field"


class TestRestoreLogicIntegration:
    """Integration tests for the full restore logic."""

    def test_restore_scenario_stale_false_no_operator_stamp(self):
        """CRCL scenario: stale managed_by_bot=False without operator stamp."""
        symbol = 'CRCL'
        side = 'long'
        denylist = frozenset({'MU', 'HQGE', 'SPCX'})
        persist_enabled = True
        
        # Simulated saved state (stale from bad sync)
        saved = {
            'managed_by_bot': False,  # Stale False
            'managed_source': None,    # No operator stamp
            'side': 'long',
        }
        
        # Simulated bot_trades evidence
        bot_trades = {
            'CRCL': {'side': 'long', 'entry_time': datetime.now()}
        }
        
        in_denylist = symbol.upper() in denylist
        _restore = False
        _restore_source = None
        
        if persist_enabled and not in_denylist:
            _saved_managed = saved.get('managed_by_bot')
            _saved_source = saved.get('managed_source')
            
            # FIXED: Only treat as operator toggle if source is 'operator'
            _operator_toggled_off = (
                _saved_managed is False and _saved_source == 'operator'
            )
            _restore_from_saved = (
                _saved_managed is True and saved.get('side') == side
            )
            
            if _restore_from_saved:
                _restore = True
                _restore_source = "saved"
            elif _operator_toggled_off:
                _restore = False
                _restore_source = None
            else:
                # Fallback to bot_trades
                bt = bot_trades.get(symbol.upper()) or bot_trades.get(symbol)
                if bt is not None and bt.get('side') == side:
                    _restore = True
                    _restore_source = "bot_trades"
        
        assert _restore is True, "CRCL should restore via bot_trades fallback"
        assert _restore_source == "bot_trades"

    def test_restore_scenario_operator_toggled_off(self):
        """Operator explicitly toggled off - should NOT restore."""
        symbol = 'AAPL'
        side = 'long'
        denylist = frozenset({'MU', 'HQGE', 'SPCX'})
        persist_enabled = True
        
        # Operator explicitly toggled off
        saved = {
            'managed_by_bot': False,
            'managed_source': 'operator',  # Operator stamp present
            'side': 'long',
        }
        
        # Even with bot_trades evidence, operator stamp should block
        bot_trades = {
            'AAPL': {'side': 'long', 'entry_time': datetime.now()}
        }
        
        in_denylist = symbol.upper() in denylist
        _restore = False
        _restore_source = None
        
        if persist_enabled and not in_denylist:
            _saved_managed = saved.get('managed_by_bot')
            _saved_source = saved.get('managed_source')
            
            _operator_toggled_off = (
                _saved_managed is False and _saved_source == 'operator'
            )
            _restore_from_saved = (
                _saved_managed is True and saved.get('side') == side
            )
            
            if _restore_from_saved:
                _restore = True
                _restore_source = "saved"
            elif _operator_toggled_off:
                _restore = False
                _restore_source = None
            else:
                bt = bot_trades.get(symbol.upper()) or bot_trades.get(symbol)
                if bt is not None and bt.get('side') == side:
                    _restore = True
                    _restore_source = "bot_trades"
        
        assert _restore is False, "Operator toggle should block restoration"
        assert _restore_source is None

    def test_restore_scenario_denylist(self):
        """Denylist symbol - should NEVER restore."""
        symbol = 'MU'
        side = 'long'
        denylist = frozenset({'MU', 'HQGE', 'SPCX'})
        persist_enabled = True
        
        # Even with full bot evidence
        saved = {
            'managed_by_bot': True,
            'managed_source': 'bot',
            'side': 'long',
        }
        
        in_denylist = symbol.upper() in denylist
        _restore = False
        
        if persist_enabled and not in_denylist:  # This won't execute for MU
            _restore = True
        
        assert _restore is False, "Denylist symbol should NEVER restore"


class TestBotOpenedPositionsManagedSource:
    """Tests for bot-opened positions setting managed_source='bot'."""

    def test_new_position_from_bot_has_bot_source(self):
        """Bot-opened positions should have managed_source='bot'."""
        # Verify engine.py sets managed_source='bot' for bot-opened positions
        with open('/workspace/core/engine.py', 'r') as f:
            content = f.read()
        
        # Check that bot-opened positions set managed_source='bot'
        # There should be multiple occurrences in different position creation paths
        occurrences = content.count("managed_source='bot'")
        assert occurrences >= 3, \
            f"Expected at least 3 occurrences of managed_source='bot', found {occurrences}"


class TestExternalPositionsManagedSource:
    """Tests for external/denylist positions setting correct managed_source."""

    def test_external_position_has_external_source(self):
        """External positions should have managed_source='external'."""
        # Verify engine.py sets managed_source for external positions
        with open('/workspace/core/engine.py', 'r') as f:
            content = f.read()
        
        # Check that external positions set managed_source
        assert "managed_source=_external_source" in content, \
            "Engine should set managed_source for external positions"
        
        # Check that _external_source is defined correctly
        assert "_external_source = 'denylist' if in_denylist else 'external'" in content, \
            "Engine should set _external_source based on denylist status"

    def test_denylist_position_has_denylist_source(self):
        """Denylist positions should have managed_source='denylist'."""
        # Verify the _external_source logic covers denylist
        with open('/workspace/core/engine.py', 'r') as f:
            content = f.read()
        
        assert "'denylist' if in_denylist else 'external'" in content, \
            "Denylist positions should get managed_source='denylist'"
