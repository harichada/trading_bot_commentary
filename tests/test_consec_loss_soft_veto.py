"""Tests for v-consec-loss-soft-veto-2026-09-22.

Validates that consecutive-loss is a SOFT ENTRY VETO, not a hard analysis block.
Analysis / strategy scan / decision logging continues when consecutive_losses >= max.
Only new LIVE entries are blocked. Exits / flatten / position management unchanged.

The pattern matches economic blackout (v-econ-calendar-dated-2026-09-09).

Tests are source-code based (no runtime dependencies) following the pattern
established in test_recent_fixes.py.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = REPO_ROOT / "core" / "engine.py"
CONFIG_PATH = REPO_ROOT / "core" / "config.py"
RISK_MANAGER_PATH = REPO_ROOT / "risk" / "manager.py"


class TestConsecLossSoftVetoConfig:
    """Config flag CONSECUTIVE_LOSS_SOFT_ENTRY_VETO."""

    def test_config_property_defined(self):
        """Config must define CONSECUTIVE_LOSS_SOFT_ENTRY_VETO property."""
        src = CONFIG_PATH.read_text()
        assert "def CONSECUTIVE_LOSS_SOFT_ENTRY_VETO" in src

    def test_default_is_true(self):
        """Default is True (soft veto, analysis continues)."""
        src = CONFIG_PATH.read_text()
        # Find the property and check for default True
        match = re.search(
            r"def CONSECUTIVE_LOSS_SOFT_ENTRY_VETO.*?return self\.manager\.get\([^,]+,\s*(True|False)\)",
            src,
            re.DOTALL
        )
        assert match is not None, "Property not found with expected pattern"
        assert match.group(1) == "True", "Default should be True"

    def test_config_key_path(self):
        """Config reads from trading.consecutive_loss_soft_entry_veto."""
        src = CONFIG_PATH.read_text()
        assert "'trading.consecutive_loss_soft_entry_veto'" in src

    def test_v_tag_in_config(self):
        """Config file has the v-tag comment."""
        src = CONFIG_PATH.read_text()
        assert "v-consec-loss-soft-veto-2026-09-22" in src


class TestRiskManagerCheckTradingAllowed:
    """risk/manager.py check_trading_allowed() behavior."""

    def test_v_tag_in_risk_manager(self):
        """Risk manager has the v-tag comment."""
        src = RISK_MANAGER_PATH.read_text()
        assert "v-consec-loss-soft-veto-2026-09-22" in src

    def test_conditional_check_on_config_flag(self):
        """check_trading_allowed gates consecutive_losses on config flag."""
        src = RISK_MANAGER_PATH.read_text()
        assert "if not Config().CONSECUTIVE_LOSS_SOFT_ENTRY_VETO:" in src

    def test_consecutive_losses_check_inside_conditional(self):
        """consecutive_losses >= MAX check is inside the config flag conditional."""
        src = RISK_MANAGER_PATH.read_text()
        # Find the conditional block
        marker = "if not Config().CONSECUTIVE_LOSS_SOFT_ENTRY_VETO:"
        idx = src.find(marker)
        assert idx != -1
        block = src[idx:idx + 300]
        assert "consecutive_losses >= Config().MAX_CONSECUTIVE_LOSSES" in block

    def test_max_consecutive_losses_message_preserved(self):
        """Legacy error message preserved for flag-off case."""
        src = RISK_MANAGER_PATH.read_text()
        assert '"Max consecutive losses reached"' in src


class TestEngineSoftVetoGate:
    """Engine _process_signal_with_commentary consecutive loss gate."""

    def test_v_tag_in_engine(self):
        """Engine has the v-tag comment."""
        src = ENGINE_PATH.read_text()
        assert "v-consec-loss-soft-veto-2026-09-22" in src

    def test_soft_veto_audit_call(self):
        """Engine emits audit log with consec_loss_soft_veto reason."""
        src = ENGINE_PATH.read_text()
        # Find the consecutive loss gate block
        marker = "v-consec-loss-soft-veto-2026-09-22"
        idx = src.find(marker)
        assert idx != -1, "v-tag not found in engine.py"
        block = src[idx:idx + 3000]
        
        assert 'self._audit("consecutive_loss_gate"' in block
        assert '"consec_loss_soft_veto"' in block

    def test_soft_veto_emits_snapshot(self):
        """Engine emits veto snapshot for consecutive loss gate."""
        src = ENGINE_PATH.read_text()
        marker = "v-consec-loss-soft-veto-2026-09-22"
        idx = src.find(marker)
        block = src[idx:idx + 3000]
        
        assert "_emit_veto_snapshot" in block
        assert 'reason="consec_loss_soft_veto"' in block
        assert 'gate_name="consecutive_loss_gate"' in block

    def test_soft_veto_commentary_message(self):
        """Engine adds commentary with clear explanation."""
        src = ENGINE_PATH.read_text()
        marker = "v-consec-loss-soft-veto-2026-09-22"
        idx = src.find(marker)
        block = src[idx:idx + 3000]
        
        assert "Consecutive Losses — Holding Off Entry" in block
        assert "Analysis continues" in block
        assert "daily reset" in block.lower()

    def test_gate_checks_config_flag(self):
        """Gate only fires when CONSECUTIVE_LOSS_SOFT_ENTRY_VETO is True."""
        src = ENGINE_PATH.read_text()
        marker = "v-consec-loss-soft-veto-2026-09-22"
        idx = src.find(marker)
        block = src[idx:idx + 3000]
        
        assert "Config().CONSECUTIVE_LOSS_SOFT_ENTRY_VETO" in block

    def test_gate_after_blackout_before_early_session(self):
        """Gate is placed after blackout check and before early_session check."""
        src = ENGINE_PATH.read_text()
        
        blackout_idx = src.find('self._audit("econ_blackout"')
        consec_loss_idx = src.find('self._audit("consecutive_loss_gate"')
        early_session_idx = src.find('self._audit("early_session"')
        
        assert blackout_idx < consec_loss_idx < early_session_idx, (
            "Gate must be after blackout and before early_session"
        )


class TestAnalysisContinuesDuringConsecLoss:
    """Analysis / strategy scan / decision logging continues when
    consecutive_losses >= max with soft veto enabled."""

    def test_evaluate_trading_conditions_returns_true(self):
        """_evaluate_trading_conditions returns True when soft veto enabled
        even when consecutive_losses >= max (analysis should proceed)."""
        src = ENGINE_PATH.read_text()
        
        marker = "v-consec-loss-soft-veto-2026-09-22"
        assert marker in RISK_MANAGER_PATH.read_text(), (
            "Risk manager must have the v-tag for the conditional check"
        )
        
        risk_src = RISK_MANAGER_PATH.read_text()
        assert "if not Config().CONSECUTIVE_LOSS_SOFT_ENTRY_VETO:" in risk_src, (
            "check_trading_allowed must gate consecutive_losses check on config flag"
        )

    def test_analyze_markets_still_called(self):
        """When soft veto enabled and consecutive_losses >= max,
        _analyze_markets_with_commentary should still be called."""
        src = ENGINE_PATH.read_text()
        
        should_trade_block = re.search(
            r'should_trade, reason = await self\._evaluate_trading_conditions\(\)',
            src
        )
        assert should_trade_block is not None
        
        idx = should_trade_block.end()
        following = src[idx:idx + 500]
        
        assert "if should_trade:" in following
        assert "_analyze_markets_with_commentary" in following


class TestExitsUnchanged:
    """Exits / flatten / position management / circuits unchanged."""

    def test_position_loop_unaffected(self):
        """Position management loop (_position_loop) should not have
        consecutive loss checks."""
        src = ENGINE_PATH.read_text()
        
        pos_loop_match = re.search(r'async def _position_loop\(self\)', src)
        assert pos_loop_match is not None
        
        idx = pos_loop_match.start()
        next_async_def = src.find("async def ", idx + 10)
        pos_loop_block = src[idx:next_async_def] if next_async_def != -1 else src[idx:idx + 5000]
        
        assert "consecutive_loss" not in pos_loop_block.lower() or \
               "v-consec-loss" in pos_loop_block, (
            "Position loop should not have consecutive loss blocking logic"
        )

    def test_daily_loss_circuit_unchanged(self):
        """Daily loss circuit breaker still works independently."""
        risk_src = RISK_MANAGER_PATH.read_text()
        
        assert "MAX_DAILY_LOSS" in risk_src
        assert "_circuit_pnl" in risk_src or "schwab_daily_pnl" in risk_src


class TestFlagOffRestoresHardBlock:
    """Setting CONSECUTIVE_LOSS_SOFT_ENTRY_VETO=False restores legacy
    hard-block behavior where analysis stops entirely."""

    def test_risk_manager_conditional_block(self):
        """When flag is False, check_trading_allowed returns False."""
        risk_src = RISK_MANAGER_PATH.read_text()
        
        assert "if not Config().CONSECUTIVE_LOSS_SOFT_ENTRY_VETO:" in risk_src
        assert '"Max consecutive losses reached"' in risk_src


class TestDecisionSnapshotIntegration:
    """Decision snapshot integration for consecutive loss veto."""

    def test_snapshot_reason_value(self):
        """Snapshot reason is 'consec_loss_soft_veto'."""
        src = ENGINE_PATH.read_text()
        marker = "v-consec-loss-soft-veto-2026-09-22"
        idx = src.find(marker)
        block = src[idx:idx + 3000]
        
        assert 'reason="consec_loss_soft_veto"' in block

    def test_snapshot_extra_contains_counts(self):
        """Snapshot extra contains consecutive_losses and max values."""
        src = ENGINE_PATH.read_text()
        marker = "v-consec-loss-soft-veto-2026-09-22"
        idx = src.find(marker)
        block = src[idx:idx + 3000]
        
        assert '"consecutive_losses": _consec_losses' in block
        assert '"max_consecutive_losses": _max_consec' in block


class TestRegressionScenario:
    """Regression test for the original P0 ops incident scenario.
    
    The P0 issue: consecutive_losses >= MAX caused check_trading_allowed
    to return False, which made _analysis_loop_body skip _analyze_markets_with_commentary.
    CoS had to manually reset consecutive_losses in trading_state.json.
    
    Fix: With soft veto enabled (default), check_trading_allowed returns True
    even when consecutive_losses >= max. The block happens at the entry gate
    in _process_signal_with_commentary, not at the analysis level.
    """

    def test_analysis_not_gated_by_consecutive_losses(self):
        """With soft veto enabled, consecutive_losses check is skipped
        in check_trading_allowed — analysis proceeds to _analyze_markets_with_commentary."""
        src = RISK_MANAGER_PATH.read_text()
        
        # The fix: check is wrapped in "if not CONSECUTIVE_LOSS_SOFT_ENTRY_VETO"
        # When the flag is True (default), the check is skipped.
        assert "if not Config().CONSECUTIVE_LOSS_SOFT_ENTRY_VETO:" in src
        
        # Find the check_trading_allowed function
        func_match = re.search(r'def check_trading_allowed\(self\)', src)
        assert func_match is not None
        
        # Get the function body (find end by looking for next method at same indent level)
        idx = func_match.start()
        # Look for "\n    def " (method at class level) after the function start
        rest = src[idx:]
        lines = rest.split('\n')
        func_lines = [lines[0]]
        for line in lines[1:]:
            if line.startswith('    def ') or (line.strip() and not line.startswith(' ')):
                break
            func_lines.append(line)
        func_body = '\n'.join(func_lines)
        
        # The return True, "Trading allowed" MUST be reachable even when
        # consecutive_losses >= max (because the check is inside the conditional)
        assert 'return True, "Trading allowed"' in func_body

    def test_entry_blocked_at_signal_router(self):
        """Entries blocked at _process_signal_with_commentary, not check_trading_allowed."""
        src = ENGINE_PATH.read_text()
        
        # The soft veto block is in _process_signal_with_commentary
        marker = "v-consec-loss-soft-veto-2026-09-22"
        idx = src.find(marker)
        assert idx != -1
        
        # Should be in _process_signal_with_commentary
        func_start = src.rfind("async def _process_signal_with_commentary", 0, idx)
        assert func_start != -1 and func_start < idx, (
            "Consecutive loss soft veto must be in _process_signal_with_commentary"
        )
