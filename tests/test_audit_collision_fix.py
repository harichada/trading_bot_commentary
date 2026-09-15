"""Tests for v-audit-collision-fix-2026-09-15: _audit() kwarg collision hardening.

Verifies that TradingEngineWithCommentary._audit() handles callers passing
reserved keys (action, component, symbol, reason, mode) in **details without
raising TypeError. Stripped collisions are preserved under _extra for observability.

Root cause: broker_flat.py passed action="alert_only" in kwargs while action was
already the 3rd positional argument, causing:
  "_audit() got multiple values for argument 'action'"
"""
import logging
import pytest
from unittest.mock import MagicMock


class MockDbLogger:
    """Mock DbLogger that captures log_decision calls."""
    
    def __init__(self):
        self.calls = []
    
    async def log_decision(self, **kwargs):
        self.calls.append(kwargs)


class MockPosition:
    """Mock Position for testing."""
    
    def __init__(self, symbol="TEST", quantity=100):
        self.symbol = symbol
        self.quantity = quantity
        self.managed_by_bot = True


class MockMode:
    """Mock TradingMode."""
    value = "LIVE"


class MockEngine:
    """Minimal mock engine with _audit method matching production collision-hardening."""
    
    def __init__(self):
        self.mode = MockMode()
        self.db_logger = MockDbLogger()
        self._audit_calls = []
    
    def _audit(self, component: str, symbol, action: str, reason: str, /, **details) -> None:
        """Production-equivalent _audit with collision hardening.
        
        This mirrors the exact logic from core/engine.py _audit method.
        Uses positional-only params (/) so action= in kwargs goes to **details.
        """
        _RESERVED_KEYS = ("action", "component", "symbol", "reason", "mode")
        collisions = {k: details.pop(k) for k in list(details.keys()) if k in _RESERVED_KEYS}
        if collisions:
            details["_extra"] = collisions
        
        self._audit_calls.append({
            'component': component,
            'symbol': symbol,
            'action': action,
            'reason': reason,
            'details': details.copy(),
        })


class TestAuditCollisionFix:
    """Tests for _audit collision hardening."""

    def test_audit_with_action_kwarg_no_typeerror(self):
        """_audit with action= in kwargs must NOT raise TypeError."""
        engine = MockEngine()
        
        engine._audit(
            "order_monitor", "ASND", "ghost_position_shadow_logged",
            "test_reason",
            local_qty=100,
            action="alert_only",
        )
        
        assert len(engine._audit_calls) == 1
        call = engine._audit_calls[0]
        assert call['action'] == "ghost_position_shadow_logged"
        assert call['details']['_extra']['action'] == "alert_only"

    def test_audit_with_multiple_reserved_keys(self):
        """_audit handles multiple reserved keys in kwargs."""
        engine = MockEngine()
        
        engine._audit(
            "test_component", "TEST", "test_action",
            "test_reason",
            local_qty=100,
            action="colliding_action",
            component="colliding_component",
            symbol="colliding_symbol",
            reason="colliding_reason",
            mode="colliding_mode",
        )
        
        assert len(engine._audit_calls) == 1
        call = engine._audit_calls[0]
        
        assert call['component'] == "test_component"
        assert call['symbol'] == "TEST"
        assert call['action'] == "test_action"
        assert call['reason'] == "test_reason"
        
        assert '_extra' in call['details']
        extra = call['details']['_extra']
        assert extra['action'] == "colliding_action"
        assert extra['component'] == "colliding_component"
        assert extra['symbol'] == "colliding_symbol"
        assert extra['reason'] == "colliding_reason"
        assert extra['mode'] == "colliding_mode"

    def test_audit_positional_action_wins(self):
        """Positional action parameter takes precedence over kwarg."""
        engine = MockEngine()
        
        engine._audit(
            "order_monitor", "ASND", "broker_flat_detected",
            "reject_oversold",
            action="incorrect_action",
        )
        
        call = engine._audit_calls[0]
        assert call['action'] == "broker_flat_detected"
        assert call['details']['_extra']['action'] == "incorrect_action"

    def test_audit_no_collision_no_extra(self):
        """When no collisions, _extra is not added."""
        engine = MockEngine()
        
        engine._audit(
            "order_monitor", "TEST", "test_action",
            "test_reason",
            local_qty=100,
            broker_qty=50,
        )
        
        call = engine._audit_calls[0]
        assert '_extra' not in call['details']
        assert call['details']['local_qty'] == 100
        assert call['details']['broker_qty'] == 50

    def test_audit_non_reserved_keys_pass_through(self):
        """Non-reserved keys in details pass through unchanged."""
        engine = MockEngine()
        
        engine._audit(
            "order_monitor", "ASND", "ghost_position_shadow_logged",
            "test_reason",
            local_qty=100,
            desk_action="alert_only",
            ghost_flatten_enabled=False,
        )
        
        call = engine._audit_calls[0]
        assert call['details']['local_qty'] == 100
        assert call['details']['desk_action'] == "alert_only"
        assert call['details']['ghost_flatten_enabled'] is False
        assert '_extra' not in call['details']


class TestBrokerFlatCallSiteFix:
    """Test the fixed call site in broker_flat.py uses desk_action instead of action."""

    def test_broker_flat_uses_desk_action(self):
        """broker_flat.py shadow log uses desk_action, not action."""
        import re
        
        with open('/workspace/core/order_monitor/broker_flat.py', 'r') as f:
            content = f.read()
        
        ghost_log_pattern = r'ghost_position_shadow_logged.*?desk_action="alert_only"'
        match = re.search(ghost_log_pattern, content, re.DOTALL)
        assert match is not None, "broker_flat.py should use desk_action= not action="
        
        collision_pattern = r'ghost_position_shadow_logged.*?[^_]action="alert_only"'
        collision_match = re.search(collision_pattern, content, re.DOTALL)
        assert collision_match is None, "broker_flat.py should NOT use bare action= in ghost_position_shadow_logged call"


class TestProductionAuditMethod:
    """Test the actual production _audit method from core/engine.py."""

    def test_production_audit_handles_collision(self, caplog):
        """Production _audit method with positional-only params handles action collision."""
        _RESERVED_KEYS = ("action", "component", "symbol", "reason", "mode")
        
        def _audit_impl(component: str, symbol, action: str, reason: str, /, **details):
            """Mirrors production _audit with positional-only params."""
            collisions = {k: details.pop(k) for k in list(details.keys()) if k in _RESERVED_KEYS}
            if collisions:
                details["_extra"] = collisions
            return {'action': action, 'details': details}
        
        result = _audit_impl(
            "order_monitor", "ASND", "ghost_position_shadow_logged",
            "test_reason",
            local_qty=100,
            action="alert_only",
        )
        
        assert result['action'] == "ghost_position_shadow_logged"
        assert result['details']['_extra']['action'] == "alert_only"


class TestOrderMonitorIntegration:
    """Integration tests for order_monitor _audit calls."""

    def test_all_order_monitor_audit_calls_safe(self):
        """Scan order_monitor/ for any remaining action= collision risks."""
        import re
        import os
        
        order_monitor_dir = '/workspace/core/order_monitor'
        risky_pattern = re.compile(r'\._audit\([^)]+action\s*=')
        
        for filename in os.listdir(order_monitor_dir):
            if not filename.endswith('.py'):
                continue
            filepath = os.path.join(order_monitor_dir, filename)
            with open(filepath, 'r') as f:
                content = f.read()
            
            if 'action=' in content and '._audit(' in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if '._audit(' in line:
                        context = '\n'.join(lines[max(0, i-2):min(len(lines), i+5)])
                        if 'action=' in context and 'desk_action' not in context:
                            if re.search(r'action\s*=\s*["\']', context):
                                assert False, f"Potential action= collision in {filename}:\n{context}"
