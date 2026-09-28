"""Tests for the Risk Console card on the dashboard Settings tab.

Covers: Task 5 of the 2026-09-28 risk console plan — markup/JS presence
checks against templates/dashboard.html, mirroring tests/test_activity_page.py's
string-assertion style for template tests.
"""
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "templates" / "dashboard.html"


def test_settings_tab_has_risk_console():
    html = TEMPLATE.read_text()
    settings = html[html.index('id="tab-settings"'):]
    for needle in ('id="risk-console-card"', 'id="rc-block-btn"', 'id="rc-status"', 'id="rc-audit"',
                   'function loadRiskConsole', 'function applyRiskKnob', 'function toggleBlockEntries',
                   "/api/console"):
        assert needle in html, needle
    assert 'loadRiskConsole()' in html[html.index("tabName === 'settings'"):][:300]


def test_risk_console_apply_and_cancel_use_data_attributes_not_inline_handlers():
    """Fix round 1: server-supplied key/id values must never be interpolated
    into an inline onclick="" JS-string-literal context (HTML-entity
    escaping does not protect that context) — they must flow through
    data-* attributes read by an addEventListener wired up in JS."""
    html = TEMPLATE.read_text()
    assert 'data-key=' in html
    assert 'data-pending-id=' in html
    assert 'onclick="applyRiskKnob(' not in html
    assert 'onclick="cancelRiskPending(' not in html
