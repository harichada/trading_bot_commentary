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
