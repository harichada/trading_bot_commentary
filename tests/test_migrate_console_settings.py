"""Tests for scripts/migrate_console_settings.py (env -> yaml Risk Console migration).

TDD for Task 6 of the Risk Console plan (.superpowers/sdd/2026-09-28-risk-console/
task-6-brief.md). Exercises `migrate()` directly against tmp_path fixtures only --
never touches the real .env / Config().yaml, and never spawns a subprocess.
"""
from __future__ import annotations

import glob
from datetime import datetime

import pytest
import yaml

from scripts.migrate_console_settings import main, migrate

ENV_CONTENT = (
    "BOT_CAPITAL_ALLOCATION=5000\n"
    "RISK_PER_TRADE_PCT=0.01\n"
    "MAX_DAILY_LOSS=0.03\n"
    'FIT_LANE_LIVE_ALLOWLIST="mr_20d:long,mom_12_1:long:risk_on|mixed"\n'
    "FIT_LANES_ENABLED=mr_20d,mom_12_1\n"
    "DESK_STREAK_SINCE=2026-09-28\n"
    "ORB_LIVE_ENTRIES_ENABLED=1\n"
    "TRADING_API_KEY=secret-xyz\n"
)

YAML_CONTENT = (
    "schwab: {}\n"
    "trading: {}\n"
    "commentary: {}\n"
    "technical_analysis: {}\n"
)

NOW = datetime(2026, 9, 28, 16, 5)


def _write_fixtures(tmp_path, env_content=ENV_CONTENT, yaml_content=YAML_CONTENT):
    env_path = tmp_path / ".env"
    yaml_path = tmp_path / "Config().yaml"
    env_path.write_text(env_content)
    yaml_path.write_text(yaml_content)
    return str(env_path), str(yaml_path)


def test_dry_run_mentions_knobs_and_never_leaks_other_env_lines(tmp_path, capsys):
    env_path, yaml_path = _write_fixtures(tmp_path)

    result = migrate(env_path, yaml_path, apply=False, now=NOW)

    captured = capsys.readouterr()
    assert "BOT_CAPITAL_ALLOCATION" in captured.out
    assert "secret-xyz" not in captured.out
    assert "TRADING_API_KEY" not in captured.out
    assert result["migrated"]

    # dry-run must not touch either file
    assert open(env_path, encoding="utf-8").read() == ENV_CONTENT
    assert open(yaml_path, encoding="utf-8").read() == YAML_CONTENT
    assert not glob.glob(env_path + ".bak_console_*")


def test_apply_migrates_yaml_and_comments_out_env_preserving_other_lines(tmp_path):
    env_path, yaml_path = _write_fixtures(tmp_path)

    result = migrate(env_path, yaml_path, apply=True, now=NOW)
    assert result["applied"] is True

    with open(yaml_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    trading = cfg["trading"]
    assert trading["bot_capital_allocation"] == 5000.0
    assert trading["fit_lane_live_allowlist"] == [
        "mr_20d:long", "mom_12_1:long:risk_on|mixed",
    ]
    assert trading["orb_live_entries_enabled"] is True
    assert trading["settings_console"]["allocation_ceiling"] == 5000.0

    env_text = open(env_path, encoding="utf-8").read()
    assert "FIT_LANES_ENABLED=mr_20d,mom_12_1" in env_text
    assert "DESK_STREAK_SINCE=2026-09-28" in env_text
    assert "TRADING_API_KEY=secret-xyz" in env_text

    for line in env_text.splitlines():
        if "BOT_CAPITAL_ALLOCATION" in line:
            assert line.startswith("# migrated-to-yaml")

    assert glob.glob(env_path + ".bak_console_*")
    assert glob.glob(yaml_path + ".bak_console_*")


def test_max_daily_loss_legacy_dollar_value_is_not_migrated(tmp_path):
    env_content = ENV_CONTENT.replace("MAX_DAILY_LOSS=0.03\n", "MAX_DAILY_LOSS=500\n")
    env_path, yaml_path = _write_fixtures(tmp_path, env_content=env_content)

    migrate(env_path, yaml_path, apply=True, now=NOW)

    with open(yaml_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert "max_daily_loss" not in cfg["trading"]
    env_text = open(env_path, encoding="utf-8").read()
    assert "MAX_DAILY_LOSS=500" in env_text


def test_exit_code_2_when_nothing_to_migrate(tmp_path, capsys):
    env_path = tmp_path / ".env"
    yaml_path = tmp_path / "Config().yaml"
    env_path.write_text("TRADING_API_KEY=secret-xyz\nDESK_STREAK_SINCE=2026-09-28\n")
    yaml_path.write_text(YAML_CONTENT)

    with pytest.raises(SystemExit) as exc_info:
        main(["--env", str(env_path), "--yaml", str(yaml_path)])
    assert exc_info.value.code == 2


def test_main_dry_run_exits_zero(tmp_path):
    env_path, yaml_path = _write_fixtures(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        main(["--env", env_path, "--yaml", yaml_path])
    assert exc_info.value.code == 0
    # dry-run (no --apply) must not have written anything
    assert not glob.glob(yaml_path + ".bak_console_*")
