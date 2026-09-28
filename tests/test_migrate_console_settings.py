"""Tests for scripts/migrate_console_settings.py (env -> yaml Risk Console migration).

TDD for Task 6 of the Risk Console plan (.superpowers/sdd/2026-09-28-risk-console/
task-6-brief.md, fix round 1). Exercises `migrate()` directly against tmp_path
fixtures only -- never touches the real .env / Config().yaml, and never spawns
a subprocess.
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
    # a registered-but-absent knob is reported, not silently dropped
    assert "absent: SWING_MAX_HOLD_SESSIONS" in captured.out

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

    result = migrate(env_path, yaml_path, apply=True, now=NOW)

    assert any(s["env_name"] == "MAX_DAILY_LOSS" for s in result["skipped_invalid"])
    with open(yaml_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert "max_daily_loss" not in cfg["trading"]
    env_text = open(env_path, encoding="utf-8").read()
    assert "MAX_DAILY_LOSS=500" in env_text


def test_exit_code_2_when_nothing_to_migrate(tmp_path):
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


def test_invalid_value_skipped_without_write(tmp_path):
    """An unrelated knob's bad value must not block, or be conflated with,
    a valid one -- and must never be written or have its .env line touched."""
    env_content = ENV_CONTENT.replace(
        "ORB_LIVE_ENTRIES_ENABLED=1\n", "ORB_LIVE_ENTRIES_ENABLED=maybe\n")
    env_path, yaml_path = _write_fixtures(tmp_path, env_content=env_content)

    result = migrate(env_path, yaml_path, apply=True, now=NOW)

    assert any(s["env_name"] == "ORB_LIVE_ENTRIES_ENABLED" for s in result["skipped_invalid"])
    with open(yaml_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert "orb_live_entries_enabled" not in cfg["trading"]
    assert cfg["trading"]["bot_capital_allocation"] == 5000.0  # unaffected sibling knob

    env_text = open(env_path, encoding="utf-8").read()
    assert "ORB_LIVE_ENTRIES_ENABLED=maybe" in env_text
    for line in env_text.splitlines():
        if "ORB_LIVE_ENTRIES_ENABLED" in line:
            assert not line.startswith("# migrated-to-yaml")


def test_validation_band_abort_exit_3_no_writes(tmp_path):
    """0.5 is a legal knob-level fraction (0, 1] but outside ConfigManager's
    own [0.01, 0.20] runtime band for max_daily_loss -- must abort before any
    write, leaving both files and backups untouched."""
    env_content = ENV_CONTENT.replace("MAX_DAILY_LOSS=0.03\n", "MAX_DAILY_LOSS=0.5\n")
    env_path, yaml_path = _write_fixtures(tmp_path, env_content=env_content)
    orig_env = open(env_path, encoding="utf-8").read()
    orig_yaml = open(yaml_path, encoding="utf-8").read()

    with pytest.raises(SystemExit) as exc_info:
        main(["--env", env_path, "--yaml", yaml_path, "--apply"])
    assert exc_info.value.code == 3

    assert open(env_path, encoding="utf-8").read() == orig_env
    assert open(yaml_path, encoding="utf-8").read() == orig_yaml
    assert not glob.glob(env_path + ".bak_console_*")
    assert not glob.glob(yaml_path + ".bak_console_*")
    assert not glob.glob(yaml_path + ".validate_tmp_*")


def test_apply_leaves_no_tmp_file(tmp_path):
    env_path, yaml_path = _write_fixtures(tmp_path)

    migrate(env_path, yaml_path, apply=True, now=NOW)

    assert not glob.glob(env_path + ".tmp")


def test_ceiling_preserved_on_second_apply(tmp_path):
    env_path, yaml_path = _write_fixtures(tmp_path)
    migrate(env_path, yaml_path, apply=True, now=NOW)
    with open(yaml_path, encoding="utf-8") as fh:
        assert yaml.safe_load(fh)["trading"]["settings_console"]["allocation_ceiling"] == 5000.0

    second_env = tmp_path / ".env2"
    second_env.write_text("DAY_TRADE_LIVE_ENTRIES_ENABLED=1\n")

    result = migrate(str(second_env), yaml_path, apply=True, now=NOW)
    assert result["ceiling_written"] is False

    with open(yaml_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["trading"]["settings_console"]["allocation_ceiling"] == 5000.0
    assert cfg["trading"]["day_trade_live_entries_enabled"] is True


def test_export_prefix_and_inline_comment(tmp_path):
    env_content = (
        "export ORB_LIVE_ENTRIES_ENABLED=1 # turn on ORB\n"
        "TRADING_API_KEY=secret-xyz\n"
    )
    env_path, yaml_path = _write_fixtures(tmp_path, env_content=env_content)

    result = migrate(env_path, yaml_path, apply=True, now=NOW)

    assert any(m["env_name"] == "ORB_LIVE_ENTRIES_ENABLED" for m in result["migrated"])
    with open(yaml_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["trading"]["orb_live_entries_enabled"] is True

    env_text = open(env_path, encoding="utf-8").read()
    assert "TRADING_API_KEY=secret-xyz" in env_text
    assert "# migrated-to-yaml 2026-09-28: export ORB_LIVE_ENTRIES_ENABLED=1 # turn on ORB" in env_text


def test_crlf_line_endings_preserved(tmp_path):
    env_content = "BOT_CAPITAL_ALLOCATION=5000\r\nTRADING_API_KEY=secret-xyz\r\n"
    env_path, yaml_path = _write_fixtures(tmp_path, env_content=env_content)

    migrate(env_path, yaml_path, apply=True, now=NOW)

    raw = open(env_path, "rb").read()
    assert b"# migrated-to-yaml 2026-09-28: BOT_CAPITAL_ALLOCATION=5000\r\n" in raw
    assert b"TRADING_API_KEY=secret-xyz\r\n" in raw


def test_duplicate_env_keys_all_commented_out(tmp_path):
    """dotenv keeps the last declaration; after migration no earlier duplicate
    may survive as the only active line."""
    from scripts.migrate_console_settings import migrate
    env = tmp_path / ".env"
    env.write_text("RISK_PER_TRADE_PCT=0.02\nOTHER=keep\nRISK_PER_TRADE_PCT=0.01\n")
    y = tmp_path / "c.yaml"
    y.write_text("schwab: {}\ntrading: {}\ncommentary: {}\ntechnical_analysis: {}\n")
    migrate(str(env), str(y), True, datetime(2026, 9, 28, 17, 0))
    lines = env.read_text().splitlines()
    assert "OTHER=keep" in lines
    assert not [l for l in lines if l.startswith("RISK_PER_TRADE_PCT=")]
    assert sum(l.startswith("# migrated-to-yaml") for l in lines) == 2
    import yaml
    assert yaml.safe_load(y.read_text())["trading"]["risk_per_trade_pct"] == 0.01


def test_apply_twice_is_a_noop_exit_2(tmp_path, capsys):
    from scripts.migrate_console_settings import main
    env = tmp_path / ".env"
    env.write_text("RISK_PER_TRADE_PCT=0.01\n")
    y = tmp_path / "c.yaml"
    y.write_text("schwab: {}\ntrading: {}\ncommentary: {}\ntechnical_analysis: {}\n")
    argv = ["--env", str(env), "--yaml", str(y), "--apply"]

    def _run():
        try:
            main(argv)
        except SystemExit as exc:
            return exc.code
        return 0

    assert _run() in (0, None)
    before = env.read_text()
    assert _run() == 2
    assert env.read_text() == before
