#!/usr/bin/env python3
"""v-risk-console-2026-09-28: env -> yaml migration for Risk Console knobs.

An env-set console knob is *env-shadowed* (env wins over yaml at read time,
see `Config()` properties), so a console edit silently does nothing until
the env line is removed. This is the one-time cutover: for each console
knob present in `.env`, print (dry-run) or write (`--apply`) the equivalent
yaml key, then comment out the migrated `.env` line.

Not migrated: `FIT_LANES_ENABLED` (`restart_required=True`, console refuses
writes to it); `DESK_STREAK_SINCE` (not a console knob, read directly from
the environment by `core/loops/desk_loop.py`); `MAX_DAILY_LOSS` unless its
value parses as a fraction in `(0, 1]` (`.env` here once held a *dollar*
value, see `core/config.py::_env_fraction` -- migrating that literally would
set the daily-loss circuit to 500x equity).
Dry-run (default) prints only the knobs it would migrate and the
`trading.settings_console.allocation_ceiling` line -- never any other
`.env` line, so a secret sitting in `.env` is never echoed.
Usage:
    python scripts/migrate_console_settings.py --env .env --yaml "Config().yaml" [--apply]

Exit codes: 0 on success, 2 when no console knob is present in `.env`.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from typing import Any

from core.config import ConfigManager
from core.settings_console import (ALLOCATION_CEILING_PATH,
                                    DEFAULT_ALLOCATION_CEILING, KNOBS)

_ENV_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_TRUE_STRINGS = {"1", "true", "yes", "on"}
_LEFT_IN_ENV_ALWAYS = ("FIT_LANES_ENABLED", "DESK_STREAK_SINCE")

def _strip_quotes(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value

def _parse_env_lines(lines: list[str]) -> dict[str, tuple[int, str]]:
    """Return {NAME: (line_index, raw_value)} for simple `NAME=value` lines."""
    found: dict[str, tuple[int, str]] = {}
    for idx, line in enumerate(lines):
        stripped = line.rstrip("\r\n")
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        match = _ENV_LINE_RE.match(stripped)
        if match:
            found[match.group(1)] = (idx, match.group(2))
    return found

def _coerce_env_value(kind: str, raw: str) -> Any:
    value = _strip_quotes(raw)
    if kind == "bool":
        return value.strip().lower() in _TRUE_STRINGS
    if kind == "csv":
        return [item.strip() for item in value.split(",") if item.strip()]
    if kind == "int":
        return int(value.strip())
    if kind in ("money", "fraction"):
        return float(value.strip())
    raise ValueError(f"unsupported knob kind: {kind}")

def _max_daily_loss_eligible(raw: str) -> bool:
    try:
        value = float(_strip_quotes(raw))
    except ValueError:
        return False
    return 0 < value <= 1

def _find_migratable(found: dict[str, tuple[int, str]]) -> list[dict]:
    migrated = []
    for knob in KNOBS:
        if not knob.env_name or knob.restart_required:
            continue
        entry = found.get(knob.env_name)
        if entry is None:
            continue
        idx, raw = entry
        if knob.key == "max_daily_loss" and not _max_daily_loss_eligible(raw):
            continue
        try:
            value = _coerce_env_value(knob.kind, raw)
        except ValueError:
            continue
        migrated.append({"knob": knob, "line_index": idx, "raw": raw, "value": value})
    return migrated

def _resolve_ceiling(found: dict[str, tuple[int, str]]) -> float:
    entry = found.get("BOT_CAPITAL_ALLOCATION")
    if entry is None:
        return DEFAULT_ALLOCATION_CEILING
    try:
        return float(_strip_quotes(entry[1]))
    except ValueError:
        return DEFAULT_ALLOCATION_CEILING

def _print_report(migrated: list[dict], ceiling: float) -> None:
    for item in migrated:
        knob = item["knob"]
        print(f"{knob.env_name}={item['raw']} → {knob.yaml_path}: "
              f"{json.dumps(item['value'])}")
    print(f"{ALLOCATION_CEILING_PATH}: {json.dumps(ceiling)}")

def _apply_changes(env_path: str, yaml_path: str, env_lines: list[str],
                    migrated: list[dict], ceiling: float, now: datetime) -> dict[str, str]:
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    env_backup = f"{env_path}.bak_console_{timestamp}"
    yaml_backup = f"{yaml_path}.bak_console_{timestamp}"
    shutil.copy2(env_path, env_backup)
    shutil.copy2(yaml_path, yaml_backup)
    manager = ConfigManager(config_path=yaml_path)
    for item in migrated:
        manager.update(item["knob"].yaml_path, item["value"])
    manager.update(ALLOCATION_CEILING_PATH, ceiling)
    date_str = now.strftime("%Y-%m-%d")
    by_index = {item["line_index"]: item for item in migrated}
    new_lines = []
    for idx, line in enumerate(env_lines):
        item = by_index.get(idx)
        if item is None:
            new_lines.append(line)
            continue
        ending = "\n" if line.endswith("\n") else ""
        new_lines.append(
            f"# migrated-to-yaml {date_str}: {item['knob'].env_name}={item['raw']}{ending}"
        )
    with open(env_path, "w", encoding="utf-8") as fh:
        fh.writelines(new_lines)
    return {"env": env_backup, "yaml": yaml_backup}

def migrate(env_path: str, yaml_path: str, apply: bool, now: datetime) -> dict:
    """Migrate console knobs from `env_path` into `yaml_path`.

    Dry-run (`apply=False`) only prints; `apply=True` also backs up both
    files, writes the yaml, and comments out the migrated `.env` lines.
    Returns a summary dict; never touches any `.env` line that isn't a
    migrated console knob.
    """
    with open(env_path, "r", encoding="utf-8") as fh:
        env_lines = fh.readlines()
    found = _parse_env_lines(env_lines)
    migrated = _find_migratable(found)
    left_in_env = sorted(name for name in _LEFT_IN_ENV_ALWAYS if name in found)
    ceiling = _resolve_ceiling(found)
    _print_report(migrated, ceiling)
    result: dict[str, Any] = {
        "migrated": [
            {"env_name": item["knob"].env_name, "yaml_path": item["knob"].yaml_path,
             "value": item["value"]}
            for item in migrated
        ],
        "left_in_env": left_in_env,
        "ceiling": ceiling,
        "applied": False,
        "backups": None,
    }
    if apply and migrated:
        result["backups"] = _apply_changes(env_path, yaml_path, env_lines, migrated,
                                            ceiling, now)
        result["applied"] = True
    return result

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Migrate Risk Console knobs from .env into Config().yaml.")
    parser.add_argument("--env", default=".env", help="path to the .env file")
    parser.add_argument("--yaml", default="Config().yaml", help="path to the yaml config")
    parser.add_argument("--apply", action="store_true",
                         help="write the changes (default: dry-run only)")
    args = parser.parse_args(argv)

    result = migrate(args.env, args.yaml, apply=args.apply, now=datetime.now())

    if not result["migrated"]:
        print("Nothing to migrate: no console knob found in the env file.")
        sys.exit(2)

    if result["left_in_env"]:
        print("Left in .env (file-only / restart-scoped): "
              + ", ".join(result["left_in_env"]))

    if result["applied"]:
        print(f"Applied. Backups: {result['backups']['env']}, {result['backups']['yaml']}")
    else:
        print("Dry-run only. Re-run with --apply to write these changes.")

    sys.exit(0)

if __name__ == "__main__":
    main()
