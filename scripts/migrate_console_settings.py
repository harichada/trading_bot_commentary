#!/usr/bin/env python3
"""v-risk-console-2026-09-28: env -> yaml migration for Risk Console knobs.

An env-set console knob is *env-shadowed* (env wins over yaml at read time,
see `Config()` properties), so a console edit silently does nothing until
the env line is removed. This is the one-time cutover: for each console
knob present in `.env`, validate it through the console's own rules
(`core.settings_console.coerce`) and print (dry-run) or write (`--apply`)
the equivalent yaml key, then comment out the migrated `.env` line.

Every knob with a non-empty `env_name` is reported as exactly one of
`migrate:`, `skipped (invalid):`, `left in .env:` (restart-scoped) or
`absent:`. `DESK_STREAK_SINCE` (not a console knob) is reported separately
when present. Nothing else from `.env` is ever printed.
Before any file is touched, the full set of planned changes is validated
against `ConfigManager`'s own runtime band (`_validate_config`) by loading a
throwaway copy of the candidate yaml -- a knob can individually validate
(e.g. `MAX_DAILY_LOSS` in `(0, 1]`) yet still fall outside the stricter
`[0.01, 0.20]` band `_validate_config` enforces. A band failure aborts with
no writes to either file (exit 3).
Usage:
    python scripts/migrate_console_settings.py --env .env --yaml "Config().yaml" [--apply]

Exit codes: 0 on success, 2 when no console knob is present in `.env`,
3 when the planned changes fail `ConfigManager`'s validation band.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Any

import yaml

from core.config import ConfigManager
from core.settings_console import (ALLOCATION_CEILING_PATH,
                                    DEFAULT_ALLOCATION_CEILING, KNOBS, coerce)
from scripts.migrate_console_env import (clean_value, line_ending,
                                          parse_env_lines, write_env_atomic)

_STREAK_NAME = "DESK_STREAK_SINCE"

class ValidationBandError(Exception):
    """The planned yaml changes fail `ConfigManager._validate_config`'s band."""

def _resolve_ceiling(found: dict[str, tuple[int, str]]) -> float:
    entry = found.get("BOT_CAPITAL_ALLOCATION")
    if entry is None:
        return DEFAULT_ALLOCATION_CEILING
    try:
        return float(clean_value(entry[1]))
    except ValueError:
        return DEFAULT_ALLOCATION_CEILING

def _classify_knobs(found: dict[str, tuple[int, str]], ceiling: float) -> dict[str, list]:
    """Classify every console knob against `found`, printing one diagnostic
    line per knob (and for `DESK_STREAK_SINCE`, when present). Returns the
    buckets needed to plan writes."""
    migrated: list[dict] = []
    skipped: list[dict] = []
    for knob in KNOBS:
        if not knob.env_name:
            continue
        entry = found.get(knob.env_name)
        if entry is None:
            print(f"absent: {knob.env_name}")
            continue
        idx, raw = entry
        if knob.restart_required:
            print(f"left in .env: {knob.env_name}={raw} (restart_required)")
            continue
        try:
            value = coerce(knob, clean_value(raw), ceiling=ceiling)
        except ValueError as exc:
            print(f"skipped (invalid): {knob.env_name}={raw} — {exc}")
            skipped.append({"knob": knob, "raw": raw, "reason": str(exc)})
            continue
        if knob.kind == "csv":
            value = list(value)
        print(f"migrate: {knob.env_name}={raw} → {knob.yaml_path}: {json.dumps(value)}")
        migrated.append({"knob": knob, "line_index": idx, "raw": raw, "value": value})
    streak_entry = found.get(_STREAK_NAME)
    if streak_entry is not None:
        print(f"left in .env: {_STREAK_NAME}={streak_entry[1]} (not a console knob)")
    return {"migrated": migrated, "skipped": skipped}

def _set_dotted(config: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        if not isinstance(node.get(key), dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value

def _validate_candidate(manager: ConfigManager, migrated: list[dict],
                         ceiling_to_write: float | None, yaml_path: str) -> None:
    """Raise `ValidationBandError` if applying `migrated` (+ the ceiling, if
    it's being added) would fail `ConfigManager._validate_config`. Never
    touches `yaml_path` itself -- validates a throwaway temp copy."""
    candidate = copy.deepcopy(manager.config)
    for item in migrated:
        _set_dotted(candidate, item["knob"].yaml_path, item["value"])
    if ceiling_to_write is not None:
        _set_dotted(candidate, ALLOCATION_CEILING_PATH, ceiling_to_write)
    tmp_path = f"{yaml_path}.validate_tmp_{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(candidate, fh)
        ConfigManager(config_path=tmp_path)
    except ValueError as exc:
        raise ValidationBandError(str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def _apply_changes(env_path: str, yaml_path: str, env_lines: list[str], migrated: list[dict],
                    ceiling_to_write: float | None, now: datetime,
                    manager: ConfigManager) -> dict[str, str]:
    _validate_candidate(manager, migrated, ceiling_to_write, yaml_path)

    timestamp = now.strftime("%Y%m%d_%H%M%S")
    env_backup = f"{env_path}.bak_console_{timestamp}"
    yaml_backup = f"{yaml_path}.bak_console_{timestamp}"
    shutil.copy2(env_path, env_backup)
    shutil.copy2(yaml_path, yaml_backup)

    for item in migrated:
        manager.update(item["knob"].yaml_path, item["value"])
    if ceiling_to_write is not None:
        manager.update(ALLOCATION_CEILING_PATH, ceiling_to_write)

    date_str = now.strftime("%Y-%m-%d")
    replacements = {
        item["line_index"]: (f"# migrated-to-yaml {date_str}: {item['knob'].env_name}="
                              f"{item['raw']}{line_ending(env_lines[item['line_index']])}")
        for item in migrated
    }
    write_env_atomic(env_path, env_lines, replacements)
    return {"env": env_backup, "yaml": yaml_backup}

def migrate(env_path: str, yaml_path: str, apply: bool, now: datetime) -> dict:
    """Migrate console knobs from `env_path` into `yaml_path`. Dry-run
    (`apply=False`) only prints; `apply=True` validates the full candidate
    against `ConfigManager`'s band first (no writes on failure), then backs
    up both files, writes the yaml, and comments out the migrated lines."""
    with open(env_path, "r", encoding="utf-8", newline="") as fh:
        env_lines = fh.readlines()
    found = parse_env_lines(env_lines)

    manager = ConfigManager(config_path=yaml_path)
    existing_ceiling = manager.get(ALLOCATION_CEILING_PATH, None)
    ceiling = existing_ceiling if existing_ceiling is not None else _resolve_ceiling(found)
    ceiling_will_write = existing_ceiling is None

    buckets = _classify_knobs(found, ceiling)
    migrated, skipped = buckets["migrated"], buckets["skipped"]
    tag = "migrate" if ceiling_will_write else "kept"
    print(f"{tag}: {ALLOCATION_CEILING_PATH}: {json.dumps(ceiling)}")

    result: dict[str, Any] = {
        "migrated": [{"env_name": m["knob"].env_name, "yaml_path": m["knob"].yaml_path,
                      "value": m["value"]} for m in migrated],
        "skipped_invalid": [{"env_name": s["knob"].env_name, "reason": s["reason"]}
                             for s in skipped],
        "ceiling": ceiling,
        "ceiling_written": False,
        "applied": False,
        "aborted": False,
        "abort_reason": "",
        "backups": None,
    }
    if apply and (migrated or ceiling_will_write):
        try:
            result["backups"] = _apply_changes(
                env_path, yaml_path, env_lines, migrated,
                ceiling if ceiling_will_write else None, now, manager)
        except ValidationBandError as exc:
            result["aborted"] = True
            result["abort_reason"] = str(exc)
            return result
        result["applied"] = True
        result["ceiling_written"] = ceiling_will_write
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

    if result["aborted"]:
        print(f"Aborted: validation band rejected the planned changes: {result['abort_reason']}")
        sys.exit(3)
    if not result["migrated"] and not result["skipped_invalid"] and not result["ceiling_written"]:
        print("Nothing to migrate: no console knob found in the env file.")
        sys.exit(2)
    if result["applied"]:
        print(f"Applied. Backups: {result['backups']['env']}, {result['backups']['yaml']}")
    else:
        print("Dry-run only. Re-run with --apply to write these changes.")
    sys.exit(0)

if __name__ == "__main__":
    main()
