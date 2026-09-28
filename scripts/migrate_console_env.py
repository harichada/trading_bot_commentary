"""v-risk-console-2026-09-28: low-level `.env` line helpers for
`scripts/migrate_console_settings.py`. Kept in its own module so the
orchestration script stays under its line budget.

Parsing is intentionally narrow: simple `[export ]NAME=value` lines only
(matching this repo's `.env` style, see `.env.example`). An unquoted value
may carry a trailing ` # comment`, which is preserved verbatim in
`raw` (used for diagnostics and the migration comment) but stripped, along
with any surrounding quotes, before a value is handed to
`core.settings_console.coerce`.
"""
from __future__ import annotations

import os
import re
import shutil

_ENV_LINE_RE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


def parse_env_lines(lines: list[str]) -> dict[str, tuple[int, str]]:
    """Return {NAME: (line_index, raw_value)} for simple `[export ]NAME=value`
    lines. `raw_value` is the untouched right-hand side (quotes and any
    inline comment intact)."""
    found: dict[str, tuple[int, str]] = {}
    for idx, line in enumerate(lines):
        stripped = line.rstrip("\r\n")
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        match = _ENV_LINE_RE.match(stripped)
        if match:
            found[match.group(1)] = (idx, match.group(2))
    return found


def clean_value(raw: str) -> str:
    """Strip surrounding quotes, or (when unquoted) a trailing ` # comment`."""
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    idx = value.find(" #")
    if idx != -1:
        value = value[:idx].rstrip()
    return value


def duplicate_indices(lines: list[str], name: str) -> list[int]:
    """Every line index declaring `name` (dotenv keeps the last one; the
    migration must comment out all of them so no stale earlier declaration
    survives as the only active one)."""
    out: list[int] = []
    for idx, line in enumerate(lines):
        stripped = line.rstrip("\r\n")
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        match = _ENV_LINE_RE.match(stripped)
        if match and match.group(1) == name:
            out.append(idx)
    return out


def line_ending(line: str) -> str:
    """The original line terminator of `line` (`\\r\\n`, `\\n`, `\\r`, or none)."""
    for ending in ("\r\n", "\n", "\r"):
        if line.endswith(ending):
            return ending
    return ""


def write_env_atomic(env_path: str, env_lines: list[str],
                      replacements: dict[int, str]) -> None:
    """Rewrite `env_path`, substituting `replacements[idx]` for `env_lines[idx]`
    (already carrying its own line ending) and copying every other line
    through byte-for-byte. Atomic: write `<path>.tmp`, copy the original file
    mode, then `os.replace`."""
    new_lines = [replacements.get(idx, line) for idx, line in enumerate(env_lines)]
    tmp_path = f"{env_path}.tmp"
    # Created owner-only from the first byte: the file holds secrets.
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
        fh.writelines(new_lines)
    shutil.copymode(env_path, tmp_path)
    os.replace(tmp_path, env_path)
