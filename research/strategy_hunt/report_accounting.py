"""The report's "Search accounting" section: how much searching bought how many PASS rows.

Extracted from `report.py` in Batch-2 Task F (that module is at its 400-line budget) when the
grid-point count learned about `--grid-pin`. The question this section answers is the
multiple-testing one -- families x grid combos x sides evaluated, next to the verdicts that
survived -- so the counts must describe the search that actually ran: a pinned axis was never
searched and must not be charged for.
"""
from __future__ import annotations

from typing import Optional

from research.strategy_hunt.walkforward.runner import pin_grid
from strategies.families import get_family
from strategies.families.common import grid_combos


def grid_pins_of(run_manifest: Optional[dict]) -> dict[str, str]:
    """The run's `--grid-pin` axes (Batch-2 Task F), `{}` when there were none or the manifest's
    section is malformed -- like every other optional section the report reads from disk."""
    pins = (run_manifest or {}).get("grid_pins")
    return {str(k): str(v) for k, v in pins.items()} if isinstance(pins, dict) else {}


def grid_pin_note(pins: dict[str, str]) -> str:
    """The report header's one-line statement of the pins. A pinned run is one arm of an A/B, not a
    search: every number in the report is conditional on it, so it belongs in the first line."""
    if not pins:
        return ""
    return (" Grid pins: " + ", ".join(f"{k}={v}" for k, v in sorted(pins.items()))
            + " (one arm of an A/B: every family with that axis searched only this value).")


def family_grid_points(name: str, pins: Optional[dict[str, str]] = None) -> tuple[int, int, int]:
    """(grid combos, sides, combos x sides) for one registered family, counted *after* pinning.

    `(0, 0, 0)` for a name the registry does not know -- a manifest written before that family was
    registered, or a hand-edited one -- so the table degrades instead of raising
    (strategy-search-batch1 task 2)."""
    try:
        fam = get_family(name)
    except KeyError:
        return 0, 0, 0
    try:
        grid = pin_grid(fam.grid, pins or {})
    except ValueError:                 # an unusable pin in a hand-edited manifest: report the search
        grid = fam.grid                # as it was defined rather than failing the whole report
    combos = len(grid_combos(grid))
    return combos, len(fam.sides), combos * len(fam.sides)


def searched_families(rows: list[dict], run_manifest: Optional[dict]) -> list[str]:
    """The families this search ran: the run manifest's `families` list when one is available
    (the authoritative record of what was actually searched, including families with zero PASS
    rows), else the families that appear in the unconditional (walkforward) rows themselves."""
    manifest_families = (run_manifest or {}).get("families")
    if manifest_families:
        return sorted(manifest_families)
    return sorted({r["strategy"] for r in rows if r.get("source") == "walkforward"})


def search_accounting_section(rows: list[dict], run_manifest: Optional[dict]) -> str:
    """Families x grid points x sides evaluated vs PASS/WATCH, so the multiple-testing burden
    of a search is visible next to its results -- not just the surviving PASS rows."""
    families = searched_families(rows, run_manifest)
    if not families:
        return "_no families run_"
    pins = grid_pins_of(run_manifest)
    wf = [r for r in rows if r.get("source") == "walkforward"]
    lines = ["| family | grid combos × sides | unconditional rows scored | PASS | WATCH |",
             "|---|---|---|---|---|"]
    total_points = total_rows = total_pass = total_watch = 0
    for fam in families:
        combos, sides, points = family_grid_points(fam, pins)
        fam_rows = [r for r in wf if r["strategy"] == fam]
        n_pass = sum(1 for r in fam_rows if r["verdict"] == "PASS")
        n_watch = sum(1 for r in fam_rows if r["verdict"] == "WATCH")
        lines.append(f"| {fam} | {combos} × {sides} = {points} | {len(fam_rows)} | {n_pass} | {n_watch} |")
        total_points += points
        total_rows += len(fam_rows)
        total_pass += n_pass
        total_watch += n_watch
    lines.append(f"| **total** | {total_points} | {total_rows} | {total_pass} | {total_watch} |")
    lines.append("")
    lines.append(f"{len(families)} families, {total_points} grid points × sides evaluated, {total_pass} PASS")
    return "\n".join(lines)
