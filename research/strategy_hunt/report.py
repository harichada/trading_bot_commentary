# research/strategy_hunt/report.py  (replaced in Task 22)
from __future__ import annotations
from pathlib import Path
from research.strategy_hunt.run import REPORTS_DIR


def write_report(root: Path, rows: list[dict], computed_at: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    p = REPORTS_DIR / f"{computed_at[:10]}-strategy-hunt.md"
    p.write_text("# Strategy hunt report (stub)\n")
    return p
