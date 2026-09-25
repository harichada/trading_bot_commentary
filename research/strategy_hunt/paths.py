"""Variant-aware artefact paths under a strategy-hunt data root.

v-playbook-desk-b1-5 task 10. A *variant* is a control run that shares the root's inputs
(`bars5/`, `daily/`, `news/`, `context/`, `calibration.json`) but writes its own outputs:

    <root>/trades/<SYM>.parquet            variant None  — the real tape
    <root>/trades_shuffle/<SYM>.parquet    variant "shuffle" — the shuffled baseline

Before this, the shuffle control needed a whole separate root, which meant re-exporting
bars and made the baseline invisible to the scorer. Keeping it in-root is what lets
`score_regime_root` compute a same-regime baseline and require pooled rows to beat it.
The `--cost-mult` controls still use separate roots: they change the cost model, so their
`trades/` are a different measurement of the same tape rather than a control tape.

Split out of `run.py` so both the runner and the scorer share one definition of where a
variant's files live, and so `run.py` stays small.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

SHUFFLE_VARIANT = "shuffle"
# Every variant the pipeline knows. An unknown variant is a caller bug, not a new
# directory to silently create: a typo would write a control run nothing ever reads.
VARIANTS = (SHUFFLE_VARIANT,)


def variant_dir_name(kind: str, variant: Optional[str] = None) -> str:
    """`"trades"` for the real tape, `"trades_shuffle"` for the shuffle variant."""
    if variant is None:
        return kind
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected None or one of {VARIANTS}")
    return f"{kind}_{variant}"


def variant_dir(root: Path, kind: str, variant: Optional[str] = None) -> Path:
    return Path(root) / variant_dir_name(kind, variant)


def trades_dir(root: Path, variant: Optional[str] = None) -> Path:
    return variant_dir(root, "trades", variant)


def trades_path(root: Path, symbol: str, variant: Optional[str] = None) -> Path:
    return trades_dir(root, variant) / f"{symbol.upper()}.parquet"


def checkpoint_path(root: Path, symbol: str, variant: Optional[str] = None) -> Path:
    return variant_dir(root, "checkpoints", variant) / f"{symbol.upper()}.json"


def choices_path(root: Path, symbol: str, variant: Optional[str] = None) -> Path:
    return variant_dir(root, "choices", variant) / f"{symbol.upper()}.json"


def reports_dir(root: Path) -> Path:
    """Where a root's manifest, reports, and scorecards belong.

    v-playbook-desk-b1-5 task 12: `write_manifest` / `write_report` / `write_scorecard`
    used to write unconditionally into the committed `research/reports/strategy_hunt/`,
    keyed by date or run stamp. A control run on a different root (e.g.
    `--root data/strategy_hunt_cost2`) then overwrote the real run's manifest record and
    dropped its report/equity-CSV/json companion into the committed directory. Artefacts
    belong to the root they came from: the real run's root (`DEFAULT_ROOT`, resolved so a
    caller's relative/absolute spelling doesn't matter) still gets the committed dir; any
    other root gets `<root>/reports`.

    Imports `DEFAULT_ROOT` from `run.py` lazily (inside the function body, not at module
    load) because `run.py` imports this module at load time -- a module-level import here
    would be circular.
    """
    from research.strategy_hunt.run import DEFAULT_ROOT

    root = Path(root)
    if root.resolve() == DEFAULT_ROOT.resolve():
        return Path("research/reports/strategy_hunt")
    return root / "reports"
