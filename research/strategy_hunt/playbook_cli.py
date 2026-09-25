"""v-playbook-desk-b2-2026-09-25 (Task 3, fix r1 finding 9): presentation for the
`python -m research.strategy_hunt playbook` subcommand.

Split out of `playbook_update.py` so both that module and `cli.py` stay inside the
400-line budget: everything here is printing and argument plumbing, and the module
holds no rules. `playbook_update` is imported as a module (not its functions) so a
test can monkeypatch `playbook_update.update_playbook` and see it take effect here.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional, Sequence

from core.desk.playbook import DEFAULT_DESK_DIR, Playbook, playbook_path

logger = logging.getLogger("strategy_hunt.playbook_cli")

__all__ = ["render_playbook", "run_playbook_command", "status_counts"]


def status_counts(playbook: Playbook) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in playbook.entries.values():
        counts[entry.status] = counts.get(entry.status, 0) + 1
    return counts


def render_playbook(playbook: Playbook, path: Any) -> str:
    """One line per key (worst status first) plus a summary line."""
    lines = [f"{e.family:18} {e.side:5} {e.regime:8} {e.status:9} x{e.size_mult:.2f} "
             f"n={e.n_recent:3} exp_r={e.exp_r_recent:+.3f} t={e.t_stat_recent:+.2f} {e.reason}"
             for e in sorted(playbook.entries.values(),
                             key=lambda e: (e.status, e.family, e.side, e.regime))]
    lines.append(f"playbook: {len(playbook.entries)} key(s) {status_counts(playbook)} -> {path}")
    return "\n".join(lines)


def run_playbook_command(args: Any, families: Optional[Sequence[str]] = None) -> int:
    """The `playbook` subcommand body (see `cli.cmd_playbook`).

    `--desk-dir` exists only on that subparser, so an `all` namespace falls back to the
    module default instead of raising AttributeError mid-pipeline. Without `--postgres`
    there is no live tape (engine=None) and the shadow ledgers alone drive the update.

    Engine construction is inside the try (fix r1, finding 5): a missing `POSTGRES_DSN`
    or an unparseable one is one clean line and rc=1, never a traceback in a nightly log.
    The engine is disposed before returning so the run leaves no open pool behind."""
    from research.strategy_hunt import playbook_update as PU
    from research.strategy_hunt.fit.writer import load_snapshot

    desk_dir = getattr(args, "desk_dir", None) or DEFAULT_DESK_DIR
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    engine = None
    try:
        if getattr(args, "postgres", False):
            from sqlalchemy import create_engine

            dsn = os.environ.get("POSTGRES_DSN")
            if not dsn:
                raise RuntimeError("--postgres needs POSTGRES_DSN in the environment")
            engine = create_engine(dsn)
        playbook = PU.update_playbook(args.root, load_snapshot(args.root), now, desk_dir=desk_dir,
                                      families=families, engine=engine)
    except Exception as exc:
        print(f"playbook ERROR {exc}")
        return 1
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception as exc:                        # pragma: no cover - defensive
                logger.warning("playbook: engine dispose failed: %s", exc)
    print(render_playbook(playbook, playbook_path(desk_dir)))
    return 0
