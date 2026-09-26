"""Subparser registration for the strategy-hunt CLI.

Extracted from `cli.py` in Batch-2 Task F so that module stays inside the 400-line budget while
gaining the `splits-audit` subcommand: `cli.py` owns the command *bodies*, `cli_args.py` owns the
flags every pipeline subcommand shares, and this module owns the parser tree that binds the two.

`build_parser` takes the handlers as an argument rather than importing them, because `cli.py`
imports this module -- passing them in is what keeps that a one-way dependency.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Mapping

from core.desk.playbook import DEFAULT_DESK_DIR as PLAYBOOK_DESK_DIR
from research.strategy_hunt.cli_args import add_common_args
from research.strategy_hunt.run import DEFAULT_ROOT

# The subcommands that share the whole pipeline flag surface (`cli_args.add_common_args`). `all`
# is the one that must refuse a user-set --shuffle-seed/--shuffle-mode: it runs the controls itself.
PIPELINE_COMMANDS = ("export", "context", "earnings", "calibrate", "run", "score", "report", "all")


def build_parser(handlers: Mapping[str, Callable[[argparse.Namespace], int]]) -> argparse.ArgumentParser:
    """The full parser. `handlers` maps each subcommand name to the function that runs it; a name
    this module does not know about is a KeyError here rather than a subcommand that silently
    disappears from the CLI."""
    p = argparse.ArgumentParser(prog="python -m research.strategy_hunt")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in PIPELINE_COMMANDS:
        sp = sub.add_parser(name)
        add_common_args(sp, DEFAULT_ROOT, user_shuffle_seed=(name != "all"))
        sp.set_defaults(fn=handlers[name])

    # v-playbook-desk-b1-2026-09-25 (Task 7b): `news` gets its own subparser (like
    # universe-context below) so --rescore only exists there, not on every other subcommand.
    sp = sub.add_parser("news")
    add_common_args(sp, DEFAULT_ROOT)
    sp.add_argument("--rescore", action="store_true",
                    help="re-score the existing cache with the configured sentiment backend "
                         "instead of fetching new rows (no Alpaca creds required)")
    sp.set_defaults(fn=handlers["news"])

    # v-playbook-desk-b1-2026-09-25 (fix round 1, finding 5): rebuild-only-the-context-file takes
    # no symbols/families/cost-mult/etc -- just --root, like `promote`.
    sp = sub.add_parser("universe-context")
    sp.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sp.set_defaults(fn=handlers["universe-context"])

    # Batch-2 Task E/F: the split-adjustment audit. --symbols (comma list or @file) narrows it;
    # nothing else applies -- it reads bars5/ and context/universe_daily.parquet only.
    sp = sub.add_parser("splits-audit")
    sp.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sp.add_argument("--symbols", default=None, help="comma list or @file; default every symbol in bars5/")
    sp.set_defaults(fn=handlers["splits-audit"])

    sp = sub.add_parser("scorecard")
    sp.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sp.add_argument("--family", required=True)
    sp.add_argument("--ledger", type=Path, default=None, help="default: data/shadow_fit_<family>.ndjson")
    sp.add_argument("--cost-mult", type=float, default=1.0)
    sp.set_defaults(fn=handlers["scorecard"])

    # v-playbook-desk-b2-2026-09-25 (Task 3): --desk-dir lives only here (like `promote`'s
    # --root), so no other subcommand grows a flag it ignores.
    sp = sub.add_parser("playbook")
    sp.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sp.add_argument("--families", default=None, help="comma list; default all")
    sp.add_argument("--desk-dir", type=Path, default=PLAYBOOK_DESK_DIR,
                    help="where playbook.json is written (default data/desk)")
    sp.add_argument("--postgres", action="store_true",
                    help="also read the live bot_trades tape, bounded to the last 120 days "
                         "(needs POSTGRES_DSN; `all --postgres` passes this through)")
    sp.set_defaults(fn=handlers["playbook"])

    sp = sub.add_parser("promote")
    sp.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sp.set_defaults(fn=handlers["promote"])
    return p
