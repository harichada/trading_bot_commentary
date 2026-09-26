"""Shared argparse surface for the strategy-hunt subcommands.

Extracted from `cli.py` in v-playbook-desk-b1-6 (final review) so that module stays inside the
400-line budget: `cli.py` owns the commands, this module owns the flags every pipeline
subcommand shares and the one flag `all` must refuse.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# v-playbook-desk-b1-5 task 10: `all` runs the shuffled baseline with this fixed seed, so the
# control is reproducible and the same across nightly runs.
BASELINE_SHUFFLE_SEED = 7


class RejectShuffleSeed(argparse.Action):
    """`--shuffle-seed` on `all`: a parser error, not a silently honoured flag.

    v-playbook-desk-b1-6 (final review, Minor 1). `all` already runs both tapes -- the real one
    and the fixed-seed shuffled baseline (`BASELINE_SHUFFLE_SEED`). A user-supplied seed turned
    the *real* leg into a second shuffled run, so `trades/` was never written and `score`,
    `report` and `playbook` all measured the control against itself. The flag still exists on
    the subparser (so `cmd_all`'s namespace keeps its `shuffle_seed=None` for
    `_baseline_namespace`), it just cannot be set.
    """

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: D102
        parser.error(f"{option_string} is not accepted by `all`: it runs the real tape plus the "
                     f"fixed-seed (--shuffle-seed {BASELINE_SHUFFLE_SEED}) baseline itself. "
                     f"Use `run --shuffle-seed N` for a one-off control.")


def add_common_args(sp: argparse.ArgumentParser, default_root: Path,
                    user_shuffle_seed: bool = True) -> None:
    """The flags shared by export/context/news/earnings/calibrate/run/score/report/all.

    `user_shuffle_seed=False` (only `all`) keeps the flag present with its `None` default but
    makes setting it a parser error -- see `RejectShuffleSeed`.
    """
    sp.add_argument("--root", type=Path, default=default_root)
    sp.add_argument("--symbols", default=None, help="comma list or @file")
    sp.add_argument("--families", default=None, help="comma list; default all")
    sp.add_argument("--min-days", type=int, default=250)
    sp.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--cost-mult", type=float, default=1.0)
    if user_shuffle_seed:
        sp.add_argument("--shuffle-seed", type=int, default=None)
    else:
        sp.add_argument("--shuffle-seed", action=RejectShuffleSeed, type=int, default=None,
                        help=argparse.SUPPRESS)
    sp.add_argument("--notional", type=float, default=6000.0)
    sp.add_argument("--holdout-months", type=int, default=0,
                    help="withhold this many months of the most recent out-of-sample data from "
                         "build_windows (strategy-search-batch1 task 2); 0 (default) uses the "
                         "full history. A search run sets this > 0, then a confirmation run "
                         "with --holdout-months 0 re-scores only the withheld months")
    sp.add_argument("--no-coldstart", action="store_true")
    sp.add_argument("--no-baseline", action="store_true",
                    help="`all` only: skip the shuffled baseline run. Whatever baseline is "
                         "already in the root is reused; scoring flags it baseline_mismatch / "
                         "baseline_stale if it is partial or old, and pooled rows fall back to "
                         "no_baseline only when the root has none at all")
    sp.add_argument("--postgres", action="store_true")
