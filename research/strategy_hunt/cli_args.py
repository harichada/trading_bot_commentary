"""Shared argparse surface for the strategy-hunt subcommands.

Extracted from `cli.py` in v-playbook-desk-b1-6 (final review) so that module stays inside the
400-line budget: `cli.py` owns the commands, this module owns the flags every pipeline
subcommand shares and the one flag `all` must refuse.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

GRID_PIN_SEP = "="

# v-playbook-desk-b1-5 task 10: `all` runs the shuffled baseline with this fixed seed, so the
# control is reproducible and the same across nightly runs.
BASELINE_SHUFFLE_SEED = 7

# Batch-2 Task A: which control `run --shuffle-seed N` produces.
#   "path"      -- permute 5-minute returns inside each session (`run.shuffle_bars`) and re-run the
#                  families on the permuted tape -> `trades_shuffle/`. Tests intraday timing.
#   "selection" -- keep the real trade schedule and replay each trade on a random peer symbol
#                  (`selection_baseline.replay_selection`) -> `trades_selection/`. Tests selection.
SHUFFLE_MODE_PATH = "path"
SHUFFLE_MODE_SELECTION = "selection"
SHUFFLE_MODES = (SHUFFLE_MODE_PATH, SHUFFLE_MODE_SELECTION)


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


class RejectShuffleMode(argparse.Action):
    """`--shuffle-mode` on `all`: a parser error, for the same reason `--shuffle-seed` is.

    `all` runs both controls itself (path shuffle, then selection). A user-supplied mode would apply
    to the *real* leg as well -- `all --shuffle-mode selection` would replay a `trades/` tape that
    this very invocation has not written yet. The flag stays registered so `cmd_all`'s namespace
    keeps its `shuffle_mode` default for `_baseline_namespace`/`_selection_namespace`.
    """

    def __call__(self, parser, namespace, values, option_string=None):  # noqa: D102
        parser.error(f"{option_string} is not accepted by `all`: it runs the real tape plus both "
                     f"controls ({', '.join(SHUFFLE_MODES)}) itself. Use "
                     f"`run --shuffle-seed N --shuffle-mode MODE` for a one-off control.")


def parse_grid_pins(values: Optional[Sequence[str]]) -> dict[str, str]:
    """`["entry_style=limit_open", ...]` -> `{"entry_style": "limit_open"}` (Batch-2 Task F).

    Values stay strings here: only the family's own grid knows whether an axis is int, float or str,
    so the typing happens in `walkforward.runner.coerce_pin` at the point of use -- which also keeps
    the pins picklable for the worker pool and JSON-serialisable for the manifest. A malformed token
    raises `ValueError`; `cmd_run` turns that into a message and a non-zero exit, because a pin the
    run quietly ignored would produce an A/B whose arms are the same run."""
    pins: dict[str, str] = {}
    for token in values or ():
        key, _, value = str(token).partition(GRID_PIN_SEP)
        key = key.strip()
        if not key or not _:
            raise ValueError(f"--grid-pin expects KEY{GRID_PIN_SEP}VALUE, got {token!r}")
        pins[key] = value
    return pins


def baseline_namespace(a: argparse.Namespace) -> argparse.Namespace:
    """A copy of `a` configured for the path-shuffle baseline run. A copy, not a mutation: the
    later steps must still see the real run's `shuffle_seed=None` and the user's `--force`."""
    return argparse.Namespace(**{**vars(a), "shuffle_seed": BASELINE_SHUFFLE_SEED, "force": True,
                                 "shuffle_mode": SHUFFLE_MODE_PATH})


def selection_namespace(a: argparse.Namespace) -> argparse.Namespace:
    """A copy of `a` configured for the selection baseline replay (Batch-2 Task A)."""
    return argparse.Namespace(**{**vars(a), "shuffle_seed": BASELINE_SHUFFLE_SEED, "force": True,
                                 "shuffle_mode": SHUFFLE_MODE_SELECTION})


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
        sp.add_argument("--shuffle-mode", choices=SHUFFLE_MODES, default=SHUFFLE_MODE_PATH,
                        help="which control `--shuffle-seed` produces: `path` (default) permutes "
                             "5-minute returns inside each session and re-runs the families; "
                             "`selection` keeps the real trade schedule and replays every trade on "
                             "a seeded random peer symbol (Batch-2 Task A)")
    else:
        sp.add_argument("--shuffle-seed", action=RejectShuffleSeed, type=int, default=None,
                        help=argparse.SUPPRESS)
        sp.add_argument("--shuffle-mode", action=RejectShuffleMode, default=SHUFFLE_MODE_PATH,
                        help=argparse.SUPPRESS)
    # Batch-2 Task F: repeatable, and shared with `all` (unlike --shuffle-seed, a pin applies to the
    # real leg *and* both controls -- a control has to search the same collapsed grid as its arm).
    sp.add_argument("--grid-pin", action="append", default=None, metavar="KEY=VALUE",
                    help="pin one grid axis to a single value in every family that has it, e.g. "
                         "--grid-pin entry_style=limit_open. Repeatable. Part of the checkpoint "
                         "config key, the run manifest and the report's search accounting, so the "
                         "two arms of an A/B never share checkpoints or a grid-point count")
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
