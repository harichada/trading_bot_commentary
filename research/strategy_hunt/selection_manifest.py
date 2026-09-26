"""Per-group pairing counts for the selection control (Batch-2 Task A fix r1, Important 2).

The selection control is *paired* with the real tape: one replayed trade per real trade. Drops break
that pairing -- a trade whose peer had no bar at that timestamp, or whose peer could not take the
entry (a session's first bar, a non-positive price, an intraday signal that would cross a session
boundary) -- and a broken pairing does not announce itself in the pooled numbers, where a smaller
control group simply looks like a different population.

So the replay records, per pooled group `(family, side, regime)`, how many real trades there were and
how many were replayed. The manifest carries those counts; the scorer reaches the same conclusion
independently from the two tapes' group sizes (`score.regime_rows.BASELINE_UNDERREPLAYED`), and the
report shows the ratio per row. Three views of one fact, none of them derived from the others, so a
disagreement is visible rather than silent.

The group of a replayed trade is taken from its *source* trade's `regime` column rather than from the
control trade's own stamp. They are the same label -- the regime lookup is a pure function of
`entry_ts`, which the replay preserves -- but reading it from the source is what makes these counts a
statement about the pairing rather than about the control tape.

This module also owns the `run --shuffle-mode selection` subcommand: the counts above only mean
something once they are written down, and keeping the record's shape beside the code that computes it
is what stops the manifest and the scorer from describing different runs. Split out of
`selection_baseline.py`, which is at its 400-line budget.
"""
from __future__ import annotations

import time
from typing import Any, Iterable

import pandas as pd

from research.strategy_hunt.paths import SELECTION_VARIANT
from research.strategy_hunt.run import write_manifest
from research.strategy_hunt.score.regime_rows import BASELINE_MIN_REPLAYED_FRAC

# The pooled group's identity, flattened for JSON. `|` because no family, side or regime label
# contains one, so the key can be split back apart by a reader.
GROUP_SEP = "|"


def group_key(family: Any, side: Any, regime: Any) -> str:
    return f"{family}{GROUP_SEP}{side}{GROUP_SEP}{regime}"


def real_group_counts(trades: pd.DataFrame) -> dict[str, int]:
    """How many real trades each `(family, side, regime)` group holds -- the pairing's denominator."""
    if trades is None or trades.empty:
        return {}
    counts: dict[str, int] = {}
    for family, side, regime in zip(trades["family"], trades["side"], trades["regime"]):
        key = group_key(family, side, regime)
        counts[key] = counts.get(key, 0) + 1
    return counts


def replayed_group_counts(records: Iterable[dict]) -> dict[str, int]:
    """The per-peer `groups` counts summed over every peer record."""
    counts: dict[str, int] = {}
    for record in records:
        for key, n in (record.get("groups") or {}).items():
            counts[key] = counts.get(key, 0) + int(n)
    return counts


def pairing_counts(real: dict[str, int], replayed: dict[str, int]) -> dict[str, dict[str, Any]]:
    """`group -> {real, replayed, dropped, replayed_frac}`, ordered by group key.

    Every group the *real* tape has appears, including one the control replayed nothing of -- that is
    the case the operator most needs to see. A group only the control has (which would mean the two
    tapes disagree about a trade's regime) appears too, with `real` 0 and `replayed_frac` 0.0."""
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(set(real) | set(replayed)):
        n_real, n_replayed = int(real.get(key, 0)), int(replayed.get(key, 0))
        out[key] = {"real": n_real, "replayed": n_replayed, "dropped": n_real - n_replayed,
                    "replayed_frac": (n_replayed / n_real) if n_real else 0.0}
    return out


def underreplayed_groups(pairing: dict[str, dict[str, Any]],
                         min_frac: float = BASELINE_MIN_REPLAYED_FRAC) -> list[str]:
    """The groups the scorer will flag `baseline_underreplayed`, named in the manifest so the
    operator sees them without having to score the run first. One threshold, defined in
    `score.regime_rows`, so the manifest and the verdict can never drift apart."""
    return [key for key, c in pairing.items() if float(c["replayed_frac"]) < min_frac]


def run_selection_command(a) -> int:
    """`run --shuffle-mode selection`: the replay plus its own manifest record. Presentation lives
    here rather than in `cli.py` (the precedent is `playbook_cli.py`), beside the counts it records
    and inside both modules' line budgets. Keyed `run_selection` because `write_manifest` merges by
    `dict.update`: a shared key would erase the real run's record for that date. `replay_selection`
    is imported in the body, not at module load: the algorithm imports these counts, so the
    dependency has to run one way at import time."""
    from research.strategy_hunt.selection_baseline import replay_selection

    if a.shuffle_seed is None:
        print("run: --shuffle-mode selection needs --shuffle-seed N (the peer draw must be reproducible)")
        return 1
    t0 = time.time()
    s = replay_selection(a.root, seed=a.shuffle_seed, workers=a.workers, notional=a.notional,
                         cost_mult=a.cost_mult)
    print(f"run (selection): {s['replayed']}/{s['trades']} trades replayed onto {s['peers']} peers, "
          f"dropped {s['selection_dropped']} {s['dropped'] or ''} elapsed={time.time() - t0:.0f}s")
    under = s.get("underreplayed_groups") or []
    if under:
        print(f"run (selection): under-replayed groups (< {100 * BASELINE_MIN_REPLAYED_FRAC:.0f}% paired, "
              f"cannot PASS): {', '.join(under)}")
    write_manifest(a.root, {"date": time.strftime("%Y-%m-%d"),
                            "run_selection": {**s, "variant": SELECTION_VARIANT, "shuffle_seed": a.shuffle_seed,
                                              "cost_mult": a.cost_mult, "notional": a.notional,
                                              "elapsed_sec": round(time.time() - t0)},
                            "errors_selection": s["errors"]})
    # The same tolerance as the real leg (`cli.cmd_run`): a handful of unreadable symbols must not
    # stop `all` before score/playbook, and the failures are on the manifest either way (fix r1,
    # Important 1). More than a tenth of the peers failing is a broken run, not a few bad symbols.
    return 0 if len(s["errors"]) <= 0.1 * max(1, int(s.get("peers_total") or 0)) else 1
