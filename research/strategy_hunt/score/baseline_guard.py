"""Which control tape a pooled row is scored against, and whether that tape is trustworthy.

Split out of `regime_rows.py` (Batch-2 follow-up: that module was at its 400-line ceiling) so the
`baseline_kind` enum and the coverage/staleness checks that read it stay in one small, independently
testable module. `regime_rows.py` re-exports everything here, so callers and tests keep importing
from `regime_rows` unchanged.
"""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd

# Batch-2 Task A: *which* control a pooled row was measured against, recorded on the row because the
# two controls answer different questions and an excess over one is not an excess over the other.
#   "path"      -- the shuffled tape (`trades_shuffle/`): destroys the intraday path, keeps each
#                  session's return. The honest control for an `intraday` hold.
#   "selection" -- the random-peer replay (`trades_selection/`): keeps the schedule, redraws the
#                  symbol. The only one of the two that tests a `multiday`/`overnight` hold.
#   "none"      -- no matching control group; the row also carries the `no_baseline` failure.
BASELINE_KIND_PATH = "path"
BASELINE_KIND_SELECTION = "selection"
BASELINE_KIND_NONE = "none"
BASELINE_KINDS = (BASELINE_KIND_PATH, BASELINE_KIND_SELECTION, BASELINE_KIND_NONE)

# v-playbook-desk-b1-5 task 10 fix r1. A *leftover* baseline is worse than none: it silently
# becomes the yardstick for every pooled verdict while looking exactly like a fresh one. These
# two failure keys say so on the row itself, so a stale/partial control can never mint a PASS.
BASELINE_MISMATCH = "baseline_mismatch"      # the baseline skipped too much of the universe
BASELINE_STALE = "baseline_stale"            # the baseline stopped too long before the real tape
BASELINE_MAX_MISSING_FRAC = 0.05             # strictly more than 5% of real symbols missing
BASELINE_MAX_STALE_DAYS = 7                  # strictly more than 7 calendar days behind


def _symbols(trades: Optional[pd.DataFrame]) -> set[str]:
    if trades is None or trades.empty or "symbol" not in trades.columns:
        return set()
    return {str(s) for s in trades["symbol"].unique()}


def _last_exit(trades: Optional[pd.DataFrame]) -> Optional[pd.Timestamp]:
    if trades is None or trades.empty or "exit_ts" not in trades.columns:
        return None
    last = pd.to_datetime(trades["exit_ts"]).max()
    return None if pd.isna(last) else pd.Timestamp(last)


def _last_entry(trades: Optional[pd.DataFrame]) -> Optional[pd.Timestamp]:
    if trades is None or trades.empty or "entry_ts" not in trades.columns:
        return None
    last = pd.to_datetime(trades["entry_ts"]).max()
    return None if pd.isna(last) else pd.Timestamp(last)


def _day_gap(later: Optional[pd.Timestamp], earlier: Optional[pd.Timestamp]) -> int:
    """`later` minus `earlier` in whole calendar days, or 0 when either side is missing."""
    return (later.date() - earlier.date()).days if later is not None and earlier is not None else 0


def baseline_coverage(real_trades: Optional[pd.DataFrame],
                      baseline_trades: Optional[pd.DataFrame]) -> dict[str, Any]:
    """How much of the real tape the baseline actually covers, and how far behind it is.

    Pure: the counts and dates `baseline_guard` decides on and `score_regime_root` logs, so the
    warning and the verdict can never disagree about what they saw.

    Both the last *exit* (`real_last_exit`, `baseline_last_exit`, `stale_days`) and the last
    *entry* (`real_last_entry`, `baseline_last_entry`) of each tape are reported. `stale_days`
    itself is always the exit gap here -- it is `baseline_guard` that picks entries instead for a
    `"selection"` control (see its docstring), so this function's contract never changes underfoot
    for the log message that already reads `stale_days` as an exit gap."""
    real_syms, base_syms = _symbols(real_trades), _symbols(baseline_trades)
    missing = sorted(real_syms - base_syms)
    real_last_exit, base_last_exit = _last_exit(real_trades), _last_exit(baseline_trades)
    real_last_entry, base_last_entry = _last_entry(real_trades), _last_entry(baseline_trades)
    return {"n_real_symbols": len(real_syms), "n_baseline_symbols": len(base_syms),
            "n_missing": len(missing), "missing": missing[:20],
            "missing_frac": len(missing) / len(real_syms) if real_syms else 0.0,
            "real_last_exit": str(real_last_exit)[:10] if real_last_exit is not None else "",
            "baseline_last_exit": str(base_last_exit)[:10] if base_last_exit is not None else "",
            "real_last_entry": str(real_last_entry)[:10] if real_last_entry is not None else "",
            "baseline_last_entry": str(base_last_entry)[:10] if base_last_entry is not None else "",
            "stale_days": _day_gap(real_last_exit, base_last_exit)}


def baseline_guard(real_trades: Optional[pd.DataFrame],
                   baseline_trades: Optional[pd.DataFrame],
                   baseline_kind: str = BASELINE_KIND_PATH) -> list[str]:
    """Failure keys for a baseline that is present but not trustworthy.

    `BASELINE_MISMATCH` when the control skipped more than `BASELINE_MAX_MISSING_FRAC` of the
    real universe's symbols, `BASELINE_STALE` when the tapes are more than `BASELINE_MAX_STALE_DAYS`
    calendar days apart. A baseline that is *newer* than the real tape is not stale. An absent
    baseline returns `[]` — that case is already `no_baseline`, and saying it twice would only
    obscure which one happened.

    The coverage test is skipped for `baseline_kind == "selection"` (fix r1, Important 3): that
    control replays each trade on a *random peer*, so which symbols it lands on is a draw, not a
    measure of completeness -- the flag would fire on every correct run. A dropped trade is measured
    per group instead, by `BASELINE_UNDERREPLAYED`.

    Staleness compares last *exits* for every kind except `"selection"` (fix r2). A selection
    control shares the real tape's *entry* schedule by construction -- it redraws the symbol, not
    the calendar -- so comparing exits flags a perfectly fresh control the moment one long real
    trade outlives every replayed one (observed: a holdout run where the real tape's last exit
    landed 13 days after the control's, with every other trade paired same-day, flagged every
    pooled row `baseline_stale`). Comparing entries instead asks the question the pairing actually
    supports: did the control keep drawing peers up through the real tape's most recent entries?"""
    cov = baseline_coverage(real_trades, baseline_trades)
    if not cov["n_real_symbols"] or not cov["n_baseline_symbols"]:
        return []
    flags = []
    if baseline_kind != BASELINE_KIND_SELECTION and cov["missing_frac"] > BASELINE_MAX_MISSING_FRAC:
        flags.append(BASELINE_MISMATCH)
    if baseline_kind == BASELINE_KIND_SELECTION:
        stale_days = _day_gap(_last_entry(real_trades), _last_entry(baseline_trades))
    else:
        stale_days = cov["stale_days"]
    if stale_days > BASELINE_MAX_STALE_DAYS:
        flags.append(BASELINE_STALE)
    return flags
