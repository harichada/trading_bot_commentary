"""Regime-conditioned fit rows, universe-pooled regime lanes, and their shuffled baseline (spec §6).

Three extra row families on top of the unconditional per-(symbol, family, side)
rows `fit/writer.build_row` produces:

* `regime_rows(...)` — one row per market regime the symbol traded in *often enough
  to measure*, scored against the same floors as the unconditional row. A symbol
  whose edge only exists in `risk_on` shows PASS there and FAIL elsewhere, which is
  what the live gate needs in order to stand a lane down when the tape turns.
* `pooled_rows(...)` — one row per (family, side, regime) across the whole
  universe, symbol `"*"`, source `"pooled"`, scored against `POOLED_FLOORS`.
  A pooled PASS is the family-level statement "this lane works in this regime",
  which lets a lane trade a symbol that has no history of its own yet.
* `baseline_pooled_rows(...)` — the same pooled rows computed over a *control*
  tape, stamped `source="baseline"` and `baseline_kind`. They exist so
  `pooled_rows(..., baseline=...)` can require positive excess over them, and they
  live only under the snapshot's own `baseline_pooled_rows` key. Since Batch-2 Task A
  there are two control tapes -- the path shuffle (`trades_shuffle/`) for intraday
  holds and the random-peer selection replay (`trades_selection/`) for multi-day and
  overnight ones -- and `run.score_regime_root` decides which one a family's rows are
  scored against.

Three sample rules matter, because a row here is read by the live gate:

* **No evidence is not a FAIL** (v-playbook-desk-b1-2026-09-25, review r1). Under
  `REGIME_ROW_MIN_N` trades a row could only ever come out FAIL — `stability` and
  `decay` are structurally 0 below four windows and `min_n` is 150 — and
  `StrategyFit.lookup_regime` would let that manufactured FAIL outrank both the
  pooled PASS and the symbol's own unconditional PASS. So a thin regime gets no
  row at all and the lookup falls through; its trades still count in the pooled
  and unconditional rows.
* **Pooled windows are calendar quarters.** `window_id` is a per-symbol
  walk-forward index, so grouping pooled trades by it would average
  non-contemporaneous windows together. Pooled rows are re-bucketed by the
  calendar quarter of `entry_ts` before the metrics are computed; per-symbol
  regime rows keep `window_id`, which is contemporaneous within one symbol.
* **A pooled row must beat its own shuffled control** (v-playbook-desk-b1-5 task 10).
  The B1 run's shuffle control — 5-minute returns permuted within each session, every
  session's open-to-close return preserved — reached a pooled PASS while the real tape
  did not, because a pooled row over a drifting regime measures the drift. `excess_r`
  is `exp_r` minus the matching baseline group's `exp_r`, `min_excess_r` is the floor,
  and a row with no baseline records `no_baseline` and can never PASS.

Only the three real regime labels are scored (`REGIMES`). The entry-time lookup in
`data/regime.py` returns "unknown" whenever the SPY/VIX context is missing for a
session; a row built from those trades would describe the data gap, not a regime,
and the gate never asks for it — so those trades are dropped from both row families
(they remain in the unconditional row, which is computed over every trade).
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

import pandas as pd

from research.strategy_hunt.fit.writer import ROW_KEYS, build_row
from research.strategy_hunt.models import OPTIONAL_TRADE_COLUMNS, TRADE_COLUMNS, FitMetrics, SimTrade
from research.strategy_hunt.score.verdict import WATCH_FLOORS

REGIMES = ("risk_on", "risk_off", "mixed")
# The row shape of both new families. fit/writer.py imports this (lazily, to avoid the
# import cycle: this module imports build_row from there) to lay out their Parquet files.
REGIME_ROW_KEYS = ROW_KEYS + ("regime",)
# Pooled and baseline rows additionally record which control they relate to (Batch-2 Task A). Their
# own key list, not `ROW_KEYS`: `baseline_kind` is meaningless for an unconditional or per-symbol
# regime row, and `ROW_KEYS` is also the `strategy_fit` SQL column list the live gate reads.
POOLED_ROW_KEYS = REGIME_ROW_KEYS + ("baseline_kind",)
# Fewer trades than the WATCH floor is not a measurement: emit no row (see the docstring).
REGIME_ROW_MIN_N = WATCH_FLOORS["min_n"]

# Pooled floors are deliberately stricter than the per-symbol floors: a pooled row
# aggregates hundreds of symbols, so n is cheap and only a large t-stat and a real
# profit factor distinguish a family-wide edge from the average of many small ones.
# There is no max_dd_r cap here (v-playbook-desk-b1-5 task 9): a flat R drawdown
# ceiling is not scale-invariant — the B1 run showed a 13k-trade pooled row with a
# 639R drawdown by arithmetic, which a fixed 60R cap would fail regardless of how many
# times net R clears it. min_recovery (net_r / max_dd_r) asks the scale-invariant
# question instead. Likewise min_t_stat alone treats same-session trades across
# symbols as independent observations, inflating significance; min_t_stat_session
# clusters by session (the true independent unit) on top of it.
# min_excess_r (v-playbook-desk-b1-5 task 10) is the floor the B1 run showed was missing: the
# shuffle control -- 5-minute returns permuted inside each session, each session's open-to-close
# return preserved -- reached a pooled PASS while the real tape did not, because a pooled row
# over a drifting regime measures the drift and not the rule. A pooled lane now has to beat the
# same (family, side, regime) shuffled baseline by this much expectancy before it can PASS.
POOLED_FLOORS = {"min_n": 500, "min_sessions": 60, "min_pf": 1.2, "min_wr": 0.45, "min_exp_r": 0.05,
                 "min_recovery": 3.0, "min_stability": 0.6, "min_decay": 0.5, "min_t_stat": 3.0,
                 "min_t_stat_session": 3.0, "min_exp_r_lb": 0.0, "min_excess_r": 0.05}
POOLED_WATCH_FLOORS = {"min_n": 200, "min_pf": 1.05}
POOLED_SYMBOL = "*"
# String contract with core/strategy_fit_rows.py: FitRow.from_dict flags a row pooled when
# its "source" is this value. Change one side and the live gate stops seeing pooled lanes.
POOLED_SOURCE = "pooled"
# Baseline rows are shuffled-control pooled rows. Deliberately *not* POOLED_SOURCE: they are
# written under their own snapshot section and the live gate must never read one as a lane.
BASELINE_SOURCE = "baseline"
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
# `excess_r` when no baseline group matched. NaN, not 0.0: "not measured" and "exactly matched
# the shuffled control" are different claims and only the second one is evidence.
NO_BASELINE = float("nan")

# v-playbook-desk-b1-5 task 10 fix r1. A *leftover* baseline is worse than none: it silently
# becomes the yardstick for every pooled verdict while looking exactly like a fresh one. These
# two failure keys say so on the row itself, so a stale/partial control can never mint a PASS.
BASELINE_MISMATCH = "baseline_mismatch"      # the baseline skipped too much of the universe
BASELINE_STALE = "baseline_stale"            # the baseline stopped too long before the real tape
BASELINE_MAX_MISSING_FRAC = 0.05             # strictly more than 5% of real symbols missing
BASELINE_MAX_STALE_DAYS = 7                  # strictly more than 7 calendar days behind

# v-playbook-desk-b1-6 (final review, Important 2). `baseline_guard` above compares the two
# *tapes*; this compares the two *groups*. A control group with a fifth of the real group's
# trades is measuring a different population, so the difference of their expectancies is not
# the "excess over the regime's drift" the verdict claims it is -- most often it means the
# shuffled tape stopped proposing this lane, not that the real one has an edge over it.
BASELINE_POPULATION_MISMATCH = "baseline_population_mismatch"
BASELINE_POPULATION_MIN_RATIO = 0.5         # baseline_n below this multiple of n is a mismatch
BASELINE_POPULATION_MAX_RATIO = 2.0         # ...and so is above this multiple

# Batch-2 Task A fix r1 (Important 2). The selection control is *paired* -- one replayed trade per
# real trade -- so a group smaller than its real group means trades were dropped, not that the lane
# beat the control, and the 2x population band is far too loose to catch it. Only `"selection"` is
# judged by this: the shuffled tape re-runs the families, where a different n is the expected outcome.
BASELINE_UNDERREPLAYED = "baseline_underreplayed"
BASELINE_MIN_REPLAYED_FRAC = 0.9


def _symbols(trades: Optional[pd.DataFrame]) -> set[str]:
    if trades is None or trades.empty or "symbol" not in trades.columns:
        return set()
    return {str(s) for s in trades["symbol"].unique()}


def _last_exit(trades: Optional[pd.DataFrame]) -> Optional[pd.Timestamp]:
    if trades is None or trades.empty or "exit_ts" not in trades.columns:
        return None
    last = pd.to_datetime(trades["exit_ts"]).max()
    return None if pd.isna(last) else pd.Timestamp(last)


def baseline_coverage(real_trades: Optional[pd.DataFrame],
                      baseline_trades: Optional[pd.DataFrame]) -> dict[str, Any]:
    """How much of the real tape the shuffled baseline actually covers, and how far behind it is.

    Pure: the counts and dates `baseline_guard` decides on and `score_regime_root` logs, so the
    warning and the verdict can never disagree about what they saw."""
    real_syms, base_syms = _symbols(real_trades), _symbols(baseline_trades)
    missing = sorted(real_syms - base_syms)
    real_last, base_last = _last_exit(real_trades), _last_exit(baseline_trades)
    stale_days = (real_last.date() - base_last.date()).days if real_last is not None and base_last is not None else 0
    return {"n_real_symbols": len(real_syms), "n_baseline_symbols": len(base_syms),
            "n_missing": len(missing), "missing": missing[:20],
            "missing_frac": len(missing) / len(real_syms) if real_syms else 0.0,
            "real_last_exit": str(real_last)[:10] if real_last is not None else "",
            "baseline_last_exit": str(base_last)[:10] if base_last is not None else "",
            "stale_days": stale_days}


def baseline_guard(real_trades: Optional[pd.DataFrame],
                   baseline_trades: Optional[pd.DataFrame],
                   baseline_kind: str = BASELINE_KIND_PATH) -> list[str]:
    """Failure keys for a baseline that is present but not trustworthy.

    `BASELINE_MISMATCH` when the control skipped more than `BASELINE_MAX_MISSING_FRAC` of the
    real universe's symbols, `BASELINE_STALE` when its last exit is more than
    `BASELINE_MAX_STALE_DAYS` calendar days behind the real tape's. A baseline that is *newer*
    than the real tape is not stale. An absent baseline returns `[]` — that case is already
    `no_baseline`, and saying it twice would only obscure which one happened.

    The coverage test is skipped for `baseline_kind == "selection"` (fix r1, Important 3): that
    control replays each trade on a *random peer*, so which symbols it lands on is a draw, not a
    measure of completeness -- the flag would fire on every correct run. A dropped trade is measured
    per group instead, by `BASELINE_UNDERREPLAYED`. Staleness still applies to both kinds."""
    cov = baseline_coverage(real_trades, baseline_trades)
    if not cov["n_real_symbols"] or not cov["n_baseline_symbols"]:
        return []
    flags = []
    if baseline_kind != BASELINE_KIND_SELECTION and cov["missing_frac"] > BASELINE_MAX_MISSING_FRAC:
        flags.append(BASELINE_MISMATCH)
    if cov["stale_days"] > BASELINE_MAX_STALE_DAYS:
        flags.append(BASELINE_STALE)
    return flags


def _population_ok(n: int, baseline_n: Optional[int]) -> bool:
    """Whether the baseline group is close enough in size to the real group to be its yardstick.

    `None` means the caller did not measure the control's population at all (the
    `verdict_pooled` default, kept so a metrics-only caller keeps its old contract);
    `pooled_rows` always passes the real number."""
    if baseline_n is None:
        return True
    lo = BASELINE_POPULATION_MIN_RATIO * float(n)
    hi = BASELINE_POPULATION_MAX_RATIO * float(n)
    return lo <= float(baseline_n) <= hi


def _replayed_ok(n: int, baseline_n: Optional[int], baseline_kind: str) -> bool:
    """Whether a *paired* control replayed enough of the real group to be its yardstick.

    Only the selection control is paired; for any other kind this is vacuously true."""
    if baseline_kind != BASELINE_KIND_SELECTION or baseline_n is None or n <= 0:
        return True
    return float(baseline_n) / float(n) >= BASELINE_MIN_REPLAYED_FRAC


def _pooled_failures(m: FitMetrics, excess_r: float = NO_BASELINE,
                     guard: tuple[str, ...] = (),
                     baseline_n: Optional[int] = None,
                     baseline_kind: str = BASELINE_KIND_NONE) -> list[str]:
    # A missing baseline is its own failure key: without one there is nothing to show the
    # row is more than the regime's drift, so it can WATCH but never PASS. The population
    # check only applies when a baseline *was* measured -- with none, "no_baseline" is the
    # whole story and a second key would only obscure which case happened.
    if math.isnan(excess_r):
        baseline_checks = {"no_baseline": False}
    else:
        baseline_checks = {"min_excess_r": excess_r >= POOLED_FLOORS["min_excess_r"],
                           BASELINE_POPULATION_MISMATCH: _population_ok(m.n, baseline_n),
                           BASELINE_UNDERREPLAYED: _replayed_ok(m.n, baseline_n, baseline_kind)}
    checks = {
        "min_n": m.n >= POOLED_FLOORS["min_n"], "min_sessions": m.sessions >= POOLED_FLOORS["min_sessions"],
        "min_pf": m.pf >= POOLED_FLOORS["min_pf"], "min_wr": m.wr >= POOLED_FLOORS["min_wr"],
        "min_exp_r": m.exp_r >= POOLED_FLOORS["min_exp_r"],
        "min_recovery": m.recovery >= POOLED_FLOORS["min_recovery"],
        "min_stability": m.stability >= POOLED_FLOORS["min_stability"],
        "min_decay": m.decay >= POOLED_FLOORS["min_decay"],
        "min_t_stat": m.t_stat >= POOLED_FLOORS["min_t_stat"],
        "min_t_stat_session": m.t_stat_session >= POOLED_FLOORS["min_t_stat_session"],
        "min_exp_r_lb": m.exp_r_lb >= POOLED_FLOORS["min_exp_r_lb"],
        **baseline_checks,
        # `baseline_guard` keys are already decided; they enter as checks that cannot pass.
        **{g: False for g in guard},
    }
    return [k for k, ok in checks.items() if not ok]


def verdict_pooled(m: FitMetrics, excess_r: float = NO_BASELINE,
                   guard: tuple[str, ...] = (),
                   baseline_n: Optional[int] = None,
                   baseline_kind: str = BASELINE_KIND_NONE) -> tuple[str, list[str]]:
    """PASS / WATCH / FAIL against `POOLED_FLOORS` (mirrors `score.verdict.verdict`).

    `excess_r` is `exp_r` minus the same-regime shuffled baseline's `exp_r`, or NaN when no
    baseline was measured. The default is that NaN rather than a pass-through value, so a
    caller that forgets the baseline gets "no_baseline" instead of a free PASS.

    `guard` carries `baseline_guard`'s keys for a baseline that exists but is partial or
    stale; like `no_baseline` they permit WATCH and forbid PASS.

    `baseline_n` is the matching baseline group's trade count; when it is outside
    [`BASELINE_POPULATION_MIN_RATIO`, `BASELINE_POPULATION_MAX_RATIO`] x `m.n` the row carries
    `baseline_population_mismatch` and cannot PASS (v-playbook-desk-b1-6, Important 2). `None`
    -- the default -- means the caller measured no population and the check is skipped.

    `baseline_kind` selects the pairing floor: a `"selection"` control must have replayed at least
    `BASELINE_MIN_REPLAYED_FRAC` of the real group or the row carries `baseline_underreplayed` and
    cannot PASS (Batch-2 Task A fix r1). The default skips that check, since only a paired control
    can be judged by it."""
    failed = _pooled_failures(m, excess_r, guard, baseline_n, baseline_kind)
    if not failed:
        return "PASS", []
    if m.n >= POOLED_WATCH_FLOORS["min_n"] and m.exp_r > 0 and m.pf >= POOLED_WATCH_FLOORS["min_pf"]:
        return "WATCH", failed
    return "FAIL", failed


def trades_from_frame(df: pd.DataFrame) -> list[SimTrade]:
    """Rebuild `SimTrade`s from a trades Parquet frame (extra columns ignored).

    A column in `OPTIONAL_TRADE_COLUMNS` may be absent: `entry_style` was added in Batch-2 Task B,
    and a trades file written before it existed is a market-fill run by definition, so the
    dataclass default is the right value and the frame is read on the columns it actually has.
    A *required* column that is missing still fails loudly, as a missing `SimTrade` argument."""
    if df is None or df.empty:
        return []
    cols = [c for c in TRADE_COLUMNS if c in df.columns or c not in OPTIONAL_TRADE_COLUMNS]
    return [SimTrade(**{**r, "entry_ts": pd.Timestamp(r["entry_ts"]), "exit_ts": pd.Timestamp(r["exit_ts"])})
            for r in df[cols].to_dict("records")]


def _window_end(trades: list[SimTrade]) -> str:
    return str(max(t.exit_ts for t in trades))[:10] if trades else ""


def quarter_index(ts) -> int:
    """Chronological integer id of the calendar quarter `ts` falls in (2025Q4 < 2026Q1)."""
    period = pd.Timestamp(ts).to_period("Q")
    return int(period.year) * 4 + int(period.quarter)


def bucket_by_quarter(trades: list[SimTrade]) -> list[SimTrade]:
    """Copies of `trades` whose `window_id` is the calendar quarter of `entry_ts`.

    `score.metrics` measures stability and decay by grouping on `window_id`; re-labelling
    it is how pooled rows get contemporaneous buckets without a second metrics path.
    Returns new frozen trades — nothing is mutated."""
    return [dataclasses.replace(t, window_id=quarter_index(t.entry_ts)) for t in trades]


def regime_rows(symbol: str, family: str, side: str, trades: list[SimTrade], n_windows: int, computed_at: str,
                latest_params: Optional[dict] = None) -> list[dict[str, Any]]:
    """One `build_row`-shaped dict (plus `regime`) per regime in `REGIMES` with at least
    `REGIME_ROW_MIN_N` trades. Each row's `window_end` is that regime's own last exit."""
    by_regime: dict[str, list[SimTrade]] = {}
    for t in trades:
        if t.regime in REGIMES:
            by_regime.setdefault(t.regime, []).append(t)
    out = []
    for regime in REGIMES:
        sub = by_regime.get(regime, [])
        if len(sub) < REGIME_ROW_MIN_N:
            continue
        row = build_row(symbol, family, side, sub, n_windows, latest_params or {}, computed_at,
                        _window_end(sub))
        out.append({**row, "regime": regime})
    return out


def _baseline_index(baseline: Optional[list[dict[str, Any]]]) -> dict[tuple[str, str, str], dict]:
    """(family, side, regime) -> baseline row, for the lookup `pooled_rows` does per group."""
    return {(r["strategy"], r["side"], r.get("regime", "")): r for r in baseline or []}


def pooled_rows(all_trades: Optional[pd.DataFrame], computed_at: str,
                baseline: Optional[list[dict[str, Any]]] = None,
                guard: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """One pooled row per (family, side, regime) over the whole universe's trades.

    Windows are calendar quarters here, not per-symbol `window_id`s (see the module
    docstring), so `stability`/`decay` compare the same stretch of tape across symbols.

    `baseline` is the shuffled control's pooled rows (`baseline_pooled_rows` over
    `<root>/trades_shuffle/`). Each row gains `baseline_exp_r`, `baseline_n` and
    `excess_r = exp_r - baseline_exp_r`, and its verdict is scored against that excess:
    with no matching baseline group the excess is NaN and the row can never PASS, and with a
    baseline group whose population is far from this row's it carries
    `baseline_population_mismatch` and cannot PASS either.

    `guard` is `baseline_guard(real_trades, baseline_trades)`: keys stamped on *every* row
    because a partial or stale control is a property of the run, not of one lane.

    `baseline_kind` (Batch-2 Task A) is copied from the matching baseline row, so a row says which
    control produced its excess; with no match it is `BASELINE_KIND_NONE`. The caller decides which
    control a family *should* be measured against (`run.score_regime_root` picks by hold kind) --
    this function only records what it was handed, which is why passing the wrong control cannot be
    hidden by the row."""
    trades = trades_from_frame(all_trades)
    index = _baseline_index(baseline)
    groups: dict[tuple[str, str, str], list[SimTrade]] = {}
    for t in trades:
        if t.regime in REGIMES:
            groups.setdefault((t.family, t.side, t.regime), []).append(t)
    out = []
    for (family, side, regime), sub in sorted(groups.items()):
        bucketed = bucket_by_quarter(sub)
        n_windows = len({t.window_id for t in bucketed})
        base = index.get((family, side, regime))
        base_exp = float(base["exp_r"]) if base is not None else NO_BASELINE
        base_n = int(base["n"]) if base is not None else 0
        base_kind = str(base.get("baseline_kind") or BASELINE_KIND_NONE) if base is not None else BASELINE_KIND_NONE
        # The verdict needs the excess, which needs this row's own exp_r -- so it is derived
        # from the metrics the builder just computed rather than by scoring the group twice.
        row = build_row(POOLED_SYMBOL, family, side, bucketed, n_windows, {}, computed_at, _window_end(sub),
                        source=POOLED_SOURCE,
                        verdict_fn=lambda m, b=base_exp, bn=base_n, bk=base_kind: verdict_pooled(
                            m, m.exp_r - b, guard, baseline_n=bn, baseline_kind=bk))
        out.append({**row, "regime": regime, "baseline_exp_r": base_exp, "baseline_n": base_n,
                    "excess_r": row["exp_r"] - base_exp, "baseline_kind": base_kind})
    return out


def baseline_pooled_rows(all_trades: Optional[pd.DataFrame], computed_at: str,
                         baseline_kind: str = BASELINE_KIND_PATH) -> list[dict[str, Any]]:
    """`pooled_rows` over a control tape's trades, stamped `source="baseline"`.

    Identical grouping, quarter bucketing and floors as the real pooled rows, so the two are
    comparable line for line. A baseline row has no baseline of its own, so it carries
    `excess_r` NaN and the `no_baseline` failure — it is read for its `exp_r` and `n`, never
    as a verdict, and never by the live gate.

    `baseline_kind` says which control these rows *are* (Batch-2 Task A); `pooled_rows` copies it
    onto every real row it scores against them. It defaults to the path shuffle, the only control
    that existed before, so an older caller keeps describing its rows correctly."""
    if baseline_kind not in BASELINE_KINDS:
        raise ValueError(f"baseline_kind must be one of {BASELINE_KINDS}, got {baseline_kind!r}")
    return [{**r, "source": BASELINE_SOURCE, "baseline_kind": baseline_kind}
            for r in pooled_rows(all_trades, computed_at)]
