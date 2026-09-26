"""Batch-2 Task A: which control a pooled row is scored against, and what it records about it.

An `intraday` family's pooled row keeps the path-shuffle baseline (`trades_shuffle/`) -- permuting
5-minute returns inside a session is a real control for an intraday hold. A `multiday`/`overnight`
family's row must use the selection baseline (`trades_selection/`) instead, because the shuffle
preserves each session's open-to-close return and therefore cannot test a multi-day hold at all.
Every pooled row says which control it used in `baseline_kind`, and a row whose needed control is
missing records `no_baseline` rather than quietly borrowing the other one.
"""
import json
import math

import pandas as pd
import pytest

from research.strategy_hunt.models import TRADE_COLUMNS

INTRADAY_FAM = "toy_intra"
MULTI_FAM = "toy_multi"
COMPUTED_AT = "2026-09-26T00:00:00"


def _rows(symbol: str, family: str, r: float, n: int = 12) -> list[dict]:
    day = pd.Timestamp("2026-01-05 10:00")
    out = []
    for k in range(n):
        ts = day + pd.Timedelta(days=k)
        out.append({"symbol": symbol, "family": family, "side": "long", "entry_ts": ts,
                    "exit_ts": ts + pd.Timedelta(hours=2), "entry_px": 100.0, "stop_px": 99.0,
                    "target_px": 103.0, "exit_px": 101.0, "entry_fill": 100.0, "exit_fill": 101.0,
                    "r_multiple": r, "pnl_pct": 1.0, "cost_pct": 0.05, "exit_reason": "target",
                    "hold_bars": 5, "hold_sessions": 1, "tag": family, "regime": "mixed",
                    "window_id": 1, "params_json": json.dumps({"time_stop_days": 2})})
    return out


def _write(root, symbol: str, rows: list[dict], variant=None) -> None:
    from research.strategy_hunt.paths import trades_path

    tp = trades_path(root, symbol, variant)
    tp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).reindex(columns=list(TRADE_COLUMNS)).to_parquet(tp, index=False)


@pytest.fixture
def families(monkeypatch):
    from strategies.families import registry
    from strategies.families.common import Family

    for name, kind in ((INTRADAY_FAM, "intraday"), (MULTI_FAM, "multiday")):
        monkeypatch.setitem(registry.FAMILIES, name,
                            Family(name, lambda *a, **kw: [], {"w": [1]}, kind, ("long",)))
    return (INTRADAY_FAM, MULTI_FAM)


@pytest.fixture
def root(tmp_path, families):
    """Real tape at exp_R 0.30, path-shuffle tape at 0.10 (both families), selection tape at 0.20."""
    _write(tmp_path, "AAA", _rows("AAA", INTRADAY_FAM, 0.30) + _rows("AAA", MULTI_FAM, 0.30))
    _write(tmp_path, "AAA", _rows("AAA", INTRADAY_FAM, 0.10) + _rows("AAA", MULTI_FAM, 0.10),
           variant="shuffle")
    _write(tmp_path, "AAA", _rows("AAA", MULTI_FAM, 0.20), variant="selection")
    return tmp_path


def _pooled_by_family(pooled: list[dict]) -> dict[str, dict]:
    return {r["strategy"]: r for r in pooled}


def test_an_intraday_pooled_row_uses_the_path_shuffle_baseline(root, families):
    from research.strategy_hunt.run import score_regime_root
    from research.strategy_hunt.score.regime_rows import BASELINE_KIND_PATH

    _per_symbol, pooled, _baseline = score_regime_root(root, list(families), COMPUTED_AT)

    row = _pooled_by_family(pooled)[INTRADAY_FAM]
    assert row["baseline_kind"] == BASELINE_KIND_PATH
    assert row["baseline_exp_r"] == pytest.approx(0.10)
    assert row["excess_r"] == pytest.approx(0.20)


def test_a_multiday_pooled_row_uses_the_selection_baseline(root, families):
    from research.strategy_hunt.run import score_regime_root
    from research.strategy_hunt.score.regime_rows import BASELINE_KIND_SELECTION

    _per_symbol, pooled, _baseline = score_regime_root(root, list(families), COMPUTED_AT)

    row = _pooled_by_family(pooled)[MULTI_FAM]
    assert row["baseline_kind"] == BASELINE_KIND_SELECTION
    assert row["baseline_exp_r"] == pytest.approx(0.20)       # the selection tape, not the 0.10 shuffle
    assert row["excess_r"] == pytest.approx(0.10)


def test_baseline_rows_record_which_control_they_are(root, families):
    from research.strategy_hunt.run import score_regime_root
    from research.strategy_hunt.score.regime_rows import (BASELINE_KIND_PATH, BASELINE_KIND_SELECTION,
                                                          BASELINE_SOURCE)

    _per_symbol, _pooled, baseline = score_regime_root(root, list(families), COMPUTED_AT)

    kinds = {(r["strategy"], r["baseline_kind"]) for r in baseline}
    assert kinds == {(INTRADAY_FAM, BASELINE_KIND_PATH), (MULTI_FAM, BASELINE_KIND_SELECTION)}
    assert {r["source"] for r in baseline} == {BASELINE_SOURCE}


def test_a_multiday_row_without_a_selection_tape_is_no_baseline_not_the_shuffle_one(root, families):
    from research.strategy_hunt.paths import trades_path
    from research.strategy_hunt.run import score_regime_root
    from research.strategy_hunt.score.regime_rows import BASELINE_KIND_NONE

    trades_path(root, "AAA", "selection").unlink()

    _per_symbol, pooled, baseline = score_regime_root(root, list(families), COMPUTED_AT)

    row = _pooled_by_family(pooled)[MULTI_FAM]
    assert row["baseline_kind"] == BASELINE_KIND_NONE
    assert math.isnan(row["baseline_exp_r"]) and math.isnan(row["excess_r"])
    assert "no_baseline" in json.loads(row["failed_floors"])
    assert row["verdict"] != "PASS"
    # the intraday row still has its own control
    assert _pooled_by_family(pooled)[INTRADAY_FAM]["baseline_exp_r"] == pytest.approx(0.10)
    assert {r["strategy"] for r in baseline} == {INTRADAY_FAM}


def test_an_intraday_row_without_a_shuffle_tape_is_no_baseline(root, families):
    from research.strategy_hunt.paths import trades_path
    from research.strategy_hunt.run import score_regime_root
    from research.strategy_hunt.score.regime_rows import BASELINE_KIND_NONE

    trades_path(root, "AAA", "shuffle").unlink()

    _per_symbol, pooled, _baseline = score_regime_root(root, list(families), COMPUTED_AT)

    row = _pooled_by_family(pooled)[INTRADAY_FAM]
    assert row["baseline_kind"] == BASELINE_KIND_NONE
    assert "no_baseline" in json.loads(row["failed_floors"])


def test_pooled_rows_are_emitted_for_every_family_and_sorted(root, families):
    from research.strategy_hunt.run import score_regime_root

    _per_symbol, pooled, _baseline = score_regime_root(root, list(families), COMPUTED_AT)

    keys = [(r["strategy"], r["side"], r["regime"]) for r in pooled]
    assert keys == sorted(keys)
    assert {k[0] for k in keys} == set(families)


# ──────────────────────────────  fix round 1  ──────────────────────────────


def test_a_selection_control_that_replayed_too_few_trades_cannot_pass(root, families):
    """Fix round 1, Important 2: the selection control is *paired* with the real tape, so a control
    group materially smaller than its real group means trades were dropped, not that the lane has an
    edge over it. `baseline_population_mismatch` (a 2x band) is far too loose to catch that."""
    from research.strategy_hunt.paths import trades_path
    from research.strategy_hunt.run import score_regime_root
    from research.strategy_hunt.score.regime_rows import BASELINE_UNDERREPLAYED

    _write(root, "AAA", _rows("AAA", MULTI_FAM, 0.20, n=10), variant="selection")   # 10 of 12 replayed

    _per_symbol, pooled, _baseline = score_regime_root(root, list(families), COMPUTED_AT)

    row = _pooled_by_family(pooled)[MULTI_FAM]
    assert row["baseline_n"] == 10 and row["n"] == 12
    assert BASELINE_UNDERREPLAYED in json.loads(row["failed_floors"])
    assert row["verdict"] != "PASS"
    # the fully paired intraday row against its path control carries no such key
    assert BASELINE_UNDERREPLAYED not in json.loads(_pooled_by_family(pooled)[INTRADAY_FAM]["failed_floors"])


def test_a_fully_replayed_selection_control_carries_no_underreplayed_key(root, families):
    from research.strategy_hunt.run import score_regime_root
    from research.strategy_hunt.score.regime_rows import BASELINE_UNDERREPLAYED

    _per_symbol, pooled, _baseline = score_regime_root(root, list(families), COMPUTED_AT)

    row = _pooled_by_family(pooled)[MULTI_FAM]
    assert row["baseline_n"] == row["n"] == 12
    assert BASELINE_UNDERREPLAYED not in json.loads(row["failed_floors"])


def test_a_thin_path_control_is_not_judged_by_the_pairing_floor(root, families):
    """The shuffled tape re-runs the families on a different tape, so its group sizes legitimately
    differ; only the 2x population band applies to it."""
    from research.strategy_hunt.run import score_regime_root
    from research.strategy_hunt.score.regime_rows import BASELINE_UNDERREPLAYED

    _write(root, "AAA", _rows("AAA", INTRADAY_FAM, 0.10, n=10) + _rows("AAA", MULTI_FAM, 0.10, n=10),
           variant="shuffle")

    _per_symbol, pooled, _baseline = score_regime_root(root, list(families), COMPUTED_AT)

    assert BASELINE_UNDERREPLAYED not in json.loads(_pooled_by_family(pooled)[INTRADAY_FAM]["failed_floors"])


def test_baseline_guard_skips_the_coverage_test_for_the_selection_control(root, families):
    """Fix round 1, Important 3: peers are a random sample of the universe, so "this control has no
    row for symbol X" is the design, not a gap. Staleness still applies."""
    import pandas as pd

    from research.strategy_hunt.score.regime_rows import (BASELINE_KIND_PATH, BASELINE_KIND_SELECTION,
                                                          BASELINE_MISMATCH, BASELINE_STALE, baseline_guard)

    real = pd.concat([pd.DataFrame(_rows(f"S{i:02d}", MULTI_FAM, 0.3)) for i in range(20)],
                     ignore_index=True)
    thin = pd.concat([pd.DataFrame(_rows(f"S{i:02d}", MULTI_FAM, 0.2)) for i in range(5)],
                     ignore_index=True)

    assert baseline_guard(real, thin) == [BASELINE_MISMATCH]
    assert baseline_guard(real, thin, baseline_kind=BASELINE_KIND_PATH) == [BASELINE_MISMATCH]
    assert baseline_guard(real, thin, baseline_kind=BASELINE_KIND_SELECTION) == []

    stale = thin.copy()
    stale["exit_ts"] = pd.to_datetime(stale["exit_ts"]) - pd.Timedelta(days=30)
    assert baseline_guard(real, stale, baseline_kind=BASELINE_KIND_SELECTION) == [BASELINE_STALE]


def test_a_selection_control_covering_few_symbols_is_not_flagged_on_the_pooled_row(root, families):
    from research.strategy_hunt.paths import trades_path
    from research.strategy_hunt.run import score_regime_root
    from research.strategy_hunt.score.regime_rows import BASELINE_MISMATCH

    # the real multiday tape spans 20 symbols; the control lands on 2 peers, as a random draw does
    for i in range(20):
        _write(root, f"S{i:02d}", _rows(f"S{i:02d}", MULTI_FAM, 0.30))
    trades_path(root, "AAA", "selection").unlink()
    for peer in ("P1", "P2"):
        _write(root, peer, _rows(peer, MULTI_FAM, 0.20, n=120), variant="selection")

    _per_symbol, pooled, _baseline = score_regime_root(root, list(families), COMPUTED_AT)

    failed = json.loads(_pooled_by_family(pooled)[MULTI_FAM]["failed_floors"])
    assert BASELINE_MISMATCH not in failed
