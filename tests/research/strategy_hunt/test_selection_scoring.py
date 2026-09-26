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
