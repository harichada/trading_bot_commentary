import json
import math
import time

import numpy as np
import pandas as pd
import pytest

from strategies.families.common import Family, make_entry, one_per_session


def _bars(start="2023-01-03", sessions=480, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=sessions)
    frames, px = [], 100.0
    for d in days:
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", name="ts")
        close = px + rng.normal(0, 0.05, 78).cumsum()
        frames.append(pd.DataFrame({"open": close - 0.02, "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 1000}, index=idx))
        px = float(close[-1])
    return pd.concat(frames)


def _toy(bars5, ind, daily, news, params):
    out = []
    for i in np.flatnonzero(ind["bar_of_day"].to_numpy() == 10):
        if i + 1 < len(bars5):
            px = float(bars5["open"].iloc[i + 1])
            e = make_entry(bars5, int(i), "long", px - params["w"], px + 2 * params["w"], 30, "intraday", "toy")
            if e:
                out.append(e)
    return one_per_session(out)


TOY = Family("toy", _toy, {"w": [0.5, 1.0]}, "intraday", ("long",))


@pytest.fixture
def toy_in_registry(monkeypatch):
    """Register `toy` so the scorer can read its hold kind. Batch-2 Task A: `score_regime_root`
    picks a pooled row's control by the family's `hold_kind`, and warns when a trades file names a
    family the registry does not have -- which these tapes otherwise would."""
    from strategies.families import registry

    monkeypatch.setitem(registry.FAMILIES, "toy", TOY)
    return TOY


@pytest.fixture
def root(tmp_path, monkeypatch):
    from research.strategy_hunt.data.bars import _write, bars5_path, daily_path, to_daily
    from strategies.families import registry
    b = _bars()
    for sym in ("AAA", "SPY"):
        _write(bars5_path(tmp_path, sym), b, "ts")
        _write(daily_path(tmp_path, sym), to_daily(b), "date")
    monkeypatch.setitem(registry.FAMILIES, "toy", TOY)
    return tmp_path


def test_shuffle_bars_preserves_session_return_multiset():
    from research.strategy_hunt.run import shuffle_bars
    b = _bars(sessions=3)
    s = shuffle_bars(b, seed=1)
    assert s.index.equals(b.index) and not np.allclose(s["close"], b["close"])
    for day in b.index.normalize().unique():
        o, n = b.loc[str(day.date())], s.loc[str(day.date())]
        ro = np.sort(np.diff(np.log(np.concatenate([[o["open"].iloc[0]], o["close"].to_numpy()]))))
        rn = np.sort(np.diff(np.log(np.concatenate([[n["open"].iloc[0]], n["close"].to_numpy()]))))
        assert np.allclose(ro, rn)
        assert (n["high"] >= n["close"]).all() and (n["low"] <= n["close"]).all()
    assert shuffle_bars(b, seed=1).equals(s)


def test_process_symbol_writes_outputs_then_skips(root):
    from research.strategy_hunt.run import process_symbol, trades_path, checkpoint_path, choices_path
    rec = process_symbol("AAA", root, ["toy"])
    assert rec["status"] == "done" and rec["n_trades"] > 0
    t = pd.read_parquet(trades_path(root, "AAA"))
    assert set(t["family"]) == {"toy"} and {"r_multiple", "window_id", "params_json", "regime"} <= set(t.columns)
    assert json.loads(choices_path(root, "AAA").read_text())["toy"][0]["window_id"] == 0
    assert checkpoint_path(root, "AAA").exists()
    assert process_symbol("AAA", root, ["toy"])["status"] == "skipped"
    assert process_symbol("AAA", root, ["toy"], force=True)["status"] == "done"
    assert process_symbol("AAA", root, ["toy"], cost_mult=2.0)["status"] == "done"     # different config re-runs
    assert process_symbol("ZZZ", root, ["toy"])["status"] == "empty"


def test_run_symbols_serial_and_score_root(root):
    from research.strategy_hunt.run import run_symbols, score_root
    recs = run_symbols(["AAA"], root, ["toy"], workers=1)
    assert [r["symbol"] for r in recs] == ["AAA"] and recs[0]["status"] == "done"
    rows = score_root(root, ["toy"], "2025-09-25T00:00:00", coldstart=False)
    assert len(rows) == 1 and rows[0]["strategy"] == "toy" and rows[0]["side"] == "long" and rows[0]["source"] == "walkforward"
    assert rows[0]["n"] > 0 and rows[0]["verdict"] in ("PASS", "WATCH", "FAIL")


def test_cli_run_and_score_end_to_end(root, monkeypatch, capsys, tmp_path):
    from research.strategy_hunt import cli
    from research.strategy_hunt import report as R
    from research.strategy_hunt import run as RUN
    from research.strategy_hunt.fit.writer import load_snapshot
    # v-strategy-fit-review2-2026-09-25 (Important 8): `run`'s manifest and
    # `score`'s report both land in REPORTS_DIR, which defaults to the real
    # repo path research/reports/strategy_hunt/ — a test must never write
    # there. run.py owns the constant; report.py binds its own module-level
    # copy at import time, so both need redirecting.
    reports = tmp_path / "reports"
    monkeypatch.setattr(RUN, "REPORTS_DIR", reports)
    monkeypatch.setattr(R, "REPORTS_DIR", reports)
    assert cli.main(["run", "--root", str(root), "--symbols", "AAA", "--families", "toy", "--workers", "1"]) == 0
    assert cli.main(["score", "--root", str(root), "--families", "toy", "--no-coldstart"]) == 0
    snap = load_snapshot(root)
    assert "AAA" in snap["rows"] and "toy:long" in snap["rows"]["AAA"]
    out = capsys.readouterr().out
    assert "AAA" in out


def test_process_symbol_uses_split_adjusted_bars(root, monkeypatch):
    from research.strategy_hunt import run as R
    from research.strategy_hunt.data import splits as S
    seen = {}
    real = S.load_adjusted
    def spy_load(r, s):
        seen["called"] = True
        return real(r, s)
    monkeypatch.setattr(R, "load_adjusted", spy_load)
    assert R.process_symbol("AAA", root, ["toy"], force=True)["status"] == "done" and seen["called"]


def test_checkpoint_invalidated_by_calibration_change(root):
    from research.strategy_hunt.run import process_symbol
    from research.strategy_hunt.sim.calibrate import write_calibration
    assert process_symbol("AAA", root, ["toy"])["status"] in ("done", "skipped")
    assert process_symbol("AAA", root, ["toy"])["status"] == "skipped"
    write_calibration(root, {"n": 40, "median_frac": 0.0003, "p75_frac": 0.001, "by_bucket": {}, "usable": True})
    assert process_symbol("AAA", root, ["toy"])["status"] == "done"


# ──────────────────────
# v-playbook-desk-b1-2026-09-25 — process_symbol threads the universe context into families
# ──────────────────────

def _ctx_toy(bars5, ind, daily, news, params):
    from strategies.families.context import in_play
    out = []
    mask = in_play(ind, 1) & (ind["bar_of_day"].to_numpy() == 10)
    for i in np.flatnonzero(mask):
        if i + 1 < len(bars5):
            px = float(bars5["open"].iloc[i + 1])
            e = make_entry(bars5, int(i), "long", px - 1.0, px + 2.0, 30, "intraday", "ctxtoy")
            if e:
                out.append(e)
    return one_per_session(out)


CTX_TOY = Family("ctxtoy", _ctx_toy, {"w": [1.0]}, "intraday", ("long",))


def _write_universe_context(root, symbol, dates):
    """Hand-written <root>/context/universe_daily.parquet with ctx_rvol_rank5 == 1 for `symbol`
    on every date -- everything `strategies.families.context.in_play` needs, without depending
    on `build_universe_context`'s earnings/news file plumbing."""
    from strategies.families.context import CTX_COLUMNS
    df = pd.DataFrame({c: np.nan for c in CTX_COLUMNS}, index=pd.DatetimeIndex(dates, name="date"))
    df["rvol_rank5"] = 1.0
    out = df.reset_index()
    out.insert(1, "symbol", symbol.upper())
    out = out[["date", "symbol", *CTX_COLUMNS]]
    ctx_dir = root / "context"
    ctx_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(ctx_dir / "universe_daily.parquet", index=False)


def test_process_symbol_passes_context_slice_to_families(root, monkeypatch):
    """A family that only proposes entries where ctx_rvol_rank5 ranks in the top-N must be silent
    when process_symbol has no universe context on disk yet (load_context_slice returns an empty
    frame), and must fire once research/strategy_hunt/context.py::build_universe_context (or, here,
    a hand-written equivalent) has written one."""
    from research.strategy_hunt.run import process_symbol
    from strategies.families import registry
    monkeypatch.setitem(registry.FAMILIES, "ctxtoy", CTX_TOY)

    rec = process_symbol("AAA", root, ["ctxtoy"], force=True)
    assert rec["status"] == "done" and rec["n_trades"] == 0

    dates = sorted(set(_bars().index.normalize()))
    _write_universe_context(root, "AAA", dates)
    rec2 = process_symbol("AAA", root, ["ctxtoy"], force=True)
    assert rec2["status"] == "done" and rec2["n_trades"] > 0


def test_version_bump_invalidates_checkpoint(root):
    """__version__ feeds _config_key, which is part of the checkpoint's stored config -- bumping
    it (0.2.0 -> 0.3.0 for this phase, so the newly-wired context reaches every symbol on the next
    nightly run) must force a re-run even though last_bar and every other config field match."""
    import json as _json
    from research.strategy_hunt.run import checkpoint_path, process_symbol
    assert process_symbol("AAA", root, ["toy"])["status"] == "done"
    assert process_symbol("AAA", root, ["toy"])["status"] == "skipped"
    cp = checkpoint_path(root, "AAA")
    rec = _json.loads(cp.read_text())
    rec["config"]["version"] = "0.0.1-stale"
    cp.write_text(_json.dumps(rec))
    assert process_symbol("AAA", root, ["toy"])["status"] == "done"


def test_version_is_0_3_0():
    from research.strategy_hunt import __version__
    assert __version__ == "0.3.0"


# ──────────────────────
# v-playbook-desk-b1-2026-09-25 — regime-conditioned and pooled scoring
# ──────────────────────

def _trade(symbol, family, side, day, w, regime, r):
    from research.strategy_hunt.models import SimTrade
    ts = pd.Timestamp(f"{day} 10:00")
    return SimTrade(symbol, family, side, ts, ts + pd.Timedelta(minutes=30), 100.0, 99.0, 102.0, 100.0 + r,
                    100.0, 100.0 + r, r, r, 0.0, "target" if r > 0 else "stop", 6, 1, f"{family}_{side}",
                    regime, w, '{"w": 1.0}')


def _write_trades(root, symbol, trades, variant=None):
    from research.strategy_hunt.models import TRADE_COLUMNS
    from research.strategy_hunt.run import choices_path, trades_path
    tp, cp = trades_path(root, symbol, variant), choices_path(root, symbol, variant)
    tp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([t.to_dict() for t in trades], columns=list(TRADE_COLUMNS)).to_parquet(tp, index=False)
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps({"toy": [{"window_id": w, "params": {"w": 1.0}} for w in range(3)]}))


def _regime_trades(symbol):
    out = []
    for di, d in enumerate(pd.bdate_range("2025-01-02", periods=30)):
        for j in range(10):
            regime = ("unknown", "risk_on", "mixed")[di % 3]
            out.append(_trade(symbol, "toy", "long", d.strftime("%Y-%m-%d"), di // 10, regime,
                              1.0 if (di * 10 + j) % 3 else -1.0))
    return out


def test_score_regime_root_builds_symbol_regime_and_pooled_rows(tmp_path):
    from research.strategy_hunt.run import score_regime_root
    from research.strategy_hunt.score.regime_rows import REGIMES
    for sym in ("AAA", "BBB"):
        _write_trades(tmp_path, sym, _regime_trades(sym))

    reg_rows, pool_rows, base_rows = score_regime_root(tmp_path, ["toy"], "2025-09-25T00:00:00")

    assert base_rows == []          # no trades_shuffle/ in this root
    assert {r["symbol"] for r in reg_rows} == {"AAA", "BBB"}
    assert {r["regime"] for r in reg_rows} <= set(REGIMES) and "unknown" not in {r["regime"] for r in reg_rows}
    assert all(r["strategy"] == "toy" and r["side"] == "long" for r in reg_rows)
    assert all(r["source"] == "walkforward" and r["computed_at"] == "2025-09-25T00:00:00" for r in reg_rows)
    assert all(json.loads(r["params_json"]) == {"w": 1.0} for r in reg_rows)
    # each regime row's window_end is that regime's own last exit date, not the family's
    assert len({r["window_end"] for r in reg_rows if r["symbol"] == "AAA"}) == 2

    assert pool_rows and all(r["symbol"] == "*" and r["source"] == "pooled" for r in pool_rows)
    by_regime = {r["regime"]: r for r in pool_rows}
    # pooled rows aggregate both symbols for the same (family, side, regime)
    per_symbol = {r["regime"]: r for r in reg_rows if r["symbol"] == "AAA"}
    for regime, prow in by_regime.items():
        assert prow["n"] == 2 * per_symbol[regime]["n"]


def test_score_regime_root_empty_root_and_score_root_unchanged(root):
    from research.strategy_hunt.run import run_symbols, score_regime_root, score_root
    assert score_regime_root(root, ["toy"], "2025-09-25T00:00:00") == ([], [], [])   # no trades yet
    # score_root keeps returning only the unconditional lane rows (the report and the
    # cold-start model consume that list).
    run_symbols(["AAA"], root, ["toy"], workers=1)
    rows = score_root(root, ["toy"], "2025-09-25T00:00:00", coldstart=False)
    assert len(rows) == 1 and "regime" not in rows[0]


def test_scoring_contains_one_unreadable_trades_file(tmp_path, caplog, toy_in_registry):
    """v-playbook-desk-b1-2026-09-25 (review r1, Minor 5): one corrupt Parquet must not
    abort the whole scoring pass -- the symbol is logged and skipped."""
    import logging

    from research.strategy_hunt.run import score_regime_root, score_root
    for sym in ("AAA", "BBB"):
        _write_trades(tmp_path, sym, _regime_trades(sym))
    (tmp_path / "trades" / "BAD.parquet").write_bytes(b"this is not a parquet file")

    with caplog.at_level(logging.WARNING, logger="strategy_hunt.run"):
        rows = score_root(tmp_path, ["toy"], "2025-09-25T00:00:00", coldstart=False)
        reg_rows, pool_rows, _base = score_regime_root(tmp_path, ["toy"], "2025-09-25T00:00:00")

    assert {r["symbol"] for r in rows} == {"AAA", "BBB"}
    assert {r["symbol"] for r in reg_rows} == {"AAA", "BBB"} and pool_rows
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 2 and all("BAD" in m for m in warnings)      # both scorers, once each


# ──────────────────────
# v-playbook-desk-b1-5 task 10 — the shuffled baseline is a variant inside the same root
# ──────────────────────

def test_variant_paths_use_suffixed_directories(tmp_path):
    from research.strategy_hunt.paths import SHUFFLE_VARIANT, VARIANTS
    from research.strategy_hunt.run import checkpoint_path, choices_path, trades_path

    assert trades_path(tmp_path, "AAA") == tmp_path / "trades" / "AAA.parquet"
    assert trades_path(tmp_path, "aaa", "shuffle") == tmp_path / "trades_shuffle" / "AAA.parquet"
    assert checkpoint_path(tmp_path, "AAA", "shuffle") == tmp_path / "checkpoints_shuffle" / "AAA.json"
    assert choices_path(tmp_path, "AAA", "shuffle") == tmp_path / "choices_shuffle" / "AAA.json"
    assert SHUFFLE_VARIANT == "shuffle" and "shuffle" in VARIANTS
    with pytest.raises(ValueError):
        trades_path(tmp_path, "AAA", "nonsense")


def test_process_symbol_variant_writes_only_the_shuffle_dirs(root):
    """The shuffle control no longer needs its own root: `variant="shuffle"` sends every
    artefact to the `_shuffle` siblings so the real tape's trades/ stays untouched."""
    from research.strategy_hunt.run import checkpoint_path, choices_path, process_symbol, trades_path

    rec = process_symbol("AAA", root, ["toy"], shuffle_seed=7, variant="shuffle")
    assert rec["status"] == "done" and rec["n_trades"] >= 0
    assert trades_path(root, "AAA", "shuffle").exists()
    assert checkpoint_path(root, "AAA", "shuffle").exists()
    assert choices_path(root, "AAA", "shuffle").exists()
    assert not (root / "trades").exists() and not (root / "checkpoints").exists()
    # and the variant's checkpoint is its own: a second call skips without touching trades/
    assert process_symbol("AAA", root, ["toy"], shuffle_seed=7, variant="shuffle")["status"] == "skipped"
    assert not (root / "trades").exists()


def test_run_symbols_threads_the_variant(root):
    from research.strategy_hunt.run import run_symbols, trades_path

    recs = run_symbols(["AAA"], root, ["toy"], workers=1, variant="shuffle", shuffle_seed=7)
    assert [r["symbol"] for r in recs] == ["AAA"]
    assert trades_path(root, "AAA", "shuffle").exists() and not (root / "trades").exists()


def _flat_trades(symbol, r, sessions=60, per_day=10, regime="risk_on", start="2025-01-02", step=5):
    """Constant-r trades spread over five calendar quarters (pooled rows bucket by quarter)."""
    out = []
    for di, d in enumerate(pd.bdate_range(start, periods=sessions * step)[::step]):
        for _ in range(per_day):
            out.append(_trade(symbol, "toy", "long", d.strftime("%Y-%m-%d"), di // 10, regime, r))
    return out


def test_score_regime_root_computes_a_baseline_from_trades_shuffle(tmp_path):
    from research.strategy_hunt.run import score_regime_root
    from research.strategy_hunt.score.regime_rows import BASELINE_SOURCE

    _write_trades(tmp_path, "AAA", _flat_trades("AAA", 0.30))
    _write_trades(tmp_path, "AAA", _flat_trades("AAA", 0.10), variant="shuffle")

    reg_rows, pool_rows, base_rows = score_regime_root(tmp_path, ["toy"], "2025-09-25T00:00:00")

    assert base_rows and all(r["source"] == BASELINE_SOURCE == "baseline" for r in base_rows)
    assert all(r["symbol"] == "*" and r["regime"] == "risk_on" for r in base_rows)
    assert base_rows[0]["exp_r"] == pytest.approx(0.10)
    assert len(pool_rows) == 1
    assert pool_rows[0]["baseline_exp_r"] == pytest.approx(0.10) and pool_rows[0]["baseline_n"] == 600
    assert pool_rows[0]["excess_r"] == pytest.approx(0.20)
    # per-symbol regime rows are untouched by the baseline
    assert reg_rows and all("baseline_exp_r" in r for r in reg_rows)
    assert all(math.isnan(r["baseline_exp_r"]) for r in reg_rows)


def test_score_regime_root_without_trades_shuffle_reports_no_baseline(tmp_path):
    import json as _json
    from research.strategy_hunt.run import score_regime_root

    _write_trades(tmp_path, "AAA", _flat_trades("AAA", 0.30))
    reg_rows, pool_rows, base_rows = score_regime_root(tmp_path, ["toy"], "2025-09-25T00:00:00")
    assert base_rows == [] and len(pool_rows) == 1
    assert math.isnan(pool_rows[0]["excess_r"]) and pool_rows[0]["baseline_n"] == 0
    assert "no_baseline" in _json.loads(pool_rows[0]["failed_floors"])
    assert pool_rows[0]["verdict"] != "PASS"


# ──────────────────────
# v-playbook-desk-b1-5 task 10 fix r1 — baseline coverage / staleness guard in score_regime_root
# ──────────────────────

def test_score_regime_root_flags_a_partial_baseline(tmp_path, caplog):
    """A leftover / half-finished `trades_shuffle/` is worse than none: it silently becomes the
    yardstick for every pooled verdict. Scoring flags it instead of trusting it."""
    import logging

    from research.strategy_hunt.run import score_regime_root

    real = [f"S{i:03d}" for i in range(20)]
    for sym in real:
        _write_trades(tmp_path, sym, _flat_trades(sym, 0.30))
    for sym in real[:5]:                                   # 75% of the universe has no baseline
        _write_trades(tmp_path, sym, _flat_trades(sym, 0.05), variant="shuffle")

    with caplog.at_level(logging.WARNING, logger="strategy_hunt.run"):
        _reg, pool, base = score_regime_root(tmp_path, ["toy"], "2025-09-25T00:00:00")

    assert base and pool
    failed = json.loads(pool[0]["failed_floors"])
    assert "baseline_mismatch" in failed and pool[0]["verdict"] != "PASS"
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("baseline_mismatch" in m for m in warnings)
    assert any("20" in m and "5" in m for m in warnings)    # the counts are in the warning


def test_score_regime_root_does_not_flag_a_complete_fresh_baseline(tmp_path, caplog, toy_in_registry):
    import logging

    from research.strategy_hunt.run import score_regime_root

    real = [f"S{i:03d}" for i in range(20)]
    for sym in real:
        _write_trades(tmp_path, sym, _flat_trades(sym, 0.30))
        _write_trades(tmp_path, sym, _flat_trades(sym, 0.05), variant="shuffle")

    with caplog.at_level(logging.WARNING, logger="strategy_hunt.run"):
        _reg, pool, _base = score_regime_root(tmp_path, ["toy"], "2025-09-25T00:00:00")

    failed = json.loads(pool[0]["failed_floors"])
    assert "baseline_mismatch" not in failed and "baseline_stale" not in failed
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []


# ──────────────────────
# strategy-search-batch1 task 2 — holdout_months threads through process_symbol/run_symbols,
# invalidates the checkpoint, and is recorded in the run manifest
# ──────────────────────

def test_config_key_includes_holdout_months():
    from research.strategy_hunt.run import _config_key
    k0 = _config_key(["toy"], 1.0, None, 6000.0, None, holdout_months=0)
    k6 = _config_key(["toy"], 1.0, None, 6000.0, None, holdout_months=6)
    assert k0["holdout_months"] == 0 and k6["holdout_months"] == 6 and k0 != k6
    # default stays 0 so every existing call site (no holdout_months kwarg) is unaffected
    assert _config_key(["toy"], 1.0, None, 6000.0, None) == k0


def test_process_symbol_holdout_months_reduces_windows_and_invalidates_checkpoint(root):
    from research.strategy_hunt.run import process_symbol
    base = process_symbol("AAA", root, ["toy"], force=True)
    assert base["status"] == "done"
    held = process_symbol("AAA", root, ["toy"], holdout_months=6, force=True)
    assert held["status"] == "done" and held["n_windows"] < base["n_windows"]
    # same holdout_months, no --force: checkpoint from the held-out run is reused
    assert process_symbol("AAA", root, ["toy"], holdout_months=6)["status"] == "skipped"
    # holdout_months is part of _config_key, so going back to 0 re-runs rather than skipping
    assert process_symbol("AAA", root, ["toy"], holdout_months=0)["status"] == "done"


def test_run_symbols_threads_holdout_months(root):
    from research.strategy_hunt.run import run_symbols
    base = run_symbols(["AAA"], root, ["toy"], workers=1, force=True)[0]
    held = run_symbols(["AAA"], root, ["toy"], workers=1, force=True, holdout_months=6)[0]
    assert held["status"] == "done" and held["n_windows"] < base["n_windows"]


def test_cli_run_records_holdout_months_and_families_in_manifest(root, monkeypatch, tmp_path):
    from research.strategy_hunt import cli
    from research.strategy_hunt import report as R
    from research.strategy_hunt import run as RUN

    reports = tmp_path / "reports"
    monkeypatch.setattr(RUN, "REPORTS_DIR", reports)
    monkeypatch.setattr(R, "REPORTS_DIR", reports)
    assert cli.main(["run", "--root", str(root), "--symbols", "AAA", "--families", "toy",
                     "--workers", "1", "--holdout-months", "6"]) == 0
    manifest = json.loads((RUN.reports_dir(root) / f"{time.strftime('%Y-%m-%d')}.json").read_text())
    assert manifest["run"]["holdout_months"] == 6
    assert manifest["run"]["families"] == ["toy"]
