"""CLI-level tests: universe-context wiring and cmd_score -> score_regime_root -> write_fit."""
import numpy as np
import pandas as pd
import pytest

from strategies.families.common import Family, make_entry, one_per_session


def _bars(start="2023-01-03", sessions=60, seed=0):
    rng = np.random.default_rng(seed)
    frames, px = [], 100.0
    for d in pd.bdate_range(start, periods=sessions):
        idx = pd.date_range(f"{d.date()} 09:30", periods=78, freq="5min", name="ts")
        close = px + rng.normal(0, 0.05, 78).cumsum()
        frames.append(pd.DataFrame({"open": close - 0.02, "high": close + 0.3, "low": close - 0.3,
                                    "close": close, "volume": 1000}, index=idx))
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


TOY = Family("toy", _toy, {"w": [1.0]}, "intraday", ("long",))


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


def test_cmd_score_calls_score_regime_root_and_snapshot_has_regime_and_pooled_keys(root, monkeypatch, tmp_path):
    """The controller's Task 6 report flagged that cmd_score never called score_regime_root, so
    production snapshots kept empty regime_rows/pooled_rows forever (lookup_regime always falling
    back to lookup()). This proves the wiring: score_regime_root is actually invoked by `score`,
    and its two return values land in the persisted strategy_fit.json under those two keys -- even
    if this toy dataset is too small to produce any rows in them (no VIX means every trade's regime
    is "unknown", which regime_rows/pooled_rows deliberately drop; see score/regime_rows.py).

    Fix round 1 (finding 4): also proves write_fit is handed *exactly* the two lists
    score_regime_root returned, not merely that the persisted json has the right shape."""
    import json as _json

    from research.strategy_hunt import cli
    from research.strategy_hunt import report as R
    from research.strategy_hunt import run as RUN
    from research.strategy_hunt.fit.writer import fit_dir

    reports = tmp_path / "reports"
    monkeypatch.setattr(RUN, "REPORTS_DIR", reports)
    monkeypatch.setattr(R, "REPORTS_DIR", reports)

    calls = []
    regime_results = []
    real_score_regime_root = cli.score_regime_root

    def _recording(root_, families, computed_at):
        calls.append((root_, families, computed_at))
        result = real_score_regime_root(root_, families, computed_at)
        regime_results.append(result)
        return result

    monkeypatch.setattr(cli, "score_regime_root", _recording)

    write_fit_calls = []
    real_write_fit = cli.write_fit

    def _recording_write_fit(rows, root_, computed_at, **kw):
        write_fit_calls.append(kw)
        return real_write_fit(rows, root_, computed_at, **kw)

    monkeypatch.setattr(cli, "write_fit", _recording_write_fit)

    assert cli.main(["run", "--root", str(root), "--symbols", "AAA", "--families", "toy", "--workers", "1"]) == 0
    assert cli.main(["score", "--root", str(root), "--families", "toy", "--no-coldstart"]) == 0

    assert len(calls) == 1
    called_root, called_families, _ = calls[0]
    assert str(called_root) == str(root) and called_families == ["toy"]

    snap = _json.loads((fit_dir(root) / "strategy_fit.json").read_text())
    assert "regime_rows" in snap and "pooled_rows" in snap and "baseline_pooled_rows" in snap
    assert isinstance(snap["regime_rows"], dict) and isinstance(snap["pooled_rows"], dict)

    # finding 4: write_fit received exactly what score_regime_root returned -- not merely
    # something of the right shape.
    assert len(write_fit_calls) == 1 and len(regime_results) == 1
    reg_rows, pool_rows, base_rows = regime_results[0]
    assert write_fit_calls[0]["regime_rows"] == reg_rows
    assert write_fit_calls[0]["pooled_rows"] == pool_rows
    assert write_fit_calls[0]["baseline_pooled_rows"] == base_rows


# ──────────────────────
# Fix round 1 (task-5-findings-r1.md) — cmd_all order, fatal universe-context build failure
# ──────────────────────

_ALL_STEPS = ("cmd_export", "cmd_news", "cmd_earnings", "cmd_context", "cmd_calibrate", "cmd_run", "cmd_score")


def _record_all_steps(cli, monkeypatch):
    """Replace every cmd_all step with a recorder of (name, shuffle_seed) and return the log."""
    order = []

    def _step(name):
        def _fn(a):
            order.append((name, getattr(a, "shuffle_seed", None)))
            return 0
        return _fn

    for name in _ALL_STEPS:
        monkeypatch.setattr(cli, name, _step(name))
    return order


def test_cmd_all_runs_context_after_news_and_earnings(monkeypatch, tmp_path):
    """Finding 1: the universe context must be built after news/earnings land, so its ctx_news_*/
    ctx_earnings_* columns reflect tonight's data by the time `run` consumes them -- not before.

    v-playbook-desk-b1-5 task 10: `run` now happens twice -- the real tape, then the shuffled
    baseline (seed 7) into the same root's `_shuffle` dirs -- before `score` reads both."""
    from research.strategy_hunt import cli

    order = _record_all_steps(cli, monkeypatch)
    assert cli.cmd_all(cli.build_parser().parse_args(["all", "--root", str(tmp_path)])) == 0
    assert order == [("cmd_export", None), ("cmd_news", None), ("cmd_earnings", None), ("cmd_context", None),
                     ("cmd_calibrate", None), ("cmd_run", None), ("cmd_run", cli.BASELINE_SHUFFLE_SEED),
                     ("cmd_score", None)]


def test_cmd_all_no_baseline_skips_the_shuffled_run(monkeypatch, tmp_path):
    from research.strategy_hunt import cli

    order = _record_all_steps(cli, monkeypatch)
    assert cli.cmd_all(cli.build_parser().parse_args(["all", "--root", str(tmp_path), "--no-baseline"])) == 0
    assert [n for n, _ in order] == list(_ALL_STEPS)
    assert [seed for _, seed in order] == [None] * len(_ALL_STEPS)


def test_cmd_all_baseline_run_is_forced_and_leaves_the_real_run_namespace_alone(monkeypatch, tmp_path):
    """The baseline must re-run even when the real run's checkpoints are warm (force=True), and
    must not mutate the namespace the later steps see."""
    from research.strategy_hunt import cli

    seen = []
    for name in ("cmd_export", "cmd_news", "cmd_earnings", "cmd_context", "cmd_calibrate", "cmd_score"):
        monkeypatch.setattr(cli, name, lambda a: 0)
    monkeypatch.setattr(cli, "cmd_run", lambda a: seen.append((a.shuffle_seed, a.force)) or 0)

    a = cli.build_parser().parse_args(["all", "--root", str(tmp_path)])
    assert cli.cmd_all(a) == 0
    assert seen == [(None, False), (cli.BASELINE_SHUFFLE_SEED, True)]
    assert a.shuffle_seed is None and a.force is False


def test_cmd_all_stops_when_the_baseline_run_fails(monkeypatch, tmp_path):
    from research.strategy_hunt import cli

    calls = []
    for name in ("cmd_export", "cmd_news", "cmd_earnings", "cmd_context", "cmd_calibrate"):
        monkeypatch.setattr(cli, name, lambda a: 0)
    monkeypatch.setattr(cli, "cmd_run", lambda a: 1 if a.shuffle_seed is not None else 0)
    monkeypatch.setattr(cli, "cmd_score", lambda a: calls.append("cmd_score") or 0)

    assert cli.cmd_all(cli.build_parser().parse_args(["all", "--root", str(tmp_path)])) == 1
    assert calls == []


def test_cmd_run_with_a_shuffle_seed_uses_the_shuffle_variant(monkeypatch, tmp_path):
    from research.strategy_hunt import cli
    from research.strategy_hunt import run as RUN

    monkeypatch.setattr(RUN, "REPORTS_DIR", tmp_path / "reports")
    seen = {}

    def _fake_run_symbols(symbols, root, families, workers=1, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(cli, "run_symbols", _fake_run_symbols)
    assert cli.main(["run", "--root", str(tmp_path), "--symbols", "AAA", "--families", "toy"]) == 0
    assert seen["variant"] is None
    assert cli.main(["run", "--root", str(tmp_path), "--symbols", "AAA", "--families", "toy",
                     "--shuffle-seed", "7"]) == 0
    assert seen["variant"] == "shuffle" and seen["shuffle_seed"] == 7


def test_cmd_run_manifest_keeps_the_baseline_run_separate(monkeypatch, tmp_path):
    """Both runs write a manifest for the same date; keying the shuffled one separately stops it
    from overwriting the real run's record."""
    import json as _json
    from research.strategy_hunt import cli
    from research.strategy_hunt import run as RUN

    reports = tmp_path / "reports"
    monkeypatch.setattr(RUN, "REPORTS_DIR", reports)
    monkeypatch.setattr(cli, "run_symbols", lambda symbols, root, families, workers=1, **kw: [])
    assert cli.main(["run", "--root", str(tmp_path), "--symbols", "AAA", "--families", "toy"]) == 0
    assert cli.main(["run", "--root", str(tmp_path), "--symbols", "AAA", "--families", "toy",
                     "--shuffle-seed", "7"]) == 0
    rec = _json.loads(next(reports.glob("*.json")).read_text())
    assert rec["run"]["shuffle_seed"] is None and rec["run_shuffle"]["shuffle_seed"] == 7


def test_cmd_context_returns_1_and_prints_clean_error_when_universe_context_build_fails(monkeypatch, tmp_path, capsys):
    """Finding 2: a build_universe_context failure (e.g. SPY daily missing) must be fatal -- a
    clean one-line error and rc=1, not a swallowed warning that lets the pipeline limp on with a
    stale/missing context file."""
    from research.strategy_hunt import cli
    from research.strategy_hunt import context as ctx_module
    from research.strategy_hunt.data import context as data_context

    monkeypatch.setattr(data_context, "ensure_context", lambda root, sess, key, secret: {"spy": 0, "vix": 0})

    def _boom(root):
        raise ValueError("SPY daily data missing/empty for 'SPY' under root (every symbol's rs_* columns depend on it)")

    monkeypatch.setattr(ctx_module, "build_universe_context", _boom)

    rc = cli.main(["context", "--root", str(tmp_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "context: universe ERROR" in out and "SPY daily data missing" in out
    assert "Traceback" not in out


def test_cmd_all_stops_before_calibrate_when_universe_context_build_fails(monkeypatch, tmp_path):
    """Finding 2 (cmd_all integration): a fatal cmd_context must stop the pipeline before
    cmd_calibrate/cmd_run/cmd_score ever run."""
    from research.strategy_hunt import cli
    from research.strategy_hunt import context as ctx_module
    from research.strategy_hunt.data import context as data_context

    monkeypatch.setattr(cli, "cmd_export", lambda a: 0)
    monkeypatch.setattr(cli, "cmd_news", lambda a: 0)
    monkeypatch.setattr(cli, "cmd_earnings", lambda a: 0)
    monkeypatch.setattr(data_context, "ensure_context", lambda root, sess, key, secret: {"spy": 0, "vix": 0})
    monkeypatch.setattr(ctx_module, "build_universe_context", lambda root: (_ for _ in ()).throw(ValueError("SPY missing")))

    downstream_calls = []
    monkeypatch.setattr(cli, "cmd_calibrate", lambda a: downstream_calls.append("cmd_calibrate") or 0)
    monkeypatch.setattr(cli, "cmd_run", lambda a: downstream_calls.append("cmd_run") or 0)
    monkeypatch.setattr(cli, "cmd_score", lambda a: downstream_calls.append("cmd_score") or 0)

    rc = cli.main(["all", "--root", str(tmp_path)])
    assert rc == 1
    assert downstream_calls == []


def test_cmd_universe_context_returns_1_and_prints_clean_error_on_failure(monkeypatch, tmp_path, capsys):
    """Finding 2 also covers the standalone `universe-context` subcommand: today it lets a raw
    traceback through since it has no try/except at all."""
    from research.strategy_hunt import cli
    from research.strategy_hunt import context as ctx_module

    def _boom(root):
        raise ValueError("SPY daily data missing/empty for 'SPY'")

    monkeypatch.setattr(ctx_module, "build_universe_context", _boom)

    rc = cli.main(["universe-context", "--root", str(tmp_path)])
    assert rc == 1
    out = capsys.readouterr().out
    assert "universe-context ERROR" in out and "SPY daily data missing" in out
    assert "Traceback" not in out


# ──────────────────────
# Task 7b — `news --rescore`
# ──────────────────────


def test_cmd_news_rescore_calls_rescore_news_per_symbol_and_never_update_news(monkeypatch, tmp_path):
    """--rescore must re-score the existing cache via rescore_news for every requested symbol,
    print its count, and never touch update_news (no fetch, no Alpaca creds needed)."""
    from research.strategy_hunt import cli
    from research.strategy_hunt.data import news as N

    rescore_calls = []
    monkeypatch.setattr(N, "rescore_news", lambda root, sym, backend=None: rescore_calls.append(sym) or 3)

    update_calls = []
    monkeypatch.setattr(N, "update_news", lambda *a, **kw: update_calls.append(a) or 0)

    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    rc = cli.main(["news", "--root", str(tmp_path), "--symbols", "AAA,BBB", "--rescore"])
    assert rc == 0
    assert rescore_calls == ["AAA", "BBB"]
    assert update_calls == []


def test_cmd_news_rescore_continues_past_per_symbol_errors(monkeypatch, tmp_path, capsys):
    from research.strategy_hunt import cli
    from research.strategy_hunt.data import news as N

    def _rescore(root, sym, backend=None):
        if sym == "AAA":
            raise RuntimeError("boom")
        return 1

    monkeypatch.setattr(N, "rescore_news", _rescore)

    rc = cli.main(["news", "--root", str(tmp_path), "--symbols", "AAA,BBB", "--rescore"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "news AAA: ERROR boom" in out
    assert "news BBB: rescored 1" in out


def test_news_rescore_flag_only_on_news_subparser():
    from research.strategy_hunt.cli import build_parser

    p = build_parser()
    ns = p.parse_args(["news", "--rescore"])
    assert ns.rescore is True
    with pytest.raises(SystemExit):
        p.parse_args(["export", "--rescore"])


def test_universe_context_subcommand_has_only_root_flag():
    """Finding 5: `universe-context` doesn't need --symbols/--families/--workers/etc -- it takes
    no arguments beyond --root, like `promote`."""
    from research.strategy_hunt.cli import build_parser

    p = build_parser()
    ns = p.parse_args(["universe-context", "--root", "/tmp/somewhere"])
    assert str(ns.root) == "/tmp/somewhere"
    with pytest.raises(SystemExit):
        p.parse_args(["universe-context", "--symbols", "AAA"])
