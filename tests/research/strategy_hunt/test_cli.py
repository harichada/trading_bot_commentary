"""CLI-level tests: universe-context wiring and cmd_score -> score_regime_root -> write_fit."""
import argparse

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

_ALL_STEPS = ("cmd_export", "cmd_news", "cmd_earnings", "cmd_context", "cmd_calibrate", "cmd_run",
              "cmd_score", "cmd_playbook")


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
    baseline (seed 7) into the same root's `_shuffle` dirs -- before `score` reads both.
    Batch-2 Task A adds a third `run`: the selection control (same seed, `_selection` dirs). Which
    leg is which is asserted by mode in test_cli_selection.py."""
    from research.strategy_hunt import cli

    order = _record_all_steps(cli, monkeypatch)
    assert cli.cmd_all(cli.build_parser().parse_args(["all", "--root", str(tmp_path)])) == 0
    assert order == [("cmd_export", None), ("cmd_news", None), ("cmd_earnings", None), ("cmd_context", None),
                     ("cmd_calibrate", None), ("cmd_run", None), ("cmd_run", cli.BASELINE_SHUFFLE_SEED),
                     ("cmd_run", cli.BASELINE_SHUFFLE_SEED),
                     ("cmd_score", None), ("cmd_playbook", None)]


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
    for name in ("cmd_export", "cmd_news", "cmd_earnings", "cmd_context", "cmd_calibrate", "cmd_score",
                 "cmd_playbook"):
        monkeypatch.setattr(cli, name, lambda a: 0)
    monkeypatch.setattr(cli, "cmd_run", lambda a: seen.append((a.shuffle_seed, a.force)) or 0)

    a = cli.build_parser().parse_args(["all", "--root", str(tmp_path)])
    assert cli.cmd_all(a) == 0
    assert seen == [(None, False), (cli.BASELINE_SHUFFLE_SEED, True), (cli.BASELINE_SHUFFLE_SEED, True)]
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


# ──────────────────────
# v-playbook-desk-b1-5 task 10 fix r1 — per-variant manifest errors, tolerant cmd_report
# ──────────────────────

def test_cmd_run_manifest_keeps_both_error_lists(monkeypatch, tmp_path):
    """`write_manifest` merges with `dict.update`, so a single top-level `errors` key meant the
    baseline run wiped the real run's failures off the record."""
    import json as _json
    from research.strategy_hunt import cli
    from research.strategy_hunt import run as RUN

    reports = tmp_path / "reports"
    monkeypatch.setattr(RUN, "REPORTS_DIR", reports)

    def _fake(symbols, root, families, workers=1, **kw):
        tag = "shuffle" if kw.get("variant") else "real"
        return [{"symbol": f"BAD_{tag}", "status": "error", "n_trades": 0, "error": f"{tag} boom"}]

    monkeypatch.setattr(cli, "run_symbols", _fake)
    assert cli.main(["run", "--root", str(tmp_path), "--symbols", "AAA", "--families", "toy"]) == 1
    assert cli.main(["run", "--root", str(tmp_path), "--symbols", "AAA", "--families", "toy",
                     "--shuffle-seed", "7"]) == 1

    rec = _json.loads(next(reports.glob("*.json")).read_text())
    assert [e["symbol"] for e in rec["errors"]] == ["BAD_real"]
    assert [e["symbol"] for e in rec["errors_shuffle"]] == ["BAD_shuffle"]
    assert rec["run"]["shuffle_seed"] is None and rec["run_shuffle"]["shuffle_seed"] == 7


def test_cmd_report_tolerates_a_malformed_pooled_rows_section(monkeypatch, tmp_path, capsys):
    """`core/strategy_fit.py` never raises on a malformed section; neither may the report path."""
    import json as _json
    from research.strategy_hunt import cli
    from research.strategy_hunt import report as R
    from research.strategy_hunt.fit.writer import fit_dir

    monkeypatch.setattr(R, "REPORTS_DIR", tmp_path / "reports")
    d = fit_dir(tmp_path)
    d.mkdir(parents=True)
    row = {"symbol": "NVDA", "strategy": "orb", "side": "long", "verdict": "PASS", "exp_r": 0.2,
           "exp_r_lb": 0.02, "t_stat": 2.5, "pf": 1.4, "wr": 0.5, "max_dd_r": 5.0, "stability": 0.7,
           "decay": 0.8, "score": 0.17, "n": 200, "sessions": 40, "computed_at": "2025-09-25T01:00:00",
           "window_end": "2025-09-24", "failed_floors": "[]", "params_json": "{}",
           "regime_exp_json": "{}", "source": "walkforward"}
    (d / "strategy_fit.json").write_text(_json.dumps(
        {"computed_at": "2025-09-25T01:00:00", "floors": {}, "rows": {"NVDA": {"orb:long": row}},
         "regime_rows": {}, "pooled_rows": ["nope"]}))

    assert cli.main(["report", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "report:" in out
    md = (tmp_path / "reports" / "2025-09-25T0100-strategy-hunt.md").read_text()
    assert "_no pooled rows_" in md.split("## Pooled rows vs baseline")[1].split("## Controls")[0]


# ──────────────────────
# v-playbook-desk-b2-2026-09-25 (Task 3) — the `playbook` subcommand
# ──────────────────────

def test_playbook_subparser_takes_root_desk_dir_and_postgres():
    from research.strategy_hunt.cli import build_parser

    ns = build_parser().parse_args(["playbook", "--root", "/tmp/r", "--desk-dir", "/tmp/desk", "--postgres"])
    assert str(ns.root) == "/tmp/r" and str(ns.desk_dir) == "/tmp/desk" and ns.postgres is True

    default = build_parser().parse_args(["playbook"])
    assert default.desk_dir.as_posix() == "data/desk" and default.postgres is False


def test_cmd_playbook_calls_update_playbook_with_the_parsed_root_and_desk_dir(monkeypatch, tmp_path, capsys):
    """No Postgres: without --postgres the updater must be handed engine=None, and the desk dir
    from the command line (never the live Config default)."""
    from core.desk.playbook import Playbook, PlaybookEntry
    from research.strategy_hunt import cli
    from research.strategy_hunt import playbook_update as PU

    seen = {}
    entry = PlaybookEntry(family="orb", side="long", regime="risk_on", status="suspended", size_mult=0.0,
                          exp_r_recent=-0.4, t_stat_recent=-2.0, n_recent=20, updated_at="now",
                          reason="suspended_negative_expectancy")

    def _fake(root, snapshot, now, desk_dir=None, families=None, engine=None, **kw):
        seen.update(root=root, snapshot=snapshot, now=now, desk_dir=desk_dir, families=families, engine=engine)
        return Playbook(entries={}, updated_at=now).with_entry(entry)

    monkeypatch.setattr(PU, "update_playbook", _fake)

    rc = cli.main(["playbook", "--root", str(tmp_path), "--desk-dir", str(tmp_path / "desk")])
    assert rc == 0
    assert str(seen["root"]) == str(tmp_path) and str(seen["desk_dir"]) == str(tmp_path / "desk")
    assert seen["engine"] is None and isinstance(seen["snapshot"], dict)
    out = capsys.readouterr().out
    assert "playbook" in out and "suspended" in out


def test_cmd_playbook_from_cmd_all_namespace_uses_the_default_desk_dir(monkeypatch, tmp_path):
    """`all` has no --desk-dir flag, so cmd_playbook must fall back to the default rather than
    raising AttributeError halfway through the nightly pipeline."""
    from core.desk.playbook import DEFAULT_DESK_DIR, Playbook
    from research.strategy_hunt import cli
    from research.strategy_hunt import playbook_update as PU

    seen = {}

    def _fake(root, snapshot, now, desk_dir=None, families=None, engine=None, **kw):
        seen["desk_dir"] = desk_dir
        return Playbook(entries={}, updated_at=now)

    monkeypatch.setattr(PU, "update_playbook", _fake)

    a = cli.build_parser().parse_args(["all", "--root", str(tmp_path)])
    assert cli.cmd_playbook(a) == 0
    assert str(seen["desk_dir"]) == str(DEFAULT_DESK_DIR)


def test_cmd_playbook_returns_1_and_prints_a_clean_error_on_failure(monkeypatch, tmp_path, capsys):
    from research.strategy_hunt import cli
    from research.strategy_hunt import playbook_update as PU

    def _boom(*a, **kw):
        raise ValueError("desk dir is a file")

    monkeypatch.setattr(PU, "update_playbook", _boom)

    assert cli.main(["playbook", "--root", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "playbook ERROR" in out and "desk dir is a file" in out and "Traceback" not in out


# ──────────────────────
# v-playbook-desk-b2 fix r1 — DSN handling and `all --postgres` passthrough
# ──────────────────────

def test_cmd_playbook_with_postgres_and_no_dsn_is_one_clean_line(monkeypatch, tmp_path, capsys):
    """Fix r1 finding 5: the engine is built inside the try, so a missing POSTGRES_DSN is rc=1
    and a single line — not a KeyError traceback in the nightly log."""
    from research.strategy_hunt import cli

    monkeypatch.delenv("POSTGRES_DSN", raising=False)

    rc = cli.main(["playbook", "--root", str(tmp_path), "--desk-dir", str(tmp_path / "desk"),
                   "--postgres"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "playbook ERROR" in out and "POSTGRES_DSN" in out and "Traceback" not in out
    assert not (tmp_path / "desk" / "playbook.json").exists()


def test_cmd_playbook_builds_and_disposes_the_engine_when_a_dsn_is_set(monkeypatch, tmp_path):
    from core.desk.playbook import Playbook
    from research.strategy_hunt import playbook_cli as PC
    from research.strategy_hunt import playbook_update as PU

    monkeypatch.setenv("POSTGRES_DSN", "sqlite://")
    seen = {}

    def _fake(root, snapshot, now, **kw):
        seen["engine"] = kw["engine"]
        return Playbook(entries={}, updated_at=now)

    monkeypatch.setattr(PU, "update_playbook", _fake)

    args = argparse.Namespace(root=tmp_path, desk_dir=tmp_path / "desk", postgres=True, families=None)
    assert PC.run_playbook_command(args, ["toy"]) == 0
    assert seen["engine"] is not None


def test_cmd_all_passes_postgres_through_to_the_playbook_step(monkeypatch, tmp_path):
    """Fix r1 finding 6: `all --postgres` must reach the playbook step, or the nightly run
    silently updates the playbook from the shadow ledgers only."""
    from research.strategy_hunt import cli

    for name in ("cmd_export", "cmd_news", "cmd_earnings", "cmd_context", "cmd_calibrate",
                 "cmd_run", "cmd_score"):
        monkeypatch.setattr(cli, name, lambda a: 0)
    seen = []
    monkeypatch.setattr(cli, "cmd_playbook", lambda a: seen.append(a.postgres) or 0)

    assert cli.cmd_all(cli.build_parser().parse_args(["all", "--root", str(tmp_path),
                                                     "--postgres"])) == 0
    assert seen == [True]
