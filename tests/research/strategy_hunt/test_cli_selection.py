"""Batch-2 Task A: the `--shuffle-mode` flag and where the selection replay sits in `all`.

`run --shuffle-seed N --shuffle-mode selection` replays the real trade schedule on random peers
instead of simulating the families again, and `all` runs both controls -- path shuffle, then
selection -- between the real run and `score`.
"""
import json

import pytest

_ALL_STEPS = ("cmd_export", "cmd_news", "cmd_earnings", "cmd_context", "cmd_calibrate", "cmd_run",
              "cmd_score", "cmd_playbook")


def _record_all_steps(cli, monkeypatch) -> list[tuple]:
    order: list[tuple] = []

    def _step(name):
        def _fn(a):
            order.append((name, getattr(a, "shuffle_seed", None), getattr(a, "shuffle_mode", None)))
            return 0
        return _fn

    for name in _ALL_STEPS:
        monkeypatch.setattr(cli, name, _step(name))
    return order


def _summary(seed, peers_total, n_errors):
    return {"symbols": peers_total, "peers": peers_total - n_errors, "peers_total": peers_total,
            "trades": 10 * peers_total, "replayed": 10 * (peers_total - n_errors),
            "selection_dropped": 10 * n_errors, "dropped": {}, "selection_groups": {},
            "underreplayed_groups": [],
            "errors": [{"symbol": f"E{i}", "error": "ValueError: boom"} for i in range(n_errors)],
            "seed": seed}


def test_shuffle_mode_defaults_to_path_and_accepts_selection():
    from research.strategy_hunt.cli import build_parser
    from research.strategy_hunt.cli_args import SHUFFLE_MODES, SHUFFLE_MODE_PATH, SHUFFLE_MODE_SELECTION

    assert SHUFFLE_MODES == (SHUFFLE_MODE_PATH, SHUFFLE_MODE_SELECTION) == ("path", "selection")
    assert build_parser().parse_args(["run"]).shuffle_mode == SHUFFLE_MODE_PATH
    assert build_parser().parse_args(["run", "--shuffle-mode", "selection"]).shuffle_mode == "selection"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--shuffle-mode", "sideways"])


def test_all_refuses_a_user_set_shuffle_mode_but_keeps_the_default(capsys):
    """Like `--shuffle-seed`, `--shuffle-mode` on `all` would turn the *real* leg into a control."""
    from research.strategy_hunt.cli import build_parser
    from research.strategy_hunt.cli_args import SHUFFLE_MODE_PATH

    assert build_parser().parse_args(["all"]).shuffle_mode == SHUFFLE_MODE_PATH
    with pytest.raises(SystemExit):
        build_parser().parse_args(["all", "--shuffle-mode", "selection"])
    assert "not accepted by `all`" in capsys.readouterr().err


def test_cmd_run_selection_replays_instead_of_simulating_the_families(monkeypatch, tmp_path):
    from research.strategy_hunt import cli
    from research.strategy_hunt import selection_baseline as SB

    def _no_simulation(*a, **kw):
        raise AssertionError("the selection control must not re-run the family simulation")

    monkeypatch.setattr(cli, "run_symbols", _no_simulation)
    seen = {}

    def _fake_replay(root, seed, workers=1, **kw):
        seen.update({"root": root, "seed": seed, "workers": workers, **kw})
        return {**_summary(seed, 3, 0), "replayed": 3, "selection_dropped": 1,
                "dropped": {"no_peer_bar": 1}}

    monkeypatch.setattr(SB, "replay_selection", _fake_replay)

    rc = cli.main(["run", "--root", str(tmp_path), "--shuffle-seed", "7", "--shuffle-mode", "selection",
                   "--workers", "1"])

    assert rc == 0
    assert seen["seed"] == 7 and seen["workers"] == 1 and str(seen["root"]) == str(tmp_path)


def test_cmd_run_selection_records_the_dropped_count_in_the_manifest(monkeypatch, tmp_path):
    from research.strategy_hunt import cli
    from research.strategy_hunt import selection_baseline as SB
    from research.strategy_hunt.paths import reports_dir

    monkeypatch.setattr(SB, "replay_selection", lambda root, seed, workers=1, **kw: {
        **_summary(seed, 3, 0), "trades": 4, "replayed": 3, "selection_dropped": 1,
        "dropped": {"no_peer_bar": 1},
        "selection_groups": {"toy|long|mixed": {"real": 4, "replayed": 3, "dropped": 1,
                                               "replayed_frac": 0.75}},
        "underreplayed_groups": ["toy|long|mixed"]})

    assert cli.main(["run", "--root", str(tmp_path), "--shuffle-seed", "7",
                     "--shuffle-mode", "selection"]) == 0

    rec = json.loads(next(reports_dir(tmp_path).glob("*.json")).read_text())
    assert rec["run_selection"]["selection_dropped"] == 1
    assert rec["run_selection"]["variant"] == "selection"
    assert rec["run_selection"]["shuffle_seed"] == 7
    assert "run" not in rec                       # the real leg's record is untouched


def test_cmd_run_selection_without_a_seed_fails_cleanly(tmp_path, capsys):
    from research.strategy_hunt import cli

    assert cli.main(["run", "--root", str(tmp_path), "--shuffle-mode", "selection"]) == 1
    assert "--shuffle-seed" in capsys.readouterr().out


def test_cmd_run_selection_tolerates_a_few_failed_peers(monkeypatch, tmp_path):
    """Fix round 1, Important 1: the same tolerance as the real leg -- a handful of bad symbols must
    not stop `all` before score/playbook, and the failures are still on the manifest."""
    from research.strategy_hunt import cli
    from research.strategy_hunt import selection_baseline as SB
    from research.strategy_hunt.paths import reports_dir

    monkeypatch.setattr(SB, "replay_selection",
                        lambda root, seed, workers=1, **kw: _summary(seed, 20, 1))

    assert cli.main(["run", "--root", str(tmp_path), "--shuffle-seed", "7",
                     "--shuffle-mode", "selection"]) == 0
    rec = json.loads(next(reports_dir(tmp_path).glob("*.json")).read_text())
    assert [e["symbol"] for e in rec["errors_selection"]] == ["E0"]


def test_cmd_run_selection_returns_1_when_more_than_a_tenth_of_peers_failed(monkeypatch, tmp_path):
    from research.strategy_hunt import cli
    from research.strategy_hunt import selection_baseline as SB

    monkeypatch.setattr(SB, "replay_selection",
                        lambda root, seed, workers=1, **kw: _summary(seed, 20, 3))

    assert cli.main(["run", "--root", str(tmp_path), "--shuffle-seed", "7",
                     "--shuffle-mode", "selection"]) == 1


def test_cmd_all_runs_the_real_tape_then_both_controls_then_score(monkeypatch, tmp_path):
    from research.strategy_hunt import cli
    from research.strategy_hunt.cli_args import BASELINE_SHUFFLE_SEED, SHUFFLE_MODE_PATH, SHUFFLE_MODE_SELECTION

    order = _record_all_steps(cli, monkeypatch)
    assert cli.cmd_all(cli.build_parser().parse_args(["all", "--root", str(tmp_path)])) == 0
    assert order == [("cmd_export", None, SHUFFLE_MODE_PATH), ("cmd_news", None, SHUFFLE_MODE_PATH),
                     ("cmd_earnings", None, SHUFFLE_MODE_PATH), ("cmd_context", None, SHUFFLE_MODE_PATH),
                     ("cmd_calibrate", None, SHUFFLE_MODE_PATH), ("cmd_run", None, SHUFFLE_MODE_PATH),
                     ("cmd_run", BASELINE_SHUFFLE_SEED, SHUFFLE_MODE_PATH),
                     ("cmd_run", BASELINE_SHUFFLE_SEED, SHUFFLE_MODE_SELECTION),
                     ("cmd_score", None, SHUFFLE_MODE_PATH), ("cmd_playbook", None, SHUFFLE_MODE_PATH)]


def test_cmd_all_no_baseline_skips_both_controls(monkeypatch, tmp_path):
    from research.strategy_hunt import cli

    order = _record_all_steps(cli, monkeypatch)
    assert cli.cmd_all(cli.build_parser().parse_args(["all", "--root", str(tmp_path),
                                                      "--no-baseline"])) == 0
    assert [n for n, _seed, _mode in order] == list(_ALL_STEPS)


def test_cmd_all_stops_when_the_selection_control_fails(monkeypatch, tmp_path):
    from research.strategy_hunt import cli
    from research.strategy_hunt.cli_args import SHUFFLE_MODE_SELECTION

    calls = []
    for name in ("cmd_export", "cmd_news", "cmd_earnings", "cmd_context", "cmd_calibrate"):
        monkeypatch.setattr(cli, name, lambda a: 0)
    monkeypatch.setattr(cli, "cmd_run",
                        lambda a: 1 if a.shuffle_mode == SHUFFLE_MODE_SELECTION else 0)
    monkeypatch.setattr(cli, "cmd_score", lambda a: calls.append("cmd_score") or 0)

    assert cli.cmd_all(cli.build_parser().parse_args(["all", "--root", str(tmp_path)])) == 1
    assert calls == []


def test_the_selection_namespace_forces_the_run_and_leaves_the_caller_alone():
    from research.strategy_hunt import cli
    from research.strategy_hunt.cli_args import BASELINE_SHUFFLE_SEED, SHUFFLE_MODE_PATH, SHUFFLE_MODE_SELECTION

    a = cli.build_parser().parse_args(["all"])
    sel = cli._selection_namespace(a)
    assert (sel.shuffle_seed, sel.force, sel.shuffle_mode) == (BASELINE_SHUFFLE_SEED, True,
                                                               SHUFFLE_MODE_SELECTION)
    assert (a.shuffle_seed, a.force, a.shuffle_mode) == (None, False, SHUFFLE_MODE_PATH)
