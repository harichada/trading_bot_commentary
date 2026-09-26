"""v-playbook-desk-b1-6 (final review, Minor 1 and 2) — the `all` shuffle guard and where
`promote` writes its markdown. Kept out of `test_cli.py`, which is already at its line budget.
"""
import argparse

import pytest

from research.strategy_hunt import cli


# ──────────────────────
# M1 — `all` must never be handed a shuffle seed
# ──────────────────────

def test_all_rejects_a_shuffle_seed_with_a_parser_error(capsys):
    """`all` runs the real tape *and* the fixed-seed baseline. A user-supplied seed would turn
    the real run into a second shuffled one, so `trades/` would never be written and every
    later step would score the control against itself."""
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["all", "--shuffle-seed", "7"])
    assert exc.value.code == 2
    assert "shuffle-seed" in capsys.readouterr().err


def test_all_still_has_a_none_shuffle_seed_for_the_baseline_namespace():
    a = cli.build_parser().parse_args(["all"])
    assert a.shuffle_seed is None
    assert cli._baseline_namespace(a).shuffle_seed == cli.BASELINE_SHUFFLE_SEED


def test_run_still_accepts_a_shuffle_seed():
    assert cli.build_parser().parse_args(["run", "--shuffle-seed", "7"]).shuffle_seed == 7
    assert cli.build_parser().parse_args(["score", "--shuffle-seed", "7"]).shuffle_seed == 7


# ──────────────────────
# M2 — promotion markdown belongs to the root it was computed from
# ──────────────────────

def test_promote_writes_under_reports_dir_of_the_given_root(monkeypatch, tmp_path):
    from research.strategy_hunt import promote as promote_mod
    from research.strategy_hunt.paths import reports_dir

    monkeypatch.chdir(tmp_path)
    root = tmp_path / "hunt_root"
    monkeypatch.setattr(cli, "family_names", lambda: ["orb"])
    monkeypatch.setattr("research.strategy_hunt.fit.writer.load_snapshot", lambda r: {"rows": {}})
    monkeypatch.setattr(promote_mod, "promotion_table", lambda *a, **kw: [])
    monkeypatch.setattr(promote_mod, "render_promotion", lambda rows: "# promotion\n")

    assert cli.cmd_promote(argparse.Namespace(root=root)) == 0
    out = reports_dir(root) / "strategy_hunt_promotion.md"
    assert out.exists() and out.read_text() == "# promotion\n"
    assert not (tmp_path / "docs" / "research" / "strategy_hunt_promotion.md").exists()


def test_promote_uses_the_committed_reports_dir_for_the_default_root(monkeypatch, tmp_path):
    from research.strategy_hunt import promote as promote_mod
    from research.strategy_hunt.run import DEFAULT_ROOT

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "family_names", lambda: [])
    monkeypatch.setattr("research.strategy_hunt.fit.writer.load_snapshot", lambda r: {"rows": {}})
    monkeypatch.setattr(promote_mod, "promotion_table", lambda *a, **kw: [])
    monkeypatch.setattr(promote_mod, "render_promotion", lambda rows: "# promotion\n")

    assert cli.cmd_promote(argparse.Namespace(root=DEFAULT_ROOT)) == 0
    assert (tmp_path / "research" / "reports" / "strategy_hunt" / "strategy_hunt_promotion.md").exists()


# ──────────────────────
# strategy-search-batch1 task 2 — --holdout-months is a shared flag, default 0
# ──────────────────────

def test_holdout_months_flag_default_and_settable():
    a = cli.build_parser().parse_args(["run"])
    assert a.holdout_months == 0
    a2 = cli.build_parser().parse_args(["run", "--holdout-months", "6"])
    assert a2.holdout_months == 6
    # shared via add_common_args -- also present on `all` and `score`, not just `run`
    assert cli.build_parser().parse_args(["all"]).holdout_months == 0
    assert cli.build_parser().parse_args(["score", "--holdout-months", "3"]).holdout_months == 3


# ──────────────────────
# Batch-2 Task F item 5 — the splits-audit subcommand is reachable from the CLI
# ──────────────────────

def test_splits_audit_subcommand_is_registered_and_dispatches(monkeypatch, tmp_path, capsys):
    """Task E wrote the audit body but left it unreachable, so the one-off diagnostic could only be
    run through a hand-written `python -c`. It takes --root and --symbols and nothing else."""
    from research.strategy_hunt import splits_audit as SA

    seen = {}
    monkeypatch.setattr(SA, "audit_splits",
                        lambda root, symbols=None: seen.update(root=root, symbols=symbols) or
                        [{"symbol": "LCID", "reasons": ["minmax_ratio=1e+08"]}])

    assert cli.main(["splits-audit", "--root", str(tmp_path), "--symbols", "LCID,AAA"]) == 0
    assert str(seen["root"]) == str(tmp_path) and seen["symbols"] == "LCID,AAA"
    out = capsys.readouterr().out
    assert "LCID" in out and "1 symbol(s) flagged" in out


def test_splits_audit_defaults_root_and_symbols():
    from research.strategy_hunt.run import DEFAULT_ROOT

    a = cli.build_parser().parse_args(["splits-audit"])
    assert a.root == DEFAULT_ROOT and a.symbols is None
