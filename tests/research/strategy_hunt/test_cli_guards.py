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
