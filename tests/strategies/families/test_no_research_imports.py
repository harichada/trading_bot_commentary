"""Verify that strategies/families/ does not import research, core, or api."""


def test_families_package_does_not_import_research_or_core():
    import pathlib
    import re

    root = pathlib.Path("strategies/families")
    for p in root.glob("*.py"):
        src = p.read_text()
        assert not re.search(r"^\s*(from|import)\s+(research|core|api)\b", src, re.M), p
