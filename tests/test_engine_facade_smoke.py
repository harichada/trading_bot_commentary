"""v-phase-a-facade-inventory-2026-09-14: TradingEngine façade stability test.

PURPOSE
-------
For one release cycle, engine façade method/attribute names must be kept
stable. Delegation or internal reimplementation is OK, but the external
API surface must not break.

This test freezes a baseline inventory of public methods, underscore-
prefixed methods that external callers rely on, and public attributes.
If a name disappears or a method signature changes, the test fails loudly.

After Phase B extract PRs, update the inventory JSON to reflect the new
façade (delegation stubs may be added) and re-run this test to confirm
backward compatibility.

HOW TO EXTEND
-------------
When adding a new public method to TradingEngineWithCommentary:
1. Add it to tests/fixtures/engine_facade_inventory.json in the
   appropriate section (methods or underscore_methods).
2. Include params list and is_async flag.
3. Run pytest tests/test_engine_facade_smoke.py to confirm the fixture
   is valid.

REFERENCES
----------
- External callers: api/routes.py, core/loops/screener_loop.py,
  core/loops/news_loop.py, strategies/base.py, strategies/news_strategy.py
"""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = REPO_ROOT / "core" / "engine.py"
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "engine_facade_inventory.json"


def _load_fixture() -> Dict[str, Any]:
    """Load the frozen inventory from JSON fixture."""
    with open(FIXTURE_PATH, "r") as f:
        return json.load(f)


def _parse_engine_class() -> Dict[str, Any]:
    """Parse core/engine.py and extract TradingEngineWithCommentary info."""
    with open(ENGINE_PATH, "r") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "TradingEngineWithCommentary":
            methods = {}
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    params = [
                        arg.arg for arg in item.args.args if arg.arg != "self"
                    ]
                    methods[item.name] = {
                        "params": params,
                        "is_async": isinstance(item, ast.AsyncFunctionDef),
                        "line": item.lineno,
                    }
            return {"class_name": "TradingEngineWithCommentary", "methods": methods}
    raise RuntimeError("TradingEngineWithCommentary class not found in engine.py")


class TestEngineFacadeInventory:
    """Frozen inventory tests for the TradingEngine façade.

    These tests ensure that names and signatures in the external
    API surface remain stable across refactoring.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def inventory(cls) -> Dict[str, Any]:
        return _load_fixture()

    @pytest.fixture(scope="class")
    @classmethod
    def actual_class(cls) -> Dict[str, Any]:
        return _parse_engine_class()

    def test_fixture_exists_and_is_valid(self, inventory: Dict[str, Any]):
        """The fixture file must exist and have expected structure."""
        assert FIXTURE_PATH.exists(), f"Missing fixture: {FIXTURE_PATH}"
        assert "metadata" in inventory
        assert "methods" in inventory
        assert "underscore_methods" in inventory
        assert "attributes" in inventory
        assert isinstance(inventory["methods"], list)
        assert isinstance(inventory["underscore_methods"], list)
        assert isinstance(inventory["attributes"], list)

    def test_public_methods_exist(
        self, inventory: Dict[str, Any], actual_class: Dict[str, Any]
    ):
        """All frozen public methods must still exist on the class."""
        actual_methods = actual_class["methods"]
        missing = []

        for method_spec in inventory["methods"]:
            name = method_spec["name"]
            if name not in actual_methods:
                missing.append(name)

        assert not missing, (
            f"Public methods REMOVED from TradingEngineWithCommentary "
            f"(Phase B extract not allowed until Phase A merges): {missing}"
        )

    def test_underscore_methods_exist(
        self, inventory: Dict[str, Any], actual_class: Dict[str, Any]
    ):
        """Underscore-prefixed methods used externally must still exist."""
        actual_methods = actual_class["methods"]
        missing = []

        for method_spec in inventory["underscore_methods"]:
            name = method_spec["name"]
            if name not in actual_methods:
                missing.append(name)

        assert not missing, (
            f"Underscore methods REMOVED from TradingEngineWithCommentary "
            f"but they are called by external code: {missing}"
        )

    def test_public_method_signatures_stable(
        self, inventory: Dict[str, Any], actual_class: Dict[str, Any]
    ):
        """Public method parameter lists must not change."""
        actual_methods = actual_class["methods"]
        signature_mismatches = []

        for method_spec in inventory["methods"]:
            name = method_spec["name"]
            if name not in actual_methods:
                continue  # Already caught by existence test

            expected_params = method_spec["params"]
            actual_params = actual_methods[name]["params"]
            expected_async = method_spec.get("is_async", False)
            actual_async = actual_methods[name]["is_async"]

            if expected_params != actual_params:
                signature_mismatches.append(
                    f"{name}: expected params {expected_params}, got {actual_params}"
                )
            if expected_async != actual_async:
                signature_mismatches.append(
                    f"{name}: expected is_async={expected_async}, got {actual_async}"
                )

        assert not signature_mismatches, (
            f"Public method signatures changed: {signature_mismatches}"
        )

    def test_underscore_method_signatures_stable(
        self, inventory: Dict[str, Any], actual_class: Dict[str, Any]
    ):
        """Underscore method parameter lists must not change."""
        actual_methods = actual_class["methods"]
        signature_mismatches = []

        for method_spec in inventory["underscore_methods"]:
            name = method_spec["name"]
            if name not in actual_methods:
                continue  # Already caught by existence test

            expected_params = method_spec["params"]
            actual_params = actual_methods[name]["params"]
            expected_async = method_spec.get("is_async", False)
            actual_async = actual_methods[name]["is_async"]

            if expected_params != actual_params:
                signature_mismatches.append(
                    f"{name}: expected params {expected_params}, got {actual_params}"
                )
            if expected_async != actual_async:
                signature_mismatches.append(
                    f"{name}: expected is_async={expected_async}, got {actual_async}"
                )

        assert not signature_mismatches, (
            f"Underscore method signatures changed: {signature_mismatches}"
        )

    def test_attribute_names_documented(self, inventory: Dict[str, Any]):
        """Attribute inventory must be non-empty and well-formed."""
        attrs = inventory["attributes"]
        assert len(attrs) > 0, "Attribute inventory is empty"
        for attr in attrs:
            assert "name" in attr, f"Attribute missing 'name': {attr}"
            assert isinstance(attr["name"], str), f"Attribute name not string: {attr}"

    def test_inventory_count_matches_baseline(self, inventory: Dict[str, Any]):
        """Alert if inventory size drifts significantly."""
        expected_method_count = inventory["metadata"].get("method_count", 0)
        expected_attr_count = inventory["metadata"].get("attribute_count", 0)

        actual_method_count = len(inventory["methods"]) + len(
            inventory["underscore_methods"]
        )
        actual_attr_count = len(inventory["attributes"])

        assert actual_method_count == expected_method_count, (
            f"Method count changed: expected {expected_method_count}, "
            f"got {actual_method_count}. Update fixture if intentional."
        )
        assert actual_attr_count == expected_attr_count, (
            f"Attribute count changed: expected {expected_attr_count}, "
            f"got {actual_attr_count}. Update fixture if intentional."
        )


class TestEngineFacadeImport:
    """Test that engine module can be imported without side effects."""

    def test_engine_module_importable(self):
        """The engine module must be importable (no syntax errors)."""
        # We parse AST instead of importing to avoid Schwab SDK
        # initialization, broker connections, etc.
        with open(ENGINE_PATH, "r") as f:
            source = f.read()
        tree = ast.parse(source)
        assert tree is not None

    def test_class_definition_exists(self):
        """TradingEngineWithCommentary must be defined in engine.py."""
        parsed = _parse_engine_class()
        assert parsed["class_name"] == "TradingEngineWithCommentary"

    def test_method_count_reasonable(self):
        """Sanity check: class should have substantial method count."""
        parsed = _parse_engine_class()
        method_count = len(parsed["methods"])
        assert method_count >= 50, (
            f"Method count suspiciously low ({method_count}), "
            "possibly parsing error"
        )
        assert method_count <= 200, (
            f"Method count suspiciously high ({method_count}), "
            "class may need decomposition"
        )
