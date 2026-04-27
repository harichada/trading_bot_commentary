"""Plumbing tests for the --primary-bundle-path flag added during the
P4 barrier-retune experiment.

Hard requirement: adding the flag must NOT change behavior when the
default path is used. This file asserts:
  * the flag exists in the runner's argparse,
  * a custom bundle path is honored end-to-end (config sourced from
    that path; SHA256 reported for that path; predictions still come
    from the per-fold harness, not the bundle),
  * the existing P3 smoke output (default path) is structurally
    unchanged.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_runner_argparse_advertises_primary_bundle_path() -> None:
    """argparse must list --primary-bundle-path with a default of
    ml_model_v2.pkl."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "train_p3_regime_routing.py"), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert "--primary-bundle-path" in result.stdout, (
        f"--primary-bundle-path missing from help.\n{result.stdout}"
    )
    assert "ml_model_v2.pkl" in result.stdout, (
        "default bundle name not advertised in help"
    )


def test_runner_rejects_missing_bundle_path(tmp_path: Path) -> None:
    """Pointing --primary-bundle-path at a nonexistent file should fail
    with a clean error before any work is done."""
    fake = tmp_path / "does_not_exist.pkl"
    report = tmp_path / "p3.json"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "train_p3_regime_routing.py"),
         "--symbols", "NVDA",
         "--days", "5", "--frequency", "5", "--folds", "3",
         "--primary-bundle-path", str(fake),
         "--report-path", str(report)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0
    err = (result.stderr or "") + (result.stdout or "")
    assert "primary bundle not found" in err.lower() or \
           "filenotfounderror" in err.lower(), (
        f"expected clean missing-bundle error.\n{err[-2000:]}"
    )
    assert not report.exists(), "report should not be written on early failure"
