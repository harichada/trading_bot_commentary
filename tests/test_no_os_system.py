"""
Test that command injection via os.system is not present in target files.

These tests enforce the security requirement that no file uses os.system()
for package installation, which is vulnerable to command injection.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

TARGET_FILES = [
    REPO_ROOT / "setup_wizard.py",
    REPO_ROOT / "backtesting_engine.py",
]

OS_SYSTEM_PATTERN = re.compile(r'\bos\.system\s*\(')


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_setup_wizard_has_no_os_system():
    source = _read_source(REPO_ROOT / "setup_wizard.py")
    matches = OS_SYSTEM_PATTERN.findall(source)
    assert matches == [], (
        f"setup_wizard.py contains {len(matches)} os.system() call(s) — "
        "replace with subprocess.run([sys.executable, '-m', 'pip', ...]) "
        "to prevent command injection."
    )


def test_backtesting_engine_has_no_os_system():
    source = _read_source(REPO_ROOT / "backtesting_engine.py")
    matches = OS_SYSTEM_PATTERN.findall(source)
    assert matches == [], (
        f"backtesting_engine.py contains {len(matches)} os.system() call(s) — "
        "replace with a plain ImportError to prevent command injection."
    )


def test_all_target_files_exist():
    """Guard: ensure the files under test actually exist and were found."""
    for path in TARGET_FILES:
        assert path.exists(), f"Expected source file not found: {path}"
