"""v-trade-record-ml-columns: regression tests.

2026-09-02: both live trades (NVAX, HYMC) landed in bot_trades with
meta_proba=NULL and confidence=NULL even though the meta-model scored
both signals (0.6235 / 0.6117) and the values sat in reasoning_json.
The bot-driven close path passed `confidence=getattr(position,
"confidence", None)` (Position has no such attribute) and never passed
meta_proba / kelly_fraction at all — unlike the external-close
reconcile path, which plumbs all three from position.reasoning.

NULL columns silently break the shadow-soak program: post-hoc
"would gating at thr_065 have helped?" analysis needs those columns.

Static-source assertions in the style of tests/test_recent_fixes.py.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ENGINE_SRC = (REPO / "core" / "engine.py").read_text()
MANAGER_SRC = (REPO / "risk" / "manager.py").read_text()


def _bot_close_log_trade_block() -> str:
    """Extract the bot-driven close's log_trade(...) call — the one
    using `exit_reason=reason` (the external path uses a literal)."""
    match = re.search(
        r"self\.db_logger\.log_trade\((?:[^()]|\([^()]*\))*exit_reason=reason,"
        r"(?:[^()]|\([^()]*\))*\)",
        ENGINE_SRC,
    )
    assert match, "bot-driven close log_trade call not found in core/engine.py"
    return match.group(0)


class TestBotCloseTradeRecord:
    def test_meta_proba_plumbed_from_reasoning(self):
        block = _bot_close_log_trade_block()
        assert 'meta_proba=(position.reasoning or {}).get("meta_proba")' in block, (
            "bot-driven close log_trade must pass meta_proba (it is "
            "already present in position.reasoning at close time)"
        )

    def test_kelly_fraction_plumbed_from_reasoning(self):
        block = _bot_close_log_trade_block()
        assert (
            'kelly_fraction=(position.reasoning or {}).get("kelly_fraction")'
            in block
        )

    def test_confidence_read_from_reasoning_not_phantom_attr(self):
        block = _bot_close_log_trade_block()
        assert 'getattr(position, "confidence"' not in block, (
            "Position has no `confidence` attribute — reading it via "
            "getattr always yields None; confidence must come from "
            "position.reasoning"
        )
        assert 'confidence=(position.reasoning or {}).get("confidence")' in block


class TestReasoningSurvivesRestart:
    """Code-review finding: live positions_data in trading_state.json
    omitted `reasoning` (sim positions save it), and the restart
    ownership-restore path rebuilt Position without it — so any trade
    that outlived a restart still wrote NULL ML columns."""

    def test_live_position_save_includes_reasoning(self):
        match = re.search(
            r"positions_data\[symbol\] = \{(.*?)\n            \}",
            ENGINE_SRC, re.DOTALL,
        )
        assert match, "positions_data save block not found"
        assert "'reasoning'" in match.group(1), (
            "live positions_data must persist reasoning across restarts "
            "(sim_data already does)"
        )

    def test_ownership_restore_rehydrates_reasoning(self):
        anchor = ENGINE_SRC.find("position_ownership_restored")
        assert anchor != -1, "ownership-restore audit site not found"
        block = ENGINE_SRC[max(0, anchor - 2500):anchor]
        assert re.search(r"reasoning\s*=\s*saved\.get\([\"']reasoning", block), (
            "ownership-restore path must rehydrate position.reasoning "
            "from the saved record"
        )


class TestSizingStashesMlFields:
    """The sizing path computes confidence + kelly; it must persist
    them into signal.reasoning so the close path can recover them."""

    def test_confidence_written_to_signal_reasoning(self):
        assert re.search(
            r"signal\.reasoning\[[\"']confidence[\"']\]\s*=", MANAGER_SRC
        ), "risk/manager.py must stash confidence into signal.reasoning"

    def test_kelly_written_to_signal_reasoning(self):
        assert re.search(
            r"signal\.reasoning\[[\"']kelly_fraction[\"']\]\s*=", MANAGER_SRC
        ), "risk/manager.py must stash kelly_fraction into signal.reasoning"
