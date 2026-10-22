"""v-learning-loop-2026-10-22: the daily realized-loss halt sits in the NEW-entry
router right after the console kill switch, is flag-gated, and the router is the
same one BLOCK_NEW_ENTRIES relies on never carrying exits."""
import inspect


def test_halt_hook_is_in_router_after_block_new_entries():
    from core import engine
    src = inspect.getsource(engine.TradingEngineWithCommentary._process_signal_with_commentary) \
        if hasattr(engine, "TradingEngineWithCommentary") else open(engine.__file__).read()
    i_block = src.index("entries_blocked_by_console")
    i_halt = src.index("ENABLE_DAILY_REALIZED_LOSS_HALT")
    i_meta = src.index("_shadow_meta_evaluate(signal)")
    assert i_block < i_halt < i_meta
    assert 'getattr(Config(), "ENABLE_DAILY_REALIZED_LOSS_HALT", False) is True' in src
    assert "HANDS_OFF_DENYLIST" in src[i_halt:i_meta]
