import inspect


def test_signal_router_checks_block_switch_before_any_other_gate():
    import core.engine as eng
    src = inspect.getsource(eng.TradingEngineWithCommentary._process_signal_with_commentary)
    received = src.index('"received"')
    block = src.index("BLOCK_NEW_ENTRIES")
    assert received < block
    assert "entries_blocked_by_console" in src
    # the block must come before the shadow meta-model call, i.e. before any other work
    assert block < src.index("_shadow_meta_evaluate")
