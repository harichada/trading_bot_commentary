"""Batch-2 Task F item 2: a lane whose winning params use a non-market entry style cannot be
promoted to live entries.

Batch-2 Task B gave the research simulator a second execution style (`limit_open`: a resting limit
a fixed offset inside the fill bar's open, filled only if the bar trades through it, at maker cost).
The live adapter places **market** entries only, so a lane whose walk-forward winner is a limit
entry is not the lane the bot would run -- its expectancy was earned by fills the bot cannot get.
That is a promotion blocker (`entry_style_unsupported_live`), not a footnote.

Kept out of `test_promote.py`, which is already at its line budget.
"""
import json

ENFORCING = {"gate_enabled": True, "gate_shadow": False}


def _row(symbol, strategy, side="long", verdict="PASS", params=None, window_end="2026-09-25",
         source="walkforward"):
    row = {"symbol": symbol, "strategy": strategy, "side": side, "verdict": verdict,
           "source": source, "window_end": window_end}
    if params is not None:
        row["params_json"] = params if isinstance(params, str) else json.dumps(params)
    return row


def _snapshot(rows):
    nested: dict = {}
    for r in rows:
        nested.setdefault(r["symbol"], {})[f"{r['strategy']}:{r['side']}"] = r
    return {"computed_at": "2026-09-26T00:00:00", "floors": {}, "rows": nested}


def _table(rows, **kw):
    from research.strategy_hunt.promote import promotion_table

    opts = {"lanes_enabled": ("orb",), "live_enabled": (), "swing_allowed": False, **ENFORCING, **kw}
    return {r["family"]: r for r in promotion_table(_snapshot(rows), {"orb": {"verdict": "GO"}}, **opts)}


def _orb_rows(params, n=12, window_end="2026-09-25"):
    return [_row(f"SYM{i}", "orb", params=params, window_end=window_end) for i in range(n)]


def test_a_limit_entry_lane_is_blocked_from_live_promotion():
    orb = _table(_orb_rows({"range_minutes": 30, "entry_style": "limit_open"}))["orb"]
    assert orb["blocker"] == "entry_style_unsupported_live"
    assert orb["recommendation"] == "hold"
    assert orb["entry_style"] == "limit_open"


def test_a_market_entry_lane_is_not_blocked_by_the_new_gate():
    orb = _table(_orb_rows({"range_minutes": 30, "entry_style": "market"}))["orb"]
    assert orb["blocker"] == "" and orb["recommendation"] == "promote_live"
    assert orb["entry_style"] == "market"


def test_params_without_an_entry_style_key_are_market_by_definition():
    """Every snapshot written before Batch-2 Task B is a market-fill search, so the absent key
    must not read as "unknown" and block every existing lane."""
    orb = _table(_orb_rows({"range_minutes": 30}))["orb"]
    assert orb["blocker"] == "" and orb["entry_style"] == "market"
    no_params = _table([_row(f"SYM{i}", "orb") for i in range(12)])["orb"]
    assert no_params["blocker"] == "" and no_params["entry_style"] == "market"


def test_a_malformed_params_json_degrades_to_market_rather_than_raising():
    orb = _table(_orb_rows("{not json"))["orb"]
    assert orb["blocker"] == "" and orb["entry_style"] == "market"


def test_only_the_latest_window_decides_the_style():
    """A limit-entry winner from an earlier window is history: what matters is the params the lane
    would be promoted *with*. A newer window that chose `market` clears the blocker."""
    rows = (_orb_rows({"entry_style": "limit_open"}, n=6, window_end="2026-06-30")
            + _orb_rows({"entry_style": "market"}, n=6, window_end="2026-09-25"))
    # distinct symbols per generation, so both survive the snapshot's (symbol, family:side) keying
    rows = [{**r, "symbol": f"{r['symbol']}_{r['window_end'][:7]}"} for r in rows]
    assert _table(rows)["orb"]["blocker"] == ""

    stale_market = [{**r, "window_end": "2026-06-30"} for r in _orb_rows({"entry_style": "market"}, n=6)]
    fresh_limit = [{**r, "symbol": f"{r['symbol']}x"}
                   for r in _orb_rows({"entry_style": "limit_open"}, n=6, window_end="2026-09-25")]
    assert _table(stale_market + fresh_limit)["orb"]["blocker"] == "entry_style_unsupported_live"


def test_the_blocker_sits_below_the_gate_and_evidence_blockers():
    """Precedence is a diagnosis: telling an operator the entry style is unsupported while the
    lane has no evidence, is unregistered, or the gate is off would be the wrong next action."""
    from research.strategy_hunt.promote import _blocker

    limit = {"entry_style": "limit_open"}
    assert _table(_orb_rows(limit, n=3))["orb"]["blocker"] == "no_pass_symbols"
    assert _table(_orb_rows(limit), lanes_enabled=())["orb"]["blocker"] == "not_registered"
    assert _table(_orb_rows(limit), gate_enabled=False)["orb"]["blocker"] == "gate_not_enforcing"
    # ... and the default keeps every pre-Task-F call site's behaviour
    assert _blocker(12, True, "GO", "intraday", False, True, False) == ""
    assert _blocker(12, True, "GO", "intraday", False, True, False, 0, "limit_open") == \
        "entry_style_unsupported_live"


def test_the_style_is_rendered_in_the_promotion_markdown():
    from research.strategy_hunt.promote import promotion_table, render_promotion

    rows = promotion_table(_snapshot(_orb_rows({"entry_style": "limit_open"})), {"orb": {"verdict": "GO"}},
                           lanes_enabled=("orb",), live_enabled=(), swing_allowed=False, **ENFORCING)
    md = render_promotion(rows)
    assert "entry_style" in md and "limit_open" in md
    assert "entry_style_unsupported_live" in md
