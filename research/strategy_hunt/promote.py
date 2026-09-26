# research/strategy_hunt/promote.py
"""Builds the strategy-fit promotion table and its markdown/`.env`-lines report.

Pure functions only (`promotion_table`, `render_promotion`); the CLI's `promote` command
(`research/strategy_hunt/cli.py`) loads the snapshot, re-derives each family's Stage A
scorecard from its shadow ledger, reads the live-facing flags off `Config()`, and passes
plain data in here so this module stays testable without a filesystem or `core` import.

Blocker precedence (first that applies wins):
  1. "no_pass_symbols"        -- fewer than 10 walk-forward PASS rows for the family AND no
                                 pooled regime PASS row (v-playbook-desk-b1-2026-09-25)
  2. "not_registered"         -- family not in FIT_LANES_ENABLED
  3. "scorecard_missing"      -- no Stage A scorecard yet (no shadow ledger / never resolved)
  4. "scorecard_no_go"        -- latest scorecard verdict is NO_GO
  5. "gate_not_enforcing"     -- ENABLE_STRATEGY_FIT_GATE off, or STRATEGY_FIT_SHADOW on
  6. "swing_overnight_disabled" -- family holds past the close (multiday or overnight) but
                                 SWING_ALLOW_OVERNIGHT is off
  7. "entry_style_unsupported_live" -- the lane's latest winning params use a non-market entry
                                 style (Batch-2 Task B's `limit_open`), which the live adapter
                                 cannot place
  else ""                     -- clear to promote

`entry_style_unsupported_live` (Batch-2 Task F): the research simulator can fill a resting limit at
maker cost, the live adapter places market entries only. A lane whose walk-forward winner is a limit
entry therefore earned its expectancy on fills the bot cannot get -- it is not the lane that would go
live. It sits last in the precedence because it is the narrowest diagnosis: while a lane has no
evidence, is unregistered, or the gate is off, the entry style is not the operator's next action.

`gate_not_enforcing` (v-strategy-fit-review2-2026-09-25, Critical 1): promoting a lane to live
entries while the strategy-fit gate is disabled or still in shadow mode means the engine-side
per-symbol check is inert, so the lane's live entries would be governed by nothing but the fit
row the adapter itself reads. The gate must be enabled AND enforcing before any lane goes live.

Recommendation follows the blocker: no_pass_symbols/scorecard_missing -> "collect_shadow";
not_registered -> "register_shadow"; scorecard_no_go/gate_not_enforcing/swing_overnight_disabled
-> "hold"; no blocker -> "promote_live" -- unless the family is already live (in `live_enabled`),
in which case there's nothing to do and it's "hold" instead, so an already-promoted family is
never recommended for (re-)promotion.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from strategies.families import family_names, get_family

logger = logging.getLogger("strategy_hunt.promote")
MIN_PASS_SYMBOLS = 10
MIN_POOLED_PASS_REGIMES = 1
# The only entry style the live adapter can place (strategies/fit_lane.py -> core order placement).
# Anything else in a lane's winning params is a promotion blocker, not a detail.
LIVE_ENTRY_STYLE = "market"

# v-batch1-task1-r1 (Important 1): every hold kind that survives the close. "overnight" is held
# from 15:55 to the next session's open, so SWING_ALLOW_OVERNIGHT governs it exactly as it governs
# "multiday" -- and the engine's own gate (core/engine.py's swing_overnight_gate, which reads
# core.swing_exits.is_multiday) already drops those signals when the flag is off, so promoting such
# a family while it is off would promote a lane that cannot trade.
OVERNIGHT_HOLD_KINDS = ("multiday", "overnight")


def _scorecard_verdict(scorecards: dict[str, Any], family: str) -> str:
    card = scorecards.get(family, "none")
    if isinstance(card, dict):
        return card.get("verdict", "none")
    return "none"


def _blocker(pass_symbols: int, registered: bool, scorecard_verdict: str, hold_kind: str, swing_allowed: bool,
             gate_enabled: bool, gate_shadow: bool, pooled_pass_regimes: int = 0,
             entry_style: str = LIVE_ENTRY_STYLE) -> str:
    # v-playbook-desk-b1-2026-09-25: a pooled regime PASS is family-level evidence measured
    # over the whole universe against stricter floors (score/regime_rows.POOLED_FLOORS), so
    # it answers the same question `pass_symbols` does -- "is there an edge here at all?" --
    # for a family whose per-symbol samples are each too thin to clear the floors alone.
    if pass_symbols < MIN_PASS_SYMBOLS and pooled_pass_regimes < MIN_POOLED_PASS_REGIMES:
        return "no_pass_symbols"
    if not registered:
        return "not_registered"
    if scorecard_verdict == "none":
        return "scorecard_missing"
    if scorecard_verdict == "NO_GO":
        return "scorecard_no_go"
    if not gate_enabled or gate_shadow:
        return "gate_not_enforcing"
    if hold_kind in OVERNIGHT_HOLD_KINDS and not swing_allowed:
        return "swing_overnight_disabled"
    if entry_style != LIVE_ENTRY_STYLE:
        return "entry_style_unsupported_live"
    return ""


_RECOMMENDATION_BY_BLOCKER = {
    "no_pass_symbols": "collect_shadow",
    "not_registered": "register_shadow",
    "scorecard_missing": "collect_shadow",
    "scorecard_no_go": "hold",
    "gate_not_enforcing": "hold",
    "swing_overnight_disabled": "hold",
    "entry_style_unsupported_live": "hold",
    "": "promote_live",
}


def _row_entry_style(row: dict) -> str:
    """One row's winning entry style, from its `params_json`. `market` for a row with no params, no
    `entry_style` key (every snapshot written before Batch-2 Task B is a market-fill search) or an
    unparsable one -- a malformed snapshot must degrade, and defaulting the other way would block
    every existing lane on a technicality."""
    raw = row.get("params_json")
    if not isinstance(raw, str) or not raw:
        return LIVE_ENTRY_STYLE
    try:
        params = json.loads(raw)
    except ValueError:
        logger.warning("promote: unparsable params_json on %s/%s; assuming a market entry",
                       row.get("symbol"), row.get("strategy"))
        return LIVE_ENTRY_STYLE
    style = params.get("entry_style") if isinstance(params, dict) else None
    return str(style) if isinstance(style, str) and style else LIVE_ENTRY_STYLE


def _family_entry_style(frows: list[dict]) -> str:
    """The entry style a family would be promoted *with*: the style of its latest generation of
    walk-forward rows, where "latest" is the newest `window_end` any of them reached.

    Earlier generations are history -- a limit-entry winner from six months ago says nothing about
    the lane as it now stands. Within that newest generation a single non-market row is enough,
    because promotion is family-wide: the lane cannot be run live as it was searched."""
    wf = [r for r in frows if r.get("source", "walkforward") == "walkforward"]
    if not wf:
        return LIVE_ENTRY_STYLE
    latest = max(str(r.get("window_end") or "") for r in wf)
    styles = {_row_entry_style(r) for r in wf if str(r.get("window_end") or "") == latest}
    other = sorted(s for s in styles if s != LIVE_ENTRY_STYLE)
    return other[0] if other else LIVE_ENTRY_STYLE


def _pooled_pass_regimes(snapshot: dict) -> dict[str, set[str]]:
    """{family: {regimes with a pooled PASS row, either side}} from `snapshot["pooled_rows"]`.

    Absent in snapshots written before the pooled lanes existed, so every read defaults."""
    out: dict[str, set[str]] = {}
    for key, row in (snapshot.get("pooled_rows") or {}).items():
        if not isinstance(row, dict) or row.get("verdict") != "PASS":
            continue
        parts = str(key).split(":")
        regime = row.get("regime") or (parts[2] if len(parts) > 2 else "")
        family = row.get("strategy") or (parts[0] if parts else "")
        if family and regime:
            out.setdefault(family, set()).add(regime)
    return out


def promotion_table(snapshot: dict, scorecards: dict[str, dict], lanes_enabled: tuple[str, ...],
                     live_enabled: tuple[str, ...], swing_allowed: bool,
                     gate_enabled: bool = False, gate_shadow: bool = True) -> list[dict]:
    """One row per known family (`strategies.families.family_names`), independent of
    whether the family has any rows in `snapshot` yet.

    `gate_enabled`/`gate_shadow` are `Config().ENABLE_STRATEGY_FIT_GATE` /
    `STRATEGY_FIT_SHADOW`; their defaults mirror the shipped config defaults, which block
    promotion (`gate_not_enforcing`)."""
    rows_by_family: dict[str, list[dict]] = {}
    for sym_rows in snapshot.get("rows", {}).values():
        for row in sym_rows.values():
            rows_by_family.setdefault(row["strategy"], []).append(row)
    pooled_pass = _pooled_pass_regimes(snapshot)

    out: list[dict] = []
    for name in family_names():
        family = get_family(name)
        frows = rows_by_family.get(name, [])
        pass_symbols = sum(1 for r in frows if r.get("verdict") == "PASS" and r.get("source") == "walkforward")
        watch_symbols = sum(1 for r in frows if r.get("verdict") == "WATCH" and r.get("source") == "walkforward")
        registered = name in lanes_enabled
        live = name in live_enabled
        scorecard_verdict = _scorecard_verdict(scorecards, name)

        pooled_pass_regimes = len(pooled_pass.get(name, set()))
        entry_style = _family_entry_style(frows)
        blocker = _blocker(pass_symbols, registered, scorecard_verdict, family.hold_kind, swing_allowed,
                           gate_enabled, gate_shadow, pooled_pass_regimes, entry_style)
        if live and blocker == "":
            recommendation = "hold"  # already promoted; nothing to do
        else:
            recommendation = _RECOMMENDATION_BY_BLOCKER[blocker]

        out.append({
            "family": name, "pass_symbols": pass_symbols, "watch_symbols": watch_symbols,
            "pooled_pass_regimes": pooled_pass_regimes, "entry_style": entry_style,
            "hold_kind": family.hold_kind, "registered": registered, "live": live,
            "scorecard_verdict": scorecard_verdict, "blocker": blocker, "recommendation": recommendation,
        })
    return out


def render_promotion(rows: list[dict]) -> str:
    """Markdown table plus the exact `.env` lines to apply. Never writes `.env` itself --
    the caller (CLI) only prints/saves this text."""
    lines = [
        "# Strategy-hunt promotion",
        "",
        "| family | pass_symbols | watch_symbols | pooled_pass_regimes | hold_kind | entry_style | "
        "registered | live | scorecard_verdict | blocker | recommendation |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['family']} | {r['pass_symbols']} | {r['watch_symbols']} | "
                     f"{r.get('pooled_pass_regimes', 0)} | {r['hold_kind']} | "
                     f"{r.get('entry_style', LIVE_ENTRY_STYLE)} | "
                     f"{r['registered']} | {r['live']} | {r['scorecard_verdict']} | {r['blocker'] or '-'} | "
                     f"{r['recommendation']} |")

    register = sorted({r["family"] for r in rows if r["registered"] or r["recommendation"] == "register_shadow"})
    newly_live = {r["family"] for r in rows if r["recommendation"] == "promote_live"}
    # Only print a live-entries line when something is actually being newly promoted -- an
    # already-live family with nothing new to promote gets no line of its own (recommendation
    # is "hold", not "promote_live"). When something else *is* newly promoted, the union
    # naturally includes any already-live families too, which is fine (it's still the correct
    # full value for the flag).
    live = sorted({r["family"] for r in rows if r["live"]} | newly_live) if newly_live else []
    swing_blocked = sorted({r["family"] for r in rows if r["blocker"] == "swing_overnight_disabled"})

    lines += ["", "## Apply", ""]
    applied_any = False
    if register:
        lines.append(f"FIT_LANES_ENABLED={','.join(register)}")
        applied_any = True
    if live:
        lines.append(f"FIT_LANE_LIVE_ENTRIES_ENABLED={','.join(live)}")
        applied_any = True
    if swing_blocked:
        lines.append(f"SWING_ALLOW_OVERNIGHT=1  # unblocks: {','.join(swing_blocked)}")
        applied_any = True
    if not applied_any:
        lines.append("(no changes recommended)")

    lines += ["", "Rollback: unset (or revert) the flag in `.env` and restart via `./deploy.sh`."]
    return "\n".join(lines) + "\n"
