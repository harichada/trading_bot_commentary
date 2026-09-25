# research/strategy_hunt/promote.py
"""Builds the strategy-fit promotion table and its markdown/`.env`-lines report.

Pure functions only (`promotion_table`, `render_promotion`); the CLI's `promote` command
(`research/strategy_hunt/cli.py`) loads the snapshot, re-derives each family's Stage A
scorecard from its shadow ledger, reads the live-facing flags off `Config()`, and passes
plain data in here so this module stays testable without a filesystem or `core` import.

Blocker precedence (first that applies wins):
  1. "no_pass_symbols"        -- fewer than 10 walk-forward PASS rows for the family
  2. "not_registered"         -- family not in FIT_LANES_ENABLED
  3. "scorecard_missing"      -- no Stage A scorecard yet (no shadow ledger / never resolved)
  4. "scorecard_no_go"        -- latest scorecard verdict is NO_GO
  5. "gate_not_enforcing"     -- ENABLE_STRATEGY_FIT_GATE off, or STRATEGY_FIT_SHADOW on
  6. "swing_overnight_disabled" -- multiday family but SWING_ALLOW_OVERNIGHT is off
  else ""                     -- clear to promote

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

from typing import Any

from strategies.families import family_names, get_family

MIN_PASS_SYMBOLS = 10


def _scorecard_verdict(scorecards: dict[str, Any], family: str) -> str:
    card = scorecards.get(family, "none")
    if isinstance(card, dict):
        return card.get("verdict", "none")
    return "none"


def _blocker(pass_symbols: int, registered: bool, scorecard_verdict: str, hold_kind: str, swing_allowed: bool,
             gate_enabled: bool, gate_shadow: bool) -> str:
    if pass_symbols < MIN_PASS_SYMBOLS:
        return "no_pass_symbols"
    if not registered:
        return "not_registered"
    if scorecard_verdict == "none":
        return "scorecard_missing"
    if scorecard_verdict == "NO_GO":
        return "scorecard_no_go"
    if not gate_enabled or gate_shadow:
        return "gate_not_enforcing"
    if hold_kind == "multiday" and not swing_allowed:
        return "swing_overnight_disabled"
    return ""


_RECOMMENDATION_BY_BLOCKER = {
    "no_pass_symbols": "collect_shadow",
    "not_registered": "register_shadow",
    "scorecard_missing": "collect_shadow",
    "scorecard_no_go": "hold",
    "gate_not_enforcing": "hold",
    "swing_overnight_disabled": "hold",
    "": "promote_live",
}


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

    out: list[dict] = []
    for name in family_names():
        family = get_family(name)
        frows = rows_by_family.get(name, [])
        pass_symbols = sum(1 for r in frows if r.get("verdict") == "PASS" and r.get("source") == "walkforward")
        watch_symbols = sum(1 for r in frows if r.get("verdict") == "WATCH" and r.get("source") == "walkforward")
        registered = name in lanes_enabled
        live = name in live_enabled
        scorecard_verdict = _scorecard_verdict(scorecards, name)

        blocker = _blocker(pass_symbols, registered, scorecard_verdict, family.hold_kind, swing_allowed,
                           gate_enabled, gate_shadow)
        if live and blocker == "":
            recommendation = "hold"  # already promoted; nothing to do
        else:
            recommendation = _RECOMMENDATION_BY_BLOCKER[blocker]

        out.append({
            "family": name, "pass_symbols": pass_symbols, "watch_symbols": watch_symbols,
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
        "| family | pass_symbols | watch_symbols | hold_kind | registered | live | scorecard_verdict | blocker | "
        "recommendation |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['family']} | {r['pass_symbols']} | {r['watch_symbols']} | {r['hold_kind']} | "
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
