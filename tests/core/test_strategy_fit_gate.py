import json
from datetime import datetime

import pytest


def _fit(tmp_path, rows, computed_at="2025-09-25T01:00:00", now=datetime(2025, 9, 25, 10)):
    from core.strategy_fit import StrategyFit
    p = tmp_path / "fit.json"
    p.write_text(json.dumps({"computed_at": computed_at, "floors": {}, "rows": rows}))
    f = StrategyFit(str(p), max_age_days=7, clock=lambda: now)
    f.reload_if_changed()
    return f


def _row(sym, strat, side, verdict, score):
    return {"symbol": sym, "strategy": strat, "side": side, "verdict": verdict, "score": score, "exp_r": score, "exp_r_lb": 0.0,
            "t_stat": 2.0, "pf": 1.3, "wr": 0.5, "stability": 0.6, "decay": 0.8, "n": 160, "source": "walkforward", "params_json": "{}"}


def test_disabled_and_ungoverned(tmp_path):
    from core.strategy_fit_gate import evaluate_fit_gate, family_for
    fit = _fit(tmp_path, {})
    d = evaluate_fit_gate(fit, "day_trade_momentum", "long", "NVDA", enabled=False, shadow=True)
    assert (d.action, d.reason, d.governed) == ("proceed", "gate_disabled", False)
    d2 = evaluate_fit_gate(fit, "breakout", "long", "NVDA", enabled=True, shadow=True)
    assert (d2.action, d2.reason, d2.governed) == ("proceed", "fit_not_governed", False)
    assert family_for("fit:vwap_reversion") == "vwap_reversion" and family_for("orb_contraction_rvol") == "orb" and family_for("x") is None


def test_pass_watch_fail_missing_in_shadow_and_enforce(tmp_path):
    from core.strategy_fit_gate import evaluate_fit_gate
    fit = _fit(tmp_path, {"NVDA": {"dt_momentum:long": _row("NVDA", "dt_momentum", "long", "PASS", 0.3),
                                   "orb:long": _row("NVDA", "orb", "long", "WATCH", 0.1),
                                   "mr_20d:long": _row("NVDA", "mr_20d", "long", "FAIL", -0.1)}})
    assert evaluate_fit_gate(fit, "day_trade_momentum", "long", "NVDA", True, True).reason == "fit_pass"
    # v-strategy-fit-review2-2026-09-25 (Important 2): "news" was never a real
    # strategy identifier and is no longer in LANE_MAP; free_news_sentiment is
    # (strategies/news_strategy.py:1641) and has no row here.
    for strat, reason in (("orb_contraction_rvol", "fit_watch_shadow"), ("fit:mr_20d", "fit_fail"),
                          ("free_news_sentiment", "fit_missing_row")):
        d_shadow = evaluate_fit_gate(fit, strat, "long", "NVDA", True, True)
        d_live = evaluate_fit_gate(fit, strat, "long", "NVDA", True, False)
        assert (d_shadow.action, d_shadow.reason) == ("shadow", reason) and (d_live.action, d_live.reason) == ("skip", reason)
        assert d_shadow.governed and d_live.governed


def test_stale_and_missing_snapshot(tmp_path):
    from core.strategy_fit_gate import evaluate_fit_gate
    stale = _fit(tmp_path, {"NVDA": {"dt_momentum:long": _row("NVDA", "dt_momentum", "long", "PASS", 0.3)}}, computed_at="2025-09-01T00:00:00")
    d = evaluate_fit_gate(stale, "day_trade_momentum", "long", "NVDA", True, False)
    assert (d.action, d.reason) == ("skip", "fit_stale") and d.age_days > 7
    from core.strategy_fit import StrategyFit
    missing = StrategyFit(str(tmp_path / "none.json"), clock=lambda: datetime(2025, 9, 25))
    missing.reload_if_changed()
    assert evaluate_fit_gate(missing, "day_trade_momentum", "long", "NVDA", True, True).reason == "fit_missing"


def test_priority_between_competing_pass_lanes(tmp_path):
    from core.strategy_fit_gate import evaluate_fit_gate
    fit = _fit(tmp_path, {"NVDA": {"dt_momentum:long": _row("NVDA", "dt_momentum", "long", "PASS", 0.3),
                                   "orb:long": _row("NVDA", "orb", "long", "PASS", 0.5)}})
    d = evaluate_fit_gate(fit, "day_trade_momentum", "long", "NVDA", True, False, competing_families=["orb"])
    assert (d.action, d.reason) == ("skip", "fit_lower_priority")
    d2 = evaluate_fit_gate(fit, "orb_contraction_rvol", "long", "NVDA", True, False, competing_families=["dt_momentum"])
    assert d2.reason == "fit_pass"
    assert set(d2.audit_fields()) >= {"family", "verdict", "score", "exp_r", "stability", "source", "age_days"}


# ──────────────────────────────────────────────────────────────────────
# v-strategy-fit-review2-2026-09-25 — Important 2: LANE_MAP keys must be
# real live strategy identifiers.
# ──────────────────────────────────────────────────────────────────────

def test_every_lane_map_key_is_a_real_strategy_literal():
    """A LANE_MAP key that no strategy ever writes into
    `reasoning["strategy"]` means the gate silently governs nothing for that
    lane. Anchor every key to a `'strategy': '<key>'` literal under
    strategies/*.py (grep-based, offline, no imports)."""
    import re
    from pathlib import Path

    from core.strategy_fit_gate import LANE_MAP

    sources = {p.name: p.read_text() for p in sorted(Path("strategies").glob("*.py"))}
    assert sources, "no strategy sources found — wrong working directory?"

    missing = []
    for key in LANE_MAP:
        pattern = re.compile(r"""['"]strategy['"]\s*:\s*['"]""" + re.escape(key) + r"""['"]""")
        if not any(pattern.search(src) for src in sources.values()):
            missing.append(key)
    assert not missing, f"LANE_MAP keys with no matching strategy literal under strategies/*.py: {missing}"


def test_lane_map_covers_the_known_live_strategy_identifiers():
    """The other direction: the governed strategies the manual names must all
    still resolve to a family, and identifiers that never existed must not."""
    from core.strategy_fit_gate import family_for

    assert family_for("mean_reversion") == "mean_rev_intraday"
    assert family_for("mean_reversion_short") == "mean_rev_intraday"
    assert family_for("oversold_bounce_v2") == "mean_rev_intraday"
    assert family_for("day_trade_momentum") == "dt_momentum"
    assert family_for("day_trade_momentum_short") == "dt_momentum"
    assert family_for("orb_contraction_rvol") == "orb"
    assert family_for("news_sentiment") == "news_catalyst"
    assert family_for("free_news_sentiment") == "news_catalyst"
    # Retired placeholders that no strategy ever emitted.
    assert family_for("oversold_v2") is None
    assert family_for("news") is None


def test_nan_competing_score_is_not_higher_priority(tmp_path):
    """Minor: a competing PASS lane with a NaN score must not win the
    priority comparison (and must not crash it)."""
    from core.strategy_fit_gate import evaluate_fit_gate

    nan_row = _row("NVDA", "orb", "long", "PASS", 0.5)
    nan_row["score"] = "not-a-number"          # FitRow.from_dict -> nan
    fit = _fit(tmp_path, {"NVDA": {"dt_momentum:long": _row("NVDA", "dt_momentum", "long", "PASS", 0.3),
                                   "orb:long": nan_row}})
    d = evaluate_fit_gate(fit, "day_trade_momentum", "long", "NVDA", True, False, competing_families=["orb"])
    assert (d.action, d.reason) == ("proceed", "fit_pass")


# ──────────────────────
# v-playbook-desk-b1-2026-09-25 — regime-conditioned gate
# ──────────────────────

def _rrow(sym, strat, side, verdict, score, regime, source="walkforward"):
    row = _row(sym, strat, side, verdict, score)
    row["regime"] = regime
    row["source"] = source
    return row


def _regime_fit(tmp_path, rows, regime_rows=None, pooled_rows=None, computed_at="2025-09-25T01:00:00"):
    from core.strategy_fit import StrategyFit
    p = tmp_path / "fit_regime.json"
    p.write_text(json.dumps({"computed_at": computed_at, "floors": {}, "rows": rows,
                             "regime_rows": regime_rows or {}, "pooled_rows": pooled_rows or {}}))
    f = StrategyFit(str(p), max_age_days=7, clock=lambda: datetime(2025, 9, 25, 10))
    f.reload_if_changed()
    return f


def test_regime_row_governs_when_a_live_regime_is_known(tmp_path):
    from core.strategy_fit_gate import evaluate_fit_gate
    fit = _regime_fit(tmp_path,
                      rows={"NVDA": {"dt_momentum:long": _row("NVDA", "dt_momentum", "long", "PASS", 0.3)}},
                      regime_rows={"NVDA": {"dt_momentum:long:risk_off": _rrow("NVDA", "dt_momentum", "long",
                                                                              "FAIL", -0.1, "risk_off")}})
    # unconditional row is PASS, but the risk_off row for this symbol FAILs -> blocked
    d = evaluate_fit_gate(fit, "day_trade_momentum", "long", "NVDA", True, False, regime="risk_off")
    assert (d.action, d.reason) == ("skip", "fit_fail")
    assert d.audit_fields()["regime"] == "risk_off" and d.audit_fields()["pooled"] is False
    # no regime -> the unconditional row still governs
    assert evaluate_fit_gate(fit, "day_trade_momentum", "long", "NVDA", True, False).reason == "fit_pass"
    # an unknown / unrecognised regime is treated as "no regime": the unconditional row
    # governs and nothing is audited as the governing regime
    for regime in ("unknown", "", None, "sideways"):
        d2 = evaluate_fit_gate(fit, "day_trade_momentum", "long", "NVDA", True, False, regime=regime)
        assert d2.reason == "fit_pass", regime
        assert "regime" not in d2.audit_fields(), regime


def test_pooled_regime_row_lets_a_lane_trade_a_symbol_with_no_rows(tmp_path):
    from core.strategy_fit_gate import evaluate_fit_gate
    fit = _regime_fit(tmp_path, rows={},
                      pooled_rows={"orb:long:risk_on": _rrow("*", "orb", "long", "PASS", 0.6, "risk_on",
                                                             source="pooled")})
    d = evaluate_fit_gate(fit, "orb_contraction_rvol", "long", "TSLA", True, False, regime="risk_on")
    assert (d.action, d.reason) == ("proceed", "fit_pass")
    fields = d.audit_fields()
    assert fields["pooled"] is True and fields["regime"] == "risk_on" and fields["source"] == "pooled"
    # a regime with no pooled row and no symbol row is still missing
    assert evaluate_fit_gate(fit, "orb_contraction_rvol", "long", "TSLA", True, False,
                             regime="risk_off").reason == "fit_missing_row"


def test_regime_lookup_also_applies_to_competing_families(tmp_path):
    from core.strategy_fit_gate import evaluate_fit_gate
    fit = _regime_fit(tmp_path,
                      rows={"NVDA": {"dt_momentum:long": _row("NVDA", "dt_momentum", "long", "PASS", 0.9),
                                     "orb:long": _row("NVDA", "orb", "long", "PASS", 0.1)}},
                      regime_rows={"NVDA": {"dt_momentum:long:risk_on": _rrow("NVDA", "dt_momentum", "long",
                                                                             "PASS", 0.2, "risk_on"),
                                            "orb:long:risk_on": _rrow("NVDA", "orb", "long", "PASS", 0.8,
                                                                      "risk_on")}})
    # in risk_on the orb regime row outscores dt_momentum's, reversing the unconditional order
    d = evaluate_fit_gate(fit, "day_trade_momentum", "long", "NVDA", True, False, competing_families=["orb"],
                          regime="risk_on")
    assert (d.action, d.reason) == ("skip", "fit_lower_priority")
    d2 = evaluate_fit_gate(fit, "day_trade_momentum", "long", "NVDA", True, False, competing_families=["orb"])
    assert d2.reason == "fit_pass"


def test_gate_tolerates_a_reader_without_lookup_regime(tmp_path):
    """The gate must not require the newer reader API (an older or stubbed
    `fit` object simply falls back to `lookup`)."""
    from core.strategy_fit_gate import evaluate_fit_gate

    class _Legacy:
        def __init__(self, row):
            self._row = row

        def age_days(self):
            return 1.0

        def is_stale(self):
            return False

        def summary(self):
            return {"computed_at": "2025-09-25T01:00:00"}

        def lookup(self, symbol, strategy, side):
            return self._row

    from core.strategy_fit import FitRow
    legacy = _Legacy(FitRow.from_dict(_row("NVDA", "dt_momentum", "long", "PASS", 0.3)))
    d = evaluate_fit_gate(legacy, "day_trade_momentum", "long", "NVDA", True, False, regime="risk_on")
    assert (d.action, d.reason) == ("proceed", "fit_pass")


def test_stale_snapshot_still_reports_the_regime(tmp_path):
    from core.strategy_fit_gate import evaluate_fit_gate
    stale = _regime_fit(tmp_path, rows={"NVDA": {"dt_momentum:long": _row("NVDA", "dt_momentum", "long", "PASS", 0.3)}},
                        computed_at="2025-09-01T00:00:00")
    d = evaluate_fit_gate(stale, "day_trade_momentum", "long", "NVDA", True, False, regime="risk_on")
    assert (d.action, d.reason) == ("skip", "fit_stale") and d.audit_fields()["regime"] == "risk_on"
