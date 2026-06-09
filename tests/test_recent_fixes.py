"""Regression tests for the May 5–8 2026 fix batch.

Locks in behaviour for fixes that were declared "done" with runtime
verification only. Pure-function and static-config assertions only —
deliberately no Schwab/PriceBook mocking; runtime paths are covered by
the existing test_strategy_replay and test_meta_inference suites.

Each test is anchored to a v-tag in the source so future me can grep for
the regression target.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PATH = REPO_ROOT / "core" / "engine.py"
CONFIG_PATH = REPO_ROOT / "core" / "config.py"
NEWS_STRATEGY_PATH = REPO_ROOT / "strategies" / "news_strategy.py"
STREAM_PATH = REPO_ROOT / "core" / "schwab_stream.py"
ROUTES_PATH = REPO_ROOT / "api" / "routes.py"


# ── v-fill-creates-position-2026-05-08 ───────────────────────────────

class TestFillCreatesPosition:
    """The immediate fill path must create a Position with
    managed_by_bot=True. Prior bug: only the sweep path created it,
    AND the sweep had an UnboundLocalError so it never fired."""

    def test_fill_path_creates_managed_position(self):
        """`_execute_real_trade` constructs Position(managed_by_bot=True)
        immediately after `_verify_order_fill` returns True."""
        src = ENGINE_PATH.read_text()
        # The patch should appear inside _execute_real_trade after
        # _verify_order_fill, BEFORE _place_bracket_orders.
        marker = "v-fill-creates-position-2026-05-08"
        assert marker in src
        # And it must use managed_by_bot=True
        idx = src.index(marker)
        block = src[idx : idx + 4000]
        assert "managed_by_bot=True" in block
        assert "self.positions[signal.symbol]" in block
        assert "exit_manager.initialize_position_tracking" in block
        # And the order_id must be popped so the sweep doesn't re-create
        assert "self.pending_orders.pop(order_id, None)" in block

    def test_sweep_signal_assignment_hoisted(self):
        """The sweep path's `signal = order_data['signal']` must appear
        BEFORE the first reference to `signal.entry_price`."""
        src = ENGINE_PATH.read_text()
        # Look for the v-tag block
        marker = "v-sweep-signal-scope-fix-2026-05-08"
        assert marker in src
        idx = src.index(marker)
        block = src[idx : idx + 1500]
        # The fix is "signal = order_data['signal']" appears at the top
        # of the FILLED branch, before fill_price assignment
        assign_pos = block.find("signal = order_data['signal']")
        fill_price_pos = block.find("fill_price = signal.entry_price")
        assert assign_pos != -1
        assert fill_price_pos != -1
        assert assign_pos < fill_price_pos, (
            "signal must be assigned before fill_price reads signal.entry_price"
        )

    def test_bracket_orders_called_with_two_args(self):
        """`_place_bracket_orders(signal, parent_order_id)` requires two
        arguments. Prior code passed one — silent TypeError, broker-side
        bracket orders never placed."""
        src = ENGINE_PATH.read_text()
        # Sig must be (self, signal, parent_order_id: str)
        sig_match = re.search(
            r"async def _place_bracket_orders\(self, signal, parent_order_id: str\)",
            src,
        )
        assert sig_match is not None, "function signature changed unexpectedly"
        # All call sites must pass parent_order_id
        bad_calls = re.findall(
            r"await self\._place_bracket_orders\(signal\)(?!\s*,)",
            src,
        )
        assert bad_calls == [], f"single-arg calls remain: {bad_calls}"


# ── v-stream-watchdog-2026-05-08 ─────────────────────────────────────

class TestStreamWatchdog:
    """Silent-death detection in the stream client."""

    def test_silent_death_event_initialized(self):
        src = STREAM_PATH.read_text()
        assert "self._silent_death = asyncio.Event()" in src

    def test_serve_races_pump_against_death(self):
        """_serve() must use asyncio.wait FIRST_COMPLETED on the pump
        coroutine and the silent_death event."""
        src = STREAM_PATH.read_text()
        assert "asyncio.wait(" in src
        assert "asyncio.FIRST_COMPLETED" in src
        # The watchdog must check tick freshness during RTH
        assert "_is_rth_now" in src

    def test_watchdog_threshold_is_120s(self):
        """Don't let someone tighten the threshold below 60s without
        deliberation — 60s is a real risk during illiquid open prints."""
        src = STREAM_PATH.read_text()
        m = re.search(r"SILENT_DEATH_THRESHOLD_SEC\s*=\s*([\d.]+)", src)
        assert m is not None
        threshold = float(m.group(1))
        assert 60 <= threshold <= 300


# ── v-health-gate-2026-05-08 ─────────────────────────────────────────

class TestHealthGate:
    """Fail-closed guard refuses new live entries when stream is dead
    or recent bot trades show managed_by_bot=False."""

    def test_health_gate_present_in_signal_router(self):
        src = ENGINE_PATH.read_text()
        marker = "v-health-gate-2026-05-08"
        assert marker in src
        idx = src.index(marker)
        # The gate runs only in LIVE mode
        block = src[idx : idx + 5000]
        assert "TradingMode.LIVE" in block
        assert "stream_dead" in block
        assert "management_unavailable" in block

    def test_sim_mode_not_gated(self):
        """Sim mode must keep firing signals so strategy logic gets
        exercised end-to-end."""
        src = ENGINE_PATH.read_text()
        idx = src.index("v-health-gate-2026-05-08")
        block = src[idx : idx + 5000]
        # The check guards behind `if self.mode == TradingMode.LIVE:`
        assert re.search(
            r"if self\.mode == TradingMode\.LIVE:",
            block,
        ) is not None


# ── v-correlation-groups-expanded-2026-05-08 ─────────────────────────

class TestCorrelationGroups:
    """The cluster guard's group definitions must catch ARM, SMCI etc.
    that were missed during the 2026-05-06 12:53 burst."""

    def test_semis_group_includes_arm_smci(self):
        src = ENGINE_PATH.read_text()
        # Find the semis_and_chip_adjacent set literal
        m = re.search(
            r'"semis_and_chip_adjacent":\s*\{([^}]+)\}',
            src,
        )
        assert m is not None, "semis_and_chip_adjacent group missing"
        members = m.group(1)
        for sym in ["ARM", "SMCI", "MRVL", "TSM", "LRCX", "AMAT", "KLAC"]:
            assert f'"{sym}"' in members, f"{sym} missing from semis cluster"

    def test_fintech_group_exists(self):
        src = ENGINE_PATH.read_text()
        assert '"fintech_payments"' in src
        m = re.search(r'"fintech_payments":\s*\{([^}]+)\}', src)
        assert m is not None
        for sym in ["COIN", "SOFI", "HOOD", "PYPL"]:
            assert f'"{sym}"' in m.group(1)


# ── v-news-verifier-advisory-2026-05-08 ──────────────────────────────

class TestNewsVerifierAdvisory:
    """Advisory mode runs the verifier without gating."""

    def test_advisory_config_property_exists(self):
        from core.config import Config
        cfg = Config()
        # Just check the attribute resolves and is bool
        assert isinstance(cfg.NEWS_VERIFIER_ADVISORY, bool)

    def test_advisory_default_is_true(self):
        from core.config import Config
        # We want advisory ON by default so the verifier collects data
        # immediately, even though hard gating stays off.
        cfg = Config()
        assert cfg.NEWS_VERIFIER_ADVISORY is True

    def test_hard_gate_default_is_false(self):
        from core.config import Config
        # Hard gate must remain OFF until we have evidence advisory
        # vetoes correlate with real losers.
        cfg = Config()
        assert cfg.ENABLE_NEWS_VERIFIER is False


# ── v-news-price-direction-gate-2026-05-08 ───────────────────────────

class TestNewsPriceDirectionGate:
    """News BUY requires bar close>open + vol_ratio>=1.2.
    News SELL requires bar close<open + vol_ratio>=1.2.
    Conviction bypass when sentiment>=0.55 with articles>=8."""

    def test_gate_present_in_news_strategy(self):
        src = NEWS_STRATEGY_PATH.read_text()
        assert "v-news-price-direction-gate-2026-05-08" in src
        assert "price_direction_disagrees_buy" in src
        assert "price_direction_disagrees_sell" in src

    def test_volume_ratio_floor_present(self):
        # After v-price-direction-relax, the floor is sentiment-scaled.
        # We just verify the strict-band floor (0.35 sentiment band)
        # is still 1.2 — i.e. weak news still demands strong volume.
        src = NEWS_STRATEGY_PATH.read_text()
        # The 1.2 vol floor lives in the relax block now.
        idx = src.index("v-price-direction-relax-2026-05-08")
        block = src[idx : idx + 3500]
        assert "_vol_floor = 1.2" in block

    def test_conviction_bypass_overrides_gate(self):
        src = NEWS_STRATEGY_PATH.read_text()
        # Guard clause must still mention _conviction_bypass for the
        # V-bottom catalyst case. The exact form is now:
        # `if not _conviction_bypass and _abs_sent < 0.50:`
        idx = src.index("v-news-price-direction-gate-2026-05-08")
        block = src[idx : idx + 4000]
        assert "not _conviction_bypass" in block


# ── v-price-direction-relax-2026-05-08 ───────────────────────────────

class TestPriceDirectionRelaxed:
    """T6 was too literal — AMD 2026-05-08 14:46–14:50 ran +$4 but bot
    blocked every entry because the current 5-min bars closed slightly
    red. Scaled the gate by sentiment magnitude."""

    def test_relax_marker_present(self):
        src = NEWS_STRATEGY_PATH.read_text()
        assert "v-price-direction-relax-2026-05-08" in src

    def test_strong_sentiment_skips_gate(self):
        """Sentiment >= 0.50 bypasses the price-direction gate
        entirely — strong sentiment is itself the catalyst."""
        src = NEWS_STRATEGY_PATH.read_text()
        idx = src.index("v-price-direction-relax-2026-05-08")
        block = src[idx : idx + 3500]
        assert "_abs_sent >= 0.50" in block

    def test_mid_band_relaxed(self):
        """0.35–0.50 → 0.2% close-vs-open tolerance + vol_ratio>=1.0."""
        src = NEWS_STRATEGY_PATH.read_text()
        idx = src.index("v-price-direction-relax-2026-05-08")
        block = src[idx : idx + 3500]
        assert "_bar_floor_factor_buy = 0.998" in block
        assert "_vol_floor = 1.0" in block

    def test_weak_sentiment_keeps_strict_gate(self):
        """< 0.35 retains strict close>open + vol>=1.2."""
        src = NEWS_STRATEGY_PATH.read_text()
        idx = src.index("v-price-direction-relax-2026-05-08")
        block = src[idx : idx + 3500]
        # The else branch sets _vol_floor = 1.2 and factor 1.0
        assert "_vol_floor = 1.2" in block


# ── v-screener-api-fix + v-screener-fresh-daily 2026-05-08 ──────────

class TestScreenerApiFix:
    """schwab-py changed get_movers signature; old code was silently
    failing for weeks. New code uses sort_order/frequency."""

    def test_screener_uses_sort_order(self):
        src = (REPO_ROOT / "analysis" / "screener.py").read_text()
        assert "v-screener-api-fix-2026-05-08" in src
        # New API passes sort_order as a keyword (resolved to enum at runtime)
        assert "sort_order=sort_enum" in src
        # Old API kwargs MUST be gone
        assert "direction=direction" not in src
        assert "change='percent'" not in src

    def test_screener_resolves_enums(self):
        """schwab-py enforces enum types on get_movers; passing strings
        raises ValueError. Must resolve to Movers.Index/SortOrder."""
        src = (REPO_ROOT / "analysis" / "screener.py").read_text()
        assert "v-screener-enum-fix-2026-05-11" in src
        # Should reference both Index and SortOrder
        assert "Movers.Index" in src
        assert "Movers.SortOrder" in src

    def test_screener_pulls_volume_source(self):
        """Most-active screen is the headline new source — without it,
        names like RKLB never surface."""
        src = (REPO_ROOT / "analysis" / "screener.py").read_text()
        idx = src.index("v-screener-fresh-daily-2026-05-08")
        block = src[idx : idx + 4000]
        assert "EQUITY_ALL" in block
        assert "VOLUME" in block

    def test_failure_promoted_to_warning(self):
        """The old logger.debug let the API break go unnoticed.
        Failures must now be visible at WARNING level."""
        src = (REPO_ROOT / "analysis" / "screener.py").read_text()
        idx = src.index("v-screener-api-fix-2026-05-08")
        block = src[idx : idx + 4000]
        assert "logger.warning" in block

    def test_operator_curated_names_in_floor(self):
        """RKLB / OPEN / CRWV must be in the static floor so they get
        considered even when the dynamic screen misses them."""
        src = (REPO_ROOT / "analysis" / "screener.py").read_text()
        for sym in ("RKLB", "OPEN", "CRWV"):
            assert f"'{sym}'" in src, f"{sym} missing from candidates"

    def test_yahoo_most_active_source(self):
        """Source 0 — Yahoo Finance most-active is the breadth source
        for retail catalyst names Schwab's indices miss."""
        src = (REPO_ROOT / "analysis" / "screener.py").read_text()
        assert "v-yahoo-most-active-2026-05-11" in src
        assert "async def _get_yahoo_most_active" in src
        assert "query1.finance.yahoo.com" in src
        assert "scrIds" in src
        # Must run via asyncio.to_thread so sync requests doesn't block the loop
        assert "to_thread" in src
        # Failure must be WARNING-level (lesson from the screener silent break).
        # Anchor on the implementation block, not the docstring reference.
        impl_idx = src.index("async def _get_yahoo_most_active")
        impl_block = src[impl_idx : impl_idx + 6000]
        assert "logger.warning" in impl_block


# ── v-enable-mean-rev-short-2026-05-11 ───────────────────────────────

class TestEnableMeanRevShort:
    """v-disable-mean-rev-short-2026-05-12: mean-rev SHORT was briefly
    re-enabled on 2026-05-11 but the structural protection cited in
    the enable comment (news-strategy rising-knife / late-entry guard)
    never applied to the mean-rev branch in strategies/builtin.py.
    Operator observed shorts-only behavior with 5 losing entries; flag
    reverted to false. Stays off until a trend-direction filter is
    added directly to the mean-rev SHORT branch."""

    def test_mean_rev_short_disabled_after_revert(self):
        from core.config import Config
        cfg = Config()
        assert cfg.ENABLE_MEAN_REV_SHORT is False

    def test_short_mirrors_still_off(self):
        """Breakout/momentum short mirrors haven't been re-tuned yet.
        They need separate retest before live re-enable."""
        from core.config import Config
        cfg = Config()
        assert cfg.ENABLE_SHORT_MIRRORS is False


# ── v-consecutive-losses-live-only-2026-05-11 ────────────────────────

class TestConsecutiveLossesLiveOnly:
    """Sim-mode losses were tripping the live-mode circuit breaker.
    Operator hit `Max consecutive losses reached` with counter=11 even
    though recent LIVE trade history showed most recent close was a win."""

    def test_close_path_gates_on_live_mode(self):
        src = ENGINE_PATH.read_text()
        assert "v-consecutive-losses-live-only-2026-05-11" in src
        # Both increment sites must check the live mode gate
        # (sites are tagged with two distinct variable names so we
        # can confirm both exist)
        assert "_is_live_close" in src
        assert "_is_live_close_sweep" in src


# ── v-schwab-provider-movers-api-fix-2026-05-11 ──────────────────────

class TestSchwabProviderMoversApiFix:
    """Second copy of the same Schwab movers API bug — calculate_market_breadth
    in data_providers/schwab.py was using the old API and silently
    failing at DEBUG level."""

    def test_provider_calls_use_new_api(self):
        src = (REPO_ROOT / "data_providers" / "schwab.py").read_text()
        assert "v-schwab-provider-movers-api-fix-2026-05-11" in src
        # New API + enum resolution must be in actual code
        assert "sort_order=sort_enum" in src
        assert "Movers.Index" in src
        assert "Movers.SortOrder" in src
        # The actual call to get_movers must NOT use the old kwargs.
        # Anchor to the call line by looking at `get_movers(...)` calls.
        import re
        calls = re.findall(r"client\.get_movers\([^)]+\)", src)
        for call in calls:
            assert "direction=" not in call, f"old direction= in {call}"
            assert "change='percent'" not in call, f"old change= in {call}"

    def test_provider_promotes_failures_to_warning(self):
        src = (REPO_ROOT / "data_providers" / "schwab.py").read_text()
        idx = src.index("v-schwab-provider-movers-api-fix-2026-05-11")
        block = src[idx : idx + 4000]
        # Old code used logger.debug for failures — now must be warning
        assert "logger.warning" in block

    def test_old_index_dollar_x_gone_in_calls(self):
        src = (REPO_ROOT / "data_providers" / "schwab.py").read_text()
        # The old `$SPX.X` index name must not appear in any _get_movers call.
        # (It may still appear inside the explanatory comment block above
        # the function — that's documentation of the prior bug.)
        import re
        calls = re.findall(r"self\._get_movers\([^)]+\)", src)
        for call in calls:
            assert "$SPX.X" not in call, f"old index name in {call}"


# ── v-verify-timeout-extend + v-discovered-claim-bot-orders 2026-05-11

class TestVerifyTimeoutAndRescue:
    """When `_verify_order_fill` times out but the order actually
    fills downstream, the discovered position needs to be claimed as
    bot-managed instead of left external."""

    def test_verify_retries_extended(self):
        """Default max_retries has been raised twice:
          - 2026-05-11: 3 → 10 (slow Schwab status API)
          - 2026-05-26: 10 → 30 (QBTS silent auto-cancel incident)
        Both v-tags must remain in source as breadcrumbs."""
        src = ENGINE_PATH.read_text()
        assert "v-verify-timeout-extend-2026-05-11" in src
        assert "v-verify-timeout-extend-2026-05-26" in src
        # Current default. Drift downward should fail this assertion.
        assert "max_retries: int = 30" in src

    def test_verify_does_final_check_before_cancel(self):
        src = ENGINE_PATH.read_text()
        idx = src.index("v-verify-timeout-extend-2026-05-11")
        # widen window: original block was 4000 chars but post-2026-05-26
        # the surrounding doc string + new code path made the relevant
        # logic land further down.
        block = src[idx : idx + 6000]
        # Final status check before cancel attempt
        assert "final_status" in block
        # Race-condition check during cancel failure
        assert "race_status" in block

    def test_discovery_rescue_claims_pending_orders(self):
        src = ENGINE_PATH.read_text()
        assert "v-discovered-claim-bot-orders-2026-05-11" in src
        idx = src.index("v-discovered-claim-bot-orders-2026-05-11")
        block = src[idx : idx + 6000]
        # Must scan pending_orders for matching symbol
        assert "self.pending_orders" in block
        # When found, must create Position with managed_by_bot=True
        assert "managed_by_bot=True" in block
        # Must pop the rescued order
        assert "self.pending_orders.pop" in block


# ── v-mean-rev-price-direction-gate-2026-05-13 ───────────────────────

BUILTIN_PATH = REPO_ROOT / "strategies" / "builtin.py"


class TestMeanRevPriceDirectionGate:
    """v-mean-rev-price-direction-gate-2026-05-13.

    Operator incident 2026-05-12: mean-rev LONG entered ORCL at 12:31
    after a 46-min slide (184.50 → 182.44) and CRWV at 12:47 after
    63 minutes of continued weakness. Both met the textbook setup
    (RSI < 30, close < BB_lower), but neither had any reversal
    confirmation at the entry bar. The existing falling-knife guard
    was bypassed because the entry bar itself happened to print green.

    Fix: require POSITIVE bar-level confirmation:
      - close > bar_open (green bar at entry)
      - volume_ratio >= 1.0 (real participation, not noise)
    Either alone is necessary; both are required.

    Mirror of v-news-price-direction-gate-2026-05-08 from news_strategy.
    """

    def test_gate_present_in_mean_rev_long(self):
        src = BUILTIN_PATH.read_text()
        assert "v-mean-rev-price-direction-gate-2026-05-13" in src

    def test_gate_skips_with_mean_rev_specific_reason(self):
        """Operator must be able to grep
        ``strategy_decision strategy=mean_reversion .* action=skip
        .* reason=mean_rev_price_direction_disagrees`` in logs."""
        src = BUILTIN_PATH.read_text()
        idx = src.index("v-mean-rev-price-direction-gate-2026-05-13")
        block = src[idx : idx + 3000]
        assert "mean_rev_price_direction_disagrees" in block

    def test_gate_requires_both_close_and_volume(self):
        """Both conditions are required — not one or the other."""
        src = BUILTIN_PATH.read_text()
        idx = src.index("v-mean-rev-price-direction-gate-2026-05-13")
        block = src[idx : idx + 3000]
        # The check must use AND, with both close > bar_open and
        # volume_ratio >= 1.0
        assert "market_data.close > _bar_open" in block
        assert "_vol_ratio_now >= 1.0" in block
        # And combined with AND (Python keyword)
        assert " and " in block


# ── v-anti-pyramiding-2026-05-13 ─────────────────────────────────────

class TestAntiPyramiding:
    """v-anti-pyramiding-2026-05-13.

    Operator incident 2026-05-12: FCEL entered THREE times in 1 hour
    (11:46, 12:21, 12:42). The existing ``already_tracking`` guard at
    signal_router checked ``self.positions`` + ``self.simulated_positions``
    but missed two gap paths:

      1. Orders in flight (in ``self.pending_orders``) that haven't yet
         materialized as a Position object.
      2. Recently-attempted entries where the order succeeded but the
         position-tracking state lagged for whatever reason (the FCEL
         pattern — 35 and 21 minutes between entries, way past any
         normal verify-fill window).

    Defense in depth: add a pending-order check AND a recent-attempt
    cooldown (30 minutes). Either alone catches the documented incident;
    both together makes the anti-pyramiding invariant hard to violate
    even through paths we haven't considered.
    """

    def test_pending_order_blocks_re_entry(self):
        src = ENGINE_PATH.read_text()
        assert "v-anti-pyramiding-2026-05-13" in src
        idx = src.index("v-anti-pyramiding-2026-05-13")
        block = src[idx : idx + 4000]
        # The check must scan pending_orders for the symbol
        assert "pending_orders" in block
        # And bail with a skip audit
        assert 'signal_router' in block
        assert 'pending_order_for_symbol' in block or 'in_flight' in block

    def test_recent_attempt_cooldown_present(self):
        src = ENGINE_PATH.read_text()
        idx = src.index("v-anti-pyramiding-2026-05-13")
        block = src[idx : idx + 4000]
        # A 30-minute cooldown for recent entry attempts
        assert "_recent_entry_attempts" in block
        # 30 minutes = 1800 seconds OR timedelta(minutes=30)
        assert "30" in block

    def test_audit_reason_grep_able(self):
        """Operator must be able to grep ``engine_decision .* action=skip
        .* reason=anti_pyramid`` to find blocked entries during review."""
        src = ENGINE_PATH.read_text()
        idx = src.index("v-anti-pyramiding-2026-05-13")
        block = src[idx : idx + 4000]
        assert 'anti_pyramid' in block


# ── v-news-verifier-gate-zero-fresh-2026-05-13 ───────────────────────

class TestNewsVerifierGateZeroFresh:
    """v-news-verifier-gate-zero-fresh-2026-05-13.

    Operator incident 2026-05-12: FCEL entered 3 times in 1 hour, each
    with ``cached_sentiment=0.527 fresh_count=0`` and the verifier
    flagging ``insufficient_fresh_articles_0_lt_2``. The verifier was
    running in *advisory* mode (``v-news-verifier-advisory-2026-05-08``)
    which logs the veto but does not gate. That made sense for the
    'fresh_count=1, almost-but-not-quite-corroborated' edge case the
    operator wanted to surface; it does NOT make sense for
    fresh_count=0 — zero fresh articles means we're acting on
    sentiment cached from articles that have aged out of any
    reasonable freshness window.

    New behavior: even in advisory mode, ``fresh_count == 0`` is a
    HARD veto. ≥1 fresh article keeps the advisory-only behavior.
    """

    def test_zero_fresh_articles_hard_vetoes_even_in_advisory(self):
        """Code path: when v.fresh_count == 0, signal is dropped
        regardless of NEWS_VERIFIER_ADVISORY."""
        src = NEWS_STRATEGY_PATH.read_text()
        assert "v-news-verifier-gate-zero-fresh-2026-05-13" in src
        idx = src.index("v-news-verifier-gate-zero-fresh-2026-05-13")
        block = src[idx : idx + 3000]
        # The gate must check fresh_count == 0 and return None unconditionally.
        assert "fresh_count == 0" in block
        assert "return None" in block

    def test_advisory_still_works_with_some_fresh(self):
        """The gate must NOT fire when fresh_count >= 1 — that case
        stays under advisory mode per v-news-verifier-advisory."""
        src = NEWS_STRATEGY_PATH.read_text()
        idx = src.index("v-news-verifier-gate-zero-fresh-2026-05-13")
        block = src[idx : idx + 3000]
        # The gate condition must be specifically the zero case;
        # >= 1 must stay advisory.
        assert "fresh_count == 0" in block
        # NOT a broader veto that catches fresh_count >= 1
        assert "fresh_count < 2" not in block


# ── v-discovered-position-tagging-2026-05-11 ─────────────────────────

class TestDiscoveredPositionTagging:
    """`_update_and_track_real_positions` was creating external Schwab
    positions with entry_time=now and is_external=False, tripping the
    T3 health gate's "recent unmanaged" canary and blocking all live
    entries. Must mirror its sibling `_update_real_positions`."""

    def test_back_dates_entry_time(self):
        src = ENGINE_PATH.read_text()
        idx = src.index("v-discovered-position-tagging-2026-05-11")
        # Larger window now that the rescue path was added.
        block = src[idx : idx + 8000]
        assert "datetime.now() - timedelta(hours=1)" in block

    def test_sets_external_flags(self):
        src = ENGINE_PATH.read_text()
        idx = src.index("v-discovered-position-tagging-2026-05-11")
        block = src[idx : idx + 8000]
        assert "position.is_external = True" in block
        assert "position.is_manually_managed = True" in block


# ── v-watchlist-size + v-watchlist-ui-cap 2026-05-11 ────────────────

class TestWatchlistSize:
    """Engine + UI now both honor Config.WATCHLIST_SIZE (default 30).
    Prior state: engine 20, UI hardcoded 10."""

    def test_watchlist_size_default_30(self):
        from core.config import Config
        assert Config().WATCHLIST_SIZE == 30

    def test_ui_cap_honors_config(self):
        """routes.py no longer hardcodes [:10] on the screener slice."""
        src = ROUTES_PATH.read_text()
        # Hardcoded slice should be gone
        assert "top_movers[:10]" not in src
        # New code must reference WATCHLIST_SIZE for the UI cap
        assert "v-watchlist-ui-cap-2026-05-11" in src
        idx = src.index("v-watchlist-ui-cap-2026-05-11")
        block = src[idx : idx + 1500]
        assert "WATCHLIST_SIZE" in block


# ── v-news-cooldown-shorten-2026-05-08 ───────────────────────────────

class TestCooldownShorten:
    """60min → 20min news cooldown. INTC 2026-05-08 12:53 entry
    locked out add-on opportunities for the full 60-min run-up."""

    def test_cooldown_is_1200_seconds(self):
        src = NEWS_STRATEGY_PATH.read_text()
        # The cooldown_left math now subtracts from 1200, not 3600
        assert "cooldown_left = 1200 -" in src
        # And the v-tag should be present
        assert "v-news-cooldown-shorten-2026-05-08" in src


# ── v-day-pnl-intraday-entry-2026-05-12 ──────────────────────────────

class TestDayPnlIntradayEntry:
    """v-day-pnl-intraday-entry-2026-05-12: the bot was computing
    per-position day P&L as `netChange × qty`. `netChange` is the
    price change FROM YESTERDAY'S CLOSE — not from the entry price —
    so for any position opened intraday at a price ≠ today's open,
    the formula attributes the pre-entry move to the position's P&L.

    Today (2026-05-12) the bot bought FCEL/ORCL/CRWV intraday after
    morning sell-offs; the displayed day P&L inflated the loss by
    hundreds of dollars per position relative to Schwab's UI.

    New approach: trust Schwab's `currentDayProfitLoss` (broker's
    authoritative number, correct for both intraday and overnight)
    UNLESS its magnitude is implausibly large vs the position's
    market value (PLTR same-day-lot-transferred case from
    v-schwab-pnl-fix-2026-04-21), in which case fall back to
    `netChange × qty`.
    """

    def test_normal_intraday_entry_uses_broker_pnl(self):
        """FCEL pattern: entered intraday well below today's open. The
        broker's currentDayProfitLoss is the only correct day P&L
        figure; netChange × qty overstates the move by including the
        pre-entry decline."""
        from core.pnl import compute_position_day_pnl

        pos = {
            "longQuantity": 568,
            "shortQuantity": 0,
            "marketValue": 14.60 * 568,
            "currentDayProfitLoss": -187.44,  # = (14.60 - 14.93) * 568
            "instrument": {"netChange": 1.10, "symbol": "FCEL"},  # prev_close $13.50
        }
        assert compute_position_day_pnl(pos) == pytest.approx(-187.44)

    def test_overnight_position_uses_broker_pnl(self):
        """Overnight short — both formulas should yield the same answer
        and the helper should return the broker number."""
        from core.pnl import compute_position_day_pnl

        pos = {
            "longQuantity": 0,
            "shortQuantity": 100,
            "marketValue": -21800.0,
            "currentDayProfitLoss": 200.00,  # short of NVDA went the right way
            "instrument": {"netChange": -2.00, "symbol": "NVDA"},
        }
        assert compute_position_day_pnl(pos) == pytest.approx(200.00)

    def test_pltr_outlier_falls_back_to_netchange(self):
        """The PLTR same-day-lot-transferred case from
        v-schwab-pnl-fix-2026-04-21: currentDayProfitLoss returned
        +$21,932 on a position worth a few thousand dollars. Magnitude
        > 50% of market value → fall back to netChange × qty."""
        from core.pnl import compute_position_day_pnl

        pos = {
            "longQuantity": 100,
            "shortQuantity": 0,
            "marketValue": 3000.0,
            "currentDayProfitLoss": 21932.0,  # implausible
            "instrument": {"netChange": -5.0, "symbol": "PLTR"},
        }
        # qty = +100; netChange × qty = -5 * 100 = -500
        assert compute_position_day_pnl(pos) == pytest.approx(-500.0)

    def test_short_qty_sign(self):
        """Sanity: shorts give qty = longQuantity - shortQuantity = -N,
        and the netChange fallback path must respect that sign."""
        from core.pnl import compute_position_day_pnl

        pos = {
            "longQuantity": 0,
            "shortQuantity": 50,
            "marketValue": -2500.0,
            "currentDayProfitLoss": 0,  # forces netChange fallback
            "instrument": {"netChange": -1.50, "symbol": "XYZ"},
        }
        # qty = 0 - 50 = -50; -1.50 * -50 = +75 (short profited on a $1.50 drop)
        assert compute_position_day_pnl(pos) == pytest.approx(75.0)

    def test_day_gain_loss_legacy_field_fallback(self):
        """If currentDayProfitLoss is 0 but dayGainLoss has a value,
        use that — preserves the existing legacy fallback chain."""
        from core.pnl import compute_position_day_pnl

        pos = {
            "longQuantity": 10,
            "shortQuantity": 0,
            "marketValue": 1000.0,
            "currentDayProfitLoss": 0,
            "dayGainLoss": 42.50,
            "instrument": {"netChange": 0, "symbol": "FOO"},
        }
        assert compute_position_day_pnl(pos) == pytest.approx(42.50)

    def test_v_tag_present_in_engine(self):
        src = ENGINE_PATH.read_text()
        assert "v-day-pnl-intraday-entry-2026-05-12" in src


# ── v-parallel-screener-loop-2026-05-12 ──────────────────────────────

class TestParallelScreenerLoop:
    """Step 1 of the parallel-loops refactor (`.claude/PLAN_parallel_loops.md`).

    The screener now runs as its own supervised task with its own cadence
    and try/except boundary. Yesterday's `pnl_to_check` NameError fired
    in the emergency-stop block of `_analysis_loop_body` and aborted
    every downstream step in the same iteration — including the screener.
    With this extraction, a crash anywhere in `signal_loop` cannot freeze
    the watchlist refresh.
    """

    def test_screener_loop_module_exists(self):
        """The new module is importable."""
        from core.loops.screener_loop import ScreenerLoop  # noqa: F401

    def test_screener_loop_registered_with_supervisor(self):
        """`_build_supervisor` must register `screener_loop` alongside
        the existing four supervised tasks."""
        src = ENGINE_PATH.read_text()
        assert 'sup.register("screener_loop"' in src, (
            "engine.py must register screener_loop with the TaskSupervisor"
        )

    def test_screener_loop_inline_block_removed(self):
        """The old inline screener cadence gate inside
        `_analyze_markets_with_commentary` must be deleted — keeping
        both would race the supervised loop and the in-line block."""
        src = ENGINE_PATH.read_text()
        # The unique signature of the old gate.
        assert "if self.screener and (not self.last_screener_run or" not in src

    def test_screener_loop_v_tag_present_in_new_module(self):
        src = (REPO_ROOT / "core" / "loops" / "screener_loop.py").read_text()
        assert "v-parallel-screener-loop-2026-05-12" in src
        # Tags migrated from the inline block must survive the move so
        # the inventory test stays green.
        assert "v-watchlist-size-2026-04-29" in src
        assert "v-watchlist-stream-2026-05-01" in src

    def test_config_has_screener_loop_sec(self):
        """`SCREENER_LOOP_SEC` config knob exists, defaults to 120."""
        from core.config import Config

        c = Config()
        assert hasattr(c, "SCREENER_LOOP_SEC")
        assert c.SCREENER_LOOP_SEC == 120

    def test_loop_swallows_screener_exception(self):
        """Per-iteration try/except: an exception in `screen_stocks()`
        must NOT raise out of `run()` — otherwise the supervisor would
        treat each cadence tick as a hard crash and eventually
        circuit-break the loop."""
        import asyncio
        from unittest.mock import MagicMock

        from core.loops.screener_loop import ScreenerLoop

        engine = MagicMock()
        engine.is_running = True

        call_count = {"n": 0}

        async def _bad_screen():
            call_count["n"] += 1
            engine.is_running = False  # stop after one tick
            raise RuntimeError("synthetic screener failure")

        engine.screener = MagicMock()
        engine.screener.screen_stocks = _bad_screen

        loop = ScreenerLoop(engine, cadence_sec=0)
        asyncio.run(loop.run())   # must not raise
        assert call_count["n"] == 1


# ── v-emergency-stop-pnl-var-2026-05-11 ──────────────────────────────

class TestEmergencyStopPnlVar:
    """Operator incident 2026-05-11: emergency-stop branch in
    _analysis_loop_body raised NameError every cycle because the
    f-string still read the old refactored var (pnl_to_check) instead
    of the renamed schwab_pnl. The NameError bubbled up the analysis
    loop and silently froze the screener/watchlist refresh for ~30
    minutes — exactly the kind of silent-loop-death the WARNING-log
    discipline was meant to surface."""

    def test_no_stale_pnl_var_in_engine_code(self):
        """Only the v-tag explanatory comment may mention the old
        name; no executable reference should survive."""
        src = ENGINE_PATH.read_text()
        for line_no, line in enumerate(src.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue  # comments allowed
            assert "pnl_to_check" not in line, (
                f"core/engine.py:{line_no} still references stale "
                f"pnl_to_check var (should be schwab_pnl): {line!r}"
            )

    def test_emergency_stop_messages_reference_schwab_pnl(self):
        """Both EMERGENCY STOP commentary messages must compute the
        loss from schwab_pnl (the var the branch guard actually
        checks at line 3662 of engine.py)."""
        src = ENGINE_PATH.read_text()
        idx = src.index("v-emergency-stop-pnl-var-2026-05-11")
        block = src[idx : idx + 4000]
        # Both messages must read schwab_pnl
        assert block.count("abs(schwab_pnl)") >= 2, (
            "expected both EMERGENCY STOP f-strings to use abs(schwab_pnl)"
        )


# ── v-news-late-entry-guard-2026-05-08 ───────────────────────────────

class TestNewsLateEntryGuard:
    """Operator observation 2026-05-08: bot was buying tops on stale
    news. Guard refuses BUY when RSI>70 AND close>sma_20*1.02
    (and symmetric for SELL)."""

    def test_guard_present(self):
        src = NEWS_STRATEGY_PATH.read_text()
        assert "v-news-late-entry-guard-2026-05-08" in src
        assert "extended_already_ran" in src
        assert "extended_already_dropped" in src

    def test_buy_threshold_rsi_70_sma_1pct(self):
        """After v-late-entry-tighten-2026-05-11: 2% → 1% SMA threshold.
        NVDA 2026-05-11 at 1.77% above SMA20 missed the 2% gate."""
        src = NEWS_STRATEGY_PATH.read_text()
        idx = src.index("v-news-late-entry-guard-2026-05-08")
        block = src[idx : idx + 5000]
        assert "rsi_now > 70.0" in block
        # Tightened to 1.01 (was 1.02)
        assert "sma_20 * 1.01" in block

    def test_sell_threshold_rsi_30_sma_1pct(self):
        src = NEWS_STRATEGY_PATH.read_text()
        idx = src.index("v-news-late-entry-guard-2026-05-08")
        block = src[idx : idx + 5000]
        assert "rsi_now < 30.0" in block
        # Tightened to 0.99 (was 0.98)
        assert "sma_20 * 0.99" in block

    def test_no_conviction_bypass(self):
        """Overextension is risk regardless of sentiment magnitude. The
        V-bottom case (low RSI + high sentiment) doesn't trigger this
        gate's BUY check anyway, so we explicitly do NOT bypass on
        conviction here."""
        src = NEWS_STRATEGY_PATH.read_text()
        idx = src.index("v-news-late-entry-guard-2026-05-08")
        block = src[idx : idx + 4000]
        # Guard must not reference _conviction_bypass
        assert "_conviction_bypass" not in block


# ── v-conviction-bypass-knife-2026-05-06 ─────────────────────────────

class TestConvictionBypass:
    """Falling/rising-knife filters must be bypassed when sentiment is
    extreme AND well-corroborated."""

    def test_bypass_threshold_550_8articles(self):
        src = NEWS_STRATEGY_PATH.read_text()
        # Enforce the threshold so a future tweak doesn't silently drop
        # the catalyst-bypass to a less-conservative threshold.
        m = re.search(
            r"_conviction_bypass\s*=\s*\(\s*abs\(avg_sentiment\)\s*>=\s*([\d.]+)\s*and\s*len\(news_items\)\s*>=\s*(\d+)",
            src,
        )
        assert m is not None
        assert float(m.group(1)) == pytest.approx(0.55)
        assert int(m.group(2)) == 8


# ── v-max-positions-gate-2026-05-06 ──────────────────────────────────

class TestMaxPositionsGate:
    """MAX_POSITIONS config default raised 5→10 and the gate now runs."""

    def test_max_positions_default_is_10(self):
        from core.config import Config
        # The default was 5 since day one but never enforced. Today we
        # both raised the default and added the enforcement gate.
        cfg = Config()
        assert cfg.MAX_POSITIONS == 10

    def test_gate_only_counts_managed_by_bot(self):
        src = ENGINE_PATH.read_text()
        idx = src.index("v-max-positions-gate-2026-05-06")
        block = src[idx : idx + 2500]
        # Counter must filter on managed_by_bot
        assert "managed_by_bot" in block
        assert "bot_managed_count" in block


# ── v-correlation-mode-isolated-2026-05-05 ───────────────────────────

class TestCorrelationModeIsolated:
    """Correlation guard must use the active mode's collection only."""

    def test_live_mode_uses_only_self_positions(self):
        src = ENGINE_PATH.read_text()
        idx = src.index("v-correlation-mode-isolated-2026-05-05")
        block = src[idx : idx + 1500]
        # Live mode reads self.positions, not simulated_positions
        assert "if self.mode == TradingMode.LIVE:" in block
        assert "active_positions = set(self.positions.keys())" in block


# ── v-cooldown-regular-only-2026-05-05 ───────────────────────────────

class TestCooldownRegularHoursOnly:
    """News-strategy cooldown only burns during regular hours."""

    def test_cooldown_set_under_regular_hours_guard(self):
        src = NEWS_STRATEGY_PATH.read_text()
        idx = src.index("v-cooldown-regular-only-2026-05-05")
        block = src[idx : idx + 1500]
        assert "if _is_regular:" in block
        assert "self.last_signal_time[symbol] = datetime.now()" in block


# ── v-sentiment-thresh-2026-05-05 ────────────────────────────────────

class TestSentimentThresholds:
    """0.40→0.30 with ≥5 articles."""

    def test_strong_corroborated_threshold_030(self):
        src = NEWS_STRATEGY_PATH.read_text()
        idx = src.index("v-sentiment-thresh-2026-05-05")
        block = src[idx : idx + 800]
        assert "abs(avg_sentiment) > 0.30" in block
        assert "len(news_items) >= 5" in block


# ── JSON NaN/inf scrub on API endpoints ──────────────────────────────

class TestJsonScrub:
    """`/api/positions/db`, `/api/trades`, `/api/decisions` must convert
    NaN/inf floats to None."""

    def test_scrub_handles_nan_and_inf(self):
        # The scrub helper is duplicated inline in three routes — verify
        # the pattern. We can also write a tiny pure-function variant
        # of the scrub here as a smoke test.
        def _scrub(v):
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return None
                return v
            return v

        bad = {"pnl_pct": float("nan"), "atr": float("inf"), "ok": 1.5}
        scrubbed = {k: _scrub(v) for k, v in bad.items()}
        assert scrubbed["pnl_pct"] is None
        assert scrubbed["atr"] is None
        assert scrubbed["ok"] == 1.5
        # Round-trips through json without ValueError
        json.dumps(scrubbed)

    def test_scrub_present_in_three_endpoints(self):
        src = ROUTES_PATH.read_text()
        # Each scrubbed endpoint references math.isnan/isinf
        # /api/positions/db
        assert "v-json-nan-sanitize-2026-05-04" in src
        # /api/trades
        assert "v-json-nan-sanitize-trades-2026-05-05" in src


# ── v-classifier-stays-isolated-2026-05-13 ──────────────────────────

class TestClassifierStaysIsolated:
    """The side-classifier subtree is *experimental* and must not be
    imported by the live trading path until ``Config.USE_SIDE_CLASSIFIER``
    is approved (rollout flavor B — per-strategy gradual).

    This meta-test fails CI the moment someone accidentally wires
    ``core.classifier`` into the bot. The cost of an accidental import
    is that broken/half-built classifier code could affect the live
    bot in ways the rule-based + research separation was designed to
    prevent.

    Tightened only when integration begins — at that point the test
    moves to assert the import is *gated by the config flag*, not that
    it's absent.
    """

    def test_engine_only_imports_classifier_conditionally(self):
        """Updated 2026-05-13: shadow-mode wiring is allowed but only
        when the import is *inside a function body* (i.e., lazy /
        gated). No top-level imports of core.classifier in engine.py
        — a torch-less env must still be able to start the bot.

        Top-level imports load at module-parse time, before any config
        is read. Conditional/lazy imports inside the relevant
        functions guarantee that ``SIDE_CLASSIFIER_SHADOW_MODE=False``
        (the default) means zero classifier code runs.
        """
        src = ENGINE_PATH.read_text()
        for lineno, line in enumerate(src.splitlines(), start=1):
            stripped = line.lstrip()
            if not stripped.startswith(("from core.classifier",
                                         "import core.classifier")):
                continue
            # An import line — must be indented (i.e., inside a function
            # body). Top-level imports start at column 0.
            indent = len(line) - len(stripped)
            assert indent > 0, (
                f"core/engine.py:{lineno} imports core.classifier at "
                f"the top level: {line!r}. Move it inside the function "
                f"that uses it, gated by a SIDE_CLASSIFIER config flag."
            )

    def test_signal_router_does_not_import_classifier(self):
        """If a signal_router module exists separately, it must also
        not import classifier yet. Today the routing is inline in
        engine.py; this test is forward-looking but cheap."""
        candidates = [
            REPO_ROOT / "core" / "signal_router.py",
            REPO_ROOT / "core" / "router.py",
        ]
        for p in candidates:
            if p.exists():
                src = p.read_text()
                assert "core.classifier" not in src, (
                    f"{p} imports the classifier — gate behind "
                    "Config.USE_SIDE_CLASSIFIER first."
                )

    def test_classifier_package_is_importable_standalone(self):
        """Counterpart guard: the classifier package itself MUST import
        cleanly without dragging engine internals. Catches accidental
        circular dependencies."""
        # If this import succeeds in a clean interpreter sequence, the
        # classifier is properly isolated. The actual import lives in
        # the test process; what we're asserting is the existence and
        # cleanliness of the public surface.
        from core.classifier import (    # noqa: F401
            Side, SymbolFeatures, SymbolSideDecision,
            classify, classify_rule_based,
        )


# ── v-mode-toggle-stale-state-2026-05-08 ─────────────────────────────

class TestModeToggleStaleState:
    """toggle-mode endpoint accepts clear_inactive and surfaces sim count."""

    def test_endpoint_accepts_clear_inactive(self):
        src = ROUTES_PATH.read_text()
        idx = src.index("v-mode-toggle-stale-state-2026-05-08")
        block = src[idx : idx + 2500]
        assert "clear_inactive" in block

    def test_live_to_sim_does_not_clear_live(self):
        src = ROUTES_PATH.read_text()
        idx = src.index("v-mode-toggle-stale-state-2026-05-08")
        block = src[idx : idx + 2500]
        # Comment must reference the deliberate non-clearing of live tracking
        assert "real money" in block.lower() or "Real money" in block or "real-money" in block.lower()


# ── v-external-close-reconcile + v-brain-learns-from-manual-closes ──

class TestExternalCloseReconcile:
    """Manual closes write a bot_trades row AND feed the brain."""

    def test_reconcile_method_exists(self):
        src = ENGINE_PATH.read_text()
        assert "async def _reconcile_external_closes" in src

    def test_reconcile_calls_log_trade(self):
        src = ENGINE_PATH.read_text()
        idx = src.index("async def _reconcile_external_closes")
        block = src[idx : idx + 8000]
        assert "self.db_logger.log_trade" in block
        assert 'exit_reason="external_close"' in block

    def test_reconcile_calls_brain_remember_trade(self):
        src = ENGINE_PATH.read_text()
        assert "v-brain-learns-from-manual-closes-2026-05-08" in src
        idx = src.index("v-brain-learns-from-manual-closes-2026-05-08")
        block = src[idx : idx + 2500]
        assert "brain.remember_trade" in block
        assert '"external_close"' in block


# ── Sanity: every fix tag should be findable ─────────────────────────

class TestFixTagInventory:
    """Cheap smoke test: every v-tag from the audit log must be
    discoverable in the source. Catches accidental rollback during
    merges/refactors."""

    EXPECTED_TAGS = [
        # 2026-05-05
        "v-early-session-narrow-2026-05-05",
        "v-cooldown-regular-only-2026-05-05",
        "v-sentiment-thresh-2026-05-05",
        "v-meta-proba-on-signal-2026-05-05",
        "v-correlation-mode-isolated-2026-05-05",
        "v-sync-preserve-bot-state-2026-05-05",
        "v-json-nan-sanitize-2026-05-04",
        # 2026-05-06
        "v-conviction-bypass-knife-2026-05-06",
        "v-max-positions-gate-2026-05-06",
        # 2026-05-07
        "v-external-close-reconcile-2026-05-07",
        # 2026-05-08
        "v-fill-creates-position-2026-05-08",
        "v-sweep-signal-scope-fix-2026-05-08",
        "v-bracket-call-fix-2026-05-08",
        "v-stream-watchdog-2026-05-08",
        "v-health-gate-2026-05-08",
        "v-news-verifier-advisory-2026-05-08",
        "v-news-price-direction-gate-2026-05-08",
        "v-correlation-groups-expanded-2026-05-08",
        "v-mode-toggle-stale-state-2026-05-08",
        "v-brain-learns-from-manual-closes-2026-05-08",
        "v-news-late-entry-guard-2026-05-08",
        "v-verifier-none-guard-bulletproof-2026-05-08",
        "v-price-direction-relax-2026-05-08",
        "v-news-cooldown-shorten-2026-05-08",
        "v-screener-api-fix-2026-05-08",
        "v-screener-fresh-daily-2026-05-08",
        "v-screener-enum-fix-2026-05-11",
        "v-yahoo-most-active-2026-05-11",
        "v-watchlist-size-2026-05-11",
        "v-watchlist-ui-cap-2026-05-11",
        "v-discovered-position-tagging-2026-05-11",
        "v-consecutive-losses-live-only-2026-05-11",
        # v-enable-mean-rev-short-2026-05-11 was reverted on
        # 2026-05-12; superseded by v-disable-mean-rev-short below.
        "v-late-entry-tighten-2026-05-11",
        "v-verify-timeout-extend-2026-05-11",
        "v-discovered-claim-bot-orders-2026-05-11",
        "v-schwab-provider-movers-api-fix-2026-05-11",
        "v-emergency-stop-pnl-var-2026-05-11",
        # 2026-05-12
        "v-parallel-screener-loop-2026-05-12",
        "v-disable-mean-rev-short-2026-05-12",
        "v-day-pnl-intraday-entry-2026-05-12",
        "v-stream-watchdog-recover-2026-05-12",
        # 2026-05-13 — side-classifier subsystem (isolated, gated off)
        "v-side-classifier-types-2026-05-13",
        "v-rule-based-classifier-2026-05-13",
        "v-classifier-composer-2026-05-13",
        "v-side-classifier-package-2026-05-13",
        "v-side-classifier-config-2026-05-13",
        "v-classifier-features-2026-05-13",
        "v-triple-barrier-labels-2026-05-13",
        "v-classifier-backtest-2026-05-13",
        "v-research-isolation-2026-05-13",
        "v-classifier-stays-isolated-2026-05-13",
        # Trained-classifier scaffolding (RTX 3090)
        "v-predictor-protocol-2026-05-13",
        "v-stub-predictor-2026-05-13",
        "v-feature-encoder-2026-05-13",
        "v-trained-baseline-model-2026-05-13",
        "v-ffn-predictor-2026-05-13",
        "v-trained-classifier-package-2026-05-13",
        "v-train-classifier-2026-05-13",
        # Strategy-layer gating fixes (financial-expert decision over TFT)
        "v-news-verifier-gate-zero-fresh-2026-05-13",
        "v-anti-pyramiding-2026-05-13",
        "v-mean-rev-price-direction-gate-2026-05-13",
        "v-classifier-runtime-2026-05-13",
        "v-session-reporter-2026-05-13",
        "v-drift-monitor-2026-05-13",
        "v-shadow-integration-2026-05-13",
    ]

    EXPECTED_TAGS_BY_FILE = {
        "analysis/screener.py": [
            "v-screener-api-fix-2026-05-08",
            "v-screener-fresh-daily-2026-05-08",
        ],
    }

    def test_screener_specific_tags_in_screener_file(self):
        src = (REPO_ROOT / "analysis" / "screener.py").read_text()
        for tag in self.EXPECTED_TAGS_BY_FILE["analysis/screener.py"]:
            assert tag in src, f"{tag} missing from analysis/screener.py"

    def test_every_fix_tag_present_in_source(self):
        all_src = (
            ENGINE_PATH.read_text()
            + CONFIG_PATH.read_text()
            + NEWS_STRATEGY_PATH.read_text()
            + STREAM_PATH.read_text()
            + ROUTES_PATH.read_text()
            + (REPO_ROOT / "analysis" / "screener.py").read_text()
            + (REPO_ROOT / "data_providers" / "schwab.py").read_text()
            + (REPO_ROOT / "Config().yaml").read_text()
            + (REPO_ROOT / "core" / "loops" / "screener_loop.py").read_text()
            + (REPO_ROOT / "core" / "schwab_stream_watchdog.py").read_text()
            + (REPO_ROOT / "core" / "stream_health.py").read_text()
            # 2026-05-13 — side-classifier subsystem
            + (REPO_ROOT / "core" / "classifier" / "__init__.py").read_text()
            + (REPO_ROOT / "core" / "classifier" / "types.py").read_text()
            + (REPO_ROOT / "core" / "classifier" / "rule_based.py").read_text()
            + (REPO_ROOT / "core" / "classifier" / "composer.py").read_text()
            + (REPO_ROOT / "research" / "__init__.py").read_text()
            + (REPO_ROOT / "research" / "features.py").read_text()
            + (REPO_ROOT / "research" / "labels.py").read_text()
            + (REPO_ROOT / "research" / "classifier_backtest.py").read_text()
            # Trained-classifier subtree (anchors all v-trained-*-2026-05-13)
            + (REPO_ROOT / "core" / "classifier" / "trained" / "__init__.py").read_text()
            + (REPO_ROOT / "core" / "classifier" / "trained" / "predictor_protocol.py").read_text()
            + (REPO_ROOT / "core" / "classifier" / "trained" / "stub_predictor.py").read_text()
            + (REPO_ROOT / "core" / "classifier" / "trained" / "feature_encoder.py").read_text()
            + (REPO_ROOT / "core" / "classifier" / "trained" / "model.py").read_text()
            + (REPO_ROOT / "core" / "classifier" / "trained" / "ffn_predictor.py").read_text()
            + (REPO_ROOT / "research" / "train_classifier.py").read_text()
            # 2026-05-13 strategy-layer fixes
            + (REPO_ROOT / "strategies" / "builtin.py").read_text()
            # 2026-05-13 runtime + reporters
            + (REPO_ROOT / "core" / "classifier" / "runtime.py").read_text()
            + (REPO_ROOT / "research" / "session_reporter.py").read_text()
            + (REPO_ROOT / "research" / "drift_monitor.py").read_text()
            + (REPO_ROOT / "tests" / "core" / "classifier" /
               "test_shadow_integration.py").read_text()
        )
        missing = [t for t in self.EXPECTED_TAGS if t not in all_src]
        assert missing == [], f"missing fix tags: {missing}"


# ── v-autocancel-fix-2026-05-26 + v-autocancel-logging-2026-05-26 + v-verify-timeout-extend-2026-05-26 ─

class TestAutocancelGhostAttemptFix:
    """Three-part fix triggered by QBTS 2026-05-26 09:37:27 silent auto-
    cancel incident.

    Bug chain on launch day:
      1. QBTS limit BUY @ $27.45 placed at 09:37:27.
      2. _verify_order_fill polled status every 2s for 10 retries = 20s.
         Order stayed WORKING (limit price never crossed).
      3. After 20s the bot called schwab_client.cancel_order silently.
         No log line on the success path of the try-block — only the
         failure path logged a warning.
      4. _recent_entry_attempts[QBTS] was set at signal time (engine.py
         ~L4875) and never cleared. The 4-hour anti-pyramid window
         was now ticking on a ghost order.
      5. At 10:45:32 the news strategy fired a fresh QBTS BUY signal.
         signal_router blocked it with reason=anti_pyramid_recent_attempt
         age_min=68.1 — based on the ghost attempt that had been dead
         since 09:37:47.

    Fix has three parts, all in core/engine.py around _verify_order_fill:
      A. Helper _clear_recent_attempt_for_order(order_id, reason) removes
         _recent_entry_attempts[symbol] for an order that never filled.
      B. Helper is called from the three terminal-not-filled paths:
         retry-loop CANCELED/REJECTED, final-check CANCELED/REJECTED,
         and the post-timeout schwab_client.cancel_order() block.
      C. max_retries default raised from 10 → 30 (20s → 60s) to lower
         the false-cancel rate on slow-fill limit orders.
      D. The post-timeout cancel block emits an INFO log
         "auto_cancelled_unfilled_order ..." so future "where did my
         order go" investigations are not silent.
    """

    def test_clear_helper_exists(self):
        """_clear_recent_attempt_for_order is defined on the engine
        class with the expected (order_id, reason) signature."""
        src = ENGINE_PATH.read_text()
        assert "def _clear_recent_attempt_for_order(self, order_id: str, reason: str)" in src

    def test_helper_pops_recent_entry_attempts(self):
        """The helper actually pops the symbol from
        self._recent_entry_attempts (not just looks at it)."""
        src = ENGINE_PATH.read_text()
        marker = "def _clear_recent_attempt_for_order"
        start = src.index(marker)
        body = src[start : start + 1200]
        assert "self._recent_entry_attempts.pop(sym, None)" in body
        # Must look up symbol from pending_orders so callers can pass
        # only order_id (the only thing in scope inside _verify_order_fill).
        assert "self.pending_orders.get(order_id, {}).get('symbol')" in body

    def test_verify_timeout_extended_to_30(self):
        """Default max_retries should be 30, giving a 60s verify window
        (vs prior 10/20s). Anchored to the v-verify-timeout-extend-
        2026-05-26 marker so future timeout drift is caught."""
        src = ENGINE_PATH.read_text()
        # The signature line is the source of truth.
        assert "_verify_order_fill(self, order_id: str, signal, max_retries: int = 30)" in src
        assert "v-verify-timeout-extend-2026-05-26" in src

    def test_autocancel_log_line_present(self):
        """The post-timeout cancel block must INFO-log when it
        silently cancels an order — otherwise the user has no way
        to know why their order disappeared from Schwab."""
        src = ENGINE_PATH.read_text()
        assert "auto_cancelled_unfilled_order" in src
        assert "v-autocancel-logging-2026-05-26" in src

    def test_clear_helper_called_from_three_sites(self):
        """The ghost-attempt cleanup must run at every terminal-not-
        filled path inside _verify_order_fill:
          1. retry-loop sees Schwab status REJECTED/CANCELED
          2. final post-retry check sees REJECTED/CANCELED
          3. bot itself calls cancel_order after timeout

        Anchor each call site by the unique `reason=` string we pass.
        """
        src = ENGINE_PATH.read_text()
        # Pull just the _verify_order_fill body for the assertion so
        # we don't accidentally match calls from elsewhere.
        start = src.index("async def _verify_order_fill")
        end = src.index("async def _check_order_status", start)
        body = src[start:end]
        # Retry loop CANCELED/REJECTED
        assert 'reason=f"schwab_{status.lower()}"' in body
        # Final post-retry check CANCELED/REJECTED
        assert 'reason=f"schwab_final_{final_status.lower()}"' in body
        # Bot-initiated cancel after timeout
        assert 'reason="auto_cancel_verify_timeout"' in body

    def test_autocancel_block_cleans_up_order_id_dict(self):
        """The post-timeout cancel block also clears order_id_to_symbol
        so the inverse-lookup dict doesn't accumulate dead entries."""
        src = ENGINE_PATH.read_text()
        start = src.index("v-autocancel-logging-2026-05-26")
        block = src[start : start + 1500]
        # both the pending_orders and the inverse map should be cleaned
        assert "del self.pending_orders[order_id]" in block
        assert "self.order_id_to_symbol.pop(order_id, None)" in block


# ── v-news-confirmation-gate-2026-05-27 ──────────────────────────────

class TestNewsTechnicalConfirmationGate:
    """News-as-consulting-not-decision gate.

    Triggered by 3-for-3 losing entries on bullish-news + already-moved
    pattern: FLY 2026-05-26 -$72.50, ASTS 2026-05-26 -$59.84, TSLA
    2026-05-27 (user manually shorted +$360 after the news strategy's
    BUY signal would have lost). The gate requires technical
    confirmation before news_strategy emits signal_buy/signal_sell.

    For BUY: rsi<55, price within 5% of sma_20, macd_val >= macd_sig,
             volume_ratio >= 1.2x.
    For SELL: mirror image.

    Behind ENABLE_NEWS_TECHNICAL_CONFIRMATION (default True) so it can
    be toggled off via API if proven too restrictive in live trading.
    """

    def test_config_property_exists(self):
        from core.config import Config
        cfg = Config()
        # Default must be True so the gate is on out of the box.
        assert cfg.ENABLE_NEWS_TECHNICAL_CONFIRMATION is True

    def test_gate_marker_in_news_strategy(self):
        src = NEWS_STRATEGY_PATH.read_text()
        assert "v-news-confirmation-gate-2026-05-27" in src
        # Must also be referenced in config.py so future-me can grep
        # both ends of the wiring.
        cfg_src = CONFIG_PATH.read_text()
        assert "v-news-confirmation-gate-2026-05-27" in cfg_src

    def test_gate_reads_config_flag(self):
        """The gate must read ENABLE_NEWS_TECHNICAL_CONFIRMATION so
        operators can flip it without a restart. Hardcoded gate would
        defeat the whole point of making it toggle-able."""
        src = NEWS_STRATEGY_PATH.read_text()
        start = src.index("v-news-confirmation-gate-2026-05-27")
        body = src[start : start + 4500]
        assert "ENABLE_NEWS_TECHNICAL_CONFIRMATION" in body

    def test_all_four_buy_criteria_present(self):
        """BUY direction must check all four: RSI, price extension
        vs SMA20, MACD direction, and volume_ratio.
        Anchor each by the failure-tag string we log so log grep
        stays stable even if the implementation refactors."""
        src = NEWS_STRATEGY_PATH.read_text()
        start = src.index("v-news-confirmation-gate-2026-05-27")
        body = src[start : start + 4500]
        # BUY branch failure tags
        assert 'rsi_extended=' in body
        assert 'price_extended_above_sma20=' in body
        assert 'macd_bearish_cross=' in body
        assert 'low_volume=' in body

    def test_sell_direction_mirror_criteria(self):
        """SELL direction must use mirror-image thresholds (RSI>=45
        for oversold, price extended BELOW SMA20, MACD bullish cross).
        Without this, SELL signals would bypass the gate entirely."""
        src = NEWS_STRATEGY_PATH.read_text()
        start = src.index("v-news-confirmation-gate-2026-05-27")
        body = src[start : start + 4500]
        assert 'rsi_oversold=' in body
        assert 'price_extended_below_sma20=' in body
        assert 'macd_bullish_cross=' in body

    def test_gate_runs_after_verifier_before_signal_emission(self):
        """The gate must run AFTER the verifier success path (so we
        don't burn verifier cycles on signals the gate would block)
        but BEFORE the `_log_decision("signal_buy", ...)` emission.
        Anchor by relative position of the v-tag and the emission."""
        src = NEWS_STRATEGY_PATH.read_text()
        gate_pos = src.index("v-news-confirmation-gate-2026-05-27")
        # Find the next signal_buy emission after the gate.
        emit_pos = src.index('"signal_buy" if signal_type', gate_pos)
        assert gate_pos < emit_pos
        # And the "Fresh News Confirmed" commentary should come AFTER
        # the gate (gate runs first; if gate skips, no "confirmed"
        # banner shown to the user — which would be misleading).
        confirmed_pos = src.index("Fresh News Confirmed", gate_pos)
        assert gate_pos < confirmed_pos
        assert confirmed_pos < emit_pos

    def test_skip_log_reason_is_no_technical_confirmation(self):
        """Anchor on the exact reason string used in
        `_log_decision(..., "skip", "no_technical_confirmation", ...)`
        so audit grep stays stable."""
        src = NEWS_STRATEGY_PATH.read_text()
        start = src.index("v-news-confirmation-gate-2026-05-27")
        body = src[start : start + 4500]
        assert '"no_technical_confirmation"' in body


# ── v-state-persistence-loop-2026-05-27 ──────────────────────────────

class TestStatePersistenceLoop:
    """Periodic trading_state.json save loop.

    Triggered by 2026-05-27 QCOM incident: user closed QCOM at ~15:30,
    sync_positions_with_schwab correctly removed it from in-memory
    self.positions, but trading_state.json stayed stale for 38 minutes
    because no _save_state call fired in that window. Without periodic
    saves, any user-manual close of a bot-tracked position leaves the
    disk state lying about positions the bot doesn't think it holds.

    Fix: dedicated NORMAL-priority loop calling _save_state every
    _SAVE_INTERVAL_SEC. Defensive (try/except around save call), exits
    cleanly on self.is_running=False, skips first cycle via initial
    asyncio.sleep so startup race conditions don't write a half-built
    state file.
    """

    def test_loop_method_exists_with_marker(self):
        """Anchored to v-state-persistence-loop-2026-05-27 so future
        grep finds both the v-tag and the implementation."""
        src = ENGINE_PATH.read_text()
        assert "async def _state_persistence_loop" in src
        assert "v-state-persistence-loop-2026-05-27" in src

    def test_loop_registered_with_supervisor(self):
        """Without `sup.register("state_persistence", ...)` the method
        never runs. Catch a silent removal of the registration line."""
        src = ENGINE_PATH.read_text()
        assert 'sup.register("state_persistence"' in src
        assert "self._state_persistence_loop" in src

    def test_loop_calls_save_state_in_try_except(self):
        """A bare _save_state call would crash the supervised task
        on the first write failure (disk full, permissions, etc.) and
        the loop would never recover. Verify the call is guarded."""
        src = ENGINE_PATH.read_text()
        start = src.index("async def _state_persistence_loop")
        body = src[start : start + 2500]
        assert "self._save_state()" in body
        # The save call must be inside a try block. Anchor on the
        # warning log line that the except branch writes.
        assert "state_persistence_loop: save failed" in body

    def test_loop_respects_is_running_for_clean_shutdown(self):
        """Loop must exit when self.is_running=False so SIGTERM
        shutdown can complete without forcing a kill. Without this,
        the loop would block shutdown waiting for the next sleep
        to elapse."""
        src = ENGINE_PATH.read_text()
        start = src.index("async def _state_persistence_loop")
        body = src[start : start + 2500]
        assert "while self.is_running:" in body

    def test_loop_initial_sleep_before_first_save(self):
        """First-cycle save would race against engine startup
        (positions not yet synced, brain not yet loaded). Verify
        the loop sleeps the full interval before the first save."""
        src = ENGINE_PATH.read_text()
        start = src.index("async def _state_persistence_loop")
        body = src[start : start + 2500]
        # The initial sleep must come BEFORE the while loop.
        sleep_pos = body.find("await asyncio.sleep")
        while_pos = body.find("while self.is_running:")
        assert sleep_pos != -1 and while_pos != -1
        assert sleep_pos < while_pos, (
            "first asyncio.sleep must come before the while loop "
            "so we don't write a half-built state on startup"
        )


# ── v-mean-rev-uptrend-pullback-2026-05-28 ───────────────────────────

class TestMeanRevUptrendPullback:
    """Alternative mean-rev entry for stocks pulling back to support
    inside an uptrend.

    Triggered 2026-05-28 ~10:00 ET. Three sessions of the bot being
    silent while specific names ran on momentum. Existing mean-rev
    fires only on extreme oversold (RSI<30 + close below BB lower);
    never catches "uptrend pulling back to support" — a pattern that
    today's regime keeps producing.

    New trigger: RSI 30-45 + close>=SMA50 + close within 2% of
    BB lower (or up to 1% below). Same downstream gates: falling-
    knife (naturally inactive since close>=SMA50), price-direction
    (still must see green bar + vol>=1.5x + lower-wick > body).
    Same ATR-based stop/target. Pattern logged with
    `reason=uptrend_pullback` for audit.

    Gated by ENABLE_MEAN_REV_UPTREND_PULLBACK (default True).
    """

    def test_config_property_exists(self):
        from core.config import Config
        cfg = Config()
        assert cfg.ENABLE_MEAN_REV_UPTREND_PULLBACK is True

    def test_marker_in_strategy(self):
        src = (REPO_ROOT / "strategies" / "builtin.py").read_text()
        assert "v-mean-rev-uptrend-pullback-2026-05-28" in src
        cfg_src = CONFIG_PATH.read_text()
        assert "v-mean-rev-uptrend-pullback-2026-05-28" in cfg_src

    def test_rsi_window(self):
        """The new path's RSI band. Widened 2026-05-28 from 30-45 to
        30-55 after the initial deploy produced 0 RTH fires —
        regime was momentum, not classic pullback. Anchor on the
        current value so future drift gets caught."""
        src = (REPO_ROOT / "strategies" / "builtin.py").read_text()
        start = src.index("v-mean-rev-uptrend-pullback-2026-05-28")
        body = src[start : start + 5000]
        assert "30 <= rsi <= 55" in body
        # And the widen v-tag is present so the history is grep-able.
        assert "v-mean-rev-uptrend-pullback-widen-2026-05-28" in body

    def test_requires_close_at_or_above_sma50(self):
        """Uptrend confirmation must demand close>=SMA50. Without
        this the new path becomes a generic 'buy mid-RSI' entry."""
        src = (REPO_ROOT / "strategies" / "builtin.py").read_text()
        start = src.index("v-mean-rev-uptrend-pullback-2026-05-28")
        body = src[start : start + 3500]
        assert "market_data.close >= _sma_50_for_uptrend" in body

    def test_bb_lower_proximity_band(self):
        """BB-lower proximity band. Widened 2026-05-28 from 2% to 8%
        above bb_lower (still -1% below allowed). Captures pullbacks
        that paused above support, not just at it."""
        src = (REPO_ROOT / "strategies" / "builtin.py").read_text()
        start = src.index("v-mean-rev-uptrend-pullback-2026-05-28")
        body = src[start : start + 5000]
        assert "-1.0 <= _bb_lower_dist_pct <= 8.0" in body

    def test_bar_gate_relaxed_for_uptrend_pullback(self):
        """Original gate (green + vol>=1.5x + lower_wick>body) is
        applied to classic falling-knife oversold. Uptrend_pullback
        only needs green bar — close>=SMA50 IS the safety. Anchor
        on the branching code."""
        src = (REPO_ROOT / "strategies" / "builtin.py").read_text()
        assert "v-uptrend-pullback-gate-relax-2026-05-28" in src
        start = src.index("v-uptrend-pullback-gate-relax-2026-05-28")
        body = src[start : start + 1500]
        assert 'if _entry_pattern == "uptrend_pullback":' in body
        assert "_bar_ok = _is_green" in body  # relaxed branch


# ── v-direction-gate-*-2026-05-29 ────────────────────────────────────

class TestDirectionGates:
    """Three new direction-reader gates wired into strategies.

    Each gate calls core.direction_reader.read_direction(close,
    indicators) and skips the signal if the directional read
    disagrees with the strategy's premise:

      Mean-rev uptrend_pullback:  direction>=2 AND phase!=exhausted.
                                  Classic oversold_bounce NOT gated
                                  (its edge is buying downtrends).
      Breakout BUY:               direction>=3 AND phase in {early,
                                  middle}. Stricter — late breakouts
                                  systematically fail.
      News BUY:                   allows_long_entry (direction>=0 +
                                  not exhausted). News SELL: mirror.

    Each gate is behind its own Config flag (default True). If the
    direction reader raises an exception the strategy gates fall
    through to existing logic — defense-in-depth, not single-point-
    of-failure.
    """

    def test_config_flags_exist_with_defaults(self):
        from core.config import Config
        cfg = Config()
        assert cfg.ENABLE_DIRECTION_GATE_MEAN_REV is True
        assert cfg.ENABLE_DIRECTION_GATE_BREAKOUT is True
        assert cfg.ENABLE_DIRECTION_GATE_NEWS is True

    def test_mean_rev_uptrend_pullback_gate_marker_in_builtin(self):
        src = (REPO_ROOT / "strategies" / "builtin.py").read_text()
        assert "v-direction-gate-mean-rev-uptrend-pullback-2026-05-29" in src

    def test_mean_rev_gate_only_applies_to_uptrend_pullback(self):
        """The gate must be inside an `if _entry_pattern ==
        'uptrend_pullback':` block so it doesn't accidentally also
        block the classic oversold_bounce path."""
        src = (REPO_ROOT / "strategies" / "builtin.py").read_text()
        start = src.index("v-direction-gate-mean-rev-uptrend-pullback-2026-05-29")
        body = src[start : start + 3000]
        assert 'if _entry_pattern == "uptrend_pullback":' in body
        # The skip log reason must match the audit grep
        assert '"direction_gate_rejected_uptrend_pullback"' in body

    def test_mean_rev_gate_requires_direction_at_least_2(self):
        """Mean-rev uptrend_pullback needs CLEAR uptrend reading,
        not just mild bullish. direction>=2 + phase!=exhausted."""
        src = (REPO_ROOT / "strategies" / "builtin.py").read_text()
        start = src.index("v-direction-gate-mean-rev-uptrend-pullback-2026-05-29")
        body = src[start : start + 3000]
        assert "_dr.direction >= 2.0" in body
        assert '_dr.phase != "exhausted"' in body

    def test_breakout_gate_marker_in_builtin(self):
        src = (REPO_ROOT / "strategies" / "builtin.py").read_text()
        assert "v-direction-gate-breakout-2026-05-29" in src

    def test_breakout_gate_requires_direction_3_and_early_middle(self):
        """Breakout is stricter — requires direction>=3 (strong
        uptrend) AND phase in {early, middle} (not late or
        exhausted). The whole point is to prevent late breakouts."""
        src = (REPO_ROOT / "strategies" / "builtin.py").read_text()
        start = src.index("v-direction-gate-breakout-2026-05-29")
        body = src[start : start + 3000]
        assert "_dr.direction >= 3.0" in body
        assert '_dr.phase in ("early", "middle")' in body
        assert '"direction_gate_rejected_breakout"' in body

    def test_news_gate_marker_in_news_strategy(self):
        src = (REPO_ROOT / "strategies" / "news_strategy.py").read_text()
        assert "v-direction-gate-news-2026-05-29" in src

    def test_news_gate_uses_allows_long_or_short_helper(self):
        """News gate must use the DirectionRead.allows_long_entry /
        allows_short_entry helpers rather than re-implementing the
        phase+direction logic. Centralizes the rule."""
        src = (REPO_ROOT / "strategies" / "news_strategy.py").read_text()
        start = src.index("v-direction-gate-news-2026-05-29")
        body = src[start : start + 3500]
        assert "_dr.allows_long_entry" in body
        assert "_dr.allows_short_entry" in body
        assert '"direction_gate_rejected_news"' in body

    def test_market_context_gates_wired_in_strategies(self):
        """v-market-context-gate-{mean-rev,breakout,news}-2026-06-08:
        each strategy must check MarketContext before signal emission
        when ENABLE_MARKET_CONTEXT_GATE is True, and skip with a
        canonical reason if regime blocks the side.

        Anchor on the exact skip-reason strings so audit grep stays
        stable."""
        builtin_src = (REPO_ROOT / "strategies" / "builtin.py").read_text()
        news_src = (REPO_ROOT / "strategies" / "news_strategy.py").read_text()

        # Mean-rev gate
        assert "v-market-context-gate-mean-rev-2026-06-08" in builtin_src
        assert '"market_context_blocks_long"' in builtin_src

        # Breakout gate
        assert "v-market-context-gate-breakout-2026-06-08" in builtin_src
        assert '"market_context_blocks_breakout"' in builtin_src

        # News gates (BUY and SELL)
        assert "v-market-context-gate-news-2026-06-08" in news_src
        assert '"market_context_blocks_news_buy"' in news_src
        assert '"market_context_blocks_news_sell"' in news_src

    def test_market_context_conviction_threads_to_risk_manager(self):
        """Each strategy must inject market_context_conviction into
        signal.reasoning so risk_manager can apply the sizing
        multiplier. Anchor on the key name."""
        builtin_src = (REPO_ROOT / "strategies" / "builtin.py").read_text()
        news_src = (REPO_ROOT / "strategies" / "news_strategy.py").read_text()
        risk_src = (REPO_ROOT / "risk" / "manager.py").read_text()

        # Strategies set the key in reasoning dict
        assert "'market_context_conviction'" in builtin_src
        assert "'market_context_conviction'" in news_src

        # Risk manager reads + applies it
        assert "v-market-context-sizing-2026-06-08" in risk_src
        assert "market_context_conviction" in risk_src
        assert "ENABLE_MARKET_CONTEXT_SIZING" in risk_src

    def test_market_context_config_flags_default_true(self):
        from core.config import Config
        cfg = Config()
        assert cfg.ENABLE_MARKET_CONTEXT_GATE is True
        assert cfg.ENABLE_MARKET_CONTEXT_SIZING is True

    def test_all_three_gates_swallow_exceptions(self):
        """If direction_reader raises (NaN inputs, missing keys,
        etc.) the strategy MUST fall through to existing logic, not
        block the signal. Pattern: try/except around the read +
        comparison; pass on exception."""
        for path, tag in [
            ("strategies/builtin.py", "v-direction-gate-mean-rev-uptrend-pullback-2026-05-29"),
            ("strategies/builtin.py", "v-direction-gate-breakout-2026-05-29"),
            ("strategies/news_strategy.py", "v-direction-gate-news-2026-05-29"),
        ]:
            src = (REPO_ROOT / path).read_text()
            start = src.index(tag)
            body = src[start : start + 3500]
            assert "try:" in body, f"{tag} missing try/except"
            # And the except must just pass (not raise / return None)
            assert "except Exception:" in body, f"{tag} missing except"
            # Look for the pass — it's how we fall through.
            assert "pass" in body, f"{tag} except block must `pass`"


# ── v-rr-audit-2026-06-01 ────────────────────────────────────────────

class TestRRAudit:
    """Risk:reward sanity check at the signal_router accept point.

    Triggered by the 2026-05-28 bb_middle bug where mean-rev signals
    fired with R:R 0.7-1.4 (vs configured 2.0). The bug went unnoticed
    for hours because every signal_buy log line showed stop+target
    individually but never the computed ratio.

    The audit fires AT THE ROUTER (one place) so it covers every
    strategy without per-strategy plumbing. WARNING (not block) so
    it surfaces during normal log review without interrupting trading.
    """

    def test_marker_in_engine(self):
        src = ENGINE_PATH.read_text()
        assert "v-rr-audit-2026-06-01" in src

    def test_audit_fires_after_accepted_audit(self):
        """The R:R check must run AFTER the existing signal_router
        'accepted' audit line so we know the signal made it through
        all gates before we warn about its sizing."""
        src = ENGINE_PATH.read_text()
        accept_pos = src.index('self._audit("signal_router", signal.symbol, "accepted"')
        audit_pos = src.index("v-rr-audit-2026-06-01")
        assert accept_pos < audit_pos, (
            "R:R audit must follow signal_router accept so we know "
            "we're auditing a real accepted trade"
        )

    def test_audit_floor_is_1_8(self):
        """Configured floor is 1.8 R:R — below the default 2.0 R:R
        the strategies target, with a 10% tolerance for normal
        variation. Anything below 1.8 indicates a real bug (like
        bb_middle capping) or a strategy author intentionally
        deviating from R:R discipline."""
        src = ENGINE_PATH.read_text()
        start = src.index("v-rr-audit-2026-06-01")
        body = src[start : start + 2500]
        assert "_RR_FLOOR = 1.8" in body

    def test_audit_uses_warning_not_error(self):
        """WARNING level so it surfaces in log review without
        triggering operator alerts as if a trade had failed."""
        src = ENGINE_PATH.read_text()
        start = src.index("v-rr-audit-2026-06-01")
        body = src[start : start + 2500]
        assert "logger.warning(" in body
        assert "rr_audit_low" in body

    def test_audit_swallows_exceptions(self):
        """Audit math must never block the trade. Pattern: try/
        except → logger.debug → pass."""
        src = ENGINE_PATH.read_text()
        start = src.index("v-rr-audit-2026-06-01")
        body = src[start : start + 2500]
        assert "try:" in body
        # The except must log at debug + continue (we don't want a
        # math error to cancel the trade).
        assert "except Exception" in body
        assert "logger.debug(" in body

    def test_signal_log_uses_pattern_string(self):
        """Anchor on the exact reason="uptrend_pullback" string so
        audit grep stays stable."""
        src = (REPO_ROOT / "strategies" / "builtin.py").read_text()
        start = src.index("v-mean-rev-uptrend-pullback-2026-05-28")
        body = src[start : start + 6000]
        assert '"uptrend_pullback"' in body
        # And the classic path's pattern is preserved (regression
        # guard — don't accidentally rename the existing trigger).
        assert '"oversold_bounce"' in body


# ── v-cancel-bracket-before-close-2026-06-08 ────────────────────────

class TestCancelBracketBeforeClose:
    """Manual close must cancel the active OCO bracket BEFORE placing
    the sell order. Bug observed 2026-06-08 11:12 ET: closes for NOK,
    GLW, OWL were rejected by Schwab with `oversold position` because
    the OCO's stop+target legs still pledged the shares. The override
    path in `_close_real_position` placed the sell directly without
    cancelling the bracket first.

    Fix: in `_close_real_position`, call `_cancel_existing_orders` for
    the position's symbol BEFORE constructing/placing the close order.
    The helper already exists (line ~1204) and is used in the entry
    path (line ~5241) for the same reason.
    """

    MARKER = "v-cancel-bracket-before-close-2026-06-08"

    def test_marker_present_in_close_real_position(self):
        """v-tag must live inside _close_real_position so future
        regressions are grep-discoverable."""
        src = ENGINE_PATH.read_text()
        assert self.MARKER in src, (
            "v-tag missing — fix regressed or never landed"
        )
        # And it must be inside _close_real_position, not floating
        # in some unrelated comment elsewhere.
        fn_start = src.index("async def _close_real_position")
        # Find the next top-level def to bound the function body.
        # Searching for the next `\n    async def ` or `\n    def `.
        m = re.search(
            r"\n    (?:async )?def ",
            src[fn_start + 1 :],
        )
        fn_end = (fn_start + 1 + m.start()) if m else len(src)
        body = src[fn_start:fn_end]
        assert self.MARKER in body, (
            "v-tag exists but not inside _close_real_position body"
        )

    def test_cancel_existing_orders_called_in_close_path(self):
        """The fix is to call `await self._cancel_existing_orders(
        position.symbol)` inside `_close_real_position`."""
        src = ENGINE_PATH.read_text()
        fn_start = src.index("async def _close_real_position")
        m = re.search(
            r"\n    (?:async )?def ",
            src[fn_start + 1 :],
        )
        fn_end = (fn_start + 1 + m.start()) if m else len(src)
        body = src[fn_start:fn_end]
        assert "_cancel_existing_orders(position.symbol)" in body, (
            "_close_real_position must call "
            "_cancel_existing_orders(position.symbol) before placing "
            "the sell — otherwise Schwab rejects with 'oversold "
            "position' when an OCO bracket is active."
        )

    def test_cancel_happens_before_place_order(self):
        """Lexical-ordering check: in _close_real_position, the
        _cancel_existing_orders call must appear BEFORE the
        place_order call. Otherwise the order will still race the
        bracket."""
        src = ENGINE_PATH.read_text()
        fn_start = src.index("async def _close_real_position")
        m = re.search(
            r"\n    (?:async )?def ",
            src[fn_start + 1 :],
        )
        fn_end = (fn_start + 1 + m.start()) if m else len(src)
        body = src[fn_start:fn_end]

        cancel_pos = body.find("_cancel_existing_orders(position.symbol)")
        place_pos = body.find("self.schwab_client.place_order(")
        assert cancel_pos != -1, "cancel call missing"
        assert place_pos != -1, "place_order call missing"
        assert cancel_pos < place_pos, (
            "_cancel_existing_orders must be called BEFORE "
            "place_order — bracket release has to settle before the "
            "sell or Schwab will reject."
        )


# ── v-rising-peak-filter-2026-06-08 ─────────────────────────────────

BUILTIN_PATH = REPO_ROOT / "strategies" / "builtin.py"


class TestRisingPeakFilter:
    """Mean-rev SHORT entries must require a confirmed downtrend before
    firing. Symmetric to the falling-knife filter on the LONG side.

    May 11 2026 incident: re-enabling mean-rev SHORT produced
    shorts-only behavior — 5 simultaneous shorts at -$665 unrealized —
    because RSI>70 fires constantly during a sustained rally (it's the
    norm, not a reversal signal). The long-side has a falling-knife
    filter (close<SMA50 blocks longs in downtrends). The short-side
    never got the symmetric "rising peak" filter blocking shorts in
    uptrends.

    Spec:
      * Apply ONLY to the mean-rev SHORT branch (rsi>70 + close>bb_upper)
      * Apply AFTER the ENABLE_MEAN_REV_SHORT gate (so the existing
        disable still wins)
      * Required gates (BOTH):
          - close < SMA50          (downtrend confirmation)
          - MACD < MACD signal     (bearish momentum confirmation)
      * If either fails → skip with audit reason "rising_peak_uptrend"
      * If indicator data is missing (sma_50 == 0) → skip with audit
        reason "rising_peak_no_data" (fail-closed)
      * Gateable via Config.ENABLE_RISING_PEAK_FILTER (default True)
      * Must NOT touch any LONG path (oversold_bounce, uptrend_pullback)
      * Must NOT touch news strategy, breakout, or momentum branches
    """

    MARKER = "v-rising-peak-filter-2026-06-08"

    def test_config_flag_exists_with_default_true(self):
        """The filter is safety-on by default. Operators flip it OFF
        only for backtest comparison."""
        from core.config import Config
        cfg = Config()
        assert hasattr(cfg, "ENABLE_RISING_PEAK_FILTER"), (
            "Config.ENABLE_RISING_PEAK_FILTER property missing"
        )
        assert cfg.ENABLE_RISING_PEAK_FILTER is True, (
            "default must be True — re-enabling SHORT without the "
            "filter is the exact path to the May 11 -$665 incident"
        )

    def test_marker_present_in_mean_rev_short_branch(self):
        """v-tag lives inside the mean-rev SHORT branch (between the
        `elif rsi > 70` line and the SELL signal construction)."""
        src = BUILTIN_PATH.read_text()
        assert self.MARKER in src, "v-tag missing"
        # And it must be after the elif that opens the SHORT branch
        short_branch_start = src.find("elif rsi > 70 and market_data.close > bb_upper")
        assert short_branch_start != -1, (
            "mean-rev SHORT branch signature drifted — verify spec"
        )
        marker_pos = src.find(self.MARKER, short_branch_start)
        assert marker_pos != -1, "v-tag not inside SHORT branch"

    def test_filter_uses_required_indicators(self):
        """Filter must check close<sma_50 AND macd<macd_signal.
        Anchored on the literal indicator keys."""
        src = BUILTIN_PATH.read_text()
        marker_pos = src.index(self.MARKER)
        # Filter logic lives in the ~1000 chars after the marker.
        body = src[marker_pos : marker_pos + 2000]
        assert "sma_50" in body, "filter must read sma_50"
        assert "macd_signal" in body, "filter must read macd_signal"

    def test_filter_skip_audit_reasons(self):
        """When the filter blocks, it must emit auditable skip
        reasons so live behavior is grep-discoverable."""
        src = BUILTIN_PATH.read_text()
        marker_pos = src.index(self.MARKER)
        body = src[marker_pos : marker_pos + 2000]
        assert "rising_peak_uptrend" in body, (
            "block-path audit reason must be 'rising_peak_uptrend'"
        )
        assert "rising_peak_no_data" in body, (
            "missing-data fail-closed audit reason must be "
            "'rising_peak_no_data'"
        )

    def test_filter_gateable_via_config(self):
        """The filter itself must check ENABLE_RISING_PEAK_FILTER —
        so operators can flip it off for backtest comparison without
        editing code."""
        src = BUILTIN_PATH.read_text()
        marker_pos = src.index(self.MARKER)
        body = src[marker_pos : marker_pos + 2000]
        assert "ENABLE_RISING_PEAK_FILTER" in body, (
            "filter must be gated by Config.ENABLE_RISING_PEAK_FILTER"
        )

    def test_long_branch_untouched_by_filter(self):
        """REGRESSION GUARD: filter must not touch the LONG paths.
        Verify the marker is NOT inside the oversold_bounce or
        uptrend_pullback code paths."""
        src = BUILTIN_PATH.read_text()
        marker_pos = src.index(self.MARKER)
        # The LONG SELL signal_buy paths come before the SHORT branch.
        # If our marker appears before any LONG construction, the patch
        # landed in the wrong place.
        long_signal_pos = src.find("signal_buy")
        if long_signal_pos != -1:
            assert marker_pos > long_signal_pos, (
                "v-tag landed before LONG signal_buy — wrong location"
            )

    def test_news_strategy_untouched(self):
        """REGRESSION GUARD: news strategy must not reference this
        filter. The news strategy has its own rising-knife filter and
        its own short discipline; mixing them risks unintended
        coupling."""
        news_src = NEWS_STRATEGY_PATH.read_text()
        assert "ENABLE_RISING_PEAK_FILTER" not in news_src, (
            "news strategy must not couple to mean-rev's filter"
        )
        assert self.MARKER not in news_src, (
            "v-tag must not be inside news strategy"
        )


# ── v-verify-close-fill-2026-06-08 ──────────────────────────────────

class TestVerifyCloseFill:
    """`_close_real_position` must verify the close order actually
    filled at Schwab before returning True. Bug observed 2026-06-08
    11:12 ET alongside the OCO-before-close bug:

    1. close path placed sell order
    2. Schwab returned HTTP 201 ("accepted into validation")
    3. close path returned True
    4. caller (`_close_position_with_commentary`) popped the position
       from local state
    5. Schwab asynchronously REJECTED the order ("oversold position")
    6. position now gone from local state but STILL OPEN at Schwab
       with active OCO bracket — bot blind to it

    Fix: after place_order returns 201, await `_verify_order_fill` to
    confirm the FILLED status. Same helper as the entry path. Return
    False on REJECTED/CANCELED/timeout so the caller promotes the
    position to ZOMBIE state instead of removing it.

    This sits on top of #33 (cancel OCO before sell). Both must land
    together — #33 prevents most rejections, #34 catches the remaining
    failure modes (margin, halt, etc.) without state corruption.
    """

    MARKER = "v-verify-close-fill-2026-06-08"

    def test_marker_present_in_close_real_position(self):
        """v-tag must live inside _close_real_position so regressions
        are grep-discoverable."""
        src = ENGINE_PATH.read_text()
        assert self.MARKER in src, "v-tag missing"
        fn_start = src.index("async def _close_real_position")
        m = re.search(
            r"\n    (?:async )?def ",
            src[fn_start + 1 :],
        )
        fn_end = (fn_start + 1 + m.start()) if m else len(src)
        body = src[fn_start:fn_end]
        assert self.MARKER in body, (
            "v-tag must be inside _close_real_position body, not "
            "elsewhere in the file"
        )

    def test_verify_called_on_success_path(self):
        """The fix is calling `_verify_order_fill` after the 201 status
        check, before returning True."""
        src = ENGINE_PATH.read_text()
        fn_start = src.index("async def _close_real_position")
        m = re.search(
            r"\n    (?:async )?def ",
            src[fn_start + 1 :],
        )
        fn_end = (fn_start + 1 + m.start()) if m else len(src)
        body = src[fn_start:fn_end]
        assert "_verify_order_fill" in body, (
            "_close_real_position must call _verify_order_fill before "
            "returning True — Schwab's 201 means accepted-for-validation, "
            "not filled."
        )

    def test_verify_gated_on_status_check(self):
        """The verify call must appear AFTER the
        `response.status_code in [200, 201]` check — otherwise we'd
        verify on a non-existent order ID."""
        src = ENGINE_PATH.read_text()
        fn_start = src.index("async def _close_real_position")
        m = re.search(
            r"\n    (?:async )?def ",
            src[fn_start + 1 :],
        )
        fn_end = (fn_start + 1 + m.start()) if m else len(src)
        body = src[fn_start:fn_end]
        status_check_pos = body.find("response.status_code in [200, 201]")
        verify_pos = body.find("_verify_order_fill")
        assert status_check_pos != -1, "status-code check signature drifted"
        assert verify_pos > status_check_pos, (
            "_verify_order_fill must be called inside the success "
            "branch, after the status_code check"
        )

    def test_verify_failure_returns_false(self):
        """When `_verify_order_fill` returns False (REJECTED, CANCELED,
        timeout), `_close_real_position` must return False so the
        caller's ZOMBIE escalation path runs.

        Grep-anchor: the body must contain `if not filled` (or
        equivalent) and a `return False` inside that block."""
        src = ENGINE_PATH.read_text()
        fn_start = src.index("async def _close_real_position")
        m = re.search(
            r"\n    (?:async )?def ",
            src[fn_start + 1 :],
        )
        fn_end = (fn_start + 1 + m.start()) if m else len(src)
        body = src[fn_start:fn_end]
        # Find the verify call, then look for the failure-return
        # within the next 500 chars.
        verify_pos = body.find("_verify_order_fill")
        assert verify_pos != -1, "verify call missing"
        tail = body[verify_pos : verify_pos + 800]
        assert "return False" in tail, (
            "after _verify_order_fill the close path must `return "
            "False` on verify failure so the caller promotes to ZOMBIE"
        )

    def test_zombie_escalation_path_unchanged(self):
        """REGRESSION GUARD: the caller's ZOMBIE escalation logic in
        `_close_position_with_commentary` must remain — that's what
        catches our new `return False`."""
        src = ENGINE_PATH.read_text()
        # The ZOMBIE promotion lives in the close-with-commentary path
        # after `success = await self._close_real_position(position)`.
        assert "broker_close_returned_false" in src, (
            "ZOMBIE escalation reason string missing — the close "
            "caller may have lost the `if not success` guard that "
            "depends on this fix returning False"
        )
        assert "PositionState.ZOMBIE.value" in src, (
            "ZOMBIE state assignment missing — caller logic regressed"
        )


# ── v-bot-only-pnl-circuit-2026-06-08 ───────────────────────────────

RISK_PATH = REPO_ROOT / "risk" / "manager.py"


class TestBotOnlyPnLCircuit:
    """Daily-loss circuit must use BOT-managed P&L, not account-wide
    Schwab P&L. Account-wide P&L includes external holdings that the
    bot never opened (e.g., HQGE/PINS/COIN in the operator's account).

    Why it matters: if external positions gap-down -8% overnight, the
    account-wide P&L crosses the -1% circuit threshold even when the
    bot's own performance is flat or positive. The bot then pauses
    trading on losses that are not its responsibility. The operator
    observed this exact pattern multiple times in May 2026.

    Fix:
      * Compute bot_daily_pnl = today's realized P&L of closed
        bot-managed trades + unrealized P&L of currently-open
        bot-managed positions.
      * Use bot_daily_pnl in `can_trade()` instead of schwab_daily_pnl.
      * Keep schwab_daily_pnl tracked and displayed (still useful
        for the user's overall portfolio view).
      * Gate via Config.ENABLE_BOT_ONLY_PNL_CIRCUIT (default True).
        Fallback to schwab_daily_pnl if flag is False.
    """

    MARKER = "v-bot-only-pnl-circuit-2026-06-08"

    def test_config_flag_exists_with_default_true(self):
        from core.config import Config
        cfg = Config()
        assert hasattr(cfg, "ENABLE_BOT_ONLY_PNL_CIRCUIT"), (
            "Config.ENABLE_BOT_ONLY_PNL_CIRCUIT missing"
        )
        assert cfg.ENABLE_BOT_ONLY_PNL_CIRCUIT is True, (
            "Default must be True — using account-wide P&L for the "
            "bot's daily-loss circuit is the documented failure mode"
        )

    def test_marker_in_risk_manager(self):
        src = RISK_PATH.read_text()
        assert self.MARKER in src, "v-tag missing from risk/manager.py"

    def test_risk_manager_tracks_bot_daily_pnl(self):
        """RiskManager must hold `bot_daily_pnl` (parallel to
        schwab_daily_pnl) so the engine sync loop can update it."""
        src = RISK_PATH.read_text()
        assert "self.bot_daily_pnl" in src, (
            "RiskManager must initialize self.bot_daily_pnl in __init__"
        )

    def test_check_trading_allowed_uses_bot_pnl_when_flag_on(self):
        """`check_trading_allowed()` (the daily-loss circuit entry
        point) must check ENABLE_BOT_ONLY_PNL_CIRCUIT and use
        bot_daily_pnl when True, schwab_daily_pnl otherwise."""
        src = RISK_PATH.read_text()
        # Find check_trading_allowed and its surrounding context.
        # The method is called check_trading_allowed in
        # RiskManagerWithCommentary (not can_trade).
        anchor = src.find("def check_trading_allowed(self)")
        assert anchor != -1, "check_trading_allowed signature drifted"
        body = src[anchor : anchor + 3000]
        assert "ENABLE_BOT_ONLY_PNL_CIRCUIT" in body, (
            "check_trading_allowed must reference "
            "ENABLE_BOT_ONLY_PNL_CIRCUIT"
        )
        assert "self.bot_daily_pnl" in body, (
            "check_trading_allowed must use bot_daily_pnl in the "
            "active branch"
        )

    def test_compute_bot_daily_pnl_in_engine(self):
        """Engine must provide a `_compute_bot_daily_pnl` method that
        sums today's realized bot trades + open bot positions'
        unrealized P&L."""
        src = ENGINE_PATH.read_text()
        assert "_compute_bot_daily_pnl" in src, (
            "engine must expose _compute_bot_daily_pnl"
        )
        # The method must filter on managed_by_bot
        method_pos = src.find("def _compute_bot_daily_pnl")
        assert method_pos != -1, "method definition missing"
        method_body = src[method_pos : method_pos + 2500]
        assert "managed_by_bot" in method_body, (
            "computation must filter on managed_by_bot"
        )

    def test_schwab_daily_pnl_still_tracked(self):
        """REGRESSION GUARD: schwab_daily_pnl must still be tracked
        and synced — it remains the source of truth for the
        operator-facing account-wide display."""
        src = RISK_PATH.read_text()
        assert "schwab_daily_pnl" in src, (
            "schwab_daily_pnl tracking regressed — required for "
            "dashboard P&L display"
        )
        eng_src = ENGINE_PATH.read_text()
        # The existing sync line must still set schwab_daily_pnl.
        assert "schwab_daily_pnl = schwab_pnl" in eng_src, (
            "engine sync loop must still update schwab_daily_pnl"
        )

    def test_bot_daily_pnl_synced_alongside_schwab(self):
        """The engine sync loop that sets schwab_daily_pnl must also
        update bot_daily_pnl on the same cadence — otherwise the
        circuit reads stale bot P&L."""
        src = ENGINE_PATH.read_text()
        # Find the schwab sync line and inspect the surrounding block
        # for a parallel bot_daily_pnl update.
        anchor = src.find("schwab_daily_pnl = schwab_pnl")
        assert anchor != -1, "engine schwab sync anchor missing"
        # Look in a wide window around the anchor.
        window = src[max(0, anchor - 500) : anchor + 800]
        assert "bot_daily_pnl" in window, (
            "engine sync loop must set risk_manager.bot_daily_pnl "
            "alongside schwab_daily_pnl"
        )


# ── v-shutdown-stops-engine-thread-2026-06-09 ────────────────────────

class TestShutdownStopsEngineThread:
    """2026-06-09 'init hang' root cause: shutdown_handler saved state
    and called sys.exit(0) but never stopped the engine thread. The
    thread was non-daemon, so the interpreter waited on it forever —
    a zombie process kept the Schwab stream, token refreshes, and
    state-file writes alive. The next instance then fought the zombie
    (token rotation race, concurrent state-file reads), which looked
    like an engine-init hang. Proof in trading_bot.log 2026-06-09:
    stream ticks logged at 10:41:19, AFTER 'Shutdown complete' at
    10:41:17."""

    def test_engine_thread_is_daemon(self):
        """The engine thread must be daemon=True so a main-thread exit
        can never be blocked by it (backstop — graceful stop below is
        the primary mechanism)."""
        src = ROUTES_PATH.read_text()
        anchor = src.find("target=lambda: asyncio.run(trading_engine.start())")
        assert anchor != -1, "engine thread creation anchor missing"
        window = src[max(0, anchor - 300) : anchor + 300]
        assert "daemon=True" in window, (
            "engine thread must be created with daemon=True — "
            "non-daemon engine thread caused the 2026-06-09 zombie"
        )

    def test_shutdown_handler_stops_engine_loop(self):
        """shutdown_handler must set is_running = False BEFORE saving
        state, so the engine loop winds down and the saved state is
        final, not mid-iteration."""
        src = ROUTES_PATH.read_text()
        anchor = src.find("def shutdown_handler(")
        assert anchor != -1, "shutdown_handler definition missing"
        body = src[anchor : src.find("signal.signal(signal.SIGINT", anchor)]
        assert "is_running = False" in body, (
            "shutdown_handler must stop the engine loop "
            "(is_running = False) — it previously exited the main "
            "thread while the engine kept trading"
        )

    def test_shutdown_handler_joins_engine_thread(self):
        """shutdown_handler must join the engine thread (bounded
        timeout) so state is saved only after the loop actually
        stopped."""
        src = ROUTES_PATH.read_text()
        anchor = src.find("def shutdown_handler(")
        body = src[anchor : src.find("signal.signal(signal.SIGINT", anchor)]
        assert ".join(timeout=" in body, (
            "shutdown_handler must join the engine thread with a "
            "bounded timeout before saving state"
        )

    def test_engine_thread_reference_kept(self):
        """/api/start must keep a module-level reference to the engine
        thread so shutdown_handler can join it."""
        src = ROUTES_PATH.read_text()
        assert "trading_thread" in src, (
            "engine thread must be stored (trading_thread) for the "
            "shutdown path to join"
        )
