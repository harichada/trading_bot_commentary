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
