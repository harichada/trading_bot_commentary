"""v-schwab-stream-2026-05-01 (Phase 4a): Schwab websocket quote stream.

Replaces the per-symbol REST polling in `_quote_streamer_loop` with a
single push subscription to LEVELONE_EQUITIES. Tick updates land in the
shared QuoteCache at sub-100ms latency; the rest of the engine —
position_loop, FSM, dashboard — reads from the same cache and now sees
real-time freshness for free.

Architecture:
  - One StreamClient per engine, authenticated against the existing
    schwab_client. Login is async; we keep one persistent connection.
  - Subscriptions are dynamic: as positions open/close the engine calls
    update_subscriptions() to add/remove symbols. The streamer diffs
    the requested set against the active set and issues subs/unsubs.
  - Disconnects auto-recover with exponential backoff (1s → 5s → 30s,
    capped). After 5 consecutive failures the stream marks itself
    unhealthy and the polling fallback (Phase 2's _quote_streamer_loop)
    takes over — that loop is unchanged and stays as a belt-and-
    suspenders backup.

Failure modes deliberately handled:
  - schwab-py library not present at import → falls back gracefully
    (the polling path still works)
  - StreamClient.login() raises (token issue, network down) →
    backoff + retry
  - Tick handler raises on a malformed message → log and continue
    (one bad tick must not kill the stream)
  - Subscription update during in-flight reconnect → enqueue and
    apply post-reconnect

Failure modes NOT handled here (out of scope for Phase 4a):
  - Account-activity stream (orders/fills) — that's the bracket-order
    Phase 4b. Quote stream and order events are separate concerns.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional, Set, TYPE_CHECKING

logger = logging.getLogger("TradingBot")

if TYPE_CHECKING:
    from core.quote_cache import QuoteCache

# Import the StreamClient lazily so a missing schwab-py at install time
# doesn't break the rest of the engine. The polling path remains the
# fallback if streaming is unavailable.
try:
    from schwab.streaming import StreamClient as _SchwabStreamClient
    STREAMING_AVAILABLE = True
except Exception as _exc:
    _SchwabStreamClient = None
    STREAMING_AVAILABLE = False
    logger.warning("schwab streaming unavailable: %s — falling back to polling", _exc)


class SchwabQuoteStream:
    """Single-process owner of the LEVELONE_EQUITIES subscription.

    Lifecycle:
      run()                 — long-running coroutine, supervised
      update_subscriptions(set[str]) — diff-based; safe to call any time
      is_healthy() -> bool  — returns False after consecutive failures;
                              the polling fallback uses this to decide
                              whether to re-enable itself
    """

    # Login + reconnect tuning. Conservative on the schedule because Schwab's
    # streamer doesn't love rapid reconnect attempts.
    _RECONNECT_BACKOFF_SEC = (1, 5, 15, 30, 60)
    _UNHEALTHY_AFTER_FAILURES = 5

    def __init__(
        self,
        schwab_client,
        account_hash: str,
        quote_cache: "QuoteCache",
        on_raw_message=None,
    ) -> None:
        self._client = schwab_client
        self._account_hash = account_hash
        self._cache = quote_cache
        # v-stream-raw-2026-05-01: optional callback — receives every
        # raw Schwab stream message before any parsing. Engine wires
        # this to ConnectionManager.broadcast so the /stream page can
        # render a live log of exactly what the broker is sending.
        self._on_raw_message = on_raw_message
        # Captured at run() start so the sync tick handler always
        # schedules cache writes against the SAME loop the engine runs.
        # asyncio.get_event_loop() inside the handler is unreliable
        # under modern asyncio (returns deprecation warnings or wrong
        # loop). v-stream-loop-capture-2026-05-01.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stream: Optional[_SchwabStreamClient] = None  # lazily built
        # The set of symbols we want subscribed at this moment.
        # update_subscriptions() writes here; the run loop diffs against
        # _active_symbols and issues subs/unsubs accordingly.
        self._desired_symbols: Set[str] = set()
        self._active_symbols: Set[str] = set()
        self._desired_lock = asyncio.Lock()
        # Health
        self._consecutive_failures: int = 0
        self._last_tick_at: Optional[datetime] = None
        self._stop = asyncio.Event()
        # The handler is set once on the first connect; we re-bind on
        # reconnect so it stays attached even if the StreamClient
        # rebuilds internal state.
        self._handler_attached: bool = False
        # v-stream-watchdog-2026-05-08: silent-death detector. When the
        # underlying websocket disconnects without raising (TCP RST,
        # broker-side close without protocol close), `handle_message()`
        # blocks forever and `run()`'s reconnect loop never fires.
        # Production observed two 10–22 hour silent-death windows. The
        # watchdog checks tick freshness during regular trading hours and
        # signals this event when no ticks have arrived for 120s, which
        # _serve() races against handle_message() to break out.
        self._silent_death = asyncio.Event()

    # ── public API ────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Whether streaming is even possible (library + client present)."""
        return STREAMING_AVAILABLE and self._client is not None

    def is_healthy(self) -> bool:
        """Polling fallback consults this. Healthy = recent successful
        connect AND tick freshness. Unhealthy → polling resumes."""
        if not self.is_available():
            return False
        if self._consecutive_failures >= self._UNHEALTHY_AFTER_FAILURES:
            return False
        # If we've been connected but no tick has arrived in 60s, treat
        # the stream as silently broken (e.g. weekend / market closed
        # is fine; during RTH a 60s gap is suspect).
        if self._last_tick_at is not None:
            age = (datetime.now(timezone.utc) - self._last_tick_at).total_seconds()
            if age > 60:
                return False
        return True

    async def update_subscriptions(self, symbols: Set[str]) -> None:
        """Atomically declare the set of symbols we want streaming.

        Diff is applied lazily by the run loop on its next iteration;
        this avoids inflight subscription edits during reconnects.
        Empty set is fine and means "unsubscribe everything."
        """
        async with self._desired_lock:
            # Defensive copy so the engine can't mutate it under us.
            self._desired_symbols = set(s for s in symbols if s)

    async def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                await asyncio.wait_for(self._stream.logout(), timeout=3.0)
            except Exception:
                pass

    # ── tick handler — receives every quote message ───────────────────

    _tick_count: int = 0
    _tick_log_every: int = 100
    _per_symbol_ticks: dict = None  # populated per-instance in _on_message

    def _on_message(self, msg) -> None:
        """LEVELONE_EQUITIES handler. schwab-py decodes the JSON for us.

        msg is a dict-like structure with `content` = list of per-symbol
        records. Field names follow Schwab's spec:
          'key' = symbol, '3' = LAST_PRICE, '1' = BID_PRICE,
          '2' = ASK_PRICE. Some library versions surface readable names
          like 'BID_PRICE' / 'ASK_PRICE' / 'LAST_PRICE' instead of the
          raw numeric keys; we handle both.
        """
        # v-stream-raw-2026-05-01: forward EVERY raw message to the
        # observer callback (typically wired to broadcast for /stream
        # debug view). Fire before any parsing so the UI sees the
        # genuine Schwab payload, including null fields, metadata
        # updates, and anything we'd otherwise drop.
        if self._on_raw_message is not None:
            try:
                self._on_raw_message(msg)
            except Exception as exc:
                logger.debug("on_raw_message hook failed: %s", exc)
        try:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not content:
                # Probe what we got — log the keys of the first such msg so
                # we know what the library hands us.
                if isinstance(msg, dict) and self._tick_count == 0:
                    logger.info("schwab_stream: first msg keys=%s sample=%s",
                                list(msg.keys())[:8], str(msg)[:200])
                return
            self._last_tick_at = datetime.now(timezone.utc)
            for record in content:
                sym = record.get("key") or record.get("KEY")
                if not sym:
                    continue
                # v-stream-partial-merge-2026-05-01: Schwab sends PARTIAL
                # updates — most ticks for liquid symbols carry only
                # bid/ask, no LAST_PRICE. Previously we skipped those,
                # so PLTR/AAPL/etc looked frozen on the dashboard while
                # ORCL (which happens to send full snapshots more often)
                # was the only symbol updating. Now: merge the partial
                # update into whatever's in the cache, only updating the
                # fields that arrived in THIS tick.
                last = record.get("LAST_PRICE")
                if last is None:
                    last = record.get("3") or record.get("MARK") or record.get("CLOSE_PRICE")
                bid = record.get("BID_PRICE")
                if bid is None:
                    bid = record.get("1")
                ask = record.get("ASK_PRICE")
                if ask is None:
                    ask = record.get("2")
                # v-stream-netchange-2026-06-11: NET_CHANGE has been in
                # the subscription since v-stream-rich-fields-2026-05-01
                # but was discarded here. "18" is the LEVELONE_EQUITIES
                # numeric field id, same fallback pattern as "1"/"2"/"3".
                net_change = record.get("NET_CHANGE")
                if net_change is None:
                    net_change = record.get("18")

                # v-null-tick-noop-2026-05-01: per architectural
                # directive — if Schwab sends a tick that has NO price
                # fields (last, bid, ask all None — i.e. only a size
                # or metadata change like LAST_ID/EXCHANGE_NAME), it
                # is a NO-OP. We do not touch the cache, do not refresh
                # the timestamp, do not change anything the UI reads.
                # The user observes the symbol's last meaningful price
                # remain on screen unchanged — which is correct,
                # because nothing meaningful changed.
                if last is None and bid is None and ask is None:
                    continue
                if self._per_symbol_ticks is None:
                    self._per_symbol_ticks = {}
                self._per_symbol_ticks[sym] = self._per_symbol_ticks.get(sym, 0) + 1
                self._tick_count += 1

                # First tick logging — capture record shape so we know
                # what Schwab actually sends
                if self._tick_count == 1:
                    logger.info(
                        "schwab_stream: FIRST TICK sym=%s last=%s bid=%s ask=%s record_keys=%s",
                        sym, last, bid, ask, list(record.keys())[:15],
                    )
                # Periodic distribution log — shows which symbols are
                # actually getting traffic, not just the last one
                if self._tick_count % self._tick_log_every == 0:
                    top = sorted(self._per_symbol_ticks.items(),
                                 key=lambda x: -x[1])[:8]
                    logger.info(
                        "schwab_stream: %d ticks total. Top per-symbol: %s",
                        self._tick_count,
                        ", ".join(f"{s}:{n}" for s, n in top),
                    )

                # Canonical write — PriceBook handles partial-merge.
                try:
                    from core.price_book import PriceBook
                    pb = PriceBook.instance()
                    if pb is not None:
                        pb.apply_tick_sync(
                            sym,
                            last=last,
                            bid=bid,
                            ask=ask,
                            net_change=net_change,
                            source="stream",
                        )
                except Exception as exc:
                    logger.debug("stream pricebook write failed for %s: %s", sym, exc)
        except Exception as exc:
            # One bad tick must not kill the stream.
            logger.debug("stream _on_message error: %s", exc)

    # ── run loop ──────────────────────────────────────────────────────

    async def run(self) -> None:
        """Long-running supervised task. Connects, subscribes, processes
        ticks; reconnects with backoff on failure. Returns only when
        stop() is called."""
        if not self.is_available():
            logger.info("schwab_stream: not available, exiting")
            return

        # Capture the running loop NOW so the sync tick handler can
        # schedule cache writes against it deterministically.
        self._loop = asyncio.get_running_loop()
        logger.info("schwab_stream: captured event loop for cache writes")

        attempt = 0
        while not self._stop.is_set():
            try:
                # Build / rebuild the StreamClient against the engine's
                # authed Schwab client. The library expects the http
                # client + account hash.
                self._stream = _SchwabStreamClient(
                    self._client,
                    account_id=self._account_hash,
                )
                logger.info("schwab_stream: logging in")
                await self._stream.login()
                self._consecutive_failures = 0
                attempt = 0

                # Bind tick handler exactly once per connection.
                self._stream.add_level_one_equity_handler(self._on_message)
                self._handler_attached = True
                self._active_symbols = set()  # fresh connection → no active subs

                # Reconcile subscriptions and process messages until disconnect.
                # _serve() requests an EXPANDED set of fields beyond just
                # LAST_PRICE so the cache updates on every quote change,
                # not just on trade prints (which are sparse for quiet
                # symbols). v-stream-rich-fields-2026-05-01.
                await self._serve()

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._consecutive_failures += 1
                attempt += 1
                delay = self._RECONNECT_BACKOFF_SEC[
                    min(attempt - 1, len(self._RECONNECT_BACKOFF_SEC) - 1)
                ]
                logger.warning(
                    "schwab_stream: connection failed (attempt %d, failures=%d): %s. "
                    "Reconnecting in %ds.",
                    attempt, self._consecutive_failures, exc, delay,
                )
                if self._consecutive_failures >= self._UNHEALTHY_AFTER_FAILURES:
                    logger.warning(
                        "schwab_stream: unhealthy (%d consecutive failures). "
                        "Polling fallback will take over.",
                        self._consecutive_failures,
                    )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                    return  # stop signalled during backoff
                except asyncio.TimeoutError:
                    pass  # keep going, retry

    async def _serve(self) -> None:
        """Inner loop: reconcile subs, then handle messages forever."""
        # v-stream-coalesce-2026-05-01: rate-limit + coalesce the
        # subscribe/unsubscribe traffic so we don't blast Schwab with
        # bursty churn (which appeared to trigger an upstream drop /
        # rate-limit). Two new safeguards:
        #   - MAX_PER_BATCH = 10: cap subscriptions per reconcile pass.
        #     A 30-symbol watchlist gets staged across 3 passes (6 sec)
        #     instead of one giant burst.
        #   - PENDING_UNSUB_HOLDDOWN_SEC = 8: a symbol must be absent
        #     from desired for at least this long before we unsubscribe.
        #     Defends against a transient empty-set registration (e.g.
        #     position list briefly empty during a race) yanking the
        #     subscription out from under us, only to re-add it 2 sec
        #     later. The Schwab connection is happier with stable subs.
        MAX_PER_BATCH = 10
        PENDING_UNSUB_HOLDDOWN_SEC = 8.0
        pending_unsubs: dict[str, float] = {}

        async def _reconcile_periodically():
            while not self._stop.is_set():
                try:
                    async with self._desired_lock:
                        desired = set(self._desired_symbols)
                    now = asyncio.get_event_loop().time()

                    to_add = desired - self._active_symbols
                    naive_to_remove = self._active_symbols - desired

                    # Stage unsubs through the hold-down dict.
                    # Any symbol that just BECAME un-desired starts
                    # its hold-down timer; symbols still in desired
                    # cancel their timer.
                    for sym in naive_to_remove:
                        if sym not in pending_unsubs:
                            pending_unsubs[sym] = now
                    for sym in desired:
                        pending_unsubs.pop(sym, None)

                    # Only emit unsubs for symbols that have been pending
                    # for >= holddown.
                    to_remove = {
                        sym for sym, t0 in pending_unsubs.items()
                        if (now - t0) >= PENDING_UNSUB_HOLDDOWN_SEC
                    }

                    if to_add:
                        # Rich field set so we get every quote-change tick.
                        F = _SchwabStreamClient.LevelOneEquityFields
                        rich_fields = [
                            F.LAST_PRICE, F.BID_PRICE, F.ASK_PRICE,
                            F.MARK, F.NET_CHANGE, F.TOTAL_VOLUME,
                            F.HIGH_PRICE, F.LOW_PRICE, F.LAST_SIZE,
                            F.BID_SIZE, F.ASK_SIZE,
                        ]
                        rich_fields = [f for f in rich_fields if f is not None]
                        # Cap this batch — bigger sets get staged across
                        # the next few reconcile passes.
                        batch = list(to_add)[:MAX_PER_BATCH]
                        # v-stream-subs-vs-add-2026-05-04: Schwab's streamer
                        # protocol distinguishes SUBS (replace the entire
                        # subscription set) from ADDS (append to it).
                        # schwab-py exposes:
                        #   level_one_equity_subs() → SUBS (replace)
                        #   level_one_equity_add()  → ADDS (append)
                        # The coalescing reconciler stages the watchlist
                        # across multiple batches. If every batch calls
                        # _subs, each batch silently REPLACES the prior
                        # subscription — the LAST batch wins, all earlier
                        # symbols stop ticking. (Symptom observed in prod:
                        # only RIOT — the last batch — kept ticking;
                        # 30 other symbols frozen for 8+ minutes despite
                        # the stream being healthy.)
                        # Fix: use SUBS only for the first batch when there
                        # are no active symbols, and ADDS for every
                        # subsequent batch.
                        if not self._active_symbols:
                            await self._stream.level_one_equity_subs(
                                batch, fields=rich_fields,
                            )
                            sub_method = "subs"
                        else:
                            await self._stream.level_one_equity_add(
                                batch, fields=rich_fields,
                            )
                            sub_method = "add"
                        self._active_symbols = self._active_symbols | set(batch)
                        logger.info(
                            "schwab_stream: subscribed (%s) +%s (%d active, %d remaining queued, fields=%d)",
                            sub_method,
                            sorted(batch),
                            len(self._active_symbols),
                            max(0, len(to_add) - len(batch)),
                            len(rich_fields),
                        )
                    if to_remove:
                        await self._stream.level_one_equity_unsubs(list(to_remove))
                        for sym in to_remove:
                            pending_unsubs.pop(sym, None)
                        self._active_symbols = self._active_symbols - to_remove
                        logger.info(
                            "schwab_stream: unsubscribed -%s (held %ds before unsub)",
                            sorted(to_remove), int(PENDING_UNSUB_HOLDDOWN_SEC),
                        )
                except Exception as exc:
                    logger.debug("schwab_stream: reconcile error: %s", exc)
                await asyncio.sleep(2.0)

        # v-stream-watchdog-2026-05-08 / v-stream-watchdog-recover-2026-05-12:
        # tick-freshness watchdog. Original implementation hid silent
        # death because (a) any exception in the body was logged at
        # DEBUG and the loop swallowed it, (b) the task itself was
        # unsupervised, and (c) there was no way to tell from logs
        # whether the watchdog was even running. The 2026-05-12
        # incident drove 75 minutes blind during RTH with no watchdog
        # output at all. Replaced with the observable StreamWatchdog
        # + DeadMansSwitch from core.schwab_stream_watchdog. See
        # docs/incidents/2026-05-12-stream-watchdog.md.
        SILENT_DEATH_THRESHOLD_SEC = 120.0
        WATCHDOG_PERIOD_SEC = 30.0
        WATCHDOG_CONNECT_GRACE_SEC = 60.0
        # Dead-man's switch: if the watchdog stops emitting heartbeats
        # (counter doesn't advance) for this many seconds, exit(42).
        DEAD_MANS_SWITCH_SEC = 90.0
        connect_at = datetime.now(timezone.utc)
        # Reset the silent-death flag at the start of each _serve() call
        # so a previous death's signal doesn't immediately trip the new
        # connection.
        self._silent_death.clear()

        def _is_rth_now() -> bool:
            """ET regular trading hours: weekday 09:30–16:00 ET."""
            try:
                import pytz as _pytz
                _et = datetime.now(_pytz.timezone("America/New_York"))
            except Exception:
                return False
            if _et.weekday() >= 5:
                return False
            mins = _et.hour * 60 + _et.minute
            return 570 <= mins < 960  # 09:30 → 16:00

        from core.schwab_stream_watchdog import (
            StreamWatchdog, DeadMansSwitch, supervise_task,
        )

        # The new watchdog signals death by calling on_death, which
        # sets self._silent_death (preserving the existing race in
        # _serve's outer asyncio.wait).
        def _on_death() -> None:
            self._silent_death.set()

        watchdog_obj = StreamWatchdog(
            period_sec=WATCHDOG_PERIOD_SEC,
            threshold_sec=SILENT_DEATH_THRESHOLD_SEC,
            connect_at=connect_at,
            connect_grace_sec=WATCHDOG_CONNECT_GRACE_SEC,
            get_last_tick=lambda: self._last_tick_at,
            on_death=_on_death,
            stop_event=self._stop,
            is_rth_now=_is_rth_now,
        )
        dms = DeadMansSwitch(
            check_period_sec=WATCHDOG_PERIOD_SEC,
            dead_after_sec=DEAD_MANS_SWITCH_SEC,
            get_heartbeat_count=lambda: watchdog_obj.heartbeat_count,
            stop_event=self._stop,
        )

        # handle_message blocks the coroutine that calls it indefinitely;
        # run reconciliation and the watchdog alongside it.
        reconciler = asyncio.create_task(_reconcile_periodically(),
                                          name="schwab_stream_reconciler")
        watchdog = asyncio.create_task(watchdog_obj.run(),
                                        name="schwab_stream_watchdog")
        dms_task = asyncio.create_task(dms.run(),
                                        name="schwab_stream_dead_mans_switch")
        # Supervise both: if either dies unexpectedly we log CRITICAL
        # with traceback so a silent task cancellation cannot recur.
        supervise_task(watchdog, "schwab_stream_watchdog")
        supervise_task(dms_task, "schwab_stream_dead_mans_switch")
        try:
            while not self._stop.is_set():
                # Race the message pump against the silent-death signal.
                # handle_message() can block forever when the underlying
                # websocket has silently disconnected; the watchdog will
                # set self._silent_death and we raise out of _serve() so
                # run()'s reconnect loop fires.
                pump_task = asyncio.create_task(
                    self._stream.handle_message(),
                    name="schwab_stream_pump",
                )
                death_task = asyncio.create_task(
                    self._silent_death.wait(),
                    name="schwab_stream_death_wait",
                )
                done, pending = await asyncio.wait(
                    {pump_task, death_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
                if death_task in done:
                    # Drain any pending pump exception, then bubble.
                    raise RuntimeError(
                        "schwab_stream: silent death detected, reconnecting"
                    )
                # pump_task completed normally — propagate any exception
                # from handle_message itself.
                exc = pump_task.exception()
                if exc is not None:
                    raise exc
        finally:
            watchdog.cancel()
            try:
                await asyncio.wait_for(watchdog, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
            dms_task.cancel()
            try:
                await asyncio.wait_for(dms_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
            reconciler.cancel()
            try:
                await asyncio.wait_for(reconciler, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
