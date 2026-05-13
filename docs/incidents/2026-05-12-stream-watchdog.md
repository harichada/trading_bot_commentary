# 2026-05-12 — Stream Watchdog Failed to Recover from Silent Death

## Symptom

- Bot restarted at **14:30:09 ET**, stream connected and subscribed at 14:31:33.
- Last `schwab_stream: NNN ticks total` log line at **14:46:29** (12,800 ticks).
- From 14:46:30 to 16:01 (when operator queried), **75 minutes elapsed with**:
  - Zero ticks from any of the 35 subscribed symbols.
  - Zero log lines from the `schwab_stream` module.
  - Zero reconnect attempts (`connection failed`, `logging in`, `subscribed` all absent).
  - Zero `silent_death` or watchdog activity logs.
- `health_gate` correctly blocked all 21 live signals with `reason=stream_dead`.
- Strategy/screener/account-info paths kept running; only the streaming subsystem went silent.

This is a watchdog failure, not a stream failure. The stream went silent
(expected — silent websocket disconnects happen); the watchdog did not
recover from it (not expected — that's its sole purpose).

## Code under examination

`core/schwab_stream.py:447-517` — the `_watchdog()` coroutine defined inside `_serve()`.
`core/schwab_stream.py:519-565` — the `_serve()` asyncio race that consumes
`self._silent_death`.

Relevant signaling chain:

```
_watchdog() detects stale ticks (line 506)
  → self._silent_death.set() (line 512)
  → death_task in _serve()'s asyncio.wait completes (line 540)
  → RuntimeError raised (line 552)
  → _serve()'s finally cancels watchdog + reconciler (lines 561-569)
  → run()'s outer try/except catches (line 316)
  → backoff sleep, then reconnect (line 334)
```

## Diagnosis against the three candidate root causes

### Candidate 1 — Watchdog task silently cancelled (no exception surfaced)

**Evidence for**: zero log output from the watchdog over 75 minutes during RTH.
A cancelled task would explain the total silence.

**Evidence against**: the only cancellation site is `_serve()`'s `finally`
block (line 561), which only runs when `_serve()` exits. `_serve()` exits
only when the outer `while not self._stop.is_set():` loop terminates, which
requires either `_stop.set()` (engine shutdown) or an exception escaping the
race. Neither happened — the bot kept running, no reconnect log line
appeared, and the engine never entered shutdown.

**Verdict**: not directly provable. Possible but no positive evidence.

### Candidate 2 — Watchdog alive but stuck on a future

**Evidence for**: the only `await` in the watchdog body is `asyncio.sleep(30)`,
which is bounded and cannot get stuck. There's no other awaitable to block on.

**Evidence against**: same — bounded. If the watchdog were running, it would
have hit the `if age > SILENT_DEATH_THRESHOLD_SEC:` branch at line 506
within 30 seconds of the tick gap exceeding 120 s (so by 14:48:30 at the
latest), and emitted the WARNING at line 507-511. **It did not.**

**Verdict**: ruled out. The watchdog body cannot be stuck on a future
because it doesn't have one to be stuck on.

### Candidate 3 — `asyncio.wait(FIRST_COMPLETED)` swallowed completion

**Evidence for**: the race at line 540 awaits two tasks; if `pump_task`
returns normally each iteration (it does — `handle_message()` returns per
batch of ticks during healthy operation), the loop reschedules a fresh
`death_task` each iteration. There's an in-principle race: the watchdog
could set `_silent_death` between the previous iteration's `await asyncio.wait`
returning and the new iteration creating its `death_task`. The new
`death_task = asyncio.create_task(self._silent_death.wait())` should still
fire immediately on its first `await` because the Event is already set —
but this depends on `asyncio.Event.wait()` correctly observing the prior
`.set()` across task boundaries. In modern asyncio this is reliable, so
this candidate would require an asyncio bug, which is unlikely.

**Evidence against**: same — `asyncio.Event.wait()` returns immediately if
the event is already set. No window for the signal to be lost.

**Verdict**: very low likelihood.

### The actual most-likely root cause

Looking at the exception handler at lines 514-517:

```python
except asyncio.CancelledError:
    raise
except Exception as exc:
    logger.debug("schwab_stream: watchdog error: %s", exc)
```

`logger.debug(...)` writes at DEBUG level. The default log level in this
project is **INFO** (see `core/config.py` logging setup). **Any exception
that fires inside the watchdog body is logged at a level that is filtered
out**, then the `while` loop iterates. The watchdog becomes a hot loop that
spins on the same failing iteration, emitting one invisible DEBUG line per
30-s tick, never reaching the `_silent_death.set()` call.

We cannot prove this from the current log because — by definition — it
emits no INFO/WARNING/ERROR-level output. The fact that **we cannot tell
whether the watchdog ran at all is itself the bug.** The watchdog is
unobservable in any failure mode short of the happy path firing the WARNING
at line 507.

**Conclusion**: insufficient runtime evidence to discriminate between
candidates 1, 3, or the hidden-exception scenario. **What we know for
certain is that the watchdog is unobservable, untimed, and unsupervised.**
The fix must eliminate the entire class.

## Fix posture

The instrumentation requirement is non-negotiable: heartbeat log every
30 s even on the happy path, supervised task with `add_done_callback`
that logs traceback at CRITICAL on exit, dead-man's-switch sibling that
calls `os._exit(42)` when the heartbeat goes stale. With those three
in place, all three candidate failure modes become impossible-to-be-silent:

- Candidate 1 (silent cancellation): `add_done_callback` logs CRITICAL.
- Candidate 2 (stuck await): no await > 30 s; dead-man's-switch fires at 90 s.
- Candidate 3 (lost signal): heartbeat shows tick age every 30 s regardless.
- Hidden exception: callback logs traceback; dead-man's-switch fires.

Move the pure decision logic (`is_stream_silently_dead`) out of the
watchdog body and into `core/stream_health.py` so it's unit-testable in
isolation with table-driven inputs. The watchdog becomes a thin shell:
sleep, log heartbeat, call pure function, act on result.
