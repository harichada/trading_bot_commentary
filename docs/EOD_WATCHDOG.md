# EOD Watchdog — Architecture, Specifications, and Runbooks

## 1. Architecture

### Task Dependency Diagram

```
FastAPI lifespan()
    |
    +-- asyncio.create_task(_watchdog_heartbeat)    [MONITOR]
    |       |
    |       +-- Monitors trading loop health (90s stale threshold)
    |       +-- Monitors EOD watchdog health (auto-restart if dead)
    |       +-- Pings systemd every 30s
    |
    +-- asyncio.create_task(_startup_reconcile)     [STARTUP]
    +-- asyncio.create_task(_prewarm_volume_cache)  [STARTUP]
    +-- asyncio.create_task(_delayed_auto_start)    [STARTUP]
            |
            +-- live_trader.start()
                    |
                    +-- asyncio.create_task(_trading_loop)    [TRADING]
                    |       |
                    |       +-- Scans, entries, LLM calls, standdowns
                    |       +-- Checks _trading_halted flag (yields if set)
                    |       +-- NO EOD logic — fully delegated to watchdog
                    |
                    +-- asyncio.create_task(_eod_watchdog)    [EOD GUARDIAN]
                            |
                            +-- Independent wall-clock scheduler
                            +-- Cannot be blocked by trading loop
                            +-- Fires at exact times regardless of system state
```

### Key Design Principle

The EOD watchdog and the trading loop share NO control flow. They communicate
through exactly one flag: `_trading_halted` (bool). The watchdog sets it; the
trading loop reads it. There are no locks, no shared queues, no function calls
between them during the critical EOD window.

### Data Flow

```
_eod_watchdog                          _trading_loop
     |                                      |
     |  sets _trading_halted = True         |
     |  --------------------------------->  |  reads _trading_halted
     |                                      |  -> blocks _enter_positions
     |                                      |  -> blocks _intraday_scan_and_enter
     |                                      |  -> yields control (sleep 5s loop)
     |                                      |
     |  calls _eod_close()                  |
     |  (shared _position_lock)             |
     |                                      |
     |  After 4 PM:                         |  After 4 PM:
     |  sleeps until next day               |  resets _trading_halted
     |                                      |  sleeps until next day
```

## 2. EOD Timeline

```
Time (ET)    Phase              Action
---------    -----              ------
  3:45 PM    CIRCUIT BREAKER    _trading_halted = True
                                - _enter_positions returns immediately
                                - _intraday_scan_and_enter returns immediately
                                - Alert sent: "EOD Circuit Breaker"
                                - UI status: "eod_closing"

  3:50 PM    PRIMARY CLOSE      _eod_close() with 60s hard timeout
                                - Cancel all stop orders per symbol
                                - Submit market cover orders
                                - Verify with broker (3 retries)
                                - Fallback: Alpaca DELETE API
                                - Performance snapshot recorded
                                - LLM debrief (15s timeout, non-critical)

  3:55 PM    FORCE CLOSE        Per-symbol Alpaca DELETE /v2/positions/{sym}
                                - Cancel orders, wait 1s, close position
                                - Alert: "EOD FORCE CLOSE"
                                - Each symbol attempted independently

  4:00 PM    EMERGENCY          Mark remaining for close-on-open
                                - position.close_on_open = True
                                - State saved to PostgreSQL
                                - Alert: "EMERGENCY: Positions Open After Hours"

  4:05 PM    AUDIT              Verify broker state directly
                                - GET /v2/positions from Alpaca
                                - Alert if any positions remain
                                - Log broker confirmation if clean
                                - Sleep until tomorrow 6 AM ET
```

## 3. Timeout Specifications

| Operation                  | Timeout | Enforcement         | On Timeout                          |
|---------------------------|---------|---------------------|-------------------------------------|
| _eod_close() primary      | 60s     | asyncio.wait_for    | Escalate to force close at 3:55     |
| LLM eod_debrief           | 15s     | asyncio.wait_for    | Skip debrief, log warning           |
| Broker order cancel       | 10s     | asyncio.to_thread   | Log error, continue to next symbol  |
| Broker position close     | 10s     | asyncio.to_thread   | Log error, retry at next phase      |
| Force close per symbol    | 10s     | asyncio.to_thread   | Log error, mark close-on-open       |
| Broker position verify    | 10s     | asyncio.to_thread   | Log error, send alert               |
| Watchdog heartbeat cycle  | 30s     | asyncio.sleep       | systemd kills process at 120s       |
| Trading loop stale check  | 90s     | monotonic time diff  | Skip systemd ping → restart at 120s |
| EOD watchdog error retry  | 10s     | asyncio.sleep       | Retry entire watchdog loop          |

## 4. Watchdog Specifications

### _watchdog_heartbeat (systemd)

| Check                     | Trigger                          | Action                              |
|--------------------------|----------------------------------|-------------------------------------|
| Trading loop stale       | No iteration in 90s (market hrs) | Skip WATCHDOG=1 → systemd kills at 120s |
| EOD watchdog dead        | task.done() == True              | Auto-restart via create_task        |
| EOD watchdog exception   | task.exception() != None         | Alert + auto-restart                |
| EOD watchdog cancelled   | task.cancelled() == True         | Alert + auto-restart                |

### _eod_watchdog (self-monitoring)

| Check                     | Trigger                          | Action                              |
|--------------------------|----------------------------------|-------------------------------------|
| Unhandled exception      | Any exception in main loop       | Log, alert, sleep 10s, retry        |
| CancelledError           | Task cancelled (stop() called)   | Re-raise (clean shutdown)           |
| Weekend detection        | weekday() >= 5                   | Sleep until Monday 6 AM             |
| Day transition           | today != _last_date              | Reset _trading_halted               |

### /api/health endpoint

Reports:
- `eod_watchdog`: `"running"` | `"dead"` | `"not_started"`
- `trading_halted`: `true` | `false`

## 5. Circuit Breaker Enforcement Points

```python
# 1. Gap fade entries
async def _enter_positions(self, ...):
    if self._trading_halted:
        self._add_message('system', 'Entry blocked — EOD circuit breaker active')
        return

# 2. Intraday entries
async def _intraday_scan_and_enter(self):
    if self._trading_halted:
        return

# 3. Trading loop main body
if self._trading_halted:
    if self.engine.positions:
        await asyncio.sleep(5)  # yield — watchdog is closing
        continue
```

## 6. SLA Guarantees

| Metric                         | Target   | Enforcement                        |
|-------------------------------|----------|------------------------------------|
| EOD close execution rate       | 100%     | Independent task, auto-restart     |
| Circuit breaker activation     | 3:45 PM ±5s | Wall-clock check every 5s      |
| Primary close initiation       | 3:50 PM ±5s | Wall-clock check every 5s      |
| Force close initiation         | 3:55 PM ±5s | Wall-clock check every 5s      |
| Max time positions unprotected | 60s      | _eod_close timeout                 |
| Watchdog self-heal time        | 30s      | _watchdog_heartbeat cycle          |
| Alert delivery on failure      | Every phase | AlertNotifier at each escalation |

## 7. Recovery Procedures

### Procedure A: EOD Watchdog Not Running

**Detection**: `/api/health` shows `eod_watchdog: dead` or `not_started`

**Automatic recovery**: `_watchdog_heartbeat` detects dead task every 30s and
restarts it automatically. No manual action needed.

**Manual recovery** (if systemd watchdog also failed):
```bash
# Restart the container — all tasks re-created on boot
cd ~/claude/infra && make restart-prod

# Verify
sleep 15
curl -s http://localhost:8003/api/health | python3 -m json.tool
# Confirm eod_watchdog: "running"
```

### Procedure B: _eod_close Timed Out

**Detection**: Alert "EOD Close Timeout" + log "TIMED OUT after 60s"

**Automatic recovery**: Watchdog escalates to force close at 3:55 PM using
Alpaca's DELETE /v2/positions/{symbol} API.

**Manual recovery** (if force close also failed):
```bash
# Check what's still open
curl -s -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
     https://paper-api.alpaca.markets/v2/positions | python3 -m json.tool

# Force close everything via Alpaca API
curl -s -X DELETE \
     -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
     https://paper-api.alpaca.markets/v2/positions
```

### Procedure C: Positions Open After Hours

**Detection**: Alert "EMERGENCY: Positions Open After Hours"

**Automatic recovery**: Positions marked `close_on_open = True`. On next
trading day, the trading loop's close-on-open handler submits market orders
within the first 5 minutes of the session.

**Manual recovery**:
```bash
# Option 1: Close via Alpaca dashboard
# https://app.alpaca.markets/paper/dashboard/overview

# Option 2: Close via API
curl -s -X DELETE \
     -H "APCA-API-KEY-ID: $ALPACA_API_KEY" \
     -H "APCA-API-SECRET-KEY: $ALPACA_SECRET_KEY" \
     https://paper-api.alpaca.markets/v2/positions/{SYMBOL}

# Option 3: Wait for close-on-open (automatic next market day)
```

### Procedure D: Broker API Down

**Detection**: Alert "EOD Watchdog Error" + repeated force close exceptions

**What happens**: Each phase catches exceptions independently. If broker API
is completely down:
- 3:50: _eod_close fails → logged, escalates
- 3:55: Force close fails → logged, escalates
- 4:00: Positions marked close-on-open → state saved locally
- 4:05: Audit fails → alert sent

**Manual recovery**: Positions are protected by broker-side stop orders
(placed when position was opened). If the broker is down, those stops are
also not executable — this is a broker-level incident, not an engine issue.
Contact Alpaca support.

### Procedure E: Time Drift

**Detection**: EOD phases fire at wrong times

**Prevention**: The watchdog uses `datetime.now(ET)` which relies on system
time. Docker containers inherit host time.

**Fix**:
```bash
# Check system time
date
timedatectl status

# Force NTP sync
sudo timedatectl set-ntp true
sudo systemctl restart systemd-timesyncd

# Verify
ntpq -p  # or timedatectl timesync-status
```

### Procedure F: Watchdog + Trading Loop Both Dead

**Detection**: systemd kills the process (WatchdogSec=120)

**Automatic recovery**: systemd `Restart=on-failure` restarts the container.
On boot, lifespan creates _watchdog_heartbeat, which monitors everything.
If auto_start is True and within trading hours, trading resumes automatically.

**Verify**:
```bash
systemctl status gap-fade.service
journalctl -u gap-fade.service --since "5 minutes ago"
```

## 8. Incident Response Checklist

### EOD Close Failure — During Market Hours

```
1. [ ] Check /api/health — is eod_watchdog running?
2. [ ] Check Telegram/alerts — what phase failed?
3. [ ] Check broker directly:
       curl alpaca /v2/positions — are positions still open?
4. [ ] If positions open AND market still open:
       curl -X DELETE alpaca /v2/positions  (close all)
5. [ ] If market closed:
       Positions will be handled by close-on-open tomorrow
6. [ ] Review logs:
       make logs-prod | grep "EOD WATCHDOG"
7. [ ] Post-incident:
       - What time did circuit breaker engage?
       - What time did primary close attempt?
       - What error caused escalation?
       - Was the watchdog task alive?
```

### EOD Watchdog Task Died

```
1. [ ] Check /api/health — eod_watchdog: "dead"
2. [ ] _watchdog_heartbeat should auto-restart within 30s
3. [ ] If not restarted:
       make restart-prod
4. [ ] Verify:
       curl /api/health | grep eod_watchdog
5. [ ] Review logs for root cause:
       make logs-prod | grep "EOD WATCHDOG"
```

### Exchange Down / Broker Outage

```
1. [ ] Cannot close positions — this is expected during outage
2. [ ] Positions are marked close-on-open automatically
3. [ ] Broker-side stop orders provide some protection
4. [ ] When exchange reopens:
       - close-on-open handler fires automatically
       - Verify via /api/state
5. [ ] Contact Alpaca support if positions remain
6. [ ] Document the incident
```

## 9. Testing

### Running Tests

```bash
# Full EOD watchdog test suite (21 tests)
python -m pytest test_eod_watchdog.py -v

# Specific test categories
python -m pytest test_eod_watchdog.py -k "CircuitBreaker" -v
python -m pytest test_eod_watchdog.py -k "Independence" -v
python -m pytest test_eod_watchdog.py -k "Recovery" -v
python -m pytest test_eod_watchdog.py -k "FullEODSequence" -v
```

### Test Coverage Matrix

| Test Class                    | Tests | What's Validated                                    |
|------------------------------|-------|-----------------------------------------------------|
| TestCircuitBreaker           | 2     | Entries blocked at 3:45 (gap + intraday)            |
| TestEODWatchdogPhases        | 4     | 3:45, 3:50, 3:55 phases; timeout escalation         |
| TestWatchdogIndependence     | 2     | Fires during standdown; fires during LLM block      |
| TestWatchdogRecovery         | 2     | Survives exceptions; dead task detected              |
| TestWeekendHandling          | 1     | Sleeps through weekends                              |
| TestNoPositions              | 1     | Graceful no-op, snapshot still recorded              |
| TestHealthEndpoint           | 2     | Reports running/dead watchdog status                 |
| TestConcurrentExecution      | 2     | 100/100 iterations; tasks truly independent          |
| TestFullEODSequence          | 1     | End-to-end 3:45 → 4:00 sequence                     |
| TestTradingLoopCleanup       | 4     | _did_eod removed; circuit breaker in all entry points|

## 10. Monitoring Commands

```bash
# Health check (includes watchdog status)
curl -s http://localhost:8003/api/health | python3 -m json.tool

# Expected output:
# {
#   "status": "ok",
#   "trader_status": "trading",
#   "eod_watchdog": "running",
#   "trading_halted": false,
#   ...
# }

# Check EOD-related logs
make logs-prod 2>&1 | grep "EOD WATCHDOG"

# Check circuit breaker status
curl -s http://localhost:8003/api/health | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(f'Halted: {d[\"trading_halted\"]}, Watchdog: {d[\"eod_watchdog\"]}')"

# Verify positions after EOD
curl -s http://localhost:8003/api/state | python3 -m json.tool | grep -A5 '"positions"'
```
