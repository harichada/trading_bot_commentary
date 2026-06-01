# Bot Monitor

Local agent that polls the trading bot and produces structured
Markdown reports for Claude to consume on demand.

## Quick start

```bash
# Start the monitor in the background (60s poll)
tools/start_monitor.sh

# Stop it
tools/start_monitor.sh stop

# Status
tools/start_monitor.sh status
```

Reports land at `docs/monitor/latest.md`. Tell Claude "check the
monitor report" or "what's the bot status" and Claude will read
this file.

Timestamped history goes to `docs/monitor/<YYYY-MM-DD>/<HH-MM>.md`
so you can audit retroactively.

## What it captures

Each report has six sections:

1. **Process health** — bot PID, uptime, Schwab token TTL, stream
   heartbeat freshness, mode (live/simulation).
2. **P&L** — current day P&L, peak/trough in window, distance to
   daily-loss circuit breaker.
3. **Positions** — open positions table.
4. **Signal activity** — counts by strategy of signals fired,
   accepted by router, blocked at router (with reasons).
5. **Direction-gate skips** — table of recent signals blocked by
   the direction reader (with phase + direction score).
6. **R:R audit warnings** — any signal whose actual R:R fell below
   the 1.8 floor. **Any entry indicates a strategy producing tight-
   target signals — investigate.**
7. **Errors** — log entries at ERROR or CRITICAL level.
8. **Market regime** — current indices snapshot via API.

## Architecture

PURE OBSERVER. The monitor never writes to:

* `trading_bot.log`
* `trading_state.json`
* `trading_brain.json`
* `trading_commentary.json`
* `Config().yaml`
* The bot's REST API (no PUT/POST/DELETE)

Only reads:

* `trading_bot.log` (tail last hour by default)
* `trading_state.json`
* `token_1.json`
* `GET /api/status`
* `GET /api/positions/db`
* `GET /api/market-indices`

The bot is a black box from the monitor's perspective. Even if
the monitor crashes or runs out of memory, the bot is unaffected.

## CLI flags

```bash
python tools/bot_monitor.py [flags]

  --interval SECONDS   poll interval (default 60)
  --window SECONDS     log window to scan (default 3600 = 1h)
  --once               generate one report and exit
  --quiet              don't print to stdout (file only)
  --no-history         skip timestamped history files
```

## How Claude reads it

In a chat with Claude, say something like:

* "Check the monitor report"
* "How's the bot doing?"
* "Anything in the latest status?"

Claude will open `docs/monitor/latest.md` and summarize what's
relevant. The Markdown is structured so Claude can extract specific
sections (e.g., "just the warnings") without re-parsing.

## Operational notes

* The monitor consumes the same port (9000) the dashboard uses,
  via REST API calls. If the bot is down, the API calls fail
  gracefully and the report says "Bot: NOT RUNNING."
* The monitor's own logs go to `/tmp/bot_monitor.out`. Truncate
  periodically; it grows unbounded.
* History files in `docs/monitor/<date>/` accumulate — clean up
  with `find docs/monitor -mindepth 1 -mtime +30 -delete` if needed.
