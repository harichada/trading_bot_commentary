# Trading Safety Rules — MANDATORY

These rules are non-negotiable. Any code change touching order execution,
stop losses, or position management MUST pass all checks below.

## Stop Orders
- **NEVER use stop_limit for protective stops.** Use stop (market) only.
- Stop-limit orders can fail to fill when price gaps through the limit.
- A stop that doesn't execute is worse than no stop — it gives false safety.

## Signal/Data Access
- **NEVER use dict["key"] for optional fields.** Use dict.get("key", default).
- If a signal dict may or may not contain a field, use .get() with a fallback.
- Log messages must never crash the trading loop.

## Error Handling in Trading Loop
- **NEVER let a logging/display error crash the trading loop.**
- Wrap all non-critical operations (logging, UI updates, metrics) in try/except.
- The trading loop must continue even if cosmetic operations fail.

## Reconciliation
- When removing orphaned positions, ALWAYS look up the exit fill price.
- Record the trade with actual P&L — never silently delete positions.

## Pre-Deployment Checklist
Before any change to order execution code:
- [ ] Stop orders use type='stop' (market), not 'stop_limit'
- [ ] All dict accesses on signal/tick data use .get()
- [ ] Error in logging/display cannot crash the trading loop
- [ ] Test: what happens if price gaps 5% in one tick?
- [ ] Test: what happens if broker API returns unexpected data?
- [ ] Test: what happens if this code throws — does the position stay protected?
