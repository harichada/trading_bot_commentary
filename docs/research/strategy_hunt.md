# Strategy hunt

Research and operations notes for the offline strategy-fit pipeline
(`research/strategy_hunt/`) and the engine-side gate that consults its
output. This file is expected to grow as later tasks in the
strategy-hunt-full-system project land (fit-lane adapters, swing
lanes, promotion tooling); for now it covers the two things Task 29
adds: the nightly scheduling of the offline hunt and the flags that
control the live engine gate.

## Scheduling

The strategy hunt (`research.strategy_hunt`) runs offline, outside the
trading engine process, and writes its snapshot to
`data/strategy_hunt/fit/strategy_fit.json` (path configurable via
`STRATEGY_FIT_PATH`). It is scheduled to run nightly after the close,
via crontab, on this host:

```
0 22 * * 1-5 cd /home/nvidia/claude/grok/trading_bot_commentary && set -a && . ./.env && set +a && /home/nvidia/anaconda3/envs/trading-bot/bin/python -m research.strategy_hunt all --workers 12 >> data/strategy_hunt/cron.log 2>&1
```

Runs 22:00 local, Monday–Friday. It sources `.env` for API
credentials, then invokes the hunt's `all` subcommand across 12
worker processes, appending stdout/stderr to
`data/strategy_hunt/cron.log`. Installing this line is an operator
step (`crontab -e`); it is not applied automatically by any code
change.

`GET /api/strategy-fit` (read-only) and the "Strategy fit" section of
`tools/bot_monitor.py`'s report both read the snapshot this job
writes, so a missed or failed cron run shows up as a stale or
`_unavailable_` reading in the morning monitor check rather than
silently going unnoticed.

## Engine gate

The strategy-fit gate (`core/strategy_fit_gate.py`, wired into
`core/engine.py` right after the regime gate) checks each governed
signal's (symbol, strategy-family, side) verdict against the
strategy-fit snapshot before letting it proceed. It is off by default
everywhere it is currently wired — see
`docs/BOT_OPERATIONS_MANUAL.md` § "Strategy fit gate" for the
pipeline position, audit line format, and fail-closed behavior.

| Flag | Default | Purpose |
|---|---|---|
| `ENABLE_STRATEGY_FIT_GATE` | `False` | Master switch. Gate is fully inert (no audit lines, no blocking) until this is `True`. |
| `STRATEGY_FIT_SHADOW` | `True` | When the gate is enabled, run it in shadow (log the decision, never block a live entry). Must be explicitly set `False` to let the gate skip signals. |
| `STRATEGY_FIT_PATH` | `data/strategy_hunt/fit/strategy_fit.json` | Path to the snapshot file the gate and `/api/strategy-fit` read. |
| `STRATEGY_FIT_MAX_AGE_DAYS` | `7` | Snapshot age (days) beyond which it is treated as stale. |
| `FIT_LANES_ENABLED` | `""` (empty) | Comma-separated family names to register live strategy-fit lane adapters for. No lanes registered by default. |
| `FIT_LANE_LIVE_ENTRIES_ENABLED` | `""` (empty) | Comma-separated family names allowed to place LIVE entries via their fit lane; a family absent here is shadow-only even if present in `FIT_LANES_ENABLED`. |
| `SWING_ALLOW_OVERNIGHT` | `False` | Allow swing-lane positions to hold across sessions instead of forced flat by EOD. |
| `SWING_MAX_HOLD_SESSIONS` | `10` | Maximum sessions a swing-lane position may be held before a forced exit. |

All seven flags follow the config manager's env-var-then-yaml-then-default
precedence (`env ENABLE_STRATEGY_FIT_GATE=1`, yaml
`trading.enable_strategy_fit_gate: true`, etc. — see
`core/config.py`, search `v-strategy-fit-2026-09-25`).
