# Daily-loss circuit — pause NEW entries (v-daily-loss-pause-entries-2026-11-05)

Owner rule (2026-11-05): the daily-loss circuit only pauses **new entries**.
It never stops analysis, position monitoring, exits / stops / re-brackets,
position sync, P&L refresh, the heartbeat, decision logging or the dashboard.

## What trips it
| Circuit | Where | Threshold | Measure |
|---|---|---|---|
| MAX_DAILY_LOSS limit | `risk/manager.py` `daily_loss_limit_breach()` | `pnl <= -MAX_DAILY_LOSS x effective_balance` | see "measure" below |
| 5% emergency stop | `core/engine.py` `_check_emergency_daily_loss()` (called from `_analysis_loop_body` every cycle; P&L refreshed every 5th) | `pnl < 0 and |pnl| > 5% x effective_balance` | same |

## Flags
| Key (yaml) | Env override | Default | Meaning |
|---|---|---|---|
| `risk.daily_loss_action` | `DAILY_LOSS_ACTION` | `pause_entries` | `pause_entries`: latch "entries paused" for the rest of the ET day; nothing closed, all loops run. `halt`: legacy — emergency stop closes bot-managed positions (never hands-off) and sets `is_running=False`; MAX_DAILY_LOSS makes the analysis loop skip analysis. Unknown values -> `pause_entries`. |
| `risk.daily_loss_bot_managed_only` | `DAILY_LOSS_BOT_MANAGED_ONLY` | `false` | When true, both circuits measure **bot-managed P&L only** (`engine._compute_bot_daily_pnl`: today's realized bot trades + unrealized of open `managed_by_bot` positions, excluding `trading.hands_off_denylist`, external, manually-managed and long-term positions), even if `ENABLE_BOT_ONLY_PNL_CIRCUIT` is false. When false, the pre-existing selector below decides, unchanged. |
| `trading.enable_bot_only_pnl_circuit` (pre-existing) | `ENABLE_BOT_ONLY_PNL_CIRCUIT` | `true` | Pre-existing measure selector (bot-only vs account-wide Schwab day P&L). Untouched. |

Env beats yaml (same convention as `ENABLE_BOT_ONLY_PNL_CIRCUIT`, `DAY_TRADE_LIVE_ENTRIES_ENABLED`).

## Pause semantics
* Latched per ET date in `DailyLossGate` (`risk/daily_loss_gate.py`), mirrored to
  `daily_loss_pause.json` (cwd, gitignored) so a same-day restart starts paused.
  It clears itself at the ET date rollover. To clear intraday on purpose: stop
  the bot, delete `daily_loss_pause.json`, start (it re-trips if still breached).
* Blocked: everything through `_process_signal_with_commentary` (long, short,
  day_trade, mean_rev, ORB, swing, adds / pyramids; audit
  `reason=daily_loss_entries_paused`) and `_execute_real_trade` (defense in
  depth; covers retry paths).
* Never touched: exits, stops, OCO / re-brackets, position sync, P&L refresh,
  heartbeat, consecutive_losses state and its clearing, `day_trade_live_entries_enabled`,
  LIVE flags, hands-off positions (HQGE, SPCX).

## Observability
`GET /api/status` adds `entries_paused`, `entries_paused_reason`
(`emergency_stop` | `daily_loss_limit`), `entries_paused_since`,
`daily_loss_action`, `daily_loss_measure` (`account` | `bot_managed`),
`circuit_halted`, `circuit_halt_reason`.

`scripts/bot_watchdog.sh`: `entries_paused: true` is healthy (`health ok ...
entries_paused=<reason>`); `circuit_halted: true` (legacy halt) is logged as a
deliberate halt and is **not** restarted (no restart-budget burn).
