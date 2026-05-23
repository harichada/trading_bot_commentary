# Tuesday 2026-05-26 — Quarter-Size Live Launch

**Decision**: Option B from the 2026-05-23 walk-forward review. Going live at
**quarter size** to accelerate learning, paired with tight circuit breakers
to cap the downside.

---

## What changed this weekend

All in `Config().yaml`, `core/engine.py`, `core/config.py`, `risk/manager.py`:

| Change | Setting | Value |
|---|---|---|
| Live-launch global size dial | `live_size_multiplier` | **0.25** (quarter size) |
| Mean-rev strategy multiplier | `strategy_size_multipliers.mean_reversion` | 0.5 (half-size, N<20) |
| Effective news strategy size | (composed) | 25% of normal |
| Effective mean-rev size | (composed) | 12.5% of normal |
| Daily loss cap | `max_daily_loss` | **0.01** (1% account ≈ \$290) |
| Breakout strategy | `enable_breakout_long` | **false** (PF 1.14 / N=2) |
| Opening blackout | hard block 09:30–09:34 | no bypass |
| Late-day blackout | `late_entry_cutoff_hour:minute` | **15:30 ET** |
| Token TTL warning | `core/token_health.py` | logs at startup; deploy.sh aborts if <8h |

The chop-zone veto was attempted then reverted (corrected data showed
break-even PF 1.03, not -PF 0.85 from the original report).

---

## Tuesday morning runbook

### 09:00 ET — Pre-flight

```bash
cd /home/nvidia/claude/trading_bot_commentary
./deploy.sh
```

`deploy.sh` validates:
- All 6 safety config values match the launch plan
- Schwab refresh-token has ≥8h of life remaining (your token expires
  Wednesday 2026-05-27 14:57)

If any check fails, **fix it before continuing** — do not bypass.

### 09:15 ET — Dashboard
- Open http://localhost:8000 in browser
- Verify the dashboard loads and shows live mode = OFF (sim)
- Verify news feed is populating

### 09:30 ET — Flip to live
- In the dashboard, toggle trading mode to **LIVE**
  (Settings → Trading Mode, or whatever the UI calls it)
- Confirm the toggle succeeded (commentary line should say so)
- **Do not** click anything else until you see actual price quotes flowing

### 09:30–09:34 ET — Opening blackout
- Bot will REFUSE all signals in this window (hard block)
- Expected log: `engine_decision component=early_session action=skip reason=first_5min_hard_block`
- This is by design — first 5 min lost \$247 last week

### 09:35 ET onward — Active trading
- Bot may take entries
- News strategy enters at ~25% of normal size
- Mean-rev strategy enters at ~12.5% of normal size
- breakout strategy is OFF — should see no breakout entries

### 15:30 ET — Late-day blackout
- Bot will REFUSE all NEW signals
- Existing positions stay open (Schwab OCO manages them)

### 16:00 ET — Market close
- Any position still open at close = OCO sitting at Schwab overnight
- This is acceptable for Tuesday's launch but flag the open positions
- Tuesday evening: **manually close any overnight positions** unless you
  want the gap risk
- Tuesday evening: **re-auth Schwab token** (expires Wednesday afternoon)

---

## Circuit breakers — when to stop

| Condition | Bot action | What to do |
|---|---|---|
| Daily loss > 1% (~\$290) | bot stops trading | Review the trades, decide if you keep going Wed |
| 7 consecutive losses | bot stops trading | Same |
| Any unexpected error in log | bot may keep trading | **You** decide — Ctrl-C if uncomfortable |

**Manual stop**: Ctrl-C the bot process, or kill it. Existing OCO orders at
Schwab remain (they're broker-side). To cancel them: use Schwab UI.

---

## What you're really testing this week

Not "is the bot profitable" — the sample is too small for that.

You're testing:
1. **Do the new gates fire correctly?** (9:35 blackout, 15:30 cutoff, strategy multipliers)
2. **Does the news strategy hold up under live execution slippage?** (sim and live PF can diverge)
3. **Does the operational layer survive a full session?** (token health, error recovery, log integrity)

Friday evening, run `research/news_strategy_validation.py` again. If the
bootstrap CI lower bound clears 1.0 with the additional samples, graduate
to `live_size_multiplier=0.5` for week 2. Otherwise stay at 0.25.

---

## Post-launch hardening (next week — NOT blocking Tuesday)

These were deferred from Task #9 because they need live-trading data to
validate:
- **Real-time P&L dashboard** (existing one is OK but could be cleaner)
- **Live `exit_audit` ledger** — pair every Schwab fill to its OCO leg
  state so we can attribute exits cleanly (resolves Task #8 measurement gap)
- **EOD flatten logic** — close all bot-managed positions at 15:55 ET
  so OCO orders don't sit overnight (resolves Task #6 deferred work)

---

## Emergency contacts

- Schwab API status: https://developer.schwab.com/
- Bot logs: `tail -f trading_bot.log`
- State file: `trading_state.json` (positions, today's P&L, circuit breaker state)
- If everything goes sideways: Ctrl-C the bot, manually close positions in
  Schwab UI, audit logs to understand what happened.
