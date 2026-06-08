# 2026-06-08 Evening Deploy — Close-Path Fixes + Bot-Only P&L Circuit

**Window**: After 16:00 ET market close, before 22:00 ET (give yourself room).
**Time required**: ~10 minutes if everything goes clean.
**What's deploying**: 4 source-code changes, all currently inactive on the running bot. None change live trading behavior until restart.

---

## What's in this deploy

| Task | File(s) | Effect after restart |
|---|---|---|
| **#33** Cancel OCO before close | `core/engine.py` | Manual close API will work first try instead of 6-cancel Schwab dance |
| **#34** Verify fill before state cleanup | `core/engine.py` | Rejected/timed-out close orders no longer corrupt local state — position stays tracked + escalates to ZOMBIE |
| **#31** Bot-only P&L circuit | `core/engine.py`, `risk/manager.py`, `core/config.py` | Daily-loss circuit no longer trips on external (HQGE/PINS/COIN) gap-downs |
| **#35** Rising-peak filter for SHORT | `strategies/builtin.py`, `core/config.py` | No live behavior change — SHORT branch still disabled. Filter ready for re-enable after sim soak. |

All 4 land as one atomic restart. No partial deploys.

---

## Pre-flight (before stopping the bot)

### 1. Market is closed

```bash
date  # confirm > 16:00 ET / 21:00 UTC EDT
```

If market is still open: **stop**. Wait for close.

### 2. OWL still has its OCO bracket at Schwab

In your Schwab app: Order Status → confirm OWL has 2 pending sell orders (SELL_LIMIT @ $9.96 and SELL_STOP @ $9.26). These are what protects OWL through the restart. If they're missing, replace them before restarting the bot.

### 3. Git working-tree review

```bash
git status --short core/ risk/ strategies/ tests/
```

Expected exactly:
```
 M core/config.py
 M core/engine.py
 M risk/manager.py
 M strategies/builtin.py
 M tests/test_recent_fixes.py
```

If anything else shows: investigate. Don't ship surprise changes.

### 4. Tests green

```bash
/home/nvidia/anaconda3/envs/trading-bot/bin/python -m pytest tests/test_recent_fixes.py -q
```

Expected: the 4 new test classes (`TestCancelBracketBeforeClose`, `TestRisingPeakFilter`, `TestVerifyCloseFill`, `TestBotOnlyPnLCircuit`) all pass. The 8 pre-existing failures (DirectionGate defaults, NewsVerifier, MaxPositions, etc.) are unrelated to this deploy — they were failing before the day started. Do not block on them.

### 5. Token has > 8h life

```bash
/home/nvidia/anaconda3/envs/trading-bot/bin/python -c "
import json
from datetime import datetime
with open('token_1.json') as f:
    tok = json.load(f)
ttl = tok.get('creation_timestamp', 0) + 7*24*3600
rem = (ttl - datetime.now().timestamp()) / 3600
print(f'refresh-token TTL remaining: {rem:.1f} hours')
"
```

If < 8h: **stop**. Re-auth before deploying — you don't want to discover a dead token during tomorrow's open.

---

## STOP conditions

Do **not** proceed if any of these is true:

- Market is open
- OWL is gone or its OCO is partial (one leg missing)
- `git status` shows unexpected modified files
- Test suite shows NEW failures (not the 8 pre-existing)
- Refresh-token < 8h
- You can't open the bot's port 9000 right now (it's broken — investigate before adding more changes)

---

## Deploy

### 6. Commit

```bash
cd /home/nvidia/claude/trading_bot_commentary

git add core/config.py core/engine.py risk/manager.py strategies/builtin.py tests/test_recent_fixes.py

git commit -m "$(cat <<'EOF'
fix: 2026-06-08 close-path + circuit + SHORT-filter shipping

Four fixes triggered by the 2026-06-08 manual close incident on
NOK/GLW/OWL (Schwab rejected 3 closes for 'oversold position'
because the bot's close path did not cancel the OCO bracket first):

* #33 (v-cancel-bracket-before-close-2026-06-08): _close_real_position
  cancels the active OCO bracket before placing the sell. Reuses
  existing _cancel_existing_orders helper.

* #34 (v-verify-close-fill-2026-06-08): _close_real_position now awaits
  _verify_order_fill before returning True. 201 ('accepted') is no
  longer treated as filled; rejections/timeouts return False so the
  caller promotes the position to ZOMBIE instead of removing it.

* #31 (v-bot-only-pnl-circuit-2026-06-08): daily-loss circuit now reads
  bot-managed P&L (realized today + unrealized open) instead of
  account-wide schwab_daily_pnl. Stops external holdings
  (HQGE/PINS/COIN) gap-downs from pausing the bot on losses that
  aren't its responsibility. Gated by ENABLE_BOT_ONLY_PNL_CIRCUIT
  (default True). schwab_daily_pnl still tracked for display.

* #35 (v-rising-peak-filter-2026-06-08): mean-rev SHORT branch now
  requires close<SMA50 AND MACD<MACD_signal before firing. Symmetric
  to the falling-knife filter on LONG. Closes the structural gap that
  produced the 2026-05-11 -$665 shorts-pile-on. Gated by
  ENABLE_RISING_PEAK_FILTER (default True). Does NOT enable SHORT
  (ENABLE_MEAN_REV_SHORT stays False); the filter is a precondition
  for safe re-enable after sim soak.

22 new tests across 4 classes in tests/test_recent_fixes.py.
EOF
)"

git log -1 --stat
```

Verify the commit shows exactly 5 files changed.

### 7. Stop the bot

```bash
kill -TERM 604579
# wait ~5 sec
sleep 5
ps -p 604579 -o pid,etime,cmd 2>/dev/null
```

Expected: process gone. If still running, `kill -KILL 604579` and investigate why it didn't respond to TERM (could be stuck in a Schwab call).

### 8. Restart

```bash
cd /home/nvidia/claude/trading_bot_commentary
nohup /home/nvidia/anaconda3/envs/trading-bot/bin/python trading_bot_commentary_updated.py > trading_bot.stdout.log 2>&1 &
echo $!  # capture new PID
sleep 8
```

### 9. Verify startup

```bash
# Port 9000 listening
ss -tlnp 2>/dev/null | grep :9000 || echo "PORT 9000 NOT LISTENING — ABORT"

# Bot's API responds
API_KEY=$(grep '^TRADING_API_KEY=' .env | cut -d= -f2-)
curl -s http://localhost:9000/api/health -H "Authorization: Bearer $API_KEY" | head -1

# State file shows positions
/home/nvidia/anaconda3/envs/trading-bot/bin/python -c "
import json
with open('trading_state.json') as f:
    s = json.load(f)
pd = s.get('positions_data', {})
print(f'positions: {list(pd.keys())}')
print(f'schwab_pnl: {s.get(\"schwab_pnl\")}')
"
```

Expected:
- Port 9000 listening
- Health endpoint returns success
- `positions` includes at least `OWL` (your bot-managed) plus `HQGE`, `PINS`, `COIN`, anything else you hold

If positions list is empty: wait another 30s and re-check. `sync_positions_with_schwab` runs in the background and re-discovers from Schwab on startup.

### 10. Smoke-test the new code paths

These are non-mutating checks — just observation.

```bash
# (A) Config flags are loaded with the new defaults
/home/nvidia/anaconda3/envs/trading-bot/bin/python -c "
from core.config import Config
c = Config()
print(f'ENABLE_BOT_ONLY_PNL_CIRCUIT = {c.ENABLE_BOT_ONLY_PNL_CIRCUIT}')
print(f'ENABLE_RISING_PEAK_FILTER   = {c.ENABLE_RISING_PEAK_FILTER}')
print(f'ENABLE_MEAN_REV_SHORT       = {c.ENABLE_MEAN_REV_SHORT}')
"
# Expected: True, True, False

# (B) Bot-only P&L is being computed (look in bot log)
grep "Bot-only P&L Check" trading_bot.log | tail -1
# Expected: one line within the first ~2 min of restart, after the first 5-cycle Schwab sync
```

### 11. Watch the log for any anomaly in the first 5 min

```bash
tail -f trading_bot.log
```

Look for: `ERROR`, `Exception`, `unexpected`, anything mentioning ZOMBIE, anything mentioning the new v-tags. Most likely you see nothing alarming and the bot quietly logs its startup sequence.

`Ctrl-C` after 5 clean minutes.

---

## Rollback (if anything is wrong)

```bash
cd /home/nvidia/claude/trading_bot_commentary

# Stop new bot
kill -TERM $(pgrep -f trading_bot_commentary_updated.py)
sleep 5

# Revert the commit
git revert HEAD --no-edit

# Restart
nohup /home/nvidia/anaconda3/envs/trading-bot/bin/python trading_bot_commentary_updated.py > trading_bot.stdout.log 2>&1 &
```

OWL's OCO at Schwab is unaffected by either deploy or rollback — it's a broker-side order. The position is protected either way.

---

## What to watch tomorrow morning at the open

These are the new behaviors that haven't been exercised in live yet. Worth a glance at 09:30 ET.

### From #31 (bot-only P&L)
- Log line `Bot-only P&L Check: $X (Schwab P&L: $Y, Limit: $Z)` should appear every ~25s (every 5 analysis cycles × ~5s).
- If `Bot P&L` and `Schwab P&L` show different numbers, that's working as intended — the difference is exactly the externals.

### From #33 + #34 (close path)
- These only matter if you manually close a position. If you don't manually close anything, you'll see no effect.
- If you DO manually close, the workflow is now: one API call (or one dashboard click) → cancel OCO + place sell + verify fill → state cleanup. No 6-cancel Schwab dance.

### From #35 (rising-peak filter)
- No live effect — SHORT branch still gated by `ENABLE_MEAN_REV_SHORT=False`.
- The filter only fires when SHORT is enabled. Tomorrow there's nothing new to see for this one.

---

## Reference

- Incident: 2026-06-08 11:12 ET (this morning, NOK/GLW/OWL close rejections, +$1,007 day P&L recovered to +$533 realized after user manually closed via Schwab app)
- Earlier shorts incident (informs #35): 2026-05-11 (-$665 unrealized from 5 simultaneous mean-rev SHORTS)
- Earlier launch checklist style: `docs/TUESDAY_2026-05-26_GO_LIVE.md`
- Rollback script template: `rollback_launch.sh` (from c4ab0b2)
- New v-tags (for grepping the code post-deploy):
  - `v-cancel-bracket-before-close-2026-06-08`
  - `v-verify-close-fill-2026-06-08`
  - `v-bot-only-pnl-circuit-2026-06-08`
  - `v-rising-peak-filter-2026-06-08`
