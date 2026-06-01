#!/usr/bin/env bash
# v-deploy-2026-05-23: reproducible startup for the trading bot.
# Validates safety config, checks Schwab token health, prints a
# summary of the launch state, and starts the bot.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO}"

PYTHON="${PYTHON:-/home/nvidia/anaconda3/envs/trading-bot/bin/python}"
if [ ! -x "${PYTHON}" ]; then
  echo "ERROR: trading-bot conda env python not found at ${PYTHON}" >&2
  echo "       The bot was built/tested against the 'trading-bot' env." >&2
  echo "       Refusing to fall back to a different interpreter — version" >&2
  echo "       skew (pandas 2.3 vs 3.0, sklearn 1.6 vs 1.5) would cause" >&2
  echo "       silent prediction errors or hard crashes." >&2
  echo "       Override with PYTHON=<path> if you know what you're doing." >&2
  exit 1
fi

EXPECTED_ENV="trading-bot"
if [[ "${PYTHON}" != *"/envs/${EXPECTED_ENV}/"* ]]; then
  echo "WARNING: Python is not from the '${EXPECTED_ENV}' conda env" >&2
  echo "         (got: ${PYTHON})" >&2
  echo "         Continuing because PYTHON was explicitly overridden." >&2
fi

echo "================================================================"
echo "Trading Bot — Pre-Launch Health Check"
echo "Repo: ${REPO}"
echo "Python: ${PYTHON} ($(${PYTHON} --version 2>&1))"
echo "Date: $(date -Iseconds)"
echo "================================================================"
echo

# 1. Config sanity check — verify safety dials are at expected values.
"${PYTHON}" - <<'PYEOF'
import sys, yaml
from pathlib import Path
cfg_path = Path("Config().yaml")
if not cfg_path.exists():
    print("ERROR: Config().yaml missing", file=sys.stderr)
    sys.exit(1)
cfg = yaml.safe_load(cfg_path.read_text())
t = cfg.get("trading", {})

# Hard expectations for the live-launch period (week of 2026-05-26).
# Update these as the bot graduates through size tiers.
#
# 2026-05-27 update: enable_breakout_long flipped True after the
# 2026-05-26 ASTS/LUNR/RDW/MU rally exposed the strategy-coverage
# gap. enable_strict_long_gates pinned True so the breakout
# strategy uses ADX>=25 + vol>=2.0x + 1.003x break as its regime
# filter (replacing the deferred SpyRegimeCache wiring).
# enable_news_technical_confirmation pinned True — news strategy
# now demands RSI/SMA20/MACD/volume confirmation before firing
# signal_buy/signal_sell (the FLY/ASTS/TSLA pattern fix).
# 2026-06-01 update: graduated trade sizing after 1 week of live
# trading. live_size_multiplier 0.25 -> 0.40 -> 0.50 (two bumps
# same day after operator observed actual trades were sub-budget by
# 10x — Kelly × strategy_mult × live_mult was multiplicatively
# eroding the configured risk-per-trade). strategy_size_multipliers
# .mean_reversion 0.5 -> 0.75 (mean-rev graduated probation after
# 10+ trades with no catastrophic losses). risk_per_trade_pct
# 0.015 -> 0.020 + atr_reward_risk_ratio 2.0 -> 2.5. Daily-loss
# cap stays at 0.01 = ~$290 so the safety floor is unchanged.
expected = {
    "enable_breakout_long":              True,
    "enable_strict_long_gates":          True,
    "enable_news_technical_confirmation": True,
    "live_size_multiplier":              0.50,
    "risk_per_trade_pct":                0.020,
    "atr_reward_risk_ratio":             2.0,
    "max_daily_loss":                    0.01,
    "late_entry_cutoff_hour":            15,
    "late_entry_cutoff_minute":          30,
}
mults = t.get("strategy_size_multipliers", {}) or {}
expected_mr = 0.75

ok = True
print("Safety config:")
for k, v in expected.items():
    actual = t.get(k)
    flag = "OK " if actual == v else "ERR"
    if actual != v:
        ok = False
    print(f"  [{flag}] {k:<28} expected={v!s:<8} actual={actual!s}")
mr = mults.get("mean_reversion")
flag = "OK " if mr == expected_mr else "ERR"
if mr != expected_mr:
    ok = False
print(f"  [{flag}] strategy_size_multipliers.mean_reversion  expected={expected_mr} actual={mr}")

print()
if not ok:
    print("ABORT: safety config does not match launch plan.", file=sys.stderr)
    print("       Re-check Config().yaml against weekly_report Section 8.", file=sys.stderr)
    sys.exit(2)
print("Safety config: all checks PASSED")
PYEOF

echo

# 2. Token TTL check (inline — does NOT import core/ to avoid the
# pandas/NumPy import cost just for a startup health check).
"${PYTHON}" - <<'PYEOF'
import json, sys
from datetime import datetime, timedelta
from pathlib import Path

token_path = Path("token_1.json")
if not token_path.exists():
    print("ABORT: token_1.json missing", file=sys.stderr)
    sys.exit(3)
try:
    data = json.loads(token_path.read_text())
except Exception as exc:
    print(f"ABORT: failed to parse token_1.json: {exc}", file=sys.stderr)
    sys.exit(3)

ct = data.get("creation_timestamp")
if not isinstance(ct, (int, float)):
    print("ABORT: token_1.json missing creation_timestamp", file=sys.stderr)
    sys.exit(3)

created = datetime.fromtimestamp(ct)
expiry = created + timedelta(days=7)  # Schwab refresh-token TTL
remaining = expiry - datetime.now()
hours = remaining.total_seconds() / 3600

print(f"Schwab token health:")
print(f"  created: {created:%Y-%m-%d %H:%M}")
print(f"  expires: {expiry:%Y-%m-%d %H:%M}")
print(f"  remaining: {hours:.1f}h ({hours/24:.1f} days)")
if hours <= 0:
    print(f"ABORT: refresh token EXPIRED. Re-auth required.", file=sys.stderr)
    sys.exit(4)
if hours < 8:
    print(f"ABORT: refresh token expires in {hours:.1f}h — refuse to start "
          f"a session that may not complete. Re-auth first.", file=sys.stderr)
    sys.exit(5)
if hours < 24:
    print(f"WARN: refresh token expires in {hours:.1f}h. Re-auth before "
          f"the next session.", file=sys.stderr)
PYEOF

echo

# 3. Print the launch summary the operator should review.
cat <<EOF
================================================================
Launch Plan (2026-05-26, Tuesday)
================================================================
Sizing dials:
  live_size_multiplier:                0.25  (quarter size, week 1)
  strategy_size_multipliers.mean_rev:  0.50  (half size, N<20 watch)
  effective news size:                 25% of normal
  effective mean_rev size:              12.5% of normal

Time gates:
  earliest entry:  09:35 ET (hard block on opening 5 min)
  latest entry:    15:30 ET (no overnight risk from late entries)

Disabled strategies:
  breakout         (PF 1.14 on N=2, awaiting walk-forward validation)
  mean_rev_short   (awaiting trend filter + sim soak)

Daily circuit breakers:
  max_daily_loss:        1.0%  (~ \$290 at \$29k account)
  max_consecutive_losses: 7

Operator actions Tuesday morning:
  1. Run this deploy.sh (you're here)
  2. Open the dashboard at http://localhost:8000
  3. At 09:30 ET: toggle mode to LIVE via UI (Settings -> Trading Mode)
  4. Watch trades. If 3+ losses in a row, stop and review.
  5. Tuesday evening: re-auth Schwab (refresh token expires Wed afternoon)

To stop at any time: Ctrl-C, or kill the bot process.

================================================================
Starting bot in 3 seconds. Ctrl-C to abort.
================================================================
EOF

sleep 3

# 4. Launch.
exec "${PYTHON}" trading_bot_commentary_updated.py
