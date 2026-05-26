#!/bin/bash
# v-eod-runner-2026-05-13: End-of-day autonomous reporter pipeline.
#
# Run this after market close (16:00 ET) — manually or via cron:
#   0 17 * * 1-5  /home/nvidia/claude/trading_bot_commentary/research/eod_runner.sh
#
# It:
#   1. Generates today's session report (research/session_reporter.py)
#   2. Generates a 7-day drift / auto-flip recommendation
#   3. Prints both summaries to stdout for human-readable scanning
#
# No interactivity. Safe to run multiple times — outputs are idempotent
# (overwrite the same dated files).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOT_PY="/home/nvidia/anaconda3/envs/trading-bot/bin/python3.10"
DATE=$(date '+%Y-%m-%d')

cd "$REPO_ROOT"

echo "============================================================"
echo "[eod] $DATE — generating session report"
echo "============================================================"
"$BOT_PY" research/session_reporter.py --date "$DATE"
echo
SESSION_REPORT="docs/training_runs/session_${DATE}.md"
if [ -f "$SESSION_REPORT" ]; then
    echo "----- session report -----"
    cat "$SESSION_REPORT"
fi

echo
echo "============================================================"
echo "[eod] $DATE — generating 7-day drift / auto-flip recommendation"
echo "============================================================"
"$BOT_PY" research/drift_monitor.py --days 7
echo
DRIFT_REPORT="docs/training_runs/drift_${DATE}.md"
if [ -f "$DRIFT_REPORT" ]; then
    echo "----- drift report -----"
    cat "$DRIFT_REPORT"
fi
