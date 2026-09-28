#!/bin/bash
# Nightly research refresh for the live fit/desk layers: export → news → earnings → context → calibrate → run →
# run(shuffle) → run(selection) → score --postgres → playbook. Run by cron after the 21:00 ET Alpaca bar ingest.
# Logs to data/strategy_hunt/nightly/<date>.log; never prints secrets. Exit code is the pipeline's.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
set -a; . ./.env; set +a
export POSTGRES_DSN="${POSTGRES_DSN:-$(/home/nvidia/anaconda3/envs/trading-bot/bin/python -c 'from backtest.engine import DEFAULT_DSN; print(DEFAULT_DSN)')}"
mkdir -p data/strategy_hunt/nightly
LOG="data/strategy_hunt/nightly/$(date +%F).log"
echo "== nightly strategy hunt start $(date -Iseconds)" >> "$LOG"
/home/nvidia/anaconda3/envs/trading-bot/bin/python -m research.strategy_hunt all --workers 12 --postgres >> "$LOG" 2>&1
rc=$?
echo "== nightly strategy hunt end rc=$rc $(date -Iseconds)" >> "$LOG"
exit $rc
