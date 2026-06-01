#!/usr/bin/env bash
# v-bot-monitor-2026-06-01: launch the bot monitor in the background.
#
# Usage:
#   tools/start_monitor.sh           # start in background, 60s poll
#   tools/start_monitor.sh stop      # find and kill the running monitor
#   tools/start_monitor.sh status    # is it running?
#
# Logs go to /tmp/bot_monitor.out; the monitor's reports land in
# docs/monitor/latest.md (+ timestamped history in
# docs/monitor/<date>/<hour-min>.md).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/home/nvidia/anaconda3/envs/trading-bot/bin/python}"
LOG="/tmp/bot_monitor.out"
PIDFILE="/tmp/bot_monitor.pid"

cmd="${1:-start}"

case "$cmd" in
  start)
    if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
      echo "monitor already running (pid $(cat "${PIDFILE}"))"
      exit 0
    fi
    cd "${REPO}"
    nohup "${PY}" tools/bot_monitor.py --quiet > "${LOG}" 2>&1 &
    echo $! > "${PIDFILE}"
    disown
    echo "monitor started pid=$(cat "${PIDFILE}") log=${LOG}"
    echo "reports: ${REPO}/docs/monitor/latest.md"
    ;;
  stop)
    if [ -f "${PIDFILE}" ]; then
      pid=$(cat "${PIDFILE}")
      if kill -0 "${pid}" 2>/dev/null; then
        kill -TERM "${pid}"
        echo "stopped pid=${pid}"
      else
        echo "no live process for pid=${pid}"
      fi
      rm -f "${PIDFILE}"
    else
      echo "no pidfile"
    fi
    ;;
  status)
    if [ -f "${PIDFILE}" ] && kill -0 "$(cat "${PIDFILE}")" 2>/dev/null; then
      pid=$(cat "${PIDFILE}")
      echo "running pid=${pid}"
      ps -p "${pid}" -o pid,etime,stat,comm
    else
      echo "not running"
    fi
    ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
