#!/bin/bash
# v-wedge-fix-2026-09-28 (Task 4, ruling 4): external restarter for the bot.
#
# WHY THIS IS A SHELL SCRIPT AND NOT A PYTHON THREAD
# On 2026-09-28 the bot process sat wedged from 09:37 ET to the close: CPU 0 %,
# 247 of 251 threads in futex_wait, the main thread in poll, one Yahoo socket
# with ~1.5 KB stuck in its send queue. A hang that holds the GIL freezes every
# in-process watchdog with it (see core/loop_heartbeat.py). Only something
# outside the process can recover it.
#
# WHAT IT DOES
#   1. GET /api/status with `curl -m 30`.
#   2. Two consecutive failures (state kept across cron runs) and fewer than
#      MAX_RESTARTS_PER_DAY restarts today  ->  restart:
#        TERM the bot, wait 20 s, KILL if still alive;
#        relaunch exactly as docs/BOT_OPERATIONS_MANUAL.md documents:
#          nohup python trading_bot_commentary_updated.py
#          POST /api/start
#          POST /api/toggle-mode {"mode":"live"}
#   3. Everything is appended to bot_watchdog.log. The API key is read from
#      .env and never printed, logged, or passed on a command line that shows
#      up in `ps` (curl reads the header from a file).
#
# CRON (installed by the controller, not by this script) — every 5 min,
# 06:00-20:00 ET, Mon-Fri:
#   */5 6-19 * * 1-5  cd /home/nvidia/claude/grok/trading_bot_commentary && \
#                     TZ=America/New_York bash scripts/bot_watchdog.sh >> bot_watchdog.log 2>&1
#
# USAGE
#   bash scripts/bot_watchdog.sh            # check, restart if warranted
#   bash scripts/bot_watchdog.sh --dry-run  # report only: never kills, never
#                                           # launches, never writes state
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "bot_watchdog: unknown argument: $arg" >&2; exit 2 ;;
    esac
done

BOT_PORT="${TRADING_BOT_PORT:-9000}"
BOT_HOST="${WATCHDOG_HOST:-127.0.0.1}"
BOT_SCRIPT="trading_bot_commentary_updated.py"
PYTHON_BIN="${PYTHON_BIN:-/home/nvidia/anaconda3/envs/trading-bot/bin/python}"
LOG="${WATCHDOG_LOG:-bot_watchdog.log}"
STATE="${WATCHDOG_STATE:-bot_watchdog.state}"
MAX_RESTARTS_PER_DAY="${WATCHDOG_MAX_RESTARTS:-2}"
FAILS_BEFORE_RESTART="${WATCHDOG_FAILS_BEFORE_RESTART:-2}"
CURL_TIMEOUT="${WATCHDOG_CURL_TIMEOUT:-30}"
TERM_GRACE_SEC="${WATCHDOG_TERM_GRACE:-20}"
BOOT_WAIT_SEC="${WATCHDOG_BOOT_WAIT:-120}"
ENGINE_WARMUP_SEC="${WATCHDOG_ENGINE_WARMUP:-25}"

# The API key comes from the environment when one is already exported (that is
# how the tests run), otherwise from .env. Sourcing .env is the only reason
# this script reads a secret at all; nothing below ever echoes it.
if [[ -z "${TRADING_API_KEY:-}" && -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
fi

# curl reads the Authorization header from a file so the key never appears in
# the process table. Removed on every exit path.
HEADER_FILE=""
cleanup() {
    [[ -n "$HEADER_FILE" && -f "$HEADER_FILE" ]] && rm -f "$HEADER_FILE"
    return 0
}
trap cleanup EXIT

log() {
    local line
    line="$(date -Iseconds) bot_watchdog $*"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "$line"
    else
        echo "$line" | tee -a "$LOG"
    fi
}

status_code() {
    # Echoes the HTTP status of GET /api/status, or 000 when unreachable.
    local args=(-s -o /dev/null -w '%{http_code}' -m "$CURL_TIMEOUT")
    if [[ -n "${TRADING_API_KEY:-}" ]]; then
        HEADER_FILE="$(mktemp)"
        chmod 600 "$HEADER_FILE"
        printf 'Authorization: Bearer %s\n' "$TRADING_API_KEY" > "$HEADER_FILE"
        args+=(-H "@$HEADER_FILE")
    fi
    local out
    # curl's own -w output is "000" when it never got a response; a
    # non-zero exit (timeout, refused) leaves it empty. Normalise both.
    out="$(curl "${args[@]}" "http://${BOT_HOST}:${BOT_PORT}/api/status" 2>/dev/null)"
    [[ -n "$HEADER_FILE" ]] && rm -f "$HEADER_FILE" && HEADER_FILE=""
    echo "${out:-000}"
    return 0
}

post_api() {
    # post_api <path> [json-body] -> echoes the HTTP status.
    local path="$1" body="${2:-}"
    local args=(-s -o /dev/null -w '%{http_code}' -m "$CURL_TIMEOUT" -X POST)
    if [[ -n "${TRADING_API_KEY:-}" ]]; then
        HEADER_FILE="$(mktemp)"
        chmod 600 "$HEADER_FILE"
        printf 'Authorization: Bearer %s\n' "$TRADING_API_KEY" > "$HEADER_FILE"
        args+=(-H "@$HEADER_FILE")
    fi
    if [[ -n "$body" ]]; then
        args+=(-H 'Content-Type: application/json' -d "$body")
    fi
    local out
    out="$(curl "${args[@]}" "http://${BOT_HOST}:${BOT_PORT}${path}" 2>/dev/null)"
    [[ -n "$HEADER_FILE" ]] && rm -f "$HEADER_FILE" && HEADER_FILE=""
    echo "${out:-000}"
    return 0
}

# ── state: consecutive failures + restarts booked today ────────────────
TODAY="$(date +%F)"
FAIL_COUNT=0
RESTART_DATE="$TODAY"
RESTART_COUNT=0
if [[ -f "$STATE" ]]; then
    # shellcheck disable=SC1090
    . "$STATE" 2>/dev/null || true
    FAIL_COUNT="${FAIL_COUNT:-0}"
    RESTART_DATE="${RESTART_DATE:-$TODAY}"
    RESTART_COUNT="${RESTART_COUNT:-0}"
fi
if [[ "$RESTART_DATE" != "$TODAY" ]]; then
    RESTART_DATE="$TODAY"
    RESTART_COUNT=0
fi

save_state() {
    [[ "$DRY_RUN" -eq 1 ]] && return 0
    {
        echo "FAIL_COUNT=$FAIL_COUNT"
        echo "RESTART_DATE=$RESTART_DATE"
        echo "RESTART_COUNT=$RESTART_COUNT"
    } > "$STATE"
}

bot_pids() {
    pgrep -f "$BOT_SCRIPT" 2>/dev/null | grep -v "^$$\$" || true
}

restart_bot() {
    local pids
    pids="$(bot_pids)"
    if [[ -n "$pids" ]]; then
        log "restart: sending TERM to pids=$(echo "$pids" | tr '\n' ' ')"
        if [[ "$DRY_RUN" -eq 0 ]]; then
            # shellcheck disable=SC2086
            kill -TERM $pids 2>/dev/null || true
            local waited=0
            while [[ "$waited" -lt "$TERM_GRACE_SEC" ]] && [[ -n "$(bot_pids)" ]]; do
                sleep 1
                waited=$((waited + 1))
            done
            pids="$(bot_pids)"
            if [[ -n "$pids" ]]; then
                log "restart: still alive after ${TERM_GRACE_SEC}s — sending KILL"
                # shellcheck disable=SC2086
                kill -KILL $pids 2>/dev/null || true
                sleep 2
            fi
        fi
    else
        log "restart: no running $BOT_SCRIPT found (nothing to kill)"
    fi

    log "restart: relaunching nohup $BOT_SCRIPT (port $BOT_PORT)"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        log "dry-run: would launch, then POST /api/start, then POST /api/toggle-mode {\"mode\":\"live\"}"
        return 0
    fi
    nohup "$PYTHON_BIN" "$BOT_SCRIPT" >> nohup.out 2>&1 &
    disown || true

    local waited=0 code="000"
    while [[ "$waited" -lt "$BOOT_WAIT_SEC" ]]; do
        sleep 5
        waited=$((waited + 5))
        code="$(status_code)"
        [[ "$code" != "000" ]] && break
    done
    if [[ "$code" == "000" ]]; then
        log "restart: FAILED — /api/status still unreachable after ${BOOT_WAIT_SEC}s"
        return 1
    fi
    log "restart: server answering (http=$code after ${waited}s)"

    code="$(post_api /api/start)"
    log "restart: POST /api/start http=$code"
    sleep "$ENGINE_WARMUP_SEC"
    code="$(post_api /api/toggle-mode '{"mode":"live"}')"
    log "restart: POST /api/toggle-mode live http=$code"
    return 0
}

# ── the check ──────────────────────────────────────────────────────────
CODE="$(status_code)"
if [[ "$CODE" == "200" ]]; then
    if [[ "$FAIL_COUNT" -gt 0 ]]; then
        log "health ok (http=200) — clearing fail_count=$FAIL_COUNT"
    fi
    FAIL_COUNT=0
    save_state
    [[ "$DRY_RUN" -eq 1 ]] && log "dry-run: healthy, no action"
    exit 0
fi

FAIL_COUNT=$((FAIL_COUNT + 1))
log "health FAILED http=$CODE fail_count=$FAIL_COUNT/$FAILS_BEFORE_RESTART restarts_today=$RESTART_COUNT/$MAX_RESTARTS_PER_DAY"

if [[ "$FAIL_COUNT" -lt "$FAILS_BEFORE_RESTART" ]]; then
    log "waiting for a second consecutive failure before restarting"
    save_state
    exit 1
fi

if [[ "$RESTART_COUNT" -ge "$MAX_RESTARTS_PER_DAY" ]]; then
    log "restart budget exhausted for $TODAY — OPERATOR ACTION REQUIRED"
    save_state
    exit 1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    log "dry-run: would restart now (no process signalled, no state written)"
    restart_bot
    exit 1
fi

RESTART_COUNT=$((RESTART_COUNT + 1))
FAIL_COUNT=0
save_state
if restart_bot; then
    log "restart complete (restarts_today=$RESTART_COUNT)"
    exit 0
fi
log "restart attempt did not bring the bot back"
exit 1
