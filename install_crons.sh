#!/bin/bash
# Install Rudra Trading Engine cron jobs

PYTHON="/home/nvidia/anaconda3/envs/trading-bot/bin/python"
APP_DIR="/home/nvidia/claude/singleton/trading_bot_commentary"

# Remove existing Rudra crons
crontab -l 2>/dev/null | grep -v "rudra\|backfill_daily\|dead_man_switch" > /tmp/crontab_clean

# Add new crons
cat >> /tmp/crontab_clean << CRON
# ── Rudra Trading Engine Crons ──
# Daily bar backfill: 5 PM ET Mon-Fri (ensures tomorrow's scan has data)
0 17 * * 1-5 cd $APP_DIR && $PYTHON backfill_daily_bars.py >> /tmp/daily_bar_backfill.log 2>&1

# Dead man's switch: 3:55 PM ET Mon-Fri (emergency position liquidation)
55 15 * * 1-5 cd $APP_DIR && $PYTHON eod_dead_man_switch.py >> /tmp/eod_dead_man_switch.log 2>&1

# Weekly swing scan alert: Sunday 6 PM ET (scan + Telegram notification)
0 18 * * 0 cd $APP_DIR && $PYTHON rudra_swing_engine.py --scan >> /tmp/swing_scan.log 2>&1
CRON

crontab /tmp/crontab_clean
echo "Crons installed:"
crontab -l | grep -v "^#$" | grep -v "^$"
