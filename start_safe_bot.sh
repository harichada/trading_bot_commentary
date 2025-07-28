#!/bin/bash

echo "Starting Professional Trading Bot with Safe Imports..."
echo "=================================================="

# Kill any existing processes
pkill -f "python run_professional_bot.py" 2>/dev/null || true
sleep 1

# Check if port 8000 is free
if lsof -i :8000 | grep -q LISTEN; then
    echo "Port 8000 is in use. Killing existing process..."
    lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill -9 2>/dev/null || true
    sleep 1
fi

# Start the bot
echo "Starting bot with safe imports enabled..."
python run_professional_bot.py