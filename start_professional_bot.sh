#!/bin/bash

echo "Starting Professional Trading Bot with Enhanced Dashboard..."
echo "=================================================="

# Check if enhanced dashboard exists
if [ -f "professional_dashboard_enhanced.html" ]; then
    echo "✓ Enhanced dashboard found"
else
    echo "✗ Enhanced dashboard not found!"
    exit 1
fi

# Clear browser cache reminder
echo ""
echo "IMPORTANT: If you see the old UI, please:"
echo "1. Clear your browser cache (Ctrl+Shift+R or Cmd+Shift+R)"
echo "2. Or open in an incognito/private window"
echo "3. Or append ?v=2 to the URL: http://localhost:8000/?v=2"
echo ""

# Start the bot
echo "Starting bot..."
python run_professional_bot.py