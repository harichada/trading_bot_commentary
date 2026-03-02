#!/bin/bash
# Install gap-fade as a systemd service for 24/7 operation
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CURRENT_USER="$(whoami)"
PYTHON_PATH="$(which python3 || which python)"

echo "Installing Rudra Trading Engine service..."
echo "  User:    $CURRENT_USER"
echo "  Dir:     $SCRIPT_DIR"
echo "  Python:  $PYTHON_PATH"
echo ""

# Generate main service file with correct paths
cat > /tmp/gap-fade.service <<EOF
[Unit]
Description=Rudra Trading Engine
After=network.target
OnFailure=gap-fade-notify-failure@%n.service

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$PYTHON_PATH $SCRIPT_DIR/gap_fade_app.py
Restart=always
RestartSec=10
StartLimitIntervalSec=300
StartLimitBurst=5

# Environment
EnvironmentFile=$SCRIPT_DIR/.env

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=gap-fade

# Watchdog
WatchdogSec=120

# Resource limits
MemoryMax=2G
CPUQuota=80%

[Install]
WantedBy=multi-user.target
EOF

# Generate failure notification service
cat > /tmp/gap-fade-notify-failure@.service <<EOF
[Unit]
Description=Send failure notification for %i

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'curl -s -X POST "https://api.telegram.org/bot\${TELEGRAM_BOT_TOKEN}/sendMessage" -d chat_id="\${TELEGRAM_CHAT_ID}" -d "text=SYSTEMD: %i crashed and is restarting"'
EnvironmentFile=$SCRIPT_DIR/.env
EOF

# Stop if already running
sudo systemctl stop gap-fade 2>/dev/null || true

# Copy service files
sudo cp /tmp/gap-fade.service /etc/systemd/system/gap-fade.service
sudo cp /tmp/gap-fade-notify-failure@.service /etc/systemd/system/gap-fade-notify-failure@.service

# Reload systemd
sudo systemctl daemon-reload

# Enable on boot
sudo systemctl enable gap-fade

# Start it
sudo systemctl start gap-fade

sleep 2
sudo systemctl status gap-fade --no-pager

echo ""
echo "Done! Rudra Trading Engine is running as a systemd service."
echo ""
echo "Useful commands:"
echo "  sudo systemctl status gap-fade    # Check status"
echo "  sudo systemctl restart gap-fade   # Restart"
echo "  sudo systemctl stop gap-fade      # Stop"
echo "  journalctl -u gap-fade -f         # Follow logs"
echo "  curl localhost:8002/api/health     # Health check"
