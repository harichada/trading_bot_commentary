#!/bin/bash
# Install gap-fade as a systemd service for 24/7 operation (Docker mode)
# Uses the infra Makefile targets: deploy-dev, down-dev, restart-dev, logs-dev
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INFRA_DIR="${INFRA_DIR:-/home/nvidia/claude/infra}"
CURRENT_USER="$(whoami)"
RUDRA_ENV="${RUDRA_ENV:-dev}"
GAP_FADE_PORT="${GAP_FADE_PORT:-8004}"

echo "Installing Rudra Trading Engine service (Docker)..."
echo "  User:      $CURRENT_USER"
echo "  Infra dir: $INFRA_DIR"
echo "  Env:       $RUDRA_ENV"
echo "  Port:      $GAP_FADE_PORT"
echo ""

# Verify docker compose is available
if ! docker-compose version &>/dev/null; then
    echo "ERROR: 'docker-compose' not found. Install Docker Compose."
    exit 1
fi

# Verify infra directory and Makefile exist
if [ ! -f "$INFRA_DIR/Makefile" ]; then
    echo "ERROR: Makefile not found in $INFRA_DIR"
    echo "Set INFRA_DIR to the correct path."
    exit 1
fi

# Generate main service file
cat > /tmp/gap-fade.service <<EOF
[Unit]
Description=Rudra Trading Engine (Docker)
After=docker.service network-online.target
Requires=docker.service
Wants=network-online.target
OnFailure=gap-fade-notify-failure@%n.service

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$INFRA_DIR

# Use the infra Makefile targets
ExecStartPre=/usr/bin/make deploy-$RUDRA_ENV
ExecStart=/usr/bin/make logs-$RUDRA_ENV
ExecStop=/usr/bin/make down-$RUDRA_ENV
ExecReload=/usr/bin/make restart-$RUDRA_ENV

Restart=always
RestartSec=10
StartLimitIntervalSec=300
StartLimitBurst=5

# Watchdog
WatchdogSec=120

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=gap-fade

# Resource limits (container limits in docker-compose.yml take precedence)
MemoryMax=5G

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
EnvironmentFile=$INFRA_DIR/.env_$RUDRA_ENV
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

sleep 3
sudo systemctl status gap-fade --no-pager

echo ""
echo "Done! Rudra Trading Engine is running as a Docker-backed systemd service."
echo ""
echo "Useful commands:"
echo "  sudo systemctl status gap-fade       # Check service status"
echo "  sudo systemctl restart gap-fade      # Restart container"
echo "  sudo systemctl stop gap-fade         # Stop container"
echo "  journalctl -u gap-fade -f            # Follow logs (journald)"
echo "  cd $INFRA_DIR && make logs-$RUDRA_ENV   # Docker logs"
echo "  cd $INFRA_DIR && make status         # All environments"
echo "  curl localhost:$GAP_FADE_PORT/api/health  # Health check"
