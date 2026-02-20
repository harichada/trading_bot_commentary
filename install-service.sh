#!/bin/bash
# Install gap-fade as a systemd service for 24/7 operation
set -e

SERVICE_FILE="gap-fade.service"
INSTALL_PATH="/etc/systemd/system/$SERVICE_FILE"

echo "Installing Gap Fade service..."

# Stop if already running
sudo systemctl stop gap-fade 2>/dev/null || true

# Copy service file
sudo cp "$SERVICE_FILE" "$INSTALL_PATH"

# Reload systemd
sudo systemctl daemon-reload

# Enable on boot
sudo systemctl enable gap-fade

# Start it
sudo systemctl start gap-fade

echo ""
echo "Done! Gap Fade is running as a systemd service."
echo ""
echo "Useful commands:"
echo "  sudo systemctl status gap-fade    # Check status"
echo "  sudo systemctl restart gap-fade   # Restart"
echo "  sudo systemctl stop gap-fade      # Stop"
echo "  journalctl -u gap-fade -f         # Follow logs"
echo "  journalctl -u gap-fade --since '1 hour ago'  # Recent logs"
echo "  curl localhost:8002/api/health     # Health check"
echo ""
echo "The service will:"
echo "  - Auto-restart on crash (after 10s)"
echo "  - Restart if unresponsive for 120s (watchdog)"
echo "  - Start on boot"
echo "  - Cap memory at 2GB, CPU at 80%"
