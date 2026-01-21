#!/usr/bin/env python3
"""
Stop Trading Bot
================
Safely stop the trading bot with graceful shutdown.
"""

import os
import sys
import signal
import subprocess
import argparse

try:
    import requests
except ImportError:
    requests = None


def stop_via_api(port=8000):
    """Try graceful shutdown via API"""
    if not requests:
        return False

    try:
        response = requests.post(
            f"http://localhost:{port}/api/shutdown",
            timeout=5
        )
        if response.status_code == 200:
            print("✓ Graceful shutdown initiated via API")
            return True
    except requests.exceptions.RequestException:
        pass
    return False


def stop_via_signal():
    """Find and send SIGTERM to trading bot processes"""
    stopped = False

    # Find trading bot processes
    try:
        result = subprocess.run(
            ["pgrep", "-f", "trading_bot_commentary_updated|start_trading"],
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                try:
                    # Don't kill ourselves
                    if int(pid) == os.getpid():
                        continue
                    os.kill(int(pid), signal.SIGTERM)
                    print(f"✓ Sent SIGTERM to process {pid}")
                    stopped = True
                except (ValueError, ProcessLookupError, PermissionError) as e:
                    print(f"⚠ Could not stop process {pid}: {e}")
    except FileNotFoundError:
        # pgrep not available, try alternative
        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True
            )
            for line in result.stdout.split("\n"):
                if "trading_bot" in line and "stop_trading" not in line:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            if pid != os.getpid():
                                os.kill(pid, signal.SIGTERM)
                                print(f"✓ Sent SIGTERM to process {pid}")
                                stopped = True
                        except (ValueError, ProcessLookupError):
                            pass
        except Exception:
            pass

    return stopped


def main():
    parser = argparse.ArgumentParser(description='Stop the Trading Bot')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port the bot is running on')
    parser.add_argument('--force', action='store_true',
                        help='Force kill (SIGKILL) if SIGTERM fails')
    args = parser.parse_args()

    print("🛑 Stopping Trading Bot...")
    print()

    # Try graceful shutdown via API first
    if stop_via_api(args.port):
        print("\n✅ Shutdown command sent successfully")
        print("   Bot will save state and exit gracefully")
        return

    # Fallback to signal-based shutdown
    print("   API not responding, trying signal-based shutdown...")
    if stop_via_signal():
        print("\n✅ Stop signal sent")
        print("   Bot should shut down within 30 seconds")
    else:
        print("\n❌ No trading bot process found")
        print("   The bot may not be running")


if __name__ == "__main__":
    main()
