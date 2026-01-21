#!/usr/bin/env python3
"""
EMERGENCY STOP
==============
Immediately close all positions and stop trading.
USE ONLY IN EMERGENCIES!

This script will:
1. Close ALL open positions at market price
2. Cancel ALL pending orders
3. Stop the trading bot
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


def close_all_positions(port=8000):
    """Close all positions via API"""
    if not requests:
        return False

    try:
        response = requests.post(
            f"http://localhost:{port}/api/emergency/close-all",
            timeout=60  # Longer timeout for position closing
        )
        if response.status_code == 200:
            result = response.json()
            closed = result.get('closed_positions', [])
            print(f"✓ Closed {len(closed)} position(s)")
            for pos in closed:
                print(f"   - {pos}")
            return True
        else:
            print(f"⚠ Position close returned: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"⚠ API error: {e}")
        return False


def cancel_all_orders(port=8000):
    """Cancel all pending orders via API"""
    if not requests:
        return False

    try:
        response = requests.post(
            f"http://localhost:{port}/api/orders/cancel-all",
            timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            cancelled = result.get('cancelled_orders', [])
            print(f"✓ Cancelled {len(cancelled)} order(s)")
            return True
    except requests.exceptions.RequestException:
        pass
    return False


def shutdown_bot(port=8000):
    """Shutdown the bot via API"""
    if not requests:
        return False

    try:
        response = requests.post(
            f"http://localhost:{port}/api/shutdown",
            timeout=10
        )
        if response.status_code == 200:
            print("✓ Shutdown command sent")
            return True
    except requests.exceptions.RequestException:
        pass
    return False


def force_kill():
    """Force kill all trading bot processes"""
    killed = False

    try:
        # Find and kill processes
        result = subprocess.run(
            ["pkill", "-9", "-f", "trading_bot"],
            capture_output=True
        )
        if result.returncode == 0:
            print("✓ Force killed trading bot processes")
            killed = True
    except FileNotFoundError:
        # pkill not available
        try:
            result = subprocess.run(
                ["pgrep", "-f", "trading_bot"],
                capture_output=True,
                text=True
            )
            for pid in result.stdout.strip().split("\n"):
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        print(f"✓ Force killed process {pid}")
                        killed = True
                    except (ValueError, ProcessLookupError):
                        pass
        except Exception:
            pass

    return killed


def main():
    parser = argparse.ArgumentParser(
        description='EMERGENCY STOP - Close all positions and stop trading'
    )
    parser.add_argument('--port', type=int, default=8000,
                        help='Port the bot is running on')
    parser.add_argument('--no-confirm', action='store_true',
                        help='Skip confirmation prompt (DANGEROUS)')
    parser.add_argument('--force', action='store_true',
                        help='Force kill processes even if API works')
    args = parser.parse_args()

    print("\n" + "!" * 60)
    print("       🚨 EMERGENCY STOP 🚨")
    print("!" * 60)
    print("\nThis will:")
    print("  1. Close ALL open positions at market price")
    print("  2. Cancel ALL pending orders")
    print("  3. Stop the trading bot")
    print()

    if not args.no_confirm:
        confirm = input("Type 'EMERGENCY' to confirm: ")
        if confirm != 'EMERGENCY':
            print("\nAborted. No changes made.")
            return

    print("\n⚠️  Executing emergency stop...")
    print("-" * 60)

    api_available = True

    # Step 1: Try to close all positions
    print("\n[1/3] Closing all positions...")
    if not close_all_positions(args.port):
        api_available = False
        print("   Could not close via API - check Schwab account manually!")

    # Step 2: Cancel all orders
    print("\n[2/3] Cancelling all orders...")
    if not cancel_all_orders(args.port):
        print("   Could not cancel via API - check Schwab account manually!")

    # Step 3: Shutdown bot
    print("\n[3/3] Shutting down bot...")
    if api_available and not args.force:
        if shutdown_bot(args.port):
            print("   Graceful shutdown initiated")
        else:
            print("   API shutdown failed, force killing...")
            force_kill()
    else:
        print("   Force killing processes...")
        if not force_kill():
            print("   No processes found to kill")

    print("\n" + "=" * 60)
    print("✅ Emergency stop complete")
    print()
    print("⚠️  IMPORTANT: Verify positions in your Schwab account!")
    print("   https://client.schwab.com/")
    print()
    print("   The API may not have been able to close all positions.")
    print("   Always verify manually after an emergency stop.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
