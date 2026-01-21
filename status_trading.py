#!/usr/bin/env python3
"""
Trading Bot Status
==================
Check the current status of the trading bot.
"""

import sys
import json
import argparse
from datetime import datetime

try:
    import requests
except ImportError:
    print("❌ 'requests' package required. Run: pip install requests")
    sys.exit(1)


def get_json(url, timeout=5):
    """Fetch JSON from URL"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None


def format_currency(amount):
    """Format number as currency"""
    if amount is None:
        return "N/A"
    return f"${amount:,.2f}"


def format_percent(value):
    """Format number as percentage"""
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def main():
    parser = argparse.ArgumentParser(description='Check Trading Bot Status')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port the bot is running on')
    parser.add_argument('--json', action='store_true',
                        help='Output raw JSON')
    parser.add_argument('--watch', action='store_true',
                        help='Continuously watch status')
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"

    # If JSON mode, just output raw data
    if args.json:
        status = get_json(f"{base_url}/api/status")
        print(json.dumps(status, indent=2) if status else '{"error": "not running"}')
        return

    # Pretty printed status
    print("\n" + "═" * 60)
    print("       TRADING BOT STATUS")
    print("═" * 60)

    # Health check
    health = get_json(f"{base_url}/health")

    if not health:
        print("\n❌ Bot is NOT running or not responding")
        print(f"   Dashboard URL: {base_url}")
        print("   Start with: python start_trading.py")
        print("\n" + "═" * 60 + "\n")
        return

    # Basic status
    status = get_json(f"{base_url}/api/status") or {}

    state = status.get('state', health.get('status', 'Unknown'))
    mode = status.get('mode', 'Unknown')
    uptime = status.get('uptime', 'Unknown')

    print(f"\n✅ Bot is RUNNING")
    print(f"   State: {state}")
    print(f"   Mode: {mode}")
    print(f"   Uptime: {uptime}")

    # Positions
    positions = get_json(f"{base_url}/api/positions")
    if positions and isinstance(positions, list):
        print(f"\n📊 POSITIONS ({len(positions)} open)")
        print("   " + "-" * 50)

        if len(positions) == 0:
            print("   No open positions")
        else:
            total_value = 0
            total_pnl = 0

            for pos in positions:
                symbol = pos.get('symbol', '?')
                qty = pos.get('quantity', 0)
                value = pos.get('market_value', 0)
                pnl = pos.get('unrealized_pnl', 0)
                pnl_pct = pos.get('unrealized_pnl_percent', 0)

                pnl_icon = '🟢' if pnl >= 0 else '🔴'
                print(f"   {symbol:6} | {qty:>6} shares | "
                      f"Value: {format_currency(value):>10} | "
                      f"P&L: {pnl_icon} {format_currency(pnl):>10} ({format_percent(pnl_pct)})")

                total_value += value or 0
                total_pnl += pnl or 0

            print("   " + "-" * 50)
            pnl_icon = '🟢' if total_pnl >= 0 else '🔴'
            print(f"   TOTAL   | Value: {format_currency(total_value):>10} | "
                  f"P&L: {pnl_icon} {format_currency(total_pnl):>10}")

    # P&L Summary
    pnl_data = get_json(f"{base_url}/api/pnl")
    if pnl_data:
        daily = pnl_data.get('daily_pnl', 0)
        total = pnl_data.get('total_pnl', 0)
        win_rate = pnl_data.get('win_rate')
        trades = pnl_data.get('trade_count', 0)

        print(f"\n💰 P&L SUMMARY")
        print("   " + "-" * 50)
        print(f"   Today:      {'🟢' if daily >= 0 else '🔴'} {format_currency(daily)}")
        print(f"   Total:      {'🟢' if total >= 0 else '🔴'} {format_currency(total)}")
        if win_rate is not None:
            print(f"   Win Rate:   {format_percent(win_rate)}")
        print(f"   Trades:     {trades}")

    # Risk Status
    risk = get_json(f"{base_url}/api/risk")
    if risk:
        print(f"\n⚠️  RISK STATUS")
        print("   " + "-" * 50)
        var = risk.get('var_95')
        if var:
            print(f"   VaR (95%):  {format_currency(var)}")
        drawdown = risk.get('drawdown')
        if drawdown:
            print(f"   Drawdown:   {format_percent(drawdown)}")
        exposure = risk.get('total_exposure')
        if exposure:
            print(f"   Exposure:   {format_currency(exposure)}")

    # Circuit Breakers
    circuit_breakers = status.get('circuit_breakers', {})
    if circuit_breakers:
        open_breakers = [k for k, v in circuit_breakers.items()
                        if v.get('state') == 'OPEN']
        if open_breakers:
            print(f"\n🚨 CIRCUIT BREAKERS OPEN: {', '.join(open_breakers)}")

    # Footer
    print("\n" + "═" * 60)
    print(f"   Dashboard: {base_url}")
    print(f"   Checked:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
