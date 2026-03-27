#!/usr/bin/env python3
"""Daily bar backfill — runs via cron at 5 PM ET to ensure tomorrow's scan has data.

Usage:
    python backfill_daily_bars.py              # Backfill today + yesterday
    python backfill_daily_bars.py --days 5     # Backfill last 5 days

Cron entry (add with: crontab -e):
    0 17 * * 1-5 /home/nvidia/anaconda3/envs/trading-bot/bin/python backfill_daily_bars.py >> /tmp/daily_bar_backfill.log 2>&1
"""

import argparse
import os
import time
from datetime import datetime, timedelta

import psycopg2
import requests
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.environ.get('DATABASE_URL', '') or 'postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev'
API_KEY = os.environ.get('ALPACA_API_KEY', '')
SECRET = os.environ.get('ALPACA_SECRET_KEY', '')


def backfill(days=2):
    if not API_KEY or not SECRET:
        print("ERROR: No Alpaca credentials")
        return

    headers = {'APCA-API-KEY-ID': API_KEY, 'APCA-API-SECRET-KEY': SECRET}

    # Get all tradeable assets
    resp = requests.get('https://api.alpaca.markets/v2/assets', headers=headers,
                        params={'status': 'active', 'asset_class': 'us_equity'}, timeout=60)
    assets = [a['symbol'] for a in resp.json()
              if a.get('tradable') and a.get('exchange') in ('NYSE', 'NASDAQ', 'ARCA', 'AMEX', 'BATS', 'NYSEARCA')]
    print(f"{datetime.now()}: {len(assets)} tradeable assets")

    start = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    stored = 0

    for i in range(0, len(assets), 1500):
        batch = assets[i:i + 1500]
        try:
            resp = requests.get('https://data.alpaca.markets/v2/stocks/bars', headers=headers,
                                params={'symbols': ','.join(batch), 'timeframe': '1Day',
                                        'start': start, 'end': end, 'limit': '10000',
                                        'feed': 'sip', 'adjustment': 'raw'}, timeout=30)
            if resp.status_code == 200:
                for sym, bars in resp.json().get('bars', {}).items():
                    for bar in bars:
                        cur.execute(
                            'INSERT INTO daily_bars (symbol, date, open, high, low, close, volume) '
                            'VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (symbol, date) DO NOTHING',
                            (sym, bar['t'][:10], bar['o'], bar['h'], bar['l'], bar['c'], bar['v']))
                        stored += 1
                conn.commit()
        except Exception as e:
            print(f"Batch {i // 1500} error: {e}")
            conn.rollback()
        time.sleep(0.1)

    # Verify
    cur.execute("SELECT date, COUNT(*) FROM daily_bars WHERE date >= %s GROUP BY date ORDER BY date", (start,))
    print(f"Stored {stored} bars. Coverage:")
    for r in cur.fetchall():
        status = "OK" if r[1] >= 10000 else "LOW"
        print(f"  {r[0]}: {r[1]} symbols [{status}]")
    conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=2)
    args = parser.parse_args()
    backfill(args.days)
