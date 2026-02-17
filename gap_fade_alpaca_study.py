#!/usr/bin/env python3
"""
Gap Fade Deep Study — Using Alpaca Market Data API (real broker data).
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import time, os, sys

# ── Load credentials ─────────────────────────────────────────────────
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())

load_env()
API_KEY = os.environ.get('ALPACA_API_KEY', '')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', '')
BASE_URL = 'https://data.alpaca.markets/v2/stocks'
HEADERS = {
    'APCA-API-KEY-ID': API_KEY,
    'APCA-API-SECRET-KEY': SECRET_KEY,
}

# ── Universe ─────────────────────────────────────────────────────────
LARGE_CAP = [
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','JPM','JNJ',
    'V','PG','UNH','HD','MA','DIS','PYPL','BAC','INTC','VZ',
    'NFLX','ADBE','CRM','CMCSA','PFE','TMO','ABT','CSCO','PEP','AVGO',
    'ACN','COST','NKE','MRK','WMT','LLY','MCD','DHR','TXN','QCOM',
    'MDT','HON','UPS','LOW','MS','GS','BLK','ISRG','SCHW','AXP',
    'BA','CAT','DE','GE','LMT','RTX','MMM','IBM','ORCL','AMD',
    'NOW','SNOW','UBER','ABNB','COIN','SHOP','MELI','SE','SPOT',
    'ZM','DOCU','CRWD','ZS','DDOG','NET','MDB','TEAM','OKTA',
]

MID_CAP = [
    'ROKU','SNAP','PINS','ETSY','DASH','LYFT','HOOD','DKNG','AFRM','SOFI',
    'UPST','RBLX','U','PATH','BILL','HUBS','VEEV','GNRC',
    'ENPH','FSLR','SEDG','RUN','PLUG','CHPT','QS',
    'RIVN','LCID','XPEV','LI','NIO',
    'W','CHWY','PTON','BYND',
    'PLTR','IONQ','RKLB','JOBY','SPCE','ASTS',
    'AI','SOUN','CRSP','EDIT','NTLA','BEAM','IONS','REGN','VRTX','BIIB',
    'MRNA','BNTX',
]

SMALL_CAP = [
    'GME','AMC','KOSS','BB','NOK',
    'MARA','RIOT','HUT','BITF','CIFR','CLSK','IREN','WULF',
    'SMCI','RGTI','DNA','GEVO',
    'SKLZ','FUBO','LMND','ROOT',
    'SPWR','MAXN','ARRY','STEM',
    'LAZR','OUST','MVIS',
    'BILI','PDD','JD','BABA','BIDU',
]

ETFS = ['SPY','QQQ','IWM','ARKK','XBI']

UNIVERSE = list(set(LARGE_CAP + MID_CAP + SMALL_CAP + ETFS))

GAP_THRESHOLD = 0.05
END_DATE = datetime(2026, 2, 14)
START_DATE = datetime(2021, 2, 15)

# ── Alpaca data fetch (paginated) ────────────────────────────────────
def fetch_alpaca_daily(symbol, start, end):
    """Fetch daily bars from Alpaca with pagination."""
    all_bars = []
    url = f'{BASE_URL}/{symbol}/bars'
    params = {
        'timeframe': '1Day',
        'start': start.strftime('%Y-%m-%dT00:00:00Z'),
        'end': end.strftime('%Y-%m-%dT00:00:00Z'),
        'limit': 10000,
        'feed': 'sip',
        'adjustment': 'split',
    }

    while True:
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(1)
            continue
        if resp.status_code != 200:
            return None
        data = resp.json()
        bars = data.get('bars', [])
        if not bars:
            break
        all_bars.extend(bars)
        npt = data.get('next_page_token')
        if not npt:
            break
        params['page_token'] = npt

    if not all_bars:
        return None

    df = pd.DataFrame(all_bars)
    df['datetime'] = pd.to_datetime(df['t'])
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    df = df.set_index('datetime').sort_index()
    df = df[['open','high','low','close','volume']].apply(pd.to_numeric, errors='coerce')
    df.index.name = symbol
    return df.dropna()

# ── Categorize ───────────────────────────────────────────────────────
CAP_MAP = {}
for s in LARGE_CAP + ETFS: CAP_MAP[s] = 'Large/ETF'
for s in MID_CAP: CAP_MAP[s] = 'Mid'
for s in SMALL_CAP: CAP_MAP[s] = 'Small/Micro'

# ── Find gap-ups ─────────────────────────────────────────────────────
def find_gaps(all_data):
    records = []
    for symbol, df in all_data.items():
        if len(df) < 20:
            continue
        o = df['open'].values
        c = df['close'].values
        h = df['high'].values
        l = df['low'].values
        v = df['volume'].values
        dates = df.index
        avg_vol = pd.Series(v).rolling(20).mean().values

        for i in range(1, len(df)):
            prev_c = c[i-1]
            if prev_c <= 0 or o[i] <= 0:
                continue
            gap = (o[i] - prev_c) / prev_c
            if gap < GAP_THRESHOLD:
                continue

            cur_avg_vol = avg_vol[i] if not np.isnan(avg_vol[i]) else v[i]

            records.append({
                'symbol': symbol,
                'date': dates[i],
                'year': dates[i].year,
                'month': dates[i].month,
                'weekday': dates[i].strftime('%a'),
                'cap_bucket': CAP_MAP.get(symbol, 'Unknown'),
                'prev_close': round(prev_c, 2),
                'open': round(o[i], 2),
                'high': round(h[i], 2),
                'low': round(l[i], 2),
                'close': round(c[i], 2),
                'volume': int(v[i]),
                'avg_vol_20': int(cur_avg_vol) if cur_avg_vol > 0 else int(v[i]),
                'price': round(o[i], 2),
                'gap_pct': round(gap * 100, 2),
                'vol_ratio': round(v[i] / cur_avg_vol, 2) if cur_avg_vol > 0 else 1.0,
                'faded': c[i] < o[i],
                'filled': c[i] <= prev_c,
                'close_vs_open_pct': round((c[i] - o[i]) / o[i] * 100, 2),
                'high_vs_open_pct': round((h[i] - o[i]) / o[i] * 100, 2),
                'low_vs_open_pct': round((l[i] - o[i]) / o[i] * 100, 2),
                'short_pnl': round((o[i] - c[i]) / o[i] * 100, 2),
            })
    return pd.DataFrame(records)

# ── Analysis helpers ─────────────────────────────────────────────────
def pnl_with_stop(df, stop_pct):
    stop_price = df['open'] * (1 + stop_pct)
    stopped = df['high'] >= stop_price
    pnl = np.where(stopped, -stop_pct * 100, df['short_pnl'])
    return pnl, stopped

def section(title):
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")

def print_table(headers, rows, widths):
    hdr = '  '.join(f'{h:>{w}}' for h, w in zip(headers, widths))
    sep = '  '.join('-' * w for w in widths)
    print(f"  {hdr}")
    print(f"  {sep}")
    for row in rows:
        print(f"  {'  '.join(f'{str(v):>{w}}' for v, w in zip(row, widths))}")

# ── Main report ──────────────────────────────────────────────────────
def report(df):
    N = len(df)
    faded = df['faded'].sum()
    filled = df['filled'].sum()
    fr = faded / N * 100
    flr = filled / N * 100

    section(f"GAP FADE STUDY — ALPACA SIP DATA — {N:,} events")
    print(f"  Source:          Alpaca Markets (SIP feed, split-adjusted)")
    print(f"  Symbols:         {df['symbol'].nunique()} stocks")
    print(f"  Period:          {START_DATE.date()} → {END_DATE.date()} (5 years)")
    print(f"  Gap threshold:   >{GAP_THRESHOLD*100:.0f}%")

    # ── 1. Core question ────────────────────────────────────
    section("1. CORE QUESTION — What % fade?")
    print(f"""
  ┌───────────────────────────────────────────────────────────┐
  │  Closed BELOW open (faded):      {faded:>5,} / {N:>5,}  = {fr:5.1f}%    │
  │  Closed ABOVE open (held):       {N-faded:>5,} / {N:>5,}  = {100-fr:5.1f}%    │
  │  Full gap fill (< prev close):   {filled:>5,} / {N:>5,}  = {flr:5.1f}%    │
  └───────────────────────────────────────────────────────────┘""")

    # ── 2. By gap size ──────────────────────────────────────
    section("2. BY GAP SIZE")
    bins = [(5,7),(7,10),(10,15),(15,20),(20,30),(30,50),(50,100),(100,500)]
    rows = []
    for lo, hi in bins:
        s = df[(df['gap_pct'] >= lo) & (df['gap_pct'] < hi)]
        if len(s) < 3: continue
        rows.append((f'{lo}-{hi}%', f'{len(s):,}', f'{s["faded"].mean()*100:.1f}%',
                      f'{s["filled"].mean()*100:.1f}%', f'{s["short_pnl"].mean():+.2f}%',
                      f'{s["short_pnl"].median():+.2f}%'))
    print_table(['Gap','Count','Fade%','Fill%','AvgPnL','MedPnL'], rows, [10,7,7,7,8,8])

    # ── 3. By market cap ────────────────────────────────────
    section("3. BY MARKET CAP")
    rows = []
    for b in ['Large/ETF','Mid','Small/Micro']:
        s = df[df['cap_bucket'] == b]
        if len(s) < 5: continue
        rows.append((b, f'{len(s):,}', f'{s["faded"].mean()*100:.1f}%',
                      f'{s["short_pnl"].mean():+.2f}%', f'{s["gap_pct"].mean():.1f}%'))
    print_table(['Cap','Count','Fade%','AvgPnL','AvgGap'], rows, [12,7,7,8,8])

    # ── 4. By year ──────────────────────────────────────────
    section("4. BY YEAR")
    rows = []
    for yr in sorted(df['year'].unique()):
        s = df[df['year'] == yr]
        if len(s) < 5: continue
        rows.append((str(yr), f'{len(s):,}', f'{s["faded"].mean()*100:.1f}%',
                      f'{s["short_pnl"].mean():+.2f}%', f'{s["short_pnl"].median():+.2f}%'))
    print_table(['Year','Count','Fade%','AvgPnL','MedPnL'], rows, [6,7,7,8,8])

    # ── 5. By day of week ───────────────────────────────────
    section("5. BY DAY OF WEEK")
    rows = []
    for day in ['Mon','Tue','Wed','Thu','Fri']:
        s = df[df['weekday'] == day]
        if len(s) < 5: continue
        rows.append((day, f'{len(s):,}', f'{s["faded"].mean()*100:.1f}%',
                      f'{s["short_pnl"].mean():+.2f}%'))
    print_table(['Day','Count','Fade%','AvgPnL'], rows, [5,7,7,8])

    # ── 6. Volume spike ─────────────────────────────────────
    section("6. VOLUME RATIO (gap day vol / 20d avg)")
    vol_bins = [(0,0.5),(0.5,1),(1,2),(2,3),(3,5),(5,10),(10,100)]
    rows = []
    for lo, hi in vol_bins:
        s = df[(df['vol_ratio'] >= lo) & (df['vol_ratio'] < hi)]
        if len(s) < 5: continue
        rows.append((f'{lo}-{hi}x', f'{len(s):,}', f'{s["faded"].mean()*100:.1f}%',
                      f'{s["short_pnl"].mean():+.2f}%', f'{s["gap_pct"].mean():.1f}%'))
    print_table(['VolRatio','Count','Fade%','AvgPnL','AvgGap'], rows, [10,7,7,8,8])

    # ── 7. Price level ──────────────────────────────────────
    section("7. BY STOCK PRICE")
    pbins = [(0,5),(5,15),(15,30),(30,50),(50,100),(100,300),(300,5000)]
    rows = []
    for lo, hi in pbins:
        s = df[(df['price'] >= lo) & (df['price'] < hi)]
        if len(s) < 5: continue
        rows.append((f'${lo}-${hi}', f'{len(s):,}', f'{s["faded"].mean()*100:.1f}%',
                      f'{s["short_pnl"].mean():+.2f}%'))
    print_table(['Price','Count','Fade%','AvgPnL'], rows, [12,7,7,8])

    # ── 8. Liquidity filter ─────────────────────────────────
    section("8. LIQUIDITY FILTER (avg daily volume)")
    for vmin, label in [(0,'All'), (100_000,'Vol>100K'), (500_000,'Vol>500K'),
                        (1_000_000,'Vol>1M'), (5_000_000,'Vol>5M')]:
        s = df[df['avg_vol_20'] >= vmin]
        if len(s) < 5: continue
        print(f"  {label:<12}  {len(s):>5,} gaps   Fade: {s['faded'].mean()*100:.1f}%   Avg short: {s['short_pnl'].mean():+.2f}%")

    # ── 9. Stop loss comparison ─────────────────────────────
    section("9. STOP LOSS LEVELS")
    rows = []
    for stop in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
        pnl, stopped = pnl_with_stop(df, stop)
        wins = (pnl > 0).sum()
        total = len(pnl)
        avg_w = pnl[pnl > 0].mean() if wins > 0 else 0
        avg_l = pnl[pnl <= 0].mean() if (total - wins) > 0 else 0
        pf = abs(pnl[pnl > 0].sum() / pnl[pnl <= 0].sum()) if pnl[pnl <= 0].sum() != 0 else 999
        rows.append((f'{stop*100:.0f}%', f'{wins/total*100:.1f}%', f'{pnl.mean():+.2f}%',
                      f'{np.median(pnl):+.2f}%', f'{avg_w:+.2f}%', f'{avg_l:+.2f}%',
                      f'{pf:.2f}', f'{stopped.mean()*100:.1f}%'))
    print_table(['Stop','Win%','AvgPnL','MedPnL','AvgWin','AvgLoss','PF','Stopped%'], rows,
                [5,6,8,8,8,8,6,9])

    # ── 10. Intraday range ──────────────────────────────────
    section("10. INTRADAY RANGE")
    print(f"  Avg high above open:   {df['high_vs_open_pct'].mean():+.2f}%  (adverse for shorts)")
    print(f"  Med high above open:   {df['high_vs_open_pct'].median():+.2f}%")
    print(f"  Avg low below open:    {df['low_vs_open_pct'].mean():+.2f}%  (favorable for shorts)")
    print(f"  Med low below open:    {df['low_vs_open_pct'].median():+.2f}%")
    print(f"  Avg close vs open:     {df['close_vs_open_pct'].mean():+.2f}%")
    print(f"  Med close vs open:     {df['close_vs_open_pct'].median():+.2f}%")

    # ── 11. Combined filters ────────────────────────────────
    section("11. COMBINED FILTERS — WHERE IS THE EDGE?")
    filters = [
        ('ALL (baseline)',                df),
        ('Gap 5-10%',                     df[(df['gap_pct']>=5)&(df['gap_pct']<10)]),
        ('Gap 10-20%',                    df[(df['gap_pct']>=10)&(df['gap_pct']<20)]),
        ('Gap 20%+',                      df[df['gap_pct']>=20]),
        ('Small/Micro only',              df[df['cap_bucket']=='Small/Micro']),
        ('Small + Gap 5-10%',             df[(df['cap_bucket']=='Small/Micro')&(df['gap_pct']>=5)&(df['gap_pct']<10)]),
        ('Small + Gap 10%+',              df[(df['cap_bucket']=='Small/Micro')&(df['gap_pct']>=10)]),
        ('Vol < 1x (low vol gap)',        df[df['vol_ratio']<1]),
        ('Vol < 0.5x (very low vol)',     df[df['vol_ratio']<0.5]),
        ('Vol 1-2x (normal)',             df[(df['vol_ratio']>=1)&(df['vol_ratio']<2)]),
        ('Vol > 3x (high vol)',           df[df['vol_ratio']>=3]),
        ('Vol > 5x (extreme vol)',        df[df['vol_ratio']>=5]),
        ('Price < $15',                   df[df['price']<15]),
        ('Price < $5',                    df[df['price']<5]),
        ('Small + Vol<1x',               df[(df['cap_bucket']=='Small/Micro')&(df['vol_ratio']<1)]),
        ('Small + Vol<1x + Gap<10%',     df[(df['cap_bucket']=='Small/Micro')&(df['vol_ratio']<1)&(df['gap_pct']<10)]),
        ('Large + Vol<1x',               df[(df['cap_bucket']=='Large/ETF')&(df['vol_ratio']<1)]),
        ('Price<$15 + Vol<1x',           df[(df['price']<15)&(df['vol_ratio']<1)]),
        ('Vol<1x + Gap 5-10%',           df[(df['vol_ratio']<1)&(df['gap_pct']>=5)&(df['gap_pct']<10)]),
        ('Vol<1x + Gap 10%+',            df[(df['vol_ratio']<1)&(df['gap_pct']>=10)]),
        ('AvgVol>500K + Vol<1x',         df[(df['avg_vol_20']>=500_000)&(df['vol_ratio']<1)]),
        ('AvgVol>1M + Vol<1x',           df[(df['avg_vol_20']>=1_000_000)&(df['vol_ratio']<1)]),
    ]
    rows = []
    for label, sub in filters:
        if len(sub) < 5:
            rows.append((label, '<5', '-', '-', '-'))
            continue
        fr = sub['faded'].mean() * 100
        mark = ' <<<' if fr >= 70 else (' <<' if fr >= 65 else (' <' if fr >= 60 else ''))
        rows.append((label, f'{len(sub):,}', f'{fr:.1f}%',
                      f'{sub["short_pnl"].mean():+.2f}%',
                      f'{sub["short_pnl"].median():+.2f}%{mark}'))
    print_table(['Filter','Count','Fade%','AvgPnL','MedPnL'], rows, [30,7,7,8,14])

    # ── 12. Top faders ──────────────────────────────────────
    section("12. TOP 20 SYMBOLS — HIGHEST FADE RATE (min 10 gaps)")
    stats = df.groupby('symbol').agg(
        count=('faded','size'), fade_rate=('faded','mean'),
        avg_pnl=('short_pnl','mean'), avg_gap=('gap_pct','mean'),
        cap=('cap_bucket','first'),
    )
    top = stats[stats['count']>=10].sort_values('fade_rate', ascending=False).head(20)
    rows = []
    for sym, r in top.iterrows():
        rows.append((sym, r['cap'], f'{int(r["count"])}', f'{r["fade_rate"]*100:.1f}%',
                      f'{r["avg_pnl"]:+.2f}%', f'{r["avg_gap"]:.1f}%'))
    print_table(['Symbol','Cap','Gaps','Fade%','AvgPnL','AvgGap'], rows, [8,12,5,7,8,7])

    # ── 13. Worst faders ────────────────────────────────────
    section("13. BOTTOM 15 — GAPS THAT HOLD (don't short these)")
    worst = stats[stats['count']>=10].sort_values('fade_rate').head(15)
    rows = []
    for sym, r in worst.iterrows():
        rows.append((sym, r['cap'], f'{int(r["count"])}', f'{r["fade_rate"]*100:.1f}%',
                      f'{r["avg_pnl"]:+.2f}%', f'{r["avg_gap"]:.1f}%'))
    print_table(['Symbol','Cap','Gaps','Fade%','AvgPnL','AvgGap'], rows, [8,12,5,7,8,7])

    # ── 14. Kelly criterion estimate ────────────────────────
    section("14. KELLY CRITERION ESTIMATE (3% stop)")
    pnl3, _ = pnl_with_stop(df, 0.03)
    p = (pnl3 > 0).mean()
    q = 1 - p
    avg_w = pnl3[pnl3 > 0].mean()
    avg_l = abs(pnl3[pnl3 <= 0].mean())
    b = avg_w / avg_l if avg_l > 0 else 1
    kelly = (p * b - q) / b if b > 0 else 0
    print(f"  Win rate (p):       {p:.3f}")
    print(f"  Loss rate (q):      {q:.3f}")
    print(f"  Avg win / avg loss: {b:.3f}")
    print(f"  Full Kelly:         {kelly:.3f}  ({kelly*100:.1f}% of capital)")
    print(f"  Half Kelly:         {kelly/2:.3f}  ({kelly*100/2:.1f}% of capital)")
    print(f"  Quarter Kelly:      {kelly/4:.3f}  ({kelly*100/4:.1f}% of capital)")

    # ── Summary ─────────────────────────────────────────────
    section("SUMMARY")
    print(f"""
  Total events:          {N:,}
  Overall fade rate:     {fr:.1f}%
  Avg short P&L:         {df['short_pnl'].mean():+.2f}%
  Median short P&L:      {df['short_pnl'].median():+.2f}%
""")

    # Best filter
    best_label, best_fr, best_n = 'Baseline', df['faded'].mean()*100, N
    for label, sub in filters:
        if len(sub) >= 20:
            f = sub['faded'].mean() * 100
            if f > best_fr:
                best_fr, best_label, best_n = f, label, len(sub)

    print(f"  Best filter:  '{best_label}'")
    print(f"    → {best_n:,} trades, {best_fr:.1f}% fade rate")
    if best_fr >= 75:
        print(f"\n  >>> The 80% edge EXISTS with the right filters <<<")
    elif best_fr >= 65:
        print(f"\n  Edge exists ({best_fr:.0f}%) but below the 80% claim.")
    else:
        print(f"\n  No filter reaches 80%. Max found: {best_fr:.0f}%.")
    print()


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if not API_KEY:
        print("ERROR: ALPACA_API_KEY not set. Check .env file.")
        sys.exit(1)

    print(f"Fetching daily bars from Alpaca ({len(UNIVERSE)} symbols)...")
    all_data = {}
    failed = []
    for idx, sym in enumerate(sorted(UNIVERSE)):
        pct = (idx + 1) / len(UNIVERSE) * 100
        sys.stdout.write(f'\r  [{idx+1}/{len(UNIVERSE)}] {pct:.0f}% — {sym:<8}')
        sys.stdout.flush()
        df = fetch_alpaca_daily(sym, START_DATE, END_DATE)
        if df is not None and len(df) > 20:
            all_data[sym] = df
        else:
            failed.append(sym)
        # Rate limit: Alpaca free tier = 200 req/min
        if (idx + 1) % 100 == 0:
            time.sleep(2)

    print(f'\n  Loaded {len(all_data)} symbols, {len(failed)} failed')
    if failed:
        print(f'  Failed: {", ".join(failed[:20])}{"..." if len(failed) > 20 else ""}')

    gaps = find_gaps(all_data)
    if len(gaps) > 0:
        report(gaps)
        gaps.to_csv('gap_fade_alpaca_results.csv', index=False)
        print(f"  Raw data saved to gap_fade_alpaca_results.csv ({len(gaps):,} rows)\n")
    else:
        print("No gap events found.")
