#!/usr/bin/env python3
"""
Gap Fade Deep Study — Comprehensive analysis with 500+ stocks.
Goal: Find where the 79-80% fade rate actually lives (if anywhere).
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings, sys
warnings.filterwarnings('ignore')

# ── Build a broad universe ───────────────────────────────────────────
# S&P 500 representative sample + mid/small caps + high-gap-probability names

LARGE_CAP = [
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','BRK-B','JPM','JNJ',
    'V','PG','UNH','HD','MA','DIS','PYPL','BAC','INTC','VZ',
    'NFLX','ADBE','CRM','CMCSA','PFE','TMO','ABT','CSCO','PEP','AVGO',
    'ACN','COST','NKE','MRK','WMT','LLY','MCD','DHR','TXN','QCOM',
    'MDT','HON','UPS','LOW','MS','GS','BLK','ISRG','SCHW','AXP',
    'BA','CAT','DE','GE','LMT','RTX','MMM','IBM','ORCL','AMD',
    'NOW','SNOW','UBER','ABNB','COIN','SQ','SHOP','MELI','SE','SPOT',
    'ZM','DOCU','CRWD','ZS','DDOG','NET','MDB','TEAM','OKTA','TWLO',
]

MID_CAP = [
    'ROKU','SNAP','PINS','ETSY','DASH','LYFT','HOOD','DKNG','AFRM','SOFI',
    'UPST','RBLX','U','PATH','BILL','HUBS','VEEV','PAYC','PCTY','GNRC',
    'ENPH','FSLR','SEDG','RUN','NOVA','PLUG','CHPT','QS','BLNK','EVGO',
    'RIVN','LCID','FSR','NKLA','GOEV','XPEV','LI','NIO','POLESTAR-AUTO.ST',
    'W','CHWY','PTON','BYND','OATLY','TTCF','OPEN','OFFERPAD','CLOV','WISH',
    'PLTR','IONQ','RKLB','JOBY','SPCE','ASTS','RDW','ASTR','MNTS','VORB',
    'AI','BBAI','SOUN','BIGB','PRCT','RXRX','DNAY','TWST','CRSP','EDIT',
    'NTLA','BEAM','VERV','IONS','REGN','VRTX','BIIB','MRNA','BNTX','ARCT',
]

SMALL_CAP_HIGH_VOL = [
    'GME','AMC','BBBY','KOSS','BB','NOK','EXPR','CLOV','WISH','WKHS',
    'MARA','RIOT','BTBT','HUT','BITF','CIFR','CLSK','IREN','CORZ','WULF',
    'SMCI','SOUN','IONQ','RGTI','QUBT','QBTS','DNA','GEVO','TELL','CLNE',
    'MULN','FFIE','GOEV','RIDE','HYLN','XL','ARVL','REE','FREY','MVST',
    'SKLZ','GENI','RSI','BETZ','FUBO','OPEN','UWMC','RKT','LMND','ROOT',
    'SPWR','MAXN','ARRY','STEM','BLDP','PTRA','NUVB','ACHR','LILM','EVTL',
    'LAZR','VLDR','OUST','AEVA','CPTN','INVZ','MVIS','LIDR','AEYE',
    'IQ','HUYA','DOYU','BILI','TME','PDD','JD','BABA','BIDU','NTES',
]

# ETFs for baseline comparison
ETFS = ['SPY','QQQ','IWM','ARKK','ARKG','ARKF','XBI','SOXL','TQQQ','TNA']

UNIVERSE = list(set(LARGE_CAP + MID_CAP + SMALL_CAP_HIGH_VOL + ETFS))

GAP_THRESHOLD = 0.05
LOOKBACK_YEARS = 5
END_DATE = datetime(2026, 2, 14)
START_DATE = END_DATE - timedelta(days=LOOKBACK_YEARS * 365)

def fetch_data():
    print(f"Downloading {len(UNIVERSE)} symbols ({START_DATE.date()} → {END_DATE.date()})...")
    all_data = {}
    batch_size = 50
    for i in range(0, len(UNIVERSE), batch_size):
        batch = UNIVERSE[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}/{(len(UNIVERSE)-1)//batch_size + 1} ({len(batch)} symbols)...")
        try:
            data = yf.download(batch, start=START_DATE, end=END_DATE,
                             auto_adjust=False, progress=False, threads=True)
            if data.empty:
                continue
            # yfinance returns MultiIndex columns: (Price, Ticker)
            if isinstance(data.columns, pd.MultiIndex):
                # Get list of tickers in the downloaded data
                tickers_in_data = data.columns.get_level_values(1).unique().tolist()
                for sym in tickers_in_data:
                    try:
                        df = data.xs(sym, level=1, axis=1).dropna(subset=['Open','Close'])
                        if len(df) > 10:
                            all_data[sym] = df
                    except:
                        pass
            else:
                # Single ticker
                if len(batch) == 1:
                    df = data.dropna(subset=['Open','Close'])
                    if len(df) > 10:
                        all_data[batch[0]] = df
        except Exception as e:
            print(f"    Batch error: {e}")
    print(f"  Got data for {len(all_data)} symbols")
    return all_data

def compute_market_cap_bucket(symbol):
    """Rough categorization."""
    if symbol in LARGE_CAP or symbol in ETFS:
        return 'Large/ETF'
    elif symbol in MID_CAP:
        return 'Mid'
    else:
        return 'Small/Micro'

def find_gaps(all_data):
    records = []
    for symbol, df in all_data.items():
        try:
            df = df.dropna(subset=['Open','Close','High','Low','Volume'])
            if len(df) < 20:
                continue
            o = df['Open'].values
            c = df['Close'].values
            h = df['High'].values
            l = df['Low'].values
            v = df['Volume'].values
            dates = df.index

            # Compute avg volume (20-day rolling)
            avg_vol_20 = pd.Series(v).rolling(20).mean().values

            for i in range(1, len(df)):
                prev_c = c[i-1]
                if prev_c <= 0 or o[i] <= 0:
                    continue
                gap = (o[i] - prev_c) / prev_c
                if gap < GAP_THRESHOLD:
                    continue

                cur_avg_vol = avg_vol_20[i] if not np.isnan(avg_vol_20[i]) else v[i]
                price = o[i]

                records.append({
                    'symbol': symbol,
                    'date': dates[i],
                    'year': dates[i].year,
                    'month': dates[i].month,
                    'weekday': dates[i].strftime('%a'),
                    'cap_bucket': compute_market_cap_bucket(symbol),
                    'prev_close': round(prev_c, 2),
                    'open': round(o[i], 2),
                    'high': round(h[i], 2),
                    'low': round(l[i], 2),
                    'close': round(c[i], 2),
                    'volume': int(v[i]),
                    'avg_vol_20': int(cur_avg_vol),
                    'price': round(price, 2),
                    'gap_pct': round(gap * 100, 2),
                    'vol_ratio': round(v[i] / cur_avg_vol, 2) if cur_avg_vol > 0 else 1.0,
                    # Key metrics
                    'faded': c[i] < o[i],
                    'filled': c[i] <= prev_c,
                    'close_vs_open_pct': round((c[i] - o[i]) / o[i] * 100, 2),
                    'high_vs_open_pct': round((h[i] - o[i]) / o[i] * 100, 2),
                    'low_vs_open_pct': round((l[i] - o[i]) / o[i] * 100, 2),
                    'short_pnl': round((o[i] - c[i]) / o[i] * 100, 2),
                })
        except Exception as e:
            continue
    return pd.DataFrame(records)

def pnl_with_stop(df, stop_pct):
    """Short at open, stop at open*(1+stop_pct), cover at close or stop."""
    stop_price = df['open'] * (1 + stop_pct)
    stopped = df['high'] >= stop_price
    pnl = np.where(stopped, -stop_pct * 100, df['short_pnl'])
    return pnl, stopped

def section(title):
    w = 78
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}")

def subsection(title):
    w = 78
    print(f"\n{'-'*w}")
    print(f"  {title}")
    print(f"{'-'*w}")

def print_table(headers, rows, col_widths=None):
    """Print a formatted table."""
    if col_widths is None:
        col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=4)) + 2
                      for i, h in enumerate(headers)]

    hdr = '  '.join(f'{h:>{w}}' for h, w in zip(headers, col_widths))
    sep = '  '.join('-' * w for w in col_widths)
    print(f"  {hdr}")
    print(f"  {sep}")
    for row in rows:
        line = '  '.join(f'{str(v):>{w}}' for v, w in zip(row, col_widths))
        print(f"  {line}")

def report(df):
    N = len(df)
    if N == 0:
        print("No gap-ups found!")
        return

    # ════════════════════════════════════════════════════════════
    section(f"GAP FADE DEEP STUDY — {N:,} gap-up events")
    print(f"  Symbols with data:   {df['symbol'].nunique()}")
    print(f"  Period:              {START_DATE.date()} → {END_DATE.date()}")
    print(f"  Gap threshold:       >{GAP_THRESHOLD*100:.0f}%")
    print(f"  Avg gap size:        {df['gap_pct'].mean():.1f}%")
    print(f"  Median gap size:     {df['gap_pct'].median():.1f}%")

    # ════════════════════════════════════════════════════════════
    section("1. THE CORE QUESTION — What % of gap-ups close below open?")

    faded = df['faded'].sum()
    filled = df['filled'].sum()
    fr = faded / N * 100
    flr = filled / N * 100

    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  Closed BELOW open (faded):    {faded:>5,} / {N:,}  = {fr:.1f}%  │
  │  Closed ABOVE open (held):     {N-faded:>5,} / {N:,}  = {100-fr:.1f}%  │
  │  Full gap fill (below prev C): {filled:>5,} / {N:,}  = {flr:.1f}%  │
  └─────────────────────────────────────────────────────┘

  Verdict vs 79-80% claim:  {"CONFIRMED" if fr >= 75 else "PARTIAL (" + f"{fr:.0f}%" + ")" if fr >= 60 else "NOT SUPPORTED (" + f"{fr:.0f}%" + ")"}
""")

    # ════════════════════════════════════════════════════════════
    section("2. FADE RATE BY GAP SIZE")
    print("  Does gap size predict fade probability?\n")

    bins = [(5,7), (7,10), (10,15), (15,20), (20,30), (30,50), (50,100), (100,500)]
    rows = []
    for lo, hi in bins:
        m = (df['gap_pct'] >= lo) & (df['gap_pct'] < hi)
        s = df[m]
        if len(s) < 3:
            continue
        rows.append((
            f'{lo}-{hi}%',
            f'{len(s):,}',
            f'{s["faded"].mean()*100:.1f}%',
            f'{s["filled"].mean()*100:.1f}%',
            f'{s["short_pnl"].mean():+.2f}%',
            f'{s["short_pnl"].median():+.2f}%',
        ))
    print_table(['Gap Range','Count','Fade%','Fill%','Avg Short','Med Short'], rows,
                [10, 7, 7, 7, 10, 10])

    # ════════════════════════════════════════════════════════════
    section("3. FADE RATE BY MARKET CAP")
    print("  Small caps vs large caps — where does the edge live?\n")

    rows = []
    for bucket in ['Large/ETF','Mid','Small/Micro']:
        s = df[df['cap_bucket'] == bucket]
        if len(s) < 5:
            continue
        rows.append((
            bucket,
            f'{len(s):,}',
            f'{s["faded"].mean()*100:.1f}%',
            f'{s["filled"].mean()*100:.1f}%',
            f'{s["short_pnl"].mean():+.2f}%',
            f'{s["short_pnl"].median():+.2f}%',
            f'{s["gap_pct"].mean():.1f}%',
        ))
    print_table(['Cap','Count','Fade%','Fill%','Avg Short','Med Short','Avg Gap'], rows,
                [12, 7, 7, 7, 10, 10, 8])

    # ════════════════════════════════════════════════════════════
    section("4. FADE RATE BY YEAR")
    print("  Is the edge consistent or regime-dependent?\n")

    rows = []
    for yr in sorted(df['year'].unique()):
        s = df[df['year'] == yr]
        if len(s) < 5:
            continue
        rows.append((
            str(yr),
            f'{len(s):,}',
            f'{s["faded"].mean()*100:.1f}%',
            f'{s["short_pnl"].mean():+.2f}%',
            f'{s["short_pnl"].median():+.2f}%',
        ))
    print_table(['Year','Count','Fade%','Avg Short','Med Short'], rows,
                [6, 7, 7, 10, 10])

    # ════════════════════════════════════════════════════════════
    section("5. FADE RATE BY DAY OF WEEK")
    print("  Monday gaps vs Friday gaps?\n")

    rows = []
    for day in ['Mon','Tue','Wed','Thu','Fri']:
        s = df[df['weekday'] == day]
        if len(s) < 5:
            continue
        rows.append((
            day,
            f'{len(s):,}',
            f'{s["faded"].mean()*100:.1f}%',
            f'{s["short_pnl"].mean():+.2f}%',
        ))
    print_table(['Day','Count','Fade%','Avg Short'], rows,
                [5, 7, 7, 10])

    # ════════════════════════════════════════════════════════════
    section("6. VOLUME SPIKE ANALYSIS")
    print("  Does abnormal volume on gap day predict fade?\n")

    vol_bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 100)]
    rows = []
    for lo, hi in vol_bins:
        m = (df['vol_ratio'] >= lo) & (df['vol_ratio'] < hi)
        s = df[m]
        if len(s) < 5:
            continue
        rows.append((
            f'{lo}-{hi}x',
            f'{len(s):,}',
            f'{s["faded"].mean()*100:.1f}%',
            f'{s["short_pnl"].mean():+.2f}%',
            f'{s["gap_pct"].mean():.1f}%',
        ))
    print_table(['Vol Ratio','Count','Fade%','Avg Short','Avg Gap'], rows,
                [10, 7, 7, 10, 8])

    # ════════════════════════════════════════════════════════════
    section("7. PRICE LEVEL ANALYSIS")
    print("  Do cheap stocks fade more than expensive ones?\n")

    price_bins = [(0, 5), (5, 15), (15, 30), (30, 50), (50, 100), (100, 500), (500, 5000)]
    rows = []
    for lo, hi in price_bins:
        m = (df['price'] >= lo) & (df['price'] < hi)
        s = df[m]
        if len(s) < 5:
            continue
        rows.append((
            f'${lo}-${hi}',
            f'{len(s):,}',
            f'{s["faded"].mean()*100:.1f}%',
            f'{s["short_pnl"].mean():+.2f}%',
            f'{s["gap_pct"].mean():.1f}%',
        ))
    print_table(['Price Range','Count','Fade%','Avg Short','Avg Gap'], rows,
                [12, 7, 7, 10, 8])

    # ════════════════════════════════════════════════════════════
    section("8. AVERAGE VOLUME FILTER (Liquidity)")
    print("  Your strategy requires avg vol > 500K. How does that affect the edge?\n")

    for vol_min, label in [(0, 'All stocks'), (100_000, 'Vol > 100K'), (500_000, 'Vol > 500K'),
                           (1_000_000, 'Vol > 1M'), (5_000_000, 'Vol > 5M')]:
        s = df[df['avg_vol_20'] >= vol_min]
        if len(s) < 5:
            continue
        fr = s['faded'].mean() * 100
        avg = s['short_pnl'].mean()
        print(f"  {label:<16}  {len(s):>5,} gaps   Fade: {fr:.1f}%   Avg short: {avg:+.2f}%")

    # ════════════════════════════════════════════════════════════
    section("9. STOP LOSS COMPARISON")
    print("  Short at open with various stop levels:\n")

    rows = []
    for stop in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        pnl, stopped = pnl_with_stop(df, stop)
        wins = (pnl > 0).sum()
        total = len(pnl)
        avg_w = pnl[pnl > 0].mean() if wins > 0 else 0
        avg_l = pnl[pnl <= 0].mean() if (total - wins) > 0 else 0
        pf = abs(pnl[pnl > 0].sum() / pnl[pnl <= 0].sum()) if pnl[pnl <= 0].sum() != 0 else 999
        rows.append((
            f'{stop*100:.0f}%',
            f'{wins/total*100:.1f}%',
            f'{pnl.mean():+.2f}%',
            f'{np.median(pnl):+.2f}%',
            f'{avg_w:+.2f}%',
            f'{avg_l:+.2f}%',
            f'{pf:.2f}',
            f'{stopped.mean()*100:.1f}%',
        ))
    print_table(['Stop','Win%','AvgPnL','MedPnL','AvgWin','AvgLoss','PF','Stopped%'], rows,
                [5, 6, 8, 8, 8, 8, 6, 9])

    # ════════════════════════════════════════════════════════════
    section("10. INTRADAY RANGE (from daily OHLC)")
    print("  How much room do gaps have intraday?\n")

    print(f"  Avg high above open:   {df['high_vs_open_pct'].mean():+.2f}%  (adverse move for shorts)")
    print(f"  Med high above open:   {df['high_vs_open_pct'].median():+.2f}%")
    print(f"  Avg low below open:    {df['low_vs_open_pct'].mean():+.2f}%  (favorable move for shorts)")
    print(f"  Med low below open:    {df['low_vs_open_pct'].median():+.2f}%")
    print(f"  Avg close vs open:     {df['close_vs_open_pct'].mean():+.2f}%")
    print(f"  Med close vs open:     {df['close_vs_open_pct'].median():+.2f}%")

    # ════════════════════════════════════════════════════════════
    section("11. COMBINED FILTERS — SEARCHING FOR THE 80% EDGE")
    print("  Applying multiple filters to find the sweet spot:\n")

    filters = [
        ('Baseline (all)', df),
        ('Gap 5-10%', df[(df['gap_pct'] >= 5) & (df['gap_pct'] < 10)]),
        ('Gap 10-20%', df[(df['gap_pct'] >= 10) & (df['gap_pct'] < 20)]),
        ('Gap 20%+', df[df['gap_pct'] >= 20]),
        ('Small/Micro cap', df[df['cap_bucket'] == 'Small/Micro']),
        ('Small + Gap 5-10%', df[(df['cap_bucket'] == 'Small/Micro') & (df['gap_pct'] >= 5) & (df['gap_pct'] < 10)]),
        ('Small + Gap 10%+', df[(df['cap_bucket'] == 'Small/Micro') & (df['gap_pct'] >= 10)]),
        ('Vol ratio > 3x', df[df['vol_ratio'] >= 3]),
        ('Vol ratio > 5x', df[df['vol_ratio'] >= 5]),
        ('Small + Vol>3x', df[(df['cap_bucket'] == 'Small/Micro') & (df['vol_ratio'] >= 3)]),
        ('Price < $15', df[df['price'] < 15]),
        ('Price < $15 + Gap>10%', df[(df['price'] < 15) & (df['gap_pct'] >= 10)]),
        ('Price < $5', df[df['price'] < 5]),
        ('Large + Gap 5-10%', df[(df['cap_bucket'] == 'Large/ETF') & (df['gap_pct'] >= 5) & (df['gap_pct'] < 10)]),
        ('Large + Gap 10%+', df[(df['cap_bucket'] == 'Large/ETF') & (df['gap_pct'] >= 10)]),
        ('AvgVol>500K + Gap 5-10%', df[(df['avg_vol_20'] >= 500_000) & (df['gap_pct'] >= 5) & (df['gap_pct'] < 10)]),
        ('AvgVol>1M + Small', df[(df['avg_vol_20'] >= 1_000_000) & (df['cap_bucket'] == 'Small/Micro')]),
    ]

    rows = []
    for label, sub in filters:
        if len(sub) < 5:
            rows.append((label, '<5', '-', '-', '-'))
            continue
        fr = sub['faded'].mean() * 100
        avg = sub['short_pnl'].mean()
        med = sub['short_pnl'].median()
        marker = ' <<<' if fr >= 70 else (' <<' if fr >= 65 else (' <' if fr >= 60 else ''))
        rows.append((
            label,
            f'{len(sub):,}',
            f'{fr:.1f}%',
            f'{avg:+.2f}%',
            f'{med:+.2f}%{marker}',
        ))
    print_table(['Filter','Count','Fade%','AvgShort','MedShort'], rows,
                [26, 7, 7, 9, 14])

    # ════════════════════════════════════════════════════════════
    section("12. TOP 20 SYMBOLS — HIGHEST FADE RATE (min 10 gaps)")

    sym_stats = df.groupby('symbol').agg(
        count=('faded','size'),
        fade_rate=('faded','mean'),
        avg_pnl=('short_pnl','mean'),
        avg_gap=('gap_pct','mean'),
        cap=('cap_bucket','first'),
    )
    sym_stats = sym_stats[sym_stats['count'] >= 10].sort_values('fade_rate', ascending=False).head(20)

    rows = []
    for sym, r in sym_stats.iterrows():
        rows.append((
            sym,
            r['cap'],
            f'{int(r["count"])}',
            f'{r["fade_rate"]*100:.1f}%',
            f'{r["avg_pnl"]:+.2f}%',
            f'{r["avg_gap"]:.1f}%',
        ))
    print_table(['Symbol','Cap','Gaps','Fade%','AvgShort','AvgGap'], rows,
                [8, 12, 5, 7, 10, 7])

    # ════════════════════════════════════════════════════════════
    section("13. WORST 15 SYMBOLS — LOWEST FADE RATE (min 10 gaps)")
    print("  These HOLD their gaps — don't short these:\n")

    worst = df.groupby('symbol').agg(
        count=('faded','size'),
        fade_rate=('faded','mean'),
        avg_pnl=('short_pnl','mean'),
        avg_gap=('gap_pct','mean'),
        cap=('cap_bucket','first'),
    )
    worst = worst[worst['count'] >= 10].sort_values('fade_rate', ascending=True).head(15)

    rows = []
    for sym, r in worst.iterrows():
        rows.append((
            sym,
            r['cap'],
            f'{int(r["count"])}',
            f'{r["fade_rate"]*100:.1f}%',
            f'{r["avg_pnl"]:+.2f}%',
            f'{r["avg_gap"]:.1f}%',
        ))
    print_table(['Symbol','Cap','Gaps','Fade%','AvgShort','AvgGap'], rows,
                [8, 12, 5, 7, 10, 7])

    # ════════════════════════════════════════════════════════════
    section("SUMMARY")
    print(f"""
  Total gap-up events analyzed:  {N:,}
  Overall fade rate:             {df['faded'].mean()*100:.1f}%
  Overall avg short P&L:         {df['short_pnl'].mean():+.2f}%
  Overall median short P&L:      {df['short_pnl'].median():+.2f}%
""")

    # Find best filter
    best_label, best_fr, best_n = 'Baseline', df['faded'].mean()*100, N
    for label, sub in filters:
        if len(sub) >= 20:
            fr = sub['faded'].mean() * 100
            if fr > best_fr:
                best_fr = fr
                best_label = label
                best_n = len(sub)

    print(f"  Best filter found:  '{best_label}'")
    print(f"    → {best_n:,} trades, {best_fr:.1f}% fade rate")

    if best_fr >= 75:
        print(f"\n  ✓ The 79-80% edge EXISTS in filtered data.")
    elif best_fr >= 65:
        print(f"\n  ~ There's an edge ({best_fr:.0f}%) but not as strong as 79-80%.")
    else:
        print(f"\n  ✗ Even with filters, the 79-80% claim doesn't hold.")

    print()


if __name__ == '__main__':
    data = fetch_data()
    gaps = find_gaps(data)
    if len(gaps) > 0:
        report(gaps)
        gaps.to_csv('gap_fade_deep_results.csv', index=False)
        print(f"  Raw data: gap_fade_deep_results.csv ({len(gaps):,} rows)\n")
    else:
        print("No gap events found.")
