#!/usr/bin/env python3
"""
Gap Fade Study: Do stocks that gap up >5% at open tend to close below open?

Thesis: ~79-80% of stocks gapping up >5% close below their open ("gap and crap")
Test: Pull historical daily data, find all gap-ups, measure fade rate and short P&L.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Universe: liquid, popular stocks across sectors ──────────────────────
UNIVERSE = [
    # Mega-cap tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'CRM',
    # Semi / tech
    'INTC', 'AVGO', 'MU', 'QCOM', 'MRVL', 'ARM', 'SMCI', 'PLTR', 'SNOW', 'SHOP',
    # Meme / high-vol
    'GME', 'AMC', 'BBBY', 'RIVN', 'LCID', 'SOFI', 'NIO', 'PLUG', 'MARA', 'RIOT',
    # Biotech (gap-heavy)
    'MRNA', 'BNTX', 'CRSP', 'EDIT', 'NTLA', 'BEAM', 'IONS', 'REGN', 'VRTX', 'BIIB',
    # Finance / industrial
    'JPM', 'GS', 'BAC', 'C', 'WFC', 'BA', 'CAT', 'DE', 'GE', 'LMT',
    # Consumer / retail
    'WMT', 'TGT', 'COST', 'HD', 'LOW', 'NKE', 'SBUX', 'MCD', 'DIS', 'ABNB',
    # Energy / materials
    'XOM', 'CVX', 'OXY', 'SLB', 'FSLR', 'ENPH', 'LI', 'XPEV',
    # SPACs / recent IPOs (gap-prone)
    'IONQ', 'RKLB', 'JOBY', 'AFRM', 'HOOD', 'COIN', 'DKNG', 'DASH', 'UBER', 'LYFT',
    # ETFs for reference
    'SPY', 'QQQ', 'IWM', 'ARKK',
]

GAP_THRESHOLD = 0.05      # 5% gap-up minimum
STOP_LOSS_PCTS = [0.02, 0.03, 0.05]  # Stop losses to test: 2%, 3%, 5% above open
LOOKBACK_YEARS = 5
END_DATE = datetime(2026, 2, 14)
START_DATE = END_DATE - timedelta(days=LOOKBACK_YEARS * 365)

def fetch_all_data():
    """Download daily data for the universe."""
    print(f"Fetching {len(UNIVERSE)} symbols, {START_DATE.date()} to {END_DATE.date()}...")
    # yfinance batch download
    data = yf.download(
        UNIVERSE,
        start=START_DATE,
        end=END_DATE,
        group_by='ticker',
        auto_adjust=False,
        progress=True,
        threads=True,
    )
    return data

def find_gap_ups(data):
    """Find all instances where a stock gapped up > threshold at open."""
    records = []

    for symbol in UNIVERSE:
        try:
            # Extract single-stock OHLCV
            if len(UNIVERSE) > 1:
                df = data[symbol].dropna(subset=['Open', 'Close', 'High', 'Low', 'Volume'])
            else:
                df = data.dropna(subset=['Open', 'Close', 'High', 'Low', 'Volume'])

            if len(df) < 10:
                continue

            opens  = df['Open'].values
            closes = df['Close'].values
            highs  = df['High'].values
            lows   = df['Low'].values
            vols   = df['Volume'].values
            dates  = df.index

            for i in range(1, len(df)):
                prev_close = closes[i - 1]
                if prev_close <= 0:
                    continue

                gap_pct = (opens[i] - prev_close) / prev_close

                if gap_pct >= GAP_THRESHOLD:
                    o = opens[i]
                    c = closes[i]
                    h = highs[i]
                    l = lows[i]

                    # Intraday range info
                    high_from_open_pct = (h - o) / o   # How far above open it went
                    low_from_open_pct  = (l - o) / o    # How far below open it went
                    close_vs_open_pct  = (c - o) / o    # Close relative to open

                    # Short P&L (short at open, cover at close)
                    short_pnl_pct = (o - c) / o  # Positive = profit

                    # Faded = closed below open
                    faded = c < o

                    # Filled gap = closed below previous close
                    filled_gap = c < prev_close

                    records.append({
                        'symbol': symbol,
                        'date': dates[i],
                        'prev_close': round(prev_close, 2),
                        'open': round(o, 2),
                        'high': round(h, 2),
                        'low': round(l, 2),
                        'close': round(c, 2),
                        'volume': int(vols[i]),
                        'gap_pct': round(gap_pct * 100, 2),
                        'high_above_open_pct': round(high_from_open_pct * 100, 2),
                        'low_below_open_pct': round(low_from_open_pct * 100, 2),
                        'close_vs_open_pct': round(close_vs_open_pct * 100, 2),
                        'short_pnl_pct': round(short_pnl_pct * 100, 2),
                        'faded': faded,
                        'filled_gap': filled_gap,
                    })
        except Exception as e:
            print(f"  Skip {symbol}: {e}")
            continue

    return pd.DataFrame(records)

def analyze_with_stops(gaps_df, stop_pcts=STOP_LOSS_PCTS):
    """Calculate short P&L with stop losses (using intraday high as worst case)."""
    results = {}
    for stop in stop_pcts:
        stop_price_col = f'stop_{int(stop*100)}pct'
        pnl_col = f'pnl_stop_{int(stop*100)}pct'

        # Stop price = open * (1 + stop%)
        gaps_df[stop_price_col] = gaps_df['open'] * (1 + stop)

        # If high >= stop price, we got stopped out at the stop price
        # Otherwise, we cover at close
        stopped = gaps_df['high'] >= gaps_df[stop_price_col]

        # P&L: stopped → lose stop%, not stopped → (open - close) / open
        pnl = np.where(
            stopped,
            -stop * 100,  # Lost stop%
            gaps_df['short_pnl_pct']  # Normal short P&L
        )
        gaps_df[pnl_col] = np.round(pnl, 2)

        win = (pnl > 0).sum()
        lose = (pnl <= 0).sum()
        total = len(pnl)

        results[stop] = {
            'stop_pct': stop * 100,
            'stopped_out': stopped.sum(),
            'stopped_out_pct': round(stopped.mean() * 100, 1),
            'win_rate': round(win / total * 100, 1) if total > 0 else 0,
            'avg_pnl': round(pnl.mean(), 2),
            'median_pnl': round(np.median(pnl), 2),
            'total_trades': total,
            'avg_win': round(pnl[pnl > 0].mean(), 2) if win > 0 else 0,
            'avg_loss': round(pnl[pnl <= 0].mean(), 2) if lose > 0 else 0,
            'profit_factor': round(abs(pnl[pnl > 0].sum() / pnl[pnl <= 0].sum()), 2) if pnl[pnl <= 0].sum() != 0 else float('inf'),
        }

    return results

def print_report(gaps_df, stop_results):
    """Print the full analysis."""
    N = len(gaps_df)
    if N == 0:
        print("No gap-ups found!")
        return

    print("\n" + "="*70)
    print("  GAP FADE STUDY: Stocks Gapping Up >5% at Open")
    print("="*70)
    print(f"  Universe:     {len(UNIVERSE)} symbols")
    print(f"  Period:       {START_DATE.date()} to {END_DATE.date()} ({LOOKBACK_YEARS} years)")
    print(f"  Gap threshold: >{GAP_THRESHOLD*100:.0f}%")
    print(f"  Total gap-ups found: {N:,}")
    print(f"  Unique symbols:      {gaps_df['symbol'].nunique()}")

    # ── Core question: What % close below open? ─────────────
    faded = gaps_df['faded'].sum()
    fade_rate = faded / N * 100
    filled = gaps_df['filled_gap'].sum()
    fill_rate = filled / N * 100

    print("\n" + "-"*70)
    print("  THE KEY QUESTION: Do gap-ups fade?")
    print("-"*70)
    print(f"  Closed BELOW open (faded):       {faded:>5,} / {N:,}  = {fade_rate:.1f}%")
    print(f"  Closed ABOVE open (held):        {N - faded:>5,} / {N:,}  = {100 - fade_rate:.1f}%")
    print(f"  Closed BELOW prev close (filled): {filled:>5,} / {N:,}  = {fill_rate:.1f}%")

    if fade_rate >= 75:
        print(f"\n  ✓ CONFIRMED: {fade_rate:.1f}% fade rate supports the 79-80% thesis")
    elif fade_rate >= 60:
        print(f"\n  ~ PARTIAL: {fade_rate:.1f}% fade — edge exists but weaker than 79-80%")
    else:
        print(f"\n  ✗ BUSTED: {fade_rate:.1f}% fade — thesis does NOT hold in this data")

    # ── Raw short P&L (no stop) ─────────────────────────────
    avg_short = gaps_df['short_pnl_pct'].mean()
    med_short = gaps_df['short_pnl_pct'].median()
    print("\n" + "-"*70)
    print("  RAW SHORT P&L (short at open, cover at close, no stop)")
    print("-"*70)
    print(f"  Average P&L:  {avg_short:+.2f}%")
    print(f"  Median P&L:   {med_short:+.2f}%")
    print(f"  Best trade:   {gaps_df['short_pnl_pct'].max():+.2f}%")
    print(f"  Worst trade:  {gaps_df['short_pnl_pct'].min():+.2f}%")
    print(f"  Std dev:      {gaps_df['short_pnl_pct'].std():.2f}%")

    # ── Short P&L with stops ────────────────────────────────
    print("\n" + "-"*70)
    print("  SHORT P&L WITH STOP LOSS (short at open, stop above open)")
    print("-"*70)
    print(f"  {'Stop':>6}  {'Win%':>6}  {'AvgPnL':>8}  {'MedPnL':>8}  {'AvgWin':>8}  {'AvgLoss':>8}  {'PF':>6}  {'Stopped%':>9}")
    print(f"  {'----':>6}  {'----':>6}  {'------':>8}  {'------':>8}  {'------':>8}  {'-------':>8}  {'--':>6}  {'--------':>9}")
    for stop, r in stop_results.items():
        print(f"  {r['stop_pct']:5.0f}%  {r['win_rate']:5.1f}%  {r['avg_pnl']:+7.2f}%  {r['median_pnl']:+7.2f}%  {r['avg_win']:+7.2f}%  {r['avg_loss']:+7.2f}%  {r['profit_factor']:5.2f}  {r['stopped_out_pct']:7.1f}%")

    # ── Breakdown by gap size ───────────────────────────────
    print("\n" + "-"*70)
    print("  BREAKDOWN BY GAP SIZE")
    print("-"*70)
    bins = [(5, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 200)]
    print(f"  {'Gap Range':>12}  {'Count':>6}  {'Fade%':>6}  {'Fill%':>6}  {'AvgShortPnL':>12}  {'MedShortPnL':>12}")
    print(f"  {'---------':>12}  {'-----':>6}  {'-----':>6}  {'-----':>6}  {'-----------':>12}  {'-----------':>12}")
    for lo, hi in bins:
        mask = (gaps_df['gap_pct'] >= lo) & (gaps_df['gap_pct'] < hi)
        sub = gaps_df[mask]
        if len(sub) == 0:
            continue
        fr = sub['faded'].mean() * 100
        flr = sub['filled_gap'].mean() * 100
        avg = sub['short_pnl_pct'].mean()
        med = sub['short_pnl_pct'].median()
        print(f"  {lo:>4}-{hi:<4}%   {len(sub):>6,}  {fr:5.1f}%  {flr:5.1f}%  {avg:+11.2f}%  {med:+11.2f}%")

    # ── Breakdown by stock type ─────────────────────────────
    print("\n" + "-"*70)
    print("  TOP 15 SYMBOLS BY GAP-UP COUNT")
    print("-"*70)
    top = gaps_df.groupby('symbol').agg(
        count=('faded', 'size'),
        fade_rate=('faded', 'mean'),
        avg_pnl=('short_pnl_pct', 'mean'),
    ).sort_values('count', ascending=False).head(15)

    print(f"  {'Symbol':>8}  {'Gaps':>5}  {'Fade%':>6}  {'AvgShortPnL':>12}")
    print(f"  {'------':>8}  {'----':>5}  {'-----':>6}  {'-----------':>12}")
    for sym, row in top.iterrows():
        print(f"  {sym:>8}  {int(row['count']):>5}  {row['fade_rate']*100:5.1f}%  {row['avg_pnl']:+11.2f}%")

    # ── Intraday behavior ───────────────────────────────────
    print("\n" + "-"*70)
    print("  INTRADAY BEHAVIOR (avg across all gap-ups)")
    print("-"*70)
    print(f"  Avg move above open (high):  {gaps_df['high_above_open_pct'].mean():+.2f}%")
    print(f"  Avg move below open (low):   {gaps_df['low_below_open_pct'].mean():+.2f}%")
    print(f"  Avg close vs open:           {gaps_df['close_vs_open_pct'].mean():+.2f}%")
    print(f"  Median close vs open:        {gaps_df['close_vs_open_pct'].median():+.2f}%")

    print("\n" + "="*70)
    print("  CONCLUSION")
    print("="*70)
    if fade_rate >= 70 and avg_short > 0:
        print(f"  Gap fade has a statistical edge: {fade_rate:.1f}% fade rate, {avg_short:+.2f}% avg short P&L.")
        print(f"  This could be a viable systematic strategy with proper risk management.")
    elif fade_rate >= 60:
        print(f"  Moderate edge: {fade_rate:.1f}% fade rate, but P&L depends heavily on stops.")
        print(f"  Needs filtering (gap size, sector, volume) to improve.")
    else:
        print(f"  Weak edge: {fade_rate:.1f}% fade rate. The 79-80% claim is overstated in this data.")
    print("="*70)


if __name__ == '__main__':
    raw = fetch_all_data()
    gaps = find_gap_ups(raw)

    if len(gaps) == 0:
        print("No gap-up events found. Check data.")
    else:
        stop_results = analyze_with_stops(gaps)
        print_report(gaps, stop_results)

        # Save raw data for further analysis
        gaps.to_csv('gap_fade_study_results.csv', index=False)
        print(f"\nRaw data saved to gap_fade_study_results.csv ({len(gaps):,} rows)")
