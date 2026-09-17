"""Day-trade momentum SHORT backtest sample generator — emits shadow entries from bars replay.

v-day-trade-short-bt-sample-2026-09-17. Reads historical minute bars from
--bars-dir (parquet/CSV) or Postgres, detects day trade SHORT patterns using
the same logic as DayTradeMomentumShortStrategy, and emits shadow entries
to data/day_trade_short_shadow.ndjson for Stage A evaluation.

This tool generates the INPUT for the resolver (research/day_trade_short_resolver.py).
Use this to create a historical shadow sample when no live shadow data exists.

Entry Pattern Detection (mirrors strategies/builtin.py):
  1. Breakdown: price < 20-bar low, ADX > 20, breakdown dist < 3%
  2. Continuation-down: RSI 30-50, price < SMA20, MACD < signal
  3. Rejection: near 20-bar high but failing, MACD < signal, ADX > 20

Filters:
  - HANDS_OFF symbols excluded (MU, HQGE, SPCX)
  - Weak RS vs SPY configurable (--min-weak-rs, default 0.5%)
  - Risk_on filter configurable (--allow-risk-on for historical replay)
  - Volume ratio minimum (--min-volume-ratio, default 1.5)
  - RSI bounds: floor 30, ceiling 80

Usage (one-liner for Research):
    python -m research.day_trade_short_bt_sample \\
        --bars-dir /path/to/bars/ \\
        --symbols NVDA AAPL TSLA \\
        --out data/day_trade_short_shadow.ndjson \\
        --allow-risk-on

With Postgres bars:
    python -m research.day_trade_short_bt_sample \\
        --dsn postgresql://... \\
        --symbols NVDA AAPL TSLA \\
        --days 90 \\
        --out data/day_trade_short_shadow.ndjson

Outputs:
    NDJSON shadow entries compatible with research/day_trade_short_resolver.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import pandas as pd
import ta

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("day_trade_short_bt_sample")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

HANDS_OFF_SYMBOLS = frozenset({"MU", "HQGE", "SPCX"})
DEDUPE_WINDOW_MIN = 15
DEFAULT_MIN_WEAK_RS = 0.5
DEFAULT_MIN_VOLUME_RATIO = 1.5
DEFAULT_RSI_FLOOR = 30
DEFAULT_RSI_CEILING = 80
DEFAULT_ATR_STOP_MULT = 1.5
DEFAULT_RR_RATIO = 2.0
DEFAULT_DSN = "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev"


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators on OHLCV DataFrame.
    
    Returns DataFrame with indicator columns added. Requires at least 50 rows.
    """
    if len(df) < 50:
        return pd.DataFrame()
    
    df = df.copy()
    
    df["rsi"] = ta.momentum.rsi(df["Close"], window=14)
    df["sma_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["sma_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    
    macd_obj = ta.trend.MACD(df["Close"])
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    
    adx_obj = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=14)
    df["adx"] = adx_obj.adx()
    
    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    
    df["high_20"] = df["High"].rolling(window=20).max()
    df["low_20"] = df["Low"].rolling(window=20).min()
    
    df["volume_ma"] = df["Volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["Volume"] / df["volume_ma"].replace(0, 1)
    
    df["day_change_pct"] = (df["Close"] - df["Open"]) / df["Open"] * 100
    
    return df.dropna()


def detect_entry_pattern(
    close: float,
    low_20: float,
    high_20: float,
    sma_20: float,
    rsi: float,
    adx: float,
    macd: float,
    macd_signal: float,
) -> Optional[tuple[str, float]]:
    """Detect day trade SHORT entry pattern.
    
    Returns (pattern_name, confidence) or None if no pattern matches.
    """
    if low_20 <= 0 or high_20 <= 0:
        return None
    
    if close < low_20 and adx > 20:
        breakdown_dist_pct = ((low_20 - close) / low_20) * 100
        if breakdown_dist_pct < 3.0:
            confidence = 0.70 + (adx / 100) * 0.2
            return ("breakdown", confidence)
    
    if (30 <= rsi <= 50 and
        sma_20 > 0 and
        close < sma_20 and
        macd < macd_signal):
        confidence = 0.65
        return ("continuation_down", confidence)
    
    if (high_20 > 0 and
        abs((close - high_20) / high_20) < 0.02 and
        macd < macd_signal and
        adx > 20):
        confidence = 0.60 + (adx / 100) * 0.15
        return ("rejection", confidence)
    
    return None


def make_file_bars_loader(bars_dir: Path) -> Callable[[str], pd.DataFrame]:
    """Return a bars loader from local files."""
    def _load(symbol: str) -> pd.DataFrame:
        parquet_path = bars_dir / f"{symbol.upper()}.parquet"
        csv_path = bars_dir / f"{symbol.upper()}.csv"
        
        df = pd.DataFrame()
        try:
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
            elif csv_path.exists():
                df = pd.read_csv(csv_path)
            else:
                logger.debug("No bars file for %s in %s", symbol, bars_dir)
                return df
                
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"])
                df = df.set_index("ts")
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            
            col_map = {"open": "Open", "high": "High", "low": "Low", 
                       "close": "Close", "volume": "Volume"}
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            
            return df
        except Exception as exc:
            logger.warning("Failed to load bars for %s: %s", symbol, exc)
            return pd.DataFrame()
    
    return _load


def make_postgres_bars_loader(dsn: str, days: int = 90) -> Callable[[str], pd.DataFrame]:
    """Return a bars loader from Postgres."""
    from data_providers.postgres import PostgresDataProvider
    provider = PostgresDataProvider(dsn)
    
    def _load(symbol: str) -> pd.DataFrame:
        try:
            df = provider.get_market_data(
                symbol,
                period_type="day" if days < 30 else "month",
                period=days if days < 30 else max(1, days // 30),
                frequency_type="minute",
                frequency=5,
            )
            return df
        except Exception as exc:
            logger.warning("Postgres bars load failed for %s: %s", symbol, exc)
            return pd.DataFrame()
    
    return _load


def estimate_spy_change(bar_ts: datetime, spy_df: Optional[pd.DataFrame]) -> float:
    """Estimate SPY change % at the given bar timestamp.
    
    If SPY data available, returns intraday change %. Otherwise returns 0.
    """
    if spy_df is None or spy_df.empty:
        return 0.0
    
    try:
        bar_date = bar_ts.date() if hasattr(bar_ts, 'date') else bar_ts
        if hasattr(spy_df.index, 'date'):
            day_mask = spy_df.index.date == bar_date
            day_data = spy_df[day_mask]
        else:
            day_data = spy_df
        
        if day_data.empty:
            return 0.0
        
        prior_bars = day_data[day_data.index <= bar_ts]
        if prior_bars.empty:
            return 0.0
        
        day_open = day_data["Open"].iloc[0]
        current_close = prior_bars["Close"].iloc[-1]
        
        if day_open > 0:
            return ((current_close - day_open) / day_open) * 100
        return 0.0
    except Exception:
        return 0.0


def classify_regime(spy_change_pct: float) -> str:
    """Classify market regime based on SPY change.
    
    Simple heuristic: SPY > +0.5% = risk_on, SPY < -0.5% = risk_off, else mixed.
    For BT without live data, this is approximate.
    """
    if spy_change_pct > 0.5:
        return "risk_on"
    elif spy_change_pct < -0.5:
        return "risk_off"
    return "mixed"


def process_symbol(
    symbol: str,
    bars_loader: Callable[[str], pd.DataFrame],
    spy_df: Optional[pd.DataFrame],
    min_weak_rs: float,
    min_volume_ratio: float,
    rsi_floor: float,
    rsi_ceiling: float,
    allow_risk_on: bool,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> list[dict]:
    """Process a single symbol and return list of shadow entry dicts."""
    if symbol.upper() in HANDS_OFF_SYMBOLS:
        logger.info("Skipping %s (hands-off)", symbol)
        return []
    
    df = bars_loader(symbol)
    if df.empty:
        logger.info("No bars for %s", symbol)
        return []
    
    if start_date is not None:
        df = df[df.index.date >= start_date]
    if end_date is not None:
        df = df[df.index.date <= end_date]
    
    if len(df) < 50:
        logger.info("Insufficient bars for %s (%d)", symbol, len(df))
        return []
    
    df = compute_indicators(df)
    if df.empty:
        logger.info("Indicator computation failed for %s", symbol)
        return []
    
    entries = []
    last_entry_ts: Optional[datetime] = None
    
    for i in range(len(df)):
        bar = df.iloc[i]
        bar_ts = df.index[i]
        
        if last_entry_ts is not None:
            delta_min = (bar_ts - last_entry_ts).total_seconds() / 60
            if delta_min < DEDUPE_WINDOW_MIN:
                continue
        
        close = float(bar["Close"])
        rsi = float(bar["rsi"])
        adx = float(bar["adx"])
        volume_ratio = float(bar["volume_ratio"])
        low_20 = float(bar["low_20"])
        high_20 = float(bar["high_20"])
        sma_20 = float(bar["sma_20"])
        sma_50 = float(bar["sma_50"]) if bar["sma_50"] > 0 else 0.0
        macd = float(bar["macd"])
        macd_signal = float(bar["macd_signal"])
        atr = float(bar["atr"])
        day_change_pct = float(bar["day_change_pct"])
        
        if rsi <= rsi_floor or rsi >= rsi_ceiling:
            continue
        
        if volume_ratio < min_volume_ratio:
            continue
        
        spy_change = estimate_spy_change(bar_ts, spy_df)
        regime = classify_regime(spy_change)
        
        if regime == "risk_on" and not allow_risk_on:
            continue
        
        rs_vs_spy = day_change_pct - spy_change
        if rs_vs_spy > -min_weak_rs:
            continue
        
        pattern_result = detect_entry_pattern(
            close, low_20, high_20, sma_20, rsi, adx, macd, macd_signal
        )
        if pattern_result is None:
            continue
        
        entry_pattern, confidence = pattern_result
        
        atr_floored = max(atr, close * 0.005)
        stop_distance = DEFAULT_ATR_STOP_MULT * atr_floored
        stop_loss = close + stop_distance
        take_profit = close - (DEFAULT_RR_RATIO * stop_distance)
        
        if stop_distance > 0:
            rr_ratio = (close - take_profit) / stop_distance
        else:
            rr_ratio = DEFAULT_RR_RATIO
        
        session_id = bar_ts.strftime("%Y-%m-%d") if hasattr(bar_ts, 'strftime') else str(bar_ts.date())
        
        shadow_entry = {
            "timestamp": bar_ts.isoformat() if hasattr(bar_ts, 'isoformat') else str(bar_ts),
            "symbol": symbol.upper(),
            "signal_type": "SHORT",
            "strategy": "day_trade_momentum_short",
            "entry_pattern": entry_pattern,
            "signal_close": round(close, 4),
            "rsi": round(rsi, 2),
            "rs_vs_spy": round(rs_vs_spy, 2),
            "volume_ratio": round(volume_ratio, 2),
            "adx": round(adx, 2),
            "low_20": round(low_20, 4),
            "high_20": round(high_20, 4),
            "sma_20": round(sma_20, 4) if sma_20 > 0 else 0,
            "sma_50": round(sma_50, 4) if sma_50 > 0 else 0,
            "macd": round(macd, 6),
            "macd_signal": round(macd_signal, 6),
            "atr": round(atr, 4),
            "hypothetical_stop": round(stop_loss, 4),
            "hypothetical_target": round(take_profit, 4),
            "rr_ratio": round(rr_ratio, 2),
            "stop_dist": round(stop_distance, 4),
            "market_context_regime": regime,
            "market_context_spy_change": round(spy_change, 2),
            "market_context_vix_change": 0.0,
            "market_context_sector": None,
            "direction_score": 0.0,
            "direction_phase": "unknown",
            "session_id": session_id,
        }
        
        entries.append(shadow_entry)
        last_entry_ts = bar_ts
    
    logger.info("Found %d shadow entries for %s", len(entries), symbol)
    return entries


def write_shadow_log(entries: list[dict], path: Path, append: bool = False) -> None:
    """Write shadow entries to NDJSON file."""
    mode = "a" if append else "w"
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, mode) as f:
        for entry in entries:
            f.write(json.dumps(entry, default=str) + "\n")
    
    logger.info("Wrote %d entries to %s", len(entries), path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bars-dir",
        type=Path,
        default=None,
        help="Directory with {SYMBOL}.parquet or {SYMBOL}.csv files",
    )
    parser.add_argument(
        "--dsn",
        default=None,
        help=f"Postgres DSN. Default: POSTGRES_DSN env or {DEFAULT_DSN}",
    )
    parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        default=None,
        help="Symbols to process (default: all in bars-dir)",
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=90,
        help="Days of history when using Postgres (default: 90)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date filter (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--out", "-o",
        type=Path,
        default=REPO_ROOT / "data" / "day_trade_short_shadow.ndjson",
        help="Output NDJSON path (default: data/day_trade_short_shadow.ndjson)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing file instead of overwriting",
    )
    parser.add_argument(
        "--min-weak-rs",
        type=float,
        default=DEFAULT_MIN_WEAK_RS,
        help=f"Minimum weak RS vs SPY %% (default: {DEFAULT_MIN_WEAK_RS})",
    )
    parser.add_argument(
        "--min-volume-ratio",
        type=float,
        default=DEFAULT_MIN_VOLUME_RATIO,
        help=f"Minimum volume ratio (default: {DEFAULT_MIN_VOLUME_RATIO})",
    )
    parser.add_argument(
        "--rsi-floor",
        type=float,
        default=DEFAULT_RSI_FLOOR,
        help=f"RSI floor - no short below (default: {DEFAULT_RSI_FLOOR})",
    )
    parser.add_argument(
        "--rsi-ceiling",
        type=float,
        default=DEFAULT_RSI_CEILING,
        help=f"RSI ceiling - no short above (default: {DEFAULT_RSI_CEILING})",
    )
    parser.add_argument(
        "--allow-risk-on",
        action="store_true",
        help="Allow entries in risk_on regime (live blocks risk_on, BT may want to score)",
    )
    parser.add_argument(
        "--spy-bars-file",
        type=Path,
        default=None,
        help="Path to SPY bars file for RS computation (optional)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be emitted without writing",
    )
    args = parser.parse_args(argv)
    
    if args.bars_dir is not None and args.bars_dir.exists():
        logger.info("Using file-based bars loader from: %s", args.bars_dir)
        bars_loader = make_file_bars_loader(args.bars_dir)
        
        if args.symbols is None:
            from data_providers.file_bars import FileBarsProvider
            provider = FileBarsProvider(args.bars_dir)
            symbols = provider.list_symbols()
        else:
            symbols = [s.upper() for s in args.symbols]
    else:
        dsn = args.dsn or os.environ.get("POSTGRES_DSN", DEFAULT_DSN)
        logger.info("Using Postgres bars loader: %s", dsn[:30] + "...")
        bars_loader = make_postgres_bars_loader(dsn, args.days)
        
        if args.symbols is None:
            logger.error("--symbols required when using Postgres")
            return 1
        symbols = [s.upper() for s in args.symbols]
    
    symbols = [s for s in symbols if s not in HANDS_OFF_SYMBOLS]
    
    if not symbols:
        logger.error("No symbols to process")
        return 1
    
    logger.info("Processing %d symbols", len(symbols))
    
    spy_df = None
    if args.spy_bars_file and args.spy_bars_file.exists():
        try:
            spy_loader = make_file_bars_loader(args.spy_bars_file.parent)
            spy_df = spy_loader("SPY")
            spy_df = compute_indicators(spy_df) if not spy_df.empty else None
            if spy_df is not None and not spy_df.empty:
                logger.info("Loaded SPY bars for RS computation")
        except Exception as exc:
            logger.warning("Failed to load SPY bars: %s", exc)
    
    if "SPY" in symbols:
        try:
            spy_df = bars_loader("SPY")
            if not spy_df.empty:
                logger.info("Loaded SPY bars for RS computation")
        except Exception:
            pass
    
    start_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    
    end_date = None
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    
    all_entries = []
    for symbol in symbols:
        if symbol == "SPY":
            continue
        
        entries = process_symbol(
            symbol=symbol,
            bars_loader=bars_loader,
            spy_df=spy_df,
            min_weak_rs=args.min_weak_rs,
            min_volume_ratio=args.min_volume_ratio,
            rsi_floor=args.rsi_floor,
            rsi_ceiling=args.rsi_ceiling,
            allow_risk_on=args.allow_risk_on,
            start_date=start_date,
            end_date=end_date,
        )
        all_entries.extend(entries)
    
    all_entries.sort(key=lambda e: e["timestamp"])
    
    logger.info("Total shadow entries: %d", len(all_entries))
    
    by_pattern = defaultdict(int)
    by_regime = defaultdict(int)
    for e in all_entries:
        by_pattern[e["entry_pattern"]] += 1
        by_regime[e["market_context_regime"]] += 1
    
    print(f"\n{'='*60}")
    print("DAY TRADE SHORT BT SAMPLE — SUMMARY")
    print(f"{'='*60}")
    print(f"  Total entries:     {len(all_entries)}")
    print(f"  Symbols processed: {len(symbols)}")
    print(f"  By pattern:        {dict(by_pattern)}")
    print(f"  By regime:         {dict(by_regime)}")
    print(f"{'='*60}")
    
    if args.dry_run:
        print(f"\nDry run — would write to: {args.out}")
        if all_entries:
            print("\nSample entries (first 5):")
            for e in all_entries[:5]:
                print(f"  {e['timestamp']} {e['symbol']} @ ${e['signal_close']:.2f} [{e['entry_pattern']}]")
        return 0
    
    if not all_entries:
        logger.warning("No entries to write")
        return 0
    
    write_shadow_log(all_entries, args.out, append=args.append)
    
    print(f"\nOutput: {args.out}")
    print(f"Next: python -m research.day_trade_short_resolver --log {args.out} --out /tmp/results/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
