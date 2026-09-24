#!/usr/bin/env python3
"""Historical mean-rev LONG emitter — scan minute bars for oversold setups.

v-mean-rev-bt-sample-2026-09-17. Backtest/historical path that emits mean_rev
LONG shadow entries into `data/shadow_long_log.ndjson` from local minute bars.
Enables Stage A validation without requiring live soak or Postgres bars lag.

USAGE (one-liner for Research):
    python -m research.mean_rev_long_bt_sample --bars-dir /path/to/bars

Detection mirrors OversoldBounceV2 gates (technical only):
    Gate A — Statistical extreme (RSI < 30 AND close < BB_lower)
    Gate B — Capitulation volume (volume_ratio > 2.0)
    Gate C — Bullish bar structure (hammer/doji or reclaimed prev_low)
    Gate D — Support level nearby (swing low within 0.5 ATR)

Gates E (market regime) and F (news) are omitted — they require live context.
This is intentional: the shadow resolver handles regime exclusions downstream.

HANDS_OFF exclusions: MU, HQGE, SPCX — permanently skipped.

Outputs:
    data/shadow_long_log.ndjson — NDJSON ledger compatible with shadow_long_resolver.py

Schema (per line):
    {
        "timestamp": "2024-01-15T10:30:00+00:00",
        "symbol": "AAPL",
        "signal_type": "LONG",
        "reason": "oversold_bounce",
        "signal_close": 150.0,
        "entry_price": 150.0,
        "rsi": 24.5,
        "rsi_14": 24.5,
        "bb_lower": 148.0,
        "bb_middle": 152.0,
        "sma_50": 145.0,
        "macd": -0.5,
        "macd_signal": -0.3,
        "atr": 3.0,
        "hypothetical_stop": 142.5,
        "hypothetical_target": 165.0,
        "rr_ratio": 2.0,
        "stop_dist": 7.5,
        "market_context_regime": null,
        "falling_knife_pass": true,
        "_source": "mean_rev_long_bt_sample"
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ta

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("mean_rev_long_bt_sample")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

HANDS_OFF_SYMBOLS = frozenset({"MU", "HQGE", "SPCX"})
DEDUPE_WINDOW_MIN = 15
DEFAULT_RR_RATIO = 2.0


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators required for mean-rev detection.
    
    Returns a DataFrame indexed like df with columns:
        rsi, bb_lower, bb_middle, bb_upper, sma_50, macd, macd_signal,
        atr, volume_ratio, prev_low, lows_50 (embedded per bar downstream)
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]
    
    ind = pd.DataFrame(index=df.index)
    
    ind["rsi"] = ta.momentum.rsi(close, window=14)
    
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    ind["bb_lower"] = bb.bollinger_lband()
    ind["bb_middle"] = bb.bollinger_mavg()
    ind["bb_upper"] = bb.bollinger_hband()
    
    ind["sma_50"] = close.rolling(50).mean()
    
    macd_ind = ta.trend.MACD(close)
    ind["macd"] = macd_ind.macd()
    ind["macd_signal"] = macd_ind.macd_signal()
    
    ind["atr"] = ta.volatility.average_true_range(high, low, close, window=14)
    
    ind["volume_ratio"] = volume / (volume.rolling(20).mean() + 1e-10)
    
    ind["prev_low"] = low.shift(1)
    
    return ind


def bar_has_capitulation_structure(
    open_: float,
    high: float,
    low: float,
    close: float,
    prev_low: Optional[float],
) -> bool:
    """Check if bar shows bullish reversal structure (Gate C).
    
    Two acceptance shapes:
        1. Long lower wick (lower_wick > 1.5 * body) — classic hammer
        2. close >= prev_low — reclaimed previous support
    """
    try:
        o = float(open_)
        h = float(high)
        lo = float(low)
        c = float(close)
    except (TypeError, ValueError):
        return False

    lower_wick = min(o, c) - lo
    body = abs(c - o)

    if body > 0 and lower_wick > 1.5 * body:
        return True
    if body == 0 and lower_wick > 0 and (h - lo) > 0 and lower_wick / (h - lo) > 0.6:
        return True

    if prev_low is not None:
        try:
            if c >= float(prev_low):
                return True
        except (TypeError, ValueError):
            pass

    return False


def find_nearest_swing_low(
    lows: list[float],
    current_price: float,
    atr: float,
) -> Optional[float]:
    """Find most recent swing low within 0.5 * ATR (Gate D)."""
    if atr <= 0 or lows is None:
        return None

    arr = np.asarray(lows, dtype=float)
    if arr.size < 3:
        return None

    threshold = 0.5 * float(atr)
    for i in range(arr.size - 2, 0, -1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            if abs(float(current_price) - float(arr[i])) <= threshold:
                return float(arr[i])
    return None


def compute_stop_target(
    close: float,
    atr: float,
    bar_range: float,
    rr_ratio: float = DEFAULT_RR_RATIO,
) -> tuple[float, float, float]:
    """Compute (stop_dist, stop_loss, take_profit) for a LONG entry.
    
    stop_distance = max(2 * ATR, 0.5 * bar_range)
    stop_loss     = close - stop_distance
    take_profit   = close + (rr_ratio + 1) * stop_distance
    """
    stop_dist = max(2.0 * float(atr), 0.5 * bar_range)
    stop_loss = close - stop_dist
    take_profit = close + (rr_ratio + 1) * stop_dist
    return stop_dist, stop_loss, take_profit


def scan_symbol_for_entries(
    symbol: str,
    df: pd.DataFrame,
    ind: pd.DataFrame,
) -> list[dict]:
    """Scan a symbol's bars for mean-rev LONG setups.
    
    Returns list of shadow entry dicts ready for NDJSON output.
    """
    entries = []
    _low_arr = df["Low"].to_numpy()
    
    for i in range(50, len(df)):
        bar = df.iloc[i]
        indicators = ind.iloc[i]
        
        if pd.isna(indicators["rsi"]) or pd.isna(indicators["bb_lower"]):
            continue
        if pd.isna(indicators["atr"]) or indicators["atr"] <= 0:
            continue
            
        close = float(bar["Close"])
        open_ = float(bar["Open"])
        high = float(bar["High"])
        low = float(bar["Low"])
        rsi = float(indicators["rsi"])
        bb_lower = float(indicators["bb_lower"])
        bb_middle = float(indicators["bb_middle"])
        volume_ratio = float(indicators["volume_ratio"])
        atr = float(indicators["atr"])
        prev_low = indicators["prev_low"]
        if pd.notna(prev_low):
            prev_low = float(prev_low)
        else:
            prev_low = None
        
        if not (rsi < 30 and close < bb_lower):
            continue
        
        if not (volume_ratio > 2.0):
            continue
        
        if not bar_has_capitulation_structure(open_, high, low, close, prev_low):
            continue
        
        lows_50 = _low_arr[max(0, i - 50):i].tolist()
        nearest_swing = find_nearest_swing_low(lows_50, close, atr)
        if nearest_swing is None:
            continue
        
        bar_range = high - low
        stop_dist, stop_loss, take_profit = compute_stop_target(close, atr, bar_range)
        
        ts = df.index[i]
        if hasattr(ts, "isoformat"):
            ts_str = ts.isoformat()
        else:
            ts_str = str(ts)
        
        sma_50 = float(indicators["sma_50"]) if pd.notna(indicators["sma_50"]) else 0.0
        macd = float(indicators["macd"]) if pd.notna(indicators["macd"]) else 0.0
        macd_signal = float(indicators["macd_signal"]) if pd.notna(indicators["macd_signal"]) else 0.0
        
        entry = {
            "timestamp": ts_str,
            "symbol": symbol.upper(),
            "signal_type": "LONG",
            "reason": "oversold_bounce",
            "signal_close": round(close, 4),
            "entry_price": round(close, 4),
            "rsi": round(rsi, 2),
            "rsi_14": round(rsi, 2),
            "bb_lower": round(bb_lower, 4),
            "bb_middle": round(bb_middle, 4),
            "sma_50": round(sma_50, 4),
            "macd": round(macd, 4),
            "macd_signal": round(macd_signal, 4),
            "atr": round(atr, 4),
            "hypothetical_stop": round(stop_loss, 4),
            "hypothetical_target": round(take_profit, 4),
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "rr_ratio": DEFAULT_RR_RATIO,
            "stop_dist": round(stop_dist, 4),
            "market_context_regime": None,
            "falling_knife_pass": True,
            "volume_ratio": round(volume_ratio, 2),
            "nearest_swing_low": round(nearest_swing, 4),
            "_source": "mean_rev_long_bt_sample",
        }
        entries.append(entry)
    
    return entries


def load_bars_from_dir(bars_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all bar files from directory.
    
    Returns dict mapping symbol -> DataFrame.
    """
    bars = {}
    
    if not bars_dir.exists():
        logger.error("Bars directory does not exist: %s", bars_dir)
        return bars
    
    for f in bars_dir.iterdir():
        if f.suffix == ".parquet":
            symbol = f.stem.upper()
            try:
                df = pd.read_parquet(f)
                bars[symbol] = df
            except Exception as exc:
                logger.warning("Failed to load %s: %s", f, exc)
        elif f.suffix == ".csv":
            symbol = f.stem.upper()
            if symbol not in bars:
                try:
                    df = pd.read_csv(f)
                    bars[symbol] = df
                except Exception as exc:
                    logger.warning("Failed to load %s: %s", f, exc)
    
    logger.info("Loaded %d symbols from %s", len(bars), bars_dir)
    return bars


def normalize_bars_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize bar DataFrame to expected format.
    
    Expected output: DatetimeIndex, columns [Open, High, Low, Close, Volume].
    """
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    column_map = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume"
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
    
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df


def dedupe_entries(entries: list[dict]) -> list[dict]:
    """Dedupe entries: same symbol within 15 minutes → keep first."""
    entries.sort(key=lambda e: e["timestamp"])
    
    last_seen: dict[str, str] = {}
    deduped: list[dict] = []
    
    for entry in entries:
        symbol = entry["symbol"]
        ts_str = entry["timestamp"]
        
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except ValueError:
            ts = datetime.now(timezone.utc)
        
        prev_ts_str = last_seen.get(symbol)
        if prev_ts_str is not None:
            try:
                prev_ts = datetime.fromisoformat(prev_ts_str.replace("Z", "+00:00"))
                delta_min = (ts - prev_ts).total_seconds() / 60.0
                if delta_min < DEDUPE_WINDOW_MIN:
                    continue
            except ValueError:
                pass
        
        last_seen[symbol] = ts_str
        deduped.append(entry)
    
    return deduped


def write_ndjson(entries: list[dict], output_path: Path, append: bool = False) -> int:
    """Write entries to NDJSON file.
    
    Returns number of entries written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    mode = "a" if append else "w"
    with open(output_path, mode) as f:
        for entry in entries:
            f.write(json.dumps(entry, default=str) + "\n")
    
    return len(entries)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bars-dir", "-b",
        type=Path,
        required=True,
        help="Directory with {SYMBOL}.parquet or {SYMBOL}.csv bar files",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=REPO_ROOT / "data" / "shadow_long_log.ndjson",
        help="Output path (default: data/shadow_long_log.ndjson)",
    )
    parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        default=None,
        help="Specific symbols to scan (default: all available)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing output file instead of overwriting",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and count entries without writing",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args(argv)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    bars_data = load_bars_from_dir(args.bars_dir)
    if not bars_data:
        logger.error("No bar data found in %s", args.bars_dir)
        return 1
    
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
        symbols = [s for s in symbols if s in bars_data]
    else:
        symbols = list(bars_data.keys())
    
    symbols = [s for s in symbols if s not in HANDS_OFF_SYMBOLS]
    excluded = [s for s in (args.symbols or bars_data.keys()) 
                if s.upper() in HANDS_OFF_SYMBOLS]
    if excluded:
        logger.info("Excluding HANDS_OFF symbols: %s", excluded)
    
    logger.info("Scanning %d symbols for mean-rev LONG setups...", len(symbols))
    
    all_entries: list[dict] = []
    for i, symbol in enumerate(symbols, 1):
        try:
            df = normalize_bars_df(bars_data[symbol].copy())
        except Exception as exc:
            logger.warning("Failed to normalize bars for %s: %s", symbol, exc)
            continue
        
        if len(df) < 100:
            logger.debug("Skipping %s: insufficient bars (%d)", symbol, len(df))
            continue
        
        ind = compute_indicators(df)
        entries = scan_symbol_for_entries(symbol, df, ind)
        
        if entries:
            logger.debug("%s: found %d entries", symbol, len(entries))
            all_entries.extend(entries)
        
        if i % 10 == 0 or i == len(symbols):
            logger.info("Progress: %d/%d symbols, %d entries so far",
                       i, len(symbols), len(all_entries))
    
    all_entries = dedupe_entries(all_entries)
    logger.info("After deduplication: %d entries", len(all_entries))
    
    if not all_entries:
        logger.warning("No mean-rev LONG entries found in bar data")
        return 0
    
    if args.dry_run:
        print(f"\n[DRY RUN] Would write {len(all_entries)} entries to {args.output}")
        print(f"\nSample entries:")
        for entry in all_entries[:5]:
            print(f"  {entry['timestamp']} {entry['symbol']} "
                  f"@ {entry['signal_close']:.2f} RSI={entry['rsi']:.1f}")
        if len(all_entries) > 5:
            print(f"  ... and {len(all_entries) - 5} more")
        
        symbols_seen = set(e["symbol"] for e in all_entries)
        print(f"\nSymbols with entries ({len(symbols_seen)}): {sorted(symbols_seen)}")
        
        if all_entries:
            ts_min = min(e["timestamp"] for e in all_entries)
            ts_max = max(e["timestamp"] for e in all_entries)
            print(f"Date range: {ts_min[:10]} to {ts_max[:10]}")
        return 0
    
    written = write_ndjson(all_entries, args.output, append=args.append)
    logger.info("Wrote %d entries to %s", written, args.output)
    
    print(f"\n{'=' * 60}")
    print(f"MEAN_REV LONG BT SAMPLE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Symbols scanned:    {len(symbols)}")
    print(f"  Entries emitted:    {len(all_entries)}")
    print(f"  Output:             {args.output}")
    
    symbols_with_entries = set(e["symbol"] for e in all_entries)
    print(f"  Symbols with hits:  {len(symbols_with_entries)}")
    
    if all_entries:
        ts_min = min(e["timestamp"] for e in all_entries)
        ts_max = max(e["timestamp"] for e in all_entries)
        print(f"  Date range:         {ts_min[:10]} to {ts_max[:10]}")
    
    print(f"\nNext step: resolve trades against bars:")
    print(f"  python -m research.shadow_long_resolver \\")
    print(f"      --log {args.output} \\")
    print(f"      --bars-dir {args.bars_dir} \\")
    print(f"      --out /tmp/shadow_results/")
    print(f"{'=' * 60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
