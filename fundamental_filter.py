#!/usr/bin/env python3
"""
FundamentalFilter — Pluggable fundamental quality gate for swing trading candidates.

Called by the weekly scanner after technical Stage 2 filtering.
Adds revenue growth, debt safety, institutional ownership,
earnings proximity, and insider activity checks.

Data source: yfinance (free, no API key needed).
Fallback: graceful degradation — if yfinance fails for a symbol,
it PASSES (don't block on data failure).

Usage:
    python fundamental_filter.py AAPL NVDA TSLA MSFT AMD    # Check specific symbols
    python fundamental_filter.py --scan                       # Check current Stage 2 watchlist
    python fundamental_filter.py --clear-cache                # Clear cached data
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg2
import psycopg2.extras

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fundamental_filter")

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev",
)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "min_revenue_growth": 0.20,
    "max_debt_equity": 1.5,
    "min_institutional_pct": 0.30,
    "earnings_blackout_before": 5,
    "earnings_blackout_after": 2,
    "max_insider_sell_pct": 0.01,
    "cache_ttl_hours": 168,
    "use_llm_overlay": False,
    "llm_model": "gpt-oss:20b",
    "llm_url": "http://localhost:11434",
    "yfinance_timeout": 10,
    "max_workers": 5,
    "rate_limit_delay": 1.0,
}


def _make_empty_result(symbol: str) -> Dict[str, Any]:
    """Return a result dict with all fields defaulted to passing / unavailable."""
    return {
        "symbol": symbol,
        "passes": True,
        "revenue_growth": None,
        "revenue_pass": True,
        "debt_equity": None,
        "debt_pass": True,
        "institutional_pct": None,
        "institutional_pass": True,
        "earnings_near": False,
        "earnings_date": None,
        "earnings_pass": True,
        "insider_net_sell_pct": None,
        "insider_pass": True,
        "reasons": [],
        "data_quality": "unavailable",
    }


def _trading_days_between(d1: date, d2: date) -> int:
    """Approximate trading days between two dates (excludes weekends)."""
    if d1 > d2:
        d1, d2 = d2, d1
    total = 0
    current = d1
    while current < d2:
        current += timedelta(days=1)
        if current.weekday() < 5:
            total += 1
    return total


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_db_connection():
    """Create a new database connection."""
    try:
        return psycopg2.connect(DB_URL)
    except Exception as exc:
        logger.warning("Cannot connect to PostgreSQL: %s", exc)
        return None


def _ensure_cache_table(conn) -> None:
    """Create the fundamental_cache table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fundamental_cache (
                symbol TEXT PRIMARY KEY,
                data_json TEXT NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
    conn.commit()


def _load_cached(conn, symbol: str, ttl_hours: int) -> Optional[Dict[str, Any]]:
    """Load a cached result if it exists and is within TTL."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT data_json, updated_at
                FROM fundamental_cache
                WHERE symbol = %s
                  AND updated_at > NOW() - INTERVAL '%s hours'
                """,
                (symbol, ttl_hours),
            )
            row = cur.fetchone()
            if row:
                return json.loads(row[0])
    except Exception as exc:
        logger.debug("Cache read failed for %s: %s", symbol, exc)
    return None


def _save_cached(conn, symbol: str, data: Dict[str, Any]) -> None:
    """Upsert a result into the cache table."""
    try:
        data_json = json.dumps(data, default=str)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO fundamental_cache (symbol, data_json, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (symbol)
                DO UPDATE SET data_json = EXCLUDED.data_json,
                              updated_at = NOW()
                """,
                (symbol, data_json),
            )
        conn.commit()
    except Exception as exc:
        logger.debug("Cache write failed for %s: %s", symbol, exc)


def _query_earnings_from_db(conn, symbol: str) -> Optional[date]:
    """Check the earnings_calendar table for next earnings date."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT earnings_date FROM earnings_calendar
                WHERE symbol = %s AND earnings_date >= CURRENT_DATE
                ORDER BY earnings_date ASC LIMIT 1
                """,
                (symbol,),
            )
            row = cur.fetchone()
            if row:
                val = row[0]
                if isinstance(val, date):
                    return val
                return datetime.strptime(str(val), "%Y-%m-%d").date()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# yfinance data fetchers (each handles its own errors)
# ---------------------------------------------------------------------------

def _fetch_revenue_growth(ticker) -> Optional[float]:
    """Return YoY revenue growth as a decimal (0.35 = 35%)."""
    try:
        # Try quarterly first for TTM
        qf = getattr(ticker, "quarterly_financials", None)
        if qf is not None and not qf.empty:
            rev_row = None
            for label in ["Total Revenue", "Revenue"]:
                if label in qf.index:
                    rev_row = qf.loc[label]
                    break
            if rev_row is not None and len(rev_row) >= 8:
                # TTM = sum of last 4 quarters; prior TTM = sum of quarters 5-8
                recent_ttm = rev_row.iloc[:4].sum()
                prior_ttm = rev_row.iloc[4:8].sum()
                if prior_ttm > 0:
                    return float((recent_ttm - prior_ttm) / prior_ttm)

        # Fall back to annual
        af = getattr(ticker, "financials", None)
        if af is not None and not af.empty:
            rev_row = None
            for label in ["Total Revenue", "Revenue"]:
                if label in af.index:
                    rev_row = af.loc[label]
                    break
            if rev_row is not None and len(rev_row) >= 2:
                recent = rev_row.iloc[0]
                prior = rev_row.iloc[1]
                if prior > 0:
                    return float((recent - prior) / prior)
    except Exception as exc:
        logger.debug("Revenue growth fetch failed: %s", exc)
    return None


def _fetch_debt_equity(ticker) -> Optional[float]:
    """Return debt-to-equity ratio."""
    try:
        bs = getattr(ticker, "balance_sheet", None)
        if bs is None or bs.empty:
            return None

        # Find total debt
        total_debt = None
        for label in ["Total Debt", "Long Term Debt", "Total Non Current Liabilities Net Minority Interest"]:
            if label in bs.index:
                val = bs.loc[label].iloc[0]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    total_debt = float(val)
                    break

        # Find stockholders equity
        equity = None
        for label in [
            "Stockholders Equity",
            "Total Stockholders Equity",
            "Stockholders' Equity",
            "Common Stock Equity",
        ]:
            if label in bs.index:
                val = bs.loc[label].iloc[0]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    equity = float(val)
                    break

        if total_debt is None:
            # No debt found — could be zero debt
            return 0.0
        if equity is None or equity <= 0:
            return None

        return total_debt / equity
    except Exception as exc:
        logger.debug("Debt/equity fetch failed: %s", exc)
    return None


def _fetch_institutional_pct(ticker) -> Optional[float]:
    """Return institutional ownership as a decimal (0.60 = 60%)."""
    try:
        info = ticker.info or {}
        # yfinance .info may contain institutionHoldPercent directly
        inst_pct = info.get("heldPercentInstitutions")
        if inst_pct is not None:
            return float(inst_pct)

        # Fallback: major_holders table
        mh = getattr(ticker, "major_holders", None)
        if mh is not None and not mh.empty:
            for _, row_data in mh.iterrows():
                val_str = str(row_data.iloc[0])
                label_str = str(row_data.iloc[1]) if len(row_data) > 1 else ""
                if "institution" in label_str.lower() and "%" in val_str:
                    pct = float(val_str.replace("%", "").strip()) / 100.0
                    return pct
    except Exception as exc:
        logger.debug("Institutional ownership fetch failed: %s", exc)
    return None


def _fetch_earnings_date(ticker) -> Optional[date]:
    """Return the next earnings date from yfinance."""
    try:
        cal = getattr(ticker, "calendar", None)
        if cal is None:
            return None

        # yfinance .calendar can be a dict or DataFrame
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if ed:
                if isinstance(ed, list) and len(ed) > 0:
                    val = ed[0]
                else:
                    val = ed
                if hasattr(val, "date"):
                    return val.date()
                return datetime.strptime(str(val)[:10], "%Y-%m-%d").date()
        else:
            # DataFrame
            if hasattr(cal, "loc"):
                for label in ["Earnings Date"]:
                    if label in cal.index:
                        val = cal.loc[label].iloc[0]
                        if hasattr(val, "date"):
                            return val.date()
                        return datetime.strptime(str(val)[:10], "%Y-%m-%d").date()
    except Exception as exc:
        logger.debug("Earnings date fetch failed: %s", exc)
    return None


def _fetch_insider_selling(ticker) -> Optional[float]:
    """Return net insider selling as fraction of shares outstanding.

    Positive = net selling, negative = net buying.
    """
    try:
        txns = getattr(ticker, "insider_transactions", None)
        if txns is None or txns.empty:
            return None

        info = ticker.info or {}
        shares_outstanding = info.get("sharesOutstanding")
        if not shares_outstanding or shares_outstanding <= 0:
            return None

        cutoff = datetime.now() - timedelta(days=180)
        net_sold = 0.0

        for _, row in txns.iterrows():
            txn_date = row.get("Start Date") or row.get("startDate") or row.get("Date")
            if txn_date is None:
                continue
            if hasattr(txn_date, "to_pydatetime"):
                txn_date = txn_date.to_pydatetime()
            elif isinstance(txn_date, str):
                try:
                    txn_date = datetime.strptime(txn_date[:10], "%Y-%m-%d")
                except ValueError:
                    continue

            if txn_date < cutoff:
                continue

            shares = row.get("Shares") or row.get("shares") or 0
            txn_type = str(row.get("Transaction") or row.get("Text") or "").lower()

            if shares is None:
                continue
            shares = abs(float(shares))

            if "sale" in txn_type or "sell" in txn_type:
                net_sold += shares
            elif "purchase" in txn_type or "buy" in txn_type:
                net_sold -= shares

        return net_sold / float(shares_outstanding)
    except Exception as exc:
        logger.debug("Insider selling fetch failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# FundamentalFilter
# ---------------------------------------------------------------------------

class FundamentalFilter:
    """Pluggable fundamental quality gate for swing trading candidates.

    Called by the weekly scanner after technical Stage 2 filtering.
    Adds revenue growth, debt safety, institutional ownership,
    earnings proximity, and insider activity checks.

    Data source: yfinance (free, no API key needed).
    Fallback: graceful degradation — if yfinance fails for a symbol,
    it PASSES (don't block on data failure).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise with optional config overrides.

        Config keys:
        - min_revenue_growth: float = 0.20  (20% YoY)
        - max_debt_equity: float = 1.5
        - min_institutional_pct: float = 0.30 (30%)
        - earnings_blackout_before: int = 5 (trading days)
        - earnings_blackout_after: int = 2
        - max_insider_sell_pct: float = 0.01 (1% of float)
        - cache_ttl_hours: int = 168 (7 days — refresh weekly)
        - use_llm_overlay: bool = False
        - llm_model: str = 'gpt-oss:20b'
        - llm_url: str = 'http://localhost:11434'
        - yfinance_timeout: int = 10
        - max_workers: int = 5
        - rate_limit_delay: float = 1.0
        """
        merged = {**DEFAULT_CONFIG}
        if config:
            merged.update(config)

        self._min_revenue_growth: float = merged["min_revenue_growth"]
        self._max_debt_equity: float = merged["max_debt_equity"]
        self._min_institutional_pct: float = merged["min_institutional_pct"]
        self._earnings_blackout_before: int = merged["earnings_blackout_before"]
        self._earnings_blackout_after: int = merged["earnings_blackout_after"]
        self._max_insider_sell_pct: float = merged["max_insider_sell_pct"]
        self._cache_ttl_hours: int = merged["cache_ttl_hours"]
        self._use_llm: bool = merged["use_llm_overlay"]
        self._llm_model: str = merged["llm_model"]
        self._llm_url: str = merged["llm_url"]
        self._yf_timeout: int = merged["yfinance_timeout"]
        self._max_workers: int = merged["max_workers"]
        self._rate_limit_delay: float = merged["rate_limit_delay"]

        # In-memory cache: {symbol: {result: dict, timestamp: float}}
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Database connection (lazy)
        self._conn = None
        self._db_ready = False

    # -- DB helpers ----------------------------------------------------------

    def _get_conn(self):
        """Return a database connection, creating if needed."""
        if self._conn is None or self._conn.closed:
            self._conn = _get_db_connection()
            if self._conn is not None and not self._db_ready:
                _ensure_cache_table(self._conn)
                self._db_ready = True
        return self._conn

    def _load_from_cache(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check in-memory cache, then DB cache."""
        now = time.time()
        ttl_seconds = self._cache_ttl_hours * 3600

        # In-memory
        entry = self._cache.get(symbol)
        if entry and (now - entry["timestamp"]) < ttl_seconds:
            return entry["result"]

        # DB
        conn = self._get_conn()
        if conn:
            result = _load_cached(conn, symbol, self._cache_ttl_hours)
            if result:
                self._cache[symbol] = {"result": result, "timestamp": now}
                return result

        return None

    def _save_to_cache(self, symbol: str, result: Dict[str, Any]) -> None:
        """Save to both in-memory and DB cache."""
        self._cache[symbol] = {"result": result, "timestamp": time.time()}
        conn = self._get_conn()
        if conn:
            _save_cached(conn, symbol, result)

    # -- Core check ----------------------------------------------------------

    def check(self, symbol: str, check_date: Optional[str] = None) -> Dict[str, Any]:
        """Run all fundamental checks on a symbol.

        Parameters
        ----------
        symbol : str
            Ticker symbol (e.g. 'AAPL').
        check_date : str, optional
            Date string 'YYYY-MM-DD' for earnings proximity.
            Defaults to today.

        Returns
        -------
        dict with keys: symbol, passes, revenue_growth, revenue_pass,
            debt_equity, debt_pass, institutional_pct, institutional_pass,
            earnings_near, earnings_date, earnings_pass,
            insider_net_sell_pct, insider_pass, reasons, data_quality.
        """
        # Check cache first
        cached = self._load_from_cache(symbol)
        if cached is not None:
            logger.debug("Cache hit for %s", symbol)
            return cached

        result = _make_empty_result(symbol)
        reasons: List[str] = []
        fields_available = 0
        fields_total = 5

        ref_date = (
            datetime.strptime(check_date, "%Y-%m-%d").date()
            if check_date
            else date.today()
        )

        # Import yfinance lazily to avoid import-time overhead
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed — all checks pass by default")
            result["data_quality"] = "unavailable"
            self._save_to_cache(symbol, result)
            return result

        try:
            ticker = yf.Ticker(symbol)
        except Exception as exc:
            logger.warning("yfinance Ticker(%s) failed: %s", symbol, exc)
            result["data_quality"] = "unavailable"
            self._save_to_cache(symbol, result)
            return result

        # 1. Revenue growth
        rev_growth = _fetch_revenue_growth(ticker)
        if rev_growth is not None:
            fields_available += 1
            result["revenue_growth"] = round(rev_growth, 4)
            if rev_growth < self._min_revenue_growth:
                result["revenue_pass"] = False
                reasons.append(
                    f"Revenue growth {rev_growth * 100:.1f}% < {self._min_revenue_growth * 100:.0f}% min"
                )
        # If None, pass by default (already True)

        # 2. Debt / Equity
        de_ratio = _fetch_debt_equity(ticker)
        if de_ratio is not None:
            fields_available += 1
            result["debt_equity"] = round(de_ratio, 4)
            if de_ratio > self._max_debt_equity:
                result["debt_pass"] = False
                reasons.append(
                    f"Debt/Equity {de_ratio:.2f} > {self._max_debt_equity:.1f} max"
                )

        # 3. Institutional ownership
        inst_pct = _fetch_institutional_pct(ticker)
        if inst_pct is not None:
            fields_available += 1
            result["institutional_pct"] = round(inst_pct, 4)
            if inst_pct < self._min_institutional_pct:
                result["institutional_pass"] = False
                reasons.append(
                    f"Institutional ownership {inst_pct * 100:.1f}% < {self._min_institutional_pct * 100:.0f}% min"
                )

        # 4. Earnings proximity
        earnings_dt = _fetch_earnings_date(ticker)

        # Fallback: check DB earnings_calendar
        if earnings_dt is None:
            conn = self._get_conn()
            if conn:
                earnings_dt = _query_earnings_from_db(conn, symbol)

        if earnings_dt is not None:
            fields_available += 1
            result["earnings_date"] = earnings_dt.isoformat()
            days_until = _trading_days_between(ref_date, earnings_dt)

            if earnings_dt >= ref_date:
                # Earnings in the future
                if days_until <= self._earnings_blackout_before:
                    result["earnings_near"] = True
                    result["earnings_pass"] = False
                    reasons.append(
                        f"Earnings in {days_until} trading days ({earnings_dt}) — blackout zone"
                    )
            else:
                # Earnings in the past
                if days_until <= self._earnings_blackout_after:
                    result["earnings_near"] = True
                    result["earnings_pass"] = False
                    reasons.append(
                        f"Earnings {days_until} trading days ago ({earnings_dt}) — post-earnings blackout"
                    )

        # 5. Insider selling
        insider_pct = _fetch_insider_selling(ticker)
        if insider_pct is not None:
            fields_available += 1
            result["insider_net_sell_pct"] = round(insider_pct, 6)
            if insider_pct > self._max_insider_sell_pct:
                result["insider_pass"] = False
                reasons.append(
                    f"Net insider selling {insider_pct * 100:.3f}% > {self._max_insider_sell_pct * 100:.1f}% max"
                )

        # Aggregate
        result["reasons"] = reasons
        result["passes"] = all([
            result["revenue_pass"],
            result["debt_pass"],
            result["institutional_pass"],
            result["earnings_pass"],
            result["insider_pass"],
        ])

        if fields_available == fields_total:
            result["data_quality"] = "full"
        elif fields_available > 0:
            result["data_quality"] = "partial"
        else:
            result["data_quality"] = "unavailable"

        self._save_to_cache(symbol, result)
        return result

    # -- Batch filter --------------------------------------------------------

    def filter_candidates(
        self, symbols: List[str], check_date: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Filter a list of symbols. Returns {symbol: check_result}.

        Runs checks in parallel (ThreadPoolExecutor) for speed.
        Caches results for cache_ttl_hours.
        """
        results: Dict[str, Dict[str, Any]] = {}

        # Separate cached from uncached
        uncached: List[str] = []
        for sym in symbols:
            cached = self._load_from_cache(sym)
            if cached is not None:
                results[sym] = cached
            else:
                uncached.append(sym)

        if not uncached:
            return results

        def _check_with_delay(sym: str) -> tuple:
            time.sleep(self._rate_limit_delay)
            return sym, self.check(sym, check_date)

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(_check_with_delay, sym): sym for sym in uncached
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    _, result = future.result(timeout=self._yf_timeout + 30)
                    results[sym] = result
                except Exception as exc:
                    logger.warning("Check failed for %s: %s", sym, exc)
                    fallback = _make_empty_result(sym)
                    results[sym] = fallback

        return results

    # -- LLM enhancement (optional) ------------------------------------------

    def llm_analysis(self, symbol: str, fundamental_data: Dict[str, Any]) -> Optional[str]:
        """Ask local Ollama for a swing trade quality verdict.

        OFF by default (use_llm_overlay: false).
        Returns the LLM response string or None on failure.
        """
        if not self._use_llm:
            return None

        try:
            import requests
        except ImportError:
            logger.warning("requests not installed — LLM overlay unavailable")
            return None

        prompt = (
            f"Given these fundamentals for {symbol}, is this stock a quality "
            f"swing trade candidate? Answer YES or NO with one sentence.\n\n"
            f"Revenue Growth: {fundamental_data.get('revenue_growth')}\n"
            f"Debt/Equity: {fundamental_data.get('debt_equity')}\n"
            f"Institutional Ownership: {fundamental_data.get('institutional_pct')}\n"
            f"Earnings Date: {fundamental_data.get('earnings_date')}\n"
            f"Net Insider Selling %: {fundamental_data.get('insider_net_sell_pct')}\n"
            f"Pass/Fail reasons: {fundamental_data.get('reasons')}\n"
        )

        try:
            resp = requests.post(
                f"{self._llm_url}/api/generate",
                json={
                    "model": self._llm_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as exc:
            logger.warning("LLM analysis failed for %s: %s", symbol, exc)
            return None

    # -- Pretty printing -----------------------------------------------------

    def summary(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Pretty-print summary of filter results."""
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("FUNDAMENTAL FILTER RESULTS")
        lines.append("=" * 72)

        passed = [s for s, r in results.items() if r.get("passes")]
        failed = [s for s, r in results.items() if not r.get("passes")]

        lines.append(f"Total: {len(results)}  |  PASS: {len(passed)}  |  FAIL: {len(failed)}")
        lines.append("-" * 72)

        for sym in sorted(results.keys()):
            r = results[sym]
            status = "PASS" if r["passes"] else "FAIL"
            quality = r.get("data_quality", "?")
            lines.append(f"\n  {sym:6s}  [{status}]  (data: {quality})")

            # Show metrics
            rev = r.get("revenue_growth")
            rev_str = f"{rev * 100:.1f}%" if rev is not None else "N/A"
            rev_mark = "ok" if r.get("revenue_pass") else "FAIL"
            lines.append(f"    Revenue Growth:     {rev_str:>8s}  [{rev_mark}]")

            de = r.get("debt_equity")
            de_str = f"{de:.2f}" if de is not None else "N/A"
            de_mark = "ok" if r.get("debt_pass") else "FAIL"
            lines.append(f"    Debt/Equity:        {de_str:>8s}  [{de_mark}]")

            inst = r.get("institutional_pct")
            inst_str = f"{inst * 100:.1f}%" if inst is not None else "N/A"
            inst_mark = "ok" if r.get("institutional_pass") else "FAIL"
            lines.append(f"    Institutional Own:  {inst_str:>8s}  [{inst_mark}]")

            earn_date = r.get("earnings_date") or "N/A"
            earn_mark = "ok" if r.get("earnings_pass") else "FAIL"
            near = " (NEAR)" if r.get("earnings_near") else ""
            lines.append(f"    Earnings Date:      {str(earn_date):>8s}{near}  [{earn_mark}]")

            ins = r.get("insider_net_sell_pct")
            ins_str = f"{ins * 100:.3f}%" if ins is not None else "N/A"
            ins_mark = "ok" if r.get("insider_pass") else "FAIL"
            lines.append(f"    Insider Net Sell:   {ins_str:>8s}  [{ins_mark}]")

            if r.get("reasons"):
                for reason in r["reasons"]:
                    lines.append(f"      -> {reason}")

        lines.append("\n" + "=" * 72)
        return "\n".join(lines)

    # -- Cache management ----------------------------------------------------

    def clear_cache(self) -> None:
        """Clear both in-memory and DB caches."""
        self._cache.clear()
        conn = self._get_conn()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM fundamental_cache")
                conn.commit()
                logger.info("Cache cleared")
            except Exception as exc:
                logger.warning("Failed to clear DB cache: %s", exc)

    def close(self) -> None:
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_stage2_watchlist() -> List[str]:
    """Load current Stage 2 symbols from the database."""
    conn = _get_db_connection()
    if conn is None:
        logger.error("Cannot connect to database for --scan")
        return []

    try:
        with conn.cursor() as cur:
            # Check if a watchlist / stage2 table exists
            cur.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN ('swing_watchlist', 'stage2_candidates', 'watchlist')
                LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                cur.execute(f"SELECT DISTINCT symbol FROM {row[0]} ORDER BY symbol")
                return [r[0] for r in cur.fetchall()]

            # Fallback: pick top liquid symbols from daily_bars
            cur.execute("""
                SELECT symbol
                FROM daily_bars
                WHERE date >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY symbol
                HAVING AVG(volume) > 1000000
                ORDER BY AVG(volume * close) DESC
                LIMIT 50
            """)
            return [r[0] for r in cur.fetchall()]
    except Exception as exc:
        logger.error("Failed to load watchlist: %s", exc)
        return []
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fundamental quality filter for swing trading candidates"
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        help="Ticker symbols to check (e.g. AAPL NVDA TSLA)",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Check current Stage 2 watchlist from database",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear cached fundamental data",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Reference date for earnings check (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM overlay analysis",
    )

    args = parser.parse_args()

    config = {}
    if args.llm:
        config["use_llm_overlay"] = True

    ff = FundamentalFilter(config)

    try:
        if args.clear_cache:
            ff.clear_cache()
            logger.info("Cache cleared successfully")
            if not args.symbols and not args.scan:
                return

        symbols: List[str] = []
        if args.scan:
            symbols = _load_stage2_watchlist()
            if not symbols:
                logger.error("No symbols found in watchlist")
                return
            logger.info("Loaded %d symbols from watchlist", len(symbols))
        elif args.symbols:
            symbols = [s.upper() for s in args.symbols]
        else:
            parser.print_help()
            return

        logger.info("Checking %d symbols: %s", len(symbols), ", ".join(symbols))
        results = ff.filter_candidates(symbols, check_date=args.date)

        summary_text = ff.summary(results)
        print(summary_text)

        # LLM overlay if enabled
        if ff._use_llm:
            print("\n--- LLM Analysis ---")
            for sym, data in results.items():
                verdict = ff.llm_analysis(sym, data)
                if verdict:
                    print(f"  {sym}: {verdict}")

        passed = [s for s, r in results.items() if r.get("passes")]
        failed = [s for s, r in results.items() if not r.get("passes")]
        print(f"\nPassed: {', '.join(sorted(passed)) or 'none'}")
        print(f"Failed: {', '.join(sorted(failed)) or 'none'}")

    finally:
        ff.close()


if __name__ == "__main__":
    main()
