"""core/scalp_screener.py — Score and select symbols for the scalp shadow loop.

Standalone module: no imports from core.engine or other bot modules.
Requires only yfinance, pandas, asyncio, and dataclasses.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import yfinance as yf

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SymbolScore:
    """Immutable score record for a single symbol."""

    symbol: str
    gap_pct: float
    rvol: float
    news_score: float
    composite: float
    price: float
    avg_daily_volume: int  # historical daily average volume (prior days)


class ScalpScreener:
    """Score a universe of symbols by gap, relative volume, and optional news sentiment.

    Args:
        universe:       List of ticker symbols to screen.
        top_n:          Maximum symbols returned by :meth:`select`.
        min_price:      Minimum price filter (inclusive).
        max_price:      Maximum price filter (inclusive).
        min_avg_volume: Minimum effective daily volume filter (inclusive).
    """

    def __init__(
        self,
        universe: list[str],
        top_n: int = 5,
        min_price: float = 3.0,
        max_price: float = 500.0,
        min_avg_volume: int = 500_000,
    ) -> None:
        self.universe = list(universe)
        self.top_n = top_n
        self.min_price = min_price
        self.max_price = max_price
        self.min_avg_volume = min_avg_volume

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _fetch_quotes(self) -> dict[str, dict]:
        """Download 30-day daily OHLCV for all universe symbols via yfinance.

        Returns a dict ``{symbol: {prev_close, open, volume, avg_daily_volume, price}}``.
        avg_daily_volume is computed from prior days (all rows except the most recent),
        so it is a true historical baseline uncontaminated by today's partial bar.
        Symbols with no data are silently omitted.
        """
        if not self.universe:
            return {}

        loop = asyncio.get_running_loop()
        tickers = self.universe if len(self.universe) > 1 else self.universe[0]

        def _download() -> object:
            return yf.download(
                tickers,
                period="30d",
                interval="1d",
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )

        try:
            df = await loop.run_in_executor(None, _download)
        except Exception as exc:
            log.warning("yfinance download failed: %s", exc)
            return {}

        if df is None or df.empty:
            return {}

        quotes: dict[str, dict] = {}

        for sym in self.universe:
            try:
                # Single-symbol download → flat columns; multi-symbol → grouped by ticker
                if len(self.universe) == 1:
                    sym_df = df
                else:
                    top_level = df.columns.get_level_values(0).unique()
                    if sym not in top_level:
                        continue
                    sym_df = df[sym]

                sym_df = sym_df.dropna(how="all")
                if len(sym_df) < 2:
                    continue

                if len(sym_df) >= 2:
                    avg_daily_volume = int(sym_df.iloc[:-1]["Volume"].mean())
                    current_volume = int(sym_df.iloc[-1]["Volume"])
                else:
                    avg_daily_volume = 0
                    current_volume = 0

                quotes[sym] = {
                    "prev_close": float(sym_df["Close"].iloc[-2]),
                    "open": float(sym_df["Open"].iloc[-1]),
                    "volume": float(current_volume),
                    "avg_daily_volume": float(avg_daily_volume),
                    "price": float(sym_df["Close"].iloc[-1]),
                }
            except Exception as exc:
                log.debug("Skipping %s: %s", sym, exc)
                continue

        return quotes

    def _score_quote(self, q: dict, symbol: str = "") -> SymbolScore:
        """Compute a :class:`SymbolScore` for a single quote dict.

        Args:
            q:      Quote dict with keys prev_close, open, volume, avg_daily_volume, price.
                    avg_daily_volume is the historical daily average (prior days only).
                    volume is today's current (possibly partial) bar volume.
            symbol: Ticker string stored on the resulting :class:`SymbolScore`.
        """
        prev_close: float = q.get("prev_close", 0.0)
        open_price: float = q.get("open", 0.0)
        raw_volume: float = q.get("volume", 0.0)
        hist_avg: float = q.get("avg_daily_volume", 0.0)
        price: float = q.get("price", 0.0)

        # Gap percent
        if prev_close > 0:
            gap_pct = (open_price - prev_close) / prev_close * 100.0
        else:
            gap_pct = 0.0

        # Effective today volume: pre-market fallback when actual volume is tiny
        today_vol = raw_volume if raw_volume > 1000 else hist_avg

        # Relative volume: today vs historical daily average
        if hist_avg > 0:
            rvol = today_vol / hist_avg
        else:
            rvol = 1.0

        # Normalised component scores (each in [0, 1])
        gap_score = min(abs(gap_pct) / 10.0, 1.0)
        rvol_score = min(rvol / 5.0, 1.0)

        composite = gap_score * 0.5 + rvol_score * 0.5

        return SymbolScore(
            symbol=symbol,
            gap_pct=gap_pct,
            rvol=rvol,
            news_score=0.0,
            composite=composite,
            price=price,
            avg_daily_volume=int(hist_avg),
        )

    def _filter(self, scores: list[SymbolScore]) -> list[SymbolScore]:
        """Return only scores that meet price and volume criteria."""
        return [
            s for s in scores
            if self.min_price <= s.price <= self.max_price
            and s.avg_daily_volume >= self.min_avg_volume
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def score(self, news_bus: Optional[object] = None) -> list[SymbolScore]:
        """Score all universe symbols and return them sorted by composite descending.

        Args:
            news_bus: Optional object with ``async get_aggregate_sentiment(symbol)``
                      method. When provided, news score is blended in with weights
                      gap=0.4, rvol=0.4, news=0.2.

        Returns:
            Filtered and sorted list of :class:`SymbolScore` objects.
        """
        try:
            quotes = await self._fetch_quotes()
        except Exception as exc:
            log.error("_fetch_quotes raised unexpectedly: %s", exc)
            return []

        raw_scores: list[SymbolScore] = []
        for sym in self.universe:
            q = quotes.get(sym)
            if q is None:
                continue
            try:
                raw_scores.append(self._score_quote(q, sym))
            except Exception as exc:
                log.debug("Skipping %s during scoring: %s", sym, exc)

        filtered = self._filter(raw_scores)

        if news_bus is not None:
            enriched: list[SymbolScore] = []
            for s in filtered:
                news_score = 0.0
                try:
                    news_score = float(
                        await news_bus.get_aggregate_sentiment(s.symbol)
                    )
                except Exception as exc:
                    log.debug("News bus error for %s: %s", s.symbol, exc)

                gap_score = min(abs(s.gap_pct) / 10.0, 1.0)
                rvol_score = min(s.rvol / 5.0, 1.0)
                news_component = min(abs(news_score), 1.0)
                composite = (
                    gap_score * 0.4
                    + rvol_score * 0.4
                    + news_component * 0.2
                )
                enriched.append(
                    SymbolScore(
                        symbol=s.symbol,
                        gap_pct=s.gap_pct,
                        rvol=s.rvol,
                        news_score=news_score,
                        composite=composite,
                        price=s.price,
                        avg_daily_volume=s.avg_daily_volume,
                    )
                )
            filtered = enriched

        return sorted(filtered, key=lambda s: s.composite, reverse=True)

    async def select(self, news_bus: Optional[object] = None) -> list[str]:
        """Return the top ``top_n`` symbol strings from :meth:`score`.

        Args:
            news_bus: Forwarded to :meth:`score`.

        Returns:
            List of ticker strings, highest composite first, length ≤ ``top_n``.
        """
        scored = await self.score(news_bus)
        return [s.symbol for s in scored][: self.top_n]
