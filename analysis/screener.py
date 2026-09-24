import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from core.models import CommentaryType
from core.commentary import TradingCommentary

logger = logging.getLogger('TradingBot')


# v-universe-quality-filter-2026-05-22: leveraged / inverse / volatility
# ETFs to always exclude from the bot's tradeable universe. These
# decay daily regardless of direction, so mean-reversion and breakout
# logic both break. SOXS -$228 loss on 5/22 was the prompt for this.
LEVERAGED_ETF_BLOCKLIST = frozenset({
    # 2x / 3x long
    "TQQQ", "SQQQ", "UPRO", "SPXU", "SPXL", "SPXS",
    "TNA", "TZA", "FAS", "FAZ", "TMF", "TMV",
    "QLD", "QID", "SSO", "SDS",
    "SOXL", "SOXS",
    "LABU", "LABD",
    "NUGT", "DUST",
    "JNUG", "JDST",
    "ERX", "ERY",
    "URTY", "SRTY",
    "TECL", "TECS",
    # Volatility
    "UVXY", "SVXY", "VXX", "VIXY", "TVIX",
    # Single-name 2x/3x
    "TSLL", "TSLT", "TSLZ", "NVDL", "NVDS", "NVDD",
    "MSTU", "MSTX", "AMDL", "AMDS",
    # Currency / commodity leveraged
    "UCO", "SCO", "USO", "USL",
    "AGQ", "ZSL", "UGL", "GLL",
})


class StockScreener:
    """Real-time stock screener using Schwab API"""

    def __init__(self, schwab_client, commentary_system):
        self.client = schwab_client
        self.commentary = commentary_system
        self.top_movers = []
        self.last_screen_time = None
        self.screen_interval = 120  # 2 minutes
        self._movers_cache = {}
        self._cache_timeout = 60  # 1 minute cache for API efficiency
        # v-universe-quality-filter-2026-05-22: batch-fetched 60d daily
        # bars cached for ~30 min so we don't hammer yfinance every
        # screener cycle. SMA50 / relative strength / 5d return all
        # derive from this cache.
        self._quality_data_cache: Dict[str, pd.DataFrame] = {}
        self._quality_data_timestamp: float = 0.0
        self._quality_data_ttl_s: int = 30 * 60
        # Track filter rejections per cycle for observability.
        self._quality_filter_rejects: Dict[str, int] = {}

    async def screen_stocks(self) -> List[Dict[str, Any]]:
        """Screen for top moving and volatile stocks.

        v-screener-fresh-daily-2026-05-08: prior implementation called
        the schwab-py movers API with `direction=` and `change=` kwargs
        which were removed when schwab-py migrated to the new
        sort_order/frequency API. Every dynamic source raised
        TypeError, got swallowed by logger.debug, and the screener
        silently fell back to the static volatile_candidates pool. As a
        result, RKLB / OPEN / CRWV style names never appeared even on
        days when they were the market's biggest movers.

        v-yahoo-most-active-2026-05-11: prepended Yahoo Finance
        most-active as Source 0. Yahoo's screener pulls retail-favorite
        catalyst names (RKLB, IREN, RDW, EOSE, etc.) that Schwab's
        index-bound movers consistently miss. The endpoint is the
        undocumented but stable internal Yahoo screener JSON, same data
        that powers https://finance.yahoo.com/markets/stocks/most-active/.
        No auth, no quota, ~30 names per call.

        Sources in priority order (deduped at the end):
          0. Yahoo most-active (30 names) — retail catalyst breadth
          1. EQUITY_ALL by VOLUME            — Schwab volume leaders
          2. EQUITY_ALL by PERCENT_CHANGE_UP  — Schwab % gainers
          3. EQUITY_ALL by PERCENT_CHANGE_DOWN — Schwab % losers
          4. $SPX  by PERCENT_CHANGE_UP/DOWN  — S&P leaders
          5. $COMPX by PERCENT_CHANGE_UP/DOWN — NASDAQ leaders
          6. volatile_candidates              — operator's curated floor
        """
        try:
            # Get movers from major indices and broad equity universe
            all_movers = []

            # Source 0: Yahoo Finance most-active (best breadth).
            # Catches RKLB / IREN / RDW / EOSE class catalyst names that
            # Schwab's index movers regularly miss because those indices
            # are market-cap-weighted toward large caps.
            yahoo_active = await self._get_yahoo_most_active()
            all_movers.extend(yahoo_active)

            # Source 0b (v-yahoo-top-movers-2026-05-20): Yahoo day_gainers
            # + day_losers. Most-active sorts by VOLUME, missing big-%-move
            # names with moderate volume. ARM (+14% on 5/20, ~6.8M vol)
            # was the canonical example — invisible to the bot all day
            # because it never cleared the most-active top-30. Day_gainers
            # is %-ranked and catches these directly.
            yahoo_top = await self._get_yahoo_top_movers()
            all_movers.extend(yahoo_top)

            # Source 1: EQUITY_ALL by VOLUME — today's most-active.
            # Catches RKLB-class names that are heavily traded but not
            # in the % movers leaderboard. This is THE source for the
            # "names everyone is watching today" signal.
            most_active = await self._get_index_movers(
                'EQUITY_ALL', sort_order='VOLUME',
            )
            all_movers.extend(most_active)

            # Sources 2–3: broad-universe % gainers / losers
            equity_up = await self._get_index_movers(
                'EQUITY_ALL', sort_order='PERCENT_CHANGE_UP',
            )
            all_movers.extend(equity_up)
            equity_down = await self._get_index_movers(
                'EQUITY_ALL', sort_order='PERCENT_CHANGE_DOWN',
            )
            all_movers.extend(equity_down)

            # Sources 4–5: S&P + NASDAQ leaders for index-aware flow
            for idx in ('$SPX', '$COMPX'):
                up = await self._get_index_movers(idx, sort_order='PERCENT_CHANGE_UP')
                all_movers.extend(up)
                down = await self._get_index_movers(idx, sort_order='PERCENT_CHANGE_DOWN')
                all_movers.extend(down)

            # Source 6: operator's curated floor — names you always want
            # the bot to consider regardless of today's tape.
            volatile_stocks = await self._get_volatile_stocks()
            all_movers.extend(volatile_stocks)

            # Remove duplicates and filter
            seen = set()
            unique_movers = []

            # v-universe-quality-filter-2026-05-22: refresh quality data
            # cache once per cycle if stale, then apply 3-gate filter
            # alongside existing _is_tradeable + dedup. Filter rejections
            # are logged per-symbol with reason for backtestability.
            #
            # v-mover-quality-relax-2026-09-10: when ENABLE_MOVER_QUALITY_RELAX
            # is True, symbols from Yahoo day_gainers/day_losers/most_active
            # bypass the SMA50/RS filters. They still respect leveraged ETF
            # blocklist, price floor, and volume floor.
            try:
                from core.config import Config as _CfgUQ
                _enabled = _CfgUQ().ENABLE_UNIVERSE_QUALITY_FILTER
                _mover_relax = _CfgUQ().ENABLE_MOVER_QUALITY_RELAX
            except Exception:
                _enabled = True
                _mover_relax = True
            if _enabled:
                _candidate_syms = list({m['symbol'] for m in all_movers
                                        if m.get('symbol')})
                await self._refresh_quality_data(_candidate_syms)
                self._quality_filter_rejects = {}  # reset per cycle
                self._mover_quality_relax_passes = {}  # v-mover-quality-relax-2026-09-10

            for mover in all_movers:
                sym = mover.get('symbol')
                if sym in seen:
                    continue
                if not self._is_tradeable(mover):
                    continue
                
                # v-mover-quality-relax-2026-09-10: check if this is a mover source
                # that qualifies for relaxed quality filtering
                _source = mover.get('source', '')
                _is_mover_source = _source in (
                    'yahoo_most_active', 'yahoo_day_gainers', 'yahoo_day_losers'
                )
                mover['is_mover'] = _is_mover_source  # Tag for downstream use
                
                if _enabled:
                    if _mover_relax and _is_mover_source:
                        # Use relaxed filter for movers
                        if not self._passes_mover_quality_filter(mover):
                            continue
                        self._mover_quality_relax_passes[sym] = True
                    else:
                        # Use standard quality filter
                        if not self._passes_quality_filter(sym):
                            continue
                
                seen.add(sym)
                unique_movers.append(mover)

            if _enabled and self._quality_filter_rejects:
                logger.info(
                    "screener: universe quality filter rejected %d candidates: %s",
                    sum(self._quality_filter_rejects.values()),
                    ", ".join(f"{k}={v}" for k, v in self._quality_filter_rejects.items()),
                )
            
            # v-mover-quality-relax-2026-09-10: log how many movers passed via relaxed filter
            if _mover_relax and hasattr(self, '_mover_quality_relax_passes') and self._mover_quality_relax_passes:
                logger.info(
                    "screener: mover quality relax passed %d movers: %s",
                    len(self._mover_quality_relax_passes),
                    ", ".join(list(self._mover_quality_relax_passes.keys())[:10]),
                )

            # Sort by combined score (volatility + volume)
            scored_movers = []
            for mover in unique_movers:
                score = self._calculate_mover_score(mover)
                mover['score'] = score
                scored_movers.append(mover)

            # v-watchlist-size-2026-04-29: cap raised 10 → 30 so
            # get_watchlist_symbols can serve up to Config.WATCHLIST_SIZE
            # without re-running the screener.
            scored_movers.sort(key=lambda x: x['score'], reverse=True)
            self.top_movers = scored_movers[:30]

            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.MARKET_ANALYSIS,
                symbol=None,
                title="\U0001f50d Stock Screener Update",
                message=f"Found {len(self.top_movers)} high-opportunity stocks",
                data={
                    'top_gainer': self.top_movers[0]['symbol'] if self.top_movers else None,
                    # v-screener-volatility-keyerror-2026-09-02: movers
                    # only carry 'volatility' when Schwab quote enrichment
                    # succeeded. Hard access here killed every cycle while
                    # the token was expired (7/29–7/31), disabling the
                    # Yahoo-only fallback exactly when it was needed.
                    'avg_volatility': np.mean([m.get('volatility', 0) for m in self.top_movers]) if self.top_movers else 0
                },
                importance=6
            ))

            self.last_screen_time = datetime.now()
            return self.top_movers

        except Exception as e:
            logger.error(f"Screener error: {e}")
            return []

    async def _get_index_movers(
        self,
        index: str,
        sort_order: str = 'PERCENT_CHANGE_UP',
    ) -> List[Dict[str, Any]]:
        """Get movers for a specific index, sorted by VOLUME, TRADES,
        PERCENT_CHANGE_UP, or PERCENT_CHANGE_DOWN.

        v-screener-api-fix-2026-05-08: schwab-py migrated this endpoint
        from `(index, direction=, change=)` to
        `(index, sort_order=, frequency=)`. The prior call signature
        raised TypeError on every invocation, was swallowed by the
        debug logger, and the screener silently became static-only.

        Direction is implicit in the sort_order:
          PERCENT_CHANGE_UP   → top gainers
          PERCENT_CHANGE_DOWN → top losers
          VOLUME              → most active (direction=neutral)
          TRADES              → most traded (direction=neutral)

        Schwab returns up to 10 names per call (the API uses to be 20;
        new docs say 10). We take whatever is returned. The response
        shape also changed: items live under the 'screeners' key in
        the new payload, with fields described, lastPrice, netChange,
        netPercentChange, volume.
        """
        cache_key = f"movers_{index}_{sort_order}"

        # Check cache
        if cache_key in self._movers_cache:
            cached_data, timestamp = self._movers_cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data

        movers = []
        direction_for_label = (
            'up' if sort_order == 'PERCENT_CHANGE_UP'
            else 'down' if sort_order == 'PERCENT_CHANGE_DOWN'
            else 'neutral'
        )

        # v-screener-enum-fix-2026-05-11: schwab-py enforces enum types
        # by default (enforce_enums=True). Passing strings raises
        # ValueError. Resolve the enum members from the client class so
        # the call is type-safe regardless of enforce_enums setting.
        try:
            _Index = type(self.client).Movers.Index
            _SortOrder = type(self.client).Movers.SortOrder
            index_enum = getattr(_Index, index.lstrip('$'), None) or _Index(index)
            sort_enum = getattr(_SortOrder, sort_order, None) or _SortOrder(sort_order)
        except Exception as exc:
            logger.warning(
                "screener: enum resolution failed for index=%s sort_order=%s: %s",
                index, sort_order, exc,
            )
            self._movers_cache[cache_key] = ([], time.time())
            return []

        try:
            response = self.client.get_movers(index_enum, sort_order=sort_enum)

            if response.status_code == 200:
                payload = response.json()
                # New API: screeners list. Old fallback: a bare list.
                items = (
                    payload.get('screeners')
                    if isinstance(payload, dict)
                    else (payload if isinstance(payload, list) else [])
                ) or []

                for item in items:
                    sym = item.get('symbol') or item.get('description')
                    if not sym:
                        continue
                    last_price = (
                        item.get('lastPrice')
                        or item.get('last')
                        or 0
                    )
                    net_change = (
                        item.get('netChange')
                        or item.get('change')
                        or 0
                    )
                    pct_change = (
                        item.get('netPercentChange')
                        or item.get('changePercent')
                        or 0
                    )
                    volume = (
                        item.get('volume')
                        or item.get('totalVolume')
                        or 0
                    )

                    mover = {
                        'symbol': sym,
                        'description': item.get('description', ''),
                        'last': last_price,
                        'change': net_change,
                        'percent_change': pct_change,
                        'volume': volume,
                        'direction': direction_for_label,
                        'source': f"{index}/{sort_order}",
                    }

                    # Get detailed quote for more data
                    try:
                        quote = self._get_quote_data(sym)
                        if quote:
                            mover.update(quote)
                    except Exception as exc:
                        logger.debug(f"quote fetch failed for {sym}: {exc}")

                    movers.append(mover)
            else:
                # Promote to warning so the operator sees a broken sub-source
                logger.warning(
                    "screener: get_movers(%s, %s) returned HTTP %s",
                    index, sort_order, response.status_code,
                )
        except Exception as e:
            # Promote to warning. The old debug-level log let the API
            # signature break go unnoticed for weeks.
            logger.warning(
                "screener: get_movers(%s, %s) raised %s: %s",
                index, sort_order, type(e).__name__, e,
            )

        # Cache results
        self._movers_cache[cache_key] = (movers, time.time())
        return movers

    async def _get_yahoo_most_active(self) -> List[Dict[str, Any]]:
        """Fetch Yahoo Finance most-active list.

        v-yahoo-most-active-2026-05-11: hits the same undocumented JSON
        endpoint that powers
        https://finance.yahoo.com/markets/stocks/most-active/. Returns
        the top ~30 most-traded names by current-session volume, no
        auth required. Operator's #1 ask after noticing the bot's
        Schwab-only screener kept missing retail catalyst names.

        Defensive against:
          - Yahoo rate-limiting (HTTP 429)
          - schema drift (every field has a safe default)
          - network hiccups (timeout=10s)
          - sync `requests` blocking the event loop (offloaded via
            asyncio.to_thread)
        """
        cache_key = "movers_yahoo_most_active"
        if cache_key in self._movers_cache:
            cached_data, timestamp = self._movers_cache[cache_key]
            if time.time() - timestamp < self._cache_timeout:
                return cached_data

        import asyncio as _asyncio
        movers: List[Dict[str, Any]] = []

        def _fetch():
            import requests
            url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            params = {
                "formatted": "true",
                "lang": "en-US",
                "region": "US",
                "scrIds": "most_actives",
                "count": 30,
            }
            headers = {
                # Yahoo blocks requests without a UA.
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            }
            return requests.get(url, params=params, headers=headers, timeout=10)

        try:
            response = await _asyncio.to_thread(_fetch)
            if response.status_code != 200:
                logger.warning(
                    "screener: yahoo most-active returned HTTP %s",
                    response.status_code,
                )
                self._movers_cache[cache_key] = ([], time.time())
                return []

            payload = response.json()
            quotes = (
                payload.get("finance", {})
                .get("result", [{}])[0]
                .get("quotes", [])
            )

            def _scalar(v):
                """Yahoo returns either raw value or {raw, fmt} dict."""
                if isinstance(v, dict):
                    return v.get("raw", 0)
                return v or 0

            for q in quotes:
                sym = q.get("symbol")
                if not sym:
                    continue
                last_price = float(_scalar(q.get("regularMarketPrice")) or 0)
                net_change = float(_scalar(q.get("regularMarketChange")) or 0)
                pct_change = float(_scalar(q.get("regularMarketChangePercent")) or 0)
                volume = float(_scalar(q.get("regularMarketVolume")) or 0)

                mover = {
                    "symbol": sym,
                    "description": q.get("shortName", "") or q.get("longName", ""),
                    "last": last_price,
                    "change": net_change,
                    "percent_change": pct_change,
                    "volume": volume,
                    "direction": (
                        "up" if pct_change > 0
                        else "down" if pct_change < 0
                        else "neutral"
                    ),
                    "source": "yahoo_most_active",
                }

                # Best-effort enrich from Schwab for full-quote fields
                # downstream code expects (spread, bid/ask). Failure is
                # fine — Yahoo data alone is enough for ranking.
                try:
                    quote = self._get_quote_data(sym)
                    if quote:
                        mover.update(quote)
                except Exception as exc:
                    logger.debug(f"quote enrich failed for {sym}: {exc}")

                movers.append(mover)

            logger.info(
                "screener: yahoo most-active returned %d names (top: %s)",
                len(movers),
                ", ".join(m["symbol"] for m in movers[:5]),
            )
        except Exception as exc:
            logger.warning(
                "screener: yahoo most-active raised %s: %s",
                type(exc).__name__, exc,
            )

        self._movers_cache[cache_key] = (movers, time.time())
        return movers

    async def _get_yahoo_top_movers(self) -> List[Dict[str, Any]]:
        """v-yahoo-top-movers-2026-05-20: fetch Yahoo day_gainers + day_losers.

        Closes the "big-move, moderate-volume" blind spot in the screener.
        On 5/20 the bot missed ARM (+14.4%, $32 gap) because ARM was not in
        Yahoo's most-active list (volume-ranked). Day_gainers IS percent-
        ranked, so ARM would appear there.

        Returns the union of top 30 gainers + top 30 losers (deduplicated
        by symbol). Defensive against the same failure modes as
        _get_yahoo_most_active.
        """
        cache_key = "movers_yahoo_top_movers"
        if cache_key in self._movers_cache:
            cached_data, ts = self._movers_cache[cache_key]
            if time.time() - ts < self._cache_timeout:
                return cached_data

        import asyncio as _asyncio
        movers: List[Dict[str, Any]] = []
        seen_symbols: set = set()

        def _fetch(scr_id: str):
            import requests
            url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            params = {
                "formatted": "true",
                "lang": "en-US",
                "region": "US",
                "scrIds": scr_id,
                "count": 30,
            }
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            }
            return requests.get(url, params=params, headers=headers, timeout=10)

        def _scalar(v):
            if isinstance(v, dict):
                return v.get("raw", 0)
            return v or 0

        for scr_id, source_tag in (("day_gainers", "yahoo_day_gainers"),
                                   ("day_losers", "yahoo_day_losers")):
            try:
                response = await _asyncio.to_thread(_fetch, scr_id)
                if response.status_code != 200:
                    logger.warning(
                        "screener: yahoo %s returned HTTP %s",
                        scr_id, response.status_code,
                    )
                    continue
                payload = response.json()
                quotes = (
                    payload.get("finance", {})
                    .get("result", [{}])[0]
                    .get("quotes", [])
                )
                added = 0
                for q in quotes:
                    sym = q.get("symbol")
                    if not sym or sym in seen_symbols:
                        continue
                    seen_symbols.add(sym)
                    last_price = float(_scalar(q.get("regularMarketPrice")) or 0)
                    net_change = float(_scalar(q.get("regularMarketChange")) or 0)
                    pct_change = float(_scalar(q.get("regularMarketChangePercent")) or 0)
                    volume = float(_scalar(q.get("regularMarketVolume")) or 0)
                    mover = {
                        "symbol": sym,
                        "description": q.get("shortName", "") or q.get("longName", ""),
                        "last": last_price,
                        "change": net_change,
                        "percent_change": pct_change,
                        "volume": volume,
                        "direction": (
                            "up" if pct_change > 0
                            else "down" if pct_change < 0
                            else "neutral"
                        ),
                        "source": source_tag,
                    }
                    # Best-effort Schwab enrich (same pattern as most_active).
                    try:
                        quote = self._get_quote_data(sym)
                        if quote:
                            mover.update(quote)
                    except Exception as exc:
                        logger.debug("quote enrich failed for %s: %s", sym, exc)
                    movers.append(mover)
                    added += 1
                logger.info(
                    "screener: yahoo %s returned %d names (added %d new)",
                    scr_id, len(quotes), added,
                )
            except Exception as exc:
                logger.warning(
                    "screener: yahoo %s raised %s: %s",
                    scr_id, type(exc).__name__, exc,
                )

        self._movers_cache[cache_key] = (movers, time.time())
        return movers

    async def _get_volatile_stocks(self) -> List[Dict[str, Any]]:
        """Find additional volatile stocks using a pre-screened watchlist.

        v-watchlist-size-2026-04-29: expanded candidate list 20 → 50 names
        and dropped the per-candidate volume gate from 500k → 100k. The
        general _is_tradeable filter still applies downstream; this
        function is the SOURCING step, not the filtering step.
        """
        volatile_candidates = [
            # Mega-cap tech / AI
            'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'AMD', 'TSLA',
            'AVGO', 'ORCL', 'CRM', 'ADBE', 'NFLX', 'INTC', 'QCOM', 'CSCO',
            # AI / semiconductor / cloud
            'PLTR', 'SMCI', 'ARM', 'MU', 'AMAT', 'LRCX', 'KLAC', 'MRVL',
            'CRWV',  # v-watchlist-add-2026-05-08: AI/GPU cloud infra
            # Crypto / fintech
            'COIN', 'MARA', 'RIOT', 'SQ', 'PYPL', 'HOOD', 'SOFI',
            'OPEN',  # v-watchlist-add-2026-05-08: real-estate iBuying, news-sensitive
            # EV / clean energy
            'NIO', 'XPEV', 'LI', 'RIVN', 'LCID', 'PLUG', 'FCEL',
            # Consumer / social
            'ROKU', 'SNAP', 'PINS', 'DKNG', 'PENN', 'FUBO', 'CHWY', 'PTON',
            # Biotech volatility
            'MRNA', 'BNTX', 'NVAX',
            # v-aerospace-space-2026-05-08: aerospace / space / eVTOL
            # category was missing entirely. Operator flagged RKLB as a
            # notable absence 2026-05-08. These names regularly produce
            # double-digit moves on launch / contract / analyst news and
            # match the bot's news-strategy edge.
            'RKLB', 'JOBY', 'ACHR', 'LUNR', 'ASTS',
            # v-watchlist-add-2026-05-26: LRCX rallied +15% on 2026-05-26
            # with zero bot evaluations because it wasn't in any input
            # source (not yahoo-active that day, no quality-filter ever
            # reached). Add it + adjacent semi-equipment names. ASML is
            # the ADR. Adding cloud/marketplace names that move on
            # earnings + analyst notes but lack steady high-vol sourcing.
            'ASML', 'NOW', 'SHOP', 'UBER', 'ABNB',
            # Index / sector ETFs (will be filtered by _is_tradeable
            # if they're on the ETF block-list, but kept here for sourcing)
            'TQQQ', 'SQQQ',
        ]

        volatile_stocks = []
        for symbol in volatile_candidates:
            try:
                quote = self._get_quote_data(symbol)
                # Lower bar — sourcing stage. _is_tradeable does the real
                # liquidity/spread filtering with current market context.
                if quote and quote.get('volume', 0) > 100_000:
                    volatile_stocks.append(quote)
            except Exception as e:
                logger.debug(f"Failed to get quote for {symbol}: {e}")
                continue

        return volatile_stocks

    def _get_quote_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed quote data for a symbol"""
        try:
            response = self.client.get_quote(symbol)
            if response.status_code == 200:
                data = response.json()
                quote_data = data.get(symbol, {})

                if 'quote' in quote_data:
                    quote = quote_data['quote']
                else:
                    quote = quote_data

                # Calculate volatility metrics
                high = quote.get('highPrice', 0)
                low = quote.get('lowPrice', 0)
                open_price = quote.get('openPrice', 1)
                last = quote.get('lastPrice', 0)

                # Intraday range percentage (best for day trading)
                volatility = ((high - low) / open_price * 100) if open_price > 0 else 0

                return {
                    'symbol': symbol,
                    'description': quote.get('description', ''),
                    'last': last,
                    'change': quote.get('netChange', 0),
                    # Schwab field is netPercentChange (the legacy
                    # netPercentChangeInDouble name returned None, which made
                    # every ticker show 0.00% on the dashboard).
                    'percent_change': quote.get('netPercentChange',
                        quote.get('netPercentChangeInDouble', 0)),
                    'volume': quote.get('totalVolume', 0),
                    'high': high,
                    'low': low,
                    'open': open_price,
                    'volatility': volatility,
                    'bid': quote.get('bidPrice', 0),
                    'ask': quote.get('askPrice', 0),
                    'spread': quote.get('askPrice', 0) - quote.get('bidPrice', 0)
                }
        except Exception as e:
            logger.debug(f"Quote error for {symbol}: {e}")

        return None

    async def _refresh_quality_data(self, symbols: List[str]) -> None:
        """v-universe-quality-filter-2026-05-22: batch-fetch 60d daily
        bars for all candidates + SPY in one yfinance call. Cached for
        30 min so screener cycles (every ~3 min) don't hammer yfinance.

        SMA50 and 5d-return derive from this cache. Pure side-effect:
        populates self._quality_data_cache. Never raises — yfinance
        failures degrade to "no quality data → all pass the filter".
        """
        now = time.time()
        if now - self._quality_data_timestamp < self._quality_data_ttl_s:
            # Cache still valid; only fetch symbols we don't have yet.
            missing = [s for s in symbols if s not in self._quality_data_cache]
            if not missing:
                return
        else:
            missing = list(symbols)

        # Always include SPY for relative-strength comparisons.
        if 'SPY' not in self._quality_data_cache or now - self._quality_data_timestamp >= self._quality_data_ttl_s:
            if 'SPY' not in missing:
                missing.append('SPY')

        if not missing:
            return

        import asyncio as _asyncio

        def _fetch():
            try:
                import yfinance as yf
            except Exception as exc:
                logger.warning("screener: yfinance import failed: %s", exc)
                return {}
            try:
                # 60 trading days ≈ 90 calendar days. Daily bars only.
                # auto_adjust=True → splits/dividends handled.
                df = yf.download(
                    tickers=missing,
                    period='90d',
                    interval='1d',
                    group_by='ticker',
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                return df
            except Exception as exc:
                logger.warning("screener: yfinance batch fetch failed: %s", exc)
                return {}

        result = await _asyncio.to_thread(_fetch)
        # Result shape: MultiIndex columns when multiple tickers,
        # flat DataFrame when single. Normalize to {sym: DataFrame}.
        try:
            if isinstance(result, pd.DataFrame):
                if isinstance(result.columns, pd.MultiIndex):
                    for sym in missing:
                        if sym in result.columns.get_level_values(0):
                            sym_df = result[sym].dropna()
                            if not sym_df.empty:
                                self._quality_data_cache[sym] = sym_df
                elif len(missing) == 1:
                    self._quality_data_cache[missing[0]] = result.dropna()
        except Exception as exc:
            logger.warning("screener: quality data normalize failed: %s", exc)

        self._quality_data_timestamp = now
        logger.info(
            "screener: quality_data refreshed for %d symbols (cache size %d)",
            len(missing), len(self._quality_data_cache),
        )

    def _passes_quality_filter(self, symbol: str) -> bool:
        """v-universe-quality-filter-2026-05-22: three-gate quality check.

        1. Leveraged/inverse ETF blocklist — daily-decay instruments
           break mean-rev and breakout logic; exclude entirely.
        2. Trend filter — close > SMA50. Don't fade names making new
           50-day lows (avoids catching falling knives at universe time).
        3. Relative strength — 5-day return > SPY 5-day return. Filters
           "everyone's selling this name" candidates.

        Returns True if symbol passes all three. Logs reason via
        self._quality_filter_rejects on any reject.

        Conservative: if data is missing for a symbol, the trend +
        relative-strength gates pass-through (don't block on missing
        data). Only the blocklist is hard.
        """
        # Gate 1: hard blocklist
        if symbol in LEVERAGED_ETF_BLOCKLIST:
            self._quality_filter_rejects[symbol] = self._quality_filter_rejects.get(symbol, 0) + 1
            return False

        sym_df = self._quality_data_cache.get(symbol)
        spy_df = self._quality_data_cache.get('SPY')

        # If no price history, fall through (don't block).
        if sym_df is None or sym_df.empty:
            return True

        try:
            closes = sym_df['Close'] if 'Close' in sym_df.columns else None
            if closes is None or len(closes) < 50:
                return True  # not enough history; pass
            last_close = float(closes.iloc[-1])
            sma50 = float(closes.tail(50).mean())

            # Gate 2: trend filter
            if last_close < sma50:
                self._quality_filter_rejects[symbol] = self._quality_filter_rejects.get(symbol, 0) + 1
                return False

            # Gate 3: relative strength — only blocks CATASTROPHIC
            # underperformance (3+ percentage points worse than SPY).
            # Tier-1 names lagging SPY by 1-2pp should still pass —
            # they're noise, not "everyone's selling this name."
            # Threshold determined empirically on 5/22 data: MSFT was
            # 1.7pp under SPY but otherwise healthy → pass. SOXS was
            # 18pp under → catastrophic → correctly blocked.
            # v-rs-window-widen-2026-05-26: 5d → 10d window. The 5d
            # window over-amplifies single-session rotations. Mega-cap
            # names (NVDA, GOOG, META, MSFT, etc.) showed -4 to -6pp
            # vs SPY on the 5d window today during a broad-market
            # rotation, blowing through the -3pp gate. 10d window
            # smooths single-day moves; threshold loosening below
            # picks up where window-widening leaves off.
            if spy_df is not None and 'Close' in spy_df.columns and len(spy_df) >= 11:
                spy_closes = spy_df['Close']
                if len(closes) >= 11:
                    sym_ret = (last_close / float(closes.iloc[-11])) - 1.0
                    spy_ret = (float(spy_closes.iloc[-1]) / float(spy_closes.iloc[-11])) - 1.0
                    # v-rs-threshold-loosen-2026-05-26: -0.03 → -0.05.
                    # Today (2026-05-26) NVDA was -5.49pp vs SPY over
                    # the original 5d window — straight into the reject
                    # bucket while it appeared in yahoo_most_active. -3pp
                    # was calibrated 5/22 against MSFT (-1.7pp pass) and
                    # SOXS (-18pp block); -5pp keeps the SOXS block
                    # while admitting normal mega-cap pullbacks.
                    if (sym_ret - spy_ret) < -0.05:
                        self._quality_filter_rejects[symbol] = self._quality_filter_rejects.get(symbol, 0) + 1
                        return False
        except Exception as exc:
            logger.debug("screener: quality eval error for %s: %s", symbol, exc)
            return True  # don't block on eval error

        return True

    def _passes_mover_quality_filter(self, mover: Dict[str, Any]) -> bool:
        """v-mover-quality-relax-2026-09-10: relaxed quality filter for movers.
        
        For Yahoo day_gainers/day_losers/most_active, we bypass SMA50/RS
        filters because:
          1. Day-gainers ARE the momentum names — requiring them to also
             be above SMA50 is circular (they're moving BECAUSE something
             changed today)
          2. Quality-for-swing is wrong for intraday momentum
        
        Still enforced:
          - Leveraged ETF blocklist (hard block)
          - MIN_MOVER_PRICE floor ($5 default)
          - MIN_MOVER_VOLUME floor (500k default)
          - Spread filter (1% max, same as _is_tradeable)
        
        Returns True if mover passes the relaxed filter.
        """
        symbol = mover.get('symbol', '')
        
        # Gate 1: hard blocklist (always enforced)
        if symbol in LEVERAGED_ETF_BLOCKLIST:
            self._quality_filter_rejects[symbol] = self._quality_filter_rejects.get(symbol, 0) + 1
            logger.debug(
                "screener: mover_quality_relax blocked %s: leveraged_etf",
                symbol,
            )
            return False
        
        # Gate 2: price floor
        try:
            from core.config import Config as _CfgMQ
            _min_price = _CfgMQ().MIN_MOVER_PRICE
            _min_volume = _CfgMQ().MIN_MOVER_VOLUME
        except Exception:
            _min_price = 5.0
            _min_volume = 500000
        
        price = mover.get('last', 0)
        if price < _min_price:
            self._quality_filter_rejects[symbol] = self._quality_filter_rejects.get(symbol, 0) + 1
            logger.debug(
                "screener: mover_quality_relax blocked %s: price %.2f < %.2f",
                symbol, price, _min_price,
            )
            return False
        
        # Gate 3: volume floor
        volume = mover.get('volume', 0)
        if volume < _min_volume:
            self._quality_filter_rejects[symbol] = self._quality_filter_rejects.get(symbol, 0) + 1
            logger.debug(
                "screener: mover_quality_relax blocked %s: volume %d < %d",
                symbol, volume, _min_volume,
            )
            return False
        
        # Gate 4: spread check (same as _is_tradeable but logged)
        spread = mover.get('spread', 0)
        if price > 0 and spread / price > 0.01:
            self._quality_filter_rejects[symbol] = self._quality_filter_rejects.get(symbol, 0) + 1
            logger.debug(
                "screener: mover_quality_relax blocked %s: spread %.4f > 1%%",
                symbol, spread / price if price > 0 else 0,
            )
            return False
        
        # Passed all mover filters
        logger.debug(
            "screener: mover_quality_relax PASSED %s: price=%.2f vol=%d spread_pct=%.4f",
            symbol, price, volume, spread / price if price > 0 else 0,
        )
        return True

    def _is_tradeable(self, mover: Dict[str, Any]) -> bool:
        """Filter for tradeable stocks.

        v-watchlist-size-2026-04-29: relaxed thresholds so pre-market and
        low-vol periods don't shrink the watchlist to 5 symbols. Filters
        are still meaningful — just calibrated for actual intraday data
        rather than full-RTH peak liquidity.
        """
        # Volume filter — pre-market sees ~10% of regular-session volume,
        # so 250k is the equivalent gate for pre-market that 2.5M would
        # be at peak. Original 500k rejected most names before 09:30.
        if mover.get('volume', 0) < 250_000:
            return False

        # Price filter ($3-$1000 — wider than $5-$500. Lower bound catches
        # popular sub-$5 names (LCID, NIO sometimes); upper accommodates
        # NVDA/AVGO/SMCI without rejection.)
        price = mover.get('last', 0)
        if price < 3 or price > 1000:
            return False

        # Spread filter — was 0.5%, now 1% to allow mid-cap intraday.
        # In tight RTH spreads stay under 0.1%; pre-market 0.5-1% is normal.
        spread = mover.get('spread', 0)
        if price > 0 and spread / price > 0.01:
            return False

        # Exclude leveraged / inverse ETFs that don't behave like stocks
        # under our trade rules. SPY/QQQ/IWM are fine to trade if they
        # ever appear, but the legacy block-list leaves them out.
        symbol = mover.get('symbol', '')
        excluded = ('SPY', 'QQQ', 'IWM', 'DIA', 'VXX', 'UVXY',
                    'TQQQ', 'SQQQ', 'TSLL', 'NVDL', 'TMF', 'TLT')
        if symbol in excluded:
            return False

        return True

    def _calculate_mover_score(self, mover: Dict[str, Any]) -> float:
        """Calculate a score for ranking movers"""
        # Volatility component (40% weight)
        volatility_score = min(mover.get('volatility', 0) / 5, 1) * 0.4

        # Volume component (30% weight) - normalized by average
        volume = mover.get('volume', 0)
        volume_score = min(volume / 10_000_000, 1) * 0.3

        # Price change component (30% weight)
        change_score = min(abs(mover.get('percent_change', 0)) / 10, 1) * 0.3

        return volatility_score + volume_score + change_score

    def get_watchlist_symbols(self, limit: int = 20) -> List[str]:
        """Get current top mover symbols for the watchlist.

        v-watchlist-size-2026-04-29: previously hardcoded to 5 — too narrow,
        the bot would burn cycles on the same handful of names. Caller now
        passes Config.WATCHLIST_SIZE so the cap is centrally tunable.
        """
        return [mover['symbol'] for mover in self.top_movers[:limit]]
