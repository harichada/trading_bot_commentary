import asyncio
import logging
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import numpy as np
import feedparser
import aiohttp
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer

from core.models import CommentaryType, SignalType, NewsImpact, TradingSignal, NewsItem, MarketData
from core.commentary import TradingCommentary
from strategies.base import TradingStrategyWithCommentary
from strategies.builtin import _floored_atr

logger = logging.getLogger('TradingBot')


# v-async-safe-news-2026-04-29: yfinance and feedparser are SYNCHRONOUS.
# Calling them inside an async coroutine blocks the entire event loop —
# observed today after MRVL: Yahoo's API hung, the analysis loop froze
# for 7+ minutes, every other async task (WebSocket, position management)
# stalled until the bot was force-restarted.
#
# This helper runs any sync callable in the default thread executor
# with a hard timeout, so a hung external API can never block the loop.
async def _run_sync_with_timeout(fn, timeout: float = 6.0, default=None):
    """Run `fn()` in a worker thread, return its value or `default` on
    timeout / exception. Logs the timeout for observability."""
    try:
        loop = asyncio.get_event_loop()
        return await asyncio.wait_for(
            loop.run_in_executor(None, fn),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"sync_call_timeout fn={getattr(fn, '__name__', repr(fn))} after {timeout}s"
        )
        return default
    except Exception as exc:
        logger.debug(f"sync_call_error fn={fn}: {exc}")
        return default


class FreeNewsAggregator:
    """Aggregates news from free sources"""

    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes

        # Free RSS feeds
        self.rss_feeds = {
            'yahoo': "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US",
            'google': "https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en",
            'investing': "https://www.investing.com/rss/news.rss",
            'marketwatch': "https://feeds.marketwatch.com/marketwatch/topstories/"
        }

    async def fetch_news(self, symbol: str, hours: int = 24) -> List[NewsItem]:
        """Fetch news from all free sources"""
        all_news = []

        # Yahoo Finance (most reliable free source)
        yahoo_news = await self._fetch_yahoo_news(symbol, hours)
        all_news.extend(yahoo_news)

        # RSS Feeds
        rss_news = await self._fetch_rss_news(symbol, hours)
        all_news.extend(rss_news)

        # MarketWatch scraping
        marketwatch_news = await self._fetch_marketwatch_news(symbol, hours)
        all_news.extend(marketwatch_news)

        # Remove duplicates
        unique_news = {}
        for item in all_news:
            key = item.headline[:50]  # Use first 50 chars as key
            if key not in unique_news:
                unique_news[key] = item

        # Sort by relevance and time
        sorted_news = sorted(
            unique_news.values(),
            key=lambda x: (x.relevance_score, -x.age_hours()),
            reverse=True
        )

        return sorted_news[:20]  # Top 20 news items

    async def _fetch_yahoo_news(self, symbol: str, hours: int) -> List[NewsItem]:
        """Fetch from Yahoo Finance"""
        try:
            import yfinance as yf

            # v-async-safe-news-2026-04-29: yfinance is sync; wrap in
            # executor with hard timeout to prevent event-loop blocking
            # if Yahoo's API hangs (root cause of 11:13 MRVL freeze).
            def _yahoo_fetch():
                return yf.Ticker(symbol).news or []
            news_data = await _run_sync_with_timeout(_yahoo_fetch, timeout=6.0, default=[])

            news_items = []
            for article in news_data:
                # Check age
                pub_time = datetime.fromtimestamp(article.get('providerPublishTime', 0))
                if (datetime.now() - pub_time).total_seconds() / 3600 > hours:
                    continue

                # Create news item
                news_item = NewsItem(
                    id=hashlib.md5(article['link'].encode()).hexdigest()[:10],
                    symbol=symbol,
                    headline=article.get('title', ''),
                    summary=article.get('summary', ''),
                    source=article.get('publisher', 'Yahoo Finance'),
                    url=article.get('link', ''),
                    published_time=pub_time,
                    sentiment_score=0.0,  # Will be analyzed
                    sentiment_confidence=0.0,
                    impact=NewsImpact.MEDIUM,
                    relevance_score=0.8  # Yahoo pre-filters by ticker
                )

                news_items.append(news_item)

            # Also get RSS feed (feedparser also sync — wrap with timeout)
            rss_url = self.rss_feeds['yahoo'].format(symbol=symbol)
            feed = await _run_sync_with_timeout(
                lambda: feedparser.parse(rss_url),
                timeout=5.0,
                default=type('Empty', (), {'entries': []})(),
            )

            for entry in feed.entries[:10]:  # Last 10 entries
                # Parse time
                pub_time = datetime(*entry.published_parsed[:6])
                if (datetime.now() - pub_time).total_seconds() / 3600 > hours:
                    continue

                news_item = NewsItem(
                    id=hashlib.md5(entry.link.encode()).hexdigest()[:10],
                    symbol=symbol,
                    headline=entry.title,
                    summary=entry.get('summary', ''),
                    source='Yahoo Finance RSS',
                    url=entry.link,
                    published_time=pub_time,
                    sentiment_score=0.0,
                    sentiment_confidence=0.0,
                    impact=NewsImpact.MEDIUM,
                    relevance_score=0.8
                )

                news_items.append(news_item)

            return news_items

        except Exception as e:
            logger.error(f"Yahoo news error for {symbol}: {e}")
            return []

    async def _fetch_rss_news(self, symbol: str, hours: int) -> List[NewsItem]:
        """Fetch from various RSS feeds"""
        news_items = []

        # Google News RSS (feedparser is sync — wrap with timeout)
        try:
            google_url = self.rss_feeds['google'].format(symbol=symbol)
            feed = await _run_sync_with_timeout(
                lambda: feedparser.parse(google_url),
                timeout=5.0,
                default=type('Empty', (), {'entries': []})(),
            )

            for entry in feed.entries[:5]:  # Top 5 from Google
                # Extract actual source from Google News title
                title = entry.title
                source = 'Google News'
                if ' - ' in title:
                    parts = title.rsplit(' - ', 1)
                    title = parts[0]
                    source = parts[1]

                # Parse time
                pub_time = datetime(*entry.published_parsed[:6])
                if (datetime.now() - pub_time).total_seconds() / 3600 > hours:
                    continue

                # Check relevance
                if symbol.upper() not in title.upper():
                    continue

                news_item = NewsItem(
                    id=hashlib.md5(entry.link.encode()).hexdigest()[:10],
                    symbol=symbol,
                    headline=title,
                    summary=entry.get('summary', '')[:200],
                    source=source,
                    url=entry.link,
                    published_time=pub_time,
                    sentiment_score=0.0,
                    sentiment_confidence=0.0,
                    impact=NewsImpact.MEDIUM,
                    relevance_score=0.6
                )

                news_items.append(news_item)

        except Exception as e:
            logger.debug(f"Google News error: {e}")

        return news_items

    async def _fetch_marketwatch_news(self, symbol: str, hours: int) -> List[NewsItem]:
        """Scrape news from MarketWatch"""
        news_items = []

        try:
            # MarketWatch URLs
            urls = [
                f"https://www.marketwatch.com/investing/stock/{symbol}",
                f"https://www.marketwatch.com/investing/stock/{symbol}/news"
            ]

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            timeout = aiohttp.ClientTimeout(total=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                for url in urls:
                    try:
                        async with session.get(url, headers=headers) as response:
                            if response.status != 200:
                                continue

                            html = await response.text()
                            soup = BeautifulSoup(html, 'html.parser')

                            # Find news articles
                            articles = soup.find_all('div', class_='article__content')
                            if not articles:
                                # Try alternative selectors
                                articles = soup.find_all('div', {'class': re.compile('element--article')})

                            for article in articles[:10]:  # Get up to 10 articles
                                try:
                                    # Extract headline
                                    headline_elem = article.find(['h3', 'h2', 'a'], class_=re.compile('headline|title'))
                                    if not headline_elem:
                                        continue

                                    headline = headline_elem.get_text(strip=True)

                                    # Extract link
                                    link_elem = article.find('a', href=True)
                                    if link_elem:
                                        link = link_elem['href']
                                        if not link.startswith('http'):
                                            link = f"https://www.marketwatch.com{link}"
                                    else:
                                        continue

                                    # Extract time
                                    time_elem = article.find(['span', 'time'], class_=re.compile('time|date|timestamp'))
                                    if time_elem:
                                        time_text = time_elem.get_text(strip=True)
                                        pub_time = self._parse_time(time_text)
                                    else:
                                        pub_time = datetime.now()

                                    # Check age
                                    if (datetime.now() - pub_time).total_seconds() / 3600 > hours:
                                        continue

                                    # Extract summary
                                    summary_elem = article.find(['p', 'div'], class_=re.compile('summary|teaser|deck'))
                                    summary = summary_elem.get_text(strip=True) if summary_elem else ''

                                    # Check relevance
                                    if symbol.upper() not in headline.upper() and symbol.upper() not in summary.upper():
                                        relevance = 0.3
                                    else:
                                        relevance = 0.7

                                    news_item = NewsItem(
                                        id=hashlib.md5(link.encode()).hexdigest()[:10],
                                        symbol=symbol,
                                        headline=headline,
                                        summary=summary[:200],
                                        source='MarketWatch',
                                        url=link,
                                        published_time=pub_time,
                                        sentiment_score=0.0,
                                        sentiment_confidence=0.0,
                                        impact=NewsImpact.MEDIUM,
                                        relevance_score=relevance
                                    )

                                    news_items.append(news_item)

                                except Exception as e:
                                    logger.debug(f"Error parsing article: {e}")
                                    continue

                    except Exception as e:
                        logger.debug(f"MarketWatch scraping error for {url}: {e}")

        except Exception as e:
            logger.error(f"MarketWatch general error: {e}")

        return news_items

    def _parse_time(self, time_text: str) -> datetime:
        """Parse various time formats from MarketWatch"""
        try:
            time_text = time_text.strip()

            # Handle relative times
            if 'ago' in time_text:
                if 'minute' in time_text:
                    minutes = int(re.search(r'(\d+)', time_text).group(1))
                    return datetime.now() - timedelta(minutes=minutes)
                elif 'hour' in time_text:
                    hours = int(re.search(r'(\d+)', time_text).group(1))
                    return datetime.now() - timedelta(hours=hours)
                elif 'day' in time_text:
                    days = int(re.search(r'(\d+)', time_text).group(1))
                    return datetime.now() - timedelta(days=days)

            # Handle "Today" or "Yesterday"
            if 'today' in time_text.lower():
                return datetime.now()
            elif 'yesterday' in time_text.lower():
                return datetime.now() - timedelta(days=1)

            # Try parsing absolute dates
            for fmt in ['%b. %d, %Y', '%B %d, %Y', '%m/%d/%Y']:
                try:
                    return datetime.strptime(time_text, fmt)
                except ValueError:
                    continue

        except (ValueError, AttributeError) as e:
            logger.debug(f"Could not parse time text: {e}")

        return datetime.now()  # Default to now


class FreeNewsSignalStrategy(TradingStrategyWithCommentary):
    """News strategy using only free sources"""
    from nltk.sentiment import SentimentIntensityAnalyzer

    name = "news"

    def __init__(self, commentary_system):
        super().__init__(commentary_system)
        self.aggregator = FreeNewsAggregator()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()  # VADER only
        self.last_signal_time = {}
        self.profiles = {}
        # v-news-verify-2026-04-28: fresh-news re-verification gate
        from analysis.news_verifier import NewsVerifier
        # v-news-verify-window-2026-04-28: widened freshness window 30min → 4h.
        # 30-min window vetoed every signal during pre-market / lunch lulls when
        # no new articles are published despite the original news still being
        # the active catalyst. 4-hour window is intraday-realistic — captures
        # morning news for an afternoon trade and vice versa.
        self.verifier = NewsVerifier(
            sentiment_analyzer=self.sentiment_analyzer,
            freshness_minutes=240,
            min_fresh_articles=2,
            min_match_strength=0.10,
        )

    async def generate_signal_with_commentary(self, market_data) -> Optional[TradingSignal]:
        """Generate signal from free news sources"""
        symbol = market_data.symbol

        # Check cooldown
        if symbol in self.last_signal_time:
            # v-news-cooldown-shorten-2026-05-08: 60min → 20min. The 60-min
            # cooldown locked out add-on opportunities after legitimate
            # entries (INTC 2026-05-08 12:53 entry → no re-engagement
            # for 60 minutes while INTC moved from $124.75 to $127 then
            # back). 20 minutes is enough to prevent same-bar churn but
            # short enough to react to developing news on a moving name.
            cooldown_left = 1200 - (market_data.timestamp - self.last_signal_time[symbol]).total_seconds()  # v-determinism-2026-05-19
            if cooldown_left > 0:
                self._log_decision(market_data, "skip", "cooldown",
                                   cooldown_remaining_s=int(cooldown_left))
                return None

        # Fetch news
        news_items = await self.aggregator.fetch_news(symbol, 24)

        # v-news-verify-2026-04-28: minimum 5 articles (was 3) — primary gate
        # for upstream signal quality. Fresh-news verification later catches
        # the "stale news" failure mode (PLTR/TSLA 2026-04-28 entries faded
        # within 35 min because cached news was already priced in).
        if len(news_items) < 5:
            self._log_decision(market_data, "skip", "insufficient_news",
                               articles=len(news_items))
            return None

        # Analyze sentiment
        sentiments = []
        high_impact_news = []

        for item in news_items:
            # Simple sentiment analysis
            text = f"{item.headline} {item.summary}"
            scores = self.sentiment_analyzer.polarity_scores(text)
            item.sentiment_score = scores['compound']
            item.sentiment_confidence = abs(scores['compound'])

            # Detect high impact
            if any(keyword in text.lower() for keyword in ['earnings', 'beat', 'miss', 'sec', 'investigation']):
                item.impact = NewsImpact.HIGH
                high_impact_news.append(item)

            sentiments.append(item.sentiment_score)

        # Calculate average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments)

        # v-news-verify-2026-04-28: tightened sentiment threshold 0.3 → 0.4.
        # Compound 0.3 captured a lot of marginal sentiment that didn't move
        # price (37 trades over 5 days at PF 0.58). 0.4 cuts volume ~40% and
        # leaves only meaningfully bullish/bearish reads.
        #
        # v-news-min-strength-2026-04-29: require BOTH a minimum sentiment
        # magnitude AND (strong sentiment OR impact keyword). The previous
        # `or high_impact_news` arm let weak signals through: QCOM
        # (avg=0.012) and HOOD (avg=0.065) both got strength≈0 entries
        # because one article happened to contain "earnings"/"sec". Those
        # near-zero-strength positions then got Kelly-floored sizing and
        # bled out on noise. Floor at 0.20 keeps the impact signal but
        # cuts the noise floor.
        # v-sentiment-thresh-2026-05-05: 0.40 → 0.30 when ≥5 articles
        # corroborate. 509/3000 recent skips were weak_sentiment, mostly
        # in the 0.25–0.40 band where article count was already strong.
        # Falling-knife / rising-knife trend filters downstream still
        # block bad-trend traps. 0.40 stays as the floor when articles<5.
        _strong_corroborated = (abs(avg_sentiment) > 0.30 and len(news_items) >= 5)
        _strong_uncorroborated = (abs(avg_sentiment) > 0.40)
        if _strong_corroborated or _strong_uncorroborated or (high_impact_news and abs(avg_sentiment) >= 0.20):
            signal_type = SignalType.BUY if avg_sentiment > 0 else SignalType.SELL
            confidence = min(abs(avg_sentiment) + 0.3, 0.8)

            # Trend filter: don't buy positive news into a confirmed downtrend.
            # Same LCID-style "falling knife" pattern the mean-rev strategy had.
            # Keep symmetric short side open since short-of-uptrend-on-bad-news
            # is a legitimate fade setup (exhaustion, reversal).
            # Compute trend context once — both BUY and SELL guards use it.
            indicators = market_data.indicators or {}
            try:
                sma_50 = float(indicators.get("sma_50", 0))
                macd_val = float(indicators.get("macd", 0))
                macd_sig = float(indicators.get("macd_signal", 0))
            except (TypeError, ValueError):
                sma_50, macd_val, macd_sig = 0.0, 0.0, 0.0

            # v-conviction-bypass-knife-2026-05-06: when sentiment is
            # extreme AND well-corroborated, the falling/rising-knife
            # guards become the wrong call. They block the very V-bottom
            # / V-top reversals the news strategy is supposed to catch
            # — MSFT 2026-05-06 09:27–09:31 ET fired sentiment 0.642 →
            # 0.707 with 10 articles at the bottom ($406.64), trend
            # filter killed each tick, bot eventually entered at $417.65
            # after MACD turned. The MACD/SMA50 are lagging trend signals
            # while sentiment > 0.55 with ≥8 articles is a leading
            # catalyst signal. Trust the catalyst.
            _conviction_bypass = (
                abs(avg_sentiment) >= 0.55 and len(news_items) >= 8
            )
            _knife_would_block_buy = (
                signal_type == SignalType.BUY
                and sma_50 > 0
                and market_data.close < sma_50
                and macd_val < macd_sig
            )
            _knife_would_block_sell = (
                signal_type == SignalType.SELL
                and sma_50 > 0
                and market_data.close > sma_50
                and macd_val > macd_sig
            )
            if _conviction_bypass and (_knife_would_block_buy or _knife_would_block_sell):
                self._log_decision(
                    market_data, "bypass", "conviction_overrides_knife",
                    sentiment=round(avg_sentiment, 3),
                    articles=len(news_items),
                    sma_50=round(sma_50, 2),
                    macd=round(macd_val, 4),
                    side=("buy" if signal_type == SignalType.BUY else "sell"),
                )

            if signal_type == SignalType.BUY:
                if (sma_50 > 0 and market_data.close < sma_50
                        and macd_val < macd_sig
                        and not _conviction_bypass):
                    self._log_decision(
                        market_data, "skip", "falling_knife_news_buy",
                        sentiment=round(avg_sentiment, 3),
                        sma_50=round(sma_50, 2),
                        macd=round(macd_val, 4),
                        articles=len(news_items),
                    )
                    # v-news-veto-tracker-2026-04-28: record this veto for
                    # post-hoc evaluation. Compute would-be stop/target the
                    # same way the strategy WOULD have on a non-vetoed signal.
                    try:
                        from core.config import Config as _Cfg
                        _atr = _floored_atr(market_data.indicators.get('atr', market_data.close * 0.02), market_data.close)
                        _stop_dist = _Cfg().ATR_STOP_MULTIPLIER * _atr
                        _rr = _Cfg().ATR_REWARD_RISK_RATIO
                        wb_stop = market_data.close - _stop_dist
                        wb_target = market_data.close + (_rr * _stop_dist)
                        # We have access to the engine via commentary if needed
                        # but db_logger lives on engine; emit through brain as a
                        # fire-and-forget through the engine if available.
                        engine = getattr(self.commentary, 'engine_ref', None)
                        if engine is not None and getattr(engine, 'db_logger', None):
                            import asyncio as _asyncio
                            _asyncio.get_event_loop().create_task(
                                engine.db_logger.log_news_veto(
                                    symbol=symbol, side='long',
                                    veto_reason='falling_knife_news_buy',
                                    veto_source='falling_knife',
                                    cached_sentiment=float(avg_sentiment),
                                    would_entry_price=float(market_data.close),
                                    would_stop_loss=float(wb_stop),
                                    would_take_profit=float(wb_target),
                                )
                            )
                    except Exception as _e:
                        logger.debug(f"news veto tracker (falling_knife) skipped: {_e}")
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=symbol,
                        title=f"⛔ News BUY Skipped — Falling Knife",
                        message=(f"Positive news sentiment ({avg_sentiment:+.2f}) "
                                 f"but price below MA50 and MACD bearish. "
                                 "Avoiding counter-trend news entry."),
                        data={"sentiment": avg_sentiment, "sma_50": sma_50,
                              "macd": macd_val, "macd_signal": macd_sig},
                        importance=6,
                    ))
                    return None

            # v-rising-knife-2026-05-01: symmetric guard for SHORT entries.
            # The cost of NOT having this: RIOT 2026-05-01 — bot shorted at
            # $18.75 on negative news (sentiment −0.579) AFTER price had
            # spiked from $18.05 to $19.20 in 25 minutes (uptrend confirmed:
            # close > sma_50, MACD bullish, RSI 88). Stopped out at $19.15
            # for −$218. The mean_reversion strategy refused to short on
            # the same setup ('short_disabled') but news_strategy had no
            # equivalent guard and fired anyway. This is the rising-knife
            # mirror of falling_knife: don't fade momentum on a counter-
            # trend news signal.
            if signal_type == SignalType.SELL:
                if (sma_50 > 0 and market_data.close > sma_50
                        and macd_val > macd_sig
                        and not _conviction_bypass):
                    self._log_decision(
                        market_data, "skip", "rising_knife_news_sell",
                        sentiment=round(avg_sentiment, 3),
                        sma_50=round(sma_50, 2),
                        macd=round(macd_val, 4),
                        articles=len(news_items),
                    )
                    # Same shadow-tracker write so post-hoc analysis can
                    # tell us whether the rising-knife veto saved a loss
                    # or missed a winner.
                    try:
                        from core.config import Config as _Cfg
                        _atr = _floored_atr(
                            market_data.indicators.get('atr', market_data.close * 0.02),
                            market_data.close,
                        )
                        _stop_dist = _Cfg().ATR_STOP_MULTIPLIER * _atr
                        _rr = _Cfg().ATR_REWARD_RISK_RATIO
                        # Mirrored: short stop is ABOVE entry, target BELOW
                        wb_stop = market_data.close + _stop_dist
                        wb_target = market_data.close - (_rr * _stop_dist)
                        engine = getattr(self.commentary, 'engine_ref', None)
                        if engine is not None and getattr(engine, 'db_logger', None):
                            import asyncio as _asyncio
                            _asyncio.get_event_loop().create_task(
                                engine.db_logger.log_news_veto(
                                    symbol=symbol, side='short',
                                    veto_reason='rising_knife_news_sell',
                                    veto_source='rising_knife',
                                    cached_sentiment=float(avg_sentiment),
                                    would_entry_price=float(market_data.close),
                                    would_stop_loss=float(wb_stop),
                                    would_take_profit=float(wb_target),
                                )
                            )
                    except Exception as _e:
                        logger.debug(f"news veto tracker (rising_knife) skipped: {_e}")
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=symbol,
                        title=f"⛔ News SELL Skipped — Rising Knife",
                        message=(f"Negative news sentiment ({avg_sentiment:+.2f}) "
                                 f"but price above MA50 and MACD bullish. "
                                 "Refusing to short into a confirmed uptrend."),
                        data={"sentiment": avg_sentiment, "sma_50": sma_50,
                              "macd": macd_val, "macd_signal": macd_sig},
                        importance=6,
                    ))
                    return None

            # v-news-price-direction-gate-2026-05-08: require the latest
            # bar's price action to AGREE with the news direction before
            # placing the trade. Pure-sentiment entries got bot into
            # PINS / TSLA / META "buy the falling knife" trades multiple
            # times — sentiment was positive (cached, multi-hour rolling
            # average) but the live tape was making lower highs.
            #
            # Gate:
            #   BUY  → require close > open AND volume_ratio >= 1.2
            #   SELL → require close < open AND volume_ratio >= 1.2
            # The volume_ratio floor (current bar volume vs avg) ensures
            # we only act on bars with real participation, not noise.
            #
            # Conviction-bypass override: same as the falling/rising-knife
            # filter — when sentiment is extreme (≥0.55, articles ≥8) we
            # trust the catalyst signal even if the latest bar hasn't
            # confirmed yet. The MSFT 09:27–09:31 V-bottom that we want
            # to catch will sometimes have the catalyst arriving DURING
            # the down-bar and we'd miss it otherwise.
            try:
                bar_open = float(getattr(market_data, "open", market_data.close) or market_data.close)
            except Exception:
                bar_open = market_data.close
            volume_ratio = float(indicators.get("volume_ratio", 1.0) or 1.0)
            # v-price-direction-relax-2026-05-08: AMD 2026-05-08 14:46–14:50
            # ran $444→$448 (+$4 over an hour) but bot blocked every entry
            # because individual 5-min bars closed slightly red on
            # consolidation pullbacks within an obvious uptrend. The
            # original gate was too literal (current bar must be green).
            #
            # Relaxed semantics:
            #   - **Strong sentiment ≥0.50**: skip the gate entirely. The
            #     news IS the catalyst; demanding bar-level confirmation
            #     on top of strong news is overgated.
            #   - **0.35–0.50**: require close within 0.2% of open
            #     (`>= bar_open * 0.998`) AND volume_ratio >= 1.0.
            #     Permits normal in-bar consolidation while still
            #     requiring real participation.
            #   - **< 0.35**: keep the strict original gate
            #     (`close > bar_open` AND vol_ratio >= 1.2). Weak
            #     sentiment needs every confirmation it can get.
            _abs_sent = abs(avg_sentiment)
            if _abs_sent >= 0.50:
                _bar_floor_factor_buy = 0.0  # any close passes
                _bar_ceil_factor_sell = float("inf")
                _vol_floor = 0.0
            elif _abs_sent >= 0.35:
                _bar_floor_factor_buy = 0.998   # close >= open * 0.998
                _bar_ceil_factor_sell = 1.002   # close <= open * 1.002
                _vol_floor = 1.0
            else:
                _bar_floor_factor_buy = 1.0     # close > open
                _bar_ceil_factor_sell = 1.0     # close < open
                _vol_floor = 1.2
            _MIN_VOL_RATIO = _vol_floor  # kept for log-line readability
            if not _conviction_bypass and _abs_sent < 0.50:
                if signal_type == SignalType.BUY:
                    if not (market_data.close >= bar_open * _bar_floor_factor_buy
                            and volume_ratio >= _vol_floor):
                        self._log_decision(
                            market_data, "skip", "price_direction_disagrees_buy",
                            sentiment=round(avg_sentiment, 3),
                            close=round(market_data.close, 2),
                            bar_open=round(bar_open, 2),
                            volume_ratio=round(volume_ratio, 2),
                            articles=len(news_items),
                        )
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.RISK_ASSESSMENT,
                            symbol=symbol,
                            title=f"⛔ News BUY Skipped — Tape Doesn't Confirm",
                            message=(
                                f"Positive news sentiment ({avg_sentiment:+.2f}) "
                                f"but the latest bar is "
                                f"{'down' if market_data.close < bar_open else 'flat'} "
                                f"({bar_open:.2f}→{market_data.close:.2f}) and "
                                f"volume ratio {volume_ratio:.2f} is below {_MIN_VOL_RATIO}. "
                                "Waiting for the tape to confirm the news direction."
                            ),
                            data={
                                "sentiment": avg_sentiment,
                                "open": bar_open,
                                "close": market_data.close,
                                "volume_ratio": volume_ratio,
                            },
                            importance=6,
                        ))
                        return None
                else:  # SELL
                    if not (market_data.close <= bar_open * _bar_ceil_factor_sell
                            and volume_ratio >= _vol_floor):
                        self._log_decision(
                            market_data, "skip", "price_direction_disagrees_sell",
                            sentiment=round(avg_sentiment, 3),
                            close=round(market_data.close, 2),
                            bar_open=round(bar_open, 2),
                            volume_ratio=round(volume_ratio, 2),
                            articles=len(news_items),
                        )
                        self.commentary.add_commentary(TradingCommentary(
                            timestamp=datetime.now(),
                            type=CommentaryType.RISK_ASSESSMENT,
                            symbol=symbol,
                            title=f"⛔ News SELL Skipped — Tape Doesn't Confirm",
                            message=(
                                f"Negative news sentiment ({avg_sentiment:+.2f}) "
                                f"but the latest bar is "
                                f"{'up' if market_data.close > bar_open else 'flat'} "
                                f"({bar_open:.2f}→{market_data.close:.2f}) and "
                                f"volume ratio {volume_ratio:.2f} is below {_MIN_VOL_RATIO}. "
                                "Waiting for the tape to confirm the news direction."
                            ),
                            data={
                                "sentiment": avg_sentiment,
                                "open": bar_open,
                                "close": market_data.close,
                                "volume_ratio": volume_ratio,
                            },
                            importance=6,
                        ))
                        return None

            # v-news-late-entry-guard-2026-05-08: refuse entries where the
            # move has already happened. Operator's observation 2026-05-08:
            # the bot kept buying tops — INTC entered $124.75 on 148-min-
            # old news with close 11% above bb_lower, MSFT yesterday
            # entered at bb_upper (RSI 83), INTC yesterday at $109 with
            # RSI 86–94. Pure-sentiment entries with no overextension
            # check confirm the late-entry pattern.
            #
            # Gate (no conviction-bypass — overextension is risk regardless
            # of sentiment magnitude; the V-bottom case fires at LOW RSI
            # with high sentiment, which this gate doesn't block):
            #
            # v-late-entry-tighten-2026-05-11: NVDA 2026-05-11 09:54 bought
            # at $218.34 with RSI 75.48 and ~1.77% above sma_20. Original
            # 2%-SMA threshold + RSI>70 AND-combo missed it by 23 bps.
            # Tightened to 1% (was 2%). NVDA at 1.77% above + RSI 75 now
            # triggers the gate. LUNR earlier (RSI 84, 3.22% above SMA20)
            # also still triggers — was 64% above the old threshold,
            # now 222% above.
            #
            # Why not OR-of-two: pure RSI>70 in a sustained trend is
            # legitimate continuation; pure 1%-above-SMA on quiet stocks
            # is noise. The combination is the late-entry pattern.
            try:
                rsi_now = float(indicators.get("rsi", 50) or 50)
                sma_20 = float(indicators.get("sma_20", 0) or 0)
            except Exception:
                rsi_now, sma_20 = 50.0, 0.0
            if signal_type == SignalType.BUY:
                if (
                    rsi_now > 70.0
                    and sma_20 > 0
                    and market_data.close > sma_20 * 1.01
                ):
                    pct_above = ((market_data.close / sma_20) - 1.0) * 100.0
                    self._log_decision(
                        market_data, "skip", "extended_already_ran",
                        sentiment=round(avg_sentiment, 3),
                        rsi=round(rsi_now, 2),
                        close=round(market_data.close, 2),
                        sma_20=round(sma_20, 2),
                        pct_above_sma20=round(pct_above, 2),
                        articles=len(news_items),
                    )
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=symbol,
                        title=f"⛔ News BUY Skipped — Already Extended",
                        message=(
                            f"Positive news sentiment ({avg_sentiment:+.2f}) "
                            f"but RSI {rsi_now:.0f} and price is "
                            f"{pct_above:.1f}% above SMA20 (${sma_20:.2f}). "
                            "The move has already happened — chasing here "
                            "is buying the top. Waiting for a pullback or "
                            "skipping this catalyst."
                        ),
                        data={
                            "sentiment": avg_sentiment,
                            "rsi": rsi_now,
                            "close": market_data.close,
                            "sma_20": sma_20,
                            "pct_above_sma20": pct_above,
                        },
                        importance=7,
                    ))
                    return None
            else:  # SELL
                # v-late-entry-tighten-2026-05-11: symmetric tightening
                # 0.98 → 0.99 (1% below SMA, was 2%).
                if (
                    rsi_now < 30.0
                    and sma_20 > 0
                    and market_data.close < sma_20 * 0.99
                ):
                    pct_below = (1.0 - (market_data.close / sma_20)) * 100.0
                    self._log_decision(
                        market_data, "skip", "extended_already_dropped",
                        sentiment=round(avg_sentiment, 3),
                        rsi=round(rsi_now, 2),
                        close=round(market_data.close, 2),
                        sma_20=round(sma_20, 2),
                        pct_below_sma20=round(pct_below, 2),
                        articles=len(news_items),
                    )
                    self.commentary.add_commentary(TradingCommentary(
                        timestamp=datetime.now(),
                        type=CommentaryType.RISK_ASSESSMENT,
                        symbol=symbol,
                        title=f"⛔ News SELL Skipped — Already Extended",
                        message=(
                            f"Negative news sentiment ({avg_sentiment:+.2f}) "
                            f"but RSI {rsi_now:.0f} and price is "
                            f"{pct_below:.1f}% below SMA20 (${sma_20:.2f}). "
                            "The drop has already happened — shorting here "
                            "is selling the bottom. Waiting for a bounce or "
                            "skipping this catalyst."
                        ),
                        data={
                            "sentiment": avg_sentiment,
                            "rsi": rsi_now,
                            "close": market_data.close,
                            "sma_20": sma_20,
                            "pct_below_sma20": pct_below,
                        },
                        importance=7,
                    ))
                    return None

            # v-news-verifier-toggle-2026-04-29: gate verifier behind config.
            # When disabled, news_strategy fires on cached sentiment alone
            # (original behaviour pre-v-news-verify-2026-04-28).
            #
            # v-news-verifier-advisory-2026-05-08: NEW advisory mode runs
            # the verifier purely for observability. The verdict is
            # logged but does not gate. When advisory is on AND the hard
            # gate is off, the verifier still executes and we log
            # `news_verifier_advisory action=advisory reason=<verdict>`
            # for every news entry — over a week we can grade whether
            # its vetoes would have improved P&L, then re-enable hard
            # gating with evidence.
            from core.config import Config as _CfgNV
            _verifier_enabled = _CfgNV().ENABLE_NEWS_VERIFIER
            _verifier_advisory = (
                (not _verifier_enabled) and _CfgNV().NEWS_VERIFIER_ADVISORY
            )
            _verifier_runs = _verifier_enabled or _verifier_advisory

            # Detection commentary — wording depends on whether verifier
            # will run. Honest about the two-stage process either way.
            _verify_msg = (
                "\n\nRe-verifying with fresh news from Alpaca/Yahoo before entering..."
                if _verifier_enabled else
                "\n\nVerifier disabled — proceeding to entry on cached sentiment."
            )
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.OPPORTUNITY,
                symbol=symbol,
                title=(f"📰 News Signal Detected: {signal_type.name} {symbol}"
                       + (" — Verifying" if _verifier_enabled else "")),
                message=(
                    f"Cached sentiment: {'Positive' if avg_sentiment > 0 else 'Negative'} ({avg_sentiment:+.2f})\n"
                    f"Articles: {len(news_items)}\n\n"
                    "Headlines:\n" +
                    "\n".join([f"• {item.headline}" for item in news_items[:3]]) +
                    _verify_msg
                ),
                data={
                    'sentiment': avg_sentiment,
                    'news_count': len(news_items),
                    'high_impact': len(high_impact_news) > 0,
                    'sources': list(set(item.source for item in news_items[:5]))
                },
                confidence=confidence,
                importance=7,
            ))

            # ATR-scaled stops
            from core.config import Config
            atr = _floored_atr(market_data.indicators.get('atr', market_data.close * 0.02), market_data.close)
            atr_mult = Config().ATR_STOP_MULTIPLIER
            rr_ratio = Config().ATR_REWARD_RISK_RATIO
            stop_distance = atr_mult * atr

            if signal_type == SignalType.BUY:
                stop_loss = market_data.close - stop_distance
                take_profit = market_data.close + (rr_ratio * stop_distance)
            else:
                stop_loss = market_data.close + stop_distance
                take_profit = market_data.close - (rr_ratio * stop_distance)

            # v-news-verify-2026-04-28: re-verify with fresh news from
            # Alpaca (or yfinance fallback) before committing to the trade.
            # Cached RSS news is often hours old and already priced in;
            # require ≥2 fresh articles in last 4 hours with sentiment in
            # the same direction. Skip cleanly if not verified.
            #
            # v-news-verifier-toggle-2026-04-29: when disabled, skip the
            # entire verifier block and treat the signal as already
            # confirmed. The shadow-tracker also doesn't fire (no veto
            # to track) — that data resumes when verifier is re-enabled.
            v = None
            if _verifier_runs:
                try:
                    expected_dir = 1 if signal_type == SignalType.BUY else -1
                    v = await self.verifier.verify(symbol, expected_dir)
                except Exception as exc:
                    # Verifier failure is non-fatal — log and treat as a skip
                    # rather than blindly trusting cached sentiment WHEN
                    # the verifier is in gating mode. In advisory mode,
                    # treat verifier failure as "no advisory data" and
                    # let the signal pass through.
                    logger.debug(f"news verifier exception for {symbol}: {exc}")
                    if _verifier_enabled:
                        self._log_decision(
                            market_data, "skip", "verifier_error",
                            err=str(exc)[:80], sentiment=round(avg_sentiment, 3),
                        )
                        return None
                    else:
                        self._log_decision(
                            market_data, "advisory", "verifier_error",
                            err=str(exc)[:80], sentiment=round(avg_sentiment, 3),
                        )
                        v = None

            # v-news-verifier-advisory-2026-05-08: when advisory mode is
            # on, log every verifier verdict (pass OR fail) for the
            # signal that's about to fire. We log but never veto.
            if _verifier_advisory and v is not None:
                self._log_decision(
                    market_data,
                    "advisory",
                    ("would_pass" if v.is_verified else "would_veto"),
                    cached_sentiment=round(avg_sentiment, 3),
                    fresh_count=v.fresh_count,
                    fresh_avg=round(v.avg_fresh_sentiment, 3),
                    source=v.source,
                    verifier_reason=v.reason,
                    latest_age_min=(
                        round(v.latest_age_min, 1)
                        if v.latest_age_min is not None else None
                    ),
                )

            # v-news-verifier-gate-zero-fresh-2026-05-13: even in
            # advisory mode, fresh_count == 0 is a hard veto. Zero
            # fresh articles means the bot is acting on cached
            # sentiment from articles that have aged out of any
            # reasonable freshness window. Operator incident 2026-05-12:
            # FCEL entered 3 times in 1 hour, each with fresh_count=0
            # and the verifier flagging insufficient_fresh_articles_0_lt_2.
            # Advisory mode was correct for the fresh_count==1 case
            # (rare but possible early news) but wrong for zero.
            # ≥1 fresh keeps the advisory behavior unchanged.
            if v is not None and v.fresh_count == 0:
                self._log_decision(
                    market_data,
                    "skip",
                    "zero_fresh_articles_hard_veto",
                    cached_sentiment=round(avg_sentiment, 3),
                    fresh_count=v.fresh_count,
                    source=v.source,
                    verifier_reason=v.reason,
                )
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.RISK_ASSESSMENT,
                    symbol=symbol,
                    title=f"⛔ {symbol}: Stale News Hard-Vetoed",
                    message=(
                        f"Cached sentiment {avg_sentiment:+.2f} but ZERO fresh "
                        f"articles in the verifier window — refusing to trade on "
                        f"stale data regardless of advisory mode."
                    ),
                    importance=7,
                ))
                return None

            if _verifier_enabled and v is not None and not v.is_verified:
                self._log_decision(
                    market_data, "skip", "fresh_news_unverified",
                    cached_sentiment=round(avg_sentiment, 3),
                    fresh_count=v.fresh_count,
                    fresh_avg=round(v.avg_fresh_sentiment, 3),
                    source=v.source,
                    verifier_reason=v.reason,  # renamed from 'reason' to avoid clobbering positional arg
                    latest_age_min=(round(v.latest_age_min, 1)
                                    if v.latest_age_min is not None else None),
                )
                # v-news-veto-tracker-2026-04-28: persist the would-be trade
                # for later evaluation (correct_veto vs missed_winner).
                try:
                    engine = getattr(self.commentary, 'engine_ref', None)
                    if engine is not None and getattr(engine, 'db_logger', None):
                        import asyncio as _asyncio
                        _asyncio.get_event_loop().create_task(
                            engine.db_logger.log_news_veto(
                                symbol=symbol,
                                side=('long' if signal_type == SignalType.BUY else 'short'),
                                veto_reason=v.reason,
                                veto_source=v.source,
                                cached_sentiment=float(avg_sentiment),
                                fresh_count=v.fresh_count,
                                fresh_avg_sentiment=float(v.avg_fresh_sentiment),
                                latest_age_min=(float(v.latest_age_min)
                                                if v.latest_age_min is not None else None),
                                would_entry_price=float(market_data.close),
                                would_stop_loss=float(stop_loss),
                                would_take_profit=float(take_profit),
                            )
                        )
                except Exception as _e:
                    logger.debug(f"news veto tracker (fresh_news_unverified) skipped: {_e}")
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.RISK_ASSESSMENT,
                    symbol=symbol,
                    title=f"⛔ News Signal Vetoed — Fresh Check Failed",
                    message=(
                        f"Cached sentiment was {avg_sentiment:+.2f} ({signal_type.name}), "
                        f"but {v.source} verification "
                        f"({v.fresh_count} fresh articles, avg {v.avg_fresh_sentiment:+.2f}) "
                        f"failed: {v.reason}. News is likely stale or already priced in."
                    ),
                    importance=7,
                ))
                return None

            # Verified (or verifier disabled) — announce in commentary so the
            # user can see WHY this signal made it past the gate.
            if v is not None:
                # v-verifier-commentary-none-guard-2026-05-08: latest_age_min
                # can be None (e.g. yfinance source returns no age) AND
                # avg_fresh_sentiment can be None when fresh_count==0.
                # Pre-T5 the verifier was disabled by default so this
                # block never ran; advisory mode (T5) now exercises it
                # and the unguarded f-string formats raised TypeError.
                #
                # v-verifier-none-guard-bulletproof-2026-05-08: replaced
                # conditional-expression guards with explicit if/else
                # because production was still throwing
                # `unsupported format string passed to NoneType.__format__`
                # at the conditional-expression line. Whatever the cause
                # of the original guard not protecting (a coercion to
                # numpy NaN, a stale .pyc, or a Python edge case I don't
                # see), the if/else form is unambiguous.
                _lam = getattr(v, "latest_age_min", None)
                if _lam is None:
                    _age_str = "unknown"
                else:
                    try:
                        _age_str = f"{float(_lam):.1f} min ago"
                    except (TypeError, ValueError):
                        _age_str = "unknown"
                _afs = getattr(v, "avg_fresh_sentiment", None)
                try:
                    _avg_fresh = float(_afs) if _afs is not None else 0.0
                except (TypeError, ValueError):
                    _avg_fresh = 0.0
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.DECISION,
                    symbol=symbol,
                    title=f"✅ Fresh News Confirmed — {signal_type.name} {symbol}",
                    message=(
                        f"Verification passed via {v.source}.\n"
                        f"  Fresh articles (last 4h):     {v.fresh_count}\n"
                        f"  Avg fresh sentiment:          {_avg_fresh:+.3f} "
                        f"({'bullish' if _avg_fresh > 0 else 'bearish'})\n"
                        f"  Most recent article:          {_age_str}\n"
                        f"  Cached sentiment:             {avg_sentiment:+.3f}\n\n"
                        f"News thesis is fresh and matches signal direction. Proceeding to entry."
                    ),
                    data={
                        'verifier_source': v.source,
                        'fresh_count': v.fresh_count,
                        'fresh_avg_sentiment': v.avg_fresh_sentiment,
                        'latest_age_min': v.latest_age_min,
                        'cached_sentiment': avg_sentiment,
                    },
                    confidence=confidence,
                    importance=8,
                ))
            # v-cooldown-regular-only-2026-05-05: only burn the per-symbol
            # cooldown when the signal had a chance of being placed — i.e.
            # we're inside regular hours. Previously, after-hours news ticks
            # set this timer; the engine then rejected for session_closed,
            # but the next-day open inherited a 60-min cooldown and silently
            # blocked clean entries (LRCX/NVDA/TSLA/SOFI all dead 09:40 ET
            # on 2026-05-05 with cooldown_remaining 30–53 min).
            try:
                import pytz as _pytz
                # v-determinism-2026-05-19: derive session check from
                # market_data.timestamp (bar time) instead of wall-clock,
                # so replay backtests reproduce the same gate decisions.
                _et_tz = _pytz.timezone("America/New_York")
                _bar_ts = market_data.timestamp
                if _bar_ts.tzinfo is None:
                    _et_now = _et_tz.localize(_bar_ts)
                else:
                    _et_now = _bar_ts.astimezone(_et_tz)
                _is_regular = (
                    _et_now.weekday() < 5
                    and ((_et_now.hour == 9 and _et_now.minute >= 30)
                         or 10 <= _et_now.hour < 16)
                )
            except Exception:
                _is_regular = True
            if _is_regular:
                self.last_signal_time[symbol] = market_data.timestamp  # v-determinism-2026-05-19

            self._log_decision(
                market_data,
                "signal_buy" if signal_type == SignalType.BUY else "signal_sell",
                "high_impact_news" if high_impact_news else "strong_sentiment",
                sentiment=round(avg_sentiment, 3),
                articles=len(news_items),
                fresh_count=(v.fresh_count if v else None),
                fresh_avg=(round(v.avg_fresh_sentiment, 3) if v else None),
                fresh_source=(v.source if v else "verifier_disabled"),
                latest_age_min=(round(v.latest_age_min, 1)
                                if v and v.latest_age_min is not None else None),
                stop=round(stop_loss, 2),
                target=round(take_profit, 2),
                atr=round(atr, 3), stop_dist=round(stop_distance, 2),
            )
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=abs(avg_sentiment),
                entry_price=market_data.close,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=0,
                reasoning={
                    'strategy': 'free_news_sentiment',
                    'sentiment': avg_sentiment,
                    'news_count': len(news_items),
                    'sources': [item.source for item in news_items[:3]],
                    'atr': atr, 'atr_mult': atr_mult,
                    'stop_distance': stop_distance,
                },
                confidence=confidence
            )

        self._log_decision(market_data, "skip", "weak_sentiment",
                           sentiment=round(avg_sentiment, 3), articles=len(news_items))
        return None
