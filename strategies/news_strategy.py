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

            # Use yfinance for news
            ticker = yf.Ticker(symbol)
            news_data = ticker.news

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

            # Also get RSS feed
            rss_url = self.rss_feeds['yahoo'].format(symbol=symbol)
            feed = feedparser.parse(rss_url)

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

        # Google News RSS
        try:
            google_url = self.rss_feeds['google'].format(symbol=symbol)
            feed = feedparser.parse(google_url)

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
            cooldown_left = 3600 - (datetime.now() - self.last_signal_time[symbol]).total_seconds()
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
        if abs(avg_sentiment) > 0.4 or high_impact_news:
            signal_type = SignalType.BUY if avg_sentiment > 0 else SignalType.SELL
            confidence = min(abs(avg_sentiment) + 0.3, 0.8)

            # Trend filter: don't buy positive news into a confirmed downtrend.
            # Same LCID-style "falling knife" pattern the mean-rev strategy had.
            # Keep symmetric short side open since short-of-uptrend-on-bad-news
            # is a legitimate fade setup (exhaustion, reversal).
            if signal_type == SignalType.BUY:
                indicators = market_data.indicators or {}
                try:
                    sma_50 = float(indicators.get("sma_50", 0))
                    macd_val = float(indicators.get("macd", 0))
                    macd_sig = float(indicators.get("macd_signal", 0))
                except (TypeError, ValueError):
                    sma_50, macd_val, macd_sig = 0.0, 0.0, 0.0
                if (sma_50 > 0 and market_data.close < sma_50
                        and macd_val < macd_sig):
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

            # v-news-verify-2026-04-28: announce that we DETECTED a candidate
            # signal but haven't fired yet — verification follows. Keeps the
            # commentary timeline honest about the two-stage process.
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.OPPORTUNITY,
                symbol=symbol,
                title=f"📰 News Signal Detected: {signal_type.name} {symbol} — Verifying",
                message=(
                    f"Cached sentiment: {'Positive' if avg_sentiment > 0 else 'Negative'} ({avg_sentiment:+.2f})\n"
                    f"Articles: {len(news_items)}\n\n"
                    "Headlines:\n" +
                    "\n".join([f"• {item.headline}" for item in news_items[:3]]) +
                    "\n\nRe-verifying with fresh news from Alpaca/Yahoo before entering..."
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
            try:
                expected_dir = 1 if signal_type == SignalType.BUY else -1
                v = await self.verifier.verify(symbol, expected_dir)
            except Exception as exc:
                # Verifier failure is non-fatal — log and treat as a skip
                # rather than blindly trusting cached sentiment.
                logger.debug(f"news verifier exception for {symbol}: {exc}")
                self._log_decision(
                    market_data, "skip", "verifier_error",
                    err=str(exc)[:80], sentiment=round(avg_sentiment, 3),
                )
                return None

            if not v.is_verified:
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

            # Verified — announce the confirmation in commentary so the user
            # can see WHY this signal made it past the gate (vs a vetoed one).
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.DECISION,
                symbol=symbol,
                title=f"✅ Fresh News Confirmed — {signal_type.name} {symbol}",
                message=(
                    f"Verification passed via {v.source}.\n"
                    f"  Fresh articles (last 4h):     {v.fresh_count}\n"
                    f"  Avg fresh sentiment:          {v.avg_fresh_sentiment:+.3f} "
                    f"({'bullish' if v.avg_fresh_sentiment > 0 else 'bearish'})\n"
                    f"  Most recent article:          "
                    f"{v.latest_age_min:.1f} min ago\n"
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
            self.last_signal_time[symbol] = datetime.now()

            self._log_decision(
                market_data,
                "signal_buy" if signal_type == SignalType.BUY else "signal_sell",
                "high_impact_news" if high_impact_news else "strong_sentiment",
                sentiment=round(avg_sentiment, 3),
                articles=len(news_items),
                fresh_count=v.fresh_count,
                fresh_avg=round(v.avg_fresh_sentiment, 3),
                fresh_source=v.source,
                latest_age_min=round(v.latest_age_min, 1) if v.latest_age_min is not None else None,
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
