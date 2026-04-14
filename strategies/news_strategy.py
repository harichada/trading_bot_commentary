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

        if len(news_items) < 3:  # Need at least 3 articles
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

        # Generate signal if strong sentiment
        if abs(avg_sentiment) > 0.3 or high_impact_news:
            signal_type = SignalType.BUY if avg_sentiment > 0 else SignalType.SELL
            confidence = min(abs(avg_sentiment) + 0.3, 0.8)

            # Commentary
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.OPPORTUNITY,
                symbol=symbol,
                title=f"📰 News Signal: {signal_type.name} {symbol}",
                message=f"News sentiment: {'Positive' if avg_sentiment > 0 else 'Negative'} ({avg_sentiment:.2f})\n\n"
                       f"Headlines:\n" +
                       "\n".join([f"• {item.headline}" for item in news_items[:3]]),
                data={
                    'sentiment': avg_sentiment,
                    'news_count': len(news_items),
                    'high_impact': len(high_impact_news) > 0,
                    'sources': list(set(item.source for item in news_items[:5]))
                },
                confidence=confidence,
                importance=8 if high_impact_news else 7
            ))

            # Calculate stops
            atr = market_data.indicators.get('atr', market_data.close * 0.02)

            if signal_type == SignalType.BUY:
                stop_loss = market_data.close - (2 * atr)
                take_profit = market_data.close + (4 * atr)
            else:
                stop_loss = market_data.close + (2 * atr)
                take_profit = market_data.close - (4 * atr)

            self.last_signal_time[symbol] = datetime.now()

            self._log_decision(
                market_data,
                "signal_buy" if signal_type == SignalType.BUY else "signal_sell",
                "high_impact_news" if high_impact_news else "strong_sentiment",
                sentiment=round(avg_sentiment, 3),
                articles=len(news_items),
                stop=round(stop_loss, 2),
                target=round(take_profit, 2),
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
                    'sources': [item.source for item in news_items[:3]]
                },
                confidence=confidence
            )

        self._log_decision(market_data, "skip", "weak_sentiment",
                           sentiment=round(avg_sentiment, 3), articles=len(news_items))
        return None
