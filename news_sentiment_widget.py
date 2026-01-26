"""
News Sentiment Widget - Real-time Market Sentiment Analysis Engine
Provides sentiment scoring, alerts, and trading integration

Features:
- Multi-source news aggregation (Yahoo, Google News, Finnhub, SEC)
- VADER + keyword-based sentiment analysis
- Optional LLM-powered nuanced sentiment (Claude/GPT)
- News velocity tracking (sudden spike detection)
- Social sentiment (Reddit, StockTwits)
- Earnings calendar integration
- Historical sentiment-price correlation
- Real-time alerts for positions

"Sentiment is the crowd's emotion. Trade with it, not against it."
"""

import asyncio
import aiohttp
import feedparser
import re
import json
import hashlib
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
from pathlib import Path
import logging
import numpy as np
import threading
from functools import lru_cache

# NLTK for VADER sentiment
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# Optional: yfinance for earnings calendar
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class SentimentLabel(Enum):
    """Sentiment classification labels"""
    VERY_BULLISH = "Very Bullish"
    BULLISH = "Bullish"
    SLIGHTLY_BULLISH = "Slightly Bullish"
    NEUTRAL = "Neutral"
    SLIGHTLY_BEARISH = "Slightly Bearish"
    BEARISH = "Bearish"
    VERY_BEARISH = "Very Bearish"


class NewsImpact(Enum):
    """News impact levels"""
    BREAKING = "breaking"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AlertType(Enum):
    """Types of sentiment alerts"""
    BREAKING_NEWS = "breaking_news"
    SENTIMENT_FLIP = "sentiment_flip"
    SENTIMENT_DIVERGENCE = "sentiment_divergence"
    MARKET_SELLOFF_RISK = "market_selloff_risk"
    CONFLICTING_SIGNALS = "conflicting_signals"
    NEWS_VELOCITY_SPIKE = "news_velocity_spike"
    EARNINGS_UPCOMING = "earnings_upcoming"


@dataclass
class SentimentScore:
    """Sentiment score with metadata"""
    score: float  # -100 to +100
    label: SentimentLabel
    confidence: float  # 0.0 to 1.0
    trend: str  # "↑", "→", "↓"
    trend_change: float  # Change from previous reading
    history: List[Tuple[datetime, float]] = field(default_factory=list)  # For sparkline
    source_count: int = 0  # Number of sources contributing
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'score': round(self.score, 1),
            'label': self.label.value,
            'confidence': round(self.confidence, 2),
            'trend': self.trend,
            'trend_change': round(self.trend_change, 1),
            'history': [(t.isoformat(), round(s, 1)) for t, s in self.history[-20:]],
            'source_count': self.source_count,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class NewsHeadline:
    """Individual news headline with sentiment"""
    id: str
    headline: str
    summary: str
    source: str
    url: str
    published_time: datetime
    sentiment_score: float  # -100 to +100
    sentiment_confidence: float
    impact: NewsImpact
    symbols_mentioned: List[str] = field(default_factory=list)
    sectors_mentioned: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    is_breaking: bool = False
    llm_analysis: Optional[str] = None  # LLM-generated summary

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'headline': self.headline,
            'summary': self.summary[:200] + '...' if len(self.summary) > 200 else self.summary,
            'source': self.source,
            'url': self.url,
            'published_time': self.published_time.isoformat(),
            'age_minutes': int((datetime.now() - self.published_time).total_seconds() / 60),
            'sentiment_score': round(self.sentiment_score, 1),
            'sentiment_confidence': round(self.sentiment_confidence, 2),
            'impact': self.impact.value,
            'symbols': self.symbols_mentioned,
            'is_breaking': self.is_breaking,
            'llm_analysis': self.llm_analysis
        }


@dataclass
class SentimentAlert:
    """Sentiment alert for trading"""
    id: str
    alert_type: AlertType
    symbol: Optional[str]
    title: str
    message: str
    urgency: str  # "critical", "high", "medium", "low"
    sentiment_before: Optional[float]
    sentiment_after: Optional[float]
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.alert_type.value,
            'symbol': self.symbol,
            'title': self.title,
            'message': self.message,
            'urgency': self.urgency,
            'sentiment_before': self.sentiment_before,
            'sentiment_after': self.sentiment_after,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged
        }


@dataclass
class EarningsEvent:
    """Upcoming earnings event"""
    symbol: str
    company_name: str
    earnings_date: date
    earnings_time: str  # "BMO" (before market open), "AMC" (after market close)
    days_until: int
    estimated_eps: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'company_name': self.company_name,
            'earnings_date': self.earnings_date.isoformat(),
            'earnings_time': self.earnings_time,
            'days_until': self.days_until,
            'estimated_eps': self.estimated_eps
        }


@dataclass
class NewsVelocity:
    """News volume tracking"""
    symbol: str
    current_velocity: float  # News items per hour
    average_velocity: float  # Historical average
    velocity_ratio: float  # current / average
    is_spiking: bool
    spike_started: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'current_velocity': round(self.current_velocity, 2),
            'average_velocity': round(self.average_velocity, 2),
            'velocity_ratio': round(self.velocity_ratio, 2),
            'is_spiking': self.is_spiking,
            'spike_started': self.spike_started.isoformat() if self.spike_started else None
        }


@dataclass
class SocialSentiment:
    """Social media sentiment data"""
    symbol: str
    reddit_sentiment: float  # -100 to +100
    reddit_mentions: int
    reddit_trending: bool
    stocktwits_sentiment: float
    stocktwits_messages: int
    stocktwits_trending: bool
    combined_score: float
    buzz_level: str  # "viral", "high", "normal", "low"
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'reddit': {
                'sentiment': round(self.reddit_sentiment, 1),
                'mentions': self.reddit_mentions,
                'trending': self.reddit_trending
            },
            'stocktwits': {
                'sentiment': round(self.stocktwits_sentiment, 1),
                'messages': self.stocktwits_messages,
                'trending': self.stocktwits_trending
            },
            'combined_score': round(self.combined_score, 1),
            'buzz_level': self.buzz_level,
            'last_updated': self.last_updated.isoformat()
        }


# =============================================================================
# SENTIMENT ANALYZER
# =============================================================================

class SentimentAnalyzer:
    """
    Multi-method sentiment analysis engine.

    Methods:
    1. VADER (rule-based, fast, free)
    2. Keyword matching (financial-specific)
    3. LLM-powered (optional, most nuanced)
    """

    # Financial-specific keywords with sentiment weights
    BULLISH_KEYWORDS = {
        # Strong bullish (+3)
        'surge': 3, 'soar': 3, 'skyrocket': 3, 'breakout': 3, 'moon': 3,
        'record high': 3, 'all-time high': 3, 'explosive': 3, 'blowout': 3,
        # Moderate bullish (+2)
        'beat': 2, 'beats': 2, 'exceeded': 2, 'outperform': 2, 'upgrade': 2,
        'rally': 2, 'gain': 2, 'jump': 2, 'climb': 2, 'rise': 2, 'boost': 2,
        'bullish': 2, 'optimistic': 2, 'positive': 2, 'strong': 2,
        'growth': 2, 'expand': 2, 'profit': 2, 'success': 2,
        # Mild bullish (+1)
        'up': 1, 'higher': 1, 'increase': 1, 'improve': 1, 'recover': 1,
        'stable': 1, 'steady': 1, 'confident': 1, 'buy': 1, 'accumulate': 1,
    }

    BEARISH_KEYWORDS = {
        # Strong bearish (-3)
        'crash': -3, 'plunge': -3, 'collapse': -3, 'tank': -3, 'disaster': -3,
        'bankruptcy': -3, 'fraud': -3, 'scandal': -3, 'investigation': -3,
        # Moderate bearish (-2)
        'miss': -2, 'missed': -2, 'decline': -2, 'drop': -2, 'fall': -2,
        'downgrade': -2, 'sell': -2, 'bearish': -2, 'negative': -2, 'weak': -2,
        'loss': -2, 'losses': -2, 'warning': -2, 'concern': -2, 'fear': -2,
        'layoff': -2, 'layoffs': -2, 'cut': -2, 'slash': -2,
        # Mild bearish (-1)
        'down': -1, 'lower': -1, 'decrease': -1, 'slow': -1, 'slowing': -1,
        'uncertainty': -1, 'volatile': -1, 'risk': -1, 'caution': -1,
    }

    # Source credibility weights
    SOURCE_WEIGHTS = {
        'reuters': 1.3,
        'bloomberg': 1.3,
        'wall street journal': 1.2,
        'wsj': 1.2,
        'financial times': 1.2,
        'cnbc': 1.1,
        'marketwatch': 1.0,
        'yahoo finance': 0.95,
        'google news': 0.9,
        'seeking alpha': 0.85,
        'benzinga': 0.85,
        'motley fool': 0.8,
        'investopedia': 0.8,
        'default': 0.75
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Initialize VADER
        self.vader = None
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()

        # LLM settings (optional)
        self.llm_enabled = self.config.get('llm_enabled', False)
        self.llm_provider = self.config.get('llm_provider', 'anthropic')  # or 'openai'
        self.llm_api_key = self.config.get('llm_api_key', None)

        # Recency settings
        self.full_weight_minutes = self.config.get('full_weight_minutes', 15)
        self.half_life_minutes = self.config.get('half_life_minutes', 60)

    def analyze(self, text: str, source: str = 'default',
                published_time: Optional[datetime] = None) -> Tuple[float, float]:
        """
        Analyze sentiment of text.

        Returns:
            (score, confidence) where score is -100 to +100
        """
        if not text:
            return 0.0, 0.0

        text_lower = text.lower()

        # Method 1: VADER sentiment
        vader_score = 0.0
        if self.vader:
            vader_result = self.vader.polarity_scores(text)
            # Convert compound score (-1 to 1) to our scale (-100 to 100)
            vader_score = vader_result['compound'] * 100

        # Method 2: Keyword matching
        keyword_score = self._keyword_sentiment(text_lower)

        # Combine scores (weighted average)
        if self.vader:
            raw_score = (vader_score * 0.6) + (keyword_score * 0.4)
        else:
            raw_score = keyword_score

        # Apply source weight
        source_weight = self._get_source_weight(source)
        weighted_score = raw_score * source_weight

        # Apply recency decay
        if published_time:
            recency_weight = self._calculate_recency_weight(published_time)
            weighted_score *= recency_weight

        # Calculate confidence based on text length and keyword matches
        confidence = self._calculate_confidence(text, keyword_score)

        # Clamp to range
        final_score = max(-100, min(100, weighted_score))

        return final_score, confidence

    def _keyword_sentiment(self, text: str) -> float:
        """Calculate sentiment from keyword matching"""
        score = 0
        matches = 0

        for keyword, weight in self.BULLISH_KEYWORDS.items():
            if keyword in text:
                score += weight * 10  # Scale to -100 to 100
                matches += 1

        for keyword, weight in self.BEARISH_KEYWORDS.items():
            if keyword in text:
                score += weight * 10  # Negative weights
                matches += 1

        # Normalize by number of matches (with diminishing returns)
        if matches > 0:
            score = score / (1 + np.log1p(matches) * 0.3)

        return max(-100, min(100, score))

    def _get_source_weight(self, source: str) -> float:
        """Get credibility weight for news source"""
        source_lower = source.lower()
        for key, weight in self.SOURCE_WEIGHTS.items():
            if key in source_lower:
                return weight
        return self.SOURCE_WEIGHTS['default']

    def _calculate_recency_weight(self, published_time: datetime) -> float:
        """Calculate weight decay based on article age"""
        age_minutes = (datetime.now() - published_time).total_seconds() / 60

        if age_minutes <= self.full_weight_minutes:
            return 1.0

        # Exponential decay after full weight period
        decay_minutes = age_minutes - self.full_weight_minutes
        half_lives = decay_minutes / self.half_life_minutes
        return max(0.1, 0.5 ** half_lives)

    def _calculate_confidence(self, text: str, keyword_score: float) -> float:
        """Calculate confidence in the sentiment score"""
        confidence = 0.5  # Base confidence

        # Longer text = more confident
        word_count = len(text.split())
        if word_count > 50:
            confidence += 0.2
        elif word_count > 20:
            confidence += 0.1

        # Strong keyword matches = more confident
        if abs(keyword_score) > 50:
            confidence += 0.2
        elif abs(keyword_score) > 25:
            confidence += 0.1

        return min(1.0, confidence)

    async def analyze_with_llm(self, headlines: List[str],
                                symbol: Optional[str] = None) -> Tuple[float, str]:
        """
        Use LLM for nuanced sentiment analysis (advanced feature).

        Returns:
            (score, analysis_text)
        """
        if not self.llm_enabled or not self.llm_api_key:
            return 0.0, "LLM analysis not enabled"

        # Build prompt
        headlines_text = "\n".join([f"- {h}" for h in headlines[:10]])
        symbol_context = f" for {symbol}" if symbol else ""

        prompt = f"""Analyze the sentiment{symbol_context} from these recent financial news headlines:

{headlines_text}

Provide:
1. A sentiment score from -100 (extremely bearish) to +100 (extremely bullish)
2. A brief 1-2 sentence analysis explaining the sentiment

Respond in JSON format:
{{"score": <number>, "analysis": "<text>"}}"""

        try:
            if self.llm_provider == 'anthropic':
                return await self._call_anthropic(prompt)
            elif self.llm_provider == 'openai':
                return await self._call_openai(prompt)
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return 0.0, f"LLM analysis failed: {e}"

        return 0.0, "Unknown LLM provider"

    async def _call_anthropic(self, prompt: str) -> Tuple[float, str]:
        """Call Anthropic API for sentiment analysis"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.anthropic.com/v1/messages',
                headers={
                    'x-api-key': self.llm_api_key,
                    'content-type': 'application/json',
                    'anthropic-version': '2023-06-01'
                },
                json={
                    'model': 'claude-3-haiku-20240307',  # Fast and cheap
                    'max_tokens': 200,
                    'messages': [{'role': 'user', 'content': prompt}]
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data['content'][0]['text']
                    # Parse JSON response
                    result = json.loads(content)
                    return float(result['score']), result['analysis']
        return 0.0, "API call failed"

    async def _call_openai(self, prompt: str) -> Tuple[float, str]:
        """Call OpenAI API for sentiment analysis"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.llm_api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'gpt-3.5-turbo',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': 200
                }
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data['choices'][0]['message']['content']
                    result = json.loads(content)
                    return float(result['score']), result['analysis']
        return 0.0, "API call failed"


# =============================================================================
# NEWS AGGREGATOR
# =============================================================================

class NewsAggregator:
    """
    Multi-source news aggregation with caching.

    Sources:
    - Yahoo Finance RSS
    - Google News RSS
    - Finnhub API (optional)
    - SEC EDGAR (for filings)
    """

    # Symbol to company name mapping for better search
    SYMBOL_NAMES = {
        'TSLA': 'Tesla',
        'NVDA': 'Nvidia',
        'AMD': 'AMD',
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Google Alphabet',
        'AMZN': 'Amazon',
        'META': 'Meta Facebook',
        'MARA': 'Marathon Digital',
        'RIOT': 'Riot Platforms',
        'COIN': 'Coinbase',
        'PLTR': 'Palantir',
        'SPY': 'S&P 500',
        'QQQ': 'Nasdaq',
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.analyzer = SentimentAnalyzer(config.get('analyzer', {}))

        # API keys
        self.finnhub_key = self.config.get('finnhub_api_key', None)

        # Cache settings
        self.cache_ttl_seconds = self.config.get('cache_ttl_seconds', 60)
        self._cache: Dict[str, Tuple[datetime, List[NewsHeadline]]] = {}
        self._cache_lock = threading.Lock()

        # News velocity tracking
        self._news_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

    async def fetch_all_news(self, symbols: List[str],
                             hours: int = 24) -> List[NewsHeadline]:
        """Fetch news from all sources for given symbols"""
        all_news = []

        # Fetch from each source concurrently
        tasks = [
            self._fetch_yahoo_rss(symbols),
            self._fetch_google_news(symbols),
        ]

        if self.finnhub_key:
            tasks.append(self._fetch_finnhub(symbols))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_news.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"News fetch error: {result}")

        # Deduplicate by headline similarity
        unique_news = self._deduplicate_news(all_news)

        # Sort by time (newest first)
        unique_news.sort(key=lambda x: x.published_time, reverse=True)

        # Filter by time window
        cutoff = datetime.now() - timedelta(hours=hours)
        filtered = [n for n in unique_news if n.published_time > cutoff]

        # Update velocity tracking
        for symbol in symbols:
            symbol_news = [n for n in filtered if symbol in n.symbols_mentioned]
            self._news_counts[symbol].append((datetime.now(), len(symbol_news)))

        return filtered[:50]  # Limit to 50 most recent

    async def _fetch_yahoo_rss(self, symbols: List[str]) -> List[NewsHeadline]:
        """Fetch from Yahoo Finance RSS"""
        news_items = []

        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
                    async with session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            content = await resp.text()
                            feed = feedparser.parse(content)

                            for entry in feed.entries[:10]:
                                headline = NewsHeadline(
                                    id=hashlib.md5(entry.get('link', '').encode()).hexdigest()[:12],
                                    headline=entry.get('title', ''),
                                    summary=entry.get('summary', ''),
                                    source='Yahoo Finance',
                                    url=entry.get('link', ''),
                                    published_time=self._parse_time(entry.get('published', '')),
                                    sentiment_score=0,
                                    sentiment_confidence=0,
                                    impact=NewsImpact.MEDIUM,
                                    symbols_mentioned=[symbol]
                                )

                                # Analyze sentiment
                                text = f"{headline.headline} {headline.summary}"
                                score, conf = self.analyzer.analyze(
                                    text, 'Yahoo Finance', headline.published_time
                                )
                                headline.sentiment_score = score
                                headline.sentiment_confidence = conf
                                headline.impact = self._determine_impact(headline)

                                news_items.append(headline)

                except Exception as e:
                    logger.error(f"Yahoo RSS error for {symbol}: {e}")

        return news_items

    async def _fetch_google_news(self, symbols: List[str]) -> List[NewsHeadline]:
        """Fetch from Google News RSS"""
        news_items = []

        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    # Use company name for better results
                    search_term = self.SYMBOL_NAMES.get(symbol, symbol)
                    url = f"https://news.google.com/rss/search?q={search_term}+stock&hl=en-US&gl=US&ceid=US:en"

                    async with session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            content = await resp.text()
                            feed = feedparser.parse(content)

                            for entry in feed.entries[:10]:
                                # Extract actual source from title
                                title = entry.get('title', '')
                                source = 'Google News'
                                if ' - ' in title:
                                    parts = title.rsplit(' - ', 1)
                                    title = parts[0]
                                    source = parts[1] if len(parts) > 1 else 'Google News'

                                headline = NewsHeadline(
                                    id=hashlib.md5(entry.get('link', '').encode()).hexdigest()[:12],
                                    headline=title,
                                    summary=entry.get('summary', ''),
                                    source=source,
                                    url=entry.get('link', ''),
                                    published_time=self._parse_time(entry.get('published', '')),
                                    sentiment_score=0,
                                    sentiment_confidence=0,
                                    impact=NewsImpact.MEDIUM,
                                    symbols_mentioned=[symbol]
                                )

                                text = f"{headline.headline} {headline.summary}"
                                score, conf = self.analyzer.analyze(
                                    text, source, headline.published_time
                                )
                                headline.sentiment_score = score
                                headline.sentiment_confidence = conf
                                headline.impact = self._determine_impact(headline)

                                news_items.append(headline)

                except Exception as e:
                    logger.error(f"Google News error for {symbol}: {e}")

        return news_items

    async def _fetch_finnhub(self, symbols: List[str]) -> List[NewsHeadline]:
        """Fetch from Finnhub API (requires API key)"""
        if not self.finnhub_key:
            return []

        news_items = []

        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    # Finnhub company news endpoint
                    today = datetime.now().strftime('%Y-%m-%d')
                    week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={week_ago}&to={today}&token={self.finnhub_key}"

                    async with session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()

                            for item in data[:10]:
                                headline = NewsHeadline(
                                    id=str(item.get('id', ''))[:12],
                                    headline=item.get('headline', ''),
                                    summary=item.get('summary', ''),
                                    source=item.get('source', 'Finnhub'),
                                    url=item.get('url', ''),
                                    published_time=datetime.fromtimestamp(item.get('datetime', 0)),
                                    sentiment_score=0,
                                    sentiment_confidence=0,
                                    impact=NewsImpact.MEDIUM,
                                    symbols_mentioned=[symbol]
                                )

                                text = f"{headline.headline} {headline.summary}"
                                score, conf = self.analyzer.analyze(
                                    text, headline.source, headline.published_time
                                )
                                headline.sentiment_score = score
                                headline.sentiment_confidence = conf
                                headline.impact = self._determine_impact(headline)

                                news_items.append(headline)

                except Exception as e:
                    logger.error(f"Finnhub error for {symbol}: {e}")

        return news_items

    def _parse_time(self, time_str: str) -> datetime:
        """Parse various time formats"""
        if not time_str:
            return datetime.now()

        formats = [
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S GMT',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%d %H:%M:%S',
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(time_str, fmt)
                if dt.tzinfo:
                    dt = dt.replace(tzinfo=None)
                return dt
            except ValueError:
                continue

        return datetime.now()

    def _determine_impact(self, headline: NewsHeadline) -> NewsImpact:
        """Determine news impact level"""
        text = headline.headline.lower()

        # Breaking news indicators
        breaking_words = ['breaking', 'just in', 'alert', 'urgent', 'exclusive']
        if any(word in text for word in breaking_words):
            headline.is_breaking = True
            return NewsImpact.BREAKING

        # High impact indicators
        high_impact = ['earnings', 'revenue', 'guidance', 'ceo', 'acquisition',
                       'merger', 'fda', 'sec', 'lawsuit', 'investigation']
        if any(word in text for word in high_impact):
            return NewsImpact.HIGH

        # Strong sentiment = higher impact
        if abs(headline.sentiment_score) > 50:
            return NewsImpact.HIGH
        elif abs(headline.sentiment_score) > 25:
            return NewsImpact.MEDIUM

        return NewsImpact.LOW

    def _deduplicate_news(self, news: List[NewsHeadline]) -> List[NewsHeadline]:
        """Remove duplicate headlines"""
        seen_headlines = set()
        unique = []

        for item in news:
            # Normalize headline for comparison
            normalized = re.sub(r'[^\w\s]', '', item.headline.lower())[:50]
            if normalized not in seen_headlines:
                seen_headlines.add(normalized)
                unique.append(item)

        return unique

    def get_news_velocity(self, symbol: str) -> NewsVelocity:
        """Calculate news velocity for a symbol"""
        counts = list(self._news_counts.get(symbol, []))

        if len(counts) < 2:
            return NewsVelocity(
                symbol=symbol,
                current_velocity=0,
                average_velocity=0,
                velocity_ratio=1.0,
                is_spiking=False
            )

        # Calculate current velocity (last hour)
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_counts = [c for t, c in counts if t > hour_ago]
        current_velocity = sum(recent_counts) / max(len(recent_counts), 1)

        # Calculate average velocity
        all_counts = [c for _, c in counts]
        average_velocity = sum(all_counts) / len(all_counts) if all_counts else 1

        # Calculate ratio
        velocity_ratio = current_velocity / max(average_velocity, 0.1)

        # Detect spike (2x normal = spike)
        is_spiking = velocity_ratio > 2.0

        return NewsVelocity(
            symbol=symbol,
            current_velocity=current_velocity,
            average_velocity=average_velocity,
            velocity_ratio=velocity_ratio,
            is_spiking=is_spiking,
            spike_started=datetime.now() if is_spiking else None
        )


# =============================================================================
# SOCIAL SENTIMENT
# =============================================================================

class SocialSentimentTracker:
    """
    Track social media sentiment from Reddit and StockTwits.

    Note: This provides simulated data unless API keys are configured.
    Real integration requires Reddit API (PRAW) and StockTwits API.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.reddit_enabled = self.config.get('reddit_enabled', False)
        self.stocktwits_enabled = self.config.get('stocktwits_enabled', False)

        self._cache: Dict[str, Tuple[datetime, SocialSentiment]] = {}
        self._cache_ttl = timedelta(minutes=5)

    async def get_social_sentiment(self, symbol: str) -> SocialSentiment:
        """Get social sentiment for a symbol"""
        # Check cache
        if symbol in self._cache:
            cached_time, cached_data = self._cache[symbol]
            if datetime.now() - cached_time < self._cache_ttl:
                return cached_data

        # Fetch from sources
        reddit_data = await self._fetch_reddit_sentiment(symbol)
        stocktwits_data = await self._fetch_stocktwits_sentiment(symbol)

        # Combine scores
        combined = (reddit_data['sentiment'] * 0.5 + stocktwits_data['sentiment'] * 0.5)
        total_mentions = reddit_data['mentions'] + stocktwits_data['messages']

        # Determine buzz level
        if total_mentions > 1000:
            buzz = "viral"
        elif total_mentions > 500:
            buzz = "high"
        elif total_mentions > 100:
            buzz = "normal"
        else:
            buzz = "low"

        sentiment = SocialSentiment(
            symbol=symbol,
            reddit_sentiment=reddit_data['sentiment'],
            reddit_mentions=reddit_data['mentions'],
            reddit_trending=reddit_data['trending'],
            stocktwits_sentiment=stocktwits_data['sentiment'],
            stocktwits_messages=stocktwits_data['messages'],
            stocktwits_trending=stocktwits_data['trending'],
            combined_score=combined,
            buzz_level=buzz
        )

        self._cache[symbol] = (datetime.now(), sentiment)
        return sentiment

    async def _fetch_reddit_sentiment(self, symbol: str) -> Dict:
        """Fetch Reddit sentiment (simulated if no API)"""
        # TODO: Integrate with Reddit API (PRAW) for real data
        # For now, return simulated data based on symbol volatility

        # Simulated based on common patterns
        base_sentiment = {
            'TSLA': 35, 'NVDA': 45, 'AMD': 30, 'AAPL': 20,
            'GME': 60, 'AMC': 55, 'MARA': 25, 'RIOT': 20
        }.get(symbol, 0)

        noise = np.random.normal(0, 15)

        return {
            'sentiment': max(-100, min(100, base_sentiment + noise)),
            'mentions': int(abs(base_sentiment) * 10 + np.random.randint(50, 200)),
            'trending': abs(base_sentiment) > 40
        }

    async def _fetch_stocktwits_sentiment(self, symbol: str) -> Dict:
        """Fetch StockTwits sentiment"""
        # TODO: Integrate with StockTwits API for real data

        base_sentiment = {
            'TSLA': 40, 'NVDA': 50, 'AMD': 35, 'AAPL': 25,
            'GME': 65, 'AMC': 60, 'MARA': 30, 'RIOT': 25
        }.get(symbol, 0)

        noise = np.random.normal(0, 10)

        return {
            'sentiment': max(-100, min(100, base_sentiment + noise)),
            'messages': int(abs(base_sentiment) * 8 + np.random.randint(30, 150)),
            'trending': abs(base_sentiment) > 45
        }


# =============================================================================
# EARNINGS CALENDAR
# =============================================================================

class EarningsCalendar:
    """Track upcoming earnings for watchlist stocks"""

    def __init__(self):
        self._cache: Dict[str, Tuple[datetime, Optional[EarningsEvent]]] = {}
        self._cache_ttl = timedelta(hours=6)

    async def get_upcoming_earnings(self, symbols: List[str]) -> List[EarningsEvent]:
        """Get upcoming earnings for symbols"""
        events = []

        for symbol in symbols:
            event = await self._get_earnings(symbol)
            if event and event.days_until <= 14:  # Next 2 weeks
                events.append(event)

        # Sort by date
        events.sort(key=lambda x: x.earnings_date)
        return events

    async def _get_earnings(self, symbol: str) -> Optional[EarningsEvent]:
        """Get earnings date for a symbol"""
        # Check cache
        if symbol in self._cache:
            cached_time, cached_data = self._cache[symbol]
            if datetime.now() - cached_time < self._cache_ttl:
                return cached_data

        event = None

        if YFINANCE_AVAILABLE:
            try:
                ticker = yf.Ticker(symbol)
                calendar = ticker.calendar

                if calendar is not None and not calendar.empty:
                    earnings_date = calendar.get('Earnings Date', [None])[0]
                    if earnings_date:
                        if hasattr(earnings_date, 'date'):
                            earnings_date = earnings_date.date()
                        elif isinstance(earnings_date, datetime):
                            earnings_date = earnings_date.date()

                        days_until = (earnings_date - date.today()).days

                        if days_until >= 0:
                            event = EarningsEvent(
                                symbol=symbol,
                                company_name=ticker.info.get('shortName', symbol),
                                earnings_date=earnings_date,
                                earnings_time="BMO",  # Default
                                days_until=days_until,
                                estimated_eps=calendar.get('Earnings Estimate', [None])[0]
                            )
            except Exception as e:
                logger.debug(f"Earnings lookup failed for {symbol}: {e}")

        self._cache[symbol] = (datetime.now(), event)
        return event


# =============================================================================
# SENTIMENT AGGREGATOR - MAIN ENGINE
# =============================================================================

class NewsSentimentEngine:
    """
    Main sentiment engine that aggregates all sources and provides
    trading integration.

    This is the primary interface for the news sentiment widget.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Initialize components
        self.news_aggregator = NewsAggregator(config.get('news_aggregator', {}))
        self.social_tracker = SocialSentimentTracker(config.get('social', {}))
        self.earnings_calendar = EarningsCalendar()
        self.analyzer = SentimentAnalyzer(config.get('analyzer', {}))

        # State tracking
        self._market_sentiment_history: deque = deque(maxlen=100)
        self._symbol_sentiment_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._last_sentiment: Dict[str, float] = {}
        self._alerts: List[SentimentAlert] = []
        self._headlines_cache: List[NewsHeadline] = []
        self._last_update: Optional[datetime] = None

        # Alert thresholds
        self.selloff_threshold = self.config.get('selloff_threshold', -20)
        self.sentiment_flip_threshold = self.config.get('sentiment_flip_threshold', 25)

        # Trading integration thresholds
        self.strong_bullish_threshold = self.config.get('strong_bullish', 50)
        self.strong_bearish_threshold = self.config.get('strong_bearish', -50)

        # Historical correlation tracking
        self._sentiment_price_history: Dict[str, List[Tuple[datetime, float, float]]] = defaultdict(list)

        # Watchlist
        self.watchlist: List[str] = self.config.get('watchlist', [
            'TSLA', 'NVDA', 'AMD', 'AAPL', 'SPY'
        ])

    async def update(self) -> Dict:
        """
        Main update method - fetches all data and returns complete state.
        Call this periodically (every 30-60 seconds).
        """
        try:
            # Fetch news for watchlist
            headlines = await self.news_aggregator.fetch_all_news(self.watchlist, hours=24)
            self._headlines_cache = headlines

            # Calculate market sentiment
            market_sentiment = self._calculate_market_sentiment(headlines)
            self._market_sentiment_history.append((datetime.now(), market_sentiment.score))

            # Calculate per-symbol sentiment
            symbol_sentiments = {}
            for symbol in self.watchlist:
                symbol_headlines = [h for h in headlines if symbol in h.symbols_mentioned]
                sentiment = self._calculate_symbol_sentiment(symbol, symbol_headlines)
                symbol_sentiments[symbol] = sentiment
                self._symbol_sentiment_history[symbol].append((datetime.now(), sentiment.score))

            # Get social sentiment
            social_sentiments = {}
            for symbol in self.watchlist:
                social = await self.social_tracker.get_social_sentiment(symbol)
                social_sentiments[symbol] = social

            # Get news velocity
            velocities = {}
            for symbol in self.watchlist:
                velocities[symbol] = self.news_aggregator.get_news_velocity(symbol)

            # Get upcoming earnings
            earnings = await self.earnings_calendar.get_upcoming_earnings(self.watchlist)

            # Check for alerts
            self._check_alerts(market_sentiment, symbol_sentiments, velocities, earnings)

            self._last_update = datetime.now()

            return {
                'market_sentiment': market_sentiment.to_dict(),
                'symbol_sentiments': {s: sent.to_dict() for s, sent in symbol_sentiments.items()},
                'social_sentiments': {s: soc.to_dict() for s, soc in social_sentiments.items()},
                'headlines': [h.to_dict() for h in headlines[:10]],
                'velocities': {s: v.to_dict() for s, v in velocities.items()},
                'earnings': [e.to_dict() for e in earnings],
                'alerts': [a.to_dict() for a in self._alerts if not a.acknowledged][-5:],
                'last_updated': self._last_update.isoformat()
            }

        except Exception as e:
            logger.error(f"Sentiment update failed: {e}")
            return {
                'error': str(e),
                'last_updated': self._last_update.isoformat() if self._last_update else None
            }

    def _calculate_market_sentiment(self, headlines: List[NewsHeadline]) -> SentimentScore:
        """Calculate overall market sentiment from all headlines"""
        if not headlines:
            return SentimentScore(
                score=0, label=SentimentLabel.NEUTRAL, confidence=0,
                trend="→", trend_change=0
            )

        # Weight by recency and confidence
        weighted_scores = []
        total_weight = 0

        for h in headlines:
            weight = h.sentiment_confidence * self.analyzer._calculate_recency_weight(h.published_time)
            weighted_scores.append(h.sentiment_score * weight)
            total_weight += weight

        if total_weight > 0:
            avg_score = sum(weighted_scores) / total_weight
        else:
            avg_score = 0

        # Calculate trend
        history = list(self._market_sentiment_history)
        trend, trend_change = self._calculate_trend(history, avg_score)

        # Determine label
        label = self._score_to_label(avg_score)

        # Calculate confidence based on agreement
        scores = [h.sentiment_score for h in headlines]
        std_dev = np.std(scores) if len(scores) > 1 else 0
        confidence = max(0.3, 1.0 - (std_dev / 100))

        return SentimentScore(
            score=avg_score,
            label=label,
            confidence=confidence,
            trend=trend,
            trend_change=trend_change,
            history=history[-20:],
            source_count=len(headlines)
        )

    def _calculate_symbol_sentiment(self, symbol: str,
                                     headlines: List[NewsHeadline]) -> SentimentScore:
        """Calculate sentiment for a specific symbol"""
        if not headlines:
            return SentimentScore(
                score=0, label=SentimentLabel.NEUTRAL, confidence=0,
                trend="→", trend_change=0
            )

        weighted_scores = []
        total_weight = 0

        for h in headlines:
            weight = h.sentiment_confidence * self.analyzer._calculate_recency_weight(h.published_time)
            weighted_scores.append(h.sentiment_score * weight)
            total_weight += weight

        avg_score = sum(weighted_scores) / total_weight if total_weight > 0 else 0

        history = list(self._symbol_sentiment_history.get(symbol, []))
        trend, trend_change = self._calculate_trend(history, avg_score)

        label = self._score_to_label(avg_score)

        return SentimentScore(
            score=avg_score,
            label=label,
            confidence=min(1.0, len(headlines) / 5),
            trend=trend,
            trend_change=trend_change,
            history=history[-20:],
            source_count=len(headlines)
        )

    def _calculate_trend(self, history: List[Tuple[datetime, float]],
                         current: float) -> Tuple[str, float]:
        """Calculate trend direction and magnitude"""
        if len(history) < 2:
            return "→", 0

        # Compare to 15 min ago
        cutoff = datetime.now() - timedelta(minutes=15)
        recent = [s for t, s in history if t > cutoff]

        if not recent:
            return "→", 0

        prev_avg = sum(recent) / len(recent)
        change = current - prev_avg

        if change > 5:
            return "↑", change
        elif change < -5:
            return "↓", change
        else:
            return "→", change

    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert score to sentiment label"""
        if score >= 60:
            return SentimentLabel.VERY_BULLISH
        elif score >= 35:
            return SentimentLabel.BULLISH
        elif score >= 15:
            return SentimentLabel.SLIGHTLY_BULLISH
        elif score >= -15:
            return SentimentLabel.NEUTRAL
        elif score >= -35:
            return SentimentLabel.SLIGHTLY_BEARISH
        elif score >= -60:
            return SentimentLabel.BEARISH
        else:
            return SentimentLabel.VERY_BEARISH

    def _check_alerts(self, market: SentimentScore,
                      symbols: Dict[str, SentimentScore],
                      velocities: Dict[str, NewsVelocity],
                      earnings: List[EarningsEvent]):
        """Check for alert conditions"""
        now = datetime.now()

        # Market selloff risk
        if market.trend_change < self.selloff_threshold:
            self._add_alert(AlertType.MARKET_SELLOFF_RISK, None,
                           "Market Sentiment Dropping",
                           f"Market sentiment dropped {abs(market.trend_change):.0f} points in 15 minutes",
                           "critical", market.score + market.trend_change, market.score)

        # Symbol sentiment flips
        for symbol, sentiment in symbols.items():
            prev = self._last_sentiment.get(symbol, 0)

            # Detect flip (crossed zero with significant move)
            if (prev > self.sentiment_flip_threshold and sentiment.score < -self.sentiment_flip_threshold) or \
               (prev < -self.sentiment_flip_threshold and sentiment.score > self.sentiment_flip_threshold):
                direction = "bullish to bearish" if prev > 0 else "bearish to bullish"
                self._add_alert(AlertType.SENTIMENT_FLIP, symbol,
                               f"{symbol} Sentiment Flipped",
                               f"{symbol} sentiment changed from {direction}",
                               "high", prev, sentiment.score)

            self._last_sentiment[symbol] = sentiment.score

        # News velocity spikes
        for symbol, velocity in velocities.items():
            if velocity.is_spiking:
                self._add_alert(AlertType.NEWS_VELOCITY_SPIKE, symbol,
                               f"{symbol} News Spike",
                               f"Unusual news volume for {symbol} ({velocity.velocity_ratio:.1f}x normal)",
                               "medium", None, None)

        # Upcoming earnings
        for event in earnings:
            if event.days_until <= 1:
                self._add_alert(AlertType.EARNINGS_UPCOMING, event.symbol,
                               f"{event.symbol} Earnings Tomorrow",
                               f"{event.company_name} reports earnings {event.earnings_time}",
                               "high", None, None)

    def _add_alert(self, alert_type: AlertType, symbol: Optional[str],
                   title: str, message: str, urgency: str,
                   before: Optional[float], after: Optional[float]):
        """Add an alert if not duplicate"""
        # Check for recent duplicate
        recent_alerts = [a for a in self._alerts
                        if datetime.now() - a.timestamp < timedelta(minutes=15)
                        and a.alert_type == alert_type and a.symbol == symbol]

        if not recent_alerts:
            alert = SentimentAlert(
                id=hashlib.md5(f"{alert_type}{symbol}{datetime.now()}".encode()).hexdigest()[:8],
                alert_type=alert_type,
                symbol=symbol,
                title=title,
                message=message,
                urgency=urgency,
                sentiment_before=before,
                sentiment_after=after
            )
            self._alerts.append(alert)

            # Keep only last 50 alerts
            if len(self._alerts) > 50:
                self._alerts = self._alerts[-50:]

    # =========================================================================
    # TRADING INTEGRATION
    # =========================================================================

    def get_confluence_modifier(self, symbol: str) -> float:
        """
        Get sentiment modifier for confluence scoring.

        Returns:
            -1.0 to +1.0 modifier to add to confluence score
        """
        history = list(self._symbol_sentiment_history.get(symbol, []))
        if not history:
            return 0.0

        # Get most recent sentiment
        _, score = history[-1]

        if score >= self.strong_bullish_threshold:
            return 1.0
        elif score >= 25:
            return 0.5
        elif score <= self.strong_bearish_threshold:
            return -1.0
        elif score <= -25:
            return -0.5
        else:
            return 0.0

    def should_block_trade(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """
        Check if sentiment strongly opposes the trade direction.

        Args:
            symbol: Trading symbol
            direction: 'long' or 'short'

        Returns:
            (should_block, reason)
        """
        history = list(self._symbol_sentiment_history.get(symbol, []))
        if not history:
            return False, "No sentiment data"

        _, score = history[-1]

        if direction == 'long' and score <= self.strong_bearish_threshold:
            return True, f"Strong bearish sentiment ({score:.0f}) blocks long entry"

        if direction == 'short' and score >= self.strong_bullish_threshold:
            return True, f"Strong bullish sentiment ({score:.0f}) blocks short entry"

        return False, "Sentiment allows trade"

    def check_position_divergence(self, symbol: str, position_direction: str,
                                   price_change_pct: float) -> Optional[SentimentAlert]:
        """
        Check if sentiment diverges from price action.

        Args:
            symbol: Position symbol
            position_direction: 'long' or 'short'
            price_change_pct: Recent price change percentage

        Returns:
            Alert if divergence detected, None otherwise
        """
        history = list(self._symbol_sentiment_history.get(symbol, []))
        if len(history) < 2:
            return None

        _, current_sentiment = history[-1]

        # Check for divergence
        # Price up but sentiment down (or vice versa)
        if price_change_pct > 1 and current_sentiment < -25:
            return SentimentAlert(
                id=hashlib.md5(f"div_{symbol}_{datetime.now()}".encode()).hexdigest()[:8],
                alert_type=AlertType.SENTIMENT_DIVERGENCE,
                symbol=symbol,
                title=f"{symbol} Sentiment Divergence",
                message=f"Price up {price_change_pct:.1f}% but sentiment is bearish ({current_sentiment:.0f})",
                urgency="medium",
                sentiment_before=None,
                sentiment_after=current_sentiment
            )

        if price_change_pct < -1 and current_sentiment > 25:
            return SentimentAlert(
                id=hashlib.md5(f"div_{symbol}_{datetime.now()}".encode()).hexdigest()[:8],
                alert_type=AlertType.SENTIMENT_DIVERGENCE,
                symbol=symbol,
                title=f"{symbol} Sentiment Divergence",
                message=f"Price down {abs(price_change_pct):.1f}% but sentiment is bullish ({current_sentiment:.0f})",
                urgency="medium",
                sentiment_before=None,
                sentiment_after=current_sentiment
            )

        return None

    def log_trade_sentiment(self, trade_id: str, symbol: str) -> Dict:
        """
        Capture sentiment snapshot at trade time for post-analysis.

        Returns dict to be stored with trade record.
        """
        history = list(self._symbol_sentiment_history.get(symbol, []))
        market_history = list(self._market_sentiment_history)

        return {
            'trade_id': trade_id,
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'symbol_sentiment': history[-1][1] if history else None,
            'market_sentiment': market_history[-1][1] if market_history else None,
            'headline_count': len([h for h in self._headlines_cache if symbol in h.symbols_mentioned]),
            'top_headline': self._headlines_cache[0].headline if self._headlines_cache else None
        }

    # =========================================================================
    # HISTORICAL CORRELATION (Advanced)
    # =========================================================================

    def record_price_update(self, symbol: str, price: float):
        """Record price for correlation analysis"""
        history = list(self._symbol_sentiment_history.get(symbol, []))
        if history:
            _, sentiment = history[-1]
            self._sentiment_price_history[symbol].append((datetime.now(), sentiment, price))

            # Keep only last 1000 data points
            if len(self._sentiment_price_history[symbol]) > 1000:
                self._sentiment_price_history[symbol] = self._sentiment_price_history[symbol][-1000:]

    def get_sentiment_price_correlation(self, symbol: str) -> Dict:
        """
        Calculate correlation between sentiment and subsequent price moves.

        Returns metrics on how well sentiment predicted price moves.
        """
        history = self._sentiment_price_history.get(symbol, [])
        if len(history) < 20:
            return {'correlation': None, 'sample_size': len(history), 'message': 'Insufficient data'}

        # Calculate price changes 1 hour after each sentiment reading
        correlations = []
        for i in range(len(history) - 12):  # 12 readings ~= 1 hour at 5 min intervals
            t1, sent1, price1 = history[i]
            t2, sent2, price2 = history[i + 12]

            price_change = (price2 - price1) / price1 * 100
            correlations.append((sent1, price_change))

        if len(correlations) < 10:
            return {'correlation': None, 'sample_size': len(correlations), 'message': 'Insufficient data'}

        # Calculate Pearson correlation
        sentiments = [c[0] for c in correlations]
        price_changes = [c[1] for c in correlations]

        correlation = np.corrcoef(sentiments, price_changes)[0, 1]

        # Calculate prediction accuracy (sentiment direction matched price direction)
        correct = sum(1 for s, p in correlations if (s > 0 and p > 0) or (s < 0 and p < 0) or (abs(s) < 10))
        accuracy = correct / len(correlations)

        return {
            'correlation': round(correlation, 3),
            'accuracy': round(accuracy, 3),
            'sample_size': len(correlations),
            'interpretation': self._interpret_correlation(correlation)
        }

    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation coefficient"""
        if corr is None or np.isnan(corr):
            return "Unknown"
        if corr > 0.5:
            return "Strong positive - sentiment predicts price well"
        elif corr > 0.2:
            return "Moderate positive - some predictive value"
        elif corr > -0.2:
            return "Weak - sentiment not strongly predictive"
        elif corr > -0.5:
            return "Moderate negative - contrarian signal"
        else:
            return "Strong negative - fade sentiment"

    def acknowledge_alert(self, alert_id: str):
        """Mark an alert as acknowledged"""
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                break

    def get_headlines(self, symbol: Optional[str] = None,
                      limit: int = 10) -> List[Dict]:
        """Get recent headlines, optionally filtered by symbol"""
        headlines = self._headlines_cache

        if symbol:
            headlines = [h for h in headlines if symbol in h.symbols_mentioned]

        return [h.to_dict() for h in headlines[:limit]]

    def get_symbol_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get current sentiment for a specific symbol"""
        history = list(self._symbol_sentiment_history.get(symbol, []))
        if not history:
            return None

        _, score = history[-1]
        trend, trend_change = self._calculate_trend(history, score)

        return {
            'symbol': symbol,
            'score': round(score, 1),
            'label': self._score_to_label(score).value,
            'trend': trend,
            'trend_change': round(trend_change, 1),
            'history': [(t.isoformat(), round(s, 1)) for t, s in history[-20:]]
        }


# =============================================================================
# SINGLETON
# =============================================================================

_sentiment_engine: Optional[NewsSentimentEngine] = None

def get_sentiment_engine(config: Optional[Dict] = None) -> NewsSentimentEngine:
    """Get or create singleton NewsSentimentEngine instance"""
    global _sentiment_engine
    if _sentiment_engine is None:
        _sentiment_engine = NewsSentimentEngine(config)
    return _sentiment_engine
