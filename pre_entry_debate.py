"""
Rudra Pre-Entry Debate — Bull/Bear analysis before every trade.

Uses Claude API (via Anthropic SDK) for high-quality reasoning,
falls back to local Ollama if no API key is set.

Before any trade fires, this module runs a 3-step debate:
1. Bull case: strongest argument FOR the trade
2. Bear case: strongest argument AGAINST the trade
3. Verdict: PROCEED or REJECT with reasoning

Cost: ~$0.01-0.03 per evaluation using Claude Haiku.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger('PreEntryDebate')


@dataclass
class DebateResult:
    bull_case: str
    bear_case: str
    verdict: str
    approved: bool
    confidence: float  # 0-1
    model_used: str
    latency_ms: float


class PreEntryDebate:
    """Run bull/bear stress test before order execution."""

    ANTHROPIC_API = 'https://api.anthropic.com/v1/messages'
    OLLAMA_API = 'http://localhost:11434/api/chat'

    def __init__(self):
        self.anthropic_key = os.environ.get('ANTHROPIC_API_KEY', '')
        self.ollama_model = os.environ.get('LLM_MODEL', 'gpt-oss:20b')
        self.enabled = True
        self._cache = {}  # symbol -> (result, timestamp)
        self._CACHE_TTL = 300  # 5 min cache per symbol

    @property
    def has_claude(self) -> bool:
        return bool(self.anthropic_key)

    async def evaluate(self, symbol: str, setup: dict) -> DebateResult:
        """Run the full bull/bear debate on a trade setup.

        Args:
            symbol: Stock ticker
            setup: Dict with keys like:
                gap_pct, direction, entry_price, stop_price, or_high,
                prev_close, vol_ratio, strategy_id, regime, etc.

        Returns:
            DebateResult with approved=True/False
        """
        # Check cache
        cached = self._cache.get(symbol)
        if cached and time.time() - cached[1] < self._CACHE_TTL:
            return cached[0]

        if not self.enabled:
            return DebateResult('', '', 'PROCEED (debate disabled)', True, 1.0, 'none', 0)

        t0 = time.time()
        context = self._build_context(symbol, setup)

        try:
            if self.has_claude:
                result = await self._run_claude(context)
            else:
                result = await self._run_ollama(context)
        except Exception as e:
            logger.warning(f"Pre-entry debate failed for {symbol}: {e}")
            # On failure, default to PROCEED (don't block trades on debate errors)
            result = DebateResult(
                bull_case='Debate unavailable',
                bear_case='Debate unavailable',
                verdict='PROCEED (debate error)',
                approved=True,
                confidence=0.5,
                model_used='error',
                latency_ms=(time.time() - t0) * 1000,
            )

        result.latency_ms = (time.time() - t0) * 1000
        self._cache[symbol] = (result, time.time())
        return result

    def _build_context(self, symbol: str, setup: dict) -> str:
        parts = [f"Symbol: {symbol}"]
        for key in ['direction', 'gap_pct', 'entry_price', 'stop_price', 'or_high',
                     'prev_close', 'vol_ratio', 'strategy_id', 'regime']:
            if key in setup:
                val = setup[key]
                if isinstance(val, float):
                    val = f"{val:.4f}" if 'pct' in key else f"{val:.2f}"
                parts.append(f"{key}: {val}")
        return '\n'.join(parts)

    async def _run_claude(self, context: str) -> DebateResult:
        """Run debate via Anthropic Claude API (Haiku for speed/cost)."""
        prompt = f"""You are a trading risk analyst. A gap fade bot wants to enter this trade:

{context}

Respond in EXACTLY this JSON format:
{{"bull": "2 sentence bull case FOR the trade", "bear": "2 sentence bear case AGAINST the trade", "verdict": "PROCEED or REJECT", "confidence": 0.0-1.0, "reason": "1 sentence"}}

Be concise. Focus on: is the gap likely to fade? Is the risk/reward favorable? Any red flags?"""

        headers = {
            'x-api-key': self.anthropic_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        }
        body = {
            'model': 'claude-haiku-4-5-20251001',
            'max_tokens': 200,
            'messages': [{'role': 'user', 'content': prompt}],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.ANTHROPIC_API, headers=headers,
                                    json=body, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise Exception(f"Claude API {resp.status}: {text[:200]}")
                data = await resp.json()

        reply = data.get('content', [{}])[0].get('text', '{}')
        try:
            parsed = json.loads(reply)
        except json.JSONDecodeError:
            # Try to extract verdict from free text
            approved = 'PROCEED' in reply.upper()
            return DebateResult(reply, '', reply, approved, 0.5, 'claude-haiku', 0)

        return DebateResult(
            bull_case=parsed.get('bull', ''),
            bear_case=parsed.get('bear', ''),
            verdict=f"{parsed.get('verdict', 'PROCEED')}: {parsed.get('reason', '')}",
            approved=parsed.get('verdict', 'PROCEED').upper().startswith('PROCEED'),
            confidence=float(parsed.get('confidence', 0.5)),
            model_used='claude-haiku',
            latency_ms=0,
        )

    async def _run_ollama(self, context: str) -> DebateResult:
        """Run debate via local Ollama (fallback)."""
        prompt = f"""You are a trading risk analyst. A gap fade bot wants to enter this trade:

{context}

Give a brief bull case (FOR), bear case (AGAINST), and verdict (PROCEED or REJECT).
Keep each to 1-2 sentences. End with: VERDICT: PROCEED or VERDICT: REJECT"""

        body = {
            'model': self.ollama_model,
            'messages': [{'role': 'user', 'content': prompt}],
            'stream': False,
            'options': {'temperature': 0.3, 'num_predict': 200},
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.OLLAMA_API, json=body,
                                    timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    raise Exception(f"Ollama {resp.status}")
                data = await resp.json()

        reply = data.get('message', {}).get('content', '')
        approved = 'VERDICT: PROCEED' in reply.upper() or 'PROCEED' in reply.upper()

        return DebateResult(
            bull_case=reply,
            bear_case='',
            verdict='PROCEED' if approved else 'REJECT',
            approved=approved,
            confidence=0.6 if approved else 0.4,
            model_used=f'ollama:{self.ollama_model}',
            latency_ms=0,
        )


# CLI test
if __name__ == '__main__':
    import sys

    async def test():
        debate = PreEntryDebate()
        print(f"Claude available: {debate.has_claude}")
        print(f"Model: {'claude-haiku' if debate.has_claude else debate.ollama_model}")

        result = await debate.evaluate('BHVN', {
            'direction': 'short',
            'gap_pct': 0.071,
            'entry_price': 9.09,
            'stop_price': 9.17,
            'or_high': 9.15,
            'prev_close': 8.49,
            'vol_ratio': 1.2,
            'strategy_id': 'exhaustion_gap_fade',
        })

        print(f"\nBull: {result.bull_case}")
        print(f"Bear: {result.bear_case}")
        print(f"Verdict: {result.verdict}")
        print(f"Approved: {result.approved}")
        print(f"Confidence: {result.confidence}")
        print(f"Model: {result.model_used}")
        print(f"Latency: {result.latency_ms:.0f}ms")

    asyncio.run(test())
