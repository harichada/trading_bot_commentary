import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from core.models import CommentaryType
from core.commentary import TradingCommentary

logger = logging.getLogger('TradingBot')

@dataclass
class TradingMemory:
    """Represents a trading memory/lesson learned"""
    timestamp: datetime
    symbol: str
    pattern: str
    outcome: str
    lesson: str
    confidence_adjustment: float
    context: Dict[str, Any]

class TradingBrain:
    """Persistent memory and learning system - acts like a human trader's experience"""

    def __init__(self, memory_path: Path = Path("trading_brain.json")):
        self.memory_path = memory_path
        self.memories: List[TradingMemory] = []
        self.pattern_success_rates: Dict[str, Dict[str, float]] = {}
        self.symbol_behaviors: Dict[str, Dict[str, Any]] = {}
        self.emotional_state = {
            'confidence': 0.7,
            'risk_appetite': 0.5,
            'patience': 0.8,
            'fear_level': 0.2,
            'greed_level': 0.3,
            'last_update': datetime.now()
        }
        self._load_memories()

    def reset_for_new_session(self):
        """v-brain-session-reset-2026-05-19: pull emotional state 60%
        toward neutral (0.5) at the start of each RTH day.

        Without this, yesterday's pain (or yesterday's hot streak) dictates
        today's behavior for the first few hours of trading. Called from
        engine.py startup AFTER memories are loaded but BEFORE the first
        trading tick. Confidence is left alone — that's a slower-moving
        skill estimate, not a session-emotional state. last_update is NOT
        touched so `recover_confidence` still computes time-since-last-trade
        correctly.
        """
        before = {k: self.emotional_state.get(k) for k in ('greed_level', 'fear_level')}
        for key in ('greed_level', 'fear_level'):
            current = self.emotional_state.get(key, 0.5)
            self.emotional_state[key] = current + 0.60 * (0.5 - current)
        after = {k: round(self.emotional_state[k], 3) for k in ('greed_level', 'fear_level')}
        logger.info(
            "brain_session_reset before=%s after=%s confidence=%.3f",
            before, after, self.emotional_state.get('confidence', 0.5),
        )

    def recover_confidence(self):
        """Gradually recover confidence AND decay extreme emotional states toward neutral.

        v-brain-decay-2026-05-19: previously only confidence and fear decayed;
        greed_level had no decay path and would lock to 0.8 once tripped.
        Now ALL emotional states drift toward neutral (0.5) over time so
        a single hot streak or losing streak doesn't dictate behavior for
        the rest of the session/week.
        """
        # Handle both string and datetime objects
        if isinstance(self.emotional_state['last_update'], str):
            last_update = datetime.fromisoformat(self.emotional_state['last_update'])
        else:
            last_update = self.emotional_state['last_update']

        time_since_update_s = (datetime.now() - last_update).total_seconds()

        # Recover 1% confidence per hour of no trading
        if time_since_update_s > 3600:
            hours = time_since_update_s / 3600.0
            recovery = hours * 0.01
            self.emotional_state['confidence'] = min(0.7, self.emotional_state['confidence'] + recovery)
            # v-brain-decay-2026-05-19: per-hour 20% drift toward neutral 0.5
            # for greed AND fear. Half-life ~3.1 hours — quick enough to
            # clear an overnight elevated state, slow enough that a fresh
            # win/loss still moves the needle within a session.
            decay_per_hour = 0.20
            for key in ('fear_level', 'greed_level'):
                current = self.emotional_state.get(key, 0.5)
                step = min(1.0, decay_per_hour * hours)
                self.emotional_state[key] = current + step * (0.5 - current)
            self.emotional_state['last_update'] = datetime.now().isoformat()

    def _load_memories(self):
        """Load memories from disk"""
        if self.memory_path.exists():
            try:
                with open(self.memory_path, 'r') as f:
                    data = json.load(f)
                    self.pattern_success_rates = data.get('pattern_success_rates', {})
                    self.symbol_behaviors = data.get('symbol_behaviors', {})
                    self.emotional_state = data.get('emotional_state', self.emotional_state)

                    # Reconstruct memories
                    for mem_data in data.get('memories', []):
                        self.memories.append(TradingMemory(
                            timestamp=datetime.fromisoformat(mem_data['timestamp']),
                            symbol=mem_data['symbol'],
                            pattern=mem_data['pattern'],
                            outcome=mem_data['outcome'],
                            lesson=mem_data['lesson'],
                            confidence_adjustment=mem_data['confidence_adjustment'],
                            context=mem_data['context']
                        ))
                logger.info(f"Loaded {len(self.memories)} trading memories")
            except Exception as e:
                logger.error(f"Error loading memories: {e}")

    def save_memories(self):
        """Save memories to disk"""
        self.emotional_state['last_update'] = self.emotional_state['last_update'].isoformat() if isinstance(self.emotional_state['last_update'], datetime) else self.emotional_state['last_update']
        data = {
            'memories': [
                {
                    'timestamp': mem.timestamp.isoformat(),
                    'symbol': mem.symbol,
                    'pattern': mem.pattern,
                    'outcome': mem.outcome,
                    'lesson': mem.lesson,
                    'confidence_adjustment': mem.confidence_adjustment,
                    'context': mem.context
                }
                for mem in self.memories[-1000:]  # Keep last 1000 memories
            ],
            'pattern_success_rates': self.pattern_success_rates,
            'symbol_behaviors': self.symbol_behaviors,
            'emotional_state': self.emotional_state
        }

        with open(self.memory_path, 'w') as f:
            json.dump(data, f, indent=2)

    def remember_trade(self, symbol: str, pattern: str, outcome: str,
                      pnl_percent: float, context: Dict[str, Any]):
        """Remember a trade outcome and learn from it"""
        # Update pattern success rates
        if pattern not in self.pattern_success_rates:
            self.pattern_success_rates[pattern] = {'wins': 0, 'losses': 0, 'total': 0}

        self.pattern_success_rates[pattern]['total'] += 1
        if pnl_percent > 0:
            self.pattern_success_rates[pattern]['wins'] += 1
        else:
            self.pattern_success_rates[pattern]['losses'] += 1

        # Update symbol-specific behaviors
        if symbol not in self.symbol_behaviors:
            self.symbol_behaviors[symbol] = {
                'avg_holding_time': 0,
                'best_exit_timing': {},
                'volatility_profile': {},
                'success_rate': 0.5
            }

        # Generate human-like lesson
        lesson = self._generate_lesson(symbol, pattern, outcome, pnl_percent, context)

        # Adjust emotional state based on outcome
        self._adjust_emotional_state(pnl_percent)

        # Create memory
        memory = TradingMemory(
            timestamp=datetime.now(),
            symbol=symbol,
            pattern=pattern,
            outcome=outcome,
            lesson=lesson,
            confidence_adjustment=self._calculate_confidence_adjustment(pnl_percent),
            context=context
        )

        self.memories.append(memory)
        self.save_memories()

        return memory

    def _generate_lesson(self, symbol: str, pattern: str, outcome: str,
                        pnl_percent: float, context: Dict[str, Any]) -> str:
        """Generate human-like lessons from trades"""
        if pnl_percent > 2:
            lessons = [
                f"Nice! {symbol} responds well to {pattern} patterns. Should have held longer though...",
                f"Good call on {symbol}. Next time, use a trailing stop to capture more upside.",
                f"{symbol} momentum was stronger than expected. Mental note: be more patient with winners."
            ]
        elif pnl_percent > 0:
            lessons = [
                f"Small win on {symbol}, but a win is a win. Maybe tighten the entry next time.",
                f"Profitable, but left money on the table. {symbol} had more room to run.",
                f"Decent trade. Should trust my analysis more when the setup is this clean."
            ]
        elif pnl_percent > -1:
            lessons = [
                f"Small loss on {symbol}. The setup was right but timing was off.",
                f"Stopped out for a small loss. Good risk management, bad entry timing.",
                f"{symbol} didn't follow through. Need to watch for false breakouts here."
            ]
        else:
            lessons = [
                f"Ouch. {symbol} completely reversed. Need to respect the trend more.",
                f"Bad read on {symbol}. Market conditions changed - should've adapted.",
                f"Lesson learned: don't fight the tape on {symbol} when volume is weak."
            ]

        return np.random.choice(lessons)

    def _adjust_emotional_state(self, pnl_percent: float):
        """Adjust emotional state based on trade outcomes.

        v-brain-semantics-2026-05-19: previously winners INCREASED greed
        (winners → +0.05 greed), which is backwards for a trading bot —
        the goal is to avoid chasing returns, not to celebrate them.
        Combined with no decay this locked the bot into all-day veto mode
        after a single win. Corrected: winners REDUCE greed (we banked
        the win, no need to chase) and slightly reduce fear; losers raise
        fear but do NOT raise greed (revenge trading is the trap, not a
        feature to model).
        """
        # Confidence adjustment
        if pnl_percent > 0:
            self.emotional_state['confidence'] = min(0.95, self.emotional_state['confidence'] + 0.02)
            # FIXED: winners reduce greed (banked the win) and reduce fear
            self.emotional_state['greed_level'] = max(0.2, self.emotional_state['greed_level'] - 0.05)
            self.emotional_state['fear_level'] = max(0.2, self.emotional_state['fear_level'] - 0.02)
        else:
            self.emotional_state['confidence'] = max(0.3, self.emotional_state['confidence'] - 0.03)
            self.emotional_state['fear_level'] = min(0.8, self.emotional_state['fear_level'] + 0.05)
            # Losers do NOT raise greed — revenge trading is the trap.

        # Patience adjustment based on recent outcomes
        recent_trades = self.memories[-5:] if len(self.memories) >= 5 else self.memories
        if recent_trades:
            avg_outcome = np.mean([m.confidence_adjustment for m in recent_trades])
            if avg_outcome < -0.02:
                self.emotional_state['patience'] = min(0.95, self.emotional_state['patience'] + 0.1)
                self.emotional_state['risk_appetite'] = max(0.2, self.emotional_state['risk_appetite'] - 0.1)

        self.emotional_state['last_update'] = datetime.now()

    def _calculate_confidence_adjustment(self, pnl_percent: float) -> float:
        """Calculate how much to adjust confidence based on outcome"""
        if pnl_percent > 3:
            return 0.05
        elif pnl_percent > 1:
            return 0.02
        elif pnl_percent > 0:
            return 0.01
        elif pnl_percent > -1:
            return -0.01
        elif pnl_percent > -2:
            return -0.02
        else:
            return -0.05

    def get_pattern_confidence(self, pattern: str) -> float:
        """Get confidence in a pattern based on history"""
        if pattern not in self.pattern_success_rates:
            return 0.5

        stats = self.pattern_success_rates[pattern]
        if stats['total'] < 3:
            return 0.5  # Not enough data

        win_rate = stats['wins'] / stats['total']
        # Adjust by emotional state
        return win_rate * self.emotional_state['confidence']

    def should_take_trade(self, symbol: str, pattern: str, base_confidence: float) -> Tuple[bool, str]:
        """Decide if we should take a trade based on memory and emotional state"""
        # Recover confidence gradually
        self.recover_confidence()

        # Lower threshold during low confidence periods
        if self.emotional_state['confidence'] < 0.5:
            threshold = 0.3  # Much lower bar
        else:
            threshold = 0.6 * self.emotional_state['confidence']
        pattern_confidence = self.get_pattern_confidence(pattern)

        # Check recent memories for this symbol
        recent_symbol_memories = [m for m in self.memories[-20:] if m.symbol == symbol]
        if recent_symbol_memories:
            recent_success = sum(1 for m in recent_symbol_memories if 'win' in m.outcome.lower())
            recent_rate = recent_success / len(recent_symbol_memories)

            if recent_rate < 0.2 and len(recent_symbol_memories) >= 3:
                return False, f"I've been wrong on {symbol} lately. Sitting this one out."

        # Emotional state check
        if self.emotional_state['fear_level'] > 0.9:
            if base_confidence < 0.6:
                return False, "I'm feeling cautious after recent losses. Need a stronger setup."

        if self.emotional_state['greed_level'] > 0.7:
            if base_confidence < 0.6:
                return False, "Getting greedy here. Need to calm down and wait for quality setups."

        # Confidence threshold
        combined_confidence = (base_confidence + pattern_confidence) / 2
        threshold = 0.6 * self.emotional_state['confidence']

        if combined_confidence < threshold:
            return False, f"Not confident enough. Setup confidence: {combined_confidence:.1%}"

        return True, "Looks good based on my experience"
