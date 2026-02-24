#!/usr/bin/env python3
"""Generate instruction-tuning JSONL from gap fade trade history and state snapshots.

Produces training examples in the Alpaca/ShareGPT format for LoRA fine-tuning:
  {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

Usage:
  python generate_training_data.py                    # from state file
  python generate_training_data.py --db               # from SQLite price DB + state
  python generate_training_data.py --output data.jsonl # custom output path

The generated JSONL can be used with:
  - Unsloth (fastest, recommended): unsloth.ai
  - Axolotl: github.com/OpenAccess-AI-Collective/axolotl
  - Ollama fine-tune (planned): ollama.com
"""

import json
import os
import sys
import random
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path

STATE_FILE = 'gap_fade_state.json'
OUTPUT_FILE = 'training_data.jsonl'

SYSTEM_PROMPT = (
    "You are the autonomous trading supervisor for a gap fade intraday short-selling bot. "
    "You analyze market conditions, positions, and candidates to make trading decisions. "
    "Respond with clear reasoning and a JSON action block when action is needed."
)


def load_state():
    """Load saved state from gap_fade_state.json."""
    if not os.path.exists(STATE_FILE):
        print(f"Error: {STATE_FILE} not found. Run the bot first to generate state.")
        sys.exit(1)
    with open(STATE_FILE) as f:
        return json.load(f)


def generate_entry_examples(trades: list) -> list:
    """Generate entry decision examples from completed trades."""
    examples = []
    for t in trades:
        if not t.get('entry_price') or not t.get('exit_price'):
            continue
        side = t.get('side', 'short')
        gap_pct = t.get('gap_pct', 0)
        vol_ratio = t.get('vol_ratio', 0)
        catalyst = t.get('catalyst', '')
        score = t.get('score', 0)
        pnl = t.get('pnl', 0)
        pnl_pct = t.get('pnl_pct', 0)
        was_winner = pnl > 0

        user = (
            f"Should I enter {t['symbol']}?\n"
            f"- Direction: {side}\n"
            f"- Gap: {gap_pct:.1%}\n"
            f"- Volume ratio: {vol_ratio:.2f}x\n"
            f"- Catalyst: {catalyst or 'none (noise gap)'}\n"
            f"- Score: {score:.0f}\n"
            f"- Current positions: 2/5"
        )

        if was_winner:
            assistant = (
                f"Yes, enter {t['symbol']} {side}. "
                f"Gap of {gap_pct:.1%} on low volume ({vol_ratio:.2f}x) with "
                f"{'no catalyst — clean noise gap, high fade probability' if not catalyst else catalyst + ' catalyst'}. "
                f"Score {score:.0f} is {'strong' if score > 25 else 'acceptable'}.\n\n"
                f"```action\n"
                f'{{"action": "enter", "symbols": ["{t["symbol"]}"]}}\n'
                f"```"
            )
        else:
            exit_reason = t.get('exit_reason', 'stop')
            assistant = (
                f"I would be cautious on {t['symbol']}. "
                f"{'The catalyst (' + catalyst + ') suggests this gap may not fade.' if catalyst in ('earnings', 'fda', 'ma') else ''}"
                f"{'Volume ratio of ' + f'{vol_ratio:.2f}x is elevated — less likely to fade.' if vol_ratio > 1.5 else ''}"
                f" Consider {'skipping' if not was_winner else 'entering with reduced size'}."
            )

        examples.append({
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': assistant.strip()},
            ],
            'metadata': {'type': 'entry_decision', 'symbol': t['symbol'], 'outcome': 'win' if was_winner else 'loss'}
        })
    return examples


def generate_exit_examples(trades: list) -> list:
    """Generate exit/profit-taking examples from completed trades."""
    examples = []
    for t in trades:
        pnl = t.get('pnl', 0)
        pnl_pct = t.get('pnl_pct', 0)
        exit_reason = t.get('exit_reason', 'unknown')
        side = t.get('side', 'short')
        holding_min = t.get('holding_minutes', 0)

        if exit_reason in ('stop', 'eod', 'time_exit', 'manual'):
            # These are mechanical exits — teach the LLM about them
            user = (
                f"Position check for {t['symbol']}:\n"
                f"- Direction: {side}, {t.get('shares', 0)} shares\n"
                f"- Entry: ${t['entry_price']:.2f}\n"
                f"- Current: ${t['exit_price']:.2f}\n"
                f"- P&L: {pnl_pct:+.1%} (${pnl:+,.2f})\n"
                f"- Time held: {holding_min} minutes\n"
                f"- Stop: ${t['entry_price'] * (1 + 0.015):.2f}"
            )

            if exit_reason == 'stop':
                assistant = (
                    f"Stop hit on {t['symbol']}. Price moved against us to ${t['exit_price']:.2f}, "
                    f"triggering the 1.5% stop loss. Loss of ${abs(pnl):.2f} ({pnl_pct:.1%}). "
                    f"This is a mechanical exit — respect the stop, don't override more than 2 times."
                )
            elif exit_reason == 'time_exit':
                assistant = (
                    f"Time exit on {t['symbol']} at 3:00 PM ET. "
                    f"Position held for {holding_min} minutes. "
                    f"{'Profit of' if pnl > 0 else 'Loss of'} ${abs(pnl):.2f} ({pnl_pct:+.1%}). "
                    f"Gap fades work best in the morning — afternoon holds rarely add value."
                )
            elif exit_reason == 'eod':
                assistant = (
                    f"EOD emergency close on {t['symbol']}. No overnight risk allowed. "
                    f"Final P&L: ${pnl:+,.2f} ({pnl_pct:+.1%})."
                )
            else:
                assistant = f"Closed {t['symbol']} manually. P&L: ${pnl:+,.2f} ({pnl_pct:+.1%})."

            examples.append({
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': user},
                    {'role': 'assistant', 'content': assistant},
                ],
                'metadata': {'type': 'exit_decision', 'reason': exit_reason, 'symbol': t['symbol']}
            })

        elif exit_reason in ('partial', 'full_target', 'llm_profit'):
            # Winning exits — teach profit-taking
            user = (
                f"Evaluate profit on {t['symbol']}:\n"
                f"- Direction: {side}, {t.get('shares', 0)} shares\n"
                f"- Entry: ${t['entry_price']:.2f}, Current: ${t['exit_price']:.2f}\n"
                f"- Unrealized P&L: {pnl_pct:+.1%} (${pnl:+,.2f})\n"
                f"- Time held: {holding_min} minutes"
            )

            if exit_reason == 'full_target':
                assistant = (
                    f"Full target hit on {t['symbol']}! Price faded back to previous close. "
                    f"Profit: ${pnl:+,.2f} ({pnl_pct:+.1%}). "
                    f"This is the ideal outcome — 100% gap fill."
                )
            elif exit_reason == 'partial':
                assistant = (
                    f"Partial cover on {t['symbol']} — covering 33% at 50% gap fill. "
                    f"Lock in ${pnl:+,.2f} on this tranche. "
                    f"Keep the remaining shares for the full target."
                )
            else:
                assistant = (
                    f"Taking profit on {t['symbol']}. "
                    f"P&L: ${pnl:+,.2f} ({pnl_pct:+.1%}) after {holding_min} minutes. "
                    f"{'Gap momentum is fading — better to lock gains.' if holding_min > 60 else 'Quick fade, nice trade.'}"
                )

            examples.append({
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': user},
                    {'role': 'assistant', 'content': assistant},
                ],
                'metadata': {'type': 'profit_taking', 'reason': exit_reason, 'symbol': t['symbol']}
            })

    return examples


def generate_risk_examples() -> list:
    """Generate risk management Q&A examples."""
    examples = []

    scenarios = [
        {
            'user': "I've lost 3 trades in a row. What should I do?",
            'assistant': (
                "Three consecutive losses triggers the standdown circuit breaker. "
                "The bot should stop opening new positions for the rest of the day. "
                "Let existing profitable positions run but don't add new risk. "
                "This is a normal part of the strategy — not every day has good gap fade setups.\n\n"
                "```action\n{\"action\": \"config\", \"params\": {\"max_positions\": 0}}\n```"
            )
        },
        {
            'user': "Daily P&L is -$450 on a $25,000 account. Should I keep trading?",
            'assistant': (
                "That's -1.8% of equity, approaching the 2% daily loss limit. "
                "I recommend pausing new entries. The daily_loss_limit is set to 2% ($500), "
                "so one more losing trade could trigger the automatic halt. "
                "Better to preserve capital and trade tomorrow.\n\n"
                "```action\n{\"action\": \"pause\"}\n```"
            )
        },
        {
            'user': "SPY is gapping up 2% today. Should we still trade?",
            'assistant': (
                "A 2% SPY gap-up creates a broad market tailwind that works AGAINST short gap fades. "
                "When the whole market is surging, individual gap-ups are less likely to fade. "
                "I recommend enabling the regime filter to block entries on days like this.\n\n"
                "```action\n{\"action\": \"config\", \"params\": {\"regime_filter\": true, \"regime_spy_gap_limit\": 0.01}}\n```"
            )
        },
        {
            'user': "The bot has been running for 2 hours with no entries. Is something wrong?",
            'assistant': (
                "Not necessarily. If no stocks meet the gap fade criteria (gap >= 7%, low volume, "
                "no catalyst), then it's correct to not trade. Forcing trades on bad setups is "
                "worse than sitting out. Let me check if there were candidates.\n\n"
                "```action\n{\"action\": \"scan\"}\n```"
            )
        },
        {
            'user': "What's the best stop loss percentage for this strategy?",
            'assistant': (
                "The default 1.5% stop works well for most gap fades based on backtesting. "
                "Key considerations:\n"
                "- 1.0%: Tighter, more stops triggered, but smaller losses per trade\n"
                "- 1.5%: Default, good balance of room-to-breathe vs risk\n"
                "- 2.0%: Wider, fewer stops but larger losses when wrong\n\n"
                "For adaptive stops (scales with gap size), enable `adaptive_stops` — "
                "bigger gaps get wider stops since they have more room to chop."
            )
        },
    ]

    for s in scenarios:
        examples.append({
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': s['user']},
                {'role': 'assistant', 'content': s['assistant']},
            ],
            'metadata': {'type': 'risk_management'}
        })
    return examples


def generate_api_examples() -> list:
    """Generate API usage examples."""
    examples = []

    commands = [
        ("Start the bot", "Starting the trading bot. It will begin scanning for gap fade candidates at the next scheduled time.\n\n```action\n{\"action\": \"start\"}\n```"),
        ("Stop everything and close all positions", "Stopping the bot and closing all open positions immediately. All broker-side stop orders will be cancelled and positions covered at market.\n\n```action\n{\"action\": \"stop\"}\n```"),
        ("Scan for new candidates", "Running a fresh scan across all tradeable symbols to find new gap fade candidates.\n\n```action\n{\"action\": \"scan\"}\n```"),
        ("Enter TSLA and NVDA with half size", "Entering TSLA and NVDA with 50% position size (reduced conviction).\n\n```action\n{\"action\": \"enter\", \"symbols\": [\"TSLA\", \"NVDA\"], \"size_mult\": 0.5}\n```"),
        ("Set max positions to 3", "Reducing max simultaneous positions to 3. This limits exposure and keeps the portfolio more concentrated.\n\n```action\n{\"action\": \"config\", \"params\": {\"max_positions\": 3}}\n```"),
        ("Enable adaptive stops and set gap fraction to 0.2", "Enabling adaptive stops — stops will scale with gap size. A 20% gap fraction means the stop is set at 20% of the gap distance above entry.\n\n```action\n{\"action\": \"config\", \"params\": {\"adaptive_stops\": true, \"stop_gap_fraction\": 0.2}}\n```"),
        ("Pause trading but keep my positions", "Pausing the bot — no new trades will be opened, but existing positions will continue to be monitored with stops intact.\n\n```action\n{\"action\": \"pause\"}\n```"),
        ("Enable re-entry after stop-outs with 15 min cooldown", "Enabling re-entry: if a position gets stopped out, the bot will re-enter after a 15-minute cooldown if the price moves favorably.\n\n```action\n{\"action\": \"config\", \"params\": {\"reentry_enabled\": true, \"reentry_cooldown_minutes\": 15}}\n```"),
    ]

    for user, assistant in commands:
        examples.append({
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user},
                {'role': 'assistant', 'content': assistant},
            ],
            'metadata': {'type': 'api_usage'}
        })
    return examples


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate training data for gap fade LLM')
    parser.add_argument('--output', default=OUTPUT_FILE, help='Output JSONL file')
    parser.add_argument('--db', action='store_true', help='Also pull from SQLite price DB')
    args = parser.parse_args()

    state = load_state()
    all_examples = []

    # From trade history
    trades = state.get('trade_log', []) + state.get('all_trades', [])
    print(f"Loaded {len(trades)} trades from state file")

    if trades:
        entry_ex = generate_entry_examples(trades)
        exit_ex = generate_exit_examples(trades)
        all_examples.extend(entry_ex)
        all_examples.extend(exit_ex)
        print(f"  Entry decision examples: {len(entry_ex)}")
        print(f"  Exit decision examples: {len(exit_ex)}")

    # Static examples (always included)
    risk_ex = generate_risk_examples()
    api_ex = generate_api_examples()
    all_examples.extend(risk_ex)
    all_examples.extend(api_ex)
    print(f"  Risk management examples: {len(risk_ex)}")
    print(f"  API usage examples: {len(api_ex)}")

    # Shuffle for training
    random.shuffle(all_examples)

    # Write JSONL
    with open(args.output, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\nWrote {len(all_examples)} training examples to {args.output}")
    print(f"\nNext steps for fine-tuning:")
    print(f"  1. Install unsloth: pip install unsloth")
    print(f"  2. Fine-tune: python finetune_gapfade.py --data {args.output}")
    print(f"  3. Convert to GGUF and import to Ollama")
    print(f"  Or wait for more trades to accumulate — more data = better model")


if __name__ == '__main__':
    main()
