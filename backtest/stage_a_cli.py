#!/usr/bin/env python3
"""
Backtest CLI for Trading Bot Commentary

Usage:
    python -m backtest.stage_a_cli stage-a --lane mean_reversion --direction LONG
    python -m backtest.stage_a_cli stage-a --lane day_trade_momentum --direction SHORT
    python -m backtest.stage_a_cli stage-a --all
    python -m backtest.stage_a_cli scorecard --output research/reports/

Data Sources (priority order):
    1. Postgres minute_bars (if POSTGRES_DSN set)
    2. Shadow ledger NDJSON files (if path exists)
    3. yfinance fallback (for demonstration only - marks INCOMPLETE)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.stage_a_scorer import (
    StageAMetrics,
    compute_stage_a_from_trades,
    create_incomplete_metrics,
    check_floors,
    STAGE_A_FLOORS,
    HANDS_OFF_SYMBOLS,
)


LANES = {
    'mean_reversion_long': {
        'strategy_id': 'mean_reversion',
        'setup_type': 'oversold_v2',
        'direction': 'LONG',
        'shadow_ledger': 'shadow_long_log.ndjson',
    },
    'mean_reversion_short': {
        'strategy_id': 'mean_reversion',
        'setup_type': 'overbought_fade',
        'direction': 'SHORT',
        'shadow_ledger': 'shadow_short_log.ndjson',
    },
    'day_trade_momentum_long': {
        'strategy_id': 'momentum',
        'setup_type': 'day_trade_continuation',
        'direction': 'LONG',
        'shadow_ledger': None,
    },
    'day_trade_momentum_short': {
        'strategy_id': 'momentum',
        'setup_type': 'day_trade_continuation',
        'direction': 'SHORT',
        'shadow_ledger': 'day_trade_short_shadow.ndjson',
    },
    'orb': {
        'strategy_id': 'orb',
        'setup_type': 'orb_15',
        'direction': 'LONG',
        'shadow_ledger': None,
    },
}


def check_postgres_available() -> Optional[str]:
    """Check if Postgres DSN is available."""
    dsn = os.environ.get('POSTGRES_DSN') or os.environ.get('DATABASE_URL')
    if dsn:
        try:
            import psycopg2
            conn = psycopg2.connect(dsn)
            conn.close()
            return dsn
        except Exception:
            pass
    return None


def check_shadow_ledger(lane_config: Dict) -> Optional[Path]:
    """Check if shadow ledger file exists."""
    ledger_name = lane_config.get('shadow_ledger')
    if not ledger_name:
        return None
    
    search_paths = [
        Path('/workspace') / ledger_name,
        Path('/workspace/data') / ledger_name,
        Path.home() / 'claude' / 'trading_bot_commentary' / ledger_name,
        Path.home() / 'claude' / 'grok' / 'trading_bot_commentary' / ledger_name,
    ]
    
    for path in search_paths:
        if path.exists():
            return path
    
    return None


def load_shadow_ledger(path: Path) -> pd.DataFrame:
    """Load and parse shadow ledger NDJSON."""
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(records)


def check_strategy_implemented(lane_id: str) -> bool:
    """Check if strategy is implemented in the codebase."""
    strategies_implemented = {
        'mean_reversion_long': True,
        'mean_reversion_short': True,
        'day_trade_momentum_long': True,
        'day_trade_momentum_short': True,
        'orb': False,
    }
    return strategies_implemented.get(lane_id, False)


def run_stage_a_for_lane(lane_id: str) -> StageAMetrics:
    """
    Run Stage A evaluation for a single lane.
    Returns metrics with appropriate verdict.
    """
    if lane_id not in LANES:
        return create_incomplete_metrics(
            lane=lane_id,
            direction='UNKNOWN',
            reason=f"Unknown lane: {lane_id}"
        )
    
    config = LANES[lane_id]
    direction = config['direction']
    
    if not check_strategy_implemented(lane_id):
        return create_incomplete_metrics(
            lane=lane_id,
            direction=direction,
            reason=f"Strategy not implemented in codebase. Engine ask: implement {config['strategy_id']} {direction} in load_strategies."
        )
    
    postgres_dsn = check_postgres_available()
    shadow_path = check_shadow_ledger(config)
    
    if postgres_dsn:
        return create_incomplete_metrics(
            lane=lane_id,
            direction=direction,
            reason="Postgres available but minute_bars query not implemented. Engine ask: add resolved trade query."
        )
    
    if shadow_path:
        try:
            df = load_shadow_ledger(shadow_path)
            df = df[~df['symbol'].isin(HANDS_OFF_SYMBOLS)]
            
            return create_incomplete_metrics(
                lane=lane_id,
                direction=direction,
                reason=f"Shadow ledger found ({len(df)} rows) but resolver not implemented. "
                       f"Engine ask: implement barrier resolver for {lane_id}. "
                       f"File: {shadow_path}"
            )
        except Exception as e:
            return create_incomplete_metrics(
                lane=lane_id,
                direction=direction,
                reason=f"Shadow ledger load failed: {e}"
            )
    
    return create_incomplete_metrics(
        lane=lane_id,
        direction=direction,
        reason=f"No data source available. Need: Postgres minute_bars or shadow ledger ({config.get('shadow_ledger', 'N/A')})"
    )


def get_historical_metrics(lane_id: str) -> Optional[StageAMetrics]:
    """
    Return historical metrics from prior Research briefs where available.
    These are documented NO_GO / INCOMPLETE results from actual measurements.
    """
    if lane_id == 'mean_reversion_short':
        metrics = StageAMetrics(
            lane='mean_reversion_short',
            direction='SHORT',
            n=88,
            sessions=10,
            pf=0.779,
            wr=0.421,
            exp_r=-0.11,
            max_dd_pct=0.0,
            max_losing_day_r=13.12,
            verdict="FAIL",
            recommendation="LIVE off",
            notes="Historical shadow resolve (2026-09-11 brief). "
                  "5 floor failures. Flags stay OFF: ENABLE_MEAN_REV_SHORT, ENABLE_SHORT_MIRRORS. "
                  "Prior live short clusters ~PF 0.69. Shadow soak only until improvement.",
            risk_off_excluded=True,
            sample_start="2026-06-11",
            sample_end="2026-09-08"
        )
        _, failures = check_floors(metrics)
        metrics.floor_failures = failures
        return metrics
    
    elif lane_id == 'mean_reversion_long':
        return StageAMetrics(
            lane='mean_reversion_long',
            direction='LONG',
            n=0,
            sessions=0,
            pf=0.0,
            wr=0.0,
            exp_r=0.0,
            max_dd_pct=0.0,
            max_losing_day_r=0.0,
            verdict="INCOMPLETE",
            recommendation="LIVE off",
            notes="LONG resolver missing (2026-09-17 brief). "
                  "Shadow ledger emits exist (MEAN_REV_SHADOW_LEDGER_ENABLED) but no barrier-resolved sample. "
                  "Engine ask: (1) confirm shadow ledger emits on KiddoKingdom for mean_rev_buy/oversold_v2, "
                  "(2) implement LONG barrier resolver (or shared resolver with side=LONG), "
                  "(3) resolve sample then apply Stage A floors. "
                  "Currently LIVE OFF.",
            risk_off_excluded=True
        )
    
    elif lane_id == 'day_trade_momentum_long':
        return StageAMetrics(
            lane='day_trade_momentum_long',
            direction='LONG',
            n=0,
            sessions=0,
            pf=0.0,
            wr=0.0,
            exp_r=0.0,
            max_dd_pct=0.0,
            max_losing_day_r=0.0,
            verdict="INCOMPLETE",
            recommendation="LIVE on (baseline)",
            notes="Currently LIVE (baseline for keep/kill). "
                  "Strategy implemented (MomentumStrategyWithCommentary). "
                  "No resolved Stage A backtest sample in this harness - "
                  "need Postgres minute_bars or shadow ledger with setup_type=momentum tagging. "
                  "Engine ask: emit setup_type=momentum, session_id, regime tags for Stage A audit trail.",
            risk_off_excluded=True
        )
    
    elif lane_id == 'day_trade_momentum_short':
        return StageAMetrics(
            lane='day_trade_momentum_short',
            direction='SHORT',
            n=0,
            sessions=0,
            pf=0.0,
            wr=0.0,
            exp_r=0.0,
            max_dd_pct=0.0,
            max_losing_day_r=0.0,
            verdict="INCOMPLETE",
            recommendation="LIVE off",
            notes="Shadow + resolver ready (#83 shadow, v-day-trade-short-resolver-2026-09-17). "
                  "DayTradeMomentumShortStrategy emits to data/day_trade_short_shadow.ndjson. "
                  "Resolver: python -m research.day_trade_short_resolver --log data/day_trade_short_shadow.ndjson "
                  "--bars-dir <path> --out /tmp/day_trade_short_results/. "
                  "Need shadow sample to accumulate (n>=150 or sessions>=10) before scoring.",
            risk_off_excluded=True
        )
    
    elif lane_id == 'orb':
        return StageAMetrics(
            lane='orb',
            direction='LONG',
            n=0,
            sessions=0,
            pf=0.0,
            wr=0.0,
            exp_r=0.0,
            max_dd_pct=0.0,
            max_losing_day_r=0.0,
            verdict="INCOMPLETE",
            recommendation="LIVE off",
            notes="ORB NOT WIRED in load_strategies (2026-09-10 brief confirms design only). "
                  "No ORB strategy class in trading_bot_commentary_updated.py. "
                  "Engine ask: implement ORBStrategy class with setup_type=orb/orb_15/orb_30, "
                  "or_width_atr, contraction filter, session_id tagging. "
                  "Then wire into self.strategies list. "
                  "Stage A cannot proceed until implementation exists.",
            risk_off_excluded=True
        )
    
    return None


def generate_scorecard(output_dir: Path) -> Dict:
    """Generate unified Stage A scorecard for all lanes."""
    scorecard = {
        'generated_at': datetime.now().isoformat(),
        'stage_a_floors': STAGE_A_FLOORS,
        'hands_off_symbols': list(HANDS_OFF_SYMBOLS),
        'lanes': {}
    }
    
    priority_order = [
        'mean_reversion_long',
        'day_trade_momentum_short',
        'day_trade_momentum_long',
        'mean_reversion_short',
        'orb',
    ]
    
    for lane_id in priority_order:
        historical = get_historical_metrics(lane_id)
        if historical:
            metrics = historical
        else:
            metrics = run_stage_a_for_lane(lane_id)
        
        scorecard['lanes'][lane_id] = metrics.to_dict()
    
    return scorecard


def write_markdown_scorecard(scorecard: Dict, output_path: Path):
    """Write scorecard as formatted markdown."""
    lines = [
        "# Stage A Backtest Scorecard",
        f"**Generated:** {scorecard['generated_at'][:10]}",
        "",
        "## Summary",
        "",
        "| Lane | Direction | Verdict | Recommend |",
        "|------|-----------|---------|-----------|",
    ]
    
    for lane_id, metrics in scorecard['lanes'].items():
        lines.append(f"| {lane_id} | {metrics['direction']} | **{metrics['verdict']}** | {metrics['recommendation']} |")
    
    lines.extend([
        "",
        "---",
        "",
        "## Locked Stage A Floors (DO NOT SOFTEN)",
        "",
        "| Metric | Floor |",
        "|--------|-------|",
        f"| n (closed trades) | ≥ {STAGE_A_FLOORS['min_n']} |",
        f"| sessions | ≥ {STAGE_A_FLOORS['min_sessions']} |",
        f"| PF (fees+slip) | ≥ {STAGE_A_FLOORS['min_pf']:.2f} |",
        f"| WR (scratches out) | ≥ {STAGE_A_FLOORS['min_wr']*100:.0f}% |",
        f"| exp_R | ≥ +{STAGE_A_FLOORS['min_exp_r']:.2f} |",
        f"| maxDD | ≤ {STAGE_A_FLOORS['max_dd_pct']*100:.0f}% allocated |",
        f"| max_losing_day | ≤ {STAGE_A_FLOORS['max_losing_day_r']:.1f}R |",
        "",
        f"**Hands-off symbols (excluded):** {', '.join(sorted(HANDS_OFF_SYMBOLS))}",
        "",
        "---",
        "",
        "## Lane Details",
        "",
    ])
    
    for i, (lane_id, metrics) in enumerate(scorecard['lanes'].items(), 1):
        verdict_emoji = {
            'PASS': '✅',
            'FAIL': '❌',
            'INCOMPLETE': '⚠️'
        }.get(metrics['verdict'], '❓')
        
        lines.extend([
            f"### {i}. {lane_id.replace('_', ' ').title()} ({metrics['direction']})",
            "",
            f"**Verdict:** {verdict_emoji} **{metrics['verdict']}**  ",
            f"**Recommendation:** {metrics['recommendation']}",
            "",
        ])
        
        if metrics['verdict'] != 'INCOMPLETE':
            lines.extend([
                "| Metric | Value | Floor | Status |",
                "|--------|-------|-------|--------|",
            ])
            
            checks = [
                ('n', metrics['n'], STAGE_A_FLOORS['min_n'], '≥'),
                ('sessions', metrics['sessions'], STAGE_A_FLOORS['min_sessions'], '≥'),
                ('PF', metrics['pf'], STAGE_A_FLOORS['min_pf'], '≥'),
                ('WR', f"{metrics['wr']*100:.1f}%", f"{STAGE_A_FLOORS['min_wr']*100:.0f}%", '≥'),
                ('exp_R', f"{metrics['exp_r']:+.3f}", f"+{STAGE_A_FLOORS['min_exp_r']:.2f}", '≥'),
                ('maxDD', f"{metrics['max_dd_pct']*100:.1f}%", f"{STAGE_A_FLOORS['max_dd_pct']*100:.0f}%", '≤'),
                ('max_losing_day', f"{metrics['max_losing_day_r']:.2f}R", f"{STAGE_A_FLOORS['max_losing_day_r']:.1f}R", '≤'),
            ]
            
            for name, val, floor, op in checks:
                if isinstance(val, (int, float)):
                    if op == '≥':
                        status = '✅' if val >= floor else '❌'
                    else:
                        status = '✅' if val <= floor else '❌'
                else:
                    status = '—'
                lines.append(f"| {name} | {val} | {op} {floor} | {status} |")
            
            lines.append("")
        
        if metrics.get('floor_failures'):
            lines.extend([
                "**Floor failures:**",
                ""
            ])
            for failure in metrics['floor_failures']:
                lines.append(f"- {failure}")
            lines.append("")
        
        if metrics.get('notes'):
            lines.extend([
                f"**Notes:** {metrics['notes']}",
                ""
            ])
        
        if metrics.get('sample_start') and metrics.get('sample_end'):
            lines.append(f"**Sample period:** {metrics['sample_start']} to {metrics['sample_end']}")
            lines.append("")
        
        lines.append("")
    
    lines.extend([
        "---",
        "",
        "## Harness Gaps (Engine Asks)",
        "",
    ])
    
    gaps = []
    for lane_id, metrics in scorecard['lanes'].items():
        if metrics['verdict'] == 'INCOMPLETE' and metrics.get('notes'):
            gaps.append(f"- **{lane_id}**: {metrics['notes']}")
    
    if gaps:
        lines.extend(gaps)
    else:
        lines.append("- No gaps identified")
    
    lines.extend([
        "",
        "---",
        "",
        "*Scorecard generated by `python -m backtest.stage_a_cli scorecard`*",
        ""
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Stage A Backtest CLI for Trading Bot Commentary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    stage_a_parser = subparsers.add_parser('stage-a', help='Run Stage A evaluation for a lane')
    stage_a_parser.add_argument('--lane', type=str, help='Lane ID to evaluate')
    stage_a_parser.add_argument('--direction', type=str, choices=['LONG', 'SHORT'], help='Trade direction')
    stage_a_parser.add_argument('--all', action='store_true', help='Evaluate all lanes')
    
    scorecard_parser = subparsers.add_parser('scorecard', help='Generate unified scorecard')
    scorecard_parser.add_argument('--output', type=str, default='research/reports/', help='Output directory')
    scorecard_parser.add_argument('--json', action='store_true', help='Also output JSON')
    
    args = parser.parse_args()
    
    if args.command == 'stage-a':
        if args.all:
            for lane_id in LANES:
                metrics = run_stage_a_for_lane(lane_id)
                print(f"\n{lane_id}: {metrics.verdict}")
                if metrics.notes:
                    print(f"  Notes: {metrics.notes}")
        elif args.lane:
            if args.direction:
                lane_id = f"{args.lane}_{args.direction.lower()}"
            else:
                lane_id = args.lane
            metrics = run_stage_a_for_lane(lane_id)
            print(json.dumps(metrics.to_dict(), indent=2))
        else:
            stage_a_parser.print_help()
    
    elif args.command == 'scorecard':
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scorecard = generate_scorecard(output_dir)
        
        date_str = datetime.now().strftime('%Y-%m-%d')
        md_path = output_dir / f'{date_str}-stage-a-scorecard.md'
        write_markdown_scorecard(scorecard, md_path)
        print(f"Markdown scorecard: {md_path}")
        
        if args.json:
            json_path = output_dir / f'{date_str}-stage-a-scorecard.json'
            with open(json_path, 'w') as f:
                json.dump(scorecard, f, indent=2)
            print(f"JSON scorecard: {json_path}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
